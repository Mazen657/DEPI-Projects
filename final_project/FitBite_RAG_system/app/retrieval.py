"""Hybrid retriever: ChromaDB vector search + BM25, fused with weighted RRF."""

import logging
import re
import threading
from collections import defaultdict
from typing import Callable

import chromadb
import numpy as np
from rank_bm25 import BM25Okapi

from app.config import settings

logger = logging.getLogger(__name__)

# Light stopword list so BM25 focuses on informative terms.
_STOPWORDS = frozenset(
    "a an the and or of to in on for with is are was were be been what how "
    "should can could do does i my me it its this that these those at by from "
    "as about".split()
)

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

# Arabic normalisation: strip diacritics/tatweel, unify alef/yaa/taa-marbuta.
_AR_DIACRITICS = re.compile(r"[\u064B-\u0652\u0640]")
_AR_MAP = str.maketrans(
    {"أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي", "ة": "ه"}
)

CATEGORY_RULES = {
    "Sports nutrition": [
        "athlete", "protein intake", "exercise", "muscle", "workout", "sports",
    ],
    "Nutrient reference values": [
        "rda", "upper limit", "vitamin", "mineral", "daily intake",
    ],
    "Healthy diet: WHO/FAO guidance": [
        "healthy diet", "salt", "sugar", "fruit", "vegetables",
    ],
    "Dietary patterns / national guidelines": [
        "dash diet", "mediterranean diet", "dietary guidelines",
    ],
    "Clinical guideline": [
        "anemia", "iron deficiency", "clinical guideline",
    ],
}

# Pre-compiled word-boundary patterns (so "salt" doesn't match "salty snack"
# by accident, and "rda" doesn't match inside other words).
_CATEGORY_PATTERNS = {
    category: [
        re.compile(rf"\b{re.escape(keyword)}\b", re.IGNORECASE)
        for keyword in keywords
    ]
    for category, keywords in CATEGORY_RULES.items()
}


# ----------------------------------------------------------------------
# Query rewriting
# ----------------------------------------------------------------------

_FILLER_RE = re.compile(
    r"^\s*(?:please\s+)?"
    r"(?:(?:can|could|would)\s+you\s+(?:please\s+)?)?"
    r"(?:tell|explain|show|give)\s+me\s+(?:about\s+)?"
    r"|^\s*i\s+(?:want|need)\s+to\s+know\s+(?:about\s+)?"
    r"|^\s*please\s+",
    re.IGNORECASE,
)
_TRAILING_PUNCT_RE = re.compile(r"[\s?!.,;:]+$")

ABBREVIATIONS = {
    "rda": "recommended dietary allowance",
    "dri": "dietary reference intakes",
    "bmi": "body mass index",
    "ida": "iron deficiency anemia",
    "vit": "vitamin",
}

SYNONYMS = {
    "anemia": ["anaemia"],
    "anaemia": ["anemia"],
    "salt": ["sodium"],
    "sodium": ["salt"],
    "sugar": ["free sugars", "added sugars"],
    "workout": ["exercise", "training"],
    "athlete": ["athletes", "sports"],
    "fruit": ["fruits"],
    "vegetables": ["veggies"],
}


def _dedupe_key(text: str) -> str:
    return " ".join(text.lower().split())


class QueryRewriter:
    """
    Turns one user query into a few retrieval-friendly variants.

    Rule-based by default (filler removal, abbreviation expansion, synonym
    swaps). Optionally pass `llm_rewriter`, a function `str -> list[str]`,
    to add LLM-generated rewrites; if it fails, the rule-based variants
    are still used.
    """

    def __init__(
        self,
        llm_rewriter: Callable[[str], list[str]] | None = None,
        max_variants: int = 3,
    ):
        self.llm_rewriter = llm_rewriter
        self.max_variants = max_variants

    @staticmethod
    def _clean(query: str) -> str:
        query = _FILLER_RE.sub("", query.strip())
        return _TRAILING_PUNCT_RE.sub("", query).strip()

    @staticmethod
    def _expand_abbreviations(query: str) -> str:
        def replace(match):
            return ABBREVIATIONS[match.group(0).lower()]

        pattern = r"\b(?:" + "|".join(map(re.escape, ABBREVIATIONS)) + r")\b"
        return re.sub(pattern, replace, query, flags=re.IGNORECASE)

    @staticmethod
    def _synonym_variants(query: str) -> list[str]:
        variants = []
        for term, alternatives in SYNONYMS.items():
            pattern = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
            if pattern.search(query):
                for alt in alternatives:
                    variants.append(pattern.sub(alt, query))
        return variants

    def rewrite(self, query: str) -> list[str]:
        """Return [original, variant1, ...] with no duplicates."""
        cleaned = self._clean(query)
        candidates = [cleaned, self._expand_abbreviations(cleaned)]

        if self.llm_rewriter:
            try:
                candidates.extend(self.llm_rewriter(query) or [])
            except Exception:
                logger.exception("LLM query rewrite failed; using rules only")

        candidates.extend(self._synonym_variants(cleaned))

        results = [query]
        seen = {_dedupe_key(query)}
        for candidate in candidates:
            key = _dedupe_key(candidate or "")
            if not key or key in seen:
                continue
            seen.add(key)
            results.append(candidate)
            if len(results) > self.max_variants:
                break
        return results


class NutritionRetriever:
    """Hybrid (vector + BM25) retriever with query rewriting and
    category-aware filtering."""

    def __init__(self, rewriter: QueryRewriter | None = None):
        if not settings.CHROMA_DB_PATH:
            raise ValueError("CHROMA_DB_PATH is missing in .env")

        self.rrf_k = getattr(settings, "RRF_K", 60)
        self.vector_weight = getattr(settings, "VECTOR_WEIGHT", 1.0)
        self.bm25_weight = getattr(settings, "BM25_WEIGHT", 1.0)
        # Rewritten queries count a bit less than the user's own wording.
        self.rewrite_weight = getattr(settings, "QUERY_REWRITE_WEIGHT", 0.7)
        self.rewriter = rewriter or QueryRewriter()

        self._lock = threading.Lock()
        self.client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)

        try:
            self.collection = self.client.get_collection(
                name=settings.CHROMA_COLLECTION
            )
        except Exception as error:
            raise RuntimeError(
                f"Could not load collection "
                f"'{settings.CHROMA_COLLECTION}': {error}"
            ) from error

        self._load_bm25_index()

    # ------------------------------------------------------------------
    # Index
    # ------------------------------------------------------------------

    def _load_bm25_index(self):
        """(Re)build the in-memory BM25 index and lookup tables."""
        data = self.collection.get(include=["documents", "metadatas"])

        documents = data["documents"] or []
        if not documents:
            raise RuntimeError("ChromaDB collection is empty")

        metadatas = [m or {} for m in (data["metadatas"] or [{}] * len(documents))]
        ids = data["ids"]

        bm25 = BM25Okapi([self._tokenize(doc) for doc in documents])

        # Category -> boolean mask, so BM25 filtering is a vectorised op.
        categories = np.array([m.get("category") for m in metadatas], dtype=object)

        with self._lock:
            self.documents = documents
            self.metadatas = metadatas
            self.ids = ids
            self.id_to_index = {doc_id: i for i, doc_id in enumerate(ids)}
            self._categories = categories
            self.bm25 = bm25

        logger.info("BM25 index loaded: %d documents", len(documents))

    def refresh(self):
        """Call after adding/removing documents in the collection."""
        self._load_bm25_index()

    # ------------------------------------------------------------------
    # Text processing
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(text: str) -> str:
        text = text.lower()
        text = _AR_DIACRITICS.sub("", text)
        return text.translate(_AR_MAP)

    @classmethod
    def _tokenize(cls, text: str) -> list[str]:
        tokens = _TOKEN_RE.findall(cls._normalize(text))
        return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]

    # ------------------------------------------------------------------
    # Category detection
    # ------------------------------------------------------------------

    def detect_categories(self, query: str) -> list[str]:
        """Return the best-scoring categories (several if tied), or []."""
        scores = {
            category: sum(1 for p in patterns if p.search(query))
            for category, patterns in _CATEGORY_PATTERNS.items()
        }
        best = max(scores.values(), default=0)
        if best == 0:
            return []
        return [c for c, s in scores.items() if s == best]

    def detect_category(self, query: str):
        """Backward-compatible single-category helper."""
        categories = self.detect_categories(query)
        return categories[0] if categories else None

    @staticmethod
    def _as_list(category) -> list[str]:
        if not category:
            return []
        return [category] if isinstance(category, str) else list(category)

    # ------------------------------------------------------------------
    # Vector search
    # ------------------------------------------------------------------

    def _vector_search(self, query, candidate_k, categories):
        args = {
            "query_texts": [query],
            "n_results": min(candidate_k, len(self.documents)),
        }
        if len(categories) == 1:
            args["where"] = {"category": categories[0]}
        elif len(categories) > 1:
            args["where"] = {"category": {"$in": categories}}

        try:
            results = self.collection.query(**args)
        except Exception:
            logger.exception("Vector search failed; falling back to BM25 only")
            return []

        return results["ids"][0] if results.get("ids") else []

    # ------------------------------------------------------------------
    # BM25 search
    # ------------------------------------------------------------------

    def _bm25_search(self, query, candidate_k, categories):
        tokens = self._tokenize(query)
        if not tokens:
            return []

        scores = np.asarray(self.bm25.get_scores(tokens), dtype=float)

        if categories:
            mask = np.isin(self._categories, categories)
            scores = np.where(mask, scores, -np.inf)

        # Drop documents with no lexical overlap: they are noise in the fusion.
        valid = np.flatnonzero(scores > 0)
        if valid.size == 0:
            return []

        top = valid[np.argsort(scores[valid])[::-1][:candidate_k]]
        return [self.ids[i] for i in top]

    # ------------------------------------------------------------------
    # Fusion
    # ------------------------------------------------------------------

    def _rrf_fusion(self, rankings):
        """
        Weighted Reciprocal Rank Fusion over any number of rankings.
        `rankings` is a list of (source_name, ordered_ids, weight).
        Returns [(id, score, sources)].
        """
        scores = defaultdict(float)
        sources = defaultdict(set)

        for name, ranking, weight in rankings:
            for rank, doc_id in enumerate(ranking):
                scores[doc_id] += weight / (self.rrf_k + rank + 1)
                sources[doc_id].add(name)

        ranked = sorted(scores, key=scores.get, reverse=True)
        return [(d, scores[d], sorted(sources[d])) for d in ranked]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int | None = None,
        category: str | list[str] | None = None,
        rewrite: bool = True,
    ):
        """
        Hybrid search with optional query rewriting. Returns a list of dicts:
        {"id", "content", "metadata", "score", "sources"}
        ordered by fused relevance.

        `category` may be a string, a list of strings, or None (auto-detect).
        If the category filter returns nothing, the search is retried
        without the filter so the caller never gets an empty answer
        purely because of a wrong auto-detected category.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        query = query.strip()
        top_k = top_k or settings.RETRIEVAL_TOP_K

        # [(text, weight)]: the original query first, rewrites after it.
        queries = [(query, 1.0)]
        if rewrite:
            variants = self.rewriter.rewrite(query)
            queries += [(v, self.rewrite_weight) for v in variants[1:]]
            logger.debug("Query variants: %s", [q for q, _ in queries])

        categories = (
            self._as_list(category)
            or self.detect_categories(query)
            # Expanded text can reveal a category the raw query hid
            # (e.g. "vit d" -> "vitamin d").
            or self.detect_categories(" ".join(q for q, _ in queries))
        )
        candidate_k = min(top_k * 3, len(self.documents))

        fused = self._run(queries, candidate_k, categories)

        if not fused and categories:
            logger.info("No hits in %s; retrying without category", categories)
            fused = self._run(queries, candidate_k, [])

        results = []
        for doc_id, score, srcs in fused[:top_k]:
            index = self.id_to_index.get(doc_id)
            if index is None:
                continue
            results.append(
                {
                    "id": doc_id,
                    "content": self.documents[index],
                    "metadata": self.metadatas[index],
                    "score": round(score, 6),
                    "sources": srcs,
                }
            )
        return results

    def _run(self, queries, candidate_k, categories):
        """Run vector + BM25 for every query variant and fuse all rankings."""
        rankings = []
        for text, query_weight in queries:
            rankings.append(
                (
                    "vector",
                    self._vector_search(text, candidate_k, categories),
                    self.vector_weight * query_weight,
                )
            )
            rankings.append(
                (
                    "bm25",
                    self._bm25_search(text, candidate_k, categories),
                    self.bm25_weight * query_weight,
                )
            )
        return self._rrf_fusion(rankings)
