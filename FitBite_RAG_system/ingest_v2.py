"""
Build NutriFit AI ChromaDB V2 with metadata from kb_categorization.csv.

Expected project structure (edit the paths below if yours differs):
Fit_bite/
├── data/
│   ├── pdfs/                  # Put the source PDFs here
│   └── csv/                   # Put ALL categorization CSV files here
│       ├── categorization_part_1.csv
│       ├── categorization_part_2.csv
│       └── ...
├── chroma_db/
└── ingest_v2.py

Install:
pip install chromadb sentence-transformers pypdf langchain-text-splitters pandas
"""

from pathlib import Path
from collections import defaultdict
import hashlib
import re
import sys

import pandas as pd
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ============================================================
# 1. CONFIGURATION — EDIT THESE PATHS / SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

PDF_DIR = BASE_DIR / "data" / "pdfs"
# Put ALL categorization CSV files in this folder.
# Example: data/csv/part_1.csv, data/csv/part_2.csv, ...
CSV_DIR = BASE_DIR / "data" / "csv"

CHROMA_PATH = BASE_DIR / "chroma_db"

# This creates a separate V2 collection; the old collection is untouched.
COLLECTION_NAME = "nutrition_knowledge_v2"

# IMPORTANT:
# Set this to EXACTLY the same embedding model used by your current
# nutrition_knowledge collection if you want comparable embeddings.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

CHUNK_SIZE = 700
CHUNK_OVERLAP = 100
BATCH_SIZE = 200

# True: delete and rebuild ONLY nutrition_knowledge_v2 every run.
# False: keep the V2 collection and upsert the generated chunks.
REBUILD_V2_EVERY_RUN = True

# CSV page numbers are assumed to be 1-based (page 1 = first PDF page).
CSV_PAGES_START_AT_ONE = True


# ============================================================
# 2. CSV METADATA MAPPING
# ============================================================

def parse_page_numbers(value):
    """
    Converts values such as:
      '9'       -> {9}
      '11-12'   -> {11, 12}
      '1-4, 7'  -> {1, 2, 3, 4, 7}
    """
    pages = set()
    value = str(value).strip()

    # Extract individual page numbers and ranges.
    parts = re.split(r"[,;،]", value)

    for part in parts:
        part = part.strip()
        if not part:
            continue

        match = re.fullmatch(r"(\d+)\s*-\s*(\d+)", part)

        if match:
            start, end = map(int, match.groups())
            if end < start:
                start, end = end, start
            pages.update(range(start, end + 1))
        else:
            match = re.search(r"\d+", part)
            if match:
                pages.add(int(match.group()))

    return pages


class MetadataMapper:
    def __init__(self, csv_dir):
        if not csv_dir.exists():
            raise FileNotFoundError(
                f"CSV folder not found: {csv_dir}\\n"
                "Create data/csv and put all categorization CSV files inside it."
            )

        csv_files = sorted(csv_dir.glob("*.csv"))

        if not csv_files:
            raise FileNotFoundError(f"No CSV files found inside: {csv_dir}")

        frames = []

        required_columns = {
            "Category", "PDF_Name", "Topic", "Subtopic", "Page_Numbers"
        }

        for csv_file in csv_files:
            print(f"[INFO] Reading metadata file: {csv_file.name}")
            frame = pd.read_csv(csv_file).fillna("")

            missing = required_columns - set(frame.columns)
            if missing:
                raise ValueError(
                    f"{csv_file.name} is missing columns: {sorted(missing)}"
                )

            frames.append(frame)

        self.df = pd.concat(frames, ignore_index=True).fillna("")

        # filename -> page number -> rows matching that page
        self.mapping = defaultdict(lambda: defaultdict(list))

        for _, row in self.df.iterrows():
            filename = Path(str(row["PDF_Name"]).strip()).name
            pages = parse_page_numbers(row["Page_Numbers"])

            for page in pages:
                self.mapping[filename][page].append({
                    "category": str(row["Category"]).strip(),
                    "topic": str(row["Topic"]).strip(),
                    "subtopic": str(row["Subtopic"]).strip(),
                })

        print(f"[INFO] Loaded {len(self.df)} CSV records.")
        print(f"[INFO] CSV contains {len(self.mapping)} PDF filenames.")

    def get_metadata_groups(self, filename, page_number):
        """
        Returns one metadata group per category.
        If a PDF page has multiple topics in the same category,
        their topic/subtopic values are combined into strings.
        """
        rows = self.mapping.get(filename, {}).get(page_number, [])

        if not rows:
            return [{
                "category": "Uncategorized",
                "topic": "Not mapped in CSV",
                "subtopic": "Not mapped in CSV",
            }]

        grouped = defaultdict(lambda: {"topics": [], "subtopics": []})

        for row in rows:
            category = row["category"] or "Uncategorized"

            if row["topic"] and row["topic"] not in grouped[category]["topics"]:
                grouped[category]["topics"].append(row["topic"])

            if row["subtopic"] and row["subtopic"] not in grouped[category]["subtopics"]:
                grouped[category]["subtopics"].append(row["subtopic"])

        result = []

        for category, values in grouped.items():
            result.append({
                "category": category,
                "topic": " | ".join(values["topics"]) or "Not specified",
                "subtopic": " | ".join(values["subtopics"]) or "Not specified",
            })

        return result


# ============================================================
# 3. PDF READING + PAGE-LEVEL CHUNKING
# ============================================================

def make_chunk_id(filename, page_number, chunk_index, category):
    raw = f"{filename}|{page_number}|{chunk_index}|{category}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def process_pdf(pdf_path, mapper, splitter):
    """
    Extract text page by page so the CSV page mapping is accurate.
    Each resulting chunk keeps its original PDF page number.
    """
    filename = pdf_path.name
    reader = PdfReader(str(pdf_path))

    output = []

    for zero_based_index, pdf_page in enumerate(reader.pages):
        page_number = zero_based_index + 1 if CSV_PAGES_START_AT_ONE else zero_based_index

        text = (pdf_page.extract_text() or "").strip()

        if not text:
            print(f"[WARN] Empty / image-only page: {filename}, page {page_number}")
            continue

        page_chunks = splitter.split_text(text)
        metadata_groups = mapper.get_metadata_groups(filename, page_number)

        for chunk_index, chunk_text in enumerate(page_chunks):
            for mapped_metadata in metadata_groups:
                metadata = {
                    "category": mapped_metadata["category"],
                    "topic": mapped_metadata["topic"],
                    "subtopic": mapped_metadata["subtopic"],
                    "source": filename,
                    "page": int(page_number),
                    "chunk_index": int(chunk_index),
                }

                output.append({
                    "id": make_chunk_id(
                        filename,
                        page_number,
                        chunk_index,
                        mapped_metadata["category"],
                    ),
                    "text": chunk_text,
                    "metadata": metadata,
                })

    return output


# ============================================================
# 4. CHROMADB V2 COLLECTION
# ============================================================

def create_v2_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    existing_names = {
        item.name for item in client.list_collections()
    }

    if REBUILD_V2_EVERY_RUN and COLLECTION_NAME in existing_names:
        print(f"[INFO] Deleting existing {COLLECTION_NAME} collection...")
        client.delete_collection(name=COLLECTION_NAME)

    embedding_function = SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function,
        metadata={"description": "NutriFit AI knowledge base with CSV metadata"},
    )

    return client, collection


def store_in_batches(collection, records):
    total = len(records)

    for start in range(0, total, BATCH_SIZE):
        batch = records[start:start + BATCH_SIZE]

        collection.upsert(
            ids=[item["id"] for item in batch],
            documents=[item["text"] for item in batch],
            metadatas=[item["metadata"] for item in batch],
        )

        print(f"[INFO] Stored {min(start + BATCH_SIZE, total)} / {total} chunks")


# ============================================================
# 5. MAIN
# ============================================================

def main():
    if not PDF_DIR.exists():
        raise FileNotFoundError(
            f"PDF folder not found: {PDF_DIR}\n"
            "Create data/pdfs and put your source PDFs inside it, "
            "or edit PDF_DIR in this script."
        )

    pdf_files = sorted(PDF_DIR.rglob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found inside: {PDF_DIR}")

    mapper = MetadataMapper(CSV_DIR)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    print(f"[INFO] Found {len(pdf_files)} PDF files.")

    all_records = []
    found_filenames = set()

    for pdf_path in pdf_files:
        print(f"\n[PDF] Processing: {pdf_path.name}")
        found_filenames.add(pdf_path.name)

        try:
            records = process_pdf(pdf_path, mapper, splitter)
            all_records.extend(records)
            print(f"[INFO] Generated {len(records)} chunks from {pdf_path.name}")
        except Exception as exc:
            print(f"[ERROR] Failed to process {pdf_path.name}: {exc}")

    csv_filenames = set(mapper.mapping.keys())
    missing_pdfs = sorted(csv_filenames - found_filenames)

    if missing_pdfs:
        print("\n[WARN] These CSV PDFs were not found in PDF_DIR:")
        for name in missing_pdfs:
            print(f"  - {name}")

    if not all_records:
        raise RuntimeError("No chunks were generated. Check the PDF files.")

    print(f"\n[INFO] Total chunks to store: {len(all_records)}")

    _, collection = create_v2_collection()
    store_in_batches(collection, all_records)

    print("\n========== INGESTION COMPLETE ==========")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Chroma path: {CHROMA_PATH}")
    print(f"Stored chunks: {collection.count()}")
    print("Old collection was not modified.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\n[FATAL] {error}")
        sys.exit(1)
