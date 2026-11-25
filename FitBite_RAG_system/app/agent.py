import re
from pathlib import PurePath

from app.llm import GroqLLM
from app.retrieval import NutritionRetriever
from app.prompts import SYSTEM_PROMPT


ARABIC_RE = re.compile(r"[؀-ۿ]")
NO_ANSWER = "NO_ANSWER"

# Metadata keys differ between loaders, so try several.
SOURCE_KEYS = (
    "source", "source_file", "file_name", "filename", "file",
    "title", "book", "document", "doc_name", "file_path", "path",
)
PAGE_KEYS = ("page", "page_number", "page_num", "page_label")


CLASSIFICATION_PROMPT = """
You are an intent classifier for FitBite, a Nutrition Assistant.
The user's message may be in English or Arabic (including Egyptian dialect).

Classify the message into exactly one category:

greeting:
- Hello, hi, hey, thanks, goodbye, or small talk that opens or ends
  a conversation (e.g. "hi", "اهلا", "السلام عليكم", "شكرا", "ازيك").

nutrition:
- Food, nutrients, protein, carbs, fats, calories, and macros.
- Vitamins, minerals, and hydration.
- Diet plans and meal recommendations.
- Weight management.
- Nutrition for exercise and fitness.
- Nutrition-related health questions.
- Questions about benefits or sources of any nutrient or food.

out_of_domain:
- Mathematics unrelated to food.
- Programming and technology.
- Politics and entertainment.
- General knowledge unrelated to nutrition.

Return ONLY one word:
greeting
or
nutrition
or
out_of_domain
"""

REWRITE_PROMPT = """
Translate the user's nutrition question (Arabic, possibly Egyptian dialect)
into a concise English search query for a nutrition textbook.
Use standard nutrition terminology (for example: benefits, functions,
sources, daily requirement).
Output ONLY the English query, with no quotes or explanation.
"""

MESSAGES = {
    "greeting": {
        "en": (
            "Hi! I'm FitBite, your Nutrition Assistant. Ask me about food, "
            "calories, nutrients, or healthy eating. What would you like "
            "to know?"
        ),
        "ar": (
            "أهلاً بيك! أنا FitBite، مساعدك للتغذية. تقدر تسألني عن الأكل "
            "والسعرات الحرارية والفيتامينات والأكل الصحي. تحب نبدأ بإيه؟"
        ),
    },
    "out_of_domain": {
        "en": (
            "I'm FitBite, your Nutrition Assistant. I can only help with "
            "nutrition, food, calories, and healthy eating questions."
        ),
        "ar": (
            "أنا FitBite، مساعد التغذية بتاعك. أقدر أساعدك بس في أسئلة "
            "الأكل والسعرات الحرارية والتغذية الصحية."
        ),
    },
    "no_answer": {
        "en": (
            "I couldn't find enough relevant information in my nutrition "
            "knowledge base to answer your question reliably."
        ),
        "ar": (
            "معلش، مش لاقي معلومات كفاية في قاعدة المعرفة بتاعتي "
            "عشان أجاوب على سؤالك بشكل موثوق."
        ),
    },
}


def detect_language(text: str) -> str:
    return "ar" if ARABIC_RE.search(text) else "en"


def _first_value(metadata: dict, keys) -> str | None:
    for key in keys:
        value = metadata.get(key)
        if value not in (None, ""):
            return str(value)
    return None


class NutritionAgent:

    def __init__(self):
        self.llm = GroqLLM()
        self.retriever = NutritionRetriever()

    # ------------------------------------------------------------------
    # Intent
    # ------------------------------------------------------------------
    def classify_intent(self, user_message: str) -> str:

        if not user_message or not user_message.strip():
            raise ValueError("User message cannot be empty")

        result = self.llm.generate(
            messages=[
                {"role": "system", "content": CLASSIFICATION_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0,
        )

        result = result.strip().lower()

        if "out_of_domain" in result:
            return "out_of_domain"
        if "greeting" in result:
            return "greeting"
        # Default to nutrition: retrieval plus the NO_ANSWER check
        # safely handle anything the classifier was unsure about.
        return "nutrition"

    # ------------------------------------------------------------------
    # Retrieval helpers
    # ------------------------------------------------------------------
    def build_search_query(self, user_message: str) -> str:
        """The knowledge base is English and the embedding model
        (all-MiniLM-L6-v2) is English-only, so translate Arabic
        questions to English before searching."""

        if detect_language(user_message) == "en":
            return user_message

        try:
            rewritten = self.llm.generate(
                messages=[
                    {"role": "system", "content": REWRITE_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=0,
            )
        except Exception:
            return user_message

        rewritten = (rewritten or "").strip().strip("\"'")
        return rewritten or user_message

    @staticmethod
    def describe_source(metadata: dict) -> str | None:
        """Return 'Book.pdf, page 12', or None if metadata has neither."""

        source = _first_value(metadata, SOURCE_KEYS)
        page = _first_value(metadata, PAGE_KEYS)

        if source:
            source = PurePath(source).name or source

        if source and page:
            return f"{source}, page {page}"
        if source:
            return source
        if page:
            return f"Page {page}"
        return None

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------
    def chat(self, user_message: str):

        lang = detect_language(user_message)

        # Step 1: Classify user intent
        intent = self.classify_intent(user_message)

        if intent in ("greeting", "out_of_domain"):
            return MESSAGES[intent][lang]

        # Step 2: Retrieve relevant nutrition knowledge
        search_query = self.build_search_query(user_message)
        documents = self.retriever.search(search_query)

        if not documents:
            return MESSAGES["no_answer"][lang]

        # Step 3: Build context
        context_parts = []
        sources = []

        for doc in documents:
            label = self.describe_source(doc["metadata"])

            header = f"Source: {label}\n" if label else ""
            context_parts.append(f"{header}Content: {doc['content']}")

            if label and label not in sources:
                sources.append(label)

        context = "\n\n".join(context_parts)

        # Step 4: Generate grounded answer
        response = self.llm.generate(
            messages=[
                {
                    "role": "system",
                    "content": (
                        SYSTEM_PROMPT
                        + "\n\n"
                        + "Answer the user's question using "
                        "the retrieved context below. "
                        "Do not invent facts. If the context "
                        "genuinely does not contain the answer, "
                        f"reply with exactly {NO_ANSWER} and nothing else."
                        + "\n\nRETRIEVED CONTEXT:\n"
                        + context
                    ),
                },
                {"role": "user", "content": user_message},
            ],
            temperature=0.3,
        )

        response = (response or "").strip()

        # No grounded answer: say so and do not attach unrelated sources.
        if not response or NO_ANSWER in response:
            return MESSAGES["no_answer"][lang]

        # Step 5: Attach source information
        if not sources:
            return response

        heading = "المصادر" if lang == "ar" else "Sources"
        source_text = "\n".join(f"- {source}" for source in sources)

        return f"{response}\n\n{heading}:\n{source_text}"
