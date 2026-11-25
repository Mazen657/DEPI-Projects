
import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from the project root
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


class Settings:
    # ChromaDB
    CHROMA_DB_PATH: str = os.getenv("CHROMA_DB_PATH", "")
    CHROMA_COLLECTION: str = os.getenv(
        "CHROMA_COLLECTION",
        "nutrition_knowledge_v2"
    )

    # Embeddings
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    # PostgreSQL
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")

    # Retrieval
    RETRIEVAL_TOP_K: int = int(
        os.getenv("RETRIEVAL_TOP_K", "5")
    )
    # Groq LLM
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv(
        "GROQ_MODEL",
        "llama-3.3-70b-versatile"
    )


settings = Settings()
