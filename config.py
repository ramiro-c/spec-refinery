from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
BASE_DIR = Path(__file__).resolve().parent

def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

LLM_PROVIDER = _env_str("LLM_PROVIDER", "gemini")
VECTOR_BACKEND = _env_str("VECTOR_BACKEND", "chroma")
CHECKPOINT_PATH = _env_str("CHECKPOINT_PATH", str(BASE_DIR / "checkpoints.sqlite"))
CORPUS_DIR = BASE_DIR / "corpus"
CHROMA_DIR = BASE_DIR / ".chroma"
MAX_STEPS = 8
RECURSION_LIMIT = 20
TOP_K = 5
EMBEDDING_MODEL = _env_str("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = _env_str("INDEX_NAME", "spec-refinery-rag")
