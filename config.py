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


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return int(raw.strip())


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return float(raw.strip())

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

LLM_PROVIDER = _env_str("LLM_PROVIDER", "gemini")
# live = Vertex/OpenRouter. fake = dummy nodes for QA/Docker without credentials.
GRAPH_MODE = _env_str("SPEC_REFINERY_GRAPH", "live")
VECTOR_BACKEND = _env_str("VECTOR_BACKEND", "chroma")
CHECKPOINT_PATH = _env_str("CHECKPOINT_PATH", str(BASE_DIR / "checkpoints.sqlite"))
CORPUS_DIR = BASE_DIR / "corpus"
CHROMA_DIR = BASE_DIR / ".chroma"
MAX_STEPS = 8
# Interrogation budget, shown to the PM in the UI: at most MAX_QUESTIONS_PER_ROUND
# questions per round, and at most MAX_ROUNDS rounds before the spec is frozen.
MAX_QUESTIONS_PER_ROUND = 3
MAX_ROUNDS = 5
RECURSION_LIMIT = 20
TOP_K = 5
EMBEDDING_MODEL = _env_str("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
# Model IDs are configurable (rubric: no hardcoded models). Empty means
# "use the provider default" resolved in clients/factory.py.
INTERROGATOR_MODEL = _env_str("INTERROGATOR_MODEL", "")
WRITER_MODEL = _env_str("WRITER_MODEL", "")
# Per-request timeout (seconds) and bounded SDK retries for every model call:
# a hung or throttled provider must fail fast instead of leaving a turn open.
LLM_TIMEOUT_SECONDS = _env_float("LLM_TIMEOUT_SECONDS", 90.0)
LLM_MAX_RETRIES = _env_int("LLM_MAX_RETRIES", 2)
