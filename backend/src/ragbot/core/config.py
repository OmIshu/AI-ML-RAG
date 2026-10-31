import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[3]
load_dotenv(ROOT / ".env")


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default) or default


DATA_DIR = ROOT / "data"
PDF_DIR = DATA_DIR / "pdfs"
QDRANT_PATH = DATA_DIR / "qdrant"
DATA_DIR.mkdir(exist_ok=True)
PDF_DIR.mkdir(exist_ok=True)

DATALAB_API_KEY = _env("DATALAB_API_KEY")
DATALAB_URL = "https://www.datalab.to/api/v1/marker"
OCR_BATCH_PAGES = 40  # pages per Datalab request

LLM_BASE_URL = _env("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_API_KEY = _env("LLM_API_KEY")
LLM_MODEL = _env("LLM_MODEL", "openai/gpt-oss-20b")

LLM_REASONING_EFFORT = _env("LLM_REASONING_EFFORT", "medium")  # gpt-oss only: low | medium | high

EMBED_MODEL =_env("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
QUERY_PREFIX = os.environ.get("QUERY_PREFIX", "Represent this sentence for searching relevant passages: ")
RERANK_MODEL = _env("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
DEVICE = _env("DEVICE", "auto")

QDRANT_URL = _env("QDRANT_URL")
COLLECTION = _env("QDRANT_COLLECTION", "pdf_chunks")

CHUNK_TOKENS = int(_env("CHUNK_TOKENS", "500"))
CHUNK_OVERLAP = float(_env("CHUNK_OVERLAP", "0.15"))

TOP_K_CANDIDATES = int(_env("TOP_K_CANDIDATES", "15"))
TOP_N_CONTEXT = int(_env("TOP_N_CONTEXT", "4"))

# page is treated as scanned when native text is below this many characters
MIN_NATIVE_CHARS = 100
