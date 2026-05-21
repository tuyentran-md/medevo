from __future__ import annotations

import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("MEDEVO_DATA_DIR", ROOT_DIR / "data"))
ARTIFACTS_DIR = DATA_DIR / "artifacts"
DB_PATH = DATA_DIR / "medevo.db"
YEARS = (10, 20, 30)
MAX_CLAIMS = 6
DEFAULT_OLLAMA_BASE_URL = os.environ.get(
    "MEDEVO_OLLAMA_BASE_URL", "http://127.0.0.1:11434"
)
DEFAULT_OLLAMA_MODEL = os.environ.get("MEDEVO_OLLAMA_MODEL", "gemma3:12b")
DEFAULT_GEMINI_BASE_URL = os.environ.get(
    "MEDEVO_GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai"
)
DEFAULT_GEMINI_MODEL = os.environ.get("MEDEVO_GEMINI_MODEL", "gemini-3-flash")
DEFAULT_RATE_LIMIT = int(os.environ.get("MEDEVO_MAX_CONCURRENT_RUNS", "3"))
DEFAULT_PUBMED_EMAIL = os.environ.get("MEDEVO_PUBMED_EMAIL")
DEFAULT_PUBMED_API_KEY = os.environ.get("MEDEVO_PUBMED_API_KEY")
DEFAULT_PUBMED_MIN_INTERVAL_SECONDS = float(
    os.environ.get("MEDEVO_PUBMED_MIN_INTERVAL_SECONDS", "0.34")
)
