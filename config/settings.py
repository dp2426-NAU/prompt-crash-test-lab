"""Central configuration for Prompt Crash-Test Lab."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SCHEMAS_DIR = DATA_DIR / "schemas"
BASE_PROMPTS_DIR = DATA_DIR / "base_prompts"
VARIANTS_DIR = DATA_DIR / "variants"
RESULTS_DIR = DATA_DIR / "results"
CACHE_DB_PATH = PROJECT_ROOT / "cache.db"

# Ensure directories exist
for d in [VARIANTS_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── API Keys ───────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "")

# ── Model Configuration ───────────────────────────────────
MODELS = {
    "gpt-4-turbo": {
        "provider": "openai",
        "model_id": "gpt-4-turbo",
        "max_tokens": 1024,
        "temperature": 0.0,
    },
    "claude-3.5-sonnet": {
        "provider": "anthropic",
        "model_id": "claude-3-5-sonnet-20241022",
        "max_tokens": 1024,
        "temperature": 0.0,
    },
    "gemini-1.5-pro": {
        "provider": "google",
        "model_id": "gemini-1.5-pro",
        "max_tokens": 1024,
        "temperature": 0.0,
    },
    "llama-3.1-70b": {
        "provider": "together",
        "model_id": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        "max_tokens": 1024,
        "temperature": 0.0,
    },
}

# ── Evaluation Settings ───────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
VARIANT_TYPES = ["paraphrase", "format", "role", "constraint", "template"]
VARIANTS_PER_TYPE = {"paraphrase": 5, "format": 4, "role": 3, "constraint": 3, "template": 5}
TOTAL_VARIANTS_PER_PROMPT = sum(VARIANTS_PER_TYPE.values())  # 20

# ── Rate Limiting ─────────────────────────────────────────
RATE_LIMIT_DELAY = 1.0  # seconds between API calls
MAX_RETRIES = 3
RETRY_DELAY = 5.0  # seconds between retries
