import os

from .env_loader import load_local_env

load_local_env()


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


PREDICTION_MODEL = os.getenv("OPENAI_PREDICTION_MODEL", "gpt-4.1-nano")
SEARCH_MODEL = os.getenv("OPENAI_SEARCH_MODEL", "gpt-4.1-mini")

SEARCH_MAX_OUTPUT_TOKENS = _env_int("OPENAI_SEARCH_MAX_OUTPUT_TOKENS", 500)
PREDICTION_MAX_OUTPUT_TOKENS = _env_int("OPENAI_PREDICTION_MAX_OUTPUT_TOKENS", 500)
SEARCH_EVIDENCE_WORD_LIMIT = _env_int("OPENAI_SEARCH_EVIDENCE_WORD_LIMIT", 220)

KALSHI_API_BASE_URL = os.getenv(
    "KALSHI_API_BASE_URL", "https://api.elections.kalshi.com/trade-api/v2"
).rstrip("/")
USE_MARKET_PRIOR = os.getenv("USE_MARKET_PRIOR", "1") not in {"0", "false", "False"}
MODEL_WEIGHT = max(0.0, min(1.0, _env_float("MODEL_WEIGHT", 0.8)))
