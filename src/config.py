import os

from .env_loader import load_local_env

load_local_env()


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


PREDICTION_MODEL = os.getenv("OPENAI_PREDICTION_MODEL", "gpt-4.1-nano")
SEARCH_MODEL = os.getenv("OPENAI_SEARCH_MODEL", "gpt-4.1-mini")

SEARCH_MAX_OUTPUT_TOKENS = _env_int("OPENAI_SEARCH_MAX_OUTPUT_TOKENS", 500)
PREDICTION_MAX_OUTPUT_TOKENS = _env_int("OPENAI_PREDICTION_MAX_OUTPUT_TOKENS", 500)
SEARCH_EVIDENCE_WORD_LIMIT = _env_int("OPENAI_SEARCH_EVIDENCE_WORD_LIMIT", 220)
