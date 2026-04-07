from typing import Optional

from openai import APIError, OpenAI

from .config import (
    SEARCH_EVIDENCE_WORD_LIMIT,
    SEARCH_MAX_OUTPUT_TOKENS,
    SEARCH_MODEL,
)
from .time_utils import format_close_time


def gather_openai_web_evidence(client: OpenAI, event: dict) -> Optional[str]:
    """Use OpenAI web search tool to gather concise forecasting evidence."""
    title = event.get("title", "N/A")
    description = event.get("description", "N/A")
    category = event.get("category", "N/A")
    close_time_str = format_close_time(event.get("close_time"))

    prompt = f"""You are collecting forecasting evidence for a prediction market event.

Event:
- Title: {title}
- Description: {description}
- Category: {category}
- Close time: {close_time_str}

Use web search to find the most relevant, recent, and credible information.
Return a concise evidence brief under {SEARCH_EVIDENCE_WORD_LIMIT} words with this structure:
- Key facts
- Latest signals
- Main uncertainties
- Sources (3-6 links)

Be concrete and prioritize facts that could change the forecast.
"""

    try:
        response = client.responses.create(
            model=SEARCH_MODEL,
            tools=[{"type": "web_search_preview"}],
            input=[
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}],
                }
            ],
            max_output_tokens=SEARCH_MAX_OUTPUT_TOKENS,
        )
        text = (response.output_text or "").strip()
        return text or None
    except APIError:
        return None
    except Exception:
        return None
