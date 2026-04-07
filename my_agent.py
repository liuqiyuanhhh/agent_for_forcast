"""
Forecasting agent for prediction markets using OpenAI API.

Usage:
    prophet forecast predict --events events.json --local my_agent
"""

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from openai import APIError, OpenAI


def _load_local_env() -> None:
    """Load KEY=VALUE pairs from .env in this project directory."""
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_local_env()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CATEGORY_HINTS = {
    "Financials": (
        "Focus on: recent price action, macroeconomic conditions, historical volatility, "
        "market sentiment, and time until resolution. Financial markets are efficient - "
        "unless there is strong directional evidence, lean toward 0.5."
    ),
    "Politics": (
        "Focus on: polling data, historical base rates, incumbency advantage, "
        "structural factors (electoral college, partisan lean), and prediction market consensus. "
        "Weight aggregated forecasts heavily."
    ),
    "Sports": (
        "Focus on: team/player rankings, recent form, home advantage, head-to-head records, "
        "injuries, and statistical models. Use Elo or similar quantitative bases where possible."
    ),
    "Science": (
        "Focus on: peer-reviewed literature, expert consensus, publication timelines, "
        "regulatory processes, and historical success rates for similar milestones."
    ),
    "Technology": (
        "Focus on: company roadmaps, release history, competitive landscape, regulatory risk, "
        "and credibility of announcements. Be skeptical of aspirational timelines."
    ),
    "Geopolitics": (
        "Focus on: historical base rates for similar events, diplomatic signals, military posture, "
        "incentive structures, and international pressure. Avoid overconfidence."
    ),
    "Economics": (
        "Focus on: leading indicators, central bank signals, consensus forecasts, "
        "historical patterns in similar macro environments."
    ),
    "Health": (
        "Focus on: clinical trial phases, regulatory timelines, epidemiological trends, "
        "historical base rates for FDA approvals or disease milestones."
    ),
    "Entertainment": (
        "Focus on: box office tracking, streaming trends, awards history, critical reception, "
        "and comparable titles."
    ),
    "Climate": (
        "Focus on: scientific models, historical weather patterns, seasonal baselines, "
        "and climate change trend data."
    ),
}

DEFAULT_HINT = (
    "Use base rates, available evidence, and domain expertise. "
    "Avoid extreme probabilities unless evidence strongly supports them."
)


def _format_close_time(close_time) -> str:
    if not close_time:
        return "unknown"
    try:
        if hasattr(close_time, "isoformat"):
            return close_time.isoformat()
        return str(close_time)
    except Exception:
        return str(close_time)


def _is_closed(close_time) -> bool:
    """Return True if the event has already closed."""
    try:
        if close_time is None:
            return False
        if isinstance(close_time, str):
            # Try parsing ISO 8601
            close_time = datetime.fromisoformat(close_time.replace("Z", "+00:00"))
        if hasattr(close_time, "tzinfo"):
            now = datetime.now(timezone.utc)
            if close_time.tzinfo is None:
                close_time = close_time.replace(tzinfo=timezone.utc)
            return close_time < now
    except Exception:
        pass
    return False


def _build_prompt(event: dict) -> str:
    category = event.get("category", "")
    hint = CATEGORY_HINTS.get(category, DEFAULT_HINT)
    close_time_str = _format_close_time(event.get("close_time"))
    now_str = datetime.now(timezone.utc).isoformat()

    return f"""You are an expert prediction market forecaster with deep knowledge across finance, politics, sports, science, and geopolitics.

Current UTC time: {now_str}

MARKET DETAILS
--------------
Title: {event.get("title", "N/A")}
Description: {event.get("description", "N/A")}
Category: {category or "N/A"}
Closes: {close_time_str}
Event ticker: {event.get("event_ticker", "N/A")}
Market ticker: {event.get("market_ticker", "N/A")}

DOMAIN GUIDANCE
---------------
{hint}

TASK
----
Estimate the probability that this market resolves YES.

Think carefully and systematically:
1. Identify the exact resolution condition.
2. Consider base rates and historical precedents.
3. Incorporate any domain-specific signals mentioned above.
4. Account for the time remaining until resolution.
5. Calibrate your confidence - avoid extreme values (below 0.03 or above 0.97) unless evidence is overwhelming.

Respond in this exact JSON format (no markdown, no extra text):
{{
  "p_yes": <float between 0.01 and 0.99>,
  "rationale": "<concise explanation, 1-3 sentences>"
}}"""


def predict(event: dict) -> dict:
    """Receive a prediction market event, return a probability estimate.

    Args:
        event: dict with keys: event_ticker, market_ticker, title,
               description, category, close_time, etc.

    Returns:
        dict with "p_yes" (float 0.01-0.99) and "rationale" (str).
        Returns {"p_yes": None} to skip if the event has closed or an
        unrecoverable error occurs.
    """
    # Skip already-closed events
    close_time = event.get("close_time")
    if _is_closed(close_time):
        return {
            "p_yes": None,
            "rationale": "Event has already closed; skipping.",
        }

    prompt = _build_prompt(event)

    try:
        response = client.responses.create(
            model="gpt-5-mini",
            input=[
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}],
                }
            ],
            max_output_tokens=500,
        )

        raw_text = (response.output_text or "").strip()
        if not raw_text:
            return {"p_yes": None, "rationale": "No text response from model."}

        # Parse JSON from the response (tolerant: strip markdown fences if present)
        json_str = raw_text
        json_str = re.sub(r"^```(?:json)?\\s*", "", json_str)
        json_str = re.sub(r"\\s*```$", "", json_str)

        parsed = json.loads(json_str)
        p_yes = float(parsed["p_yes"])

        # Clamp to valid range
        p_yes = max(0.01, min(0.99, p_yes))

        rationale: Optional[str] = parsed.get("rationale")

        result: dict = {"p_yes": p_yes}
        if rationale:
            result["rationale"] = str(rationale)
        return result

    except json.JSONDecodeError as exc:
        return {
            "p_yes": None,
            "rationale": f"JSON parse error: {exc}. Raw response: {raw_text[:200]}",
        }
    except APIError as exc:
        return {
            "p_yes": None,
            "rationale": f"API error: {exc}",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "p_yes": None,
            "rationale": f"Unexpected error: {exc}",
        }
