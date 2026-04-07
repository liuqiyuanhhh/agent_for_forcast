from datetime import datetime, timezone
from typing import Optional

from .time_utils import format_close_time

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


def build_prediction_prompt(event: dict, web_context: Optional[str]) -> str:
    category = event.get("category", "")
    hint = CATEGORY_HINTS.get(category, DEFAULT_HINT)
    close_time_str = format_close_time(event.get("close_time"))
    now_str = datetime.now(timezone.utc).isoformat()

    extra_context = web_context or "No web context available. Use only provided event details and prior knowledge."

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

WEB EVIDENCE SUMMARY
--------------------
{extra_context}

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
