import json
import re
from typing import Optional

from openai import APIError, OpenAI

from .config import PREDICTION_MAX_OUTPUT_TOKENS, PREDICTION_MODEL
from .evidence import gather_openai_web_evidence
from .prompting import build_prediction_prompt
from .time_utils import is_closed

client = OpenAI()
FORECAST_RESPONSE_SCHEMA = {
    "name": "forecast_response",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "p_yes": {"type": "number", "minimum": 0.01, "maximum": 0.99},
            "rationale": {"type": "string"},
        },
        "required": ["p_yes", "rationale"],
        "additionalProperties": False,
    },
}


def _parse_prediction_json(raw_text: str) -> dict:
    json_str = raw_text.strip()
    json_str = re.sub(r"^```(?:json)?\s*", "", json_str)
    json_str = re.sub(r"\s*```$", "", json_str)

    parsed = json.loads(json_str)
    p_yes = float(parsed["p_yes"])
    p_yes = max(0.01, min(0.99, p_yes))

    result = {"p_yes": p_yes}
    rationale: Optional[str] = parsed.get("rationale")
    if rationale:
        result["rationale"] = str(rationale)
    return result


def _gather_hybrid_evidence(event: dict) -> Optional[str]:
    return gather_openai_web_evidence(client, event)


def predict(event: dict) -> dict:
    """Receive a prediction market event, return a probability estimate."""
    if is_closed(event.get("close_time")):
        return {
            "p_yes": None,
            "rationale": "Event has already closed; skipping.",
        }

    evidence = _gather_hybrid_evidence(event)
    prompt = build_prediction_prompt(event, web_context=evidence)

    try:
        response = client.chat.completions.create(
            model=PREDICTION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            response_format={
                "type": "json_schema",
                "json_schema": FORECAST_RESPONSE_SCHEMA,
            },
            max_tokens=PREDICTION_MAX_OUTPUT_TOKENS,
        )

        raw_text = (response.choices[0].message.content or "").strip()
        if not raw_text:
            return {"p_yes": None, "rationale": "No text response from model."}

        return _parse_prediction_json(raw_text)

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
    except Exception as exc:
        return {
            "p_yes": None,
            "rationale": f"Unexpected error: {exc}",
        }
