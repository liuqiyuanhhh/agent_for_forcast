import json
import re
from typing import Optional

from openai import APIError, OpenAI

from .config import (
    MODEL_WEIGHT,
    PREDICTION_MAX_OUTPUT_TOKENS,
    PREDICTION_MODEL,
    USE_MARKET_PRIOR,
)
from .evidence import gather_openai_web_evidence
from .kalshi import prior_from_event_fields, prior_from_kalshi_api
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


def _blend_with_market_prior(result: dict, event: dict) -> dict:
    """Blend model forecast with Kalshi market prior when available."""
    if not USE_MARKET_PRIOR:
        return result

    p_model = result.get("p_yes")
    if p_model is None:
        return result

    prior, source = prior_from_event_fields(event)
    if prior is None:
        prior, source = prior_from_kalshi_api(event)
    if prior is None:
        return result

    p_final = MODEL_WEIGHT * p_model + (1.0 - MODEL_WEIGHT) * prior
    p_final = max(0.01, min(0.99, p_final))

    out = dict(result)
    out["p_yes"] = p_final
    if "rationale" in out:
        out["rationale"] = (
            f"{out['rationale']} Blended with Kalshi prior ({source})={prior:.2f} "
            f"using model_weight={MODEL_WEIGHT:.2f}."
        )
    return out


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

        parsed = _parse_prediction_json(raw_text)
        return _blend_with_market_prior(parsed, event)

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
