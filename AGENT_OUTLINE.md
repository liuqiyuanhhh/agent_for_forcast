# Agent Outline

## Goal
Predict `p_yes` for a prediction-market event using:
1. Web evidence gathering
2. Prompt-based reasoning
3. Schema-constrained JSON output

## Input/Output Contract
- Input: one event dict, typically with:
  - `title`
  - `description`
  - `category`
  - `close_time`
  - optional `event_ticker`, `market_ticker`
- Output:
  - success: `{"p_yes": float, "rationale": str}`
  - skip/error: `{"p_yes": None, "rationale": str}`

## Pipeline
1. `is_closed(close_time)` check
- If event already closed, return skip result.

2. Gather web evidence (`src/evidence.py`)
- Uses OpenAI Responses API + `web_search_preview`.
- Produces short evidence brief:
  - key facts
  - latest signals
  - uncertainties
  - source links

3. Build final forecast prompt (`src/prompting.py`)
- Injects event fields + domain hint + web evidence summary.

4. Forecast (`src/agent.py`)
- Uses OpenAI Chat Completions with JSON schema response format.
- Enforces object shape:
  - `p_yes` in `[0.01, 0.99]`
  - `rationale` string

5. Parse and clamp output
- Defensive parse and clamp ensure stable return shape.

## Main Modules
- `src/agent.py`: orchestration + model call + parse
- `src/evidence.py`: web evidence collection
- `src/prompting.py`: forecasting prompt template
- `src/time_utils.py`: close-time parsing/logic
- `src/config.py`: tunable model/token/word settings
- `src/env_loader.py`: load `.env`
- `run_agent.py`: CLI runner for local JSON events

## Tunable Parameters
From `.env`:
- `OPENAI_API_KEY`
- `OPENAI_SEARCH_MODEL`
- `OPENAI_PREDICTION_MODEL`
- `OPENAI_SEARCH_EVIDENCE_WORD_LIMIT`
- `OPENAI_SEARCH_MAX_OUTPUT_TOKENS`
- `OPENAI_PREDICTION_MAX_OUTPUT_TOKENS`

## Current Strengths
- End-to-end working baseline
- Web-informed forecasting
- Configurable cost/quality knobs
- Structured output contract

## Next Upgrades
- Add confidence score and uncertainty bands
- Add evidence quality scoring
- Add caching to cut search cost
- Add backtest harness (Brier/log-loss)
