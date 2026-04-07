# agent_for_forcast

A lightweight forecasting agent that uses OpenAI web search + LLM reasoning to estimate event probability (`p_yes`).

## Features
- Web evidence gathering via OpenAI `web_search_preview`
- Domain-aware prompt construction
- JSON-schema constrained forecast output
- Simple local runner for event JSON files
- Env-based tuning for model and output lengths

## Project Structure
- `src/agent.py`: main `predict(event)` pipeline
- `src/evidence.py`: web evidence retrieval/summarization
- `src/prompting.py`: final forecasting prompt builder
- `src/time_utils.py`: close-time utilities
- `src/config.py`: env-driven settings
- `src/env_loader.py`: `.env` loader
- `run_agent.py`: local CLI runner
- `events.sample.json`: sample input
- `AGENT_OUTLINE.md`: design overview

## Environment
Create `.env` in project root:
```bash
OPENAI_API_KEY=sk-...
OPENAI_SEARCH_MODEL=gpt-4.1-mini
OPENAI_PREDICTION_MODEL=gpt-4.1-nano
OPENAI_SEARCH_EVIDENCE_WORD_LIMIT=220
OPENAI_SEARCH_MAX_OUTPUT_TOKENS=500
OPENAI_PREDICTION_MAX_OUTPUT_TOKENS=500
```

## Run
```bash
cd /home/qiyuanliu/agent_for_forcast
source .venv/bin/activate
python run_agent.py events.sample.json
```

If no path is provided, `run_agent.py` defaults to `events.sample.json`.

## Input Format
`run_agent.py` accepts either:
- a list of events
- a single event object

Example event:
```json
{
  "event_ticker": "EVT-BTC-2026-12-31",
  "market_ticker": "MKT-BTC-ABOVE-100K-2026",
  "title": "Will Bitcoin close above $100,000 on December 31, 2026?",
  "description": "Resolves YES if BTC/USD spot price is above 100,000 at 23:59:59 UTC on 2026-12-31.",
  "category": "Financials",
  "close_time": "2026-12-31T23:59:59Z"
}
```

## Output Format
On success:
```json
{
  "p_yes": 0.63,
  "rationale": "..."
}
```

On skip/error:
```json
{
  "p_yes": null,
  "rationale": "..."
}
```
