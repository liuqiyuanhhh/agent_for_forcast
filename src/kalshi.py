import json
from typing import Optional, Tuple
from urllib.request import urlopen

from .config import KALSHI_API_BASE_URL


def _safe_float(value) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _clamp01(x: float) -> float:
    return max(0.01, min(0.99, x))


def _best_book_price(levels) -> Optional[float]:
    """Extract best price from orderbook levels like [[price, size], ...]."""
    if not levels:
        return None
    first = levels[0]
    if not isinstance(first, (list, tuple)) or not first:
        return None
    return _safe_float(first[0])


def prior_from_event_fields(event: dict) -> Tuple[Optional[float], Optional[str]]:
    """Get a market prior directly from event fields when available."""
    yes_bid = _safe_float(event.get("yes_bid"))
    yes_ask = _safe_float(event.get("yes_ask"))
    if yes_bid is not None and yes_ask is not None:
        return _clamp01((yes_bid + yes_ask) / 2.0), "event_yes_bid_ask_mid"

    for key in ("yes_price", "market_price", "p_yes_market", "last_price"):
        val = _safe_float(event.get(key))
        if val is not None:
            return _clamp01(val), f"event_{key}"

    return None, None


def prior_from_kalshi_api(event: dict, timeout_sec: int = 10) -> Tuple[Optional[float], Optional[str]]:
    """Fetch Kalshi market data by ticker and derive market-implied prior."""
    ticker = str(event.get("market_ticker") or "").strip()
    if not ticker:
        return None, None

    url = f"{KALSHI_API_BASE_URL}/markets/{ticker}"
    try:
        with urlopen(url, timeout=timeout_sec) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None, None

    market = payload.get("market", payload)

    yes_bid = _safe_float(market.get("yes_bid"))
    yes_ask = _safe_float(market.get("yes_ask"))
    if yes_bid is not None and yes_ask is not None:
        return _clamp01((yes_bid + yes_ask) / 2.0), "kalshi_yes_bid_ask_mid"

    for key in ("yes_price", "last_price"):
        val = _safe_float(market.get(key))
        if val is not None:
            return _clamp01(val), f"kalshi_{key}"

    # Fallback to orderbook when top-level price fields are empty.
    try:
        ob_url = f"{KALSHI_API_BASE_URL}/markets/{ticker}/orderbook"
        with urlopen(ob_url, timeout=timeout_sec) as resp:
            ob_payload = json.loads(resp.read().decode("utf-8"))
        orderbook = ob_payload.get("orderbook_fp", ob_payload)

        yes_levels = orderbook.get("yes_dollars") or []
        no_levels = orderbook.get("no_dollars") or []

        yes_px = _best_book_price(yes_levels)
        if yes_px is not None:
            return _clamp01(yes_px), "kalshi_orderbook_yes_dollars"

        no_px = _best_book_price(no_levels)
        if no_px is not None:
            return _clamp01(1.0 - no_px), "kalshi_orderbook_no_dollars"
    except Exception:
        pass

    return None, None
