from datetime import datetime, timezone


def format_close_time(close_time) -> str:
    if not close_time:
        return "unknown"
    try:
        if hasattr(close_time, "isoformat"):
            return close_time.isoformat()
        return str(close_time)
    except Exception:
        return str(close_time)


def is_closed(close_time) -> bool:
    """Return True if the event has already closed."""
    try:
        if close_time is None:
            return False
        if isinstance(close_time, str):
            close_time = datetime.fromisoformat(close_time.replace("Z", "+00:00"))
        if hasattr(close_time, "tzinfo"):
            now = datetime.now(timezone.utc)
            if close_time.tzinfo is None:
                close_time = close_time.replace(tzinfo=timezone.utc)
            return close_time < now
    except Exception:
        pass
    return False
