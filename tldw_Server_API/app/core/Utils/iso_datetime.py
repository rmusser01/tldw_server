"""UTC-aware ISO-8601 timestamp helpers.

One rule for every caller: results are timezone-aware UTC. A naive input is
taken to already be UTC (what SQLite ``CURRENT_TIMESTAMP`` and the legacy
``datetime.utcnow()`` writers produce), whatever format it arrived in.
Unparseable input returns ``None``; the parser never substitutes "now", so the
caller decides what a missing timestamp means.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def utc_now_iso() -> str:
    """Current time as an aware UTC ISO-8601 string (``...+00:00``)."""
    return datetime.now(timezone.utc).isoformat()


def parse_iso_utc(value: Any) -> datetime | None:
    """Parse a datetime or ISO-8601 string into an aware UTC datetime.

    Accepts ``T`` or space separators, fractional seconds, a trailing ``Z`` and
    explicit offsets. Returns ``None`` for ``None``, blank or unparseable input.
    """
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value.strip():
        try:
            parsed = datetime.fromisoformat(value.strip())
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)
