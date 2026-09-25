from __future__ import annotations

from datetime import datetime, timezone


def _strip_tzinfo(dt: datetime) -> datetime:
    """
    Strip timezone info for backend-agnostic timestamp storage.

    Converts an aware datetime to naive UTC. Naive datetimes are preferred for
    consistent storage across database backends, and stored naive values are
    read back as UTC, so an aware value is converted to UTC before its offset
    is dropped (dropping it alone would shift non-UTC inputs by their offset).

    Args:
        dt: A datetime object (aware or naive).

    Returns:
        A naive datetime in UTC.
    """
    return dt.astimezone(timezone.utc).replace(tzinfo=None) if dt.tzinfo else dt
