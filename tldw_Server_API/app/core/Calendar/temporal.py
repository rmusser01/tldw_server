"""Provider duration arithmetic that distinguishes civil days from elapsed hours."""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta, timezone

from icalendar.prop import vDuration

from tldw_Server_API.app.core.Calendar.errors import CalendarValidationError


def add_ical_duration(start: date | datetime, value: str) -> date | datetime:
    """Add RFC 5545 nominal days first, then accurate elapsed hours/minutes/seconds."""
    try:
        duration = vDuration.from_ical(value)
        nominal = re.fullmatch(r"\+?P(?:(\d+)W|(\d+)D)?", value.upper().split("T", 1)[0])
        if nominal is None or duration <= timedelta(0):
            raise ValueError("Duration must be positive")
        days = int(nominal[1] or 0) * 7 + int(nominal[2] or 0)
        if not isinstance(start, datetime):
            if "T" in value.upper():
                raise ValueError("All-day duration must use weeks or days")
            return start + timedelta(days=days)
        end = start + timedelta(days=days) if days else start
        elapsed = duration - timedelta(days=days)
        if end.tzinfo is not None:
            return (end.astimezone(timezone.utc) + elapsed).astimezone(end.tzinfo)
        return end + elapsed
    except (TypeError, ValueError, OverflowError) as exc:
        raise CalendarValidationError("Invalid or out-of-range VEVENT duration") from exc


__all__ = ["add_ical_duration"]
