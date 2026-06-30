"""
governance_utils.py

Pure utility functions for governance policy schedule and chat-type filtering.
No DB or service dependencies — suitable for use by both supervised_policy.py
and self_monitoring_service.py engines.
"""
from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from zoneinfo import ZoneInfo


_DAY_MAP = {
    "mon": 0, "monday": 0,
    "tue": 1, "tuesday": 1,
    "wed": 2, "wednesday": 2,
    "thu": 3, "thursday": 3,
    "fri": 4, "friday": 4,
    "sat": 5, "saturday": 5,
    "sun": 6, "sunday": 6,
}


def _parse_time_minutes(hhmm: str) -> int | None:
    """Parse 'HH:MM' into minutes since midnight. Returns None on failure."""
    try:
        parts = hhmm.strip().split(":")
        if len(parts) != 2:
            return None
        h, m = int(parts[0]), int(parts[1])
        if not (0 <= h <= 23 and 0 <= m <= 59):
            return None
        return h * 60 + m
    except (ValueError, TypeError, AttributeError):
        return None


def _get_tz(timezone_str: str | None) -> ZoneInfo:
    """Get a ZoneInfo object, falling back to UTC on any error."""
    from zoneinfo import ZoneInfo
    if not timezone_str:
        return ZoneInfo("UTC")
    try:
        return ZoneInfo(timezone_str)
    except (KeyError, ValueError, TypeError):
        logger.debug("Invalid schedule_timezone, falling back to UTC")
        return ZoneInfo("UTC")


def is_schedule_active(
    schedule_start: str | None,
    schedule_end: str | None,
    schedule_days: str | None,
    schedule_timezone: str | None,
) -> bool:
    """Check whether the current time falls within a governance policy's schedule.

    All-None/empty -> always active (return True).
    Fail-open: any parse error -> treat as active.
    """
    # All empty -> always active
    has_time = bool(schedule_start) or bool(schedule_end)
    has_days = bool(schedule_days)
    if not has_time and not has_days:
        return True

    try:
        tz = _get_tz(schedule_timezone)
        now = datetime.now(tz)

        # Day-of-week check
        if has_days:
            day_names = [d.strip().lower() for d in schedule_days.split(",") if d.strip()]
            if day_names:
                allowed_weekdays = set()
                for name in day_names:
                    weekday = _DAY_MAP.get(name)
                    if weekday is not None:
                        allowed_weekdays.add(weekday)
                if allowed_weekdays and now.weekday() not in allowed_weekdays:
                    return False

        # Time-of-day check
        if has_time:
            start_min = _parse_time_minutes(schedule_start) if schedule_start else None
            end_min = _parse_time_minutes(schedule_end) if schedule_end else None

            if start_min is not None and end_min is not None:
                current_min = now.hour * 60 + now.minute
                if start_min <= end_min:
                    # Normal range: e.g. 09:00-17:00
                    if not (start_min <= current_min < end_min):
                        return False
                else:
                    # Overnight range: e.g. 22:00-06:00
                    if not (current_min >= start_min or current_min < end_min):
                        return False
            # If only one bound is set or parse failed, fail-open (active)

    except Exception:
        logger.debug("Schedule evaluation error, failing open")
        return True

    return True


def chat_type_matches(
    scope_chat_types: str | None,
    chat_type: str | None,
) -> bool:
    """Check whether a chat_type falls within the governance policy's scope.

    scope_chat_types "all" or empty -> True.
    chat_type None -> defaults to "regular".
    Otherwise comma-split scope and check membership (case-insensitive).
    """
    if not scope_chat_types or scope_chat_types.strip().lower() == "all":
        return True

    effective_chat_type = (chat_type or "regular").strip().lower()
    allowed = {s.strip().lower() for s in scope_chat_types.split(",") if s.strip()}
    if not allowed:
        return True

    return effective_chat_type in allowed
