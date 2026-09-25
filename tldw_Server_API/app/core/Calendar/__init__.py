"""Calendar domain package."""

from __future__ import annotations

from .constants import (
    CALENDAR_DOMAIN,
    CALENDAR_SYNC_JOB_TYPE,
    DEFAULT_SYNC_LOOKAHEAD_DAYS,
    DEFAULT_SYNC_LOOKBACK_DAYS,
    MAX_EXPANDED_OCCURRENCES,
    MAX_QUERY_WINDOW_DAYS,
)
from .errors import (
    CalendarItemNotFound,
    CalendarNotFound,
    CalendarPermissionDenied,
    CalendarReadOnlyError,
    CalendarSyncError,
    CalendarValidationError,
)

__all__ = [
    "CALENDAR_DOMAIN",
    "CALENDAR_SYNC_JOB_TYPE",
    "DEFAULT_SYNC_LOOKAHEAD_DAYS",
    "DEFAULT_SYNC_LOOKBACK_DAYS",
    "MAX_EXPANDED_OCCURRENCES",
    "MAX_QUERY_WINDOW_DAYS",
    "CalendarItemNotFound",
    "CalendarNotFound",
    "CalendarPermissionDenied",
    "CalendarReadOnlyError",
    "CalendarSyncError",
    "CalendarValidationError",
]
