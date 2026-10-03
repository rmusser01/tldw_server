"""Compatibility exports for centrally defined Calendar domain exceptions."""

from tldw_Server_API.app.core.exception_types import (
    CalendarError,
    CalendarItemNotFound,
    CalendarNotFound,
    CalendarPermissionDenied,
    CalendarReadOnlyError,
    CalendarSyncError,
    CalendarValidationError,
)

__all__ = [
    "CalendarError",
    "CalendarItemNotFound",
    "CalendarNotFound",
    "CalendarPermissionDenied",
    "CalendarReadOnlyError",
    "CalendarSyncError",
    "CalendarValidationError",
]
