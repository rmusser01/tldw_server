"""Calendar domain exceptions."""

from __future__ import annotations


class CalendarError(Exception):
    """Base class for Calendar domain errors."""


class CalendarNotFound(CalendarError):
    """Raised when a calendar row cannot be found."""


class CalendarPermissionDenied(CalendarError):
    """Raised when the caller is not allowed to perform a calendar action."""


class CalendarValidationError(CalendarError):
    """Raised when calendar input violates domain constraints."""


class CalendarItemNotFound(CalendarError):
    """Raised when a calendar item row cannot be found."""


class CalendarReadOnlyError(CalendarError):
    """Raised when a read-only calendar entity is mutated."""


class CalendarSyncError(CalendarError):
    """Raised when calendar sync state cannot be recorded or applied."""
