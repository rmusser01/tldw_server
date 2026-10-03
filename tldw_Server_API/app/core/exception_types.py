"""Framework-neutral exception types shared by core services."""

from __future__ import annotations


class CalendarError(Exception):
    """Base class for Calendar domain errors."""


class CalendarNotFound(CalendarError):
    """A calendar or external account is unavailable."""


class CalendarItemNotFound(CalendarError):
    """A Calendar item or its local context is unavailable."""


class CalendarPermissionDenied(CalendarError):
    """The current principal cannot perform a Calendar operation."""


class CalendarReadOnlyError(CalendarError):
    """A provider-managed or projected Calendar entity cannot be mutated."""


class CalendarValidationError(CalendarError):
    """Calendar input violates a domain constraint."""


class CalendarSyncError(CalendarError):
    """Calendar synchronization state cannot be applied or recorded."""


class PersonaArtworkValidationError(ValueError):
    """A pack's artwork credit record or carrier violates the native contract.

    The message is a stable metadata error code, without imported record contents.
    ValueError compatibility preserves existing import and export error handling.
    """


class PromptCatalogError(Exception):
    """Sanitized prompt catalog error suitable for MCP protocol mapping."""

    def __init__(self, code: str, message: str, internal: bool = False) -> None:
        """Create a sanitized prompt catalog error.

        Args:
            code: Stable machine-readable prompt catalog error code.
            message: Safe public message for MCP protocol responses.
            internal: Whether the underlying failure should be mapped to a
                generic internal error for clients.
        """
        super().__init__(message)
        self.code = code
        self.message = message
        self.internal = internal
