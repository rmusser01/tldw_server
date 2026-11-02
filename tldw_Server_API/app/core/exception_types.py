"""Framework-neutral exception types shared by core services."""

from __future__ import annotations

MODERATION_EXECUTION_REASONS = frozenset(
    {
        "deadline_exceeded",
        "capacity_exhausted",
        "size_exceeded",
        "worker_unavailable",
        "worker_crashed",
        "invalid_response",
        "cleanup_failed",
    }
)


class ModerationExecutionError(RuntimeError):
    """Content-free failure at the moderation execution boundary."""

    def __init__(self, reason: str) -> None:
        if not isinstance(reason, str) or reason not in MODERATION_EXECUTION_REASONS:
            raise ValueError("invalid moderation execution reason")
        self.reason = reason
        super().__init__(reason)


class ProviderUsageReservationError(Exception):
    """Stable, content-free reservation error without a backend exception graph."""

    _CODES = frozenset(
        {
            "invalid_reservation",
            "integer_overflow",
            "duplicate_execution",
            "quota_exceeded",
            "snapshot_unavailable",
            "reservation_not_found",
            "invalid_transition",
            "conflicting_replay",
            "actuals_exceed_reservation",
            "usage_write_failed",
            "reservation_store_unavailable",
        }
    )

    def __init__(self, code: str) -> None:
        """Expose only a recognized public failure code."""
        self.code = code if code in self._CODES else "reservation_store_unavailable"
        super().__init__(self.code)


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
