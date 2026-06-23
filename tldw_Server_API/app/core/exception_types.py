"""Framework-neutral exception types shared by core services."""

from __future__ import annotations


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
