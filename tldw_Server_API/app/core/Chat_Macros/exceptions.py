"""Compatibility imports for exceptions centralized in the core package."""

from tldw_Server_API.app.core.exceptions import (
    MacroExecutionError,
    MacroNotFoundError,
    MacroStorageError,
    MacroValidationError,
)

__all__ = [
    "MacroExecutionError",
    "MacroNotFoundError",
    "MacroStorageError",
    "MacroValidationError",
]
