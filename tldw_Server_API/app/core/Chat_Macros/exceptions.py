"""Exceptions raised by the Chat_Macros core package."""


class MacroValidationError(ValueError):
    """Raised when a macro definition or invocation args fail validation."""


class MacroStorageError(RuntimeError):
    """Raised when macro definition or run storage fails."""


class MacroNotFoundError(MacroStorageError):
    """Raised when a requested macro definition or run record is missing."""


class MacroExecutionError(RuntimeError):
    """Raised when macro execution fails."""
