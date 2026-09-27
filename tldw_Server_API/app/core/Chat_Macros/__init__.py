"""Chat macro definition models and parsers."""

from .exceptions import MacroExecutionError, MacroNotFoundError, MacroStorageError, MacroValidationError
from .models import (
    MacroArgSpec,
    MacroBranchRecord,
    MacroContext,
    MacroDefinition,
    MacroExecution,
    MacroPermissions,
    MacroRunRecord,
    MacroStep,
    OutputProfile,
)
from .parser import load_macro_definition, parse_macro_args

__all__ = [
    "MacroArgSpec",
    "MacroBranchRecord",
    "MacroContext",
    "MacroDefinition",
    "MacroExecution",
    "MacroExecutionError",
    "MacroNotFoundError",
    "MacroPermissions",
    "MacroRunRecord",
    "MacroStep",
    "MacroStorageError",
    "MacroValidationError",
    "OutputProfile",
    "load_macro_definition",
    "parse_macro_args",
]
