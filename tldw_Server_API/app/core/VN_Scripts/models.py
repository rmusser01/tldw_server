"""Internal models for VN script validation and service responses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class VNScriptDiagnostic:
    """A script validation error or warning."""

    code: str
    message: str
    path: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable diagnostic."""
        return {
            "code": self.code,
            "message": self.message,
            "path": self.path,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class VNScriptValidationResult:
    """Result returned by the pure VN script validator."""

    valid: bool
    errors: list[dict[str, Any]]
    warnings: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable validation result."""
        return {
            "valid": self.valid,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }
