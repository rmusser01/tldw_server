"""Typed VN script authoring errors."""

from __future__ import annotations

from typing import Any


class VNScriptAuthoringError(ValueError):
    """Error raised by pure VN script authoring helpers."""

    def __init__(
        self,
        code: str,
        message: str,
        status_code: int = 400,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.details = details or {}
