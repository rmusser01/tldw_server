"""Closed, source-free contracts for standalone HTML validation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StandaloneHtmlValidationResult:
    """Derived scalar metadata from one accepted standalone document."""

    title: str
    slide_count: int
    html_bytes: int
    html_sha256: str
    indexable_text: str


class StandaloneHtmlValidationError(RuntimeError):
    """Bounded public validation failure that never carries source text."""

    __slots__ = ("code", "status_code", "retry_after", "reason", "line", "column")

    def __init__(
        self,
        code: str,
        *,
        status_code: int,
        retry_after: int | None = None,
        reason: str | None = None,
        line: int | None = None,
        column: int | None = None,
    ) -> None:
        self.code = code
        self.status_code = status_code
        self.retry_after = retry_after
        self.reason = reason
        self.line = line if isinstance(line, int) and 1 <= line <= 1_000_000 else None
        self.column = column if isinstance(column, int) and 1 <= column <= 1_000_000 else None
        super().__init__(code)


__all__ = ["StandaloneHtmlValidationError", "StandaloneHtmlValidationResult"]
