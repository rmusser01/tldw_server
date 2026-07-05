"""Failure contract objects used by web scraping compatibility adapters."""

from __future__ import annotations

from dataclasses import dataclass

from .statuses import WebScrapingStatus


@dataclass(frozen=True, slots=True)
class RuntimeFailure:
    """Sanitized failure data for compatibility conversion and future boundaries."""

    status: WebScrapingStatus
    public_message: str
    reason: str | None = None
    mode: str | None = None
    stage: str | None = None
    source: str | None = None
    backend: str | None = None
    provider: str | None = None

    def as_policy_fields(self) -> dict[str, str]:
        """Return legacy policy metadata fields for public API payloads."""
        fields: dict[str, str] = {}
        if self.reason:
            fields["policy_reason"] = self.reason
        if self.mode:
            fields["policy_mode"] = self.mode
        if self.stage:
            fields["policy_stage"] = self.stage
        if self.source:
            fields["policy_source"] = self.source
        return fields
