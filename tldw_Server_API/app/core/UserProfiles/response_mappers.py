"""
Response mapping helpers for UserProfiles command flows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass(frozen=True)
class LegacyProfileCommandResult:
    status_code: int = 200
    profile_version: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    applied: tuple[str, ...] = ()
    skipped: tuple[dict[str, str], ...] = ()
    error_code: str | None = None
    detail: str | None = None
