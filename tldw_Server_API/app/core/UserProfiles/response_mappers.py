"""
Response mapping helpers for UserProfiles command flows.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType


@dataclass(frozen=True)
class LegacyProfileCommandResult:
    status_code: int = 200
    profile_version: datetime | None = None
    applied: tuple[str, ...] = ()
    skipped: tuple[Mapping[str, str], ...] = field(default_factory=tuple)
    error_code: str | None = None
    detail: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "skipped",
            tuple(MappingProxyType(dict(item)) for item in self.skipped),
        )
