"""Typed outcomes for authoritative provider credential scope resolution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any


class ProviderScopeStatus(str, Enum):
    """Only authorized absence permits advancing credential precedence."""

    RESOLVED = "resolved"
    AUTHORIZED_ABSENT = "authorized_absent"
    UNAUTHORIZED = "unauthorized"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class ProviderScopeResult:
    """A frozen outcome with a detached, top-level read-only, repr-redacted record."""

    status: ProviderScopeStatus
    record: Mapping[str, Any] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Reject inconsistent outcomes and detach normalized records from callers."""
        if not isinstance(self.status, ProviderScopeStatus):
            raise ValueError("status must be a ProviderScopeStatus")
        if self.status is ProviderScopeStatus.RESOLVED:
            if not isinstance(self.record, Mapping):
                raise ValueError("resolved record must be a normalized Mapping")
            object.__setattr__(self, "record", MappingProxyType(dict(self.record)))
        elif self.record is not None:
            raise ValueError("only resolved status may carry a record")


def _validate_positive_id(value: int, name: str) -> None:
    """Reject bools, integer subclasses, coercible values, and nonpositive IDs."""
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive exact int")
