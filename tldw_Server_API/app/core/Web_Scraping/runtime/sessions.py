"""Session and cookie contracts for Web_Scraping runtime adapters."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


def _freeze_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType({str(key): item for key, item in dict(value or {}).items()})


@dataclass(frozen=True, slots=True)
class RuntimeCookie:
    """Normalized cookie state for fetch and browser adapters."""

    name: str
    value: str
    domain: str | None = None
    path: str = "/"

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "value", str(self.value))
        if self.domain is not None:
            object.__setattr__(self, "domain", str(self.domain))
        object.__setattr__(self, "path", str(self.path or "/"))


@dataclass(frozen=True, slots=True)
class RuntimeSessionState:
    """Immutable session state passed into runtime adapters."""

    cookies: tuple[RuntimeCookie, ...] = ()
    headers: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "cookies", tuple(self.cookies or ()))
        object.__setattr__(self, "headers", _freeze_mapping(self.headers))
