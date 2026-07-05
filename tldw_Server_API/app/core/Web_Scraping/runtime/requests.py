"""Low-level runtime request contracts for Web_Scraping."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


_FALSE_STRINGS = frozenset({"false", "0", "no", "off"})
_TRUE_STRINGS = frozenset({"true", "1", "yes", "on"})


def _normalize_bool(value: Any, *, field_name: str) -> bool:
    """Normalize boolean values accepted at the runtime boundary."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUE_STRINGS:
            return True
        if normalized in _FALSE_STRINGS:
            return False
    raise ValueError(f"{field_name} must be a boolean or boolean-like string")


def _freeze_value(value: Any) -> Any:
    """Recursively freeze nested mappings and sequences."""
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if isinstance(value, list | tuple):
        return tuple(_freeze_value(item) for item in value)
    return value


def _freeze_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Return an immutable copy of a mapping with string keys."""
    if not value:
        return MappingProxyType({})
    return MappingProxyType({str(key): _freeze_value(item) for key, item in value.items()})


def _freeze_proxy_value(value: Mapping[str, str] | str | None) -> Mapping[str, str] | str | None:
    """Normalize proxy configuration while preserving string proxy URLs."""
    if value is None:
        return None
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): str(item) for key, item in value.items()})
    return str(value)


@dataclass(frozen=True, slots=True)
class RuntimeRequestContext:
    """Context metadata carried into low-level runtime operations."""

    source: str = "web_scraping"
    stage: str = "runtime"
    user_id: str | int | None = None
    request_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", str(self.source or "web_scraping"))
        object.__setattr__(self, "stage", str(self.stage or "runtime"))
        if self.user_id is not None:
            object.__setattr__(self, "user_id", str(self.user_id))
        if self.request_id is not None:
            object.__setattr__(self, "request_id", str(self.request_id))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class FetchRequest:
    """Low-level HTTP fetch request used by Web_Scraping runtime adapters."""

    url: str
    method: str = "GET"
    headers: Mapping[str, str] = field(default_factory=dict)
    cookies: Mapping[str, str] = field(default_factory=dict)
    timeout: float | None = None
    backend: str = "httpx"
    allow_redirects: bool | str = True
    impersonate: str | None = None
    proxies: Mapping[str, str] | str | None = None
    context: RuntimeRequestContext = field(default_factory=RuntimeRequestContext)

    def __post_init__(self) -> None:
        normalized_url = str(self.url or "").strip()
        if not normalized_url:
            raise ValueError("url is required")
        object.__setattr__(self, "url", normalized_url)
        object.__setattr__(self, "method", str(self.method or "GET").strip().upper() or "GET")
        object.__setattr__(self, "headers", _freeze_mapping(self.headers))
        object.__setattr__(self, "cookies", _freeze_mapping(self.cookies))
        if self.timeout is not None:
            if isinstance(self.timeout, bool):
                raise ValueError("timeout must be a float or int, not a boolean")
            normalized_timeout = float(self.timeout)
            if not math.isfinite(normalized_timeout):
                raise ValueError("timeout must be finite")
            if normalized_timeout < 0:
                raise ValueError("timeout must be non-negative")
            object.__setattr__(self, "timeout", normalized_timeout)
        object.__setattr__(self, "backend", str(self.backend or "httpx").strip().lower() or "httpx")
        object.__setattr__(
            self,
            "allow_redirects",
            _normalize_bool(self.allow_redirects, field_name="allow_redirects"),
        )
        if self.impersonate is not None:
            object.__setattr__(self, "impersonate", str(self.impersonate))
        object.__setattr__(self, "proxies", _freeze_proxy_value(self.proxies))
