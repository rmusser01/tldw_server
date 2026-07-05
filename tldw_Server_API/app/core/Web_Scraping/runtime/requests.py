"""Low-level runtime request contracts for Web_Scraping."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if isinstance(value, list | tuple):
        return tuple(_freeze_value(item) for item in value)
    return value


def _freeze_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType({str(key): _freeze_value(item) for key, item in dict(value or {}).items()})


def _freeze_proxy_value(value: Mapping[str, str] | str | None) -> Mapping[str, str] | str | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): str(item) for key, item in dict(value).items()})
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
    allow_redirects: bool = True
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
            object.__setattr__(self, "timeout", float(self.timeout))
        object.__setattr__(self, "backend", str(self.backend or "httpx").strip().lower() or "httpx")
        object.__setattr__(self, "allow_redirects", bool(self.allow_redirects))
        if self.impersonate is not None:
            object.__setattr__(self, "impersonate", str(self.impersonate))
        object.__setattr__(self, "proxies", _freeze_proxy_value(self.proxies))
