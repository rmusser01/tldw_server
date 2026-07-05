"""Low-level runtime response contracts for Web_Scraping."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


_FALSE_STRINGS = frozenset({"false", "0", "no", "off"})
_TRUE_STRINGS = frozenset({"true", "1", "yes", "on"})


def _normalize_bool(value: Any, *, field_name: str) -> bool:
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
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if isinstance(value, list | tuple):
        return tuple(_freeze_value(item) for item in value)
    return value


def _freeze_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType({str(key): _freeze_value(item) for key, item in dict(value or {}).items()})


def _raw_get(raw: Any, key: str, default: Any = None) -> Any:
    if isinstance(raw, Mapping):
        return raw.get(key, default)
    try:
        return raw[key]  # type: ignore[index]
    except (AttributeError, KeyError, LookupError, TypeError):
        pass
    value = getattr(raw, key, None)
    if value is not None:
        return value
    data = getattr(raw, "data", None)
    if isinstance(data, Mapping):
        return data.get(key, default)
    return default


@dataclass(frozen=True, slots=True)
class FetchResponse:
    """Normalized response from a runtime fetch adapter."""

    url: str
    status: int = 0
    headers: Mapping[str, Any] = field(default_factory=dict)
    text: str = ""
    backend: str = "httpx"
    elapsed_seconds: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "url", str(self.url or ""))
        object.__setattr__(self, "status", int(self.status or 0))
        object.__setattr__(self, "headers", _freeze_mapping(self.headers))
        object.__setattr__(self, "text", str(self.text or ""))
        object.__setattr__(self, "backend", str(self.backend or "httpx"))
        object.__setattr__(self, "elapsed_seconds", max(0.0, float(self.elapsed_seconds or 0.0)))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    @classmethod
    def from_raw(
        cls,
        raw: Any,
        *,
        fallback_url: str,
        fallback_backend: str | None = None,
        elapsed_seconds: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "FetchResponse":
        status = _raw_get(raw, "status")
        if status is None:
            status = _raw_get(raw, "status_code", 0)
        return cls(
            url=str(_raw_get(raw, "url", fallback_url) or fallback_url),
            status=int(status or 0),
            headers=dict(_raw_get(raw, "headers", {}) or {}),
            text=str(_raw_get(raw, "text", "") or ""),
            backend=str(_raw_get(raw, "backend", fallback_backend or "httpx") or fallback_backend or "httpx"),
            elapsed_seconds=float(elapsed_seconds or 0.0),
            metadata=metadata or {},
        )


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    """Policy decision shape consumed by runtime-aware scrape code."""

    allowed: bool | str
    mode: str
    reason: str
    stage: str
    source: str
    details: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "allowed", _normalize_bool(self.allowed, field_name="allowed"))
        object.__setattr__(self, "mode", str(self.mode or "compat"))
        object.__setattr__(self, "reason", str(self.reason or "allowed"))
        object.__setattr__(self, "stage", str(self.stage or "runtime"))
        object.__setattr__(self, "source", str(self.source or "web_scraping"))
        if self.details is not None:
            object.__setattr__(self, "details", _freeze_mapping(self.details))
