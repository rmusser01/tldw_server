"""Analyzer-facing contracts for governed HTTP, browser, and tool probes."""

from __future__ import annotations

import math
from collections.abc import Mapping
from contextlib import AbstractAsyncContextManager as AsyncContextManager
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Protocol

_SAFE_ERROR_MESSAGES: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "policy_denied": frozenset({"Probe destination was denied."}),
        "policy_error": frozenset({"Probe destination was denied."}),
        "budget_exhausted": frozenset({"Probe budget exhausted."}),
        "timeout": frozenset({"Probe timed out."}),
        "unavailable": frozenset({"Probe capability is unavailable."}),
        "missing_dependency": frozenset({"Probe dependency is unavailable."}),
        "browser_transport_disabled": frozenset({"Safe browser transport is unavailable."}),
        "browser_transport_unattested": frozenset({"Safe browser transport is unavailable."}),
        "browser_transport_config_invalid": frozenset({"Safe browser transport is unavailable."}),
        "external_tool_disabled": frozenset({"External tool probing is disabled."}),
        "redirect_loop": frozenset({"Redirect loop detected."}),
        "invalid_redirect": frozenset({"Redirect target is invalid."}),
        "too_many_redirects": frozenset({"Redirect limit exceeded."}),
        "probe_error": frozenset({"Probe failed.", "HTTP probe failed."}),
    }
)


def _required_url(value: Any) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError("url is required")
    return normalized


def _freeze_string_mapping(value: Mapping[str, Any] | None) -> Mapping[str, str]:
    return MappingProxyType({str(key): str(item) for key, item in dict(value or {}).items()})


def _freeze_proxies(
    value: Mapping[str, str] | str | None,
) -> Mapping[str, str] | str | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return _freeze_string_mapping(value)
    return str(value)


def _positive_timeout(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("timeout_s must be a positive finite number")
    try:
        normalized = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("timeout_s must be a positive finite number") from exc
    if not math.isfinite(normalized) or normalized <= 0:
        raise ValueError("timeout_s must be a positive finite number")
    return normalized


def _viewport_dimension(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class ProbeHttpRequest:
    """Immutable HTTP request supplied to the governed probe adapter."""

    url: str
    headers: Mapping[str, str] = field(default_factory=dict)
    cookies: Mapping[str, str] = field(default_factory=dict)
    timeout_s: float | None = None
    impersonate: str | None = None
    proxies: Mapping[str, str] | str | None = None
    allow_redirects: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "url", _required_url(self.url))
        object.__setattr__(self, "headers", _freeze_string_mapping(self.headers))
        object.__setattr__(self, "cookies", _freeze_string_mapping(self.cookies))
        if self.timeout_s is not None:
            object.__setattr__(self, "timeout_s", _positive_timeout(self.timeout_s))
        if self.impersonate is not None:
            object.__setattr__(self, "impersonate", str(self.impersonate))
        object.__setattr__(self, "proxies", _freeze_proxies(self.proxies))
        object.__setattr__(self, "allow_redirects", bool(self.allow_redirects))


@dataclass(frozen=True, slots=True)
class ProbeHttpResponse:
    """Immutable response returned by the governed HTTP probe adapter."""

    url: str
    status: int
    headers: Mapping[str, str] = field(default_factory=dict)
    text: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "url", _required_url(self.url))
        object.__setattr__(self, "status", int(self.status))
        object.__setattr__(self, "headers", _freeze_string_mapping(self.headers))
        object.__setattr__(self, "text", str(self.text or ""))


@dataclass(frozen=True, slots=True)
class BrowserProbeOptions:
    """Immutable options used to create one governed browser page."""

    user_agent: str | None = None
    extra_headers: Mapping[str, str] = field(default_factory=dict)
    viewport_width: int = 1280
    viewport_height: int = 720
    block_resource_types: tuple[str, ...] = ()
    init_scripts: tuple[str, ...] = ()
    capture_requests: bool = False

    def __post_init__(self) -> None:
        if self.user_agent is not None:
            object.__setattr__(self, "user_agent", str(self.user_agent))
        object.__setattr__(
            self,
            "extra_headers",
            _freeze_string_mapping(self.extra_headers),
        )
        object.__setattr__(
            self,
            "viewport_width",
            _viewport_dimension(self.viewport_width, field_name="viewport_width"),
        )
        object.__setattr__(
            self,
            "viewport_height",
            _viewport_dimension(self.viewport_height, field_name="viewport_height"),
        )
        object.__setattr__(
            self,
            "block_resource_types",
            tuple(str(item) for item in (self.block_resource_types or ())),
        )
        object.__setattr__(
            self,
            "init_scripts",
            tuple(str(item) for item in (self.init_scripts or ())),
        )
        object.__setattr__(self, "capture_requests", bool(self.capture_requests))


@dataclass(frozen=True, slots=True)
class ExternalToolResult:
    """Captured external-tool process result retained inside the adapter boundary."""

    returncode: int
    stdout: str
    stderr: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "returncode", int(self.returncode))
        object.__setattr__(self, "stdout", str(self.stdout or ""))
        object.__setattr__(self, "stderr", str(self.stderr or ""))


class HttpProbe(Protocol):
    async def get(self, request: ProbeHttpRequest) -> ProbeHttpResponse:
        raise NotImplementedError


class BrowserProbePage(Protocol):
    async def goto(self, url: str, *, wait_until: str, timeout_ms: float) -> None:
        raise NotImplementedError

    async def reload(self, *, wait_until: str, timeout_ms: float) -> None:
        raise NotImplementedError

    async def wait_for_load_state(self, state: str, *, timeout_ms: float) -> None:
        raise NotImplementedError

    async def wait_for_timeout(self, timeout_ms: float) -> None:
        raise NotImplementedError

    async def content(self) -> str:
        raise NotImplementedError

    async def evaluate(self, expression: str, argument: Any = None) -> Any:
        raise NotImplementedError

    async def link_count(self) -> int:
        raise NotImplementedError

    async def link_is_visible(self, index: int) -> bool:
        raise NotImplementedError

    def captured_request_urls(self) -> tuple[str, ...]:
        raise NotImplementedError

    def clear_captured_request_urls(self) -> None:
        raise NotImplementedError


class BrowserProbe(Protocol):
    def open_page(
        self,
        options: BrowserProbeOptions,
    ) -> AsyncContextManager[BrowserProbePage]:
        raise NotImplementedError


class ExternalToolProbe(Protocol):
    async def run_waf(
        self,
        url: str,
        *,
        find_all: bool,
        enabled: bool | None,
    ) -> ExternalToolResult:
        raise NotImplementedError


class ProbeError(Exception):
    """Safe analyzer-scoped failure containing only stable public fields."""

    __slots__ = ("_error_code", "_public_message")

    def __init__(self, error_code: str, public_message: str) -> None:
        normalized_code = str(error_code)
        normalized_message = str(public_message)
        approved_messages = _SAFE_ERROR_MESSAGES.get(normalized_code)
        if approved_messages is None or normalized_message not in approved_messages:
            raise ValueError("unsupported probe error payload")
        self._error_code = normalized_code
        self._public_message = normalized_message
        super().__init__(self.public_message)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"error_code", "public_message", "args"}:
            raise AttributeError(f"{name} is immutable")
        super().__setattr__(name, value)

    @property
    def error_code(self) -> str:
        return self._error_code

    @property
    def public_message(self) -> str:
        return self._public_message


class ProbeBudgetExhausted(ProbeError):
    """Raised when an atomic probe reservation would exceed its limit."""

    def __init__(self) -> None:
        super().__init__("budget_exhausted", "Probe budget exhausted.")


class ProbeTimeout(ProbeError):
    """Raised for a local analyzer operation timeout."""

    def __init__(self) -> None:
        super().__init__("timeout", "Probe timed out.")


class ProbeUnavailable(ProbeError):
    """Raised when an optional probe capability cannot be used."""

    def __init__(
        self,
        *,
        error_code: Literal[
            "unavailable",
            "missing_dependency",
            "browser_transport_disabled",
            "browser_transport_unattested",
            "browser_transport_config_invalid",
        ] = "unavailable",
    ) -> None:
        if error_code == "missing_dependency":
            message = "Probe dependency is unavailable."
        elif error_code == "unavailable":
            message = "Probe capability is unavailable."
        else:
            message = "Safe browser transport is unavailable."
        super().__init__(error_code, message)
