"""Immutable planning contracts for governed article acquisition."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from tldw_Server_API.app.core.Web_Scraping.scraper_router import DEFAULT_HANDLER
from tldw_Server_API.app.core.Web_Scraping.ua_profiles import build_browser_headers

DEFAULT_MAX_ARTICLE_BYTES = 16_777_216
DEFAULT_MAX_BROWSER_TRANSFER_BYTES = 67_108_864
MAX_CONFIGURED_RESPONSE_BYTES = 1_073_741_824
DEFAULT_BROWSER_RETRIES = 3
DEFAULT_BROWSER_TIMEOUT_MS = 60_000
DEFAULT_STEALTH_WAIT_MS = 5_000

PUBLIC_FAILURE_CODES = frozenset(
    {
        "policy_error",
        "regex_invalid",
        "regex_too_large",
        "regex_timeout",
        "selector_invalid",
        "provider_error",
        "fetch_error",
        "browser_error",
        "browser_transport_unavailable",
        "response_too_large",
        "extraction_error",
    }
)

_BACKENDS = frozenset({"auto", "curl", "httpx", "playwright"})
_TRUE_STRINGS = frozenset({"1", "true", "yes", "y", "on"})
_BROWSER_TRANSPORT_CAPABILITY_KEYS = frozenset(
    {
        "name",
        "available",
        "configured_mode",
        "effective_mode",
        "dns_peer_attested",
        "reason",
    }
)
_BROWSER_TRANSPORT_CONFIGURED_MODES = frozenset(
    {"auto", "disabled", "url_guarded", "attested_proxy"}
)
_BROWSER_TRANSPORT_DENIAL_REASONS = frozenset(
    {
        "browser_transport_disabled",
        "browser_transport_unattested",
        "browser_transport_config_invalid",
    }
)
_INVALID_BROWSER_TRANSPORT_CAPABILITY: Mapping[str, str | bool] = MappingProxyType(
    {
        "name": "safe_browser_transport",
        "available": False,
        "configured_mode": "disabled",
        "effective_mode": "disabled",
        "dns_peer_attested": False,
        "reason": "browser_transport_config_invalid",
    }
)


def _freeze_value(value: Any) -> Any:
    """Recursively copy mutable request values into immutable equivalents."""
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if isinstance(value, list | tuple):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, set | frozenset):
        return frozenset(_freeze_value(item) for item in value)
    if isinstance(value, bytearray | memoryview):
        return bytes(value)
    return value


def _freeze_mapping(value: Mapping[Any, Any] | None) -> Mapping[str, Any]:
    """Return an immutable string-keyed snapshot of a mapping."""
    if not isinstance(value, Mapping):
        return MappingProxyType({})
    return MappingProxyType({str(key): _freeze_value(item) for key, item in value.items()})


def _positive_integer_or_default(value: Any, default: int) -> int:
    """Accept positive integer values without applying a domain-specific cap."""
    if type(value) is int:
        return value if value > 0 else default
    if type(value) is str:
        normalized = value.strip()
        if normalized.isascii() and normalized.isdecimal():
            try:
                parsed = int(normalized)
            except (ValueError, OverflowError):
                return default
            return parsed if parsed > 0 else default
    return default


def _response_budget_or_default(value: Any, default: int) -> int:
    """Accept positive response budgets up to the explicit server ceiling."""
    if type(value) is int:
        return value if 0 < value <= MAX_CONFIGURED_RESPONSE_BYTES else default
    if type(value) is str:
        normalized = value.strip()
        if (
            normalized.isascii()
            and normalized.isdecimal()
            and len(normalized) <= len(str(MAX_CONFIGURED_RESPONSE_BYTES))
        ):
            try:
                parsed = int(normalized)
            except (ValueError, OverflowError):
                return default
            return parsed if 0 < parsed <= MAX_CONFIGURED_RESPONSE_BYTES else default
    return default


def _legacy_non_negative_integer(value: Any, default: int) -> int:
    """Preserve the browser's existing non-negative numeric normalization."""
    if isinstance(value, bool):
        return default
    try:
        normalized = int(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return max(0, normalized)


def _legacy_bool(value: Any, default: bool = False) -> bool:
    """Normalize the existing string-or-boolean stealth configuration."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in _TRUE_STRINGS
    if value is None:
        return default
    return bool(value)


def _config_values(config: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Normalize direct or full loaded config without changing the input mapping."""
    if not isinstance(config, Mapping):
        return MappingProxyType({})
    legacy = config.get("web_scraper")
    raw = config.get("Web-Scraping")
    if not isinstance(legacy, Mapping) and not isinstance(raw, Mapping):
        return config

    values = dict(legacy) if isinstance(legacy, Mapping) else {}
    if isinstance(raw, Mapping):
        for key in (
            "web_scraper_max_article_bytes",
            "web_scraper_max_browser_transfer_bytes",
            "stealth_wait_ms",
        ):
            if key in raw:
                values[key] = raw[key]
    return MappingProxyType(values)


def _mapping_strings(value: Any) -> dict[str, str]:
    """Copy mapping values into the string form expected by HTTP acquisition."""
    if not isinstance(value, Mapping):
        return {}
    return {str(key): str(item) for key, item in value.items()}


def _strategy_order(value: Any) -> tuple[str, ...] | None:
    """Snapshot the optional routing strategy list without changing its meaning."""
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return None
    return tuple(str(item) for item in value)


@dataclass(frozen=True, slots=True)
class ArticleLimits:
    """Response-size budgets captured once for one article request."""

    max_article_bytes: int = DEFAULT_MAX_ARTICLE_BYTES
    max_browser_transfer_bytes: int = DEFAULT_MAX_BROWSER_TRANSFER_BYTES

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "max_article_bytes",
            _response_budget_or_default(self.max_article_bytes, DEFAULT_MAX_ARTICLE_BYTES),
        )
        object.__setattr__(
            self,
            "max_browser_transfer_bytes",
            _response_budget_or_default(
                self.max_browser_transfer_bytes,
                DEFAULT_MAX_BROWSER_TRANSFER_BYTES,
            ),
        )

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any] | None) -> ArticleLimits:
        """Build immutable limits from the existing Web-Scraper settings mapping."""
        values = _config_values(config)
        return cls(
            max_article_bytes=_response_budget_or_default(
                values.get("web_scraper_max_article_bytes"),
                DEFAULT_MAX_ARTICLE_BYTES,
            ),
            max_browser_transfer_bytes=_response_budget_or_default(
                values.get("web_scraper_max_browser_transfer_bytes"),
                DEFAULT_MAX_BROWSER_TRANSFER_BYTES,
            ),
        )

    from_config = from_mapping


@dataclass(frozen=True, slots=True)
class DirectBrowserProfile:
    """Direct Playwright inputs deliberately isolated from lightweight settings."""

    user_agent: str
    custom_cookies: tuple[Mapping[str, Any], ...]
    retries: int
    timeout_ms: int
    stealth_enabled: bool
    stealth_wait_ms: int
    viewport_width: int = 1280
    viewport_height: int = 720

    def __post_init__(self) -> None:
        object.__setattr__(self, "user_agent", str(self.user_agent or ""))
        cookies = self.custom_cookies or ()
        object.__setattr__(
            self,
            "custom_cookies",
            tuple(_freeze_mapping(cookie) for cookie in cookies if isinstance(cookie, Mapping)),
        )
        object.__setattr__(self, "retries", _legacy_non_negative_integer(self.retries, DEFAULT_BROWSER_RETRIES))
        object.__setattr__(
            self, "timeout_ms", _legacy_non_negative_integer(self.timeout_ms, DEFAULT_BROWSER_TIMEOUT_MS)
        )
        object.__setattr__(self, "stealth_enabled", _legacy_bool(self.stealth_enabled))
        object.__setattr__(
            self,
            "stealth_wait_ms",
            _legacy_non_negative_integer(self.stealth_wait_ms, DEFAULT_STEALTH_WAIT_MS),
        )
        object.__setattr__(self, "viewport_width", _positive_integer_or_default(self.viewport_width, 1280))
        object.__setattr__(self, "viewport_height", _positive_integer_or_default(self.viewport_height, 720))


@dataclass(frozen=True, slots=True)
class ArticlePlan:
    """A fully snapshotted route, extraction, limit, and browser request plan."""

    url: str
    domain: str
    browser: DirectBrowserProfile
    backend: str = "auto"
    handler: str = DEFAULT_HANDLER
    ua_profile: str = "chrome_120_win"
    impersonate: str | None = None
    headers: Mapping[str, str] = field(default_factory=dict)
    cookies: Mapping[str, str] = field(default_factory=dict)
    proxies: Mapping[str, str] = field(default_factory=dict)
    respect_robots: bool = True
    strategy_order: tuple[str, ...] | None = None
    schema_rules: Mapping[str, Any] | None = None
    llm_settings: Mapping[str, Any] | None = None
    regex_settings: Mapping[str, Any] | None = None
    cluster_settings: Mapping[str, Any] | None = None
    limits: ArticleLimits = field(default_factory=ArticleLimits)

    def __post_init__(self) -> None:
        object.__setattr__(self, "url", str(self.url or "").strip())
        object.__setattr__(self, "domain", str(self.domain or "").strip().lower())
        backend = str(self.backend or "auto").strip().lower()
        object.__setattr__(self, "backend", backend if backend in _BACKENDS else "auto")
        object.__setattr__(self, "handler", str(self.handler or DEFAULT_HANDLER))
        object.__setattr__(self, "ua_profile", str(self.ua_profile or "chrome_120_win"))
        if self.impersonate is not None:
            object.__setattr__(self, "impersonate", str(self.impersonate))
        object.__setattr__(self, "headers", _freeze_mapping(self.headers))
        object.__setattr__(self, "cookies", _freeze_mapping(self.cookies))
        object.__setattr__(self, "proxies", _freeze_mapping(self.proxies))
        object.__setattr__(self, "respect_robots", bool(self.respect_robots))
        object.__setattr__(self, "strategy_order", _strategy_order(self.strategy_order))
        for field_name in ("schema_rules", "llm_settings", "regex_settings", "cluster_settings"):
            value = getattr(self, field_name)
            object.__setattr__(self, field_name, _freeze_mapping(value) if value is not None else None)
        object.__setattr__(
            self,
            "limits",
            self.limits if isinstance(self.limits, ArticleLimits) else ArticleLimits(),
        )

    @property
    def direct_browser(self) -> DirectBrowserProfile:
        """Return the browser-only profile for direct Playwright acquisition."""
        return self.browser

    @classmethod
    def from_routing_plan(
        cls,
        routing_plan: Any,
        config: Mapping[str, Any] | None,
        custom_cookies: Sequence[Mapping[str, Any]] | None = None,
    ) -> ArticlePlan:
        """Copy a mutable router plan into one immutable article request snapshot."""
        values = _config_values(config)
        ua_profile = str(getattr(routing_plan, "ua_profile", "chrome_120_win") or "chrome_120_win")
        headers = build_browser_headers(ua_profile, accept_lang="en-US,en;q=0.9")
        headers.update(_mapping_strings(getattr(routing_plan, "extra_headers", None)))
        configured_backend = str(values.get("web_scraper_default_backend", "auto") or "auto").strip().lower()
        requested_backend = str(getattr(routing_plan, "backend", "auto") or "auto").strip().lower()
        backend = (
            configured_backend if requested_backend == "auto" and configured_backend in _BACKENDS else requested_backend
        )
        retries = _legacy_non_negative_integer(
            values.get("web_scraper_retry_count"),
            DEFAULT_BROWSER_RETRIES,
        )
        timeout_seconds = _legacy_non_negative_integer(
            values.get("web_scraper_retry_timeout"),
            DEFAULT_BROWSER_TIMEOUT_MS // 1000,
        )
        stealth_wait_ms = _legacy_non_negative_integer(
            values.get("stealth_wait_ms", values.get("STEALTH_WAIT_MS", values.get("web_scraper_stealth_wait_ms"))),
            DEFAULT_STEALTH_WAIT_MS,
        )
        browser = DirectBrowserProfile(
            user_agent=headers.get("User-Agent", ""),
            custom_cookies=tuple(custom_cookies or ()),
            retries=retries,
            timeout_ms=timeout_seconds * 1000,
            stealth_enabled=_legacy_bool(values.get("web_scraper_stealth_playwright")),
            stealth_wait_ms=stealth_wait_ms,
        )
        return cls(
            url=getattr(routing_plan, "url", ""),
            domain=getattr(routing_plan, "domain", ""),
            browser=browser,
            backend=backend,
            handler=getattr(routing_plan, "handler", DEFAULT_HANDLER),
            ua_profile=ua_profile,
            impersonate=getattr(routing_plan, "impersonate", None),
            headers=headers,
            cookies=_mapping_strings(getattr(routing_plan, "cookies", None)),
            proxies=_mapping_strings(getattr(routing_plan, "proxies", None)),
            respect_robots=bool(getattr(routing_plan, "respect_robots", True)),
            strategy_order=_strategy_order(getattr(routing_plan, "strategy_order", None)),
            schema_rules=getattr(routing_plan, "schema_rules", None),
            llm_settings=getattr(routing_plan, "llm_settings", None),
            regex_settings=getattr(routing_plan, "regex_settings", None),
            cluster_settings=getattr(routing_plan, "cluster_settings", None),
            limits=ArticleLimits.from_mapping(values),
        )


class ArticleFailure(Exception):
    """A stable public failure code with an internal orchestration stage."""

    def __init__(
        self,
        code: str,
        stage: str,
        *,
        capability: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(code)
        self.code = code
        self.stage = stage
        self.capability = _freeze_mapping(capability) if capability is not None else None
        self.retry_suppressed = False


def _validated_browser_transport_capability(
    value: Mapping[str, Any] | None,
) -> dict[str, str | bool]:
    """Return only a coherent bounded denial capability, or a safe fallback."""
    invalid = dict(_INVALID_BROWSER_TRANSPORT_CAPABILITY)
    if not isinstance(value, Mapping) or frozenset(value) != _BROWSER_TRANSPORT_CAPABILITY_KEYS:
        return invalid
    configured_mode = value.get("configured_mode")
    reason = value.get("reason")
    if (
        value.get("name") != "safe_browser_transport"
        or value.get("available") is not False
        or type(configured_mode) is not str
        or configured_mode not in _BROWSER_TRANSPORT_CONFIGURED_MODES
        or value.get("effective_mode") != "disabled"
        or value.get("dns_peer_attested") is not False
        or type(reason) is not str
        or reason not in _BROWSER_TRANSPORT_DENIAL_REASONS
    ):
        return invalid
    return {
        "name": "safe_browser_transport",
        "available": False,
        "configured_mode": configured_mode,
        "effective_mode": "disabled",
        "dns_peer_attested": False,
        "reason": reason,
    }


def article_failure_result(failure: ArticleFailure | str) -> dict[str, Any]:
    """Return the stable sanitized article failure dictionary for public callers."""
    code = failure.code if isinstance(failure, ArticleFailure) else failure
    safe_code = code if type(code) is str and code in PUBLIC_FAILURE_CODES else "extraction_error"
    result: dict[str, Any] = {
        "title": "N/A",
        "author": "N/A",
        "date": "N/A",
        "content": "",
        "extraction_successful": False,
        "error": safe_code,
    }
    if safe_code == "browser_transport_unavailable":
        capability = failure.capability if isinstance(failure, ArticleFailure) else None
        result["capability"] = _validated_browser_transport_capability(capability)
    return result


__all__ = [
    "DEFAULT_MAX_ARTICLE_BYTES",
    "DEFAULT_MAX_BROWSER_TRANSFER_BYTES",
    "PUBLIC_FAILURE_CODES",
    "ArticleFailure",
    "ArticleLimits",
    "ArticlePlan",
    "DirectBrowserProfile",
    "article_failure_result",
]
