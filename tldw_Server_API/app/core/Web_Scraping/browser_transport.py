"""Fail-closed admission contract for existing browser retrieval adapters."""

from __future__ import annotations

import configparser
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Literal, cast
from urllib.parse import unquote, urlsplit

from loguru import logger

from tldw_Server_API.app.core.config import (
    load_comprehensive_config,
    web_browser_transport_mode,
    web_outbound_policy_mode,
)
from tldw_Server_API.app.core.config_sections.auth import load_auth_config
from tldw_Server_API.app.core.Security.egress import public_url_policy_active

ConfiguredBrowserTransportMode = Literal[
    "auto",
    "disabled",
    "url_guarded",
    "attested_proxy",
]
EffectiveBrowserTransportMode = Literal[
    "disabled",
    "url_guarded",
    "attested_proxy",
]
BrowserTransportReason = Literal[
    "browser_transport_allowed_legacy",
    "browser_transport_allowed_attested",
    "browser_transport_disabled",
    "browser_transport_unattested",
    "browser_transport_config_invalid",
]

_CONFIGURED_MODES = frozenset({"auto", "disabled", "url_guarded", "attested_proxy"})


@dataclass(frozen=True, slots=True)
class BrowserTransportAttestation:
    """In-process evidence supplied only by a governed browser transport."""

    mechanism: Literal["governed_proxy"]
    routes_all_requests: bool
    dns_pinned: bool
    peer_verified: bool
    proxy_binding: Mapping[str, str] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.proxy_binding is not None:
            object.__setattr__(self, "proxy_binding", MappingProxyType(dict(self.proxy_binding)))


@dataclass(frozen=True, slots=True)
class BrowserTransportDecision:
    """Deterministic browser admission plus bounded public capability data."""

    allowed: bool
    configured_mode: ConfiguredBrowserTransportMode
    effective_mode: EffectiveBrowserTransportMode
    dns_peer_attested: bool
    reason: BrowserTransportReason
    proxy_binding: Mapping[str, str] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.proxy_binding is not None:
            object.__setattr__(self, "proxy_binding", MappingProxyType(dict(self.proxy_binding)))

    def to_capability_metadata(self) -> dict[str, str | bool]:
        """Return the fixed safe metadata contract without configuration details."""
        return {
            "name": "safe_browser_transport",
            "available": self.allowed,
            "configured_mode": self.configured_mode,
            "effective_mode": self.effective_mode,
            "dns_peer_attested": self.dns_peer_attested,
            "reason": self.reason,
        }


def decide_browser_transport(
    *,
    configured_mode: object,
    auth_mode: object,
    outbound_policy_mode: object,
    attestation: BrowserTransportAttestation | None = None,
) -> BrowserTransportDecision:
    """Decide whether browser transport is admissible for the supplied profile."""
    configured = _normalize_configured_mode(configured_mode)
    if configured is None:
        return BrowserTransportDecision(
            allowed=False,
            configured_mode="disabled",
            effective_mode="disabled",
            dns_peer_attested=False,
            reason="browser_transport_config_invalid",
        )
    if configured == "disabled":
        return BrowserTransportDecision(
            allowed=False,
            configured_mode=configured,
            effective_mode="disabled",
            dns_peer_attested=False,
            reason="browser_transport_disabled",
        )
    if configured == "attested_proxy":
        if _is_complete_attestation(attestation):
            return BrowserTransportDecision(
                allowed=True,
                configured_mode=configured,
                effective_mode="attested_proxy",
                dns_peer_attested=True,
                reason="browser_transport_allowed_attested",
                proxy_binding=attestation.proxy_binding,
            )
        return BrowserTransportDecision(
            allowed=False,
            configured_mode=configured,
            effective_mode="disabled",
            dns_peer_attested=False,
            reason="browser_transport_unattested",
        )

    auth = _normalize_profile_value(auth_mode)
    outbound = "strict" if public_url_policy_active() else _normalize_profile_value(outbound_policy_mode)
    if auth == "single_user" and outbound == "compat":
        return BrowserTransportDecision(
            allowed=True,
            configured_mode=configured,
            effective_mode="url_guarded",
            dns_peer_attested=False,
            reason="browser_transport_allowed_legacy",
        )
    return BrowserTransportDecision(
        allowed=False,
        configured_mode=configured,
        effective_mode="disabled",
        dns_peer_attested=False,
        reason="browser_transport_unattested",
    )


def default_browser_transport_decision(
    *,
    attestation: BrowserTransportAttestation | None = None,
    environ: Mapping[str, str] | None = None,
) -> BrowserTransportDecision:
    """Resolve the default profile without importing or initializing AuthNZ."""
    environment = os.environ if environ is None else environ
    try:
        configured_mode: object = web_browser_transport_mode(environment=environment)
        outbound_mode: object = web_outbound_policy_mode(environment=environment)
        auth_mode: object = _effective_auth_mode(environment)
    except Exception as exc:  # noqa: BLE001 - configuration errors must fail closed
        logger.bind(
            component="browser_transport",
            operation="resolve_default_configuration",
            exception_type=type(exc).__name__,
        ).warning("Browser transport configuration resolution failed.")
        configured_mode = None
        outbound_mode = None
        auth_mode = None
    return decide_browser_transport(
        configured_mode=configured_mode,
        auth_mode=auth_mode,
        outbound_policy_mode=outbound_mode,
        attestation=attestation,
    )


def resolve_browser_transport_decision(
    provider: Callable[[], BrowserTransportDecision],
    *,
    component: str,
) -> BrowserTransportDecision:
    """Resolve a provider fail-closed while logging only bounded diagnostics."""
    try:
        decision = provider()
    except Exception as exc:  # noqa: BLE001 - provider errors must fail closed
        logger.bind(
            component=component,
            operation="resolve_transport",
            exception_type=type(exc).__name__,
        ).warning("Browser transport decision provider failed.")
        return _invalid_browser_transport_decision()
    if not isinstance(decision, BrowserTransportDecision):
        return _invalid_browser_transport_decision()
    return decision


def browser_transport_failure_result(
    url: str,
    decision: BrowserTransportDecision,
) -> dict[str, Any]:
    """Return the fixed public failure contract for denied browser retrieval."""
    return {
        "url": url,
        "error": "browser_transport_unavailable",
        "extraction_successful": False,
        "capability": decision.to_capability_metadata(),
    }


def selected_browser_proxy(url: str, proxies: Mapping[str, str]) -> dict[str, str] | None:
    """Validate through shared HTTP policy and normalize the selected browser proxy."""
    if not proxies:
        return None
    from tldw_Server_API.app.core import http_client

    snapshot = dict(proxies)
    http_client.validate_proxies_or_raise(snapshot)
    selected = http_client._resolve_proxy_for_url(url, snapshot)
    if not selected:
        return None
    parsed = urlsplit(selected if "://" in selected else f"http://{selected}")
    if parsed.scheme not in {"http", "https", "socks4", "socks5"} or not parsed.hostname:
        raise ValueError("invalid browser proxy")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise ValueError("invalid browser proxy")
    host = f"[{parsed.hostname}]" if ":" in parsed.hostname else parsed.hostname
    server = f"{parsed.scheme}://{host}" + (f":{parsed.port}" if parsed.port is not None else "")
    result = {"server": server}
    if parsed.username is not None:
        result["username"] = unquote(parsed.username)
    if parsed.password is not None:
        result["password"] = unquote(parsed.password)
    return result


def bind_selected_browser_proxy(
    decision: BrowserTransportDecision, proxy: Mapping[str, str] | None
) -> BrowserTransportDecision:
    """Never treat allowlisting as evidence for a different attested transport."""
    if decision.allowed and decision.effective_mode == "attested_proxy" and proxy is not None:
        if decision.proxy_binding is None or dict(decision.proxy_binding) != dict(proxy):
            return replace(
                decision,
                allowed=False,
                effective_mode="disabled",
                dns_peer_attested=False,
                reason="browser_transport_unattested",
                proxy_binding=None,
            )
    return decision


def _invalid_browser_transport_decision() -> BrowserTransportDecision:
    """Return the fixed decision used for invalid provider output."""
    return BrowserTransportDecision(
        allowed=False,
        configured_mode="disabled",
        effective_mode="disabled",
        dns_peer_attested=False,
        reason="browser_transport_config_invalid",
    )


def _effective_auth_mode(environment: Mapping[str, str]) -> str:
    """Resolve absent auth mode canonically while rejecting explicit invalid input."""
    explicit = environment.get("AUTH_MODE")
    if explicit is not None:
        normalized = _normalize_profile_value(explicit)
        return normalized if normalized in {"single_user", "multi_user"} else ""
    config = load_comprehensive_config() or configparser.ConfigParser()
    return load_auth_config(config, environment).mode


def _normalize_configured_mode(
    value: object,
) -> ConfiguredBrowserTransportMode | None:
    """Normalize a supported configured mode without retaining invalid values."""
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if normalized not in _CONFIGURED_MODES:
        return None
    return cast(ConfiguredBrowserTransportMode, normalized)


def _normalize_profile_value(value: object) -> str:
    """Normalize a deployment-profile value to a lowercase string."""
    return value.strip().lower() if isinstance(value, str) else ""


def _is_complete_attestation(
    attestation: BrowserTransportAttestation | None,
) -> bool:
    """Return whether governed transport evidence covers every required property."""
    return (
        isinstance(attestation, BrowserTransportAttestation)
        and attestation.mechanism == "governed_proxy"
        and attestation.routes_all_requests is True
        and attestation.dns_pinned is True
        and attestation.peer_verified is True
    )
