"""Fail-closed admission contract for existing browser retrieval adapters."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, cast

from tldw_Server_API.app.core.config import (
    web_browser_transport_mode,
    web_outbound_policy_mode,
)

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

_CONFIGURED_MODES = frozenset(
    {"auto", "disabled", "url_guarded", "attested_proxy"}
)


@dataclass(frozen=True, slots=True)
class BrowserTransportAttestation:
    """In-process evidence supplied only by a governed browser transport."""

    mechanism: Literal["governed_proxy"]
    routes_all_requests: bool
    dns_pinned: bool
    peer_verified: bool


@dataclass(frozen=True, slots=True)
class BrowserTransportDecision:
    """Deterministic browser admission plus bounded public capability data."""

    allowed: bool
    configured_mode: ConfiguredBrowserTransportMode
    effective_mode: EffectiveBrowserTransportMode
    dns_peer_attested: bool
    reason: BrowserTransportReason

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
            )
        return BrowserTransportDecision(
            allowed=False,
            configured_mode=configured,
            effective_mode="disabled",
            dns_peer_attested=False,
            reason="browser_transport_unattested",
        )

    auth = _normalize_profile_value(auth_mode)
    outbound = _normalize_profile_value(outbound_policy_mode)
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
        configured_mode: object = web_browser_transport_mode()
        outbound_mode: object = web_outbound_policy_mode()
    except Exception:
        configured_mode = None
        outbound_mode = None
    return decide_browser_transport(
        configured_mode=configured_mode,
        auth_mode=environment.get("AUTH_MODE"),
        outbound_policy_mode=outbound_mode,
        attestation=attestation,
    )


def _normalize_configured_mode(
    value: object,
) -> ConfiguredBrowserTransportMode | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if normalized not in _CONFIGURED_MODES:
        return None
    return cast(ConfiguredBrowserTransportMode, normalized)


def _normalize_profile_value(value: object) -> str:
    return value.strip().lower() if isinstance(value, str) else ""


def _is_complete_attestation(
    attestation: BrowserTransportAttestation | None,
) -> bool:
    return (
        isinstance(attestation, BrowserTransportAttestation)
        and attestation.mechanism == "governed_proxy"
        and attestation.routes_all_requests is True
        and attestation.dns_pinned is True
        and attestation.peer_verified is True
    )
