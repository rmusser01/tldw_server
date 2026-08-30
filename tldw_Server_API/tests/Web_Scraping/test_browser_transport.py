"""Tests for browser-transport admission and bounded capability metadata."""

from __future__ import annotations

import json

import pytest

from tldw_Server_API.app.core.Web_Scraping.browser_transport import (
    BrowserTransportAttestation,
    decide_browser_transport,
    default_browser_transport_decision,
)


pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("configured", "auth_mode", "policy_mode", "allowed", "reason"),
    [
        ("auto", "single_user", "compat", True, "browser_transport_allowed_legacy"),
        (
            "url_guarded",
            "single_user",
            "compat",
            True,
            "browser_transport_allowed_legacy",
        ),
        ("auto", "multi_user", "compat", False, "browser_transport_unattested"),
        ("auto", "single_user", "strict", False, "browser_transport_unattested"),
        (
            "url_guarded",
            "multi_user",
            "strict",
            False,
            "browser_transport_unattested",
        ),
        (
            "disabled",
            "single_user",
            "compat",
            False,
            "browser_transport_disabled",
        ),
        (
            "bogus",
            "single_user",
            "compat",
            False,
            "browser_transport_config_invalid",
        ),
    ],
)
def test_browser_transport_decision_matrix(
    configured: str,
    auth_mode: str,
    policy_mode: str,
    allowed: bool,
    reason: str,
) -> None:
    decision = decide_browser_transport(
        configured_mode=configured,
        auth_mode=auth_mode,
        outbound_policy_mode=policy_mode,
    )

    assert decision.allowed is allowed
    assert decision.reason == reason


@pytest.mark.parametrize(
    "attestation",
    [
        None,
        BrowserTransportAttestation(
            mechanism="governed_proxy",
            routes_all_requests=False,
            dns_pinned=True,
            peer_verified=True,
        ),
        BrowserTransportAttestation(
            mechanism="governed_proxy",
            routes_all_requests=True,
            dns_pinned=False,
            peer_verified=True,
        ),
        BrowserTransportAttestation(
            mechanism="governed_proxy",
            routes_all_requests=True,
            dns_pinned=True,
            peer_verified=False,
        ),
        BrowserTransportAttestation(
            mechanism="self_asserted",  # type: ignore[arg-type]
            routes_all_requests=True,
            dns_pinned=True,
            peer_verified=True,
        ),
    ],
)
def test_attested_proxy_denies_incomplete_or_unapproved_evidence(
    attestation: BrowserTransportAttestation | None,
) -> None:
    decision = decide_browser_transport(
        configured_mode="attested_proxy",
        auth_mode="multi_user",
        outbound_policy_mode="strict",
        attestation=attestation,
    )

    assert decision.allowed is False
    assert decision.effective_mode == "disabled"
    assert decision.dns_peer_attested is False
    assert decision.reason == "browser_transport_unattested"


def test_attested_proxy_allows_only_complete_governed_evidence() -> None:
    attestation = BrowserTransportAttestation(
        mechanism="governed_proxy",
        routes_all_requests=True,
        dns_pinned=True,
        peer_verified=True,
    )

    decision = decide_browser_transport(
        configured_mode="attested_proxy",
        auth_mode="multi_user",
        outbound_policy_mode="strict",
        attestation=attestation,
    )

    assert decision.allowed is True
    assert decision.effective_mode == "attested_proxy"
    assert decision.dns_peer_attested is True
    assert decision.reason == "browser_transport_allowed_attested"


def test_attestation_is_ignored_unless_proxy_mode_is_explicit() -> None:
    attestation = BrowserTransportAttestation(
        mechanism="governed_proxy",
        routes_all_requests=True,
        dns_pinned=True,
        peer_verified=True,
    )

    decision = decide_browser_transport(
        configured_mode="auto",
        auth_mode="multi_user",
        outbound_policy_mode="strict",
        attestation=attestation,
    )

    assert decision.allowed is False
    assert decision.reason == "browser_transport_unattested"


def test_malformed_mode_is_sanitized_and_metadata_is_exactly_bounded() -> None:
    decision = decide_browser_transport(
        configured_mode="https://proxy.internal/?credential=secret",
        auth_mode="multi_user-header-cookie-secret",
        outbound_policy_mode="strict",
    )

    metadata = decision.to_capability_metadata()

    assert metadata == {
        "name": "safe_browser_transport",
        "available": False,
        "configured_mode": "disabled",
        "effective_mode": "disabled",
        "dns_peer_attested": False,
        "reason": "browser_transport_config_invalid",
    }
    serialized = json.dumps(metadata).lower()
    for forbidden in (
        "proxy.internal",
        "https://",
        "address",
        "credential",
        "header",
        "cookie",
        "secret",
        "multi_user",
    ):
        assert forbidden not in serialized


def test_default_decision_uses_injected_auth_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    module = __import__(
        "tldw_Server_API.app.core.Web_Scraping.browser_transport",
        fromlist=["browser_transport"],
    )
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setattr(module, "web_browser_transport_mode", lambda: "auto")
    monkeypatch.setattr(module, "web_outbound_policy_mode", lambda: "compat")

    decision = default_browser_transport_decision(
        environ={"AUTH_MODE": "single_user"}
    )

    assert decision.allowed is True
    assert decision.reason == "browser_transport_allowed_legacy"


def test_default_decision_fails_closed_when_config_resolution_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = __import__(
        "tldw_Server_API.app.core.Web_Scraping.browser_transport",
        fromlist=["browser_transport"],
    )
    monkeypatch.setattr(
        module,
        "web_browser_transport_mode",
        lambda: (_ for _ in ()).throw(RuntimeError("config unavailable")),
    )

    decision = default_browser_transport_decision(
        environ={"AUTH_MODE": "single_user"}
    )

    assert decision.allowed is False
    assert decision.reason == "browser_transport_config_invalid"
