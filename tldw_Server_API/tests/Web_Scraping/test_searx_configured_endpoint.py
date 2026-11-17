"""Configured Searx authority never belongs to request-supplied endpoints."""

import httpx
import pytest

from tldw_Server_API.app.core import http_client
from tldw_Server_API.app.core.Security import egress
from tldw_Server_API.app.core.Web_Scraping import WebSearch_APIs as ws

pytestmark = pytest.mark.unit
LOCAL = "http://127.0.0.1:18990/search"


@pytest.fixture
def search_transport(monkeypatch):
    """Keep real checked egress while replacing only network I/O and jitter."""
    settings = {"search_engines": {"searx_search_api_url": LOCAL}}
    calls = []
    redirect = []
    for key in ("EGRESS_ALLOWLIST", "EGRESS_DENYLIST", "WORKFLOWS_EGRESS_ALLOWLIST", "WORKFLOWS_EGRESS_DENYLIST"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("WORKFLOWS_EGRESS_BLOCK_PRIVATE", "true")
    monkeypatch.setenv("WORKFLOWS_EGRESS_PROFILE", "strict")
    monkeypatch.setenv("WORKFLOWS_EGRESS_ALLOWED_PORTS", "80,443,8080")
    monkeypatch.setattr(ws, "load_and_log_configs", lambda: settings)
    monkeypatch.setattr(ws, "get_loaded_config", lambda: settings)
    monkeypatch.setattr(ws.time, "sleep", lambda _: None)

    def reply(request):
        calls.append(request.url)
        if redirect:
            return httpx.Response(302, headers={"Location": redirect[0]})
        return httpx.Response(
            200, json={"results": [{"title": "Example", "url": "https://example.com/", "content": "Example domain"}]}
        )

    fetch = ws.fetch
    with httpx.Client(transport=httpx.MockTransport(reply), trust_env=False) as client:
        monkeypatch.setattr(
            ws, "fetch", lambda **kwargs: fetch(client=client, retry=http_client.RetryPolicy(attempts=1), **kwargs)
        )
        yield settings, calls, redirect


@pytest.mark.parametrize("profile", ["strict", "permissive"])
def test_server_configured_searx_reaches_checked_transport(search_transport, monkeypatch, profile):
    monkeypatch.setenv("WORKFLOWS_EGRESS_PROFILE", profile)
    _, calls, _ = search_transport
    result = ws.search_web_searx("example", json_mode=True)
    assert result["results"][0]["link"] == "https://example.com/"  # nosec B101 # Controlled result contract.
    assert len(calls) == 1 and calls[0].host == "127.0.0.1"  # nosec B101 # Only the configured origin reached transport.


def test_same_origin_request_does_not_widen_server_scope(search_transport):
    _, calls, _ = search_transport
    ws.search_web_searx("example", searx_url="http://127.0.0.1:18990/other")
    assert len(calls) == 1 and calls[0].path == "/other"  # nosec B101 # Paths retain the existing exact-origin contract.


@pytest.mark.parametrize(
    "override", ["http://127.0.0.2:18990/search", "http://127.0.0.1:18991/search", "https://127.0.0.1:18990/search"]
)
def test_request_cannot_grant_another_local_origin(search_transport, override):
    _, calls, _ = search_transport
    with pytest.raises(ValueError, match="Blocked by outbound policy"):
        ws.search_web_searx("example", searx_url=override)
    assert calls == []  # nosec B101 # Request origins carry no local authority.


def test_stale_cached_config_does_not_grant_previous_origin(search_transport, monkeypatch):
    settings, calls, _ = search_transport
    old = {"search_engines": {"searx_search_api_url": LOCAL}}
    monkeypatch.setattr(ws, "get_loaded_config", lambda: old)
    settings["search_engines"]["searx_search_api_url"] = "http://127.0.0.2:18990/search"
    with pytest.raises(ValueError, match="Blocked by outbound policy"):
        ws.search_web_searx("example", searx_url=LOCAL)
    assert calls == []  # nosec B101 # Fresh server configuration owns authority.


@pytest.mark.parametrize("target", ["http://169.254.169.254:18990/search", "http://[::ffff:127.0.0.1]:18990/search"])
def test_forbidden_configured_addresses_stay_denied(search_transport, target):
    settings, calls, _ = search_transport
    settings["search_engines"]["searx_search_api_url"] = target
    with pytest.raises(ValueError, match="Blocked by outbound policy"):
        ws.search_web_searx("example")
    assert calls == []  # nosec B101 # Configured origin is not permission for forbidden address classes.


def test_global_denylist_wins_for_configured_searx(search_transport, monkeypatch):
    _, calls, _ = search_transport
    monkeypatch.setenv("EGRESS_DENYLIST", "127.0.0.1")
    with pytest.raises(ValueError, match="Blocked by outbound policy"):
        ws.search_web_searx("example")
    assert calls == []  # nosec B101 # Administrator denylist remains authoritative.


def test_configured_searx_cannot_escape_origin_on_redirect(search_transport):
    _, calls, redirect = search_transport
    redirect.append("http://127.0.0.2:18990/search")
    with pytest.raises(http_client.EgressPolicyError) as failure:
        ws.search_web_searx("example")
    assert failure.value.reason_code == "origin_mismatch"  # nosec B101 # Real checked redirect rejection.
    assert len(calls) == 1  # nosec B101 # No request reached the redirect target.


def test_public_capture_scope_cannot_gain_configured_searx_authority(search_transport):
    _, calls, _ = search_transport
    with egress.public_url_policy_scope(), pytest.raises(ValueError, match="Blocked by outbound policy"):
        ws.search_web_searx("example")
    assert calls == []  # nosec B101 # Public capture cannot inherit provider authority.


def test_ordinary_private_url_policy_remains_blocked_after_search(search_transport):
    ws.search_web_searx("example")
    assert egress.evaluate_url_policy(LOCAL).allowed is False  # nosec B101 # No ambient or global permission leaked.


def test_different_public_override_keeps_ordinary_policy(search_transport, monkeypatch):
    _, calls, _ = search_transport
    monkeypatch.setenv("EGRESS_ALLOWLIST", "8.8.8.8")
    ws.search_web_searx("example", searx_url="https://8.8.8.8/search")
    assert len(calls) == 1 and calls[0].host == "8.8.8.8"  # nosec B101 # A public override does not inherit the configured local scope.


@pytest.mark.parametrize("configured", ["", "not a URL", "http://user:secret@127.0.0.1:18990/search"])
def test_invalid_server_configuration_grants_no_private_authority(search_transport, configured):
    settings, calls, _ = search_transport
    settings["search_engines"]["searx_search_api_url"] = configured
    with pytest.raises(ValueError, match="Blocked by outbound policy"):
        ws.search_web_searx("example", searx_url=LOCAL)
    assert calls == []  # nosec B101 # Blank, malformed and credential-bearing configuration cannot authorize a request.


def test_configured_dns_change_is_rejected_before_redirect_transport(search_transport, monkeypatch):
    settings, calls, redirect = search_transport
    settings["search_engines"]["searx_search_api_url"] = "http://searx.internal:18990/search"
    redirect.append("http://searx.internal:18990/results")
    monkeypatch.setattr(egress, "_resolve_host_ips", lambda *_args, **_kwargs: ["127.0.0.2" if calls else "127.0.0.1"])
    with pytest.raises(http_client.EgressPolicyError) as failure:
        ws.search_web_searx("example")
    assert failure.value.reason_code == "dns_changed"  # nosec B101 # Redirect retains the checked transport's pinned DNS set.
    assert len(calls) == 1  # nosec B101 # Changed address never reached transport.
