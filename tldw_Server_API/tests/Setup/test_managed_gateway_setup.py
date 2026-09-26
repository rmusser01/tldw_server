"""Behavioral setup checks for the authenticated private managed gateway."""

import os
import subprocess
import sys
from pathlib import Path

import httpx
import pytest
import yaml
from fastapi import FastAPI, HTTPException
from hypothesis import given, strategies as st
from starlette.requests import Request

from tldw_Server_API.app.api.v1.API_Deps import setup_deps
from tldw_Server_API.app.api.v1.endpoints import setup as setup_endpoint

SECRET = "managed-gateway-secret-with-at-least-32-characters"
ORIGIN = "http://127.0.0.1:18090"


@pytest.fixture(autouse=True)
def managed_environment(monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("TLDW_MANAGED_GATEWAY", "1")
    monkeypatch.setenv("TLDW_GATEWAY_HOP_SECRET", SECRET)
    monkeypatch.setenv("TLDW_MANAGED_PUBLIC_ORIGIN", ORIGIN)
    monkeypatch.delenv("TLDW_SETUP_ALLOW_REMOTE", raising=False)
    setup_deps.reset_remote_access_cache(False)
    monkeypatch.setattr(
        setup_endpoint.setup_manager,
        "get_status_snapshot",
        lambda: {
            "auth_mode": "single_user",
            "needs_setup": True,
        },
    )
    yield
    setup_deps.reset_remote_access_cache(None)


def gateway_request(*, peer="172.18.0.3", changes=None, extra=(), path="/api/v1/setup/first-run/metadata"):
    headers = {
        "host": "127.0.0.1:18090",
        "origin": ORIGIN,
        "x-tldw-gateway-hop": SECRET,
        "x-forwarded-for": "172.18.0.1",
        "x-forwarded-host": "127.0.0.1:18090",
        "x-forwarded-port": "18090",
        "x-forwarded-proto": "http",
    }
    for name, value in (changes or {}).items():
        if value is None:
            headers.pop(name, None)
        else:
            headers[name] = value
    return Request(
        {
            "type": "http",
            "scheme": "http",
            "method": "POST",
            "path": path,
            "server": ("app", 8000),
            "client": (peer, 40000),
            "query_string": b"",
            "headers": [(name.encode(), value.encode()) for name, value in headers.items()] + list(extra),
        }
    )


@pytest.mark.asyncio
async def test_measured_docker_hop_allows_setup_and_bundled_metadata():
    request = gateway_request()
    await setup_deps.require_local_setup_access(request)
    metadata = await setup_endpoint.get_first_run_metadata(request)
    assert metadata.connection.browser_access == "local"
    assert metadata.bundled_single_user_auth_available is True
    assert metadata.manual_auth_required is False
    assert SECRET not in metadata.model_dump_json()


@pytest.mark.asyncio
async def test_managed_metadata_uses_same_hop_validation():
    metadata = setup_endpoint.build_first_run_metadata(gateway_request())
    assert metadata.manual_auth_required is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "changes",
    [
        {"x-tldw-gateway-hop": None},
        {"x-tldw-gateway-hop": "wrong"},
        {"x-tldw-gateway-hop": "é" * 40},
        {"host": "127.0.0.1:18091"},
        {"host": "["},
        {"origin": "http://attacker.test"},
        {"origin": ORIGIN + "/"},
        {"x-forwarded-for": "8.8.8.8"},
        {"x-forwarded-for": "172.18.0.1, 127.0.0.1"},
        {"x-forwarded-for": "172.18.0.1:1234"},
        {"x-forwarded-for": " 172.18.0.1"},
        {"x-forwarded-for": "fe80::1%eth0"},
        {"x-forwarded-for": None},
        {"x-forwarded-host": "localhost:18090"},
        {"x-forwarded-port": "018090"},
        {"x-forwarded-port": None},
        {"x-forwarded-proto": "https"},
        {"forwarded": "for=172.18.0.1"},
        {"x-real-ip": "172.18.0.1"},
        {"x-forwarded-prefix": "/api"},
        {"x-tldw-gateway-extra": "forged"},
    ],
)
async def test_invalid_envelope_cannot_get_local_setup_or_bundled_metadata(changes):
    request = gateway_request(changes=changes)
    with pytest.raises(HTTPException) as error:
        await setup_deps.require_local_setup_access(request)
    assert error.value.status_code == 403
    assert setup_endpoint.build_first_run_metadata(request).manual_auth_required is True
    assert SECRET not in str(error.value.detail)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name",
    [
        "host",
        "origin",
        "x-tldw-gateway-hop",
        "x-forwarded-for",
        "x-forwarded-host",
        "x-forwarded-port",
        "x-forwarded-proto",
    ],
)
async def test_duplicate_envelope_headers_are_denied(name):
    request = gateway_request(extra=[(name.encode(), b"duplicate")])
    with pytest.raises(HTTPException):
        await setup_deps.require_local_setup_access(request)
    assert setup_endpoint.build_first_run_metadata(request).manual_auth_required is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name,value",
    [
        ("AUTH_MODE", "multi_user"),
        ("TLDW_MANAGED_GATEWAY", "0"),
        ("TLDW_GATEWAY_HOP_SECRET", "short"),
        ("TLDW_GATEWAY_HOP_SECRET", ""),
        ("TLDW_MANAGED_PUBLIC_ORIGIN", "http://example.test:18090"),
        ("TLDW_MANAGED_PUBLIC_ORIGIN", ORIGIN + "/"),
        ("TLDW_MANAGED_PUBLIC_ORIGIN", ""),
        ("TLDW_MANAGED_PUBLIC_ORIGIN", "http://127.0.0.1:99999"),
    ],
)
async def test_disabled_or_invalid_managed_configuration_is_denied(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    request = gateway_request()
    with pytest.raises(HTTPException):
        await setup_deps.require_local_setup_access(request)
    assert setup_endpoint.build_first_run_metadata(request).manual_auth_required is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "peer",
    [
        "8.8.8.8",
        "203.0.113.2",
        "192.0.2.3",
        "198.51.100.2",
        "169.254.1.1",
        "0.0.0.0",
        "100.64.0.1",
        "2001:db8::1",
        "fe80::1",
        "::",
    ],
)
async def test_public_and_reserved_peers_cannot_use_managed_capability(peer):
    request = gateway_request(peer=peer)
    with pytest.raises(HTTPException):
        await setup_deps.require_local_setup_access(request)
    assert setup_endpoint.build_first_run_metadata(request).manual_auth_required is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "peer,client",
    [
        ("127.0.0.1", "127.0.0.1"),
        ("10.0.0.2", "192.168.1.1"),
        ("fd12::2", "fd12::1"),
        ("::ffff:172.18.0.3", "::ffff:172.18.0.1"),
    ],
)
async def test_bounded_private_address_families_and_absent_origin_are_allowed(peer, client):
    request = gateway_request(peer=peer, changes={"x-forwarded-for": client, "origin": None})
    await setup_deps.require_local_setup_access(request)
    assert setup_endpoint.build_first_run_metadata(request).manual_auth_required is False


@pytest.mark.asyncio
@given(octet=st.integers(min_value=0, max_value=255))
async def test_documentation_address_range_never_gains_managed_setup(octet):
    request = gateway_request(changes={"x-forwarded-for": f"192.0.2.{octet}"})
    with pytest.raises(HTTPException):
        await setup_deps.require_local_setup_access(request)
    assert setup_endpoint.build_first_run_metadata(request).manual_auth_required is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path", ["/api/v1/media", "/api/v1/setup-other", "/api/v1/setup/%2e%2e/config", "/api/v1/setup//config"]
)
async def test_setup_capability_cannot_authorize_other_or_noncanonical_paths(path):
    with pytest.raises(HTTPException):
        await setup_deps.require_local_setup_access(gateway_request(path=path))


def test_managed_compose_origin_is_the_only_trusted_cookie_websocket_origin():
    compose_path = Path(__file__).resolve().parents[3] / "Dockerfiles/app-bundle/compose.yaml"
    environment = yaml.safe_load(compose_path.read_text())["services"]["app"]["environment"]
    allowed = environment["ALLOWED_ORIGINS"].replace("${TLDW_PUBLIC_PORT:?Public port is required}", "18090")
    # Runtime config reads ALLOWED_ORIGINS once at process startup. Exercise that
    # real boundary without reloading shared config used by other tests.
    code = """
import asyncio
from starlette.websockets import WebSocket
from tldw_Server_API.app.core.AuthNZ.settings import get_settings
from tldw_Server_API.app.core.AuthNZ.websocket_session_auth import (
    trusted_webui_origins, resolve_single_user_cookie_websocket, cookie_websocket_rejection_code,
)
assert trusted_webui_origins() == {"http://127.0.0.1:18090"}
async def unexpected_io(*args):
    raise AssertionError("Origin refusal should not perform WebSocket I/O")
ws = WebSocket({"type": "websocket", "path": "/api/v1/audio/stream/transcribe", "query_string": b"",
    "headers": [(b"origin", b"http://127.0.0.1:18091"),
        (b"cookie", (get_settings().SINGLE_USER_SESSION_COOKIE_NAME + "=invalid-session").encode())]},
    receive=unexpected_io, send=unexpected_io)
assert asyncio.run(resolve_single_user_cookie_websocket(ws)) is None
assert cookie_websocket_rejection_code(ws) == 4403
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "ALLOWED_ORIGINS": allowed},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.asyncio
@pytest.mark.parametrize("hop,want_status", [(SECRET, 200), ("forged", 403)])
async def test_metadata_api_applies_the_same_managed_guard(hop, want_status):
    app = FastAPI()
    app.include_router(setup_endpoint.router, prefix="/api/v1")
    headers = dict(gateway_request(changes={"x-tldw-gateway-hop": hop}).headers)
    transport = httpx.ASGITransport(app=app, client=("172.18.0.3", 40000))
    async with httpx.AsyncClient(transport=transport, base_url=ORIGIN) as client:
        response = await client.get("/api/v1/setup/first-run/metadata", headers=headers)
    assert response.status_code == want_status
    if want_status == 200:
        assert response.json()["manual_auth_required"] is False
        assert response.json()["connection"]["browser_access"] == "local"
    assert SECRET not in response.text
