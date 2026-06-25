import ipaddress
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

import tldw_Server_API.app.core.Security.setup_access_guard as guard
from tldw_Server_API.app.core.Security.setup_access_guard import SetupAccessGuardMiddleware


pytestmark = pytest.mark.unit


def _make_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(SetupAccessGuardMiddleware)

    @app.get("/setup/ping")
    async def setup_ping():
        return {"ok": True}

    return app


def _set_remote_ip(monkeypatch, ip: str):
    monkeypatch.setattr(SetupAccessGuardMiddleware, "_resolve_client_ip", lambda self, request, proxies: ip)
    monkeypatch.setattr(guard, "_is_loopback", lambda _ip: False)


def test_setup_allowlist_blocks_non_matching_ip(monkeypatch):
    monkeypatch.setenv("TLDW_SETUP_ALLOW_REMOTE", "1")
    monkeypatch.setenv("TLDW_SETUP_ALLOWLIST", "203.0.113.5")
    _set_remote_ip(monkeypatch, "198.51.100.20")

    client = TestClient(_make_app())
    resp = client.get("/setup/ping")
    assert resp.status_code == 403
    assert "allowlist" in resp.text.lower()


def test_setup_allowlist_allows_matching_ip(monkeypatch):
    monkeypatch.setenv("TLDW_SETUP_ALLOW_REMOTE", "1")
    monkeypatch.setenv("TLDW_SETUP_ALLOWLIST", "203.0.113.5")
    _set_remote_ip(monkeypatch, "203.0.113.5")

    client = TestClient(_make_app())
    resp = client.get("/setup/ping")
    assert resp.status_code == 200


def test_setup_blocks_remote_when_toggle_off(monkeypatch):
    monkeypatch.delenv("TLDW_SETUP_ALLOW_REMOTE", raising=False)
    monkeypatch.delenv("TLDW_SETUP_ALLOWLIST", raising=False)
    _set_remote_ip(monkeypatch, "198.51.100.20")

    client = TestClient(_make_app())
    resp = client.get("/setup/ping")
    assert resp.status_code == 403


def test_trusted_proxy_ignores_spoofed_leftmost_x_forwarded_for():
    middleware = SetupAccessGuardMiddleware(_make_app())
    request = SimpleNamespace(
        client=SimpleNamespace(host="10.0.0.10"),
        headers={
            "x-forwarded-for": "127.0.0.1, 198.51.100.20, 10.0.0.10",
        },
    )

    client_ip = middleware._resolve_client_ip(
        request,
        [ipaddress.ip_network("10.0.0.0/24")],
    )

    assert client_ip == "198.51.100.20"


def test_trusted_proxy_skips_malformed_x_forwarded_for_entries():
    middleware = SetupAccessGuardMiddleware(_make_app())
    request = SimpleNamespace(
        client=SimpleNamespace(host="10.0.0.10"),
        headers={
            "x-forwarded-for": "not-an-ip, 198.51.100.20, 10.0.0.10",
        },
    )

    client_ip = middleware._resolve_client_ip(
        request,
        [ipaddress.ip_network("10.0.0.0/24")],
    )

    assert client_ip == "198.51.100.20"
