"""Regression tests for HTTP-layer security guards."""

import pytest
from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request

from tldw_Server_API.app.core.MCP_unified.security import ip_filter
from tldw_Server_API.app.core.MCP_unified.security.request_guards import (
    enforce_client_certificate_headers,
    enforce_http_security,
    enforce_request_body_limit,
)


def _build_guarded_app() -> FastAPI:
    app = FastAPI()

    @app.post("/guarded")
    async def guarded_endpoint(
        _guard: None = Depends(enforce_http_security),  # pragma: no cover - dependency handles logic
    ):
        return {"status": "ok"}

    @app.get("/guarded-get")
    async def guarded_get_endpoint(
        _guard: None = Depends(enforce_http_security),  # pragma: no cover - dependency handles logic
    ):
        return {"status": "ok"}

    return app


def test_enforce_http_security_rejects_large_payload(monkeypatch):
    from types import SimpleNamespace

    cfg = SimpleNamespace(
        allowed_client_ips=[],
        blocked_client_ips=[],
        trust_x_forwarded_for=False,
        trusted_proxy_depth=0,
        trusted_proxy_ips=[],
        http_max_body_bytes=32,
        client_cert_required=False,
        client_cert_header=None,
        client_cert_header_value=None,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.MCP_unified.security.request_guards.get_config",
        lambda: cfg,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.MCP_unified.security.ip_filter.get_config",
        lambda: cfg,
    )
    try:
        ip_filter.get_ip_access_controller.cache_clear()  # type: ignore[attr-defined]
    except Exception:
        _ = None
    client = TestClient(_build_guarded_app())
    try:
        # Small payload passes
        r_small = client.post("/guarded", json={"msg": "ok"})
        assert r_small.status_code == 200
        # Oversized payload rejected with 413
        big_value = "a" * 64
        r_big = client.post("/guarded", json={"msg": big_value})
        assert r_big.status_code == 413
    finally:
        client.close()


def test_enforce_http_security_get_skips_body_read(monkeypatch):
    from types import SimpleNamespace

    cfg = SimpleNamespace(
        allowed_client_ips=[],
        blocked_client_ips=[],
        trust_x_forwarded_for=False,
        trusted_proxy_depth=0,
        trusted_proxy_ips=[],
        http_max_body_bytes=32,
        client_cert_required=False,
        client_cert_header=None,
        client_cert_header_value=None,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.MCP_unified.security.request_guards.get_config",
        lambda: cfg,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.MCP_unified.security.ip_filter.get_config",
        lambda: cfg,
    )
    try:
        ip_filter.get_ip_access_controller.cache_clear()  # type: ignore[attr-defined]
    except Exception:
        _ = None

    body_call_count = {"count": 0}

    async def _raise_if_called(_self) -> bytes:
        body_call_count["count"] += 1
        raise AssertionError("request.body() should not be called for GET")

    monkeypatch.setattr(Request, "body", _raise_if_called)

    client = TestClient(_build_guarded_app())
    try:
        response = client.get("/guarded-get")
        assert response.status_code == 200
        assert body_call_count["count"] == 0
    finally:
        client.close()


@pytest.mark.asyncio
async def test_enforce_request_body_limit_handles_client_disconnect(monkeypatch):
    from types import SimpleNamespace

    cfg = SimpleNamespace(
        allowed_client_ips=[],
        blocked_client_ips=[],
        trust_x_forwarded_for=False,
        trusted_proxy_depth=0,
        trusted_proxy_ips=[],
        http_max_body_bytes=1024,
        client_cert_required=False,
        client_cert_header=None,
        client_cert_header_value=None,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.MCP_unified.security.request_guards.get_config",
        lambda: cfg,
    )

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/api/v1/mcp/health",
        "raw_path": b"/api/v1/mcp/health",
        "query_string": b"",
        "headers": [],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
    }

    async def _receive_disconnect():
        return {"type": "http.disconnect"}

    request = Request(scope, _receive_disconnect)

    # Should not propagate starlette.requests.ClientDisconnect
    await enforce_request_body_limit(request)


@pytest.mark.asyncio
async def test_ip_allowlist_normalizes_missing_client_ip_in_test_mode(monkeypatch):
    from types import SimpleNamespace

    cfg = SimpleNamespace(
        allowed_client_ips=["127.0.0.1"],
        blocked_client_ips=[],
        trust_x_forwarded_for=False,
        trusted_proxy_depth=0,
        trusted_proxy_ips=[],
        http_max_body_bytes=524288,
        client_cert_required=False,
        client_cert_header=None,
        client_cert_header_value=None,
    )
    monkeypatch.setenv("TLDW_TEST_MODE", "1")
    monkeypatch.setattr(
        "tldw_Server_API.app.core.MCP_unified.security.ip_filter.get_config",
        lambda: cfg,
    )
    try:
        ip_filter.get_ip_access_controller.cache_clear()  # type: ignore[attr-defined]
    except Exception:
        _ = None

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/api/v1/mcp/health",
        "raw_path": b"/api/v1/mcp/health",
        "query_string": b"",
        "headers": [],
        "client": None,
        "server": ("testserver", 80),
    }

    async def _receive_empty():
        return {"type": "http.request", "body": b""}

    request = Request(scope, _receive_empty)

    await ip_filter.enforce_ip_allowlist(request)


def test_client_certificate_guard_allows_testclient_only_in_test_mode(monkeypatch):
    from types import SimpleNamespace

    cfg = SimpleNamespace(
        allowed_client_ips=[],
        blocked_client_ips=[],
        trust_x_forwarded_for=False,
        trusted_proxy_depth=0,
        trusted_proxy_ips=[],
        http_max_body_bytes=524288,
        client_cert_required=True,
        client_cert_header="x-client-cert",
        client_cert_header_value="verified",
    )
    monkeypatch.delenv("TEST_MODE", raising=False)
    monkeypatch.delenv("TLDW_TEST_MODE", raising=False)
    monkeypatch.setattr(
        "tldw_Server_API.app.core.MCP_unified.security.request_guards.get_config",
        lambda: cfg,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.MCP_unified.security.ip_filter.get_config",
        lambda: cfg,
    )
    try:
        ip_filter.get_ip_access_controller.cache_clear()  # type: ignore[attr-defined]
    except Exception:
        _ = None

    with pytest.raises(HTTPException) as rejected:
        enforce_client_certificate_headers(
            {"x-client-cert": "verified"},
            remote_addr="testclient",
        )
    assert rejected.value.status_code == 403

    monkeypatch.setenv("TLDW_TEST_MODE", "1")

    enforce_client_certificate_headers(
        {"x-client-cert": "verified"},
        remote_addr="testclient",
    )

    with pytest.raises(HTTPException) as bad_cert:
        enforce_client_certificate_headers(
            {"x-client-cert": "denied"},
            remote_addr="testclient",
        )
    assert bad_cert.value.status_code == 403


def test_enforce_http_security_requires_client_certificate(monkeypatch):
    from types import SimpleNamespace

    cfg = SimpleNamespace(
        allowed_client_ips=[],
        blocked_client_ips=[],
        trust_x_forwarded_for=False,
        trusted_proxy_depth=0,
        trusted_proxy_ips=[],
        http_max_body_bytes=524288,
        client_cert_required=True,
        client_cert_header="x-ssl-client-verify",
        # Stricter policy requires explicit expected value
        client_cert_header_value="success",
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.MCP_unified.security.request_guards.get_config",
        lambda: cfg,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.MCP_unified.security.ip_filter.get_config",
        lambda: cfg,
    )
    try:
        ip_filter.get_ip_access_controller.cache_clear()  # type: ignore[attr-defined]
    except Exception:
        _ = None
    monkeypatch.setattr(
        ip_filter.IPAccessController,
        "_is_trusted_proxy",
        lambda self, ip: ip in {"testclient", "127.0.0.1"},
    )

    client = TestClient(_build_guarded_app())
    try:
        # Missing certificate header → 403
        r_missing = client.post("/guarded", json={"msg": "hello"})
        assert r_missing.status_code == 403
        # Valid header (default SUCCESS sentinel) → 200
        r_valid = client.post(
            "/guarded",
            json={"msg": "hello"},
            headers={"x-ssl-client-verify": "SUCCESS"},
        )
        assert r_valid.status_code == 200
        # Raw PEM payloads are rejected under strict policy (value must match expected)
        r_pem = client.post(
            "/guarded",
            json={"msg": "pem"},
            headers={"x-ssl-client-verify": "-----BEGIN CERTIFICATE-----\nMIIB..."},
        )
        assert r_pem.status_code == 403
    finally:
        client.close()


def test_ip_filter_ignores_untrusted_forwarded_header():
    controller = ip_filter.IPAccessController(
        allowed=[],
        blocked=[],
        trust_x_forwarded_for=True,
        trusted_proxy_depth=0,
        trusted_proxies=[],
    )
    # Public client tries to spoof XFF; remote peer is also public → ignore header
    resolved = controller.resolve_client_ip("198.51.100.10", "203.0.113.5", None)
    assert resolved == "198.51.100.10"


def test_ip_filter_ignores_untrusted_real_ip_header():
    controller = ip_filter.IPAccessController(
        allowed=[],
        blocked=[],
        trust_x_forwarded_for=True,
        trusted_proxy_depth=0,
        trusted_proxies=[],
    )
    resolved = controller.resolve_client_ip("198.51.100.10", None, "203.0.113.5")
    assert resolved == "198.51.100.10"


def test_ip_filter_falls_back_to_remote_when_proxy_depth_insufficient():
    controller = ip_filter.IPAccessController(
        allowed=[],
        blocked=[],
        trust_x_forwarded_for=True,
        trusted_proxy_depth=2,
        trusted_proxies=["127.0.0.1/32"],
    )
    resolved = controller.resolve_client_ip("127.0.0.1", "203.0.113.5", None)
    assert resolved == "127.0.0.1"


def test_ip_filter_uses_single_hop_xff_when_depth_one():
    controller = ip_filter.IPAccessController(
        allowed=[],
        blocked=[],
        trust_x_forwarded_for=True,
        trusted_proxy_depth=1,
        trusted_proxies=["127.0.0.1/32"],
    )
    # Single trusted proxy should surface the original client when depth is 1.
    resolved = controller.resolve_client_ip("127.0.0.1", "203.0.113.5", None)
    assert resolved == "203.0.113.5"


def test_ip_filter_does_not_trust_real_ip_without_remote_peer():
    controller = ip_filter.IPAccessController(
        allowed=[],
        blocked=[],
        trust_x_forwarded_for=True,
        trusted_proxy_depth=0,
        trusted_proxies=["127.0.0.1/32"],
    )
    resolved = controller.resolve_client_ip(None, None, "203.0.113.5")
    assert resolved is None


def test_ip_filter_accepts_real_ip_from_trusted_proxy_when_enabled():
    controller = ip_filter.IPAccessController(
        allowed=[],
        blocked=[],
        trust_x_forwarded_for=True,
        trusted_proxy_depth=0,
        trusted_proxies=["127.0.0.1/32"],
    )
    resolved = controller.resolve_client_ip("127.0.0.1", None, "10.0.0.1")
    assert resolved == "10.0.0.1"


def test_enforce_http_security_rejects_invalid_cert_value(monkeypatch):
    from types import SimpleNamespace

    cfg = SimpleNamespace(
        allowed_client_ips=[],
        blocked_client_ips=[],
        trust_x_forwarded_for=False,
        trusted_proxy_depth=0,
        trusted_proxy_ips=[],
        http_max_body_bytes=524288,
        client_cert_required=True,
        client_cert_header="x-client-cert",
        client_cert_header_value="verified",
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.MCP_unified.security.request_guards.get_config",
        lambda: cfg,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.MCP_unified.security.ip_filter.get_config",
        lambda: cfg,
    )
    try:
        ip_filter.get_ip_access_controller.cache_clear()  # type: ignore[attr-defined]
    except Exception:
        _ = None

    # Treat test client as coming via trusted proxy for header acceptance
    monkeypatch.setattr(
        ip_filter.IPAccessController,
        "_is_trusted_proxy",
        lambda self, ip: ip in {"testclient", "127.0.0.1"},
    )
    client = TestClient(_build_guarded_app())
    try:
        r_missing = client.post("/guarded", json={"msg": "bad"})
        assert r_missing.status_code == 403
        r_wrong = client.post(
            "/guarded",
            json={"msg": "bad"},
            headers={"x-client-cert": "denied"},
        )
        assert r_wrong.status_code == 403
        r_ok = client.post(
            "/guarded",
            json={"msg": "good"},
            headers={"x-client-cert": "verified"},
        )
        assert r_ok.status_code == 200
    finally:
        client.close()


def test_enforce_http_security_enforces_ip_allowlist(monkeypatch):
    from types import SimpleNamespace

    cfg = SimpleNamespace(
        allowed_client_ips=["10.0.0.1"],
        blocked_client_ips=[],
        trust_x_forwarded_for=True,
        trusted_proxy_depth=0,
        trusted_proxy_ips=["127.0.0.1/32"],
        http_max_body_bytes=524288,
        client_cert_required=False,
        client_cert_header=None,
        client_cert_header_value=None,
    )
    monkeypatch.setattr(
        ip_filter.IPAccessController,
        "_is_trusted_proxy",
        lambda self, ip: ip in {"testclient", "127.0.0.1"},
    )

    controller = ip_filter.IPAccessController(
        allowed=cfg.allowed_client_ips,
        blocked=cfg.blocked_client_ips,
        trust_x_forwarded_for=cfg.trust_x_forwarded_for,
        trusted_proxy_depth=cfg.trusted_proxy_depth,
        trusted_proxies=cfg.trusted_proxy_ips,
    )
    resolved = controller.resolve_client_ip("testclient", "10.0.0.1", None)
    assert resolved == "10.0.0.1"
    assert controller.is_allowed(resolved) is True
