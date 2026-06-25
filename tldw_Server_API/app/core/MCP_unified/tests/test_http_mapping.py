import base64
import ipaddress
import json
import os

import pytest
from fastapi.testclient import TestClient


def _setup_env():
    # Keep the app light for tests
    os.environ["TEST_MODE"] = "true"
    # Single-user mode with API key for simplicity
    os.environ["AUTH_MODE"] = "single_user"
    os.environ["SINGLE_USER_API_KEY"] = "test-api-key-1234567890"
    os.environ["SINGLE_USER_FIXED_ID"] = "1"
    # Ensure MCP unified config has required secrets
    os.environ["MCP_JWT_SECRET"] = "x" * 64
    os.environ["MCP_API_KEY_SALT"] = "s" * 64


def _build_mcp_client():
    from fastapi import FastAPI

    from tldw_Server_API.app.api.v1.endpoints.mcp_unified_endpoint import router as mcp_router

    app = FastAPI()
    app.include_router(mcp_router, prefix="/api/v1")
    return TestClient(app)


@pytest.fixture(scope="module")
def client():
    _setup_env()
    from fastapi import FastAPI, HTTPException, Request, status

    from tldw_Server_API.app.api.v1.API_Deps import auth_deps
    from tldw_Server_API.app.api.v1.endpoints.mcp_unified_endpoint import router as mcp_router
    from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
    app = FastAPI()
    app.include_router(mcp_router, prefix="/api/v1")

    async def _fake_get_auth_principal(request: Request) -> AuthPrincipal:  # type: ignore[override]
        # Treat any presented credential as an admin-style principal with system.logs
        auth = request.headers.get("Authorization")
        x_api_key = request.headers.get("X-API-KEY")
        if not auth and not x_api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required",
            )
        return AuthPrincipal(
            kind="user",
            user_id=1,
            api_key_id=None,
            subject=None,
            token_type="access",
            jti=None,
            roles=["admin"],
            permissions=["system.logs"],
            is_admin=True,
            org_ids=[],
            team_ids=[],
        )

    app.dependency_overrides[auth_deps.get_auth_principal] = _fake_get_auth_principal
    with TestClient(app) as c:
        yield c


def test_tools_list_unauth_forbidden_with_hint(client: TestClient):
    r = client.get("/api/v1/mcp/tools")
    assert r.status_code == 403
    body = r.json()
    # Endpoint maps -32001 to 403 with a hint under 'detail'
    assert isinstance(body, dict) and isinstance(body.get("detail"), dict)
    assert "hint" in body["detail"]


def test_tools_list_via_request_bearer_token_allowed(client: TestClient):
    # Use MCP JWT bearer for auth to avoid AuthNZ DB/RBAC complexity
    from tldw_Server_API.app.core.MCP_unified.auth.jwt_manager import get_jwt_manager
    os.environ.setdefault("MCP_JWT_SECRET", "x" * 64)
    token = get_jwt_manager().create_access_token(subject="1")
    payload = {"jsonrpc": "2.0", "method": "tools/list", "params": {}, "id": 1}
    r = client.post("/api/v1/mcp/request", json=payload, headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body, dict) and isinstance(body.get("result"), dict)
    assert "tools" in body["result"]
    assert "error" not in body


def test_initialize_request_sets_session_header(client: TestClient):
    payload = {"jsonrpc": "2.0", "method": "initialize", "params": {"clientInfo": {"name": "pytest"}}, "id": 1}
    # Provide a small safe_config via base64
    cfg = base64.b64encode(json.dumps({"snippet_length": 200}).encode("utf-8")).decode("utf-8")
    r = client.post(
        "/api/v1/mcp/request",
        json=payload,
        params={"config": cfg},
    )
    assert r.status_code == 200
    assert r.headers.get("mcp-session-id") is not None
    body = r.json()
    assert isinstance(body, dict) and body.get("result") is not None
    assert "error" not in body
    result = body["result"]
    assert result.get("serverInfo", {}).get("name") == "tldw-mcp-unified"


def test_http_notification_returns_204(client: TestClient):
    payload = {"jsonrpc": "2.0", "method": "ping"}
    r = client.post("/api/v1/mcp/request", json=payload)
    assert r.status_code == 204
    assert r.content in (b"",)


def test_resources_and_modules_mappings_unauth_forbidden(client: TestClient):
    # Unauthenticated should be forbidden for resources/modules
    r0 = client.get("/api/v1/mcp/resources")
    assert r0.status_code == 403

    r1 = client.get("/api/v1/mcp/modules")
    assert r1.status_code == 403


def test_tools_list_get_bearer_ok(client: TestClient):
    from tldw_Server_API.app.core.MCP_unified.auth.jwt_manager import get_jwt_manager
    os.environ.setdefault("MCP_JWT_SECRET", "x" * 64)
    token = get_jwt_manager().create_access_token(subject="1")
    r = client.get("/api/v1/mcp/tools", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, dict) and "tools" in data


def test_modules_health_admin_bearer_ok(client: TestClient):
    # Use MCP JWT with admin role so it passes regardless of auth mode
    from tldw_Server_API.app.core.MCP_unified.auth.jwt_manager import get_jwt_manager
    os.environ.setdefault("MCP_JWT_SECRET", "x" * 64)
    token = get_jwt_manager().create_access_token(subject="1", roles=["admin"])  # embed admin role
    r = client.get("/api/v1/mcp/modules/health", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, dict) and "health" in data


def test_prometheus_requires_auth(client: TestClient, monkeypatch):
    # No auth → 401/403 via require_permissions(SYSTEM_LOGS)
    r0 = client.get("/api/v1/mcp/metrics/prometheus")
    assert r0.status_code in (401, 403)
    # With X-API-KEY in single_user mode → 200
    r1 = client.get(
        "/api/v1/mcp/metrics/prometheus",
        headers={"X-API-KEY": os.environ["SINGLE_USER_API_KEY"]},
    )
    assert r1.status_code == 200
    assert isinstance(r1.text, str) and len(r1.text) > 0


def test_demo_auth_requires_secret(monkeypatch):
    _setup_env()
    monkeypatch.setenv("MCP_ENABLE_DEMO_AUTH", "1")
    monkeypatch.delenv("MCP_DEMO_AUTH_SECRET", raising=False)
    from tldw_Server_API.app.core.MCP_unified import config as config_module
    try:
        config_module.get_config.cache_clear()  # type: ignore[attr-defined]
    except Exception:
        _ = None
    with _build_mcp_client() as temp_client:
        resp = temp_client.post(
            "/api/v1/mcp/auth/token",
            json={"username": "admin", "password": "anything"},
        )
        assert resp.status_code == 501


def test_demo_auth_issues_token_with_secret(monkeypatch):
    _setup_env()
    monkeypatch.setenv("MCP_ENABLE_DEMO_AUTH", "1")
    secret = "supersecretvalue12345"
    monkeypatch.setenv("MCP_DEMO_AUTH_SECRET", secret)
    from tldw_Server_API.app.core.MCP_unified import config as config_module
    from tldw_Server_API.app.core.MCP_unified.security import ip_filter
    try:
        config_module.get_config.cache_clear()  # type: ignore[attr-defined]
        ip_filter.get_ip_access_controller.cache_clear()  # type: ignore[attr-defined]
    except Exception:
        _ = None

    # Treat the starlette test client host as loopback/private.
    real_ip_address = ipaddress.ip_address

    class _LoopbackPeer:
        def __init__(self):
            self._impl = real_ip_address("127.0.0.1")

        def __str__(self):
            return str(self._impl)

        @property
        def is_loopback(self):
            return True

        @property
        def is_private(self):
            return True

    def _patched_ip_address(value):
        if value == "testclient":
            return _LoopbackPeer()
        return real_ip_address(value)

    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.mcp_unified_endpoint.ipaddress.ip_address",
        _patched_ip_address,
    )

    with _build_mcp_client() as temp_client:
        resp = temp_client.post(
            "/api/v1/mcp/auth/token",
            json={"username": "admin", "password": secret},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "access_token" in body and body["token_type"] == "bearer"


def test_refresh_endpoint_rejects_query_token_and_accepts_body(monkeypatch):
    _setup_env()
    monkeypatch.setenv("MCP_ENABLE_DEMO_AUTH", "1")
    monkeypatch.setenv("MCP_DEMO_AUTH_SECRET", "supersecretvalue12345")
    from tldw_Server_API.app.core.MCP_unified import config as config_module
    from tldw_Server_API.app.core.MCP_unified.auth.jwt_manager import get_jwt_manager
    from tldw_Server_API.app.core.MCP_unified.security import ip_filter

    try:
        config_module.get_config.cache_clear()  # type: ignore[attr-defined]
        ip_filter.get_ip_access_controller.cache_clear()  # type: ignore[attr-defined]
    except Exception:
        _ = None

    mgr = get_jwt_manager()
    refresh, token_id = mgr.create_refresh_token(subject="1")

    with _build_mcp_client() as temp_client:
        # Query transport is intentionally unsupported for refresh tokens.
        r_query = temp_client.post(
            "/api/v1/mcp/auth/refresh",
            params={"refresh_token": "qtoken", "token_id": "qid"},
        )
        assert r_query.status_code == 422

        # Body transport is supported.
        r_body = temp_client.post(
            "/api/v1/mcp/auth/refresh",
            json={"refresh_token": refresh, "token_id": token_id},
        )
        assert r_body.status_code == 200
        data = r_body.json()
        assert "access_token" in data and data["token_type"] == "bearer"


def test_refresh_endpoint_disabled_without_demo_auth(monkeypatch):
    _setup_env()
    monkeypatch.delenv("MCP_ENABLE_DEMO_AUTH", raising=False)
    from tldw_Server_API.app.core.MCP_unified import config as config_module
    from tldw_Server_API.app.core.MCP_unified.auth.jwt_manager import get_jwt_manager
    from tldw_Server_API.app.core.MCP_unified.security import ip_filter

    try:
        config_module.get_config.cache_clear()  # type: ignore[attr-defined]
        ip_filter.get_ip_access_controller.cache_clear()  # type: ignore[attr-defined]
    except Exception:
        _ = None

    refresh, token_id = get_jwt_manager().create_refresh_token(subject="1")

    with _build_mcp_client() as temp_client:
        r_body = temp_client.post(
            "/api/v1/mcp/auth/refresh",
            json={"refresh_token": refresh, "token_id": token_id},
        )

    assert r_body.status_code == 501


def test_request_guard_enforces_body_size_limit(monkeypatch):
    _setup_env()
    from tldw_Server_API.app.core.MCP_unified import config as config_module
    from tldw_Server_API.app.core.MCP_unified.security import ip_filter

    config_module.get_config.cache_clear()  # type: ignore[attr-defined]
    base_cfg = config_module.get_config()
    override_cfg = base_cfg.model_copy(update={"http_max_body_bytes": 128})

    def _override_config():
        return override_cfg

    monkeypatch.setattr(
        "tldw_Server_API.app.core.MCP_unified.config.get_config",
        _override_config,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.MCP_unified.security.request_guards.get_config",
        _override_config,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.MCP_unified.security.ip_filter.get_config",
        _override_config,
    )
    try:
        ip_filter.get_ip_access_controller.cache_clear()  # type: ignore[attr-defined]
    except Exception:
        _ = None

    # Treat test client peer as trusted proxy for header acceptance
    monkeypatch.setattr(
        ip_filter.IPAccessController,
        "_is_trusted_proxy",
        lambda self, ip: ip in {"testclient", "127.0.0.1"},
    )

    with _build_mcp_client() as temp_client:
        large_payload = {
            "jsonrpc": "2.0",
            "method": "ping",
            "params": {"blob": "a" * 1024},
            "id": 1,
        }
        resp = temp_client.post("/api/v1/mcp/request", json=large_payload)
        assert resp.status_code == 413


def test_request_guard_requires_client_certificate(monkeypatch):
    _setup_env()
    from tldw_Server_API.app.core.MCP_unified import config as config_module
    from tldw_Server_API.app.core.MCP_unified.security import ip_filter

    config_module.get_config.cache_clear()  # type: ignore[attr-defined]
    base_cfg = config_module.get_config()
    override_cfg = base_cfg.model_copy(
        update={
            "client_cert_required": True,
            "client_cert_header": "x-ssl-client-verify",
            # Stricter policy: require explicit expected value
            "client_cert_header_value": "success",
        }
    )

    def _override_config():
        return override_cfg

    monkeypatch.setattr(
        "tldw_Server_API.app.core.MCP_unified.config.get_config",
        _override_config,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.MCP_unified.security.request_guards.get_config",
        _override_config,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.MCP_unified.security.ip_filter.get_config",
        _override_config,
    )
    try:
        ip_filter.get_ip_access_controller.cache_clear()  # type: ignore[attr-defined]
    except Exception:
        _ = None

    payload = {"jsonrpc": "2.0", "method": "ping", "id": 1}
    with _build_mcp_client() as temp_client:
        r_missing = temp_client.post("/api/v1/mcp/request", json=payload)
        assert r_missing.status_code == 403

        r_valid = temp_client.post(
            "/api/v1/mcp/request",
            json=payload,
            headers={"x-ssl-client-verify": "SUCCESS"},
        )
        assert r_valid.status_code == 200
