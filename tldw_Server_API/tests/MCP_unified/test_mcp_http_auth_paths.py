import builtins
import os
from typing import Any, Dict, List, Optional

from fastapi import Response
from fastapi.testclient import TestClient
from loguru import logger
import pytest
from starlette.requests import Request

from tldw_Server_API.app.main import app
from tldw_Server_API.app.api.v1.endpoints import mcp_unified_endpoint as mcp_ep
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
from tldw_Server_API.app.core.MCP_unified.auth import UserRole


# Disable HTTP security guard for these tests (IP allowlist/mTLS) to focus on auth behavior.
try:
    from tldw_Server_API.app.core.MCP_unified.security.request_guards import (  # type: ignore[attr-defined]
        enforce_http_security as _ehs,
    )

    app.dependency_overrides[_ehs] = lambda: None
except Exception as exc:  # pragma: no cover - defensive
    logger.debug(f"Unable to override enforce_http_security for MCP tests: {exc}")


client = TestClient(app)


class _DummyProtocol:
    async def process_request(self, payload, ctx):
        from tldw_Server_API.app.core.MCP_unified.protocol import MCPResponse

        if isinstance(payload, list):
            return [MCPResponse(result={"ok": True}, id=getattr(p, "id", None)) for p in payload]
        return MCPResponse(result={"ok": True}, id=getattr(payload, "id", None))


class _DummyServer:
    def __init__(self):
        self.initialized = True
        self.protocol = _DummyProtocol()
        self.last_metadata: Optional[Dict[str, Any]] = None

    async def initialize(self):
        self.initialized = True

    async def handle_http_request(
        self,
        request,
        client_id: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        from tldw_Server_API.app.core.MCP_unified.protocol import MCPResponse

        self.last_metadata = metadata or {}
        return MCPResponse(result={"ok": True}, id=getattr(request, "id", None))

    async def handle_http_batch(
        self,
        requests,
        client_id: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        from tldw_Server_API.app.core.MCP_unified.protocol import MCPResponse

        self.last_metadata = metadata or {}
        return [MCPResponse(result={"ok": True}, id=getattr(req, "id", None)) for req in requests]


class _LoggerStub:
    def __init__(self) -> None:
        self.debug_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def debug(self, *args: Any, **kwargs: Any) -> None:
        self.debug_calls.append((args, kwargs))


def _install_dummy_server(monkeypatch) -> _DummyServer:


    server = _DummyServer()
    monkeypatch.setattr(mcp_ep, "get_mcp_server", lambda: server)
    return server


def test_mcp_get_client_ip_sanitizes_resolution_failure_log(monkeypatch):
    logger_stub = _LoggerStub()
    monkeypatch.setattr(mcp_ep, "logger", logger_stub)

    def _raise_resolution_failure(*_args: Any, **_kwargs: Any) -> str:
        raise RuntimeError("client IP parser leaked /private/mcp-client-ip.txt")

    monkeypatch.setattr(mcp_ep, "resolve_client_ip", _raise_resolution_failure)

    result = mcp_ep._get_client_ip(Request({"type": "http", "headers": []}))

    assert result is None
    assert logger_stub.debug_calls == [(("Failed to extract client IP",), {})]
    assert "/private/mcp-client-ip.txt" not in repr(logger_stub.debug_calls)


@pytest.mark.asyncio
async def test_mcp_initialize_session_id_logs_are_sanitized(monkeypatch):
    logger_stub = _LoggerStub()
    monkeypatch.setattr(mcp_ep, "logger", logger_stub)
    _install_dummy_server(monkeypatch)

    original_import = builtins.__import__

    def _raise_uuid_import(name: str, *args: Any, **kwargs: Any):
        if name == "uuid":
            raise RuntimeError("uuid generator leaked /private/mcp-session.txt")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _raise_uuid_import)
    auth = mcp_ep.McpAuthContext(user=None, principal=None, api_key_info=None, raw_api_key=None)
    http_request = Request({"type": "http", "headers": []})

    await mcp_ep.mcp_request(
        mcp_ep.MCPRequest(method="initialize", id=1),
        http_request,
        client_id=None,
        auth=auth,
        mcp_session_id=None,
        config=None,
        response=Response(),
        _guard=None,
    )
    await mcp_ep.mcp_request_batch(
        [mcp_ep.MCPRequest(method="initialize", id=2)],
        http_request,
        client_id=None,
        auth=auth,
        mcp_session_id=None,
        config=None,
        response=Response(),
        _guard=None,
    )

    assert [args[0] for args, _kwargs in logger_stub.debug_calls if args] == [
        "Failed to generate session ID for initialize request",
        "Failed to generate session ID for batch initialize",
    ]
    assert all(not kwargs.get("exc_info") for _args, kwargs in logger_stub.debug_calls)
    assert "/private/mcp-session.txt" not in repr(logger_stub.debug_calls)


class _RBACAllow:
    async def check_permission(self, *_args, **_kwargs):
        return True


class _ScopeServer:
    def __init__(self):
        from tldw_Server_API.app.core.MCP_unified.protocol import MCPProtocol

        self.initialized = True
        self.protocol = MCPProtocol()
        self.protocol.rbac_policy = _RBACAllow()

    async def initialize(self):
        self.initialized = True

    async def handle_http_request(
        self,
        request,
        client_id: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        from tldw_Server_API.app.core.MCP_unified.protocol import RequestContext

        ctx = RequestContext(
            request_id=str(getattr(request, "id", "http")),
            user_id=user_id,
            client_id=client_id,
            session_id=None,
            metadata=metadata or {},
        )
        return await self.protocol.process_request(request, ctx)


@pytest.mark.asyncio
async def test_mcp_http_requests_use_single_api_key_validation(monkeypatch):
    """
    Ensure /mcp/request and /mcp/request/batch validate a given API key exactly once
    per request and reuse the resolved metadata (org_id/team_id) without re-validating.
    """

    # Force multi-user API key path so compat token resolution uses APIKeyManager instead of
    # the single-user shortcut.
    monkeypatch.setattr(mcp_ep, "is_single_user_profile_mode", lambda: False)

    calls: List[Dict[str, Any]] = []

    class _DummyApiManager:
        async def validate_api_key(self, key: str, ip_address: Optional[str] = None) -> Dict[str, Any]:
            calls.append({"key": key, "ip": ip_address})
            return {"user_id": "123", "org_id": 9, "team_id": 7}

    async def _fake_get_api_key_manager():
        return _DummyApiManager()

    monkeypatch.setattr(mcp_ep, "get_api_key_manager", _fake_get_api_key_manager)

    server = _install_dummy_server(monkeypatch)

    headers = {"X-API-KEY": "test-api-key-123", "X-Real-IP": "127.0.0.1"}

    # Single MCP HTTP request
    body = {"jsonrpc": "2.0", "method": "status", "id": 1}
    r1 = client.post("/api/v1/mcp/request", headers=headers, json=body)
    assert r1.status_code == 200, r1.text
    data1 = r1.json()
    assert isinstance(data1, dict)
    assert data1.get("result", {}).get("ok") is True

    # API key should have been validated exactly once
    assert len(calls) == 1
    assert calls[0]["key"] == "test-api-key-123"

    # org/team metadata from the key should be propagated to the MCP server
    assert server.last_metadata is not None
    assert server.last_metadata.get("org_id") == 9
    assert server.last_metadata.get("team_id") == 7
    assert "roles" in server.last_metadata  # derived from TokenData roles=["api_client"]

    # Batch MCP HTTP request
    batch_body = [
        {"jsonrpc": "2.0", "method": "status", "id": 2},
        {"jsonrpc": "2.0", "method": "status", "id": 3},
    ]
    r2 = client.post("/api/v1/mcp/request/batch", headers=headers, json=batch_body)
    assert r2.status_code == 200, r2.text
    data2 = r2.json()
    assert isinstance(data2, list)
    assert all(isinstance(item, dict) for item in data2)

    # API key should have been validated exactly once for the batch request as well
    assert len(calls) == 2
    assert calls[1]["key"] == "test-api-key-123"

    # org/team metadata should also be present for the batch path
    assert server.last_metadata is not None
    assert server.last_metadata.get("org_id") == 9
    assert server.last_metadata.get("team_id") == 7


def test_http_api_key_scopes_enforced(monkeypatch):
    """
    Ensure API key scopes propagate to HTTP metadata and enforce mcp: scope checks.
    """
    # Force multi-user API key path (no single-user shim).
    monkeypatch.setenv("MCP_SINGLE_USER_COMPAT_SHIM", "0")
    monkeypatch.setattr(mcp_ep, "is_single_user_profile_mode", lambda: False)

    class _DummyApiManager:
        async def validate_api_key(self, key: str, ip_address: Optional[str] = None) -> Dict[str, Any]:
            scope = ["mcp:tool:media.search"] if key == "scope-match" else ["mcp:tool:other"]
            return {"user_id": "123", "org_id": 9, "team_id": 7, "scope": scope}

    async def _fake_get_api_key_manager():
        return _DummyApiManager()

    monkeypatch.setattr(mcp_ep, "get_api_key_manager", _fake_get_api_key_manager)

    server = _ScopeServer()
    monkeypatch.setattr(mcp_ep, "get_mcp_server", lambda: server)

    payload = {"jsonrpc": "2.0", "method": "tools/call", "params": {"name": "media.search"}, "id": 1}

    # Scope mismatch -> authorization error (403)
    r0 = client.post("/api/v1/mcp/request", headers={"X-API-KEY": "scope-mismatch"}, json=payload)
    assert r0.status_code == 403

    # Scope match -> passes auth (handler will fail later with tool-not-found)
    r1 = client.post("/api/v1/mcp/request", headers={"X-API-KEY": "scope-match"}, json=payload)
    assert r1.status_code == 200
    body = r1.json()
    assert body.get("error", {}).get("code") == -32602


def test_tools_list_attaches_api_key_metadata(monkeypatch):
    """
    Ensure /mcp/tools attaches API key metadata (org/team/scopes) for convenience endpoints.
    """
    monkeypatch.setenv("MCP_SINGLE_USER_COMPAT_SHIM", "0")
    monkeypatch.setattr(mcp_ep, "is_single_user_profile_mode", lambda: False)

    class _DummyApiManager:
        async def validate_api_key(self, key: str, ip_address: Optional[str] = None) -> Dict[str, Any]:
            return {"user_id": "123", "org_id": 9, "team_id": 7, "scope": ["read"]}

    async def _fake_get_api_key_manager():
        return _DummyApiManager()

    monkeypatch.setattr(mcp_ep, "get_api_key_manager", _fake_get_api_key_manager)

    server = _install_dummy_server(monkeypatch)

    r = client.get("/api/v1/mcp/tools", headers={"X-API-KEY": "test-api-key"})
    assert r.status_code == 200, r.text

    assert server.last_metadata is not None
    assert server.last_metadata.get("org_id") == 9
    assert server.last_metadata.get("team_id") == 7
    assert server.last_metadata.get("api_key_scopes") == ["read"]


def test_mcp_request_attaches_workspace_headers_to_metadata(monkeypatch):
    monkeypatch.setattr(mcp_ep, "is_single_user_profile_mode", lambda: True)
    server = _install_dummy_server(monkeypatch)

    headers = {
        "X-API-KEY": os.getenv("SINGLE_USER_API_KEY", "THIS-IS-A-SECURE-KEY-123-FAKE-KEY"),
        "x-tldw-workspace-id": "workspace-direct",
        "x-tldw-cwd": "src/app",
    }
    body = {"jsonrpc": "2.0", "method": "status", "id": 1}

    response = client.post("/api/v1/mcp/request", headers=headers, json=body)

    assert response.status_code == 200, response.text
    assert server.last_metadata is not None
    assert server.last_metadata.get("workspace_id") == "workspace-direct"
    assert server.last_metadata.get("cwd") == "src/app"


def test_mcp_request_batch_attaches_workspace_headers_to_metadata(monkeypatch):
    monkeypatch.setattr(mcp_ep, "is_single_user_profile_mode", lambda: True)
    server = _install_dummy_server(monkeypatch)

    headers = {
        "X-API-KEY": os.getenv("SINGLE_USER_API_KEY", "THIS-IS-A-SECURE-KEY-123-FAKE-KEY"),
        "x-tldw-workspace-id": "workspace-direct",
        "x-tldw-cwd": "src/app",
    }
    body = [
        {"jsonrpc": "2.0", "method": "status", "id": 1},
        {"jsonrpc": "2.0", "method": "status", "id": 2},
    ]

    response = client.post("/api/v1/mcp/request/batch", headers=headers, json=body)

    assert response.status_code == 200, response.text
    assert server.last_metadata is not None
    assert server.last_metadata.get("workspace_id") == "workspace-direct"
    assert server.last_metadata.get("cwd") == "src/app"


def test_tools_execute_attaches_workspace_headers_to_metadata(monkeypatch):
    monkeypatch.setattr(mcp_ep, "is_single_user_profile_mode", lambda: True)
    server = _install_dummy_server(monkeypatch)

    headers = {
        "X-API-KEY": os.getenv("SINGLE_USER_API_KEY", "THIS-IS-A-SECURE-KEY-123-FAKE-KEY"),
        "x-tldw-workspace-id": "workspace-direct",
        "x-tldw-cwd": "src/app",
    }
    payload = {"tool_name": "media.search", "arguments": {"query": "hello"}}

    response = client.post("/api/v1/mcp/tools/execute", headers=headers, json=payload)

    assert response.status_code in {200, 400}, response.text
    assert server.last_metadata is not None
    assert server.last_metadata.get("workspace_id") == "workspace-direct"
    assert server.last_metadata.get("cwd") == "src/app"


def test_tools_execute_api_key_scopes_enforced(monkeypatch):
    """
    Ensure /mcp/tools/execute respects API key scopes.
    """
    monkeypatch.setenv("MCP_SINGLE_USER_COMPAT_SHIM", "0")
    monkeypatch.setattr(mcp_ep, "is_single_user_profile_mode", lambda: False)

    class _DummyApiManager:
        async def validate_api_key(self, key: str, ip_address: Optional[str] = None) -> Dict[str, Any]:
            scope = ["mcp:tool:media.search"] if key == "scope-match" else ["mcp:tool:other"]
            return {"user_id": "123", "org_id": 9, "team_id": 7, "scope": scope}

    async def _fake_get_api_key_manager():
        return _DummyApiManager()

    monkeypatch.setattr(mcp_ep, "get_api_key_manager", _fake_get_api_key_manager)

    server = _ScopeServer()
    monkeypatch.setattr(mcp_ep, "get_mcp_server", lambda: server)

    payload = {"tool_name": "media.search", "arguments": {"query": "hello"}}

    # Scope mismatch -> authorization error (403)
    r0 = client.post("/api/v1/mcp/tools/execute", headers={"X-API-KEY": "scope-mismatch"}, json=payload)
    assert r0.status_code == 403

    # Scope match -> passes auth (handler will fail later with tool-not-found)
    r1 = client.post("/api/v1/mcp/tools/execute", headers={"X-API-KEY": "scope-match"}, json=payload)
    assert r1.status_code == 400


@pytest.mark.asyncio
async def test_modules_health_uses_principal_metadata(monkeypatch):
    """
    Ensure /mcp/modules/health uses the principal returned by require_permissions(SYSTEM_LOGS)
    and forwards its roles/permissions into the MCP metadata.
    """
    from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
    from tldw_Server_API.app.core.AuthNZ.permissions import SYSTEM_LOGS
    from tldw_Server_API.app.api.v1.API_Deps import auth_deps

    server = _install_dummy_server(monkeypatch)

    calls: list[AuthPrincipal] = []

    principal = AuthPrincipal(
        kind="user",
        user_id=42,
        roles=["observer", "admin"],
        permissions=[SYSTEM_LOGS, "other.permission"],
        is_admin=False,
    )

    async def _fake_resolve_auth_principal(request) -> AuthPrincipal:  # type: ignore[override]
        calls.append(principal)
        return principal

    # Patch the core resolver used by get_auth_principal so that require_permissions
    # sees our synthetic principal.
    monkeypatch.setattr(auth_deps, "_resolve_auth_principal", _fake_resolve_auth_principal)

    r = client.get("/api/v1/mcp/modules/health")
    assert r.status_code == 200, r.text

    # Principal should have been resolved exactly once for this request.
    assert len(calls) == 1

    assert server.last_metadata is not None
    # Endpoint should pass through roles/permissions from the principal
    assert server.last_metadata.get("roles") == principal.roles
    assert server.last_metadata.get("permissions") == principal.permissions
    # Admin override flag should be set as documented
    assert server.last_metadata.get("admin_override") is True


@pytest.mark.asyncio
async def test_optional_token_data_prefers_claim_first_principal(monkeypatch):
    from starlette.requests import Request

    principal = AuthPrincipal(
        kind="user",
        user_id=42,
        username="claim-user",
        roles=["admin"],
        permissions=["mcp:read"],
        token_type="access",
    )

    async def _fail_compat_resolver(**_kwargs):
        raise AssertionError("claim-first token data path should not call compat token resolver")

    monkeypatch.setattr(mcp_ep, "_resolve_token_data_compat", _fail_compat_resolver)

    request = Request({"type": "http", "headers": []})
    token = await mcp_ep._get_optional_token_data(
        request=request,
        principal=principal,
        credentials=None,
        x_api_key=None,
    )

    assert isinstance(token, mcp_ep.TokenData)
    assert token.sub == "42"
    assert token.username == "claim-user"
    assert token.roles == ["admin"]
    assert token.permissions == ["mcp:read"]


@pytest.mark.asyncio
async def test_optional_token_data_falls_back_to_legacy_user(monkeypatch):
    from starlette.requests import Request

    expected = mcp_ep.TokenData(
        sub="legacy-123",
        username="legacy",
        roles=["api_client"],
        permissions=["read"],
        token_type="access",
    )

    async def _fake_compat_resolver(**_kwargs):
        return expected

    monkeypatch.setattr(mcp_ep, "_resolve_token_data_compat", _fake_compat_resolver)

    request = Request({"type": "http", "headers": []})
    token = await mcp_ep._get_optional_token_data(
        request=request,
        principal=None,
        credentials=None,
        x_api_key=None,
    )

    assert token is expected


@pytest.mark.asyncio
async def test_mcp_single_user_api_key_flag_enabled_uses_compat_shim(monkeypatch):
    """
    When MCP_SINGLE_USER_COMPAT_SHIM is enabled and the runtime is single-user,
    X-API-KEY matching SINGLE_USER_API_KEY should be treated as a single-user
    admin principal without hitting the multi-user API key manager path.
    """
    # Enable shim and force single-user runtime.
    monkeypatch.setenv("MCP_SINGLE_USER_COMPAT_SHIM", "1")
    monkeypatch.setattr(mcp_ep, "is_single_user_profile_mode", lambda: True)

    # Provide minimal settings for the shim.
    class _Settings:
        SINGLE_USER_API_KEY = "single-user-admin-key"
        SINGLE_USER_FIXED_ID = 999

    monkeypatch.setattr(mcp_ep, "get_settings", lambda: _Settings())

    server = _install_dummy_server(monkeypatch)

    # Also ensure HTTP security guard is disabled for this test.
    try:
        from tldw_Server_API.app.core.MCP_unified.security.request_guards import enforce_http_security as _ehs

        app.dependency_overrides[_ehs] = lambda: None
    except Exception:
        _ = None

    headers = {"X-API-KEY": "single-user-admin-key"}
    body = {"jsonrpc": "2.0", "method": "status", "id": 10}

    r = client.post("/api/v1/mcp/request", headers=headers, json=body)
    assert r.status_code == 200, r.text

    assert server.last_metadata is not None
    # TokenData produced by the shim should carry admin-style role.
    assert "roles" in server.last_metadata
    assert server.last_metadata["roles"] == [UserRole.ADMIN.value]


@pytest.mark.asyncio
async def test_mcp_single_user_api_key_respects_ip_allowlist(monkeypatch):
    """
    Single-user compat auth should reject API keys when the client IP is not
    in SINGLE_USER_ALLOWED_IPS, and accept them when it is.
    """
    from starlette.requests import Request

    monkeypatch.setenv("MCP_SINGLE_USER_COMPAT_SHIM", "1")
    monkeypatch.setattr(mcp_ep, "is_single_user_profile_mode", lambda: True)

    class _Settings:
        SINGLE_USER_API_KEY = "single-user-admin-key"
        SINGLE_USER_FIXED_ID = 999
        SINGLE_USER_ALLOWED_IPS = ["203.0.113.10"]

    monkeypatch.setattr(mcp_ep, "get_settings", lambda: _Settings())

    request_denied = Request({"type": "http", "client": ("198.51.100.5", 12345)})
    user_denied = await mcp_ep._resolve_token_data_compat(
        credentials=None,
        x_api_key="single-user-admin-key",
        request=request_denied,
    )
    assert user_denied is None

    request_allowed = Request({"type": "http", "client": ("203.0.113.10", 12345)})
    user_allowed = await mcp_ep._resolve_token_data_compat(
        credentials=None,
        x_api_key="single-user-admin-key",
        request=request_allowed,
    )
    assert isinstance(user_allowed, mcp_ep.TokenData)
    assert user_allowed.sub == str(_Settings.SINGLE_USER_FIXED_ID)


@pytest.mark.asyncio
async def test_mcp_single_user_api_key_flag_disabled_uses_api_key_manager(monkeypatch):
    """
    When MCP_SINGLE_USER_COMPAT_SHIM is disabled, even in single-user runtime
    the API key should be validated via the multi-user API key manager path.
    """
    monkeypatch.setenv("MCP_SINGLE_USER_COMPAT_SHIM", "0")
    monkeypatch.setattr(mcp_ep, "is_single_user_profile_mode", lambda: True)

    calls: list[dict[str, Any]] = []

    class _DummyApiManager:
        async def validate_api_key(self, key: str, ip_address: Optional[str] = None) -> Dict[str, Any]:
            calls.append({"key": key, "ip": ip_address})
            return {"user_id": "123", "org_id": 9, "team_id": 7}

    async def _fake_get_api_key_manager():
        return _DummyApiManager()

    monkeypatch.setattr(mcp_ep, "get_api_key_manager", _fake_get_api_key_manager)

    server = _install_dummy_server(monkeypatch)

    try:
        from tldw_Server_API.app.core.MCP_unified.security.request_guards import enforce_http_security as _ehs

        app.dependency_overrides[_ehs] = lambda: None
    except Exception:
        _ = None

    headers = {"X-API-KEY": "test-api-key-123"}
    body = {"jsonrpc": "2.0", "method": "status", "id": 11}

    r = client.post("/api/v1/mcp/request", headers=headers, json=body)
    assert r.status_code == 200, r.text

    # API key manager should have been used exactly once.
    assert len(calls) == 1
    assert calls[0]["key"] == "test-api-key-123"

    assert server.last_metadata is not None
    # TokenData from manager path should carry api_client role.
    assert "roles" in server.last_metadata
    assert server.last_metadata["roles"] == ["api_client"]


@pytest.mark.asyncio
async def test_get_current_user_authnz_jwt_failure_falls_back_to_mcp_jwt(monkeypatch):
    """
    If AuthNZ JWT decode fails but an MCP JWT is valid, compat token resolution should
    return the MCP TokenData instead of raising or propagating a 500-style error.
    """
    from fastapi.security.http import HTTPAuthorizationCredentials
    from fastapi import HTTPException, status

    async def _fail_verify_jwt(_request, _token: str):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
        )

    # Simulate invalid AuthNZ JWT: verify_jwt_and_fetch_user raises.
    monkeypatch.setattr(mcp_ep, "verify_jwt_and_fetch_user", _fail_verify_jwt)

    # Provide a successful MCP JWT verification path.
    expected = mcp_ep.TokenData(
        sub="mcp-user-123",
        username="mcp-user",
        roles=["mcp_role"],
        permissions=["tools.execute:sample"],
        token_type="access",
    )

    class _DummyJwtManager:
        def verify_token(self, token: str):
            assert token == "mcp.jwt.token"
            return expected

    monkeypatch.setattr(mcp_ep, "get_jwt_manager", lambda: _DummyJwtManager())

    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="mcp.jwt.token")

    user = await mcp_ep._resolve_token_data_compat(credentials=creds, x_api_key=None, request=None)

    assert isinstance(user, mcp_ep.TokenData)
    assert user.sub == expected.sub
    assert user.roles == expected.roles
    assert user.permissions == expected.permissions


@pytest.mark.asyncio
async def test_get_current_user_authnz_revoked_does_not_fallback(monkeypatch):
    """
    If an AuthNZ JWT verifies but is revoked/inactive, do not fall back to MCP JWT.
    """
    from fastapi.security.http import HTTPAuthorizationCredentials
    from fastapi import HTTPException, status

    async def _revoked_verify(_request, _token: str):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
        )

    class _FailingJwtManager:
        def verify_token(self, _token: str):
            raise AssertionError("MCP JWT fallback should not be attempted")

    revoked_token = ".".join(("revoked", "jwt", "token"))
    monkeypatch.setattr(mcp_ep, "verify_jwt_and_fetch_user", _revoked_verify)
    monkeypatch.setattr(mcp_ep, "_is_authnz_access_token", lambda token: token == revoked_token)
    monkeypatch.setattr(mcp_ep, "get_jwt_manager", lambda: _FailingJwtManager())

    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=revoked_token)

    user = await mcp_ep._resolve_token_data_compat(credentials=creds, x_api_key=None, request=None)

    assert user is None


@pytest.mark.asyncio
async def test_get_current_user_authnz_and_mcp_failure_use_api_key_and_set_state(monkeypatch):
    """
    When both AuthNZ JWT and MCP JWT fail, compat token resolution should fall back to
    the multi-user API key path, returning a TokenData and attaching API key
    metadata to request.state.mcp_api_key_info.
    """
    from fastapi.security.http import HTTPAuthorizationCredentials
    from starlette.requests import Request

    from fastapi import HTTPException, status

    async def _fail_verify_jwt(_request, _token: str):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
        )

    # AuthNZ JWT validation always fails.
    monkeypatch.setattr(mcp_ep, "verify_jwt_and_fetch_user", _fail_verify_jwt)

    # MCP JWT verify also fails to force API key fallback.
    class _FailingJwtManager:
        def verify_token(self, token: str):
            raise RuntimeError("invalid MCP token")

    monkeypatch.setattr(mcp_ep, "get_jwt_manager", lambda: _FailingJwtManager())

    # Force multi-user API key path (no single-user compat shim).
    monkeypatch.setenv("MCP_SINGLE_USER_COMPAT_SHIM", "0")
    monkeypatch.setattr(mcp_ep, "is_single_user_profile_mode", lambda: False)

    # Dummy API key manager that returns metadata for the key.
    calls: list[dict[str, Any]] = []

    class _DummyApiManager:
        async def validate_api_key(self, key: str, ip_address: Optional[str] = None) -> Dict[str, Any]:
            calls.append({"key": key, "ip": ip_address})
            return {"user_id": "42", "org_id": 99, "team_id": 7, "scope": ["read"]}

    async def _fake_get_api_key_manager():
        return _DummyApiManager()

    monkeypatch.setattr(mcp_ep, "get_api_key_manager", _fake_get_api_key_manager)

    scope = {"type": "http", "client": ("203.0.113.1", 12345)}
    request = Request(scope)

    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="bad.jwt")

    user = await mcp_ep._resolve_token_data_compat(credentials=creds, x_api_key="api-key-xyz", request=request)

    assert isinstance(user, mcp_ep.TokenData)
    assert user.sub == "42"
    assert user.roles == ["api_client"]
    assert user.permissions == ["read"]

    # API key should have been validated exactly once.
    assert len(calls) == 1
    assert calls[0]["key"] == "api-key-xyz"
    # Request state should carry the resolved API key metadata.
    assert getattr(request.state, "mcp_api_key_info", None) == {
        "user_id": "42",
        "org_id": 99,
        "team_id": 7,
        "scope": ["read"],
    }


@pytest.mark.asyncio
async def test_single_user_test_api_key_allowed_in_test_mode_dev_context(monkeypatch):
    """
    In TEST_MODE with a clear dev/test context, SINGLE_USER_TEST_API_KEY should
    be accepted via the single-user compat shim and produce an admin-style TokenData.
    """
    from starlette.requests import Request

    test_key = "test-key-dev-123"

    # Enable compat shim and single-user profile.
    monkeypatch.delenv("MCP_SINGLE_USER_COMPAT_SHIM", raising=False)
    monkeypatch.setattr(mcp_ep, "is_single_user_profile_mode", lambda: True)

    # TEST_MODE enabled and explicit test key configured.
    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("SINGLE_USER_TEST_API_KEY", test_key)

    # Dev-like config: debug_mode True.
    class _Cfg:
        debug_mode = True

    monkeypatch.setattr(mcp_ep, "get_config", lambda: _Cfg())

    # Single-user settings.
    class _Settings:
        SINGLE_USER_API_KEY = "real-single-user-key"
        SINGLE_USER_FIXED_ID = 999

    monkeypatch.setattr(mcp_ep, "get_settings", lambda: _Settings())

    # Guard that multi-user API key manager is not used in this path.
    async def _fail_get_api_key_manager():
        raise AssertionError("API key manager should not be used for SINGLE_USER_TEST_API_KEY in dev TEST_MODE")

    monkeypatch.setattr(mcp_ep, "get_api_key_manager", _fail_get_api_key_manager)

    scope = {"type": "http", "client": ("127.0.0.1", 8000)}
    request = Request(scope)

    user = await mcp_ep._resolve_token_data_compat(credentials=None, x_api_key=test_key, request=request)

    assert isinstance(user, mcp_ep.TokenData)
    assert user.sub == str(_Settings.SINGLE_USER_FIXED_ID)
    assert user.username == "single_user"
    # Admin-style role and wildcard permissions in TEST_MODE.
    assert user.roles == [UserRole.ADMIN.value]
    assert user.permissions == ["*"]


@pytest.mark.asyncio
async def test_single_user_test_api_key_uses_api_key_manager_outside_dev_context(monkeypatch):
    """
    When TEST_MODE is enabled but the environment is non-dev/prod-like, the
    SINGLE_USER_TEST_API_KEY should not be accepted via the single-user compat
    shim; instead it should flow through the multi-user API key manager path.
    """
    from starlette.requests import Request
    import sys as _sys

    test_key = "test-key-prod-456"

    # Enable compat shim and single-user profile.
    monkeypatch.delenv("MCP_SINGLE_USER_COMPAT_SHIM", raising=False)
    monkeypatch.setattr(mcp_ep, "is_single_user_profile_mode", lambda: True)

    # TEST_MODE enabled but environment marked as production-like.
    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("SINGLE_USER_TEST_API_KEY", test_key)
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.delenv("APP_ENV", raising=False)
    monkeypatch.delenv("ENV", raising=False)
    # Prevent the dev/test shortcuts that key off pytest internals.
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delitem(_sys.modules, "pytest", raising=False)

    # Config with debug_mode False to emulate production.
    class _Cfg:
        debug_mode = False

    monkeypatch.setattr(mcp_ep, "get_config", lambda: _Cfg())

    # Single-user settings (real admin API key is different from test key).
    class _Settings:
        SINGLE_USER_API_KEY = "real-single-user-key"
        SINGLE_USER_FIXED_ID = 999

    monkeypatch.setattr(mcp_ep, "get_settings", lambda: _Settings())

    # Multi-user API key manager should be used for the test key in this context.
    calls: list[dict[str, Any]] = []

    class _DummyApiManager:
        async def validate_api_key(self, key: str, ip_address: Optional[str] = None) -> Dict[str, Any]:
            calls.append({"key": key, "ip": ip_address})
            return {"user_id": "777", "org_id": 1, "team_id": 2, "scope": ["read"]}

    async def _fake_get_api_key_manager():
        return _DummyApiManager()

    monkeypatch.setattr(mcp_ep, "get_api_key_manager", _fake_get_api_key_manager)

    scope = {"type": "http", "client": ("198.51.100.5", 9000)}
    request = Request(scope)

    user = await mcp_ep._resolve_token_data_compat(credentials=None, x_api_key=test_key, request=request)

    # The test key should not yield the single-user admin; it should be treated
    # as a regular multi-user API key.
    assert isinstance(user, mcp_ep.TokenData)
    assert user.sub == "777"
    assert user.roles == ["api_client"]
    assert user.permissions == ["read"]

    assert len(calls) == 1
    assert calls[0]["key"] == test_key
