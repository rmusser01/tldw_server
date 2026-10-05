"""Mounted MCP JSON-RPC HTTP transport contract tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI, HTTPException, status
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect


@pytest.fixture(autouse=True)
def mounted_mcp_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scope mounted MCP environment defaults to each test."""
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", "test-api-key-1234567890")
    monkeypatch.setenv("SINGLE_USER_FIXED_ID", "1")
    monkeypatch.setenv("MCP_JWT_SECRET", "x" * 64)
    monkeypatch.setenv("MCP_API_KEY_SALT", "s" * 64)
    monkeypatch.setenv("MCP_ALLOWED_IPS", "")


def build_mcp_admin_auth_override():
    """Return a dependency override representing an authenticated MCP admin."""
    from tldw_Server_API.app.api.v1.endpoints.mcp_unified_endpoint import McpAuthContext
    from tldw_Server_API.app.core.MCP_unified.auth.jwt_manager import TokenData

    async def _override() -> McpAuthContext:
        return McpAuthContext(
            user=TokenData(sub="1", roles=["admin"], permissions=["*"]),
            principal=None,
            api_key_info=None,
            raw_api_key=None,
        )

    return _override


def build_mcp_test_client(auth_principal_override: Any | None = None) -> TestClient:
    """Build a minimal app with the mounted MCP router and optional auth override."""
    from tldw_Server_API.app.api.v1.endpoints import mcp_unified_endpoint

    app = FastAPI()
    app.include_router(mcp_unified_endpoint.router, prefix="/api/v1")
    if auth_principal_override is not None:
        app.dependency_overrides[mcp_unified_endpoint.get_mcp_auth_context] = auth_principal_override
    return TestClient(app)


def _auth_headers() -> dict[str, str]:
    return {"Authorization": "Bearer test"}


class _NoStoredApiKeyManager:
    async def validate_api_key(self, *_args: Any, **_kwargs: Any) -> None:
        return None


class _SingleUserSettings:
    AUTH_MODE = "single_user"
    SINGLE_USER_FIXED_ID = 1
    SINGLE_USER_ALLOWED_IPS: list[str] = []

    def __init__(self, api_key: str) -> None:
        self.SINGLE_USER_API_KEY = api_key


class _DebugConfig:
    def __init__(self, *, debug_mode: bool) -> None:
        self.debug_mode = debug_mode


def _install_mounted_http_single_user_compat(
    monkeypatch: pytest.MonkeyPatch,
    *,
    api_key: str,
    test_mode: bool,
    debug_mode: bool = False,
    ip_allowed: bool = True,
    test_key: str | None = None,
) -> None:
    from tldw_Server_API.app.api.v1.endpoints import mcp_unified_endpoint

    async def _no_principal(_request: Any) -> None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="No principal")

    async def _get_api_key_manager() -> _NoStoredApiKeyManager:
        return _NoStoredApiKeyManager()

    monkeypatch.setattr(mcp_unified_endpoint, "get_auth_principal", _no_principal)
    monkeypatch.setattr(mcp_unified_endpoint, "is_single_user_profile_mode", lambda: True)
    monkeypatch.setattr(mcp_unified_endpoint, "is_test_mode", lambda: test_mode)
    monkeypatch.setattr(mcp_unified_endpoint, "env_flag_enabled", lambda _name: False)
    monkeypatch.setattr(mcp_unified_endpoint, "get_settings", lambda: _SingleUserSettings(api_key))
    monkeypatch.setattr(mcp_unified_endpoint, "get_config", lambda: _DebugConfig(debug_mode=debug_mode))
    monkeypatch.setattr(mcp_unified_endpoint, "is_single_user_ip_allowed", lambda _ip, _settings: ip_allowed)
    monkeypatch.setattr(mcp_unified_endpoint, "get_api_key_manager", _get_api_key_manager)
    if test_key is None:
        monkeypatch.delenv("SINGLE_USER_TEST_API_KEY", raising=False)
    else:
        monkeypatch.setenv("SINGLE_USER_TEST_API_KEY", test_key)


class _RecordingHttpAuthServer:
    initialized = True

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def initialize(self) -> None:
        self.initialized = True

    async def handle_http_request(self, request: Any, *_args: Any, **kwargs: Any):
        from tldw_Server_API.app.core.MCP_unified import MCPResponse
        from tldw_Server_API.app.core.MCP_unified.protocol import MCPError

        self.calls.append(
            {
                "request": request,
                "user_id": kwargs.get("user_id"),
                "metadata": dict(kwargs.get("metadata") or {}),
                "server_auth_scope": kwargs.get("server_auth_scope"),
            }
        )
        if kwargs.get("user_id") is None:
            return MCPResponse(error=MCPError(code=-32001, message="Insufficient permissions"), id=request.id)
        return MCPResponse(result={"tools": []}, id=request.id)

    async def handle_http_batch(self, requests: list[Any], *_args: Any, **kwargs: Any):
        from tldw_Server_API.app.core.MCP_unified import MCPResponse

        self.calls.append(
            {
                "requests": requests,
                "user_id": kwargs.get("user_id"),
                "metadata": dict(kwargs.get("metadata") or {}),
                "server_auth_scope": kwargs.get("server_auth_scope"),
            }
        )
        return [MCPResponse(result={"ok": True}, id=request.id) for request in requests]


def test_mounted_http_transports_forward_authenticated_scope_separately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.api.v1.endpoints import mcp_unified_endpoint
    from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
    from tldw_Server_API.app.core.MCP_unified.auth.jwt_manager import TokenData
    from tldw_Server_API.app.core.MCP_unified.protocol_types import AuthenticatedExecutionScope

    scope = AuthenticatedExecutionScope(active_org_id=7, active_team_id=11)

    async def _scoped_auth_override() -> Any:
        return mcp_unified_endpoint.McpAuthContext(
            user=TokenData(sub="41", roles=["admin"], permissions=["*"]),
            principal=AuthPrincipal(
                kind="user",
                user_id=41,
                active_org_id=7,
                active_team_id=11,
            ),
            api_key_info=None,
            raw_api_key=None,
        )

    server = _RecordingHttpAuthServer()
    monkeypatch.setattr(mcp_unified_endpoint, "get_mcp_server", lambda: server)

    with build_mcp_test_client(auth_principal_override=_scoped_auth_override) as client:
        response = client.post(
            "/api/v1/mcp/request",
            json={"jsonrpc": "2.0", "method": "tools/list", "id": "scoped-request"},
            headers=_auth_headers(),
        )
        batch_response = client.post(
            "/api/v1/mcp/request/batch",
            json=[{"jsonrpc": "2.0", "method": "ping", "id": "scoped-batch"}],
            headers=_auth_headers(),
        )

    assert response.status_code == 200
    assert batch_response.status_code == 200
    assert [call["server_auth_scope"] for call in server.calls] == [scope, scope]
    assert all("server_auth_scope" not in call["metadata"] for call in server.calls)


def test_mounted_http_principal_dependency_projects_authenticated_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.api.v1.endpoints import mcp_unified_endpoint
    from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
    from tldw_Server_API.app.core.MCP_unified.protocol_types import AuthenticatedExecutionScope

    principal = AuthPrincipal(
        kind="user",
        user_id=41,
        roles=["admin"],
        permissions=["*"],
        active_org_id=7,
        active_team_id=11,
    )

    async def _scoped_principal(_request: Any) -> AuthPrincipal:
        return principal

    server = _RecordingHttpAuthServer()
    monkeypatch.setattr(mcp_unified_endpoint, "get_auth_principal", _scoped_principal)
    monkeypatch.setattr(mcp_unified_endpoint, "get_mcp_server", lambda: server)

    with build_mcp_test_client() as client:
        response = client.post(
            "/api/v1/mcp/request",
            json={"jsonrpc": "2.0", "method": "tools/list", "id": "scoped-principal"},
            headers=_auth_headers(),
        )

    assert response.status_code == 200
    assert server.calls[-1]["user_id"] == "41"
    assert server.calls[-1]["server_auth_scope"] == AuthenticatedExecutionScope(
        active_org_id=7,
        active_team_id=11,
    )


def test_mounted_http_single_user_api_key_attaches_trusted_mounted_metadata(
    monkeypatch: pytest.MonkeyPatch,
):
    from tldw_Server_API.app.api.v1.endpoints import mcp_unified_endpoint

    api_key = "primary-single-user-key-12345"
    _install_mounted_http_single_user_compat(
        monkeypatch,
        api_key=api_key,
        test_mode=False,
    )
    server = _RecordingHttpAuthServer()
    monkeypatch.setattr(mcp_unified_endpoint, "get_mcp_server", lambda: server)

    with build_mcp_test_client() as client:
        response = client.post(
            "/api/v1/mcp/request",
            json={"jsonrpc": "2.0", "method": "tools/list", "id": "http-primary-list"},
            headers={"X-API-KEY": api_key},
        )

    assert response.status_code == 200
    assert response.json()["result"] == {"tools": []}
    assert server.calls[-1]["user_id"] == "1"
    metadata = server.calls[-1]["metadata"]
    assert metadata["auth_via"] == "single_user_api_key"
    assert metadata["trusted_auth_claims"] is True
    assert metadata["compat_claims_source"] == "mounted_http"


def test_mounted_http_single_user_test_api_key_rejected_when_test_mode_false(
    monkeypatch: pytest.MonkeyPatch,
):
    from tldw_Server_API.app.api.v1.endpoints import mcp_unified_endpoint

    api_key = "primary-single-user-key-12345"
    test_key = "test-single-user-key-12345"
    _install_mounted_http_single_user_compat(
        monkeypatch,
        api_key=api_key,
        test_mode=False,
        test_key=test_key,
    )
    server = _RecordingHttpAuthServer()
    monkeypatch.setattr(mcp_unified_endpoint, "get_mcp_server", lambda: server)

    with build_mcp_test_client() as client:
        response = client.post(
            "/api/v1/mcp/request",
            json={"jsonrpc": "2.0", "method": "tools/list", "id": "http-test-reject"},
            headers={"X-API-KEY": test_key},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["error"]["code"] == -32001
    assert server.calls[-1]["user_id"] is None
    metadata = server.calls[-1]["metadata"]
    assert "trusted_auth_claims" not in metadata
    assert metadata.get("auth_via") != "single_user_test_api_key"


def test_mounted_http_single_user_test_api_key_attaches_test_metadata_with_guard(
    monkeypatch: pytest.MonkeyPatch,
):
    from tldw_Server_API.app.api.v1.endpoints import mcp_unified_endpoint

    api_key = "primary-single-user-key-12345"
    test_key = "test-single-user-key-12345"
    _install_mounted_http_single_user_compat(
        monkeypatch,
        api_key=api_key,
        test_mode=True,
        debug_mode=True,
        test_key=test_key,
    )
    server = _RecordingHttpAuthServer()
    monkeypatch.setattr(mcp_unified_endpoint, "get_mcp_server", lambda: server)

    with build_mcp_test_client() as client:
        response = client.post(
            "/api/v1/mcp/request",
            json={"jsonrpc": "2.0", "method": "tools/list", "id": "http-test-list"},
            headers={"X-API-KEY": test_key},
        )

    assert response.status_code == 200
    assert response.json()["result"] == {"tools": []}
    metadata = server.calls[-1]["metadata"]
    assert metadata["auth_via"] == "single_user_test_api_key"
    assert metadata["trusted_auth_claims"] is True
    assert metadata["compat_claims_source"] == "mounted_http"


def test_mounted_http_single_user_api_key_does_not_attach_trust_when_ip_rejected(
    monkeypatch: pytest.MonkeyPatch,
):
    from tldw_Server_API.app.api.v1.endpoints import mcp_unified_endpoint
    from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal

    api_key = "primary-single-user-key-12345"
    _install_mounted_http_single_user_compat(
        monkeypatch,
        api_key=api_key,
        test_mode=False,
        ip_allowed=False,
    )

    async def _valid_principal(_request: Any) -> AuthPrincipal:
        return AuthPrincipal(kind="user", user_id=7, roles=["user"], permissions=[])

    monkeypatch.setattr(mcp_unified_endpoint, "get_auth_principal", _valid_principal)
    server = _RecordingHttpAuthServer()
    monkeypatch.setattr(mcp_unified_endpoint, "get_mcp_server", lambda: server)

    with build_mcp_test_client() as client:
        response = client.post(
            "/api/v1/mcp/request",
            json={"jsonrpc": "2.0", "method": "tools/list", "id": "http-ip-reject"},
            headers={"X-API-KEY": api_key},
        )

    assert response.status_code == 200
    assert response.json()["result"] == {"tools": []}
    assert server.calls[-1]["user_id"] == "7"
    metadata = server.calls[-1]["metadata"]
    assert metadata.get("auth_via") != "single_user_api_key"
    assert "trusted_auth_claims" not in metadata
    assert "compat_claims_source" not in metadata
    assert not any(str(key).startswith("_server_auth_") for key in metadata)


class _RejectingWsAuthProvider:
    async def authenticate_authnz_websocket_token(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    async def validate_api_key(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def normalize_api_key_permissions(self, _info: Any) -> list[str]:
        return []

    def is_authnz_access_token(self, _token: str) -> bool:
        return False


class _RecordingWsProtocol:
    def __init__(self) -> None:
        self.contexts: list[Any] = []

    async def process_request(self, request: Any, context: Any) -> Any:
        from tldw_Server_API.app.core.MCP_unified import MCPResponse

        self.contexts.append(context)
        request_id = request.get("id") if isinstance(request, dict) else getattr(request, "id", None)
        return MCPResponse(result={"tools": []}, id=request_id)


def _install_mounted_ws_single_user_compat(
    monkeypatch: pytest.MonkeyPatch,
    *,
    server: Any,
    api_key: str,
    test_mode: bool,
    debug_mode: bool = False,
    test_key: str | None = None,
) -> _RecordingWsProtocol:
    from tldw_Server_API.app.api.v1.endpoints import mcp_unified_endpoint
    from tldw_Server_API.app.core.MCP_unified import server as mcp_server_module

    protocol = _RecordingWsProtocol()
    monkeypatch.setattr(server, "initialized", True)
    monkeypatch.setattr(server, "protocol", protocol)
    monkeypatch.setattr(server, "auth_provider", _RejectingWsAuthProvider())
    monkeypatch.setattr(server.config, "ws_auth_required", True)
    monkeypatch.setattr(server.config, "ws_allow_query_auth", True)
    monkeypatch.setattr(server.config, "allowed_client_ips", [])
    monkeypatch.setattr(server.config, "blocked_client_ips", [])
    monkeypatch.setattr(server.config, "debug_mode", debug_mode)
    monkeypatch.setattr(server, "_is_test_mode", lambda: test_mode)
    monkeypatch.setattr(server, "_is_explicit_pytest_runtime", lambda: test_mode)
    monkeypatch.setattr(server, "_env_flag_enabled", lambda _name: False)
    monkeypatch.setattr(mcp_server_module, "is_single_user_profile_mode", lambda: True, raising=False)
    monkeypatch.setattr(mcp_server_module, "get_settings", lambda: _SingleUserSettings(api_key), raising=False)
    monkeypatch.setattr(mcp_server_module, "is_single_user_ip_allowed", lambda _ip, _settings: True, raising=False)
    monkeypatch.setattr(mcp_unified_endpoint, "get_mcp_server", lambda: server)
    if test_key is None:
        monkeypatch.delenv("SINGLE_USER_TEST_API_KEY", raising=False)
    else:
        monkeypatch.setenv("SINGLE_USER_TEST_API_KEY", test_key)
    return protocol


class _ScopedWsAuthProvider:
    def __init__(
        self,
        *,
        identity: Any | None = None,
        api_key_info: dict[str, Any] | None = None,
        is_authnz_token: bool | None = None,
    ) -> None:
        self.identity = identity
        self.api_key_info = api_key_info
        self.is_authnz_token = identity is not None if is_authnz_token is None else is_authnz_token

    async def authenticate_authnz_websocket_token(self, *_args: Any, **_kwargs: Any) -> Any | None:
        return self.identity

    async def validate_api_key(self, *_args: Any, **_kwargs: Any) -> dict[str, Any] | None:
        return self.api_key_info

    def normalize_api_key_permissions(self, _info: Any) -> list[str]:
        return []

    def is_authnz_access_token(self, _token: str) -> bool:
        return self.is_authnz_token


@pytest.mark.parametrize("auth_kind", ["authnz_jwt", "api_key"])
def test_mounted_ws_propagates_authenticated_scope(
    monkeypatch: pytest.MonkeyPatch,
    auth_kind: str,
) -> None:
    from tldw_Server_API.app.core.MCP_unified import get_mcp_server
    from tldw_Server_API.app.core.MCP_unified.interfaces.runtime import AuthenticatedIdentity
    from tldw_Server_API.app.core.MCP_unified.protocol_types import AuthenticatedExecutionScope

    server = get_mcp_server()
    protocol = _install_mounted_ws_single_user_compat(
        monkeypatch,
        server=server,
        api_key="unrelated-single-user-key-12345",
        test_mode=True,
    )
    if auth_kind == "authnz_jwt":
        server.auth_provider = _ScopedWsAuthProvider(
            identity=AuthenticatedIdentity(
                user_id="41",
                roles=["user"],
                permissions=["mcp:read"],
                active_org_id=7,
                active_team_id=11,
            )
        )
        headers = {"Authorization": "Bearer scoped-authnz-token"}
    else:
        server.auth_provider = _ScopedWsAuthProvider(
            api_key_info={"user_id": "41", "org_id": 7, "team_id": 11}
        )
        headers = {"X-API-KEY": "scoped-api-key"}

    with build_mcp_test_client() as client:
        with client.websocket_connect(
            f"/api/v1/mcp/ws?client_id=ws-scoped-{auth_kind}",
            headers=headers,
        ) as ws:
            ws.send_json({"jsonrpc": "2.0", "method": "tools/list", "id": f"ws-{auth_kind}"})
            body = ws.receive_json()

    assert body["result"] == {"tools": []}
    assert protocol.contexts[-1].server_auth_scope == AuthenticatedExecutionScope(
        active_org_id=7,
        active_team_id=11,
    )


def test_mounted_ws_authnz_identity_takes_precedence_over_api_key_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.MCP_unified import get_mcp_server
    from tldw_Server_API.app.core.MCP_unified.interfaces.runtime import AuthenticatedIdentity
    from tldw_Server_API.app.core.MCP_unified.protocol_types import AuthenticatedExecutionScope

    class _RecordingAuthProvider(_ScopedWsAuthProvider):
        def __init__(self) -> None:
            super().__init__(
                identity=AuthenticatedIdentity(
                    user_id="41",
                    active_org_id=7,
                    active_team_id=11,
                ),
                api_key_info={"user_id": "99", "org_id": 13, "team_id": 17},
            )
            self.api_key_calls = 0

        async def validate_api_key(self, *_args: Any, **_kwargs: Any) -> dict[str, Any] | None:
            self.api_key_calls += 1
            return self.api_key_info

    server = get_mcp_server()
    protocol = _install_mounted_ws_single_user_compat(
        monkeypatch,
        server=server,
        api_key="unrelated-single-user-key-12345",
        test_mode=True,
    )
    auth_provider = _RecordingAuthProvider()
    server.auth_provider = auth_provider

    with build_mcp_test_client() as client:
        with client.websocket_connect(
            "/api/v1/mcp/ws?client_id=ws-mixed-credentials",
            headers={
                "Authorization": "Bearer valid-authnz-token",
                "X-API-KEY": "ignored-api-key",
            },
        ) as ws:
            ws.send_json({"jsonrpc": "2.0", "method": "tools/list", "id": "mixed-credentials"})
            body = ws.receive_json()

    assert body["result"] == {"tools": []}
    assert auth_provider.api_key_calls == 0
    assert protocol.contexts[-1].user_id == "41"
    assert protocol.contexts[-1].server_auth_scope == AuthenticatedExecutionScope(
        active_org_id=7,
        active_team_id=11,
    )


def test_mounted_ws_rejects_malformed_api_key_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.MCP_unified import get_mcp_server

    server = get_mcp_server()
    _install_mounted_ws_single_user_compat(
        monkeypatch,
        server=server,
        api_key="unrelated-single-user-key-12345",
        test_mode=True,
    )
    server.auth_provider = _ScopedWsAuthProvider(
        api_key_info={"user_id": "41", "org_id": True}
    )

    with build_mcp_test_client() as client:
        with pytest.raises(WebSocketDisconnect) as exc_info:
            with client.websocket_connect(
                "/api/v1/mcp/ws?client_id=ws-malformed-scope",
                headers={"X-API-KEY": "malformed-scope-key"},
            ):
                pass

    assert exc_info.value.code == 1008
    assert exc_info.value.reason == "Authentication failed"


def test_mounted_ws_malformed_authnz_identity_cannot_fall_back_to_mcp_jwt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.MCP_unified import get_mcp_server

    class _RecordingJwtManager:
        def __init__(self) -> None:
            self.calls = 0

        def verify_token(self, _token: str) -> None:
            self.calls += 1
            raise HTTPException(status_code=401, detail="invalid")

    server = get_mcp_server()
    _install_mounted_ws_single_user_compat(
        monkeypatch,
        server=server,
        api_key="unrelated-single-user-key-12345",
        test_mode=True,
    )
    server.auth_provider = _ScopedWsAuthProvider(
        identity=SimpleNamespace(
            user_id="41",
            roles=[],
            permissions=[],
            active_org_id=True,
            active_team_id=None,
        ),
        is_authnz_token=False,
    )
    jwt_manager = _RecordingJwtManager()
    monkeypatch.setattr(server, "jwt_manager", jwt_manager)

    with build_mcp_test_client() as client:
        with pytest.raises(WebSocketDisconnect) as exc_info:
            with client.websocket_connect(
                "/api/v1/mcp/ws?client_id=ws-malformed-authnz",
                headers={"Authorization": "Bearer malformed-authnz-token"},
            ):
                pass

    assert jwt_manager.calls == 0
    assert exc_info.value.code == 1008
    assert exc_info.value.reason == "Authentication failed"


def test_mounted_ws_malformed_authnz_identity_cannot_fall_back_to_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.MCP_unified import get_mcp_server

    class _RecordingAuthProvider(_ScopedWsAuthProvider):
        def __init__(self) -> None:
            super().__init__(
                identity=SimpleNamespace(
                    user_id="41",
                    roles=[],
                    permissions=[],
                    active_org_id=True,
                    active_team_id=None,
                ),
                api_key_info={"user_id": "99", "org_id": 7},
                is_authnz_token=False,
            )
            self.api_key_calls = 0

        async def validate_api_key(self, *_args: Any, **_kwargs: Any) -> dict[str, Any] | None:
            self.api_key_calls += 1
            return self.api_key_info

    server = get_mcp_server()
    _install_mounted_ws_single_user_compat(
        monkeypatch,
        server=server,
        api_key="unrelated-single-user-key-12345",
        test_mode=True,
    )
    auth_provider = _RecordingAuthProvider()
    server.auth_provider = auth_provider

    with build_mcp_test_client() as client:
        with pytest.raises(WebSocketDisconnect) as exc_info:
            with client.websocket_connect(
                "/api/v1/mcp/ws?client_id=ws-malformed-authnz-api-key",
                headers={
                    "Authorization": "Bearer malformed-authnz-token",
                    "X-API-KEY": "fallback-api-key",
                },
            ) as ws:
                ws.send_json({"jsonrpc": "2.0", "method": "tools/list", "id": "scope-downgrade"})
                ws.receive_json()

    assert exc_info.value.code == 1008
    assert exc_info.value.reason == "Authentication failed"
    assert auth_provider.api_key_calls == 0


def test_mounted_ws_authnz_identity_without_user_cannot_fall_back_to_mcp_jwt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.MCP_unified import get_mcp_server

    class _RecordingJwtManager:
        def __init__(self) -> None:
            self.calls = 0

        def verify_token(self, _token: str) -> None:
            self.calls += 1
            raise HTTPException(status_code=401, detail="invalid")

    server = get_mcp_server()
    _install_mounted_ws_single_user_compat(
        monkeypatch,
        server=server,
        api_key="unrelated-single-user-key-12345",
        test_mode=True,
    )
    server.auth_provider = _ScopedWsAuthProvider(
        identity=SimpleNamespace(
            user_id="",
            roles=[],
            permissions=[],
            active_org_id=None,
            active_team_id=None,
        ),
        is_authnz_token=False,
    )
    jwt_manager = _RecordingJwtManager()
    monkeypatch.setattr(server, "jwt_manager", jwt_manager)

    with build_mcp_test_client() as client:
        with pytest.raises(WebSocketDisconnect) as exc_info:
            with client.websocket_connect(
                "/api/v1/mcp/ws?client_id=ws-missing-authnz-user",
                headers={"Authorization": "Bearer malformed-authnz-token"},
            ):
                pass

    assert jwt_manager.calls == 0
    assert exc_info.value.code == 1008
    assert exc_info.value.reason == "Authentication failed"


def test_mounted_ws_personal_mcp_jwt_preserves_absent_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.MCP_unified import get_mcp_server

    server = get_mcp_server()
    protocol = _install_mounted_ws_single_user_compat(
        monkeypatch,
        server=server,
        api_key="unrelated-single-user-key-12345",
        test_mode=True,
    )
    token = server.jwt_manager.create_access_token(subject="41")

    with build_mcp_test_client() as client:
        with client.websocket_connect(
            "/api/v1/mcp/ws?client_id=ws-personal-jwt",
            headers={"Authorization": f"Bearer {token}"},
        ) as ws:
            ws.send_json({"jsonrpc": "2.0", "method": "tools/list", "id": "ws-personal-jwt"})
            body = ws.receive_json()

    assert body["result"] == {"tools": []}
    assert protocol.contexts[-1].server_auth_scope is None


def test_mounted_ws_single_user_cookie_preserves_absent_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.api.v1.endpoints import mcp_unified_endpoint
    from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
    from tldw_Server_API.app.core.MCP_unified import get_mcp_server

    server = get_mcp_server()
    protocol = _install_mounted_ws_single_user_compat(
        monkeypatch,
        server=server,
        api_key="unrelated-single-user-key-12345",
        test_mode=True,
    )

    async def _cookie_identity(websocket: Any) -> Any:
        websocket.state.single_user_session_id = 17
        websocket.state.user_id = 41
        websocket.state.single_user_cookie_websocket_close_code = None
        websocket.state.auth_principal = AuthPrincipal(
            kind="user",
            user_id=41,
            token_type="single_user_session",
            roles=["admin"],
            permissions=["*"],
        )
        return object()

    monkeypatch.setattr(
        mcp_unified_endpoint,
        "resolve_single_user_cookie_websocket",
        _cookie_identity,
    )

    with build_mcp_test_client() as client:
        with client.websocket_connect("/api/v1/mcp/ws?client_id=ws-cookie") as ws:
            ws.send_json({"jsonrpc": "2.0", "method": "tools/list", "id": "ws-cookie"})
            body = ws.receive_json()

    assert body["result"] == {"tools": []}
    assert protocol.contexts[-1].server_auth_scope is None


def test_mounted_ws_single_user_api_key_attaches_trusted_mounted_metadata(
    monkeypatch: pytest.MonkeyPatch,
):
    from tldw_Server_API.app.core.MCP_unified import get_mcp_server

    api_key = "primary-single-user-key-12345"
    server = get_mcp_server()
    protocol = _install_mounted_ws_single_user_compat(
        monkeypatch,
        server=server,
        api_key=api_key,
        test_mode=False,
    )

    with build_mcp_test_client() as client:
        with client.websocket_connect(
            "/api/v1/mcp/ws?client_id=ws-primary",
            headers={"X-API-KEY": api_key},
        ) as ws:
            ws.send_json({"jsonrpc": "2.0", "method": "tools/list", "id": "ws-primary-list"})
            body = ws.receive_json()

    assert body["result"] == {"tools": []}
    context = protocol.contexts[-1]
    assert context.user_id == "1"
    assert context.metadata["auth_via"] == "single_user_api_key"
    assert context.metadata["trusted_auth_claims"] is True
    assert context.metadata["compat_claims_source"] == "mounted_ws"
    assert context.server_auth_scope is None


def test_mounted_ws_single_user_test_api_key_rejected_when_test_mode_false(
    monkeypatch: pytest.MonkeyPatch,
):
    from tldw_Server_API.app.core.MCP_unified import get_mcp_server

    api_key = "primary-single-user-key-12345"
    test_key = "test-single-user-key-12345"
    server = get_mcp_server()
    _install_mounted_ws_single_user_compat(
        monkeypatch,
        server=server,
        api_key=api_key,
        test_mode=False,
        test_key=test_key,
    )

    with build_mcp_test_client() as client:
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(
                "/api/v1/mcp/ws?client_id=ws-test-reject",
                headers={"X-API-KEY": test_key},
            ):
                pass


def test_mounted_ws_single_user_test_api_key_attaches_test_metadata_with_guard(
    monkeypatch: pytest.MonkeyPatch,
):
    from tldw_Server_API.app.core.MCP_unified import get_mcp_server

    api_key = "primary-single-user-key-12345"
    test_key = "test-single-user-key-12345"
    server = get_mcp_server()
    protocol = _install_mounted_ws_single_user_compat(
        monkeypatch,
        server=server,
        api_key=api_key,
        test_mode=True,
        debug_mode=True,
        test_key=test_key,
    )

    with build_mcp_test_client() as client:
        with client.websocket_connect(
            "/api/v1/mcp/ws?client_id=ws-test",
            headers={"X-API-KEY": test_key},
        ) as ws:
            ws.send_json({"jsonrpc": "2.0", "method": "tools/list", "id": "ws-test-list"})
            body = ws.receive_json()

    assert body["result"] == {"tools": []}
    context = protocol.contexts[-1]
    assert context.user_id == "1"
    assert context.metadata["auth_via"] == "single_user_test_api_key"
    assert context.metadata["trusted_auth_claims"] is True
    assert context.metadata["compat_claims_source"] == "mounted_ws"
    assert context.server_auth_scope is None


def test_mounted_request_success_omits_error():
    with build_mcp_test_client(auth_principal_override=build_mcp_admin_auth_override()) as client:
        response = client.post(
            "/api/v1/mcp/request",
            json={"jsonrpc": "2.0", "method": "ping", "id": "ping-1"},
            headers=_auth_headers(),
        )

    assert response.status_code == 200
    body = response.json()
    assert body["jsonrpc"] == "2.0"
    assert body["id"] == "ping-1"
    assert "result" in body
    assert "error" not in body


def test_mounted_request_invalid_json_returns_jsonrpc_parse_error():
    with build_mcp_test_client(auth_principal_override=build_mcp_admin_auth_override()) as client:
        response = client.post(
            "/api/v1/mcp/request",
            content=b"{not-json",
            headers={"content-type": "application/json", **_auth_headers()},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["jsonrpc"] == "2.0"
    assert body["id"] is None
    assert body["error"]["code"] == -32700
    assert "result" not in body


def test_mounted_request_invalid_envelope_returns_jsonrpc_invalid_request():
    with build_mcp_test_client(auth_principal_override=build_mcp_admin_auth_override()) as client:
        response = client.post(
            "/api/v1/mcp/request",
            json={"jsonrpc": "2.0", "params": {}, "id": "bad-envelope"},
            headers=_auth_headers(),
        )

    assert response.status_code == 200
    body = response.json()
    assert body["jsonrpc"] == "2.0"
    assert body["id"] == "bad-envelope"
    assert body["error"]["code"] == -32600
    assert "result" not in body


def test_mounted_request_initialized_notification_returns_204_empty_body():
    with build_mcp_test_client(auth_principal_override=build_mcp_admin_auth_override()) as client:
        response = client.post(
            "/api/v1/mcp/request",
            json={"jsonrpc": "2.0", "method": "notifications/initialized"},
            headers=_auth_headers(),
        )

    assert response.status_code == 204
    assert response.content == b""


def test_mounted_request_notification_is_delivered_to_server_before_response_suppression(
    monkeypatch: pytest.MonkeyPatch,
):
    from tldw_Server_API.app.api.v1.endpoints import mcp_unified_endpoint
    from tldw_Server_API.app.core.MCP_unified import MCPRequest, MCPResponse

    class _RecordingServer:
        initialized = True

        def __init__(self) -> None:
            self.requests: list[MCPRequest] = []

        async def initialize(self) -> None:
            self.initialized = True

        async def handle_http_request(
            self,
            request: MCPRequest,
            *_args: Any,
            **_kwargs: Any,
        ) -> MCPResponse:
            self.requests.append(request)
            return MCPResponse(result={"should": "be suppressed"}, id=request.id)

    server = _RecordingServer()
    monkeypatch.setattr(mcp_unified_endpoint, "get_mcp_server", lambda: server)

    with build_mcp_test_client(auth_principal_override=build_mcp_admin_auth_override()) as client:
        response = client.post(
            "/api/v1/mcp/request",
            json={"jsonrpc": "2.0", "method": "notifications/initialized"},
            headers=_auth_headers(),
        )

    assert response.status_code == 204
    assert response.content == b""
    assert [request.method for request in server.requests] == ["notifications/initialized"]


def test_mounted_request_explicit_null_id_returns_null_id_response():
    with build_mcp_test_client(auth_principal_override=build_mcp_admin_auth_override()) as client:
        response = client.post(
            "/api/v1/mcp/request",
            json={"jsonrpc": "2.0", "method": "ping", "id": None},
            headers=_auth_headers(),
        )

    assert response.status_code == 200
    body = response.json()
    assert body["jsonrpc"] == "2.0"
    assert body["id"] is None
    assert "result" in body
    assert "error" not in body


def test_mounted_request_post_protocol_authz_failure_stays_jsonrpc_200(monkeypatch: pytest.MonkeyPatch):
    from tldw_Server_API.app.api.v1.endpoints import mcp_unified_endpoint
    from tldw_Server_API.app.core.MCP_unified import MCPResponse
    from tldw_Server_API.app.core.MCP_unified.protocol import MCPError

    class _DenyingServer:
        initialized = True

        async def initialize(self) -> None:
            self.initialized = True

        async def handle_http_request(self, *_args: Any, **_kwargs: Any) -> MCPResponse:
            return MCPResponse(error=MCPError(code=-32001, message="Insufficient permissions"), id="deny-1")

    monkeypatch.setattr(mcp_unified_endpoint, "get_mcp_server", lambda: _DenyingServer())

    with build_mcp_test_client(auth_principal_override=build_mcp_admin_auth_override()) as client:
        response = client.post(
            "/api/v1/mcp/request",
            json={
                "jsonrpc": "2.0",
                "method": "tools/call",
                "params": {"name": "restricted", "arguments": {}},
                "id": "deny-1",
            },
            headers=_auth_headers(),
        )

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == "deny-1"
    assert body["error"]["code"] == -32001
    assert "result" not in body


def test_mounted_request_pre_protocol_auth_dependency_failure_stays_http_error():
    async def _auth_failure():
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden before JSON-RPC")

    with build_mcp_test_client(auth_principal_override=_auth_failure) as client:
        response = client.post(
            "/api/v1/mcp/request",
            json={"jsonrpc": "2.0", "method": "ping", "id": "pre-auth"},
            headers=_auth_headers(),
        )

    assert response.status_code == 403
    assert response.json()["detail"] == "Forbidden before JSON-RPC"


def test_mounted_batch_notification_is_delivered_to_server_and_omitted_from_response(
    monkeypatch: pytest.MonkeyPatch,
):
    from tldw_Server_API.app.api.v1.endpoints import mcp_unified_endpoint
    from tldw_Server_API.app.core.MCP_unified import MCPRequest, MCPResponse
    from tldw_Server_API.app.core.MCP_unified.protocol import MCPError

    class _RecordingBatchServer:
        initialized = True

        def __init__(self) -> None:
            self.batch_requests: list[MCPRequest] = []

        async def initialize(self) -> None:
            self.initialized = True

        async def handle_http_batch(
            self,
            requests: list[MCPRequest],
            *_args: Any,
            **_kwargs: Any,
        ) -> list[MCPResponse]:
            self.batch_requests.extend(requests)
            return [
                MCPResponse(
                    error=MCPError(code=-32601, message="notification response should be suppressed"),
                    id=None,
                ),
                MCPResponse(result={"pong": True}, id="batch-ping"),
            ]

    server = _RecordingBatchServer()
    monkeypatch.setattr(mcp_unified_endpoint, "get_mcp_server", lambda: server)

    with build_mcp_test_client(auth_principal_override=build_mcp_admin_auth_override()) as client:
        response = client.post(
            "/api/v1/mcp/request/batch",
            json=[
                {"jsonrpc": "2.0", "method": "notifications/initialized"},
                {"jsonrpc": "2.0", "method": "ping", "id": "batch-ping"},
            ],
            headers=_auth_headers(),
        )

    assert response.status_code == 200
    body = response.json()
    assert body == [{"jsonrpc": "2.0", "id": "batch-ping", "result": {"pong": True}}]
    assert [request.method for request in server.batch_requests] == [
        "notifications/initialized",
        "ping",
    ]
