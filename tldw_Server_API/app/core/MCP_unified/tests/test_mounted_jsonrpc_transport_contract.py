"""Mounted MCP JSON-RPC HTTP transport contract tests."""

from __future__ import annotations

import os
from typing import Any

import pytest
from fastapi import FastAPI, HTTPException, status
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect


os.environ.setdefault("TEST_MODE", "true")
os.environ.setdefault("AUTH_MODE", "single_user")
os.environ.setdefault("SINGLE_USER_API_KEY", "test-api-key-1234567890")
os.environ.setdefault("SINGLE_USER_FIXED_ID", "1")
os.environ.setdefault("MCP_JWT_SECRET", "x" * 64)
os.environ.setdefault("MCP_API_KEY_SALT", "s" * 64)
os.environ.setdefault("MCP_ALLOWED_IPS", "")


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
            }
        )
        if kwargs.get("user_id") is None:
            return MCPResponse(error=MCPError(code=-32001, message="Insufficient permissions"), id=request.id)
        return MCPResponse(result={"tools": []}, id=request.id)


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
    server.initialized = True
    server.protocol = protocol
    server.auth_provider = _RejectingWsAuthProvider()
    server.config.ws_auth_required = True
    server.config.ws_allow_query_auth = True
    server.config.allowed_client_ips = []
    server.config.blocked_client_ips = []
    server.config.debug_mode = debug_mode
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
