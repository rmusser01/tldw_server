"""Mounted MCP JSON-RPC HTTP transport contract tests."""

from __future__ import annotations

import os
from typing import Any

import pytest
from fastapi import FastAPI, HTTPException, status
from fastapi.testclient import TestClient


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
