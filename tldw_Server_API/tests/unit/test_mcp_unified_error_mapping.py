"""Error mapping tests for MCP unified HTTP endpoints."""
from __future__ import annotations

from typing import Any

import base64

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from tldw_Server_API.app.api.v1.endpoints import mcp_unified_endpoint as mcp
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.MCP_unified.auth.jwt_manager import TokenData
from tldw_Server_API.app.core.MCP_unified.protocol import MCPError, MCPResponse

pytestmark = pytest.mark.unit


class _ErroringMcpServer:
    initialized = True

    def __init__(self, *, code: int = -32000, message: str = "mcp backend exploded at /private/db/path") -> None:
        self.code = code
        self.message = message
        self.requests: list[tuple[Any, str | None, dict[str, Any] | None]] = []

    async def initialize(self) -> None:  # pragma: no cover - initialized starts true
        self.initialized = True

    async def handle_http_request(
        self,
        request: Any,
        *,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MCPResponse:
        self.requests.append((request, user_id, metadata))
        return MCPResponse(
            id=getattr(request, "id", None),
            error=MCPError(code=self.code, message=self.message),
        )


class _StatusMcpServer:
    initialized = True

    async def get_status(self) -> dict[str, str]:
        return {"status": "degraded: redis backend exploded at /private/mcp"}


class _SuccessfulMcpServer:
    initialized = True

    def __init__(self) -> None:
        self.metadata: dict[str, Any] | None = None

    async def initialize(self) -> None:  # pragma: no cover - initialized starts true
        self.initialized = True

    async def handle_http_request(
        self,
        request: Any,
        *,
        client_id: str | None = None,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MCPResponse:
        self.metadata = metadata
        return MCPResponse(id=getattr(request, "id", None), result={"ok": True})

    async def handle_http_batch(
        self,
        requests: list[Any],
        *,
        client_id: str | None = None,
        user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[MCPResponse]:
        self.metadata = metadata
        return [MCPResponse(id=getattr(request, "id", None), result={"ok": True}) for request in requests]


class _LoggerStub:
    def __init__(self) -> None:
        self.debug_messages: list[str] = []
        self.error_messages: list[str] = []

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.debug_messages.append(message.format(*args) if args else message)

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.error_messages.append(message.format(*args) if args else message)


class _FailingJwtManager:
    _refresh_tokens: dict[str, Any] = {}

    def rotate_refresh_token(self, refresh_token: str, token_id: str) -> tuple[str, str, str]:
        raise RuntimeError("refresh token backend leaked /private/mcp/tokens.db")


def _request() -> Request:
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/api/v1/mcp",
            "headers": [],
        }
    )


def _auth() -> mcp.McpAuthContext:
    return mcp.McpAuthContext(
        user=TokenData(
            sub="123",
            username="mcp-user",
            roles=["user"],
            permissions=["tools.list", "tools.execute:*"],
        ),
        principal=None,
        api_key_info=None,
        raw_api_key=None,
    )


def _principal() -> AuthPrincipal:
    return AuthPrincipal(
        kind="user",
        user_id=123,
        username="mcp-user",
        roles=["admin"],
        permissions=["system.logs"],
        is_admin=True,
    )


@pytest.mark.asyncio
async def test_list_tools_sanitizes_generic_mcp_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mcp, "get_mcp_server", lambda: _ErroringMcpServer())

    with pytest.raises(HTTPException) as excinfo:
        await mcp.list_tools(
            http_request=_request(),
            module=None,
            catalog=None,
            catalog_id=None,
            catalog_strict=None,
            auth=_auth(),
            _guard=None,
        )

    assert excinfo.value.status_code == 500
    assert excinfo.value.detail == "Failed to list MCP tools"


@pytest.mark.asyncio
async def test_execute_tool_sanitizes_generic_mcp_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mcp, "get_mcp_server", lambda: _ErroringMcpServer())

    with pytest.raises(HTTPException) as excinfo:
        await mcp.execute_tool(
            request=mcp.ToolExecutionRequest(tool_name="demo_tool", arguments={}),
            http_request=_request(),
            auth=_auth(),
            _guard=None,
        )

    assert excinfo.value.status_code == 500
    assert excinfo.value.detail == "MCP tool execution failed"


@pytest.mark.asyncio
async def test_list_modules_sanitizes_generic_mcp_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mcp, "get_mcp_server", lambda: _ErroringMcpServer())

    with pytest.raises(HTTPException) as excinfo:
        await mcp.list_modules(http_request=_request(), auth=_auth(), _guard=None)

    assert excinfo.value.status_code == 500
    assert excinfo.value.detail == "Failed to list MCP modules"


@pytest.mark.asyncio
async def test_get_modules_health_sanitizes_generic_mcp_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mcp, "get_mcp_server", lambda: _ErroringMcpServer())

    with pytest.raises(HTTPException) as excinfo:
        await mcp.get_modules_health(http_request=_request(), principal=_principal(), _guard=None)

    assert excinfo.value.status_code == 500
    assert excinfo.value.detail == "Failed to get MCP module health"


@pytest.mark.asyncio
async def test_list_resources_sanitizes_generic_mcp_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mcp, "get_mcp_server", lambda: _ErroringMcpServer())

    with pytest.raises(HTTPException) as excinfo:
        await mcp.list_resources(http_request=_request(), auth=_auth(), _guard=None)

    assert excinfo.value.status_code == 500
    assert excinfo.value.detail == "Failed to list MCP resources"


@pytest.mark.asyncio
async def test_list_prompts_sanitizes_generic_mcp_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mcp, "get_mcp_server", lambda: _ErroringMcpServer())

    with pytest.raises(HTTPException) as excinfo:
        await mcp.list_prompts(http_request=_request(), auth=_auth(), _guard=None)

    assert excinfo.value.status_code == 500
    assert excinfo.value.detail == "Failed to list MCP prompts"


@pytest.mark.asyncio
async def test_health_check_sanitizes_nonhealthy_status(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mcp, "get_mcp_server", lambda: _StatusMcpServer())

    with pytest.raises(HTTPException) as excinfo:
        await mcp.health_check(_guard=None)

    assert excinfo.value.status_code == 503
    assert excinfo.value.detail == "MCP server is not healthy"


@pytest.mark.asyncio
async def test_request_sanitizes_safe_config_parse_log(monkeypatch: pytest.MonkeyPatch) -> None:
    server = _SuccessfulMcpServer()
    logger = _LoggerStub()
    monkeypatch.setattr(mcp, "get_mcp_server", lambda: server)
    monkeypatch.setattr(mcp, "logger", logger)

    def _raise_leaky_parse_error(_value: str) -> bytes:
        raise ValueError("safe config leaked /private/mcp/config.json")

    monkeypatch.setattr(base64, "b64decode", _raise_leaky_parse_error)

    response = await mcp.mcp_request(
        request=mcp.MCPRequest(method="tools/list", id="req-1"),
        http_request=_request(),
        client_id=None,
        auth=_auth(),
        mcp_session_id=None,
        config="invalid-config",
        response=None,
        _guard=None,
    )

    assert response.result == {"ok": True}
    assert logger.debug_messages == ["Failed to parse safe config"]
    assert "/private/mcp/config.json" not in logger.debug_messages[0]


@pytest.mark.asyncio
async def test_batch_request_sanitizes_safe_config_parse_log(monkeypatch: pytest.MonkeyPatch) -> None:
    server = _SuccessfulMcpServer()
    logger = _LoggerStub()
    monkeypatch.setattr(mcp, "get_mcp_server", lambda: server)
    monkeypatch.setattr(mcp, "logger", logger)

    def _raise_leaky_parse_error(_value: str) -> bytes:
        raise ValueError("batch safe config leaked /private/mcp/config.json")

    monkeypatch.setattr(base64, "b64decode", _raise_leaky_parse_error)

    responses = await mcp.mcp_request_batch(
        requests=[mcp.MCPRequest(method="tools/list", id="req-1")],
        http_request=_request(),
        client_id=None,
        auth=_auth(),
        mcp_session_id=None,
        config="invalid-config",
        response=None,
        _guard=None,
    )

    assert responses[0].result == {"ok": True}
    assert logger.debug_messages == ["Batch failed to parse safe config"]
    assert "/private/mcp/config.json" not in logger.debug_messages[0]


@pytest.mark.asyncio
async def test_refresh_token_sanitizes_rotation_failure_log(monkeypatch: pytest.MonkeyPatch) -> None:
    logger = _LoggerStub()
    monkeypatch.setattr(mcp, "logger", logger)
    monkeypatch.setattr(mcp, "get_jwt_manager", lambda: _FailingJwtManager())

    with pytest.raises(HTTPException) as excinfo:
        await mcp.refresh_token(
            auth_request=mcp.AuthRefreshRequest(refresh_token="refresh-token", token_id="token-id"),
            _guard=None,
        )

    assert excinfo.value.status_code == 500
    assert excinfo.value.detail == "Failed to refresh token"
    assert logger.error_messages == ["Refresh token rotation failed"]
    assert "/private/mcp/tokens.db" not in logger.error_messages[0]
