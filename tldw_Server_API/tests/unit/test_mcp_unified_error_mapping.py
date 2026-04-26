"""Error mapping tests for MCP unified HTTP endpoints."""
from __future__ import annotations

from typing import Any

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
