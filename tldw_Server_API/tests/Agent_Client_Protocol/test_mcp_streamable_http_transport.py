"""Tests for MCPStreamableHTTPTransport — streamable HTTP MCP transport."""
from __future__ import annotations

import json

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from tldw_Server_API.app.core.Agent_Client_Protocol.adapters.mcp_transports.streamable_http import (
    MCPStreamableHTTPTransport,
)

pytestmark = pytest.mark.unit


# ---- Helpers ----


def make_json_response(result: dict, request_id: str = "1") -> MagicMock:
    """Create a fake JSON response object."""
    resp = MagicMock()
    resp.status_code = 200
    resp.headers = {"content-type": "application/json"}
    resp.json.return_value = {"jsonrpc": "2.0", "id": request_id, "result": result}
    resp.raise_for_status = MagicMock()
    return resp


def make_sse_response(result: dict, request_id: str = "1") -> MagicMock:
    """Create a fake SSE response object."""
    resp = MagicMock()
    resp.status_code = 200
    resp.headers = {"content-type": "text/event-stream"}
    resp.text = (
        f'data: {json.dumps({"jsonrpc": "2.0", "id": request_id, "result": result})}\n\n'
    )
    resp.raise_for_status = MagicMock()
    return resp


def make_json_error_response(
    message: str = "Method not found",
    request_id: str = "1",
) -> MagicMock:
    """Create a fake JSON-RPC error response."""
    resp = MagicMock()
    resp.status_code = 200
    resp.headers = {"content-type": "application/json"}
    resp.json.return_value = {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": -32601, "message": message},
    }
    resp.raise_for_status = MagicMock()
    return resp


def make_notify_response() -> MagicMock:
    """Create a fake notification response."""
    resp = MagicMock()
    resp.status_code = 200
    resp.headers = {"content-type": "application/json"}
    resp.raise_for_status = MagicMock()
    return resp


# ---- Fixtures ----


@pytest.fixture
def transport():
    return MCPStreamableHTTPTransport(
        endpoint="http://localhost:8080/mcp",
        headers={"Authorization": "Bearer test"},
        timeout_sec=15,
        allow_private_network=True,
    )


# ---- Tests ----


def test_streamable_http_not_connected_initially(transport):
    """A freshly created transport should not be connected."""
    assert transport.is_connected is False


@pytest.mark.asyncio
async def test_streamable_http_connect_handshake(transport):
    """connect() should POST initialize then send initialized notification."""
    init_result = {
        "protocolVersion": "2024-11-05",
        "serverInfo": {"name": "test-server"},
        "capabilities": {},
    }
    # First call: initialize (JSON-RPC call with id "1")
    # Second call: initialized notification (no id)
    mock_post = AsyncMock(
        side_effect=[
            make_json_response(init_result, request_id="1"),
            make_notify_response(),
        ]
    )

    instance = MagicMock()
    instance.post = mock_post
    instance.aclose = AsyncMock()
    with patch.object(transport, "_create_http_client", return_value=instance):
        await transport.connect()

    assert transport.is_connected is True
    assert mock_post.await_count == 2

    # Verify initialize call
    first_call = mock_post.call_args_list[0]
    payload = first_call.kwargs.get("json") or first_call[1].get("json")
    assert payload["method"] == "initialize"
    assert payload["params"]["protocolVersion"] == "2024-11-05"
    assert payload["params"]["clientInfo"]["name"] == "tldw_acp_harness"
    assert "id" in payload

    # Verify initialized notification
    second_call = mock_post.call_args_list[1]
    notify_payload = second_call.kwargs.get("json") or second_call[1].get("json")
    assert notify_payload["method"] == "initialized"
    assert "id" not in notify_payload


@pytest.mark.asyncio
async def test_streamable_http_connect_cleans_up_on_handshake_failure(transport):
    """Handshake failures should close and clear the HTTP client."""
    mock_client = MagicMock()
    mock_client.aclose = AsyncMock()

    with patch.object(transport, "_create_http_client", return_value=mock_client):
        with patch.object(
            transport,
            "_json_rpc_call",
            new_callable=AsyncMock,
            side_effect=RuntimeError("initialize failed"),
        ):
            with pytest.raises(RuntimeError, match="initialize failed"):
                await transport.connect()

    mock_client.aclose.assert_awaited_once()
    assert transport.is_connected is False
    assert transport._http_client is None


@pytest.mark.asyncio
async def test_streamable_http_list_tools_json_mode(transport):
    """list_tools() should parse tools from a JSON response."""
    tools = [
        {"name": "echo", "description": "echoes input"},
        {"name": "add", "description": "adds numbers"},
    ]
    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=make_json_response({"tools": tools}, request_id="1"))
    mock_client.aclose = AsyncMock()
    transport._http_client = mock_client
    transport._connected = True
    transport._next_id = 1

    result = await transport.list_tools()

    assert result == tools
    mock_client.post.assert_awaited_once()


@pytest.mark.asyncio
async def test_streamable_http_call_tool_sse_mode(transport):
    """call_tool() should parse result from an SSE response."""
    tool_result = {"content": [{"type": "text", "text": "hello"}]}
    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=make_sse_response(tool_result, request_id="1"))
    mock_client.aclose = AsyncMock()
    transport._http_client = mock_client
    transport._connected = True
    transport._next_id = 1

    result = await transport.call_tool("echo", {"message": "hello"})

    assert result == tool_result
    # Verify the POST payload
    call_args = mock_client.post.call_args
    payload = call_args.kwargs.get("json") or call_args[1].get("json")
    assert payload["method"] == "tools/call"
    assert payload["params"]["name"] == "echo"
    assert payload["params"]["arguments"] == {"message": "hello"}


@pytest.mark.asyncio
async def test_streamable_http_close(transport):
    """close() should call aclose on the HTTP client and mark disconnected."""
    mock_client = MagicMock()
    mock_client.aclose = AsyncMock()
    transport._http_client = mock_client
    transport._connected = True

    await transport.close()

    mock_client.aclose.assert_awaited_once()
    assert transport.is_connected is False


@pytest.mark.asyncio
async def test_streamable_http_health_check(transport):
    """health_check() returns True when connected, False otherwise."""
    assert await transport.health_check() is False

    transport._connected = True
    assert await transport.health_check() is True


@pytest.mark.asyncio
async def test_streamable_http_error_response(transport):
    """A JSON-RPC error in the response should raise RuntimeError."""
    mock_client = MagicMock()
    mock_client.post = AsyncMock(
        return_value=make_json_error_response("Method not found", request_id="1")
    )
    mock_client.aclose = AsyncMock()
    transport._http_client = mock_client
    transport._connected = True
    transport._next_id = 1

    with pytest.raises(RuntimeError, match="Method not found"):
        await transport.list_tools()


@pytest.mark.asyncio
async def test_streamable_http_list_tools_not_connected(transport):
    """list_tools() raises RuntimeError when not connected."""
    with pytest.raises(RuntimeError, match="Not connected"):
        await transport.list_tools()


@pytest.mark.asyncio
async def test_streamable_http_call_tool_not_connected(transport):
    """call_tool() raises RuntimeError when not connected."""
    with pytest.raises(RuntimeError, match="Not connected"):
        await transport.call_tool("echo", {})


@pytest.mark.asyncio
async def test_streamable_http_rejects_loopback_endpoint_before_client_creation():
    transport = MCPStreamableHTTPTransport(endpoint="http://127.0.0.1:8080/mcp")

    with patch.object(transport, "_create_http_client") as create_client:
        with pytest.raises(ValueError):
            await transport.connect()

    create_client.assert_not_called()
