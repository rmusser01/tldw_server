"""Tests for MCPStreamableHTTPTransport — streamable HTTP MCP transport."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from tldw_Server_API.app.core.Agent_Client_Protocol.adapters.mcp_transports.streamable_http import (
    MCPStreamableHTTPTransport,
)
from tldw_Server_API.app.core.exceptions import EgressPolicyError
from tldw_Server_API.app.core.http_client import create_async_client

pytestmark = pytest.mark.unit


# ---- Helpers ----

_PUBLIC_ENDPOINT = "http://93.184.216.34/mcp"


def _mock_client(handler):
    """A core.http_client client whose socket layer is an httpx.MockTransport."""
    return create_async_client(transport=httpx.MockTransport(handler))


def _rpc_server(replies: list[dict], *, sse: bool = False, seen: list | None = None):
    """MockTransport handler answering each JSON-RPC request with the next reply.

    A reply is the ``result`` or ``error`` part of the JSON-RPC message. Notifications
    get an empty 202. Every request is appended to ``seen``.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        body = json.loads(request.content)
        if "id" not in body:
            return httpx.Response(202, request=request)
        message = {"jsonrpc": "2.0", "id": body["id"], **replies.pop(0)}
        if sse:
            return httpx.Response(
                200,
                request=request,
                headers={"content-type": "text/event-stream"},
                content=f"data: {json.dumps(message)}\n\n".encode(),
            )
        return httpx.Response(200, request=request, json=message)

    return handler


# ---- Fixtures ----


@pytest.fixture
def transport():
    return MCPStreamableHTTPTransport(
        endpoint=_PUBLIC_ENDPOINT,
        headers={"Authorization": "Bearer test"},
        timeout_sec=15,
    )


# ---- Tests ----


def test_streamable_http_not_connected_initially(transport):
    """A freshly created transport should not be connected."""
    assert transport.is_connected is False


@pytest.mark.asyncio
async def test_streamable_http_connect_handshake(transport, mcp_http_egress):
    """connect() should POST initialize then send initialized notification."""
    init_result = {
        "protocolVersion": "2024-11-05",
        "serverInfo": {"name": "test-server"},
        "capabilities": {},
    }
    seen: list[httpx.Request] = []
    client = _mock_client(_rpc_server([{"result": init_result}], seen=seen))
    with patch.object(transport, "_create_http_client", return_value=client):
        await transport.connect()

    assert transport.is_connected is True
    assert len(seen) == 2
    assert all(str(request.url) == _PUBLIC_ENDPOINT for request in seen)
    assert all(request.headers["authorization"] == "Bearer test" for request in seen)

    # Verify initialize call
    payload = json.loads(seen[0].content)
    assert payload["method"] == "initialize"
    assert payload["params"]["protocolVersion"] == "2024-11-05"
    assert payload["params"]["clientInfo"]["name"] == "tldw_acp_harness"
    assert "id" in payload

    # Verify initialized notification
    notify_payload = json.loads(seen[1].content)
    assert notify_payload["method"] == "initialized"
    assert "id" not in notify_payload
    await transport.close()


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
async def test_streamable_http_list_tools_json_mode(transport, mcp_http_egress):
    """list_tools() should parse tools from a JSON response."""
    tools = [
        {"name": "echo", "description": "echoes input"},
        {"name": "add", "description": "adds numbers"},
    ]
    seen: list[httpx.Request] = []
    transport._http_client = _mock_client(_rpc_server([{"result": {"tools": tools}}], seen=seen))
    transport._connected = True
    transport._next_id = 1

    result = await transport.list_tools()

    assert result == tools
    assert len(seen) == 1
    await transport.close()


@pytest.mark.asyncio
async def test_streamable_http_call_tool_sse_mode(transport, mcp_http_egress):
    """call_tool() should parse result from an SSE response."""
    tool_result = {"content": [{"type": "text", "text": "hello"}]}
    seen: list[httpx.Request] = []
    transport._http_client = _mock_client(_rpc_server([{"result": tool_result}], sse=True, seen=seen))
    transport._connected = True
    transport._next_id = 1

    result = await transport.call_tool("echo", {"message": "hello"})

    assert result == tool_result
    # Verify the POST payload
    payload = json.loads(seen[0].content)
    assert payload["method"] == "tools/call"
    assert payload["params"]["name"] == "echo"
    assert payload["params"]["arguments"] == {"message": "hello"}
    await transport.close()


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
async def test_streamable_http_error_response(transport, mcp_http_egress):
    """A JSON-RPC error in the response should raise RuntimeError."""
    transport._http_client = _mock_client(
        _rpc_server([{"error": {"code": -32601, "message": "Method not found"}}])
    )
    transport._connected = True
    transport._next_id = 1

    with pytest.raises(RuntimeError, match="Method not found"):
        await transport.list_tools()
    await transport.close()


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


# ---- Egress: every request goes through core.http_client (TASK-13950.11) ----


def _counting_server(requested: list[str]):
    """A server that records each URL and would complete the handshake if reached."""
    replies = [{"result": {"protocolVersion": "2024-11-05", "capabilities": {}}}]
    inner = _rpc_server(replies)

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        return inner(request)

    return handler


def test_streamable_http_client_ignores_env_proxies():
    """The HTTP client comes from core.http_client, which does not read HTTP(S)_PROXY."""
    transport = MCPStreamableHTTPTransport(endpoint=_PUBLIC_ENDPOINT)
    client = transport._create_http_client()

    assert client.trust_env is False


@pytest.mark.asyncio
async def test_streamable_http_refuses_dns_rebinding_to_loopback(mcp_dns_rebinding):
    """validate_mcp_http_url saw a public address; the connect-time answer is loopback."""
    transport = MCPStreamableHTTPTransport(endpoint="http://mcp.rebind.test/mcp")
    requested: list[str] = []

    with patch.object(transport, "_create_http_client", return_value=_mock_client(_counting_server(requested))):
        with pytest.raises(EgressPolicyError):
            await transport.connect()

    assert requested == []
    assert transport.is_connected is False


@pytest.mark.asyncio
async def test_streamable_http_private_opt_in_still_refuses_metadata(mcp_http_egress):
    """allow_private_network admits the configured private origin, never cloud metadata."""
    transport = MCPStreamableHTTPTransport(
        endpoint="http://169.254.169.254/latest/mcp",
        allow_private_network=True,
    )
    requested: list[str] = []

    with patch.object(transport, "_create_http_client", return_value=_mock_client(_counting_server(requested))):
        with pytest.raises(EgressPolicyError):
            await transport.connect()

    assert requested == []


@pytest.mark.asyncio
async def test_streamable_http_private_opt_in_reaches_loopback_server(mcp_http_egress):
    """A local MCP server on a non-standard port still works under the opt-in."""
    endpoint = "http://127.0.0.1:3001/mcp"
    transport = MCPStreamableHTTPTransport(endpoint=endpoint, allow_private_network=True)
    requested: list[str] = []

    with patch.object(transport, "_create_http_client", return_value=_mock_client(_counting_server(requested))):
        await transport.connect()

    assert transport.is_connected is True
    assert requested == [endpoint, endpoint]
    await transport.close()


@pytest.mark.asyncio
async def test_streamable_http_does_not_follow_redirect_to_private_address(transport, mcp_http_egress):
    """A redirect is an HTTP error, as before; the private target is never requested."""
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        return httpx.Response(307, request=request, headers={"Location": "http://169.254.169.254/mcp"})

    with patch.object(transport, "_create_http_client", return_value=_mock_client(handler)):
        with pytest.raises(httpx.HTTPStatusError):
            await transport.connect()

    assert requested == [_PUBLIC_ENDPOINT]
    assert transport.is_connected is False
