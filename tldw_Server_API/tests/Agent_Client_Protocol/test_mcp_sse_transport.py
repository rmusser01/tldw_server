"""Tests for MCPSSETransport — SSE-based MCP transport."""
from __future__ import annotations

import asyncio
import json

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from tldw_Server_API.app.core.Agent_Client_Protocol.adapters.mcp_transports.sse import (
    MCPSSETransport,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_transport(
    sse_url: str = "http://localhost:8080/sse",
    post_url: str | None = "http://localhost:8080/messages",
    **kwargs,
) -> MCPSSETransport:
    kwargs.setdefault("allow_private_network", True)
    return MCPSSETransport(sse_url=sse_url, post_url=post_url, **kwargs)


async def _resolve_pending(transport: MCPSSETransport, req_id: str, result: dict):
    """Simulate an SSE response arriving for a pending request."""
    # Give the call a moment to register the pending future
    for _ in range(50):
        if req_id in transport._pending:
            break
        await asyncio.sleep(0.01)
    future = transport._pending.get(req_id)
    if future and not future.done():
        future.set_result(result)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_sse_transport_not_connected_initially():
    """A freshly created transport should not be connected."""
    t = _make_transport()
    assert t.is_connected is False


@pytest.mark.asyncio
async def test_sse_transport_connect_with_explicit_post_url():
    """connect() with explicit post_url skips discovery and performs handshake."""
    t = _make_transport(post_url="http://localhost:8080/messages")
    mock_http = AsyncMock()
    mock_http.post = AsyncMock(return_value=MagicMock(status_code=200))

    with patch.object(t, "_create_http_client", return_value=mock_http):
        # Mock the reader loop so it doesn't actually run
        with patch.object(t, "_sse_reader_loop", new_callable=AsyncMock):
            # We need _json_rpc_call to work for the handshake.
            # Mock it to return a valid initialize response.
            with patch.object(t, "_json_rpc_call", new_callable=AsyncMock) as mock_call:
                mock_call.return_value = {
                    "protocolVersion": "2024-11-05",
                    "serverInfo": {"name": "test"},
                    "capabilities": {},
                }
                with patch.object(t, "_json_rpc_notify", new_callable=AsyncMock) as mock_notify:
                    await t.connect()

    # Verify handshake
    mock_call.assert_awaited_once()
    call_args = mock_call.call_args
    assert call_args[0][0] == "initialize"
    init_params = call_args[0][1]
    assert init_params["protocolVersion"] == "2024-11-05"
    assert init_params["clientInfo"]["name"] == "tldw_acp_harness"

    mock_notify.assert_awaited_once_with("initialized", {})
    assert t.is_connected is True


@pytest.mark.asyncio
async def test_sse_transport_connect_cleans_up_on_handshake_failure():
    """Failed connect handshakes should close the client and clear partial state."""
    t = _make_transport(post_url="http://localhost:8080/messages")
    mock_http = AsyncMock()
    mock_http.post = AsyncMock(return_value=MagicMock(status_code=200))

    with patch.object(t, "_create_http_client", return_value=mock_http):
        with patch.object(t, "_sse_reader_loop", new_callable=AsyncMock):
            with patch.object(
                t,
                "_json_rpc_call",
                new_callable=AsyncMock,
                side_effect=RuntimeError("initialize failed"),
            ):
                with pytest.raises(RuntimeError, match="initialize failed"):
                    await t.connect()

    mock_http.aclose.assert_awaited_once()
    assert t.is_connected is False
    assert t._http_client is None
    assert t._reader_task is None
    assert t._pending == {}


@pytest.mark.asyncio
async def test_sse_transport_list_tools():
    """list_tools() should call tools/list and return the tools array."""
    t = _make_transport()
    t._connected = True

    tools = [
        {"name": "echo", "description": "echoes input"},
        {"name": "add", "description": "adds numbers"},
    ]
    with patch.object(t, "_json_rpc_call", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = {"tools": tools}
        result = await t.list_tools()

    assert result == tools
    mock_call.assert_awaited_once_with("tools/list", {})


@pytest.mark.asyncio
async def test_sse_transport_call_tool():
    """call_tool() should forward name and arguments correctly."""
    t = _make_transport()
    t._connected = True

    expected = {"content": [{"type": "text", "text": "hello"}]}
    with patch.object(t, "_json_rpc_call", new_callable=AsyncMock) as mock_call:
        mock_call.return_value = expected
        result = await t.call_tool("echo", {"message": "hello"})

    assert result == expected
    mock_call.assert_awaited_once_with(
        "tools/call", {"name": "echo", "arguments": {"message": "hello"}}
    )


@pytest.mark.asyncio
async def test_sse_transport_close():
    """close() should cancel reader task, close HTTP client, mark disconnected."""
    t = _make_transport()
    t._connected = True

    mock_task = MagicMock()
    t._reader_task = mock_task

    mock_http = AsyncMock()
    t._http_client = mock_http

    await t.close()

    mock_task.cancel.assert_called_once()
    mock_http.aclose.assert_awaited_once()
    assert t.is_connected is False


@pytest.mark.asyncio
async def test_sse_transport_health_check():
    """health_check() returns connection status."""
    t = _make_transport()
    assert await t.health_check() is False

    t._connected = True
    assert await t.health_check() is True


@pytest.mark.asyncio
async def test_sse_transport_list_tools_not_connected():
    """list_tools() raises RuntimeError when not connected."""
    t = _make_transport()
    with pytest.raises(RuntimeError, match="Not connected"):
        await t.list_tools()


@pytest.mark.asyncio
async def test_sse_transport_call_tool_not_connected():
    """call_tool() raises RuntimeError when not connected."""
    t = _make_transport()
    with pytest.raises(RuntimeError, match="Not connected"):
        await t.call_tool("echo", {})


@pytest.mark.asyncio
async def test_sse_transport_json_rpc_call():
    """_json_rpc_call posts JSON-RPC and resolves from pending future."""
    t = _make_transport()
    t._post_url = "http://localhost:8080/messages"
    mock_http = AsyncMock()
    mock_http.post = AsyncMock(return_value=MagicMock(status_code=200))
    t._http_client = mock_http

    expected_result = {"tools": []}

    async def resolve_soon():
        await _resolve_pending(t, "1", expected_result)

    # Run the call and the resolver concurrently
    resolve_task = asyncio.create_task(resolve_soon())
    result = await t._json_rpc_call("tools/list", {})
    await resolve_task

    assert result == expected_result
    # Verify the POST was made with correct JSON-RPC payload
    mock_http.post.assert_awaited_once()
    posted_json = mock_http.post.call_args[1]["json"]
    assert posted_json["jsonrpc"] == "2.0"
    assert posted_json["id"] == "1"
    assert posted_json["method"] == "tools/list"
    assert posted_json["params"] == {}
    assert "1" not in t._pending


@pytest.mark.asyncio
async def test_sse_transport_json_rpc_call_requires_post_url():
    """_json_rpc_call should require that post_url has been initialized."""
    t = _make_transport(post_url=None)
    mock_http = AsyncMock()
    mock_http.post = AsyncMock(return_value=MagicMock(status_code=200))
    t._http_client = mock_http

    with pytest.raises(RuntimeError, match="post_url must be set"):
        await t._json_rpc_call("tools/list", {})


@pytest.mark.asyncio
async def test_sse_transport_json_rpc_call_cleans_pending_on_timeout():
    """Timed-out JSON-RPC calls should not leak pending futures."""
    t = _make_transport(timeout_sec=0)
    t._post_url = "http://localhost:8080/messages"
    mock_http = AsyncMock()
    mock_http.post = AsyncMock(return_value=MagicMock(status_code=200))
    t._http_client = mock_http

    with pytest.raises(asyncio.TimeoutError):
        await t._json_rpc_call("tools/list", {})

    assert t._pending == {}


@pytest.mark.asyncio
async def test_sse_transport_json_rpc_call_cleans_pending_on_post_error():
    """POST failures should remove any pending future for the request."""
    t = _make_transport()
    t._post_url = "http://localhost:8080/messages"
    mock_http = AsyncMock()
    mock_http.post = AsyncMock(side_effect=RuntimeError("post failed"))
    t._http_client = mock_http

    with pytest.raises(RuntimeError, match="post failed"):
        await t._json_rpc_call("tools/list", {})

    assert t._pending == {}


@pytest.mark.asyncio
async def test_sse_transport_json_rpc_notify():
    """_json_rpc_notify posts JSON-RPC notification (no id field)."""
    t = _make_transport()
    t._post_url = "http://localhost:8080/messages"
    mock_http = AsyncMock()
    mock_http.post = AsyncMock(return_value=MagicMock(status_code=200))
    t._http_client = mock_http

    await t._json_rpc_notify("initialized", {})

    mock_http.post.assert_awaited_once()
    posted_json = mock_http.post.call_args[1]["json"]
    assert posted_json["jsonrpc"] == "2.0"
    assert posted_json["method"] == "initialized"
    assert posted_json["params"] == {}
    assert "id" not in posted_json


@pytest.mark.asyncio
async def test_sse_transport_json_rpc_notify_requires_post_url():
    """_json_rpc_notify should require that post_url has been initialized."""
    t = _make_transport(post_url=None)
    mock_http = AsyncMock()
    mock_http.post = AsyncMock(return_value=MagicMock(status_code=200))
    t._http_client = mock_http

    with pytest.raises(RuntimeError, match="post_url must be set"):
        await t._json_rpc_notify("initialized", {})


@pytest.mark.asyncio
async def test_sse_transport_parse_sse_events():
    """_parse_sse_line correctly accumulates and yields SSE events."""
    t = _make_transport()

    # Simulate SSE lines
    lines = [
        "event: endpoint",
        "data: /messages",
        "",  # blank = event boundary
        "event: message",
        'data: {"jsonrpc": "2.0", "id": "1", "result": {"ok": true}}',
        "",
    ]

    events = []
    for line in lines:
        evt = t._parse_sse_line(line)
        if evt is not None:
            events.append(evt)

    assert len(events) == 2
    assert events[0] == ("endpoint", "/messages")
    assert events[1] == ("message", '{"jsonrpc": "2.0", "id": "1", "result": {"ok": true}}')


def test_sse_transport_parse_sse_multiline_data_defaults_to_message():
    """Data-only SSE frames should default to 'message' and preserve all data lines."""
    t = _make_transport()

    assert t._parse_sse_line("data: first line") is None
    assert t._parse_sse_line("data: second line") is None
    assert t._parse_sse_line("") == ("message", "first line\nsecond line")


@pytest.mark.asyncio
async def test_sse_transport_reader_loop_tears_down_on_stream_end():
    """Reader loop should mark the transport disconnected and fail pending RPCs on EOF."""
    t = _make_transport()
    t._connected = True
    pending = asyncio.get_running_loop().create_future()
    t._pending["1"] = pending

    async def empty_lines():
        if False:
            yield ""

    mock_response = AsyncMock()
    mock_response.aiter_lines = empty_lines
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=False)

    mock_http = AsyncMock()
    mock_http.stream = MagicMock(return_value=mock_response)
    t._http_client = mock_http

    await t._sse_reader_loop()

    assert t.is_connected is False
    assert t._pending == {}
    assert pending.done() is True
    assert isinstance(pending.exception(), RuntimeError)


@pytest.mark.asyncio
async def test_sse_transport_discover_post_url():
    """_discover_post_url resolves relative URL against sse_url base."""
    t = MCPSSETransport(sse_url="http://localhost:8080/sse", allow_private_network=True)

    # Mock the HTTP client stream to return an endpoint event
    async def mock_stream_lines():
        yield "event: endpoint"
        yield "data: /messages"
        yield ""

    mock_response = AsyncMock()
    mock_response.aiter_lines = mock_stream_lines
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=False)

    mock_http = AsyncMock()
    mock_http.stream = MagicMock(return_value=mock_response)
    t._http_client = mock_http

    url = await t._discover_post_url()
    assert url == "http://localhost:8080/messages"


@pytest.mark.asyncio
async def test_sse_transport_rejects_loopback_endpoint_before_client_creation():
    t = MCPSSETransport(
        sse_url="http://127.0.0.1:8080/sse",
        post_url="http://127.0.0.1:8080/messages",
    )

    with patch.object(t, "_create_http_client") as create_client:
        with pytest.raises(ValueError):
            await t.connect()

    create_client.assert_not_called()


@pytest.mark.asyncio
async def test_sse_transport_rejects_cross_origin_discovered_post_url():
    t = MCPSSETransport(sse_url="https://mcp.example.com/sse", allow_private_network=True)

    async def mock_stream_lines():
        yield "event: endpoint"
        yield "data: https://evil.example.com/messages"
        yield ""

    mock_response = AsyncMock()
    mock_response.aiter_lines = mock_stream_lines
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=False)

    mock_http = AsyncMock()
    mock_http.stream = MagicMock(return_value=mock_response)
    t._http_client = mock_http

    with pytest.raises(ValueError, match="same origin"):
        await t._discover_post_url()


@pytest.mark.asyncio
async def test_sse_transport_close_idempotent():
    """close() should be safe to call even when nothing is initialized."""
    t = _make_transport()
    await t.close()  # Should not raise
    assert t.is_connected is False
