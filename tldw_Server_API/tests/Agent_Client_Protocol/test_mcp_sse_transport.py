"""Tests for MCPSSETransport — SSE-based MCP transport."""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from tldw_Server_API.app.core.Agent_Client_Protocol.adapters.mcp_transports.sse import (
    MCPSSETransport,
)
from tldw_Server_API.app.core.exceptions import EgressPolicyError, NetworkError
from tldw_Server_API.app.core.http_client import create_async_client

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SSE_URL = "http://93.184.216.34/sse"
_POST_URL = "http://93.184.216.34/messages"


def _make_transport(
    sse_url: str = _SSE_URL,
    post_url: str | None = _POST_URL,
    **kwargs,
) -> MCPSSETransport:
    return MCPSSETransport(sse_url=sse_url, post_url=post_url, **kwargs)


def _mock_client(handler):
    """A core.http_client client whose socket layer is an httpx.MockTransport."""
    return create_async_client(transport=httpx.MockTransport(handler))


def _sse_response(request: httpx.Request, body: bytes) -> httpx.Response:
    return httpx.Response(200, request=request, headers={"content-type": "text/event-stream"}, content=body)


def _accepting(seen: list[httpx.Request]):
    """Handler that records each request and answers 202 Accepted."""

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(202, request=request)

    return handler


class _ReplyStream(httpx.AsyncByteStream):
    """An SSE body that sends ``first`` and then whatever the server queues."""

    def __init__(self, first: bytes, queue: asyncio.Queue) -> None:
        self._first = first
        self._queue = queue

    async def __aiter__(self):
        yield self._first
        while True:
            yield await self._queue.get()


class _MCPSSEServer:
    """In-memory MCP-over-SSE server behind an httpx.MockTransport.

    Every GET opens a stream that starts with the ``endpoint`` event; JSON-RPC
    replies to POSTed requests are queued onto the streams, as multi-line
    ``message`` events.
    """

    def __init__(self, tools: list[dict]) -> None:
        self.requests: list[httpx.Request] = []
        self._tools = tools
        self._replies: asyncio.Queue = asyncio.Queue()

    async def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.method == "GET":
            endpoint = b"event: endpoint\ndata: /messages?session=abc\n\n"
            return httpx.Response(
                200,
                request=request,
                headers={"content-type": "text/event-stream"},
                stream=_ReplyStream(endpoint, self._replies),
            )
        body = json.loads(request.content)
        if "id" in body:
            result = {"tools": self._tools} if body["method"] == "tools/list" else {"capabilities": {}}
            message = json.dumps({"jsonrpc": "2.0", "id": body["id"], "result": result}, indent=1)
            data = "data: " + message.replace("\n", "\ndata: ")
            self._replies.put_nowait(f"event: message\n{data}\n\n".encode())
        return httpx.Response(202, request=request)


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
    t = _make_transport(post_url=_POST_URL)
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
    t = _make_transport(post_url=_POST_URL)
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
async def test_sse_transport_json_rpc_call(mcp_http_egress):
    """_json_rpc_call posts JSON-RPC and resolves from pending future."""
    t = _make_transport()
    seen: list[httpx.Request] = []
    t._http_client = _mock_client(_accepting(seen))

    expected_result = {"tools": []}

    async def resolve_soon():
        await _resolve_pending(t, "1", expected_result)

    # Run the call and the resolver concurrently
    resolve_task = asyncio.create_task(resolve_soon())
    result = await t._json_rpc_call("tools/list", {})
    await resolve_task

    assert result == expected_result
    # Verify the POST was made with correct JSON-RPC payload
    assert [str(request.url) for request in seen] == [_POST_URL]
    posted_json = json.loads(seen[0].content)
    assert posted_json["jsonrpc"] == "2.0"
    assert posted_json["id"] == "1"
    assert posted_json["method"] == "tools/list"
    assert posted_json["params"] == {}
    assert "1" not in t._pending
    await t.close()


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
async def test_sse_transport_json_rpc_call_cleans_pending_on_timeout(mcp_http_egress):
    """Timed-out JSON-RPC calls should not leak pending futures."""
    t = _make_transport(timeout_sec=0)
    t._http_client = _mock_client(_accepting([]))

    with pytest.raises(asyncio.TimeoutError):
        await t._json_rpc_call("tools/list", {})

    assert t._pending == {}


@pytest.mark.asyncio
async def test_sse_transport_json_rpc_call_cleans_pending_on_post_error(mcp_http_egress):
    """POST failures should remove any pending future for the request."""
    t = _make_transport()
    t._http_client = _mock_client(lambda request: httpx.Response(500, request=request))

    with pytest.raises(httpx.HTTPStatusError):
        await t._json_rpc_call("tools/list", {})

    assert t._pending == {}


@pytest.mark.asyncio
async def test_sse_transport_json_rpc_notify(mcp_http_egress):
    """_json_rpc_notify posts JSON-RPC notification (no id field)."""
    t = _make_transport()
    seen: list[httpx.Request] = []
    t._http_client = _mock_client(_accepting(seen))

    await t._json_rpc_notify("initialized", {})

    assert [str(request.url) for request in seen] == [_POST_URL]
    posted_json = json.loads(seen[0].content)
    assert posted_json["jsonrpc"] == "2.0"
    assert posted_json["method"] == "initialized"
    assert posted_json["params"] == {}
    assert "id" not in posted_json
    await t.close()


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
async def test_sse_transport_reader_routes_message_events(mcp_http_egress):
    """Message events (including data-only frames) resolve pending RPCs; other events are ignored."""
    t = _make_transport()
    t._connected = True
    loop = asyncio.get_running_loop()
    first, second = loop.create_future(), loop.create_future()
    t._pending.update({"1": first, "2": second})
    body = (
        b"event: endpoint\ndata: /messages\n\n"
        b'data: {"jsonrpc": "2.0", "id": "1",\ndata: "result": {"ok": true}}\n\n'
        b"event: message\n"
        b'data: {"jsonrpc": "2.0", "id": "2", "result": {"n": 2}}\n\n'
    )
    t._http_client = _mock_client(lambda request: _sse_response(request, body))

    await t._sse_reader_loop()

    assert first.result() == {"ok": True}
    assert second.result() == {"n": 2}


@pytest.mark.asyncio
async def test_sse_transport_reader_loop_tears_down_on_stream_end(mcp_http_egress):
    """Reader loop should mark the transport disconnected and fail pending RPCs on EOF."""
    t = _make_transport()
    t._connected = True
    pending = asyncio.get_running_loop().create_future()
    t._pending["1"] = pending
    t._http_client = _mock_client(lambda request: _sse_response(request, b""))

    await t._sse_reader_loop()

    assert t.is_connected is False
    assert t._pending == {}
    assert pending.done() is True
    assert isinstance(pending.exception(), RuntimeError)


@pytest.mark.asyncio
async def test_sse_transport_discover_post_url(mcp_http_egress):
    """_discover_post_url resolves relative URL against sse_url base."""
    t = MCPSSETransport(sse_url=_SSE_URL)
    t._http_client = _mock_client(
        lambda request: _sse_response(request, b"event: endpoint\ndata: /messages\n\n")
    )

    url = await t._discover_post_url()
    assert url == _POST_URL
    await t.close()


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
async def test_sse_transport_rejects_cross_origin_discovered_post_url(mcp_http_egress):
    t = MCPSSETransport(sse_url=_SSE_URL)
    t._http_client = _mock_client(
        lambda request: _sse_response(request, b"event: endpoint\ndata: http://93.184.216.35/messages\n\n")
    )

    with pytest.raises(ValueError, match="same origin"):
        await t._discover_post_url()
    await t.close()


@pytest.mark.asyncio
async def test_sse_transport_close_idempotent():
    """close() should be safe to call even when nothing is initialized."""
    t = _make_transport()
    await t.close()  # Should not raise
    assert t.is_connected is False


# ---------------------------------------------------------------------------
# Egress: every request goes through core.http_client (TASK-13950.11)
# ---------------------------------------------------------------------------


def _recording_server(requested: list[str]):
    """A server that records each URL and would hand out an endpoint if reached."""

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        if request.method == "GET":
            return _sse_response(request, b"event: endpoint\ndata: /messages\n\n")
        return httpx.Response(202, request=request)

    return handler


@pytest.mark.asyncio
async def test_sse_transport_end_to_end_over_http_client(mcp_http_egress):
    """Discovery, handshake and an RPC round-trip work against a public server."""
    server = _MCPSSEServer(tools=[{"name": "echo"}])
    t = MCPSSETransport(sse_url=_SSE_URL, headers={"Authorization": "Bearer tok"})

    with patch.object(t, "_create_http_client", return_value=_mock_client(server.handler)):
        await t.connect()
    try:
        assert t.is_connected is True
        assert t._post_url == "http://93.184.216.34/messages?session=abc"
        assert await t.list_tools() == [{"name": "echo"}]
    finally:
        await t.close()

    gets = [request for request in server.requests if request.method == "GET"]
    posts = [request for request in server.requests if request.method == "POST"]
    assert [str(request.url) for request in gets] == [_SSE_URL, _SSE_URL]
    assert all(request.headers["accept"] == "text/event-stream" for request in gets)
    assert all(str(request.url) == t._post_url for request in posts)
    assert [json.loads(request.content)["method"] for request in posts] == ["initialize", "initialized", "tools/list"]
    assert all(request.headers["authorization"] == "Bearer tok" for request in server.requests)


def test_sse_transport_client_ignores_env_proxies():
    """The HTTP client comes from core.http_client, which does not read HTTP(S)_PROXY."""
    client = MCPSSETransport(sse_url=_SSE_URL)._create_http_client()

    assert client.trust_env is False


@pytest.mark.asyncio
async def test_sse_transport_refuses_dns_rebinding_to_loopback(mcp_dns_rebinding):
    """validate_mcp_http_url saw a public address; the connect-time answer is loopback."""
    t = MCPSSETransport(sse_url="http://mcp.rebind.test/sse", timeout_sec=1)
    requested: list[str] = []

    with patch.object(t, "_create_http_client", return_value=_mock_client(_recording_server(requested))):
        with pytest.raises(EgressPolicyError):
            await t.connect()

    assert requested == []
    assert t.is_connected is False


@pytest.mark.asyncio
async def test_sse_transport_private_opt_in_still_refuses_metadata(mcp_http_egress):
    """allow_private_network admits the configured private origin, never cloud metadata."""
    t = MCPSSETransport(
        sse_url="http://169.254.169.254/sse",
        allow_private_network=True,
        timeout_sec=1,
    )
    requested: list[str] = []

    with patch.object(t, "_create_http_client", return_value=_mock_client(_recording_server(requested))):
        with pytest.raises(EgressPolicyError):
            await t.connect()

    assert requested == []


@pytest.mark.asyncio
async def test_sse_transport_private_opt_in_reaches_loopback_server(mcp_http_egress):
    """A local MCP server on a non-standard port still works under the opt-in."""
    t = MCPSSETransport(sse_url="http://127.0.0.1:3001/sse", allow_private_network=True)
    requested: list[str] = []
    t._http_client = _mock_client(_recording_server(requested))

    t._post_url = await t._discover_post_url()
    await t._json_rpc_notify("initialized", {})

    assert t._post_url == "http://127.0.0.1:3001/messages"
    assert requested == ["http://127.0.0.1:3001/sse", "http://127.0.0.1:3001/messages"]
    await t.close()


@pytest.mark.asyncio
async def test_sse_transport_refuses_redirect_to_private_address_at_the_hop(mcp_http_egress):
    """A redirect from the SSE URL into link-local metadata is refused before it is requested."""
    t = MCPSSETransport(sse_url=_SSE_URL)
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        return httpx.Response(302, request=request, headers={"Location": "http://169.254.169.254/sse"})

    t._http_client = _mock_client(handler)

    with pytest.raises(EgressPolicyError):
        await t._discover_post_url()

    assert requested == [_SSE_URL]
    await t.close()


@pytest.mark.asyncio
async def test_sse_transport_stream_timeout_bounds_a_silent_server(mcp_http_egress):
    """timeout_sec bounds the wait for SSE bytes, enforced by http_client's stream timeouts."""

    class _SilentStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            await asyncio.Event().wait()
            yield b""

    t = MCPSSETransport(sse_url=_SSE_URL, timeout_sec=0.2)
    t._http_client = _mock_client(
        lambda request: httpx.Response(200, request=request, stream=_SilentStream())
    )

    with pytest.raises(NetworkError, match="StreamTimeout"):
        await asyncio.wait_for(t._discover_post_url(), timeout=2)
    await t.close()
