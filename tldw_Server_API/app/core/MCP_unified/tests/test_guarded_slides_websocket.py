"""Real-stack coverage for the guarded MCP WebSocket transport."""

from __future__ import annotations

import asyncio
import json
import socket
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest
import uvicorn
import websockets
from uvicorn.protocols.websockets.websockets_sansio_impl import WebSocketsSansIOProtocol
from websockets.exceptions import ConnectionClosed

from tldw_Server_API.app.core.Security.standalone_html_request_guard import (
    SCANNER_CHUNK_BYTES,
    ShallowStandaloneFieldScanner,
)


class _EchoApplication:
    """Capture ASGI scopes and echo complete text messages byte-for-byte."""

    def __init__(self) -> None:
        self.scope: dict[str, Any] | None = None
        self.messages: list[str] = []
        self.disconnect: dict[str, Any] | None = None

    async def __call__(self, scope, receive, send) -> None:
        self.scope = scope
        assert (await receive())["type"] == "websocket.connect"
        await send({"type": "websocket.accept"})
        while True:
            event = await receive()
            if event["type"] == "websocket.disconnect":
                self.disconnect = event
                return
            text = event.get("text")
            if isinstance(text, str):
                self.messages.append(text)
                await send({"type": "websocket.send", "text": text})


@asynccontextmanager
async def _serve_websocket(
    app: _EchoApplication,
    protocol_class: type[WebSocketsSansIOProtocol],
) -> AsyncIterator[str]:
    """Run the pinned Uvicorn stack on an ephemeral pre-bound socket."""

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(128)
    listener.setblocking(False)
    port = int(listener.getsockname()[1])
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        ws=protocol_class,
        ws_per_message_deflate=False,
        lifespan="off",
        access_log=False,
        log_level="warning",
    )
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve(sockets=[listener]))
    try:
        for _ in range(200):
            if server.started:
                break
            if task.done():
                await task
            await asyncio.sleep(0.01)
        assert server.started
        yield f"ws://127.0.0.1:{port}/api/v1/mcp/ws"
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, timeout=5)


def _guarded_transport_module():
    from tldw_Server_API.app.core.MCP_unified.transport import guarded_slides_websocket

    return guarded_slides_websocket


@pytest.mark.asyncio
async def test_guarded_websocket_replays_large_fragmented_message_exactly_without_compression(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guarded = _guarded_transport_module()
    feed_sizes: list[int] = []
    scanners: list[Any] = []

    class _RecordingScanner:
        def __init__(self, *, mode: str) -> None:
            self.delegate = ShallowStandaloneFieldScanner(mode=mode)
            self.finish_calls = 0
            scanners.append(self)

        @property
        def retained_bytes(self) -> int:
            return self.delegate.retained_bytes

        @property
        def requires_value_lookbehind(self) -> bool:
            return self.delegate.requires_value_lookbehind

        def feed(self, chunk: bytes) -> int | None:
            feed_sizes.append(len(chunk))
            return self.delegate.feed(chunk)

        def finish(self) -> None:
            self.finish_calls += 1
            self.delegate.finish()

    monkeypatch.setattr(guarded, "ShallowStandaloneFieldScanner", _RecordingScanner)
    protocols: list[Any] = []

    class _InspectingGuardedProtocol(guarded.GuardedSlidesWebSocketProtocol):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            protocols.append(self)

    app = _EchoApplication()
    arguments = {
        "padding": "x" * (SCANNER_CHUNK_BYTES * 2 + 4096),
        "content_kind": "structured_slides",
        "patch": {"content_kind": "structured_slides"},
    }
    message = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "notes.echo", "arguments": arguments},
        },
        separators=(",", ":"),
    )
    split_at = message.index("structured_slides") + len("struct")

    async with _serve_websocket(app, _InspectingGuardedProtocol) as uri:
        async with websockets.connect(uri, compression="deflate", max_size=None) as client:
            assert client.response.headers.get("Sec-WebSocket-Extensions") is None
            await client.send([message[:split_at], message[split_at:]])
            assert await client.recv() == message

    assert app.messages == [message]
    assert app.scope is not None
    assert guarded.is_guarded_slides_websocket_scope(app.scope)
    assert feed_sizes
    assert max(feed_sizes) <= SCANNER_CHUNK_BYTES
    assert scanners and all(scanner.retained_bytes == 0 for scanner in scanners)
    assert all(scanner.finish_calls >= 1 for scanner in scanners)
    assert protocols and all(protocol.guard_retained_bytes == 0 for protocol in protocols)


@pytest.mark.asyncio
async def test_guarded_websocket_rejects_before_html_value_and_redacts_logs(caplog) -> None:
    guarded = _guarded_transport_module()
    protocols: list[Any] = []

    class _InspectingGuardedProtocol(guarded.GuardedSlidesWebSocketProtocol):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            protocols.append(self)

    app = _EchoApplication()
    secret = "TOP-SECRET-WEBSOCKET-SOURCE"
    prefix = (
        '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":'
        '{"name":"slides.presentations.patch","arguments":{"patch":{"html_document":'
    )

    async def _fragments() -> AsyncIterator[str]:
        yield prefix
        await asyncio.sleep(0.1)
        yield json.dumps(secret) + "}}}"

    async with _serve_websocket(app, _InspectingGuardedProtocol) as uri:
        async with websockets.connect(uri, compression=None) as client:
            with pytest.raises(ConnectionClosed) as closed:
                await client.send(_fragments())
                await client.recv()

    assert closed.value.code == 1008
    assert app.messages == []
    assert app.disconnect is not None
    assert app.disconnect["code"] == 1008
    assert secret not in caplog.text
    assert prefix not in caplog.text
    assert protocols and all(protocol.guard_retained_bytes == 0 for protocol in protocols)


@pytest.mark.asyncio
async def test_guarded_websocket_rejects_split_forbidden_content_kind_without_replay() -> None:
    guarded = _guarded_transport_module()
    app = _EchoApplication()
    first = (
        '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":'
        '{"name":"slides.presentations.create","arguments":{"content_kind":"standalone_'
    )
    second = 'html","title":"Never delivered"}}}'

    async with _serve_websocket(app, guarded.GuardedSlidesWebSocketProtocol) as uri:
        async with websockets.connect(uri, compression=None) as client:
            with pytest.raises(ConnectionClosed) as closed:
                await client.send([first, second])
                await client.recv()

    assert closed.value.code == 1008
    assert app.messages == []


@pytest.mark.asyncio
async def test_standard_uvicorn_websocket_cannot_forge_guard_marker() -> None:
    guarded = _guarded_transport_module()
    app = _EchoApplication()

    async with _serve_websocket(app, WebSocketsSansIOProtocol) as uri:
        forged_uri = f"{uri}?_tldw_guarded_slides_websocket=true"
        async with websockets.connect(
            forged_uri,
            additional_headers={"X-TLDW-Guarded-Slides-WebSocket": "true"},
            compression=None,
        ) as client:
            await client.send("ordinary")
            assert await client.recv() == "ordinary"

    assert app.scope is not None
    assert not guarded.is_guarded_slides_websocket_scope(app.scope)


class _ModuleStub:
    def __init__(self, name: str, tool_names: list[str]) -> None:
        self.name = name
        self._tool_names = tool_names

    async def get_tools(self) -> list[dict[str, Any]]:
        return [{"name": name, "inputSchema": {"type": "object"}} for name in self._tool_names]

    def is_write_tool_def(self, _tool: dict[str, Any]) -> bool:
        return False


class _RegistryStub:
    def __init__(self) -> None:
        self.modules = {
            "slides": _ModuleStub("Slides", ["slides.presentations.list"]),
            "notes": _ModuleStub("Notes", ["notes.search"]),
        }

    async def get_all_modules(self) -> dict[str, _ModuleStub]:
        return self.modules


@pytest.mark.asyncio
async def test_protocol_filters_slides_per_websocket_request_without_registry_mutation() -> None:
    guarded = _guarded_transport_module()
    from tldw_Server_API.app.core.MCP_unified.protocol import MCPProtocol, RequestContext
    from tldw_Server_API.app.core.MCP_unified.server import _websocket_transport_metadata

    registry = _RegistryStub()
    protocol = MCPProtocol()
    protocol.module_registry = registry

    async def _allow(*_args, **_kwargs) -> bool:
        return True

    protocol._has_module_permission = _allow
    protocol._has_tool_permission = _allow

    guarded_app = _EchoApplication()
    async with _serve_websocket(guarded_app, guarded.GuardedSlidesWebSocketProtocol) as uri:
        async with websockets.connect(uri, compression=None) as client:
            await client.send("scope")
            assert await client.recv() == "scope"
    assert guarded_app.scope is not None

    guarded_context = RequestContext(
        "guarded",
        metadata=_websocket_transport_metadata(guarded_app.scope),
    )
    unguarded_context = RequestContext(
        "unguarded",
        metadata=_websocket_transport_metadata(
            {
                "type": "websocket",
                guarded.GUARDED_SLIDES_SCOPE_KEY: True,
            }
        ),
    )
    http_context = RequestContext("http", metadata={"mcp_transport": "http"})

    guarded_names = {tool["name"] for tool in (await protocol._handle_tools_list({}, guarded_context))["tools"]}
    unguarded_names = {tool["name"] for tool in (await protocol._handle_tools_list({}, unguarded_context))["tools"]}
    http_names = {tool["name"] for tool in (await protocol._handle_tools_list({}, http_context))["tools"]}

    assert guarded_names == {"slides.presentations.list", "notes.search"}
    assert unguarded_names == {"notes.search"}
    assert http_names == {"slides.presentations.list", "notes.search"}
    assert set(registry.modules) == {"slides", "notes"}


@pytest.mark.asyncio
async def test_protocol_rejects_direct_slides_call_only_on_unguarded_websocket() -> None:
    guarded = _guarded_transport_module()
    from tldw_Server_API.app.core.MCP_unified.protocol import MCPProtocol, RequestContext
    from tldw_Server_API.app.core.MCP_unified.server import _websocket_transport_metadata

    calls: list[str] = []

    class _ExecutionStub:
        async def handle_tools_call(self, params, _context):
            calls.append(params["name"])
            return {"success": True, "tool": params["name"]}

    protocol = MCPProtocol()
    protocol._tool_execution = _ExecutionStub()
    scope: dict[str, Any] = {}
    guarded._mark_guarded_slides_websocket_scope(scope)
    guarded_context = RequestContext(
        "guarded-call",
        metadata=_websocket_transport_metadata(scope),
    )
    unguarded_context = RequestContext(
        "unguarded-call",
        metadata={"mcp_transport": "websocket", guarded.GUARDED_SLIDES_SCOPE_KEY: True},
    )

    rejected = await protocol._handle_tools_call(
        {"name": "slides.presentations.list", "arguments": {}},
        unguarded_context,
    )
    unrelated = await protocol._handle_tools_call(
        {"name": "notes.search", "arguments": {}},
        unguarded_context,
    )
    permitted = await protocol._handle_tools_call(
        {"name": "slides.presentations.list", "arguments": {}},
        guarded_context,
    )

    assert rejected == {
        "success": False,
        "error": {
            "code": "slides_websocket_guard_required",
            "operation": "slides.presentations.list",
        },
    }
    assert unrelated["success"] is True
    assert permitted["success"] is True
    assert calls == ["notes.search", "slides.presentations.list"]


def test_guarded_launcher_uses_exact_protocol_and_disables_compression(monkeypatch) -> None:
    from tldw_Server_API.scripts import run_server_guarded_mcp

    captured: dict[str, Any] = {}

    def _run(app: str, **kwargs: Any) -> None:
        captured["app"] = app
        captured.update(kwargs)

    monkeypatch.setattr(run_server_guarded_mcp.uvicorn, "run", _run)
    run_server_guarded_mcp.main(
        [
            "--host",
            "127.0.0.2",
            "--port",
            "8123",
            "--workers",
            "2",
            "--log-level",
            "debug",
            "--no-proxy-headers",
            "--forwarded-allow-ips",
            "10.0.0.1",
        ]
    )

    assert captured == {
        "app": "tldw_Server_API.app.main:app",
        "host": "127.0.0.2",
        "port": 8123,
        "workers": 2,
        "log_level": "debug",
        "proxy_headers": False,
        "forwarded_allow_ips": "10.0.0.1",
        "ws": run_server_guarded_mcp.GuardedSlidesWebSocketProtocol,
        "ws_per_message_deflate": False,
    }
