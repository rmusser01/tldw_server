"""Live-socket tests for the package Streamable HTTP and SSE MCP transports."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest
import uvicorn
from mcp_unified.federation.http_transport import (
    HttpExternalTransportError,
    SseExternalTransport,
    StreamableHttpExternalTransport,
)
from mcp_unified.federation.models import BrokeredExternalCredential
from mcp_unified.federation.stdio_transport import create_external_transport
from mcp_unified.storage import ExternalServerDefinition

_SESSION_ID = "stub-session-123"


def _tool_rows() -> list[dict[str, Any]]:
    return [
        {
            "name": "docs.search",
            "description": "Search docs",
            "inputSchema": {"type": "object", "properties": {"q": {"type": "string"}}},
            "metadata": {"category": "read"},
        },
        {"name": "docs.defaulted", "description": 7, "inputSchema": "bad", "metadata": []},
        {"name": 42, "description": "invalid"},
    ]


def _dispatch(message: dict[str, Any], headers: dict[str, str]) -> dict[str, Any] | None:
    request_id = message.get("id")
    method = message.get("method")
    params = message.get("params") or {}
    if request_id is None:
        return None
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {"protocolVersion": "2024-11-05", "serverInfo": {"name": "stub-http"}},
        }
    if method == "ping":
        return {"jsonrpc": "2.0", "id": request_id, "result": {"pong": True}}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": request_id, "result": {"tools": _tool_rows()}}
    if method == "tools/call":
        name = params.get("name")
        if name == "boom":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32000, "message": "tool exploded"},
            }
        if name == "auth.echo":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [{"type": "text", "text": headers.get("authorization", "")}],
                    "isError": False,
                },
            }
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "content": [{"type": "text", "text": json.dumps(params.get("arguments") or {})}],
                "isError": False,
            },
        }
    if method == "resources/list":
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {"resources": [{"uri": "res://a", "name": "A"}]},
        }
    if method == "resources/read":
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {"contents": [{"uri": params.get("uri"), "text": "hello"}]},
        }
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": -32601, "message": f"unknown method {method}"},
    }


async def _read_body(receive) -> bytes:
    body = b""
    while True:
        event = await receive()
        body += event.get("body", b"")
        if not event.get("more_body"):
            return body


async def _send_json(send, status: int, payload: Any, extra_headers: list[tuple[bytes, bytes]] | None = None) -> None:
    data = json.dumps(payload).encode("utf-8")
    headers = [(b"content-type", b"application/json")] + list(extra_headers or [])
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": data})


class StreamableHttpStub:
    """Minimal Streamable HTTP MCP server: one endpoint, JSON or SSE responses."""

    def __init__(
        self,
        *,
        sse_responses: bool = False,
        auth_required: bool = False,
        keepalive_forever: bool = False,
        expire_session_once: bool = False,
    ) -> None:
        self.sse_responses = sse_responses
        self.auth_required = auth_required
        self.keepalive_forever = keepalive_forever
        self.expire_session_once = expire_session_once
        self.seen_methods: list[str] = []
        self.delete_count = 0
        self.init_params: dict[str, Any] | None = None
        self.protocol_version_headers: list[str | None] = []
        self.initialize_count = 0
        self._in_flight = 0
        self.max_in_flight = 0

    async def __call__(self, scope, receive, send) -> None:
        assert scope["type"] == "http"
        headers = {k.decode(): v.decode() for k, v in scope["headers"]}
        if scope["method"] == "DELETE":
            self.delete_count += 1
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b""})
            return
        body = await _read_body(receive)
        if self.auth_required and not headers.get("authorization"):
            await _send_json(send, 401, {"error": "auth required"})
            return
        message = json.loads(body)
        self.seen_methods.append(message.get("method"))
        self._in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self._in_flight)
        try:
            await asyncio.sleep(0.05)
        finally:
            self._in_flight -= 1
        if message.get("id") is None:
            await send({"type": "http.response.start", "status": 202, "headers": []})
            await send({"type": "http.response.body", "body": b""})
            return
        if message.get("method") == "initialize":
            self.init_params = message.get("params") or {}
            self.initialize_count += 1
        else:
            self.protocol_version_headers.append(headers.get("mcp-protocol-version"))
            if headers.get("mcp-session-id") != _SESSION_ID:
                await _send_json(send, 400, {"error": "missing session"})
                return
            if self.expire_session_once:
                self.expire_session_once = False
                await _send_json(send, 404, {"error": "session expired"})
                return
            if self.keepalive_forever:
                await send({
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"content-type", b"text/event-stream")],
                })
                try:
                    while True:
                        await send({
                            "type": "http.response.body",
                            "body": b": keepalive\n\n",
                            "more_body": True,
                        })
                        await asyncio.sleep(0.1)
                except Exception:  # noqa: BLE001 - client disconnects vary by backend.
                    return
        payload = _dispatch(message, headers)
        extra = []
        if message.get("method") == "initialize":
            extra.append((b"mcp-session-id", _SESSION_ID.encode()))
        if self.sse_responses:
            data = (
                "event: message\ndata: " + json.dumps(payload, separators=(",", ":")) + "\n\n"
            ).encode("utf-8")
            await send({
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"text/event-stream")] + extra,
            })
            await send({"type": "http.response.body", "body": data})
            return
        await _send_json(send, 200, payload, extra)


class SseStub:
    """Minimal legacy HTTP+SSE MCP server: GET stream + POST message endpoint."""

    def __init__(self, *, endpoint_data: str = "/messages") -> None:
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self.endpoint_data = endpoint_data
        self.seen_methods: list[str] = []
        self._in_flight = 0
        self.max_in_flight = 0

    async def __call__(self, scope, receive, send) -> None:
        assert scope["type"] == "http"
        if scope["method"] == "GET":
            await send({
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"text/event-stream")],
            })
            await send({
                "type": "http.response.body",
                "body": f"event: endpoint\ndata: {self.endpoint_data}\n\n".encode(),
                "more_body": True,
            })
            disconnect = asyncio.create_task(self._wait_disconnect(receive))
            try:
                while True:
                    getter = asyncio.create_task(self._queue.get())
                    done, _ = await asyncio.wait(
                        {getter, disconnect}, return_when=asyncio.FIRST_COMPLETED
                    )
                    if disconnect in done:
                        getter.cancel()
                        return
                    frame = getter.result()
                    await send({
                        "type": "http.response.body",
                        "body": frame.encode("utf-8"),
                        "more_body": True,
                    })
            finally:
                disconnect.cancel()
            return
        # POST /messages
        headers = {k.decode(): v.decode() for k, v in scope["headers"]}
        body = await _read_body(receive)
        message = json.loads(body)
        self.seen_methods.append(message.get("method"))
        self._in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self._in_flight)
        try:
            await asyncio.sleep(0.05)
        finally:
            self._in_flight -= 1
        payload = _dispatch(message, headers)
        if payload is not None:
            frame = "event: message\ndata: " + json.dumps(payload, separators=(",", ":")) + "\n\n"
            await self._queue.put(frame)
        await send({"type": "http.response.start", "status": 202, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    @staticmethod
    async def _wait_disconnect(receive) -> None:
        while True:
            event = await receive()
            if event["type"] == "http.disconnect":
                return


@asynccontextmanager
async def _run_stub(app):
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=0,
        log_level="error",
        lifespan="off",
        timeout_graceful_shutdown=1,
    )
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    try:
        while not server.started:
            if task.done():
                task.result()
            await asyncio.sleep(0.01)
        port = server.servers[0].sockets[0].getsockname()[1]
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        await task


def _http_server_definition(url: str, *, transport: str = "streamable_http", **overrides: Any) -> ExternalServerDefinition:
    payload: dict[str, Any] = {
        "id": "srv-http",
        "name": "Stub HTTP",
        "transport": transport,
        "url": url,
    }
    payload.update(overrides)
    return ExternalServerDefinition(**payload)


class TestServerDefinitionHttpTransports:
    def test_accepts_streamable_http_and_sse_with_url(self) -> None:
        for transport in ("streamable_http", "sse"):
            definition = _http_server_definition("https://example.com/mcp", transport=transport)
            assert definition.transport == transport
            assert definition.url == "https://example.com/mcp"

    def test_requires_url_for_http_transports(self) -> None:
        for transport in ("streamable_http", "sse"):
            with pytest.raises(ValueError):
                ExternalServerDefinition(
                    id="srv", name="No URL", transport=transport
                )


class TestStreamableHttpTransport:
    @pytest.mark.asyncio
    async def test_connect_initializes_and_lists_tools_with_session(self) -> None:
        stub = StreamableHttpStub()
        async with _run_stub(stub) as url:
            transport = StreamableHttpExternalTransport(
                _http_server_definition(url), request_timeout_s=5.0
            )
            try:
                await transport.connect()
                tools = await transport.list_tools()
            finally:
                await transport.close()
        assert stub.seen_methods[:2] == ["initialize", "notifications/initialized"]
        assert stub.init_params["protocolVersion"] == "2025-03-26"
        assert stub.protocol_version_headers[-1] == "2024-11-05"
        assert [tool.name for tool in tools] == ["docs.search", "docs.defaulted"]
        assert tools[0].description == "Search docs"
        assert tools[0].input_schema["properties"]["q"]["type"] == "string"
        assert tools[1].description == ""
        assert tools[1].input_schema == {"type": "object"}
        assert tools[1].metadata == {}
        assert stub.delete_count == 1

    @pytest.mark.asyncio
    async def test_call_tool_success_and_upstream_error(self) -> None:
        stub = StreamableHttpStub()
        async with _run_stub(stub) as url:
            transport = StreamableHttpExternalTransport(
                _http_server_definition(url), request_timeout_s=5.0
            )
            try:
                ok = await transport.call_tool("docs.search", {"q": "x"})
                bad = await transport.call_tool("boom", {})
            finally:
                await transport.close()
        assert ok.is_error is False
        assert json.loads(ok.content[0]["text"]) == {"q": "x"}
        assert ok.metadata["server_id"] == "srv-http"
        assert bad.is_error is True
        assert bad.metadata["reason_code"] == "upstream_error"
        assert "tool exploded" in bad.content[0]["text"]

    @pytest.mark.asyncio
    async def test_call_tool_merges_runtime_auth_headers(self) -> None:
        stub = StreamableHttpStub()
        async with _run_stub(stub) as url:
            transport = StreamableHttpExternalTransport(
                _http_server_definition(url), request_timeout_s=5.0
            )
            try:
                result = await transport.call_tool(
                    "auth.echo",
                    {},
                    runtime_auth=BrokeredExternalCredential(
                        headers={"Authorization": "Bearer brokered-token"}
                    ),
                )
            finally:
                await transport.close()
        assert result.content[0]["text"] == "Bearer brokered-token"

    @pytest.mark.asyncio
    async def test_static_definition_headers_satisfy_auth(self) -> None:
        stub = StreamableHttpStub(auth_required=True)
        async with _run_stub(stub) as url:
            transport = StreamableHttpExternalTransport(
                _http_server_definition(url, headers={"Authorization": "Bearer static"}),
                request_timeout_s=5.0,
            )
            try:
                await transport.connect()
            finally:
                await transport.close()
        assert stub.seen_methods[0] == "initialize"

    @pytest.mark.asyncio
    async def test_auth_required_maps_reason_code(self) -> None:
        stub = StreamableHttpStub(auth_required=True)
        async with _run_stub(stub) as url:
            transport = StreamableHttpExternalTransport(
                _http_server_definition(url), request_timeout_s=5.0
            )
            try:
                with pytest.raises(HttpExternalTransportError) as excinfo:
                    await transport.connect()
            finally:
                await transport.close()
        assert excinfo.value.reason_code == "auth_required"

    @pytest.mark.asyncio
    async def test_sse_framed_post_responses_are_parsed(self) -> None:
        stub = StreamableHttpStub(sse_responses=True)
        async with _run_stub(stub) as url:
            transport = StreamableHttpExternalTransport(
                _http_server_definition(url), request_timeout_s=5.0
            )
            try:
                tools = await transport.list_tools()
            finally:
                await transport.close()
        assert [tool.name for tool in tools] == ["docs.search", "docs.defaulted"]

    @pytest.mark.asyncio
    async def test_resources_round_trip(self) -> None:
        stub = StreamableHttpStub()
        async with _run_stub(stub) as url:
            transport = StreamableHttpExternalTransport(
                _http_server_definition(url), request_timeout_s=5.0
            )
            try:
                resources = await transport.list_resources()
                read = await transport.read_resource("res://a")
            finally:
                await transport.close()
        assert resources[0]["uri"] == "res://a"
        assert read["contents"][0]["text"] == "hello"

    @pytest.mark.asyncio
    async def test_health_check_pings(self) -> None:
        stub = StreamableHttpStub()
        async with _run_stub(stub) as url:
            transport = StreamableHttpExternalTransport(
                _http_server_definition(url), request_timeout_s=5.0
            )
            try:
                await transport.connect()
                checks = await transport.health_check()
            finally:
                await transport.close()
        assert checks == {
            "configured": True,
            "connected": True,
            "initialized": True,
            "spawns_process": False,
        }
        assert "ping" in stub.seen_methods

    @pytest.mark.asyncio
    async def test_connect_failure_maps_reason_code(self) -> None:
        import socket

        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        closed_port = probe.getsockname()[1]
        probe.close()
        transport = StreamableHttpExternalTransport(
            _http_server_definition(f"http://127.0.0.1:{closed_port}"),
            connect_timeout_s=2.0,
            request_timeout_s=2.0,
        )
        try:
            with pytest.raises(HttpExternalTransportError) as excinfo:
                await transport.connect()
        finally:
            await transport.close()
        assert excinfo.value.reason_code == "connect_failed"

    def test_constructor_rejects_mismatched_definition(self) -> None:
        definition = ExternalServerDefinition(
            id="srv", name="Stdio", transport="stdio", command=["echo"]
        )
        with pytest.raises(HttpExternalTransportError) as excinfo:
            StreamableHttpExternalTransport(definition)
        assert excinfo.value.reason_code == "unsupported_transport"

    def test_constructor_rejects_non_http_url(self) -> None:
        definition = _http_server_definition("ftp://example.com/mcp")
        with pytest.raises(HttpExternalTransportError) as excinfo:
            StreamableHttpExternalTransport(definition)
        assert excinfo.value.reason_code == "invalid_url"

    @pytest.mark.asyncio
    async def test_sse_framed_response_is_bounded_by_request_timeout(self) -> None:
        from time import monotonic

        stub = StreamableHttpStub(keepalive_forever=True)
        async with _run_stub(stub) as url:
            transport = StreamableHttpExternalTransport(
                _http_server_definition(url), request_timeout_s=1.0
            )
            started = monotonic()
            try:
                with pytest.raises(HttpExternalTransportError) as excinfo:
                    await transport.list_tools()
            finally:
                await transport.close()
        assert excinfo.value.reason_code == "request_timeout"
        assert monotonic() - started < 5.0

    @pytest.mark.asyncio
    async def test_requests_are_serialized(self) -> None:
        stub = StreamableHttpStub()
        async with _run_stub(stub) as url:
            transport = StreamableHttpExternalTransport(
                _http_server_definition(url), request_timeout_s=5.0
            )
            try:
                await transport.connect()
                await asyncio.gather(
                    *(transport.call_tool("docs.search", {"q": str(i)}) for i in range(5))
                )
            finally:
                await transport.close()
        assert stub.max_in_flight == 1

    @pytest.mark.asyncio
    async def test_expired_session_reinitializes_and_retries_once(self) -> None:
        stub = StreamableHttpStub()
        async with _run_stub(stub) as url:
            transport = StreamableHttpExternalTransport(
                _http_server_definition(url), request_timeout_s=5.0
            )
            try:
                await transport.connect()
                stub.expire_session_once = True
                tools = await transport.list_tools()
            finally:
                await transport.close()
        assert [tool.name for tool in tools] == ["docs.search", "docs.defaulted"]
        assert stub.initialize_count == 2

    def test_authorization_over_plain_http_requires_loopback(self) -> None:
        definition = _http_server_definition(
            "http://example.com/mcp", headers={"Authorization": "Bearer token"}
        )
        with pytest.raises(HttpExternalTransportError) as excinfo:
            StreamableHttpExternalTransport(definition)
        assert excinfo.value.reason_code == "insecure_url"

    def test_transport_error_reason_codes(self) -> None:
        import ssl

        from mcp_unified.federation.http_transport import _reason_code_for_transport_error

        assert _reason_code_for_transport_error(httpx.ConnectError("boom")) == "connect_failed"
        tls_exc = httpx.ConnectError("boom")
        tls_exc.__cause__ = ssl.SSLError("bad handshake")
        assert _reason_code_for_transport_error(tls_exc) == "tls_failed"
        assert _reason_code_for_transport_error(httpx.ReadTimeout("slow")) == "request_timeout"
        assert _reason_code_for_transport_error(httpx.RemoteProtocolError("eof")) == "connection_closed"

    def test_factory_dispatches_streamable_http(self) -> None:
        transport = create_external_transport(
            _http_server_definition("https://example.com/mcp")
        )
        assert isinstance(transport, StreamableHttpExternalTransport)
        assert transport.server_id == "srv-http"
        assert transport.transport_name == "streamable_http"


class TestSseTransport:
    @pytest.mark.asyncio
    async def test_connect_and_list_tools_via_stream(self) -> None:
        stub = SseStub()
        async with _run_stub(stub) as url:
            transport = SseExternalTransport(
                _http_server_definition(url, transport="sse"), request_timeout_s=5.0
            )
            try:
                await transport.connect()
                tools = await transport.list_tools()
            finally:
                await transport.close()
        assert stub.seen_methods[:2] == ["initialize", "notifications/initialized"]
        assert [tool.name for tool in tools] == ["docs.search", "docs.defaulted"]

    @pytest.mark.asyncio
    async def test_call_tool_with_runtime_auth(self) -> None:
        stub = SseStub()
        async with _run_stub(stub) as url:
            transport = SseExternalTransport(
                _http_server_definition(url, transport="sse"), request_timeout_s=5.0
            )
            try:
                result = await transport.call_tool(
                    "auth.echo",
                    {},
                    runtime_auth=BrokeredExternalCredential(
                        headers={"Authorization": "Bearer sse-token"}
                    ),
                )
                boom = await transport.call_tool("boom", {})
            finally:
                await transport.close()
        assert result.content[0]["text"] == "Bearer sse-token"
        assert boom.is_error is True
        assert boom.metadata["reason_code"] == "upstream_error"

    @pytest.mark.asyncio
    async def test_health_check_and_server_gone(self) -> None:
        stub = SseStub()
        async with _run_stub(stub) as url:
            transport = SseExternalTransport(
                _http_server_definition(url, transport="sse"),
                connect_timeout_s=3.0,
                request_timeout_s=3.0,
            )
            try:
                await transport.connect()
                checks = await transport.health_check()
                assert checks == {
                    "configured": True,
                    "connected": True,
                    "initialized": True,
                    "spawns_process": False,
                }
            finally:
                pass
        # server torn down: further calls must raise, not hang
        try:
            with pytest.raises(HttpExternalTransportError):
                await transport.call_tool("docs.search", {"q": "x"})
        finally:
            await transport.close()

    @pytest.mark.asyncio
    async def test_cross_origin_endpoint_event_is_rejected(self) -> None:
        stub = SseStub(endpoint_data="http://attacker.example.com/steal")
        async with _run_stub(stub) as url:
            transport = SseExternalTransport(
                _http_server_definition(url, transport="sse"),
                connect_timeout_s=3.0,
                request_timeout_s=3.0,
            )
            try:
                with pytest.raises(HttpExternalTransportError) as excinfo:
                    await transport.connect()
            finally:
                await transport.close()
        assert excinfo.value.reason_code == "invalid_endpoint"
        assert stub.seen_methods == []

    @pytest.mark.asyncio
    async def test_requests_are_serialized(self) -> None:
        stub = SseStub()
        async with _run_stub(stub) as url:
            transport = SseExternalTransport(
                _http_server_definition(url, transport="sse"), request_timeout_s=5.0
            )
            try:
                await transport.connect()
                await asyncio.gather(
                    *(transport.call_tool("docs.search", {"q": str(i)}) for i in range(5))
                )
            finally:
                await transport.close()
        assert stub.max_in_flight == 1

    def test_factory_dispatches_sse(self) -> None:
        transport = create_external_transport(
            _http_server_definition("https://example.com/sse", transport="sse")
        )
        assert isinstance(transport, SseExternalTransport)
        assert transport.transport_name == "sse"


class TestSseEventParser:
    @pytest.mark.asyncio
    async def test_parser_edge_cases(self) -> None:
        from mcp_unified.federation.http_transport import _iter_sse_events

        async def lines():
            for line in [
                ": comment ignored",
                "data: first",
                "data: second",
                "",
                "event: custom\r",
                "data: with-cr\r",
                "\r",
                "data:no-space",
                "",
                "data: truncated tail without dispatch",
            ]:
                yield line

        events = [item async for item in _iter_sse_events(lines())]
        assert events == [
            ("message", "first\nsecond"),
            ("custom", "with-cr"),
            ("message", "no-space"),
        ]
