from __future__ import annotations

import asyncio
import json

import pytest

from tldw_Server_API.app.core.exceptions import NetworkError
from tldw_Server_API.app.core.MCP_unified.external_servers.config_schema import (
    ExternalAuthConfig,
    ExternalAuthMode,
    ExternalMCPServerConfig,
    ExternalTimeoutConfig,
    ExternalTransportType,
    ExternalWebSocketConfig,
)
from tldw_Server_API.app.core.MCP_unified.external_servers.transports.websocket_adapter import (
    WebSocketExternalMCPAdapter,
)
from tldw_Server_API.app.core.MCP_unified.external_servers.transports.base import (
    BrokeredExternalCredential,
)


class _FakeWebSocket:
    def __init__(self) -> None:
        self.sent_messages: list[dict] = []
        self._recv_queue: asyncio.Queue[dict | str | bytes | Exception] = asyncio.Queue()
        self.closed = False

    def enqueue(self, payload: dict | str | bytes | Exception) -> None:
        self._recv_queue.put_nowait(payload)

    async def send(self, data: str) -> None:
        self.sent_messages.append(json.loads(data))

    async def recv(self) -> str | bytes:
        payload = await self._recv_queue.get()
        if isinstance(payload, Exception):
            raise payload
        if isinstance(payload, dict):
            return json.dumps(payload)
        return payload

    async def close(self) -> None:
        self.closed = True


def _server_config(
    *,
    request_seconds: float = 0.2,
    auth: ExternalAuthConfig | None = None,
) -> ExternalMCPServerConfig:
    return ExternalMCPServerConfig(
        id="docs",
        name="Docs",
        transport=ExternalTransportType.WEBSOCKET,
        websocket=ExternalWebSocketConfig(
            url="wss://example.test/mcp",
            subprotocols=["mcp"],
            headers={"x-client": "tldw"},
        ),
        auth=auth or ExternalAuthConfig(),
        timeouts=ExternalTimeoutConfig(connect_seconds=1.0, request_seconds=request_seconds),
    )


async def _wait_for_sent(ws: _FakeWebSocket, expected_count: int, timeout: float = 0.5) -> None:
    end = asyncio.get_running_loop().time() + timeout
    while len(ws.sent_messages) < expected_count:
        if asyncio.get_running_loop().time() >= end:
            raise AssertionError(
                f"Expected at least {expected_count} sent websocket messages, got {len(ws.sent_messages)}"
            )
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_websocket_adapter_connect_initializes_and_applies_auth_headers(monkeypatch) -> None:
    monkeypatch.setenv("EXTERNAL_DOCS_TOKEN", "token-123")
    cfg = _server_config(
        auth=ExternalAuthConfig(mode=ExternalAuthMode.BEARER_ENV, token_env="EXTERNAL_DOCS_TOKEN")
    )
    ws = _FakeWebSocket()
    ws.enqueue({"jsonrpc": "2.0", "id": 1, "result": {"serverInfo": {"name": "stub"}}})
    seen: dict[str, object] = {}

    async def _connector(*, url: str, subprotocols: list[str], headers: dict[str, str], connect_timeout: float):
        seen["url"] = url
        seen["subprotocols"] = list(subprotocols)
        seen["headers"] = dict(headers)
        seen["connect_timeout"] = connect_timeout
        return ws

    adapter = WebSocketExternalMCPAdapter(cfg, ws_connector=_connector)
    try:
        await adapter.connect()
        health = await adapter.health_check()
        assert health["configured"] is True
        assert health["connected"] is True
        assert health["initialized"] is True
        assert seen["url"] == "wss://example.test/mcp"
        assert seen["subprotocols"] == ["mcp"]
        assert seen["connect_timeout"] == 1.0
        headers = seen["headers"]
        assert isinstance(headers, dict)
        assert headers["x-client"] == "tldw"
        assert headers["Authorization"] == "Bearer token-123"
        assert ws.sent_messages[0]["method"] == "initialize"
        assert ws.sent_messages[0]["jsonrpc"] == "2.0"
    finally:
        await adapter.close()


@pytest.mark.asyncio
async def test_websocket_adapter_list_tools_normalizes_response() -> None:
    cfg = _server_config()
    ws = _FakeWebSocket()
    ws.enqueue({"jsonrpc": "2.0", "id": 1, "result": {"serverInfo": {"name": "stub"}}})

    async def _connector(**kwargs):
        return ws

    adapter = WebSocketExternalMCPAdapter(cfg, ws_connector=_connector)
    try:
        await adapter.connect()
        list_task = asyncio.create_task(adapter.list_tools())
        await _wait_for_sent(ws, expected_count=2)
        list_request_id = ws.sent_messages[1]["id"]
        ws.enqueue(
            {
                "jsonrpc": "2.0",
                "id": list_request_id,
                "result": {
                    "tools": [
                        {
                            "name": "docs.search",
                            "description": "Search docs",
                            "inputSchema": {"type": "object", "properties": {"q": {"type": "string"}}},
                            "metadata": {"scope": "read"},
                        },
                        {"name": 7, "description": "invalid"},
                    ]
                },
            }
        )
        tools = await list_task
        assert len(tools) == 1
        assert tools[0].name == "docs.search"
        assert tools[0].description == "Search docs"
        assert tools[0].input_schema["type"] == "object"
        assert tools[0].metadata["scope"] == "read"
    finally:
        await adapter.close()


@pytest.mark.asyncio
async def test_websocket_adapter_call_tool_maps_upstream_error() -> None:
    cfg = _server_config()
    ws = _FakeWebSocket()
    ws.enqueue({"jsonrpc": "2.0", "id": 1, "result": {"serverInfo": {"name": "stub"}}})

    async def _connector(**kwargs):
        return ws

    adapter = WebSocketExternalMCPAdapter(cfg, ws_connector=_connector)
    try:
        await adapter.connect()
        call_task = asyncio.create_task(adapter.call_tool("docs.search", {"q": "x"}))
        await _wait_for_sent(ws, expected_count=2)
        call_request_id = ws.sent_messages[1]["id"]
        ws.enqueue(
            {
                "jsonrpc": "2.0",
                "id": call_request_id,
                "error": {"code": -32042, "message": "upstream failed"},
            }
        )
        result = await call_task
        assert result.is_error is True
        assert isinstance(result.content, list)
        assert result.content[0]["text"] == "upstream failed"
        assert result.metadata["upstream_error"]["code"] == -32042
    finally:
        await adapter.close()


@pytest.mark.asyncio
async def test_websocket_adapter_lists_and_reads_resources() -> None:
    cfg = _server_config()
    ws = _FakeWebSocket()
    ws.enqueue({"jsonrpc": "2.0", "id": 1, "result": {"serverInfo": {"name": "stub"}}})

    async def _connector(**kwargs):
        return ws

    adapter = WebSocketExternalMCPAdapter(cfg, ws_connector=_connector)
    try:
        await adapter.connect()
        list_task = asyncio.create_task(adapter.list_resources())
        await _wait_for_sent(ws, expected_count=2)
        list_request_id = ws.sent_messages[1]["id"]
        ws.enqueue(
            {
                "jsonrpc": "2.0",
                "id": list_request_id,
                "result": {
                    "resources": [
                        {
                            "uri": "docs://one",
                            "name": "Docs One",
                            "description": "Reference",
                            "mimeType": "text/plain",
                            "metadata": {"scope": "read"},
                        },
                        {"uri": 7, "name": "invalid"},
                    ]
                },
            }
        )
        resources = await list_task

        read_task = asyncio.create_task(adapter.read_resource("docs://one"))
        await _wait_for_sent(ws, expected_count=3)
        read_request = ws.sent_messages[2]
        ws.enqueue(
            {
                "jsonrpc": "2.0",
                "id": read_request["id"],
                "result": {
                    "contents": [
                        {
                            "uri": "docs://one",
                            "mimeType": "text/plain",
                            "text": "hello",
                        }
                    ]
                },
            }
        )
        result = await read_task

        assert resources == [
            {
                "uri": "docs://one",
                "name": "Docs One",
                "description": "Reference",
                "mimeType": "text/plain",
                "metadata": {"scope": "read"},
            }
        ]
        assert read_request["method"] == "resources/read"
        assert read_request["params"] == {"uri": "docs://one"}
        assert result["contents"][0]["text"] == "hello"
    finally:
        await adapter.close()


@pytest.mark.asyncio
async def test_websocket_adapter_resource_errors_raise_network_error() -> None:
    cfg = _server_config()
    adapter = WebSocketExternalMCPAdapter(cfg)
    responses = [
        {"error": {"code": -32000, "message": "list failed"}},
        {"error": {"code": -32001, "message": "read failed"}},
    ]

    async def _ensure_connected() -> None:
        return None

    async def _request(method: str, params: dict[str, object]) -> dict[str, object]:
        assert method in {"resources/list", "resources/read"}
        assert isinstance(params, dict)
        return responses.pop(0)

    adapter._ensure_connected = _ensure_connected  # type: ignore[method-assign]
    adapter._jsonrpc_request = _request  # type: ignore[method-assign]

    with pytest.raises(NetworkError, match="resources/list failed.*list failed"):
        await adapter.list_resources()
    with pytest.raises(NetworkError, match="resources/read failed.*read failed"):
        await adapter.read_resource("docs://one")


@pytest.mark.asyncio
async def test_websocket_adapter_resource_payloads_are_deep_copied() -> None:
    cfg = _server_config()
    adapter = WebSocketExternalMCPAdapter(cfg)
    list_response = {
        "result": {
            "resources": [
                {
                    "uri": "docs://one",
                    "metadata": {"nested": {"value": "original"}},
                }
            ]
        }
    }
    read_response = {
        "result": {
            "contents": [
                {
                    "uri": "docs://one",
                    "metadata": {"nested": {"value": "original"}},
                }
            ]
        }
    }

    async def _ensure_connected() -> None:
        return None

    async def _request(method: str, params: dict[str, object]) -> dict[str, object]:
        del params
        return list_response if method == "resources/list" else read_response

    adapter._ensure_connected = _ensure_connected  # type: ignore[method-assign]
    adapter._jsonrpc_request = _request  # type: ignore[method-assign]

    resources = await adapter.list_resources()
    resources[0]["metadata"]["nested"]["value"] = "changed"
    result = await adapter.read_resource("docs://one")
    result["contents"][0]["metadata"]["nested"]["value"] = "changed"

    assert list_response["result"]["resources"][0]["metadata"]["nested"]["value"] == "original"
    assert read_response["result"]["contents"][0]["metadata"]["nested"]["value"] == "original"


@pytest.mark.asyncio
async def test_websocket_adapter_request_timeout_clears_pending() -> None:
    cfg = _server_config(request_seconds=0.1)
    ws = _FakeWebSocket()
    ws.enqueue({"jsonrpc": "2.0", "id": 1, "result": {"serverInfo": {"name": "stub"}}})

    async def _connector(**kwargs):
        return ws

    adapter = WebSocketExternalMCPAdapter(cfg, ws_connector=_connector)
    try:
        await adapter.connect()
        with pytest.raises(TimeoutError, match="timed out"):
            await adapter.list_tools()
        assert adapter._pending == {}  # internal safety check for timeout cleanup
    finally:
        await adapter.close()


@pytest.mark.asyncio
async def test_websocket_adapter_correlates_out_of_order_responses() -> None:
    cfg = _server_config()
    ws = _FakeWebSocket()
    ws.enqueue({"jsonrpc": "2.0", "id": 1, "result": {"serverInfo": {"name": "stub"}}})

    async def _connector(**kwargs):
        return ws

    adapter = WebSocketExternalMCPAdapter(cfg, ws_connector=_connector)
    try:
        await adapter.connect()

        list_task = asyncio.create_task(adapter.list_tools())
        call_task = asyncio.create_task(adapter.call_tool("docs.search", {"q": "x"}))

        await _wait_for_sent(ws, expected_count=3)
        request_ids: dict[str, int] = {}
        for message in ws.sent_messages[1:3]:
            request_ids[message["method"]] = message["id"]

        ws.enqueue(
            {
                "jsonrpc": "2.0",
                "id": request_ids["tools/call"],
                "result": {"content": [{"type": "text", "text": "ok"}]},
            }
        )
        ws.enqueue(
            {
                "jsonrpc": "2.0",
                "id": request_ids["tools/list"],
                "result": {"tools": [{"name": "docs.search", "description": "Search docs"}]},
            }
        )

        tools = await list_task
        call_result = await call_task
        assert len(tools) == 1
        assert tools[0].name == "docs.search"
        assert call_result.is_error is False
        assert call_result.content == [{"type": "text", "text": "ok"}]
    finally:
        await adapter.close()


@pytest.mark.asyncio
async def test_websocket_adapter_uses_ephemeral_runtime_headers_without_mutating_base_session() -> None:
    cfg = _server_config()
    base_ws = _FakeWebSocket()
    runtime_ws = _FakeWebSocket()
    base_ws.enqueue({"jsonrpc": "2.0", "id": 1, "result": {"serverInfo": {"name": "base"}}})
    runtime_ws.enqueue({"jsonrpc": "2.0", "id": 1, "result": {"serverInfo": {"name": "runtime"}}})
    seen_headers: list[dict[str, str]] = []
    connect_count = 0

    async def _connector(*, url: str, subprotocols: list[str], headers: dict[str, str], connect_timeout: float):
        nonlocal connect_count
        del url, subprotocols, connect_timeout
        seen_headers.append(dict(headers))
        connect_count += 1
        if connect_count == 1:
            return base_ws
        return runtime_ws

    adapter = WebSocketExternalMCPAdapter(cfg, ws_connector=_connector)
    try:
        await adapter.connect()

        call_task = asyncio.create_task(
            adapter.call_tool(
                "docs.search",
                {"q": "runtime"},
                runtime_auth=BrokeredExternalCredential(
                    headers={"Authorization": "Bearer ephemeral-token"},
                ),
            )
        )
        await _wait_for_sent(runtime_ws, expected_count=2)
        runtime_request_id = runtime_ws.sent_messages[1]["id"]
        runtime_ws.enqueue(
            {
                "jsonrpc": "2.0",
                "id": runtime_request_id,
                "result": {"content": [{"type": "text", "text": "ok"}]},
            }
        )
        result = await call_task

        assert result.is_error is False
        assert connect_count == 2
        assert seen_headers[0] == {"x-client": "tldw"}
        assert seen_headers[1]["x-client"] == "tldw"
        assert seen_headers[1]["Authorization"] == "Bearer ephemeral-token"
        assert cfg.websocket is not None
        assert cfg.websocket.headers == {"x-client": "tldw"}
        assert adapter._ws is base_ws
        assert base_ws.sent_messages == [{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05", "clientInfo": {"name": "tldw_external_federation", "version": "0.1.0"}}}]
    finally:
        await adapter.close()


@pytest.mark.asyncio
async def test_websocket_adapter_prefers_runtime_authorization_header_over_static_auth(monkeypatch) -> None:
    monkeypatch.setenv("EXTERNAL_DOCS_TOKEN", "static-token")
    cfg = _server_config(
        auth=ExternalAuthConfig(mode=ExternalAuthMode.BEARER_ENV, token_env="EXTERNAL_DOCS_TOKEN")
    )
    base_ws = _FakeWebSocket()
    runtime_ws = _FakeWebSocket()
    base_ws.enqueue({"jsonrpc": "2.0", "id": 1, "result": {"serverInfo": {"name": "base"}}})
    runtime_ws.enqueue({"jsonrpc": "2.0", "id": 1, "result": {"serverInfo": {"name": "runtime"}}})
    seen_headers: list[dict[str, str]] = []
    connect_count = 0

    async def _connector(*, url: str, subprotocols: list[str], headers: dict[str, str], connect_timeout: float):
        nonlocal connect_count
        del url, subprotocols, connect_timeout
        connect_count += 1
        seen_headers.append(dict(headers))
        return base_ws if connect_count == 1 else runtime_ws

    adapter = WebSocketExternalMCPAdapter(cfg, ws_connector=_connector)
    try:
        await adapter.connect()

        call_task = asyncio.create_task(
            adapter.call_tool(
                "docs.search",
                {"q": "runtime"},
                runtime_auth=BrokeredExternalCredential(
                    headers={"Authorization": "Bearer runtime-token"},
                ),
            )
        )
        await _wait_for_sent(runtime_ws, expected_count=2)
        runtime_request_id = runtime_ws.sent_messages[1]["id"]
        runtime_ws.enqueue(
            {
                "jsonrpc": "2.0",
                "id": runtime_request_id,
                "result": {"content": [{"type": "text", "text": "ok"}]},
            }
        )
        result = await call_task

        assert result.is_error is False
        assert seen_headers[0]["Authorization"] == "Bearer static-token"
        assert seen_headers[1]["Authorization"] == "Bearer runtime-token"
    finally:
        await adapter.close()
