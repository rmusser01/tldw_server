from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from tldw_Server_API.app.core.MCP_unified.browser_cdp.client import (
    CDPBrowserClient,
    CDPClientConfig,
    CDPClientError,
    CDPPageTarget,
)


def test_cdp_client_requires_operator_configured_debugger_url() -> None:
    client = CDPBrowserClient(CDPClientConfig(debugger_url=None))

    with pytest.raises(CDPClientError) as exc_info:
        _ = client.debugger_base_url

    assert exc_info.value.reason_code == "cdp_not_configured"  # nosec B101


@pytest.mark.parametrize(
    "url, expected",
    [
        ("http://127.0.0.1:9222", "http://127.0.0.1:9222"),
        ("http://127.12.0.9:9222/", "http://127.12.0.9:9222"),
        ("http://localhost:9222", "http://localhost:9222"),
        ("http://[::1]:9222", "http://[::1]:9222"),
    ],
)
def test_cdp_client_accepts_loopback_debugger_urls(url: str, expected: str) -> None:
    client = CDPBrowserClient(CDPClientConfig(debugger_url=url))

    assert client.debugger_base_url == expected  # nosec B101


def test_cdp_client_rejects_non_loopback_debugger_url_by_default() -> None:
    client = CDPBrowserClient(CDPClientConfig(debugger_url="http://example.com:9222"))

    with pytest.raises(CDPClientError) as exc_info:
        _ = client.debugger_base_url

    assert exc_info.value.reason_code == "cdp_endpoint_not_allowed"  # nosec B101


def test_cdp_client_can_allow_operator_configured_non_loopback_url() -> None:
    client = CDPBrowserClient(
        CDPClientConfig(
            debugger_url="http://browser-host.internal:9222",
            allow_non_loopback=True,
        )
    )

    assert client.debugger_base_url == "http://browser-host.internal:9222"  # nosec B101


class _HTTPFakeClient(CDPBrowserClient):
    def __init__(self, responses: dict[str, Any] | None = None, *, fail: bool = False) -> None:
        super().__init__(CDPClientConfig(debugger_url="http://127.0.0.1:9222"))
        self.responses = responses or {}
        self.fail = fail
        self.paths: list[str] = []

    async def _get_json(self, path: str) -> Any:
        self.paths.append(path)
        if self.fail:
            raise RuntimeError("connection refused")
        return self.responses[path]


@pytest.mark.asyncio
async def test_cdp_client_normalizes_version_and_page_targets() -> None:
    client = _HTTPFakeClient(
        {
            "/json/version": {
                "Browser": "Chrome/126.0",
                "Protocol-Version": "1.3",
                "webSocketDebuggerUrl": "ws://127.0.0.1:9222/devtools/browser/1",
            },
            "/json/list": [
                {
                    "id": "page-1",
                    "type": "page",
                    "title": "App",
                    "url": "http://127.0.0.1:5173/",
                    "webSocketDebuggerUrl": "ws://127.0.0.1:9222/devtools/page/1",
                },
                {
                    "id": "worker-1",
                    "type": "service_worker",
                    "title": "Worker",
                    "url": "chrome-extension://abc/bg.js",
                    "webSocketDebuggerUrl": "ws://127.0.0.1:9222/devtools/worker/1",
                },
            ],
        }
    )

    version = await client.get_version()
    pages = await client.list_pages()

    assert client.paths == ["/json/version", "/json/list"]  # nosec B101
    assert version["browser"] == "Chrome/126.0"  # nosec B101
    assert version["protocol_version"] == "1.3"  # nosec B101
    assert [page.target_id for page in pages] == ["page-1"]  # nosec B101
    assert pages[0].title == "App"  # nosec B101
    assert pages[0].websocket_url == "ws://127.0.0.1:9222/devtools/page/1"  # nosec B101


@pytest.mark.asyncio
async def test_cdp_client_maps_http_failures_to_unreachable() -> None:
    client = _HTTPFakeClient(fail=True)

    with pytest.raises(CDPClientError) as exc_info:
        await client.get_version()

    assert exc_info.value.reason_code == "cdp_unreachable"  # nosec B101


class _FakeWebSocket:
    def __init__(self, messages: list[dict[str, Any] | BaseException]) -> None:
        self.messages = list(messages)
        self.sent: list[dict[str, Any]] = []

    async def __aenter__(self) -> _FakeWebSocket:
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None

    async def send(self, payload: str) -> None:
        self.sent.append(json.loads(payload))

    async def recv(self) -> str:
        if not self.messages:
            raise asyncio.TimeoutError
        message = self.messages.pop(0)
        if isinstance(message, BaseException):
            raise message
        return json.dumps(message)


class _WebSocketFakeClient(CDPBrowserClient):
    def __init__(self, websocket: _FakeWebSocket) -> None:
        super().__init__(CDPClientConfig(debugger_url="http://127.0.0.1:9222"))
        self.websocket = websocket

    def _connect_websocket(self, websocket_url: str) -> _FakeWebSocket:
        assert websocket_url == "ws://127.0.0.1:9222/devtools/page/1"  # nosec B101
        return self.websocket


def _page() -> CDPPageTarget:
    return CDPPageTarget(
        target_id="page-1",
        title="App",
        url="http://127.0.0.1:5173/",
        type="page",
        websocket_url="ws://127.0.0.1:9222/devtools/page/1",
    )


@pytest.mark.asyncio
async def test_cdp_client_sends_command_and_returns_matching_result() -> None:
    websocket = _FakeWebSocket(
        [
            {"method": "Runtime.consoleAPICalled", "params": {"type": "log"}},
            {"id": 1, "result": {"product": "Chrome/126.0"}},
        ]
    )
    client = _WebSocketFakeClient(websocket)

    result = await client.send_command(_page(), "Browser.getVersion")

    assert result == {"product": "Chrome/126.0"}  # nosec B101
    assert websocket.sent == [{"id": 1, "method": "Browser.getVersion", "params": {}}]  # nosec B101


@pytest.mark.asyncio
async def test_cdp_client_maps_protocol_errors() -> None:
    websocket = _FakeWebSocket(
        [
            {
                "id": 1,
                "error": {"code": -32601, "message": "No such method"},
            }
        ]
    )
    client = _WebSocketFakeClient(websocket)

    with pytest.raises(CDPClientError) as exc_info:
        await client.send_command(_page(), "Missing.method")

    assert exc_info.value.reason_code == "cdp_protocol_error"  # nosec B101


@pytest.mark.asyncio
async def test_cdp_client_observes_bounded_events() -> None:
    websocket = _FakeWebSocket(
        [
            {"id": 1, "result": {}},
            {"method": "Runtime.consoleAPICalled", "params": {"type": "log", "args": []}},
            {"method": "Runtime.consoleAPICalled", "params": {"type": "error", "args": []}},
        ]
    )
    client = _WebSocketFakeClient(websocket)

    observed = await client.observe_events(
        _page(),
        enable_methods=["Runtime.enable"],
        event_names={"Runtime.consoleAPICalled"},
        window_ms=100,
        max_events=1,
    )

    assert websocket.sent == [{"id": 1, "method": "Runtime.enable", "params": {}}]  # nosec B101
    assert observed["truncated"] is True  # nosec B101
    assert observed["events"] == [  # nosec B101
        {"method": "Runtime.consoleAPICalled", "params": {"type": "log", "args": []}}
    ]
