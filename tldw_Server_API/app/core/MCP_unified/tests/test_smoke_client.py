"""Unit tests for the standalone MCP smoke client core."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
import websockets

REPO_ROOT = Path(__file__).resolve().parents[5]
FIXTURE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "smoke_stdio_server.py"
)


def test_smoke_report_redacts_sensitive_details(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TLDW_SMOKE_TEST_API_KEY", "env-secret-value-12345")

    from mcp_unified.smoke.reporting import summarize_result

    summary = summarize_result(
        {
            "headers": {"authorization": "Bearer secret-token"},
            "env": {"TLDW_SMOKE_TEST_API_KEY": "env-secret-value-12345"},
            "path": "/Users/example/private/file.txt",
            "message": "token from env-secret-value-12345",
            "content": [{"type": "text", "text": "x" * 5000}],
        }
    )

    rendered = repr(summary)
    assert "secret-token" not in rendered  # nosec B101
    assert "Bearer secret-token" not in rendered  # nosec B101
    assert "env-secret-value-12345" not in rendered  # nosec B101
    assert "/Users/example" not in rendered  # nosec B101
    assert "x" * 500 not in rendered  # nosec B101
    assert summary["content"] == {  # nosec B101
        "summary": "[summarized content]",
        "item_count": 1,
    }
    assert len(rendered) < 1200  # nosec B101


def test_smoke_report_redacts_top_level_sensitive_summary_fields() -> None:
    from mcp_unified.smoke.reporting import summarize_result

    summary = summarize_result(
        {
            "arguments": {"query": "full user supplied tool arguments"},
            "content": "short full file contents",
            "env": {"CUSTOM_ENV": "raw-env-value"},
            "path": "/opt/app/data/file.txt",
            "authorization": "Bearer top-level-token",
        }
    )

    rendered = repr(summary)
    assert "full user supplied tool arguments" not in rendered  # nosec B101
    assert "short full file contents" not in rendered  # nosec B101
    assert "raw-env-value" not in rendered  # nosec B101
    assert "/opt/app/data/file.txt" not in rendered  # nosec B101
    assert "top-level-token" not in rendered  # nosec B101


def test_smoke_report_redacts_camel_case_content_and_argument_keys() -> None:
    from mcp_unified.smoke.reporting import summarize_result

    summary = summarize_result(
        {
            "structuredContent": {"documentText": "secret short body"},
            "fileContent": "file secret",
            "fileContents": "files secret",
            "contentBytes": "bytes secret",
            "toolArguments": {"q": "secret arg"},
            "toolArgs": {"q": "secret tool arg"},
        }
    )

    rendered = repr(summary)
    assert "secret short body" not in rendered  # nosec B101
    assert "file secret" not in rendered  # nosec B101
    assert "files secret" not in rendered  # nosec B101
    assert "bytes secret" not in rendered  # nosec B101
    assert "secret arg" not in rendered  # nosec B101
    assert "secret tool arg" not in rendered  # nosec B101


def test_smoke_report_summarizes_resource_contents_and_file_uris() -> None:
    from mcp_unified.smoke.reporting import summarize_result

    summary = summarize_result(
        {
            "contents": [
                {
                    "uri": "file:///Users/example/private.txt",
                    "mimeType": "text/plain",
                    "text": "short resource text body",
                },
                {
                    "uri": "file:///tmp/private.txt",
                    "mimeType": "application/octet-stream",
                    "blob": "base64-resource-body",
                },
            ]
        }
    )

    rendered = repr(summary)
    assert "short resource text body" not in rendered  # nosec B101
    assert "base64-resource-body" not in rendered  # nosec B101
    assert "file:///Users/example/private.txt" not in rendered  # nosec B101
    assert "file:///tmp/private.txt" not in rendered  # nosec B101
    assert summary["contents"] == {  # nosec B101
        "summary": "[summarized content]",
        "item_count": 2,
    }


def test_smoke_report_redacts_file_uri_paths() -> None:
    from mcp_unified.smoke.reporting import summarize_result

    summary = summarize_result(
        {
            "primary_uri": "file:///Users/example/private.txt",
            "tmp_uri": "file:///tmp/private.txt",
        }
    )

    rendered = repr(summary)
    assert "file:///Users/example/private.txt" not in rendered  # nosec B101
    assert "file:///tmp/private.txt" not in rendered  # nosec B101
    assert "/Users/example/private.txt" not in rendered  # nosec B101
    assert "/tmp/private.txt" not in rendered  # nosec B101 B108


def test_smoke_report_json_sanitizes_step_details() -> None:
    from mcp_unified.smoke.reporting import (
        SmokeReport,
        SmokeStepReport,
        SmokeTraceSummary,
        report_to_json,
    )

    report = SmokeReport(
        transport="stdio",
        steps=[
            SmokeStepReport(
                name="tools/call",
                ok=False,
                method="tools/call",
                request_id="smoke-1",
                detail={
                    "arguments": {"query": "full user supplied tool arguments"},
                    "error": "failed near /private/tmp/project/secret.txt",
                    "headers": {"Authorization": "Bearer report-token"},
                },
            )
        ],
        traces=[
            SmokeTraceSummary(
                request_id="smoke-1",
                method="tools/call",
                detail={"payload": "y" * 5000},
            )
        ],
    )

    payload = report_to_json(report)

    rendered = repr(payload)
    assert "full user supplied tool arguments" not in rendered  # nosec B101
    assert "report-token" not in rendered  # nosec B101
    assert "/private/tmp/project" not in rendered  # nosec B101
    assert "y" * 500 not in rendered  # nosec B101
    assert payload["ok"] is False  # nosec B101
    assert payload["steps"][0]["ok"] is False  # nosec B101


def test_smoke_report_json_summarizes_resource_read_details() -> None:
    from mcp_unified.smoke.reporting import SmokeReport, SmokeStepReport, report_to_json

    report = SmokeReport(
        transport="inprocess",
        steps=[
            SmokeStepReport(
                name="resources/read",
                ok=True,
                method="resources/read",
                detail={
                    "response": {
                        "contents": [
                            {
                                "uri": "file:///Users/example/private.txt",
                                "text": "short resource detail body",
                            },
                            {
                                "uri": "file:///tmp/private.txt",
                                "blob": "resource-detail-blob",
                            },
                        ]
                    }
                },
            )
        ],
    )

    payload = report_to_json(report)

    rendered = repr(payload)
    assert "short resource detail body" not in rendered  # nosec B101
    assert "resource-detail-blob" not in rendered  # nosec B101
    assert "file:///Users/example/private.txt" not in rendered  # nosec B101
    assert "file:///tmp/private.txt" not in rendered  # nosec B101


class _ExplodingItemsDict(dict):
    def __len__(self) -> int:
        return 1000

    def items(self):
        for index in range(1000):
            if index > 32:
                raise AssertionError("iterated beyond bounded mapping summary")
            yield f"k{index}", index


class _ExplodingList(list):
    def __len__(self) -> int:
        return 1000

    def __iter__(self):
        for index in range(1000):
            if index > 32:
                raise AssertionError("iterated beyond bounded sequence summary")
            yield index


def test_smoke_report_summaries_do_not_materialize_full_containers() -> None:
    from mcp_unified.smoke.reporting import summarize_result

    mapping_summary = summarize_result(_ExplodingItemsDict())
    sequence_summary = summarize_result({"items": _ExplodingList()})

    assert mapping_summary["omitted_keys"] > 0  # nosec B101
    assert sequence_summary["items"][-1]["omitted_items"] > 0  # nosec B101


class _RecordingTransport:
    def __init__(self, responses: list[object] | None = None) -> None:
        self.responses = list(responses or [])
        self.payloads: list[dict[str, object]] = []
        self.notifications: list[dict[str, object]] = []

    async def request(self, payload: dict[str, object] | list[object]) -> object | None:
        assert isinstance(payload, dict)  # nosec B101
        self.payloads.append(payload)
        if not self.responses:
            return None
        return self.responses.pop(0)

    async def notify(self, payload: dict[str, object]) -> None:
        self.notifications.append(payload)


@pytest.mark.asyncio
async def test_smoke_client_request_assigns_id_and_returns_result() -> None:
    from mcp_unified.smoke.client import McpSmokeClient

    transport = _RecordingTransport(
        [{"jsonrpc": "2.0", "id": "smoke-1", "result": {"pong": True}}]
    )
    client = McpSmokeClient(transport)

    result = await client.request("ping")

    assert result == {"pong": True}  # nosec B101
    assert transport.payloads[0] == {  # nosec B101
        "jsonrpc": "2.0",
        "id": "smoke-1",
        "method": "ping",
    }


@pytest.mark.asyncio
async def test_smoke_client_uses_stable_sequential_request_ids() -> None:
    from mcp_unified.smoke.client import McpSmokeClient

    transport = _RecordingTransport(
        [
            {"jsonrpc": "2.0", "id": "smoke-1", "result": {}},
            {"jsonrpc": "2.0", "id": "smoke-2", "result": {"tools": []}},
        ]
    )
    client = McpSmokeClient(transport)

    await client.ping()
    await client.list_tools()

    assert [payload["id"] for payload in transport.payloads] == [  # nosec B101
        "smoke-1",
        "smoke-2",
    ]
    assert [payload["method"] for payload in transport.payloads] == [  # nosec B101
        "ping",
        "tools/list",
    ]


@pytest.mark.asyncio
async def test_smoke_client_notification_has_no_id() -> None:
    from mcp_unified.smoke.client import McpSmokeClient

    transport = _RecordingTransport()
    client = McpSmokeClient(transport)

    await client.notify("notifications/initialized", {"ready": True})

    assert transport.notifications == [  # nosec B101
        {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {"ready": True},
        }
    ]


@pytest.mark.asyncio
async def test_smoke_client_helpers_send_expected_methods_and_params() -> None:
    from mcp_unified.smoke.client import McpSmokeClient

    transport = _RecordingTransport(
        [
            {"jsonrpc": "2.0", "id": "smoke-1", "result": {"serverInfo": {}}},
            {"jsonrpc": "2.0", "id": "smoke-2", "result": {"content": []}},
            {"jsonrpc": "2.0", "id": "smoke-3", "result": {"resources": []}},
            {"jsonrpc": "2.0", "id": "smoke-4", "result": {"contents": []}},
            {"jsonrpc": "2.0", "id": "smoke-5", "result": {"prompts": []}},
            {"jsonrpc": "2.0", "id": "smoke-6", "result": {"messages": []}},
        ]
    )
    client = McpSmokeClient(transport)

    await client.initialize(client_name="unit-client")
    await client.call_tool("echo.search", {"query": "hello"})
    await client.list_resources()
    await client.read_resource("resource://smoke/doc")
    await client.list_prompts()
    await client.get_prompt("smoke.review", {"topic": "client"})

    assert [payload["method"] for payload in transport.payloads] == [  # nosec B101
        "initialize",
        "tools/call",
        "resources/list",
        "resources/read",
        "prompts/list",
        "prompts/get",
    ]
    assert transport.payloads[0]["params"] == {  # nosec B101
        "clientInfo": {"name": "unit-client"},
        "capabilities": {},
    }
    assert transport.payloads[1]["params"] == {  # nosec B101
        "name": "echo.search",
        "arguments": {"query": "hello"},
    }
    assert transport.payloads[3]["params"] == {  # nosec B101
        "uri": "resource://smoke/doc"
    }
    assert transport.payloads[5]["params"] == {  # nosec B101
        "name": "smoke.review",
        "arguments": {"topic": "client"},
    }


@pytest.mark.asyncio
async def test_smoke_client_rejects_malformed_jsonrpc_responses() -> None:
    from mcp_unified.smoke.client import McpSmokeClient, McpSmokeClientError

    transport = _RecordingTransport([{"jsonrpc": "2.0", "id": "wrong", "result": {}}])
    client = McpSmokeClient(transport)

    with pytest.raises(McpSmokeClientError):
        await client.ping()


@pytest.mark.asyncio
async def test_inprocess_gateway_transport_runs_ping() -> None:
    from mcp_unified.smoke.client import McpSmokeClient
    from mcp_unified.smoke.fixtures import SmokeFixtureGatewayRuntime
    from mcp_unified.smoke.transports import InProcessGatewayTransport

    transport = InProcessGatewayTransport(SmokeFixtureGatewayRuntime())
    await transport.start()
    client = McpSmokeClient(transport)

    try:
        assert await client.ping() == {"pong": True}  # nosec B101
    finally:
        await transport.close()


@pytest.mark.asyncio
async def test_inprocess_gateway_transport_exposes_fixture_tools_resources_and_prompts() -> None:
    from mcp_unified.smoke.client import McpSmokeClient
    from mcp_unified.smoke.fixtures import SmokeFixtureGatewayRuntime
    from mcp_unified.smoke.transports import InProcessGatewayTransport

    client = McpSmokeClient(InProcessGatewayTransport(SmokeFixtureGatewayRuntime()))

    initialized = await client.initialize()
    tools = await client.list_tools()
    called = await client.call_tool("echo.search", {"query": "needle"})
    resources = await client.list_resources()
    resource = await client.read_resource("resource://smoke/doc")
    prompts = await client.list_prompts()
    prompt = await client.get_prompt("smoke.review", {"topic": "transport"})

    assert initialized["serverInfo"] == {  # nosec B101
        "name": "smoke-fixture-gateway",
        "version": "0.0-test",
    }
    assert initialized["capabilities"]["resources"]["available"] is True  # nosec B101
    assert initialized["capabilities"]["prompts"]["available"] is True  # nosec B101
    assert tools["tools"][0]["name"] == "echo.search"  # nosec B101
    assert called["content"][0]["text"] == "echo.search:needle"  # nosec B101
    assert resources["resources"][0]["uri"] == "resource://smoke/doc"  # nosec B101
    assert resource["contents"][0]["uri"] == "resource://smoke/doc"  # nosec B101
    assert prompts["prompts"][0]["name"] == "smoke.review"  # nosec B101
    assert prompt["messages"][0]["content"]["text"] == "Review transport"  # nosec B101


@pytest.mark.asyncio
async def test_inprocess_fastapi_transport_uses_asgi_request_path() -> None:
    from mcp_unified.gateway.fastapi import create_gateway_app
    from mcp_unified.smoke.client import McpSmokeClient
    from mcp_unified.smoke.fixtures import SmokeFixtureGatewayRuntime
    from mcp_unified.smoke.transports import InProcessFastApiTransport

    app = create_gateway_app(SmokeFixtureGatewayRuntime(), prefix="/mcp")
    transport = InProcessFastApiTransport(app, request_path="/mcp/request")
    await transport.start()
    client = McpSmokeClient(transport)

    try:
        assert await client.ping() == {"pong": True}  # nosec B101
    finally:
        await transport.close()


@pytest.mark.asyncio
async def test_live_http_transport_sends_profile_header() -> None:
    from mcp_unified.smoke.transports import LiveHttpTransport

    seen_headers: list[str | None] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen_headers.append(request.headers.get("x-mcp-profile"))
        return httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": "smoke-1",
                "result": {"pong": True},
            },
        )

    transport = LiveHttpTransport(
        "http://mcp.test/request",
        profile_id="reviewer",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    try:
        response = await transport.request(
            {"jsonrpc": "2.0", "id": "smoke-1", "method": "ping"}
        )
    finally:
        await transport.close()

    assert response == {  # nosec B101
        "jsonrpc": "2.0",
        "id": "smoke-1",
        "result": {"pong": True},
    }
    assert seen_headers == ["reviewer"]  # nosec B101


@pytest.mark.asyncio
async def test_live_http_transport_sends_auth_headers() -> None:
    from mcp_unified.smoke.transports import LiveHttpTransport

    seen_headers: list[tuple[str | None, str | None]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen_headers.append(
            (
                request.headers.get("authorization"),
                request.headers.get("x-api-key"),
            )
        )
        return httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": "smoke-1",
                "result": {"pong": True},
            },
        )

    auth_material = ("jwt-token", "api-key-value")
    transport = LiveHttpTransport(
        "http://mcp.test/request",
        bearer_token=auth_material[0],
        api_key=auth_material[1],
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    try:
        await transport.request({"jsonrpc": "2.0", "id": "smoke-1", "method": "ping"})
    finally:
        await transport.close()

    assert seen_headers == [(f"Bearer {auth_material[0]}", auth_material[1])]  # nosec B101


@pytest.mark.asyncio
async def test_live_http_transport_replaces_managed_headers_case_insensitively() -> None:
    from mcp_unified.smoke.transports import LiveHttpTransport

    seen_raw_headers: list[list[tuple[str, str]]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen_raw_headers.append(
            [
                (
                    name.decode("ascii").lower(),
                    value.decode("ascii"),
                )
                for name, value in request.headers.raw
            ]
        )
        return httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": "smoke-1",
                "result": {"pong": True},
            },
        )

    auth_material = ("new-jwt-token", "new-api-key")
    transport = LiveHttpTransport(
        "http://mcp.test/request",
        bearer_token=auth_material[0],
        api_key=auth_material[1],
        profile_id="reviewer",
        headers={
            "authorization": "Bearer stale-token",
            "X-API-Key": "stale-api-key",
            "x-MCP-profile": "stale-profile",
            "X-MCP-Profile-ID": "stale-profile-alias",
            "X-Custom-Smoke": "custom-value",
        },
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    try:
        await transport.request({"jsonrpc": "2.0", "id": "smoke-1", "method": "ping"})
    finally:
        await transport.close()

    raw_headers = seen_raw_headers[0]
    names = [name for name, _value in raw_headers]

    assert [  # nosec B101
        value for name, value in raw_headers if name == "authorization"
    ] == [f"Bearer {auth_material[0]}"]
    assert [value for name, value in raw_headers if name == "x-api-key"] == [  # nosec B101
        auth_material[1]
    ]
    assert [value for name, value in raw_headers if name == "x-mcp-profile"] == [  # nosec B101
        "reviewer"
    ]
    assert "x-mcp-profile-id" not in names  # nosec B101
    assert ("x-custom-smoke", "custom-value") in raw_headers  # nosec B101


@pytest.mark.asyncio
async def test_live_http_transport_preserves_managed_headers_without_replacements() -> None:
    from mcp_unified.smoke.transports import LiveHttpTransport

    seen_raw_headers: list[list[tuple[str, str]]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen_raw_headers.append(
            [
                (
                    name.decode("ascii").lower(),
                    value.decode("ascii"),
                )
                for name, value in request.headers.raw
            ]
        )
        return httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": "smoke-1",
                "result": {"pong": True},
            },
        )

    headers = {
        "authorization": "Bearer caller-token",
        "X-API-Key": "caller-api-key",
        "x-MCP-profile": "caller-profile",
        "X-MCP-Profile-ID": "caller-profile-id",
        "X-Custom-Smoke": "custom-value",
    }
    transport = LiveHttpTransport(
        "http://mcp.test/request",
        headers=headers,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    try:
        await transport.request({"jsonrpc": "2.0", "id": "smoke-1", "method": "ping"})
    finally:
        await transport.close()

    raw_headers = seen_raw_headers[0]

    assert ("authorization", headers["authorization"]) in raw_headers  # nosec B101
    assert ("x-api-key", headers["X-API-Key"]) in raw_headers  # nosec B101
    assert ("x-mcp-profile", headers["x-MCP-profile"]) in raw_headers  # nosec B101
    assert ("x-mcp-profile-id", headers["X-MCP-Profile-ID"]) in raw_headers  # nosec B101
    assert ("x-custom-smoke", headers["X-Custom-Smoke"]) in raw_headers  # nosec B101


@pytest.mark.asyncio
async def test_live_http_transport_treats_204_as_notification_success() -> None:
    from mcp_unified.smoke.transports import LiveHttpTransport

    payloads: list[dict[str, object]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads((await request.aread()).decode("utf-8")))
        return httpx.Response(204)

    transport = LiveHttpTransport(
        "http://mcp.test/request",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    try:
        await transport.notify(
            {"jsonrpc": "2.0", "method": "notifications/initialized"}
        )
    finally:
        await transport.close()

    assert payloads == [  # nosec B101
        {"jsonrpc": "2.0", "method": "notifications/initialized"}
    ]


@pytest.mark.asyncio
async def test_live_http_transport_parses_jsonrpc_response_body() -> None:
    from mcp_unified.smoke.client import McpSmokeClient
    from mcp_unified.smoke.transports import LiveHttpTransport

    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads((await request.aread()).decode("utf-8"))
        return httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": payload["id"],
                "result": {"tools": []},
            },
        )

    transport = LiveHttpTransport(
        "http://mcp.test/request",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    client = McpSmokeClient(transport)

    try:
        assert await client.list_tools() == {"tools": []}  # nosec B101
    finally:
        await transport.close()


@pytest.mark.asyncio
async def test_live_http_transport_wraps_connect_failures() -> None:
    from mcp_unified.smoke.transports import LiveHttpTransport, McpSmokeTransportError

    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    transport = LiveHttpTransport(
        "http://mcp.test/request",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    with pytest.raises(McpSmokeTransportError, match="transport_http_request_failed"):
        await transport.request({"jsonrpc": "2.0", "id": "smoke-1", "method": "ping"})

    await transport.close()


@pytest.mark.asyncio
async def test_live_http_retry_skips_transmitted_tool_call_on_5xx() -> None:
    from mcp_unified.smoke.transports import LiveHttpTransport, McpSmokeTransportError

    received_payloads: list[dict[str, object]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        received_payloads.append(json.loads((await request.aread()).decode("utf-8")))
        return httpx.Response(503, json={"error": "unavailable"})

    transport = LiveHttpTransport(
        "http://mcp.test/request",
        max_retries=1,
        retry_methods={"ping"},
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    with pytest.raises(
        McpSmokeTransportError,
        match="transport_retry_skipped_non_idempotent",
    ):
        await transport.request(
            {
                "jsonrpc": "2.0",
                "id": "smoke-1",
                "method": "tools/call",
                "params": {"name": "echo.search", "arguments": {"query": "needle"}},
            }
        )

    await transport.close()

    assert len(received_payloads) == 1  # nosec B101


@pytest.mark.asyncio
async def test_live_http_retry_skips_transmitted_tool_call_on_disconnect() -> None:
    from mcp_unified.smoke.transports import LiveHttpTransport, McpSmokeTransportError

    received_payloads: list[dict[str, object]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        received_payloads.append(json.loads((await request.aread()).decode("utf-8")))
        raise httpx.RemoteProtocolError("server disconnected", request=request)

    transport = LiveHttpTransport(
        "http://mcp.test/request",
        max_retries=1,
        retry_methods={"ping"},
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    with pytest.raises(
        McpSmokeTransportError,
        match="transport_retry_skipped_non_idempotent",
    ):
        await transport.request(
            {
                "jsonrpc": "2.0",
                "id": "smoke-1",
                "method": "tools/call",
                "params": {"name": "echo.search", "arguments": {"query": "needle"}},
            }
        )

    await transport.close()

    assert len(received_payloads) == 1  # nosec B101


@pytest.mark.asyncio
async def test_live_http_retry_replays_configured_idempotent_method() -> None:
    from mcp_unified.smoke.transports import LiveHttpTransport

    received_payloads: list[dict[str, object]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads((await request.aread()).decode("utf-8"))
        received_payloads.append(payload)
        if len(received_payloads) == 1:
            return httpx.Response(503, json={"error": "temporarily unavailable"})
        return httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": payload["id"],
                "result": {"pong": True},
            },
        )

    transport = LiveHttpTransport(
        "http://mcp.test/request",
        max_retries=1,
        retry_methods={"ping"},
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    try:
        response = await transport.request(
            {"jsonrpc": "2.0", "id": "smoke-1", "method": "ping"}
        )
    finally:
        await transport.close()

    assert response == {  # nosec B101
        "jsonrpc": "2.0",
        "id": "smoke-1",
        "result": {"pong": True},
    }
    assert [payload["method"] for payload in received_payloads] == [  # nosec B101
        "ping",
        "ping",
    ]


@pytest.mark.asyncio
async def test_live_websocket_transport_correlates_out_of_order_responses() -> None:
    from mcp_unified.smoke.client import McpSmokeClient
    from mcp_unified.smoke.transports import LiveWebSocketTransport

    async def handler(websocket) -> None:
        first = json.loads(await websocket.recv())
        second = json.loads(await websocket.recv())

        await websocket.send(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": second["id"],
                    "result": {"method": second["method"]},
                }
            )
        )
        await websocket.send(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": first["id"],
                    "result": {"method": first["method"]},
                }
            )
        )

    async with websockets.serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        transport = LiveWebSocketTransport(f"ws://127.0.0.1:{port}/mcp")
        client = McpSmokeClient(transport)

        try:
            first_task = asyncio.create_task(client.request("tools/list"))
            second_task = asyncio.create_task(client.request("ping"))
            first_result, second_result = await asyncio.gather(first_task, second_task)
        finally:
            await transport.close()

    assert first_result == {"method": "tools/list"}  # nosec B101
    assert second_result == {"method": "ping"}  # nosec B101


@pytest.mark.asyncio
async def test_live_websocket_transport_suppresses_server_notifications() -> None:
    from mcp_unified.smoke.client import McpSmokeClient
    from mcp_unified.smoke.transports import LiveWebSocketTransport

    received_payloads: list[dict[str, object]] = []

    async def handler(websocket) -> None:
        notification = json.loads(await websocket.recv())
        received_payloads.append(notification)
        await websocket.send(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "method": "notifications/progress",
                    "params": {"progress": 1},
                }
            )
        )

        ping = json.loads(await websocket.recv())
        received_payloads.append(ping)
        await websocket.send(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": ping["id"],
                    "result": {"pong": True},
                }
            )
        )

    async with websockets.serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        transport = LiveWebSocketTransport(f"ws://127.0.0.1:{port}/mcp")
        client = McpSmokeClient(transport)

        try:
            await client.notify("notifications/initialized")
            ping_result = await client.ping()
        finally:
            await transport.close()

    assert ping_result == {"pong": True}  # nosec B101
    assert received_payloads == [  # nosec B101
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": "smoke-1", "method": "ping"},
    ]


@pytest.mark.asyncio
async def test_live_websocket_transport_rejects_malformed_frames() -> None:
    from mcp_unified.smoke.transports import (
        LiveWebSocketTransport,
        McpSmokeTransportError,
    )

    async def handler(websocket) -> None:
        await websocket.recv()
        await websocket.send("not-json")

    async with websockets.serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        transport = LiveWebSocketTransport(f"ws://127.0.0.1:{port}/mcp")

        try:
            with pytest.raises(
                McpSmokeTransportError,
                match="transport_invalid_json_response",
            ):
                await transport.request(
                    {"jsonrpc": "2.0", "id": "smoke-1", "method": "ping"}
                )
        finally:
            await transport.close()


@pytest.mark.asyncio
async def test_live_websocket_transport_adds_profile_header_and_auth_headers() -> None:
    from mcp_unified.smoke.transports import LiveWebSocketTransport

    bearer_value = "jwt-token"
    api_key_value = "api-key-value"
    seen_path: str | None = None
    seen_authorization: str | None = None
    seen_api_key: str | None = None
    seen_profile: str | None = None
    seen_profile_alias: str | None = None

    async def handler(websocket) -> None:
        nonlocal seen_path, seen_authorization, seen_api_key, seen_profile
        nonlocal seen_profile_alias
        seen_path = websocket.request.path
        seen_authorization = websocket.request.headers.get("authorization")
        seen_api_key = websocket.request.headers.get("x-api-key")
        seen_profile = websocket.request.headers.get("x-mcp-profile")
        seen_profile_alias = websocket.request.headers.get("x-mcp-profile-id")
        payload = json.loads(await websocket.recv())
        await websocket.send(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": payload["id"],
                    "result": {"pong": True},
                }
            )
        )

    async with websockets.serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        transport = LiveWebSocketTransport(
            f"ws://127.0.0.1:{port}/mcp?client_id=smoke",
            bearer_token=bearer_value,
            api_key=api_key_value,
            profile_id="reviewer",
            headers={
                "x-MCP-profile": "stale-profile",
                "X-MCP-Profile-ID": "stale-profile-alias",
                "X-Custom-Smoke": "custom-value",
            },
        )

        try:
            response = await transport.request(
                {"jsonrpc": "2.0", "id": "smoke-1", "method": "ping"}
            )
        finally:
            await transport.close()

    assert response == {  # nosec B101
        "jsonrpc": "2.0",
        "id": "smoke-1",
        "result": {"pong": True},
    }
    assert seen_authorization == f"Bearer {bearer_value}"  # nosec B101
    assert seen_api_key == api_key_value  # nosec B101
    assert seen_profile == "reviewer"  # nosec B101
    assert seen_profile_alias is None  # nosec B101
    assert seen_path is not None  # nosec B101
    query = parse_qs(urlsplit(seen_path).query)
    assert query["client_id"] == ["smoke"]  # nosec B101
    assert "profile" not in query  # nosec B101
    assert "profile_id" not in query  # nosec B101


@pytest.mark.asyncio
async def test_live_websocket_pending_registration_rolls_back_batch_conflicts() -> None:
    from mcp_unified.smoke.transports import (
        LiveWebSocketTransport,
        McpSmokeTransportError,
    )

    transport = LiveWebSocketTransport("ws://127.0.0.1/mcp")
    transport._register_pending(["existing"])

    with pytest.raises(
        McpSmokeTransportError,
        match="transport_duplicate_request_id",
    ):
        transport._register_pending(["new", "existing"])

    assert "existing" in transport._pending  # nosec B101
    assert "new" not in transport._pending  # nosec B101


@pytest.mark.asyncio
async def test_stdio_subprocess_transport_exchanges_object_payloads() -> None:
    from mcp_unified.smoke.client import McpSmokeClient
    from mcp_unified.smoke.transports import StdioSubprocessTransport

    transport = StdioSubprocessTransport(
        command=sys.executable,
        args=[str(FIXTURE_PATH)],
        cwd=str(REPO_ROOT),
        env_allowlist=["PYTHONPATH"],
    )
    client = McpSmokeClient(transport)

    try:
        initialized = await client.initialize()
        ping = await client.ping()
        tools = await client.list_tools()
    finally:
        await transport.close()

    assert initialized["serverInfo"]["name"] == "smoke-stdio-fixture"  # nosec B101
    assert ping == {"pong": True}  # nosec B101
    assert tools["tools"][0]["name"] == "echo.search"  # nosec B101


@pytest.mark.asyncio
async def test_stdio_subprocess_transport_batch_suppresses_notification_response() -> None:
    from mcp_unified.smoke.transports import StdioSubprocessTransport

    transport = StdioSubprocessTransport(
        command=sys.executable,
        args=[str(FIXTURE_PATH)],
        cwd=str(REPO_ROOT),
        env_allowlist=["PYTHONPATH"],
    )

    try:
        response = await transport.request(
            [
                {"jsonrpc": "2.0", "method": "notifications/initialized"},
                {"jsonrpc": "2.0", "id": "smoke-batch-ping", "method": "ping"},
            ]
        )
    finally:
        await transport.close()

    assert response == [  # nosec B101
        {
            "jsonrpc": "2.0",
            "id": "smoke-batch-ping",
            "result": {"pong": True},
        }
    ]


@pytest.mark.asyncio
async def test_stdio_subprocess_transport_ignores_server_notifications_before_response() -> None:
    from mcp_unified.smoke.transports import StdioSubprocessTransport

    transport = StdioSubprocessTransport(
        command=sys.executable,
        args=[str(FIXTURE_PATH)],
        cwd=str(REPO_ROOT),
        env_allowlist=["PYTHONPATH"],
    )

    try:
        response = await transport.request(
            {
                "jsonrpc": "2.0",
                "id": "smoke-notification",
                "method": "smoke/server-notification-before-response",
            }
        )
    finally:
        await transport.close()

    assert response == {  # nosec B101
        "jsonrpc": "2.0",
        "id": "smoke-notification",
        "result": {"after": "notification"},
    }


@pytest.mark.asyncio
async def test_stdio_subprocess_transport_rejects_wrong_response_id() -> None:
    from mcp_unified.smoke.transports import (
        McpSmokeTransportError,
        StdioSubprocessTransport,
    )

    transport = StdioSubprocessTransport(
        command=sys.executable,
        args=[str(FIXTURE_PATH)],
        cwd=str(REPO_ROOT),
        env_allowlist=["PYTHONPATH"],
    )

    with pytest.raises(
        McpSmokeTransportError,
        match="transport_unexpected_stdio_response",
    ):
        await transport.request(
            {
                "jsonrpc": "2.0",
                "id": "smoke-expected-id",
                "method": "smoke/wrong-id-response",
            }
        )

    assert transport._process is None  # nosec B101


@pytest.mark.asyncio
async def test_stdio_subprocess_transport_close_terminates_process() -> None:
    from mcp_unified.smoke.transports import StdioSubprocessTransport

    transport = StdioSubprocessTransport(
        command=sys.executable,
        args=[str(FIXTURE_PATH)],
        cwd=str(REPO_ROOT),
        env_allowlist=["PYTHONPATH"],
    )
    await transport.start()
    process = transport._process

    await transport.close()

    assert process is not None  # nosec B101
    await asyncio.wait_for(process.wait(), timeout=1.0)
    assert process.returncode is not None  # nosec B101


@pytest.mark.asyncio
async def test_stdio_subprocess_transport_redacts_secret_stderr_on_error() -> None:
    from mcp_unified.smoke.transports import (
        McpSmokeTransportError,
        StdioSubprocessTransport,
    )

    transport = StdioSubprocessTransport(
        command=sys.executable,
        args=[str(FIXTURE_PATH)],
        cwd=str(REPO_ROOT),
        env_allowlist=["PYTHONPATH"],
    )

    with pytest.raises(McpSmokeTransportError) as caught:
        await transport.request(
            {"jsonrpc": "2.0", "id": "smoke-secret", "method": "smoke/secret-stderr"}
        )

    await transport.close()

    rendered_error = str(caught.value)
    assert "stdio-secret-token" not in rendered_error  # nosec B101
    assert "Bearer stdio-secret-token" not in rendered_error  # nosec B101
    assert "stderr=" in rendered_error  # nosec B101


@pytest.mark.asyncio
async def test_stdio_subprocess_transport_duplicate_batch_ids_do_not_start_process() -> None:
    from mcp_unified.smoke.transports import (
        McpSmokeTransportError,
        StdioSubprocessTransport,
    )

    transport = StdioSubprocessTransport(
        command=sys.executable,
        args=[str(FIXTURE_PATH)],
        cwd=str(REPO_ROOT),
        env_allowlist=["PYTHONPATH"],
    )

    with pytest.raises(
        McpSmokeTransportError,
        match="transport_duplicate_request_id",
    ):
        await transport.request(
            [
                {"jsonrpc": "2.0", "id": "duplicate", "method": "ping"},
                {"jsonrpc": "2.0", "id": "duplicate", "method": "tools/list"},
            ]
        )

    assert transport._process is None  # nosec B101


@pytest.mark.asyncio
async def test_stdio_subprocess_transport_timeout_closes_process() -> None:
    from mcp_unified.smoke.transports import (
        McpSmokeTransportError,
        StdioSubprocessTransport,
    )

    transport = StdioSubprocessTransport(
        command=sys.executable,
        args=[str(FIXTURE_PATH)],
        cwd=str(REPO_ROOT),
        env_allowlist=["PYTHONPATH"],
        request_timeout=0.05,
    )

    with pytest.raises(
        McpSmokeTransportError,
        match="transport_stdio_response_timeout",
    ):
        await transport.request(
            {"jsonrpc": "2.0", "id": "smoke-timeout", "method": "smoke/hang"}
        )

    assert transport._process is None  # nosec B101


@pytest.mark.asyncio
async def test_stdio_subprocess_transport_exited_process_request_clears_state() -> None:
    from mcp_unified.smoke.transports import (
        McpSmokeTransportError,
        StdioSubprocessTransport,
    )

    transport = StdioSubprocessTransport(
        command=sys.executable,
        args=[str(FIXTURE_PATH)],
        cwd=str(REPO_ROOT),
        env_allowlist=["PYTHONPATH"],
    )

    await transport.start()
    process = transport._process
    assert process is not None  # nosec B101
    process.terminate()
    await asyncio.wait_for(process.wait(), timeout=1.0)

    with pytest.raises(
        McpSmokeTransportError,
        match="transport_stdio_process_exited",
    ):
        await transport.request({"jsonrpc": "2.0", "id": "after-exit", "method": "ping"})

    assert transport._process is None  # nosec B101


@pytest.mark.asyncio
async def test_baseline_scenario_passes_in_best_effort_mode() -> None:
    from mcp_unified.smoke.fixtures import SmokeFixtureGatewayRuntime
    from mcp_unified.smoke.scenarios import run_baseline_scenario
    from mcp_unified.smoke.transports import InProcessGatewayTransport

    report = await run_baseline_scenario(
        InProcessGatewayTransport(SmokeFixtureGatewayRuntime(include_denied_tool=True)),
        mode="best_effort",
    )

    assert report.ok is True  # nosec B101
    assert [step.name for step in report.steps] == [  # nosec B101
        "initialize",
        "notifications/initialized",
        "ping",
        "tools/list",
        "tools/call",
        "tools/call:unknown",
        "profile-filtered visibility",
        "resources",
        "prompts",
        "json-rpc batch",
        "malformed request",
        "policy denial",
    ]


@pytest.mark.asyncio
async def test_baseline_scenario_requires_followup_ping_after_initialized_notification() -> None:
    from mcp_unified.smoke.scenarios import run_baseline_scenario

    class _NotificationOnlyTransport:
        async def start(self) -> None:
            return None

        async def request(
            self,
            payload: dict[str, object] | list[object],
        ) -> object | None:
            assert isinstance(payload, dict)  # nosec B101
            if payload["method"] == "initialize":
                return {
                    "jsonrpc": "2.0",
                    "id": payload["id"],
                    "result": {
                        "capabilities": {"tools": {"available": True}},
                        "serverInfo": {"name": "unit", "version": "0"},
                    },
                }
            return {
                "jsonrpc": "2.0",
                "id": payload["id"],
                "error": {"code": -32601, "message": "ping unavailable"},
            }

        async def notify(self, payload: dict[str, object]) -> None:
            return None

        async def close(self) -> None:
            return None

    report = await run_baseline_scenario(_NotificationOnlyTransport(), mode="best_effort")

    notification_step = next(
        step for step in report.steps if step.name == "notifications/initialized"
    )
    assert report.ok is False  # nosec B101
    assert notification_step.ok is False  # nosec B101
    assert notification_step.reason_code == "followup_ping_failed"  # nosec B101


@pytest.mark.asyncio
async def test_baseline_scenario_skips_unavailable_resources_and_prompts_in_best_effort() -> None:
    from mcp_unified.smoke.scenarios import run_baseline_scenario

    class _NoOptionalCapabilityTransport:
        async def start(self) -> None:
            return None

        async def request(
            self,
            payload: dict[str, object] | list[object],
        ) -> object | None:
            if isinstance(payload, list):
                return [
                    {
                        "jsonrpc": "2.0",
                        "id": item["id"],
                        "result": {"pong": True} if item["method"] == "ping" else {"tools": []},
                    }
                    for item in payload
                    if isinstance(item, dict) and "id" in item
                ]
            assert isinstance(payload, dict)  # nosec B101
            method = payload.get("method")
            if method == "initialize":
                return {
                    "jsonrpc": "2.0",
                    "id": payload["id"],
                    "result": {
                        "capabilities": {
                            "tools": {"available": True},
                            "resources": {"available": False},
                            "prompts": {"available": False},
                        },
                        "serverInfo": {"name": "unit", "version": "0"},
                    },
                }
            if method == "ping":
                return {"jsonrpc": "2.0", "id": payload["id"], "result": {"pong": True}}
            if method == "tools/list":
                return {"jsonrpc": "2.0", "id": payload["id"], "result": {"tools": []}}
            if method == "tools/call":
                return {
                    "jsonrpc": "2.0",
                    "id": payload["id"],
                    "error": {"code": -32601, "message": "Method not found"},
                }
            return {
                "jsonrpc": "2.0",
                "id": payload.get("id"),
                "error": {"code": -32600, "message": "Invalid request"},
            }

        async def notify(self, payload: dict[str, object]) -> None:
            return None

        async def close(self) -> None:
            return None

    report = await run_baseline_scenario(
        _NoOptionalCapabilityTransport(),
        mode="best_effort",
    )
    steps = {step.name: step for step in report.steps}

    assert report.ok is True  # nosec B101
    assert steps["resources"].reason_code == "capability_unavailable"  # nosec B101
    assert steps["prompts"].reason_code == "capability_unavailable"  # nosec B101


@pytest.mark.asyncio
async def test_baseline_scenario_fails_unadvertised_resources_in_strict_mode() -> None:
    from mcp_unified.smoke.scenarios import run_baseline_scenario

    class _NoResourceCapabilityTransport:
        async def start(self) -> None:
            return None

        async def request(self, payload: dict[str, object] | list[object]) -> object | None:
            assert isinstance(payload, dict)  # nosec B101
            if payload["method"] == "initialize":
                return {
                    "jsonrpc": "2.0",
                    "id": payload["id"],
                    "result": {
                        "capabilities": {
                            "tools": {"available": True},
                            "resources": {"available": False},
                        },
                        "serverInfo": {"name": "unit", "version": "0"},
                    },
                }
            return {"jsonrpc": "2.0", "id": payload["id"], "result": {"pong": True}}

        async def notify(self, payload: dict[str, object]) -> None:
            return None

        async def close(self) -> None:
            return None

    report = await run_baseline_scenario(
        _NoResourceCapabilityTransport(),
        mode="strict",
    )
    resources_step = next(step for step in report.steps if step.name == "resources")

    assert report.ok is False  # nosec B101
    assert resources_step.ok is False  # nosec B101
    assert resources_step.reason_code == "required_capability_unavailable"  # nosec B101


@pytest.mark.asyncio
async def test_baseline_scenario_rejects_malformed_batch_items() -> None:
    from mcp_unified.smoke.fixtures import SmokeFixtureGatewayRuntime
    from mcp_unified.smoke.scenarios import run_baseline_scenario
    from mcp_unified.smoke.transports import InProcessGatewayTransport

    class _MalformedBatchTransport(InProcessGatewayTransport):
        async def request(
            self,
            payload: dict[str, object] | list[object],
        ) -> object | None:
            if isinstance(payload, list):
                return [
                    {"id": "smoke-batch-ping"},
                    {"id": "smoke-batch-tools"},
                ]
            return await super().request(payload)

    report = await run_baseline_scenario(
        _MalformedBatchTransport(SmokeFixtureGatewayRuntime()),
        mode="best_effort",
    )
    batch_step = next(step for step in report.steps if step.name == "json-rpc batch")

    assert report.ok is False  # nosec B101
    assert batch_step.ok is False  # nosec B101
    assert batch_step.reason_code == "invalid_batch_item"  # nosec B101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("batch_response", "expected_reason_code"),
    [
        (
            [
                {"jsonrpc": "2.0", "id": "smoke-batch-ping", "result": {"pong": True}},
                {"jsonrpc": "2.0", "id": "smoke-batch-tools", "result": {"tools": []}},
                "malformed-extra-item",
            ],
            "invalid_batch_response_count",
        ),
        (
            [
                {"jsonrpc": "2.0", "id": "smoke-batch-ping", "result": {"pong": True}},
                {"jsonrpc": "2.0", "id": "smoke-batch-tools", "result": {"tools": []}},
                {"jsonrpc": "2.0", "id": "smoke-batch-extra", "result": {}},
            ],
            "invalid_batch_response_count",
        ),
        (
            [
                {"id": "smoke-batch-ping"},
                {"jsonrpc": "2.0", "id": "smoke-batch-ping", "result": {"pong": True}},
                {"jsonrpc": "2.0", "id": "smoke-batch-tools", "result": {"tools": []}},
            ],
            "invalid_batch_response_count",
        ),
        (
            [
                {"jsonrpc": "2.0", "id": "smoke-batch-ping", "result": {"pong": True}},
                {"jsonrpc": "2.0", "id": "smoke-batch-ping", "result": {"pong": True}},
            ],
            "duplicate_batch_id",
        ),
    ],
)
async def test_baseline_scenario_rejects_batch_extras_and_duplicate_ids(
    batch_response: list[object],
    expected_reason_code: str,
) -> None:
    from mcp_unified.smoke.fixtures import SmokeFixtureGatewayRuntime
    from mcp_unified.smoke.scenarios import run_baseline_scenario
    from mcp_unified.smoke.transports import InProcessGatewayTransport

    class _MalformedBatchTransport(InProcessGatewayTransport):
        async def request(
            self,
            payload: dict[str, object] | list[object],
        ) -> object | None:
            if isinstance(payload, list):
                return batch_response
            return await super().request(payload)

    report = await run_baseline_scenario(
        _MalformedBatchTransport(SmokeFixtureGatewayRuntime()),
        mode="best_effort",
    )
    batch_step = next(step for step in report.steps if step.name == "json-rpc batch")

    assert report.ok is False  # nosec B101
    assert batch_step.ok is False  # nosec B101
    assert batch_step.reason_code == expected_reason_code  # nosec B101


@pytest.mark.asyncio
async def test_baseline_scenario_preserves_empty_safe_tool_and_prompt_arguments() -> None:
    from mcp_unified.smoke.fixtures import SmokeFixtureGatewayRuntime
    from mcp_unified.smoke.scenarios import run_baseline_scenario
    from mcp_unified.smoke.transports import InProcessGatewayTransport

    runtime = SmokeFixtureGatewayRuntime()

    report = await run_baseline_scenario(
        InProcessGatewayTransport(runtime),
        mode="best_effort",
        safe_tool_arguments={},
        safe_prompt_arguments={},
    )

    safe_tool_call = next(
        request for request in runtime.call_requests if request[0] == "echo.search"
    )
    assert report.ok is True  # nosec B101
    assert safe_tool_call[1] == {}  # nosec B101
    assert runtime.prompt_gets[0][1] == {}  # nosec B101


@pytest.mark.asyncio
async def test_baseline_scenario_redacts_direct_client_error_details() -> None:
    from mcp_unified.smoke.scenarios import run_baseline_scenario

    class _SensitiveInitializeErrorTransport:
        async def start(self) -> None:
            return None

        async def request(
            self,
            payload: dict[str, object] | list[object],
        ) -> object | None:
            assert isinstance(payload, dict)  # nosec B101
            return {
                "jsonrpc": "2.0",
                "id": payload["id"],
                "error": {
                    "code": -32603,
                    "message": "Bearer secret-token failed near /Users/example/private.txt",
                    "data": {
                        "arguments": {"query": "full user supplied tool arguments"},
                        "authorization": "Bearer nested-secret",
                    },
                },
            }

        async def notify(self, payload: dict[str, object]) -> None:
            return None

        async def close(self) -> None:
            return None

    report = await run_baseline_scenario(
        _SensitiveInitializeErrorTransport(),
        mode="best_effort",
    )
    rendered_detail = repr(report.steps[0].detail)

    assert report.steps[0].ok is False  # nosec B101
    assert "secret-token" not in rendered_detail  # nosec B101
    assert "nested-secret" not in rendered_detail  # nosec B101
    assert "/Users/example" not in rendered_detail  # nosec B101
    assert "full user supplied tool arguments" not in rendered_detail  # nosec B101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("target_id", "error_code", "step_name"),
    [
        ("smoke-unknown-tool", -32601, "tools/call:unknown"),
        ("smoke-malformed", -32600, "malformed request"),
        ("smoke-policy-denial", -32001, "policy denial"),
    ],
)
async def test_baseline_scenario_rejects_malformed_error_envelopes(
    target_id: str,
    error_code: int,
    step_name: str,
) -> None:
    from mcp_unified.smoke.fixtures import SmokeFixtureGatewayRuntime
    from mcp_unified.smoke.scenarios import run_baseline_scenario
    from mcp_unified.smoke.transports import InProcessGatewayTransport

    class _MalformedErrorEnvelopeTransport(InProcessGatewayTransport):
        async def request(
            self,
            payload: dict[str, object] | list[object],
        ) -> object | None:
            if isinstance(payload, dict) and payload.get("id") == target_id:
                return {"error": {"code": error_code}}
            return await super().request(payload)

    report = await run_baseline_scenario(
        _MalformedErrorEnvelopeTransport(
            SmokeFixtureGatewayRuntime(include_denied_tool=True)
        ),
        mode="best_effort",
    )
    target_step = next(step for step in report.steps if step.name == step_name)

    assert report.ok is False  # nosec B101
    assert target_step.ok is False  # nosec B101
    assert target_step.reason_code == "malformed_error_envelope"  # nosec B101
