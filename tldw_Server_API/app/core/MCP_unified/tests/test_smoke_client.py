"""Unit tests for the standalone MCP smoke client core."""

from __future__ import annotations

import pytest


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
    assert summary["content_count"] == 1  # nosec B101
    assert len(rendered) < 1200  # nosec B101


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
