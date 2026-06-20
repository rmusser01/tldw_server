"""
JSON-RPC notification behavior tests for MCP Unified.
"""

import os

import pytest

# Minimize startup side-effects for tests (protocol-level only, but keep consistent)
os.environ.setdefault("TEST_MODE", "true")
os.environ.setdefault("ENABLE_TRACING", "false")
os.environ.setdefault("OTEL_METRICS_EXPORTER", "console")
from tldw_Server_API.app.core.MCP_unified.protocol import ErrorCode, MCPProtocol, MCPRequest, RequestContext


@pytest.mark.asyncio
async def test_notification_no_response():
    protocol = MCPProtocol()
    # Send a ping notification (no id) and ensure None response
    req = {"jsonrpc": "2.0", "method": "ping"}
    resp = await protocol.process_request(req, RequestContext(request_id="n-1", client_id="notif"))
    assert resp is None


@pytest.mark.asyncio
async def test_initialized_notification_no_response():
    protocol = MCPProtocol()
    req = {"jsonrpc": "2.0", "method": "notifications/initialized"}
    resp = await protocol.process_request(req, RequestContext(request_id="n-init", client_id="notif"))
    assert resp is None


@pytest.mark.asyncio
async def test_tool_call_notification_with_invalid_params_returns_no_response():
    protocol = MCPProtocol()
    req = {"jsonrpc": "2.0", "method": "tools/call", "params": {}}

    resp = await protocol.process_request(req, RequestContext(request_id="n-bad-params", client_id="notif"))

    assert resp is None


@pytest.mark.asyncio
async def test_notification_unknown_method_returns_no_response():
    protocol = MCPProtocol()
    req = {"jsonrpc": "2.0", "method": "missing/method"}

    resp = await protocol.process_request(req, RequestContext(request_id="n-missing-method", client_id="notif"))

    assert resp is None


@pytest.mark.asyncio
async def test_explicit_null_id_ping_returns_response():
    protocol = MCPProtocol()
    req = {"jsonrpc": "2.0", "method": "ping", "id": None}

    resp = await protocol.process_request(req, RequestContext(request_id="null-id", client_id="notif"))

    assert resp is not None
    assert resp.id is None
    assert resp.result is not None
    assert resp.result["pong"] is True


@pytest.mark.asyncio
async def test_mcp_request_explicit_null_id_ping_returns_response():
    protocol = MCPProtocol()
    req = MCPRequest(method="ping", id=None)

    resp = await protocol.process_request(req, RequestContext(request_id="model-null-id", client_id="notif"))

    assert resp is not None
    assert resp.id is None
    assert resp.result is not None
    assert resp.result["pong"] is True


@pytest.mark.asyncio
async def test_mcp_request_without_id_remains_notification():
    protocol = MCPProtocol()
    req = MCPRequest(method="ping")

    resp = await protocol.process_request(req, RequestContext(request_id="model-notif", client_id="notif"))

    assert resp is None


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_id", [True, 1.25])
async def test_invalid_request_id_returns_invalid_request_with_null_id(invalid_id):
    protocol = MCPProtocol()
    req = {"jsonrpc": "2.0", "method": "ping", "id": invalid_id}

    resp = await protocol.process_request(req, RequestContext(request_id="bad-id", client_id="notif"))

    assert resp is not None
    assert resp.id is None
    assert resp.error is not None
    assert resp.error.code == ErrorCode.INVALID_REQUEST


@pytest.mark.asyncio
async def test_batch_of_notifications_returns_none():
    protocol = MCPProtocol()
    # Two notifications (no ids) should yield None overall
    batch = [
        {"jsonrpc": "2.0", "method": "ping"},
        {"jsonrpc": "2.0", "method": "initialize", "params": {"clientInfo": {"name": "N Batch"}}},
    ]
    # initialize without id is also a notification
    resp = await protocol.process_request(batch, RequestContext(request_id="n-2", client_id="notif"))
    assert resp is None
