"""
HTTP batch endpoint tests for MCP Unified.
"""

import os as _os

import pytest

_os.environ.setdefault("TEST_MODE", "true")
_os.environ.setdefault("ENABLE_TRACING", "false")
_os.environ.setdefault("OTEL_METRICS_EXPORTER", "console")
_os.environ.setdefault("MCP_WS_AUTH_REQUIRED", "false")
_os.environ.setdefault("MCP_ALLOWED_IPS", "")

from fastapi.testclient import TestClient

from tldw_Server_API.app.main import app

client = TestClient(app)


def test_http_batch_initialize_and_ping():
    payload = [
        {"jsonrpc": "2.0", "method": "initialize", "params": {"clientInfo": {"name": "HTTP Batch"}}, "id": 1},
        {"jsonrpc": "2.0", "method": "ping", "id": 2},
    ]
    resp = client.post("/api/v1/mcp/request/batch", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    ids = sorted(item.get("id") for item in data)
    assert ids == [1, 2]
    for item in data:
        assert item.get("jsonrpc") == "2.0"
        assert "error" not in item


def test_http_batch_empty_returns_invalid_request():
    resp = client.post("/api/v1/mcp/request/batch", json=[])
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, dict)
    error = data.get("error")
    assert error is not None
    assert error.get("code") == -32600


def test_http_batch_non_array_returns_single_invalid_request_object():
    resp = client.post("/api/v1/mcp/request/batch", json={"jsonrpc": "2.0", "method": "ping", "id": 1})
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, dict)
    assert data["id"] is None
    assert data["error"]["code"] == -32600
    assert "result" not in data


@pytest.mark.parametrize("bad_id", [True, 1.5])
def test_http_batch_invalid_element_unsafe_id_returns_null_id(bad_id):
    resp = client.post(
        "/api/v1/mcp/request/batch",
        json=[{"jsonrpc": "2.0", "method": "ping", "params": {}, "id": bad_id}],
    )
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["id"] is None
    assert data[0]["error"]["code"] == -32600
    assert "result" not in data[0]


def test_http_batch_invalid_and_valid_items_preserve_response_order():
    resp = client.post(
        "/api/v1/mcp/request/batch",
        json=[
            {"jsonrpc": "2.0", "method": "ping", "params": {}, "id": True},
            {"jsonrpc": "2.0", "method": "ping", "id": "ordered-ping"},
        ],
    )
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert [item["id"] for item in data] == [None, "ordered-ping"]
    assert data[0]["error"]["code"] == -32600
    assert "error" not in data[1]


def test_http_batch_mixed_notification_and_request_omits_notification_item():
    resp = client.post(
        "/api/v1/mcp/request/batch",
        json=[
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "method": "ping", "id": "batch-ping"},
        ],
    )
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["id"] == "batch-ping"
    assert "result" in data[0]
    assert "error" not in data[0]


def test_http_batch_all_notifications_returns_204_empty_body():
    resp = client.post(
        "/api/v1/mcp/request/batch",
        json=[
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {}},
        ],
    )
    assert resp.status_code == 204
    assert resp.content == b""
