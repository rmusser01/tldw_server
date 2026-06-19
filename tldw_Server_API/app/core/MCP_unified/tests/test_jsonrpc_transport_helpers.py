import pytest

from tldw_Server_API.app.core.MCP_unified.jsonrpc_transport import (
    is_jsonrpc_keepalive,
    jsonrpc_payload_has_id,
    mcp_response_to_json,
    safe_jsonrpc_id,
)
from tldw_Server_API.app.core.MCP_unified.protocol import MCPError, MCPResponse


def test_serialize_success_omits_error_but_preserves_id():
    response = MCPResponse(result={"ok": True}, id="ok-1")
    assert mcp_response_to_json(response) == {
        "jsonrpc": "2.0",
        "id": "ok-1",
        "result": {"ok": True},
    }


def test_serialize_error_omits_null_data_and_preserves_null_id():
    response = MCPResponse(
        error=MCPError(code=-32700, message="Parse error"),
        id=None,
    )
    assert mcp_response_to_json(response) == {
        "jsonrpc": "2.0",
        "id": None,
        "error": {"code": -32700, "message": "Parse error"},
    }


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"jsonrpc": "2.0", "method": "ping"}, False),
        ({"jsonrpc": "2.0", "method": "ping", "id": None}, True),
    ],
)
def test_has_request_id_preserves_absent_vs_explicit_null(payload, expected):
    assert jsonrpc_payload_has_id(payload) is expected


@pytest.mark.parametrize("value", ["abc", 1, None])
def test_safe_jsonrpc_id_accepts_strings_integers_and_null(value):
    assert safe_jsonrpc_id(value) == value


@pytest.mark.parametrize("value", [True, False, 1.2, [], {}])
def test_safe_jsonrpc_id_rejects_unsafe_values(value):
    assert safe_jsonrpc_id(value) is None


@pytest.mark.parametrize("frame", [{"type": "ping"}, {"type": "pong"}])
def test_exact_keepalive_allowlist(frame):
    assert is_jsonrpc_keepalive(frame) is True


@pytest.mark.parametrize(
    "frame",
    [{"type": "ping", "id": 1}, {"type": "other"}, {"jsonrpc": "2.0"}],
)
def test_non_keepalive_frames_are_not_allowlisted(frame):
    assert is_jsonrpc_keepalive(frame) is False
