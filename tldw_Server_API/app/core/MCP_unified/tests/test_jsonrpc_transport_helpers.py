import pytest

from tldw_Server_API.app.core.MCP_unified.jsonrpc_transport import (
    invalid_request_response,
    is_jsonrpc_keepalive,
    jsonrpc_payload_has_id,
    mcp_response_to_json,
    mcp_responses_to_json,
    parse_error_response,
    parse_jsonrpc_body,
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


def test_mcp_responses_to_json_serializes_response_list():
    responses = [
        MCPResponse(result={"ok": True}, id="ok-1"),
        MCPResponse(error=MCPError(code=-32600, message="Invalid request"), id=2),
    ]

    assert mcp_responses_to_json(responses) == [
        {
            "jsonrpc": "2.0",
            "id": "ok-1",
            "result": {"ok": True},
        },
        {
            "jsonrpc": "2.0",
            "id": 2,
            "error": {"code": -32600, "message": "Invalid request"},
        },
    ]


def test_parse_jsonrpc_body_valid_raw_json_bytes_returns_payload():
    payload = parse_jsonrpc_body(b'{"jsonrpc":"2.0","method":"ping","id":"raw-1"}')

    assert payload == {"jsonrpc": "2.0", "method": "ping", "id": "raw-1"}


def test_parse_jsonrpc_body_invalid_json_returns_serializable_parse_error():
    response = parse_jsonrpc_body(b'{"jsonrpc":"2.0",')

    assert isinstance(response, MCPResponse)
    assert mcp_response_to_json(response) == {
        "jsonrpc": "2.0",
        "id": None,
        "error": {"code": -32700, "message": "Parse error"},
    }


def test_parse_error_response_serializes_as_parse_error_with_null_id():
    response = parse_error_response()

    assert mcp_response_to_json(response) == {
        "jsonrpc": "2.0",
        "id": None,
        "error": {"code": -32700, "message": "Parse error"},
    }


def test_invalid_request_response_serializes_error_and_preserves_id():
    response = invalid_request_response("Malformed request", request_id="bad-1")

    assert mcp_response_to_json(response) == {
        "jsonrpc": "2.0",
        "id": "bad-1",
        "error": {"code": -32600, "message": "Malformed request"},
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
