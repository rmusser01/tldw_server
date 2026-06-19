"""JSON-RPC transport helpers for mounted MCP boundaries."""

from __future__ import annotations

import json
from typing import Any

from .protocol import MCPError, MCPResponse


def mcp_response_to_json(response: MCPResponse) -> dict[str, Any]:
    """Serialize an MCP response without optional null response fields."""
    if hasattr(response, "model_dump"):
        data = response.model_dump(mode="json", exclude_none=True)
    else:
        data = response.dict(exclude_none=True)

    data["id"] = response.id
    if response.error is not None:
        data.pop("result", None)
        error = data.get("error")
        if isinstance(error, dict) and error.get("data") is None:
            error.pop("data", None)
    else:
        data.pop("error", None)
    return data


def mcp_responses_to_json(
    response: MCPResponse | list[MCPResponse],
) -> dict[str, Any] | list[dict[str, Any]]:
    """Serialize one MCP response or a batch of MCP responses."""
    if isinstance(response, list):
        return [mcp_response_to_json(item) for item in response]
    return mcp_response_to_json(response)


def parse_jsonrpc_body(raw_body: bytes) -> Any | MCPResponse:
    """Parse a JSON-RPC request body, returning a parse-error response on failure."""
    try:
        return json.loads(raw_body)
    except (TypeError, ValueError, json.JSONDecodeError):
        return parse_error_response()


def jsonrpc_payload_has_id(payload: Any) -> bool:
    """Return whether a JSON-RPC payload contains an id member."""
    return isinstance(payload, dict) and "id" in payload


def safe_jsonrpc_id(value: Any) -> str | int | None:
    """Return a JSON-RPC id only when it is safe for local response use."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (str, int)):
        return value
    return None


def invalid_request_response(
    message: str,
    request_id: str | int | None = None,
) -> MCPResponse:
    """Create a JSON-RPC invalid-request error response."""
    return MCPResponse(
        error=MCPError(code=-32600, message=message),
        id=request_id,
    )


def parse_error_response(message: str = "Parse error") -> MCPResponse:
    """Create a JSON-RPC parse-error response."""
    return MCPResponse(
        error=MCPError(code=-32700, message=message),
        id=None,
    )


def is_jsonrpc_keepalive(payload: Any) -> bool:
    """Return whether a payload is an exact mounted MCP keepalive frame."""
    return payload in ({"type": "ping"}, {"type": "pong"})
