"""JSON-RPC transport helpers for mounted MCP boundaries."""

from __future__ import annotations

import json
import secrets
from dataclasses import dataclass
from typing import Any

from .protocol import MCPError, MCPRequest, MCPResponse

_JSONRPC_EXPLICIT_NULL_ID_PREFIX = "__tldw_jsonrpc_explicit_null_id_"


@dataclass
class JSONRPCRequestPlan:
    """Prepared JSON-RPC request state for an HTTP transport boundary."""

    request: MCPRequest | None
    error: MCPResponse | None
    explicit_null_id: bool
    is_notification: bool
    null_id_sentinel: str


@dataclass
class JSONRPCBatchPlan:
    """Prepared JSON-RPC batch state for an HTTP transport boundary."""

    requests: list[MCPRequest]
    entries: list[tuple[str, MCPRequest | MCPResponse]]
    explicit_null_id_sentinels: set[str]


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
        data["result"] = response.result
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
    except (TypeError, ValueError):
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


def is_valid_jsonrpc_id_value(value: Any) -> bool:
    """Return whether a JSON-RPC id value is valid for this transport."""
    if value is None:
        return True
    if isinstance(value, bool):
        return False
    return isinstance(value, (str, int))


def jsonrpc_null_id_sentinel(index: int | None = None) -> str:
    """Return a request-local sentinel for explicit JSON-RPC null ids."""
    suffix = secrets.token_hex(8)
    if index is None:
        return f"{_JSONRPC_EXPLICIT_NULL_ID_PREFIX}{suffix}__"
    return f"{_JSONRPC_EXPLICIT_NULL_ID_PREFIX}{index}_{suffix}__"


def invalid_request_response(
    message: str,
    request_id: str | int | None = None,
) -> MCPResponse:
    """Create a JSON-RPC invalid-request error response."""
    return MCPResponse(
        error=MCPError(code=-32600, message=message),
        id=request_id,
    )


def jsonrpc_request_id_error(payload: Any, message: str) -> MCPResponse:
    """Build an invalid-request response using only safe request ids."""
    request_id = None
    if isinstance(payload, dict) and jsonrpc_payload_has_id(payload):
        request_id = safe_jsonrpc_id(payload.get("id"))
    return invalid_request_response(message, request_id=request_id)


def parse_error_response(message: str = "Parse error") -> MCPResponse:
    """Create a JSON-RPC parse-error response."""
    return MCPResponse(
        error=MCPError(code=-32700, message=message),
        id=None,
    )


def is_jsonrpc_keepalive(payload: Any) -> bool:
    """Return whether a payload is an exact mounted MCP keepalive frame."""
    return payload in ({"type": "ping"}, {"type": "pong"})


def mcp_request_from_jsonrpc_payload(
    payload: Any,
    *,
    null_id_sentinel: str | None = None,
) -> tuple[MCPRequest | None, MCPResponse | None, bool]:
    """Validate a JSON-RPC object and construct an MCPRequest when valid."""
    if not isinstance(payload, dict):
        return None, invalid_request_response("Invalid request", request_id=None), False

    id_present = jsonrpc_payload_has_id(payload)
    if id_present and not is_valid_jsonrpc_id_value(payload.get("id")):
        return None, invalid_request_response("Invalid request id", request_id=None), False

    request_payload = dict(payload)
    explicit_null_id = id_present and payload.get("id") is None
    if explicit_null_id:
        request_payload["id"] = null_id_sentinel or jsonrpc_null_id_sentinel()

    try:
        return MCPRequest(**request_payload), None, explicit_null_id
    except (TypeError, ValueError) as exc:
        return (
            None,
            jsonrpc_request_id_error(payload, f"Invalid request format: {exc}"),
            False,
        )


def prepare_jsonrpc_request(payload: Any) -> JSONRPCRequestPlan:
    """Prepare a single JSON-RPC payload for MCP protocol dispatch."""
    null_id_sentinel = jsonrpc_null_id_sentinel()
    request_obj, error_obj, explicit_null_id = mcp_request_from_jsonrpc_payload(
        payload,
        null_id_sentinel=null_id_sentinel,
    )
    return JSONRPCRequestPlan(
        request=request_obj,
        error=error_obj,
        explicit_null_id=explicit_null_id,
        is_notification=isinstance(payload, dict) and not jsonrpc_payload_has_id(payload),
        null_id_sentinel=null_id_sentinel,
    )


def prepare_jsonrpc_batch(payload: list[Any]) -> JSONRPCBatchPlan:
    """Prepare a JSON-RPC batch for MCP protocol dispatch."""
    requests: list[MCPRequest] = []
    entries: list[tuple[str, MCPRequest | MCPResponse]] = []
    explicit_null_id_sentinels: set[str] = set()
    for index, item in enumerate(payload):
        sentinel = jsonrpc_null_id_sentinel(index)
        request_obj, error_obj, explicit_null_id = mcp_request_from_jsonrpc_payload(
            item,
            null_id_sentinel=sentinel,
        )
        if error_obj is not None or request_obj is None:
            entries.append(("error", error_obj or invalid_request_response("Invalid request", request_id=None)))
            continue
        if explicit_null_id:
            explicit_null_id_sentinels.add(sentinel)
        requests.append(request_obj)
        if not jsonrpc_payload_has_id(item):
            continue
        entries.append(("request", request_obj))

    return JSONRPCBatchPlan(
        requests=requests,
        entries=entries,
        explicit_null_id_sentinels=explicit_null_id_sentinels,
    )


def restore_explicit_null_jsonrpc_ids(response: MCPResponse | list[MCPResponse], sentinels: set[str]) -> None:
    """Restore explicit JSON-RPC null ids that were preserved via sentinels."""
    responses = response if isinstance(response, list) else [response]
    for item in responses:
        if isinstance(item.id, str) and item.id in sentinels:
            item.id = None
