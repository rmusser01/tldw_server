"""Transport-neutral JSON-RPC handling for standalone MCP gateways."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

from loguru import logger
from pydantic import BaseModel, Field, ValidationError

try:
    from pydantic import field_validator as _pydantic_field_validator
except ImportError:  # pragma: no cover - pydantic v1 fallback
    from pydantic import validator as _pydantic_validator  # type: ignore

    def _before_validator(*fields: str) -> Any:
        """Create a pydantic v1 pre-validation decorator."""

        return _pydantic_validator(*fields, pre=True, allow_reuse=True)

else:

    def _before_validator(*fields: str) -> Any:
        """Create a pydantic v2 before-validation decorator."""

        return _pydantic_field_validator(*fields, mode="before")

try:
    from pydantic.json import pydantic_encoder as _pydantic_encoder
except ImportError:  # pragma: no cover - defensive fallback for future pydantic changes
    _pydantic_encoder = None

from .runtime import GatewayPolicyDenied, GatewayRequestContext, GatewayRuntime

PROTOCOL_VERSION = "2024-11-05"
JSONRPC_VERSION = "2.0"

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
POLICY_DENIED = -32001
_GATEWAY_RUNTIME_ERRORS = Exception
_JSON_PARSE_ERRORS = (json.JSONDecodeError, UnicodeDecodeError)
_RESERVED_CONTEXT_METADATA_KEYS = frozenset({"client_host", "method", "path"})


class GatewayJSONRPCRequest(BaseModel):
    """Validated JSON-RPC request envelope accepted by the gateway."""

    jsonrpc: Literal["2.0"]
    method: str = Field(..., min_length=1, max_length=100)
    params: Any | None = None
    id: str | int | None = None

    @_before_validator("id")
    def _validate_id(cls: type[GatewayJSONRPCRequest], value: Any) -> str | int | None:
        """Reject JSON-RPC ids that cannot be safely echoed for correlation."""

        return validate_request_id(value)


class GatewayJSONRPCError(BaseModel):
    """JSON-RPC error object returned by the gateway."""

    code: int
    message: str
    data: Any | None = None


class GatewayJSONRPCSuccessResponse(BaseModel):
    """JSON-RPC success response envelope returned by the gateway."""

    jsonrpc: Literal["2.0"] = JSONRPC_VERSION
    result: Any
    id: str | int | None = None


class GatewayJSONRPCErrorResponse(BaseModel):
    """JSON-RPC error response envelope returned by the gateway."""

    jsonrpc: Literal["2.0"] = JSONRPC_VERSION
    error: GatewayJSONRPCError
    id: str | int | None = None


@dataclass(frozen=True, slots=True)
class GatewayNoResponse:
    """Sentinel for JSON-RPC notifications that intentionally produce no response."""


GatewayJSONRPCResponse = GatewayJSONRPCSuccessResponse | GatewayJSONRPCErrorResponse
GatewayJSONRPCResult = GatewayJSONRPCResponse | list[GatewayJSONRPCResponse] | GatewayNoResponse
GATEWAY_RESPONSE_TYPES = (GatewayJSONRPCSuccessResponse, GatewayJSONRPCErrorResponse)


def runtime_name(runtime: GatewayRuntime) -> str:
    """Return the runtime display name with a standalone default."""

    return str(getattr(runtime, "name", "mcp-unified-gateway"))


def runtime_version(runtime: GatewayRuntime) -> str:
    """Return the runtime version with a standalone default."""

    return str(getattr(runtime, "version", "0.1.0"))


def validate_request_id(value: Any) -> str | int | None:
    """Validate the gateway's JSON-RPC id contract before response echoing."""

    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError("id must be a string, number, or null")
    return value


def jsonrpc_result(result: Any, request_id: str | int | None) -> GatewayJSONRPCSuccessResponse:
    """Build a successful JSON-RPC response envelope."""

    return GatewayJSONRPCSuccessResponse(result=result, id=request_id)


def jsonrpc_error(
    code: int,
    message: str,
    request_id: str | int | None,
    *,
    data: Any | None = None,
) -> GatewayJSONRPCErrorResponse:
    """Build a JSON-RPC error response envelope."""

    return GatewayJSONRPCErrorResponse(
        error=GatewayJSONRPCError(code=code, message=message, data=data),
        id=request_id,
    )


def response_to_json(response: GatewayJSONRPCResponse) -> dict[str, Any]:
    """Return a JSON-serializable response mapping for transport sends."""

    try:
        return response.model_dump(mode="json")  # type: ignore[attr-defined]
    except AttributeError:  # pragma: no cover - pydantic v1 fallback
        if hasattr(response, "json"):
            return json.loads(response.json())
        return _json_compatible(response.dict())


def _json_compatible(value: Any) -> Any:
    """Convert pydantic v1 `.dict()` output into JSON-compatible values."""

    return json.loads(json.dumps(value, default=_json_default))


def _json_default(value: Any) -> Any:
    """Serialize common non-JSON scalar values without importing FastAPI."""

    if _pydantic_encoder is not None:
        try:
            return _pydantic_encoder(value)
        except TypeError:
            pass
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, (Decimal, Path, UUID)):
        return str(value)
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def object_or_empty(value: Any, message: str) -> dict[str, Any]:
    """Return an object payload or reject non-object values without coercion."""

    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    raise ValueError(message)


def required_string(value: Any, message: str) -> str:
    """Return a stripped string value or reject missing and non-string input."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(message)
    return value.strip()


def runtime_supports(runtime: GatewayRuntime, *method_names: str) -> bool:
    """Return whether the injected runtime exposes each named method."""

    return all(callable(getattr(runtime, method_name, None)) for method_name in method_names)


def request_context(
    payload: GatewayJSONRPCRequest,
    *,
    path: str,
    client_host: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> GatewayRequestContext:
    """Derive the host-neutral gateway request context from a JSON-RPC request."""

    context_metadata: dict[str, Any] = {
        key: value
        for key, value in (metadata or {}).items()
        if key not in _RESERVED_CONTEXT_METADATA_KEYS
    }
    context_metadata.update(
        {
            "method": payload.method,
            "path": path,
        }
    )
    if client_host is not None:
        context_metadata["client_host"] = client_host
    return GatewayRequestContext(
        request_id=str(payload.id if payload.id is not None else "notification"),
        metadata=context_metadata,
    )


async def handle_initialize(
    runtime: GatewayRuntime,
) -> dict[str, Any]:
    """Return the gateway's initial MCP capability advertisement."""

    return {
        "protocolVersion": PROTOCOL_VERSION,
        "capabilities": {
            "tools": {"available": True},
            "resources": {"available": runtime_supports(runtime, "list_resources", "read_resource")},
            "prompts": {"available": runtime_supports(runtime, "list_prompts", "get_prompt")},
        },
        "serverInfo": {
            "name": runtime_name(runtime),
            "version": runtime_version(runtime),
        },
    }


async def dispatch_jsonrpc(
    runtime: GatewayRuntime,
    payload: GatewayJSONRPCRequest,
    *,
    path: str,
    client_host: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Any:
    """Dispatch one validated JSON-RPC request to the injected runtime."""

    method = payload.method
    params = object_or_empty(payload.params, "params must be an object")

    context = request_context(payload, path=path, client_host=client_host, metadata=metadata)

    if method == "initialize":
        return await handle_initialize(runtime)
    if method == "ping":
        return {"pong": True}
    if method == "tools/list":
        return {"tools": await runtime.list_tools(context)}
    if method == "tools/call":
        tool_name = params.get("name")
        arguments = object_or_empty(
            params.get("arguments"),
            "tools/call arguments must be an object",
        )
        tool_name = required_string(tool_name, "tools/call requires a non-empty string name")
        return await runtime.call_tool(tool_name, arguments, context)
    if method == "resources/list":
        return {"resources": await runtime.list_resources(context)}
    if method == "resources/read":
        uri = required_string(params.get("uri"), "resources/read requires a non-empty string uri")
        return await runtime.read_resource(uri, context)
    if method == "prompts/list":
        return {"prompts": await runtime.list_prompts(context)}
    if method == "prompts/get":
        name = required_string(params.get("name"), "prompts/get requires a non-empty string name")
        arguments = object_or_empty(
            params.get("arguments"),
            "prompts/get arguments must be an object",
        )
        return await runtime.get_prompt(name, arguments, context)
    if method == "modules/list":
        return {"modules": await runtime.list_modules(context)}
    if method == "modules/health":
        return {"health": await runtime.get_modules_health(context)}

    raise NotImplementedError(str(method))


def validate_jsonrpc_request(payload: dict[str, Any]) -> GatewayJSONRPCRequest:
    """Validate a raw mapping as a gateway JSON-RPC request envelope."""

    try:
        return GatewayJSONRPCRequest.model_validate(payload)
    except AttributeError:  # pragma: no cover - pydantic v1 fallback
        return GatewayJSONRPCRequest.parse_obj(payload)


async def handle_single_jsonrpc(
    runtime: GatewayRuntime,
    payload: Any,
    *,
    path: str,
    client_host: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> GatewayJSONRPCResponse | GatewayNoResponse:
    """Handle one JSON-RPC request, including notifications and error mapping."""

    if not isinstance(payload, dict):
        return jsonrpc_error(INVALID_REQUEST, "Request must be an object", None)

    try:
        request_id = validate_request_id(payload.get("id"))
    except ValueError as exc:
        return jsonrpc_error(INVALID_REQUEST, str(exc), None)

    try:
        gateway_request = validate_jsonrpc_request(payload)
    except ValidationError as exc:
        return jsonrpc_error(INVALID_REQUEST, "Invalid request", request_id, data=exc.errors())

    try:
        result = await dispatch_jsonrpc(
            runtime,
            gateway_request,
            path=path,
            client_host=client_host,
            metadata=metadata,
        )
    except GatewayPolicyDenied as exc:
        error = jsonrpc_error(POLICY_DENIED, str(exc), request_id, data=exc.to_error_data())
    except ValueError as exc:
        error = jsonrpc_error(INVALID_PARAMS, str(exc), request_id)
    except NotImplementedError:
        error = jsonrpc_error(METHOD_NOT_FOUND, f"Method not found: {gateway_request.method}", request_id)
    except _GATEWAY_RUNTIME_ERRORS as exc:  # noqa: BLE001 - JSON-RPC requires mapping runtime failures to -32603.
        logger.opt(exception=True).error(
            "Gateway runtime error while handling method={!r} request_id={!r}",
            gateway_request.method,
            request_id,
        )
        error = jsonrpc_error(INTERNAL_ERROR, "Internal server error", request_id, data=exc.__class__.__name__)
    else:
        if request_id is None:
            return GatewayNoResponse()
        return jsonrpc_result(result, request_id)

    if request_id is None:
        return GatewayNoResponse()
    return error


async def handle_jsonrpc(
    runtime: GatewayRuntime,
    payload: Any,
    *,
    path: str,
    client_host: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> GatewayJSONRPCResult:
    """Handle single or batch JSON-RPC payloads for a gateway transport."""

    if isinstance(payload, list):
        if not payload:
            return jsonrpc_error(INVALID_REQUEST, "Batch request must not be empty", None)
        responses: list[GatewayJSONRPCResponse] = []
        for item in payload:
            response = await handle_single_jsonrpc(
                runtime,
                item,
                path=path,
                client_host=client_host,
                metadata=metadata,
            )
            if isinstance(response, GATEWAY_RESPONSE_TYPES):
                responses.append(response)
        if not responses:
            return GatewayNoResponse()
        return responses

    return await handle_single_jsonrpc(
        runtime,
        payload,
        path=path,
        client_host=client_host,
        metadata=metadata,
    )


def parse_json_payload(payload: str | bytes) -> Any | GatewayJSONRPCErrorResponse:
    """Parse a raw JSON-RPC payload or return a JSON-RPC parse error."""

    if not payload:
        return jsonrpc_error(PARSE_ERROR, "Parse error: Empty request body", None)
    try:
        return json.loads(payload)
    except _JSON_PARSE_ERRORS:
        return jsonrpc_error(PARSE_ERROR, "Parse error: Invalid JSON", None)


__all__ = [
    "GATEWAY_RESPONSE_TYPES",
    "INVALID_REQUEST",
    "POLICY_DENIED",
    "GatewayJSONRPCError",
    "GatewayJSONRPCErrorResponse",
    "GatewayJSONRPCRequest",
    "GatewayJSONRPCResponse",
    "GatewayJSONRPCResult",
    "GatewayJSONRPCSuccessResponse",
    "GatewayNoResponse",
    "dispatch_jsonrpc",
    "handle_initialize",
    "handle_jsonrpc",
    "handle_single_jsonrpc",
    "jsonrpc_error",
    "jsonrpc_result",
    "parse_json_payload",
    "request_context",
    "response_to_json",
    "runtime_name",
    "runtime_supports",
    "runtime_version",
    "validate_jsonrpc_request",
    "validate_request_id",
]
