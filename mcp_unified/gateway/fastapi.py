"""FastAPI transport skeleton for standalone MCP gateway runtimes."""

from __future__ import annotations

import json
from typing import Any, Literal

from fastapi import APIRouter, FastAPI, Request, Response
from loguru import logger
from pydantic import BaseModel, Field, ValidationError

try:
    from pydantic import field_validator as _pydantic_field_validator
except ImportError:  # pragma: no cover - pydantic v1 fallback
    from pydantic import validator as _pydantic_validator  # type: ignore

    def _before_validator(*fields: str):
        """Create a pydantic v1 pre-validation decorator."""

        return _pydantic_validator(*fields, pre=True, allow_reuse=True)

else:

    def _before_validator(*fields: str):
        """Create a pydantic v2 before-validation decorator."""

        return _pydantic_field_validator(*fields, mode="before")

from .runtime import GatewayRequestContext, GatewayRuntime

_PROTOCOL_VERSION = "2024-11-05"
_JSONRPC_VERSION = "2.0"

_PARSE_ERROR = -32700
_INVALID_REQUEST = -32600
_METHOD_NOT_FOUND = -32601
_INVALID_PARAMS = -32602
_INTERNAL_ERROR = -32603
_GATEWAY_RUNTIME_ERRORS = Exception
_JSON_PARSE_ERRORS = (json.JSONDecodeError, UnicodeDecodeError)


class GatewayJSONRPCRequest(BaseModel):
    """Validated JSON-RPC request envelope accepted by the gateway."""

    jsonrpc: Literal["2.0"]
    method: str = Field(..., min_length=1, max_length=100)
    params: Any | None = None
    id: str | int | None = None

    @_before_validator("id")
    def _validate_id(cls, value: Any) -> str | int | None:
        """Reject JSON-RPC ids that cannot be safely echoed for correlation."""

        return _validate_request_id(value)


class GatewayJSONRPCError(BaseModel):
    """JSON-RPC error object returned by the gateway."""

    code: int
    message: str
    data: Any | None = None


class GatewayJSONRPCSuccessResponse(BaseModel):
    """JSON-RPC success response envelope returned by the gateway."""

    jsonrpc: Literal["2.0"] = _JSONRPC_VERSION
    result: Any
    id: str | int | None = None


class GatewayJSONRPCErrorResponse(BaseModel):
    """JSON-RPC error response envelope returned by the gateway."""

    jsonrpc: Literal["2.0"] = _JSONRPC_VERSION
    error: GatewayJSONRPCError
    id: str | int | None = None


GatewayJSONRPCResponse = GatewayJSONRPCSuccessResponse | GatewayJSONRPCErrorResponse
_GATEWAY_RESPONSE_TYPES = (GatewayJSONRPCSuccessResponse, GatewayJSONRPCErrorResponse)


def _runtime_name(runtime: GatewayRuntime) -> str:
    """Return the runtime display name with a standalone default."""

    return str(getattr(runtime, "name", "mcp-unified-gateway"))


def _runtime_version(runtime: GatewayRuntime) -> str:
    """Return the runtime version with a standalone default."""

    return str(getattr(runtime, "version", "0.1.0"))


def _validate_request_id(value: Any) -> str | int | None:
    """Validate the gateway's JSON-RPC id contract before response echoing."""

    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError("id must be a string, number, or null")
    return value


def _jsonrpc_result(result: Any, request_id: str | int | None) -> GatewayJSONRPCSuccessResponse:
    """Build a successful JSON-RPC response envelope."""

    return GatewayJSONRPCSuccessResponse(result=result, id=request_id)


def _jsonrpc_error(
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


def _object_or_empty(value: Any, message: str) -> dict[str, Any]:
    """Return an object payload or reject non-object values without coercion."""

    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    raise ValueError(message)


def _required_string(value: Any, message: str) -> str:
    """Return a stripped string value or reject missing and non-string input."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(message)
    return value.strip()


def _runtime_supports(runtime: GatewayRuntime, *method_names: str) -> bool:
    """Return whether the injected runtime exposes each named method."""

    return all(callable(getattr(runtime, method_name, None)) for method_name in method_names)


def _request_context(
    payload: GatewayJSONRPCRequest,
    request: Request,
) -> GatewayRequestContext:
    """Derive the host-neutral gateway request context from a JSON-RPC request."""

    metadata: dict[str, Any] = {
        "method": payload.method,
        "path": str(request.url.path),
    }
    if request.client is not None:
        metadata["client_host"] = request.client.host
    return GatewayRequestContext(
        request_id=str(payload.id if payload.id is not None else "notification"),
        metadata=metadata,
    )


async def _handle_initialize(
    runtime: GatewayRuntime,
) -> dict[str, Any]:
    """Return the gateway's initial MCP capability advertisement."""

    return {
        "protocolVersion": _PROTOCOL_VERSION,
        "capabilities": {
            "tools": {"available": True},
            "resources": {"available": _runtime_supports(runtime, "list_resources", "read_resource")},
            "prompts": {"available": _runtime_supports(runtime, "list_prompts", "get_prompt")},
        },
        "serverInfo": {
            "name": _runtime_name(runtime),
            "version": _runtime_version(runtime),
        },
    }


async def _dispatch_jsonrpc(
    runtime: GatewayRuntime,
    payload: GatewayJSONRPCRequest,
    request: Request,
) -> Any:
    """Dispatch one validated JSON-RPC request to the injected runtime."""

    method = payload.method
    params = _object_or_empty(payload.params, "params must be an object")

    context = _request_context(payload, request)

    if method == "initialize":
        return await _handle_initialize(runtime)
    if method == "ping":
        return {"pong": True}
    if method == "tools/list":
        return {"tools": await runtime.list_tools(context)}
    if method == "tools/call":
        tool_name = params.get("name")
        arguments = _object_or_empty(
            params.get("arguments"),
            "tools/call arguments must be an object",
        )
        tool_name = _required_string(tool_name, "tools/call requires a non-empty string name")
        return await runtime.call_tool(tool_name, arguments, context)
    if method == "resources/list":
        return {"resources": await runtime.list_resources(context)}
    if method == "resources/read":
        uri = _required_string(params.get("uri"), "resources/read requires a non-empty string uri")
        return await runtime.read_resource(uri, context)
    if method == "prompts/list":
        return {"prompts": await runtime.list_prompts(context)}
    if method == "prompts/get":
        name = _required_string(params.get("name"), "prompts/get requires a non-empty string name")
        arguments = _object_or_empty(
            params.get("arguments"),
            "prompts/get arguments must be an object",
        )
        return await runtime.get_prompt(name, arguments, context)
    if method == "modules/list":
        return {"modules": await runtime.list_modules(context)}
    if method == "modules/health":
        return {"health": await runtime.get_modules_health(context)}

    raise NotImplementedError(str(method))


def _validate_jsonrpc_request(payload: dict[str, Any]) -> GatewayJSONRPCRequest:
    """Validate a raw mapping as a gateway JSON-RPC request envelope."""

    try:
        return GatewayJSONRPCRequest.model_validate(payload)
    except AttributeError:  # pragma: no cover - pydantic v1 fallback
        return GatewayJSONRPCRequest.parse_obj(payload)


async def _handle_single_jsonrpc(
    runtime: GatewayRuntime,
    payload: Any,
    request: Request,
) -> GatewayJSONRPCResponse | Response:
    """Handle one JSON-RPC request, including notifications and error mapping."""

    if not isinstance(payload, dict):
        return _jsonrpc_error(_INVALID_REQUEST, "Request must be an object", None)

    try:
        request_id = _validate_request_id(payload.get("id"))
    except ValueError as exc:
        return _jsonrpc_error(_INVALID_REQUEST, str(exc), None)

    try:
        gateway_request = _validate_jsonrpc_request(payload)
    except ValidationError as exc:
        return _jsonrpc_error(_INVALID_REQUEST, "Invalid request", request_id, data=exc.errors())

    try:
        result = await _dispatch_jsonrpc(runtime, gateway_request, request)
    except ValueError as exc:
        error = _jsonrpc_error(_INVALID_PARAMS, str(exc), request_id)
    except NotImplementedError:
        error = _jsonrpc_error(_METHOD_NOT_FOUND, f"Method not found: {gateway_request.method}", request_id)
    except _GATEWAY_RUNTIME_ERRORS as exc:  # noqa: BLE001 - JSON-RPC requires mapping runtime failures to -32603.
        logger.opt(exception=True).error(
            "Gateway runtime error while handling method={!r} request_id={!r}",
            gateway_request.method,
            request_id,
        )
        error = _jsonrpc_error(_INTERNAL_ERROR, "Internal server error", request_id, data=exc.__class__.__name__)
    else:
        if request_id is None:
            return Response(status_code=204)
        return _jsonrpc_result(result, request_id)

    if request_id is None:
        return Response(status_code=204)
    return error


async def _handle_jsonrpc(
    runtime: GatewayRuntime,
    payload: Any,
    request: Request,
) -> GatewayJSONRPCResponse | list[GatewayJSONRPCResponse] | Response:
    """Handle single or batch JSON-RPC payloads for the gateway endpoint."""

    if isinstance(payload, list):
        if not payload:
            return _jsonrpc_error(_INVALID_REQUEST, "Batch request must not be empty", None)
        responses: list[GatewayJSONRPCResponse] = []
        for item in payload:
            response = await _handle_single_jsonrpc(runtime, item, request)
            if isinstance(response, _GATEWAY_RESPONSE_TYPES):
                responses.append(response)
        if not responses:
            return Response(status_code=204)
        return responses

    return await _handle_single_jsonrpc(runtime, payload, request)


async def _parse_json_body(request: Request) -> Any | GatewayJSONRPCErrorResponse:
    """Parse raw JSON so malformed bodies return JSON-RPC parse errors."""

    body = await request.body()
    if not body:
        return _jsonrpc_error(_PARSE_ERROR, "Parse error: Empty request body", None)
    try:
        return json.loads(body)
    except _JSON_PARSE_ERRORS:
        return _jsonrpc_error(_PARSE_ERROR, "Parse error: Invalid JSON", None)


def create_gateway_router(runtime: GatewayRuntime) -> APIRouter:
    """Create a package-owned FastAPI router for a standalone MCP runtime."""

    router = APIRouter()

    @router.get("/status")
    async def gateway_status() -> dict[str, str]:
        """Return lightweight gateway health and runtime identity metadata."""

        return {
            "status": "ok",
            "name": _runtime_name(runtime),
            "version": _runtime_version(runtime),
        }

    @router.post(
        "/request",
        response_model=GatewayJSONRPCResponse | list[GatewayJSONRPCResponse],
        responses={204: {"description": "JSON-RPC notification accepted."}},
    )
    async def gateway_request(request: Request) -> GatewayJSONRPCResponse | list[GatewayJSONRPCResponse] | Response:
        """Process a raw JSON-RPC HTTP request body for the standalone gateway."""

        payload = await _parse_json_body(request)
        if isinstance(payload, _GATEWAY_RESPONSE_TYPES):
            return payload
        return await _handle_jsonrpc(runtime, payload, request)

    return router


def create_gateway_app(runtime: GatewayRuntime, *, prefix: str = "/mcp") -> FastAPI:
    """Create a minimal FastAPI app exposing the standalone MCP gateway router."""

    app = FastAPI(title="MCP Unified Gateway", version=_runtime_version(runtime))
    app.include_router(create_gateway_router(runtime), prefix=prefix)
    return app
