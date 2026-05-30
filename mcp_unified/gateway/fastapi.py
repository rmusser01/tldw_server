"""FastAPI transport skeleton for standalone MCP gateway runtimes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, FastAPI, Request, Response

from .runtime import GatewayRequestContext, GatewayRuntime

_PROTOCOL_VERSION = "2024-11-05"
_JSONRPC_VERSION = "2.0"

_INVALID_REQUEST = -32600
_METHOD_NOT_FOUND = -32601
_INVALID_PARAMS = -32602
_INTERNAL_ERROR = -32603
_GATEWAY_RUNTIME_ERRORS = (
    AttributeError,
    LookupError,
    RuntimeError,
    TypeError,
)


def _runtime_name(runtime: GatewayRuntime) -> str:
    return str(getattr(runtime, "name", "mcp-unified-gateway"))


def _runtime_version(runtime: GatewayRuntime) -> str:
    return str(getattr(runtime, "version", "0.1.0"))


def _jsonrpc_result(result: Any, request_id: str | int | None) -> dict[str, Any]:
    return {"jsonrpc": _JSONRPC_VERSION, "result": result, "id": request_id}


def _jsonrpc_error(
    code: int,
    message: str,
    request_id: str | int | None,
    *,
    data: Any | None = None,
) -> dict[str, Any]:
    error: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return {"jsonrpc": _JSONRPC_VERSION, "error": error, "id": request_id}


def _request_context(
    payload: dict[str, Any],
    request: Request,
) -> GatewayRequestContext:
    request_id = payload.get("id")
    metadata: dict[str, Any] = {
        "method": payload.get("method"),
        "path": str(request.url.path),
    }
    if request.client is not None:
        metadata["client_host"] = request.client.host
    return GatewayRequestContext(
        request_id=str(request_id if request_id is not None else "notification"),
        metadata=metadata,
    )


async def _handle_initialize(
    runtime: GatewayRuntime,
) -> dict[str, Any]:
    return {
        "protocolVersion": _PROTOCOL_VERSION,
        "capabilities": {
            "tools": {"available": True},
            "resources": {"available": False},
            "prompts": {"available": False},
        },
        "serverInfo": {
            "name": _runtime_name(runtime),
            "version": _runtime_version(runtime),
        },
    }


async def _dispatch_jsonrpc(
    runtime: GatewayRuntime,
    payload: dict[str, Any],
    request: Request,
) -> Any:
    method = payload.get("method")
    params = payload.get("params") or {}
    if not isinstance(params, dict):
        raise ValueError("params must be an object")

    context = _request_context(payload, request)

    if method == "initialize":
        return await _handle_initialize(runtime)
    if method == "ping":
        return {"pong": True}
    if method == "tools/list":
        return {"tools": await runtime.list_tools(context)}
    if method == "tools/call":
        tool_name = params.get("name")
        arguments = params.get("arguments") or {}
        if not isinstance(tool_name, str) or not tool_name.strip():
            raise ValueError("tools/call requires a non-empty string name")
        if not isinstance(arguments, dict):
            raise ValueError("tools/call arguments must be an object")
        return await runtime.call_tool(tool_name, arguments, context)

    raise NotImplementedError(str(method))


async def _handle_single_jsonrpc(
    runtime: GatewayRuntime,
    payload: Any,
    request: Request,
) -> dict[str, Any] | Response:
    if not isinstance(payload, dict):
        return _jsonrpc_error(_INVALID_REQUEST, "Request must be an object", None)

    request_id = payload.get("id")
    if payload.get("jsonrpc", _JSONRPC_VERSION) != _JSONRPC_VERSION:
        return _jsonrpc_error(_INVALID_REQUEST, "jsonrpc must be 2.0", request_id)

    method = payload.get("method")
    if not isinstance(method, str) or not method:
        return _jsonrpc_error(_INVALID_REQUEST, "method must be a non-empty string", request_id)

    try:
        result = await _dispatch_jsonrpc(runtime, payload, request)
    except ValueError as exc:
        return _jsonrpc_error(_INVALID_PARAMS, str(exc), request_id)
    except NotImplementedError:
        return _jsonrpc_error(_METHOD_NOT_FOUND, f"Method not found: {method}", request_id)
    except _GATEWAY_RUNTIME_ERRORS as exc:
        return _jsonrpc_error(_INTERNAL_ERROR, "Internal server error", request_id, data=exc.__class__.__name__)

    if request_id is None:
        return Response(status_code=204)
    return _jsonrpc_result(result, request_id)


async def _handle_jsonrpc(
    runtime: GatewayRuntime,
    payload: Any,
    request: Request,
) -> dict[str, Any] | list[dict[str, Any]] | Response:
    if isinstance(payload, list):
        if not payload:
            return _jsonrpc_error(_INVALID_REQUEST, "Batch request must not be empty", None)
        responses: list[dict[str, Any]] = []
        for item in payload:
            response = await _handle_single_jsonrpc(runtime, item, request)
            if isinstance(response, dict):
                responses.append(response)
        if not responses:
            return Response(status_code=204)
        return responses

    return await _handle_single_jsonrpc(runtime, payload, request)


def create_gateway_router(runtime: GatewayRuntime) -> APIRouter:
    """Create a package-owned FastAPI router for a standalone MCP runtime."""

    router = APIRouter()

    @router.get("/status")
    async def gateway_status() -> dict[str, str]:
        return {
            "status": "ok",
            "name": _runtime_name(runtime),
            "version": _runtime_version(runtime),
        }

    @router.post("/request")
    async def gateway_request(request: Request, payload: Any = Body(...)) -> Any:
        return await _handle_jsonrpc(runtime, payload, request)

    return router


def create_gateway_app(runtime: GatewayRuntime, *, prefix: str = "/mcp") -> FastAPI:
    """Create a minimal FastAPI app exposing the standalone MCP gateway router."""

    app = FastAPI(title="MCP Unified Gateway", version=_runtime_version(runtime))
    app.include_router(create_gateway_router(runtime), prefix=prefix)
    return app
