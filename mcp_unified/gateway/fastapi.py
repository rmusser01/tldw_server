"""FastAPI transport skeleton for standalone MCP gateway runtimes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, FastAPI, Request, Response, WebSocket, WebSocketDisconnect

from .jsonrpc import (
    GATEWAY_RESPONSE_TYPES as _GATEWAY_RESPONSE_TYPES,
)
from .jsonrpc import (
    INVALID_REQUEST as _INVALID_REQUEST,
)
from .jsonrpc import (
    GatewayJSONRPCResponse,
    GatewayJSONRPCResult,
    GatewayNoResponse,
    handle_jsonrpc,
)
from .jsonrpc import (
    jsonrpc_error as _jsonrpc_error,
)
from .jsonrpc import (
    parse_json_payload as _parse_json_payload,
)
from .jsonrpc import (
    response_to_json as _response_to_json,
)
from .jsonrpc import (
    runtime_name as _runtime_name,
)
from .jsonrpc import (
    runtime_version as _runtime_version,
)
from .runtime import GatewayRuntime

_PROFILE_HEADER_NAMES = ("x-mcp-profile", "x-mcp-profile-id")
_PROFILE_QUERY_NAMES = ("profile_id", "profileId")


async def _parse_json_body(request: Request) -> Any:
    """Parse raw JSON so malformed bodies return JSON-RPC parse errors."""

    return _parse_json_payload(await request.body())


def _client_host(request: Request | WebSocket) -> str | None:
    """Return the peer host when the transport exposes one."""

    if request.client is None:
        return None
    return request.client.host


def _request_metadata(request: Request | WebSocket) -> dict[str, Any]:
    """Build host-neutral metadata from lightweight transport selectors."""

    metadata: dict[str, Any] = {}
    profile_id = _profile_id_from_transport(request)
    if profile_id is not None:
        metadata["profile_id"] = profile_id
    return metadata


def _profile_id_from_transport(request: Request | WebSocket) -> str | None:
    """Return an optional profile id selected by header or query parameter."""

    headers = getattr(request, "headers", None)
    if headers is not None:
        for header_name in _PROFILE_HEADER_NAMES:
            value = headers.get(header_name)
            if isinstance(value, str) and value.strip():
                return value.strip()

    query_params = getattr(request, "query_params", None)
    if query_params is not None:
        for query_name in _PROFILE_QUERY_NAMES:
            value = query_params.get(query_name)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


async def _send_websocket_response(
    websocket: WebSocket,
    response: GatewayJSONRPCResult,
) -> None:
    """Send JSON-RPC responses while suppressing notification-only results."""

    if isinstance(response, GatewayNoResponse):
        return
    if isinstance(response, list):
        await websocket.send_json([_response_to_json(item) for item in response])
        return
    await websocket.send_json(_response_to_json(response))


def _websocket_message_payload(message: dict[str, Any]) -> str | bytes | None:
    """Extract a JSON-RPC payload from a raw WebSocket ASGI message."""

    if message.get("type") == "websocket.disconnect":
        raise WebSocketDisconnect(
            code=message.get("code", 1000),
            reason=message.get("reason", ""),
        )
    raw_payload = message.get("text")
    if raw_payload is not None:
        return raw_payload
    raw_payload = message.get("bytes")
    if raw_payload is not None:
        return raw_payload
    return None


def _to_http_response(response: GatewayJSONRPCResult) -> GatewayJSONRPCResponse | list[GatewayJSONRPCResponse] | Response:
    """Convert a transport-neutral gateway response into FastAPI's HTTP contract."""

    if isinstance(response, GatewayNoResponse):
        return Response(status_code=204)
    return response


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
        response = await handle_jsonrpc(
            runtime,
            payload,
            path=str(request.url.path),
            client_host=_client_host(request),
            metadata=_request_metadata(request),
        )
        return _to_http_response(response)

    @router.websocket("/ws")
    async def gateway_websocket(websocket: WebSocket) -> None:
        """Process JSON-RPC messages over a standalone gateway WebSocket."""

        await websocket.accept()
        try:
            while True:
                raw_payload = _websocket_message_payload(await websocket.receive())
                if raw_payload is None:
                    await websocket.send_json(
                        _response_to_json(_jsonrpc_error(_INVALID_REQUEST, "Invalid WebSocket message", None))
                    )
                    continue
                payload = _parse_json_payload(raw_payload)
                if isinstance(payload, _GATEWAY_RESPONSE_TYPES):
                    await websocket.send_json(_response_to_json(payload))
                    continue
                response = await handle_jsonrpc(
                    runtime,
                    payload,
                    path=str(websocket.url.path),
                    client_host=_client_host(websocket),
                    metadata=_request_metadata(websocket),
                )
                await _send_websocket_response(websocket, response)
        except (WebSocketDisconnect, RuntimeError):
            return

    return router


def create_gateway_app(runtime: GatewayRuntime, *, prefix: str = "/mcp") -> FastAPI:
    """Create a minimal FastAPI app exposing the standalone MCP gateway router."""

    app = FastAPI(title="MCP Unified Gateway", version=_runtime_version(runtime))
    app.include_router(create_gateway_router(runtime), prefix=prefix)
    return app
