"""FastAPI transport skeleton for standalone MCP gateway runtimes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from fastapi import APIRouter, FastAPI, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from .bootstrap import GatewayProfileBootstrap
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
from .profiles import GatewayProfileManagementError, GatewayProfileManager
from .runtime import GatewayRuntime

_PROFILE_HEADER_NAMES = ("x-mcp-profile", "x-mcp-profile-id")
_PROFILE_QUERY_NAMES = ("profile_id", "profileId")
_PROFILE_MANAGEMENT_STATUS_CODES = {
    "profile_not_found": 404,
    "preset_not_found": 404,
    "default_profile_not_configured": 404,
    "profile_disabled": 409,
    "profile_is_default": 409,
    "profile_has_assignments": 409,
    "profile_already_exists": 409,
    "invalid_profile_request": 422,
    "invalid_profile_patch": 422,
    "profile_store_unavailable": 503,
    "assignment_store_unavailable": 503,
    "unexpected_delete_result": 500,
}


class DuplicatePresetRequest(BaseModel):
    """Request body for duplicating a built-in gateway profile preset."""

    preset_id: str
    profile_id: str | None = None
    name: str | None = None


class SetDefaultProfileRequest(BaseModel):
    """Request body for changing the gateway default profile."""

    profile_id: str


class CreateProfileRequest(BaseModel):
    """Request body for creating a user-editable gateway profile."""

    model_config = ConfigDict(extra="allow")

    id: str
    name: str


class PatchProfileRequest(BaseModel):
    """Request body for patching a user-editable gateway profile."""

    model_config = ConfigDict(extra="allow")


class StoreMetadataResponse(BaseModel):
    """Response metadata describing the active gateway profile store."""

    kind: Literal["memory", "sqlite"]
    persistent: bool


class ProfileListResponse(BaseModel):
    """Response body for listing stored gateway profiles."""

    ok: bool
    profiles: list[dict[str, Any]]
    store: StoreMetadataResponse


class ProfileResponse(BaseModel):
    """Response body for returning one stored gateway profile."""

    ok: bool
    profile: dict[str, Any]
    store: StoreMetadataResponse


class DuplicatePresetResponse(BaseModel):
    """Response body for duplicating a built-in profile preset."""

    ok: bool
    preset_id: str | None = None
    preset_version: str | None = None
    profile: dict[str, Any]
    store: StoreMetadataResponse


class DefaultProfileResponse(BaseModel):
    """Response body for reading or updating the gateway default profile."""

    ok: bool
    profile: dict[str, Any]
    assignment: dict[str, Any] | None
    default: dict[str, Any]
    store: StoreMetadataResponse


class DeleteProfileResponse(BaseModel):
    """Response body for deleting one stored gateway profile."""

    ok: bool
    profile_id: str
    store: StoreMetadataResponse


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


def _profile_management_error_response(exc: GatewayProfileManagementError) -> JSONResponse:
    """Translate expected profile-management errors into HTTP JSON responses."""

    return JSONResponse(
        status_code=_PROFILE_MANAGEMENT_STATUS_CODES.get(exc.reason_code, 500),
        content=exc.to_payload(),
    )


def _normalize_duplicate_preset_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Expose duplicated preset metadata at the top level for HTTP clients."""

    profile = payload.get("profile")
    if not isinstance(profile, Mapping):
        return payload
    if profile.get("preset_id") is None and profile.get("preset_version") is None:
        return payload

    normalized = dict(payload)
    if profile.get("preset_id") is not None:
        normalized["preset_id"] = profile["preset_id"]
    if profile.get("preset_version") is not None:
        normalized["preset_version"] = profile["preset_version"]
    return normalized


def _resolve_profile_manager(
    *,
    profile_manager: GatewayProfileManager | None,
    profile_bootstrap: GatewayProfileBootstrap | None,
    enable_profile_management: bool,
) -> GatewayProfileManager | None:
    """Return the configured profile manager or reject invalid explicit gating."""

    resolved_manager = profile_manager
    if resolved_manager is None and profile_bootstrap is not None:
        resolved_manager = profile_bootstrap.profile_manager
    if enable_profile_management and resolved_manager is None:
        raise ValueError(
            "profile management requires profile_manager or profile_bootstrap"
        )
    return resolved_manager


def _mount_profile_management_routes(
    router: APIRouter,
    manager: GatewayProfileManager,
) -> None:
    """Mount profile-management endpoints on the package gateway router."""

    @router.get("/profiles", response_model=ProfileListResponse)
    async def list_profiles() -> ProfileListResponse | JSONResponse:
        """Return editable MCP profiles from the configured profile manager."""
        try:
            return await manager.list_profiles()
        except GatewayProfileManagementError as exc:
            return _profile_management_error_response(exc)

    @router.post("/profiles", response_model=ProfileResponse)
    async def create_profile(
        request: CreateProfileRequest,
    ) -> ProfileResponse | JSONResponse:
        """Create a user-editable MCP profile."""
        try:
            return await manager.create_profile(request.model_dump(mode="json"))
        except GatewayProfileManagementError as exc:
            return _profile_management_error_response(exc)

    @router.patch("/profiles/{profile_id}", response_model=ProfileResponse)
    async def patch_profile(
        profile_id: str,
        request: PatchProfileRequest,
    ) -> ProfileResponse | JSONResponse:
        """Apply an allowed semantic patch to an editable MCP profile."""
        try:
            return await manager.patch_profile(
                profile_id,
                request.model_dump(mode="json", exclude_unset=True),
            )
        except GatewayProfileManagementError as exc:
            return _profile_management_error_response(exc)

    @router.delete("/profiles/{profile_id}", response_model=DeleteProfileResponse)
    async def delete_profile(profile_id: str) -> DeleteProfileResponse | JSONResponse:
        """Delete an unassigned, non-default editable MCP profile."""
        try:
            return await manager.delete_profile(profile_id)
        except GatewayProfileManagementError as exc:
            return _profile_management_error_response(exc)

    @router.post("/profiles/from-preset", response_model=DuplicatePresetResponse)
    async def duplicate_preset(
        request: DuplicatePresetRequest,
    ) -> DuplicatePresetResponse | JSONResponse:
        """Duplicate a built-in profile preset into editable profile storage."""
        try:
            payload = await manager.duplicate_preset(
                request.preset_id,
                profile_id=request.profile_id,
                name=request.name,
            )
        except GatewayProfileManagementError as exc:
            return _profile_management_error_response(exc)
        return _normalize_duplicate_preset_payload(payload)

    @router.get("/profiles/default", response_model=DefaultProfileResponse)
    async def get_default_profile() -> DefaultProfileResponse | JSONResponse:
        """Return the currently effective default MCP profile."""
        try:
            return await manager.get_default_profile()
        except GatewayProfileManagementError as exc:
            return _profile_management_error_response(exc)

    @router.put("/profiles/default", response_model=DefaultProfileResponse)
    async def set_default_profile(
        request: SetDefaultProfileRequest,
    ) -> DefaultProfileResponse | JSONResponse:
        """Set the gateway default MCP profile assignment."""
        try:
            return await manager.set_default_profile(request.profile_id)
        except GatewayProfileManagementError as exc:
            return _profile_management_error_response(exc)

    @router.get("/profiles/{profile_id}", response_model=ProfileResponse)
    async def show_profile(profile_id: str) -> ProfileResponse | JSONResponse:
        """Return a single editable MCP profile by id."""
        try:
            return await manager.show_profile(profile_id)
        except GatewayProfileManagementError as exc:
            return _profile_management_error_response(exc)


def create_gateway_router(
    runtime: GatewayRuntime,
    *,
    profile_manager: GatewayProfileManager | None = None,
    profile_bootstrap: GatewayProfileBootstrap | None = None,
    enable_profile_management: bool = False,
) -> APIRouter:
    """Create a package-owned FastAPI router for a standalone MCP runtime."""

    router = APIRouter()
    resolved_profile_manager = _resolve_profile_manager(
        profile_manager=profile_manager,
        profile_bootstrap=profile_bootstrap,
        enable_profile_management=enable_profile_management,
    )
    if resolved_profile_manager is not None:
        _mount_profile_management_routes(router, resolved_profile_manager)

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


def create_gateway_app(
    runtime: GatewayRuntime,
    *,
    prefix: str = "/mcp",
    profile_manager: GatewayProfileManager | None = None,
    profile_bootstrap: GatewayProfileBootstrap | None = None,
    enable_profile_management: bool = False,
) -> FastAPI:
    """Create a minimal FastAPI app exposing the standalone MCP gateway router."""

    app = FastAPI(title="MCP Unified Gateway", version=_runtime_version(runtime))
    app.include_router(
        create_gateway_router(
            runtime,
            profile_manager=profile_manager,
            profile_bootstrap=profile_bootstrap,
            enable_profile_management=enable_profile_management,
        ),
        prefix=prefix,
    )
    return app
