"""FastAPI transport skeleton for standalone MCP gateway runtimes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from fastapi import APIRouter, FastAPI, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from mcp_unified.storage.models import ExternalServerDefinition

from .bootstrap import GatewayProfileBootstrap
from .external_registry import (
    GatewayExternalRegistryManagementError,
    GatewayExternalRegistryManager,
)
from .external_runtime import (
    GatewayExternalRuntimeError,
    GatewayExternalRuntimeManager,
)
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
_EXTERNAL_REGISTRY_STATUS_CODES = {
    "external_registry_store_unavailable": 503,
    "credential_grant_store_unavailable": 503,
    "external_server_not_found": 404,
    "external_server_already_exists": 409,
    "external_server_has_credential_grants": 409,
    "credential_slot_change_requires_disabled_server": 409,
    "invalid_external_server_request": 422,
    "invalid_external_server_patch": 422,
    "unexpected_external_server_delete_result": 500,
}
_EXTERNAL_RUNTIME_STATUS_CODES = {
    "external_server_not_found": 404,
    "external_virtual_tool_not_found": 404,
    "external_server_disabled": 409,
    "external_server_start_failed": 503,
    "external_server_stop_failed": 503,
    "external_server_discovery_failed": 503,
    "external_server_transport_unavailable": 503,
    "external_tool_call_failed": 503,
    "credential_broker_unavailable": 503,
    "invalid_external_runtime_request": 422,
}
_EXTERNAL_SERVER_RESERVED_IDS = frozenset({"runtime", "refresh", "reconcile"})


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


class CreateExternalServerRequest(ExternalServerDefinition):
    """Request body for creating a stored external MCP server definition."""


class PatchExternalServerRequest(BaseModel):
    """Request body for patching a stored external MCP server definition."""

    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    transport: Literal["stdio", "websocket"] | None = None
    command: list[str] | None = None
    url: str | None = None
    cwd: str | None = None
    env_allowlist: list[str] | None = None
    credential_slots: list[str] | None = None
    metadata: dict[str, Any] | None = None
    provenance: dict[str, Any] | None = None
    enabled: bool | None = None
    auto_start: bool | None = None


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


class ExternalServerListResponse(BaseModel):
    """Response body for listing stored external MCP servers."""

    ok: bool
    servers: list[ExternalServerDefinition]
    store: StoreMetadataResponse


class ExternalServerResponse(BaseModel):
    """Response body for returning one stored external MCP server."""

    ok: bool
    server: ExternalServerDefinition
    store: StoreMetadataResponse


class DeleteExternalServerResponse(BaseModel):
    """Response body for deleting one stored external MCP server."""

    ok: bool
    server_id: str
    store: StoreMetadataResponse


class ExternalRuntimeServerStatusResponse(BaseModel):
    """Response row for one external MCP server runtime."""

    model_config = ConfigDict(extra="allow")

    id: str
    status: str
    name: str | None = None
    transport: str | None = None
    enabled: bool | None = None
    tool_count: int | None = None
    checks: dict[str, Any] | None = None
    last_error: str | None = None


class ExternalRuntimeServerListResponse(BaseModel):
    """Response body for listing external MCP server runtime status."""

    model_config = ConfigDict(extra="allow")

    servers: list[ExternalRuntimeServerStatusResponse]
    ok: bool | None = None
    total_servers: int | None = None


class ExternalRuntimeOperationResponse(BaseModel):
    """Response body for external MCP server lifecycle operations."""

    model_config = ConfigDict(extra="allow")

    ok: bool
    reason_code: str
    server_id: str | None = None
    error: str | None = None
    errors: dict[str, Any] | None = None


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


def _external_registry_error_response(
    exc: GatewayExternalRegistryManagementError,
) -> JSONResponse:
    """Translate expected external-registry errors into HTTP JSON responses."""

    return JSONResponse(
        status_code=_EXTERNAL_REGISTRY_STATUS_CODES.get(exc.reason_code, 500),
        content=exc.to_payload(),
    )


def _external_runtime_error_response(
    exc: GatewayExternalRuntimeError,
) -> JSONResponse:
    """Translate expected external-runtime errors into HTTP JSON responses."""

    return JSONResponse(
        status_code=_EXTERNAL_RUNTIME_STATUS_CODES.get(exc.reason_code, 500),
        content=exc.to_payload(),
    )


def _external_registry_store_unavailable_response(
    *,
    server_id: str | None = None,
) -> JSONResponse:
    """Return a deterministic response for unexpected registry store failures."""

    return _external_registry_error_response(
        GatewayExternalRegistryManagementError(
            "External registry store unavailable",
            reason_code="external_registry_store_unavailable",
            server_id=server_id,
        )
    )


def _is_reserved_external_server_id(server_id: str) -> bool:
    """Return true when a server id would collide with gateway routes."""

    return server_id.strip().lower() in _EXTERNAL_SERVER_RESERVED_IDS


def _reserved_external_server_id_response(server_id: str) -> JSONResponse:
    """Return a registry error for route-reserved external server ids."""

    return _external_registry_error_response(
        GatewayExternalRegistryManagementError(
            f"External server id '{server_id}' is reserved",
            reason_code="invalid_external_server_request",
            server_id=server_id,
        )
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
        resolved_manager = getattr(profile_bootstrap, "profile_manager", None)
    if enable_profile_management and resolved_manager is None:
        raise ValueError(
            "profile management requires profile_manager or profile_bootstrap"
        )
    return resolved_manager


def _resolve_external_registry_manager(
    *,
    external_registry_manager: GatewayExternalRegistryManager | None,
    profile_bootstrap: GatewayProfileBootstrap | None,
    enable_external_registry_management: bool,
) -> GatewayExternalRegistryManager | None:
    """Return the configured external registry manager or reject invalid gating."""

    resolved = external_registry_manager
    if resolved is None and profile_bootstrap is not None:
        resolved = getattr(profile_bootstrap, "external_registry_manager", None)
    if enable_external_registry_management and resolved is None:
        raise ValueError(
            "external registry management requires external_registry_manager or profile_bootstrap"
        )
    return resolved


def _resolve_external_runtime_manager(
    *,
    external_runtime_manager: GatewayExternalRuntimeManager | None,
    profile_bootstrap: GatewayProfileBootstrap | None,
    enable_external_runtime_management: bool,
) -> GatewayExternalRuntimeManager | None:
    """Return the configured external runtime manager or reject invalid gating."""

    resolved = external_runtime_manager
    if resolved is None and profile_bootstrap is not None:
        resolved = getattr(profile_bootstrap, "external_runtime_manager", None)
    if enable_external_runtime_management and resolved is None:
        raise ValueError(
            "external runtime management requires external_runtime_manager or profile_bootstrap"
        )
    return resolved


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


def _mount_external_registry_routes(
    router: APIRouter,
    manager: GatewayExternalRegistryManager,
) -> None:
    """Mount external-registry management endpoints on the package gateway router."""

    @router.get("/external-servers", response_model=ExternalServerListResponse)
    async def list_external_servers(
        enabled: bool | None = None,
    ) -> ExternalServerListResponse | JSONResponse:
        """Return stored external MCP servers from the configured registry manager."""
        try:
            return await manager.list_servers(enabled=enabled)
        except GatewayExternalRegistryManagementError as exc:
            return _external_registry_error_response(exc)
        except Exception:  # noqa: BLE001
            return _external_registry_store_unavailable_response()

    @router.get("/external-servers/{server_id}", response_model=ExternalServerResponse)
    async def show_external_server(
        server_id: str,
    ) -> ExternalServerResponse | JSONResponse:
        """Return a single stored external MCP server by id."""
        if _is_reserved_external_server_id(server_id):
            return _reserved_external_server_id_response(server_id)
        try:
            return await manager.show_server(server_id)
        except GatewayExternalRegistryManagementError as exc:
            return _external_registry_error_response(exc)
        except Exception:  # noqa: BLE001
            return _external_registry_store_unavailable_response(server_id=server_id)

    @router.post("/external-servers", response_model=ExternalServerResponse)
    async def create_external_server(
        request: CreateExternalServerRequest,
    ) -> ExternalServerResponse | JSONResponse:
        """Create a stored external MCP server definition."""
        if _is_reserved_external_server_id(request.id):
            return _reserved_external_server_id_response(request.id)
        try:
            return await manager.create_server(
                request.model_dump(mode="json", exclude_unset=True)
            )
        except GatewayExternalRegistryManagementError as exc:
            return _external_registry_error_response(exc)
        except Exception:  # noqa: BLE001
            return _external_registry_store_unavailable_response(server_id=request.id)

    @router.patch("/external-servers/{server_id}", response_model=ExternalServerResponse)
    async def patch_external_server(
        server_id: str,
        request: PatchExternalServerRequest,
    ) -> ExternalServerResponse | JSONResponse:
        """Apply an allowed patch to a stored external MCP server definition."""
        if _is_reserved_external_server_id(server_id):
            return _reserved_external_server_id_response(server_id)
        try:
            return await manager.patch_server(
                server_id,
                request.model_dump(mode="json", exclude_unset=True),
            )
        except GatewayExternalRegistryManagementError as exc:
            return _external_registry_error_response(exc)
        except Exception:  # noqa: BLE001
            return _external_registry_store_unavailable_response(server_id=server_id)

    @router.delete(
        "/external-servers/{server_id}",
        response_model=DeleteExternalServerResponse,
    )
    async def delete_external_server(
        server_id: str,
    ) -> DeleteExternalServerResponse | JSONResponse:
        """Delete a stored external MCP server definition."""
        if _is_reserved_external_server_id(server_id):
            return _reserved_external_server_id_response(server_id)
        try:
            return await manager.delete_server(server_id)
        except GatewayExternalRegistryManagementError as exc:
            return _external_registry_error_response(exc)
        except Exception:  # noqa: BLE001
            return _external_registry_store_unavailable_response(server_id=server_id)


def _mount_external_runtime_routes(
    router: APIRouter,
    manager: GatewayExternalRuntimeManager,
) -> None:
    """Mount external runtime lifecycle endpoints on the package gateway router."""

    @router.get(
        "/external-servers/runtime",
        response_model=ExternalRuntimeServerListResponse,
        response_model_exclude_none=True,
    )
    async def list_external_runtime_servers() -> ExternalRuntimeServerListResponse | JSONResponse:
        """Return external server runtime status rows."""
        try:
            return await manager.list_runtime_servers()
        except GatewayExternalRuntimeError as exc:
            return _external_runtime_error_response(exc)

    @router.post(
        "/external-servers/{server_id}/start",
        response_model=ExternalRuntimeOperationResponse,
    )
    async def start_external_runtime_server(
        server_id: str,
    ) -> ExternalRuntimeOperationResponse | JSONResponse:
        """Start one configured external MCP server runtime."""
        try:
            return await manager.start_server(server_id)
        except GatewayExternalRuntimeError as exc:
            return _external_runtime_error_response(exc)

    @router.post(
        "/external-servers/{server_id}/stop",
        response_model=ExternalRuntimeOperationResponse,
    )
    async def stop_external_runtime_server(
        server_id: str,
    ) -> ExternalRuntimeOperationResponse | JSONResponse:
        """Stop one configured external MCP server runtime."""
        try:
            return await manager.stop_server(server_id)
        except GatewayExternalRuntimeError as exc:
            return _external_runtime_error_response(exc)

    @router.post(
        "/external-servers/{server_id}/restart",
        response_model=ExternalRuntimeOperationResponse,
    )
    async def restart_external_runtime_server(
        server_id: str,
    ) -> ExternalRuntimeOperationResponse | JSONResponse:
        """Restart one configured external MCP server runtime."""
        try:
            return await manager.restart_server(server_id)
        except GatewayExternalRuntimeError as exc:
            return _external_runtime_error_response(exc)

    @router.post(
        "/external-servers/refresh",
        response_model=ExternalRuntimeOperationResponse,
    )
    async def refresh_external_runtime_servers() -> ExternalRuntimeOperationResponse | JSONResponse:
        """Refresh discovery for all active external MCP server runtimes."""
        try:
            return await manager.refresh_server(None)
        except GatewayExternalRuntimeError as exc:
            return _external_runtime_error_response(exc)

    @router.post(
        "/external-servers/{server_id}/refresh",
        response_model=ExternalRuntimeOperationResponse,
    )
    async def refresh_external_runtime_server(
        server_id: str,
    ) -> ExternalRuntimeOperationResponse | JSONResponse:
        """Refresh discovery for one active external MCP server runtime."""
        try:
            return await manager.refresh_server(server_id)
        except GatewayExternalRuntimeError as exc:
            return _external_runtime_error_response(exc)

    @router.post(
        "/external-servers/reconcile",
        response_model=ExternalRuntimeOperationResponse,
    )
    async def reconcile_external_runtime_servers() -> ExternalRuntimeOperationResponse | JSONResponse:
        """Reconcile all configured external MCP server runtimes."""
        try:
            return await manager.reconcile(None)
        except GatewayExternalRuntimeError as exc:
            return _external_runtime_error_response(exc)

    @router.post(
        "/external-servers/{server_id}/reconcile",
        response_model=ExternalRuntimeOperationResponse,
    )
    async def reconcile_external_runtime_server(
        server_id: str,
    ) -> ExternalRuntimeOperationResponse | JSONResponse:
        """Reconcile one configured external MCP server runtime."""
        try:
            return await manager.reconcile(server_id)
        except GatewayExternalRuntimeError as exc:
            return _external_runtime_error_response(exc)

    @router.post(
        "/external-servers/{server_id}/install",
        response_model=ExternalRuntimeOperationResponse,
    )
    async def install_external_runtime_server(
        server_id: str,
    ) -> ExternalRuntimeOperationResponse | JSONResponse:
        """Run the configured install contract for one external MCP server."""
        try:
            return await manager.install_server(server_id)
        except GatewayExternalRuntimeError as exc:
            return _external_runtime_error_response(exc)

    @router.post(
        "/external-servers/{server_id}/update",
        response_model=ExternalRuntimeOperationResponse,
    )
    async def update_external_runtime_server(
        server_id: str,
    ) -> ExternalRuntimeOperationResponse | JSONResponse:
        """Run the configured update contract for one external MCP server."""
        try:
            return await manager.update_server(server_id)
        except GatewayExternalRuntimeError as exc:
            return _external_runtime_error_response(exc)


def create_gateway_router(
    runtime: GatewayRuntime,
    *,
    profile_manager: GatewayProfileManager | None = None,
    profile_bootstrap: GatewayProfileBootstrap | None = None,
    enable_profile_management: bool = False,
    external_registry_manager: GatewayExternalRegistryManager | None = None,
    enable_external_registry_management: bool = False,
    external_runtime_manager: GatewayExternalRuntimeManager | None = None,
    enable_external_runtime_management: bool = False,
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
    resolved_external_runtime_manager = _resolve_external_runtime_manager(
        external_runtime_manager=external_runtime_manager,
        profile_bootstrap=profile_bootstrap,
        enable_external_runtime_management=enable_external_runtime_management,
    )
    if resolved_external_runtime_manager is not None:
        _mount_external_runtime_routes(router, resolved_external_runtime_manager)
    resolved_external_registry_manager = _resolve_external_registry_manager(
        external_registry_manager=external_registry_manager,
        profile_bootstrap=profile_bootstrap,
        enable_external_registry_management=enable_external_registry_management,
    )
    if resolved_external_registry_manager is not None:
        _mount_external_registry_routes(router, resolved_external_registry_manager)

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
    external_registry_manager: GatewayExternalRegistryManager | None = None,
    enable_external_registry_management: bool = False,
    external_runtime_manager: GatewayExternalRuntimeManager | None = None,
    enable_external_runtime_management: bool = False,
) -> FastAPI:
    """Create a minimal FastAPI app exposing the standalone MCP gateway router."""

    app = FastAPI(title="MCP Unified Gateway", version=_runtime_version(runtime))
    app.include_router(
        create_gateway_router(
            runtime,
            profile_manager=profile_manager,
            profile_bootstrap=profile_bootstrap,
            enable_profile_management=enable_profile_management,
            external_registry_manager=external_registry_manager,
            enable_external_registry_management=enable_external_registry_management,
            external_runtime_manager=external_runtime_manager,
            enable_external_runtime_management=enable_external_runtime_management,
        ),
        prefix=prefix,
    )
    return app
