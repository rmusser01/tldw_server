"""FastAPI transport skeleton for standalone MCP gateway runtimes."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import APIRouter, Depends, FastAPI, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from loguru import logger
from pydantic import BaseModel, ConfigDict

from mcp_unified.storage.models import CredentialGrant, ExternalServerDefinition
from mcp_unified.interfaces.storage import AuditStore
from mcp_unified.package_metadata import package_metadata_summary
from mcp_unified.profiles import MCPProfile

from .admin_auth import (
    DefaultGatewayAdminPermissionChecker,
    GatewayAdminAuthConfig,
    GatewayAdminAuthError,
    GatewayAdminIdentity,
    GatewayAdminPermissionChecker,
    GatewayAdminPermissionError,
    gateway_admin_auth_dependencies,
    gateway_admin_auth_error_response,
    gateway_admin_identity_dependency,
    normalize_gateway_admin_auth_config,
)
from .bootstrap import GatewayProfileBootstrap
from .credential_grants import (
    CREDENTIAL_GRANT_SENSITIVE_MATERIAL_REJECTED_ERROR,
    CREDENTIAL_GRANT_SENSITIVE_MATERIAL_REJECTED_REASON,
    GatewayCredentialGrantManagementError,
    GatewayCredentialGrantManager,
)
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
from .lifecycle import (
    GatewayExternalRuntimeLifecycleConfig,
    normalize_external_runtime_lifecycle_config,
)
from .profiles import GatewayProfileManagementError, GatewayProfileManager
from .policy_explain import (
    GatewayPolicyExplainError,
    GatewayPolicyExplainService,
    PolicyExplainErrorResponse,
    PolicyExplainResponse,
    ProfileToolPreviewResponse,
    parse_policy_explain_request,
    parse_profile_tool_preview_request,
)
from .runtime import GatewayRequestContext, GatewayRuntime

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
    "permission_change_denied": 403,
    "permission_change_requires_approval": 409,
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
_CREDENTIAL_GRANT_STATUS_CODES = {
    "credential_grant_store_unavailable": 503,
    "profile_store_unavailable": 503,
    "external_registry_store_unavailable": 503,
    "credential_grant_not_found": 404,
    "profile_not_found": 404,
    "external_server_not_found": 404,
    "credential_grant_already_exists": 409,
    "invalid_credential_grant_request": 422,
    "invalid_credential_grant_patch": 422,
    CREDENTIAL_GRANT_SENSITIVE_MATERIAL_REJECTED_REASON: 422,
}
_POLICY_EXPLAIN_STATUS_CODES = {
    "profile_not_found": 404,
    "invalid_policy_explain_request": 422,
    "invalid_policy_preview_request": 422,
    "audit_store_unavailable": 503,
    "profile_resolution_failed": 503,
    "policy_evaluation_failed": 422,
}
POLICY_EXPLAIN_RATE_LIMIT_MAX_REQUESTS = 120
POLICY_EXPLAIN_RATE_LIMIT_WINDOW_SECONDS = 60.0
_POLICY_EXPLAIN_RATE_LIMIT_MAX_KEYS = 2048
_EXTERNAL_SERVER_RESERVED_IDS = frozenset({"runtime", "refresh", "reconcile"})
_PROFILE_MANAGEMENT_PUBLIC_ERRORS = {
    "permission_change_denied": "Permission change denied",
    "permission_change_requires_approval": "Permission change requires approval",
    "profile_store_unavailable": "Profile store unavailable",
    "assignment_store_unavailable": "Profile assignment store unavailable",
}
_EXTERNAL_REGISTRY_PUBLIC_ERRORS = {
    "external_registry_store_unavailable": "External registry store unavailable",
    "credential_grant_store_unavailable": "Credential grant store unavailable",
}
_EXTERNAL_RUNTIME_PUBLIC_ERRORS = {
    "external_server_start_failed": "External server start failed",
    "external_server_stop_failed": "External server stop failed",
    "external_server_discovery_failed": "External server discovery failed",
    "external_server_transport_unavailable": "External server transport unavailable",
    "external_tool_call_failed": "External tool call failed",
    "credential_broker_unavailable": "Credential broker unavailable",
}
_CREDENTIAL_GRANT_PUBLIC_ERRORS = {
    "credential_grant_store_unavailable": "Credential grant store unavailable",
    "profile_store_unavailable": "Profile store unavailable",
    "external_registry_store_unavailable": "External registry store unavailable",
    CREDENTIAL_GRANT_SENSITIVE_MATERIAL_REJECTED_REASON: (
        CREDENTIAL_GRANT_SENSITIVE_MATERIAL_REJECTED_ERROR
    ),
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


class CreateExternalServerRequest(ExternalServerDefinition):
    """Request body for creating a stored external MCP server definition."""


class PatchExternalServerRequest(BaseModel):
    """Request body for patching a stored external MCP server definition."""

    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    transport: Literal["stdio", "websocket", "streamable_http", "sse"] | None = None
    command: list[str] | None = None
    url: str | None = None
    cwd: str | None = None
    headers: dict[str, str] | None = None
    env_allowlist: list[str] | None = None
    credential_slots: list[str] | None = None
    metadata: dict[str, Any] | None = None
    provenance: dict[str, Any] | None = None
    enabled: bool | None = None
    auto_start: bool | None = None


class CreateCredentialGrantRequest(CredentialGrant):
    """Request body for creating stored credential broker grant metadata."""


class PatchCredentialGrantRequest(BaseModel):
    """Request body for patching stored credential broker grant metadata."""

    model_config = ConfigDict(extra="forbid")

    broker_id: str | None = None
    credential_slot: str | None = None
    external_server_id: str | None = None
    scopes: list[str] | None = None
    metadata: dict[str, Any] | None = None
    provenance: dict[str, Any] | None = None
    enabled: bool | None = None


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


class CredentialGrantListResponse(BaseModel):
    """Response body for listing stored credential grants."""

    ok: bool
    grants: list[CredentialGrant]
    store: StoreMetadataResponse


class CredentialGrantResponse(BaseModel):
    """Response body for returning one stored credential grant."""

    ok: bool
    grant: CredentialGrant
    store: StoreMetadataResponse


class DeleteCredentialGrantResponse(BaseModel):
    """Response body for deleting one stored credential grant."""

    ok: bool
    grant_id: str
    store: StoreMetadataResponse


class GatewayStatusStoreResponse(BaseModel):
    """Best-effort readiness metadata for a gateway backing store."""

    model_config = ConfigDict(extra="allow")

    kind: str
    persistent: bool | None = None


class GatewayStatusPackageResponse(BaseModel):
    """Package boundary metadata exposed by the gateway status endpoint."""

    model_config = ConfigDict(extra="allow")

    package_name: str | None = None
    package_import_name: str | None = None
    package_status: str | None = None
    publishing_status: str | None = None
    source_distribution: str | None = None
    dependency_version_policy: str | None = None


class GatewayStatusTransportResponse(BaseModel):
    """Transport metadata exposed by the gateway status endpoint."""

    model_config = ConfigDict(extra="allow")

    base_path: str
    mount_path: str


class GatewayStatusDefaultProfileResponse(BaseModel):
    """Default profile readiness metadata."""

    model_config = ConfigDict(extra="allow")

    configured: bool
    profile_id: str | None = None
    source: str


class GatewayStatusAdminAuthResponse(BaseModel):
    """Admin-auth readiness metadata without secret values."""

    model_config = ConfigDict(extra="allow")

    enabled: bool
    configured: bool
    header_name: str | None = None


class GatewayStatusExternalServersResponse(BaseModel):
    """External server readiness counts."""

    model_config = ConfigDict(extra="allow")

    total: int
    enabled: int
    unavailable: int


class GatewayStatusWarningResponse(BaseModel):
    """Non-secret readiness warning returned by gateway status."""

    reason_code: str
    message: str


class GatewayStatusResponse(BaseModel):
    """Response body for package-local gateway readiness status."""

    model_config = ConfigDict(extra="allow")

    status: str
    name: str
    version: str
    transport: GatewayStatusTransportResponse
    package: GatewayStatusPackageResponse
    profile_store: GatewayStatusStoreResponse
    default_profile: GatewayStatusDefaultProfileResponse
    admin_auth: GatewayStatusAdminAuthResponse
    external_registry_store: GatewayStatusStoreResponse
    external_servers: GatewayStatusExternalServersResponse
    warnings: list[GatewayStatusWarningResponse]
    next_actions: list[str]


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
    installer: dict[str, Any] | None = None


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
    available: bool | None = None
    installer: str | None = None
    version: str | None = None
    installed_version: str | None = None
    latest_version: str | None = None
    message: str | None = None
    details: dict[str, Any] | None = None
    required_fields: list[str] | None = None
    warnings: list[Any] | None = None
    error: str | None = None
    errors: dict[str, Any] | None = None


class GatewayStatusWarning(BaseModel):
    """Non-secret readiness warning for the package-local gateway status."""

    reason_code: str
    message: str


class GatewayPackageStatus(BaseModel):
    """Package-boundary metadata for the package-local gateway status."""

    package_name: str | None = None
    package_import_name: str | None = None
    package_status: str | None = None
    publishing_status: str | None = None
    source_distribution: str | None = None
    dependency_version_policy: str | None = None


class GatewayTransportStatus(BaseModel):
    """Transport metadata for the package-local gateway status."""

    base_path: str
    mount_path: str


class GatewayStoreReadinessStatus(BaseModel):
    """Non-secret store readiness metadata for the package-local gateway status."""

    model_config = ConfigDict(extra="allow")

    kind: str
    persistent: bool | None = None


class GatewayDefaultProfileStatus(BaseModel):
    """Default profile readiness metadata for the package-local gateway status."""

    configured: bool
    profile_id: str | None = None
    source: str


class GatewayAdminAuthStatus(BaseModel):
    """Admin authentication readiness metadata without secret values."""

    enabled: bool
    configured: bool
    header_name: str | None = None


class GatewayExternalServersStatus(BaseModel):
    """External MCP server count summary for the package-local gateway status."""

    total: int
    enabled: int
    unavailable: int


class GatewayReadinessStatusResponse(BaseModel):
    """Response body for package-local gateway readiness status."""

    status: str
    name: str
    version: str
    transport: GatewayTransportStatus
    package: GatewayPackageStatus
    profile_store: GatewayStoreReadinessStatus
    default_profile: GatewayDefaultProfileStatus
    admin_auth: GatewayAdminAuthStatus
    external_registry_store: GatewayStoreReadinessStatus
    external_servers: GatewayExternalServersStatus
    warnings: list[GatewayStatusWarning]
    next_actions: list[str]


class _GatewayAdminAuthHandlingRoute(APIRoute):
    """Route wrapper that keeps router-only admin auth failures JSON-stable."""

    def get_route_handler(self) -> Callable[[Request], Any]:
        original_route_handler = super().get_route_handler()

        async def route_handler(request: Request) -> Response:
            try:
                return await original_route_handler(request)
            except GatewayAdminAuthError as exc:
                return gateway_admin_auth_error_response(request, exc)

        return route_handler


class _GatewayAdminRouteRateLimiter:
    """Small in-memory rate limiter for standalone admin route surfaces."""

    def __init__(
        self,
        *,
        max_requests: int,
        window_seconds: float,
        max_keys: int = _POLICY_EXPLAIN_RATE_LIMIT_MAX_KEYS,
    ) -> None:
        self.max_requests = max(1, int(max_requests))
        self.window_seconds = max(1.0, float(window_seconds))
        self.max_keys = max(1, int(max_keys))
        self._hits_by_key: dict[str, deque[float]] = {}
        self._lock = asyncio.Lock()

    async def allow(self, key: str) -> bool:
        """Return whether one request is within this limiter's window."""

        now = time.monotonic()
        cutoff = now - self.window_seconds
        async with self._lock:
            self._prune_locked(cutoff)
            hits = self._hits_by_key.setdefault(key, deque())
            while hits and hits[0] <= cutoff:
                hits.popleft()
            if len(hits) >= self.max_requests:
                return False
            hits.append(now)
            if len(self._hits_by_key) > self.max_keys:
                self._prune_oldest_key_locked()
            return True

    def _prune_locked(self, cutoff: float) -> None:
        empty_keys: list[str] = []
        for key, hits in self._hits_by_key.items():
            while hits and hits[0] <= cutoff:
                hits.popleft()
            if not hits:
                empty_keys.append(key)
        for key in empty_keys:
            del self._hits_by_key[key]

    def _prune_oldest_key_locked(self) -> None:
        oldest_key: str | None = None
        oldest_hit: float | None = None
        for key, hits in self._hits_by_key.items():
            if not hits:
                oldest_key = key
                break
            if oldest_hit is None or hits[0] < oldest_hit:
                oldest_key = key
                oldest_hit = hits[0]
        if oldest_key is not None:
            del self._hits_by_key[oldest_key]


async def _parse_json_body(request: Request) -> Any:
    """Parse raw JSON so malformed bodies return JSON-RPC parse errors."""

    return _parse_json_payload(await request.body())


async def _parse_policy_explain_json_body(
    request: Request,
    *,
    reason_code: str,
) -> Any:
    """Parse a policy-explain JSON body without exposing raw invalid input."""

    try:
        return await request.json()
    except ValueError:
        raise GatewayPolicyExplainError(
            "Invalid JSON request",
            reason_code=reason_code,
        ) from None


def _client_host(request: Request | WebSocket) -> str | None:
    """Return the peer host when the transport exposes one."""

    if request.client is None:
        return None
    return request.client.host


def _policy_explain_rate_limit_key(
    request: Request,
    identity: GatewayAdminIdentity,
) -> str:
    """Return the admin identity/IP key used for policy explain throttling."""

    return f"{identity.actor_id}:{_client_host(request) or 'unknown'}"


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


def _to_http_response(response: GatewayJSONRPCResult) -> Response:
    """Convert a transport-neutral gateway response into FastAPI's HTTP contract."""

    if isinstance(response, GatewayNoResponse):
        return Response(status_code=204)
    if isinstance(response, list):
        return JSONResponse(content=[_response_to_json(item) for item in response])
    return JSONResponse(content=_response_to_json(response))


def _profile_management_error_response(exc: GatewayProfileManagementError) -> JSONResponse:
    """Translate expected profile-management errors into HTTP JSON responses."""

    return JSONResponse(
        status_code=_PROFILE_MANAGEMENT_STATUS_CODES.get(exc.reason_code, 500),
        content=_profile_management_error_payload(exc),
    )


def _external_registry_error_response(
    exc: GatewayExternalRegistryManagementError,
) -> JSONResponse:
    """Translate expected external-registry errors into HTTP JSON responses."""

    return JSONResponse(
        status_code=_EXTERNAL_REGISTRY_STATUS_CODES.get(exc.reason_code, 500),
        content=_external_registry_error_payload(exc),
    )


def _external_runtime_error_response(
    exc: GatewayExternalRuntimeError,
) -> JSONResponse:
    """Translate expected external-runtime errors into HTTP JSON responses."""

    return JSONResponse(
        status_code=_EXTERNAL_RUNTIME_STATUS_CODES.get(exc.reason_code, 500),
        content=_external_runtime_error_payload(exc),
    )


def _credential_grant_error_response(
    exc: GatewayCredentialGrantManagementError,
) -> JSONResponse:
    """Translate expected credential-grant errors into HTTP JSON responses."""

    return JSONResponse(
        status_code=_CREDENTIAL_GRANT_STATUS_CODES.get(exc.reason_code, 500),
        content=_credential_grant_error_payload(exc),
    )


def _policy_explain_error_response(exc: GatewayPolicyExplainError) -> JSONResponse:
    """Translate expected policy-explain errors into HTTP JSON responses."""

    return JSONResponse(
        status_code=_POLICY_EXPLAIN_STATUS_CODES.get(exc.reason_code, 500),
        content=exc.to_payload().model_dump(mode="json"),
    )


def _policy_explain_permission_error_response(
    exc: GatewayAdminPermissionError,
) -> JSONResponse:
    """Return the policy-explain error envelope for admin permission failures."""

    payload = PolicyExplainErrorResponse(
        message=exc.payload["error"],
        reason_code=exc.reason_code,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=payload.model_dump(mode="json"),
    )


def _policy_explain_rate_limit_error_response() -> JSONResponse:
    """Return the policy-explain error envelope for rate-limit denials."""

    payload = PolicyExplainErrorResponse(
        message="Policy explain rate limit exceeded",
        reason_code="policy_explain_rate_limited",
    )
    return JSONResponse(
        status_code=429,
        content=payload.model_dump(mode="json"),
    )


def _profile_tool_preview_payload_for_path(
    payload: Any,
    *,
    profile_id: str,
) -> Any:
    """Return preview payload with a validated path-canonical profile id."""

    if not isinstance(payload, Mapping):
        return payload
    body_profile_id = payload.get("profile_id")
    if body_profile_id is None:
        return {**payload, "profile_id": profile_id}
    if body_profile_id == profile_id:
        return payload
    raise GatewayPolicyExplainError(
        "Invalid policy preview request",
        reason_code="invalid_policy_preview_request",
    )


def _profile_management_error_payload(exc: GatewayProfileManagementError) -> dict[str, Any]:
    """Return a public profile-management error payload without raw exception text."""

    payload: dict[str, Any] = {
        "ok": False,
        "error": _PROFILE_MANAGEMENT_PUBLIC_ERRORS.get(
            exc.reason_code,
            "Gateway profile management request failed",
        ),
        "reason_code": exc.reason_code,
    }
    if exc.profile_id is not None:
        payload["profile_id"] = exc.profile_id
    if exc.preset_id is not None:
        payload["preset_id"] = exc.preset_id
    return payload


def _external_registry_error_payload(
    exc: GatewayExternalRegistryManagementError,
) -> dict[str, Any]:
    """Return a public external-registry error payload without raw exception text."""

    payload: dict[str, Any] = {
        "ok": False,
        "error": _EXTERNAL_REGISTRY_PUBLIC_ERRORS.get(
            exc.reason_code,
            "Gateway external registry request failed",
        ),
        "reason_code": exc.reason_code,
    }
    if exc.server_id is not None:
        payload["server_id"] = exc.server_id
    return payload


def _external_runtime_error_payload(exc: GatewayExternalRuntimeError) -> dict[str, Any]:
    """Return a public external-runtime error payload without raw exception text."""

    payload: dict[str, Any] = {
        "ok": False,
        "error": _EXTERNAL_RUNTIME_PUBLIC_ERRORS.get(
            exc.reason_code,
            "Gateway external runtime request failed",
        ),
        "reason_code": exc.reason_code,
    }
    if exc.server_id is not None:
        payload["server_id"] = exc.server_id
    return payload


def _credential_grant_error_payload(
    exc: GatewayCredentialGrantManagementError,
) -> dict[str, Any]:
    """Return a public credential-grant error payload without raw exception text."""

    payload: dict[str, Any] = {
        "ok": False,
        "error": _CREDENTIAL_GRANT_PUBLIC_ERRORS.get(
            exc.reason_code,
            "Gateway credential grant request failed",
        ),
        "reason_code": exc.reason_code,
    }
    if exc.grant_id is not None:
        payload["grant_id"] = exc.grant_id
    if exc.profile_id is not None:
        payload["profile_id"] = exc.profile_id
    if exc.external_server_id is not None:
        payload["external_server_id"] = exc.external_server_id
    return payload


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


def _resolve_credential_grant_manager(
    *,
    credential_grant_manager: GatewayCredentialGrantManager | None,
    profile_bootstrap: GatewayProfileBootstrap | None,
    enable_credential_grant_management: bool,
) -> GatewayCredentialGrantManager | None:
    """Return the configured credential-grant manager or reject invalid gating."""

    resolved = credential_grant_manager
    if resolved is None and profile_bootstrap is not None:
        resolved = getattr(profile_bootstrap, "credential_grant_manager", None)
    if enable_credential_grant_management and resolved is None:
        raise ValueError(
            "credential grant management requires credential_grant_manager or profile_bootstrap"
        )
    return resolved


def _resolve_external_runtime_lifecycle(
    *,
    external_runtime_lifecycle: (
        GatewayExternalRuntimeLifecycleConfig | Mapping[str, Any] | None
    ),
    profile_bootstrap: GatewayProfileBootstrap | None,
    external_runtime_manager: GatewayExternalRuntimeManager | None,
) -> GatewayExternalRuntimeLifecycleConfig:
    """Return resolved external runtime lifecycle behavior for one app."""

    lifecycle = external_runtime_lifecycle
    if lifecycle is None and profile_bootstrap is not None:
        lifecycle = getattr(profile_bootstrap, "external_runtime_lifecycle", None)
    resolved = normalize_external_runtime_lifecycle_config(lifecycle)
    if resolved.enabled and external_runtime_manager is None:
        raise ValueError(
            "external runtime lifecycle requires external_runtime_manager or profile_bootstrap"
        )
    return resolved


def _resolve_admin_auth_config(
    *,
    admin_auth: GatewayAdminAuthConfig | Mapping[str, Any] | None,
    profile_bootstrap: GatewayProfileBootstrap | None,
) -> GatewayAdminAuthConfig:
    """Return explicit or bootstrap-carried admin auth configuration."""

    if admin_auth is not None:
        return normalize_gateway_admin_auth_config(admin_auth)
    if profile_bootstrap is not None:
        bootstrap_admin_auth = getattr(profile_bootstrap, "admin_auth", None)
        if bootstrap_admin_auth is not None:
            return normalize_gateway_admin_auth_config(bootstrap_admin_auth)
    return GatewayAdminAuthConfig()


def _external_runtime_lifecycle_error_payload(
    reason_code: str,
    exc: BaseException,
) -> dict[str, Any]:
    """Return compact lifecycle failure metadata without traceback or secrets."""

    return {
        "ok": False,
        "reason_code": reason_code,
        "error_type": type(exc).__name__,
        "error": "External runtime lifecycle operation failed",
    }


def _log_external_runtime_lifecycle_error(
    reason_code: str,
    exc: BaseException,
) -> None:
    """Log lifecycle failure context without exposing raw exception text."""

    logger.opt(exception=True).error(
        "External runtime lifecycle operation failed reason_code={!r} error_type={!r}",
        reason_code,
        type(exc).__name__,
    )


async def _run_external_runtime_startup(
    manager: GatewayExternalRuntimeManager,
) -> dict[str, Any]:
    """Run best-effort external runtime startup reconciliation."""

    try:
        return await manager.reconcile()
    except Exception as exc:  # noqa: BLE001 - app startup must report, not crash.
        _log_external_runtime_lifecycle_error(
            "external_runtime_startup_failed",
            exc,
        )
        return _external_runtime_lifecycle_error_payload(
            "external_runtime_startup_failed",
            exc,
        )


async def _run_external_runtime_shutdown(
    manager: GatewayExternalRuntimeManager,
) -> dict[str, Any]:
    """Run best-effort external runtime shutdown cleanup."""

    try:
        return await manager.stop_all()
    except Exception as exc:  # noqa: BLE001 - shutdown cleanup is best-effort.
        _log_external_runtime_lifecycle_error(
            "external_runtime_shutdown_failed",
            exc,
        )
        return _external_runtime_lifecycle_error_payload(
            "external_runtime_shutdown_failed",
            exc,
        )


def _create_external_runtime_lifespan(
    *,
    manager: GatewayExternalRuntimeManager,
    lifecycle: GatewayExternalRuntimeLifecycleConfig,
) -> Any:
    """Create a FastAPI lifespan context for external runtime lifecycle work."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if lifecycle.reconcile_on_startup:
            app.state.external_runtime_startup = await _run_external_runtime_startup(
                manager,
            )
        try:
            yield
        finally:
            if lifecycle.stop_on_shutdown:
                app.state.external_runtime_shutdown = await _run_external_runtime_shutdown(
                    manager,
                )

    return lifespan


def _mount_profile_management_routes(
    router: APIRouter,
    manager: GatewayProfileManager,
    *,
    admin_dependencies: list[Depends] | None = None,
) -> None:
    """Mount profile-management endpoints on the package gateway router."""

    dependencies = admin_dependencies or []

    @router.get(
        "/profiles",
        response_model=ProfileListResponse,
        dependencies=dependencies,
    )
    async def list_profiles() -> ProfileListResponse | JSONResponse:
        """Return editable MCP profiles from the configured profile manager."""
        try:
            return await manager.list_profiles()
        except GatewayProfileManagementError as exc:
            return _profile_management_error_response(exc)

    @router.post(
        "/profiles",
        response_model=ProfileResponse,
        dependencies=dependencies,
    )
    async def create_profile(
        request: CreateProfileRequest,
    ) -> ProfileResponse | JSONResponse:
        """Create a user-editable MCP profile."""
        try:
            return await manager.create_profile(request.model_dump(mode="json"))
        except GatewayProfileManagementError as exc:
            return _profile_management_error_response(exc)

    @router.patch(
        "/profiles/{profile_id}",
        response_model=ProfileResponse,
        dependencies=dependencies,
    )
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

    @router.delete(
        "/profiles/{profile_id}",
        response_model=DeleteProfileResponse,
        dependencies=dependencies,
    )
    async def delete_profile(profile_id: str) -> DeleteProfileResponse | JSONResponse:
        """Delete an unassigned, non-default editable MCP profile."""
        try:
            return await manager.delete_profile(profile_id)
        except GatewayProfileManagementError as exc:
            return _profile_management_error_response(exc)

    @router.post(
        "/profiles/from-preset",
        response_model=DuplicatePresetResponse,
        dependencies=dependencies,
    )
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

    @router.get(
        "/profiles/default",
        response_model=DefaultProfileResponse,
        dependencies=dependencies,
    )
    async def get_default_profile() -> DefaultProfileResponse | JSONResponse:
        """Return the currently effective default MCP profile."""
        try:
            return await manager.get_default_profile()
        except GatewayProfileManagementError as exc:
            return _profile_management_error_response(exc)

    @router.put(
        "/profiles/default",
        response_model=DefaultProfileResponse,
        dependencies=dependencies,
    )
    async def set_default_profile(
        request: SetDefaultProfileRequest,
    ) -> DefaultProfileResponse | JSONResponse:
        """Set the gateway default MCP profile assignment."""
        try:
            return await manager.set_default_profile(request.profile_id)
        except GatewayProfileManagementError as exc:
            return _profile_management_error_response(exc)

    @router.get(
        "/profiles/{profile_id}",
        response_model=ProfileResponse,
        dependencies=dependencies,
    )
    async def show_profile(profile_id: str) -> ProfileResponse | JSONResponse:
        """Return a single editable MCP profile by id."""
        try:
            return await manager.show_profile(profile_id)
        except GatewayProfileManagementError as exc:
            return _profile_management_error_response(exc)


def _mount_external_registry_routes(
    router: APIRouter,
    manager: GatewayExternalRegistryManager,
    *,
    admin_dependencies: list[Depends] | None = None,
) -> None:
    """Mount external-registry management endpoints on the package gateway router."""

    dependencies = admin_dependencies or []

    @router.get(
        "/external-servers",
        response_model=ExternalServerListResponse,
        dependencies=dependencies,
    )
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

    @router.get(
        "/external-servers/{server_id}",
        response_model=ExternalServerResponse,
        dependencies=dependencies,
    )
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

    @router.post(
        "/external-servers",
        response_model=ExternalServerResponse,
        dependencies=dependencies,
    )
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

    @router.patch(
        "/external-servers/{server_id}",
        response_model=ExternalServerResponse,
        dependencies=dependencies,
    )
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
        dependencies=dependencies,
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
    *,
    admin_dependencies: list[Depends] | None = None,
) -> None:
    """Mount external runtime lifecycle endpoints on the package gateway router."""

    dependencies = admin_dependencies or []

    @router.get(
        "/external-servers/runtime",
        response_model=ExternalRuntimeServerListResponse,
        response_model_exclude_none=True,
        dependencies=dependencies,
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
        dependencies=dependencies,
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
        dependencies=dependencies,
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
        dependencies=dependencies,
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
        dependencies=dependencies,
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
        dependencies=dependencies,
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
        dependencies=dependencies,
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
        dependencies=dependencies,
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
        dependencies=dependencies,
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
        dependencies=dependencies,
    )
    async def update_external_runtime_server(
        server_id: str,
    ) -> ExternalRuntimeOperationResponse | JSONResponse:
        """Run the configured update contract for one external MCP server."""
        try:
            return await manager.update_server(server_id)
        except GatewayExternalRuntimeError as exc:
            return _external_runtime_error_response(exc)


def _mount_credential_grant_routes(
    router: APIRouter,
    manager: GatewayCredentialGrantManager,
    *,
    admin_dependencies: list[Depends] | None = None,
) -> None:
    """Mount credential-grant management endpoints on the package gateway router."""

    dependencies = admin_dependencies or []

    @router.get(
        "/credential-grants",
        response_model=CredentialGrantListResponse,
        dependencies=dependencies,
    )
    async def list_credential_grants(
        profile_id: str | None = None,
        external_server_id: str | None = None,
    ) -> CredentialGrantListResponse | JSONResponse:
        """Return stored credential grant metadata from the configured manager."""
        try:
            return await manager.list_grants(
                profile_id=profile_id,
                external_server_id=external_server_id,
            )
        except GatewayCredentialGrantManagementError as exc:
            return _credential_grant_error_response(exc)

    @router.post(
        "/credential-grants",
        response_model=CredentialGrantResponse,
        dependencies=dependencies,
    )
    async def create_credential_grant(
        request: CreateCredentialGrantRequest,
    ) -> CredentialGrantResponse | JSONResponse:
        """Create stored credential grant metadata."""
        try:
            return await manager.create_grant(
                request.model_dump(mode="json", exclude_unset=True)
            )
        except GatewayCredentialGrantManagementError as exc:
            return _credential_grant_error_response(exc)

    @router.get(
        "/credential-grants/{grant_id}",
        response_model=CredentialGrantResponse,
        dependencies=dependencies,
    )
    async def show_credential_grant(
        grant_id: str,
    ) -> CredentialGrantResponse | JSONResponse:
        """Return a single credential grant by id."""
        try:
            return await manager.show_grant(grant_id)
        except GatewayCredentialGrantManagementError as exc:
            return _credential_grant_error_response(exc)

    @router.patch(
        "/credential-grants/{grant_id}",
        response_model=CredentialGrantResponse,
        dependencies=dependencies,
    )
    async def patch_credential_grant(
        grant_id: str,
        request: PatchCredentialGrantRequest,
    ) -> CredentialGrantResponse | JSONResponse:
        """Apply an allowed patch to stored credential grant metadata."""
        try:
            return await manager.patch_grant(
                grant_id,
                request.model_dump(mode="json", exclude_unset=True),
            )
        except GatewayCredentialGrantManagementError as exc:
            return _credential_grant_error_response(exc)

    @router.delete(
        "/credential-grants/{grant_id}",
        response_model=DeleteCredentialGrantResponse,
        dependencies=dependencies,
    )
    async def delete_credential_grant(
        grant_id: str,
    ) -> DeleteCredentialGrantResponse | JSONResponse:
        """Delete stored credential grant metadata."""
        try:
            return await manager.delete_grant(grant_id)
        except GatewayCredentialGrantManagementError as exc:
            return _credential_grant_error_response(exc)


def _mount_policy_explain_routes(
    router: APIRouter,
    runtime: GatewayRuntime,
    *,
    admin_auth: GatewayAdminAuthConfig,
    policy_explain_service: GatewayPolicyExplainService | None = None,
    policy_explain_profile_resolver: (
        Callable[[str], MCPProfile | Awaitable[MCPProfile | None]] | None
    ) = None,
    policy_explain_audit_store: AuditStore | None = None,
    policy_explain_permission_checker: GatewayAdminPermissionChecker | None = None,
) -> None:
    """Mount read-only policy explanation endpoints on the gateway router."""

    if policy_explain_service is None and policy_explain_profile_resolver is None:
        raise ValueError(
            "policy explain management requires policy_explain_service "
            "or policy_explain_profile_resolver"
        )

    identity_dependency = gateway_admin_identity_dependency(admin_auth)
    permission_checker = (
        policy_explain_permission_checker
        or DefaultGatewayAdminPermissionChecker()
    )

    async def installed_tool_catalog() -> list[dict[str, Any]]:
        context = GatewayRequestContext(request_id="policy-explain-tool-catalog")
        admin_catalog = getattr(
            runtime,
            "list_backend_tools_for_admin_catalog",
            None,
        )
        if callable(admin_catalog):
            return await admin_catalog(context)
        return await runtime.list_tools(context)

    def service_for_identity(
        identity: GatewayAdminIdentity,
    ) -> GatewayPolicyExplainService:
        if policy_explain_service is not None:
            return policy_explain_service.with_request_context(
                actor_id=identity.actor_id,
                installed_tool_catalog=installed_tool_catalog,
            )
        return GatewayPolicyExplainService(
            profile_resolver=policy_explain_profile_resolver,
            audit_store=policy_explain_audit_store,
            actor_id=identity.actor_id,
            installed_tool_catalog=installed_tool_catalog,
        )

    route_rate_limiter = _GatewayAdminRouteRateLimiter(
        max_requests=POLICY_EXPLAIN_RATE_LIMIT_MAX_REQUESTS,
        window_seconds=POLICY_EXPLAIN_RATE_LIMIT_WINDOW_SECONDS,
    )

    async def policy_explain_rate_limit_response(
        request: Request,
        identity: GatewayAdminIdentity,
    ) -> JSONResponse | None:
        key = _policy_explain_rate_limit_key(request, identity)
        if await route_rate_limiter.allow(key):
            return None
        logger.warning(
            "Gateway policy explain route rate limit exceeded "
            "(actor_id={}, client_host={})",
            identity.actor_id,
            _client_host(request) or "unknown",
        )
        return _policy_explain_rate_limit_error_response()

    @router.post(
        "/policy/explain",
        response_model=PolicyExplainResponse,
        response_model_exclude_none=True,
    )
    async def explain_policy(
        request: Request,
        identity: GatewayAdminIdentity = Depends(identity_dependency),
    ) -> PolicyExplainResponse | JSONResponse:
        """Return a redacted explanation for one profile/tool policy decision."""

        try:
            await permission_checker.require_permission(
                identity,
                "mcp.policy.explain",
            )
            rate_limit_response = await policy_explain_rate_limit_response(
                request,
                identity,
            )
            if rate_limit_response is not None:
                return rate_limit_response
            payload = await _parse_policy_explain_json_body(
                request,
                reason_code="invalid_policy_explain_request",
            )
            return await service_for_identity(identity).explain_tool_call(
                parse_policy_explain_request(payload)
            )
        except GatewayAdminPermissionError as exc:
            return _policy_explain_permission_error_response(exc)
        except GatewayPolicyExplainError as exc:
            return _policy_explain_error_response(exc)

    @router.post(
        "/profiles/{profile_id}/tool-preview",
        response_model=ProfileToolPreviewResponse,
        response_model_exclude_none=True,
    )
    async def preview_profile_tools(
        profile_id: str,
        request: Request,
        identity: GatewayAdminIdentity = Depends(identity_dependency),
    ) -> ProfileToolPreviewResponse | JSONResponse:
        """Return a redacted profile preview across installed gateway tools."""

        try:
            await permission_checker.require_permission(
                identity,
                "mcp.policy.explain",
            )
            rate_limit_response = await policy_explain_rate_limit_response(
                request,
                identity,
            )
            if rate_limit_response is not None:
                return rate_limit_response
            payload = await _parse_policy_explain_json_body(
                request,
                reason_code="invalid_policy_preview_request",
            )
            payload = _profile_tool_preview_payload_for_path(
                payload,
                profile_id=profile_id,
            )
            return await service_for_identity(identity).preview_profile_tools(
                parse_profile_tool_preview_request(payload)
            )
        except GatewayAdminPermissionError as exc:
            return _policy_explain_permission_error_response(exc)
        except GatewayPolicyExplainError as exc:
            return _policy_explain_error_response(exc)


def _read_store_metadata(manager: Any) -> dict[str, Any]:
    """Return non-secret store metadata from a gateway manager."""

    store_metadata = getattr(manager, "store_metadata", None)
    to_payload = getattr(store_metadata, "to_payload", None)
    if callable(to_payload):
        try:
            payload = to_payload()
            if isinstance(payload, dict):
                return dict(payload)
        except Exception:  # noqa: BLE001 - status must remain best-effort.
            logger.debug("Gateway status store metadata lookup failed", exc_info=True)
    return {"kind": "unknown", "persistent": None}


def _store_payload_from_result(result: dict[str, Any], fallback: dict[str, Any]) -> dict[str, Any]:
    """Return a store payload from a manager result when present."""

    store = result.get("store") if isinstance(result, dict) else None
    if isinstance(store, dict):
        return dict(store)
    return fallback


def _status_warning(reason_code: str, message: str) -> dict[str, str]:
    """Return a small non-secret readiness warning object."""

    return {"reason_code": reason_code, "message": message}


async def _gateway_readiness_status(
    runtime: GatewayRuntime,
    *,
    profile_manager: GatewayProfileManager | None,
    external_registry_manager: GatewayExternalRegistryManager | None,
    admin_auth: GatewayAdminAuthConfig | None,
    status_path: str,
) -> dict[str, Any]:
    """Return best-effort package-local gateway readiness metadata."""

    warnings: list[dict[str, str]] = []
    next_actions: list[str] = []
    mount_path = status_path.rsplit("/status", 1)[0] if status_path.endswith("/status") else status_path
    if not mount_path:
        mount_path = "/"
    package_summary = package_metadata_summary()
    package_payload = {
        key: package_summary.get(key)
        for key in (
            "package_name",
            "package_import_name",
            "package_status",
            "publishing_status",
            "source_distribution",
            "dependency_version_policy",
        )
    }
    if package_payload.get("publishing_status") == "not-published":
        warnings.append(
            _status_warning(
                "package_not_published",
                "Package metadata is internal/experimental and not published.",
            )
        )
        next_actions.append(
            "Install from apps/mcp-unified in this repository; do not use public PyPI install guidance."
        )

    profile_store = {"kind": "unknown", "persistent": None}
    default_profile: dict[str, Any] = {"configured": False, "profile_id": None, "source": "none"}
    if profile_manager is not None:
        profile_store = _read_store_metadata(profile_manager)
        try:
            default_result = await profile_manager.get_default_profile()
            profile_store = _store_payload_from_result(default_result, profile_store)
            default = default_result.get("default") if isinstance(default_result, dict) else None
            profile = default_result.get("profile") if isinstance(default_result, dict) else None
            profile_id = None
            source = "unknown"
            if isinstance(default, dict):
                profile_id = default.get("profile_id")
                source = str(default.get("source") or "unknown")
            if profile_id is None and isinstance(profile, dict):
                profile_id = profile.get("id")
            default_profile = {
                "configured": bool(profile_id),
                "profile_id": profile_id,
                "source": source,
            }
        except GatewayProfileManagementError as exc:
            warnings.append(
                _status_warning(
                    exc.reason_code,
                    "Default profile readiness check failed.",
                )
            )
            next_actions.append("Configure a default profile before relying on profile-scoped gateway calls.")
        except Exception:  # noqa: BLE001 - status must remain best-effort.
            logger.opt(exception=True).warning(
                "Gateway default profile readiness check failed"
            )
            warnings.append(
                _status_warning(
                    "default_profile_status_unavailable",
                    "Default profile readiness check failed.",
                )
            )
    else:
        next_actions.append("Mount profile management if this host should expose package-local profile readiness.")

    if profile_store.get("persistent") is False:
        warnings.append(
            _status_warning(
                "profile_store_not_persistent",
                "Profile management is using a non-persistent store.",
            )
        )

    external_registry_store = {"kind": "unknown", "persistent": None}
    external_servers = {"total": 0, "enabled": 0, "unavailable": 0}
    if external_registry_manager is not None:
        external_registry_store = _read_store_metadata(external_registry_manager)
        try:
            all_result = await external_registry_manager.list_servers(enabled=None)
            external_registry_store = _store_payload_from_result(all_result, external_registry_store)
            all_servers = all_result.get("servers", []) if isinstance(all_result, dict) else []
            total = len(all_servers) if isinstance(all_servers, list) else 0
            enabled = 0
            if isinstance(all_servers, list):
                enabled = sum(
                    1
                    for server in all_servers
                    if (
                        server.get("enabled", True)
                        if isinstance(server, dict)
                        else getattr(server, "enabled", True)
                    )
                )
            external_servers = {
                "total": total,
                "enabled": enabled,
                "unavailable": 0,
            }
        except GatewayExternalRegistryManagementError as exc:
            warnings.append(
                _status_warning(
                    exc.reason_code,
                    "External server registry readiness check failed.",
                )
            )
            external_servers["unavailable"] = external_servers["total"]
        except Exception:  # noqa: BLE001 - status must remain best-effort.
            logger.opt(exception=True).warning(
                "Gateway external registry readiness check failed"
            )
            warnings.append(
                _status_warning(
                    "external_registry_status_unavailable",
                    "External registry readiness check failed.",
                )
            )
            next_actions.append("Check external registry store configuration before using remote MCP servers.")

    if external_registry_store.get("persistent") is False:
        warnings.append(
            _status_warning(
                "external_registry_store_not_persistent",
                "External server registry is using a non-persistent store.",
            )
        )

    admin_auth_enabled = bool(admin_auth.enabled) if admin_auth is not None else False
    admin_auth_configured = (
        bool(admin_auth.api_key is not None or admin_auth.verifier is not None)
        if admin_auth is not None
        else False
    )
    admin_auth_payload = {
        "enabled": admin_auth_enabled,
        "configured": admin_auth_configured,
        "header_name": admin_auth.header_name if admin_auth_enabled and admin_auth is not None else None,
    }
    if admin_auth_enabled and not admin_auth_payload["configured"]:
        warnings.append(
            _status_warning(
                "admin_auth_not_configured",
                "Admin auth is enabled but no verifier is configured.",
            )
        )

    return {
        "status": "ok",
        "name": _runtime_name(runtime),
        "version": _runtime_version(runtime),
        "transport": {"base_path": "package-local-mounted", "mount_path": mount_path},
        "package": package_payload,
        "profile_store": profile_store,
        "default_profile": default_profile,
        "admin_auth": admin_auth_payload,
        "external_registry_store": external_registry_store,
        "external_servers": external_servers,
        "warnings": warnings,
        "next_actions": next_actions,
    }


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
    admin_auth: GatewayAdminAuthConfig | Mapping[str, Any] | None = None,
    credential_grant_manager: GatewayCredentialGrantManager | None = None,
    enable_credential_grant_management: bool = False,
    enable_policy_explain_management: bool = False,
    policy_explain_service: GatewayPolicyExplainService | None = None,
    policy_explain_profile_resolver: (
        Callable[[str], MCPProfile | Awaitable[MCPProfile | None]] | None
    ) = None,
    policy_explain_audit_store: AuditStore | None = None,
    policy_explain_permission_checker: GatewayAdminPermissionChecker | None = None,
) -> APIRouter:
    """Create a package-owned FastAPI router for a standalone MCP runtime."""

    router = APIRouter(route_class=_GatewayAdminAuthHandlingRoute)
    resolved_admin_auth = _resolve_admin_auth_config(
        admin_auth=admin_auth,
        profile_bootstrap=profile_bootstrap,
    )
    admin_dependencies = gateway_admin_auth_dependencies(resolved_admin_auth)
    resolved_profile_manager = _resolve_profile_manager(
        profile_manager=profile_manager,
        profile_bootstrap=profile_bootstrap,
        enable_profile_management=enable_profile_management,
    )
    if resolved_profile_manager is not None:
        _mount_profile_management_routes(
            router,
            resolved_profile_manager,
            admin_dependencies=admin_dependencies,
        )
    resolved_external_runtime_manager = _resolve_external_runtime_manager(
        external_runtime_manager=external_runtime_manager,
        profile_bootstrap=profile_bootstrap,
        enable_external_runtime_management=enable_external_runtime_management,
    )
    if resolved_external_runtime_manager is not None:
        _mount_external_runtime_routes(
            router,
            resolved_external_runtime_manager,
            admin_dependencies=admin_dependencies,
        )
    resolved_external_registry_manager = _resolve_external_registry_manager(
        external_registry_manager=external_registry_manager,
        profile_bootstrap=profile_bootstrap,
        enable_external_registry_management=enable_external_registry_management,
    )
    if resolved_external_registry_manager is not None:
        _mount_external_registry_routes(
            router,
            resolved_external_registry_manager,
            admin_dependencies=admin_dependencies,
        )
    resolved_credential_grant_manager = _resolve_credential_grant_manager(
        credential_grant_manager=credential_grant_manager,
        profile_bootstrap=profile_bootstrap,
        enable_credential_grant_management=enable_credential_grant_management,
    )
    if resolved_credential_grant_manager is not None:
        _mount_credential_grant_routes(
            router,
            resolved_credential_grant_manager,
            admin_dependencies=admin_dependencies,
        )
    if enable_policy_explain_management:
        _mount_policy_explain_routes(
            router,
            runtime,
            admin_auth=resolved_admin_auth,
            policy_explain_service=policy_explain_service,
            policy_explain_profile_resolver=policy_explain_profile_resolver,
            policy_explain_audit_store=policy_explain_audit_store,
            policy_explain_permission_checker=policy_explain_permission_checker,
        )

    @router.get("/status", response_model=GatewayReadinessStatusResponse)
    async def gateway_status(request: Request) -> dict[str, Any]:
        """Return best-effort package-local gateway readiness metadata."""

        return await _gateway_readiness_status(
            runtime,
            profile_manager=resolved_profile_manager,
            external_registry_manager=resolved_external_registry_manager,
            admin_auth=resolved_admin_auth,
            status_path=str(request.url.path),
        )

    @router.post(
        "/request",
        response_model=GatewayJSONRPCResponse | list[GatewayJSONRPCResponse],
        responses={204: {"description": "JSON-RPC notification accepted."}},
    )
    async def gateway_request(request: Request) -> GatewayJSONRPCResponse | list[GatewayJSONRPCResponse] | Response:
        """Process a raw JSON-RPC HTTP request body for the standalone gateway."""

        payload = await _parse_json_body(request)
        if isinstance(payload, _GATEWAY_RESPONSE_TYPES):
            return _to_http_response(payload)
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
    external_runtime_lifecycle: (
        GatewayExternalRuntimeLifecycleConfig | Mapping[str, Any] | None
    ) = None,
    admin_auth: GatewayAdminAuthConfig | Mapping[str, Any] | None = None,
    credential_grant_manager: GatewayCredentialGrantManager | None = None,
    enable_credential_grant_management: bool = False,
    enable_policy_explain_management: bool = False,
    policy_explain_service: GatewayPolicyExplainService | None = None,
    policy_explain_profile_resolver: (
        Callable[[str], MCPProfile | Awaitable[MCPProfile | None]] | None
    ) = None,
    policy_explain_audit_store: AuditStore | None = None,
    policy_explain_permission_checker: GatewayAdminPermissionChecker | None = None,
) -> FastAPI:
    """Create a minimal FastAPI app exposing the standalone MCP gateway router."""

    resolved_external_runtime_manager = _resolve_external_runtime_manager(
        external_runtime_manager=external_runtime_manager,
        profile_bootstrap=profile_bootstrap,
        enable_external_runtime_management=enable_external_runtime_management,
    )
    resolved_lifecycle = _resolve_external_runtime_lifecycle(
        external_runtime_lifecycle=external_runtime_lifecycle,
        profile_bootstrap=profile_bootstrap,
        external_runtime_manager=resolved_external_runtime_manager,
    )
    lifespan = (
        _create_external_runtime_lifespan(
            manager=resolved_external_runtime_manager,
            lifecycle=resolved_lifecycle,
        )
        if resolved_external_runtime_manager is not None and resolved_lifecycle.enabled
        else None
    )
    app = FastAPI(
        title="MCP Unified Gateway",
        version=_runtime_version(runtime),
        lifespan=lifespan,
    )
    app.add_exception_handler(
        GatewayAdminAuthError,
        gateway_admin_auth_error_response,
    )
    app.include_router(
        create_gateway_router(
            runtime,
            profile_manager=profile_manager,
            profile_bootstrap=profile_bootstrap,
            enable_profile_management=enable_profile_management,
            external_registry_manager=external_registry_manager,
            enable_external_registry_management=enable_external_registry_management,
            external_runtime_manager=resolved_external_runtime_manager,
            enable_external_runtime_management=enable_external_runtime_management,
            admin_auth=admin_auth,
            credential_grant_manager=credential_grant_manager,
            enable_credential_grant_management=enable_credential_grant_management,
            enable_policy_explain_management=enable_policy_explain_management,
            policy_explain_service=policy_explain_service,
            policy_explain_profile_resolver=policy_explain_profile_resolver,
            policy_explain_audit_store=policy_explain_audit_store,
            policy_explain_permission_checker=policy_explain_permission_checker,
        ),
        prefix=prefix,
    )
    return app
