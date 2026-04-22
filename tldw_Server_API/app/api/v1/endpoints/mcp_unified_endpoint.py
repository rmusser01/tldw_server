"""
Unified MCP API Endpoints

Production-ready endpoints for the unified MCP module with enhanced security and monitoring.

Environment:
    - ``MCP_SINGLE_USER_COMPAT_SHIM``: When set to ``0``/``false``/``off``, disables
      the single-user API key compatibility shim so that all API keys are validated
      via the multi-user API key manager path regardless of AUTH_MODE/PROFILE.
"""

import contextlib
import ipaddress
import os
import secrets
from dataclasses import dataclass
from typing import Any, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, Response, Security, WebSocket, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from loguru import logger
from pydantic import BaseModel, Field

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
    get_auth_principal,
    get_db_transaction,
    require_permissions,
)
from tldw_Server_API.app.api.v1.schemas.admin_schemas import ToolCatalogResponse
from tldw_Server_API.app.core.AuthNZ.api_key_manager import get_api_key_manager
from tldw_Server_API.app.core.AuthNZ.ip_allowlist import (
    is_single_user_ip_allowed,
    resolve_client_ip,
)
from tldw_Server_API.app.core.AuthNZ.orgs_teams import (
    list_active_team_memberships_for_user,
    list_org_memberships_for_user,
)
from tldw_Server_API.app.core.AuthNZ.permissions import SYSTEM_CONFIGURE, SYSTEM_LOGS
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.settings import (
    get_settings,
    is_single_user_profile_mode,
)
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import verify_jwt_and_fetch_user
from tldw_Server_API.app.core.MCP_unified import MCPRequest, MCPResponse, get_config, get_mcp_server
from tldw_Server_API.app.core.MCP_unified.auth import UserRole
from tldw_Server_API.app.core.MCP_unified.auth.jwt_manager import TokenData, get_jwt_manager
from tldw_Server_API.app.core.MCP_unified.monitoring.metrics import get_metrics_collector
from tldw_Server_API.app.core.MCP_unified.security.request_guards import enforce_http_security
from tldw_Server_API.app.core.MCP_unified.server import _is_authnz_access_token
from tldw_Server_API.app.core.Security.url_validation import assert_url_safe
from tldw_Server_API.app.core.feature_flags import is_mcp_hub_policy_enforcement_enabled
from tldw_Server_API.app.core.testing import env_flag_enabled, is_test_mode
from tldw_Server_API.app.services import admin_tool_catalog_service

_MCP_UNIFIED_NONCRITICAL_EXCEPTIONS = (
    OSError,
    ValueError,
    TypeError,
    KeyError,
    RuntimeError,
    AttributeError,
    ConnectionError,
    TimeoutError,
    ipaddress.AddressValueError,
    ipaddress.NetmaskValueError,
)
_ACCESS_TOKEN_TYPE = "access"  # nosec B105

# Create router
router = APIRouter(prefix="/mcp", tags=["mcp-unified"])

# Security
security = HTTPBearer(auto_error=False)


def _normalize_optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


# Request/Response models
class ServerStatusResponse(BaseModel):
    """Server status response"""
    status: str
    version: str
    uptime_seconds: float
    connections: dict[str, int]
    modules: dict[str, int]


class ServerMetricsResponse(BaseModel):
    """Server metrics response"""
    connections: dict[str, Any]
    modules: dict[str, dict[str, Any]]


class ToolExecutionRequest(BaseModel):
    """Tool execution request"""
    tool_name: str = Field(..., min_length=1, max_length=100)
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolExecutionResponse(BaseModel):
    """Tool execution response"""
    result: Any
    execution_time_ms: float
    module: str


class ModuleHealthResponse(BaseModel):
    """Module health response"""
    module_id: str
    status: str
    message: str
    checks: dict[str, bool]
    metrics: Optional[dict[str, Any]] = None


class AuthTokenRequest(BaseModel):
    """Authentication token request"""
    username: str
    password: Optional[str] = None
    api_key: Optional[str] = None


class AuthTokenResponse(BaseModel):
    """Authentication token response"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    refresh_token: Optional[str] = None


class AuthRefreshRequest(BaseModel):
    """Refresh authentication request payload."""
    refresh_token: str = Field(..., description="Refresh token value. Must be sent in the JSON body.")
    token_id: Optional[str] = Field(
        default=None,
        description="Optional refresh token identifier when known by the client.",
    )


def _get_derived_user_id(user: Optional[TokenData]) -> Optional[str]:
    """Return user.sub when available, otherwise None."""
    return user.sub if user else None


def _extract_api_key_permissions(info: Optional[dict[str, Any]]) -> list[str]:
    """Normalize API key scopes into MCP permissions."""
    if not info:
        return []
    try:
        from tldw_Server_API.app.core.AuthNZ.api_key_manager import normalize_scope
    except _MCP_UNIFIED_NONCRITICAL_EXCEPTIONS:
        return []

    raw_scopes = info.get("scopes")
    if raw_scopes is None:
        raw_scopes = info.get("scope")

    try:
        scopes = normalize_scope(raw_scopes)
    except _MCP_UNIFIED_NONCRITICAL_EXCEPTIONS:
        scopes = set()

    return sorted(scopes) if scopes else []


def _should_use_single_user_api_key_compat() -> bool:
    """
    Decide whether to use the single-user API key compatibility shim.

    Behaviour:
    - When MCP_SINGLE_USER_COMPAT_SHIM is explicitly disabled (\"0\"/\"false\"/\"off\"),
      the shim is turned off regardless of AUTH_MODE/PROFILE and API keys are
      always validated via the multi-user API key manager path.
    - Otherwise (default), the shim is enabled only when the runtime profile
      indicates a single-user-style deployment, mirroring the existing behaviour.
    """
    flag = os.getenv("MCP_SINGLE_USER_COMPAT_SHIM", "").strip().lower()
    if flag in {"0", "false", "off"}:
        return False
    try:
        return bool(is_single_user_profile_mode())
    except _MCP_UNIFIED_NONCRITICAL_EXCEPTIONS:
        logger.debug(
            "MCP unified: single-user profile detection failed; defaulting compat shim to False",
            extra={"auth_method": "single_user_compat_shim"},
            exc_info=True,
        )
        return False


def _get_client_ip(request: Optional[Request]) -> Optional[str]:
    """Extract client IP from the incoming request."""
    if request is None:
        return None
    try:
        return resolve_client_ip(request, get_settings())
    except _MCP_UNIFIED_NONCRITICAL_EXCEPTIONS:
        logger.debug("Failed to extract client IP", exc_info=True)
    return None


@dataclass
class McpAuthContext:
    """Resolved authentication context for MCP HTTP endpoints."""
    user: Optional[TokenData]
    principal: Optional[AuthPrincipal]
    api_key_info: Optional[dict[str, Any]]
    raw_api_key: Optional[str]


def _principal_to_token_data(principal: AuthPrincipal) -> Optional[TokenData]:
    """Best-effort conversion from claim-first principal to MCP TokenData."""
    sub: str = ""
    if principal.user_id is not None:
        sub = str(principal.user_id)
    elif principal.subject:
        sub = str(principal.subject)
    elif principal.api_key_id is not None:
        sub = f"api_key:{principal.api_key_id}"
    if not sub:
        return None
    return TokenData(
        sub=sub,
        username=principal.username,
        roles=list(principal.roles or []),
        permissions=list(principal.permissions or []),
        token_type=str(principal.token_type or "access"),
    )


# Dependency functions

async def _resolve_token_data_compat(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Security(security),
    x_api_key: Optional[str] = Header(None, alias="X-API-KEY"),
) -> Optional[TokenData]:
    """Compatibility resolver for TokenData (AuthNZ JWT, MCP JWT, or API key).

    The resolution order deliberately mirrors the main AuthNZ stack:
    1) AuthNZ access JWT (multi-user).
    2) MCP JWT (for tools integrations).
    3) API key:
       - Single-user mode: treat SINGLE_USER_API_KEY (and, in TEST_MODE,
         SINGLE_USER_TEST_API_KEY) as an admin-style principal.
       - Multi-user: validate via APIKeyManager and map to a user id.

    When the multi-user API key path succeeds, this helper also attaches the
    resolved API key metadata to ``request.state.mcp_api_key_info`` so that
    downstream handlers (for example, :func:`get_mcp_auth_context` and
    ``_attach_api_key_metadata``) can reuse the information without re-validating
    the same key or double-counting usage/audit events.
    """
    # Try AuthNZ JWT first (multi-user)
    authnz_token_failed = False
    token = credentials.credentials if credentials and credentials.credentials else None
    try:
        if token:
            if request is None:
                from starlette.requests import Request as _Request

                request = _Request({"type": "http", "headers": []})
            user = await verify_jwt_and_fetch_user(request, token)
            uid = str(getattr(user, "id", None))
            if uid:
                return TokenData(
                    sub=uid,
                    username=getattr(user, "username", None),
                    roles=list(getattr(user, "roles", []) or []),
                    permissions=list(getattr(user, "permissions", []) or []),
                    token_type=_ACCESS_TOKEN_TYPE,
                )
    except HTTPException:
        logger.debug(
            "AuthNZ JWT verification rejected the token",
            extra={"auth_method": "authnz_jwt"},
            exc_info=True,
        )
        if token and _is_authnz_access_token(token):
            logger.debug(
                "AuthNZ JWT rejected; not falling back to MCP JWT",
                extra={"auth_method": "authnz_jwt"},
                exc_info=True,
            )
            authnz_token_failed = True
            if not x_api_key:
                return None
        logger.debug(
            "AuthNZ JWT check failed; falling back to MCP JWT / API key",
            extra={"auth_method": "authnz_jwt"},
            exc_info=True,
        )
    except _MCP_UNIFIED_NONCRITICAL_EXCEPTIONS:
        logger.debug(
            "AuthNZ JWT verification raised an exception",
            extra={"auth_method": "authnz_jwt"},
            exc_info=True,
        )
        if token and _is_authnz_access_token(token):
            logger.debug(
                "AuthNZ JWT rejected; not falling back to MCP JWT",
                extra={"auth_method": "authnz_jwt"},
                exc_info=True,
            )
            authnz_token_failed = True
            if not x_api_key:
                return None
        logger.debug(
            "AuthNZ JWT check failed; falling back to MCP JWT / API key",
            extra={"auth_method": "authnz_jwt"},
            exc_info=True,
        )

    # MCP JWT fallback
    try:
        if not authnz_token_failed and credentials and credentials.credentials:
            jwt_manager = get_jwt_manager()
            return jwt_manager.verify_token(credentials.credentials)
    except _MCP_UNIFIED_NONCRITICAL_EXCEPTIONS:
        logger.debug(
            "MCP token verification failed; falling back to API key",
            extra={"auth_method": "mcp_jwt"},
            exc_info=True,
        )

    # API key fallback
    try:
        if x_api_key:
            # Single-user mode: accept the configured SINGLE_USER_API_KEY directly,
            # mirroring the semantics in get_request_user/get_auth_principal.
            try:
                if _should_use_single_user_api_key_compat():
                    settings = get_settings()
                    test_mode = is_test_mode()
                    if test_mode:
                        # Guard against accidental production use of TEST_MODE-based
                        # SINGLE_USER_TEST_API_KEY shortcuts. Only honor TEST_MODE
                        # in clear dev/test contexts (debug or explicit env/pytest).
                        try:
                            cfg = get_config()
                        except _MCP_UNIFIED_NONCRITICAL_EXCEPTIONS:
                            cfg = None
                        env = (
                            os.getenv("ENVIRONMENT")
                            or os.getenv("APP_ENV")
                            or os.getenv("ENV")
                            or ""
                        ).lower()
                        prod_flag = env_flag_enabled("tldw_production")
                        is_dev_ctx = bool(cfg and getattr(cfg, "debug_mode", False))
                        if os.getenv("PYTEST_CURRENT_TEST") is not None:
                            is_dev_ctx = True
                        try:
                            import sys as _sys

                            if "pytest" in _sys.modules:
                                is_dev_ctx = True
                        except _MCP_UNIFIED_NONCRITICAL_EXCEPTIONS:
                            pass
                        if env in {"dev", "development", "test", "ci"}:
                            is_dev_ctx = True
                        if prod_flag:
                            is_dev_ctx = False
                        if not is_dev_ctx:
                            logger.error(
                                "TEST_MODE enabled outside dev/test context; refusing SINGLE_USER_TEST_API_KEY",
                                extra={"audit": True, "env": env, "debug_mode": bool(cfg and getattr(cfg, 'debug_mode', False))},
                            )
                            test_mode = False
                    allowed: set[str] = set()
                    if settings.SINGLE_USER_API_KEY:
                        allowed.add(settings.SINGLE_USER_API_KEY)
                    if test_mode:
                        test_key = os.getenv("SINGLE_USER_TEST_API_KEY")
                        if test_key:
                            allowed.add(test_key)
                    if x_api_key in {a for a in allowed if a}:
                        client_ip = _get_client_ip(request)
                        if not is_single_user_ip_allowed(client_ip, settings):
                            return None
                        roles = [UserRole.ADMIN.value]
                        perms = ["*"] if test_mode else []
                        return TokenData(
                            sub=str(settings.SINGLE_USER_FIXED_ID),
                            username="single_user",
                            roles=roles,
                            permissions=perms,
                            token_type=_ACCESS_TOKEN_TYPE,
                        )
            except _MCP_UNIFIED_NONCRITICAL_EXCEPTIONS:
                # Fall through to multi-user API key validation
                logger.debug(
                    "Single-user API key check failed, falling through to multi-user validation",
                    extra={"auth_method": "single_user_api_key"},
                    exc_info=True,
                )
            api_mgr = await get_api_key_manager()
            client_ip = _get_client_ip(request)
            info = await api_mgr.validate_api_key(x_api_key, ip_address=client_ip)
            if info and info.get("user_id"):
                # Attach API key metadata to the request state so downstream
                # handlers can reuse it without re-validating (avoids double
                # usage/audit updates when the compatibility resolver is used as a
                # dependency alongside per-endpoint validate_api_key calls).
                try:
                    if request is not None:
                        request.state.mcp_api_key_info = info
                except _MCP_UNIFIED_NONCRITICAL_EXCEPTIONS:
                    logger.debug(
                        "MCP unified: failed to attach API key info to request state",
                        extra={"auth_method": "api_key"},
                        exc_info=True,
                    )
                return TokenData(
                    sub=str(info["user_id"]),
                    username=None,
                    roles=["api_client"],
                    permissions=_extract_api_key_permissions(info),
                    token_type=_ACCESS_TOKEN_TYPE,
                )
    except _MCP_UNIFIED_NONCRITICAL_EXCEPTIONS:
        logger.debug(
            "API key check failed in MCP unified compatibility token resolver",
            extra={"auth_method": "api_key"},
            exc_info=True,
        )

    return None


async def _get_optional_auth_principal(request: Request) -> Optional[AuthPrincipal]:
    """Best-effort principal resolver for optional MCP auth context."""
    try:
        current = getattr(request.state, "auth", None)
        if isinstance(current, AuthPrincipal):
            return current
    except _MCP_UNIFIED_NONCRITICAL_EXCEPTIONS:
        logger.debug(
            "MCP unified: failed to read request.state.auth for optional principal",
            exc_info=True,
        )

    try:
        return await get_auth_principal(request)
    except HTTPException:
        return None
    except _MCP_UNIFIED_NONCRITICAL_EXCEPTIONS:
        logger.debug(
            "MCP unified: optional principal resolution failed",
            exc_info=True,
        )
        return None


async def _get_optional_token_data(
    request: Request,
    principal: Optional[AuthPrincipal] = Depends(_get_optional_auth_principal),
    credentials: HTTPAuthorizationCredentials = Security(security),
    x_api_key: Optional[str] = Header(None, alias="X-API-KEY"),
) -> Optional[TokenData]:
    """Claim-first TokenData resolver with compatibility fallback."""
    if principal is not None:
        token_data = _principal_to_token_data(principal)
        if token_data is not None:
            return token_data
    return await _resolve_token_data_compat(
        request=request,
        credentials=credentials,
        x_api_key=x_api_key,
    )


async def get_mcp_auth_context(
    request: Request,
    user: Optional[TokenData] = Depends(_get_optional_token_data),
    principal: Optional[AuthPrincipal] = Depends(_get_optional_auth_principal),
    x_api_key: Optional[str] = Header(None, alias="X-API-KEY"),
) -> McpAuthContext:
    """Resolve MCP auth context (user + API key metadata) for HTTP endpoints.

    Prefers the claim-first principal for user claims and falls back to
    :func:`_resolve_token_data_compat` only when needed for compatibility. Surfaces any
    API key metadata attached to ``request.state.mcp_api_key_info`` by the
    multi-user API key path. HTTP handlers should rely on the resulting
    :class:`McpAuthContext` (and helper functions like ``_attach_api_key_metadata``)
    rather than re-validating API keys themselves.
    """
    api_key_info: Optional[dict[str, Any]] = None
    try:
        if request is not None:
            api_key_info = getattr(request.state, "mcp_api_key_info", None)
    except _MCP_UNIFIED_NONCRITICAL_EXCEPTIONS:
        logger.debug(
            "MCP unified: failed to read API key info from request state",
            exc_info=True,
        )
        api_key_info = None
    return McpAuthContext(
        user=user,
        principal=principal,
        api_key_info=api_key_info,
        raw_api_key=x_api_key,
    )


async def _attach_api_key_metadata(
    auth: McpAuthContext,
    http_request: Optional[Request],
    *,
    log_on_error: bool = False,
    log_prefix: str = "HTTP",
) -> dict[str, Any]:
    """Attach API-key-derived metadata (org/team) for MCP HTTP endpoints.

    Prefers any API key info already attached to the auth context and falls back
    to validating the raw API key when needed.

    Re-validation can trigger usage and audit side effects via the API key
    manager (for example, incrementing usage counters or emitting audit logs),
    so this helper intentionally avoids re-validating when cached API key info
    is already available on the auth context.
    """
    metadata: dict[str, Any] = {}
    api_key_info: Optional[dict[str, Any]] = None

    if auth.api_key_info is not None:
        api_key_info = auth.api_key_info

    if auth.raw_api_key and api_key_info is None:
        try:
            api_mgr = await get_api_key_manager()
            client_ip = _get_client_ip(http_request)
            api_key_info = await api_mgr.validate_api_key(auth.raw_api_key, ip_address=client_ip)
        except _MCP_UNIFIED_NONCRITICAL_EXCEPTIONS:
            if log_on_error:
                logger.debug(
                    "MCP unified {} API key metadata attach failed",
                    log_prefix,
                    exc_info=True,
                )
            api_key_info = None

    if api_key_info:
        if api_key_info.get("org_id") is not None:
            metadata["org_id"] = api_key_info.get("org_id")
        if api_key_info.get("team_id") is not None:
            metadata["team_id"] = api_key_info.get("team_id")
        try:
            from tldw_Server_API.app.core.AuthNZ.api_key_manager import normalize_scope
            raw_scopes = api_key_info.get("scopes")
            if raw_scopes is None:
                raw_scopes = api_key_info.get("scope")
            scopes = normalize_scope(raw_scopes)
            if scopes:
                metadata["api_key_scopes"] = sorted(scopes)
                metadata["auth_via"] = "api_key"
        except _MCP_UNIFIED_NONCRITICAL_EXCEPTIONS:
            logger.debug(
                "MCP unified {} API key scope normalization failed",
                log_prefix,
                exc_info=True,
            )

    _attach_rg_ingress_metadata(metadata, http_request)
    _attach_workspace_ingress_metadata(metadata, http_request)
    return metadata


def _attach_rg_ingress_metadata(metadata: dict[str, Any], http_request: Optional[Request]) -> None:
    """Attach RG ingress metadata when policy has been enforced upstream."""
    if not http_request:
        return
    try:
        policy_id = getattr(http_request.state, "rg_policy_id", None)
        if policy_id:
            metadata["rg_ingress_enforced"] = True
            metadata["rg_policy_id"] = str(policy_id)
    except _MCP_UNIFIED_NONCRITICAL_EXCEPTIONS as exc:
        logger.debug(
            "Failed to read rg_policy_id from request state",
            error=str(exc),
            exc_info=True,
        )


def _attach_workspace_ingress_metadata(
    metadata: dict[str, Any],
    http_request: Optional[Request],
) -> None:
    """Attach trusted direct-ingress workspace metadata for MCP path scoping."""
    if not http_request:
        return
    try:
        workspace_id = str(http_request.headers.get("x-tldw-workspace-id") or "").strip()
        if workspace_id:
            metadata["workspace_id"] = workspace_id
        cwd = str(http_request.headers.get("x-tldw-cwd") or "").strip()
        if cwd:
            metadata["cwd"] = cwd
    except _MCP_UNIFIED_NONCRITICAL_EXCEPTIONS as exc:
        logger.debug(
            "Failed to read direct MCP workspace headers",
            error=str(exc),
            exc_info=True,
        )


def _resolve_auth_principal_from_request(request: Optional[Request]) -> Optional[AuthPrincipal]:
    """Best-effort resolver for AuthPrincipal already attached to request state."""
    if request is None:
        return None
    try:
        principal = getattr(request.state, "auth", None)
        if isinstance(principal, AuthPrincipal):
            return principal
    except _MCP_UNIFIED_NONCRITICAL_EXCEPTIONS:
        logger.debug(
            "MCP unified: failed to resolve request.state.auth principal",
            exc_info=True,
        )
    return None


def _principal_has_admin_claims(principal: Optional[AuthPrincipal]) -> bool:
    """Return True when principal carries admin-style claims."""
    if principal is None:
        return False
    try:
        roles = {
            str(role).strip().lower()
            for role in (getattr(principal, "roles", []) or [])
            if str(role).strip()
        }
        permissions = {
            str(perm).strip().lower()
            for perm in (getattr(principal, "permissions", []) or [])
            if str(perm).strip()
        }
        if "admin" in roles:
            return True
        if "*" in permissions:
            return True
        if SYSTEM_CONFIGURE in permissions:
            return True
    except _MCP_UNIFIED_NONCRITICAL_EXCEPTIONS:
        return False
    return False


def _is_catalog_admin_context(
    request: Optional[Request],
    user: Optional[TokenData],
    principal: Optional[AuthPrincipal] = None,
) -> bool:
    """Claim-first admin check for MCP tool catalogs visibility."""
    if principal is not None:
        return _principal_has_admin_claims(principal)

    principal = _resolve_auth_principal_from_request(request)
    if principal is not None:
        return _principal_has_admin_claims(principal)

    # Compatibility fallback when no principal is attached to request.state.
    try:
        roles = list(getattr(user, "roles", []) or [])
    except _MCP_UNIFIED_NONCRITICAL_EXCEPTIONS:
        return False
    return any(str(role).lower() == "admin" for role in roles)


# WebSocket endpoint

@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    client_id: Optional[str] = Query(None, description="Client identifier"),
    token: Optional[str] = Query(None, description="Authentication token"),
    api_key: Optional[str] = Query(None, description="API key for authentication"),
    mcp_session_id: Optional[str] = Query(None, description="Stable MCP session identifier"),
    workspace_id: Optional[str] = Query(None, description="Trusted workspace identifier"),
    cwd: Optional[str] = Query(None, description="Current working directory within the workspace"),
):
    """
    WebSocket endpoint for MCP protocol.

    Supports:
    - Full MCP protocol over WebSocket
    - Optional authentication via token parameter
    - Real-time bidirectional communication
    - Automatic reconnection support

    Example:
    ```javascript
    const ws = new WebSocket('ws://localhost:8000/api/v1/mcp/ws?client_id=my-client&token=jwt-token');

    ws.onopen = () => {
        ws.send(JSON.stringify({
            jsonrpc: "2.0",
            method: "initialize",
            params: {
                clientInfo: {
                    name: "My Client",
                    version: "1.0.0"
                }
            },
            id: 1
        }));
    };
    ```
    """
    server = get_mcp_server()

    # Ensure server is initialized
    if not server.initialized:
        await server.initialize()

    # Handle WebSocket connection
    await server.handle_websocket(
        websocket,
        client_id=_normalize_optional_text(client_id),
        auth_token=_normalize_optional_text(token),
        api_key=_normalize_optional_text(api_key),
        mcp_session_id=_normalize_optional_text(mcp_session_id),
        workspace_id=_normalize_optional_text(workspace_id),
        cwd=_normalize_optional_text(cwd),
    )


# HTTP endpoints

@router.post("/request", response_model=MCPResponse)
async def mcp_request(
    request: MCPRequest,
    http_request: Request,
    client_id: Optional[str] = Query(None, description="Client identifier"),
    auth: McpAuthContext = Depends(get_mcp_auth_context),
    mcp_session_id: Optional[str] = Header(None, alias="mcp-session-id"),
    config: Optional[str] = Query(None, description="Base64-encoded JSON safe config for this request"),
    response: Response = None,
    _guard: None = Depends(enforce_http_security),
):
    """
    Process an MCP request via HTTP.

    This endpoint provides a simpler alternative to WebSocket for clients
    that don't need real-time bidirectional communication.
    """
    server = get_mcp_server()

    # Ensure server is initialized
    if not server.initialized:
        await server.initialize()

    # Process request
    # Attach org/team metadata when auth via API key. Prefer any metadata attached
    # by the compatibility resolver to avoid re-validating the same key and
    # double-counting usage/audit; fall back to a direct lookup when needed.
    # usage/audit; fall back to a direct lookup when needed.
    metadata = await _attach_api_key_metadata(auth, http_request)

    # Derive user id from the authenticated token user when present.
    derived_user_id = _get_derived_user_id(auth.user)

    # Parse optional safe config (base64-encoded JSON)
    safe_config: dict[str, Any] = {}
    if config:
        try:
            import base64
            import json as _json
            decoded = base64.b64decode(config).decode("utf-8")
            cfg = _json.loads(decoded)
            if isinstance(cfg, dict):
                safe_config = cfg
        except _MCP_UNIFIED_NONCRITICAL_EXCEPTIONS as e:
            logger.debug(f"Failed to parse safe config: {e}")

    # Session lifecycle: if initialize and no session id provided, generate one and return header
    try:
        if request.method == "initialize" and not mcp_session_id:
            import uuid as _uuid
            mcp_session_id = _uuid.uuid4().hex
            if response is not None:
                response.headers["mcp-session-id"] = mcp_session_id
    except _MCP_UNIFIED_NONCRITICAL_EXCEPTIONS:
        logger.debug("Failed to generate session ID for initialize request", exc_info=True)

    if auth.user:
        if auth.user.roles:
            metadata.setdefault("roles", auth.user.roles)
        if auth.user.permissions:
            metadata.setdefault("permissions", auth.user.permissions)

    metadata["mcp_policy_context_enabled"] = is_mcp_hub_policy_enforcement_enabled()
    if mcp_session_id:
        metadata["session_id"] = mcp_session_id
    if safe_config:
        metadata["safe_config"] = safe_config

    resp_obj = await server.handle_http_request(
        request,
        client_id=client_id,
        user_id=derived_user_id,
        metadata=metadata or None
    )
    if resp_obj is None:
        return Response(status_code=204)
    # Convert authorization errors to HTTP 403 with hint for HTTP clients
    if resp_obj.error and resp_obj.error.code == -32001:
        hint = None
        try:
            if request.method == "tools/call":
                tname = (request.params or {}).get("name") if isinstance(request.params, dict) else None
                if tname:
                    hint = f"Permission denied. Ask an admin to grant tools.execute:{tname} or tools.execute:* to your role (Admin → Access Control)."
        except _MCP_UNIFIED_NONCRITICAL_EXCEPTIONS:
            hint = None
        raise HTTPException(status_code=403, detail={
            "message": resp_obj.error.message or "Insufficient permissions",
            "hint": hint or "Insufficient permissions for this operation"
        })

    return resp_obj


@router.post("/request/batch", response_model=list[MCPResponse])
async def mcp_request_batch(
    requests: list[MCPRequest],
    http_request: Request,
    client_id: Optional[str] = Query(None, description="Client identifier"),
    auth: McpAuthContext = Depends(get_mcp_auth_context),
    mcp_session_id: Optional[str] = Header(None, alias="mcp-session-id"),
    config: Optional[str] = Query(None, description="Base64-encoded JSON safe config for this request"),
    response: Response = None,
    _guard: None = Depends(enforce_http_security),
):
    """
    Process a batch of MCP requests via HTTP.
    Accepts a JSON array of JSON-RPC 2.0 requests and returns an array
    of responses. Notifications (no id) are dropped per spec.
    """
    server = get_mcp_server()
    if not server.initialized:
        await server.initialize()

    # Attach org/team metadata when auth via API key. Prefer any metadata attached
    # by the compatibility resolver to avoid re-validating the same key and
    # double-counting usage/audit; fall back to a direct lookup when needed.
    # usage/audit; fall back to a direct lookup when needed.
    metadata = await _attach_api_key_metadata(
        auth,
        http_request,
        log_on_error=True,
        log_prefix="batch",
    )

    # Derive user id from the authenticated token user when present.
    derived_user_id = _get_derived_user_id(auth.user)

    # Optional safe config
    safe_config: dict[str, Any] = {}
    if config:
        try:
            import base64
            import json as _json
            decoded = base64.b64decode(config).decode("utf-8")
            cfg = _json.loads(decoded)
            if isinstance(cfg, dict):
                safe_config = cfg
        except _MCP_UNIFIED_NONCRITICAL_EXCEPTIONS as e:
            logger.debug(f"Batch failed to parse safe config: {e}")

    # If any initialize request is present and no session id was provided, generate one.
    try:
        if not mcp_session_id and any(req.method == "initialize" for req in requests):
            import uuid as _uuid
            mcp_session_id = _uuid.uuid4().hex
            if response is not None:
                response.headers["mcp-session-id"] = mcp_session_id
    except _MCP_UNIFIED_NONCRITICAL_EXCEPTIONS:
        logger.debug("Failed to generate session ID for batch initialize", exc_info=True)

    # Build metadata for batch processing
    if auth.user:
        if auth.user.roles:
            metadata.setdefault("roles", auth.user.roles)
        if auth.user.permissions:
            metadata.setdefault("permissions", auth.user.permissions)
    metadata["mcp_policy_context_enabled"] = is_mcp_hub_policy_enforcement_enabled()
    if mcp_session_id:
        metadata["session_id"] = mcp_session_id
    if safe_config:
        metadata["safe_config"] = safe_config

    resp = await server.handle_http_batch(
        requests,
        client_id=client_id,
        user_id=derived_user_id,
        metadata=metadata or None
    )
    # If only notifications were sent, return empty list
    return resp or []


@router.get("/status", response_model=ServerStatusResponse)
async def get_server_status(
    _guard: None = Depends(enforce_http_security),
):
    """
    Get MCP server status.

    Returns:
    - Server health status
    - Uptime
    - Connection statistics
    - Module health summary
    """
    server = get_mcp_server()

    # Ensure server is initialized
    if not server.initialized:
        await server.initialize()

    server_status = await server.get_status()
    return ServerStatusResponse(**server_status)


@router.get("/metrics", response_model=ServerMetricsResponse)
async def get_server_metrics(
    _principal: AuthPrincipal = Depends(require_permissions(SYSTEM_LOGS)),
    _guard: None = Depends(enforce_http_security),
):
    """
    Get detailed server metrics (requires `system.logs` permission or admin).

    Returns:
    - Connection metrics
    - Module performance metrics
    - Error rates
    - Latency statistics
    """
    server = get_mcp_server()

    if not server.initialized:
        await server.initialize()

    metrics = await server.get_metrics()
    return ServerMetricsResponse(**metrics)


@router.get("/metrics/prometheus")
async def get_prometheus_metrics(
    _principal: AuthPrincipal = Depends(require_permissions(SYSTEM_LOGS)),
    _guard: None = Depends(enforce_http_security),
):
    """
    Prometheus scrape endpoint for MCP metrics.

    Security: Requires an authenticated principal with the `system.logs`
    permission (or admin-style claims via require_permissions). External
    ingress or Prometheus-side configuration should be used to handle any
    additional network-level access controls.
    """
    collector = get_metrics_collector()
    content = collector.get_prometheus_metrics()
    # Use standard Prometheus content type regardless of availability
    return Response(content=content, media_type="text/plain; version=0.0.4; charset=utf-8")


@router.get(
    "/tools",
    summary="List MCP tools",
    description=(
        "List available MCP tools. Tools are filtered by RBAC; catalog filters shape discovery only.\n\n"
        "Parameters:\n"
        "- `module`: Filter by module id (repeat or comma-separate for multiple modules).\n"
        "- `catalog`: Filter by tool catalog name. The server resolves the name with precedence\n"
        "  `team > org > global` based on the authenticated context (API key or JWT). When both\n"
        "  `catalog` and `catalog_id` are provided, `catalog_id` takes precedence.\n"
        "- `catalog_id`: Filter by tool catalog id directly.\n\n"
        "- `catalog_strict`: When true, unresolved catalogs return an empty tool list instead\n"
        "  of falling back to the full discovery set.\n\n"
        "Behavior:\n"
        "- If the catalog name/id cannot be resolved, the server fails open (no catalog filter),\n"
        "  but RBAC is still enforced unless `catalog_strict` is enabled.\n"
        "- `canExecute` reflects the caller's permissions. Catalog membership does not grant\n"
        "  execution rights; it only affects discovery."
    ),
)
async def list_tools(
    http_request: Request,
    module: Optional[list[str]] = Query(None, description="Filter by module(s)"),
    catalog: Optional[str] = Query(None, description="Filter by tool catalog name"),
    catalog_id: Optional[int] = Query(None, description="Filter by tool catalog id"),
    catalog_strict: Optional[bool] = Query(None, description="Fail closed on unresolved catalogs"),
    auth: McpAuthContext = Depends(get_mcp_auth_context),
    _guard: None = Depends(enforce_http_security),
):
    """
    List available MCP tools.

    Returns tools with their descriptions and required parameters.
    Tools are filtered based on user permissions if authenticated.
    """
    params: dict[str, Any] = {}
    if module:
        module_values: list[str] = []
        for item in module:
            if not isinstance(item, str):
                continue
            for part in item.split(","):
                part = part.strip()
                if part:
                    module_values.append(part)
        if module_values:
            params["module"] = module_values
    if catalog is not None:
        params["catalog"] = catalog
    if catalog_id is not None:
        params["catalog_id"] = catalog_id
    if catalog_strict is not None:
        params["catalog_strict"] = catalog_strict
    request = MCPRequest(method="tools/list", params=params, id="http-tools-list")

    server = get_mcp_server()
    if not server.initialized:
        await server.initialize()

    # Derive user id from the authenticated token user when present.
    user = auth.user
    derived_user_id = _get_derived_user_id(user)

    metadata = await _attach_api_key_metadata(auth, http_request)
    if user:
        if user.roles:
            metadata["roles"] = user.roles
        if user.permissions:
            metadata["permissions"] = user.permissions

    response = await server.handle_http_request(
        request,
        user_id=derived_user_id,
        metadata=metadata or None
    )

    if response.error:
        if response.error.code == -32001:
            raise HTTPException(status_code=403, detail={
                "message": response.error.message or "Insufficient permissions",
                "hint": "Permission denied for listing tools. Contact an admin."
            })
        raise HTTPException(status_code=500, detail=response.error.message)

    return response.result


@router.get(
    "/tool_catalogs",
    response_model=list[ToolCatalogResponse],
    summary="List MCP tool catalogs visible to the current user",
    description=(
        "Return MCP tool catalogs scoped to the current user context.\n\n"
        "Scopes:\n"
        "- global: org_id and team_id are NULL\n"
        "- org: org_id matches any org the user belongs to\n"
        "- team: team_id matches any team the user belongs to\n\n"
        "Admins can see all catalogs.\n"
    ),
)
async def list_tool_catalogs(
    http_request: Request,
    scope: str = Query("all", description="Filter by scope: all|global|org|team"),
    auth: McpAuthContext = Depends(get_mcp_auth_context),
    _guard: None = Depends(enforce_http_security),
    db=Depends(get_db_transaction),
) -> list[ToolCatalogResponse]:
    user = auth.user
    principal = auth.principal
    metadata = await _attach_api_key_metadata(auth, http_request)

    if user is None and principal is None and not metadata:
        raise HTTPException(status_code=403, detail="Authentication required")

    admin_all = _is_catalog_admin_context(http_request, user, principal)

    org_ids: set[int] = set()
    team_ids: set[int] = set()
    if metadata.get("org_id") is not None:
        with contextlib.suppress(_MCP_UNIFIED_NONCRITICAL_EXCEPTIONS):
            org_ids.add(int(metadata["org_id"]))
    if metadata.get("team_id") is not None:
        with contextlib.suppress(_MCP_UNIFIED_NONCRITICAL_EXCEPTIONS):
            team_ids.add(int(metadata["team_id"]))

    if principal is not None and not admin_all:
        for principal_org_id in list(getattr(principal, "org_ids", []) or []):
            with contextlib.suppress(_MCP_UNIFIED_NONCRITICAL_EXCEPTIONS):
                org_ids.add(int(principal_org_id))
        for principal_team_id in list(getattr(principal, "team_ids", []) or []):
            with contextlib.suppress(_MCP_UNIFIED_NONCRITICAL_EXCEPTIONS):
                team_ids.add(int(principal_team_id))

    if user and not admin_all:
        try:
            uid = int(user.sub)
        except _MCP_UNIFIED_NONCRITICAL_EXCEPTIONS:
            uid = None
        if uid is not None:
            try:
                memberships = await list_org_memberships_for_user(uid)
                for m in memberships:
                    try:
                        org_id = int(m.get("org_id"))
                        org_ids.add(org_id)
                    except _MCP_UNIFIED_NONCRITICAL_EXCEPTIONS:
                        continue
            except _MCP_UNIFIED_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug(f"MCP tool catalogs: org membership lookup failed: {exc}")

            try:
                memberships = await list_active_team_memberships_for_user(uid)
                for membership in memberships:
                    try:
                        team_id_val = (
                            membership.get("team_id")
                            if isinstance(membership, dict)
                            else None
                        )
                        if team_id_val is not None:
                            team_ids.add(int(team_id_val))
                    except _MCP_UNIFIED_NONCRITICAL_EXCEPTIONS:
                        continue
            except _MCP_UNIFIED_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug(f"MCP tool catalogs: team membership lookup failed: {exc}")

    scope_norm = scope.strip().lower()
    if scope_norm not in {"all", "global", "org", "team"}:
        raise HTTPException(status_code=422, detail="Invalid scope value")

    result_rows = await admin_tool_catalog_service.list_visible_tool_catalogs(
        db,
        scope_norm=scope_norm,
        admin_all=admin_all,
        org_ids=org_ids,
        team_ids=team_ids,
    )

    unique: dict[int, ToolCatalogResponse] = {}
    for row in result_rows:
        try:
            entry = ToolCatalogResponse(**row)
            unique[entry.id] = entry
        except _MCP_UNIFIED_NONCRITICAL_EXCEPTIONS:
            continue
    return list(unique.values())


@router.post("/tools/execute", response_model=ToolExecutionResponse)
async def execute_tool(
    request: ToolExecutionRequest,
    http_request: Request,
    auth: McpAuthContext = Depends(get_mcp_auth_context),
    _guard: None = Depends(enforce_http_security),
):
    """
    Execute a specific tool (requires authentication).

    Tools are executed with user context for permission checking.
    """
    import time
    start_time = time.time()

    mcp_request = MCPRequest(
        method="tools/call",
        params={
            "name": request.tool_name,
            "arguments": request.arguments
        },
        id=f"http-tools-execute:{request.tool_name}"
    )

    server = get_mcp_server()
    if not server.initialized:
        await server.initialize()

    if not auth.user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = auth.user
    metadata = await _attach_api_key_metadata(auth, http_request, log_on_error=True, log_prefix="tools_execute")
    if user.roles:
        metadata["roles"] = user.roles
    if user.permissions:
        metadata["permissions"] = user.permissions

    derived_user_id = _get_derived_user_id(user)

    response = await server.handle_http_request(
        mcp_request,
        user_id=derived_user_id,
        metadata=metadata or None
    )

    if response is None:
        logger.error("MCP server returned no response for tools/call", tool=request.tool_name)
        raise HTTPException(status_code=502, detail="MCP tool execution returned no response")

    if response.error:
        # Map authorization failures to 403 with a helpful hint
        if response.error.code == -32001:  # AUTHORIZATION_ERROR
            hint = {
                "message": response.error.message or "Insufficient permissions",
                "hint": (
                    f"Permission denied. Ask an admin to grant tools.execute:{request.tool_name} "
                    f"or tools.execute:* to your role (Admin → Access Control)."
                )
            }
            raise HTTPException(status_code=403, detail=hint)
        # Invalid params
        if response.error.code == -32602:
            raise HTTPException(status_code=400, detail=response.error.message)
        # Other errors
        raise HTTPException(status_code=500, detail=response.error.message)

    execution_time = (time.time() - start_time) * 1000

    result_payload = response.result
    # Back-compat: unwrap plain text content to raw string when possible
    display_result = result_payload
    try:
        if isinstance(result_payload, dict):
            content = result_payload.get("content")
            if isinstance(content, list) and content:
                first = content[0]
                if isinstance(first, dict) and first.get("type") == "text" and isinstance(first.get("text"), str):
                    display_result = first.get("text")
    except _MCP_UNIFIED_NONCRITICAL_EXCEPTIONS:
        display_result = result_payload

    served_by = None
    try:
        if isinstance(result_payload, dict):
            served_by = result_payload.get("module")
    except _MCP_UNIFIED_NONCRITICAL_EXCEPTIONS:
        served_by = None

    return ToolExecutionResponse(
        result=display_result,
        execution_time_ms=execution_time,
        module=served_by or "unknown"
    )


@router.get("/modules")
async def list_modules(
    http_request: Request,
    auth: McpAuthContext = Depends(get_mcp_auth_context),
    _guard: None = Depends(enforce_http_security),
):
    """
    List registered MCP modules.

    Returns module information including status and capabilities.
    """
    request = MCPRequest(method="modules/list", id="http-modules-list")

    server = get_mcp_server()
    if not server.initialized:
        await server.initialize()

    # Derive user id from the authenticated token user when present.
    user = auth.user
    derived_user_id = _get_derived_user_id(user)

    metadata = await _attach_api_key_metadata(auth, http_request)
    if user:
        if user.roles:
            metadata["roles"] = user.roles
        if user.permissions:
            metadata["permissions"] = user.permissions

    response = await server.handle_http_request(
        request,
        user_id=derived_user_id,
        metadata=metadata or None
    )

    if response.error:
        if response.error.code == -32001:
            raise HTTPException(status_code=403, detail={
                "message": response.error.message or "Insufficient permissions",
                "hint": "Permission denied for listing tools. Contact an admin."
            })
        raise HTTPException(status_code=500, detail=response.error.message)

    return response.result


@router.get("/modules/health")
async def get_modules_health(
    http_request: Request,
    principal: AuthPrincipal = Depends(require_permissions(SYSTEM_LOGS)),
    _guard: None = Depends(enforce_http_security),
):
    """
    Get detailed health status of all modules; requires `system.logs` permission (or admin).

    Returns health checks and metrics for each module.
    """
    request = MCPRequest(method="modules/health", id="http-modules-health")

    server = get_mcp_server()
    if not server.initialized:
        await server.initialize()

    meta: dict[str, Any] = {"admin_override": True}
    if principal.roles:
        meta["roles"] = principal.roles
    if principal.permissions:
        meta["permissions"] = principal.permissions
    _attach_rg_ingress_metadata(meta, http_request)

    response = await server.handle_http_request(request, user_id=principal.principal_id, metadata=meta)

    if response.error:
        if response.error.code == -32001:
            raise HTTPException(status_code=403, detail={
                "message": response.error.message or "Insufficient permissions",
                "hint": "Permission denied for listing modules. Contact an admin."
            })
        raise HTTPException(status_code=500, detail=response.error.message)

    return response.result


@router.get("/resources")
async def list_resources(
    http_request: Request,
    auth: McpAuthContext = Depends(get_mcp_auth_context),
    _guard: None = Depends(enforce_http_security),
):
    """
    List available MCP resources.

    Resources are filtered based on user permissions if authenticated.
    """
    request = MCPRequest(method="resources/list", id="http-resources-list")

    server = get_mcp_server()
    if not server.initialized:
        await server.initialize()

    # Derive user id from the authenticated token user when present.
    user = auth.user
    derived_user_id = _get_derived_user_id(user)

    metadata = await _attach_api_key_metadata(auth, http_request)
    if user:
        if user.roles:
            metadata["roles"] = user.roles
        if user.permissions:
            metadata["permissions"] = user.permissions

    response = await server.handle_http_request(
        request,
        user_id=derived_user_id,
        metadata=metadata or None
    )

    if response.error:
        if response.error.code == -32001:
            raise HTTPException(status_code=403, detail={
                "message": response.error.message or "Insufficient permissions",
                "hint": "Permission denied for listing resources. Contact an admin."
            })
        raise HTTPException(status_code=500, detail=response.error.message)

    return response.result


@router.get("/prompts")
async def list_prompts(
    http_request: Request,
    auth: McpAuthContext = Depends(get_mcp_auth_context),
    _guard: None = Depends(enforce_http_security),
):
    """
    List available MCP prompts.

    Prompts are filtered based on user permissions if authenticated.
    """
    request = MCPRequest(method="prompts/list", id="http-prompts-list")

    server = get_mcp_server()
    if not server.initialized:
        await server.initialize()

    # Derive user id from the authenticated token user when present.
    user = auth.user
    derived_user_id = _get_derived_user_id(user)

    metadata = await _attach_api_key_metadata(auth, http_request)
    if user:
        if user.roles:
            metadata["roles"] = user.roles
        if user.permissions:
            metadata["permissions"] = user.permissions

    response = await server.handle_http_request(
        request,
        user_id=derived_user_id,
        metadata=metadata or None
    )

    if response.error:
        if response.error.code == -32001:
            raise HTTPException(status_code=403, detail={
                "message": response.error.message or "Insufficient permissions",
                "hint": "Permission denied for listing prompts. Contact an admin."
            })
        raise HTTPException(status_code=500, detail=response.error.message)

    return response.result


# Authentication endpoints

@router.post("/auth/token", response_model=AuthTokenResponse)
async def create_token(
    auth_request: AuthTokenRequest,
    request: Request,
    _guard: None = Depends(enforce_http_security),
):
    """
    Issue an MCP access token.

    Note: Direct username/password authentication for MCP is disabled by default.
    Use the primary AuthNZ login flow to obtain a JWT, or enable this endpoint
    explicitly via MCP_ENABLE_DEMO_AUTH=1 for development/testing only.
    """
    if not env_flag_enabled("MCP_ENABLE_DEMO_AUTH"):
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Direct MCP auth is disabled. Use AuthNZ bearer tokens.",
        )

    cfg = get_config()
    test_mode = is_test_mode()
    if not cfg.debug_mode and not test_mode:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Demo auth is restricted to debug/test environments.",
        )

    demo_secret = os.getenv("MCP_DEMO_AUTH_SECRET", "")
    if len(demo_secret) < 16:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Demo auth secret not configured; set MCP_DEMO_AUTH_SECRET to enable.",
        )

    client_host = getattr(request.client, "host", None)
    try:
        peer_ip = ipaddress.ip_address(client_host) if client_host else None
    except ValueError:
        peer_ip = None
    if not peer_ip or (not peer_ip.is_loopback and not peer_ip.is_private):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Demo auth is only available from loopback/private addresses.",
        )

    provided_secret = auth_request.api_key or auth_request.password or ""
    if not provided_secret or not secrets.compare_digest(provided_secret, demo_secret):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid demo auth secret.",
        )

    allowed_user = os.getenv("MCP_DEMO_AUTH_USER", "admin")
    username = auth_request.username or allowed_user
    if username != allowed_user:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid demo auth username.",
        )

    subject = os.getenv("MCP_DEMO_AUTH_SUBJECT", "admin_user")
    jwt_manager = get_jwt_manager()
    access_token = jwt_manager.create_access_token(
        subject=subject,
        username=username,
        roles=[UserRole.ADMIN.value],
        permissions=["*"],
    )
    refresh_token, _ = jwt_manager.create_refresh_token(subject)

    logger.warning(
        "Issued demo MCP token for development/testing use only.",
        extra={"audit": True, "client_ip": str(peer_ip)},
    )

    return AuthTokenResponse(
        access_token=access_token,
        expires_in=jwt_manager.config.jwt_access_token_expire_minutes * 60,
        refresh_token=refresh_token,
    )


@router.post("/auth/refresh", response_model=AuthTokenResponse)
async def refresh_token(
    auth_request: AuthRefreshRequest,
    _guard: None = Depends(enforce_http_security),
):
    """
    Refresh authentication token.

    Exchange a valid refresh token for a new access token using rotation.
    The refresh token must be provided in the request JSON body.
    If `token_id` is not provided, the system attempts to locate it by scanning
    active refresh tokens (acceptable for in-memory DEV mode).
    """
    jwt_manager = get_jwt_manager()
    refresh_token_value = auth_request.refresh_token
    token_id = auth_request.token_id
    # Attempt to find token_id if not provided
    resolved_token_id = token_id
    try:
        if not resolved_token_id:
            # Scan in-memory store (DEV): find first token_id matching token
            for tid, rt in jwt_manager._refresh_tokens.items():  # noqa: SLF001
                if rt.token == refresh_token_value and not rt.revoked:
                    resolved_token_id = tid
                    break
    except _MCP_UNIFIED_NONCRITICAL_EXCEPTIONS:
        resolved_token_id = token_id

    if not resolved_token_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token")

    try:
        new_access, new_refresh, _new_tid = jwt_manager.rotate_refresh_token(refresh_token_value, resolved_token_id)
    except HTTPException:
        raise
    except _MCP_UNIFIED_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Refresh token rotation failed: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to refresh token") from e

    return AuthTokenResponse(
        access_token=new_access,
        expires_in=jwt_manager.config.jwt_access_token_expire_minutes * 60,
        refresh_token=new_refresh,
    )


# Health check endpoint

@router.get("/health")
async def health_check(
    _guard: None = Depends(enforce_http_security),
):
    """
    Health check endpoint for load balancers.

    Returns 200 if server is healthy, 503 if not.
    """
    server = get_mcp_server()

    if not server.initialized:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server not initialized",
        )

    server_status = await server.get_status()

    status_value = "unhealthy"
    if isinstance(server_status, dict):
        raw_status = server_status.get("status", "unhealthy")
        status_value = raw_status if isinstance(raw_status, str) else "unhealthy"

    if status_value != "healthy":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Server status: {status_value}",
        )

    return {"status": "healthy"}


# ---------------------------------------------------------------------------
# Catalog endpoints
# ---------------------------------------------------------------------------


class MCPConnectionTestRequest(BaseModel):
    """Request body for testing connectivity to an external MCP server."""

    url: str
    auth_type: str = "none"
    secret: str | None = None
    auth_key_name: str | None = None


class MCPConnectionTestResponse(BaseModel):
    """Response for an MCP server connection test."""

    reachable: bool
    tools_discovered: list[str] = Field(default_factory=list)
    error: str | None = None


@router.get("/catalog")
async def list_mcp_catalog(
    archetype_key: str | None = None,
    _guard: None = Depends(enforce_http_security),
):
    """Return the curated external MCP server catalog."""
    from tldw_Server_API.app.core.MCP_unified.catalog_loader import list_catalog_entries

    return list_catalog_entries(archetype_key=archetype_key)


def _is_private_ip(host: str) -> bool:
    """Return True if *host* resolves to a non-public IP address."""
    import socket

    try:
        resolved = socket.getaddrinfo(host, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
        for _family, _type, _proto, _canonname, sockaddr in resolved:
            ip = ipaddress.ip_address(sockaddr[0])
            if (
                ip.is_private
                or ip.is_loopback
                or ip.is_link_local
                or ip.is_reserved
                or ip.is_multicast
                or ip.is_unspecified
                or not ip.is_global
            ):
                return True
    except (socket.gaierror, ValueError):
        return True  # fail closed: unresolvable hosts are rejected
    return False


def _canonicalize_catalog_probe_url(url: str) -> str | None:
    from urllib.parse import urlsplit, urlunsplit

    try:
        parsed = urlsplit(url)
    except ValueError:
        return None

    scheme = (parsed.scheme or "").lower()
    hostname = (parsed.hostname or "").lower()
    if scheme not in ("http", "https") or not hostname or parsed.username or parsed.password:
        return None

    try:
        port = parsed.port
    except ValueError:
        return None

    default_port = 443 if scheme == "https" else 80
    netloc = hostname if port in (None, default_port) else f"{hostname}:{port}"
    normalized_path = (parsed.path or "").rstrip("/")
    return urlunsplit((scheme, netloc, normalized_path, "", ""))


def _resolve_catalog_probe_url(url: str) -> str | None:
    from tldw_Server_API.app.core.MCP_unified.catalog_loader import list_catalog_entries

    candidate = _canonicalize_catalog_probe_url(url)
    if candidate is None:
        return None

    for entry in list_catalog_entries():
        allowed = _canonicalize_catalog_probe_url(entry.url_template)
        if allowed and allowed == candidate:
            return allowed
    return None


async def _probe_mcp_connection(url: str, headers: dict[str, str]) -> None:
    from tldw_Server_API.app.core.http_client import _get_httpx_async_client

    http = _get_httpx_async_client()
    resp = await http.get(url, headers=headers, timeout=10.0)
    resp.raise_for_status()


@router.post("/catalog/test-connection", response_model=MCPConnectionTestResponse)
async def check_mcp_connection(
    req: MCPConnectionTestRequest,
    _guard: None = Depends(enforce_http_security),
):
    """Test connectivity to an external MCP server URL.

    Currently validates reachability only. The ``tools_discovered`` field
    is reserved for a future enhancement that will parse the MCP server's
    tool listing response.
    """
    from urllib.parse import urlparse

    catalog_probe_url = _resolve_catalog_probe_url(req.url)
    if catalog_probe_url is None:
        return MCPConnectionTestResponse(
            reachable=False,
            error="URL must match a curated MCP catalog entry",
        )

    parsed = urlparse(catalog_probe_url)
    if parsed.scheme not in ("http", "https"):
        return MCPConnectionTestResponse(
            reachable=False, error="Only http and https URLs are allowed"
        )

    supported_auth_types = {"none", "bearer", "api_key"}
    if req.auth_type not in supported_auth_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported auth_type: {req.auth_type}",
        )

    # SSRF protection: reject private/reserved IP ranges.
    hostname = parsed.hostname or ""
    if not hostname or _is_private_ip(hostname):
        return MCPConnectionTestResponse(
            reachable=False, error="Cannot connect to private or reserved addresses"
        )
    assert_url_safe(catalog_probe_url)

    headers: dict[str, str] = {}
    if req.auth_type == "bearer" and req.secret:
        headers["Authorization"] = f"Bearer {req.secret}"
    elif req.auth_type == "api_key" and req.secret:
        headers[req.auth_key_name or "X-API-Key"] = req.secret
    try:
        await _probe_mcp_connection(catalog_probe_url, headers)
        return MCPConnectionTestResponse(reachable=True)
    except HTTPException:
        raise
    except Exception:
        logger.opt(exception=True).warning(
            "MCP connection test failed for {}", req.url
        )
        return MCPConnectionTestResponse(reachable=False, error="Connection failed")


# OpenAPI documentation customization

def customize_openapi():
    """Customize OpenAPI schema for better documentation"""
    from fastapi.openapi.utils import get_openapi

    def custom_openapi():
        if router.openapi_schema:
            return router.openapi_schema

        openapi_schema = get_openapi(
            title="MCP Unified API",
            version="3.0.0",
            description="""
            # Model Context Protocol (MCP) Unified API

            Production-ready MCP implementation with enhanced security and monitoring.

            ## Features
            - 🔒 **Secure by Default**: JWT authentication, RBAC, rate limiting
            - 🚀 **High Performance**: Connection pooling, caching, circuit breakers
            - 📊 **Observable**: Health checks, metrics, distributed tracing
            - 🔧 **Modular**: Extensible module system with hot-reload

            ## Authentication
            Most endpoints support optional authentication via Bearer token.
            Some endpoints require authentication or admin role.

            ## WebSocket
            The `/ws` endpoint provides full MCP protocol support over WebSocket.
            """,
            routes=router.routes,
        )

        # Add security schemes
        openapi_schema["components"]["securitySchemes"] = {
            "bearerAuth": {
                "type": "http",
                "scheme": "bearer",
                "bearerFormat": "JWT"
            }
        }

        router.openapi_schema = openapi_schema
        return router.openapi_schema

    return custom_openapi


router.openapi = customize_openapi()
