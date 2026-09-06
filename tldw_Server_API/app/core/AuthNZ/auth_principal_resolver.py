"""
auth_principal_resolver.py

Resolver for the unified AuthPrincipal / AuthContext models.

This module provides a single entry point for deriving the authenticated
principal for a request. It intentionally reuses existing AuthNZ helpers
so that behavior stays aligned with current authentication flows.
"""

import os
from typing import Any, Optional

from fastapi import HTTPException, Request, status
from loguru import logger

from tldw_Server_API.app.core.AuthNZ import User_DB_Handling
from tldw_Server_API.app.core.AuthNZ.exceptions import (
    InvalidTokenError,
    SessionRevokedException,
    TokenExpiredError,
)
from tldw_Server_API.app.core.AuthNZ.ip_allowlist import (
    is_service_token_ip_allowed,
    is_single_user_ip_allowed,
    resolve_client_ip,
)
from tldw_Server_API.app.core.AuthNZ.jwt_service import get_jwt_service
from tldw_Server_API.app.core.AuthNZ.principal_model import (
    AuthContext,
    AuthPrincipal,
    PrincipalKind,
)
from tldw_Server_API.app.core.AuthNZ.session_manager import get_session_manager
from tldw_Server_API.app.core.AuthNZ.settings import get_settings
from tldw_Server_API.app.core.AuthNZ.single_user_session import validate_single_user_session
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
from tldw_Server_API.app.core.exceptions import InactiveUserError
from tldw_Server_API.app.core.testing import env_flag_enabled

_RESOLVER_MODE_EXCEPTIONS = (
    AttributeError,
    ImportError,
    RuntimeError,
    TypeError,
    ValueError,
)
_TOKEN_PEEK_EXCEPTIONS = (
    AttributeError,
    ImportError,
    KeyError,
    RuntimeError,
    TypeError,
    ValueError,
)
_USER_VALUE_EXCEPTIONS = (AttributeError, TypeError, ValueError)
_STATE_ATTACH_EXCEPTIONS = (AttributeError, RuntimeError, TypeError, ValueError)
_SINGLE_USER_COMPAT_EXCEPTIONS = (
    AttributeError,
    ConnectionError,
    HTTPException,
    ImportError,
    KeyError,
    LookupError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)
_PLATFORM_ADMIN_ROLES = frozenset({"admin", "owner", "super_admin"})
_ADMIN_CLAIM_PERMISSIONS = frozenset({"*", "system.configure", "admin"})


def _extract_bearer_token(request: Request) -> Optional[str]:
    """Extract Bearer token from Authorization header."""
    try:
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            return None
        scheme, _, credential = auth_header.partition(" ")
        if scheme.lower() != "bearer" or not credential:
            return None
        return credential.strip()
    except (AttributeError, TypeError):
        return None


def _extract_api_key(request: Request) -> Optional[str]:
    """Extract API key from X-API-KEY header."""
    try:
        api_key = request.headers.get("X-API-KEY")
        if isinstance(api_key, str) and api_key.strip():
            return api_key.strip()
    except (AttributeError, TypeError):
        return None


def _looks_like_jwt(token: Optional[str]) -> bool:
    if not isinstance(token, str):
        return False
    return token.count(".") == 2


def _peek_jwt_token_type(token: str) -> Optional[str]:
    """Return the unverified token type claim when available."""
    try:
        from tldw_Server_API.app.core.Utils import jwt_compat as _jwt

        claims = _jwt.get_unverified_claims(token)
        if isinstance(claims, dict):
            raw_type = claims.get("type")
            if raw_type:
                return str(raw_type)
    except _TOKEN_PEEK_EXCEPTIONS:
        return None
    except Exception:  # noqa: BLE001 - malformed JWT segments should not block API-key fallback
        return None
    return None


def _normalized_claim_values(
    values: list[Any] | tuple[Any, ...] | set[Any] | None,
) -> set[str]:
    return {
        str(value).strip().lower()
        for value in (values or [])
        if str(value).strip()
    }


def _claims_mark_admin(
    *,
    roles: list[str] | tuple[str, ...] | set[str] | None,
    permissions: list[str] | tuple[str, ...] | set[str] | None,
) -> bool:
    normalized_roles = _normalized_claim_values(roles)
    if normalized_roles & _PLATFORM_ADMIN_ROLES:
        return True
    normalized_permissions = _normalized_claim_values(permissions)
    return bool(normalized_permissions & _ADMIN_CLAIM_PERMISSIONS)


def _resolve_service_identity(payload: dict) -> tuple[str, list[str], list[str]]:
    """Derive service name and claims from a verified service token."""
    service_name = payload.get("service")
    if not service_name:
        subject = payload.get("sub")
        if isinstance(subject, str) and subject.startswith("service:"):
            service_name = subject.split("service:", 1)[1]
        else:
            service_name = subject or "service"
    permissions_raw = payload.get("permissions") or []
    if isinstance(permissions_raw, str):
        permissions = [permissions_raw]
    elif isinstance(permissions_raw, (list, tuple, set)):
        permissions = [str(p) for p in permissions_raw if p]
    else:
        permissions = []

    roles_raw = payload.get("roles") or []
    if isinstance(roles_raw, str):
        roles = [roles_raw]
    elif isinstance(roles_raw, (list, tuple, set)):
        roles = [str(r) for r in roles_raw if r]
    else:
        roles = []
    return str(service_name), permissions, roles


def _is_test_context() -> bool:
    """Return True when running in pytest or explicit test-mode contexts."""
    if os.getenv("PYTEST_CURRENT_TEST") is not None:
        return True
    for flag in ("TEST_MODE", "TLDW_TEST_MODE", "TESTING"):
        if env_flag_enabled(flag):
            return True
    return False


def _build_principal_from_user(
    user: User,
    *,
    kind: PrincipalKind,
    request: Request,
    token_type: Optional[str] = None,
    jti: Optional[str] = None,
    subject: Optional[str] = None,
    api_key_id: Optional[int] = None,
) -> AuthPrincipal:
    """
    Construct an AuthPrincipal from an authenticated User and request context.

    This helper derives organization and team membership from request.state
    for backwards compatibility with existing middleware and DB scoping logic.
    """
    # Best-effort numeric user id
    try:
        user_id_int = user.id_int
    except (AttributeError, TypeError, ValueError):
        logger.debug("Could not extract numeric user_id from user object")
        user_id_int = None

    # Membership/context from request.state (if present)
    org_ids: list[int] = []
    team_ids: list[int] = []
    active_org_id: Optional[int] = None
    active_team_id: Optional[int] = None
    try:
        raw_org_ids = getattr(request.state, "org_ids", None)
        if isinstance(raw_org_ids, (list, tuple)):
            org_ids = [int(o) for o in raw_org_ids if o is not None]
    except (AttributeError, TypeError, ValueError):
        org_ids = []
    try:
        raw_team_ids = getattr(request.state, "team_ids", None)
        if isinstance(raw_team_ids, (list, tuple)):
            team_ids = [int(t) for t in raw_team_ids if t is not None]
    except (AttributeError, TypeError, ValueError):
        team_ids = []
    try:
        raw_active_org = getattr(request.state, "active_org_id", None)
        if raw_active_org is not None:
            active_org_id = int(raw_active_org)
    except (AttributeError, TypeError, ValueError):
        active_org_id = None
    try:
        raw_active_team = getattr(request.state, "active_team_id", None)
        if raw_active_team is not None:
            active_team_id = int(raw_active_team)
    except (AttributeError, TypeError, ValueError):
        active_team_id = None

    # Claims on the User model are the canonical source of truth.
    raw_roles = list(getattr(user, "roles", []) or [])
    roles: list[str] = []
    _seen_roles: set[str] = set()
    for role in raw_roles:
        role_text = str(role).strip()
        if not role_text:
            continue
        role_key = role_text.lower()
        if role_key in _seen_roles:
            continue
        _seen_roles.add(role_key)
        roles.append(role_text)
    permissions = [
        str(permission)
        for permission in (getattr(user, "permissions", []) or [])
        if str(permission).strip()
    ]
    is_admin = _claims_mark_admin(roles=roles, permissions=permissions)
    username = None
    email = None
    try:
        raw_username = getattr(user, "username", None)
        if raw_username:
            username = str(raw_username)
    except _USER_VALUE_EXCEPTIONS:
        username = None
    try:
        raw_email = getattr(user, "email", None)
        if raw_email:
            email = str(raw_email)
    except _USER_VALUE_EXCEPTIONS:
        email = None

    principal = AuthPrincipal(
        kind=kind,
        user_id=user_id_int,
        api_key_id=api_key_id,
        username=username,
        email=email,
        subject=subject,
        token_type=token_type,
        jti=jti,
        roles=roles,
        permissions=permissions,
        is_admin=is_admin,
        org_ids=org_ids,
        team_ids=team_ids,
        active_org_id=active_org_id,
        active_team_id=active_team_id,
    )
    return principal


def _build_context(
    principal: AuthPrincipal,
    request: Request,
) -> AuthContext:
    """Construct AuthContext from principal and request metadata."""
    try:
        ip = request.client.host if request.client else None  # type: ignore[union-attr]
    except (AttributeError, TypeError):
        ip = None
    try:
        user_agent = request.headers.get("User-Agent")
    except (AttributeError, TypeError):
        user_agent = None
    try:
        request_id = (
            request.headers.get("X-Request-ID")
            or getattr(request.state, "request_id", None)
        )
        # getattr(request, "state", ...) can raise if state is not present or misconfigured
    except (AttributeError, TypeError):
        request_id = None

    return AuthContext(
        principal=principal,
        ip=ip,
        user_agent=user_agent,
        request_id=request_id,
    )


async def get_auth_principal(request: Request) -> AuthPrincipal:
    """
    Resolve the AuthPrincipal for the current request.

    Behavior:
    - If request.state.auth is already populated with an AuthContext, reuse it.
    - Prefer Bearer JWT tokens; fall back to API keys via X-API-KEY or non-JWT Bearer tokens.
    - On missing or invalid credentials, raise HTTP 401 with stable semantics.
    """
    # Fast-path: reuse existing AuthContext if present
    existing = getattr(request.state, "auth", None)
    if isinstance(existing, AuthContext):
        return existing.principal
    # Prefer Bearer JWT, fall back to X-API-KEY
    has_authorization_header = request.headers.get("Authorization") is not None
    has_api_key_header = request.headers.get("X-API-KEY") is not None
    token = _extract_bearer_token(request)
    api_key = _extract_api_key(request)

    raw_token = token
    token_is_jwt = _looks_like_jwt(token)
    test_context = _is_test_context()

    # In single-user mode, always treat Bearer tokens as API keys. In multi-user
    # mode, keep non-JWT tokens as API keys unless we are in a test context.
    if token and not api_key:
        try:
            settings = get_settings()
            auth_mode = getattr(settings, "AUTH_MODE", None)
        except _RESOLVER_MODE_EXCEPTIONS:
            auth_mode = None
        if auth_mode == "single_user":
            # In single-user mode, treat Bearer tokens as API keys (no legacy bearer tokens).
            api_key = token
            token = None
        elif not token_is_jwt and not test_context:
            api_key = token
            token = None

    if not token and not api_key:
        if not has_authorization_header and not has_api_key_header:
            try:
                identity = await validate_single_user_session(request)
            except SessionRevokedException:
                identity = None
            if identity is not None:
                user = User_DB_Handling.get_single_user_instance()
                principal = _build_principal_from_user(
                    user=user,
                    kind="user",
                    request=request,
                    token_type="single_user_session",  # nosec B106
                    subject="single_user",
                )
                request.state.single_user_session_id = identity.session_id
                request.state.user_id = identity.user_id
                request.state.auth = _build_context(principal, request)
                request.state._auth_user = user
                return principal
        # Align with existing 401 semantics when no credentials are provided
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated (provide Bearer token or X-API-KEY)",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # JWT path
    if token:
        # Service token path (planned usage): verify and construct a service principal.
        if token_is_jwt:
            token_type = _peek_jwt_token_type(token)
            if token_type == "service":
                try:
                    jwt_service = get_jwt_service()
                    payload = jwt_service.verify_service_token(token)
                except (InvalidTokenError, TokenExpiredError) as exc:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Could not validate credentials",
                        headers={"WWW-Authenticate": "Bearer"},
                    ) from exc
                except Exception as exc:
                    logger.exception("Error verifying service token: {}", exc)
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Could not validate credentials",
                        headers={"WWW-Authenticate": "Bearer"},
                    ) from exc

                # Enforce token revocation/blacklist for service tokens.
                try:
                    session_manager = await get_session_manager()
                    if await session_manager.is_token_blacklisted(token, payload.get("jti")):
                        raise HTTPException(
                            status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Could not validate credentials",
                            headers={"WWW-Authenticate": "Bearer"},
                        )
                except HTTPException:
                    raise
                except Exception as exc:
                    logger.debug("Service token blacklist check failed: {}", exc)
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Could not validate credentials",
                        headers={"WWW-Authenticate": "Bearer"},
                    ) from exc

                try:
                    settings = get_settings()
                except _RESOLVER_MODE_EXCEPTIONS:
                    settings = None
                client_ip = resolve_client_ip(request, settings)
                if not is_service_token_ip_allowed(client_ip, settings):
                    logger.warning(
                        "Service token rejected due to non-local client_ip={}",
                        client_ip or "<unknown>",
                    )
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Service tokens are restricted to local/internal requests",
                    )

                service_name, permissions, roles = _resolve_service_identity(payload)
                principal = AuthPrincipal(
                    kind="service",
                    user_id=None,
                    api_key_id=None,
                    username=None,
                    email=None,
                    subject=f"service:{service_name}",
                    token_type="service",
                    jti=payload.get("jti"),
                    roles=roles,
                    permissions=permissions,
                    is_admin=_claims_mark_admin(roles=roles, permissions=permissions),
                    org_ids=[],
                    team_ids=[],
                    active_org_id=None,
                    active_team_id=None,
                )
                ctx = _build_context(principal, request)
                try:
                    request.state.auth = ctx
                    request.state.api_key_id = None
                    request.state.user_id = None
                except _STATE_ATTACH_EXCEPTIONS as state_exc:
                    logger.debug(
                        "auth_principal_resolver: unable to attach service principal context: {}",
                        state_exc,
                    )
                return principal
        try:
            user = await User_DB_Handling.verify_jwt_and_fetch_user(request, token)
        except InactiveUserError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Inactive user",
            ) from exc
        except HTTPException as exc:
            if exc.status_code == status.HTTP_401_UNAUTHORIZED and api_key:
                token = None
            elif raw_token and not token_is_jwt and test_context:
                api_key = raw_token
                token = None
            else:
                raise
        except Exception as exc:
            if raw_token and not token_is_jwt and test_context:
                api_key = raw_token
                token = None
            else:
                logger.exception("Error resolving principal from JWT: {}", exc)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Authentication failed due to internal error",
                ) from exc

        if token is not None:
            # verify_jwt_and_fetch_user populates request.state.auth / _auth_user; reuse if present.
            try:
                ctx = getattr(request.state, "auth", None)
                if isinstance(ctx, AuthContext):
                    return ctx.principal
            except (AttributeError, TypeError) as exc:
                # Fall back to rebuilding principal if state is missing/misconfigured.
                logger.debug("Could not access request.state.auth after JWT validation: {}", exc)

            principal = _build_principal_from_user(
                user=user,
                kind="user",
                request=request,
                token_type="access",
                jti=None,
                api_key_id=None,
            )
            ctx = _build_context(principal, request)
            try:
                request.state.auth = ctx
                # Cache the resolved user for downstream dependencies
                request.state._auth_user = user
            except Exception as exc:  # noqa: BLE001 - defensive: caching failures must not break auth
                logger.exception("Unable to cache auth context/user: {}", exc)
            return principal

    # API key path
    if api_key:
        # Single-user compatibility: treat SINGLE_USER_API_KEY/SINGLE_USER_TEST_API_KEY
        # as a user-kind principal without requiring API key store lookups.
        try:
            from tldw_Server_API.app.core.AuthNZ.settings import get_settings as _get_settings  # noqa: WPS433

            settings = _get_settings()
            if getattr(settings, "AUTH_MODE", None) == "single_user":
                allowed_keys: set[str] = set()
                primary_key = getattr(settings, "SINGLE_USER_API_KEY", None)
                if primary_key:
                    allowed_keys.add(primary_key)
                if test_context:
                    test_key = os.getenv("SINGLE_USER_TEST_API_KEY")
                    if test_key:
                        allowed_keys.add(test_key)
                if api_key in allowed_keys:
                    client_ip = resolve_client_ip(request, settings)
                    if not is_single_user_ip_allowed(client_ip, settings):
                        raise HTTPException(
                            status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Not authenticated (provide Bearer token or X-API-KEY)",
                            headers={"WWW-Authenticate": "Bearer"},
                        )
                    user = User_DB_Handling.get_single_user_instance()
                    principal = _build_principal_from_user(
                        user=user,
                        kind="user",
                        request=request,
                        token_type="api_key",
                        jti=None,
                        subject="single_user",
                        api_key_id=None,
                    )
                    ctx = _build_context(principal, request)
                    try:
                        request.state.auth = ctx
                        request.state._auth_user = user
                        request.state.user_id = user.id
                        request.state.api_key_id = None
                        request.state.team_ids = []
                        request.state.org_ids = []
                    except _STATE_ATTACH_EXCEPTIONS as state_exc:
                        logger.debug(
                            "auth_principal_resolver: unable to attach single-user state context: {}",
                            state_exc,
                        )
                    return principal
        except _SINGLE_USER_COMPAT_EXCEPTIONS as single_exc:
            logger.debug(
                "auth_principal_resolver: single-user API key compat path failed; falling back: {}",
                single_exc,
            )

        try:
            user = await User_DB_Handling.authenticate_api_key_user(request, api_key)
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Error resolving principal from API key: {}", exc)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Authentication failed due to internal error",
            ) from exc

        # authenticate_api_key_user populates request.state.auth / _auth_user; reuse if present.
        try:
            ctx = getattr(request.state, "auth", None)
            if isinstance(ctx, AuthContext):
                return ctx.principal
        except (AttributeError, TypeError) as exc:
            logger.debug("Could not access request.state.auth after API key validation: {}", exc)

        api_key_id: Optional[int] = None
        try:
            raw_api_key_id = getattr(request.state, "api_key_id", None)
            if raw_api_key_id is not None:
                api_key_id = int(raw_api_key_id)
        except (AttributeError, TypeError, ValueError):
            api_key_id = None

        principal = _build_principal_from_user(
            user=user,
            kind="api_key",
            request=request,
            token_type="api_key",
            jti=None,
            api_key_id=api_key_id,
        )
        ctx = _build_context(principal, request)
        try:
            request.state.auth = ctx
            # Cache the resolved user for downstream dependencies
            request.state._auth_user = user
        except Exception as exc:  # noqa: BLE001 - defensive: caching failures must not break auth
            logger.exception("Unable to cache auth context/user: {}", exc)
        return principal

    # Fallback (should not be reached)
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
