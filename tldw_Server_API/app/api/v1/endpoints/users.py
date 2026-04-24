# users.py
# Description: User management endpoints for profile, password, and session management
#
# Imports
import contextlib
import os
from datetime import datetime
from typing import Any, Optional

#
# 3rd-party imports
from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, Response, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel, Field

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
    get_auth_principal,
    get_db_transaction,
    get_password_service_dep,
    get_session_manager_dep,
    get_storage_service_dep,
    require_api_key_scope,
)
from tldw_Server_API.app.api.v1.schemas.api_key_schemas import (
    APIKeyCreateRequest,
    APIKeyCreateResponse,
    APIKeyMetadata,
    APIKeyRotateRequest,
)

#
# Local imports
from tldw_Server_API.app.api.v1.schemas.auth_schemas import (
    DeprecatedUserResponse,
    MessageResponse,
    PasswordChangeRequest,
    SessionResponse,
    StorageQuotaResponse,
    UpdateProfileRequest,
)
from tldw_Server_API.app.api.v1.schemas.user_profile_schemas import (
    UserProfileCatalogResponse,
    UserProfileErrorDetail,
    UserProfileErrorResponse,
    UserProfileResponse,
    UserProfileUpdateError,
    UserProfileUpdateRequest,
    UserProfileUpdateResponse,
)
from tldw_Server_API.app.api.v1.utils.cache import generate_etag, is_not_modified
from tldw_Server_API.app.api.v1.utils.deprecation import build_deprecation_headers
from tldw_Server_API.app.api.v1.utils.profile_errors import (
    classify_profile_update_skips,
)
from tldw_Server_API.app.core.AuthNZ.api_key_manager import get_api_key_manager
from tldw_Server_API.app.core.AuthNZ.database import (
    get_db_pool,
    is_postgres_backend,
)
from tldw_Server_API.app.core.AuthNZ.exceptions import (
    DatabaseError,
    StorageError,
    WeakPasswordError,
)
from tldw_Server_API.app.core.AuthNZ.password_service import PasswordService
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal, is_single_user_principal
from tldw_Server_API.app.core.AuthNZ.repos.users_repo import AuthnzUsersRepo
from tldw_Server_API.app.core.AuthNZ.session_manager import SessionManager
from tldw_Server_API.app.core.Audit.unified_audit_service import MandatoryAuditWriteError
from tldw_Server_API.app.core.testing import is_truthy
from tldw_Server_API.app.core.UserProfiles.service import UserProfileService
from tldw_Server_API.app.core.UserProfiles.update_service import UserProfileUpdateService
from tldw_Server_API.app.core.UserProfiles.user_profile_catalog import load_user_profile_catalog
from tldw_Server_API.app.services.storage_quota_service import StorageQuotaService

_USERS_AUDIT_EXCEPTIONS = (
    AttributeError,
    ConnectionError,
    DatabaseError,
    ImportError,
    KeyError,
    LookupError,
    OSError,
    RuntimeError,
    StorageError,
    TypeError,
    ValueError,
)
_USERS_ENDPOINT_EXCEPTIONS = (
    AttributeError,
    ConnectionError,
    DatabaseError,
    KeyError,
    LookupError,
    OSError,
    RuntimeError,
    StorageError,
    TypeError,
    ValueError,
)
_USERS_HEADER_EXCEPTIONS = (AttributeError, TypeError, ValueError)


def _legacy_user_me_enabled() -> bool:
    raw = os.getenv("ENABLE_LEGACY_USER_ME_ENDPOINTS", "true")
    return is_truthy(str(raw).strip().lower())


def _legacy_warning_payload(successor: str) -> dict[str, str]:
    return {"warning": "deprecated_endpoint", "successor": successor}


def _profile_error_response(
    *,
    status_code: int,
    error_code: str,
    detail: str,
    errors: Optional[list[UserProfileErrorDetail]] = None,
) -> JSONResponse:
    payload = UserProfileErrorResponse(
        error_code=error_code,
        detail=detail,
        errors=errors or [],
    )
    return JSONResponse(status_code=status_code, content=payload.model_dump())


def _principal_user_id(principal: AuthPrincipal) -> int:
    raw_id = principal.user_id
    try:
        user_id = int(raw_id)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid user context") from exc
    if user_id <= 0:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid user context")
    return user_id


def _principal_primary_role(principal: AuthPrincipal) -> str:
    roles = {
        str(role).strip().lower()
        for role in (principal.roles or [])
        if str(role).strip()
    }
    permissions = {
        str(permission).strip().lower()
        for permission in (principal.permissions or [])
        if str(permission).strip()
    }
    if "admin" in roles or "*" in permissions or "system.configure" in permissions:
        return "admin"
    for role in principal.roles or []:
        role_text = str(role).strip().lower()
        if role_text:
            return role_text
    for role_text in roles:
        return role_text
    return "user"


def _principal_username(principal: AuthPrincipal, user_id: int) -> str:
    username = (principal.username or "").strip()
    if username:
        return username
    email = (principal.email or "").strip()
    if email:
        return email.split("@", 1)[0]
    return f"user-{user_id}"


def _principal_email(principal: AuthPrincipal) -> str:
    return (principal.email or "").strip()


def _coerce_bool_flag(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off", ""}:
            return False
    return bool(value)


def _extract_affected_rows(execute_result: Any) -> int | None:
    rowcount = getattr(execute_result, "rowcount", None)
    if isinstance(rowcount, int):
        return rowcount
    if isinstance(execute_result, str):
        parts = execute_result.strip().split()
        if parts and parts[-1].isdigit():
            try:
                return int(parts[-1])
            except ValueError:
                return None
    return None


async def _fetch_password_hash_for_user(db: Any, user_id: int) -> str | None:
    """
    Compatibility helper retained for backend-selection unit tests.
    """
    if await is_postgres_backend():
        value = await db.fetchval(
            "SELECT password_hash FROM users WHERE id = $1",
            user_id,
        )
        return str(value) if value else None

    cursor = await db.execute(
        "SELECT password_hash FROM users WHERE id = ?",
        (user_id,),
    )
    row = await cursor.fetchone()
    if not row:
        return None
    if isinstance(row, dict):
        value = row.get("password_hash")
        return str(value) if value else None
    try:
        value = row[0]
    except (IndexError, KeyError, TypeError):
        value = None
    return str(value) if value else None


def _require_active_verified_user(user_context: dict[str, Any]) -> None:
    if not _coerce_bool_flag(user_context.get("is_active"), default=False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive",
        )
    if not _coerce_bool_flag(user_context.get("is_verified"), default=False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Email verification required",
        )


async def _resolve_user_context(
    principal: AuthPrincipal,
    *,
    allow_missing: bool = False,
) -> dict[str, Any]:
    user_id = _principal_user_id(principal)
    fallback: dict[str, Any] = {
        "id": user_id,
        "username": _principal_username(principal, user_id),
        "email": _principal_email(principal),
        "role": _principal_primary_role(principal),
        "is_active": True,
        "is_verified": True,
        "storage_quota_mb": 5120,
        "storage_used_mb": 0.0,
        "created_at": datetime.utcnow(),
        "last_login": None,
    }
    repo = await AuthnzUsersRepo.from_pool()
    user = await repo.get_user_by_id(user_id)
    if not user:
        if allow_missing:
            return fallback
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    user_dict = dict(user)
    user_dict.pop("password_hash", None)
    return {**fallback, **user_dict}


async def _require_principal_active_verified(
    principal: AuthPrincipal,
    *,
    allow_missing_single_user: bool = True,
) -> dict[str, Any]:
    """
    Resolve user context from principal and enforce active/verified account state.

    Single-user mode may operate with a synthetic principal identity that does not
    have a persisted users table row; allow fallback context only for that case.
    """
    allow_missing = bool(allow_missing_single_user and is_single_user_principal(principal))
    user_context = await _resolve_user_context(principal, allow_missing=allow_missing)
    _require_active_verified_user(user_context)
    return user_context


async def _emit_user_profile_audit_event(
    request: Request,
    *,
    user_id: int,
    update_keys: list[str],
    applied_count: int,
    skipped_count: int,
    dry_run: bool,
) -> None:
    try:
        from tldw_Server_API.app.api.v1.API_Deps.Audit_DB_Deps import (
            get_or_create_audit_service_for_user_id,
        )
        from tldw_Server_API.app.core.Audit.unified_audit_service import (
            AuditContext,
            AuditEventCategory,
            AuditEventType,
        )

        audit_service = await get_or_create_audit_service_for_user_id(user_id)
        correlation_id = (
            request.headers.get("X-Correlation-ID")
            or getattr(request.state, "correlation_id", None)
        )
        request_id = (
            request.headers.get("X-Request-ID")
            or getattr(request.state, "request_id", None)
            or ""
        )
        ctx = AuditContext(
            user_id=str(user_id),
            correlation_id=correlation_id,
            request_id=request_id,
            ip_address=(request.client.host if request.client else None),
            user_agent=request.headers.get("user-agent"),
            endpoint=str(request.url.path),
            method=request.method,
        )
        await audit_service.log_event(
            event_type=AuditEventType.DATA_READ if dry_run else AuditEventType.DATA_UPDATE,
            category=AuditEventCategory.DATA_ACCESS if dry_run else AuditEventCategory.DATA_MODIFICATION,
            context=ctx,
            resource_type="user_profile",
            resource_id=str(user_id),
            action="user_profile.update_preview" if dry_run else "user_profile.update",
            metadata={
                "dry_run": dry_run,
                "update_keys": update_keys,
                "applied_count": applied_count,
                "skipped_count": skipped_count,
            },
        )
    except _USERS_AUDIT_EXCEPTIONS as exc:
        logger.debug("User profile audit emission skipped: {}", exc)

#######################################################################################################################
#
# Router Configuration

router = APIRouter(
    prefix="/users",
    tags=["users"],
    responses={404: {"description": "Not found"}}
)


#######################################################################################################################
#
# User Profile Endpoints

@router.get("/profile/catalog", response_model=UserProfileCatalogResponse)
async def get_user_profile_catalog(
    principal: AuthPrincipal = Depends(get_auth_principal),
    if_none_match: Optional[str] = Header(None),
) -> Response:
    """
    Return the user profile config catalog with caching headers.

    Requires authentication; catalog data is shared across users.
    """
    await _require_principal_active_verified(principal)
    catalog = load_user_profile_catalog()
    payload = jsonable_encoder(catalog)
    etag = generate_etag(payload)
    cache_headers = {"ETag": etag, "Cache-Control": "max-age=3600"}

    if is_not_modified(etag, if_none_match):
        return Response(status_code=status.HTTP_304_NOT_MODIFIED, headers=cache_headers)

    response = JSONResponse(content=payload)
    response.headers.update(cache_headers)
    return response


@router.get("/me/profile", response_model=UserProfileResponse, response_model_exclude_none=True)
async def get_current_user_profile_view(
    sections: Optional[str] = Query(
        None, description="Comma-separated list of sections to include"
    ),
    include_sources: bool = Query(
        False, description="Include per-field source attribution"
    ),
    include_raw: bool = Query(
        False, description="Admin-only; include raw stored overrides"
    ),
    mask_secrets: bool = Query(
        True, description="Mask secret values in the response"
    ),
    principal: AuthPrincipal = Depends(get_auth_principal),
    session_manager: SessionManager = Depends(get_session_manager_dep),
) -> UserProfileResponse:
    """
    Get the authenticated user's unified profile.
    """
    if include_raw or not mask_secrets:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="admin_only_query_parameters",
        )
    db_pool = await get_db_pool()
    service = UserProfileService(db_pool)
    user_context = await _resolve_user_context(principal)
    _require_active_verified_user(user_context)
    user_dict: dict[str, Any] = dict(user_context)
    requested = service.parse_sections(sections)
    api_mgr = await get_api_key_manager()
    security = await service.build_security(
        user_id=int(user_context["id"]),
        session_manager=session_manager,
        api_key_manager=api_mgr,
    )
    profile = await service.build_profile(
        user=user_dict,
        sections=requested,
        security=security,
        include_sources=include_sources,
        include_raw=include_raw,
        mask_secrets=mask_secrets,
        metrics_scope="self",
    )
    return UserProfileResponse(**profile)


@router.patch("/me/profile", response_model=UserProfileUpdateResponse)
async def update_current_user_profile(
    payload: UserProfileUpdateRequest,
    http_request: Request,
    principal: AuthPrincipal = Depends(get_auth_principal),
    db=Depends(get_db_transaction),
) -> UserProfileUpdateResponse:
    """
    Update the authenticated user's profile preferences.
    """
    if not payload.updates:
        return _profile_error_response(
            status_code=status.HTTP_400_BAD_REQUEST,
            error_code="profile_update_invalid",
            detail="No updates provided",
            errors=[UserProfileErrorDetail(key="updates", message="missing")],
        )

    user_context = await _resolve_user_context(principal)
    _require_active_verified_user(user_context)
    user_id = int(user_context["id"])
    db_pool = await get_db_pool()
    profile_service = UserProfileService(db_pool)
    current_version = await profile_service.get_profile_version(user_id=user_id)
    if payload.profile_version is not None:
        if not profile_service.versions_match(current_version, payload.profile_version):
            return _profile_error_response(
                status_code=status.HTTP_409_CONFLICT,
                error_code="profile_version_mismatch",
                detail="profile_version_mismatch",
                errors=[UserProfileErrorDetail(key="profile_version", message="mismatch")],
            )

    service = UserProfileUpdateService(db_pool)
    updates = [(entry.key, entry.value) for entry in payload.updates]
    preflight = await service.apply_updates(
        user_id=user_id,
        updates=updates,
        roles={"user"},
        dry_run=True,
        db_conn=db,
        updated_by=user_id,
    )

    error_payload = classify_profile_update_skips(preflight.skipped)
    if error_payload:
        status_code, error_code, detail, errors = error_payload
        return _profile_error_response(
            status_code=status_code,
            error_code=error_code,
            detail=detail,
            errors=errors,
        )

    if payload.dry_run:
        return UserProfileUpdateResponse(
            profile_version=current_version,
            applied=preflight.applied,
            skipped=[],
        )

    result = await service.apply_updates(
        user_id=user_id,
        updates=updates,
        roles={"user"},
        dry_run=False,
        db_conn=db,
        updated_by=user_id,
    )

    current_version = await profile_service.get_profile_version(user_id=user_id)
    skipped = [UserProfileUpdateError(**item) for item in result.skipped]
    response = UserProfileUpdateResponse(
        profile_version=current_version,
        applied=result.applied,
        skipped=skipped,
    )
    with contextlib.suppress(_USERS_AUDIT_EXCEPTIONS):
        await _emit_user_profile_audit_event(
            http_request,
            user_id=user_id,
            update_keys=[entry.key for entry in payload.updates],
            applied_count=len(result.applied),
            skipped_count=len(result.skipped),
            dry_run=False,
        )
    return response


@router.get("/me", response_model=DeprecatedUserResponse, deprecated=True)
async def get_current_user_profile(
    principal: AuthPrincipal = Depends(get_auth_principal),
    response: Response = None,
) -> DeprecatedUserResponse:
    """
    Deprecated: use /api/v1/users/me/profile.

    Returns the authenticated user's profile information.

    Returns:
        UserResponse with user details
    """
    successor = "/api/v1/users/me/profile"
    if not _legacy_user_me_enabled():
        return JSONResponse(
            status_code=status.HTTP_410_GONE,
            content=_legacy_warning_payload(successor),
        )
    try:
        if response is not None:
            response.headers.update(build_deprecation_headers(successor))
    except _USERS_HEADER_EXCEPTIONS:
        pass
    user_context = await _require_principal_active_verified(
        principal,
        allow_missing_single_user=True,
    )
    return DeprecatedUserResponse(
        warning="deprecated_endpoint",
        successor=successor,
        id=user_context['id'],
        uuid=user_context.get('uuid') or None,
        username=user_context['username'],
        email=user_context.get('email') or "",
        role=user_context.get('role', 'user'),
        is_active=user_context.get('is_active', True),
        is_verified=user_context.get('is_verified', False),
        created_at=user_context.get('created_at', datetime.utcnow()),
        last_login=user_context.get('last_login'),
        storage_quota_mb=user_context.get('storage_quota_mb', 5120),
        storage_used_mb=user_context.get('storage_used_mb', 0.0)
    )


@router.put("/me", response_model=DeprecatedUserResponse, deprecated=True)
async def update_user_profile(
    request: UpdateProfileRequest,
    principal: AuthPrincipal = Depends(get_auth_principal),
    db=Depends(get_db_transaction),
    response: Response = None,
) -> DeprecatedUserResponse:
    """
    Deprecated: use /api/v1/users/me/profile.

    Allows users to update their email address.
    Username changes are not allowed for security reasons.

    Args:
        request: UpdateProfileRequest with new email address (optional)

    Returns:
        Updated UserResponse
    """
    successor = "/api/v1/users/me/profile"
    if not _legacy_user_me_enabled():
        return JSONResponse(
            status_code=status.HTTP_410_GONE,
            content=_legacy_warning_payload(successor),
        )
    try:
        if response is not None:
            response.headers.update(build_deprecation_headers(successor))
    except _USERS_HEADER_EXCEPTIONS:
        pass
    try:
        user_context = await _resolve_user_context(principal, allow_missing=False)
        _require_active_verified_user(user_context)
        user_id = int(user_context["id"])
        updates_made = False

        if request.email and request.email != user_context.get('email'):
            # Update email
            # Use Postgres-style placeholders; test adapters and SQLite shims
            # normalize `$N` to `?` automatically.
            execute_result = await db.execute(
                "UPDATE users SET email = $1, updated_at = CURRENT_TIMESTAMP WHERE id = $2",
                request.email.lower(),
                user_id,
            )
            affected_rows = _extract_affected_rows(execute_result)
            if affected_rows == 0:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="User not found",
                )

            updates_made = True
            user_context['email'] = request.email.lower()
            logger.info(f"Updated email for user {user_context['username']} (ID: {user_id})")

        if not updates_made:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No updates provided"
            )

        # Return updated user info
        return DeprecatedUserResponse(
            warning="deprecated_endpoint",
            successor=successor,
            id=user_context['id'],
            uuid=user_context.get('uuid') or None,
            username=user_context['username'],
            email=user_context.get('email') or "",
            role=user_context.get('role', 'user'),
            is_active=user_context.get('is_active', True),
            is_verified=user_context.get('is_verified', False),
            created_at=user_context.get('created_at', datetime.utcnow()),
            last_login=user_context.get('last_login'),
            storage_quota_mb=user_context.get('storage_quota_mb', 5120),
            storage_used_mb=user_context.get('storage_used_mb', 0.0)
        )

    except HTTPException:
        raise
    except _USERS_ENDPOINT_EXCEPTIONS as e:
        logger.error(f"Failed to update user profile: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update profile"
        ) from e


#######################################################################################################################
#
# Password Management

@router.post("/change-password", response_model=MessageResponse)
async def change_password(
    request: PasswordChangeRequest,
    principal: AuthPrincipal = Depends(get_auth_principal),
    password_service: PasswordService = Depends(get_password_service_dep),
    db=Depends(get_db_transaction)
) -> MessageResponse:
    """
    Change user password

    Allows users to change their password by providing the current password.

    Args:
        request: PasswordChangeRequest with current and new passwords

    Returns:
        MessageResponse confirming password change

    Raises:
        HTTPException: 401 if current password is incorrect, 400 if new password is weak
    """
    try:
        user_context = await _require_principal_active_verified(principal)
        user_id = int(user_context["id"])
        # Prefer the request-scoped DB transaction for compatibility with legacy
        # tests that override get_db_transaction.
        password_hash = await _fetch_password_hash_for_user(db, user_id)
        user_row: dict[str, Any] = {}
        try:
            users_repo = await AuthnzUsersRepo.from_pool()
            fetched_row = await users_repo.get_user_by_id(user_id)
            if isinstance(fetched_row, dict):
                user_row = dict(fetched_row)
                if not password_hash and user_row.get("password_hash"):
                    password_hash = str(user_row.get("password_hash"))
        except _USERS_ENDPOINT_EXCEPTIONS as repo_exc:
            logger.debug("User repo lookup skipped for password change: {}", repo_exc)

        username = str(
            user_row.get("username")
            or user_context.get("username")
            or principal.username
            or f"user-{user_id}"
        )

        if not password_hash:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        # Verify current password
        is_valid, _ = password_service.verify_password(
            request.current_password,
            password_hash
        )
        if not is_valid:
            logger.warning(f"Failed password change attempt for user {username}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Current password is incorrect"
            )

        # Validate new password strength
        try:
            password_service.validate_password_strength(
                request.new_password,
                username
            )
        except WeakPasswordError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            ) from e

        # Hash new password
        new_hash = password_service.hash_password(request.new_password)

        # Update password in database
        await db.execute(
            """
            UPDATE users
            SET password_hash = $1,
                password_changed_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = $2
            """,
            new_hash,
            user_id,
        )
        await db.execute(
            "INSERT INTO password_history (user_id, password_hash) VALUES ($1, $2)",
            user_id,
            new_hash,
        )

        logger.info(f"Password changed for user {username} (ID: {user_id})")

        return MessageResponse(
            message="Password changed successfully",
            details={"user_id": user_id}
        )

    except HTTPException:
        raise
    except _USERS_ENDPOINT_EXCEPTIONS as e:
        logger.error(f"Failed to change password: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to change password"
        ) from e


#######################################################################################################################
#
# API Key Management (per-user)

@router.get(
    "/api-keys",
    response_model=list[APIKeyMetadata],
    dependencies=[Depends(require_api_key_scope("read"))],
)
async def list_api_keys(
    principal: AuthPrincipal = Depends(get_auth_principal)
) -> list[APIKeyMetadata]:
    """List active API keys for the current user (metadata only)."""
    user_context = await _require_principal_active_verified(principal)
    user_id = int(user_context["id"])
    api_mgr = await get_api_key_manager()
    rows = await api_mgr.list_user_keys(user_id=user_id)
    # list_user_keys does not return the raw key; return metadata only
    return [APIKeyMetadata(**row) for row in rows]


@router.post(
    "/api-keys",
    response_model=APIKeyCreateResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_api_key_scope("write"))],
)
async def create_api_key(
    payload: APIKeyCreateRequest,
    request: Request,
    principal: AuthPrincipal = Depends(get_auth_principal)
) -> APIKeyCreateResponse:
    """Create a new API key for the current user and return the key once."""
    user_context = await _require_principal_active_verified(principal)
    user_id = int(user_context["id"])
    api_mgr = await get_api_key_manager()
    try:
        result = await api_mgr.create_api_key(
            user_id=user_id,
            name=payload.name,
            description=payload.description,
            scope=payload.scope,
            expires_in_days=payload.expires_in_days,
            actor_user_id=_principal_user_id(principal),
            actor_subject=principal.subject,
            actor_kind=principal.kind,
            actor_roles=list(principal.roles or []),
        )
    except MandatoryAuditWriteError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Mandatory audit persistence unavailable",
        ) from exc
    return APIKeyCreateResponse(**result)


class SelfVirtualAPIKeyRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    expires_in_days: Optional[int] = Field(30, ge=1)
    allowed_endpoints: Optional[list[str]] = None
    # Generic constraints
    allowed_methods: Optional[list[str]] = None
    allowed_paths: Optional[list[str]] = None
    max_calls: Optional[int] = Field(None, ge=0)
    max_runs: Optional[int] = Field(None, ge=0)
    # Optional LLM budgets (if used by client tools)
    budget_day_tokens: Optional[int] = None
    budget_month_tokens: Optional[int] = None
    budget_day_usd: Optional[float] = None
    budget_month_usd: Optional[float] = None


@router.post(
    "/api-keys/virtual",
    response_model=APIKeyCreateResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_api_key_scope("write"))],
)
async def create_virtual_api_key(
    payload: SelfVirtualAPIKeyRequest,
    request: Request,
    principal: AuthPrincipal = Depends(get_auth_principal)
) -> APIKeyCreateResponse:
    """Create a constrained (virtual/burnable) API key for the current user."""
    user_context = await _require_principal_active_verified(principal)
    user_id = int(user_context["id"])
    api_mgr = await get_api_key_manager()
    try:
        result = await api_mgr.create_virtual_key(
            user_id=user_id,
            name=payload.name,
            description=payload.description,
            expires_in_days=payload.expires_in_days,
            allowed_endpoints=payload.allowed_endpoints,
            budget_day_tokens=payload.budget_day_tokens,
            budget_month_tokens=payload.budget_month_tokens,
            budget_day_usd=payload.budget_day_usd,
            budget_month_usd=payload.budget_month_usd,
            allowed_methods=payload.allowed_methods,
            allowed_paths=payload.allowed_paths,
            max_calls=payload.max_calls,
            max_runs=payload.max_runs,
            actor_user_id=_principal_user_id(principal),
            actor_subject=principal.subject,
            actor_kind=principal.kind,
            actor_roles=list(principal.roles or []),
        )
    except MandatoryAuditWriteError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Mandatory audit persistence unavailable",
        ) from exc
    return APIKeyCreateResponse(**result)


@router.post(
    "/api-keys/{key_id}/rotate",
    response_model=APIKeyCreateResponse,
    dependencies=[Depends(require_api_key_scope("write"))],
)
async def rotate_api_key(
    key_id: int,
    payload: APIKeyRotateRequest,
    request: Request,
    principal: AuthPrincipal = Depends(get_auth_principal)
) -> APIKeyCreateResponse:
    """Rotate an API key (revoke old; create new) and return the new key once."""
    user_context = await _require_principal_active_verified(principal)
    user_id = int(user_context["id"])
    api_mgr = await get_api_key_manager()
    try:
        result = await api_mgr.rotate_api_key(
            key_id=key_id,
            user_id=user_id,
            expires_in_days=payload.expires_in_days,
            actor_user_id=_principal_user_id(principal),
            actor_subject=principal.subject,
            actor_kind=principal.kind,
            actor_roles=list(principal.roles or []),
        )
    except MandatoryAuditWriteError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Mandatory audit persistence unavailable",
        ) from exc
    return APIKeyCreateResponse(**result)


@router.delete(
    "/api-keys/{key_id}",
    response_model=MessageResponse,
    dependencies=[Depends(require_api_key_scope("write"))],
)
async def revoke_api_key(
    key_id: int,
    request: Request,
    principal: AuthPrincipal = Depends(get_auth_principal)
) -> MessageResponse:
    """Revoke an API key for the current user."""
    user_context = await _require_principal_active_verified(principal)
    user_id = int(user_context["id"])
    api_mgr = await get_api_key_manager()
    try:
        success = await api_mgr.revoke_api_key(
            key_id=key_id,
            user_id=user_id,
            actor_user_id=_principal_user_id(principal),
            actor_subject=principal.subject,
            actor_kind=principal.kind,
            actor_roles=list(principal.roles or []),
        )
    except MandatoryAuditWriteError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Mandatory audit persistence unavailable",
        ) from exc
    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="API key not found")
    return MessageResponse(message="API key revoked")


#######################################################################################################################
#
# Session Management

@router.get("/sessions", response_model=list[SessionResponse])
async def list_user_sessions(
    principal: AuthPrincipal = Depends(get_auth_principal),
    session_manager: SessionManager = Depends(get_session_manager_dep)
) -> list[SessionResponse]:
    """
    List all active sessions for the current user.

    Returns:
        List of SessionResponse objects
    """
    from tldw_Server_API.app.api.v1.endpoints.auth import list_user_sessions as _auth_list_sessions

    user_context = await _require_principal_active_verified(principal)
    user_id = int(user_context["id"])
    current_user = {
        "id": user_id,
        "username": str(user_context.get("username") or _principal_username(principal, user_id)),
    }
    return await _auth_list_sessions(
        current_user=current_user,
        session_manager=session_manager,
    )


@router.delete("/sessions/{session_id}", response_model=MessageResponse)
async def revoke_session(
    session_id: int,
    principal: AuthPrincipal = Depends(get_auth_principal),
    session_manager: SessionManager = Depends(get_session_manager_dep)
) -> MessageResponse:
    """
    Revoke a specific session.

    Allows users to log out specific sessions (e.g., on other devices).

    Args:
        session_id: ID of the session to revoke

    Returns:
        MessageResponse confirming revocation

    Raises:
        HTTPException: 404 if session not found or doesn't belong to user
    """
    from tldw_Server_API.app.api.v1.endpoints.auth import revoke_session as _auth_revoke_session

    user_context = await _require_principal_active_verified(principal)
    user_id = int(user_context["id"])
    current_user = {
        "id": user_id,
        "username": str(user_context.get("username") or _principal_username(principal, user_id)),
    }
    return await _auth_revoke_session(
        session_id=session_id,
        current_user=current_user,
        session_manager=session_manager,
    )


@router.post("/sessions/revoke-all", response_model=MessageResponse)
async def revoke_all_sessions(
    principal: AuthPrincipal = Depends(get_auth_principal),
    session_manager: SessionManager = Depends(get_session_manager_dep)
) -> MessageResponse:
    """
    Revoke all sessions for the current user.

    Logs out the user from all devices.

    Returns:
        MessageResponse confirming revocation
    """
    from tldw_Server_API.app.api.v1.endpoints.auth import revoke_all_sessions as _auth_revoke_all_sessions

    user_context = await _require_principal_active_verified(principal)
    user_id = int(user_context["id"])
    current_user = {
        "id": user_id,
        "username": str(user_context.get("username") or _principal_username(principal, user_id)),
    }
    return await _auth_revoke_all_sessions(
        current_user=current_user,
        session_manager=session_manager,
    )


#######################################################################################################################
#
# Storage Management

@router.get("/storage", response_model=StorageQuotaResponse)
async def get_storage_quota(
    principal: AuthPrincipal = Depends(get_auth_principal),
    storage_service: StorageQuotaService = Depends(get_storage_service_dep)
) -> StorageQuotaResponse:
    """
    Get storage quota information for current user

    Returns:
        StorageQuotaResponse with usage details
    """
    user_context: dict[str, Any] | None = None
    try:
        user_context = await _require_principal_active_verified(principal)
        user_id = int(user_context["id"])
        # Get storage info from service
        storage_info = await storage_service.calculate_user_storage(
            user_id,
            update_database=False  # Don't update unless explicitly requested
        )

        return StorageQuotaResponse(
            user_id=user_id,
            storage_used_mb=storage_info['total_mb'],
            storage_quota_mb=storage_info['quota_mb'],
            available_mb=storage_info['available_mb'],
            usage_percentage=storage_info['usage_percentage']
        )

    except _USERS_ENDPOINT_EXCEPTIONS as e:
        logger.error(f"Failed to get storage quota: {e}")
        if user_context is None:
            user_context = await _resolve_user_context(
                principal,
                allow_missing=is_single_user_principal(principal),
            )
        quota = float(user_context.get('storage_quota_mb', 5120))
        used = float(user_context.get('storage_used_mb', 0.0))
        # Return from database values if calculation fails
        return StorageQuotaResponse(
            user_id=int(user_context['id']),
            storage_used_mb=used,
            storage_quota_mb=quota,
            available_mb=max(0, quota - used),
            usage_percentage=round((used / quota * 100) if quota > 0 else 0, 1),
        )


@router.post("/storage/recalculate", response_model=StorageQuotaResponse)
async def recalculate_storage(
    principal: AuthPrincipal = Depends(get_auth_principal),
    storage_service: StorageQuotaService = Depends(get_storage_service_dep)
) -> StorageQuotaResponse:
    """
    Recalculate storage usage for current user

    Forces a recalculation of actual disk usage and updates the database.

    Returns:
        StorageQuotaResponse with updated usage details
    """
    try:
        user_context = await _require_principal_active_verified(principal)
        user_id = int(user_context["id"])
        username = str(user_context.get("username") or _principal_username(principal, user_id))
        # Recalculate and update database
        storage_info = await storage_service.calculate_user_storage(
            user_id,
            update_database=True
        )

        logger.info(f"Recalculated storage for user {username}: {storage_info['total_mb']:.2f}MB")

        return StorageQuotaResponse(
            user_id=user_id,
            storage_used_mb=storage_info['total_mb'],
            storage_quota_mb=storage_info['quota_mb'],
            available_mb=storage_info['available_mb'],
            usage_percentage=storage_info['usage_percentage']
        )

    except _USERS_ENDPOINT_EXCEPTIONS as e:
        logger.error(f"Failed to recalculate storage: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to recalculate storage"
        ) from e


#
# End of users.py
#######################################################################################################################
