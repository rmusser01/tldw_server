# auth.py
# Description: Authentication endpoints for user login, logout, refresh, and registration
#
# Imports
import asyncio
import base64
import contextlib
import json
import os
import re
import secrets
import string
import time
from datetime import datetime, timedelta, timezone
from importlib import import_module
from typing import Any, Optional

#
# 3rd-party imports
from fastapi import APIRouter, Depends, Form, HTTPException, Query, Request, Response, status
from fastapi.responses import RedirectResponse
from fastapi.security import OAuth2PasswordRequestForm
from loguru import logger
from pydantic import BaseModel, EmailStr, Field

from tldw_Server_API.app.api.v1.API_Deps.Audit_DB_Deps import (
    get_or_create_audit_service_for_user_id,
)
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
    check_auth_rate_limit,
    get_auth_principal,
    get_current_active_user,  # compat export used by integration tests
    get_db_transaction,
    get_jwt_service_dep,
    get_password_service_dep,
    get_rate_limiter_dep,
    get_registration_service_dep,
    get_session_manager_dep,
)
from tldw_Server_API.app.api.v1.API_Deps.federation_deps import (
    get_federation_provisioning_service_dep,
    get_identity_provider_repo_dep,
    get_oidc_federation_service_dep,
    require_enterprise_federation,
)
from tldw_Server_API.app.api.v1.utils.deprecation import build_deprecation_headers

#
# Local imports
from tldw_Server_API.app.api.v1.schemas.auth_schemas import (
    DeprecatedUserResponse,
    MagicLinkRequest,
    MagicLinkVerifyRequest,
    MessageResponse,
    MFAChallengeResponse,
    RefreshTokenRequest,
    RegisterRequest,
    RegistrationResponse,
    SessionResponse,
    TokenResponse,
)
from tldw_Server_API.app.core.Audit.unified_audit_service import (
    AuditContext,
    AuditEventType,
    MandatoryAuditWriteError,
)
from tldw_Server_API.app.core.AuthNZ.api_key_manager import get_api_key_manager
from tldw_Server_API.app.core.AuthNZ.auth_governor import get_auth_governor
from tldw_Server_API.app.core.AuthNZ.csrf_protection import (
    global_settings as _csrf_globals,
)
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, is_postgres_backend
from tldw_Server_API.app.core.AuthNZ.exceptions import (
    DatabaseError,
    DuplicateOrganizationError,
    DuplicateUserError,
    InvalidRegistrationCodeError,
    InvalidTokenError,
    RegistrationError,
    SessionError,
    TokenExpiredError,
    WeakPasswordError,
)
from tldw_Server_API.app.core.AuthNZ.input_validation import get_input_validator
from tldw_Server_API.app.core.AuthNZ.ip_allowlist import (
    is_single_user_ip_allowed,
    resolve_client_ip,
)
from tldw_Server_API.app.core.AuthNZ.jwt_service import JWTService
from tldw_Server_API.app.core.AuthNZ.orgs_teams import list_memberships_for_user
from tldw_Server_API.app.core.AuthNZ.password_service import PasswordService
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.rate_limiter import RateLimiter
from tldw_Server_API.app.core.AuthNZ.repos.identity_provider_repo import IdentityProviderRepo
from tldw_Server_API.app.core.AuthNZ.session_manager import SessionManager
from tldw_Server_API.app.core.AuthNZ.settings import Settings, get_profile, get_settings
from tldw_Server_API.app.core.AuthNZ.token_blacklist import get_token_blacklist
from tldw_Server_API.app.core.Resource_Governance.governor import MemoryResourceGovernor, RGRequest
from tldw_Server_API.app.core.Resource_Governance.policy_loader import default_policy_loader
from tldw_Server_API.app.core.Resource_Governance.tenant import hash_entity
from tldw_Server_API.app.core.Metrics.metrics_logger import log_counter, log_histogram
from tldw_Server_API.app.services.auth_service import (
    apply_password_reset as _svc_apply_password_reset,
    fetch_active_user_by_id,
    fetch_password_reset_token_record as _svc_fetch_password_reset_token_record,
    fetch_user_by_email_for_password_reset as _svc_fetch_user_by_email_for_password_reset,
    fetch_user_by_email_for_verification as _svc_fetch_user_by_email_for_verification,
    fetch_user_by_login_identifier,
    mark_user_verified as _svc_mark_user_verified,
    store_password_reset_token as _svc_store_password_reset_token,
    verify_user_email_once as _svc_verify_user_email_once,
    update_user_last_login,
    update_user_password_hash,
)
from tldw_Server_API.app.services.registration_service import RegistrationService
from tldw_Server_API.app.core.AuthNZ.federation.claim_mapping import preview_claim_mapping
from tldw_Server_API.app.core.AuthNZ.federation.oidc_service import OIDCFederationService
from tldw_Server_API.app.core.AuthNZ.federation.provisioning_service import FederationProvisioningService
from tldw_Server_API.app.core.AuthNZ.federation.state_repo import FederationStateRepo
from tldw_Server_API.app.core.testing import (
    env_flag_enabled as _env_flag_enabled,
    is_explicit_pytest_runtime as _is_explicit_pytest_runtime,
    is_test_mode as _is_test_mode,
    is_truthy as _is_truthy,
)

_AUTH_NONCRITICAL_EXCEPTIONS = (
    asyncio.CancelledError,
    AssertionError,
    AttributeError,
    ConnectionError,
    FileNotFoundError,
    IndexError,
    KeyError,
    LookupError,
    OSError,
    PermissionError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    UnicodeDecodeError,
    json.JSONDecodeError,
    HTTPException,
    DatabaseError,
    DuplicateOrganizationError,
    DuplicateUserError,
    InvalidRegistrationCodeError,
    InvalidTokenError,
    RegistrationError,
    SessionError,
    TokenExpiredError,
    WeakPasswordError,
)

#######################################################################################################################
#
# Router Configuration

router = APIRouter(
    prefix="/auth",
    tags=["authentication"],
    responses={404: {"description": "Not found"}}
)
_AUTH_ENDPOINT_RG_LOCK: Optional[asyncio.Lock] = None
_AUTH_RG_DIAGNOSTICS_SHIM_LOGGED: set[str] = set()

def _extract_bearer_token(auth_header: Optional[str]) -> str:
    """Parse Authorization header and return Bearer token (case-insensitive)."""
    if not auth_header:
        return ""
    try:
        scheme, _, credential = auth_header.strip().partition(" ")
        if scheme.lower() != "bearer" or not credential.strip():
            return ""
        return credential.strip()
    except _AUTH_NONCRITICAL_EXCEPTIONS:
        return ""


def _legacy_user_me_enabled() -> bool:
    raw = os.getenv("ENABLE_LEGACY_USER_ME_ENDPOINTS", "true")
    return _is_truthy(str(raw).strip().lower())


def _legacy_warning_payload(successor: str) -> dict[str, str]:
    return {"warning": "deprecated_endpoint", "successor": successor}


def _get_admin_module_ensure_sqlite_authnz_ready_if_test_mode():
    from tldw_Server_API.app.api.v1.endpoints import admin as admin_mod

    return admin_mod._ensure_sqlite_authnz_ready_if_test_mode


def _get_federation_state_repo(session_manager: SessionManager) -> FederationStateRepo:
    return FederationStateRepo(session_manager=session_manager)


async def _resolve_federation_provider(
    *,
    repo: IdentityProviderRepo,
    provider_slug: str,
    org_id: int | None = None,
) -> dict[str, Any] | None:
    normalized_slug = str(provider_slug or "").strip()
    if org_id is not None:
        provider = await repo.get_provider_by_slug(
            slug=normalized_slug,
            owner_scope_type="org",
            owner_scope_id=int(org_id),
        )
        if provider:
            return provider
    return await repo.get_provider_by_slug(
        slug=normalized_slug,
        owner_scope_type="global",
        owner_scope_id=None,
    )


def _get_email_service():
    """Resolve the email service lazily to honor monkeypatched modules in tests."""
    module = import_module("tldw_Server_API.app.core.AuthNZ.email_service")
    return module.get_email_service()


def _get_mfa_service():
    """Resolve the MFA service lazily to honor monkeypatched modules in tests."""
    module = import_module("tldw_Server_API.app.core.AuthNZ.mfa_service")
    return module.get_mfa_service()


async def _is_mfa_backend_supported() -> bool:
    """Return True when MFA storage is supported by the active AuthNZ backend."""
    try:
        mfa_service = _get_mfa_service()
        initializer = getattr(mfa_service, "initialize", None)
        if callable(initializer):
            await initializer()
        supports_backend = getattr(mfa_service, "supports_backend", None)
        if callable(supports_backend):
            return bool(supports_backend())
        # Test stubs may omit supports_backend(); fall back to canonical backend detection.
        return await is_postgres_backend()
    except _AUTH_NONCRITICAL_EXCEPTIONS:
        logger.debug("Failed to resolve MFA service backend support flag", exc_info=True)
    return False


async def _ensure_mfa_available():
    """Validate MFA endpoints are allowed under current configuration."""
    settings = get_settings()
    if settings.AUTH_MODE != "multi_user":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="MFA is only available in multi-user deployments",
        )
    if not await _is_mfa_backend_supported():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="MFA requires a PostgreSQL database backend",
        )


def _mfa_setup_cache_key(user_id: int) -> str:
    return f"mfa:setup:{user_id}"


def _get_mfa_setup_ttl_seconds() -> int:
    try:
        ttl = int(os.getenv("MFA_SETUP_TTL_SECONDS", "600"))
    except (TypeError, ValueError):
        ttl = 600
    return max(ttl, 60)

def _mfa_login_cache_key(session_token: str) -> str:
    return f"mfa:login:{session_token}"


def _get_mfa_login_ttl_seconds() -> int:
    try:
        ttl = int(os.getenv("MFA_LOGIN_TTL_SECONDS", "600"))
    except (TypeError, ValueError):
        ttl = 600
    return max(ttl, 60)


def _normalize_magic_email(email: str) -> str:
    return str(email or "").strip().lower()


async def _safe_audit_log_login_event(
    *,
    user_id: int,
    username: str,
    ip: str,
    user_agent: str,
    success: bool,
) -> None:
    try:
        svc = await get_or_create_audit_service_for_user_id(user_id)
        await svc.log_login(
            user_id=user_id,
            username=username,
            ip_address=ip,
            user_agent=user_agent,
            success=success,
        )
        flush_on_login = _env_flag_enabled("AUDIT_FLUSH_ON_LOGIN")
        test_mode = _is_test_mode()
        if flush_on_login or test_mode:
            await svc.flush()
    except _AUTH_NONCRITICAL_EXCEPTIONS as exc:
        logger.debug(
            "Federated login audit failed for user_id={}: {}",
            user_id,
            exc,
            exc_info=True,
        )


async def _issue_multi_user_tokens(
    *,
    request: Request,
    db: Any,
    user: dict[str, Any],
    jwt_service: JWTService,
    session_manager: SessionManager,
    settings: Settings,
) -> TokenResponse:
    user_id = int(user["id"])
    username = str(user.get("username") or "")
    client_ip = _auth_request_client_ip(request)
    user_agent = request.headers.get("User-Agent", "Unknown")

    temp_access = f"temp_access_{secrets.token_urlsafe(16)}"
    temp_refresh = f"temp_refresh_{secrets.token_urlsafe(16)}"
    temp_session_info = await session_manager.create_session(
        user_id=user_id,
        access_token=temp_access,
        refresh_token=temp_refresh,
        ip_address=client_ip,
        user_agent=user_agent,
    )
    session_id = int(temp_session_info["session_id"])

    if settings.AUTH_MODE == "multi_user":
        await _ensure_user_org_membership(user_id, user.get("username"))

    scope_claims = await _build_scope_claims(user_id)
    add_claims = dict(scope_claims)
    add_claims["session_id"] = session_id
    access_token = jwt_service.create_access_token(
        user_id=user_id,
        username=username,
        role=str(user.get("role") or settings.DEFAULT_USER_ROLE),
        additional_claims=add_claims,
    )
    refresh_token = jwt_service.create_refresh_token(
        user_id=user_id,
        username=username,
        additional_claims=add_claims,
    )
    await session_manager.update_session_tokens(
        session_id=session_id,
        access_token=access_token,
        refresh_token=refresh_token,
    )
    await update_user_last_login(db, user_id, datetime.utcnow())
    await _safe_audit_log_login_event(
        user_id=user_id,
        username=username,
        ip=client_ip,
        user_agent=user_agent,
        success=True,
    )
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


@router.get(
    "/federation/{provider_slug}/login",
    dependencies=[Depends(check_auth_rate_limit)],
)
async def federation_login(
    provider_slug: str,
    request: Request,
    org_id: int | None = Query(None, ge=1),
    session_manager: SessionManager = Depends(get_session_manager_dep),
) -> RedirectResponse:
    await _get_admin_module_ensure_sqlite_authnz_ready_if_test_mode()()
    require_enterprise_federation()
    client_ip = _auth_request_client_ip(request)
    allowed, retry_after = await _reserve_auth_rg_requests(
        request,
        policy_id="authnz.federation.login",
        entity=f"ip:{client_ip}",
        tags={
            "auth_endpoint": "federation_login",
            "provider_slug": str(provider_slug or ""),
        },
    )
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many federation login attempts. Please try again later.",
            headers={"Retry-After": str(int(retry_after or 1))},
        )

    repo = await get_identity_provider_repo_dep()
    provider = await _resolve_federation_provider(
        repo=repo,
        provider_slug=provider_slug,
        org_id=org_id,
    )
    if not provider or not bool(provider.get("enabled")):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Identity provider not found",
        )

    redirect_uri = str(request.url_for("federation_callback", provider_slug=provider_slug))
    oidc_service = get_oidc_federation_service_dep()
    try:
        auth_request = await oidc_service.build_authorization_request(
            provider=provider,
            redirect_uri=redirect_uri,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc

    state_repo = _get_federation_state_repo(session_manager)
    await state_repo.create_state(
        state=auth_request["state"],
        payload={
            "provider_id": int(provider["id"]),
            "provider_slug": str(provider["slug"]),
            "owner_scope_type": str(provider.get("owner_scope_type") or "global"),
            "owner_scope_id": provider.get("owner_scope_id"),
            "redirect_uri": redirect_uri,
            "nonce": auth_request["nonce"],
            "code_verifier": auth_request["code_verifier"],
            "expires_at": auth_request["expires_at"].isoformat(),
        },
        ttl_seconds=int(auth_request["ttl_seconds"]),
    )
    return RedirectResponse(url=str(auth_request["auth_url"]), status_code=status.HTTP_307_TEMPORARY_REDIRECT)

@router.get(
    "/federation/{provider_slug}/callback",
    name="federation_callback",
    response_model=TokenResponse,
    dependencies=[Depends(check_auth_rate_limit)],
)
async def federation_callback(
    provider_slug: str,
    code: str,
    state: str,
    request: Request,
    jwt_service: JWTService = Depends(get_jwt_service_dep),
    session_manager: SessionManager = Depends(get_session_manager_dep),
    settings: Settings = Depends(get_settings),
    provisioning_service: FederationProvisioningService = Depends(get_federation_provisioning_service_dep),
) -> TokenResponse:
    await _get_admin_module_ensure_sqlite_authnz_ready_if_test_mode()()
    require_enterprise_federation()

    code_value = str(code or "").strip()
    state_value = str(state or "").strip()
    if not code_value or not state_value:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing OIDC callback parameters",
        )
    client_ip = _auth_request_client_ip(request)
    allowed, retry_after = await _reserve_auth_rg_requests(
        request,
        policy_id="authnz.federation.callback",
        entity=f"ip:{client_ip}",
        tags={
            "auth_endpoint": "federation_callback",
            "provider_slug": str(provider_slug or ""),
        },
    )
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many federation callback attempts. Please try again later.",
            headers={"Retry-After": str(int(retry_after or 1))},
        )

    state_repo = _get_federation_state_repo(session_manager)
    state_record = await state_repo.consume_state(state=state_value)
    if not state_record:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or expired OIDC state",
        )

    repo = await get_identity_provider_repo_dep()
    provider_id = int(state_record.get("provider_id") or 0)
    provider = await repo.get_provider(provider_id) if provider_id > 0 else None
    if not provider or not bool(provider.get("enabled")):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Identity provider not found",
        )
    if str(provider.get("slug") or "").strip() != str(provider_slug or "").strip():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="OIDC state provider mismatch",
        )

    redirect_uri = str(state_record.get("redirect_uri") or "").strip()
    code_verifier = str(state_record.get("code_verifier") or "").strip()
    if not redirect_uri or not code_verifier:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="OIDC state is incomplete",
        )

    oidc_service = get_oidc_federation_service_dep()
    try:
        claims = await oidc_service.exchange_authorization_code(
            provider=provider,
            code=code_value,
            redirect_uri=redirect_uri,
            code_verifier=code_verifier,
            nonce=(str(state_record.get("nonce")).strip() if state_record.get("nonce") else None),
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    mapped_claims = preview_claim_mapping(provider.get("claim_mapping"), claims)
    subject = str(mapped_claims.get("subject") or "").strip()
    if not subject:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="OIDC claims did not resolve a subject",
        )

    db_pool = provisioning_service.db_pool
    user = await provisioning_service.resolve_user_for_login(
        provider=provider,
        mapped_claims=mapped_claims,
    )
    if not user:
        policy = provider.get("provisioning_policy") or {}
        if not bool(policy.get("jit_create")):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Federated account is not linked to a local user",
            )
        user = await _provision_federated_user(
            mapped_claims=mapped_claims,
            registration_service=RegistrationService(db_pool=db_pool),
            default_role=settings.DEFAULT_USER_ROLE,
        )
    if not bool(user.get("is_active", True)):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is inactive. Please contact support.",
        )

    await provisioning_service.upsert_identity_link(
        provider=provider,
        user_id=int(user["id"]),
        mapped_claims=mapped_claims,
        raw_claims=claims,
    )
    await provisioning_service.apply_mapped_grants(
        provider=provider,
        user_id=int(user["id"]),
        mapped_claims=mapped_claims,
    )
    db_pool = await get_db_pool()
    async with db_pool.transaction() as db:
        return await _issue_multi_user_tokens(
            request=request,
            db=db,
            user=user,
            jwt_service=jwt_service,
            session_manager=session_manager,
            settings=settings,
        )


def _current_user_value(user: Any, key: str, default: Any = None) -> Any:
    """Read a user field from either dict-backed or attribute-backed user objects."""
    if isinstance(user, dict):
        return user.get(key, default)
    return getattr(user, key, default)


def _current_user_id(user: Any) -> Optional[int]:
    raw = _current_user_value(user, "id", None)
    if raw is None:
        raw = _current_user_value(user, "user_id", None)
    try:
        return int(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


_PLATFORM_ADMIN_ROLES = frozenset({"admin"})
_ADMIN_CLAIM_PERMISSIONS = frozenset({"*", "system.configure"})


def _normalized_claim_values(values: Any) -> set[str]:
    if not isinstance(values, (list, tuple, set)):
        return set()
    return {
        str(value).strip().lower()
        for value in values
        if str(value).strip()
    }


def _current_user_has_admin_claims(user: Any) -> bool:
    role = str(_current_user_value(user, "role", "") or "").strip().lower()
    if role in _PLATFORM_ADMIN_ROLES:
        return True

    roles = _normalized_claim_values(_current_user_value(user, "roles", []))
    if roles & _PLATFORM_ADMIN_ROLES:
        return True

    permissions = _normalized_claim_values(_current_user_value(user, "permissions", []))
    return bool(permissions & _ADMIN_CLAIM_PERMISSIONS)


def _principal_primary_role(principal: AuthPrincipal) -> str:
    roles = [
        str(role).strip().lower()
        for role in (principal.roles or [])
        if str(role).strip()
    ]
    if any(role in _PLATFORM_ADMIN_ROLES for role in roles):
        return "admin"
    permissions = _normalized_claim_values(principal.permissions)
    if permissions & _ADMIN_CLAIM_PERMISSIONS:
        return "admin"
    return roles[0] if roles else "user"


async def _fetch_user_by_email_for_password_reset(db: Any, email: str) -> dict[str, Any] | None:
    return await _svc_fetch_user_by_email_for_password_reset(db, email)


async def _store_password_reset_token(
    db: Any,
    *,
    user_id: int,
    token_hash: str,
    expires_at: datetime,
    ip_address: str,
) -> None:
    await _svc_store_password_reset_token(
        db,
        user_id=user_id,
        token_hash=token_hash,
        expires_at=expires_at,
        ip_address=ip_address,
    )


async def _fetch_password_reset_token_record(
    db: Any,
    *,
    user_id: int,
    hash_candidates: list[str],
) -> tuple[Optional[int], Optional[Any]]:
    return await _svc_fetch_password_reset_token_record(
        db,
        user_id=user_id,
        hash_candidates=hash_candidates,
    )


async def _apply_password_reset(
    db: Any,
    *,
    user_id: int,
    new_password_hash: str,
    token_record_id: int,
    now_utc: datetime,
) -> None:
    await _svc_apply_password_reset(
        db,
        user_id=user_id,
        new_password_hash=new_password_hash,
        token_record_id=token_record_id,
        now_utc=now_utc,
    )


async def _verify_user_email_once(
    db: Any,
    *,
    user_id: int,
    email: str,
    now_utc: datetime,
) -> int:
    return await _svc_verify_user_email_once(
        db,
        user_id=user_id,
        email=email,
        now_utc=now_utc,
    )


async def _fetch_user_by_email_for_verification(db: Any, email: str) -> dict[str, Any] | None:
    return await _svc_fetch_user_by_email_for_verification(db, email)


async def _mark_user_verified(db: Any, user_id: int, now_utc: datetime) -> None:
    await _svc_mark_user_verified(db, user_id, now_utc)


def _current_user_username(user: Any) -> str:
    username = _current_user_value(user, "username", None)
    if username:
        return str(username)
    email = _current_user_value(user, "email", None)
    if email:
        return str(email).split("@", 1)[0]
    uid = _current_user_id(user)
    return f"user-{uid}" if uid is not None else "user"


def _current_user_primary_role(user: Any) -> str:
    role = _current_user_value(user, "role", None)
    if role:
        role_text = str(role).strip().lower()
        if role_text:
            return role_text
    raw_roles = _current_user_value(user, "roles", [])
    if isinstance(raw_roles, (list, tuple, set)):
        for raw_role in raw_roles:
            role_text = str(raw_role).strip().lower()
            if role_text:
                return role_text
    if _current_user_has_admin_claims(user):
        return "admin"
    return "user"


def _derive_username_from_email(email: str) -> str:
    local = email.split("@", 1)[0].strip().lower()
    local = re.sub(r"[^a-z0-9_-]+", "-", local).strip("-")
    if len(local) < 3:
        local = f"user-{secrets.token_hex(2)}"
    return local[:32]


def _derive_username_from_claims(mapped_claims: dict[str, Any]) -> str:
    username_hint = str(mapped_claims.get("username") or "").strip().lower()
    if username_hint:
        username_hint = re.sub(r"[^a-z0-9_-]+", "-", username_hint).strip("-")
        if len(username_hint) >= 3:
            return username_hint[:32]
    return _derive_username_from_email(str(mapped_claims.get("email") or "user"))


def _username_with_suffix(base: str, suffix: str) -> str:
    if len(base) + len(suffix) <= 32:
        return f"{base}{suffix}"
    return f"{base[: 32 - len(suffix)]}{suffix}"


def _generate_magic_password(length: int = 16) -> str:
    # Ensure a mix of upper, lower, digit, and special characters.
    specials = "!@#$%^&*"
    choices = string.ascii_letters + string.digits + specials
    base = [
        secrets.choice(string.ascii_lowercase),
        secrets.choice(string.ascii_uppercase),
        secrets.choice(string.digits),
        secrets.choice(specials),
    ]
    remaining = max(length - len(base), 0)
    base.extend(secrets.choice(choices) for _ in range(remaining))
    secrets.SystemRandom().shuffle(base)
    return "".join(base)


async def _provision_federated_user(
    *,
    mapped_claims: dict[str, Any],
    registration_service: RegistrationService,
    default_role: str,
) -> dict[str, Any]:
    email = str(mapped_claims.get("email") or "").strip().lower()
    if not email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="OIDC claims did not resolve an email for JIT provisioning",
        )

    password = _generate_magic_password()
    username_base = _derive_username_from_claims(mapped_claims)
    user_info: Optional[dict[str, Any]] = None

    for attempt in range(5):
        suffix = f"-{secrets.token_hex(2)}"
        username = username_base if attempt == 0 else _username_with_suffix(username_base, suffix)
        try:
            user_info = await registration_service.register_user(
                username=username,
                email=email,
                password=password,
                role_override=default_role,
                is_verified_override=True,
                system_provisioning=True,
            )
            break
        except DuplicateUserError as exc:
            if getattr(exc, "field", None) == "username":
                continue
            if getattr(exc, "field", None) == "email":
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Federated email already belongs to a local user",
                ) from exc
            raise
        except RegistrationError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc

    if not user_info:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Could not provision a unique username for the federated user",
        )

    return {
        "id": int(user_info["user_id"]),
        "username": str(user_info["username"]),
        "email": str(user_info["email"]),
        "role": str(user_info["role"]),
        "is_active": bool(user_info.get("is_active", True)),
        "is_verified": bool(user_info.get("is_verified", False)),
    }


async def _ensure_user_org_membership(user_id: int, username: Optional[str] = None) -> None:
    """Ensure a user has at least one org membership; create a personal org if not."""
    try:
        memberships = await list_memberships_for_user(int(user_id))
    except _AUTH_NONCRITICAL_EXCEPTIONS:
        memberships = []
    if memberships:
        return

    base_name = f"{username or 'Personal'} Workspace"
    base_name = base_name.strip() or "Personal Workspace"

    try:
        from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
        from tldw_Server_API.app.core.AuthNZ.orgs_teams import create_organization
        from tldw_Server_API.app.core.AuthNZ.repos.orgs_teams_repo import AuthnzOrgsTeamsRepo

        org = None
        for attempt in range(3):
            name = base_name if attempt == 0 else f"{base_name}-{secrets.token_hex(2)}"
            try:
                org = await create_organization(name=name, owner_user_id=int(user_id))
                break
            except DuplicateOrganizationError:
                continue

        if not org:
            return

        pool = await get_db_pool()
        repo = AuthnzOrgsTeamsRepo(db_pool=pool)
        await repo.add_org_member(org_id=org["id"], user_id=int(user_id), role="owner")
    except Exception as exc:
        # Best-effort bootstrap: org creation failures must not block login flows.
        logger.warning("Org bootstrap failed for user {}: {}", user_id, exc)

def _is_pytest_context() -> bool:
    """Return True when running under pytest or explicit test-mode flags."""
    try:
        if _is_explicit_pytest_runtime():
            return True
        if _is_test_mode():
            return True
        if _env_flag_enabled("TESTING"):
            return True
    except _AUTH_NONCRITICAL_EXCEPTIONS:
        return False
    return False


def _is_enterprise_admin_ui_mode() -> bool:
    raw_value = os.getenv("ADMIN_UI_ENTERPRISE_MODE", "")
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _auth_request_client_ip(request: Request) -> str:
    try:
        settings = get_settings()
    except _AUTH_NONCRITICAL_EXCEPTIONS:
        settings = None
    try:
        resolved = resolve_client_ip(request, settings)
        if resolved:
            return str(resolved)
    except _AUTH_NONCRITICAL_EXCEPTIONS:
        pass
    try:
        return request.client.host if request.client else "unknown"
    except _AUTH_NONCRITICAL_EXCEPTIONS:
        return "unknown"


def _auth_hashed_entity(raw_value: str) -> str:
    try:
        return hash_entity(raw_value)
    except _AUTH_NONCRITICAL_EXCEPTIONS:
        return raw_value


def _log_auth_rg_diagnostics_only_shim(
    *,
    reason: str,
    policy_id: str,
    exc: Optional[Exception] = None,
) -> None:
    """Emit a one-shot warning when AuthNZ ingress falls back to diagnostics-only mode."""
    if reason in _AUTH_RG_DIAGNOSTICS_SHIM_LOGGED:
        return
    _AUTH_RG_DIAGNOSTICS_SHIM_LOGGED.add(reason)
    if exc is None:
        logger.warning(
            "Auth endpoint ResourceGovernor unavailable for policy {} (reason={}); "
            "allowing request via diagnostics-only shim.",
            policy_id,
            reason,
        )
        return
    logger.warning(
        "Auth endpoint ResourceGovernor unavailable for policy {} (reason={}); "
        "allowing request via diagnostics-only shim. error={}",
        policy_id,
        reason,
        exc,
    )


def _auth_rg_policy_defined(request: Request, policy_id: str, governor: Any) -> bool:
    if not policy_id:
        return False
    loader = None
    try:
        loader = getattr(request.app.state, "rg_policy_loader", None)
    except _AUTH_NONCRITICAL_EXCEPTIONS:
        loader = None
    if loader is None:
        loader = getattr(governor, "_policy_loader", None)
    if loader is None:
        return True
    try:
        policy = loader.get_policy(policy_id)
    except _AUTH_NONCRITICAL_EXCEPTIONS:
        return True
    return bool(policy)


async def _get_auth_endpoint_rg_governor(request: Request) -> Optional[Any]:
    try:
        app = request.app
        state = getattr(app, "state", None)
    except _AUTH_NONCRITICAL_EXCEPTIONS:
        app = None
        state = None
    if app is None or state is None:
        return None

    try:
        governor = getattr(state, "rg_governor", None) or getattr(state, "auth_endpoint_rg_governor", None)
    except _AUTH_NONCRITICAL_EXCEPTIONS:
        governor = None
    if governor is not None:
        return governor

    global _AUTH_ENDPOINT_RG_LOCK

    if _AUTH_ENDPOINT_RG_LOCK is None:
        _AUTH_ENDPOINT_RG_LOCK = asyncio.Lock()

    async with _AUTH_ENDPOINT_RG_LOCK:
        governor = getattr(state, "rg_governor", None) or getattr(state, "auth_endpoint_rg_governor", None)
        if governor is not None:
            return governor
        try:
            loader = getattr(state, "rg_policy_loader", None)
            if loader is None:
                loader = default_policy_loader()
                await loader.load_once()
                setattr(state, "rg_policy_loader", loader)
            fallback = MemoryResourceGovernor(policy_loader=loader)
            setattr(state, "auth_endpoint_rg_governor", fallback)
            return fallback
        except _AUTH_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug("Auth endpoint RG fallback governor init failed: {}", exc)
            return None


async def _reserve_auth_rg_requests(
    request: Request,
    *,
    policy_id: str,
    entity: Optional[str] = None,
    tags: Optional[dict[str, str]] = None,
    fail_open: bool = True,
) -> tuple[bool, Optional[int]]:
    _ = fail_open  # Compatibility parameter; legacy fallback limiter is retired.

    rg_entity = entity or f"ip:{_auth_request_client_ip(request)}"

    governor = await _get_auth_endpoint_rg_governor(request)
    if governor is None:
        _log_auth_rg_diagnostics_only_shim(reason="governor_unavailable", policy_id=policy_id)
        return True, None

    if not _auth_rg_policy_defined(request, policy_id, governor):
        _log_auth_rg_diagnostics_only_shim(reason="policy_missing", policy_id=policy_id)
        return True, None

    op_id = f"auth-rg-{policy_id}-{time.time_ns()}"
    metadata = {
        "policy_id": policy_id,
        "module": "auth",
        "endpoint": str(getattr(getattr(request, "url", None), "path", "/auth")),
    }
    if tags:
        metadata.update({str(k): str(v) for k, v in tags.items()})

    try:
        decision, handle_id = await governor.reserve(
            RGRequest(
                entity=rg_entity,
                categories={"requests": {"units": 1}},
                tags=metadata,
            ),
            op_id=op_id,
        )
        if handle_id:
            with contextlib.suppress(_AUTH_NONCRITICAL_EXCEPTIONS):
                await governor.commit(handle_id, actuals={"requests": 1}, op_id=op_id)
        if decision.allowed:
            return True, None
        return False, int(decision.retry_after or 1)
    except _AUTH_NONCRITICAL_EXCEPTIONS as exc:
        _log_auth_rg_diagnostics_only_shim(reason="rg_reserve_failed", policy_id=policy_id, exc=exc)
        return True, None


async def _ensure_mfa_cache_available(session_manager: SessionManager, settings: Settings) -> None:
    """Ensure MFA ephemeral storage is backed by Redis (mandatory for MFA flows)."""
    test_context = _is_pytest_context()

    def _supports_ephemeral_stub(sm: SessionManager) -> bool:
        return callable(getattr(sm, "store_ephemeral_value", None)) and callable(
            getattr(sm, "get_ephemeral_value", None)
        )

    if not settings.REDIS_URL:
        if test_context and _supports_ephemeral_stub(session_manager):
            return
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="MFA requires Redis-backed ephemeral storage",
        )
    try:
        if not getattr(session_manager, "_initialized", False):
            await session_manager.initialize()
    except _AUTH_NONCRITICAL_EXCEPTIONS:
        # If initialization fails, we'll fall through to the redis_client check below.
        logger.debug(
            "MFA cache init: session_manager.initialize() failed (redis_url={}); continuing to redis_client check.",
            settings.REDIS_URL,
            exc_info=True,
        )
    if getattr(session_manager, "redis_client", None) is None:
        if test_context and _supports_ephemeral_stub(session_manager):
            return
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="MFA requires Redis-backed ephemeral storage",
        )


#######################################################################################################################
#
# Enhanced Auth Schemas

class ForgotPasswordRequest(BaseModel):
    """Request for password reset."""
    email: EmailStr = Field(..., description="Email address to send reset link")


class ResetPasswordRequest(BaseModel):
    """Request to reset password with token."""
    token: str = Field(..., description="Password reset token")
    new_password: str = Field(..., min_length=8, description="New password")


class VerifyEmailRequest(BaseModel):
    """Request to verify email."""
    token: str = Field(..., description="Email verification token")


class ResendVerificationRequest(BaseModel):
    """Request to resend verification email."""
    email: EmailStr = Field(..., description="Email address to resend verification")


class MFASetupResponse(BaseModel):
    """Response for MFA setup initiation."""
    secret: str = Field(..., description="TOTP secret (store securely)")
    qr_code: str = Field(..., description="QR code image as base64")
    backup_codes: list[str] = Field(..., description="Backup codes for recovery")


class MFAVerifyRequest(BaseModel):
    """Request to verify MFA setup."""
    token: str = Field(..., description="6-digit TOTP token")


class MFALoginRequest(BaseModel):
    """Request for MFA during login."""
    session_token: str = Field(..., description="Temporary session token from initial login")
    mfa_token: str = Field(..., description="6-digit TOTP token or backup code")


class LogoutRequest(BaseModel):
    """Request for logout."""
    all_devices: bool = Field(default=False, description="Logout from all devices")


#######################################################################################################################
#
# Register endpoint diagnostics (test-only)

async def _build_scope_claims(user_id: int) -> dict[str, Any]:
    try:
        memberships = await list_memberships_for_user(int(user_id))
    except _AUTH_NONCRITICAL_EXCEPTIONS:
        return {}

    team_ids = sorted({m.get("team_id") for m in memberships if m.get("team_id") is not None})
    org_ids = sorted({m.get("org_id") for m in memberships if m.get("org_id") is not None})

    claims: dict[str, Any] = {}
    if team_ids:
        claims["team_ids"] = team_ids
    if org_ids:
        claims["org_ids"] = org_ids
    if len(team_ids) == 1:
        claims["active_team_id"] = team_ids[0]
    if len(org_ids) == 1:
        claims["active_org_id"] = org_ids[0]
    return claims

async def _register_runtime_diag(request: Request, response: Response):
    """Small request-time diagnostic for tests.

    When TEST_MODE is enabled, annotate the response with:
    - X-TLDW-DB: 'postgres' or 'sqlite' based on get_db_pool()
    - X-TLDW-CSRF-Enabled: 'true' or 'false' from runtime settings
    - X-TLDW-Register-Duration-ms: handler duration in ms (set after return)
    """
    test_mode = _is_test_mode()
    if not test_mode:
        # No-op when not in test mode
        return

    start_ts = time.perf_counter()
    try:
        # Resolve DB pool and set backend header
        pool = await get_db_pool()
        db_backend = "postgres" if getattr(pool, "pool", None) is not None else "sqlite"
        response.headers["X-TLDW-DB"] = db_backend

        # Reflect runtime CSRF state exposed via global settings
        csrf_enabled = _csrf_globals.get("CSRF_ENABLED", None)
        response.headers["X-TLDW-CSRF-Enabled"] = "true" if bool(csrf_enabled) else "false"
    finally:
        # Use a background callback via request.state to add duration later
        request.state._register_diag_start = start_ts

def _finalize_register_diag(request: Request, response: Response):
    """Attach handler duration header if start was captured by the diagnostic dep."""
    try:
        start_ts = getattr(request.state, "_register_diag_start", None)
        if start_ts is None:
            return
        dur_ms = int((time.perf_counter() - start_ts) * 1000)
        response.headers["X-TLDW-Register-Duration-ms"] = str(dur_ms)
    except _AUTH_NONCRITICAL_EXCEPTIONS:
        # Diagnostics must never interfere with the response
        pass


#######################################################################################################################
#
# Login Endpoint

async def _login_runtime_diag(request: Request, response: Response):
    """Diagnostics for login (TEST_MODE only): annotate DB backend and CSRF state, capture start time."""
    test_mode = _is_test_mode()
    if not test_mode:
        return
    import time as _t
    try:
        pool = await get_db_pool()
        db_backend = "postgres" if getattr(pool, "pool", None) is not None else "sqlite"
        response.headers["X-TLDW-DB"] = db_backend
        csrf_enabled = _csrf_globals.get("CSRF_ENABLED", None)
        response.headers["X-TLDW-CSRF-Enabled"] = "true" if bool(csrf_enabled) else "false"
    finally:
        request.state._login_diag_start = _t.perf_counter()

def _finalize_login_diag(request: Request, response: Response):
    try:
        import time as _t
        start_ts = getattr(request.state, "_login_diag_start", None)
        if start_ts is None:
            return
        dur_ms = int((_t.perf_counter() - start_ts) * 1000)
        response.headers["X-TLDW-Login-Duration-ms"] = str(dur_ms)
    except _AUTH_NONCRITICAL_EXCEPTIONS:
        pass


# ---------------- Self-service virtual keys (scoped JWT) ----------------


class SelfVirtualKeyRequest(BaseModel):
    ttl_minutes: int = Field(60, ge=1, le=1440)
    scope: str = Field("workflows")
    schedule_id: Optional[str] = None
    allowed_endpoints: Optional[list[str]] = None
    allowed_methods: Optional[list[str]] = None
    allowed_paths: Optional[list[str]] = None
    max_calls: Optional[int] = Field(None, ge=0)
    max_runs: Optional[int] = Field(None, ge=0)
    not_before: Optional[str] = Field(None, description="Optional ISO timestamp when token becomes valid")


@router.post("/virtual-key")
async def mint_self_virtual_key(
    body: SelfVirtualKeyRequest,
    current_user: AuthPrincipal = Depends(get_auth_principal),
    settings: Settings = Depends(get_settings),
):
    """Mint a short-lived, scoped JWT for the current user.

    This is intended for automation and integrations to act on behalf of the
    requesting user with constrained scope, time window, and optional endpoint
    allowlists.
    """
    if settings.AUTH_MODE != "multi_user":
        raise HTTPException(status_code=400, detail="Virtual keys require multi-user mode")
    try:
        user_id = int(current_user.user_id) if current_user.user_id is not None else 0
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Invalid user context") from exc
    if user_id <= 0:
        raise HTTPException(status_code=400, detail="Invalid user context")

    username = str(current_user.username or current_user.email or "user")
    role = _principal_primary_role(current_user)

    try:
        svc = JWTService(settings)
        add_claims: dict[str, Any] = {}
        if body.allowed_endpoints:
            add_claims["allowed_endpoints"] = [str(x) for x in body.allowed_endpoints]
        if body.allowed_methods:
            add_claims["allowed_methods"] = [str(x).upper() for x in body.allowed_methods]
        if body.allowed_paths:
            add_claims["allowed_paths"] = [str(x) for x in body.allowed_paths]
        if body.max_calls is not None:
            add_claims["max_calls"] = int(body.max_calls)
        if body.max_runs is not None:
            add_claims["max_runs"] = int(body.max_runs)
        scope_claims = await _build_scope_claims(user_id)
        if scope_claims:
            add_claims.update(scope_claims)
        if body.not_before:
            # Store as standard JWT 'nbf' if parseable; otherwise ignore
            try:
                from datetime import datetime
                nbf_dt = datetime.fromisoformat(str(body.not_before).replace("Z", "+00:00"))
                add_claims["nbf"] = int(nbf_dt.timestamp())
            except _AUTH_NONCRITICAL_EXCEPTIONS:
                pass
        token = svc.create_virtual_access_token(
            user_id=user_id,
            username=username,
            role=role,
            scope=str(body.scope or "workflows"),
            ttl_minutes=int(body.ttl_minutes),
            schedule_id=(str(body.schedule_id) if body.schedule_id else None),
            additional_claims=add_claims or None,
        )
        from datetime import datetime, timedelta
        exp = datetime.utcnow() + timedelta(minutes=int(body.ttl_minutes))
        return {
            "token": token,
            "expires_at": exp.isoformat(),
            "scope": str(body.scope or "workflows"),
            "schedule_id": (str(body.schedule_id) if body.schedule_id else None),
        }
    except HTTPException:
        raise
    except _AUTH_NONCRITICAL_EXCEPTIONS as e:
        if settings.PII_REDACT_LOGS:
            logger.exception("Failed to mint self virtual key [redacted]")
        else:
            logger.exception(
                "Failed to mint self virtual key for user_id={} scope={} ttl_minutes={} error_type={}",
                user_id,
                body.scope,
                body.ttl_minutes,
                type(e).__name__,
            )
        raise HTTPException(status_code=500, detail="Failed to mint token") from e

@router.post(
    "/login",
    response_model=TokenResponse | MFAChallengeResponse,
    dependencies=[Depends(check_auth_rate_limit)],
    responses={status.HTTP_202_ACCEPTED: {"model": MFAChallengeResponse}},
)
async def login(
    request: Request,
    response: Response,
    _diag=Depends(_login_runtime_diag),  # noqa: B008
    form_data: OAuth2PasswordRequestForm = Depends(),
    db=Depends(get_db_transaction),
    jwt_service: JWTService = Depends(get_jwt_service_dep),
    password_service: PasswordService = Depends(get_password_service_dep),
    session_manager: SessionManager = Depends(get_session_manager_dep),
    rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
    settings: Settings = Depends(get_settings)
) -> TokenResponse | MFAChallengeResponse:
    """
    OAuth2 compatible login endpoint

    Authenticates user with username/password and returns JWT tokens.
    Compatible with OAuth2PasswordRequestForm for standard OAuth2 clients.

    Args:
        form_data: OAuth2 form with username and password

    Returns:
        TokenResponse with access_token, refresh_token, and expiry

    Raises:
        HTTPException: 401 if credentials invalid, 403 if account inactive
    """
    start_time = time.perf_counter()
    log_counter("auth_login_attempt")
    auth_gov = await get_auth_governor()
    try:
        # Get client info
        client_ip = _auth_request_client_ip(request)
        user_agent = request.headers.get("User-Agent", "Unknown")
        # PII-aware logging
        if settings.PII_REDACT_LOGS:
            logger.info("Login attempt [redacted]")
        else:
            logger.info(f"Login attempt for user: {form_data.username} from IP: {client_ip}")

        # Check if IP is locked out (only when rate limiting is enabled)
        is_locked = False
        lockout_expires = None
        if getattr(rate_limiter, 'enabled', False):
            is_locked, lockout_expires = await auth_gov.check_lockout(
                client_ip,
                attempt_type="login",
                rate_limiter=rate_limiter,
            )
        if is_locked:
            logger.warning(f"Login attempt from locked IP: {client_ip}")
            log_counter("auth_login_locked_ip")
            retry_after_seconds = 900
            if isinstance(lockout_expires, datetime):
                try:
                    from datetime import timezone as _tz

                    now = datetime.now(lockout_expires.tzinfo or _tz.utc)
                    retry_after_seconds = max(0, int((lockout_expires - now).total_seconds()))
                except _AUTH_NONCRITICAL_EXCEPTIONS as exc:
                    # Fallback to default retry_after_seconds, but keep it observable.
                    logger.debug(
                        f"login: failed to compute lockout expiry; using default Retry-After: {exc}"
                    )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many failed login attempts. Please try again later.",
                headers={"Retry-After": str(retry_after_seconds)}
            )

        # Sanitize input (lightweight). For login, avoid strict validation to not block
        # legitimate existing accounts (e.g., reserved usernames like 'admin').
        login_identifier = form_data.username.strip()

        # Helper to attempt audit logging without hard dependency (safe no-op in tests)
        async def _safe_audit_log_login(user_id: int, username: str, ip: str, ua: str, success: bool):
            try:
                svc = await get_or_create_audit_service_for_user_id(user_id)
                await svc.log_login(
                    user_id=user_id,
                    username=username,
                    ip_address=ip,
                    user_agent=ua,
                    success=success,
                )
                # Persist immediately for observability (tests/admin tools) without relying on background loops.
                flush_on_login = _env_flag_enabled("AUDIT_FLUSH_ON_LOGIN")
                test_mode = _is_test_mode()
                if flush_on_login or test_mode:
                    await svc.flush()
            except _AUTH_NONCRITICAL_EXCEPTIONS as exc:
                # Never block auth on audit issues
                logger.debug(
                    "Login audit failed for user_id={}: {}",
                    user_id,
                    exc,
                    exc_info=True,
                )

        # Fetch user from database using sanitized identifier
        user = await fetch_user_by_login_identifier(db, login_identifier)

        # Check if user exists
        if not user:
            # Log failed attempt
            if settings.PII_REDACT_LOGS:
                logger.warning("Failed login: user not found [redacted]")
            else:
                logger.warning(f"Failed login: User not found - {login_identifier}")

            # Track failed attempt by IP (only when rate limiting is enabled)
            if getattr(rate_limiter, 'enabled', False):
                await auth_gov.record_auth_failure(
                    identifier=client_ip,
                    attempt_type="login",
                    rate_limiter=rate_limiter,
                )

            log_counter("auth_login_user_not_found")
            # finalize diag headers in test mode for visibility
            with contextlib.suppress(_AUTH_NONCRITICAL_EXCEPTIONS):
                _finalize_login_diag(request, response)
            extra_headers = {"WWW-Authenticate": "Bearer"}
            if _is_test_mode():
                extra_headers["X-TLDW-Login-Reason"] = "user-not-found"
                # Stage marker to aid triage in test runs
                extra_headers["X-TLDW-Login-Stage"] = "user_fetch"
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers=extra_headers
            )

        # Convert to dict if needed; guard against stray AsyncMock/awaitables in tests
        import inspect as _inspect
        if _inspect.isawaitable(user):
            try:
                user = await user  # resolve unexpected awaitables from mocks
            except _AUTH_NONCRITICAL_EXCEPTIONS:
                pass
        # user already normalized to dict in service

        # Enforce username lockout before password verification (true lockout).
        if getattr(rate_limiter, 'enabled', False):
            user_locked, user_lockout_expires = await auth_gov.check_lockout(
                user['username'],
                attempt_type="login",
                rate_limiter=rate_limiter,
            )
            if user_locked:
                logger.warning(f"Login attempt for locked account: {user['username']}")
                log_counter("auth_login_locked_user")
                retry_after_seconds = 900
                if isinstance(user_lockout_expires, datetime):
                    try:
                        from datetime import timezone as _tz

                        now = datetime.now(user_lockout_expires.tzinfo or _tz.utc)
                        retry_after_seconds = max(0, int((user_lockout_expires - now).total_seconds()))
                    except _AUTH_NONCRITICAL_EXCEPTIONS as exc:
                        logger.debug(
                            f"login: failed to compute user lockout expiry; using default Retry-After: {exc}"
                        )
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Too many failed login attempts. Account temporarily locked.",
                    headers={"Retry-After": str(retry_after_seconds)},
                )

        # Verify password
        is_test_mode = _is_test_mode()
        is_valid, needs_rehash = password_service.verify_password(form_data.password, user['password_hash'])
        # In TEST_MODE, double-check with a fresh PasswordService to rule out stale settings
        if not is_valid and is_test_mode:
            try:
                fresh_ps = PasswordService(settings)
                is_valid2, needs_rehash2 = fresh_ps.verify_password(form_data.password, user['password_hash'])
                if is_valid2:
                    is_valid = True
                    needs_rehash = needs_rehash2
                    password_service = fresh_ps
                    # annotate header to indicate re-verify succeeded
                    with contextlib.suppress(_AUTH_NONCRITICAL_EXCEPTIONS):
                        response.headers["X-TLDW-Login-Reverify"] = "ok"
            except _AUTH_NONCRITICAL_EXCEPTIONS:
                # fall back to original result
                pass
        if not is_valid:
            # Log failed attempt
            if settings.PII_REDACT_LOGS:
                logger.warning("Failed login: invalid password [redacted]")
            else:
                logger.warning(f"Failed login: Invalid password for user {user['username']}")

            # Audit log failed login
            await _safe_audit_log_login(
                user_id=user['id'],
                username=user['username'],
                ip=client_ip,
                ua=user_agent,
                success=False,
            )

            # Track failed attempt by IP and username
            ip_result = {"is_locked": False, "remaining_attempts": 5}
            user_result = {"is_locked": False, "remaining_attempts": 5}
            if getattr(rate_limiter, 'enabled', False):
                ip_result = await auth_gov.record_auth_failure(
                    identifier=client_ip,
                    attempt_type="login",
                    rate_limiter=rate_limiter,
                )
                user_result = await auth_gov.record_auth_failure(
                    identifier=user['username'],
                    attempt_type="login",
                    rate_limiter=rate_limiter,
                )

            # Provide informative error if locked out
            if ip_result['is_locked'] or user_result['is_locked']:
                log_counter("auth_login_locked_user")
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Too many failed login attempts. Account temporarily locked.",
                    headers={"Retry-After": "900"}  # 15 minutes
                )

            # Otherwise generic error
            remaining = min(ip_result.get('remaining_attempts', 5), user_result.get('remaining_attempts', 5))
            logger.info(f"Remaining login attempts: {remaining}")

            log_counter("auth_login_invalid_password")
            with contextlib.suppress(_AUTH_NONCRITICAL_EXCEPTIONS):
                _finalize_login_diag(request, response)
            extra_headers = {"WWW-Authenticate": "Bearer"}
            if _is_test_mode():
                extra_headers["X-TLDW-Login-Reason"] = "invalid-password"
                # Stage marker to aid triage in test runs
                extra_headers["X-TLDW-Login-Stage"] = "verify_password"
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers=extra_headers
            )

        # Check if account is active
        if not user['is_active']:
            logger.warning(f"Failed login: Inactive account - {user['username']}")
            log_counter("auth_login_inactive_account")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account is inactive. Please contact support."
            )

        # If password needs rehashing, update it
        if needs_rehash:
            new_hash = password_service.hash_password(form_data.password)
            await update_user_password_hash(db, int(user['id']), new_hash)
            logger.info(f"Updated password hash for user {user['username']} with new parameters")

        # Attach user_id for downstream middleware (e.g., CSRF binding) on successful auth.
        try:
            request.state.user_id = int(user["id"])
        except _AUTH_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug("Failed to set request.state.user_id during login: {}", exc)

        # Determine whether MFA is required (multi-user + PostgreSQL only).
        mfa_required = False
        if settings.AUTH_MODE == "multi_user" and await _is_mfa_backend_supported():
            try:
                mfa_service = _get_mfa_service()
                mfa_status = await mfa_service.get_user_mfa_status(int(user["id"]))
                mfa_required = bool(mfa_status.get("enabled"))
            except _AUTH_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug(
                    "MFA status lookup failed during login; treating as disabled: {}",
                    exc,
                )

        # Create session first to get session_id
        user_agent = request.headers.get("User-Agent", "Unknown")

        # Generate tokens based on auth mode
        if settings.AUTH_MODE == "single_user":
            if _is_enterprise_admin_ui_mode():
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Enterprise Admin UI requires multi-user authentication; single-user login is disabled.",
                )
            # For single-user mode, return the configured API key as the access token.
            single_user_key = (
                settings.SINGLE_USER_API_KEY
                or os.getenv("SINGLE_USER_API_KEY")
                or os.getenv("API_KEY")
            )
            if not single_user_key:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Single-user API key is not configured",
                )
            access_token = single_user_key
            refresh_token = single_user_key

            # Create session with tokens
            await session_manager.create_session(
                user_id=user['id'],
                access_token=access_token,
                refresh_token=refresh_token,
                ip_address=client_ip,
                user_agent=user_agent
            )
        else:
            # For multi-user mode, create a temporary session first
            # Use unique placeholders to avoid duplicate token hash constraints
            temp_access = f"temp_access_{secrets.token_urlsafe(16)}"
            temp_refresh = f"temp_refresh_{secrets.token_urlsafe(16)}"
            mfa_expires_at = None
            mfa_ttl_seconds = None
            if mfa_required:
                mfa_ttl_seconds = _get_mfa_login_ttl_seconds()
                mfa_expires_at = datetime.now(timezone.utc) + timedelta(seconds=mfa_ttl_seconds)

            temp_session_info = await session_manager.create_session(
                user_id=user['id'],
                access_token=temp_access,  # Will update with actual token
                refresh_token=temp_refresh,  # Will update with actual token
                ip_address=client_ip,
                user_agent=user_agent,
                expires_at_override=mfa_expires_at,
                refresh_expires_at_override=mfa_expires_at,
            )

            session_id = temp_session_info['session_id']

            if mfa_required:
                await _ensure_mfa_cache_available(session_manager, settings)
                session_token = secrets.token_urlsafe(32)
                ttl_seconds = mfa_ttl_seconds or _get_mfa_login_ttl_seconds()
                payload = {
                    "user_id": int(user["id"]),
                    "session_id": int(session_id),
                }
                try:
                    await session_manager.store_ephemeral_value(
                        _mfa_login_cache_key(session_token),
                        json.dumps(payload),
                        ttl_seconds,
                    )
                except _AUTH_NONCRITICAL_EXCEPTIONS as exc:
                    logger.error("Failed to cache MFA login session: {}", exc)
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="Failed to initiate MFA login",
                    ) from exc
                response.status_code = status.HTTP_202_ACCEPTED
                log_counter("auth_login_mfa_required")
                with contextlib.suppress(_AUTH_NONCRITICAL_EXCEPTIONS):
                    _finalize_login_diag(request, response)
                return MFAChallengeResponse(
                    session_token=session_token,
                    mfa_required=True,
                    expires_in=ttl_seconds,
                    message="MFA required. Submit your TOTP or backup code.",
                )

            # Ensure org membership before issuing tokens (so claims include org_id)
            if settings.AUTH_MODE == "multi_user":
                await _ensure_user_org_membership(int(user["id"]), user.get("username"))

            # Create JWT tokens with session_id
            scope_claims = await _build_scope_claims(int(user["id"]))
            add_claims = dict(scope_claims)
            add_claims["session_id"] = session_id
            access_token = jwt_service.create_access_token(
                user_id=user['id'],
                username=user['username'],
                role=user['role'],
                additional_claims=add_claims
            )

            refresh_token = jwt_service.create_refresh_token(
                user_id=user['id'],
                username=user['username'],
                additional_claims=add_claims
            )

            # Update session with actual tokens
            await session_manager.update_session_tokens(
                session_id=session_id,
                access_token=access_token,
                refresh_token=refresh_token
            )


        # Update last login time
        await update_user_last_login(db, int(user['id']), datetime.utcnow())

        # Log successful login
        if settings.PII_REDACT_LOGS:
            logger.info("Successful login [redacted]")
        else:
            logger.info(f"Successful login for user: {user['username']} (ID: {user['id']})")

        # Reset failed login attempts on successful login
        if getattr(rate_limiter, 'enabled', False):
            try:
                await rate_limiter.reset_failed_attempts(client_ip, "login")
                await rate_limiter.reset_failed_attempts(user['username'], "login")
            except _AUTH_NONCRITICAL_EXCEPTIONS as rl_exc:
                # Guardrails must not break successful logins; log and continue.
                logger.debug(f"rate_limiter.reset_failed_attempts failed: {rl_exc}")

        # Audit log successful login
        await _safe_audit_log_login(
            user_id=user['id'],
            username=user['username'],
            ip=client_ip,
            ua=user_agent,
            success=True,
        )

        log_counter("auth_login_success")
        log_histogram("auth_login_duration", time.perf_counter() - start_time)
        result = TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
        )
        with contextlib.suppress(_AUTH_NONCRITICAL_EXCEPTIONS):
            _finalize_login_diag(request, response)
        return result

    except HTTPException:
        log_counter("auth_login_http_error")
        log_histogram("auth_login_duration", time.perf_counter() - start_time)
        raise
    except _AUTH_NONCRITICAL_EXCEPTIONS:
        logger.exception("Login error")
        log_counter("auth_login_unexpected_error")
        log_histogram("auth_login_duration", time.perf_counter() - start_time)
        if _is_test_mode():  # expose details in tests
            try:
                response.headers["X-TLDW-Login-Error"] = "internal-error"
                _finalize_login_diag(request, response)
            except _AUTH_NONCRITICAL_EXCEPTIONS:
                pass
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An error occurred during login"
            ) from None
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during login"
        ) from None


#######################################################################################################################
#
# Logout Endpoint

@router.post("/logout", response_model=MessageResponse)
async def logout(
    request: Request,
    data: Optional[LogoutRequest] = None,
    current_user: AuthPrincipal = Depends(get_auth_principal),
    session_manager: SessionManager = Depends(get_session_manager_dep),
    jwt_service: JWTService = Depends(get_jwt_service_dep),
) -> MessageResponse:
    """
    Logout current session or all sessions.

    Revokes the current token and optionally all user tokens/sessions.
    """
    start_time = time.perf_counter()
    log_counter("auth_logout_attempt")
    try:
        all_devices = bool(getattr(data, "all_devices", False)) if data is not None else False

        def _user_id_from(obj: Any) -> int:
            if isinstance(obj, AuthPrincipal):
                return int(obj.user_id or 0)
            if isinstance(obj, dict):
                return int(obj.get("id") or obj.get("user_id") or 0)
            return int(getattr(obj, "user_id", 0) or getattr(obj, "id", 0) or 0)

        user_id = _user_id_from(current_user)
        if user_id <= 0:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid user context")

        if all_devices:
            try:
                count = int(
                    await session_manager.revoke_all_user_sessions(
                        user_id=user_id,
                        reason="User requested logout from all devices",
                    )
                )
            except _AUTH_NONCRITICAL_EXCEPTIONS as revoke_exc:
                logger.error(f"Failed to revoke user sessions during logout-all: {revoke_exc}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to revoke sessions during logout",
                ) from revoke_exc
            message = f"Logged out from {count} device(s)"
        else:
            # Revoke current access token and session.
            auth_header = request.headers.get("Authorization", "") if request is not None else ""
            token = _extract_bearer_token(auth_header)
            if not token:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Current-session logout requires a bearer token. Use all_devices=true to revoke all sessions.",
                )
            blacklist = get_token_blacklist()
            payload = {}
            try:
                # NOTE: Using sync verify_token() here is acceptable - we're extracting
                # claims for revocation cleanup, not making authorization decisions.
                # The user has already been authenticated via the claim-first dependency.
                payload = jwt_service.verify_token(token)
            except _AUTH_NONCRITICAL_EXCEPTIONS:
                payload = {}

            try:
                jti = jwt_service.extract_jti(token)
            except _AUTH_NONCRITICAL_EXCEPTIONS:
                jti = None
            if jti:
                try:
                    exp = payload.get("exp")
                    expires_at = datetime.utcfromtimestamp(exp) if exp else datetime.utcnow()
                    await blacklist.revoke_token(
                        jti=jti,
                        expires_at=expires_at,
                        user_id=user_id,
                        token_type="access",
                        reason="User logout",
                    )
                except _AUTH_NONCRITICAL_EXCEPTIONS as revoke_exc:
                    logger.error(f"Failed to revoke access token for logout: {revoke_exc}")
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="Failed to revoke access token during logout",
                    ) from revoke_exc
            session_id = payload.get("session_id")
            if session_id is not None:
                try:
                    await session_manager.revoke_session(
                        session_id=session_id,
                        revoked_by=user_id,
                        reason="User logout",
                    )
                except _AUTH_NONCRITICAL_EXCEPTIONS as cleanup_exc:
                    logger.error(f"Failed to revoke session {session_id} during logout: {cleanup_exc}")
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="Failed to revoke session during logout",
                    ) from cleanup_exc
            else:
                # Fallback to full session revoke if the token lacks a session id.
                try:
                    await session_manager.revoke_all_user_sessions(user_id=user_id)
                except _AUTH_NONCRITICAL_EXCEPTIONS as cleanup_exc:
                    logger.error(f"Failed to revoke user sessions during logout: {cleanup_exc}")
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="Failed to revoke sessions during logout",
                    ) from cleanup_exc

            message = "Successfully logged out"

        # PII-aware logging
        try:
            _settings = get_settings()
        except _AUTH_NONCRITICAL_EXCEPTIONS:
            _settings = None
        if _settings and getattr(_settings, 'PII_REDACT_LOGS', False):
            logger.info("User logged out [redacted]")
        else:
            logger.info(f"User logged out: {user_id}")

        log_counter("auth_logout_success")
        log_histogram("auth_logout_duration", time.perf_counter() - start_time)
        return MessageResponse(message=message, details={"user_id": user_id})

    except HTTPException:
        log_counter("auth_logout_error")
        log_histogram("auth_logout_duration", time.perf_counter() - start_time)
        raise
    except _AUTH_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Logout error: {e}")
        log_counter("auth_logout_error")
        log_histogram("auth_logout_duration", time.perf_counter() - start_time)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to complete logout",
        ) from e


#######################################################################################################################
#
# Session Management (auth-scoped)

@router.get("/sessions", response_model=list[SessionResponse])
async def list_user_sessions(
    current_user: AuthPrincipal = Depends(get_auth_principal),
    session_manager: SessionManager = Depends(get_session_manager_dep)
) -> list[SessionResponse]:
    """
    List all active sessions for the current user.
    """
    try:
        user_id = _current_user_id(current_user)
        if user_id is None or user_id <= 0:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials",
            )
        sessions = await session_manager.get_user_sessions(user_id)

        return [
            SessionResponse(
                id=session['id'],
                ip_address=session.get('ip_address'),
                user_agent=session.get('user_agent'),
                created_at=session['created_at'],
                last_activity=session['last_activity'],
                expires_at=session['expires_at']
            )
            for session in sessions
        ]

    except _AUTH_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Failed to list user sessions: {e}")
        # In test mode, surface the underlying error to aid debugging
        if _is_test_mode():
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to retrieve sessions: {e}"
            ) from e
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve sessions"
        ) from e


@router.delete("/sessions/{session_id}", response_model=MessageResponse)
async def revoke_session(
    session_id: int,
    current_user: AuthPrincipal = Depends(get_auth_principal),
    session_manager: SessionManager = Depends(get_session_manager_dep)
) -> MessageResponse:
    """
    Revoke a specific session for the current user.
    """
    try:
        user_id = _current_user_id(current_user)
        if user_id is None or user_id <= 0:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials",
            )
        username = _current_user_username(current_user)

        # Get session to verify ownership
        sessions = await session_manager.get_user_sessions(user_id)
        session_ids = [s['id'] for s in sessions]

        if session_id not in session_ids:
            # Return success for idempotency - session is already not active
            logger.info(
                f"Session {session_id} not found for user {user_id} - treating as already revoked"
            )
            return MessageResponse(
                message="Session revoked successfully",
                details={"session_id": session_id, "note": "Session was already inactive or did not exist"}
            )

        # Revoke the session
        await session_manager.revoke_session(
            session_id,
            revoked_by=user_id,
            reason="User requested revocation"
        )

        logger.info(f"User {username} revoked session {session_id}")

        return MessageResponse(
            message="Session revoked successfully",
            details={"session_id": session_id}
        )

    except HTTPException:
        raise
    except SessionError as e:
        logger.error(f"Failed to revoke session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to revoke session"
        ) from e
    except _AUTH_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Unexpected error revoking session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while revoking the session"
        ) from e


@router.post("/sessions/revoke-all", response_model=MessageResponse)
async def revoke_all_sessions(
    current_user: AuthPrincipal = Depends(get_auth_principal),
    session_manager: SessionManager = Depends(get_session_manager_dep)
) -> MessageResponse:
    """
    Revoke all sessions for the current user.
    """
    try:
        user_id = _current_user_id(current_user)
        if user_id is None or user_id <= 0:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials",
            )
        username = _current_user_username(current_user)

        count = await session_manager.revoke_all_user_sessions(
            user_id=user_id,
            reason="User requested logout from all devices"
        )
        revoked_count = int(count)

        logger.info(f"User {username} revoked all {revoked_count} sessions")

        return MessageResponse(
            message=f"Successfully revoked {revoked_count} sessions",
            details={"sessions_revoked": revoked_count}
        )

    except _AUTH_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Failed to revoke all sessions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to revoke sessions"
        ) from e


#######################################################################################################################
#
# Token Refresh Endpoint

@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    payload: RefreshTokenRequest,
    response: Response,
    http_request: Request,
    jwt_service: JWTService = Depends(get_jwt_service_dep),
    session_manager: SessionManager = Depends(get_session_manager_dep),
    db=Depends(get_db_transaction),
    settings: Settings = Depends(get_settings)
) -> TokenResponse:
    """
    Refresh access token using refresh token

    Args:
        payload: RefreshTokenRequest with refresh token
        http_request: FastAPI request (for IP allowlist and CSRF binding)

    Returns:
        TokenResponse with new access_token

    Raises:
        HTTPException: 401 if refresh token invalid or expired
    """
    start_time = time.perf_counter()
    log_counter("auth_refresh_attempt")
    try:
        # TEST_MODE diagnostics (set DB and CSRF headers for easier triage)
        if _is_test_mode():
            try:
                from tldw_Server_API.app.core.AuthNZ.database import get_db_pool as _get_pool
                pool = await _get_pool()
                db_backend = "postgres" if getattr(pool, "pool", None) is not None else "sqlite"
                from tldw_Server_API.app.core.AuthNZ.csrf_protection import global_settings as _csrf_globals
                response.headers["X-TLDW-DB"] = db_backend
                response.headers["X-TLDW-CSRF-Enabled"] = "true" if bool(_csrf_globals.get("CSRF_ENABLED", None)) else "false"
            except _AUTH_NONCRITICAL_EXCEPTIONS:
                pass

        # Handle based on auth mode
        if settings.AUTH_MODE == "single_user":
            # Simple token validation for single-user mode
            single_user_key = (
                settings.SINGLE_USER_API_KEY
                or os.getenv("SINGLE_USER_API_KEY")
                or os.getenv("API_KEY")
            )
            if single_user_key and payload.refresh_token == single_user_key:
                client_ip = resolve_client_ip(http_request, settings)
                if not is_single_user_ip_allowed(client_ip, settings):
                    raise InvalidTokenError("Invalid refresh token format")
                user_id = int(getattr(settings, "SINGLE_USER_FIXED_ID", 1))
            else:
                if _is_test_mode():
                    try:
                        response.headers["X-TLDW-Refresh-Stage"] = "validate"
                        response.headers["X-TLDW-Refresh-Reason"] = "invalid-format"
                    except _AUTH_NONCRITICAL_EXCEPTIONS:
                        pass
                raise InvalidTokenError("Invalid refresh token format")
        else:
            # JWT validation for multi-user mode
            try:
                token_payload = jwt_service.decode_refresh_token(payload.refresh_token)
            except _AUTH_NONCRITICAL_EXCEPTIONS as _e:
                if _is_test_mode():
                    try:
                        response.headers["X-TLDW-Refresh-Stage"] = "decode"
                        response.headers["X-TLDW-Refresh-Reason"] = f"invalid-token:{type(_e).__name__}"
                    except _AUTH_NONCRITICAL_EXCEPTIONS:
                        pass
                raise

            # Check if token is blacklisted
            if await session_manager.is_token_blacklisted(payload.refresh_token, token_payload.get("jti")):
                if _is_test_mode():
                    try:
                        response.headers["X-TLDW-Refresh-Stage"] = "blacklist"
                        response.headers["X-TLDW-Refresh-Reason"] = "revoked"
                    except _AUTH_NONCRITICAL_EXCEPTIONS:
                        pass
                raise InvalidTokenError("Refresh token has been revoked")

            # JWT standard uses 'sub' for subject (user ID)
            user_id = token_payload.get("sub") or token_payload.get("user_id")
            if not user_id:
                if _is_test_mode():
                    try:
                        response.headers["X-TLDW-Refresh-Stage"] = "decode"
                        response.headers["X-TLDW-Refresh-Reason"] = "missing-user-id"
                    except _AUTH_NONCRITICAL_EXCEPTIONS:
                        pass
                raise InvalidTokenError("Invalid refresh token payload")

            # Convert to int if it's a string
            try:
                user_id = int(user_id)
            except (ValueError, TypeError):
                if _is_test_mode():
                    try:
                        response.headers["X-TLDW-Refresh-Stage"] = "decode"
                        response.headers["X-TLDW-Refresh-Reason"] = "invalid-user-id"
                    except _AUTH_NONCRITICAL_EXCEPTIONS:
                        pass
                raise InvalidTokenError("Invalid user ID in refresh token") from None
            # Capture session association when present
            session_id = token_payload.get("session_id")

        # Fetch user
        user = await fetch_active_user_by_id(db, user_id)

        if not user:
            if _is_test_mode():
                try:
                    response.headers["X-TLDW-Refresh-Stage"] = "fetch-user"
                    response.headers["X-TLDW-Refresh-Reason"] = "user-not-found"
                except _AUTH_NONCRITICAL_EXCEPTIONS:
                    pass
            raise InvalidTokenError("Invalid or expired refresh token")

        # Convert to dict
        if not isinstance(user, dict):
            if hasattr(user, 'keys'):
                user = dict(user)
            else:
                columns = ['id', 'uuid', 'username', 'email', 'password_hash', 'role']
                user = dict(zip(columns[:len(user)], user))

        # Attach user_id for middleware that binds CSRF tokens on refresh responses.
        try:
            http_request.state.user_id = int(user.get("id")) if isinstance(user, dict) else None
        except _AUTH_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug("Failed to set request.state.user_id during refresh: {}", exc)

        # Generate new tokens based on auth mode and session linkage
        if settings.AUTH_MODE == "single_user":
            single_user_key = (
                settings.SINGLE_USER_API_KEY
                or os.getenv("SINGLE_USER_API_KEY")
                or os.getenv("API_KEY")
            )
            if not single_user_key:
                raise InvalidTokenError("Single-user API key is not configured")
            new_access_token = single_user_key
            new_refresh_token = single_user_key
        else:
            # Access token always refreshed; preserve session_id claim when available
            scope_claims = await _build_scope_claims(int(user["id"]))
            add_claims = dict(scope_claims)
            if session_id:
                add_claims["session_id"] = session_id
            if not add_claims:
                add_claims = None
            new_access_token = jwt_service.create_access_token(
                user_id=user['id'],
                username=user['username'],
                role=user['role'],
                additional_claims=add_claims
            )
            # Rotate refresh token if enabled
            new_refresh_token = payload.refresh_token
            if getattr(settings, "ROTATE_REFRESH_TOKENS", False):
                new_refresh_token = jwt_service.create_refresh_token(
                    user_id=user['id'],
                    username=user['username'],
                    additional_claims=add_claims
                )
            # Update backing session to reflect new token(s)
            try:
                await session_manager.refresh_session(
                    refresh_token=payload.refresh_token,
                    new_access_token=new_access_token,
                    new_refresh_token=(new_refresh_token if new_refresh_token != payload.refresh_token else None)
                )
            except _AUTH_NONCRITICAL_EXCEPTIONS as _sess_e:
                # Treat missing/invalid session mapping as invalid token usage
                if _is_test_mode():
                    try:
                        response.headers.setdefault("X-TLDW-Refresh-Stage", "session")
                        response.headers.setdefault("X-TLDW-Refresh-Reason", f"session-error:{type(_sess_e).__name__}")
                    except _AUTH_NONCRITICAL_EXCEPTIONS:
                        pass
                raise InvalidTokenError("Invalid or expired session for refresh token") from _sess_e

            # Always blacklist the prior refresh token's JTI to prevent reuse
            try:
                from datetime import datetime as _dt

                from tldw_Server_API.app.core.AuthNZ.token_blacklist import get_token_blacklist as _get_bl
                old_jti = token_payload.get("jti") if isinstance(token_payload, dict) else None
                old_exp = token_payload.get("exp") if isinstance(token_payload, dict) else None
                if old_jti and isinstance(old_exp, (int, float)) and new_refresh_token != payload.refresh_token:
                    expires_at = _dt.utcfromtimestamp(old_exp)
                    bl = _get_bl()
                    # Best-effort revoke; do not fail refresh if blacklist write fails
                    await bl.revoke_token(
                        jti=old_jti,
                        expires_at=expires_at,
                        user_id=int(user['id']),
                        token_type="refresh",
                        reason="refresh-rotated",
                        revoked_by=None,
                        ip_address=(_auth_request_client_ip(http_request) if http_request else None),
                    )
            except _AUTH_NONCRITICAL_EXCEPTIONS as _bl_e:
                with contextlib.suppress(_AUTH_NONCRITICAL_EXCEPTIONS):
                    logger.debug(f"Refresh: blacklist prior token best-effort failed: {_bl_e}")

        if settings.PII_REDACT_LOGS:
            logger.info("Token refreshed [redacted]")
        else:
            logger.info(f"Token refreshed for user: {user['username']} (ID: {user['id']})")

        log_counter("auth_refresh_success")
        log_histogram("auth_refresh_duration", time.perf_counter() - start_time)
        # TEST_MODE: include simple duration metric header (non-breaking)
        if _is_test_mode():
            with contextlib.suppress(_AUTH_NONCRITICAL_EXCEPTIONS):
                response.headers["X-TLDW-Refresh-Duration-ms"] = str(int((time.perf_counter() - start_time) * 1000))
        return TokenResponse(
            access_token=new_access_token,
            refresh_token=new_refresh_token,
            token_type="bearer",
            expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
        )

    except (TokenExpiredError, InvalidTokenError) as e:
        logger.warning(f"Token refresh failed: {e}")
        log_counter("auth_refresh_token_error", labels={"type": type(e).__name__})
        log_histogram("auth_refresh_duration", time.perf_counter() - start_time)
        if _is_test_mode():
            try:
                response.headers.setdefault("X-TLDW-Refresh-Stage", "error")
                response.headers.setdefault("X-TLDW-Refresh-Reason", type(e).__name__)
            except _AUTH_NONCRITICAL_EXCEPTIONS:
                pass
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
            headers={"WWW-Authenticate": "Bearer"}
        ) from e
    except _AUTH_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Token refresh error: {e}")
        log_counter("auth_refresh_unexpected_error")
        log_histogram("auth_refresh_duration", time.perf_counter() - start_time)
        if _is_test_mode():
            try:
                response.headers["X-TLDW-Refresh-Stage"] = "unexpected"
                response.headers["X-TLDW-Refresh-Reason"] = "internal-error"
                response.headers["X-TLDW-Refresh-Duration-ms"] = str(int((time.perf_counter() - start_time) * 1000))
            except _AUTH_NONCRITICAL_EXCEPTIONS:
                pass
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during token refresh"
        ) from e


#######################################################################################################################
#
# Password Reset Endpoints

@router.post("/forgot-password", status_code=status.HTTP_200_OK)
async def forgot_password(
    request: Request,
    data: ForgotPasswordRequest,
    db=Depends(get_db_transaction),
    jwt_service: JWTService = Depends(get_jwt_service_dep),
) -> dict[str, str]:
    """
    Request password reset email.

    Sends a password reset link to the user's email if the account exists.
    Returns success even if email doesn't exist (security best practice).
    """
    try:
        # Get client info
        client_ip = _auth_request_client_ip(request)
        allowed, _retry_after = await _reserve_auth_rg_requests(
            request,
            policy_id="authnz.forgot_password",
            entity=f"ip:{client_ip}",
            tags={"auth_endpoint": "forgot_password"},
        )
        if not allowed:
            return {"message": "If the email exists, a reset link has been sent"}

        # Validate email format
        validator = get_input_validator()
        is_valid, _error_msg = validator.validate_email(data.email)
        if not is_valid:
            # Return success anyway for security
            return {"message": "If the email exists, a reset link has been sent"}

        # Check if user exists
        user = await _fetch_user_by_email_for_password_reset(db, data.email)

        if user and user["is_active"]:
            # Generate reset token
            reset_token = jwt_service.create_password_reset_token(
                user_id=user["id"],
                email=user["email"],
                expires_in_hours=1,
            )

            # Store token in database for validation
            await _store_password_reset_token(
                db,
                user_id=int(user["id"]),
                token_hash=jwt_service.hash_password_reset_token(reset_token),
                expires_at=datetime.utcnow() + timedelta(hours=1),
                ip_address=client_ip,
            )

            # Send email
            email_service = _get_email_service()
            await email_service.send_password_reset_email(
                to_email=user["email"],
                username=user["username"],
                reset_token=reset_token,
                ip_address=client_ip,
            )

            logger.info(f"Password reset requested for user {user['id']} from IP {client_ip}")

        # Always return success for security
        return {"message": "If the email exists, a reset link has been sent"}

    except _AUTH_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Password reset error: {e}")
        # Still return success for security
        return {"message": "If the email exists, a reset link has been sent"}


@router.post("/reset-password", status_code=status.HTTP_200_OK)
async def reset_password(
    data: ResetPasswordRequest,
    request: Request,
    db=Depends(get_db_transaction),
    jwt_service: JWTService = Depends(get_jwt_service_dep),
    password_service: PasswordService = Depends(get_password_service_dep),
) -> dict[str, str]:
    """
    Reset password with valid token.

    Validates the reset token and updates the user's password.
    """
    try:
        ip_addr = _auth_request_client_ip(request)
        allowed, retry_after = await _reserve_auth_rg_requests(
            request,
            policy_id="authnz.reset_password",
            entity=f"ip:{ip_addr}",
            tags={"auth_endpoint": "reset_password"},
        )
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many password reset attempts. Please try again later.",
                headers={"Retry-After": str(int(retry_after or 1))},
            )
        # Verify token cryptographically
        # NOTE: Using sync verify_token() is acceptable for password_reset tokens because:
        # 1. These are single-use tokens with additional database validation below
        # 2. The token hash is verified against password_reset_tokens table
        # 3. used_at check prevents token reuse even if not blacklisted
        try:
            payload = jwt_service.verify_token(data.token, token_type="password_reset")
        except _AUTH_NONCRITICAL_EXCEPTIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired reset token",
            ) from None

        user_id = int(payload["sub"])
        if hasattr(jwt_service, "hash_password_reset_token_candidates"):
            hash_candidates = jwt_service.hash_password_reset_token_candidates(data.token)
        elif hasattr(jwt_service, "hash_password_reset_token"):
            hashed = jwt_service.hash_password_reset_token(data.token)
            hash_candidates = [hashed] if hashed else []
        elif hasattr(jwt_service, "hash_token_candidates"):
            # Backwards compatibility for older JWT service stubs
            hash_candidates = jwt_service.hash_token_candidates(data.token)
        else:
            hash_candidates = []
        if not hash_candidates:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired reset token",
            )

        # Check if token was already used
        token_record_id, token_used_at = await _fetch_password_reset_token_record(
            db,
            user_id=user_id,
            hash_candidates=hash_candidates,
        )

        if not token_record_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired reset token",
            )

        if token_used_at:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="This reset token has already been used",
            )

        # Validate new password (service raises WeakPasswordError on failure)
        try:
            password_service.validate_password_strength(data.new_password)
        except WeakPasswordError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e),
            ) from e

        # Hash new password
        new_password_hash = password_service.hash_password(data.new_password)

        now_utc = datetime.utcnow()
        await _apply_password_reset(
            db,
            user_id=user_id,
            new_password_hash=new_password_hash,
            token_record_id=int(token_record_id),
            now_utc=now_utc,
        )

        # Revoke all existing sessions for security
        blacklist = get_token_blacklist()
        await blacklist.revoke_all_user_tokens(user_id, "Password reset")

        if get_settings().PII_REDACT_LOGS:
            logger.info("Password reset completed for authenticated user (details redacted)")
        else:
            logger.info(f"Password reset completed for user {user_id}")
        return {"message": "Password has been reset successfully"}

    except HTTPException:
        raise
    except _AUTH_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Password reset error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to reset password",
        ) from e


# Email Verification Endpoints

@router.get("/verify-email", status_code=status.HTTP_200_OK)
async def verify_email(
    token: str = Query(..., description="Email verification token"),
    db=Depends(get_db_transaction),
    jwt_service: JWTService = Depends(get_jwt_service_dep),
) -> dict[str, str]:
    """
    Verify email address with token.

    Marks the user's email as verified.
    """
    try:
        # Verify token cryptographically
        # NOTE: Using sync verify_token() is acceptable for email_verification tokens.
        try:
            payload = jwt_service.verify_token(token, token_type="email_verification")
        except _AUTH_NONCRITICAL_EXCEPTIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired verification token",
            ) from None

        try:
            user_id = int(payload.get("sub"))
            email = str(payload.get("email") or "").strip()
            if not email:
                raise ValueError("Missing email claim")
        except _AUTH_NONCRITICAL_EXCEPTIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired verification token",
            )

        # Update user's verification status only when it is currently unverified.
        updated_rows = await _verify_user_email_once(
            db,
            user_id=user_id,
            email=email,
            now_utc=datetime.utcnow(),
        )

        if updated_rows < 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired verification token",
            )

        if get_settings().PII_REDACT_LOGS:
            logger.info("Email verified for authenticated user (details redacted)")
        else:
            logger.info(f"Email verified for user {user_id}")
        return {"message": "Email verified successfully"}

    except HTTPException:
        raise
    except _AUTH_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Email verification error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to verify email",
        ) from e


@router.post("/resend-verification", status_code=status.HTTP_200_OK)
async def resend_verification(
    data: ResendVerificationRequest,
    request: Request,
    db=Depends(get_db_transaction),
    jwt_service: JWTService = Depends(get_jwt_service_dep),
) -> dict[str, str]:
    """
    Resend email verification link.

    Sends a new verification email if the account exists and is not verified.
    """
    try:
        # Per-IP throttling to mitigate verification email abuse.
        client_ip = _auth_request_client_ip(request)
        allowed, _retry_after = await _reserve_auth_rg_requests(
            request,
            policy_id="authnz.resend_verification",
            entity=f"ip:{client_ip}",
            tags={"auth_endpoint": "resend_verification"},
        )
        if not allowed:
            return {"message": "If the account exists and needs verification, an email has been sent"}
        # Check if user exists and needs verification
        user = await _fetch_user_by_email_for_verification(db, data.email)

        if user and not user["is_verified"]:
            # Generate verification token
            verification_token = jwt_service.create_email_verification_token(
                user_id=user["id"],
                email=user["email"],
                expires_in_hours=24,
            )

            # Send email
            email_service = _get_email_service()
            await email_service.send_verification_email(
                to_email=user["email"],
                username=user["username"],
                verification_token=verification_token,
            )

            logger.info(f"Verification email resent for user {user['id']}")

        # Always return success for security
        return {"message": "If the account exists and needs verification, an email has been sent"}

    except _AUTH_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Resend verification error: {e}")
        return {"message": "If the account exists and needs verification, an email has been sent"}


@router.post(
    "/magic-link/request",
    response_model=MessageResponse,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(check_auth_rate_limit)],
)
async def request_magic_link(
    data: MagicLinkRequest,
    request: Request,
    db=Depends(get_db_transaction),
    jwt_service: JWTService = Depends(get_jwt_service_dep),
) -> MessageResponse:
    """
    Request a magic link sign-in email.

    Always returns a generic success message to avoid user enumeration.
    """
    settings = get_settings()
    if settings.AUTH_MODE != "multi_user":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Magic link sign-in is only available in multi-user mode",
        )

    email = _normalize_magic_email(data.email)
    if not email:
        return MessageResponse(message="If the account exists, a sign-in link has been sent")
    try:
        validator = get_input_validator()
        is_valid, _error_msg = validator.validate_email(email)
        if not is_valid:
            return MessageResponse(message="If the account exists, a sign-in link has been sent")
    except _AUTH_NONCRITICAL_EXCEPTIONS:
        # Never fail the request on validation errors; keep response generic
        pass

    # Rate limit by IP and by normalized email to preserve anti-abuse behavior.
    client_ip = _auth_request_client_ip(request)
    ip_allowed, _retry_after = await _reserve_auth_rg_requests(
        request,
        policy_id="authnz.magic_link.request",
        entity=f"ip:{client_ip}",
        tags={"auth_endpoint": "magic_link_request", "scope": "ip"},
    )
    if not ip_allowed:
        return MessageResponse(message="If the account exists, a sign-in link has been sent")
    email_allowed, _retry_after = await _reserve_auth_rg_requests(
        request,
        policy_id="authnz.magic_link.email",
        entity=f"email:{_auth_hashed_entity(email)}",
        tags={"auth_endpoint": "magic_link_request", "scope": "email"},
    )
    if not email_allowed:
        return MessageResponse(message="If the account exists, a sign-in link has been sent")

    user = await fetch_user_by_login_identifier(db, email)
    user_id = int(user["id"]) if user and user.get("id") is not None else None

    # If registration is disabled and user doesn't exist, do not send an email.
    if not user and not settings.ENABLE_REGISTRATION:
        return MessageResponse(message="If the account exists, a sign-in link has been sent")

    try:
        token = jwt_service.create_magic_link_token(
            email=email,
            user_id=user_id,
            expires_in_minutes=settings.MAGIC_LINK_EXPIRE_MINUTES,
        )

        email_service = _get_email_service()
        await email_service.send_magic_link_email(
            to_email=email,
            magic_token=token,
            expires_in_minutes=settings.MAGIC_LINK_EXPIRE_MINUTES,
            username=user.get("username") if user else None,
        )
    except _AUTH_NONCRITICAL_EXCEPTIONS as exc:
        logger.error("Failed to send magic link email: {}", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to send magic link email",
        ) from exc

    return MessageResponse(message="If the account exists, a sign-in link has been sent")


@router.post(
    "/admin/reauth/request",
    response_model=MessageResponse,
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(check_auth_rate_limit)],
)
async def request_admin_reauth(
    request: Request,
    current_user: AuthPrincipal = Depends(get_auth_principal),
    db=Depends(get_db_transaction),
    jwt_service: JWTService = Depends(get_jwt_service_dep),
) -> MessageResponse:
    """Email a purpose-scoped admin reauthentication link to the acting admin."""
    settings = get_settings()
    if settings.AUTH_MODE != "multi_user":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Admin reauthentication is only available in multi-user mode",
        )
    if not _current_user_has_admin_claims(current_user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin reauthentication is only available to administrators",
        )

    user_id = _current_user_id(current_user)
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )

    email = _normalize_magic_email(_current_user_value(current_user, "email", ""))
    username = _current_user_username(current_user)
    if not email:
        actor = await fetch_active_user_by_id(db, int(user_id))
        if actor:
            email = _normalize_magic_email(actor.get("email") or "")
            username = str(actor.get("username") or username)
    if not email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Authenticated admin account is missing an email address",
        )

    expires_in_minutes = max(1, min(int(settings.MAGIC_LINK_EXPIRE_MINUTES), 10))

    try:
        token = jwt_service.create_admin_reauth_token(
            email=email,
            user_id=int(user_id),
            expires_in_minutes=expires_in_minutes,
        )

        email_service = _get_email_service()
        await email_service.send_admin_reauth_email(
            to_email=email,
            reauth_token=token,
            expires_in_minutes=expires_in_minutes,
            username=username,
        )
    except HTTPException:
        raise
    except _AUTH_NONCRITICAL_EXCEPTIONS as exc:
        logger.error("Failed to send admin reauthentication email: {}", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to send admin reauthentication email",
        ) from exc

    return MessageResponse(message="Admin reauthentication link sent")


@router.post(
    "/magic-link/verify",
    response_model=TokenResponse,
    status_code=status.HTTP_200_OK,
)
async def verify_magic_link(
    data: MagicLinkVerifyRequest,
    request: Request,
    db=Depends(get_db_transaction),
    jwt_service: JWTService = Depends(get_jwt_service_dep),
    session_manager: SessionManager = Depends(get_session_manager_dep),
    registration_service: RegistrationService = Depends(get_registration_service_dep),
) -> TokenResponse:
    """Verify a magic link token and return access/refresh tokens."""
    settings = get_settings()
    if settings.AUTH_MODE != "multi_user":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Magic link sign-in is only available in multi-user mode",
        )

    try:
        payload = await jwt_service.verify_token_async(data.token, token_type="magic_link")
    except TokenExpiredError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Magic link token has expired",
        ) from None
    except InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid magic link token",
        ) from None
    except _AUTH_NONCRITICAL_EXCEPTIONS as exc:
        logger.error("Magic link verification failed: {}", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid magic link token",
        ) from exc

    if str(payload.get("purpose") or "").strip().lower() == "admin_reauth":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Admin reauthentication tokens cannot be used for sign-in",
        )

    email = _normalize_magic_email(payload.get("email") or "")
    user_id: Optional[int] = None
    raw_user_id = payload.get("user_id")
    if isinstance(raw_user_id, int):
        user_id = raw_user_id
    elif isinstance(raw_user_id, str) and raw_user_id.isdigit():
        user_id = int(raw_user_id)
    else:
        sub = payload.get("sub")
        if isinstance(sub, int):
            user_id = sub
        elif isinstance(sub, str) and sub.isdigit():
            user_id = int(sub)
        elif not email and isinstance(sub, str) and "@" in sub:
            email = _normalize_magic_email(sub)

    user: Optional[dict[str, Any]] = None
    if user_id is not None:
        user = await fetch_active_user_by_id(db, int(user_id))
    if not user and email:
        user = await fetch_user_by_login_identifier(db, email)

    if not user:
        if not settings.ENABLE_REGISTRATION:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Registration is disabled",
            )

        # Create a new user on first magic-link sign-in
        username_base = _derive_username_from_email(email or "user")
        password = _generate_magic_password()
        user_info: Optional[dict[str, Any]] = None
        for attempt in range(5):
            username = username_base if attempt == 0 else f"{username_base}-{secrets.token_hex(2)}"
            try:
                user_info = await registration_service.register_user(
                    username=username,
                    email=email,
                    password=password,
                    registration_code=None,
                )
                break
            except DuplicateUserError as dupe:
                if getattr(dupe, "field", None) == "email":
                    user = await fetch_user_by_login_identifier(db, email)
                    break
            except _AUTH_NONCRITICAL_EXCEPTIONS as exc:
                logger.error("Magic link registration failed: {}", exc)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to register user",
                ) from exc

        if not user and user_info:
            user_id = int(user_info["user_id"])
            # Mark user verified (magic link serves as email verification)
            await _mark_user_verified(db, user_id, datetime.utcnow())
            user = await fetch_active_user_by_id(db, user_id)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or inactive user",
        )
    if user.get("is_active") is False:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or inactive user",
        )

    # If an existing account wasn't verified, magic link serves as verification
    try:
        if user.get("is_verified") is False:
            await _mark_user_verified(db, int(user["id"]), datetime.utcnow())
            user["is_verified"] = True
    except _AUTH_NONCRITICAL_EXCEPTIONS as exc:
        logger.debug("Magic link verification: failed to mark user verified: {}", exc)

    if settings.AUTH_MODE == "multi_user":
        await _ensure_user_org_membership(int(user["id"]), user.get("username"))

    # Create a session and issue tokens (similar to login flow)
    user_agent = request.headers.get("User-Agent", "Unknown")
    temp_access = f"temp_access_{secrets.token_urlsafe(16)}"
    temp_refresh = f"temp_refresh_{secrets.token_urlsafe(16)}"
    temp_session_info = await session_manager.create_session(
        user_id=user["id"],
        access_token=temp_access,
        refresh_token=temp_refresh,
        ip_address=_auth_request_client_ip(request),
        user_agent=user_agent,
    )
    session_id = temp_session_info["session_id"]

    scope_claims = await _build_scope_claims(int(user["id"]))
    add_claims = dict(scope_claims)
    add_claims["session_id"] = session_id
    access_token = jwt_service.create_access_token(
        user_id=user["id"],
        username=user["username"],
        role=user["role"],
        additional_claims=add_claims,
    )
    refresh_token = jwt_service.create_refresh_token(
        user_id=user["id"],
        username=user["username"],
        additional_claims=add_claims,
    )
    await session_manager.update_session_tokens(
        session_id=session_id,
        access_token=access_token,
        refresh_token=refresh_token,
    )

    await update_user_last_login(db, int(user["id"]), datetime.utcnow())

    # Blacklist the magic link token after use (one-time)
    try:
        jti = payload.get("jti")
        raw_exp = payload.get("exp")
        exp_dt = None
        if isinstance(raw_exp, (int, float)):
            exp_dt = datetime.fromtimestamp(raw_exp, tz=timezone.utc)
        elif isinstance(raw_exp, datetime):
            exp_dt = raw_exp
        if jti and exp_dt:
            from tldw_Server_API.app.core.AuthNZ.token_blacklist import get_token_blacklist
            bl = get_token_blacklist()
            await bl.revoke_token(
                jti=jti,
                expires_at=exp_dt,
                user_id=int(user["id"]),
                token_type="magic_link",
                reason="magic_link_used",
                revoked_by=int(user["id"]),
                ip_address=_auth_request_client_ip(request),
            )
    except _AUTH_NONCRITICAL_EXCEPTIONS as exc:
        logger.debug("Magic link token blacklist failed: {}", exc)

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


# MFA Endpoints

@router.post("/mfa/setup", response_model=MFASetupResponse)
async def setup_mfa(
    current_user: AuthPrincipal = Depends(get_auth_principal),
    db=Depends(get_db_transaction),
    session_manager: SessionManager = Depends(get_session_manager_dep),
) -> MFASetupResponse:
    """
    Initialize MFA setup for current user.

    Generates TOTP secret and backup codes but doesn't enable MFA yet.
    User must verify with a TOTP token first.
    """
    try:
        await _ensure_mfa_available()
        await _ensure_mfa_cache_available(session_manager, get_settings())
        mfa_service = _get_mfa_service()
        user_id = _current_user_id(current_user)
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials",
            )
        username = _current_user_username(current_user)

        # Check if MFA is already enabled
        try:
            mfa_status = await mfa_service.get_user_mfa_status(user_id)
        except DatabaseError as exc:
            logger.debug("MFA status lookup failed due to database error; assuming disabled: {}", exc)
            mfa_status = {"enabled": False}
        if mfa_status["enabled"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="MFA is already enabled for this account",
            )

        # Generate TOTP secret
        secret = mfa_service.generate_secret()

        # Generate QR code
        totp_uri = mfa_service.generate_totp_uri(secret, username)
        qr_code_bytes = mfa_service.generate_qr_code(totp_uri)
        qr_code_base64 = base64.b64encode(qr_code_bytes).decode("utf-8")

        # Generate backup codes
        backup_codes = mfa_service.generate_backup_codes()

        try:
            payload = json.dumps({
                "secret": secret,
                "backup_codes": backup_codes,
            })
            await session_manager.store_ephemeral_value(
                _mfa_setup_cache_key(user_id),
                payload,
                _get_mfa_setup_ttl_seconds(),
            )
        except _AUTH_NONCRITICAL_EXCEPTIONS as exc:
            logger.error("Failed to cache MFA setup secret: {}", exc)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to setup MFA",
            ) from exc

        return MFASetupResponse(
            secret=secret,
            qr_code=f"data:image/png;base64,{qr_code_base64}",
            backup_codes=backup_codes,
        )

    except HTTPException:
        raise
    except _AUTH_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"MFA setup error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to setup MFA",
        ) from e


@router.post("/mfa/verify", status_code=status.HTTP_200_OK)
async def verify_mfa_setup(
    data: MFAVerifyRequest,
    request: Request,
    current_user: AuthPrincipal = Depends(get_auth_principal),
    session_manager: SessionManager = Depends(get_session_manager_dep),
) -> dict[str, Any]:
    """
    Verify and enable MFA with TOTP token.

    Completes MFA setup by verifying the user can generate valid tokens.
    """
    try:
        await _ensure_mfa_available()
        await _ensure_mfa_cache_available(session_manager, get_settings())
        mfa_service = _get_mfa_service()
        user_id = _current_user_id(current_user)
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials",
            )
        username = _current_user_username(current_user)

        cached = await session_manager.get_ephemeral_value(_mfa_setup_cache_key(user_id))
        if not cached:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="MFA setup not found or expired. Please restart setup.",
            )
        secret = None
        backup_codes = None
        try:
            parsed = json.loads(cached)
            if isinstance(parsed, dict):
                secret = parsed.get("secret")
                backup_codes = parsed.get("backup_codes")
        except _AUTH_NONCRITICAL_EXCEPTIONS:
            secret = None
            backup_codes = None
        if not secret:
            secret = cached

        allowed, retry_after = await _reserve_auth_rg_requests(
            request,
            policy_id="authnz.mfa.verify",
            entity=f"user:{user_id}",
            tags={"auth_endpoint": "mfa_verify"},
        )
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many attempts",
                headers={"Retry-After": str(int(retry_after or 1))},
            )
        # Verify TOTP token
        if not mfa_service.verify_totp(secret, data.token):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid TOTP token",
            )

        # Generate final backup codes if none were staged during setup
        if not isinstance(backup_codes, list) or not backup_codes:
            backup_codes = mfa_service.generate_backup_codes()

        # Enable MFA
        success = await mfa_service.enable_mfa(
            user_id=user_id,
            secret=secret,
            backup_codes=backup_codes,
        )

        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to enable MFA",
            )

        await session_manager.delete_ephemeral_value(_mfa_setup_cache_key(user_id))

        # Send email with backup codes (best effort; MFA enablement already committed).
        email = _current_user_value(current_user, "email")
        if email:
            email_service = _get_email_service()
            client_ip = _auth_request_client_ip(request)
            try:
                await email_service.send_mfa_enabled_email(
                    to_email=str(email),
                    username=username,
                    backup_codes=backup_codes,
                    ip_address=client_ip,
                )
            except _AUTH_NONCRITICAL_EXCEPTIONS as email_exc:
                logger.warning("MFA enabled but notification email failed for user {}: {}", user_id, email_exc)
        else:
            logger.info("MFA enabled for user {} without notification email (no email on profile)", user_id)

        logger.info(f"MFA enabled for user {user_id}")
        return {"message": "MFA has been enabled successfully", "backup_codes": backup_codes}

    except HTTPException:
        raise
    except _AUTH_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"MFA verification error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to verify MFA",
        ) from e


@router.post("/mfa/disable", status_code=status.HTTP_200_OK)
async def disable_mfa(
    current_user: AuthPrincipal = Depends(get_auth_principal),
    password: str = Form(..., description="Current password for verification"),
    db=Depends(get_db_transaction),
    password_service: PasswordService = Depends(get_password_service_dep),
) -> dict[str, str]:
    """
    Disable MFA for current user.

    Requires password verification for security.
    """
    try:
        await _ensure_mfa_available()
        user_id = _current_user_id(current_user)
        if user_id is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

        user_record = await fetch_active_user_by_id(db, int(user_id))
        if not isinstance(user_record, dict):
            try:
                user_record = dict(user_record) if user_record is not None else None
            except _AUTH_NONCRITICAL_EXCEPTIONS:
                user_record = None
        if not user_record:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

        password_hash = user_record.get("password_hash")
        if not isinstance(password_hash, str) or not password_hash:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

        try:
            password_service.hasher.verify(password_hash, password)
        except _AUTH_NONCRITICAL_EXCEPTIONS:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials",
            ) from None

        mfa_service = _get_mfa_service()
        success = await mfa_service.disable_mfa(int(user_id))

        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to disable MFA",
            )

        logger.info(f"MFA disabled for user {user_id}")
        return {"message": "MFA has been disabled"}

    except HTTPException:
        raise
    except _AUTH_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"MFA disable error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to disable MFA",
        ) from e


@router.post("/mfa/login", response_model=TokenResponse)
async def mfa_login(
    data: MFALoginRequest,
    request: Request,
    response: Response,
    db=Depends(get_db_transaction),
    jwt_service: JWTService = Depends(get_jwt_service_dep),
    session_manager: SessionManager = Depends(get_session_manager_dep),
    rate_limiter: RateLimiter = Depends(get_rate_limiter_dep),
    settings: Settings = Depends(get_settings),
) -> TokenResponse:
    """
    Complete login with MFA token.
    """
    start_time = time.perf_counter()
    log_counter("auth_mfa_login_attempt")
    try:
        await _ensure_mfa_available()
        await _ensure_mfa_cache_available(session_manager, settings)

        cache_key = _mfa_login_cache_key(data.session_token)
        payload_raw = await session_manager.get_ephemeral_value(cache_key)
        if not payload_raw:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="MFA session expired or invalid",
            )
        try:
            payload = json.loads(payload_raw)
        except json.JSONDecodeError:
            payload = {}

        session_id = payload.get("session_id")
        user_id = payload.get("user_id")
        if not session_id or not user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="MFA session expired or invalid",
            )

        user_id = int(user_id)
        session_id = int(session_id)

        allowed, retry_after = await _reserve_auth_rg_requests(
            request,
            policy_id="authnz.mfa.login",
            entity=f"user:{user_id}",
            tags={"auth_endpoint": "mfa_login"},
        )
        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many MFA attempts. Please try again later.",
                headers={"Retry-After": str(int(retry_after or 1))},
            )

        user = await fetch_active_user_by_id(db, user_id)
        if not isinstance(user, dict):
            try:
                user = dict(user) if user is not None else None
            except _AUTH_NONCRITICAL_EXCEPTIONS:
                user = None
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials",
            )

        mfa_service = _get_mfa_service()
        mfa_status = await mfa_service.get_user_mfa_status(user_id)
        if not mfa_status.get("enabled"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="MFA is not enabled for this account",
            )

        secret = await mfa_service.get_user_totp_secret(user_id)
        token_ok = False
        if secret and mfa_service.verify_totp(secret, data.mfa_token):
            token_ok = True
        else:
            token_ok = await mfa_service.verify_backup_code(user_id, data.mfa_token)

        if not token_ok:
            log_counter("auth_mfa_login_invalid_token")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid MFA token",
                headers={"WWW-Authenticate": "Bearer"},
            )

        if settings.AUTH_MODE == "multi_user":
            await _ensure_user_org_membership(user_id, user.get("username"))

        scope_claims = await _build_scope_claims(user_id)
        add_claims = dict(scope_claims)
        add_claims["session_id"] = session_id
        access_token = jwt_service.create_access_token(
            user_id=user_id,
            username=user.get("username", ""),
            role=user.get("role", "user"),
            additional_claims=add_claims,
        )
        refresh_token = jwt_service.create_refresh_token(
            user_id=user_id,
            username=user.get("username", ""),
            additional_claims=add_claims,
        )

        await session_manager.update_session_tokens(
            session_id=session_id,
            access_token=access_token,
            refresh_token=refresh_token,
        )

        await update_user_last_login(db, user_id, datetime.utcnow())

        # Audit log successful login
        client_ip = _auth_request_client_ip(request)
        user_agent = request.headers.get("User-Agent", "Unknown")
        async def _safe_audit_log_login(user_id: int, username: str, ip: str, ua: str, success: bool):
            try:
                svc = await get_or_create_audit_service_for_user_id(user_id)
                await svc.log_login(
                    user_id=user_id,
                    username=username,
                    ip_address=ip,
                    user_agent=ua,
                    success=success,
                )
                flush_on_login = _env_flag_enabled("AUDIT_FLUSH_ON_LOGIN")
                test_mode = _is_test_mode()
                if flush_on_login or test_mode:
                    await svc.flush()
            except _AUTH_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug(
                    "MFA login audit failed for user_id={}: {}",
                    user_id,
                    exc,
                    exc_info=True,
                )

        await _safe_audit_log_login(
            user_id=user_id,
            username=str(user.get("username", "")),
            ip=client_ip,
            ua=user_agent,
            success=True,
        )

        # Reset failed login attempts on successful MFA login
        if getattr(rate_limiter, 'enabled', False):
            try:
                await rate_limiter.reset_failed_attempts(client_ip, "login")
                await rate_limiter.reset_failed_attempts(user.get("username", ""), "login")
            except _AUTH_NONCRITICAL_EXCEPTIONS as rl_exc:
                logger.debug(f"rate_limiter.reset_failed_attempts failed: {rl_exc}")

        with contextlib.suppress(_AUTH_NONCRITICAL_EXCEPTIONS):
            await session_manager.delete_ephemeral_value(cache_key)

        log_counter("auth_login_success")
        log_counter("auth_mfa_login_success")
        log_histogram("auth_login_duration", time.perf_counter() - start_time)

        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
        )

    except HTTPException:
        log_counter("auth_mfa_login_http_error")
        log_histogram("auth_login_duration", time.perf_counter() - start_time)
        raise
    except _AUTH_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"MFA login error: {e}")
        log_counter("auth_mfa_login_unexpected_error")
        log_histogram("auth_login_duration", time.perf_counter() - start_time)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to complete MFA login",
        ) from e


#######################################################################################################################
#
# Registration Endpoint

@router.post("/register", response_model=RegistrationResponse, dependencies=[Depends(check_auth_rate_limit)])
async def register(
    payload: RegisterRequest,
    http_request: Request,
    response: Response,
    _diag=Depends(_register_runtime_diag),  # noqa: B008
    registration_service: RegistrationService = Depends(get_registration_service_dep)
) -> RegistrationResponse:
    """
    Register a new user

    Creates a new user account with the provided credentials.
    May require a registration code if configured.

    Args:
        payload: RegisterRequest with user details

    Returns:
        RegistrationResponse with user information

    Raises:
        HTTPException: 400 if validation fails, 409 if user exists
    """
    start_time = time.perf_counter()
    log_counter("auth_register_attempt")
    settings = get_settings()
    try:
        # Hard constraint for local-single-user profile: do not allow
        # registration of additional users beyond the bootstrapped admin.
        profile = get_profile()
        if isinstance(profile, str) and profile.strip().lower() in {"local-single-user", "single_user"}:
            if getattr(settings, "PII_REDACT_LOGS", False):
                logger.warning("Registration attempt rejected in local-single-user profile [redacted]")
            else:
                logger.warning(
                    "Registration attempt rejected in local-single-user profile for username={}",
                    payload.username,
                )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User registration is not allowed in local-single-user profile",
            )

        # Register the user
        user_info = await registration_service.register_user(
            username=payload.username,
            email=payload.email,
            password=payload.password,
            registration_code=payload.registration_code,
        )

        logger.info(f"New user registered: {user_info['username']} (ID: {user_info['user_id']})")

        # If using SQLite for AuthNZ, generate a per-user API key so UI can present it
        api_key_value = None
        try:
            if isinstance(settings.DATABASE_URL, str) and settings.DATABASE_URL.startswith("sqlite"):
                api_mgr = await get_api_key_manager()
                key_result = await api_mgr.create_api_key(
                    user_id=int(user_info['user_id']),
                    name="Default API Key",
                    description="Auto-generated on registration",
                    scope="write",
                    expires_in_days=365
                )
                api_key_value = key_result.get('key')
        except MandatoryAuditWriteError as exc:
            rollback_ok = await registration_service.rollback_user_registration(int(user_info["user_id"]))
            logger.error(
                "Mandatory audit write failed while auto-generating default API key for new user {} (rollback_ok={}): {}",
                user_info["user_id"],
                rollback_ok,
                exc,
                exc_info=exc,
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "error": {
                        "message": "Mandatory audit persistence unavailable",
                        "type": "audit_persistence_failure",
                        "code": "audit_persistence_failure",
                    }
                },
            ) from exc
        except _AUTH_NONCRITICAL_EXCEPTIONS as _e:
            logger.warning(f"Failed to auto-generate API key for new user {user_info['user_id']}: {_e}")

        log_counter("auth_register_success")
        log_histogram("auth_register_duration", time.perf_counter() - start_time)
        # Attach diagnostics (if enabled)
        _finalize_register_diag(http_request, response)

        async def _safe_audit_log_registration_code() -> None:
            try:
                code_id = user_info.get("registration_code_id")
                if not payload.registration_code or code_id is None:
                    return
                svc = await get_or_create_audit_service_for_user_id(int(user_info["user_id"]))
                correlation_id = (
                    http_request.headers.get("X-Correlation-ID")
                    or getattr(http_request.state, "correlation_id", None)
                )
                request_id = (
                    http_request.headers.get("X-Request-ID")
                    or getattr(http_request.state, "request_id", None)
                    or ""
                )
                ctx = AuditContext(
                    user_id=str(user_info["user_id"]),
                    correlation_id=correlation_id,
                    request_id=request_id,
                    ip_address=(_auth_request_client_ip(http_request) if http_request else None),
                    user_agent=http_request.headers.get("user-agent"),
                    endpoint=str(http_request.url.path),
                    method=http_request.method,
                )
                await svc.log_event(
                    event_type=AuditEventType.DATA_UPDATE,
                    context=ctx,
                    resource_type="registration_code",
                    resource_id=str(code_id),
                    action="registration_code.redeemed",
                    metadata={
                        "registration_code_id": code_id,
                        "org_id": user_info.get("registration_code_org_id"),
                        "org_role": user_info.get("registration_code_org_role"),
                        "team_id": user_info.get("registration_code_team_id"),
                    },
                )
            except _AUTH_NONCRITICAL_EXCEPTIONS as exc:
                logger.debug("Registration code audit failed: {}", exc)

        await _safe_audit_log_registration_code()
        return RegistrationResponse(
            message="Registration successful",
            user_id=user_info['user_id'],
            username=user_info['username'],
            email=user_info['email'],
            requires_verification=not user_info['is_verified'],
            api_key=api_key_value
        )

    except DuplicateUserError as e:
        logger.warning(f"Registration failed - duplicate user: {e}")
        log_counter("auth_register_duplicate")
        log_histogram("auth_register_duration", time.perf_counter() - start_time)
        # Attach diagnostics (if enabled)
        if _is_test_mode():
            with contextlib.suppress(_AUTH_NONCRITICAL_EXCEPTIONS):
                response.headers["X-TLDW-Register-Error"] = "duplicate-user"
        _finalize_register_diag(http_request, response)
        detail = "Username or email already exists."
        if payload.registration_code:
            detail = (
                "Username or email already exists. If you're joining an organization, "
                "log in and accept the invite in the WebUI "
                "or POST /api/v1/orgs/invites/accept."
            )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=detail
        ) from e
    except WeakPasswordError as e:
        logger.warning(f"Registration failed - weak password: {e}")
        log_counter("auth_register_weak_password")
        log_histogram("auth_register_duration", time.perf_counter() - start_time)
        if _is_test_mode():
            with contextlib.suppress(_AUTH_NONCRITICAL_EXCEPTIONS):
                response.headers["X-TLDW-Register-Error"] = "weak-password"
        _finalize_register_diag(http_request, response)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password does not meet requirements."
        ) from e
    except InvalidRegistrationCodeError as e:
        logger.warning(f"Registration failed - invalid code: {e}")
        log_counter("auth_register_invalid_code")
        log_histogram("auth_register_duration", time.perf_counter() - start_time)
        if _is_test_mode():
            with contextlib.suppress(_AUTH_NONCRITICAL_EXCEPTIONS):
                response.headers["X-TLDW-Register-Error"] = "invalid-registration-code"
        _finalize_register_diag(http_request, response)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid registration code."
        ) from e
    except RegistrationError as e:
        logger.error(f"Registration error: {e}")
        log_counter("auth_register_error")
        log_histogram("auth_register_duration", time.perf_counter() - start_time)
        if _is_test_mode():
            with contextlib.suppress(_AUTH_NONCRITICAL_EXCEPTIONS):
                response.headers["X-TLDW-Register-Error"] = "registration-error"
        _finalize_register_diag(http_request, response)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Registration failed."
        ) from e
    except HTTPException:
        # Propagate explicit HTTPException responses (for example the
        # local-single-user profile guard) without wrapping them as 500.
        _finalize_register_diag(http_request, response)
        raise
    except _AUTH_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Unexpected registration error: {e}")
        log_counter("auth_register_unexpected_error")
        log_histogram("auth_register_duration", time.perf_counter() - start_time)
        if _is_test_mode():
            with contextlib.suppress(_AUTH_NONCRITICAL_EXCEPTIONS):
                response.headers["X-TLDW-Register-Error"] = "internal-error"
        duration = time.perf_counter() - start_time
        _finalize_register_diag(http_request, response)
        if _is_test_mode():
            try:
                pool = await get_db_pool()
                db_backend = "postgres" if getattr(pool, "pool", None) is not None else "sqlite"
            except _AUTH_NONCRITICAL_EXCEPTIONS:
                db_backend = "unknown"
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": "An error occurred during registration"},
                headers={
                    "X-TLDW-DB": db_backend,
                    "X-TLDW-CSRF-Enabled": "true" if bool(_csrf_globals.get("CSRF_ENABLED", None)) else "false",
                    "X-TLDW-Register-Error": "internal-error",
                    "X-TLDW-Register-Duration-ms": str(int(duration * 1000)),
                },
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during registration"
        ) from e


#######################################################################################################################
#
# Test Endpoint

@router.get("/me", response_model=DeprecatedUserResponse, deprecated=True)
async def get_current_user_info(
    current_user: AuthPrincipal = Depends(get_auth_principal),
    response: Response = None,
) -> DeprecatedUserResponse:
    """
    Deprecated: use /api/v1/users/me/profile.

    Returns:
        UserResponse with current user details
    """
    successor = "/api/v1/users/me/profile"
    if not _legacy_user_me_enabled():
        from fastapi.responses import JSONResponse

        return JSONResponse(
            status_code=status.HTTP_410_GONE,
            content=_legacy_warning_payload(successor),
        )
    try:
        if response is not None:
            response.headers.update(build_deprecation_headers(successor))
    except _AUTH_NONCRITICAL_EXCEPTIONS:
        pass

    user_id = _current_user_id(current_user)
    if user_id is None or user_id <= 0:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )

    return DeprecatedUserResponse(
        warning="deprecated_endpoint",
        successor=successor,
        id=user_id,
        uuid=_current_user_value(current_user, "uuid", None) or None,
        username=_current_user_username(current_user),
        email=str(_current_user_value(current_user, "email", "") or ""),
        role=_current_user_primary_role(current_user),
        is_active=bool(_current_user_value(current_user, "is_active", True)),
        is_verified=bool(_current_user_value(current_user, "is_verified", True)),
        created_at=_current_user_value(current_user, "created_at", datetime.utcnow()),
        last_login=_current_user_value(current_user, "last_login"),
        storage_quota_mb=_current_user_value(current_user, "storage_quota_mb", 1000),
        storage_used_mb=_current_user_value(current_user, "storage_used_mb", 0.0),
    )


#
# End of auth.py
#######################################################################################################################
