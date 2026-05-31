# User_DB_Handling.py
# Description: Handles user authentication and identification based on application mode.
#
# Imports
import contextlib
import os
from typing import Any, Optional, Union
from uuid import UUID

#
# 3rd-Party Libraries
from fastapi import Depends, Header, HTTPException, Request, status

# Utils
from loguru import logger
from pydantic import BaseModel, Field, ValidationError

# API Dependencies
from tldw_Server_API.app.api.v1.API_Deps.v1_endpoint_deps import oauth2_scheme
from tldw_Server_API.app.core.AuthNZ.api_key_manager import get_api_key_manager
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
from tldw_Server_API.app.core.AuthNZ.exceptions import DatabaseError, InvalidTokenError, TokenExpiredError
from tldw_Server_API.app.core.AuthNZ.ip_allowlist import (
    is_single_user_ip_allowed,
    resolve_client_ip,
)

# New JWT service
from tldw_Server_API.app.core.AuthNZ.jwt_service import get_jwt_service
from tldw_Server_API.app.core.AuthNZ.org_rbac import apply_scoped_permissions
from tldw_Server_API.app.core.AuthNZ.orgs_teams import (
    list_memberships_for_user,
    list_org_memberships_for_user,
)
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.repos.rbac_repo import AuthnzRbacRepo
from tldw_Server_API.app.core.AuthNZ.session_manager import get_session_manager
from tldw_Server_API.app.core.testing import env_flag_enabled, is_test_mode, is_truthy

#
# Local Imports
# New unified settings
from tldw_Server_API.app.core.AuthNZ.settings import get_settings
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.DB_Management.scope_context import set_scope
from tldw_Server_API.app.core.DB_Management.backends.base import (
    DatabaseError as BackendDatabaseError,
)
from tldw_Server_API.app.core.exceptions import InactiveUserError

_USER_DB_NONCRITICAL_EXCEPTIONS = (
    AssertionError,
    AttributeError,
    ConnectionError,
    FileNotFoundError,
    ImportError,
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
    DatabaseError,
    BackendDatabaseError,
    HTTPException,
    InvalidTokenError,
    TokenExpiredError,
    ValidationError,
)

#######################################################################################################################

def is_single_user_mode() -> bool:
    """Compatibility helper for tests and legacy callers."""
    try:
        return get_settings().AUTH_MODE == "single_user"
    except _USER_DB_NONCRITICAL_EXCEPTIONS:
        return False


def is_multi_user_mode() -> bool:
    """Compatibility helper for tests and legacy callers."""
    return get_settings().AUTH_MODE == "multi_user"


def get_effective_permissions(user_id: int) -> list[str]:
    """
    Compatibility helper that returns effective permission codes for a user.

    Several integration tests monkeypatch this symbol on the User_DB_Handling
    module to inject additional permission codes. The canonical implementation
    delegates to the RBAC repository facade.
    """
    repo = AuthnzRbacRepo(client_id="authnz_effective_permissions")
    try:
        return list(repo.get_effective_permissions(int(user_id)))
    except _USER_DB_NONCRITICAL_EXCEPTIONS:
        logger.exception("Failed to resolve effective permissions for user_id={}", user_id)
        return []


def _enrich_user_with_rbac(
    user_id: Optional[int],
    user_data: dict,
    *,
    pii_redact_logs: bool = False,
) -> tuple[list[str], list[str], bool]:
    """
    Fetch roles/permissions/admin flag for a user from central RBAC tables.
    """
    repo = AuthnzRbacRepo(client_id="authnz_user_enrichment")
    roles: list[str] = []
    perms: list[str] = []
    is_admin_flag = bool(user_data.get("is_superuser") or user_data.get("is_admin"))
    if user_id is None:
        return roles, perms, is_admin_flag

    base_perms: set[str] = set()

    # Roles from centralized RBAC repository
    try:
        role_rows = repo.get_user_roles(int(user_id))
        for row in role_rows or []:
            role_name = row.get("name") or row.get("role") or row.get("role_name")
            if role_name:
                role_str = str(role_name)
                if role_str not in roles:
                    roles.append(role_str)
        if "admin" in roles:
            is_admin_flag = True
    except _USER_DB_NONCRITICAL_EXCEPTIONS as rb_exc:
        if pii_redact_logs:
            logger.debug("RBAC enrichment failed for user roles (details redacted)")
        else:
            logger.debug(f"RBAC enrichment failed for user {user_id} roles: {rb_exc}")

    def _merge_role_permissions(role_name: str) -> None:
        """Merge role_permissions-derived permission codes into base_perms (best-effort)."""
        nonlocal base_perms
        try:
            role_row_id = repo.get_role_id_by_name(str(role_name))
        except _USER_DB_NONCRITICAL_EXCEPTIONS as rb_exc_lookup:  # pragma: no cover - best-effort lookup
            role_row_id = None
            if pii_redact_logs:
                logger.debug("RBAC role permission lookup failed [redacted]")
            else:
                logger.debug(
                    f"RBAC role permission lookup failed for role {role_name}: {rb_exc_lookup}"
                )
        if role_row_id is None:
            return
        try:
            role_perms = repo.get_role_effective_permissions(int(role_row_id))
            for pname in role_perms.get("all_permissions", []):
                if pname:
                    base_perms.add(str(pname))
        except _USER_DB_NONCRITICAL_EXCEPTIONS as rb_exc_role:  # pragma: no cover - best-effort
            if pii_redact_logs:
                logger.debug("RBAC role permission expansion failed [redacted]")
            else:
                logger.debug(
                    f"RBAC role permission expansion failed for role {role_name}: {rb_exc_role}"
                )

    # Effective permissions from RBAC helper (roles + user overrides)
    try:
        base_perms = set(get_effective_permissions(int(user_id)))
    except _USER_DB_NONCRITICAL_EXCEPTIONS as rb_exc:
        if pii_redact_logs:
            logger.debug("RBAC enrichment failed for permissions (details redacted)")
        else:
            logger.debug(f"RBAC enrichment failed for user {user_id} permissions: {rb_exc}")

    # If we learned roles but permission expansion failed (schema drift, stale caches, etc),
    # fall back to role-permission mappings so baseline permission gates keep working.
    if roles and not base_perms:
        for role_name in roles:
            _merge_role_permissions(role_name)

    # Fallback: honor legacy role column when user_roles entries are absent
    if not roles:
        implicit_role = user_data.get("role")
        if implicit_role:
            try:
                rname = str(implicit_role)
                roles.append(rname)
                if rname == "admin":
                    is_admin_flag = True
                _merge_role_permissions(rname)
            except _USER_DB_NONCRITICAL_EXCEPTIONS as rb_exc_outer:  # pragma: no cover - guard against unexpected shapes
                if pii_redact_logs:
                    logger.debug("RBAC fallback (role column) outer failure [redacted]")
                else:
                    logger.debug(
                        f"RBAC fallback (role column) outer failure for role {implicit_role}: {rb_exc_outer}"
                    )

    perms = sorted(base_perms)
    if is_admin_flag:
        perms = sorted(set(perms) | {"system.configure"})

    return roles, perms, is_admin_flag


def _coerce_int_list(raw: Any) -> list[int]:
    if not isinstance(raw, (list, tuple, set)):
        return []
    out: list[int] = []
    for value in raw:
        try:
            out.append(int(value))
        except (TypeError, ValueError):
            continue
    return out


def _normalize_active_id(raw: Any, ids: list[int]) -> Optional[int]:
    if raw is None:
        return ids[0] if len(ids) == 1 else None
    try:
        active = int(raw)
    except (TypeError, ValueError):
        return None
    if active in ids:
        return active
    return None

# --- User Model ---
# Standardized User object, used even for the dummy single user.
class User(BaseModel):
    # Accept either integer DB ids or string tenant-style ids in tests
    id: Union[int, str]
    uuid: Optional[UUID] = None
    username: str
    email: Optional[str] = None
    role: str = "user"
    is_active: bool = True
    is_verified: bool = True
    is_superuser: bool = False
    # Optional tenant field for multi-tenant-aware endpoints/tests
    tenant_id: Optional[str] = None
    # RBAC/claims exposure
    roles: list[str] = Field(default_factory=list)
    permissions: list[str] = Field(default_factory=list)
    is_admin: bool = False

    # Convenience properties for downstream code that expects int ids
    @property
    def id_int(self) -> Optional[int]:
        try:
            return int(self.id)  # type: ignore[arg-type]
        except _USER_DB_NONCRITICAL_EXCEPTIONS:
            return None

    @property
    def id_str(self) -> str:
        try:
            return str(self.id)
        except _USER_DB_NONCRITICAL_EXCEPTIONS:
            return ""

# --- Single User "Dummy" Object ---
# Created when in single-user mode using values from the settings
_single_user_instance: Optional[User] = None

def get_single_user_instance() -> User:
    """Get or create the single user instance"""
    global _single_user_instance
    settings = get_settings()
    desired_id = settings.SINGLE_USER_FIXED_ID

    default_permissions = list(getattr(settings, "SINGLE_USER_DEFAULT_PERMISSIONS", []) or [])
    if not default_permissions:
        # Defensive baseline to satisfy permission-gated routes in single-user mode.
        default_permissions = ["system.configure", "media.read", "media.create"]

    if (
        _single_user_instance is None
        or getattr(_single_user_instance, "id", None) != desired_id
    ):
        _single_user_instance = User(
            id=desired_id,
            username="single_user",
            email="",
            role="admin",
            is_active=True,
            is_verified=True,
            roles=["admin"],
            permissions=default_permissions,
            is_admin=True,
        )
    return _single_user_instance

# Eagerly initialize for environments/tests that mutate the module-level reference directly.
try:
    _single_user_instance = get_single_user_instance()
except _USER_DB_NONCRITICAL_EXCEPTIONS:
    _single_user_instance = None


def _is_test_context() -> bool:
    """Return True when running in pytest or explicit test-mode contexts."""
    try:
        if getattr(get_settings(), "TEST_MODE", False):
            return True
    except _USER_DB_NONCRITICAL_EXCEPTIONS:
        pass
    if os.getenv("PYTEST_CURRENT_TEST") is not None:
        return True
    return is_test_mode() or env_flag_enabled("TESTING")


def _is_strict_test_bypass_context() -> bool:
    """
    Return True only when an explicit test runtime is detected.

    This is intentionally stricter than `_is_test_context()` so that accidental
    `TESTING=1` environment leakage does not enable auth bypasses in non-test
    deployments.
    """
    if os.getenv("PYTEST_CURRENT_TEST") is not None:
        return True
    if is_test_mode():
        return True
    try:
        if getattr(get_settings(), "TEST_MODE", False):
            return True
    except _USER_DB_NONCRITICAL_EXCEPTIONS:
        pass
    return False


def _is_production_like_env() -> bool:
    """
    Detect production-like runtime from common deployment environment variables.

    This intentionally checks multiple keys because not all deployments set
    `tldw_production`.
    """
    if is_truthy(os.getenv("tldw_production", "").strip().lower()):
        return True

    production_values = {"production", "prod", "live"}
    for key in (
        "ENVIRONMENT",
        "APP_ENV",
        "DEPLOYMENT_ENV",
        "FASTAPI_ENV",
        "TLDW_ENV",
    ):
        value = os.getenv(key, "").strip().lower()
        if value in production_values:
            return True
    return False


def _looks_like_jwt(token: Optional[str]) -> bool:
    if not isinstance(token, str):
        return False
    return token.count(".") == 2


def _raise_user_id_error(detail: str, *, status_code: int, raise_http: bool) -> None:
    if raise_http:
        raise HTTPException(status_code=status_code, detail=detail)
    raise ValueError(detail)


def resolve_user_id_value(
    user_id: Optional[Union[int, str]],
    *,
    allow_none: bool = False,
    as_int: bool = False,
    allow_test_user_ids: Optional[bool] = None,
    error_status: int = status.HTTP_400_BAD_REQUEST,
    missing_detail: str = "user_id is required in multi-user mode",
    invalid_detail: str = "invalid user_id",
    raise_http: bool = False,
) -> Optional[Union[int, str]]:
    """Normalize user_id values with single-user fallback and test-friendly rules."""
    if allow_test_user_ids is None:
        allow_test_user_ids = _is_test_context()

    missing = user_id is None or (isinstance(user_id, str) and not user_id.strip())
    if missing:
        if is_single_user_mode():
            user_id = DatabasePaths.get_single_user_id()
        elif allow_none:
            return None
        else:
            _raise_user_id_error(missing_detail, status_code=error_status, raise_http=raise_http)

    if as_int:
        try:
            return int(user_id)  # type: ignore[arg-type]
        except _USER_DB_NONCRITICAL_EXCEPTIONS:
            if is_single_user_mode():
                return DatabasePaths.get_single_user_id()
            _raise_user_id_error(invalid_detail, status_code=error_status, raise_http=raise_http)

    if not allow_test_user_ids:
        try:
            int(user_id)  # type: ignore[arg-type]
        except _USER_DB_NONCRITICAL_EXCEPTIONS:
            if is_single_user_mode():
                return str(DatabasePaths.get_single_user_id())
            _raise_user_id_error(invalid_detail, status_code=error_status, raise_http=raise_http)

    return str(user_id)


def resolve_user_id_for_request(
    current_user: Optional["User"],
    *,
    allow_none: bool = False,
    as_int: bool = False,
    allow_test_user_ids: Optional[bool] = None,
    error_status: int = status.HTTP_400_BAD_REQUEST,
    missing_detail: str = "user_id is required in multi-user mode",
    invalid_detail: str = "invalid user_id",
) -> Optional[Union[int, str]]:
    """Normalize current_user.id for HTTP requests and raise HTTPException on errors."""
    user_id = getattr(current_user, "id", None)
    return resolve_user_id_value(
        user_id,
        allow_none=allow_none,
        as_int=as_int,
        allow_test_user_ids=allow_test_user_ids,
        error_status=error_status,
        missing_detail=missing_detail,
        invalid_detail=invalid_detail,
        raise_http=True,
    )

#######################################################################################################################

# --- Verification Dependencies ---

async def verify_single_user_api_key(
    _request: Request,
    api_key: Optional[str] = Header(None, alias="X-API-KEY"),
    authorization: Optional[str] = Header(None, alias="Authorization"),
) -> bool:
    """
    Dependency to verify that the provided credentials correspond to the
    bootstrapped single-user admin principal.

    This helper no longer branches on AUTH_MODE. Instead, it:
    - Extracts the candidate API key from headers (X-API-KEY or Bearer).
    - Uses the standard API-key authentication flow to resolve a User.
    - Asserts that the resolved user id matches SINGLE_USER_FIXED_ID.
    """
    settings = get_settings()
    expected_user_id = getattr(settings, "SINGLE_USER_FIXED_ID", 1)

    # Derive the presented API key from headers
    provided = api_key or ""
    if not provided and authorization:
        try:
            scheme, _, credential = authorization.partition(" ")
            if scheme.lower() == "bearer":
                provided = credential.strip()
        except _USER_DB_NONCRITICAL_EXCEPTIONS:
            provided = ""

    if not provided:
        if settings.PII_REDACT_LOGS:
            logger.warning("Invalid or missing API Key for single-user verification (no credentials)")
        else:
            logger.warning("Invalid or missing API Key for single-user verification: '<missing>'")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key",
        )

    # Use the shared API-key authentication path so that verification is
    # driven by AuthNZ tables and RBAC rather than inline comparisons.
    user = await authenticate_api_key_user(_request, provided)

    try:
        user_id_val = getattr(user, "id_int", None)
        if user_id_val is None:
            user_id_val = int(user.id)
    except _USER_DB_NONCRITICAL_EXCEPTIONS:
        user_id_val = None

    if user_id_val != expected_user_id:
        if settings.PII_REDACT_LOGS:
            logger.warning("API Key resolved to non-single-user principal during single-user verification")
        else:
            logger.warning(
                f"API Key resolved to user_id={user_id_val} during single-user verification; "
                f"expected {expected_user_id}"
            )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key",
        )

    logger.debug("Single-user API Key verified successfully via AuthNZ store.")
    return True


async def verify_jwt_and_fetch_user(request: Request, token: str = Depends(oauth2_scheme)) -> User:
    """
    Dependency to verify JWT and fetch user details.

    Uses the new JWT service for token validation and is agnostic to
    AUTH_MODE; callers decide whether JWTs are appropriate for their
    deployment via configuration and routing, not via this helper.
    """

    # Resolve users via the AuthNZ repository to keep backend differences localized.
    try:
        from tldw_Server_API.app.core.AuthNZ.repos.users_repo import AuthnzUsersRepo
    except ImportError:
        logger.error("AuthNZ users repository is unavailable; cannot resolve JWT subjects.")
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="AuthNZ user lookup is unavailable.",
        ) from None

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    pii_redact_logs = get_settings().PII_REDACT_LOGS

    # Use new JWT service to decode token
    jwt_service = get_jwt_service()
    try:
        payload = jwt_service.decode_access_token(token)
        raw_subject = payload.get("user_id") or payload.get("sub")  # Handle both formats
        if raw_subject is None:
            logger.warning("Token payload missing user_id/sub claim")
            raise credentials_exception

        user_id_int: Optional[int] = None
        if isinstance(raw_subject, int):
            user_id_int = raw_subject
        elif isinstance(raw_subject, str):
            try:
                user_id_int = int(raw_subject)
            except ValueError:
                user_id_int = None
        else:
            # Leave as-is; downstream lookups will attempt to resolve
            user_id_int = None
        token_org_ids = _coerce_int_list(payload.get("org_ids"))
        token_team_ids = _coerce_int_list(payload.get("team_ids"))
        token_active_org_id = payload.get("active_org_id")
        token_active_team_id = payload.get("active_team_id")
    except (InvalidTokenError, TokenExpiredError) as e:
        logger.warning(f"Token validation failed: {e}")
        raise credentials_exception from e
    except _USER_DB_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Unexpected error decoding token: {e}")
        raise credentials_exception from e

    # Enforce scope checks for scoped tokens to prevent privilege expansion.
    try:
        scoped_claims = (
            "scope",
            "allowed_endpoints",
            "allowed_methods",
            "allowed_paths",
            "max_calls",
            "max_runs",
            "schedule_id",
        )
        has_scoped_claim = any(payload.get(claim) is not None for claim in scoped_claims)
    except _USER_DB_NONCRITICAL_EXCEPTIONS:
        has_scoped_claim = False

    if has_scoped_claim:
        def _route_declares_scope_enforcement(req: Request) -> bool:
            try:
                route = getattr(req, "scope", {}).get("route") if req is not None else None
                dependant = getattr(route, "dependant", None)
                if dependant is None:
                    return False
                stack = list(getattr(dependant, "dependencies", []) or [])
                while stack:
                    dep = stack.pop()
                    call = getattr(dep, "call", None)
                    if getattr(call, "_tldw_token_scope", False):
                        return True
                    stack.extend(getattr(dep, "dependencies", []) or [])
            except _USER_DB_NONCRITICAL_EXCEPTIONS:
                return False
            return False

        try:
            scope_enforced = bool(getattr(request.state, "_token_scope_enforced", False))
        except _USER_DB_NONCRITICAL_EXCEPTIONS:
            scope_enforced = False
        if not scope_enforced and not _route_declares_scope_enforcement(request):
            if pii_redact_logs:
                logger.warning("Scoped token used without scope enforcement (details redacted)")
            else:
                logger.warning("Scoped token used without scope enforcement for subject {}", raw_subject)
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Scoped token requires endpoint scope enforcement",
            )

    if pii_redact_logs:
        logger.debug("Token decoded successfully for authenticated subject (redacted)")
    else:
        logger.debug(f"Token decoded successfully for subject: {raw_subject}")

    # Enforce blacklist revocation (fail-closed)
    try:
        session_manager = await get_session_manager()
        if await session_manager.is_token_blacklisted(token, payload.get("jti")):
            logger.warning("Token has been revoked according to blacklist/session state")
            raise credentials_exception
    except HTTPException:
        raise
    except _USER_DB_NONCRITICAL_EXCEPTIONS as exc:
        logger.error(f"Error checking token blacklist: {exc}")
        raise credentials_exception from exc

    # --- Fetch and Validate User Data ---
    subject_identifier = user_id_int if user_id_int is not None else raw_subject
    user_data: Optional[dict] = None
    try:
        repo = await AuthnzUsersRepo.from_pool()
        if user_id_int is not None:
            user_data = await repo.get_user_by_id(user_id_int)
        else:
            identifier_str = str(raw_subject)
            user_data = await repo.get_user_by_uuid(identifier_str)
            if not user_data and payload.get("username"):
                # Fallback to username claim when UUID lookup misses
                user_data = await repo.get_user_by_username(str(payload["username"]))

        if not user_data:
            if pii_redact_logs:
                logger.warning("User record for authenticated subject not found.")
            else:
                logger.warning(f"User record for subject '{subject_identifier}' not found.")
            raise credentials_exception

        if not isinstance(user_data, dict):
            data_type = type(user_data)
            if pii_redact_logs:
                logger.error(f"Data retrieved for authenticated subject is not a dictionary (type: {data_type}).")
            else:
                logger.error(f"Data retrieved for subject {subject_identifier} is not a dictionary (type: {data_type}).")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal error retrieving user data format."
            )

    except HTTPException:
        raise
    except _USER_DB_NONCRITICAL_EXCEPTIONS as e:
        if pii_redact_logs:
            logger.error("Error fetching user (details redacted) from AuthNZ user store", exc_info=True)
        else:
            logger.error(
                f"Error fetching user {subject_identifier} from AuthNZ user store: {e}",
                exc_info=True,
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving user information."
        ) from e

    # Prepare numeric ID (if available) for downstream lookups
    subject_db_id_raw = user_data.get("id")
    try:
        subject_db_id_int = int(subject_db_id_raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        subject_db_id_int = None

    # --- Enrich with roles/permissions from central AuthNZ RBAC tables ---
    roles, perms, is_admin = _enrich_user_with_rbac(
        subject_db_id_int, user_data, pii_redact_logs=pii_redact_logs
    )

    # --- Create and validate the User Pydantic model ---
    try:
        user = User(**{**user_data, "roles": roles, "permissions": perms, "is_admin": is_admin})
    except ValidationError as e:  # Catch Pydantic validation errors specifically
        if pii_redact_logs:
            logger.error("Failed to validate user data for authenticated user into User model (details redacted)", exc_info=True)
        else:
            logger.error(f"Failed to validate user data for user {subject_identifier} into User model: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing user data: Invalid format."
        ) from e
    except _USER_DB_NONCRITICAL_EXCEPTIONS as e:  # Catch other potential errors during model creation
        if pii_redact_logs:
            logger.error("Unexpected error creating User model for authenticated user (details redacted)", exc_info=True)
        else:
            logger.error(f"Unexpected error creating User model for user {subject_identifier}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal error processing user data."
        ) from e

    # --- Final User Status Check ---
    if not user.is_active:
        if pii_redact_logs:
            logger.warning("Authentication attempt by inactive user (details redacted)")
        else:
            logger.warning(f"Authentication attempt by inactive user: {user.username} (ID: {user.id})")
        raise InactiveUserError("Inactive user")

    # Attach user id for downstream context (usage logging, RBAC rate limits)
    with contextlib.suppress(_USER_DB_NONCRITICAL_EXCEPTIONS):
        request.state.user_id = user.id

    team_ids: list[int] = []
    org_ids: list[int] = []
    active_team_id: Optional[int] = None
    active_org_id: Optional[int] = None
    membership_error_detail = "Token membership could not be validated"
    try:
        membership_lookup_id = user.id_int
        if membership_lookup_id is None:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=membership_error_detail,
            )

        memberships = await list_memberships_for_user(membership_lookup_id)
        member_team_ids = _coerce_int_list(
            [m.get("team_id") for m in memberships if m.get("team_id") is not None]
        )
        member_org_ids = sorted(
            set(_coerce_int_list([m.get("org_id") for m in memberships if m.get("org_id") is not None]))
        )
        team_to_org: dict[int, int] = {}
        for m in memberships:
            if m.get("team_id") is None or m.get("org_id") is None:
                continue
            try:
                team_to_org[int(m["team_id"])] = int(m["org_id"])
            except (TypeError, ValueError):
                continue

        member_team_set = set(member_team_ids)
        member_org_set = set(member_org_ids)

        if token_team_ids:
            missing_teams = [tid for tid in token_team_ids if tid not in member_team_set]
            if missing_teams:
                if pii_redact_logs:
                    logger.warning("Token team memberships are no longer valid (details redacted)")
                else:
                    logger.warning(
                        f"Token team IDs not permitted for user {subject_identifier}: {missing_teams}"
                    )
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Token membership is no longer valid",
                )
        if token_org_ids:
            missing_orgs = [oid for oid in token_org_ids if oid not in member_org_set]
            if missing_orgs:
                if pii_redact_logs:
                    logger.warning("Token org memberships are no longer valid (details redacted)")
                else:
                    logger.warning(
                        f"Token org IDs not permitted for user {subject_identifier}: {missing_orgs}"
                    )
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Token membership is no longer valid",
                )

        if token_team_ids:
            team_ids = list(token_team_ids)
            if token_org_ids:
                org_ids = list(token_org_ids)
            else:
                org_ids = sorted(
                    {team_to_org.get(tid) for tid in team_ids if team_to_org.get(tid) is not None}
                )
        elif token_org_ids:
            org_ids = list(token_org_ids)
            token_org_set = set(org_ids)
            team_ids = [tid for tid in member_team_ids if team_to_org.get(tid) in token_org_set]
        else:
            team_ids = list(member_team_ids)
            org_ids = list(member_org_ids)

        active_team_claim = None
        if token_active_team_id is not None:
            try:
                active_team_claim = int(token_active_team_id)
            except (TypeError, ValueError):
                active_team_claim = None
        if active_team_claim is not None and active_team_claim not in team_ids:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Token active team is no longer valid",
            )
        active_org_claim = None
        if token_active_org_id is not None:
            try:
                active_org_claim = int(token_active_org_id)
            except (TypeError, ValueError):
                active_org_claim = None
        if active_org_claim is not None and active_org_claim not in org_ids:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Token active organization is no longer valid",
            )

        active_team_id = _normalize_active_id(token_active_team_id, team_ids)
        active_org_id = _normalize_active_id(token_active_org_id, org_ids)
    except HTTPException:
        raise
    except _USER_DB_NONCRITICAL_EXCEPTIONS as membership_exc:
        if pii_redact_logs:
            logger.warning("JWT membership validation failed (details redacted)")
        else:
            logger.warning(
                f"JWT membership validation failed for subject {subject_identifier}: {membership_exc}"
            )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=membership_error_detail,
        ) from membership_exc

    try:
        scoped_result = await apply_scoped_permissions(
            user_id=user.id_int,
            base_permissions=list(user.permissions or []),
            org_ids=org_ids,
            team_ids=team_ids,
            active_org_id=active_org_id,
            active_team_id=active_team_id,
        )
    except HTTPException:
        raise
    except _USER_DB_NONCRITICAL_EXCEPTIONS as scope_exc:
        if pii_redact_logs:
            logger.warning("JWT scoped permission application failed (details redacted)")
        else:
            logger.warning(
                f"JWT scoped permission application failed for subject {subject_identifier}: {scope_exc}"
            )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=membership_error_detail,
        ) from scope_exc

    user.permissions = list(scoped_result.permissions or [])
    active_org_id = scoped_result.active_org_id
    active_team_id = scoped_result.active_team_id
    try:
        request.state.team_ids = team_ids
        request.state.org_ids = org_ids
        request.state.active_team_id = active_team_id
        request.state.active_org_id = active_org_id
    except _USER_DB_NONCRITICAL_EXCEPTIONS:
        pass

    try:
        set_scope(
            user_id=user.id_int,
            org_ids=org_ids,
            team_ids=team_ids,
            active_org_id=active_org_id,
            active_team_id=active_team_id,
            is_admin=bool(user.is_admin),
        )
    except _USER_DB_NONCRITICAL_EXCEPTIONS as scope_exc:
        if pii_redact_logs:
            logger.debug(f"Failed to set scope context for authenticated user (redacted): {scope_exc}")
        else:
            logger.debug(f"Failed to set scope context for user {user.id}: {scope_exc}")

    # Populate AuthContext for compatibility with the new principal model
    try:
        principal = AuthPrincipal(
            kind="user",
            user_id=user.id_int,
            api_key_id=None,
            username=getattr(user, "username", None),
            email=getattr(user, "email", None),
            subject=None,
            token_type="access",
            jti=None,
            roles=list(user.roles or []),
            permissions=list(user.permissions or []),
            is_admin=bool(user.is_admin),
            org_ids=org_ids,
            team_ids=team_ids,
            active_org_id=active_org_id,
            active_team_id=active_team_id,
        )
        ip = request.client.host if getattr(request, "client", None) else None
        user_agent = request.headers.get("User-Agent") if getattr(request, "headers", None) else None
        request_id = (
            request.headers.get("X-Request-ID") if getattr(request, "headers", None) else None
        ) or getattr(request.state, "request_id", None)
        request.state.auth = AuthContext(
            principal=principal,
            ip=ip,
            user_agent=user_agent,
            request_id=request_id,
        )
    except _USER_DB_NONCRITICAL_EXCEPTIONS:
        logger.exception("Unable to populate AuthContext in verify_jwt_and_fetch_user")

    if pii_redact_logs:
        logger.info("Authenticated active user (details redacted)")
    else:
        logger.info(f"Authenticated active user: {user.username} (ID: {user.id})")
    return user


# --- Combined Primary Authentication Dependency ---


async def authenticate_api_key_user(request: Request, api_key: str) -> User:
    """
    Validate an API key and return the associated User.

    This helper centralizes API-key authentication so that both legacy
    dependencies (get_request_user) and the AuthPrincipal resolver can
    share the same behavior and context population. In single-user mode,
    it also supports the bootstrapped SINGLE_USER_API_KEY /
    SINGLE_USER_TEST_API_KEY without requiring a backing AuthNZ database
    row.
    """
    settings = get_settings()

    # Single-user compatibility: treat the configured single-user API key(s)
    # as an admin-style principal without touching the AuthNZ API key store.
    try:
        if getattr(settings, "AUTH_MODE", None) == "single_user":
            allowed_keys: set[str] = set()
            primary_key = getattr(settings, "SINGLE_USER_API_KEY", None)
            if primary_key:
                allowed_keys.add(primary_key)
            if _is_test_context():
                test_key = os.getenv("SINGLE_USER_TEST_API_KEY")
                if test_key:
                    allowed_keys.add(test_key)
            if api_key in allowed_keys:
                client_ip = resolve_client_ip(request, settings)
                if not is_single_user_ip_allowed(client_ip, settings):
                    if settings.PII_REDACT_LOGS:
                        logger.warning("Single-user API key rejected due to client IP allowlist (details redacted)")
                    else:
                        logger.warning(
                            "Single-user API key rejected due to client IP allowlist (ip={})",
                            client_ip,
                        )
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Invalid or missing API Key",
                    )
                user = get_single_user_instance()
                # Ensure admin-style claims are present
                if not user.roles:
                    user.roles = ["admin"]
                if not user.permissions:
                    default_permissions = list(
                        getattr(settings, "SINGLE_USER_DEFAULT_PERMISSIONS", []) or []
                    )
                    user.permissions = default_permissions or ["system.configure", "media.read", "media.create"]
                user.is_admin = True
                # Attach minimal context for downstream consumers
                try:
                    request.state.user_id = user.id
                    request.state.api_key_id = None
                    request.state.team_ids = []
                    request.state.org_ids = []
                except _USER_DB_NONCRITICAL_EXCEPTIONS:
                    pass
                with contextlib.suppress(_USER_DB_NONCRITICAL_EXCEPTIONS):
                    set_scope(
                        user_id=user.id_int,
                        org_ids=[],
                        team_ids=[],
                        is_admin=True,
                    )
                try:
                    principal = AuthPrincipal(
                        kind="user",
                        user_id=user.id_int,
                        api_key_id=None,
                        username=getattr(user, "username", None),
                        email=getattr(user, "email", None),
                        subject="single_user",
                        token_type="api_key",
                        jti=None,
                        roles=list(user.roles or []),
                        permissions=list(user.permissions or []),
                        is_admin=True,
                        org_ids=[],
                        team_ids=[],
                    )
                    ip = request.client.host if getattr(request, "client", None) else None
                    user_agent = (
                        request.headers.get("User-Agent")
                        if getattr(request, "headers", None)
                        else None
                    )
                    request_id = (
                        request.headers.get("X-Request-ID")
                        if getattr(request, "headers", None)
                        else None
                    ) or getattr(request.state, "request_id", None)
                    request.state.auth = AuthContext(
                        principal=principal,
                        ip=ip,
                        user_agent=user_agent,
                        request_id=request_id,
                    )
                    request.state._auth_user = user
                    with contextlib.suppress(_USER_DB_NONCRITICAL_EXCEPTIONS):
                        request.state.user_id = user.id_int
                except _USER_DB_NONCRITICAL_EXCEPTIONS:
                    logger.debug("Unable to populate AuthContext for single-user API key")
                return user
            logger.warning("Single-user mode received a non-matching API key.")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API Key",
            )
    except HTTPException:
        # Preserve explicit auth failures (e.g., IP allowlist rejection).
        raise
    except _USER_DB_NONCRITICAL_EXCEPTIONS as single_exc:
        logger.debug(
            "authenticate_api_key_user: single-user API key path failed; falling back to multi-user flow: {}",
            single_exc,
        )

    try:
        api_mgr = await get_api_key_manager()
        client_ip = resolve_client_ip(request, settings)

        usage_details: dict[str, Any] | None = None
        try:
            endpoint_id = getattr(request.state, "_auth_endpoint_id", None)
            action = getattr(request.state, "_auth_action", None)
            scope_name = getattr(request.state, "_auth_scope_name", None)
            if endpoint_id is not None or action is not None or scope_name is not None:
                usage_details = {}
                if endpoint_id is not None:
                    usage_details["endpoint_id"] = str(endpoint_id)
                if action is not None:
                    usage_details["action"] = str(action)
                if scope_name is not None:
                    usage_details["scope"] = str(scope_name)
                path = getattr(getattr(request, "url", None), "path", None) or getattr(request, "scope", {}).get("path")
                if path:
                    usage_details["path"] = str(path)
                method = getattr(request, "method", None)
                if method:
                    usage_details["method"] = str(method).upper()
        except _USER_DB_NONCRITICAL_EXCEPTIONS:
            usage_details = None

        if usage_details is None:
            key_info = await api_mgr.validate_api_key(api_key=api_key, ip_address=client_ip)
        else:
            try:
                key_info = await api_mgr.validate_api_key(
                    api_key=api_key,
                    ip_address=client_ip,
                    usage_details=usage_details,
                )
            except TypeError:
                key_info = await api_mgr.validate_api_key(api_key=api_key, ip_address=client_ip)
        if not key_info:
            logger.warning("Multi-User Mode: Invalid X-API-KEY presented.")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
            )

        user_id = key_info.get("user_id")
        if not isinstance(user_id, int):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
            )

        from tldw_Server_API.app.core.AuthNZ.repos.users_repo import AuthnzUsersRepo

        user_data = None
        try:
            users_repo = await AuthnzUsersRepo.from_pool()
            user_data = await users_repo.get_user_by_id(user_id)
        except _USER_DB_NONCRITICAL_EXCEPTIONS:
            if not _is_test_context():
                raise

        if not user_data and _is_test_context():
            try:
                from tldw_Server_API.app.core.DB_Management.Users_DB import (
                    get_user_by_id as legacy_get_user_by_id,
                )

                user_data = await legacy_get_user_by_id(user_id)
            except _USER_DB_NONCRITICAL_EXCEPTIONS:
                user_data = None
        if not user_data:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
            )

        # Normalize active flag
        is_active_value = user_data.get("is_active", True)
        is_active_normalized = bool(is_active_value)
        user_data["is_active"] = is_active_normalized
        if not is_active_normalized:
            if settings.PII_REDACT_LOGS:
                logger.warning("Authentication attempt by inactive user (API key)")
            else:
                logger.warning(
                    f"Authentication attempt by inactive user (API key): {user_data.get('username', user_id)}"
                )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Inactive user",
            )

        if user_data.get("is_superuser"):
            user_data.setdefault("is_admin", True)

        roles, perms, is_admin_flag = _enrich_user_with_rbac(
            user_id, user_data, pii_redact_logs=getattr(settings, "PII_REDACT_LOGS", False)
        )

        user_data["roles"] = roles
        user_data["permissions"] = perms
        user_data["is_admin"] = bool(is_admin_flag)

        def _coerce_int(value: Any) -> Optional[int]:
            try:
                if value is None:
                    return None
                return int(value)
            except (TypeError, ValueError):
                return None

        key_org_id = _coerce_int(key_info.get("org_id"))
        key_team_id = _coerce_int(key_info.get("team_id"))
        if key_org_id is None and key_team_id is None and key_info.get("id") is not None:
            try:
                from tldw_Server_API.app.core.AuthNZ.repos.api_keys_repo import AuthnzApiKeysRepo

                db_pool = await get_db_pool()
                api_keys_repo = AuthnzApiKeysRepo(db_pool)
                row = await api_keys_repo.fetch_key_for_user(
                    key_id=int(key_info["id"]),
                    user_id=int(user_id),
                )
                if row:
                    key_org_id = _coerce_int(row.get("org_id"))
                    key_team_id = _coerce_int(row.get("team_id"))
            except Exception as exc:
                logger.debug("API key scope fallback lookup failed: {}", exc)

        # Attach context for downstream consumers
        try:
            request.state.user_id = user_id
            request.state.api_key_id = key_info.get("id")
            # Store scope for require_api_key_scope() dependency enforcement
            request.state._api_key_scope = key_info.get("scope", "read")
            # Attach org/team context if present (virtual keys)
            try:
                if key_org_id is not None:
                    request.state.org_id = key_org_id
                if key_team_id is not None:
                    request.state.team_id = key_team_id
            except _USER_DB_NONCRITICAL_EXCEPTIONS as e:
                logger.debug(f"Unable to attach org/team context: {e}")
        except _USER_DB_NONCRITICAL_EXCEPTIONS as ctx_state_exc:
            logger.debug(f"Unable to attach user/api_key context to request.state: {ctx_state_exc}")

        user_obj = User(**user_data)

        team_ids: list[int] = []
        org_ids: list[int] = []
        active_team_id: Optional[int] = None
        active_org_id: Optional[int] = None
        memberships: list[dict[str, Any]] = []
        try:
            memberships = await list_memberships_for_user(int(user_id))
        except _USER_DB_NONCRITICAL_EXCEPTIONS as memberships_exc:
            logger.debug(f"Membership lookup failed for user {user_id}: {memberships_exc}")

        member_team_ids = [
            m.get("team_id")
            for m in memberships
            if m.get("team_id") is not None
        ]
        member_org_ids = sorted(
            {m.get("org_id") for m in memberships if m.get("org_id") is not None}
        )
        team_to_org = {
            int(m["team_id"]): int(m["org_id"])
            for m in memberships
            if m.get("team_id") is not None and m.get("org_id") is not None
        }

        if key_team_id is not None:
            if key_team_id not in [int(t) for t in member_team_ids if t is not None]:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="API key team scope is not permitted",
                )
            team_ids = [key_team_id]
            org_for_team = team_to_org.get(key_team_id)
            if org_for_team is None:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="API key team scope could not be resolved",
                )
            if key_org_id is not None and key_org_id != org_for_team:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="API key org scope does not match team scope",
                )
            org_ids = [org_for_team]
        elif key_org_id is not None:
            if key_org_id not in [int(o) for o in member_org_ids if o is not None]:
                try:
                    org_memberships = await list_org_memberships_for_user(int(user_id))
                    org_membership_ids = {
                        int(m.get("org_id"))
                        for m in org_memberships
                        if m.get("org_id") is not None
                    }
                except _USER_DB_NONCRITICAL_EXCEPTIONS:
                    org_membership_ids = set()
                if key_org_id not in org_membership_ids:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="API key org scope is not permitted",
                    )
            org_ids = [key_org_id]
            team_ids = [
                int(tid)
                for tid in member_team_ids
                if tid is not None and team_to_org.get(int(tid)) == key_org_id
            ]
        else:
            team_ids = [int(tid) for tid in member_team_ids if tid is not None]
            org_ids = [int(oid) for oid in member_org_ids if oid is not None]

        active_team_id = _normalize_active_id(None, team_ids)
        active_org_id = _normalize_active_id(None, org_ids)
        scoped_result = await apply_scoped_permissions(
            user_id=user_obj.id_int,
            base_permissions=list(user_obj.permissions or []),
            org_ids=org_ids,
            team_ids=team_ids,
            active_org_id=active_org_id,
            active_team_id=active_team_id,
        )
        user_obj.permissions = list(scoped_result.permissions or [])
        active_org_id = scoped_result.active_org_id
        active_team_id = scoped_result.active_team_id

        try:
            request.state.team_ids = team_ids
            request.state.org_ids = org_ids
            request.state.active_team_id = active_team_id
            request.state.active_org_id = active_org_id
            if org_ids:
                request.state.org_id = org_ids[0]
            if team_ids:
                request.state.team_id = team_ids[0]
        except _USER_DB_NONCRITICAL_EXCEPTIONS as team_ctx_exc:
            logger.debug(f"Unable to attach team/org ids to request.state: {team_ctx_exc}")

        try:
            set_scope(
                user_id=user_obj.id_int,
                org_ids=org_ids,
                team_ids=team_ids,
                active_org_id=active_org_id,
                active_team_id=active_team_id,
                is_admin=bool(user_obj.is_admin),
            )
        except _USER_DB_NONCRITICAL_EXCEPTIONS as scope_exc:
            logger.debug(
                f"Scope context setup failed for API key user {user_id}: {scope_exc}"
            )

        # Populate AuthContext for API-key-based principals
        try:
            api_key_id_val = None
            raw_key_id = key_info.get("id")
            if raw_key_id is not None:
                try:
                    api_key_id_val = int(raw_key_id)
                except _USER_DB_NONCRITICAL_EXCEPTIONS:
                    api_key_id_val = None

            subject_val: Optional[str] = None
            try:
                if getattr(settings, "AUTH_MODE", None) == "single_user":
                    single_id = getattr(settings, "SINGLE_USER_FIXED_ID", None)
                    if single_id is not None and user_obj.id_int == int(single_id):
                        subject_val = "single_user"
            except _USER_DB_NONCRITICAL_EXCEPTIONS:
                subject_val = None

            principal = AuthPrincipal(
                kind="api_key",
                user_id=user_obj.id_int,
                api_key_id=api_key_id_val,
                username=getattr(user_obj, "username", None),
                email=getattr(user_obj, "email", None),
                subject=subject_val,
                token_type="api_key",
                jti=None,
                roles=list(user_obj.roles or []),
                permissions=list(user_obj.permissions or []),
                is_admin=bool(user_obj.is_admin),
                org_ids=org_ids,
                team_ids=team_ids,
                active_org_id=active_org_id,
                active_team_id=active_team_id,
            )
            ip = request.client.host if getattr(request, "client", None) else None
            user_agent = (
                request.headers.get("User-Agent")
                if getattr(request, "headers", None)
                else None
            )
            request_id = (
                request.headers.get("X-Request-ID")
                if getattr(request, "headers", None)
                else None
            ) or getattr(request.state, "request_id", None)

            request.state.auth = AuthContext(
                principal=principal,
                ip=ip,
                user_agent=user_agent,
                request_id=request_id,
            )
            # Cache the resolved user for downstream adapters
            try:
                request.state._auth_user = user_obj
            except _USER_DB_NONCRITICAL_EXCEPTIONS as cache_exc:
                logger.debug(f"Failed to cache _auth_user on request.state: {cache_exc}")
        except _USER_DB_NONCRITICAL_EXCEPTIONS as ctx_exc:
            logger.debug(
                f"Unable to populate AuthContext in authenticate_api_key_user: {ctx_exc}"
            )

        return user_obj
    except HTTPException:
        raise
    except _USER_DB_NONCRITICAL_EXCEPTIONS as e:
        # Avoid logging exception messages directly to prevent leaking secrets
        # from API key validation paths; log only the exception type.
        logger.error(
            "Error validating API key in multi-user mode (type={})",
            type(e).__name__,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication failed due to internal error",
        ) from e


def _coerce_user_id_for_log(user: User) -> Optional[int]:
    with contextlib.suppress(_USER_DB_NONCRITICAL_EXCEPTIONS):
        if getattr(user, "id_int", None) is not None:
            return int(user.id_int)  # type: ignore[arg-type]
    with contextlib.suppress(_USER_DB_NONCRITICAL_EXCEPTIONS):
        return int(user.id)  # type: ignore[arg-type]
    return None


def _warn_single_user_context_mismatch(
    request: Request,
    user: User,
    auth_source: str,
) -> None:
    try:
        settings = get_settings()
        if getattr(settings, "AUTH_MODE", None) != "single_user":
            return

        expected_id = int(getattr(settings, "SINGLE_USER_FIXED_ID", 1))
        resolved_id = _coerce_user_id_for_log(user)
        if resolved_id is None or resolved_id == expected_id:
            return

        has_auth_header = bool(
            request.headers.get("Authorization")
            if getattr(request, "headers", None)
            else False
        )
        has_api_key_header = bool(
            request.headers.get("X-API-KEY")
            if getattr(request, "headers", None)
            else False
        )

        logger.warning(
            "Single-user mode resolved unexpected principal via {}: resolved_user_id={}, "
            "expected_user_id={}, has_authorization_header={}, has_api_key_header={}",
            auth_source,
            resolved_id,
            expected_id,
            has_auth_header,
            has_api_key_header,
        )
    except _USER_DB_NONCRITICAL_EXCEPTIONS as e:
        logger.debug(
            "Could not emit single-user context mismatch warning (source={}): {}",
            auth_source,
            e,
        )


async def get_request_user(
    request: Request,
    api_key: Optional[str] = Header(None, alias="X-API-KEY"),
    token: Optional[str] = Depends(oauth2_scheme),  # Bearer from Authorization
    legacy_token_header: Optional[str] = Header(None, alias="Token"),  # Deprecated legacy header (ignored)
) -> User:
    """
    Unified authentication dependency for endpoints that require a User.

    Behavior:
    - Uses the shared JWT/API-key helpers (verify_jwt_and_fetch_user,
      authenticate_api_key_user) for all deployments.
    - Does not branch on AUTH_MODE; single-user deployments authenticate
      via the same AuthNZ tables and RBAC as multi-user, with the
      bootstrapped admin treated as a normal user with roles/permissions.
    - Treats non-JWT Bearer tokens as API keys for compatibility.
    """
    # Test-mode bypasses are disabled in production for safety
    try:
        import os as _os
        _prod = _is_production_like_env()
        # Test-mode bypass for evaluations when admin gating is explicitly disabled
        if (
            not _prod
            and env_flag_enabled("TESTING")
            and not is_truthy(_os.getenv("EVALS_HEAVY_ADMIN_ONLY", "true"))
            and _is_strict_test_bypass_context()
        ):
            logger.info(
                "TESTING with EVALS_HEAVY_ADMIN_ONLY disabled: "
                "bypassing auth, returning single-user test instance"
            )
            return get_single_user_instance()
    except _USER_DB_NONCRITICAL_EXCEPTIONS as env_exc:
        logger.debug(
            f"get_request_user: test-mode bypass env detection failed; continuing with normal auth: {env_exc}"
        )

    # Warn if test flags are present in production deployments
    try:
        import os as _os
        _prod = _is_production_like_env()
        if _prod and (
            is_test_mode()
            or env_flag_enabled("TESTING")
        ) and not getattr(get_request_user, "_warned_testflags_prod", False):
            logger.warning(
                "TEST flags detected while tldw_production=true; "
                "test-only auth bypasses are disabled."
            )
            get_request_user._warned_testflags_prod = True
    except _USER_DB_NONCRITICAL_EXCEPTIONS as warn_exc:
        logger.debug(
            f"get_request_user: production test-flag warning emission failed; continuing without warning: {warn_exc}"
        )

    # Fast-path: if an AuthPrincipal has already been resolved, reuse the cached
    # _auth_user instead of re-running authentication logic.
    try:
        existing_ctx = getattr(request.state, "auth", None)
        cached_user = getattr(request.state, "_auth_user", None)
        if isinstance(existing_ctx, AuthContext) and isinstance(cached_user, User):
            logger.debug("get_request_user: Reusing cached AuthPrincipal/_auth_user from request.state.")
            return cached_user
    except _USER_DB_NONCRITICAL_EXCEPTIONS as fastpath_exc:
        # Fall through to normal auth paths on any issues, but make failures observable.
        logger.debug(
            f"get_request_user fast-path reuse failed; falling back to normal auth: {fastpath_exc}"
        )

    # Legacy Token header is intentionally ignored (no longer supported).
    _ = legacy_token_header

    # Prefer Bearer JWT when present; in single-user mode treat Bearer as API key.
    if token:
        try:
            settings = get_settings()
        except _USER_DB_NONCRITICAL_EXCEPTIONS:
            settings = None
        token_is_jwt = _looks_like_jwt(token)
        if settings is not None and getattr(settings, "AUTH_MODE", None) == "single_user":
            logger.debug("get_request_user: Treating Bearer token as API key in single-user mode.")
            user = await authenticate_api_key_user(request, token)
            _warn_single_user_context_mismatch(request, user, "bearer_single_user")
            return user
        if not token_is_jwt:
            logger.debug("get_request_user: Treating Bearer token as API key (non-JWT token).")
            user = await authenticate_api_key_user(request, token)
            _warn_single_user_context_mismatch(request, user, "bearer_non_jwt")
            return user
        logger.debug("get_request_user: Attempting JWT-based authentication.")
        try:
            user = await verify_jwt_and_fetch_user(request, token)
        except InactiveUserError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Inactive user",
            ) from exc
        # verify_jwt_and_fetch_user already sets request.state.auth; cache user for fast-path reuse.
        try:
            request.state._auth_user = user
        except _USER_DB_NONCRITICAL_EXCEPTIONS as cache_exc:
            logger.debug(f"Failed to cache _auth_user on request.state: {cache_exc}")
        _warn_single_user_context_mismatch(request, user, "bearer_jwt")
        return user

    if api_key:
        logger.debug("get_request_user: Attempting API-key-based authentication.")
        user = await authenticate_api_key_user(request, api_key)
        _warn_single_user_context_mismatch(request, user, "x_api_key")
        return user

    # Neither Bearer token nor API key provided
    logger.warning(
        "get_request_user: No credentials provided (missing Bearer token or X-API-KEY)."
    )
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not authenticated (provide Bearer token or X-API-KEY)",
        headers={"WWW-Authenticate": "Bearer"},
    )



#
# End of User_DB_Handling.py
#######################################################################################################################
