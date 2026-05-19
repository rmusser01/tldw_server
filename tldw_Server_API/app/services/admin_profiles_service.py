"""
Admin user profile orchestration helpers.
"""
from __future__ import annotations

import os
import time
from typing import Any

from fastapi import HTTPException, status
from fastapi.responses import JSONResponse
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.org_deps import ROLE_HIERARCHY
from tldw_Server_API.app.api.v1.schemas.user_profile_schemas import (
    UserProfileBatchResponse,
    UserProfileBulkUpdateDiff,
    UserProfileBulkUpdateRequest,
    UserProfileBulkUpdateResponse,
    UserProfileBulkUpdateUserResult,
    UserProfileErrorDetail,
    UserProfileErrorResponse,
    UserProfileResponse,
    UserProfileUpdateEntry,
    UserProfileUpdateError,
    UserProfileUpdateRequest,
    UserProfileUpdateResponse,
)
from tldw_Server_API.app.api.v1.utils.pagination import build_page_pagination_meta
from tldw_Server_API.app.api.v1.utils.profile_errors import classify_profile_update_skips
from tldw_Server_API.app.core.AuthNZ.api_key_manager import get_api_key_manager
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
from tldw_Server_API.app.core.AuthNZ.exceptions import UserNotFoundError
from tldw_Server_API.app.core.AuthNZ.orgs_teams import (
    get_team,
    list_memberships_for_user,
    list_org_memberships_for_user,
    list_team_members,
)
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal, is_single_user_principal
from tldw_Server_API.app.core.AuthNZ.repos.users_repo import AuthnzUsersRepo
from tldw_Server_API.app.core.config import load_comprehensive_config
from tldw_Server_API.app.core.UserProfiles.service import UserProfileService
from tldw_Server_API.app.core.UserProfiles.update_service import (
    ProfileUpdateScope,
    UserProfileUpdateService,
)
from tldw_Server_API.app.core.UserProfiles.user_profile_catalog import load_user_profile_catalog
from tldw_Server_API.app.services import admin_scope_service

REQUIRED_ADMIN_RANK = ROLE_HIERARCHY.get("admin", 3)
REQUIRED_TEAM_ADMIN_RANK = ROLE_HIERARCHY.get("lead", 2)
_PLATFORM_ADMIN_ROLES = frozenset({"admin", "owner", "super_admin"})
_ADMIN_CLAIM_PERMISSIONS = frozenset({"*", "system.configure"})


def _role_rank(role: str | None) -> int:
    """Map role strings to numeric rank using the shared hierarchy."""
    return admin_scope_service.role_rank(role)


def _normalized_claim_values(values: list[Any] | tuple[Any, ...] | set[Any] | None) -> set[str]:
    return {
        str(value).strip().lower()
        for value in (values or [])
        if str(value).strip()
    }


def _principal_has_platform_admin_claims(principal: AuthPrincipal) -> bool:
    roles = _normalized_claim_values(principal.roles)
    permissions = _normalized_claim_values(principal.permissions)
    if roles & _PLATFORM_ADMIN_ROLES:
        return True
    return bool(permissions & _ADMIN_CLAIM_PERMISSIONS)


def _derive_profile_update_roles(principal: AuthPrincipal) -> set[str]:
    """Build the role set used for profile update enforcement."""
    roles = {str(role).strip().lower() for role in (principal.roles or []) if role}
    has_platform_admin_claims = _principal_has_platform_admin_claims(principal)
    if has_platform_admin_claims or "admin" in roles:
        roles.add("admin")
        roles.update({"org_admin", "team_admin"})
    if has_platform_admin_claims or admin_scope_service.is_platform_admin(principal):
        roles.add("platform_admin")
    return roles


def _get_bulk_confirm_threshold() -> int:
    """Resolve the bulk update confirmation threshold from env/config."""
    raw_env = os.getenv("BULK_UPDATE_CONFIRM_THRESHOLD")
    if raw_env:
        try:
            return max(1, int(raw_env))
        except Exception:
            return 1000
    try:
        config_parser = load_comprehensive_config()
        for section in ("user_profile", "profile", "admin"):
            if config_parser and config_parser.has_section(section):
                raw_cfg = config_parser.get(section, "bulkUpdateConfirmThreshold", fallback="").strip()
                if raw_cfg:
                    return max(1, int(raw_cfg))
    except Exception as threshold_error:
        logger.bind(error_type=type(threshold_error).__name__).debug(
            "Failed to load bulk-update threshold from config; using default"
        )
    return 1000


def _coerce_bulk_candidate_user_id(user: Any) -> int | None:
    """Return a candidate user id, or None when a malformed repo row should be skipped."""
    try:
        return int(user.get("id"))
    except Exception as user_id_error:
        logger.bind(error_type=type(user_id_error).__name__).debug(
            "Skipping bulk user candidate with invalid id"
        )
        return None


def _profile_error_response(
    *,
    status_code: int,
    error_code: str,
    detail: str,
    errors: list[UserProfileErrorDetail] | None = None,
) -> JSONResponse:
    """Return a structured profile error response payload."""
    payload = UserProfileErrorResponse(
        error_code=error_code,
        detail=detail,
        errors=errors or [],
    )
    return JSONResponse(status_code=status_code, content=payload.model_dump())


def _mask_profile_diff_value(
    profile_service: UserProfileService,
    catalog_map: dict[str, Any],
    key: str,
    value: Any,
) -> Any:
    """Format and mask per-field diff values for bulk updates."""
    try:
        return profile_service._format_value(
            key,
            value,
            include_sources=False,
            source=None,
            catalog_map=catalog_map,
            mask_secrets=True,
        )
    except Exception:
        return value


async def _build_bulk_update_before_values(
    *,
    user_id: int,
    updates: list[UserProfileUpdateEntry],
    profile_service: UserProfileService,
    user_repo: AuthnzUsersRepo,
    catalog_map: dict[str, Any],
) -> dict[str, Any]:
    """Compute before-values for bulk update diffs."""
    keys = [entry.key for entry in updates]
    key_set = set(keys)
    before: dict[str, Any] = {}

    identity_keys = {
        "identity.email",
        "identity.role",
        "identity.is_active",
        "identity.is_verified",
        "identity.is_locked",
    }
    needs_user_record = bool(key_set & (identity_keys | {"limits.storage_quota_mb"}))
    user_row: dict[str, Any] | None = None
    if needs_user_record:
        user = await user_repo.get_user_by_id(int(user_id))
        if not user:
            return before
        user_row = dict(user)

    if user_row and (key_set & identity_keys):
        identity = profile_service._build_identity(user_row)
        await profile_service._attach_lockout_status(identity)
        identity_map = {
            "identity.email": identity.get("email"),
            "identity.role": identity.get("role"),
            "identity.is_active": identity.get("is_active"),
            "identity.is_verified": identity.get("is_verified"),
            "identity.is_locked": identity.get("is_locked"),
        }
        for key, value in identity_map.items():
            if key in key_set:
                before[key] = _mask_profile_diff_value(
                    profile_service,
                    catalog_map,
                    key,
                    value,
                )

    if user_row and "limits.storage_quota_mb" in key_set:
        before["limits.storage_quota_mb"] = _mask_profile_diff_value(
            profile_service,
            catalog_map,
            "limits.storage_quota_mb",
            user_row.get("storage_quota_mb"),
        )

    if any(key.startswith("memberships.") for key in key_set):
        org_memberships = await list_org_memberships_for_user(user_id)
        team_memberships = await list_memberships_for_user(user_id)
        org_roles = {
            int(m.get("org_id")): m.get("role")
            for m in org_memberships
            if m.get("org_id") is not None
        }
        team_roles = {
            int(m.get("team_id")): m.get("role")
            for m in team_memberships
            if m.get("team_id") is not None
        }

        for entry in updates:
            if entry.key == "memberships.orgs.role":
                if isinstance(entry.value, dict) and "org_id" in entry.value:
                    try:
                        org_id = int(entry.value.get("org_id"))
                    except (TypeError, ValueError):
                        continue
                    before_val = org_roles.get(org_id)
                    before[entry.key] = _mask_profile_diff_value(
                        profile_service,
                        catalog_map,
                        entry.key,
                        before_val,
                    )
            elif entry.key == "memberships.teams.role":
                if isinstance(entry.value, dict) and "team_id" in entry.value:
                    try:
                        team_id = int(entry.value.get("team_id"))
                    except (TypeError, ValueError):
                        continue
                    before_val = team_roles.get(team_id)
                    before[entry.key] = _mask_profile_diff_value(
                        profile_service,
                        catalog_map,
                        entry.key,
                        before_val,
                    )
            elif entry.key == "memberships.teams.member":
                if isinstance(entry.value, dict) and "team_id" in entry.value:
                    try:
                        team_id = int(entry.value.get("team_id"))
                    except (TypeError, ValueError):
                        continue
                    role = team_roles.get(team_id)
                    before_val = {"member": team_id in team_roles, "role": role}
                    before[entry.key] = _mask_profile_diff_value(
                        profile_service,
                        catalog_map,
                        entry.key,
                        before_val,
                    )

    needs_effective = any(
        key not in identity_keys
        and not key.startswith("memberships.")
        and key != "limits.storage_quota_mb"
        for key in key_set
    )
    if needs_effective:
        effective = await profile_service._build_effective_config(
            int(user_id),
            include_sources=False,
            mask_secrets=True,
        )
        for key in key_set:
            if key in before:
                continue
            if key in effective:
                before[key] = effective.get(key)

    return before


def _parse_user_id_list(raw: str | None) -> list[int] | None:
    """Parse a comma-separated list of user IDs."""
    if raw is None:
        return None
    values: list[int] = []
    for part in str(raw).split(","):
        piece = part.strip()
        if not piece:
            continue
        try:
            values.append(int(piece))
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="invalid_user_ids",
            ) from exc
    return values or None


class _ProfileAdminScope:
    """Resolved admin scope for profile batch operations."""

    def __init__(
        self,
        *,
        org_admin_ids: set[int] | None,
        team_admin_ids: set[int],
        team_admin_org_ids: set[int],
        team_admin_org_map: dict[int, int],
    ) -> None:
        self.org_admin_ids = org_admin_ids
        self.team_admin_ids = team_admin_ids
        self.team_admin_org_ids = team_admin_org_ids
        self.team_admin_org_map = team_admin_org_map

    @property
    def is_platform_admin(self) -> bool:
        return self.org_admin_ids is None


async def _get_profile_admin_scope(principal: AuthPrincipal) -> _ProfileAdminScope:
    """Resolve the principal's organization/team admin scope."""
    if is_single_user_principal(principal) or admin_scope_service.is_platform_admin(principal):
        return _ProfileAdminScope(
            org_admin_ids=None,
            team_admin_ids=set(),
            team_admin_org_ids=set(),
            team_admin_org_map={},
        )
    if principal.user_id is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to manage users",
        )
    org_memberships = await list_org_memberships_for_user(principal.user_id)
    org_admin_ids = {
        int(m.get("org_id"))
        for m in org_memberships
        if m.get("org_id") is not None
        and _role_rank(m.get("role")) >= REQUIRED_ADMIN_RANK
    }

    team_memberships = await list_memberships_for_user(principal.user_id)
    team_admin_ids: set[int] = set()
    team_admin_org_ids: set[int] = set()
    team_admin_org_map: dict[int, int] = {}
    for membership in team_memberships:
        team_id = membership.get("team_id")
        org_id = membership.get("org_id")
        if team_id is None or org_id is None:
            continue
        if _role_rank(membership.get("role")) < REQUIRED_TEAM_ADMIN_RANK:
            continue
        team_id_int = int(team_id)
        org_id_int = int(org_id)
        team_admin_ids.add(team_id_int)
        team_admin_org_ids.add(org_id_int)
        team_admin_org_map[team_id_int] = org_id_int

    if not org_admin_ids and not team_admin_ids:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to manage users",
        )

    return _ProfileAdminScope(
        org_admin_ids=org_admin_ids,
        team_admin_ids=team_admin_ids,
        team_admin_org_ids=team_admin_org_ids,
        team_admin_org_map=team_admin_org_map,
    )


async def _load_team_user_ids(team_ids: set[int]) -> set[int]:
    """Collect active user IDs from the specified teams."""
    user_ids: set[int] = set()
    for team_id in team_ids:
        members = await list_team_members(int(team_id))
        for member in members:
            if member.get("user_id") is None:
                continue
            status_val = member.get("status")
            if status_val is not None and str(status_val).lower() != "active":
                continue
            user_ids.add(int(member.get("user_id")))
    return user_ids


async def _load_bulk_user_candidates(
    *,
    principal: AuthPrincipal,
    org_id: int | None,
    team_id: int | None,
    role: str | None,
    is_active: bool | None,
    search: str | None,
    user_ids: list[int] | None,
) -> list[int]:
    """Resolve target user IDs for batch profile operations."""
    repo = await AuthnzUsersRepo.from_pool()
    scope = await _get_profile_admin_scope(principal)
    org_ids: list[int] | None = None
    team_user_ids: set[int] | None = None
    restrict_to_team_scope = False

    if org_id is not None:
        if not scope.is_platform_admin:
            if org_id not in scope.org_admin_ids and org_id not in scope.team_admin_org_ids:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Not authorized to access this organization",
                )
        org_ids = [org_id]
        if not scope.is_platform_admin and org_id not in scope.org_admin_ids:
            restrict_to_team_scope = True

    if team_id is not None:
        team = await get_team(team_id)
        if not team:
            raise HTTPException(status_code=404, detail="team_not_found")
        team_org_id = int(team.get("org_id"))
        if org_id is not None and team_org_id != int(org_id):
            return []
        if not scope.is_platform_admin:
            if team_id not in scope.team_admin_ids and team_org_id not in scope.org_admin_ids:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Not authorized to access this team",
                )
        team_user_ids = await _load_team_user_ids({int(team_id)})
        if org_ids is None:
            org_ids = [team_org_id]
        elif team_org_id not in org_ids:
            return []

    if not scope.is_platform_admin and org_ids is None:
        if scope.org_admin_ids:
            org_ids = sorted(scope.org_admin_ids)
        elif scope.team_admin_org_ids:
            org_ids = sorted(scope.team_admin_org_ids)
            restrict_to_team_scope = True

    if restrict_to_team_scope and team_user_ids is None:
        allowed_team_ids = set(scope.team_admin_ids)
        if org_id is not None:
            allowed_team_ids = {
                team_id for team_id, team_org in scope.team_admin_org_map.items()
                if team_org == int(org_id)
            }
        if not allowed_team_ids:
            return []
        team_user_ids = await _load_team_user_ids(allowed_team_ids)
        if not team_user_ids:
            return []

    target_ids: set[int] = set()
    offset = 0
    limit = 500
    while True:
        users, total = await repo.list_users(
            offset=offset,
            limit=limit,
            role=role,
            is_active=is_active,
            search=search,
            org_ids=org_ids,
        )
        for user in users:
            user_id = _coerce_bulk_candidate_user_id(user)
            if user_id is None:
                continue
            target_ids.add(user_id)
        offset += limit
        if len(target_ids) >= total:
            break

    if user_ids:
        target_ids &= {int(uid) for uid in user_ids}
    if team_user_ids is not None:
        target_ids &= team_user_ids

    return sorted(target_ids)


async def list_user_profiles(
    *,
    principal: AuthPrincipal,
    sections: str | None,
    include_sources: bool,
    include_raw: bool,
    mask_secrets: bool,
    user_ids: str | None,
    org_id: int | None,
    team_id: int | None,
    role: str | None,
    is_active: bool | None,
    search: str | None,
    page: int,
    limit: int,
    session_manager,
) -> tuple[UserProfileBatchResponse, dict[str, Any]]:
    """Return a batch of user profiles within admin scope."""
    batch_start = time.perf_counter()
    user_id_list = _parse_user_id_list(user_ids)
    target_ids = await _load_bulk_user_candidates(
        principal=principal,
        org_id=org_id,
        team_id=team_id,
        role=role,
        is_active=is_active,
        search=search,
        user_ids=user_id_list,
    )
    total = len(target_ids)
    offset = (page - 1) * limit
    page_ids = target_ids[offset : offset + limit]

    db_pool = await get_db_pool()
    service = UserProfileService(db_pool)
    requested = service.parse_sections(sections)
    if requested is None:
        requested = {"identity", "memberships", "quotas"}
    api_mgr = await get_api_key_manager()

    repo = await AuthnzUsersRepo.from_pool()
    profiles: list[UserProfileResponse] = []
    for user_id in page_ids:
        user = await repo.get_user_by_id(int(user_id))
        if not user:
            continue
        user_dict: dict[str, Any] = dict(user)
        user_dict.pop("password_hash", None)

        security: dict[str, Any] | None = None
        if "security" in requested:
            security = await service.build_security(
                user_id=int(user_id),
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
            metrics_scope="batch",
        )
        profiles.append(UserProfileResponse(**profile))

    pages = (total + limit - 1) // limit if limit else 0
    response = UserProfileBatchResponse(
        profiles=profiles,
        total=total,
        page=page,
        limit=limit,
        pages=pages,
        pagination=build_page_pagination_meta(
            page=page,
            per_page=limit,
            total=total,
            total_pages=pages,
        ),
    )

    try:
        registry = service._get_metrics_registry()
        if registry:
            page_size = len(page_ids)
            latency_ms = (time.perf_counter() - batch_start) * 1000.0
            registry.observe(
                "profile_batch_latency_ms",
                latency_ms,
                labels={"page_size": str(page_size)},
            )
            threshold_ms = UserProfileService.batch_sla_threshold_ms(max(1, page_size))
            if latency_ms > threshold_ms:
                registry.increment(
                    "profile_batch_sla_breach_total",
                    1,
                    labels={"page_size": str(page_size)},
                )
                logger.warning(
                    "Profile batch SLA exceeded: {:.2f}ms for page_size={} (threshold {}ms)",
                    latency_ms,
                    page_size,
                    threshold_ms,
                )
            timeout_ms = UserProfileService.batch_timeout_ms()
            if latency_ms > timeout_ms:
                registry.increment(
                    "profile_batch_timeout_total",
                    1,
                    labels={"page_size": str(page_size)},
                )
                logger.warning(
                    "Profile batch timeout threshold exceeded: {:.2f}ms for page_size={} (timeout {}ms)",
                    latency_ms,
                    page_size,
                    timeout_ms,
                )
    except Exception as telemetry_error:
        logger.bind(error_type=type(telemetry_error).__name__).debug(
            "Bulk profile update telemetry failed; continuing"
        )

    audit_metadata = {
        "filters": {
            "user_ids_count": len(user_id_list or []),
            "org_id": org_id,
            "team_id": team_id,
            "role": role,
            "is_active": is_active,
            "search": search,
        },
        "page": page,
        "limit": limit,
        "total": total,
        "sections": sorted(requested or []),
        "include_sources": include_sources,
        "include_raw": include_raw,
        "mask_secrets": mask_secrets,
    }
    audit_info = {
        "event_type": "data.read",
        "category": "data_access",
        "resource_type": "user_profile",
        "resource_id": None,
        "action": "user_profile.batch_read",
        "metadata": audit_metadata,
    }
    return response, audit_info


async def get_user_profile(
    *,
    user_id: int,
    principal: AuthPrincipal,
    sections: str | None,
    include_sources: bool,
    include_raw: bool,
    mask_secrets: bool,
    session_manager,
) -> tuple[UserProfileResponse, dict[str, Any]]:
    """Return a single user profile within admin scope."""
    try:
        await admin_scope_service.enforce_admin_user_scope(
            principal,
            user_id,
            require_hierarchy=False,
        )

        repo = await AuthnzUsersRepo.from_pool()
        user = await repo.get_user_by_id(user_id)
        if not user:
            raise UserNotFoundError(f"User {user_id}")

        user_dict: dict[str, Any] = dict(user)
        user_dict.pop("password_hash", None)

        db_pool = await get_db_pool()
        service = UserProfileService(db_pool)
        requested = service.parse_sections(sections)
        api_mgr = await get_api_key_manager()
        security = await service.build_security(
            user_id=int(user_id),
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
            metrics_scope="admin",
        )
        response = UserProfileResponse(**profile)
        audit_metadata = {
            "sections": sorted(requested or []),
            "include_sources": include_sources,
            "include_raw": include_raw,
            "mask_secrets": mask_secrets,
        }
        audit_info = {
            "event_type": "data.read",
            "category": "data_access",
            "resource_type": "user_profile",
            "resource_id": str(user_id),
            "action": "user_profile.read",
            "metadata": audit_metadata,
        }
        return response, audit_info

    except UserNotFoundError as err:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User {user_id} not found",
        ) from err
    except HTTPException:
        raise
    except Exception as exc:
        logger.bind(error_type=type(exc).__name__).error(
            "Failed to build profile for user {}",
            user_id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user profile",
        ) from exc


async def update_user_profile(
    *,
    user_id: int,
    payload: UserProfileUpdateRequest,
    principal: AuthPrincipal,
    db,
) -> tuple[UserProfileUpdateResponse | JSONResponse, dict[str, Any] | None]:
    """Update a user's profile within admin scope."""
    if not payload.updates:
        return (
            _profile_error_response(
                status_code=status.HTTP_400_BAD_REQUEST,
                error_code="profile_update_invalid",
                detail="No updates provided",
                errors=[UserProfileErrorDetail(key="updates", message="missing")],
            ),
            None,
        )

    await admin_scope_service.enforce_admin_user_scope(
        principal,
        user_id,
        require_hierarchy=True,
    )

    db_pool = await get_db_pool()
    repo = await AuthnzUsersRepo.from_pool()
    user = await repo.get_user_by_id(int(user_id))
    if not user:
        return (
            _profile_error_response(
                status_code=status.HTTP_404_NOT_FOUND,
                error_code="profile_update_not_found",
                detail=f"User {user_id} not found",
                errors=[UserProfileErrorDetail(key="user_id", message="not_found")],
            ),
            None,
        )
    profile_service = UserProfileService(db_pool)
    current_version = await profile_service.get_profile_version(user_id=int(user_id))
    if payload.profile_version is not None:
        if not profile_service.versions_match(current_version, payload.profile_version):
            return (
                _profile_error_response(
                    status_code=status.HTTP_409_CONFLICT,
                    error_code="profile_version_mismatch",
                    detail="profile_version_mismatch",
                    errors=[UserProfileErrorDetail(key="profile_version", message="mismatch")],
                ),
                None,
            )

    roles = _derive_profile_update_roles(principal)
    service = UserProfileUpdateService(db_pool)
    updates = [(entry.key, entry.value) for entry in payload.updates]
    preflight = await service.apply_updates(
        user_id=int(user_id),
        updates=updates,
        roles=roles,
        dry_run=True,
        db_conn=db,
        updated_by=principal.user_id,
        scope=ProfileUpdateScope(
            actor_user_id=principal.user_id,
            active_org_id=principal.active_org_id,
            active_team_id=principal.active_team_id,
        ),
    )

    error_payload = classify_profile_update_skips(preflight.skipped)
    if error_payload:
        status_code, error_code, detail, errors = error_payload
        return (
            _profile_error_response(
                status_code=status_code,
                error_code=error_code,
                detail=detail,
                errors=errors,
            ),
            None,
        )

    if payload.dry_run:
        response = UserProfileUpdateResponse(
            profile_version=current_version,
            applied=preflight.applied,
            skipped=[],
        )
    else:
        result = await service.apply_updates(
            user_id=int(user_id),
            updates=updates,
            roles=roles,
            dry_run=False,
            db_conn=db,
            updated_by=principal.user_id,
            scope=ProfileUpdateScope(
                actor_user_id=principal.user_id,
                active_org_id=principal.active_org_id,
                active_team_id=principal.active_team_id,
            ),
        )
        current_version = await profile_service.get_profile_version(user_id=int(user_id))
        skipped = [UserProfileUpdateError(**item) for item in result.skipped]
        response = UserProfileUpdateResponse(
            profile_version=current_version,
            applied=result.applied,
            skipped=skipped,
        )

    audit_metadata = {
        "dry_run": payload.dry_run,
        "update_keys": [entry.key for entry in payload.updates],
        "applied_count": len(response.applied),
        "skipped_count": len(response.skipped),
    }
    audit_info = {
        "event_type": "data.read" if payload.dry_run else "data.update",
        "category": "data_access" if payload.dry_run else "data_modification",
        "resource_type": "user_profile",
        "resource_id": str(user_id),
        "action": "user_profile.update_preview" if payload.dry_run else "user_profile.update",
        "metadata": audit_metadata,
    }
    return response, audit_info


async def bulk_update_user_profiles(
    *,
    payload: UserProfileBulkUpdateRequest,
    principal: AuthPrincipal,
) -> tuple[UserProfileBulkUpdateResponse, dict[str, Any]]:
    """Bulk update user profiles within admin scope."""
    if not payload.updates:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No updates provided")

    target_ids = await _load_bulk_user_candidates(
        principal=principal,
        org_id=payload.org_id,
        team_id=payload.team_id,
        role=payload.role,
        is_active=payload.is_active,
        search=payload.search,
        user_ids=payload.user_ids,
    )
    total_targets = len(target_ids)
    threshold = _get_bulk_confirm_threshold()
    if not payload.dry_run and total_targets > threshold and not payload.confirm:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "error": "bulk_update_confirm_required",
                "target_count": total_targets,
                "threshold": threshold,
            },
        )

    db_pool = await get_db_pool()
    update_service = UserProfileUpdateService(db_pool)
    profile_service = UserProfileService(db_pool)
    catalog = load_user_profile_catalog()
    catalog_map = {entry.key: entry for entry in catalog.entries}
    user_repo = await AuthnzUsersRepo.from_pool()
    roles = _derive_profile_update_roles(principal)
    updates = [(entry.key, entry.value) for entry in payload.updates]
    results: list[UserProfileBulkUpdateUserResult] = []
    updated_count = 0
    skipped_count = 0
    failed_count = 0

    for user_id in target_ids:
        try:
            await admin_scope_service.enforce_admin_user_scope(
                principal,
                int(user_id),
                require_hierarchy=True,
            )
        except HTTPException as exc:
            failed_count += 1
            results.append(
                UserProfileBulkUpdateUserResult(
                    user_id=int(user_id),
                    error=str(exc.detail) if exc.detail else "forbidden",
                )
            )
            continue

        try:
            before_values = await _build_bulk_update_before_values(
                user_id=int(user_id),
                updates=payload.updates,
                profile_service=profile_service,
                user_repo=user_repo,
                catalog_map=catalog_map,
            )

            if payload.dry_run:
                result = await update_service.apply_updates(
                    user_id=int(user_id),
                    updates=updates,
                    roles=roles,
                    dry_run=True,
                    db_conn=db_pool,
                    updated_by=principal.user_id,
                    scope=ProfileUpdateScope(
                        actor_user_id=principal.user_id,
                        active_org_id=principal.active_org_id,
                        active_team_id=principal.active_team_id,
                    ),
                )
            else:
                async with db_pool.transaction() as conn:
                    result = await update_service.apply_updates(
                        user_id=int(user_id),
                        updates=updates,
                        roles=roles,
                        dry_run=False,
                        db_conn=conn,
                        updated_by=principal.user_id,
                        scope=ProfileUpdateScope(
                            actor_user_id=principal.user_id,
                            active_org_id=principal.active_org_id,
                            active_team_id=principal.active_team_id,
                        ),
                    )

            profile_version = await profile_service.get_profile_version(user_id=int(user_id))
            skipped_entries = [UserProfileUpdateError(**item) for item in result.skipped]
            applied_keys = set(result.applied)
            diffs = [
                UserProfileBulkUpdateDiff(
                    key=entry.key,
                    before=before_values.get(entry.key),
                    after=_mask_profile_diff_value(
                        profile_service,
                        catalog_map,
                        entry.key,
                        entry.value,
                    ),
                )
                for entry in payload.updates
                if entry.key in applied_keys
            ]
            results.append(
                UserProfileBulkUpdateUserResult(
                    user_id=int(user_id),
                    profile_version=profile_version,
                    applied=result.applied,
                    skipped=skipped_entries,
                    diffs=diffs,
                )
            )
            if result.applied:
                updated_count += 1
            else:
                skipped_count += 1
        except HTTPException as exc:
            failed_count += 1
            results.append(
                UserProfileBulkUpdateUserResult(
                    user_id=int(user_id),
                    error=str(exc.detail) if exc.detail else "update_failed",
                )
            )
        except Exception as exc:
            logger.bind(error_type=type(exc).__name__).error(
                "Bulk profile update failed for user {}",
                user_id,
            )
            failed_count += 1
            results.append(
                UserProfileBulkUpdateUserResult(
                    user_id=int(user_id),
                    error="update_failed",
                )
            )

    try:
        registry = profile_service._get_metrics_registry()
        if registry:
            registry.increment(
                "profile_bulk_update_total",
                total_targets,
                labels={"dry_run": str(payload.dry_run).lower()},
            )
    except Exception as metrics_error:
        logger.bind(error_type=type(metrics_error).__name__).debug(
            "Bulk profile update metrics emission failed; continuing"
        )

    response = UserProfileBulkUpdateResponse(
        total_targets=total_targets,
        updated=updated_count,
        skipped=skipped_count,
        failed=failed_count,
        dry_run=payload.dry_run,
        results=results,
    )

    update_keys = [entry.key for entry in payload.updates]
    audit_metadata = {
        "dry_run": payload.dry_run,
        "confirm": payload.confirm,
        "target_count": total_targets,
        "updated": updated_count,
        "skipped": skipped_count,
        "failed": failed_count,
        "filters": {
            "org_id": payload.org_id,
            "team_id": payload.team_id,
            "role": payload.role,
            "is_active": payload.is_active,
            "search": payload.search,
            "user_ids_count": len(payload.user_ids or []),
        },
        "update_keys": update_keys,
    }
    audit_info = {
        "event_type": "data.read" if payload.dry_run else "data.update",
        "category": "data_access" if payload.dry_run else "data_modification",
        "resource_type": "user_profile",
        "resource_id": None,
        "action": "user_profile.bulk_preview" if payload.dry_run else "user_profile.bulk_update",
        "metadata": audit_metadata,
    }
    return response, audit_info
