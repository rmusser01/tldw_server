from __future__ import annotations

import json
from typing import Any

from fastapi import HTTPException, status
from loguru import logger

from tldw_Server_API.app.api.v1.utils.pagination import build_offset_pagination_meta
from tldw_Server_API.app.api.v1.schemas.org_team_schemas import (
    OrganizationCreateRequest,
    OrganizationResponse,
    OrganizationSTTSettingsResponse,
    OrganizationSTTSettingsUpdate,
    OrganizationWatchlistsSettingsResponse,
    OrganizationWatchlistsSettingsUpdate,
    OrgMemberAddRequest,
    OrgMemberListItem,
    OrgMemberRemoveResponse,
    OrgMemberResponse,
    OrgMemberRoleUpdateRequest,
    OrgMembershipItem,
    TeamCreateRequest,
    TeamMemberAddRequest,
    TeamMemberRoleUpdateRequest,
    TeamMemberRemoveResponse,
    TeamMembershipItem,
    TeamMemberResponse,
    TeamResponse,
)
from tldw_Server_API.app.core.AuthNZ.database import DatabasePool, get_db_pool
from tldw_Server_API.app.core.config_sections.stt import _parse_bool
from tldw_Server_API.app.core.AuthNZ.exceptions import (
    DuplicateOrganizationError,
    DuplicateTeamError,
    UserRegistrationException,
)
from tldw_Server_API.app.core.AuthNZ.orgs_teams import (
    add_org_member as core_add_org_member,
)
from tldw_Server_API.app.core.AuthNZ.orgs_teams import (
    add_team_member as core_add_team_member,
)
from tldw_Server_API.app.core.AuthNZ.orgs_teams import (
    create_organization as core_create_organization,
)
from tldw_Server_API.app.core.AuthNZ.orgs_teams import (
    create_team as core_create_team,
)
from tldw_Server_API.app.core.AuthNZ.orgs_teams import (
    list_org_members as core_list_org_members,
)
from tldw_Server_API.app.core.AuthNZ.orgs_teams import (
    list_org_memberships_for_user as core_list_org_memberships_for_user,
)
from tldw_Server_API.app.core.AuthNZ.orgs_teams import (
    list_memberships_for_user as core_list_team_memberships_for_user,
)
from tldw_Server_API.app.core.AuthNZ.orgs_teams import (
    list_organizations as core_list_organizations,
)
from tldw_Server_API.app.core.AuthNZ.orgs_teams import (
    list_team_members as core_list_team_members,
)
from tldw_Server_API.app.core.AuthNZ.orgs_teams import (
    remove_org_member as core_remove_org_member,
)
from tldw_Server_API.app.core.AuthNZ.orgs_teams import (
    remove_team_member as core_remove_team_member,
)
from tldw_Server_API.app.core.AuthNZ.orgs_teams import (
    update_team_member_role as core_update_team_member_role,
)
from tldw_Server_API.app.core.AuthNZ.orgs_teams import (
    update_org_member_role as core_update_org_member_role,
)
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.repos.org_stt_settings_repo import AuthnzOrgSttSettingsRepo
from tldw_Server_API.app.core.config import get_stt_config
from tldw_Server_API.app.core.DB_Management.backends.base import (
    DatabaseError as BackendDatabaseError,
)
from tldw_Server_API.app.services import admin_scope_service

_ADMIN_ORGS_NONCRITICAL_EXCEPTIONS = (
    AssertionError,
    AttributeError,
    BackendDatabaseError,
    ConnectionError,
    FileNotFoundError,
    ImportError,
    IndexError,
    json.JSONDecodeError,
    KeyError,
    LookupError,
    OSError,
    PermissionError,
    RuntimeError,
    TimeoutError,
    TypeError,
    UnicodeDecodeError,
    UserRegistrationException,
    ValueError,
)


def _is_db_pool_object(db: Any) -> bool:
    return isinstance(db, DatabasePool)


def _is_postgres_connection(db: Any) -> bool:
    """Resolve backend mode from connection shape without global probes."""
    if _is_db_pool_object(db):
        return getattr(db, "pool", None) is not None

    sqlite_hint = getattr(db, "_is_sqlite", None)
    if isinstance(sqlite_hint, bool):
        return not sqlite_hint

    if getattr(db, "_c", None) is not None:
        return False

    module_name = getattr(type(db), "__module__", "")
    if isinstance(module_name, str) and module_name.startswith("asyncpg"):
        return True

    return callable(getattr(db, "fetchrow", None))


async def _list_teams_by_org_conn(db, org_id: int, limit: int, offset: int) -> list[dict[str, Any]]:
    pg = _is_postgres_connection(db)
    if pg:
        rows = await db.fetch(
            "SELECT id, org_id, name, slug, description, COALESCE(is_active,TRUE) as is_active, created_at, updated_at FROM teams WHERE org_id = $1 ORDER BY created_at DESC LIMIT $2 OFFSET $3",
            org_id, limit, offset,
        )
        return [dict(r) for r in rows]
    cur = await db.execute(
        "SELECT id, org_id, name, slug, description, COALESCE(is_active,1), created_at, updated_at FROM teams WHERE org_id = ? ORDER BY created_at DESC LIMIT ? OFFSET ?",
        (org_id, limit, offset),
    )
    rows = await cur.fetchall()
    return [
        {
            "id": r[0],
            "org_id": r[1],
            "name": r[2],
            "slug": r[3],
            "description": r[4],
            "is_active": bool(r[5]),
            "created_at": r[6],
            "updated_at": r[7],
        }
        for r in rows
    ]


async def list_teams_by_org(
    org_id: int,
    limit: int = 1000,
    offset: int = 0,
    db=None,
) -> list[dict[str, Any]]:
    try:
        if db is None:
            pool = await get_db_pool()
            async with pool.acquire() as conn:
                return await _list_teams_by_org_conn(conn, org_id, limit, offset)
        return await _list_teams_by_org_conn(db, org_id, limit, offset)
    except Exception:
        logger.error("Failed to list teams by organization")
        raise


async def _emit_membership_audit_event(
    request,
    *,
    resource_type: str,
    resource_id: str,
    action: str,
    event_type_name: str,
    metadata: dict[str, Any],
) -> None:
    try:
        actor_id = getattr(request.state, "user_id", None)
        if not isinstance(actor_id, int):
            return
        from tldw_Server_API.app.api.v1.API_Deps.Audit_DB_Deps import (
            get_audit_service_for_user,
            get_or_create_audit_service_for_user_id,
        )
        from tldw_Server_API.app.core.Audit.unified_audit_service import (
            AuditContext,
            AuditEventCategory,
            AuditEventType,
        )
        from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User as _User
        from tldw_Server_API.app.core.DB_Management.Users_DB import get_user_by_id as _get_user

        _ud = await _get_user(actor_id)
        _svc = None
        if _ud:
            _user = _User(**_ud)
            _svc = await get_audit_service_for_user(_user)
        if _svc is None:
            _svc = await get_or_create_audit_service_for_user_id(int(actor_id))

        _ctx = AuditContext(
            user_id=str(actor_id),
            ip_address=(request.client.host if request.client else None),
            user_agent=request.headers.get("user-agent"),
            endpoint=str(request.url.path),
            method=request.method,
        )

        event_type = getattr(AuditEventType, event_type_name, AuditEventType.DATA_WRITE)
        await _svc.log_event(
            event_type=event_type,
            category=AuditEventCategory.AUTHORIZATION,
            context=_ctx,
            resource_type=resource_type,
            resource_id=str(resource_id),
            action=action,
            metadata=metadata,
        )
    except _ADMIN_ORGS_NONCRITICAL_EXCEPTIONS as exc:
        logger.debug(f"Audit ({action}) skipped/failed: {exc}")


async def create_org(
    payload: OrganizationCreateRequest,
    principal: AuthPrincipal,
) -> OrganizationResponse:
    try:
        admin_scope_service.require_platform_admin(principal)
        row = await core_create_organization(
            name=payload.name,
            owner_user_id=payload.owner_user_id,
            slug=payload.slug,
        )
        return OrganizationResponse(**row)
    except DuplicateOrganizationError as dup:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Organization with {dup.field} '{dup.value}' already exists",
        ) from dup
    except _ADMIN_ORGS_NONCRITICAL_EXCEPTIONS as exc:
        logger.error("Failed to create organization")
        raise HTTPException(status_code=500, detail="Failed to create organization") from exc


async def list_orgs(
    *,
    principal: AuthPrincipal,
    limit: int,
    offset: int,
    q: str | None,
    org_id: int | None,
    wants_wrapper: bool,
) -> Any:
    try:
        org_ids = await admin_scope_service.get_admin_org_ids(principal)
        if org_id is not None:
            org_ids = [org_id] if org_ids is None else [org_id] if org_id in org_ids else []

        result = await core_list_organizations(
            limit=limit,
            offset=offset,
            q=q,
            org_ids=org_ids,
            with_total=wants_wrapper,
        )
        if wants_wrapper:
            if not isinstance(result, tuple) or len(result) != 2:
                logger.error(
                    f"list_organizations returned unexpected format: {type(result).__name__}"
                )
                raise HTTPException(
                    status_code=500,
                    detail="Failed to list organizations",
                )
            rows, total = result
        else:
            rows = result[0] if isinstance(result, tuple) else result
            total = 0
        items = [OrganizationResponse(**r).model_dump() for r in rows]

        if wants_wrapper:
            total_count = int(total or 0)
            has_more = (offset + len(items)) < int(total or 0)
            return {
                "items": items,
                "total": total_count,
                "limit": limit,
                "offset": offset,
                "has_more": has_more,
                "pagination": build_offset_pagination_meta(
                    total=total_count,
                    limit=limit,
                    offset=offset,
                    count=len(items),
                ),
            }
        return items
    except HTTPException:
        raise
    except _ADMIN_ORGS_NONCRITICAL_EXCEPTIONS as exc:
        logger.error("Failed to list organizations")
        raise HTTPException(status_code=500, detail="Failed to list organizations") from exc


async def create_team(
    org_id: int,
    payload: TeamCreateRequest,
    principal: AuthPrincipal,
) -> TeamResponse:
    try:
        await admin_scope_service.enforce_admin_org_access(principal, org_id, require_admin=True)
        row = await core_create_team(
            org_id=org_id,
            name=payload.name,
            slug=payload.slug,
            description=payload.description,
        )
        return TeamResponse(**row)
    except DuplicateTeamError as dup:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Team with {dup.field} '{dup.value}' already exists in org {org_id}",
        ) from dup
    except _ADMIN_ORGS_NONCRITICAL_EXCEPTIONS as exc:
        logger.error("Failed to create team")
        raise HTTPException(status_code=500, detail="Failed to create team") from exc


async def list_teams(
    org_id: int,
    *,
    principal: AuthPrincipal,
    limit: int,
    offset: int,
    db,
) -> list[TeamResponse]:
    try:
        await admin_scope_service.enforce_admin_org_access(principal, org_id, require_admin=True)
        rows = await list_teams_by_org(org_id, limit=limit, offset=offset, db=db)
        return [TeamResponse(**r) for r in rows]
    except _ADMIN_ORGS_NONCRITICAL_EXCEPTIONS as exc:
        logger.error("Failed to list teams")
        raise HTTPException(status_code=500, detail="Failed to list teams") from exc


async def get_team(
    team_id: int,
    principal: AuthPrincipal,
) -> TeamResponse:
    try:
        team = await admin_scope_service.get_scoped_team(team_id, principal, require_admin=True)
        return TeamResponse(**team)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to fetch team")
        raise HTTPException(status_code=500, detail="Failed to fetch team") from exc


async def _ensure_org_exists(db: Any, org_id: int) -> None:
    pg = _is_postgres_connection(db)
    if pg:
        row = await db.fetchrow("SELECT id FROM organizations WHERE id = $1", org_id)
    else:
        cur = await db.execute("SELECT id FROM organizations WHERE id = ?", (org_id,))
        row = await cur.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="organization_not_found")


def _default_org_stt_settings_payload(org_id: int) -> OrganizationSTTSettingsResponse:
    config = get_stt_config()
    if isinstance(config, dict):
        delete_audio_after_success = _parse_bool(config.get("delete_audio_after_success", True), True)
        audio_retention_hours = float(config.get("audio_retention_hours", 0.0))
        redact_pii = _parse_bool(config.get("redact_pii", False), False)
        allow_unredacted_partials = _parse_bool(config.get("allow_unredacted_partials", False), False)
        raw_categories = config.get("redact_categories", [])
    else:
        delete_audio_after_success = _parse_bool(getattr(config, "delete_audio_after_success", True), True)
        audio_retention_hours = float(getattr(config, "audio_retention_hours", 0.0))
        redact_pii = _parse_bool(getattr(config, "redact_pii", False), False)
        allow_unredacted_partials = _parse_bool(getattr(config, "allow_unredacted_partials", False), False)
        raw_categories = getattr(config, "redact_categories", [])

    if isinstance(raw_categories, str):
        try:
            parsed_categories = json.loads(raw_categories)
        except _ADMIN_ORGS_NONCRITICAL_EXCEPTIONS:
            parsed_categories = [part.strip() for part in raw_categories.split(",")]
        raw_categories = parsed_categories

    redact_categories: list[str] = []
    seen_categories: set[str] = set()
    for raw in raw_categories if isinstance(raw_categories, list) else []:
        value = str(raw).strip().lower()
        if not value or value in seen_categories:
            continue
        redact_categories.append(value)
        seen_categories.add(value)

    return OrganizationSTTSettingsResponse(
        org_id=org_id,
        delete_audio_after_success=delete_audio_after_success,
        audio_retention_hours=audio_retention_hours,
        redact_pii=redact_pii,
        allow_unredacted_partials=allow_unredacted_partials,
        redact_categories=redact_categories,
    )


async def update_org_stt_settings(
    org_id: int,
    payload: OrganizationSTTSettingsUpdate,
    *,
    principal: AuthPrincipal,
    db,
) -> OrganizationSTTSettingsResponse:
    try:
        await admin_scope_service.enforce_admin_org_access(principal, org_id, require_admin=True)
        await _ensure_org_exists(db, org_id)
        repo = AuthnzOrgSttSettingsRepo(db)
        await repo.ensure_tables()
        current = await repo.get_settings(org_id)
        if current is None:
            current = _default_org_stt_settings_payload(org_id).model_dump()

        updated = await repo.upsert_settings(
            org_id=org_id,
            delete_audio_after_success=(
                current["delete_audio_after_success"]
                if payload.delete_audio_after_success is None
                else bool(payload.delete_audio_after_success)
            ),
            audio_retention_hours=(
                current["audio_retention_hours"]
                if payload.audio_retention_hours is None
                else float(payload.audio_retention_hours)
            ),
            redact_pii=current["redact_pii"] if payload.redact_pii is None else bool(payload.redact_pii),
            allow_unredacted_partials=(
                current["allow_unredacted_partials"]
                if payload.allow_unredacted_partials is None
                else bool(payload.allow_unredacted_partials)
            ),
            redact_categories=(
                current["redact_categories"]
                if payload.redact_categories is None
                else payload.redact_categories
            ),
            updated_by=getattr(principal, "user_id", None),
        )
        return OrganizationSTTSettingsResponse(**updated)
    except HTTPException:
        raise
    except _ADMIN_ORGS_NONCRITICAL_EXCEPTIONS as exc:
        logger.error("Failed to update org STT settings")
        raise HTTPException(status_code=500, detail="failed_to_update_org_stt_settings") from exc


async def get_org_stt_settings(
    org_id: int,
    *,
    principal: AuthPrincipal,
    db,
) -> OrganizationSTTSettingsResponse:
    try:
        await admin_scope_service.enforce_admin_org_access(principal, org_id, require_admin=True)
        await _ensure_org_exists(db, org_id)
        repo = AuthnzOrgSttSettingsRepo(db)
        await repo.ensure_tables()
        current = await repo.get_settings(org_id)
        if current is None:
            return _default_org_stt_settings_payload(org_id)
        return OrganizationSTTSettingsResponse(**current)
    except HTTPException:
        raise
    except _ADMIN_ORGS_NONCRITICAL_EXCEPTIONS as exc:
        logger.error("Failed to fetch org STT settings")
        raise HTTPException(status_code=500, detail="failed_to_fetch_org_stt_settings") from exc


async def update_org_watchlists_settings(
    org_id: int,
    payload: OrganizationWatchlistsSettingsUpdate,
    *,
    principal: AuthPrincipal,
    db,
) -> OrganizationWatchlistsSettingsResponse:
    try:
        await admin_scope_service.enforce_admin_org_access(principal, org_id, require_admin=True)
        is_pg = _is_postgres_connection(db)
        if is_pg:
            row = await db.fetchrow("SELECT metadata FROM organizations WHERE id = $1", org_id)
            meta_raw = row.get("metadata") if row else None
        else:
            cur = await db.execute("SELECT metadata FROM organizations WHERE id = ?", (org_id,))
            row = await cur.fetchone()
            meta_raw = row[0] if row else None
        if not row:
            raise HTTPException(status_code=404, detail="organization_not_found")
        meta: dict[str, Any]
        try:
            meta = json.loads(meta_raw) if meta_raw else {}
            if not isinstance(meta, dict):
                meta = {}
        except _ADMIN_ORGS_NONCRITICAL_EXCEPTIONS:
            meta = {}

        wl = meta.get("watchlists") if isinstance(meta.get("watchlists"), dict) else {}
        changed = False
        if payload.require_include_default is not None:
            wl["require_include_default"] = bool(payload.require_include_default)
            changed = True
        if changed:
            meta["watchlists"] = wl
            if is_pg:
                await db.execute(
                    "UPDATE organizations SET metadata = $1, updated_at = CURRENT_TIMESTAMP WHERE id = $2",
                    json.dumps(meta), org_id,
                )
            else:
                await db.execute(
                    "UPDATE organizations SET metadata = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (json.dumps(meta), org_id),
                )
        return OrganizationWatchlistsSettingsResponse(
            org_id=org_id,
            require_include_default=wl.get("require_include_default"),
        )
    except HTTPException:
        raise
    except _ADMIN_ORGS_NONCRITICAL_EXCEPTIONS as exc:
        logger.error("Failed to update org watchlists settings")
        raise HTTPException(status_code=500, detail="failed_to_update_org_watchlists_settings") from exc


async def get_org_watchlists_settings(
    org_id: int,
    *,
    principal: AuthPrincipal,
    db,
) -> OrganizationWatchlistsSettingsResponse:
    try:
        await admin_scope_service.enforce_admin_org_access(principal, org_id, require_admin=True)
        is_pg = _is_postgres_connection(db)
        if is_pg:
            row = await db.fetchrow("SELECT metadata FROM organizations WHERE id = $1", org_id)
            meta_raw = row.get("metadata") if row else None
        else:
            cur = await db.execute("SELECT metadata FROM organizations WHERE id = ?", (org_id,))
            row = await cur.fetchone()
            meta_raw = row[0] if row else None
        if not row:
            raise HTTPException(status_code=404, detail="organization_not_found")
        require_include_default = None
        if meta_raw:
            try:
                meta = json.loads(meta_raw)
                if isinstance(meta, dict):
                    wl = meta.get("watchlists") if isinstance(meta.get("watchlists"), dict) else None
                    if isinstance(wl, dict) and isinstance(wl.get("require_include_default"), bool):
                        require_include_default = bool(wl.get("require_include_default"))
                    elif isinstance(meta.get("watchlists_require_include_default"), bool):
                        require_include_default = bool(meta.get("watchlists_require_include_default"))
            except _ADMIN_ORGS_NONCRITICAL_EXCEPTIONS:
                pass
        return OrganizationWatchlistsSettingsResponse(org_id=org_id, require_include_default=require_include_default)
    except HTTPException:
        raise
    except _ADMIN_ORGS_NONCRITICAL_EXCEPTIONS as exc:
        logger.error("Failed to fetch org watchlists settings")
        raise HTTPException(status_code=500, detail="failed_to_fetch_org_watchlists_settings") from exc


async def add_team_member(
    team_id: int,
    payload: TeamMemberAddRequest,
    request,
    principal: AuthPrincipal,
) -> TeamMemberResponse:
    try:
        await admin_scope_service.get_scoped_team(team_id, principal, require_admin=True)
        row = await core_add_team_member(
            team_id=team_id,
            user_id=payload.user_id,
            role=payload.role or "member",
        )
        await _emit_membership_audit_event(
            request,
            resource_type="team",
            resource_id=str(team_id),
            action="team_member.add",
            event_type_name="DATA_WRITE",
            metadata={"target_user_id": payload.user_id, "role": payload.role or "member"},
        )
        return TeamMemberResponse(**row)
    except HTTPException:
        raise
    except _ADMIN_ORGS_NONCRITICAL_EXCEPTIONS as exc:
        logger.error("Failed to add team member")
        raise HTTPException(status_code=500, detail="Failed to add team member") from exc


async def list_team_members(
    team_id: int,
    *,
    principal: AuthPrincipal,
) -> list[TeamMemberResponse]:
    try:
        team = await admin_scope_service.get_scoped_team(team_id, principal, require_admin=True)
        rows = await core_list_team_members(team_id)
        org_id = team.get("org_id") if isinstance(team, dict) else None
        items: list[TeamMemberResponse] = []
        for row in rows:
            payload = dict(row)
            payload["team_id"] = team_id
            if org_id is not None:
                payload["org_id"] = org_id
            items.append(TeamMemberResponse(**payload))
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to list team members")
        raise HTTPException(status_code=500, detail="Failed to list team members") from exc
    return items


async def remove_team_member(
    team_id: int,
    user_id: int,
    request,
    principal: AuthPrincipal,
) -> TeamMemberRemoveResponse:
    try:
        await admin_scope_service.get_scoped_team(team_id, principal, require_admin=True)
        res = await core_remove_team_member(team_id=team_id, user_id=user_id)
        removed = bool(res.get("removed"))
        if not removed:
            return TeamMemberRemoveResponse(
                message="No membership found",
                team_id=int(res.get("team_id", team_id)),
                user_id=int(res.get("user_id", user_id)),
                removed=False,
            )
        await _emit_membership_audit_event(
            request,
            resource_type="team",
            resource_id=str(team_id),
            action="team_member.remove",
            event_type_name="DATA_DELETE",
            metadata={"target_user_id": user_id},
        )
        return TeamMemberRemoveResponse(
            message="Team member removed",
            team_id=int(res.get("team_id", team_id)),
            user_id=int(res.get("user_id", user_id)),
            removed=True,
        )
    except HTTPException:
        raise
    except _ADMIN_ORGS_NONCRITICAL_EXCEPTIONS as exc:
        logger.error("Failed to remove team member")
        raise HTTPException(status_code=500, detail="Failed to remove team member") from exc


async def update_team_member_role(
    team_id: int,
    user_id: int,
    payload: TeamMemberRoleUpdateRequest,
    request,
    principal: AuthPrincipal,
) -> TeamMemberResponse:
    try:
        team = await admin_scope_service.get_scoped_team(team_id, principal, require_admin=True)
        row = await core_update_team_member_role(team_id=team_id, user_id=user_id, role=payload.role)
        if not row:
            raise HTTPException(status_code=404, detail="Team membership not found")
        await _emit_membership_audit_event(
            request,
            resource_type="team",
            resource_id=str(team_id),
            action="team_member.update",
            event_type_name="DATA_UPDATE",
            metadata={"target_user_id": user_id, "new_role": payload.role},
        )
        response_payload = dict(row)
        response_payload["org_id"] = team.get("org_id") if isinstance(team, dict) else None
        return TeamMemberResponse(**response_payload)
    except HTTPException:
        raise
    except _ADMIN_ORGS_NONCRITICAL_EXCEPTIONS as exc:
        logger.error("Failed to update team member role")
        raise HTTPException(status_code=500, detail="Failed to update team member role") from exc


async def add_org_member(
    org_id: int,
    payload: OrgMemberAddRequest,
    request,
    principal: AuthPrincipal,
) -> OrgMemberResponse:
    try:
        await admin_scope_service.enforce_admin_org_access(principal, org_id, require_admin=True)
        row = await core_add_org_member(
            org_id=org_id,
            user_id=payload.user_id,
            role=payload.role or "member",
        )
        await _emit_membership_audit_event(
            request,
            resource_type="organization",
            resource_id=str(org_id),
            action="org_member.add",
            event_type_name="DATA_WRITE",
            metadata={"target_user_id": payload.user_id, "role": payload.role or "member"},
        )
        return OrgMemberResponse(**row)
    except HTTPException:
        raise
    except _ADMIN_ORGS_NONCRITICAL_EXCEPTIONS as exc:
        logger.error("Failed to add org member")
        raise HTTPException(status_code=500, detail="Failed to add org member") from exc


async def list_org_members(
    org_id: int,
    *,
    limit: int,
    offset: int,
    role: str | None,
    status_filter: str | None,
    principal: AuthPrincipal,
) -> list[OrgMemberListItem]:
    try:
        await admin_scope_service.enforce_admin_org_access(principal, org_id, require_admin=True)
        rows = await core_list_org_members(org_id=org_id, limit=limit, offset=offset, role=role, status=status_filter)
        out: list[OrgMemberListItem] = []
        for r in rows:
            d = dict(r)
            try:
                from datetime import datetime as _dt
                if isinstance(d.get("added_at"), _dt):
                    d["added_at"] = d["added_at"].isoformat()
            except _ADMIN_ORGS_NONCRITICAL_EXCEPTIONS:
                pass
            out.append(OrgMemberListItem(**d))
        return out
    except HTTPException:
        raise
    except _ADMIN_ORGS_NONCRITICAL_EXCEPTIONS as exc:
        logger.error("Failed to list org members")
        raise HTTPException(status_code=500, detail="Failed to list org members") from exc


async def remove_org_member(
    org_id: int,
    user_id: int,
    request,
    principal: AuthPrincipal,
) -> OrgMemberRemoveResponse:
    try:
        await admin_scope_service.enforce_admin_org_access(principal, org_id, require_admin=True)
        res = await core_remove_org_member(org_id=org_id, user_id=user_id)
        if res.get("error") == "owner_required":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Organization must retain at least one owner",
            )
        removed = bool(res.get("removed"))
        if not removed:
            return OrgMemberRemoveResponse(
                message="No membership found",
                org_id=int(res.get("org_id", org_id)),
                user_id=int(res.get("user_id", user_id)),
                removed=False,
            )
        await _emit_membership_audit_event(
            request,
            resource_type="organization",
            resource_id=str(org_id),
            action="org_member.remove",
            event_type_name="DATA_DELETE",
            metadata={"target_user_id": user_id},
        )
        return OrgMemberRemoveResponse(
            message="Org member removed",
            org_id=int(res.get("org_id", org_id)),
            user_id=int(res.get("user_id", user_id)),
            removed=True,
        )
    except HTTPException:
        raise
    except _ADMIN_ORGS_NONCRITICAL_EXCEPTIONS as exc:
        logger.error("Failed to remove org member")
        raise HTTPException(status_code=500, detail="Failed to remove org member") from exc


async def update_org_member_role(
    org_id: int,
    user_id: int,
    payload: OrgMemberRoleUpdateRequest,
    request,
    principal: AuthPrincipal,
) -> OrgMemberResponse:
    try:
        await admin_scope_service.enforce_admin_org_access(principal, org_id, require_admin=True)
        row = await core_update_org_member_role(org_id=org_id, user_id=user_id, role=payload.role)
        if not row:
            raise HTTPException(status_code=404, detail="Org membership not found")
        if row.get("error") == "owner_required":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Organization must retain at least one owner",
            )
        await _emit_membership_audit_event(
            request,
            resource_type="organization",
            resource_id=str(org_id),
            action="org_member.update",
            event_type_name="DATA_UPDATE",
            metadata={"target_user_id": user_id, "new_role": payload.role},
        )
        return OrgMemberResponse(**row)
    except HTTPException:
        raise
    except _ADMIN_ORGS_NONCRITICAL_EXCEPTIONS as exc:
        logger.error("Failed to update org member role")
        raise HTTPException(status_code=500, detail="Failed to update org member role") from exc


async def list_user_org_memberships(
    user_id: int,
    principal: AuthPrincipal,
) -> list[OrgMembershipItem]:
    try:
        await admin_scope_service.enforce_admin_user_scope(principal, user_id, require_hierarchy=False)
        rows = await core_list_org_memberships_for_user(user_id)
        return [OrgMembershipItem(**r) for r in rows]
    except _ADMIN_ORGS_NONCRITICAL_EXCEPTIONS as exc:
        logger.error("Failed to list org memberships")
        raise HTTPException(status_code=500, detail="Failed to list org memberships") from exc


async def list_user_team_memberships(
    user_id: int,
    principal: AuthPrincipal,
) -> list[TeamMembershipItem]:
    try:
        await admin_scope_service.enforce_admin_user_scope(principal, user_id, require_hierarchy=False)
        rows = await core_list_team_memberships_for_user(user_id)
        results: list[TeamMembershipItem] = []
        for row in rows:
            payload = dict(row)
            payload["team_id"] = int(payload.get("team_id"))
            payload["org_id"] = int(payload.get("org_id"))
            payload["role"] = str(payload.get("role") or "member")
            team_name = payload.get("team_name")
            org_name = payload.get("org_name")
            payload["team_name"] = str(team_name) if isinstance(team_name, str) else None
            payload["org_name"] = str(org_name) if isinstance(org_name, str) else None
            results.append(TeamMembershipItem(**payload))
        return results
    except HTTPException:
        raise
    except _ADMIN_ORGS_NONCRITICAL_EXCEPTIONS as exc:
        logger.error("Failed to list team memberships")
        raise HTTPException(status_code=500, detail="Failed to list team memberships") from exc
