from __future__ import annotations

from collections.abc import Awaitable
from datetime import datetime, timezone
from typing import Any, Callable

from fastapi import HTTPException, status
from loguru import logger

from tldw_Server_API.app.api.v1.schemas.api_key_schemas import (
    APIKeyAuditEntry,
    APIKeyAuditListResponse,
    APIKeyCreateRequest,
    APIKeyCreateResponse,
    APIKeyMetadata,
    APIKeyRotateRequest,
    APIKeyUpdateRequest,
)
from tldw_Server_API.app.api.v1.schemas.org_team_schemas import VirtualKeyCreateRequest
from tldw_Server_API.app.core.Audit.unified_audit_service import MandatoryAuditWriteError
from tldw_Server_API.app.core.AuthNZ.api_key_manager import get_api_key_manager
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.services import admin_scope_service
from tldw_Server_API.app.services.admin_service import update_api_key_metadata

_ADMIN_API_KEYS_NONCRITICAL_EXCEPTIONS = (
    AttributeError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)


def _principal_actor_user_id(principal: Any) -> int | None:
    raw_user_id = getattr(principal, "user_id", None)
    if raw_user_id is None:
        return None
    try:
        return int(raw_user_id)
    except (TypeError, ValueError):
        return None


def _principal_actor_roles(principal: Any) -> list[str]:
    raw_roles = getattr(principal, "roles", None)
    if not raw_roles:
        return []
    if isinstance(raw_roles, (list, tuple, set)):
        return [str(role) for role in raw_roles]
    return [str(raw_roles)]


def _principal_actor_kwargs(principal: Any) -> dict[str, Any]:
    return {
        "actor_user_id": _principal_actor_user_id(principal),
        "actor_subject": getattr(principal, "subject", None),
        "actor_kind": getattr(principal, "kind", None),
        "actor_roles": _principal_actor_roles(principal),
    }


async def list_user_api_keys(
    principal: AuthPrincipal,
    user_id: int,
    *,
    include_revoked: bool,
) -> list[APIKeyMetadata]:
    try:
        await admin_scope_service.enforce_admin_user_scope(principal, user_id, require_hierarchy=False)
        api_mgr = await get_api_key_manager()
        rows = await api_mgr.list_user_keys(user_id=user_id, include_revoked=include_revoked)
        return [APIKeyMetadata(**row) for row in rows]
    except HTTPException:
        raise
    except _ADMIN_API_KEYS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Admin failed to list API keys for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to list API keys") from e


async def create_user_api_key(
    principal: AuthPrincipal,
    user_id: int,
    request: APIKeyCreateRequest,
) -> APIKeyCreateResponse:
    try:
        await admin_scope_service.enforce_admin_user_scope(principal, user_id, require_hierarchy=True)
        api_mgr = await get_api_key_manager()
        result = await api_mgr.create_api_key(
            user_id=user_id,
            name=request.name,
            description=request.description,
            scope=request.scope,
            expires_in_days=request.expires_in_days,
            **_principal_actor_kwargs(principal),
        )
        return APIKeyCreateResponse(**result)
    except HTTPException:
        raise
    except MandatoryAuditWriteError as e:
        logger.error(f"Admin mandatory audit write failed while creating API key for user {user_id}: {e}")
        raise HTTPException(status_code=503, detail="Mandatory audit persistence unavailable") from e
    except _ADMIN_API_KEYS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Admin failed to create API key for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to create API key") from e


async def rotate_user_api_key(
    principal: AuthPrincipal,
    user_id: int,
    key_id: int,
    request: APIKeyRotateRequest,
) -> APIKeyCreateResponse:
    try:
        await admin_scope_service.enforce_admin_user_scope(principal, user_id, require_hierarchy=True)
        api_mgr = await get_api_key_manager()
        result = await api_mgr.rotate_api_key(
            key_id=key_id,
            user_id=user_id,
            expires_in_days=request.expires_in_days,
            **_principal_actor_kwargs(principal),
        )
        return APIKeyCreateResponse(**result)
    except HTTPException:
        raise
    except MandatoryAuditWriteError as e:
        logger.error(f"Admin mandatory audit write failed while rotating API key {key_id} for user {user_id}: {e}")
        raise HTTPException(status_code=503, detail="Mandatory audit persistence unavailable") from e
    except _ADMIN_API_KEYS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Admin failed to rotate API key {key_id} for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to rotate API key") from e


async def revoke_user_api_key(
    principal: AuthPrincipal,
    user_id: int,
    key_id: int,
) -> dict[str, Any]:
    try:
        await admin_scope_service.enforce_admin_user_scope(principal, user_id, require_hierarchy=True)
        api_mgr = await get_api_key_manager()
        success = await api_mgr.revoke_api_key(
            key_id=key_id,
            user_id=user_id,
            **_principal_actor_kwargs(principal),
        )
        if not success:
            raise HTTPException(status_code=404, detail="API key not found")
        return {"message": "API key revoked", "user_id": user_id, "key_id": key_id}
    except HTTPException:
        raise
    except MandatoryAuditWriteError as e:
        logger.error(f"Admin mandatory audit write failed while revoking API key {key_id} for user {user_id}: {e}")
        raise HTTPException(status_code=503, detail="Mandatory audit persistence unavailable") from e
    except _ADMIN_API_KEYS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Admin failed to revoke API key {key_id} for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to revoke API key") from e


async def update_user_api_key(
    principal: AuthPrincipal,
    user_id: int,
    key_id: int,
    request: APIKeyUpdateRequest,
    db,
    *,
    is_pg_fn: Callable[[], Awaitable[bool]],
) -> APIKeyMetadata:
    try:
        await admin_scope_service.enforce_admin_user_scope(principal, user_id, require_hierarchy=True)
        try:
            is_pg = await is_pg_fn()
            row = await update_api_key_metadata(
                db,
                user_id=user_id,
                key_id=key_id,
                rate_limit=request.rate_limit,
                allowed_ips=request.allowed_ips,
                is_postgres=is_pg,
            )
        except ValueError:
            raise HTTPException(status_code=400, detail="No updates provided") from None
        except LookupError:
            raise HTTPException(status_code=404, detail="API key not found") from None
        return APIKeyMetadata(**row)
    except HTTPException:
        raise
    except _ADMIN_API_KEYS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Admin failed to update API key {key_id} for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to update API key") from e


async def create_virtual_key(
    principal: AuthPrincipal,
    user_id: int,
    payload: VirtualKeyCreateRequest,
) -> dict[str, Any]:
    try:
        await admin_scope_service.enforce_admin_user_scope(principal, user_id, require_hierarchy=True)
        if payload.org_id is not None:
            await admin_scope_service.enforce_admin_org_access(principal, payload.org_id, require_admin=True)
        if payload.team_id is not None:
            team = await admin_scope_service.get_scoped_team(payload.team_id, principal, require_admin=True)
            if payload.org_id is not None and int(team.get("org_id")) != int(payload.org_id):
                raise HTTPException(status_code=400, detail="team_id does not belong to org_id")
        api_mgr = await get_api_key_manager()
        result = await api_mgr.create_virtual_key(
            user_id=user_id,
            name=payload.name,
            description=payload.description,
            expires_in_days=payload.expires_in_days,
            org_id=payload.org_id,
            team_id=payload.team_id,
            allowed_endpoints=payload.allowed_endpoints,
            allowed_providers=payload.allowed_providers,
            allowed_models=payload.allowed_models,
            budget_day_tokens=payload.budget_day_tokens,
            budget_month_tokens=payload.budget_month_tokens,
            budget_day_usd=payload.budget_day_usd,
            budget_month_usd=payload.budget_month_usd,
            allowed_methods=payload.allowed_methods,
            allowed_paths=payload.allowed_paths,
            max_calls=payload.max_calls,
            max_runs=payload.max_runs,
            **_principal_actor_kwargs(principal),
        )
        return result
    except HTTPException:
        raise
    except MandatoryAuditWriteError as e:
        logger.error(f"Admin mandatory audit write failed while creating virtual key for user {user_id}: {e}")
        raise HTTPException(status_code=503, detail="Mandatory audit persistence unavailable") from e
    except _ADMIN_API_KEYS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Admin failed to create virtual key for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to create virtual key") from e


async def list_virtual_keys(
    principal: AuthPrincipal,
    user_id: int,
    *,
    db,
    name: str | None,
    status_filter: str | None,
    org_id: int | None,
    team_id: int | None,
    created_after: datetime | None,
    created_before: datetime | None,
    is_pg_fn: Callable[[], Awaitable[bool]],
) -> list[APIKeyMetadata]:
    try:
        await admin_scope_service.enforce_admin_user_scope(principal, user_id, require_hierarchy=False)

        def _normalize_ts(value: datetime) -> datetime:
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value.astimezone(timezone.utc)

        name_filter = name.strip() if isinstance(name, str) and name.strip() else None
        status_filter = status_filter.strip() if isinstance(status_filter, str) and status_filter.strip() else None
        created_after_dt = _normalize_ts(created_after) if created_after else None
        created_before_dt = _normalize_ts(created_before) if created_before else None
        if created_after_dt and created_before_dt and created_after_dt > created_before_dt:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="created_before must be >= created_after",
            )

        if isinstance(user_id, (tuple, list)):
            user_id = user_id[0]
        user_id = int(user_id)
        is_pg = await is_pg_fn()
        if is_pg:
            conditions = ["user_id = $1", "COALESCE(is_virtual, FALSE) = TRUE"]
            params: list[Any] = [user_id]
            param_idx = 1
            if name_filter:
                param_idx += 1
                conditions.append(f"LOWER(name) LIKE ${param_idx}")
                params.append(f"%{name_filter.lower()}%")
            if status_filter:
                param_idx += 1
                conditions.append(f"status = ${param_idx}")
                params.append(status_filter)
            if org_id is not None:
                param_idx += 1
                conditions.append(f"org_id = ${param_idx}")
                params.append(org_id)
            if team_id is not None:
                param_idx += 1
                conditions.append(f"team_id = ${param_idx}")
                params.append(team_id)
            if created_after_dt:
                param_idx += 1
                conditions.append(f"created_at >= ${param_idx}")
                params.append(created_after_dt.replace(tzinfo=None))
            if created_before_dt:
                param_idx += 1
                conditions.append(f"created_at <= ${param_idx}")
                params.append(created_before_dt.replace(tzinfo=None))
            where_clause = " AND ".join(conditions)
            list_api_keys_sql_template = (
                "SELECT id, key_prefix, name, description, scope, status, created_at, expires_at, usage_count, last_used_at, last_used_ip "
                "FROM api_keys WHERE {where_clause} ORDER BY created_at DESC"
            )
            list_api_keys_sql = list_api_keys_sql_template.format_map(locals())  # nosec B608
            rows = await db.fetch(
                list_api_keys_sql,
                *params,
            )
            items = [APIKeyMetadata(**dict(r)) for r in rows]
        else:
            conditions = ["user_id = ?", "COALESCE(is_virtual,0) = 1"]
            params2: list[Any] = [user_id]
            if name_filter:
                conditions.append("LOWER(name) LIKE ?")
                params2.append(f"%{name_filter.lower()}%")
            if status_filter:
                conditions.append("status = ?")
                params2.append(status_filter)
            if org_id is not None:
                conditions.append("org_id = ?")
                params2.append(org_id)
            if team_id is not None:
                conditions.append("team_id = ?")
                params2.append(team_id)
            if created_after_dt:
                conditions.append("datetime(created_at) >= datetime(?)")
                params2.append(created_after_dt.strftime("%Y-%m-%d %H:%M:%S"))
            if created_before_dt:
                conditions.append("datetime(created_at) <= datetime(?)")
                params2.append(created_before_dt.strftime("%Y-%m-%d %H:%M:%S"))
            where_clause = " AND ".join(conditions)
            list_api_keys_sql_template = (
                "SELECT id, key_prefix, name, description, scope, status, created_at, expires_at, usage_count, last_used_at, last_used_ip "
                "FROM api_keys WHERE {where_clause} ORDER BY datetime(created_at) DESC"
            )
            list_api_keys_sql = list_api_keys_sql_template.format_map(locals())  # nosec B608
            cur = await db.execute(
                list_api_keys_sql,
                tuple(params2),
            )
            rows = await cur.fetchall()
            items = [
                APIKeyMetadata(
                    id=r[0],
                    key_prefix=r[1],
                    name=r[2],
                    description=r[3],
                    scope=r[4],
                    status=r[5],
                    created_at=r[6],
                    expires_at=r[7],
                    usage_count=r[8],
                    last_used_at=r[9],
                    last_used_ip=r[10],
                ) for r in rows
            ]
        return items
    except HTTPException:
        raise
    except _ADMIN_API_KEYS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Admin failed to list virtual keys for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to list virtual keys") from e


async def get_api_key_audit_log(
    principal: AuthPrincipal,
    key_id: int,
    *,
    limit: int,
    offset: int,
    db,
    is_pg_fn: Callable[[], Awaitable[bool]],
) -> APIKeyAuditListResponse:
    try:
        is_pg = await is_pg_fn()
        if is_pg:
            key_owner = await db.fetchval("SELECT user_id FROM api_keys WHERE id = $1", key_id)
        else:
            cur = await db.execute("SELECT user_id FROM api_keys WHERE id = ?", (key_id,))
            row = await cur.fetchone()
            key_owner = row[0] if row else None
        if key_owner is None:
            raise HTTPException(status_code=404, detail="API key not found")
        await admin_scope_service.enforce_admin_user_scope(principal, int(key_owner), require_hierarchy=False)
        if is_pg:
            rows = await db.fetch(
                """
                SELECT id, api_key_id, action, user_id, ip_address, user_agent, details, created_at
                FROM api_key_audit_log
                WHERE api_key_id = $1
                ORDER BY created_at DESC
                LIMIT $2 OFFSET $3
                """,
                key_id, limit, offset
            )
        else:
            cursor = await db.execute(
                """
                SELECT id, api_key_id, action, user_id, ip_address, user_agent, details, created_at
                FROM api_key_audit_log
                WHERE api_key_id = ?
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
                """,
                (key_id, limit, offset)
            )
            rows = await cursor.fetchall()

        items: list[APIKeyAuditEntry] = []
        for r in rows:
            if isinstance(r, dict):
                items.append(APIKeyAuditEntry(**r))
            else:
                items.append(APIKeyAuditEntry(
                    id=r[0], api_key_id=r[1], action=r[2], user_id=r[3], ip_address=r[4], user_agent=r[5], details=r[6], created_at=r[7]
                ))
        return APIKeyAuditListResponse(key_id=key_id, items=items)
    except HTTPException:
        raise
    except _ADMIN_API_KEYS_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Admin failed to fetch audit log for key {key_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to load audit log") from e
