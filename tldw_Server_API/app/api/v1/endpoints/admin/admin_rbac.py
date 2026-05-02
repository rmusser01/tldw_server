from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
    check_rate_limit,
    get_auth_principal,
    get_db_transaction,
)
from tldw_Server_API.app.api.v1.schemas.admin_rbac_schemas import (
    EffectivePermissionsResponse,
    OverrideEffect,
    PermissionCreateRequest,
    PermissionResponse,
    RoleCreateRequest,
    RoleEffectivePermissionsResponse,
    RolePermissionBooleanMatrixResponse,
    RolePermissionGrant,
    RolePermissionMatrixResponse,
    RoleResponse,
    UserOverrideEntry,
    UserOverridesResponse,
    UserOverrideUpsertRequest,
    UserRoleListResponse,
)
from tldw_Server_API.app.api.v1.schemas.admin_schemas import (
    KanbanFtsMaintenanceResponse,
    ToolPermissionBatchRequest,
    ToolPermissionCreateRequest,
    ToolPermissionGrantRequest,
    ToolPermissionPrefixRequest,
    ToolPermissionResponse,
)
from tldw_Server_API.app.api.v1.utils.http_errors import map_db_error_to_http
from tldw_Server_API.app.core.AuthNZ.exceptions import DuplicateRoleError
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.rbac import get_effective_permissions
from tldw_Server_API.app.core.AuthNZ.repos.rbac_repo import AuthnzRbacRepo
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.DB_Management.Kanban_DB import InputError, KanbanDB, KanbanDBError
from tldw_Server_API.app.core.exceptions import ResourceNotFoundError
from tldw_Server_API.app.services.admin_roles_permissions_service import (
    create_role as svc_create_role,
)
from tldw_Server_API.app.services.admin_roles_permissions_service import (
    delete_role as svc_delete_role,
)
from tldw_Server_API.app.services.admin_roles_permissions_service import (
    delete_tool_permission as svc_delete_tool_permission,
)
from tldw_Server_API.app.services.admin_roles_permissions_service import (
    grant_tool_permission_to_role as svc_grant_tool_perm,
)
from tldw_Server_API.app.services.admin_roles_permissions_service import (
    list_role_permissions as svc_list_role_permissions,
)
from tldw_Server_API.app.services.admin_roles_permissions_service import (
    list_roles as svc_list_roles,
)
from tldw_Server_API.app.services.admin_roles_permissions_service import (
    list_tool_permissions as svc_list_tool_permissions,
)
from tldw_Server_API.app.services.admin_roles_permissions_service import (
    revoke_tool_permission_from_role as svc_revoke_tool_perm,
)

router = APIRouter()


_RBAC_NONCRITICAL_EXCEPTIONS = (
    asyncio.CancelledError,
    asyncio.TimeoutError,
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
    HTTPException,
    DuplicateRoleError,
    InputError,
    KanbanDBError,
    ResourceNotFoundError,
)


def _get_rbac_repo() -> AuthnzRbacRepo:
    from tldw_Server_API.app.api.v1.endpoints import admin as admin_mod

    return admin_mod._get_rbac_repo()


async def _enforce_admin_user_scope(
    principal: AuthPrincipal,
    target_user_id: int,
    *,
    require_hierarchy: bool,
) -> None:
    from tldw_Server_API.app.api.v1.endpoints import admin as admin_mod

    await admin_mod._enforce_admin_user_scope(
        principal,
        target_user_id,
        require_hierarchy=require_hierarchy,
    )


def _get_is_postgres_backend_fn() -> Callable[[], Awaitable[bool]]:
    from tldw_Server_API.app.api.v1.endpoints import admin as admin_mod

    return admin_mod._is_postgres_backend


def _get_kanban_db_for_user_id(user_id: int) -> KanbanDB:
    db_path = DatabasePaths.get_kanban_db_path(user_id)
    return KanbanDB(db_path=str(db_path), user_id=str(user_id))


@router.post(
    "/kanban/fts/{action}",
    response_model=KanbanFtsMaintenanceResponse,
    dependencies=[Depends(check_rate_limit)],
)
async def admin_kanban_fts_maintenance(
    action: Literal["optimize", "rebuild"],
    user_id: int = Query(..., ge=1),
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> KanbanFtsMaintenanceResponse:
    db: KanbanDB | None = None
    try:
        await _enforce_admin_user_scope(principal, user_id, require_hierarchy=True)
        db = _get_kanban_db_for_user_id(user_id)
        try:
            if action == "rebuild":
                db.rebuild_fts()
            else:
                db.optimize_fts()
        finally:
            db.close()
    except InputError as exc:
        raise map_db_error_to_http(exc, default_detail="Kanban FTS maintenance failed") from exc
    except KanbanDBError as exc:
        logger.error("Kanban FTS maintenance failed")
        raise map_db_error_to_http(exc, default_detail="Kanban FTS maintenance failed") from exc
    except _RBAC_NONCRITICAL_EXCEPTIONS as exc:
        logger.error("Kanban FTS maintenance failed")
        raise HTTPException(status_code=500, detail="Kanban FTS maintenance failed") from exc
    return KanbanFtsMaintenanceResponse(user_id=user_id, action=action, status="ok")


#######################################################################################################################
#
# RBAC: Roles, Permissions, Assignments, Overrides


@router.get("/roles", response_model=list[RoleResponse])
async def list_roles(db=Depends(get_db_transaction)) -> list[RoleResponse]:
    try:
        rows = await svc_list_roles(db)
        return [RoleResponse(**row) for row in rows]
    except _RBAC_NONCRITICAL_EXCEPTIONS as e:
        logger.error("Failed to list roles")
        raise HTTPException(status_code=500, detail="Failed to list roles") from e


@router.post("/roles", response_model=RoleResponse)
async def create_role(payload: RoleCreateRequest, db=Depends(get_db_transaction)) -> RoleResponse:
    try:
        row = await svc_create_role(db, payload.name, payload.description, False)
        return RoleResponse(**row)
    except DuplicateRoleError as dup:
        raise HTTPException(status_code=409, detail=f"Role '{dup.name}' already exists") from dup
    except _RBAC_NONCRITICAL_EXCEPTIONS as e:
        logger.error("Failed to create role")
        raise HTTPException(status_code=500, detail="Failed to create role") from e


@router.delete("/roles/{role_id}")
async def delete_role(role_id: int, db=Depends(get_db_transaction)) -> dict:
    try:
        await svc_delete_role(db, role_id)
        return {"message": "Role deleted"}
    except _RBAC_NONCRITICAL_EXCEPTIONS as e:
        logger.error("Failed to delete role")
        raise HTTPException(status_code=500, detail="Failed to delete role") from e


@router.get("/roles/{role_id}/permissions", response_model=list[PermissionResponse])
async def list_role_permissions(role_id: int, db=Depends(get_db_transaction)) -> list[PermissionResponse]:
    """List permissions granted to a specific role (read-only matrix row)."""
    try:
        rows = await svc_list_role_permissions(db, role_id)
        return [PermissionResponse(**r) for r in rows]
    except _RBAC_NONCRITICAL_EXCEPTIONS as e:
        logger.error("Failed to list role permissions")
        raise HTTPException(status_code=500, detail="Failed to list role permissions") from e


@router.get("/permissions/tools", response_model=list[ToolPermissionResponse])
async def list_tool_permissions(db=Depends(get_db_transaction)) -> list[ToolPermissionResponse]:
    """List tool execution permissions (name starts with 'tools.execute:')."""
    try:
        rows = await svc_list_tool_permissions(db)
        return [ToolPermissionResponse(**r) for r in rows]
    except _RBAC_NONCRITICAL_EXCEPTIONS as e:
        logger.error("Failed to list tool permissions")
        raise HTTPException(status_code=500, detail="Failed to list tool permissions") from e


@router.post("/permissions/tools", response_model=ToolPermissionResponse)
async def create_tool_permission(
    payload: ToolPermissionCreateRequest, db=Depends(get_db_transaction)
) -> ToolPermissionResponse:
    """Create a tool execution permission.

    - tool_name='*' → creates tools.execute:*
    - tool_name='<name>' → creates tools.execute:<name>
    """
    try:
        tool = payload.tool_name.strip()
        name = f"tools.execute:{'*' if tool == '*' else tool}"
        desc = payload.description or ("Wildcard tool execution" if tool == "*" else f"Execute tool {tool}")

        is_pg = await _get_is_postgres_backend_fn()()
        if is_pg:
            await db.execute(
                "INSERT INTO permissions (name, description, category) VALUES ($1, $2, $3) ON CONFLICT (name) DO NOTHING",
                name,
                desc,
                "tools",
            )
            row = await db.fetchrow(
                "SELECT name, description, category FROM permissions WHERE name = $1",
                name,
            )
            return ToolPermissionResponse(**dict(row))
        else:
            # SQLite doesn't support upsert on all versions; emulate
            cur = await db.execute("SELECT name, description, category FROM permissions WHERE name = ?", (name,))
            r = await cur.fetchone()
            if not r:
                await db.execute(
                    "INSERT INTO permissions (name, description, category) VALUES (?, ?, ?)",
                    (name, desc, "tools"),
                )
                await db.commit()
                cur = await db.execute("SELECT name, description, category FROM permissions WHERE name = ?", (name,))
                r = await cur.fetchone()
            return ToolPermissionResponse(name=r[0], description=r[1], category=r[2])
    except _RBAC_NONCRITICAL_EXCEPTIONS as e:
        logger.error("Failed to create tool permission")
        raise HTTPException(status_code=500, detail="Failed to create tool permission") from e


@router.delete("/permissions/tools/{perm_name}")
async def delete_tool_permission(perm_name: str, db=Depends(get_db_transaction)) -> dict:
    """Delete a tool execution permission by full name (e.g., tools.execute:my_tool)."""
    try:
        if not perm_name.startswith("tools.execute:"):
            raise HTTPException(status_code=400, detail="Invalid tool permission name")
        await svc_delete_tool_permission(db, perm_name)
        return {"message": "Tool permission deleted", "name": perm_name}
    except HTTPException:
        raise
    except _RBAC_NONCRITICAL_EXCEPTIONS as e:
        logger.error("Failed to delete tool permission")
        raise HTTPException(status_code=500, detail="Failed to delete tool permission") from e


@router.post("/roles/{role_id}/permissions/tools", response_model=ToolPermissionResponse)
async def grant_tool_permission_to_role(
    role_id: int, payload: ToolPermissionGrantRequest, db=Depends(get_db_transaction)
) -> ToolPermissionResponse:
    """Grant a tool execution permission to a role.

    - tool_name='*' → grants tools.execute:*
    - tool_name='<name>' → grants tools.execute:<name>
    Creates the permission in catalog if missing.
    """
    tool = payload.tool_name.strip()
    name = f"tools.execute:{'*' if tool == '*' else tool}"
    desc = "Wildcard tool execution" if tool == "*" else f"Execute tool {tool}"
    try:
        perm = await svc_grant_tool_perm(db, role_id, name, desc)
        return ToolPermissionResponse(
            name=perm["name"], description=perm.get("description"), category=perm.get("category")
        )
    except HTTPException:
        raise
    except _RBAC_NONCRITICAL_EXCEPTIONS as e:
        logger.error("Failed to grant tool permission")
        raise HTTPException(status_code=500, detail="Failed to grant tool permission") from e


@router.delete("/roles/{role_id}/permissions/tools/{tool_name}")
async def revoke_tool_permission_from_role(role_id: int, tool_name: str, db=Depends(get_db_transaction)) -> dict:
    """Revoke a tool execution permission from a role.

    tool_name '*' refers to tools.execute:*
    """
    name = f"tools.execute:{'*' if tool_name.strip() == '*' else tool_name.strip()}"
    try:
        ok = await svc_revoke_tool_perm(db, role_id, name)
        if not ok:
            return {"message": "Permission not found; nothing to revoke", "name": name}
        return {"message": "Tool permission revoked", "name": name}
    except _RBAC_NONCRITICAL_EXCEPTIONS as e:
        logger.error("Failed to revoke tool permission")
        raise HTTPException(status_code=500, detail="Failed to revoke tool permission") from e


@router.get("/roles/{role_id}/permissions/tools", response_model=list[ToolPermissionResponse])
async def list_role_tool_permissions(role_id: int, db=Depends(get_db_transaction)) -> list[ToolPermissionResponse]:
    """List tool execution permissions assigned to a role."""
    try:
        is_pg = await _get_is_postgres_backend_fn()()
        if is_pg:
            rows = await db.fetch(
                """
                SELECT p.name, p.description, p.category
                FROM permissions p
                JOIN role_permissions rp ON rp.permission_id = p.id
                WHERE rp.role_id = $1 AND p.name LIKE 'tools.execute:%'
                ORDER BY p.name
                """,
                role_id,
            )
            return [ToolPermissionResponse(**dict(r)) for r in rows]
        else:
            cur = await db.execute(
                """
                SELECT p.name, p.description, p.category
                FROM permissions p
                JOIN role_permissions rp ON rp.permission_id = p.id
                WHERE rp.role_id = ? AND p.name LIKE 'tools.execute:%'
                ORDER BY p.name
                """,
                (role_id,),
            )
            rows = await cur.fetchall()
            return [ToolPermissionResponse(name=r[0], description=r[1], category=r[2]) for r in rows]
    except _RBAC_NONCRITICAL_EXCEPTIONS as e:
        logger.error("Failed to list role tool permissions")
        raise HTTPException(status_code=500, detail="Failed to list role tool permissions") from e


@router.post("/roles/{role_id}/permissions/tools/batch", response_model=list[ToolPermissionResponse])
async def grant_tool_permissions_batch(
    role_id: int, payload: ToolPermissionBatchRequest, db=Depends(get_db_transaction)
) -> list[ToolPermissionResponse]:
    """Grant multiple tool execution permissions to a role in one call."""
    try:
        is_pg = await _get_is_postgres_backend_fn()()
        results: list[ToolPermissionResponse] = []
        # Reuse single-grant logic inline
        for tool in payload.tool_names:
            tool = tool.strip()
            if not tool:
                continue
            name = f"tools.execute:{'*' if tool == '*' else tool}"
            desc = "Wildcard tool execution" if tool == "*" else f"Execute tool {tool}"

            if is_pg:
                await db.execute(
                    "INSERT INTO permissions (name, description, category) VALUES ($1, $2, $3) ON CONFLICT (name) DO NOTHING",
                    name,
                    desc,
                    "tools",
                )
                row = await db.fetchrow("SELECT id, name, description, category FROM permissions WHERE name = $1", name)
                if not row:
                    continue
                await db.execute(
                    "INSERT INTO role_permissions (role_id, permission_id) VALUES ($1, $2) ON CONFLICT DO NOTHING",
                    role_id,
                    row["id"],
                )
                results.append(
                    ToolPermissionResponse(name=row["name"], description=row["description"], category=row["category"])
                )
            else:
                cur = await db.execute(
                    "SELECT id, name, description, category FROM permissions WHERE name = ?", (name,)
                )
                r = await cur.fetchone()
                if not r:
                    await db.execute(
                        "INSERT INTO permissions (name, description, category) VALUES (?, ?, ?)", (name, desc, "tools")
                    )
                    await db.commit()
                    cur = await db.execute(
                        "SELECT id, name, description, category FROM permissions WHERE name = ?", (name,)
                    )
                    r = await cur.fetchone()
                await db.execute(
                    "INSERT OR IGNORE INTO role_permissions (role_id, permission_id) VALUES (?, ?)", (role_id, r[0])
                )
                await db.commit()
                results.append(ToolPermissionResponse(name=r[1], description=r[2], category=r[3]))
        return results
    except _RBAC_NONCRITICAL_EXCEPTIONS as e:
        logger.error("Failed to grant tool permissions")
        raise HTTPException(status_code=500, detail="Failed to grant tool permissions") from e


@router.post("/roles/{role_id}/permissions/tools/batch/revoke")
async def revoke_tool_permissions_batch(
    role_id: int, payload: ToolPermissionBatchRequest, db=Depends(get_db_transaction)
) -> dict:
    """Revoke multiple tool execution permissions from a role."""
    try:
        is_pg = await _get_is_postgres_backend_fn()()
        revoked: list[str] = []
        for tool in payload.tool_names:
            tool = tool.strip()
            if not tool:
                continue
            name = f"tools.execute:{'*' if tool == '*' else tool}"
            if is_pg:
                row = await db.fetchrow("SELECT id FROM permissions WHERE name = $1", name)
                if row:
                    await db.execute(
                        "DELETE FROM role_permissions WHERE role_id = $1 AND permission_id = $2", role_id, row["id"]
                    )
                    revoked.append(name)
            else:
                cur = await db.execute("SELECT id FROM permissions WHERE name = ?", (name,))
                r = await cur.fetchone()
                if r:
                    await db.execute(
                        "DELETE FROM role_permissions WHERE role_id = ? AND permission_id = ?", (role_id, r[0])
                    )
                    await db.commit()
                    revoked.append(name)
        return {"revoked": revoked, "count": len(revoked)}
    except _RBAC_NONCRITICAL_EXCEPTIONS as e:
        logger.error("Failed to revoke tool permissions")
        raise HTTPException(status_code=500, detail="Failed to revoke tool permissions") from e


def _normalize_tool_prefix(raw_prefix: str) -> str:
    px = raw_prefix.strip()
    if not px:
        return "tools.execute:"
    if not px.startswith("tools.execute:"):
        px = "tools.execute:" + px
    return px


@router.post("/roles/{role_id}/permissions/tools/prefix/grant", response_model=list[ToolPermissionResponse])
async def grant_tool_permissions_by_prefix(
    role_id: int, payload: ToolPermissionPrefixRequest, db=Depends(get_db_transaction)
) -> list[ToolPermissionResponse]:
    """Grant all existing tool permissions with names starting with the prefix."""
    try:
        is_pg = await _get_is_postgres_backend_fn()()
        prefix = _normalize_tool_prefix(payload.prefix)
        results: list[ToolPermissionResponse] = []
        if is_pg:
            rows = await db.fetch(
                "SELECT id, name, description, category FROM permissions WHERE name LIKE $1", prefix + "%"
            )
            for r in rows:
                await db.execute(
                    "INSERT INTO role_permissions (role_id, permission_id) VALUES ($1, $2) ON CONFLICT DO NOTHING",
                    role_id,
                    r["id"],
                )
                results.append(
                    ToolPermissionResponse(name=r["name"], description=r["description"], category=r["category"])
                )
        else:
            cur = await db.execute(
                "SELECT id, name, description, category FROM permissions WHERE name LIKE ?", (prefix + "%",)
            )
            rows = await cur.fetchall()
            for r in rows:
                await db.execute(
                    "INSERT OR IGNORE INTO role_permissions (role_id, permission_id) VALUES (?, ?)", (role_id, r[0])
                )
                await db.commit()
                results.append(ToolPermissionResponse(name=r[1], description=r[2], category=r[3]))
        return results
    except _RBAC_NONCRITICAL_EXCEPTIONS as e:
        logger.error("Failed to grant tool permissions by prefix")
        raise HTTPException(status_code=500, detail="Failed to grant permissions by prefix") from e


@router.post("/roles/{role_id}/permissions/tools/prefix/revoke")
async def revoke_tool_permissions_by_prefix(
    role_id: int, payload: ToolPermissionPrefixRequest, db=Depends(get_db_transaction)
) -> dict:
    """Revoke all tool permissions with names starting with the prefix from a role."""
    try:
        is_pg = await _get_is_postgres_backend_fn()()
        prefix = _normalize_tool_prefix(payload.prefix)
        names: list[str] = []
        if is_pg:
            rows = await db.fetch("SELECT id, name FROM permissions WHERE name LIKE $1", prefix + "%")
            for r in rows:
                await db.execute(
                    "DELETE FROM role_permissions WHERE role_id = $1 AND permission_id = $2", role_id, r["id"]
                )
                names.append(r["name"])
        else:
            cur = await db.execute("SELECT id, name FROM permissions WHERE name LIKE ?", (prefix + "%",))
            rows = await cur.fetchall()
            for r in rows:
                await db.execute(
                    "DELETE FROM role_permissions WHERE role_id = ? AND permission_id = ?", (role_id, r[0])
                )
                await db.commit()
                names.append(r[1])
        return {"revoked": names, "count": len(names)}
    except _RBAC_NONCRITICAL_EXCEPTIONS as e:
        logger.error("Failed to revoke tool permissions by prefix")
        raise HTTPException(status_code=500, detail="Failed to revoke permissions by prefix") from e


@router.get("/roles/matrix", response_model=RolePermissionMatrixResponse)
async def get_roles_matrix(
    category: str | None = Query(None),
    search: str | None = Query(None),
    role_search: str | None = Query(None),
    role_names: list[str] | None = Query(None),
    roles_limit: int | None = Query(100, ge=1, le=10000),
    roles_offset: int | None = Query(0, ge=0),
    db=Depends(get_db_transaction),
) -> RolePermissionMatrixResponse:
    """Return roles, filtered permissions, and grants (matrix view).

    Optional filters:
    - category: permission category exact match
    - search: substring match on name/description (case-insensitive)
    """
    try:
        is_pg = await _get_is_postgres_backend_fn()()
        # Role filters + pagination
        role_clauses = []
        role_params: list[Any] = []
        total_roles = 0
        if is_pg:
            # Postgres
            if role_search:
                role_clauses.append(f"name ILIKE ${len(role_params)+1}")
                role_params.append(f"%{role_search}%")
            if role_names:
                role_clauses.append(f"name = ANY(${len(role_params)+1})")
                role_params.append(role_names)
            role_where = (" WHERE " + " AND ".join(role_clauses)) if role_clauses else ""
            # total count
            total_roles = await db.fetchval(
                f"SELECT COUNT(*) FROM roles{role_where}",  # nosec B608
                *role_params,
            )
            # fetch with limit/offset
            role_rows = await db.fetch(
                f"SELECT id, name, description, COALESCE(is_system,0) as is_system FROM roles{role_where} ORDER BY name LIMIT ${len(role_params)+1} OFFSET ${len(role_params)+2}",  # nosec B608
                *role_params,
                roles_limit,
                roles_offset,
            )
            roles = [RoleResponse(**dict(r)) for r in role_rows]
        else:
            # SQLite
            if role_search:
                role_clauses.append("name LIKE ?")
                role_params.append(f"%{role_search}%")
            if role_names:
                placeholders = ",".join(["?"] * len(role_names))
                role_clauses.append(f"name IN ({placeholders})")
                role_params.extend(role_names)
            role_where = (" WHERE " + " AND ".join(role_clauses)) if role_clauses else ""
            # total count
            cur = await db.execute(f"SELECT COUNT(*) FROM roles{role_where}", role_params)  # nosec B608
            row = await cur.fetchone()
            total_roles = int(row[0]) if row else 0
            # fetch with limit/offset
            cur = await db.execute(
                f"SELECT id, name, description, COALESCE(is_system,0) FROM roles{role_where} ORDER BY name LIMIT ? OFFSET ?",  # nosec B608
                [*role_params, roles_limit, roles_offset],
            )
            role_rows = await cur.fetchall()
            roles = [
                RoleResponse(id=row[0], name=row[1], description=row[2], is_system=bool(row[3])) for row in role_rows
            ]

        # Build WHERE for permissions
        clauses = []
        params: list[Any] = []
        if is_pg:
            # Postgres
            if category:
                clauses.append(f"category = ${len(params)+1}")
                params.append(category)
            if search:
                idx = len(params) + 1
                clauses.append(f"(name ILIKE ${idx} OR description ILIKE ${idx})")
                params.append(f"%{search}%")
            where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
            perm_rows = await db.fetch(
                f"SELECT id, name, description, category FROM permissions{where} ORDER BY name", *params  # nosec B608
            )
            permissions = [PermissionResponse(**dict(r)) for r in perm_rows]

            # Grants limited to filtered permissions via join
            grant_sql = (  # nosec B608
                "SELECT rp.role_id, rp.permission_id "  # nosec B608
                "FROM role_permissions rp "
                "JOIN permissions p ON p.id = rp.permission_id"
                f"{where}"
            )
            grant_rows = await db.fetch(grant_sql, *params)  # nosec B608
            grants = [RolePermissionGrant(role_id=r["role_id"], permission_id=r["permission_id"]) for r in grant_rows]
        else:
            # SQLite
            if category:
                clauses.append("category = ?")
                params.append(category)
            if search:
                clauses.append("(name LIKE ? OR description LIKE ?)")
                params.extend([f"%{search}%", f"%{search}%"])
            where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
            cur = await db.execute(
                f"SELECT id, name, description, category FROM permissions{where} ORDER BY name",  # nosec B608
                params,
            )
            perm_rows = await cur.fetchall()
            permissions = [
                PermissionResponse(id=row[0], name=row[1], description=row[2], category=row[3]) for row in perm_rows
            ]

            grant_sql = (  # nosec B608
                "SELECT rp.role_id, rp.permission_id "  # nosec B608
                "FROM role_permissions rp "
                "JOIN permissions p ON p.id = rp.permission_id"
                f"{where}"
            )
            cur = await db.execute(grant_sql, params)  # nosec B608
            grant_rows = await cur.fetchall()
            grants = [RolePermissionGrant(role_id=row[0], permission_id=row[1]) for row in grant_rows]

        return RolePermissionMatrixResponse(
            roles=roles, permissions=permissions, grants=grants, total_roles=total_roles
        )
    except _RBAC_NONCRITICAL_EXCEPTIONS as e:
        logger.error("Failed to build roles/permissions matrix")
        raise HTTPException(status_code=500, detail="Failed to fetch role-permission matrix") from e


@router.get("/roles/matrix-boolean", response_model=RolePermissionBooleanMatrixResponse)
async def get_roles_matrix_boolean(
    category: str | None = Query(None),
    search: str | None = Query(None),
    role_search: str | None = Query(None),
    role_names: list[str] | None = Query(None),
    roles_limit: int | None = Query(100, ge=1, le=10000),
    roles_offset: int | None = Query(0, ge=0),
    db=Depends(get_db_transaction),
) -> RolePermissionBooleanMatrixResponse:
    """Return a compact boolean matrix: roles x permission_names, with optional filters."""
    try:
        is_pg = await _get_is_postgres_backend_fn()()
        # Roles with filters + pagination
        role_clauses = []
        role_params: list[Any] = []
        total_roles = 0
        if is_pg:
            if role_search:
                role_clauses.append(f"name ILIKE ${len(role_params)+1}")
                role_params.append(f"%{role_search}%")
            if role_names:
                role_clauses.append(f"name = ANY(${len(role_params)+1})")
                role_params.append(role_names)
            role_where = (" WHERE " + " AND ".join(role_clauses)) if role_clauses else ""
            total_roles = await db.fetchval(
                f"SELECT COUNT(*) FROM roles{role_where}",  # nosec B608
                *role_params,
            )
            role_rows = await db.fetch(
                f"SELECT id, name, description, COALESCE(is_system,0) as is_system FROM roles{role_where} ORDER BY name LIMIT ${len(role_params)+1} OFFSET ${len(role_params)+2}",  # nosec B608
                *role_params,
                roles_limit,
                roles_offset,
            )
            roles = [RoleResponse(**dict(r)) for r in role_rows]
        else:
            if role_search:
                role_clauses.append("name LIKE ?")
                role_params.append(f"%{role_search}%")
            if role_names:
                placeholders = ",".join(["?"] * len(role_names))
                role_clauses.append(f"name IN ({placeholders})")
                role_params.extend(role_names)
            role_where = (" WHERE " + " AND ".join(role_clauses)) if role_clauses else ""
            cur = await db.execute(f"SELECT COUNT(*) FROM roles{role_where}", role_params)  # nosec B608
            row = await cur.fetchone()
            total_roles = int(row[0]) if row else 0
            cur = await db.execute(
                f"SELECT id, name, description, COALESCE(is_system,0) FROM roles{role_where} ORDER BY name LIMIT ? OFFSET ?",  # nosec B608
                [*role_params, roles_limit, roles_offset],
            )
            role_rows = await cur.fetchall()
            roles = [
                RoleResponse(id=row[0], name=row[1], description=row[2], is_system=bool(row[3])) for row in role_rows
            ]

        # Build WHERE for permissions
        clauses = []
        params: list[Any] = []
        if is_pg:
            if category:
                clauses.append(f"category = ${len(params)+1}")
                params.append(category)
            if search:
                idx = len(params) + 1
                clauses.append(f"(name ILIKE ${idx} OR description ILIKE ${idx})")
                params.append(f"%{search}%")
            where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
            perm_rows = await db.fetch(
                f"SELECT id, name FROM permissions{where} ORDER BY name",  # nosec B608
                *params,
            )
            perm_ids = [r["id"] for r in perm_rows]
            perm_names = [r["name"] for r in perm_rows]
        else:
            if category:
                clauses.append("category = ?")
                params.append(category)
            if search:
                clauses.append("(name LIKE ? OR description LIKE ?)")
                params.extend([f"%{search}%", f"%{search}%"])
            where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
            cur = await db.execute(f"SELECT id, name FROM permissions{where} ORDER BY name", params)  # nosec B608
            perm_rows = await cur.fetchall()
            perm_ids = [row[0] for row in perm_rows]
            perm_names = [row[1] for row in perm_rows]

        # Grants set (also restrict to selected roles if any)
        if is_pg:
            role_ids = [r.id for r in roles]
            grant_sql = (  # nosec B608
                "SELECT rp.role_id, rp.permission_id "  # nosec B608
                "FROM role_permissions rp "
                "JOIN permissions p ON p.id = rp.permission_id"
                f"{where}"
            )
            grant_params = list(params)
            if role_ids:
                grant_sql += f" AND rp.role_id = ANY(${len(grant_params)+1})"
                grant_params.append(role_ids)
            grant_rows = await db.fetch(grant_sql, *grant_params)  # nosec B608
            grants_set = {(r["role_id"], r["permission_id"]) for r in grant_rows}
        else:
            role_ids = [r.id for r in roles]
            grant_sql = (  # nosec B608
                "SELECT rp.role_id, rp.permission_id "  # nosec B608
                "FROM role_permissions rp "
                "JOIN permissions p ON p.id = rp.permission_id"
                f"{where}"
            )
            grant_params = list(params)
            if role_ids:
                placeholders = ",".join(["?"] * len(role_ids))
                grant_sql += f" AND rp.role_id IN ({placeholders})"
                grant_params.extend(role_ids)
            cur = await db.execute(grant_sql, grant_params)  # nosec B608
            grant_rows = await cur.fetchall()
            grants_set = {(row[0], row[1]) for row in grant_rows}

        # Build matrix: rows per role, cols per permission (same order as perm_names)
        role_ids = [r.id for r in roles]
        matrix: list[list[bool]] = []
        for rid in role_ids:
            row = [(rid, pid) in grants_set for pid in perm_ids]
            matrix.append(row)

        return RolePermissionBooleanMatrixResponse(
            roles=roles,
            permission_names=perm_names,
            matrix=matrix,
            total_roles=total_roles,
        )
    except _RBAC_NONCRITICAL_EXCEPTIONS as e:
        logger.error("Failed to build boolean matrix")
        raise HTTPException(status_code=500, detail="Failed to fetch boolean matrix") from e


@router.get("/permissions/categories", response_model=list[str])
async def list_permission_categories(db=Depends(get_db_transaction)) -> list[str]:
    """List distinct permission categories (for UI filters)."""
    try:
        is_pg = await _get_is_postgres_backend_fn()()
        if is_pg:
            rows = await db.fetch(
                "SELECT DISTINCT category FROM permissions WHERE category IS NOT NULL ORDER BY category"
            )
            return [r["category"] for r in rows]
        else:
            cur = await db.execute(
                "SELECT DISTINCT category FROM permissions WHERE category IS NOT NULL ORDER BY category"
            )
            rows = await cur.fetchall()
            return [row[0] for row in rows]
    except _RBAC_NONCRITICAL_EXCEPTIONS:
        logger.error("Failed to list permission categories")
        return []


@router.get("/permissions", response_model=list[PermissionResponse])
async def list_permissions(
    category: str | None = None, search: str | None = None, db=Depends(get_db_transaction)
) -> list[PermissionResponse]:
    try:
        is_pg = await _get_is_postgres_backend_fn()()
        clauses = []
        params = []
        if category:
            clauses.append("category = $1" if is_pg else "category = ?")
            params.append(category)
        if search:
            if is_pg:
                clauses.append(f"(name ILIKE ${len(params)+1} OR description ILIKE ${len(params)+1})")
                params.append(f"%{search}%")
            else:
                clauses.append("(name LIKE ? OR description LIKE ?)")
                params.append(f"%{search}%")
                params.append(f"%{search}%")
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        if is_pg:
            rows = await db.fetch(
                f"SELECT id, name, description, category FROM permissions{where} ORDER BY name",  # nosec B608
                *params,
            )
            return [PermissionResponse(**dict(r)) for r in rows]
        else:
            cur = await db.execute(
                f"SELECT id, name, description, category FROM permissions{where} ORDER BY name",  # nosec B608
                params,
            )
            rows = await cur.fetchall()
            return [PermissionResponse(id=row[0], name=row[1], description=row[2], category=row[3]) for row in rows]
    except _RBAC_NONCRITICAL_EXCEPTIONS as e:
        logger.error("Failed to list permissions")
        raise HTTPException(status_code=500, detail="Failed to list permissions") from e


@router.post("/permissions", response_model=PermissionResponse)
async def create_permission(payload: PermissionCreateRequest, db=Depends(get_db_transaction)) -> PermissionResponse:
    try:
        is_pg = await _get_is_postgres_backend_fn()()
        if is_pg:
            # Pre-check (case-insensitive)
            exists = await db.fetchrow("SELECT 1 FROM permissions WHERE LOWER(name) = LOWER($1)", payload.name)
            if exists:
                raise HTTPException(status_code=409, detail=f"Permission '{payload.name}' already exists")
            row = await db.fetchrow(
                "INSERT INTO permissions (name, description, category) VALUES ($1, $2, $3) RETURNING id, name, description, category",
                payload.name,
                payload.description,
                payload.category,
            )
            return PermissionResponse(**dict(row))
        else:
            # SQLite: explicit pre-check, return 409 if exists (case-insensitive)
            curx = await db.execute(
                "SELECT 1 FROM permissions WHERE LOWER(name) = LOWER(?)",
                (payload.name,),
            )
            if await curx.fetchone():
                raise HTTPException(status_code=409, detail=f"Permission '{payload.name}' already exists")
            await db.execute(
                "INSERT INTO permissions (name, description, category) VALUES (?, ?, ?)",
                (payload.name, payload.description, payload.category),
            )
            # Fetch the row via adapter
            cur = await db.execute(
                "SELECT id, name, description, category FROM permissions WHERE name = ?",
                (payload.name,),
            )
            row = await cur.fetchone()
            try:
                if isinstance(row, dict):
                    return PermissionResponse(**row)
            except _RBAC_NONCRITICAL_EXCEPTIONS:
                pass
            return PermissionResponse(id=row[0], name=row[1], description=row[2], category=row[3])
    except HTTPException:
        # Preserve explicit status codes like 409 Conflict
        raise
    except _RBAC_NONCRITICAL_EXCEPTIONS as e:
        logger.error("Failed to create permission")
        raise HTTPException(status_code=500, detail="Failed to create permission") from e


@router.post("/roles/{role_id}/permissions/{permission_id}")
async def grant_permission_to_role(role_id: int, permission_id: int, db=Depends(get_db_transaction)) -> dict:
    try:
        is_pg = await _get_is_postgres_backend_fn()()
        if is_pg:
            await db.execute(
                "INSERT INTO role_permissions (role_id, permission_id) VALUES ($1, $2) ON CONFLICT DO NOTHING",
                role_id,
                permission_id,
            )
        else:
            await db.execute(
                "INSERT OR IGNORE INTO role_permissions (role_id, permission_id) VALUES (?, ?)",
                (role_id, permission_id),
            )
            await db.commit()
        return {"message": "Permission granted to role"}
    except _RBAC_NONCRITICAL_EXCEPTIONS as e:
        logger.error("Failed to grant permission to role")
        raise HTTPException(status_code=500, detail="Failed to grant permission to role") from e


@router.delete("/roles/{role_id}/permissions/{permission_id}")
async def revoke_permission_from_role(role_id: int, permission_id: int, db=Depends(get_db_transaction)) -> dict:
    try:
        is_pg = await _get_is_postgres_backend_fn()()
        if is_pg:
            await db.execute(
                "DELETE FROM role_permissions WHERE role_id = $1 AND permission_id = $2", role_id, permission_id
            )
        else:
            await db.execute(
                "DELETE FROM role_permissions WHERE role_id = ? AND permission_id = ?", (role_id, permission_id)
            )
            await db.commit()
        return {"message": "Permission revoked from role"}
    except _RBAC_NONCRITICAL_EXCEPTIONS as e:
        logger.error("Failed to revoke permission from role")
        raise HTTPException(status_code=500, detail="Failed to revoke permission from role") from e


@router.get("/users/{user_id}/roles", response_model=UserRoleListResponse)
async def get_user_roles_admin(
    user_id: int,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> UserRoleListResponse:
    try:
        await _enforce_admin_user_scope(principal, user_id, require_hierarchy=False)
        repo = _get_rbac_repo()
        loop = asyncio.get_event_loop()
        rows = await loop.run_in_executor(None, repo.get_user_roles, int(user_id))
        roles = [
            RoleResponse(
                id=int(r.get("id")),
                name=str(r.get("name")),
                description=str(r.get("description") or ""),
                is_system=bool(r.get("is_system")),
            )
            for r in rows
        ]
        return UserRoleListResponse(user_id=user_id, roles=roles)
    except _RBAC_NONCRITICAL_EXCEPTIONS as e:
        logger.error("Failed to get user roles")
        raise HTTPException(status_code=500, detail="Failed to get user roles") from e


@router.post("/users/{user_id}/roles/{role_id}")
async def add_role_to_user(
    user_id: int,
    role_id: int,
    principal: AuthPrincipal = Depends(get_auth_principal),
    db=Depends(get_db_transaction),
) -> dict:
    try:
        await _enforce_admin_user_scope(principal, user_id, require_hierarchy=True)
        is_pg = await _get_is_postgres_backend_fn()()
        if is_pg:
            await db.execute(
                "INSERT INTO user_roles (user_id, role_id) VALUES ($1, $2) ON CONFLICT (user_id, role_id) DO NOTHING",
                user_id,
                role_id,
            )
        else:
            await db.execute(
                "INSERT OR IGNORE INTO user_roles (user_id, role_id) VALUES (?, ?)",
                (user_id, role_id),
            )
            await db.commit()
        return {"message": "Role added to user"}
    except _RBAC_NONCRITICAL_EXCEPTIONS as e:
        logger.error("Failed to add role to user")
        raise HTTPException(status_code=500, detail="Failed to add role to user") from e


@router.delete("/users/{user_id}/roles/{role_id}")
async def remove_role_from_user(
    user_id: int,
    role_id: int,
    principal: AuthPrincipal = Depends(get_auth_principal),
    db=Depends(get_db_transaction),
) -> dict:
    try:
        await _enforce_admin_user_scope(principal, user_id, require_hierarchy=True)
        is_pg = await _get_is_postgres_backend_fn()()
        if is_pg:
            await db.execute("DELETE FROM user_roles WHERE user_id = $1 AND role_id = $2", user_id, role_id)
        else:
            await db.execute("DELETE FROM user_roles WHERE user_id = ? AND role_id = ?", (user_id, role_id))
            await db.commit()
        return {"message": "Role removed from user"}
    except _RBAC_NONCRITICAL_EXCEPTIONS as e:
        logger.error("Failed to remove role from user")
        raise HTTPException(status_code=500, detail="Failed to remove role from user") from e


@router.get("/users/{user_id}/overrides", response_model=UserOverridesResponse)
async def list_user_overrides(
    user_id: int,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> UserOverridesResponse:
    try:
        await _enforce_admin_user_scope(principal, user_id, require_hierarchy=False)
        repo = _get_rbac_repo()
        rows = repo.get_user_overrides(user_id=int(user_id))
        entries = [
            UserOverrideEntry(
                permission_id=int(r.get("permission_id")),
                permission_name=str(r.get("permission_name")),
                granted=bool(r.get("granted")),
                expires_at=str(r.get("expires_at")) if r.get("expires_at") else None,
            )
            for r in rows
        ]
        return UserOverridesResponse(user_id=user_id, overrides=entries)
    except _RBAC_NONCRITICAL_EXCEPTIONS as e:
        logger.error("Failed to list user overrides")
        raise HTTPException(status_code=500, detail="Failed to list user overrides") from e


@router.post("/users/{user_id}/overrides")
async def upsert_user_override(
    user_id: int,
    payload: UserOverrideUpsertRequest,
    principal: AuthPrincipal = Depends(get_auth_principal),
    db=Depends(get_db_transaction),
) -> dict:
    try:
        await _enforce_admin_user_scope(principal, user_id, require_hierarchy=True)
        from tldw_Server_API.app.core.AuthNZ.settings import get_settings as _get_settings

        _settings = _get_settings()
        _is_pg = await _get_is_postgres_backend_fn()()
        # In single-user mode, ensure the fixed user row exists before applying overrides (SQLite/PG FK safety)
        from tldw_Server_API.app.core.AuthNZ.settings import is_single_user_mode as _is_single

        if _is_single() and int(user_id) == int(getattr(_settings, "SINGLE_USER_FIXED_ID", 1)):
            if _is_pg:
                await db.execute(
                    """
                    INSERT INTO users (id, username, email, password_hash, is_active, is_verified, role)
                    VALUES ($1, $2, $3, $4, TRUE, TRUE, COALESCE((SELECT role FROM users WHERE id=$1),'user'))
                    ON CONFLICT (id) DO NOTHING
                    """,
                    user_id,
                    "single_user",
                    "single_user@example.local",
                    "",
                )
            else:
                # SQLite path: insert a stub single_user row with default role 'user'
                cur = await db.execute(
                    """
                    INSERT OR IGNORE INTO users (id, username, email, password_hash, is_active, is_verified, role)
                    VALUES (?, ?, ?, ?, 1, 1, 'user')
                    """,
                    user_id,
                    "single_user",
                    "single_user@example.local",
                    "",
                )
                if not _is_pg:
                    await db.commit()
        # Resolve permission_id if only name provided
        perm_id = payload.permission_id
        if not perm_id and payload.permission_name:
            if _is_pg:
                perm_id = await db.fetchval("SELECT id FROM permissions WHERE name = $1", payload.permission_name)
            else:
                cur = await db.execute("SELECT id FROM permissions WHERE name = ?", (payload.permission_name,))
                row = await cur.fetchone()
                perm_id = row[0] if row else None
        if not perm_id:
            raise HTTPException(status_code=400, detail="permission_id or permission_name required")

        granted = payload.effect == OverrideEffect.allow
        if _is_pg:
            await db.execute(
                """
                INSERT INTO user_permissions (user_id, permission_id, granted, expires_at)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (user_id, permission_id)
                DO UPDATE SET granted = EXCLUDED.granted, expires_at = EXCLUDED.expires_at
                """,
                user_id,
                perm_id,
                granted,
                payload.expires_at,
            )
        else:
            cur = await db.execute(
                """
                INSERT OR REPLACE INTO user_permissions (user_id, permission_id, granted, expires_at)
                VALUES (?, ?, ?, ?)
                """,
                user_id,
                perm_id,
                granted,
                payload.expires_at,
            )
            # Commit on SQLite acquire()-based connection
            if not _is_pg:
                await db.commit()
        return {"message": "Override upserted"}
    except HTTPException:
        raise
    except _RBAC_NONCRITICAL_EXCEPTIONS as e:
        logger.exception("Failed to upsert user override")
        raise HTTPException(status_code=500, detail="Failed to upsert user override") from e


@router.delete("/users/{user_id}/overrides/{permission_id}")
async def delete_user_override(
    user_id: int,
    permission_id: int,
    principal: AuthPrincipal = Depends(get_auth_principal),
    db=Depends(get_db_transaction),
) -> dict:
    try:
        await _enforce_admin_user_scope(principal, user_id, require_hierarchy=True)
        _is_pg = await _get_is_postgres_backend_fn()()
        if _is_pg:
            await db.execute(
                "DELETE FROM user_permissions WHERE user_id = $1 AND permission_id = $2", user_id, permission_id
            )
        else:
            await db.execute(
                "DELETE FROM user_permissions WHERE user_id = ? AND permission_id = ?", (user_id, permission_id)
            )
            if not _is_pg:
                await db.commit()
        return {"message": "Override deleted"}
    except _RBAC_NONCRITICAL_EXCEPTIONS as e:
        logger.exception("Failed to delete user override")
        raise HTTPException(status_code=500, detail="Failed to delete user override") from e


@router.get("/users/{user_id}/effective-permissions", response_model=EffectivePermissionsResponse)
async def get_effective_permissions_admin(
    user_id: int,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> EffectivePermissionsResponse:
    """Compute effective permissions for a user.

    Delegates to the central RBAC helper, which in turn uses the AuthNZ
    repository layer (`AuthnzRbacRepo` / `UserDatabase_v2`) so that both
    SQLite and Postgres backends share the same logic.
    """
    try:
        await _enforce_admin_user_scope(principal, user_id, require_hierarchy=False)
        loop = asyncio.get_event_loop()
        perms = await loop.run_in_executor(None, get_effective_permissions, user_id)
        return EffectivePermissionsResponse(user_id=user_id, permissions=sorted(perms))
    except _RBAC_NONCRITICAL_EXCEPTIONS as e:
        logger.error("Failed to compute effective permissions")
        raise HTTPException(status_code=500, detail="Failed to compute effective permissions") from e


@router.get("/roles/{role_id}/permissions/effective", response_model=RoleEffectivePermissionsResponse)
async def get_role_effective_permissions(
    role_id: int,
    db=Depends(get_db_transaction),
) -> RoleEffectivePermissionsResponse:
    """Return a convenience view combining a role's granted permissions and tool permissions.

    - permissions: non-tool permission names (e.g., media.read)
    - tool_permissions: tool execution permission names (tools.execute:...)
    - all_permissions: union of both, sorted
    """
    try:
        _is_pg = await _get_is_postgres_backend_fn()()
        if _is_pg:
            role_row = await db.fetchrow(
                "SELECT id, name FROM roles WHERE id = $1",
                int(role_id),
            )
        else:
            cur = await db.execute(
                "SELECT id, name FROM roles WHERE id = ?",
                (int(role_id),),
            )
            role_row = await cur.fetchone()
        if not role_row:
            raise HTTPException(status_code=404, detail="Role not found")

        if _is_pg:
            role_name = str(role_row["name"])
        else:
            role_name = str(role_row[1])

        perm_rows = await svc_list_role_permissions(db, int(role_id))
        names = sorted(str(row.get("name")) for row in perm_rows if row.get("name"))
        tool_prefix = "tools.execute:"
        tool_permissions = [name for name in names if name.startswith(tool_prefix)]
        permissions = [name for name in names if not name.startswith(tool_prefix)]
        all_permissions = sorted(set(tool_permissions + permissions))

        return RoleEffectivePermissionsResponse(
            role_id=role_id,
            role_name=role_name,
            permissions=permissions,
            tool_permissions=tool_permissions,
            all_permissions=all_permissions,
        )
    except HTTPException:
        raise
    except _RBAC_NONCRITICAL_EXCEPTIONS as e:
        logger.error("Failed to compute role effective permissions")
        raise HTTPException(status_code=500, detail="Failed to compute role effective permissions") from e
