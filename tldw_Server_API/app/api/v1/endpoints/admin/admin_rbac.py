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
from tldw_Server_API.app.core.AuthNZ.profile_version import VersionedUserWriteGateway
from tldw_Server_API.app.core.AuthNZ.rbac import get_effective_permissions
from tldw_Server_API.app.core.AuthNZ.repos import rbac_admin_repo
from tldw_Server_API.app.core.AuthNZ.repos.rbac_repo import AuthnzRbacRepo
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.DB_Management.Kanban_DB import InputError, KanbanDB, KanbanDBError
from tldw_Server_API.app.core.exceptions import ResourceNotFoundError
from tldw_Server_API.app.core.testing import is_test_mode as _shared_is_test_mode
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


def is_test_mode() -> bool:
    """Return the shared test-mode flag for legacy admin RBAC patch points."""
    return _shared_is_test_mode()


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
        row = await rbac_admin_repo.ensure_permission(
            db, is_postgres=await _get_is_postgres_backend_fn()(), name=name, description=desc, category="tools"
        )
        return ToolPermissionResponse(name=row["name"], description=row["description"], category=row["category"])
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
        rows = await rbac_admin_repo.list_role_tool_permissions(
            db, is_postgres=await _get_is_postgres_backend_fn()(), role_id=role_id
        )
        return [ToolPermissionResponse(**row) for row in rows]
    except _RBAC_NONCRITICAL_EXCEPTIONS as e:
        logger.error("Failed to list role tool permissions")
        raise HTTPException(status_code=500, detail="Failed to list role tool permissions") from e


@router.post("/roles/{role_id}/permissions/tools/batch", response_model=list[ToolPermissionResponse])
async def grant_tool_permissions_batch(
    role_id: int, payload: ToolPermissionBatchRequest, db=Depends(get_db_transaction)
) -> list[ToolPermissionResponse]:
    """Grant multiple tool execution permissions to a role in one call."""
    try:
        wanted: list[tuple[str, str]] = []
        for tool in payload.tool_names:
            tool = tool.strip()
            if not tool:
                continue
            wanted.append(
                (
                    f"tools.execute:{'*' if tool == '*' else tool}",
                    "Wildcard tool execution" if tool == "*" else f"Execute tool {tool}",
                )
            )
        rows = await rbac_admin_repo.grant_tool_permissions(
            db, is_postgres=await _get_is_postgres_backend_fn()(), role_id=role_id, permissions=wanted
        )
        return [ToolPermissionResponse(name=r["name"], description=r["description"], category=r["category"]) for r in rows]
    except _RBAC_NONCRITICAL_EXCEPTIONS as e:
        logger.error("Failed to grant tool permissions")
        raise HTTPException(status_code=500, detail="Failed to grant tool permissions") from e


@router.post("/roles/{role_id}/permissions/tools/batch/revoke")
async def revoke_tool_permissions_batch(
    role_id: int, payload: ToolPermissionBatchRequest, db=Depends(get_db_transaction)
) -> dict:
    """Revoke multiple tool execution permissions from a role."""
    try:
        names = [
            f"tools.execute:{'*' if tool.strip() == '*' else tool.strip()}" for tool in payload.tool_names if tool.strip()
        ]
        revoked = await rbac_admin_repo.revoke_tool_permissions(
            db, is_postgres=await _get_is_postgres_backend_fn()(), role_id=role_id, names=names
        )
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
        rows = await rbac_admin_repo.permissions_with_prefix(
            db, is_postgres=is_pg, prefix=_normalize_tool_prefix(payload.prefix)
        )
        for row in rows:
            await rbac_admin_repo.grant_permission(db, is_postgres=is_pg, role_id=role_id, permission_id=int(row["id"]))
        return [ToolPermissionResponse(name=r["name"], description=r["description"], category=r["category"]) for r in rows]
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
        rows = await rbac_admin_repo.permissions_with_prefix(
            db, is_postgres=is_pg, prefix=_normalize_tool_prefix(payload.prefix)
        )
        for row in rows:
            await rbac_admin_repo.revoke_permission(db, is_postgres=is_pg, role_id=role_id, permission_id=int(row["id"]))
        names = [str(row["name"]) for row in rows]
        return {"revoked": names, "count": len(names)}
    except _RBAC_NONCRITICAL_EXCEPTIONS as e:
        logger.error("Failed to revoke tool permissions by prefix")
        raise HTTPException(status_code=500, detail="Failed to revoke permissions by prefix") from e


def _role_response(row: dict[str, Any]) -> RoleResponse:
    return RoleResponse(id=row["id"], name=row["name"], description=row["description"], is_system=bool(row["is_system"]))


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
        total_roles, role_rows = await rbac_admin_repo.roles_page(
            db, is_postgres=is_pg, role_search=role_search, role_names=role_names, limit=roles_limit, offset=roles_offset
        )
        permissions = await rbac_admin_repo.list_permissions(db, is_postgres=is_pg, category=category, search=search)
        grants = await rbac_admin_repo.role_permission_grants(db, is_postgres=is_pg, category=category, search=search)
        return RolePermissionMatrixResponse(
            roles=[_role_response(row) for row in role_rows],
            permissions=[PermissionResponse(**row) for row in permissions],
            grants=[RolePermissionGrant(role_id=role_id, permission_id=perm_id) for role_id, perm_id in grants],
            total_roles=total_roles,
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
        total_roles, role_rows = await rbac_admin_repo.roles_page(
            db, is_postgres=is_pg, role_search=role_search, role_names=role_names, limit=roles_limit, offset=roles_offset
        )
        roles = [_role_response(row) for row in role_rows]
        permissions = await rbac_admin_repo.list_permissions(db, is_postgres=is_pg, category=category, search=search)
        role_ids = [role.id for role in roles]
        grants_set = set(
            await rbac_admin_repo.role_permission_grants(
                db, is_postgres=is_pg, category=category, search=search, role_ids=role_ids
            )
        )
        perm_ids = [int(row["id"]) for row in permissions]
        return RolePermissionBooleanMatrixResponse(
            roles=roles,
            permission_names=[str(row["name"]) for row in permissions],
            matrix=[[(rid, pid) in grants_set for pid in perm_ids] for rid in role_ids],
            total_roles=total_roles,
        )
    except _RBAC_NONCRITICAL_EXCEPTIONS as e:
        logger.error("Failed to build boolean matrix")
        raise HTTPException(status_code=500, detail="Failed to fetch boolean matrix") from e


@router.get("/permissions/categories", response_model=list[str])
async def list_permission_categories(db=Depends(get_db_transaction)) -> list[str]:
    """List distinct permission categories (for UI filters)."""
    try:
        return await rbac_admin_repo.permission_categories(db, is_postgres=await _get_is_postgres_backend_fn()())
    except _RBAC_NONCRITICAL_EXCEPTIONS:
        logger.error("Failed to list permission categories")
        return []


@router.get("/permissions", response_model=list[PermissionResponse])
async def list_permissions(
    category: str | None = None, search: str | None = None, db=Depends(get_db_transaction)
) -> list[PermissionResponse]:
    try:
        rows = await rbac_admin_repo.list_permissions(
            db, is_postgres=await _get_is_postgres_backend_fn()(), category=category, search=search
        )
        return [PermissionResponse(**row) for row in rows]
    except _RBAC_NONCRITICAL_EXCEPTIONS as e:
        logger.error("Failed to list permissions")
        raise HTTPException(status_code=500, detail="Failed to list permissions") from e


@router.post("/permissions", response_model=PermissionResponse)
async def create_permission(payload: PermissionCreateRequest, db=Depends(get_db_transaction)) -> PermissionResponse:
    try:
        row = await rbac_admin_repo.create_permission(
            db,
            is_postgres=await _get_is_postgres_backend_fn()(),
            name=payload.name,
            description=payload.description,
            category=payload.category,
        )
        if row is None:
            raise HTTPException(status_code=409, detail=f"Permission '{payload.name}' already exists")
        return PermissionResponse(**row)
    except HTTPException:
        # Preserve explicit status codes like 409 Conflict
        raise
    except _RBAC_NONCRITICAL_EXCEPTIONS as e:
        logger.error("Failed to create permission")
        raise HTTPException(status_code=500, detail="Failed to create permission") from e


@router.post("/roles/{role_id}/permissions/{permission_id}")
async def grant_permission_to_role(role_id: int, permission_id: int, db=Depends(get_db_transaction)) -> dict:
    try:
        await rbac_admin_repo.grant_permission(
            db, is_postgres=await _get_is_postgres_backend_fn()(), role_id=role_id, permission_id=permission_id
        )
        return {"message": "Permission granted to role"}
    except _RBAC_NONCRITICAL_EXCEPTIONS as e:
        logger.error("Failed to grant permission to role")
        raise HTTPException(status_code=500, detail="Failed to grant permission to role") from e


@router.delete("/roles/{role_id}/permissions/{permission_id}")
async def revoke_permission_from_role(role_id: int, permission_id: int, db=Depends(get_db_transaction)) -> dict:
    try:
        await rbac_admin_repo.revoke_permission(
            db, is_postgres=await _get_is_postgres_backend_fn()(), role_id=role_id, permission_id=permission_id
        )
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
        await rbac_admin_repo.add_user_role(
            db, is_postgres=await _get_is_postgres_backend_fn()(), user_id=user_id, role_id=role_id
        )
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
        await rbac_admin_repo.remove_user_role(
            db, is_postgres=await _get_is_postgres_backend_fn()(), user_id=user_id, role_id=role_id
        )
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
            gateway = VersionedUserWriteGateway("postgres" if _is_pg else "sqlite")
            if _is_pg:
                await gateway.insert_user(
                    db,
                    values={
                        "id": user_id,
                        "username": "single_user",
                        "email": "single_user@example.local",
                        # This stub is never accepted as an authenticating password.
                        "password_hash": "",  # nosec B105
                        "is_active": True,
                        "is_verified": True,
                        "role": "user",
                    },
                    ignore_conflict=True,
                )
            else:
                # SQLite path: insert a stub single_user row with default role 'user'
                await gateway.insert_user(
                    db,
                    values={
                        "id": user_id,
                        "username": "single_user",
                        "email": "single_user@example.local",
                        # This stub is never accepted as an authenticating password.
                        "password_hash": "",  # nosec B105
                        "is_active": 1,
                        "is_verified": 1,
                        "role": "user",
                    },
                    ignore_conflict=True,
                )
        # Resolve permission_id if only name provided
        perm_id = payload.permission_id
        if not perm_id and payload.permission_name:
            perm_id = await rbac_admin_repo.permission_id_by_name(db, is_postgres=_is_pg, name=payload.permission_name)
        if not perm_id:
            raise HTTPException(status_code=400, detail="permission_id or permission_name required")

        await rbac_admin_repo.upsert_user_override(
            db,
            is_postgres=_is_pg,
            user_id=user_id,
            permission_id=int(perm_id),
            granted=payload.effect == OverrideEffect.allow,
            expires_at=payload.expires_at,
        )
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
        await rbac_admin_repo.delete_user_override(
            db, is_postgres=await _get_is_postgres_backend_fn()(), user_id=user_id, permission_id=permission_id
        )
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
        role_row = await rbac_admin_repo.get_role(db, is_postgres=await _get_is_postgres_backend_fn()(), role_id=role_id)
        if not role_row:
            raise HTTPException(status_code=404, detail="Role not found")
        role_name = str(role_row["name"])

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
