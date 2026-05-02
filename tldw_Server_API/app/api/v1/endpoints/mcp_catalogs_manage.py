"""
MCP Tool Catalog Management for Org/Team leads (non-admin paths)

Allows organization owners/admins and team leads/admins to manage tool catalogs
within their scope. Admins also permitted.
"""

from __future__ import annotations

from sqlite3 import Error as SQLiteError

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal, get_db_transaction
from tldw_Server_API.app.api.v1.schemas.admin_schemas import (
    ToolCatalogCreateRequest,
    ToolCatalogEntryCreateRequest,
    ToolCatalogEntryResponse,
    ToolCatalogResponse,
)
from tldw_Server_API.app.core.AuthNZ.orgs_teams import list_org_members, list_team_members
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.exceptions import ToolCatalogConflictError
from tldw_Server_API.app.services import admin_tool_catalog_service
from tldw_Server_API.app.services.mcp_hub_service import emit_mcp_hub_audit

router = APIRouter(prefix="", tags=["mcp-catalogs-scope"])
_CATALOG_NONCRITICAL_EXCEPTIONS = (
    AttributeError,
    LookupError,
    OSError,
    RuntimeError,
    SQLiteError,
    TimeoutError,
    TypeError,
    ValueError,
)
_CATALOG_MEMBER_PARSE_EXCEPTIONS = (AttributeError, TypeError, ValueError)
_CATALOG_PAGE_SIZE = 1000
_ADMIN_CLAIM_PERMISSIONS = frozenset({"*", "system.configure"})


def _is_manager(role: str | None) -> bool:
    if not role:
        return False
    r = str(role).lower()
    return r in {"owner", "admin", "lead"}


def _is_admin_principal(principal: AuthPrincipal) -> bool:
    roles = {str(role).strip().lower() for role in (principal.roles or []) if str(role).strip()}
    if "admin" in roles:
        return True
    permissions = {
        str(permission).strip().lower() for permission in (principal.permissions or []) if str(permission).strip()
    }
    return bool(permissions & _ADMIN_CLAIM_PERMISSIONS)


async def _require_org_manager(principal: AuthPrincipal, org_id: int) -> None:
    # Admin bypass
    if _is_admin_principal(principal):
        return
    if principal.user_id is None:
        raise HTTPException(status_code=403, detail="Org manager role required")
    # Check org membership
    try:
        members = await list_org_members(org_id=org_id, limit=1000, offset=0)
        uid = int(principal.user_id)
        for m in members:
            try:
                if int(m.get("user_id")) == uid and _is_manager(m.get("role")):
                    return
            except _CATALOG_MEMBER_PARSE_EXCEPTIONS:
                pass
    except _CATALOG_NONCRITICAL_EXCEPTIONS:
        logger.debug("Org manager check failed")
    raise HTTPException(status_code=403, detail="Org manager role required")


async def _require_team_manager(principal: AuthPrincipal, team_id: int) -> None:
    # Admin bypass
    if _is_admin_principal(principal):
        return
    if principal.user_id is None:
        raise HTTPException(status_code=403, detail="Team manager role required")
    try:
        members = await list_team_members(team_id)
        uid = int(principal.user_id)
        for m in members:
            try:
                if int(m.get("user_id")) == uid and _is_manager(m.get("role")):
                    return
            except _CATALOG_MEMBER_PARSE_EXCEPTIONS:
                pass
    except _CATALOG_NONCRITICAL_EXCEPTIONS:
        logger.debug("Team manager check failed")
    raise HTTPException(status_code=403, detail="Team manager role required")


async def _list_scoped_catalogs(
    db,
    *,
    org_id: int | None = None,
    team_id: int | None = None,
) -> list[dict]:
    """List all catalogs for a given scope via paged service queries."""
    rows: list[dict] = []
    offset = 0
    while True:
        batch = await admin_tool_catalog_service.list_tool_catalogs(
            db,
            org_id=org_id,
            team_id=team_id,
            limit=_CATALOG_PAGE_SIZE,
            offset=offset,
        )
        rows.extend(batch)
        if len(batch) < _CATALOG_PAGE_SIZE:
            break
        offset += _CATALOG_PAGE_SIZE
    return rows


async def _get_scoped_catalog(
    db,
    *,
    catalog_id: int,
    org_id: int | None = None,
    team_id: int | None = None,
) -> dict | None:
    """Fetch catalog by id and ensure it belongs to the requested scope."""
    catalog = await admin_tool_catalog_service.get_tool_catalog(db, catalog_id)
    if not catalog:
        return None
    if org_id is not None and int(catalog.get("org_id") or -1) != int(org_id):
        return None
    if team_id is not None and int(catalog.get("team_id") or -1) != int(team_id):
        return None
    return catalog


@router.get(
    "/orgs/{org_id}/mcp/tool_catalogs",
    response_model=list[ToolCatalogResponse],
    summary="List org-scoped MCP tool catalogs",
    description=(
        "List all MCP tool catalogs scoped to the specified organization.\n\n"
        "RBAC: Requires an org manager (owner/admin/lead) in this organization or a global admin.\n\n"
        "Scope: Results include only catalogs with `org_id` matching the path and `team_id` NULL."
    ),
)
async def list_org_tool_catalogs(
    org_id: int,
    principal: AuthPrincipal = Depends(get_auth_principal),
    db=Depends(get_db_transaction),
) -> list[ToolCatalogResponse]:
    await _require_org_manager(principal, org_id)
    try:
        rows = await _list_scoped_catalogs(db, org_id=org_id)
        return [ToolCatalogResponse(**r) for r in rows]
    except _CATALOG_NONCRITICAL_EXCEPTIONS as e:
        logger.error("Failed to list org tool catalogs")
        raise HTTPException(status_code=500, detail="Failed to list org tool catalogs") from e


@router.post(
    "/orgs/{org_id}/mcp/tool_catalogs",
    response_model=ToolCatalogResponse,
    status_code=201,
    summary="Create new org-scoped MCP tool catalog",
    description=(
        "Create an MCP tool catalog owned by the specified organization.\n\n"
        "RBAC: Requires an org manager (owner/admin/lead) in this organization or a global admin.\n\n"
        "Scope: Catalog is created with `org_id` set and `team_id` NULL. Name must be unique within the org scope."
    ),
)
async def create_org_tool_catalog(
    org_id: int,
    payload: ToolCatalogCreateRequest,
    principal: AuthPrincipal = Depends(get_auth_principal),
    db=Depends(get_db_transaction),
) -> ToolCatalogResponse:
    await _require_org_manager(principal, org_id)
    # Force scope to org
    name = payload.name.strip()
    desc = payload.description
    is_active = bool(payload.is_active if payload.is_active is not None else True)
    try:
        row = await admin_tool_catalog_service.create_tool_catalog(
            db,
            name=name,
            description=desc,
            org_id=org_id,
            team_id=None,
            is_active=is_active,
        )
        await emit_mcp_hub_audit(
            action="tool_catalog.created",
            actor_id=principal.user_id,
            resource_type="mcp_tool_catalog",
            resource_id=str(row.get("id")),
            metadata={"owner_scope_type": "org", "owner_scope_id": org_id, "name": row.get("name")},
        )
        return ToolCatalogResponse(**row)
    except ToolCatalogConflictError as exc:
        raise HTTPException(status_code=409, detail="Catalog already exists") from exc
    except HTTPException:
        raise
    except _CATALOG_NONCRITICAL_EXCEPTIONS as e:
        logger.error("Failed to create org tool catalog")
        raise HTTPException(status_code=500, detail="Failed to create tool catalog") from e


@router.get(
    "/teams/{team_id}/mcp/tool_catalogs",
    response_model=list[ToolCatalogResponse],
    summary="List team-scoped MCP tool catalogs",
    description=(
        "List all MCP tool catalogs scoped to the specified team.\n\n"
        "RBAC: Requires a team manager (owner/admin/lead) on this team or a global admin.\n\n"
        "Scope: Results include only catalogs with `team_id` matching the path."
    ),
)
async def list_team_tool_catalogs(
    team_id: int,
    principal: AuthPrincipal = Depends(get_auth_principal),
    db=Depends(get_db_transaction),
) -> list[ToolCatalogResponse]:
    await _require_team_manager(principal, team_id)
    try:
        rows = await _list_scoped_catalogs(db, team_id=team_id)
        return [ToolCatalogResponse(**r) for r in rows]
    except _CATALOG_NONCRITICAL_EXCEPTIONS as e:
        logger.error("Failed to list team tool catalogs")
        raise HTTPException(status_code=500, detail="Failed to list team tool catalogs") from e


@router.post(
    "/teams/{team_id}/mcp/tool_catalogs",
    response_model=ToolCatalogResponse,
    status_code=201,
    summary="Create new team-scoped MCP tool catalog",
    description=(
        "Create an MCP tool catalog owned by the specified team.\n\n"
        "RBAC: Requires a team manager (owner/admin/lead) on this team or a global admin.\n\n"
        "Scope: Catalog is created with `team_id` set. Name must be unique within the team scope."
    ),
)
async def create_team_tool_catalog(
    team_id: int,
    payload: ToolCatalogCreateRequest,
    principal: AuthPrincipal = Depends(get_auth_principal),
    db=Depends(get_db_transaction),
) -> ToolCatalogResponse:
    await _require_team_manager(principal, team_id)
    name = payload.name.strip()
    desc = payload.description
    is_active = bool(payload.is_active if payload.is_active is not None else True)
    try:
        row = await admin_tool_catalog_service.create_tool_catalog(
            db,
            name=name,
            description=desc,
            org_id=None,
            team_id=team_id,
            is_active=is_active,
        )
        await emit_mcp_hub_audit(
            action="tool_catalog.created",
            actor_id=principal.user_id,
            resource_type="mcp_tool_catalog",
            resource_id=str(row.get("id")),
            metadata={"owner_scope_type": "team", "owner_scope_id": team_id, "name": row.get("name")},
        )
        return ToolCatalogResponse(**row)
    except ToolCatalogConflictError as exc:
        raise HTTPException(status_code=409, detail="Catalog already exists") from exc
    except HTTPException:
        raise
    except _CATALOG_NONCRITICAL_EXCEPTIONS as e:
        logger.error("Failed to create team tool catalog")
        raise HTTPException(status_code=500, detail="Failed to create tool catalog") from e


@router.post(
    "/orgs/{org_id}/mcp/tool_catalogs/{catalog_id}/entries",
    response_model=ToolCatalogEntryResponse,
    status_code=201,
    summary="Add tool entry to an org-scoped catalog",
    description=(
        "Add a tool entry to an org-scoped catalog. Idempotent per (catalog_id, tool_name).\n\n"
        "RBAC: Requires an org manager (owner/admin/lead) in this organization or a global admin.\n\n"
        "Scope: Fails with 404 if the catalog is not owned by the specified org."
    ),
)
async def add_org_catalog_entry(
    org_id: int,
    catalog_id: int,
    payload: ToolCatalogEntryCreateRequest,
    principal: AuthPrincipal = Depends(get_auth_principal),
    db=Depends(get_db_transaction),
) -> ToolCatalogEntryResponse:
    await _require_org_manager(principal, org_id)
    # Confirm catalog scope
    owner = await _get_scoped_catalog(db, catalog_id=catalog_id, org_id=org_id)
    if not owner:
        raise HTTPException(status_code=404, detail="Catalog not found in org")
    # Upsert entry
    tool = payload.tool_name.strip()
    module_id = payload.module_id.strip() if payload.module_id else None
    try:
        row = await admin_tool_catalog_service.add_tool_catalog_entry(
            db,
            catalog_id,
            tool,
            module_id,
        )
        await emit_mcp_hub_audit(
            action="tool_catalog_entry.created",
            actor_id=principal.user_id,
            resource_type="mcp_tool_catalog_entry",
            resource_id=f"{catalog_id}:{tool}",
            metadata={
                "owner_scope_type": "org",
                "owner_scope_id": org_id,
                "catalog_id": catalog_id,
                "tool_name": tool,
                "module_id": module_id,
            },
        )
        return ToolCatalogEntryResponse(**row)
    except _CATALOG_NONCRITICAL_EXCEPTIONS as e:
        logger.error("Failed to add org tool catalog entry")
        raise HTTPException(status_code=500, detail="Failed to add tool catalog entry") from e


@router.post(
    "/teams/{team_id}/mcp/tool_catalogs/{catalog_id}/entries",
    response_model=ToolCatalogEntryResponse,
    status_code=201,
    summary="Add tool entry to a team-scoped catalog",
    description=(
        "Add a tool entry to a team-scoped catalog. Idempotent per (catalog_id, tool_name).\n\n"
        "RBAC: Requires a team manager (owner/admin/lead) on this team or a global admin.\n\n"
        "Scope: Fails with 404 if the catalog is not owned by the specified team."
    ),
)
async def add_team_catalog_entry(
    team_id: int,
    catalog_id: int,
    payload: ToolCatalogEntryCreateRequest,
    principal: AuthPrincipal = Depends(get_auth_principal),
    db=Depends(get_db_transaction),
) -> ToolCatalogEntryResponse:
    await _require_team_manager(principal, team_id)
    owner = await _get_scoped_catalog(db, catalog_id=catalog_id, team_id=team_id)
    if not owner:
        raise HTTPException(status_code=404, detail="Catalog not found in team")
    tool = payload.tool_name.strip()
    module_id = payload.module_id.strip() if payload.module_id else None
    try:
        row = await admin_tool_catalog_service.add_tool_catalog_entry(
            db,
            catalog_id,
            tool,
            module_id,
        )
        await emit_mcp_hub_audit(
            action="tool_catalog_entry.created",
            actor_id=principal.user_id,
            resource_type="mcp_tool_catalog_entry",
            resource_id=f"{catalog_id}:{tool}",
            metadata={
                "owner_scope_type": "team",
                "owner_scope_id": team_id,
                "catalog_id": catalog_id,
                "tool_name": tool,
                "module_id": module_id,
            },
        )
        return ToolCatalogEntryResponse(**row)
    except _CATALOG_NONCRITICAL_EXCEPTIONS as e:
        logger.error("Failed to add team tool catalog entry")
        raise HTTPException(status_code=500, detail="Failed to add tool catalog entry") from e


@router.delete(
    "/orgs/{org_id}/mcp/tool_catalogs/{catalog_id}",
    summary="Delete an org-scoped catalog (cascade entries)",
    description=(
        "Delete an org-owned catalog. Entries are removed via ON DELETE CASCADE.\n\n"
        "RBAC: Requires an org manager (owner/admin/lead) in this organization or a global admin.\n\n"
        "Scope: Returns 404 if the catalog does not belong to the specified org."
    ),
)
async def delete_org_tool_catalog(
    org_id: int, catalog_id: int, principal: AuthPrincipal = Depends(get_auth_principal), db=Depends(get_db_transaction)
) -> dict:
    """Delete an org-scoped tool catalog (entries cascade).

    Requires the requester to be an org manager (owner/admin/lead) or global admin.
    """
    await _require_org_manager(principal, org_id)
    try:
        owner = await _get_scoped_catalog(db, catalog_id=catalog_id, org_id=org_id)
        if not owner:
            raise HTTPException(status_code=404, detail="Catalog not found in org")
        await admin_tool_catalog_service.delete_tool_catalog(db, catalog_id)
        await emit_mcp_hub_audit(
            action="tool_catalog.deleted",
            actor_id=principal.user_id,
            resource_type="mcp_tool_catalog",
            resource_id=str(catalog_id),
            metadata={"owner_scope_type": "org", "owner_scope_id": org_id, "name": owner.get("name")},
        )
        return {"message": "Catalog deleted", "id": catalog_id, "scope": {"org_id": org_id}}
    except HTTPException:
        raise
    except _CATALOG_NONCRITICAL_EXCEPTIONS as e:
        logger.error("Failed to delete org tool catalog")
        raise HTTPException(status_code=500, detail="Failed to delete tool catalog") from e


@router.delete(
    "/orgs/{org_id}/mcp/tool_catalogs/{catalog_id}/entries/{tool_name}",
    summary="Remove tool entry from an org-scoped catalog",
    description=(
        "Remove a tool entry from an org-owned catalog.\n\n"
        "RBAC: Requires an org manager (owner/admin/lead) in this organization or a global admin.\n\n"
        "Scope: Returns 404 if the catalog does not belong to the specified org."
    ),
)
async def delete_org_catalog_entry(
    org_id: int,
    catalog_id: int,
    tool_name: str,
    principal: AuthPrincipal = Depends(get_auth_principal),
    db=Depends(get_db_transaction),
) -> dict:
    """Remove a tool entry from an org-scoped catalog.

    Requires the requester to be an org manager (owner/admin/lead) or global admin.
    """
    await _require_org_manager(principal, org_id)
    try:
        owner = await _get_scoped_catalog(db, catalog_id=catalog_id, org_id=org_id)
        if not owner:
            raise HTTPException(status_code=404, detail="Catalog not found in org")
        await admin_tool_catalog_service.delete_tool_catalog_entry(
            db,
            catalog_id,
            tool_name,
        )
        await emit_mcp_hub_audit(
            action="tool_catalog_entry.deleted",
            actor_id=principal.user_id,
            resource_type="mcp_tool_catalog_entry",
            resource_id=f"{catalog_id}:{tool_name}",
            metadata={
                "owner_scope_type": "org",
                "owner_scope_id": org_id,
                "catalog_id": catalog_id,
                "tool_name": tool_name,
            },
        )
        return {
            "message": "Entry deleted",
            "catalog_id": catalog_id,
            "tool_name": tool_name,
            "scope": {"org_id": org_id},
        }
    except HTTPException:
        raise
    except _CATALOG_NONCRITICAL_EXCEPTIONS as e:
        logger.error("Failed to delete org tool catalog entry")
        raise HTTPException(status_code=500, detail="Failed to delete tool catalog entry") from e


@router.delete(
    "/teams/{team_id}/mcp/tool_catalogs/{catalog_id}",
    summary="Delete a team-scoped catalog (cascade entries)",
    description=(
        "Delete a team-owned catalog. Entries are removed via ON DELETE CASCADE.\n\n"
        "RBAC: Requires a team manager (owner/admin/lead) on this team or a global admin.\n\n"
        "Scope: Returns 404 if the catalog does not belong to the specified team."
    ),
)
async def delete_team_tool_catalog(
    team_id: int,
    catalog_id: int,
    principal: AuthPrincipal = Depends(get_auth_principal),
    db=Depends(get_db_transaction),
) -> dict:
    """Delete a team-scoped tool catalog (entries cascade).

    Requires the requester to be a team manager (owner/admin/lead) or global admin.
    """
    await _require_team_manager(principal, team_id)
    try:
        owner = await _get_scoped_catalog(db, catalog_id=catalog_id, team_id=team_id)
        if not owner:
            raise HTTPException(status_code=404, detail="Catalog not found in team")
        await admin_tool_catalog_service.delete_tool_catalog(db, catalog_id)
        await emit_mcp_hub_audit(
            action="tool_catalog.deleted",
            actor_id=principal.user_id,
            resource_type="mcp_tool_catalog",
            resource_id=str(catalog_id),
            metadata={"owner_scope_type": "team", "owner_scope_id": team_id, "name": owner.get("name")},
        )
        return {"message": "Catalog deleted", "id": catalog_id, "scope": {"team_id": team_id}}
    except HTTPException:
        raise
    except _CATALOG_NONCRITICAL_EXCEPTIONS as e:
        logger.error("Failed to delete team tool catalog")
        raise HTTPException(status_code=500, detail="Failed to delete tool catalog") from e


@router.delete(
    "/teams/{team_id}/mcp/tool_catalogs/{catalog_id}/entries/{tool_name}",
    summary="Remove tool entry from a team-scoped catalog",
    description=(
        "Remove a tool entry from a team-owned catalog.\n\n"
        "RBAC: Requires a team manager (owner/admin/lead) on this team or a global admin.\n\n"
        "Scope: Returns 404 if the catalog does not belong to the specified team."
    ),
)
async def delete_team_catalog_entry(
    team_id: int,
    catalog_id: int,
    tool_name: str,
    principal: AuthPrincipal = Depends(get_auth_principal),
    db=Depends(get_db_transaction),
) -> dict:
    """Remove a tool entry from a team-scoped catalog.

    Requires the requester to be a team manager (owner/admin/lead) or global admin.
    """
    await _require_team_manager(principal, team_id)
    try:
        owner = await _get_scoped_catalog(db, catalog_id=catalog_id, team_id=team_id)
        if not owner:
            raise HTTPException(status_code=404, detail="Catalog not found in team")
        await admin_tool_catalog_service.delete_tool_catalog_entry(
            db,
            catalog_id,
            tool_name,
        )
        await emit_mcp_hub_audit(
            action="tool_catalog_entry.deleted",
            actor_id=principal.user_id,
            resource_type="mcp_tool_catalog_entry",
            resource_id=f"{catalog_id}:{tool_name}",
            metadata={
                "owner_scope_type": "team",
                "owner_scope_id": team_id,
                "catalog_id": catalog_id,
                "tool_name": tool_name,
            },
        )
        return {
            "message": "Entry deleted",
            "catalog_id": catalog_id,
            "tool_name": tool_name,
            "scope": {"team_id": team_id},
        }
    except HTTPException:
        raise
    except _CATALOG_NONCRITICAL_EXCEPTIONS as e:
        logger.error("Failed to delete team tool catalog entry")
        raise HTTPException(status_code=500, detail="Failed to delete tool catalog entry") from e
