from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from loguru import logger
from pydantic import BaseModel, Field

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
    RequireRole,
    get_db_transaction,
)
from tldw_Server_API.app.api.v1.schemas.admin_schemas import (
    ToolCatalogCreateRequest,
    ToolCatalogEntryCreateRequest,
    ToolCatalogEntryResponse,
    ToolCatalogResponse,
)
from tldw_Server_API.app.core.exceptions import ToolCatalogConflictError
from tldw_Server_API.app.services import admin_tool_catalog_service

router = APIRouter()


@router.get(
    "/mcp/tool_catalogs",
    response_model=list[ToolCatalogResponse],
    summary="List MCP tool catalogs (admin)",
    description=(
        "List MCP tool catalogs across global/org/team scopes.\n\n"
        "RBAC: Admin-only.\n\n"
        "Filters: Optional `org_id` and/or `team_id` parameters restrict results to a given scope.\n"
        "Without filters, returns all catalogs."
    ),
    dependencies=[Depends(RequireRole("admin"))],
)
async def list_tool_catalogs(
    org_id: int | None = Query(None),
    team_id: int | None = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    db: Any = Depends(get_db_transaction),
) -> list[ToolCatalogResponse]:
    """List tool catalogs with optional org/team filtering."""
    try:
        rows = await admin_tool_catalog_service.list_tool_catalogs(
            db,
            org_id=org_id,
            team_id=team_id,
            limit=limit,
            offset=offset,
        )
        return [ToolCatalogResponse(**r) for r in rows]
    except Exception as exc:
        logger.error("Failed to list tool catalogs")
        raise HTTPException(status_code=500, detail="Failed to list tool catalogs") from exc


@router.post(
    "/mcp/tool_catalogs",
    response_model=ToolCatalogResponse,
    status_code=201,
    summary="Create MCP tool catalog (admin)",
    description=(
        "Create a new MCP tool catalog in the chosen scope.\n\n"
        "RBAC: Admin-only.\n\n"
        "Scope: Set `org_id` for org-owned, `team_id` for team-owned, or neither for global.\n"
        "Name must be unique per (name, org_id, team_id)."
    ),
    dependencies=[Depends(RequireRole("admin"))],
)
async def create_tool_catalog(
    payload: ToolCatalogCreateRequest,
    db: Any = Depends(get_db_transaction),
) -> ToolCatalogResponse:
    """Create a tool catalog."""
    try:
        name = payload.name.strip()
        desc = payload.description
        org_id = payload.org_id
        team_id = payload.team_id
        is_active = bool(payload.is_active if payload.is_active is not None else True)
        row = await admin_tool_catalog_service.create_tool_catalog(
            db,
            name=name,
            description=desc,
            org_id=org_id,
            team_id=team_id,
            is_active=is_active,
        )
        return ToolCatalogResponse(**row)
    except ToolCatalogConflictError as exc:
        logger.warning("Tool catalog already exists: {}", payload.name)
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Catalog already exists") from exc
    except ValueError as exc:
        logger.warning("Invalid tool catalog payload: {}", exc)
        raise HTTPException(status_code=400, detail="Invalid tool catalog") from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to create tool catalog")
        raise HTTPException(status_code=500, detail="Failed to create tool catalog") from exc


@router.delete(
    "/mcp/tool_catalogs/{catalog_id}",
    summary="Delete MCP tool catalog (admin)",
    description=(
        "Delete a catalog by id. Entries are removed via ON DELETE CASCADE.\n\n"
        "RBAC: Admin-only.\n\n"
        "Scope: Works for any catalog (global/org/team)."
    ),
    dependencies=[Depends(RequireRole("admin"))],
)
async def delete_tool_catalog(
    catalog_id: int,
    db: Any = Depends(get_db_transaction),
) -> dict:
    """Delete a tool catalog (entries cascade)."""
    try:
        catalog = await admin_tool_catalog_service.get_tool_catalog(db, catalog_id)
        if not catalog:
            raise HTTPException(status_code=404, detail="Tool catalog not found")
        await admin_tool_catalog_service.delete_tool_catalog(db, catalog_id)
        return {"message": "Catalog deleted", "id": catalog_id}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to delete tool catalog")
        raise HTTPException(status_code=500, detail="Failed to delete tool catalog") from exc


@router.get(
    "/mcp/tool_catalogs/{catalog_id}/entries",
    response_model=list[ToolCatalogEntryResponse],
    summary="List catalog entries (admin)",
    description=(
        "List tools included in the specified catalog.\n\n"
        "RBAC: Admin-only."
    ),
    dependencies=[Depends(RequireRole("admin"))],
)
async def list_tool_catalog_entries(
    catalog_id: int,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    db: Any = Depends(get_db_transaction),
) -> list[ToolCatalogEntryResponse]:
    """List entries in a tool catalog."""
    try:
        catalog = await admin_tool_catalog_service.get_tool_catalog(db, catalog_id)
        if not catalog:
            raise HTTPException(status_code=404, detail="Tool catalog not found")
        rows = await admin_tool_catalog_service.list_tool_catalog_entries(
            db,
            catalog_id,
            limit=limit,
            offset=offset,
        )
        return [ToolCatalogEntryResponse(**r) for r in rows]
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to list tool catalog entries")
        raise HTTPException(status_code=500, detail="Failed to list tool catalog entries") from exc


@router.post(
    "/mcp/tool_catalogs/{catalog_id}/entries",
    response_model=ToolCatalogEntryResponse,
    status_code=201,
    summary="Add tool to catalog (admin)",
    description=(
        "Add a tool entry to the catalog. Idempotent per (catalog_id, tool_name).\n\n"
        "RBAC: Admin-only."
    ),
    dependencies=[Depends(RequireRole("admin"))],
)
async def add_tool_catalog_entry(
    catalog_id: int,
    payload: ToolCatalogEntryCreateRequest,
    db: Any = Depends(get_db_transaction),
) -> ToolCatalogEntryResponse:
    """Add a tool entry to a catalog (idempotent)."""
    try:
        catalog = await admin_tool_catalog_service.get_tool_catalog(db, catalog_id)
        if not catalog:
            raise HTTPException(status_code=404, detail="Tool catalog not found")
        tool = payload.tool_name.strip()
        module_id = payload.module_id.strip() if payload.module_id else None
        row = await admin_tool_catalog_service.add_tool_catalog_entry(db, catalog_id, tool, module_id)
        return ToolCatalogEntryResponse(**row)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to add tool catalog entry")
        raise HTTPException(status_code=500, detail="Failed to add tool catalog entry") from exc


@router.delete(
    "/mcp/tool_catalogs/{catalog_id}/entries/{tool_name}",
    summary="Remove tool from catalog (admin)",
    description=(
        "Remove a tool entry from the catalog. Returns 200 whether or not the entry existed.\n\n" "RBAC: Admin-only."
    ),
    dependencies=[Depends(RequireRole("admin"))],
)
async def delete_tool_catalog_entry(
    catalog_id: int,
    tool_name: str,
    db: Any = Depends(get_db_transaction),
) -> dict:
    """Remove a tool from a catalog."""
    try:
        catalog = await admin_tool_catalog_service.get_tool_catalog(db, catalog_id)
        if not catalog:
            raise HTTPException(status_code=404, detail="Tool catalog not found")
        await admin_tool_catalog_service.delete_tool_catalog_entry(db, catalog_id, tool_name)
        return {"message": "Entry deleted", "catalog_id": catalog_id, "tool_name": tool_name}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to delete tool catalog entry")
        raise HTTPException(status_code=500, detail="Failed to delete tool catalog entry") from exc


class ModuleToolUsage(BaseModel):
    calls: int = 0
    avg_latency_ms: float = 0


class MCPToolUsageResponse(BaseModel):
    period_seconds: int
    modules: dict[str, ModuleToolUsage] = Field(default_factory=dict)
    tools: dict[str, ModuleToolUsage] = Field(default_factory=dict)


@router.get("/mcp/tool-usage", response_model=MCPToolUsageResponse)
async def get_mcp_tool_usage(
    period_seconds: int = Query(3600, ge=60, le=86400),
) -> MCPToolUsageResponse:
    """Aggregate MCP tool invocation counts per module from internal metrics."""
    try:
        from tldw_Server_API.app.core.MCP_unified.monitoring.metrics import (
            get_metrics_collector,
        )

        collector = get_metrics_collector()
        raw = collector.get_internal_metrics(period_seconds=period_seconds)

        module_usage: dict[str, ModuleToolUsage] = {}
        tool_usage: dict[str, ModuleToolUsage] = {}
        module_latency_totals: dict[str, float] = {}
        tool_latency_totals: dict[str, float] = {}

        def _accumulate(
            target: dict[str, ModuleToolUsage],
            latency_totals: dict[str, float],
            key: str,
            *,
            count: int,
            avg_seconds: float,
        ) -> None:
            bucket = target.setdefault(key, ModuleToolUsage())
            bucket.calls += count
            if count > 0:
                latency_totals[key] = latency_totals.get(key, 0.0) + (avg_seconds * 1000 * count)

        for key, data in raw.items():
            if not key.startswith("module_") or "_tools_call" not in key:
                continue
            groups = data.get("labels", [])
            for group in groups:
                labels = dict(group.get("labels", []))
                module = labels.get("module", "unknown")
                tool = labels.get("tool") or labels.get("tool_name")
                count = group.get("count", 0)
                avg = group.get("avg", 0)
                if not isinstance(module, str) or not module:
                    module = "unknown"
                _accumulate(module_usage, module_latency_totals, module, count=count, avg_seconds=avg)
                if isinstance(tool, str) and tool:
                    tool_key = f"{module}.{tool}"
                    _accumulate(tool_usage, tool_latency_totals, tool_key, count=count, avg_seconds=avg)

        for name, total_latency_ms in module_latency_totals.items():
            calls = module_usage[name].calls
            if calls > 0:
                module_usage[name].avg_latency_ms = round(total_latency_ms / calls, 1)
        for name, total_latency_ms in tool_latency_totals.items():
            calls = tool_usage[name].calls
            if calls > 0:
                tool_usage[name].avg_latency_ms = round(total_latency_ms / calls, 1)

        return MCPToolUsageResponse(
            period_seconds=period_seconds,
            modules=module_usage,
            tools=tool_usage,
        )
    except ImportError:
        logger.warning("MCP metrics module unavailable")
        return MCPToolUsageResponse(period_seconds=period_seconds, modules={}, tools={})
