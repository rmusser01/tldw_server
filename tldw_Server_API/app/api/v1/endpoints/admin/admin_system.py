from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, Query, Response
from fastapi.responses import PlainTextResponse

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
    get_auth_principal,
    get_db_transaction,
)
from tldw_Server_API.app.api.v1.schemas.admin_schemas import (
    ActivitySummaryResponse,
    AdminPermissionDebugRequest,
    AdminPermissionDebugResponse,
    AdminTokenDecodeRequest,
    AdminTokenDecodeResponse,
    AuditLogResponse,
    ErrorBreakdownResponse,
    RateLimitSummaryResponse,
    SecurityAlertStatusResponse,
    SystemLogsResponse,
    SystemStatsResponse,
)
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.services import admin_system_service

router = APIRouter()


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


@router.get("/security/alert-status", response_model=SecurityAlertStatusResponse)
async def get_security_alert_status() -> SecurityAlertStatusResponse:
    return await admin_system_service.get_security_alert_status()


@router.post("/debug/resolve-permissions", response_model=AdminPermissionDebugResponse)
async def debug_resolve_permissions(
    payload: AdminPermissionDebugRequest,
    principal: AuthPrincipal = Depends(get_auth_principal),
    db=Depends(get_db_transaction),
) -> AdminPermissionDebugResponse:
    """Resolve effective permissions for a given user (debug tool)."""
    await _enforce_admin_user_scope(principal, payload.user_id, require_hierarchy=False)
    result = await admin_system_service.debug_resolve_permissions(payload.user_id, db)
    return AdminPermissionDebugResponse(**result)


@router.post("/debug/validate-token", response_model=AdminTokenDecodeResponse)
async def debug_decode_token(
    payload: AdminTokenDecodeRequest,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> AdminTokenDecodeResponse:
    """Decode a JWT or API token without verifying its signature."""
    result = await admin_system_service.debug_decode_token(payload.token.get_secret_value())
    return AdminTokenDecodeResponse(**result)


@router.get("/dependencies/health")
async def get_all_dependencies_health(
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> dict:
    """Probe all external service dependencies and return health status."""
    return await admin_system_service.get_all_dependencies_health()


@router.get("/dependencies/uptime-history")
async def get_dependencies_uptime_history(
    principal: AuthPrincipal = Depends(get_auth_principal),
    service: str | None = None,
    range_days: int = 30,
) -> dict:
    """Return uptime history for dependency health probes.

    If *service* is given, returns history for that service only.
    Otherwise returns history for all tracked services.
    """
    from tldw_Server_API.app.services.admin_uptime_history_service import (
        get_admin_uptime_history_service,
    )

    svc = get_admin_uptime_history_service()
    if service:
        history = await svc.get_uptime_history(service, range_days=range_days)
        return {"service": service, "range_days": range_days, "history": history}
    all_history = await svc.get_all_services_uptime(range_days=range_days)
    return {"range_days": range_days, "services": all_history}


@router.get("/security/key-age-stats")
async def get_key_age_stats(
    principal: AuthPrincipal = Depends(get_auth_principal),
    db=Depends(get_db_transaction),
) -> dict:
    """Get API key age distribution without per-user fan-out."""
    return await admin_system_service.get_key_age_stats(db)


@router.get("/stats", response_model=SystemStatsResponse)
async def get_system_stats(
    db=Depends(get_db_transaction),
) -> SystemStatsResponse:
    return await admin_system_service.get_system_stats(db)


@router.get("/activity", response_model=ActivitySummaryResponse)
async def get_dashboard_activity(
    days: int = Query(7, ge=1, le=30),
    granularity: Literal["hour", "day", "auto"] = Query("auto"),
    db=Depends(get_db_transaction),
) -> ActivitySummaryResponse:
    return await admin_system_service.get_dashboard_activity(days, db, granularity)


@router.get("/audit-log", response_model=AuditLogResponse)
async def get_audit_log(
    user_id: int | None = None,
    action: str | None = None,
    resource: str | None = Query(None, description="Filter by resource type or type:id"),
    start: str | None = Query(None, description="ISO date or datetime (start)"),
    end: str | None = Query(None, description="ISO date or datetime (end)"),
    days: int = Query(7, ge=1, le=90),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    org_id: int | None = Query(None, description="Restrict to a specific organization"),
    principal: AuthPrincipal = Depends(get_auth_principal),
    db=Depends(get_db_transaction),
) -> AuditLogResponse:
    return await admin_system_service.get_audit_log(
        user_id=user_id,
        action=action,
        resource=resource,
        start=start,
        end=end,
        days=days,
        limit=limit,
        offset=offset,
        org_id=org_id,
        principal=principal,
        db=db,
    )


@router.get(
    "/audit-log/export",
    responses={
        200: {
            "description": "Audit log export as JSON or CSV.",
            "content": {
                "application/json": {},
                "text/csv": {},
            },
        },
    },
)
async def export_audit_log(
    user_id: int | None = None,
    action: str | None = None,
    resource: str | None = Query(None, description="Filter by resource type or type:id"),
    start: str | None = Query(None, description="ISO date or datetime (start)"),
    end: str | None = Query(None, description="ISO date or datetime (end)"),
    days: int = Query(30, ge=1, le=365),
    limit: int = Query(10000, ge=1, le=50000),
    offset: int = Query(0, ge=0),
    org_id: int | None = Query(None, description="Restrict to a specific organization"),
    format: str = Query("csv", pattern="^(csv|json)$"),
    filename: str | None = Query(None, description="Optional filename for Content-Disposition"),
    principal: AuthPrincipal = Depends(get_auth_principal),
    db=Depends(get_db_transaction),
) -> Response:
    content, media_type, default_filename = await admin_system_service.export_audit_log(
        user_id=user_id,
        action=action,
        resource=resource,
        start=start,
        end=end,
        days=days,
        limit=limit,
        offset=offset,
        org_id=org_id,
        format=format,
        principal=principal,
        db=db,
    )
    if format == "json":
        resp = Response(content=content, media_type=media_type)
    else:
        resp = PlainTextResponse(content=content, media_type=media_type)
    if not filename:
        filename = default_filename
    if filename:
        safe = filename.replace("\n", " ").replace("\r", " ").replace("\"", "_")
        resp.headers["Content-Disposition"] = f"attachment; filename=\"{safe}\""
    return resp


@router.get("/system/logs", response_model=SystemLogsResponse)
async def list_system_logs(
    start: str | None = Query(None, description="ISO date or datetime (start)"),
    end: str | None = Query(None, description="ISO date or datetime (end)"),
    level: str | None = Query(None, description="Log level (INFO, ERROR, etc.)"),
    service: str | None = Query(None, description="Logger or module filter"),
    query: str | None = Query(None, description="Substring search"),
    org_id: int | None = Query(None, description="Restrict to a specific organization"),
    user_id: int | None = Query(None, description="Restrict to a specific user"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> SystemLogsResponse:
    return await admin_system_service.list_system_logs(
        start=start,
        end=end,
        level=level,
        service=service,
        query=query,
        org_id=org_id,
        user_id=user_id,
        limit=limit,
        offset=offset,
        principal=principal,
    )


@router.get("/errors/breakdown", response_model=ErrorBreakdownResponse)
async def get_error_breakdown(
    hours: int = Query(24, ge=1, le=168, description="Lookback period in hours"),
    principal: AuthPrincipal = Depends(get_auth_principal),
    db=Depends(get_db_transaction),
) -> ErrorBreakdownResponse:
    """Aggregate recent errors from audit log grouped by action and status code."""
    return await admin_system_service.get_error_breakdown(
        principal=principal,
        db=db,
        hours=hours,
    )


@router.get("/rate-limits/summary", response_model=RateLimitSummaryResponse)
async def get_rate_limit_summary(
    hours: int = Query(24, ge=1, le=168, description="Lookback period in hours"),
) -> RateLimitSummaryResponse:
    """Aggregate rate-limit throttle events and policy headroom."""
    return await admin_system_service.get_rate_limit_summary(hours=hours)
