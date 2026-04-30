from __future__ import annotations

import os
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from loguru import logger
from pydantic import BaseModel, Field
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal, get_request_user, RequirePermission, TokenScopeGuard, User

from tldw_Server_API.app.core.AuthNZ.permissions import WORKFLOWS_ADMIN
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.Scheduler import get_global_scheduler
from tldw_Server_API.app.services.workflows_scheduler import (
    build_schedule_payload,
    get_workflows_scheduler,
    resolve_schedule_submission_target,
)

router = APIRouter(prefix="/api/v1/scheduler/workflows", tags=["scheduler", "workflows"])

_ADMIN_RESCAN_SCOPE_DEP = TokenScopeGuard(
    "workflows",
    require_if_present=True,
    endpoint_id="scheduler.workflows.admin_rescan",
)
_ADMIN_RESCAN_PERMISSIONS_DEP = RequirePermission(WORKFLOWS_ADMIN)


class ScheduleCreateRequest(BaseModel):
    workflow_id: int | None = Field(None, description="Saved workflow ID; optional if definition snapshot is used")
    name: str | None = None
    cron: str = Field(..., description="Cron expression, e.g., '*/15 * * * *'")
    timezone: str | None = Field(
        None,
        description=(
            "IANA timezone name (e.g., 'UTC', 'America/New_York'). "
            "See tz database list: https://en.wikipedia.org/wiki/List_of_tz_database_time_zones"
        ),
    )
    inputs: dict[str, Any] = Field(default_factory=dict)
    run_mode: str = Field("async", pattern="^(async|sync)$")
    validation_mode: str = Field("block", pattern="^(block|non-block)$")
    enabled: bool = True
    concurrency_mode: str = Field("skip", pattern="^(skip|queue)$", description="skip: drop overlaps; queue: allow overlaps")
    misfire_grace_sec: int = Field(300, ge=0, le=86400)
    coalesce: bool = Field(True, description="Coalesce misfires into single run")
    require_online: bool = Field(False, description="If true, only run when the user has an active session")


class ScheduleUpdateRequest(BaseModel):
    name: str | None = None
    cron: str | None = None
    timezone: str | None = Field(
        None,
        description=(
            "IANA timezone name (e.g., 'UTC', 'America/New_York'). "
            "See tz database list: https://en.wikipedia.org/wiki/List_of_tz_database_time_zones"
        ),
    )
    inputs: dict[str, Any] | None = None
    run_mode: str | None = Field(None, pattern="^(async|sync)$")
    validation_mode: str | None = Field(None, pattern="^(block|non-block)$")
    enabled: bool | None = None
    concurrency_mode: str | None = Field(None, pattern="^(skip|queue)$")
    misfire_grace_sec: int | None = Field(None, ge=0, le=86400)
    coalesce: bool | None = None
    require_online: bool | None = Field(None, description="Toggle presence gating for this schedule")


class ScheduleResponse(BaseModel):
    id: str
    workflow_id: int | None
    name: str | None
    cron: str
    timezone: str | None
    inputs: dict[str, Any]
    run_mode: str
    validation_mode: str
    enabled: bool
    tenant_id: str
    user_id: str
    concurrency_mode: str
    misfire_grace_sec: int
    coalesce: bool
    require_online: bool
    last_run_at: str | None
    next_run_at: str | None
    last_status: str | None


def _normalize_claim_values(raw: Any) -> list[str]:
    values = raw if isinstance(raw, (list, tuple, set)) else ([raw] if raw is not None else [])
    out: list[str] = []
    for value in values:
        text = str(value).strip().lower()
        if text:
            out.append(text)
    return out


def _is_scheduler_admin_user(current_user: User) -> bool:
    """
    Resolve scheduler admin authorization from explicit role/permission claims.

    Legacy profile booleans/columns like ``is_admin`` are intentionally not
    trusted for cross-user scheduler authorization paths.
    """
    try:
        if "admin" in _normalize_claim_values(getattr(current_user, "roles", [])):
            return True
        permission_values = _normalize_claim_values(getattr(current_user, "permissions", []))
        if WORKFLOWS_ADMIN.lower() in permission_values:
            return True
        if "*" in permission_values:
            return True
        if "system.configure" in permission_values:
            return True
    except Exception:
        return False
    return False


@router.post(
    "",
    response_model=dict[str, str],
    status_code=201,
    dependencies=[Depends(TokenScopeGuard("workflows", require_if_present=True, endpoint_id="scheduler.workflows.create"))],
)
async def create_schedule(
    body: ScheduleCreateRequest,
    current_user: User = Depends(get_request_user),
):
    _validate_cron_or_422(body.cron, body.timezone)
    svc = get_workflows_scheduler()
    tenant_id = str(getattr(current_user, "tenant_id", "default"))
    sid = svc.create(
        tenant_id=tenant_id,
        user_id=str(current_user.id),
        workflow_id=body.workflow_id,
        name=body.name,
        cron=body.cron,
        timezone=body.timezone,
        inputs=body.inputs,
        run_mode=body.run_mode,
        validation_mode=body.validation_mode,
        enabled=body.enabled,
        concurrency_mode=body.concurrency_mode,
        misfire_grace_sec=body.misfire_grace_sec,
        coalesce=body.coalesce,
        require_online=body.require_online,
    )
    return {"id": sid}


@router.post(
    "/admin/rescan",
    response_model=dict[str, Any],
    status_code=200,
    dependencies=[
        Depends(_ADMIN_RESCAN_SCOPE_DEP),
        Depends(_ADMIN_RESCAN_PERMISSIONS_DEP),
    ],
)
async def admin_rescan(
    _principal: AuthPrincipal = Depends(get_auth_principal),  # noqa: B008
):
    """Force a one-shot rescan of all users' schedules.

    Requires the ``workflows.admin`` permission (via
    :data:`tldw_Server_API.app.core.AuthNZ.permissions.WORKFLOWS_ADMIN`) or an
    admin principal. Returns the number of registered APScheduler jobs after
    rescan.
    """
    svc = get_workflows_scheduler()
    try:
        await svc._rescan_once()  # type: ignore[attr-defined]
    except Exception as e:
        logger.warning("Admin rescan failed")
        raise HTTPException(status_code=500, detail="Rescan failed") from e
    jobs = 0
    try:
        jobs = len(svc._aps.get_jobs()) if getattr(svc, "_aps", None) else 0  # type: ignore[attr-defined]
    except Exception:
        logger.debug("Failed to collect APScheduler job count after admin rescan")
    return {"ok": True, "jobs": jobs}


@router.get(
    "",
    response_model=list[ScheduleResponse],
    dependencies=[Depends(TokenScopeGuard("workflows", require_if_present=True, endpoint_id="scheduler.workflows.list"))],
)
async def list_schedules(
    owner: str | None = Query(None, description="Admin-only: filter by owner user_id"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_request_user),
):
    svc = get_workflows_scheduler()
    tenant_id = str(getattr(current_user, "tenant_id", "default"))
    is_admin = _is_scheduler_admin_user(current_user)
    user_filter: str | None = None
    if owner:
        if not is_admin:
            raise HTTPException(status_code=403, detail="Admin-only owner filter")
        user_filter = owner
    else:
        user_filter = str(current_user.id)
    rows = svc.list(tenant_id=tenant_id, user_id=user_filter, limit=limit, offset=offset)
    out: list[ScheduleResponse] = []
    import json
    for r in rows:
        try:
            inputs = json.loads(r.inputs_json or "{}")
        except Exception:
            inputs = {}
        out.append(
            ScheduleResponse(
                id=r.id,
                workflow_id=r.workflow_id,
                name=r.name,
                cron=r.cron,
                timezone=r.timezone,
                inputs=inputs,
                run_mode=r.run_mode or "async",
                validation_mode=r.validation_mode or "block",
                enabled=bool(r.enabled),
                tenant_id=r.tenant_id,
                user_id=r.user_id,
                concurrency_mode=r.concurrency_mode,
                misfire_grace_sec=r.misfire_grace_sec,
                coalesce=bool(r.coalesce),
                require_online=bool(getattr(r, 'require_online', False)),
                last_run_at=r.last_run_at,
                next_run_at=r.next_run_at,
                last_status=r.last_status,
            )
        )
    return out


@router.get(
    "/{schedule_id}",
    response_model=ScheduleResponse,
    dependencies=[Depends(TokenScopeGuard("workflows", require_if_present=True, endpoint_id="scheduler.workflows.get"))],
)
async def get_schedule(
    schedule_id: str,
    current_user: User = Depends(get_request_user),
):
    svc = get_workflows_scheduler()
    s = svc.get(schedule_id)
    if not s:
        raise HTTPException(status_code=404, detail="Not found")
    is_admin = _is_scheduler_admin_user(current_user)
    if str(current_user.id) != s.user_id and not is_admin:
        raise HTTPException(status_code=403, detail="Forbidden")
    import json
    # Best-effort: compute next_run_at if missing (e.g., freshly created)
    if not s.next_run_at:
        try:
            from apscheduler.triggers.cron import CronTrigger
            tz = s.timezone or os.getenv("WORKFLOWS_SCHEDULER_TZ", "UTC")
            trigger = CronTrigger.from_crontab(s.cron, timezone=tz)
            from datetime import datetime
            now = datetime.now(trigger.timezone)
            nxt = trigger.get_next_fire_time(None, now)
            if nxt is not None:
                try:
                    svc._get_db(int(s.user_id)).set_history(s.id, next_run_at=nxt.isoformat())  # type: ignore[attr-defined]
                except Exception:
                    logger.debug("Failed to persist computed next_run_at")
                # Refresh s to reflect persisted value
                s = svc.get(schedule_id) or s
        except Exception:
            logger.debug("Failed to compute next_run_at from crontab")
    try:
        inputs = json.loads(s.inputs_json or "{}")
    except Exception:
        inputs = {}
    return ScheduleResponse(
        id=s.id,
        workflow_id=s.workflow_id,
        name=s.name,
        cron=s.cron,
        timezone=s.timezone,
        inputs=inputs,
        run_mode=s.run_mode or "async",
        validation_mode=s.validation_mode or "block",
        enabled=bool(s.enabled),
        tenant_id=s.tenant_id,
        user_id=s.user_id,
        concurrency_mode=s.concurrency_mode,
        misfire_grace_sec=s.misfire_grace_sec,
        coalesce=bool(s.coalesce),
        require_online=bool(getattr(s, 'require_online', False)),
        last_run_at=s.last_run_at,
        next_run_at=s.next_run_at,
        last_status=s.last_status,
    )


@router.patch(
    "/{schedule_id}",
    response_model=dict[str, bool],
    dependencies=[Depends(TokenScopeGuard("workflows", require_if_present=True, endpoint_id="scheduler.workflows.update"))],
)
async def update_schedule(
    schedule_id: str,
    body: ScheduleUpdateRequest,
    current_user: User = Depends(get_request_user),
):
    svc = get_workflows_scheduler()
    s = svc.get(schedule_id)
    if not s:
        raise HTTPException(status_code=404, detail="Not found")
    is_admin = _is_scheduler_admin_user(current_user)
    if str(current_user.id) != s.user_id and not is_admin:
        raise HTTPException(status_code=403, detail="Forbidden")
    update: dict[str, Any] = {}
    if body.name is not None:
        update["name"] = body.name
    if body.cron is not None:
        _validate_cron_or_422(body.cron, body.timezone or s.timezone)
        update["cron"] = body.cron
    if body.timezone is not None:
        update["timezone"] = body.timezone
    if body.inputs is not None:
        update["inputs"] = body.inputs
    if body.run_mode is not None:
        update["run_mode"] = body.run_mode
    if body.validation_mode is not None:
        update["validation_mode"] = body.validation_mode
    if body.enabled is not None:
        update["enabled"] = body.enabled
    if body.concurrency_mode is not None:
        update["concurrency_mode"] = body.concurrency_mode
    if body.misfire_grace_sec is not None:
        update["misfire_grace_sec"] = int(body.misfire_grace_sec)
    if body.coalesce is not None:
        update["coalesce"] = bool(body.coalesce)
    if body.require_online is not None:
        update["require_online"] = bool(body.require_online)
    ok = svc.update(schedule_id, update)
    return {"ok": bool(ok)}


@router.delete(
    "/{schedule_id}",
    response_model=dict[str, bool],
    dependencies=[Depends(TokenScopeGuard("workflows", require_if_present=True, endpoint_id="scheduler.workflows.delete"))],
)
async def delete_schedule(
    schedule_id: str,
    current_user: User = Depends(get_request_user),
):
    svc = get_workflows_scheduler()
    s = svc.get(schedule_id)
    if not s:
        return {"ok": False}
    is_admin = _is_scheduler_admin_user(current_user)
    if str(current_user.id) != s.user_id and not is_admin:
        raise HTTPException(status_code=403, detail="Forbidden")
    ok = svc.delete(schedule_id)
    return {"ok": bool(ok)}


@router.post(
    "/{schedule_id}/run-now",
    response_model=dict[str, str],
    dependencies=[Depends(TokenScopeGuard("workflows", require_if_present=True, require_schedule_match=True, schedule_path_param="schedule_id", allow_admin_bypass=True, endpoint_id="scheduler.workflows.run_now", count_as="run"))],
)
async def run_now(
    schedule_id: str,
    current_user: User = Depends(get_request_user),
):
    svc = get_workflows_scheduler()
    s = svc.get(schedule_id)
    if not s:
        raise HTTPException(status_code=404, detail="Not found")
    is_admin = _is_scheduler_admin_user(current_user)
    if str(current_user.id) != s.user_id and not is_admin:
        raise HTTPException(status_code=403, detail="Forbidden")
    # Submit immediate job to core Scheduler
    payload = build_schedule_payload(s)
    handler_name, queue_name = resolve_schedule_submission_target(payload)
    # Use the global scheduler instance to avoid duplicate worker pools
    core = await get_global_scheduler()
    task_id = await core.submit(handler_name, payload=payload, queue_name=queue_name, metadata={"user_id": s.user_id})
    return {"task_id": task_id}


class DryRunRequest(BaseModel):
    cron: str
    timezone: str | None = Field(
        None,
        description=(
            "IANA timezone name (e.g., 'UTC', 'America/New_York'). "
            "See tz database list: https://en.wikipedia.org/wiki/List_of_tz_database_time_zones"
        ),
    )
    inputs: dict[str, Any] = Field(default_factory=dict)


@router.post(
    "/dry-run",
    response_model=dict[str, Any],
    dependencies=[Depends(TokenScopeGuard("workflows", require_if_present=True, endpoint_id="scheduler.workflows.dry_run"))],
)
async def dry_run_schedule(body: DryRunRequest, current_user: User = Depends(get_request_user)):
    """Validate cron/timezone and return next run time and echo inputs.

    Notes:
    - Timezone must be an IANA tz name (e.g., 'UTC', 'America/New_York').
      See: https://en.wikipedia.org/wiki/List_of_tz_database_time_zones
    - Does not persist a schedule or submit a run.
    """
    _validate_cron_or_422(body.cron, body.timezone)
    from apscheduler.triggers.cron import CronTrigger
    tz = body.timezone or os.getenv("WORKFLOWS_SCHEDULER_TZ", "UTC")
    trigger = CronTrigger.from_crontab(body.cron, timezone=tz)
    from datetime import datetime
    now = datetime.now(trigger.timezone)
    nxt = trigger.get_next_fire_time(None, now)
    return {
        "valid": True,
        "next_run_at": nxt.isoformat() if nxt else None,
        "inputs_preview": body.inputs,
    }
def _validate_cron_or_422(cron: str, timezone: str | None) -> None:
    try:
        from apscheduler.triggers.cron import CronTrigger
        CronTrigger.from_crontab(cron, timezone=timezone or "UTC")
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Invalid cron or timezone. Timezone must be an IANA name. Details: {e}"
            ),
        ) from e
