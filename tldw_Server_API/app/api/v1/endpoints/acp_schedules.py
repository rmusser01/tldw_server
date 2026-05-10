"""ACP Schedules sub-module.

Provides CRUD endpoints for scheduled ACP agent runs (cron-style triggers).
Schedules are stored in the same ``workflow_schedules`` table but with an
``acp_config_json`` column that marks them as ACP schedules and carries the
ACP-specific configuration (prompt, cwd, agent_type, etc.).
"""
from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from apscheduler.triggers.cron import CronTrigger
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_request_user, User


router = APIRouter(prefix="/acp/schedules", tags=["acp-schedules"])


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ACPScheduleCreate(BaseModel):
    name: str = Field(..., description="Human-readable schedule name")
    cron: str = Field(..., description="Cron expression (5-field)")
    prompt: Any = Field(..., description="Prompt string or message list for the agent")
    cwd: str = Field(".", description="Working directory for the agent")
    agent_type: str | None = Field(None, description="Agent type identifier")
    model: str | None = Field(None, description="LLM model override")
    token_budget: int | None = Field(None, description="Token budget for the run")
    persona_id: str | None = Field(None, description="Persona context")
    workspace_id: str | None = Field(None, description="Workspace context")
    sandbox_enabled: bool = Field(False, description="Run in sandbox")
    enabled: bool = Field(True, description="Whether the schedule is active")
    timezone: str = Field("UTC", description="IANA timezone for cron evaluation")
    concurrency_mode: str = Field("skip", description="Overlapping run behavior: 'skip' or 'queue'")
    misfire_grace_sec: int = Field(300, description="Seconds a missed fire may still enqueue")
    coalesce: bool = Field(True, description="Coalesce missed runs into one execution")


class ACPScheduleUpdate(BaseModel):
    name: str | None = None
    cron: str | None = None
    prompt: Any | None = None
    cwd: str | None = None
    agent_type: str | None = None
    model: str | None = None
    token_budget: int | None = None
    persona_id: str | None = None
    workspace_id: str | None = None
    sandbox_enabled: bool | None = None
    enabled: bool | None = None
    timezone: str | None = None
    concurrency_mode: str | None = None
    misfire_grace_sec: int | None = None
    coalesce: bool | None = None


class ACPScheduleResponse(BaseModel):
    id: str
    name: str | None = None
    cron: str
    acp_config: dict[str, Any]
    enabled: bool
    timezone: str | None = None
    last_run_at: str | None = None
    next_run_at: str | None = None
    last_status: str | None = None
    concurrency_mode: str = "skip"
    misfire_grace_sec: int = 300
    coalesce: bool = True
    created_at: str | None = None


class ACPScheduleDeleteResponse(BaseModel):
    status: str
    id: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_scheduler():
    """Return the global WorkflowsScheduler instance."""
    from tldw_Server_API.app.services.workflows_scheduler import get_workflows_scheduler
    return get_workflows_scheduler()


def _validate_cron(cron: str, timezone: str = "UTC") -> None:
    """Validate a cron expression; raise HTTPException on failure."""
    try:
        CronTrigger.from_crontab(cron, timezone=timezone)
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid cron expression '{cron}': {exc}",
        )


def _validate_concurrency_mode(mode: str) -> None:
    """Validate ACP schedule overlapping-run behavior."""
    if mode not in {"skip", "queue"}:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid concurrency_mode '{mode}'; must be 'skip' or 'queue'",
        )


def _build_acp_config(body: ACPScheduleCreate | ACPScheduleUpdate) -> dict[str, Any]:
    """Extract ACP-specific fields from the request body into a config dict."""
    cfg: dict[str, Any] = {}
    if getattr(body, "prompt", None) is not None:
        cfg["prompt"] = body.prompt
    if getattr(body, "cwd", None) is not None:
        cfg["cwd"] = body.cwd
    if getattr(body, "agent_type", None) is not None:
        cfg["agent_type"] = body.agent_type
    if getattr(body, "model", None) is not None:
        cfg["model"] = body.model
    if getattr(body, "token_budget", None) is not None:
        cfg["token_budget"] = body.token_budget
    if getattr(body, "persona_id", None) is not None:
        cfg["persona_id"] = body.persona_id
    if getattr(body, "workspace_id", None) is not None:
        cfg["workspace_id"] = body.workspace_id
    if getattr(body, "sandbox_enabled", None) is not None:
        cfg["sandbox_enabled"] = body.sandbox_enabled
    return cfg


def _schedule_to_response(s) -> dict[str, Any]:
    """Convert a WorkflowSchedule dataclass to an ACPScheduleResponse-compatible dict."""
    acp_config: dict[str, Any] = {}
    if s.acp_config_json:
        try:
            acp_config = json.loads(s.acp_config_json) if isinstance(s.acp_config_json, str) else s.acp_config_json
        except Exception:
            acp_config = {}
    return {
        "id": s.id,
        "name": s.name,
        "cron": s.cron,
        "acp_config": acp_config,
        "enabled": s.enabled,
        "timezone": s.timezone,
        "last_run_at": s.last_run_at,
        "next_run_at": s.next_run_at,
        "last_status": s.last_status,
        "concurrency_mode": s.concurrency_mode,
        "misfire_grace_sec": s.misfire_grace_sec,
        "coalesce": s.coalesce,
        "created_at": s.created_at,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("", response_model=ACPScheduleResponse, status_code=201)
async def create_acp_schedule(
    body: ACPScheduleCreate,
    user: User = Depends(get_request_user),
) -> dict[str, Any]:
    """Create a recurring ACP agent schedule."""
    _validate_cron(body.cron, body.timezone)
    _validate_concurrency_mode(body.concurrency_mode)

    acp_config = _build_acp_config(body)
    acp_config_str = json.dumps(acp_config)

    svc = _get_scheduler()
    sid = svc.create(
        tenant_id="default",
        user_id=str(user.id),
        workflow_id=None,
        name=body.name,
        cron=body.cron,
        timezone=body.timezone,
        inputs={},
        run_mode="async",
        validation_mode="block",
        enabled=body.enabled,
        concurrency_mode=body.concurrency_mode,
        misfire_grace_sec=body.misfire_grace_sec,
        coalesce=body.coalesce,
        acp_config_json=acp_config_str,
    )

    s = svc.get(sid)
    if s is None:
        raise HTTPException(status_code=500, detail="Failed to create schedule")
    return _schedule_to_response(s)


@router.get("", response_model=list[ACPScheduleResponse])
async def list_acp_schedules(
    user: User = Depends(get_request_user),
    limit: int = 50,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """List ACP agent schedules for the current user.

    Only returns schedules that have ``acp_config_json`` set (filtering
    out plain workflow schedules).
    """
    svc = _get_scheduler()
    all_schedules = svc.list(
        tenant_id="default",
        user_id=str(user.id),
        limit=limit,
        offset=offset,
    )
    # Filter to ACP-only schedules
    return [
        _schedule_to_response(s)
        for s in all_schedules
        if s.acp_config_json
    ]


@router.put("/{schedule_id}", response_model=ACPScheduleResponse)
async def update_acp_schedule(
    schedule_id: str,
    body: ACPScheduleUpdate,
    user: User = Depends(get_request_user),
) -> dict[str, Any]:
    """Update an existing ACP schedule."""
    svc = _get_scheduler()
    existing = svc.get(schedule_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="Schedule not found")
    # Verify it belongs to the current user
    if str(existing.user_id) != str(user.id):
        raise HTTPException(status_code=403, detail="Not your schedule")
    # Verify this is an ACP schedule
    if not existing.acp_config_json:
        raise HTTPException(status_code=400, detail="Schedule is not an ACP schedule")

    update_dict: dict[str, Any] = {}

    # Handle cron update with validation
    new_cron = body.cron if body.cron is not None else existing.cron
    new_tz = body.timezone if body.timezone is not None else (existing.timezone or "UTC")
    if body.cron is not None:
        _validate_cron(body.cron, new_tz)
        update_dict["cron"] = body.cron
    if body.timezone is not None:
        update_dict["timezone"] = body.timezone
    if body.name is not None:
        update_dict["name"] = body.name
    if body.enabled is not None:
        update_dict["enabled"] = body.enabled
    if body.concurrency_mode is not None:
        _validate_concurrency_mode(body.concurrency_mode)
        update_dict["concurrency_mode"] = body.concurrency_mode
    if body.misfire_grace_sec is not None:
        update_dict["misfire_grace_sec"] = body.misfire_grace_sec
    if body.coalesce is not None:
        update_dict["coalesce"] = body.coalesce

    # Merge ACP config: start from existing, overlay provided fields
    try:
        current_config = json.loads(existing.acp_config_json) if isinstance(existing.acp_config_json, str) else (existing.acp_config_json or {})
    except Exception:
        current_config = {}

    new_fields = _build_acp_config(body)
    if new_fields:
        current_config.update(new_fields)
        update_dict["acp_config_json"] = json.dumps(current_config)

    if update_dict:
        svc.update(schedule_id, update_dict)

    updated = svc.get(schedule_id)
    if updated is None:
        raise HTTPException(status_code=500, detail="Schedule disappeared after update")
    return _schedule_to_response(updated)


@router.delete("/{schedule_id}", response_model=ACPScheduleDeleteResponse)
async def delete_acp_schedule(
    schedule_id: str,
    user: User = Depends(get_request_user),
) -> ACPScheduleDeleteResponse:
    """Delete an ACP schedule."""
    svc = _get_scheduler()
    existing = svc.get(schedule_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="Schedule not found")
    if str(existing.user_id) != str(user.id):
        raise HTTPException(status_code=403, detail="Not your schedule")
    if not existing.acp_config_json:
        raise HTTPException(status_code=400, detail="Schedule is not an ACP schedule")

    svc.delete(schedule_id)
    return ACPScheduleDeleteResponse(status="deleted", id=schedule_id)
