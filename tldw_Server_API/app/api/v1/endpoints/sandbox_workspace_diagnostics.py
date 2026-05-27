from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Mapping
from functools import lru_cache
from typing import Any
from urllib.parse import quote

from fastapi import APIRouter, Depends, Path, Query
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import User, get_request_user
from tldw_Server_API.app.api.v1.endpoints.sandbox_service import sandbox_service
from tldw_Server_API.app.api.v1.schemas.sandbox_schemas import (
    SandboxRunStatusReasonDetails,
    SandboxWorkspaceDiagnosticState,
    SandboxWorkspaceDiagnosticsLinks,
    SandboxWorkspaceDiagnosticsResponse,
    SandboxWorkspaceDiagnosticsRunList,
    SandboxWorkspaceDiagnosticsRunSummary,
)
from tldw_Server_API.app.core.Sandbox.models import RunPhase
from tldw_Server_API.app.core.Sandbox.orchestrator import IdempotencyConflict, QueueFull
from tldw_Server_API.app.core.Sandbox.policy import SandboxPolicy
from tldw_Server_API.app.core.Sandbox.run_status_taxonomy import (
    normalize_run_status_reason,
    run_status_reason_details,
)
from tldw_Server_API.app.core.Sandbox.service import SandboxService
from tldw_Server_API.app.core.config import settings as app_settings
from tldw_Server_API.app.core.testing import is_truthy

router = APIRouter(prefix="/sandbox", tags=["sandbox"])

_service = sandbox_service

_SANDBOX_WORKSPACE_DIAGNOSTICS_NONCRITICAL_EXCEPTIONS = (
    asyncio.CancelledError,
    AttributeError,
    ConnectionError,
    FileNotFoundError,
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
    json.JSONDecodeError,
    IdempotencyConflict,
    QueueFull,
    SandboxPolicy.RuntimeUnavailable,
    SandboxPolicy.PolicyUnsupported,
    SandboxService.InvalidSpecVersion,
    SandboxService.InvalidFirecrackerConfig,
)


def _status_reason_code(
    *,
    phase: RunPhase | str | None,
    message: str | None,
    exit_code: int | str | None,
    resource_usage: Mapping[str, Any] | None,
) -> str:
    return normalize_run_status_reason(
        phase=phase,
        message=message,
        exit_code=exit_code,
        resource_usage=resource_usage if isinstance(resource_usage, dict) else None,
    )


@lru_cache(maxsize=32)
def _cached_status_reason_details(
    code: str,
) -> SandboxRunStatusReasonDetails:
    return SandboxRunStatusReasonDetails.model_validate(run_status_reason_details(code))


def _status_reason_details(
    code: str | None,
) -> SandboxRunStatusReasonDetails:
    return _cached_status_reason_details(str(code or "unknown").strip())


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except _SANDBOX_WORKSPACE_DIAGNOSTICS_NONCRITICAL_EXCEPTIONS:
        return default


def _sandbox_run_route_enabled() -> bool:
    try:
        from tldw_Server_API.app.core import config as app_config

        return bool(app_config.route_enabled("sandbox", default_stable=False))
    except _SANDBOX_WORKSPACE_DIAGNOSTICS_NONCRITICAL_EXCEPTIONS as exc:
        logger.debug("Sandbox workspace diagnostics could not read route policy: {}", exc)
        return False


def _sandbox_execution_enabled() -> bool:
    try:
        env_exec = os.getenv("SANDBOX_ENABLE_EXECUTION")
        if env_exec is not None:
            return is_truthy(env_exec)
        return bool(getattr(app_settings, "SANDBOX_ENABLE_EXECUTION", False))
    except _SANDBOX_WORKSPACE_DIAGNOSTICS_NONCRITICAL_EXCEPTIONS as exc:
        logger.debug("Sandbox workspace diagnostics could not read execution gate: {}", exc)
        return False


def _sandbox_workspace_runtime_state() -> tuple[SandboxWorkspaceDiagnosticState, SandboxWorkspaceDiagnosticState]:
    try:
        diagnostics = _service.runtime_diagnostics_summary()
    except _SANDBOX_WORKSPACE_DIAGNOSTICS_NONCRITICAL_EXCEPTIONS as exc:
        logger.debug("Sandbox workspace diagnostics could not read runtime summary: {}", exc)
        runtime = SandboxWorkspaceDiagnosticState(
            state="unknown",
            reason_code="sandbox_not_available",
            message="Sandbox diagnostics are unavailable right now.",
            management_surface="sandbox_settings",
        )
        admission = SandboxWorkspaceDiagnosticState(
            state="blocked",
            reason_code="sandbox_not_available",
            message="Sandboxed workspace actions are blocked until sandbox diagnostics recover.",
            management_surface="sandbox_settings",
        )
        return runtime, admission

    summary = diagnostics.get("summary") if isinstance(diagnostics, Mapping) else {}
    summary_map = summary if isinstance(summary, Mapping) else {}
    total = _coerce_int(summary_map.get("total"))
    ready = _coerce_int(summary_map.get("ready"))
    unavailable = _coerce_int(summary_map.get("unavailable"))

    if ready > 0:
        runtime = SandboxWorkspaceDiagnosticState(
            state="available",
            reason_code=None,
            message="A sandbox runtime is available for workspace actions.",
            management_surface="sandbox_settings",
        )
        if not _sandbox_run_route_enabled():
            admission = SandboxWorkspaceDiagnosticState(
                state="blocked",
                reason_code="sandbox_route_disabled",
                message=(
                    "Sandboxed workspace actions are blocked because the sandbox "
                    "API route is disabled by route policy."
                ),
                management_surface="sandbox_settings",
            )
            return runtime, admission

        if not _sandbox_execution_enabled():
            admission = SandboxWorkspaceDiagnosticState(
                state="blocked",
                reason_code="sandbox_execution_disabled",
                message=(
                    "Sandboxed workspace actions are blocked because sandbox "
                    "execution is disabled."
                ),
                management_surface="sandbox_settings",
            )
            return runtime, admission

        admission = SandboxWorkspaceDiagnosticState(
            state="available",
            reason_code=None,
            message="Sandboxed workspace actions may run.",
            management_surface="sandbox_settings",
        )
        return runtime, admission

    if total <= 0:
        runtime = SandboxWorkspaceDiagnosticState(
            state="not_configured",
            reason_code="sandbox_no_runtimes_discovered",
            message="No sandbox runtimes are available for workspace actions.",
            management_surface="sandbox_settings",
        )
        admission = SandboxWorkspaceDiagnosticState(
            state="blocked",
            reason_code="sandbox_not_configured",
            message="Enable a sandbox runtime before sandboxed workspace actions can run.",
            management_surface="sandbox_settings",
        )
        return runtime, admission

    if unavailable >= total:
        runtime = SandboxWorkspaceDiagnosticState(
            state="unavailable",
            reason_code="sandbox_runtime_unavailable",
            message="Sandbox runtime discovery failed or all runtimes are unavailable.",
            management_surface="sandbox_settings",
        )
        admission = SandboxWorkspaceDiagnosticState(
            state="blocked",
            reason_code="sandbox_runtime_unavailable",
            message="Sandboxed workspace actions are blocked until a runtime is healthy.",
            management_surface="sandbox_settings",
        )
        return runtime, admission

    runtime = SandboxWorkspaceDiagnosticState(
        state="unknown",
        reason_code="sandbox_not_available",
        message="Sandbox readiness could not be determined for this workspace.",
        management_surface="sandbox_settings",
    )
    admission = SandboxWorkspaceDiagnosticState(
        state="blocked",
        reason_code="sandbox_not_available",
        message="Sandboxed workspace actions are blocked until sandbox readiness is known.",
        management_surface="sandbox_settings",
    )
    return runtime, admission


def _workspace_diagnostics_run_summary(row: Mapping[str, Any]) -> SandboxWorkspaceDiagnosticsRunSummary:
    status_reason_code = _status_reason_code(
        phase=row.get("phase"),
        message=row.get("message"),
        exit_code=row.get("exit_code"),
        resource_usage=row.get("resource_usage"),
    )
    return SandboxWorkspaceDiagnosticsRunSummary(
        id=str(row.get("id")),
        runtime=row.get("runtime"),
        runtime_version=row.get("runtime_version"),
        base_image=row.get("base_image"),
        phase=row.get("phase"),
        status_reason_code=status_reason_code,
        status_reason_details=_status_reason_details(status_reason_code),
        exit_code=row.get("exit_code"),
        started_at=row.get("started_at"),
        finished_at=row.get("finished_at"),
        message=row.get("message"),
        session_id=row.get("session_id"),
        persona_id=row.get("persona_id"),
        workspace_id=row.get("workspace_id"),
        workspace_group_id=row.get("workspace_group_id"),
        scope_snapshot_id=row.get("scope_snapshot_id"),
    )


@router.get(
    "/workspaces/{workspace_id}/diagnostics",
    response_model=SandboxWorkspaceDiagnosticsResponse,
    summary="Get sandbox diagnostics scoped to a workspace",
)
async def get_workspace_sandbox_diagnostics(
    workspace_id: str = Path(..., description="Canonical workspace id"),
    source_label: str = Query("research_workspace", description="Canonical workspace source label"),
    limit: int = Query(10, ge=1, le=100),
    current_user: User = Depends(get_request_user),
) -> SandboxWorkspaceDiagnosticsResponse:
    del source_label
    runtime, admission = _sandbox_workspace_runtime_state()
    user_id = str(current_user.id)
    rows = _service._orch.list_runs(  # type: ignore[attr-defined]
        user_id=user_id,
        workspace_id=workspace_id,
        limit=limit,
        offset=0,
        sort_desc=True,
    )
    total = int(
        _service._orch.count_runs(  # type: ignore[attr-defined]
            user_id=user_id,
            workspace_id=workspace_id,
        )
    )
    items = [_workspace_diagnostics_run_summary(row) for row in rows]
    encoded_workspace_id = quote(str(workspace_id), safe="")
    return SandboxWorkspaceDiagnosticsResponse(
        workspace_id=workspace_id,
        source_label="research_workspace",
        runtime=runtime,
        admission=admission,
        runs=SandboxWorkspaceDiagnosticsRunList(
            total=total,
            limit=int(limit),
            has_more=len(items) < total,
            items=items,
        ),
        links=SandboxWorkspaceDiagnosticsLinks(
            runtime_config="/admin/runtime-config",
            admin_runs=f"/admin/monitoring?focus=sandbox&workspace_id={encoded_workspace_id}",
        ),
    )
