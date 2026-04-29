from __future__ import annotations

"""Admin monitoring control-plane endpoints for shared alert rules and overlay state."""

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.api.v1.schemas.admin_schemas import (
    AdminAlertAssignRequest,
    AdminAlertEscalateRequest,
    AdminAlertEventResponse,
    AdminAlertHistoryListResponse,
    AdminAlertRuleCreateRequest,
    AdminAlertRuleCreateResponse,
    AdminAlertRuleDeleteResponse,
    AdminAlertRuleListResponse,
    AdminAlertRuleResponse,
    AdminAlertSnoozeRequest,
    AdminAlertStateMutationResponse,
    AdminAlertStateResponse,
)
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.repos.admin_monitoring_repo import (
    AuthnzAdminMonitoringRepo,
)
from tldw_Server_API.app.core.AuthNZ.repos.users_repo import AuthnzUsersRepo
from tldw_Server_API.app.core.DB_Management.TopicMonitoring_DB import TopicMonitoringDB

router = APIRouter()

_RUNTIME_ALERT_ID_PREFIX = "alert:"


def _resolve_monitoring_alerts_db_path() -> str:
    """Resolve the monitoring alerts DB path the same way the monitoring service does.

    Anchors relative paths to the project root instead of relying on the
    process working directory.
    """
    raw = os.getenv("MONITORING_ALERTS_DB", "Databases/monitoring_alerts.db")
    db_p = Path(raw)
    if db_p.is_absolute():
        return str(db_p)
    try:
        from tldw_Server_API.app.core.Utils.Utils import get_project_root as _gpr

        return str((Path(_gpr()).resolve() / db_p).resolve())
    except Exception:
        # Fallback: walk parents of this file looking for repo root markers
        start = Path(__file__).resolve()
        for candidate in (start.parent, *start.parent.parents):
            if (candidate / "pyproject.toml").is_file() and (candidate / "tldw_Server_API").is_dir():
                return str((candidate / db_p).resolve())
            if (candidate / ".git").exists():
                return str((candidate / db_p).resolve())
        return raw


_MONITORING_NONCRITICAL_EXCEPTIONS = (
    asyncio.TimeoutError,
    AssertionError,
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
)


async def _emit_admin_audit_event(
    request: Request,
    principal: AuthPrincipal,
    *,
    event_type: str,
    category: str,
    resource_type: str,
    resource_id: str | None,
    action: str,
    metadata: dict[str, Any],
) -> None:
    from tldw_Server_API.app.api.v1.endpoints import admin as admin_mod

    await admin_mod._emit_admin_audit_event(
        request,
        principal,
        event_type=event_type,
        category=category,
        resource_type=resource_type,
        resource_id=resource_id,
        action=action,
        metadata=metadata,
    )


async def _get_monitoring_repo() -> AuthnzAdminMonitoringRepo:
    """Return the shared admin monitoring repository for the active AuthNZ backend."""
    pool = await get_db_pool()
    return AuthnzAdminMonitoringRepo(pool)


async def _get_users_repo() -> AuthnzUsersRepo:
    """Return the AuthNZ users repository for assignee resolution."""
    pool = await get_db_pool()
    return AuthnzUsersRepo(db_pool=pool)


def _principal_actor_id(principal: AuthPrincipal) -> int | None:
    try:
        return int(principal.user_id) if principal.user_id is not None else None
    except (TypeError, ValueError):
        return None


def _parse_event_details(details_json: str | None) -> dict[str, Any] | None:
    if not details_json:
        return None
    try:
        parsed = json.loads(details_json)
    except json.JSONDecodeError:
        return {"raw": details_json}
    return parsed if isinstance(parsed, dict) else {"value": parsed}


def _event_response_from_row(row: dict[str, Any]) -> AdminAlertEventResponse:
    return AdminAlertEventResponse(
        id=int(row["id"]),
        alert_identity=str(row["alert_identity"]),
        action=str(row["action"]),
        actor_user_id=row.get("actor_user_id"),
        details=_parse_event_details(row.get("details_json")),
        created_at=row.get("created_at"),
    )


def _require_runtime_alert_identity(
    alert_identity: str,
    monitoring_db: TopicMonitoringDB,
) -> int:
    """Validate that an admin overlay mutation targets an existing runtime alert."""
    if not alert_identity.startswith(_RUNTIME_ALERT_ID_PREFIX):
        raise HTTPException(status_code=422, detail="unsupported_alert_identity")

    raw_alert_id = alert_identity[len(_RUNTIME_ALERT_ID_PREFIX) :]
    try:
        alert_id = int(raw_alert_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="malformed_alert_identity") from None

    if monitoring_db.get_alert(alert_id) is None:
        raise HTTPException(status_code=404, detail="unknown_alert")

    return alert_id


async def _get_runtime_alerts_db() -> TopicMonitoringDB:
    """Return the runtime monitoring alerts DB without blocking the event loop."""
    return await asyncio.to_thread(
        TopicMonitoringDB,
        _resolve_monitoring_alerts_db_path(),
    )


async def _require_runtime_alert_identity_for_mutation(
    alert_identity: str,
    monitoring_db: TopicMonitoringDB,
) -> str:
    """Validate an overlay mutation target and return its canonical alert identity."""
    alert_id = await asyncio.to_thread(
        _require_runtime_alert_identity,
        alert_identity,
        monitoring_db,
    )
    return f"{_RUNTIME_ALERT_ID_PREFIX}{alert_id}"


@router.on_event("startup")
async def _admin_monitoring_startup() -> None:
    """Ensure admin monitoring tables exist before request handling begins."""
    repo = await _get_monitoring_repo()
    await repo.ensure_schema_ready_once()


@router.get("/monitoring/alert-rules", response_model=AdminAlertRuleListResponse)
async def list_alert_rules(
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> AdminAlertRuleListResponse:
    """Return platform-wide admin alert rules."""
    try:
        del principal
        repo = await _get_monitoring_repo()
        items = [AdminAlertRuleResponse(**row) for row in await repo.list_rules()]
        return AdminAlertRuleListResponse(items=items)
    except HTTPException:
        raise
    except asyncio.CancelledError:
        raise
    except _MONITORING_NONCRITICAL_EXCEPTIONS as exc:
        logger.error("Failed to list admin alert rules")
        raise HTTPException(status_code=500, detail="Failed to list alert rules") from exc


@router.post("/monitoring/alert-rules", response_model=AdminAlertRuleCreateResponse)
async def create_alert_rule(
    payload: AdminAlertRuleCreateRequest,
    request: Request,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> AdminAlertRuleCreateResponse:
    """Create a shared admin monitoring alert rule."""
    try:
        repo = await _get_monitoring_repo()
        actor_id = _principal_actor_id(principal)
        created = await repo.create_rule(
            metric=payload.metric,
            operator=payload.operator,
            threshold=payload.threshold,
            duration_minutes=payload.duration_minutes,
            severity=payload.severity,
            enabled=payload.enabled,
            created_by_user_id=actor_id,
        )
        await _emit_admin_audit_event(
            request,
            principal,
            event_type="config.changed",
            category="system",
            resource_type="admin_alert_rule",
            resource_id=str(created["id"]),
            action="monitoring.rule.create",
            metadata={
                "metric": payload.metric,
                "operator": payload.operator,
                "severity": payload.severity,
            },
        )
        return AdminAlertRuleCreateResponse(item=AdminAlertRuleResponse(**created))
    except HTTPException:
        raise
    except asyncio.CancelledError:
        raise
    except _MONITORING_NONCRITICAL_EXCEPTIONS as exc:
        logger.error("Failed to create admin alert rule")
        raise HTTPException(status_code=500, detail="Failed to create alert rule") from exc


@router.delete("/monitoring/alert-rules/{rule_id}", response_model=AdminAlertRuleDeleteResponse)
async def delete_alert_rule(
    rule_id: int,
    request: Request,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> AdminAlertRuleDeleteResponse:
    """Delete a shared admin monitoring alert rule."""
    try:
        repo = await _get_monitoring_repo()
        existing = await repo.get_rule(rule_id)
        if existing is None:
            raise HTTPException(status_code=404, detail="unknown_rule")
        deleted = await repo.delete_rule(rule_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="unknown_rule")
        await _emit_admin_audit_event(
            request,
            principal,
            event_type="config.changed",
            category="system",
            resource_type="admin_alert_rule",
            resource_id=str(rule_id),
            action="monitoring.rule.delete",
            metadata={"metric": existing.get("metric")},
        )
        return AdminAlertRuleDeleteResponse(status="deleted", id=rule_id)
    except HTTPException:
        raise
    except asyncio.CancelledError:
        raise
    except _MONITORING_NONCRITICAL_EXCEPTIONS as exc:
        logger.error("Failed to delete admin alert rule")
        raise HTTPException(status_code=500, detail="Failed to delete alert rule") from exc


@router.post("/monitoring/alerts/{alert_identity}/assign", response_model=AdminAlertStateMutationResponse)
async def assign_alert(
    alert_identity: str,
    payload: AdminAlertAssignRequest,
    request: Request,
    principal: AuthPrincipal = Depends(get_auth_principal),
    runtime_alerts_db: TopicMonitoringDB = Depends(_get_runtime_alerts_db),  # noqa: B008
) -> AdminAlertStateMutationResponse:
    """Assign or unassign a runtime-backed monitoring alert in authoritative overlay state."""
    try:
        if payload.assigned_to_user_id is not None:
            users_repo = await _get_users_repo()
            assignee = await users_repo.get_user_by_id(payload.assigned_to_user_id)
            if assignee is None:
                raise HTTPException(status_code=404, detail="unknown_user")
        canonical_alert_identity = await _require_runtime_alert_identity_for_mutation(
            alert_identity,
            runtime_alerts_db,
        )
        repo = await _get_monitoring_repo()
        actor_id = _principal_actor_id(principal)
        event_action = "assigned" if payload.assigned_to_user_id is not None else "unassigned"
        audit_action = (
            "monitoring.alert.assign" if payload.assigned_to_user_id is not None else "monitoring.alert.unassign"
        )
        state = await repo.upsert_alert_state(
            alert_identity=canonical_alert_identity,
            assigned_to_user_id=payload.assigned_to_user_id,
            updated_by_user_id=actor_id,
        )
        await repo.append_alert_event(
            alert_identity=canonical_alert_identity,
            action=event_action,
            actor_user_id=actor_id,
            details_json=json.dumps({"assigned_to_user_id": payload.assigned_to_user_id}),
        )
        await _emit_admin_audit_event(
            request,
            principal,
            event_type="data.update",
            category="system",
            resource_type="monitoring_alert",
            resource_id=canonical_alert_identity,
            action=audit_action,
            metadata={"assigned_to_user_id": payload.assigned_to_user_id},
        )
        return AdminAlertStateMutationResponse(item=AdminAlertStateResponse(**state))
    except HTTPException:
        raise
    except asyncio.CancelledError:
        raise
    except _MONITORING_NONCRITICAL_EXCEPTIONS as exc:
        logger.error("Failed to assign monitoring alert")
        raise HTTPException(status_code=500, detail="Failed to assign alert") from exc


@router.post("/monitoring/alerts/{alert_identity}/snooze", response_model=AdminAlertStateMutationResponse)
async def snooze_alert(
    alert_identity: str,
    payload: AdminAlertSnoozeRequest,
    request: Request,
    principal: AuthPrincipal = Depends(get_auth_principal),
    runtime_alerts_db: TopicMonitoringDB = Depends(_get_runtime_alerts_db),  # noqa: B008
) -> AdminAlertStateMutationResponse:
    """Snooze a runtime-backed monitoring alert in authoritative overlay state."""
    try:
        canonical_alert_identity = await _require_runtime_alert_identity_for_mutation(
            alert_identity,
            runtime_alerts_db,
        )
        repo = await _get_monitoring_repo()
        actor_id = _principal_actor_id(principal)
        state = await repo.upsert_alert_state(
            alert_identity=canonical_alert_identity,
            snoozed_until=payload.snoozed_until.isoformat(),
            updated_by_user_id=actor_id,
        )
        await repo.append_alert_event(
            alert_identity=canonical_alert_identity,
            action="snoozed",
            actor_user_id=actor_id,
            details_json=json.dumps({"snoozed_until": payload.snoozed_until.isoformat()}),
        )
        await _emit_admin_audit_event(
            request,
            principal,
            event_type="data.update",
            category="system",
            resource_type="monitoring_alert",
            resource_id=canonical_alert_identity,
            action="monitoring.alert.snooze",
            metadata={"snoozed_until": payload.snoozed_until.isoformat()},
        )
        return AdminAlertStateMutationResponse(item=AdminAlertStateResponse(**state))
    except HTTPException:
        raise
    except asyncio.CancelledError:
        raise
    except _MONITORING_NONCRITICAL_EXCEPTIONS as exc:
        logger.error("Failed to snooze monitoring alert")
        raise HTTPException(status_code=500, detail="Failed to snooze alert") from exc


@router.post("/monitoring/alerts/{alert_identity}/escalate", response_model=AdminAlertStateMutationResponse)
async def escalate_alert(
    alert_identity: str,
    payload: AdminAlertEscalateRequest,
    request: Request,
    principal: AuthPrincipal = Depends(get_auth_principal),
    runtime_alerts_db: TopicMonitoringDB = Depends(_get_runtime_alerts_db),  # noqa: B008
) -> AdminAlertStateMutationResponse:
    """Escalate a runtime-backed monitoring alert in authoritative overlay state."""
    try:
        canonical_alert_identity = await _require_runtime_alert_identity_for_mutation(
            alert_identity,
            runtime_alerts_db,
        )
        repo = await _get_monitoring_repo()
        actor_id = _principal_actor_id(principal)
        state = await repo.upsert_alert_state(
            alert_identity=canonical_alert_identity,
            escalated_severity=payload.severity,
            updated_by_user_id=actor_id,
        )
        await repo.append_alert_event(
            alert_identity=canonical_alert_identity,
            action="escalated",
            actor_user_id=actor_id,
            details_json=json.dumps({"severity": payload.severity}),
        )
        await _emit_admin_audit_event(
            request,
            principal,
            event_type="data.update",
            category="system",
            resource_type="monitoring_alert",
            resource_id=canonical_alert_identity,
            action="monitoring.alert.escalate",
            metadata={"severity": payload.severity},
        )
        return AdminAlertStateMutationResponse(item=AdminAlertStateResponse(**state))
    except HTTPException:
        raise
    except asyncio.CancelledError:
        raise
    except _MONITORING_NONCRITICAL_EXCEPTIONS as exc:
        logger.error("Failed to escalate monitoring alert")
        raise HTTPException(status_code=500, detail="Failed to escalate alert") from exc


@router.get("/monitoring/alerts/history", response_model=AdminAlertHistoryListResponse)
async def list_alert_history(
    alert_identity: str | None = Query(None),
    limit: int = Query(50, ge=1, le=500),
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> AdminAlertHistoryListResponse:
    """List recent authoritative admin monitoring alert history entries."""
    try:
        del principal
        repo = await _get_monitoring_repo()
        items = [
            _event_response_from_row(row)
            for row in await repo.list_alert_events(alert_identity=alert_identity, limit=limit)
        ]
        return AdminAlertHistoryListResponse(items=items)
    except HTTPException:
        raise
    except asyncio.CancelledError:
        raise
    except _MONITORING_NONCRITICAL_EXCEPTIONS as exc:
        logger.error("Failed to list admin alert history")
        raise HTTPException(status_code=500, detail="Failed to list alert history") from exc
