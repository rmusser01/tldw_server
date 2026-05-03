from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Awaitable
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Callable

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.api.v1.endpoints._pagination_utils import (
    build_offset_pagination_meta,
)
from tldw_Server_API.app.api.v1.schemas.admin_schemas import (
    EmailDeliveryItem,
    EmailDeliveryListResponse,
    FeatureFlagItem,
    FeatureFlagsResponse,
    FeatureFlagUpsertRequest,
    IncidentCreateRequest,
    IncidentEventCreateRequest,
    IncidentItem,
    IncidentListResponse,
    IncidentNotifyRequest,
    IncidentNotifyResponse,
    IncidentUpdateRequest,
    MaintenanceRotationRunCreateRequest,
    MaintenanceRotationRunCreateResponse,
    MaintenanceRotationRunItem,
    MaintenanceRotationRunListResponse,
    MaintenanceState,
    MaintenanceUpdateRequest,
    WebhookCreateRequest,
    WebhookCreateResponse,
    WebhookDeliveryItem,
    WebhookDeliveryListResponse,
    WebhookItem,
    WebhookListResponse,
    WebhookUpdateRequest,
)
from tldw_Server_API.app.api.v1.schemas.auth_schemas import MessageResponse
from tldw_Server_API.app.core.AuthNZ.database import DatabasePool, get_db_pool
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.repos.rbac_repo import AuthnzRbacRepo
from tldw_Server_API.app.core.AuthNZ.repos.users_repo import AuthnzUsersRepo
from tldw_Server_API.app.core.Chat.chat_service import (
    invalidate_model_alias_caches,
)
from tldw_Server_API.app.core.Usage.pricing_catalog import reset_pricing_catalog
from tldw_Server_API.app.services.admin_maintenance_rotation_service import (
    AdminMaintenanceRotationService,
)
from tldw_Server_API.app.services.admin_system_ops_service import (
    add_incident_event as svc_add_incident_event,
)
from tldw_Server_API.app.services.admin_system_ops_service import (
    create_incident as svc_create_incident,
)
from tldw_Server_API.app.services.admin_system_ops_service import (
    create_report_schedule as svc_create_report_schedule,
)
from tldw_Server_API.app.services.admin_system_ops_service import (
    create_webhook as svc_create_webhook,
)
from tldw_Server_API.app.services.admin_system_ops_service import (
    delete_feature_flag as svc_delete_feature_flag,
)
from tldw_Server_API.app.services.admin_system_ops_service import (
    delete_incident as svc_delete_incident,
)
from tldw_Server_API.app.services.admin_system_ops_service import (
    delete_report_schedule as svc_delete_report_schedule,
)
from tldw_Server_API.app.services.admin_system_ops_service import (
    delete_webhook as svc_delete_webhook,
)
from tldw_Server_API.app.services.admin_system_ops_service import (
    get_digest_preference as svc_get_digest_preference,
)
from tldw_Server_API.app.services.admin_system_ops_service import (
    get_incident as svc_get_incident,
)
from tldw_Server_API.app.services.admin_system_ops_service import (
    get_maintenance_state as svc_get_maintenance_state,
)
from tldw_Server_API.app.services.admin_system_ops_service import (
    get_uptime_stats as svc_get_uptime_stats,
)
from tldw_Server_API.app.services.admin_system_ops_service import (
    list_email_deliveries as svc_list_email_deliveries,
)
from tldw_Server_API.app.services.admin_system_ops_service import (
    list_feature_flags as svc_list_feature_flags,
)
from tldw_Server_API.app.services.admin_system_ops_service import (
    list_incidents as svc_list_incidents,
)
from tldw_Server_API.app.services.admin_system_ops_service import (
    list_report_schedules as svc_list_report_schedules,
)
from tldw_Server_API.app.services.admin_system_ops_service import (
    list_webhook_deliveries as svc_list_webhook_deliveries,
)
from tldw_Server_API.app.services.admin_system_ops_service import (
    list_webhooks as svc_list_webhooks,
)
from tldw_Server_API.app.services.admin_system_ops_service import (
    mark_report_schedule_sent as svc_mark_report_schedule_sent,
)
from tldw_Server_API.app.services.admin_system_ops_service import (
    notify_incident_stakeholders as svc_notify_incident_stakeholders,
)
from tldw_Server_API.app.services.admin_system_ops_service import (
    record_health_snapshot as svc_record_health_snapshot,
)
from tldw_Server_API.app.services.admin_system_ops_service import (
    send_test_webhook as svc_send_test_webhook,
)
from tldw_Server_API.app.services.admin_system_ops_service import (
    set_digest_preference as svc_set_digest_preference,
)
from tldw_Server_API.app.services.admin_system_ops_service import (
    update_incident as svc_update_incident,
)
from tldw_Server_API.app.services.admin_system_ops_service import (
    update_maintenance_state as svc_update_maintenance_state,
)
from tldw_Server_API.app.services.admin_system_ops_service import (
    update_report_schedule as svc_update_report_schedule,
)
from tldw_Server_API.app.services.admin_system_ops_service import (
    update_webhook as svc_update_webhook,
)
from tldw_Server_API.app.services.admin_system_ops_service import (
    upsert_feature_flag as svc_upsert_feature_flag,
)

if TYPE_CHECKING:
    from tldw_Server_API.app.core.AuthNZ.repos.maintenance_rotation_runs_repo import (
        AuthnzMaintenanceRotationRunsRepo,
    )

router = APIRouter()

_INCIDENT_ASSIGNABLE_ROLES = frozenset({"admin", "owner", "super_admin"})

_OPS_NONCRITICAL_EXCEPTIONS = (
    asyncio.CancelledError,
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
    HTTPException,
)


def _public_exception_label(exc: Exception) -> str:
    return exc.__class__.__name__ or "operation_failed"


def _raise_internal_admin_error(message: str, exc: Exception) -> None:
    """Raise a generic 500 without reflecting backend exception details to clients."""
    logger.warning("{}: {}", message, exc.__class__.__name__)
    raise HTTPException(status_code=500, detail=message) from exc


def _require_platform_admin(principal: AuthPrincipal) -> None:
    from tldw_Server_API.app.api.v1.endpoints import admin as admin_mod

    return admin_mod._require_platform_admin(principal)


def _enforce_domain_scope_unified(principal: AuthPrincipal, domain: str | None) -> Any:
    """Delegate domain-scope enforcement to the shared jobs admin helper."""
    from tldw_Server_API.app.api.v1.endpoints import jobs_admin as jobs_admin_mod

    return jobs_admin_mod._enforce_domain_scope_unified(principal, domain)


def _user_has_incident_admin_role(user: dict[str, Any], role_rows: list[dict[str, Any]]) -> bool:
    """Return whether the user is eligible for incident assignment."""

    role_names = {
        str(role.get("name") or "").strip().lower() for role in role_rows if str(role.get("name") or "").strip()
    }
    legacy_role = str(user.get("role") or "").strip().lower()
    if legacy_role:
        role_names.add(legacy_role)
    return bool(role_names & _INCIDENT_ASSIGNABLE_ROLES) or bool(user.get("is_superuser"))


async def _resolve_incident_assignee(user_id: int) -> dict[str, Any]:
    """Resolve persisted assignee fields for an incident update.

    Raises:
        ValueError: ``assignee_not_found`` when the user does not exist.
        ValueError: ``incident_assignee_must_be_admin`` when the user is not admin-capable.
    """

    repo = await AuthnzUsersRepo.from_pool()
    user = await repo.get_user_by_id(int(user_id))
    if not user:
        raise ValueError("assignee_not_found")

    rbac_repo = AuthnzRbacRepo()
    role_rows = await asyncio.to_thread(rbac_repo.get_user_roles, int(user_id))
    if not _user_has_incident_admin_role(user, role_rows):
        raise ValueError("incident_assignee_must_be_admin")

    label = str(user.get("email") or "").strip() or str(user.get("username") or "").strip() or str(user["id"])
    return {
        "assigned_to_user_id": int(user["id"]),
        "assigned_to_label": label,
    }


async def _get_admin_org_ids(principal: AuthPrincipal) -> list[int] | None:
    from tldw_Server_API.app.api.v1.endpoints import admin as admin_mod

    return await admin_mod._get_admin_org_ids(principal)


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


@router.get("/maintenance", response_model=MaintenanceState)
async def get_maintenance_mode(
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> MaintenanceState:
    del principal
    state = svc_get_maintenance_state()
    return MaintenanceState(**state)


@router.put("/maintenance", response_model=MaintenanceState)
async def update_maintenance_mode(
    payload: MaintenanceUpdateRequest,
    request: Request,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> MaintenanceState:
    _require_platform_admin(principal)
    actor = principal.email or principal.username or (str(principal.user_id) if principal.user_id is not None else None)
    state = svc_update_maintenance_state(
        enabled=payload.enabled,
        message=payload.message,
        allowlist_user_ids=payload.allowlist_user_ids,
        allowlist_emails=payload.allowlist_emails,
        actor=actor,
    )
    await _emit_admin_audit_event(
        request,
        principal,
        event_type="config.changed",
        category="system",
        resource_type="maintenance",
        resource_id="maintenance_mode",
        action="maintenance.update",
        metadata={"enabled": payload.enabled},
    )
    return MaintenanceState(**state)


async def _get_maintenance_rotation_runs_repo(
    pool: DatabasePool = Depends(get_db_pool),
) -> AuthnzMaintenanceRotationRunsRepo:
    """Build the maintenance rotation runs repository from the shared AuthNZ pool."""
    from tldw_Server_API.app.core.AuthNZ.repos.maintenance_rotation_runs_repo import (
        AuthnzMaintenanceRotationRunsRepo,
    )

    repo = AuthnzMaintenanceRotationRunsRepo(pool)
    await repo.ensure_schema()
    return repo


async def get_admin_maintenance_rotation_service(
    repo: AuthnzMaintenanceRotationRunsRepo = Depends(_get_maintenance_rotation_runs_repo),
) -> AdminMaintenanceRotationService:
    """Build the maintenance rotation service from the injected repository."""
    return AdminMaintenanceRotationService(repo=repo)


def get_maintenance_rotation_job_enqueuer() -> Callable[[dict[str, Any]], Awaitable[str]]:
    """Return the callable used to enqueue maintenance rotation Jobs."""
    from tldw_Server_API.app.services.admin_maintenance_rotation_jobs_worker import (
        enqueue_maintenance_rotation_run,
    )

    return enqueue_maintenance_rotation_run


def _get_maintenance_rotation_allowed_domains(principal: AuthPrincipal) -> list[str] | None:
    """Return the effective domain allowlist for rotation-run history, or None when unrestricted."""
    from tldw_Server_API.app.core.testing import env_flag_enabled

    if not env_flag_enabled("JOBS_DOMAIN_SCOPED_RBAC"):
        return None
    if principal.user_id is None:
        return None
    raw_allowlist = os.getenv(f"JOBS_DOMAIN_ALLOWLIST_{principal.user_id}", "").strip()
    if not raw_allowlist:
        return None
    allowed_domains = [entry.strip() for entry in raw_allowlist.split(",") if entry.strip()]
    return sorted(set(allowed_domains))


@router.post("/maintenance/rotation-runs", response_model=MaintenanceRotationRunCreateResponse)
async def create_maintenance_rotation_run(
    payload: MaintenanceRotationRunCreateRequest,
    request: Request,
    principal: AuthPrincipal = Depends(get_auth_principal),
    repo: AuthnzMaintenanceRotationRunsRepo = Depends(_get_maintenance_rotation_runs_repo),
    service: AdminMaintenanceRotationService = Depends(get_admin_maintenance_rotation_service),
    enqueue_run: Callable[[dict[str, Any]], Awaitable[str]] = Depends(get_maintenance_rotation_job_enqueuer),
) -> MaintenanceRotationRunCreateResponse:
    """Create one authoritative maintenance rotation run and enqueue its Jobs execution."""
    from tldw_Server_API.app.services.admin_maintenance_rotation_jobs_worker import (
        maintenance_rotation_worker_enabled,
    )

    if not maintenance_rotation_worker_enabled():
        raise HTTPException(status_code=503, detail="maintenance_rotation_worker_unavailable")
    _enforce_domain_scope_unified(principal, payload.domain)
    actor_label = (
        principal.email or principal.username or (str(principal.user_id) if principal.user_id is not None else None)
    )
    item = await service.create_run(
        mode=payload.mode,
        domain=payload.domain,
        queue=payload.queue,
        job_type=payload.job_type,
        fields=payload.fields,
        limit=payload.limit,
        confirmed=payload.confirmed,
        requested_by_user_id=principal.user_id,
        requested_by_label=actor_label,
    )
    try:
        await enqueue_run(item)
    except _OPS_NONCRITICAL_EXCEPTIONS:
        await repo.mark_failed(item["id"], error_message="enqueue_failed")
        raise HTTPException(status_code=503, detail="maintenance_rotation_enqueue_failed") from None
    await _emit_admin_audit_event(
        request,
        principal,
        event_type="maintenance.rotation_requested",
        category="system",
        resource_type="maintenance_rotation_run",
        resource_id=item["id"],
        action="maintenance.rotation.create",
        metadata={
            "mode": item["mode"],
            "domain": item.get("domain"),
            "queue": item.get("queue"),
            "job_type": item.get("job_type"),
            "limit": item.get("limit"),
            "confirmation_recorded": item["confirmation_recorded"],
        },
    )
    return MaintenanceRotationRunCreateResponse(item=MaintenanceRotationRunItem(**item))


@router.get("/maintenance/rotation-runs", response_model=MaintenanceRotationRunListResponse)
async def list_maintenance_rotation_runs(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    principal: AuthPrincipal = Depends(get_auth_principal),
    service: AdminMaintenanceRotationService = Depends(get_admin_maintenance_rotation_service),
) -> MaintenanceRotationRunListResponse:
    """List maintenance rotation runs visible to the caller with truthful pagination metadata."""
    payload = await service.list_runs(
        limit=limit,
        offset=offset,
        allowed_domains=_get_maintenance_rotation_allowed_domains(principal),
    )
    return MaintenanceRotationRunListResponse(
        items=[MaintenanceRotationRunItem(**item) for item in payload["items"]],
        total=payload["total"],
        limit=payload["limit"],
        offset=payload["offset"],
        pagination=build_offset_pagination_meta(
            total=payload["total"],
            limit=payload["limit"],
            offset=payload["offset"],
            count=len(payload["items"]),
        ),
    )


@router.get("/maintenance/rotation-runs/{run_id}", response_model=MaintenanceRotationRunItem)
async def get_maintenance_rotation_run(
    run_id: str,
    principal: AuthPrincipal = Depends(get_auth_principal),
    service: AdminMaintenanceRotationService = Depends(get_admin_maintenance_rotation_service),
) -> MaintenanceRotationRunItem:
    """Return one maintenance rotation run after enforcing the caller's domain scope."""
    item = await service.get_run(run_id)
    if item is None:
        raise HTTPException(status_code=404, detail="maintenance_rotation_run_not_found")
    _enforce_domain_scope_unified(principal, item.get("domain"))
    return MaintenanceRotationRunItem(**item)


@router.get("/feature-flags", response_model=FeatureFlagsResponse)
async def list_feature_flags(
    scope: str | None = Query(None, description="global|org|user"),
    org_id: int | None = Query(None, description="Organization ID for org-scoped flags"),
    user_id: int | None = Query(None, description="User ID for user-scoped flags"),
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> FeatureFlagsResponse:
    org_ids = await _get_admin_org_ids(principal)
    if org_id is not None and org_ids is not None:
        org_ids = [org_id] if org_id in org_ids else []
    if org_ids is not None and len(org_ids) == 0:
        return FeatureFlagsResponse(items=[], total=0)
    try:
        items = svc_list_feature_flags(
            scope=scope,
            org_id=org_id if org_ids is None else None,
            user_id=user_id,
        )
    except ValueError as exc:
        detail = str(exc)
        if detail in {"invalid_scope", "missing_org_id", "missing_user_id"}:
            raise HTTPException(status_code=400, detail=detail) from exc
        raise HTTPException(status_code=400, detail="invalid_feature_flag") from exc
    if org_ids is not None:
        items = [item for item in items if item.get("org_id") in org_ids]
    return FeatureFlagsResponse(items=[FeatureFlagItem(**item) for item in items], total=len(items))


@router.put("/feature-flags/{flag_key}", response_model=FeatureFlagItem)
async def upsert_feature_flag(
    flag_key: str,
    payload: FeatureFlagUpsertRequest,
    request: Request,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> FeatureFlagItem:
    _require_platform_admin(principal)
    actor = principal.email or principal.username or (str(principal.user_id) if principal.user_id is not None else None)
    try:
        flag = svc_upsert_feature_flag(
            key=flag_key,
            scope=payload.scope,
            enabled=payload.enabled,
            description=payload.description,
            org_id=payload.org_id,
            user_id=payload.user_id,
            target_user_ids=payload.target_user_ids,
            rollout_percent=payload.rollout_percent,
            variant_value=payload.variant_value,
            actor=actor,
            note=payload.note,
        )
    except ValueError as exc:
        detail = str(exc)
        if detail in {
            "invalid_scope",
            "missing_org_id",
            "missing_user_id",
            "invalid_key",
            "invalid_rollout_percent",
        }:
            raise HTTPException(status_code=400, detail=detail) from exc
        raise HTTPException(status_code=400, detail="invalid_feature_flag") from exc
    await _emit_admin_audit_event(
        request,
        principal,
        event_type="config.changed",
        category="system",
        resource_type="feature_flag",
        resource_id=flag_key,
        action="feature_flag.upsert",
        metadata={
            "scope": payload.scope,
            "enabled": payload.enabled,
            "rollout_percent": payload.rollout_percent,
            "target_user_count": len(payload.target_user_ids or []),
            "variant_value": payload.variant_value,
        },
    )
    return FeatureFlagItem(**flag)


@router.delete("/feature-flags/{flag_key}", response_model=MessageResponse)
async def delete_feature_flag(
    flag_key: str,
    request: Request,
    scope: str = Query(..., description="global|org|user"),
    org_id: int | None = Query(None),
    user_id: int | None = Query(None),
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> MessageResponse:
    _require_platform_admin(principal)
    try:
        svc_delete_feature_flag(key=flag_key, scope=scope, org_id=org_id, user_id=user_id)
    except ValueError as exc:
        detail = str(exc)
        if detail in {"invalid_scope", "missing_org_id", "missing_user_id"}:
            raise HTTPException(status_code=400, detail=detail) from exc
        if detail == "not_found":
            raise HTTPException(status_code=404, detail="feature_flag_not_found") from exc
        raise HTTPException(status_code=400, detail="invalid_feature_flag") from exc
    if request is not None:
        await _emit_admin_audit_event(
            request,
            principal,
            event_type="config.changed",
            category="system",
            resource_type="feature_flag",
            resource_id=flag_key,
            action="feature_flag.delete",
            metadata={"scope": scope, "org_id": org_id, "user_id": user_id},
        )
    return MessageResponse(message="feature_flag_deleted")


@router.get("/incidents", response_model=IncidentListResponse)
async def list_incidents(
    status: str | None = Query(None, description="Incident status"),
    severity: str | None = Query(None, description="Incident severity"),
    tag: str | None = Query(None, description="Filter by tag"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> IncidentListResponse:
    del principal
    items, total = svc_list_incidents(
        status=status,
        severity=severity,
        tag=tag,
        limit=limit,
        offset=offset,
    )
    return IncidentListResponse(
        items=[IncidentItem(**item) for item in items],
        total=total,
        limit=limit,
        offset=offset,
        pagination=build_offset_pagination_meta(
            total=total,
            limit=limit,
            offset=offset,
            count=len(items),
        ),
    )


@router.post("/incidents", response_model=IncidentItem)
async def create_incident(
    payload: IncidentCreateRequest,
    request: Request,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> IncidentItem:
    _require_platform_admin(principal)
    actor = principal.email or principal.username or (str(principal.user_id) if principal.user_id is not None else None)
    try:
        incident = svc_create_incident(
            title=payload.title,
            status=payload.status,
            severity=payload.severity,
            summary=payload.summary,
            tags=payload.tags,
            actor=actor,
        )
    except ValueError as exc:
        detail = str(exc)
        if detail in {"invalid_title", "invalid_status", "invalid_severity"}:
            raise HTTPException(status_code=400, detail=detail) from exc
        raise HTTPException(status_code=400, detail="invalid_incident") from exc
    await _emit_admin_audit_event(
        request,
        principal,
        event_type="ops.incident",
        category="system",
        resource_type="incident",
        resource_id=incident.get("id"),
        action="incident.create",
        metadata={"status": incident.get("status"), "severity": incident.get("severity")},
    )
    return IncidentItem(**incident)


@router.get("/incidents/metrics/sla", response_model=dict)
async def get_incident_sla_metrics(
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> dict:
    """Compute SLA metrics across all incidents."""
    _require_platform_admin(principal)
    incidents, _ = svc_list_incidents(
        status=None,
        severity=None,
        tag=None,
        limit=10000,
        offset=0,
    )

    mtta_values: list[float] = []
    mttr_values: list[float] = []

    for inc in incidents:
        created = inc.get("created_at")
        acknowledged = inc.get("acknowledged_at")
        resolved = inc.get("resolved_at")

        if created and acknowledged:
            try:
                c = datetime.fromisoformat(str(created))
                a = datetime.fromisoformat(str(acknowledged))
                mtta_minutes = (a - c).total_seconds() / 60
                if mtta_minutes >= 0:
                    mtta_values.append(mtta_minutes)
            except (ValueError, TypeError):
                pass

        if created and resolved:
            try:
                c = datetime.fromisoformat(str(created))
                r = datetime.fromisoformat(str(resolved))
                mttr_minutes = (r - c).total_seconds() / 60
                if mttr_minutes >= 0:
                    mttr_values.append(mttr_minutes)
            except (ValueError, TypeError):
                pass

    def _avg(values: list[float]) -> float | None:
        return round(sum(values) / len(values), 1) if values else None

    def _p95(values: list[float]) -> float | None:
        if not values:
            return None
        sorted_v = sorted(values)
        idx = int(len(sorted_v) * 0.95)
        return round(sorted_v[min(idx, len(sorted_v) - 1)], 1)

    return {
        "total_incidents": len(incidents),
        "resolved_count": sum(1 for i in incidents if i.get("resolved_at")),
        "acknowledged_count": sum(1 for i in incidents if i.get("acknowledged_at")),
        "avg_mtta_minutes": _avg(mtta_values),
        "avg_mttr_minutes": _avg(mttr_values),
        "p95_mtta_minutes": _p95(mtta_values),
        "p95_mttr_minutes": _p95(mttr_values),
    }


@router.patch("/incidents/{incident_id}", response_model=IncidentItem)
async def update_incident(
    incident_id: str,
    payload: IncidentUpdateRequest,
    request: Request,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> IncidentItem:
    _require_platform_admin(principal)
    actor = principal.email or principal.username or (str(principal.user_id) if principal.user_id is not None else None)
    update_fields = payload.model_dump(exclude_unset=True)
    workflow_fields: dict[str, Any] = {}
    for field_name in ("root_cause", "impact", "runbook_url", "action_items"):
        if field_name in payload.model_fields_set:
            workflow_fields[field_name] = update_fields[field_name]
    try:
        assignee_fields: dict[str, Any] = {}
        if "assigned_to_user_id" in payload.model_fields_set:
            assignee_user_id = payload.assigned_to_user_id
            if assignee_user_id is None:
                assignee_fields = {
                    "assigned_to_user_id": None,
                    "assigned_to_label": None,
                }
            else:
                assignee_fields = await _resolve_incident_assignee(int(assignee_user_id))

        incident = svc_update_incident(
            incident_id=incident_id,
            title=payload.title,
            status=payload.status,
            severity=payload.severity,
            summary=payload.summary,
            tags=payload.tags,
            **assignee_fields,
            **workflow_fields,
            update_message=payload.update_message,
            actor=actor,
        )
    except ValueError as exc:
        detail = str(exc)
        if detail == "not_found":
            raise HTTPException(status_code=404, detail="incident_not_found") from exc
        if detail == "assignee_not_found":
            raise HTTPException(status_code=404, detail=detail) from exc
        if detail in {"invalid_status", "invalid_severity", "incident_assignee_must_be_admin"}:
            raise HTTPException(status_code=400, detail=detail) from exc
        raise HTTPException(status_code=400, detail="invalid_incident") from exc
    await _emit_admin_audit_event(
        request,
        principal,
        event_type="ops.incident",
        category="system",
        resource_type="incident",
        resource_id=incident_id,
        action="incident.update",
        metadata={"status": incident.get("status"), "severity": incident.get("severity")},
    )
    return IncidentItem(**incident)


@router.post("/incidents/{incident_id}/events", response_model=IncidentItem)
async def add_incident_event(
    incident_id: str,
    payload: IncidentEventCreateRequest,
    request: Request,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> IncidentItem:
    _require_platform_admin(principal)
    actor = principal.email or principal.username or (str(principal.user_id) if principal.user_id is not None else None)
    try:
        incident = svc_add_incident_event(
            incident_id=incident_id,
            message=payload.message,
            actor=actor,
        )
    except ValueError as exc:
        detail = str(exc)
        if detail == "invalid_message":
            raise HTTPException(status_code=400, detail=detail) from exc
        if detail == "not_found":
            raise HTTPException(status_code=404, detail="incident_not_found") from exc
        raise HTTPException(status_code=400, detail="invalid_incident") from exc
    await _emit_admin_audit_event(
        request,
        principal,
        event_type="ops.incident",
        category="system",
        resource_type="incident",
        resource_id=incident_id,
        action="incident.event",
        metadata={"message": payload.message},
    )
    return IncidentItem(**incident)


@router.delete("/incidents/{incident_id}", response_model=MessageResponse)
async def delete_incident(
    incident_id: str,
    request: Request,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> MessageResponse:
    _require_platform_admin(principal)
    try:
        svc_delete_incident(incident_id=incident_id)
    except ValueError as exc:
        detail = str(exc)
        if detail == "not_found":
            raise HTTPException(status_code=404, detail="incident_not_found") from exc
        raise HTTPException(status_code=400, detail="invalid_incident") from exc
    await _emit_admin_audit_event(
        request,
        principal,
        event_type="ops.incident",
        category="system",
        resource_type="incident",
        resource_id=incident_id,
        action="incident.delete",
        metadata={},
    )
    return MessageResponse(message="incident_deleted")


@router.post("/incidents/{incident_id}/notify", response_model=IncidentNotifyResponse)
async def notify_incident_stakeholders(
    incident_id: str,
    payload: IncidentNotifyRequest,
    request: Request,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> IncidentNotifyResponse:
    """Send email notifications to stakeholders about an incident."""
    _require_platform_admin(principal)
    actor = principal.email or principal.username or (str(principal.user_id) if principal.user_id is not None else None)
    if not payload.recipients:
        raise HTTPException(status_code=400, detail="recipients_required")
    try:
        result = svc_notify_incident_stakeholders(
            incident_id=incident_id,
            recipients=payload.recipients,
            message=payload.message,
            actor=actor,
        )
    except ValueError as exc:
        detail = str(exc)
        if detail == "not_found":
            raise HTTPException(status_code=404, detail="incident_not_found") from exc
        raise HTTPException(status_code=400, detail="invalid_notification") from exc
    await _emit_admin_audit_event(
        request,
        principal,
        event_type="ops.incident",
        category="system",
        resource_type="incident",
        resource_id=incident_id,
        action="incident.notify",
        metadata={"recipient_count": len(payload.recipients)},
    )
    return IncidentNotifyResponse(**result)


@router.post("/incidents/{incident_id}/notify-webhooks", response_model=IncidentNotifyResponse)
async def notify_incident_webhooks(
    incident_id: str,
    request: Request,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> IncidentNotifyResponse:
    """Send a notification about an incident to configured webhook channels.

    Dispatches an ``incident.notify`` event to all active admin webhooks
    with the incident's severity, title, status, and affected services.
    """
    _require_platform_admin(principal)

    # Fetch the incident
    incident = svc_get_incident(incident_id=incident_id)
    if incident is None:
        raise HTTPException(status_code=404, detail="incident_not_found")

    from tldw_Server_API.app.services.admin_webhooks_service import get_admin_webhooks_service

    svc = get_admin_webhooks_service()
    wh_payload = {
        "incident_id": incident.get("id"),
        "title": incident.get("title"),
        "severity": incident.get("severity"),
        "status": incident.get("status"),
        "summary": incident.get("summary"),
        "tags": incident.get("tags", []),
        "notified_by": principal.email or principal.username or str(principal.user_id),
    }
    delivered = await svc.dispatch_event("incident.notify", wh_payload)

    await _emit_admin_audit_event(
        request,
        principal,
        event_type="ops.incident",
        category="system",
        resource_type="incident",
        resource_id=incident_id,
        action="incident.notify",
        metadata={"delivered_to": delivered},
    )

    return IncidentNotifyResponse(
        notified=True,
        incident_id=incident_id,
        webhooks_delivered=delivered,
    )


# ---------------------------------------------------------------------------------------------------------------------
# Webhooks
# ---------------------------------------------------------------------------------------------------------------------


@router.get("/webhooks", response_model=WebhookListResponse)
async def list_webhooks(
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> WebhookListResponse:
    """List all configured webhooks (secrets redacted)."""
    _require_platform_admin(principal)
    items = svc_list_webhooks()
    limit = len(items)
    return WebhookListResponse(
        items=[WebhookItem(**item) for item in items],
        total=len(items),
        pagination=build_offset_pagination_meta(
            total=len(items),
            limit=limit,
            offset=0,
            count=len(items),
        ),
    )


@router.post("/webhooks", response_model=WebhookCreateResponse)
async def create_webhook(
    payload: WebhookCreateRequest,
    request: Request,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> WebhookCreateResponse:
    """Create a new webhook. The secret is returned once in this response."""
    _require_platform_admin(principal)
    try:
        webhook = svc_create_webhook(
            url=payload.url,
            events=payload.events,
            enabled=payload.enabled,
        )
    except ValueError as exc:
        detail = str(exc)
        if detail in {"invalid_url", "invalid_events"}:
            raise HTTPException(status_code=400, detail=detail) from exc
        raise HTTPException(status_code=400, detail="invalid_webhook") from exc
    await _emit_admin_audit_event(
        request,
        principal,
        event_type="config.changed",
        category="system",
        resource_type="webhook",
        resource_id=webhook.get("id"),
        action="webhook.create",
        metadata={"url": webhook.get("url"), "events": webhook.get("events")},
    )
    return WebhookCreateResponse(**webhook)


@router.patch("/webhooks/{webhook_id}", response_model=WebhookItem)
async def update_webhook(
    webhook_id: str,
    payload: WebhookUpdateRequest,
    request: Request,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> WebhookItem:
    """Update a webhook. Secret is never returned after creation."""
    _require_platform_admin(principal)
    try:
        webhook = svc_update_webhook(
            webhook_id=webhook_id,
            url=payload.url,
            events=payload.events,
            enabled=payload.enabled,
        )
    except ValueError as exc:
        detail = str(exc)
        if detail == "not_found":
            raise HTTPException(status_code=404, detail="webhook_not_found") from exc
        if detail in {"invalid_url", "invalid_events"}:
            raise HTTPException(status_code=400, detail=detail) from exc
        raise HTTPException(status_code=400, detail="invalid_webhook") from exc
    await _emit_admin_audit_event(
        request,
        principal,
        event_type="config.changed",
        category="system",
        resource_type="webhook",
        resource_id=webhook_id,
        action="webhook.update",
        metadata={"url": webhook.get("url"), "events": webhook.get("events")},
    )
    return WebhookItem(**webhook)


@router.delete("/webhooks/{webhook_id}", response_model=MessageResponse)
async def delete_webhook(
    webhook_id: str,
    request: Request,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> MessageResponse:
    """Delete a webhook."""
    _require_platform_admin(principal)
    try:
        svc_delete_webhook(webhook_id=webhook_id)
    except ValueError as exc:
        detail = str(exc)
        if detail == "not_found":
            raise HTTPException(status_code=404, detail="webhook_not_found") from exc
        raise HTTPException(status_code=400, detail="invalid_webhook") from exc
    await _emit_admin_audit_event(
        request,
        principal,
        event_type="config.changed",
        category="system",
        resource_type="webhook",
        resource_id=webhook_id,
        action="webhook.delete",
        metadata={},
    )
    return MessageResponse(message="webhook_deleted")


@router.get("/webhooks/{webhook_id}/deliveries", response_model=WebhookDeliveryListResponse)
async def list_webhook_deliveries(
    webhook_id: str,
    limit: int = Query(default=50, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> WebhookDeliveryListResponse:
    """List delivery history for a webhook, newest first."""
    _require_platform_admin(principal)
    items, total = svc_list_webhook_deliveries(
        webhook_id=webhook_id,
        limit=limit,
        offset=offset,
    )
    return WebhookDeliveryListResponse(
        items=[WebhookDeliveryItem(**item) for item in items],
        total=total,
        limit=limit,
        offset=offset,
        pagination=build_offset_pagination_meta(
            total=total,
            limit=limit,
            offset=offset,
            count=len(items),
        ),
    )


@router.post("/webhooks/{webhook_id}/test", response_model=WebhookDeliveryItem)
async def test_webhook(
    webhook_id: str,
    request: Request,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> WebhookDeliveryItem:
    """Send a test payload to a webhook and record the delivery result."""
    _require_platform_admin(principal)
    try:
        delivery = await asyncio.to_thread(svc_send_test_webhook, webhook_id=webhook_id)
    except ValueError as exc:
        detail = str(exc)
        if detail == "not_found":
            raise HTTPException(status_code=404, detail="webhook_not_found") from exc
        raise HTTPException(status_code=400, detail="test_delivery_failed") from exc
    await _emit_admin_audit_event(
        request,
        principal,
        event_type="config.changed",
        category="system",
        resource_type="webhook",
        resource_id=webhook_id,
        action="webhook.test",
        metadata={"success": delivery.get("success"), "status_code": delivery.get("status_code")},
    )
    return WebhookDeliveryItem(**delivery)


# ---------------------------------------------------------------------------------------------------------------------
# Billing Analytics
# ---------------------------------------------------------------------------------------------------------------------


@router.get("/billing/analytics")
async def get_billing_analytics(
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> dict:
    """Revenue and subscription analytics.

    Computes MRR, subscriber counts by status, plan distribution,
    and trial conversion rate from the billing database.  Returns
    zeroed metrics when billing is disabled or no subscriptions exist.
    """
    _require_platform_admin(principal)

    from tldw_Server_API.app.core.Billing.runtime_flags import is_billing_enabled

    # Default empty response for when billing is disabled or no data exists.
    empty_response: dict = {
        "mrr_cents": 0,
        "subscriber_count": 0,
        "active_count": 0,
        "trialing_count": 0,
        "past_due_count": 0,
        "canceled_count": 0,
        "plan_distribution": [],
        "trial_conversion_rate_pct": 0.0,
    }

    if not is_billing_enabled():
        return empty_response

    try:
        from tldw_Server_API.app.core.AuthNZ.repos.billing_repo import AuthnzBillingRepo

        pool = await get_db_pool()
        repo = AuthnzBillingRepo(db_pool=pool)

        subscriptions = await repo.list_all_subscriptions()
        if not subscriptions:
            return empty_response

        # Aggregate metrics
        mrr_cents = 0
        active_count = 0
        trialing_count = 0
        past_due_count = 0
        canceled_count = 0
        plan_counts: dict[str, int] = {}
        total_trials_ever = 0
        converted_trials = 0

        for sub in subscriptions:
            status = str(sub.get("status") or "").strip().lower()
            plan_display = str(sub.get("plan_display_name") or sub.get("plan_name") or "Unknown")
            plan_counts[plan_display] = plan_counts.get(plan_display, 0) + 1

            if status == "active":
                active_count += 1
                # MRR: use monthly price for active subs
                price_monthly = sub.get("price_usd_monthly")
                if price_monthly is not None:
                    try:
                        # price_usd_monthly is stored as dollars (float/int);
                        # convert to cents for the response.
                        monthly_cents = int(round(float(price_monthly) * 100))
                    except (ValueError, TypeError):
                        monthly_cents = 0
                    billing_cycle = str(sub.get("billing_cycle") or "monthly").lower()
                    if billing_cycle == "yearly":
                        # For yearly billing, MRR = yearly_price / 12
                        price_yearly = sub.get("price_usd_yearly")
                        if price_yearly is not None:
                            try:
                                yearly_cents = int(round(float(price_yearly) * 100))
                                mrr_cents += yearly_cents // 12
                            except (ValueError, TypeError):
                                mrr_cents += monthly_cents
                        else:
                            mrr_cents += monthly_cents
                    else:
                        mrr_cents += monthly_cents
            elif status == "trialing":
                trialing_count += 1
            elif status == "past_due":
                past_due_count += 1
            elif status == "canceled":
                canceled_count += 1

            # Trial tracking: if there's a trial_end field, this sub had a trial.
            if sub.get("trial_end"):
                total_trials_ever += 1
                # A converted trial is one that reached active status after trialing.
                if status == "active":
                    converted_trials += 1

        subscriber_count = active_count + trialing_count + past_due_count
        trial_conversion_rate = round(converted_trials / total_trials_ever * 100, 1) if total_trials_ever > 0 else 0.0

        plan_distribution = [
            {"plan_name": name, "count": count} for name, count in sorted(plan_counts.items(), key=lambda x: -x[1])
        ]

        return {
            "mrr_cents": mrr_cents,
            "subscriber_count": subscriber_count,
            "active_count": active_count,
            "trialing_count": trialing_count,
            "past_due_count": past_due_count,
            "canceled_count": canceled_count,
            "plan_distribution": plan_distribution,
            "trial_conversion_rate_pct": trial_conversion_rate,
        }

    except Exception:
        logger.warning("billing/analytics: failed to compute metrics")
        return empty_response


# ---------------------------------------------------------------------------------------------------------------------
# Realtime Stats
# ---------------------------------------------------------------------------------------------------------------------


@router.get("/stats/realtime")
async def get_realtime_stats(
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> dict:
    """Realtime operational statistics for dashboard KPIs.

    Returns active ACP session count and aggregate token consumption
    across all sessions.  Gracefully degrades when the ACP session store
    has not been initialised (e.g. ACP is disabled).
    """
    _require_platform_admin(principal)

    active_sessions = 0
    tokens_today: dict[str, int] = {"prompt": 0, "completion": 0, "total": 0}

    try:
        from tldw_Server_API.app.services.admin_acp_sessions_service import (
            get_acp_session_store,
        )

        store = await get_acp_session_store()

        # Count active sessions via list_sessions with status filter
        _records, total_active = await store.list_sessions(status="active", limit=1, offset=0)
        active_sessions = total_active

        # Aggregate token consumption from agent metrics
        metrics = await store.get_agent_metrics()
        for m in metrics:
            tokens_today["prompt"] += int(m.get("total_prompt_tokens", 0) or 0)
            tokens_today["completion"] += int(m.get("total_completion_tokens", 0) or 0)
            tokens_today["total"] += int(m.get("total_tokens", 0) or 0)
    except Exception:
        logger.debug("Realtime stats: ACP session store unavailable")

    return {
        "active_sessions": active_sessions,
        "tokens_today": tokens_today,
    }


# ---------------------------------------------------------------------------------------------------------------------
# Pricing Catalog Management
# ---------------------------------------------


@router.post("/llm-usage/pricing/reload", response_model=dict)
async def reload_llm_pricing_catalog() -> dict:
    """Reload the LLM pricing catalog from environment and config file (admin-only).

    Picks up changes in PRICING_OVERRIDES and Config_Files/model_pricing.json
    without restarting the server.
    """
    try:
        reset_pricing_catalog()
        return {"status": "ok"}
    except _OPS_NONCRITICAL_EXCEPTIONS as e:
        logger.error("Failed to reload pricing catalog")
        raise HTTPException(status_code=500, detail="Failed to reload pricing catalog") from e


# ---------------------------------------------
# Chat Model Alias Cache Management
# ---------------------------------------------


@router.post("/chat/model-aliases/reload", response_model=dict)
async def reload_chat_model_alias_caches() -> dict:
    """Invalidate cached chat model lists and alias overrides (admin-only).

    Clears module-scope lru_caches used by chat model alias resolution so
    updates to Config_Files/model_pricing.json or env vars take effect
    without restarting the server.
    """
    try:
        invalidate_model_alias_caches()
        return {"status": "ok"}
    except _OPS_NONCRITICAL_EXCEPTIONS as e:
        logger.error("Failed to reload chat model alias caches")
        raise HTTPException(status_code=500, detail="Failed to reload chat model alias caches") from e


# ---------------------------------------------------------------------------------------------------------------------
# Compliance Posture
# ---------------------------------------------------------------------------------------------------------------------


@router.get("/compliance/posture", response_model=dict)
async def get_compliance_posture(
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> dict[str, Any]:
    """Aggregate compliance posture metrics for the admin dashboard.

    Returns MFA adoption rate, API key rotation compliance, and an overall
    compliance score.  This endpoint is designed to be called infrequently
    (page-load) so two aggregate queries are acceptable.
    """
    _require_platform_admin(principal)

    pool = await get_db_pool()
    is_pg = bool(getattr(pool, "pool", None))

    # --- MFA adoption ----------------------------------------------------------
    mfa_enabled = 0
    total_users = 0
    try:
        if is_pg:
            row = await pool.fetchone(
                "SELECT COUNT(*) AS total,"
                " COUNT(*) FILTER (WHERE COALESCE(two_factor_enabled, FALSE) = TRUE) AS mfa_on"
                " FROM users WHERE is_active = TRUE"
            )
        else:
            row = await pool.fetchone(
                "SELECT COUNT(*) AS total,"
                " SUM(CASE WHEN COALESCE(two_factor_enabled, 0) = 1 THEN 1 ELSE 0 END) AS mfa_on"
                " FROM users WHERE is_active = 1"
            )
        if row:
            r = dict(row) if hasattr(row, "keys") or isinstance(row, dict) else {"total": row[0], "mfa_on": row[1]}
            total_users = int(r.get("total") or 0)
            mfa_enabled = int(r.get("mfa_on") or 0)
    except Exception:
        logger.warning("compliance/posture: MFA query failed")

    # --- API key rotation compliance (keys older than 180 days need rotation) --
    keys_compliant = 0
    keys_total = 0
    keys_needing_rotation = 0
    rotation_threshold_days = 180
    try:
        now = datetime.now(timezone.utc)
        if is_pg:
            row = await pool.fetchone(
                "SELECT COUNT(*) AS total,"
                " COUNT(*) FILTER (WHERE created_at >= $1) AS compliant"
                " FROM api_keys WHERE status = 'active'",
                now - timedelta(days=rotation_threshold_days),
            )
        else:
            threshold_iso = (now - timedelta(days=rotation_threshold_days)).isoformat()
            row = await pool.fetchone(
                "SELECT COUNT(*) AS total,"
                " SUM(CASE WHEN created_at >= ? THEN 1 ELSE 0 END) AS compliant"
                " FROM api_keys WHERE status = 'active'",
                threshold_iso,
            )
        if row:
            r = dict(row) if hasattr(row, "keys") or isinstance(row, dict) else {"total": row[0], "compliant": row[1]}
            keys_total = int(r.get("total") or 0)
            keys_compliant = int(r.get("compliant") or 0)
            keys_needing_rotation = keys_total - keys_compliant
    except Exception:
        logger.warning("compliance/posture: API key rotation query failed")

    mfa_pct = round((mfa_enabled / total_users * 100), 1) if total_users > 0 else 0.0
    key_rotation_pct = round((keys_compliant / keys_total * 100), 1) if keys_total > 0 else 0.0

    # Simple weighted overall score: MFA 40%, key rotation 40%, 20% baseline for audit logging
    overall = round(mfa_pct * 0.4 + key_rotation_pct * 0.4 + 20, 1)

    return {
        "overall_score": min(overall, 100.0),
        "mfa_adoption_pct": mfa_pct,
        "mfa_enabled_count": mfa_enabled,
        "total_users": total_users,
        "key_rotation_compliance_pct": key_rotation_pct,
        "keys_needing_rotation": keys_needing_rotation,
        "keys_total": keys_total,
        "rotation_threshold_days": rotation_threshold_days,
        "audit_logging_enabled": True,
    }


# ---------------------------------------------------------------------------
# Unified dependency health
# ---------------------------------------------------------------------------


async def _check_dep(
    name: str,
    check_fn: Callable[[], Any],
    timeout: float = 5.0,
) -> dict[str, Any]:
    """Run a single dependency health check with timeout."""
    start = time.monotonic()
    try:
        result = await asyncio.wait_for(check_fn(), timeout=timeout)
        latency = round((time.monotonic() - start) * 1000, 1)
        # Allow check_fn to return a dict with extra metadata
        meta = result if isinstance(result, dict) else {}
        return {
            "name": name,
            "status": meta.get("status", "healthy"),
            "latency_ms": latency,
            "error": meta.get("error"),
            "metadata": {k: v for k, v in meta.items() if k not in ("status", "error")},
        }
    except asyncio.TimeoutError:
        return {"name": name, "status": "down", "latency_ms": 5000.0, "error": "Timeout", "metadata": {}}
    except _OPS_NONCRITICAL_EXCEPTIONS:
        latency = round((time.monotonic() - start) * 1000, 1)
        return {
            "name": name,
            "status": "degraded",
            "latency_ms": latency,
            "error": f"{name} health check failed",
            "metadata": {},
        }


async def _check_authnz_database() -> dict[str, Any]:
    """Check AuthNZ database pool connectivity."""
    pool = await get_db_pool()
    health = await pool.health_check()
    return {
        "status": health.get("status", "unhealthy"),
        "type": health.get("type", "unknown"),
        "pool_size": health.get("pool_size"),
        "idle_connections": health.get("idle_connections"),
    }


async def _check_chacha_notes() -> dict[str, Any]:
    """Check ChaChaNotes DB health snapshot."""
    try:
        from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_health_snapshot

        snapshot = get_chacha_health_snapshot()
        return {
            "status": snapshot.get("status", "unknown"),
            "cached_instances": snapshot.get("cached_instances"),
            "init_failures": snapshot.get("init_failures"),
        }
    except _OPS_NONCRITICAL_EXCEPTIONS:
        return {"status": "unhealthy", "error": "ChaChaNotes health check failed"}


async def _check_workflows_engine() -> dict[str, Any]:
    """Check the workflow scheduler engine."""
    try:
        from tldw_Server_API.app.core.Workflows.engine import WorkflowScheduler

        sched = WorkflowScheduler.instance()
        qd = sched.queue_depth()
        return {"status": "healthy", "queue_depth": qd}
    except _OPS_NONCRITICAL_EXCEPTIONS:
        return {"status": "degraded", "error": "Workflows engine health check failed"}


async def _check_embeddings_service() -> dict[str, Any]:
    """Check embedding service availability."""
    try:
        from tldw_Server_API.app.core.Embeddings.async_embeddings import get_embedding_service

        svc = await get_embedding_service()
        provider_status = await svc.get_provider_status()
        healthy = sum(1 for v in provider_status.values() if v.get("status") == "healthy")
        total = len(provider_status)
        overall = "healthy" if healthy == total else ("degraded" if healthy > 0 else "unhealthy")
        return {"status": overall, "providers_healthy": healthy, "providers_total": total}
    except _OPS_NONCRITICAL_EXCEPTIONS:
        return {"status": "degraded", "error": "Embeddings service health check failed"}


async def _check_metrics_registry() -> dict[str, Any]:
    """Check if the metrics registry is available."""
    try:
        from tldw_Server_API.app.core.Metrics.metrics_manager import get_metrics_registry

        reg = get_metrics_registry()
        return {"status": "healthy" if bool(reg) else "degraded"}
    except _OPS_NONCRITICAL_EXCEPTIONS:
        return {"status": "degraded", "error": "Metrics registry health check failed"}


@router.get("/dependencies")
async def get_all_dependencies(
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> dict[str, Any]:
    """Aggregated health status of all system dependencies.

    Returns a list of dependency checks with status, latency, and error
    information for the admin dashboard system-dependencies view.
    """
    _require_platform_admin(principal)

    checks: list[tuple[str, Callable[[], Any]]] = [
        ("AuthNZ Database", _check_authnz_database),
        ("ChaChaNotes", _check_chacha_notes),
        ("Workflows Engine", _check_workflows_engine),
        ("Embeddings Service", _check_embeddings_service),
        ("Metrics Registry", _check_metrics_registry),
    ]

    results = await asyncio.gather(*[_check_dep(name, fn) for name, fn in checks])

    # Record health snapshot for historical uptime tracking (fire-and-forget)
    try:
        await asyncio.to_thread(svc_record_health_snapshot, list(results))
    except _OPS_NONCRITICAL_EXCEPTIONS:
        pass

    return {
        "items": list(results),
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/dependencies/{name}/uptime")
async def get_dependency_uptime(
    name: str,
    days: int = Query(default=30, ge=1, le=90),
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> dict[str, Any]:
    """Historical uptime statistics for a single system dependency.

    Returns uptime percentage, average latency, downtime estimate,
    and an hourly sparkline for the last *days* (or 7 days if shorter).
    """
    _require_platform_admin(principal)
    try:
        stats = await asyncio.to_thread(svc_get_uptime_stats, name, days)
    except _OPS_NONCRITICAL_EXCEPTIONS as exc:
        _raise_internal_admin_error("Failed to get dependency uptime", exc)
    return stats


# ──────────────────────────────────────────────────────────────────────────────
# Email Delivery Log
# ──────────────────────────────────────────────────────────────────────────────


@router.get("/email/deliveries", response_model=EmailDeliveryListResponse)
async def list_email_deliveries(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    status: str | None = Query(default=None),
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> EmailDeliveryListResponse:
    """List email delivery log entries with optional status filter and pagination."""
    _require_platform_admin(principal)
    try:
        items, total = await asyncio.to_thread(svc_list_email_deliveries, limit=limit, offset=offset, status=status)
    except _OPS_NONCRITICAL_EXCEPTIONS as exc:
        _raise_internal_admin_error("Failed to list email deliveries", exc)
    pagination = build_offset_pagination_meta(
        total=total,
        limit=limit,
        offset=offset,
        count=len(items),
    )
    return EmailDeliveryListResponse(
        items=[EmailDeliveryItem(**item) for item in items],
        total=total,
        limit=limit,
        offset=offset,
        pagination=pagination,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Compliance Report Schedules
# ──────────────────────────────────────────────────────────────────────────────


@router.get("/compliance/report-schedules")
async def get_report_schedules(
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> dict[str, Any]:
    """List all compliance report schedules."""
    _require_platform_admin(principal)
    try:
        items = await asyncio.to_thread(svc_list_report_schedules)
    except _OPS_NONCRITICAL_EXCEPTIONS as exc:
        _raise_internal_admin_error("Failed to list report schedules", exc)
    return {"items": items, "total": len(items)}


@router.post("/compliance/report-schedules")
async def create_report_schedule(
    request: Request,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> dict[str, Any]:
    """Create a new compliance report schedule."""
    _require_platform_admin(principal)
    body = await request.json()
    try:
        schedule = await asyncio.to_thread(
            svc_create_report_schedule,
            frequency=body.get("frequency", "weekly"),
            recipients=body.get("recipients", []),
            report_format=body.get("format", "html"),
            enabled=body.get("enabled", True),
        )
    except ValueError as exc:
        error_key = str(exc)
        if error_key == "too_many_report_schedules":
            raise HTTPException(
                status_code=429,
                detail="Maximum number of report schedules reached.",
            ) from exc
        raise HTTPException(status_code=400, detail=error_key) from exc
    except _OPS_NONCRITICAL_EXCEPTIONS as exc:
        _raise_internal_admin_error("Failed to create report schedule.", exc)
    return schedule


@router.patch("/compliance/report-schedules/{schedule_id}")
async def update_report_schedule(
    schedule_id: str,
    request: Request,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> dict[str, Any]:
    """Update an existing compliance report schedule."""
    _require_platform_admin(principal)
    body = await request.json()
    try:
        schedule = await asyncio.to_thread(
            svc_update_report_schedule,
            schedule_id=schedule_id,
            frequency=body.get("frequency"),
            recipients=body.get("recipients"),
            report_format=body.get("format"),
            enabled=body.get("enabled"),
        )
    except ValueError as exc:
        error_key = str(exc)
        if error_key == "not_found":
            raise HTTPException(status_code=404, detail="Schedule not found.") from exc
        raise HTTPException(status_code=400, detail=error_key) from exc
    except _OPS_NONCRITICAL_EXCEPTIONS as exc:
        _raise_internal_admin_error("Failed to update report schedule.", exc)
    return schedule


@router.delete("/compliance/report-schedules/{schedule_id}")
async def delete_report_schedule(
    schedule_id: str,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> dict[str, Any]:
    """Delete a compliance report schedule."""
    _require_platform_admin(principal)
    try:
        removed = await asyncio.to_thread(svc_delete_report_schedule, schedule_id=schedule_id)
    except ValueError as exc:
        error_key = str(exc)
        if error_key == "not_found":
            raise HTTPException(status_code=404, detail="Schedule not found.") from exc
        raise HTTPException(status_code=400, detail=error_key) from exc
    except _OPS_NONCRITICAL_EXCEPTIONS as exc:
        _raise_internal_admin_error("Failed to delete report schedule.", exc)
    return {"message": "Schedule deleted.", "schedule": removed}


@router.post("/compliance/report-schedules/{schedule_id}/send-now")
async def send_report_now(
    schedule_id: str,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> dict[str, Any]:
    """Generate and send a compliance report immediately for a schedule.

    Reuses the compliance posture data and sends it via the email service
    to the schedule's recipients.
    """
    _require_platform_admin(principal)

    # Fetch the schedule
    try:
        schedules = await asyncio.to_thread(svc_list_report_schedules)
    except _OPS_NONCRITICAL_EXCEPTIONS as exc:
        _raise_internal_admin_error("Failed to load report schedules.", exc)

    schedule = next((s for s in schedules if s.get("id") == schedule_id), None)
    if not schedule:
        raise HTTPException(status_code=404, detail="Schedule not found.")

    # Generate posture data (reuse the existing endpoint logic inline)
    posture = await get_compliance_posture(principal=principal)

    # Build report content
    recipients = schedule.get("recipients", [])
    report_format = schedule.get("format", "html")
    if report_format == "json":
        import json as _json

        report_body = _json.dumps(posture, indent=2)
    else:
        report_body = _build_compliance_html_report(posture)

    # Send via email service (best-effort)
    sent_count = 0
    errors: list[str] = []
    try:
        from tldw_Server_API.app.core.AuthNZ.email_service import get_email_service
        from tldw_Server_API.app.services.admin_system_ops_service import (
            record_email_delivery as svc_record_email_delivery,
        )

        email_service = get_email_service()
        for recipient in recipients:
            try:
                ok = await email_service.send_email(
                    to_email=recipient,
                    subject="Compliance Report",
                    html_body=report_body if report_format == "html" else "<pre>" + report_body + "</pre>",
                    text_body=report_body if report_format != "html" else None,
                    _template="compliance_report",
                )
                if ok:
                    sent_count += 1
                    svc_record_email_delivery(
                        recipient=recipient,
                        subject="Compliance Report",
                        template="compliance_report",
                        status="sent",
                    )
                else:
                    errors.append(f"{recipient}: delivery returned false")
                    svc_record_email_delivery(
                        recipient=recipient,
                        subject="Compliance Report",
                        template="compliance_report",
                        status="failed",
                        error="delivery returned false",
                    )
            except Exception as send_exc:
                errors.append(f"{recipient}: delivery failed")
                svc_record_email_delivery(
                    recipient=recipient,
                    subject="Compliance Report",
                    template="compliance_report",
                    status="failed",
                    error=_public_exception_label(send_exc),
                )
    except Exception as exc:
        logger.warning(
            "Compliance report send-now: email service unavailable: {}",
            exc.__class__.__name__,
        )
        errors.append("email service: unavailable")

    # Mark as sent
    try:
        await asyncio.to_thread(svc_mark_report_schedule_sent, schedule_id=schedule_id)
    except _OPS_NONCRITICAL_EXCEPTIONS:
        pass

    return {
        "sent_count": sent_count,
        "total_recipients": len(recipients),
        "errors": errors,
    }


def _build_compliance_html_report(posture: dict[str, Any]) -> str:
    """Build a simple HTML compliance report from posture data."""
    score = posture.get("overall_score", 0)
    mfa_pct = posture.get("mfa_adoption_pct", 0)
    mfa_count = posture.get("mfa_enabled_count", 0)
    total_users = posture.get("total_users", 0)
    key_pct = posture.get("key_rotation_compliance_pct", 0)
    keys_needing = posture.get("keys_needing_rotation", 0)
    keys_total = posture.get("keys_total", 0)
    audit = posture.get("audit_logging_enabled", False)

    return f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>Compliance Report</title></head>
<body style="font-family:sans-serif;max-width:600px;margin:0 auto;padding:20px;">
<h1 style="color:#333;">Compliance Report</h1>
<p>Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</p>
<table style="width:100%;border-collapse:collapse;">
<tr><td style="padding:8px;border-bottom:1px solid #eee;font-weight:bold;">Overall Score</td>
<td style="padding:8px;border-bottom:1px solid #eee;">{score:.0f}/100</td></tr>
<tr><td style="padding:8px;border-bottom:1px solid #eee;font-weight:bold;">MFA Adoption</td>
<td style="padding:8px;border-bottom:1px solid #eee;">{mfa_pct:.0f}% ({mfa_count}/{total_users} users)</td></tr>
<tr><td style="padding:8px;border-bottom:1px solid #eee;font-weight:bold;">Key Rotation</td>
<td style="padding:8px;border-bottom:1px solid #eee;">{key_pct:.0f}% ({keys_needing} of {keys_total} need rotation)</td></tr>
<tr><td style="padding:8px;border-bottom:1px solid #eee;font-weight:bold;">Audit Logging</td>
<td style="padding:8px;border-bottom:1px solid #eee;">{"Enabled" if audit else "Disabled"}</td></tr>
</table>
</body>
</html>"""


# ──────────────────────────────────────────────────────────────────────────────
# Email Digest Preferences
# ──────────────────────────────────────────────────────────────────────────────


@router.get("/digest/preference")
async def get_digest_preference(
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> dict[str, Any]:
    """Get the current user's email digest preference."""
    user_id = str(getattr(principal, "user_id", None) or "")
    if not user_id:
        raise HTTPException(status_code=401, detail="User identity required.")
    try:
        pref = await asyncio.to_thread(svc_get_digest_preference, user_id=user_id)
    except _OPS_NONCRITICAL_EXCEPTIONS as exc:
        _raise_internal_admin_error("Failed to get digest preference", exc)
    if pref is None:
        return {"user_id": user_id, "email": "", "frequency": "off", "enabled": False}
    return pref


@router.put("/digest/preference")
async def set_digest_preference(
    request: Request,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> dict[str, Any]:
    """Set or update the current user's email digest preference."""
    user_id = str(getattr(principal, "user_id", None) or "")
    if not user_id:
        raise HTTPException(status_code=401, detail="User identity required.")
    body = await request.json()
    email = body.get("email", "")
    frequency = body.get("frequency", "off")
    try:
        pref = await asyncio.to_thread(
            svc_set_digest_preference,
            user_id=user_id,
            email=email,
            frequency=frequency,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except _OPS_NONCRITICAL_EXCEPTIONS as exc:
        _raise_internal_admin_error("Failed to set digest preference", exc)
    return pref
