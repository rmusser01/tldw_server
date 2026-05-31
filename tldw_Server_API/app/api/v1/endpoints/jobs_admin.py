from __future__ import annotations

import os
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

#
try:
    from pydantic import field_validator  # Pydantic v2
except ImportError:  # Fallback to v1 naming
    from pydantic import validator as field_validator  # type: ignore

import asyncio
import contextlib
import json as _json

from fastapi.responses import StreamingResponse

from tldw_Server_API.app.api.v1.API_Deps.Audit_DB_Deps import get_audit_service_for_user
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
    RequireRole,
    get_auth_principal,
)
from tldw_Server_API.app.core.Audit.unified_audit_service import AuditContext, AuditEventType
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.testing import (
    env_flag_enabled,
    is_test_mode,
    is_truthy as _shared_is_truthy,
)

_JOBS_ADMIN_NONCRITICAL_EXCEPTIONS = (
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
    _json.JSONDecodeError,
    HTTPException,
)

router = APIRouter(
    dependencies=[Depends(RequireRole("admin"))],
)


def _is_truthy(v: str | None) -> bool:
    return _shared_is_truthy(v)


def _make_admin_user_from_principal(principal: AuthPrincipal) -> dict[str, Any]:
    """Derive the minimal jobs-admin user dict from an AuthPrincipal.

    For user-backed principals, the id field reflects the numeric user_id so
    existing domain allowlist and RLS semantics continue to apply. For
    non-user principals (e.g., service or API-key callers) where user_id may
    be None, id is left as None and username falls back to subject or
    principal_id (or kind) for audit/diagnostics. Downstream RBAC and RLS
    enforcement must rely only on the id field; username is informational.
    """
    user_id = principal.user_id
    if user_id is not None:
        # Prefer a stable, human-readable label when available.
        username = principal.subject or principal.principal_id or f"user:{user_id}"
    else:
        # Preserve RLS/allowlist behavior by not fabricating a synthetic id.
        # Use subject/principal_id/kind for human-readable/audit labels instead.
        username = principal.subject or principal.principal_id or principal.kind or "service"
    return {
        "id": user_id,
        "username": str(username),
    }


def _enforce_domain_scope(user: dict, domain: str | None) -> None:
    """Optional domain-scoped RBAC enforcement.

    Enabled when JOBS_DOMAIN_SCOPED_RBAC=true. If JOBS_REQUIRE_DOMAIN_FILTER=true,
    a domain filter must be provided. If an allowlist is configured for the user
    via JOBS_DOMAIN_ALLOWLIST_<USER_ID>, the requested domain must be in it.
    """
    try:
        if not _is_truthy(os.getenv("JOBS_DOMAIN_SCOPED_RBAC")):
            return
        # Be robust to user being a Pydantic model or dict
        try:
            uid_val = getattr(user, "id", None)
        except _JOBS_ADMIN_NONCRITICAL_EXCEPTIONS:
            uid_val = None
        if uid_val is None and isinstance(user, dict):
            uid_val = user.get("id")
        uid = str(uid_val or "")
        if _is_truthy(os.getenv("JOBS_REQUIRE_DOMAIN_FILTER")) and not (domain and domain.strip()):
            raise HTTPException(status_code=403, detail="Domain filter is required for this operation")
        allow = os.getenv(f"JOBS_DOMAIN_ALLOWLIST_{uid}", "").strip()
        if allow:
            allowed = {d.strip() for d in allow.split(",") if d.strip()}
            if domain and domain.strip():
                if domain not in allowed:
                    raise HTTPException(status_code=403, detail=f"Not allowed for domain {domain}")
            else:
                # denying broad queries when allowlist is present and domain missing
                raise HTTPException(status_code=403, detail="Domain filter required for allowlisted admin")
    except HTTPException:
        raise
    except _JOBS_ADMIN_NONCRITICAL_EXCEPTIONS as _rbac_exc:
        # Fail-closed in forced mode (tests), otherwise fail-open to avoid lockout
        if _is_truthy(os.getenv("JOBS_RBAC_FORCE")):
            raise HTTPException(status_code=403, detail="RBAC enforcement error") from _rbac_exc
        return


def _enforce_domain_scope_from_principal(principal: AuthPrincipal, domain: str | None) -> None:
    """
    Domain-scoped RBAC enforcement using AuthPrincipal (feature-flagged path).

    When JOBS_DOMAIN_RBAC_PRINCIPAL=1, jobs admin endpoints call this helper
    instead of passing a user dict directly to _enforce_domain_scope as part
    of the principal-driven RBAC migration.
    The underlying semantics (JOBS_DOMAIN_SCOPED_RBAC, JOBS_REQUIRE_DOMAIN_FILTER,
    JOBS_DOMAIN_ALLOWLIST_*, JOBS_RBAC_FORCE) remain unchanged.
    """
    if not _is_truthy(os.getenv("JOBS_DOMAIN_RBAC_PRINCIPAL")):
        return
    user = _make_admin_user_from_principal(principal)
    _enforce_domain_scope(user, domain)


def _enforce_domain_scope_unified(
    principal: AuthPrincipal,
    domain: str | None,
) -> dict[str, Any]:
    """
    Unified domain-scoped RBAC enforcement for jobs admin endpoints.

    All jobs-admin endpoints in this module use this helper alongside the
    router-level RequireRole(\"admin\") guard. It always derives the
    admin_user from the AuthPrincipal so callers can reuse the same user
    mapping for downstream operations (e.g., Postgres RLS). When
    JOBS_DOMAIN_RBAC_PRINCIPAL is enabled, enforcement is driven from the
    AuthPrincipal; otherwise the legacy user-dict path is used.
    """
    admin_user = _make_admin_user_from_principal(principal)
    if _is_truthy(os.getenv("JOBS_DOMAIN_RBAC_PRINCIPAL")):
        _enforce_domain_scope_from_principal(principal, domain)
    else:
        _enforce_domain_scope(admin_user, domain)
    return admin_user


def _set_pg_rls_for_user(user: dict, domain: str | None) -> None:
    """Set per-request RLS context for Postgres sessions using contextvars.

    This avoids global env and is concurrency-safe within the request task.
    """
    try:
        from tldw_Server_API.app.core.Jobs.manager import JobManager as _JM
        uid = str(user.get("id") or "")
        _JM.set_rls_context(is_admin=True, domain_allowlist=str(domain) if domain else None, owner_user_id=uid)
    except _JOBS_ADMIN_NONCRITICAL_EXCEPTIONS:
        pass


class PruneRequest(BaseModel):
    statuses: list[str] = Field(default_factory=lambda: ["completed", "failed", "cancelled"], description="Statuses to prune")
    older_than_days: int = Field(ge=1, le=3650, default=30, description="Delete jobs older than N days")
    # Optional scope filters
    domain: str | None = Field(default=None, description="Limit prune to a specific domain")
    queue: str | None = Field(default=None, description="Limit prune to a specific queue")
    job_type: str | None = Field(default=None, description="Limit prune to a specific job type")
    dry_run: bool = Field(default=False, description="When true, return count only without deleting")
    detail_top_k: int = Field(default=0, ge=0, le=100, description="When dry_run is true, optionally compute top-K groups by count")

    @field_validator("statuses", mode="before")
    @classmethod
    def _norm_statuses(cls, v):
        # Expect list; normalize each item to allowed set
        allowed = {"queued", "processing", "completed", "failed", "cancelled"}
        try:
            items = list(v) if isinstance(v, (list, tuple)) else [v]
        except _JOBS_ADMIN_NONCRITICAL_EXCEPTIONS:
            raise ValueError("statuses must be a list of strings") from None
        out = []
        for item in items:
            s = str(item or "").strip().lower()
            if s not in allowed:
                raise ValueError(f"Unsupported status: {s}")
            out.append(s)
        return out

    @field_validator("domain", "queue", "job_type", mode="before")
    @classmethod
    def _trim_optional(cls, v: str | None) -> str | None:
        s = str(v or "").strip()
        return s or None

    model_config = ConfigDict(json_schema_extra={
            "example": {
                "statuses": ["completed", "failed"],
                "older_than_days": 30,
                "domain": "chatbooks",
                "queue": "default",
                "job_type": "export",
                "dry_run": True,
            }
        })


class PruneResponse(BaseModel):
    deleted: int


@router.post(
    "/jobs/prune",
    response_model=PruneResponse,
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "dryRun": {
                            "summary": "Dry run prune (scoped)",
                            "value": {
                                "statuses": ["completed", "failed", "cancelled"],
                                "older_than_days": 30,
                                "domain": "chatbooks",
                                "queue": "default",
                                "job_type": "export",
                                "dry_run": True
                            },
                        },
                        "execute": {
                            "summary": "Execute prune (requires X-Confirm: true)",
                            "value": {
                                "statuses": ["completed", "failed"],
                                "older_than_days": 14,
                                "domain": "chatbooks",
                                "queue": "default",
                                "job_type": "export",
                                "dry_run": False
                            },
                        },
                    }
                }
            }
        },
        "responses": {
            "200": {"content": {"application/json": {"example": {"deleted": 42}}}},
            "400": {"description": "Missing X-Confirm header for destructive action"},
        },
    },
)
async def prune_jobs_endpoint(
    request: Request,
    audit_service=Depends(get_audit_service_for_user),
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> PruneResponse:
    """Delete jobs matching statuses and older than threshold.

    Requires authentication. Use cautiously.
    """
    try:
        # Pre-parse raw JSON to enforce RBAC before model validation to avoid 422s
        try:
            raw_body = await request.json()
        except _JOBS_ADMIN_NONCRITICAL_EXCEPTIONS:
            raw_body = {}
        # Enforce domain-scoped RBAC (403) even if request body is incomplete.
        # When JOBS_DOMAIN_RBAC_PRINCIPAL is enabled, drive enforcement from
        # AuthPrincipal; otherwise fall back to the legacy user dict path.
        domain_val = (raw_body or {}).get("domain")
        admin_user = _enforce_domain_scope_unified(principal, domain_val)
        # Confirm header for destructive action (skip when dry_run or in TEST_MODE)
        if not bool((raw_body or {}).get("dry_run")):
            is_test = is_test_mode()
            require_confirm_env = env_flag_enabled("JOBS_REQUIRE_CONFIRM")
            try:
                older = int((raw_body or {}).get("older_than_days") or 0)
            except _JOBS_ADMIN_NONCRITICAL_EXCEPTIONS:
                older = 0
            # Require confirmation when explicitly enabled (except in TEST_MODE),
            # or when pruning with immediate threshold (older_than_days <= 0)
            if (require_confirm_env and not is_test) or (older <= 0):
                hdr = str(request.headers.get("x-confirm", "")).lower()
                if not _is_truthy(hdr):
                    raise HTTPException(status_code=400, detail="Confirmation required: set X-Confirm: true")
        # Now validate the request body
        req = PruneRequest(**(raw_body or {}))

        db_url = os.getenv("JOBS_DB_URL")
        backend = "postgres" if (db_url and db_url.startswith("postgres")) else None
        if backend == "postgres":
            _set_pg_rls_for_user(admin_user, req.domain)
        jm = JobManager(backend=backend, db_url=db_url)
        deleted = jm.prune_jobs(
            statuses=req.statuses,
            older_than_days=req.older_than_days,
            domain=req.domain,
            queue=req.queue,
            job_type=req.job_type,
            dry_run=req.dry_run,
            detail_top_k=req.detail_top_k,
        )
        # Optionally refresh gauges for a fully-scoped prune (avoid heavy recompute by default)
        try:
            if (
                not req.dry_run
                and req.domain
                and req.queue
                and req.job_type
                and env_flag_enabled("JOBS_UPDATE_GAUGES_ON_PRUNE")
            ):
                jm._update_gauges(domain=req.domain, queue=req.queue, job_type=req.job_type)
        except _JOBS_ADMIN_NONCRITICAL_EXCEPTIONS:
            pass
        # Best-effort audit logging for admin prune action
        try:
            ctx = AuditContext(
                user_id=str(admin_user.get("id") or principal.principal_id),
                endpoint="/api/v1/jobs/prune",
                method="POST",
            )
            await audit_service.log_event(
                event_type=AuditEventType.DATA_DELETE,
                context=ctx,
                resource_type="jobs",
                action="prune",
                result="success",
                result_count=deleted,
                metadata={
                    "statuses": req.statuses,
                    "older_than_days": req.older_than_days,
                    "domain": req.domain,
                    "queue": req.queue,
                    "job_type": req.job_type,
                    "dry_run": req.dry_run,
                },
            )
        except _JOBS_ADMIN_NONCRITICAL_EXCEPTIONS:
            # Never fail prune due to audit logging issues
            pass
        return PruneResponse(deleted=deleted)
    except HTTPException:
        # Preserve intended HTTP errors (e.g., RBAC 403)
        raise
    except _JOBS_ADMIN_NONCRITICAL_EXCEPTIONS as e:
        raise HTTPException(status_code=500, detail="Prune failed") from e


# --- Queue controls (pause/resume/drain) ---
class QueueControlRequest(BaseModel):
    domain: str
    queue: str
    action: str = Field(description="pause|resume|drain")


class QueueFlagsResponse(BaseModel):
    paused: bool
    drain: bool


@router.post("/jobs/queue/control", response_model=QueueFlagsResponse)
async def queue_control_endpoint(
    req: QueueControlRequest,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> QueueFlagsResponse:
    admin_user = _enforce_domain_scope_unified(principal, req.domain)
    db_url = os.getenv("JOBS_DB_URL")
    backend = "postgres" if (db_url and db_url.startswith("postgres")) else None
    if backend == "postgres":
        _set_pg_rls_for_user(admin_user, req.domain)
    jm = JobManager(backend=backend, db_url=db_url)
    try:
        flags = jm.set_queue_control(req.domain, req.queue, req.action)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve)) from ve
    return QueueFlagsResponse(**flags)


@router.get("/jobs/queue/status", response_model=QueueFlagsResponse)
async def queue_status_endpoint(
    domain: str,
    queue: str,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> QueueFlagsResponse:
    admin_user = _enforce_domain_scope_unified(principal, domain)
    db_url = os.getenv("JOBS_DB_URL")
    backend = "postgres" if (db_url and db_url.startswith("postgres")) else None
    if backend == "postgres":
        _set_pg_rls_for_user(admin_user, domain)
    jm = JobManager(backend=backend, db_url=db_url)
    flags = jm._get_queue_flags(domain, queue)
    return QueueFlagsResponse(**flags)


# --- Reschedule / Retry-now ---
class RescheduleRequest(BaseModel):
    domain: str | None = None
    queue: str | None = None
    job_type: str | None = None
    status: str | None = Field(default=None, description="Optional status filter")
    set_now: bool = True
    delta_seconds: int | None = None
    dry_run: bool = False


class AffectedResponse(BaseModel):
    affected: int


@router.post("/jobs/reschedule", response_model=AffectedResponse)
async def reschedule_jobs_endpoint(
    req: RescheduleRequest,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> AffectedResponse:
    admin_user = _enforce_domain_scope_unified(principal, req.domain)
    db_url = os.getenv("JOBS_DB_URL")
    backend = "postgres" if (db_url and db_url.startswith("postgres")) else None
    if backend == "postgres":
        _set_pg_rls_for_user(admin_user, req.domain)
    jm = JobManager(backend=backend, db_url=db_url)
    try:
        n = jm.reschedule_jobs(domain=req.domain, queue=req.queue, job_type=req.job_type, status=req.status, set_now=req.set_now, delta_seconds=req.delta_seconds, dry_run=req.dry_run)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve)) from ve
    return AffectedResponse(affected=int(n))


class RetryNowRequest(BaseModel):
    domain: str | None = None
    queue: str | None = None
    job_type: str | None = None
    job_id: int | None = None
    only_failed: bool = True
    dry_run: bool = False


@router.post("/jobs/retry-now", response_model=AffectedResponse)
async def retry_now_jobs_endpoint(
    req: RetryNowRequest,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> AffectedResponse:
    admin_user = _enforce_domain_scope_unified(principal, req.domain)
    db_url = os.getenv("JOBS_DB_URL")
    backend = "postgres" if (db_url and db_url.startswith("postgres")) else None
    if backend == "postgres":
        _set_pg_rls_for_user(admin_user, req.domain)
    jm = JobManager(backend=backend, db_url=db_url)
    n = jm.retry_now_jobs(
        job_id=req.job_id,
        domain=req.domain,
        queue=req.queue,
        job_type=req.job_type,
        only_failed=req.only_failed,
        dry_run=req.dry_run,
    )
    return AffectedResponse(affected=int(n))


# --- Attachments ---
class AttachmentRequest(BaseModel):
    kind: str = Field(description="log|artifact|tag")
    content_text: str | None = None
    url: str | None = None


class AttachmentItem(BaseModel):
    id: int
    kind: str
    content_text: str | None
    url: str | None
    created_at: str


def _normalize_attachment_item(item: dict[str, Any]) -> AttachmentItem:
    created_at = item.get("created_at")
    if isinstance(created_at, datetime):
        created_at = created_at.isoformat()
    elif created_at is None:
        created_at = ""
    else:
        created_at = str(created_at)
    payload = dict(item)
    payload["created_at"] = created_at
    return AttachmentItem(**payload)


@router.post("/jobs/{job_id}/attachments", response_model=AttachmentItem)
async def add_job_attachment_endpoint(
    job_id: int,
    req: AttachmentRequest,
    principal: AuthPrincipal = Depends(get_auth_principal),
    domain: str | None = None,
) -> AttachmentItem:
    admin_user = _enforce_domain_scope_unified(principal, domain)
    db_url = os.getenv("JOBS_DB_URL")
    backend = "postgres" if (db_url and db_url.startswith("postgres")) else None
    if backend == "postgres":
        _set_pg_rls_for_user(admin_user, domain)
    jm = JobManager(backend=backend, db_url=db_url)
    try:
        rid = jm.add_job_attachment(job_id, kind=req.kind, content_text=req.content_text, url=req.url)
        items = jm.list_job_attachments(job_id, limit=1_000)
        item = next((i for i in items if int(i.get('id')) == int(rid)), None)
        if not item:
            raise HTTPException(status_code=500, detail="Failed to read back attachment")
        return _normalize_attachment_item(item)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve)) from ve


@router.get("/jobs/{job_id}/attachments", response_model=list[AttachmentItem])
async def list_job_attachments_endpoint(
    job_id: int,
    principal: AuthPrincipal = Depends(get_auth_principal),
    domain: str | None = None,
) -> list[AttachmentItem]:
    admin_user = _enforce_domain_scope_unified(principal, domain)
    db_url = os.getenv("JOBS_DB_URL")
    backend = "postgres" if (db_url and db_url.startswith("postgres")) else None
    if backend == "postgres":
        _set_pg_rls_for_user(admin_user, domain)
    jm = JobManager(backend=backend, db_url=db_url)
    items = jm.list_job_attachments(job_id, limit=500)
    return [_normalize_attachment_item(i) for i in items]


# --- SLA policies ---
class SlaPolicyRequest(BaseModel):
    domain: str
    queue: str
    job_type: str
    max_queue_latency_seconds: int | None = None
    max_duration_seconds: int | None = None
    enabled: bool = True


@router.post("/jobs/sla/policy")
async def upsert_sla_policy_endpoint(
    req: SlaPolicyRequest,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> dict:
    admin_user = _enforce_domain_scope_unified(principal, req.domain)
    db_url = os.getenv("JOBS_DB_URL")
    backend = "postgres" if (db_url and db_url.startswith("postgres")) else None
    # When Postgres RLS is enabled, scope SLA policy mutations to the same domain
    # context as jobs to keep admin behavior consistent across tables.
    if backend == "postgres":
        _set_pg_rls_for_user(admin_user, req.domain)
    jm = JobManager(backend=backend, db_url=db_url)
    jm.upsert_sla_policy(
        domain=req.domain,
        queue=req.queue,
        job_type=req.job_type,
        max_queue_latency_seconds=req.max_queue_latency_seconds,
        max_duration_seconds=req.max_duration_seconds,
        enabled=req.enabled,
    )
    return {"ok": True}


@router.get("/jobs/sla/policies")
async def list_sla_policies_endpoint(
    domain: str | None = None,
    queue: str | None = None,
    job_type: str | None = None,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> list[dict]:
    admin_user = _enforce_domain_scope_unified(principal, domain)
    db_url = os.getenv("JOBS_DB_URL")
    backend = "postgres" if (db_url and db_url.startswith("postgres")) else None
    jm = JobManager(backend=backend, db_url=db_url)
    # Simple fetch via manager's internal connection
    conn = jm._connect()
    try:
        if jm.backend == "postgres":
            # Apply Postgres RLS context so listings respect the same domain policies as jobs.
            _set_pg_rls_for_user(admin_user, domain)
            with jm._pg_cursor(conn) as cur:
                where = ["1=1"]
                params: list = []
                if domain:
                    where.append("domain=%s")
                    params.append(domain)
                if queue:
                    where.append("queue=%s")
                    params.append(queue)
                if job_type:
                    where.append("job_type=%s")
                    params.append(job_type)
                cur.execute(
                    f"SELECT * FROM job_sla_policies WHERE {' AND '.join(where)} ORDER BY domain,queue,job_type",  # nosec B608
                    tuple(params),
                )
                rows = cur.fetchall() or []
                return [dict(r) for r in rows]
        else:
            where = ["1=1"]
            params2: list = []
            if domain:
                where.append("domain=?")
                params2.append(domain)
            if queue:
                where.append("queue=?")
                params2.append(queue)
            if job_type:
                where.append("job_type=?")
                params2.append(job_type)
            rows = conn.execute(
                f"SELECT * FROM job_sla_policies WHERE {' AND '.join(where)} ORDER BY domain,queue,job_type",  # nosec B608
                tuple(params2),
            ).fetchall() or []
            return [dict(r) for r in rows]
    finally:
        conn.close()


class SlaPolicyDeleteRequest(BaseModel):
    domain: str
    queue: str
    job_type: str


@router.delete("/jobs/sla/policy")
async def delete_sla_policy_endpoint(
    req: SlaPolicyDeleteRequest,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> dict:
    _enforce_domain_scope_unified(principal, req.domain)
    db_url = os.getenv("JOBS_DB_URL")
    backend = "postgres" if (db_url and db_url.startswith("postgres")) else None
    jm = JobManager(backend=backend, db_url=db_url)
    deleted = jm.delete_sla_policy(
        domain=req.domain,
        queue=req.queue,
        job_type=req.job_type,
    )
    if not deleted:
        raise HTTPException(status_code=404, detail="sla_policy_not_found")
    return {"ok": True}


@router.get("/jobs/sla/breaches")
async def list_sla_breaches_endpoint(
    domain: str | None = None,
    queue: str | None = None,
    job_type: str | None = None,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> list[dict]:
    """Return active jobs that currently breach their SLA policy thresholds.

    Compares processing time and wait time of active (queued/processing) jobs
    against the configured SLA policies.
    """
    admin_user = _enforce_domain_scope_unified(principal, domain)
    db_url = os.getenv("JOBS_DB_URL")
    backend = "postgres" if (db_url and db_url.startswith("postgres")) else None
    jm = JobManager(backend=backend, db_url=db_url)
    conn = jm._connect()
    try:
        # 1. Load enabled SLA policies
        policies: list[dict] = []
        if backend == "postgres":
            _set_pg_rls_for_user(admin_user, domain)
            with jm._pg_cursor(conn) as cur:
                cur.execute(
                    "SELECT * FROM job_sla_policies WHERE enabled=true ORDER BY domain,queue,job_type"
                )
                policies = [dict(r) for r in (cur.fetchall() or [])]
        else:
            policies = [
                dict(r) for r in conn.execute(
                    "SELECT * FROM job_sla_policies WHERE enabled=1 ORDER BY domain,queue,job_type"
                ).fetchall()
            ]

        if not policies:
            return []

        # Build lookup: (domain, queue, job_type) -> policy
        policy_lookup: dict[tuple[str, str, str], dict] = {}
        for pol in policies:
            key = (str(pol.get("domain", "")), str(pol.get("queue", "")), str(pol.get("job_type", "")))
            policy_lookup[key] = pol

        # 2. Load active jobs (queued + processing)
        active_jobs: list[dict] = []
        if backend == "postgres":
            with jm._pg_cursor(conn) as cur:
                where = ["status IN ('queued', 'processing')"]
                params: list = []
                if domain:
                    where.append("domain=%s")
                    params.append(domain)
                if queue:
                    where.append("queue=%s")
                    params.append(queue)
                if job_type:
                    where.append("job_type=%s")
                    params.append(job_type)
                cur.execute(
                    f"SELECT id, domain, queue, job_type, status, created_at, acquired_at, started_at FROM jobs WHERE {' AND '.join(where)} ORDER BY created_at",  # nosec B608
                    tuple(params),
                )
                active_jobs = [dict(r) for r in (cur.fetchall() or [])]
        else:
            where2 = ["status IN ('queued', 'processing')"]
            params2: list = []
            if domain:
                where2.append("domain=?")
                params2.append(domain)
            if queue:
                where2.append("queue=?")
                params2.append(queue)
            if job_type:
                where2.append("job_type=?")
                params2.append(job_type)
            active_jobs = [
                dict(r) for r in conn.execute(
                    f"SELECT id, domain, queue, job_type, status, created_at, acquired_at, started_at FROM jobs WHERE {' AND '.join(where2)} ORDER BY created_at",  # nosec B608
                    tuple(params2),
                ).fetchall()
            ]

        # 3. Check each active job against its policy
        now = datetime.utcnow()
        breaches: list[dict] = []
        for job in active_jobs:
            jd = str(job.get("domain", ""))
            jq = str(job.get("queue", ""))
            jjt = str(job.get("job_type", ""))
            pol = policy_lookup.get((jd, jq, jjt))
            if not pol:
                continue

            breach_kinds: list[str] = []
            breach_details: dict[str, Any] = {}

            # Check wait time (time in queue before being acquired)
            max_qlat = pol.get("max_queue_latency_seconds")
            if max_qlat is not None:
                created_at = _parse_dt_safe(job.get("created_at"))
                if created_at:
                    if job.get("status") == "queued":
                        wait_seconds = max(0.0, (now - created_at).total_seconds())
                    else:
                        acquired_at = _parse_dt_safe(job.get("acquired_at"))
                        wait_seconds = max(0.0, (acquired_at - created_at).total_seconds()) if acquired_at else 0.0
                    if wait_seconds > float(max_qlat):
                        breach_kinds.append("queue_latency")
                        breach_details["wait_seconds"] = round(wait_seconds, 1)
                        breach_details["max_wait_seconds"] = int(max_qlat)

            # Check processing duration
            max_dur = pol.get("max_duration_seconds")
            if max_dur is not None and job.get("status") == "processing":
                started_at = _parse_dt_safe(job.get("started_at")) or _parse_dt_safe(job.get("acquired_at"))
                if started_at:
                    processing_seconds = max(0.0, (now - started_at).total_seconds())
                    if processing_seconds > float(max_dur):
                        breach_kinds.append("duration")
                        breach_details["processing_seconds"] = round(processing_seconds, 1)
                        breach_details["max_processing_seconds"] = int(max_dur)

            if breach_kinds:
                breaches.append({
                    "job_id": job.get("id"),
                    "domain": jd,
                    "queue": jq,
                    "job_type": jjt,
                    "status": job.get("status"),
                    "breach_kinds": breach_kinds,
                    **breach_details,
                })

        return breaches
    finally:
        conn.close()


def _parse_dt_safe(value: Any) -> datetime | None:
    """Parse a datetime string safely, returning None on failure."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00").replace("+00:00", ""))
    except (ValueError, TypeError):
        return None


# --- Maintenance: Encryption key rotation ---
class CryptoRotateRequest(BaseModel):
    old_key_b64: str
    new_key_b64: str
    domain: str | None = None
    queue: str | None = None
    job_type: str | None = None
    fields: list[str] = Field(default_factory=lambda: ["payload", "result"])
    limit: int = 1000
    dry_run: bool = False


class CryptoRotateResponse(BaseModel):
    affected: int


@router.post("/jobs/crypto/rotate", response_model=CryptoRotateResponse)
async def rotate_crypto_endpoint(
    request: Request,
    body: CryptoRotateRequest,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> CryptoRotateResponse:
    admin_user = _enforce_domain_scope_unified(principal, body.domain)
    # Require confirmation for destructive changes
    if not body.dry_run:
        hdr = str(request.headers.get("x-confirm", "")).lower()
        if not _is_truthy(hdr):
            raise HTTPException(status_code=400, detail="Confirmation required: set X-Confirm: true")
    db_url = os.getenv("JOBS_DB_URL")
    backend = "postgres" if (db_url and db_url.startswith("postgres")) else None
    if backend == "postgres":
        _set_pg_rls_for_user(admin_user, body.domain)
    jm = JobManager(backend=backend, db_url=db_url)
    try:
        n = jm.rotate_encryption_keys(
            domain=body.domain,
            queue=body.queue,
            job_type=body.job_type,
            old_key_b64=body.old_key_b64,
            new_key_b64=body.new_key_b64,
            fields=body.fields,
            limit=int(body.limit),
            dry_run=bool(body.dry_run),
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve)) from ve
    return CryptoRotateResponse(affected=int(n))


class TTLSweepRequest(BaseModel):
    age_seconds: int | None = Field(default=None, ge=1, description="Cancel/fail queued jobs older than this many seconds (created_at)")
    runtime_seconds: int | None = Field(default=None, ge=1, description="Cancel/fail processing jobs running longer than this many seconds")
    action: str = Field(default="cancel", pattern="^(cancel|fail)$", description="Action to apply to matching jobs")
    domain: str | None = Field(default=None)
    queue: str | None = Field(default=None)
    job_type: str | None = Field(default=None)

    model_config = ConfigDict(json_schema_extra={
            "example": {
                "age_seconds": 86400,
                "runtime_seconds": 7200,
                "action": "cancel",
                "domain": "chatbooks",
                "queue": "default",
                "job_type": None,
            }
        })


# -------------------- Job Events Outbox (CDC) --------------------

class JobEvent(BaseModel):
    id: int
    job_id: int | None = None
    domain: str | None = None
    queue: str | None = None
    job_type: str | None = None
    event_type: str
    attrs: dict = Field(default_factory=dict)
    owner_user_id: str | None = None
    request_id: str | None = None
    trace_id: str | None = None
    created_at: str


@router.get("/jobs/events", response_model=list[JobEvent])
async def list_job_events(
    after_id: int = 0,
    limit: int = 200,
    domain: str | None = None,
    queue: str | None = None,
    job_type: str | None = None,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> list[JobEvent]:
    """Return job events from the append-only outbox with a cursor (after_id).

    Intended for reliable polling by dashboards or external sinks.
    """
    admin_user = _enforce_domain_scope_unified(principal, domain)
    db_url = os.getenv("JOBS_DB_URL")
    backend = "postgres" if (db_url and db_url.startswith("postgres")) else None
    if backend == "postgres":
        # Admin list; allow admin bypass with optional domain filter
        _set_pg_rls_for_user(admin_user, domain)
    jm = JobManager(backend=backend, db_url=db_url)
    conn = jm._connect()
    try:
        rows = []
        if jm.backend == "postgres":
            with jm._pg_cursor(conn) as cur:
                query = "SELECT id, job_id, domain, queue, job_type, event_type, attrs_json, owner_user_id, request_id, trace_id, created_at FROM job_events WHERE id > %s"
                params = [int(after_id)]
                if domain:
                    query += " AND domain = %s"
                    params.append(domain)
                if queue:
                    query += " AND queue = %s"
                    params.append(queue)
                if job_type:
                    query += " AND job_type = %s"
                    params.append(job_type)
                query += " ORDER BY id ASC LIMIT %s"
                params.append(int(min(1000, max(1, limit))))
                cur.execute(query, tuple(params))
                rows = cur.fetchall() or []
        else:
            query = "SELECT id, job_id, domain, queue, job_type, event_type, attrs_json, owner_user_id, request_id, trace_id, created_at FROM job_events WHERE id > ?"
            params = [int(after_id)]
            if domain:
                query += " AND domain = ?"
                params.append(domain)
            if queue:
                query += " AND queue = ?"
                params.append(queue)
            if job_type:
                query += " AND job_type = ?"
                params.append(job_type)
            query += " ORDER BY id ASC LIMIT ?"
            params.append(int(min(1000, max(1, limit))))
            rows = conn.execute(query, tuple(params)).fetchall() or []
        events: list[JobEvent] = []
        for r in rows:
            try:
                # r can be dict-row or tuple
                if isinstance(r, dict):
                    attrs = r.get("attrs_json")
                    try:
                        attrs_obj = _json.loads(attrs) if isinstance(attrs, str) else (attrs or {})
                    except _JOBS_ADMIN_NONCRITICAL_EXCEPTIONS:
                        attrs_obj = {}
                    events.append(JobEvent(
                        id=int(r.get("id")), job_id=(r.get("job_id")), domain=r.get("domain"), queue=r.get("queue"), job_type=r.get("job_type"),
                        event_type=str(r.get("event_type")), attrs=attrs_obj, owner_user_id=r.get("owner_user_id"), request_id=r.get("request_id"), trace_id=r.get("trace_id"), created_at=str(r.get("created_at"))
                    ))
                else:
                    attrs_val = r[6]
                    try:
                        attrs_obj = _json.loads(attrs_val) if isinstance(attrs_val, str) else (attrs_val or {})
                    except _JOBS_ADMIN_NONCRITICAL_EXCEPTIONS:
                        attrs_obj = {}
                    events.append(JobEvent(
                        id=int(r[0]), job_id=(r[1]), domain=r[2], queue=r[3], job_type=r[4], event_type=str(r[5]), attrs=attrs_obj, owner_user_id=r[7], request_id=r[8], trace_id=r[9], created_at=str(r[10])
                    ))
            except _JOBS_ADMIN_NONCRITICAL_EXCEPTIONS:
                continue
        return events
    finally:
        with contextlib.suppress(_JOBS_ADMIN_NONCRITICAL_EXCEPTIONS):
            conn.close()


@router.get(
    "/jobs/events/stream",
    response_class=StreamingResponse,
    responses={
        200: {
            "description": "Server-sent events stream of job outbox events.",
            "content": {
                "text/event-stream": {},
            },
        },
    },
)
async def stream_job_events(
    after_id: int = 0,
    domain: str | None = None,
    queue: str | None = None,
    job_type: str | None = None,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> StreamingResponse:
    """Server-Sent Events stream of job events from the outbox.

    This is a simple tailer that polls the outbox and emits events without loss.
    """
    admin_user = _enforce_domain_scope_unified(principal, domain)
    db_url = os.getenv("JOBS_DB_URL")
    backend = "postgres" if (db_url and db_url.startswith("postgres")) else None
    if backend == "postgres":
        _set_pg_rls_for_user(admin_user, domain)
    jm = JobManager(backend=backend, db_url=db_url)

    import time as _time

    from tldw_Server_API.app.core.Metrics.metrics_manager import (
        MetricDefinition,
        MetricType,
        get_metrics_registry,
    )
    from tldw_Server_API.app.core.Streaming.streams import SSEStream

    nonlocal_after_id = after_id  # keep compatibility with inner mutation
    poll_interval = float(os.getenv("JOBS_EVENTS_POLL_INTERVAL", "1.0") or "1.0")

    # Register a lightweight gauge for the last event time (epoch seconds)
    try:
        _reg = get_metrics_registry()
        _reg.register_metric(
            MetricDefinition(
                name="jobs_events_last_ts_seconds",
                type=MetricType.GAUGE,
                description="Epoch seconds of the last emitted job event",
                unit="s",
                labels=["component", "endpoint"],
            )
        )
    except _JOBS_ADMIN_NONCRITICAL_EXCEPTIONS:
        _reg = get_metrics_registry()

    # In test mode, bound the stream duration to avoid teardown hangs in CI/sandbox
    try:
        _test_mode = is_test_mode()
    except _JOBS_ADMIN_NONCRITICAL_EXCEPTIONS:
        _test_mode = False
    try:
        _max_s = float(os.getenv("JOBS_SSE_TEST_MAX_SECONDS", "1.0")) if _test_mode else None
    except _JOBS_ADMIN_NONCRITICAL_EXCEPTIONS:
        _max_s = 1.0 if _test_mode else None

    stream = SSEStream(
        heartbeat_interval_s=poll_interval,
        heartbeat_mode="data",
        max_duration_s=_max_s,
        labels={"component": "jobs", "endpoint": "jobs_events_sse"},
    )

    async def _producer() -> None:
        nonlocal nonlocal_after_id
        # Initial small event to prompt client streaming
        with contextlib.suppress(_JOBS_ADMIN_NONCRITICAL_EXCEPTIONS):
            await stream.send_event("ping", {})
        while True:
            # Terminate promptly if the stream has been closed (e.g., max_duration or client done)
            try:
                if getattr(stream, "_closed", False):
                    break
            except _JOBS_ADMIN_NONCRITICAL_EXCEPTIONS:
                pass
            conn = jm._connect()
            try:
                if jm.backend == "postgres":
                    with jm._pg_cursor(conn) as cur:
                        query = "SELECT id, event_type, attrs_json FROM job_events WHERE id > %s"
                        params: list[Any] = [int(nonlocal_after_id)]
                        if domain:
                            query += " AND domain = %s"
                            params.append(domain)
                        if queue:
                            query += " AND queue = %s"
                            params.append(queue)
                        if job_type:
                            query += " AND job_type = %s"
                            params.append(job_type)
                        query += " ORDER BY id ASC LIMIT 500"
                        cur.execute(query, tuple(params))
                        rows = cur.fetchall() or []
                else:
                    query = "SELECT id, event_type, attrs_json FROM job_events WHERE id > ?"
                    params2: list[Any] = [int(nonlocal_after_id)]
                    if domain:
                        query += " AND domain = ?"
                        params2.append(domain)
                    if queue:
                        query += " AND queue = ?"
                        params2.append(queue)
                    if job_type:
                        query += " AND job_type = ?"
                        params2.append(job_type)
                    query += " ORDER BY id ASC LIMIT 500"
                    rows = conn.execute(query, tuple(params2)).fetchall() or []
                if rows:
                    for r in rows:
                        if isinstance(r, dict):
                            eid = int(r.get("id"))
                            et = str(r.get("event_type"))
                            attrs = r.get("attrs_json")
                        else:
                            eid = int(r[0])
                            et = str(r[1])
                            attrs = r[2]
                        try:
                            attrs_obj = _json.loads(attrs) if isinstance(attrs, str) else (attrs or {})
                        except _JOBS_ADMIN_NONCRITICAL_EXCEPTIONS:
                            attrs_obj = {}
                        # Preserve SSE id line for clients using Last-Event-ID
                        await stream.send_event("job", {"event": et, "attrs": attrs_obj}, event_id=str(eid))
                        with contextlib.suppress(_JOBS_ADMIN_NONCRITICAL_EXCEPTIONS):
                            _reg.set_gauge(
                                "jobs_events_last_ts_seconds",
                                float(_time.time()),
                                {"component": "jobs", "endpoint": "jobs_events_sse"},
                            )
                        nonlocal_after_id = eid
                # If no rows, rely on heartbeat to keep connection alive
                await asyncio.sleep(poll_interval)
            except (asyncio.CancelledError, GeneratorExit):
                break
            except _JOBS_ADMIN_NONCRITICAL_EXCEPTIONS:
                # Swallow transient errors and continue after a short delay; heartbeat covers liveness
                await asyncio.sleep(poll_interval)
            finally:
                with contextlib.suppress(_JOBS_ADMIN_NONCRITICAL_EXCEPTIONS):
                    conn.close()

    async def _gen():
        prod_task = asyncio.create_task(_producer())
        try:
            async for ln in stream.iter_sse():
                yield ln
        except asyncio.CancelledError:
            # Client cancelled: cancel producer promptly and re-raise
            if not prod_task.done():
                with contextlib.suppress(_JOBS_ADMIN_NONCRITICAL_EXCEPTIONS):
                    prod_task.cancel()
                with contextlib.suppress(_JOBS_ADMIN_NONCRITICAL_EXCEPTIONS):
                    await prod_task
            raise
        else:
            # Normal shutdown: ensure producer completes without forced cancel
            if not prod_task.done():
                with contextlib.suppress(_JOBS_ADMIN_NONCRITICAL_EXCEPTIONS):
                    await prod_task
        finally:
            # Ensure producer task never leaks on unexpected exceptions
            if not prod_task.done():
                with contextlib.suppress(_JOBS_ADMIN_NONCRITICAL_EXCEPTIONS):
                    prod_task.cancel()
                try:
                    await prod_task
                except asyncio.CancelledError:
                    pass
                except _JOBS_ADMIN_NONCRITICAL_EXCEPTIONS:
                    # Swallow any cleanup-time errors to avoid propagating
                    pass

    # Advise proxies/servers not to buffer SSE
    sse_headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    return StreamingResponse(_gen(), media_type="text/event-stream", headers=sse_headers)



class TTLSweepResponse(BaseModel):
    affected: int


@router.post(
    "/jobs/ttl/sweep",
    response_model=TTLSweepResponse,
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "cancel": {
                            "summary": "Cancel expired queued/processing jobs (requires X-Confirm)",
                            "value": {
                                "age_seconds": 86400,
                                "runtime_seconds": 7200,
                                "action": "cancel",
                                "domain": "chatbooks",
                                "queue": "default"
                            },
                        },
                        "fail": {
                            "summary": "Fail expired jobs (requires X-Confirm)",
                            "value": {
                                "age_seconds": 604800,
                                "runtime_seconds": 14400,
                                "action": "fail",
                                "domain": "chatbooks"
                            },
                        },
                    }
                }
            }
        },
        "responses": {
            "200": {"content": {"application/json": {"example": {"affected": 10}}}},
            "400": {"description": "Missing X-Confirm header for destructive action"},
        },
    },
)
async def ttl_sweep_endpoint(
    request: Request,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> TTLSweepResponse:
    try:
        # Correlation IDs and diagnostics
        from tldw_Server_API.app.core.Logging.log_context import ensure_request_id, ensure_traceparent, get_ps_logger
        rid = ensure_request_id(request)
        ensure_traceparent(request)
        # Pre-parse raw to enforce RBAC and confirm header before validation
        try:
            raw = await request.json()
        except _JOBS_ADMIN_NONCRITICAL_EXCEPTIONS:
            raw = {}
        raw_domain = (raw or {}).get("domain")
        domain_val = str(raw_domain or "").strip() or None
        admin_user = _enforce_domain_scope_unified(principal, domain_val)
        # Confirm header for destructive action (check before model validation for consistent 400s)
        hdr = str(request.headers.get("x-confirm", "")).lower()
        if not _is_truthy(hdr):
            # Special-case: when domain-scoped RBAC is enforced and request is scoped
            # in TEST_MODE, allow a no-op (affected=0) response without destructive
            # changes. This preserves the guardrail while enabling RBAC-focused checks
            # in tests and non-production environments while keeping production
            # behavior (400 without X-Confirm) unchanged.
            domain_scoped = _is_truthy(os.getenv("JOBS_DOMAIN_SCOPED_RBAC"))
            forced = _is_truthy(os.getenv("JOBS_RBAC_FORCE"))
            is_test = is_test_mode()
            if is_test and domain_scoped and forced and (raw or {}).get("domain"):
                return TTLSweepResponse(affected=0)
            raise HTTPException(status_code=400, detail="Confirmation required: set X-Confirm: true")
        db_url = os.getenv("JOBS_DB_URL")
        backend = "postgres" if (db_url and db_url.startswith("postgres")) else None
        if backend == "postgres":
            _set_pg_rls_for_user(admin_user, domain_val)
        jm = JobManager(backend=backend, db_url=db_url)
        # Now validate the request model
        req = TTLSweepRequest(**(raw or {}))
        # Capture a single reference time to avoid boundary drift between age/runtime calculations
        try:
            from datetime import datetime
            from datetime import timezone as _tz
            ref_now = datetime.now(tz=_tz.utc)
        except _JOBS_ADMIN_NONCRITICAL_EXCEPTIONS:
            ref_now = None
        # Diagnostics before executing
        with contextlib.suppress(_JOBS_ADMIN_NONCRITICAL_EXCEPTIONS):
            get_ps_logger(request_id=rid, ps_component="endpoint", ps_job_kind="jobs").info(
                "TTL sweep request: action=%s domain=%s queue=%s job_type=%s age=%s runtime=%s backend=%s ref_now=%s",
                req.action, req.domain, req.queue, req.job_type, req.age_seconds, req.runtime_seconds, (backend or "sqlite"), str(ref_now) if ref_now else ""
            )
        affected = jm.apply_ttl_policies(
            age_seconds=req.age_seconds,
            runtime_seconds=req.runtime_seconds,
            action=req.action,
            domain=req.domain,
            queue=req.queue,
            job_type=req.job_type,
            reference_time=ref_now,
        )
        # Refresh gauges when fully scoped to avoid stale metrics
        try:
            if req.domain and req.queue and req.job_type:
                jm._update_gauges(domain=req.domain, queue=req.queue, job_type=req.job_type)
        except _JOBS_ADMIN_NONCRITICAL_EXCEPTIONS:
            pass
        # Diagnostics after executing
        with contextlib.suppress(_JOBS_ADMIN_NONCRITICAL_EXCEPTIONS):
            get_ps_logger(request_id=rid, ps_component="endpoint", ps_job_kind="jobs").info(
                "TTL sweep result: affected=%s action=%s domain=%s queue=%s job_type=%s",
                int(affected), req.action, req.domain, req.queue, req.job_type
            )
        return TTLSweepResponse(affected=int(affected))
    except HTTPException:
        raise
    except _JOBS_ADMIN_NONCRITICAL_EXCEPTIONS as e:
        raise HTTPException(status_code=500, detail="TTL sweep failed") from e


class IntegritySweepRequest(BaseModel):
    fix: bool = Field(default=False, description="When true, attempt to repair invalid states")
    domain: str | None = Field(default=None)
    queue: str | None = Field(default=None)
    job_type: str | None = Field(default=None)

    model_config = ConfigDict(json_schema_extra={
            "example": {
                "fix": False,
                "domain": "chatbooks",
                "queue": "default",
                "job_type": None,
            }
        })


class IntegritySweepResponse(BaseModel):
    non_processing_with_lease: int
    processing_expired: int
    fixed: int

    model_config = ConfigDict(json_schema_extra={
            "example": {
                "non_processing_with_lease": 3,
                "processing_expired": 1,
                "fixed": 2,
            }
        })


@router.post(
    "/jobs/integrity/sweep",
    response_model=IntegritySweepResponse,
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "dryRun": {
                            "summary": "Dry run integrity check (scoped)",
                            "value": {"fix": False, "domain": "chatbooks", "queue": "default"},
                        },
                        "fix": {
                            "summary": "Fix invalid states globally",
                            "value": {"fix": True},
                        },
                    }
                }
            }
        },
        "responses": {
            "200": {
                "content": {
                    "application/json": {
                        "example": {"non_processing_with_lease": 3, "processing_expired": 1, "fixed": 2}
                    }
                }
            }
        },
    },
)
async def integrity_sweep_endpoint(
    req: IntegritySweepRequest,
    principal: AuthPrincipal = Depends(get_auth_principal),
):
    try:
        admin_user = _enforce_domain_scope_unified(principal, req.domain)
        db_url = os.getenv("JOBS_DB_URL")
        backend = "postgres" if (db_url and db_url.startswith("postgres")) else None
        if backend == "postgres":
            _set_pg_rls_for_user(admin_user, req.domain)
        jm = JobManager(backend=backend, db_url=db_url)
        stats = jm.integrity_sweep(fix=req.fix, domain=req.domain, queue=req.queue, job_type=req.job_type)
        return IntegritySweepResponse(**stats)
    except HTTPException:
        raise
    except _JOBS_ADMIN_NONCRITICAL_EXCEPTIONS as e:
        raise HTTPException(status_code=500, detail="Integrity sweep failed") from e


class QueueStatsResponse(BaseModel):
    domain: str
    queue: str
    job_type: str
    queued: int
    scheduled: int
    processing: int
    quarantined: int

    model_config = ConfigDict(json_schema_extra={
            "example": {
                "domain": "chatbooks",
                "queue": "default",
                "job_type": "export",
                "queued": 3,
                "scheduled": 2,
                "processing": 1,
                "quarantined": 0,
            }
        })


@router.get("/jobs/stats", response_model=list[QueueStatsResponse])
async def get_jobs_stats(
    domain: str | None = None,
    queue: str | None = None,
    job_type: str | None = None,
    principal: AuthPrincipal = Depends(get_auth_principal),
):
    """Aggregate counts grouped by domain/queue/job_type for the WebUI."""
    try:
        admin_user = _enforce_domain_scope_unified(principal, domain)
        db_url = os.getenv("JOBS_DB_URL")
        backend = "postgres" if (db_url and db_url.startswith("postgres")) else None
        if backend == "postgres":
            # Correct RLS scoping for Postgres path
            _set_pg_rls_for_user(admin_user, domain)
        jm = JobManager(backend=backend, db_url=db_url)
        stats = jm.get_queue_stats(domain=domain, queue=queue, job_type=job_type)
        return [QueueStatsResponse(**s) for s in stats]
    except HTTPException:
        raise
    except _JOBS_ADMIN_NONCRITICAL_EXCEPTIONS as e:
        raise HTTPException(status_code=500, detail="Stats failed") from e


class ArchiveMetaResponse(BaseModel):
    job_id: int
    payload_present: bool
    result_present: bool
    payload_compressed_present: bool
    result_compressed_present: bool


@router.get("/jobs/archive/meta", response_model=ArchiveMetaResponse)
async def get_archive_meta(
    job_id: int,
    domain: str | None = None,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> ArchiveMetaResponse:
    """Return archive compression metadata for a given job id (if archived).

    When domain-scoped RBAC is enabled, this endpoint applies the same
    domain allowlist semantics and Postgres RLS context as other jobs admin
    surfaces. Global admins without domain RBAC enabled can inspect any
    archived job.
    """
    admin_user = _enforce_domain_scope_unified(principal, domain)
    db_url = os.getenv("JOBS_DB_URL")
    backend = "postgres" if (db_url and db_url.startswith("postgres")) else None
    jm = JobManager(backend=backend, db_url=db_url)
    conn = jm._connect()
    try:
        if jm.backend == "postgres":
            _set_pg_rls_for_user(admin_user, domain)
            with jm._pg_cursor(conn) as cur:
                cur.execute(
                    "SELECT payload, result, payload_compressed, result_compressed FROM jobs_archive WHERE id = %s",
                    (int(job_id),),
                )
                row = cur.fetchone()
        else:
            row = conn.execute(
                "SELECT payload, result, payload_compressed, result_compressed FROM jobs_archive WHERE id = ?",
                (int(job_id),),
            ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Archive row not found for job_id")
        # row can be dict or tuple
        def _get(ix_or_key):
            if isinstance(row, dict):
                return row.get(ix_or_key)
            return row[ix_or_key]
        payload_present = _get(0) is not None
        result_present = _get(1) is not None
        payload_compressed_present = _get(2) is not None
        result_compressed_present = _get(3) is not None
        return ArchiveMetaResponse(
            job_id=int(job_id),
            payload_present=bool(payload_present),
            result_present=bool(result_present),
            payload_compressed_present=bool(payload_compressed_present),
            result_compressed_present=bool(result_compressed_present),
        )
    finally:
        with contextlib.suppress(_JOBS_ADMIN_NONCRITICAL_EXCEPTIONS):
            conn.close()


class JobItem(BaseModel):
    id: int
    uuid: str | None = None
    domain: str
    queue: str
    job_type: str
    status: str
    priority: int | None = None
    retry_count: int | None = None
    max_retries: int | None = None
    available_at: str | None = None
    created_at: str | None = None
    acquired_at: str | None = None
    started_at: str | None = None
    leased_until: str | None = None
    completed_at: str | None = None


class JobDetailResponse(BaseModel):
    id: int
    uuid: str | None = None
    domain: str
    queue: str
    job_type: str
    status: str
    payload: Any | None = None
    result: Any | None = None
    archived: bool = False

    model_config = ConfigDict(extra="allow")


@router.get("/jobs/list", response_model=list[JobItem])
async def list_jobs_endpoint(
    domain: str | None = None,
    queue: str | None = None,
    status: str | None = None,
    owner_user_id: str | None = None,
    job_type: str | None = None,
    limit: int = 100,
    sort_by: str | None = None,
    sort_order: str | None = None,
    principal: AuthPrincipal = Depends(get_auth_principal),
):
    try:
        admin_user = _enforce_domain_scope_unified(principal, domain)
        db_url = os.getenv("JOBS_DB_URL")
        backend = "postgres" if (db_url and db_url.startswith("postgres")) else None
        if backend == "postgres":
            _set_pg_rls_for_user(admin_user, domain)
        jm = JobManager(backend=backend, db_url=db_url)
        rows = jm.list_jobs(
            domain=domain,
            queue=queue,
            status=status,
            owner_user_id=owner_user_id,
            job_type=job_type,
            limit=limit,
            sort_by=(sort_by or "created_at"),
            sort_order=(sort_order or "desc"),
        )
        items: list[JobItem] = []
        for r in rows:
            # Keep minimal fields for listing
            items.append(
                JobItem(
                    id=int(r.get("id")),
                    uuid=r.get("uuid"),
                    domain=str(r.get("domain")),
                    queue=str(r.get("queue")),
                    job_type=str(r.get("job_type")),
                    status=str(r.get("status")),
                    priority=r.get("priority"),
                    retry_count=r.get("retry_count"),
                    max_retries=r.get("max_retries"),
                    available_at=str(r.get("available_at")) if r.get("available_at") is not None else None,
                    created_at=str(r.get("created_at")) if r.get("created_at") is not None else None,
                    acquired_at=str(r.get("acquired_at")) if r.get("acquired_at") is not None else None,
                    started_at=str(r.get("started_at")) if r.get("started_at") is not None else None,
                    leased_until=str(r.get("leased_until")) if r.get("leased_until") is not None else None,
                    completed_at=str(r.get("completed_at")) if r.get("completed_at") is not None else None,
                )
            )
        return items
    except HTTPException:
        raise
    except _JOBS_ADMIN_NONCRITICAL_EXCEPTIONS as e:
        raise HTTPException(status_code=500, detail="List failed") from e


class StaleGroup(BaseModel):
    domain: str
    queue: str
    count: int


@router.get("/jobs/stale", response_model=list[StaleGroup])
async def stale_processing_endpoint(
    domain: str | None = None,
    queue: str | None = None,
    principal: AuthPrincipal = Depends(get_auth_principal),
):
    try:
        admin_user = _enforce_domain_scope_unified(principal, domain)
        # Use explicit backend/db_url selection for consistency with other admin endpoints
        db_url = os.getenv("JOBS_DB_URL")
        backend = "postgres" if (db_url and db_url.startswith("postgres")) else None
        if backend == "postgres":
            _set_pg_rls_for_user(admin_user, domain)
        jm = JobManager(backend=backend, db_url=db_url)
        conn = jm._connect()
        out: list[StaleGroup] = []
        try:
            if jm.backend == "postgres":
                with jm._pg_cursor(conn) as cur:
                    where = ["status='processing'", "(leased_until IS NULL OR leased_until <= NOW())"]
                    params: list = []
                    if domain:
                        where.append("domain = %s")
                        params.append(domain)
                    if queue:
                        where.append("queue = %s")
                        params.append(queue)
                    cur.execute(
                        f"SELECT domain, queue, COUNT(*) FROM jobs WHERE {' AND '.join(where)} GROUP BY domain, queue",  # nosec B608
                        tuple(params),
                    )
                    for (d, q, c) in cur.fetchall():
                        out.append(StaleGroup(domain=str(d), queue=str(q), count=int(c)))
            else:
                where = ["status='processing'", "(leased_until IS NULL OR leased_until <= DATETIME('now'))"]
                params2: list = []
                if domain:
                    where.append("domain = ?")
                    params2.append(domain)
                if queue:
                    where.append("queue = ?")
                    params2.append(queue)
                sql = f"SELECT domain, queue, COUNT(*) FROM jobs WHERE {' AND '.join(where)} GROUP BY domain, queue"  # nosec B608
                for (d, q, c) in conn.execute(sql, tuple(params2)).fetchall():
                    out.append(StaleGroup(domain=str(d), queue=str(q), count=int(c)))
        finally:
            try:
                conn.close()
            except _JOBS_ADMIN_NONCRITICAL_EXCEPTIONS:
                logger.opt(exception=True).debug("Failed to close connection in list_stale_groups")
        return out
    except HTTPException:
        raise
    except _JOBS_ADMIN_NONCRITICAL_EXCEPTIONS as e:
        raise HTTPException(status_code=500, detail="Stale groups failed") from e


@router.get("/jobs/{job_id}", response_model=JobDetailResponse)
async def get_job_detail(
    job_id: int,
    principal: AuthPrincipal = Depends(get_auth_principal),
    domain: str | None = None,
) -> JobDetailResponse:
    admin_user = _enforce_domain_scope_unified(principal, domain)
    db_url = os.getenv("JOBS_DB_URL")
    backend = "postgres" if (db_url and db_url.startswith("postgres")) else None
    if backend == "postgres":
        _set_pg_rls_for_user(admin_user, domain)
    jm = JobManager(backend=backend, db_url=db_url)
    job = jm.get_job_or_archived(job_id, domain=domain)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobDetailResponse(**job)


class BatchCancelRequest(BaseModel):
    domain: str
    queue: str | None = None
    job_type: str | None = None
    job_id: int | None = None
    dry_run: bool = False


class BatchCancelResponse(BaseModel):
    affected: int


@router.post("/jobs/batch/cancel", response_model=BatchCancelResponse)
async def batch_cancel_endpoint(
    req: BatchCancelRequest,
    request: Request,
    principal: AuthPrincipal = Depends(get_auth_principal),
):
    try:
        admin_user = _enforce_domain_scope_unified(principal, req.domain)
        # Require confirm header unless dry_run
        if not req.dry_run:
            hdr = str(request.headers.get("x-confirm", "")).lower()
            if not _is_truthy(hdr):
                raise HTTPException(status_code=400, detail="Confirmation required: set X-Confirm: true")
        db_url = os.getenv("JOBS_DB_URL")
        backend = "postgres" if (db_url and db_url.startswith("postgres")) else None
        if backend == "postgres":
            _set_pg_rls_for_user(admin_user, req.domain)
        jm = JobManager(backend=backend, db_url=db_url)
        conn = jm._connect()
        try:
            where = ["domain = %s"] if jm.backend == "postgres" else ["domain = ?"]
            params: list = [req.domain]
            if req.queue:
                where.append("queue = %s" if jm.backend == "postgres" else "queue = ?")
                params.append(req.queue)
            if req.job_type:
                where.append("job_type = %s" if jm.backend == "postgres" else "job_type = ?")
                params.append(req.job_type)
            if req.job_id is not None:
                where.append("id = %s" if jm.backend == "postgres" else "id = ?")
                params.append(int(req.job_id))
            # Allow cancelling queued or processing (processing will be terminally cancelled)
            if jm.backend == "postgres":
                with jm._pg_cursor(conn) as cur:
                    if req.dry_run:
                        cur.execute(
                            f"SELECT COUNT(*) FROM jobs WHERE ({' AND '.join(where)}) AND status IN ('queued','processing')",  # nosec B608
                            tuple(params),
                        )
                        c = cur.fetchone()
                        count = int(c.get("count") or 0) if isinstance(c, dict) else int(c[0] if c else 0)
                        return BatchCancelResponse(affected=count)
                    # Counters pre-measure per group
                    counters_enabled = env_flag_enabled("JOBS_COUNTERS_ENABLED")
                    grp_ready = []
                    grp_sched = []
                    grp_proc = []
                    if counters_enabled:
                        cur.execute(
                            (
                                f"SELECT domain, queue, job_type, COUNT(*) c FROM jobs WHERE ({' AND '.join(where)}) "  # nosec B608
                                "AND status='queued' AND (available_at IS NULL OR available_at <= NOW()) GROUP BY domain,queue,job_type"
                            ),
                            tuple(params),
                        )
                        grp_ready = cur.fetchall() or []
                        cur.execute(
                            (
                                f"SELECT domain, queue, job_type, COUNT(*) c FROM jobs WHERE ({' AND '.join(where)}) "  # nosec B608
                                "AND status='queued' AND (available_at IS NOT NULL AND available_at > NOW()) GROUP BY domain,queue,job_type"
                            ),
                            tuple(params),
                        )
                        grp_sched = cur.fetchall() or []
                        cur.execute(
                            f"SELECT domain, queue, job_type, COUNT(*) c FROM jobs WHERE ({' AND '.join(where)}) AND status='processing' GROUP BY domain,queue,job_type",  # nosec B608
                            tuple(params),
                        )
                        grp_proc = cur.fetchall() or []
                    # queued immediate cancel
                    cur.execute(
                        f"UPDATE jobs SET status='cancelled', cancelled_at = NOW(), cancellation_reason='batch_cancel' WHERE ({' AND '.join(where)}) AND status = 'queued'",  # nosec B608
                        tuple(params),
                    )
                    affected = cur.rowcount or 0
                    # processing terminal cancel
                    cur.execute(
                        f"UPDATE jobs SET status='cancelled', cancelled_at = NOW(), cancellation_reason='batch_cancel', leased_until = NULL WHERE ({' AND '.join(where)}) AND status = 'processing'",  # nosec B608
                        tuple(params),
                    )
                    affected += cur.rowcount or 0
                    # Adjust counters and refresh gauges per group
                    try:
                        if counters_enabled:
                            for r in grp_ready:
                                d = r["domain"] if isinstance(r, dict) else r[0]
                                q = r["queue"] if isinstance(r, dict) else r[1]
                                jt = r["job_type"] if isinstance(r, dict) else r[2]
                                c = int(r["c"] if isinstance(r, dict) else r[3])
                                cur.execute(
                                    "UPDATE job_counters SET ready_count = GREATEST(ready_count - %s, 0), updated_at = NOW() WHERE domain=%s AND queue=%s AND job_type=%s",
                                    (c, d, q, jt),
                                )
                            for r in grp_sched:
                                d = r["domain"] if isinstance(r, dict) else r[0]
                                q = r["queue"] if isinstance(r, dict) else r[1]
                                jt = r["job_type"] if isinstance(r, dict) else r[2]
                                c = int(r["c"] if isinstance(r, dict) else r[3])
                                cur.execute(
                                    "UPDATE job_counters SET scheduled_count = GREATEST(scheduled_count - %s, 0), updated_at = NOW() WHERE domain=%s AND queue=%s AND job_type=%s",
                                    (c, d, q, jt),
                                )
                            for r in grp_proc:
                                d = r["domain"] if isinstance(r, dict) else r[0]
                                q = r["queue"] if isinstance(r, dict) else r[1]
                                jt = r["job_type"] if isinstance(r, dict) else r[2]
                                c = int(r["c"] if isinstance(r, dict) else r[3])
                                cur.execute(
                                    "UPDATE job_counters SET processing_count = GREATEST(processing_count - %s, 0), updated_at = NOW() WHERE domain=%s AND queue=%s AND job_type=%s",
                                    (c, d, q, jt),
                                )
                    except _JOBS_ADMIN_NONCRITICAL_EXCEPTIONS:
                        pass
                    with contextlib.suppress(_JOBS_ADMIN_NONCRITICAL_EXCEPTIONS):
                        conn.commit()
                    try:
                        # If fully scoped, refresh gauges
                        if req.domain and req.queue and req.job_type:
                            jm._update_gauges(domain=req.domain, queue=req.queue, job_type=req.job_type)
                    except _JOBS_ADMIN_NONCRITICAL_EXCEPTIONS:
                        pass
                    return BatchCancelResponse(affected=int(affected))
            else:
                if req.dry_run:
                    cur = conn.execute(
                        f"SELECT COUNT(*) FROM jobs WHERE ({' AND '.join(where)}) AND status IN ('queued','processing')",  # nosec B608
                        tuple(params),
                    )
                    r = cur.fetchone()
                    return BatchCancelResponse(affected=int(r[0] if r else 0))
                # Counters pre-measure
                counters_enabled = env_flag_enabled("JOBS_COUNTERS_ENABLED")
                grp_ready2 = []
                grp_sched2 = []
                grp_proc2 = []
                if counters_enabled:
                    grp_ready2 = conn.execute(
                        (
                            f"SELECT domain, queue, job_type, COUNT(*) FROM jobs WHERE ({' AND '.join(where)}) "  # nosec B608
                            "AND status='queued' AND (available_at IS NULL OR available_at <= DATETIME('now')) GROUP BY domain,queue,job_type"
                        ),
                        tuple(params),
                    ).fetchall() or []
                    grp_sched2 = conn.execute(
                        (
                            f"SELECT domain, queue, job_type, COUNT(*) FROM jobs WHERE ({' AND '.join(where)}) "  # nosec B608
                            "AND status='queued' AND (available_at IS NOT NULL AND available_at > DATETIME('now')) GROUP BY domain,queue,job_type"
                        ),
                        tuple(params),
                    ).fetchall() or []
                    grp_proc2 = conn.execute(
                        f"SELECT domain, queue, job_type, COUNT(*) FROM jobs WHERE ({' AND '.join(where)}) AND status='processing' GROUP BY domain,queue,job_type",  # nosec B608
                        tuple(params),
                    ).fetchall() or []
                before = conn.total_changes or 0
                conn.execute(
                    f"UPDATE jobs SET status='cancelled', cancelled_at = DATETIME('now'), cancellation_reason='batch_cancel' WHERE ({' AND '.join(where)}) AND status = 'queued'",  # nosec B608
                    tuple(params),
                )
                mid = conn.total_changes or 0
                conn.execute(
                    f"UPDATE jobs SET status='cancelled', cancelled_at = DATETIME('now'), cancellation_reason='batch_cancel', leased_until = NULL WHERE ({' AND '.join(where)}) AND status = 'processing'",  # nosec B608
                    tuple(params),
                )
                after = conn.total_changes or 0
                affected = (mid - before) + (after - mid)
                # Adjust counters
                try:
                    if counters_enabled:
                        for d, q, jt, c in grp_ready2:
                            conn.execute(
                                "UPDATE job_counters SET ready_count = CASE WHEN (ready_count - ?) < 0 THEN 0 ELSE ready_count - ? END, updated_at = DATETIME('now') WHERE domain=? AND queue=? AND job_type=?",
                                (int(c), int(c), d, q, jt),
                            )
                        for d, q, jt, c in grp_sched2:
                            conn.execute(
                                "UPDATE job_counters SET scheduled_count = CASE WHEN (scheduled_count - ?) < 0 THEN 0 ELSE scheduled_count - ? END, updated_at = DATETIME('now') WHERE domain=? AND queue=? AND job_type=?",
                                (int(c), int(c), d, q, jt),
                            )
                        for d, q, jt, c in grp_proc2:
                            conn.execute(
                                "UPDATE job_counters SET processing_count = CASE WHEN (processing_count - ?) < 0 THEN 0 ELSE processing_count - ? END, updated_at = DATETIME('now') WHERE domain=? AND queue=? AND job_type=?",
                                (int(c), int(c), d, q, jt),
                            )
                except _JOBS_ADMIN_NONCRITICAL_EXCEPTIONS:
                    pass
                # Ensure changes are persisted for subsequent reads
                with contextlib.suppress(_JOBS_ADMIN_NONCRITICAL_EXCEPTIONS):
                    conn.commit()
                try:
                    if req.domain and req.queue and req.job_type:
                        jm._update_gauges(domain=req.domain, queue=req.queue, job_type=req.job_type)
                except _JOBS_ADMIN_NONCRITICAL_EXCEPTIONS:
                    pass
                return BatchCancelResponse(affected=int(affected))
        finally:
            with contextlib.suppress(_JOBS_ADMIN_NONCRITICAL_EXCEPTIONS):
                conn.close()
    except HTTPException:
        raise
    except _JOBS_ADMIN_NONCRITICAL_EXCEPTIONS as e:
        raise HTTPException(status_code=500, detail="Batch cancel failed") from e


class BatchRescheduleRequest(BaseModel):
    domain: str
    queue: str | None = None
    job_type: str | None = None
    delay_seconds: int = Field(ge=0, default=0)
    dry_run: bool = False


class BatchRescheduleResponse(BaseModel):
    affected: int


@router.post("/jobs/batch/reschedule", response_model=BatchRescheduleResponse)
async def batch_reschedule_endpoint(
    req: BatchRescheduleRequest,
    request: Request,
    principal: AuthPrincipal = Depends(get_auth_principal),
):
    try:
        admin_user = _enforce_domain_scope_unified(principal, req.domain)
        if not req.dry_run:
            hdr = str(request.headers.get("x-confirm", "")).lower()
            if not _is_truthy(hdr):
                raise HTTPException(status_code=400, detail="Confirmation required: set X-Confirm: true")
        db_url = os.getenv("JOBS_DB_URL")
        backend = "postgres" if (db_url and db_url.startswith("postgres")) else None
        if backend == "postgres":
            _set_pg_rls_for_user(admin_user, req.domain)
        jm = JobManager(backend=backend, db_url=db_url)
        conn = jm._connect()
        try:
            where = ["domain = %s", "status = 'queued'"] if jm.backend == "postgres" else ["domain = ?", "status = 'queued'"]
            params: list = [req.domain]
            if req.queue:
                where.append("queue = %s" if jm.backend == "postgres" else "queue = ?")
                params.append(req.queue)
            if req.job_type:
                where.append("job_type = %s" if jm.backend == "postgres" else "job_type = ?")
                params.append(req.job_type)
            if jm.backend == "postgres":
                with jm._pg_cursor(conn) as cur:
                    if req.dry_run:
                        cur.execute(
                            f"SELECT COUNT(*) FROM jobs WHERE {' AND '.join(where)}",  # nosec B608
                            tuple(params),
                        )
                        r = cur.fetchone()
                        return BatchRescheduleResponse(affected=int(r[0] if r else 0))
                    counters_enabled = env_flag_enabled("JOBS_COUNTERS_ENABLED")
                    grp_ready = []
                    if counters_enabled:
                        cur.execute(
                            (
                                f"SELECT domain, queue, job_type, COUNT(*) c FROM jobs WHERE {' AND '.join(where)} "  # nosec B608
                                "AND (available_at IS NULL OR available_at <= NOW()) GROUP BY domain,queue,job_type"
                            ),
                            tuple(params),
                        )
                        grp_ready = cur.fetchall() or []
                    cur.execute(
                        f"UPDATE jobs SET available_at = NOW() + (%s || ' seconds')::interval WHERE {' AND '.join(where)}",  # nosec B608
                        tuple([int(req.delay_seconds)] + params),
                    )
                    # Update counters: ready -> scheduled for affected
                    try:
                        if counters_enabled and grp_ready:
                            for r in grp_ready:
                                d = r["domain"] if isinstance(r, dict) else r[0]
                                q = r["queue"] if isinstance(r, dict) else r[1]
                                jt = r["job_type"] if isinstance(r, dict) else r[2]
                                c = int(r["c"] if isinstance(r, dict) else r[3])
                                cur.execute(
                                    "UPDATE job_counters SET ready_count = GREATEST(ready_count - %s, 0), scheduled_count = job_counters.scheduled_count + %s, updated_at = NOW() WHERE domain=%s AND queue=%s AND job_type=%s",
                                    (c, c, d, q, jt),
                                )
                    except _JOBS_ADMIN_NONCRITICAL_EXCEPTIONS:
                        pass
                    with contextlib.suppress(_JOBS_ADMIN_NONCRITICAL_EXCEPTIONS):
                        conn.commit()
                    try:
                        if req.domain and req.queue and req.job_type:
                            jm._update_gauges(domain=req.domain, queue=req.queue, job_type=req.job_type)
                    except _JOBS_ADMIN_NONCRITICAL_EXCEPTIONS:
                        pass
                    return BatchRescheduleResponse(affected=int(cur.rowcount or 0))
            else:
                if req.dry_run:
                    cur = conn.execute(
                        f"SELECT COUNT(*) FROM jobs WHERE {' AND '.join(where)}",  # nosec B608
                        tuple(params),
                    )
                    r = cur.fetchone()
                    return BatchRescheduleResponse(affected=int(r[0] if r else 0))
                counters_enabled = env_flag_enabled("JOBS_COUNTERS_ENABLED")
                grp_ready2 = []
                if counters_enabled:
                    grp_ready2 = conn.execute(
                        (
                            f"SELECT domain, queue, job_type, COUNT(*) FROM jobs WHERE {' AND '.join(where)} AND (available_at IS NULL OR available_at <= DATETIME('now')) GROUP BY domain,queue,job_type"  # nosec B608
                        ),
                        tuple(params),
                    ).fetchall() or []
                before = conn.total_changes or 0
                conn.execute(
                    f"UPDATE jobs SET available_at = DATETIME('now', ?) WHERE {' AND '.join(where)}",  # nosec B608
                    tuple([f"+{int(req.delay_seconds)} seconds"] + params),
                )
                after = conn.total_changes or 0
                affected = after - before
                # Update counters
                try:
                    if counters_enabled and grp_ready2:
                        for d, q, jt, c in grp_ready2:
                            conn.execute(
                                (
                                    "UPDATE job_counters SET ready_count = CASE WHEN (ready_count - ?) < 0 THEN 0 ELSE ready_count - ? END, "
                                    "scheduled_count = scheduled_count + ?, updated_at = DATETIME('now') WHERE domain=? AND queue=? AND job_type=?"
                                ),
                                (int(c), int(c), int(c), d, q, jt),
                            )
                except _JOBS_ADMIN_NONCRITICAL_EXCEPTIONS:
                    pass
                with contextlib.suppress(_JOBS_ADMIN_NONCRITICAL_EXCEPTIONS):
                    conn.commit()
                try:
                    if req.domain and req.queue and req.job_type:
                        jm._update_gauges(domain=req.domain, queue=req.queue, job_type=req.job_type)
                except _JOBS_ADMIN_NONCRITICAL_EXCEPTIONS:
                    pass
                return BatchRescheduleResponse(affected=int(affected))
        finally:
            with contextlib.suppress(_JOBS_ADMIN_NONCRITICAL_EXCEPTIONS):
                conn.close()
    except HTTPException:
        raise
    except _JOBS_ADMIN_NONCRITICAL_EXCEPTIONS as e:
        raise HTTPException(status_code=500, detail="Batch reschedule failed") from e


class BatchRequeueQuarantinedRequest(BaseModel):
    domain: str
    queue: str | None = None
    job_type: str | None = None
    job_id: int | None = None
    dry_run: bool = False

    model_config = ConfigDict(json_schema_extra={
            "example": {
                "domain": "chatbooks",
                "queue": "default",
                "job_type": "export",
                "dry_run": True
            }
        })


class BatchRequeueQuarantinedResponse(BaseModel):
    affected: int

    model_config = ConfigDict(json_schema_extra={"example": {"affected": 5}})


@router.post(
    "/jobs/batch/requeue_quarantined",
    operation_id="batch_requeue_quarantined",
    response_model=BatchRequeueQuarantinedResponse,
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "dryRun": {
                            "summary": "Dry run requeue for a scoped set",
                            "value": {"domain": "chatbooks", "queue": "default", "job_type": "export", "dry_run": True},
                        },
                        "requeue": {
                            "summary": "Requeue quarantined jobs (requires X-Confirm: true)",
                            "value": {"domain": "chatbooks", "queue": "default", "job_type": "export", "dry_run": False},
                        },
                    }
                }
            }
        },
        "responses": {
            "200": {"content": {"application/json": {"example": {"affected": 12}}}},
            "400": {"description": "Missing confirmation header for destructive action"},
        },
    },
)
@router.post(
    "/jobs/batch/requeue-quarantined",
    operation_id="batch_requeue_quarantined_alias",
    response_model=BatchRequeueQuarantinedResponse,
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "dryRun": {
                            "summary": "Dry run requeue for a scoped set",
                            "value": {"domain": "chatbooks", "queue": "default", "job_type": "export", "dry_run": True},
                        },
                        "requeue": {
                            "summary": "Requeue quarantined jobs (requires X-Confirm: true)",
                            "value": {"domain": "chatbooks", "queue": "default", "job_type": "export", "dry_run": False},
                        },
                    }
                }
            }
        },
        "responses": {
            "200": {"content": {"application/json": {"example": {"affected": 12}}}},
            "400": {"description": "Missing confirmation header for destructive action"},
        },
    },
)
async def batch_requeue_quarantined_endpoint(
    req: BatchRequeueQuarantinedRequest,
    request: Request,
    principal: AuthPrincipal = Depends(get_auth_principal),
):
    try:
        admin_user = _enforce_domain_scope_unified(principal, req.domain)
        if not req.dry_run:
            hdr = str(request.headers.get("x-confirm", "")).lower()
            if not _is_truthy(hdr):
                raise HTTPException(status_code=400, detail="Confirmation required: set X-Confirm: true")
        db_url = os.getenv("JOBS_DB_URL")
        backend = "postgres" if (db_url and db_url.startswith("postgres")) else None
        if backend == "postgres":
            _set_pg_rls_for_user(admin_user, req.domain)
        jm = JobManager(backend=backend, db_url=db_url)
        conn = jm._connect()
        try:
            if jm.backend == "postgres":
                where = ["domain = %s", "status = 'quarantined'"]
                params: list = [req.domain]
                if req.queue:
                    where.append("queue = %s")
                    params.append(req.queue)
                if req.job_type:
                    where.append("job_type = %s")
                    params.append(req.job_type)
                if req.job_id is not None:
                    where.append("id = %s")
                    params.append(int(req.job_id))
                with conn:
                    with jm._pg_cursor(conn) as cur:
                        if req.dry_run:
                            cur.execute(f"SELECT COUNT(*) AS c FROM jobs WHERE {' AND '.join(where)}", tuple(params))  # nosec B608
                            r = cur.fetchone()
                            count = int(r.get("c") or 0) if isinstance(r, dict) else int(r[0] if r else 0)
                            return BatchRequeueQuarantinedResponse(affected=count)
                        # Compute group counts to adjust counters post-update when enabled
                        counters_enabled = env_flag_enabled("JOBS_COUNTERS_ENABLED")
                        grp_rows: list = []
                        if counters_enabled:
                            cur.execute(
                                f"SELECT domain, queue, job_type, COUNT(*) c FROM jobs WHERE {' AND '.join(where)} GROUP BY domain, queue, job_type",  # nosec B608
                                tuple(params),
                            )
                            grp_rows = cur.fetchall() or []
                        cur.execute(
                            f"UPDATE jobs SET status='queued', failure_streak_count = 0, failure_streak_code = NULL, quarantined_at = NULL, available_at = NOW(), leased_until = NULL, worker_id = NULL, lease_id = NULL WHERE {' AND '.join(where)}",  # nosec B608
                            tuple(params),
                        )
                        affected = int(cur.rowcount or 0)
                        # Adjust job_counters: quarantined -> ready
                        try:
                            if counters_enabled and grp_rows:
                                for r in grp_rows:
                                    d = r["domain"] if isinstance(r, dict) else r[0]
                                    q = r["queue"] if isinstance(r, dict) else r[1]
                                    jt = r["job_type"] if isinstance(r, dict) else r[2]
                                    c = int(r["c"] if isinstance(r, dict) else r[3])
                                    cur.execute(
                                        (
                                            "INSERT INTO job_counters(domain,queue,job_type,ready_count,scheduled_count,processing_count,quarantined_count) VALUES(%s,%s,%s,0,0,0,0) "
                                            "ON CONFLICT(domain,queue,job_type) DO UPDATE SET ready_count = job_counters.ready_count + %s, quarantined_count = GREATEST(job_counters.quarantined_count - %s, 0), updated_at = NOW()"
                                        ),
                                        (d, q, jt, c, c),
                                    )
                        except _JOBS_ADMIN_NONCRITICAL_EXCEPTIONS:
                            pass
                        try:
                            # Refresh gauges for the scope (best-effort)
                            if req.domain and req.queue and req.job_type:
                                jm._update_gauges(domain=req.domain, queue=req.queue, job_type=req.job_type)
                        except _JOBS_ADMIN_NONCRITICAL_EXCEPTIONS:
                            pass
                        return BatchRequeueQuarantinedResponse(affected=affected)
            else:
                where = ["domain = ?", "status = 'quarantined'"]
                params2: list = [req.domain]
                if req.queue:
                    where.append("queue = ?")
                    params2.append(req.queue)
                if req.job_type:
                    where.append("job_type = ?")
                    params2.append(req.job_type)
                if req.job_id is not None:
                    where.append("id = ?")
                    params2.append(int(req.job_id))
                if req.dry_run:
                    cur = conn.execute(f"SELECT COUNT(*) FROM jobs WHERE {' AND '.join(where)}", tuple(params2))  # nosec B608
                    r = cur.fetchone()
                    return BatchRequeueQuarantinedResponse(affected=int(r[0] if r else 0))
                with conn:
                    # Measure groups for counters before update
                    counters_enabled = env_flag_enabled("JOBS_COUNTERS_ENABLED")
                    grp_rows2 = []
                    if counters_enabled:
                        grp_rows2 = conn.execute(
                            f"SELECT domain, queue, job_type, COUNT(*) FROM jobs WHERE {' AND '.join(where)} GROUP BY domain, queue, job_type",  # nosec B608
                            tuple(params2),
                        ).fetchall() or []
                    conn.execute(
                        f"UPDATE jobs SET status='queued', failure_streak_count = 0, failure_streak_code = NULL, quarantined_at = NULL, available_at = DATETIME('now'), leased_until = NULL, worker_id = NULL, lease_id = NULL WHERE {' AND '.join(where)}",  # nosec B608
                        tuple(params2),
                    )
                    affected2 = int(conn.total_changes or 0)
                    # Adjust job_counters
                    try:
                        if counters_enabled and grp_rows2:
                            for d, q, jt, c in grp_rows2:
                                conn.execute(
                                    (
                                        "INSERT INTO job_counters(domain,queue,job_type,ready_count,scheduled_count,processing_count,quarantined_count) VALUES(?,?,?, ?, 0,0,0) "
                                        "ON CONFLICT(domain,queue,job_type) DO UPDATE SET ready_count = ready_count + ?, quarantined_count = CASE WHEN (quarantined_count - ?) < 0 THEN 0 ELSE quarantined_count - ? END, updated_at = DATETIME('now')"
                                    ),
                                    (d, q, jt, int(c), int(c), int(c), int(c)),
                                )
                    except _JOBS_ADMIN_NONCRITICAL_EXCEPTIONS:
                        pass
                    try:
                        if req.domain and req.queue and req.job_type:
                            jm._update_gauges(domain=req.domain, queue=req.queue, job_type=req.job_type)
                    except _JOBS_ADMIN_NONCRITICAL_EXCEPTIONS:
                        pass
                    return BatchRequeueQuarantinedResponse(affected=affected2)
        finally:
            with contextlib.suppress(_JOBS_ADMIN_NONCRITICAL_EXCEPTIONS):
                conn.close()
    except HTTPException:
        raise
    except _JOBS_ADMIN_NONCRITICAL_EXCEPTIONS as e:
        raise HTTPException(status_code=500, detail="Batch requeue quarantined failed") from e
