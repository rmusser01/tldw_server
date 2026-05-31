"""
Prompt Studio Status/Health API

Provides lightweight observability for the Prompt Studio job queue,
including queue depth, processing counts, and lease health.
"""

import contextlib
import os
from typing import Any, Optional

from fastapi import APIRouter, Depends, Query
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.prompt_studio_deps import (
    get_prompt_studio_user,
)
from tldw_Server_API.app.api.v1.schemas.prompt_studio_base import StandardResponse
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Metrics.metrics_manager import get_metrics_registry
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.monitoring import prompt_studio_metrics

_PROMPT_STUDIO_DOMAIN = "prompt_studio"


def _get_jobs_manager() -> JobManager:
    db_url = (os.getenv("JOBS_DB_URL") or "").strip()
    if not db_url:
        return JobManager()
    backend = "postgres" if db_url.startswith("postgres") else None
    return JobManager(backend=backend, db_url=db_url)


def _get_prompt_studio_queue() -> str:
    queue = (os.getenv("PROMPT_STUDIO_JOBS_QUEUE") or "default").strip()
    return queue or "default"


def _build_job_filters(
    *,
    backend: str,
    domain: str,
    queue: str,
    owner_user_id: Optional[str],
) -> tuple[str, list[Any]]:
    token = "%s" if backend == "postgres" else "?"
    clauses: list[str] = [f"domain = {token}", f"queue = {token}"]
    params: list[Any] = [domain, queue]
    if owner_user_id is not None:
        clauses.append(f"owner_user_id = {token}")
        params.append(owner_user_id)
    return " AND ".join(clauses), params


def _fetch_all(jm: JobManager, sql: str, params: list[Any]) -> list[Any]:
    conn = jm._connect()
    try:
        if jm.backend == "postgres":
            with jm._pg_cursor(conn) as cur:
                cur.execute(sql, params)
                return list(cur.fetchall() or [])
        return list(conn.execute(sql, params).fetchall() or [])
    finally:
        with contextlib.suppress(Exception):
            conn.close()


def _fetch_one(jm: JobManager, sql: str, params: list[Any]) -> Optional[Any]:
    rows = _fetch_all(jm, sql, params)
    return rows[0] if rows else None


def _row_value(row: Any, key: str, index: int, default: Any = None) -> Any:
    if row is None:
        return default
    if isinstance(row, dict):
        return row.get(key, default)
    try:
        return row[index]
    except Exception:
        return default


def _get_by_status(
    jm: JobManager,
    *,
    domain: str,
    queue: str,
    owner_user_id: Optional[str],
) -> dict[str, int]:
    where_sql, params = _build_job_filters(
        backend=jm.backend,
        domain=domain,
        queue=queue,
        owner_user_id=owner_user_id,
    )
    sql = f"SELECT status, COUNT(*) AS c FROM jobs WHERE {where_sql} GROUP BY status"  # nosec B608
    rows = _fetch_all(jm, sql, params)
    counts: dict[str, int] = {}
    for row in rows:
        status = _row_value(row, "status", 0)
        count = _row_value(row, "c", 1, 0)
        if status:
            try:
                counts[str(status)] = int(count or 0)
            except Exception:
                counts[str(status)] = 0
    return counts


def _get_by_type_and_status(
    jm: JobManager,
    *,
    domain: str,
    queue: str,
    owner_user_id: Optional[str],
) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    where_sql, params = _build_job_filters(
        backend=jm.backend,
        domain=domain,
        queue=queue,
        owner_user_id=owner_user_id,
    )
    sql = """
        SELECT job_type, status, COUNT(*) AS c
        FROM jobs
        WHERE {where_sql}
        GROUP BY job_type, status
    """.format_map(locals())  # nosec B608
    rows = _fetch_all(jm, sql, params)
    totals: dict[str, int] = {}
    queued: dict[str, int] = {}
    processing: dict[str, int] = {}
    for row in rows:
        job_type = _row_value(row, "job_type", 0)
        status = _row_value(row, "status", 1)
        count = _row_value(row, "c", 2, 0)
        if not job_type:
            continue
        job_type_str = str(job_type)
        count_int = int(count or 0)
        totals[job_type_str] = totals.get(job_type_str, 0) + count_int
        if status == "queued":
            queued[job_type_str] = queued.get(job_type_str, 0) + count_int
        if status == "processing":
            processing[job_type_str] = processing.get(job_type_str, 0) + count_int
    return totals, queued, processing


def _get_avg_processing_time_seconds(
    jm: JobManager,
    *,
    domain: str,
    queue: str,
    owner_user_id: Optional[str],
) -> float:
    where_sql, params = _build_job_filters(
        backend=jm.backend,
        domain=domain,
        queue=queue,
        owner_user_id=owner_user_id,
    )
    if jm.backend == "postgres":
        sql = (
            "SELECT AVG(EXTRACT(EPOCH FROM (completed_at - started_at))) AS avg_seconds "  # nosec B608
            f"FROM jobs WHERE {where_sql} AND status = 'completed' AND started_at IS NOT NULL AND completed_at IS NOT NULL"
        )
    else:
        sql = (
            "SELECT AVG((julianday(completed_at) - julianday(started_at)) * 86400.0) AS avg_seconds "  # nosec B608
            f"FROM jobs WHERE {where_sql} AND status = 'completed' AND started_at IS NOT NULL AND completed_at IS NOT NULL"
        )
    row = _fetch_one(jm, sql, params)
    value = _row_value(row, "avg_seconds", 0)
    try:
        return float(value) if value is not None else 0.0
    except Exception:
        return 0.0


def _get_success_rate(
    jm: JobManager,
    *,
    domain: str,
    queue: str,
    owner_user_id: Optional[str],
) -> float:
    where_sql, params = _build_job_filters(
        backend=jm.backend,
        domain=domain,
        queue=queue,
        owner_user_id=owner_user_id,
    )
    if jm.backend == "postgres":
        sql = (
            "SELECT "  # nosec B608
            "COUNT(*) FILTER (WHERE status = 'completed') * 100.0 / "
            "NULLIF(COUNT(*) FILTER (WHERE status IN ('completed', 'failed')), 0) AS success_rate "
            f"FROM jobs WHERE {where_sql} AND status IN ('completed', 'failed')"
        )
    else:
        sql = (
            "SELECT "  # nosec B608
            "SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) * 100.0 / "
            "NULLIF(SUM(CASE WHEN status IN ('completed', 'failed') THEN 1 ELSE 0 END), 0) AS success_rate "
            f"FROM jobs WHERE {where_sql} AND status IN ('completed', 'failed')"
        )
    row = _fetch_one(jm, sql, params)
    value = _row_value(row, "success_rate", 0)
    try:
        return float(value) if value is not None else 0.0
    except Exception:
        return 0.0


def _get_lease_stats(
    jm: JobManager,
    *,
    domain: str,
    queue: str,
    owner_user_id: Optional[str],
    warn_seconds: int,
) -> dict[str, int]:
    where_sql, params = _build_job_filters(
        backend=jm.backend,
        domain=domain,
        queue=queue,
        owner_user_id=owner_user_id,
    )
    warn_seconds = max(1, min(3600, int(warn_seconds)))
    if jm.backend == "postgres":
        sql = (
            "SELECT "  # nosec B608
            "COUNT(*) FILTER (WHERE status = 'processing' AND leased_until IS NOT NULL AND leased_until > NOW()) AS active, "
            "COUNT(*) FILTER (WHERE status = 'processing' AND leased_until IS NOT NULL AND leased_until > NOW() "
            f"AND leased_until <= NOW() + INTERVAL '{warn_seconds} seconds') AS expiring_soon, "
            "COUNT(*) FILTER (WHERE status = 'processing' AND (leased_until IS NULL OR leased_until <= NOW())) AS stale_processing "
            f"FROM jobs WHERE {where_sql}"
        )
    else:
        sql = (
            "SELECT "  # nosec B608
            "SUM(CASE WHEN status = 'processing' AND leased_until IS NOT NULL AND leased_until > DATETIME('now') THEN 1 ELSE 0 END) AS active, "
            "SUM(CASE WHEN status = 'processing' AND leased_until IS NOT NULL AND leased_until > DATETIME('now') "
            f"AND leased_until <= DATETIME('now', '+{warn_seconds} seconds') THEN 1 ELSE 0 END) AS expiring_soon, "
            "SUM(CASE WHEN status = 'processing' AND (leased_until IS NULL OR leased_until <= DATETIME('now')) THEN 1 ELSE 0 END) AS stale_processing "
            f"FROM jobs WHERE {where_sql}"
        )
    row = _fetch_one(jm, sql, params)
    return {
        "active": int(_row_value(row, "active", 0, 0) or 0),
        "expiring_soon": int(_row_value(row, "expiring_soon", 1, 0) or 0),
        "stale_processing": int(_row_value(row, "stale_processing", 2, 0) or 0),
    }

router = APIRouter(
    prefix="/api/v1/prompt-studio/status",
    tags=["prompt-studio"],
)


@router.get("", response_model=StandardResponse, openapi_extra={
    "responses": {
        "200": {
            "description": "Prompt Studio queue health and status",
            "content": {
                "application/json": {
                    "examples": {
                        "ok": {
                            "summary": "Queue health",
                            "value": {
                                "success": True,
                                "data": {
                                    "queue_depth": 0,
                                    "processing": 0,
                                    "leases": {"active": 0, "expiring_soon": 0, "stale_processing": 0},
                                    "by_status": {"queued": 0, "processing": 0},
                                    "by_type": {"optimization": 0},
                                    "avg_processing_time_seconds": 0,
                                    "success_rate": 100.0
                                }
                            }
                        }
                    }
                }
            }
        }
    }
})
async def get_prompt_studio_status(
    warn_seconds: int = Query(30, ge=1, le=3600, description="Threshold for expiring leases"),
    user_context: dict = Depends(get_prompt_studio_user),
) -> StandardResponse:
    """Return queue depth, processing count, and lease health stats."""
    try:
        jm = _get_jobs_manager()
        owner_user_id = user_context.get("user_id")
        owner_user_id = str(owner_user_id) if owner_user_id is not None else None
        queue = _get_prompt_studio_queue()
        JobManager.set_rls_context(
            is_admin=bool(user_context.get("is_admin", False)),
            domain_allowlist=_PROMPT_STUDIO_DOMAIN,
            owner_user_id=owner_user_id,
        )
        try:
            by_status = _get_by_status(
                jm,
                domain=_PROMPT_STUDIO_DOMAIN,
                queue=queue,
                owner_user_id=owner_user_id,
            )
            by_type, queued_by_type, processing_by_type = _get_by_type_and_status(
                jm,
                domain=_PROMPT_STUDIO_DOMAIN,
                queue=queue,
                owner_user_id=owner_user_id,
            )
            avg_processing_time_seconds = _get_avg_processing_time_seconds(
                jm,
                domain=_PROMPT_STUDIO_DOMAIN,
                queue=queue,
                owner_user_id=owner_user_id,
            )
            success_rate = _get_success_rate(
                jm,
                domain=_PROMPT_STUDIO_DOMAIN,
                queue=queue,
                owner_user_id=owner_user_id,
            )
            leases = _get_lease_stats(
                jm,
                domain=_PROMPT_STUDIO_DOMAIN,
                queue=queue,
                owner_user_id=owner_user_id,
                warn_seconds=warn_seconds,
            )
        finally:
            JobManager.clear_rls_context()

        data = {
            "queue_depth": int(by_status.get("queued", 0) or 0),
            "processing": int(by_status.get("processing", 0) or 0),
            "leases": leases,
            "by_status": by_status,
            "by_type": by_type,
            "avg_processing_time_seconds": avg_processing_time_seconds,
            "success_rate": success_rate,
        }
        # Prometheus hook: export gauges for queue/lease metrics
        try:
            backend_label = jm.backend or "unknown"
            reg = get_metrics_registry()
            reg.set_gauge("prompt_studio_queue_depth", float(data["queue_depth"]), labels={"backend": backend_label})
            reg.set_gauge("prompt_studio_processing", float(data["processing"]), labels={"backend": backend_label})
            reg.set_gauge("prompt_studio_leases_active", float(leases.get("active", 0)), labels={"backend": backend_label})
            reg.set_gauge("prompt_studio_leases_expiring_soon", float(leases.get("expiring_soon", 0)), labels={"backend": backend_label})
            reg.set_gauge("prompt_studio_leases_stale_processing", float(leases.get("stale_processing", 0)), labels={"backend": backend_label})
            # Periodic refresh of per-type gauges (queued/processing/backlog) based on current DB counts
            try:
                for jt in by_type:
                    q = int(queued_by_type.get(jt, 0))
                    p = int(processing_by_type.get(jt, 0))
                    prompt_studio_metrics.update_job_queue_size(jt, q)
                    prompt_studio_metrics.metrics_manager.set_gauge(
                        "jobs.processing", float(p), labels={"job_type": jt}
                    )
                    backlog = max(0, q - p)
                    prompt_studio_metrics.metrics_manager.set_gauge(
                        "jobs.backlog", float(backlog), labels={"job_type": jt}
                    )
                # Aggregate stale processing value
                prompt_studio_metrics.metrics_manager.set_gauge(
                    "jobs.stale_processing",
                    float(leases.get("stale_processing", 0)),
                )
            except Exception:
                logger.debug("Failed to refresh per-type gauges")
        except Exception:
            logger.debug("Failed to set Prompt Studio gauges")

        return StandardResponse(success=True, data=data)
    except Exception:  # noqa: BLE001
        logger.error("Failed to compute Prompt Studio status")
        return StandardResponse(success=False, error="Failed to compute Prompt Studio status")
