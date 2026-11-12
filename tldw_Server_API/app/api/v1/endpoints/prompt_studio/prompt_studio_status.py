"""
Prompt Studio Status/Health API

Provides lightweight observability for the Prompt Studio job queue,
including queue depth, processing counts, and lease health.
"""

import os

from fastapi import APIRouter, Depends, Query
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.prompt_studio_deps import (
    get_prompt_studio_user,
)
from tldw_Server_API.app.api.v1.schemas.prompt_studio_base import StandardResponse
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.queue_stats import (
    get_avg_processing_time_seconds as _get_avg_processing_time_seconds,
    get_by_status as _get_by_status,
    get_by_type_and_status as _get_by_type_and_status,
    get_lease_stats as _get_lease_stats,
    get_success_rate as _get_success_rate,
)
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
