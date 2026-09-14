"""Bounded optional Jobs enrichment for workspace source status reads."""
from __future__ import annotations

from typing import Any

from loguru import logger

from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Workspaces.source_jobs import (
    WORKSPACE_SOURCE_JOB_DOMAIN,
    WORKSPACE_SOURCE_JOB_QUEUE,
    WORKSPACE_SOURCE_JOB_TYPE,
)


def _dedupe_jobs_by_identity(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[Any] = set()
    for job in jobs:
        key = job.get("id") or job.get("uuid")
        if key is None:
            key = (
                job.get("domain"),
                job.get("queue"),
                job.get("job_type"),
                job.get("created_at"),
                str(job.get("payload") or ""),
            )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(job)
    return deduped


def _safe_list_jobs(job_manager: JobManager, **kwargs: Any) -> list[dict[str, Any]]:
    try:
        return job_manager.list_jobs(**kwargs)
    except Exception as exc:  # noqa: BLE001 - optional enrichment must fail open
        logger.warning("Workspace job list failed for filters {}: {}", kwargs, exc)
        return []


def list_recent_workspace_source_ingest_jobs(
    job_manager: JobManager | None,
    *,
    owner_user_id: int | str,
) -> list[dict[str, Any]]:
    """Return both current and legacy source-ingest Jobs, failing open."""
    if job_manager is None:
        return []
    workspace_source_jobs = _safe_list_jobs(
        job_manager,
        domain=WORKSPACE_SOURCE_JOB_DOMAIN,
        queue=WORKSPACE_SOURCE_JOB_QUEUE,
        owner_user_id=str(owner_user_id),
        job_type=WORKSPACE_SOURCE_JOB_TYPE,
        limit=500,
        sort_by="created_at",
        sort_order="desc",
    )
    legacy_media_jobs = _safe_list_jobs(
        job_manager,
        domain=WORKSPACE_SOURCE_JOB_DOMAIN,
        owner_user_id=str(owner_user_id),
        limit=500,
        sort_by="created_at",
        sort_order="desc",
    )
    return _dedupe_jobs_by_identity(workspace_source_jobs + legacy_media_jobs)
