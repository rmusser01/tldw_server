"""Shared Jobs helpers for workspace source lifecycle tracking."""

from __future__ import annotations

from typing import Any

from loguru import logger

WORKSPACE_SOURCE_JOB_DOMAIN = "media_ingest"
WORKSPACE_SOURCE_JOB_QUEUE = "default"
WORKSPACE_SOURCE_JOB_TYPE = "workspace_source_ingest"
WORKSPACE_SOURCE_JOB_STAGES = ["ingestion", "extraction", "chunking", "indexing"]


def enqueue_workspace_source_ingest_job(
    *,
    jm: Any | None,
    owner_user_id: int | str,
    workspace_id: str,
    src: dict[str, Any],
) -> None:
    """Submit a user-visible lifecycle job after the workspace source row exists."""
    if jm is None:
        return
    source_id = str(src["id"])
    media_id = int(src["media_id"])
    payload = {
        "workspace_id": workspace_id,
        "workspace_source_id": source_id,
        "source_id": source_id,
        "media_id": media_id,
        "source_type": str(src["source_type"]),
        "title": str(src["title"]),
        "url": src.get("url"),
        "requested_stages": WORKSPACE_SOURCE_JOB_STAGES,
    }
    try:
        jm.create_job(
            domain=WORKSPACE_SOURCE_JOB_DOMAIN,
            queue=WORKSPACE_SOURCE_JOB_QUEUE,
            job_type=WORKSPACE_SOURCE_JOB_TYPE,
            payload=payload,
            owner_user_id=str(owner_user_id),
            idempotency_key=f"workspace-source:{workspace_id}:{source_id}:{media_id}",
            max_retries=3,
        )
    except Exception as exc:
        logger.warning(
            "Workspace source ingest job enqueue failed for workspace={} source={}: {}",
            workspace_id,
            source_id,
            exc,
        )
