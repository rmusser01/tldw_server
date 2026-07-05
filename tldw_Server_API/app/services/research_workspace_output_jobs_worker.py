"""Jobs worker for Research Workspace media output requests."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any

from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user_id
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.DB_Management.media_db.api import managed_media_database
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.worker_sdk import WorkerConfig, WorkerSDK
from tldw_Server_API.app.core.Jobs.worker_utils import coerce_int, jobs_manager_from_env
from tldw_Server_API.app.core.Research_Workspace.output_jobs import (
    RESEARCH_WORKSPACE_OUTPUT_JOB_DOMAIN,
    RESEARCH_WORKSPACE_OUTPUT_JOB_TYPE,
    ResearchWorkspaceOutputJobError,
    normalize_research_workspace_output_payload,
    process_research_workspace_output_payload,
    research_workspace_output_jobs_queue,
)


@dataclass
class _ProgressState:
    percent: float | None = None
    message: str | None = None


def _build_worker_config(*, worker_id: str, queue: str) -> WorkerConfig:
    return WorkerConfig(
        domain=RESEARCH_WORKSPACE_OUTPUT_JOB_DOMAIN,
        queue=queue,
        worker_id=worker_id,
        lease_seconds=coerce_int(os.getenv("RESEARCH_WORKSPACE_OUTPUT_JOBS_LEASE_SECONDS"), 180),
        renew_jitter_seconds=coerce_int(os.getenv("RESEARCH_WORKSPACE_OUTPUT_JOBS_RENEW_JITTER_SECONDS"), 5),
        renew_threshold_seconds=coerce_int(os.getenv("RESEARCH_WORKSPACE_OUTPUT_JOBS_RENEW_THRESHOLD_SECONDS"), 20),
        backoff_base_seconds=coerce_int(os.getenv("RESEARCH_WORKSPACE_OUTPUT_JOBS_BACKOFF_BASE_SECONDS"), 2),
        backoff_max_seconds=coerce_int(os.getenv("RESEARCH_WORKSPACE_OUTPUT_JOBS_BACKOFF_MAX_SECONDS"), 30),
        retry_on_exception=True,
        retry_backoff_seconds=coerce_int(os.getenv("RESEARCH_WORKSPACE_OUTPUT_JOBS_RETRY_BACKOFF_SECONDS"), 10),
    )


async def process_research_workspace_output_job(
    job: dict[str, Any],
    *,
    job_manager: JobManager,
    worker_id: str = "research-workspace-output-worker",
    progress: _ProgressState | None = None,
) -> dict[str, Any]:
    payload = normalize_research_workspace_output_payload(job.get("payload"))
    if str(job.get("job_type") or "").lower() != RESEARCH_WORKSPACE_OUTPUT_JOB_TYPE:
        raise ResearchWorkspaceOutputJobError("unsupported_job_type", retryable=False)

    user_id = resolve_research_workspace_output_job_user_id(job, payload)
    workspace_db = await open_research_workspace_output_notes_db(user_id)
    try:
        with managed_media_database(
            "research_workspace_output_worker",
            db_path=str(DatabasePaths.get_media_db_path(user_id)),
            initialize=False,
        ) as media_db:
            return await process_research_workspace_output_payload(
                job=job,
                payload=payload,
                workspace_db=workspace_db,
                media_db=media_db,
                user_id=user_id,
                job_manager=job_manager,
                progress=progress,
            )
    finally:
        close_research_workspace_output_notes_db(workspace_db)


def resolve_research_workspace_output_job_user_id(job: dict[str, Any], payload: dict[str, Any]) -> int:
    owner = payload.get("user_id") or job.get("owner_user_id")
    try:
        user_id = int(owner)
    except (TypeError, ValueError) as exc:
        raise ResearchWorkspaceOutputJobError("missing_owner_user_id", retryable=False) from exc
    if user_id <= 0:
        raise ResearchWorkspaceOutputJobError("missing_owner_user_id", retryable=False)
    return user_id


async def open_research_workspace_output_notes_db(user_id: int) -> CharactersRAGDB:
    return await get_chacha_db_for_user_id(
        user_id,
        client_id=f"research-workspace-output-worker-{user_id}",
    )


def close_research_workspace_output_notes_db(db: Any) -> None:
    if hasattr(db, "release_context_connection"):
        db.release_context_connection()
    elif hasattr(db, "close_connection"):
        db.close_connection()


async def run_research_workspace_output_jobs_worker(stop_event: asyncio.Event | None = None) -> None:
    worker_id = (os.getenv("RESEARCH_WORKSPACE_OUTPUT_JOBS_WORKER_ID") or "research-workspace-output-worker").strip()
    jm = jobs_manager_from_env()
    sdk = WorkerSDK(
        jm,
        _build_worker_config(
            worker_id=worker_id,
            queue=research_workspace_output_jobs_queue(),
        ),
    )
    progress = _ProgressState()

    async def _handle_job(job: dict[str, Any]) -> dict[str, Any]:
        progress.percent = None
        progress.message = None
        return await process_research_workspace_output_job(
            job,
            job_manager=jm,
            worker_id=worker_id,
            progress=progress,
        )

    def _progress_cb() -> dict[str, Any]:
        update: dict[str, Any] = {}
        if progress.percent is not None:
            update["progress_percent"] = progress.percent
        if progress.message:
            update["progress_message"] = progress.message
        return update

    async def _watch_stop() -> None:
        if stop_event is None:
            return
        await stop_event.wait()
        sdk.stop()

    logger.info("Starting Research Workspace Output Jobs worker")
    stop_task = asyncio.create_task(_watch_stop())
    try:
        await sdk.run(
            handler=_handle_job,
            progress_cb=_progress_cb,
            job_type=RESEARCH_WORKSPACE_OUTPUT_JOB_TYPE,
        )
    finally:
        stop_task.cancel()
