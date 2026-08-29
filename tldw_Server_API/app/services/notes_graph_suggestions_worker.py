"""Jobs worker service for Notes graph suggestion generation."""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import (
    get_chacha_db_for_user_id,
)
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.worker_sdk import WorkerConfig, WorkerSDK
from tldw_Server_API.app.core.Notes_Graph.suggestion_jobs import (
    JOB_DOMAIN,
    JOB_QUEUE,
    JOB_TYPE,
    SuggestionPublisher,
)
from tldw_Server_API.app.core.Notes_Graph.suggestion_provider import (
    resolve_generation_capability,
)
from tldw_Server_API.app.core.Notes_Graph.suggestion_service import (
    SuggestionWorker,
    SuggestionWorkerCancelled,
)


def build_worker_config(*, worker_id: str) -> WorkerConfig:
    return WorkerConfig(
        domain=JOB_DOMAIN,
        queue=JOB_QUEUE,
        worker_id=worker_id,
        lease_seconds=int(os.getenv("NOTES_GRAPH_SUGGESTIONS_LEASE_SECONDS", "180") or "180"),
        renew_threshold_seconds=15,
        renew_jitter_seconds=0,
        retry_on_exception=False,
        bind_completion_token=True,
    )


async def _cancellation_requested(job: dict[str, Any], *, jobs: JobManager) -> bool:
    current = await asyncio.to_thread(
        jobs.get_job_or_archived_by_uuid,
        str(job.get("uuid") or ""),
        domain=JOB_DOMAIN,
        owner_user_id=str(job.get("owner_user_id") or ""),
    )
    return bool(current and (current.get("status") == "cancelled" or current.get("cancel_requested_at")))


async def _open_owner_database(owner_user_id: str) -> Any:
    try:
        user_id = int(owner_user_id)
    except (TypeError, ValueError) as exc:
        raise ValueError("notes_graph_job_owner_invalid") from exc
    return await get_chacha_db_for_user_id(
        user_id,
        client_id=str(user_id),
    )


def _close_database(db: Any) -> None:
    if hasattr(db, "release_context_connection"):
        db.release_context_connection()
    elif hasattr(db, "close_connection"):
        db.close_connection()


async def handle_notes_graph_suggestions_job(
    job: dict[str, Any],
    *,
    jobs: JobManager,
    worker_id: str,
) -> dict[str, Any]:
    owner = str(job.get("owner_user_id") or "")
    db = await _open_owner_database(owner)
    try:
        worker = SuggestionWorker(
            store_factory=lambda _owner: db.note_graph_suggestion_store,
            resolve_capability=resolve_generation_capability,
            cancellation_requested=lambda row: _cancellation_requested(row, jobs=jobs),
            sync_cleanup=lambda: _close_database(db),
        )
        return await worker.handle(job)
    except SuggestionWorkerCancelled:
        await asyncio.to_thread(
            jobs.finalize_cancelled,
            int(job["id"]),
            reason="requested",
            expected_uuid=str(job["uuid"]),
            worker_id=worker_id,
            lease_id=str(job["lease_id"]),
        )
        raise
    finally:
        await asyncio.to_thread(_close_database, db)


async def _publish_completed(
    job: dict[str, Any],
    result: dict[str, Any],
    *,
    jobs: JobManager,
) -> None:
    owner = str(job["owner_user_id"])
    db = await _open_owner_database(owner)
    dataset_id = str(job["payload"]["dataset_id"])

    def publish() -> None:
        try:
            run = db.note_graph_suggestion_store.get_run(
                dataset_id=dataset_id,
                run_id=str(result["run_id"]),
            )
            SuggestionPublisher(
                jobs=jobs,
                store_factory=lambda _owner: db.note_graph_suggestion_store,
            ).publish(
                run=run,
                job_uuid=str(job["uuid"]),
                owner_user_id=owner,
                dataset_id=dataset_id,
                now=datetime.now(timezone.utc),
            )
        finally:
            _close_database(db)

    await asyncio.to_thread(publish)


async def run_notes_graph_suggestions_worker(
    stop_event: asyncio.Event | None = None,
) -> None:
    worker_id = (
        os.getenv("NOTES_GRAPH_SUGGESTIONS_WORKER_ID") or f"notes-graph-suggestions-worker-{os.getpid()}"
    ).strip()
    jobs = JobManager()
    sdk = WorkerSDK(jobs, build_worker_config(worker_id=worker_id))
    stop_waiter = None
    if stop_event is not None:

        async def watch_stop() -> None:
            await stop_event.wait()
            sdk.stop()

        stop_waiter = asyncio.create_task(watch_stop())
    logger.info("Notes graph suggestions Jobs worker starting")
    try:
        await sdk.run(
            handler=lambda job: handle_notes_graph_suggestions_job(
                job,
                jobs=jobs,
                worker_id=worker_id,
            ),
            job_type=JOB_TYPE,
            cancel_check=lambda job: _cancellation_requested(job, jobs=jobs),
            on_completed=lambda job, result: _publish_completed(job, result, jobs=jobs),
        )
    finally:
        if stop_waiter is not None:
            stop_waiter.cancel()


__all__ = [
    "build_worker_config",
    "handle_notes_graph_suggestions_job",
    "resolve_generation_capability",
    "run_notes_graph_suggestions_worker",
]
