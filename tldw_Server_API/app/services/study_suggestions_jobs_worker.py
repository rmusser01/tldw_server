"""Jobs worker for asynchronous study-suggestion snapshot refreshes."""

from __future__ import annotations

import asyncio
import os
from typing import Any

from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user_id
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.worker_sdk import WorkerConfig, WorkerSDK
from tldw_Server_API.app.core.StudySuggestions import snapshot_service
from tldw_Server_API.app.core.StudySuggestions.jobs import (
    STUDY_SUGGESTIONS_DOMAIN,
    STUDY_SUGGESTIONS_REFRESH_JOB_TYPE,
    study_suggestions_jobs_queue,
)


def _close_worker_database(db: Any) -> None:
    if db is None:
        return
    if hasattr(db, "release_context_connection"):
        db.release_context_connection()
        return
    if hasattr(db, "close_connection"):
        db.close_connection()


async def _get_databases_for_user(user_id: str) -> tuple[Any, Any]:
    try:
        normalized_user_id = int(str(user_id).strip())
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid owner_user_id for study-suggestions worker: {user_id!r}") from exc
    note_db = await get_chacha_db_for_user_id(
        normalized_user_id,
        client_id=f"study-suggestions-worker-{normalized_user_id}",
    )
    return note_db, None


async def handle_study_suggestions_job(job: dict[str, Any]) -> dict[str, Any]:
    if str(job.get("job_type") or "").strip().lower() != STUDY_SUGGESTIONS_REFRESH_JOB_TYPE:
        raise ValueError("unsupported_study_suggestions_job_type")

    owner_user_id = str(job.get("owner_user_id") or "").strip()
    if not owner_user_id:
        raise ValueError("missing_owner_user_id")

    payload = job.get("payload") or {}
    anchor_type = str(payload.get("anchor_type") or "").strip()
    anchor_id = payload.get("anchor_id")
    refreshed_from_snapshot_id = payload.get("snapshot_id")
    if not anchor_type or anchor_id is None:
        raise ValueError("missing_study_suggestions_anchor")

    note_db = None
    aux_db = None
    try:
        note_db, aux_db = await _get_databases_for_user(owner_user_id)
        principal = AuthPrincipal(
            kind="user",
            user_id=int(owner_user_id),
            roles=[],
            permissions=[],
            is_admin=False,
        )
        snapshot_id = snapshot_service.refresh_snapshot_for_anchor(
            note_db=note_db,
            anchor_type=anchor_type,
            anchor_id=int(anchor_id),
            refreshed_from_snapshot_id=(
                int(refreshed_from_snapshot_id) if refreshed_from_snapshot_id is not None else None
            ),
            principal=principal,
        )
        return {"snapshot_id": int(snapshot_id)}
    finally:
        for db in (aux_db, note_db):
            try:
                _close_worker_database(db)
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                logger.debug(
                    "Study-suggestions worker cleanup skipped for {}.",
                    type(db).__name__ if db is not None else "None",
                )


async def _should_cancel(
    job: dict[str, Any],
    *,
    job_manager: JobManager,
) -> bool:
    jm = job_manager
    job_id = int(job["id"])
    current = jm.get_job(job_id)
    if not current:
        return False
    status = str(current.get("status") or "").strip().lower()
    if status == "cancelled":
        return True
    if current.get("cancel_requested_at"):
        jm.finalize_cancelled(job_id, reason=str(current.get("cancellation_reason") or "requested"))
        return True
    return False


async def run_study_suggestions_jobs_worker(stop_event: asyncio.Event | None = None) -> None:
    worker_id = (os.getenv("STUDY_SUGGESTIONS_JOBS_WORKER_ID") or f"study-suggestions-worker-{os.getpid()}").strip()
    cfg = WorkerConfig(
        domain=STUDY_SUGGESTIONS_DOMAIN,
        queue=study_suggestions_jobs_queue(),
        worker_id=worker_id,
        lease_seconds=int(
            os.getenv("STUDY_SUGGESTIONS_JOBS_LEASE_SECONDS", os.getenv("JOBS_LEASE_SECONDS", "120")) or "120"
        ),
        renew_threshold_seconds=int(os.getenv("STUDY_SUGGESTIONS_JOBS_RENEW_THRESHOLD_SECONDS", "10") or "10"),
        renew_jitter_seconds=int(os.getenv("STUDY_SUGGESTIONS_JOBS_RENEW_JITTER_SECONDS", "0") or "0"),
    )
    jm = JobManager()
    sdk = WorkerSDK(jm, cfg)

    stop_waiter = None
    if stop_event is not None:

        async def _watch_stop() -> None:
            await stop_event.wait()
            sdk.stop()

        stop_waiter = asyncio.create_task(_watch_stop(), name="study_suggestions_jobs_worker_stop_waiter")

    logger.info("Study-suggestions Jobs worker starting: queue={} worker_id={}", cfg.queue, worker_id)
    try:
        await sdk.run(
            handler=handle_study_suggestions_job,
            cancel_check=lambda job_row: _should_cancel(job_row, job_manager=jm),
        )
    finally:
        if stop_waiter is not None:
            stop_waiter.cancel()


__all__ = [
    "_get_databases_for_user",
    "_should_cancel",
    "handle_study_suggestions_job",
    "run_study_suggestions_jobs_worker",
]
