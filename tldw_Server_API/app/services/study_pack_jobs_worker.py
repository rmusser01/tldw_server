"""Jobs worker for study-pack generation requests."""

from __future__ import annotations

import asyncio
import os
from typing import Any

from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user_id
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_owner
from tldw_Server_API.app.api.v1.schemas.study_packs import StudyPackCreateJobRequest
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.worker_sdk import WorkerConfig, WorkerSDK
from tldw_Server_API.app.core.StudyPacks.generation_service import StudyPackGenerationService
from tldw_Server_API.app.core.StudyPacks.jobs import (
    STUDY_PACKS_DOMAIN,
    STUDY_PACKS_JOB_TYPE,
    build_study_pack_job_result,
    study_pack_jobs_queue,
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
    normalized_user_id = int(str(user_id).strip())
    note_db = await get_chacha_db_for_user_id(
        normalized_user_id,
        client_id=f"study-pack-worker-{normalized_user_id}",
    )
    try:
        media_db = get_media_db_for_owner(normalized_user_id)
    except Exception:
        _close_worker_database(note_db)
        raise
    return note_db, media_db


async def handle_study_pack_job(job: dict[str, Any]) -> dict[str, Any]:
    if str(job.get("job_type") or STUDY_PACKS_JOB_TYPE).strip().lower() != STUDY_PACKS_JOB_TYPE:
        raise ValueError("unsupported_study_pack_job_type")

    owner_user_id = str(job.get("owner_user_id") or "").strip()
    if not owner_user_id:
        raise ValueError("missing_owner_user_id")

    payload = job.get("payload") or {}
    request = StudyPackCreateJobRequest.model_validate(payload)
    regenerate_from_pack_id = payload.get("regenerate_from_pack_id")
    if regenerate_from_pack_id is not None:
        regenerate_from_pack_id = int(regenerate_from_pack_id)
    expected_version = payload.get("expected_version")
    if expected_version is not None:
        expected_version = int(expected_version)

    note_db = None
    media_db = None
    try:
        note_db, media_db = await _get_databases_for_user(owner_user_id)
        service = StudyPackGenerationService(
            note_db=note_db,
            media_db=media_db,
            provider=None,
            model=None,
        )
        created = await service.create_from_request(
            request,
            regenerate_from_pack_id=regenerate_from_pack_id,
            expected_regenerate_version=expected_version,
        )
        return build_study_pack_job_result(
            pack_id=created.pack_id,
            deck_id=created.deck_id,
            deck_name=created.deck_name,
            regenerated_from_pack_id=created.regenerated_from_pack_id,
        )
    finally:
        for db in (media_db, note_db):
            try:
                _close_worker_database(db)
            except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
                logger.debug("Study-pack worker cleanup skipped for {}.", type(db).__name__ if db is not None else "None")


async def _should_cancel(
    job: dict[str, Any],
    *,
    job_manager: JobManager | None = None,
) -> bool:
    jm = job_manager or JobManager()
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


async def run_study_pack_jobs_worker(stop_event: asyncio.Event | None = None) -> None:
    worker_id = (os.getenv("STUDY_PACK_JOBS_WORKER_ID") or f"study-pack-worker-{os.getpid()}").strip()
    cfg = WorkerConfig(
        domain=STUDY_PACKS_DOMAIN,
        queue=study_pack_jobs_queue(),
        worker_id=worker_id,
        lease_seconds=int(os.getenv("STUDY_PACK_JOBS_LEASE_SECONDS", os.getenv("JOBS_LEASE_SECONDS", "120")) or "120"),
        renew_threshold_seconds=int(os.getenv("STUDY_PACK_JOBS_RENEW_THRESHOLD_SECONDS", "10") or "10"),
        renew_jitter_seconds=int(os.getenv("STUDY_PACK_JOBS_RENEW_JITTER_SECONDS", "0") or "0"),
    )
    jm = JobManager()
    sdk = WorkerSDK(jm, cfg)

    stop_waiter = None
    if stop_event is not None:
        async def _watch_stop() -> None:
            await stop_event.wait()
            sdk.stop()

        stop_waiter = asyncio.create_task(_watch_stop(), name="study_pack_jobs_worker_stop_waiter")

    logger.info(
        "Study-pack Jobs worker starting: queue={} worker_id={}",
        cfg.queue,
        worker_id,
    )
    try:
        await sdk.run(
            handler=handle_study_pack_job,
            cancel_check=lambda job_row: _should_cancel(job_row, job_manager=jm),
        )
    finally:
        if stop_waiter is not None:
            stop_waiter.cancel()


__all__ = [
    "_should_cancel",
    "handle_study_pack_job",
    "run_study_pack_jobs_worker",
]
