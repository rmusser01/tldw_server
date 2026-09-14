"""Jobs worker for Visual Identity expression pack imports."""

from __future__ import annotations

import asyncio
import contextlib
import os
from pathlib import Path
from typing import Any

from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user_id
from tldw_Server_API.app.core.DB_Management.VisualIdentity_DB import VisualIdentityRepository
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.worker_sdk import WorkerConfig, WorkerSDK
from tldw_Server_API.app.core.testing import env_flag_enabled
from tldw_Server_API.app.core.Visual_Identities.archive_import import (
    import_visual_identity_expression_zip,
)
from tldw_Server_API.app.core.Visual_Identities.jobs import (
    VISUAL_IDENTITIES_DOMAIN,
    VISUAL_IDENTITY_IMPORT_ZIP_JOB_TYPE,
    visual_identity_jobs_queue,
)


def _close_worker_database(db: Any) -> None:
    if db is None:
        return
    if hasattr(db, "release_context_connection"):
        db.release_context_connection()
        return
    if hasattr(db, "close_connection"):
        db.close_connection()


async def handle_visual_identity_import_zip_job(
    job: dict[str, Any],
    *,
    job_manager: JobManager | None = None,
    storage_root: str | Path | None = None,
) -> dict[str, Any]:
    """Process one Visual Identity ZIP import job."""
    del job_manager
    job_type = str(job.get("job_type") or "").strip()
    if job_type != VISUAL_IDENTITY_IMPORT_ZIP_JOB_TYPE:
        raise ValueError("unsupported_visual_identity_worker_job_type")

    payload = job.get("payload") or {}
    owner_user_id = _job_owner_user_id(job)
    payload_owner_user_id = _payload_int(payload, "owner_user_id", error_code="missing_owner_user_id")
    if payload_owner_user_id != owner_user_id:
        raise ValueError("visual_identity_job_owner_mismatch")

    draft_id = _payload_int(payload, "draft_id", error_code="missing_draft_id")
    upload_path = _payload_str(payload, "upload_path", error_code="missing_upload_path")

    note_db = await get_chacha_db_for_user_id(
        owner_user_id,
        client_id=f"visual-identity-worker-{owner_user_id}",
    )
    try:
        repo = VisualIdentityRepository.initialized(note_db)
        imported = import_visual_identity_expression_zip(
            repo,
            owner_user_id=owner_user_id,
            draft_id=draft_id,
            archive_path=upload_path,
            storage_root=storage_root,
        )
        return {
            "draft_id": int(imported["id"]),
            "status": str(imported.get("status") or ""),
            "source_filename": str(imported.get("source_filename") or ""),
        }
    finally:
        _close_worker_database(note_db)


def _job_owner_user_id(job: dict[str, Any]) -> int:
    try:
        owner_user_id = int(str(job.get("owner_user_id") or "").strip())
    except (TypeError, ValueError) as exc:
        raise ValueError("missing_owner_user_id") from exc
    if owner_user_id <= 0:
        raise ValueError("missing_owner_user_id")
    return owner_user_id


def _payload_int(payload: Any, key: str, *, error_code: str) -> int:
    try:
        value = int(payload[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(error_code) from exc
    if value <= 0:
        raise ValueError(error_code)
    return value


def _payload_str(payload: Any, key: str, *, error_code: str) -> str:
    try:
        value = str(payload[key]).strip()
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(error_code) from exc
    if not value:
        raise ValueError(error_code)
    return value


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
    return bool(current.get("cancel_requested_at"))


async def run_visual_identity_jobs_worker(stop_event: asyncio.Event | None = None) -> None:
    """Run the Visual Identity Jobs worker until stopped."""
    worker_id = (
        os.getenv("VISUAL_IDENTITY_JOBS_WORKER_ID")
        or f"visual-identity-worker-{os.getpid()}"
    ).strip()
    cfg = WorkerConfig(
        domain=VISUAL_IDENTITIES_DOMAIN,
        queue=visual_identity_jobs_queue(),
        worker_id=worker_id,
        lease_seconds=int(
            os.getenv("VISUAL_IDENTITY_JOBS_LEASE_SECONDS", os.getenv("JOBS_LEASE_SECONDS", "120")) or "120"
        ),
        renew_threshold_seconds=int(os.getenv("VISUAL_IDENTITY_JOBS_RENEW_THRESHOLD_SECONDS", "10") or "10"),
        renew_jitter_seconds=int(os.getenv("VISUAL_IDENTITY_JOBS_RENEW_JITTER_SECONDS", "0") or "0"),
    )
    jm = JobManager()
    sdk = WorkerSDK(jm, cfg)

    stop_task: asyncio.Task[None] | None = None
    if stop_event is not None:

        async def _watch_stop() -> None:
            await stop_event.wait()
            sdk.stop()

        stop_task = asyncio.create_task(_watch_stop(), name="visual_identity_jobs_worker_stop_watch")

    logger.info("Visual Identity Jobs worker starting: queue={} worker_id={}", cfg.queue, worker_id)
    try:
        await sdk.run(
            handler=lambda job_row: handle_visual_identity_import_zip_job(
                job_row,
                job_manager=jm,
            ),
            cancel_check=lambda job_row: _should_cancel(job_row, job_manager=jm),
            job_type=VISUAL_IDENTITY_IMPORT_ZIP_JOB_TYPE,
        )
    finally:
        if stop_task is not None:
            stop_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await stop_task


async def start_visual_identity_jobs_worker(
    stop_event: asyncio.Event | None = None,
) -> asyncio.Task | None:
    if not env_flag_enabled("VISUAL_IDENTITY_JOBS_WORKER_ENABLED"):
        return None
    return asyncio.create_task(
        run_visual_identity_jobs_worker(stop_event),
        name="visual_identity_jobs_worker",
    )


__all__ = [
    "_close_worker_database",
    "_should_cancel",
    "handle_visual_identity_import_zip_job",
    "run_visual_identity_jobs_worker",
    "start_visual_identity_jobs_worker",
]
