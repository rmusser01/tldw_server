from __future__ import annotations

import asyncio
import contextlib
import os
from typing import TYPE_CHECKING, Any, Callable

from loguru import logger

from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.worker_sdk import WorkerConfig, WorkerSDK
from tldw_Server_API.app.core.Storage.backup_schedule_jobs import (
    BACKUP_SCHEDULE_DOMAIN,
    BACKUP_SCHEDULE_JOB_TYPE,
    backup_schedule_queue,
    parse_backup_schedule_job_payload,
)
from tldw_Server_API.app.core.testing import env_flag_enabled
from tldw_Server_API.app.services.admin_data_ops_service import create_backup_snapshot

if TYPE_CHECKING:
    from tldw_Server_API.app.core.AuthNZ.repos.backup_schedules_repo import (
        AuthnzBackupSchedulesRepo,
    )


async def _get_repo() -> AuthnzBackupSchedulesRepo:
    """Build the backup schedules repository for worker execution."""
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.AuthNZ.repos.backup_schedules_repo import (
        AuthnzBackupSchedulesRepo,
    )

    pool = await get_db_pool()
    repo = AuthnzBackupSchedulesRepo(pool)
    await repo.ensure_schema()
    return repo


async def handle_backup_schedule_job(
    job: dict[str, Any],
    *,
    repo=None,
    create_backup_snapshot_fn: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Execute one scheduled backup job using the existing backup service."""
    payload = parse_backup_schedule_job_payload(job.get("payload") or {})
    repo = repo or await _get_repo()
    create_fn = create_backup_snapshot_fn or create_backup_snapshot

    schedule = await repo.get_schedule(payload["schedule_id"], include_deleted=True)
    if not schedule:
        raise ValueError("missing_schedule")
    run = await repo.get_run(payload["run_id"])
    if not run:
        raise ValueError("missing_run")
    if str(run.get("schedule_id")) != payload["schedule_id"]:
        raise ValueError("run_schedule_mismatch")
    if run.get("job_id") is not None and str(run["job_id"]) != str(job.get("id")):
        raise ValueError("job_id_mismatch")
    if str(schedule.get("dataset") or "").strip().lower() != payload["dataset"]:
        raise ValueError("schedule_payload_mismatch")
    if schedule.get("target_user_id") != payload["target_user_id"]:
        raise ValueError("schedule_payload_mismatch")

    await repo.mark_run_running(run_id=payload["run_id"])
    try:
        backup_item = await asyncio.to_thread(
            create_fn,
            dataset=str(schedule["dataset"]),
            user_id=schedule.get("target_user_id"),
            backup_type="full",
            max_backups=int(schedule["retention_count"]),
        )
    except Exception as exc:
        await repo.mark_run_failed(run_id=payload["run_id"], error=str(exc), last_status="failed")
        raise

    await repo.mark_run_succeeded(run_id=payload["run_id"])
    logger.info(
        "Scheduled backup job handled successfully: schedule_id={} run_id={} backup_id={}",
        payload["schedule_id"],
        payload["run_id"],
        getattr(backup_item, "filename", None),
    )
    return {
        "status": "succeeded",
        "schedule_id": payload["schedule_id"],
        "run_id": payload["run_id"],
        "backup_id": getattr(backup_item, "filename", None),
        "dataset": getattr(backup_item, "dataset", str(schedule["dataset"])),
        "user_id": getattr(backup_item, "user_id", schedule.get("target_user_id")),
    }


async def run_admin_backup_jobs_worker(stop_event: asyncio.Event | None = None) -> None:
    worker_id = (os.getenv("ADMIN_BACKUP_JOBS_WORKER_ID") or f"admin-backup-{os.getpid()}").strip()
    cfg = WorkerConfig(
        domain=BACKUP_SCHEDULE_DOMAIN,
        queue=backup_schedule_queue(),
        worker_id=worker_id,
    )
    jm = JobManager()
    sdk = WorkerSDK(jm, cfg)
    stop_task: asyncio.Task[None] | None = None
    if stop_event is not None:
        async def _watch_stop() -> None:
            await stop_event.wait()
            sdk.stop()

        stop_task = asyncio.create_task(
            _watch_stop(),
            name="admin_backup_jobs_worker_stop_watch",
        )
    logger.info("Admin backup Jobs worker starting: queue={} worker_id={}", cfg.queue, worker_id)
    try:
        await sdk.run(handler=handle_backup_schedule_job)
    finally:
        if stop_task is not None:
            stop_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await stop_task


async def start_admin_backup_jobs_worker(stop_event: asyncio.Event | None = None) -> asyncio.Task | None:
    if not env_flag_enabled("ADMIN_BACKUP_JOBS_WORKER_ENABLED"):
        return None
    return asyncio.create_task(
        run_admin_backup_jobs_worker(stop_event),
        name="admin_backup_jobs_worker",
    )


__all__ = [
    "handle_backup_schedule_job",
    "run_admin_backup_jobs_worker",
    "start_admin_backup_jobs_worker",
]
