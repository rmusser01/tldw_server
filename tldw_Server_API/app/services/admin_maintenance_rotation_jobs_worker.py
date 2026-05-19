from __future__ import annotations

import asyncio
import contextlib
import json
import os
from typing import TYPE_CHECKING, Any, Callable

from loguru import logger

from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.worker_sdk import WorkerConfig, WorkerSDK
from tldw_Server_API.app.core.testing import env_flag_enabled

if TYPE_CHECKING:
    from tldw_Server_API.app.core.AuthNZ.repos.maintenance_rotation_runs_repo import (
        AuthnzMaintenanceRotationRunsRepo,
    )


MAINTENANCE_ROTATION_DOMAIN = "maintenance"
MAINTENANCE_ROTATION_JOB_TYPE = "crypto_rotation"
MAINTENANCE_ROTATION_KEY_SOURCE = "env:jobs_crypto_rotate"


def maintenance_rotation_queue() -> str:
    """Return the queue used for authoritative maintenance rotation jobs."""
    return (os.getenv("ADMIN_MAINTENANCE_ROTATION_JOBS_QUEUE") or "default").strip() or "default"


def maintenance_rotation_worker_enabled() -> bool:
    """Return True when the authoritative maintenance rotation worker path is enabled."""
    return env_flag_enabled("ADMIN_MAINTENANCE_ROTATION_JOBS_WORKER_ENABLED")


def build_maintenance_rotation_job_payload(*, run_id: str) -> dict[str, Any]:
    """Build the opaque Jobs payload for a maintenance rotation run."""
    return {"run_id": str(run_id)}


def build_maintenance_rotation_idempotency_key(*, run_id: str) -> str:
    """Build the idempotency key for one maintenance rotation run enqueue."""
    return f"maintenance-rotation:{run_id}"


def resolve_rotation_keys(*, key_source: str) -> tuple[str, str]:
    """Resolve server-side configured rotation keys for the supported key source."""
    if str(key_source).strip() != MAINTENANCE_ROTATION_KEY_SOURCE:
        raise ValueError("unsupported_rotation_key_source")
    old_key = os.getenv("JOBS_CRYPTO_ROTATE_OLD_KEY", "").strip()
    new_key = os.getenv("JOBS_CRYPTO_ROTATE_NEW_KEY", "").strip()
    if not old_key or not new_key:
        raise ValueError("rotation_key_source_unavailable")
    return old_key, new_key


async def _get_repo() -> AuthnzMaintenanceRotationRunsRepo:
    """Build the maintenance rotation repo for worker execution."""
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.AuthNZ.repos.maintenance_rotation_runs_repo import (
        AuthnzMaintenanceRotationRunsRepo,
    )

    pool = await get_db_pool()
    repo = AuthnzMaintenanceRotationRunsRepo(pool)
    await repo.ensure_schema()
    return repo


async def enqueue_maintenance_rotation_run(
    run: dict[str, Any],
    *,
    job_manager: JobManager | None = None,
) -> str:
    """Enqueue a maintenance rotation run into Jobs and return the job id."""
    jobs = job_manager or JobManager()
    job = jobs.create_job(
        domain=MAINTENANCE_ROTATION_DOMAIN,
        queue=maintenance_rotation_queue(),
        job_type=MAINTENANCE_ROTATION_JOB_TYPE,
        payload=build_maintenance_rotation_job_payload(run_id=str(run["id"])),
        owner_user_id=(
            str(run["requested_by_user_id"]) if run.get("requested_by_user_id") is not None else None
        ),
        idempotency_key=build_maintenance_rotation_idempotency_key(run_id=str(run["id"])),
    )
    return str(job.get("id"))


async def handle_maintenance_rotation_job(
    job: dict[str, Any],
    *,
    repo=None,
    rotate_encryption_keys_fn: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Execute one authoritative maintenance rotation run from the Jobs queue."""
    payload = job.get("payload") or {}
    run_id = str(payload.get("run_id") or "").strip()
    if not run_id:
        raise ValueError("missing_run_id")

    repo = repo or await _get_repo()
    run = await repo.get_run(run_id)
    if not run:
        raise ValueError("missing_run")

    job_id = str(job.get("id")) if job.get("id") is not None else None
    await repo.mark_running(run_id, job_id=job_id)

    rotate_fn = rotate_encryption_keys_fn or JobManager().rotate_encryption_keys
    try:
        old_key, new_key = resolve_rotation_keys(key_source=str(run.get("key_source") or ""))
        fields = json.loads(str(run.get("fields_json") or "[]"))
        affected = await asyncio.to_thread(
            rotate_fn,
            domain=run.get("domain"),
            queue=run.get("queue"),
            job_type=run.get("job_type"),
            old_key_b64=old_key,
            new_key_b64=new_key,
            fields=fields,
            limit=int(run.get("limit") or 1000),
            dry_run=str(run.get("mode") or "") == "dry_run",
        )
    except Exception as exc:
        await repo.mark_failed(run_id, error_message=str(exc))
        raise

    await repo.mark_complete(run_id, affected_count=int(affected))
    logger.info("Maintenance rotation job completed: run_id={} job_id={}", run_id, job_id)
    return {
        "status": "complete",
        "run_id": run_id,
        "job_id": job_id,
        "affected_count": int(affected),
    }


async def run_admin_maintenance_rotation_jobs_worker(
    stop_event: asyncio.Event | None = None,
) -> None:
    """Run the WorkerSDK loop for authoritative maintenance rotation jobs."""
    worker_id = (
        os.getenv("ADMIN_MAINTENANCE_ROTATION_JOBS_WORKER_ID") or f"admin-maintenance-{os.getpid()}"
    ).strip()
    cfg = WorkerConfig(
        domain=MAINTENANCE_ROTATION_DOMAIN,
        queue=maintenance_rotation_queue(),
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
            name="admin_maintenance_rotation_jobs_worker_stop_watch",
        )
    logger.info(
        "Admin maintenance rotation Jobs worker starting: queue={} worker_id={}",
        cfg.queue,
        worker_id,
    )
    try:
        await sdk.run(handler=handle_maintenance_rotation_job)
    finally:
        if stop_task is not None:
            stop_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await stop_task


async def start_admin_maintenance_rotation_jobs_worker(
    stop_event: asyncio.Event | None = None,
) -> asyncio.Task | None:
    """Start the maintenance rotation Jobs worker when explicitly enabled."""
    if not maintenance_rotation_worker_enabled():
        return None
    return asyncio.create_task(
        run_admin_maintenance_rotation_jobs_worker(stop_event),
        name="admin_maintenance_rotation_jobs_worker",
    )


__all__ = [
    "MAINTENANCE_ROTATION_DOMAIN",
    "MAINTENANCE_ROTATION_JOB_TYPE",
    "MAINTENANCE_ROTATION_KEY_SOURCE",
    "build_maintenance_rotation_idempotency_key",
    "build_maintenance_rotation_job_payload",
    "enqueue_maintenance_rotation_run",
    "handle_maintenance_rotation_job",
    "maintenance_rotation_queue",
    "maintenance_rotation_worker_enabled",
    "resolve_rotation_keys",
    "run_admin_maintenance_rotation_jobs_worker",
    "start_admin_maintenance_rotation_jobs_worker",
]
