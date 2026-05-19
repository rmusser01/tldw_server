from __future__ import annotations

import asyncio
import os

from loguru import logger

from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Reminders.reminder_jobs import (
    REMINDERS_DOMAIN,
    REMINDER_JOB_TYPE,
    handle_reminder_job,
)
from tldw_Server_API.app.core.testing import env_flag_enabled

_REMINDER_WORKER_NONCRITICAL_EXCEPTIONS = (
    AttributeError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)


def reminder_jobs_queue() -> str:
    queue = (os.getenv("REMINDER_JOBS_QUEUE") or "default").strip()
    return queue or "default"


async def run_reminder_jobs_worker(stop_event: asyncio.Event | None = None) -> None:
    jm = JobManager()
    worker_id = "reminder-jobs-worker"
    queue = reminder_jobs_queue()
    poll_sleep = float(os.getenv("JOBS_POLL_INTERVAL_SECONDS", "1.0") or "1.0")
    logger.info("Starting Reminder Jobs worker")
    while True:
        if stop_event and stop_event.is_set():
            logger.info("Stopping Reminder Jobs worker on shutdown signal")
            return
        try:
            lease_seconds = int(os.getenv("JOBS_LEASE_SECONDS", "120") or "120")
            job = jm.acquire_next_job(
                domain=REMINDERS_DOMAIN,
                queue=queue,
                lease_seconds=lease_seconds,
                worker_id=worker_id,
            )
            if not job:
                await asyncio.sleep(poll_sleep)
                continue

            lease_id = str(job.get("lease_id"))
            if str(job.get("job_type") or "").lower() != REMINDER_JOB_TYPE:
                jm.fail_job(
                    int(job["id"]),
                    error="unsupported reminder job_type",
                    retryable=False,
                    worker_id=worker_id,
                    lease_id=lease_id,
                    completion_token=lease_id,
                )
                continue

            try:
                result = await handle_reminder_job(job)
                jm.complete_job(
                    int(job["id"]),
                    result=result,
                    worker_id=worker_id,
                    lease_id=lease_id,
                    completion_token=lease_id,
                )
            except _REMINDER_WORKER_NONCRITICAL_EXCEPTIONS as exc:
                jm.fail_job(
                    int(job["id"]),
                    error=str(exc),
                    retryable=False,
                    worker_id=worker_id,
                    lease_id=lease_id,
                    completion_token=lease_id,
                )
        except _REMINDER_WORKER_NONCRITICAL_EXCEPTIONS as exc:
            logger.error("Reminder Jobs worker loop error: {}", exc)
            await asyncio.sleep(poll_sleep)


async def start_reminder_jobs_worker(stop_event: asyncio.Event | None = None) -> asyncio.Task | None:
    if not env_flag_enabled("REMINDER_JOBS_WORKER_ENABLED"):
        return None
    managed_stop_event = stop_event or asyncio.Event()
    return asyncio.create_task(
        run_reminder_jobs_worker(managed_stop_event),
        name="reminder_jobs_worker",
    )


__all__ = [
    "reminder_jobs_queue",
    "run_reminder_jobs_worker",
    "start_reminder_jobs_worker",
]
