"""Jobs worker for scheduled automation definitions (TASK-13021).

Mirrors ``reminder_jobs_worker.py`` exactly: env-gated worker loop,
lease-based job acquisition on the ``scheduled_tasks`` domain, and
completion/failure reporting back to the Jobs pipeline. Enable together
with the feed (`SCHEDULED_TASKS_AUTOMATION_SCHEDULER_ENABLED`) — arming
without a consumer only queues jobs nobody executes.
"""

from __future__ import annotations

import asyncio
import os

from loguru import logger

from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Scheduled_Tasks.agent_task_jobs import (
    AUTOMATION_DOMAIN,
    AUTOMATION_JOB_TYPE,
    handle_agent_task_job,
)
from tldw_Server_API.app.core.testing import env_flag_enabled

_AGENT_WORKER_NONCRITICAL_EXCEPTIONS = (
    AttributeError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)


def agent_task_jobs_queue() -> str:
    """Return the Jobs queue name for automation work (AUTOMATION_JOBS_QUEUE)."""
    queue = (os.getenv("AUTOMATION_JOBS_QUEUE") or "default").strip()
    return queue or "default"


async def run_agent_task_jobs_worker(stop_event: asyncio.Event | None = None) -> None:
    """Poll the automation queue and consume agent_task_run Jobs until stopped."""
    jm = JobManager()
    worker_id = "agent-task-jobs-worker"
    queue = agent_task_jobs_queue()
    poll_sleep = float(os.getenv("JOBS_POLL_INTERVAL_SECONDS", "1.0") or "1.0")
    logger.info("Starting Agent Task Jobs worker")
    while True:
        if stop_event and stop_event.is_set():
            logger.info("Stopping Agent Task Jobs worker on shutdown signal")
            return
        try:
            lease_seconds = int(os.getenv("JOBS_LEASE_SECONDS", "120") or "120")
            job = jm.acquire_next_job(
                domain=AUTOMATION_DOMAIN,
                queue=queue,
                lease_seconds=lease_seconds,
                worker_id=worker_id,
            )
            if not job:
                await asyncio.sleep(poll_sleep)
                continue

            lease_id = str(job.get("lease_id"))
            if str(job.get("job_type") or "").lower() != AUTOMATION_JOB_TYPE:
                jm.fail_job(
                    int(job["id"]),
                    error="unsupported automation job_type",
                    retryable=False,
                    worker_id=worker_id,
                    lease_id=lease_id,
                    completion_token=lease_id,
                )
                continue

            try:
                result = await handle_agent_task_job(job)
                jm.complete_job(
                    int(job["id"]),
                    result=result,
                    worker_id=worker_id,
                    lease_id=lease_id,
                    completion_token=lease_id,
                )
            except _AGENT_WORKER_NONCRITICAL_EXCEPTIONS as exc:
                jm.fail_job(
                    int(job["id"]),
                    error=str(exc),
                    retryable=False,
                    worker_id=worker_id,
                    lease_id=lease_id,
                    completion_token=lease_id,
                )
        except _AGENT_WORKER_NONCRITICAL_EXCEPTIONS as exc:
            logger.error("Agent Task Jobs worker loop error: {}", exc)
            await asyncio.sleep(poll_sleep)


async def start_agent_task_jobs_worker(
    stop_event: asyncio.Event | None = None,
) -> asyncio.Task | None:
    """Start the worker when its env gate is enabled (AGENT_TASK_JOBS_WORKER_ENABLED)."""
    if not env_flag_enabled("AGENT_TASK_JOBS_WORKER_ENABLED"):
        return None
    managed_stop_event = stop_event or asyncio.Event()
    return asyncio.create_task(
        run_agent_task_jobs_worker(managed_stop_event),
        name="agent_task_jobs_worker",
    )


__all__ = [
    "agent_task_jobs_queue",
    "run_agent_task_jobs_worker",
    "start_agent_task_jobs_worker",
]
