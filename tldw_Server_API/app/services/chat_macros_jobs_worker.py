"""Jobs worker for chat macro runs."""

from __future__ import annotations

import asyncio
import os

from loguru import logger

from tldw_Server_API.app.core.Chat_Macros.jobs import (
    CHAT_MACROS_DOMAIN,
    chat_macro_jobs_queue,
    handle_chat_macro_job,
    should_cancel_chat_macro_job,
)
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.worker_sdk import WorkerConfig, WorkerSDK


async def run_chat_macros_jobs_worker(stop_event: asyncio.Event | None = None) -> None:
    """Run the WorkerSDK loop for chat macro jobs."""

    worker_id = (os.getenv("CHAT_MACROS_JOBS_WORKER_ID") or f"chat-macro-worker-{os.getpid()}").strip()
    cfg = WorkerConfig(
        domain=CHAT_MACROS_DOMAIN,
        queue=chat_macro_jobs_queue(),
        worker_id=worker_id,
        lease_seconds=int(os.getenv("CHAT_MACROS_JOBS_LEASE_SECONDS", os.getenv("JOBS_LEASE_SECONDS", "120")) or "120"),
        renew_threshold_seconds=int(os.getenv("CHAT_MACROS_JOBS_RENEW_THRESHOLD_SECONDS", "10") or "10"),
        renew_jitter_seconds=int(os.getenv("CHAT_MACROS_JOBS_RENEW_JITTER_SECONDS", "0") or "0"),
    )
    jm = JobManager()
    sdk = WorkerSDK(jm, cfg)

    stop_waiter = None
    if stop_event is not None:

        async def _watch_stop() -> None:
            await stop_event.wait()
            sdk.stop()

        stop_waiter = asyncio.create_task(_watch_stop(), name="chat_macros_jobs_worker_stop_waiter")

    logger.info("Chat macro Jobs worker starting: queue={} worker_id={}", cfg.queue, worker_id)
    try:
        await sdk.run(
            handler=handle_chat_macro_job,
            cancel_check=lambda job_row: should_cancel_chat_macro_job(job_row, job_manager=jm),
        )
    finally:
        if stop_waiter is not None:
            stop_waiter.cancel()


__all__ = ["run_chat_macros_jobs_worker"]
