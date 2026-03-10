"""Worker entrypoint for managed vLLM lifecycle jobs."""

from __future__ import annotations

import asyncio
import contextlib
import os

from loguru import logger

from tldw_Server_API.app.core.Jobs.worker_sdk import WorkerConfig, WorkerSDK
from tldw_Server_API.app.core.Jobs.worker_utils import coerce_int as _coerce_int
from tldw_Server_API.app.core.Jobs.worker_utils import jobs_manager_from_env as _jobs_manager
from tldw_Server_API.app.core.VLLM_Management.job_handlers import handle_vllm_management_job
from tldw_Server_API.app.core.VLLM_Management.service import (
    VLLM_MANAGEMENT_DOMAIN,
    VLLMManagementService,
    vllm_management_queue,
)


async def run_vllm_management_worker(
    stop_event: asyncio.Event | None = None,
    *,
    job_manager=None,
    service: VLLMManagementService | None = None,
) -> None:
    worker_id = (os.getenv("VLLM_MANAGEMENT_WORKER_ID") or f"vllm-management-{os.getpid()}").strip()
    lease_seconds = _coerce_int(
        os.getenv("VLLM_MANAGEMENT_LEASE_SECONDS") or os.getenv("JOBS_LEASE_SECONDS"),
        60,
    )
    renew_jitter = _coerce_int(
        os.getenv("VLLM_MANAGEMENT_RENEW_JITTER_SECONDS") or os.getenv("JOBS_LEASE_RENEW_JITTER_SECONDS"),
        5,
    )
    renew_threshold = _coerce_int(
        os.getenv("VLLM_MANAGEMENT_RENEW_THRESHOLD_SECONDS") or os.getenv("JOBS_LEASE_RENEW_THRESHOLD_SECONDS"),
        10,
    )
    queue = vllm_management_queue()
    jm = job_manager or _jobs_manager()
    resolved_service = service or VLLMManagementService(job_manager=jm)
    cfg = WorkerConfig(
        domain=VLLM_MANAGEMENT_DOMAIN,
        queue=queue,
        worker_id=worker_id,
        lease_seconds=lease_seconds,
        renew_jitter_seconds=renew_jitter,
        renew_threshold_seconds=renew_threshold,
    )
    sdk = WorkerSDK(jm, cfg)
    stop_watcher_task: asyncio.Task[None] | None = None
    if stop_event is not None:
        async def _watch_stop() -> None:
            await stop_event.wait()
            sdk.stop()

        stop_watcher_task = asyncio.create_task(_watch_stop())

    async def _handler(job: dict[str, object]) -> dict[str, object]:
        result = await handle_vllm_management_job(job, service=resolved_service)
        if stop_event is not None and stop_event.is_set():
            sdk.stop()
        return dict(result)

    logger.info("Managed vLLM worker starting: queue={} worker_id={}", queue, worker_id)
    try:
        await sdk.run(handler=_handler)
    finally:
        if stop_watcher_task is not None and not stop_watcher_task.done():
            stop_watcher_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await stop_watcher_task


if __name__ == "__main__":
    asyncio.run(run_vllm_management_worker())
