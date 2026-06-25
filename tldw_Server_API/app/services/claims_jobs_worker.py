"""Lifecycle-managed Jobs worker for Claims background jobs."""

from __future__ import annotations

import asyncio
import contextlib
import os

from loguru import logger

from tldw_Server_API.app.core.Claims_Extraction.claims_job_contracts import (
    CLAIMS_JOBS_DOMAIN,
)
from tldw_Server_API.app.core.Claims_Extraction.claims_job_handlers import (
    process_claims_job,
)
from tldw_Server_API.app.core.Claims_Extraction.claims_jobs import (
    claims_jobs_queue,
    claims_jobs_worker_enabled,
)
from tldw_Server_API.app.core.Jobs.worker_sdk import WorkerConfig, WorkerSDK
from tldw_Server_API.app.core.Jobs.worker_utils import (
    coerce_int,
    jobs_manager_from_env,
)
from tldw_Server_API.app.services.lifecycle_worker_specs import (
    ShutdownPhase,
    WorkerLifecycleContext,
    WorkerSpec,
    stop_event_worker_spec,
)


def _worker_id() -> str:
    return (
        os.getenv("CLAIMS_JOBS_WORKER_ID") or "claims-jobs-worker"
    ).strip() or "claims-jobs-worker"


def build_claims_worker_config() -> WorkerConfig:
    """Build the WorkerSDK configuration for Claims Jobs."""

    return WorkerConfig(
        domain=CLAIMS_JOBS_DOMAIN,
        queue=claims_jobs_queue(),
        worker_id=_worker_id(),
        lease_seconds=coerce_int(os.getenv("CLAIMS_JOBS_LEASE_SECONDS"), 120),
        renew_jitter_seconds=coerce_int(
            os.getenv("CLAIMS_JOBS_RENEW_JITTER_SECONDS"),
            5,
        ),
        renew_threshold_seconds=coerce_int(
            os.getenv("CLAIMS_JOBS_RENEW_THRESHOLD_SECONDS"),
            15,
        ),
        backoff_base_seconds=coerce_int(
            os.getenv("CLAIMS_JOBS_BACKOFF_BASE_SECONDS"),
            2,
        ),
        backoff_max_seconds=coerce_int(os.getenv("CLAIMS_JOBS_BACKOFF_MAX_SECONDS"), 30),
        retry_on_exception=True,
        retry_backoff_seconds=coerce_int(
            os.getenv("CLAIMS_JOBS_RETRY_BACKOFF_SECONDS"),
            10,
        ),
    )


async def start_claims_jobs_worker(stop_event: asyncio.Event | None = None) -> None:
    """Run the Claims Jobs worker until lifecycle shutdown requests a stop."""

    manager = jobs_manager_from_env()
    config = build_claims_worker_config()
    sdk = WorkerSDK(manager, config)

    async def _watch_stop() -> None:
        if stop_event is None:
            return
        await stop_event.wait()
        sdk.stop()

    stop_task = asyncio.create_task(_watch_stop(), name="claims_jobs_worker_stop_waiter")
    logger.info("Claims Jobs worker starting: queue={}", config.queue)
    try:
        await sdk.run(handler=process_claims_job)
    except asyncio.CancelledError:
        sdk.stop()
        raise
    finally:
        sdk.stop()
        stop_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await stop_task
        logger.info("Claims Jobs worker stopped")


def _claims_jobs_worker_enabled(context: WorkerLifecycleContext) -> bool:
    return claims_jobs_worker_enabled(context.settings)


def provide_claims_jobs_worker_specs(
    _context: WorkerLifecycleContext | None = None,
) -> tuple[WorkerSpec, ...]:
    """Return lifecycle specs for the Claims Jobs worker."""

    return (
        stop_event_worker_spec(
            name="claims_jobs_task",
            worker_service=start_claims_jobs_worker,
            category="jobs",
            phase=ShutdownPhase.JOB_POLLER_QUIESCE,
            enabled=_claims_jobs_worker_enabled,
        ),
    )


__all__ = [
    "build_claims_worker_config",
    "process_claims_job",
    "provide_claims_jobs_worker_specs",
    "start_claims_jobs_worker",
]
