"""Worker entrypoint for Explainer generation Jobs."""

from __future__ import annotations

import asyncio
import contextlib
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.DB_Management.Explainer_DB import open_explainer_db
from tldw_Server_API.app.core.Explainer.repository import ExplainerRepository
from tldw_Server_API.app.core.Jobs.worker_sdk import WorkerConfig, WorkerSDK
from tldw_Server_API.app.core.Jobs.worker_utils import coerce_int as _coerce_int
from tldw_Server_API.app.core.Jobs.worker_utils import jobs_manager_from_env as _jobs_manager

from .jobs import (
    EXPLAINER_DOMAIN,
    EXPLAINER_QUEUE,
    ExplainerGenerator,
    handle_explainer_node_expansion_job,
    make_configured_explainer_generator,
)
from .retrieval import ExplainerRetriever


async def run_explainer_jobs_worker(
    stop_event: asyncio.Event | None = None,
) -> None:
    """Run the Explainer Jobs worker loop until stopped."""

    worker_id = (os.getenv("EXPLAINER_JOBS_WORKER_ID") or f"explainer-jobs-{os.getpid()}").strip()
    lease_seconds = _coerce_int(os.getenv("EXPLAINER_JOBS_LEASE_SECONDS") or os.getenv("JOBS_LEASE_SECONDS"), 60)
    renew_jitter = _coerce_int(
        os.getenv("EXPLAINER_JOBS_RENEW_JITTER_SECONDS") or os.getenv("JOBS_LEASE_RENEW_JITTER_SECONDS"),
        5,
    )
    renew_threshold = _coerce_int(
        os.getenv("EXPLAINER_JOBS_RENEW_THRESHOLD_SECONDS") or os.getenv("JOBS_LEASE_RENEW_THRESHOLD_SECONDS"),
        10,
    )
    cfg = WorkerConfig(
        domain=EXPLAINER_DOMAIN,
        queue=EXPLAINER_QUEUE,
        worker_id=worker_id,
        lease_seconds=lease_seconds,
        renew_jitter_seconds=renew_jitter,
        renew_threshold_seconds=renew_threshold,
    )
    sdk = WorkerSDK(_jobs_manager(), cfg)
    stop_task: asyncio.Task[None] | None = None

    if stop_event is not None:

        async def _watch_stop() -> None:
            await stop_event.wait()
            sdk.stop()

        stop_task = asyncio.create_task(_watch_stop())

    handler = build_explainer_job_handler()

    logger.info("Explainer Jobs worker starting: queue={} worker_id={}", EXPLAINER_QUEUE, worker_id)
    try:
        await sdk.run(handler=handler)
    finally:
        if stop_task is not None and not stop_task.done():
            stop_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await stop_task


def build_explainer_job_handler(
    *,
    db_path_resolver: Callable[[str], Path] | None = None,
    generator_factory: Callable[[], ExplainerGenerator] = make_configured_explainer_generator,
    retriever_factory: Callable[[], ExplainerRetriever | None] | None = None,
) -> Callable[[dict[str, Any]], Any]:
    """Build the concrete job handler used by the worker loop."""

    resolved_db_path = db_path_resolver or _explainer_db_path_for_owner

    async def _handler(job: dict[str, Any]) -> dict[str, Any]:
        owner_user_id = str(job.get("owner_user_id") or "").strip()
        if not owner_user_id:
            raise ValueError("Explainer job is missing owner_user_id")
        configured_generator: ExplainerGenerator | None = None

        def _lazy_generator(prompt):
            nonlocal configured_generator
            if configured_generator is None:
                configured_generator = generator_factory()
            return configured_generator(prompt)

        with open_explainer_db(owner_user_id, db_path=resolved_db_path(owner_user_id)) as db:
            retriever = retriever_factory() if retriever_factory is not None else None
            return await handle_explainer_node_expansion_job(
                job,
                repo=ExplainerRepository(db),
                generator=_lazy_generator,
                retriever=retriever,
            )

    return _handler


def _explainer_db_path_for_owner(owner_user_id: str) -> Path:
    return DatabasePaths.get_explainer_db_path(int(owner_user_id))


if __name__ == "__main__":
    asyncio.run(run_explainer_jobs_worker())
