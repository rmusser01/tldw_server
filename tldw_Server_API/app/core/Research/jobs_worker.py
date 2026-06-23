"""Worker entrypoint for deep research Jobs slices."""

from __future__ import annotations

import asyncio
import contextlib
import os
from pathlib import Path

from loguru import logger

from tldw_Server_API.app.core.Jobs.worker_sdk import WorkerConfig, WorkerSDK
from tldw_Server_API.app.core.Jobs.worker_utils import coerce_int as _coerce_int
from tldw_Server_API.app.core.Jobs.worker_utils import jobs_manager_from_env as _jobs_manager
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths

from .jobs import RESEARCH_DOMAIN, RESEARCH_QUEUE, handle_research_phase_job


async def run_research_jobs_worker(
    stop_event: asyncio.Event | None = None,
    *,
    research_db_path: str | Path | None = None,
    outputs_dir: str | Path | None = None,
) -> None:
    """Run the deep research worker loop until stopped."""
    worker_id = (os.getenv("RESEARCH_JOBS_WORKER_ID") or f"research-jobs-{os.getpid()}").strip()
    lease_seconds = _coerce_int(os.getenv("RESEARCH_JOBS_LEASE_SECONDS") or os.getenv("JOBS_LEASE_SECONDS"), 60)
    renew_jitter = _coerce_int(os.getenv("RESEARCH_JOBS_RENEW_JITTER_SECONDS") or os.getenv("JOBS_LEASE_RENEW_JITTER_SECONDS"), 5)
    renew_threshold = _coerce_int(os.getenv("RESEARCH_JOBS_RENEW_THRESHOLD_SECONDS") or os.getenv("JOBS_LEASE_RENEW_THRESHOLD_SECONDS"), 10)
    cfg = WorkerConfig(
        domain=RESEARCH_DOMAIN,
        queue=RESEARCH_QUEUE,
        worker_id=worker_id,
        lease_seconds=lease_seconds,
        renew_jitter_seconds=renew_jitter,
        renew_threshold_seconds=renew_threshold,
    )
    sdk = WorkerSDK(_jobs_manager(), cfg)
    _stop_watcher_task: asyncio.Task[None] | None = None

    if stop_event is not None:
        async def _watch_stop() -> None:
            await stop_event.wait()
            sdk.stop()

        _stop_watcher_task = asyncio.create_task(_watch_stop())

    research_db_path_override = research_db_path or os.getenv("RESEARCH_SESSIONS_DB_PATH")
    outputs_dir_override = outputs_dir or os.getenv("RESEARCH_OUTPUTS_DIR")
    resolved_research_db_path = Path(research_db_path_override) if research_db_path_override else None
    resolved_outputs_dir = Path(outputs_dir_override) if outputs_dir_override else None

    def _paths_for_job(job: dict[str, object]) -> tuple[Path, Path]:
        payload = job.get("payload") if isinstance(job.get("payload"), dict) else {}
        owner_user_id = str(job.get("owner_user_id") or payload.get("owner_user_id") or "").strip()
        if (resolved_research_db_path is None or resolved_outputs_dir is None) and not owner_user_id:
            raise ValueError("missing research owner_user_id")
        research_path = resolved_research_db_path or DatabasePaths.get_research_sessions_db_path(owner_user_id)
        output_path = resolved_outputs_dir or DatabasePaths.get_user_outputs_dir(owner_user_id)
        return research_path, output_path

    async def _cancel_check(job: dict[str, object]) -> bool:
        return bool(job.get("cancel_requested_at") or job.get("cancelled_at"))

    async def _handler(job: dict[str, object]) -> dict[str, object]:
        research_path, output_path = _paths_for_job(job)
        return await handle_research_phase_job(
            job,
            research_db_path=research_path,
            outputs_dir=output_path,
        )

    logger.info("Research Jobs worker starting: queue={} worker_id={}", RESEARCH_QUEUE, worker_id)
    try:
        await sdk.run(handler=_handler, cancel_check=_cancel_check)
    finally:
        if _stop_watcher_task is not None and not _stop_watcher_task.done():
            _stop_watcher_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await _stop_watcher_task


if __name__ == "__main__":
    asyncio.run(run_research_jobs_worker())
