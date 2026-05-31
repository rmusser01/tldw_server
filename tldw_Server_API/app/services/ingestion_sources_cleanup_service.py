from __future__ import annotations

import asyncio
import os

from loguru import logger

from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
from tldw_Server_API.app.core.Ingestion_Sources.archive_snapshot import (
    prune_archive_source_retention,
)
from tldw_Server_API.app.core.Ingestion_Sources.service import (
    list_sources_for_retention_cleanup,
)
from tldw_Server_API.app.core.testing import env_flag_enabled

_NONCRITICAL_EXCEPTIONS = (
    AttributeError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)


def _cleanup_interval_seconds() -> int:
    try:
        value = int(os.getenv("INGESTION_SOURCES_CLEANUP_INTERVAL_SEC", "3600") or "3600")
    except _NONCRITICAL_EXCEPTIONS:
        value = 3600
    return max(60, value)


async def run_ingestion_sources_cleanup_once() -> dict[str, int]:
    pool = await get_db_pool()
    async with pool.transaction() as db:
        rows = await list_sources_for_retention_cleanup(db)

    results = {
        "scanned": len(rows),
        "pruned": 0,
        "skipped_active": 0,
        "failed": 0,
    }

    for source in rows:
        source_id = int(source["id"])
        if str(source.get("active_job_id") or "").strip():
            results["skipped_active"] += 1
            continue
        try:
            async with pool.transaction() as db:
                await prune_archive_source_retention(db, source_id=source_id)
            results["pruned"] += 1
        except _NONCRITICAL_EXCEPTIONS as exc:
            results["failed"] += 1
            logger.bind(error_type=type(exc).__name__).warning(
                "Ingestion sources cleanup failed for source_id={}",
                source_id,
            )
    return results


async def run_ingestion_sources_cleanup_loop(stop_event: asyncio.Event | None = None) -> None:
    interval_sec = _cleanup_interval_seconds()
    logger.info("Starting ingestion sources cleanup worker (every {}s)", interval_sec)

    while True:
        if stop_event and stop_event.is_set():
            logger.info("Stopping ingestion sources cleanup worker on shutdown signal")
            return
        try:
            results = await run_ingestion_sources_cleanup_once()
            if any(results.values()):
                logger.info(
                    "Ingestion sources cleanup scanned={} pruned={} skipped_active={} failed={}",
                    results["scanned"],
                    results["pruned"],
                    results["skipped_active"],
                    results["failed"],
                )
        except _NONCRITICAL_EXCEPTIONS as exc:
            logger.bind(error_type=type(exc).__name__).warning("Ingestion sources cleanup loop error")
        if await _wait_for_interval_or_stop(interval_sec, stop_event):
            logger.info("Stopping ingestion sources cleanup worker on shutdown signal")
            return


async def _wait_for_interval_or_stop(
    interval_sec: int,
    stop_event: asyncio.Event | None,
) -> bool:
    """Wait for the cleanup interval or shutdown.

    Returns True when ``stop_event`` fires; False means the interval elapsed.
    """

    if stop_event is None:
        await asyncio.sleep(interval_sec)
        return False

    try:
        await asyncio.wait_for(stop_event.wait(), timeout=interval_sec)
    except asyncio.TimeoutError:
        return False
    return True


async def start_ingestion_sources_cleanup_scheduler() -> asyncio.Task | None:
    if not env_flag_enabled("INGESTION_SOURCES_CLEANUP_ENABLED"):
        return None
    task = asyncio.create_task(
        run_ingestion_sources_cleanup_loop(),
        name="ingestion-sources-cleanup-scheduler",
    )
    logger.info("Started ingestion sources cleanup scheduler: interval={}s", _cleanup_interval_seconds())
    return task
