"""
Data tables jobs worker shutdown helper extracted from the application lifespan.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable

from loguru import logger


@dataclass
class DataTablesJobsShutdownHandles:
    """Updated data tables jobs worker handles after shutdown processing."""

    data_tables_jobs_task: Any | None = None
    data_tables_jobs_stop_event: Any | None = None


async def shutdown_data_tables_jobs_worker(
    *,
    data_tables_jobs_task: Any | None,
    data_tables_jobs_stop_event: Any | None,
    should_run_late_stop: Callable[[str, Any], bool],
    guard_exceptions: tuple[type[BaseException], ...],
) -> DataTablesJobsShutdownHandles:
    """Stop the data tables jobs worker while preserving legacy late-stop semantics."""
    await _shutdown_data_tables_jobs_worker(
        task=data_tables_jobs_task,
        stop_event=data_tables_jobs_stop_event,
        should_run_late_stop=should_run_late_stop,
        guard_exceptions=guard_exceptions,
    )
    return DataTablesJobsShutdownHandles(
        data_tables_jobs_task=data_tables_jobs_task,
        data_tables_jobs_stop_event=data_tables_jobs_stop_event,
    )


async def _shutdown_data_tables_jobs_worker(
    *,
    task: Any | None,
    stop_event: Any | None,
    should_run_late_stop: Callable[[str, Any], bool],
    guard_exceptions: tuple[type[BaseException], ...],
) -> None:
    if not should_run_late_stop("data_tables_jobs_task", task):
        return
    if stop_event:
        try:
            stop_event.set()
            await _wait_for_task(task, timeout=5.0)
            logger.info("Data Tables Jobs worker stopped via stop_event")
        except guard_exceptions:
            task.cancel()
    else:
        task.cancel()


async def _wait_for_task(task: Any, *, timeout: float) -> Any:
    return await asyncio.wait_for(task, timeout=timeout)
