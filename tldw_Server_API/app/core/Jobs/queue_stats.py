"""Job-queue statistics for one (domain, queue[, owner]) slice of the jobs table.

Moved out of the Prompt Studio status endpoint (TASK-13317). The SQL and connection
handling live in ``DB_Management/jobs_queue_stats_repository.py``; this module turns
the raw aggregates into typed numbers.

Conversion contract: NULL (an empty aggregate) is 0. A value that is present but not
numeric is corruption, not zero: it is logged by metric name and the ValueError
propagates, so callers report a failed read instead of a healthy-looking empty queue.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from tldw_Server_API.app.core.DB_Management.jobs_queue_stats_repository import JobsQueueStatsRepository
from tldw_Server_API.app.core.Jobs.manager import JobManager

_CONVERSION_ERRORS = (TypeError, ValueError, OverflowError)


def _to_int(value: Any, metric: str) -> int:
    """Convert an aggregate count; NULL is 0, anything non-numeric raises ValueError."""
    if value is None:
        return 0
    try:
        return int(value)
    except _CONVERSION_ERRORS as exc:
        logger.warning("Jobs queue stats: non-numeric value for {}", metric)
        raise ValueError(f"non-numeric jobs queue stat: {metric}") from exc  # noqa: TRY003


def _to_float(value: Any, metric: str) -> float:
    """Convert an aggregate measure; NULL is 0.0, anything non-numeric raises ValueError."""
    if value is None:
        return 0.0
    try:
        return float(value)
    except _CONVERSION_ERRORS as exc:
        logger.warning("Jobs queue stats: non-numeric value for {}", metric)
        raise ValueError(f"non-numeric jobs queue stat: {metric}") from exc  # noqa: TRY003


def get_by_status(
    jm: JobManager,
    *,
    domain: str,
    queue: str,
    owner_user_id: str | None,
) -> dict[str, int]:
    """Return ``{status: job count}`` for the slice."""
    rows = JobsQueueStatsRepository(jm).count_by_status(domain=domain, queue=queue, owner_user_id=owner_user_id)
    return {str(row["status"]): _to_int(row["c"], "by_status") for row in rows if row.get("status")}


def get_by_type_and_status(
    jm: JobManager,
    *,
    domain: str,
    queue: str,
    owner_user_id: str | None,
) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    """Return per-job-type (total, queued, processing) counts for the slice."""
    rows = JobsQueueStatsRepository(jm).count_by_type_and_status(
        domain=domain, queue=queue, owner_user_id=owner_user_id
    )
    totals: dict[str, int] = {}
    queued: dict[str, int] = {}
    processing: dict[str, int] = {}
    for row in rows:
        if not row.get("job_type"):
            continue
        job_type = str(row["job_type"])
        count = _to_int(row["c"], "by_type")
        totals[job_type] = totals.get(job_type, 0) + count
        if row.get("status") == "queued":
            queued[job_type] = queued.get(job_type, 0) + count
        if row.get("status") == "processing":
            processing[job_type] = processing.get(job_type, 0) + count
    return totals, queued, processing


def get_avg_processing_time_seconds(
    jm: JobManager,
    *,
    domain: str,
    queue: str,
    owner_user_id: str | None,
) -> float:
    """Return mean seconds from start to completion over completed jobs (0.0 if none)."""
    value = JobsQueueStatsRepository(jm).avg_processing_seconds(
        domain=domain, queue=queue, owner_user_id=owner_user_id
    )
    return _to_float(value, "avg_processing_time_seconds")


def get_success_rate(
    jm: JobManager,
    *,
    domain: str,
    queue: str,
    owner_user_id: str | None,
) -> float:
    """Return the completed share of finished jobs as a percentage (0.0 if none finished)."""
    value = JobsQueueStatsRepository(jm).success_rate(domain=domain, queue=queue, owner_user_id=owner_user_id)
    return _to_float(value, "success_rate")


def get_lease_stats(
    jm: JobManager,
    *,
    domain: str,
    queue: str,
    owner_user_id: str | None,
    warn_seconds: int,
) -> dict[str, int]:
    """Return active / expiring-soon / stale-processing lease counts.

    ``warn_seconds`` is clamped to 1..3600 before it reaches the query.
    """
    warn_seconds = max(1, min(3600, int(warn_seconds)))
    row = JobsQueueStatsRepository(jm).lease_counts(
        domain=domain, queue=queue, owner_user_id=owner_user_id, warn_seconds=warn_seconds
    )
    return {key: _to_int(row.get(key), f"leases.{key}") for key in ("active", "expiring_soon", "stale_processing")}
