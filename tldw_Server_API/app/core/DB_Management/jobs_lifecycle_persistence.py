"""Lifecycle persistence inside the caller-owned Jobs transaction."""

from __future__ import annotations

import json
from typing import Any


def _insert_lifecycle_event(
    executor: Any,
    *,
    backend: str,
    event_type: str,
    job: dict[str, Any],
    attrs: dict[str, Any] | None = None,
) -> None:
    """Append a durable lifecycle event inside the caller's transaction."""

    values = (
        int(job["id"]),
        job.get("domain"),
        job.get("queue"),
        job.get("job_type"),
        event_type,
        json.dumps(attrs or {}),
        job.get("owner_user_id"),
        job.get("request_id"),
        job.get("trace_id"),
    )
    if backend == "postgres":
        executor.execute(
            (
                "INSERT INTO job_events(job_id, domain, queue, job_type, event_type, "
                "attrs_json, owner_user_id, request_id, trace_id, created_at) "
                "VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, NOW())"
            ),
            values,
        )
        return
    executor.execute(
        (
            "INSERT INTO job_events(job_id, domain, queue, job_type, event_type, "
            "attrs_json, owner_user_id, request_id, trace_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, DATETIME('now'))"
        ),
        values,
    )


def _reconcile_lifecycle_counter_row(
    executor: Any,
    *,
    backend: str,
    domain: Any,
    queue: Any,
    job_type: Any,
) -> None:
    """Rebuild one missing counter row from transaction-local job state."""

    params = (domain, queue, job_type)
    if backend == "postgres":
        executor.execute(
            (
                "SELECT "
                "COUNT(*) FILTER (WHERE status='queued' AND available_at IS NULL) AS ready_count, "
                "COUNT(*) FILTER (WHERE status='queued' AND available_at IS NOT NULL) AS scheduled_count, "
                "COUNT(*) FILTER (WHERE status='processing') AS processing_count, "
                "COUNT(*) FILTER (WHERE status='quarantined') AS quarantined_count "
                "FROM jobs WHERE domain=%s AND queue=%s AND job_type=%s"
            ),
            params,
        )
        row = executor.fetchone()
        if isinstance(row, dict):
            counts = tuple(
                int(row.get(name) or 0)
                for name in (
                    "ready_count",
                    "scheduled_count",
                    "processing_count",
                    "quarantined_count",
                )
            )
        else:
            counts = tuple(int(value or 0) for value in (row or (0, 0, 0, 0)))
        executor.execute(
            (
                "INSERT INTO job_counters(domain,queue,job_type,ready_count,scheduled_count,processing_count,quarantined_count) "
                "VALUES(%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (domain,queue,job_type) DO UPDATE SET "
                "ready_count=EXCLUDED.ready_count, scheduled_count=EXCLUDED.scheduled_count, "
                "processing_count=EXCLUDED.processing_count, quarantined_count=EXCLUDED.quarantined_count, "
                "updated_at=NOW()"
            ),
            (*params, *counts),
        )
        return

    row = executor.execute(
        (
            "SELECT "
            "COALESCE(SUM(CASE WHEN status='queued' AND available_at IS NULL THEN 1 ELSE 0 END),0), "
            "COALESCE(SUM(CASE WHEN status='queued' AND available_at IS NOT NULL THEN 1 ELSE 0 END),0), "
            "COALESCE(SUM(CASE WHEN status='processing' THEN 1 ELSE 0 END),0), "
            "COALESCE(SUM(CASE WHEN status='quarantined' THEN 1 ELSE 0 END),0) "
            "FROM jobs WHERE domain=? AND queue=? AND job_type=?"
        ),
        params,
    ).fetchone()
    counts = tuple(int(value or 0) for value in (row or (0, 0, 0, 0)))
    executor.execute(
        (
            "INSERT INTO job_counters(domain,queue,job_type,ready_count,scheduled_count,processing_count,quarantined_count) "
            "VALUES(?,?,?,?,?,?,?) ON CONFLICT(domain,queue,job_type) DO UPDATE SET "
            "ready_count=excluded.ready_count, scheduled_count=excluded.scheduled_count, "
            "processing_count=excluded.processing_count, quarantined_count=excluded.quarantined_count, "
            "updated_at=DATETIME('now')"
        ),
        (*params, *counts),
    )
