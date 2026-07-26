"""SQLite-backed Jobs single-job lifecycle operations."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Any

from tldw_Server_API.app.core.Jobs.operations.contracts import (
    AcquireJobCommand,
    LifecycleResult,
    NoTransitionReason,
)

_ORDER_CLAUSES = {
    ("ASC", "ASC"): (
        " ORDER BY priority ASC, COALESCE(available_at, created_at) ASC, id ASC LIMIT 1"
    ),
    ("ASC", "DESC"): (
        " ORDER BY priority ASC, COALESCE(available_at, created_at) DESC, id DESC LIMIT 1"
    ),
    ("DESC", "ASC"): (
        " ORDER BY priority DESC, COALESCE(available_at, created_at) ASC, id ASC LIMIT 1"
    ),
    ("DESC", "DESC"): (
        " ORDER BY priority DESC, COALESCE(available_at, created_at) DESC, id DESC LIMIT 1"
    ),
}


def _sqlite_timestamp(value: datetime) -> str:
    """Return the UTC timestamp representation used by the Jobs table."""

    normalized = value
    if value.tzinfo is not None:
        normalized = value.astimezone(timezone.utc).replace(tzinfo=None)
    return normalized.strftime("%Y-%m-%d %H:%M:%S")


def _dependency_condition() -> str:
    return (
        " AND (status != 'queued' OR NOT EXISTS ("
        "SELECT 1 FROM job_dependencies jd "
        "LEFT JOIN jobs dep ON dep.uuid = jd.depends_on_job_uuid "
        "WHERE jd.job_uuid = jobs.uuid AND "
        "COALESCE(dep.status, jd.depends_on_terminal_status, 'missing') <> 'completed'"
        "))"
    )


def _candidate_sql(
    conn: sqlite3.Connection,
    *,
    command: AcquireJobCommand,
    now_sql: str,
) -> tuple[str, list[Any]]:
    sql = (
        "SELECT id FROM jobs WHERE domain = ? AND queue = ? "
        "AND status = 'queued' AND (available_at IS NULL OR available_at <= DATETIME(?))"
    )
    sql += _dependency_condition()
    params: list[Any] = [command.domain, command.queue, now_sql]
    if command.owner_user_id:
        sql += " AND owner_user_id = ?"
        params.append(command.owner_user_id)
    if command.job_type:
        sql += " AND job_type = ?"
        params.append(command.job_type)

    if command.tie_break == "fifo":
        tie_direction = "ASC"
    elif command.tie_break == "lifo":
        tie_direction = "DESC"
    elif command.single_update or command.domain != "chatbooks":
        tie_direction = "ASC"
    else:
        scheduled_sql = (
            "SELECT 1 FROM jobs WHERE domain=? AND queue=? "
            "AND status='queued' AND available_at IS NOT NULL AND available_at > DATETIME(?)"
        )
        scheduled_params: list[Any] = [command.domain, command.queue, now_sql]
        if command.job_type:
            scheduled_sql += " AND job_type=?"
            scheduled_params.append(command.job_type)
        scheduled_sql += " LIMIT 1"
        tie_direction = "ASC" if conn.execute(scheduled_sql, scheduled_params).fetchone() else "DESC"

    sql += _ORDER_CLAUSES[(command.priority_direction, tie_direction)]
    return sql, params


def _bump_acquired_counters(conn: sqlite3.Connection, *, acquired: dict[str, Any]) -> None:
    is_scheduled = acquired.get("available_at") is not None
    conn.execute(
        (
            "INSERT INTO job_counters(domain,queue,job_type,ready_count,scheduled_count,processing_count,"
            "quarantined_count) VALUES(?,?,?,?,?,?,?) ON CONFLICT(domain,queue,job_type) DO UPDATE SET "
            "ready_count = MAX(ready_count + ?, 0), "
            "scheduled_count = MAX(scheduled_count + ?, 0), "
            "processing_count = processing_count + 1, updated_at = DATETIME('now')"
        ),
        (
            acquired.get("domain"),
            acquired.get("queue"),
            acquired.get("job_type"),
            0,
            0,
            1,
            0,
            -1 if not is_scheduled else 0,
            -1 if is_scheduled else 0,
        ),
    )


def acquire_job(
    conn: sqlite3.Connection,
    *,
    command: AcquireJobCommand,
    counters_enabled: bool,
    now: datetime,
) -> LifecycleResult:
    """Atomically acquire the next eligible SQLite job."""

    now_sql = _sqlite_timestamp(now)
    with conn:
        # Selection and transition must exclude concurrent dependency writers.
        conn.execute("BEGIN IMMEDIATE")
        if command.max_inflight_quota and command.owner_user_id:
            inflight_row = conn.execute(
                (
                    "SELECT COUNT(*) FROM jobs WHERE domain=? AND owner_user_id=? AND status='processing' "
                    "AND leased_until IS NOT NULL AND leased_until > DATETIME(?)"
                ),
                (command.domain, command.owner_user_id, now_sql),
            ).fetchone()
            if int(inflight_row[0] or 0) >= command.max_inflight_quota:
                return LifecycleResult.no_transition(NoTransitionReason.NO_ELIGIBLE_JOB)

        candidate_sql, candidate_params = _candidate_sql(conn, command=command, now_sql=now_sql)
        job_id: int | None = None
        if command.single_update:
            update_sql = "".join(
                [
                    "UPDATE jobs SET status='processing', ",
                    "retry_count = CASE WHEN status='processing' THEN retry_count + 1 ELSE retry_count END, ",
                    "started_at = COALESCE(started_at, DATETIME(?)), ",
                    "acquired_at = COALESCE(acquired_at, DATETIME(?)), ",
                    "leased_until = DATETIME(?, ?), worker_id = ?, lease_id = ?, completion_token = NULL ",
                    "WHERE id IN (",
                    candidate_sql,
                    ")",
                ]
            )
            changed = conn.execute(
                update_sql,
                (
                    now_sql,
                    now_sql,
                    now_sql,
                    f"+{command.lease_seconds} seconds",
                    command.worker_id,
                    command.lease_id,
                    *candidate_params,
                ),
            )
            if changed.rowcount != 1:
                return LifecycleResult.no_transition(NoTransitionReason.NO_ELIGIBLE_JOB)
            row = conn.execute(
                "SELECT * FROM jobs WHERE lease_id = ?",
                (command.lease_id,),
            ).fetchone()
        else:
            candidate = conn.execute(candidate_sql, candidate_params).fetchone()
            if not candidate:
                return LifecycleResult.no_transition(NoTransitionReason.NO_ELIGIBLE_JOB)
            job_id = int(candidate[0])
            changed = conn.execute(
                (
                    "UPDATE jobs SET status = 'processing', "
                    "retry_count = CASE WHEN status = 'processing' THEN retry_count + 1 ELSE retry_count END, "
                    "started_at = COALESCE(started_at, DATETIME(?)), "
                    "acquired_at = COALESCE(acquired_at, DATETIME(?)), "
                    "leased_until = DATETIME(?, ?), worker_id = ?, lease_id = ?, completion_token = NULL "
                    "WHERE id = ? AND status = 'queued'"
                ),
                (
                    now_sql,
                    now_sql,
                    now_sql,
                    f"+{command.lease_seconds} seconds",
                    command.worker_id,
                    command.lease_id,
                    job_id,
                ),
            )
            if changed.rowcount != 1:
                return LifecycleResult.no_transition(NoTransitionReason.NO_ELIGIBLE_JOB)
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()

        if not row:
            return LifecycleResult.no_transition(NoTransitionReason.NO_ELIGIBLE_JOB)
        acquired = dict(row)
        if counters_enabled:
            _bump_acquired_counters(conn, acquired=acquired)
        return LifecycleResult.applied(row=acquired)
