"""SQLite-backed Jobs single-job lifecycle operations."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
from typing import Any

from tldw_Server_API.app.core.DB_Management.sqlite_policy import (
    begin_immediate_if_needed,
)
from tldw_Server_API.app.core.Jobs.operations.contracts import (
    AcquireJobCommand,
    BatchRenewLeasesCommand,
    BatchRenewLeasesResult,
    LifecycleResult,
    NoTransitionReason,
    ReleaseJobCommand,
    RenewLeaseCommand,
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

_RENEW_SQL_VARIANTS = {
    (False, False, False): (
        "UPDATE jobs SET leased_until = "
        "MAX(COALESCE(leased_until, DATETIME(?)), DATETIME(?, '+' || ? || ' seconds')) "
        "WHERE id = ? AND status = 'processing'"
    ),
    (False, True, False): (
        "UPDATE jobs SET leased_until = "
        "MAX(COALESCE(leased_until, DATETIME(?)), DATETIME(?, '+' || ? || ' seconds')), "
        "progress_percent = ? WHERE id = ? AND status = 'processing'"
    ),
    (False, False, True): (
        "UPDATE jobs SET leased_until = "
        "MAX(COALESCE(leased_until, DATETIME(?)), DATETIME(?, '+' || ? || ' seconds')), "
        "progress_message = ? WHERE id = ? AND status = 'processing'"
    ),
    (False, True, True): (
        "UPDATE jobs SET leased_until = "
        "MAX(COALESCE(leased_until, DATETIME(?)), DATETIME(?, '+' || ? || ' seconds')), "
        "progress_percent = ?, progress_message = ? "
        "WHERE id = ? AND status = 'processing'"
    ),
    (True, False, False): (
        "UPDATE jobs SET leased_until = "
        "MAX(COALESCE(leased_until, DATETIME(?)), DATETIME(?, '+' || ? || ' seconds')) "
        "WHERE id = ? AND status = 'processing' AND worker_id = ? AND lease_id = ?"
    ),
    (True, True, False): (
        "UPDATE jobs SET leased_until = "
        "MAX(COALESCE(leased_until, DATETIME(?)), DATETIME(?, '+' || ? || ' seconds')), "
        "progress_percent = ? "
        "WHERE id = ? AND status = 'processing' AND worker_id = ? AND lease_id = ?"
    ),
    (True, False, True): (
        "UPDATE jobs SET leased_until = "
        "MAX(COALESCE(leased_until, DATETIME(?)), DATETIME(?, '+' || ? || ' seconds')), "
        "progress_message = ? "
        "WHERE id = ? AND status = 'processing' AND worker_id = ? AND lease_id = ?"
    ),
    (True, True, True): (
        "UPDATE jobs SET leased_until = "
        "MAX(COALESCE(leased_until, DATETIME(?)), DATETIME(?, '+' || ? || ' seconds')), "
        "progress_percent = ?, progress_message = ? "
        "WHERE id = ? AND status = 'processing' AND worker_id = ? AND lease_id = ?"
    ),
}

_RELEASE_SQL = (
    "UPDATE jobs SET status = 'queued', available_at = NULL, leased_until = NULL, "
    "worker_id = NULL, lease_id = NULL, acquired_at = NULL, started_at = NULL, "
    "completion_token = NULL, updated_at = DATETIME('now') "
    "WHERE id = ? AND status = 'processing'"
)

_RELEASE_ENFORCED_SQL = (
    "UPDATE jobs SET status = 'queued', available_at = NULL, leased_until = NULL, "
    "worker_id = NULL, lease_id = NULL, acquired_at = NULL, started_at = NULL, "
    "completion_token = NULL, updated_at = DATETIME('now') "
    "WHERE id = ? AND status = 'processing' AND worker_id = ? AND lease_id = ?"
)

_RELEASE_COUNTER_SQL = (
    "INSERT INTO job_counters(domain, queue, job_type, ready_count, scheduled_count, "
    "processing_count, quarantined_count) VALUES(?, ?, ?, 1, 0, 0, 0) "
    "ON CONFLICT(domain, queue, job_type) DO UPDATE SET "
    "ready_count = ready_count + 1, "
    "processing_count = CASE WHEN processing_count > 0 THEN processing_count - 1 ELSE 0 END, "
    "updated_at = DATETIME('now')"
)


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


def _classify_lifecycle_no_transition(
    conn: sqlite3.Connection,
    *,
    job_id: int,
    enforce: bool,
    worker_id: str | None,
    lease_id: str | None,
) -> LifecycleResult:
    """Classify why a SQLite lifecycle update did not change a row."""

    row = conn.execute(
        "SELECT id, status, worker_id, lease_id FROM jobs WHERE id = ?",
        (job_id,),
    ).fetchone()
    if row is None:
        return LifecycleResult.no_transition(NoTransitionReason.MISSING)
    current = dict(row)
    if current.get("status") != "processing":
        return LifecycleResult.no_transition(NoTransitionReason.WRONG_STATUS, row=current)
    if enforce and (
        current.get("worker_id") != worker_id or current.get("lease_id") != lease_id
    ):
        return LifecycleResult.no_transition(NoTransitionReason.STALE_LEASE, row=current)
    return LifecycleResult.no_transition(NoTransitionReason.WRONG_STATUS, row=current)


def _renew_lease_statement(
    command: RenewLeaseCommand,
    *,
    now: datetime,
) -> tuple[str, tuple[Any, ...]]:
    """Build the SQLite statement for one lease renewal attempt."""

    now_sql = _sqlite_timestamp(now)
    has_percent = command.progress_percent is not None
    has_message = command.progress_message is not None
    sql = _RENEW_SQL_VARIANTS[(command.enforce, has_percent, has_message)]
    params: list[Any] = [now_sql, now_sql, command.seconds]
    if has_percent:
        params.append(float(command.progress_percent))
    if has_message:
        params.append(str(command.progress_message))
    params.append(command.job_id)
    if command.enforce:
        params.extend((command.worker_id, command.lease_id))
    return sql, tuple(params)


@contextmanager
def _batch_renew_transaction(conn: sqlite3.Connection) -> Iterator[None]:
    """Own a transaction or isolate the batch inside the caller's transaction."""

    if not conn.in_transaction:
        with conn:
            yield
        return

    conn.execute("SAVEPOINT jobs_batch_renew_leases")
    try:
        yield
    except BaseException:
        conn.execute("ROLLBACK TO SAVEPOINT jobs_batch_renew_leases")
        conn.execute("RELEASE SAVEPOINT jobs_batch_renew_leases")
        raise
    else:
        conn.execute("RELEASE SAVEPOINT jobs_batch_renew_leases")


def renew_lease(
    conn: sqlite3.Connection,
    *,
    command: RenewLeaseCommand,
    now: datetime,
) -> LifecycleResult:
    """Renew one processing SQLite job lease without shortening it."""

    sql, params = _renew_lease_statement(command, now=now)

    transaction = nullcontext(conn) if conn.in_transaction else conn
    with transaction:
        changed = conn.execute(sql, params)
        if changed.rowcount != 1:
            return _classify_lifecycle_no_transition(
                conn,
                job_id=command.job_id,
                enforce=command.enforce,
                worker_id=command.worker_id,
                lease_id=command.lease_id,
            )
        row = conn.execute(
            (
                "SELECT id, leased_until, progress_percent, progress_message "
                "FROM jobs WHERE id = ?"
            ),
            (command.job_id,),
        ).fetchone()
        if row is None:
            return LifecycleResult.no_transition(NoTransitionReason.MISSING)
        return LifecycleResult.applied(row=dict(row))


def renew_leases_batch(
    conn: sqlite3.Connection,
    *,
    command: BatchRenewLeasesCommand,
    clock: Callable[[], datetime],
) -> BatchRenewLeasesResult:
    """Renew an ordered SQLite lease batch in one transaction."""

    applied_count = 0
    with _batch_renew_transaction(conn):
        for item in command.items:
            item_command = RenewLeaseCommand(
                job_id=item.job_id,
                seconds=item.seconds,
                enforce=command.enforce,
                worker_id=item.worker_id,
                lease_id=item.lease_id,
            )
            sql, params = _renew_lease_statement(item_command, now=clock())
            changed = conn.execute(sql, params)
            applied_count += int(changed.rowcount or 0)
        return BatchRenewLeasesResult(
            requested_count=len(command.items),
            applied_count=applied_count,
        )


def release_job(
    conn: sqlite3.Connection,
    *,
    command: ReleaseJobCommand,
    counters_enabled: bool,
) -> LifecycleResult:
    """Release one processing SQLite job back to the ready queue."""

    owns_transaction = begin_immediate_if_needed(conn)
    transaction = conn if owns_transaction else nullcontext(conn)
    with transaction:
        selected = conn.execute(
            (
                "SELECT id, domain, queue, job_type, status, worker_id, lease_id "
                "FROM jobs WHERE id = ?"
            ),
            (command.job_id,),
        ).fetchone()
        if selected is None:
            return LifecycleResult.no_transition(NoTransitionReason.MISSING)
        current = dict(selected)
        if current.get("status") != "processing":
            return LifecycleResult.no_transition(NoTransitionReason.WRONG_STATUS, row=current)
        if command.enforce and (
            current.get("worker_id") != command.worker_id
            or current.get("lease_id") != command.lease_id
        ):
            return LifecycleResult.no_transition(NoTransitionReason.STALE_LEASE, row=current)

        if command.enforce:
            changed = conn.execute(
                _RELEASE_ENFORCED_SQL,
                (command.job_id, command.worker_id, command.lease_id),
            )
        else:
            changed = conn.execute(_RELEASE_SQL, (command.job_id,))
        if changed.rowcount != 1:
            return _classify_lifecycle_no_transition(
                conn,
                job_id=command.job_id,
                enforce=command.enforce,
                worker_id=command.worker_id,
                lease_id=command.lease_id,
            )
        if counters_enabled:
            conn.execute(
                _RELEASE_COUNTER_SQL,
                (current["domain"], current["queue"], current["job_type"]),
            )
        row = conn.execute(
            (
                "SELECT id, domain, queue, job_type, status, available_at, leased_until, "
                "worker_id, lease_id, acquired_at, started_at, completion_token, updated_at "
                "FROM jobs WHERE id = ?"
            ),
            (command.job_id,),
        ).fetchone()
        if row is None:
            return LifecycleResult.no_transition(NoTransitionReason.MISSING)
        return LifecycleResult.applied(row=dict(row))


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
