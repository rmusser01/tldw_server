"""Postgres-backed Jobs single-job lifecycle operations."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from contextlib import AbstractContextManager
from datetime import datetime
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.Jobs.operations.contracts import (
    AcquireJobCommand,
    BatchRenewLeasesCommand,
    BatchRenewLeasesResult,
    LifecycleResult,
    NoTransitionReason,
    ReleaseJobCommand,
    RenewLeaseCommand,
)

try:
    import psycopg as _psycopg  # type: ignore
except ImportError:  # pragma: no cover - optional dependency path
    _PG_ERRORS = ()
else:
    _PG_ERRORS: tuple[type[BaseException], ...] = (_psycopg.Error,)


_COUNTER_NONCRITICAL_ERRORS: tuple[type[BaseException], ...] = (
    AttributeError,
    RuntimeError,
    TypeError,
    ValueError,
    *_PG_ERRORS,
)

_ORDER_CLAUSES = {
    ("ASC", "ASC"): (
        " ORDER BY priority ASC, COALESCE(available_at, created_at) ASC, "
        "id ASC LIMIT 1 FOR UPDATE SKIP LOCKED"
    ),
    ("ASC", "DESC"): (
        " ORDER BY priority ASC, COALESCE(available_at, created_at) DESC, "
        "id DESC LIMIT 1 FOR UPDATE SKIP LOCKED"
    ),
    ("DESC", "ASC"): (
        " ORDER BY priority DESC, COALESCE(available_at, created_at) ASC, "
        "id ASC LIMIT 1 FOR UPDATE SKIP LOCKED"
    ),
    ("DESC", "DESC"): (
        " ORDER BY priority DESC, COALESCE(available_at, created_at) DESC, "
        "id DESC LIMIT 1 FOR UPDATE SKIP LOCKED"
    ),
}

_RENEW_SQL_VARIANTS = {
    (False, False, False): (
        "UPDATE jobs SET leased_until = "
        "GREATEST(COALESCE(leased_until, %s), %s + (%s || ' seconds')::interval) "
        "WHERE id = %s AND status = 'processing' "
        "RETURNING id, leased_until, progress_percent, progress_message"
    ),
    (False, True, False): (
        "UPDATE jobs SET leased_until = "
        "GREATEST(COALESCE(leased_until, %s), %s + (%s || ' seconds')::interval), "
        "progress_percent = %s WHERE id = %s AND status = 'processing' "
        "RETURNING id, leased_until, progress_percent, progress_message"
    ),
    (False, False, True): (
        "UPDATE jobs SET leased_until = "
        "GREATEST(COALESCE(leased_until, %s), %s + (%s || ' seconds')::interval), "
        "progress_message = %s WHERE id = %s AND status = 'processing' "
        "RETURNING id, leased_until, progress_percent, progress_message"
    ),
    (False, True, True): (
        "UPDATE jobs SET leased_until = "
        "GREATEST(COALESCE(leased_until, %s), %s + (%s || ' seconds')::interval), "
        "progress_percent = %s, progress_message = %s "
        "WHERE id = %s AND status = 'processing' "
        "RETURNING id, leased_until, progress_percent, progress_message"
    ),
    (True, False, False): (
        "UPDATE jobs SET leased_until = "
        "GREATEST(COALESCE(leased_until, %s), %s + (%s || ' seconds')::interval) "
        "WHERE id = %s AND status = 'processing' AND worker_id = %s AND lease_id = %s "
        "RETURNING id, leased_until, progress_percent, progress_message"
    ),
    (True, True, False): (
        "UPDATE jobs SET leased_until = "
        "GREATEST(COALESCE(leased_until, %s), %s + (%s || ' seconds')::interval), "
        "progress_percent = %s "
        "WHERE id = %s AND status = 'processing' AND worker_id = %s AND lease_id = %s "
        "RETURNING id, leased_until, progress_percent, progress_message"
    ),
    (True, False, True): (
        "UPDATE jobs SET leased_until = "
        "GREATEST(COALESCE(leased_until, %s), %s + (%s || ' seconds')::interval), "
        "progress_message = %s "
        "WHERE id = %s AND status = 'processing' AND worker_id = %s AND lease_id = %s "
        "RETURNING id, leased_until, progress_percent, progress_message"
    ),
    (True, True, True): (
        "UPDATE jobs SET leased_until = "
        "GREATEST(COALESCE(leased_until, %s), %s + (%s || ' seconds')::interval), "
        "progress_percent = %s, progress_message = %s "
        "WHERE id = %s AND status = 'processing' AND worker_id = %s AND lease_id = %s "
        "RETURNING id, leased_until, progress_percent, progress_message"
    ),
}

_RELEASE_SQL = (
    "UPDATE jobs SET status = 'queued', available_at = NULL, leased_until = NULL, "
    "worker_id = NULL, lease_id = NULL, acquired_at = NULL, started_at = NULL, "
    "completion_token = NULL, updated_at = NOW() "
    "WHERE id = %s AND status = 'processing' "
    "RETURNING id, domain, queue, job_type, status, available_at, leased_until, "
    "worker_id, lease_id, acquired_at, started_at, completion_token, updated_at"
)

_RELEASE_ENFORCED_SQL = (
    "UPDATE jobs SET status = 'queued', available_at = NULL, leased_until = NULL, "
    "worker_id = NULL, lease_id = NULL, acquired_at = NULL, started_at = NULL, "
    "completion_token = NULL, updated_at = NOW() "
    "WHERE id = %s AND status = 'processing' AND worker_id = %s AND lease_id = %s "
    "RETURNING id, domain, queue, job_type, status, available_at, leased_until, "
    "worker_id, lease_id, acquired_at, started_at, completion_token, updated_at"
)

_RELEASE_COUNTER_SQL = (
    "INSERT INTO job_counters(domain, queue, job_type, ready_count, scheduled_count, "
    "processing_count, quarantined_count) VALUES(%s, %s, %s, 1, 0, 0, 0) "
    "ON CONFLICT (domain, queue, job_type) DO UPDATE SET "
    "ready_count = job_counters.ready_count + %s, "
    "scheduled_count = job_counters.scheduled_count + %s, "
    "processing_count = GREATEST(job_counters.processing_count - 1, 0), "
    "updated_at = NOW()"
)


def _pg_advisory_key(*parts: str) -> int:
    """Match the legacy advisory key used by deployed workers."""

    material = (":".join(["jobs"] + [part or "" for part in parts])).encode(
        "utf-8",
        "ignore",
    )
    value = int.from_bytes(
        hashlib.sha1(material, usedforsecurity=False).digest()[:8],
        "big",
        signed=False,
    )
    if value >= 2**63:
        value -= 2**63
    return int(value)


def _count_from_row(row: Any) -> int:
    if row is None:
        return 0
    if isinstance(row, dict):
        return int(row.get("c") or 0)
    return int(row[0] or 0)


def _classify_lifecycle_no_transition(
    cur: Any,
    *,
    job_id: int,
    enforce: bool,
    worker_id: str | None,
    lease_id: str | None,
) -> LifecycleResult:
    """Classify why a Postgres lifecycle update did not change a row."""

    cur.execute(
        "SELECT id, status, worker_id, lease_id FROM jobs WHERE id = %s",
        (job_id,),
    )
    row = cur.fetchone()
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
    """Build the PostgreSQL statement for one lease renewal attempt."""

    has_percent = command.progress_percent is not None
    has_message = command.progress_message is not None
    sql = _RENEW_SQL_VARIANTS[(command.enforce, has_percent, has_message)]
    params: list[Any] = [now, now, command.seconds]
    if has_percent:
        params.append(float(command.progress_percent))
    if has_message:
        params.append(str(command.progress_message))
    params.append(command.job_id)
    if command.enforce:
        params.extend((command.worker_id, command.lease_id))
    return sql, tuple(params)


def renew_lease(
    conn: Any,
    cursor_factory: Callable[[Any], AbstractContextManager[Any]],
    *,
    command: RenewLeaseCommand,
    now: datetime,
) -> LifecycleResult:
    """Renew one processing Postgres job lease without shortening it."""

    sql, params = _renew_lease_statement(command, now=now)

    with conn:
        with cursor_factory(conn) as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
            if row is None:
                return _classify_lifecycle_no_transition(
                    cur,
                    job_id=command.job_id,
                    enforce=command.enforce,
                    worker_id=command.worker_id,
                    lease_id=command.lease_id,
                )
            return LifecycleResult.applied(row=dict(row))


def renew_leases_batch(
    conn: Any,
    cursor_factory: Callable[[Any], AbstractContextManager[Any]],
    *,
    command: BatchRenewLeasesCommand,
    clock: Callable[[], datetime],
) -> BatchRenewLeasesResult:
    """Renew an ordered PostgreSQL lease batch in one transaction."""

    applied_count = 0
    with conn:
        with cursor_factory(conn) as cur:
            now = clock()
            for item in command.items:
                item_command = RenewLeaseCommand(
                    job_id=item.job_id,
                    seconds=item.seconds,
                    enforce=command.enforce,
                    worker_id=item.worker_id,
                    lease_id=item.lease_id,
                )
                sql, params = _renew_lease_statement(item_command, now=now)
                cur.execute(sql, params)
                if cur.fetchone() is not None:
                    applied_count += 1
            return BatchRenewLeasesResult(
                requested_count=len(command.items),
                applied_count=applied_count,
            )


def release_job(
    conn: Any,
    cursor_factory: Callable[[Any], AbstractContextManager[Any]],
    *,
    command: ReleaseJobCommand,
    counters_enabled: bool,
) -> LifecycleResult:
    """Release one processing Postgres job back to the ready queue."""

    with conn:
        with cursor_factory(conn) as cur:
            cur.execute(
                (
                    "SELECT id, domain, queue, job_type, status, worker_id, lease_id "
                    "FROM jobs WHERE id = %s FOR UPDATE"
                ),
                (command.job_id,),
            )
            selected = cur.fetchone()
            if selected is None:
                return LifecycleResult.no_transition(NoTransitionReason.MISSING)
            current = dict(selected)
            if current.get("status") != "processing":
                return LifecycleResult.no_transition(
                    NoTransitionReason.WRONG_STATUS,
                    row=current,
                )
            if command.enforce and (
                current.get("worker_id") != command.worker_id
                or current.get("lease_id") != command.lease_id
            ):
                return LifecycleResult.no_transition(
                    NoTransitionReason.STALE_LEASE,
                    row=current,
                )

            if command.enforce:
                cur.execute(
                    _RELEASE_ENFORCED_SQL,
                    (command.job_id, command.worker_id, command.lease_id),
                )
            else:
                cur.execute(_RELEASE_SQL, (command.job_id,))
            row = cur.fetchone()
            if row is None:
                return _classify_lifecycle_no_transition(
                    cur,
                    job_id=command.job_id,
                    enforce=command.enforce,
                    worker_id=command.worker_id,
                    lease_id=command.lease_id,
                )
            released = dict(row)
            if counters_enabled:
                cur.execute(
                    _RELEASE_COUNTER_SQL,
                    (
                        released["domain"],
                        released["queue"],
                        released["job_type"],
                        1,
                        0,
                    ),
                )
            return LifecycleResult.applied(row=released)


def _dependency_condition() -> str:
    return (
        " AND (status != 'queued' OR NOT EXISTS ("
        "SELECT 1 FROM job_dependencies jd "
        "LEFT JOIN jobs dep ON dep.uuid = jd.depends_on_job_uuid "
        "WHERE jd.job_uuid = jobs.uuid AND "
        "COALESCE(dep.status, jd.depends_on_terminal_status, 'missing') <> 'completed'"
        "))"
    )


def _order_clause(command: AcquireJobCommand) -> str:
    tie_direction = "DESC" if command.tie_break == "lifo" else "ASC"
    return _ORDER_CLAUSES[(command.priority_direction, tie_direction)]


def _candidate_sql(command: AcquireJobCommand) -> tuple[str, list[Any]]:
    sql = (
        "SELECT id FROM jobs WHERE domain = %s AND queue = %s "
        "AND status = 'queued' AND (available_at IS NULL OR available_at <= NOW())"
    )
    sql += _dependency_condition()
    params: list[Any] = [command.domain, command.queue]
    if command.owner_user_id:
        sql += " AND owner_user_id = %s"
        params.append(command.owner_user_id)
    if command.job_type:
        sql += " AND job_type = %s"
        params.append(command.job_type)
    sql += _order_clause(command)
    return sql, params


def _single_update_acquire(cur: Any, *, command: AcquireJobCommand) -> dict[str, Any] | None:
    candidate_sql, candidate_params = _candidate_sql(command)
    sql = "".join(
        [
            "WITH picked AS (",
            "  ",
            candidate_sql,
            ") ",
            "UPDATE jobs SET status='processing', "
            "retry_count = CASE WHEN status='processing' THEN retry_count + 1 ELSE retry_count END, "
            "started_at = COALESCE(started_at, NOW()), acquired_at = COALESCE(acquired_at, NOW()), "
            "leased_until = NOW() + (%s || ' seconds')::interval, worker_id = %s, lease_id = %s, "
            "completion_token = NULL WHERE id IN (SELECT id FROM picked) RETURNING *",
        ]
    )
    cur.execute(
        sql,
        (*candidate_params, command.lease_seconds, command.worker_id, command.lease_id),
    )
    row = cur.fetchone()
    return dict(row) if row else None


def _two_step_acquire(cur: Any, *, command: AcquireJobCommand) -> dict[str, Any] | None:
    candidate_sql, candidate_params = _candidate_sql(command)
    cur.execute(candidate_sql, candidate_params)
    candidate = cur.fetchone()
    if not candidate:
        return None
    job_id = int(candidate["id"] if isinstance(candidate, dict) else candidate[0])
    cur.execute(
        (
            "UPDATE jobs SET status = 'processing', "
            "retry_count = CASE WHEN status = 'processing' THEN retry_count + 1 ELSE retry_count END, "
            "started_at = COALESCE(started_at, NOW()), acquired_at = COALESCE(acquired_at, NOW()), "
            "leased_until = NOW() + (%s || ' seconds')::interval, "
            "worker_id = %s, lease_id = %s, completion_token = NULL WHERE id = %s"
        ),
        (command.lease_seconds, command.worker_id, command.lease_id, job_id),
    )
    cur.execute("SELECT * FROM jobs WHERE id = %s", (job_id,))
    row = cur.fetchone()
    return dict(row) if row else None


def _bump_acquired_counters(cur: Any, *, acquired: dict[str, Any]) -> None:
    is_scheduled = acquired.get("available_at") is not None
    cur.execute(
        (
            "INSERT INTO job_counters(domain,queue,job_type,ready_count,scheduled_count,processing_count,"
            "quarantined_count) VALUES(%s,%s,%s,0,0,1,0) "
            "ON CONFLICT (domain,queue,job_type) DO UPDATE SET "
            "ready_count = GREATEST(job_counters.ready_count + %s, 0), "
            "scheduled_count = GREATEST(job_counters.scheduled_count + %s, 0), "
            "processing_count = job_counters.processing_count + 1, updated_at = NOW()"
        ),
        (
            acquired.get("domain"),
            acquired.get("queue"),
            acquired.get("job_type"),
            -1 if not is_scheduled else 0,
            -1 if is_scheduled else 0,
        ),
    )


def _bump_acquired_counters_best_effort(cur: Any, *, acquired: dict[str, Any]) -> None:
    cur.execute("SAVEPOINT jobs_acquire_counter_update")
    try:
        _bump_acquired_counters(cur, acquired=acquired)
    except _COUNTER_NONCRITICAL_ERRORS as exc:
        cur.execute("ROLLBACK TO SAVEPOINT jobs_acquire_counter_update")
        cur.execute("RELEASE SAVEPOINT jobs_acquire_counter_update")
        logger.warning(
            "Non-critical Postgres jobs acquisition counter update failed for {}:{}:{}: {}",
            acquired.get("domain"),
            acquired.get("queue"),
            acquired.get("job_type"),
            exc,
        )
    else:
        cur.execute("RELEASE SAVEPOINT jobs_acquire_counter_update")


def acquire_job(
    conn: Any,
    cursor_factory: Callable[[Any], AbstractContextManager[Any]],
    *,
    command: AcquireJobCommand,
    counters_enabled: bool,
    now: datetime,
) -> LifecycleResult:
    """Atomically acquire the next eligible Postgres job."""

    del now  # Postgres remains authoritative for lease and eligibility timestamps.
    with conn:
        with cursor_factory(conn) as cur:
            if command.max_inflight_quota and command.owner_user_id:
                cur.execute(
                    "SELECT pg_advisory_xact_lock(%s)",
                    (_pg_advisory_key("max-inflight", command.domain, command.owner_user_id),),
                )
                cur.execute(
                    "SELECT COUNT(*) AS c FROM jobs WHERE domain=%s AND owner_user_id=%s "
                    "AND status='processing' AND leased_until IS NOT NULL AND leased_until > NOW()",
                    (command.domain, command.owner_user_id),
                )
                if _count_from_row(cur.fetchone()) >= command.max_inflight_quota:
                    return LifecycleResult.no_transition(NoTransitionReason.NO_ELIGIBLE_JOB)

            acquired = (
                _single_update_acquire(cur, command=command)
                if command.single_update
                else _two_step_acquire(cur, command=command)
            )
            if acquired is None:
                return LifecycleResult.no_transition(NoTransitionReason.NO_ELIGIBLE_JOB)
            if counters_enabled:
                _bump_acquired_counters_best_effort(cur, acquired=acquired)
            return LifecycleResult.applied(row=acquired)


__all__ = ["acquire_job", "release_job", "renew_lease"]
