"""SQLite-backed Jobs single-job lifecycle operations."""

from __future__ import annotations

import json
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
    ApplyPreparedDispositionCommand,
    BatchRenewLeasesCommand,
    BatchRenewLeasesResult,
    EnsureLeaseHorizonCommand,
    FindJobByIdentityCommand,
    JobIdentityLookupResult,
    JobIdentityLookupState,
    LeaseHorizonResult,
    LifecycleResult,
    NoTransitionReason,
    OperationOutcome,
    PreparedDispositionKind,
    PreparedDispositionOrigin,
    PreparedDispositionResult,
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
    except BaseException as primary_error:
        try:
            conn.execute("ROLLBACK TO SAVEPOINT jobs_batch_renew_leases")
            conn.execute("RELEASE SAVEPOINT jobs_batch_renew_leases")
        except BaseException as cleanup_error:
            raise primary_error from cleanup_error
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
    """Renew an ordered SQLite lease batch in one atomic scope."""

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


def _parse_json_object(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _aware_sqlite_timestamp(value: str | datetime | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _database_times(conn: sqlite3.Connection) -> tuple[datetime, datetime]:
    row = conn.execute(
        "SELECT STRFTIME('%Y-%m-%dT%H:%M:%fZ','now'), "
        "STRFTIME('%Y-%m-%dT%H:%M:%fZ','now','+30 seconds')"
    ).fetchone()
    if row is None:
        raise RuntimeError("Jobs database clock unavailable")
    now = _aware_sqlite_timestamp(row[0])
    deferred = _aware_sqlite_timestamp(row[1])
    if now is None or deferred is None:
        raise RuntimeError("Jobs database clock returned an invalid timestamp")
    return now, deferred


def _marker_not_before(marker: dict[str, Any]) -> datetime | None:
    return _aware_sqlite_timestamp(marker.get("original_not_before_at"))


def _marker_matches(
    marker: dict[str, Any],
    command: ApplyPreparedDispositionCommand,
) -> bool:
    if _aware_sqlite_timestamp(marker.get("applied_at")) is None:
        return False
    disposition = command.disposition
    expected = {
        "schema_version": 1,
        "token": disposition.token,
        "kind": disposition.kind.value,
        "origin": disposition.origin.value,
        "delivery_id": disposition.delivery_id,
    }
    if disposition.attempt_id is not None:
        expected["attempt_id"] = disposition.attempt_id
    if disposition.not_before_at is not None:
        expected["original_not_before_at"] = disposition.not_before_at.isoformat()
    elif disposition.origin is PreparedDispositionOrigin.INFRASTRUCTURE:
        if _marker_not_before(marker) is None:
            return False
        expected["original_not_before_at"] = marker["original_not_before_at"]
    comparable = {key: marker.get(key) for key in expected}
    return comparable == expected and set(marker) == {*expected, "applied_at"}


def _identity_matches(
    row: dict[str, Any],
    *,
    domain: str,
    queue: str,
    job_type: str,
    expected_payload: dict[str, Any],
) -> bool:
    return (
        row.get("domain") == domain
        and row.get("queue") == queue
        and row.get("job_type") == job_type
        and _parse_json_object(row.get("payload")) == expected_payload
    )


def _prepared_counter_transition(
    conn: sqlite3.Connection,
    *,
    row: dict[str, Any],
    new_status: str,
) -> None:
    old_status = str(row.get("status") or "")
    old_scheduled = old_status == "queued" and row.get("available_at") is not None
    ready_delta = int(new_status == "queued") - int(old_status == "queued" and not old_scheduled)
    scheduled_delta = int(new_status == "queued") - int(old_scheduled)
    if new_status == "queued":
        ready_delta = 0
        scheduled_delta = 1 - int(old_scheduled)
    processing_delta = -int(old_status == "processing")
    quarantined_delta = int(new_status == "quarantined") - int(old_status == "quarantined")
    conn.execute(
        "INSERT INTO job_counters(domain,queue,job_type,ready_count,scheduled_count,"
        "processing_count,quarantined_count) VALUES(?,?,?,?,?,?,?) "
        "ON CONFLICT(domain,queue,job_type) DO UPDATE SET "
        "ready_count=MAX(job_counters.ready_count + excluded.ready_count,0), "
        "scheduled_count=MAX(job_counters.scheduled_count + excluded.scheduled_count,0), "
        "processing_count=MAX(job_counters.processing_count + excluded.processing_count,0), "
        "quarantined_count=MAX(job_counters.quarantined_count + excluded.quarantined_count,0), "
        "updated_at=DATETIME('now')",
        (
            row.get("domain"),
            row.get("queue"),
            row.get("job_type"),
            ready_delta,
            scheduled_delta,
            processing_delta,
            quarantined_delta,
        ),
    )


def _insert_prepared_event(
    conn: sqlite3.Connection,
    *,
    row: dict[str, Any],
    event_type: str,
    marker: dict[str, Any],
) -> None:
    attrs = {
        "kind": marker["kind"],
        "origin": marker["origin"],
        "reason_code": row.get("error_code") or row.get("cancellation_reason"),
    }
    conn.execute(
        "INSERT INTO job_events(job_id,domain,queue,job_type,event_type,attrs_json,"
        "owner_user_id,request_id,trace_id,created_at) "
        "VALUES(?,?,?,?,?,?,?,?,?,DATETIME('now'))",
        (
            row.get("id"),
            row.get("domain"),
            row.get("queue"),
            row.get("job_type"),
            event_type,
            json.dumps(attrs, separators=(",", ":"), sort_keys=True),
            row.get("owner_user_id"),
            row.get("request_id"),
            row.get("trace_id"),
        ),
    )


def apply_prepared_disposition(
    conn: sqlite3.Connection,
    *,
    command: ApplyPreparedDispositionCommand,
    counters_enabled: bool,
    outbox_enabled: bool,
) -> PreparedDispositionResult:
    """Apply one exact prepared transition under a SQLite write transaction."""

    conn.execute("BEGIN IMMEDIATE")
    try:
        selected = conn.execute(
            "SELECT * FROM jobs WHERE id=?",
            (command.job_id,),
        ).fetchone()
        if selected is None:
            conn.commit()
            return PreparedDispositionResult.no_transition(NoTransitionReason.MISSING)
        row = dict(selected)
        if not _identity_matches(
            row,
            domain=command.domain,
            queue=command.queue,
            job_type=command.job_type,
            expected_payload=command.expected_payload,
        ):
            conn.commit()
            return PreparedDispositionResult.conflict(state=str(row.get("status")))

        marker = _parse_json_object(row.get("result"))
        if marker is not None and marker.get("token") == command.disposition.token:
            if not _marker_matches(marker, command):
                conn.commit()
                return PreparedDispositionResult.conflict(state=str(row.get("status")))
            result = PreparedDispositionResult.applied(
                state=str(row.get("status")),
                metadata=marker,
                already_applied=True,
                not_before_at=(
                    _aware_sqlite_timestamp(row.get("available_at"))
                    or _marker_not_before(marker)
                ),
            )
            conn.commit()
            return result

        disposition = command.disposition
        status = str(row.get("status") or "")
        queued_cancel = (
            disposition.kind is PreparedDispositionKind.CANCEL
            and status == "queued"
            and command.worker_id is None
            and command.lease_id is None
        )
        if not queued_cancel:
            if status != "processing":
                conn.commit()
                return PreparedDispositionResult.no_transition(
                    NoTransitionReason.WRONG_STATUS,
                    state=status,
                )
            if (
                command.worker_id is None
                or command.lease_id is None
                or row.get("worker_id") != command.worker_id
                or row.get("lease_id") != command.lease_id
            ):
                conn.commit()
                return PreparedDispositionResult.no_transition(
                    NoTransitionReason.STALE_LEASE,
                    state=status,
                )

        database_now, infrastructure_not_before = _database_times(conn)
        not_before: datetime | None = None
        if disposition.kind is PreparedDispositionKind.RETRY:
            not_before = max(database_now, disposition.not_before_at)
        elif disposition.origin is PreparedDispositionOrigin.INFRASTRUCTURE:
            not_before = infrastructure_not_before
        elif disposition.origin is PreparedDispositionOrigin.RECOVERY:
            not_before = max(database_now, disposition.not_before_at)

        marker = {
            "schema_version": 1,
            "token": disposition.token,
            "kind": disposition.kind.value,
            "origin": disposition.origin.value,
            "delivery_id": disposition.delivery_id,
        }
        if disposition.attempt_id is not None:
            marker["attempt_id"] = disposition.attempt_id
        if not_before is not None:
            marker["original_not_before_at"] = (
                disposition.not_before_at.isoformat()
                if disposition.not_before_at is not None
                else not_before.isoformat()
            )
        marker["applied_at"] = database_now.isoformat()
        marker_json = json.dumps(marker, separators=(",", ":"), sort_keys=True)

        event_type: str
        new_status: str
        if disposition.kind is PreparedDispositionKind.COMPLETE:
            new_status = "completed"
            event_type = "job.completed"
            changed = conn.execute(
                "UPDATE jobs SET status='completed', result=?, completed_at=DATETIME('now'), "
                "leased_until=NULL,worker_id=NULL,lease_id=NULL,completion_token=? "
                "WHERE id=? AND status='processing' AND worker_id=? AND lease_id=?",
                (
                    marker_json,
                    disposition.token,
                    command.job_id,
                    command.worker_id,
                    command.lease_id,
                ),
            )
        elif disposition.kind is PreparedDispositionKind.RETRY:
            current_streak = int(row.get("failure_streak_count") or 0)
            next_streak = (
                current_streak + 1
                if row.get("failure_streak_code") == disposition.reason_code
                else 1
            )
            threshold = row.get("quarantine_threshold")
            quarantine = threshold is not None and next_streak >= int(threshold)
            new_status = "quarantined" if quarantine else "queued"
            event_type = "job.quarantined" if quarantine else "job.retry_scheduled"
            changed = conn.execute(
                "UPDATE jobs SET status=?, result=?, retry_count=COALESCE(retry_count,0)+1, "
                "failure_streak_code=?, failure_streak_count=?, error_code=?, "
                "available_at=?, quarantined_at=CASE WHEN ? THEN DATETIME('now') ELSE quarantined_at END, "
                "completed_at=CASE WHEN ? THEN DATETIME('now') ELSE completed_at END, "
                "leased_until=NULL,worker_id=NULL,lease_id=NULL,completion_token=?, "
                "acquired_at=NULL,started_at=NULL WHERE id=? AND status='processing' "
                "AND worker_id=? AND lease_id=?",
                (
                    new_status,
                    marker_json,
                    disposition.reason_code,
                    next_streak,
                    disposition.reason_code,
                    _sqlite_timestamp(not_before),
                    int(quarantine),
                    int(quarantine),
                    disposition.token if quarantine else None,
                    command.job_id,
                    command.worker_id,
                    command.lease_id,
                ),
            )
        elif disposition.kind is PreparedDispositionKind.FAIL:
            new_status = "failed"
            event_type = "job.failed"
            changed = conn.execute(
                "UPDATE jobs SET status='failed', result=?, error_code=?, error_message=?, "
                "last_error=?, completed_at=DATETIME('now'), leased_until=NULL,worker_id=NULL,"
                "lease_id=NULL,completion_token=? WHERE id=? AND status='processing' "
                "AND worker_id=? AND lease_id=?",
                (
                    marker_json,
                    disposition.reason_code,
                    disposition.reason_code,
                    disposition.reason_code,
                    disposition.token,
                    command.job_id,
                    command.worker_id,
                    command.lease_id,
                ),
            )
        elif disposition.kind is PreparedDispositionKind.CANCEL:
            new_status = "cancelled"
            event_type = "job.cancelled"
            if queued_cancel:
                changed = conn.execute(
                    "UPDATE jobs SET status='cancelled', result=?, cancellation_reason=?, "
                    "cancelled_at=DATETIME('now'), completed_at=DATETIME('now'), "
                    "completion_token=? WHERE id=? AND status='queued'",
                    (
                        marker_json,
                        disposition.reason_code,
                        disposition.token,
                        command.job_id,
                    ),
                )
            else:
                changed = conn.execute(
                    "UPDATE jobs SET status='cancelled', result=?, cancellation_reason=?, "
                    "cancelled_at=DATETIME('now'), completed_at=DATETIME('now'), "
                    "leased_until=NULL,worker_id=NULL,lease_id=NULL,completion_token=? "
                    "WHERE id=? AND status='processing' AND worker_id=? AND lease_id=?",
                    (
                        marker_json,
                        disposition.reason_code,
                        disposition.token,
                        command.job_id,
                        command.worker_id,
                        command.lease_id,
                    ),
                )
        else:
            new_status = "queued"
            event_type = "job.deferred"
            changed = conn.execute(
                "UPDATE jobs SET status='queued', result=?, available_at=?, "
                "leased_until=NULL,worker_id=NULL,lease_id=NULL,completion_token=NULL, "
                "acquired_at=NULL,started_at=NULL WHERE id=? AND status='processing' "
                "AND worker_id=? AND lease_id=?",
                (
                    marker_json,
                    _sqlite_timestamp(not_before),
                    command.job_id,
                    command.worker_id,
                    command.lease_id,
                ),
            )

        if changed.rowcount != 1:
            conn.rollback()
            return PreparedDispositionResult.no_transition(
                NoTransitionReason.STALE_LEASE,
                state=status,
            )
        if counters_enabled:
            _prepared_counter_transition(conn, row=row, new_status=new_status)
        updated = dict(
            conn.execute("SELECT * FROM jobs WHERE id=?", (command.job_id,)).fetchone()
        )
        if outbox_enabled:
            _insert_prepared_event(
                conn,
                row=updated,
                event_type=event_type,
                marker=marker,
            )
        conn.commit()
        persisted_not_before = (
            _aware_sqlite_timestamp(updated.get("available_at"))
            if new_status == "queued"
            else None
        )
        return PreparedDispositionResult.applied(
            state=new_status,
            metadata=marker,
            already_applied=False,
            not_before_at=persisted_not_before,
        )
    except Exception:
        conn.rollback()
        raise


def ensure_lease_horizon(
    conn: sqlite3.Connection,
    *,
    command: EnsureLeaseHorizonCommand,
) -> LeaseHorizonResult:
    """Extend but never shorten one exact SQLite processing lease."""

    conn.execute("BEGIN IMMEDIATE")
    try:
        selected = conn.execute(
            "SELECT * FROM jobs WHERE id=?",
            (command.job_id,),
        ).fetchone()
        if selected is None:
            conn.commit()
            return LeaseHorizonResult.no_transition(NoTransitionReason.MISSING)
        row = dict(selected)
        observed = _aware_sqlite_timestamp(row.get("leased_until"))
        if not _identity_matches(
            row,
            domain=command.domain,
            queue=command.queue,
            job_type=command.job_type,
            expected_payload=command.expected_payload,
        ):
            conn.commit()
            return LeaseHorizonResult(
                outcome=OperationOutcome.BACKEND_CONFLICT,
                ensured=False,
                leased_until=observed,
            )
        if row.get("status") != "processing":
            conn.commit()
            return LeaseHorizonResult.no_transition(
                NoTransitionReason.WRONG_STATUS,
                leased_until=observed,
            )
        if row.get("worker_id") != command.worker_id or row.get("lease_id") != command.lease_id:
            conn.commit()
            return LeaseHorizonResult.no_transition(
                NoTransitionReason.STALE_LEASE,
                leased_until=observed,
            )
        conn.execute(
            "UPDATE jobs SET leased_until=MAX(COALESCE(leased_until,DATETIME('now')), "
            "DATETIME('now','+' || ? || ' seconds')) WHERE id=? AND status='processing' "
            "AND worker_id=? AND lease_id=?",
            (
                command.minimum_seconds,
                command.job_id,
                command.worker_id,
                command.lease_id,
            ),
        )
        leased_until = _aware_sqlite_timestamp(
            conn.execute(
                "SELECT leased_until FROM jobs WHERE id=?",
                (command.job_id,),
            ).fetchone()[0]
        )
        conn.commit()
        return LeaseHorizonResult.applied(leased_until=leased_until)
    except Exception:
        conn.rollback()
        raise


def find_job_by_identity(
    conn: sqlite3.Connection,
    *,
    command: FindJobByIdentityCommand,
) -> JobIdentityLookupResult:
    """Find one exact active or archived SQLite job without inserting work."""

    with conn:
        active_rows = conn.execute(
            "SELECT * FROM jobs WHERE domain=? AND queue=? AND job_type=? "
            "AND idempotency_key=?",
            (
                command.domain,
                command.queue,
                command.job_type,
                command.idempotency_key,
            ),
        ).fetchall()
        archived_rows = conn.execute(
            "SELECT * FROM jobs_archive WHERE domain=? AND queue=? AND job_type=? "
            "AND idempotency_key=?",
            (
                command.domain,
                command.queue,
                command.job_type,
                command.idempotency_key,
            ),
        ).fetchall()
    matches = [
        (JobIdentityLookupState.ACTIVE, dict(row)) for row in active_rows
    ] + [
        (JobIdentityLookupState.ARCHIVED, dict(row)) for row in archived_rows
    ]
    if not matches:
        return JobIdentityLookupResult.missing()
    if len(matches) != 1:
        return JobIdentityLookupResult.conflict()
    state, row = matches[0]
    payload = _parse_json_object(row.get("payload"))
    if payload != command.expected_payload:
        return JobIdentityLookupResult.conflict()
    row["payload"] = payload
    result = _parse_json_object(row.get("result"))
    if result is not None:
        row["result"] = result
    return JobIdentityLookupResult.found(state, row)
