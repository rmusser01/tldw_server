"""Postgres-backed Jobs single-job lifecycle operations."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from contextlib import AbstractContextManager
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.Jobs.migrations import (
    SlidesArchiveNormalizationError,
    normalize_slides_archive_projection,
    slides_archive_values_equal,
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
    canonical_admin_webhook_row_matches,
    is_admin_webhook_delivery_queue,
    prepared_disposition_fingerprint,
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
            "completion_token = NULL, no_attempt_recovery_fingerprint = NULL "
            "WHERE id IN (SELECT id FROM picked) RETURNING *",
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
            "worker_id = %s, lease_id = %s, completion_token = NULL, "
            "no_attempt_recovery_fingerprint = NULL WHERE id = %s"
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


def _aware_timestamp(value: Any) -> datetime | None:
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


def _marker_matches(
    marker: dict[str, Any],
    command: ApplyPreparedDispositionCommand,
) -> bool:
    if _aware_timestamp(marker.get("applied_at")) is None:
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
        if _aware_timestamp(marker.get("original_not_before_at")) is None:
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
    archived: bool = False,
) -> bool:
    payload = _parse_json_object(row.get("payload"))
    if is_admin_webhook_delivery_queue(domain, queue):
        return canonical_admin_webhook_row_matches(
            {**row, "payload": payload},
            expected_payload=expected_payload,
            archived=archived,
        )
    return (
        row.get("domain") == domain
        and row.get("queue") == queue
        and row.get("job_type") == job_type
        and slides_archive_values_equal(payload, expected_payload)
    )


def _prepared_counter_transition(
    cur: Any,
    *,
    row: dict[str, Any],
    new_status: str,
) -> None:
    old_status = str(row.get("status") or "")
    old_scheduled = old_status == "queued" and row.get("available_at") is not None
    ready_delta = -int(old_status == "queued" and not old_scheduled)
    scheduled_delta = -int(old_scheduled)
    if new_status == "queued":
        ready_delta = 0
        scheduled_delta += 1
    processing_delta = -int(old_status == "processing")
    quarantined_delta = int(new_status == "quarantined") - int(old_status == "quarantined")
    cur.execute(
        "INSERT INTO job_counters(domain,queue,job_type,ready_count,scheduled_count,"
        "processing_count,quarantined_count) VALUES(%s,%s,%s,%s,%s,%s,%s) "
        "ON CONFLICT(domain,queue,job_type) DO UPDATE SET "
        "ready_count=GREATEST(job_counters.ready_count + EXCLUDED.ready_count,0), "
        "scheduled_count=GREATEST(job_counters.scheduled_count + EXCLUDED.scheduled_count,0), "
        "processing_count=GREATEST(job_counters.processing_count + EXCLUDED.processing_count,0), "
        "quarantined_count=GREATEST(job_counters.quarantined_count + EXCLUDED.quarantined_count,0), "
        "updated_at=NOW()",
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
    cur: Any,
    *,
    row: dict[str, Any],
    event_type: str,
    marker: dict[str, Any],
    reason_code: str | None,
) -> None:
    attrs = {
        "kind": marker["kind"],
        "origin": marker["origin"],
        "reason_code": reason_code,
    }
    cur.execute(
        "INSERT INTO job_events(job_id,domain,queue,job_type,event_type,attrs_json,"
        "owner_user_id,request_id,trace_id,created_at) "
        "VALUES(%s,%s,%s,%s,%s,%s::jsonb,%s,%s,%s,NOW())",
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
    conn: Any,
    cursor_factory: Callable[[Any], AbstractContextManager[Any]],
    *,
    command: ApplyPreparedDispositionCommand,
    counters_enabled: bool,
    outbox_enabled: bool,
) -> PreparedDispositionResult:
    """Apply one exact prepared transition under a Postgres row lock."""

    with conn:
        with cursor_factory(conn) as cur:
            cur.execute("SELECT * FROM jobs WHERE id=%s FOR UPDATE", (command.job_id,))
            selected = cur.fetchone()
            if selected is None:
                return PreparedDispositionResult.no_transition(NoTransitionReason.MISSING)
            row = dict(selected)
            if not _identity_matches(
                row,
                domain=command.domain,
                queue=command.queue,
                job_type=command.job_type,
                expected_payload=command.expected_payload,
            ):
                return PreparedDispositionResult.conflict(state=str(row.get("status")))

            facts_fingerprint = prepared_disposition_fingerprint(command.disposition)
            marker = _parse_json_object(row.get("result"))
            if marker is not None and marker.get("token") == command.disposition.token:
                if (
                    not _marker_matches(marker, command)
                    or row.get("prepared_disposition_fingerprint")
                    != facts_fingerprint
                ):
                    return PreparedDispositionResult.conflict(state=str(row.get("status")))
                return PreparedDispositionResult.applied(
                    state=str(row.get("status")),
                    metadata=marker,
                    already_applied=True,
                    not_before_at=(
                        _aware_timestamp(row.get("available_at"))
                        or _aware_timestamp(marker.get("original_not_before_at"))
                    ),
                )

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
                    return PreparedDispositionResult.no_transition(
                        NoTransitionReason.STALE_LEASE,
                        state=status,
                    )

            cur.execute(
                "SELECT NOW() AS database_now, "
                "NOW() + interval '30 seconds' AS infrastructure_not_before"
            )
            clock_row = cur.fetchone()
            if isinstance(clock_row, dict):
                database_now = _aware_timestamp(clock_row.get("database_now"))
                infrastructure_not_before = _aware_timestamp(
                    clock_row.get("infrastructure_not_before")
                )
            else:
                clock_values = tuple(clock_row)
                database_now = _aware_timestamp(clock_values[0])
                infrastructure_not_before = _aware_timestamp(clock_values[1])
            if database_now is None or infrastructure_not_before is None:
                raise RuntimeError("Jobs database clock returned an invalid timestamp")
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
                cur.execute(
                    "UPDATE jobs SET status='completed',result=%s::jsonb,"
                    "prepared_disposition_fingerprint=%s,completed_at=NOW(),"
                    "leased_until=NULL,worker_id=NULL,lease_id=NULL,completion_token=%s "
                    "WHERE id=%s AND status='processing' AND worker_id=%s AND lease_id=%s",
                    (
                        marker_json,
                        facts_fingerprint,
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
                cur.execute(
                    "UPDATE jobs SET status=%s,result=%s::jsonb,"
                    "prepared_disposition_fingerprint=%s,"
                    "retry_count=COALESCE(retry_count,0)+1,"
                    "failure_streak_code=%s,failure_streak_count=%s,error_code=%s,available_at=%s,"
                    "quarantined_at=CASE WHEN %s THEN NOW() ELSE quarantined_at END,"
                    "completed_at=CASE WHEN %s THEN NOW() ELSE completed_at END,"
                    "leased_until=NULL,worker_id=NULL,lease_id=NULL,completion_token=%s,"
                    "acquired_at=NULL,started_at=NULL WHERE id=%s AND status='processing' "
                    "AND worker_id=%s AND lease_id=%s",
                    (
                        new_status,
                        marker_json,
                        facts_fingerprint,
                        disposition.reason_code,
                        next_streak,
                        disposition.reason_code,
                        not_before,
                        quarantine,
                        quarantine,
                        disposition.token if quarantine else None,
                        command.job_id,
                        command.worker_id,
                        command.lease_id,
                    ),
                )
            elif disposition.kind is PreparedDispositionKind.FAIL:
                new_status = "failed"
                event_type = "job.failed"
                cur.execute(
                    "UPDATE jobs SET status='failed',result=%s::jsonb,"
                    "prepared_disposition_fingerprint=%s,error_code=%s,"
                    "error_message=%s,last_error=%s,completed_at=NOW(),leased_until=NULL,"
                    "worker_id=NULL,lease_id=NULL,completion_token=%s WHERE id=%s "
                    "AND status='processing' AND worker_id=%s AND lease_id=%s",
                    (
                        marker_json,
                        facts_fingerprint,
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
                    cur.execute(
                        "UPDATE jobs SET status='cancelled',result=%s::jsonb,"
                        "prepared_disposition_fingerprint=%s,"
                        "cancellation_reason=%s,cancelled_at=NOW(),completed_at=NOW(),"
                        "completion_token=%s,no_attempt_recovery_fingerprint=NULL "
                        "WHERE id=%s AND status='queued'",
                        (
                            marker_json,
                            facts_fingerprint,
                            disposition.reason_code,
                            disposition.token,
                            command.job_id,
                        ),
                    )
                else:
                    cur.execute(
                        "UPDATE jobs SET status='cancelled',result=%s::jsonb,"
                        "prepared_disposition_fingerprint=%s,"
                        "cancellation_reason=%s,cancelled_at=NOW(),completed_at=NOW(),"
                        "leased_until=NULL,worker_id=NULL,lease_id=NULL,completion_token=%s "
                        "WHERE id=%s AND status='processing' AND worker_id=%s AND lease_id=%s",
                        (
                            marker_json,
                            facts_fingerprint,
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
                cur.execute(
                    "UPDATE jobs SET status='queued',result=%s::jsonb,"
                    "prepared_disposition_fingerprint=%s,available_at=%s,"
                    "leased_until=NULL,worker_id=NULL,lease_id=NULL,completion_token=NULL,"
                    "acquired_at=NULL,started_at=NULL WHERE id=%s AND status='processing' "
                    "AND worker_id=%s AND lease_id=%s",
                    (
                        marker_json,
                        facts_fingerprint,
                        not_before,
                        command.job_id,
                        command.worker_id,
                        command.lease_id,
                    ),
                )

            if cur.rowcount != 1:
                return PreparedDispositionResult.no_transition(
                    NoTransitionReason.STALE_LEASE,
                    state=status,
                )
            if counters_enabled:
                _prepared_counter_transition(cur, row=row, new_status=new_status)
            cur.execute("SELECT * FROM jobs WHERE id=%s", (command.job_id,))
            updated = dict(cur.fetchone())
            if outbox_enabled:
                _insert_prepared_event(
                    cur,
                    row=updated,
                    event_type=event_type,
                    marker=marker,
                    reason_code=disposition.reason_code,
                )
            return PreparedDispositionResult.applied(
                state=new_status,
                metadata=marker,
                already_applied=False,
                not_before_at=not_before,
            )


def ensure_lease_horizon(
    conn: Any,
    cursor_factory: Callable[[Any], AbstractContextManager[Any]],
    *,
    command: EnsureLeaseHorizonCommand,
) -> LeaseHorizonResult:
    """Extend but never shorten one exact Postgres processing lease."""

    with conn:
        with cursor_factory(conn) as cur:
            cur.execute("SELECT * FROM jobs WHERE id=%s FOR UPDATE", (command.job_id,))
            selected = cur.fetchone()
            if selected is None:
                return LeaseHorizonResult.no_transition(NoTransitionReason.MISSING)
            row = dict(selected)
            observed = _aware_timestamp(row.get("leased_until"))
            if not _identity_matches(
                row,
                domain=command.domain,
                queue=command.queue,
                job_type=command.job_type,
                expected_payload=command.expected_payload,
            ):
                return LeaseHorizonResult(
                    outcome=OperationOutcome.BACKEND_CONFLICT,
                    ensured=False,
                    leased_until=observed,
                )
            if row.get("status") != "processing":
                return LeaseHorizonResult.no_transition(
                    NoTransitionReason.WRONG_STATUS,
                    leased_until=observed,
                )
            if row.get("worker_id") != command.worker_id or row.get("lease_id") != command.lease_id:
                return LeaseHorizonResult.no_transition(
                    NoTransitionReason.STALE_LEASE,
                    leased_until=observed,
                )
            cur.execute(
                "UPDATE jobs SET leased_until=GREATEST(COALESCE(leased_until,NOW()),"
                "NOW()+(%s || ' seconds')::interval) WHERE id=%s AND status='processing' "
                "AND worker_id=%s AND lease_id=%s RETURNING leased_until",
                (
                    command.minimum_seconds,
                    command.job_id,
                    command.worker_id,
                    command.lease_id,
                ),
            )
            changed = cur.fetchone()
            if changed is None:
                return LeaseHorizonResult.no_transition(NoTransitionReason.STALE_LEASE)
            leased_until = (
                changed.get("leased_until")
                if isinstance(changed, dict)
                else changed[0]
            )
            return LeaseHorizonResult.applied(
                leased_until=leased_until,
                guaranteed_seconds=command.minimum_seconds,
            )


def find_job_by_identity(
    conn: Any,
    cursor_factory: Callable[[Any], AbstractContextManager[Any]],
    *,
    command: FindJobByIdentityCommand,
) -> JobIdentityLookupResult:
    """Find one exact active or archived Postgres job without inserting work."""

    with conn:
        with cursor_factory(conn) as cur:
            params = (
                command.domain,
                command.queue,
                command.job_type,
                command.idempotency_key,
            )
            cur.execute(
                "SELECT * FROM jobs WHERE domain=%s AND queue=%s AND job_type=%s "
                "AND idempotency_key=%s",
                params,
            )
            active_rows = cur.fetchall() or []
            cur.execute(
                "SELECT *, payload IS NOT NULL AS __slides_archive_payload_present, "
                "result IS NOT NULL AS __slides_archive_result_present "
                "FROM jobs_archive WHERE domain=%s AND queue=%s AND job_type=%s "
                "AND idempotency_key=%s",
                params,
            )
            archived_rows = cur.fetchall() or []
    matches = [
        (JobIdentityLookupState.ACTIVE, dict(row)) for row in active_rows
    ] + [
        (JobIdentityLookupState.ARCHIVED, dict(row)) for row in archived_rows
    ]
    if not matches:
        return JobIdentityLookupResult.missing()
    if len(matches) != 1:
        return JobIdentityLookupResult.conflict()
    state, raw_row = matches[0]
    try:
        row = (
            normalize_slides_archive_projection(raw_row)
            if state is JobIdentityLookupState.ARCHIVED
            else raw_row
        )
    except SlidesArchiveNormalizationError:
        return JobIdentityLookupResult.conflict()
    if not _identity_matches(
        row,
        domain=command.domain,
        queue=command.queue,
        job_type=command.job_type,
        expected_payload=command.expected_payload,
        archived=state is JobIdentityLookupState.ARCHIVED,
    ):
        return JobIdentityLookupResult.conflict()
    payload = _parse_json_object(row.get("payload"))
    row["payload"] = payload
    result = _parse_json_object(row.get("result"))
    if result is not None:
        row["result"] = result
    return JobIdentityLookupResult.found(state, row)


__all__ = [
    "acquire_job",
    "apply_prepared_disposition",
    "ensure_lease_horizon",
    "find_job_by_identity",
    "release_job",
    "renew_lease",
]
