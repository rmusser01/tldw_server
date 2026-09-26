"""Conditional, owner-scoped Jobs retry admission and submission accounting."""

from __future__ import annotations

import json
from collections.abc import Callable
from contextlib import AbstractContextManager, closing, nullcontext, suppress
from datetime import datetime
from time import monotonic, sleep
from typing import Any

from tldw_Server_API.app.core.Jobs.operations.contracts import AdmissionResult, CreateJobCommand

_PG_RETRY_ADMISSION_INDEX_LOCK = "tldw.jobs.job_events.retry_admission_index.v1"

_SQL = {
    "sqlite": {
        "job": "SELECT * FROM jobs WHERE id=? AND owner_user_id=?",
        "control_insert": "INSERT OR IGNORE INTO job_queue_controls(domain,queue,paused,drain) VALUES(?,?,0,0)",
        "control": "SELECT paused,drain FROM job_queue_controls WHERE domain=? AND queue=?",
        "retry": """UPDATE jobs SET status='queued', retry_count=0, available_at=NULL,
            result=NULL, completed_at=NULL, started_at=NULL, acquired_at=NULL,
            worker_id=NULL, lease_id=NULL, leased_until=NULL, completion_token=NULL,
            last_error=NULL, error_message=NULL, error_code=NULL, error_class=NULL,
            error_stack=NULL, failure_streak_code=NULL, failure_streak_count=0,
            updated_at=? WHERE id=? AND owner_user_id=? AND uuid=? AND status='failed'""",
        "event": """INSERT INTO job_events(job_id,domain,queue,job_type,event_type,attrs_json,
            owner_user_id,request_id,trace_id,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)""",
    },
    "postgres": {
        "job": "SELECT * FROM jobs WHERE id=%s AND owner_user_id=%s FOR UPDATE",
        "control_insert": """INSERT INTO job_queue_controls(domain,queue,paused,drain)
            VALUES(%s,%s,FALSE,FALSE) ON CONFLICT(domain,queue) DO NOTHING""",
        "control": "SELECT paused,drain FROM job_queue_controls WHERE domain=%s AND queue=%s FOR SHARE",
        "retry": """UPDATE jobs SET status='queued', retry_count=0, available_at=NULL,
            result=NULL, completed_at=NULL, started_at=NULL, acquired_at=NULL,
            worker_id=NULL, lease_id=NULL, leased_until=NULL, completion_token=NULL,
            last_error=NULL, error_message=NULL, error_code=NULL, error_class=NULL,
            error_stack=NULL, failure_streak_code=NULL, failure_streak_count=0,
            updated_at=%s WHERE id=%s AND owner_user_id=%s AND uuid=%s AND status='failed'""",
        "event": """INSERT INTO job_events(job_id,domain,queue,job_type,event_type,attrs_json,
            owner_user_id,request_id,trace_id,created_at) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
    },
}


def _pg_retry_admission_index_state(executor: Any) -> tuple[bool, bool, bool] | None:
    """Return exact owned definition, validity and readiness for the literal index."""
    executor.execute(
        """SELECT CASE WHEN i.indexrelid IS NULL THEN FALSE ELSE (
            i.indrelid = to_regclass(format('%I.job_events', current_schema()))
            AND pg_get_indexdef(i.indexrelid) = format(
                'CREATE INDEX idx_job_events_retry_admissions ON %I.job_events USING btree '
                '(domain, owner_user_id, created_at) WHERE (event_type = ''job.retry_admitted''::text)',
                current_schema())
            AND NOT EXISTS (SELECT 1 FROM pg_constraint con WHERE con.conindid=i.indexrelid)
        ) END, i.indisvalid, i.indisready
        FROM pg_class idx JOIN pg_namespace ns ON ns.oid=idx.relnamespace
        LEFT JOIN pg_index i ON i.indexrelid=idx.oid
        WHERE ns.nspname=current_schema() AND idx.relname='idx_job_events_retry_admissions'"""
    )
    row = executor.fetchone()
    return (bool(row[0]), bool(row[1]), bool(row[2])) if row is not None else None


def _lock_pg_retry_admission_index(executor: Any) -> None:
    """Acquire without a held snapshot, using positive timeouts or a 30s fallback."""
    executor.execute(
        "SELECT MIN(NULLIF(setting,'0')::bigint) FROM pg_settings "
        "WHERE name IN ('lock_timeout','statement_timeout')"
    )
    deadline = monotonic() + float(executor.fetchone()[0] or 30_000) / 1000
    # A blocking advisory-lock SELECT retains a snapshot that a partial index
    # builder can wait for. Each autocommit try releases its snapshot instead.
    while True:
        executor.execute(
            "SELECT pg_try_advisory_lock(hashtextextended(%s, 0))", (_PG_RETRY_ADMISSION_INDEX_LOCK,),
        )
        if executor.fetchone()[0]:
            return
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise RuntimeError("Jobs retry-admission index advisory lock timeout")
        sleep(min(0.05, remaining))


def ensure_retry_admission_index(executor: Any, *, backend: str) -> None:
    """Index only explicit retry admissions through the existing schema ensure.

    PostgreSQL callers use an autocommit connection for its established
    concurrent-index phase; SQLite callers own the schema transaction.
    Repeated ensures preserve ready indexes. PostgreSQL repairs only this exact
    owned definition after failed concurrent builds, under a session advisory
    lock and the caller's configured timeouts. Foreign collisions fail closed.
    Lock contention uses the minimum positive timeout, or 30s if both are zero.
    """
    if backend == "sqlite":
        executor.execute(
            "CREATE INDEX IF NOT EXISTS idx_job_events_retry_admissions "
            "ON job_events(domain,owner_user_id,created_at) WHERE event_type='job.retry_admitted'"
        )
    elif backend == "postgres":
        import psycopg

        _lock_pg_retry_admission_index(executor)
        try:
            state = _pg_retry_admission_index_state(executor)
            if state is not None and not state[0]:
                raise RuntimeError("Jobs retry-admission index definition collision; refusing replacement")
            if state is None or not all(state[1:]):
                if state is not None:
                    executor.execute("DROP INDEX CONCURRENTLY IF EXISTS idx_job_events_retry_admissions")
                executor.execute(
                    "CREATE INDEX CONCURRENTLY idx_job_events_retry_admissions "
                    "ON job_events(domain,owner_user_id,created_at) WHERE event_type='job.retry_admitted'"
                )
            if _pg_retry_admission_index_state(executor) != (True, True, True):
                raise RuntimeError("Jobs retry-admission index verification failed")
        finally:
            with suppress(psycopg.Error):
                executor.execute(
                    "SELECT pg_advisory_unlock(hashtextextended(%s, 0))", (_PG_RETRY_ADMISSION_INDEX_LOCK,),
                )
    else:
        raise ValueError("Unsupported Jobs admission backend")


def count_recent_job_admissions(
    executor: Any, *, backend: str, domain: str, owner_user_id: str, now: datetime | str,
) -> int:
    """Count initial Jobs and explicit same-row retry admissions in the rate window.

    Called under the existing owner quota lock; legacy admin retry events are
    deliberately excluded. A duplicate queued replay emits no admission event.
    """
    if backend == "sqlite":
        row = executor.execute(
            """SELECT (
                SELECT COUNT(*) FROM jobs WHERE domain=? AND owner_user_id=?
                    AND created_at >= DATETIME(?, '-60 seconds')
            ) + (
                SELECT COUNT(*) FROM job_events WHERE domain=? AND owner_user_id=?
                    AND event_type='job.retry_admitted' AND created_at >= DATETIME(?, '-60 seconds')
            ) AS c""",
            (domain, owner_user_id, now, domain, owner_user_id, now),
        ).fetchone()
    elif backend == "postgres":
        executor.execute(
            """SELECT (
                SELECT COUNT(*) FROM jobs WHERE domain=%s AND owner_user_id=%s
                    AND created_at >= (%s - interval '60 seconds')
            ) + (
                SELECT COUNT(*) FROM job_events WHERE domain=%s AND owner_user_id=%s
                    AND event_type='job.retry_admitted' AND created_at >= (%s - interval '60 seconds')
            ) AS c""",
            (domain, owner_user_id, now, domain, owner_user_id, now),
        )
        row = executor.fetchone()
    else:
        raise ValueError("Unsupported Jobs admission backend")
    return int(row["c"] if isinstance(row, dict) else row[0])


def retry_failed_job_admission(
    conn: Any,
    *,
    backend: str,
    cursor_factory: Callable[[Any], AbstractContextManager[Any]],
    command: CreateJobCommand,
    job_id: int,
    expected_uuid: str,
    now: datetime,
    max_queued_quota: int,
    submits_per_minute_quota: int,
    counters_enabled: bool,
    decode_payload: Callable[[Any], Any],
    check_policy: Callable[[], None],
) -> AdmissionResult:
    """Requeue only an exactly matched failed Job with transactional admission.

    SQLite uses its write lock; PostgreSQL shares create's owner advisory lock
    and READ COMMITTED isolation, then locks the Job and queue-control row.
    Concurrent queued/processing replays precede policy checks. Any rejection
    or bookkeeping error leaves state, counters and events unchanged.
    """
    # Reuse create's serialized policy and counter operations, without insertion.
    from tldw_Server_API.app.core.Jobs.operations.postgres import admission as pg
    from tldw_Server_API.app.core.Jobs.operations.sqlite import admission as sqlite

    sql = _SQL[backend]
    if backend == "sqlite":
        conn.execute("BEGIN IMMEDIATE")
        transaction_policy = nullcontext()
        now_value = sqlite._sqlite_timestamp(now)
    else:
        transaction_policy = pg._read_committed_quota_transaction(conn, enabled=True)
        now_value = now
    with transaction_policy, conn:
        cursor_context = cursor_factory(conn) if backend == "postgres" else closing(conn.cursor())
        with cursor_context as cur:
            if backend == "postgres":
                cur.execute("SELECT pg_advisory_xact_lock(%s)", (pg._quota_lock_key(command),))
            cur.execute(sql["job"], (job_id, command.owner_user_id))
            found = cur.fetchone()
            if found is None:
                raise ValueError("Jobs retry identity mismatch")
            row = dict(found)
            payload = row.get("payload")
            if isinstance(payload, str):
                payload = json.loads(payload)
            payload = decode_payload(payload)
            if (
                row.get("uuid") != expected_uuid
                or row.get("domain") != command.domain
                or row.get("queue") != command.queue
                or row.get("job_type") != command.job_type
                or row.get("owner_user_id") != command.owner_user_id
                or row.get("idempotency_key") != command.idempotency_key
                or payload != command.payload
                or not isinstance(payload, dict)
                or any(type(value) is not int for value in payload.values())
            ):
                raise ValueError("Jobs retry identity mismatch")
            row["payload"] = payload
            if row["status"] in {"queued", "processing"}:
                return AdmissionResult.existing(row=row)
            if row["status"] != "failed" or row.get("cancel_requested_at") is not None:
                raise ValueError("Jobs retry requires a failed job")
            check_policy()
            cur.execute(sql["control_insert"], (command.domain, command.queue))
            cur.execute(sql["control"], (command.domain, command.queue))
            flags = dict(cur.fetchone())
            if flags["paused"] or flags["drain"]:
                raise ValueError("Jobs queue paused or draining")
            if backend == "sqlite":
                rejection = sqlite._quota_rejection(
                    conn, command=command, now_sql=now_value,
                    max_queued_quota=max_queued_quota, submits_per_minute_quota=submits_per_minute_quota,
                )
            else:
                rejection = pg._quota_rejection(
                    cur, command=command, now=now,
                    max_queued_quota=max_queued_quota, submits_per_minute_quota=submits_per_minute_quota,
                )
            if rejection is not None:
                return rejection
            cur.execute(sql["retry"], (now_value, job_id, command.owner_user_id, expected_uuid))
            if cur.rowcount != 1:
                raise ValueError("Jobs retry transition lost")
            cur.execute(sql["event"], (
                job_id, command.domain, command.queue, command.job_type, "job.retry_admitted",
                json.dumps({"previous_retry_count": row["retry_count"], "retry_count": 0}),
                command.owner_user_id, row.get("request_id"), row.get("trace_id"), now_value,
            ))
            if counters_enabled:
                if backend == "sqlite":
                    sqlite._bump_counters(conn, command=command, available_at_sql=None)
                else:
                    pg._bump_counters(cur, command=command, available_at=None)
            cur.execute(sql["job"], (job_id, command.owner_user_id))
            updated = dict(cur.fetchone())
            updated["payload"] = payload
            return AdmissionResult.applied(row=updated)
