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
    """Ensure the partial index for explicit retry admissions; return None.

    Args:
        executor: SQLite connection/cursor with ``execute``, or a psycopg 3
            cursor with ``execute`` and tuple-row ``fetchone`` on an autocommit
            connection. The Jobs ``job_events`` table must already exist in
            the current schema; PostgreSQL also reads system catalogs/settings.
        backend: Exactly ``"sqlite"`` or ``"postgres"``.

    Ownership and side effects:
        The caller owns and closes the executor/connection. This helper does
        not begin, commit or roll back a transaction. SQLite executes CREATE
        INDEX IF NOT EXISTS in the caller's schema transaction. PostgreSQL's
        concurrent-index phase requires autocommit, acquires a session advisory
        lock, verifies ready indexes and repairs only the exact owned definition
        after failed builds. It releases the lock best-effort, suppressing only
        psycopg errors from unlock. Foreign definition collisions fail closed.
        Lock contention uses the minimum positive lock/statement timeout, or
        30s when both are zero. No Jobs rows, counters or events are changed;
        PostgreSQL failed concurrent DDL can leave an invalid index for retry.

    Raises:
        ValueError: Unsupported backend.
        RuntimeError: PostgreSQL advisory-lock timeout, foreign index collision
            or failed index verification.
        ImportError: The PostgreSQL branch cannot import psycopg.
        sqlite3.Error / psycopg.Error: Native DDL, catalog, timeout, permission
            or transaction-mode failures propagate unchanged (except unlock).
        Other executor/result errors also propagate; there are no callbacks.
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
    """Return the initial-job plus explicit retry count in a 60-second window.

    Args:
        executor: SQLite connection/cursor whose ``execute`` returns a cursor
            with ``fetchone``, or a psycopg 3 cursor with both methods. Count
            rows may be positional (including sqlite3.Row) or dicts with ``c``.
        backend: Exactly ``"sqlite"`` or ``"postgres"``.
        domain: Jobs domain string matched exactly in both tables.
        owner_user_id: Owner string matched exactly; no unscoped fallback.
        now: SQLite-compatible datetime/timestamp string, or a PostgreSQL
            datetime, used as the inclusive lower boundary ``now - 60s``.

    Ownership and side effects:
        The caller holds the existing owner quota lock and owns the transaction,
        executor and connection. This helper neither acquires that lock nor
        begins/ends transactions or closes resources. It performs one read-only
        SELECT against jobs/job_events. The nonnegative integer sum counts
        initial jobs and ``job.retry_admitted`` events, excluding legacy admin
        retry events. There is no upper time boundary or status filter; a queued
        replay contributes no new event, but prior admissions still count.

    Raises:
        ValueError: Unsupported backend, or failed integer conversion.
        sqlite3.Error / psycopg.Error: Native query, binding, missing-schema and
            connection failures propagate unchanged.
        Malformed executor/result shape errors propagate; no callbacks run.
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
    """Transactionally admit one exactly matched failed Job, returning facts.

    Args:
        conn: Idle SQLite connection with sqlite3.Row-compatible mapping rows,
            or an idle, non-autocommit psycopg 3 connection. Do not call inside
            an existing transaction. The caller supplies initialized Jobs schema.
        backend: Exactly ``"sqlite"`` or ``"postgres"``; this lower-level helper
            indexes the backend map directly, unlike the other public helpers.
        cursor_factory: PostgreSQL connection-to-context-manager callable
            yielding a cursor with mapping rows; ignored on SQLite, which
            creates and closes its own cursor.
        command: CreateJobCommand containing the expected domain, queue,
            job_type, owner_user_id, idempotency_key and payload. Its decoded
            payload must be an exactly matching dict with integer-only values
            (bools excluded). Other command fields do not replace Job controls.
        job_id: Target Jobs integer ID, scoped by command.owner_user_id.
        expected_uuid: Exact immutable UUID string for that row.
        now: Admission datetime; SQLite stores its normalized timestamp.
        max_queued_quota: Domain/owner queued limit; zero disables it.
        submits_per_minute_quota: Domain/owner 60-second admission limit;
            zero disables it. Quota values are expected to be nonnegative;
            this helper does not validate them.
        counters_enabled: Whether to increment the ready counter on admission.
        decode_payload: Callable accepting persisted payload (JSON strings are
            parsed first) and returning its decoded value; runs even on replay.
        check_policy: Zero-argument policy callback, run only after exact
            identity and failed/non-cancelled checks, before queue/quota checks.

    Ownership and side effects:
        The caller creates the connection and must close it on all exit paths.
        This helper owns the transaction: SQLite BEGIN IMMEDIATE serializes
        writes; PostgreSQL temporarily selects READ COMMITTED, shares create's
        owner advisory lock, then locks the Job and queue-control row. Its
        connection context commits on normal return and rolls back exceptions.
        SQLite leaves conn open; psycopg 3's context closes it. Isolation is
        restored only if the PostgreSQL connection remains open. Errors before
        entering the connection context leave cleanup to the caller.

        An applied AdmissionResult contains the requeued row with decoded
        payload: retry_count resets to zero, execution/result/error state clears,
        updated_at advances, one job.retry_admitted event is inserted, and ready
        counters optionally advance. ID, UUID, payload, idempotency key and
        immutable execution controls are preserved; no new Job is inserted.
        APPLIED's was_inserted flag follows the existing AdmissionResult
        convention, not physical insertion. Matching queued/processing rows
        return an existing/no-transition result before policy checks, without
        a new event or charge. Quotas return an admission-rejected result rather
        than raising. A quota rejection may commit a newly created queue-control
        row, but does not change the Job, counters or admission events.

    Raises:
        KeyError: Unsupported backend.
        ValueError: Identity/payload mismatch, non-failed/cancel-requested Job,
            paused/draining queue, or lost conditional transition. Malformed
            persisted JSON raises json.JSONDecodeError (a ValueError).
        RuntimeError: PostgreSQL transaction support cannot import psycopg.
        sqlite3.Error / psycopg.Error: Native SQL, binding, connection, isolation,
            locking and bookkeeping failures propagate unchanged.
        Exceptions from cursor_factory, decode_payload and check_policy
            (including policy rejection) propagate unchanged, as do malformed
            row/result errors. Once the transaction context is entered, errors
            roll back Job/counter/event writes; this helper does not translate
            errors into SDK dispositions or facade HTTP/policy exceptions.
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
