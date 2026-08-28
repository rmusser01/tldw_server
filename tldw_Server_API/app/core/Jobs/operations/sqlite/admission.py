"""SQLite-backed Jobs admission operations."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from tldw_Server_API.app.core.Jobs.operations.contracts import (
    AdmissionRejectionReason,
    AdmissionResult,
    CreateJobCommand,
    OperationOutcome,
)

_MAX_QUEUED_MESSAGE = "Quota exceeded: max queued per user/domain"
_SUBMITS_PER_MINUTE_MESSAGE = "Quota exceeded: submits per minute"
_EXECUTION_CONTROL_CONFLICT_MESSAGE = "Idempotent job execution controls conflict"


def _execution_controls_match(row: dict[str, Any], command: CreateJobCommand) -> bool:
    """Return whether an existing row has the requested immutable controls."""

    return (
        row.get("expired_lease_policy") == command.expired_lease_policy.value
        and row.get("quarantine_threshold") == command.quarantine_threshold
    )


def _sqlite_timestamp(value: datetime) -> str:
    """Return the SQLite timestamp representation used by the Jobs table."""

    normalized = value
    if value.tzinfo is not None:
        normalized = value.astimezone(timezone.utc).replace(tzinfo=None)
    return normalized.strftime("%Y-%m-%d %H:%M:%S")


def _future_available_at(value: datetime | None, *, now: datetime) -> datetime | None:
    """Keep only future schedule times; immediate jobs use a NULL ready marker."""

    if value is None:
        return None
    normalized_value = value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)
    normalized_now = now.replace(tzinfo=timezone.utc) if now.tzinfo is None else now.astimezone(timezone.utc)
    return normalized_value if normalized_value > normalized_now else None


def _row_to_dict(row: Any) -> dict[str, Any]:
    """Convert sqlite row-like values to plain dictionaries."""

    return dict(row) if row is not None else {}


def _created_event_fact(
    *,
    row: dict[str, Any],
    idempotent: bool,
    request_id: str | None,
    trace_id: str | None,
) -> dict[str, Any]:
    """Build the facade-facing fact for a persisted job.created event."""

    attrs = {
        "idempotent": idempotent,
        "owner_user_id": row.get("owner_user_id"),
        "retry_count": int(row.get("retry_count") or 0),
    }
    return {
        "event_type": "job.created",
        "attrs": attrs,
        "request_id": request_id,
        "trace_id": trace_id,
        "job_id": int(row.get("id")),
        "domain": row.get("domain"),
        "queue": row.get("queue"),
        "job_type": row.get("job_type"),
        "owner_user_id": row.get("owner_user_id"),
    }


def _insert_created_event(
    conn: sqlite3.Connection,
    *,
    row: dict[str, Any],
    idempotent: bool,
    request_id: str | None,
    trace_id: str | None,
) -> dict[str, Any]:
    """Persist a transactional job.created event and return its fact payload."""

    event = _created_event_fact(row=row, idempotent=idempotent, request_id=request_id, trace_id=trace_id)
    conn.execute(
        (
            "INSERT INTO job_events(job_id, domain, queue, job_type, event_type, attrs_json, "
            "owner_user_id, request_id, trace_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, DATETIME('now'))"
        ),
        (
            event["job_id"],
            event["domain"],
            event["queue"],
            event["job_type"],
            event["event_type"],
            json.dumps(event["attrs"]),
            event["owner_user_id"],
            event["request_id"],
            event["trace_id"],
        ),
    )
    return event


def _bump_counters(
    conn: sqlite3.Connection,
    *,
    command: CreateJobCommand,
    available_at_sql: str | None,
) -> None:
    """Increment ready/scheduled counters for a newly inserted job."""

    is_scheduled = bool(available_at_sql)
    conn.execute(
        (
            "INSERT INTO job_counters(domain,queue,job_type,ready_count,scheduled_count,processing_count,"
            "quarantined_count) VALUES(?,?,?,?,?,0,0) "
            "ON CONFLICT(domain,queue,job_type) DO UPDATE SET ready_count = ready_count + ?, "
            "scheduled_count = scheduled_count + ?, updated_at = DATETIME('now')"
        ),
        (
            command.domain,
            command.queue,
            command.job_type,
            0 if is_scheduled else 1,
            1 if is_scheduled else 0,
            0 if is_scheduled else 1,
            1 if is_scheduled else 0,
        ),
    )


def _quota_rejection(
    conn: sqlite3.Connection,
    *,
    command: CreateJobCommand,
    now_sql: str,
    max_queued_quota: int,
    submits_per_minute_quota: int,
) -> AdmissionResult | None:
    """Return a quota rejection result when admission exceeds configured limits."""

    if not command.owner_user_id:
        return None

    if max_queued_quota:
        row = conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE domain=? AND owner_user_id=? AND status='queued'",
            (command.domain, command.owner_user_id),
        ).fetchone()
        if int(row[0] if row else 0) >= max_queued_quota:
            return AdmissionResult.rejected(
                AdmissionRejectionReason.QUOTA_EXCEEDED,
                message=_MAX_QUEUED_MESSAGE,
            )

    if submits_per_minute_quota:
        row = conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE domain=? AND owner_user_id=? AND created_at >= DATETIME(?, '-60 seconds')",
            (command.domain, command.owner_user_id, now_sql),
        ).fetchone()
        if int(row[0] if row else 0) >= submits_per_minute_quota:
            return AdmissionResult.rejected(
                AdmissionRejectionReason.QUOTA_EXCEEDED,
                message=_SUBMITS_PER_MINUTE_MESSAGE,
            )

    return None


def _insert_job(
    conn: sqlite3.Connection,
    *,
    command: CreateJobCommand,
    uuid_value: str,
    payload_json: str,
    now_sql: str,
    available_at_sql: str | None,
    ignore_idempotency_conflict: bool,
) -> int | None:
    """Insert a queued job and return its row id when SQLite exposes one."""

    if ignore_idempotency_conflict:
        sql = """
        INSERT OR IGNORE INTO jobs (
          uuid, domain, queue, job_type, owner_user_id, project_id, batch_group,
          idempotency_key, payload, result, status, priority, max_retries,
          expired_lease_policy, quarantine_threshold, retry_count, available_at,
          created_at, updated_at, request_id, trace_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, 'queued', ?, ?, ?, ?, 0, ?, ?, ?, ?, ?)
        """
    else:
        sql = """
        INSERT INTO jobs (
          uuid, domain, queue, job_type, owner_user_id, project_id, batch_group,
          idempotency_key, payload, result, status, priority, max_retries,
          expired_lease_policy, quarantine_threshold, retry_count, available_at,
          created_at, updated_at, request_id, trace_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, 'queued', ?, ?, ?, ?, 0, ?, ?, ?, ?, ?)
        """
    before_changes = int(getattr(conn, "total_changes", 0))
    conn.execute(
        sql,
        (
            uuid_value,
            command.domain,
            command.queue,
            command.job_type,
            command.owner_user_id,
            command.project_id,
            command.batch_group,
            command.idempotency_key,
            payload_json,
            command.priority,
            command.max_retries,
            command.expired_lease_policy.value,
            command.quarantine_threshold,
            available_at_sql,
            now_sql,
            now_sql,
            command.request_id,
            command.trace_id,
        ),
    )
    if int(getattr(conn, "total_changes", 0)) <= before_changes:
        return None
    row = conn.execute("SELECT last_insert_rowid()").fetchone()
    return int(row[0]) if row else None


def create_job_admission(
    conn: sqlite3.Connection,
    *,
    command: CreateJobCommand,
    uuid_value: str,
    now: datetime,
    max_queued_quota: int,
    submits_per_minute_quota: int,
    counters_enabled: bool,
    begin_immediate: bool = False,
    pre_admission_lookup: Callable[
        [sqlite3.Connection], dict[str, Any] | None
    ] | None = None,
) -> AdmissionResult:
    """Create or replay a queued job admission inside a SQLite transaction."""

    payload_json = json.dumps(command.payload)
    now_sql = _sqlite_timestamp(now)
    available_at = _future_available_at(command.available_at, now=now)
    available_at_sql = _sqlite_timestamp(available_at) if available_at else None

    quota_enabled = bool(command.owner_user_id and (max_queued_quota or submits_per_minute_quota))
    if quota_enabled or begin_immediate:
        conn.execute("BEGIN IMMEDIATE")

    with conn:
        if pre_admission_lookup is not None:
            existing = pre_admission_lookup(conn)
            if existing is not None:
                return AdmissionResult.existing(row=existing)
        idempotent_replay = False
        if quota_enabled and command.idempotency_key:
            row = conn.execute(
                "SELECT 1 FROM jobs WHERE domain = ? AND queue = ? AND job_type = ? AND idempotency_key = ?",
                (command.domain, command.queue, command.job_type, command.idempotency_key),
            ).fetchone()
            idempotent_replay = row is not None

        if not idempotent_replay:
            quota_result = _quota_rejection(
                conn,
                command=command,
                now_sql=now_sql,
                max_queued_quota=max_queued_quota,
                submits_per_minute_quota=submits_per_minute_quota,
            )
            if quota_result is not None:
                return quota_result

        if command.idempotency_key:
            row_id = _insert_job(
                conn,
                command=command,
                uuid_value=uuid_value,
                payload_json=payload_json,
                now_sql=now_sql,
                available_at_sql=available_at_sql,
                ignore_idempotency_conflict=True,
            )
            inserted = row_id is not None
            row = _row_to_dict(
                conn.execute(
                    "SELECT * FROM jobs WHERE domain = ? AND queue = ? AND job_type = ? AND idempotency_key = ?",
                    (command.domain, command.queue, command.job_type, command.idempotency_key),
                ).fetchone()
            )
            if not row:
                row = {
                    "uuid": uuid_value,
                    "status": "queued",
                    "domain": command.domain,
                    "queue": command.queue,
                    "job_type": command.job_type,
                }
            if not inserted and not _execution_controls_match(row, command):
                return AdmissionResult(
                    outcome=OperationOutcome.BACKEND_CONFLICT,
                    row=row,
                    message=_EXECUTION_CONTROL_CONFLICT_MESSAGE,
                )
            if inserted and counters_enabled:
                _bump_counters(conn, command=command, available_at_sql=available_at_sql)
            event = _insert_created_event(
                conn,
                row=row,
                idempotent=not inserted,
                request_id=command.request_id,
                trace_id=command.trace_id,
            )
            if inserted:
                return AdmissionResult.applied(row=row, durable_events=(event,))
            return AdmissionResult.existing(row=row, durable_events=(event,))

        job_id = _insert_job(
            conn,
            command=command,
            uuid_value=uuid_value,
            payload_json=payload_json,
            now_sql=now_sql,
            available_at_sql=available_at_sql,
            ignore_idempotency_conflict=False,
        )
        row = _row_to_dict(conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone())
        if not row:
            row = {
                "id": job_id,
                "uuid": uuid_value,
                "status": "queued",
                "domain": command.domain,
                "queue": command.queue,
                "job_type": command.job_type,
            }
        if counters_enabled:
            _bump_counters(conn, command=command, available_at_sql=available_at_sql)
        event = _insert_created_event(
            conn,
            row=row,
            idempotent=False,
            request_id=command.request_id,
            trace_id=command.trace_id,
        )
        return AdmissionResult.applied(row=row, durable_events=(event,))
