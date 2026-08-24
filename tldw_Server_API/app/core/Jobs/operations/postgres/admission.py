"""Postgres-backed Jobs admission operations."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from contextlib import AbstractContextManager, contextmanager
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.Jobs.operations.contracts import (
    AdmissionRejectionReason,
    AdmissionResult,
    CreateJobCommand,
)

_MAX_QUEUED_MESSAGE = "Quota exceeded: max queued per user/domain"
_SUBMITS_PER_MINUTE_MESSAGE = "Quota exceeded: submits per minute"
_PSYCOPG_REQUIRED_MESSAGE = "psycopg is required for PostgreSQL quota admission"
_IDEMPOTENT_CONFLICT_ATTEMPTS = 3
_IDEMPOTENT_CONFLICT_LOST_MESSAGE = "Idempotent job conflict repeatedly disappeared during admission"

try:
    import psycopg as _psycopg  # type: ignore
except ImportError:  # pragma: no cover - optional dependency path
    _psycopg = None
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


@contextmanager
def _read_committed_quota_transaction(conn: Any, *, enabled: bool):
    if not enabled:
        yield
        return

    if _psycopg is None:
        raise RuntimeError(_PSYCOPG_REQUIRED_MESSAGE)

    previous_isolation = conn.isolation_level
    conn.isolation_level = _psycopg.IsolationLevel.READ_COMMITTED
    try:
        yield
    finally:
        if not getattr(conn, "closed", False):
            conn.isolation_level = previous_isolation


def _quota_lock_key(command: CreateJobCommand) -> int:
    material = f"jobs:admission-quota\x00{command.domain}\x00{command.owner_user_id}".encode()
    digest = hashlib.blake2b(material, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=True)


def _count_from_row(row: Any) -> int:
    if row is None:
        return 0
    if isinstance(row, dict):
        return int(row.get("c") or 0)
    return int(row[0] or 0)


def _row_to_dict(row: Any) -> dict[str, Any]:
    return dict(row) if row is not None else {}


def _future_available_at(value: datetime | None, *, now: datetime) -> datetime | None:
    """Keep only future schedule times; immediate jobs use a NULL ready marker."""

    if value is None:
        return None
    normalized_value = value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)
    normalized_now = now.replace(tzinfo=timezone.utc) if now.tzinfo is None else now.astimezone(timezone.utc)
    return normalized_value if normalized_value > normalized_now else None


def _created_event_fact(
    *,
    row: dict[str, Any],
    idempotent: bool,
    request_id: str | None,
    trace_id: str | None,
) -> dict[str, Any]:
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
    cur: Any,
    *,
    row: dict[str, Any],
    idempotent: bool,
    request_id: str | None,
    trace_id: str | None,
) -> dict[str, Any]:
    event = _created_event_fact(row=row, idempotent=idempotent, request_id=request_id, trace_id=trace_id)
    cur.execute(
        (
            "INSERT INTO job_events("
            "job_id, domain, queue, job_type, event_type, attrs_json, owner_user_id, request_id, trace_id, created_at"
            ") VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())"
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
    cur: Any,
    *,
    command: CreateJobCommand,
    available_at: datetime | None,
) -> None:
    is_scheduled = bool(available_at)
    cur.execute(
        (
            "INSERT INTO job_counters(domain,queue,job_type,ready_count,scheduled_count,processing_count,quarantined_count) VALUES(%s,%s,%s,%s,%s,0,0) "
            "ON CONFLICT (domain,queue,job_type) DO UPDATE SET ready_count = job_counters.ready_count + EXCLUDED.ready_count, scheduled_count = job_counters.scheduled_count + EXCLUDED.scheduled_count, updated_at = NOW()"
        ),
        (
            command.domain,
            command.queue,
            command.job_type,
            0 if is_scheduled else 1,
            1 if is_scheduled else 0,
        ),
    )


def _bump_counters_best_effort(
    cur: Any,
    *,
    command: CreateJobCommand,
    available_at: datetime | None,
) -> None:
    cur.execute("SAVEPOINT jobs_admission_counter_update")
    try:
        _bump_counters(cur, command=command, available_at=available_at)
    except _COUNTER_NONCRITICAL_ERRORS as exc:
        cur.execute("ROLLBACK TO SAVEPOINT jobs_admission_counter_update")
        cur.execute("RELEASE SAVEPOINT jobs_admission_counter_update")
        logger.warning(
            "Non-critical Postgres jobs counter update failed for {}:{}:{}: {}",
            command.domain,
            command.queue,
            command.job_type,
            exc,
        )
    else:
        cur.execute("RELEASE SAVEPOINT jobs_admission_counter_update")


def _quota_rejection(
    cur: Any,
    *,
    command: CreateJobCommand,
    now: datetime,
    max_queued_quota: int,
    submits_per_minute_quota: int,
) -> AdmissionResult | None:
    if not command.owner_user_id:
        return None

    if max_queued_quota:
        cur.execute(
            "SELECT COUNT(*) AS c FROM jobs WHERE domain=%s AND owner_user_id=%s AND status='queued'",
            (command.domain, command.owner_user_id),
        )
        if _count_from_row(cur.fetchone()) >= max_queued_quota:
            return AdmissionResult.rejected(
                AdmissionRejectionReason.QUOTA_EXCEEDED,
                message=_MAX_QUEUED_MESSAGE,
            )

    if submits_per_minute_quota:
        cur.execute(
            "SELECT COUNT(*) AS c FROM jobs WHERE domain=%s AND owner_user_id=%s AND created_at >= (%s - interval '60 seconds')",
            (command.domain, command.owner_user_id, now),
        )
        if _count_from_row(cur.fetchone()) >= submits_per_minute_quota:
            return AdmissionResult.rejected(
                AdmissionRejectionReason.QUOTA_EXCEEDED,
                message=_SUBMITS_PER_MINUTE_MESSAGE,
            )

    return None


def _insert_job(
    cur: Any,
    *,
    command: CreateJobCommand,
    uuid_value: str,
    payload_json: str,
    available_at: datetime | None,
    idempotent_insert: bool,
) -> dict[str, Any] | None:
    if idempotent_insert:
        sql = (
            "INSERT INTO jobs (uuid, domain, queue, job_type, owner_user_id, project_id, batch_group, idempotency_key, payload, result, status, priority, max_retries, retry_count, available_at, created_at, updated_at, request_id, trace_id) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, NULL, 'queued', %s, %s, 0, %s, NOW(), NOW(), %s, %s) "
            "ON CONFLICT (domain, queue, job_type, idempotency_key) DO NOTHING RETURNING *"
        )
    else:
        sql = (
            "INSERT INTO jobs (uuid, domain, queue, job_type, owner_user_id, project_id, batch_group, idempotency_key, payload, result, status, priority, max_retries, retry_count, available_at, created_at, updated_at, request_id, trace_id) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, NULL, 'queued', %s, %s, 0, %s, NOW(), NOW(), %s, %s) RETURNING *"
        )
    cur.execute(
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
            available_at,
            command.request_id,
            command.trace_id,
        ),
    )
    row = cur.fetchone()
    return _row_to_dict(row) if row else None


def _insert_or_lock_idempotent_job(
    cur: Any,
    *,
    command: CreateJobCommand,
    uuid_value: str,
    payload_json: str,
    available_at: datetime | None,
) -> tuple[dict[str, Any], bool]:
    for _ in range(_IDEMPOTENT_CONFLICT_ATTEMPTS):
        row = _insert_job(
            cur,
            command=command,
            uuid_value=uuid_value,
            payload_json=payload_json,
            available_at=available_at,
            idempotent_insert=True,
        )
        if row is not None:
            return row, True

        cur.execute(
            "SELECT * FROM jobs WHERE domain = %s AND queue = %s AND job_type = %s "
            "AND idempotency_key = %s FOR KEY SHARE",
            (command.domain, command.queue, command.job_type, command.idempotency_key),
        )
        row = _row_to_dict(cur.fetchone())
        if row:
            return row, False

    raise RuntimeError(_IDEMPOTENT_CONFLICT_LOST_MESSAGE)


def create_job_admission(
    conn: Any,
    cursor_factory: Callable[[Any], AbstractContextManager[Any]],
    *,
    command: CreateJobCommand,
    uuid_value: str,
    now: datetime,
    max_queued_quota: int,
    submits_per_minute_quota: int,
    counters_enabled: bool,
    advisory_xact_lock_key: int | None = None,
    pre_admission_lookup: Callable[[Any], dict[str, Any] | None] | None = None,
) -> AdmissionResult:
    """Create or replay a queued job admission inside a Postgres transaction."""

    payload_json = json.dumps(command.payload)
    available_at = _future_available_at(command.available_at, now=now)
    quota_enabled = bool(command.owner_user_id and (max_queued_quota or submits_per_minute_quota))

    with _read_committed_quota_transaction(conn, enabled=quota_enabled), conn:
        with cursor_factory(conn) as cur:
            if advisory_xact_lock_key is not None:
                cur.execute(
                    "SELECT pg_advisory_xact_lock(%s)",
                    (int(advisory_xact_lock_key),),
                )
            if pre_admission_lookup is not None:
                existing = pre_admission_lookup(cur)
                if existing is not None:
                    return AdmissionResult.existing(row=existing)
            if quota_enabled:
                cur.execute("SELECT pg_advisory_xact_lock(%s)", (_quota_lock_key(command),))

            idempotent_replay = False
            if quota_enabled and command.idempotency_key:
                cur.execute(
                    "SELECT 1 FROM jobs WHERE domain = %s AND queue = %s AND job_type = %s AND idempotency_key = %s FOR KEY SHARE",
                    (command.domain, command.queue, command.job_type, command.idempotency_key),
                )
                idempotent_replay = cur.fetchone() is not None

            if not idempotent_replay:
                quota_result = _quota_rejection(
                    cur,
                    command=command,
                    now=now,
                    max_queued_quota=max_queued_quota,
                    submits_per_minute_quota=submits_per_minute_quota,
                )
                if quota_result is not None:
                    return quota_result

            if command.idempotency_key:
                row, inserted = _insert_or_lock_idempotent_job(
                    cur,
                    command=command,
                    uuid_value=uuid_value,
                    payload_json=payload_json,
                    available_at=available_at,
                )
                if inserted and counters_enabled:
                    _bump_counters_best_effort(cur, command=command, available_at=available_at)
                event = _insert_created_event(
                    cur,
                    row=row,
                    idempotent=not inserted,
                    request_id=command.request_id,
                    trace_id=command.trace_id,
                )
                if inserted:
                    return AdmissionResult.applied(row=row, durable_events=(event,))
                return AdmissionResult.existing(row=row, durable_events=(event,))

            row = _insert_job(
                cur,
                command=command,
                uuid_value=uuid_value,
                payload_json=payload_json,
                available_at=available_at,
                idempotent_insert=False,
            )
            if not row:
                row = {
                    "uuid": uuid_value,
                    "status": "queued",
                    "domain": command.domain,
                    "queue": command.queue,
                    "job_type": command.job_type,
                }
            if counters_enabled:
                _bump_counters_best_effort(cur, command=command, available_at=available_at)
            event = _insert_created_event(
                cur,
                row=row,
                idempotent=False,
                request_id=command.request_id,
                trace_id=command.trace_id,
            )
            return AdmissionResult.applied(row=row, durable_events=(event,))
