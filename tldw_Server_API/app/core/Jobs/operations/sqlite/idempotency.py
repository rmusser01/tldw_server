"""SQLite admission for durable owner-scoped idempotent operations."""

from __future__ import annotations

import contextlib
import json
import secrets
import sqlite3
from datetime import datetime, timedelta
from typing import Any

from tldw_Server_API.app.core.Jobs.migrations import (
    SLIDES_ARCHIVE_EXACT_FIELDS,
    SlidesArchiveNormalizationError,
    normalize_slides_archive_projection,
)
from tldw_Server_API.app.core.Jobs.operations.contracts import (
    IdempotentOperationAdmission,
    IdempotentOperationCommand,
    IdempotentOperationConflict,
    IdempotentOperationConflictReason,
    IdempotentOperationUnavailableError,
)

from .admission import (
    _bump_counters,
    _future_available_at,
    _insert_created_event,
    _insert_job,
    _quota_rejection,
    _sqlite_timestamp,
)


def _row_to_dict(row: Any) -> dict[str, Any]:
    return dict(row) if row is not None else {}


def _find_exact_receipt(
    conn: sqlite3.Connection,
    command: IdempotentOperationCommand,
) -> dict[str, Any] | None:
    row = conn.execute(
        """
        SELECT *
        FROM job_idempotency_receipts
        WHERE domain = ? AND queue = ? AND job_type = ?
          AND owner_user_id = ? AND key_digest = ?
        """,
        (
            command.job.domain,
            command.job.queue,
            command.job.job_type,
            command.job.owner_user_id,
            command.key_digest,
        ),
    ).fetchone()
    return _row_to_dict(row) or None


def get_job_or_archived_by_uuid(
    conn: sqlite3.Connection,
    job_uuid: str,
    *,
    domain: str | None = None,
    owner_user_id: str | None = None,
) -> dict[str, Any] | None:
    """Read one UUID from active/archive storage in a single DB snapshot."""

    if not isinstance(job_uuid, str) or not job_uuid.strip() or len(job_uuid) > 200:
        raise ValueError("job_uuid must be a non-empty string of at most 200 characters")

    clauses = ["uuid = ?"]
    filter_values: list[Any] = [job_uuid]
    if domain is not None:
        clauses.append("domain = ?")
        filter_values.append(domain)
    if owner_user_id is not None:
        clauses.append("owner_user_id = ?")
        filter_values.append(owner_user_id)
    where_sql = " AND ".join(clauses)
    projection = ", ".join(("id", *SLIDES_ARCHIVE_EXACT_FIELDS))
    rows = conn.execute(
        f"""
        SELECT {projection}, NULL AS payload_compressed,
               NULL AS result_compressed, 0 AS archived
        FROM jobs
        WHERE {where_sql}
        UNION ALL
        SELECT {projection}, payload_compressed,
               result_compressed, 1 AS archived
        FROM jobs_archive
        WHERE {where_sql}
        """,  # nosec B608
        (*filter_values, *filter_values),
    ).fetchall()
    if len(rows) > 1:
        raise IdempotentOperationUnavailableError(
            "job UUID does not resolve to exactly one Job"
        )
    if not rows:
        return None
    job = None
    with contextlib.suppress(SlidesArchiveNormalizationError):
        job = normalize_slides_archive_projection(rows[0])
    if job is None:
        raise IdempotentOperationUnavailableError(
            "job archive projection is unavailable"
        )
    job["archived"] = bool(job.get("archived"))
    return job


def get_job_or_archived_by_idempotency_key(
    conn: sqlite3.Connection,
    *,
    idempotency_key: str,
    domain: str,
    queue: str,
    job_type: str,
    owner_user_id: str,
) -> dict[str, Any] | None:
    """Resolve one exact stable idempotency authority in a single snapshot."""

    values = (idempotency_key, domain, queue, job_type, owner_user_id)
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError("scoped idempotency lookup values must be non-empty strings")
    projection = ", ".join(("id", *SLIDES_ARCHIVE_EXACT_FIELDS))
    where_sql = "idempotency_key=? AND domain=? AND queue=? AND job_type=? AND owner_user_id=?"
    rows = conn.execute(
        f"""
        SELECT {projection}, NULL AS payload_compressed,
               NULL AS result_compressed, 0 AS archived
        FROM jobs
        WHERE {where_sql}
        UNION ALL
        SELECT {projection}, payload_compressed,
               result_compressed, 1 AS archived
        FROM jobs_archive
        WHERE {where_sql}
        """,  # nosec B608
        (*values, *values),
    ).fetchall()
    if len(rows) > 1:
        raise IdempotentOperationUnavailableError("scoped idempotency key does not resolve to exactly one Job")
    if not rows:
        return None
    job = normalize_slides_archive_projection(rows[0])
    job["archived"] = bool(job.get("archived"))
    return job


def _resolve_receipt_job(
    conn: sqlite3.Connection,
    receipt: dict[str, Any],
) -> dict[str, Any]:
    try:
        job = get_job_or_archived_by_uuid(
            conn,
            receipt.get("job_uuid"),
            domain=receipt.get("domain"),
            owner_user_id=receipt.get("owner_user_id"),
        )
    except ValueError as exc:
        raise IdempotentOperationUnavailableError(
            "receipt contains an invalid Job UUID"
        ) from exc
    if job is None:
        raise IdempotentOperationUnavailableError(
            "receipt does not resolve to exactly one Job"
        )
    valid = (
        job.get("uuid") == receipt.get("job_uuid")
        and job.get("id") == receipt.get("job_id")
        and job.get("domain") == receipt.get("domain")
        and job.get("queue") == receipt.get("queue")
        and job.get("job_type") == receipt.get("job_type")
        and job.get("owner_user_id") == receipt.get("owner_user_id")
        and job.get("batch_group") == receipt.get("operation_scope")
    )
    if not valid:
        raise IdempotentOperationUnavailableError(
            "receipt and Job correlation do not match"
        )
    return job


def _find_active_scope_job(
    conn: sqlite3.Connection,
    command: IdempotentOperationCommand,
) -> tuple[dict[str, Any], str] | None:
    rows = conn.execute(
        """
        SELECT jobs.*
        FROM jobs
        WHERE jobs.domain = ? AND jobs.queue = ? AND jobs.job_type = ?
          AND jobs.owner_user_id = ? AND jobs.batch_group = ?
          AND jobs.status IN ('queued', 'processing')
        ORDER BY jobs.id
        """,
        (
            command.job.domain,
            command.job.queue,
            command.job.job_type,
            command.job.owner_user_id,
            command.operation_scope,
        ),
    ).fetchall()
    if not rows:
        return None
    if len(rows) != 1:
        raise IdempotentOperationUnavailableError(
            "operation scope resolves to multiple active Jobs"
        )
    job = _row_to_dict(rows[0])
    receipt_rows = conn.execute(
        """
        SELECT *
        FROM job_idempotency_receipts
        WHERE job_uuid = ? OR job_id = ?
        ORDER BY receipt_id
        """,
        (
            job["uuid"],
            job["id"],
        ),
    ).fetchall()
    receipts = [_row_to_dict(row) for row in receipt_rows]
    if any(
        receipt.get("job_uuid") != job.get("uuid")
        or receipt.get("job_id") != job.get("id")
        or receipt.get("domain") != job.get("domain")
        or receipt.get("queue") != job.get("queue")
        or receipt.get("job_type") != job.get("job_type")
        or receipt.get("owner_user_id") != job.get("owner_user_id")
        or receipt.get("operation_scope") != job.get("batch_group")
        for receipt in receipts
    ):
        raise IdempotentOperationUnavailableError(
            "active operation receipt correlation does not match"
        )
    fingerprints = {
        str(receipt["request_fingerprint"]) for receipt in receipts
    }
    if len(fingerprints) != 1:
        raise IdempotentOperationUnavailableError(
            "active operation receipts do not have one fingerprint"
        )
    return job, next(iter(fingerprints))


def _insert_receipt(
    conn: sqlite3.Connection,
    *,
    command: IdempotentOperationCommand,
    job: dict[str, Any],
    now_sql: str,
) -> None:
    conn.execute(
        """
        INSERT INTO job_idempotency_receipts (
          domain, queue, job_type, owner_user_id, key_digest,
          request_fingerprint, operation_scope, job_uuid, job_id,
          created_at, expires_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            command.job.domain,
            command.job.queue,
            command.job.job_type,
            command.job.owner_user_id,
            command.key_digest,
            command.request_fingerprint,
            command.operation_scope,
            job["uuid"],
            job["id"],
            now_sql,
            _sqlite_timestamp(command.receipt_expires_at),
        ),
    )


def replay_idempotent_operation(
    conn: sqlite3.Connection,
    command: IdempotentOperationCommand,
) -> IdempotentOperationAdmission | None:
    """Return a validated exact-key replay without applying admission policy."""

    receipt = _find_exact_receipt(conn, command)
    if receipt is None:
        return None
    job = _resolve_receipt_job(conn, receipt)
    if (
        receipt.get("operation_scope") != command.operation_scope
        or not secrets.compare_digest(
            str(receipt.get("request_fingerprint") or ""),
            command.request_fingerprint,
        )
    ):
        raise IdempotentOperationConflict(
            IdempotentOperationConflictReason.KEY_REUSED,
            job_uuid=str(job["uuid"]),
        )
    return IdempotentOperationAdmission.replayed(job)


def admit_idempotent_operation(
    conn: sqlite3.Connection,
    *,
    command: IdempotentOperationCommand,
    uuid_value: str,
    now: datetime,
    max_queued_quota: int,
    submits_per_minute_quota: int,
    counters_enabled: bool,
) -> IdempotentOperationAdmission:
    """Atomically create, replay, or converge one receipt-backed Job."""

    now_sql = _sqlite_timestamp(now)
    available_at = _future_available_at(command.job.available_at, now=now)
    available_at_sql = _sqlite_timestamp(available_at) if available_at else None
    payload_json = json.dumps(command.job.payload)

    conn.execute("BEGIN IMMEDIATE")
    with conn:
        replay = replay_idempotent_operation(conn, command)
        if replay is not None:
            return replay

        if command.receipt_expires_at < now + timedelta(days=30):
            raise ValueError(
                "receipt_expires_at must retain replay authority for at least 30 days"
            )

        active = _find_active_scope_job(conn, command)
        if active is not None:
            job, active_fingerprint = active
            if not secrets.compare_digest(
                active_fingerprint,
                command.request_fingerprint,
            ):
                raise IdempotentOperationConflict(
                    IdempotentOperationConflictReason.SCOPE_ACTIVE,
                    job_uuid=str(job["uuid"]),
                )
            _insert_receipt(conn, command=command, job=job, now_sql=now_sql)
            return IdempotentOperationAdmission.converged(job)

        quota_result = _quota_rejection(
            conn,
            command=command.job,
            now_sql=now_sql,
            max_queued_quota=max_queued_quota,
            submits_per_minute_quota=submits_per_minute_quota,
        )
        if quota_result is not None:
            raise ValueError(quota_result.message or "Quota exceeded")

        job_id = _insert_job(
            conn,
            command=command.job,
            uuid_value=uuid_value,
            payload_json=payload_json,
            now_sql=now_sql,
            available_at_sql=available_at_sql,
            ignore_idempotency_conflict=False,
        )
        job = _row_to_dict(
            conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        )
        if not job:
            raise IdempotentOperationUnavailableError(
                "newly inserted Job could not be read back"
            )
        if counters_enabled:
            _bump_counters(
                conn,
                command=command.job,
                available_at_sql=available_at_sql,
            )
        _insert_created_event(
            conn,
            row=job,
            idempotent=False,
            request_id=command.job.request_id,
            trace_id=command.job.trace_id,
        )
        _insert_receipt(conn, command=command, job=job, now_sql=now_sql)
        return IdempotentOperationAdmission.created(job)
