"""PostgreSQL admission for durable owner-scoped idempotent operations."""

from __future__ import annotations

import hashlib
import json
import secrets
from collections.abc import Callable
from contextlib import AbstractContextManager
from datetime import datetime, timedelta
from typing import Any

from tldw_Server_API.app.core.Jobs.migrations import (
    SLIDES_ARCHIVE_EXACT_FIELDS,
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
    _bump_counters_best_effort,
    _future_available_at,
    _insert_created_event,
    _insert_job,
    _quota_lock_key,
    _quota_rejection,
    _read_committed_quota_transaction,
)


def _row_to_dict(row: Any) -> dict[str, Any]:
    return dict(row) if row is not None else {}


def _row_value(row: Any, name: str, position: int) -> Any:
    if isinstance(row, dict):
        return row.get(name)
    return row[position]


def _operation_lock_key(kind: str, command: IdempotentOperationCommand) -> int:
    value = command.key_digest if kind == "key" else command.operation_scope
    material = "\x00".join(
        (
            "jobs:idempotent-operation",
            kind,
            command.job.domain,
            command.job.queue,
            command.job.job_type,
            str(command.job.owner_user_id),
            value,
        )
    ).encode("utf-8")
    digest = hashlib.blake2b(material, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=True)


def _operation_lock_keys(command: IdempotentOperationCommand) -> tuple[int, ...]:
    return tuple(
        sorted(
            {
                _operation_lock_key("key", command),
                _operation_lock_key("scope", command),
            }
        )
    )


def _find_exact_receipt(cur: Any, command: IdempotentOperationCommand):
    cur.execute(
        """
        SELECT *
        FROM job_idempotency_receipts
        WHERE domain = %s AND queue = %s AND job_type = %s
          AND owner_user_id = %s AND key_digest = %s
        """,
        (
            command.job.domain,
            command.job.queue,
            command.job.job_type,
            command.job.owner_user_id,
            command.key_digest,
        ),
    )
    return _row_to_dict(cur.fetchone()) or None


def get_job_or_archived_by_uuid(
    cur: Any,
    job_uuid: str,
    *,
    domain: str | None = None,
    owner_user_id: str | None = None,
) -> dict[str, Any] | None:
    """Read one UUID from active/archive storage in a single DB snapshot."""

    if not isinstance(job_uuid, str) or not job_uuid.strip() or len(job_uuid) > 200:
        raise ValueError("job_uuid must be a non-empty string of at most 200 characters")

    clauses = ["uuid = %s"]
    filter_values: list[Any] = [job_uuid]
    if domain is not None:
        clauses.append("domain = %s")
        filter_values.append(domain)
    if owner_user_id is not None:
        clauses.append("owner_user_id = %s")
        filter_values.append(owner_user_id)
    where_sql = " AND ".join(clauses)
    projection = ", ".join(("id", *SLIDES_ARCHIVE_EXACT_FIELDS))
    cur.execute(
        f"""
        SELECT {projection}, NULL AS payload_compressed,
               NULL AS result_compressed, FALSE AS archived
        FROM jobs
        WHERE {where_sql}
        UNION ALL
        SELECT {projection}, payload_compressed,
               result_compressed, TRUE AS archived
        FROM jobs_archive
        WHERE {where_sql}
        """,  # nosec B608
        (*filter_values, *filter_values),
    )
    rows = cur.fetchall()
    if len(rows) > 1:
        raise IdempotentOperationUnavailableError(
            "job UUID does not resolve to exactly one Job"
        )
    if not rows:
        return None
    job = normalize_slides_archive_projection(rows[0])
    job["archived"] = bool(job.get("archived"))
    return job


def _resolve_receipt_job(cur: Any, receipt: dict[str, Any]) -> dict[str, Any]:
    try:
        job = get_job_or_archived_by_uuid(
            cur,
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


def _replay_with_cursor(
    cur: Any,
    command: IdempotentOperationCommand,
) -> IdempotentOperationAdmission | None:
    receipt = _find_exact_receipt(cur, command)
    if receipt is None:
        return None
    job = _resolve_receipt_job(cur, receipt)
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


def replay_idempotent_operation(
    conn: Any,
    cursor_factory: Callable[[Any], AbstractContextManager[Any]],
    command: IdempotentOperationCommand,
) -> IdempotentOperationAdmission | None:
    """Return a validated exact-key replay without applying admission policy."""

    with cursor_factory(conn) as cur:
        return _replay_with_cursor(cur, command)


def _find_active_scope_job(
    cur: Any,
    command: IdempotentOperationCommand,
) -> tuple[dict[str, Any], str] | None:
    cur.execute(
        """
        SELECT jobs.*
        FROM jobs
        WHERE jobs.domain = %s AND jobs.queue = %s AND jobs.job_type = %s
          AND jobs.owner_user_id = %s AND jobs.batch_group = %s
          AND jobs.status IN ('queued', 'processing')
          AND EXISTS (
            SELECT 1 FROM job_idempotency_receipts AS receipts
            WHERE receipts.job_uuid = jobs.uuid AND receipts.job_id = jobs.id
              AND receipts.domain = jobs.domain
              AND receipts.queue = jobs.queue
              AND receipts.job_type = jobs.job_type
              AND receipts.owner_user_id = jobs.owner_user_id
              AND receipts.operation_scope = jobs.batch_group
          )
        ORDER BY jobs.id
        FOR KEY SHARE
        """,
        (
            command.job.domain,
            command.job.queue,
            command.job.job_type,
            command.job.owner_user_id,
            command.operation_scope,
        ),
    )
    rows = cur.fetchall()
    if not rows:
        return None
    if len(rows) != 1:
        raise IdempotentOperationUnavailableError(
            "operation scope resolves to multiple active Jobs"
        )
    job = _row_to_dict(rows[0])
    cur.execute(
        """
        SELECT DISTINCT request_fingerprint
        FROM job_idempotency_receipts
        WHERE job_uuid = %s AND job_id = %s
          AND domain = %s AND queue = %s AND job_type = %s
          AND owner_user_id = %s AND operation_scope = %s
        """,
        (
            job["uuid"],
            job["id"],
            command.job.domain,
            command.job.queue,
            command.job.job_type,
            command.job.owner_user_id,
            command.operation_scope,
        ),
    )
    fingerprints = {
        str(_row_value(row, "request_fingerprint", 0)) for row in cur.fetchall()
    }
    if len(fingerprints) != 1:
        raise IdempotentOperationUnavailableError(
            "active operation receipts do not have one fingerprint"
        )
    return job, next(iter(fingerprints))


def _insert_receipt(
    cur: Any,
    *,
    command: IdempotentOperationCommand,
    job: dict[str, Any],
    now: datetime,
) -> None:
    cur.execute(
        """
        INSERT INTO job_idempotency_receipts (
          domain, queue, job_type, owner_user_id, key_digest,
          request_fingerprint, operation_scope, job_uuid, job_id,
          created_at, expires_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
            now,
            command.receipt_expires_at,
        ),
    )


def admit_idempotent_operation(
    conn: Any,
    cursor_factory: Callable[[Any], AbstractContextManager[Any]],
    *,
    command: IdempotentOperationCommand,
    uuid_value: str,
    now: datetime,
    max_queued_quota: int,
    submits_per_minute_quota: int,
    counters_enabled: bool,
) -> IdempotentOperationAdmission:
    """Atomically create, replay, or converge one receipt-backed Job."""

    payload_json = json.dumps(command.job.payload)
    available_at = _future_available_at(command.job.available_at, now=now)
    quota_enabled = bool(
        command.job.owner_user_id
        and (max_queued_quota or submits_per_minute_quota)
    )

    with _read_committed_quota_transaction(conn, enabled=quota_enabled), conn:
        with cursor_factory(conn) as cur:
            for lock_key in _operation_lock_keys(command):
                cur.execute("SELECT pg_advisory_xact_lock(%s)", (lock_key,))

            replay = _replay_with_cursor(cur, command)
            if replay is not None:
                return replay
            if command.receipt_expires_at < now + timedelta(days=30):
                raise ValueError(
                    "receipt_expires_at must retain replay authority for at least 30 days"
                )

            active = _find_active_scope_job(cur, command)
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
                _insert_receipt(cur, command=command, job=job, now=now)
                return IdempotentOperationAdmission.converged(job)

            if quota_enabled:
                cur.execute(
                    "SELECT pg_advisory_xact_lock(%s)",
                    (_quota_lock_key(command.job),),
                )
            quota_result = _quota_rejection(
                cur,
                command=command.job,
                now=now,
                max_queued_quota=max_queued_quota,
                submits_per_minute_quota=submits_per_minute_quota,
            )
            if quota_result is not None:
                raise ValueError(quota_result.message or "Quota exceeded")

            job = _insert_job(
                cur,
                command=command.job,
                uuid_value=uuid_value,
                payload_json=payload_json,
                available_at=available_at,
                idempotent_insert=False,
            )
            if not job:
                raise IdempotentOperationUnavailableError(
                    "newly inserted Job could not be read back"
                )
            if counters_enabled:
                _bump_counters_best_effort(
                    cur,
                    command=command.job,
                    available_at=available_at,
                )
            _insert_created_event(
                cur,
                row=job,
                idempotent=False,
                request_id=command.job.request_id,
                trace_id=command.job.trace_id,
            )
            _insert_receipt(cur, command=command, job=job, now=now)
            return IdempotentOperationAdmission.created(job)
