"""
Jobs module migrations (SQLite-focused).

Provides a simple helper to ensure the `jobs` table exists in a given SQLite
database path. This scaffolds the future core JobManager backend.
"""

from __future__ import annotations

import base64
import binascii
import contextlib
import json
import math
import os
import sqlite3
import zlib
from pathlib import Path
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.DB_Management.sqlite_policy import (
    configure_sqlite_connection,
)
from tldw_Server_API.app.core.Jobs.operations.contracts import (
    ADMIN_WEBHOOK_DELIVERY_DOMAIN,
    ADMIN_WEBHOOK_DELIVERY_JOB_TYPE,
    ADMIN_WEBHOOK_DELIVERY_QUARANTINE_THRESHOLD,
    ADMIN_WEBHOOK_DELIVERY_QUEUE,
    ExpiredLeasePolicy,
    SlidesArchiveNormalizationError,
    reconstruct_legacy_admin_webhook_archive_fingerprint,
)

_JOBS_PATH_EXCEPTIONS = (ImportError, OSError, RuntimeError, TypeError, ValueError)
_JOBS_DB_EXCEPTIONS = (sqlite3.Error, OSError, RuntimeError, TypeError, ValueError)
_SLIDES_AUDIT_EXCEPTIONS = (
    *_JOBS_DB_EXCEPTIONS,
    SlidesArchiveNormalizationError,
)

SQLITE_ARCHIVE_CURSOR_SENTINEL = "0001-01-01 00:00:00"
_SQLITE_ARCHIVE_ISO_DATE_GLOB = (
    "[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]*"
)
_SQLITE_ARCHIVE_CREATED_AT_ISO_SQL = (
    f"trim(created_at) GLOB '{_SQLITE_ARCHIVE_ISO_DATE_GLOB}' "
    "AND substr(trim(created_at), 1, 4) <> '0000'"
)
_SQLITE_ARCHIVE_ARCHIVED_AT_ISO_SQL = (
    f"trim(archived_at) GLOB '{_SQLITE_ARCHIVE_ISO_DATE_GLOB}' "
    "AND substr(trim(archived_at), 1, 4) <> '0000'"
)
SQLITE_ARCHIVE_CURSOR_TIME_SQL = (
    "COALESCE("
    f"CASE WHEN {_SQLITE_ARCHIVE_CREATED_AT_ISO_SQL} "
    "THEN julianday(trim(created_at)) END, "
    f"CASE WHEN {_SQLITE_ARCHIVE_ARCHIVED_AT_ISO_SQL} "
    "THEN julianday(trim(archived_at)) END, "
    "1721425.5)"
)
SQLITE_ARCHIVE_CURSOR_OUTPUT_SQL = (
    "COALESCE("
    f"CASE WHEN {_SQLITE_ARCHIVE_CREATED_AT_ISO_SQL} "
    "THEN strftime('%Y-%m-%d %H:%M:%f', julianday(trim(created_at))) END, "
    f"CASE WHEN {_SQLITE_ARCHIVE_ARCHIVED_AT_ISO_SQL} "
    "THEN strftime('%Y-%m-%d %H:%M:%f', julianday(trim(archived_at))) END, "
    f"'{SQLITE_ARCHIVE_CURSOR_SENTINEL}')"
)
SQLITE_ARCHIVE_CURSOR_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_jobs_archive_cursor_v2 "
    "ON jobs_archive(domain, job_type, "
    f"{SQLITE_ARCHIVE_CURSOR_TIME_SQL}, "
    "id, COALESCE(uuid, ''), archive_id)"
)
SQLITE_ARCHIVE_LOOKUP_ID_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_jobs_archive_lookup_id "
    "ON jobs_archive(id, archive_id DESC)"
)
SQLITE_ARCHIVE_BATCH_GROUP_SCOPE_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_jobs_archive_batch_group_scope "
    "ON jobs_archive(batch_group, domain, owner_user_id, job_type, "
    "archive_id DESC)"
)
_SQLITE_ARCHIVE_BATCH_READ_INDEX_SPECS = (
    (
        "idx_jobs_archive_lookup_id",
        SQLITE_ARCHIVE_LOOKUP_ID_INDEX_SQL,
        (("id", False), ("archive_id", True)),
    ),
    (
        "idx_jobs_archive_batch_group_scope",
        SQLITE_ARCHIVE_BATCH_GROUP_SCOPE_INDEX_SQL,
        (
            ("batch_group", False),
            ("domain", False),
            ("owner_user_id", False),
            ("job_type", False),
            ("archive_id", True),
        ),
    ),
)


class _SlidesAuditSafetyError(Exception):
    """Raised when standalone readiness cannot be made durably fail-closed."""

SLIDES_ARCHIVE_EXACT_FIELDS = (
    "uuid",
    "domain",
    "queue",
    "job_type",
    "owner_user_id",
    "project_id",
    "batch_group",
    "idempotency_key",
    "payload",
    "result",
    "status",
    "priority",
    "max_retries",
    "expired_lease_policy",
    "quarantine_threshold",
    "prepared_disposition_fingerprint",
    "no_attempt_recovery_fingerprint",
    "retry_count",
    "available_at",
    "started_at",
    "leased_until",
    "lease_id",
    "worker_id",
    "acquired_at",
    "error_message",
    "error_code",
    "last_error",
    "cancel_requested_at",
    "cancelled_at",
    "cancellation_reason",
    "completion_token",
    "failure_streak_code",
    "failure_streak_count",
    "quarantined_at",
    "progress_percent",
    "progress_message",
    "request_id",
    "trace_id",
    "failure_timeline",
    "created_at",
    "updated_at",
    "completed_at",
)

SLIDES_ARCHIVE_COMPRESSED_FIELDS = ("payload_compressed", "result_compressed")
SLIDES_ARCHIVE_PAYLOAD_PRESENT = "__slides_archive_payload_present"
SLIDES_ARCHIVE_RESULT_PRESENT = "__slides_archive_result_present"

# Jobs payload JSON defaults to a 1 MiB admission cap. Archive readback uses the
# same fixed logical limit plus bounded gzip overhead for compressed input.
JOBS_ARCHIVE_JSON_MAX_BYTES = 1_048_576
JOBS_ARCHIVE_COMPRESSED_MAX_BYTES = JOBS_ARCHIVE_JSON_MAX_BYTES + 65_536
_JOBS_ARCHIVE_BASE64_MAX_CHARS = (
    4 * ((JOBS_ARCHIVE_COMPRESSED_MAX_BYTES + 2) // 3)
)
_JOBS_ARCHIVE_GZIP_CHUNK_BYTES = 65_536


def _parse_slides_archive_json(value: Any) -> Any:
    """Normalize a stored JSON value without requiring a Jobs manager instance."""
    if isinstance(value, memoryview):
        value = value.tobytes()
    if isinstance(value, (bytes, bytearray)):
        try:
            value = bytes(value).decode("utf-8")
        except UnicodeDecodeError:
            return bytes(value)
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (TypeError, ValueError):
            return value
    return value


def slides_archive_values_equal(left: Any, right: Any) -> bool:
    """Compare logical archive values with exact recursive JSON semantics."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            slides_archive_values_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            slides_archive_values_equal(left_item, right_item)
            for left_item, right_item in zip(left, right)
        )
    if isinstance(left, float) and math.isnan(left):
        return math.isnan(right)
    return left == right


def _bounded_gzip_decompress(compressed: bytes) -> bytes:
    """Decode one complete gzip member without exceeding archive bounds."""

    if not 1 <= len(compressed) <= JOBS_ARCHIVE_COMPRESSED_MAX_BYTES:
        raise ValueError("archive compressed input is outside the fixed bound")
    decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
    output = bytearray()
    for offset in range(0, len(compressed), _JOBS_ARCHIVE_GZIP_CHUNK_BYTES):
        remaining = JOBS_ARCHIVE_JSON_MAX_BYTES - len(output)
        if remaining <= 0:
            raise ValueError("archive JSON exceeds the fixed bound")
        chunk = compressed[offset : offset + _JOBS_ARCHIVE_GZIP_CHUNK_BYTES]
        decoded = decompressor.decompress(chunk, remaining)
        output.extend(decoded)
        if decompressor.unconsumed_tail:
            raise ValueError("archive JSON exceeds the fixed bound")
        if decompressor.unused_data:
            raise ValueError("archive gzip contains trailing or concatenated data")
    remaining = JOBS_ARCHIVE_JSON_MAX_BYTES - len(output)
    if remaining > 0:
        output.extend(decompressor.flush(remaining))
    if (
        len(output) > JOBS_ARCHIVE_JSON_MAX_BYTES
        or not decompressor.eof
        or decompressor.unused_data
        or decompressor.unconsumed_tail
    ):
        raise ValueError("archive gzip stream is incomplete or exceeds its bound")
    return bytes(output)


def _strict_archive_compressed_bytes(value: Any) -> bytes:
    """Validate one backend archive encoding before allocating decoded bytes."""

    if isinstance(value, memoryview):
        if value.nbytes > JOBS_ARCHIVE_COMPRESSED_MAX_BYTES:
            raise ValueError("archive compressed input is outside the fixed bound")
        return value.tobytes()
    if isinstance(value, (bytes, bytearray)):
        if len(value) > JOBS_ARCHIVE_COMPRESSED_MAX_BYTES:
            raise ValueError("archive compressed input is outside the fixed bound")
        return bytes(value)
    if isinstance(value, str) and value.startswith("gzip64:"):
        encoded = value[len("gzip64:") :]
        if (
            not encoded
            or len(encoded) > _JOBS_ARCHIVE_BASE64_MAX_CHARS
            or len(encoded) % 4 != 0
        ):
            raise ValueError("archive base64 input is outside the fixed bound")
        encoded_bytes = encoded.encode("ascii")
        compressed = base64.b64decode(encoded_bytes, validate=True)
        if len(compressed) > JOBS_ARCHIVE_COMPRESSED_MAX_BYTES:
            raise ValueError("archive compressed input is outside the fixed bound")
        if base64.b64encode(compressed) != encoded_bytes:
            raise ValueError("archive base64 input is not canonically encoded")
        return compressed
    raise ValueError("archive compressed input uses an unsupported encoding")


def _decode_slides_archive_blob(value: Any) -> Any:
    """Decode one bounded, strictly framed SQLite/PostgreSQL archive blob."""
    if value is None:
        return None
    try:
        compressed = _strict_archive_compressed_bytes(value)
        decoded = _bounded_gzip_decompress(compressed).decode("utf-8")
        return json.loads(decoded)
    except (binascii.Error, TypeError, ValueError, UnicodeError, zlib.error):
        pass
    raise SlidesArchiveNormalizationError


def normalize_slides_archive_projection(row: Any) -> dict[str, Any]:
    """Return one logical projection or reject an invalid compressed field."""
    normalized = dict(row)
    presence_fields = {
        "payload": SLIDES_ARCHIVE_PAYLOAD_PRESENT,
        "result": SLIDES_ARCHIVE_RESULT_PRESENT,
    }
    for field, presence_field in presence_fields.items():
        raw_primary = normalized.get(field)
        presence = normalized.pop(presence_field, None)
        if presence is None:
            primary_present = raw_primary is not None
        elif type(presence) is bool:
            primary_present = presence
        elif type(presence) is int and presence in (0, 1):
            primary_present = bool(presence)
        else:
            raise SlidesArchiveNormalizationError
        if not primary_present and raw_primary is not None:
            raise SlidesArchiveNormalizationError
        primary = _parse_slides_archive_json(raw_primary)
        compressed = normalized.get(f"{field}_compressed")
        if compressed is not None:
            sidecar = _decode_slides_archive_blob(compressed)
            if primary_present and not slides_archive_values_equal(
                primary,
                sidecar,
            ):
                raise SlidesArchiveNormalizationError
            if not primary_present:
                primary = sidecar
        normalized[field] = primary
    return normalized


def slides_archive_projection_ready_sqlite(conn: sqlite3.Connection) -> bool:
    """Return whether active/archive tables expose the complete exact projection."""
    try:
        active_columns = {row[1] for row in conn.execute("PRAGMA table_info(jobs)")}
        archive_columns = {row[1] for row in conn.execute("PRAGMA table_info(jobs_archive)")}
    except sqlite3.Error:
        return False
    return (
        {"id", *SLIDES_ARCHIVE_EXACT_FIELDS} <= active_columns
        and {
            "id",
            "archived_at",
            *SLIDES_ARCHIVE_EXACT_FIELDS,
            *SLIDES_ARCHIVE_COMPRESSED_FIELDS,
        }
        <= archive_columns
    )

_SLIDES_ARCHIVE_INDEXES = {
    "idx_jobs_archive_slides_scope": (
        False,
        (
            ("domain", False),
            ("queue", False),
            ("job_type", False),
            ("idempotency_key", False),
            ("owner_user_id", False),
            ("archived_at", True),
        ),
        "idempotency_key is not null",
    ),
    "idx_jobs_archive_uuid_unique": (
        True,
        (("uuid", False),),
        "uuid is not null",
    ),
}

JOBS_SQLITE_DDL = """
CREATE TABLE IF NOT EXISTS jobs (
  id INTEGER PRIMARY KEY,
  uuid TEXT UNIQUE,
  domain TEXT NOT NULL,
  queue TEXT NOT NULL,
  job_type TEXT NOT NULL,
  owner_user_id TEXT,
  project_id INTEGER,
  batch_group TEXT,
  idempotency_key TEXT,
  payload TEXT,
  result TEXT,
  -- status now includes 'quarantined' for poison message handling
  status TEXT NOT NULL CHECK (status IN ('queued','processing','completed','failed','cancelled','quarantined')),
  priority INTEGER DEFAULT 5 CHECK (priority >= 1 AND priority <= 10),
  max_retries INTEGER DEFAULT 3 CHECK (max_retries >= 0 AND max_retries <= 100),
  expired_lease_policy TEXT NOT NULL DEFAULT 'consume_retry' CHECK (expired_lease_policy IN ('consume_retry','requeue_no_attempt')),
  quarantine_threshold INTEGER CHECK (quarantine_threshold IS NULL OR quarantine_threshold > 0),
  prepared_disposition_fingerprint TEXT CHECK (
    prepared_disposition_fingerprint IS NULL OR (
      LENGTH(prepared_disposition_fingerprint) = 64 AND
      prepared_disposition_fingerprint NOT GLOB '*[^0-9a-f]*'
    )
  ),
  no_attempt_recovery_fingerprint TEXT CHECK (
    no_attempt_recovery_fingerprint IS NULL OR (
      LENGTH(no_attempt_recovery_fingerprint) = 64 AND
      no_attempt_recovery_fingerprint NOT GLOB '*[^0-9a-f]*'
    )
  ),
  retry_count INTEGER DEFAULT 0,
  available_at TEXT,
  started_at TEXT,
  leased_until TEXT,
  lease_id TEXT,
  worker_id TEXT,
  acquired_at TEXT,
  error_message TEXT,
  error_code TEXT,
  error_class TEXT,
  error_stack TEXT,
  last_error TEXT,
  cancel_requested_at TEXT,
  cancelled_at TEXT,
  cancellation_reason TEXT,
  -- completion token for exactly-once finalize semantics
  completion_token TEXT,
  -- failure streak tracking for poison message quarantine
  failure_streak_code TEXT,
  failure_streak_count INTEGER DEFAULT 0,
  quarantined_at TEXT,
  progress_percent REAL CHECK (progress_percent IS NULL OR (progress_percent >= 0 AND progress_percent <= 100)),
  progress_message TEXT,
  -- correlation
  request_id TEXT,
  trace_id TEXT,
  -- structured failure history (JSON array of {ts, error_code, retry_backoff})
  failure_timeline TEXT,
  created_at TEXT DEFAULT (DATETIME('now')),
  updated_at TEXT DEFAULT (DATETIME('now')),
  completed_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_jobs_lookup ON jobs(domain, queue, status, available_at, priority, created_at);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_lease ON jobs(leased_until);
CREATE INDEX IF NOT EXISTS idx_jobs_owner_status ON jobs(owner_user_id, status, created_at);
CREATE INDEX IF NOT EXISTS idx_jobs_batch_group ON jobs(batch_group);
-- Cover ready vs scheduled scans
CREATE INDEX IF NOT EXISTS idx_jobs_status_available_at ON jobs(status, available_at);

-- Emulate Postgres partial unique index: scope idempotency to (domain,queue,job_type) when key is not NULL
CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_idempotent
  ON jobs(domain, queue, job_type, idempotency_key)
  WHERE idempotency_key IS NOT NULL;

-- Keep updated_at current
CREATE TRIGGER IF NOT EXISTS trg_jobs_updated_at
AFTER UPDATE ON jobs
FOR EACH ROW
BEGIN
  UPDATE jobs SET updated_at = DATETIME('now') WHERE id = NEW.id;
END;

-- Optional archive table (schema-aligned, used when JOBS_ARCHIVE_BEFORE_DELETE=true)
CREATE TABLE IF NOT EXISTS jobs_archive (
  archive_id INTEGER PRIMARY KEY AUTOINCREMENT,
  id INTEGER,
  uuid TEXT,
  domain TEXT NOT NULL,
  queue TEXT NOT NULL,
  job_type TEXT NOT NULL,
  owner_user_id TEXT,
  project_id INTEGER,
  batch_group TEXT,
  idempotency_key TEXT,
  payload TEXT,
  result TEXT,
  status TEXT NOT NULL,
  priority INTEGER,
  max_retries INTEGER,
  expired_lease_policy TEXT NOT NULL DEFAULT 'consume_retry' CHECK (expired_lease_policy IN ('consume_retry','requeue_no_attempt')),
  quarantine_threshold INTEGER CHECK (quarantine_threshold IS NULL OR quarantine_threshold > 0),
  prepared_disposition_fingerprint TEXT CHECK (
    prepared_disposition_fingerprint IS NULL OR (
      LENGTH(prepared_disposition_fingerprint) = 64 AND
      prepared_disposition_fingerprint NOT GLOB '*[^0-9a-f]*'
    )
  ),
  no_attempt_recovery_fingerprint TEXT CHECK (
    no_attempt_recovery_fingerprint IS NULL OR (
      LENGTH(no_attempt_recovery_fingerprint) = 64 AND
      no_attempt_recovery_fingerprint NOT GLOB '*[^0-9a-f]*'
    )
  ),
  retry_count INTEGER,
  available_at TEXT,
  started_at TEXT,
  leased_until TEXT,
  lease_id TEXT,
  worker_id TEXT,
  acquired_at TEXT,
  error_message TEXT,
  error_code TEXT,
  last_error TEXT,
  cancel_requested_at TEXT,
  cancelled_at TEXT,
  cancellation_reason TEXT,
  completion_token TEXT,
  failure_streak_code TEXT,
  failure_streak_count INTEGER,
  quarantined_at TEXT,
  progress_percent REAL,
  progress_message TEXT,
  request_id TEXT,
  trace_id TEXT,
  failure_timeline TEXT,
  -- Optional compressed blobs (base64-gz) for payload/result when archiving
  payload_compressed TEXT,
  result_compressed TEXT,
  created_at TEXT,
  updated_at TEXT,
  completed_at TEXT,
  archived_at TEXT DEFAULT (DATETIME('now'))
);
CREATE INDEX IF NOT EXISTS idx_jobs_archive_migration
  ON jobs_archive(
    domain,
    job_type,
    status,
    COALESCE(created_at, archived_at, '0001-01-01 00:00:00'),
    id,
    COALESCE(uuid, '')
  );

-- Immutable correlation for user-facing idempotent operations. Job state,
-- progress, result, and errors remain exclusively in jobs/jobs_archive.
CREATE TABLE IF NOT EXISTS job_idempotency_receipts (
  receipt_id INTEGER PRIMARY KEY AUTOINCREMENT,
  domain TEXT NOT NULL CHECK (LENGTH(domain) BETWEEN 1 AND 64),
  queue TEXT NOT NULL CHECK (LENGTH(queue) BETWEEN 1 AND 64),
  job_type TEXT NOT NULL CHECK (LENGTH(job_type) BETWEEN 1 AND 128),
  owner_user_id TEXT NOT NULL CHECK (LENGTH(owner_user_id) BETWEEN 1 AND 200),
  key_digest TEXT NOT NULL CHECK (
    LENGTH(key_digest) = 64 AND key_digest NOT GLOB '*[^0-9a-f]*'
  ),
  request_fingerprint TEXT NOT NULL CHECK (
    LENGTH(request_fingerprint) = 64
    AND request_fingerprint NOT GLOB '*[^0-9a-f]*'
  ),
  operation_scope TEXT NOT NULL CHECK (LENGTH(operation_scope) BETWEEN 1 AND 200),
  job_uuid TEXT NOT NULL CHECK (LENGTH(job_uuid) BETWEEN 1 AND 64),
  job_id INTEGER NOT NULL,
  created_at TEXT NOT NULL,
  expires_at TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_job_idempotency_receipts_owner_key
  ON job_idempotency_receipts(
    domain, queue, job_type, owner_user_id, key_digest
  );
CREATE INDEX IF NOT EXISTS idx_job_idempotency_receipts_job_uuid
  ON job_idempotency_receipts(job_uuid);
CREATE INDEX IF NOT EXISTS idx_job_idempotency_receipts_job_id
  ON job_idempotency_receipts(job_id);
CREATE INDEX IF NOT EXISTS idx_job_idempotency_receipts_scope
  ON job_idempotency_receipts(operation_scope, owner_user_id, expires_at);

-- Source-free standalone-HTML digest-key metadata. Secrets and digests never
-- belong in the shared Jobs store.
CREATE TABLE IF NOT EXISTS slides_standalone_key_registry (
  key_id TEXT PRIMARY KEY,
  state TEXT NOT NULL CHECK (state IN ('current','retiring')),
  activated_at TEXT NOT NULL,
  retired_at TEXT,
  config_revision TEXT NOT NULL,
  CHECK (
    (state = 'current' AND retired_at IS NULL)
    OR (state = 'retiring' AND retired_at IS NOT NULL)
  )
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_slides_standalone_one_current_key
  ON slides_standalone_key_registry(state)
  WHERE state = 'current';

-- Singleton fencing/reconciliation state. Diagnostic state is deliberately
-- bounded to a code, count, and timestamp and contains no source material.
CREATE TABLE IF NOT EXISTS slides_standalone_reconciliation (
  singleton_id INTEGER PRIMARY KEY CHECK (singleton_id = 1),
  holder_uuid TEXT,
  lease_expires_at TEXT,
  fencing_token INTEGER NOT NULL DEFAULT 0 CHECK (fencing_token >= 0),
  cursor TEXT,
  config_revision TEXT,
  startup_complete_epoch TEXT,
  last_complete_epoch REAL,
  lag INTEGER NOT NULL DEFAULT 0 CHECK (lag >= 0),
  diagnostic_code TEXT CHECK (
    diagnostic_code IS NULL
    OR diagnostic_code IN ('duplicate_archive_uuid','ambiguous_generation_legacy_row')
  ),
  diagnostic_count INTEGER NOT NULL DEFAULT 0 CHECK (diagnostic_count >= 0),
  diagnostic_at TEXT,
  sweep_key_id TEXT,
  sweep_started_at TEXT,
  sweep_completed_at TEXT,
  sweep_complete INTEGER NOT NULL DEFAULT 0 CHECK (sweep_complete IN (0,1)),
  unexpired_reference_count INTEGER NOT NULL DEFAULT 0
    CHECK (unexpired_reference_count >= 0)
);
INSERT OR IGNORE INTO slides_standalone_reconciliation(singleton_id) VALUES (1);

-- Append-only outbox for job events (CDC/event bus)
CREATE TABLE IF NOT EXISTS job_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  job_id INTEGER,
  domain TEXT,
  queue TEXT,
  job_type TEXT,
  event_type TEXT NOT NULL,
  attrs_json TEXT,
  owner_user_id TEXT,
  request_id TEXT,
  trace_id TEXT,
  created_at TEXT NOT NULL DEFAULT (DATETIME('now'))
);
CREATE INDEX IF NOT EXISTS idx_job_events_id ON job_events(id);
CREATE INDEX IF NOT EXISTS idx_job_events_job_id ON job_events(job_id);
\n-- Lightweight per-queue counters to avoid frequent COUNT(*) scans
CREATE TABLE IF NOT EXISTS job_counters (
  domain TEXT NOT NULL,
  queue TEXT NOT NULL,
  job_type TEXT NOT NULL,
  ready_count INTEGER DEFAULT 0,
  scheduled_count INTEGER DEFAULT 0,
  processing_count INTEGER DEFAULT 0,
  quarantined_count INTEGER DEFAULT 0,
  updated_at TEXT DEFAULT (DATETIME('now')),
  PRIMARY KEY (domain, queue, job_type)
);
CREATE INDEX IF NOT EXISTS idx_job_counters_domain_queue ON job_counters(domain, queue);

-- Queue-level controls (pause/drain) per domain/queue
CREATE TABLE IF NOT EXISTS job_queue_controls (
  domain TEXT NOT NULL,
  queue TEXT NOT NULL,
  paused INTEGER DEFAULT 0,
  drain INTEGER DEFAULT 0,
  updated_at TEXT DEFAULT (DATETIME('now')),
  PRIMARY KEY (domain, queue)
);

-- Per-job attachments/logs (small text or URL)
CREATE TABLE IF NOT EXISTS job_attachments (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  job_id INTEGER NOT NULL,
  kind TEXT NOT NULL, -- log|artifact|tag
  content_text TEXT,
  url TEXT,
  created_at TEXT NOT NULL DEFAULT (DATETIME('now'))
);
CREATE INDEX IF NOT EXISTS idx_job_attachments_job ON job_attachments(job_id);

-- SLA policies per job_type
CREATE TABLE IF NOT EXISTS job_sla_policies (
  domain TEXT NOT NULL,
  queue TEXT NOT NULL,
  job_type TEXT NOT NULL,
  max_queue_latency_seconds INTEGER,
  max_duration_seconds INTEGER,
  enabled INTEGER DEFAULT 1,
  updated_at TEXT DEFAULT (DATETIME('now')),
  PRIMARY KEY (domain, queue, job_type)
);

-- Job dependencies (DAG edges)
CREATE TABLE IF NOT EXISTS job_dependencies (
  job_uuid TEXT NOT NULL,
  depends_on_job_uuid TEXT NOT NULL,
  depends_on_terminal_status TEXT,
  depends_on_cancellation_reason TEXT,
  created_at TEXT DEFAULT (DATETIME('now')),
  PRIMARY KEY (job_uuid, depends_on_job_uuid)
);
CREATE INDEX IF NOT EXISTS idx_job_dependencies_job ON job_dependencies(job_uuid);
CREATE INDEX IF NOT EXISTS idx_job_dependencies_depends_on ON job_dependencies(depends_on_job_uuid);
"""


def _sqlite_archive_locator_schema_ready(conn: sqlite3.Connection) -> bool:
    """Return whether a SQLite archive has a stable locator allocator."""
    archive_columns = {
        str(row[1]): str(row[2] or "")
        for row in conn.execute("PRAGMA table_info(jobs_archive)").fetchall()
    }
    if "archive_id" not in archive_columns:
        return False
    if "INT" not in archive_columns["archive_id"].upper():
        return False
    if conn.execute(
        "SELECT 1 FROM jobs_archive WHERE archive_id IS NULL LIMIT 1"
    ).fetchone() is not None:
        return False

    archive_index = next(
        (
            row
            for row in conn.execute("PRAGMA index_list(jobs_archive)").fetchall()
            if str(row[1]) == "idx_jobs_archive_id"
        ),
        None,
    )
    if not (
        archive_index is not None
        and bool(archive_index[2])
        and len(archive_index) > 4
        and not bool(archive_index[4])
    ):
        return False
    archive_index_columns = [
        str(row[2])
        for row in conn.execute(
            "PRAGMA index_info(idx_jobs_archive_id)"
        ).fetchall()
    ]
    if archive_index_columns != ["archive_id"]:
        return False

    trigger_row = conn.execute(
        "SELECT sql FROM sqlite_master "
        "WHERE type = 'trigger' AND name = 'trg_jobs_archive_id' "
        "AND tbl_name = 'jobs_archive'"
    ).fetchone()
    trigger_sql = " ".join(
        str(trigger_row[0] or "").split()
    ) if trigger_row is not None else ""
    return all(
        fragment in trigger_sql
        for fragment in (
            "WHEN NEW.archive_id IS NULL",
            "MAX(archive_id)",
            "+ 1",
            "WHERE rowid = NEW.rowid",
        )
    )


def _sqlite_archive_migration_busy_timeout_ms() -> int:
    """Return the bounded lock wait used by archive schema migrations."""
    try:
        return max(
            1_000,
            int(
                os.getenv(
                    "JOBS_SQLITE_ARCHIVE_MIGRATION_BUSY_TIMEOUT_MS",
                    "60000",
                )
            ),
        )
    except (TypeError, ValueError):
        return 60_000


def _sqlite_archive_batch_read_index_state(
    conn: sqlite3.Connection,
    index_name: str,
) -> tuple[bool, bool, tuple[tuple[str, bool], ...]] | None:
    """Return uniqueness, partial, and ordered key metadata for an index."""

    object_row = conn.execute(
        "SELECT tbl_name FROM sqlite_master "
        "WHERE type = 'index' AND name = ?",
        (index_name,),
    ).fetchone()
    if object_row is None:
        return None
    if str(object_row[0]) != "jobs_archive":
        raise RuntimeError(f"{index_name} belongs to another table")

    index_row = next(
        (
            row
            for row in conn.execute(
                "PRAGMA index_list(jobs_archive)"
            ).fetchall()
            if str(row[1]) == index_name
        ),
        None,
    )
    if index_row is None:
        return None
    key_columns = tuple(
        (str(row[2]), bool(row[3]))
        for row in conn.execute(
            f"PRAGMA index_xinfo({index_name})"  # nosec B608
        ).fetchall()
        if bool(row[5])
    )
    return bool(index_row[2]), bool(index_row[4]), key_columns


def _sqlite_archive_batch_read_index_ready(
    state: tuple[bool, bool, tuple[tuple[str, bool], ...]] | None,
    expected_columns: tuple[tuple[str, bool], ...],
) -> bool:
    """Return whether one archive lookup index has the canonical shape."""

    return bool(
        state is not None
        and not state[0]
        and not state[1]
        and state[2] == expected_columns
    )


def _ensure_sqlite_archive_batch_read_indexes(
    conn: sqlite3.Connection,
) -> None:
    """Atomically create, repair, and verify archive batch-read indexes."""

    conn.commit()
    conn.execute(
        f"PRAGMA busy_timeout = {_sqlite_archive_migration_busy_timeout_ms()}"
    )
    conn.execute("BEGIN IMMEDIATE")
    try:
        for index_name, create_sql, expected_columns in (
            _SQLITE_ARCHIVE_BATCH_READ_INDEX_SPECS
        ):
            state = _sqlite_archive_batch_read_index_state(conn, index_name)
            if not _sqlite_archive_batch_read_index_ready(
                state, expected_columns
            ):
                if state is not None:
                    conn.execute(f"DROP INDEX {index_name}")  # nosec B608
                conn.execute(create_sql)
            if not _sqlite_archive_batch_read_index_ready(
                _sqlite_archive_batch_read_index_state(conn, index_name),
                expected_columns,
            ):
                raise RuntimeError(f"{index_name} verification failed")
        conn.commit()
    except _JOBS_DB_EXCEPTIONS:
        conn.rollback()
        raise


def _ensure_sqlite_archive_locators(conn: sqlite3.Connection) -> None:
    """Atomically add and validate stable locators for legacy archives."""

    if _sqlite_archive_locator_schema_ready(conn):
        return
    conn.commit()
    conn.execute(
        f"PRAGMA busy_timeout = {_sqlite_archive_migration_busy_timeout_ms()}"
    )
    conn.execute("BEGIN IMMEDIATE")
    try:
        if _sqlite_archive_locator_schema_ready(conn):
            conn.commit()
            return
        archive_columns = {
            str(row[1])
            for row in conn.execute("PRAGMA table_info(jobs_archive)").fetchall()
        }
        if "archive_id" not in archive_columns:
            conn.execute("ALTER TABLE jobs_archive ADD COLUMN archive_id INTEGER")
        archive_column = next(
            row
            for row in conn.execute("PRAGMA table_info(jobs_archive)").fetchall()
            if str(row[1]) == "archive_id"
        )
        if "INT" not in str(archive_column[2] or "").upper():
            raise RuntimeError(
                "jobs_archive.archive_id must have INTEGER affinity"
            )
        if conn.execute(
            "SELECT 1 FROM jobs_archive WHERE archive_id IS NOT NULL "
            "AND typeof(archive_id) <> 'integer' LIMIT 1"
        ).fetchone() is not None:
            raise RuntimeError(
                "jobs_archive.archive_id contains non-integer values"
            )
        for object_type, object_name in (
            ("index", "idx_jobs_archive_id"),
            ("trigger", "trg_jobs_archive_id"),
        ):
            object_row = conn.execute(
                "SELECT tbl_name FROM sqlite_master "
                "WHERE type = ? AND name = ?",
                (object_type, object_name),
            ).fetchone()
            if object_row is not None and str(object_row[0]) != "jobs_archive":
                raise RuntimeError(
                    f"{object_name} belongs to another table"
                )

        conn.execute(
            "WITH base(max_id) AS MATERIALIZED ("
            "SELECT COALESCE(MAX(archive_id), 0) FROM jobs_archive), "
            "missing(target_rowid, locator_offset) AS MATERIALIZED ("
            "SELECT rowid, ROW_NUMBER() OVER (ORDER BY rowid) "
            "FROM jobs_archive WHERE archive_id IS NULL) "
            "UPDATE jobs_archive SET archive_id = ("
            "SELECT base.max_id + missing.locator_offset "
            "FROM base, missing WHERE missing.target_rowid = jobs_archive.rowid"
            ") WHERE archive_id IS NULL"
        )

        archive_index = next(
            (
                row
                for row in conn.execute(
                    "PRAGMA index_list(jobs_archive)"
                ).fetchall()
                if str(row[1]) == "idx_jobs_archive_id"
            ),
            None,
        )
        archive_index_columns = (
            [
                str(row[2])
                for row in conn.execute(
                    "PRAGMA index_info(idx_jobs_archive_id)"
                ).fetchall()
            ]
            if archive_index is not None
            else []
        )
        if not (
            archive_index is not None
            and bool(archive_index[2])
            and len(archive_index) > 4
            and not bool(archive_index[4])
            and archive_index_columns == ["archive_id"]
        ):
            conn.execute("DROP INDEX IF EXISTS idx_jobs_archive_id")
            conn.execute(
                "CREATE UNIQUE INDEX idx_jobs_archive_id "
                "ON jobs_archive(archive_id)"
            )

        conn.execute("DROP TRIGGER IF EXISTS trg_jobs_archive_id")
        conn.execute(
            "CREATE TRIGGER trg_jobs_archive_id "
            "AFTER INSERT ON jobs_archive FOR EACH ROW "
            "WHEN NEW.archive_id IS NULL BEGIN "
            "UPDATE jobs_archive SET archive_id = "
            "COALESCE((SELECT MAX(archive_id) FROM jobs_archive), 0) + 1 "
            "WHERE rowid = NEW.rowid; END"
        )
        if not _sqlite_archive_locator_schema_ready(conn):
            raise RuntimeError("SQLite Jobs archive locator migration failed")
        conn.commit()
    except _JOBS_DB_EXCEPTIONS:
        conn.rollback()
        raise


def _ensure_sqlite_dependency_snapshot_columns(conn: sqlite3.Connection) -> None:
    """Add and verify dependency snapshots required by acquisition queries."""

    additions = {
        "depends_on_terminal_status": (
            "ALTER TABLE job_dependencies "
            "ADD COLUMN depends_on_terminal_status TEXT"
        ),
        "depends_on_cancellation_reason": (
            "ALTER TABLE job_dependencies "
            "ADD COLUMN depends_on_cancellation_reason TEXT"
        ),
    }
    try:
        columns = {
            str(row[1])
            for row in conn.execute(
                "PRAGMA table_info(job_dependencies)"
            ).fetchall()
        }
        for column, statement in additions.items():
            if column not in columns:
                conn.execute(statement)
        conn.execute(
            "SELECT depends_on_terminal_status, "
            "depends_on_cancellation_reason FROM job_dependencies LIMIT 0"
        )
    except _JOBS_DB_EXCEPTIONS as exc:
        conn.rollback()
        raise RuntimeError(
            "SQLite Jobs dependency snapshot migration failed"
        ) from exc


def _sqlite_archive_index_matches(
    conn: sqlite3.Connection,
    *,
    index_name: str,
    unique: bool,
    columns: tuple[tuple[str, bool], ...],
    predicate: str,
) -> bool:
    row = conn.execute(
        "SELECT tbl_name, sql FROM sqlite_master WHERE type='index' AND name=?",
        (index_name,),
    ).fetchone()
    if row is None or row[0] != "jobs_archive" or not row[1]:
        return False
    listed = next(
        (item for item in conn.execute("PRAGMA index_list('jobs_archive')") if item[1] == index_name),
        None,
    )
    if listed is None or bool(listed[2]) is not unique or not bool(listed[4]):
        return False
    key_rows = tuple(
        item
        for item in conn.execute(
            "SELECT * FROM pragma_index_xinfo(?)",
            (index_name,),
        )
        if bool(item[5])
    )
    key_columns = tuple((item[2], bool(item[3])) for item in key_rows)
    key_collations = tuple(str(item[4] or "").upper() for item in key_rows)
    normalized_sql = " ".join(str(row[1]).lower().split())
    actual_predicate = normalized_sql.rsplit(" where ", 1)[1] if " where " in normalized_sql else ""
    return (
        key_columns == columns
        and all(collation == "BINARY" for collation in key_collations)
        and actual_predicate == predicate
    )


def slides_archive_indexes_ready_sqlite(conn: sqlite3.Connection) -> bool:
    """Return whether both standalone archive indexes have their exact definitions."""
    return all(
        _sqlite_archive_index_matches(
            conn,
            index_name=index_name,
            unique=unique,
            columns=columns,
            predicate=predicate,
        )
        for index_name, (unique, columns, predicate) in _SLIDES_ARCHIVE_INDEXES.items()
    )


def _audit_and_index_slides_generation(conn: sqlite3.Connection) -> None:
    """Audit legacy generation correlations before adding archive indexes."""
    if not slides_archive_projection_ready_sqlite(conn):
        conn.execute(
            """
            UPDATE slides_standalone_reconciliation
            SET diagnostic_code=CASE
                  WHEN diagnostic_code='duplicate_archive_uuid'
                  THEN diagnostic_code ELSE 'ambiguous_generation_legacy_row' END,
                diagnostic_count=CASE
                  WHEN diagnostic_code='duplicate_archive_uuid'
                  THEN diagnostic_count ELSE MAX(diagnostic_count, 1) END,
                diagnostic_at=CASE
                  WHEN diagnostic_code='duplicate_archive_uuid'
                  THEN diagnostic_at ELSE DATETIME('now') END
            WHERE singleton_id=1
            """
        )
        return

    duplicate_row = conn.execute(
        """
        SELECT COALESCE(SUM(candidate_count), 0)
        FROM (
          SELECT COUNT(*) AS candidate_count
          FROM jobs_archive
          WHERE uuid IS NOT NULL
          GROUP BY uuid
          HAVING COUNT(*) > 1
        )
        """
    ).fetchone()
    duplicate_count = int(duplicate_row[0] or 0) if duplicate_row else 0
    invalid_row = conn.execute(
        """
        SELECT COUNT(*) FROM (
          SELECT uuid, owner_user_id, idempotency_key
          FROM jobs
          WHERE domain='slides' AND queue='default' AND job_type='presentation.generate'
          UNION ALL
          SELECT uuid, owner_user_id, idempotency_key
          FROM jobs_archive
          WHERE domain='slides' AND queue='default' AND job_type='presentation.generate'
        ) scoped
        WHERE uuid IS NULL OR TRIM(uuid) = ''
           OR owner_user_id IS NULL OR TRIM(owner_user_id) = ''
           OR idempotency_key IS NULL OR TRIM(idempotency_key) = ''
        """
    ).fetchone()
    invalid_count = int(invalid_row[0] or 0) if invalid_row else 0
    active_projection = ", ".join(f"active.{field}" for field in SLIDES_ARCHIVE_EXACT_FIELDS)
    archived_projection = ", ".join(f"archived.{field}" for field in SLIDES_ARCHIVE_EXACT_FIELDS)
    cross_table_rows = conn.execute(
        f"""
        SELECT {active_projection}, {archived_projection},
               archived.payload_compressed, archived.result_compressed
        FROM jobs active
        JOIN jobs_archive archived ON archived.uuid = active.uuid
        WHERE active.uuid IS NOT NULL AND TRIM(active.uuid) <> ''
          AND (
            (active.domain='slides' AND active.queue='default'
             AND active.job_type='presentation.generate')
            OR
            (archived.domain='slides' AND archived.queue='default'
             AND archived.job_type='presentation.generate')
          )
        """  # nosec B608 - field names come from a closed module constant
    ).fetchall()
    projection_size = len(SLIDES_ARCHIVE_EXACT_FIELDS)
    cross_table_count = 0
    for row in cross_table_rows:
        active = normalize_slides_archive_projection(
            dict(zip(SLIDES_ARCHIVE_EXACT_FIELDS, row[:projection_size]))
        )
        archived_values = dict(
            zip(
                SLIDES_ARCHIVE_EXACT_FIELDS,
                row[projection_size : 2 * projection_size],
            )
        )
        archived_values["payload_compressed"] = row[2 * projection_size]
        archived_values["result_compressed"] = row[2 * projection_size + 1]
        archived = normalize_slides_archive_projection(archived_values)
        if any(
            not slides_archive_values_equal(
                active.get(field),
                archived.get(field),
            )
            for field in SLIDES_ARCHIVE_EXACT_FIELDS
        ):
            cross_table_count += 1

    diagnostic_code: str | None = None
    diagnostic_count = 0
    if duplicate_count:
        diagnostic_code = "duplicate_archive_uuid"
        diagnostic_count = duplicate_count
    elif invalid_count or cross_table_count:
        diagnostic_code = "ambiguous_generation_legacy_row"
        diagnostic_count = invalid_count + cross_table_count
    for index_name, (unique, columns, predicate) in _SLIDES_ARCHIVE_INDEXES.items():
        if not _sqlite_archive_index_matches(
            conn,
            index_name=index_name,
            unique=unique,
            columns=columns,
            predicate=predicate,
        ):
            if index_name == "idx_jobs_archive_slides_scope":
                conn.execute("DROP INDEX IF EXISTS idx_jobs_archive_slides_scope")
            else:
                conn.execute("DROP INDEX IF EXISTS idx_jobs_archive_uuid_unique")

    conn.execute(
        """
        UPDATE slides_standalone_reconciliation
        SET diagnostic_code=?, diagnostic_count=?,
            diagnostic_at=CASE WHEN ? IS NULL THEN NULL ELSE DATETIME('now') END
        WHERE singleton_id=1
        """,
        (diagnostic_code, diagnostic_count, diagnostic_code),
    )

    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_jobs_archive_slides_scope
        ON jobs_archive(
          domain, queue, job_type, idempotency_key, owner_user_id, archived_at DESC
        )
        WHERE idempotency_key IS NOT NULL
        """
    )
    conn.execute("DROP TRIGGER IF EXISTS trg_jobs_slides_generation_uuid_immutable")
    conn.execute(
        """
        CREATE TRIGGER trg_jobs_slides_generation_uuid_immutable
        BEFORE UPDATE ON jobs
        FOR EACH ROW
        WHEN (
          NEW.domain='slides' AND NEW.queue='default'
            AND NEW.job_type='presentation.generate'
            AND (NEW.uuid IS NULL OR TRIM(NEW.uuid)='')
        ) OR (
          OLD.domain='slides' AND OLD.queue='default'
            AND OLD.job_type='presentation.generate'
            AND (
              NEW.uuid IS NOT OLD.uuid
              OR NEW.domain IS NOT OLD.domain
              OR NEW.queue IS NOT OLD.queue
              OR NEW.job_type IS NOT OLD.job_type
            )
        )
        BEGIN
          SELECT RAISE(ABORT, 'presentation.generate job UUID is immutable');
        END
        """
    )
    conn.execute("DROP TRIGGER IF EXISTS trg_jobs_archive_slides_generation_uuid_immutable")
    conn.execute(
        """
        CREATE TRIGGER trg_jobs_archive_slides_generation_uuid_immutable
        BEFORE UPDATE ON jobs_archive
        FOR EACH ROW
        WHEN (
          NEW.domain='slides' AND NEW.queue='default'
            AND NEW.job_type='presentation.generate'
            AND (NEW.uuid IS NULL OR TRIM(NEW.uuid)='')
        ) OR (
          OLD.domain='slides' AND OLD.queue='default'
            AND OLD.job_type='presentation.generate'
            AND (
              NEW.uuid IS NOT OLD.uuid
              OR NEW.domain IS NOT OLD.domain
              OR NEW.queue IS NOT OLD.queue
              OR NEW.job_type IS NOT OLD.job_type
            )
        )
        BEGIN
          SELECT RAISE(ABORT, 'presentation.generate archive UUID is immutable');
        END
        """
    )
    if duplicate_count == 0:
        try:
            conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_archive_uuid_unique
                ON jobs_archive(uuid)
                WHERE uuid IS NOT NULL
                """
            )
        except sqlite3.IntegrityError:
            raced_row = conn.execute(
                """
                SELECT COALESCE(SUM(candidate_count), 0)
                FROM (
                  SELECT COUNT(*) AS candidate_count
                  FROM jobs_archive
                  WHERE uuid IS NOT NULL
                  GROUP BY uuid
                  HAVING COUNT(*) > 1
                )
                """
            ).fetchone()
            conn.execute(
                """
                UPDATE slides_standalone_reconciliation
                SET diagnostic_code='duplicate_archive_uuid', diagnostic_count=?,
                    diagnostic_at=DATETIME('now')
                WHERE singleton_id=1
                """,
                (int(raced_row[0] or 1) if raced_row else 1,),
            )

    conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS trg_jobs_slides_generation_uuid_required
        BEFORE INSERT ON jobs
        FOR EACH ROW
        WHEN NEW.domain='slides'
          AND NEW.queue='default'
          AND NEW.job_type='presentation.generate'
          AND (NEW.uuid IS NULL OR TRIM(NEW.uuid)='')
        BEGIN
          SELECT RAISE(ABORT, 'presentation.generate jobs require an immutable UUID');
        END
        """
    )


def _record_slides_audit_failure_sqlite(conn: sqlite3.Connection) -> None:
    """Persist a bounded fail-closed diagnostic after an audit rollback."""
    result = conn.execute(
        """
        UPDATE slides_standalone_reconciliation
        SET diagnostic_code=CASE
              WHEN diagnostic_code='duplicate_archive_uuid'
              THEN diagnostic_code ELSE 'ambiguous_generation_legacy_row' END,
            diagnostic_count=CASE
              WHEN diagnostic_code='duplicate_archive_uuid'
              THEN diagnostic_count ELSE MAX(COALESCE(diagnostic_count, 0), 1) END,
            diagnostic_at=CASE
              WHEN diagnostic_code='duplicate_archive_uuid'
              THEN diagnostic_at ELSE DATETIME('now') END
        WHERE singleton_id=1
        """
    )
    if result.rowcount != 1:
        raise _SlidesAuditSafetyError(
            "standalone audit readiness singleton is unavailable"
        )


def _upgrade_legacy_admin_webhook_archives_sqlite(
    conn: sqlite3.Connection,
) -> None:
    """Backfill only strictly reconstructable reserved archive evidence."""

    cursor = conn.execute(
        "SELECT * FROM jobs_archive WHERE domain=? AND queue=? AND job_type=?",
        (
            ADMIN_WEBHOOK_DELIVERY_DOMAIN,
            ADMIN_WEBHOOK_DELIVERY_QUEUE,
            ADMIN_WEBHOOK_DELIVERY_JOB_TYPE,
        ),
    )
    columns = tuple(str(item[0]) for item in cursor.description or ())
    for values in cursor.fetchall():
        row = normalize_slides_archive_projection(dict(zip(columns, values)))
        fingerprint = reconstruct_legacy_admin_webhook_archive_fingerprint(row)
        if fingerprint is None:
            continue
        updated = conn.execute(
            "UPDATE jobs_archive SET expired_lease_policy=?, "
            "quarantine_threshold=?, prepared_disposition_fingerprint=?, "
            "no_attempt_recovery_fingerprint=NULL WHERE archive_id=? "
            "AND expired_lease_policy=? "
            "AND quarantine_threshold IS NULL "
            "AND prepared_disposition_fingerprint IS NULL "
            "AND no_attempt_recovery_fingerprint IS NULL",
            (
                ExpiredLeasePolicy.REQUEUE_NO_ATTEMPT.value,
                ADMIN_WEBHOOK_DELIVERY_QUARANTINE_THRESHOLD,
                fingerprint,
                row.get("archive_id"),
                ExpiredLeasePolicy.CONSUME_RETRY.value,
            ),
        )
        if updated.rowcount != 1:
            raise RuntimeError(
                "canonical admin webhook legacy archive upgrade lost its row"
            )


def ensure_jobs_tables(db_path: Path | None = None) -> Path:
    """Ensure the jobs table exists in the given SQLite database.

    Args:
        db_path: Optional path to the SQLite database; defaults to Databases/jobs.db

    Returns:
        Path to the database used
    """
    if db_path is None:
        environment_path = str(os.getenv("JOBS_DB_PATH") or "").strip()
        if environment_path:
            db_path = Path(environment_path).expanduser()

    if db_path is None:
        # Anchor default path to project root to avoid CWD effects
        try:
            from tldw_Server_API.app.core.Utils.Utils import get_project_root as _gpr

            db_path = (Path(_gpr()) / "Databases" / "jobs.db").resolve()
        except _JOBS_PATH_EXCEPTIONS:
            db_path = (Path(__file__).resolve().parents[5] / "Databases" / "jobs.db").resolve()
    else:
        try:
            db_path = Path(db_path)
            if not db_path.is_absolute():
                from tldw_Server_API.app.core.Utils.Utils import get_project_root as _gpr

                db_path = (Path(_gpr()) / db_path).resolve()
        except _JOBS_PATH_EXCEPTIONS:
            db_path = Path(db_path)
    with contextlib.suppress(_JOBS_PATH_EXCEPTIONS):
        db_path.parent.mkdir(parents=True, exist_ok=True)
    archive_locator_verified = False
    try:
        with sqlite3.connect(db_path) as conn:
            # SQLite tuning for better concurrency
            try:
                configure_sqlite_connection(
                    conn,
                    use_wal=True,
                    synchronous="NORMAL",
                    busy_timeout_ms=5000,
                    foreign_keys=False,
                    temp_store=None,
                )
            except _JOBS_DB_EXCEPTIONS:
                pass
            conn.execute(
                "PRAGMA busy_timeout = "
                f"{_sqlite_archive_migration_busy_timeout_ms()}"
            )
            conn.executescript(JOBS_SQLITE_DDL)
            if not conn.in_transaction:
                conn.execute("BEGIN IMMEDIATE")
            with contextlib.suppress(_JOBS_DB_EXCEPTIONS):
                conn.execute("ALTER TABLE jobs ADD COLUMN batch_group TEXT")
            columns = {
                str(row[1])
                for row in conn.execute("PRAGMA table_info(jobs)").fetchall()
            }
            if "expired_lease_policy" not in columns:
                conn.execute(
                    "ALTER TABLE jobs ADD COLUMN expired_lease_policy TEXT "
                    "NOT NULL DEFAULT 'consume_retry' CHECK "
                    "(expired_lease_policy IN ('consume_retry','requeue_no_attempt'))"
                )
            if "quarantine_threshold" not in columns:
                conn.execute(
                    "ALTER TABLE jobs ADD COLUMN quarantine_threshold INTEGER "
                    "CHECK (quarantine_threshold IS NULL OR quarantine_threshold > 0)"
                )
            if "prepared_disposition_fingerprint" not in columns:
                conn.execute(
                    "ALTER TABLE jobs ADD COLUMN prepared_disposition_fingerprint TEXT "
                    "CHECK (prepared_disposition_fingerprint IS NULL OR "
                    "(LENGTH(prepared_disposition_fingerprint) = 64 AND "
                    "prepared_disposition_fingerprint NOT GLOB '*[^0-9a-f]*'))"
                )
            if "no_attempt_recovery_fingerprint" not in columns:
                conn.execute(
                    "ALTER TABLE jobs ADD COLUMN no_attempt_recovery_fingerprint TEXT "
                    "CHECK (no_attempt_recovery_fingerprint IS NULL OR "
                    "(LENGTH(no_attempt_recovery_fingerprint) = 64 AND "
                    "no_attempt_recovery_fingerprint NOT GLOB '*[^0-9a-f]*'))"
                )
            archive_columns = {
                str(row[1])
                for row in conn.execute(
                    "PRAGMA table_info(jobs_archive)"
                ).fetchall()
            }
            if "expired_lease_policy" not in archive_columns:
                conn.execute(
                    "ALTER TABLE jobs_archive ADD COLUMN expired_lease_policy TEXT "
                    "NOT NULL DEFAULT 'consume_retry' CHECK "
                    "(expired_lease_policy IN ('consume_retry','requeue_no_attempt'))"
                )
            if "quarantine_threshold" not in archive_columns:
                conn.execute(
                    "ALTER TABLE jobs_archive ADD COLUMN quarantine_threshold INTEGER "
                    "CHECK (quarantine_threshold IS NULL OR quarantine_threshold > 0)"
                )
            if "prepared_disposition_fingerprint" not in archive_columns:
                conn.execute(
                    "ALTER TABLE jobs_archive ADD COLUMN "
                    "prepared_disposition_fingerprint TEXT CHECK "
                    "(prepared_disposition_fingerprint IS NULL OR "
                    "(LENGTH(prepared_disposition_fingerprint) = 64 AND "
                    "prepared_disposition_fingerprint NOT GLOB '*[^0-9a-f]*'))"
                )
            if "no_attempt_recovery_fingerprint" not in archive_columns:
                conn.execute(
                    "ALTER TABLE jobs_archive ADD COLUMN "
                    "no_attempt_recovery_fingerprint TEXT CHECK "
                    "(no_attempt_recovery_fingerprint IS NULL OR "
                    "(LENGTH(no_attempt_recovery_fingerprint) = 64 AND "
                    "no_attempt_recovery_fingerprint NOT GLOB '*[^0-9a-f]*'))"
                )
            with contextlib.suppress(_JOBS_DB_EXCEPTIONS):
                conn.execute("ALTER TABLE jobs_archive ADD COLUMN batch_group TEXT")
            with contextlib.suppress(_JOBS_DB_EXCEPTIONS):
                conn.execute(
                    "ALTER TABLE jobs_archive ADD COLUMN owner_user_id TEXT"
                )
            for column_sql in (
                "completion_token TEXT",
                "failure_streak_code TEXT",
                "failure_streak_count INTEGER",
                "quarantined_at TEXT",
                "request_id TEXT",
                "trace_id TEXT",
                "failure_timeline TEXT",
                "error_code TEXT",
            ):
                with contextlib.suppress(_JOBS_DB_EXCEPTIONS):
                    conn.execute(f"ALTER TABLE jobs_archive ADD COLUMN {column_sql}")  # nosec B608
            _upgrade_legacy_admin_webhook_archives_sqlite(conn)
            _ensure_sqlite_dependency_snapshot_columns(conn)
            try:
                _record_slides_audit_failure_sqlite(conn)
                conn.execute("SAVEPOINT slides_generation_audit")
            except _JOBS_DB_EXCEPTIONS as poison_error:
                conn.rollback()
                raise _SlidesAuditSafetyError(
                    "standalone audit could not establish fail-closed readiness"
                ) from poison_error
            try:
                _audit_and_index_slides_generation(conn)
            except _SLIDES_AUDIT_EXCEPTIONS as audit_error:
                try:
                    conn.execute("ROLLBACK TO SAVEPOINT slides_generation_audit")
                    conn.execute("RELEASE SAVEPOINT slides_generation_audit")
                    conn.commit()
                except _JOBS_DB_EXCEPTIONS as persistence_error:
                    conn.rollback()
                    raise _SlidesAuditSafetyError(
                        "standalone audit failure could not be persisted"
                    ) from persistence_error
                logger.warning(
                    "Jobs standalone audit failed closed ({})",
                    type(audit_error).__name__,
                )
            else:
                try:
                    conn.execute("RELEASE SAVEPOINT slides_generation_audit")
                    conn.commit()
                except _JOBS_DB_EXCEPTIONS as persistence_error:
                    conn.rollback()
                    raise _SlidesAuditSafetyError(
                        "standalone audit result could not be persisted"
                    ) from persistence_error
            _ensure_sqlite_archive_locators(conn)
            _ensure_sqlite_archive_batch_read_indexes(conn)
            archive_locator_verified = True
            with contextlib.suppress(_JOBS_DB_EXCEPTIONS):
                conn.execute(SQLITE_ARCHIVE_CURSOR_INDEX_SQL)
        try:
            logger.info(f"Ensured Jobs schema at {Path(db_path).resolve()}")
        except _JOBS_PATH_EXCEPTIONS:
            logger.info(f"Ensured Jobs schema at {db_path}")
    except _JOBS_DB_EXCEPTIONS as e:
        logger.warning("Failed to ensure Jobs schema ({})", type(e).__name__)
        if not archive_locator_verified:
            raise
    return db_path
