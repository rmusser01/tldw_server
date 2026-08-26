"""
Jobs module migrations (PostgreSQL).

Provides SQL DDL to provision a `jobs` table compatible with the core JobManager
semantics. This module does not connect to Postgres directly; callers should
apply this DDL using their own connection or via a future Postgres JobManager.
"""

import contextlib
import os
from typing import Any

from tldw_Server_API.app.core.testing import is_truthy as _is_truthy

from .migrations import (
    SLIDES_ARCHIVE_COMPRESSED_FIELDS,
    SLIDES_ARCHIVE_EXACT_FIELDS,
    normalize_slides_archive_projection,
)

_JOBS_PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS = (
    AssertionError,
    AttributeError,
    ConnectionError,
    FileNotFoundError,
    ImportError,
    IndexError,
    KeyError,
    LookupError,
    OSError,
    PermissionError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    UnicodeDecodeError,
)

POSTGRES_ARCHIVE_CURSOR_TIME_SQL = (
    "COALESCE(created_at, archived_at, "
    "TIMESTAMPTZ '0001-01-01 00:00:00+00')"
)
POSTGRES_ARCHIVE_CURSOR_INDEX_SQL = (
    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_jobs_archive_cursor_v2 "
    "ON jobs_archive(domain, job_type, "
    f"{POSTGRES_ARCHIVE_CURSOR_TIME_SQL}, "
    "id, COALESCE(uuid, ''), archive_id)"
)
POSTGRES_ARCHIVE_LOOKUP_ID_INDEX_SQL = (
    "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_jobs_archive_lookup_id "
    "ON jobs_archive(id, archive_id DESC)"
)
POSTGRES_ARCHIVE_BATCH_GROUP_SCOPE_INDEX_SQL = (
    "CREATE INDEX CONCURRENTLY IF NOT EXISTS "
    "idx_jobs_archive_batch_group_scope "
    "ON jobs_archive(batch_group, domain, owner_user_id, job_type, "
    "archive_id DESC)"
)
_POSTGRES_ARCHIVE_BATCH_READ_INDEX_LOCK = (
    "tldw.jobs_archive.batch_read_indexes.v1"
)
_POSTGRES_ARCHIVE_BATCH_READ_INDEX_SPECS = (
    (
        "idx_jobs_archive_lookup_id",
        POSTGRES_ARCHIVE_LOOKUP_ID_INDEX_SQL,
        "DROP INDEX CONCURRENTLY idx_jobs_archive_lookup_id",
        ("id", "archive_id DESC"),
    ),
    (
        "idx_jobs_archive_batch_group_scope",
        POSTGRES_ARCHIVE_BATCH_GROUP_SCOPE_INDEX_SQL,
        "DROP INDEX CONCURRENTLY idx_jobs_archive_batch_group_scope",
        (
            "batch_group",
            "domain",
            "owner_user_id",
            "job_type",
            "archive_id DESC",
        ),
    ),
)
_POSTGRES_ARCHIVE_SEQUENCE = "jobs_archive_archive_id_seq"
_POSTGRES_ARCHIVE_MIGRATION_LOCK = "tldw.jobs_archive.archive_id.v1"

_SLIDES_ARCHIVE_INDEXES_PG = {
    "idx_jobs_archive_slides_scope": (
        False,
        (
            "domain",
            "queue",
            "job_type",
            "idempotency_key",
            "owner_user_id",
            "archived_at desc",
        ),
        "idempotency_key is not null",
    ),
    "idx_jobs_archive_uuid_unique": (
        True,
        ("uuid",),
        "uuid is not null",
    ),
}


def _normalize_pg_index_expression(value: Any) -> str:
    normalized = " ".join(str(value or "").replace('"', "").lower().split())
    while normalized.startswith("(") and normalized.endswith(")"):
        normalized = normalized[1:-1].strip()
    return normalized


def _pg_archive_index_matches(
    cur: Any,
    *,
    index_name: str,
    unique: bool,
    columns: tuple[str, ...],
    predicate: str,
) -> bool:
    """Return whether one archive index has the exact required catalog shape."""
    cur.execute(
        """
        SELECT index_state.indisvalid AS is_valid,
               index_state.indisready AS is_ready,
               index_state.indisunique AS is_unique,
               index_state.indnatts AS total_attributes,
               ARRAY(
                 SELECT pg_get_indexdef(index_state.indexrelid, position, TRUE)
                 FROM generate_series(1, index_state.indnkeyatts) AS positions(position)
                 ORDER BY position
               ) AS key_columns,
               pg_get_expr(index_state.indpred, index_state.indrelid, TRUE) AS predicate
        FROM pg_class AS index_class
        JOIN pg_index AS index_state ON index_state.indexrelid=index_class.oid
        JOIN pg_class AS table_class ON table_class.oid=index_state.indrelid
        JOIN pg_namespace AS namespace ON namespace.oid=index_class.relnamespace
        WHERE namespace.nspname=current_schema()
          AND index_class.relname=%s
          AND table_class.relname='jobs_archive'
        """,
        (index_name,),
    )
    row = cur.fetchone()
    if row is None:
        return False
    if isinstance(row, dict):
        is_valid = row["is_valid"]
        is_ready = row["is_ready"]
        is_unique = row["is_unique"]
        total_attributes = row["total_attributes"]
        actual_columns = row["key_columns"]
        actual_predicate = row["predicate"]
    else:
        is_valid, is_ready, is_unique, total_attributes, actual_columns, actual_predicate = row
    return (
        bool(is_valid)
        and bool(is_ready)
        and bool(is_unique) is unique
        and int(total_attributes) == len(columns)
        and tuple(_normalize_pg_index_expression(item) for item in (actual_columns or ())) == columns
        and _normalize_pg_index_expression(actual_predicate) == predicate
    )


def slides_archive_indexes_ready_pg(cur: Any) -> bool:
    """Return whether both PostgreSQL archive indexes exactly match the contract."""
    return all(
        _pg_archive_index_matches(
            cur,
            index_name=index_name,
            unique=unique,
            columns=columns,
            predicate=predicate,
        )
        for index_name, (unique, columns, predicate) in _SLIDES_ARCHIVE_INDEXES_PG.items()
    )


def slides_archive_projection_ready_pg(cur: Any) -> bool:
    """Return whether active/archive tables expose the complete exact projection."""
    cur.execute(
        """
        SELECT table_name, column_name
        FROM information_schema.columns
        WHERE table_schema=current_schema()
          AND table_name IN ('jobs', 'jobs_archive')
        """
    )
    columns: dict[str, set[str]] = {"jobs": set(), "jobs_archive": set()}
    for row in cur.fetchall() or ():
        if isinstance(row, dict):
            table_name = row["table_name"]
            column_name = row["column_name"]
        else:
            table_name, column_name = row
        columns.setdefault(str(table_name), set()).add(str(column_name))
    return (
        {"id", *SLIDES_ARCHIVE_EXACT_FIELDS} <= columns["jobs"]
        and {
            "id",
            "archived_at",
            *SLIDES_ARCHIVE_EXACT_FIELDS,
            *SLIDES_ARCHIVE_COMPRESSED_FIELDS,
        }
        <= columns["jobs_archive"]
    )


JOBS_POSTGRES_DDL = """
CREATE TABLE IF NOT EXISTS jobs (
  id SERIAL PRIMARY KEY,
  uuid TEXT UNIQUE,
  domain TEXT NOT NULL,
  queue TEXT NOT NULL,
  job_type TEXT NOT NULL,
  owner_user_id TEXT,
  project_id INTEGER,
  batch_group TEXT,
  idempotency_key TEXT,
  payload JSONB,
  result JSONB,
  -- include 'quarantined' for poison message handling
  status TEXT NOT NULL CHECK (status IN ('queued','processing','completed','failed','cancelled','quarantined')),
  priority INTEGER DEFAULT 5 CHECK (priority >= 1 AND priority <= 10),
  max_retries INTEGER DEFAULT 3 CHECK (max_retries >= 0 AND max_retries <= 100),
  retry_count INTEGER DEFAULT 0,
  available_at TIMESTAMPTZ,
  started_at TIMESTAMPTZ,
  leased_until TIMESTAMPTZ,
  lease_id TEXT,
  worker_id TEXT,
  acquired_at TIMESTAMPTZ,
  error_message TEXT,
  error_code TEXT,
  error_class TEXT,
  error_stack JSONB,
  last_error TEXT,
  cancel_requested_at TIMESTAMPTZ,
  cancelled_at TIMESTAMPTZ,
  cancellation_reason TEXT,
  completion_token TEXT,
  failure_streak_code TEXT,
  failure_streak_count INTEGER DEFAULT 0,
  quarantined_at TIMESTAMPTZ,
  progress_percent REAL CHECK (progress_percent IS NULL OR (progress_percent >= 0 AND progress_percent <= 100)),
  progress_message TEXT,
  -- correlation
  request_id TEXT,
  trace_id TEXT,
  -- structured failure history (JSONB array of objects)
  failure_timeline JSONB,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  completed_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_jobs_lookup
  ON jobs(domain, queue, status, available_at, priority, created_at);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_lease ON jobs(leased_until);
CREATE INDEX IF NOT EXISTS idx_jobs_owner_status ON jobs(owner_user_id, status, created_at);
CREATE INDEX IF NOT EXISTS idx_jobs_batch_group ON jobs(batch_group);

-- updated_at trigger
CREATE OR REPLACE FUNCTION set_jobs_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at := NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_jobs_updated_at ON jobs;
CREATE TRIGGER trg_jobs_updated_at
BEFORE UPDATE ON jobs
FOR EACH ROW EXECUTE FUNCTION set_jobs_updated_at();

-- Append-only outbox for job events (CDC)
CREATE TABLE IF NOT EXISTS job_events (
  id BIGSERIAL PRIMARY KEY,
  job_id INTEGER,
  domain TEXT,
  queue TEXT,
  job_type TEXT,
  event_type TEXT NOT NULL,
  attrs_json JSONB,
  owner_user_id TEXT,
  request_id TEXT,
  trace_id TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_job_events_id ON job_events(id);
CREATE INDEX IF NOT EXISTS idx_job_events_job_id ON job_events(job_id);

-- Optional archive table (used when JOBS_ARCHIVE_BEFORE_DELETE=true)
CREATE TABLE IF NOT EXISTS jobs_archive (
  archive_id BIGSERIAL CONSTRAINT idx_jobs_archive_id PRIMARY KEY,
  id INTEGER,
  uuid TEXT,
  domain TEXT NOT NULL,
  queue TEXT NOT NULL,
  job_type TEXT NOT NULL,
  owner_user_id TEXT,
  project_id INTEGER,
  batch_group TEXT,
  idempotency_key TEXT,
  payload JSONB,
  result JSONB,
  status TEXT NOT NULL,
  priority INTEGER,
  max_retries INTEGER,
  retry_count INTEGER,
  available_at TIMESTAMPTZ,
  started_at TIMESTAMPTZ,
  leased_until TIMESTAMPTZ,
  lease_id TEXT,
  worker_id TEXT,
  acquired_at TIMESTAMPTZ,
  error_message TEXT,
  error_code TEXT,
  last_error TEXT,
  cancel_requested_at TIMESTAMPTZ,
  cancelled_at TIMESTAMPTZ,
  cancellation_reason TEXT,
  completion_token TEXT,
  failure_streak_code TEXT,
  failure_streak_count INTEGER,
  quarantined_at TIMESTAMPTZ,
  progress_percent REAL,
  progress_message TEXT,
  request_id TEXT,
  trace_id TEXT,
  failure_timeline JSONB,
  -- Optional compressed copies for payload/result when archiving (BYTEA)
  payload_compressed BYTEA,
  result_compressed BYTEA,
  created_at TIMESTAMPTZ,
  updated_at TIMESTAMPTZ,
  completed_at TIMESTAMPTZ,
  archived_at TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_jobs_archive_migration
  ON jobs_archive(
    domain,
    job_type,
    status,
    COALESCE(
      created_at,
      archived_at,
      TIMESTAMPTZ '0001-01-01 00:00:00+00'
    ),
    id,
    COALESCE(uuid, '')
  );

-- Immutable correlation for user-facing idempotent operations. Job state,
-- progress, result, and errors remain exclusively in jobs/jobs_archive.
CREATE TABLE IF NOT EXISTS job_idempotency_receipts (
  receipt_id BIGSERIAL PRIMARY KEY,
  domain TEXT NOT NULL CHECK (LENGTH(domain) BETWEEN 1 AND 64),
  queue TEXT NOT NULL CHECK (LENGTH(queue) BETWEEN 1 AND 64),
  job_type TEXT NOT NULL CHECK (LENGTH(job_type) BETWEEN 1 AND 128),
  owner_user_id TEXT NOT NULL CHECK (LENGTH(owner_user_id) BETWEEN 1 AND 200),
  key_digest TEXT NOT NULL CHECK (
    LENGTH(key_digest) = 64 AND key_digest ~ '^[0-9a-f]{64}$'
  ),
  request_fingerprint TEXT NOT NULL CHECK (
    LENGTH(request_fingerprint) = 64
    AND request_fingerprint ~ '^[0-9a-f]{64}$'
  ),
  operation_scope TEXT NOT NULL CHECK (LENGTH(operation_scope) BETWEEN 1 AND 200),
  job_uuid TEXT NOT NULL CHECK (LENGTH(job_uuid) BETWEEN 1 AND 64),
  job_id INTEGER NOT NULL,
  created_at TIMESTAMPTZ NOT NULL,
  expires_at TIMESTAMPTZ NOT NULL
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

CREATE OR REPLACE FUNCTION enforce_slides_generation_uuid_immutable()
RETURNS TRIGGER AS $$
BEGIN
  IF (
    NEW.domain='slides' AND NEW.queue='default'
    AND NEW.job_type='presentation.generate'
    AND (NEW.uuid IS NULL OR BTRIM(NEW.uuid)='')
  ) OR (
    OLD.domain='slides' AND OLD.queue='default'
    AND OLD.job_type='presentation.generate'
    AND (
      NEW.uuid IS DISTINCT FROM OLD.uuid
      OR NEW.domain IS DISTINCT FROM OLD.domain
      OR NEW.queue IS DISTINCT FROM OLD.queue
      OR NEW.job_type IS DISTINCT FROM OLD.job_type
    )
  ) THEN
    RAISE EXCEPTION 'presentation.generate UUID is required and immutable';
  END IF;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_jobs_slides_generation_uuid_immutable ON jobs;
CREATE TRIGGER trg_jobs_slides_generation_uuid_immutable
BEFORE UPDATE ON jobs
FOR EACH ROW EXECUTE FUNCTION enforce_slides_generation_uuid_immutable();

DROP TRIGGER IF EXISTS trg_jobs_archive_slides_generation_uuid_immutable ON jobs_archive;
CREATE TRIGGER trg_jobs_archive_slides_generation_uuid_immutable
BEFORE UPDATE ON jobs_archive
FOR EACH ROW EXECUTE FUNCTION enforce_slides_generation_uuid_immutable();

-- Source-free standalone-HTML key metadata. No secret or digest material is
-- persisted in the shared Jobs database.
CREATE TABLE IF NOT EXISTS slides_standalone_key_registry (
  key_id TEXT PRIMARY KEY,
  state TEXT NOT NULL CHECK (state IN ('current','retiring')),
  activated_at TIMESTAMPTZ NOT NULL,
  retired_at TIMESTAMPTZ,
  config_revision TEXT NOT NULL,
  CHECK (
    (state = 'current' AND retired_at IS NULL)
    OR (state = 'retiring' AND retired_at IS NOT NULL)
  )
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_slides_standalone_one_current_key
  ON slides_standalone_key_registry(state)
  WHERE state = 'current';

CREATE TABLE IF NOT EXISTS slides_standalone_reconciliation (
  singleton_id SMALLINT PRIMARY KEY CHECK (singleton_id = 1),
  holder_uuid TEXT,
  lease_expires_at TIMESTAMPTZ,
  fencing_token BIGINT NOT NULL DEFAULT 0 CHECK (fencing_token >= 0),
  cursor TEXT,
  config_revision TEXT,
  startup_complete_epoch TEXT,
  last_complete_epoch DOUBLE PRECISION,
  lag BIGINT NOT NULL DEFAULT 0 CHECK (lag >= 0),
  diagnostic_code TEXT CHECK (
    diagnostic_code IS NULL
    OR diagnostic_code IN ('duplicate_archive_uuid','ambiguous_generation_legacy_row')
  ),
  diagnostic_count BIGINT NOT NULL DEFAULT 0 CHECK (diagnostic_count >= 0),
  diagnostic_at TIMESTAMPTZ,
  sweep_key_id TEXT,
  sweep_started_at TIMESTAMPTZ,
  sweep_completed_at TIMESTAMPTZ,
  sweep_complete BOOLEAN NOT NULL DEFAULT FALSE,
  unexpired_reference_count BIGINT NOT NULL DEFAULT 0
    CHECK (unexpired_reference_count >= 0)
);
INSERT INTO slides_standalone_reconciliation(singleton_id) VALUES (1)
ON CONFLICT (singleton_id) DO NOTHING;

-- Job dependencies (DAG edges)
CREATE TABLE IF NOT EXISTS job_dependencies (
  job_uuid TEXT NOT NULL,
  depends_on_job_uuid TEXT NOT NULL,
  depends_on_terminal_status TEXT,
  depends_on_cancellation_reason TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  PRIMARY KEY (job_uuid, depends_on_job_uuid)
);
CREATE INDEX IF NOT EXISTS idx_job_dependencies_job ON job_dependencies(job_uuid);
CREATE INDEX IF NOT EXISTS idx_job_dependencies_depends_on ON job_dependencies(depends_on_job_uuid);

-- Status-focused partial indexes to speed common counts and lookups
CREATE INDEX IF NOT EXISTS idx_jobs_status_queued ON jobs(domain, queue, job_type, priority, available_at, created_at) WHERE status='queued';
CREATE INDEX IF NOT EXISTS idx_jobs_status_processing ON jobs(domain, queue, job_type, leased_until) WHERE status='processing';

-- Composite uniqueness for idempotency scoped by domain/queue/job_type (NULL key allowed)
-- A unique index is created outside the DDL block using autocommit.
"""


def _pg_archive_locator_index_state(cur: Any) -> tuple[Any, ...] | None:
    """Return validity metadata for the canonical archive-locator index."""
    cur.execute(
        "SELECT i.indisunique, i.indisvalid, i.indpred IS NULL, "
        "i.indnkeyatts, pg_get_indexdef(i.indexrelid, 1, true), "
        "con.conname "
        "FROM pg_class idx "
        "JOIN pg_namespace ns ON ns.oid = idx.relnamespace "
        "JOIN pg_index i ON i.indexrelid = idx.oid "
        "LEFT JOIN pg_constraint con ON con.conindid = idx.oid "
        "WHERE ns.nspname = current_schema() "
        "AND idx.relname = 'idx_jobs_archive_id' "
        "AND i.indrelid = 'jobs_archive'::regclass"
    )
    return cur.fetchone()


def _pg_archive_locator_index_ready(archive_index: tuple[Any, ...] | None) -> bool:
    """Return whether index metadata enforces one valid locator per row."""
    return bool(
        archive_index is not None
        and archive_index[0]
        and archive_index[1]
        and archive_index[2]
        and int(archive_index[3]) == 1
        and str(archive_index[4]).strip() == "archive_id"
    )


def _pg_archive_batch_read_index_state(
    cur: Any,
    index_name: str,
) -> tuple[Any, ...] | None:
    """Return catalog metadata for one archive batch-read index."""

    cur.execute(
        "SELECT i.indrelid = 'jobs_archive'::regclass, i.indisvalid, "
        "i.indisready, i.indisunique, i.indpred IS NULL, "
        "i.indexprs IS NULL, am.amname, i.indnkeyatts, "
        "ARRAY(SELECT pg_get_indexdef(i.indexrelid, key_position, true) "
        "FROM generate_series(1, i.indnkeyatts) AS key_position "
        "ORDER BY key_position), "
        "ARRAY(SELECT i.indoption[key_position - 1]::integer "
        "FROM generate_series(1, i.indnkeyatts) AS key_position "
        "ORDER BY key_position), con.conname "
        "FROM pg_class idx "
        "JOIN pg_namespace ns ON ns.oid = idx.relnamespace "
        "JOIN pg_index i ON i.indexrelid = idx.oid "
        "JOIN pg_am am ON am.oid = idx.relam "
        "LEFT JOIN pg_constraint con ON con.conindid = idx.oid "
        "WHERE ns.nspname = current_schema() AND idx.relname = %s",
        (index_name,),
    )
    return cur.fetchone()


def _pg_archive_batch_read_index_ready(
    state: tuple[Any, ...] | None,
    expected_columns: tuple[str, ...],
) -> bool:
    """Return whether one PostgreSQL archive lookup index is canonical."""

    expected_names = tuple(
        column.removesuffix(" DESC") for column in expected_columns
    )
    expected_options = tuple(
        3 if column.endswith(" DESC") else 0 for column in expected_columns
    )
    return bool(
        state is not None
        and state[0]
        and state[1]
        and state[2]
        and not state[3]
        and state[4]
        and state[5]
        and str(state[6]) == "btree"
        and int(state[7]) == len(expected_columns)
        and tuple(state[8]) == expected_names
        and tuple(int(option) for option in state[9]) == expected_options
        and state[10] is None
    )


def _ensure_pg_archive_batch_read_indexes(cur: Any) -> None:
    """Create or repair archive lookup indexes once under an advisory lock."""

    lock_acquired = False
    cur.execute(
        "SELECT pg_advisory_lock(hashtextextended(%s, 0))",
        (_POSTGRES_ARCHIVE_BATCH_READ_INDEX_LOCK,),
    )
    lock_acquired = True
    try:
        for index_name, create_sql, drop_sql, expected_columns in (
            _POSTGRES_ARCHIVE_BATCH_READ_INDEX_SPECS
        ):
            state = _pg_archive_batch_read_index_state(cur, index_name)
            if state is not None and not state[0]:
                raise RuntimeError(f"{index_name} belongs to another table")
            if state is not None and state[10] is not None:
                raise RuntimeError(
                    f"{index_name} is a misdefined constraint-backed index"
                )
            if not _pg_archive_batch_read_index_ready(
                state, expected_columns
            ):
                if state is not None:
                    cur.execute(drop_sql)
                cur.execute(create_sql)
            if not _pg_archive_batch_read_index_ready(
                _pg_archive_batch_read_index_state(cur, index_name),
                expected_columns,
            ):
                raise RuntimeError(f"{index_name} verification failed")
    finally:
        if lock_acquired:
            with contextlib.suppress(
                _JOBS_PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS
            ):
                cur.execute(
                    "SELECT pg_advisory_unlock(hashtextextended(%s, 0))",
                    (_POSTGRES_ARCHIVE_BATCH_READ_INDEX_LOCK,),
                )


def _pg_archive_column_default_uses_locator_sequence(cur: Any) -> bool:
    """Return whether archive_id's default depends on the canonical sequence."""
    cur.execute(
        "SELECT dep.refobjid = %s::regclass "
        "FROM pg_attrdef defaults "
        "JOIN pg_attribute attr ON attr.attrelid = defaults.adrelid "
        "AND attr.attnum = defaults.adnum "
        "JOIN pg_depend dep ON dep.classid = 'pg_attrdef'::regclass "
        "AND dep.objid = defaults.oid "
        "JOIN pg_class referenced ON referenced.oid = dep.refobjid "
        "WHERE defaults.adrelid = 'jobs_archive'::regclass "
        "AND attr.attname = 'archive_id' "
        "AND dep.refclassid = 'pg_class'::regclass "
        "AND referenced.relkind = 'S'",
        (_POSTGRES_ARCHIVE_SEQUENCE,),
    )
    sequence_dependencies = [bool(row[0]) for row in cur.fetchall() or []]
    return bool(sequence_dependencies) and all(sequence_dependencies)


def _pg_archive_locator_schema_ready(cur: Any) -> bool:
    """Return whether archive locators are safe for concurrent allocation."""

    cur.execute(
        "SELECT column_default, is_nullable, data_type "
        "FROM information_schema.columns "
        "WHERE table_schema = current_schema() "
        "AND table_name = 'jobs_archive' AND column_name = 'archive_id'"
    )
    column = cur.fetchone()
    if not (
        column is not None
        and str(column[1]) == "NO"
        and str(column[2]) == "bigint"
    ):
        return False

    cur.execute(
        "SELECT pg_get_serial_sequence("
        "quote_ident(current_schema()) || '.jobs_archive', 'archive_id')"
    )
    owned_sequence = cur.fetchone()[0]
    if not (
        owned_sequence
        and str(owned_sequence).split(".")[-1].strip('"')
        == _POSTGRES_ARCHIVE_SEQUENCE
    ):
        return False
    if not _pg_archive_column_default_uses_locator_sequence(cur):
        return False
    if _pg_archive_sequence_conflict(cur) is not None:
        return False

    if not _pg_archive_locator_index_ready(
        _pg_archive_locator_index_state(cur)
    ):
        return False

    cur.execute("SELECT to_regclass(%s)", (_POSTGRES_ARCHIVE_SEQUENCE,))
    if cur.fetchone()[0] is None:
        return False
    cur.execute("SELECT COALESCE(MAX(archive_id), 0) FROM jobs_archive")
    max_locator = int(cur.fetchone()[0])
    cur.execute(
        "SELECT last_value, is_called FROM jobs_archive_archive_id_seq"
    )
    last_value, is_called = cur.fetchone()
    next_locator = int(last_value) + (1 if bool(is_called) else 0)
    return next_locator > max_locator


def _pg_archive_migration_timeout_ms(
    name: str,
    *,
    default: int,
    minimum: int,
) -> int:
    """Return a bounded integer timeout for archive schema migration work."""
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


def _configure_pg_archive_migration_session(
    cur: Any,
    *,
    local: bool = True,
) -> None:
    """Override request-oriented DSN timeouts for bounded schema migration work."""
    statement_timeout_ms = _pg_archive_migration_timeout_ms(
        "JOBS_PG_ARCHIVE_MIGRATION_STATEMENT_TIMEOUT_MS",
        default=300_000,
        minimum=0,
    )
    lock_timeout_ms = _pg_archive_migration_timeout_ms(
        "JOBS_PG_ARCHIVE_MIGRATION_LOCK_TIMEOUT_MS",
        default=30_000,
        minimum=1_000,
    )
    cur.execute(
        "SELECT set_config('statement_timeout', %s, %s)",
        (f"{statement_timeout_ms}ms", local),
    )
    cur.execute(
        "SELECT set_config('lock_timeout', %s, %s)",
        (f"{lock_timeout_ms}ms", local),
    )


def _pg_archive_sequence_conflict(cur: Any) -> str | None:
    """Describe any non-archive ownership or default use of the locator sequence."""
    cur.execute("SELECT to_regclass(%s)", (_POSTGRES_ARCHIVE_SEQUENCE,))
    if cur.fetchone()[0] is None:
        return None

    target_owner = False
    cur.execute(
        "SELECT dep.refobjid = 'jobs_archive'::regclass, attr.attname "
        "FROM pg_depend dep "
        "JOIN pg_attribute attr ON attr.attrelid = dep.refobjid "
        "AND attr.attnum = dep.refobjsubid "
        "WHERE dep.classid = 'pg_class'::regclass "
        "AND dep.objid = %s::regclass "
        "AND dep.refclassid = 'pg_class'::regclass "
        "AND dep.deptype IN ('a', 'i')",
        (_POSTGRES_ARCHIVE_SEQUENCE,),
    )
    for owns_archive, column_name in cur.fetchall() or []:
        if bool(owns_archive) and str(column_name) == "archive_id":
            target_owner = True
        else:
            return "owned by another table or column"

    target_default = False
    cur.execute(
        "SELECT defaults.adrelid = 'jobs_archive'::regclass, attr.attname "
        "FROM pg_attrdef defaults "
        "JOIN pg_depend dep ON dep.classid = 'pg_attrdef'::regclass "
        "AND dep.objid = defaults.oid "
        "JOIN pg_attribute attr ON attr.attrelid = defaults.adrelid "
        "AND attr.attnum = defaults.adnum "
        "WHERE dep.refclassid = 'pg_class'::regclass "
        "AND dep.refobjid = %s::regclass",
        (_POSTGRES_ARCHIVE_SEQUENCE,),
    )
    for defaults_archive, column_name in cur.fetchall() or []:
        if bool(defaults_archive) and str(column_name) == "archive_id":
            target_default = True
        else:
            return "used by another table or column default"
    if not (target_owner or target_default):
        return "present without archive ownership or default binding"
    return None


def _ensure_pg_archive_locators(dsn: str) -> None:
    """Run the legacy archive-locator upgrade once under migration locks."""

    import psycopg
    from psycopg import sql

    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        _configure_pg_archive_migration_session(cur)
        if _pg_archive_locator_schema_ready(cur):
            return

        cur.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
            (_POSTGRES_ARCHIVE_MIGRATION_LOCK,),
        )
        cur.execute("LOCK TABLE jobs_archive IN ACCESS EXCLUSIVE MODE")
        if _pg_archive_locator_schema_ready(cur):
            return

        cur.execute(
            "ALTER TABLE jobs_archive ADD COLUMN IF NOT EXISTS archive_id BIGINT"
        )
        cur.execute(
            "SELECT data_type FROM information_schema.columns "
            "WHERE table_schema = current_schema() "
            "AND table_name = 'jobs_archive' AND column_name = 'archive_id'"
        )
        archive_id_type = str(cur.fetchone()[0])
        if archive_id_type in {"smallint", "integer"}:
            cur.execute(
                "ALTER TABLE jobs_archive ALTER COLUMN archive_id "
                "TYPE BIGINT USING archive_id::bigint"
            )
        elif archive_id_type != "bigint":
            raise RuntimeError(
                "jobs_archive.archive_id must use an integer-compatible type"
            )

        archive_index = _pg_archive_locator_index_state(cur)
        archive_index_ready = _pg_archive_locator_index_ready(archive_index)
        constraint_name = archive_index[5] if archive_index else None
        if not archive_index_ready:
            if constraint_name:
                raise RuntimeError(
                    "idx_jobs_archive_id is a misdefined constraint-backed index"
                )
            if archive_index is None:
                cur.execute("SELECT to_regclass('idx_jobs_archive_id')")
                conflicting_index = cur.fetchone()[0]
                if conflicting_index is not None:
                    raise RuntimeError(
                        "idx_jobs_archive_id belongs to another table"
                    )

        sequence_conflict = _pg_archive_sequence_conflict(cur)
        if sequence_conflict:
            raise RuntimeError(
                f"jobs_archive_archive_id_seq is {sequence_conflict}"
            )
        if not archive_index_ready:
            cur.execute(
                "SELECT archive_id FROM jobs_archive "
                "WHERE archive_id IS NOT NULL GROUP BY archive_id "
                "HAVING COUNT(*) > 1 LIMIT 1"
            )
            if cur.fetchone() is not None:
                raise RuntimeError(
                    "jobs_archive contains duplicate archive_id values"
                )

        cur.execute(
            "CREATE SEQUENCE IF NOT EXISTS jobs_archive_archive_id_seq"
        )
        cur.execute(
            "ALTER SEQUENCE jobs_archive_archive_id_seq "
            "OWNED BY jobs_archive.archive_id"
        )
        cur.execute(
            "ALTER TABLE jobs_archive ALTER COLUMN archive_id "
            "SET DEFAULT nextval('jobs_archive_archive_id_seq'::regclass)"
        )
        cur.execute(
            "SELECT pg_catalog.setval("
            "'jobs_archive_archive_id_seq'::regclass, "
            "GREATEST("
            "COALESCE((SELECT MAX(archive_id) FROM jobs_archive), 0) + 1, "
            "(SELECT CASE WHEN is_called THEN last_value + 1 "
            "ELSE last_value END FROM jobs_archive_archive_id_seq)"
            "), false)"
        )
        cur.execute(
            "UPDATE jobs_archive SET archive_id = "
            "nextval('jobs_archive_archive_id_seq'::regclass) "
            "WHERE archive_id IS NULL"
        )
        if not archive_index_ready:
            if archive_index is not None:
                cur.execute(
                    sql.SQL("DROP INDEX {}").format(
                        sql.Identifier("idx_jobs_archive_id")
                    )
                )
            cur.execute(
                "CREATE UNIQUE INDEX idx_jobs_archive_id "
                "ON jobs_archive(archive_id)"
            )
        cur.execute(
            "ALTER TABLE jobs_archive ALTER COLUMN archive_id SET NOT NULL"
        )
        if not _pg_archive_locator_schema_ready(cur):
            raise RuntimeError("PostgreSQL Jobs archive locator migration failed")


def _ensure_pg_dependency_snapshot_columns(cur: Any) -> None:
    """Add and verify dependency snapshots required by acquisition queries."""

    cur.execute(
        "ALTER TABLE job_dependencies "
        "ADD COLUMN IF NOT EXISTS depends_on_terminal_status TEXT"
    )
    cur.execute(
        "ALTER TABLE job_dependencies "
        "ADD COLUMN IF NOT EXISTS depends_on_cancellation_reason TEXT"
    )
    cur.execute(
        "SELECT depends_on_terminal_status, "
        "depends_on_cancellation_reason FROM job_dependencies LIMIT 0"
    )

def _audit_slides_generation_pg(cur) -> tuple[str | None, int]:
    """Persist bounded legacy diagnostics before archive index creation."""
    cur.execute(
        """
        SELECT singleton_id
        FROM slides_standalone_reconciliation
        WHERE singleton_id=1
        FOR UPDATE
        """
    )
    if cur.fetchone() is None:
        return "ambiguous_generation_legacy_row", 1
    if not slides_archive_projection_ready_pg(cur):
        cur.execute(
            """
            UPDATE slides_standalone_reconciliation
            SET diagnostic_code=CASE
                  WHEN diagnostic_code='duplicate_archive_uuid'
                  THEN diagnostic_code ELSE 'ambiguous_generation_legacy_row' END,
                diagnostic_count=CASE
                  WHEN diagnostic_code='duplicate_archive_uuid'
                  THEN diagnostic_count ELSE GREATEST(diagnostic_count, 1) END,
                diagnostic_at=CASE
                  WHEN diagnostic_code='duplicate_archive_uuid'
                  THEN diagnostic_at ELSE NOW() END
            WHERE singleton_id=1
            """
        )
        return "ambiguous_generation_legacy_row", 1

    cur.execute(
        """
        SELECT COALESCE(SUM(candidate_count), 0) FROM (
          SELECT COUNT(*) AS candidate_count
          FROM jobs_archive
          WHERE uuid IS NOT NULL
          GROUP BY uuid
          HAVING COUNT(*) > 1
        ) duplicates
        """
    )
    duplicate_count = int((cur.fetchone() or [0])[0] or 0)
    cur.execute(
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
        WHERE uuid IS NULL OR BTRIM(uuid) = ''
           OR owner_user_id IS NULL OR BTRIM(owner_user_id) = ''
           OR idempotency_key IS NULL OR BTRIM(idempotency_key) = ''
        """
    )
    invalid_count = int((cur.fetchone() or [0])[0] or 0)
    active_projection = ", ".join(
        f"active.{field} AS active_{field}" for field in SLIDES_ARCHIVE_EXACT_FIELDS
    )
    archived_projection = ", ".join(
        f"archived.{field} AS archived_{field}" for field in SLIDES_ARCHIVE_EXACT_FIELDS
    )
    cur.execute(
        f"""
        SELECT {active_projection}, {archived_projection},
               archived.payload_compressed AS archived_payload_compressed,
               archived.result_compressed AS archived_result_compressed
        FROM jobs active
        JOIN jobs_archive archived ON archived.uuid = active.uuid
        WHERE active.uuid IS NOT NULL AND BTRIM(active.uuid) <> ''
          AND (
            (active.domain='slides' AND active.queue='default'
             AND active.job_type='presentation.generate')
            OR
            (archived.domain='slides' AND archived.queue='default'
             AND archived.job_type='presentation.generate')
          )
        """  # nosec B608 - field names and aliases come from a closed module constant
    )
    projection_size = len(SLIDES_ARCHIVE_EXACT_FIELDS)
    cross_table_count = 0
    for row in cur.fetchall() or ():
        if isinstance(row, dict):
            active_values = {
                field: row.get(f"active_{field}")
                for field in SLIDES_ARCHIVE_EXACT_FIELDS
            }
            archived_values = {
                field: row.get(f"archived_{field}")
                for field in SLIDES_ARCHIVE_EXACT_FIELDS
            }
            archived_values["payload_compressed"] = row.get(
                "archived_payload_compressed"
            )
            archived_values["result_compressed"] = row.get(
                "archived_result_compressed"
            )
        else:
            active_values = dict(
                zip(SLIDES_ARCHIVE_EXACT_FIELDS, row[:projection_size])
            )
            archived_values = dict(
                zip(
                    SLIDES_ARCHIVE_EXACT_FIELDS,
                    row[projection_size : 2 * projection_size],
                )
            )
            archived_values["payload_compressed"] = row[2 * projection_size]
            archived_values["result_compressed"] = row[2 * projection_size + 1]
        active = normalize_slides_archive_projection(active_values)
        archived = normalize_slides_archive_projection(archived_values)
        if any(
            active.get(field) != archived.get(field)
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
    cur.execute(
        """
        UPDATE slides_standalone_reconciliation
        SET diagnostic_code=%s, diagnostic_count=%s,
            diagnostic_at=CASE WHEN %s IS NULL THEN NULL ELSE NOW() END
        WHERE singleton_id=1
        """,
        (diagnostic_code, diagnostic_count, diagnostic_code),
    )
    return diagnostic_code, diagnostic_count


def _record_duplicate_archive_uuid_pg(dsn: str) -> None:
    """Translate a concurrent unique-index race into standalone diagnostics."""
    import psycopg

    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT COALESCE(SUM(candidate_count), 1) FROM (
              SELECT COUNT(*) AS candidate_count
              FROM jobs_archive
              WHERE uuid IS NOT NULL
              GROUP BY uuid
              HAVING COUNT(*) > 1
            ) duplicates
            """
        )
        count = int((cur.fetchone() or [1])[0] or 1)
        cur.execute(
            """
            UPDATE slides_standalone_reconciliation
            SET diagnostic_code='duplicate_archive_uuid', diagnostic_count=%s,
                diagnostic_at=NOW()
            WHERE singleton_id=1
            """,
            (count,),
        )


def _mark_slides_audit_failure_pg(cur) -> None:
    """Mark standalone readiness fail-closed on the caller's transaction."""
    cur.execute(
        """
        UPDATE slides_standalone_reconciliation
        SET diagnostic_code=CASE
              WHEN diagnostic_code='duplicate_archive_uuid'
              THEN diagnostic_code ELSE 'ambiguous_generation_legacy_row' END,
            diagnostic_count=CASE
              WHEN diagnostic_code='duplicate_archive_uuid'
              THEN diagnostic_count ELSE GREATEST(diagnostic_count, 1) END,
            diagnostic_at=CASE
              WHEN diagnostic_code='duplicate_archive_uuid'
              THEN diagnostic_at ELSE NOW() END
        WHERE singleton_id=1
        """
    )
    if cur.rowcount != 1:
        raise RuntimeError("standalone audit readiness singleton is unavailable")


def _record_slides_audit_failure_pg(dsn: str) -> None:
    """Persist a bounded fail-closed diagnostic when the archive audit errors."""
    import psycopg

    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        _mark_slides_audit_failure_pg(cur)


def ensure_jobs_tables_pg(db_url: str) -> str:
    """Ensure the jobs table exists in the given PostgreSQL database.

    Returns the db_url passed through for convenience.
    """
    try:
        import psycopg
    except _JOBS_PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS as e:  # pragma: no cover - environment dependent
        raise RuntimeError("psycopg is required for PostgreSQL Jobs backend. Install extras 'db_postgres'.") from e

    from .pg_util import negotiate_pg_dsn
    _dsn = negotiate_pg_dsn(db_url)
    try:
        with psycopg.connect(_dsn) as conn, conn.cursor() as cur:
            _configure_pg_archive_migration_session(cur)
            cur.execute(JOBS_POSTGRES_DDL)
            # Additional objects: queue controls, attachments, SLA policies
            cur.execute(
                """
                    CREATE TABLE IF NOT EXISTS job_queue_controls (
                      domain TEXT NOT NULL,
                      queue TEXT NOT NULL,
                      paused BOOLEAN DEFAULT FALSE,
                      drain BOOLEAN DEFAULT FALSE,
                      updated_at TIMESTAMPTZ DEFAULT NOW(),
                      PRIMARY KEY (domain, queue)
                    );
                    """
            )
            cur.execute(
                """
                    CREATE TABLE IF NOT EXISTS job_attachments (
                      id SERIAL PRIMARY KEY,
                      job_id INTEGER NOT NULL,
                      kind TEXT NOT NULL,
                      content_text TEXT,
                      url TEXT,
                      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    );
                    """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_job_attachments_job ON job_attachments(job_id)")
            cur.execute(
                """
                    CREATE TABLE IF NOT EXISTS job_sla_policies (
                      domain TEXT NOT NULL,
                      queue TEXT NOT NULL,
                      job_type TEXT NOT NULL,
                      max_queue_latency_seconds INTEGER,
                      max_duration_seconds INTEGER,
                      enabled BOOLEAN DEFAULT TRUE,
                      updated_at TIMESTAMPTZ DEFAULT NOW(),
                      PRIMARY KEY (domain, queue, job_type)
                    );
                    """
            )
            conn.commit()
        # Forward-migrate older installs: add missing columns that newer code expects
        with psycopg.connect(_dsn, autocommit=True) as cfix, cfix.cursor() as f:
            _configure_pg_archive_migration_session(f, local=False)
            required_migration_exceptions = (
                psycopg.Error,
                *_JOBS_PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS,
            )
            try:
                _ensure_pg_dependency_snapshot_columns(f)
            except required_migration_exceptions as exc:
                raise RuntimeError(
                    "PostgreSQL Jobs dependency snapshot migration failed"
                ) from exc
            try:
                f.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS completion_token TEXT")
                f.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS failure_streak_code TEXT")
                f.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS failure_streak_count INTEGER DEFAULT 0")
                f.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS quarantined_at TIMESTAMPTZ")
                f.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS progress_percent REAL")
                f.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS progress_message TEXT")
                f.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS request_id TEXT")
                f.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS trace_id TEXT")
                f.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS failure_timeline JSONB")
                f.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS error_code TEXT")
                f.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS error_class TEXT")
                f.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS error_stack JSONB")
                f.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS batch_group TEXT")
                f.execute(
                    """
                    DO $$
                    BEGIN
                      IF NOT EXISTS (
                        SELECT 1 FROM pg_constraint
                        WHERE conname='jobs_slides_generation_uuid_required'
                          AND conrelid='jobs'::regclass
                      ) THEN
                        ALTER TABLE jobs ADD CONSTRAINT jobs_slides_generation_uuid_required
                        CHECK (
                          domain <> 'slides' OR queue <> 'default'
                          OR job_type <> 'presentation.generate'
                          OR (uuid IS NOT NULL AND BTRIM(uuid) <> '')
                        ) NOT VALID;
                      END IF;
                    END
                    $$
                    """
                )
                # Forward-migrate archive table compressed columns (if table exists)
                try:
                    f.execute("ALTER TABLE jobs_archive ADD COLUMN IF NOT EXISTS payload_compressed BYTEA")
                    f.execute("ALTER TABLE jobs_archive ADD COLUMN IF NOT EXISTS result_compressed BYTEA")
                    f.execute("ALTER TABLE jobs_archive ADD COLUMN IF NOT EXISTS batch_group TEXT")
                    f.execute("ALTER TABLE jobs_archive ADD COLUMN IF NOT EXISTS owner_user_id TEXT")
                    f.execute("ALTER TABLE jobs_archive ADD COLUMN IF NOT EXISTS completion_token TEXT")
                    f.execute("ALTER TABLE jobs_archive ADD COLUMN IF NOT EXISTS failure_streak_code TEXT")
                    f.execute("ALTER TABLE jobs_archive ADD COLUMN IF NOT EXISTS failure_streak_count INTEGER")
                    f.execute("ALTER TABLE jobs_archive ADD COLUMN IF NOT EXISTS quarantined_at TIMESTAMPTZ")
                    f.execute("ALTER TABLE jobs_archive ADD COLUMN IF NOT EXISTS request_id TEXT")
                    f.execute("ALTER TABLE jobs_archive ADD COLUMN IF NOT EXISTS trace_id TEXT")
                    f.execute("ALTER TABLE jobs_archive ADD COLUMN IF NOT EXISTS failure_timeline JSONB")
                    f.execute("ALTER TABLE jobs_archive ADD COLUMN IF NOT EXISTS error_code TEXT")
                except required_migration_exceptions:
                    pass
            except required_migration_exceptions:
                # Best-effort; existing installs may restrict optional columns.
                pass
        _ensure_pg_archive_locators(_dsn)
        # Audit before creating the standalone archive indexes.
        slides_diagnostic: str | None = "ambiguous_generation_legacy_row"
        with psycopg.connect(_dsn) as audit_conn, audit_conn.cursor() as audit_cur:
            _mark_slides_audit_failure_pg(audit_cur)
            audit_cur.execute("SAVEPOINT slides_generation_audit")
            try:
                slides_diagnostic, _ = _audit_slides_generation_pg(audit_cur)
            except psycopg.Error:
                audit_cur.execute("ROLLBACK TO SAVEPOINT slides_generation_audit")
                audit_cur.execute("RELEASE SAVEPOINT slides_generation_audit")
            except _JOBS_PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS:
                audit_cur.execute("ROLLBACK TO SAVEPOINT slides_generation_audit")
                audit_cur.execute("RELEASE SAVEPOINT slides_generation_audit")
            else:
                audit_cur.execute("RELEASE SAVEPOINT slides_generation_audit")
        # Create hot-path indexes concurrently (outside transaction) when possible
        archive_batch_read_indexes_verified = False
        try:
            with psycopg.connect(_dsn, autocommit=True) as c2:
                with c2.cursor() as k:
                    _configure_pg_archive_migration_session(k, local=False)
                    _ensure_pg_archive_batch_read_indexes(k)
                    archive_batch_read_indexes_verified = True
                    for index_name, (unique, columns, predicate) in _SLIDES_ARCHIVE_INDEXES_PG.items():
                        if _pg_archive_index_matches(
                            k,
                            index_name=index_name,
                            unique=unique,
                            columns=columns,
                            predicate=predicate,
                        ):
                            continue
                        if index_name == "idx_jobs_archive_slides_scope":
                            k.execute("DROP INDEX CONCURRENTLY IF EXISTS idx_jobs_archive_slides_scope")
                        else:
                            k.execute("DROP INDEX CONCURRENTLY IF EXISTS idx_jobs_archive_uuid_unique")
                    # Ready vs scheduled scans
                    k.execute(
                        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_jobs_status_available_at ON jobs(status, available_at)"
                    )
                    # Composite unique for idempotency (NULLs are allowed and do not conflict)
                    k.execute(
                        "CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS idx_jobs_idempotent_unique ON jobs(domain, queue, job_type, idempotency_key)"
                    )
                    # Optional partial index to speed common hot-path queries
                    with contextlib.suppress(_JOBS_PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS):
                        k.execute("CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_jobs_hot ON jobs(domain, queue, job_type, priority, available_at, created_at) WHERE status IN ('queued','processing')")
                    with contextlib.suppress(_JOBS_PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS):
                        k.execute(POSTGRES_ARCHIVE_CURSOR_INDEX_SQL)
                    # Acquisition ordering index: priority ASC (lower number = higher priority),
                    # then available/created, then id; queued only. The ORDER BY in queries
                    # is explicit; this index simply supports that access pattern.
                    try:
                        k.execute(
                            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_jobs_acquire_order ON jobs (priority, COALESCE(available_at, created_at), id) WHERE status = 'queued'"
                        )
                    except _JOBS_PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS:
                        # Older PG versions or permission issues: non-fatal
                        pass
                    k.execute(
                        """
                        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_jobs_archive_slides_scope
                        ON jobs_archive(
                          domain, queue, job_type, idempotency_key, owner_user_id, archived_at DESC
                        )
                        WHERE idempotency_key IS NOT NULL
                        """
                    )
                    if slides_diagnostic != "duplicate_archive_uuid":
                        try:
                            k.execute(
                                """
                                CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS idx_jobs_archive_uuid_unique
                                ON jobs_archive(uuid)
                                WHERE uuid IS NOT NULL
                                """
                            )
                        except psycopg.Error as exc:
                            if getattr(exc, "sqlstate", None) != "23505":
                                raise
                            with contextlib.suppress(psycopg.Error):
                                k.execute("DROP INDEX CONCURRENTLY IF EXISTS idx_jobs_archive_uuid_unique")
                            _record_duplicate_archive_uuid_pg(_dsn)
        except (
            psycopg.Error,
            *_JOBS_PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS,
        ) as exc:
            if not archive_batch_read_indexes_verified:
                if isinstance(exc, RuntimeError):
                    raise
                raise RuntimeError(
                    "PostgreSQL Jobs archive batch-read index migration failed"
                ) from exc
            if isinstance(exc, psycopg.Error):
                # Optional standalone index/readiness setup must not break
                # generic Jobs after the required archive indexes are ready.
                with contextlib.suppress(psycopg.Error):
                    _record_slides_audit_failure_pg(_dsn)
            # Best-effort; not fatal
            pass
        # Ensure job_events exists (idempotent helper) for deployments created before inlined DDL
        with contextlib.suppress(_JOBS_PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS):
            ensure_job_events_pg(db_url)
        # Ensure job_counters exists for counters-enabled deployments
        with contextlib.suppress(_JOBS_PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS):
            ensure_job_counters_pg(db_url)
        # Optional: enable RLS on core tables when requested via env.
        try:
            import os as _os

            import psycopg  # noqa: F401

            if _is_truthy(_os.getenv("JOBS_PG_RLS_ENABLE", "")):
                with psycopg.connect(_dsn, autocommit=True) as _c_rls, _c_rls.cursor() as _p:
                    with contextlib.suppress(_JOBS_PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS):
                        _p.execute("ALTER TABLE jobs ENABLE ROW LEVEL SECURITY")
                    with contextlib.suppress(_JOBS_PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS):
                        _p.execute("ALTER TABLE job_events ENABLE ROW LEVEL SECURITY")
                    with contextlib.suppress(_JOBS_PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS):
                        _p.execute("ALTER TABLE job_counters ENABLE ROW LEVEL SECURITY")
                    with contextlib.suppress(_JOBS_PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS):
                        _p.execute("ALTER TABLE job_queue_controls ENABLE ROW LEVEL SECURITY")
                    with contextlib.suppress(_JOBS_PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS):
                        _p.execute("ALTER TABLE job_attachments ENABLE ROW LEVEL SECURITY")
                    with contextlib.suppress(_JOBS_PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS):
                        _p.execute("ALTER TABLE job_sla_policies ENABLE ROW LEVEL SECURITY")
                    with contextlib.suppress(_JOBS_PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS):
                        _p.execute("ALTER TABLE job_dependencies ENABLE ROW LEVEL SECURITY")
                    with contextlib.suppress(_JOBS_PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS):
                        _p.execute("ALTER TABLE jobs_archive ENABLE ROW LEVEL SECURITY")
                    with contextlib.suppress(_JOBS_PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS):
                        _p.execute(
                            "ALTER TABLE job_idempotency_receipts ENABLE ROW LEVEL SECURITY"
                        )
        except _JOBS_PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS:
            # Ignore in environments without permissions or when tables don't exist yet
            pass
    except _JOBS_PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS as e:
        # Attempt to create database if it doesn't exist, then retry
        msg = str(e)
        if "does not exist" in msg and "database" in msg:
            try:
                base = db_url.rsplit("/", 1)[0] + "/postgres"
                db_name = db_url.rsplit("/", 1)[1].split("?")[0]
                with psycopg.connect(base, autocommit=True) as conn2, conn2.cursor() as cur2:
                    cur2.execute("SELECT 1 FROM pg_database WHERE datname=%s", (db_name,))
                    if cur2.fetchone() is None:
                        cur2.execute(f"CREATE DATABASE {db_name}")
                # Retry DDL
                with psycopg.connect(_dsn) as conn3:
                    with conn3.cursor() as cur3:
                        _configure_pg_archive_migration_session(cur3)
                        cur3.execute(JOBS_POSTGRES_DDL)
                    conn3.commit()
            except _JOBS_PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS as e2:
                raise RuntimeError(f"Failed to ensure Jobs schema in Postgres: {e2}") from e2
        else:
            # Re-raise with context for other errors
            raise RuntimeError(f"Failed to ensure Jobs schema in Postgres: {e}") from e
    # Optionally enable RLS policies for domain scoping when requested
    try:
        import os as _os_rls
        if _is_truthy(_os_rls.getenv("JOBS_PG_RLS_ENABLE", "")):
            with contextlib.suppress(_JOBS_PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS):
                ensure_jobs_rls_policies_pg(db_url)
    except _JOBS_PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS:
        pass
    return db_url

def ensure_job_events_pg(db_url: str) -> None:
    """Ensure the job_events table and indexes exist in Postgres."""
    try:
        import psycopg
    except _JOBS_PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS:
        return
    from .pg_util import negotiate_pg_dsn
    _dsn = negotiate_pg_dsn(db_url)
    _rls_debug = _is_truthy(os.getenv("JOBS_PG_RLS_DEBUG", ""))
    try:
        with psycopg.connect(_dsn, autocommit=True) as conn, conn.cursor() as cur:
            cur.execute(
                """
                    CREATE TABLE IF NOT EXISTS job_events (
                      id BIGSERIAL PRIMARY KEY,
                      job_id INTEGER,
                      domain TEXT,
                      queue TEXT,
                      job_type TEXT,
                      event_type TEXT NOT NULL,
                      attrs_json JSONB,
                      owner_user_id TEXT,
                      request_id TEXT,
                      trace_id TEXT,
                      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    );
                    """
            )
            try:
                cur.execute("CREATE INDEX IF NOT EXISTS idx_job_events_id ON job_events(id)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_job_events_job_id ON job_events(job_id)")
            except _JOBS_PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS:
                pass
    except _JOBS_PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS:
        return


def ensure_jobs_rls_policies_pg(db_url: str) -> None:
    """Enable Postgres Row Level Security (RLS) for domain scoping.

    Policies restrict SELECT/UPDATE/DELETE to rows where jobs.domain is in
    current_setting('app.domain_allowlist', true), if set.
    """
    try:
        import psycopg  # type: ignore
        from psycopg import sql as _sql  # type: ignore
    except _JOBS_PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS:
        return
    import os
    import re as _re

    from .pg_util import negotiate_pg_dsn
    _dsn = negotiate_pg_dsn(db_url)
    debug = _is_truthy(os.getenv("JOBS_PG_RLS_DEBUG", ""))
    try:
        with psycopg.connect(_dsn, autocommit=True) as conn, conn.cursor() as cur:
            role = str(os.getenv("JOBS_PG_RLS_ROLE", "")).strip()
            if role and _re.match(r"^[A-Za-z0-9_]+$", role):
                try:
                    cur.execute("SELECT current_schema()")
                    schema_row = cur.fetchone()
                    schema_name = (schema_row[0] if schema_row else None) or "public"
                    cur.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (role,))
                    if not cur.fetchone():
                        cur.execute(
                            _sql.SQL("CREATE ROLE {} NOLOGIN").format(
                                _sql.Identifier(role),
                            )
                        )
                    try:
                        cur.execute("SELECT current_user")
                        user_row = cur.fetchone()
                        current_user = (user_row[0] if user_row else None) or None
                        if current_user and _re.match(r"^[A-Za-z0-9_]+$", str(current_user)):
                            cur.execute(
                                _sql.SQL("GRANT {} TO {}").format(
                                    _sql.Identifier(role),
                                    _sql.Identifier(str(current_user)),
                                )
                            )
                    except _JOBS_PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS:
                        pass
                    cur.execute(
                        _sql.SQL("GRANT USAGE ON SCHEMA {} TO {}").format(
                            _sql.Identifier(schema_name),
                            _sql.Identifier(role),
                        )
                    )
                    cur.execute(
                        _sql.SQL(
                            "GRANT SELECT, UPDATE, DELETE ON ALL TABLES IN SCHEMA {} TO {}"
                        ).format(
                            _sql.Identifier(schema_name),
                            _sql.Identifier(role),
                        )
                    )
                    cur.execute(
                        _sql.SQL("GRANT INSERT ON TABLE {}.job_counters TO {}").format(
                            _sql.Identifier(schema_name),
                            _sql.Identifier(role),
                        )
                    )
                    cur.execute(
                        _sql.SQL(
                            "GRANT INSERT ON TABLE "
                            "{}.slides_standalone_key_registry TO {}"
                        ).format(
                            _sql.Identifier(schema_name),
                            _sql.Identifier(role),
                        )
                    )
                    cur.execute(
                        psycopg.sql.SQL("GRANT INSERT ON {}.job_events TO {}").format(
                            psycopg.sql.Identifier(schema_name),
                            psycopg.sql.Identifier(role),
                        )
                    )
                    cur.execute(
                        psycopg.sql.SQL(
                            "GRANT USAGE, SELECT ON SEQUENCE {}.job_events_id_seq TO {}"
                        ).format(
                            psycopg.sql.Identifier(schema_name),
                            psycopg.sql.Identifier(role),
                        )
                    )
                    cur.execute(
                        _sql.SQL(
                            "GRANT INSERT ON TABLE "
                            "{}.job_idempotency_receipts TO {}"
                        ).format(
                            _sql.Identifier(schema_name),
                            _sql.Identifier(role),
                        )
                    )
                    cur.execute(
                        _sql.SQL(
                            "GRANT USAGE, SELECT ON SEQUENCE "
                            "{}.job_idempotency_receipts_receipt_id_seq TO {}"
                        ).format(
                            _sql.Identifier(schema_name),
                            _sql.Identifier(role),
                        )
                    )
                except _JOBS_PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS:
                    pass

            def _enable_rls(table: str) -> None:
                with contextlib.suppress(_JOBS_PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS):
                    cur.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
                with contextlib.suppress(_JOBS_PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS):
                    cur.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")

            # Enable and enforce RLS on all Jobs tables
            for _table in (
                "jobs",
                "job_events",
                "job_counters",
                "job_queue_controls",
                "job_sla_policies",
                "job_attachments",
                "job_dependencies",
                "job_idempotency_receipts",
            ):
                _enable_rls(_table)
            admin_expr = "COALESCE(NULLIF(current_setting('app.is_admin', true), ''), '') = 'true'"
            domain_expr = "NULLIF(current_setting('app.domain_allowlist', true), '')"
            owner_expr = "NULLIF(current_setting('app.owner_user_id', true), '')"
            domain_filter = f"({domain_expr} IS NULL OR domain = ANY(string_to_array({domain_expr}, ',')))"
            owner_filter = f"({owner_expr} IS NULL OR owner_user_id = {owner_expr})"

            cur.execute("DROP POLICY IF EXISTS jobs_domain_select ON jobs")
            cur.execute(
                f"""
                    CREATE POLICY jobs_domain_select ON jobs FOR SELECT
                    USING (
                      {admin_expr} OR (
                        {domain_filter}
                        AND {owner_filter}
                      )
                    )
                    """
            )
            cur.execute("DROP POLICY IF EXISTS jobs_domain_modify ON jobs")
            cur.execute(
                f"""
                    CREATE POLICY jobs_domain_modify ON jobs FOR ALL
                    USING (
                      {admin_expr} OR (
                        {domain_filter}
                        AND {owner_filter}
                      )
                    )
                    """
            )
            cur.execute(
                "DROP POLICY IF EXISTS job_idempotency_receipts_select "
                "ON job_idempotency_receipts"
            )
            cur.execute(
                f"""
                    CREATE POLICY job_idempotency_receipts_select
                    ON job_idempotency_receipts FOR SELECT
                    USING (
                      {admin_expr} OR (
                        {domain_filter}
                        AND {owner_filter}
                      )
                    )
                    """
            )
            cur.execute(
                "DROP POLICY IF EXISTS job_idempotency_receipts_modify "
                "ON job_idempotency_receipts"
            )
            cur.execute(
                f"""
                    CREATE POLICY job_idempotency_receipts_modify
                    ON job_idempotency_receipts FOR ALL
                    USING (
                      {admin_expr} OR (
                        {domain_filter}
                        AND {owner_filter}
                      )
                    )
                    WITH CHECK (
                      {admin_expr} OR (
                        {domain_filter}
                        AND {owner_filter}
                      )
                    )
                    """
            )
            # job_events policies (domain + owner, with admin bypass)
            try:
                cur.execute("DROP POLICY IF EXISTS job_events_select ON job_events")
                cur.execute(
                    f"""
                        CREATE POLICY job_events_select ON job_events FOR SELECT
                        USING (
                          {admin_expr} OR (
                            {domain_filter}
                            AND {owner_filter}
                          )
                        )
                        """
                )
                cur.execute("DROP POLICY IF EXISTS job_events_modify ON job_events")
                cur.execute(
                    f"""
                        CREATE POLICY job_events_modify ON job_events FOR ALL
                        USING (
                          {admin_expr} OR (
                            {domain_filter}
                            AND {owner_filter}
                          )
                        )
                        """
                )
            except _JOBS_PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS:
                pass
            # job_counters policies (domain only, with admin bypass)
            try:
                cur.execute("DROP POLICY IF EXISTS job_counters_select ON job_counters")
                cur.execute(
                    f"""
                        CREATE POLICY job_counters_select ON job_counters FOR SELECT
                        USING (
                          {admin_expr} OR (
                            {domain_filter}
                          )
                        )
                        """
                )
                cur.execute("DROP POLICY IF EXISTS job_counters_modify ON job_counters")
                cur.execute(
                    f"""
                        CREATE POLICY job_counters_modify ON job_counters FOR ALL
                        USING (
                          {admin_expr} OR (
                            {domain_filter}
                          )
                        )
                        """
                )
            except _JOBS_PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS:
                pass
            # job_queue_controls policies (domain only, with admin bypass)
            try:
                cur.execute("DROP POLICY IF EXISTS job_queue_controls_select ON job_queue_controls")
                cur.execute(
                    f"""
                        CREATE POLICY job_queue_controls_select ON job_queue_controls FOR SELECT
                        USING (
                          {admin_expr} OR (
                            {domain_filter}
                          )
                        )
                        """
                )
                cur.execute("DROP POLICY IF EXISTS job_queue_controls_modify ON job_queue_controls")
                cur.execute(
                    f"""
                        CREATE POLICY job_queue_controls_modify ON job_queue_controls FOR ALL
                        USING (
                          {admin_expr} OR (
                            {domain_filter}
                          )
                        )
                        """
                )
            except _JOBS_PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS:
                pass
            # job_attachments policies (join to jobs for domain/owner)
            try:
                cur.execute("DROP POLICY IF EXISTS job_attachments_select ON job_attachments")
                job_attachments_select_policy_template = """
                        CREATE POLICY job_attachments_select ON job_attachments FOR SELECT
                        USING (
                          {admin_expr} OR EXISTS (
                            SELECT 1 FROM jobs j
                            WHERE j.id = job_attachments.job_id
                              AND ({domain_expr} IS NULL OR j.domain = ANY(string_to_array({domain_expr}, ',')))
                              AND ({owner_expr} IS NULL OR j.owner_user_id = {owner_expr})
                          )
                        )
                        """
                job_attachments_select_policy_sql = job_attachments_select_policy_template.format_map(locals())  # nosec B608
                cur.execute(
                    job_attachments_select_policy_sql
                )
                cur.execute("DROP POLICY IF EXISTS job_attachments_modify ON job_attachments")
                job_attachments_modify_policy_template = """
                        CREATE POLICY job_attachments_modify ON job_attachments FOR ALL
                        USING (
                          {admin_expr} OR EXISTS (
                            SELECT 1 FROM jobs j
                            WHERE j.id = job_attachments.job_id
                              AND ({domain_expr} IS NULL OR j.domain = ANY(string_to_array({domain_expr}, ',')))
                              AND ({owner_expr} IS NULL OR j.owner_user_id = {owner_expr})
                          )
                        )
                        """
                job_attachments_modify_policy_sql = job_attachments_modify_policy_template.format_map(locals())  # nosec B608
                cur.execute(
                    job_attachments_modify_policy_sql
                )
            except _JOBS_PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS:
                pass
            # job_dependencies policies (join to jobs for domain/owner)
            try:
                cur.execute("DROP POLICY IF EXISTS job_dependencies_select ON job_dependencies")
                job_dependencies_select_policy_template = """
                        CREATE POLICY job_dependencies_select ON job_dependencies FOR SELECT
                        USING (
                          {admin_expr} OR EXISTS (
                            SELECT 1 FROM jobs j
                            WHERE j.uuid = job_dependencies.job_uuid
                              AND ({domain_expr} IS NULL OR j.domain = ANY(string_to_array({domain_expr}, ',')))
                              AND ({owner_expr} IS NULL OR j.owner_user_id = {owner_expr})
                          )
                        )
                        """
                job_dependencies_select_policy_sql = job_dependencies_select_policy_template.format_map(locals())  # nosec B608
                cur.execute(
                    job_dependencies_select_policy_sql
                )
                cur.execute("DROP POLICY IF EXISTS job_dependencies_modify ON job_dependencies")
                job_dependencies_modify_policy_template = """
                        CREATE POLICY job_dependencies_modify ON job_dependencies FOR ALL
                        USING (
                          {admin_expr} OR EXISTS (
                            SELECT 1 FROM jobs j
                            WHERE j.uuid = job_dependencies.job_uuid
                              AND ({domain_expr} IS NULL OR j.domain = ANY(string_to_array({domain_expr}, ',')))
                              AND ({owner_expr} IS NULL OR j.owner_user_id = {owner_expr})
                          )
                        )
                        """
                job_dependencies_modify_policy_sql = job_dependencies_modify_policy_template.format_map(locals())  # nosec B608
                cur.execute(
                    job_dependencies_modify_policy_sql
                )
            except _JOBS_PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS:
                pass
            # job_sla_policies policies (domain only)
            try:
                cur.execute("DROP POLICY IF EXISTS job_sla_policies_select ON job_sla_policies")
                cur.execute(
                    f"""
                        CREATE POLICY job_sla_policies_select ON job_sla_policies FOR SELECT
                        USING (
                          {admin_expr} OR (
                            {domain_filter}
                          )
                        )
                        """
                )
                cur.execute("DROP POLICY IF EXISTS job_sla_policies_modify ON job_sla_policies")
                cur.execute(
                    f"""
                        CREATE POLICY job_sla_policies_modify ON job_sla_policies FOR ALL
                        USING (
                          {admin_expr} OR (
                            {domain_filter}
                          )
                        )
                        """
                )
            except _JOBS_PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS:
                pass
            # jobs_archive policies (domain + owner, with admin bypass)
            try:
                cur.execute("DROP POLICY IF EXISTS jobs_archive_select ON jobs_archive")
                cur.execute(
                    f"""
                        CREATE POLICY jobs_archive_select ON jobs_archive FOR SELECT
                        USING (
                          {admin_expr} OR (
                            {domain_filter}
                            AND {owner_filter}
                          )
                        )
                        """
                )
                cur.execute("DROP POLICY IF EXISTS jobs_archive_modify ON jobs_archive")
                cur.execute(
                    f"""
                        CREATE POLICY jobs_archive_modify ON jobs_archive FOR ALL
                        USING (
                          {admin_expr} OR (
                            {domain_filter}
                            AND {owner_filter}
                          )
                        )
                        """
                )
            except _JOBS_PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS:
                pass
            if debug:
                try:
                    cur.execute(
                        """
                            SELECT tablename, polname
                            FROM pg_policies
                            WHERE schemaname = current_schema()
                              AND tablename IN (
                                'jobs','job_events','job_counters','job_queue_controls',
                                'job_attachments','job_sla_policies','job_dependencies','jobs_archive',
                                'job_idempotency_receipts'
                              )
                            ORDER BY tablename, polname
                            """
                    )
                    rows = cur.fetchall()
                    print(f"[jobs-rls-debug] policies={rows}")
                except _JOBS_PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS:
                    pass
    except _JOBS_PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS:
        if debug:
            with contextlib.suppress(_JOBS_PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS):
                print("[jobs-rls-debug] failed to apply RLS policies")
        return


def ensure_job_counters_pg(db_url: str) -> None:
    """Ensure per-queue counters table exists in PG."""
    try:
        import psycopg
    except _JOBS_PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS:
        return
    # Normalize DSN to include timeouts and libpq options, similar to other helpers
    try:
        from .pg_util import normalize_pg_dsn
        _dsn = normalize_pg_dsn(db_url)
    except _JOBS_PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS:
        _dsn = db_url
    try:
        with psycopg.connect(_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS job_counters (
                      domain TEXT NOT NULL,
                      queue TEXT NOT NULL,
                      job_type TEXT NOT NULL,
                      ready_count INTEGER DEFAULT 0,
                      scheduled_count INTEGER DEFAULT 0,
                      processing_count INTEGER DEFAULT 0,
                      quarantined_count INTEGER DEFAULT 0,
                      updated_at TIMESTAMPTZ DEFAULT NOW(),
                      PRIMARY KEY (domain, queue, job_type)
                    );
                    """
                )
                with contextlib.suppress(_JOBS_PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS):
                    cur.execute("CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_job_counters_domain_queue ON job_counters(domain, queue)")
    except _JOBS_PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS:
        return
