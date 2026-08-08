"""
Jobs module migrations (SQLite-focused).

Provides a simple helper to ensure the `jobs` table exists in a given SQLite
database path. This scaffolds the future core JobManager backend.
"""

from __future__ import annotations

import contextlib
import os
import sqlite3
from pathlib import Path

from loguru import logger

from tldw_Server_API.app.core.DB_Management.sqlite_policy import (
    configure_sqlite_connection,
)

_JOBS_PATH_EXCEPTIONS = (ImportError, OSError, RuntimeError, TypeError, ValueError)
_JOBS_DB_EXCEPTIONS = (sqlite3.Error, OSError, RuntimeError, TypeError, ValueError)

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
  retry_count INTEGER,
  available_at TEXT,
  started_at TEXT,
  leased_until TEXT,
  lease_id TEXT,
  worker_id TEXT,
  acquired_at TEXT,
  error_message TEXT,
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


def ensure_jobs_tables(db_path: Path | None = None) -> Path:
    """Ensure the jobs table exists in the given SQLite database.

    Args:
        db_path: Optional path to the SQLite database; defaults to Databases/jobs.db

    Returns:
        Path to the database used
    """
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
            conn.commit()
            with contextlib.suppress(_JOBS_DB_EXCEPTIONS):
                conn.execute("ALTER TABLE jobs ADD COLUMN batch_group TEXT")
            with contextlib.suppress(_JOBS_DB_EXCEPTIONS):
                conn.execute("ALTER TABLE jobs_archive ADD COLUMN batch_group TEXT")
            with contextlib.suppress(_JOBS_DB_EXCEPTIONS):
                conn.execute(
                    "ALTER TABLE jobs_archive ADD COLUMN owner_user_id TEXT"
                )
            _ensure_sqlite_dependency_snapshot_columns(conn)
            conn.commit()
            _ensure_sqlite_archive_locators(conn)
            _ensure_sqlite_archive_batch_read_indexes(conn)
            archive_locator_verified = True
            with contextlib.suppress(_JOBS_DB_EXCEPTIONS):
                conn.execute(SQLITE_ARCHIVE_CURSOR_INDEX_SQL)
            conn.commit()
        try:
            logger.info(f"Ensured Jobs schema at {Path(db_path).resolve()}")
        except _JOBS_PATH_EXCEPTIONS:
            logger.info(f"Ensured Jobs schema at {db_path}")
    except _JOBS_DB_EXCEPTIONS as e:
        logger.warning("Failed to ensure Jobs schema ({})", type(e).__name__)
        if not archive_locator_verified:
            raise
    return db_path
