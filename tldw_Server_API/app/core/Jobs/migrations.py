"""
Jobs module migrations (SQLite-focused).

Provides a simple helper to ensure the `jobs` table exists in a given SQLite
database path. This scaffolds the future core JobManager backend.
"""

from __future__ import annotations

import contextlib
import sqlite3
from pathlib import Path

from loguru import logger

from tldw_Server_API.app.core.DB_Management.sqlite_policy import (
    configure_sqlite_connection,
)

_JOBS_PATH_EXCEPTIONS = (ImportError, OSError, RuntimeError, TypeError, ValueError)
_JOBS_DB_EXCEPTIONS = (sqlite3.Error, OSError, RuntimeError, TypeError, ValueError)

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
  created_at TEXT DEFAULT (DATETIME('now')),
  PRIMARY KEY (job_uuid, depends_on_job_uuid)
);
CREATE INDEX IF NOT EXISTS idx_job_dependencies_job ON job_dependencies(job_uuid);
CREATE INDEX IF NOT EXISTS idx_job_dependencies_depends_on ON job_dependencies(depends_on_job_uuid);
"""


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
            conn.executescript(JOBS_SQLITE_DDL)
            conn.commit()
            with contextlib.suppress(_JOBS_DB_EXCEPTIONS):
                conn.execute("ALTER TABLE jobs ADD COLUMN batch_group TEXT")
            with contextlib.suppress(_JOBS_DB_EXCEPTIONS):
                conn.execute("ALTER TABLE jobs_archive ADD COLUMN batch_group TEXT")
            conn.commit()
        try:
            logger.info(f"Ensured Jobs schema at {Path(db_path).resolve()}")
        except _JOBS_PATH_EXCEPTIONS:
            logger.info(f"Ensured Jobs schema at {db_path}")
    except _JOBS_DB_EXCEPTIONS as e:
        logger.warning("Failed to ensure Jobs schema ({})", type(e).__name__)
    return db_path
