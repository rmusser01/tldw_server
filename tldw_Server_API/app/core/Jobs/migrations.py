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

-- Owner-scoped immutable playlist inspection snapshots
CREATE TABLE IF NOT EXISTS playlist_preflights (
  preflight_id TEXT NOT NULL PRIMARY KEY,
  owner_user_id TEXT NOT NULL,
  status TEXT NOT NULL,
  source_url TEXT NOT NULL,
  source_kind TEXT NOT NULL,
  playlist_id TEXT,
  job_id INTEGER,
  summary_json TEXT CHECK (summary_json IS NULL OR json_valid(summary_json)),
  error_json TEXT CHECK (error_json IS NULL OR json_valid(error_json)),
  created_at TEXT NOT NULL DEFAULT (DATETIME('now')),
  updated_at TEXT NOT NULL DEFAULT (DATETIME('now')),
  expires_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_playlist_preflights_owner_status
  ON playlist_preflights(owner_user_id, status);
CREATE INDEX IF NOT EXISTS idx_playlist_preflights_job ON playlist_preflights(job_id);
CREATE INDEX IF NOT EXISTS idx_playlist_preflights_expiry ON playlist_preflights(expires_at);

CREATE TABLE IF NOT EXISTS playlist_preflight_items (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  preflight_id TEXT NOT NULL,
  owner_user_id TEXT NOT NULL,
  occurrence_id TEXT NOT NULL UNIQUE,
  ordinal INTEGER NOT NULL CHECK (ordinal >= 1),
  occurrence_index_for_source INTEGER NOT NULL CHECK (occurrence_index_for_source >= 1),
  source_url TEXT,
  normalized_source_id TEXT,
  source_kind TEXT NOT NULL,
  availability TEXT NOT NULL,
  duplicate_status TEXT NOT NULL,
  duplicate_of_occurrence_id TEXT,
  selected_by_default INTEGER NOT NULL DEFAULT 1 CHECK (selected_by_default IN (0, 1)),
  display_metadata_json TEXT CHECK (display_metadata_json IS NULL OR json_valid(display_metadata_json)),
  UNIQUE (preflight_id, ordinal)
);
CREATE INDEX IF NOT EXISTS idx_playlist_preflight_items_owner_source
  ON playlist_preflight_items(owner_user_id, normalized_source_id);

-- Owner-bound queue records copied from a completed preflight
CREATE TABLE IF NOT EXISTS playlist_materializations (
  materialization_id TEXT NOT NULL PRIMARY KEY,
  preflight_id TEXT NOT NULL,
  owner_user_id TEXT NOT NULL,
  status TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (DATETIME('now')),
  updated_at TEXT NOT NULL DEFAULT (DATETIME('now')),
  expires_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_playlist_materializations_owner_status
  ON playlist_materializations(owner_user_id, status);
CREATE INDEX IF NOT EXISTS idx_playlist_materializations_expiry
  ON playlist_materializations(expires_at);

CREATE TABLE IF NOT EXISTS playlist_materialization_items (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  materialization_id TEXT NOT NULL,
  owner_user_id TEXT NOT NULL,
  occurrence_id TEXT NOT NULL,
  ordinal INTEGER NOT NULL CHECK (ordinal >= 1),
  source_url TEXT NOT NULL,
  normalized_source_id TEXT,
  source_kind TEXT NOT NULL,
  display_metadata_json TEXT CHECK (display_metadata_json IS NULL OR json_valid(display_metadata_json)),
  UNIQUE (materialization_id, ordinal),
  UNIQUE (materialization_id, occurrence_id)
);
CREATE INDEX IF NOT EXISTS idx_playlist_materialization_items_owner_source
  ON playlist_materialization_items(owner_user_id, normalized_source_id);
CREATE INDEX IF NOT EXISTS idx_playlist_materialization_items_occurrence
  ON playlist_materialization_items(occurrence_id);

-- Lightweight manifest connecting selected occurrences to Jobs
CREATE TABLE IF NOT EXISTS media_ingest_runs (
  run_id TEXT NOT NULL PRIMARY KEY,
  owner_user_id TEXT NOT NULL,
  client_request_id TEXT,
  request_fingerprint TEXT,
  initialization_token TEXT,
  initialization_expires_at TEXT,
  status TEXT NOT NULL,
  collection_id INTEGER,
  processing_options_json TEXT CHECK (processing_options_json IS NULL OR json_valid(processing_options_json)),
  playlist_summaries_json TEXT CHECK (playlist_summaries_json IS NULL OR json_valid(playlist_summaries_json)),
  batch_ids_json TEXT CHECK (batch_ids_json IS NULL OR json_valid(batch_ids_json)),
  version INTEGER NOT NULL DEFAULT 1 CHECK (version >= 1),
  created_at TEXT NOT NULL DEFAULT (DATETIME('now')),
  updated_at TEXT NOT NULL DEFAULT (DATETIME('now')),
  expires_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_media_ingest_runs_owner_status
  ON media_ingest_runs(owner_user_id, status);
CREATE INDEX IF NOT EXISTS idx_media_ingest_runs_expiry ON media_ingest_runs(expires_at);

CREATE TABLE IF NOT EXISTS media_ingest_run_items (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL,
  owner_user_id TEXT NOT NULL,
  occurrence_id TEXT NOT NULL,
  ordinal INTEGER NOT NULL CHECK (ordinal >= 1),
  input_kind TEXT NOT NULL,
  materialization_id TEXT,
  source_url TEXT,
  normalized_source_id TEXT,
  source_kind TEXT,
  display_metadata_json TEXT CHECK (display_metadata_json IS NULL OR json_valid(display_metadata_json)),
  duplicate_policy TEXT CHECK (
    duplicate_policy IS NULL OR duplicate_policy IN ('skip','include_existing','update_metadata_only','overwrite')
  ),
  metadata_patch_json TEXT CHECK (metadata_patch_json IS NULL OR json_valid(metadata_patch_json)),
  state TEXT NOT NULL CHECK (
    state IN (
      'staged','preparing','awaiting_upload','submit_pending','queued','running',
      'cancellation_requested','status_unavailable','terminal'
    )
  ),
  outcome TEXT CHECK (
    outcome IS NULL OR outcome IN (
      'completed','included_existing','metadata_updated','skipped_existing',
      'submit_failed','processing_failed','metadata_update_failed','cancelled'
    )
  ),
  job_id INTEGER,
  batch_id TEXT,
  attempt INTEGER NOT NULL DEFAULT 1 CHECK (attempt >= 1),
  idempotency_identity TEXT,
  submission_queue TEXT,
  staging_temp_dir TEXT,
  submission_lease_token TEXT,
  submission_lease_expires_at TEXT,
  submission_lease_generation INTEGER NOT NULL DEFAULT 0 CHECK (submission_lease_generation >= 0),
  planned_collection_item_id INTEGER,
  progress_percent REAL CHECK (progress_percent IS NULL OR (progress_percent >= 0 AND progress_percent <= 100)),
  progress_message TEXT,
  retryable INTEGER NOT NULL DEFAULT 0 CHECK (retryable IN (0, 1)),
  media_id INTEGER,
  error_code TEXT,
  error_message TEXT,
  created_at TEXT NOT NULL DEFAULT (DATETIME('now')),
  updated_at TEXT NOT NULL DEFAULT (DATETIME('now')),
  UNIQUE (run_id, ordinal),
  UNIQUE (run_id, occurrence_id),
  UNIQUE (run_id, occurrence_id, attempt),
  CHECK (
    (state = 'terminal' AND outcome IS NOT NULL)
    OR (state <> 'terminal' AND outcome IS NULL)
  )
);
CREATE INDEX IF NOT EXISTS idx_media_ingest_run_items_owner_state
  ON media_ingest_run_items(owner_user_id, state);
CREATE INDEX IF NOT EXISTS idx_media_ingest_run_items_source
  ON media_ingest_run_items(normalized_source_id);
CREATE INDEX IF NOT EXISTS idx_media_ingest_run_items_job ON media_ingest_run_items(job_id);

CREATE TABLE IF NOT EXISTS media_ingest_run_events (
  event_id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL,
  owner_user_id TEXT NOT NULL,
  occurrence_id TEXT,
  job_id INTEGER,
  batch_id TEXT,
  event_type TEXT NOT NULL,
  state TEXT CHECK (
    state IS NULL OR state IN (
      'staged','preparing','awaiting_upload','submit_pending','queued','running',
      'cancellation_requested','status_unavailable','terminal'
    )
  ),
  outcome TEXT CHECK (
    outcome IS NULL OR outcome IN (
      'completed','included_existing','metadata_updated','skipped_existing',
      'submit_failed','processing_failed','metadata_update_failed','cancelled'
    )
  ),
  progress_percent REAL CHECK (progress_percent IS NULL OR (progress_percent >= 0 AND progress_percent <= 100)),
  progress_message TEXT,
  attrs_json TEXT CHECK (attrs_json IS NULL OR json_valid(attrs_json)),
  occurred_at TEXT NOT NULL DEFAULT (DATETIME('now')),
  CHECK (
    (state IS NULL AND outcome IS NULL)
    OR (state IS NOT NULL AND state = 'terminal' AND outcome IS NOT NULL)
    OR (state IS NOT NULL AND state <> 'terminal' AND outcome IS NULL)
  )
);
CREATE INDEX IF NOT EXISTS idx_media_ingest_run_events_owner_run_event
  ON media_ingest_run_events(owner_user_id, run_id, event_id);
CREATE INDEX IF NOT EXISTS idx_media_ingest_run_events_occurrence
  ON media_ingest_run_events(run_id, occurrence_id, event_id);
CREATE INDEX IF NOT EXISTS idx_media_ingest_run_events_job ON media_ingest_run_events(job_id);
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
            with contextlib.suppress(_JOBS_DB_EXCEPTIONS):
                conn.execute("ALTER TABLE media_ingest_runs ADD COLUMN client_request_id TEXT")
            with contextlib.suppress(_JOBS_DB_EXCEPTIONS):
                conn.execute("ALTER TABLE media_ingest_runs ADD COLUMN request_fingerprint TEXT")
            with contextlib.suppress(_JOBS_DB_EXCEPTIONS):
                conn.execute("ALTER TABLE media_ingest_runs ADD COLUMN initialization_token TEXT")
            with contextlib.suppress(_JOBS_DB_EXCEPTIONS):
                conn.execute("ALTER TABLE media_ingest_runs ADD COLUMN initialization_expires_at TEXT")
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_media_ingest_runs_owner_client_request "
                "ON media_ingest_runs(owner_user_id, client_request_id)"
            )
            with contextlib.suppress(_JOBS_DB_EXCEPTIONS):
                conn.execute("ALTER TABLE media_ingest_run_items ADD COLUMN submission_queue TEXT")
            with contextlib.suppress(_JOBS_DB_EXCEPTIONS):
                conn.execute("ALTER TABLE media_ingest_run_items ADD COLUMN staging_temp_dir TEXT")
            with contextlib.suppress(_JOBS_DB_EXCEPTIONS):
                conn.execute("ALTER TABLE media_ingest_run_items ADD COLUMN submission_lease_token TEXT")
            with contextlib.suppress(_JOBS_DB_EXCEPTIONS):
                conn.execute("ALTER TABLE media_ingest_run_items ADD COLUMN submission_lease_expires_at TEXT")
            with contextlib.suppress(_JOBS_DB_EXCEPTIONS):
                conn.execute(
                    "ALTER TABLE media_ingest_run_items "
                    "ADD COLUMN submission_lease_generation INTEGER NOT NULL DEFAULT 0"
                )
            conn.commit()
        try:
            logger.info(f"Ensured Jobs schema at {Path(db_path).resolve()}")
        except _JOBS_PATH_EXCEPTIONS:
            logger.info(f"Ensured Jobs schema at {db_path}")
    except _JOBS_DB_EXCEPTIONS as e:
        logger.warning("Failed to ensure Jobs schema ({})", type(e).__name__)
    return db_path
