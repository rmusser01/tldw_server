"""
Jobs module migrations (PostgreSQL).

Provides SQL DDL to provision a `jobs` table compatible with the core JobManager
semantics. This module does not connect to Postgres directly; callers should
apply this DDL using their own connection or via a future Postgres JobManager.
"""

import contextlib
import os

from tldw_Server_API.app.core.testing import is_truthy as _is_truthy

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

_PLAYLIST_RLS_TABLES = (
    "playlist_preflights",
    "playlist_preflight_items",
    "playlist_materializations",
    "playlist_materialization_items",
    "media_ingest_runs",
    "media_ingest_run_items",
    "media_ingest_run_events",
)
_PLAYLIST_RLS_SEQUENCES = (
    "playlist_preflight_items_id_seq",
    "playlist_materialization_items_id_seq",
    "media_ingest_run_items_id_seq",
    "media_ingest_run_events_event_id_seq",
)
_JOBS_RLS_INSERT_TABLES = (
    "jobs",
    "job_events",
    "job_counters",
    "job_queue_controls",
    "job_sla_policies",
    "job_attachments",
    "job_dependencies",
    "jobs_archive",
)
_JOBS_RLS_SEQUENCES = (
    "jobs_id_seq",
    "job_events_id_seq",
    "job_attachments_id_seq",
)


class JobsRLSInstallationError(RuntimeError):
    """Raised when a security-critical Postgres RLS install step fails."""

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

-- Job dependencies (DAG edges)
CREATE TABLE IF NOT EXISTS job_dependencies (
  job_uuid TEXT NOT NULL,
  depends_on_job_uuid TEXT NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  PRIMARY KEY (job_uuid, depends_on_job_uuid)
);
CREATE INDEX IF NOT EXISTS idx_job_dependencies_job ON job_dependencies(job_uuid);
CREATE INDEX IF NOT EXISTS idx_job_dependencies_depends_on ON job_dependencies(depends_on_job_uuid);

-- Status-focused partial indexes to speed common counts and lookups
CREATE INDEX IF NOT EXISTS idx_jobs_status_queued ON jobs(domain, queue, job_type, priority, available_at, created_at) WHERE status='queued';
CREATE INDEX IF NOT EXISTS idx_jobs_status_processing ON jobs(domain, queue, job_type, leased_until) WHERE status='processing';

-- Owner-scoped immutable playlist inspection snapshots
CREATE TABLE IF NOT EXISTS playlist_preflights (
  preflight_id TEXT PRIMARY KEY,
  owner_user_id TEXT NOT NULL,
  status TEXT NOT NULL,
  source_url TEXT NOT NULL,
  source_kind TEXT NOT NULL,
  playlist_id TEXT,
  job_id BIGINT,
  summary_json JSONB,
  error_json JSONB,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  expires_at TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_playlist_preflights_owner_status
  ON playlist_preflights(owner_user_id, status);
CREATE INDEX IF NOT EXISTS idx_playlist_preflights_job ON playlist_preflights(job_id);
CREATE INDEX IF NOT EXISTS idx_playlist_preflights_expiry ON playlist_preflights(expires_at);

CREATE TABLE IF NOT EXISTS playlist_preflight_items (
  id BIGSERIAL PRIMARY KEY,
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
  selected_by_default BOOLEAN NOT NULL DEFAULT TRUE,
  display_metadata_json JSONB,
  UNIQUE (preflight_id, ordinal)
);
CREATE INDEX IF NOT EXISTS idx_playlist_preflight_items_owner_source
  ON playlist_preflight_items(owner_user_id, normalized_source_id);

-- Owner-bound queue records copied from a completed preflight
CREATE TABLE IF NOT EXISTS playlist_materializations (
  materialization_id TEXT PRIMARY KEY,
  preflight_id TEXT NOT NULL,
  owner_user_id TEXT NOT NULL,
  status TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  expires_at TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_playlist_materializations_owner_status
  ON playlist_materializations(owner_user_id, status);
CREATE INDEX IF NOT EXISTS idx_playlist_materializations_expiry
  ON playlist_materializations(expires_at);

CREATE TABLE IF NOT EXISTS playlist_materialization_items (
  id BIGSERIAL PRIMARY KEY,
  materialization_id TEXT NOT NULL,
  owner_user_id TEXT NOT NULL,
  occurrence_id TEXT NOT NULL,
  ordinal INTEGER NOT NULL CHECK (ordinal >= 1),
  source_url TEXT NOT NULL,
  normalized_source_id TEXT,
  source_kind TEXT NOT NULL,
  display_metadata_json JSONB,
  UNIQUE (materialization_id, ordinal),
  UNIQUE (materialization_id, occurrence_id)
);
CREATE INDEX IF NOT EXISTS idx_playlist_materialization_items_owner_source
  ON playlist_materialization_items(owner_user_id, normalized_source_id);
CREATE INDEX IF NOT EXISTS idx_playlist_materialization_items_occurrence
  ON playlist_materialization_items(occurrence_id);

-- Lightweight manifest connecting selected occurrences to Jobs
CREATE TABLE IF NOT EXISTS media_ingest_runs (
  run_id TEXT PRIMARY KEY,
  owner_user_id TEXT NOT NULL,
  client_request_id TEXT,
  request_fingerprint TEXT,
  initialization_token TEXT,
  initialization_expires_at TIMESTAMPTZ,
  status TEXT NOT NULL,
  collection_id BIGINT,
  processing_options_json JSONB,
  playlist_summaries_json JSONB,
  batch_ids_json JSONB,
  version INTEGER NOT NULL DEFAULT 1 CHECK (version >= 1),
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  expires_at TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_media_ingest_runs_owner_status
  ON media_ingest_runs(owner_user_id, status);
CREATE INDEX IF NOT EXISTS idx_media_ingest_runs_expiry ON media_ingest_runs(expires_at);

CREATE TABLE IF NOT EXISTS media_ingest_run_items (
  id BIGSERIAL PRIMARY KEY,
  run_id TEXT NOT NULL,
  owner_user_id TEXT NOT NULL,
  occurrence_id TEXT NOT NULL,
  ordinal INTEGER NOT NULL CHECK (ordinal >= 1),
  input_kind TEXT NOT NULL,
  materialization_id TEXT,
  source_url TEXT,
  normalized_source_id TEXT,
  source_kind TEXT,
  display_metadata_json JSONB,
  duplicate_policy TEXT CHECK (
    duplicate_policy IS NULL OR duplicate_policy IN ('skip','include_existing','update_metadata_only','overwrite')
  ),
  metadata_patch_json JSONB,
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
  job_id BIGINT,
  batch_id TEXT,
  attempt INTEGER NOT NULL DEFAULT 1 CHECK (attempt >= 1),
  idempotency_identity TEXT,
  submission_queue TEXT,
  staging_temp_dir TEXT,
  submission_lease_token TEXT,
  submission_lease_expires_at TIMESTAMPTZ,
  submission_lease_generation INTEGER NOT NULL DEFAULT 0 CHECK (submission_lease_generation >= 0),
  planned_collection_item_id BIGINT,
  progress_percent REAL CHECK (progress_percent IS NULL OR (progress_percent >= 0 AND progress_percent <= 100)),
  progress_message TEXT,
  retryable BOOLEAN NOT NULL DEFAULT FALSE,
  media_id BIGINT,
  error_code TEXT,
  error_message TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
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
  event_id BIGSERIAL PRIMARY KEY,
  run_id TEXT NOT NULL,
  owner_user_id TEXT NOT NULL,
  occurrence_id TEXT,
  job_id BIGINT,
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
  attrs_json JSONB,
  occurred_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
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

-- Composite uniqueness for idempotency scoped by domain/queue/job_type (NULL key allowed)
-- A unique index is created outside the DDL block using autocommit.
"""


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
            # JobManager construction can race in worker/test processes. Keep the
            # multi-statement schema bootstrap in one cluster-local critical section.
            cur.execute("SELECT pg_advisory_xact_lock(hashtext('tldw_jobs_schema_bootstrap'))")
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
        try:
            with psycopg.connect(_dsn, autocommit=True) as cfix, cfix.cursor() as f:
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
                f.execute("ALTER TABLE media_ingest_runs ADD COLUMN IF NOT EXISTS client_request_id TEXT")
                f.execute("ALTER TABLE media_ingest_runs ADD COLUMN IF NOT EXISTS request_fingerprint TEXT")
                f.execute("ALTER TABLE media_ingest_runs ADD COLUMN IF NOT EXISTS initialization_token TEXT")
                f.execute(
                    "ALTER TABLE media_ingest_runs "
                    "ADD COLUMN IF NOT EXISTS initialization_expires_at TIMESTAMPTZ"
                )
                f.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS idx_media_ingest_runs_owner_client_request "
                    "ON media_ingest_runs(owner_user_id, client_request_id)"
                )
                f.execute("ALTER TABLE media_ingest_run_items ADD COLUMN IF NOT EXISTS submission_queue TEXT")
                f.execute("ALTER TABLE media_ingest_run_items ADD COLUMN IF NOT EXISTS staging_temp_dir TEXT")
                f.execute("ALTER TABLE media_ingest_run_items ADD COLUMN IF NOT EXISTS submission_lease_token TEXT")
                f.execute(
                    "ALTER TABLE media_ingest_run_items "
                    "ADD COLUMN IF NOT EXISTS submission_lease_expires_at TIMESTAMPTZ"
                )
                f.execute(
                    "ALTER TABLE media_ingest_run_items "
                    "ADD COLUMN IF NOT EXISTS submission_lease_generation INTEGER NOT NULL DEFAULT 0"
                )
                # Forward-migrate archive table compressed columns (if table exists)
                try:
                    f.execute("ALTER TABLE jobs_archive ADD COLUMN IF NOT EXISTS payload_compressed BYTEA")
                    f.execute("ALTER TABLE jobs_archive ADD COLUMN IF NOT EXISTS result_compressed BYTEA")
                    f.execute("ALTER TABLE jobs_archive ADD COLUMN IF NOT EXISTS batch_group TEXT")
                except _JOBS_PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS:
                    pass
        except _JOBS_PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS:
            # Best-effort; if the DB already has these or lacks permissions, continue
            pass
        # Create hot-path indexes concurrently (outside transaction) when possible
        try:
            with psycopg.connect(_dsn, autocommit=True) as c2:
                with c2.cursor() as k:
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
                        k.execute(
                            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_jobs_hot ON jobs(domain, queue, job_type, priority, available_at, created_at) WHERE status IN ('queued','processing')"
                        )
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
        except _JOBS_PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS:
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
                        cur3.execute(JOBS_POSTGRES_DDL)
                    conn3.commit()
            except _JOBS_PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS as e2:
                raise RuntimeError(f"Failed to ensure Jobs schema in Postgres: {e2}") from e2
        else:
            # Re-raise with context for other errors
            raise RuntimeError(f"Failed to ensure Jobs schema in Postgres: {e}") from e
    # RLS is a security boundary when enabled; startup must fail if its
    # policies cannot be installed and verified.
    if _is_truthy(os.getenv("JOBS_PG_RLS_ENABLE", "")):
        ensure_jobs_rls_policies_pg(db_url)
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
    """Install Jobs domain policies and fail-closed playlist owner policies."""
    try:
        import psycopg  # type: ignore
    except _JOBS_PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS:
        return
    legacy_rls_exceptions = (*_JOBS_PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS, psycopg.Error)
    import os
    import re as _re

    from .pg_util import negotiate_pg_dsn

    _dsn = negotiate_pg_dsn(db_url)
    debug = _is_truthy(os.getenv("JOBS_PG_RLS_DEBUG", ""))
    try:
        with psycopg.connect(_dsn, autocommit=True) as conn, conn.cursor() as cur:
            role = str(os.getenv("JOBS_PG_RLS_ROLE", "")).strip()
            if role:
                if not _re.fullmatch(r"[A-Za-z0-9_]+", role):
                    raise JobsRLSInstallationError("JOBS_PG_RLS_ROLE must be a simple PostgreSQL identifier")
                try:
                    cur.execute("SELECT current_schema()")
                    schema_row = cur.fetchone()
                    schema_name = (schema_row[0] if schema_row else None) or "public"
                    if not _re.fullmatch(r"[A-Za-z0-9_]+", str(schema_name)):
                        raise JobsRLSInstallationError("current PostgreSQL schema is not a simple identifier")
                    cur.execute(
                        "SELECT rolcanlogin, rolsuper, rolbypassrls "
                        "FROM pg_roles WHERE rolname = %s",
                        (role,),
                    )
                    role_row = cur.fetchone()
                    if role_row and bool(role_row[0]):
                        raise JobsRLSInstallationError(
                            f"JOBS_PG_RLS_ROLE {role!r} must be a NOLOGIN group role"
                        )
                    if role_row and bool(role_row[1]):
                        raise JobsRLSInstallationError(
                            f"JOBS_PG_RLS_ROLE {role!r} must not be a superuser role"
                        )
                    if role_row and bool(role_row[2]):
                        raise JobsRLSInstallationError(
                            f"JOBS_PG_RLS_ROLE {role!r} must not have BYPASSRLS"
                        )
                    if not role_row:
                        cur.execute(f"CREATE ROLE {role} NOLOGIN")
                    cur.execute("SELECT current_user")
                    user_row = cur.fetchone()
                    current_user = (user_row[0] if user_row else None) or None
                    if current_user and _re.fullmatch(r"[A-Za-z0-9_]+", str(current_user)):
                        cur.execute(f"GRANT {role} TO {current_user}")
                    cur.execute(f"GRANT USAGE ON SCHEMA {schema_name} TO {role}")
                    cur.execute(f"GRANT SELECT, UPDATE, DELETE ON ALL TABLES IN SCHEMA {schema_name} TO {role}")
                    cur.execute(
                        f"GRANT INSERT ON "  # nosec B608
                        f"{', '.join((*_JOBS_RLS_INSERT_TABLES, *_PLAYLIST_RLS_TABLES))} TO {role}"
                    )
                    for sequence in (*_JOBS_RLS_SEQUENCES, *_PLAYLIST_RLS_SEQUENCES):
                        cur.execute(
                            f"GRANT USAGE, SELECT ON SEQUENCE {sequence} TO {role}"  # nosec B608
                        )
                except JobsRLSInstallationError:
                    raise
                except _JOBS_PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS as exc:
                    raise JobsRLSInstallationError("failed to configure the Postgres RLS role") from exc

            def _enable_rls(table: str) -> None:
                with contextlib.suppress(legacy_rls_exceptions):
                    cur.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
                with contextlib.suppress(legacy_rls_exceptions):
                    cur.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")

            # Preserve best-effort installation for legacy Jobs tables.
            for _table in (
                "jobs",
                "job_events",
                "job_counters",
                "job_queue_controls",
                "job_sla_policies",
                "job_attachments",
                "job_dependencies",
            ):
                _enable_rls(_table)

            # Playlist authority tables are security boundaries: installation is
            # successful only when both ENABLE and FORCE are confirmed by Postgres.
            try:
                for _table in _PLAYLIST_RLS_TABLES:
                    cur.execute(f"ALTER TABLE {_table} ENABLE ROW LEVEL SECURITY")  # nosec B608
                    cur.execute(f"ALTER TABLE {_table} FORCE ROW LEVEL SECURITY")  # nosec B608
                    cur.execute(
                        "SELECT relrowsecurity, relforcerowsecurity "
                        "FROM pg_class WHERE oid = to_regclass(%s)",
                        (_table,),
                    )
                    rls_row = cur.fetchone()
                    if not rls_row or not bool(rls_row[0]) or not bool(rls_row[1]):
                        raise JobsRLSInstallationError(
                            f"playlist RLS was not enabled and forced for {_table}"
                        )
            except JobsRLSInstallationError:
                raise
            except _JOBS_PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS as exc:
                raise JobsRLSInstallationError("playlist RLS installation failed") from exc
            admin_expr = "COALESCE(NULLIF(current_setting('app.is_admin', true), ''), '') = 'true'"
            domain_expr = "NULLIF(current_setting('app.domain_allowlist', true), '')"
            owner_expr = "NULLIF(current_setting('app.owner_user_id', true), '')"
            domain_filter = f"({domain_expr} IS NULL OR domain = ANY(string_to_array({domain_expr}, ',')))"
            owner_filter = f"({owner_expr} IS NULL OR owner_user_id = {owner_expr})"

            def _qualified_owner_filter(table: str) -> str:
                return f"({owner_expr} IS NOT NULL AND {table}.owner_user_id = {owner_expr})"

            playlist_policy_filters = {
                "playlist_preflights": (
                    f"{admin_expr} OR {_qualified_owner_filter('playlist_preflights')}"
                ),
                "playlist_preflight_items": (
                    f"{admin_expr} OR ("  # nosec B608
                    f"{_qualified_owner_filter('playlist_preflight_items')} AND EXISTS ("
                    "SELECT 1 FROM playlist_preflights parent "
                    "WHERE parent.preflight_id = playlist_preflight_items.preflight_id "
                    "AND parent.owner_user_id = playlist_preflight_items.owner_user_id "
                    f"AND {_qualified_owner_filter('parent')}))"
                ),
                "playlist_materializations": (
                    f"{admin_expr} OR ("  # nosec B608
                    f"{_qualified_owner_filter('playlist_materializations')} AND EXISTS ("
                    "SELECT 1 FROM playlist_preflights parent "
                    "WHERE parent.preflight_id = playlist_materializations.preflight_id "
                    "AND parent.owner_user_id = playlist_materializations.owner_user_id "
                    f"AND {_qualified_owner_filter('parent')}))"
                ),
                "playlist_materialization_items": (
                    f"{admin_expr} OR ("  # nosec B608
                    f"{_qualified_owner_filter('playlist_materialization_items')} AND EXISTS ("
                    "SELECT 1 FROM playlist_materializations parent "
                    "WHERE parent.materialization_id = playlist_materialization_items.materialization_id "
                    "AND parent.owner_user_id = playlist_materialization_items.owner_user_id "
                    f"AND {_qualified_owner_filter('parent')}))"
                ),
                "media_ingest_runs": (
                    f"{admin_expr} OR {_qualified_owner_filter('media_ingest_runs')}"
                ),
                "media_ingest_run_items": (
                    f"{admin_expr} OR ("  # nosec B608
                    f"{_qualified_owner_filter('media_ingest_run_items')} AND EXISTS ("
                    "SELECT 1 FROM media_ingest_runs parent "
                    "WHERE parent.run_id = media_ingest_run_items.run_id "
                    "AND parent.owner_user_id = media_ingest_run_items.owner_user_id "
                    f"AND {_qualified_owner_filter('parent')}))"
                ),
                "media_ingest_run_events": (
                    f"{admin_expr} OR ("  # nosec B608
                    f"{_qualified_owner_filter('media_ingest_run_events')} AND EXISTS ("
                    "SELECT 1 FROM media_ingest_runs parent "
                    "WHERE parent.run_id = media_ingest_run_events.run_id "
                    "AND parent.owner_user_id = media_ingest_run_events.owner_user_id "
                    f"AND {_qualified_owner_filter('parent')}))"
                ),
            }
            for table, policy_filter in playlist_policy_filters.items():
                cur.execute(f"DROP POLICY IF EXISTS {table}_owner_select ON {table}")  # nosec B608
                select_policy_sql = f"""
                    CREATE POLICY {table}_owner_select ON {table} FOR SELECT
                    USING ({policy_filter})
                    """  # nosec B608
                cur.execute(select_policy_sql)
                cur.execute(f"DROP POLICY IF EXISTS {table}_owner_modify ON {table}")  # nosec B608
                modify_policy_sql = f"""
                    CREATE POLICY {table}_owner_modify ON {table} FOR ALL
                    USING ({policy_filter})
                    WITH CHECK ({policy_filter})
                    """  # nosec B608
                cur.execute(modify_policy_sql)

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
                job_attachments_select_policy_sql = job_attachments_select_policy_template.format_map(
                    locals()
                )  # nosec B608
                cur.execute(job_attachments_select_policy_sql)
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
                job_attachments_modify_policy_sql = job_attachments_modify_policy_template.format_map(
                    locals()
                )  # nosec B608
                cur.execute(job_attachments_modify_policy_sql)
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
                job_dependencies_select_policy_sql = job_dependencies_select_policy_template.format_map(
                    locals()
                )  # nosec B608
                cur.execute(job_dependencies_select_policy_sql)
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
                job_dependencies_modify_policy_sql = job_dependencies_modify_policy_template.format_map(
                    locals()
                )  # nosec B608
                cur.execute(job_dependencies_modify_policy_sql)
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
                                'playlist_preflights','playlist_preflight_items',
                                'playlist_materializations','playlist_materialization_items',
                                'media_ingest_runs','media_ingest_run_items','media_ingest_run_events'
                              )
                            ORDER BY tablename, polname
                            """
                    )
                    rows = cur.fetchall()
                    print(f"[jobs-rls-debug] policies={rows}")
                except _JOBS_PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS:
                    pass
    except JobsRLSInstallationError:
        raise
    except _JOBS_PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS as exc:
        if debug:
            with contextlib.suppress(_JOBS_PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS):
                print("[jobs-rls-debug] failed to apply RLS policies")
        raise JobsRLSInstallationError("failed to apply Postgres Jobs RLS policies") from exc


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
                    cur.execute(
                        "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_job_counters_domain_queue ON job_counters(domain, queue)"
                    )
    except _JOBS_PG_MIGRATIONS_NONCRITICAL_EXCEPTIONS:
        return
