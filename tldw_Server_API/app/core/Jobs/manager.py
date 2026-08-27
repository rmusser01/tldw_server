from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import secrets
import sqlite3
import time
import uuid as _uuid
from contextvars import ContextVar
from dataclasses import replace
from datetime import datetime, timedelta
from datetime import timezone as _tz
from pathlib import Path
from typing import Any, ClassVar

from loguru import logger

from tldw_Server_API.app.core.DB_Management.jobs_sql_fragments import (
    fetch_slides_archive_collision_rows,
    job_event_filter_fragment,
)
from tldw_Server_API.app.core.DB_Management.sqlite_policy import (
    configure_sqlite_connection,
)
from tldw_Server_API.app.core.exceptions import BadRequestError
from tldw_Server_API.app.core.Security.crypto import (
    decrypt_json_blob,
    decrypt_json_blob_with_key,
    encrypt_json_blob,
    encrypt_json_blob_with_key,
)
from tldw_Server_API.app.core.testing import (
    is_test_mode as _is_test_mode,
)
from tldw_Server_API.app.core.testing import (
    is_truthy as _shared_is_truthy,
)

from .audit_bridge import submit_job_audit_event
from .event_stream import emit_job_event, observe_job_event
from .fair_share import FairShareScheduler
from .metrics import (
    ensure_jobs_metrics_registered,
    increment_cancelled,
    increment_completed,
    increment_created,
    increment_failures,
    increment_json_truncated,
    increment_retries,
    increment_sla_breach,
    observe_duration,
    observe_queue_latency,
    set_queue_flag,
    set_queue_gauges,
)
from .migrations import (
    SLIDES_ARCHIVE_EXACT_FIELDS,
    SQLITE_ARCHIVE_CURSOR_OUTPUT_SQL,
    SQLITE_ARCHIVE_CURSOR_TIME_SQL,
    _ensure_sqlite_archive_batch_read_indexes,
    ensure_jobs_tables,
    normalize_slides_archive_projection,
    slides_archive_indexes_ready_sqlite,
    slides_archive_projection_ready_sqlite,
)
from .operations.contracts import (
    AcquireJobCommand,
    AdmissionRejectionReason,
    AdmissionResult,
    BatchRenewLeaseItem,
    BatchRenewLeasesCommand,
    CreateJobCommand,
    IdempotentOperationAdmission,
    IdempotentOperationCommand,
    IdempotentOperationDisposition,
    IdempotentOperationUnavailableError,
    OperationOutcome,
    ReleaseJobCommand,
    RenewLeaseCommand,
    TerminalOperationResultPatchCommand,
    TerminalOperationResultPatchOutcome,
)
from .operations.postgres import acquire_job as _postgres_acquire_job
from .operations.postgres import admit_idempotent_operation as _postgres_admit_idempotent_operation
from .operations.postgres import create_job_admission as _postgres_create_job_admission
from .operations.postgres import (
    get_job_or_archived_by_uuid as _postgres_get_job_or_archived_by_uuid,
)
from .operations.postgres import (
    patch_terminal_operation_result as _postgres_patch_terminal_operation_result,
)
from .operations.postgres import release_job as _postgres_release_job
from .operations.postgres import renew_lease as _postgres_renew_lease
from .operations.postgres import renew_leases_batch as _postgres_renew_leases_batch
from .operations.postgres import replay_idempotent_operation as _postgres_replay_idempotent_operation
from .operations.sqlite import acquire_job as _sqlite_acquire_job
from .operations.sqlite import admit_idempotent_operation as _sqlite_admit_idempotent_operation
from .operations.sqlite import create_job_admission as _sqlite_create_job_admission
from .operations.sqlite import (
    get_job_or_archived_by_uuid as _sqlite_get_job_or_archived_by_uuid,
)
from .operations.sqlite import (
    patch_terminal_operation_result as _sqlite_patch_terminal_operation_result,
)
from .operations.sqlite import release_job as _sqlite_release_job
from .operations.sqlite import renew_lease as _sqlite_renew_lease
from .operations.sqlite import renew_leases_batch as _sqlite_renew_leases_batch
from .operations.sqlite import replay_idempotent_operation as _sqlite_replay_idempotent_operation
from .pg_migrations import (
    POSTGRES_ARCHIVE_CURSOR_TIME_SQL,
    ensure_job_counters_pg,
    ensure_jobs_tables_pg,
    slides_archive_indexes_ready_pg,
    slides_archive_projection_ready_pg,
)
from .tracing import job_span

try:
    import psycopg  # type: ignore

    _PG_ERRORS: tuple[type[BaseException], ...] = (psycopg.Error,)
except ImportError:
    _PG_ERRORS = ()

_JOB_NONCRITICAL_EXCEPTIONS: tuple[type[BaseException], ...] = (
    AttributeError,
    BadRequestError,
    ConnectionError,
    ImportError,
    IndexError,
    KeyError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    UnicodeDecodeError,
    ValueError,
    json.JSONDecodeError,
    sqlite3.Error,
    *_PG_ERRORS,
)

_LEASE_EXPIRED_ERROR_CODE = "lease_expired"
_LEASE_EXPIRED_ERROR_MESSAGE = "Job lease expired; retry budget exhausted"
_DEFAULT_MAX_RETRIES = 3
_EXPIRED_RECOVERY_BATCH_DEFAULT = 100
_EXPIRED_RECOVERY_BATCH_MAX = 1000
_PRUNE_BATCH_SIZE = 1000


class JobPayloadDecryptionError(RuntimeError):
    """Raised when an encrypted Jobs value cannot be safely decoded."""

    def __init__(self, field_name: str) -> None:
        super().__init__(f"Encrypted job {field_name} could not be decrypted")
        self.field_name = field_name


_SLIDES_GENERATION_DOMAIN = "slides"
_SLIDES_GENERATION_QUEUE = "default"
_SLIDES_GENERATION_JOB_TYPE = "presentation.generate"
_SLIDES_GENERATION_CORRELATION_LOCK_PARTS = (
    _SLIDES_GENERATION_DOMAIN,
    _SLIDES_GENERATION_QUEUE,
    _SLIDES_GENERATION_JOB_TYPE,
    "correlation",
)


class SlidesGenerationJobsUnavailableError(ValueError):
    """Raised when standalone generation correlation cannot be trusted."""


def _is_slides_generation_scope(domain: object, queue: object, job_type: object) -> bool:
    return (
        domain == _SLIDES_GENERATION_DOMAIN
        and queue == _SLIDES_GENERATION_QUEUE
        and job_type == _SLIDES_GENERATION_JOB_TYPE
    )


def _require_aware_utc(value: datetime | None, *, field_name: str) -> datetime:
    if value is None:
        value = datetime.now(tz=_tz.utc)
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{field_name} must be an aware UTC timestamp")
    return value


def _sqlite_utc(value: datetime) -> str:
    return value.astimezone(_tz.utc).isoformat(timespec="microseconds")


# Module-level fair-share scheduler instance (lazy singleton)
_fair_share: FairShareScheduler | None = None
_fair_share_limits: tuple[int, int] | None = None


def _safe_increment_created_metric(*, domain: str, queue: str, job_type: str) -> None:
    """Keep job creation non-fatal while surfacing metric update failures."""
    try:
        increment_created({"domain": domain, "queue": queue, "job_type": job_type})
    except _JOB_NONCRITICAL_EXCEPTIONS as exc:
        logger.warning(
            "Non-critical jobs created metric update failed for {}:{}:{}: {}",
            domain,
            queue,
            job_type,
            exc,
        )


def _close_connection_nonfatal(conn: Any, *, operation: str) -> None:
    """Close after a transaction without masking committed post-commit work."""
    try:
        conn.close()
    except _JOB_NONCRITICAL_EXCEPTIONS as exc:
        logger.warning(
            "Non-critical Jobs connection close failed after {}: {}",
            operation,
            type(exc).__name__,
        )


def _commit_postgres_transaction(conn: Any, *, operation: str) -> None:
    """Commit a PostgreSQL mutation, rejecting already-aborted transactions."""

    from psycopg.pq import TransactionStatus

    if conn.info.transaction_status == TransactionStatus.INERROR:
        conn.rollback()
        raise RuntimeError(
            f"Cannot commit PostgreSQL {operation}: transaction is aborted"
        )
    conn.commit()


def _run_post_commit_side_effects(
    side_effects: list[tuple[Any, tuple[Any, ...], dict[str, Any]]],
) -> None:
    """Run non-critical observers only after the owning transaction commits."""

    for callback, args, kwargs in side_effects:
        with contextlib.suppress(_JOB_NONCRITICAL_EXCEPTIONS):
            callback(*args, **kwargs)


def _record_job_span(
    operation: str,
    job: dict[str, Any],
    attrs: dict[str, Any] | None = None,
) -> None:
    """Finish a terminal tracing span after durable state is visible."""

    with job_span(operation, job=job, attrs=attrs):
        pass


def _insert_lifecycle_event(
    executor: Any,
    *,
    backend: str,
    event_type: str,
    job: dict[str, Any],
    attrs: dict[str, Any] | None = None,
) -> None:
    """Append a durable lifecycle event inside the caller's transaction."""

    values = (
        int(job["id"]),
        job.get("domain"),
        job.get("queue"),
        job.get("job_type"),
        event_type,
        json.dumps(attrs or {}),
        job.get("owner_user_id"),
        job.get("request_id"),
        job.get("trace_id"),
    )
    if backend == "postgres":
        executor.execute(
            (
                "INSERT INTO job_events(job_id, domain, queue, job_type, event_type, "
                "attrs_json, owner_user_id, request_id, trace_id, created_at) "
                "VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, NOW())"
            ),
            values,
        )
        return
    executor.execute(
        (
            "INSERT INTO job_events(job_id, domain, queue, job_type, event_type, "
            "attrs_json, owner_user_id, request_id, trace_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, DATETIME('now'))"
        ),
        values,
    )


def _queue_lifecycle_event_observer(
    side_effects: list[tuple[Any, tuple[Any, ...], dict[str, Any]]],
    *,
    event_type: str,
    job: dict[str, Any],
    attrs: dict[str, Any] | None = None,
) -> None:
    """Schedule exactly one non-durable observer path after commit."""

    side_effects.append(
        (observe_job_event, (event_type,), {"job": job, "attrs": attrs})
    )


def _reconcile_lifecycle_counter_row(
    executor: Any,
    *,
    backend: str,
    domain: Any,
    queue: Any,
    job_type: Any,
) -> None:
    """Rebuild one missing counter row from transaction-local job state."""

    params = (domain, queue, job_type)
    if backend == "postgres":
        executor.execute(
            (
                "SELECT "
                "COUNT(*) FILTER (WHERE status='queued' AND available_at IS NULL) AS ready_count, "
                "COUNT(*) FILTER (WHERE status='queued' AND available_at IS NOT NULL) AS scheduled_count, "
                "COUNT(*) FILTER (WHERE status='processing') AS processing_count, "
                "COUNT(*) FILTER (WHERE status='quarantined') AS quarantined_count "
                "FROM jobs WHERE domain=%s AND queue=%s AND job_type=%s"
            ),
            params,
        )
        row = executor.fetchone()
        if isinstance(row, dict):
            counts = tuple(
                int(row.get(name) or 0)
                for name in (
                    "ready_count",
                    "scheduled_count",
                    "processing_count",
                    "quarantined_count",
                )
            )
        else:
            counts = tuple(int(value or 0) for value in (row or (0, 0, 0, 0)))
        executor.execute(
            (
                "INSERT INTO job_counters(domain,queue,job_type,ready_count,scheduled_count,processing_count,quarantined_count) "
                "VALUES(%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (domain,queue,job_type) DO UPDATE SET "
                "ready_count=EXCLUDED.ready_count, scheduled_count=EXCLUDED.scheduled_count, "
                "processing_count=EXCLUDED.processing_count, quarantined_count=EXCLUDED.quarantined_count, "
                "updated_at=NOW()"
            ),
            (*params, *counts),
        )
        return

    row = executor.execute(
        (
            "SELECT "
            "COALESCE(SUM(CASE WHEN status='queued' AND available_at IS NULL THEN 1 ELSE 0 END),0), "
            "COALESCE(SUM(CASE WHEN status='queued' AND available_at IS NOT NULL THEN 1 ELSE 0 END),0), "
            "COALESCE(SUM(CASE WHEN status='processing' THEN 1 ELSE 0 END),0), "
            "COALESCE(SUM(CASE WHEN status='quarantined' THEN 1 ELSE 0 END),0) "
            "FROM jobs WHERE domain=? AND queue=? AND job_type=?"
        ),
        params,
    ).fetchone()
    counts = tuple(int(value or 0) for value in (row or (0, 0, 0, 0)))
    executor.execute(
        (
            "INSERT INTO job_counters(domain,queue,job_type,ready_count,scheduled_count,processing_count,quarantined_count) "
            "VALUES(?,?,?,?,?,?,?) ON CONFLICT(domain,queue,job_type) DO UPDATE SET "
            "ready_count=excluded.ready_count, scheduled_count=excluded.scheduled_count, "
            "processing_count=excluded.processing_count, quarantined_count=excluded.quarantined_count, "
            "updated_at=DATETIME('now')"
        ),
        (*params, *counts),
    )


def _log_optional_sla_persistence_failure(job_id: int, error_type: str) -> None:
    """Report a recovered optional SLA write failure after the core commit."""

    logger.warning(
        "Optional completion SLA persistence failed for job {}: {}",
        job_id,
        error_type,
    )


def _fair_share_enabled() -> bool:
    """Return whether fair-share admission/priority logic is explicitly enabled."""
    return any(
        os.getenv(name) is not None
        for name in ("JOBS_MAX_PER_USER", "JOBS_MAX_PER_ORG")
    )


def _get_fair_share_limits() -> tuple[int, int]:
    """Return the effective env-backed fair-share limits."""
    return (
        int(os.getenv("JOBS_MAX_PER_USER", "5") or "5"),
        int(os.getenv("JOBS_MAX_PER_ORG", "20") or "20"),
    )


def _get_fair_share() -> FairShareScheduler:
    """Return the module-level FairShareScheduler, refreshing when limits change."""
    global _fair_share
    global _fair_share_limits

    limits = _get_fair_share_limits()
    if _fair_share is None or _fair_share_limits != limits:
        _fair_share = FairShareScheduler(
            max_per_user=limits[0],
            max_per_org=limits[1],
        )
        _fair_share_limits = limits
    return _fair_share


def _parse_dt(v: Any) -> datetime | None:
    """Parse a datetime from common storage formats."""
    if v is None:
        return None
    if isinstance(v, datetime):
        return v
    try:
        # Accept ISO8601 or SQLite default format
        s = str(v).replace("Z", "+00:00")
        # Try fromisoformat
        return datetime.fromisoformat(s)
    except _JOB_NONCRITICAL_EXCEPTIONS:
        return None


def _as_utc_datetime(value: Any) -> datetime | None:
    """Return a parsed datetime normalized to timezone-aware UTC."""

    parsed = _parse_dt(value)
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=_tz.utc)
    return parsed.astimezone(_tz.utc)


class JobManager:
    """DB-backed Job Manager with leasing, retries, and cancellation.

    Supports SQLite by default and PostgreSQL when `JOBS_DB_URL` (or `db_url`)
    is provided with a Postgres DSN. Provides helpers to create, list, acquire,
    renew, complete, fail, and cancel jobs in a generic, domain-agnostic way.

    Notes on lease enforcement:
    - Methods that acknowledge or extend work (renew/complete/fail) accept
      optional `worker_id` and `lease_id` parameters. If the environment
      variable `JOBS_ENFORCE_LEASE_ACK` is set to a truthy value, these values
      must match the current job lease or the operation is rejected.
    - Enforcement is enabled by default (unless disabled via
      `JOBS_DISABLE_LEASE_ENFORCEMENT`).
    """

    class Clock:
        def __init__(self):
            try:
                _env = os.getenv("JOBS_TEST_NOW_EPOCH")
                self._fixed_epoch = float(_env) if _env else None
            except _JOB_NONCRITICAL_EXCEPTIONS:
                self._fixed_epoch = None

        def now_utc(self) -> datetime:
            if self._fixed_epoch is not None:
                return datetime.fromtimestamp(self._fixed_epoch, tz=_tz.utc)
            return datetime.now(tz=_tz.utc)

    # In-process debounce map for gauge updates
    _GAUGE_LAST_TS: dict[tuple[str, str, str | None], float] = {}
    # RLS context (per-task via contextvars). Defaults are non-admin, unset filters.
    _RLS_IS_ADMIN: ContextVar[bool] = ContextVar("jobs_rls_is_admin", default=False)
    _RLS_DOMAIN_ALLOWLIST: ContextVar[str | None] = ContextVar("jobs_rls_domain_allowlist", default=None)
    _RLS_OWNER_USER_ID: ContextVar[str | None] = ContextVar("jobs_rls_owner_user_id", default=None)

    # Test-mode only: remember last acquired job per (domain,queue) to stabilize duplicate acquires
    _LAST_ACQUIRED_TEST: dict[tuple[str, str], dict[str, Any]] = {}

    @staticmethod
    def _is_truthy(val: str | None) -> bool:
        return _shared_is_truthy(val)

    @staticmethod
    def _expired_recovery_batch_size() -> int:
        """Return the bounded expired-lease maintenance batch size."""

        try:
            configured = int(
                os.getenv(
                    "JOBS_EXPIRED_RECOVERY_BATCH_SIZE",
                    str(_EXPIRED_RECOVERY_BATCH_DEFAULT),
                )
            )
        except (TypeError, ValueError):
            configured = _EXPIRED_RECOVERY_BATCH_DEFAULT
        return max(1, min(configured, _EXPIRED_RECOVERY_BATCH_MAX))

    def __init__(
        self,
        db_path: Path | None = None,
        *,
        backend: str | None = None,
        db_url: str | None = None,
        clock: JobManager.Clock | None = None,
        enforce_leases: bool | None = None,
    ):
        """Initialize JobManager.

        Currently supports SQLite. A future path will add Postgres support via db_url.
        """
        # Determine backend from explicit arg or env URL
        if backend is None:
            env_url = os.getenv("JOBS_DB_URL", "")
            if (db_url and str(db_url).startswith("postgres")) or env_url.startswith("postgres"):
                self.backend = "postgres"
                self.db_url = db_url or env_url
            else:
                self.backend = "sqlite"
                self.db_url = db_url
        else:
            self.backend = backend.lower()
            self.db_url = db_url
        # Time provider
        self._clock: JobManager.Clock = clock or JobManager.Clock()
        # Ensure schema for selected backend
        if self.backend == "postgres":
            if not (self.db_url and str(self.db_url).startswith("postgres")):
                raise ValueError(  # noqa: TRY003
                    "Postgres backend selected but no valid db_url provided; set JOBS_DB_URL or pass db_url"
                )
            # Normalize DSN and negotiate options for server compatibility
            try:
                from .pg_util import negotiate_pg_dsn

                self.db_url = negotiate_pg_dsn(self.db_url)
            except _JOB_NONCRITICAL_EXCEPTIONS:
                pass
            skip_schema = JobManager._is_truthy(os.getenv("JOBS_PG_SKIP_SCHEMA_INIT"))
            if not skip_schema:
                ensure_jobs_tables_pg(self.db_url)
                with contextlib.suppress(_JOB_NONCRITICAL_EXCEPTIONS):
                    ensure_job_counters_pg(self.db_url)
            self.db_path = Path(":memory:")  # unused
        else:
            # Prefer explicit db_path, then env override for tests (JOBS_DB_PATH), otherwise default
            if db_path is not None:
                self.db_path = ensure_jobs_tables(db_path)
            else:
                env_db_path = os.getenv("JOBS_DB_PATH")
                if env_db_path:
                    self.db_path = ensure_jobs_tables(Path(env_db_path))
                else:
                    self.db_path = ensure_jobs_tables(db_path)
        self._conn = None  # Lazily opened per operation

        self._enforce_override = enforce_leases
        with contextlib.suppress(_JOB_NONCRITICAL_EXCEPTIONS):
            ensure_jobs_metrics_registered()

    # Standard queues across domains
    STANDARD_QUEUES = ("default", "high", "low")
    DOMAIN_ALLOWED_QUEUES: ClassVar[dict[str, tuple[str, ...]]] = {
        "llamacpp": ("acquisition",),
        "reading": ("reading-digest",),
        "vn_assets": ("generation",),
        "persona_visuals": ("generation",),
        "sharing": ("workspace-clone",),
        "writing": ("writing-review", "writing-ai"),
        "scheduled_tasks": ("scheduled-tasks",),
    }

    # --- Shutdown/acquisition gate (process-wide) ---
    _ACQUIRE_GATE_ENABLED: bool = False

    @classmethod
    def set_acquire_gate(cls, enabled: bool) -> None:
        """Globally gate new acquisitions during graceful shutdown."""
        cls._ACQUIRE_GATE_ENABLED = bool(enabled)

    def _count_active_jobs_for_user(self, user_id: str) -> int:
        """Count jobs with active status (queued or processing) for a user.

        Args:
            user_id: The owner_user_id to count active jobs for.

        Returns:
            Number of active jobs for this user.
        """
        conn = self._connect()
        try:
            if self.backend == "postgres":
                with self._pg_cursor(conn) as cur:
                    cur.execute(
                        "SELECT COUNT(*) AS c FROM jobs WHERE owner_user_id = %s AND status IN ('queued', 'processing')",
                        (user_id,),
                    )
                    row = cur.fetchone()
                    return int(row["c"] if isinstance(row, dict) else (row[0] if row else 0))
            else:
                row = conn.execute(
                    "SELECT COUNT(*) FROM jobs WHERE owner_user_id = ? AND status IN ('queued', 'processing')",
                    (user_id,),
                ).fetchone()
                return int(row[0] if row else 0)
        except _JOB_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug(f"Failed to count active jobs for user {user_id}: {exc}")
            return 0
        finally:
            conn.close()

    @staticmethod
    def _map_fair_share_score_to_priority(score: int) -> int:
        """Convert a higher fair-share score into a higher queue priority.

        Queue priority is stored as 1..10 where lower numbers run first.
        """
        clamped_score = max(0, min(100, int(score)))
        return max(1, 10 - min(9, clamped_score // 10))

    def _get_allowed_queues(self, domain: str | None = None) -> list[str]:
        allowed = list(self.STANDARD_QUEUES)
        if domain:
            domain_key = str(domain).strip().lower()
            allowed.extend(self.DOMAIN_ALLOWED_QUEUES.get(domain_key, ()))
        extra = os.getenv("JOBS_ALLOWED_QUEUES", "").strip()
        if extra:
            allowed.extend([q.strip() for q in extra.split(",") if q.strip()])
        if domain:
            key = f"JOBS_ALLOWED_QUEUES_{str(domain).upper()}"
            extra_d = os.getenv(key, "").strip()
            if extra_d:
                allowed.extend([q.strip() for q in extra_d.split(",") if q.strip()])
        # Deduplicate preserving order
        dedup: list[str] = []
        seen = set()
        for q in allowed:
            if q not in seen:
                dedup.append(q)
                seen.add(q)
        return dedup

    def _assert_invariants(self, row: dict[str, Any]) -> None:
        try:
            status = str(row.get("status") or "")
            lease_id = row.get("lease_id")
            leased_until = _parse_dt(row.get("leased_until"))
            acquired_at = _parse_dt(row.get("acquired_at"))
            if status != "processing" and lease_id:
                logger.warning(f"Jobs invariant: non-processing job has lease_id (id={row.get('id')}, status={status})")
            if leased_until and acquired_at and leased_until < acquired_at:
                logger.warning(
                    f"Jobs invariant: leased_until < acquired_at (id={row.get('id')}, leased_until={leased_until}, acquired_at={acquired_at})"
                )
        except _JOB_NONCRITICAL_EXCEPTIONS:
            # Never raise from invariant checks
            pass

    # Connection helper
    def _connect(self):
        if self.backend == "postgres":
            import psycopg

            conn = psycopg.connect(self.db_url)
            return conn
        conn = sqlite3.connect(self.db_path)
        # Apply pragmatic SQLite settings for concurrent read/write under tests and dev
        with contextlib.suppress(_JOB_NONCRITICAL_EXCEPTIONS):
            configure_sqlite_connection(conn)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _sqlite_missing_column_error(exc: Exception, column: str) -> bool:
        if not isinstance(exc, sqlite3.OperationalError):
            return False
        msg = str(exc).lower()
        return f"no column named {column}" in msg or f"no such column: {column}" in msg

    def _sqlite_ensure_batch_group(self, conn: sqlite3.Connection) -> bool:
        try:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(jobs)").fetchall()}
            if "batch_group" not in cols:
                conn.execute("ALTER TABLE jobs ADD COLUMN batch_group TEXT")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_batch_group "
                "ON jobs(batch_group)"
            )
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='jobs_archive'"
            ).fetchone()
            if row:
                cols_arch = {r[1] for r in conn.execute("PRAGMA table_info(jobs_archive)").fetchall()}
                if "batch_group" not in cols_arch:
                    conn.execute("ALTER TABLE jobs_archive ADD COLUMN batch_group TEXT")
                _ensure_sqlite_archive_batch_read_indexes(conn)
            conn.commit()
            return True  # noqa: TRY300
        except sqlite3.Error as exc:
            with contextlib.suppress(_JOB_NONCRITICAL_EXCEPTIONS):
                conn.rollback()
            logger.warning(f"Jobs schema backfill for batch_group failed: {exc}")
            return False

    def _pg_cursor(self, conn):
        from psycopg.rows import dict_row  # type: ignore

        cur = conn.cursor(row_factory=dict_row)

        def _is_serialization_failure(exc: Exception) -> bool:
            try:
                from psycopg import errors as pg_errors  # type: ignore
            except _JOB_NONCRITICAL_EXCEPTIONS:
                return False
            return isinstance(exc, pg_errors.SerializationFailure)

        # Apply per-transaction RLS via SET LOCAL to avoid cross-request leakage
        role = str(os.getenv("JOBS_PG_RLS_ROLE", "")).strip()
        if role:
            try:
                import re as _re

                if _re.match(r"^[A-Za-z0-9_]+$", role):
                    cur.execute(f"SET ROLE {role}")
            except _JOB_NONCRITICAL_EXCEPTIONS as exc:
                if _is_serialization_failure(exc):
                    raise
                with contextlib.suppress(_JOB_NONCRITICAL_EXCEPTIONS):
                    conn.rollback()
        try:
            from psycopg import sql as _sql  # type: ignore

            is_admin = bool(JobManager._RLS_IS_ADMIN.get())
            cur.execute(_sql.SQL("SET app.is_admin = {}").format(_sql.Literal("true" if is_admin else "false")))

            def _set_or_reset(name: str, value: str | None) -> None:
                if value:
                    cur.execute(
                        _sql.SQL("SET {} = {}").format(
                            _sql.SQL(name),
                            _sql.Literal(str(value)),
                        )
                    )
                    return
                try:
                    cur.execute(_sql.SQL("RESET {}").format(_sql.SQL(name)))
                except _JOB_NONCRITICAL_EXCEPTIONS:
                    with contextlib.suppress(_JOB_NONCRITICAL_EXCEPTIONS):
                        conn.rollback()
                    try:
                        cur.execute(
                            _sql.SQL("SET {} = {}").format(
                                _sql.SQL(name),
                                _sql.Literal(""),
                            )
                        )
                    except _JOB_NONCRITICAL_EXCEPTIONS:
                        with contextlib.suppress(_JOB_NONCRITICAL_EXCEPTIONS):
                            conn.rollback()

            dom = JobManager._RLS_DOMAIN_ALLOWLIST.get()
            _set_or_reset("app.domain_allowlist", dom)
            owner = JobManager._RLS_OWNER_USER_ID.get()
            _set_or_reset("app.owner_user_id", owner)
            if JobManager._is_truthy(os.getenv("JOBS_PG_RLS_DEBUG", "")):
                try:
                    cur.execute(
                        "SELECT "
                        "current_setting('app.is_admin', true) AS is_admin, "
                        "current_setting('app.domain_allowlist', true) AS domain_allowlist, "
                        "current_setting('app.owner_user_id', true) AS owner_user_id"
                    )
                    row = cur.fetchone()
                    print(f"[jobs-rls-debug] settings={row}")
                except _JOB_NONCRITICAL_EXCEPTIONS:
                    pass
        except _JOB_NONCRITICAL_EXCEPTIONS as exc:
            if _is_serialization_failure(exc):
                raise
            # Non-fatal: continue without RLS context if GUCs unavailable
            # Some Postgres installations reject unknown GUCs (custom parameters).
            # If any SET LOCAL fails, the transaction enters an aborted state.
            # Roll back to clear the error so subsequent statements can proceed.
            with contextlib.suppress(_JOB_NONCRITICAL_EXCEPTIONS):
                conn.rollback()
        return cur

    @staticmethod
    def _normalize_slides_reconciliation_row(row: Any) -> dict[str, Any]:
        data = dict(row)
        for field_name in (
            "lease_expires_at",
            "diagnostic_at",
            "sweep_started_at",
            "sweep_completed_at",
        ):
            value = _parse_dt(data.get(field_name))
            if value is not None:
                if value.tzinfo is None:
                    value = value.replace(tzinfo=_tz.utc)
                else:
                    value = value.astimezone(_tz.utc)
            data[field_name] = value
        data["fencing_token"] = int(data.get("fencing_token") or 0)
        data["lag"] = int(data.get("lag") or 0)
        data["diagnostic_count"] = int(data.get("diagnostic_count") or 0)
        data["sweep_complete"] = bool(data.get("sweep_complete"))
        data["unexpired_reference_count"] = int(data.get("unexpired_reference_count") or 0)
        return data

    @staticmethod
    def _validate_slides_coordination_identity(*, holder_uuid: str, config_revision: str) -> None:
        if not isinstance(holder_uuid, str) or not holder_uuid.strip() or len(holder_uuid) > 128:
            raise ValueError("holder_uuid must be a bounded nonblank identifier")
        if not isinstance(config_revision, str) or not config_revision or len(config_revision) > 512:
            raise ValueError("config_revision must be a bounded opaque token")

    def get_slides_reconciliation_state(self) -> dict[str, Any]:
        """Read the singleton standalone-generation coordination state."""
        conn = self._connect()
        try:
            if self.backend == "postgres":
                with conn, self._pg_cursor(conn) as cur:
                    cur.execute("SELECT * FROM slides_standalone_reconciliation WHERE singleton_id=1")
                    row = cur.fetchone()
            else:
                row = conn.execute("SELECT * FROM slides_standalone_reconciliation WHERE singleton_id=1").fetchone()
            if row is None:
                raise RuntimeError("standalone reconciliation singleton is missing")
            return self._normalize_slides_reconciliation_row(row)
        finally:
            conn.close()

    def acquire_slides_reconciliation_lease(
        self,
        *,
        holder_uuid: str,
        lease_seconds: int,
        config_revision: str,
        now: datetime | None = None,
    ) -> dict[str, Any] | None:
        """Acquire an expired/unheld singleton lease and advance its fence."""
        self._validate_slides_coordination_identity(
            holder_uuid=holder_uuid,
            config_revision=config_revision,
        )
        if isinstance(lease_seconds, bool) or int(lease_seconds) <= 0:
            raise ValueError("lease_seconds must be positive")
        now_utc = _require_aware_utc(now, field_name="now")
        expires_at = now_utc + timedelta(seconds=int(lease_seconds))
        conn = self._connect()
        try:
            if self.backend == "postgres":
                with conn, self._pg_cursor(conn) as cur:
                    cur.execute("SELECT * FROM slides_standalone_reconciliation " "WHERE singleton_id=1 FOR UPDATE")
                    row = cur.fetchone()
                    if row is None:
                        return None
                    cur.execute(
                        "SELECT config_revision FROM slides_standalone_key_registry " "ORDER BY key_id FOR UPDATE"
                    )
                    registry_revisions = {item["config_revision"] for item in cur.fetchall()}
                    if registry_revisions and registry_revisions != {config_revision}:
                        return None
                    observed = dict(row)
                    observed_expiry = _parse_dt(observed.get("lease_expires_at"))
                    if observed_expiry is not None and observed_expiry.tzinfo is None:
                        observed_expiry = observed_expiry.replace(tzinfo=_tz.utc)
                    same_revision = observed.get("config_revision") == config_revision
                    if same_revision and observed_expiry is not None and observed_expiry > now_utc:
                        return None
                    if same_revision:
                        cur.execute(
                            """
                            UPDATE slides_standalone_reconciliation
                            SET holder_uuid=%s, lease_expires_at=%s,
                                fencing_token=fencing_token + 1,
                                sweep_key_id=NULL, sweep_started_at=NULL,
                                sweep_completed_at=NULL, sweep_complete=FALSE,
                                unexpired_reference_count=0
                            WHERE singleton_id=1
                              AND fencing_token=%s
                              AND config_revision IS NOT DISTINCT FROM %s
                              AND holder_uuid IS NOT DISTINCT FROM %s
                            RETURNING *
                            """,
                            (
                                holder_uuid,
                                expires_at,
                                int(observed.get("fencing_token") or 0),
                                observed.get("config_revision"),
                                observed.get("holder_uuid"),
                            ),
                        )
                    else:
                        cur.execute(
                            """
                            UPDATE slides_standalone_reconciliation
                            SET holder_uuid=%s, lease_expires_at=%s,
                                fencing_token=fencing_token + 1,
                                config_revision=%s, cursor=NULL,
                                startup_complete_epoch=NULL, last_complete_epoch=NULL,
                                lag=0, sweep_key_id=NULL, sweep_started_at=NULL,
                                sweep_completed_at=NULL, sweep_complete=FALSE,
                                unexpired_reference_count=0
                            WHERE singleton_id=1
                              AND fencing_token=%s
                              AND config_revision IS NOT DISTINCT FROM %s
                              AND holder_uuid IS NOT DISTINCT FROM %s
                            RETURNING *
                            """,
                            (
                                holder_uuid,
                                expires_at,
                                config_revision,
                                int(observed.get("fencing_token") or 0),
                                observed.get("config_revision"),
                                observed.get("holder_uuid"),
                            ),
                        )
                    updated = cur.fetchone()
                    return self._normalize_slides_reconciliation_row(updated) if updated is not None else None

            conn.execute("BEGIN IMMEDIATE")
            registry_revisions = {
                item[0]
                for item in conn.execute(
                    "SELECT DISTINCT config_revision " "FROM slides_standalone_key_registry"
                ).fetchall()
            }
            if registry_revisions and registry_revisions != {config_revision}:
                conn.rollback()
                return None
            row = conn.execute("SELECT * FROM slides_standalone_reconciliation WHERE singleton_id=1").fetchone()
            if row is None:
                conn.rollback()
                return None
            observed = dict(row)
            observed_expiry = _parse_dt(observed.get("lease_expires_at"))
            if observed_expiry is not None and observed_expiry.tzinfo is None:
                observed_expiry = observed_expiry.replace(tzinfo=_tz.utc)
            same_revision = observed.get("config_revision") == config_revision
            if same_revision and observed_expiry is not None and observed_expiry > now_utc:
                conn.rollback()
                return None
            common_where = (
                int(observed.get("fencing_token") or 0),
                observed.get("config_revision"),
                observed.get("holder_uuid"),
            )
            if same_revision:
                result = conn.execute(
                    """
                    UPDATE slides_standalone_reconciliation
                    SET holder_uuid=?, lease_expires_at=?, fencing_token=fencing_token + 1,
                        sweep_key_id=NULL, sweep_started_at=NULL,
                        sweep_completed_at=NULL, sweep_complete=0,
                        unexpired_reference_count=0
                    WHERE singleton_id=1 AND fencing_token=?
                      AND config_revision IS ? AND holder_uuid IS ?
                    """,
                    (holder_uuid, _sqlite_utc(expires_at), *common_where),
                )
            else:
                result = conn.execute(
                    """
                    UPDATE slides_standalone_reconciliation
                    SET holder_uuid=?, lease_expires_at=?, fencing_token=fencing_token + 1,
                        config_revision=?, cursor=NULL, startup_complete_epoch=NULL,
                        last_complete_epoch=NULL, lag=0, sweep_key_id=NULL,
                        sweep_started_at=NULL, sweep_completed_at=NULL,
                        sweep_complete=0, unexpired_reference_count=0
                    WHERE singleton_id=1 AND fencing_token=?
                      AND config_revision IS ? AND holder_uuid IS ?
                    """,
                    (
                        holder_uuid,
                        _sqlite_utc(expires_at),
                        config_revision,
                        *common_where,
                    ),
                )
            if result.rowcount != 1:
                conn.rollback()
                return None
            updated = conn.execute("SELECT * FROM slides_standalone_reconciliation WHERE singleton_id=1").fetchone()
            conn.commit()
            return self._normalize_slides_reconciliation_row(updated)
        except Exception:
            with contextlib.suppress(Exception):
                conn.rollback()
            raise
        finally:
            conn.close()

    def renew_slides_reconciliation_lease(
        self,
        *,
        holder_uuid: str,
        fencing_token: int,
        config_revision: str,
        lease_seconds: int,
        now: datetime | None = None,
    ) -> bool:
        """Renew a still-live lease using holder, fence, and revision CAS."""
        self._validate_slides_coordination_identity(
            holder_uuid=holder_uuid,
            config_revision=config_revision,
        )
        if isinstance(fencing_token, bool) or int(fencing_token) <= 0:
            raise ValueError("fencing_token must be positive")
        if isinstance(lease_seconds, bool) or int(lease_seconds) <= 0:
            raise ValueError("lease_seconds must be positive")
        now_utc = _require_aware_utc(now, field_name="now")
        expires_at = now_utc + timedelta(seconds=int(lease_seconds))
        conn = self._connect()
        try:
            if self.backend == "postgres":
                with conn, self._pg_cursor(conn) as cur:
                    cur.execute(
                        """
                        UPDATE slides_standalone_reconciliation
                        SET lease_expires_at=%s
                        WHERE singleton_id=1 AND holder_uuid=%s AND fencing_token=%s
                          AND config_revision=%s AND lease_expires_at > %s
                        """,
                        (expires_at, holder_uuid, int(fencing_token), config_revision, now_utc),
                    )
                    return cur.rowcount == 1
            with conn:
                result = conn.execute(
                    """
                    UPDATE slides_standalone_reconciliation
                    SET lease_expires_at=?
                    WHERE singleton_id=1 AND holder_uuid=? AND fencing_token=?
                      AND config_revision=? AND lease_expires_at > ?
                    """,
                    (
                        _sqlite_utc(expires_at),
                        holder_uuid,
                        int(fencing_token),
                        config_revision,
                        _sqlite_utc(now_utc),
                    ),
                )
                return result.rowcount == 1
        finally:
            conn.close()

    def checkpoint_slides_reconciliation(
        self,
        *,
        holder_uuid: str,
        fencing_token: int,
        config_revision: str,
        cursor: str | None,
        startup_complete_epoch: str | None,
        last_complete_epoch: float | None,
        lag: int,
        now: datetime | None = None,
        completed: bool = False,
        sweep_key_id: str | None = None,
        sweep_started_at: datetime | None = None,
        unexpired_reference_count: int | None = None,
    ) -> bool:
        """Publish fenced reconciliation progress or one completed full sweep."""
        self._validate_slides_coordination_identity(
            holder_uuid=holder_uuid,
            config_revision=config_revision,
        )
        if isinstance(fencing_token, bool) or int(fencing_token) <= 0:
            raise ValueError("fencing_token must be positive")
        if startup_complete_epoch not in (None, config_revision):
            raise ValueError("startup_complete_epoch must exactly match config_revision")
        if cursor is not None and (not isinstance(cursor, str) or len(cursor) > 1024):
            raise ValueError("cursor must be a bounded string")
        if isinstance(lag, bool) or not isinstance(lag, int) or lag < 0:
            raise ValueError("lag must be a nonnegative integer")
        if last_complete_epoch is not None and float(last_complete_epoch) < 0:
            raise ValueError("last_complete_epoch must be nonnegative")
        now_utc = _require_aware_utc(now, field_name="now")
        update_sweep = sweep_key_id is not None
        if update_sweep and (
            isinstance(unexpired_reference_count, bool)
            or not isinstance(unexpired_reference_count, int)
            or unexpired_reference_count < 0
        ):
            raise ValueError("unexpired_reference_count must be a nonnegative integer for a sweep")
        if not update_sweep and unexpired_reference_count is not None:
            raise ValueError("sweep_key_id is required with unexpired_reference_count")
        if sweep_started_at is not None:
            sweep_started_at = _require_aware_utc(
                sweep_started_at,
                field_name="sweep_started_at",
            )
        if completed:
            cursor = None
            lag = 0
            if last_complete_epoch is None:
                last_complete_epoch = now_utc.timestamp()
        sweep_complete = bool(completed and update_sweep)
        conn = self._connect()
        try:
            if self.backend == "postgres":
                with conn, self._pg_cursor(conn) as cur:
                    cur.execute(
                        """
                        UPDATE slides_standalone_reconciliation
                        SET cursor=%s, startup_complete_epoch=%s,
                            last_complete_epoch=%s, lag=%s,
                            sweep_key_id=CASE WHEN %s THEN %s ELSE sweep_key_id END,
                            sweep_started_at=CASE WHEN %s THEN %s ELSE sweep_started_at END,
                            sweep_completed_at=CASE WHEN %s THEN %s ELSE sweep_completed_at END,
                            sweep_complete=CASE WHEN %s THEN %s ELSE sweep_complete END,
                            unexpired_reference_count=CASE WHEN %s THEN %s ELSE unexpired_reference_count END
                        WHERE singleton_id=1 AND holder_uuid=%s AND fencing_token=%s
                          AND config_revision=%s AND lease_expires_at > %s
                        """,
                        (
                            cursor,
                            startup_complete_epoch,
                            last_complete_epoch,
                            lag,
                            update_sweep,
                            sweep_key_id,
                            update_sweep,
                            sweep_started_at,
                            update_sweep,
                            now_utc if completed else None,
                            update_sweep,
                            sweep_complete,
                            update_sweep,
                            unexpired_reference_count,
                            holder_uuid,
                            int(fencing_token),
                            config_revision,
                            now_utc,
                        ),
                    )
                    return cur.rowcount == 1
            with conn:
                result = conn.execute(
                    """
                    UPDATE slides_standalone_reconciliation
                    SET cursor=?, startup_complete_epoch=?, last_complete_epoch=?, lag=?,
                        sweep_key_id=CASE WHEN ? THEN ? ELSE sweep_key_id END,
                        sweep_started_at=CASE WHEN ? THEN ? ELSE sweep_started_at END,
                        sweep_completed_at=CASE WHEN ? THEN ? ELSE sweep_completed_at END,
                        sweep_complete=CASE WHEN ? THEN ? ELSE sweep_complete END,
                        unexpired_reference_count=CASE WHEN ? THEN ? ELSE unexpired_reference_count END
                    WHERE singleton_id=1 AND holder_uuid=? AND fencing_token=?
                      AND config_revision=? AND lease_expires_at > ?
                    """,
                    (
                        cursor,
                        startup_complete_epoch,
                        last_complete_epoch,
                        lag,
                        int(update_sweep),
                        sweep_key_id,
                        int(update_sweep),
                        _sqlite_utc(sweep_started_at) if sweep_started_at else None,
                        int(update_sweep),
                        _sqlite_utc(now_utc) if completed else None,
                        int(update_sweep),
                        int(sweep_complete),
                        int(update_sweep),
                        unexpired_reference_count,
                        holder_uuid,
                        int(fencing_token),
                        config_revision,
                        _sqlite_utc(now_utc),
                    ),
                )
                return result.rowcount == 1
        finally:
            conn.close()

    def release_slides_reconciliation_lease(
        self,
        *,
        holder_uuid: str,
        fencing_token: int,
        config_revision: str,
        now: datetime | None = None,
    ) -> bool:
        """Release the unchanged lease, including an expired non-taken-over lease."""
        self._validate_slides_coordination_identity(
            holder_uuid=holder_uuid,
            config_revision=config_revision,
        )
        if isinstance(fencing_token, bool) or int(fencing_token) <= 0:
            raise ValueError("fencing_token must be positive")
        _require_aware_utc(now, field_name="now")
        conn = self._connect()
        try:
            if self.backend == "postgres":
                with conn, self._pg_cursor(conn) as cur:
                    cur.execute(
                        """
                        UPDATE slides_standalone_reconciliation
                        SET holder_uuid=NULL, lease_expires_at=NULL
                        WHERE singleton_id=1 AND holder_uuid=%s AND fencing_token=%s
                          AND config_revision=%s
                        """,
                        (holder_uuid, int(fencing_token), config_revision),
                    )
                    return cur.rowcount == 1
            with conn:
                result = conn.execute(
                    """
                    UPDATE slides_standalone_reconciliation
                    SET holder_uuid=NULL, lease_expires_at=NULL
                    WHERE singleton_id=1 AND holder_uuid=? AND fencing_token=?
                      AND config_revision=?
                    """,
                    (holder_uuid, int(fencing_token), config_revision),
                )
                return result.rowcount == 1
        finally:
            conn.close()

    def get_slides_generation_readiness(self) -> dict[str, Any]:
        """Return standalone-only migration/index readiness diagnostics."""
        state = self.get_slides_reconciliation_state()
        conn = self._connect()
        try:
            if self.backend == "postgres":
                with conn, self._pg_cursor(conn) as cur:
                    projection_ready = slides_archive_projection_ready_pg(cur)
                    index_ready = slides_archive_indexes_ready_pg(cur)
            else:
                projection_ready = slides_archive_projection_ready_sqlite(conn)
                index_ready = slides_archive_indexes_ready_sqlite(conn)
        finally:
            conn.close()
        return {
            "ready": (
                state.get("diagnostic_code") is None
                and projection_ready
                and index_ready
            ),
            "diagnostic_code": state.get("diagnostic_code"),
            "diagnostic_count": state.get("diagnostic_count", 0),
            "diagnostic_at": state.get("diagnostic_at"),
            "archive_indexes_ready": index_ready,
            "archive_projection_ready": projection_ready,
        }

    def list_active_slides_generation_owner_ids(
        self,
        *,
        after_owner_user_id: str | None = None,
        limit: int = 100,
    ) -> list[str]:
        """List distinct owners with active standalone generation jobs."""
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
            raise ValueError("limit must be an integer between 1 and 1000")
        if after_owner_user_id is not None:
            if (
                not isinstance(after_owner_user_id, str)
                or not after_owner_user_id.strip()
                or len(after_owner_user_id.encode("utf-8")) > 512
            ):
                raise ValueError("after_owner_user_id must be a nonblank UTF-8 string of at most 512 bytes")

        conn = self._connect()
        try:
            if self.backend == "postgres":
                with conn, self._pg_cursor(conn) as cur:
                    cur.execute(
                        """
                        SELECT owner_user_id
                        FROM (
                            SELECT DISTINCT owner_user_id
                            FROM jobs
                            WHERE domain=%s AND queue=%s AND job_type=%s
                              AND status IN ('queued', 'processing')
                              AND owner_user_id IS NOT NULL
                              AND BTRIM(owner_user_id) <> ''
                        ) AS active_owners
                        WHERE (CAST(%s AS TEXT) IS NULL
                               OR owner_user_id COLLATE "C" > %s)
                        ORDER BY owner_user_id COLLATE "C" ASC
                        LIMIT %s
                        """,
                        (
                            _SLIDES_GENERATION_DOMAIN,
                            _SLIDES_GENERATION_QUEUE,
                            _SLIDES_GENERATION_JOB_TYPE,
                            after_owner_user_id,
                            after_owner_user_id,
                            limit,
                        ),
                    )
                    return [str(row["owner_user_id"]) for row in cur.fetchall()]

            rows = conn.execute(
                """
                SELECT owner_user_id
                FROM (
                    SELECT DISTINCT owner_user_id
                    FROM jobs
                    WHERE domain=? AND queue=? AND job_type=?
                      AND status IN ('queued', 'processing')
                      AND owner_user_id IS NOT NULL
                      AND TRIM(owner_user_id) <> ''
                ) AS active_owners
                WHERE (? IS NULL OR owner_user_id COLLATE BINARY > ?)
                ORDER BY owner_user_id COLLATE BINARY ASC
                LIMIT ?
                """,
                (
                    _SLIDES_GENERATION_DOMAIN,
                    _SLIDES_GENERATION_QUEUE,
                    _SLIDES_GENERATION_JOB_TYPE,
                    after_owner_user_id,
                    after_owner_user_id,
                    limit,
                ),
            ).fetchall()
            return [str(row["owner_user_id"]) for row in rows]
        finally:
            conn.close()

    def _slides_generation_ready_in_connection(
        self,
        conn: Any,
        *,
        cursor: Any | None = None,
    ) -> bool:
        """Check standalone readiness on the already-serialized create connection."""
        if self.backend == "postgres":
            if cursor is None:
                raise RuntimeError("PostgreSQL readiness check requires a cursor")
            cursor.execute(
                "SELECT diagnostic_code FROM slides_standalone_reconciliation "
                "WHERE singleton_id=1 FOR SHARE"
            )
            row = cursor.fetchone()
            diagnostic = row.get("diagnostic_code") if isinstance(row, dict) else (row[0] if row else None)
            return (
                row is not None
                and diagnostic is None
                and slides_archive_projection_ready_pg(cursor)
                and slides_archive_indexes_ready_pg(cursor)
            )
        row = conn.execute(
            "SELECT diagnostic_code FROM slides_standalone_reconciliation WHERE singleton_id=1"
        ).fetchone()
        return (
            row is not None
            and row[0] is None
            and slides_archive_projection_ready_sqlite(conn)
            and slides_archive_indexes_ready_sqlite(conn)
        )

    def _serialized_slides_generation_replay(
        self,
        *,
        owner_user_id: str,
        idempotency_key: str,
        expected_job_uuid: str | None = None,
        expected_job_id: int | None = None,
        rejection: Exception | None = None,
    ) -> dict[str, Any] | None:
        """Resolve one replay under the create/prune lock or raise its rejection."""
        conn = self._connect()
        try:
            if self.backend == "postgres":
                with conn, self._pg_cursor(conn) as cur:
                    cur.execute(
                        "SELECT pg_advisory_xact_lock(%s)",
                        (
                            self._pg_advisory_key(
                                *_SLIDES_GENERATION_CORRELATION_LOCK_PARTS
                            ),
                        ),
                    )
                    if not self._slides_generation_ready_in_connection(
                        conn,
                        cursor=cur,
                    ):
                        raise SlidesGenerationJobsUnavailableError(
                            "presentation.generate Jobs coordination is unavailable"
                        )
                    existing = self._lookup_slides_generation_job_in_connection(
                        conn,
                        owner_user_id=owner_user_id,
                        idempotency_key=idempotency_key,
                        expected_job_uuid=expected_job_uuid,
                        expected_job_id=expected_job_id,
                        cursor=cur,
                    )
                    if existing is not None:
                        return existing
                    if rejection is not None:
                        raise rejection
                    return None

            with conn:
                conn.execute("BEGIN IMMEDIATE")
                if not self._slides_generation_ready_in_connection(conn):
                    raise SlidesGenerationJobsUnavailableError(
                        "presentation.generate Jobs coordination is unavailable"
                    )
                existing = self._lookup_slides_generation_job_in_connection(
                    conn,
                    owner_user_id=owner_user_id,
                    idempotency_key=idempotency_key,
                    expected_job_uuid=expected_job_uuid,
                    expected_job_id=expected_job_id,
                )
                if existing is not None:
                    return existing
                if rejection is not None:
                    raise rejection
                return None
        finally:
            conn.close()

    def _lookup_ready_slides_generation_job_in_connection(
        self,
        conn: Any,
        *,
        owner_user_id: str,
        idempotency_key: str,
        cursor: Any | None = None,
    ) -> dict[str, Any] | None:
        """Check readiness and resolve one correlation under the same fence."""

        if not self._slides_generation_ready_in_connection(
            conn,
            cursor=cursor,
        ):
            raise SlidesGenerationJobsUnavailableError(
                "presentation.generate Jobs coordination is unavailable"
            )
        return self._lookup_slides_generation_job_in_connection(
            conn,
            owner_user_id=owner_user_id,
            idempotency_key=idempotency_key,
            cursor=cursor,
        )

    def _record_slides_generation_diagnostic(
        self,
        conn: Any,
        *,
        code: str,
        count: int,
    ) -> None:
        """Persist a bounded standalone-only archive diagnostic before failing closed."""
        if self.backend == "postgres":
            with self._pg_cursor(conn) as cur:
                cur.execute(
                    """
                    UPDATE slides_standalone_reconciliation
                    SET diagnostic_code=CASE
                          WHEN diagnostic_code='duplicate_archive_uuid'
                          THEN diagnostic_code ELSE %s END,
                        diagnostic_count=CASE
                          WHEN diagnostic_code='duplicate_archive_uuid'
                          THEN diagnostic_count ELSE %s END,
                        diagnostic_at=CASE
                          WHEN diagnostic_code='duplicate_archive_uuid'
                          THEN diagnostic_at ELSE NOW() END
                    WHERE singleton_id=1
                    """,
                    (code, max(1, int(count))),
                )
        else:
            conn.execute(
                """
                UPDATE slides_standalone_reconciliation
                SET diagnostic_code=CASE
                      WHEN diagnostic_code='duplicate_archive_uuid'
                      THEN diagnostic_code ELSE ? END,
                    diagnostic_count=CASE
                      WHEN diagnostic_code='duplicate_archive_uuid'
                      THEN diagnostic_count ELSE ? END,
                    diagnostic_at=CASE
                      WHEN diagnostic_code='duplicate_archive_uuid'
                      THEN diagnostic_at ELSE DATETIME('now') END
                WHERE singleton_id=1
                """,
                (code, max(1, int(count))),
            )
        conn.commit()

    def _lookup_slides_generation_job_in_connection(
        self,
        conn: Any,
        *,
        owner_user_id: str,
        idempotency_key: str,
        expected_job_uuid: str | None = None,
        expected_job_id: int | None = None,
        cursor: Any | None = None,
    ) -> dict[str, Any] | None:
        if self.backend == "postgres":
            cur = cursor
            if cur is None:
                raise RuntimeError("PostgreSQL generation lookup requires a cursor")
            cur.execute(
                "SELECT * FROM jobs WHERE domain='slides' AND queue='default' "
                "AND job_type='presentation.generate' AND owner_user_id=%s "
                "AND idempotency_key=%s LIMIT 2",
                (owner_user_id, idempotency_key),
            )
            rows = list(cur.fetchall() or [])
            archived = False
            if not rows:
                if expected_job_uuid is None:
                    cur.execute(
                        "SELECT * FROM jobs_archive WHERE domain='slides' AND queue='default' "
                        "AND job_type='presentation.generate' AND owner_user_id=%s "
                        "AND idempotency_key=%s ORDER BY archived_at DESC, uuid LIMIT 1",
                        (owner_user_id, idempotency_key),
                    )
                else:
                    cur.execute(
                        "SELECT * FROM jobs_archive WHERE domain='slides' AND queue='default' "
                        "AND job_type='presentation.generate' AND owner_user_id=%s "
                        "AND idempotency_key=%s AND uuid=%s "
                        "ORDER BY archived_at DESC, uuid LIMIT 2",
                        (owner_user_id, idempotency_key, expected_job_uuid),
                    )
                rows = list(cur.fetchall() or [])
                archived = bool(rows)
        else:
            rows = list(
                conn.execute(
                    "SELECT * FROM jobs WHERE domain='slides' AND queue='default' "
                    "AND job_type='presentation.generate' AND owner_user_id=? "
                    "AND idempotency_key=? LIMIT 2",
                    (owner_user_id, idempotency_key),
                ).fetchall()
            )
            archived = False
            if not rows:
                if expected_job_uuid is None:
                    archived_query = (
                        "SELECT * FROM jobs_archive WHERE domain='slides' AND queue='default' "
                        "AND job_type='presentation.generate' AND owner_user_id=? "
                        "AND idempotency_key=? ORDER BY archived_at DESC, uuid LIMIT 1"
                    )
                    archived_params = (owner_user_id, idempotency_key)
                else:
                    archived_query = (
                        "SELECT * FROM jobs_archive WHERE domain='slides' AND queue='default' "
                        "AND job_type='presentation.generate' AND owner_user_id=? "
                        "AND idempotency_key=? AND uuid=? "
                        "ORDER BY archived_at DESC, uuid LIMIT 2"
                    )
                    archived_params = (
                        owner_user_id,
                        idempotency_key,
                        expected_job_uuid,
                    )
                rows = list(conn.execute(archived_query, archived_params).fetchall())
                archived = bool(rows)
        if not rows:
            return None
        uuid_values = [str(dict(row).get("uuid") or "").strip() for row in rows]
        if not archived and len(rows) > 1:
            self._record_slides_generation_diagnostic(
                conn,
                code="ambiguous_generation_legacy_row",
                count=len(rows),
            )
            raise SlidesGenerationJobsUnavailableError("presentation.generate correlation is unsafe")
        if any(not value for value in uuid_values):
            self._record_slides_generation_diagnostic(
                conn,
                code="ambiguous_generation_legacy_row",
                count=len(rows),
            )
            raise SlidesGenerationJobsUnavailableError("presentation.generate correlation is unsafe")
        if archived and len(set(uuid_values)) < len(uuid_values):
            self._record_slides_generation_diagnostic(
                conn,
                code="duplicate_archive_uuid",
                count=len(rows),
            )
            raise SlidesGenerationJobsUnavailableError("presentation.generate correlation is unsafe")
        selected = rows[0]
        if archived and expected_job_uuid is not None:
            matching_rows = [
                row
                for row, candidate_uuid in zip(rows, uuid_values)
                if candidate_uuid == expected_job_uuid
            ]
            if not matching_rows:
                return None
            selected = matching_rows[0]
        result = self._normalize_archived_job(selected) if archived else dict(selected)
        job_uuid = str(result.get("uuid") or "").strip()
        if not job_uuid:
            self._record_slides_generation_diagnostic(
                conn,
                code="ambiguous_generation_legacy_row",
                count=1,
            )
            raise SlidesGenerationJobsUnavailableError("presentation.generate correlation is unsafe")
        if expected_job_uuid is not None and job_uuid != expected_job_uuid:
            return None
        if expected_job_id is not None:
            try:
                if result.get("id") is None or int(result["id"]) != int(expected_job_id):
                    return None
            except (TypeError, ValueError):
                return None
        if not archived:
            result["payload"] = self._maybe_decrypt_json(self._parse_json_value(result.get("payload")))
            result["result"] = self._maybe_decrypt_json(self._parse_json_value(result.get("result")))
            result["archived"] = False
        return result

    def lookup_slides_generation_job(
        self,
        *,
        owner_user_id: str,
        idempotency_key: str,
        expected_job_uuid: str | None = None,
        expected_job_id: int | None = None,
    ) -> dict[str, Any] | None:
        """Resolve the authoritative active-first generation row without requiring its UUID."""
        if not all(isinstance(value, str) and value.strip() for value in (owner_user_id, idempotency_key)):
            raise ValueError("owner_user_id and idempotency_key must be nonblank strings")
        return self._serialized_slides_generation_replay(
            owner_user_id=owner_user_id,
            idempotency_key=idempotency_key,
            expected_job_uuid=expected_job_uuid,
            expected_job_id=expected_job_id,
        )

    def resolve_slides_generation_job(
        self,
        *,
        job_uuid: str,
        owner_user_id: str,
        idempotency_key: str,
        job_id: int | None = None,
    ) -> dict[str, Any] | None:
        """Resolve one generation correlation with optional UUID and numeric-ID checks."""
        if not all(isinstance(value, str) and value.strip() for value in (job_uuid, owner_user_id, idempotency_key)):
            return None
        return self.lookup_slides_generation_job(
            owner_user_id=owner_user_id,
            idempotency_key=idempotency_key,
            expected_job_uuid=job_uuid,
            expected_job_id=job_id,
        )

    def _normalize_archived_job(self, row: Any) -> dict[str, Any]:
        """Decode one archived Jobs row without relying on its reusable numeric id."""
        result = normalize_slides_archive_projection(row)
        result["payload"] = self._maybe_decrypt_json(self._parse_json_value(result.get("payload")))
        result["result"] = self._maybe_decrypt_json(self._parse_json_value(result.get("result")))
        result["archived"] = True
        return result

    def _idempotent_slides_archive_collisions(
        self,
        conn: Any,
        *,
        where_clause: str,
        params: tuple[Any, ...],
        cursor: Any | None = None,
    ) -> set[str]:
        """Validate matching UUID collisions and return exact re-archive UUIDs."""
        exact_collisions: set[str] = set()
        collision_rows = fetch_slides_archive_collision_rows(
            conn,
            backend=self.backend,
            where_clause=where_clause,
            params=params,
            cursor=cursor,
        )
        for active_row, archived_rows in collision_rows:
            active = normalize_slides_archive_projection(active_row)
            job_uuid = str(active.get("uuid") or "").strip()
            if not archived_rows:
                continue
            if len(archived_rows) != 1:
                raise SlidesGenerationJobsUnavailableError("unsafe presentation.generate archive collision")
            archived = self._normalize_archived_job(archived_rows[0])
            active["payload"] = self._maybe_decrypt_json(
                self._parse_json_value(active.get("payload"))
            )
            active["result"] = self._maybe_decrypt_json(
                self._parse_json_value(active.get("result"))
            )
            if any(archived.get(field) != active.get(field) for field in SLIDES_ARCHIVE_EXACT_FIELDS):
                raise SlidesGenerationJobsUnavailableError("unsafe presentation.generate archive collision")
            exact_collisions.add(job_uuid)
        return exact_collisions

    @staticmethod
    def _normalize_slides_key_record(row: Any) -> dict[str, Any]:
        record = dict(row)
        for field_name in ("activated_at", "retired_at"):
            value = _parse_dt(record.get(field_name))
            if value is not None:
                if value.tzinfo is None:
                    value = value.replace(tzinfo=_tz.utc)
                else:
                    value = value.astimezone(_tz.utc)
            record[field_name] = value
        return record

    @classmethod
    def _slides_key_state_from_rows(cls, rows: list[Any]) -> dict[str, Any]:
        records = [cls._normalize_slides_key_record(row) for row in rows]
        revisions = {record["config_revision"] for record in records}
        if len(revisions) > 1:
            raise RuntimeError("standalone key registry has conflicting revisions")
        return {
            "records": records,
            "config_revision": next(iter(revisions), None),
        }

    def load_slides_digest_key_registry(self) -> dict[str, Any]:
        """Load source-free digest-key IDs, states, timestamps, and revision."""
        conn = self._connect()
        try:
            if self.backend == "postgres":
                with conn, self._pg_cursor(conn) as cur:
                    cur.execute(
                        "SELECT * FROM slides_standalone_key_registry "
                        "ORDER BY CASE state WHEN 'current' THEN 0 ELSE 1 END, key_id"
                    )
                    rows = list(cur.fetchall())
            else:
                rows = list(
                    conn.execute(
                        "SELECT * FROM slides_standalone_key_registry "
                        "ORDER BY CASE state WHEN 'current' THEN 0 ELSE 1 END, key_id"
                    ).fetchall()
                )
            return self._slides_key_state_from_rows(rows)
        finally:
            conn.close()

    @staticmethod
    def _validate_slides_key_cas_values(*, key_id: str, config_revision: str, changed_at: datetime) -> datetime:
        if not isinstance(key_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,31}", key_id):
            raise ValueError("digest key ID is invalid")
        if not isinstance(config_revision, str) or not config_revision or len(config_revision) > 512:
            raise ValueError("config revision is invalid")
        return _require_aware_utc(changed_at, field_name="changed_at")

    def compare_and_swap_slides_current_key(
        self,
        *,
        expected_current_key_id: str | None,
        expected_config_revision: str | None,
        new_current_key_id: str,
        new_config_revision: str,
        changed_at: datetime,
    ) -> dict[str, Any] | None:
        """Rotate source-free key metadata under one transactional CAS."""
        changed_at = self._validate_slides_key_cas_values(
            key_id=new_current_key_id,
            config_revision=new_config_revision,
            changed_at=changed_at,
        )
        conn = self._connect()
        try:
            if self.backend == "postgres":
                with conn, self._pg_cursor(conn) as cur:
                    cur.execute("SELECT * FROM slides_standalone_reconciliation " "WHERE singleton_id=1 FOR UPDATE")
                    reconciliation = cur.fetchone()
                    if reconciliation is None:
                        return None
                    reconciliation_revision = reconciliation.get("config_revision")
                    cur.execute("SELECT * FROM slides_standalone_key_registry ORDER BY key_id FOR UPDATE")
                    rows = list(cur.fetchall())
                    before = self._slides_key_state_from_rows(rows)
                    current = next(
                        (row for row in before["records"] if row["state"] == "current"),
                        None,
                    )
                    current_id = current["key_id"] if current else None
                    if current_id == new_current_key_id and before["config_revision"] == new_config_revision:
                        if reconciliation_revision != new_config_revision:
                            return None
                        return {"state": before, "applied_here": False}
                    if (
                        current_id != expected_current_key_id
                        or before["config_revision"] != expected_config_revision
                        or reconciliation_revision != expected_config_revision
                    ):
                        return None
                    if current_id and current_id != new_current_key_id:
                        cur.execute(
                            """
                            UPDATE slides_standalone_key_registry
                            SET state='retiring', retired_at=%s
                            WHERE key_id=%s AND state='current'
                            """,
                            (changed_at, current_id),
                        )
                    cur.execute(
                        "SELECT 1 FROM slides_standalone_key_registry WHERE key_id=%s",
                        (new_current_key_id,),
                    )
                    if cur.fetchone():
                        if current_id == new_current_key_id:
                            cur.execute(
                                """
                                UPDATE slides_standalone_key_registry
                                SET state='current', retired_at=NULL
                                WHERE key_id=%s
                                """,
                                (new_current_key_id,),
                            )
                        else:
                            cur.execute(
                                """
                                UPDATE slides_standalone_key_registry
                                SET state='current', activated_at=%s, retired_at=NULL
                                WHERE key_id=%s
                                """,
                                (changed_at, new_current_key_id),
                            )
                    else:
                        cur.execute(
                            """
                            INSERT INTO slides_standalone_key_registry(
                              key_id, state, activated_at, retired_at, config_revision
                            ) VALUES (%s, 'current', %s, NULL, %s)
                            """,
                            (new_current_key_id, changed_at, new_config_revision),
                        )
                    cur.execute(
                        "UPDATE slides_standalone_key_registry SET config_revision=%s",
                        (new_config_revision,),
                    )
                    if reconciliation_revision != new_config_revision:
                        cur.execute(
                            """
                            UPDATE slides_standalone_reconciliation
                            SET holder_uuid=NULL, lease_expires_at=NULL,
                                fencing_token=fencing_token + 1,
                                config_revision=%s, cursor=NULL,
                                startup_complete_epoch=NULL, last_complete_epoch=NULL,
                                lag=0, sweep_key_id=NULL, sweep_started_at=NULL,
                                sweep_completed_at=NULL, sweep_complete=FALSE,
                                unexpired_reference_count=0
                            WHERE singleton_id=1
                              AND config_revision IS NOT DISTINCT FROM %s
                            """,
                            (new_config_revision, expected_config_revision),
                        )
                        if cur.rowcount != 1:
                            return None
                    cur.execute(
                        "SELECT * FROM slides_standalone_key_registry "
                        "ORDER BY CASE state WHEN 'current' THEN 0 ELSE 1 END, key_id"
                    )
                    after = self._slides_key_state_from_rows(list(cur.fetchall()))
                    return {"state": after, "applied_here": True}

            conn.execute("BEGIN IMMEDIATE")
            reconciliation = conn.execute(
                "SELECT * FROM slides_standalone_reconciliation WHERE singleton_id=1"
            ).fetchone()
            if reconciliation is None:
                conn.rollback()
                return None
            reconciliation_revision = reconciliation["config_revision"]
            rows = list(conn.execute("SELECT * FROM slides_standalone_key_registry ORDER BY key_id").fetchall())
            before = self._slides_key_state_from_rows(rows)
            current = next(
                (row for row in before["records"] if row["state"] == "current"),
                None,
            )
            current_id = current["key_id"] if current else None
            if current_id == new_current_key_id and before["config_revision"] == new_config_revision:
                conn.rollback()
                if reconciliation_revision != new_config_revision:
                    return None
                return {"state": before, "applied_here": False}
            if (
                current_id != expected_current_key_id
                or before["config_revision"] != expected_config_revision
                or reconciliation_revision != expected_config_revision
            ):
                conn.rollback()
                return None
            changed_sql = _sqlite_utc(changed_at)
            if current_id and current_id != new_current_key_id:
                conn.execute(
                    """
                    UPDATE slides_standalone_key_registry
                    SET state='retiring', retired_at=?
                    WHERE key_id=? AND state='current'
                    """,
                    (changed_sql, current_id),
                )
            exists = conn.execute(
                "SELECT 1 FROM slides_standalone_key_registry WHERE key_id=?",
                (new_current_key_id,),
            ).fetchone()
            if exists:
                if current_id == new_current_key_id:
                    conn.execute(
                        """
                        UPDATE slides_standalone_key_registry
                        SET state='current', retired_at=NULL
                        WHERE key_id=?
                        """,
                        (new_current_key_id,),
                    )
                else:
                    conn.execute(
                        """
                        UPDATE slides_standalone_key_registry
                        SET state='current', activated_at=?, retired_at=NULL
                        WHERE key_id=?
                        """,
                        (changed_sql, new_current_key_id),
                    )
            else:
                conn.execute(
                    """
                    INSERT INTO slides_standalone_key_registry(
                      key_id, state, activated_at, retired_at, config_revision
                    ) VALUES (?, 'current', ?, NULL, ?)
                    """,
                    (new_current_key_id, changed_sql, new_config_revision),
                )
            conn.execute(
                "UPDATE slides_standalone_key_registry SET config_revision=?",
                (new_config_revision,),
            )
            if reconciliation_revision != new_config_revision:
                reconciled = conn.execute(
                    """
                    UPDATE slides_standalone_reconciliation
                    SET holder_uuid=NULL, lease_expires_at=NULL,
                        fencing_token=fencing_token + 1,
                        config_revision=?, cursor=NULL,
                        startup_complete_epoch=NULL, last_complete_epoch=NULL,
                        lag=0, sweep_key_id=NULL, sweep_started_at=NULL,
                        sweep_completed_at=NULL, sweep_complete=0,
                        unexpired_reference_count=0
                    WHERE singleton_id=1 AND config_revision IS ?
                    """,
                    (new_config_revision, expected_config_revision),
                )
                if reconciled.rowcount != 1:
                    conn.rollback()
                    return None
            after_rows = list(
                conn.execute(
                    "SELECT * FROM slides_standalone_key_registry "
                    "ORDER BY CASE state WHEN 'current' THEN 0 ELSE 1 END, key_id"
                ).fetchall()
            )
            conn.commit()
            return {
                "state": self._slides_key_state_from_rows(after_rows),
                "applied_here": True,
            }
        except Exception:
            with contextlib.suppress(Exception):
                conn.rollback()
            raise
        finally:
            conn.close()

    def compare_and_swap_remove_slides_key(
        self,
        *,
        key_id: str,
        expected_retired_at: datetime,
        expected_config_revision: str | None,
    ) -> dict[str, Any] | None:
        """Remove one unchanged retiring key metadata row."""
        expected_retired_at = _require_aware_utc(
            expected_retired_at,
            field_name="expected_retired_at",
        )
        checked_at = datetime.now(_tz.utc)
        conn = self._connect()
        try:
            if self.backend == "postgres":
                with conn, self._pg_cursor(conn) as cur:
                    cur.execute("SELECT * FROM slides_standalone_reconciliation " "WHERE singleton_id=1 FOR UPDATE")
                    reconciliation = cur.fetchone()
                    cur.execute(
                        "SELECT * FROM slides_standalone_key_registry WHERE key_id=%s FOR UPDATE",
                        (key_id,),
                    )
                    row = cur.fetchone()
                    if row is None:
                        cur.execute("SELECT * FROM slides_standalone_key_registry ORDER BY key_id")
                        return self._slides_key_state_from_rows(list(cur.fetchall()))
                    record = self._normalize_slides_key_record(row)
                    if (
                        reconciliation is None
                        or record["state"] != "retiring"
                        or record["retired_at"] != expected_retired_at
                        or record["config_revision"] != expected_config_revision
                        or not self._slides_zero_reference_proof_is_current(
                            reconciliation,
                            key_id=key_id,
                            config_revision=expected_config_revision,
                            retired_at=record["retired_at"],
                            checked_at=checked_at,
                        )
                    ):
                        return None
                    cur.execute(
                        "DELETE FROM slides_standalone_key_registry WHERE key_id=%s",
                        (key_id,),
                    )
                    cur.execute(
                        "SELECT * FROM slides_standalone_key_registry "
                        "ORDER BY CASE state WHEN 'current' THEN 0 ELSE 1 END, key_id"
                    )
                    return self._slides_key_state_from_rows(list(cur.fetchall()))

            conn.execute("BEGIN IMMEDIATE")
            reconciliation = conn.execute(
                "SELECT * FROM slides_standalone_reconciliation WHERE singleton_id=1"
            ).fetchone()
            row = conn.execute(
                "SELECT * FROM slides_standalone_key_registry WHERE key_id=?",
                (key_id,),
            ).fetchone()
            if row is None:
                rows = list(conn.execute("SELECT * FROM slides_standalone_key_registry").fetchall())
                conn.rollback()
                return self._slides_key_state_from_rows(rows)
            record = self._normalize_slides_key_record(row)
            if (
                reconciliation is None
                or record["state"] != "retiring"
                or record["retired_at"] != expected_retired_at
                or record["config_revision"] != expected_config_revision
                or not self._slides_zero_reference_proof_is_current(
                    reconciliation,
                    key_id=key_id,
                    config_revision=expected_config_revision,
                    retired_at=record["retired_at"],
                    checked_at=checked_at,
                )
            ):
                conn.rollback()
                return None
            conn.execute(
                "DELETE FROM slides_standalone_key_registry WHERE key_id=?",
                (key_id,),
            )
            rows = list(
                conn.execute(
                    "SELECT * FROM slides_standalone_key_registry "
                    "ORDER BY CASE state WHEN 'current' THEN 0 ELSE 1 END, key_id"
                ).fetchall()
            )
            conn.commit()
            return self._slides_key_state_from_rows(rows)
        except Exception:
            with contextlib.suppress(Exception):
                conn.rollback()
            raise
        finally:
            conn.close()

    @staticmethod
    def _slides_zero_reference_proof_is_current(
        reconciliation: Any,
        *,
        key_id: str,
        config_revision: str | None,
        retired_at: datetime,
        checked_at: datetime,
    ) -> bool:
        state = JobManager._normalize_slides_reconciliation_row(reconciliation)
        started_at = state.get("sweep_started_at")
        completed_at = state.get("sweep_completed_at")
        return bool(
            state.get("config_revision") == config_revision
            and state.get("sweep_key_id") == key_id
            and state.get("fencing_token", 0) > 0
            and state.get("sweep_complete")
            and state.get("unexpired_reference_count") == 0
            and started_at is not None
            and completed_at is not None
            and retired_at + timedelta(days=32) <= started_at <= completed_at <= checked_at
        )

    def load_slides_dormant_sweep_proof(self, *, key_id: str) -> dict[str, Any] | None:
        """Load the currently stored completed fenced dormant-reference proof."""
        state = self.get_slides_reconciliation_state()
        if not state.get("sweep_complete") or state.get("sweep_key_id") != key_id:
            return None
        started_at = state.get("sweep_started_at")
        completed_at = state.get("sweep_completed_at")
        if started_at is None or completed_at is None or state.get("config_revision") is None:
            return None
        return {
            "key_id": key_id,
            "config_revision": state["config_revision"],
            "fencing_token": state["fencing_token"],
            "sweep_started_at": started_at,
            "sweep_completed_at": completed_at,
            "complete": True,
            "unexpired_reference_count": state["unexpired_reference_count"],
        }

    # --- Acquire ordering policy (env-driven overrides) ---
    def _priority_dir_for(self, domain: str | None, backend: str) -> str:
        """Return 'ASC' or 'DESC' for priority ordering based on env.

        Env (checked in order):
          - JOBS_{BACKEND}_ACQUIRE_PRIORITY_DESC_DOMAINS (comma list)
          - JOBS_{ALIAS}_ACQUIRE_PRIORITY_DESC_DOMAINS (comma list)  # e.g., BACKEND=pg -> ALIAS=postgres
          - JOBS_ACQUIRE_PRIORITY_DESC_DOMAINS (comma list)
        If domain is listed => DESC; otherwise ASC.
        """
        try:
            dom = (domain or "").strip()
            b = (backend or "").strip().lower()
            key_backend = f"JOBS_{b.upper()}_ACQUIRE_PRIORITY_DESC_DOMAINS"
            # Support alternate alias names (e.g., pg -> postgres)
            alias = "postgres" if b == "pg" else None
            key_alias = f"JOBS_{alias.upper()}_ACQUIRE_PRIORITY_DESC_DOMAINS" if alias else None
            key_global = "JOBS_ACQUIRE_PRIORITY_DESC_DOMAINS"
            raw = os.getenv(key_backend) or (os.getenv(key_alias) if key_alias else None) or os.getenv(key_global) or ""
            listed = {d.strip().lower() for d in raw.split(",") if d.strip()}
            if dom.lower() in listed:
                return "DESC"
            # Default behavior across domains (including 'chatbooks'):
            # lower numeric value means higher priority -> ASC
            return "ASC"  # noqa: TRY300
        except _JOB_NONCRITICAL_EXCEPTIONS:
            return "ASC"

    def _tie_break_for(self, domain: str | None, backend: str) -> str | None:
        """Return 'fifo' or 'lifo' if explicitly configured, else None for default behavior.

        Env (checked in order):
          - JOBS_{BACKEND}_ACQUIRE_TIE_BREAK_{DOMAIN}
          - JOBS_{BACKEND}_ACQUIRE_TIE_BREAK
          - JOBS_ACQUIRE_TIE_BREAK_{DOMAIN}
          - JOBS_ACQUIRE_TIE_BREAK
        """
        try:
            dom = (domain or "").strip()
            b = (backend or "").strip().lower()
            # Build alias list mirroring _priority_dir_for semantics and adding common variants
            aliases: list[str] = []
            if b in {"pg", "postgres", "postgresql"}:
                # Preserve caller's preferred token first
                base_order = [b, "postgres", "postgresql", "pg"]
                seen = set()
                for t in base_order:
                    if t not in seen:
                        aliases.append(t)
                        seen.add(t)
            else:
                aliases = [b]

            cands: list[str] = []
            # Backend/alias-scoped overrides (domain-specific then general)
            for a in aliases:
                cands.append(f"JOBS_{a.upper()}_ACQUIRE_TIE_BREAK_{dom.upper()}")
            for a in aliases:
                cands.append(f"JOBS_{a.upper()}_ACQUIRE_TIE_BREAK")
            # Global fallbacks (domain-specific then general)
            cands.append(f"JOBS_ACQUIRE_TIE_BREAK_{dom.upper()}")
            cands.append("JOBS_ACQUIRE_TIE_BREAK")
            for k in cands:
                v = os.getenv(k)
                if v:
                    v2 = v.strip().lower()
                    if v2 in {"fifo", "lifo"}:
                        return v2
            return None  # noqa: TRY300
        except _JOB_NONCRITICAL_EXCEPTIONS:
            return None

    @classmethod
    def set_rls_context(cls, *, is_admin: bool, domain_allowlist: str | None, owner_user_id: str | None) -> None:
        try:
            cls._RLS_IS_ADMIN.set(bool(is_admin))
            cls._RLS_DOMAIN_ALLOWLIST.set(domain_allowlist if (domain_allowlist or "").strip() else None)
            cls._RLS_OWNER_USER_ID.set(owner_user_id if (owner_user_id or "").strip() else None)
        except _JOB_NONCRITICAL_EXCEPTIONS:
            pass

    @classmethod
    def clear_rls_context(cls) -> None:
        try:
            cls._RLS_IS_ADMIN.set(False)
            cls._RLS_DOMAIN_ALLOWLIST.set(None)
            cls._RLS_OWNER_USER_ID.set(None)
        except _JOB_NONCRITICAL_EXCEPTIONS:
            pass

    def _should_enforce_ack(self) -> bool:
        if self._enforce_override is not None:
            return bool(self._enforce_override)
        env_force = os.getenv("JOBS_ENFORCE_LEASE_ACK")
        if env_force is not None:
            return JobManager._is_truthy(env_force)
        env_disable = os.getenv("JOBS_DISABLE_LEASE_ENFORCEMENT")
        if env_disable is not None:
            return not JobManager._is_truthy(env_disable)
        return True

    def should_enforce_leases(self) -> bool:
        return self._should_enforce_ack()

    # --- Queue controls (pause/drain) ---
    def _get_queue_flags(self, domain: str, queue: str) -> dict[str, bool]:
        conn = self._connect()
        try:
            if self.backend == "postgres":
                with conn, self._pg_cursor(conn) as cur:
                    cur.execute(
                        "SELECT paused, drain FROM job_queue_controls WHERE domain=%s AND queue=%s", (domain, queue)
                    )
                    row = cur.fetchone()
                    if not row:
                        return {"paused": False, "drain": False}
                    return {"paused": bool(row.get("paused")), "drain": bool(row.get("drain"))}
            else:
                row = conn.execute(
                    "SELECT paused, drain FROM job_queue_controls WHERE domain=? AND queue=?", (domain, queue)
                ).fetchone()
                if not row:
                    return {"paused": False, "drain": False}
                return {"paused": bool(int(row[0] or 0)), "drain": bool(int(row[1] or 0))}
        finally:
            conn.close()

    def set_queue_control(self, domain: str, queue: str, action: str) -> dict[str, bool]:
        action = str(action or "").lower()
        paused = drain = None
        if action == "pause":
            paused, drain = True, False
        elif action == "resume":
            paused, drain = False, False
        elif action == "drain":
            paused, drain = True, True
        else:
            raise ValueError("Unsupported action; expected pause|resume|drain")  # noqa: TRY003
        conn = self._connect()
        _test_mode = _is_test_mode()
        try:
            if self.backend == "postgres":
                with conn:  # noqa: SIM117
                    with self._pg_cursor(conn) as cur:
                        cur.execute(
                            (
                                "INSERT INTO job_queue_controls(domain,queue,paused,drain,updated_at) VALUES(%s,%s,%s,%s,NOW()) "
                                "ON CONFLICT(domain,queue) DO UPDATE SET paused=EXCLUDED.paused, drain=EXCLUDED.drain, updated_at=NOW() RETURNING paused,drain"
                            ),
                            (domain, queue, bool(paused), bool(drain)),
                        )
                        row = cur.fetchone()
                        flags = {"paused": bool(row.get("paused")), "drain": bool(row.get("drain"))}
                        try:
                            set_queue_flag(domain, queue, "paused", flags["paused"])
                            set_queue_flag(domain, queue, "drain", flags["drain"])
                        except _JOB_NONCRITICAL_EXCEPTIONS:
                            pass
                        return flags
            else:
                with conn:
                    conn.execute(
                        (
                            "INSERT INTO job_queue_controls(domain,queue,paused,drain,updated_at) VALUES(?,?,?,?,DATETIME('now')) "
                            "ON CONFLICT(domain,queue) DO UPDATE SET paused=excluded.paused, drain=excluded.drain, updated_at=DATETIME('now')"
                        ),
                        (domain, queue, 1 if paused else 0, 1 if drain else 0),
                    )
                    row = conn.execute(
                        "SELECT paused, drain FROM job_queue_controls WHERE domain=? AND queue=?", (domain, queue)
                    ).fetchone()
                    flags = {"paused": bool(int(row[0] or 0)), "drain": bool(int(row[1] or 0))}
                    try:
                        set_queue_flag(domain, queue, "paused", flags["paused"])
                        set_queue_flag(domain, queue, "drain", flags["drain"])
                    except _JOB_NONCRITICAL_EXCEPTIONS:
                        pass
                    return flags
        finally:
            conn.close()

    def _update_gauges(self, *, domain: str, queue: str, job_type: str | None = None) -> None:
        # Optional lightweight debounce to reduce high-churn writes
        try:
            debounce_ms = int(os.getenv("JOBS_GAUGES_DEBOUNCE_MS", "0") or "0")
        except _JOB_NONCRITICAL_EXCEPTIONS:
            debounce_ms = 0
        if debounce_ms > 0:
            key = (str(domain), str(queue), str(job_type) if job_type is not None else None)
            now = time.time()
            last = JobManager._GAUGE_LAST_TS.get(key)
            if last is not None and (now - last) < (debounce_ms / 1000.0):
                return
            JobManager._GAUGE_LAST_TS[key] = now
        try:
            conn = self._connect()
            try:
                counters_enabled = JobManager._is_truthy(os.getenv("JOBS_COUNTERS_ENABLED", ""))
                if self.backend == "postgres":
                    with self._pg_cursor(conn) as cur:
                        if counters_enabled:
                            cur.execute(
                                "SELECT ready_count, scheduled_count, processing_count FROM job_counters WHERE domain=%s AND queue=%s AND job_type=%s",
                                (domain, queue, job_type),
                            )
                            rowc = cur.fetchone()
                            if rowc:
                                q_ready = int((rowc.get("ready_count") if isinstance(rowc, dict) else 0) or 0)
                                q_sched = int((rowc.get("scheduled_count") if isinstance(rowc, dict) else 0) or 0)
                                p = int((rowc.get("processing_count") if isinstance(rowc, dict) else 0) or 0)
                            else:
                                q_ready = q_sched = p = 0
                        else:
                            # Ready queued jobs use the stable NULL bucket marker.
                            cur.execute(
                                "SELECT COUNT(*) AS c FROM jobs WHERE domain=%s AND queue=%s AND job_type=%s AND status='queued' AND available_at IS NULL",
                                (domain, queue, job_type),
                            )
                            q_ready_row = cur.fetchone()
                            q_ready = int(
                                (q_ready_row.get("c") if isinstance(q_ready_row, dict) else 0)
                                if q_ready_row is not None
                                else 0
                            )
                            # Scheduled queued jobs retain a non-NULL timestamp until transition.
                            cur.execute(
                                "SELECT COUNT(*) AS c FROM jobs WHERE domain=%s AND queue=%s AND job_type=%s AND status='queued' AND available_at IS NOT NULL",
                                (domain, queue, job_type),
                            )
                            q_sched_row = cur.fetchone()
                            q_sched = int(
                                (q_sched_row.get("c") if isinstance(q_sched_row, dict) else 0)
                                if q_sched_row is not None
                                else 0
                            )
                            cur.execute(
                                "SELECT COUNT(*) AS c FROM jobs WHERE domain=%s AND queue=%s AND job_type=%s AND status='processing'",
                                (domain, queue, job_type),
                            )
                            p_row = cur.fetchone()
                            p = int((p_row.get("c") if isinstance(p_row, dict) else 0) if p_row is not None else 0)
                else:
                    if counters_enabled:
                        rowc = conn.execute(
                            "SELECT ready_count, scheduled_count, processing_count FROM job_counters WHERE domain=? AND queue=? AND job_type=?",
                            (domain, queue, job_type),
                        ).fetchone()
                        if rowc:
                            q_ready = int(rowc[0] or 0)
                            q_sched = int(rowc[1] or 0)
                            p = int(rowc[2] or 0)
                        else:
                            q_ready = q_sched = p = 0
                    else:
                        q_ready = int(
                            conn.execute(
                                "SELECT COUNT(*) FROM jobs WHERE domain=? AND queue=? AND job_type=? AND status='queued' AND available_at IS NULL",
                                (domain, queue, job_type),
                            ).fetchone()[0]
                        )
                        q_sched = int(
                            conn.execute(
                                "SELECT COUNT(*) FROM jobs WHERE domain=? AND queue=? AND job_type=? AND status='queued' AND available_at IS NOT NULL",
                                (domain, queue, job_type),
                            ).fetchone()[0]
                        )
                        p = int(
                            conn.execute(
                                "SELECT COUNT(*) FROM jobs WHERE domain=? AND queue=? AND job_type=? AND status='processing'",
                                (domain, queue, job_type),
                            ).fetchone()[0]
                        )
                set_queue_gauges(domain, queue, job_type, q_ready, p, backlog=(q_ready + q_sched), scheduled=q_sched)
            finally:
                conn.close()
        except _JOB_NONCRITICAL_EXCEPTIONS:
            pass

    # --- SLA policies ---
    def upsert_sla_policy(
        self,
        *,
        domain: str,
        queue: str,
        job_type: str,
        max_queue_latency_seconds: int | None = None,
        max_duration_seconds: int | None = None,
        enabled: bool = True,
    ) -> None:
        conn = self._connect()
        _test_mode = _is_test_mode()
        try:
            if self.backend == "postgres":
                with conn:  # noqa: SIM117
                    with self._pg_cursor(conn) as cur:
                        if _test_mode:
                            with contextlib.suppress(_JOB_NONCRITICAL_EXCEPTIONS):
                                logger.info(
                                    f"[JM TEST MUT] upsert_sla_policy domain={domain} queue={queue} job_type={job_type} backend=pg"
                                )
                        cur.execute(
                            (
                                "INSERT INTO job_sla_policies(domain,queue,job_type,max_queue_latency_seconds,max_duration_seconds,enabled,updated_at) "
                                "VALUES(%s,%s,%s,%s,%s,%s,NOW()) ON CONFLICT(domain,queue,job_type) DO UPDATE SET "
                                "max_queue_latency_seconds=EXCLUDED.max_queue_latency_seconds, max_duration_seconds=EXCLUDED.max_duration_seconds, enabled=EXCLUDED.enabled, updated_at=NOW()"
                            ),
                            (domain, queue, job_type, max_queue_latency_seconds, max_duration_seconds, enabled),
                        )
            else:
                with conn:
                    conn.execute(
                        (
                            "INSERT INTO job_sla_policies(domain,queue,job_type,max_queue_latency_seconds,max_duration_seconds,enabled,updated_at) "
                            "VALUES(?,?,?,?,?, ?, DATETIME('now')) ON CONFLICT(domain,queue,job_type) DO UPDATE SET "
                            "max_queue_latency_seconds=excluded.max_queue_latency_seconds, max_duration_seconds=excluded.max_duration_seconds, enabled=excluded.enabled, updated_at=DATETIME('now')"
                        ),
                        (domain, queue, job_type, max_queue_latency_seconds, max_duration_seconds, 1 if enabled else 0),
                    )
        finally:
            conn.close()

    def delete_sla_policy(
        self,
        *,
        domain: str,
        queue: str,
        job_type: str,
    ) -> bool:
        """Delete an SLA policy. Returns True if a row was deleted."""
        conn = self._connect()
        try:
            if self.backend == "postgres":
                with conn, self._pg_cursor(conn) as cur:
                    cur.execute(
                        "DELETE FROM job_sla_policies WHERE domain=%s AND queue=%s AND job_type=%s",
                        (domain, queue, job_type),
                    )
                    return (cur.rowcount or 0) > 0
            else:
                with conn:
                    cur = conn.execute(
                        "DELETE FROM job_sla_policies WHERE domain=? AND queue=? AND job_type=?",
                        (domain, queue, job_type),
                    )
                    return (cur.rowcount or 0) > 0
        finally:
            conn.close()

    def _get_sla_policy(self, domain: str, queue: str, job_type: str) -> dict[str, Any] | None:
        conn = self._connect()
        try:
            if self.backend == "postgres":
                with self._pg_cursor(conn) as cur:
                    cur.execute(
                        "SELECT * FROM job_sla_policies WHERE domain=%s AND queue=%s AND job_type=%s",
                        (domain, queue, job_type),
                    )
                    row = cur.fetchone()
                    return dict(row) if row else None
            else:
                row = conn.execute(
                    "SELECT * FROM job_sla_policies WHERE domain=? AND queue=? AND job_type=?",
                    (domain, queue, job_type),
                ).fetchone()
                return dict(row) if row else None
        finally:
            conn.close()

    def _duration_sla_breach(
        self,
        job: dict[str, Any],
    ) -> tuple[float, float] | None:
        """Return duration and threshold when a completed job breached its SLA."""

        try:
            policy = self._get_sla_policy(
                str(job.get("domain")),
                str(job.get("queue")),
                str(job.get("job_type")),
            )
            if not (
                policy
                and policy.get("enabled") in (True, 1)
                and policy.get("max_duration_seconds") is not None
            ):
                return None
            started_at = _as_utc_datetime(
                job.get("started_at") or job.get("acquired_at")
            )
            now = _as_utc_datetime(self._clock.now_utc())
            if started_at is None or now is None:
                return None
            duration = max(0.0, (now - started_at).total_seconds())
            threshold = float(policy["max_duration_seconds"])
            if duration <= threshold:
                return None
            return duration, threshold
        except _JOB_NONCRITICAL_EXCEPTIONS:
            return None

    def _stage_completion_sla_breach(
        self,
        executor: Any,
        *,
        job_id: int,
        job: dict[str, Any],
        outbox_enabled: bool,
        side_effects: list[tuple[Any, tuple[Any, ...], dict[str, Any]]],
    ) -> None:
        """Stage optional SLA rows in the completion transaction.

        A savepoint keeps attachment or SLA-outbox failures best-effort without
        allowing them to abort the durable core completion.
        """

        breach = self._duration_sla_breach(job)
        if breach is None:
            return
        duration, threshold = breach
        event_job = {
            "id": int(job_id),
            "domain": job.get("domain"),
            "queue": job.get("queue"),
            "job_type": job.get("job_type"),
            "owner_user_id": job.get("owner_user_id"),
            "request_id": job.get("request_id"),
            "trace_id": job.get("trace_id"),
        }
        attrs = {
            "kind": "duration",
            "value": float(duration),
            "threshold": float(threshold),
        }
        message = (
            f"SLA breach: duration={duration:.3f}s > {threshold:.3f}s"
        )
        executor.execute("SAVEPOINT job_completion_sla")
        try:
            if self.backend == "postgres":
                executor.execute(
                    "INSERT INTO job_attachments(job_id,kind,content_text) "
                    "VALUES(%s,%s,%s)",
                    (int(job_id), "tag", message),
                )
            else:
                executor.execute(
                    "INSERT INTO job_attachments(job_id,kind,content_text) "
                    "VALUES(?,?,?)",
                    (int(job_id), "tag", message),
                )
            if outbox_enabled:
                _insert_lifecycle_event(
                    executor,
                    backend=self.backend,
                    event_type="job.sla_breached",
                    job=event_job,
                    attrs=attrs,
                )
        except _JOB_NONCRITICAL_EXCEPTIONS as exc:
            executor.execute("ROLLBACK TO SAVEPOINT job_completion_sla")
            executor.execute("RELEASE SAVEPOINT job_completion_sla")
            side_effects.append(
                (
                    _log_optional_sla_persistence_failure,
                    (int(job_id), type(exc).__name__),
                    {},
                )
            )
            return
        executor.execute("RELEASE SAVEPOINT job_completion_sla")
        _queue_lifecycle_event_observer(
            side_effects,
            event_type="job.sla_breached",
            job=event_job,
            attrs=attrs,
        )
        side_effects.append(
            (
                increment_sla_breach,
                (
                    {
                        "domain": job.get("domain"),
                        "queue": job.get("queue"),
                        "job_type": job.get("job_type"),
                    },
                    "duration",
                ),
                {},
            )
        )

    def _record_sla_breach(
        self,
        job_id: int,
        domain: str,
        queue: str,
        job_type: str,
        kind: str,
        value: float,
        threshold: float,
        *,
        conn: Any | None = None,
    ) -> None:
        try:
            own_conn = conn is None
            if own_conn:
                conn = self._connect()
            try:
                msg = f"SLA breach: {kind}={value:.3f}s > {threshold:.3f}s"
                if self.backend == "postgres":
                    if own_conn:
                        with conn, self._pg_cursor(conn) as cur:
                            cur.execute(
                                "INSERT INTO job_attachments(job_id,kind,content_text) VALUES(%s,%s,%s)",
                                (int(job_id), "tag", msg),
                            )
                    else:
                        with self._pg_cursor(conn) as cur:
                            cur.execute(
                                "INSERT INTO job_attachments(job_id,kind,content_text) VALUES(%s,%s,%s)",
                                (int(job_id), "tag", msg),
                            )
                else:
                    if own_conn:
                        with conn:
                            conn.execute(
                                "INSERT INTO job_attachments(job_id,kind,content_text) VALUES(?,?,?)",
                                (int(job_id), "tag", msg),
                            )
                    else:
                        conn.execute(
                            "INSERT INTO job_attachments(job_id,kind,content_text) VALUES(?,?,?)",
                            (int(job_id), "tag", msg),
                        )
                with contextlib.suppress(_JOB_NONCRITICAL_EXCEPTIONS):
                    emit_job_event(
                        "job.sla_breached",
                        job={"id": int(job_id), "domain": domain, "queue": queue, "job_type": job_type},
                        attrs={"kind": kind, "value": float(value), "threshold": float(threshold)},
                    )
                with contextlib.suppress(_JOB_NONCRITICAL_EXCEPTIONS):
                    increment_sla_breach({"domain": domain, "queue": queue, "job_type": job_type}, kind)
            finally:
                if own_conn and conn is not None:
                    conn.close()
        except _JOB_NONCRITICAL_EXCEPTIONS:
            pass

    # --- Encryption helpers ---
    def _should_encrypt(self, domain: str | None) -> bool:
        try:
            if JobManager._is_truthy(os.getenv("JOBS_ENCRYPT", "")):
                return True
            if domain:  # noqa: SIM102
                if JobManager._is_truthy(os.getenv(f"JOBS_ENCRYPT_{str(domain).upper()}", "")):
                    return True
        except _JOB_NONCRITICAL_EXCEPTIONS:
            pass
        return False

    def _maybe_encrypt_json(self, obj: dict[str, Any] | None, domain: str | None) -> dict[str, Any] | None:
        if obj is None:
            return None
        try:
            if self._should_encrypt(domain):
                env = encrypt_json_blob(obj)
                if env:
                    return {"_encrypted": env}
        except _JOB_NONCRITICAL_EXCEPTIONS:
            pass
        return obj

    def _maybe_decrypt_json(
        self,
        obj: Any | None,
        *,
        fail_on_error: bool = False,
        field_name: str = "JSON value",
    ) -> Any | None:
        env: dict[str, Any] | None = None
        try:
            if isinstance(obj, dict):
                if obj.get("_enc") == "aesgcm:v1":
                    env = obj
                elif isinstance(obj.get("_encrypted"), dict):
                    env = obj.get("_encrypted")
                if env:
                    dec = decrypt_json_blob(env)  # returns dict or None
                    if dec is not None:
                        return dec
                    if fail_on_error:
                        raise JobPayloadDecryptionError(field_name)
                    return obj
        except JobPayloadDecryptionError:
            raise
        except _JOB_NONCRITICAL_EXCEPTIONS:
            if fail_on_error and env is not None:
                raise JobPayloadDecryptionError(field_name) from None
            return obj
        return obj

    @staticmethod
    def _parse_json_value(value: Any) -> Any:
        """Normalize JSON-ish values from DB rows into Python objects."""
        if value is None:
            return None
        if isinstance(value, (dict, list)):
            return value
        if isinstance(value, memoryview):
            value = value.tobytes()
        if isinstance(value, (bytes, bytearray)):
            try:
                return JobManager._parse_json_value(bytes(value).decode("utf-8"))
            except _JOB_NONCRITICAL_EXCEPTIONS:
                return value
        if isinstance(value, str):
            try:
                return json.loads(value)
            except _JOB_NONCRITICAL_EXCEPTIONS:
                return value
        return value

    @staticmethod
    def _decode_archive_blob(value: Any) -> Any:
        """Decode compressed archive payload/result values."""
        if value is None:
            return None
        if isinstance(value, memoryview):
            value = value.tobytes()
        if isinstance(value, (bytes, bytearray)):
            try:
                import gzip

                decoded = gzip.decompress(bytes(value)).decode("utf-8")
                return JobManager._parse_json_value(decoded)
            except _JOB_NONCRITICAL_EXCEPTIONS:
                return JobManager._parse_json_value(value)
        if isinstance(value, str) and value.startswith("gzip64:"):
            try:
                import base64
                import gzip

                payload = value[len("gzip64:") :]
                decoded = gzip.decompress(base64.b64decode(payload)).decode("utf-8")
                return JobManager._parse_json_value(decoded)
            except _JOB_NONCRITICAL_EXCEPTIONS:
                return JobManager._parse_json_value(value)
        return JobManager._parse_json_value(value)

    # --- Secret hygiene helpers ---
    def _secret_patterns(self) -> tuple[list[re.Pattern], list[str]]:
        """Return compiled regex patterns and sensitive keys for secret detection."""
        # Default key denylist (lowercased)
        default_keys = [
            "api_key",
            "apikey",
            "x-api-key",
            "authorization",
            "auth",
            "password",
            "pass",
            "secret",
            "token",
            "access_token",
            "refresh_token",
            "session",
            "cookie",
            "jwt",
        ]
        extra_keys = os.getenv("JOBS_SECRET_DENY_KEYS", "").strip()
        if extra_keys:
            default_keys.extend([k.strip().lower() for k in extra_keys.split(",") if k.strip()])
        # Default regexes for common tokens
        defaults = [
            r"sk-[A-Za-z0-9]{20,}",  # OpenAI-like
            r"AKIA[0-9A-Z]{16}",  # AWS Access Key ID
            r"ghp_[A-Za-z0-9]{36}",  # GitHub PAT
            r"eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}",  # JWT
            r"AIza[0-9A-Za-z\-_]{35}",  # Google API key
            r"xox[abpr]-[0-9A-Za-z-]{10,}",  # Slack tokens
        ]
        extra = os.getenv("JOBS_SECRET_PATTERNS", "").strip()
        if extra:
            defaults.extend([p.strip() for p in extra.split(";") if p.strip()])
        try:
            compiled = [re.compile(p, re.IGNORECASE) for p in defaults]
        except _JOB_NONCRITICAL_EXCEPTIONS:
            compiled = [re.compile(p) for p in defaults if p]
        return compiled, default_keys

    def _scan_and_redact_secrets(self, obj: Any) -> tuple[Any, bool, list[str]]:
        """Scan object for secrets. Optionally redact based on env flags.

        Returns (possibly-redacted-object, found_any, findings).
        """
        redact = JobManager._is_truthy(os.getenv("JOBS_SECRET_REDACT", ""))
        patterns, deny_keys = self._secret_patterns()
        findings: list[str] = []

        def _is_secret_str(s: str) -> bool:
            try:
                return any(pat.search(s or "") for pat in patterns)
            except _JOB_NONCRITICAL_EXCEPTIONS:
                return False

        def _recurse(x: Any, key_path: str = "") -> Any:
            nonlocal findings
            try:
                if isinstance(x, dict):
                    out: dict[str, Any] = {}
                    for k, v in x.items():
                        lk = str(k).lower()
                        kp = f"{key_path}.{k}" if key_path else str(k)
                        if lk in deny_keys:
                            findings.append(kp)
                            out[k] = "***REDACTED***" if redact else v
                        else:
                            out[k] = _recurse(v, kp)
                    return out
                if isinstance(x, list):
                    return [_recurse(v, f"{key_path}[{i}]") for i, v in enumerate(x)]
                if isinstance(x, str):
                    if _is_secret_str(x):
                        findings.append(key_path or "<root>")
                        return "***REDACTED***" if redact else x
                    return x
                return x  # noqa: TRY300
            except _JOB_NONCRITICAL_EXCEPTIONS:
                return x

        new_obj = _recurse(obj)
        return new_obj, bool(findings), findings

    # --- Quotas helpers ---
    def _quota_get(self, base: str, domain: str | None, user_id: str | None) -> int:
        def _parse(v: str | None) -> int:
            try:
                return int(str(v or "").strip() or 0)
            except _JOB_NONCRITICAL_EXCEPTIONS:
                return 0

        dom = str(domain or "").upper()
        uid = str(user_id or "").strip()
        # Precedence: domain+user, user global, domain global, global
        if dom and uid:
            v = os.getenv(f"{base}_{dom}_USER_{uid}")
            if v is not None:
                return _parse(v)
        if uid:
            v = os.getenv(f"{base}_USER_{uid}")
            if v is not None:
                return _parse(v)
        if dom:
            v = os.getenv(f"{base}_{dom}")
            if v is not None:
                return _parse(v)
        return _parse(os.getenv(base))

    @staticmethod
    def _build_create_job_command(
        *,
        domain: str,
        queue: str,
        job_type: str,
        payload: dict[str, Any],
        owner_user_id: str | None,
        project_id: int | None,
        batch_group: str | None,
        priority: int,
        max_retries: int,
        available_at: datetime | None,
        idempotency_key: str | None,
        request_id: str | None,
        trace_id: str | None,
    ) -> CreateJobCommand:
        """Build the backend-neutral create command after facade validation."""

        return CreateJobCommand(
            domain=domain,
            queue=queue,
            job_type=job_type,
            payload=payload,
            owner_user_id=owner_user_id,
            idempotency_key=idempotency_key,
            priority=priority,
            max_retries=max_retries,
            available_at=available_at,
            project_id=project_id,
            batch_group=batch_group,
            request_id=request_id,
            trace_id=trace_id,
        )

    @staticmethod
    def _map_admission_result(result: AdmissionResult) -> dict[str, Any]:
        """Map an admission result to the public create_job return row."""

        if result.outcome is OperationOutcome.ADMISSION_REJECTED:
            if result.admission_rejection_reason is AdmissionRejectionReason.QUOTA_EXCEEDED:
                raise ValueError(result.message or "Quota exceeded")  # noqa: TRY003
            raise ValueError(result.message or "Admission rejected")  # noqa: TRY003
        if result.row is None:
            raise RuntimeError("Job admission did not return a row")  # noqa: TRY003
        return result.row

    def _validate_slides_generation_admission(
        self,
        conn: Any,
        row: dict[str, Any],
        *,
        owner_user_id: str,
        idempotency_key: str,
    ) -> None:
        """Reject an idempotent replay that is not the exact Slides authority."""

        valid = (
            str(row.get("uuid") or "").strip() != ""
            and row.get("domain") == _SLIDES_GENERATION_DOMAIN
            and row.get("queue") == _SLIDES_GENERATION_QUEUE
            and row.get("job_type") == _SLIDES_GENERATION_JOB_TYPE
            and row.get("owner_user_id") == owner_user_id
            and row.get("idempotency_key") == idempotency_key
        )
        if valid:
            return
        self._record_slides_generation_diagnostic(
            conn,
            code="ambiguous_generation_legacy_row",
            count=1,
        )
        raise SlidesGenerationJobsUnavailableError(
            "presentation.generate correlation is unsafe"
        )

    def _emit_create_side_effects(
        self,
        result: AdmissionResult,
        *,
        backend: str,
        idempotency_key: str | None,
    ) -> None:
        """Emit create metrics and facade-owned event/audit side effects."""

        row = result.row
        if row is None:
            return

        if result.inserted:
            _safe_increment_created_metric(
                domain=str(row.get("domain")),
                queue=str(row.get("queue")),
                job_type=str(row.get("job_type")),
            )

        for event in result.durable_events:
            if event.get("event_type") != "job.created":
                continue
            attrs = dict(event.get("attrs") or {})
            request_id = event.get("request_id")
            trace_id = event.get("trace_id")
            outbox_enabled = JobManager._is_truthy(os.getenv("JOBS_EVENTS_OUTBOX", ""))

            def _run_create_side_effect(
                operation: str,
                job: dict[str, Any],
                emit_func: Any,
                event_attrs: dict[str, Any],
            ) -> None:
                try:
                    emit_func("job.created", job=job, attrs=event_attrs)
                except _JOB_NONCRITICAL_EXCEPTIONS as exc:
                    logger.warning(
                        "Non-critical Jobs create side effect {} failed for backend={} job_id={} domain={} queue={} job_type={}: {}",
                        operation,
                        backend,
                        job.get("id"),
                        job.get("domain"),
                        job.get("queue"),
                        job.get("job_type"),
                        exc,
                    )

            if backend == "sqlite":
                emitted_job = {**row, "request_id": request_id, "trace_id": trace_id}
                if idempotency_key:
                    if outbox_enabled:
                        _run_create_side_effect(
                            "submit_job_audit_event",
                            emitted_job,
                            submit_job_audit_event,
                            attrs,
                        )
                    else:
                        _run_create_side_effect(
                            "emit_job_event",
                            emitted_job,
                            emit_job_event,
                            attrs,
                        )
                    continue

                if not outbox_enabled:
                    _run_create_side_effect(
                        "emit_job_event",
                        emitted_job,
                        emit_job_event,
                        attrs,
                    )
                else:
                    _run_create_side_effect(
                        "submit_job_audit_event",
                        emitted_job,
                        submit_job_audit_event,
                        attrs,
                    )
                continue

            emitted_job = {**row, "request_id": request_id, "trace_id": trace_id}
            if not outbox_enabled:
                _run_create_side_effect(
                    "emit_job_event",
                    emitted_job,
                    emit_job_event,
                    attrs,
                )
            else:
                _run_create_side_effect(
                    "submit_job_audit_event",
                    emitted_job,
                    submit_job_audit_event,
                    attrs,
                )

    # --- Advisory lock helpers (Postgres) ---
    def _pg_advisory_key(self, *parts: str) -> int:
        """Compute a signed 64-bit advisory lock key from parts."""
        s = (":".join(["jobs"] + [p or "" for p in parts])).encode("utf-8", "ignore")
        h = int.from_bytes(hashlib.sha1(s, usedforsecurity=False).digest()[:8], "big", signed=False)
        # Fit into signed BIGINT range used by pg advisory locks
        if h >= 2**63:
            h = h - 2**63
        return int(h)

    def _pg_try_advisory_lock(self, key: int) -> bool:
        if self.backend != "postgres":
            return True
        conn = self._connect()
        try:
            with self._pg_cursor(conn) as cur:
                cur.execute("SELECT pg_try_advisory_lock(%s)", (int(key),))
                row = cur.fetchone()
                return bool(row[0]) if row is not None else False
        finally:
            with contextlib.suppress(_JOB_NONCRITICAL_EXCEPTIONS):
                conn.close()

    def _pg_advisory_unlock(self, key: int) -> None:
        if self.backend != "postgres":
            return
        conn = self._connect()
        try:
            with self._pg_cursor(conn) as cur, contextlib.suppress(_JOB_NONCRITICAL_EXCEPTIONS):
                cur.execute("SELECT pg_advisory_unlock(%s)", (int(key),))
        finally:
            with contextlib.suppress(_JOB_NONCRITICAL_EXCEPTIONS):
                conn.close()

    # CRUD / queries
    def replay_idempotent_operation(
        self,
        command: IdempotentOperationCommand,
    ) -> IdempotentOperationAdmission | None:
        """Read an exact owner-scoped receipt without applying admission policy."""

        conn = self._connect()
        try:
            if self.backend == "postgres":
                return _postgres_replay_idempotent_operation(
                    conn,
                    self._pg_cursor,
                    command,
                )
            return _sqlite_replay_idempotent_operation(conn, command)
        finally:
            conn.close()

    def admit_idempotent_operation(
        self,
        command: IdempotentOperationCommand,
    ) -> IdempotentOperationAdmission:
        """Atomically admit or replay one owner-scoped user operation."""

        job = command.job
        replay = self.replay_idempotent_operation(command)
        if replay is not None:
            return replay

        allowed_queues = self._get_allowed_queues(job.domain)
        if job.queue not in allowed_queues:
            raise ValueError(  # noqa: TRY003
                f"Queue '{job.queue}' not allowed for domain '{job.domain}'. "
                f"Allowed: {allowed_queues}"
            )

        allowed_job_types: list[str] = []
        env_all = os.getenv("JOBS_ALLOWED_JOB_TYPES", "").strip()
        if env_all:
            allowed_job_types.extend(
                item.strip() for item in env_all.split(",") if item.strip()
            )
        env_domain = os.getenv(
            f"JOBS_ALLOWED_JOB_TYPES_{str(job.domain).upper()}",
            "",
        ).strip()
        if env_domain:
            allowed_job_types.extend(
                item.strip() for item in env_domain.split(",") if item.strip()
            )
        if allowed_job_types and job.job_type not in allowed_job_types:
            raise ValueError(  # noqa: TRY003
                f"Job type '{job.job_type}' not allowed for domain "
                f"'{job.domain}'. Allowed: {sorted(set(allowed_job_types))}"
            )

        now = self._clock.now_utc()
        payload = job.payload
        try:
            cleaned, found, where = self._scan_and_redact_secrets(payload)
        except _JOB_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug(
                "Jobs secret hygiene scan error during idempotent admission: {}",
                type(exc).__name__,
            )
        else:
            if found and JobManager._is_truthy(os.getenv("JOBS_SECRET_REJECT", "")):
                suffix = "..." if len(where) > 3 else ""
                raise ValueError(  # noqa: TRY003
                    f"Payload appears to contain secrets at: {where[:3]}{suffix}"
                )
            if found:
                payload = cleaned

        payload = self._maybe_encrypt_json(payload, job.domain)
        payload_json = json.dumps(payload)
        payload_bytes = len(payload_json.encode("utf-8"))
        max_bytes = int(os.getenv("JOBS_MAX_JSON_BYTES", "1048576") or "1048576")
        if payload_bytes > max_bytes:
            if JobManager._is_truthy(os.getenv("JOBS_JSON_TRUNCATE", "")):
                payload = {"_truncated": True, "len_bytes": payload_bytes}
            else:
                raise ValueError(  # noqa: TRY003
                    f"Payload too large: {payload_bytes} bytes > limit {max_bytes}"
                )

        if job.owner_user_id and _fair_share_enabled():
            scheduler = _get_fair_share()
            active_count = self._count_active_jobs_for_user(job.owner_user_id)
            if not scheduler.can_submit(job.owner_user_id, active_count):
                raise BadRequestError(
                    f"User {job.owner_user_id} has reached the maximum concurrent "
                    f"job limit ({scheduler.max_per_user})"
                )
            fair_priority = scheduler.calculate_priority(
                job.owner_user_id,
                active_count,
            )
            job = replace(
                job,
                priority=min(
                    job.priority,
                    self._map_fair_share_score_to_priority(fair_priority),
                ),
            )

        if not job.trace_id:
            job = replace(job, trace_id=str(_uuid.uuid4()))
        if payload is not job.payload:
            job = replace(job, payload=payload)
        command = replace(command, job=job)

        conn = self._connect()
        try:
            admission_kwargs = {
                "command": command,
                "uuid_value": str(_uuid.uuid4()),
                "now": now,
                "max_queued_quota": self._quota_get(
                    "JOBS_QUOTA_MAX_QUEUED",
                    job.domain,
                    job.owner_user_id,
                ),
                "submits_per_minute_quota": self._quota_get(
                    "JOBS_QUOTA_SUBMITS_PER_MIN",
                    job.domain,
                    job.owner_user_id,
                ),
                "counters_enabled": JobManager._is_truthy(
                    os.getenv("JOBS_COUNTERS_ENABLED", "")
                ),
            }
            if self.backend == "postgres":
                result = _postgres_admit_idempotent_operation(
                    conn,
                    self._pg_cursor,
                    **admission_kwargs,
                )
            else:
                result = _sqlite_admit_idempotent_operation(
                    conn,
                    **admission_kwargs,
                )
        finally:
            conn.close()

        if result.disposition is IdempotentOperationDisposition.CREATED:
            _safe_increment_created_metric(
                domain=job.domain,
                queue=job.queue,
                job_type=job.job_type,
            )
        with contextlib.suppress(_JOB_NONCRITICAL_EXCEPTIONS):
            self._update_gauges(
                domain=job.domain,
                queue=job.queue,
                job_type=job.job_type,
            )
        with contextlib.suppress(_JOB_NONCRITICAL_EXCEPTIONS):
            self._assert_invariants(result.job)
        return result

    def create_job(
        self,
        *,
        domain: str,
        queue: str,
        job_type: str,
        payload: dict[str, Any],
        owner_user_id: str | None,
        project_id: int | None = None,
        batch_group: str | None = None,
        priority: int = 5,
        max_retries: int = 3,
        available_at: datetime | None = None,
        idempotency_key: str | None = None,
        request_id: str | None = None,
        trace_id: str | None = None,
    ) -> dict[str, Any]:
        """Create a new job.

        Args:
            domain: Logical domain (e.g., "chatbooks", "prompt_studio").
            queue: Queue name within the domain.
            job_type: Free-form job type string.
            payload: Opaque payload to be interpreted by the worker.
            owner_user_id: Owner of the job for scoping/quotas.
            project_id: Optional project association.
            batch_group: Optional batch identifier for client-managed grouping.
            priority: Lower number means higher priority (default 5).
            max_retries: Maximum automatic retries on failure.
            available_at: Optional schedule time before the job becomes acquirable.
            idempotency_key: If provided, duplicate creates return the same row.

        Returns:
            A dict representing the created (or existing, if idempotent) job row.
        """
        slides_generation = _is_slides_generation_scope(domain, queue, job_type)
        if slides_generation:
            if not isinstance(owner_user_id, str) or not owner_user_id.strip():
                raise ValueError("presentation.generate jobs require owner_user_id")
            if not isinstance(idempotency_key, str) or not idempotency_key.strip():
                raise ValueError("presentation.generate jobs require idempotency_key")
            existing = self._serialized_slides_generation_replay(
                owner_user_id=owner_user_id,
                idempotency_key=idempotency_key,
            )
            if existing is not None:
                return existing

        # Queue name policy
        allowed_queues = self._get_allowed_queues(domain)
        if queue not in allowed_queues:
            error = ValueError(
                f"Queue '{queue}' not allowed for domain '{domain}'. Allowed: {allowed_queues}"
            )
            if slides_generation:
                existing = self._serialized_slides_generation_replay(
                    owner_user_id=owner_user_id,
                    idempotency_key=idempotency_key,
                    rejection=error,
                )
                if existing is not None:
                    return existing
            raise error  # noqa: TRY003

        # Fair-share scheduling: enforce per-user concurrency limits and adjust priority
        if owner_user_id and _fair_share_enabled():
            try:
                scheduler = _get_fair_share()
                active_count = self._count_active_jobs_for_user(owner_user_id)
                if not scheduler.can_submit(owner_user_id, active_count):
                    raise BadRequestError(
                        f"User {owner_user_id} has reached the maximum concurrent job limit "
                        f"({scheduler.max_per_user})"
                    )
                fair_priority = scheduler.calculate_priority(owner_user_id, active_count)
                fair_priority_mapped = self._map_fair_share_score_to_priority(fair_priority)
                priority = min(priority, fair_priority_mapped)
            except BadRequestError as exc:
                if slides_generation:
                    existing = self._serialized_slides_generation_replay(
                        owner_user_id=owner_user_id,
                        idempotency_key=idempotency_key,
                        rejection=exc,
                    )
                    if existing is not None:
                        return existing
                raise
            except _JOB_NONCRITICAL_EXCEPTIONS as _fs_exc:
                logger.warning(f"Fair-share scheduling check skipped: {_fs_exc}")

        # Secret hygiene (reject/redact)
        try:
            cleaned, found, where = self._scan_and_redact_secrets(payload)
        except _JOB_NONCRITICAL_EXCEPTIONS as _sec_e:
            logger.debug(
                "Jobs secret hygiene scan error: {}",
                type(_sec_e).__name__,
            )
        else:
            if found and JobManager._is_truthy(os.getenv("JOBS_SECRET_REJECT", "")):
                suffix = "..." if len(where) > 3 else ""
                raise ValueError(
                    f"Payload appears to contain secrets at: {where[:3]}{suffix}"
                )  # noqa: TRY003 - public rejection text is an API compatibility contract.
            if found:
                payload = cleaned

        # JSON payload size cap
        max_bytes = int(os.getenv("JOBS_MAX_JSON_BYTES", "1048576") or "1048576")
        truncate = JobManager._is_truthy(os.getenv("JOBS_JSON_TRUNCATE", ""))
        # Optional encryption at rest for payload
        payload = self._maybe_encrypt_json(payload, domain)
        try:
            payload_json = json.dumps(payload)
        except (TypeError, ValueError) as exc:
            if slides_generation:
                existing = self._serialized_slides_generation_replay(
                    owner_user_id=owner_user_id,
                    idempotency_key=idempotency_key,
                    rejection=exc,
                )
                if existing is not None:
                    return existing
            raise
        payload_bytes = len(payload_json.encode("utf-8"))
        if payload_bytes > max_bytes:
            if truncate:
                payload = {"_truncated": True, "len_bytes": payload_bytes}
                payload_json = json.dumps(payload)
                with contextlib.suppress(_JOB_NONCRITICAL_EXCEPTIONS):
                    increment_json_truncated({"domain": domain, "queue": queue, "job_type": job_type}, "payload")
            else:
                error = ValueError(
                    f"Payload too large: {payload_bytes} bytes > limit {max_bytes}"
                )
                if slides_generation:
                    existing = self._serialized_slides_generation_replay(
                        owner_user_id=owner_user_id,
                        idempotency_key=idempotency_key,
                        rejection=error,
                    )
                    if existing is not None:
                        return existing
                raise error  # noqa: TRY003

        # Note: completion_token enforcement applies to finalize paths (complete/fail), not creation.
        conn = self._connect()
        try:
            try:
                with job_span(
                    "job.create",
                    job={"uuid": None, "domain": domain, "queue": queue, "job_type": job_type},
                    attrs={"idempotency_key": idempotency_key},
                ):
                    pass
            except _JOB_NONCRITICAL_EXCEPTIONS:
                pass
            # Use consistent clock
            _now_dt = self._clock.now_utc()
            uuid_val = str(_uuid.uuid4())
            if not trace_id:
                try:
                    trace_id = str(_uuid.uuid4())
                except _JOB_NONCRITICAL_EXCEPTIONS:
                    trace_id = None
            # Ensure PG receives timezone-aware timestamps
            avail_param = available_at
            if avail_param is not None and getattr(avail_param, "tzinfo", None) is None:
                avail_param = avail_param.replace(tzinfo=_tz.utc)
            # SQLite persists timestamp strings without timezone; normalize aware
            # datetimes to UTC naive to preserve the correct instant.
            avail_param_sqlite = avail_param
            if avail_param_sqlite is not None and getattr(avail_param_sqlite, "tzinfo", None) is not None:
                avail_param_sqlite = avail_param_sqlite.astimezone(_tz.utc).replace(tzinfo=None)
            # Optional job_type allowlist
            allowed_job_types: list[str] = []
            env_all = os.getenv("JOBS_ALLOWED_JOB_TYPES", "").strip()
            if env_all:
                allowed_job_types.extend([x.strip() for x in env_all.split(",") if x.strip()])
            if domain:
                env_dom = os.getenv(f"JOBS_ALLOWED_JOB_TYPES_{str(domain).upper()}", "").strip()
                if env_dom:
                    allowed_job_types.extend([x.strip() for x in env_dom.split(",") if x.strip()])
            if allowed_job_types and job_type not in allowed_job_types:
                error = ValueError(
                    f"Job type '{job_type}' not allowed for domain '{domain}'. Allowed: {sorted(set(allowed_job_types))}"
                )
                if slides_generation:
                    existing = self._serialized_slides_generation_replay(
                        owner_user_id=owner_user_id,
                        idempotency_key=idempotency_key,
                        rejection=error,
                    )
                    if existing is not None:
                        return existing
                raise error  # noqa: TRY003

            if self.backend == "postgres":
                command = self._build_create_job_command(
                    domain=domain,
                    queue=queue,
                    job_type=job_type,
                    payload=payload,
                    owner_user_id=owner_user_id,
                    project_id=project_id,
                    batch_group=batch_group,
                    priority=priority,
                    max_retries=max_retries,
                    available_at=avail_param,
                    idempotency_key=idempotency_key,
                    request_id=request_id,
                    trace_id=trace_id,
                )
                result = _postgres_create_job_admission(
                    conn,
                    self._pg_cursor,
                    command=command,
                    uuid_value=uuid_val,
                    now=_now_dt,
                    max_queued_quota=self._quota_get("JOBS_QUOTA_MAX_QUEUED", domain, owner_user_id),
                    submits_per_minute_quota=self._quota_get("JOBS_QUOTA_SUBMITS_PER_MIN", domain, owner_user_id),
                    counters_enabled=JobManager._is_truthy(os.getenv("JOBS_COUNTERS_ENABLED", "")),
                    advisory_xact_lock_key=(
                        self._pg_advisory_key(
                            *_SLIDES_GENERATION_CORRELATION_LOCK_PARTS
                        )
                        if slides_generation
                        else None
                    ),
                    pre_admission_lookup=(
                        lambda cur: self._lookup_ready_slides_generation_job_in_connection(
                            conn,
                            owner_user_id=str(owner_user_id),
                            idempotency_key=str(idempotency_key),
                            cursor=cur,
                        )
                        if slides_generation
                        else None
                    ),
                )
                d = self._map_admission_result(result)
                if slides_generation:
                    self._validate_slides_generation_admission(
                        conn,
                        d,
                        owner_user_id=str(owner_user_id),
                        idempotency_key=str(idempotency_key),
                    )
                try:
                    pol = self._get_sla_policy(d.get("domain"), d.get("queue"), d.get("job_type"))
                    if pol and (pol.get("enabled") in (True, 1)):
                        ca = _parse_dt(d.get("acquired_at"))
                        cr = _parse_dt(d.get("created_at")) if d.get("created_at") else None
                        if ca and cr and (pol.get("max_queue_latency_seconds") is not None):
                            qlat = max(0.0, (ca - cr).total_seconds())
                            if qlat > float(pol.get("max_queue_latency_seconds")):
                                self._record_sla_breach(
                                    int(d.get("id")),
                                    str(d.get("domain")),
                                    str(d.get("queue")),
                                    str(d.get("job_type")),
                                    "queue_latency",
                                    qlat,
                                    float(pol.get("max_queue_latency_seconds")),
                                    conn=conn,
                                )
                except _JOB_NONCRITICAL_EXCEPTIONS:
                    pass
                with contextlib.suppress(_JOB_NONCRITICAL_EXCEPTIONS):
                    self._assert_invariants(d)
                self._emit_create_side_effects(result, backend="postgres", idempotency_key=idempotency_key)
                return d
            else:
                command = self._build_create_job_command(
                    domain=domain,
                    queue=queue,
                    job_type=job_type,
                    payload=payload,
                    owner_user_id=owner_user_id,
                    project_id=project_id,
                    batch_group=batch_group,
                    priority=priority,
                    max_retries=max_retries,
                    available_at=avail_param_sqlite,
                    idempotency_key=idempotency_key,
                    request_id=request_id,
                    trace_id=trace_id,
                )
                for attempt in range(2):
                    try:
                        result = _sqlite_create_job_admission(
                            conn,
                            command=command,
                            uuid_value=uuid_val,
                            now=_now_dt,
                            max_queued_quota=self._quota_get("JOBS_QUOTA_MAX_QUEUED", domain, owner_user_id),
                            submits_per_minute_quota=self._quota_get(
                                "JOBS_QUOTA_SUBMITS_PER_MIN",
                                domain,
                                owner_user_id,
                            ),
                            counters_enabled=JobManager._is_truthy(os.getenv("JOBS_COUNTERS_ENABLED", "")),
                            begin_immediate=slides_generation,
                            pre_admission_lookup=(
                                lambda active_conn: self._lookup_ready_slides_generation_job_in_connection(
                                    active_conn,
                                    owner_user_id=str(owner_user_id),
                                    idempotency_key=str(idempotency_key),
                                )
                                if slides_generation
                                else None
                            ),
                        )
                        d = self._map_admission_result(result)
                        if slides_generation:
                            self._validate_slides_generation_admission(
                                conn,
                                d,
                                owner_user_id=str(owner_user_id),
                                idempotency_key=str(idempotency_key),
                            )
                        with contextlib.suppress(_JOB_NONCRITICAL_EXCEPTIONS):
                            self._update_gauges(domain=domain, queue=queue, job_type=job_type)
                        with contextlib.suppress(_JOB_NONCRITICAL_EXCEPTIONS):
                            self._assert_invariants(d)
                        self._emit_create_side_effects(result, backend="sqlite", idempotency_key=idempotency_key)
                        return d
                    except sqlite3.OperationalError as exc:
                        if attempt == 0 and self._sqlite_missing_column_error(exc, "batch_group"):  # noqa: SIM102
                            if self._sqlite_ensure_batch_group(conn):
                                continue
                        raise
        finally:
            conn.close()

    def _dependency_path_exists_in_transaction(
        self,
        executor: Any,
        start_uuid: str,
        target_uuid: str,
    ) -> bool:
        """Check graph reachability using the caller's locked transaction."""

        if self.backend == "postgres":
            query = (
                "WITH RECURSIVE dependency_path(job_uuid) AS ("
                "SELECT depends_on_job_uuid FROM job_dependencies "
                "WHERE job_uuid = %s UNION "
                "SELECT edge.depends_on_job_uuid FROM job_dependencies AS edge "
                "JOIN dependency_path AS path ON edge.job_uuid = path.job_uuid"
                ") SELECT 1 FROM dependency_path WHERE job_uuid = %s LIMIT 1"
            )
        else:
            query = (
                "WITH RECURSIVE dependency_path(job_uuid) AS ("
                "SELECT depends_on_job_uuid FROM job_dependencies "
                "WHERE job_uuid = ? UNION "
                "SELECT edge.depends_on_job_uuid FROM job_dependencies AS edge "
                "JOIN dependency_path AS path ON edge.job_uuid = path.job_uuid"
                ") SELECT 1 FROM dependency_path WHERE job_uuid = ? LIMIT 1"
            )
        cursor = executor.execute(query, (str(start_uuid), str(target_uuid)))
        return cursor.fetchone() is not None

    def add_job_dependency(self, job_uuid: str, depends_on_job_uuid: str) -> bool:
        if not job_uuid or not depends_on_job_uuid:
            raise ValueError("job_uuid and depends_on_job_uuid are required")  # noqa: TRY003
        if str(job_uuid) == str(depends_on_job_uuid):
            raise ValueError("Job cannot depend on itself")  # noqa: TRY003
        job = self.get_job_by_uuid(str(job_uuid))
        dep = self.get_job_by_uuid(str(depends_on_job_uuid))
        if not job or not dep:
            raise ValueError("Both jobs must exist to create dependency")  # noqa: TRY003
        if str(job.get("domain")) != str(dep.get("domain")):
            raise ValueError("Dependencies must share domain")  # noqa: TRY003
        if str(job.get("owner_user_id")) != str(dep.get("owner_user_id")):
            raise ValueError("Dependencies must share owner_user_id")  # noqa: TRY003

        conn = self._connect()
        try:
            if self.backend == "postgres":
                with conn, self._pg_cursor(conn) as cur:
                    cur.execute("SET TRANSACTION ISOLATION LEVEL READ COMMITTED")
                    cur.execute(
                        "SELECT pg_advisory_xact_lock(%s)",
                        (self._pg_advisory_key("dependency-graph"),),
                    )
                    if self._dependency_path_exists_in_transaction(
                        cur,
                        str(depends_on_job_uuid),
                        str(job_uuid),
                    ):
                        raise ValueError(
                            "Dependency would create a cycle"
                        )  # noqa: TRY003
                    cur.execute(
                        (
                            "WITH locked_jobs AS ("
                            "SELECT child.uuid AS job_uuid, "
                            "dependency.uuid AS depends_on_job_uuid, "
                            "CASE WHEN dependency.status IN "
                            "('completed','failed','cancelled','quarantined') "
                            "THEN dependency.status END AS terminal_status, "
                            "CASE WHEN dependency.status IN "
                            "('completed','failed','cancelled','quarantined') "
                            "THEN dependency.cancellation_reason END AS cancellation_reason "
                            "FROM jobs AS child CROSS JOIN jobs AS dependency "
                            "WHERE child.uuid = %s AND dependency.uuid = %s "
                            "AND child.status = 'queued' "
                            "AND child.domain = dependency.domain "
                            "AND child.owner_user_id IS NOT DISTINCT FROM "
                            "dependency.owner_user_id "
                            "FOR UPDATE OF child, dependency) "
                            "INSERT INTO job_dependencies "
                            "(job_uuid, depends_on_job_uuid, "
                            "depends_on_terminal_status, "
                            "depends_on_cancellation_reason) "
                            "SELECT job_uuid, depends_on_job_uuid, terminal_status, "
                            "cancellation_reason FROM locked_jobs "
                            "ON CONFLICT (job_uuid, depends_on_job_uuid) DO NOTHING"
                        ),
                        (str(job_uuid), str(depends_on_job_uuid)),
                    )
                    return cur.rowcount > 0
            else:
                with conn:
                    conn.execute("BEGIN IMMEDIATE")
                    if self._dependency_path_exists_in_transaction(
                        conn,
                        str(depends_on_job_uuid),
                        str(job_uuid),
                    ):
                        raise ValueError(
                            "Dependency would create a cycle"
                        )  # noqa: TRY003
                    cur = conn.execute(
                        (
                            "INSERT OR IGNORE INTO job_dependencies "
                            "(job_uuid, depends_on_job_uuid, "
                            "depends_on_terminal_status, "
                            "depends_on_cancellation_reason) "
                            "SELECT child.uuid, dependency.uuid, "
                            "CASE WHEN dependency.status IN "
                            "('completed','failed','cancelled','quarantined') "
                            "THEN dependency.status END, "
                            "CASE WHEN dependency.status IN "
                            "('completed','failed','cancelled','quarantined') "
                            "THEN dependency.cancellation_reason END "
                            "FROM jobs AS child CROSS JOIN jobs AS dependency "
                            "WHERE child.uuid = ? AND dependency.uuid = ? "
                            "AND child.status = 'queued' "
                            "AND child.domain = dependency.domain "
                            "AND child.owner_user_id IS dependency.owner_user_id"
                        ),
                        (str(job_uuid), str(depends_on_job_uuid)),
                    )
                    return (cur.rowcount or 0) > 0
        finally:
            conn.close()

    def _cancel_dependent_jobs(self, job_uuid: str | None, *, reason: str) -> None:
        if not job_uuid:
            return
        conn = self._connect()
        try:
            if self.backend == "postgres":
                with conn, self._pg_cursor(conn) as cur:
                    cur.execute(
                        (
                            "SELECT j.id, j.uuid, j.domain, j.job_type FROM job_dependencies jd "
                            "JOIN jobs j ON j.uuid = jd.job_uuid "
                            "WHERE jd.depends_on_job_uuid = %s AND j.status IN ('queued','processing')"
                        ),
                        (str(job_uuid),),
                    )
                    dependents = [dict(row) for row in (cur.fetchall() or [])]
            else:
                with conn:
                    rows = conn.execute(
                        (
                            "SELECT j.id, j.uuid, j.domain, j.job_type FROM job_dependencies jd "
                            "JOIN jobs j ON j.uuid = jd.job_uuid "
                            "WHERE jd.depends_on_job_uuid = ? AND j.status IN ('queued','processing')"
                        ),
                        (str(job_uuid),),
                    ).fetchall()
                    dependents = [dict(row) for row in rows]
        finally:
            conn.close()
        for dependent in dependents:
            try:
                self.cancel_job(
                    int(dependent["id"]),
                    reason=reason,
                    expected_uuid=str(dependent["uuid"]),
                    expected_domain=str(dependent["domain"]),
                    expected_job_type=str(dependent["job_type"]),
                )
            except _JOB_NONCRITICAL_EXCEPTIONS:
                continue

    def _reconcile_terminal_dependents(
        self,
        *,
        domain: str | None = None,
        queue: str | None = None,
        owner_user_id: str | None = None,
        job_type: str | None = None,
    ) -> int:
        """Cancel one bounded batch of jobs blocked by terminal dependencies."""

        batch_size = JobManager._expired_recovery_batch_size()
        conn = self._connect()
        try:
            if self.backend == "postgres":
                where = [
                    "child.status IN ('queued','processing')",
                    "COALESCE(dependency.status, jd.depends_on_terminal_status, 'missing') "
                    "IN ('failed','cancelled','quarantined','missing')",
                ]
                params: list[Any] = []
                if domain is not None:
                    where.append("child.domain = %s")
                    params.append(domain)
                if queue is not None:
                    where.append("child.queue = %s")
                    params.append(queue)
                if owner_user_id is not None:
                    where.append("child.owner_user_id = %s")
                    params.append(owner_user_id)
                if job_type is not None:
                    where.append("child.job_type = %s")
                    params.append(job_type)
                with self._pg_cursor(conn) as cur:
                    cur.execute(
                        (
                            "SELECT child.id, child.uuid, child.domain, child.job_type, "
                            "MAX(CASE WHEN COALESCE(dependency.status, "
                            "jd.depends_on_terminal_status, 'missing') "
                            "IN ('failed','quarantined','missing') "
                            "OR (COALESCE(dependency.status, jd.depends_on_terminal_status) = 'cancelled' "
                            "AND COALESCE(dependency.cancellation_reason, "
                            "jd.depends_on_cancellation_reason) = 'dependency_failed') "
                            "THEN 1 ELSE 0 END) AS dependency_failed "
                            "FROM job_dependencies jd "
                            "JOIN jobs child ON child.uuid = jd.job_uuid "
                            "LEFT JOIN jobs dependency ON dependency.uuid = jd.depends_on_job_uuid "
                            f"WHERE {' AND '.join(where)} "  # nosec B608
                            "GROUP BY child.id, child.uuid, child.domain, child.job_type "
                            "ORDER BY child.id ASC LIMIT %s"
                        ),
                        (*params, batch_size),
                    )
                    candidates = [dict(row) for row in (cur.fetchall() or [])]
            else:
                where = [
                    "child.status IN ('queued','processing')",
                    "COALESCE(dependency.status, jd.depends_on_terminal_status, 'missing') "
                    "IN ('failed','cancelled','quarantined','missing')",
                ]
                params = []
                if domain is not None:
                    where.append("child.domain = ?")
                    params.append(domain)
                if queue is not None:
                    where.append("child.queue = ?")
                    params.append(queue)
                if owner_user_id is not None:
                    where.append("child.owner_user_id = ?")
                    params.append(owner_user_id)
                if job_type is not None:
                    where.append("child.job_type = ?")
                    params.append(job_type)
                candidates = [
                    dict(row)
                    for row in conn.execute(
                        (
                            "SELECT child.id, child.uuid, child.domain, child.job_type, "
                            "MAX(CASE WHEN COALESCE(dependency.status, "
                            "jd.depends_on_terminal_status, 'missing') "
                            "IN ('failed','quarantined','missing') "
                            "OR (COALESCE(dependency.status, jd.depends_on_terminal_status) = 'cancelled' "
                            "AND COALESCE(dependency.cancellation_reason, "
                            "jd.depends_on_cancellation_reason) = 'dependency_failed') "
                            "THEN 1 ELSE 0 END) AS dependency_failed "
                            "FROM job_dependencies jd "
                            "JOIN jobs child ON child.uuid = jd.job_uuid "
                            "LEFT JOIN jobs dependency ON dependency.uuid = jd.depends_on_job_uuid "
                            f"WHERE {' AND '.join(where)} "  # nosec B608
                            "GROUP BY child.id, child.uuid, child.domain, child.job_type "
                            "ORDER BY child.id ASC LIMIT ?"
                        ),
                        (*params, batch_size),
                    ).fetchall()
                ]
        finally:
            conn.close()

        reconciled = 0
        for candidate in candidates:
            reason = (
                "dependency_failed"
                if int(candidate.get("dependency_failed") or 0)
                else "dependency_cancelled"
            )
            try:
                cancelled = self.cancel_job(
                    int(candidate["id"]),
                    reason=reason,
                    expected_uuid=str(candidate["uuid"]),
                    expected_domain=str(candidate["domain"]),
                    expected_job_type=str(candidate["job_type"]),
                    cascade_dependents=False,
                )
            except _JOB_NONCRITICAL_EXCEPTIONS:
                continue
            if cancelled:
                reconciled += 1
        return reconciled

    def get_job(self, job_id: int) -> dict[str, Any] | None:
        """Fetch a job by numeric id.

        Returns None if not found. JSON payload/result are normalized to dicts
        for SQLite; Postgres returns native JSON via the driver.
        """
        # Read-only helper; no completion_token semantics apply
        conn = self._connect()
        try:
            if self.backend == "postgres":
                with self._pg_cursor(conn) as cur:
                    cur.execute("SELECT * FROM jobs WHERE id = %s", (int(job_id),))
                    row = cur.fetchone()
                if not row:
                    return None
                d = dict(row)
                try:
                    d["payload"] = self._maybe_decrypt_json(d.get("payload"))
                    d["result"] = self._maybe_decrypt_json(d.get("result"))
                except _JOB_NONCRITICAL_EXCEPTIONS:
                    pass
                return d
            else:
                row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
                if not row:
                    return None
                d = dict(row)
                try:
                    if isinstance(d.get("payload"), str):
                        d["payload"] = json.loads(d["payload"]) if d["payload"] else {}
                    if isinstance(d.get("result"), str):
                        d["result"] = json.loads(d["result"]) if d["result"] else None
                    d["payload"] = self._maybe_decrypt_json(d.get("payload"))
                    d["result"] = self._maybe_decrypt_json(d.get("result"))
                except _JOB_NONCRITICAL_EXCEPTIONS:
                    pass
                return d
        finally:
            conn.close()

    def get_job_by_uuid(self, job_uuid: str) -> dict[str, Any] | None:
        """Fetch a job by UUID string.

        Returns None if not found. JSON payload/result are normalized to dicts
        for SQLite; Postgres returns native JSON via the driver.
        """
        conn = self._connect()
        try:
            if self.backend == "postgres":
                with self._pg_cursor(conn) as cur:
                    cur.execute("SELECT * FROM jobs WHERE uuid = %s", (str(job_uuid),))
                    row = cur.fetchone()
                if not row:
                    return None
                d = dict(row)
                try:
                    d["payload"] = self._maybe_decrypt_json(d.get("payload"))
                    d["result"] = self._maybe_decrypt_json(d.get("result"))
                except _JOB_NONCRITICAL_EXCEPTIONS:
                    pass
                return d
            else:
                row = conn.execute("SELECT * FROM jobs WHERE uuid = ?", (str(job_uuid),)).fetchone()
                if not row:
                    return None
                d = dict(row)
                try:
                    if isinstance(d.get("payload"), str):
                        d["payload"] = json.loads(d["payload"]) if d["payload"] else {}
                    if isinstance(d.get("result"), str):
                        d["result"] = json.loads(d["result"]) if d["result"] else None
                    d["payload"] = self._maybe_decrypt_json(d.get("payload"))
                    d["result"] = self._maybe_decrypt_json(d.get("result"))
                except _JOB_NONCRITICAL_EXCEPTIONS:
                    pass
                return d
        finally:
            conn.close()

    def get_job_or_archived_by_uuid(
        self,
        job_uuid: str,
        *,
        domain: str | None = None,
        owner_user_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Fetch exactly one scoped Job by UUID across active/archive storage.

        The backend performs one union query so a concurrent transactional
        archive move cannot create a false missing result. Duplicate authority
        across the two stores fails closed.
        """

        conn = self._connect()
        try:
            if self.backend == "postgres":
                with self._pg_cursor(conn) as cur:
                    job = _postgres_get_job_or_archived_by_uuid(
                        cur,
                        job_uuid,
                        domain=domain,
                        owner_user_id=owner_user_id,
                    )
            else:
                job = _sqlite_get_job_or_archived_by_uuid(
                    conn,
                    job_uuid,
                    domain=domain,
                    owner_user_id=owner_user_id,
                )
            if job is None:
                return None
            if job.get("archived"):
                return self._normalize_archived_job_row(job)
            normalized = self._normalize_active_job_row(job)
            normalized["archived"] = False
            return normalized
        finally:
            conn.close()

    def patch_terminal_operation_result(
        self,
        command: TerminalOperationResultPatchCommand,
    ) -> TerminalOperationResultPatchOutcome:
        """Replace one exact terminal operation result across active/archive storage."""

        try:
            replacement_json = json.dumps(
                command.replacement_result,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Terminal operation result must be JSON-serializable"
            ) from exc
        max_bytes = int(
            os.getenv("JOBS_MAX_JSON_BYTES", "1048576") or "1048576"
        )
        result_bytes = len(replacement_json.encode("utf-8"))
        if result_bytes > max_bytes:
            raise ValueError(
                f"Terminal operation result too large: {result_bytes} bytes > limit {max_bytes}"
            )
        stored_replacement = self._maybe_encrypt_json(
            command.replacement_result,
            command.domain,
        )
        if not isinstance(stored_replacement, dict):
            raise ValueError("Terminal operation result must be an object")

        def decode_result(raw_result: Any, compressed_result: Any) -> Any:
            parsed = self._parse_json_value(raw_result)
            if parsed is None and compressed_result is not None:
                parsed = self._decode_archive_blob(compressed_result)
            if parsed is None:
                parsed = {}
            return self._maybe_decrypt_json(
                parsed,
                fail_on_error=True,
                field_name="terminal operation result",
            )

        conn = self._connect()
        try:
            if self.backend == "postgres":
                return _postgres_patch_terminal_operation_result(
                    conn,
                    self._pg_cursor,
                    command=command,
                    stored_replacement=stored_replacement,
                    decode_result=decode_result,
                )
            return _sqlite_patch_terminal_operation_result(
                conn,
                command=command,
                stored_replacement=stored_replacement,
                decode_result=decode_result,
            )
        finally:
            conn.close()

    def _serialize_replacement_payload(
        self,
        payload: dict[str, Any],
        *,
        domain: str,
        queue: str,
        job_type: str,
    ) -> str:
        """Apply Jobs payload policy and serialize one replacement value."""

        if not isinstance(payload, dict):
            raise ValueError("Job payload must be an object")  # noqa: TRY003

        cleaned, found, where = self._scan_and_redact_secrets(payload)
        if found and JobManager._is_truthy(os.getenv("JOBS_SECRET_REJECT", "")):
            raise ValueError(  # noqa: TRY003
                "Payload appears to contain secrets at: "
                f"{where[:3]}{'...' if len(where) > 3 else ''}"
            )
        selected = cleaned if found else payload
        stored = self._maybe_encrypt_json(selected, domain)
        serialized = json.dumps(stored)
        payload_bytes = len(serialized.encode("utf-8"))
        max_bytes = int(
            os.getenv("JOBS_MAX_JSON_BYTES", "1048576") or "1048576"
        )
        if payload_bytes <= max_bytes:
            return serialized
        if not JobManager._is_truthy(os.getenv("JOBS_JSON_TRUNCATE", "")):
            raise ValueError(  # noqa: TRY003
                f"Payload too large: {payload_bytes} bytes > limit {max_bytes}"
            )
        with contextlib.suppress(_JOB_NONCRITICAL_EXCEPTIONS):
            increment_json_truncated(
                {"domain": domain, "queue": queue, "job_type": job_type},
                "payload",
            )
        return json.dumps({"_truncated": True, "len_bytes": payload_bytes})

    def _secured_prompt_archive_payload(
        self,
        payload_value: Any,
        *,
        queue: str,
    ) -> str | None:
        """Return a secret-free replacement for a legacy Prompt payload."""

        payload = self._parse_json_value(payload_value)
        payload = self._maybe_decrypt_json(
            payload,
            fail_on_error=True,
            field_name="payload",
        )
        if not isinstance(payload, dict):
            raise ValueError("Prompt Studio optimization payload must be an object")  # noqa: TRY003

        from tldw_Server_API.app.core.Prompt_Management.optimization_model_config import (
            strip_sensitive_durable_mapping,
        )

        secured = strip_sensitive_durable_mapping(payload)
        if secured == payload:
            return None
        return self._serialize_replacement_payload(
            secured,
            domain="prompt_studio",
            queue=queue,
            job_type="optimization",
        )

    def replace_job_payload(
        self,
        job_id: int,
        *,
        payload: dict[str, Any],
        expected_uuid: str | None = None,
        expected_domain: str | None = None,
    ) -> bool:
        """Atomically replace one guarded live Jobs payload."""
        if not isinstance(payload, dict):
            raise ValueError("Job payload must be an object")  # noqa: TRY003

        conn = self._connect()
        try:
            if self.backend == "postgres":
                clauses = ["id = %s"]
                guards: list[Any] = [int(job_id)]
                if expected_uuid is not None:
                    clauses.append("uuid = %s")
                    guards.append(str(expected_uuid))
                if expected_domain is not None:
                    clauses.append("domain = %s")
                    guards.append(str(expected_domain))
                where_sql = " AND ".join(clauses)
                with conn:
                    with self._pg_cursor(conn) as cur:
                        cur.execute(
                            "SELECT domain, queue, job_type FROM jobs "
                            f"WHERE {where_sql} FOR UPDATE",  # nosec B608
                            tuple(guards),
                        )
                        row = cur.fetchone()
                        if not row:
                            return False
                        serialized = self._serialize_replacement_payload(
                            payload,
                            domain=str(row["domain"]),
                            queue=str(row["queue"]),
                            job_type=str(row["job_type"]),
                        )
                        cur.execute(
                            "UPDATE jobs SET payload = %s::jsonb, "
                            f"updated_at = NOW() WHERE {where_sql}",  # nosec B608
                            (serialized, *guards),
                        )
                        return cur.rowcount == 1

            clauses = ["id = ?"]
            guards = [int(job_id)]
            if expected_uuid is not None:
                clauses.append("uuid = ?")
                guards.append(str(expected_uuid))
            if expected_domain is not None:
                clauses.append("domain = ?")
                guards.append(str(expected_domain))
            where_sql = " AND ".join(clauses)
            with conn:
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute(
                    "SELECT domain, queue, job_type FROM jobs "
                    f"WHERE {where_sql}",  # nosec B608
                    tuple(guards),
                ).fetchone()
                if not row:
                    return False
                serialized = self._serialize_replacement_payload(
                    payload,
                    domain=str(row[0]),
                    queue=str(row[1]),
                    job_type=str(row[2]),
                )
                cursor = conn.execute(
                    "UPDATE jobs SET payload = ?, updated_at = DATETIME('now') "
                    f"WHERE {where_sql}",  # nosec B608
                    (serialized, *guards),
                )
                return cursor.rowcount == 1
        finally:
            conn.close()

    def replace_archived_job_payload(
        self,
        job_id: int,
        *,
        payload: dict[str, Any],
        expected_uuid: str | None = None,
        expected_domain: str | None = None,
        expected_archive_locator: str | int | None = None,
    ) -> bool:
        """Atomically replace one guarded archive payload and stale blob copy."""
        if not isinstance(payload, dict):
            raise ValueError("Job payload must be an object")  # noqa: TRY003

        conn = self._connect()
        try:
            if self.backend == "postgres":
                clauses = ["id = %s"]
                guards: list[Any] = [int(job_id)]
                if expected_uuid is not None:
                    clauses.append("uuid = %s")
                    guards.append(str(expected_uuid))
                if expected_domain is not None:
                    clauses.append("domain = %s")
                    guards.append(str(expected_domain))
                if expected_archive_locator is not None:
                    clauses.append("archive_id = %s")
                    guards.append(int(expected_archive_locator))
                where_sql = " AND ".join(clauses)
                with conn:
                    with self._pg_cursor(conn) as cur:
                        cur.execute(
                            "SELECT domain, queue, job_type FROM jobs_archive "
                            f"WHERE {where_sql} FOR UPDATE",  # nosec B608
                            tuple(guards),
                        )
                        row = cur.fetchone()
                        if not row:
                            return False
                        serialized = self._serialize_replacement_payload(
                            payload,
                            domain=str(row["domain"]),
                            queue=str(row["queue"]),
                            job_type=str(row["job_type"]),
                        )
                        cur.execute(
                            "UPDATE jobs_archive SET payload = %s::jsonb, "
                            f"payload_compressed = NULL WHERE {where_sql}",  # nosec B608
                            (serialized, *guards),
                        )
                        return cur.rowcount > 0

            clauses = ["id = ?"]
            guards = [int(job_id)]
            if expected_uuid is not None:
                clauses.append("uuid = ?")
                guards.append(str(expected_uuid))
            if expected_domain is not None:
                clauses.append("domain = ?")
                guards.append(str(expected_domain))
            if expected_archive_locator is not None:
                clauses.append("archive_id = ?")
                guards.append(int(expected_archive_locator))
            where_sql = " AND ".join(clauses)
            with conn:
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute(
                    "SELECT domain, queue, job_type FROM jobs_archive "
                    f"WHERE {where_sql}",  # nosec B608
                    tuple(guards),
                ).fetchone()
                if not row:
                    return False
                serialized = self._serialize_replacement_payload(
                    payload,
                    domain=str(row[0]),
                    queue=str(row[1]),
                    job_type=str(row[2]),
                )
                cursor = conn.execute(
                    "UPDATE jobs_archive SET payload = ?, payload_compressed = NULL "
                    f"WHERE {where_sql}",  # nosec B608
                    (serialized, *guards),
                )
                return cursor.rowcount > 0
        finally:
            conn.close()

    def _normalize_archived_job_row(
        self,
        row: Any,
        *,
        fail_on_decryption_error: bool = False,
    ) -> dict[str, Any]:
        """Decode one archive row using the same policy as single-row reads."""

        job_data = dict(row)
        raw_payload = job_data.get("payload")
        raw_payload_compressed = job_data.get("payload_compressed")
        payload = self._parse_json_value(raw_payload)
        compressed_payload = self._decode_archive_blob(raw_payload_compressed)
        result = self._parse_json_value(job_data.get("result"))
        payload = self._maybe_decrypt_json(
            payload,
            fail_on_error=fail_on_decryption_error,
            field_name="payload",
        )
        compressed_payload = self._maybe_decrypt_json(
            compressed_payload,
            fail_on_error=fail_on_decryption_error,
            field_name="compressed payload",
        )
        if (
            str(job_data.get("domain") or "") == "prompt_studio"
            and str(job_data.get("job_type") or "") == "optimization"
        ):
            primary_is_object = isinstance(payload, dict)
            compressed_is_object = isinstance(compressed_payload, dict)
            if not primary_is_object:
                payload = compressed_payload if compressed_is_object else {}
            if not primary_is_object or raw_payload_compressed is not None:
                job_data["_archive_payload_rewrite_required"] = True
        elif payload is None:
            payload = compressed_payload
        if result is None:
            result = self._decode_archive_blob(job_data.get("result_compressed"))
        job_data["payload"] = payload
        job_data["result"] = self._maybe_decrypt_json(
            result,
            fail_on_error=fail_on_decryption_error,
            field_name="result",
        )
        job_data["archived"] = True
        return job_data

    def list_archived_jobs(
        self,
        *,
        domain: str | None = None,
        queue: str | None = None,
        status: str | None = None,
        job_type: str | None = None,
        created_before: datetime | None = None,
        before_id: int | None = None,
        before_uuid: str | None = None,
        before_archive_locator: str | int | None = None,
        fail_on_decryption_error: bool = False,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """List archived jobs for bounded migration and repair passes."""

        cursor_values = (
            created_before,
            before_id,
            before_uuid,
            before_archive_locator,
        )
        if any(value is not None for value in cursor_values) and not all(
            value is not None for value in cursor_values
        ):
            raise BadRequestError(
                "A complete archive cursor requires created_before, before_id, "
                "before_uuid, and before_archive_locator"
            )
        conn = self._connect()
        try:
            if self.backend == "postgres":
                cursor_time_sql = POSTGRES_ARCHIVE_CURSOR_TIME_SQL
                cursor_output_time_sql = cursor_time_sql
                archive_locator_sql = "archive_id"
                select_sql = (
                    "SELECT *, archive_id AS _archive_locator, "  # nosec B608
                    f"{cursor_output_time_sql} AS _archive_cursor_created_at, "
                    "COALESCE(uuid, '') AS _archive_cursor_uuid "
                    "FROM jobs_archive WHERE 1=1"
                )
            else:
                cursor_output_time_sql = SQLITE_ARCHIVE_CURSOR_OUTPUT_SQL
                cursor_time_sql = SQLITE_ARCHIVE_CURSOR_TIME_SQL
                archive_locator_sql = "archive_id"
                select_sql = (
                    "SELECT *, archive_id AS _archive_locator, "  # nosec B608
                    f"{cursor_output_time_sql} AS _archive_cursor_created_at, "
                    "COALESCE(uuid, '') AS _archive_cursor_uuid "
                    "FROM jobs_archive WHERE 1=1"
                )
            query = select_sql
            params: list[Any] = []
            placeholder = "%s" if self.backend == "postgres" else "?"
            cursor_placeholder = (
                placeholder
                if self.backend == "postgres"
                else f"julianday({placeholder})"
            )
            for column, value in (
                ("domain", domain),
                ("queue", queue),
                ("status", status),
                ("job_type", job_type),
            ):
                if value:
                    query += f" AND {column} = {placeholder}"  # nosec B608
                    params.append(value)
            if created_before is not None:
                cursor_value: Any = created_before
                if self.backend != "postgres":
                    if created_before.tzinfo is not None:
                        created_before = created_before.astimezone(_tz.utc).replace(
                            tzinfo=None
                        )
                    cursor_value = created_before.isoformat(
                        sep=" ",
                        timespec="microseconds",
                    )
                if before_id is None:
                    query += f" AND {cursor_time_sql} <= {cursor_placeholder}"  # nosec B608
                    params.append(cursor_value)
                elif before_uuid is None:
                    query += (
                        f" AND ({cursor_time_sql} < {cursor_placeholder} OR "  # nosec B608
                        f"({cursor_time_sql} = {cursor_placeholder} AND id < {placeholder}))"
                    )
                    params.extend([cursor_value, cursor_value, int(before_id)])
                elif before_archive_locator is None:
                    query += (
                        f" AND ({cursor_time_sql} < {cursor_placeholder} OR "  # nosec B608
                        f"({cursor_time_sql} = {cursor_placeholder} AND "
                        f"(id < {placeholder} OR (id = {placeholder} AND "
                        f"COALESCE(uuid, '') < {placeholder}))))"
                    )
                    params.extend(
                        [
                            cursor_value,
                            cursor_value,
                            int(before_id),
                            int(before_id),
                            str(before_uuid),
                        ]
                    )
                else:
                    query += (
                        f" AND ({cursor_time_sql} < {cursor_placeholder} OR "  # nosec B608
                        f"({cursor_time_sql} = {cursor_placeholder} AND "
                        f"(id < {placeholder} OR (id = {placeholder} AND "
                        f"(COALESCE(uuid, '') < {placeholder} OR "
                        f"(COALESCE(uuid, '') = {placeholder} AND "
                        f"{archive_locator_sql} < {placeholder}))))))"
                    )
                    params.extend(
                        [
                            cursor_value,
                            cursor_value,
                            int(before_id),
                            int(before_id),
                            str(before_uuid),
                            str(before_uuid),
                            int(before_archive_locator),
                        ]
                    )
            query += (  # nosec B608
                f" ORDER BY {cursor_time_sql} DESC, id DESC, "
                f"COALESCE(uuid, '') DESC, {archive_locator_sql} DESC "
                f"LIMIT {placeholder}"
            )
            params.append(int(limit))
            if self.backend == "postgres":
                with self._pg_cursor(conn) as cur:
                    cur.execute(query, tuple(params))
                    rows = cur.fetchall() or []
            else:
                rows = conn.execute(query, tuple(params)).fetchall() or []
            return [
                self._normalize_archived_job_row(
                    row,
                    fail_on_decryption_error=fail_on_decryption_error,
                )
                for row in rows
            ]
        finally:
            conn.close()

    def get_job_or_archived(
        self,
        job_id: int,
        domain: str | None = None,
        *,
        job_uuid: str | None = None,
        archive_locator: str | int | None = None,
    ) -> dict[str, Any] | None:
        """Fetch a job from the active table or the archive table.

        Returns a job dict with normalized payload/result and an "archived" flag.
        """
        job = self.get_job(job_id)
        if (
            job
            and archive_locator is None
            and (not domain or job.get("domain") == domain)
            and (job_uuid is None or str(job.get("uuid") or "") == str(job_uuid))
        ):
            job["archived"] = False
            return job

        conn = self._connect()
        try:
            clauses: list[str] = []
            params: list[Any] = []
            if self.backend == "postgres":
                clauses.append("id = %s")
                params.append(int(job_id))
                if domain:
                    clauses.append("domain = %s")
                    params.append(domain)
                if job_uuid is not None:
                    clauses.append("uuid = %s")
                    params.append(str(job_uuid))
                if archive_locator is not None:
                    clauses.append("archive_id = %s")
                    params.append(int(archive_locator))
                with self._pg_cursor(conn) as cur:
                    cur.execute(
                        "SELECT *, archive_id AS _archive_locator "  # nosec B608
                        "FROM jobs_archive WHERE "
                        + " AND ".join(clauses)
                        + " ORDER BY archive_id DESC LIMIT 1",
                        tuple(params),
                    )
                    row = cur.fetchone()
            else:
                clauses.append("id = ?")
                params.append(int(job_id))
                if domain:
                    clauses.append("domain = ?")
                    params.append(domain)
                if job_uuid is not None:
                    clauses.append("uuid = ?")
                    params.append(str(job_uuid))
                if archive_locator is not None:
                    clauses.append("archive_id = ?")
                    params.append(int(archive_locator))
                row = conn.execute(
                    "SELECT *, archive_id AS _archive_locator "  # nosec B608
                    "FROM jobs_archive WHERE "
                    + " AND ".join(clauses)
                    + " ORDER BY archive_id DESC LIMIT 1",
                    tuple(params),
                ).fetchone()
            if not row:
                return None
            return self._normalize_archived_job_row(row)
        finally:
            conn.close()

    def _normalize_active_job_row(self, row: Any) -> dict[str, Any]:
        """Normalize one active row using the same policy as ``get_job``."""

        job_data = dict(row)
        try:
            if self.backend != "postgres":
                if isinstance(job_data.get("payload"), str):
                    job_data["payload"] = (
                        json.loads(job_data["payload"])
                        if job_data["payload"]
                        else {}
                    )
                if isinstance(job_data.get("result"), str):
                    job_data["result"] = (
                        json.loads(job_data["result"])
                        if job_data["result"]
                        else None
                    )
            job_data["payload"] = self._maybe_decrypt_json(
                job_data.get("payload")
            )
            job_data["result"] = self._maybe_decrypt_json(job_data.get("result"))
        except _JOB_NONCRITICAL_EXCEPTIONS:
            pass
        return job_data

    def get_jobs_by_ids(
        self,
        job_ids: list[int],
        *,
        domain: str | None = None,
        owner_user_id: str | None = None,
        include_archived: bool = False,
    ) -> dict[int, dict[str, Any]]:
        """Fetch scoped active and optionally archived jobs by numeric ID."""

        if not isinstance(job_ids, list):
            raise BadRequestError("job_ids must be a list of positive integers")

        unique_ids: list[int] = []
        seen_ids: set[int] = set()
        for job_id in job_ids:
            if type(job_id) is not int or job_id <= 0:
                raise BadRequestError(
                    "job_ids must contain only positive integers"
                )
            if job_id not in seen_ids:
                unique_ids.append(job_id)
                seen_ids.add(job_id)
        if not unique_ids:
            return {}

        placeholder = "%s" if self.backend == "postgres" else "?"
        chunk_size = 1000 if self.backend == "postgres" else 400
        rows_by_id: dict[int, dict[str, Any]] = {}

        def _query_rows(
            conn: Any,
            *,
            table: str,
            ids: list[int],
        ) -> list[Any]:
            id_placeholders = ",".join([placeholder] * len(ids))
            query = f"SELECT * FROM {table} WHERE id IN ({id_placeholders})"  # nosec B608
            params: list[Any] = list(ids)
            if domain is not None:
                query += f" AND domain = {placeholder}"  # nosec B608
                params.append(domain)
            if owner_user_id is not None:
                query += f" AND owner_user_id = {placeholder}"  # nosec B608
                params.append(owner_user_id)
            if table == "jobs_archive":
                query += " ORDER BY archive_id DESC"
            if self.backend == "postgres":
                with self._pg_cursor(conn) as cur:
                    cur.execute(query, tuple(params))
                    return list(cur.fetchall() or [])
            return list(conn.execute(query, tuple(params)).fetchall() or [])

        conn = self._connect()
        try:
            for offset in range(0, len(unique_ids), chunk_size):
                chunk = unique_ids[offset : offset + chunk_size]
                for row in _query_rows(conn, table="jobs", ids=chunk):
                    job_data = self._normalize_active_job_row(row)
                    job_data["archived"] = False
                    rows_by_id[int(job_data["id"])] = job_data

            if include_archived:
                missing_ids = [
                    job_id for job_id in unique_ids if job_id not in rows_by_id
                ]
                for offset in range(0, len(missing_ids), chunk_size):
                    chunk = missing_ids[offset : offset + chunk_size]
                    for row in _query_rows(
                        conn,
                        table="jobs_archive",
                        ids=chunk,
                    ):
                        job_data = self._normalize_archived_job_row(row)
                        job_id = int(job_data["id"])
                        if job_id not in rows_by_id:
                            rows_by_id[job_id] = job_data
            return rows_by_id
        finally:
            conn.close()

    def find_job_by_batch_group(
        self,
        *,
        batch_group: str,
        domain: str,
        owner_user_id: str,
        job_type: str,
        include_archived: bool = False,
    ) -> dict[str, Any] | None:
        """Find the newest job matching one exact scoped batch group."""

        placeholder = "%s" if self.backend == "postgres" else "?"
        predicates = " AND ".join(
            f"{column} = {placeholder}"
            for column in (
                "batch_group",
                "domain",
                "owner_user_id",
                "job_type",
            )
        )
        params = (batch_group, domain, owner_user_id, job_type)
        conn = self._connect()
        try:
            def _sqlite_fetchone_with_batch_group_repair(query: str) -> Any:
                for attempt in range(2):
                    try:
                        return conn.execute(query, params).fetchone()
                    except sqlite3.OperationalError as exc:
                        if (
                            attempt == 0
                            and self._sqlite_missing_column_error(
                                exc, "batch_group"
                            )
                            and self._sqlite_ensure_batch_group(conn)
                        ):
                            continue
                        raise
                return None

            active_query = (
                f"SELECT * FROM jobs WHERE {predicates} "  # nosec B608
                "ORDER BY id DESC LIMIT 1"
            )
            if self.backend == "postgres":
                with self._pg_cursor(conn) as cur:
                    cur.execute(active_query, params)
                    row = cur.fetchone()
            else:
                row = _sqlite_fetchone_with_batch_group_repair(active_query)
            if row:
                job_data = self._normalize_active_job_row(row)
                job_data["archived"] = False
                return job_data
            if not include_archived:
                return None

            archive_query = (
                f"SELECT * FROM jobs_archive WHERE {predicates} "  # nosec B608
                "ORDER BY archive_id DESC LIMIT 1"
            )
            if self.backend == "postgres":
                with self._pg_cursor(conn) as cur:
                    cur.execute(archive_query, params)
                    row = cur.fetchone()
            else:
                row = _sqlite_fetchone_with_batch_group_repair(archive_query)
            if not row:
                return None
            return self._normalize_archived_job_row(row)
        finally:
            conn.close()

    def list_jobs(
        self,
        *,
        domain: str | None = None,
        queue: str | None = None,
        status: str | None = None,
        owner_user_id: str | None = None,
        job_type: str | None = None,
        batch_group: str | None = None,
        created_after: datetime | None = None,
        created_before: datetime | None = None,
        before_id: int | None = None,
        limit: int = 100,
        sort_by: str = "created_at",
        sort_order: str = "desc",
    ) -> list[dict[str, Any]]:
        """List jobs with optional filters.

        Args:
            domain: Filter by domain.
            queue: Filter by queue.
            status: Filter by status (queued|processing|completed|failed|cancelled).
            owner_user_id: Filter by owner id.
            batch_group: Optional indexed batch grouping key.
            limit: Max rows to return (default 100).
            before_id: Optional cursor id for stable pagination with created_before.
        """
        sort_col = sort_by if sort_by in {"created_at", "priority", "status"} else "created_at"
        sort_ord = "DESC" if str(sort_order).lower() == "desc" else "ASC"
        if before_id is not None and (
            created_before is None or sort_col != "created_at" or sort_ord != "DESC"
        ):
            raise BadRequestError("before_id requires created_before with created_at DESC ordering")  # noqa: TRY003
        conn = self._connect()
        try:
            if self.backend == "postgres":
                query = "SELECT * FROM jobs WHERE 1=1"
                params: list[Any] = []
                if domain:
                    query += " AND domain = %s"
                    params.append(domain)
                if queue:
                    query += " AND queue = %s"
                    params.append(queue)
                if status:
                    query += " AND status = %s"
                    params.append(status)
                if owner_user_id:
                    query += " AND owner_user_id = %s"
                    params.append(owner_user_id)
                if job_type:
                    query += " AND job_type = %s"
                    params.append(job_type)
                if batch_group:
                    query += " AND batch_group = %s"
                    params.append(batch_group)
                if created_after:
                    query += " AND created_at >= %s"
                    params.append(created_after)
                if created_before:
                    if before_id is not None and sort_col == "created_at" and sort_ord == "DESC":
                        query += " AND (created_at < %s OR (created_at = %s AND id < %s))"
                        params.extend([created_before, created_before, int(before_id)])
                    else:
                        query += " AND created_at <= %s"
                        params.append(created_before)
                # Add deterministic tie-breaker on id
                if sort_col == "created_at":
                    query += f" ORDER BY {sort_col} {sort_ord}, id {'DESC' if sort_ord=='DESC' else 'ASC'} LIMIT %s"
                else:
                    query += f" ORDER BY {sort_col} {sort_ord} LIMIT %s"
                params.append(limit)
                with self._pg_cursor(conn) as cur:
                    cur.execute(query, params)
                    rows = cur.fetchall()
                out = [dict(r) for r in rows]
                for d in out:
                    try:
                        d["payload"] = self._maybe_decrypt_json(d.get("payload"))
                        d["result"] = self._maybe_decrypt_json(d.get("result"))
                    except _JOB_NONCRITICAL_EXCEPTIONS:
                        pass
                return out
            else:
                query = "SELECT * FROM jobs WHERE 1=1"
                params: list[Any] = []
                if domain:
                    query += " AND domain = ?"
                    params.append(domain)
                if queue:
                    query += " AND queue = ?"
                    params.append(queue)
                if status:
                    query += " AND status = ?"
                    params.append(status)
                if owner_user_id:
                    query += " AND owner_user_id = ?"
                    params.append(owner_user_id)
                if job_type:
                    query += " AND job_type = ?"
                    params.append(job_type)
                if batch_group:
                    query += " AND batch_group = ?"
                    params.append(batch_group)

                def _sqlite_dt(dt_val: datetime) -> str:
                    if dt_val.tzinfo is not None:
                        dt_val = dt_val.astimezone(_tz.utc).replace(tzinfo=None)
                    return dt_val.strftime("%Y-%m-%d %H:%M:%S")

                if created_after:
                    query += " AND created_at >= ?"
                    params.append(_sqlite_dt(created_after))
                if created_before:
                    if before_id is not None and sort_col == "created_at" and sort_ord == "DESC":
                        created_before_str = _sqlite_dt(created_before)
                        query += " AND (created_at < ? OR (created_at = ? AND id < ?))"
                        params.extend([created_before_str, created_before_str, int(before_id)])
                    else:
                        query += " AND created_at <= ?"
                        params.append(_sqlite_dt(created_before))
                if sort_col == "created_at":
                    query += f" ORDER BY {sort_col} {sort_ord}, id {'DESC' if sort_ord=='DESC' else 'ASC'} LIMIT ?"
                else:
                    query += f" ORDER BY {sort_col} {sort_ord} LIMIT ?"
                params.append(limit)
                try:
                    rows = conn.execute(query, params).fetchall()
                except sqlite3.OperationalError as exc:
                    if (
                        batch_group
                        and self._sqlite_missing_column_error(exc, "batch_group")
                        and self._sqlite_ensure_batch_group(conn)
                    ):
                        rows = conn.execute(query, params).fetchall()
                    else:
                        raise
                out: list[dict[str, Any]] = []
                for r in rows:
                    d = dict(r)
                    try:
                        if isinstance(d.get("payload"), str):
                            d["payload"] = json.loads(d["payload"]) if d["payload"] else {}
                        if isinstance(d.get("result"), str):
                            d["result"] = json.loads(d["result"]) if d["result"] else None
                        d["payload"] = self._maybe_decrypt_json(d.get("payload"))
                        d["result"] = self._maybe_decrypt_json(d.get("result"))
                    except _JOB_NONCRITICAL_EXCEPTIONS:
                        pass
                    out.append(d)
                return out
        finally:
            conn.close()

    def count_jobs(
        self,
        *,
        domain: str | None = None,
        queue: str | None = None,
        status: str | None = None,
        owner_user_id: str | None = None,
        job_type: str | None = None,
        batch_group: str | None = None,
    ) -> int:
        """
        Return the number of jobs matching the provided filters.
        """
        conn = self._connect()
        try:
            if self.backend == "postgres":
                query = "SELECT COUNT(*) AS c FROM jobs WHERE 1=1"
                params: list[Any] = []
                if domain:
                    query += " AND domain = %s"
                    params.append(domain)
                if queue:
                    query += " AND queue = %s"
                    params.append(queue)
                if status:
                    query += " AND status = %s"
                    params.append(status)
                if owner_user_id:
                    query += " AND owner_user_id = %s"
                    params.append(owner_user_id)
                if job_type:
                    query += " AND job_type = %s"
                    params.append(job_type)
                if batch_group:
                    query += " AND batch_group = %s"
                    params.append(batch_group)
                with self._pg_cursor(conn) as cur:
                    cur.execute(query, params)
                    row = cur.fetchone()
                if not row:
                    return 0
                try:
                    return int(row["c"])
                except _JOB_NONCRITICAL_EXCEPTIONS:
                    return 0
            else:
                query = "SELECT COUNT(*) FROM jobs WHERE 1=1"
                params: list[Any] = []
                if domain:
                    query += " AND domain = ?"
                    params.append(domain)
                if queue:
                    query += " AND queue = ?"
                    params.append(queue)
                if status:
                    query += " AND status = ?"
                    params.append(status)
                if owner_user_id:
                    query += " AND owner_user_id = ?"
                    params.append(owner_user_id)
                if job_type:
                    query += " AND job_type = ?"
                    params.append(job_type)
                if batch_group:
                    query += " AND batch_group = ?"
                    params.append(batch_group)
                try:
                    row = conn.execute(query, params).fetchone()
                except sqlite3.OperationalError as exc:
                    if (
                        batch_group
                        and self._sqlite_missing_column_error(exc, "batch_group")
                        and self._sqlite_ensure_batch_group(conn)
                    ):
                        row = conn.execute(query, params).fetchone()
                    else:
                        raise
                if not row:
                    return 0
                try:
                    return int(row[0])
                except _JOB_NONCRITICAL_EXCEPTIONS:
                    return 0
        finally:
            conn.close()

    def list_job_events_after(
        self,
        *,
        after_id: int = 0,
        limit: int = 100,
        domain: str | None = None,
        queue: str | None = None,
        job_type: str | None = None,
        job_id: int | None = None,
        owner_user_id: str | None = None,
        event_types: tuple[str, ...] | list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """List raw job events after a cursor id, optionally filtered by event metadata."""

        bounded_limit = max(1, min(1000, int(limit)))
        normalized_after = max(0, int(after_id))
        normalized_types = tuple(str(v).strip() for v in (event_types or ()) if str(v).strip())
        normalized_job_id = int(job_id) if job_id is not None else None

        def _clean_filter(value: str | None) -> str | None:
            if value is None:
                return None
            cleaned = str(value).strip()
            return cleaned or None

        scalar_filters: tuple[tuple[str, Any], ...] = (
            ("domain", _clean_filter(domain)),
            ("queue", _clean_filter(queue)),
            ("job_type", _clean_filter(job_type)),
            ("job_id", normalized_job_id),
            ("owner_user_id", _clean_filter(owner_user_id)),
        )
        selected_columns = (
            "id",
            "event_type",
            "attrs_json",
            "job_id",
            "domain",
            "queue",
            "job_type",
            "owner_user_id",
            "request_id",
            "trace_id",
            "created_at",
        )

        def _row_to_event_dict(row: Any) -> dict[str, Any]:
            if isinstance(row, dict):
                return {column: row.get(column) for column in selected_columns}
            return {column: row[index] for index, column in enumerate(selected_columns)}

        conn = self._connect()
        try:
            if self.backend == "postgres":
                query = (
                    "SELECT id, event_type, attrs_json, job_id, domain, queue, job_type, owner_user_id, "
                    "request_id, trace_id, created_at FROM job_events WHERE id > %s"
                )
                params: list[Any] = [normalized_after]
                for column, value in scalar_filters:
                    if value is None:
                        continue
                    query += " AND " + job_event_filter_fragment(
                        column,
                        backend="postgres",
                    )
                    params.append(value)
                if normalized_types:
                    placeholders = ", ".join(["%s"] * len(normalized_types))
                    query += f" AND event_type IN ({placeholders})"
                    params.extend(normalized_types)
                query += " ORDER BY id ASC LIMIT %s"
                params.append(bounded_limit)
                with self._pg_cursor(conn) as cur:
                    cur.execute(query, params)
                    rows = cur.fetchall() or []
                return [_row_to_event_dict(row) for row in rows]

            query = (
                "SELECT id, event_type, attrs_json, job_id, domain, queue, job_type, owner_user_id, "
                "request_id, trace_id, created_at FROM job_events WHERE id > ?"
            )
            params = [normalized_after]
            for column, value in scalar_filters:
                if value is None:
                    continue
                query += " AND " + job_event_filter_fragment(
                    column,
                    backend="sqlite",
                )
                params.append(value)
            if normalized_types:
                placeholders = ",".join(["?"] * len(normalized_types))
                query += f" AND event_type IN ({placeholders})"
                params.extend(normalized_types)
            query += " ORDER BY id ASC LIMIT ?"
            params.append(bounded_limit)
            rows = conn.execute(query, tuple(params)).fetchall() or []
            return [_row_to_event_dict(row) for row in rows]
        finally:
            conn.close()

    def summarize_by_status(
        self,
        *,
        domain: str | None = None,
        owner_user_id: str | None = None,
    ) -> dict[str, int]:
        """
        Return a mapping of job status → count for the given filters.
        """
        conn = self._connect()
        try:
            if self.backend == "postgres":
                query = "SELECT status, COUNT(*) AS c FROM jobs WHERE 1=1"
                params: list[Any] = []
                if domain:
                    query += " AND domain = %s"
                    params.append(domain)
                if owner_user_id:
                    query += " AND owner_user_id = %s"
                    params.append(owner_user_id)
                query += " GROUP BY status"
                with self._pg_cursor(conn) as cur:
                    cur.execute(query, params)
                    rows = cur.fetchall() or []
                out: dict[str, int] = {}
                for r in rows:
                    try:
                        status_val = str(r["status"])
                        count_val = int(r["c"])
                    except _JOB_NONCRITICAL_EXCEPTIONS:
                        continue
                    if status_val:
                        out[status_val] = count_val
                return out
            else:
                query = "SELECT status, COUNT(*) FROM jobs WHERE 1=1"
                params: list[Any] = []
                if domain:
                    query += " AND domain = ?"
                    params.append(domain)
                if owner_user_id:
                    query += " AND owner_user_id = ?"
                    params.append(owner_user_id)
                query += " GROUP BY status"
                rows = conn.execute(query, params).fetchall() or []
                out: dict[str, int] = {}
                for r in rows:
                    try:
                        status_val = str(r[0])
                        count_val = int(r[1])
                    except _JOB_NONCRITICAL_EXCEPTIONS:
                        continue
                    if status_val:
                        out[status_val] = count_val
                return out
        finally:
            conn.close()

    def summarize_by_owner_and_status(
        self,
        *,
        domain: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Summarize jobs grouped by (owner_user_id, status).
        """
        conn = self._connect()
        try:
            if self.backend == "postgres":
                query = "SELECT owner_user_id, status, COUNT(*) AS c FROM jobs WHERE 1=1"
                params: list[Any] = []
                if domain:
                    query += " AND domain = %s"
                    params.append(domain)
                query += " GROUP BY owner_user_id, status"
                with self._pg_cursor(conn) as cur:
                    cur.execute(query, params)
                    rows = cur.fetchall() or []
                out: list[dict[str, Any]] = []
                for r in rows:
                    try:
                        owner = r["owner_user_id"]
                        status_val = str(r["status"])
                        count_val = int(r["c"])
                    except _JOB_NONCRITICAL_EXCEPTIONS:
                        continue
                    out.append(
                        {
                            "owner_user_id": str(owner) if owner is not None else None,
                            "status": status_val,
                            "count": count_val,
                        }
                    )
                return out
            else:
                query = "SELECT owner_user_id, status, COUNT(*) FROM jobs WHERE 1=1"
                params: list[Any] = []
                if domain:
                    query += " AND domain = ?"
                    params.append(domain)
                query += " GROUP BY owner_user_id, status"
                rows = conn.execute(query, params).fetchall() or []
                out: list[dict[str, Any]] = []
                for row in rows:
                    try:
                        owner, status_val, count_val = row
                        out.append(
                            {
                                "owner_user_id": str(owner) if owner is not None else None,
                                "status": str(status_val),
                                "count": int(count_val),
                            }
                        )
                    except _JOB_NONCRITICAL_EXCEPTIONS:
                        continue
                return out
        finally:
            conn.close()

    def _recover_expired_processing_jobs(
        self,
        *,
        domain: str | None = None,
        queue: str | None = None,
        owner_user_id: str | None = None,
        job_type: str | None = None,
    ) -> int:
        """Recover expired processing leases with their lifecycle side effects.

        Legacy nullable retry fields use the bounded schema default of three
        retries. The guarded transitions, counters, and optional durable event
        rows commit together so concurrent recovery cannot double-count or
        emit duplicate lifecycle events.
        """

        counters_enabled = JobManager._is_truthy(os.getenv("JOBS_COUNTERS_ENABLED", ""))
        outbox_enabled = JobManager._is_truthy(os.getenv("JOBS_EVENTS_OUTBOX", ""))
        batch_size = JobManager._expired_recovery_batch_size()
        recovered: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
        conn = self._connect()
        try:
            if self.backend == "postgres":
                where = ["status = 'processing'", "(leased_until IS NULL OR leased_until <= NOW())"]
                params: list[Any] = []
                for column, value in (
                    ("domain", domain),
                    ("queue", queue),
                    ("owner_user_id", owner_user_id),
                    ("job_type", job_type),
                ):
                    if value is not None:
                        where.append(f"{column} = %s")
                        params.append(value)
                with conn, self._pg_cursor(conn) as cur:
                    counter_deltas: dict[tuple[str, str, str], tuple[int, int]] = {}
                    cur.execute(
                        (
                            "SELECT id, uuid, domain, queue, job_type, owner_user_id, request_id, trace_id, "
                            "COALESCE(retry_count, 0) AS effective_retry_count, "
                            "COALESCE(max_retries, %s) AS effective_max_retries "
                            f"FROM jobs WHERE {' AND '.join(where)} "  # nosec B608
                            "ORDER BY leased_until ASC NULLS FIRST, id ASC "
                            "LIMIT %s FOR UPDATE SKIP LOCKED"
                        ),
                        (_DEFAULT_MAX_RETRIES, *params, batch_size),
                    )
                    rows = cur.fetchall() or []
                    for raw_row in rows:
                        row = dict(raw_row)
                        retry_count = int(row["effective_retry_count"])
                        max_retries = int(row["effective_max_retries"])
                        requeue = retry_count < max_retries
                        if requeue:
                            cur.execute(
                                (
                                    "UPDATE jobs SET status='queued', "
                                    "retry_count=COALESCE(retry_count, 0) + 1, "
                                    "max_retries=COALESCE(max_retries, %s), available_at=NULL, "
                                    "leased_until=NULL, worker_id=NULL, lease_id=NULL, completion_token=NULL "
                                    "WHERE id=%s AND status='processing' "
                                    "AND (leased_until IS NULL OR leased_until <= NOW()) "
                                    "AND COALESCE(retry_count, 0) < COALESCE(max_retries, %s)"
                                ),
                                (_DEFAULT_MAX_RETRIES, int(row["id"]), _DEFAULT_MAX_RETRIES),
                            )
                            event_type = "job.retry_scheduled"
                            attrs = {
                                "backoff_seconds": 0,
                                "error_code": _LEASE_EXPIRED_ERROR_CODE,
                                "retry_count": retry_count + 1,
                            }
                        else:
                            cur.execute(
                                (
                                    "UPDATE jobs SET status='failed', "
                                    "retry_count=COALESCE(retry_count, 0), "
                                    "max_retries=COALESCE(max_retries, %s), "
                                    "last_error=%s, error_message=%s, error_code=%s, completed_at=NOW(), "
                                    "leased_until=NULL, worker_id=NULL, lease_id=NULL "
                                    "WHERE id=%s AND status='processing' "
                                    "AND (leased_until IS NULL OR leased_until <= NOW()) "
                                    "AND COALESCE(retry_count, 0) >= COALESCE(max_retries, %s)"
                                ),
                                (
                                    _DEFAULT_MAX_RETRIES,
                                    _LEASE_EXPIRED_ERROR_CODE,
                                    _LEASE_EXPIRED_ERROR_MESSAGE,
                                    _LEASE_EXPIRED_ERROR_CODE,
                                    int(row["id"]),
                                    _DEFAULT_MAX_RETRIES,
                                ),
                            )
                            event_type = "job.failed"
                            attrs = {"error_code": _LEASE_EXPIRED_ERROR_CODE}
                        if cur.rowcount != 1:
                            continue

                        job = {
                            "id": int(row["id"]),
                            "uuid": row.get("uuid"),
                            "domain": row.get("domain"),
                            "queue": row.get("queue"),
                            "job_type": row.get("job_type"),
                            "owner_user_id": row.get("owner_user_id"),
                            "request_id": row.get("request_id"),
                            "trace_id": row.get("trace_id"),
                        }
                        if counters_enabled:
                            counter_key = (job["domain"], job["queue"], job["job_type"])
                            ready_delta, processing_delta = counter_deltas.get(counter_key, (0, 0))
                            counter_deltas[counter_key] = (
                                ready_delta + int(requeue),
                                processing_delta + 1,
                            )
                        if outbox_enabled:
                            cur.execute(
                                (
                                    "INSERT INTO job_events(job_id,domain,queue,job_type,event_type,attrs_json,owner_user_id,request_id,trace_id,created_at) "
                                    "VALUES(%s,%s,%s,%s,%s,%s::jsonb,%s,%s,%s,NOW())"
                                ),
                                (
                                    job["id"],
                                    job["domain"],
                                    job["queue"],
                                    job["job_type"],
                                    event_type,
                                    json.dumps(attrs),
                                    job["owner_user_id"],
                                    job["request_id"],
                                    job["trace_id"],
                                ),
                            )
                        recovered.append((event_type, job, attrs))
                    for counter_key in sorted(counter_deltas):
                        ready_delta, processing_delta = counter_deltas[counter_key]
                        cur.execute(
                            (
                                "INSERT INTO job_counters(domain,queue,job_type,ready_count,scheduled_count,processing_count,quarantined_count) "
                                "VALUES(%s,%s,%s,%s,0,0,0) ON CONFLICT (domain,queue,job_type) DO UPDATE SET "
                                "ready_count=job_counters.ready_count + EXCLUDED.ready_count, "
                                "processing_count=GREATEST(job_counters.processing_count - %s, 0), updated_at=NOW()"
                            ),
                            (*counter_key, ready_delta, processing_delta),
                        )
            else:
                where = [
                    "status = 'processing'",
                    "(leased_until IS NULL OR leased_until <= DATETIME('now'))",
                ]
                params = []
                for column, value in (
                    ("domain", domain),
                    ("queue", queue),
                    ("owner_user_id", owner_user_id),
                    ("job_type", job_type),
                ):
                    if value is not None:
                        where.append(f"{column} = ?")
                        params.append(value)
                scoped_where = " AND ".join(where)
                # Healthy acquisitions pay only a scoped read; take the write
                # lock only when an expired row may need recovery.
                precheck = conn.execute(
                    f"SELECT 1 FROM jobs WHERE {scoped_where} LIMIT 1",  # nosec B608
                    tuple(params),
                ).fetchone()
                if not precheck:
                    return 0
                with conn:
                    conn.execute("BEGIN IMMEDIATE")
                    rows = conn.execute(
                        (
                            "SELECT id, uuid, domain, queue, job_type, owner_user_id, request_id, trace_id, "
                            "COALESCE(retry_count, 0) AS effective_retry_count, "
                            "COALESCE(max_retries, ?) AS effective_max_retries "
                            f"FROM jobs WHERE {scoped_where} "  # nosec B608
                            "ORDER BY leased_until ASC, id ASC LIMIT ?"
                        ),
                        (_DEFAULT_MAX_RETRIES, *params, batch_size),
                    ).fetchall()
                    for raw_row in rows:
                        row = dict(raw_row)
                        retry_count = int(row["effective_retry_count"])
                        max_retries = int(row["effective_max_retries"])
                        requeue = retry_count < max_retries
                        if requeue:
                            changed = conn.execute(
                                (
                                    "UPDATE jobs SET status='queued', "
                                    "retry_count=COALESCE(retry_count, 0) + 1, "
                                    "max_retries=COALESCE(max_retries, ?), available_at=NULL, "
                                    "leased_until=NULL, worker_id=NULL, lease_id=NULL, completion_token=NULL "
                                    "WHERE id=? AND status='processing' "
                                    "AND (leased_until IS NULL OR leased_until <= DATETIME('now')) "
                                    "AND COALESCE(retry_count, 0) < COALESCE(max_retries, ?)"
                                ),
                                (_DEFAULT_MAX_RETRIES, int(row["id"]), _DEFAULT_MAX_RETRIES),
                            )
                            event_type = "job.retry_scheduled"
                            attrs = {
                                "backoff_seconds": 0,
                                "error_code": _LEASE_EXPIRED_ERROR_CODE,
                                "retry_count": retry_count + 1,
                            }
                        else:
                            changed = conn.execute(
                                (
                                    "UPDATE jobs SET status='failed', "
                                    "retry_count=COALESCE(retry_count, 0), "
                                    "max_retries=COALESCE(max_retries, ?), "
                                    "last_error=?, error_message=?, error_code=?, completed_at=DATETIME('now'), "
                                    "leased_until=NULL, worker_id=NULL, lease_id=NULL "
                                    "WHERE id=? AND status='processing' "
                                    "AND (leased_until IS NULL OR leased_until <= DATETIME('now')) "
                                    "AND COALESCE(retry_count, 0) >= COALESCE(max_retries, ?)"
                                ),
                                (
                                    _DEFAULT_MAX_RETRIES,
                                    _LEASE_EXPIRED_ERROR_CODE,
                                    _LEASE_EXPIRED_ERROR_MESSAGE,
                                    _LEASE_EXPIRED_ERROR_CODE,
                                    int(row["id"]),
                                    _DEFAULT_MAX_RETRIES,
                                ),
                            )
                            event_type = "job.failed"
                            attrs = {"error_code": _LEASE_EXPIRED_ERROR_CODE}
                        if changed.rowcount != 1:
                            continue

                        job = {
                            "id": int(row["id"]),
                            "uuid": row.get("uuid"),
                            "domain": row.get("domain"),
                            "queue": row.get("queue"),
                            "job_type": row.get("job_type"),
                            "owner_user_id": row.get("owner_user_id"),
                            "request_id": row.get("request_id"),
                            "trace_id": row.get("trace_id"),
                        }
                        if counters_enabled:
                            if requeue:
                                conn.execute(
                                    (
                                        "INSERT INTO job_counters(domain,queue,job_type,ready_count,scheduled_count,processing_count,quarantined_count) "
                                        "VALUES(?,?,?,1,0,0,0) ON CONFLICT(domain,queue,job_type) DO UPDATE SET "
                                        "ready_count=ready_count + 1, "
                                        "processing_count=CASE WHEN processing_count>0 THEN processing_count-1 ELSE 0 END, "
                                        "updated_at=DATETIME('now')"
                                    ),
                                    (job["domain"], job["queue"], job["job_type"]),
                                )
                            else:
                                conn.execute(
                                    (
                                        "INSERT INTO job_counters(domain,queue,job_type,ready_count,scheduled_count,processing_count,quarantined_count) "
                                        "VALUES(?,?,?,0,0,0,0) ON CONFLICT(domain,queue,job_type) DO UPDATE SET "
                                        "processing_count=CASE WHEN processing_count>0 THEN processing_count-1 ELSE 0 END, "
                                        "updated_at=DATETIME('now')"
                                    ),
                                    (job["domain"], job["queue"], job["job_type"]),
                                )
                        if outbox_enabled:
                            conn.execute(
                                (
                                    "INSERT INTO job_events(job_id,domain,queue,job_type,event_type,attrs_json,owner_user_id,request_id,trace_id,created_at) "
                                    "VALUES(?,?,?,?,?,?,?,?,?,DATETIME('now'))"
                                ),
                                (
                                    job["id"],
                                    job["domain"],
                                    job["queue"],
                                    job["job_type"],
                                    event_type,
                                    json.dumps(attrs),
                                    job["owner_user_id"],
                                    job["request_id"],
                                    job["trace_id"],
                                ),
                            )
                        recovered.append((event_type, job, attrs))
        finally:
            _close_connection_nonfatal(conn, operation="expired-job recovery")

        for event_type, job, attrs in recovered:
            with contextlib.suppress(_JOB_NONCRITICAL_EXCEPTIONS):
                if outbox_enabled:
                    submit_job_audit_event(event_type, job=job, attrs=attrs)
                else:
                    emit_job_event(event_type, job=job, attrs=attrs)
            with contextlib.suppress(_JOB_NONCRITICAL_EXCEPTIONS):
                if event_type == "job.retry_scheduled":
                    increment_retries(job)
                else:
                    increment_failures(job, reason="terminal")
            with contextlib.suppress(_JOB_NONCRITICAL_EXCEPTIONS):
                self._update_gauges(
                    domain=job.get("domain"),
                    queue=job.get("queue"),
                    job_type=job.get("job_type"),
                )
        return len(recovered)

    def acquire_next_job(
        self,
        *,
        domain: str,
        queue: str,
        lease_seconds: int,
        worker_id: str,
        owner_user_id: str | None = None,
        job_type: str | None = None,
    ) -> dict[str, Any] | None:
        """Atomically acquire the next eligible job and start a lease.

        Selection order (both SQLite and Postgres): priority ASC (lower numeric is higher priority),
        then oldest first by COALESCE(available_at, created_at), then id ASC.

        Before selection, expired processing jobs are atomically requeued or
        terminally failed according to their retry budget. Selection then
        acquires only ready queued jobs.

        When provided, `job_type` restricts acquisition to matching jobs in the
        selected domain and queue.
        """
        # Honor global acquire gate for graceful shutdown
        _test_mode = _is_test_mode()
        if _test_mode:
            with contextlib.suppress(_JOB_NONCRITICAL_EXCEPTIONS):
                logger.info(
                    f"[JM TEST] acquire_next_job enter backend={self.backend} domain={domain} queue={queue} job_type={job_type} owner={owner_user_id} gate={JobManager._ACQUIRE_GATE_ENABLED} db={(str(self.db_path) if getattr(self, 'db_path', None) else self.db_url)}"
                )
        if JobManager._ACQUIRE_GATE_ENABLED:
            with contextlib.suppress(_JOB_NONCRITICAL_EXCEPTIONS):
                logger.debug("Jobs acquire gate enabled; declining new acquisition")
            return None
        # Queue-specific pause/drain gate
        flags = self._get_queue_flags(domain, queue)
        if _test_mode:
            with contextlib.suppress(_JOB_NONCRITICAL_EXCEPTIONS):
                logger.info(f"[JM TEST] queue flags paused={flags.get('paused')} drain={flags.get('drain')}")
        if flags.get("paused"):
            return None
        reconciliation_scope = {"domain": domain, "queue": queue}
        if owner_user_id is not None:
            reconciliation_scope["owner_user_id"] = owner_user_id
        self._reconcile_terminal_dependents(**reconciliation_scope)
        # Domain/user inflight limit
        max_inflight = 0
        try:
            max_inflight = self._quota_get("JOBS_QUOTA_MAX_INFLIGHT", domain, owner_user_id)
            if _test_mode:
                with contextlib.suppress(_JOB_NONCRITICAL_EXCEPTIONS):
                    logger.info(f"[JM TEST] inflight quota={max_inflight} owner={owner_user_id}")
            if max_inflight and owner_user_id:
                conn_q = self._connect()
                try:
                    if self.backend == "postgres":
                        with self._pg_cursor(conn_q) as curq:
                            curq.execute(
                                "SELECT COUNT(*) AS c FROM jobs WHERE domain=%s AND owner_user_id=%s AND status='processing' AND leased_until IS NOT NULL AND leased_until > NOW()",
                                (domain, owner_user_id),
                            )
                            _row = curq.fetchone()
                            if int(_row.get("c") if isinstance(_row, dict) else 0) >= max_inflight:
                                return None
                    else:
                        rowc = conn_q.execute(
                            "SELECT COUNT(*) FROM jobs WHERE domain=? AND owner_user_id=? AND status='processing' AND leased_until IS NOT NULL AND leased_until > DATETIME('now')",
                            (domain, owner_user_id),
                        ).fetchone()
                        if int(rowc[0] or 0) >= max_inflight:
                            return None
                finally:
                    with contextlib.suppress(_JOB_NONCRITICAL_EXCEPTIONS):
                        conn_q.close()
        except _JOB_NONCRITICAL_EXCEPTIONS:
            pass
        max_lease = int(os.getenv("JOBS_LEASE_MAX_SECONDS", "3600") or "3600")
        # Adaptive default when seconds <= 0 and enabled
        try:
            req = int(lease_seconds)
        except _JOB_NONCRITICAL_EXCEPTIONS:
            req = 0
        if req <= 0 and JobManager._is_truthy(os.getenv("JOBS_ADAPTIVE_LEASE_ENABLE", "")):
            try:
                req = self._adaptive_lease_seconds(domain, queue, job_type)
            except _JOB_NONCRITICAL_EXCEPTIONS:
                req = 30
        lease_seconds = max(1, min(max_lease, int(req)))
        self._recover_expired_processing_jobs(
            domain=domain,
            queue=queue,
            owner_user_id=owner_user_id,
            job_type=job_type,
        )
        self._reconcile_terminal_dependents(**reconciliation_scope)
        conn = self._connect()
        acquired: dict[str, Any] | None = None
        try:
            if self.backend == "postgres":
                result = _postgres_acquire_job(
                    conn,
                    self._pg_cursor,
                    command=AcquireJobCommand(
                        domain=domain,
                        queue=queue,
                        lease_seconds=lease_seconds,
                        worker_id=worker_id,
                        lease_id=str(_uuid.uuid4()),
                        owner_user_id=owner_user_id,
                        job_type=job_type,
                        max_inflight_quota=max_inflight,
                        priority_direction=self._priority_dir_for(domain, backend="pg"),
                        tie_break=self._tie_break_for(domain, backend="pg"),
                        single_update=JobManager._is_truthy(
                            os.getenv("JOBS_PG_SINGLE_UPDATE_ACQUIRE", "")
                        ),
                    ),
                    counters_enabled=JobManager._is_truthy(os.getenv("JOBS_COUNTERS_ENABLED", "")),
                    now=self._clock.now_utc(),
                )
                if result.outcome is not OperationOutcome.APPLIED or result.row is None:
                    return None
                acquired = result.row
            else:
                result = _sqlite_acquire_job(
                    conn,
                    command=AcquireJobCommand(
                        domain=domain,
                        queue=queue,
                        lease_seconds=lease_seconds,
                        worker_id=worker_id,
                        lease_id=str(_uuid.uuid4()),
                        owner_user_id=owner_user_id,
                        job_type=job_type,
                        max_inflight_quota=max_inflight,
                        priority_direction=self._priority_dir_for(domain, backend="sqlite"),
                        tie_break=self._tie_break_for(domain, backend="sqlite"),
                        single_update=JobManager._is_truthy(
                            os.getenv("JOBS_SQLITE_SINGLE_UPDATE_ACQUIRE", "")
                        ),
                    ),
                    counters_enabled=JobManager._is_truthy(os.getenv("JOBS_COUNTERS_ENABLED", "")),
                    now=self._clock.now_utc(),
                )
                if result.outcome is not OperationOutcome.APPLIED or result.row is None:
                    return None
                acquired = result.row
                if _test_mode:
                    with contextlib.suppress(_JOB_NONCRITICAL_EXCEPTIONS):
                        logger.info(
                            f"[JM TEST] acquired id={acquired.get('id')} status={acquired.get('status')} leased_until={acquired.get('leased_until')} worker_id={acquired.get('worker_id')} lease_id={acquired.get('lease_id')}"
                        )

            if acquired is None:
                return None

            # Everything below is observational and must run only after commit.
            with contextlib.suppress(_JOB_NONCRITICAL_EXCEPTIONS):
                self._assert_invariants(acquired)
            try:
                policy = self._get_sla_policy(
                    str(acquired.get("domain")),
                    str(acquired.get("queue")),
                    str(acquired.get("job_type")),
                )
                created_at = _parse_dt(acquired.get("created_at"))
                acquired_at = _parse_dt(acquired.get("acquired_at"))
                threshold = policy.get("max_queue_latency_seconds") if policy else None
                if policy and policy.get("enabled") in (True, 1) and created_at and acquired_at and threshold is not None:
                    queue_latency = max(0.0, (acquired_at - created_at).total_seconds())
                    if queue_latency > float(threshold):
                        self._record_sla_breach(
                            int(acquired["id"]),
                            str(acquired.get("domain")),
                            str(acquired.get("queue")),
                            str(acquired.get("job_type")),
                            "queue_latency",
                            queue_latency,
                            float(threshold),
                        )
            except _JOB_NONCRITICAL_EXCEPTIONS:
                pass
            with contextlib.suppress(_JOB_NONCRITICAL_EXCEPTIONS):
                observe_queue_latency(
                    acquired,
                    _parse_dt(acquired.get("acquired_at")),
                    _parse_dt(acquired.get("created_at")),
                )
            if isinstance(acquired.get("payload"), str):
                with contextlib.suppress(_JOB_NONCRITICAL_EXCEPTIONS):
                    acquired["payload"] = json.loads(acquired["payload"]) if acquired["payload"] else {}
            with contextlib.suppress(_JOB_NONCRITICAL_EXCEPTIONS):
                acquired["payload"] = self._maybe_decrypt_json(acquired.get("payload"))
            with contextlib.suppress(_JOB_NONCRITICAL_EXCEPTIONS):
                self._update_gauges(domain=domain, queue=queue, job_type=acquired.get("job_type"))
            with contextlib.suppress(_JOB_NONCRITICAL_EXCEPTIONS):
                with job_span("job.acquire", job=acquired):
                    pass
            with contextlib.suppress(_JOB_NONCRITICAL_EXCEPTIONS):
                emit_job_event(
                    "job.acquired",
                    job=acquired,
                    attrs={
                        "worker_id": worker_id,
                        "owner_user_id": acquired.get("owner_user_id"),
                        "retry_count": int(acquired.get("retry_count") or 0),
                    },
                )
            if _test_mode:
                with contextlib.suppress(_JOB_NONCRITICAL_EXCEPTIONS):
                    JobManager._LAST_ACQUIRED_TEST[(domain, queue)] = dict(acquired)
                if self.backend == "sqlite":
                    try:
                        cq = conn.execute(
                            "SELECT COUNT(*) FROM jobs WHERE domain=? AND queue=? AND status='queued'",
                            (domain, queue),
                        ).fetchone()[0]
                        cp = conn.execute(
                            "SELECT COUNT(*) FROM jobs WHERE domain=? AND queue=? AND status='processing'",
                            (domain, queue),
                        ).fetchone()[0]
                        logger.info(f"[JM TEST] post-acquire counts queued={cq} processing={cp}")
                    except _JOB_NONCRITICAL_EXCEPTIONS:
                        pass
            return acquired
        finally:
            conn.close()

    def renew_job_lease(
        self,
        job_id: int,
        *,
        seconds: int,
        worker_id: str | None = None,
        lease_id: str | None = None,
        progress_percent: float | None = None,
        progress_message: str | None = None,
        enforce: bool | None = None,
    ) -> bool:
        """Extend the lease on a processing job.

        If `enforce` is True (or `JOBS_ENFORCE_LEASE_ACK` env is truthy), the
        current `worker_id`/`lease_id` must match to succeed. If values are not
        provided while enforcement is enabled, the operation will be rejected.
        """
        max_lease = int(os.getenv("JOBS_LEASE_MAX_SECONDS", "3600") or "3600")
        seconds = max(1, min(max_lease, int(seconds)))
        if enforce is None:
            enforce = self._should_enforce_ack()
        conn = self._connect()
        try:
            if self.backend == "postgres":
                result = _postgres_renew_lease(
                    conn,
                    self._pg_cursor,
                    command=RenewLeaseCommand(
                        job_id=int(job_id),
                        seconds=int(seconds),
                        enforce=bool(enforce),
                        worker_id=worker_id,
                        lease_id=lease_id,
                        progress_percent=progress_percent,
                        progress_message=progress_message,
                    ),
                    now=self._clock.now_utc(),
                )
                if result.outcome is not OperationOutcome.APPLIED:
                    return False
                with contextlib.suppress(_JOB_NONCRITICAL_EXCEPTIONS):
                    emit_job_event(
                        "job.lease_renewed",
                        job={"id": int(job_id)},
                        attrs={"seconds": int(seconds)},
                    )
                return True
            else:
                result = _sqlite_renew_lease(
                    conn,
                    command=RenewLeaseCommand(
                        job_id=int(job_id),
                        seconds=int(seconds),
                        enforce=bool(enforce),
                        worker_id=worker_id,
                        lease_id=lease_id,
                        progress_percent=progress_percent,
                        progress_message=progress_message,
                    ),
                    now=self._clock.now_utc(),
                )
                if result.outcome is not OperationOutcome.APPLIED:
                    return False
                with contextlib.suppress(_JOB_NONCRITICAL_EXCEPTIONS):
                    emit_job_event(
                        "job.lease_renewed",
                        job={"id": int(job_id)},
                        attrs={"seconds": int(seconds)},
                    )
                return True
        finally:
            conn.close()

    def update_job_progress(
        self,
        job_id: int,
        *,
        progress_percent: float | None = None,
        progress_message: str | None = None,
    ) -> bool:
        """Update progress fields on a job without touching lease state."""
        if progress_percent is None and progress_message is None:
            return False
        if progress_percent is not None:
            progress_percent = max(0.0, min(100.0, float(progress_percent)))
        conn = self._connect()
        try:
            if self.backend == "postgres":
                with conn, self._pg_cursor(conn) as cur:
                    sets: list[str] = []
                    params: list[Any] = []
                    if progress_percent is not None:
                        sets.append("progress_percent = %s")
                        params.append(float(progress_percent))
                    if progress_message is not None:
                        sets.append("progress_message = %s")
                        params.append(str(progress_message))
                    if not sets:
                        return False
                    params.append(int(job_id))
                    cur.execute(
                        f"UPDATE jobs SET {', '.join(sets)}, updated_at = NOW() WHERE id = %s",  # nosec B608
                        tuple(params),
                    )
                    return cur.rowcount > 0
            else:
                with conn:
                    sets2: list[str] = []
                    params2: list[Any] = []
                    if progress_percent is not None:
                        sets2.append("progress_percent = ?")
                        params2.append(float(progress_percent))
                    if progress_message is not None:
                        sets2.append("progress_message = ?")
                        params2.append(str(progress_message))
                    if not sets2:
                        return False
                    params2.append(int(job_id))
                    sql = f"UPDATE jobs SET {', '.join(sets2)}, updated_at = DATETIME('now') WHERE id = ?"  # nosec B608
                    cur2 = conn.execute(sql, tuple(params2))
                    return (cur2.rowcount or 0) > 0
        finally:
            conn.close()

    def update_job_result(
        self,
        job_id: int,
        *,
        result: dict[str, Any],
        merge: bool = True,
    ) -> bool:
        """Update the result payload on a job without changing status."""
        if result is None:
            return False
        job = self.get_job(int(job_id))
        if not job:
            return False
        existing = job.get("result")
        if merge and isinstance(existing, dict) and isinstance(result, dict):
            res_obj: dict[str, Any] = dict(existing)
            res_obj.update(result)
        else:
            res_obj = result

        max_bytes = int(os.getenv("JOBS_MAX_JSON_BYTES", "1048576") or "1048576")
        truncate = JobManager._is_truthy(os.getenv("JOBS_JSON_TRUNCATE", ""))
        try:
            res_json = json.dumps(res_obj)
        except (TypeError, ValueError) as exc:
            raise ValueError("Result payload must be JSON-serializable") from exc  # noqa: TRY003
        res_bytes = len(res_json.encode("utf-8"))
        if res_bytes > max_bytes:
            if truncate:
                res_obj = {"_truncated": True, "len_bytes": res_bytes}
            else:
                raise ValueError(f"Result too large: {res_bytes} bytes > limit {max_bytes}")  # noqa: TRY003

        with contextlib.suppress(_JOB_NONCRITICAL_EXCEPTIONS):
            res_obj = self._maybe_encrypt_json(res_obj, str(job.get("domain")))

        conn = self._connect()
        try:
            if self.backend == "postgres":
                with conn, self._pg_cursor(conn) as cur:
                    cur.execute(
                        "UPDATE jobs SET result = %s::jsonb, updated_at = NOW() WHERE id = %s",
                        (json.dumps(res_obj) if res_obj is not None else None, int(job_id)),
                    )
                    return cur.rowcount > 0
            else:
                with conn:
                    cur = conn.execute(
                        "UPDATE jobs SET result = ?, updated_at = DATETIME('now') WHERE id = ?",
                        (json.dumps(res_obj) if res_obj is not None else None, int(job_id)),
                    )
                    return (cur.rowcount or 0) > 0
        finally:
            conn.close()

    def terminalize_job_from_worker(
        self,
        *,
        job_id: int,
        job_uuid: str,
        owner_user_id: str,
        domain: str,
        queue: str,
        job_type: str,
        worker_id: str,
        lease_id: str,
        completion_token: str,
        status: str,
        error_code: str,
        error_message: str,
    ) -> str:
        """Apply one exact worker terminal CAS without numeric-ID fallback."""
        if not _is_slides_generation_scope(domain, queue, job_type):
            return "CONFLICT"
        if status not in {"failed", "cancelled"}:
            raise ValueError("terminal status must be failed or cancelled")
        if not isinstance(error_code, str) or re.fullmatch(r"[a-z][a-z0-9_.-]{0,127}", error_code) is None:
            raise ValueError("terminal error_code is invalid")
        if not isinstance(error_message, str) or len(error_message) > 1024:
            raise ValueError("terminal error_message exceeds 1024 characters")
        correlations = (
            job_uuid,
            owner_user_id,
            domain,
            queue,
            job_type,
            worker_id,
            lease_id,
            completion_token,
        )
        if not all(isinstance(value, str) and value for value in correlations):
            raise ValueError("terminal correlation values must be nonblank strings")

        event_type = "job.failed" if status == "failed" else "job.cancelled"
        event_attrs = {"error_code": error_code} if status == "failed" else {"reason": error_message, "terminal": True}
        applied_job: dict[str, Any] | None = None
        row = None
        conn = self._connect()
        try:
            if self.backend == "postgres":
                with conn, self._pg_cursor(conn) as cur:
                    cur.execute(
                        """
                        UPDATE jobs
                        SET status=%s, error_code=%s, error_message=%s, last_error=NULL,
                            completion_token=%s, completed_at=NOW(), leased_until=NULL,
                            cancelled_at=CASE WHEN %s='cancelled' THEN NOW() ELSE cancelled_at END,
                            cancellation_reason=CASE WHEN %s='cancelled' THEN %s ELSE cancellation_reason END
                        WHERE id=%s AND status='processing' AND uuid=%s
                          AND owner_user_id IS NOT DISTINCT FROM %s
                          AND domain=%s AND queue=%s AND job_type=%s
                          AND worker_id=%s AND lease_id=%s
                          AND leased_until > NOW()
                          AND (completion_token IS NULL OR completion_token=%s)
                        RETURNING *
                        """,
                        (
                            status,
                            error_code,
                            error_message,
                            completion_token,
                            status,
                            status,
                            error_message,
                            int(job_id),
                            job_uuid,
                            owner_user_id,
                            domain,
                            queue,
                            job_type,
                            worker_id,
                            lease_id,
                            completion_token,
                        ),
                    )
                    applied = cur.fetchone()
                    if applied is not None:
                        applied_job = dict(applied)
                        if JobManager._is_truthy(os.getenv("JOBS_COUNTERS_ENABLED", "")):
                            cur.execute(
                                "UPDATE job_counters SET processing_count = "
                                "GREATEST(processing_count - 1, 0), updated_at = NOW() "
                                "WHERE domain=%s AND queue=%s AND job_type=%s",
                                (domain, queue, job_type),
                            )
                        cur.execute(
                            "INSERT INTO job_events("
                            "job_id, domain, queue, job_type, event_type, attrs_json, "
                            "owner_user_id, request_id, trace_id, created_at"
                            ") VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())",
                            (
                                int(job_id),
                                domain,
                                queue,
                                job_type,
                                event_type,
                                json.dumps(event_attrs),
                                owner_user_id,
                                applied_job.get("request_id"),
                                applied_job.get("trace_id"),
                            ),
                        )
                    else:
                        cur.execute("SELECT * FROM jobs WHERE id=%s", (int(job_id),))
                        row = cur.fetchone()
            else:
                now_utc = self._clock.now_utc().astimezone(_tz.utc)
                now_sql = now_utc.strftime("%Y-%m-%d %H:%M:%S")
                with conn:
                    updated = conn.execute(
                        """
                        UPDATE jobs
                        SET status=?, error_code=?, error_message=?, last_error=NULL, completion_token=?,
                            completed_at=?, leased_until=NULL,
                            cancelled_at=CASE WHEN ?='cancelled' THEN ? ELSE cancelled_at END,
                            cancellation_reason=CASE WHEN ?='cancelled' THEN ? ELSE cancellation_reason END
                        WHERE id=? AND status='processing' AND uuid=?
                          AND owner_user_id IS ?
                          AND domain=? AND queue=? AND job_type=?
                          AND worker_id=? AND lease_id=?
                          AND leased_until > ?
                          AND (completion_token IS NULL OR completion_token=?)
                        """,
                        (
                            status,
                            error_code,
                            error_message,
                            completion_token,
                            now_sql,
                            status,
                            now_sql,
                            status,
                            error_message,
                            int(job_id),
                            job_uuid,
                            owner_user_id,
                            domain,
                            queue,
                            job_type,
                            worker_id,
                            lease_id,
                            now_sql,
                            completion_token,
                        ),
                    )
                    if updated.rowcount == 1:
                        applied = conn.execute(
                            "SELECT * FROM jobs WHERE id=?",
                            (int(job_id),),
                        ).fetchone()
                        if applied is None:
                            raise RuntimeError("terminalized job disappeared before bookkeeping")
                        applied_job = dict(applied)
                        if JobManager._is_truthy(os.getenv("JOBS_COUNTERS_ENABLED", "")):
                            conn.execute(
                                "UPDATE job_counters SET processing_count = "
                                "CASE WHEN processing_count>0 THEN processing_count-1 ELSE 0 END, "
                                "updated_at = DATETIME('now') "
                                "WHERE domain=? AND queue=? AND job_type=?",
                                (domain, queue, job_type),
                            )
                        conn.execute(
                            "INSERT INTO job_events("
                            "job_id, domain, queue, job_type, event_type, attrs_json, "
                            "owner_user_id, request_id, trace_id, created_at"
                            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, DATETIME('now'))",
                            (
                                int(job_id),
                                domain,
                                queue,
                                job_type,
                                event_type,
                                json.dumps(event_attrs),
                                owner_user_id,
                                applied_job.get("request_id"),
                                applied_job.get("trace_id"),
                            ),
                        )
                    else:
                        row = conn.execute(
                            "SELECT * FROM jobs WHERE id=?",
                            (int(job_id),),
                        ).fetchone()
            if applied_job is not None:
                with contextlib.suppress(_JOB_NONCRITICAL_EXCEPTIONS):
                    self._update_gauges(domain=domain, queue=queue, job_type=job_type)
                if status == "failed":
                    with contextlib.suppress(_JOB_NONCRITICAL_EXCEPTIONS):
                        increment_failures(applied_job, reason="terminal")
                    with contextlib.suppress(_JOB_NONCRITICAL_EXCEPTIONS):
                        from .metrics import increment_failures_by_code

                        increment_failures_by_code(applied_job, error_code)
                else:
                    with contextlib.suppress(_JOB_NONCRITICAL_EXCEPTIONS):
                        increment_cancelled(applied_job)
                if not JobManager._is_truthy(os.getenv("JOBS_EVENTS_OUTBOX", "")):
                    with contextlib.suppress(_JOB_NONCRITICAL_EXCEPTIONS):
                        emit_job_event(event_type, job=applied_job, attrs=event_attrs)
                with contextlib.suppress(_JOB_NONCRITICAL_EXCEPTIONS):
                    submit_job_audit_event(event_type, job=applied_job, attrs=event_attrs)
                return "APPLIED"
            if row is None:
                return "MISSING"
            stored = dict(row)
            stable_correlation = (
                stored.get("uuid") == job_uuid
                and stored.get("owner_user_id") == owner_user_id
                and stored.get("domain") == domain
                and stored.get("queue") == queue
                and stored.get("job_type") == job_type
            )
            exact_correlation = (
                stable_correlation and stored.get("worker_id") == worker_id and stored.get("lease_id") == lease_id
            )
            identical_terminal = (
                stored.get("status") == status
                and stored.get("completion_token") == completion_token
                and stored.get("error_code") == error_code
                and stored.get("error_message") == error_message
            )
            if exact_correlation and identical_terminal:
                return "IDEMPOTENT"
            terminal_status = stored.get("status")
            worker_terminal_winner = (
                exact_correlation
                and terminal_status in {"failed", "cancelled"}
                and stored.get("completion_token") == stored.get("lease_id")
                and stored.get("last_error") is None
                and stored.get("completed_at") is not None
            )
            if (
                stable_correlation
                and terminal_status in {"failed", "cancelled", "quarantined"}
                and not worker_terminal_winner
            ):
                return "ALREADY_TERMINAL"
            return "CONFLICT"
        finally:
            conn.close()

    def terminalize_slides_generation_job_from_reconciler(
        self,
        *,
        job_uuid: str,
        owner_user_id: str,
        expected_status: str,
        status: str,
        error_code: str,
        error_message: str,
        completion_token: str,
        job_id: int | None = None,
        require_processing_lease_expired: bool = False,
    ) -> str:
        """Apply one UUID-authoritative Slides reconciliation terminal CAS."""
        if expected_status not in {"queued", "processing"}:
            raise ValueError("reconciler expected status must be queued or processing")
        if status not in {"failed", "cancelled"}:
            raise ValueError("terminal status must be failed or cancelled")
        if not isinstance(error_code, str) or re.fullmatch(r"[a-z][a-z0-9_.-]{0,127}", error_code) is None:
            raise ValueError("terminal error_code is invalid")
        if not isinstance(error_message, str) or len(error_message) > 1024:
            raise ValueError("terminal error_message exceeds 1024 characters")
        if not all(
            isinstance(value, str) and value.strip()
            for value in (job_uuid, owner_user_id, completion_token)
        ):
            raise ValueError("terminal correlation values must be nonblank strings")
        if job_id is not None and (
            isinstance(job_id, bool) or not isinstance(job_id, int) or job_id <= 0
        ):
            raise ValueError("job_id hint must be a positive integer")
        if not isinstance(require_processing_lease_expired, bool):
            raise ValueError("require_processing_lease_expired must be a boolean")

        domain = _SLIDES_GENERATION_DOMAIN
        queue = _SLIDES_GENERATION_QUEUE
        job_type = _SLIDES_GENERATION_JOB_TYPE
        event_type = "job.failed" if status == "failed" else "job.cancelled"
        event_attrs = (
            {"error_code": error_code}
            if status == "failed"
            else {"reason": error_message, "terminal": True}
        )
        applied_job: dict[str, Any] | None = None
        row = None
        unsafe_correlation = False
        conn = self._connect()
        try:
            if self.backend == "postgres":
                lease_clause = (
                    " AND (leased_until IS NULL OR leased_until <= NOW())"
                    if expected_status == "processing" and require_processing_lease_expired
                    else ""
                )
                with conn, self._pg_cursor(conn) as cur:
                    cur.execute(
                        "SELECT pg_advisory_xact_lock(%s)",
                        (
                            self._pg_advisory_key(
                                *_SLIDES_GENERATION_CORRELATION_LOCK_PARTS
                            ),
                        ),
                    )
                    if not self._slides_generation_ready_in_connection(
                        conn,
                        cursor=cur,
                    ):
                        raise SlidesGenerationJobsUnavailableError(
                            "presentation.generate Jobs coordination is unavailable"
                        )
                    cur.execute(
                        "SELECT * FROM jobs WHERE uuid=%s "
                        "ORDER BY id LIMIT 2 FOR UPDATE",
                        (job_uuid,),
                    )
                    authority_rows = list(cur.fetchall() or [])
                    if len(authority_rows) > 1:
                        cur.execute(
                            """
                            UPDATE slides_standalone_reconciliation
                            SET diagnostic_code=CASE
                                  WHEN diagnostic_code='duplicate_archive_uuid'
                                  THEN diagnostic_code
                                  ELSE 'ambiguous_generation_legacy_row' END,
                                diagnostic_count=CASE
                                  WHEN diagnostic_code='duplicate_archive_uuid'
                                  THEN diagnostic_count
                                  ELSE GREATEST(diagnostic_count, %s) END,
                                diagnostic_at=CASE
                                  WHEN diagnostic_code='duplicate_archive_uuid'
                                  THEN diagnostic_at ELSE NOW() END
                            WHERE singleton_id=1
                            """,
                            (len(authority_rows),),
                        )
                        unsafe_correlation = True
                    row = authority_rows[0] if authority_rows else None
                    authority_id = (
                        int(row["id"])
                        if row is not None and not unsafe_correlation
                        else None
                    )
                    params: list[Any] = [
                        status,
                        error_code,
                        error_message,
                        completion_token,
                        status,
                        status,
                        error_message,
                        job_uuid,
                        owner_user_id,
                        domain,
                        queue,
                        job_type,
                        expected_status,
                        (
                            job_id
                            if job_id is not None and not unsafe_correlation
                            else authority_id
                        ),
                    ]
                    params.append(completion_token)
                    cur.execute(
                        (
                            "UPDATE jobs SET status=%s, error_code=%s, error_message=%s, "
                            "completion_token=%s, completed_at=NOW(), leased_until=NULL, "
                            "cancelled_at=CASE WHEN %s='cancelled' THEN NOW() ELSE cancelled_at END, "
                            "cancellation_reason=CASE WHEN %s='cancelled' THEN %s ELSE cancellation_reason END "
                            "WHERE uuid=%s AND owner_user_id IS NOT DISTINCT FROM %s "
                            "AND domain=%s AND queue=%s AND job_type=%s AND status=%s "
                            f"AND id=%s{lease_clause} "  # nosec B608
                            "AND (completion_token IS NULL OR completion_token=%s) RETURNING *"
                        ),
                        tuple(params),
                    )
                    applied = cur.fetchone()
                    if applied is not None:
                        applied_job = dict(applied)
                        if JobManager._is_truthy(os.getenv("JOBS_COUNTERS_ENABLED", "")):
                            available_at = _parse_dt(applied_job.get("available_at"))
                            if available_at is not None and available_at.tzinfo is None:
                                available_at = available_at.replace(tzinfo=_tz.utc)
                            scheduled = available_at is not None and available_at > self._clock.now_utc()
                            ready_delta = int(expected_status == "queued" and not scheduled)
                            scheduled_delta = int(expected_status == "queued" and scheduled)
                            processing_delta = int(expected_status == "processing")
                            cur.execute(
                                "UPDATE job_counters SET "
                                "ready_count=GREATEST(ready_count - %s, 0), "
                                "scheduled_count=GREATEST(scheduled_count - %s, 0), "
                                "processing_count=GREATEST(processing_count - %s, 0), "
                                "updated_at=NOW() WHERE domain=%s AND queue=%s AND job_type=%s",
                                (
                                    ready_delta,
                                    scheduled_delta,
                                    processing_delta,
                                    domain,
                                    queue,
                                    job_type,
                                ),
                            )
                        cur.execute(
                            "INSERT INTO job_events("
                            "job_id, domain, queue, job_type, event_type, attrs_json, "
                            "owner_user_id, request_id, trace_id, created_at"
                            ") VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())",
                            (
                                int(applied_job["id"]),
                                domain,
                                queue,
                                job_type,
                                event_type,
                                json.dumps(event_attrs),
                                owner_user_id,
                                applied_job.get("request_id"),
                                applied_job.get("trace_id"),
                            ),
                        )
            else:
                now_utc = self._clock.now_utc().astimezone(_tz.utc)
                now_sql = now_utc.strftime("%Y-%m-%d %H:%M:%S")
                lease_clause = (
                    " AND (leased_until IS NULL OR DATETIME(leased_until) <= DATETIME(?))"
                    if expected_status == "processing" and require_processing_lease_expired
                    else ""
                )
                with conn:
                    conn.execute("BEGIN IMMEDIATE")
                    if not self._slides_generation_ready_in_connection(conn):
                        raise SlidesGenerationJobsUnavailableError(
                            "presentation.generate Jobs coordination is unavailable"
                        )
                    authority_rows = list(
                        conn.execute(
                            "SELECT * FROM jobs WHERE uuid=? ORDER BY id LIMIT 2",
                            (job_uuid,),
                        ).fetchall()
                    )
                    if len(authority_rows) > 1:
                        conn.execute(
                            """
                            UPDATE slides_standalone_reconciliation
                            SET diagnostic_code=CASE
                                  WHEN diagnostic_code='duplicate_archive_uuid'
                                  THEN diagnostic_code
                                  ELSE 'ambiguous_generation_legacy_row' END,
                                diagnostic_count=CASE
                                  WHEN diagnostic_code='duplicate_archive_uuid'
                                  THEN diagnostic_count
                                  ELSE MAX(diagnostic_count, ?) END,
                                diagnostic_at=CASE
                                  WHEN diagnostic_code='duplicate_archive_uuid'
                                  THEN diagnostic_at ELSE DATETIME('now') END
                            WHERE singleton_id=1
                            """,
                            (len(authority_rows),),
                        )
                        unsafe_correlation = True
                    row = authority_rows[0] if authority_rows else None
                    authority_id = (
                        int(row["id"])
                        if row is not None and not unsafe_correlation
                        else None
                    )
                    params = [
                        status,
                        error_code,
                        error_message,
                        completion_token,
                        now_sql,
                        status,
                        now_sql,
                        status,
                        error_message,
                        job_uuid,
                        owner_user_id,
                        domain,
                        queue,
                        job_type,
                        expected_status,
                        (
                            job_id
                            if job_id is not None and not unsafe_correlation
                            else authority_id
                        ),
                    ]
                    if lease_clause:
                        params.append(now_sql)
                    params.append(completion_token)
                    updated = conn.execute(
                        (
                            "UPDATE jobs SET status=?, error_code=?, error_message=?, "
                            "completion_token=?, completed_at=?, leased_until=NULL, "
                            "cancelled_at=CASE WHEN ?='cancelled' THEN ? ELSE cancelled_at END, "
                            "cancellation_reason=CASE WHEN ?='cancelled' THEN ? ELSE cancellation_reason END "
                            "WHERE uuid=? AND owner_user_id IS ? "
                            "AND domain=? AND queue=? AND job_type=? AND status=? "
                            f"AND id=?{lease_clause} "  # nosec B608
                            "AND (completion_token IS NULL OR completion_token=?)"
                        ),
                        tuple(params),
                    )
                    if updated.rowcount == 1:
                        applied = conn.execute(
                            "SELECT * FROM jobs WHERE uuid=?",
                            (job_uuid,),
                        ).fetchone()
                        if applied is None:
                            raise RuntimeError("terminalized job disappeared before bookkeeping")
                        applied_job = dict(applied)
                        if JobManager._is_truthy(os.getenv("JOBS_COUNTERS_ENABLED", "")):
                            available_at = _parse_dt(applied_job.get("available_at"))
                            if available_at is not None and available_at.tzinfo is None:
                                available_at = available_at.replace(tzinfo=_tz.utc)
                            scheduled = available_at is not None and available_at > now_utc
                            ready_delta = int(expected_status == "queued" and not scheduled)
                            scheduled_delta = int(expected_status == "queued" and scheduled)
                            processing_delta = int(expected_status == "processing")
                            conn.execute(
                                "UPDATE job_counters SET "
                                "ready_count=MAX(ready_count - ?, 0), "
                                "scheduled_count=MAX(scheduled_count - ?, 0), "
                                "processing_count=MAX(processing_count - ?, 0), "
                                "updated_at=DATETIME('now') "
                                "WHERE domain=? AND queue=? AND job_type=?",
                                (
                                    ready_delta,
                                    scheduled_delta,
                                    processing_delta,
                                    domain,
                                    queue,
                                    job_type,
                                ),
                            )
                        conn.execute(
                            "INSERT INTO job_events("
                            "job_id, domain, queue, job_type, event_type, attrs_json, "
                            "owner_user_id, request_id, trace_id, created_at"
                            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, DATETIME('now'))",
                            (
                                int(applied_job["id"]),
                                domain,
                                queue,
                                job_type,
                                event_type,
                                json.dumps(event_attrs),
                                owner_user_id,
                                applied_job.get("request_id"),
                                applied_job.get("trace_id"),
                            ),
                        )
            if unsafe_correlation:
                raise SlidesGenerationJobsUnavailableError(
                    "presentation.generate correlation is unsafe"
                )
            if applied_job is not None:
                with contextlib.suppress(_JOB_NONCRITICAL_EXCEPTIONS):
                    self._update_gauges(domain=domain, queue=queue, job_type=job_type)
                if status == "failed":
                    with contextlib.suppress(_JOB_NONCRITICAL_EXCEPTIONS):
                        increment_failures(applied_job, reason="terminal")
                    with contextlib.suppress(_JOB_NONCRITICAL_EXCEPTIONS):
                        from .metrics import increment_failures_by_code

                        increment_failures_by_code(applied_job, error_code)
                else:
                    with contextlib.suppress(_JOB_NONCRITICAL_EXCEPTIONS):
                        increment_cancelled(applied_job)
                if not JobManager._is_truthy(os.getenv("JOBS_EVENTS_OUTBOX", "")):
                    with contextlib.suppress(_JOB_NONCRITICAL_EXCEPTIONS):
                        emit_job_event(event_type, job=applied_job, attrs=event_attrs)
                with contextlib.suppress(_JOB_NONCRITICAL_EXCEPTIONS):
                    submit_job_audit_event(event_type, job=applied_job, attrs=event_attrs)
                return "APPLIED"
            if row is None:
                return "MISSING"
            stored = dict(row)
            exact_correlation = (
                stored.get("uuid") == job_uuid
                and stored.get("owner_user_id") == owner_user_id
                and _is_slides_generation_scope(
                    stored.get("domain"),
                    stored.get("queue"),
                    stored.get("job_type"),
                )
                and (job_id is None or stored.get("id") == job_id)
            )
            identical_terminal = (
                stored.get("status") == status
                and stored.get("completion_token") == completion_token
                and stored.get("error_code") == error_code
                and stored.get("error_message") == error_message
                and stored.get("completed_at") is not None
                and stored.get("leased_until") is None
                and (
                    status != "cancelled"
                    or (
                        stored.get("cancelled_at") is not None
                        and stored.get("cancellation_reason") == error_message
                    )
                )
            )
            if exact_correlation and identical_terminal:
                return "IDEMPOTENT"
            return "CONFLICT"
        finally:
            conn.close()

    def complete_job(
        self,
        job_id: int,
        *,
        result: dict[str, Any] | None = None,
        worker_id: str | None = None,
        lease_id: str | None = None,
        completion_token: str | None = None,
        enforce: bool | None = None,
    ) -> bool:
        """Mark a job as completed and clear the lease.

        See `renew_job_lease` for enforcement semantics.
        """
        # Strong exactly-once finalize (optional): require a completion_token when enabled
        if (
            JobManager._is_truthy(os.getenv("JOBS_REQUIRE_COMPLETION_TOKEN", ""))
            and not completion_token
        ):
            raise ValueError("completion_token required by JOBS_REQUIRE_COMPLETION_TOKEN")  # noqa: TRY003
        if enforce is None:
            enforce = self._should_enforce_ack()
        # Cap result size if configured
        max_bytes = int(os.getenv("JOBS_MAX_JSON_BYTES", "1048576") or "1048576")
        truncate = JobManager._is_truthy(os.getenv("JOBS_JSON_TRUNCATE", ""))
        outbox_enabled = JobManager._is_truthy(os.getenv("JOBS_EVENTS_OUTBOX", ""))
        res_obj = result
        res_ok = False
        post_commit_side_effects: list[
            tuple[Any, tuple[Any, ...], dict[str, Any]]
        ] = []
        if res_obj is not None:
            # Serialize first; only catch serialization errors, not size checks
            try:
                res_json = json.dumps(res_obj)
            except (TypeError, ValueError):
                # Non-serializable results are handled by DB layer (stored as NULL or fail later)
                res_json = None
            if res_json is not None:
                res_bytes = len(res_json.encode("utf-8"))
                if res_bytes > max_bytes:
                    if truncate:
                        res_obj = {"_truncated": True, "len_bytes": res_bytes}
                    else:
                        raise ValueError(f"Result too large: {res_bytes} bytes > limit {max_bytes}")  # noqa: TRY003
        # Optional encryption at rest for result (requires domain; will be resolved per-backend)
        conn = self._connect()
        _test_mode = _is_test_mode()
        try:
            if self.backend == "postgres":
                with conn:  # noqa: SIM117
                    with self._pg_cursor(conn) as cur:
                        if _test_mode:
                            with contextlib.suppress(_JOB_NONCRITICAL_EXCEPTIONS):
                                logger.info(
                                    f"[JM TEST MUT] complete_job enter job_id={job_id} enforce={enforce} backend=pg"
                                )
                        # Pre-fetch for metrics and idempotency
                        cur.execute(
                            "SELECT status, completion_token, worker_id, lease_id, domain, queue, job_type, available_at, started_at, acquired_at, trace_id, request_id, owner_user_id FROM jobs WHERE id = %s",
                            (int(job_id),),
                        )
                        base = cur.fetchone()
                        if base:
                            st = str(base.get("status"))
                            ct = base.get("completion_token")
                            if st in {"completed", "failed", "cancelled", "quarantined"}:
                                # A token only replays the operation that produced its state.
                                return bool(
                                    st == "completed"
                                    and completion_token
                                    and ct
                                    and str(ct) == str(completion_token)
                                )
                        # Apply encryption if configured (domain available from base)
                        try:
                            if base:
                                res_obj = self._maybe_encrypt_json(res_obj, str(base.get("domain")))
                        except _JOB_NONCRITICAL_EXCEPTIONS:
                            pass
                        completed_from_processing = False
                        completed_from_queued = False
                        if enforce:
                            cur.execute(
                                (
                                    "UPDATE jobs SET status = 'completed', result = %s::jsonb, completed_at = NOW(), completion_token = %s, "
                                    "leased_until = NULL, worker_id = NULL, lease_id = NULL WHERE id = %s AND status = 'processing' AND worker_id = %s AND lease_id = %s AND (completion_token IS NULL OR completion_token = %s)"
                                ),
                                (
                                    json.dumps(res_obj) if res_obj is not None else None,
                                    completion_token,
                                    int(job_id),
                                    worker_id,
                                    lease_id,
                                    completion_token,
                                ),
                            )
                            ok = cur.rowcount > 0
                            completed_from_processing = ok
                            if not ok and completion_token:
                                # Idempotent retry if already completed with same token (race)
                                cur.execute("SELECT completion_token, status FROM jobs WHERE id = %s", (int(job_id),))
                                chk = cur.fetchone()
                                if (
                                    chk
                                    and str(chk.get("completion_token") or "") == str(completion_token)
                                    and str(chk.get("status")) == "completed"
                                ):
                                    return True
                        else:
                            cur.execute(
                                "UPDATE jobs SET status = 'completed', result = %s::jsonb, completed_at = NOW(), completion_token = COALESCE(completion_token, %s), leased_until = NULL, worker_id = NULL, lease_id = NULL WHERE id = %s AND status = 'processing' AND (completion_token IS NULL OR completion_token = %s)",
                                (
                                    json.dumps(res_obj) if res_obj is not None else None,
                                    completion_token,
                                    int(job_id),
                                    completion_token,
                                ),
                            )
                            completed_from_processing = cur.rowcount > 0
                            ok = completed_from_processing
                            if not ok:
                                # Admin-style finalize: optionally allow completing queued without lease when enforcement disabled
                                try:
                                    allow = {
                                        d.strip().lower()
                                        for d in os.getenv(
                                            "JOBS_ADMIN_COMPLETE_QUEUED_ALLOW_DOMAINS",
                                            "chatbooks,embeddings",
                                        ).split(",")
                                        if d.strip()
                                    }
                                    cur.execute("SELECT domain FROM jobs WHERE id = %s", (int(job_id),))
                                    row_dom = cur.fetchone()
                                    dom_val = str(row_dom.get("domain") or "").lower() if row_dom else ""
                                except _JOB_NONCRITICAL_EXCEPTIONS:
                                    allow = {"chatbooks", "embeddings"}
                                    dom_val = ""
                                if dom_val in allow:
                                    cur.execute(
                                        "UPDATE jobs SET status = 'completed', result = %s::jsonb, completed_at = NOW(), completion_token = COALESCE(completion_token, %s), leased_until = NULL, worker_id = NULL, lease_id = NULL WHERE id = %s AND status = 'queued' AND (completion_token IS NULL OR completion_token = %s)",
                                        (
                                            json.dumps(res_obj) if res_obj is not None else None,
                                            completion_token,
                                            int(job_id),
                                            completion_token,
                                        ),
                                    )
                                    completed_from_queued = cur.rowcount > 0
                                    ok = completed_from_queued
                        if _test_mode:
                            try:
                                cur.execute("SELECT id, status FROM jobs WHERE id = %s", (int(job_id),))
                                _r = cur.fetchone()
                                cur.execute("SELECT COUNT(*) AS c FROM jobs")
                                _total = (cur.fetchone() or {}).get("c", 0)
                                cur.execute("SELECT status, COUNT(*) AS c FROM jobs GROUP BY status")
                                _rows = cur.fetchall() or []
                                _dist = {str(x.get("status")): int(x.get("c") or 0) for x in _rows}
                                logger.info(
                                    f"[JM TEST MUT] complete_job affected ok={bool(ok)} row={(dict(_r) if _r else None)} total={_total} dist={_dist}"
                                )
                            except _JOB_NONCRITICAL_EXCEPTIONS:
                                pass
                        # Truncation metric (PG)
                        try:
                            if base and ok and isinstance(res_obj, dict) and res_obj.get("_truncated"):
                                dtmp = dict(base)
                                post_commit_side_effects.append(
                                    (
                                        increment_json_truncated,
                                        (
                                            {
                                                "domain": dtmp.get("domain"),
                                                "queue": dtmp.get("queue"),
                                                "job_type": dtmp.get("job_type"),
                                            },
                                            "result",
                                        ),
                                        {},
                                    )
                                )
                        except _JOB_NONCRITICAL_EXCEPTIONS:
                            pass
                        # Metrics: duration + counters
                        try:
                            if base and ok:
                                d = dict(base)
                                started_at = d.get("started_at") or d.get("acquired_at")
                                if isinstance(started_at, str):
                                    started_at = _parse_dt(started_at)
                                post_commit_side_effects.append(
                                    (
                                        observe_duration,
                                        (
                                            {
                                                "domain": d.get("domain"),
                                                "queue": d.get("queue"),
                                                "job_type": d.get("job_type"),
                                                "trace_id": d.get("trace_id"),
                                                "request_id": d.get("request_id"),
                                            },
                                            started_at,
                                            self._clock.now_utc(),
                                        ),
                                        {},
                                    )
                                )
                                post_commit_side_effects.append(
                                    (
                                        increment_completed,
                                        (
                                            {
                                                "domain": d.get("domain"),
                                                "queue": d.get("queue"),
                                                "job_type": d.get("job_type"),
                                            },
                                        ),
                                        {},
                                    )
                                )
                                post_commit_side_effects.append(
                                    (
                                        self._update_gauges,
                                        (),
                                        {
                                            "domain": d.get("domain"),
                                            "queue": d.get("queue"),
                                            "job_type": d.get("job_type"),
                                        },
                                    )
                                )
                                post_commit_side_effects.append(
                                    (_record_job_span, ("job.complete", d), {})
                                )
                        except _JOB_NONCRITICAL_EXCEPTIONS:
                            pass
                        if (
                            base
                            and ok
                            and JobManager._is_truthy(
                                os.getenv("JOBS_COUNTERS_ENABLED", "")
                            )
                        ):
                            d_counter = dict(base)
                            if completed_from_processing:
                                cur.execute(
                                    "UPDATE job_counters SET processing_count = GREATEST(processing_count - 1, 0), updated_at = NOW() WHERE domain=%s AND queue=%s AND job_type=%s",
                                    (
                                        d_counter.get("domain"),
                                        d_counter.get("queue"),
                                        d_counter.get("job_type"),
                                    ),
                                )
                            elif completed_from_queued:
                                is_sched = d_counter.get("available_at") is not None
                                cur.execute(
                                    (
                                        "UPDATE job_counters SET ready_count = GREATEST(ready_count - %s, 0), scheduled_count = GREATEST(scheduled_count - %s, 0), updated_at = NOW() "
                                        "WHERE domain=%s AND queue=%s AND job_type=%s"
                                    ),
                                    (
                                        int(not is_sched),
                                        int(is_sched),
                                        d_counter.get("domain"),
                                        d_counter.get("queue"),
                                        d_counter.get("job_type"),
                                    ),
                                )
                            if cur.rowcount == 0:
                                _reconcile_lifecycle_counter_row(
                                    cur,
                                    backend=self.backend,
                                    domain=d_counter.get("domain"),
                                    queue=d_counter.get("queue"),
                                    job_type=d_counter.get("job_type"),
                                )
                        if base and ok:
                            d_ev = dict(base)
                            self._stage_completion_sla_breach(
                                cur,
                                job_id=int(job_id),
                                job=d_ev,
                                outbox_enabled=outbox_enabled,
                                side_effects=post_commit_side_effects,
                            )
                            ev = {
                                "id": int(job_id),
                                "domain": d_ev.get("domain"),
                                "queue": d_ev.get("queue"),
                                "job_type": d_ev.get("job_type"),
                                "owner_user_id": d_ev.get("owner_user_id"),
                                "request_id": d_ev.get("request_id"),
                                "trace_id": d_ev.get("trace_id"),
                            }
                            if outbox_enabled:
                                _insert_lifecycle_event(
                                    cur,
                                    backend=self.backend,
                                    event_type="job.completed",
                                    job=ev,
                                )
                            _queue_lifecycle_event_observer(
                                post_commit_side_effects,
                                event_type="job.completed",
                                job=ev,
                            )
                            _commit_postgres_transaction(
                                conn,
                                operation="job completion",
                            )
                            _run_post_commit_side_effects(post_commit_side_effects)
                            post_commit_side_effects.clear()
                        res_ok = ok
                        # fall through to finally and return
            else:
                with conn:
                    if _test_mode:
                        with contextlib.suppress(_JOB_NONCRITICAL_EXCEPTIONS):
                            logger.info(
                                f"[JM TEST MUT] complete_job enter job_id={job_id} enforce={enforce} backend=sqlite"
                            )
                    # Pre-fetch for metrics + idempotency
                    rowm = conn.execute(
                        "SELECT status, completion_token, domain, queue, job_type, available_at, started_at, acquired_at, trace_id, request_id, owner_user_id FROM jobs WHERE id = ?",
                        (job_id,),
                    ).fetchone()
                    if rowm:
                        st = str(rowm[0])
                        ct = rowm[1]
                        if st in {"completed", "failed", "cancelled", "quarantined"}:
                            return bool(
                                st == "completed"
                                and completion_token
                                and ct
                                and str(ct) == str(completion_token)
                            )
                    # Apply encryption if configured
                    try:
                        if rowm:
                            res_obj = self._maybe_encrypt_json(res_obj, str(rowm[2]))
                    except _JOB_NONCRITICAL_EXCEPTIONS:
                        pass
                    completed_from_processing = False
                    completed_from_queued = False
                    if enforce:
                        conn.execute(
                            (
                                "UPDATE jobs SET status = 'completed', result = ?, completed_at = DATETIME('now'), completion_token = ?, "
                                "leased_until = NULL, worker_id = NULL, lease_id = NULL WHERE id = ? AND status = 'processing' AND worker_id = ? AND lease_id = ? AND (completion_token IS NULL OR completion_token = ?)"
                            ),
                            (
                                json.dumps(res_obj) if res_obj is not None else None,
                                completion_token,
                                job_id,
                                worker_id,
                                lease_id,
                                completion_token,
                            ),
                        )
                        cur = conn.execute("SELECT changes()")
                        ok = (cur.fetchone()[0] or 0) > 0
                        completed_from_processing = ok
                        if not ok and completion_token:
                            chk = conn.execute(
                                "SELECT completion_token, status FROM jobs WHERE id = ?", (job_id,)
                            ).fetchone()
                            if chk and str(chk[0] or "") == str(completion_token) and str(chk[1]) == "completed":
                                return True
                    else:
                        conn.execute(
                            (
                                "UPDATE jobs SET status = 'completed', result = ?, completed_at = DATETIME('now'), completion_token = COALESCE(completion_token, ?), leased_until = NULL, worker_id = NULL, lease_id = NULL "
                                "WHERE id = ? AND status = 'processing' AND (completion_token IS NULL OR completion_token = ?)"
                            ),
                            (
                                json.dumps(res_obj) if res_obj is not None else None,
                                completion_token,
                                job_id,
                                completion_token,
                            ),
                        )
                        cur = conn.execute("SELECT changes()")
                        completed_from_processing = (cur.fetchone()[0] or 0) > 0
                        ok = completed_from_processing
                        if not ok:
                            # Admin-style finalize: optionally allow completing queued without lease when enforcement is disabled
                            try:
                                allow = {
                                    d.strip().lower()
                                    for d in (
                                        os.getenv(
                                            "JOBS_ADMIN_COMPLETE_QUEUED_ALLOW_DOMAINS", "chatbooks,embeddings"
                                        ).split(",")
                                    )
                                    if d.strip()
                                }
                                row_dom = conn.execute("SELECT domain FROM jobs WHERE id = ?", (job_id,)).fetchone()
                                dom_val = str(row_dom[0]).lower() if row_dom and row_dom[0] else ""
                            except _JOB_NONCRITICAL_EXCEPTIONS:
                                allow = {"chatbooks", "embeddings"}
                                dom_val = ""
                            if dom_val in allow:
                                conn.execute(
                                    (
                                        "UPDATE jobs SET status = 'completed', result = ?, completed_at = DATETIME('now'), completion_token = COALESCE(completion_token, ?), leased_until = NULL, worker_id = NULL, lease_id = NULL "
                                        "WHERE id = ? AND status = 'queued' AND (completion_token IS NULL OR completion_token = ?)"
                                    ),
                                    (
                                        json.dumps(res_obj) if res_obj is not None else None,
                                        completion_token,
                                        job_id,
                                        completion_token,
                                    ),
                                )
                                cur2 = conn.execute("SELECT changes()")
                                completed_from_queued = (cur2.fetchone()[0] or 0) > 0
                                ok = completed_from_queued
                    if _test_mode:
                        try:
                            _r = conn.execute("SELECT id, status FROM jobs WHERE id = ?", (int(job_id),)).fetchone()
                            _total = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
                            _dist = {
                                str(r[0]): int(r[1])
                                for r in conn.execute("SELECT status, COUNT(*) FROM jobs GROUP BY status").fetchall()
                            }
                            logger.info(
                                f"[JM TEST MUT] complete_job affected ok={bool(ok)} row={(dict(_r) if _r else None)} total={int(_total)} dist={_dist}"
                            )
                        except _JOB_NONCRITICAL_EXCEPTIONS:
                            pass
                    # Truncation metric (SQLite)
                    try:
                        if rowm and ok and isinstance(res_obj, dict) and res_obj.get("_truncated"):
                            post_commit_side_effects.append(
                                (
                                    increment_json_truncated,
                                    (
                                        {
                                            "domain": rowm[2],
                                            "queue": rowm[3],
                                            "job_type": rowm[4],
                                        },
                                        "result",
                                    ),
                                    {},
                                )
                            )
                    except _JOB_NONCRITICAL_EXCEPTIONS:
                        pass
                    # Metrics: duration + counters
                    try:
                        if rowm and ok:
                            d = {
                                "domain": rowm[2],
                                "queue": rowm[3],
                                "job_type": rowm[4],
                                "available_at": rowm[5],
                                "started_at": rowm[6],
                                "acquired_at": rowm[7],
                                "trace_id": rowm[8] if len(rowm) > 8 else None,
                                "request_id": rowm[9] if len(rowm) > 9 else None,
                                "owner_user_id": rowm[10] if len(rowm) > 10 else None,
                            }
                            s = _parse_dt(d.get("started_at")) or _parse_dt(d.get("acquired_at"))
                            post_commit_side_effects.append(
                                (
                                    observe_duration,
                                    (
                                        {
                                            "domain": d.get("domain"),
                                            "queue": d.get("queue"),
                                            "job_type": d.get("job_type"),
                                            "trace_id": d.get("trace_id"),
                                            "request_id": d.get("request_id"),
                                        },
                                        s,
                                        datetime.utcnow(),
                                    ),
                                    {},
                                )
                            )
                            post_commit_side_effects.append(
                                (
                                    increment_completed,
                                    (
                                        {
                                            "domain": d.get("domain"),
                                            "queue": d.get("queue"),
                                            "job_type": d.get("job_type"),
                                        },
                                    ),
                                    {},
                                )
                            )
                            post_commit_side_effects.append(
                                (
                                    self._update_gauges,
                                    (),
                                    {
                                        "domain": d.get("domain"),
                                        "queue": d.get("queue"),
                                        "job_type": d.get("job_type"),
                                    },
                                )
                            )
                            post_commit_side_effects.append(
                                (_record_job_span, ("job.complete", d), {})
                            )
                    except _JOB_NONCRITICAL_EXCEPTIONS:
                        pass
                    if (
                        rowm
                        and ok
                        and JobManager._is_truthy(
                            os.getenv("JOBS_COUNTERS_ENABLED", "")
                        )
                    ):
                        d_counter = {
                            "domain": rowm[2],
                            "queue": rowm[3],
                            "job_type": rowm[4],
                            "available_at": rowm[5],
                        }
                        if completed_from_processing:
                            counter_cursor = conn.execute(
                                "UPDATE job_counters SET processing_count = CASE WHEN processing_count>0 THEN processing_count-1 ELSE 0 END, updated_at = DATETIME('now') WHERE domain=? AND queue=? AND job_type=?",
                                (
                                    d_counter["domain"],
                                    d_counter["queue"],
                                    d_counter["job_type"],
                                ),
                            )
                        else:
                            is_sched = d_counter["available_at"] is not None
                            counter_cursor = conn.execute(
                                (
                                    "UPDATE job_counters SET ready_count = CASE WHEN ready_count > ? THEN ready_count - ? ELSE 0 END, "
                                    "scheduled_count = CASE WHEN scheduled_count > ? THEN scheduled_count - ? ELSE 0 END, updated_at = DATETIME('now') "
                                    "WHERE domain=? AND queue=? AND job_type=?"
                                ),
                                (
                                    int(not is_sched),
                                    int(not is_sched),
                                    int(is_sched),
                                    int(is_sched),
                                    d_counter["domain"],
                                    d_counter["queue"],
                                    d_counter["job_type"],
                                ),
                            )
                        if (counter_cursor.rowcount or 0) == 0:
                            _reconcile_lifecycle_counter_row(
                                conn,
                                backend=self.backend,
                                domain=d_counter["domain"],
                                queue=d_counter["queue"],
                                job_type=d_counter["job_type"],
                            )
                    if rowm and ok:
                        d_sla = {
                            "domain": rowm[2],
                            "queue": rowm[3],
                            "job_type": rowm[4],
                            "started_at": rowm[6],
                            "acquired_at": rowm[7],
                            "trace_id": rowm[8] if len(rowm) > 8 else None,
                            "request_id": rowm[9] if len(rowm) > 9 else None,
                            "owner_user_id": rowm[10] if len(rowm) > 10 else None,
                        }
                        self._stage_completion_sla_breach(
                            conn,
                            job_id=int(job_id),
                            job=d_sla,
                            outbox_enabled=outbox_enabled,
                            side_effects=post_commit_side_effects,
                        )
                        ev = {
                            "id": int(job_id),
                            "domain": rowm[2],
                            "queue": rowm[3],
                            "job_type": rowm[4],
                            "owner_user_id": rowm[10] if len(rowm) > 10 else None,
                            "request_id": rowm[9] if len(rowm) > 9 else None,
                            "trace_id": rowm[8] if len(rowm) > 8 else None,
                        }
                        if outbox_enabled:
                            _insert_lifecycle_event(
                                conn,
                                backend=self.backend,
                                event_type="job.completed",
                                job=ev,
                            )
                        _queue_lifecycle_event_observer(
                            post_commit_side_effects,
                            event_type="job.completed",
                            job=ev,
                        )
                    if ok:
                        conn.commit()
                        _run_post_commit_side_effects(post_commit_side_effects)
                        post_commit_side_effects.clear()
                    res_ok = ok
        finally:
            _close_connection_nonfatal(conn, operation="job completion")
        return bool(res_ok)

    def _adaptive_lease_seconds(self, domain: str, queue: str, job_type: str | None) -> int:
        """Compute adaptive lease seconds based on recent P95 durations with headroom.

        Works for both backends; uses percentile_cont on PG and a simple
        approximate percentile for SQLite.
        """
        headroom = float(os.getenv("JOBS_ADAPTIVE_LEASE_HEADROOM", "1.3") or "1.3")
        window_h = int(os.getenv("JOBS_ADAPTIVE_LEASE_WINDOW_HOURS", "6") or "6")
        min_s = int(os.getenv("JOBS_ADAPTIVE_LEASE_MIN_SECONDS", "15") or "15")
        max_s = int(os.getenv("JOBS_LEASE_MAX_SECONDS", "3600") or "3600")
        value: float | None = None
        conn = self._connect()
        try:
            if self.backend == "postgres":
                with self._pg_cursor(conn) as cur:
                    q = (
                        "SELECT percentile_cont(0.95) WITHIN GROUP (ORDER BY EXTRACT(EPOCH FROM (completed_at - COALESCE(started_at, acquired_at)))) AS p95 "
                        "FROM jobs WHERE completed_at IS NOT NULL AND created_at >= NOW() - (%s || ' hours')::interval AND domain=%s AND queue=%s"
                    )
                    params: list[Any] = [int(window_h), domain, queue]
                    if job_type:
                        q += " AND job_type=%s"
                        params.append(job_type)
                    cur.execute(q, tuple(params))
                    row = cur.fetchone()
                    if row and (row.get("p95") is not None):
                        value = float(row.get("p95"))
            else:
                query = (
                    "SELECT (julianday(completed_at) - julianday(COALESCE(started_at, acquired_at))) * 86400.0 AS dur "
                    "FROM jobs WHERE completed_at IS NOT NULL AND created_at >= DATETIME('now', ?) AND domain=? AND queue=?"
                )
                params2: list[Any] = [f"-{int(window_h)} hours", domain, queue]
                if job_type:
                    query += " AND job_type=?"
                    params2.append(job_type)
                vals = [float(r[0]) for r in conn.execute(query, tuple(params2)).fetchall() if r and r[0] is not None]
                if vals:
                    vals.sort()
                    idx = max(0, min(len(vals) - 1, int(round(0.95 * (len(vals) - 1)))))
                    value = float(vals[idx])
        finally:
            with contextlib.suppress(_JOB_NONCRITICAL_EXCEPTIONS):
                conn.close()
        if not value or value <= 0:
            return max(min_s, 30)
        return max(min_s, min(max_s, int(value * headroom)))

    def batch_renew_leases(self, items: list[dict[str, Any]], *, enforce: bool | None = None) -> int:
        if enforce is None:
            enforce = self._should_enforce_ack()
        conn = self._connect()
        try:
            command = BatchRenewLeasesCommand(
                items=tuple(
                    BatchRenewLeaseItem(
                        seconds=max(
                            1,
                            min(
                                int(os.getenv("JOBS_LEASE_MAX_SECONDS", "3600") or "3600"),
                                int(item.get("seconds") or 0),
                            ),
                        ),
                        job_id=int(item.get("job_id")),
                        worker_id=item.get("worker_id"),
                        lease_id=item.get("lease_id"),
                    )
                    for item in items
                ),
                enforce=bool(enforce),
            )
            if self.backend == "postgres":
                result = _postgres_renew_leases_batch(
                    conn,
                    self._pg_cursor,
                    command=command,
                    clock=self._clock.now_utc,
                )
            else:
                result = _sqlite_renew_leases_batch(
                    conn,
                    command=command,
                    clock=self._clock.now_utc,
                )
            return int(result.applied_count)
        finally:
            _close_connection_nonfatal(conn, operation="batch lease renewal")

    def batch_complete_jobs(self, items: list[dict[str, Any]], *, enforce: bool | None = None) -> int:
        if enforce is None:
            enforce = self._should_enforce_ack()
        conn = self._connect()
        done = 0
        try:
            if self.backend == "postgres":
                with conn:  # noqa: SIM117
                    with self._pg_cursor(conn) as cur:
                        for it in items:
                            res_obj = it.get("result")
                            # Optional encryption at rest: prefer provided domain, otherwise fetch
                            try:
                                dom = it.get("domain")
                                if not dom:
                                    cur.execute("SELECT domain FROM jobs WHERE id=%s", (int(it.get("job_id")),))
                                    _r = cur.fetchone()
                                    dom = (_r.get("domain") if isinstance(_r, dict) else None) if _r else None
                                res_obj = self._maybe_encrypt_json(res_obj, dom)
                            except _JOB_NONCRITICAL_EXCEPTIONS:
                                pass
                            ctok = it.get("completion_token")
                            if enforce:
                                cur.execute(
                                    "UPDATE jobs SET status='completed', result=%s::jsonb, completed_at = NOW(), completion_token = %s, leased_until = NULL, worker_id = NULL, lease_id = NULL WHERE id=%s AND status='processing' AND worker_id=%s AND lease_id=%s AND (completion_token IS NULL OR completion_token = %s)",
                                    (
                                        json.dumps(res_obj) if res_obj is not None else None,
                                        ctok,
                                        int(it.get("job_id")),
                                        it.get("worker_id"),
                                        it.get("lease_id"),
                                        ctok,
                                    ),
                                )
                            else:
                                cur.execute(
                                    "UPDATE jobs SET status='completed', result=%s::jsonb, completed_at = NOW(), completion_token = COALESCE(completion_token, %s), leased_until = NULL, worker_id = NULL, lease_id = NULL WHERE id=%s AND status='processing' AND (completion_token IS NULL OR completion_token = %s)",
                                    (
                                        json.dumps(res_obj) if res_obj is not None else None,
                                        ctok,
                                        int(it.get("job_id")),
                                        ctok,
                                    ),
                                )
                            done += cur.rowcount or 0
            else:
                with conn:
                    for it in items:
                        res_obj = it.get("result")
                        # Optional encryption at rest (SQLite): prefer provided domain, otherwise fetch
                        try:
                            dom = it.get("domain")
                            if not dom:
                                rowd = conn.execute(
                                    "SELECT domain FROM jobs WHERE id = ?", (int(it.get("job_id")),)
                                ).fetchone()
                                dom = rowd[0] if rowd else None
                            res_obj = self._maybe_encrypt_json(res_obj, dom)
                        except _JOB_NONCRITICAL_EXCEPTIONS:
                            pass
                        ctok = it.get("completion_token")
                        if enforce:
                            cur = conn.execute(
                                "UPDATE jobs SET status='completed', result=?, completed_at = DATETIME('now'), completion_token = ?, leased_until = NULL, worker_id = NULL, lease_id = NULL WHERE id = ? AND status='processing' AND worker_id = ? AND lease_id = ? AND (completion_token IS NULL OR completion_token = ?)",
                                (
                                    json.dumps(res_obj) if res_obj is not None else None,
                                    ctok,
                                    int(it.get("job_id")),
                                    it.get("worker_id"),
                                    it.get("lease_id"),
                                    ctok,
                                ),
                            )
                            done += int(cur.rowcount or 0)
                        else:
                            cur = conn.execute(
                                "UPDATE jobs SET status='completed', result=?, completed_at = DATETIME('now'), completion_token = COALESCE(completion_token, ?), leased_until = NULL, worker_id = NULL, lease_id = NULL WHERE id = ? AND status='processing' AND (completion_token IS NULL OR completion_token = ?)",
                                (
                                    json.dumps(res_obj) if res_obj is not None else None,
                                    ctok,
                                    int(it.get("job_id")),
                                    ctok,
                                ),
                            )
                            done += int(cur.rowcount or 0)
            return int(done)
        finally:
            with contextlib.suppress(_JOB_NONCRITICAL_EXCEPTIONS):
                conn.close()

    def batch_fail_jobs(self, items: list[dict[str, Any]], *, enforce: bool | None = None) -> int:
        if JobManager._is_truthy(os.getenv("JOBS_REQUIRE_COMPLETION_TOKEN", "")):
            for it in items:
                if not it.get("completion_token"):
                    raise ValueError("completion_token required by JOBS_REQUIRE_COMPLETION_TOKEN")  # noqa: TRY003
        if enforce is None:
            enforce = self._should_enforce_ack()
        conn = self._connect()
        cnt = 0
        try:
            if self.backend == "postgres":
                with conn:  # noqa: SIM117
                    with self._pg_cursor(conn) as cur:
                        for it in items:
                            if enforce:
                                cur.execute(
                                    "UPDATE jobs SET status='failed', last_error=%s, error_message=%s, error_code=%s, completed_at=NOW(), leased_until=NULL, worker_id=NULL, lease_id=NULL, completion_token=%s WHERE id=%s AND status='processing' AND worker_id=%s AND lease_id=%s AND (completion_token IS NULL OR completion_token=%s)",
                                    (
                                        it.get("error_code") or it.get("error"),
                                        it.get("error"),
                                        it.get("error_code"),
                                        it.get("completion_token"),
                                        int(it.get("job_id")),
                                        it.get("worker_id"),
                                        it.get("lease_id"),
                                        it.get("completion_token"),
                                    ),
                                )
                            else:
                                cur.execute(
                                    "UPDATE jobs SET status='failed', last_error=%s, error_message=%s, error_code=%s, completed_at=NOW(), leased_until=NULL, worker_id=NULL, lease_id=NULL, completion_token=COALESCE(completion_token,%s) WHERE id=%s AND status='processing' AND (completion_token IS NULL OR completion_token=%s)",
                                    (
                                        it.get("error_code") or it.get("error"),
                                        it.get("error"),
                                        it.get("error_code"),
                                        it.get("completion_token"),
                                        int(it.get("job_id")),
                                        it.get("completion_token"),
                                    ),
                                )
                            cnt += cur.rowcount or 0
            else:
                with conn:
                    for it in items:
                        if enforce:
                            cur = conn.execute(
                                "UPDATE jobs SET status='failed', last_error=?, error_message=?, error_code=?, completed_at=DATETIME('now'), leased_until=NULL, worker_id=NULL, lease_id=NULL, completion_token=? WHERE id=? AND status='processing' AND worker_id=? AND lease_id=? AND (completion_token IS NULL OR completion_token=?)",
                                (
                                    it.get("error_code") or it.get("error"),
                                    it.get("error"),
                                    it.get("error_code"),
                                    it.get("completion_token"),
                                    int(it.get("job_id")),
                                    it.get("worker_id"),
                                    it.get("lease_id"),
                                    it.get("completion_token"),
                                ),
                            )
                            cnt += int(cur.rowcount or 0)
                        else:
                            cur = conn.execute(
                                "UPDATE jobs SET status='failed', last_error=?, error_message=?, error_code=?, completed_at=DATETIME('now'), leased_until=NULL, worker_id=NULL, lease_id=NULL, completion_token=COALESCE(completion_token,?) WHERE id=? AND status='processing' AND (completion_token IS NULL OR completion_token=?)",
                                (
                                    it.get("error_code") or it.get("error"),
                                    it.get("error"),
                                    it.get("error_code"),
                                    it.get("completion_token"),
                                    int(it.get("job_id")),
                                    it.get("completion_token"),
                                ),
                            )
                            cnt += int(cur.rowcount or 0)
            return int(cnt)
        finally:
            with contextlib.suppress(_JOB_NONCRITICAL_EXCEPTIONS):
                conn.close()

    def fail_job(
        self,
        job_id: int,
        *,
        error: str,
        retryable: bool = True,
        backoff_seconds: int = 1,
        worker_id: str | None = None,
        lease_id: str | None = None,
        enforce: bool | None = None,
        error_code: str | None = None,
        error_class: str | None = None,
        error_stack: dict[str, Any] | None = None,
        completion_token: str | None = None,
    ) -> bool:
        """Mark a job as failed; optionally reschedule with backoff if retryable.

        See `renew_job_lease` for enforcement semantics.
        """
        # Strong exactly-once finalize (optional): require a completion_token when enabled
        if (
            JobManager._is_truthy(os.getenv("JOBS_REQUIRE_COMPLETION_TOKEN", ""))
            and not completion_token
        ):
            raise ValueError("completion_token required by JOBS_REQUIRE_COMPLETION_TOKEN")  # noqa: TRY003
        if enforce is None:
            enforce = self._should_enforce_ack()
        streak_code = str(error_code or error)
        outbox_enabled = JobManager._is_truthy(os.getenv("JOBS_EVENTS_OUTBOX", ""))
        post_commit_side_effects: list[
            tuple[Any, tuple[Any, ...], dict[str, Any]]
        ] = []
        post_commit_cancel_uuid: str | None = None
        post_commit_cancel_reason: str | None = None
        conn = self._connect()
        _test_mode = _is_test_mode()
        try:
            if self.backend == "postgres":
                with conn:  # noqa: SIM117
                    with self._pg_cursor(conn) as cur:
                        # For metrics and idempotency
                        if _test_mode:
                            with contextlib.suppress(_JOB_NONCRITICAL_EXCEPTIONS):
                                logger.info(
                                    f"[JM TEST MUT] fail_job enter job_id={job_id} retryable={retryable} backoff={backoff_seconds} enforce={enforce} backend=pg"
                                )
                        cur.execute(
                            "SELECT status, completion_token, retry_count, failure_streak_code, failure_streak_count, domain, queue, job_type, uuid, request_id, trace_id, owner_user_id, available_at FROM jobs WHERE id = %s",
                            (int(job_id),),
                        )
                        elem = cur.fetchone()
                        if elem:
                            st = str(elem.get("status"))
                            ct = elem.get("completion_token")
                            if st in {"completed", "failed", "cancelled", "quarantined"}:
                                replay_states = (
                                    {"failed", "quarantined"}
                                    if retryable
                                    else {"failed"}
                                )
                                return bool(
                                    st in replay_states
                                    and completion_token
                                    and ct
                                    and str(ct) == str(completion_token)
                                )
                        if retryable:
                            cur.execute("SELECT retry_count FROM jobs WHERE id = %s", (int(job_id),))
                            row = cur.fetchone()
                            current = int(row["retry_count"]) if row else 0
                            exp_backoff = max(1, int(backoff_seconds * (2**current)))
                            test_mode = _is_test_mode()
                            jitter = (
                                0
                                if exp_backoff <= 2 or test_mode
                                else secrets.randbelow(max(1, exp_backoff // 4) + 1)
                            )
                            delay = exp_backoff + jitter
                            # In tests, enforce a generous minimum when the caller requested
                            # immediate retry (backoff_seconds=0) so that newer jobs can be
                            # acquired before recently failed ones.
                            if test_mode:
                                _outbox = JobManager._is_truthy(os.getenv("JOBS_EVENTS_OUTBOX", ""))
                                # Permit immediate retry in tests when caller requests backoff=0
                                # unless outbox mode is enabled (which needs a larger gap).
                                if not _outbox and exp_backoff <= 1:
                                    delay = 0
                                try:
                                    if _outbox and int(backoff_seconds) <= 0 and delay < 10:
                                        delay = 10
                                except _JOB_NONCRITICAL_EXCEPTIONS:
                                    if _outbox and delay < 3:
                                        delay = 3
                            # Poison message quarantine check: increment failure_streak_* and quarantine if threshold reached
                            base_thresh = int(os.getenv("JOBS_QUARANTINE_THRESHOLD", "2") or "2")
                            # In TEST_MODE with zero backoff (unit-style retry loops), avoid quarantining to allow timeline growth
                            if test_mode and int(backoff_seconds) <= 0:
                                thresh = max(base_thresh, 10**9)
                            else:
                                thresh = base_thresh
                            # Update retry path with failure streak bookkeeping
                            if enforce:
                                cur.execute(
                                    (
                                        "UPDATE jobs SET status = CASE WHEN (CASE WHEN COALESCE(failure_streak_code, '') = %s THEN COALESCE(failure_streak_count,0) + 1 ELSE 1 END) >= %s THEN 'quarantined' ELSE 'queued' END, "
                                        "retry_count = retry_count + 1, last_error = %s, error_message = %s, error_code = %s, error_class = %s, error_stack = %s::jsonb, "
                                        "failure_streak_count = CASE WHEN COALESCE(failure_streak_code, '') = %s THEN COALESCE(failure_streak_count,0) + 1 ELSE 1 END, "
                                        "failure_streak_code = %s, "
                                        "completion_token = CASE WHEN (CASE WHEN COALESCE(failure_streak_code, '') = %s THEN COALESCE(failure_streak_count,0) + 1 ELSE 1 END) >= %s THEN %s ELSE NULL END, "
                                        "available_at = CASE WHEN (CASE WHEN COALESCE(failure_streak_code, '') = %s THEN COALESCE(failure_streak_count,0) + 1 ELSE 1 END) >= %s THEN available_at WHEN %s <= 0 THEN NULL ELSE NOW() + (%s || ' seconds')::interval END, "
                                        "quarantined_at = CASE WHEN (CASE WHEN COALESCE(failure_streak_code, '') = %s THEN COALESCE(failure_streak_count,0) + 1 ELSE 1 END) >= %s THEN NOW() ELSE quarantined_at END, "
                                        "leased_until = NULL, worker_id = NULL, lease_id = NULL "
                                        "WHERE id = %s AND status = 'processing' AND retry_count < max_retries AND worker_id = %s AND lease_id = %s AND (completion_token IS NULL OR completion_token = %s)"
                                    ),
                                    (
                                        streak_code,
                                        int(thresh),
                                        streak_code,
                                        error,
                                        error_code,
                                        error_class,
                                        (json.dumps(error_stack) if error_stack is not None else None),
                                        streak_code,
                                        streak_code,
                                        streak_code,
                                        int(thresh),
                                        completion_token,
                                        streak_code,
                                        int(thresh),
                                        int(delay),
                                        int(delay),
                                        streak_code,
                                        int(thresh),
                                        int(job_id),
                                        worker_id,
                                        lease_id,
                                        completion_token,
                                    ),
                                )
                            else:
                                cur.execute(
                                    (
                                        "UPDATE jobs SET status = CASE WHEN (CASE WHEN COALESCE(failure_streak_code, '') = %s THEN COALESCE(failure_streak_count,0) + 1 ELSE 1 END) >= %s THEN 'quarantined' ELSE 'queued' END, "
                                        "retry_count = retry_count + 1, last_error = %s, error_message = %s, error_code = %s, error_class = %s, error_stack = %s::jsonb, "
                                        "failure_streak_count = CASE WHEN COALESCE(failure_streak_code, '') = %s THEN COALESCE(failure_streak_count,0) + 1 ELSE 1 END, "
                                        "failure_streak_code = %s, "
                                        "completion_token = CASE WHEN (CASE WHEN COALESCE(failure_streak_code, '') = %s THEN COALESCE(failure_streak_count,0) + 1 ELSE 1 END) >= %s THEN %s ELSE NULL END, "
                                        "available_at = CASE WHEN (CASE WHEN COALESCE(failure_streak_code, '') = %s THEN COALESCE(failure_streak_count,0) + 1 ELSE 1 END) >= %s THEN available_at WHEN %s <= 0 THEN NULL ELSE NOW() + (%s || ' seconds')::interval END, "
                                        "quarantined_at = CASE WHEN (CASE WHEN COALESCE(failure_streak_code, '') = %s THEN COALESCE(failure_streak_count,0) + 1 ELSE 1 END) >= %s THEN NOW() ELSE quarantined_at END, "
                                        "leased_until = NULL, worker_id = NULL, lease_id = NULL "
                                        "WHERE id = %s AND status = 'processing' AND retry_count < max_retries AND (completion_token IS NULL OR completion_token = %s)"
                                    ),
                                    (
                                        streak_code,
                                        int(thresh),
                                        streak_code,
                                        error,
                                        error_code,
                                        error_class,
                                        (json.dumps(error_stack) if error_stack is not None else None),
                                        streak_code,
                                        streak_code,
                                        streak_code,
                                        int(thresh),
                                        completion_token,
                                        streak_code,
                                        int(thresh),
                                        int(delay),
                                        int(delay),
                                        streak_code,
                                        int(thresh),
                                        int(job_id),
                                        completion_token,
                                    ),
                                )
                            retry_transition_changed = cur.rowcount > 0
                            if retry_transition_changed:
                                cur.execute(
                                    "SELECT status, uuid FROM jobs WHERE id = %s",
                                    (int(job_id),),
                                )
                                _srow = cur.fetchone()
                                transition_status = str((_srow or {}).get("status") or "")
                                if transition_status == "quarantined":
                                    post_commit_cancel_uuid = (_srow or {}).get("uuid")
                                    post_commit_cancel_reason = "dependency_failed"
                                try:
                                    if elem:
                                        if transition_status == "queued":
                                            post_commit_side_effects.append(
                                                (
                                                    increment_retries,
                                                    (dict(elem),),
                                                    {},
                                                )
                                            )
                                            try:
                                                from .metrics import observe_retry_after

                                                post_commit_side_effects.append(
                                                    (
                                                        observe_retry_after,
                                                        (dict(elem), float(delay)),
                                                        {},
                                                    )
                                                )
                                            except _JOB_NONCRITICAL_EXCEPTIONS:
                                                pass
                                        # Append to failure_timeline (retryable)
                                        try:
                                            cur.execute(
                                                "SELECT failure_timeline FROM jobs WHERE id = %s", (int(job_id),)
                                            )
                                            _tlrow = cur.fetchone()
                                            tl_val = _tlrow.get("failure_timeline") if _tlrow else None
                                            if isinstance(tl_val, list):
                                                tl = tl_val
                                            elif isinstance(tl_val, str):
                                                try:
                                                    tl = json.loads(tl_val)
                                                except _JOB_NONCRITICAL_EXCEPTIONS:
                                                    tl = []
                                            else:
                                                tl = []
                                            tl.append(
                                                {
                                                    "ts": datetime.utcnow().isoformat(),
                                                    "error_code": (error_code or error),
                                                    "retry_backoff": int(delay),
                                                }
                                            )
                                            tl = tl[-10:]
                                            cur.execute(
                                                "UPDATE jobs SET failure_timeline = %s::jsonb WHERE id = %s",
                                                (json.dumps(tl), int(job_id)),
                                            )
                                        except _JOB_NONCRITICAL_EXCEPTIONS:
                                            pass
                                except _JOB_NONCRITICAL_EXCEPTIONS:
                                    pass
                                if (
                                    elem
                                    and JobManager._is_truthy(
                                        os.getenv("JOBS_COUNTERS_ENABLED", "")
                                    )
                                ):
                                    will_quarantine = (
                                        transition_status == "quarantined"
                                    )
                                    add_ready = int(
                                        not will_quarantine and int(delay) <= 0
                                    )
                                    add_sched = int(
                                        not will_quarantine and int(delay) > 0
                                    )
                                    add_quar = int(will_quarantine)
                                    cur.execute(
                                        (
                                            "INSERT INTO job_counters(domain,queue,job_type,ready_count,scheduled_count,processing_count,quarantined_count) VALUES(%s,%s,%s,%s,%s,0,%s) "
                                            "ON CONFLICT (domain,queue,job_type) DO UPDATE SET "
                                            "ready_count = job_counters.ready_count + %s, "
                                            "scheduled_count = job_counters.scheduled_count + %s, "
                                            "processing_count = GREATEST(job_counters.processing_count - 1, 0), "
                                            "quarantined_count = job_counters.quarantined_count + %s, "
                                            "updated_at = NOW()"
                                        ),
                                        (
                                            elem.get("domain"),
                                            elem.get("queue"),
                                            elem.get("job_type"),
                                            add_ready,
                                            add_sched,
                                            add_quar,
                                            add_ready,
                                            add_sched,
                                            add_quar,
                                        ),
                                    )
                                if _is_test_mode():
                                    try:
                                        # Snapshot after scheduling retry (PG)
                                        cur.execute("SELECT failure_timeline FROM jobs WHERE id = %s", (int(job_id),))
                                        _tlrow = cur.fetchone()
                                        _tl_len = 0
                                        try:
                                            _tl_len = (
                                                len(json.loads(_tlrow.get("failure_timeline")))
                                                if _tlrow and _tlrow.get("failure_timeline")
                                                else 0
                                            )
                                        except _JOB_NONCRITICAL_EXCEPTIONS:
                                            _tl_len = 0
                                        cur.execute("SELECT COUNT(*) AS c FROM jobs")
                                        _total = (cur.fetchone() or {}).get("c", 0)
                                        cur.execute("SELECT status, COUNT(*) AS c FROM jobs GROUP BY status")
                                        _rows = cur.fetchall() or []
                                        _dist = {str(x.get("status")): int(x.get("c") or 0) for x in _rows}
                                        logger.info(
                                            f"[JM TEST MUT] fail_job retryable scheduled delay={int(delay)} tl_len={_tl_len} total={_total} dist={_dist}"
                                        )
                                    except _JOB_NONCRITICAL_EXCEPTIONS:
                                        pass
                                if elem:
                                    post_commit_side_effects.append(
                                        (
                                            self._update_gauges,
                                            (),
                                            {
                                                "domain": elem.get("domain"),
                                                "queue": elem.get("queue"),
                                                "job_type": elem.get("job_type"),
                                            },
                                        )
                                    )
                                    event_type = (
                                        "job.quarantined"
                                        if transition_status == "quarantined"
                                        else "job.retry_scheduled"
                                    )
                                    attrs = {
                                        "error_code": (error_code or error),
                                        "retry_count": int(current + 1),
                                    }
                                    if transition_status == "queued":
                                        attrs["backoff_seconds"] = int(delay)
                                    event_job = {
                                        "id": int(job_id),
                                        "domain": elem.get("domain"),
                                        "queue": elem.get("queue"),
                                        "job_type": elem.get("job_type"),
                                        "owner_user_id": elem.get("owner_user_id"),
                                        "request_id": elem.get("request_id"),
                                        "trace_id": elem.get("trace_id"),
                                    }
                                    if outbox_enabled:
                                        _insert_lifecycle_event(
                                            cur,
                                            backend=self.backend,
                                            event_type=event_type,
                                            job=event_job,
                                            attrs=attrs,
                                        )
                                    _queue_lifecycle_event_observer(
                                        post_commit_side_effects,
                                        event_type=event_type,
                                        job=event_job,
                                        attrs=attrs,
                                    )
                                _commit_postgres_transaction(
                                    conn,
                                    operation="job retry scheduling",
                                )
                                _run_post_commit_side_effects(post_commit_side_effects)
                                post_commit_side_effects.clear()
                                if post_commit_cancel_uuid:
                                    with contextlib.suppress(_JOB_NONCRITICAL_EXCEPTIONS):
                                        self._cancel_dependent_jobs(
                                            post_commit_cancel_uuid,
                                            reason=post_commit_cancel_reason or "dependency_failed",
                                        )
                                return True
                            if completion_token:
                                cur.execute(
                                    "SELECT status, completion_token FROM jobs WHERE id=%s",
                                    (int(job_id),),
                                )
                                replay = cur.fetchone()
                                if (
                                    replay
                                    and str(replay.get("status"))
                                    in {"failed", "quarantined"}
                                    and str(replay.get("completion_token") or "")
                                    == str(completion_token)
                                ):
                                    return True
                        # terminal failure
                        failed_from_queued = False
                        if enforce:
                            cur.execute(
                                (
                                    "UPDATE jobs SET status = 'failed', last_error = %s, error_message = %s, error_code = %s, error_class = %s, error_stack = %s::jsonb, completion_token = %s, "
                                    "completed_at = NOW(), leased_until = NULL, worker_id = NULL, lease_id = NULL WHERE id = %s AND status = 'processing' AND worker_id = %s AND lease_id = %s AND (completion_token IS NULL OR completion_token = %s)"
                                ),
                                (
                                    (error_code or error),
                                    error,
                                    error_code,
                                    error_class,
                                    (json.dumps(error_stack) if error_stack is not None else None),
                                    completion_token,
                                    int(job_id),
                                    worker_id,
                                    lease_id,
                                    completion_token,
                                ),
                            )
                            failed_from_processing = cur.rowcount > 0
                        else:
                            cur.execute(
                                (
                                    "UPDATE jobs SET status = 'failed', last_error = %s, error_message = %s, error_code = %s, error_class = %s, error_stack = %s::jsonb, completion_token = COALESCE(completion_token, %s), "
                                    "completed_at = NOW(), leased_until = NULL, worker_id = NULL, lease_id = NULL WHERE id = %s AND status = 'processing' AND (completion_token IS NULL OR completion_token = %s)"
                                ),
                                (
                                    (error_code or error),
                                    error,
                                    error_code,
                                    error_class,
                                    (json.dumps(error_stack) if error_stack is not None else None),
                                    completion_token,
                                    int(job_id),
                                    completion_token,
                                ),
                            )
                            failed_from_processing = cur.rowcount > 0
                            if not failed_from_processing:
                                try:
                                    allow = {
                                        d.strip().lower()
                                        for d in os.getenv(
                                            "JOBS_ADMIN_COMPLETE_QUEUED_ALLOW_DOMAINS",
                                            "chatbooks,embeddings",
                                        ).split(",")
                                        if d.strip()
                                    }
                                    cur.execute("SELECT domain FROM jobs WHERE id = %s", (int(job_id),))
                                    row_dom = cur.fetchone()
                                    dom_val = str(row_dom.get("domain") or "").lower() if row_dom else ""
                                except _JOB_NONCRITICAL_EXCEPTIONS:
                                    allow = {"chatbooks", "embeddings"}
                                    dom_val = ""
                                if dom_val in allow:
                                    cur.execute(
                                        (
                                            "UPDATE jobs SET status = 'failed', last_error = %s, error_message = %s, error_code = %s, error_class = %s, error_stack = %s::jsonb, completion_token = COALESCE(completion_token, %s), "
                                            "completed_at = NOW(), leased_until = NULL, worker_id = NULL, lease_id = NULL WHERE id = %s AND status = 'queued' AND (completion_token IS NULL OR completion_token = %s)"
                                        ),
                                        (
                                            (error_code or error),
                                            error,
                                            error_code,
                                            error_class,
                                            (json.dumps(error_stack) if error_stack is not None else None),
                                            completion_token,
                                            int(job_id),
                                            completion_token,
                                        ),
                                    )
                                    failed_from_queued = cur.rowcount > 0
                        ok = failed_from_processing or failed_from_queued
                        if not ok and completion_token:
                            cur.execute(
                                "SELECT status, completion_token FROM jobs WHERE id=%s",
                                (int(job_id),),
                            )
                            replay = cur.fetchone()
                            if (
                                replay
                                and str(replay.get("status")) == "failed"
                                and str(replay.get("completion_token") or "")
                                == str(completion_token)
                            ):
                                return True
                        if ok and (error_code or error_class or error_stack is not None):
                            with contextlib.suppress(_JOB_NONCRITICAL_EXCEPTIONS):
                                cur.execute(
                                    (
                                        "UPDATE jobs SET error_code = COALESCE(%s, error_code), "
                                        "error_class = COALESCE(%s, error_class), "
                                        "error_stack = COALESCE(%s::jsonb, error_stack) "
                                        "WHERE id = %s"
                                    ),
                                    (
                                        error_code,
                                        error_class,
                                        (json.dumps(error_stack) if error_stack is not None else None),
                                        int(job_id),
                                    ),
                                )
                        try:
                            if ok and elem:
                                d = dict(elem)
                                post_commit_side_effects.append(
                                    (increment_failures, (d,), {"reason": "terminal"})
                                )
                                try:
                                    if error_code:
                                        from .metrics import increment_failures_by_code

                                        post_commit_side_effects.append(
                                            (
                                                increment_failures_by_code,
                                                (d, error_code),
                                                {},
                                            )
                                        )
                                except _JOB_NONCRITICAL_EXCEPTIONS:
                                    pass
                                try:
                                    # Append terminal failure to timeline (no backoff)
                                    cur.execute(
                                        "UPDATE jobs SET failure_timeline = COALESCE(failure_timeline, '[]'::jsonb) || jsonb_build_array(jsonb_build_object('ts', NOW(), 'error_code', %s::text, 'retry_backoff', 0)) WHERE id = %s",
                                        ((error_code or error), int(job_id)),
                                    )

                                    post_commit_side_effects.append(
                                        (
                                            _record_job_span,
                                            ("job.fail", d),
                                            {
                                                "attrs": {
                                                    "retryable": False,
                                                    "error_code": error_code,
                                                }
                                            },
                                        )
                                    )
                                except _PG_ERRORS:
                                    raise
                                except _JOB_NONCRITICAL_EXCEPTIONS:
                                    pass
                                post_commit_side_effects.append(
                                    (
                                        self._update_gauges,
                                        (),
                                        {
                                            "domain": d.get("domain"),
                                            "queue": d.get("queue"),
                                            "job_type": d.get("job_type"),
                                        },
                                    )
                                )
                                post_commit_cancel_uuid = d.get("uuid")
                                post_commit_cancel_reason = "dependency_failed"
                        except _PG_ERRORS:
                            raise
                        except _JOB_NONCRITICAL_EXCEPTIONS:
                            pass
                        if (
                            ok
                            and elem
                            and JobManager._is_truthy(
                                os.getenv("JOBS_COUNTERS_ENABLED", "")
                            )
                        ):
                            d_counter = dict(elem)
                            if failed_from_queued:
                                is_scheduled = (
                                    d_counter.get("available_at") is not None
                                )
                                cur.execute(
                                    (
                                        "UPDATE job_counters SET "
                                        "ready_count = GREATEST(ready_count - %s, 0), "
                                        "scheduled_count = GREATEST(scheduled_count - %s, 0), "
                                        "updated_at = NOW() "
                                        "WHERE domain=%s AND queue=%s AND job_type=%s"
                                    ),
                                    (
                                        int(not is_scheduled),
                                        int(is_scheduled),
                                        d_counter.get("domain"),
                                        d_counter.get("queue"),
                                        d_counter.get("job_type"),
                                    ),
                                )
                            else:
                                cur.execute(
                                    "UPDATE job_counters SET processing_count = GREATEST(processing_count - 1, 0), updated_at = NOW() WHERE domain=%s AND queue=%s AND job_type=%s",
                                    (
                                        d_counter.get("domain"),
                                        d_counter.get("queue"),
                                        d_counter.get("job_type"),
                                    ),
                                )
                            if cur.rowcount == 0:
                                _reconcile_lifecycle_counter_row(
                                    cur,
                                    backend=self.backend,
                                    domain=d_counter.get("domain"),
                                    queue=d_counter.get("queue"),
                                    job_type=d_counter.get("job_type"),
                                )
                        if ok and elem:
                            event_job = {
                                "id": int(job_id),
                                "domain": elem.get("domain"),
                                "queue": elem.get("queue"),
                                "job_type": elem.get("job_type"),
                                "owner_user_id": elem.get("owner_user_id"),
                                "request_id": elem.get("request_id"),
                                "trace_id": elem.get("trace_id"),
                            }
                            attrs = {"error_code": (error_code or error)}
                            if outbox_enabled:
                                _insert_lifecycle_event(
                                    cur,
                                    backend=self.backend,
                                    event_type="job.failed",
                                    job=event_job,
                                    attrs=attrs,
                                )
                            _queue_lifecycle_event_observer(
                                post_commit_side_effects,
                                event_type="job.failed",
                                job=event_job,
                                attrs=attrs,
                            )
                        if _is_test_mode():
                            try:
                                cur.execute("SELECT COUNT(*) AS c FROM jobs")
                                _total = (cur.fetchone() or {}).get("c", 0)
                                cur.execute("SELECT status, COUNT(*) AS c FROM jobs GROUP BY status")
                                _rows = cur.fetchall() or []
                                _dist = {str(x.get("status")): int(x.get("c") or 0) for x in _rows}
                                logger.info(
                                    f"[JM TEST MUT] fail_job terminal ok={bool(ok)} total={_total} dist={_dist}"
                                )
                            except _JOB_NONCRITICAL_EXCEPTIONS:
                                pass
                        if ok:
                            _commit_postgres_transaction(
                                conn,
                                operation="terminal job failure",
                            )
                            _run_post_commit_side_effects(post_commit_side_effects)
                            post_commit_side_effects.clear()
                            if post_commit_cancel_uuid:
                                with contextlib.suppress(_JOB_NONCRITICAL_EXCEPTIONS):
                                    self._cancel_dependent_jobs(
                                        post_commit_cancel_uuid,
                                        reason=post_commit_cancel_reason or "dependency_failed",
                                    )
                        return ok
            else:
                result = False
                with conn:
                    if _test_mode:
                        with contextlib.suppress(_JOB_NONCRITICAL_EXCEPTIONS):
                            logger.info(
                                f"[JM TEST MUT] fail_job enter job_id={job_id} retryable={retryable} backoff={backoff_seconds} enforce={enforce} backend=sqlite"
                            )
                    # For metrics, fetch labels
                    rowl = conn.execute(
                        "SELECT status, completion_token, domain, queue, job_type, uuid, request_id, trace_id, owner_user_id, available_at FROM jobs WHERE id = ?",
                        (job_id,),
                    ).fetchone()
                    if rowl:
                        st = str(rowl[0])
                        ct = rowl[1]
                        if st in {"completed", "failed", "cancelled", "quarantined"}:
                            replay_states = (
                                {"failed", "quarantined"}
                                if retryable
                                else {"failed"}
                            )
                            return bool(
                                st in replay_states
                                and completion_token
                                and ct
                                and str(ct) == str(completion_token)
                            )
                    retry_scheduled = False
                    if retryable:
                        # compute jittered backoff based on current retry_count
                        row = conn.execute("SELECT retry_count FROM jobs WHERE id = ?", (job_id,)).fetchone()
                        current = int(row[0]) if row else 0
                        exp_backoff = max(1, int(backoff_seconds * (2**current)))
                        test_mode = _is_test_mode()
                        jitter = (
                            0
                            if exp_backoff <= 2 or test_mode
                            else secrets.randbelow(max(1, exp_backoff // 4) + 1)
                        )
                        delay = exp_backoff + jitter
                        base_thresh = int(os.getenv("JOBS_QUARANTINE_THRESHOLD", "2") or "2")
                        thresh = base_thresh
                        if test_mode:
                            _outbox = JobManager._is_truthy(os.getenv("JOBS_EVENTS_OUTBOX", ""))
                            if not _outbox and exp_backoff <= 1:
                                delay = 0
                            try:
                                if _outbox and int(backoff_seconds) <= 0 and delay < 10:
                                    delay = 10
                            except _JOB_NONCRITICAL_EXCEPTIONS:
                                if _outbox and delay < 3:
                                    delay = 3
                            if test_mode and int(backoff_seconds) <= 0:
                                # Respect explicit threshold in tests; otherwise, avoid quarantining to allow timeline growth
                                if os.getenv("JOBS_QUARANTINE_THRESHOLD") is None:
                                    thresh = max(base_thresh, 10**9)
                                else:
                                    thresh = base_thresh
                        # SQLite retry path with failure streak bookkeeping
                        if enforce:
                            retry_cursor = conn.execute(
                                (
                                    "UPDATE jobs SET status = CASE WHEN (CASE WHEN COALESCE(failure_streak_code, '') = ? THEN COALESCE(failure_streak_count,0) + 1 ELSE 1 END) >= ? THEN 'quarantined' ELSE 'queued' END, "
                                    "retry_count = retry_count + 1, last_error = ?, error_message = ?, error_code = ?, error_class = ?, error_stack = ?, "
                                    "failure_streak_count = CASE WHEN COALESCE(failure_streak_code, '') = ? THEN COALESCE(failure_streak_count,0) + 1 ELSE 1 END, "
                                    "failure_streak_code = ?, "
                                    "completion_token = CASE WHEN (CASE WHEN COALESCE(failure_streak_code, '') = ? THEN COALESCE(failure_streak_count,0) + 1 ELSE 1 END) >= ? THEN ? ELSE NULL END, "
                                    "available_at = CASE WHEN (CASE WHEN COALESCE(failure_streak_code, '') = ? THEN COALESCE(failure_streak_count,0) + 1 ELSE 1 END) >= ? THEN available_at WHEN ? <= 0 THEN NULL ELSE DATETIME('now', ?) END, "
                                    "quarantined_at = CASE WHEN (CASE WHEN COALESCE(failure_streak_code, '') = ? THEN COALESCE(failure_streak_count,0) + 1 ELSE 1 END) >= ? THEN DATETIME('now') ELSE quarantined_at END, "
                                    "leased_until = NULL, worker_id = NULL, lease_id = NULL "
                                    "WHERE id = ? AND status = 'processing' AND retry_count < max_retries AND worker_id = ? AND lease_id = ? AND (completion_token IS NULL OR completion_token = ?)"
                                ),
                                (
                                    streak_code,
                                    int(thresh),
                                    streak_code,
                                    error,
                                    error_code,
                                    error_class,
                                    (json.dumps(error_stack) if error_stack is not None else None),
                                    streak_code,
                                    streak_code,
                                    streak_code,
                                    int(thresh),
                                    completion_token,
                                    streak_code,
                                    int(thresh),
                                    int(delay),
                                    f"+{delay} seconds",
                                    streak_code,
                                    int(thresh),
                                    job_id,
                                    worker_id,
                                    lease_id,
                                    completion_token,
                                ),
                            )
                        else:
                            retry_cursor = conn.execute(
                                (
                                    "UPDATE jobs SET status = CASE WHEN (CASE WHEN COALESCE(failure_streak_code, '') = ? THEN COALESCE(failure_streak_count,0) + 1 ELSE 1 END) >= ? THEN 'quarantined' ELSE 'queued' END, "
                                    "retry_count = retry_count + 1, last_error = ?, error_message = ?, error_code = ?, error_class = ?, error_stack = ?, "
                                    "failure_streak_count = CASE WHEN COALESCE(failure_streak_code, '') = ? THEN COALESCE(failure_streak_count,0) + 1 ELSE 1 END, "
                                    "failure_streak_code = ?, "
                                    "completion_token = CASE WHEN (CASE WHEN COALESCE(failure_streak_code, '') = ? THEN COALESCE(failure_streak_count,0) + 1 ELSE 1 END) >= ? THEN ? ELSE NULL END, "
                                    "available_at = CASE WHEN (CASE WHEN COALESCE(failure_streak_code, '') = ? THEN COALESCE(failure_streak_count,0) + 1 ELSE 1 END) >= ? THEN available_at WHEN ? <= 0 THEN NULL ELSE DATETIME('now', ?) END, "
                                    "quarantined_at = CASE WHEN (CASE WHEN COALESCE(failure_streak_code, '') = ? THEN COALESCE(failure_streak_count,0) + 1 ELSE 1 END) >= ? THEN DATETIME('now') ELSE quarantined_at END, "
                                    "leased_until = NULL, worker_id = NULL, lease_id = NULL "
                                    "WHERE id = ? AND status = 'processing' AND retry_count < max_retries AND (completion_token IS NULL OR completion_token = ?)"
                                ),
                                (
                                    streak_code,
                                    int(thresh),
                                    streak_code,
                                    error,
                                    error_code,
                                    error_class,
                                    (json.dumps(error_stack) if error_stack is not None else None),
                                    streak_code,
                                    streak_code,
                                    streak_code,
                                    int(thresh),
                                    completion_token,
                                    streak_code,
                                    int(thresh),
                                    int(delay),
                                    f"+{delay} seconds",
                                    streak_code,
                                    int(thresh),
                                    job_id,
                                    completion_token,
                                ),
                            )
                        retry_transition_changed = (retry_cursor.rowcount or 0) > 0
                        if retry_transition_changed:
                            retry_scheduled = True
                            rowq = conn.execute(
                                "SELECT status, uuid FROM jobs WHERE id = ?",
                                (job_id,),
                            ).fetchone()
                            transition_status = str(rowq[0] if rowq else "")
                            if transition_status == "quarantined":
                                post_commit_cancel_uuid = rowq[1]
                                post_commit_cancel_reason = "dependency_failed"
                            try:
                                if rowl:
                                    dtmp = dict(rowl)
                                    if transition_status == "queued":
                                        post_commit_side_effects.append(
                                            (
                                                increment_retries,
                                                (dtmp,),
                                                {},
                                            )
                                        )
                                        try:
                                            from .metrics import observe_retry_after

                                            post_commit_side_effects.append(
                                                (
                                                    observe_retry_after,
                                                    (dtmp, float(delay)),
                                                    {},
                                                )
                                            )
                                        except _JOB_NONCRITICAL_EXCEPTIONS:
                                            pass
                                    post_commit_side_effects.append(
                                        (
                                            self._update_gauges,
                                            (),
                                            {
                                                "domain": dtmp.get("domain"),
                                                "queue": dtmp.get("queue"),
                                                "job_type": dtmp.get("job_type"),
                                            },
                                        )
                                    )
                                    # Append to failure_timeline
                                    try:
                                        row_t = conn.execute(
                                            "SELECT failure_timeline FROM jobs WHERE id = ?", (job_id,)
                                        ).fetchone()
                                        timeline_json = row_t[0] if row_t else None
                                        try:
                                            tl = json.loads(timeline_json) if timeline_json else []
                                        except _JOB_NONCRITICAL_EXCEPTIONS:
                                            tl = []
                                        tl.append(
                                            {
                                                "ts": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                                                "error_code": (error_code or error),
                                                "retry_backoff": int(delay),
                                            }
                                        )
                                        tl = tl[-10:]
                                        conn.execute(
                                            "UPDATE jobs SET failure_timeline = ? WHERE id = ?",
                                            (json.dumps(tl), int(job_id)),
                                        )
                                        # Update last-acquired snapshot for test-mode fallbacks to preserve timeline growth
                                        try:
                                            if (
                                                _is_test_mode()
                                                and rowl
                                            ):
                                                rnow = conn.execute(
                                                    "SELECT * FROM jobs WHERE id = ?", (int(job_id),)
                                                ).fetchone()
                                                if rnow:
                                                    JobManager._LAST_ACQUIRED_TEST[(rowl[2], rowl[3])] = dict(rnow)
                                        except _JOB_NONCRITICAL_EXCEPTIONS:
                                            pass
                                    except _JOB_NONCRITICAL_EXCEPTIONS:
                                        pass
                            except _JOB_NONCRITICAL_EXCEPTIONS:
                                pass
                            if (
                                rowl
                                and JobManager._is_truthy(
                                    os.getenv("JOBS_COUNTERS_ENABLED", "")
                                )
                            ):
                                will_quarantine = (
                                    transition_status == "quarantined"
                                )
                                add_ready = int(
                                    not will_quarantine and int(delay) <= 0
                                )
                                add_sched = int(
                                    not will_quarantine and int(delay) > 0
                                )
                                add_quar = int(will_quarantine)
                                conn.execute(
                                    (
                                        "INSERT INTO job_counters(domain,queue,job_type,ready_count,scheduled_count,processing_count,quarantined_count) VALUES(?,?,?, ?, ?, 0, ?) "
                                        "ON CONFLICT(domain,queue,job_type) DO UPDATE SET ready_count = ready_count + ?, scheduled_count = scheduled_count + ?, processing_count = CASE WHEN processing_count>0 THEN processing_count-1 ELSE 0 END, quarantined_count = quarantined_count + ?, updated_at = DATETIME('now')"
                                    ),
                                    (
                                        rowl[2],
                                        rowl[3],
                                        rowl[4],
                                        add_ready,
                                        add_sched,
                                        add_quar,
                                        add_ready,
                                        add_sched,
                                        add_quar,
                                    ),
                                )
                            if rowl:
                                dtmp = dict(rowl)
                                event_type = (
                                    "job.quarantined"
                                    if transition_status == "quarantined"
                                    else "job.retry_scheduled"
                                )
                                attrs = {
                                    "error_code": (error_code or error),
                                    "retry_count": int(current + 1),
                                }
                                if transition_status == "queued":
                                    attrs["backoff_seconds"] = int(delay)
                                event_job = {
                                    "id": int(job_id),
                                    "domain": dtmp.get("domain"),
                                    "queue": dtmp.get("queue"),
                                    "job_type": dtmp.get("job_type"),
                                    "owner_user_id": dtmp.get("owner_user_id"),
                                    "request_id": dtmp.get("request_id"),
                                    "trace_id": dtmp.get("trace_id"),
                                }
                                if outbox_enabled:
                                    _insert_lifecycle_event(
                                        conn,
                                        backend=self.backend,
                                        event_type=event_type,
                                        job=event_job,
                                        attrs=attrs,
                                    )
                                _queue_lifecycle_event_observer(
                                    post_commit_side_effects,
                                    event_type=event_type,
                                    job=event_job,
                                    attrs=attrs,
                                )
                            if _is_test_mode():
                                try:
                                    _row = conn.execute(
                                        "SELECT failure_timeline FROM jobs WHERE id = ?", (int(job_id),)
                                    ).fetchone()
                                    _tl_len = 0
                                    try:
                                        _tl_len = len(json.loads(_row[0])) if (_row and _row[0]) else 0
                                    except _JOB_NONCRITICAL_EXCEPTIONS:
                                        _tl_len = 0
                                    _total = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
                                    _dist = {
                                        str(r[0]): int(r[1])
                                        for r in conn.execute(
                                            "SELECT status, COUNT(*) FROM jobs GROUP BY status"
                                        ).fetchall()
                                    }
                                    logger.info(
                                        f"[JM TEST MUT] fail_job retryable scheduled delay={int(delay)} tl_len={_tl_len} total={int(_total)} dist={_dist}"
                                    )
                                except _JOB_NONCRITICAL_EXCEPTIONS:
                                    pass
                            result = True
                        elif completion_token:
                            replay = conn.execute(
                                "SELECT status, completion_token FROM jobs WHERE id=?",
                                (int(job_id),),
                            ).fetchone()
                            if (
                                replay
                                and str(replay[0])
                                in {"failed", "quarantined"}
                                and str(replay[1] or "") == str(completion_token)
                            ):
                                return True
                    if not retry_scheduled:
                        # terminal failure
                        failed_from_queued = False
                        if enforce:
                            processing_cursor = conn.execute(
                                (
                                    "UPDATE jobs SET status = 'failed', last_error = ?, error_message = ?, error_code = ?, error_class = ?, error_stack = ?, completion_token = ?, "
                                    "completed_at = DATETIME('now'), leased_until = NULL, worker_id = NULL, lease_id = NULL WHERE id = ? AND status = 'processing' AND worker_id = ? AND lease_id = ? AND (completion_token IS NULL OR completion_token = ?)"
                                ),
                                (
                                    (error_code or error),
                                    error,
                                    error_code,
                                    error_class,
                                    (json.dumps(error_stack) if error_stack is not None else None),
                                    completion_token,
                                    job_id,
                                    worker_id,
                                    lease_id,
                                    completion_token,
                                ),
                            )
                            failed_from_processing = (
                                processing_cursor.rowcount or 0
                            ) > 0
                        else:
                            # Enforcement disabled: allow failing processing without matching worker/lease,
                            # and fall back to failing queued jobs (admin-style terminalization) when appropriate.
                            processing_cursor = conn.execute(
                                (
                                    "UPDATE jobs SET status = 'failed', last_error = ?, error_message = ?, error_code = ?, error_class = ?, error_stack = ?, completion_token = COALESCE(completion_token, ?), "
                                    "completed_at = DATETIME('now'), leased_until = NULL, worker_id = NULL, lease_id = NULL WHERE id = ? AND status = 'processing' AND (completion_token IS NULL OR completion_token = ?)"
                                ),
                                (
                                    (error_code or error),
                                    error,
                                    error_code,
                                    error_class,
                                    (json.dumps(error_stack) if error_stack is not None else None),
                                    completion_token,
                                    job_id,
                                    completion_token,
                                ),
                            )
                            failed_from_processing = (
                                processing_cursor.rowcount or 0
                            ) > 0
                            if not failed_from_processing:
                                # Admin-style finalize: optionally allow failing queued jobs when enforcement is disabled
                                # Scope via allowlist of domains (default: chatbooks) to avoid global behavior in tests
                                try:
                                    allow = {
                                        d.strip().lower()
                                        for d in os.getenv(
                                            "JOBS_ADMIN_COMPLETE_QUEUED_ALLOW_DOMAINS",
                                            "chatbooks,embeddings",
                                        ).split(",")
                                        if d.strip()
                                    }
                                    row_dom = conn.execute("SELECT domain FROM jobs WHERE id = ?", (job_id,)).fetchone()
                                    dom_val = str(row_dom[0]).lower() if row_dom and row_dom[0] else ""
                                except _JOB_NONCRITICAL_EXCEPTIONS:
                                    allow = {"chatbooks", "embeddings"}
                                    dom_val = ""
                                if dom_val in allow:
                                    cur2 = conn.execute(
                                        (
                                            "UPDATE jobs SET status = 'failed', last_error = ?, error_message = ?, error_code = ?, error_class = ?, error_stack = ?, completion_token = COALESCE(completion_token, ?), "
                                            "completed_at = DATETIME('now'), leased_until = NULL, worker_id = NULL, lease_id = NULL WHERE id = ? AND status = 'queued' AND (completion_token IS NULL OR completion_token = ?)"
                                        ),
                                        (
                                            (error_code or error),
                                            error,
                                            error_code,
                                            error_class,
                                            (json.dumps(error_stack) if error_stack is not None else None),
                                            completion_token,
                                            job_id,
                                            completion_token,
                                        ),
                                    )
                                    failed_from_queued = (cur2.rowcount or 0) > 0
                        ok = failed_from_processing or failed_from_queued
                        if not ok and completion_token:
                            replay = conn.execute(
                                "SELECT status, completion_token FROM jobs WHERE id=?",
                                (int(job_id),),
                            ).fetchone()
                            if (
                                replay
                                and str(replay[0]) == "failed"
                                and str(replay[1] or "") == str(completion_token)
                            ):
                                return True
                        try:
                            if ok and rowl:
                                d = dict(rowl)
                                post_commit_side_effects.append(
                                    (increment_failures, (d,), {"reason": "terminal"})
                                )
                                try:
                                    if error_code:
                                        from .metrics import increment_failures_by_code

                                        post_commit_side_effects.append(
                                            (
                                                increment_failures_by_code,
                                                (d, error_code),
                                                {},
                                            )
                                        )
                                except _JOB_NONCRITICAL_EXCEPTIONS:
                                    pass
                                # Append terminal failure to timeline (no backoff)
                                try:
                                    row_t2 = conn.execute(
                                        "SELECT failure_timeline FROM jobs WHERE id = ?", (job_id,)
                                    ).fetchone()
                                    timeline_json2 = row_t2[0] if row_t2 else None
                                    try:
                                        tl2 = json.loads(timeline_json2) if timeline_json2 else []
                                    except _JOB_NONCRITICAL_EXCEPTIONS:
                                        tl2 = []
                                    tl2.append(
                                        {
                                            "ts": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                                            "error_code": (error_code or error),
                                            "retry_backoff": 0,
                                        }
                                    )
                                    tl2 = tl2[-10:]
                                    conn.execute(
                                        "UPDATE jobs SET failure_timeline = ? WHERE id = ?",
                                        (json.dumps(tl2), int(job_id)),
                                    )
                                except _JOB_NONCRITICAL_EXCEPTIONS:
                                    pass
                                try:
                                    post_commit_side_effects.append(
                                        (
                                            _record_job_span,
                                            ("job.fail", d),
                                            {
                                                "attrs": {
                                                    "retryable": False,
                                                    "error_code": error_code,
                                                }
                                            },
                                        )
                                    )
                                except _JOB_NONCRITICAL_EXCEPTIONS:
                                    pass
                                post_commit_side_effects.append(
                                    (
                                        self._update_gauges,
                                        (),
                                        {
                                            "domain": d.get("domain"),
                                            "queue": d.get("queue"),
                                            "job_type": d.get("job_type"),
                                        },
                                    )
                                )
                                try:
                                    if d.get("uuid"):
                                        post_commit_cancel_uuid = d.get("uuid")
                                        post_commit_cancel_reason = "dependency_failed"
                                except _JOB_NONCRITICAL_EXCEPTIONS:
                                    pass
                        except _JOB_NONCRITICAL_EXCEPTIONS:
                            pass
                        if (
                            ok
                            and rowl
                            and JobManager._is_truthy(
                                os.getenv("JOBS_COUNTERS_ENABLED", "")
                            )
                        ):
                            d_counter = dict(rowl)
                            if failed_from_queued:
                                is_scheduled = (
                                    d_counter.get("available_at") is not None
                                )
                                counter_cursor = conn.execute(
                                    (
                                        "UPDATE job_counters SET "
                                        "ready_count = CASE WHEN ready_count > ? THEN ready_count - ? ELSE 0 END, "
                                        "scheduled_count = CASE WHEN scheduled_count > ? THEN scheduled_count - ? ELSE 0 END, "
                                        "updated_at = DATETIME('now') "
                                        "WHERE domain=? AND queue=? AND job_type=?"
                                    ),
                                    (
                                        int(not is_scheduled),
                                        int(not is_scheduled),
                                        int(is_scheduled),
                                        int(is_scheduled),
                                        d_counter.get("domain"),
                                        d_counter.get("queue"),
                                        d_counter.get("job_type"),
                                    ),
                                )
                            else:
                                counter_cursor = conn.execute(
                                    "UPDATE job_counters SET processing_count = CASE WHEN processing_count>0 THEN processing_count-1 ELSE 0 END, updated_at = DATETIME('now') WHERE domain=? AND queue=? AND job_type=?",
                                    (
                                        d_counter.get("domain"),
                                        d_counter.get("queue"),
                                        d_counter.get("job_type"),
                                    ),
                                )
                            if (counter_cursor.rowcount or 0) == 0:
                                _reconcile_lifecycle_counter_row(
                                    conn,
                                    backend=self.backend,
                                    domain=d_counter.get("domain"),
                                    queue=d_counter.get("queue"),
                                    job_type=d_counter.get("job_type"),
                                )
                        if ok and rowl:
                            d_event = dict(rowl)
                            event_job = {
                                "id": int(job_id),
                                "domain": d_event.get("domain"),
                                "queue": d_event.get("queue"),
                                "job_type": d_event.get("job_type"),
                                "owner_user_id": d_event.get("owner_user_id"),
                                "request_id": d_event.get("request_id"),
                                "trace_id": d_event.get("trace_id"),
                            }
                            attrs = {"error_code": (error_code or error)}
                            if outbox_enabled:
                                _insert_lifecycle_event(
                                    conn,
                                    backend=self.backend,
                                    event_type="job.failed",
                                    job=event_job,
                                    attrs=attrs,
                                )
                            _queue_lifecycle_event_observer(
                                post_commit_side_effects,
                                event_type="job.failed",
                                job=event_job,
                                attrs=attrs,
                            )
                        if _is_test_mode():
                            try:
                                _total = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
                                _dist = {
                                    str(r[0]): int(r[1])
                                    for r in conn.execute(
                                        "SELECT status, COUNT(*) FROM jobs GROUP BY status"
                                    ).fetchall()
                                }
                                logger.info(
                                    f"[JM TEST MUT] fail_job terminal ok={bool(ok)} total={int(_total)} dist={_dist}"
                                )
                            except _JOB_NONCRITICAL_EXCEPTIONS:
                                pass
                        result = bool(ok)
            _run_post_commit_side_effects(post_commit_side_effects)
            post_commit_side_effects.clear()
            if post_commit_cancel_uuid:
                with contextlib.suppress(_JOB_NONCRITICAL_EXCEPTIONS):
                    self._cancel_dependent_jobs(
                        post_commit_cancel_uuid,
                        reason=post_commit_cancel_reason or "dependency_failed",
                    )
            return bool(result)
        finally:
            _close_connection_nonfatal(conn, operation="job failure")

    def cancel_job(
        self,
        job_id: int,
        *,
        reason: str | None = None,
        expected_uuid: str | None = None,
        expected_domain: str | None = None,
        expected_job_type: str | None = None,
        cascade_dependents: bool = True,
    ) -> bool:
        """Request cancellation or cancel queued jobs immediately.

        Optional identity guards make lookup-then-cancel callers safe against
        SQLite integer ID reuse and stale references.
        Internal maintenance callers may disable immediate dependent cascading
        so reconciliation remains bounded.

        Emits gauge updates on successful cancellation for the job's domain/queue/job_type.
        """
        conn = self._connect()
        _test_mode = _is_test_mode()
        outbox_enabled = JobManager._is_truthy(os.getenv("JOBS_EVENTS_OUTBOX", ""))
        counters_enabled = JobManager._is_truthy(os.getenv("JOBS_COUNTERS_ENABLED", ""))
        result = False
        cancelled_row: dict[str, Any] | None = None
        event_job: dict[str, Any] | None = None
        event_attrs = {"reason": reason, "terminal": True}
        post_commit_cancel_uuid: str | None = None

        def _identity_guard(placeholder: str) -> tuple[str, list[Any]]:
            clauses = [f"id = {placeholder}"]
            values: list[Any] = [int(job_id)]
            if expected_uuid is not None:
                clauses.append(f"uuid = {placeholder}")
                values.append(str(expected_uuid))
            if expected_domain is not None:
                clauses.append(f"domain = {placeholder}")
                values.append(str(expected_domain))
            if expected_job_type is not None:
                clauses.append(f"job_type = {placeholder}")
                values.append(str(expected_job_type))
            return " AND ".join(clauses), values

        def _event_job(row: dict[str, Any]) -> dict[str, Any]:
            return {
                "id": int(job_id),
                "domain": row["domain"],
                "queue": row["queue"],
                "job_type": row["job_type"],
                "owner_user_id": row["owner_user_id"],
                "request_id": row["request_id"],
                "trace_id": row["trace_id"],
            }

        def _persist_cancelled_event(executor: Any, job: dict[str, Any]) -> None:
            if not outbox_enabled:
                return
            params = (
                job["id"],
                job["domain"],
                job["queue"],
                job["job_type"],
                json.dumps(event_attrs),
                job["owner_user_id"],
                job["request_id"],
                job["trace_id"],
            )
            if self.backend == "postgres":
                executor.execute(
                    (
                        "INSERT INTO job_events(job_id,domain,queue,job_type,event_type,attrs_json,owner_user_id,request_id,trace_id,created_at) "
                        "VALUES(%s,%s,%s,%s,'job.cancelled',%s::jsonb,%s,%s,%s,NOW())"
                    ),
                    params,
                )
            else:
                executor.execute(
                    (
                        "INSERT INTO job_events(job_id,domain,queue,job_type,event_type,attrs_json,owner_user_id,request_id,trace_id,created_at) "
                        "VALUES(?,?,?,?,'job.cancelled',?,?,?,?,DATETIME('now'))"
                    ),
                    params,
                )

        try:
            if self.backend == "postgres":
                identity_sql, identity_params = _identity_guard("%s")
                with conn:  # noqa: SIM117
                    with self._pg_cursor(conn) as cur:
                        if _test_mode:
                            with contextlib.suppress(_JOB_NONCRITICAL_EXCEPTIONS):
                                logger.info(f"[JM TEST MUT] cancel_job enter job_id={job_id} backend=pg")
                        # identity_sql contains only fixed column predicates; all values stay parameterized.
                        cur.execute(
                            "SELECT id, domain, queue, job_type, uuid, owner_user_id, request_id, "  # nosec B608
                            "trace_id, status, available_at FROM jobs WHERE "
                            + identity_sql
                            + " FOR UPDATE",
                            tuple(identity_params),
                        )
                        selected = cur.fetchone()
                        if selected and str(selected.get("status")) in {"queued", "processing"}:
                            row = dict(selected)
                            previous_status = str(row["status"])
                            cur.execute(
                                "UPDATE jobs SET status = 'cancelled', cancelled_at = NOW(), "  # nosec B608
                                "cancellation_reason = %s, leased_until = NULL, worker_id = NULL, "
                                "lease_id = NULL WHERE "
                                + identity_sql
                                + " AND status = %s",
                                (reason, *identity_params, previous_status),
                            )
                            result = cur.rowcount > 0
                            if result and counters_enabled:
                                if previous_status == "queued":
                                    is_scheduled = row.get("available_at") is not None
                                    add_ready = -1 if not is_scheduled else 0
                                    add_scheduled = -1 if is_scheduled else 0
                                    cur.execute(
                                        (
                                            "INSERT INTO job_counters(domain,queue,job_type,ready_count,scheduled_count,processing_count,quarantined_count) VALUES(%s,%s,%s,0,0,0,0) "
                                            "ON CONFLICT (domain,queue,job_type) DO UPDATE SET ready_count = GREATEST(job_counters.ready_count + %s, 0), scheduled_count = GREATEST(job_counters.scheduled_count + %s, 0), updated_at = NOW()"
                                        ),
                                        (
                                            row["domain"],
                                            row["queue"],
                                            row["job_type"],
                                            add_ready,
                                            add_scheduled,
                                        ),
                                    )
                                else:
                                    cur.execute(
                                        "UPDATE job_counters SET processing_count = GREATEST(processing_count - 1, 0), updated_at = NOW() WHERE domain=%s AND queue=%s AND job_type=%s",
                                        (row["domain"], row["queue"], row["job_type"]),
                                    )
                            if result:
                                event_job = _event_job(row)
                                _persist_cancelled_event(cur, event_job)
                                cancelled_row = row
                                if cascade_dependents and row.get("uuid"):
                                    post_commit_cancel_uuid = str(row["uuid"])
            else:
                identity_sql, identity_params = _identity_guard("?")
                with conn:
                    conn.execute("BEGIN IMMEDIATE")
                    if _test_mode:
                        with contextlib.suppress(_JOB_NONCRITICAL_EXCEPTIONS):
                            logger.info(f"[JM TEST MUT] cancel_job enter job_id={job_id} backend=sqlite")
                    selected = conn.execute(
                        "SELECT id, domain, queue, job_type, uuid, owner_user_id, request_id, "
                        "trace_id, status, available_at FROM jobs WHERE "
                        + identity_sql,  # nosec B608
                        tuple(identity_params),
                    ).fetchone()
                    if selected and str(selected["status"]) in {"queued", "processing"}:
                        row = dict(selected)
                        previous_status = str(row["status"])
                        cursor = conn.execute(
                            "UPDATE jobs SET status = 'cancelled', "  # nosec B608
                            "cancelled_at = DATETIME('now'), cancellation_reason = ?, "
                            "leased_until = NULL, worker_id = NULL, lease_id = NULL WHERE "
                            + identity_sql
                            + " AND status = ?",
                            (reason, *identity_params, previous_status),
                        )
                        result = cursor.rowcount > 0
                        if result and counters_enabled:
                            if previous_status == "queued":
                                is_scheduled = row.get("available_at") is not None
                                add_ready = -1 if not is_scheduled else 0
                                add_scheduled = -1 if is_scheduled else 0
                                conn.execute(
                                    (
                                        "INSERT INTO job_counters(domain,queue,job_type,ready_count,scheduled_count,processing_count,quarantined_count) VALUES(?,?,?,0,0,0,0) "
                                        "ON CONFLICT(domain,queue,job_type) DO UPDATE SET ready_count = CASE WHEN (ready_count + ?) < 0 THEN 0 ELSE ready_count + ? END, scheduled_count = CASE WHEN (scheduled_count + ?) < 0 THEN 0 ELSE scheduled_count + ? END, updated_at = DATETIME('now')"
                                    ),
                                    (
                                        row["domain"],
                                        row["queue"],
                                        row["job_type"],
                                        add_ready,
                                        add_ready,
                                        add_scheduled,
                                        add_scheduled,
                                    ),
                                )
                            else:
                                conn.execute(
                                    "UPDATE job_counters SET processing_count = CASE WHEN processing_count>0 THEN processing_count-1 ELSE 0 END, updated_at = DATETIME('now') WHERE domain=? AND queue=? AND job_type=?",
                                    (row["domain"], row["queue"], row["job_type"]),
                                )
                        if result:
                            event_job = _event_job(row)
                            _persist_cancelled_event(conn, event_job)
                            cancelled_row = row
                            if cascade_dependents and row.get("uuid"):
                                post_commit_cancel_uuid = str(row["uuid"])
        finally:
            _close_connection_nonfatal(conn, operation="job cancellation")

        if result and cancelled_row and event_job:
            with contextlib.suppress(_JOB_NONCRITICAL_EXCEPTIONS):
                self._update_gauges(
                    domain=cancelled_row["domain"],
                    queue=cancelled_row["queue"],
                    job_type=cancelled_row["job_type"],
                )
            with contextlib.suppress(_JOB_NONCRITICAL_EXCEPTIONS):
                increment_cancelled(dict(cancelled_row))
            with contextlib.suppress(_JOB_NONCRITICAL_EXCEPTIONS):
                if outbox_enabled:
                    submit_job_audit_event("job.cancelled", job=event_job, attrs=event_attrs)
                else:
                    emit_job_event("job.cancelled", job=event_job, attrs=event_attrs)
            if post_commit_cancel_uuid:
                with contextlib.suppress(_JOB_NONCRITICAL_EXCEPTIONS):
                    self._cancel_dependent_jobs(
                        post_commit_cancel_uuid,
                        reason=reason or "dependency_cancelled",
                    )
        return bool(result)

    def release_job(
        self,
        job_id: int,
        *,
        worker_id: str | None = None,
        lease_id: str | None = None,
        reason: str | None = None,
        enforce: bool | None = None,
    ) -> bool:
        """Release a processing job back to queued without retry penalties."""
        if enforce is None:
            enforce = self._should_enforce_ack()
        if enforce and (not worker_id or not lease_id):
            return False
        released_job: dict[str, Any] | None = None
        conn = self._connect()
        try:
            if self.backend == "postgres":
                result = _postgres_release_job(
                    conn,
                    self._pg_cursor,
                    command=ReleaseJobCommand(
                        job_id=int(job_id),
                        enforce=bool(enforce),
                        worker_id=worker_id,
                        lease_id=lease_id,
                        reason=reason,
                    ),
                    counters_enabled=JobManager._is_truthy(
                        os.getenv("JOBS_COUNTERS_ENABLED", "")
                    ),
                )
                if result.outcome is OperationOutcome.APPLIED and result.row is not None:
                    released_job = {
                        "id": int(job_id),
                        "domain": result.row["domain"],
                        "queue": result.row["queue"],
                        "job_type": result.row["job_type"],
                    }
            else:
                result = _sqlite_release_job(
                    conn,
                    command=ReleaseJobCommand(
                        job_id=int(job_id),
                        enforce=bool(enforce),
                        worker_id=worker_id,
                        lease_id=lease_id,
                        reason=reason,
                    ),
                    counters_enabled=JobManager._is_truthy(
                        os.getenv("JOBS_COUNTERS_ENABLED", "")
                    ),
                )
                if result.outcome is OperationOutcome.APPLIED and result.row is not None:
                    released_job = {
                        "id": int(job_id),
                        "domain": result.row["domain"],
                        "queue": result.row["queue"],
                        "job_type": result.row["job_type"],
                    }
        finally:
            conn.close()

        if released_job is None:
            return False
        with contextlib.suppress(_JOB_NONCRITICAL_EXCEPTIONS):
            self._update_gauges(
                domain=released_job["domain"],
                queue=released_job["queue"],
                job_type=released_job["job_type"],
            )
        if reason:
            with contextlib.suppress(_JOB_NONCRITICAL_EXCEPTIONS):
                emit_job_event("job.released", job=released_job, attrs={"reason": reason})
        return True

    @staticmethod
    def _validate_receipt_candidate_rows(rows: list[Any]) -> dict[int, str]:
        """Validate receipt-to-active-Job correlations selected for pruning."""

        candidates: dict[int, str] = {}
        for raw_row in rows:
            row = dict(raw_row)
            job_id = int(row["active_id"])
            job_uuid = str(row.get("active_uuid") or "").strip()
            valid = (
                bool(job_uuid)
                and job_id == int(row["receipt_job_id"])
                and job_uuid == row.get("receipt_job_uuid")
                and row.get("active_domain") == row.get("receipt_domain")
                and row.get("active_queue") == row.get("receipt_queue")
                and row.get("active_job_type") == row.get("receipt_job_type")
                and row.get("active_owner_user_id")
                == row.get("receipt_owner_user_id")
                and row.get("active_batch_group")
                == row.get("receipt_operation_scope")
            )
            if not valid:
                raise IdempotentOperationUnavailableError(
                    "receipt and prune candidate correlation do not match"
                )
            observed_uuid = candidates.setdefault(job_id, job_uuid)
            if observed_uuid != job_uuid:
                raise IdempotentOperationUnavailableError(
                    "prune candidate resolves to multiple Job UUIDs"
                )
        return candidates

    def _receipt_backed_prune_candidates(
        self,
        conn: Any,
        *,
        where_clause: str,
        params: tuple[Any, ...],
        cursor: Any | None = None,
    ) -> dict[int, str]:
        """Return validated receipt-backed active Jobs in one prune population."""

        query = f"""
            WITH candidates AS (
              SELECT id, uuid, domain, queue, job_type, owner_user_id, batch_group
              FROM jobs{where_clause}
            )
            SELECT candidates.id AS active_id,
                   candidates.uuid AS active_uuid,
                   candidates.domain AS active_domain,
                   candidates.queue AS active_queue,
                   candidates.job_type AS active_job_type,
                   candidates.owner_user_id AS active_owner_user_id,
                   candidates.batch_group AS active_batch_group,
                   receipts.job_id AS receipt_job_id,
                   receipts.job_uuid AS receipt_job_uuid,
                   receipts.domain AS receipt_domain,
                   receipts.queue AS receipt_queue,
                   receipts.job_type AS receipt_job_type,
                   receipts.owner_user_id AS receipt_owner_user_id,
                   receipts.operation_scope AS receipt_operation_scope
            FROM candidates
            JOIN job_idempotency_receipts AS receipts
              ON receipts.job_uuid = candidates.uuid
              OR receipts.job_id = candidates.id
            ORDER BY candidates.id, receipts.receipt_id
        """  # nosec B608
        if self.backend == "postgres":
            if cursor is None:
                raise RuntimeError("PostgreSQL receipt pruning requires a cursor")
            cursor.execute(query, params)
            rows = list(cursor.fetchall() or [])
        else:
            rows = list(conn.execute(query, params).fetchall() or [])
        return self._validate_receipt_candidate_rows(rows)

    def _exact_receipt_archive_uuids(
        self,
        conn: Any,
        *,
        where_clause: str,
        params: tuple[Any, ...],
        cursor: Any | None = None,
    ) -> set[str]:
        """Return exact existing archives and reject ambiguous/corrupt copies."""

        projection_fields = ("id", *SLIDES_ARCHIVE_EXACT_FIELDS)
        active_projection = ", ".join(
            f"candidates.{field} AS active__{field}"
            for field in projection_fields
        )
        archive_projection = ", ".join(
            f"archived.{field} AS archived__{field}"
            for field in projection_fields
        )
        query = f"""
            WITH candidates AS (
              SELECT * FROM jobs{where_clause}
            )
            SELECT {active_projection}, archived.archive_id AS archived__archive_id,
                   {archive_projection},
                   archived.payload_compressed AS archived__payload_compressed,
                   archived.result_compressed AS archived__result_compressed
            FROM candidates
            LEFT JOIN jobs_archive AS archived ON archived.uuid = candidates.uuid
            ORDER BY candidates.id, archived.archive_id
        """  # nosec B608
        if self.backend == "postgres":
            if cursor is None:
                raise RuntimeError("PostgreSQL archive validation requires a cursor")
            cursor.execute(query, params)
            rows = list(cursor.fetchall() or [])
        else:
            rows = list(conn.execute(query, params).fetchall() or [])

        grouped: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
        for raw_row in rows:
            row = dict(raw_row)
            active = {
                field: row.get(f"active__{field}") for field in projection_fields
            }
            job_uuid = str(active.get("uuid") or "").strip()
            if not job_uuid:
                raise IdempotentOperationUnavailableError(
                    "receipt-backed prune candidate has no Job UUID"
                )
            _, archived_rows = grouped.setdefault(job_uuid, (active, []))
            if row.get("archived__archive_id") is not None:
                archived_rows.append(
                    {
                        field: row.get(f"archived__{field}")
                        for field in projection_fields
                    }
                    | {
                        "payload_compressed": row.get(
                            "archived__payload_compressed"
                        ),
                        "result_compressed": row.get(
                            "archived__result_compressed"
                        ),
                    }
                )

        exact_uuids: set[str] = set()
        for job_uuid, (active_raw, archived_rows) in grouped.items():
            if not archived_rows:
                continue
            if len(archived_rows) != 1:
                raise IdempotentOperationUnavailableError(
                    "receipt-backed Job has ambiguous archive authority"
                )
            active = normalize_slides_archive_projection(active_raw)
            archived = normalize_slides_archive_projection(archived_rows[0])
            if any(
                active.get(field) != archived.get(field)
                for field in projection_fields
            ):
                raise IdempotentOperationUnavailableError(
                    "receipt-backed Job archive does not match the active row"
                )
            exact_uuids.add(job_uuid)
        return exact_uuids

    def _prune_postgres_batch(
        self,
        cur: Any,
        candidate_ids: list[int],
        *,
        archive_enabled: bool,
        test_mode: bool,
    ) -> int:
        """Archive and delete one already-locked PostgreSQL prune batch."""

        if not candidate_ids:
            return 0
        candidate_where_clause = " WHERE id = ANY(%s)"
        candidate_params: tuple[Any, ...] = (candidate_ids,)
        receipt_candidates = self._receipt_backed_prune_candidates(
            None,
            where_clause=candidate_where_clause,
            params=candidate_params,
            cursor=cur,
        )
        cur.execute(
            "SELECT id FROM jobs WHERE id = ANY(%s) AND domain=%s AND queue=%s "
            "AND job_type=%s AND status IN ('completed','failed','cancelled','quarantined')",
            (candidate_ids, "notes", "graph-suggestions", "note_graph_suggestions"),
        )
        notes_graph_candidate_ids = {
            int(row["id"] if isinstance(row, dict) else row[0])
            for row in (cur.fetchall() or [])
        }
        archive_candidate_ids = (
            candidate_ids
            if archive_enabled
            else sorted(set(receipt_candidates) | notes_graph_candidate_ids)
        )
        cur.execute(
            (
                "UPDATE job_dependencies AS dependency_edge SET "
                "depends_on_terminal_status = CASE WHEN terminal_job.status IN "
                "('completed','failed','cancelled','quarantined') "
                "THEN terminal_job.status ELSE 'cancelled' END, "
                "depends_on_cancellation_reason = CASE WHEN terminal_job.status IN "
                "('completed','failed','cancelled','quarantined') "
                "THEN terminal_job.cancellation_reason ELSE 'pruned' END "
                "FROM jobs AS terminal_job WHERE "
                "terminal_job.uuid = dependency_edge.depends_on_job_uuid AND "
                "terminal_job.id = ANY(%s)"
            ),
            candidate_params,
        )
        if archive_candidate_ids:
            archive_candidate_where_clause = " WHERE id = ANY(%s)"
            archive_candidate_params: tuple[Any, ...] = (archive_candidate_ids,)
            archive_projection = ", ".join(("id", *SLIDES_ARCHIVE_EXACT_FIELDS))
            archive_select_projection = ", ".join(
                (
                    "CASE WHEN status IN ('completed','failed','cancelled','quarantined') "
                    f"THEN NULL ELSE {column} END AS {column}"
                    if column in {"leased_until", "lease_id", "worker_id"}
                    else column
                )
                for column in ("id", *SLIDES_ARCHIVE_EXACT_FIELDS)
            )
            exact_collisions = self._idempotent_slides_archive_collisions(
                None,
                where_clause=archive_candidate_where_clause,
                params=archive_candidate_params,
                cursor=cur,
            )
            if receipt_candidates:
                receipt_ids = sorted(receipt_candidates)
                receipt_where_clause = " WHERE id = ANY(%s)"
                exact_collisions.update(
                    self._exact_receipt_archive_uuids(
                        None,
                        where_clause=receipt_where_clause,
                        params=(receipt_ids,),
                        cursor=cur,
                    )
                )
            archive_where_clause = archive_candidate_where_clause
            archive_params: tuple[Any, ...] = archive_candidate_params
            if exact_collisions:
                archive_where_clause += " AND (uuid IS NULL OR NOT (uuid = ANY(%s)))"
                archive_params = (*archive_candidate_params, sorted(exact_collisions))
            prompt_archive_params = (
                *archive_candidate_params,
                "prompt_studio",
                "optimization",
            )
            cur.execute(
                f"SELECT id, queue, payload FROM jobs{archive_candidate_where_clause} "  # nosec B608
                "AND domain = %s AND job_type = %s FOR UPDATE",
                prompt_archive_params,
            )
            for prompt_row in cur.fetchall() or []:
                secured_payload = self._secured_prompt_archive_payload(
                    prompt_row["payload"],
                    queue=str(prompt_row["queue"]),
                )
                if secured_payload is not None:
                    cur.execute(
                        "UPDATE jobs SET payload = %s::jsonb "
                        "WHERE id = %s AND domain = %s AND job_type = %s",
                        (
                            secured_payload,
                            int(prompt_row["id"]),
                            "prompt_studio",
                            "optimization",
                        ),
                    )
            archive_compress = JobManager._is_truthy(
                os.getenv("JOBS_ARCHIVE_COMPRESS", "")
            )
            archive_returning = (
                " RETURNING archive_id, payload, result"
                if archive_compress
                else ""
            )
            archive_insert_sql = (
                f"WITH locked_jobs AS (SELECT * FROM jobs{archive_where_clause} FOR UPDATE) "  # nosec B608
                f"INSERT INTO jobs_archive ({archive_projection}) "
                f"SELECT {archive_select_projection} FROM locked_jobs"
                + archive_returning
            )
            cur.execute(archive_insert_sql, archive_params)
            inserted_archive_rows = (
                cur.fetchall() or [] if archive_compress else []
            )
            try:
                if archive_compress:
                    import gzip

                    drop_json = JobManager._is_truthy(
                        os.getenv("JOBS_ARCHIVE_COMPRESS_DROP_JSON", "")
                    )
                    for row in inserted_archive_rows:
                        try:
                            archive_id = (
                                int(row["archive_id"])
                                if isinstance(row, dict)
                                else int(row[0])
                            )
                            payload = row.get("payload") if isinstance(row, dict) else row[1]
                            result = row.get("result") if isinstance(row, dict) else row[2]
                            payload_bytes = (
                                gzip.compress(json.dumps(payload).encode("utf-8"))
                                if payload is not None
                                else None
                            )
                            result_bytes = (
                                gzip.compress(json.dumps(result).encode("utf-8"))
                                if result is not None
                                else None
                            )
                            if drop_json:
                                cur.execute(
                                    "UPDATE jobs_archive SET payload=NULL, result=NULL, "
                                    "payload_compressed=%s, result_compressed=%s "
                                    "WHERE archive_id=%s",
                                    (payload_bytes, result_bytes, archive_id),
                                )
                            else:
                                cur.execute(
                                    "UPDATE jobs_archive SET payload_compressed=%s, "
                                    "result_compressed=%s WHERE archive_id=%s",
                                    (payload_bytes, result_bytes, archive_id),
                                )
                        except _JOB_NONCRITICAL_EXCEPTIONS:
                            continue
            except _JOB_NONCRITICAL_EXCEPTIONS:
                pass
            if receipt_candidates:
                archived_receipt_uuids = self._exact_receipt_archive_uuids(
                    None,
                    where_clause=" WHERE id = ANY(%s)",
                    params=(sorted(receipt_candidates),),
                    cursor=cur,
                )
                if archived_receipt_uuids != set(receipt_candidates.values()):
                    raise IdempotentOperationUnavailableError(
                        "receipt-backed Jobs were not archived exactly once"
                    )

        if JobManager._is_truthy(os.getenv("JOBS_COUNTERS_ENABLED", "")):
            counter_groups = (
                ("queued", "available_at IS NULL", "ready_count"),
                ("queued", "available_at IS NOT NULL", "scheduled_count"),
                ("processing", "TRUE", "processing_count"),
                ("quarantined", "TRUE", "quarantined_count"),
            )
            for status, predicate, counter_column in counter_groups:
                cur.execute(
                    f"SELECT domain, queue, job_type, COUNT(*) AS c "  # nosec B608
                    f"FROM jobs{candidate_where_clause} AND status=%s "
                    f"AND {predicate} GROUP BY domain, queue, job_type",
                    (*candidate_params, status),
                )
                for row in cur.fetchall() or []:
                    cur.execute(
                        f"UPDATE job_counters SET {counter_column} = "  # nosec B608
                        f"GREATEST({counter_column} - %s, 0), updated_at=NOW() "
                        "WHERE domain=%s AND queue=%s AND job_type=%s",
                        (
                            int(row["c"]),
                            row["domain"],
                            row["queue"],
                            row["job_type"],
                        ),
                    )

        before_count: int | None = None
        if test_mode:
            with contextlib.suppress(_JOB_NONCRITICAL_EXCEPTIONS):
                cur.execute("SELECT COUNT(*) AS c FROM jobs")
                before_count = int((cur.fetchone() or {}).get("c", 0))
        cur.execute(
            f"DELETE FROM jobs{candidate_where_clause}",  # nosec B608
            candidate_params,
        )
        deleted = int(cur.rowcount or 0)
        if test_mode:
            with contextlib.suppress(_JOB_NONCRITICAL_EXCEPTIONS):
                cur.execute("SELECT COUNT(*) AS c FROM jobs")
                after_count = int((cur.fetchone() or {}).get("c", 0))
                logger.info(
                    "[JM TEST MUT] prune_jobs deleted={} before={} after={}",
                    deleted,
                    before_count,
                    after_count,
                )
        return deleted

    # Maintenance
    def prune_jobs(
        self,
        *,
        statuses: list[str] | None = None,
        older_than_days: int = 30,
        domain: str | None = None,
        queue: str | None = None,
        job_type: str | None = None,
        dry_run: bool = False,
        detail_top_k: int = 0,
    ) -> int:
        """Delete or count jobs in selected statuses older than the cutoff."""

        statuses = statuses or ["completed", "failed", "cancelled"]
        if not statuses:
            return 0
        prune_ref = self._clock.now_utc()
        if prune_ref.tzinfo is None:
            prune_ref = prune_ref.replace(tzinfo=_tz.utc)
        else:
            prune_ref = prune_ref.astimezone(_tz.utc)
        prune_ref_sql = prune_ref.strftime("%Y-%m-%d %H:%M:%S.%f")
        archive_enabled = JobManager._is_truthy(os.getenv("JOBS_ARCHIVE_BEFORE_DELETE", ""))
        slides_archive_fence = (
            archive_enabled
            and not dry_run
            and domain in (None, _SLIDES_GENERATION_DOMAIN)
            and queue in (None, _SLIDES_GENERATION_QUEUE)
            and job_type in (None, _SLIDES_GENERATION_JOB_TYPE)
        )
        archive_projection = ", ".join(("id", *SLIDES_ARCHIVE_EXACT_FIELDS))
        archive_select_projection = ", ".join(
            (
                "CASE WHEN status IN ('completed','failed','cancelled','quarantined') "
                f"THEN NULL ELSE {column} END AS {column}"
                if column in {"leased_until", "lease_id", "worker_id"}
                else column
            )
            for column in ("id", *SLIDES_ARCHIVE_EXACT_FIELDS)
        )
        conn = self._connect()
        slides_diagnostic_persisted = False
        _test_mode = _is_test_mode()
        deleted = 0
        try:
            if self.backend == "postgres":
                with conn:  # noqa: SIM117
                    with self._pg_cursor(conn) as cur:
                        if slides_archive_fence:
                            cur.execute(
                                "SELECT pg_advisory_xact_lock(%s)",
                                (self._pg_advisory_key(*_SLIDES_GENERATION_CORRELATION_LOCK_PARTS),),
                            )
                        where_parts: list[str] = []
                        params: list[Any] = []
                        placeholders = ",".join(["%s"] * len(statuses))
                        where_parts.append(f"status IN ({placeholders})")
                        params.extend(statuses)
                        where_parts.append(
                            "COALESCE(completed_at, created_at) <= "
                            "NOW() - (%s || ' days')::interval"
                        )
                        params.append(int(older_than_days))
                        where_parts.append(
                            "(NOT (domain=%s AND queue=%s AND job_type=%s AND status IN "
                            "('completed','failed','cancelled','quarantined')) OR "
                            "COALESCE(completed_at, created_at) <= "
                            "NOW() - INTERVAL '31 days')"
                        )
                        params.extend(
                            ["notes", "graph-suggestions", "note_graph_suggestions"]
                        )
                        for column, value in (
                            ("domain", domain),
                            ("queue", queue),
                            ("job_type", job_type),
                        ):
                            if value:
                                where_parts.append(f"{column} = %s")
                                params.append(value)
                        where_clause = " WHERE " + " AND ".join(where_parts)
                        if dry_run and detail_top_k > 0:
                            cur.execute(
                                f"SELECT domain, queue, job_type, status, "  # nosec B608
                                f"COUNT(*) AS c FROM jobs{where_clause} "
                                "GROUP BY domain, queue, job_type, status "
                                "ORDER BY c DESC LIMIT %s",
                                tuple(params + [int(detail_top_k)]),
                            )
                        if dry_run:
                            cur.execute(
                                f"SELECT COUNT(*) AS c FROM jobs{where_clause}",  # nosec B608
                                tuple(params),
                            )
                            row = cur.fetchone()
                            count = int(row["c"]) if row is not None else 0
                            with contextlib.suppress(_JOB_NONCRITICAL_EXCEPTIONS):
                                emit_job_event(
                                    "jobs.pruned",
                                    job=None,
                                    attrs={
                                        "deleted": count,
                                        "dry_run": True,
                                        "statuses": ",".join(statuses),
                                        "older_than_days": int(older_than_days),
                                        "domain": domain,
                                        "queue": queue,
                                        "job_type": job_type,
                                    },
                                )
                            return count

                        # Freeze the exact prune population under row locks before
                        # any archive, counter, or delete mutation. Process that
                        # immutable ID set in bounded batches using the current
                        # archive-before-delete helper.
                        cur.execute(
                            f"SELECT id FROM jobs{where_clause} "  # nosec B608
                            "ORDER BY id FOR UPDATE",
                            tuple(params),
                        )
                        locked_ids = [
                            int(row["id"] if isinstance(row, dict) else row[0])
                            for row in (cur.fetchall() or [])
                        ]
                        for offset in range(0, len(locked_ids), _PRUNE_BATCH_SIZE):
                            candidate_ids = locked_ids[
                                offset : offset + _PRUNE_BATCH_SIZE
                            ]
                            deleted += self._prune_postgres_batch(
                                cur,
                                candidate_ids,
                                archive_enabled=archive_enabled,
                                test_mode=_test_mode,
                            )
            else:
                with conn:
                    if not dry_run:
                        conn.execute("BEGIN IMMEDIATE")
                    if _test_mode:
                        with contextlib.suppress(_JOB_NONCRITICAL_EXCEPTIONS):
                            logger.info(
                                f"[JM TEST MUT] prune_jobs enter statuses={statuses} older_than_days={older_than_days} domain={domain} queue={queue} job_type={job_type} backend=sqlite"
                            )
                    where_parts: list[str] = []
                    params: list[Any] = []
                    placeholders = ",".join(["?"] * len(statuses))
                    where_parts.append(f"status IN ({placeholders})")
                    params.extend(statuses)
                    # Reuse one cutoff for count, dependency snapshot, archive,
                    # counters, and delete throughout this transaction.
                    where_parts.append(
                        "julianday(COALESCE(completed_at, created_at)) "
                        "<= julianday(?, ?)"
                    )
                    params.extend(
                        [prune_ref_sql, f"-{int(older_than_days)} days"]
                    )
                    where_parts.append(
                        "(NOT (domain=? AND queue=? AND job_type=? AND status IN "
                        "('completed','failed','cancelled','quarantined')) OR "
                        "julianday(COALESCE(completed_at, created_at)) "
                        "<= julianday(?, '-31 days'))"
                    )
                    params.extend(
                        [
                            "notes",
                            "graph-suggestions",
                            "note_graph_suggestions",
                            prune_ref_sql,
                        ]
                    )
                    if domain:
                        where_parts.append("domain = ?")
                        params.append(domain)
                    if queue:
                        where_parts.append("queue = ?")
                        params.append(queue)
                    if job_type:
                        where_parts.append("job_type = ?")
                        params.append(job_type)
                    where_clause = " WHERE " + " AND ".join(where_parts)
                    # Diagnostics in TEST_MODE: show which rows match the prune filter (SQLite)
                    try:
                        if _is_test_mode():
                            dbg_rows = conn.execute(
                                f"SELECT id, status, completed_at, created_at FROM jobs{where_clause}",  # nosec B608
                                tuple(params),
                            ).fetchall()
                            all_rows = conn.execute(
                                "SELECT id, status, completed_at, created_at FROM jobs", ()
                            ).fetchall()
                            logger.debug(
                                f"SQLite prune debug: total={len(all_rows)} sample={[tuple(r) for r in all_rows]}"
                            )
                            logger.debug(
                                f"SQLite prune debug: matches={len(dbg_rows)} statuses={statuses} older_than_days={older_than_days} ids={[int(r[0]) for r in dbg_rows]}"
                            )
                    except _JOB_NONCRITICAL_EXCEPTIONS:
                        pass
                    # Compute match count up front for accurate reporting
                    cur_cnt = conn.execute(f"SELECT COUNT(*) FROM jobs{where_clause}", tuple(params))  # nosec B608
                    row = cur_cnt.fetchone()
                    count = int(row[0]) if row is not None else 0
                    if dry_run:
                        with contextlib.suppress(_JOB_NONCRITICAL_EXCEPTIONS):
                            emit_job_event(
                                "jobs.pruned",
                                job=None,
                                attrs={
                                    "deleted": int(count),
                                    "dry_run": True,
                                    "statuses": ",".join(statuses),
                                    "older_than_days": int(older_than_days),
                                    "domain": domain,
                                    "queue": queue,
                                    "job_type": job_type,
                                },
                            )
                        if _test_mode:
                            with contextlib.suppress(_JOB_NONCRITICAL_EXCEPTIONS):
                                logger.info(f"[JM TEST MUT] prune_jobs dry_run count={int(count)}")
                        return count
                    receipt_candidates = self._receipt_backed_prune_candidates(
                        conn,
                        where_clause=where_clause,
                        params=tuple(params),
                    )
                    notes_graph_candidates = conn.execute(
                        f"SELECT id FROM jobs{where_clause} AND domain=? AND queue=? "  # nosec B608
                        "AND job_type=?",
                        tuple(
                            params
                            + ["notes", "graph-suggestions", "note_graph_suggestions"]
                        ),
                    ).fetchall()
                    conn.execute(
                        (
                            "UPDATE job_dependencies SET "
                            "depends_on_terminal_status = CASE WHEN ("
                            "SELECT terminal_job.status FROM jobs AS terminal_job "
                            "WHERE terminal_job.uuid = job_dependencies.depends_on_job_uuid) IN "
                            "('completed','failed','cancelled','quarantined') "
                            "THEN (SELECT terminal_job.status FROM jobs AS terminal_job "
                            "WHERE terminal_job.uuid = job_dependencies.depends_on_job_uuid) "
                            "ELSE 'cancelled' END, "
                            "depends_on_cancellation_reason = CASE WHEN ("
                            "SELECT terminal_job.status FROM jobs AS terminal_job "
                            "WHERE terminal_job.uuid = job_dependencies.depends_on_job_uuid) IN "
                            "('completed','failed','cancelled','quarantined') "
                            "THEN (SELECT terminal_job.cancellation_reason FROM jobs AS terminal_job "
                            "WHERE terminal_job.uuid = job_dependencies.depends_on_job_uuid) "
                            "ELSE 'pruned' END "
                            "WHERE depends_on_job_uuid IN ("
                            f"SELECT uuid FROM jobs{where_clause})"  # nosec B608
                        ),
                        tuple(params),
                    )
                    # Receipt-backed jobs always archive before active deletion;
                    # global archive policy still controls all other candidates.
                    if archive_enabled or receipt_candidates or notes_graph_candidates:
                        receipt_exists_clause = (
                            " EXISTS (SELECT 1 FROM job_idempotency_receipts "
                            "AS receipt WHERE receipt.job_uuid = jobs.uuid "
                            "AND receipt.job_id = jobs.id "
                            "AND receipt.domain = jobs.domain "
                            "AND receipt.queue = jobs.queue "
                            "AND receipt.job_type = jobs.job_type "
                            "AND receipt.owner_user_id = jobs.owner_user_id "
                            "AND receipt.operation_scope = jobs.batch_group)"
                        )
                        receipt_where_clause = (
                            where_clause + " AND" + receipt_exists_clause
                        )
                        archive_params = list(params)
                        if archive_enabled:
                            archive_where_clause = where_clause
                        else:
                            archive_where_clause = (
                                where_clause
                                + " AND ("
                                + receipt_exists_clause
                                + " OR (domain=? AND queue=? AND job_type=?))"
                            )
                            archive_params.extend(
                                [
                                    "notes",
                                    "graph-suggestions",
                                    "note_graph_suggestions",
                                ]
                            )
                        prompt_archive_params = tuple(
                            archive_params + ["prompt_studio", "optimization"]
                        )
                        prompt_rows = conn.execute(
                            f"SELECT id, queue, payload FROM jobs{archive_where_clause} "  # nosec B608
                            "AND domain = ? AND job_type = ?",
                            prompt_archive_params,
                        ).fetchall()
                        for prompt_row in prompt_rows:
                            secured_payload = self._secured_prompt_archive_payload(
                                prompt_row[2],
                                queue=str(prompt_row[1]),
                            )
                            if secured_payload is not None:
                                conn.execute(
                                    "UPDATE jobs SET payload = ? WHERE id = ? "
                                    "AND domain = ? AND job_type = ?",
                                    (
                                        secured_payload,
                                        int(prompt_row[0]),
                                        "prompt_studio",
                                        "optimization",
                                    ),
                                )
                        exact_collisions = self._idempotent_slides_archive_collisions(
                            conn,
                            where_clause=archive_where_clause,
                            params=tuple(archive_params),
                        )
                        if receipt_candidates:
                            exact_collisions.update(
                                self._exact_receipt_archive_uuids(
                                    conn,
                                    where_clause=receipt_where_clause,
                                    params=tuple(params),
                                )
                            )
                        if exact_collisions:
                            collision_placeholders = ",".join(
                                ["?"] * len(exact_collisions)
                            )
                            archive_where_clause += (
                                " AND (uuid IS NULL OR "
                                f"uuid NOT IN ({collision_placeholders}))"  # nosec B608
                            )
                            archive_params.extend(sorted(exact_collisions))
                        archive_compress = JobManager._is_truthy(
                            os.getenv("JOBS_ARCHIVE_COMPRESS", "")
                        )
                        archive_returning = (
                            " RETURNING rowid, uuid, domain, queue, job_type, "
                            "payload, result"
                            if archive_compress
                            else ""
                        )
                        archive_insert_sql = (
                            f"INSERT INTO jobs_archive ({archive_projection}) "  # nosec B608
                            f"SELECT {archive_select_projection} FROM jobs{archive_where_clause}"  # nosec B608
                            + archive_returning
                        )
                        archive_insert_cursor = conn.execute(
                            archive_insert_sql,
                            tuple(archive_params),
                        )
                        inserted_archive_rows = (
                            archive_insert_cursor.fetchall() or []
                            if archive_compress
                            else []
                        )
                        # Optional compression for archived payload/result (SQLite: base64-gz prefix)
                        try:
                            if archive_compress:
                                import base64
                                import gzip

                                drop_json = JobManager._is_truthy(os.getenv("JOBS_ARCHIVE_COMPRESS_DROP_JSON", ""))
                                for (
                                    archive_rowid,
                                    job_uuid,
                                    row_domain,
                                    row_queue,
                                    row_type,
                                    pl,
                                    rs,
                                ) in inserted_archive_rows:
                                    try:
                                        if not job_uuid and _is_slides_generation_scope(
                                            row_domain,
                                            row_queue,
                                            row_type,
                                        ):
                                                conn.execute(
                                                    """
                                                    UPDATE slides_standalone_reconciliation
                                                    SET diagnostic_code=CASE
                                                          WHEN diagnostic_code='duplicate_archive_uuid'
                                                          THEN diagnostic_code
                                                          ELSE 'ambiguous_generation_legacy_row' END,
                                                        diagnostic_count=CASE
                                                          WHEN diagnostic_code='duplicate_archive_uuid'
                                                          THEN diagnostic_count
                                                          ELSE MAX(diagnostic_count, 1) END,
                                                        diagnostic_at=CASE
                                                          WHEN diagnostic_code='duplicate_archive_uuid'
                                                          THEN diagnostic_at
                                                          ELSE DATETIME('now') END
                                                    WHERE singleton_id=1
                                                    """
                                                )
                                                continue
                                        p64 = None
                                        r64 = None
                                        if isinstance(pl, str) and pl:
                                            p64 = "gzip64:" + base64.b64encode(
                                                gzip.compress(pl.encode("utf-8"))
                                            ).decode("ascii")
                                        if isinstance(rs, str) and rs:
                                            r64 = "gzip64:" + base64.b64encode(
                                                gzip.compress(rs.encode("utf-8"))
                                            ).decode("ascii")
                                        if drop_json:
                                            conn.execute(
                                                "UPDATE jobs_archive SET payload=NULL, result=NULL, payload_compressed=?, result_compressed=? WHERE rowid=?",
                                                (p64, r64, int(archive_rowid)),
                                            )
                                        else:
                                            conn.execute(
                                                "UPDATE jobs_archive SET payload_compressed=?, result_compressed=? WHERE rowid=?",
                                                (p64, r64, int(archive_rowid)),
                                            )
                                    except _JOB_NONCRITICAL_EXCEPTIONS:
                                        continue
                        except _JOB_NONCRITICAL_EXCEPTIONS:
                            pass
                        if receipt_candidates:
                            archived_receipt_uuids = (
                                self._exact_receipt_archive_uuids(
                                    conn,
                                    where_clause=receipt_where_clause,
                                    params=tuple(params),
                                )
                            )
                            if archived_receipt_uuids != set(
                                receipt_candidates.values()
                            ):
                                raise IdempotentOperationUnavailableError(
                                    "receipt-backed Jobs were not archived exactly once"
                                )
                    # Counters: subtract queued/processing/quarantined rows if they are part of prune set
                    if JobManager._is_truthy(os.getenv("JOBS_COUNTERS_ENABLED", "")):
                        for r in (
                            conn.execute(
                                f"SELECT domain, queue, job_type, COUNT(*) FROM jobs{where_clause} AND status='queued' AND available_at IS NULL GROUP BY domain,queue,job_type",  # nosec B608
                                tuple(params),
                            ).fetchall()
                            or []
                        ):
                            conn.execute(
                                "UPDATE job_counters SET ready_count = CASE WHEN (ready_count - ?) < 0 THEN 0 ELSE ready_count - ? END, updated_at = DATETIME('now') WHERE domain=? AND queue=? AND job_type=?",
                                (int(r[3]), int(r[3]), r[0], r[1], r[2]),
                            )
                        for r in (
                            conn.execute(
                                f"SELECT domain, queue, job_type, COUNT(*) FROM jobs{where_clause} AND status='queued' AND available_at IS NOT NULL GROUP BY domain,queue,job_type",  # nosec B608
                                tuple(params),
                            ).fetchall()
                            or []
                        ):
                            conn.execute(
                                "UPDATE job_counters SET scheduled_count = CASE WHEN (scheduled_count - ?) < 0 THEN 0 ELSE scheduled_count - ? END, updated_at = DATETIME('now') WHERE domain=? AND queue=? AND job_type=?",
                                (int(r[3]), int(r[3]), r[0], r[1], r[2]),
                            )
                        for r in (
                            conn.execute(
                                f"SELECT domain, queue, job_type, COUNT(*) FROM jobs{where_clause} AND status='processing' GROUP BY domain,queue,job_type",  # nosec B608
                                tuple(params),
                            ).fetchall()
                            or []
                        ):
                            conn.execute(
                                "UPDATE job_counters SET processing_count = CASE WHEN (processing_count - ?) < 0 THEN 0 ELSE processing_count - ? END, updated_at = DATETIME('now') WHERE domain=? AND queue=? AND job_type=?",
                                (int(r[3]), int(r[3]), r[0], r[1], r[2]),
                            )
                        for r in (
                            conn.execute(
                                f"SELECT domain, queue, job_type, COUNT(*) FROM jobs{where_clause} AND status='quarantined' GROUP BY domain,queue,job_type",  # nosec B608
                                tuple(params),
                            ).fetchall()
                            or []
                        ):
                            conn.execute(
                                "UPDATE job_counters SET quarantined_count = CASE WHEN (quarantined_count - ?) < 0 THEN 0 ELSE quarantined_count - ? END, updated_at = DATETIME('now') WHERE domain=? AND queue=? AND job_type=?",
                                (int(r[3]), int(r[3]), r[0], r[1], r[2]),
                            )
                    if _test_mode:
                        try:
                            _before2 = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
                        except _JOB_NONCRITICAL_EXCEPTIONS:
                            _before2 = None
                    conn.execute(f"DELETE FROM jobs{where_clause}", tuple(params))  # nosec B608
                    deleted = int(count)
                    if _test_mode:
                        try:
                            _after2 = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
                            logger.info(
                                f"[JM TEST MUT] prune_jobs deleted={int(deleted)} before={_before2} after={_after2}"
                            )
                        except _JOB_NONCRITICAL_EXCEPTIONS:
                            pass
            with contextlib.suppress(_JOB_NONCRITICAL_EXCEPTIONS):
                emit_job_event(
                    "jobs.pruned",
                    job=None,
                    attrs={
                        "deleted": int(deleted),
                        "dry_run": False,
                        "statuses": ",".join(statuses),
                        "older_than_days": int(older_than_days),
                        "domain": domain,
                        "queue": queue,
                        "job_type": job_type,
                    },
                )
            return int(deleted)

        except SlidesGenerationJobsUnavailableError:
            if not slides_diagnostic_persisted:
                with contextlib.suppress(_JOB_NONCRITICAL_EXCEPTIONS):
                    with contextlib.closing(self._connect()) as diagnostic_conn:
                        self._record_slides_generation_diagnostic(
                            diagnostic_conn,
                            code="ambiguous_generation_legacy_row",
                            count=1,
                        )
            raise
        finally:
            conn.close()

    def prune_idempotency_receipts(
        self,
        *,
        now: datetime | None = None,
        limit: int = 1000,
    ) -> int:
        """Delete expired receipts only after one exact terminal archive exists."""

        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 10_000:
            raise ValueError("limit must be an integer between 1 and 10000")
        reference_time = _require_aware_utc(now, field_name="now")
        conn = self._connect()
        try:
            if self.backend == "postgres":
                with conn, self._pg_cursor(conn) as cur:
                    cur.execute(
                        """
                        WITH candidates AS (
                          SELECT receipt.receipt_id
                          FROM job_idempotency_receipts AS receipt
                          WHERE receipt.expires_at <= %s
                            AND NOT EXISTS (
                              SELECT 1 FROM jobs AS active
                              WHERE active.uuid = receipt.job_uuid
                                 OR active.id = receipt.job_id
                            )
                            AND 1 = (
                              SELECT COUNT(*) FROM jobs_archive AS archived
                              WHERE archived.uuid = receipt.job_uuid
                            )
                            AND EXISTS (
                              SELECT 1 FROM jobs_archive AS archived
                              WHERE archived.uuid = receipt.job_uuid
                                AND archived.id = receipt.job_id
                                AND archived.domain = receipt.domain
                                AND archived.queue = receipt.queue
                                AND archived.job_type = receipt.job_type
                                AND archived.owner_user_id = receipt.owner_user_id
                                AND archived.batch_group = receipt.operation_scope
                                AND archived.status IN (
                                  'completed','failed','cancelled','quarantined'
                                )
                            )
                          ORDER BY receipt.expires_at, receipt.receipt_id
                          FOR UPDATE OF receipt SKIP LOCKED
                          LIMIT %s
                        )
                        DELETE FROM job_idempotency_receipts AS receipt
                        USING candidates
                        WHERE receipt.receipt_id = candidates.receipt_id
                        RETURNING receipt.receipt_id
                        """,
                        (reference_time, limit),
                    )
                    return len(cur.fetchall() or [])

            reference_sql = _sqlite_utc(reference_time)
            conn.execute("BEGIN IMMEDIATE")
            with conn:
                rows = conn.execute(
                    """
                    SELECT receipt.receipt_id
                    FROM job_idempotency_receipts AS receipt
                    WHERE julianday(receipt.expires_at) <= julianday(?)
                      AND NOT EXISTS (
                        SELECT 1 FROM jobs AS active
                        WHERE active.uuid = receipt.job_uuid
                           OR active.id = receipt.job_id
                      )
                      AND 1 = (
                        SELECT COUNT(*) FROM jobs_archive AS archived
                        WHERE archived.uuid = receipt.job_uuid
                      )
                      AND EXISTS (
                        SELECT 1 FROM jobs_archive AS archived
                        WHERE archived.uuid = receipt.job_uuid
                          AND archived.id = receipt.job_id
                          AND archived.domain = receipt.domain
                          AND archived.queue = receipt.queue
                          AND archived.job_type = receipt.job_type
                          AND archived.owner_user_id = receipt.owner_user_id
                          AND archived.batch_group = receipt.operation_scope
                          AND archived.status IN (
                            'completed','failed','cancelled','quarantined'
                          )
                      )
                    ORDER BY receipt.expires_at, receipt.receipt_id
                    LIMIT ?
                    """,
                    (reference_sql, limit),
                ).fetchall()
                receipt_ids = [int(row[0]) for row in rows]
                if not receipt_ids:
                    return 0
                placeholders = ",".join("?" for _ in receipt_ids)
                deleted = conn.execute(
                    "DELETE FROM job_idempotency_receipts "
                    f"WHERE receipt_id IN ({placeholders})",  # nosec B608
                    tuple(receipt_ids),
                )
                return int(deleted.rowcount or 0)
        finally:
            conn.close()

    def apply_ttl_policies(
        self,
        *,
        age_seconds: int | None = None,
        runtime_seconds: int | None = None,
        action: str = "cancel",
        domain: str | None = None,
        queue: str | None = None,
        job_type: str | None = None,
        reference_time: datetime | None = None,
    ) -> int:
        """Atomically terminalize jobs selected by age or runtime TTL policies.

        PostgreSQL aggregates changed rows inside the UPDATE statement. SQLite
        snapshots grouped counts under a write lock before running the guarded
        UPDATE. Counters remain in the same transaction, while metrics and the
        sweep event run only after commit.
        """

        if action not in {"cancel", "fail"}:
            raise ValueError("action must be 'cancel' or 'fail'")  # noqa: TRY003
        age_seconds = int(age_seconds) if age_seconds is not None else None
        runtime_seconds = int(runtime_seconds) if runtime_seconds is not None else None
        if age_seconds is None and runtime_seconds is None:
            return 0

        ref_dt = reference_time or self._clock.now_utc()
        if ref_dt.tzinfo is None:
            ref_dt = ref_dt.replace(tzinfo=_tz.utc)
        else:
            ref_dt = ref_dt.astimezone(_tz.utc)

        counters_enabled = JobManager._is_truthy(
            os.getenv("JOBS_COUNTERS_ENABLED", "")
        )
        affected_age = 0
        affected_runtime = 0
        metric_facts: list[tuple[str, tuple[str, str, str], int]] = []

        def _row_value(row: Any, key: str) -> Any:
            if isinstance(row, dict):
                return row.get(key)
            return row[key]

        def _record_group_counts(
            rows: list[Any],
            *,
            reason: str,
            counter_deltas: dict[tuple[str, str, str], list[int]],
            queued: bool,
        ) -> int:
            affected = 0
            for row in rows:
                key = (
                    str(_row_value(row, "domain")),
                    str(_row_value(row, "queue")),
                    str(_row_value(row, "job_type")),
                )
                total = int(_row_value(row, "total_count") or 0)
                ready = int(_row_value(row, "ready_count") or 0)
                scheduled = int(_row_value(row, "scheduled_count") or 0)
                affected += total
                deltas = counter_deltas.setdefault(key, [0, 0, 0])
                if queued:
                    deltas[0] += ready
                    deltas[1] += scheduled
                else:
                    deltas[2] += total
                metric_facts.append((reason, key, total))
            return affected

        conn = self._connect()
        try:
            counter_deltas: dict[tuple[str, str, str], list[int]] = {}
            if self.backend == "postgres":
                with conn, self._pg_cursor(conn) as cur:
                    if age_seconds is not None:
                        where = [
                            "status='queued'",
                            "created_at <= (%s - (%s || ' seconds')::interval)",
                        ]
                        where_params: list[Any] = [ref_dt, age_seconds]
                        for column, value in (
                            ("domain", domain),
                            ("queue", queue),
                            ("job_type", job_type),
                        ):
                            if value is not None:
                                where.append(f"{column} = %s")
                                where_params.append(value)
                        terminal_set = (
                            "status='cancelled', cancelled_at=%s, "
                            "cancellation_reason='ttl_age'"
                            if action == "cancel"
                            else "status='failed', completed_at=%s, "
                            "error_message='ttl_age'"
                        )
                        cur.execute(
                            (
                                f"WITH changed AS (UPDATE jobs SET {terminal_set}, "  # nosec B608
                                "leased_until=NULL, "
                                "worker_id=NULL, lease_id=NULL "
                                f"WHERE {' AND '.join(where)} "  # nosec B608
                                "RETURNING domain, queue, job_type, available_at) "
                                "SELECT domain, queue, job_type, "
                                "COUNT(*) AS total_count, "
                                "COUNT(*) FILTER (WHERE available_at IS NULL) "
                                "AS ready_count, "
                                "COUNT(*) FILTER (WHERE available_at IS NOT NULL) "
                                "AS scheduled_count FROM changed "
                                "GROUP BY domain, queue, job_type"
                            ),
                            (ref_dt, *where_params),
                        )
                        affected_age = _record_group_counts(
                            list(cur.fetchall() or []),
                            reason="ttl_age",
                            counter_deltas=counter_deltas,
                            queued=True,
                        )

                    if runtime_seconds is not None:
                        where = [
                            "status='processing'",
                            "COALESCE(started_at, acquired_at) "
                            "<= (%s - (%s || ' seconds')::interval)",
                        ]
                        where_params = [ref_dt, runtime_seconds]
                        for column, value in (
                            ("domain", domain),
                            ("queue", queue),
                            ("job_type", job_type),
                        ):
                            if value is not None:
                                where.append(f"{column} = %s")
                                where_params.append(value)
                        terminal_set = (
                            "status='cancelled', cancelled_at=%s, "
                            "cancellation_reason='ttl_runtime'"
                            if action == "cancel"
                            else "status='failed', completed_at=%s, "
                            "error_message='ttl_runtime'"
                        )
                        cur.execute(
                            (
                                f"WITH changed AS (UPDATE jobs SET {terminal_set}, "  # nosec B608
                                "leased_until=NULL, "
                                "worker_id=NULL, lease_id=NULL "
                                f"WHERE {' AND '.join(where)} "  # nosec B608
                                "RETURNING domain, queue, job_type, available_at) "
                                "SELECT domain, queue, job_type, "
                                "COUNT(*) AS total_count, "
                                "COUNT(*) FILTER (WHERE available_at IS NULL) "
                                "AS ready_count, "
                                "COUNT(*) FILTER (WHERE available_at IS NOT NULL) "
                                "AS scheduled_count FROM changed "
                                "GROUP BY domain, queue, job_type"
                            ),
                            (ref_dt, *where_params),
                        )
                        affected_runtime = _record_group_counts(
                            list(cur.fetchall() or []),
                            reason="ttl_runtime",
                            counter_deltas=counter_deltas,
                            queued=False,
                        )

                    if counters_enabled:
                        for key, deltas in sorted(counter_deltas.items()):
                            ready, scheduled, processing = deltas
                            cur.execute(
                                (
                                    "UPDATE job_counters SET "
                                    "ready_count=GREATEST(ready_count-%s, 0), "
                                    "scheduled_count=GREATEST(scheduled_count-%s, 0), "
                                    "processing_count=GREATEST(processing_count-%s, 0), "
                                    "updated_at=NOW() "
                                    "WHERE domain=%s AND queue=%s AND job_type=%s"
                                ),
                                (ready, scheduled, processing, *key),
                            )
            else:
                now_str = ref_dt.strftime("%Y-%m-%d %H:%M:%S.%f")
                with conn:
                    conn.execute("BEGIN IMMEDIATE")
                    if age_seconds is not None:
                        where = [
                            "status='queued'",
                            "created_at <= DATETIME(?, ?)",
                        ]
                        where_params = [now_str, f"-{age_seconds} seconds"]
                        for column, value in (
                            ("domain", domain),
                            ("queue", queue),
                            ("job_type", job_type),
                        ):
                            if value is not None:
                                where.append(f"{column} = ?")
                                where_params.append(value)
                        terminal_set = (
                            "status='cancelled', cancelled_at=?, "
                            "cancellation_reason='ttl_age'"
                            if action == "cancel"
                            else "status='failed', completed_at=?, "
                            "error_message='ttl_age'"
                        )
                        age_rows = list(
                            conn.execute(
                                (
                                    "SELECT domain, queue, job_type, "
                                    "COUNT(*) AS total_count, "
                                    "SUM(CASE WHEN available_at IS NULL THEN 1 ELSE 0 END) "
                                    "AS ready_count, "
                                    "SUM(CASE WHEN available_at IS NOT NULL THEN 1 ELSE 0 END) "
                                    "AS scheduled_count FROM jobs "
                                    f"WHERE {' AND '.join(where)} "  # nosec B608
                                    "GROUP BY domain, queue, job_type"
                                ),
                                tuple(where_params),
                            ).fetchall()
                            or []
                        )
                        affected_age = _record_group_counts(
                            age_rows,
                            reason="ttl_age",
                            counter_deltas=counter_deltas,
                            queued=True,
                        )
                        cursor = conn.execute(
                            (
                                f"UPDATE jobs SET {terminal_set}, leased_until=NULL, "  # nosec B608
                                "worker_id=NULL, lease_id=NULL "
                                f"WHERE {' AND '.join(where)}"  # nosec B608
                            ),
                            (now_str, *where_params),
                        )
                        if int(cursor.rowcount or 0) != affected_age:
                            raise RuntimeError(
                                "SQLite age TTL selection changed under write lock"
                            )

                    if runtime_seconds is not None:
                        where = [
                            "status='processing'",
                            "COALESCE(started_at, acquired_at) <= DATETIME(?, ?)",
                        ]
                        where_params = [now_str, f"-{runtime_seconds} seconds"]
                        for column, value in (
                            ("domain", domain),
                            ("queue", queue),
                            ("job_type", job_type),
                        ):
                            if value is not None:
                                where.append(f"{column} = ?")
                                where_params.append(value)
                        terminal_set = (
                            "status='cancelled', cancelled_at=?, "
                            "cancellation_reason='ttl_runtime'"
                            if action == "cancel"
                            else "status='failed', completed_at=?, "
                            "error_message='ttl_runtime'"
                        )
                        runtime_rows = list(
                            conn.execute(
                                (
                                    "SELECT domain, queue, job_type, "
                                    "COUNT(*) AS total_count, "
                                    "SUM(CASE WHEN available_at IS NULL THEN 1 ELSE 0 END) "
                                    "AS ready_count, "
                                    "SUM(CASE WHEN available_at IS NOT NULL THEN 1 ELSE 0 END) "
                                    "AS scheduled_count FROM jobs "
                                    f"WHERE {' AND '.join(where)} "  # nosec B608
                                    "GROUP BY domain, queue, job_type"
                                ),
                                tuple(where_params),
                            ).fetchall()
                            or []
                        )
                        affected_runtime = _record_group_counts(
                            runtime_rows,
                            reason="ttl_runtime",
                            counter_deltas=counter_deltas,
                            queued=False,
                        )
                        cursor = conn.execute(
                            (
                                f"UPDATE jobs SET {terminal_set}, leased_until=NULL, "  # nosec B608
                                "worker_id=NULL, lease_id=NULL "
                                f"WHERE {' AND '.join(where)}"  # nosec B608
                            ),
                            (now_str, *where_params),
                        )
                        if int(cursor.rowcount or 0) != affected_runtime:
                            raise RuntimeError(
                                "SQLite runtime TTL selection changed under write lock"
                            )

                    if counters_enabled:
                        for key, deltas in sorted(counter_deltas.items()):
                            ready, scheduled, processing = deltas
                            conn.execute(
                                (
                                    "UPDATE job_counters SET "
                                    "ready_count=MAX(ready_count-?, 0), "
                                    "scheduled_count=MAX(scheduled_count-?, 0), "
                                    "processing_count=MAX(processing_count-?, 0), "
                                    "updated_at=DATETIME('now') "
                                    "WHERE domain=? AND queue=? AND job_type=?"
                                ),
                                (ready, scheduled, processing, *key),
                            )

            try:
                from tldw_Server_API.app.core.Metrics.metrics_manager import (
                    get_metrics_registry,
                )

                registry = get_metrics_registry()
                if registry:
                    for reason, key, count in metric_facts:
                        labels = {
                            "domain": key[0],
                            "queue": key[1],
                            "job_type": key[2],
                        }
                        if action == "cancel":
                            registry.increment(
                                "jobs.cancelled_total",
                                float(count),
                                labels,
                            )
                        else:
                            registry.increment(
                                "jobs.failures_total",
                                float(count),
                                {**labels, "reason": reason},
                            )
            except _JOB_NONCRITICAL_EXCEPTIONS:
                pass

            affected = affected_age + affected_runtime
            with contextlib.suppress(_JOB_NONCRITICAL_EXCEPTIONS):
                emit_job_event(
                    "jobs.ttl_sweep",
                    job=None,
                    attrs={
                        "affected": affected,
                        "affected_age": affected_age,
                        "affected_runtime": affected_runtime,
                        "action": action,
                        "age_seconds": int(age_seconds or 0),
                        "runtime_seconds": int(runtime_seconds or 0),
                        "domain": domain,
                        "queue": queue,
                        "job_type": job_type,
                    },
                )
            return affected
        finally:
            conn.close()

    def acquire_next_jobs(
        self,
        *,
        domain: str,
        queue: str,
        lease_seconds: int,
        worker_id: str,
        owner_user_id: str | None = None,
        job_type: str | None = None,
        limit: int = 1,
    ) -> list[dict[str, Any]]:
        """Acquire up to `limit` jobs. Simple loop over acquire_next_job for now."""
        limit = max(1, int(limit))
        out: list[dict[str, Any]] = []
        for _ in range(limit):
            j = self.acquire_next_job(
                domain=domain,
                queue=queue,
                lease_seconds=lease_seconds,
                worker_id=worker_id,
                owner_user_id=owner_user_id,
                job_type=job_type,
            )
            if not j:
                break
            out.append(j)
        return out

    # --- Admin reschedule / retry-now helpers ---
    def reschedule_jobs(
        self,
        *,
        domain: str | None = None,
        queue: str | None = None,
        job_type: str | None = None,
        status: str | None = None,
        set_now: bool = True,
        delta_seconds: int | None = None,
        dry_run: bool = False,
    ) -> int:
        """Reschedule jobs by adjusting available_at.

        If set_now is True, clears available_at so matched jobs are ready now.
        Otherwise, adds delta_seconds to current available_at (or sets from now if NULL).
        """
        if status and status not in {"queued", "failed", "processing", "completed", "cancelled", "quarantined"}:
            raise ValueError("Unsupported status filter")  # noqa: TRY003
        conn = self._connect()
        try:
            if self.backend == "postgres":
                with self._pg_cursor(conn) as cur:
                    where = ["1=1"]
                    params: list[Any] = []
                    if domain:
                        where.append("domain=%s")
                        params.append(domain)
                    if queue:
                        where.append("queue=%s")
                        params.append(queue)
                    if job_type:
                        where.append("job_type=%s")
                        params.append(job_type)
                    if status:
                        where.append("status=%s")
                        params.append(status)
                    wh = " AND ".join(where)
                    if dry_run:
                        dry_run_where = f"{wh} AND available_at IS NOT NULL" if set_now else wh
                        cur.execute(
                            f"SELECT COUNT(*) AS c FROM jobs WHERE {dry_run_where}",  # nosec B608
                            tuple(params),
                        )
                        count_row = cur.fetchone()
                        return int(count_row.get("c") if isinstance(count_row, dict) else 0)
                    if set_now:
                        with conn:
                            cur.execute(
                                (
                                    "WITH candidates AS ("  # nosec B608
                                    f"SELECT id FROM jobs WHERE {wh} "
                                    "AND available_at IS NOT NULL FOR UPDATE"
                                    "), updated AS ("
                                    "UPDATE jobs AS target SET available_at=NULL FROM candidates "
                                    "WHERE target.id=candidates.id AND target.available_at IS NOT NULL "
                                    "RETURNING target.domain,target.queue,target.job_type,target.status"
                                    ") SELECT domain,queue,job_type,status,COUNT(*) AS c FROM updated "
                                    "GROUP BY domain,queue,job_type,status"
                                ),
                                tuple(params),
                            )
                            grouped_rows = [dict(row) for row in (cur.fetchall() or [])]
                            count = sum(int(row.get("c") or 0) for row in grouped_rows)
                            if JobManager._is_truthy(os.getenv("JOBS_COUNTERS_ENABLED", "")):
                                for grouped_row in sorted(
                                    grouped_rows,
                                    key=lambda row: (
                                        str(row.get("domain")),
                                        str(row.get("queue")),
                                        str(row.get("job_type")),
                                    ),
                                ):
                                    if grouped_row.get("status") != "queued":
                                        continue
                                    moved = int(grouped_row.get("c") or 0)
                                    cur.execute(
                                        (
                                            "UPDATE job_counters SET "
                                            "scheduled_count=GREATEST(scheduled_count - %s, 0), "
                                            "ready_count=ready_count + %s, updated_at=NOW() "
                                            "WHERE domain=%s AND queue=%s AND job_type=%s"
                                        ),
                                        (
                                            moved,
                                            moved,
                                            grouped_row.get("domain"),
                                            grouped_row.get("queue"),
                                            grouped_row.get("job_type"),
                                        ),
                                    )
                                    if cur.rowcount == 0:
                                        _reconcile_lifecycle_counter_row(
                                            cur,
                                            backend=self.backend,
                                            domain=grouped_row.get("domain"),
                                            queue=grouped_row.get("queue"),
                                            job_type=grouped_row.get("job_type"),
                                        )
                            return count
                    else:
                        if delta_seconds is None:
                            raise ValueError("delta_seconds required when set_now=false")  # noqa: TRY003
                        with conn:
                            cur.execute(
                                (
                                    "WITH candidates AS ("  # nosec B608
                                    f"SELECT id,status,(available_at IS NULL) AS was_ready FROM jobs WHERE {wh} FOR UPDATE"
                                    "), updated AS ("
                                    "UPDATE jobs AS target SET available_at=COALESCE(target.available_at, NOW()) + "
                                    "(%s || ' seconds')::interval FROM candidates "
                                    "WHERE target.id=candidates.id "
                                    "RETURNING target.domain,target.queue,target.job_type,target.status,candidates.was_ready"
                                    ") SELECT domain,queue,job_type,COUNT(*) AS total_count,"
                                    "COUNT(*) FILTER (WHERE status='queued' AND was_ready) AS moved_count "
                                    "FROM updated GROUP BY domain,queue,job_type"
                                ),
                                (*params, int(delta_seconds)),
                            )
                            grouped_rows = [dict(row) for row in (cur.fetchall() or [])]
                            count = sum(int(row.get("total_count") or 0) for row in grouped_rows)
                            if JobManager._is_truthy(os.getenv("JOBS_COUNTERS_ENABLED", "")):
                                for grouped_row in sorted(
                                    grouped_rows,
                                    key=lambda row: (
                                        str(row.get("domain")),
                                        str(row.get("queue")),
                                        str(row.get("job_type")),
                                    ),
                                ):
                                    moved = int(grouped_row.get("moved_count") or 0)
                                    if moved == 0:
                                        continue
                                    cur.execute(
                                        (
                                            "UPDATE job_counters SET "
                                            "ready_count=GREATEST(ready_count - %s, 0), "
                                            "scheduled_count=scheduled_count + %s, updated_at=NOW() "
                                            "WHERE domain=%s AND queue=%s AND job_type=%s"
                                        ),
                                        (
                                            moved,
                                            moved,
                                            grouped_row.get("domain"),
                                            grouped_row.get("queue"),
                                            grouped_row.get("job_type"),
                                        ),
                                    )
                                    if cur.rowcount == 0:
                                        _reconcile_lifecycle_counter_row(
                                            cur,
                                            backend=self.backend,
                                            domain=grouped_row.get("domain"),
                                            queue=grouped_row.get("queue"),
                                            job_type=grouped_row.get("job_type"),
                                        )
                            return count
            else:
                where = ["1=1"]
                params: list[Any] = []
                if domain:
                    where.append("domain=?")
                    params.append(domain)
                if queue:
                    where.append("queue=?")
                    params.append(queue)
                if job_type:
                    where.append("job_type=?")
                    params.append(job_type)
                if status:
                    where.append("status=?")
                    params.append(status)
                wh = " AND ".join(where)
                if dry_run:
                    dry_run_where = f"{wh} AND available_at IS NOT NULL" if set_now else wh
                    row = conn.execute(
                        f"SELECT COUNT(*) FROM jobs WHERE {dry_run_where}",  # nosec B608
                        tuple(params),
                    ).fetchone()
                    return int(row[0]) if row else 0
                with conn:
                    if set_now:
                        conn.execute("BEGIN IMMEDIATE")
                        counter_groups = []
                        if JobManager._is_truthy(os.getenv("JOBS_COUNTERS_ENABLED", "")):
                            counter_groups = list(
                                conn.execute(
                                    (
                                        f"SELECT domain,queue,job_type,COUNT(*) AS moved_count FROM jobs WHERE {wh} "  # nosec B608
                                        "AND status='queued' AND available_at IS NOT NULL "
                                        "GROUP BY domain,queue,job_type"
                                    ),
                                    tuple(params),
                                ).fetchall()
                                or []
                            )
                        changed = conn.execute(
                            f"UPDATE jobs SET available_at=NULL WHERE {wh} AND available_at IS NOT NULL",  # nosec B608
                            tuple(params),
                        )
                        count = int(changed.rowcount)
                        for counter_group in counter_groups:
                            moved = int(counter_group[3] or 0)
                            counter_cursor = conn.execute(
                                (
                                    "UPDATE job_counters SET "
                                    "scheduled_count=MAX(scheduled_count - ?, 0), "
                                    "ready_count=ready_count + ?, updated_at=DATETIME('now') "
                                    "WHERE domain=? AND queue=? AND job_type=?"
                                ),
                                (
                                    moved,
                                    moved,
                                    counter_group[0],
                                    counter_group[1],
                                    counter_group[2],
                                ),
                            )
                            if (counter_cursor.rowcount or 0) == 0:
                                _reconcile_lifecycle_counter_row(
                                    conn,
                                    backend=self.backend,
                                    domain=counter_group[0],
                                    queue=counter_group[1],
                                    job_type=counter_group[2],
                                )
                        return count
                    else:
                        if delta_seconds is None:
                            raise ValueError("delta_seconds required when set_now=false")  # noqa: TRY003
                        conn.execute("BEGIN IMMEDIATE")
                        counter_groups = []
                        if JobManager._is_truthy(os.getenv("JOBS_COUNTERS_ENABLED", "")):
                            counter_groups = list(
                                conn.execute(
                                    (
                                        f"SELECT domain,queue,job_type,COUNT(*) AS moved_count FROM jobs WHERE {wh} "  # nosec B608
                                        "AND status='queued' AND available_at IS NULL "
                                        "GROUP BY domain,queue,job_type"
                                    ),
                                    tuple(params),
                                ).fetchall()
                                or []
                            )
                        changed = conn.execute(
                            f"UPDATE jobs SET available_at=DATETIME(COALESCE(available_at, DATETIME('now')), ?) WHERE {wh}",  # nosec B608
                            (f"{int(delta_seconds):+d} seconds", *params),
                        )
                        count = int(changed.rowcount)
                        for counter_group in counter_groups:
                            moved = int(counter_group[3] or 0)
                            counter_cursor = conn.execute(
                                (
                                    "UPDATE job_counters SET "
                                    "ready_count=MAX(ready_count - ?, 0), "
                                    "scheduled_count=scheduled_count + ?, updated_at=DATETIME('now') "
                                    "WHERE domain=? AND queue=? AND job_type=?"
                                ),
                                (
                                    moved,
                                    moved,
                                    counter_group[0],
                                    counter_group[1],
                                    counter_group[2],
                                )
                            )
                            if (counter_cursor.rowcount or 0) == 0:
                                _reconcile_lifecycle_counter_row(
                                    conn,
                                    backend=self.backend,
                                    domain=counter_group[0],
                                    queue=counter_group[1],
                                    job_type=counter_group[2],
                                )
                return count
        finally:
            conn.close()

    def retry_now_jobs(
        self,
        *,
        job_id: int | None = None,
        domain: str | None = None,
        queue: str | None = None,
        job_type: str | None = None,
        only_failed: bool = True,
        dry_run: bool = False,
    ) -> int:
        """Force immediate retry by moving eligible jobs to queued with available_at=NULL.

        By default targets failed jobs with retries remaining. If only_failed is False,
        also adjusts queued scheduled jobs by clearing available_at.
        """
        conn = self._connect()
        try:
            if self.backend == "postgres":
                where = ["1=1"]
                params: list[Any] = []
                if domain:
                    where.append("domain=%s")
                    params.append(domain)
                if queue:
                    where.append("queue=%s")
                    params.append(queue)
                if job_type:
                    where.append("job_type=%s")
                    params.append(job_type)
                if job_id is not None:
                    where.append("id=%s")
                    params.append(int(job_id))
                wh = " AND ".join(where)
                if dry_run:
                    with self._pg_cursor(conn) as cur:
                        cur.execute(
                            (
                                f"SELECT COUNT(*) AS c FROM jobs WHERE {wh} AND ("  # nosec B608
                                "(status='failed' AND retry_count < max_retries) "
                                + (
                                    " OR (status='queued' AND available_at IS NOT NULL)"
                                    if not only_failed
                                    else ""
                                )
                                + ")"
                            ),
                            tuple(params),
                        )
                        count_row = cur.fetchone()
                        return int(
                            count_row.get("c")
                            if isinstance(count_row, dict)
                            else 0
                        )

                count = 0
                counters_enabled = JobManager._is_truthy(
                    os.getenv("JOBS_COUNTERS_ENABLED", "")
                )
                with conn:  # noqa: SIM117
                    with self._pg_cursor(conn) as cur:
                        cur.execute(
                            (
                                "WITH changed AS (UPDATE jobs SET status='queued', "
                                "available_at=NULL, result=NULL, completed_at=NULL, "
                                "started_at=NULL, acquired_at=NULL, last_error=NULL, "
                                "error_message=NULL, error_code=NULL, error_class=NULL, "
                                "error_stack=NULL, completion_token=NULL "
                                f"WHERE {wh} AND status='failed' "  # nosec B608
                                "AND retry_count < max_retries "
                                "RETURNING domain,queue,job_type) "
                                "SELECT domain,queue,job_type,COUNT(*) AS c FROM changed "
                                "GROUP BY domain,queue,job_type"
                            ),
                            tuple(params),
                        )
                        failed_groups = list(cur.fetchall() or [])
                        count += sum(int(row.get("c") or 0) for row in failed_groups)
                        if counters_enabled:
                            for row in failed_groups:
                                cur.execute(
                                    (
                                        "UPDATE job_counters SET ready_count=ready_count + %s, "
                                        "updated_at=NOW() "
                                        "WHERE domain=%s AND queue=%s AND job_type=%s"
                                    ),
                                    (
                                        int(row["c"]),
                                        row["domain"],
                                        row["queue"],
                                        row["job_type"],
                                    ),
                                )
                                if cur.rowcount == 0:
                                    _reconcile_lifecycle_counter_row(
                                        cur,
                                        backend=self.backend,
                                        domain=row["domain"],
                                        queue=row["queue"],
                                        job_type=row["job_type"],
                                    )

                        if not only_failed:
                            cur.execute(
                                (
                                    "WITH changed AS (UPDATE jobs SET available_at=NULL, "
                                    "completion_token=NULL "
                                    f"WHERE {wh} AND status='queued' "  # nosec B608
                                    "AND available_at IS NOT NULL "
                                    "RETURNING domain,queue,job_type) "
                                    "SELECT domain,queue,job_type,COUNT(*) AS c FROM changed "
                                    "GROUP BY domain,queue,job_type"
                                ),
                                tuple(params),
                            )
                            scheduled_groups = list(cur.fetchall() or [])
                            count += sum(
                                int(row.get("c") or 0)
                                for row in scheduled_groups
                            )
                            if counters_enabled:
                                for row in scheduled_groups:
                                    moved = int(row["c"])
                                    cur.execute(
                                        (
                                            "UPDATE job_counters SET "
                                            "scheduled_count=GREATEST(scheduled_count - %s, 0), "
                                            "ready_count=ready_count + %s, updated_at=NOW() "
                                            "WHERE domain=%s AND queue=%s AND job_type=%s"
                                        ),
                                        (
                                            moved,
                                            moved,
                                            row["domain"],
                                            row["queue"],
                                            row["job_type"],
                                        ),
                                    )
                                    if cur.rowcount == 0:
                                        _reconcile_lifecycle_counter_row(
                                            cur,
                                            backend=self.backend,
                                            domain=row["domain"],
                                            queue=row["queue"],
                                            job_type=row["job_type"],
                                        )
                return count
            else:
                where = ["1=1"]
                params: list[Any] = []
                if domain:
                    where.append("domain=?")
                    params.append(domain)
                if queue:
                    where.append("queue=?")
                    params.append(queue)
                if job_type:
                    where.append("job_type=?")
                    params.append(job_type)
                if job_id is not None:
                    where.append("id=?")
                    params.append(int(job_id))
                wh = " AND ".join(where)
                if dry_run:
                    row = conn.execute(
                        (
                            f"SELECT COUNT(*) FROM jobs WHERE {wh} AND ("  # nosec B608
                            "(status='failed' AND retry_count < max_retries) "
                            + (
                                " OR (status='queued' AND available_at IS NOT NULL)"
                                if not only_failed
                                else ""
                            )
                            + ")"
                        ),
                        tuple(params),
                    ).fetchone()
                    return int(row[0]) if row else 0

                count = 0
                counters_enabled = JobManager._is_truthy(
                    os.getenv("JOBS_COUNTERS_ENABLED", "")
                )
                with conn:
                    conn.execute("BEGIN IMMEDIATE")
                    failed_groups = []
                    if counters_enabled:
                        failed_groups = list(
                            conn.execute(
                                (
                                    f"SELECT domain,queue,job_type,COUNT(*) FROM jobs WHERE {wh} "  # nosec B608
                                    "AND status='failed' AND retry_count < max_retries "
                                    "GROUP BY domain,queue,job_type"
                                ),
                                tuple(params),
                            ).fetchall()
                            or []
                        )
                    changed = conn.execute(
                        (
                            "UPDATE jobs SET status='queued', available_at=NULL, "
                            "result=NULL, completed_at=NULL, started_at=NULL, "
                            "acquired_at=NULL, last_error=NULL, error_message=NULL, "
                            "error_code=NULL, error_class=NULL, error_stack=NULL, "
                            f"completion_token=NULL WHERE {wh} AND status='failed' "  # nosec B608
                            "AND retry_count < max_retries"
                        ),
                        tuple(params),
                    )
                    count += int(changed.rowcount or 0)
                    if counters_enabled:
                        for row in failed_groups:
                            counter_cursor = conn.execute(
                                (
                                    "UPDATE job_counters SET ready_count=ready_count + ?, "
                                    "updated_at=DATETIME('now') "
                                    "WHERE domain=? AND queue=? AND job_type=?"
                                ),
                                (int(row[3]), row[0], row[1], row[2]),
                            )
                            if (counter_cursor.rowcount or 0) == 0:
                                _reconcile_lifecycle_counter_row(
                                    conn,
                                    backend=self.backend,
                                    domain=row[0],
                                    queue=row[1],
                                    job_type=row[2],
                                )
                    if not only_failed:
                        scheduled_groups = []
                        if counters_enabled:
                            scheduled_groups = list(
                                conn.execute(
                                    (
                                        f"SELECT domain,queue,job_type,COUNT(*) FROM jobs WHERE {wh} "  # nosec B608
                                        "AND status='queued' AND available_at IS NOT NULL "
                                        "GROUP BY domain,queue,job_type"
                                    ),
                                    tuple(params),
                                ).fetchall()
                                or []
                            )
                        changed = conn.execute(
                            f"UPDATE jobs SET available_at=NULL, completion_token=NULL WHERE {wh} AND status='queued' AND available_at IS NOT NULL",  # nosec B608
                            tuple(params),
                        )
                        count += int(changed.rowcount or 0)
                        if counters_enabled:
                            for row in scheduled_groups:
                                moved = int(row[3])
                                counter_cursor = conn.execute(
                                    (
                                        "UPDATE job_counters SET "
                                        "scheduled_count=MAX(scheduled_count - ?, 0), "
                                        "ready_count=ready_count + ?, updated_at=DATETIME('now') "
                                        "WHERE domain=? AND queue=? AND job_type=?"
                                    ),
                                    (moved, moved, row[0], row[1], row[2]),
                                )
                                if (counter_cursor.rowcount or 0) == 0:
                                    _reconcile_lifecycle_counter_row(
                                        conn,
                                        backend=self.backend,
                                        domain=row[0],
                                        queue=row[1],
                                        job_type=row[2],
                                    )
                return count
        finally:
            conn.close()

    def get_queue_stats(
        self,
        *,
        domain: str | None = None,
        queue: str | None = None,
        job_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return counts grouped by domain/queue/job_type.

        Provides queued (ready), scheduled, and processing counts per group.
        """
        conn = self._connect()
        try:
            if self.backend == "postgres":
                where = ["1=1"]
                params: list[Any] = []
                if domain:
                    where.append("domain = %s")
                    params.append(domain)
                if queue:
                    where.append("queue = %s")
                    params.append(queue)
                if job_type:
                    where.append("job_type = %s")
                    params.append(job_type)
                sql = (
                    "SELECT domain, queue, job_type, "  # nosec B608
                    "SUM(CASE WHEN status='queued' AND available_at IS NULL THEN 1 ELSE 0 END) AS queued, "
                    "SUM(CASE WHEN status='queued' AND available_at IS NOT NULL THEN 1 ELSE 0 END) AS scheduled, "
                    "SUM(CASE WHEN status='processing' THEN 1 ELSE 0 END) AS processing, "
                    "SUM(CASE WHEN status='quarantined' THEN 1 ELSE 0 END) AS quarantined "
                    f"FROM jobs WHERE {' AND '.join(where)} GROUP BY domain, queue, job_type ORDER BY domain, queue, job_type"
                )
                with self._pg_cursor(conn) as cur:
                    cur.execute(sql, params)
                    rows = cur.fetchall()
                return [
                    {
                        "domain": r["domain"],
                        "queue": r["queue"],
                        "job_type": r["job_type"],
                        "queued": int((r.get("queued") if isinstance(r, dict) else 0) or 0),
                        "scheduled": int((r.get("scheduled") if isinstance(r, dict) else 0) or 0),
                        "processing": int((r.get("processing") if isinstance(r, dict) else 0) or 0),
                        "quarantined": int((r.get("quarantined") if isinstance(r, dict) else 0) or 0),
                    }
                    for r in rows
                ]
            else:
                where = ["1=1"]
                params2: list[Any] = []
                if domain:
                    where.append("domain = ?")
                    params2.append(domain)
                if queue:
                    where.append("queue = ?")
                    params2.append(queue)
                if job_type:
                    where.append("job_type = ?")
                    params2.append(job_type)
                sql = (
                    "SELECT domain, queue, job_type, "  # nosec B608
                    "SUM(CASE WHEN status='queued' AND available_at IS NULL THEN 1 ELSE 0 END) AS queued, "
                    "SUM(CASE WHEN status='queued' AND available_at IS NOT NULL THEN 1 ELSE 0 END) AS scheduled, "
                    "SUM(CASE WHEN status='processing' THEN 1 ELSE 0 END) AS processing, "
                    "SUM(CASE WHEN status='quarantined' THEN 1 ELSE 0 END) AS quarantined "
                    f"FROM jobs WHERE {' AND '.join(where)} GROUP BY domain, queue, job_type ORDER BY domain, queue, job_type"
                )
                rows = conn.execute(sql, params2).fetchall()
                return [
                    {
                        "domain": r[0],
                        "queue": r[1],
                        "job_type": r[2],
                        "queued": int(r[3] or 0),
                        "scheduled": int(r[4] or 0),
                        "processing": int(r[5] or 0),
                        "quarantined": int(r[6] or 0),
                    }
                    for r in rows
                ]

        finally:
            with contextlib.suppress(_JOB_NONCRITICAL_EXCEPTIONS):
                conn.close()

    def count_active_processing(self, *, domain: str | None = None, queue: str | None = None) -> int:
        """Count jobs currently in processing state (optionally filtered)."""
        conn = self._connect()
        try:
            if self.backend == "postgres":
                where = ["status='processing'"]
                params: list[Any] = []
                if domain:
                    where.append("domain = %s")
                    params.append(domain)
                if queue:
                    where.append("queue = %s")
                    params.append(queue)
                with self._pg_cursor(conn) as cur:
                    cur.execute(f"SELECT COUNT(*) AS c FROM jobs WHERE {' AND '.join(where)}", tuple(params))  # nosec B608
                    row = cur.fetchone()
                    return int(row["c"]) if row is not None else 0
            else:
                where = ["status='processing'"]
                params2: list[Any] = []
                if domain:
                    where.append("domain = ?")
                    params2.append(domain)
                if queue:
                    where.append("queue = ?")
                    params2.append(queue)
                row = conn.execute(f"SELECT COUNT(*) FROM jobs WHERE {' AND '.join(where)}", tuple(params2)).fetchone()  # nosec B608
                return int(row[0]) if row else 0
        finally:
            with contextlib.suppress(_JOB_NONCRITICAL_EXCEPTIONS):
                conn.close()

    def add_job_attachment(
        self, job_id: int, *, kind: str, content_text: str | None = None, url: str | None = None
    ) -> int:
        kind = str(kind or "").strip().lower()
        if kind not in {"log", "artifact", "tag"}:
            raise ValueError("kind must be one of: log, artifact, tag")  # noqa: TRY003
        if not content_text and not url:
            raise ValueError("content_text or url is required")  # noqa: TRY003
        conn = self._connect()
        try:
            if self.backend == "postgres":
                with conn, self._pg_cursor(conn) as cur:
                    cur.execute(
                        "INSERT INTO job_attachments(job_id,kind,content_text,url) VALUES(%s,%s,%s,%s) RETURNING id",
                        (int(job_id), kind, content_text, url),
                    )
                    row = cur.fetchone()
                    return int(row["id"]) if row else 0
            else:
                with conn:
                    conn.execute(
                        "INSERT INTO job_attachments(job_id,kind,content_text,url) VALUES(?,?,?,?)",
                        (int(job_id), kind, content_text, url),
                    )
                    rid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
                    return int(rid)
        finally:
            conn.close()

    def count_processing_for_owner(self, *, domain: str, owner_user_id: str) -> int:
        """Count processing jobs with active leases for a specific owner."""
        if not owner_user_id:
            return 0
        conn = self._connect()
        try:
            if self.backend == "postgres":
                with self._pg_cursor(conn) as cur:
                    cur.execute(
                        "SELECT COUNT(*) AS c FROM jobs WHERE domain=%s AND owner_user_id=%s AND status='processing' AND leased_until IS NOT NULL AND leased_until > NOW()",
                        (domain, owner_user_id),
                    )
                    row = cur.fetchone()
                    return int((row.get("c") if isinstance(row, dict) else 0) or 0)
            row = conn.execute(
                "SELECT COUNT(*) FROM jobs WHERE domain=? AND owner_user_id=? AND status='processing' AND leased_until IS NOT NULL AND leased_until > DATETIME('now')",
                (domain, owner_user_id),
            ).fetchone()
            return int(row[0] or 0)
        except _JOB_NONCRITICAL_EXCEPTIONS:
            return 0
        finally:
            conn.close()

    def list_job_attachments(self, job_id: int, *, limit: int = 100) -> list[dict[str, Any]]:
        limit = max(1, min(1000, int(limit)))
        conn = self._connect()
        try:
            if self.backend == "postgres":
                with self._pg_cursor(conn) as cur:
                    cur.execute(
                        "SELECT id, kind, content_text, url, created_at FROM job_attachments WHERE job_id = %s ORDER BY id ASC LIMIT %s",
                        (int(job_id), limit),
                    )
                    rows = cur.fetchall() or []
                    return [dict(r) for r in rows]
            else:
                rows = (
                    conn.execute(
                        "SELECT id, kind, content_text, url, created_at FROM job_attachments WHERE job_id = ? ORDER BY id ASC LIMIT ?",
                        (int(job_id), limit),
                    ).fetchall()
                    or []
                )
                return [
                    {"id": int(r[0]), "kind": r[1], "content_text": r[2], "url": r[3], "created_at": r[4]} for r in rows
                ]
        finally:
            conn.close()

    def rotate_encryption_keys(
        self,
        *,
        domain: str | None = None,
        queue: str | None = None,
        job_type: str | None = None,
        old_key_b64: str,
        new_key_b64: str,
        fields: list[str],
        limit: int = 1000,
        dry_run: bool = False,
    ) -> int:
        """Re-encrypt encrypted JSON envelopes from old key to new key for selected rows.

        Fields may include 'payload' and/or 'result'. Returns affected row count.
        """
        fields = [f for f in (fields or []) if f in {"payload", "result"}]
        if not fields:
            raise ValueError("fields must include at least one of: payload, result")  # noqa: TRY003
        if not old_key_b64 or not new_key_b64:
            raise ValueError("old_key_b64 and new_key_b64 are required")  # noqa: TRY003
        affected = 0
        conn = self._connect()
        try:
            if self.backend == "postgres":
                with self._pg_cursor(conn) as cur:
                    where = ["1=1"]
                    params: list[Any] = []
                    if domain:
                        where.append("domain=%s")
                        params.append(domain)
                    if queue:
                        where.append("queue=%s")
                        params.append(queue)
                    if job_type:
                        where.append("job_type=%s")
                        params.append(job_type)
                    cur.execute(
                        f"SELECT id, payload, result, domain, queue, job_type FROM jobs WHERE {' AND '.join(where)} ORDER BY id ASC LIMIT %s",  # nosec B608
                        tuple(params + [int(limit)]),
                    )
                    rows = cur.fetchall() or []
                    if dry_run:
                        # Count candidates that would be re-encrypted
                        for r in rows:
                            for fld in fields:
                                val = r.get(fld)
                                env = val if isinstance(val, dict) else None
                                if env and (env.get("_enc") == "aesgcm:v1" or isinstance(env.get("_encrypted"), dict)):
                                    affected += 1
                                    break
                        return affected
                    with conn:
                        for r in rows:
                            upd = {}
                            for fld in fields:
                                val = r.get(fld)
                                obj = None
                                if isinstance(val, dict) and val.get("_enc") == "aesgcm:v1":
                                    obj = decrypt_json_blob_with_key(val, old_key_b64)
                                elif isinstance(val, dict) and isinstance(val.get("_encrypted"), dict):
                                    obj = decrypt_json_blob_with_key(val.get("_encrypted"), old_key_b64)
                                if obj is not None:
                                    env = encrypt_json_blob_with_key(obj, new_key_b64)
                                    if env:
                                        upd[fld] = {"_encrypted": env}
                            if upd:
                                sets = []
                                params_upd: list[Any] = []
                                for k, v in upd.items():
                                    sets.append(f"{k}=%s::jsonb")
                                    params_upd.append(json.dumps(v))
                                params_upd.append(int(r["id"]))
                                cur.execute(f"UPDATE jobs SET {', '.join(sets)} WHERE id = %s", tuple(params_upd))  # nosec B608
                                affected += 1
                return affected
            else:
                where = ["1=1"]
                params2: list[Any] = []
                if domain:
                    where.append("domain=?")
                    params2.append(domain)
                if queue:
                    where.append("queue=?")
                    params2.append(queue)
                if job_type:
                    where.append("job_type=?")
                    params2.append(job_type)
                sql = f"SELECT id, payload, result, domain, queue, job_type FROM jobs WHERE {' AND '.join(where)} ORDER BY id ASC LIMIT ?"  # nosec B608
                rows = conn.execute(sql, tuple(params2 + [int(limit)])).fetchall() or []
                if dry_run:
                    for _rid, pl, rs, *_ in rows:
                        for fld, val in (("payload", pl), ("result", rs)):
                            if fld not in fields:
                                continue
                            try:
                                if isinstance(val, str) and val:
                                    obj = json.loads(val)
                                elif isinstance(val, dict):
                                    obj = val
                                else:
                                    obj = None
                            except _JOB_NONCRITICAL_EXCEPTIONS:
                                obj = None
                            if isinstance(obj, dict) and (
                                obj.get("_enc") == "aesgcm:v1" or isinstance(obj.get("_encrypted"), dict)
                            ):
                                affected += 1
                                break
                    return affected
                with conn:
                    for rid, pl, rs, *_ in rows:
                        upd: dict[str, Any] = {}
                        for fld, val in (("payload", pl), ("result", rs)):
                            if fld not in fields:
                                continue
                            obj = None
                            try:
                                if isinstance(val, str) and val:
                                    val_obj = json.loads(val)
                                elif isinstance(val, dict):
                                    val_obj = val
                                else:
                                    val_obj = None
                            except _JOB_NONCRITICAL_EXCEPTIONS:
                                val_obj = None
                            if isinstance(val_obj, dict) and val_obj.get("_enc") == "aesgcm:v1":
                                obj = decrypt_json_blob_with_key(val_obj, old_key_b64)
                            elif isinstance(val_obj, dict) and isinstance(val_obj.get("_encrypted"), dict):
                                obj = decrypt_json_blob_with_key(val_obj.get("_encrypted"), old_key_b64)
                            if obj is not None:
                                env = encrypt_json_blob_with_key(obj, new_key_b64)
                                if env:
                                    upd[fld] = json.dumps({"_encrypted": env})
                        if upd:
                            sets = []
                            params_upd: list[Any] = []
                            for k, v in upd.items():
                                sets.append(f"{k} = ?")
                                params_upd.append(v)
                            params_upd.append(int(rid))
                            conn.execute(f"UPDATE jobs SET {', '.join(sets)} WHERE id = ?", tuple(params_upd))  # nosec B608
                            affected += 1
                return affected
        finally:
            with contextlib.suppress(_JOB_NONCRITICAL_EXCEPTIONS):
                conn.close()

    def finalize_cancelled(
        self,
        job_id: int,
        *,
        expected_uuid: str,
        reason: str | None = None,
        worker_id: str | None = None,
        lease_id: str | None = None,
        allow_queued: bool = False,
    ) -> bool:
        """Terminally cancel only the exact job incarnation owned by the caller.

        Processing jobs require the acquired UUID, worker ID, and lease ID. A
        queued job may be finalized only when ``allow_queued`` is explicit and
        its UUID matches. These guards prevent a stale worker from cancelling a
        reassigned job or a new job whose numeric ID was reused.
        """
        expected_uuid = str(expected_uuid or "").strip()
        worker_id = str(worker_id or "").strip() or None
        lease_id = str(lease_id or "").strip() or None
        if not expected_uuid:
            return False
        conn = self._connect()
        outbox_enabled = JobManager._is_truthy(os.getenv("JOBS_EVENTS_OUTBOX", ""))
        counters_enabled = JobManager._is_truthy(os.getenv("JOBS_COUNTERS_ENABLED", ""))
        event_job: dict[str, Any] | None = None
        event_attrs = {"reason": reason, "terminal": True}
        try:
            if self.backend == "postgres":
                with conn:  # noqa: SIM117
                    with self._pg_cursor(conn) as cur:
                        cur.execute(
                            "SELECT status, domain, queue, job_type, available_at, uuid, "
                            "owner_user_id, request_id, trace_id, worker_id, lease_id "
                            "FROM jobs WHERE id = %s FOR UPDATE",
                            (int(job_id),),
                        )
                        row = cur.fetchone()
                        if not row:
                            return False
                        if str(row.get("uuid") or "") != expected_uuid:
                            return False
                        state = str(row.get("status") or "")
                        if state == "cancelled":
                            return True
                        if state in {"completed", "failed", "quarantined"}:
                            return False
                        if state not in {"queued", "processing"}:
                            return False
                        if state == "queued":
                            if not allow_queued:
                                return False
                            update_where = "id = %s AND uuid = %s AND status = 'queued'"
                            update_params: tuple[Any, ...] = (
                                reason,
                                int(job_id),
                                expected_uuid,
                            )
                        else:
                            if not worker_id or not lease_id:
                                return False
                            if (
                                str(row.get("worker_id") or "") != worker_id
                                or str(row.get("lease_id") or "") != lease_id
                            ):
                                return False
                            update_where = (
                                "id = %s AND uuid = %s AND status = 'processing' "
                                "AND worker_id = %s AND lease_id = %s"
                            )
                            update_params = (
                                reason,
                                int(job_id),
                                expected_uuid,
                                worker_id,
                                lease_id,
                            )
                        # update_where contains only fixed predicates; values remain bound.
                        cur.execute(
                            (
                                "UPDATE jobs SET status = 'cancelled', cancelled_at = NOW(), cancellation_reason = %s, "
                                "leased_until = NULL, worker_id = NULL, lease_id = NULL WHERE "
                                + update_where  # nosec B608
                            ),
                            update_params,
                        )
                        changed = cur.rowcount > 0
                        if not changed:
                            cur.execute(
                                "SELECT status FROM jobs WHERE id = %s AND uuid = %s",
                                (int(job_id), expected_uuid),
                            )
                            row_chk = cur.fetchone()
                            return bool(row_chk and str(row_chk.get("status") or "") == "cancelled")
                        if counters_enabled:
                            if state == "processing":
                                cur.execute(
                                    (
                                        "UPDATE job_counters SET processing_count = GREATEST(processing_count - 1, 0), "
                                        "updated_at = NOW() WHERE domain=%s AND queue=%s AND job_type=%s"
                                    ),
                                    (row.get("domain"), row.get("queue"), row.get("job_type")),
                                )
                            else:
                                is_sched = row.get("available_at") is not None
                                add_ready = -1 if not is_sched else 0
                                add_sched = -1 if is_sched else 0
                                cur.execute(
                                    (
                                        "UPDATE job_counters SET "
                                        "ready_count = GREATEST(ready_count + %s, 0), "
                                        "scheduled_count = GREATEST(scheduled_count + %s, 0), "
                                        "updated_at = NOW() WHERE domain=%s AND queue=%s AND job_type=%s"
                                    ),
                                    (
                                        int(add_ready),
                                        int(add_sched),
                                        row.get("domain"),
                                        row.get("queue"),
                                        row.get("job_type"),
                                    ),
                                )
                        event_job = {
                            "id": int(job_id),
                            "uuid": row.get("uuid"),
                            "domain": row.get("domain"),
                            "queue": row.get("queue"),
                            "job_type": row.get("job_type"),
                            "owner_user_id": row.get("owner_user_id"),
                            "request_id": row.get("request_id"),
                            "trace_id": row.get("trace_id"),
                        }
                        if outbox_enabled:
                            cur.execute(
                                (
                                    "INSERT INTO job_events(job_id,domain,queue,job_type,event_type,attrs_json,"
                                    "owner_user_id,request_id,trace_id,created_at) "
                                    "VALUES(%s,%s,%s,%s,'job.cancelled',%s::jsonb,%s,%s,%s,NOW())"
                                ),
                                (
                                    event_job["id"],
                                    event_job["domain"],
                                    event_job["queue"],
                                    event_job["job_type"],
                                    json.dumps(event_attrs),
                                    event_job["owner_user_id"],
                                    event_job["request_id"],
                                    event_job["trace_id"],
                                ),
                            )
            else:
                with conn:
                    conn.execute("BEGIN IMMEDIATE")
                    row = conn.execute(
                        "SELECT status, domain, queue, job_type, available_at, uuid, "
                        "owner_user_id, request_id, trace_id, worker_id, lease_id "
                        "FROM jobs WHERE id = ?",
                        (job_id,),
                    ).fetchone()
                    if not row:
                        return False
                    if str(row["uuid"] or "") != expected_uuid:
                        return False
                    state = str(row["status"] or "")
                    if state == "cancelled":
                        return True
                    if state in {"completed", "failed", "quarantined"}:
                        return False
                    if state not in {"queued", "processing"}:
                        return False
                    if state == "queued":
                        if not allow_queued:
                            return False
                        update_where = "id = ? AND uuid = ? AND status = 'queued'"
                        update_params = (reason, job_id, expected_uuid)
                    else:
                        if not worker_id or not lease_id:
                            return False
                        if (
                            str(row["worker_id"] or "") != worker_id
                            or str(row["lease_id"] or "") != lease_id
                        ):
                            return False
                        update_where = (
                            "id = ? AND uuid = ? AND status = 'processing' "
                            "AND worker_id = ? AND lease_id = ?"
                        )
                        update_params = (
                            reason,
                            job_id,
                            expected_uuid,
                            worker_id,
                            lease_id,
                        )
                    # update_where contains only fixed predicates; values remain bound.
                    cur = conn.execute(
                        (
                            "UPDATE jobs SET status = 'cancelled', cancelled_at = DATETIME('now'), cancellation_reason = ?, "
                            "leased_until = NULL, worker_id = NULL, lease_id = NULL WHERE "
                            + update_where  # nosec B608
                        ),
                        update_params,
                    )
                    changed = (cur.rowcount or 0) > 0
                    if not changed:
                        row_chk = conn.execute(
                            "SELECT status FROM jobs WHERE id = ? AND uuid = ?",
                            (job_id, expected_uuid),
                        ).fetchone()
                        return bool(row_chk and str(row_chk["status"] or "") == "cancelled")
                    if counters_enabled:
                        if state == "processing":
                            conn.execute(
                                (
                                    "UPDATE job_counters SET processing_count = CASE WHEN processing_count>0 "
                                    "THEN processing_count-1 ELSE 0 END, updated_at = DATETIME('now') "
                                    "WHERE domain=? AND queue=? AND job_type=?"
                                ),
                                (row["domain"], row["queue"], row["job_type"]),
                            )
                        else:
                            is_sched = row["available_at"] is not None
                            add_ready = -1 if not is_sched else 0
                            add_sched = -1 if is_sched else 0
                            conn.execute(
                                (
                                    "UPDATE job_counters SET "
                                    "ready_count = CASE WHEN (ready_count + ?) < 0 THEN 0 ELSE ready_count + ? END, "
                                    "scheduled_count = CASE WHEN (scheduled_count + ?) < 0 THEN 0 ELSE scheduled_count + ? END, "
                                    "updated_at = DATETIME('now') WHERE domain=? AND queue=? AND job_type=?"
                                ),
                                (
                                    int(add_ready),
                                    int(add_ready),
                                    int(add_sched),
                                    int(add_sched),
                                    row["domain"],
                                    row["queue"],
                                    row["job_type"],
                                ),
                            )
                    event_job = {
                        "id": int(job_id),
                        "uuid": row["uuid"],
                        "domain": row["domain"],
                        "queue": row["queue"],
                        "job_type": row["job_type"],
                        "owner_user_id": row["owner_user_id"],
                        "request_id": row["request_id"],
                        "trace_id": row["trace_id"],
                    }
                    if outbox_enabled:
                        conn.execute(
                            (
                                "INSERT INTO job_events(job_id,domain,queue,job_type,event_type,attrs_json,"
                                "owner_user_id,request_id,trace_id,created_at) "
                                "VALUES(?,?,?,?,'job.cancelled',?,?,?,?,DATETIME('now'))"
                            ),
                            (
                                event_job["id"],
                                event_job["domain"],
                                event_job["queue"],
                                event_job["job_type"],
                                json.dumps(event_attrs),
                                event_job["owner_user_id"],
                                event_job["request_id"],
                                event_job["trace_id"],
                            ),
                        )

        finally:
            _close_connection_nonfatal(conn, operation="cancel finalization")

        if event_job is None:
            return False
        with contextlib.suppress(_JOB_NONCRITICAL_EXCEPTIONS):
            increment_cancelled(event_job)
        with contextlib.suppress(_JOB_NONCRITICAL_EXCEPTIONS):
            self._update_gauges(
                domain=event_job["domain"],
                queue=event_job["queue"],
                job_type=event_job["job_type"],
            )
        with contextlib.suppress(_JOB_NONCRITICAL_EXCEPTIONS):
            if outbox_enabled:
                submit_job_audit_event("job.cancelled", job=event_job, attrs=event_attrs)
            else:
                emit_job_event("job.cancelled", job=event_job, attrs=event_attrs)
        return True

    def integrity_sweep(
        self,
        *,
        fix: bool = False,
        domain: str | None = None,
        queue: str | None = None,
        job_type: str | None = None,
    ) -> dict[str, int]:
        """Validate and optionally repair impossible states.

        - non_processing_with_lease: status != processing but lease_id/worker_id/leased_until set
        - processing_expired: processing with missing/expired lease
        If fix=True, clears stale lease fields on non-processing, requeues expired
        processing jobs with retry budget, and terminally fails exhausted jobs.
        """
        conn = self._connect()
        try:
            res = {
                "non_processing_with_lease": 0,
                "processing_expired": 0,
                "fixed": 0,
            }
            if self.backend == "postgres":
                with conn:  # noqa: SIM117
                    with self._pg_cursor(conn) as cur:
                        where_np = [
                            "status <> 'processing'",
                            "(lease_id IS NOT NULL OR worker_id IS NOT NULL OR leased_until IS NOT NULL)",
                        ]
                        where_pr = ["status = 'processing'", "(leased_until IS NULL OR leased_until <= NOW())"]
                        params_np: list[Any] = []
                        params_pr: list[Any] = []
                        if domain:
                            where_np.append("domain = %s")
                            params_np.append(domain)
                            where_pr.append("domain = %s")
                            params_pr.append(domain)
                        if queue:
                            where_np.append("queue = %s")
                            params_np.append(queue)
                            where_pr.append("queue = %s")
                            params_pr.append(queue)
                        if job_type:
                            where_np.append("job_type = %s")
                            params_np.append(job_type)
                            where_pr.append("job_type = %s")
                            params_pr.append(job_type)
                        cur.execute(f"SELECT COUNT(*) AS c FROM jobs WHERE {' AND '.join(where_np)}", tuple(params_np))  # nosec B608
                        _np = cur.fetchone()
                        res["non_processing_with_lease"] = int(_np.get("c") if isinstance(_np, dict) else 0)
                        cur.execute(f"SELECT COUNT(*) AS c FROM jobs WHERE {' AND '.join(where_pr)}", tuple(params_pr))  # nosec B608
                        _pr = cur.fetchone()
                        res["processing_expired"] = int(_pr.get("c") if isinstance(_pr, dict) else 0)
                        if fix:
                            # Clear leases for non-processing
                            cur.execute(
                                f"UPDATE jobs SET lease_id = NULL, leased_until = NULL, worker_id = NULL WHERE {' AND '.join(where_np)}",  # nosec B608
                                tuple(params_np),
                            )
                            res["fixed"] += cur.rowcount or 0
            else:
                where_np = [
                    "status <> 'processing'",
                    "(lease_id IS NOT NULL OR worker_id IS NOT NULL OR leased_until IS NOT NULL)",
                ]
                where_pr = ["status = 'processing'", "(leased_until IS NULL OR leased_until <= DATETIME('now'))"]
                params_np: list[Any] = []
                params_pr: list[Any] = []
                if domain:
                    where_np.append("domain = ?")
                    params_np.append(domain)
                    where_pr.append("domain = ?")
                    params_pr.append(domain)
                if queue:
                    where_np.append("queue = ?")
                    params_np.append(queue)
                    where_pr.append("queue = ?")
                    params_pr.append(queue)
                if job_type:
                    where_np.append("job_type = ?")
                    params_np.append(job_type)
                    where_pr.append("job_type = ?")
                    params_pr.append(job_type)
                cur = conn.execute(f"SELECT COUNT(*) FROM jobs WHERE {' AND '.join(where_np)}", tuple(params_np))  # nosec B608
                res["non_processing_with_lease"] = int(cur.fetchone()[0])
                cur2 = conn.execute(f"SELECT COUNT(*) FROM jobs WHERE {' AND '.join(where_pr)}", tuple(params_pr))  # nosec B608
                res["processing_expired"] = int(cur2.fetchone()[0])
                if fix:
                    with conn:
                        cur_fix_1 = conn.execute(
                            f"UPDATE jobs SET lease_id = NULL, leased_until = NULL, worker_id = NULL WHERE {' AND '.join(where_np)}",  # nosec B608
                            tuple(params_np),
                        )
                        res["fixed"] += int(cur_fix_1.rowcount or 0)
            if fix:
                while True:
                    reconciled = self._reconcile_terminal_dependents(
                        domain=domain,
                        queue=queue,
                        job_type=job_type,
                    )
                    res["fixed"] += reconciled
                    if reconciled == 0:
                        break
                batch_size = JobManager._expired_recovery_batch_size()
                while True:
                    recovered = self._recover_expired_processing_jobs(
                        domain=domain,
                        queue=queue,
                        job_type=job_type,
                    )
                    res["fixed"] += recovered
                    if recovered < batch_size:
                        break
                while True:
                    reconciled = self._reconcile_terminal_dependents(
                        domain=domain,
                        queue=queue,
                        job_type=job_type,
                    )
                    res["fixed"] += reconciled
                    if reconciled == 0:
                        break
            with contextlib.suppress(_JOB_NONCRITICAL_EXCEPTIONS):
                emit_job_event(
                    "jobs.integrity_sweep",
                    job=None,
                    attrs={
                        "fixed": int(res.get("fixed", 0)),
                        "non_processing_with_lease": int(res.get("non_processing_with_lease", 0)),
                        "processing_expired": int(res.get("processing_expired", 0)),
                        "domain": domain,
                        "queue": queue,
                        "job_type": job_type,
                        "fix": bool(fix),
                    },
                )
            return res
        finally:
            with contextlib.suppress(_JOB_NONCRITICAL_EXCEPTIONS):
                conn.close()
