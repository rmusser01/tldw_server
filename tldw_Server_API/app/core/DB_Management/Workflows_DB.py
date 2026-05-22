"""
Workflows database adapter (SQLite by default).

Provides minimal persistence for workflow definitions, runs and events
to support v0.1 engine scaffolding.
"""

from __future__ import annotations

import contextlib
import json
import os
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from loguru import logger

from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths

from .backends.base import (
    BackendType,
    DatabaseBackend,
    QueryResult,
)
from .backends.base import (
    DatabaseError as BackendDatabaseError,
)
from .backends.query_utils import (
    prepare_backend_many_statement,
    prepare_backend_statement,
)
from .sqlite_policy import configure_sqlite_connection

_WORKFLOWS_DB_NONCRITICAL_EXCEPTIONS = (
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
    json.JSONDecodeError,
    sqlite3.Error,
    BackendDatabaseError,
)


class WorkflowsSchemaError(RuntimeError):
    """Raised when workflow schema initialization or migration fails."""

    pass


DEFAULT_DB_PATH = DatabasePaths.get_workflows_db_path(DatabasePaths.get_single_user_id())


WORKFLOWS_POSTGRES_SCHEMA = """
CREATE TABLE IF NOT EXISTS workflows (
    id SERIAL PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    name TEXT NOT NULL,
    version INTEGER NOT NULL,
    owner_id TEXT NOT NULL,
    visibility TEXT NOT NULL DEFAULT 'private',
    description TEXT,
    tags TEXT,
    definition_json TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    UNIQUE (tenant_id, name, version)
);

    CREATE TABLE IF NOT EXISTS workflow_runs (
        run_id TEXT PRIMARY KEY,
        tenant_id TEXT NOT NULL,
        workflow_id INTEGER,
        status TEXT NOT NULL,
        status_reason TEXT,
        user_id TEXT NOT NULL,
        inputs_json TEXT NOT NULL,
        outputs_json TEXT,
        error TEXT,
        duration_ms INTEGER,
        created_at TIMESTAMPTZ NOT NULL,
        started_at TIMESTAMPTZ,
        ended_at TIMESTAMPTZ,
        definition_version INTEGER,
        definition_snapshot_json TEXT,
        idempotency_key TEXT,
        session_id TEXT,
        validation_mode TEXT DEFAULT 'block',
        metadata_json TEXT,
        tokens_input INTEGER,
        tokens_output INTEGER,
        cost_usd DOUBLE PRECISION,
        cancel_requested BOOLEAN NOT NULL DEFAULT FALSE,
        FOREIGN KEY (workflow_id) REFERENCES workflows(id)
    );

CREATE TABLE IF NOT EXISTS workflow_step_runs (
    step_run_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    step_id TEXT NOT NULL,
    name TEXT,
    type TEXT,
    status TEXT,
    attempt INTEGER DEFAULT 0,
    started_at TIMESTAMPTZ,
    ended_at TIMESTAMPTZ,
    inputs_json TEXT,
    outputs_json TEXT,
    error TEXT,
    decision TEXT,
    assigned_to TEXT,
    approved_by TEXT,
    approved_at TIMESTAMPTZ,
    review_comment TEXT,
    locked_by TEXT,
    locked_at TIMESTAMPTZ,
    lock_expires_at TIMESTAMPTZ,
    heartbeat_at TIMESTAMPTZ,
    pid INTEGER,
    pgid INTEGER,
    workdir TEXT,
    stdout_path TEXT,
    stderr_path TEXT,
    FOREIGN KEY (run_id) REFERENCES workflow_runs(run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS workflow_step_attempts (
    attempt_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    step_run_id TEXT NOT NULL,
    step_id TEXT NOT NULL,
    attempt_number INTEGER NOT NULL,
    status TEXT NOT NULL,
    reason_code_core TEXT,
    reason_code_detail TEXT,
    retryable BOOLEAN,
    error_summary TEXT,
    metadata_json JSONB,
    started_at TIMESTAMPTZ NOT NULL,
    ended_at TIMESTAMPTZ,
    UNIQUE (step_run_id, attempt_number),
    FOREIGN KEY (run_id) REFERENCES workflow_runs(run_id) ON DELETE CASCADE,
    FOREIGN KEY (step_run_id) REFERENCES workflow_step_runs(step_run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS workflow_events (
    event_id BIGSERIAL PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    step_run_id TEXT,
    event_seq INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    payload_json JSONB,
    created_at TIMESTAMPTZ NOT NULL,
    FOREIGN KEY (run_id) REFERENCES workflow_runs(run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS workflow_artifacts (
    artifact_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    step_run_id TEXT,
    type TEXT,
    uri TEXT,
    size_bytes BIGINT,
    mime_type TEXT,
    checksum_sha256 TEXT,
    encryption TEXT,
    owned_by TEXT,
    metadata_json TEXT,
    created_at TIMESTAMPTZ NOT NULL,
    FOREIGN KEY (run_id) REFERENCES workflow_runs(run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS workflow_research_waits (
    wait_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    workflow_run_id TEXT NOT NULL,
    step_id TEXT NOT NULL,
    research_run_id TEXT NOT NULL,
    checkpoint_id TEXT NOT NULL,
    checkpoint_type TEXT NOT NULL,
    wait_status TEXT NOT NULL,
    wait_payload_json TEXT NOT NULL,
    active_poll_seconds DOUBLE PRECISION NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    resumed_at TIMESTAMPTZ,
    FOREIGN KEY (workflow_run_id) REFERENCES workflow_runs(run_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_workflows_owner ON workflows(owner_id);
CREATE INDEX IF NOT EXISTS idx_runs_status ON workflow_runs(status);
CREATE INDEX IF NOT EXISTS idx_runs_idempotency_lookup ON workflow_runs(tenant_id, user_id, idempotency_key, created_at);
CREATE INDEX IF NOT EXISTS idx_step_attempts_run_attempts ON workflow_step_attempts(run_id, attempt_number, started_at);
CREATE INDEX IF NOT EXISTS idx_step_attempts_run_step_attempts ON workflow_step_attempts(run_id, step_id, attempt_number, started_at);
CREATE INDEX IF NOT EXISTS idx_step_attempts_step_run_attempts ON workflow_step_attempts(step_run_id, attempt_number, started_at);
    CREATE INDEX IF NOT EXISTS idx_events_run_seq ON workflow_events(run_id, event_seq);
CREATE UNIQUE INDEX IF NOT EXISTS ux_workflow_research_wait_run_step ON workflow_research_waits(workflow_run_id, step_id);
CREATE INDEX IF NOT EXISTS idx_workflow_research_wait_lookup ON workflow_research_waits(research_run_id, checkpoint_id, wait_status);

-- Ensure uniqueness of per-run event sequence
    CREATE UNIQUE INDEX IF NOT EXISTS ux_events_run_seq ON workflow_events(run_id, event_seq);

-- Partial indexes on hot statuses for faster lookups
    CREATE INDEX IF NOT EXISTS idx_runs_status_running ON workflow_runs(status) WHERE status = 'running';
    CREATE INDEX IF NOT EXISTS idx_runs_status_queued ON workflow_runs(status) WHERE status = 'queued';
    CREATE INDEX IF NOT EXISTS idx_runs_status_succeeded ON workflow_runs(status) WHERE status = 'succeeded';
    CREATE INDEX IF NOT EXISTS idx_runs_status_failed ON workflow_runs(status) WHERE status = 'failed';

-- Per-run event sequence counters (optional optimization)
    CREATE TABLE IF NOT EXISTS workflow_event_counters (
        run_id TEXT PRIMARY KEY,
        next_seq INTEGER NOT NULL
    );

    -- Dead-letter queue for webhook deliveries (optional retry worker)
    CREATE TABLE IF NOT EXISTS workflow_webhook_dlq (
        id BIGSERIAL PRIMARY KEY,
        tenant_id TEXT NOT NULL,
        run_id TEXT NOT NULL,
        url TEXT NOT NULL,
        body_json TEXT,
        attempts INTEGER NOT NULL DEFAULT 0,
        next_attempt_at TIMESTAMPTZ,
        last_error TEXT,
        created_at TIMESTAMPTZ NOT NULL
    );
"""


def _utcnow_iso() -> str:
    return datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()


def _workflows_idempotency_ttl_hours() -> int:
    """Return idempotency TTL in hours for workflow runs."""
    raw = os.getenv("WORKFLOWS_IDEMPOTENCY_TTL_HOURS", "24")
    try:
        ttl = int(raw)
    except (TypeError, ValueError):
        logger.warning("Invalid WORKFLOWS_IDEMPOTENCY_TTL_HOURS={}; defaulting to 24", raw)
        ttl = 24
    return max(1, ttl)


class WorkflowRowAdapter:
    """Row wrapper that mimics sqlite3.Row semantics for backend results."""

    __slots__ = ("_mapping", "_columns")

    def __init__(self, mapping: dict[str, Any], columns: tuple[str, ...]):
        self._mapping = mapping
        self._columns = columns

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, int):
            column = self._columns[key]
            return self._mapping.get(column)
        return self._mapping.get(key)

    def __iter__(self):
        return iter(self.items())

    def items(self):
        for column in self._columns:
            yield column, self._mapping.get(column)

    def keys(self) -> tuple[str, ...]:
        return self._columns

    def get(self, key: str, default: Any = None) -> Any:
        return self._mapping.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        return dict(self._mapping)


class WorkflowsBackendCursorAdapter:
    """Adapter that provides sqlite-like fetch methods for backend QueryResult."""

    def __init__(self, result: QueryResult):
        self._result = result
        self._rows = self._build_rows(result)
        self._index = 0
        self.description = result.description or []

    def _build_rows(self, result: QueryResult) -> list[WorkflowRowAdapter]:
        rows: list[WorkflowRowAdapter] = []
        columns: tuple[str, ...] = ()
        if result.description:
            columns = tuple(desc[0] for desc in result.description if desc)
        for mapping in result.rows:
            mapping_dict = dict(mapping)
            if not columns:
                columns = tuple(mapping_dict.keys())
            rows.append(WorkflowRowAdapter(mapping_dict, columns))
        return rows

    def fetchone(self) -> WorkflowRowAdapter | None:
        if self._index >= len(self._rows):
            return None
        row = self._rows[self._index]
        self._index += 1
        return row

    def fetchall(self) -> list[WorkflowRowAdapter]:
        if self._index >= len(self._rows):
            return []
        rows = self._rows[self._index :]
        self._index = len(self._rows)
        return rows

    def fetchmany(self, size: int | None = None) -> list[WorkflowRowAdapter]:
        if size is None or size <= 0:
            size = len(self._rows) - self._index
        rows = self._rows[self._index : self._index + size]
        self._index += len(rows)
        return rows

    def close(self) -> None:
        self._rows = []
        self._index = 0


class WorkflowsBackendCursor:
    """Cursor wrapper that routes SQL through a DatabaseBackend."""

    def __init__(self, db: WorkflowsDatabase):
        self._db = db
        self._adapter: WorkflowsBackendCursorAdapter | None = None
        self.rowcount: int = -1
        self.lastrowid: int | None = None
        self.description = None

    def _requires_returning(self, query: str) -> bool:
        stripped = query.lstrip().upper()
        return stripped.startswith("INSERT")

    def execute(self, query: str, params: Any | None = None):
        backend = self._db.backend
        if backend is None:
            raise RuntimeError("Backend cursor cannot execute without a backend instance")

        ensure_returning = self._requires_returning(query)
        prepared_query, prepared_params = prepare_backend_statement(
            backend.backend_type,
            query,
            params,
            apply_default_transform=True,
            ensure_returning=ensure_returning,
        )

        conn = None
        try:
            conn = backend.get_pool().get_connection()
            result = backend.execute(prepared_query, prepared_params, connection=conn)
        finally:
            if conn is not None:
                backend.get_pool().return_connection(conn)

        self._adapter = WorkflowsBackendCursorAdapter(result)
        self.rowcount = result.rowcount
        self.lastrowid = result.lastrowid
        self.description = result.description
        return self

    def executemany(self, query: str, params_list: list[Any]):
        backend = self._db.backend
        if backend is None:
            raise RuntimeError("Backend cursor cannot execute without a backend instance")

        prepared_query, prepared_params_list = prepare_backend_many_statement(
            backend.backend_type,
            query,
            params_list,
            apply_default_transform=True,
        )

        conn = None
        try:
            conn = backend.get_pool().get_connection()
            result = backend.execute_many(prepared_query, prepared_params_list, connection=conn)
        finally:
            if conn is not None:
                backend.get_pool().return_connection(conn)

        self._adapter = WorkflowsBackendCursorAdapter(result)
        self.rowcount = result.rowcount
        self.lastrowid = result.lastrowid
        self.description = result.description
        return self

    def fetchone(self):
        if not self._adapter:
            return None
        return self._adapter.fetchone()

    def fetchall(self):
        if not self._adapter:
            return []
        return self._adapter.fetchall()

    def fetchmany(self, size: int | None = None):
        if not self._adapter:
            return []
        return self._adapter.fetchmany(size)

    def close(self) -> None:
        if self._adapter:
            self._adapter.close()
        self._adapter = None
        self.rowcount = -1
        self.lastrowid = None
        self.description = None


class WorkflowsBackendConnection:
    """Connection shim exposing sqlite-style helpers for backend usage."""

    def __init__(self, db: WorkflowsDatabase) -> None:
        self._db = db

    def cursor(self) -> WorkflowsBackendCursor:
        return WorkflowsBackendCursor(self._db)

    def execute(self, query: str, params: Any | None = None):
        cursor = self.cursor()
        return cursor.execute(query, params)

    def executemany(self, query: str, params_list: list[Any]):
        cursor = self.cursor()
        return cursor.executemany(query, params_list)

    def commit(self) -> None:  # pragma: no cover - compatibility
        return None

    def rollback(self) -> None:  # pragma: no cover - compatibility
        return None

    def close(self) -> None:  # pragma: no cover - compatibility
        return None


@dataclass
class WorkflowDefinition:
    id: int
    tenant_id: str
    name: str
    version: int
    owner_id: str
    visibility: str
    description: str | None
    tags: str | None
    definition_json: str
    created_at: str
    updated_at: str
    is_active: int


@dataclass
class WorkflowRun:
    run_id: str
    tenant_id: str
    workflow_id: int | None
    status: str
    status_reason: str | None
    user_id: str
    inputs_json: str
    outputs_json: str | None
    error: str | None
    duration_ms: int | None
    created_at: str
    started_at: str | None
    ended_at: str | None
    definition_version: int | None
    definition_snapshot_json: str | None
    idempotency_key: str | None
    session_id: str | None
    metadata_json: str | None = None
    cancel_requested: int | None = 0
    # Accounting fields (nullable)
    tokens_input: int | None = None
    tokens_output: int | None = None
    cost_usd: float | None = None
    validation_mode: str | None = "block"


class WorkflowsDatabase:
    _CURRENT_SCHEMA_VERSION = 9
    """Workflow persistence adapter supporting SQLite and DatabaseBackend instances."""

    def __init__(
        self,
        db_path: str | None = None,
        *,
        backend: DatabaseBackend | None = None,
    ) -> None:
        self.backend: DatabaseBackend | None = None
        self.backend_type: BackendType = BackendType.SQLITE

        if backend and backend.backend_type == BackendType.POSTGRESQL:
            self.backend = backend
            self.backend_type = backend.backend_type
            self.db_path = str(db_path or DEFAULT_DB_PATH)
            self._conn = WorkflowsBackendConnection(self)
            self._initialize_schema_backend()
            logger.debug("Workflows DB initialized using {} backend", self.backend_type.value)
            return

        # Fallback to SQLite path (default behaviour)
        url = os.getenv("DATABASE_URL_WORKFLOWS", "").strip()
        if not db_path and url:
            if url.startswith("sqlite://"):
                path = url.split("sqlite://", 1)[1]
                resolved = path if path.startswith("/") and not path.startswith("//") else path.lstrip("/")
                db_path = resolved or str(DEFAULT_DB_PATH)
            else:
                logger.warning(
                    'DATABASE_URL_WORKFLOWS={} is not a supported SQLite URI; falling back to default path',
                    url,
                )

        self.db_path = str(db_path or DEFAULT_DB_PATH)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._enable_wal()
        self._create_schema()
        # Optional lightweight SQLite connection pool for high-churn operations
        try:
            pool_size = int(os.getenv("WORKFLOWS_SQLITE_POOL_SIZE", "0"))
        except _WORKFLOWS_DB_NONCRITICAL_EXCEPTIONS:
            pool_size = 0
        self._sqlite_pool: list[sqlite3.Connection] = []
        if pool_size and pool_size > 0:
            for _ in range(max(0, pool_size - 1)):
                c = sqlite3.connect(self.db_path, check_same_thread=False)
                c.row_factory = sqlite3.Row
                try:
                    configure_sqlite_connection(c)
                    c.execute("PRAGMA wal_autocheckpoint=1000;")
                except _WORKFLOWS_DB_NONCRITICAL_EXCEPTIONS:
                    pass
                self._sqlite_pool.append(c)
        logger.debug(f"Workflows DB initialized at {self.db_path}")

    # ------------------------------------------------------------------
    # Backend helpers
    # ------------------------------------------------------------------

    def _using_backend(self) -> bool:
        return self.backend is not None and self.backend_type == BackendType.POSTGRESQL

    def _execute_backend(
        self,
        query: str,
        params: Any | None = None,
        *,
        connection: Any = None,
        ensure_returning: bool = False,
    ) -> QueryResult:
        if not self.backend:
            raise RuntimeError("Backend execution requested without configured backend")

        prepared_query, prepared_params = prepare_backend_statement(
            self.backend.backend_type,
            query,
            params,
            apply_default_transform=True,
            ensure_returning=ensure_returning,
        )
        return self.backend.execute(
            prepared_query,
            prepared_params,
            connection=connection,
        )

    def _execute_backend_many(
        self,
        query: str,
        params_list: Sequence[Any],
        *,
        connection: Any = None,
    ) -> QueryResult:
        if not self.backend:
            raise RuntimeError("Backend execution requested without configured backend")

        prepared_query, prepared_params_list = prepare_backend_many_statement(
            self.backend.backend_type,
            query,
            params_list,
            apply_default_transform=True,
        )
        return self.backend.execute_many(
            prepared_query,
            prepared_params_list,
            connection=connection,
        )

    @staticmethod
    def _rows_from_result(result: QueryResult) -> list[WorkflowRowAdapter]:
        adapter = WorkflowsBackendCursorAdapter(result)
        rows = adapter.fetchall()
        adapter.close()
        return rows

    @staticmethod
    def _row_from_result(result: QueryResult) -> WorkflowRowAdapter | None:
        adapter = WorkflowsBackendCursorAdapter(result)
        row = adapter.fetchone()
        adapter.close()
        return row

    @staticmethod
    def _row_to_dict(row: Any) -> dict[str, Any]:
        if isinstance(row, WorkflowRowAdapter):
            return row.to_dict()
        return dict(row)

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release database resources for both SQLite and backend modes."""
        if self._using_backend():
            if self.backend is not None:
                try:
                    pool = self.backend.get_pool()
                except _WORKFLOWS_DB_NONCRITICAL_EXCEPTIONS:  # noqa: BLE001 - defensive
                    return
                if hasattr(pool, "close_all"):
                    pool.close_all()
            return

        # Close pooled connections first
        if hasattr(self, "_sqlite_pool") and self._sqlite_pool:
            for c in self._sqlite_pool:
                with contextlib.suppress(_WORKFLOWS_DB_NONCRITICAL_EXCEPTIONS):
                    c.close()
            self._sqlite_pool = []
        if hasattr(self, "_conn") and self._conn is not None:
            try:
                self._conn.close()
            finally:
                self._conn = None

    def close_connection(self) -> None:
        """Backward-compatible alias expected by older callers/tests."""
        self.close()

    def _enable_wal(self) -> None:
        try:
            configure_sqlite_connection(self._conn)
            self._conn.execute("PRAGMA wal_autocheckpoint=1000;")
        except _WORKFLOWS_DB_NONCRITICAL_EXCEPTIONS as e:
            logger.warning(f"Failed to enable WAL on workflows DB: {e}")

    def _get_backend_schema_version(self, conn) -> int:
        if not self.backend:
            return 0

        backend = self.backend
        ident = backend.escape_identifier

        backend.execute(
            f"CREATE TABLE IF NOT EXISTS {ident('workflow_schema_version')} (version INTEGER NOT NULL)",  # nosec B608
            connection=conn,
        )

        result = backend.execute(
            f"SELECT version FROM {ident('workflow_schema_version')} LIMIT 1",  # nosec B608
            connection=conn,
        )
        if not result.rows:
            backend.execute(
                f"INSERT INTO {ident('workflow_schema_version')} (version) VALUES (%s)",  # nosec B608
                (0,),
                connection=conn,
            )
            return 0
        return int(result.scalar or 0)

    def _set_backend_schema_version(self, conn, version: int) -> None:
        if not self.backend:
            return

        backend = self.backend
        ident = backend.escape_identifier
        result = backend.execute(
            f"UPDATE {ident('workflow_schema_version')} SET version = %s",  # nosec B608
            (int(version),),
            connection=conn,
        )
        if result.rowcount == 0:
            backend.execute(
                f"INSERT INTO {ident('workflow_schema_version')} (version) VALUES (%s)",  # nosec B608
                (int(version),),
                connection=conn,
            )

    def _run_backend_migrations(self, conn, current_version: int, target_version: int) -> int:
        if not self.backend:
            return current_version

        migrations = self._get_backend_migrations()
        applied_version = current_version

        for version in sorted(migrations.keys()):
            if applied_version < version <= target_version:
                migrations[version](conn)
                self._set_backend_schema_version(conn, version)
                applied_version = version

        if applied_version < target_version:
            raise WorkflowsSchemaError(
                f"Incomplete migration path for workflows backend schema (reached {applied_version}, expected {target_version})."
            )

        return applied_version

    @staticmethod
    def _step_attempt_migration_score(row: dict[str, Any]) -> tuple[bool, str, str, str]:
        return (
            row.get("ended_at") is not None,
            str(row.get("ended_at") or row.get("started_at") or ""),
            str(row.get("started_at") or ""),
            str(row.get("attempt_id") or ""),
        )

    def _normalize_step_attempt_migration_rows(
        self,
        rows: Sequence[Any],
        *,
        valid_run_ids: set[str] | None = None,
        valid_step_run_ids: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        canonical_rows: dict[tuple[str, int], dict[str, Any]] = {}
        for row in rows:
            data = self._row_to_dict(row)
            run_id = str(data.get("run_id") or "").strip()
            if not run_id:
                continue
            if valid_run_ids is not None and run_id not in valid_run_ids:
                continue
            step_run_id = str(data.get("step_run_id") or "").strip()
            if not step_run_id:
                continue
            if valid_step_run_ids is not None and step_run_id not in valid_step_run_ids:
                continue
            try:
                attempt_number = int(data.get("attempt_number") or 0)
            except _WORKFLOWS_DB_NONCRITICAL_EXCEPTIONS:
                continue
            if attempt_number <= 0:
                continue
            logical_key = (step_run_id, attempt_number)
            existing = canonical_rows.get(logical_key)
            if existing is None or self._step_attempt_migration_score(data) > self._step_attempt_migration_score(existing):
                canonical_rows[logical_key] = data
        return [
            canonical_rows[key]
            for key in sorted(canonical_rows.keys(), key=lambda item: (item[0], item[1]))
        ]

    def _get_backend_migrations(self):
        return {
            1: self._backend_migrate_to_v1,
            2: self._backend_migrate_to_v2,
            3: self._backend_migrate_to_v3,
            4: self._backend_migrate_to_v4,
            5: self._backend_migrate_to_v5,
            6: self._backend_migrate_to_v6,
            7: self._backend_migrate_to_v7,
            8: self._backend_migrate_to_v8,
            9: self._backend_migrate_to_v9,
        }

    def _backend_migrate_to_v1(self, conn) -> None:
        if not self.backend:
            return

        backend = self.backend
        ident = backend.escape_identifier

        column_additions = [
            ("workflow_runs", "cancel_requested", "BOOLEAN NOT NULL DEFAULT FALSE"),
            ("workflow_runs", "tokens_input", "INTEGER"),
            ("workflow_runs", "tokens_output", "INTEGER"),
            ("workflow_runs", "cost_usd", "DOUBLE PRECISION"),
            ("workflow_step_runs", "pid", "INTEGER"),
            ("workflow_step_runs", "pgid", "INTEGER"),
            ("workflow_step_runs", "workdir", "TEXT"),
            ("workflow_step_runs", "stdout_path", "TEXT"),
            ("workflow_step_runs", "stderr_path", "TEXT"),
        ]

        for table, column, column_type in column_additions:
            backend.execute(
                f"ALTER TABLE {ident(table)} ADD COLUMN IF NOT EXISTS {ident(column)} {column_type}",
                connection=conn,
            )

        backend.execute(
            f"CREATE TABLE IF NOT EXISTS {ident('workflow_artifacts')} ("
            f"artifact_id TEXT PRIMARY KEY,"
            f"tenant_id TEXT NOT NULL,"
            f"run_id TEXT NOT NULL,"
            f"step_run_id TEXT,"
            f"type TEXT,"
            f"uri TEXT,"
            f"size_bytes BIGINT,"
            f"mime_type TEXT,"
            f"checksum_sha256 TEXT,"
            f"encryption TEXT,"
            f"owned_by TEXT,"
            f"metadata_json TEXT,"
            f"created_at TIMESTAMPTZ NOT NULL,"
            f"FOREIGN KEY ({ident('run_id')}) REFERENCES {ident('workflow_runs')}({ident('run_id')})"
            ")",
            connection=conn,
        )

        backend.execute(
            f"CREATE INDEX IF NOT EXISTS {ident('idx_workflows_owner')} ON {ident('workflows')} ({ident('owner_id')})",
            connection=conn,
        )
        backend.execute(
            f"CREATE INDEX IF NOT EXISTS {ident('idx_runs_status')} ON {ident('workflow_runs')} ({ident('status')})",
            connection=conn,
        )
        backend.execute(
            f"CREATE INDEX IF NOT EXISTS {ident('idx_events_run_seq')} ON {ident('workflow_events')} ({ident('run_id')}, {ident('event_seq')})",
            connection=conn,
        )

        # Ensure uniqueness of per-run event sequence (idempotent via unique index)
        backend.execute(
            f"CREATE UNIQUE INDEX IF NOT EXISTS {ident('ux_events_run_seq')} ON {ident('workflow_events')} ({ident('run_id')}, {ident('event_seq')})",
            connection=conn,
        )

        # Event counters table (idempotent)
        backend.execute(
            f"CREATE TABLE IF NOT EXISTS {ident('workflow_event_counters')} ("
            f"run_id TEXT PRIMARY KEY,"
            f"next_seq INTEGER NOT NULL"
            ")",
            connection=conn,
        )

    def _backend_migrate_to_v2(self, conn) -> None:
        if not self.backend:
            return
        backend = self.backend
        ident = backend.escape_identifier
        # Add validation_mode to workflow_runs
        backend.execute(
            f"ALTER TABLE {ident('workflow_runs')} ADD COLUMN IF NOT EXISTS {ident('validation_mode')} TEXT DEFAULT 'block'",
            connection=conn,
        )

    def _backend_migrate_to_v3(self, conn) -> None:
        if not self.backend:
            return
        backend = self.backend
        ident = backend.escape_identifier

        # Convert payload_json to JSONB if needed
        with contextlib.suppress(_WORKFLOWS_DB_NONCRITICAL_EXCEPTIONS):
            backend.execute(
                f"ALTER TABLE {ident('workflow_events')} "
                f"ALTER COLUMN {ident('payload_json')} TYPE JSONB USING {ident('payload_json')}::jsonb",
                connection=conn,
            )

        # Add GIN index on JSONB payloads
        with contextlib.suppress(_WORKFLOWS_DB_NONCRITICAL_EXCEPTIONS):
            backend.execute(
                f"CREATE INDEX IF NOT EXISTS {ident('idx_events_payload_json_gin')} "
                f"ON {ident('workflow_events')} USING GIN ({ident('payload_json')})",
                connection=conn,
            )

        # Recreate FK constraints to cascade on run delete
        for table in ("workflow_events", "workflow_step_runs", "workflow_artifacts"):
            with contextlib.suppress(_WORKFLOWS_DB_NONCRITICAL_EXCEPTIONS):
                backend.execute(
                    f"ALTER TABLE {ident(table)} DROP CONSTRAINT IF EXISTS {ident(f'{table}_run_id_fkey')}",
                    connection=conn,
                )
            with contextlib.suppress(_WORKFLOWS_DB_NONCRITICAL_EXCEPTIONS):
                backend.execute(
                    f"ALTER TABLE {ident(table)} ADD CONSTRAINT {ident(f'{table}_run_id_fkey')} "
                    f"FOREIGN KEY ({ident('run_id')}) REFERENCES {ident('workflow_runs')}({ident('run_id')}) ON DELETE CASCADE",
                    connection=conn,
                )

        # Partial indexes for hot statuses
        try:
            backend.execute(
                f"CREATE INDEX IF NOT EXISTS {ident('idx_runs_status_running')} ON {ident('workflow_runs')}({ident('status')}) WHERE {ident('status')} = 'running'",
                connection=conn,
            )
            backend.execute(
                f"CREATE INDEX IF NOT EXISTS {ident('idx_runs_status_queued')} ON {ident('workflow_runs')}({ident('status')}) WHERE {ident('status')} = 'queued'",
                connection=conn,
            )
        except _WORKFLOWS_DB_NONCRITICAL_EXCEPTIONS:
            pass
        # Dead-letter queue table for webhooks
        backend.execute(
            f"CREATE TABLE IF NOT EXISTS {ident('workflow_webhook_dlq')} ("
            f"id BIGSERIAL PRIMARY KEY,"
            f"tenant_id TEXT NOT NULL,"
            f"run_id TEXT NOT NULL,"
            f"url TEXT NOT NULL,"
            f"body_json TEXT,"
            f"attempts INTEGER NOT NULL DEFAULT 0,"
            f"next_attempt_at TIMESTAMPTZ,"
            f"last_error TEXT,"
            f"created_at TIMESTAMPTZ NOT NULL"
            ")",
            connection=conn,
        )

    def _backend_migrate_to_v4(self, conn) -> None:
        if not self.backend:
            return
        backend = self.backend
        ident = backend.escape_identifier
        # Add additional partial indexes for common terminal statuses
        with contextlib.suppress(_WORKFLOWS_DB_NONCRITICAL_EXCEPTIONS):
            backend.execute(
                f"CREATE INDEX IF NOT EXISTS {ident('idx_runs_status_succeeded')} ON {ident('workflow_runs')}({ident('status')}) WHERE {ident('status')} = 'succeeded'",
                connection=conn,
            )
        with contextlib.suppress(_WORKFLOWS_DB_NONCRITICAL_EXCEPTIONS):
            backend.execute(
                f"CREATE INDEX IF NOT EXISTS {ident('idx_runs_status_failed')} ON {ident('workflow_runs')}({ident('status')}) WHERE {ident('status')} = 'failed'",
                connection=conn,
            )

    def _backend_migrate_to_v5(self, conn) -> None:
        if not self.backend:
            return
        backend = self.backend
        ident = backend.escape_identifier

        # Add tenant/assignee columns to workflow_step_runs
        backend.execute(
            f"ALTER TABLE {ident('workflow_step_runs')} ADD COLUMN IF NOT EXISTS {ident('tenant_id')} TEXT",
            connection=conn,
        )
        backend.execute(
            f"ALTER TABLE {ident('workflow_step_runs')} ADD COLUMN IF NOT EXISTS {ident('assigned_to')} TEXT",
            connection=conn,
        )

        # Best-effort backfill tenant_id from workflow_runs
        with contextlib.suppress(_WORKFLOWS_DB_NONCRITICAL_EXCEPTIONS):
            backfill_tenant_sql_template = (
                "UPDATE {step_runs_table} AS s "
                "SET {tenant_column} = r.{tenant_column} "
                "FROM {runs_table} AS r "
                "WHERE s.{run_id_column} = r.{run_id_column} "
                "AND (s.{tenant_column} IS NULL OR s.{tenant_column} = '')"
            )
            step_runs_table = ident("workflow_step_runs")
            runs_table = ident("workflow_runs")
            tenant_column = ident("tenant_id")
            run_id_column = ident("run_id")
            backfill_tenant_sql = backfill_tenant_sql_template.format_map(locals())  # nosec B608
            backend.execute(
                backfill_tenant_sql,
                connection=conn,
            )

    def _backend_migrate_to_v6(self, conn) -> None:
        if not self.backend:
            return
        backend = self.backend
        ident = backend.escape_identifier
        with contextlib.suppress(_WORKFLOWS_DB_NONCRITICAL_EXCEPTIONS):
            backend.execute(
                f"CREATE INDEX IF NOT EXISTS {ident('idx_runs_idempotency_lookup')} "
                f"ON {ident('workflow_runs')} ({ident('tenant_id')}, {ident('user_id')}, "
                f"{ident('idempotency_key')}, {ident('created_at')})",
                connection=conn,
            )

    def _backend_migrate_to_v7(self, conn) -> None:
        if not self.backend:
            return
        backend = self.backend
        ident = backend.escape_identifier
        backend.execute(
            f"CREATE TABLE IF NOT EXISTS {ident('workflow_research_waits')} ("
            f"{ident('wait_id')} TEXT PRIMARY KEY,"
            f"{ident('tenant_id')} TEXT NOT NULL,"
            f"{ident('workflow_run_id')} TEXT NOT NULL,"
            f"{ident('step_id')} TEXT NOT NULL,"
            f"{ident('research_run_id')} TEXT NOT NULL,"
            f"{ident('checkpoint_id')} TEXT NOT NULL,"
            f"{ident('checkpoint_type')} TEXT NOT NULL,"
            f"{ident('wait_status')} TEXT NOT NULL,"
            f"{ident('wait_payload_json')} TEXT NOT NULL,"
            f"{ident('active_poll_seconds')} DOUBLE PRECISION NOT NULL DEFAULT 0,"
            f"{ident('created_at')} TIMESTAMPTZ NOT NULL,"
            f"{ident('updated_at')} TIMESTAMPTZ NOT NULL,"
            f"{ident('resumed_at')} TIMESTAMPTZ,"
            f"FOREIGN KEY ({ident('workflow_run_id')}) REFERENCES {ident('workflow_runs')}({ident('run_id')}) ON DELETE CASCADE"
            ")",
            connection=conn,
        )
        backend.execute(
            f"CREATE UNIQUE INDEX IF NOT EXISTS {ident('ux_workflow_research_wait_run_step')} "
            f"ON {ident('workflow_research_waits')} ({ident('workflow_run_id')}, {ident('step_id')})",
            connection=conn,
        )
        backend.execute(
            f"CREATE INDEX IF NOT EXISTS {ident('idx_workflow_research_wait_lookup')} "
            f"ON {ident('workflow_research_waits')} ({ident('research_run_id')}, {ident('checkpoint_id')}, {ident('wait_status')})",
            connection=conn,
        )

    def _backend_migrate_to_v8(self, conn) -> None:
        if not self.backend:
            return
        backend = self.backend
        ident = backend.escape_identifier
        legacy_rows: list[Any] = []
        valid_run_ids: set[str] = set()
        valid_step_run_ids: set[str] = set()

        run_result = backend.execute(
            f"SELECT {ident('run_id')} FROM {ident('workflow_runs')}",  # nosec B608
            connection=conn,
        )
        for row in self._rows_from_result(run_result):
            data = self._row_to_dict(row)
            run_id = str(data.get("run_id") or "").strip()
            if run_id:
                valid_run_ids.add(run_id)

        step_run_result = backend.execute(
            f"SELECT {ident('step_run_id')} FROM {ident('workflow_step_runs')}",  # nosec B608
            connection=conn,
        )
        for row in self._rows_from_result(step_run_result):
            data = self._row_to_dict(row)
            step_run_id = str(data.get("step_run_id") or "").strip()
            if step_run_id:
                valid_step_run_ids.add(step_run_id)

        if backend.table_exists("workflow_step_attempts", connection=conn):
            backend.execute(
                f"ALTER TABLE {ident('workflow_step_attempts')} RENAME TO {ident('workflow_step_attempts_legacy_v8')}",
                connection=conn,
            )
            result = backend.execute(
                f"SELECT * FROM {ident('workflow_step_attempts_legacy_v8')}",  # nosec B608
                connection=conn,
            )
            legacy_rows = self._rows_from_result(result)

        backend.execute(
            f"DROP TABLE IF EXISTS {ident('workflow_step_attempts')}",
            connection=conn,
        )
        backend.execute(
            f"CREATE TABLE {ident('workflow_step_attempts')} ("
            f"{ident('attempt_id')} TEXT PRIMARY KEY,"
            f"{ident('tenant_id')} TEXT NOT NULL,"
            f"{ident('run_id')} TEXT NOT NULL,"
            f"{ident('step_run_id')} TEXT NOT NULL,"
            f"{ident('step_id')} TEXT NOT NULL,"
            f"{ident('attempt_number')} INTEGER NOT NULL,"
            f"{ident('status')} TEXT NOT NULL,"
            f"{ident('reason_code_core')} TEXT,"
            f"{ident('reason_code_detail')} TEXT,"
            f"{ident('retryable')} BOOLEAN,"
            f"{ident('error_summary')} TEXT,"
            f"{ident('metadata_json')} JSONB,"
            f"{ident('started_at')} TIMESTAMPTZ NOT NULL,"
            f"{ident('ended_at')} TIMESTAMPTZ,"
            f"UNIQUE ({ident('step_run_id')}, {ident('attempt_number')}),"
            f"FOREIGN KEY ({ident('run_id')}) REFERENCES {ident('workflow_runs')}({ident('run_id')}) ON DELETE CASCADE,"
            f"FOREIGN KEY ({ident('step_run_id')}) REFERENCES {ident('workflow_step_runs')}({ident('step_run_id')}) ON DELETE CASCADE"
            ")",
            connection=conn,
        )
        for row in self._normalize_step_attempt_migration_rows(
            legacy_rows,
            valid_run_ids=valid_run_ids,
            valid_step_run_ids=valid_step_run_ids,
        ):
            metadata_raw = row.get("metadata_json")
            if isinstance(metadata_raw, (dict, list)):
                metadata_value = json.dumps(metadata_raw)
            else:
                metadata_value = metadata_raw
            backend.execute(
                f"INSERT INTO {ident('workflow_step_attempts')} ("  # nosec B608
                f"{ident('attempt_id')}, {ident('tenant_id')}, {ident('run_id')}, {ident('step_run_id')}, "
                f"{ident('step_id')}, {ident('attempt_number')}, {ident('status')}, {ident('reason_code_core')}, "
                f"{ident('reason_code_detail')}, {ident('retryable')}, {ident('error_summary')}, "
                f"{ident('metadata_json')}, {ident('started_at')}, {ident('ended_at')}"
                f") VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    str(row.get("attempt_id") or uuid4()),
                    str(row.get("tenant_id") or ""),
                    str(row.get("run_id") or ""),
                    str(row.get("step_run_id") or ""),
                    str(row.get("step_id") or ""),
                    int(row.get("attempt_number") or 0),
                    str(row.get("status") or "running"),
                    row.get("reason_code_core"),
                    row.get("reason_code_detail"),
                    row.get("retryable"),
                    row.get("error_summary"),
                    metadata_value,
                    str(row.get("started_at") or _utcnow_iso()),
                    row.get("ended_at"),
                ),
                connection=conn,
            )
        backend.execute(
            f"DROP TABLE IF EXISTS {ident('workflow_step_attempts_legacy_v8')}",
            connection=conn,
        )
        backend.execute(
            f"CREATE INDEX IF NOT EXISTS {ident('idx_step_attempts_run_attempts')} "
            f"ON {ident('workflow_step_attempts')} ({ident('run_id')}, {ident('attempt_number')}, {ident('started_at')})",
            connection=conn,
        )
        backend.execute(
            f"CREATE INDEX IF NOT EXISTS {ident('idx_step_attempts_run_step_attempts')} "
            f"ON {ident('workflow_step_attempts')} ({ident('run_id')}, {ident('step_id')}, {ident('attempt_number')}, {ident('started_at')})",
            connection=conn,
        )
        backend.execute(
            f"CREATE INDEX IF NOT EXISTS {ident('idx_step_attempts_step_run_attempts')} "
            f"ON {ident('workflow_step_attempts')} ({ident('step_run_id')}, {ident('attempt_number')}, {ident('started_at')})",
            connection=conn,
        )

    def _backend_migrate_to_v9(self, conn) -> None:
        if not self.backend:
            return
        backend = self.backend
        ident = backend.escape_identifier
        backend.execute(
            f"ALTER TABLE {ident('workflow_runs')} ADD COLUMN IF NOT EXISTS {ident('metadata_json')} TEXT",
            connection=conn,
        )

    def _initialize_schema_backend(self) -> None:
        if not self.backend:
            return

        backend = self.backend
        target_version = self._CURRENT_SCHEMA_VERSION

        try:
            with backend.transaction() as conn:
                backend.create_tables(WORKFLOWS_POSTGRES_SCHEMA, connection=conn)
                current_version = self._get_backend_schema_version(conn)

                if current_version > target_version:
                    raise WorkflowsSchemaError(
                        "Workflows schema version is newer than supported by this release."
                    )

                applied_version = current_version

                if applied_version < target_version:
                    applied_version = self._run_backend_migrations(conn, applied_version, target_version)

                if applied_version != target_version:
                    self._set_backend_schema_version(conn, target_version)
        except WorkflowsSchemaError:
            raise
        except BackendDatabaseError as exc:
            logger.error("Failed to initialise workflows schema on backend: {}", exc)
            raise
        except _WORKFLOWS_DB_NONCRITICAL_EXCEPTIONS as exc:
            logger.error("Unexpected error while initialising workflows schema: {}", exc)
            raise

    def _get_sqlite_schema_version(self) -> int:
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS workflow_schema_version (version INTEGER NOT NULL)"
        )
        row = self._conn.execute(
            "SELECT version FROM workflow_schema_version LIMIT 1"
        ).fetchone()
        if not row:
            self._conn.execute(
                "INSERT INTO workflow_schema_version (version) VALUES (0)"
            )
            self._conn.commit()
            return 0
        return int(row[0] or 0)

    def _set_sqlite_schema_version(self, version: int) -> None:
        cur = self._conn.execute(
            "UPDATE workflow_schema_version SET version = ?",
            (int(version),),
        )
        if cur.rowcount == 0:
            self._conn.execute(
                "INSERT INTO workflow_schema_version (version) VALUES (?)",
                (int(version),),
            )
        self._conn.commit()

    def _sqlite_migrate_to_v8(self) -> None:
        cur = self._conn.cursor()
        legacy_rows: list[Any] = []
        valid_run_ids = {
            str(row[0]).strip()
            for row in cur.execute("SELECT run_id FROM workflow_runs").fetchall()
            if str(row[0]).strip()
        }
        valid_step_run_ids = {
            str(row[0]).strip()
            for row in cur.execute("SELECT step_run_id FROM workflow_step_runs").fetchall()
            if str(row[0]).strip()
        }
        existing = cur.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'workflow_step_attempts'"
        ).fetchone()
        if existing:
            cur.execute("ALTER TABLE workflow_step_attempts RENAME TO workflow_step_attempts_legacy_v8")
            legacy_rows = cur.execute(
                "SELECT * FROM workflow_step_attempts_legacy_v8"
            ).fetchall()

        cur.execute("DROP TABLE IF EXISTS workflow_step_attempts")
        cur.execute(
            """
            CREATE TABLE workflow_step_attempts (
                attempt_id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                run_id TEXT NOT NULL,
                step_run_id TEXT NOT NULL,
                step_id TEXT NOT NULL,
                attempt_number INTEGER NOT NULL,
                status TEXT NOT NULL,
                reason_code_core TEXT,
                reason_code_detail TEXT,
                retryable INTEGER,
                error_summary TEXT,
                metadata_json TEXT,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                UNIQUE(step_run_id, attempt_number),
                FOREIGN KEY(run_id) REFERENCES workflow_runs(run_id) ON DELETE CASCADE,
                FOREIGN KEY(step_run_id) REFERENCES workflow_step_runs(step_run_id) ON DELETE CASCADE
            );
            """
        )
        for row in self._normalize_step_attempt_migration_rows(
            legacy_rows,
            valid_run_ids=valid_run_ids,
            valid_step_run_ids=valid_step_run_ids,
        ):
            metadata_raw = row.get("metadata_json")
            if isinstance(metadata_raw, (dict, list)):
                metadata_value = json.dumps(metadata_raw)
            else:
                metadata_value = metadata_raw
            cur.execute(
                """
                INSERT INTO workflow_step_attempts(
                    attempt_id, tenant_id, run_id, step_run_id, step_id, attempt_number, status, reason_code_core,
                    reason_code_detail, retryable, error_summary, metadata_json, started_at, ended_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(row.get("attempt_id") or uuid4()),
                    str(row.get("tenant_id") or ""),
                    str(row.get("run_id") or ""),
                    str(row.get("step_run_id") or ""),
                    str(row.get("step_id") or ""),
                    int(row.get("attempt_number") or 0),
                    str(row.get("status") or "running"),
                    row.get("reason_code_core"),
                    row.get("reason_code_detail"),
                    row.get("retryable"),
                    row.get("error_summary"),
                    metadata_value,
                    str(row.get("started_at") or _utcnow_iso()),
                    row.get("ended_at"),
                ),
            )
        cur.execute("DROP TABLE IF EXISTS workflow_step_attempts_legacy_v8")
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_step_attempts_run_attempts "
            "ON workflow_step_attempts(run_id, attempt_number, started_at)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_step_attempts_run_step_attempts "
            "ON workflow_step_attempts(run_id, step_id, attempt_number, started_at)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_step_attempts_step_run_attempts "
            "ON workflow_step_attempts(step_run_id, attempt_number, started_at)"
        )
        self._conn.commit()

    def _run_sqlite_migrations(self, current_version: int, target_version: int) -> int:
        applied_version = current_version
        if applied_version < 8 <= target_version:
            self._sqlite_migrate_to_v8()
            self._set_sqlite_schema_version(8)
            applied_version = 8
        if applied_version < 9 <= target_version:
            self._sqlite_migrate_to_v9()
            self._set_sqlite_schema_version(9)
            applied_version = 9
        return applied_version

    def _sqlite_migrate_to_v9(self) -> None:
        cur = self._conn.cursor()
        with contextlib.suppress(_WORKFLOWS_DB_NONCRITICAL_EXCEPTIONS):
            cur.execute("ALTER TABLE workflow_runs ADD COLUMN metadata_json TEXT")
            self._conn.commit()

    def _create_schema(self) -> None:
        current_version = self._get_sqlite_schema_version()
        if current_version > self._CURRENT_SCHEMA_VERSION:
            raise WorkflowsSchemaError(
                "Workflows schema version is newer than supported by this release."
            )

        cur = self._conn.cursor()
        # Definitions
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS workflows (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tenant_id TEXT NOT NULL,
                name TEXT NOT NULL,
                version INTEGER NOT NULL,
                owner_id TEXT NOT NULL,
                visibility TEXT NOT NULL DEFAULT 'private',
                description TEXT,
                tags TEXT,
                definition_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 1,
                UNIQUE(tenant_id, name, version)
            );
            """
        )

        # Runs
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS workflow_runs (
                run_id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                workflow_id INTEGER,
                status TEXT NOT NULL,
                status_reason TEXT,
                user_id TEXT NOT NULL,
                inputs_json TEXT NOT NULL,
                outputs_json TEXT,
                error TEXT,
                duration_ms INTEGER,
                created_at TEXT NOT NULL,
                started_at TEXT,
                ended_at TEXT,
                definition_version INTEGER,
                definition_snapshot_json TEXT,
                idempotency_key TEXT,
                session_id TEXT,
                validation_mode TEXT DEFAULT 'block',
                metadata_json TEXT,
                tokens_input INTEGER,
                tokens_output INTEGER,
                cost_usd REAL,
                cancel_requested INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY(workflow_id) REFERENCES workflows(id)
            );
            """
        )

        # Step runs (minimal, for human decisions later)
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS workflow_step_runs (
                step_run_id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                run_id TEXT NOT NULL,
                step_id TEXT NOT NULL,
                name TEXT,
                type TEXT,
                status TEXT,
                attempt INTEGER DEFAULT 0,
                started_at TEXT,
                ended_at TEXT,
                inputs_json TEXT,
                outputs_json TEXT,
                error TEXT,
                decision TEXT,
                assigned_to TEXT,
                approved_by TEXT,
                approved_at TEXT,
                review_comment TEXT,
                locked_by TEXT,
                locked_at TEXT,
                lock_expires_at TEXT,
                heartbeat_at TEXT,
                pid INTEGER,
                pgid INTEGER,
                workdir TEXT,
                stdout_path TEXT,
                stderr_path TEXT,
                FOREIGN KEY(run_id) REFERENCES workflow_runs(run_id) ON DELETE CASCADE
            );
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS workflow_step_attempts (
                attempt_id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                run_id TEXT NOT NULL,
                step_run_id TEXT NOT NULL,
                step_id TEXT NOT NULL,
                attempt_number INTEGER NOT NULL,
                status TEXT NOT NULL,
                reason_code_core TEXT,
                reason_code_detail TEXT,
                retryable INTEGER,
                error_summary TEXT,
                metadata_json TEXT,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                UNIQUE(step_run_id, attempt_number),
                FOREIGN KEY(run_id) REFERENCES workflow_runs(run_id) ON DELETE CASCADE,
                FOREIGN KEY(step_run_id) REFERENCES workflow_step_runs(step_run_id) ON DELETE CASCADE
            );
            """
        )

        # Events
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS workflow_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                tenant_id TEXT NOT NULL,
                run_id TEXT NOT NULL,
                step_run_id TEXT,
                event_seq INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                payload_json TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(run_id) REFERENCES workflow_runs(run_id) ON DELETE CASCADE
            );
            """
        )

        # Indices
        cur.execute("CREATE INDEX IF NOT EXISTS idx_workflows_owner ON workflows(owner_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_runs_status ON workflow_runs(status)")
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_runs_idempotency_lookup ON workflow_runs(tenant_id, user_id, idempotency_key, created_at)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_step_attempts_run_attempts "
            "ON workflow_step_attempts(run_id, attempt_number, started_at)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_step_attempts_run_step_attempts "
            "ON workflow_step_attempts(run_id, step_id, attempt_number, started_at)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_step_attempts_step_run_attempts "
            "ON workflow_step_attempts(step_run_id, attempt_number, started_at)"
        )
        # Partial indexes for frequently accessed statuses (supported on modern SQLite)
        try:
            cur.execute("CREATE INDEX IF NOT EXISTS idx_runs_status_running ON workflow_runs(status) WHERE status = 'running'")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_runs_status_queued ON workflow_runs(status) WHERE status = 'queued'")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_runs_status_succeeded ON workflow_runs(status) WHERE status = 'succeeded'")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_runs_status_failed ON workflow_runs(status) WHERE status = 'failed'")
        except _WORKFLOWS_DB_NONCRITICAL_EXCEPTIONS:
            pass
        cur.execute("CREATE INDEX IF NOT EXISTS idx_events_run_seq ON workflow_events(run_id, event_seq)")
        # Ensure uniqueness of per-run event sequence
        cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS ux_events_run_seq ON workflow_events(run_id, event_seq)")
        self._conn.commit()

        # Optional per-run event counters for SQLite
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS workflow_event_counters (
                run_id TEXT PRIMARY KEY,
                next_seq INTEGER NOT NULL
            );
            """
        )

        # Dead-letter queue for webhooks
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS workflow_webhook_dlq (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tenant_id TEXT NOT NULL,
                run_id TEXT NOT NULL,
                url TEXT NOT NULL,
                body_json TEXT,
                attempts INTEGER NOT NULL DEFAULT 0,
                next_attempt_at TEXT,
                last_error TEXT,
                created_at TEXT NOT NULL
            );
            """
        )
        self._conn.commit()

        # Attempt to add newly introduced columns if missing (SQLite tolerant pattern)
        for alter in [
            "ALTER TABLE workflow_runs ADD COLUMN tokens_input INTEGER",
            "ALTER TABLE workflow_runs ADD COLUMN tokens_output INTEGER",
            "ALTER TABLE workflow_runs ADD COLUMN cost_usd REAL",
            "ALTER TABLE workflow_runs ADD COLUMN metadata_json TEXT",
            "ALTER TABLE workflow_step_runs ADD COLUMN tenant_id TEXT",
            "ALTER TABLE workflow_step_runs ADD COLUMN assigned_to TEXT",
            "ALTER TABLE workflow_step_runs ADD COLUMN pid INTEGER",
            "ALTER TABLE workflow_step_runs ADD COLUMN pgid INTEGER",
            "ALTER TABLE workflow_step_runs ADD COLUMN workdir TEXT",
            "ALTER TABLE workflow_step_runs ADD COLUMN stdout_path TEXT",
            "ALTER TABLE workflow_step_runs ADD COLUMN stderr_path TEXT",
        ]:
            try:
                cur.execute(alter)
                self._conn.commit()
            except _WORKFLOWS_DB_NONCRITICAL_EXCEPTIONS:
                pass

        # Backfill step-run tenant_id from workflow_runs when possible
        try:
            cur.execute(
                """
                UPDATE workflow_step_runs
                SET tenant_id = (
                    SELECT tenant_id FROM workflow_runs WHERE workflow_runs.run_id = workflow_step_runs.run_id
                )
                WHERE tenant_id IS NULL OR tenant_id = ''
                """
            )
            self._conn.commit()
        except _WORKFLOWS_DB_NONCRITICAL_EXCEPTIONS:
            pass

        # Artifacts table (v0.2)
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS workflow_artifacts (
                artifact_id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                run_id TEXT NOT NULL,
                step_run_id TEXT,
                type TEXT,
                uri TEXT,
                size_bytes INTEGER,
                mime_type TEXT,
                checksum_sha256 TEXT,
                encryption TEXT,
                owned_by TEXT,
                metadata_json TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(run_id) REFERENCES workflow_runs(run_id) ON DELETE CASCADE
            );
            """
        )
        self._conn.commit()

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS workflow_research_waits (
                wait_id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                workflow_run_id TEXT NOT NULL,
                step_id TEXT NOT NULL,
                research_run_id TEXT NOT NULL,
                checkpoint_id TEXT NOT NULL,
                checkpoint_type TEXT NOT NULL,
                wait_status TEXT NOT NULL,
                wait_payload_json TEXT NOT NULL,
                active_poll_seconds REAL NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                resumed_at TEXT,
                FOREIGN KEY(workflow_run_id) REFERENCES workflow_runs(run_id) ON DELETE CASCADE
            );
            """
        )
        cur.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS ux_workflow_research_wait_run_step "
            "ON workflow_research_waits(workflow_run_id, step_id)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_workflow_research_wait_lookup "
            "ON workflow_research_waits(research_run_id, checkpoint_id, wait_status)"
        )
        self._conn.commit()

        applied_version = self._run_sqlite_migrations(
            current_version,
            self._CURRENT_SCHEMA_VERSION,
        )
        if applied_version < self._CURRENT_SCHEMA_VERSION:
            self._set_sqlite_schema_version(self._CURRENT_SCHEMA_VERSION)

    # ---------- Definitions ----------

    # SQLite write helpers with backoff to mitigate 'database is locked' under bursts
    def _sqlite_retry_execute(self, query: str, params: Any | None = None, *, max_tries: int = 5) -> None:
        import time as _time
        tries = 0
        while True:
            try:
                self._conn.execute(query, params or ())
                return
            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower() and tries < max_tries - 1:
                    _time.sleep(0.05 * (2 ** tries))
                    tries += 1
                    continue
                raise

    def _sqlite_retry_commit(self) -> None:
        import time as _time
        tries = 0
        while True:
            try:
                self._conn.commit()
                return
            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower() and tries < 4:
                    _time.sleep(0.05 * (2 ** tries))
                    tries += 1
                    continue
                raise

    def _acquire_sqlite(self) -> sqlite3.Connection:
        """Acquire a SQLite connection from pool if enabled, else return primary connection."""
        if getattr(self, "_sqlite_pool", None):
            try:
                return self._sqlite_pool.pop() if self._sqlite_pool else self._conn
            except _WORKFLOWS_DB_NONCRITICAL_EXCEPTIONS:
                return self._conn
        return self._conn

    def _release_sqlite(self, conn: sqlite3.Connection) -> None:
        if getattr(self, "_sqlite_pool", None) and conn is not self._conn:
            try:
                self._sqlite_pool.append(conn)
            except _WORKFLOWS_DB_NONCRITICAL_EXCEPTIONS:
                with contextlib.suppress(_WORKFLOWS_DB_NONCRITICAL_EXCEPTIONS):
                    conn.close()
    def create_definition(
        self,
        tenant_id: str,
        name: str,
        version: int,
        owner_id: str,
        visibility: str,
        description: str | None,
        tags: list[str] | None,
        definition: dict[str, Any],
        is_active: bool = True,
    ) -> int:
        now = _utcnow_iso()
        # For PostgreSQL backend, pass actual booleans for boolean columns;
        # SQLite accepts ints, but psycopg will map Python bool to BOOL.
        is_active_param = bool(is_active)
        params = (
            tenant_id,
            name,
            version,
            owner_id,
            visibility,
            description,
            json.dumps(tags or []),
            json.dumps(definition),
            now,
            now,
            is_active_param,
        )

        if self._using_backend():
            with self.backend.transaction() as conn:  # type: ignore[union-attr]
                result = self._execute_backend(
                    """
                    INSERT INTO workflows(tenant_id, name, version, owner_id, visibility, description, tags, definition_json, created_at, updated_at, is_active)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    params,
                    connection=conn,
                    ensure_returning=True,
                )
            row = self._row_from_result(result)
            if row:
                return int(row["id"])
            if result.lastrowid is not None:
                return int(result.lastrowid)
            raise WorkflowsSchemaError("Failed to retrieve workflow id after insert")

        cur = self._conn.cursor()
        cur.execute(
            """
            INSERT INTO workflows(tenant_id, name, version, owner_id, visibility, description, tags, definition_json, created_at, updated_at, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            params,
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def get_definition(self, workflow_id: int) -> WorkflowDefinition | None:
        if self._using_backend():
            with self.backend.transaction() as conn:  # type: ignore[union-attr]
                result = self._execute_backend(
                    "SELECT * FROM workflows WHERE id = ?",
                    (workflow_id,),
                    connection=conn,
                )
            row = self._row_from_result(result)
            if not row:
                return None
            return WorkflowDefinition(**row.to_dict())

        row = self._conn.cursor().execute("SELECT * FROM workflows WHERE id = ?", (workflow_id,)).fetchone()
        if not row:
            return None
        return WorkflowDefinition(**dict(row))

    def list_definitions(
        self, tenant_id: str | None = None, owner_id: str | None = None, include_inactive: bool = False
    ) -> list[WorkflowDefinition]:
        sql = "SELECT * FROM workflows WHERE 1=1"
        params: list[Any] = []
        if tenant_id:
            sql += " AND tenant_id = ?"
            params.append(tenant_id)
        if owner_id:
            sql += " AND owner_id = ?"
            params.append(owner_id)
        if not include_inactive:
            sql += " AND is_active = 1"
        sql += " ORDER BY name, version DESC"
        if self._using_backend():
            with self.backend.transaction() as conn:  # type: ignore[union-attr]
                result = self._execute_backend(sql, tuple(params), connection=conn)
            rows = self._rows_from_result(result)
            return [WorkflowDefinition(**row.to_dict()) for row in rows]

        rows = self._conn.cursor().execute(sql, params).fetchall()
        return [WorkflowDefinition(**dict(r)) for r in rows]

    def soft_delete_definition(self, workflow_id: int) -> bool:
        params = (_utcnow_iso(), workflow_id)
        if self._using_backend():
            with self.backend.transaction() as conn:  # type: ignore[union-attr]
                result = self._execute_backend(
                    "UPDATE workflows SET is_active = 0, updated_at = ? WHERE id = ?",
                    params,
                    connection=conn,
                )
            return result.rowcount > 0

        cur = self._conn.cursor()
        cur.execute("UPDATE workflows SET is_active = 0, updated_at = ? WHERE id = ?", params)
        self._conn.commit()
        return cur.rowcount > 0

    # ---------- Runs ----------
    def create_run(
        self,
        run_id: str,
        tenant_id: str,
        user_id: str,
        inputs: dict[str, Any],
        workflow_id: int | None = None,
        definition_version: int | None = None,
        definition_snapshot: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
        session_id: str | None = None,
        validation_mode: str = "block",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        metadata_json = json.dumps(metadata or {})
        params = (
            run_id,
            tenant_id,
            workflow_id,
            user_id,
            json.dumps(inputs or {}),
            _utcnow_iso(),
            definition_version,
            json.dumps(definition_snapshot) if definition_snapshot else None,
            idempotency_key,
            session_id,
            validation_mode,
            metadata_json,
        )

        query = """
            INSERT INTO workflow_runs(
                run_id, tenant_id, workflow_id, status, status_reason, user_id, inputs_json, outputs_json,
                error, duration_ms, created_at, started_at, ended_at, definition_version, definition_snapshot_json,
                idempotency_key, session_id, validation_mode, metadata_json
            ) VALUES (?, ?, ?, 'queued', NULL, ?, ?, NULL, NULL, NULL, ?, NULL, NULL, ?, ?, ?, ?, ?, ?)
        """

        if self._using_backend():
            with self.backend.transaction() as conn:  # type: ignore[union-attr]
                self._execute_backend(query, params, connection=conn)
            return

        try:
            self._conn.execute(query, params)
            self._conn.commit()
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower():
                # Retry with backoff on lock contention
                self._sqlite_retry_execute(query, params)
                self._sqlite_retry_commit()
            else:
                raise

    def get_run(self, run_id: str) -> WorkflowRun | None:
        # Defensive conversion to avoid sqlite binding errors when callers pass UUID/None
        run_id_param = "" if run_id is None else str(run_id)
        if self._using_backend():
            with self.backend.transaction() as conn:  # type: ignore[union-attr]
                result = self._execute_backend(
                    "SELECT * FROM workflow_runs WHERE run_id = ?",
                    (run_id_param,),
                    connection=conn,
                )
            row = self._row_from_result(result)
            return WorkflowRun(**row.to_dict()) if row else None

        conn = self._acquire_sqlite()
        try:
            row = conn.cursor().execute("SELECT * FROM workflow_runs WHERE run_id = ?", (run_id_param,)).fetchone()
            return WorkflowRun(**dict(row)) if row else None
        finally:
            self._release_sqlite(conn)

    def list_runs(
        self,
        *,
        tenant_id: str,
        user_id: str | None = None,
        statuses: list[str] | None = None,
        workflow_id: int | None = None,
        created_after: str | None = None,
        created_before: str | None = None,
        cursor_ts: str | None = None,
        cursor_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
        order_by: str = "created_at",
        order_desc: bool = True,
    ) -> list[WorkflowRun]:
        sql = "SELECT * FROM workflow_runs WHERE tenant_id = ?"
        params: list[Any] = [tenant_id]
        if user_id:
            sql += " AND user_id = ?"
            params.append(user_id)
        if statuses:
            placeholders = ",".join(["?"] * len(statuses))
            sql += f" AND status IN ({placeholders})"
            params.extend(list(statuses))
        if workflow_id is not None:
            sql += " AND workflow_id = ?"
            params.append(int(workflow_id))
        if created_after:
            sql += " AND created_at >= ?"
            params.append(created_after)
        if created_before:
            sql += " AND created_at <= ?"
            params.append(created_before)
        # Whitelist order_by to known columns
        allowed_order = {"created_at", "started_at", "ended_at"}
        ob = order_by if order_by in allowed_order else "created_at"
        # Apply cursor seek if provided (seek pagination)
        if cursor_ts and cursor_id:
            cmp = "<" if order_desc else ">"
            # Add tie-breaker on run_id; for DESC use run_id < last_id if same ts, for ASC use run_id > last_id
            tcmp = "<" if order_desc else ">"
            sql += f" AND (({ob} {cmp} ?) OR ({ob} = ? AND run_id {tcmp} ?))"
            params.extend([cursor_ts, cursor_ts, cursor_id])
            # When using cursor, ignore numeric offset to avoid skipping
            sql += f" ORDER BY {ob} {'DESC' if order_desc else 'ASC'}, run_id {'DESC' if order_desc else 'ASC'} LIMIT ?"
            params.extend([int(limit)])
        else:
            # Stable ordering with tie-breaker by run_id
            sql += f" ORDER BY {ob} {'DESC' if order_desc else 'ASC'}, run_id {'DESC' if order_desc else 'ASC'} LIMIT ? OFFSET ?"
            params.extend([int(limit), int(offset)])

        if self._using_backend():
            with self.backend.transaction() as conn:  # type: ignore[union-attr]
                result = self._execute_backend(sql, tuple(params), connection=conn)
            rows = self._rows_from_result(result)
            return [WorkflowRun(**row.to_dict()) for row in rows]

        cur = self._conn.cursor()
        rows = cur.execute(sql, params).fetchall()
        return [WorkflowRun(**dict(r)) for r in rows]

    # ---------- Quotas / Usage ----------
    def count_runs_for_user_window(
        self,
        *,
        tenant_id: str,
        user_id: str,
        window_start_iso: str,
        window_end_iso: str | None = None,
    ) -> int:
        """Count runs created by a user within an ISO time window [start, end]."""
        sql = "SELECT COUNT(*) AS c FROM workflow_runs WHERE tenant_id = ? AND user_id = ? AND created_at >= ?"
        params: list[Any] = [tenant_id, user_id, window_start_iso]
        if window_end_iso:
            sql += " AND created_at < ?"
            params.append(window_end_iso)

        if self._using_backend():
            with self.backend.transaction() as conn:  # type: ignore[union-attr]
                result = self._execute_backend(sql, tuple(params), connection=conn)
            row = self._row_from_result(result)
            try:
                return int(row[0]) if row is not None else 0
            except _WORKFLOWS_DB_NONCRITICAL_EXCEPTIONS:
                return int((row.get("c") if row else 0) or 0)

        cur = self._conn.cursor()
        row = cur.execute(sql, params).fetchone()
        if not row:
            return 0
        try:
            return int(row[0])
        except _WORKFLOWS_DB_NONCRITICAL_EXCEPTIONS:
            try:
                return int(row.get("c") or 0)  # type: ignore[attr-defined]
            except _WORKFLOWS_DB_NONCRITICAL_EXCEPTIONS:
                return 0

    def count_runs_for_tenant_window(
        self,
        *,
        tenant_id: str,
        window_start_iso: str,
        window_end_iso: str | None = None,
    ) -> int:
        """Count runs created within a tenant over a time window."""
        sql = "SELECT COUNT(*) AS c FROM workflow_runs WHERE tenant_id = ? AND created_at >= ?"
        params: list[Any] = [tenant_id, window_start_iso]
        if window_end_iso:
            sql += " AND created_at < ?"
            params.append(window_end_iso)

        if self._using_backend():
            with self.backend.transaction() as conn:  # type: ignore[union-attr]
                result = self._execute_backend(sql, tuple(params), connection=conn)
            row = self._row_from_result(result)
            try:
                return int(row[0]) if row is not None else 0
            except _WORKFLOWS_DB_NONCRITICAL_EXCEPTIONS:
                return int((row.get("c") if row else 0) or 0)

        cur = self._conn.cursor()
        row = cur.execute(sql, params).fetchone()
        if not row:
            return 0
        try:
            return int(row[0])
        except _WORKFLOWS_DB_NONCRITICAL_EXCEPTIONS:
            try:
                return int(row.get("c") or 0)  # type: ignore[attr-defined]
            except _WORKFLOWS_DB_NONCRITICAL_EXCEPTIONS:
                return 0

    def update_run_status(
        self,
        run_id: str,
        status: str,
        status_reason: str | None = None,
        outputs: dict[str, Any] | None = None,
        error: str | None = None,
        started_at: str | None = None,
        ended_at: str | None = None,
        duration_ms: int | None = None,
        tokens_input: int | None = None,
        tokens_output: int | None = None,
        cost_usd: float | None = None,
        *,
        connection: Any = None,
    ) -> None:
        params = (
            status,
            status_reason,
            json.dumps(outputs) if outputs is not None else None,
            error,
            started_at,
            ended_at,
            duration_ms,
            tokens_input,
            tokens_output,
            cost_usd,
            run_id,
        )

        query = """
            UPDATE workflow_runs
            SET status = ?, status_reason = ?, outputs_json = ?, error = ?,
                started_at = COALESCE(?, started_at), ended_at = COALESCE(?, ended_at),
                duration_ms = COALESCE(?, duration_ms),
                tokens_input = COALESCE(?, tokens_input),
                tokens_output = COALESCE(?, tokens_output),
                cost_usd = COALESCE(?, cost_usd)
            WHERE run_id = ?
        """

        if self._using_backend():
            if connection is None:
                with self.backend.transaction() as conn:  # type: ignore[union-attr]
                    self._execute_backend(query, params, connection=conn)
            else:
                self._execute_backend(query, params, connection=connection)
        else:
            try:
                self._conn.execute(query, params)
                self._conn.commit()
            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower():
                    self._sqlite_retry_execute(query, params)
                    self._sqlite_retry_commit()
                else:
                    raise
        try:
            from loguru import logger as _logger
            _logger.debug(f"WorkflowsDB: run {run_id} -> status={status}")
        except _WORKFLOWS_DB_NONCRITICAL_EXCEPTIONS:
            pass

    # ---------- Run control ----------
    def set_cancel_requested(self, run_id: str, cancel: bool = True) -> None:
        params = (bool(cancel), run_id)
        if self._using_backend():
            with self.backend.transaction() as conn:  # type: ignore[union-attr]
                self._execute_backend(
                    "UPDATE workflow_runs SET cancel_requested = ? WHERE run_id = ?",
                    params,
                    connection=conn,
                )
            return

        try:
            self._conn.execute("UPDATE workflow_runs SET cancel_requested = ? WHERE run_id = ?", params)
            self._conn.commit()
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower():
                self._sqlite_retry_execute("UPDATE workflow_runs SET cancel_requested = ? WHERE run_id = ?", params)
                self._sqlite_retry_commit()
            else:
                raise

    def is_cancel_requested(self, run_id: str) -> bool:
        if self._using_backend():
            with self.backend.transaction() as conn:  # type: ignore[union-attr]
                result = self._execute_backend(
                    "SELECT cancel_requested FROM workflow_runs WHERE run_id = ?",
                    (run_id,),
                    connection=conn,
                )
            row = self._row_from_result(result)
            return bool(row[0]) if row else False

        row = self._conn.cursor().execute(
            "SELECT cancel_requested FROM workflow_runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        return bool(row[0]) if row else False

    # ---------- Events ----------
    def append_event(
        self,
        tenant_id: str,
        run_id: str,
        event_type: str,
        payload: dict[str, Any] | None = None,
        step_run_id: str | None = None,
        *,
        connection: Any = None,
    ) -> int:
        # Prefer per-run counters when available
        if self._using_backend():
            def _append_with_connection(conn: Any) -> int:
                # Increment or initialize per-run counter atomically
                try:
                    # Use upsert to bump counter and read back the new value
                    inc = self._execute_backend(
                        """
                        INSERT INTO workflow_event_counters(run_id, next_seq)
                        VALUES (?, 1)
                        ON CONFLICT (run_id) DO UPDATE SET next_seq = workflow_event_counters.next_seq + 1
                        RETURNING next_seq
                        """,
                        (run_id,),
                        connection=conn,
                    )
                    r = self._row_from_result(inc)
                    next_seq = int(r["next_seq"]) if r else 1
                except _WORKFLOWS_DB_NONCRITICAL_EXCEPTIONS:
                    # Fallback to aggregate
                    seq_result = self._execute_backend(
                        "SELECT COALESCE(MAX(event_seq), 0) AS max_seq FROM workflow_events WHERE run_id = ?",
                        (run_id,),
                        connection=conn,
                    )
                    row = self._row_from_result(seq_result)
                    max_seq = int(row["max_seq"]) if row else 0
                    next_seq = max_seq + 1
                self._execute_backend(
                    """
                    INSERT INTO workflow_events(tenant_id, run_id, step_run_id, event_seq, event_type, payload_json, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        tenant_id,
                        run_id,
                        step_run_id,
                        next_seq,
                        event_type,
                        json.dumps(payload or {}),
                        _utcnow_iso(),
                    ),
                    connection=conn,
                )
                return next_seq

            if connection is None:
                with self.backend.transaction() as conn:  # type: ignore[union-attr]
                    return _append_with_connection(conn)
            return _append_with_connection(connection)

        # SQLite path
        conn = self._acquire_sqlite()
        try:
            tries = 0
            while True:
                try:
                    conn.execute("BEGIN IMMEDIATE")
                    break
                except sqlite3.OperationalError as e:
                    if "locked" in str(e).lower() and tries < 4:
                        import time as _time

                        _time.sleep(0.05 * (2 ** tries))
                        tries += 1
                        continue
                    raise

            cur = conn.cursor()
            try:
                # Try per-run counter with an explicit write transaction so
                # the counter increment and event insert share one serialized
                # critical section on SQLite.
                row = cur.execute(
                    """
                    INSERT INTO workflow_event_counters(run_id, next_seq)
                    VALUES (?, 1)
                    ON CONFLICT(run_id) DO UPDATE SET next_seq = workflow_event_counters.next_seq + 1
                    RETURNING next_seq
                    """,
                    (run_id,),
                ).fetchone()
                next_seq = int(row["next_seq"] if isinstance(row, dict) else row[0]) if row else 1
            except _WORKFLOWS_DB_NONCRITICAL_EXCEPTIONS:
                # Fallback for older SQLite builds or partially migrated tables.
                row = cur.execute(
                    "SELECT next_seq FROM workflow_event_counters WHERE run_id = ?",
                    (run_id,),
                ).fetchone()
                if not row:
                    next_seq = 1
                    cur.execute(
                        "INSERT OR IGNORE INTO workflow_event_counters(run_id, next_seq) VALUES (?, ?)",
                        (run_id, next_seq),
                    )
                else:
                    current = int(row[0] if not isinstance(row, dict) else row.get("next_seq", 0))
                    next_seq = current + 1
                    cur.execute(
                        "UPDATE workflow_event_counters SET next_seq = ? WHERE run_id = ?",
                        (next_seq, run_id),
                    )

            params_insert = (
                tenant_id,
                run_id,
                step_run_id,
                next_seq,
                event_type,
                json.dumps(payload or {}),
                _utcnow_iso(),
            )
            cur.execute(
                """
                INSERT INTO workflow_events(tenant_id, run_id, step_run_id, event_seq, event_type, payload_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                params_insert,
            )
            conn.commit()
            return next_seq
        except (sqlite3.Error, TypeError, ValueError):
            with contextlib.suppress(sqlite3.Error):
                conn.rollback()
            raise
        finally:
            self._release_sqlite(conn)

    def get_events(self, run_id: str, since: int | None = None, limit: int = 500, types: list[str] | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM workflow_events WHERE run_id = ?"
        params: list[Any] = [run_id]
        if since is not None:
            sql += " AND event_seq > ?"
            params.append(int(since))
        if types:
            placeholders = ",".join(["?"] * len(types))
            sql += f" AND event_type IN ({placeholders})"
            params.extend(list(types))
        # Stable ordering: primary by event_seq (per-run unique), tie-breaker by event_id
        sql += " ORDER BY event_seq ASC, event_id ASC LIMIT ?"
        params.append(int(limit))
        if self._using_backend():
            with self.backend.transaction() as conn:  # type: ignore[union-attr]
                result = self._execute_backend(sql, tuple(params), connection=conn)
            rows = self._rows_from_result(result)
        else:
            conn = self._acquire_sqlite()
            try:
                rows = conn.cursor().execute(sql, params).fetchall()
            finally:
                self._release_sqlite(conn)

        out: list[dict[str, Any]] = []
        for r in rows:
            data = self._row_to_dict(r)
            with contextlib.suppress(_WORKFLOWS_DB_NONCRITICAL_EXCEPTIONS):
                data["payload_json"] = json.loads(data.get("payload_json") or "{}")
            out.append(data)
        return out

    # ---------- Step Runs ----------
    def create_step_run(
        self,
        *,
        step_run_id: str,
        tenant_id: str,
        run_id: str,
        step_id: str,
        name: str,
        step_type: str,
        status: str = "running",
        inputs: dict[str, Any] | None = None,
        assigned_to: str | None = None,
    ) -> None:
        params = (
            step_run_id,
            tenant_id,
            run_id,
            step_id,
            name,
            step_type,
            status,
            _utcnow_iso(),
            json.dumps(inputs or {}),
            assigned_to,
        )

        query = """
            INSERT INTO workflow_step_runs(
                step_run_id, tenant_id, run_id, step_id, name, type, status, started_at, inputs_json, assigned_to
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        if self._using_backend():
            with self.backend.transaction() as conn:  # type: ignore[union-attr]
                self._execute_backend(query, params, connection=conn)
            return

        try:
            self._conn.execute(query, params)
            self._conn.commit()
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower():
                self._sqlite_retry_execute(query, params)
                self._sqlite_retry_commit()
            else:
                raise

    def complete_step_run(
        self,
        *,
        step_run_id: str,
        status: str = "succeeded",
        outputs: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        params = (
            status,
            _utcnow_iso(),
            json.dumps(outputs or {}),
            error,
            step_run_id,
        )
        query = """
            UPDATE workflow_step_runs
            SET status = ?, ended_at = ?, outputs_json = ?, error = ?
            WHERE step_run_id = ?
        """

        if self._using_backend():
            with self.backend.transaction() as conn:  # type: ignore[union-attr]
                self._execute_backend(query, params, connection=conn)
            return

        try:
            self._conn.execute(query, params)
            self._conn.commit()
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower():
                self._sqlite_retry_execute(query, params)
                self._sqlite_retry_commit()
            else:
                raise

    def get_latest_step_run(self, *, run_id: str, step_id: str) -> dict[str, Any] | None:
        """Return the most recent step run for a run/step_id pair."""
        query = """
            SELECT * FROM workflow_step_runs
            WHERE run_id = ? AND step_id = ?
            ORDER BY started_at DESC
            LIMIT 1
        """
        params = (str(run_id), str(step_id))
        if self._using_backend():
            with self.backend.transaction() as conn:  # type: ignore[union-attr]
                result = self._execute_backend(query, params, connection=conn)
            row = self._row_from_result(result)
            return row.to_dict() if row else None

        conn = self._acquire_sqlite()
        try:
            row = conn.cursor().execute(query, params).fetchone()
            return dict(row) if row else None
        finally:
            self._release_sqlite(conn)

    def list_step_runs(self, *, run_id: str) -> list[dict[str, Any]]:
        """Return step runs for a workflow run ordered by creation time."""
        query = """
            SELECT * FROM workflow_step_runs
            WHERE run_id = ?
            ORDER BY started_at ASC, step_run_id ASC
        """
        params = (str(run_id),)
        if self._using_backend():
            with self.backend.transaction() as conn:  # type: ignore[union-attr]
                result = self._execute_backend(query, params, connection=conn)
            rows = self._rows_from_result(result)
        else:
            rows = self._conn.cursor().execute(query, params).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def update_step_attempt(self, *, step_run_id: str, attempt: int) -> None:
        """Persist the current attempt count for a step run."""
        params = (int(attempt), step_run_id)
        if self._using_backend():
            with self.backend.transaction() as conn:  # type: ignore[union-attr]
                self._execute_backend(
                    "UPDATE workflow_step_runs SET attempt = ? WHERE step_run_id = ?",
                    params,
                    connection=conn,
                )
            return

        try:
            self._conn.execute(
                "UPDATE workflow_step_runs SET attempt = ? WHERE step_run_id = ?",
                params,
            )
            self._conn.commit()
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower():
                self._sqlite_retry_execute("UPDATE workflow_step_runs SET attempt = ? WHERE step_run_id = ?", params)
                self._sqlite_retry_commit()
            else:
                raise

    def create_step_attempt(
        self,
        *,
        tenant_id: str,
        run_id: str,
        step_run_id: str,
        step_id: str,
        attempt_number: int,
        status: str = "running",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        step_run_id_value = str(step_run_id or "").strip()
        if not step_run_id_value:
            raise ValueError("step_run_id is required for workflow step attempts")
        attempt_id = str(uuid4())
        params = (
            attempt_id,
            tenant_id,
            run_id,
            step_run_id_value,
            step_id,
            int(attempt_number),
            status,
            json.dumps(metadata or {}),
            _utcnow_iso(),
        )
        query = """
            INSERT INTO workflow_step_attempts(
                attempt_id, tenant_id, run_id, step_run_id, step_id, attempt_number, status, metadata_json, started_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        if self._using_backend():
            with self.backend.transaction() as conn:  # type: ignore[union-attr]
                self._execute_backend(query, params, connection=conn)
            return attempt_id

        try:
            self._conn.execute(query, params)
            self._conn.commit()
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower():
                self._sqlite_retry_execute(query, params)
                self._sqlite_retry_commit()
            else:
                raise
        return attempt_id

    def complete_step_attempt(
        self,
        *,
        attempt_id: str,
        status: str,
        reason_code_core: str | None = None,
        reason_code_detail: str | None = None,
        retryable: bool | None = None,
        error_summary: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        retryable_value = None if retryable is None else bool(retryable)
        metadata_json = None if metadata is None else json.dumps(metadata)
        params = (
            status,
            reason_code_core,
            reason_code_detail,
            retryable_value,
            error_summary,
            metadata_json,
            _utcnow_iso(),
            attempt_id,
        )
        query = """
            UPDATE workflow_step_attempts
            SET status = ?,
                reason_code_core = ?,
                reason_code_detail = ?,
                retryable = ?,
                error_summary = ?,
                metadata_json = COALESCE(?, metadata_json),
                ended_at = ?
            WHERE attempt_id = ?
        """

        if self._using_backend():
            with self.backend.transaction() as conn:  # type: ignore[union-attr]
                self._execute_backend(query, params, connection=conn)
            return

        try:
            self._conn.execute(query, params)
            self._conn.commit()
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower():
                self._sqlite_retry_execute(query, params)
                self._sqlite_retry_commit()
            else:
                raise

    def list_step_attempts(
        self,
        *,
        run_id: str,
        step_id: str | None = None,
        step_run_id: str | None = None,
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM workflow_step_attempts WHERE run_id = ?"
        params: list[Any] = [str(run_id)]
        if step_id is not None:
            sql += " AND step_id = ?"
            params.append(str(step_id))
        if step_run_id is not None:
            sql += " AND step_run_id = ?"
            params.append(str(step_run_id))
        sql += " ORDER BY attempt_number ASC, started_at ASC, attempt_id ASC"

        if self._using_backend():
            with self.backend.transaction() as conn:  # type: ignore[union-attr]
                result = self._execute_backend(sql, tuple(params), connection=conn)
            rows = self._rows_from_result(result)
        else:
            rows = self._conn.cursor().execute(sql, params).fetchall()

        attempts: list[dict[str, Any]] = []
        for row in rows:
            data = self._row_to_dict(row)
            metadata_raw = data.get("metadata_json")
            if isinstance(metadata_raw, (dict, list)):
                data["metadata_json"] = metadata_raw
            elif not metadata_raw:
                data["metadata_json"] = {}
            else:
                try:
                    data["metadata_json"] = json.loads(str(metadata_raw))
                except _WORKFLOWS_DB_NONCRITICAL_EXCEPTIONS as exc:
                    logger.warning(
                        "Workflows DB: malformed step attempt metadata_json run_id={} attempt_id={}: {}",
                        run_id,
                        data.get("attempt_id"),
                        exc,
                    )
                    data["metadata_json"] = {}
            attempts.append(data)
        return attempts

    def get_last_failed_step_id(self, run_id: str) -> str | None:
        query = (
            "SELECT step_id FROM workflow_step_runs WHERE run_id = ? AND status = 'failed' "
            "ORDER BY ended_at DESC LIMIT 1"
        )
        if self._using_backend():
            with self.backend.transaction() as conn:  # type: ignore[union-attr]
                result = self._execute_backend(query, (run_id,), connection=conn)
            row = self._row_from_result(result)
            return row[0] if row else None

        row = self._conn.cursor().execute(query, (run_id,)).fetchone()
        return row[0] if row else None

    def get_last_completed_step_run(
        self,
        *,
        run_id: str,
        before_ts: str | None = None,
    ) -> dict[str, Any] | None:
        """Return the most recent succeeded step run, optionally before a timestamp."""
        query = "SELECT * FROM workflow_step_runs WHERE run_id = ? AND status = 'succeeded'"
        params: list[Any] = [str(run_id)]
        if before_ts:
            query += " AND ended_at IS NOT NULL AND ended_at < ?"
            params.append(str(before_ts))
        query += " ORDER BY ended_at DESC LIMIT 1"
        if self._using_backend():
            with self.backend.transaction() as conn:  # type: ignore[union-attr]
                result = self._execute_backend(query, tuple(params), connection=conn)
            row = self._row_from_result(result)
            return row.to_dict() if row else None

        row = self._conn.cursor().execute(query, params).fetchone()
        return dict(row) if row else None

    def aggregate_run_token_usage(self, run_id: str) -> tuple[int | None, int | None, float | None]:
        """Aggregate token usage and cost across step outputs for a run."""
        query = "SELECT outputs_json FROM workflow_step_runs WHERE run_id = ? AND outputs_json IS NOT NULL"
        rows: list[Any]
        if self._using_backend():
            with self.backend.transaction() as conn:  # type: ignore[union-attr]
                result = self._execute_backend(query, (str(run_id),), connection=conn)
            rows = self._rows_from_result(result)
        else:
            rows = self._conn.cursor().execute(query, (str(run_id),)).fetchall()

        total_in = 0
        total_out = 0
        total_cost = 0.0
        have_tokens = False
        have_cost = False

        def _as_int(val: Any) -> int | None:
            try:
                if val is None:
                    return None
                return int(val)
            except _WORKFLOWS_DB_NONCRITICAL_EXCEPTIONS:
                return None

        def _as_float(val: Any) -> float | None:
            try:
                if val is None:
                    return None
                return float(val)
            except _WORKFLOWS_DB_NONCRITICAL_EXCEPTIONS:
                return None

        for row in rows:
            raw = None
            try:
                if isinstance(row, (WorkflowRowAdapter, dict)):
                    raw = row.get("outputs_json")
                else:
                    raw = row[0] if row else None
            except _WORKFLOWS_DB_NONCRITICAL_EXCEPTIONS:
                raw = None
            if raw is None:
                continue
            outputs: Any = raw
            if isinstance(raw, (bytes, bytearray)):
                try:
                    outputs = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    logger.debug("Skipping malformed outputs_json for run_id={}", run_id)
                    continue
            elif isinstance(raw, str):
                try:
                    outputs = json.loads(raw)
                except json.JSONDecodeError:
                    logger.debug("Skipping malformed outputs_json for run_id={}", run_id)
                    continue
            if not isinstance(outputs, dict):
                continue
            meta = outputs.get("metadata") if isinstance(outputs.get("metadata"), dict) else None
            usage = None
            if isinstance(meta, dict):
                usage = meta.get("token_usage") or meta.get("usage")
            if usage is None and isinstance(outputs.get("token_usage"), dict):
                usage = outputs.get("token_usage")
            if isinstance(usage, dict):
                in_val = _as_int(usage.get("prompt_tokens") or usage.get("input_tokens"))
                out_val = _as_int(usage.get("completion_tokens") or usage.get("output_tokens"))
                if in_val is not None:
                    total_in += in_val
                    have_tokens = True
                if out_val is not None:
                    total_out += out_val
                    have_tokens = True
            cost_val = None
            if isinstance(meta, dict) and meta.get("cost_usd") is not None:
                cost_val = _as_float(meta.get("cost_usd"))
            elif outputs.get("cost_usd") is not None:
                cost_val = _as_float(outputs.get("cost_usd"))
            if cost_val is not None:
                total_cost += cost_val
                have_cost = True

        return (
            total_in if have_tokens else None,
            total_out if have_tokens else None,
            total_cost if have_cost else None,
        )

    def get_run_by_idempotency(self, tenant_id: str, user_id: str, idempotency_key: str) -> WorkflowRun | None:
        ttl_hours = _workflows_idempotency_ttl_hours()
        try:
            from datetime import timedelta
            cutoff = datetime.utcnow().replace(tzinfo=timezone.utc) - timedelta(hours=ttl_hours)
            cutoff_iso = cutoff.isoformat()
        except _WORKFLOWS_DB_NONCRITICAL_EXCEPTIONS:
            cutoff_iso = None
        params = (tenant_id, user_id, idempotency_key)
        query = "SELECT * FROM workflow_runs WHERE tenant_id = ? AND user_id = ? AND idempotency_key = ?"
        if cutoff_iso:
            query += " AND created_at >= ?"
            params = (*params, cutoff_iso)
        query += " ORDER BY created_at DESC, run_id DESC LIMIT 1"
        if self._using_backend():
            with self.backend.transaction() as conn:  # type: ignore[union-attr]
                result = self._execute_backend(query, params, connection=conn)
            row = self._row_from_result(result)
            return WorkflowRun(**row.to_dict()) if row else None

        row = self._conn.cursor().execute(query, params).fetchone()
        return WorkflowRun(**dict(row)) if row else None

    def update_step_lock_and_heartbeat(
        self,
        *,
        step_run_id: str,
        locked_by: str | None = None,
        lock_ttl_seconds: int | None = None,
    ) -> None:
        now = datetime.utcnow().replace(tzinfo=timezone.utc)
        locked_at = now.isoformat()
        lock_expires_at = None
        if lock_ttl_seconds is not None:
            lock_expires_at = (now + __import__("datetime").timedelta(seconds=lock_ttl_seconds)).isoformat()
        params = (
            locked_by,
            locked_at,
            lock_expires_at,
            locked_at,
            step_run_id,
        )
        query = """
            UPDATE workflow_step_runs
            SET locked_by = COALESCE(?, locked_by), locked_at = ?, lock_expires_at = COALESCE(?, lock_expires_at), heartbeat_at = ?
            WHERE step_run_id = ?
        """

        if self._using_backend():
            with self.backend.transaction() as conn:  # type: ignore[union-attr]
                self._execute_backend(query, params, connection=conn)
            return

        try:
            self._conn.execute(query, params)
            self._conn.commit()
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower():
                self._sqlite_retry_execute(query, params)
                self._sqlite_retry_commit()
            else:
                raise
        try:
            from loguru import logger as _logger
            _logger.debug(f"WorkflowsDB: heartbeat step_run_id={step_run_id}")
        except _WORKFLOWS_DB_NONCRITICAL_EXCEPTIONS:
            pass

    def find_orphan_step_runs(self, cutoff_iso: str) -> list[dict[str, Any]]:
        sql = (
            "SELECT * FROM workflow_step_runs WHERE status = 'running' AND (heartbeat_at IS NULL OR heartbeat_at < ?)"
        )
        if self._using_backend():
            with self.backend.transaction() as conn:  # type: ignore[union-attr]
                result = self._execute_backend(sql, (cutoff_iso,), connection=conn)
            rows = self._rows_from_result(result)
        else:
            rows = self._conn.cursor().execute(sql, (cutoff_iso,)).fetchall()
        return [self._row_to_dict(r) for r in rows]

    # ---------- Subprocess tracking ----------
    def update_step_subprocess(
        self,
        *,
        step_run_id: str,
        pid: int | None = None,
        pgid: int | None = None,
        workdir: str | None = None,
        stdout_path: str | None = None,
        stderr_path: str | None = None,
    ) -> None:
        params = (
            pid,
            pgid,
            workdir,
            stdout_path,
            stderr_path,
            step_run_id,
        )
        query = """
            UPDATE workflow_step_runs
            SET pid = COALESCE(?, pid), pgid = COALESCE(?, pgid), workdir = COALESCE(?, workdir),
                stdout_path = COALESCE(?, stdout_path), stderr_path = COALESCE(?, stderr_path)
            WHERE step_run_id = ?
        """

        if self._using_backend():
            with self.backend.transaction() as conn:  # type: ignore[union-attr]
                self._execute_backend(query, params, connection=conn)
            return

        try:
            self._conn.execute(query, params)
            self._conn.commit()
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower():
                self._sqlite_retry_execute(query, params)
                self._sqlite_retry_commit()
            else:
                raise

    def find_running_subprocesses_for_run(self, run_id: str) -> list[dict[str, Any]]:
        sql = (
            "SELECT step_run_id, pid, pgid, workdir, stdout_path, stderr_path FROM workflow_step_runs "
            "WHERE run_id = ? AND status = 'running' AND (pid IS NOT NULL OR pgid IS NOT NULL)"
        )
        if self._using_backend():
            with self.backend.transaction() as conn:  # type: ignore[union-attr]
                result = self._execute_backend(sql, (run_id,), connection=conn)
            rows = self._rows_from_result(result)
        else:
            rows = self._conn.cursor().execute(sql, (run_id,)).fetchall()
        return [self._row_to_dict(r) for r in rows]

    # ---------- Artifacts ----------
    def add_artifact(
        self,
        *,
        artifact_id: str,
        tenant_id: str,
        run_id: str,
        step_run_id: str | None,
        type: str,
        uri: str,
        size_bytes: int | None = None,
        mime_type: str | None = None,
        checksum_sha256: str | None = None,
        encryption: str | None = None,
        owned_by: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        params = (
            artifact_id,
            tenant_id,
            run_id,
            step_run_id,
            type,
            uri,
            size_bytes,
            mime_type,
            checksum_sha256,
            encryption,
            owned_by,
            json.dumps(metadata or {}),
            _utcnow_iso(),
        )
        query = """
            INSERT INTO workflow_artifacts(
                artifact_id, tenant_id, run_id, step_run_id, type, uri, size_bytes, mime_type, checksum_sha256,
                encryption, owned_by, metadata_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        if self._using_backend():
            with self.backend.transaction() as conn:  # type: ignore[union-attr]
                self._execute_backend(query, params, connection=conn)
            return

        try:
            self._conn.execute(query, params)
            self._conn.commit()
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower():
                self._sqlite_retry_execute(query, params)
                self._sqlite_retry_commit()
            else:
                raise

    # ---------- Research wait links ----------
    def upsert_research_wait_link(
        self,
        *,
        wait_id: str,
        tenant_id: str,
        workflow_run_id: str,
        step_id: str,
        research_run_id: str,
        checkpoint_id: str,
        checkpoint_type: str,
        wait_status: str,
        wait_payload: dict[str, Any],
        active_poll_seconds: float,
    ) -> None:
        now = _utcnow_iso()
        params = (
            str(wait_id),
            str(tenant_id),
            str(workflow_run_id),
            str(step_id),
            str(research_run_id),
            str(checkpoint_id),
            str(checkpoint_type),
            str(wait_status),
            json.dumps(wait_payload or {}),
            float(active_poll_seconds),
            now,
            now,
        )
        query = """
            INSERT INTO workflow_research_waits(
                wait_id,
                tenant_id,
                workflow_run_id,
                step_id,
                research_run_id,
                checkpoint_id,
                checkpoint_type,
                wait_status,
                wait_payload_json,
                active_poll_seconds,
                created_at,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(workflow_run_id, step_id) DO UPDATE SET
                research_run_id = excluded.research_run_id,
                checkpoint_id = excluded.checkpoint_id,
                checkpoint_type = excluded.checkpoint_type,
                wait_status = excluded.wait_status,
                wait_payload_json = excluded.wait_payload_json,
                active_poll_seconds = excluded.active_poll_seconds,
                updated_at = excluded.updated_at
        """

        if self._using_backend():
            with self.backend.transaction() as conn:  # type: ignore[union-attr]
                self._execute_backend(query, params, connection=conn)
            return

        try:
            self._conn.execute(query, params)
            self._conn.commit()
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower():
                self._sqlite_retry_execute(query, params)
                self._sqlite_retry_commit()
            else:
                raise

    def get_research_wait_link(self, *, workflow_run_id: str, step_id: str) -> dict[str, Any] | None:
        query = """
            SELECT * FROM workflow_research_waits
            WHERE workflow_run_id = ? AND step_id = ?
            LIMIT 1
        """
        params = (str(workflow_run_id), str(step_id))
        if self._using_backend():
            with self.backend.transaction() as conn:  # type: ignore[union-attr]
                result = self._execute_backend(query, params, connection=conn)
            row = self._row_from_result(result)
            return row.to_dict() if row else None

        conn = self._acquire_sqlite()
        try:
            row = conn.cursor().execute(query, params).fetchone()
            return dict(row) if row else None
        finally:
            self._release_sqlite(conn)

    def claim_research_waits_for_resume(
        self,
        *,
        research_run_id: str,
        checkpoint_id: str,
    ) -> list[dict[str, Any]]:
        now = _utcnow_iso()
        select_query = """
            SELECT * FROM workflow_research_waits
            WHERE research_run_id = ? AND checkpoint_id = ? AND wait_status = 'waiting'
            ORDER BY created_at ASC
        """
        update_query = """
            UPDATE workflow_research_waits
            SET wait_status = 'resuming', updated_at = ?
            WHERE wait_id = ? AND wait_status = 'waiting'
        """
        params = (str(research_run_id), str(checkpoint_id))

        if self._using_backend():
            claimed: list[dict[str, Any]] = []
            with self.backend.transaction() as conn:  # type: ignore[union-attr]
                result = self._execute_backend(select_query, params, connection=conn)
                rows = self._rows_from_result(result)
                for row in rows:
                    wait_row = row.to_dict()
                    update_result = self._execute_backend(
                        update_query,
                        (now, wait_row["wait_id"]),
                        connection=conn,
                    )
                    if update_result.rowcount:
                        wait_row["wait_status"] = "resuming"
                        wait_row["updated_at"] = now
                        claimed.append(wait_row)
            return claimed

        conn = self._acquire_sqlite()
        try:
            cur = conn.cursor()
            rows = [dict(row) for row in cur.execute(select_query, params).fetchall()]
            claimed: list[dict[str, Any]] = []
            for row in rows:
                cur.execute(update_query, (now, row["wait_id"]))
                if cur.rowcount:
                    row["wait_status"] = "resuming"
                    row["updated_at"] = now
                    claimed.append(row)
            conn.commit()
            return claimed
        finally:
            self._release_sqlite(conn)

    def mark_research_wait_resumed(self, *, wait_id: str) -> None:
        now = _utcnow_iso()
        query = """
            UPDATE workflow_research_waits
            SET wait_status = 'resumed', resumed_at = ?, updated_at = ?
            WHERE wait_id = ?
        """
        params = (now, now, str(wait_id))
        if self._using_backend():
            with self.backend.transaction() as conn:  # type: ignore[union-attr]
                self._execute_backend(query, params, connection=conn)
            return

        try:
            self._conn.execute(query, params)
            self._conn.commit()
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower():
                self._sqlite_retry_execute(query, params)
                self._sqlite_retry_commit()
            else:
                raise

    def reset_research_wait_for_retry(self, *, wait_id: str) -> None:
        now = _utcnow_iso()
        query = """
            UPDATE workflow_research_waits
            SET wait_status = 'waiting', updated_at = ?
            WHERE wait_id = ? AND wait_status = 'resuming'
        """
        params = (now, str(wait_id))
        if self._using_backend():
            with self.backend.transaction() as conn:  # type: ignore[union-attr]
                self._execute_backend(query, params, connection=conn)
            return

        try:
            self._conn.execute(query, params)
            self._conn.commit()
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower():
                self._sqlite_retry_execute(query, params)
                self._sqlite_retry_commit()
            else:
                raise

    def cancel_research_wait_links_for_run(self, *, workflow_run_id: str) -> None:
        now = _utcnow_iso()
        query = """
            UPDATE workflow_research_waits
            SET wait_status = 'cancelled', updated_at = ?
            WHERE workflow_run_id = ? AND wait_status IN ('waiting', 'resuming')
        """
        params = (now, str(workflow_run_id))
        if self._using_backend():
            with self.backend.transaction() as conn:  # type: ignore[union-attr]
                self._execute_backend(query, params, connection=conn)
            return

        try:
            self._conn.execute(query, params)
            self._conn.commit()
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower():
                self._sqlite_retry_execute(query, params)
                self._sqlite_retry_commit()
            else:
                raise

    def list_artifacts_for_run(self, run_id: str) -> list[dict[str, Any]]:
        sql = "SELECT * FROM workflow_artifacts WHERE run_id = ? ORDER BY created_at ASC"
        if self._using_backend():
            with self.backend.transaction() as conn:  # type: ignore[union-attr]
                result = self._execute_backend(sql, (run_id,), connection=conn)
            rows = self._rows_from_result(result)
        else:
            rows = self._conn.cursor().execute(sql, (run_id,)).fetchall()

        out: list[dict[str, Any]] = []
        for r in rows:
            data = self._row_to_dict(r)
            # Decode metadata_json; attempt to decrypt if envelope present and key available
            md: dict[str, Any] = {}
            try:
                md = json.loads(data.get("metadata_json") or "{}")
                if isinstance(md, dict) and "_encrypted" in md:
                    try:
                        from tldw_Server_API.app.core.Security.crypto import decrypt_json_blob
                        dec = decrypt_json_blob(md.get("_encrypted") or {})
                        if isinstance(dec, dict):
                            md = dec
                        else:
                            # Hide encrypted content when key not available
                            md = {"_encrypted": True}
                    except _WORKFLOWS_DB_NONCRITICAL_EXCEPTIONS:
                        md = {"_encrypted": True}
            except _WORKFLOWS_DB_NONCRITICAL_EXCEPTIONS:
                pass
            data["metadata_json"] = md
            out.append(data)
        return out

    def get_artifact(self, artifact_id: str) -> dict[str, Any] | None:
        query = "SELECT * FROM workflow_artifacts WHERE artifact_id = ?"
        if self._using_backend():
            with self.backend.transaction() as conn:  # type: ignore[union-attr]
                result = self._execute_backend(query, (artifact_id,), connection=conn)
            row = self._row_from_result(result)
            if not row:
                return None
            data = row.to_dict()
        else:
            row = self._conn.cursor().execute(query, (artifact_id,)).fetchone()
            if not row:
                return None
            data = dict(row)

        try:
            md = json.loads(data.get("metadata_json") or "{}")
            if isinstance(md, dict) and "_encrypted" in md:
                try:
                    from tldw_Server_API.app.core.Security.crypto import decrypt_json_blob
                    dec = decrypt_json_blob(md.get("_encrypted") or {})
                    md = dec if isinstance(dec, dict) else {"_encrypted": True}
                except _WORKFLOWS_DB_NONCRITICAL_EXCEPTIONS:
                    md = {"_encrypted": True}
            data["metadata_json"] = md
        except _WORKFLOWS_DB_NONCRITICAL_EXCEPTIONS:
            pass
        return data

    def delete_artifact(self, artifact_id: str) -> None:
        if self._using_backend():
            with self.backend.transaction() as conn:  # type: ignore[union-attr]
                self._execute_backend("DELETE FROM workflow_artifacts WHERE artifact_id = ?", (artifact_id,), connection=conn)
            return
        cur = self._conn.cursor()
        try:
            cur.execute("DELETE FROM workflow_artifacts WHERE artifact_id = ?", (artifact_id,))
            self._conn.commit()
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower():
                self._sqlite_retry_execute("DELETE FROM workflow_artifacts WHERE artifact_id = ?", (artifact_id,))
                self._sqlite_retry_commit()
            else:
                raise

    def list_artifacts_older_than(self, cutoff_iso: str) -> list[dict[str, Any]]:
        sql = "SELECT * FROM workflow_artifacts WHERE created_at < ?"
        rows: list[Any]
        if self._using_backend():
            with self.backend.transaction() as conn:  # type: ignore[union-attr]
                result = self._execute_backend(sql, (cutoff_iso,), connection=conn)
            rows = self._rows_from_result(result)
        else:
            rows = self._conn.cursor().execute(sql, (cutoff_iso,)).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            out.append(self._row_to_dict(r))
        return out

    # ---------- Human-in-the-loop decisions ----------
    def approve_step_decision(
        self,
        *,
        run_id: str,
        step_id: str,
        approved_by: str,
        comment: str | None = None,
    ) -> None:
        """Mark step decision approved and set final status to succeeded for matching rows.

        For v0.1, we update all rows matching run_id and step_id.
        """
        params = ("approved", approved_by, _utcnow_iso(), comment or "", "succeeded", run_id, step_id)
        query = (
            "UPDATE workflow_step_runs SET decision = ?, approved_by = ?, approved_at = ?, review_comment = ?, status = ? "
            "WHERE run_id = ? AND step_id = ?"
        )
        if self._using_backend():
            with self.backend.transaction() as conn:  # type: ignore[union-attr]
                self._execute_backend(query, params, connection=conn)
            return
        cur = self._conn.cursor()
        try:
            cur.execute(query, params)
            self._conn.commit()
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower():
                self._sqlite_retry_execute(query, params)
                self._sqlite_retry_commit()
            else:
                raise

    # ---------- Webhook DLQ ----------
    def enqueue_webhook_dlq(self, *, tenant_id: str, run_id: str, url: str, body: dict[str, Any] | None = None, last_error: str | None = None) -> None:
        params = (
            tenant_id,
            run_id,
            url,
            json.dumps(body or {}),
            0,
            None,
            last_error or "",
            _utcnow_iso(),
        )
        query = """
            INSERT INTO workflow_webhook_dlq(
                tenant_id, run_id, url, body_json, attempts, next_attempt_at, last_error, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        if self._using_backend():
            with self.backend.transaction() as conn:  # type: ignore[union-attr]
                self._execute_backend(query, params, connection=conn)
            return
        cur = self._conn.cursor()
        try:
            cur.execute(query, params)
            self._conn.commit()
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower():
                self._sqlite_retry_execute(query, params)
                self._sqlite_retry_commit()
            else:
                raise

    def list_webhook_dlq_due(self, *, limit: int = 50) -> list[dict[str, Any]]:
        """Return DLQ rows that are due for retry (next_attempt_at is null or <= now).

        Results are ordered by next_attempt_at (nulls first via COALESCE to created_at) then id for stability.
        """
        if self._using_backend():
            query = (
                "SELECT id, tenant_id, run_id, url, body_json, attempts, next_attempt_at, last_error, created_at "
                "FROM workflow_webhook_dlq "
                "WHERE next_attempt_at IS NULL OR next_attempt_at <= NOW() "
                "ORDER BY COALESCE(next_attempt_at, created_at) ASC, id ASC "
                "LIMIT %s"
            )
            with self.backend.transaction() as conn:  # type: ignore[union-attr]
                rows = self._fetchall_backend(query, (limit,), connection=conn)
            return [dict(r) if isinstance(r, dict) else {
                "id": r[0], "tenant_id": r[1], "run_id": r[2], "url": r[3], "body_json": r[4],
                "attempts": r[5], "next_attempt_at": r[6], "last_error": r[7], "created_at": r[8]
            } for r in rows or []]
        cur = self._conn.cursor()
        cur.execute(
            """
            SELECT id, tenant_id, run_id, url, body_json, attempts, next_attempt_at, last_error, created_at
            FROM workflow_webhook_dlq
            WHERE next_attempt_at IS NULL OR next_attempt_at <= datetime('now')
            ORDER BY COALESCE(next_attempt_at, created_at) ASC, id ASC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cur.fetchall()
        out: list[dict[str, Any]] = []
        for r in rows or []:
            try:
                out.append({
                    "id": r[0],
                    "tenant_id": r[1],
                    "run_id": r[2],
                    "url": r[3],
                    "body_json": r[4],
                    "attempts": r[5],
                    "next_attempt_at": r[6],
                    "last_error": r[7],
                    "created_at": r[8],
                })
            except _WORKFLOWS_DB_NONCRITICAL_EXCEPTIONS:
                # Attempt dict row style access (when using row_factory)
                out.append({
                    "id": r.get("id"),
                    "tenant_id": r.get("tenant_id"),
                    "run_id": r.get("run_id"),
                    "url": r.get("url"),
                    "body_json": r.get("body_json"),
                    "attempts": r.get("attempts"),
                    "next_attempt_at": r.get("next_attempt_at"),
                    "last_error": r.get("last_error"),
                    "created_at": r.get("created_at"),
                })
        return out

    def delete_webhook_dlq(self, *, dlq_id: int) -> None:
        if self._using_backend():
            with self.backend.transaction() as conn:  # type: ignore[union-attr]
                self._execute_backend("DELETE FROM workflow_webhook_dlq WHERE id = %s", (dlq_id,), connection=conn)
            return
        cur = self._conn.cursor()
        cur.execute("DELETE FROM workflow_webhook_dlq WHERE id = ?", (dlq_id,))
        self._conn.commit()

    def update_webhook_dlq_failure(self, *, dlq_id: int, last_error: str, next_attempt_at_iso: str | None, attempts: int | None = None) -> None:
        """Update DLQ row after a failed attempt.

        If attempts is provided, set to that value; else increment by 1.
        """
        if self._using_backend():
            if attempts is None:
                query = (
                    "UPDATE workflow_webhook_dlq SET attempts = attempts + 1, last_error = %s, next_attempt_at = %s WHERE id = %s"
                )
                params = (last_error, next_attempt_at_iso, dlq_id)
            else:
                query = (
                    "UPDATE workflow_webhook_dlq SET attempts = %s, last_error = %s, next_attempt_at = %s WHERE id = %s"
                )
                params = (attempts, last_error, next_attempt_at_iso, dlq_id)
            with self.backend.transaction() as conn:  # type: ignore[union-attr]
                self._execute_backend(query, params, connection=conn)
            return
        cur = self._conn.cursor()
        if attempts is None:
            cur.execute(
                "UPDATE workflow_webhook_dlq SET attempts = attempts + 1, last_error = ?, next_attempt_at = ? WHERE id = ?",
                (last_error, next_attempt_at_iso, dlq_id),
            )
        else:
            cur.execute(
                "UPDATE workflow_webhook_dlq SET attempts = ?, last_error = ?, next_attempt_at = ? WHERE id = ?",
                (attempts, last_error, next_attempt_at_iso, dlq_id),
            )
        self._conn.commit()

    def list_webhook_dlq_all(self, *, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        """List all DLQ rows with stable ordering (admin UI)."""
        if self._using_backend():
            query = (
                "SELECT id, tenant_id, run_id, url, body_json, attempts, next_attempt_at, last_error, created_at "
                "FROM workflow_webhook_dlq ORDER BY created_at ASC, id ASC LIMIT %s OFFSET %s"
            )
            with self.backend.transaction() as conn:  # type: ignore[union-attr]
                rows = self._fetchall_backend(query, (int(limit), int(offset)), connection=conn)
            return [dict(r) if isinstance(r, dict) else {
                "id": r[0], "tenant_id": r[1], "run_id": r[2], "url": r[3], "body_json": r[4],
                "attempts": r[5], "next_attempt_at": r[6], "last_error": r[7], "created_at": r[8]
            } for r in rows or []]
        cur = self._conn.cursor()
        cur.execute(
            """
            SELECT id, tenant_id, run_id, url, body_json, attempts, next_attempt_at, last_error, created_at
            FROM workflow_webhook_dlq
            ORDER BY created_at ASC, id ASC
            LIMIT ? OFFSET ?
            """,
            (int(limit), int(offset)),
        )
        rows = cur.fetchall()
        out: list[dict[str, Any]] = []
        for r in rows or []:
            try:
                out.append({
                    "id": r[0],
                    "tenant_id": r[1],
                    "run_id": r[2],
                    "url": r[3],
                    "body_json": r[4],
                    "attempts": r[5],
                    "next_attempt_at": r[6],
                    "last_error": r[7],
                    "created_at": r[8],
                })
            except _WORKFLOWS_DB_NONCRITICAL_EXCEPTIONS:
                out.append({
                    "id": r.get("id"),
                    "tenant_id": r.get("tenant_id"),
                    "run_id": r.get("run_id"),
                    "url": r.get("url"),
                    "body_json": r.get("body_json"),
                    "attempts": r.get("attempts"),
                    "next_attempt_at": r.get("next_attempt_at"),
                    "last_error": r.get("last_error"),
                    "created_at": r.get("created_at"),
                })
        return out

    def reject_step_decision(
        self,
        *,
        run_id: str,
        step_id: str,
        approved_by: str,
        comment: str | None = None,
        connection: Any = None,
    ) -> None:
        """Mark step decision rejected and set status to failed for matching rows."""
        params = ("rejected", approved_by, _utcnow_iso(), comment or "", "failed", run_id, step_id)
        query = (
            "UPDATE workflow_step_runs SET decision = ?, approved_by = ?, approved_at = ?, review_comment = ?, status = ? "
            "WHERE run_id = ? AND step_id = ?"
        )
        if self._using_backend():
            if connection is None:
                with self.backend.transaction() as conn:  # type: ignore[union-attr]
                    self._execute_backend(query, params, connection=conn)
            else:
                self._execute_backend(query, params, connection=connection)
            return
        cur = self._conn.cursor()
        cur.execute(query, params)
        self._conn.commit()


__all__ = ["WorkflowsDatabase", "WorkflowDefinition", "WorkflowRun", "DEFAULT_DB_PATH"]
