# PromptStudioDatabase.py
# Database management for Prompt Studio feature
# Extends PromptsDatabase to add Prompt Studio specific functionality

import json
import os
import re
import sqlite3
import threading
import uuid
from collections.abc import Iterable
from configparser import ConfigParser
from contextlib import contextmanager, suppress
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional, Union

try:  # psycopg v3 preferred; fall back to psycopg2 if installed
    from psycopg import sql as psycopg_sql  # type: ignore
except ImportError:  # pragma: no cover
    try:
        from psycopg2 import sql as psycopg_sql  # type: ignore
    except ImportError:  # pragma: no cover
        psycopg_sql = None  # type: ignore

from loguru import logger

from ..Prompt_Management.optimization_model_config import (
    validate_secret_free_optimization_config,
)
from .backends.base import (
    BackendType,
    DatabaseBackend,
    QueryResult,
)
from .backends.base import (
    DatabaseError as BackendDatabaseError,
)
from .backends.base import UniqueConstraintError
from .backends.query_utils import (
    prepare_backend_many_statement,
    prepare_backend_statement,
    replace_collate_nocase,
    replace_insert_or_ignore,
)

# Local imports
from .Prompts_DB import ConflictError, DatabaseError, InputError, PromptsDatabase, SchemaError
from .prompt_studio_db.prompt_fields import _prepare_prompt_record_fields  # noqa: F401 - re-exported for endpoints
from .prompt_studio_db.repositories.evaluations import EvaluationsRepository
from .prompt_studio_db.repositories.optimizations import OptimizationsRepository
from .prompt_studio_db.repositories.test_cases import TestCasesRepository
from .prompt_studio_db.repositories.prompts import PromptsRepository
from .prompt_studio_db.repositories.projects import ProjectsRepository
from .prompt_studio_db.repositories.signatures import SignaturesRepository
from .prompt_studio_db.repositories.prompt_versions import PromptVersionsRepository
from .prompt_studio_db.repositories.test_runs import TestRunsRepository

_PROMPT_STUDIO_NONCRITICAL_EXCEPTIONS = (
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
    ConflictError,
    DatabaseError,
    InputError,
    SchemaError,
)



def _should_enable_prompt_studio_sqlite_wal() -> bool:
    """Default Prompt Studio SQLite to WAL outside CI and explicit test runtimes."""
    from tldw_Server_API.app.core.testing import (
        env_flag_enabled,
        is_explicit_pytest_runtime,
        is_test_mode,
    )

    override = os.getenv("TLDW_PS_SQLITE_WAL")
    if override is not None:
        return env_flag_enabled("TLDW_PS_SQLITE_WAL")

    if env_flag_enabled("CI") or env_flag_enabled("GITHUB_ACTIONS"):
        return False

    return not (is_explicit_pytest_runtime() or is_test_mode())


########################################################################################################################
# Backend cursor/connection helpers


class PromptStudioRowAdapter:
    """Row object that mimics sqlite3.Row semantics for consumers."""

    __slots__ = ("_mapping", "_columns")

    def __init__(self, mapping: dict[str, Any], columns: tuple[str, ...]):
        self._mapping = mapping
        self._columns = columns

    def __getitem__(self, key: Union[int, str]) -> Any:
        if isinstance(key, int):
            # Prefer named lookup when column metadata is a simple string
            try:
                col = self._columns[key]
                if isinstance(col, str) and isinstance(self._mapping, dict):
                    return self._mapping.get(col)
            except _PROMPT_STUDIO_NONCRITICAL_EXCEPTIONS:
                pass
            # Fallback: positional access over mapping values
            if isinstance(self._mapping, dict):
                try:
                    return list(self._mapping.values())[key]
                except _PROMPT_STUDIO_NONCRITICAL_EXCEPTIONS:
                    return None
            return None
        return self._mapping.get(key)

    def __iter__(self):
        for column in self._columns:
            yield self._mapping.get(column)

    def keys(self) -> tuple[str, ...]:
        return self._columns

    def items(self):
        for column in self._columns:
            yield column, self._mapping.get(column)

    def get(self, key: str, default: Any = None) -> Any:
        return self._mapping.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        return dict(self._mapping)


class PromptStudioBackendCursorAdapter:
    """Adapter that provides sqlite-like cursor behaviour for QueryResult objects."""

    def __init__(self, result: QueryResult):
        self._result = result
        self._index = 0
        self.rowcount = result.rowcount
        self.lastrowid = result.lastrowid
        self.description = result.description or []
        self._columns: tuple[str, ...] = tuple(
            (
                desc[0]
                if isinstance(desc, (list, tuple)) and desc
                else getattr(desc, "name", desc)
            )
            for desc in (self.description or [])
        )

    def _wrap_row(self, row: Any) -> PromptStudioRowAdapter:
        if isinstance(row, PromptStudioRowAdapter):
            return row
        if isinstance(row, dict):
            mapping = row
            columns = self._columns or tuple(mapping.keys())
        else:
            # Assume it's a sequence aligned with description
            columns = self._columns
            mapping = {columns[idx]: row[idx] for idx in range(len(columns))}
        return PromptStudioRowAdapter(mapping, columns)

    def fetchone(self) -> Optional[PromptStudioRowAdapter]:
        if self._index >= len(self._result.rows):
            return None
        row = self._result.rows[self._index]
        self._index += 1
        return self._wrap_row(row)

    def fetchall(self) -> list[PromptStudioRowAdapter]:
        rows = self._result.rows[self._index :]
        self._index = len(self._result.rows)
        return [self._wrap_row(row) for row in rows]

    def fetchmany(self, size: Optional[int] = None) -> list[PromptStudioRowAdapter]:
        if size is None or size <= 0:
            size = len(self._result.rows) - self._index
        end = min(self._index + size, len(self._result.rows))
        rows = self._result.rows[self._index : end]
        self._index = end
        return [self._wrap_row(row) for row in rows]

    def close(self) -> None:
        self._result = QueryResult(rows=[], rowcount=0)
        self.rowcount = 0
        self.lastrowid = None
        self.description = None
        self._columns = ()


class PromptStudioBackendCursorWrapper:
    """Cursor wrapper that routes SQL through the configured DatabaseBackend."""

    def __init__(self, db: 'BackendPromptStudioDatabaseBase', connection: Any):
        self._db = db
        self._connection = connection
        self._result: Optional[QueryResult] = None
        self._adapter: Optional[PromptStudioBackendCursorAdapter] = None
        self.rowcount: int = -1
        self.lastrowid: Optional[int] = None
        self.description = None
        self._columns: tuple[str, ...] = ()

    def execute(self, query: str, params: Optional[Union[tuple, list, dict, Any]] = None):
        import sqlite3

        prepared_query, prepared_params = self._db._prepare_backend_statement(query, params)
        try:
            self._result = self._db.backend.execute(
                prepared_query,
                prepared_params,
                connection=self._connection,
            )
        except BackendDatabaseError as exc:
            msg = str(exc)
            # The backend redacts driver messages, so the type is what identifies a
            # uniqueness conflict. ConflictError is a DatabaseError, so existing
            # handlers still catch it; the message carries no data.
            if isinstance(exc, UniqueConstraintError):
                raise ConflictError("UNIQUE constraint failed") from exc
            if "duplicate" in msg.lower() or "unique constraint" in msg.lower():
                raise sqlite3.IntegrityError(msg)  # noqa: B904
            raise DatabaseError(f"Backend query execution failed: {msg}") from exc  # noqa: TRY003

        self._adapter = PromptStudioBackendCursorAdapter(self._result)
        self.rowcount = self._result.rowcount
        self.lastrowid = self._result.lastrowid
        self.description = self._adapter.description
        self._columns = self._adapter._columns
        return self

    def executemany(self, query: str, params_list: list[Union[tuple, list, dict, Any]]):
        import sqlite3

        prepared_query, prepared_params_list = self._db._prepare_backend_many_statement(query, params_list)
        try:
            self._result = self._db.backend.execute_many(
                prepared_query,
                prepared_params_list,
                connection=self._connection,
            )
        except BackendDatabaseError as exc:
            msg = str(exc)
            # The backend redacts driver messages, so the type is what identifies a
            # uniqueness conflict. ConflictError is a DatabaseError, so existing
            # handlers still catch it; the message carries no data.
            if isinstance(exc, UniqueConstraintError):
                raise ConflictError("UNIQUE constraint failed") from exc
            if "duplicate" in msg.lower() or "unique constraint" in msg.lower():
                raise sqlite3.IntegrityError(msg)  # noqa: B904
            raise DatabaseError(f"Backend batch execution failed: {msg}") from exc  # noqa: TRY003

        self._adapter = PromptStudioBackendCursorAdapter(self._result)
        self.rowcount = self._result.rowcount
        self.lastrowid = self._result.lastrowid
        self.description = self._adapter.description
        self._columns = self._adapter._columns
        return self

    def fetchone(self) -> Optional[dict[str, Any]]:
        row = self._adapter.fetchone() if self._adapter else None
        return row

    def fetchall(self) -> list[dict[str, Any]]:
        return self._adapter.fetchall() if self._adapter else []

    def fetchmany(self, size: Optional[int] = None) -> list[dict[str, Any]]:
        return self._adapter.fetchmany(size) if self._adapter else []

    def close(self) -> None:
        if self._adapter:
            self._adapter.close()
        self._adapter = None
        self._result = None
        self.rowcount = -1
        self.lastrowid = None
        self.description = None


class PromptStudioBackendConnectionWrapper:
    """Connection wrapper exposing sqlite-like API backed by DatabaseBackend."""

    def __init__(self, db: 'BackendPromptStudioDatabaseBase', connection: Any):
        self._db = db
        self.raw_connection = connection

    def cursor(self):
        return PromptStudioBackendCursorWrapper(self._db, self.raw_connection)

    def execute(self, query: str, params: Optional[Union[tuple, list, dict, Any]] = None):
        cursor = self.cursor()
        return cursor.execute(query, params)

    def executemany(self, query: str, params_list: list[Union[tuple, list, dict, Any]]):
        cursor = self.cursor()
        return cursor.executemany(query, params_list)

    def commit(self):
        return self.raw_connection.commit()

    def rollback(self):
        return self.raw_connection.rollback()

    @property
    def closed(self) -> bool:
        return getattr(self.raw_connection, "closed", False)


class PromptStudioBackendManagedTransaction:
    """Context manager leveraging the backend's native transaction handling."""

    def __init__(self, db: 'BackendPromptStudioDatabaseBase'):
        self._db = db
        self._ctx = None
        self._conn = None

    def __enter__(self):
        self._ctx = self._db.backend.transaction()
        raw_conn = self._ctx.__enter__()
        try:
            self._db._apply_tenant_session(raw_conn)
        except BaseException as exc:
            with suppress(Exception):
                self._ctx.__exit__(exc.__class__, exc, exc.__traceback__)
            self._ctx = None
            raise
        self._conn = PromptStudioBackendConnectionWrapper(self._db, raw_conn)
        return self._conn

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._ctx is None:
            return False
        return self._ctx.__exit__(exc_type, exc_val, exc_tb)


########################################################################################################################
# Backend-aware Prompt Studio implementation (PostgreSQL)


class BackendPromptStudioDatabaseBase:
    """Common helpers for backend-backed Prompt Studio database implementations."""

    def __init__(
        self,
        db_path: Union[str, Path],
        client_id: str,
        *,
        tenant_user_id: Optional[str] = None,
        backend: Optional[DatabaseBackend] = None,
        config: Optional[ConfigParser] = None,
    ) -> None:
        if backend is None:
            raise ValueError("Prompt Studio backend database requires an explicit DatabaseBackend instance")  # noqa: TRY003

        self.backend = backend
        self.backend_type = backend.backend_type
        if self.backend_type != BackendType.POSTGRESQL:
            raise ValueError(  # noqa: TRY003
                f"BackendPromptStudioDatabaseBase only supports PostgreSQL backends; received {self.backend_type.value}"
            )

        self.client_id = client_id
        self.tenant_user_id = (
            client_id if tenant_user_id is None else str(tenant_user_id)
        )
        self._config = config
        self.db_path = Path(db_path) if not isinstance(db_path, Path) else db_path
        self.db_path_str = str(self.db_path)
        self._local = threading.local()
        self._write_lock = threading.RLock()
        self._sync_log_available: Optional[bool] = None

    # --- Connection handling ---
    def _open_new_connection(self):
        try:
            pool = self.backend.get_pool()
            return pool.get_connection()
        except BackendDatabaseError as exc:
            raise DatabaseError(f"Failed to acquire backend connection: {exc}") from exc  # noqa: TRY003

    def _apply_tenant_session(self, raw_conn: Any) -> None:
        """Apply the Prompt Studio tenant to a borrowed PostgreSQL connection."""
        try:
            cur = raw_conn.cursor()
            user_value = self.tenant_user_id
            if psycopg_sql is not None:  # type: ignore[name-defined]
                stmt = psycopg_sql.SQL("SET SESSION app.current_user_id = {}").format(
                    psycopg_sql.Literal(user_value)
                )
                cur.execute(stmt)
            else:
                cur.execute(
                    "SELECT set_config('app.current_user_id', %s, false)",
                    (user_value,),
                )
            raw_conn.commit()
        except Exception:  # noqa: BLE001 - normalize arbitrary driver failures
            with suppress(Exception):
                raw_conn.rollback()
            raise DatabaseError(
                "Failed to apply Prompt Studio tenant session"
            ) from None

    def _release_connection(
        self,
        wrapper: Optional[PromptStudioBackendConnectionWrapper],
        *,
        discard: bool = False,
    ) -> None:
        if not wrapper:
            return
        try:
            raw_conn = wrapper.raw_connection
            pool = self.backend.get_pool()
            if discard:
                discard_connection = getattr(pool, "discard_connection", None)
                if callable(discard_connection):
                    discard_connection(raw_conn)
                    return
                with suppress(Exception):
                    raw_conn.close()
            pool.return_connection(raw_conn)
        except BackendDatabaseError as exc:
            logger.warning("Error returning backend connection to pool: {}", exc)

    def _get_thread_connection(self) -> PromptStudioBackendConnectionWrapper:
        wrapper: Optional[PromptStudioBackendConnectionWrapper] = getattr(self._local, 'conn', None)
        if wrapper is not None and not wrapper.closed:
            return wrapper

        raw_conn = self._open_new_connection()
        # Apply per-tenant session guard for PostgreSQL (RLS via current_setting('app.current_user_id'))
        try:
            self._apply_tenant_session(raw_conn)
        except BaseException:
            with suppress(Exception):
                self.backend.get_pool().return_connection(raw_conn)
            raise
        wrapper = PromptStudioBackendConnectionWrapper(self, raw_conn)
        self._local.conn = wrapper
        logger.debug(
            'Acquired Prompt Studio backend connection ({}) for thread {}',
            self.backend_type.value,
            threading.get_ident(),
        )
        return wrapper

    def get_connection(self) -> PromptStudioBackendConnectionWrapper:
        return self._get_thread_connection()

    def close_connection(self) -> None:
        wrapper: Optional[PromptStudioBackendConnectionWrapper] = getattr(self._local, 'conn', None)
        if wrapper is None:
            return

        discard = False
        try:
            # Cached backend connections execute reads as external connections,
            # so psycopg may leave an implicit transaction open. Psycopg exposes
            # that state through ``info.transaction_status`` rather than the
            # sqlite-style ``in_transaction`` attribute; rollback is safe when
            # already idle and guarantees read locks are released before pooling.
            if wrapper.raw_connection:
                try:
                    wrapper.rollback()
                except Exception as exc:  # noqa: BLE001 - driver errors vary
                    discard = True
                    logger.warning(
                        "Prompt Studio connection rollback failed during release: {}",
                        type(exc).__name__,
                    )
        finally:
            try:
                # A rollback failure makes the checkout unsafe to reuse. The
                # pool still owns disposal/bookkeeping for that connection.
                self._release_connection(wrapper, discard=discard)
            finally:
                self._local.conn = None

    def close(self) -> None:
        self.close_connection()

    @contextmanager
    def transaction(self) -> Iterable[PromptStudioBackendConnectionWrapper]:
        ctx = PromptStudioBackendManagedTransaction(self)
        conn = ctx.__enter__()
        try:
            yield conn
            ctx.__exit__(None, None, None)
        except Exception as exc:  # noqa: BLE001
            ctx.__exit__(exc.__class__, exc, exc.__traceback__)
            raise

    # --- Query preparation helpers ---
    def _prepare_backend_statement(
        self,
        query: str,
        params: Optional[Union[tuple, list, dict, Any]] = None,
    ) -> tuple[str, Optional[Union[tuple, dict]]]:
        return prepare_backend_statement(
            self.backend_type,
            query,
            params,
            apply_default_transform=True,
            ensure_returning=True,
        )

    def _prepare_backend_many_statement(
        self,
        query: str,
        params_list: list[Union[tuple, list, dict, Any]],
    ) -> tuple[str, list[Optional[Union[tuple, dict]]]]:
        return prepare_backend_many_statement(
            self.backend_type,
            query,
            params_list,
            apply_default_transform=True,
            ensure_returning=False,
        )

    # Convenience for subclasses
    def _execute(
        self,
        query: str,
        params: Optional[Union[tuple, list, dict, Any]] = None,
        *,
        connection: Optional[PromptStudioBackendConnectionWrapper] = None,
    ) -> PromptStudioBackendCursorWrapper:
        conn = connection or self.get_connection()
        cursor = conn.cursor()
        return cursor.execute(query, params)

    def _executemany(
        self,
        query: str,
        params_list: list[Union[tuple, list, dict, Any]],
        *,
        connection: Optional[PromptStudioBackendConnectionWrapper] = None,
    ) -> PromptStudioBackendCursorWrapper:
        conn = connection or self.get_connection()
        cursor = conn.cursor()
        return cursor.executemany(query, params_list)


class _BackendPromptStudioDatabase(BackendPromptStudioDatabaseBase):
    """PostgreSQL-backed Prompt Studio database implementation."""

    _SCHEMA_VERSION = 1
    _MIGRATION_FILES_SQL = [
        "001_prompt_studio_schema.sql",
        "003_prompt_studio_iterations.sql",
        "002_prompt_studio_indexes.sql",
        # 003 triggers file intentionally omitted (no-op placeholder)
        # 004 FTS handled via backend abstraction
        "005_add_chunking_templates.sql",
        "006_prompt_studio_structured_prompts.sql",
    ]

    _FTS_CONFIG = (
        ("prompt_studio_projects", ["name", "description"]),
        ("prompt_studio_prompts", ["name", "system_prompt", "user_prompt"]),
        ("prompt_studio_test_cases", ["name", "description", "tags"]),
    )

    _JSON_FIELDS = {
        "metadata",
        "input_schema",
        "output_schema",
        "constraints",
        "validation_rules",
        "few_shot_examples",
        "modules_config",
        "prompt_definition",
        "model_params",
        "inputs",
        "outputs",
        "expected_outputs",
        "actual_outputs",
        "scores",
        "test_case_ids",
        "test_run_ids",
        "aggregate_metrics",
        "model_configs",
        "payload",
        "result",
        "initial_metrics",
        "final_metrics",
        "optimization_config",
        "prompt_variant",
        "metrics",
    }

    _DATETIME_FIELDS = {
        "created_at",
        "updated_at",
        "deleted_at",
        "last_modified",
        "started_at",
        "completed_at",
    }

    _MIGRATIONS_DIR = Path(__file__).parent / "migrations"

    def __init__(
        self,
        db_path: Union[str, Path],
        client_id: str,
        *,
        tenant_user_id: Optional[str] = None,
        backend: Optional[DatabaseBackend] = None,
        config: Optional[ConfigParser] = None,
    ) -> None:
        super().__init__(
            db_path,
            client_id,
            tenant_user_id=tenant_user_id,
            backend=backend,
            config=config,
        )
        self._fts_columns = {
            table: f"{table}_tsv" for table, _columns in self._FTS_CONFIG
        }
        self._initialize_schema_postgres()

    def _cursor_exec(self, conn: Any, query: str, params: Optional[Union[tuple, list, dict, Any]] = None):
        """Execute a query using the backend's parameter style.

        Converts SQLite-style placeholders to PostgreSQL, then executes using the
        provided psycopg connection. Returns a native cursor with description set.
        """
        q, p = self._prepare_backend_statement(query, params)
        cur = conn.cursor()
        if p is not None:
            cur.execute(q, p)
        else:
            cur.execute(q)
        return cur

    # --- Schema management ---
    def _initialize_schema_postgres(self) -> None:
        with self.backend.transaction() as conn:
            self._ensure_extensions(conn)
            if not self.backend.table_exists('prompt_studio_projects', connection=conn):
                self._apply_postgres_migrations(conn)
            # Ensure auxiliary tables exist even on existing DBs (idempotency mapping)
            try:
                self.backend.execute(
                    (
                        "CREATE TABLE IF NOT EXISTS prompt_studio_idempotency ("
                        " id BIGSERIAL PRIMARY KEY,"
                        " entity_type TEXT NOT NULL,"
                        " idempotency_key TEXT NOT NULL,"
                        " entity_id BIGINT NOT NULL,"
                        " user_id TEXT,"
                        " created_at TIMESTAMPTZ DEFAULT NOW()"
                        ")"
                    ),
                    connection=conn,
                )
                # Composite uniqueness per user
                self.backend.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS uq_ps_idem_user ON prompt_studio_idempotency(entity_type, idempotency_key, user_id)",
                    connection=conn,
                )
                self.backend.execute(
                    "CREATE INDEX IF NOT EXISTS idx_ps_idem_entity ON prompt_studio_idempotency(entity_type, user_id)",
                    connection=conn,
                )
                # Note: Postgres idempotency helpers are implemented in this class
                # via _idem_lookup/_idem_record and scoped by (entity_type, idempotency_key, user_id).
            except BackendDatabaseError as exc:
                raise SchemaError(f"Failed to ensure idempotency table: {exc}") from exc  # noqa: TRY003
            # Ensure leasing columns exist on job queue
            try:
                self.backend.execute(
                    "ALTER TABLE prompt_studio_job_queue ADD COLUMN IF NOT EXISTS leased_until TIMESTAMPTZ",
                    connection=conn,
                )
                self.backend.execute(
                    "ALTER TABLE prompt_studio_job_queue ADD COLUMN IF NOT EXISTS lease_owner TEXT",
                    connection=conn,
                )
                self.backend.execute(
                    "ALTER TABLE prompt_studio_optimizations ADD COLUMN IF NOT EXISTS test_case_ids JSONB",
                    connection=conn,
                )
            except BackendDatabaseError:
                # Older Postgres versions may not support IF NOT EXISTS on ADD COLUMN; fall back
                try:
                    # Probe column existence; if missing, add without IF NOT EXISTS
                    self.backend.execute(
                        "SELECT leased_until FROM prompt_studio_job_queue LIMIT 1",
                        connection=conn,
                    )
                except BackendDatabaseError:
                    self.backend.execute(
                        "ALTER TABLE prompt_studio_job_queue ADD COLUMN leased_until TIMESTAMPTZ",
                        connection=conn,
                    )
                try:
                    self.backend.execute(
                        "SELECT lease_owner FROM prompt_studio_job_queue LIMIT 1",
                        connection=conn,
                    )
                except BackendDatabaseError:
                    self.backend.execute(
                        "ALTER TABLE prompt_studio_job_queue ADD COLUMN lease_owner TEXT",
                        connection=conn,
                    )
                try:
                    self.backend.execute(
                        "SELECT test_case_ids FROM prompt_studio_optimizations LIMIT 1",
                        connection=conn,
                    )
                except BackendDatabaseError:
                    self.backend.execute(
                        "ALTER TABLE prompt_studio_optimizations ADD COLUMN test_case_ids JSONB",
                        connection=conn,
                    )
                try:
                    self.backend.execute(
                        "SELECT prompt_format FROM prompt_studio_prompts LIMIT 1",
                        connection=conn,
                    )
                except BackendDatabaseError:
                    self.backend.execute(
                        "ALTER TABLE prompt_studio_prompts ADD COLUMN prompt_format TEXT NOT NULL DEFAULT 'legacy'",
                        connection=conn,
                    )
                try:
                    self.backend.execute(
                        "SELECT prompt_schema_version FROM prompt_studio_prompts LIMIT 1",
                        connection=conn,
                    )
                except BackendDatabaseError:
                    self.backend.execute(
                        "ALTER TABLE prompt_studio_prompts ADD COLUMN prompt_schema_version INTEGER",
                        connection=conn,
                    )
                try:
                    self.backend.execute(
                        "SELECT prompt_definition FROM prompt_studio_prompts LIMIT 1",
                        connection=conn,
                    )
                except BackendDatabaseError:
                    self.backend.execute(
                        "ALTER TABLE prompt_studio_prompts ADD COLUMN prompt_definition JSONB",
                        connection=conn,
                    )
        self._ensure_postgres_fts()

    def _ensure_extensions(self, conn) -> None:
        try:
            self.backend.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto", connection=conn)
        except BackendDatabaseError as exc:
            raise SchemaError(f"Failed enabling pgcrypto extension: {exc}") from exc  # noqa: TRY003

    def _apply_postgres_migrations(self, conn) -> None:
        for filename in self._MIGRATION_FILES_SQL:
            migration_path = self._MIGRATIONS_DIR / filename
            if not migration_path.exists():
                logger.warning("Prompt Studio migration file missing: {}", migration_path)
                continue
            sql = migration_path.read_text()
            statements = self._convert_sqlite_schema_to_postgres_statements(sql)
            for statement in statements:
                try:
                    self.backend.execute(statement, connection=conn)
                except BackendDatabaseError as exc:
                    raise SchemaError(f"Failed applying migration {filename}: {exc}") from exc  # noqa: TRY003

    def _ensure_postgres_fts(self) -> None:
        for source_table, columns in self._FTS_CONFIG:
            try:
                self.backend.create_fts_table(
                    table_name=source_table,
                    source_table=source_table,
                    columns=list(columns),
                )
            except BackendDatabaseError as exc:
                raise SchemaError(f"Failed to provision Prompt Studio FTS ({source_table}): {exc}") from exc  # noqa: TRY003

    def get_fts_column(self, table_name: str) -> Optional[str]:
        return getattr(self, "_fts_columns", {}).get(table_name)

    def _convert_sqlite_schema_to_postgres_statements(self, sql: str) -> list[str]:
        statements: list[str] = []
        buffer: list[str] = []
        in_block_comment = False
        in_trigger_block = False

        for raw_line in sql.splitlines():
            stripped = raw_line.strip()

            if not stripped:
                continue

            if in_block_comment:
                if '*/' in stripped:
                    in_block_comment = False
                continue

            if stripped.startswith('/*'):
                if '*/' not in stripped:
                    in_block_comment = True
                continue

            if stripped.startswith('--'):
                continue

            upper = stripped.upper()

            if upper.startswith('PRAGMA'):
                continue

            if in_trigger_block:
                # Skip lines belonging to a trigger block until semicolon
                if ';' in stripped:
                    in_trigger_block = False
                continue

            if 'CREATE VIRTUAL TABLE' in upper:
                # handled by backend FTS helpers
                continue

            if upper.startswith('INSERT INTO') and 'FTS' in upper:
                continue

            if upper.startswith('DROP TRIGGER') or upper.startswith('CREATE TRIGGER'):
                # Skip entire trigger block (SQLite syntax not supported in Postgres)
                in_trigger_block = True
                continue

            buffer.append(raw_line)

            if stripped.endswith(';'):
                statement = '\n'.join(buffer).strip()
                buffer = []
                transformed = self._transform_sqlite_statement_for_postgres(statement)
                if transformed:
                    statements.append(transformed)

        return statements

    # --- Idempotency helpers (Postgres) ---
    def _idem_lookup(self, entity_type: str, key: str, user_id: Optional[str]) -> Optional[int]:
        try:
            if user_id is None:
                cursor = self._execute(
                    """
                    SELECT entity_id
                    FROM prompt_studio_idempotency
                    WHERE entity_type = ?
                      AND idempotency_key = ?
                      AND user_id IS NULL
                    LIMIT 1
                    """,
                    (entity_type, key),
                )
            else:
                cursor = self._execute(
                    """
                    SELECT entity_id
                    FROM prompt_studio_idempotency
                    WHERE entity_type = ?
                      AND idempotency_key = ?
                      AND user_id = ?
                    LIMIT 1
                    """,
                    (entity_type, key, user_id),
                )
            row = cursor.fetchone()
            return int(row[0]) if row else None
        except BackendDatabaseError:
            return None

    def _idem_record(self, entity_type: str, key: str, entity_id: int, user_id: Optional[str]) -> None:
        try:
            # INSERT OR IGNORE is translated to ON CONFLICT DO NOTHING for Postgres by the query adapter
            with self.transaction() as conn:
                self._execute(
                    "INSERT OR IGNORE INTO prompt_studio_idempotency (entity_type, idempotency_key, entity_id, user_id) VALUES (?, ?, ?, ?)",
                    (entity_type, key, entity_id, user_id),
                    connection=conn,
                )
        except BackendDatabaseError:
            pass

    def _transform_sqlite_statement_for_postgres(self, statement: str) -> Optional[str]:
        stmt = statement.strip()
        if not stmt:
            return None

        # Normalize whitespace for easier regex handling
        stmt = re.sub(r'\s+', ' ', stmt)

        # Column conversions
        stmt = re.sub(
            r'INTEGER PRIMARY KEY AUTOINCREMENT',
            'BIGSERIAL PRIMARY KEY',
            stmt,
            flags=re.IGNORECASE,
        )
        stmt = re.sub(
            r'INTEGER PRIMARY KEY',
            'BIGSERIAL PRIMARY KEY',
            stmt,
            flags=re.IGNORECASE,
        )

        def _replace_randomblob_default(match: re.Match[str]) -> str:
            prefix = match.group(1)
            return f"{prefix}encode(gen_random_bytes(16), 'hex')"

        stmt = re.sub(
            r'(DEFAULT\s*)\(LOWER\(HEX\(RANDOMBLOB\(16\)\)\)\)',
            _replace_randomblob_default,
            stmt,
            flags=re.IGNORECASE,
        )

        # Column-specific boolean conversions
        stmt = re.sub(r'(\bdeleted\b\s+)INTEGER\s+DEFAULT\s+0', r'\1BOOLEAN NOT NULL DEFAULT FALSE', stmt, flags=re.IGNORECASE)
        stmt = re.sub(r'(\bis_golden\b\s+)INTEGER\s+DEFAULT\s+0', r'\1BOOLEAN NOT NULL DEFAULT FALSE', stmt, flags=re.IGNORECASE)
        stmt = re.sub(r'(\bis_generated\b\s+)INTEGER\s+DEFAULT\s+0', r'\1BOOLEAN NOT NULL DEFAULT FALSE', stmt, flags=re.IGNORECASE)
        stmt = re.sub(r'(\bis_builtin\b\s+)BOOLEAN\s+DEFAULT\s+0', r'\1BOOLEAN NOT NULL DEFAULT FALSE', stmt, flags=re.IGNORECASE)

        # Handle BOOLEAN defaults regardless of NOT NULL placement
        # e.g., "BOOLEAN NOT NULL DEFAULT 0" or "BOOLEAN DEFAULT 0 NOT NULL"
        stmt = re.sub(r'BOOLEAN\s+NOT\s+NULL\s+DEFAULT\s+0', 'BOOLEAN NOT NULL DEFAULT FALSE', stmt, flags=re.IGNORECASE)
        stmt = re.sub(r'BOOLEAN\s+NOT\s+NULL\s+DEFAULT\s+1', 'BOOLEAN NOT NULL DEFAULT TRUE', stmt, flags=re.IGNORECASE)
        stmt = re.sub(r'BOOLEAN\s+DEFAULT\s+0\s+NOT\s+NULL', 'BOOLEAN NOT NULL DEFAULT FALSE', stmt, flags=re.IGNORECASE)
        stmt = re.sub(r'BOOLEAN\s+DEFAULT\s+1\s+NOT\s+NULL', 'BOOLEAN NOT NULL DEFAULT TRUE', stmt, flags=re.IGNORECASE)
        # Simple form without NOT NULL
        stmt = re.sub(r'BOOLEAN\s+DEFAULT\s+0', 'BOOLEAN DEFAULT FALSE', stmt, flags=re.IGNORECASE)
        stmt = re.sub(r'BOOLEAN\s+DEFAULT\s+1', 'BOOLEAN DEFAULT TRUE', stmt, flags=re.IGNORECASE)

        stmt = re.sub(r'JSON\b', 'JSONB', stmt, flags=re.IGNORECASE)
        stmt = re.sub(r'DATETIME', 'TIMESTAMPTZ', stmt, flags=re.IGNORECASE)

        stmt = replace_collate_nocase(stmt)
        stmt = replace_insert_or_ignore(stmt)
        # Normalize boolean comparisons in indexes/constraints (e.g., WHERE deleted = 0)
        stmt = re.sub(r'\bdeleted\s*=\s*0\b', 'deleted = FALSE', stmt, flags=re.IGNORECASE)
        stmt = re.sub(r'\bdeleted\s*=\s*1\b', 'deleted = TRUE', stmt, flags=re.IGNORECASE)
        stmt = re.sub(r'\bis_builtin\s*=\s*0\b', 'is_builtin = FALSE', stmt, flags=re.IGNORECASE)
        stmt = re.sub(r'\bis_builtin\s*=\s*1\b', 'is_builtin = TRUE', stmt, flags=re.IGNORECASE)

        if not stmt.endswith(';'):
            stmt = f"{stmt};"

        return stmt

    # NOTE: Removed a duplicated, misplaced idempotency helpers block here.
    # The correct implementations exist within the PromptStudioDatabase class
    # later in this file. Keeping only one canonical definition avoids
    # indentation/scope issues during import.

    # --- Data helpers ---
    def _row_to_dict(self, cursor, row: Optional[Any] = None) -> Optional[dict[str, Any]]:
        row_obj = cursor if row is None else row

        if row_obj is None:
            return None

        if isinstance(row_obj, PromptStudioRowAdapter):
            result = row_obj.to_dict()
        elif isinstance(row_obj, dict):
            result = dict(row_obj)
        else:
            # Fallback: attempt to build from sequence with cursor description
            if hasattr(cursor, 'description') and cursor.description:
                columns = [desc[0] if isinstance(desc, (list, tuple)) and desc else desc for desc in cursor.description]
                result = {col: row_obj[idx] for idx, col in enumerate(columns)}
            else:
                raise DatabaseError("Unable to convert row to dict; missing column metadata")  # noqa: TRY003

        # The tsvector columns are search plumbing that SQLite keeps in separate FTS
        # tables; `SELECT *` / `RETURNING *` must not leak them to callers.
        for fts_column in self._fts_columns.values():
            result.pop(fts_column, None)

        for field in self._JSON_FIELDS:
            if field in result and isinstance(result[field], str):
                with suppress(TypeError, ValueError):
                    result[field] = json.loads(result[field])
            elif field in result and isinstance(result[field], (bytes, bytearray, memoryview)):
                try:
                    result[field] = json.loads(bytes(result[field]).decode('utf-8'))
                except (TypeError, ValueError):
                    result[field] = None

        for field in self._DATETIME_FIELDS:
            value = result.get(field)
            if isinstance(value, str):
                with suppress(ValueError):
                    result[field] = datetime.fromisoformat(value)

        return result

    def _log_sync_event(self, entity: str, entity_uuid: str, operation: str, payload: dict[str, Any]) -> None:
        if not entity or not entity_uuid or not operation:
            return
        if self._sync_log_available is False:
            return

        try:
            with self.transaction() as conn:
                if self._sync_log_available is None:
                    try:
                        self._sync_log_available = self.backend.table_exists(
                            "sync_log",
                            connection=conn.raw_connection,
                        )
                    except _PROMPT_STUDIO_NONCRITICAL_EXCEPTIONS as exc:
                        logger.debug("Prompt Studio sync_log availability check failed: {}", exc)
                        self._sync_log_available = False
                        return
                if not self._sync_log_available:
                    return
                self._cursor_exec(
                    conn,
                    """
                    INSERT INTO sync_log (entity, entity_uuid, operation, client_id, version, payload, timestamp)
                    VALUES (?, ?, ?, ?, 1, ?, CURRENT_TIMESTAMP)
                    RETURNING change_id
                    """,
                    (
                        entity,
                        entity_uuid,
                        operation,
                        # Shared sync-log RLS uses this column as tenant ownership.
                        # The project/prompt rows retain the originating audit client.
                        self.tenant_user_id,
                        json.dumps(payload, separators=(',', ':')) if payload else None,
                    ),
                )
        except _PROMPT_STUDIO_NONCRITICAL_EXCEPTIONS as e:
            # sync_log is optional across backends
            err_str = str(e).lower()
            if "no such table" in err_str or "does not exist" in err_str or "relation" in err_str:
                # Table doesn't exist - expected in some deployments
                self._sync_log_available = False
                logger.debug(
                    'Prompt Studio sync_log table not available; skipping event for {}/{}',
                    entity,
                    entity_uuid,
                )
            else:
                # Actual write error - worth warning about
                logger.warning(
                    'Failed to log sync event for {}/{}: {}',
                    entity,
                    entity_uuid,
                    e,
                )

    # --- Core API ---
    # Project name constraints

    # --- Signature helpers -----------------------------------------------

    # --- Test run helpers ------------------------------------------------

    # --- Evaluation helpers ---------------------------------------------

    # --- Optimization helpers -------------------------------------------

    # --- Prompt helpers ---

    # --- Job queue helpers ---

    def create_job(
        self,
        job_type: str,
        entity_id: int,
        payload: Optional[Any],
        *,
        project_id: Optional[int] = None,
        priority: int = 5,
        status: str = "queued",
        max_retries: int = 3,
        client_id: Optional[str] = None,
    ) -> dict[str, Any]:
        job_uuid = str(uuid.uuid4())
        payload_json = json.dumps(payload) if payload is not None else json.dumps({})

        with self._write_lock, self.transaction() as conn:
            cursor = self._cursor_exec(
                conn,
                """
                    INSERT INTO prompt_studio_job_queue (
                        uuid, job_type, entity_id, project_id, priority, status,
                        payload, max_retries, client_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    RETURNING *
                    """,
                (
                    job_uuid,
                    job_type,
                    entity_id,
                    project_id,
                    priority,
                    status,
                    payload_json,
                    max_retries,
                    client_id or self.client_id,
                ),
            )
            row = cursor.fetchone()
            if not row:
                raise DatabaseError("Failed to create prompt studio job queue record")  # noqa: TRY003
            job = self._row_to_dict(cursor, row)
        return job or {}

    def get_job(self, job_id: int) -> Optional[dict[str, Any]]:
        try:
            cursor = self._execute(
                "SELECT * FROM prompt_studio_job_queue WHERE id = ? LIMIT 1",
                (job_id,),
            )
            row = cursor.fetchone()
            return self._row_to_dict(cursor, row) if row else None
        except BackendDatabaseError as exc:
            raise DatabaseError(f"Failed to fetch job {job_id}: {exc}") from exc  # noqa: TRY003

    def get_job_by_uuid(self, job_uuid: str) -> Optional[dict[str, Any]]:
        try:
            cursor = self._execute(
                "SELECT * FROM prompt_studio_job_queue WHERE uuid = ? LIMIT 1",
                (job_uuid,),
            )
            row = cursor.fetchone()
            return self._row_to_dict(cursor, row) if row else None
        except BackendDatabaseError as exc:
            raise DatabaseError(f"Failed to fetch job {job_uuid}: {exc}") from exc  # noqa: TRY003

    def list_jobs(
        self,
        *,
        status: Optional[str] = None,
        job_type: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if status:
            clauses.append("status = ?")
            params.append(status)
        if job_type:
            clauses.append("job_type = ?")
            params.append(job_type)

        where_clause = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        query = """
            SELECT *
            FROM prompt_studio_job_queue
            {where_clause}
            ORDER BY priority DESC, created_at ASC
            LIMIT ?
        """.format_map(locals())  # nosec B608
        params_with_limit = list(params) + [limit]

        try:
            cursor = self._execute(query, params_with_limit)
            rows = cursor.fetchall()
            return [self._row_to_dict(cursor, row) for row in rows if row]
        except BackendDatabaseError as exc:
            raise DatabaseError(f"Failed to list prompt studio jobs: {exc}") from exc  # noqa: TRY003

    def update_job_status(
        self,
        job_id: int,
        status: str,
        *,
        error_message: Optional[str] = None,
        result: Optional[Any] = None,
    ) -> Optional[dict[str, Any]]:
        updates = ["status = ?"]
        params: list[Any] = [status]

        if status == "processing":
            updates.append("started_at = CURRENT_TIMESTAMP")
            # Extend lease window when explicitly setting processing
            updates.append("leased_until = NOW() + INTERVAL '60 seconds'")
        elif status in {"completed", "failed", "cancelled"}:
            updates.append("completed_at = CURRENT_TIMESTAMP")
            # Clear lease on terminal states
            updates.append("leased_until = NULL")
            updates.append("lease_owner = NULL")

        if error_message is not None:
            updates.append("error_message = ?")
            params.append(error_message)

        if result is not None:
            updates.append("result = ?")
            params.append(json.dumps(result))

        params.append(job_id)
        updates_sql = ', '.join(updates)

        query = """
            UPDATE prompt_studio_job_queue
            SET {updates_sql}
            WHERE id = ?
            RETURNING *
        """.format_map(locals())  # nosec B608

        try:
            with self.transaction() as conn:
                cursor = self._cursor_exec(conn, query, params)
                row = cursor.fetchone()
                record = self._row_to_dict(cursor, row) if row else None
                # Release advisory lock on terminal states
                if record and status in {"completed", "failed", "cancelled"}:
                    with suppress(BackendDatabaseError):
                        self._execute("SELECT pg_advisory_unlock(?)", (job_id,))
                return record
        except BackendDatabaseError as exc:
            raise DatabaseError(f"Failed to update job {job_id}: {exc}") from exc  # noqa: TRY003

    def renew_job_lease(self, job_id: int, seconds: int = 60, worker_id: Optional[str] = None) -> bool:
        try:
            seconds = max(1, min(3600, int(seconds)))
        except _PROMPT_STUDIO_NONCRITICAL_EXCEPTIONS:
            seconds = 60

        owner_value: Optional[str] = None
        if worker_id:
            try:
                owner_value = str(worker_id).strip()[:128]
                if not owner_value:
                    owner_value = None
            except _PROMPT_STUDIO_NONCRITICAL_EXCEPTIONS:
                owner_value = None

        set_owner_sql = ", lease_owner = COALESCE(?, lease_owner)" if owner_value is not None else ""
        owner_guard_sql = " AND (lease_owner IS NULL OR lease_owner = ?)" if owner_value is not None else ""
        params: list[Any] = [job_id]
        if owner_value is not None:
            params = [owner_value, job_id, owner_value]

        try:
            with self.transaction() as conn:
                cursor = self._cursor_exec(
                    conn,
                    """
                    UPDATE prompt_studio_job_queue
                    SET leased_until = CASE
                            WHEN leased_until IS NOT NULL AND leased_until > NOW()
                                THEN leased_until + INTERVAL '{seconds} seconds'
                            ELSE NOW() + INTERVAL '{seconds} seconds'
                        END{set_owner_sql}
                    WHERE id = ?
                      AND status = 'processing'
                      {owner_guard_sql}
                    RETURNING id
                    """.format_map(locals()),  # nosec B608
                    tuple(params),
                )
                row = cursor.fetchone()
                return bool(row)
        except BackendDatabaseError as exc:
            raise DatabaseError(f"Failed to renew job lease for {job_id}: {exc}") from exc  # noqa: TRY003

    def acquire_next_job(self, worker_id: Optional[str] = None) -> Optional[dict[str, Any]]:
        with self._write_lock:  # noqa: SIM117
            with self.transaction() as conn:
                cursor = conn.cursor()
                owner_value: Optional[str] = None
                if worker_id:
                    try:
                        owner_value = str(worker_id).strip()[:128]
                        if not owner_value:
                            owner_value = None
                    except _PROMPT_STUDIO_NONCRITICAL_EXCEPTIONS:
                        owner_value = None
                if self.backend_type == BackendType.POSTGRESQL:
                    import os as _os
                    try:
                        _lease_secs = max(1, min(3600, int(_os.getenv("TLDW_PS_JOB_LEASE_SECONDS", "60"))))
                    except _PROMPT_STUDIO_NONCRITICAL_EXCEPTIONS:
                        _lease_secs = 60
                    # Acquire using advisory lock as a gate to avoid double-processing across processes
                    # Metrics: advisory lock attempt
                    _psm = None
                    try:
                        from tldw_Server_API.app.core.Prompt_Management.prompt_studio.monitoring import (
                            prompt_studio_metrics as _psm,
                        )
                    except _PROMPT_STUDIO_NONCRITICAL_EXCEPTIONS:
                        _psm = None

                    def _inc_metric(name: str, labels: Optional[dict[str, str]] = None) -> None:
                        if _psm is None:
                            return
                        with suppress(_PROMPT_STUDIO_NONCRITICAL_EXCEPTIONS):
                            _psm.metrics_manager.increment(name, labels=labels)

                    _inc_metric("prompt_studio.pg_advisory.lock_attempts_total")
                    cursor.execute(
                        """
                        WITH candidate AS (
                            SELECT id,
                                   (status = 'processing' AND (leased_until IS NULL OR leased_until < NOW())) AS was_reclaim
                            FROM prompt_studio_job_queue
                            WHERE (status = 'queued'
                                   OR (status = 'processing' AND (leased_until IS NULL OR leased_until < NOW())))
                            ORDER BY priority DESC, created_at ASC
                            LIMIT 10
                        ), locked AS (
                            SELECT id, was_reclaim
                            FROM candidate
                            WHERE pg_try_advisory_xact_lock(id)
                            LIMIT 1
                        )
                        UPDATE prompt_studio_job_queue AS q
                        SET status = 'processing',
                            started_at = CURRENT_TIMESTAMP,
                            leased_until = NOW() + INTERVAL '{_lease_secs} seconds',
                            lease_owner = COALESCE(%s, lease_owner)
                        FROM locked
                        WHERE q.id = locked.id
                          AND (
                              q.status = 'queued'
                              OR (q.status = 'processing' AND (q.leased_until IS NULL OR q.leased_until < NOW()))
                          )
                        RETURNING q.*, locked.was_reclaim
                        """.format_map(locals()),  # nosec B608
                        (owner_value,),
                    )
                    row = cursor.fetchone()
                    if not row:
                        return None
                    # Metrics: locks acquired
                    _inc_metric("prompt_studio.pg_advisory.locks_acquired_total")
                    _inc_metric("prompt_studio.pg_advisory.unlocks_total")
                    # Transaction-scoped advisory lock releases on commit; no manual unlock needed.
                    record = self._row_to_dict(cursor, row)
                    # Record queue latency for Postgres
                    try:
                        from datetime import datetime
                        created = record.get("created_at")
                        started = record.get("started_at")
                        def _parse(v):
                            if v is None:
                                return None
                            if isinstance(v, datetime):
                                return v
                            try:
                                return datetime.fromisoformat(str(v).replace("Z", "+00:00"))
                            except _PROMPT_STUDIO_NONCRITICAL_EXCEPTIONS:
                                return None
                        cdt = _parse(created)
                        sdt = _parse(started)
                        if cdt and sdt:
                            qlat = max(0.0, (sdt - cdt).total_seconds())
                            if _psm is not None:
                                with suppress(_PROMPT_STUDIO_NONCRITICAL_EXCEPTIONS):
                                    _psm.metrics_manager.observe(
                                        "jobs.queue_latency_seconds",
                                        qlat,
                                        labels={"job_type": str(record.get("job_type", ""))},
                                    )
                    except _PROMPT_STUDIO_NONCRITICAL_EXCEPTIONS:
                        pass
                    # Increment reclaims if applicable
                    try:
                        was_reclaim = False
                        try:
                            was_reclaim = bool(row["was_reclaim"])  # dict_row in pg
                        except _PROMPT_STUDIO_NONCRITICAL_EXCEPTIONS:
                            # positional
                            was_reclaim = bool(row[-1])
                        if was_reclaim:
                            _inc_metric("jobs.reclaims_total", labels={"job_type": str(record.get("job_type", ""))})
                    except _PROMPT_STUDIO_NONCRITICAL_EXCEPTIONS:
                        pass
                    return record
                else:
                    cursor.execute(
                        """
                        SELECT id
                        FROM prompt_studio_job_queue
                        WHERE (status = 'queued' OR (status = 'processing' AND (leased_until IS NULL OR leased_until <= CURRENT_TIMESTAMP)))
                        ORDER BY priority DESC, created_at ASC
                        LIMIT 1
                        """,
                        (owner_value,),
                    )
                    job_row = cursor.fetchone()
                    if not job_row:
                        return None
                    job_id = job_row[0]
                    import os as _os2
                    try:
                        _lease_secs2 = max(1, min(3600, int(_os2.getenv("TLDW_PS_JOB_LEASE_SECONDS", "60"))))
                    except _PROMPT_STUDIO_NONCRITICAL_EXCEPTIONS:
                        _lease_secs2 = 60
                    cursor.execute(
                        """
                        UPDATE prompt_studio_job_queue
                        SET status = 'processing',
                            started_at = CURRENT_TIMESTAMP,
                            leased_until = DATETIME('now', '+{_lease_secs2} seconds'),
                            lease_owner = COALESCE(?, lease_owner)
                        WHERE id = ?
                          AND (
                              status = 'queued'
                              OR (status = 'processing' AND (leased_until IS NULL OR leased_until <= CURRENT_TIMESTAMP))
                          )
                        RETURNING *
                        """.format_map(locals()),  # nosec B608
                        (owner_value, job_id),
                    )

                row = cursor.fetchone()
                if not row:
                    return None
                job = self._row_to_dict(cursor, row)
                # Record queue latency for SQLite
                try:
                    from datetime import datetime
                    created = job.get("created_at")
                    started = job.get("started_at")
                    def _parse(v):
                        if v is None:
                            return None
                        if isinstance(v, datetime):
                            return v
                        try:
                            return datetime.fromisoformat(str(v).replace("Z", "+00:00"))
                        except _PROMPT_STUDIO_NONCRITICAL_EXCEPTIONS:
                            return None
                    cdt = _parse(created)
                    sdt = _parse(started)
                    if cdt and sdt:
                        qlat = max(0.0, (sdt - cdt).total_seconds())
                        try:
                            from tldw_Server_API.app.core.Prompt_Management.prompt_studio.monitoring import (
                                prompt_studio_metrics as _psm2,
                            )
                            _psm2.metrics_manager.observe(
                                "jobs.queue_latency_seconds",
                                qlat,
                                labels={"job_type": str(job.get("job_type", ""))},
                            )
                        except _PROMPT_STUDIO_NONCRITICAL_EXCEPTIONS:
                            pass
                except _PROMPT_STUDIO_NONCRITICAL_EXCEPTIONS:
                    pass
        return job

    def retry_job_record(self, job_id: int) -> bool:
        with self._write_lock, self.transaction() as conn:
            cursor = self._cursor_exec(
                conn,
                """
                    UPDATE prompt_studio_job_queue
                    SET status = 'queued',
                        retry_count = retry_count + 1,
                        error_message = NULL,
                        started_at = NULL,
                        completed_at = NULL,
                        leased_until = NULL,
                        lease_owner = NULL
                    WHERE id = ?
                    RETURNING retry_count, max_retries
                    """,
                (job_id,),
            )
            row = cursor.fetchone()
            success = row is not None
            if success:
                with suppress(BackendDatabaseError):
                    self._execute("SELECT pg_advisory_unlock(?)", (job_id,))
            return success

    def cleanup_jobs(self, older_than_days: int = 30) -> int:
        cutoff = datetime.now(timezone.utc) - timedelta(days=older_than_days)
        try:
            cursor = self._execute(
                """
                DELETE FROM prompt_studio_job_queue
                WHERE status IN ('completed', 'failed', 'cancelled')
                  AND completed_at IS NOT NULL
                  AND completed_at < ?
                """,
                (cutoff.isoformat(),),
            )
            return cursor.rowcount  # noqa: TRY300
        except BackendDatabaseError as exc:
            raise DatabaseError(f"Failed cleaning up old jobs: {exc}") from exc  # noqa: TRY003

    def get_latest_job_for_entity(self, job_type: str, entity_id: int) -> Optional[dict[str, Any]]:
        query = """
            SELECT *
            FROM prompt_studio_job_queue
            WHERE job_type = ? AND entity_id = ?
            ORDER BY created_at DESC, id DESC
            LIMIT 1
        """
        try:
            cursor = self._execute(query, (job_type, entity_id))
            row = cursor.fetchone()
            return self._row_to_dict(cursor, row) if row else None
        except BackendDatabaseError as exc:
            raise DatabaseError(  # noqa: TRY003
                f"Failed fetching latest job for entity {entity_id}: {exc}"
            ) from exc

    def list_jobs_for_entity(
        self,
        job_type: str,
        entity_id: int,
        *,
        limit: int = 50,
        ascending: bool = True,
    ) -> list[dict[str, Any]]:
        order_clause = "ASC" if ascending else "DESC"
        query = """
            SELECT *
            FROM prompt_studio_job_queue
            WHERE job_type = ? AND entity_id = ?
            ORDER BY created_at {order_clause}, id {order_clause}
            LIMIT ?
        """.format_map(locals())  # nosec B608
        try:
            cursor = self._execute(query, (job_type, entity_id, limit))
            rows = cursor.fetchall()
            return [self._row_to_dict(cursor, row) for row in rows if row]
        except BackendDatabaseError as exc:
            raise DatabaseError(  # noqa: TRY003
                f"Failed listing jobs for entity {entity_id}: {exc}"
            ) from exc

    # --- Test case helpers -------------------------------------------------

########################################################################################################################
# Prompt Studio Database Class

class _SQLitePromptStudioDatabase(PromptsDatabase):
    """
    Extends PromptsDatabase with Prompt Studio specific functionality.
    Manages projects, signatures, test cases, evaluations, and optimizations.
    """

    _PROMPT_STUDIO_SCHEMA_VERSION = 1

    def _sqlite_journal_mode(self) -> str | None:
        if self.is_memory_db:
            return None
        return "WAL" if _should_enable_prompt_studio_sqlite_wal() else "DELETE"

    def __init__(self, db_path: Union[str, Path], client_id: str):
        """
        Initialize PromptStudioDatabase with path and client ID.

        Args:
            db_path: Path to the database file
            client_id: Client identifier for sync logging
        """
        # Initialize parent class
        super().__init__(db_path, client_id)
        # Mark backend type for helper branches reused from backend-aware implementation
        self.backend_type = BackendType.SQLITE

        # Create a write lock for serializing write operations
        self._write_lock = threading.RLock()

        # Initialize prompt studio schema
        self._init_prompt_studio_schema()

        logger.info(f"PromptStudioDatabase initialized for {db_path} with client {client_id}")

    def _init_prompt_studio_schema(self):
        """Initialize Prompt Studio specific schema."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            # Check if prompt studio tables exist
            cursor.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='prompt_studio_projects'
            """)

            if not cursor.fetchone():
                logger.info("Initializing Prompt Studio schema...")
                self._apply_prompt_studio_migrations(conn)
            # Ensure auxiliary tables exist even on existing DBs
            try:
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS prompt_studio_idempotency (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        entity_type TEXT NOT NULL,
                        idempotency_key TEXT NOT NULL,
                        entity_id INTEGER NOT NULL,
                        user_id TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
                # Composite uniqueness per user; SQLite treats NULLs as distinct, which is acceptable here
                cursor.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS uq_ps_idem_user ON prompt_studio_idempotency(entity_type, idempotency_key, user_id)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_ps_idem_entity ON prompt_studio_idempotency(entity_type, user_id)"
                )
                conn.commit()
            except _PROMPT_STUDIO_NONCRITICAL_EXCEPTIONS as _e:
                logger.warning(f"Failed ensuring idempotency table: {_e}")

            # Ensure leasing columns exist on job queue (SQLite)
            try:
                cursor.execute("SELECT leased_until FROM prompt_studio_job_queue LIMIT 1")
            except _PROMPT_STUDIO_NONCRITICAL_EXCEPTIONS:
                try:
                    cursor.execute("ALTER TABLE prompt_studio_job_queue ADD COLUMN leased_until TIMESTAMP")
                    conn.commit()
                except _PROMPT_STUDIO_NONCRITICAL_EXCEPTIONS:
                    pass
            try:
                cursor.execute("SELECT lease_owner FROM prompt_studio_job_queue LIMIT 1")
            except _PROMPT_STUDIO_NONCRITICAL_EXCEPTIONS:
                try:
                    cursor.execute("ALTER TABLE prompt_studio_job_queue ADD COLUMN lease_owner TEXT")
                    conn.commit()
                except _PROMPT_STUDIO_NONCRITICAL_EXCEPTIONS:
                    pass
            # Ensure optimization test_case_ids column exists (SQLite)
            try:
                cursor.execute("SELECT test_case_ids FROM prompt_studio_optimizations LIMIT 1")
            except _PROMPT_STUDIO_NONCRITICAL_EXCEPTIONS:
                try:
                    cursor.execute("ALTER TABLE prompt_studio_optimizations ADD COLUMN test_case_ids JSON")
                    conn.commit()
                except _PROMPT_STUDIO_NONCRITICAL_EXCEPTIONS:
                    pass
            try:
                cursor.execute("SELECT prompt_format FROM prompt_studio_prompts LIMIT 1")
            except _PROMPT_STUDIO_NONCRITICAL_EXCEPTIONS:
                try:
                    cursor.execute(
                        "ALTER TABLE prompt_studio_prompts ADD COLUMN prompt_format TEXT NOT NULL DEFAULT 'legacy'"
                    )
                    conn.commit()
                except _PROMPT_STUDIO_NONCRITICAL_EXCEPTIONS:
                    pass
            try:
                cursor.execute("SELECT prompt_schema_version FROM prompt_studio_prompts LIMIT 1")
            except _PROMPT_STUDIO_NONCRITICAL_EXCEPTIONS:
                try:
                    cursor.execute(
                        "ALTER TABLE prompt_studio_prompts ADD COLUMN prompt_schema_version INTEGER"
                    )
                    conn.commit()
                except _PROMPT_STUDIO_NONCRITICAL_EXCEPTIONS:
                    pass
            try:
                cursor.execute("SELECT prompt_definition FROM prompt_studio_prompts LIMIT 1")
            except _PROMPT_STUDIO_NONCRITICAL_EXCEPTIONS:
                try:
                    cursor.execute(
                        "ALTER TABLE prompt_studio_prompts ADD COLUMN prompt_definition JSON"
                    )
                    conn.commit()
                except _PROMPT_STUDIO_NONCRITICAL_EXCEPTIONS:
                    pass

        except _PROMPT_STUDIO_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Error initializing Prompt Studio schema: {e}")
            raise SchemaError(f"Failed to initialize Prompt Studio schema: {e}")  # noqa: B904, TRY003

    # Keep parity with backend helper: local execute that returns a cursor
    def _cursor_exec(self, conn: sqlite3.Connection, query: str, params: Optional[Union[tuple, list, dict, Any]] = None):
        cursor = conn.cursor()
        if params is not None:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        return cursor

    def _execute(
        self,
        query: str,
        params: Optional[Union[tuple, list, dict, Any]] = None,
        *,
        connection: Optional[sqlite3.Connection] = None,
    ):
        conn = connection or self.get_connection()
        return self._cursor_exec(conn, query, params)

    def _executemany(
        self,
        query: str,
        params_list: list[Union[tuple, list, dict, Any]],
        *,
        connection: Optional[sqlite3.Connection] = None,
    ):
        conn = connection or self.get_connection()
        cursor = conn.cursor()
        cursor.executemany(query, params_list)
        return cursor

    def _apply_prompt_studio_migrations(self, conn: sqlite3.Connection):
        """Apply Prompt Studio migration scripts."""
        migrations_dir = Path(__file__).parent / "migrations"

        # List of migration files in order (ensure iterations table exists before indexes)
        migration_files = [
            "001_prompt_studio_schema.sql",
            "003_prompt_studio_iterations.sql",
            "002_prompt_studio_indexes.sql",
            "003_prompt_studio_triggers.sql",
            "004_prompt_studio_fts.sql",
            "006_prompt_studio_structured_prompts.sql",
        ]
        # Allow explicitly skipping FTS migrations when requested, but default to running them
        try:
            import os as _os
            if _os.getenv("SKIP_PROMPT_STUDIO_FTS", "").lower() == "true":
                migration_files = [mf for mf in migration_files if not mf.startswith("004_")]
        except _PROMPT_STUDIO_NONCRITICAL_EXCEPTIONS:
            pass

        for migration_file in migration_files:
            migration_path = migrations_dir / migration_file
            if migration_path.exists():
                logger.info(f"Applying migration: {migration_file}")
                with open(migration_path) as f:
                    migration_sql = f.read()

                # Execute migration statements
                try:
                    conn.executescript(migration_sql)
                    conn.commit()
                    logger.info(f"Successfully applied {migration_file}")
                except _PROMPT_STUDIO_NONCRITICAL_EXCEPTIONS as e:
                    logger.error(f"Failed to apply {migration_file}: {e}")
                    raise SchemaError(f"Migration {migration_file} failed: {e}")  # noqa: B904, TRY003
            else:
                logger.warning(f"Migration file not found: {migration_path}")

    # --- Idempotency helpers (SQLite) ---
    def _idem_lookup(self, entity_type: str, key: str, user_id: Optional[str]) -> Optional[int]:
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            if user_id is None:
                cursor.execute(
                    """
                    SELECT entity_id
                    FROM prompt_studio_idempotency
                    WHERE entity_type = ?
                      AND idempotency_key = ?
                      AND user_id IS NULL
                    LIMIT 1
                    """,
                    (entity_type, key),
                )
            else:
                cursor.execute(
                    """
                    SELECT entity_id
                    FROM prompt_studio_idempotency
                    WHERE entity_type = ?
                      AND idempotency_key = ?
                      AND user_id = ?
                    LIMIT 1
                    """,
                    (entity_type, key, user_id),
                )
            row = cursor.fetchone()
            return int(row[0]) if row else None
        except _PROMPT_STUDIO_NONCRITICAL_EXCEPTIONS:
            return None

    def _idem_record(self, entity_type: str, key: str, entity_id: int, user_id: Optional[str]) -> None:
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR IGNORE INTO prompt_studio_idempotency (entity_type, idempotency_key, entity_id, user_id) VALUES (?, ?, ?, ?)",
                (entity_type, key, entity_id, user_id),
            )
            conn.commit()
        except _PROMPT_STUDIO_NONCRITICAL_EXCEPTIONS:
            pass

    ####################################################################################################################
    # Project Management

    ####################################################################################################################
    # Signature Management

    ####################################################################################################################
    # Test Run Management

    ####################################################################################################################
    # Evaluation Management

    ####################################################################################################################
    # Helper Methods

    def _row_to_dict(self, cursor: sqlite3.Cursor, row: tuple) -> dict[str, Any]:
        """Convert a database row to dictionary."""
        if not row:
            return None

        columns = [description[0] for description in cursor.description]
        result = dict(zip(columns, row))

        # One field list for both implementations; this copy had drifted (no
        # prompt_variant/metrics, so iteration rows came back as raw JSON strings).
        for field in _BackendPromptStudioDatabase._JSON_FIELDS:
            if field in result and result[field]:
                with suppress(json.JSONDecodeError, TypeError):
                    result[field] = json.loads(result[field])

        # Parse datetime fields
        for field in _BackendPromptStudioDatabase._DATETIME_FIELDS:
            if field in result and result[field]:
                try:
                    if isinstance(result[field], str):
                        result[field] = datetime.fromisoformat(result[field])
                except (ValueError, TypeError):
                    pass

        return result

    def _log_sync_event(self, entity: str, entity_uuid: str, operation: str, payload: dict[str, Any]):
        """Log an event to sync_log table if it exists."""
        try:
            with self.transaction() as conn:
                cursor = conn.cursor()

                # Check if sync_log table exists
                cursor.execute(
                    """
                    SELECT name FROM sqlite_master
                    WHERE type='table' AND name='sync_log'
                    """
                )

                if cursor.fetchone():
                    cursor.execute(
                        """
                        INSERT INTO sync_log (
                            entity,
                            entity_uuid,
                            operation,
                            client_id,
                            version,
                            payload,
                            timestamp
                        )
                        VALUES (?, ?, ?, ?, 1, ?, CURRENT_TIMESTAMP)
                        """,
                        (
                            entity,
                            entity_uuid,
                            operation,
                            self.client_id,
                            json.dumps(payload),
                        ),
                    )
        except _PROMPT_STUDIO_NONCRITICAL_EXCEPTIONS as e:
            err_str = str(e).lower()
            if "no such table" in err_str or "does not exist" in err_str:
                logger.debug(f"sync_log table not available: {e}")
            else:
                logger.warning(f"Failed to log sync event for {entity}/{entity_uuid}: {e}")

    # Public convenience alias matching some endpoint call sites
    def row_to_dict(self, row: tuple, cursor: sqlite3.Cursor) -> dict[str, Any]:
        """
        Convert a (row, cursor) pair to a dict. Wrapper around _row_to_dict,
        provided to match call sites that pass (row, cursor) in that order.
        """
        return self._row_to_dict(cursor, row)

    ####################################################################################################################
    # Prompt Accessors (Prompt Studio tables)

    # --- Optimization helpers -------------------------------------------------

    # --- Job queue helpers ---

    def create_job(
        self,
        job_type: str,
        entity_id: int,
        payload: Optional[Any],
        *,
        project_id: Optional[int] = None,
        priority: int = 5,
        status: str = "queued",
        max_retries: int = 3,
        client_id: Optional[str] = None,
    ) -> dict[str, Any]:
        import random
        import time

        conn = self.get_connection()
        job_uuid = str(uuid.uuid4())
        payload_json = json.dumps(payload) if payload is not None else json.dumps({})
        base_delay = 0.05

        for attempt in range(5):
            should_retry = False
            with self._write_lock:
                try:
                    cursor = conn.cursor()
                    cursor.execute(
                        """
                        INSERT INTO prompt_studio_job_queue (
                            uuid, job_type, entity_id, project_id, priority, status,
                            payload, max_retries, client_id
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            job_uuid,
                            job_type,
                            entity_id,
                            project_id,
                            priority,
                            status,
                            payload_json,
                            max_retries,
                            client_id or self.client_id,
                        ),
                    )
                    job_id = cursor.lastrowid
                    conn.commit()
                    return self.get_job(job_id) or {}
                except sqlite3.OperationalError as exc:
                    if "database is locked" in str(exc).lower() and attempt < 4:
                        should_retry = True
                        delay = base_delay * (2 ** attempt) * (0.5 + random.random())
                    else:
                        raise DatabaseError(f"Failed to create job: {exc}") from exc  # noqa: TRY003
                except sqlite3.Error as exc:  # noqa: BLE001
                    raise DatabaseError(f"Failed to create job: {exc}") from exc  # noqa: TRY003

            if should_retry:
                time.sleep(delay)

        raise DatabaseError("Failed to create job due to database locks")  # noqa: TRY003

    def get_job(self, job_id: int) -> Optional[dict[str, Any]]:
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM prompt_studio_job_queue WHERE id = ?",
                (job_id,),
            )
            row = cursor.fetchone()
            return self._row_to_dict(cursor, row) if row else None
        except sqlite3.Error as exc:  # noqa: BLE001
            raise DatabaseError(f"Failed to fetch job {job_id}: {exc}") from exc  # noqa: TRY003

    def get_job_by_uuid(self, job_uuid: str) -> Optional[dict[str, Any]]:
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM prompt_studio_job_queue WHERE uuid = ?",
                (job_uuid,),
            )
            row = cursor.fetchone()
            return self._row_to_dict(cursor, row) if row else None
        except sqlite3.Error as exc:  # noqa: BLE001
            raise DatabaseError(f"Failed to fetch job {job_uuid}: {exc}") from exc  # noqa: TRY003

    def list_jobs(
        self,
        *,
        status: Optional[str] = None,
        job_type: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            query = "SELECT * FROM prompt_studio_job_queue WHERE 1=1"
            params: list[Any] = []
            if status:
                query += " AND status = ?"
                params.append(status)
            if job_type:
                query += " AND job_type = ?"
                params.append(job_type)
            query += " ORDER BY priority DESC, created_at ASC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [self._row_to_dict(cursor, row) for row in rows if row]
        except sqlite3.Error as exc:  # noqa: BLE001
            raise DatabaseError(f"Failed to list prompt studio jobs: {exc}") from exc  # noqa: TRY003

    def update_job_status(
        self,
        job_id: int,
        status: str,
        *,
        error_message: Optional[str] = None,
        result: Optional[Any] = None,
    ) -> Optional[dict[str, Any]]:
        import random
        import time

        conn = self.get_connection()
        base_delay = 0.05

        for attempt in range(5):
            should_retry = False
            with self._write_lock:
                try:
                    cursor = conn.cursor()
                    updates = ["status = ?"]
                    params: list[Any] = [status]

                    if status == "processing":
                        updates.append("started_at = CURRENT_TIMESTAMP")
                        # Extend lease window on explicit processing state
                        import os as _os_ul
                        try:
                            _lease_secs_upd = max(1, min(3600, int(_os_ul.getenv("TLDW_PS_JOB_LEASE_SECONDS", "60"))))
                        except _PROMPT_STUDIO_NONCRITICAL_EXCEPTIONS:
                            _lease_secs_upd = 60
                        updates.append(f"leased_until = DATETIME('now', '+{_lease_secs_upd} seconds')")
                    elif status in {"completed", "failed", "cancelled"}:
                        updates.append("completed_at = CURRENT_TIMESTAMP")
                        updates.append("leased_until = NULL")
                        updates.append("lease_owner = NULL")

                    if error_message is not None:
                        updates.append("error_message = ?")
                        params.append(error_message)

                    if result is not None:
                        updates.append("result = ?")
                        params.append(json.dumps(result))

                    params.append(job_id)
                    updates_sql = ', '.join(updates)

                    cursor.execute(
                        """
                        UPDATE prompt_studio_job_queue
                        SET {updates_sql}
                        WHERE id = ?
                        """.format_map(locals()),  # nosec B608
                        params,
                    )

                    if cursor.rowcount > 0:
                        conn.commit()
                        cursor.execute(
                            "SELECT * FROM prompt_studio_job_queue WHERE id = ?",
                            (job_id,),
                        )
                        row = cursor.fetchone()
                        return self._row_to_dict(cursor, row) if row else None
                    return None  # noqa: TRY300
                except sqlite3.OperationalError as exc:
                    if "database is locked" in str(exc).lower() and attempt < 4:
                        should_retry = True
                        delay = base_delay * (2 ** attempt) * (0.5 + random.random())
                    else:
                        raise DatabaseError(f"Failed to update job {job_id}: {exc}") from exc  # noqa: TRY003
                except sqlite3.Error as exc:  # noqa: BLE001
                    raise DatabaseError(f"Failed to update job {job_id}: {exc}") from exc  # noqa: TRY003

            if should_retry:
                time.sleep(delay)

        raise DatabaseError("Failed to update job status due to database locks")  # noqa: TRY003

    def acquire_next_job(self, worker_id: Optional[str] = None) -> Optional[dict[str, Any]]:
        import random
        import time

        conn = self.get_connection()
        base_delay = 0.05
        owner_value: Optional[str] = None
        if worker_id:
            try:
                owner_value = str(worker_id).strip()[:128]
                if not owner_value:
                    owner_value = None
            except _PROMPT_STUDIO_NONCRITICAL_EXCEPTIONS:
                owner_value = None

        for attempt in range(5):
            should_retry = False
            with self._write_lock:
                try:
                    cursor = conn.cursor()
                    cursor.execute(
                        """
                        SELECT id
                        FROM prompt_studio_job_queue
                        WHERE (status = 'queued' OR (status = 'processing' AND (leased_until IS NULL OR leased_until <= CURRENT_TIMESTAMP)))
                        ORDER BY priority DESC, created_at ASC
                        LIMIT 1
                        """,
                    )
                    row = cursor.fetchone()
                    if not row:
                        return None
                    job_id = row[0]

                    # Determine lease window from env
                    import os as _os_s1
                    try:
                        _lease_secs_sqlite = max(1, min(3600, int(_os_s1.getenv("TLDW_PS_JOB_LEASE_SECONDS", "60"))))
                    except _PROMPT_STUDIO_NONCRITICAL_EXCEPTIONS:
                        _lease_secs_sqlite = 60
                    query = (
                        "UPDATE prompt_studio_job_queue "  # nosec B608
                        "SET status = 'processing', "
                        "    started_at = CURRENT_TIMESTAMP, "
                        f"    leased_until = DATETIME('now', '+{_lease_secs_sqlite} seconds'), "
                        "    lease_owner = COALESCE(?, lease_owner) "
                        "WHERE id = ? "
                        "  AND (status = 'queued' OR (status = 'processing' AND (leased_until IS NULL OR leased_until <= CURRENT_TIMESTAMP)))"
                    )
                    cursor.execute(query, (owner_value, job_id))
                    try:
                        row = cursor.fetchone()
                    except _PROMPT_STUDIO_NONCRITICAL_EXCEPTIONS:
                        row = None

                    if row is not None or cursor.rowcount > 0:
                        conn.commit()
                        job = self._row_to_dict(cursor, row) if row is not None else self.get_job(job_id)
                        # Record queue latency (started_at - created_at)
                        try:
                            from datetime import datetime
                            if job:
                                created = job.get("created_at")
                                started = job.get("started_at")
                                def _parse(v):
                                    if v is None:
                                        return None
                                    if isinstance(v, datetime):
                                        return v
                                    try:
                                        return datetime.fromisoformat(str(v).replace("Z", "+00:00"))
                                    except _PROMPT_STUDIO_NONCRITICAL_EXCEPTIONS:
                                        return None
                                cdt = _parse(created)
                                sdt = _parse(started)
                                if cdt and sdt:
                                    qlat = max(0.0, (sdt - cdt).total_seconds())
                                    try:
                                        from tldw_Server_API.app.core.Prompt_Management.prompt_studio.monitoring import (
                                            prompt_studio_metrics as _psm2,
                                        )
                                        _psm2.metrics_manager.observe(
                                            "jobs.queue_latency_seconds",
                                            qlat,
                                            labels={"job_type": str(job.get("job_type", ""))},
                                        )
                                    except _PROMPT_STUDIO_NONCRITICAL_EXCEPTIONS:
                                        pass
                        except _PROMPT_STUDIO_NONCRITICAL_EXCEPTIONS:
                            pass
                        return job
                    # Lost race to another worker updating this job; retry selection
                    should_retry = True
                    delay = base_delay * (2 ** attempt) * (0.5 + random.random())
                except sqlite3.OperationalError as exc:
                    if "database is locked" in str(exc).lower() and attempt < 4:
                        should_retry = True
                        delay = base_delay * (2 ** attempt) * (0.5 + random.random())
                    else:
                        raise DatabaseError(f"Failed to acquire job: {exc}") from exc  # noqa: TRY003
                except sqlite3.Error as exc:  # noqa: BLE001
                    raise DatabaseError(f"Failed to acquire job: {exc}") from exc  # noqa: TRY003

            if should_retry:
                try:
                    time.sleep(delay)
                except _PROMPT_STUDIO_NONCRITICAL_EXCEPTIONS:
                    time.sleep(0.01)

        raise DatabaseError("Failed to acquire job due to database locks or contention")  # noqa: TRY003

    def retry_job_record(self, job_id: int) -> bool:
        import random
        import time

        conn = self.get_connection()
        base_delay = 0.05

        for attempt in range(5):
            should_retry = False
            with self._write_lock:
                try:
                    cursor = conn.cursor()
                    cursor.execute(
                        """
                        UPDATE prompt_studio_job_queue
                        SET status = 'queued',
                            retry_count = retry_count + 1,
                            error_message = NULL,
                            started_at = NULL,
                            completed_at = NULL,
                            leased_until = NULL,
                            lease_owner = NULL
                        WHERE id = ?
                        """,
                        (job_id,),
                    )
                    success = cursor.rowcount > 0
                    if success:
                        conn.commit()
                        return True
                    # Fallback: if guard matched and row already had identical values, treat as success
                    try:
                        cursor.execute(
                            "SELECT status, lease_owner, leased_until FROM prompt_studio_job_queue WHERE id = ?",
                            (job_id,),
                        )
                        row2 = cursor.fetchone()
                        if row2:
                            st = str(row2[0]) if row2[0] is not None else ""
                            owner = str(row2[1]) if row2[1] is not None else None
                            if st.lower() == "processing" and owner is None:
                                return True
                    except _PROMPT_STUDIO_NONCRITICAL_EXCEPTIONS:
                        pass
                    return False  # noqa: TRY300
                except sqlite3.OperationalError as exc:
                    if "database is locked" in str(exc).lower() and attempt < 4:
                        should_retry = True
                        delay = base_delay * (2 ** attempt) * (0.5 + random.random())
                    else:
                        raise DatabaseError(f"Failed to reschedule job {job_id}: {exc}") from exc  # noqa: TRY003
                except sqlite3.Error as exc:  # noqa: BLE001
                    raise DatabaseError(f"Failed to reschedule job {job_id}: {exc}") from exc  # noqa: TRY003

            if should_retry:
                time.sleep(delay)

        return False

    def cleanup_jobs(self, older_than_days: int = 30) -> int:
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cutoff = (datetime.utcnow() - timedelta(days=older_than_days)).isoformat()
            cursor.execute(
                """
                DELETE FROM prompt_studio_job_queue
                WHERE status IN ('completed', 'failed', 'cancelled')
                  AND completed_at IS NOT NULL
                  AND completed_at < ?
                """,
                (cutoff,),
            )
            deleted = cursor.rowcount
            if deleted:
                conn.commit()
            return deleted  # noqa: TRY300
        except sqlite3.Error as exc:  # noqa: BLE001
            raise DatabaseError(f"Failed to clean up old jobs: {exc}") from exc  # noqa: TRY003

    def get_latest_job_for_entity(self, job_type: str, entity_id: int) -> Optional[dict[str, Any]]:
        conn = self.get_connection()
        cursor = conn.cursor()
        query = """
            SELECT *
            FROM prompt_studio_job_queue
            WHERE job_type = ? AND entity_id = ?
            ORDER BY created_at DESC, id DESC
            LIMIT 1
        """
        try:
            cursor.execute(query, (job_type, entity_id))
            row = cursor.fetchone()
            return self._row_to_dict(cursor, row) if row else None
        except sqlite3.Error as exc:  # noqa: BLE001
            raise DatabaseError(  # noqa: TRY003
                f"Failed fetching latest job for entity {entity_id}: {exc}"
            ) from exc

    def list_jobs_for_entity(
        self,
        job_type: str,
        entity_id: int,
        *,
        limit: int = 50,
        ascending: bool = True,
    ) -> list[dict[str, Any]]:
        conn = self.get_connection()
        cursor = conn.cursor()
        order_clause = "ASC" if ascending else "DESC"
        query = (
            f"SELECT * FROM prompt_studio_job_queue "  # nosec B608
            f"WHERE job_type = ? AND entity_id = ? "
            f"ORDER BY created_at {order_clause}, id {order_clause} LIMIT ?"
        )
        try:
            cursor.execute(query, (job_type, entity_id, limit))
            rows = cursor.fetchall()
            return [self._row_to_dict(cursor, row) for row in rows if row]
        except sqlite3.Error as exc:  # noqa: BLE001
            raise DatabaseError(  # noqa: TRY003
                f"Failed listing jobs for entity {entity_id}: {exc}"
            ) from exc

    def renew_job_lease(self, job_id: int, seconds: int = 60, worker_id: Optional[str] = None) -> bool:
        import random
        import time
        try:
            seconds = max(1, min(3600, int(seconds)))
        except _PROMPT_STUDIO_NONCRITICAL_EXCEPTIONS:
            seconds = 60
        owner_value: Optional[str] = None
        if worker_id:
            try:
                owner_value = str(worker_id).strip()[:128]
                if not owner_value:
                    owner_value = None
            except _PROMPT_STUDIO_NONCRITICAL_EXCEPTIONS:
                owner_value = None

        conn = self.get_connection()
        base_delay = 0.05
        for attempt in range(5):
            should_retry = False
            with self._write_lock:
                try:
                    cursor = conn.cursor()
                    set_owner_sql = ", lease_owner = COALESCE(?, lease_owner)" if owner_value is not None else ""
                    owner_guard_sql = " AND (lease_owner IS NULL OR lease_owner = ?)" if owner_value is not None else ""
                    params = (owner_value, job_id, owner_value) if owner_value is not None else (job_id,)
                    cursor.execute(
                        """
                        UPDATE prompt_studio_job_queue
                        SET leased_until = CASE
                                WHEN leased_until IS NOT NULL AND leased_until > CURRENT_TIMESTAMP
                                    THEN DATETIME(leased_until, '+{seconds} seconds')
                                ELSE DATETIME('now', '+{seconds} seconds')
                            END{set_owner_sql}
                        WHERE id = ?
                          AND status = 'processing'
                          {owner_guard_sql}
                        """.format_map(locals()),  # nosec B608
                        params,
                    )
                    success = cursor.rowcount > 0
                    if success:
                        conn.commit()
                    return success  # noqa: TRY300
                except sqlite3.OperationalError as exc:
                    if "database is locked" in str(exc).lower() and attempt < 4:
                        should_retry = True
                        delay = base_delay * (2 ** attempt) * (0.5 + random.random())
                    else:
                        raise DatabaseError(f"Failed to renew job lease for {job_id}: {exc}") from exc  # noqa: TRY003
                except sqlite3.Error as exc:  # noqa: BLE001
                    raise DatabaseError(f"Failed to renew job lease for {job_id}: {exc}") from exc  # noqa: TRY003
            if should_retry:
                time.sleep(delay)
        raise DatabaseError("Failed to renew job lease due to database locks")  # noqa: TRY003

    ####################################################################################################################
    # Test Case Methods

    ####################################################################################################################
    # Transaction Management

    @contextmanager
    def transaction(self):
        """
        Context manager for database transactions.
        Ensures atomic operations with automatic rollback on error.
        """
        conn = self.get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    ####################################################################################################################
    # Optimization helpers (SQLite)


class PromptStudioDatabase:
    """Factory wrapper that selects SQLite or backend-aware implementations."""

    def __init__(
        self,
        db_path: Union[str, Path],
        client_id: str,
        *,
        tenant_user_id: Optional[str] = None,
        backend: Optional[DatabaseBackend] = None,
        config: Optional[ConfigParser] = None,
    ) -> None:
        backend_type = backend.backend_type if backend else BackendType.SQLITE
        if backend_type == BackendType.POSTGRESQL and backend is not None:
            self._impl = _BackendPromptStudioDatabase(
                db_path,
                client_id,
                tenant_user_id=tenant_user_id,
                backend=backend,
                config=config,
            )
        else:
            self._impl = _SQLitePromptStudioDatabase(str(db_path), client_id)
        if tenant_user_id is not None:
            # The wrapper is the object passed to optimizers and other service
            # layers. Preserve the tenant on both surfaces for SQLite as well
            # as PostgreSQL so durable state can never fall back to audit ID.
            self.tenant_user_id = str(tenant_user_id)
            self._impl.tenant_user_id = str(tenant_user_id)

    def __getattr__(self, item):
        return getattr(self._impl, item)

    def __dir__(self):
        return sorted(set(dir(type(self)) + dir(self._impl)))

    def __repr__(self) -> str:  # pragma: no cover - repr helper
        return f"PromptStudioDatabase<{self._impl!r}>"

    @property
    def backend_type(self) -> BackendType:
        return getattr(self._impl, 'backend_type', BackendType.SQLITE)

    @property
    def backend(self) -> Optional[DatabaseBackend]:
        return getattr(self._impl, 'backend', None)

    # Idempotency helpers (public facade)
    def lookup_idempotency(self, entity_type: str, key: str, user_id: Optional[str]) -> Optional[int]:
        if hasattr(self._impl, '_idem_lookup'):
            return self._impl._idem_lookup(entity_type, key, user_id)  # type: ignore[attr-defined]
        return None

    def record_idempotency(self, entity_type: str, key: str, entity_id: int, user_id: Optional[str]) -> None:
        if hasattr(self._impl, '_idem_record'):
            with suppress(_PROMPT_STUDIO_NONCRITICAL_EXCEPTIONS):
                self._impl._idem_record(entity_type, key, entity_id, user_id)  # type: ignore[attr-defined]

    def update_project(self, project_id: int, updates: Optional[dict[str, Any]] = None, **fields: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if updates:
            payload.update(updates)
        if fields:
            payload.update(fields)
        return ProjectsRepository(self._impl).update(project_id, payload)

    def create_project(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return ProjectsRepository(self._impl).create(*args, **kwargs)

    def get_project(self, *args: Any, **kwargs: Any) -> Optional[dict[str, Any]]:
        return ProjectsRepository(self._impl).get(*args, **kwargs)

    def list_projects(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return ProjectsRepository(self._impl).list(*args, **kwargs)

    def delete_project(self, *args: Any, **kwargs: Any) -> bool:
        return ProjectsRepository(self._impl).delete(*args, **kwargs)

    def get_prompt(self, *args: Any, **kwargs: Any) -> Optional[dict[str, Any]]:
        return PromptsRepository(self._impl).get(*args, **kwargs)

    def get_prompt_with_project(self, *args: Any, **kwargs: Any) -> Optional[dict[str, Any]]:
        return PromptsRepository(self._impl).get_with_project(*args, **kwargs)

    def list_prompts(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return PromptsRepository(self._impl).list(*args, **kwargs)

    # Signature delegation ------------------------------------------------

    def create_signature(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return SignaturesRepository(self._impl).create(*args, **kwargs)

    def get_signature(self, *args: Any, **kwargs: Any) -> Optional[dict[str, Any]]:
        return SignaturesRepository(self._impl).get(*args, **kwargs)

    def list_signatures(self, *args: Any, **kwargs: Any) -> Union[dict[str, Any], list[dict[str, Any]]]:
        return SignaturesRepository(self._impl).list(*args, **kwargs)

    def update_signature(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return SignaturesRepository(self._impl).update(*args, **kwargs)

    def delete_signature(self, *args: Any, **kwargs: Any) -> bool:
        return SignaturesRepository(self._impl).delete(*args, **kwargs)

    def create_prompt(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return PromptsRepository(self._impl).create(*args, **kwargs)

    def ensure_prompt_stub(self, *args: Any, **kwargs: Any) -> None:
        return PromptsRepository(self._impl).ensure_stub(*args, **kwargs)

    # Job queue delegation -------------------------------------------------

    def create_job(
        self,
        job_type: str,
        entity_id: int,
        payload: Optional[Any],
        *,
        project_id: Optional[int] = None,
        priority: int = 5,
        status: str = "queued",
        max_retries: int = 3,
        client_id: Optional[str] = None,
    ) -> dict[str, Any]:
        return self._impl.create_job(
            job_type,
            entity_id,
            payload,
            project_id=project_id,
            priority=priority,
            status=status,
            max_retries=max_retries,
            client_id=client_id,
        )

    def get_job(self, job_id: int) -> Optional[dict[str, Any]]:
        return self._impl.get_job(job_id)

    def get_job_by_uuid(self, job_uuid: str) -> Optional[dict[str, Any]]:
        return self._impl.get_job_by_uuid(job_uuid)

    def list_jobs(
        self,
        *,
        status: Optional[str] = None,
        job_type: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        return self._impl.list_jobs(status=status, job_type=job_type, limit=limit)

    def update_job_status(
        self,
        job_id: int,
        status: str,
        *,
        error_message: Optional[str] = None,
        result: Optional[Any] = None,
    ) -> Optional[dict[str, Any]]:
        return self._impl.update_job_status(
            job_id,
            status,
            error_message=error_message,
            result=result,
        )

    def acquire_next_job(self, *, worker_id: Optional[str] = None) -> Optional[dict[str, Any]]:
        return self._impl.acquire_next_job(worker_id=worker_id)

    def retry_job_record(self, job_id: int) -> bool:
        return self._impl.retry_job_record(job_id)

    # Optional: renew job lease
    def renew_job_lease(self, job_id: int, seconds: int = 60, *, worker_id: Optional[str] = None) -> bool:
        if hasattr(self._impl, 'renew_job_lease'):
            try:
                return bool(self._impl.renew_job_lease(job_id, seconds, worker_id=worker_id))  # type: ignore[attr-defined]
            except _PROMPT_STUDIO_NONCRITICAL_EXCEPTIONS:
                return False
        return False

    def cleanup_jobs(self, older_than_days: int = 30) -> int:
        return self._impl.cleanup_jobs(older_than_days)

    def get_latest_job_for_entity(self, *args: Any, **kwargs: Any) -> Optional[dict[str, Any]]:
        return self._impl.get_latest_job_for_entity(*args, **kwargs)

    def list_jobs_for_entity(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        return self._impl.list_jobs_for_entity(*args, **kwargs)
    # Test case delegation -------------------------------------------------

    def create_test_case(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return TestCasesRepository(self._impl).create(*args, **kwargs)

    def get_test_case(self, *args: Any, **kwargs: Any) -> Optional[dict[str, Any]]:
        return TestCasesRepository(self._impl).get(*args, **kwargs)

    def list_test_cases(self, *args: Any, **kwargs: Any) -> Union[dict[str, Any], list[dict[str, Any]]]:
        return TestCasesRepository(self._impl).list(*args, **kwargs)

    def update_test_case(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return TestCasesRepository(self._impl).update(*args, **kwargs)

    def delete_test_case(self, *args: Any, **kwargs: Any) -> bool:
        return TestCasesRepository(self._impl).delete(*args, **kwargs)

    def create_bulk_test_cases(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        return TestCasesRepository(self._impl).create_bulk(*args, **kwargs)

    def search_test_cases(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        return TestCasesRepository(self._impl).search(*args, **kwargs)

    def get_test_cases_by_signature(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        return TestCasesRepository(self._impl).get_by_signature(*args, **kwargs)

    def get_test_case_stats(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return TestCasesRepository(self._impl).stats(*args, **kwargs)

    def get_golden_test_cases(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        return TestCasesRepository(self._impl).get_golden(*args, **kwargs)

    # Test run delegation -------------------------------------------------

    def create_test_run(self, **kwargs: Any) -> dict[str, Any]:
        return TestRunsRepository(self._impl).create(**kwargs)

    def create_prompt_version(self, prompt_id: int, **kwargs: Any) -> dict[str, Any]:
        return PromptVersionsRepository(self._impl).create(prompt_id, **kwargs)

    def revert_prompt_to_version(self, prompt_id: int, target_version: int, **kwargs: Any) -> dict[str, Any]:
        return PromptVersionsRepository(self._impl).revert(prompt_id, target_version, **kwargs)

    def list_prompt_versions(self, project_id: int, prompt_name: str, **kwargs: Any) -> list[dict[str, Any]]:
        return PromptVersionsRepository(self._impl).list(project_id, prompt_name, **kwargs)

    def get_test_cases_by_ids(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        return TestCasesRepository(self._impl).get_by_ids(*args, **kwargs)

    # Evaluation delegation -----------------------------------------------

    def create_evaluation(self, **kwargs: Any) -> dict[str, Any]:
        return EvaluationsRepository(self._impl).create(**kwargs)

    def update_evaluation(self, evaluation_id: int, updates: dict[str, Any]) -> dict[str, Any]:
        return EvaluationsRepository(self._impl).update(evaluation_id, updates)

    def get_evaluation(self, evaluation_id: int) -> Optional[dict[str, Any]]:
        return EvaluationsRepository(self._impl).get(evaluation_id)

    def list_evaluations(self, **kwargs: Any) -> dict[str, Any]:
        return EvaluationsRepository(self._impl).list(**kwargs)

    # Optimization delegation --------------------------------------------

    def create_optimization(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        if "optimization_config" in kwargs:
            kwargs = dict(kwargs)
            kwargs["optimization_config"] = validate_secret_free_optimization_config(
                kwargs["optimization_config"],
            )
        return OptimizationsRepository(self._impl).create(*args, **kwargs)

    def get_optimization(self, *args: Any, **kwargs: Any) -> Optional[dict[str, Any]]:
        return OptimizationsRepository(self._impl).get(*args, **kwargs)

    def list_optimizations(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return OptimizationsRepository(self._impl).list(*args, **kwargs)

    def update_optimization(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        positional = list(args)
        if len(positional) >= 2 and isinstance(positional[1], dict):
            updates = dict(positional[1])
            if "optimization_config" in updates:
                updates["optimization_config"] = validate_secret_free_optimization_config(
                    updates["optimization_config"],
                )
            positional[1] = updates
        elif isinstance(kwargs.get("updates"), dict):
            kwargs = dict(kwargs)
            updates = dict(kwargs["updates"])
            if "optimization_config" in updates:
                updates["optimization_config"] = validate_secret_free_optimization_config(
                    updates["optimization_config"],
                )
            kwargs["updates"] = updates
        return OptimizationsRepository(self._impl).update(*positional, **kwargs)

    def set_optimization_status(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return OptimizationsRepository(self._impl).set_status(*args, **kwargs)

    def complete_optimization(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return OptimizationsRepository(self._impl).complete(*args, **kwargs)

    def complete_optimization_with_transition(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], bool]:
        """Complete an active optimization and report whether this caller won."""

        result = OptimizationsRepository(self._impl).complete(
            *args,
            **kwargs,
            _return_transition_applied=True,
        )
        if not isinstance(result, tuple):
            raise DatabaseError("Optimization transition result is invalid")
        return result

    def record_optimization_iteration(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return OptimizationsRepository(self._impl).record_iteration(*args, **kwargs)

    def list_optimization_iterations(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return OptimizationsRepository(self._impl).list_iterations(*args, **kwargs)
