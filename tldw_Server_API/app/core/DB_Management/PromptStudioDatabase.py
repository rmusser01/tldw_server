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

from pydantic import ValidationError

try:  # psycopg v3 preferred; fall back to psycopg2 if installed
    from psycopg import sql as psycopg_sql  # type: ignore
except ImportError:  # pragma: no cover
    try:
        from psycopg2 import sql as psycopg_sql  # type: ignore
    except ImportError:  # pragma: no cover
        psycopg_sql = None  # type: ignore

from loguru import logger

from .backends.base import (
    BackendType,
    DatabaseBackend,
    QueryResult,
)
from .backends.base import (
    DatabaseError as BackendDatabaseError,
)
from .backends.fts_translator import FTSQueryTranslator
from .backends.query_utils import (
    prepare_backend_many_statement,
    prepare_backend_statement,
    replace_collate_nocase,
    replace_insert_or_ignore,
)

# Local imports
from .Prompts_DB import ConflictError, DatabaseError, InputError, PromptsDatabase, SchemaError
from ..Prompt_Management.structured_prompts import (
    PromptDefinition,
    render_legacy_snapshot,
    validate_prompt_definition,
)

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

    if is_explicit_pytest_runtime() or is_test_mode():
        return False

    return True


def _serialise_tags(tags: Optional[Union[str, Iterable[str]]]) -> Optional[str]:
    """Convert tag collections to a comma-separated string for storage."""

    if tags is None:
        return None

    if isinstance(tags, str):
        return tags

    try:
        return ",".join(
            [
                str(tag).strip()
                for tag in tags
                if str(tag).strip()
            ]
        ) or None
    except TypeError:
        return None


def _parse_tags(value: Any) -> list[str]:
    """Convert stored tag payloads into a list representation."""

    if value is None:
        return []

    if isinstance(value, list):
        return [str(tag).strip() for tag in value if str(tag).strip()]

    if isinstance(value, str):
        return [segment.strip() for segment in value.split(",") if segment.strip()]

    try:
        decoded = bytes(value).decode("utf-8") if isinstance(value, (bytes, bytearray, memoryview)) else None
    except _PROMPT_STUDIO_NONCRITICAL_EXCEPTIONS:  # pragma: no cover - defensive
        decoded = None

    if decoded:
        return [segment.strip() for segment in decoded.split(",") if segment.strip()]

    return []


def _format_test_case_record(record: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Normalise Prompt Studio test case payloads for caller consumption."""

    if record is None:
        return None

    normalised = dict(record)
    normalised["tags"] = _parse_tags(normalised.get("tags"))

    # Ensure boolean fields surface as bools for both backends
    for field in ("is_golden", "is_generated", "deleted"):
        if field in normalised and normalised[field] is not None:
            normalised[field] = bool(normalised[field])

    # JSON fields sometimes arrive as strings; best-effort decoding
    for json_field in ("inputs", "expected_outputs", "actual_outputs"):
        value = normalised.get(json_field)
        if isinstance(value, str):
            with suppress(json.JSONDecodeError, TypeError):
                normalised[json_field] = json.loads(value)

    return normalised


def _render_definition_legacy_fields(definition: PromptDefinition) -> tuple[str, str]:
    messages = [
        {"role": block.role, "content": block.content}
        for block in sorted(definition.blocks, key=lambda item: item.order)
        if block.enabled
    ]
    legacy = render_legacy_snapshot(messages, definition)
    return legacy.system_prompt, legacy.user_prompt


def _prepare_prompt_record_fields(
    *,
    prompt_format: Optional[str],
    prompt_schema_version: Optional[int],
    prompt_definition: Optional[Any],
    system_prompt: Optional[str],
    user_prompt: Optional[str],
    current_prompt: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    effective_format = prompt_format or (
        current_prompt.get("prompt_format") if current_prompt else "legacy"
    ) or "legacy"

    if effective_format == "structured":
        effective_schema_version = prompt_schema_version
        if effective_schema_version is None and current_prompt:
            effective_schema_version = current_prompt.get("prompt_schema_version")
        effective_definition = prompt_definition
        if effective_definition is None and current_prompt:
            effective_definition = current_prompt.get("prompt_definition")

        if effective_schema_version is None:
            raise InputError("Structured prompts require prompt_schema_version.")  # noqa: TRY003
        if not isinstance(effective_definition, dict):
            raise InputError("Structured prompts require prompt_definition.")  # noqa: TRY003

        try:
            definition = PromptDefinition.model_validate(effective_definition)
        except ValidationError as exc:
            raise InputError(f"Invalid prompt_definition: {exc}") from exc  # noqa: TRY003

        issues = validate_prompt_definition(definition)
        if issues:
            raise InputError(issues[0].message)  # noqa: TRY003

        definition_schema_version = int(definition.schema_version)
        if int(effective_schema_version) != definition_schema_version:
            raise InputError(
                "prompt_schema_version must match prompt_definition.schema_version."
            )  # noqa: TRY003

        derived_system_prompt, derived_user_prompt = _render_definition_legacy_fields(definition)
        return {
            "prompt_format": "structured",
            "prompt_schema_version": definition_schema_version,
            "prompt_definition": definition.model_dump(),
            "system_prompt": derived_system_prompt,
            "user_prompt": derived_user_prompt,
        }

    if prompt_definition is not None:
        raise InputError("Legacy prompts cannot include prompt_definition.")  # noqa: TRY003
    if prompt_schema_version is not None:
        raise InputError("Legacy prompts cannot include prompt_schema_version.")  # noqa: TRY003

    return {
        "prompt_format": "legacy",
        "prompt_schema_version": None,
        "prompt_definition": None,
        "system_prompt": (
            system_prompt
            if system_prompt is not None
            else current_prompt.get("system_prompt") if current_prompt else None
        ),
        "user_prompt": (
            user_prompt
            if user_prompt is not None
            else current_prompt.get("user_prompt") if current_prompt else None
        ),
    }


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

    def _release_connection(self, wrapper: Optional[PromptStudioBackendConnectionWrapper]) -> None:
        if not wrapper:
            return
        try:
            raw_conn = wrapper.raw_connection
            self.backend.get_pool().return_connection(raw_conn)
        except BackendDatabaseError as exc:
            logger.warning("Error returning backend connection to pool: {}", exc)

    def _get_thread_connection(self) -> PromptStudioBackendConnectionWrapper:
        wrapper: Optional[PromptStudioBackendConnectionWrapper] = getattr(self._local, 'conn', None)
        if wrapper is not None and not wrapper.closed:
            return wrapper

        raw_conn = self._open_new_connection()
        # Apply per-tenant session guard for PostgreSQL (RLS via current_setting('app.current_user_id'))
        try:
            if self.backend_type == BackendType.POSTGRESQL and self.client_id:
                cur = raw_conn.cursor()
                user_value = str(self.client_id)
                if psycopg_sql is not None:  # type: ignore[name-defined]
                    stmt = psycopg_sql.SQL("SET SESSION app.current_user_id = {}").format(
                        psycopg_sql.Literal(user_value)
                    )
                    cur.execute(stmt)
                else:
                    # Validate input strictly - only allow alphanumeric, dash, underscore, dot
                    import re
                    if not re.match(r'^[\w\-\.]+$', user_value):
                        logger.warning(f"Invalid client_id format for SET SESSION: {user_value[:50]}")
                    else:
                        # Use parameterized query via format_map for safety
                        safe_value = user_value.replace("'", "''").replace("\\", "\\\\")
                        cur.execute(f"SET SESSION app.current_user_id = '{safe_value}'")
                with suppress(_PROMPT_STUDIO_NONCRITICAL_EXCEPTIONS):
                    raw_conn.commit()
        except _PROMPT_STUDIO_NONCRITICAL_EXCEPTIONS:
            # Non-fatal if SET fails
            pass
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

        try:
            if wrapper.raw_connection and getattr(wrapper.raw_connection, 'in_transaction', False):
                with suppress(_PROMPT_STUDIO_NONCRITICAL_EXCEPTIONS):
                    wrapper.rollback()
            self._release_connection(wrapper)
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
        backend: Optional[DatabaseBackend] = None,
        config: Optional[ConfigParser] = None,
    ) -> None:
        super().__init__(db_path, client_id, backend=backend, config=config)
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
                    """,
                    (
                        entity,
                        entity_uuid,
                        operation,
                        self.client_id,
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
    MAX_PROJECT_NAME_LENGTH = 255
    MIN_PROJECT_NAME_LENGTH = 1

    def create_project(
        self,
        name: str,
        description: Optional[str] = None,
        status: str = "draft",
        metadata: Optional[dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> dict[str, Any]:
        # Validate project name
        if not name or not isinstance(name, str):
            raise ValueError("Project name must be a non-empty string")  # noqa: TRY003
        name = name.strip()
        if len(name) < self.MIN_PROJECT_NAME_LENGTH:
            raise ValueError("Project name cannot be empty")  # noqa: TRY003
        if len(name) > self.MAX_PROJECT_NAME_LENGTH:
            raise ValueError(f"Project name cannot exceed {self.MAX_PROJECT_NAME_LENGTH} characters")  # noqa: TRY003

        project_uuid = str(uuid.uuid4())
        payload = (
            project_uuid,
            name,
            description,
            user_id or self.client_id,
            self.client_id,
            status,
            json.dumps(metadata) if metadata is not None else None,
        )

        insert_sql = """
            INSERT INTO prompt_studio_projects
            (uuid, name, description, user_id, client_id, status, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            RETURNING id, uuid, name, description, user_id, client_id, status,
                      deleted, deleted_at, created_at, updated_at, last_modified,
                      version, metadata
        """

        try:
            with self._write_lock, self.transaction() as conn:
                cursor = self._cursor_exec(conn, insert_sql, payload)
                row = cursor.fetchone()
                project = self._row_to_dict(row)
            self._log_sync_event(
                "prompt_studio_project",
                project_uuid,
                "create",
                {
                    "name": name,
                    "description": description,
                    "status": status,
                },
            )
            return project or {}  # noqa: TRY300
        except BackendDatabaseError as exc:
            message = str(exc)
            if 'duplicate' in message.lower() and 'prompt_studio_projects_name_user_id_deleted_key' in message:
                raise ConflictError(f"Project with name '{name}' already exists for this user") from exc  # noqa: TRY003
            raise DatabaseError(f"Failed to create prompt studio project: {exc}") from exc  # noqa: TRY003
        except _PROMPT_STUDIO_NONCRITICAL_EXCEPTIONS as exc:
            # Psycopg unique violations, etc.
            msg = str(exc).lower()
            if 'duplicate' in msg or 'unique constraint' in msg or 'unique violation' in msg:
                raise ConflictError(f"Project with name '{name}' already exists for this user") from exc  # noqa: TRY003
            raise

    def get_project(self, project_id: int, include_deleted: bool = False) -> Optional[dict[str, Any]]:
        clauses = ["id = ?"]
        params: list[Any] = [project_id]
        if not include_deleted:
            clauses.append("deleted = FALSE")
        query = (
            "SELECT id, uuid, name, description, user_id, client_id, status, deleted, deleted_at, "  # nosec B608
            "created_at, updated_at, last_modified, version, metadata "
            "FROM prompt_studio_projects WHERE " + " AND ".join(clauses)
        )
        try:
            cursor = self._execute(query, params)
            row = cursor.fetchone()
            return self._row_to_dict(row)
        except BackendDatabaseError as exc:
            raise DatabaseError(f"Failed to fetch prompt studio project {project_id}: {exc}") from exc  # noqa: TRY003

    def list_projects(
        self,
        user_id: Optional[str] = None,
        status: Optional[str] = None,
        include_deleted: bool = False,
        page: int = 1,
        per_page: int = 20,
        search: Optional[str] = None,
    ) -> dict[str, Any]:
        where_clauses: list[str] = []
        params: list[Any] = []

        if not include_deleted:
            where_clauses.append("deleted = FALSE")
        if user_id:
            where_clauses.append("user_id = ?")
            params.append(user_id)
        if status:
            where_clauses.append("status = ?")
            params.append(status)
        if search:
            where_clauses.append("(name ILIKE ? OR description ILIKE ?)")
            like = f"%{search}%"
            params.extend([like, like])

        where_sql = " WHERE " + " AND ".join(where_clauses) if where_clauses else ""

        count_sql = f"SELECT COUNT(*) AS total FROM prompt_studio_projects{where_sql}"  # nosec B608
        try:
            count_cursor = self._execute(count_sql, params)
            total = count_cursor.fetchone()
            total_count = int(total.get('total', 0)) if total else 0
        except BackendDatabaseError as exc:
            raise DatabaseError(f"Failed counting prompt studio projects: {exc}") from exc  # noqa: TRY003

        offset = (page - 1) * per_page
        list_sql = """
            SELECT p.*,
                   (SELECT COUNT(*) FROM prompt_studio_prompts WHERE project_id = p.id AND deleted = FALSE) AS prompt_count,
                   (SELECT COUNT(*) FROM prompt_studio_test_cases WHERE project_id = p.id AND deleted = FALSE) AS test_case_count
            FROM prompt_studio_projects p
            {where_sql}
            ORDER BY p.updated_at DESC
            LIMIT ?
            OFFSET ?
        """.format_map(locals())  # nosec B608
        params_with_pagination = list(params) + [per_page, offset]

        try:
            cursor = self._execute(list_sql, params_with_pagination)
            rows = cursor.fetchall()
            projects = [self._row_to_dict(row) for row in rows if row]
        except BackendDatabaseError as exc:
            raise DatabaseError(f"Failed listing prompt studio projects: {exc}") from exc  # noqa: TRY003

        return {
            "projects": projects,
            "pagination": {
                "page": page,
                "per_page": per_page,
                "total": total_count,
                "total_pages": (total_count + per_page - 1) // per_page if per_page else 0,
            },
        }

    def create_prompt(
        self,
        project_id: int,
        name: str,
        *,
        signature_id: Optional[int] = None,
        version_number: int = 1,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        prompt_format: str = "legacy",
        prompt_schema_version: Optional[int] = None,
        prompt_definition: Optional[Any] = None,
        few_shot_examples: Optional[Any] = None,
        modules_config: Optional[Any] = None,
        parent_version_id: Optional[int] = None,
        change_description: Optional[str] = None,
        client_id: Optional[str] = None,
    ) -> dict[str, Any]:
        normalized_prompt_fields = _prepare_prompt_record_fields(
            prompt_format=prompt_format,
            prompt_schema_version=prompt_schema_version,
            prompt_definition=prompt_definition,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        prompt_uuid = str(uuid.uuid4())
        payload = (
            prompt_uuid,
            project_id,
            signature_id,
            version_number,
            name,
            normalized_prompt_fields["system_prompt"],
            normalized_prompt_fields["user_prompt"],
            normalized_prompt_fields["prompt_format"],
            normalized_prompt_fields["prompt_schema_version"],
            json.dumps(normalized_prompt_fields["prompt_definition"])
            if normalized_prompt_fields["prompt_definition"] is not None
            else None,
            json.dumps(few_shot_examples) if few_shot_examples is not None else None,
            json.dumps(modules_config) if modules_config is not None else None,
            parent_version_id,
            change_description,
            client_id or self.client_id,
        )

        insert_sql = """
            INSERT INTO prompt_studio_prompts (
                uuid, project_id, signature_id, version_number, name, system_prompt,
                user_prompt, prompt_format, prompt_schema_version, prompt_definition,
                few_shot_examples, modules_config, parent_version_id,
                change_description, client_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            RETURNING id, uuid, project_id, signature_id, version_number, name,
                      system_prompt, user_prompt, prompt_format, prompt_schema_version,
                      prompt_definition, few_shot_examples, modules_config, parent_version_id, change_description, client_id, deleted,
                      deleted_at, created_at, updated_at
        """

        try:
            with self._write_lock, self.transaction() as conn:
                cursor = self._cursor_exec(conn, insert_sql, payload)
                row = cursor.fetchone()
                prompt = self._row_to_dict(row)
            self._log_sync_event(
                "prompt_studio_prompt",
                prompt_uuid,
                "create",
                {
                    "project_id": project_id,
                    "name": name,
                    "version_number": version_number,
                },
            )
            return prompt or {}  # noqa: TRY300
        except BackendDatabaseError as exc:
            message = str(exc).lower()
            if 'duplicate' in message and 'prompt_studio_prompts' in message and 'name' in message:
                raise ConflictError(  # noqa: TRY003
                    f"Prompt with name '{name}' already exists in project {project_id}"
                ) from exc
            raise DatabaseError(f"Failed to create prompt studio prompt: {exc}") from exc  # noqa: TRY003
        except _PROMPT_STUDIO_NONCRITICAL_EXCEPTIONS as exc:
            msg = str(exc).lower()
            if 'duplicate' in msg or 'unique constraint' in msg or 'unique violation' in msg:
                raise ConflictError(  # noqa: TRY003
                    f"Prompt with name '{name}' already exists in project {project_id}"
                ) from exc
            raise

    def update_project(self, project_id: int, updates: dict[str, Any]) -> dict[str, Any]:
        allowed_fields = {"name", "description", "status", "metadata"}
        set_clauses: list[str] = []
        params: list[Any] = []

        # Validate name if being updated
        if "name" in updates:
            name = updates["name"]
            if not name or not isinstance(name, str):
                raise ValueError("Project name must be a non-empty string")  # noqa: TRY003
            name = name.strip()
            if len(name) < self.MIN_PROJECT_NAME_LENGTH:
                raise ValueError("Project name cannot be empty")  # noqa: TRY003
            if len(name) > self.MAX_PROJECT_NAME_LENGTH:
                raise ValueError(f"Project name cannot exceed {self.MAX_PROJECT_NAME_LENGTH} characters")  # noqa: TRY003
            updates["name"] = name

        for field, value in updates.items():
            if field not in allowed_fields:
                continue
            column = field
            if field == "metadata" and value is not None:
                value = json.dumps(value)
            set_clauses.append(f"{column} = ?")
            params.append(value)

        if not set_clauses:
            project = self.get_project(project_id, include_deleted=True)
            if project is None:
                raise InputError(f"Project {project_id} not found or already deleted")  # noqa: TRY003
            return project

        set_clauses.append("updated_at = CURRENT_TIMESTAMP")
        params.append(project_id)

        update_sql = (
            "UPDATE prompt_studio_projects SET "  # nosec B608
            + ", ".join(set_clauses)
            + " WHERE id = ? AND deleted = FALSE RETURNING *"
        )

        try:
            with self._write_lock, self.transaction() as conn:
                cursor = self._cursor_exec(conn, update_sql, params)
                row = cursor.fetchone()
                if not row:
                    raise InputError(f"Project {project_id} not found or already deleted")  # noqa: TRY003
            project = self._row_to_dict(row)
            if project:
                self._log_sync_event(
                    "prompt_studio_project",
                    project.get('uuid', ''),
                    "update",
                    {key: updates[key] for key in updates if key in allowed_fields},
                )
            return project or {}  # noqa: TRY300
        except BackendDatabaseError as exc:
            raise DatabaseError(f"Failed to update prompt studio project {project_id}: {exc}") from exc  # noqa: TRY003

    def delete_project(self, project_id: int, hard_delete: bool = False) -> bool:
        try:
            with self._write_lock, self.transaction() as conn:
                if hard_delete:
                    cursor = self._cursor_exec(
                        conn,
                        "DELETE FROM prompt_studio_projects WHERE id = ? RETURNING uuid",
                        (project_id,),
                    )
                else:
                    cursor = self._cursor_exec(
                        conn,
                        """
                            UPDATE prompt_studio_projects
                            SET deleted = TRUE, deleted_at = CURRENT_TIMESTAMP
                            WHERE id = ? AND deleted = FALSE
                            RETURNING uuid
                            """,
                        (project_id,),
                    )
                row = cursor.fetchone()
                success = row is not None
            if success and row:
                self._log_sync_event(
                    "prompt_studio_project",
                    row.get('uuid', ''),
                    "delete" if hard_delete else "soft_delete",
                    {"hard": hard_delete},
                )
            return success  # noqa: TRY300
        except BackendDatabaseError as exc:
            raise DatabaseError(f"Failed to delete prompt studio project {project_id}: {exc}") from exc  # noqa: TRY003

    # --- Signature helpers -----------------------------------------------

    def create_signature(
        self,
        project_id: int,
        name: str,
        *,
        input_schema: Iterable[Any],
        output_schema: Iterable[Any],
        constraints: Optional[Any] = None,
        validation_rules: Optional[Any] = None,
        client_id: Optional[str] = None,
    ) -> dict[str, Any]:
        if not name or not str(name).strip():
            raise InputError("Signature name cannot be empty")  # noqa: TRY003

        signature_uuid = str(uuid.uuid4())
        payload = (
            signature_uuid,
            project_id,
            str(name).strip(),
            json.dumps(list(input_schema) if input_schema is not None else []),
            json.dumps(list(output_schema) if output_schema is not None else []),
            json.dumps(constraints) if constraints is not None else None,
            json.dumps(validation_rules) if validation_rules is not None else None,
            client_id or self.client_id,
        )

        insert_sql = """
            INSERT INTO prompt_studio_signatures (
                uuid, project_id, name, input_schema, output_schema,
                constraints, validation_rules, client_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            RETURNING *
        """

        try:
            with self._write_lock, self.transaction() as conn:
                cursor = self._cursor_exec(conn, insert_sql, payload)
                row = cursor.fetchone()
                signature = self._row_to_dict(row)

            self._log_sync_event(
                "prompt_studio_signature",
                signature_uuid,
                "create",
                {
                    "project_id": project_id,
                    "name": name,
                },
            )
            return signature or {}  # noqa: TRY300
        except BackendDatabaseError as exc:
            message = str(exc).lower()
            if "duplicate" in message and "prompt_studio_signatures" in message:
                raise ConflictError(  # noqa: TRY003
                    f"Signature with name '{name}' already exists for project {project_id}"
                ) from exc
            raise DatabaseError(f"Failed to create prompt studio signature: {exc}") from exc  # noqa: TRY003

    def get_signature(
        self,
        signature_id: int,
        *,
        include_deleted: bool = False,
    ) -> Optional[dict[str, Any]]:
        clauses = ["id = ?"]
        params: list[Any] = [signature_id]
        if not include_deleted:
            clauses.append("deleted = FALSE")

        query = "SELECT * FROM prompt_studio_signatures WHERE " + " AND ".join(clauses) + " LIMIT 1"  # nosec B608

        try:
            cursor = self._execute(query, params)
            row = cursor.fetchone()
            return self._row_to_dict(cursor, row)
        except BackendDatabaseError as exc:
            raise DatabaseError(f"Failed to fetch signature {signature_id}: {exc}") from exc  # noqa: TRY003

    def list_signatures(
        self,
        project_id: int,
        *,
        include_deleted: bool = False,
        search: Optional[str] = None,
        page: int = 1,
        per_page: int = 20,
        return_pagination: bool = False,
    ) -> Union[dict[str, Any], list[dict[str, Any]]]:
        if page < 1:
            raise InputError("Page index must be >= 1")  # noqa: TRY003
        if per_page < 1:
            raise InputError("Items per page must be >= 1")  # noqa: TRY003

        conditions = ["project_id = ?"]
        params: list[Any] = [project_id]

        if not include_deleted:
            conditions.append("deleted = FALSE")

        if search:
            comparator = "ILIKE" if self.backend_type == BackendType.POSTGRESQL else "LIKE"
            conditions.append(f"name {comparator} ?")
            params.append(f"%{search}%")

        where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""

        count_sql = f"SELECT COUNT(*) FROM prompt_studio_signatures{where_clause}"  # nosec B608
        try:
            count_cursor = self._execute(count_sql, params)
            total_row = count_cursor.fetchone()
            total = int(total_row[0]) if total_row and total_row[0] is not None else 0
        except BackendDatabaseError as exc:
            raise DatabaseError(f"Failed counting signatures for project {project_id}: {exc}") from exc  # noqa: TRY003

        offset = max(page - 1, 0) * per_page
        list_sql = """
            SELECT *
            FROM prompt_studio_signatures
            {where_clause}
            ORDER BY updated_at DESC, id DESC
            LIMIT ? OFFSET ?
        """.format_map(locals())  # nosec B608
        params_with_pagination = params + [per_page, offset]

        try:
            cursor = self._execute(list_sql, params_with_pagination)
            rows = cursor.fetchall()
            signatures = [self._row_to_dict(row) for row in rows if row]
        except BackendDatabaseError as exc:
            raise DatabaseError(f"Failed listing signatures for project {project_id}: {exc}") from exc  # noqa: TRY003

        if return_pagination:
            return {
                "signatures": signatures,
                "pagination": {
                    "page": page,
                    "per_page": per_page,
                    "total": total,
                    "total_pages": (total + per_page - 1) // per_page if per_page else 0,
                },
            }
        return signatures

    def update_signature(self, signature_id: int, updates: dict[str, Any]) -> dict[str, Any]:
        allowed_fields = {
            "name",
            "input_schema",
            "output_schema",
            "constraints",
            "validation_rules",
        }

        set_clauses: list[str] = []
        params: list[Any] = []

        for field, value in updates.items():
            if field not in allowed_fields:
                continue

            if field in {"input_schema", "output_schema", "constraints", "validation_rules"} and value is not None:
                params.append(json.dumps(value))
            else:
                params.append(value)
            set_clauses.append(f"{field} = ?")

        if not set_clauses:
            signature = self.get_signature(signature_id, include_deleted=True)
            if signature is None:
                raise InputError(f"Signature {signature_id} not found or already deleted")  # noqa: TRY003
            return signature

        set_clauses.append("updated_at = CURRENT_TIMESTAMP")
        params.append(signature_id)

        update_sql = (
            "UPDATE prompt_studio_signatures SET "  # nosec B608
            + ", ".join(set_clauses)
            + " WHERE id = ? AND deleted = FALSE RETURNING *"
        )

        try:
            with self._write_lock, self.transaction() as conn:
                cursor = self._cursor_exec(conn, update_sql, params)
                row = cursor.fetchone()
                if not row:
                    raise InputError(f"Signature {signature_id} not found or already deleted")  # noqa: TRY003
                signature = self._row_to_dict(row)
            self._log_sync_event(
                "prompt_studio_signature",
                signature.get("uuid", ""),
                "update",
                {key: updates[key] for key in updates if key in allowed_fields},
            )
            return signature or {}  # noqa: TRY300
        except BackendDatabaseError as exc:
            message = str(exc).lower()
            if "duplicate" in message and "prompt_studio_signatures" in message:
                raise ConflictError(  # noqa: TRY003
                    "Signature update conflicts with an existing record"
                ) from exc
            raise DatabaseError(f"Failed to update signature {signature_id}: {exc}") from exc  # noqa: TRY003

    def delete_signature(self, signature_id: int, hard_delete: bool = False) -> bool:
        try:
            with self._write_lock, self.transaction() as conn:
                if hard_delete:
                    cursor = self._cursor_exec(
                        conn,
                        "DELETE FROM prompt_studio_signatures WHERE id = ? RETURNING uuid",
                        (signature_id,),
                    )
                else:
                    cursor = self._cursor_exec(
                        conn,
                        """
                            UPDATE prompt_studio_signatures
                            SET deleted = TRUE, deleted_at = CURRENT_TIMESTAMP
                            WHERE id = ? AND deleted = FALSE
                            RETURNING uuid
                            """,
                        (signature_id,),
                    )
                row = cursor.fetchone()
                success = row is not None
            if success and row:
                self._log_sync_event(
                    "prompt_studio_signature",
                    row.get("uuid", ""),
                    "delete" if hard_delete else "soft_delete",
                    {"hard": hard_delete},
                )
            return success  # noqa: TRY300
        except BackendDatabaseError as exc:
            raise DatabaseError(f"Failed to delete signature {signature_id}: {exc}") from exc  # noqa: TRY003

    # --- Test run helpers ------------------------------------------------

    def create_test_run(
        self,
        *,
        project_id: int,
        prompt_id: int,
        test_case_id: int,
        model_name: str,
        model_params: Optional[dict[str, Any]] = None,
        inputs: Optional[dict[str, Any]] = None,
        outputs: Optional[dict[str, Any]] = None,
        expected_outputs: Optional[dict[str, Any]] = None,
        scores: Optional[dict[str, Any]] = None,
        execution_time_ms: Optional[int] = None,
        tokens_used: Optional[int] = None,
        cost_estimate: Optional[float] = None,
        error_message: Optional[str] = None,
        client_id: Optional[str] = None,
    ) -> dict[str, Any]:
        run_uuid = str(uuid.uuid4())
        payload = (
            run_uuid,
            project_id,
            prompt_id,
            test_case_id,
            model_name,
            json.dumps(model_params) if model_params is not None else None,
            json.dumps(inputs) if inputs is not None else None,
            json.dumps(outputs) if outputs is not None else None,
            json.dumps(expected_outputs) if expected_outputs is not None else None,
            json.dumps(scores) if scores is not None else None,
            execution_time_ms,
            tokens_used,
            cost_estimate,
            error_message,
            client_id or self.client_id,
        )

        insert_sql = """
            INSERT INTO prompt_studio_test_runs (
                uuid, project_id, prompt_id, test_case_id, model_name,
                model_params, inputs, outputs, expected_outputs, scores,
                execution_time_ms, tokens_used, cost_estimate, error_message,
                client_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            RETURNING *
        """

        try:
            with self._write_lock, self.transaction() as conn:
                cursor = self._cursor_exec(conn, insert_sql, payload)
                row = cursor.fetchone()
            return self._row_to_dict(cursor, row) if row else {}
        except BackendDatabaseError as exc:
            raise DatabaseError(f"Failed to create prompt studio test run: {exc}") from exc  # noqa: TRY003

    def get_test_cases_by_ids(
        self,
        test_case_ids: Iterable[int],
        *,
        include_deleted: bool = False,
    ) -> list[dict[str, Any]]:
        identifiers = list(dict.fromkeys(test_case_ids))
        if not identifiers:
            return []

        placeholders = ",".join(["?"] * len(identifiers))
        where_clause = f"id IN ({placeholders})"
        if not include_deleted:
            where_clause += " AND deleted = FALSE"

        query = f"SELECT * FROM prompt_studio_test_cases WHERE {where_clause}"  # nosec B608

        try:
            cursor = self._execute(query, identifiers)
            rows = cursor.fetchall()
            return [self._format_test_case(row) for row in rows if row]
        except BackendDatabaseError as exc:
            raise DatabaseError(f"Failed fetching test cases: {exc}") from exc  # noqa: TRY003

    # --- Evaluation helpers ---------------------------------------------

    def create_evaluation(
        self,
        *,
        prompt_id: int,
        project_id: int,
        model_configs: Optional[dict[str, Any]] = None,
        status: str = "running",
        test_case_ids: Optional[Iterable[int]] = None,
        client_id: Optional[str] = None,
    ) -> dict[str, Any]:
        evaluation_uuid = str(uuid.uuid4())
        payload = (
            evaluation_uuid,
            prompt_id,
            project_id,
            json.dumps(model_configs) if model_configs is not None else None,
            status,
            json.dumps(list(test_case_ids) if test_case_ids is not None else []),
            client_id or self.client_id,
        )

        insert_sql = """
            INSERT INTO prompt_studio_evaluations (
                uuid, prompt_id, project_id, model_configs, status,
                test_case_ids, started_at, client_id
            ) VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?)
            RETURNING *
        """

        try:
            with self._write_lock, self.transaction() as conn:
                cursor = self._cursor_exec(conn, insert_sql, payload)
                row = cursor.fetchone()
            return self._row_to_dict(cursor, row) if row else {}
        except BackendDatabaseError as exc:
            raise DatabaseError(f"Failed to create prompt studio evaluation: {exc}") from exc  # noqa: TRY003

    def update_evaluation(self, evaluation_id: int, updates: dict[str, Any]) -> dict[str, Any]:
        if not updates:
            evaluation = self.get_evaluation(evaluation_id)
            if evaluation is None:
                raise InputError(f"Evaluation {evaluation_id} not found")  # noqa: TRY003
            return evaluation

        json_fields = {"model_configs", "test_case_ids", "test_run_ids", "aggregate_metrics"}
        set_clauses: list[str] = []
        params: list[Any] = []

        for field, value in updates.items():
            if field in json_fields and value is not None:
                params.append(json.dumps(value))
                set_clauses.append(f"{field} = ?::jsonb")
            else:
                params.append(value)
                set_clauses.append(f"{field} = ?")

        set_clause_sql = ", ".join(set_clauses)
        params.append(evaluation_id)

        update_sql = (
            "UPDATE prompt_studio_evaluations SET "  # nosec B608
            + set_clause_sql
            + " WHERE id = ? RETURNING *"
        )

        try:
            with self._write_lock, self.transaction() as conn:
                cursor = self._cursor_exec(conn, update_sql, params)
                row = cursor.fetchone()
                if not row:
                    raise InputError(f"Evaluation {evaluation_id} not found")  # noqa: TRY003
            return self._row_to_dict(cursor, row) if row else {}
        except BackendDatabaseError as exc:
            raise DatabaseError(f"Failed to update evaluation {evaluation_id}: {exc}") from exc  # noqa: TRY003

    def get_evaluation(self, evaluation_id: int) -> Optional[dict[str, Any]]:
        try:
            cursor = self._execute(
                "SELECT * FROM prompt_studio_evaluations WHERE id = ?",
                [evaluation_id],
            )
            row = cursor.fetchone()
            return self._row_to_dict(cursor, row) if row else None
        except BackendDatabaseError as exc:
            raise DatabaseError(f"Failed to fetch evaluation {evaluation_id}: {exc}") from exc  # noqa: TRY003

    def list_evaluations(
        self,
        *,
        project_id: Optional[int] = None,
        prompt_id: Optional[int] = None,
        status: Optional[str] = None,
        page: int = 1,
        per_page: int = 20,
    ) -> dict[str, Any]:
        if page < 1:
            raise InputError("Page index must be >= 1")  # noqa: TRY003
        if per_page < 1:
            raise InputError("Items per page must be >= 1")  # noqa: TRY003

        conditions: list[str] = []
        params: list[Any] = []

        if project_id is not None:
            conditions.append("project_id = ?")
            params.append(project_id)
        if prompt_id is not None:
            conditions.append("prompt_id = ?")
            params.append(prompt_id)
        if status is not None:
            conditions.append("status = ?")
            params.append(status)

        where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""

        count_sql = f"SELECT COUNT(*) FROM prompt_studio_evaluations{where_clause}"  # nosec B608
        try:
            count_cursor = self._execute(count_sql, params)
            total_row = count_cursor.fetchone()
            total = int(total_row[0]) if total_row and total_row[0] is not None else 0
        except BackendDatabaseError as exc:
            raise DatabaseError(f"Failed counting evaluations: {exc}") from exc  # noqa: TRY003

        offset = max(page - 1, 0) * per_page
        list_sql = """
            SELECT *
            FROM prompt_studio_evaluations
            {where_clause}
            ORDER BY started_at DESC NULLS LAST, id DESC
            LIMIT ? OFFSET ?
        """.format_map(locals())  # nosec B608
        params_with_page = list(params) + [per_page, offset]

        try:
            cursor = self._execute(list_sql, params_with_page)
            rows = cursor.fetchall()
            evaluations = [self._row_to_dict(row) for row in rows if row]
        except BackendDatabaseError as exc:
            raise DatabaseError(f"Failed listing evaluations: {exc}") from exc  # noqa: TRY003

        return {
            "evaluations": evaluations,
            "pagination": {
                "page": page,
                "per_page": per_page,
                "total": total,
                "total_pages": (total + per_page - 1) // per_page if per_page else 0,
            },
        }

    # --- Optimization helpers -------------------------------------------

    def create_optimization(
        self,
        *,
        project_id: int,
        name: Optional[str],
        initial_prompt_id: Optional[int],
        optimizer_type: str,
        optimization_config: Optional[dict[str, Any]] = None,
        max_iterations: Optional[int] = None,
        bootstrap_samples: Optional[int] = None,
        status: str = "pending",
        client_id: Optional[str] = None,
    ) -> dict[str, Any]:
        optimization_uuid = str(uuid.uuid4())
        payload = (
            optimization_uuid,
            project_id,
            name,
            initial_prompt_id,
            None,  # optimized_prompt_id
            optimizer_type,
            json.dumps(optimization_config) if optimization_config is not None else None,
            None,
            None,
            None,
            0,
            max_iterations,
            bootstrap_samples,
            status,
            None,
            None,
            None,
            client_id or self.client_id,
        )

        insert_sql = """
            INSERT INTO prompt_studio_optimizations (
                uuid, project_id, name, initial_prompt_id, optimized_prompt_id,
                optimizer_type, optimization_config, initial_metrics, final_metrics,
                improvement_percentage, iterations_completed, max_iterations,
                bootstrap_samples, status, error_message, total_tokens, total_cost,
                client_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            RETURNING *
        """

        try:
            with self._write_lock, self.transaction() as conn:
                cursor = self._cursor_exec(conn, insert_sql, payload)
                row = cursor.fetchone()
            optimization = self._row_to_dict(cursor, row) if row else {}
        except BackendDatabaseError as exc:
            raise DatabaseError(f"Failed to create optimization: {exc}") from exc  # noqa: TRY003

        self._log_sync_event(
            "prompt_studio_optimization",
            optimization_uuid,
            "create",
            {
                "project_id": project_id,
                "optimizer_type": optimizer_type,
                "status": status,
            },
        )
        return optimization

    def get_optimization(
        self,
        optimization_id: int,
        *,
        include_deleted: bool = False,
    ) -> Optional[dict[str, Any]]:
        clauses = ["id = ?"]
        params: list[Any] = [optimization_id]
        if not include_deleted:
            clauses.append("deleted = FALSE")

        query = "SELECT * FROM prompt_studio_optimizations WHERE " + " AND ".join(clauses) + " LIMIT 1"  # nosec B608

        try:
            cursor = self._execute(query, params)
            row = cursor.fetchone()
            return self._row_to_dict(cursor, row) if row else None
        except BackendDatabaseError as exc:
            raise DatabaseError(f"Failed to fetch optimization {optimization_id}: {exc}") from exc  # noqa: TRY003

    def list_optimizations(
        self,
        *,
        project_id: Optional[int] = None,
        status: Optional[str] = None,
        include_deleted: bool = False,
        page: int = 1,
        per_page: int = 20,
    ) -> dict[str, Any]:
        if page < 1:
            raise InputError("Page index must be >= 1")  # noqa: TRY003
        if per_page < 1:
            raise InputError("Items per page must be >= 1")  # noqa: TRY003

        conditions: list[str] = []
        params: list[Any] = []

        if project_id is not None:
            conditions.append("project_id = ?")
            params.append(project_id)
        if status is not None:
            conditions.append("status = ?")
            params.append(status)
        if not include_deleted:
            conditions.append("deleted = FALSE")

        where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""

        count_sql = f"SELECT COUNT(*) FROM prompt_studio_optimizations{where_clause}"  # nosec B608
        try:
            count_cursor = self._execute(count_sql, params)
            total_row = count_cursor.fetchone()
            total = int(total_row[0]) if total_row and total_row[0] is not None else 0
        except BackendDatabaseError as exc:
            raise DatabaseError(f"Failed counting optimizations: {exc}") from exc  # noqa: TRY003

        offset = max(page - 1, 0) * per_page
        list_sql = """
            SELECT *
            FROM prompt_studio_optimizations
            {where_clause}
            ORDER BY created_at DESC, id DESC
            LIMIT ? OFFSET ?
        """.format_map(locals())  # nosec B608
        params_with_page = list(params) + [per_page, offset]

        try:
            cursor = self._execute(list_sql, params_with_page)
            rows = cursor.fetchall()
            optimizations = [self._row_to_dict(row) for row in rows if row]
        except BackendDatabaseError as exc:
            raise DatabaseError(f"Failed listing optimizations: {exc}") from exc  # noqa: TRY003

        return {
            "optimizations": optimizations,
            "pagination": {
                "page": page,
                "per_page": per_page,
                "total": total,
                "total_pages": (total + per_page - 1) // per_page if per_page else 0,
            },
        }

    def update_optimization(
        self,
        optimization_id: int,
        updates: dict[str, Any],
        *,
        set_started_at: bool = False,
        set_completed_at: bool = False,
    ) -> dict[str, Any]:
        json_fields = {
            "optimization_config",
            "initial_metrics",
            "final_metrics",
            "test_case_ids",
            "test_run_ids",
        }
        set_clauses: list[str] = []
        params: list[Any] = []

        for field, value in updates.items():
            if field in json_fields and value is not None:
                params.append(json.dumps(value))
                set_clauses.append(f"{field} = ?::jsonb")
            else:
                params.append(value)
                set_clauses.append(f"{field} = ?")

        if set_started_at:
            set_clauses.append("started_at = CURRENT_TIMESTAMP")
        if set_completed_at:
            set_clauses.append("completed_at = CURRENT_TIMESTAMP")

        if not set_clauses:
            optimization = self.get_optimization(optimization_id, include_deleted=True)
            if optimization is None:
                raise InputError(f"Optimization {optimization_id} not found")  # noqa: TRY003
            return optimization

        params.append(optimization_id)
        update_sql = (
            "UPDATE prompt_studio_optimizations SET "  # nosec B608
            + ", ".join(set_clauses)
            + " WHERE id = ? RETURNING *"
        )

        try:
            with self._write_lock, self.transaction() as conn:
                cursor = self._cursor_exec(conn, update_sql, params)
                row = cursor.fetchone()
                if not row:
                    raise InputError(f"Optimization {optimization_id} not found")  # noqa: TRY003
            optimization = self._row_to_dict(cursor, row) if row else {}
        except BackendDatabaseError as exc:
            raise DatabaseError(f"Failed to update optimization {optimization_id}: {exc}") from exc  # noqa: TRY003

        log_payload = {}
        for key, value in updates.items():
            if isinstance(value, (dict, list)):
                try:
                    log_payload[key] = json.loads(json.dumps(value, default=str))
                except TypeError:
                    log_payload[key] = str(value)
            else:
                log_payload[key] = value

        if set_started_at:
            log_payload["started_at"] = "CURRENT_TIMESTAMP"
        if set_completed_at:
            log_payload["completed_at"] = "CURRENT_TIMESTAMP"

        self._log_sync_event(
            "prompt_studio_optimization",
            optimization.get("uuid", ""),
            "update",
            log_payload,
        )
        return optimization

    def set_optimization_status(
        self,
        optimization_id: int,
        status: str,
        *,
        error_message: Optional[str] = None,
        mark_started: bool = False,
        mark_completed: bool = False,
    ) -> dict[str, Any]:
        updates: dict[str, Any] = {"status": status}
        if error_message is not None:
            updates["error_message"] = error_message
        return self.update_optimization(
            optimization_id,
            updates,
            set_started_at=mark_started,
            set_completed_at=mark_completed,
        )

    def complete_optimization(
        self,
        optimization_id: int,
        *,
        optimized_prompt_id: Optional[int] = None,
        iterations_completed: Optional[int] = None,
        initial_metrics: Optional[dict[str, Any]] = None,
        final_metrics: Optional[dict[str, Any]] = None,
        improvement_percentage: Optional[float] = None,
        total_tokens: Optional[int] = None,
        total_cost: Optional[float] = None,
    ) -> dict[str, Any]:
        updates: dict[str, Any] = {
            "status": "completed",
            "optimized_prompt_id": optimized_prompt_id,
            "iterations_completed": iterations_completed,
            "initial_metrics": initial_metrics,
            "final_metrics": final_metrics,
            "improvement_percentage": improvement_percentage,
            "total_tokens": total_tokens,
            "total_cost": total_cost,
        }
        # Remove keys with None to avoid overriding with NULL unnecessarily
        updates = {k: v for k, v in updates.items() if v is not None}
        return self.update_optimization(
            optimization_id,
            updates,
            set_completed_at=True,
        )

    def record_optimization_iteration(
        self,
        optimization_id: int,
        *,
        iteration_number: int,
        prompt_variant: Optional[dict[str, Any]] = None,
        metrics: Optional[dict[str, Any]] = None,
        tokens_used: Optional[int] = None,
        cost: Optional[float] = None,
        note: Optional[str] = None,
    ) -> dict[str, Any]:
        payload = (
            str(uuid.uuid4()),
            optimization_id,
            iteration_number,
            json.dumps(prompt_variant) if prompt_variant is not None else None,
            json.dumps(metrics) if metrics is not None else None,
            tokens_used,
            cost,
            note,
        )

        insert_sql = """
            INSERT INTO prompt_studio_optimization_iterations (
                uuid, optimization_id, iteration_number, prompt_variant, metrics,
                tokens_used, cost, note
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            RETURNING *
        """

        try:
            with self.transaction() as conn:
                cursor = self._cursor_exec(conn, insert_sql, payload)
                row = cursor.fetchone()
        except BackendDatabaseError as exc:
            raise DatabaseError(f"Failed to record optimization iteration: {exc}") from exc  # noqa: TRY003

        record = self._row_to_dict(cursor, row) if row else {}
        self._log_sync_event(
            "prompt_studio_optimization_iteration",
            record.get("uuid", ""),
            "create",
            {
                "optimization_id": optimization_id,
                "iteration_number": iteration_number,
            },
        )
        return record

    def list_optimization_iterations(  # noqa: F811
        self,
        optimization_id: int,
        *,
        page: int = 1,
        per_page: int = 50,
    ) -> dict[str, Any]:
        """List persisted iterations for an optimization (SQLite backend)."""
        if page < 1:
            raise InputError("Page index must be >= 1")  # noqa: TRY003
        if per_page < 1:
            raise InputError("Items per page must be >= 1")  # noqa: TRY003

        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            # Count total
            cursor.execute(
                "SELECT COUNT(*) FROM prompt_studio_optimization_iterations WHERE optimization_id = ?",
                (optimization_id,),
            )
            row = cursor.fetchone()
            total = int(row[0]) if row and row[0] is not None else 0

            # Page slice
            offset = max(page - 1, 0) * per_page
            cursor.execute(
                """
                SELECT *
                FROM prompt_studio_optimization_iterations
                WHERE optimization_id = ?
                ORDER BY iteration_number ASC, id ASC
                LIMIT ? OFFSET ?
                """,
                (optimization_id, per_page, offset),
            )
            rows = cursor.fetchall()
            iterations = [self._row_to_dict(cursor, r) for r in rows if r]

            return {
                "iterations": iterations,
                "pagination": {
                    "page": page,
                    "per_page": per_page,
                    "total": total,
                    "total_pages": (total + per_page - 1) // per_page if per_page else 0,
                },
            }
        except sqlite3.Error as exc:  # noqa: BLE001
            raise DatabaseError(f"Failed to list optimization iterations: {exc}") from exc  # noqa: TRY003

    def list_optimization_iterations(  # noqa: F811
        self,
        optimization_id: int,
        *,
        page: int = 1,
        per_page: int = 50,
    ) -> dict[str, Any]:
        if page < 1:
            raise InputError("Page index must be >= 1")  # noqa: TRY003
        if per_page < 1:
            raise InputError("Items per page must be >= 1")  # noqa: TRY003

        count_sql = "SELECT COUNT(*) FROM prompt_studio_optimization_iterations WHERE optimization_id = ?"

        try:
            count_cursor = self._execute(count_sql, [optimization_id])
            total_row = count_cursor.fetchone()
            total = int(total_row[0]) if total_row and total_row[0] is not None else 0
        except BackendDatabaseError as exc:
            raise DatabaseError(f"Failed counting optimization iterations: {exc}") from exc  # noqa: TRY003

        offset = max(page - 1, 0) * per_page
        list_sql = """
            SELECT *
            FROM prompt_studio_optimization_iterations
            WHERE optimization_id = ?
            ORDER BY iteration_number ASC, id ASC
            LIMIT ? OFFSET ?
        """

        try:
            cursor = self._execute(list_sql, [optimization_id, per_page, offset])
            rows = cursor.fetchall()
            iterations = [self._row_to_dict(row) for row in rows if row]
        except BackendDatabaseError as exc:
            raise DatabaseError(f"Failed listing optimization iterations: {exc}") from exc  # noqa: TRY003

        return {
            "iterations": iterations,
            "pagination": {
                "page": page,
                "per_page": per_page,
                "total": total,
                "total_pages": (total + per_page - 1) // per_page if per_page else 0,
            },
        }

    # --- Prompt helpers ---

    def get_prompt(self, prompt_id: int, include_deleted: bool = False) -> Optional[dict[str, Any]]:
        clauses = ["id = ?"]
        params: list[Any] = [prompt_id]
        if not include_deleted:
            clauses.append("deleted = FALSE")

        query = (
            "SELECT * FROM prompt_studio_prompts WHERE " + " AND ".join(clauses) + " LIMIT 1"  # nosec B608
        )

        try:
            cursor = self._execute(query, params)
            row = cursor.fetchone()
            return self._row_to_dict(cursor, row)
        except BackendDatabaseError as exc:
            raise DatabaseError(f"Failed to fetch prompt {prompt_id}: {exc}") from exc  # noqa: TRY003

    def list_prompts(
        self,
        project_id: int,
        *,
        page: int = 1,
        per_page: int = 20,
        include_deleted: bool = False,
    ) -> dict[str, Any]:
        if page < 1:
            raise InputError("Page index must be >= 1")  # noqa: TRY003
        if per_page < 1:
            raise InputError("Items per page must be >= 1")  # noqa: TRY003

        base_conditions = ["project_id = ?"]
        params: list[Any] = [project_id]
        if not include_deleted:
            base_conditions.append("deleted = FALSE")

        where_clause = " WHERE " + " AND ".join(base_conditions)

        count_sql = f"SELECT COUNT(*) FROM prompt_studio_prompts{where_clause}"  # nosec B608
        try:
            count_cursor = self._execute(count_sql, params)
            total_row = count_cursor.fetchone()
            total = int(total_row[0]) if total_row and total_row[0] is not None else 0

            offset = (page - 1) * per_page
            list_sql = """
                SELECT *
                FROM prompt_studio_prompts
                {where_clause}
                ORDER BY updated_at DESC, version_number DESC
                LIMIT ? OFFSET ?
            """.format_map(locals())  # nosec B608
            list_params = list(params) + [per_page, offset]
            list_cursor = self._execute(list_sql, list_params)
            rows = list_cursor.fetchall()
            prompts = [self._row_to_dict(list_cursor, row) for row in rows if row]

            return {
                "prompts": prompts,
                "pagination": {
                    "page": page,
                    "per_page": per_page,
                    "total": total,
                    "total_pages": (total + per_page - 1) // per_page if per_page else 0,
                },
            }
        except BackendDatabaseError as exc:
            raise DatabaseError(  # noqa: TRY003
                f"Failed to list prompts for project {project_id}: {exc}"
            ) from exc

    def list_prompt_versions(
        self,
        project_id: int,
        prompt_name: str,
        *,
        include_deleted: bool = False,
    ) -> list[dict[str, Any]]:
        conditions = ["project_id = ?", "name = ?"]
        params: list[Any] = [project_id, prompt_name]
        if not include_deleted:
            conditions.append("deleted = FALSE")

        query_template = """
            SELECT id, uuid, version_number, name, change_description,
                   created_at, parent_version_id
            FROM prompt_studio_prompts
            WHERE {where}
            ORDER BY version_number DESC
        """
        where_sql = " AND ".join(conditions)
        query = query_template.format(where=where_sql)  # nosec B608

        try:
            cursor = self._execute(query, params)
            rows = cursor.fetchall()
            return [self._row_to_dict(cursor, row) for row in rows if row]
        except BackendDatabaseError as exc:
            raise DatabaseError(  # noqa: TRY003
                f"Failed to list versions for prompt '{prompt_name}' in project {project_id}: {exc}"
            ) from exc

    def ensure_prompt_stub(
        self,
        *,
        prompt_id: int,
        project_id: int,
        name: Optional[str] = None,
        client_id: Optional[str] = None,
    ) -> None:
        """Ensure a placeholder prompt exists for the given identifiers."""

        if not prompt_id or not project_id:
            return

        try:
            cursor = self._execute(
                "SELECT 1 FROM prompt_studio_prompts WHERE id = ?",
                [prompt_id],
            )
            if cursor.fetchone() is not None:
                return
        except BackendDatabaseError as exc:
            raise DatabaseError(  # noqa: TRY003
                f"Failed to verify prompt {prompt_id} existence: {exc}"
            ) from exc

        stub_name = name or f"Auto-Created Prompt {prompt_id}"
        params = (
            prompt_id,
            project_id,
            stub_name,
            client_id or self.client_id,
        )

        insert_sql = """
            INSERT OR IGNORE INTO prompt_studio_prompts (
                id, uuid, project_id, version_number, name, client_id
            ) VALUES (?, lower(hex(randomblob(16))), ?, 1, ?, ?)
        """

        try:
            with self._write_lock, self.transaction() as conn:
                _ = self._cursor_exec(conn, insert_sql, params)
        except BackendDatabaseError as exc:
            raise DatabaseError(  # noqa: TRY003
                f"Failed to create placeholder prompt {prompt_id}: {exc}"
            ) from exc

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

    def get_prompt_with_project(
        self,
        prompt_id: int,
        *,
        include_deleted: bool = False,
    ) -> Optional[dict[str, Any]]:
        clauses = ["p.id = ?"]
        if not include_deleted:
            clauses.append("p.deleted = FALSE")
        where_sql = ' AND '.join(clauses)
        query = """
            SELECT p.*, proj.user_id AS project_user_id
            FROM prompt_studio_prompts p
            JOIN prompt_studio_projects proj ON p.project_id = proj.id
            WHERE {where_sql}
            LIMIT 1
        """.format_map(locals())  # nosec B608
        try:
            cursor = self._execute(query, [prompt_id])
            row = cursor.fetchone()
            return self._row_to_dict(cursor, row) if row else None
        except BackendDatabaseError as exc:
            raise DatabaseError(f"Failed to fetch prompt {prompt_id}: {exc}") from exc  # noqa: TRY003

    def create_prompt_version(
        self,
        prompt_id: int,
        *,
        change_description: str,
        name: Optional[str] = None,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        prompt_format: Optional[str] = None,
        prompt_schema_version: Optional[int] = None,
        prompt_definition: Optional[Any] = None,
        few_shot_examples: Optional[Any] = None,
        modules_config: Optional[Any] = None,
        client_id: Optional[str] = None,
    ) -> dict[str, Any]:
        if not change_description:
            raise InputError("change_description is required")  # noqa: TRY003

        with self._write_lock, self.transaction() as conn:
            cursor = self._cursor_exec(
                conn,
                """
                    SELECT *
                    FROM prompt_studio_prompts
                    WHERE id = ? AND deleted = FALSE
                    LIMIT 1
                    """,
                (prompt_id,),
            )
            current_row = cursor.fetchone()
            if not current_row:
                raise InputError(f"Prompt {prompt_id} not found or already deleted")  # noqa: TRY003
            current_prompt = self._row_to_dict(cursor, current_row) or {}

            new_uuid = str(uuid.uuid4())
            new_version = int(current_prompt.get("version_number", 0)) + 1

            next_name = name if name is not None else current_prompt.get("name")
            normalized_prompt_fields = _prepare_prompt_record_fields(
                prompt_format=prompt_format,
                prompt_schema_version=prompt_schema_version,
                prompt_definition=prompt_definition,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                current_prompt=current_prompt,
            )
            next_examples = (
                few_shot_examples
                if few_shot_examples is not None
                else current_prompt.get("few_shot_examples")
            )
            next_modules = (
                modules_config
                if modules_config is not None
                else current_prompt.get("modules_config")
            )

            insert_sql = """
                    INSERT INTO prompt_studio_prompts (
                        uuid,
                        project_id,
                        signature_id,
                        version_number,
                        name,
                        system_prompt,
                        user_prompt,
                        prompt_format,
                        prompt_schema_version,
                        prompt_definition,
                        few_shot_examples,
                        modules_config,
                        parent_version_id,
                        change_description,
                        client_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    RETURNING *
                """

            payload = (
                new_uuid,
                current_prompt.get("project_id"),
                current_prompt.get("signature_id"),
                new_version,
                next_name,
                normalized_prompt_fields["system_prompt"],
                normalized_prompt_fields["user_prompt"],
                normalized_prompt_fields["prompt_format"],
                normalized_prompt_fields["prompt_schema_version"],
                json.dumps(normalized_prompt_fields["prompt_definition"])
                if normalized_prompt_fields["prompt_definition"] is not None
                else None,
                json.dumps(next_examples) if next_examples is not None else None,
                json.dumps(next_modules) if next_modules is not None else None,
                prompt_id,
                change_description,
                client_id or current_prompt.get("client_id") or self.client_id,
            )

            cursor.execute(insert_sql, payload)
            new_row = cursor.fetchone()
            prompt = self._row_to_dict(cursor, new_row)

        self._log_sync_event(
            "prompt_studio_prompt",
            prompt.get("uuid", ""),
            "version_create",
            {
                "prompt_id": prompt_id,
                "new_version": prompt.get("version_number"),
                "change_description": change_description,
            },
        )
        return prompt

    def revert_prompt_to_version(
        self,
        prompt_id: int,
        target_version: int,
        *,
        client_id: Optional[str] = None,
    ) -> dict[str, Any]:
        if target_version < 1:
            raise InputError("target_version must be >= 1")  # noqa: TRY003

        with self._write_lock, self.transaction() as conn:
            cursor = self._cursor_exec(
                conn,
                """
                    SELECT *
                    FROM prompt_studio_prompts
                    WHERE id = ? AND deleted = FALSE
                    LIMIT 1
                    """,
                (prompt_id,),
            )
            current_row = cursor.fetchone()
            if not current_row:
                raise InputError(f"Prompt {prompt_id} not found or already deleted")  # noqa: TRY003
            current_prompt = self._row_to_dict(cursor, current_row) or {}

            cursor = self._cursor_exec(
                conn,
                """
                    SELECT *
                    FROM prompt_studio_prompts
                    WHERE project_id = ? AND name = ? AND version_number = ? AND deleted = FALSE
                    LIMIT 1
                    """,
                (
                    current_prompt.get("project_id"),
                    current_prompt.get("name"),
                    target_version,
                ),
            )
            target_row = cursor.fetchone()
            if not target_row:
                raise InputError(  # noqa: TRY003
                    f"Version {target_version} not found for prompt {current_prompt.get('name')}"
                )
            target_prompt = self._row_to_dict(cursor, target_row) or {}

            cursor = self._cursor_exec(
                conn,
                """
                    SELECT COALESCE(MAX(version_number), 0)
                    FROM prompt_studio_prompts
                    WHERE project_id = ? AND name = ?
                    """,
                (current_prompt.get("project_id"), current_prompt.get("name")),
            )
            max_version_row = cursor.fetchone()
            next_version = int(max_version_row[0]) + 1 if max_version_row else 1

            new_uuid = str(uuid.uuid4())
            normalized_prompt_fields = _prepare_prompt_record_fields(
                prompt_format=target_prompt.get("prompt_format"),
                prompt_schema_version=target_prompt.get("prompt_schema_version"),
                prompt_definition=target_prompt.get("prompt_definition"),
                system_prompt=target_prompt.get("system_prompt"),
                user_prompt=target_prompt.get("user_prompt"),
            )
            insert_sql = """
                    INSERT INTO prompt_studio_prompts (
                        uuid,
                        project_id,
                        signature_id,
                        version_number,
                        name,
                        system_prompt,
                        user_prompt,
                        prompt_format,
                        prompt_schema_version,
                        prompt_definition,
                        few_shot_examples,
                        modules_config,
                        parent_version_id,
                        change_description,
                        client_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    RETURNING *
                """

            payload = (
                new_uuid,
                target_prompt.get("project_id"),
                target_prompt.get("signature_id"),
                next_version,
                target_prompt.get("name"),
                normalized_prompt_fields["system_prompt"],
                normalized_prompt_fields["user_prompt"],
                normalized_prompt_fields["prompt_format"],
                normalized_prompt_fields["prompt_schema_version"],
                json.dumps(normalized_prompt_fields["prompt_definition"])
                if normalized_prompt_fields["prompt_definition"] is not None
                else None,
                json.dumps(target_prompt.get("few_shot_examples"))
                if target_prompt.get("few_shot_examples") is not None
                else None,
                json.dumps(target_prompt.get("modules_config"))
                if target_prompt.get("modules_config") is not None
                else None,
                prompt_id,
                f"Reverted to version {target_version}",
                client_id or current_prompt.get("client_id") or self.client_id,
            )

            cursor.execute(insert_sql, payload)
            new_row = cursor.fetchone()
            prompt = self._row_to_dict(cursor, new_row)

        self._log_sync_event(
            "prompt_studio_prompt",
            prompt.get("uuid", ""),
            "version_revert",
            {
                "prompt_id": prompt_id,
                "target_version": target_version,
                "new_version": prompt.get("version_number"),
            },
        )
        return prompt

    def get_golden_test_cases(
        self,
        project_id: int,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        query = (
            """
            SELECT id, uuid, project_id, signature_id, name, description,
                   inputs, expected_outputs, actual_outputs, tags,
                   is_golden, is_generated, client_id, deleted,
                   created_at, updated_at
            FROM prompt_studio_test_cases
            WHERE project_id = ? AND is_golden = TRUE
        """
        )
        params: list[Any] = [project_id]
        query += " AND deleted = FALSE"  # Always exclude deleted in helper
        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        try:
            cursor = self._execute(query, params)
            rows = cursor.fetchall()
            return [self._format_test_case(row) for row in rows if row]
        except BackendDatabaseError as exc:
            raise DatabaseError(f"Failed to fetch golden test cases for project {project_id}: {exc}") from exc  # noqa: TRY003

    # --- Test case helpers -------------------------------------------------

    def _format_test_case(self, row: Any) -> Optional[dict[str, Any]]:
        return _format_test_case_record(self._row_to_dict(row))

    def _build_test_case_filters(
        self,
        project_id: int,
        *,
        signature_id: Optional[int] = None,
        is_golden: Optional[bool] = None,
        tags: Optional[list[str]] = None,
        include_deleted: bool = False,
    ) -> tuple[str, list[Any]]:
        conditions: list[str] = ["project_id = ?"]
        params: list[Any] = [project_id]

        if not include_deleted:
            conditions.append("deleted = FALSE" if self.backend_type == BackendType.POSTGRESQL else "deleted = 0")

        if signature_id is not None:
            conditions.append("signature_id = ?")
            params.append(signature_id)

        if is_golden is not None:
            conditions.append("is_golden = ?")
            params.append(bool(is_golden) if self.backend_type == BackendType.POSTGRESQL else int(bool(is_golden)))

        if tags:
            tag_conditions = []
            for tag in tags:
                tag_conditions.append("tags LIKE ?")
                params.append(f"%{tag}%")
            if tag_conditions:
                conditions.append(f"({' OR '.join(tag_conditions)})")

        where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""
        return where_clause, params

    def create_test_case(
        self,
        project_id: int,
        name: str,
        *,
        inputs: dict[str, Any],
        description: Optional[str] = None,
        expected_outputs: Optional[dict[str, Any]] = None,
        actual_outputs: Optional[dict[str, Any]] = None,
        tags: Optional[Iterable[str]] = None,
        is_golden: bool = False,
        is_generated: bool = False,
        signature_id: Optional[int] = None,
        client_id: Optional[str] = None,
    ) -> dict[str, Any]:
        if not name or not name.strip():
            raise InputError("Test case name cannot be empty")  # noqa: TRY003

        test_case_uuid = str(uuid.uuid4())
        payload = (
            test_case_uuid,
            project_id,
            signature_id,
            name.strip(),
            description,
            json.dumps(inputs),
            json.dumps(expected_outputs) if expected_outputs is not None else None,
            json.dumps(actual_outputs) if actual_outputs is not None else None,
            _serialise_tags(tags),
            bool(is_golden) if self.backend_type == BackendType.POSTGRESQL else int(bool(is_golden)),
            bool(is_generated) if self.backend_type == BackendType.POSTGRESQL else int(bool(is_generated)),
            client_id or self.client_id,
        )

        insert_sql = """
            INSERT INTO prompt_studio_test_cases (
                uuid, project_id, signature_id, name, description,
                inputs, expected_outputs, actual_outputs, tags,
                is_golden, is_generated, client_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            RETURNING *
        """

        try:
            with self._write_lock, self.transaction() as conn:
                cursor = self._cursor_exec(conn, insert_sql, payload)
                row = cursor.fetchone()
                test_case = self._format_test_case(row)
            return test_case or {}  # noqa: TRY300
        except BackendDatabaseError as exc:
            message = str(exc).lower()
            if "unique" in message and "prompt_studio_test_cases" in message and "name" in message:
                raise ConflictError(f"Test case with name '{name}' already exists") from exc  # noqa: TRY003
            raise DatabaseError(f"Failed to create prompt studio test case: {exc}") from exc  # noqa: TRY003

    def get_test_case(
        self,
        test_case_id: int,
        *,
        include_deleted: bool = False,
    ) -> Optional[dict[str, Any]]:
        where_clauses = ["id = ?"]
        params: list[Any] = [test_case_id]
        if not include_deleted:
            where_clauses.append("deleted = FALSE" if self.backend_type == BackendType.POSTGRESQL else "deleted = 0")

        query = (
            "SELECT * FROM prompt_studio_test_cases WHERE "  # nosec B608
            + " AND ".join(where_clauses)
            + " LIMIT 1"
        )

        try:
            cursor = self._execute(query, params)
            row = cursor.fetchone()
            return self._format_test_case(row)
        except BackendDatabaseError as exc:
            raise DatabaseError(f"Failed to fetch test case {test_case_id}: {exc}") from exc  # noqa: TRY003

    def list_test_cases(
        self,
        project_id: int,
        *,
        signature_id: Optional[int] = None,
        is_golden: Optional[bool] = None,
        tags: Optional[list[str]] = None,
        search: Optional[str] = None,
        include_deleted: bool = False,
        page: int = 1,
        per_page: int = 20,
        return_pagination: bool = False,
    ) -> Union[dict[str, Any], list[dict[str, Any]]]:
        where_clause, params = self._build_test_case_filters(
            project_id,
            signature_id=signature_id,
            is_golden=is_golden,
            tags=tags,
            include_deleted=include_deleted,
        )

        if search:
            comparator = "ILIKE" if self.backend_type == BackendType.POSTGRESQL else "LIKE"
            where_clause += f" AND (name {comparator} ? OR description {comparator} ?)"
            params.extend([f"%{search}%", f"%{search}%"])

        count_sql = f"SELECT COUNT(*) FROM prompt_studio_test_cases{where_clause}"  # nosec B608
        try:
            count_cursor = self._execute(count_sql, params)
            count_row = count_cursor.fetchone()
            total = int(count_row[0]) if count_row else 0
        except BackendDatabaseError as exc:
            raise DatabaseError(f"Failed to count test cases for project {project_id}: {exc}") from exc  # noqa: TRY003

        offset = max(page - 1, 0) * per_page
        list_sql = """
            SELECT *
            FROM prompt_studio_test_cases
            {where_clause}
            ORDER BY is_golden DESC, created_at DESC
            LIMIT ? OFFSET ?
        """.format_map(locals())  # nosec B608
        params_with_pagination = params + [per_page, offset]

        try:
            cursor = self._execute(list_sql, params_with_pagination)
            rows = cursor.fetchall()
            records = [self._format_test_case(row) for row in rows if row]
        except BackendDatabaseError as exc:
            raise DatabaseError(f"Failed to list test cases for project {project_id}: {exc}") from exc  # noqa: TRY003

        if return_pagination:
            return {
                "test_cases": records,
                "pagination": {
                    "page": page,
                    "per_page": per_page,
                    "total": total,
                    "total_pages": (total + per_page - 1) // per_page if per_page else 0,
                },
            }
        return records

    def update_test_case(self, test_case_id: int, updates: dict[str, Any]) -> dict[str, Any]:
        allowed_fields = {
            "name",
            "description",
            "inputs",
            "expected_outputs",
            "actual_outputs",
            "tags",
            "is_golden",
            "is_generated",
            "signature_id",
        }
        set_clauses: list[str] = []
        params: list[Any] = []

        for field, value in updates.items():
            if field not in allowed_fields:
                continue

            if field in {"inputs", "expected_outputs", "actual_outputs"} and value is not None:
                params.append(json.dumps(value))
            elif field in {"is_golden", "is_generated"} and value is not None:
                params.append(int(bool(value)))
            elif field == "tags":
                params.append(_serialise_tags(value))
            else:
                params.append(value)
            set_clauses.append(f"{field} = ?")

        if not set_clauses:
            existing = self.get_test_case(test_case_id)
            if existing is None:
                raise InputError(f"Test case {test_case_id} not found or already deleted")  # noqa: TRY003
            return existing

        set_clauses.append("updated_at = CURRENT_TIMESTAMP")
        params.append(test_case_id)

        deleted_clause = "deleted = FALSE" if self.backend_type == BackendType.POSTGRESQL else "deleted = 0"
        set_clause_sql = ', '.join(set_clauses)
        update_sql = """
            UPDATE prompt_studio_test_cases
            SET {set_clause_sql}
            WHERE id = ? AND {deleted_clause}
            RETURNING *
        """.format_map(locals())  # nosec B608

        try:
            with self.transaction() as conn:
                cursor = self._cursor_exec(conn, update_sql, params)
                row = cursor.fetchone()
                if not row:
                    raise InputError(f"Test case {test_case_id} not found or already deleted")  # noqa: TRY003
                return self._format_test_case(row) or {}
        except BackendDatabaseError as exc:
            raise DatabaseError(f"Failed to update test case {test_case_id}: {exc}") from exc  # noqa: TRY003

    def delete_test_case(self, test_case_id: int, *, hard_delete: bool = False) -> bool:
        try:
            with self.transaction() as conn:
                deleted_clause = "deleted = FALSE" if self.backend_type == BackendType.POSTGRESQL else "deleted = 0"
                deleted_value = "TRUE" if self.backend_type == BackendType.POSTGRESQL else "1"
                if hard_delete:
                    cursor = self._cursor_exec(
                        conn,
                        "DELETE FROM prompt_studio_test_cases WHERE id = ? RETURNING id",
                        (test_case_id,),
                    )
                else:
                    cursor = self._cursor_exec(
                        conn,
                        """
                        UPDATE prompt_studio_test_cases
                        SET deleted = {deleted_value},
                            deleted_at = CURRENT_TIMESTAMP
                        WHERE id = ? AND {deleted_clause}
                        RETURNING id
                        """.format_map(locals()),  # nosec B608
                        (test_case_id,),
                    )
                row = cursor.fetchone()
                return row is not None
        except BackendDatabaseError as exc:
            raise DatabaseError(f"Failed to delete test case {test_case_id}: {exc}") from exc  # noqa: TRY003

    def create_bulk_test_cases(
        self,
        project_id: int,
        test_cases: list[dict[str, Any]],
        *,
        signature_id: Optional[int] = None,
        client_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        created: list[dict[str, Any]] = []
        for test_case in test_cases:
            created_case = self.create_test_case(
                project_id,
                test_case.get("name", ""),
                inputs=test_case.get("inputs", {}),
                description=test_case.get("description"),
                expected_outputs=test_case.get("expected_outputs"),
                actual_outputs=test_case.get("actual_outputs"),
                tags=test_case.get("tags"),
                is_golden=test_case.get("is_golden", False),
                is_generated=test_case.get("is_generated", False),
                signature_id=signature_id or test_case.get("signature_id"),
                client_id=client_id or test_case.get("client_id"),
            )
            created.append(created_case)
        return created

    def search_test_cases(
        self,
        project_id: int,
        query: str,
        *,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        backend_query = query
        if self.backend_type == BackendType.POSTGRESQL:
            backend_query = FTSQueryTranslator.normalize_query(query, "postgresql") or query
            fts_column = self.get_fts_column("prompt_studio_test_cases") or "prompt_studio_test_cases_tsv"
            search_sql = """
                SELECT tc.*, ts_rank({fts_column}, to_tsquery('english', ?)) AS rank
                FROM prompt_studio_test_cases tc
                WHERE tc.project_id = ?
                  AND tc.deleted = FALSE
                  AND {fts_column} @@ to_tsquery('english', ?)
                ORDER BY rank DESC
                LIMIT ?
            """.format_map(locals())  # nosec B608
            params = [backend_query, project_id, backend_query, limit]
        else:
            search_sql = """
                SELECT tc.*
                FROM prompt_studio_test_cases tc
                JOIN prompt_studio_test_cases_fts ON tc.id = prompt_studio_test_cases_fts.rowid
                WHERE tc.project_id = ?
                  AND tc.deleted = 0
                  AND prompt_studio_test_cases_fts MATCH ?
                ORDER BY bm25(prompt_studio_test_cases_fts)
                LIMIT ?
            """
            params = [project_id, backend_query, limit]

        try:
            cursor = self._execute(search_sql, params)
            rows = cursor.fetchall()
            return [self._format_test_case(row) for row in rows if row]
        except BackendDatabaseError as exc:
            raise DatabaseError(f"Failed to search test cases in project {project_id}: {exc}") from exc  # noqa: TRY003

    def get_test_cases_by_signature(self, signature_id: int) -> list[dict[str, Any]]:
        query = """
            SELECT *
            FROM prompt_studio_test_cases
            WHERE signature_id = ? AND deleted = 0
            ORDER BY is_golden DESC, created_at DESC
        """
        try:
            cursor = self._execute(query, [signature_id])
            rows = cursor.fetchall()
            return [self._format_test_case(row) for row in rows if row]
        except BackendDatabaseError as exc:
            raise DatabaseError(f"Failed to fetch test cases for signature {signature_id}: {exc}") from exc  # noqa: TRY003

    def get_test_case_stats(self, project_id: int) -> dict[str, Any]:
        stats: dict[str, Any] = {}
        try:
            total_cursor = self._execute(
                "SELECT COUNT(*) FROM prompt_studio_test_cases WHERE project_id = ? AND deleted = 0",
                (project_id,),
            )
            total_row = total_cursor.fetchone()
            stats["total"] = total_row[0] if total_row else 0

            golden_cursor = self._execute(
                "SELECT COUNT(*) FROM prompt_studio_test_cases WHERE project_id = ? AND deleted = 0 AND is_golden = 1",
                (project_id,),
            )
            golden_row = golden_cursor.fetchone()
            stats["golden"] = golden_row[0] if golden_row else 0

            generated_cursor = self._execute(
                "SELECT COUNT(*) FROM prompt_studio_test_cases WHERE project_id = ? AND deleted = 0 AND is_generated = 1",
                (project_id,),
            )
            generated_row = generated_cursor.fetchone()
            stats["generated"] = generated_row[0] if generated_row else 0

            expected_cursor = self._execute(
                "SELECT COUNT(*) FROM prompt_studio_test_cases WHERE project_id = ? AND deleted = 0 AND expected_outputs IS NOT NULL",
                (project_id,),
            )
            expected_row = expected_cursor.fetchone()
            stats["with_expected"] = expected_row[0] if expected_row else 0

            signature_cursor = self._execute(
                """
                SELECT signature_id, COUNT(*)
                FROM prompt_studio_test_cases
                WHERE project_id = ? AND deleted = 0 AND signature_id IS NOT NULL
                GROUP BY signature_id
                """,
                (project_id,),
            )
            stats["by_signature"] = {
                row[0]: row[1]
                for row in signature_cursor.fetchall()
                if row and row[0] is not None
            }

            tags_cursor = self._execute(
                """
                SELECT tags
                FROM prompt_studio_test_cases
                WHERE project_id = ? AND deleted = 0 AND tags IS NOT NULL
                """,
                (project_id,),
            )
            tag_counts: dict[str, int] = {}
            for row in tags_cursor.fetchall():
                for tag in _parse_tags(row[0]):
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1

            stats["top_tags"] = sorted(tag_counts.items(), key=lambda item: item[1], reverse=True)[:10]
            return stats  # noqa: TRY300
        except BackendDatabaseError as exc:
            raise DatabaseError(f"Failed to compute test case stats for project {project_id}: {exc}") from exc  # noqa: TRY003

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

    def create_project(self, name: str, description: Optional[str] = None,
                      status: str = "draft", metadata: Optional[dict] = None,
                      user_id: Optional[str] = None) -> dict[str, Any]:
        """
        Create a new prompt studio project.

        Args:
            name: Project name
            description: Project description
            status: Project status (draft, active, archived)
            metadata: Additional metadata

        Returns:
            Created project record
        """
        import random
        import sqlite3
        import time

        project_id = None
        # Get connection before acquiring lock to avoid deadlock
        conn = self.get_connection()

        max_retries = 5
        base_delay = 0.1  # 100ms

        for attempt in range(max_retries):
            should_retry = False
            retry_delay = 0

            # Use write lock to serialize write operations
            with self._write_lock:
                try:
                    cursor = conn.cursor()

                    # Generate UUID
                    project_uuid = str(uuid.uuid4())

                    # Insert project
                    cursor.execute("""
                        INSERT INTO prompt_studio_projects
                        (uuid, name, description, user_id, client_id, status, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (project_uuid, name, description, user_id or self.client_id, self.client_id,
                          status, json.dumps(metadata) if metadata else None))

                    project_id = cursor.lastrowid
                    conn.commit()

                    # Log to sync_log
                    self._log_sync_event("prompt_studio_project", project_uuid, "create", {
                        "name": name,
                        "description": description,
                        "status": status
                    })

                    logger.info(f"Created project: {name} (ID: {project_id})")
                    break  # Success, exit retry loop

                except sqlite3.OperationalError as e:
                    if "database is locked" in str(e) and attempt < max_retries - 1:
                        # Database locked, will retry
                        should_retry = True
                        retry_delay = base_delay * (2 ** attempt) * (0.5 + random.random())
                    else:
                        raise DatabaseError(f"Failed to create project: {e}")  # noqa: B904, TRY003
                except sqlite3.IntegrityError as e:
                    if "UNIQUE" in str(e):
                        raise ConflictError(f"Project with name '{name}' already exists for this user")  # noqa: B904, TRY003
                    raise DatabaseError(f"Failed to create project: {e}")  # noqa: B904, TRY003
                except _PROMPT_STUDIO_NONCRITICAL_EXCEPTIONS as e:
                    raise DatabaseError(f"Failed to create project: {e}")  # noqa: B904, TRY003

            # Sleep outside the lock if we need to retry
            if should_retry:
                time.sleep(retry_delay)

        # Get the project after releasing the lock
        return self.get_project(project_id)

    def get_project(self, project_id: int, include_deleted: bool = False) -> Optional[dict[str, Any]]:
        """
        Get a project by ID.

        Args:
            project_id: Project ID
            include_deleted: Include soft-deleted projects

        Returns:
            Project record or None
        """
        import random
        import sqlite3
        import time
        conn = self.get_connection()
        cursor = conn.cursor()
        query = """
            SELECT
                id, uuid, name, description, user_id, client_id, status,
                deleted, deleted_at, created_at, updated_at, last_modified,
                version, metadata
            FROM prompt_studio_projects
            WHERE id = ?
        """
        if not include_deleted:
            query += " AND deleted = 0"

        max_retries = 5
        base_delay = 0.05
        for attempt in range(max_retries):
            try:
                cursor.execute(query, (project_id,))
                row = cursor.fetchone()
                if row:
                    return self._row_to_dict(cursor, row)
                return None  # noqa: TRY300
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e).lower() and attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) * (0.5 + random.random())
                    time.sleep(delay)
                    continue
                raise DatabaseError(f"Failed to get project: {e}")  # noqa: B904, TRY003
            except sqlite3.Error as e:
                raise DatabaseError(f"Failed to get project: {e}")  # noqa: B904, TRY003

    def list_projects(self, user_id: Optional[str] = None, status: Optional[str] = None,
                     include_deleted: bool = False, page: int = 1, per_page: int = 20,
                     search: Optional[str] = None) -> dict[str, Any]:
        """
        List projects with optional filtering.

        Args:
            user_id: Filter by user ID
            status: Filter by status
            include_deleted: Include soft-deleted projects
            page: Page number
            per_page: Items per page

        Returns:
            Dictionary with projects list and pagination metadata
        """
        import random
        import sqlite3
        import time
        conn = self.get_connection()
        cursor = conn.cursor()

        # Build query
        conditions = []
        params = []
        if not include_deleted:
            conditions.append("deleted = 0")
        if user_id:
            conditions.append("user_id = ?")
            params.append(user_id)
        if status:
            conditions.append("status = ?")
            params.append(status)

        if search:
            conditions.append("(name LIKE ? OR description LIKE ?)")
            like = f"%{search}%"
            params.extend([like, like])
        where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""

        # Count total with retry
        count_query = f"SELECT COUNT(*) FROM prompt_studio_projects{where_clause}"  # nosec B608
        max_retries = 5
        base_delay = 0.05
        for attempt in range(max_retries):
            try:
                cursor.execute(count_query, params)
                total = cursor.fetchone()[0]
                break
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e).lower() and attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) * (0.5 + random.random())
                    time.sleep(delay)
                    continue
                raise DatabaseError(f"Failed to list projects: {e}")  # noqa: B904, TRY003
            except sqlite3.Error as e:
                raise DatabaseError(f"Failed to list projects: {e}")  # noqa: B904, TRY003

        # Get projects with pagination (retry)
        offset = (page - 1) * per_page
        query = """
            SELECT
                p.*,
                (SELECT COUNT(*) FROM prompt_studio_prompts WHERE project_id = p.id AND deleted = 0) as prompt_count,
                (SELECT COUNT(*) FROM prompt_studio_test_cases WHERE project_id = p.id AND deleted = 0) as test_case_count
            FROM prompt_studio_projects p
            {where_clause}
            ORDER BY p.updated_at DESC
            LIMIT ? OFFSET ?
        """.format_map(locals())  # nosec B608
        params_page = list(params) + [per_page, offset]
        for attempt in range(max_retries):
            try:
                cursor.execute(query, params_page)
                projects = [self._row_to_dict(cursor, row) for row in cursor.fetchall()]
                return {
                    "projects": projects,
                    "pagination": {
                        "page": page,
                        "per_page": per_page,
                        "total": total,
                        "total_pages": (total + per_page - 1) // per_page
                    }
                }
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e).lower() and attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) * (0.5 + random.random())
                    time.sleep(delay)
                    continue
                raise DatabaseError(f"Failed to list projects: {e}")  # noqa: B904, TRY003
            except sqlite3.Error as e:
                raise DatabaseError(f"Failed to list projects: {e}")  # noqa: B904, TRY003

    def update_project(self, project_id: int, updates: dict[str, Any]) -> dict[str, Any]:
        """
        Update a project.

        Args:
            project_id: Project ID
            updates: Fields to update

        Returns:
            Updated project record
        """
        import random
        import sqlite3
        import time

        conn = self.get_connection()
        max_retries = 5
        base_delay = 0.05

        for attempt in range(max_retries):
            project_uuid = None
            try:
                with self._write_lock:
                    cursor = conn.cursor()

                    # Build update query
                    allowed_fields = ["name", "description", "status", "metadata"]
                    set_clauses: list[str] = []
                    params: list[Any] = []

                    for field in allowed_fields:
                        if field in updates:
                            set_clauses.append(f"{field} = ?")
                            value = updates[field]
                            if field == "metadata" and value is not None:
                                value = json.dumps(value)
                            params.append(value)

                    if not set_clauses:
                        return self.get_project(project_id)

                    set_clauses.append("updated_at = CURRENT_TIMESTAMP")
                    params.append(project_id)

                    query = (
                        "UPDATE prompt_studio_projects "  # nosec B608
                        f"SET {', '.join(set_clauses)} "
                        "WHERE id = ? AND deleted = 0"
                    )

                    cursor.execute(query, params)

                    if cursor.rowcount == 0:
                        raise InputError(f"Project {project_id} not found or already deleted")  # noqa: TRY003

                    cursor.execute(
                        "SELECT uuid FROM prompt_studio_projects WHERE id = ?",
                        (project_id,),
                    )
                    row = cursor.fetchone()
                    if row:
                        project_uuid = row[0]

                    conn.commit()

                if project_uuid:
                    self._log_sync_event(
                        "prompt_studio_project",
                        project_uuid,
                        "update",
                        updates,
                    )

                return self.get_project(project_id)

            except sqlite3.IntegrityError as exc:
                if "UNIQUE" in str(exc):
                    raise ConflictError("Project with name already exists")  # noqa: B904, TRY003
                raise DatabaseError(f"Failed to update project: {exc}")  # noqa: B904, TRY003
            except sqlite3.OperationalError as exc:
                if "database is locked" in str(exc).lower() and attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) * (0.5 + random.random())
                    time.sleep(delay)
                    continue
                raise DatabaseError(f"Failed to update project: {exc}")  # noqa: B904, TRY003
            except _PROMPT_STUDIO_NONCRITICAL_EXCEPTIONS as exc:  # noqa: BLE001
                raise DatabaseError(f"Failed to update project: {exc}")  # noqa: B904, TRY003

        raise DatabaseError("Failed to update project after retries")  # noqa: TRY003

    ####################################################################################################################
    # Signature Management

    def create_signature(
        self,
        project_id: int,
        name: str,
        *,
        input_schema: Iterable[Any],
        output_schema: Iterable[Any],
        constraints: Optional[Any] = None,
        validation_rules: Optional[Any] = None,
        client_id: Optional[str] = None,
    ) -> dict[str, Any]:
        import random
        import sqlite3
        import time

        if not name or not str(name).strip():
            raise InputError("Signature name cannot be empty")  # noqa: TRY003

        conn = self.get_connection()
        signature_uuid = str(uuid.uuid4())
        payload = (
            signature_uuid,
            project_id,
            str(name).strip(),
            json.dumps(list(input_schema) if input_schema is not None else []),
            json.dumps(list(output_schema) if output_schema is not None else []),
            json.dumps(constraints) if constraints is not None else None,
            json.dumps(validation_rules) if validation_rules is not None else None,
            client_id or self.client_id,
        )

        insert_sql = """
            INSERT INTO prompt_studio_signatures (
                uuid, project_id, name, input_schema, output_schema,
                constraints, validation_rules, client_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """

        max_retries = 5
        base_delay = 0.05

        for attempt in range(max_retries):
            should_retry = False
            with self._write_lock:
                try:
                    cursor = conn.cursor()
                    cursor.execute(insert_sql, payload)
                    signature_id = cursor.lastrowid
                    conn.commit()

                    cursor.execute(
                        "SELECT * FROM prompt_studio_signatures WHERE id = ?",
                        (signature_id,),
                    )
                    row = cursor.fetchone()
                    signature = self._row_to_dict(cursor, row) if row else {}

                    self._log_sync_event(
                        "prompt_studio_signature",
                        signature_uuid,
                        "create",
                        {
                            "project_id": project_id,
                            "name": name,
                        },
                    )
                    return signature  # noqa: TRY300
                except sqlite3.IntegrityError as exc:
                    message = str(exc)
                    if "UNIQUE" in message:
                        raise ConflictError(  # noqa: B904, TRY003
                            f"Signature with name '{name}' already exists for project {project_id}"
                        )
                    raise DatabaseError(f"Failed to create signature: {exc}") from exc  # noqa: TRY003
                except sqlite3.OperationalError as exc:
                    if "database is locked" in str(exc).lower() and attempt < max_retries - 1:
                        should_retry = True
                        delay = base_delay * (2 ** attempt) * (0.5 + random.random())
                    else:
                        raise DatabaseError(f"Failed to create signature: {exc}") from exc  # noqa: TRY003
                except sqlite3.Error as exc:  # noqa: BLE001
                    raise DatabaseError(f"Failed to create signature: {exc}") from exc  # noqa: TRY003

            if should_retry:
                time.sleep(delay)

        raise DatabaseError("Failed to create signature due to database locks")  # noqa: TRY003

    def get_signature(
        self,
        signature_id: int,
        *,
        include_deleted: bool = False,
    ) -> Optional[dict[str, Any]]:
        import random
        import sqlite3
        import time

        conn = self.get_connection()
        cursor = conn.cursor()

        query = "SELECT * FROM prompt_studio_signatures WHERE id = ?"
        params: list[Any] = [signature_id]
        if not include_deleted:
            query += " AND deleted = 0"

        max_retries = 5
        base_delay = 0.05
        for attempt in range(max_retries):
            try:
                cursor.execute(query, params)
                row = cursor.fetchone()
                return self._row_to_dict(cursor, row) if row else None
            except sqlite3.OperationalError as exc:
                if "database is locked" in str(exc).lower() and attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) * (0.5 + random.random())
                    time.sleep(delay)
                    continue
                raise DatabaseError(f"Failed to fetch signature {signature_id}: {exc}") from exc  # noqa: TRY003
            except sqlite3.Error as exc:  # noqa: BLE001
                raise DatabaseError(f"Failed to fetch signature {signature_id}: {exc}") from exc  # noqa: TRY003

        return None

    def list_signatures(
        self,
        project_id: int,
        *,
        include_deleted: bool = False,
        search: Optional[str] = None,
        page: int = 1,
        per_page: int = 20,
        return_pagination: bool = False,
    ) -> Union[dict[str, Any], list[dict[str, Any]]]:
        import random
        import sqlite3
        import time

        if page < 1:
            raise InputError("Page index must be >= 1")  # noqa: TRY003
        if per_page < 1:
            raise InputError("Items per page must be >= 1")  # noqa: TRY003

        conn = self.get_connection()
        cursor = conn.cursor()

        conditions = ["project_id = ?"]
        params: list[Any] = [project_id]
        if not include_deleted:
            conditions.append("deleted = 0")
        if search:
            conditions.append("name LIKE ?")
            params.append(f"%{search}%")

        where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""

        count_sql = f"SELECT COUNT(*) FROM prompt_studio_signatures{where_clause}"  # nosec B608

        max_retries = 5
        base_delay = 0.05

        for attempt in range(max_retries):
            try:
                cursor.execute(count_sql, params)
                total_row = cursor.fetchone()
                total = int(total_row[0]) if total_row else 0
                break
            except sqlite3.OperationalError as exc:
                if "database is locked" in str(exc).lower() and attempt < max_retries - 1:
                    time.sleep(base_delay * (2 ** attempt) * (0.5 + random.random()))
                    continue
                raise DatabaseError(f"Failed to count signatures: {exc}") from exc  # noqa: TRY003
            except sqlite3.Error as exc:  # noqa: BLE001
                raise DatabaseError(f"Failed to count signatures: {exc}") from exc  # noqa: TRY003
        else:
            raise DatabaseError("Failed to count signatures due to database locks")  # noqa: TRY003

        offset = max(page - 1, 0) * per_page
        list_sql = (
            f"SELECT * FROM prompt_studio_signatures{where_clause} "  # nosec B608
            "ORDER BY updated_at DESC, id DESC LIMIT ? OFFSET ?"
        )
        params_with_pagination = params + [per_page, offset]

        for attempt in range(max_retries):
            try:
                cursor.execute(list_sql, params_with_pagination)
                rows = cursor.fetchall()
                signatures = [self._row_to_dict(cursor, row) for row in rows if row]
                break
            except sqlite3.OperationalError as exc:
                if "database is locked" in str(exc).lower() and attempt < max_retries - 1:
                    time.sleep(base_delay * (2 ** attempt) * (0.5 + random.random()))
                    continue
                raise DatabaseError(f"Failed to list signatures: {exc}") from exc  # noqa: TRY003
            except sqlite3.Error as exc:  # noqa: BLE001
                raise DatabaseError(f"Failed to list signatures: {exc}") from exc  # noqa: TRY003
        else:
            raise DatabaseError("Failed to list signatures due to database locks")  # noqa: TRY003

        if return_pagination:
            return {
                "signatures": signatures,
                "pagination": {
                    "page": page,
                    "per_page": per_page,
                    "total": total,
                    "total_pages": (total + per_page - 1) // per_page if per_page else 0,
                },
            }
        return signatures

    def update_signature(self, signature_id: int, updates: dict[str, Any]) -> dict[str, Any]:
        import sqlite3

        allowed_fields = {
            "name",
            "input_schema",
            "output_schema",
            "constraints",
            "validation_rules",
        }

        set_clauses: list[str] = []
        params: list[Any] = []

        for field, value in updates.items():
            if field not in allowed_fields:
                continue

            if field in {"input_schema", "output_schema", "constraints", "validation_rules"} and value is not None:
                params.append(json.dumps(value))
            else:
                params.append(value)
            set_clauses.append(f"{field} = ?")

        if not set_clauses:
            signature = self.get_signature(signature_id, include_deleted=True)
            if signature is None:
                raise InputError(f"Signature {signature_id} not found or already deleted")  # noqa: TRY003
            return signature

        set_clauses.append("updated_at = CURRENT_TIMESTAMP")
        params.append(signature_id)

        update_sql = (
            "UPDATE prompt_studio_signatures SET "  # nosec B608
            + ", ".join(set_clauses)
            + " WHERE id = ? AND deleted = 0"
        )

        conn = self.get_connection()

        with self._write_lock:
            try:
                cursor = conn.cursor()
                cursor.execute(update_sql, params)
                if cursor.rowcount == 0:
                    raise InputError(f"Signature {signature_id} not found or already deleted")  # noqa: TRY003
                conn.commit()
                cursor.execute(
                    "SELECT * FROM prompt_studio_signatures WHERE id = ?",
                    (signature_id,),
                )
                row = cursor.fetchone()
                if not row:
                    raise DatabaseError(f"Failed to fetch updated signature {signature_id}")  # noqa: TRY003
                signature = self._row_to_dict(cursor, row)
            except sqlite3.IntegrityError as exc:
                message = str(exc)
                if "UNIQUE" in message:
                    raise ConflictError("Signature update conflicts with existing record") from exc  # noqa: TRY003
                raise DatabaseError(f"Failed to update signature: {exc}") from exc  # noqa: TRY003
            except sqlite3.Error as exc:  # noqa: BLE001
                raise DatabaseError(f"Failed to update signature: {exc}") from exc  # noqa: TRY003

        self._log_sync_event(
            "prompt_studio_signature",
            signature.get("uuid", ""),
            "update",
            {key: updates[key] for key in updates if key in allowed_fields},
        )
        return signature

    def delete_signature(self, signature_id: int, *, hard_delete: bool = False) -> bool:
        import sqlite3
        import time

        conn = self.get_connection()
        cursor = conn.cursor()
        max_retries = 5
        base_delay = 0.05

        for attempt in range(max_retries):
            try:
                if hard_delete:
                    cursor.execute(
                        "SELECT uuid FROM prompt_studio_signatures WHERE id = ?",
                        (signature_id,),
                    )
                    row = cursor.fetchone()
                    if not row:
                        return False
                    signature_uuid = row[0]
                    cursor.execute(
                        "DELETE FROM prompt_studio_signatures WHERE id = ?",
                        (signature_id,),
                    )
                else:
                    cursor.execute(
                        "SELECT uuid FROM prompt_studio_signatures WHERE id = ? AND deleted = 0",
                        (signature_id,),
                    )
                    row = cursor.fetchone()
                    if not row:
                        return False
                    signature_uuid = row[0]
                    cursor.execute(
                        """
                        UPDATE prompt_studio_signatures
                        SET deleted = 1, deleted_at = CURRENT_TIMESTAMP
                        WHERE id = ? AND deleted = 0
                        """,
                        (signature_id,),
                    )

                if cursor.rowcount > 0:
                    conn.commit()
                    self._log_sync_event(
                        "prompt_studio_signature",
                        signature_uuid,
                        "delete" if hard_delete else "soft_delete",
                        {"hard": hard_delete},
                    )
                    return True
                return False  # noqa: TRY300
            except sqlite3.OperationalError as exc:
                if "database is locked" in str(exc).lower() and attempt < max_retries - 1:
                    time.sleep(base_delay * (2 ** attempt))
                    continue
                raise DatabaseError(f"Failed to delete signature {signature_id}: {exc}") from exc  # noqa: TRY003
            except sqlite3.Error as exc:  # noqa: BLE001
                raise DatabaseError(f"Failed to delete signature {signature_id}: {exc}") from exc  # noqa: TRY003

        raise DatabaseError("Failed to delete signature due to database locks")  # noqa: TRY003

    ####################################################################################################################
    # Test Run Management

    def create_test_run(
        self,
        *,
        project_id: int,
        prompt_id: int,
        test_case_id: int,
        model_name: str,
        model_params: Optional[dict[str, Any]] = None,
        inputs: Optional[dict[str, Any]] = None,
        outputs: Optional[dict[str, Any]] = None,
        expected_outputs: Optional[dict[str, Any]] = None,
        scores: Optional[dict[str, Any]] = None,
        execution_time_ms: Optional[int] = None,
        tokens_used: Optional[int] = None,
        cost_estimate: Optional[float] = None,
        error_message: Optional[str] = None,
        client_id: Optional[str] = None,
    ) -> dict[str, Any]:
        import random
        import sqlite3
        import time

        conn = self.get_connection()
        cursor = conn.cursor()
        run_uuid = str(uuid.uuid4())

        payload = (
            run_uuid,
            project_id,
            prompt_id,
            test_case_id,
            model_name,
            json.dumps(model_params) if model_params is not None else None,
            json.dumps(inputs) if inputs is not None else None,
            json.dumps(outputs) if outputs is not None else None,
            json.dumps(expected_outputs) if expected_outputs is not None else None,
            json.dumps(scores) if scores is not None else None,
            execution_time_ms,
            tokens_used,
            cost_estimate,
            error_message,
            client_id or self.client_id,
        )

        insert_sql = """
            INSERT INTO prompt_studio_test_runs (
                uuid, project_id, prompt_id, test_case_id, model_name,
                model_params, inputs, outputs, expected_outputs, scores,
                execution_time_ms, tokens_used, cost_estimate, error_message,
                client_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        base_delay = 0.05
        for attempt in range(5):
            try:
                with self._write_lock:
                    cursor.execute(insert_sql, payload)
                    run_id = cursor.lastrowid
                    conn.commit()
                    cursor.execute(
                        "SELECT * FROM prompt_studio_test_runs WHERE id = ?",
                        (run_id,),
                    )
                    row = cursor.fetchone()
                    return self._row_to_dict(cursor, row) if row else {}
            except sqlite3.OperationalError as exc:
                if "database is locked" in str(exc).lower() and attempt < 4:
                    time.sleep(base_delay * (2 ** attempt) * (0.5 + random.random()))
                    continue
                raise DatabaseError(f"Failed to create test run: {exc}") from exc  # noqa: TRY003
            except sqlite3.Error as exc:  # noqa: BLE001
                raise DatabaseError(f"Failed to create test run: {exc}") from exc  # noqa: TRY003

        raise DatabaseError("Failed to create test run due to database locks")  # noqa: TRY003

    def get_test_cases_by_ids(
        self,
        test_case_ids: Iterable[int],
        *,
        include_deleted: bool = False,
    ) -> list[dict[str, Any]]:
        import sqlite3

        identifiers = list(dict.fromkeys(test_case_ids))
        if not identifiers:
            return []

        conn = self.get_connection()
        cursor = conn.cursor()

        placeholders = ",".join(["?"] * len(identifiers))
        query = f"SELECT * FROM prompt_studio_test_cases WHERE id IN ({placeholders})"  # nosec B608
        if not include_deleted:
            query += " AND deleted = 0"

        try:
            cursor.execute(query, identifiers)
            rows = cursor.fetchall()
            return [self._format_test_case(cursor, row) for row in rows if row]
        except sqlite3.Error as exc:  # noqa: BLE001
            raise DatabaseError(f"Failed to fetch test cases: {exc}") from exc  # noqa: TRY003

    ####################################################################################################################
    # Evaluation Management

    def create_evaluation(
        self,
        *,
        prompt_id: int,
        project_id: int,
        model_configs: Optional[dict[str, Any]] = None,
        status: str = "running",
        test_case_ids: Optional[Iterable[int]] = None,
        client_id: Optional[str] = None,
    ) -> dict[str, Any]:
        import random
        import sqlite3
        import time

        conn = self.get_connection()
        cursor = conn.cursor()

        eval_uuid = str(uuid.uuid4())
        payload = (
            eval_uuid,
            prompt_id,
            project_id,
            json.dumps(model_configs) if model_configs is not None else None,
            status,
            json.dumps(list(test_case_ids) if test_case_ids is not None else []),
            client_id or self.client_id,
        )

        insert_sql = """
            INSERT INTO prompt_studio_evaluations (
                uuid, prompt_id, project_id, model_configs, status,
                test_case_ids, started_at, client_id
            ) VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?)
        """

        base_delay = 0.05
        for attempt in range(5):
            try:
                with self._write_lock:
                    cursor.execute(insert_sql, payload)
                    eval_id = cursor.lastrowid
                    conn.commit()
                    cursor.execute(
                        "SELECT * FROM prompt_studio_evaluations WHERE id = ?",
                        (eval_id,),
                    )
                    row = cursor.fetchone()
                    return self._row_to_dict(cursor, row) if row else {}
            except sqlite3.OperationalError as exc:
                if "database is locked" in str(exc).lower() and attempt < 4:
                    time.sleep(base_delay * (2 ** attempt) * (0.5 + random.random()))
                    continue
                raise DatabaseError(f"Failed to create evaluation: {exc}") from exc  # noqa: TRY003
            except sqlite3.Error as exc:  # noqa: BLE001
                raise DatabaseError(f"Failed to create evaluation: {exc}") from exc  # noqa: TRY003

        raise DatabaseError("Failed to create evaluation due to database locks")  # noqa: TRY003

    def update_evaluation(self, evaluation_id: int, updates: dict[str, Any]) -> dict[str, Any]:
        import sqlite3

        if not updates:
            evaluation = self.get_evaluation(evaluation_id)
            if evaluation is None:
                raise InputError(f"Evaluation {evaluation_id} not found")  # noqa: TRY003
            return evaluation

        json_fields = {"model_configs", "test_case_ids", "test_run_ids", "aggregate_metrics"}
        set_clauses: list[str] = []
        params: list[Any] = []

        for field, value in updates.items():
            if field in json_fields and value is not None:
                params.append(json.dumps(value))
            else:
                params.append(value)
            set_clauses.append(f"{field} = ?")

        params.append(evaluation_id)

        query = (
            "UPDATE prompt_studio_evaluations SET "  # nosec B608
            + ", ".join(set_clauses)
            + " WHERE id = ?"
        )

        with self._write_lock:
            cursor = self.get_connection().cursor()
            try:
                cursor.execute(query, params)
                if cursor.rowcount == 0:
                    raise InputError(f"Evaluation {evaluation_id} not found")  # noqa: TRY003
                self.get_connection().commit()
                cursor.execute(
                    "SELECT * FROM prompt_studio_evaluations WHERE id = ?",
                    (evaluation_id,),
                )
                row = cursor.fetchone()
                if not row:
                    raise DatabaseError(f"Failed to fetch evaluation {evaluation_id}")  # noqa: TRY003
                return self._row_to_dict(cursor, row)
            except sqlite3.Error as exc:  # noqa: BLE001
                raise DatabaseError(f"Failed to update evaluation: {exc}") from exc  # noqa: TRY003

    def get_evaluation(self, evaluation_id: int) -> Optional[dict[str, Any]]:
        import sqlite3

        cursor = self.get_connection().cursor()
        try:
            cursor.execute(
                "SELECT * FROM prompt_studio_evaluations WHERE id = ?",
                (evaluation_id,),
            )
            row = cursor.fetchone()
            if row:
                return self._row_to_dict(cursor, row)
            return None  # noqa: TRY300
        except sqlite3.Error as exc:  # noqa: BLE001
            raise DatabaseError(f"Failed to fetch evaluation {evaluation_id}: {exc}") from exc  # noqa: TRY003

    def list_evaluations(
        self,
        project_id: Optional[int] = None,
        prompt_id: Optional[int] = None,
        status: Optional[str] = None,
        page: int = 1,
        per_page: int = 20,
    ) -> dict[str, Any]:
        import random
        import sqlite3
        import time

        if page < 1:
            raise InputError("Page index must be >= 1")  # noqa: TRY003
        if per_page < 1:
            raise InputError("Items per page must be >= 1")  # noqa: TRY003

        conn = self.get_connection()
        cursor = conn.cursor()

        conditions: list[str] = []
        params: list[Any] = []
        if project_id is not None:
            conditions.append("project_id = ?")
            params.append(project_id)
        if prompt_id is not None:
            conditions.append("prompt_id = ?")
            params.append(prompt_id)
        if status is not None:
            conditions.append("status = ?")
            params.append(status)

        where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""

        count_query = f"SELECT COUNT(*) FROM prompt_studio_evaluations{where_clause}"  # nosec B608

        base_delay = 0.05
        for attempt in range(5):
            try:
                cursor.execute(count_query, params)
                total = cursor.fetchone()[0]
                break
            except sqlite3.OperationalError as exc:
                if "database is locked" in str(exc).lower() and attempt < 4:
                    time.sleep(base_delay * (2 ** attempt) * (0.5 + random.random()))
                    continue
                raise DatabaseError(f"Failed to list evaluations: {exc}") from exc  # noqa: TRY003
            except sqlite3.Error as exc:  # noqa: BLE001
                raise DatabaseError(f"Failed to list evaluations: {exc}") from exc  # noqa: TRY003
        else:
            raise DatabaseError("Failed to list evaluations due to database locks")  # noqa: TRY003

        offset = (page - 1) * per_page
        query = """
            SELECT *
            FROM prompt_studio_evaluations
            {where_clause}
            ORDER BY started_at DESC, id DESC
            LIMIT ? OFFSET ?
        """.format_map(locals())  # nosec B608
        params_with_page = list(params) + [per_page, offset]

        for attempt in range(5):
            try:
                cursor.execute(query, params_with_page)
                rows = cursor.fetchall()
                evaluations = [self._row_to_dict(cursor, row) for row in rows if row]
                return {
                    "evaluations": evaluations,
                    "pagination": {
                        "page": page,
                        "per_page": per_page,
                        "total": total,
                        "total_pages": (total + per_page - 1) // per_page
                    }
                }
            except sqlite3.OperationalError as exc:
                if "database is locked" in str(exc).lower() and attempt < 4:
                    time.sleep(base_delay * (2 ** attempt) * (0.5 + random.random()))
                    continue
                raise DatabaseError(f"Failed to list evaluations: {exc}") from exc  # noqa: TRY003
            except sqlite3.Error as exc:  # noqa: BLE001
                raise DatabaseError(f"Failed to list evaluations: {exc}") from exc  # noqa: TRY003

        raise DatabaseError("Failed to list evaluations due to database locks")  # noqa: TRY003

    def create_prompt(
        self,
        project_id: int,
        name: str,
        *,
        signature_id: Optional[int] = None,
        version_number: int = 1,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        prompt_format: str = "legacy",
        prompt_schema_version: Optional[int] = None,
        prompt_definition: Optional[Any] = None,
        few_shot_examples: Optional[Any] = None,
        modules_config: Optional[Any] = None,
        parent_version_id: Optional[int] = None,
        change_description: Optional[str] = None,
        client_id: Optional[str] = None,
    ) -> dict[str, Any]:
        import random
        import time

        normalized_prompt_fields = _prepare_prompt_record_fields(
            prompt_format=prompt_format,
            prompt_schema_version=prompt_schema_version,
            prompt_definition=prompt_definition,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        prompt_uuid = str(uuid.uuid4())
        payload = (
            prompt_uuid,
            project_id,
            signature_id,
            version_number,
            name,
            normalized_prompt_fields["system_prompt"],
            normalized_prompt_fields["user_prompt"],
            normalized_prompt_fields["prompt_format"],
            normalized_prompt_fields["prompt_schema_version"],
            json.dumps(normalized_prompt_fields["prompt_definition"])
            if normalized_prompt_fields["prompt_definition"] is not None
            else None,
            json.dumps(few_shot_examples) if few_shot_examples is not None else None,
            json.dumps(modules_config) if modules_config is not None else None,
            parent_version_id,
            change_description,
            client_id or self.client_id,
        )

        insert_sql = """
            INSERT INTO prompt_studio_prompts (
                uuid, project_id, signature_id, version_number, name, system_prompt,
                user_prompt, prompt_format, prompt_schema_version, prompt_definition,
                few_shot_examples, modules_config, parent_version_id,
                change_description, client_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        conn = self.get_connection()
        max_retries = 5
        base_delay = 0.05

        for attempt in range(max_retries):
            should_retry = False
            with self._write_lock:
                try:
                    cursor = conn.cursor()
                    cursor.execute(insert_sql, payload)
                    prompt_id = cursor.lastrowid
                    conn.commit()
                    prompt = self.get_prompt(prompt_id)
                    self._log_sync_event(
                        "prompt_studio_prompt",
                        prompt_uuid,
                        "create",
                        {
                            "project_id": project_id,
                            "name": name,
                            "version_number": version_number,
                        },
                    )
                    return prompt or {}  # noqa: TRY300
                except sqlite3.OperationalError as exc:
                    if "database is locked" in str(exc).lower() and attempt < max_retries - 1:
                        should_retry = True
                        delay = base_delay * (2 ** attempt) * (0.5 + random.random())
                    else:
                        raise DatabaseError(f"Failed to create prompt: {exc}") from exc  # noqa: TRY003
                except sqlite3.IntegrityError as exc:
                    if "UNIQUE" in str(exc).upper():
                        raise ConflictError(  # noqa: TRY003
                            f"Prompt with name '{name}' already exists in project {project_id}"
                        ) from exc
                    raise DatabaseError(f"Failed to create prompt: {exc}") from exc  # noqa: TRY003

            if should_retry:
                time.sleep(delay)

        raise DatabaseError("Failed to create prompt after multiple retries")  # noqa: TRY003

    def delete_project(self, project_id: int, hard_delete: bool = False) -> bool:
        """
        Delete a project (soft delete by default).

        Args:
            project_id: Project ID
            hard_delete: Permanently delete if True

        Returns:
            True if deleted
        """
        import random
        import sqlite3
        import time

        conn = self.get_connection()
        max_retries = 5
        base_delay = 0.1

        for attempt in range(max_retries):
            should_retry = False
            try:
                with self._write_lock:
                    cursor = conn.cursor()
                    if hard_delete:
                        # Cascade delete all related data
                        cursor.execute("DELETE FROM prompt_studio_projects WHERE id = ?", (project_id,))
                    else:
                        # Soft delete
                        cursor.execute(
                            """
                            UPDATE prompt_studio_projects
                            SET deleted = 1, deleted_at = CURRENT_TIMESTAMP
                            WHERE id = ? AND deleted = 0
                            """,
                            (project_id,)
                        )
                    success = cursor.rowcount > 0
                    if success:
                        conn.commit()
                        logger.info(f"{'Hard' if hard_delete else 'Soft'} deleted project {project_id}")
                    return success
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e) and attempt < max_retries - 1:
                    should_retry = True
                    delay = base_delay * (2 ** attempt) * (0.5 + random.random())
                    logger.warning(f"Delete project locked, retrying in {delay:.3f}s (attempt {attempt+1})")
                    time.sleep(delay)
                else:
                    raise DatabaseError(f"Failed to delete project: {e}")  # noqa: B904, TRY003
            except _PROMPT_STUDIO_NONCRITICAL_EXCEPTIONS as e:
                raise DatabaseError(f"Failed to delete project: {e}")  # noqa: B904, TRY003

            if not should_retry:
                break

        return False

    ####################################################################################################################
    # Helper Methods

    def _row_to_dict(self, cursor: sqlite3.Cursor, row: tuple) -> dict[str, Any]:
        """Convert a database row to dictionary."""
        if not row:
            return None

        columns = [description[0] for description in cursor.description]
        result = dict(zip(columns, row))

        # Parse JSON fields
        json_fields = ["metadata", "input_schema", "output_schema", "constraints",
                      "validation_rules", "few_shot_examples", "modules_config",
                      "prompt_definition",
                      "model_params", "inputs", "outputs", "expected_outputs",
                      "actual_outputs", "scores", "test_case_ids", "test_run_ids",
                      "aggregate_metrics", "model_configs", "payload", "result",
                      "initial_metrics", "final_metrics", "optimization_config"]

        for field in json_fields:
            if field in result and result[field]:
                with suppress(json.JSONDecodeError, TypeError):
                    result[field] = json.loads(result[field])

        # Parse datetime fields
        datetime_fields = ["created_at", "updated_at", "deleted_at", "last_modified",
                          "started_at", "completed_at"]

        for field in datetime_fields:
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

    def _format_test_case(self, cursor: sqlite3.Cursor, row: Optional[sqlite3.Row]) -> Optional[dict[str, Any]]:
        if row is None:
            return None
        return _format_test_case_record(self._row_to_dict(cursor, row))

    ####################################################################################################################
    # Prompt Accessors (Prompt Studio tables)

    def get_prompt(self, prompt_id: int) -> Optional[dict[str, Any]]:
        """
        Fetch a prompt-studio prompt by id from the prompt_studio_prompts table.

        Args:
            prompt_id: ID of the prompt (prompt_studio_prompts.id)

        Returns:
            A dictionary representing the prompt or None if not found.
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT *
                FROM prompt_studio_prompts
                WHERE id = ? AND deleted = 0
                """,
                (prompt_id,)
            )
            row = cursor.fetchone()
            if not row:
                return None
            return self._row_to_dict(cursor, row)
        except _PROMPT_STUDIO_NONCRITICAL_EXCEPTIONS as e:
            logger.error(f"Failed to get prompt {prompt_id}: {e}")
            return None

    def get_prompt_with_project(
        self,
        prompt_id: int,
        *,
        include_deleted: bool = False,
    ) -> Optional[dict[str, Any]]:
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            clause = "" if include_deleted else "AND p.deleted = 0"
            cursor.execute(
                """
                SELECT p.*, proj.user_id AS project_user_id
                FROM prompt_studio_prompts p
                JOIN prompt_studio_projects proj ON p.project_id = proj.id
                WHERE p.id = ? {clause}
                """.format_map(locals()),  # nosec B608
                (prompt_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None
            return self._row_to_dict(cursor, row)
        except _PROMPT_STUDIO_NONCRITICAL_EXCEPTIONS as exc:  # noqa: BLE001
            logger.error(f"Failed to fetch prompt {prompt_id}: {exc}")
            return None

    def create_prompt_version(
        self,
        prompt_id: int,
        *,
        change_description: str,
        name: Optional[str] = None,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        prompt_format: Optional[str] = None,
        prompt_schema_version: Optional[int] = None,
        prompt_definition: Optional[Any] = None,
        few_shot_examples: Optional[Any] = None,
        modules_config: Optional[Any] = None,
        client_id: Optional[str] = None,
    ) -> dict[str, Any]:
        import random
        import time

        if not change_description:
            raise InputError("change_description is required")  # noqa: TRY003

        conn = self.get_connection()
        max_retries = 5
        base_delay = 0.05

        for attempt in range(max_retries):
            should_retry = False
            with self._write_lock:
                try:
                    cursor = conn.cursor()
                    cursor.execute(
                        """
                        SELECT *
                        FROM prompt_studio_prompts
                        WHERE id = ? AND deleted = 0
                        """,
                        (prompt_id,),
                    )
                    current_row = cursor.fetchone()
                    if not current_row:
                        raise InputError(f"Prompt {prompt_id} not found or already deleted")  # noqa: TRY003
                    current_prompt = self._row_to_dict(cursor, current_row)

                    new_uuid = str(uuid.uuid4())
                    new_version = int(current_prompt.get("version_number", 0)) + 1

                    next_name = name if name is not None else current_prompt.get("name")
                    normalized_prompt_fields = _prepare_prompt_record_fields(
                        prompt_format=prompt_format,
                        prompt_schema_version=prompt_schema_version,
                        prompt_definition=prompt_definition,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        current_prompt=current_prompt,
                    )
                    next_examples = (
                        few_shot_examples
                        if few_shot_examples is not None
                        else current_prompt.get("few_shot_examples")
                    )
                    next_modules = (
                        modules_config
                        if modules_config is not None
                        else current_prompt.get("modules_config")
                    )

                    cursor.execute(
                        """
                        INSERT INTO prompt_studio_prompts (
                            uuid, project_id, signature_id, version_number, name,
                            system_prompt, user_prompt, prompt_format, prompt_schema_version,
                            prompt_definition, few_shot_examples, modules_config,
                            parent_version_id, change_description, client_id
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            new_uuid,
                            current_prompt.get("project_id"),
                            current_prompt.get("signature_id"),
                            new_version,
                            next_name,
                            normalized_prompt_fields["system_prompt"],
                            normalized_prompt_fields["user_prompt"],
                            normalized_prompt_fields["prompt_format"],
                            normalized_prompt_fields["prompt_schema_version"],
                            json.dumps(normalized_prompt_fields["prompt_definition"])
                            if normalized_prompt_fields["prompt_definition"] is not None
                            else None,
                            json.dumps(next_examples) if next_examples is not None else None,
                            json.dumps(next_modules) if next_modules is not None else None,
                            prompt_id,
                            change_description,
                            client_id or current_prompt.get("client_id") or self.client_id,
                        ),
                    )

                    new_prompt_id = cursor.lastrowid
                    conn.commit()

                    cursor.execute(
                        "SELECT * FROM prompt_studio_prompts WHERE id = ?",
                        (new_prompt_id,),
                    )
                    row = cursor.fetchone()
                    prompt = self._row_to_dict(cursor, row) if row else {}

                    self._log_sync_event(
                        "prompt_studio_prompt",
                        prompt.get("uuid", ""),
                        "version_create",
                        {
                            "prompt_id": prompt_id,
                            "new_version": prompt.get("version_number"),
                            "change_description": change_description,
                        },
                    )
                    return prompt  # noqa: TRY300
                except sqlite3.OperationalError as exc:
                    if "database is locked" in str(exc).lower() and attempt < max_retries - 1:
                        should_retry = True
                        delay = base_delay * (2 ** attempt) * (0.5 + random.random())
                    else:
                        raise DatabaseError(f"Failed to create prompt version: {exc}") from exc  # noqa: TRY003
                except sqlite3.Error as exc:  # noqa: BLE001
                    raise DatabaseError(f"Failed to create prompt version: {exc}") from exc  # noqa: TRY003

            if should_retry:
                time.sleep(delay)

        raise DatabaseError("Failed to create prompt version due to database locks")  # noqa: TRY003

    def revert_prompt_to_version(
        self,
        prompt_id: int,
        target_version: int,
        *,
        client_id: Optional[str] = None,
    ) -> dict[str, Any]:
        import random
        import time

        if target_version < 1:
            raise InputError("target_version must be >= 1")  # noqa: TRY003

        conn = self.get_connection()
        max_retries = 5
        base_delay = 0.05

        for attempt in range(max_retries):
            should_retry = False
            with self._write_lock:
                try:
                    cursor = conn.cursor()
                    cursor.execute(
                        """
                        SELECT * FROM prompt_studio_prompts
                        WHERE id = ? AND deleted = 0
                        """,
                        (prompt_id,),
                    )
                    current_row = cursor.fetchone()
                    if not current_row:
                        raise InputError(f"Prompt {prompt_id} not found or already deleted")  # noqa: TRY003
                    current_prompt = self._row_to_dict(cursor, current_row)

                    cursor.execute(
                        """
                        SELECT * FROM prompt_studio_prompts
                        WHERE project_id = ? AND name = ? AND version_number = ? AND deleted = 0
                        """,
                        (
                            current_prompt.get("project_id"),
                            current_prompt.get("name"),
                            target_version,
                        ),
                    )
                    target_row = cursor.fetchone()
                    if not target_row:
                        raise InputError(  # noqa: TRY003
                            f"Version {target_version} not found for this prompt"
                        )
                    target_prompt = self._row_to_dict(cursor, target_row)

                    cursor.execute(
                        """
                        SELECT MAX(version_number) FROM prompt_studio_prompts
                        WHERE project_id = ? AND name = ?
                        """,
                        (current_prompt.get("project_id"), current_prompt.get("name")),
                    )
                    max_version = cursor.fetchone()[0] or 0
                    new_version = max_version + 1

                    new_uuid = str(uuid.uuid4())
                    normalized_prompt_fields = _prepare_prompt_record_fields(
                        prompt_format=target_prompt.get("prompt_format"),
                        prompt_schema_version=target_prompt.get("prompt_schema_version"),
                        prompt_definition=target_prompt.get("prompt_definition"),
                        system_prompt=target_prompt.get("system_prompt"),
                        user_prompt=target_prompt.get("user_prompt"),
                    )
                    cursor.execute(
                        """
                        INSERT INTO prompt_studio_prompts (
                            uuid, project_id, signature_id, version_number, name,
                            system_prompt, user_prompt, prompt_format, prompt_schema_version,
                            prompt_definition, few_shot_examples, modules_config,
                            parent_version_id, change_description, client_id
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            new_uuid,
                            target_prompt.get("project_id"),
                            target_prompt.get("signature_id"),
                            new_version,
                            target_prompt.get("name"),
                            normalized_prompt_fields["system_prompt"],
                            normalized_prompt_fields["user_prompt"],
                            normalized_prompt_fields["prompt_format"],
                            normalized_prompt_fields["prompt_schema_version"],
                            json.dumps(normalized_prompt_fields["prompt_definition"])
                            if normalized_prompt_fields["prompt_definition"] is not None
                            else None,
                            json.dumps(target_prompt.get("few_shot_examples"))
                            if target_prompt.get("few_shot_examples") is not None
                            else None,
                            json.dumps(target_prompt.get("modules_config"))
                            if target_prompt.get("modules_config") is not None
                            else None,
                            prompt_id,
                            f"Reverted to version {target_version}",
                            client_id or current_prompt.get("client_id") or self.client_id,
                        ),
                    )

                    new_prompt_id = cursor.lastrowid
                    conn.commit()

                    cursor.execute(
                        "SELECT * FROM prompt_studio_prompts WHERE id = ?",
                        (new_prompt_id,),
                    )
                    row = cursor.fetchone()
                    prompt = self._row_to_dict(cursor, row) if row else {}

                    self._log_sync_event(
                        "prompt_studio_prompt",
                        prompt.get("uuid", ""),
                        "version_revert",
                        {
                            "prompt_id": prompt_id,
                            "target_version": target_version,
                            "new_version": prompt.get("version_number"),
                        },
                    )
                    return prompt  # noqa: TRY300
                except sqlite3.OperationalError as exc:
                    if "database is locked" in str(exc).lower() and attempt < max_retries - 1:
                        should_retry = True
                        delay = base_delay * (2 ** attempt) * (0.5 + random.random())
                    else:
                        raise DatabaseError(f"Failed to revert prompt: {exc}") from exc  # noqa: TRY003
                except sqlite3.Error as exc:  # noqa: BLE001
                    raise DatabaseError(f"Failed to revert prompt: {exc}") from exc  # noqa: TRY003

            if should_retry:
                time.sleep(delay)

        raise DatabaseError("Failed to revert prompt due to database locks")  # noqa: TRY003

    # --- Optimization helpers -------------------------------------------------

    def get_optimization(
        self,
        optimization_id: int,
        *,
        include_deleted: bool = False,
    ) -> Optional[dict[str, Any]]:
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            clause = "" if include_deleted else " AND deleted = 0"
            cursor.execute(
                """
                SELECT *
                FROM prompt_studio_optimizations
                WHERE id = ?{clause}
                LIMIT 1
                """.format_map(locals()),  # nosec B608
                (optimization_id,),
            )
            row = cursor.fetchone()
            return self._row_to_dict(cursor, row) if row else None
        except sqlite3.Error as exc:  # noqa: BLE001
            raise DatabaseError(f"Failed to fetch optimization {optimization_id}: {exc}") from exc  # noqa: TRY003

    def update_optimization(
        self,
        optimization_id: int,
        updates: dict[str, Any],
        *,
        set_started_at: bool = False,
        set_completed_at: bool = False,
    ) -> dict[str, Any]:
        if not updates and not (set_started_at or set_completed_at):
            optimization = self.get_optimization(optimization_id, include_deleted=True)
            if optimization is None:
                raise InputError(f"Optimization {optimization_id} not found")  # noqa: TRY003
            return optimization

        json_fields = {
            "optimization_config",
            "initial_metrics",
            "final_metrics",
            "test_case_ids",
            "test_run_ids",
        }
        set_clauses: list[str] = []
        params: list[Any] = []

        for field, value in updates.items():
            if field in json_fields and value is not None:
                params.append(json.dumps(value))
            else:
                params.append(value)
            set_clauses.append(f"{field} = ?")

        if set_started_at:
            set_clauses.append("started_at = CURRENT_TIMESTAMP")
        if set_completed_at:
            set_clauses.append("completed_at = CURRENT_TIMESTAMP")

        params.append(optimization_id)
        sql = (
            "UPDATE prompt_studio_optimizations SET "  # nosec B608
            + ", ".join(set_clauses)
            + " WHERE id = ?"
        )

        try:
            with self._write_lock:
                conn = self.get_connection()
                cursor = conn.cursor()
                cursor.execute(sql, params)
                if cursor.rowcount == 0:
                    raise InputError(f"Optimization {optimization_id} not found")  # noqa: TRY003
                conn.commit()

                cursor.execute(
                    "SELECT * FROM prompt_studio_optimizations WHERE id = ?",
                    (optimization_id,),
                )
                row = cursor.fetchone()
                optimization = self._row_to_dict(cursor, row) if row else {}
        except sqlite3.Error as exc:  # noqa: BLE001
            raise DatabaseError(f"Failed to update optimization {optimization_id}: {exc}") from exc  # noqa: TRY003

        log_payload = {}
        for key, value in updates.items():
            if isinstance(value, (dict, list)):
                try:
                    log_payload[key] = json.loads(json.dumps(value, default=str))
                except TypeError:
                    log_payload[key] = str(value)
            else:
                log_payload[key] = value
        if set_started_at:
            log_payload["started_at"] = "CURRENT_TIMESTAMP"
        if set_completed_at:
            log_payload["completed_at"] = "CURRENT_TIMESTAMP"

        self._log_sync_event(
            "prompt_studio_optimization",
            optimization.get("uuid", ""),
            "update",
            log_payload,
        )
        return optimization

    def set_optimization_status(
        self,
        optimization_id: int,
        status: str,
        *,
        error_message: Optional[str] = None,
        mark_started: bool = False,
        mark_completed: bool = False,
    ) -> dict[str, Any]:
        updates: dict[str, Any] = {"status": status}
        if error_message is not None:
            updates["error_message"] = error_message
        return self.update_optimization(
            optimization_id,
            updates,
            set_started_at=mark_started,
            set_completed_at=mark_completed,
        )

    def complete_optimization(
        self,
        optimization_id: int,
        *,
        optimized_prompt_id: Optional[int] = None,
        iterations_completed: Optional[int] = None,
        initial_metrics: Optional[dict[str, Any]] = None,
        final_metrics: Optional[dict[str, Any]] = None,
        improvement_percentage: Optional[float] = None,
        total_tokens: Optional[int] = None,
        total_cost: Optional[float] = None,
    ) -> dict[str, Any]:
        updates: dict[str, Any] = {
            "status": "completed",
            "optimized_prompt_id": optimized_prompt_id,
            "iterations_completed": iterations_completed,
            "initial_metrics": initial_metrics,
            "final_metrics": final_metrics,
            "improvement_percentage": improvement_percentage,
            "total_tokens": total_tokens,
            "total_cost": total_cost,
        }
        updates = {k: v for k, v in updates.items() if v is not None}
        return self.update_optimization(
            optimization_id,
            updates,
            set_completed_at=True,
        )

    def record_optimization_iteration(
        self,
        optimization_id: int,
        *,
        iteration_number: int,
        prompt_variant: Optional[dict[str, Any]] = None,
        metrics: Optional[dict[str, Any]] = None,
        tokens_used: Optional[int] = None,
        cost: Optional[float] = None,
        note: Optional[str] = None,
    ) -> dict[str, Any]:
        payload = (
            str(uuid.uuid4()),
            optimization_id,
            iteration_number,
            json.dumps(prompt_variant) if prompt_variant is not None else None,
            json.dumps(metrics) if metrics is not None else None,
            tokens_used,
            cost,
            note,
        )

        try:
            with self._write_lock:
                conn = self.get_connection()
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO prompt_studio_optimization_iterations (
                        uuid, optimization_id, iteration_number, prompt_variant,
                        metrics, tokens_used, cost, note
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    payload,
                )
                iteration_id = cursor.lastrowid
                conn.commit()

                cursor.execute(
                    "SELECT * FROM prompt_studio_optimization_iterations WHERE id = ?",
                    (iteration_id,),
                )
                row = cursor.fetchone()
                record = self._row_to_dict(cursor, row) if row else {}
        except sqlite3.Error as exc:  # noqa: BLE001
            raise DatabaseError(f"Failed to record optimization iteration: {exc}") from exc  # noqa: TRY003

        self._log_sync_event(
            "prompt_studio_optimization_iteration",
            record.get("uuid", ""),
            "create",
            {
                "optimization_id": optimization_id,
                "iteration_number": iteration_number,
            },
        )
        return record

    def list_optimization_iterations(
        self,
        optimization_id: int,
        *,
        page: int = 1,
        per_page: int = 50,
    ) -> dict[str, Any]:
        """List persisted iterations for an optimization (SQLite backend)."""
        if page < 1:
            raise InputError("Page index must be >= 1")  # noqa: TRY003
        if per_page < 1:
            raise InputError("Items per page must be >= 1")  # noqa: TRY003

        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            cursor.execute(
                "SELECT COUNT(*) FROM prompt_studio_optimization_iterations WHERE optimization_id = ?",
                (optimization_id,),
            )
            row = cursor.fetchone()
            total = int(row[0]) if row and row[0] is not None else 0

            offset = max(page - 1, 0) * per_page
            cursor.execute(
                """
                SELECT *
                FROM prompt_studio_optimization_iterations
                WHERE optimization_id = ?
                ORDER BY iteration_number ASC, id ASC
                LIMIT ? OFFSET ?
                """,
                (optimization_id, per_page, offset),
            )
            rows = cursor.fetchall()
            iterations = [self._row_to_dict(cursor, r) for r in rows if r]

            return {
                "iterations": iterations,
                "pagination": {
                    "page": page,
                    "per_page": per_page,
                    "total": total,
                    "total_pages": (total + per_page - 1) // per_page if per_page else 0,
                },
            }
        except sqlite3.Error as exc:  # noqa: BLE001
            raise DatabaseError(f"Failed to list optimization iterations: {exc}") from exc  # noqa: TRY003

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

    def list_prompts(
        self,
        project_id: int,
        *,
        page: int = 1,
        per_page: int = 20,
        include_deleted: bool = False,
    ) -> dict[str, Any]:
        import sqlite3

        if page < 1:
            raise InputError("Page index must be >= 1")  # noqa: TRY003
        if per_page < 1:
            raise InputError("Items per page must be >= 1")  # noqa: TRY003

        try:
            conn = self.get_connection()
            cursor = conn.cursor()

            base_clause = "FROM prompt_studio_prompts WHERE project_id = ?"
            params: list[Any] = [project_id]
            if not include_deleted:
                base_clause += " AND deleted = 0"

            cursor.execute(f"SELECT COUNT(*) {base_clause}", params)
            total_row = cursor.fetchone()
            total = int(total_row[0]) if total_row and total_row[0] is not None else 0

            offset = (page - 1) * per_page
            list_query = (
                f"SELECT * {base_clause} "
                "ORDER BY updated_at DESC, version_number DESC LIMIT ? OFFSET ?"
            )
            cursor.execute(list_query, params + [per_page, offset])
            prompts = [self._row_to_dict(cursor, row) for row in cursor.fetchall()]

            return {
                "prompts": prompts,
                "pagination": {
                    "page": page,
                    "per_page": per_page,
                    "total": total,
                    "total_pages": (total + per_page - 1) // per_page if per_page else 0,
                },
            }
        except sqlite3.Error as exc:  # noqa: BLE001
            raise DatabaseError(f"Failed to list prompts: {exc}") from exc  # noqa: TRY003

    def list_prompt_versions(
        self,
        project_id: int,
        prompt_name: str,
        *,
        include_deleted: bool = False,
    ) -> list[dict[str, Any]]:
        import sqlite3

        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            deleted_clause = "" if include_deleted else "AND deleted = 0"
            cursor.execute(
                """
                SELECT id, uuid, version_number, name, change_description,
                       created_at, parent_version_id
                FROM prompt_studio_prompts
                WHERE project_id = ? AND name = ? {deleted_clause}
                ORDER BY version_number DESC
                """.format_map(locals()),  # nosec B608
                (project_id, prompt_name),
            )
            rows = cursor.fetchall()
            return [self._row_to_dict(cursor, row) for row in rows]
        except sqlite3.Error as exc:  # noqa: BLE001
            raise DatabaseError(  # noqa: TRY003
                f"Failed to list prompt versions for project {project_id}: {exc}"
            ) from exc

    def ensure_prompt_stub(
        self,
        *,
        prompt_id: int,
        project_id: int,
        name: Optional[str] = None,
        client_id: Optional[str] = None,
    ) -> None:
        import sqlite3

        if not prompt_id or not project_id:
            return

        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            cursor.execute(
                "SELECT 1 FROM prompt_studio_prompts WHERE id = ?",
                (prompt_id,),
            )
            if cursor.fetchone() is not None:
                return
        except sqlite3.Error as exc:  # noqa: BLE001
            raise DatabaseError(  # noqa: TRY003
                f"Failed to verify prompt {prompt_id} existence: {exc}"
            ) from exc

        stub_name = name or f"Auto-Created Prompt {prompt_id}"
        try:
            cursor.execute(
                """
                INSERT OR IGNORE INTO prompt_studio_prompts (
                    id, uuid, project_id, version_number, name, client_id
                ) VALUES (?, lower(hex(randomblob(16))), ?, 1, ?, ?)
                """,
                (
                    prompt_id,
                    project_id,
                    stub_name,
                    client_id or self.client_id,
                ),
            )
            conn.commit()
        except sqlite3.Error as exc:  # noqa: BLE001
            raise DatabaseError(  # noqa: TRY003
                f"Failed to create placeholder prompt {prompt_id}: {exc}"
            ) from exc

    ####################################################################################################################
    # Test Case Methods

    def create_test_case(
        self,
        project_id: int,
        name: str,
        *,
        inputs: dict[str, Any],
        description: Optional[str] = None,
        expected_outputs: Optional[dict[str, Any]] = None,
        actual_outputs: Optional[dict[str, Any]] = None,
        tags: Optional[Iterable[str]] = None,
        is_golden: bool = False,
        is_generated: bool = False,
        signature_id: Optional[int] = None,
        client_id: Optional[str] = None,
    ) -> dict[str, Any]:
        if not name or not name.strip():
            raise InputError("Test case name cannot be empty")  # noqa: TRY003

        import time

        conn = self.get_connection()
        cursor = conn.cursor()

        # Ensure uniqueness within project for active test cases
        try:
            cursor.execute(
                """
                SELECT COUNT(*) FROM prompt_studio_test_cases
                WHERE project_id = ? AND name = ? AND deleted = 0
                """,
                (project_id, name.strip()),
            )
            if cursor.fetchone()[0]:
                raise ConflictError(f"Test case with name '{name}' already exists")  # noqa: TRY003
        except sqlite3.Error as exc:  # noqa: BLE001
            raise DatabaseError(f"Failed to validate test case uniqueness: {exc}") from exc  # noqa: TRY003

        test_case_uuid = str(uuid.uuid4())
        tags_str = _serialise_tags(tags)
        payload = (
            test_case_uuid,
            project_id,
            signature_id,
            name.strip(),
            description,
            json.dumps(inputs),
            json.dumps(expected_outputs) if expected_outputs else None,
            json.dumps(actual_outputs) if actual_outputs else None,
            tags_str,
            bool(is_golden) if self.backend_type == BackendType.POSTGRESQL else int(bool(is_golden)),
            bool(is_generated) if self.backend_type == BackendType.POSTGRESQL else int(bool(is_generated)),
            client_id or self.client_id,
        )

        max_retries = 5
        base_delay = 0.05
        for attempt in range(max_retries):
            try:
                cursor.execute(
                    """
                    INSERT INTO prompt_studio_test_cases (
                        uuid, project_id, signature_id, name, description,
                        inputs, expected_outputs, actual_outputs, tags,
                        is_golden, is_generated, client_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    payload,
                )
                test_case_id = cursor.lastrowid
                conn.commit()
                created = self.get_test_case(test_case_id)
                if created:
                    logger.info(
                        'Created test case {} (ID: {}) for project {}',
                        name,
                        test_case_id,
                        project_id,
                    )
                    return created
                break
            except sqlite3.OperationalError as exc:
                if "database is locked" in str(exc).lower() and attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(
                        'create_test_case locked, retrying in {}s (attempt {}/{})',
                        delay,
                        attempt + 1,
                        max_retries,
                    )
                    time.sleep(delay)
                    continue
                raise DatabaseError(f"Failed to create test case: {exc}") from exc  # noqa: TRY003
            except sqlite3.IntegrityError as exc:  # noqa: BLE001
                raise ConflictError(f"Failed to create test case: {exc}") from exc  # noqa: TRY003
            except _PROMPT_STUDIO_NONCRITICAL_EXCEPTIONS as exc:  # noqa: BLE001
                raise DatabaseError(f"Failed to create test case: {exc}") from exc  # noqa: TRY003

        raise DatabaseError("Failed to create test case after multiple retries")  # noqa: TRY003

    def get_test_case(self, test_case_id: int, *, include_deleted: bool = False) -> Optional[dict[str, Any]]:
        conn = self.get_connection()
        cursor = conn.cursor()
        query = "SELECT * FROM prompt_studio_test_cases WHERE id = ?"
        params: list[Any] = [test_case_id]
        if not include_deleted:
            query += " AND deleted = 0"
        cursor.execute(query, params)
        row = cursor.fetchone()
        return self._format_test_case(cursor, row)

    def list_test_cases(
        self,
        project_id: int,
        *,
        signature_id: Optional[int] = None,
        is_golden: Optional[bool] = None,
        tags: Optional[list[str]] = None,
        search: Optional[str] = None,
        include_deleted: bool = False,
        page: int = 1,
        per_page: int = 20,
        return_pagination: bool = False,
    ) -> Union[dict[str, Any], list[dict[str, Any]]]:
        import time

        conn = self.get_connection()
        cursor = conn.cursor()

        conditions = ["project_id = ?"]
        params: list[Any] = [project_id]

        if not include_deleted:
            conditions.append("deleted = 0")

        if signature_id is not None:
            conditions.append("signature_id = ?")
            params.append(signature_id)

        if is_golden is not None:
            conditions.append("is_golden = ?")
            params.append(bool(is_golden) if self.backend_type == BackendType.POSTGRESQL else int(bool(is_golden)))

        if tags:
            tag_conditions = []
            for tag in tags:
                tag_conditions.append("tags LIKE ?")
                params.append(f"%{tag}%")
            if tag_conditions:
                conditions.append(f"({' OR '.join(tag_conditions)})")

        where_clause = " WHERE " + " AND ".join(conditions)

        search_clause = ""
        if search:
            search_clause = " AND (name LIKE ? OR description LIKE ?)"
            params.extend([f"%{search}%", f"%{search}%"])

        count_query = f"SELECT COUNT(*) FROM prompt_studio_test_cases{where_clause}{search_clause}"  # nosec B608
        cursor.execute(count_query, params)
        total = cursor.fetchone()[0]

        offset = (page - 1) * per_page
        list_query = """
            SELECT * FROM prompt_studio_test_cases
            {where_clause}{search_clause}
            ORDER BY is_golden DESC, created_at DESC
            LIMIT ? OFFSET ?
        """.format_map(locals())  # nosec B608

        # Retry loop mirroring historical behaviour for locked databases
        params_with_pagination = params + [per_page, offset]
        max_retries = 5
        base_delay = 0.05
        for attempt in range(max_retries):
            try:
                cursor.execute(list_query, params_with_pagination)
                break
            except sqlite3.OperationalError as exc:
                if "database is locked" in str(exc).lower() and attempt < max_retries - 1:
                    time.sleep(base_delay * (2 ** attempt))
                    continue
                raise DatabaseError(f"Failed to list test cases: {exc}") from exc  # noqa: TRY003

        records = [self._format_test_case(cursor, row) for row in cursor.fetchall() if row]

        if return_pagination:
            return {
                "test_cases": records,
                "pagination": {
                    "page": page,
                    "per_page": per_page,
                    "total": total,
                    "total_pages": (total + per_page - 1) // per_page if per_page else 0,
                },
            }
        return records

    def update_test_case(self, test_case_id: int, updates: dict[str, Any]) -> dict[str, Any]:
        conn = self.get_connection()
        cursor = conn.cursor()

        allowed_fields = {
            "name",
            "description",
            "inputs",
            "expected_outputs",
            "actual_outputs",
            "tags",
            "is_golden",
            "is_generated",
            "signature_id",
        }

        set_clauses: list[str] = []
        params: list[Any] = []

        for field, value in updates.items():
            if field not in allowed_fields:
                continue

            if field in {"inputs", "expected_outputs", "actual_outputs"} and value is not None:
                params.append(json.dumps(value))
            elif field in {"is_golden", "is_generated"} and value is not None:
                params.append(int(bool(value)))
            elif field == "tags":
                params.append(_serialise_tags(value))
            else:
                params.append(value)
            set_clauses.append(f"{field} = ?")

        if not set_clauses:
            existing = self.get_test_case(test_case_id)
            if existing is None:
                raise InputError(f"Test case {test_case_id} not found or already deleted")  # noqa: TRY003
            return existing

        set_clauses.append("updated_at = CURRENT_TIMESTAMP")
        params.append(test_case_id)
        set_clause_sql = ', '.join(set_clauses)

        try:
            cursor.execute(
                """
                UPDATE prompt_studio_test_cases
                SET {set_clause_sql}
                WHERE id = ? AND deleted = 0
                """.format_map(locals()),  # nosec B608
                params,
            )
            if cursor.rowcount == 0:
                raise InputError(f"Test case {test_case_id} not found or already deleted")  # noqa: TRY003
            conn.commit()
            updated = self.get_test_case(test_case_id)
            if updated is None:
                raise DatabaseError(f"Failed to fetch updated test case {test_case_id}")  # noqa: TRY003
            return updated  # noqa: TRY300
        except sqlite3.IntegrityError as exc:  # noqa: BLE001
            raise ConflictError(f"Failed to update test case: {exc}") from exc  # noqa: TRY003
        except sqlite3.Error as exc:  # noqa: BLE001
            raise DatabaseError(f"Failed to update test case: {exc}") from exc  # noqa: TRY003

    def delete_test_case(self, test_case_id: int, *, hard_delete: bool = False) -> bool:
        import time

        conn = self.get_connection()
        cursor = conn.cursor()
        max_retries = 5
        base_delay = 0.05

        for attempt in range(max_retries):
            try:
                if hard_delete:
                    cursor.execute("DELETE FROM prompt_studio_test_cases WHERE id = ?", (test_case_id,))
                else:
                    cursor.execute(
                        """
                        UPDATE prompt_studio_test_cases
                        SET deleted = 1, deleted_at = CURRENT_TIMESTAMP
                        WHERE id = ? AND deleted = 0
                        """,
                        (test_case_id,),
                    )
                if cursor.rowcount > 0:
                    conn.commit()
                    logger.info(
                        '{} deleted test case {}',
                        "Hard" if hard_delete else "Soft",
                        test_case_id,
                    )
                    return True
                return False  # noqa: TRY300
            except sqlite3.OperationalError as exc:
                if "database is locked" in str(exc).lower() and attempt < max_retries - 1:
                    time.sleep(base_delay * (2 ** attempt))
                    continue
                raise DatabaseError(f"Failed to delete test case {test_case_id}: {exc}") from exc  # noqa: TRY003
            except sqlite3.Error as exc:  # noqa: BLE001
                raise DatabaseError(f"Failed to delete test case {test_case_id}: {exc}") from exc  # noqa: TRY003

        return False

    def create_bulk_test_cases(
        self,
        project_id: int,
        test_cases: list[dict[str, Any]],
        *,
        signature_id: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        import time

        created: list[dict[str, Any]] = []
        with self.transaction() as conn:
            cursor = conn.cursor()
            max_retries = 5
            base_delay = 0.05

            for test_case in test_cases:
                test_case_uuid = str(uuid.uuid4())
                tags_str = _serialise_tags(test_case.get("tags"))
                params = (
                    test_case_uuid,
                    project_id,
                    signature_id or test_case.get("signature_id"),
                    test_case.get("name"),
                    test_case.get("description"),
                    json.dumps(test_case.get("inputs", {})),
                    json.dumps(test_case.get("expected_outputs")) if test_case.get("expected_outputs") else None,
                    json.dumps(test_case.get("actual_outputs")) if test_case.get("actual_outputs") else None,
                    tags_str,
                    int(bool(test_case.get("is_golden", False))),
                    int(bool(test_case.get("is_generated", False))),
                    self.client_id,
                )

                for attempt in range(max_retries):
                    try:
                        cursor.execute(
                            """
                            INSERT INTO prompt_studio_test_cases (
                                uuid, project_id, signature_id, name, description,
                                inputs, expected_outputs, actual_outputs, tags,
                                is_golden, is_generated, client_id
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            params,
                        )
                        new_id = cursor.lastrowid
                        created_case = self.get_test_case(new_id)
                        if created_case:
                            created.append(created_case)
                        break
                    except sqlite3.OperationalError as exc:
                        if "database is locked" in str(exc).lower() and attempt < max_retries - 1:
                            time.sleep(base_delay * (2 ** attempt))
                            continue
                        raise DatabaseError(f"Failed to create test case in bulk: {exc}") from exc  # noqa: TRY003

        logger.info("Created {} test cases in bulk for project {}", len(created), project_id)
        return created

    def search_test_cases(self, project_id: int, query: str, *, limit: int = 10) -> list[dict[str, Any]]:
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT tc.*
            FROM prompt_studio_test_cases tc
            JOIN prompt_studio_test_cases_fts ON tc.id = prompt_studio_test_cases_fts.rowid
            WHERE tc.project_id = ?
              AND tc.deleted = 0
              AND prompt_studio_test_cases_fts MATCH ?
            ORDER BY bm25(prompt_studio_test_cases_fts)
            LIMIT ?
            """,
            (project_id, query, limit),
        )
        return [self._format_test_case(cursor, row) for row in cursor.fetchall() if row]

    def get_test_cases_by_signature(self, signature_id: int) -> list[dict[str, Any]]:
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT *
            FROM prompt_studio_test_cases
            WHERE signature_id = ? AND deleted = 0
            ORDER BY is_golden DESC, created_at DESC
            """,
            (signature_id,),
        )
        return [self._format_test_case(cursor, row) for row in cursor.fetchall() if row]

    def get_test_case_stats(self, project_id: int) -> dict[str, Any]:
        conn = self.get_connection()
        cursor = conn.cursor()
        stats: dict[str, Any] = {}

        cursor.execute(
            "SELECT COUNT(*) FROM prompt_studio_test_cases WHERE project_id = ? AND deleted = 0",
            (project_id,),
        )
        stats["total"] = cursor.fetchone()[0]

        cursor.execute(
            """
            SELECT COUNT(*) FROM prompt_studio_test_cases
            WHERE project_id = ? AND deleted = 0 AND is_golden = 1
            """,
            (project_id,),
        )
        stats["golden"] = cursor.fetchone()[0]

        cursor.execute(
            """
            SELECT COUNT(*) FROM prompt_studio_test_cases
            WHERE project_id = ? AND deleted = 0 AND is_generated = 1
            """,
            (project_id,),
        )
        stats["generated"] = cursor.fetchone()[0]

        cursor.execute(
            """
            SELECT COUNT(*) FROM prompt_studio_test_cases
            WHERE project_id = ? AND deleted = 0 AND expected_outputs IS NOT NULL
            """,
            (project_id,),
        )
        stats["with_expected"] = cursor.fetchone()[0]

        cursor.execute(
            """
            SELECT signature_id, COUNT(*)
            FROM prompt_studio_test_cases
            WHERE project_id = ? AND deleted = 0 AND signature_id IS NOT NULL
            GROUP BY signature_id
            """,
            (project_id,),
        )
        stats["by_signature"] = {
            row[0]: row[1]
            for row in cursor.fetchall()
            if row and row[0] is not None
        }

        cursor.execute(
            """
            SELECT tags
            FROM prompt_studio_test_cases
            WHERE project_id = ? AND deleted = 0 AND tags IS NOT NULL
            """,
            (project_id,),
        )
        tag_counts: dict[str, int] = {}
        for row in cursor.fetchall():
            for tag in _parse_tags(row[0]):
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

        stats["top_tags"] = sorted(tag_counts.items(), key=lambda item: item[1], reverse=True)[:10]
        return stats

    def get_golden_test_cases(self, project_id: int, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT *
            FROM prompt_studio_test_cases
            WHERE project_id = ? AND is_golden = 1 AND deleted = 0
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
            """,
            (project_id, limit, offset),
        )
        return [self._format_test_case(cursor, row) for row in cursor.fetchall() if row]

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

    def create_optimization(
        self,
        *,
        project_id: int,
        name: Optional[str],
        initial_prompt_id: Optional[int],
        optimizer_type: str,
        optimization_config: Optional[dict[str, Any]] = None,
        max_iterations: Optional[int] = None,
        bootstrap_samples: Optional[int] = None,
        status: str = "pending",
        client_id: Optional[str] = None,
    ) -> dict[str, Any]:
        optimization_uuid = str(uuid.uuid4())
        payload = (
            optimization_uuid,
            project_id,
            name,
            initial_prompt_id,
            None,  # optimized_prompt_id
            optimizer_type,
            json.dumps(optimization_config) if optimization_config is not None else None,
            None,
            None,
            None,
            0,
            max_iterations,
            bootstrap_samples,
            status,
            None,
            None,
            None,
            client_id or self.client_id,
        )

        insert_sql = """
            INSERT INTO prompt_studio_optimizations (
                uuid, project_id, name, initial_prompt_id, optimized_prompt_id,
                optimizer_type, optimization_config, initial_metrics, final_metrics,
                improvement_percentage, iterations_completed, max_iterations,
                bootstrap_samples, status, error_message, total_tokens, total_cost,
                client_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            RETURNING *
        """

        try:
            with self._write_lock, self.transaction() as conn:
                cursor = self._cursor_exec(conn, insert_sql, payload)
                row = cursor.fetchone()
            return self._row_to_dict(cursor, row) if row else {}
        except _PROMPT_STUDIO_NONCRITICAL_EXCEPTIONS as exc:
            raise DatabaseError(f"Failed to create prompt studio optimization: {exc}") from exc  # noqa: TRY003


class PromptStudioDatabase:
    """Factory wrapper that selects SQLite or backend-aware implementations."""

    def __init__(
        self,
        db_path: Union[str, Path],
        client_id: str,
        *,
        backend: Optional[DatabaseBackend] = None,
        config: Optional[ConfigParser] = None,
    ) -> None:
        backend_type = backend.backend_type if backend else BackendType.SQLITE
        if backend_type == BackendType.POSTGRESQL and backend is not None:
            self._impl = _BackendPromptStudioDatabase(
                db_path,
                client_id,
                backend=backend,
                config=config,
            )
        else:
            self._impl = _SQLitePromptStudioDatabase(str(db_path), client_id)

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
        return self._impl.update_project(project_id, payload)

    # Signature delegation ------------------------------------------------

    def create_signature(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self._impl.create_signature(*args, **kwargs)

    def get_signature(self, *args: Any, **kwargs: Any) -> Optional[dict[str, Any]]:
        return self._impl.get_signature(*args, **kwargs)

    def list_signatures(self, *args: Any, **kwargs: Any) -> Union[dict[str, Any], list[dict[str, Any]]]:
        return self._impl.list_signatures(*args, **kwargs)

    def update_signature(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self._impl.update_signature(*args, **kwargs)

    def delete_signature(self, *args: Any, **kwargs: Any) -> bool:
        return self._impl.delete_signature(*args, **kwargs)

    def create_prompt(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self._impl.create_prompt(*args, **kwargs)

    def ensure_prompt_stub(self, *args: Any, **kwargs: Any) -> None:
        return self._impl.ensure_prompt_stub(*args, **kwargs)

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
        return self._impl.create_test_case(*args, **kwargs)

    def get_test_case(self, *args: Any, **kwargs: Any) -> Optional[dict[str, Any]]:
        return self._impl.get_test_case(*args, **kwargs)

    def list_test_cases(self, *args: Any, **kwargs: Any) -> Union[dict[str, Any], list[dict[str, Any]]]:
        return self._impl.list_test_cases(*args, **kwargs)

    def update_test_case(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self._impl.update_test_case(*args, **kwargs)

    def delete_test_case(self, *args: Any, **kwargs: Any) -> bool:
        return self._impl.delete_test_case(*args, **kwargs)

    def create_bulk_test_cases(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        return self._impl.create_bulk_test_cases(*args, **kwargs)

    def search_test_cases(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        return self._impl.search_test_cases(*args, **kwargs)

    def get_test_cases_by_signature(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        return self._impl.get_test_cases_by_signature(*args, **kwargs)

    def get_test_case_stats(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self._impl.get_test_case_stats(*args, **kwargs)

    def get_golden_test_cases(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        return self._impl.get_golden_test_cases(*args, **kwargs)

    # Test run delegation -------------------------------------------------

    def create_test_run(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self._impl.create_test_run(*args, **kwargs)

    def get_test_cases_by_ids(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        return self._impl.get_test_cases_by_ids(*args, **kwargs)

    # Evaluation delegation -----------------------------------------------

    def create_evaluation(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self._impl.create_evaluation(*args, **kwargs)

    def update_evaluation(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self._impl.update_evaluation(*args, **kwargs)

    def get_evaluation(self, *args: Any, **kwargs: Any) -> Optional[dict[str, Any]]:
        return self._impl.get_evaluation(*args, **kwargs)

    def list_evaluations(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self._impl.list_evaluations(*args, **kwargs)

    # Optimization delegation --------------------------------------------

    def create_optimization(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self._impl.create_optimization(*args, **kwargs)

    def get_optimization(self, *args: Any, **kwargs: Any) -> Optional[dict[str, Any]]:
        return self._impl.get_optimization(*args, **kwargs)

    def list_optimizations(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self._impl.list_optimizations(*args, **kwargs)

    def update_optimization(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self._impl.update_optimization(*args, **kwargs)

    def set_optimization_status(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self._impl.set_optimization_status(*args, **kwargs)

    def complete_optimization(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self._impl.complete_optimization(*args, **kwargs)

    def record_optimization_iteration(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self._impl.record_optimization_iteration(*args, **kwargs)

    def list_optimization_iterations(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return self._impl.list_optimization_iterations(*args, **kwargs)
