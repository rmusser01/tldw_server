# database.py
# Description: Database connection pooling and transaction management for user registration system
#
# Imports
from __future__ import annotations

import asyncio
import os
import re
import sqlite3
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import aiosqlite

#
# 3rd-party imports
import asyncpg
from fastapi import HTTPException
from loguru import logger

from tldw_Server_API.app.core.Audit.unified_audit_service import MandatoryAuditWriteError
from tldw_Server_API.app.core.AuthNZ.exceptions import (
    ConnectionPoolExhaustedError,
    DatabaseConcurrencyConflict,
    DatabaseError,
    DatabaseLockError,
    RollbackSignal,
    TransactionError,
    UserRegistrationException,
)
from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
from tldw_Server_API.app.core.AuthNZ.postgres_profile_version_schema import (
    ensure_postgres_profile_version_on_connection,
)
from tldw_Server_API.app.core.AuthNZ.profile_candidate_schema import (
    repair_postgres_profile_candidate_timestamps,
    validate_postgres_profile_candidate_schema,
)
from tldw_Server_API.app.core.AuthNZ.profile_user_write_guard import (
    ProfileUserWriteRejected,
    _execute_profile_users_bootstrap,
    _guard_sql,
)

#
# Local imports
from tldw_Server_API.app.core.AuthNZ.settings import Settings, get_settings
from tldw_Server_API.app.core.AuthNZ.sqlite_profile_version_schema import (
    validate_sqlite_profile_version_database,
)
from tldw_Server_API.app.core.DB_Management.sql_utils import split_sql_statements
from tldw_Server_API.app.core.DB_Management.sqlite_policy import (
    configure_sqlite_connection_async,
)
from tldw_Server_API.app.core.exceptions import TransactionPassthroughError
from tldw_Server_API.app.core.testing import is_explicit_pytest_runtime, is_test_mode

_AUTHNZ_DB_NONCRITICAL_EXCEPTIONS = (
    OSError,
    ValueError,
    TypeError,
    KeyError,
    RuntimeError,
    AttributeError,
    ConnectionError,
    TimeoutError,
    asyncpg.PostgresError,
)
_AUTHNZ_DB_INITIALIZATION_EXCEPTIONS = _AUTHNZ_DB_NONCRITICAL_EXCEPTIONS + (
    DatabaseError,
    sqlite3.Error,
)

_AUTHNZ_TRANSACTION_PASSTHROUGH_EXCEPTIONS = (
    RollbackSignal,
    UserRegistrationException,
    HTTPException,
    MandatoryAuditWriteError,
    TransactionPassthroughError,
)

_AUTHNZ_TRANSACTION_BOUNDARY_EXCEPTIONS = (DatabaseConcurrencyConflict,)

OPENAI_CREDENTIAL_LOCK_POOL_MAX_SIZE = 4
_POSTGRES_CONCURRENCY_SQLSTATES = frozenset({"40P01", "40001"})

SQLITE_REQUIRED_API_KEYS_COLUMNS = frozenset(
    {
        "id",
        "user_id",
        "key_hash",
        "key_id",
        "key_prefix",
        "name",
        "description",
        "scope",
        "status",
        "created_at",
        "expires_at",
        "last_used_at",
        "last_used_ip",
        "usage_count",
        "rate_limit",
        "allowed_ips",
        "metadata",
        "rotated_from",
        "rotated_to",
        "revoked_at",
        "revoked_by",
        "revoke_reason",
        "is_virtual",
        "parent_key_id",
        "org_id",
        "team_id",
        "llm_budget_day_tokens",
        "llm_budget_month_tokens",
        "llm_budget_day_usd",
        "llm_budget_month_usd",
        "llm_allowed_endpoints",
        "llm_allowed_providers",
        "llm_allowed_models",
    }
)


async def await_cancellation_safe_cleanup(awaitable: Any) -> Any:
    """Finish one cleanup task before propagating its first cancellation."""
    async def _capture_cleanup_outcome() -> tuple[bool, Any]:
        try:
            return True, await awaitable
        except BaseException as exc:  # noqa: BLE001 - return failure for safe consumption
            return False, exc

    cleanup_task = asyncio.create_task(_capture_cleanup_outcome())
    first_cancellation: asyncio.CancelledError | None = None

    while not cleanup_task.done():
        try:
            await asyncio.shield(cleanup_task)
        except asyncio.CancelledError as exc:
            if first_cancellation is None:
                first_cancellation = exc

    succeeded, result = cleanup_task.result()
    if not succeeded:
        cleanup_failure = result
        result = None
        if isinstance(cleanup_failure, asyncio.CancelledError):
            if first_cancellation is None:
                first_cancellation = cleanup_failure
        elif first_cancellation is None:
            raise cleanup_failure

    if first_cancellation is not None:
        raise first_cancellation from None
    return result


def _has_postgres_concurrency_sqlstate(exc: BaseException) -> bool:
    """Inspect Python's bounded effective chain for a retryable PostgreSQL state."""
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and len(seen) < 32:
        identity = id(current)
        if identity in seen:
            break
        seen.add(identity)
        for attribute in ("sqlstate", "pgcode"):
            try:
                sqlstate = getattr(current, attribute, None)
            except Exception:  # noqa: BLE001 - malformed backend exceptions are untrusted
                sqlstate = None
            if type(sqlstate) is str and sqlstate in _POSTGRES_CONCURRENCY_SQLSTATES:
                return True
        try:
            cause = current.__cause__
            context = current.__context__
            suppress_context = current.__suppress_context__
        except Exception:  # noqa: BLE001 - malformed backend exceptions are untrusted
            break
        if isinstance(cause, BaseException):
            current = cause
        elif not suppress_context and isinstance(context, BaseException):
            current = context
        else:
            current = None
    return False


def _is_transaction_control_exception(exc: BaseException) -> bool:
    """Return whether cleanup must preserve the original exception unchanged."""
    return not isinstance(exc, Exception) or (
        isinstance(exc, _AUTHNZ_TRANSACTION_PASSTHROUGH_EXCEPTIONS)
        and not isinstance(exc, _AUTHNZ_TRANSACTION_BOUNDARY_EXCEPTIONS)
    )


def select_transaction_cleanup_failure(
    primary: BaseException | None,
    cleanup: BaseException,
) -> tuple[BaseException, bool]:
    """Select the failure to propagate and whether ordinary cleanup should log."""
    if primary is not None and not isinstance(primary, Exception):
        return primary, False
    if not isinstance(cleanup, Exception):
        return cleanup, False
    if primary is None:
        return cleanup, False
    should_log = not _is_transaction_control_exception(
        primary
    ) and not _is_transaction_control_exception(cleanup)
    return primary, should_log


async def _release_postgres_connection(
    pool: Any,
    connection: Any,
    timeout: float | None,
    primary: BaseException | None,
) -> BaseException | None:
    """Finish one pool release and combine it with any active body failure."""
    try:
        release = (
            pool.release(connection)
            if timeout is None
            else pool.release(connection, timeout=timeout)
        )
        await await_cancellation_safe_cleanup(release)
    except BaseException as cleanup_exc:  # noqa: BLE001 - preserve control precedence
        if primary is None and isinstance(cleanup_exc, Exception):
            primary = TransactionError("PostgreSQL connection release")
            should_log = True
        else:
            primary, should_log = select_transaction_cleanup_failure(
                primary,
                cleanup_exc,
            )
        if should_log:
            logger.bind(
                backend="postgresql",
                operation="release",
                error_type=type(cleanup_exc).__name__,
            ).error("PostgreSQL connection release failed")
    return primary


async def _close_sqlite_connection(
    connection: Any,
    primary: BaseException | None,
) -> BaseException | None:
    """Finish one SQLite close and combine it with any active body failure."""
    try:
        await await_cancellation_safe_cleanup(connection.close())
    except BaseException as cleanup_exc:  # noqa: BLE001 - preserve control precedence
        if primary is None and isinstance(cleanup_exc, Exception):
            primary = TransactionError("SQLite connection close")
            should_log = True
        else:
            primary, should_log = select_transaction_cleanup_failure(
                primary,
                cleanup_exc,
            )
        if should_log:
            logger.bind(
                backend="sqlite",
                operation="close",
                error_type=type(cleanup_exc).__name__,
            ).error("SQLite connection close failed")
    return primary


SQLITE_REQUIRED_API_KEY_AUDIT_COLUMNS = frozenset(
    {
        "id",
        "api_key_id",
        "action",
        "user_id",
        "ip_address",
        "user_agent",
        "details",
        "created_at",
    }
)

#######################################################################################################################
#
# SQL Query Helpers


def build_sqlite_in_clause(values: list) -> tuple[str, tuple]:
    """
    Build a parameterized IN clause for SQLite queries.

    This helper function generates safe SQL placeholders for IN clauses,
    avoiding f-string interpolation patterns that could introduce SQL injection
    vulnerabilities if modified incorrectly.

    SECURITY NOTE: The returned placeholders contain ONLY '?' characters joined
    by commas. This function does NOT include any user-provided values in the
    SQL string itself - all values are returned as parameters to be bound safely
    by the database driver.

    Args:
        values: List of values to include in the IN clause

    Returns:
        Tuple of (placeholders_string, values_tuple) where:
        - placeholders_string: e.g., "?,?,?" for 3 values
        - values_tuple: tuple of the values for parameter binding

    Raises:
        ValueError: If values list is empty

    Example:
        >>> placeholders, params = build_sqlite_in_clause(['a', 'b', 'c'])
        >>> query = f"SELECT * FROM table WHERE col IN ({placeholders})"
        >>> # query = "SELECT * FROM table WHERE col IN (?,?,?)"
        >>> # params = ('a', 'b', 'c')
        >>> cursor.execute(query, params)
    """
    if not values:
        raise ValueError("Cannot build IN clause for empty values list")
    # Generate only '?' placeholders - never include actual values in SQL string
    placeholders = ",".join("?" for _ in values)
    return placeholders, tuple(values)


def build_postgres_in_clause(values: list, start_param: int = 1) -> tuple[str, list]:
    """
    Build a parameterized IN clause for PostgreSQL queries.

    This helper function generates safe SQL placeholders for IN clauses using
    PostgreSQL's $N placeholder syntax.

    SECURITY NOTE: The returned placeholders contain ONLY '$N' patterns.
    This function does NOT include any user-provided values in the SQL string
    itself - all values are returned as parameters to be bound safely by the
    database driver.

    Args:
        values: List of values to include in the IN clause
        start_param: Starting parameter number (default 1)

    Returns:
        Tuple of (placeholders_string, values_list) where:
        - placeholders_string: e.g., "$1,$2,$3" for 3 values starting at 1
        - values_list: list of the values for parameter binding

    Raises:
        ValueError: If values list is empty

    Example:
        >>> placeholders, params = build_postgres_in_clause(['a', 'b', 'c'])
        >>> query = f"SELECT * FROM table WHERE col IN ({placeholders})"
        >>> # query = "SELECT * FROM table WHERE col IN ($1,$2,$3)"
        >>> # params = ['a', 'b', 'c']
        >>> conn.fetch(query, *params)

        >>> # With offset for additional parameters
        >>> placeholders, params = build_postgres_in_clause(['x', 'y'], start_param=3)
        >>> # placeholders = "$3,$4", params = ['x', 'y']
    """
    if not values:
        raise ValueError("Cannot build IN clause for empty values list")
    # Generate only '$N' placeholders - never include actual values in SQL string
    placeholders = ",".join(f"${i}" for i in range(start_param, start_param + len(values)))
    return placeholders, list(values)


def _apply_single_user_fallback(url: str, auth_mode: str | None = None) -> str:
    """Apply single-user non-sqlite DATABASE_URL fallback to default SQLite path.

    When running in single-user mode and the provided URL uses a non-sqlite/file
    scheme, ignore it and return the default SQLite users DB path instead. This
    guards against leaking a Postgres DSN from tests/CI into local single-user
    runs.
    """
    try:
        if auth_mode is None:
            # Local import form retained for defensive use in non-pooled contexts.
            from tldw_Server_API.app.core.AuthNZ.settings import (  # type: ignore
                get_settings as _get_settings,
            )

            auth_mode_value = getattr(_get_settings(), "AUTH_MODE", "single_user")
        else:
            auth_mode_value = auth_mode
        mode = str(auth_mode_value).strip().lower()
    except _AUTHNZ_DB_NONCRITICAL_EXCEPTIONS:
        mode = "single_user"

    try:
        parsed = urlparse(url)
        scheme = (parsed.scheme or "").lower()
    except _AUTHNZ_DB_NONCRITICAL_EXCEPTIONS:
        scheme = ""

    if mode == "single_user" and scheme and not scheme.startswith("sqlite") and not scheme.startswith("file"):
        # Keep integration tests free to exercise Postgres-backed single-user bootstrap
        # behavior. Production/runtime safety still applies outside explicit tests.
        if is_test_mode() or is_explicit_pytest_runtime():
            return url
        with suppress(_AUTHNZ_DB_NONCRITICAL_EXCEPTIONS):
            logger.warning("Single-user mode: ignoring non-SQLite DATABASE_URL")
        return "sqlite:///./Databases/users.db"

    return url


def should_enforce_sqlite_schema_strictness(sqlite_fs_path: str | None) -> bool:
    """Return True when persisted SQLite schema drift should fail fast."""
    if not sqlite_fs_path or sqlite_fs_path == ":memory:":
        return False
    return not (is_test_mode() or is_explicit_pytest_runtime())


def _sqlite_missing_required_columns(
    conn: sqlite3.Connection,
    *,
    table_name: str,
    required_columns: frozenset[str],
) -> list[str]:
    # SECURITY: table_name must be a trusted, hardcoded value.
    # PRAGMA does not support parameterized queries.
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    present_columns = {row[1] for row in rows}
    return sorted(required_columns - present_columns)


def _sqlite_table_info_by_name(conn: sqlite3.Connection, table_name: str) -> dict[str, tuple]:
    # SECURITY: table_name must be a trusted, hardcoded value.
    # PRAGMA does not support parameterized queries.
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {row[1]: row for row in rows}


def validate_required_sqlite_api_key_schema(sqlite_fs_path: str | None) -> None:
    """Raise when persisted SQLite API-key schema drift is detected."""
    if not sqlite_fs_path or sqlite_fs_path == ":memory:":
        return

    with sqlite3.connect(sqlite_fs_path) as conn:
        api_key_table_info = _sqlite_table_info_by_name(conn, "api_keys")
        missing_api_key_cols = _sqlite_missing_required_columns(
            conn,
            table_name="api_keys",
            required_columns=SQLITE_REQUIRED_API_KEYS_COLUMNS,
        )
        if missing_api_key_cols:
            raise RuntimeError(
                "SQLite api_keys schema missing required columns: "
                + ", ".join(missing_api_key_cols)
            )

        scope_info = api_key_table_info.get("scope")
        if scope_info is None:
            raise RuntimeError(
                "SQLite api_keys schema missing 'scope' column"
            )
        scope_default = scope_info[4]
        if scope_default is not None:
            raise RuntimeError(
                "SQLite api_keys.scope must not define a default; "
                f"found {scope_default}"
            )

        missing_audit_cols = _sqlite_missing_required_columns(
            conn,
            table_name="api_key_audit_log",
            required_columns=SQLITE_REQUIRED_API_KEY_AUDIT_COLUMNS,
        )
        if missing_audit_cols:
            raise RuntimeError(
                "SQLite api_key_audit_log schema missing required columns: "
                + ", ".join(missing_audit_cols)
            )

#######################################################################################################################
#
# Database Pool Manager


class _GuardedAsyncpgConnection(asyncpg.Connection):
    """asyncpg connection class enforcing the managed AuthNZ users firewall."""

    _authnz_profile_user_backend = "postgres"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._authnz_profile_user_guard_identity = object()

    def _guard(self, query: Any, *, operation: str) -> str:
        return _guard_sql(
            query,
            backend="postgres",
            connection_identity=self._authnz_profile_user_guard_identity,
            operation=operation,
        )

    async def reset(self, *, timeout: float | None = None) -> None:
        """Reset pooled driver state without treating driver SQL as caller SQL."""
        async with asyncio.timeout(timeout):
            await asyncpg.Connection._reset(self)
            reset_query = asyncpg.Connection.get_reset_query(self)
            if reset_query:
                await asyncpg.Connection.execute(self, reset_query, timeout=timeout)

    async def execute(
        self,
        query: Any,
        *args: Any,
        timeout: float | None = None,
    ) -> str:
        return await super().execute(
            self._guard(query, operation="execute"),
            *args,
            timeout=timeout,
        )

    async def executemany(
        self,
        command: Any,
        args: Any,
        *,
        timeout: float | None = None,
    ) -> None:
        return await super().executemany(
            self._guard(command, operation="executemany"),
            args,
            timeout=timeout,
        )

    async def fetch(
        self,
        query: Any,
        *args: Any,
        timeout: float | None = None,
        record_class: type | None = None,
    ) -> list[Any]:
        return await super().fetch(
            self._guard(query, operation="fetch"),
            *args,
            timeout=timeout,
            record_class=record_class,
        )

    async def fetchrow(
        self,
        query: Any,
        *args: Any,
        timeout: float | None = None,
        record_class: type | None = None,
    ) -> Any:
        return await super().fetchrow(
            self._guard(query, operation="fetchrow"),
            *args,
            timeout=timeout,
            record_class=record_class,
        )

    async def fetchval(
        self,
        query: Any,
        *args: Any,
        column: int = 0,
        timeout: float | None = None,
    ) -> Any:
        return await super().fetchval(
            self._guard(query, operation="fetchval"),
            *args,
            column=column,
            timeout=timeout,
        )

    async def prepare(
        self,
        query: Any,
        *,
        name: str | None = None,
        timeout: float | None = None,
        record_class: type | None = None,
    ) -> Any:
        return await super().prepare(
            self._guard(query, operation="prepare"),
            name=name,
            timeout=timeout,
            record_class=record_class,
        )

    def cursor(
        self,
        query: Any,
        *args: Any,
        prefetch: int | None = None,
        timeout: float | None = None,
        record_class: type | None = None,
    ) -> Any:
        return super().cursor(
            self._guard(query, operation="cursor"),
            *args,
            prefetch=prefetch,
            timeout=timeout,
            record_class=record_class,
        )

    async def copy_from_query(self, query: Any, *args: Any, **kwargs: Any) -> str:
        return await super().copy_from_query(
            self._guard(query, operation="copy_from_query"),
            *args,
            **kwargs,
        )

    async def copy_to_table(
        self,
        table_name: str,
        *,
        schema_name: str | None = None,
        **kwargs: Any,
    ) -> str:
        _guard_postgres_copy_target(table_name)
        return await super().copy_to_table(
            table_name,
            schema_name=schema_name,
            **kwargs,
        )

    async def copy_records_to_table(
        self,
        table_name: str,
        *,
        schema_name: str | None = None,
        **kwargs: Any,
    ) -> str:
        _guard_postgres_copy_target(table_name)
        return await super().copy_records_to_table(
            table_name,
            schema_name=schema_name,
            **kwargs,
        )


def _guard_postgres_copy_target(table_name: Any) -> None:
    if type(table_name) is not str or not table_name.strip():
        raise ProfileUserWriteRejected()
    target = table_name.strip().rsplit(".", 1)[-1]
    if len(target) >= 2 and target[0] == target[-1] and target[0] in {'"', "'"}:
        target = target[1:-1]
    if target.lower() == "users":
        raise ProfileUserWriteRejected()


class _GuardedSQLiteCursor:
    """Caller-side cursor proxy that preserves the AuthNZ SQL firewall."""

    def __init__(self, cursor: Any, connection: _GuardedSQLiteConnection) -> None:
        self._cursor = cursor
        self._connection = connection

    @property
    def connection(self) -> _GuardedSQLiteConnection:
        return self._connection

    @property
    def rowcount(self) -> int:
        return self._cursor.rowcount

    @property
    def lastrowid(self) -> Any:
        return self._cursor.lastrowid

    @property
    def description(self) -> Any:
        return self._cursor.description

    @property
    def arraysize(self) -> int:
        return self._cursor.arraysize

    @arraysize.setter
    def arraysize(self, value: int) -> None:
        self._cursor.arraysize = value

    async def execute(self, query: Any, parameters: Any = ()) -> _GuardedSQLiteCursor:
        guarded = self._connection._guard(query, operation="execute")
        await self._cursor.execute(guarded, parameters)
        return self

    async def executemany(
        self,
        query: Any,
        parameters: Any,
    ) -> _GuardedSQLiteCursor:
        guarded = self._connection._guard(query, operation="executemany")
        await self._cursor.executemany(guarded, parameters)
        return self

    async def executescript(self, query: Any) -> _GuardedSQLiteCursor:
        guarded = self._connection._guard(query, operation="executescript")
        await self._cursor.executescript(guarded)
        return self

    async def fetchone(self) -> Any:
        return await self._cursor.fetchone()

    async def fetchmany(self, size: int | None = None) -> Any:
        if size is None:
            return await self._cursor.fetchmany()
        return await self._cursor.fetchmany(size)

    async def fetchall(self) -> Any:
        return await self._cursor.fetchall()

    async def close(self) -> None:
        await self._cursor.close()

    def __aiter__(self) -> _GuardedSQLiteCursor:
        return self

    async def __anext__(self) -> Any:
        return await self._cursor.__anext__()

    async def __aenter__(self) -> _GuardedSQLiteCursor:
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        await self.close()


class _GuardedSQLiteConnection:
    """Managed AuthNZ aiosqlite connection with no public raw SQL bypass."""

    _authnz_profile_user_backend = "sqlite"

    def __init__(self, connection: Any) -> None:
        self._connection = connection
        self._authnz_profile_user_guard_identity = object()

    def _guard(self, query: Any, *, operation: str) -> str:
        guarded = _guard_sql(
            query,
            backend="sqlite",
            connection_identity=self._authnz_profile_user_guard_identity,
            operation=operation,
        )
        return _normalize_sqlite_sql(guarded)

    async def execute(self, query: Any, *args: Any) -> _GuardedSQLiteCursor:
        guarded = self._guard(query, operation="execute")
        parameters = _sqlite_parameters(args)
        cursor = (
            await self._connection.execute(guarded)
            if parameters is None
            else await self._connection.execute(guarded, parameters)
        )
        return _GuardedSQLiteCursor(cursor, self)

    async def executemany(self, query: Any, parameters: Any) -> _GuardedSQLiteCursor:
        guarded = self._guard(query, operation="executemany")
        cursor = await self._connection.executemany(guarded, parameters)
        return _GuardedSQLiteCursor(cursor, self)

    async def executescript(self, query: Any) -> _GuardedSQLiteCursor:
        guarded = self._guard(query, operation="executescript")
        cursor = await self._connection.executescript(guarded)
        return _GuardedSQLiteCursor(cursor, self)

    async def execute_fetchall(self, query: Any, parameters: Any = ()) -> Any:
        guarded = self._guard(query, operation="execute_fetchall")
        return await self._connection.execute_fetchall(guarded, parameters)

    async def execute_insert(self, query: Any, parameters: Any = ()) -> Any:
        guarded = self._guard(query, operation="execute_insert")
        return await self._connection.execute_insert(guarded, parameters)

    async def cursor(self) -> _GuardedSQLiteCursor:
        return _GuardedSQLiteCursor(await self._connection.cursor(), self)

    async def commit(self) -> None:
        await self._connection.commit()

    async def rollback(self) -> None:
        await self._connection.rollback()

    async def close(self) -> None:
        await self._connection.close()

    @property
    def row_factory(self) -> Any:
        return self._connection.row_factory

    @row_factory.setter
    def row_factory(self, value: Any) -> None:
        self._connection.row_factory = value

    @property
    def in_transaction(self) -> bool:
        return self._connection.in_transaction

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[_GuardedSQLiteConnection]:
        """Run a short atomic unit on this managed connection."""
        if self.in_transaction:
            yield self
            return

        await self._connection.execute("BEGIN IMMEDIATE")
        failure: BaseException | None = None
        try:
            yield self
        except BaseException as exc:  # noqa: BLE001 - rollback must cover cancellation
            failure = exc

        if failure is None:
            try:
                await await_cancellation_safe_cleanup(self._connection.commit())
            except BaseException as exc:  # noqa: BLE001 - rollback after commit failure
                failure = exc

        if failure is not None and self.in_transaction:
            try:
                await await_cancellation_safe_cleanup(self._connection.rollback())
            except BaseException as cleanup_exc:  # noqa: BLE001 - preserve failure order
                failure, should_log = select_transaction_cleanup_failure(
                    failure,
                    cleanup_exc,
                )
                if should_log:
                    logger.bind(
                        backend="sqlite",
                        operation="rollback",
                        error_type=type(cleanup_exc).__name__,
                    ).error("SQLite managed-connection rollback failed")

        if failure is not None:
            raise failure

    @property
    def total_changes(self) -> int:
        return self._connection.total_changes


def _sqlite_parameters(args: tuple[Any, ...]) -> Any | None:
    if not args:
        return None
    parameters = (
        args[0]
        if len(args) == 1 and isinstance(args[0], (list, tuple, dict))
        else args
    )
    return parameters if isinstance(parameters, dict) else tuple(parameters)

class DatabasePool:
    """Database connection pool manager supporting both PostgreSQL and SQLite"""

    def __init__(self, settings: Settings | None = None):
        """Initialize database pool manager"""
        self.settings = settings or get_settings()
        self.pool: asyncpg.Pool | None = None
        self._openai_credential_lock_pool: asyncpg.Pool | None = None
        self.db_path: str | None = None
        self._sqlite_fs_path: str | None = None
        self._sqlite_uri: bool = False
        self._initialized = False
        self._lock = asyncio.Lock()
        # Track the event loop this pool is attached to (Postgres only)
        self._loop: asyncio.AbstractEventLoop | None = None

    async def initialize(self):
        """Initialize database connection pool"""
        if self._initialized:
            return

        async with self._lock:
            if self._initialized:
                return

            try:
                if self._should_use_postgres():
                    # PostgreSQL with connection pooling
                    logger.info("Initializing PostgreSQL connection pool...")

                    self.pool = await asyncpg.create_pool(
                        self.settings.DATABASE_URL,
                        min_size=self.settings.DATABASE_POOL_MIN_SIZE,
                        max_size=self.settings.DATABASE_POOL_MAX_SIZE,
                        max_queries=self.settings.DATABASE_MAX_QUERIES,
                        max_inactive_connection_lifetime=self.settings.DATABASE_MAX_INACTIVE_CONNECTION_LIFETIME,
                        command_timeout=60,
                        connection_class=_GuardedAsyncpgConnection,
                    )
                    self._openai_credential_lock_pool = await asyncpg.create_pool(
                        self.settings.DATABASE_URL,
                        min_size=0,
                        max_size=OPENAI_CREDENTIAL_LOCK_POOL_MAX_SIZE,
                        max_queries=self.settings.DATABASE_MAX_QUERIES,
                        max_inactive_connection_lifetime=self.settings.DATABASE_MAX_INACTIVE_CONNECTION_LIFETIME,
                        command_timeout=60,
                        connection_class=_GuardedAsyncpgConnection,
                    )
                    # Remember loop for compatibility checks
                    try:
                        self._loop = asyncio.get_running_loop()
                    except RuntimeError:
                        # Fallback for contexts without a running loop
                        self._loop = None

                    # Test connection
                    async with self.pool.acquire() as conn:
                        version = await conn.fetchval("SELECT version()")
                        logger.info(f"PostgreSQL connected: {version[:50]}...")

                    # Create schema if needed
                    await self._create_postgresql_schema()

                else:
                    # SQLite for single-user mode or fallback
                    # Defensive hardening: if AUTH_MODE is single_user but DATABASE_URL
                    # is a non-sqlite scheme (e.g., a Postgres DSN leaked from CI),
                    # ignore it and fall back to the default SQLite users DB.
                    _raw_url = _apply_single_user_fallback(
                        self.settings.DATABASE_URL,
                        auth_mode=getattr(self.settings, "AUTH_MODE", "single_user"),
                    )

                    self.db_path, self._sqlite_uri, self._sqlite_fs_path = self._resolve_sqlite_paths(_raw_url)

                    # Ensure directory exists
                    if self._sqlite_fs_path and self._sqlite_fs_path != ":memory:":
                        db_dir = Path(self._sqlite_fs_path).parent
                        db_dir.mkdir(parents=True, exist_ok=True)

                    logger.bind(
                        database_kind=(
                            "memory" if self.db_path == ":memory:" else "file"
                        )
                    ).info("Using SQLite database")

                    # Initialize SQLite schema
                    await self._create_sqlite_schema()

                self._initialized = True
                logger.info("Database pool initialized successfully")

            except asyncio.CancelledError:
                await self._close_postgres_pools()
                raise
            except _AUTHNZ_DB_INITIALIZATION_EXCEPTIONS as e:
                cancelled = await self._close_postgres_pools()
                if cancelled:
                    raise asyncio.CancelledError from None
                logger.bind(
                    operation="database_pool_initialize",
                    exception_type=type(e).__name__,
                ).error("Failed to initialize database pool")
                if "profile_version" in str(e).lower():
                    raise DatabaseError(
                        "AuthNZ users.profile_version readiness validation failed"
                    ) from None
                raise DatabaseError("Database initialization failed") from None

    def _should_use_postgres(self) -> bool:
        """Return True if the configured DATABASE_URL resolves to PostgreSQL.

        In production, PostgreSQL is only used when AUTH_MODE is ``multi_user``.
        For test contexts (``TEST_MODE=1``) we allow exercising single-user
        bootstrap and RBAC seed paths against a Postgres backend when a
        Postgres DSN is configured. This keeps local single-user deployments
        safely on SQLite by default while enabling the User-Unification
        Postgres test coverage.
        """
        parsed = urlparse(self.settings.DATABASE_URL)
        scheme = (parsed.scheme or "").lower()
        if not scheme or not scheme.startswith("postgres"):
            return False

        mode = getattr(self.settings, "AUTH_MODE", "multi_user")
        if mode == "multi_user":
            return True

        # Allow Postgres in single-user mode only in explicit test contexts,
        # so production single-user profiles continue to fall back to SQLite.
        try:
            test_mode = is_test_mode()
        except _AUTHNZ_DB_NONCRITICAL_EXCEPTIONS:
            test_mode = False
        # Also allow Postgres when running under pytest even if TEST_MODE is not set,
        # to keep Postgres-backed tests deterministic without requiring extra env wiring.
        try:
            import sys as _sys  # local import to avoid module-level side effects

            pytest_active = bool(os.getenv("PYTEST_CURRENT_TEST")) or ("pytest" in _sys.modules)
        except _AUTHNZ_DB_NONCRITICAL_EXCEPTIONS:
            pytest_active = False
        return test_mode or pytest_active

    @property
    def backend_type(self) -> str:
        """Return the configured AuthNZ database backend without exposing internals."""
        if self.pool is not None:
            return "postgres"
        if self._initialized:
            return "sqlite"
        return "postgres" if self._should_use_postgres() else "sqlite"

    @staticmethod
    def _resolve_sqlite_paths(url: str) -> tuple[str, bool, str | None]:
        """Resolve sqlite connection string, uri flag, and filesystem path.

        The provided URL is assumed to have already passed through any
        single-user fallback handling; this helper focuses purely on parsing
        SQLite and file-style URLs into a usable connection string and
        filesystem path.
        """
        parsed = urlparse(url)
        scheme = (parsed.scheme or "").lower()
        if scheme.startswith("file"):
            fs_path = parsed.path or ""
            if fs_path.startswith("//"):
                fs_path = fs_path[1:]
            fs_path = unquote(fs_path or "")
            if re.match(r"^/[A-Za-z]:[\\/]", fs_path):
                fs_path = fs_path[1:]
            return url, True, fs_path or None

        if not scheme.startswith("sqlite"):
            # Fallback: treat entire string as path
            return url, False, url

        path_part = parsed.path or ""
        netloc = parsed.netloc or ""
        combined = f"{netloc}{path_part}" if netloc else path_part
        combined = unquote(combined or "")

        if combined in (":memory:", "/:memory:"):
            filesystem_path = ":memory:"
        else:
            if path_part.startswith("//") or netloc:
                filesystem_path = "/" + combined.lstrip("/")
            elif combined.startswith("/"):
                filesystem_path = combined.lstrip("/")
            else:
                filesystem_path = combined

        if filesystem_path.startswith("///"):
            filesystem_path = filesystem_path.lstrip("/")

        if parsed.query:
            if filesystem_path.startswith("/") or filesystem_path:
                uri = f"file:{filesystem_path}?{parsed.query}"
            else:
                uri = f"file:?{parsed.query}"
            return uri, True, filesystem_path or None

        return filesystem_path, False, filesystem_path or None

    async def _create_postgresql_schema(self):
        """Create PostgreSQL schema if it doesn't exist"""
        schema_file = Path(__file__).parent.parent.parent.parent / "Databases" / "Postgres" / "Schema" / "postgresql_users.sql"

        schema_available = schema_file.exists()
        if not schema_available:
            # This path is expected in current builds: schema is provisioned by initialize.py/migrations.
            logger.warning(
                "PostgreSQL schema file not found at {}. Run 'python -m tldw_Server_API.app.core.AuthNZ.initialize' or apply DB migrations to create schema.",
                schema_file,
            )

        try:
            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    exists = await conn.fetchval(
                        """
                        SELECT EXISTS (
                            SELECT 1
                            FROM information_schema.tables
                            WHERE table_schema = 'public'
                              AND table_name = 'users'
                        )
                        """
                    )

                    if schema_available:
                        logger.info("Ensuring PostgreSQL schema...")
                        schema_sql = schema_file.read_text()
                        for statement in split_sql_statements(schema_sql):
                            if "CREATE TABLE IF NOT EXISTS public.users" in statement:
                                await _execute_profile_users_bootstrap(
                                    conn,
                                    statement,
                                    backend="postgres",
                                )
                            else:
                                await conn.execute(statement)
                        logger.info("PostgreSQL schema ensured successfully")
                    elif not exists:
                        raise RuntimeError(
                            "PostgreSQL AuthNZ bootstrap schema is unavailable"
                        )

                    await ensure_postgres_profile_version_on_connection(conn)
                    await repair_postgres_profile_candidate_timestamps(conn)
                    await validate_postgres_profile_candidate_schema(conn)

        except Exception as exc:  # noqa: BLE001 - sanitize bootstrap failures
            logger.bind(
                backend="postgres",
                operation="authnz_schema_readiness",
                exception_type=type(exc).__name__,
            ).error("PostgreSQL AuthNZ schema readiness failed")
            raise DatabaseError(
                "PostgreSQL AuthNZ schema readiness failed"
            ) from None

    async def _create_sqlite_schema(self):
        """Create SQLite schema if it doesn't exist"""
        schema_file = Path(__file__).parent.parent.parent.parent / "Databases" / "SQLite" / "Schema" / "sqlite_users.sql"

        schema_available = schema_file.exists()
        if not schema_available:
            logger.warning("Packaged SQLite schema file not found")

        try:
            async with aiosqlite.connect(self.db_path, uri=self._sqlite_uri) as conn:
                await configure_sqlite_connection_async(conn)

                # Check if users table exists
                cursor = await conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='users'"
                )
                exists = await cursor.fetchone()

                if not exists and schema_available:
                    logger.info("Creating SQLite schema...")
                    schema_sql = schema_file.read_text()
                    await conn.executescript(schema_sql)
                    await conn.commit()
                    logger.info("SQLite schema created successfully")
                else:
                    logger.debug("SQLite schema already exists")

            # Ensure AuthNZ migrations are up to date (handles legacy columns)
            try:
                if self._sqlite_fs_path and self._sqlite_fs_path != ":memory:":
                    logger.info("SQLite schema harmonization: ensuring AuthNZ tables")
                    await asyncio.to_thread(ensure_authnz_tables, Path(self._sqlite_fs_path))
                    if should_enforce_sqlite_schema_strictness(self._sqlite_fs_path):
                        await asyncio.to_thread(
                            validate_required_sqlite_api_key_schema,
                            self._sqlite_fs_path,
                        )
            except _AUTHNZ_DB_NONCRITICAL_EXCEPTIONS as migration_error:
                if should_enforce_sqlite_schema_strictness(self._sqlite_fs_path):
                    raise
                logger.bind(
                    operation="sqlite_schema_harmonization",
                    exception_type=type(migration_error).__name__,
                ).debug("SQLite migration harmonization skipped")

        except _AUTHNZ_DB_NONCRITICAL_EXCEPTIONS as e:
            logger.bind(
                operation="sqlite_schema_readiness",
                exception_type=type(e).__name__,
            ).error("Failed to create SQLite schema")
            if should_enforce_sqlite_schema_strictness(self._sqlite_fs_path):
                raise
            # Don't raise in permissive contexts - schema might already exist

        if self._sqlite_fs_path and self._sqlite_fs_path != ":memory:":
            await asyncio.to_thread(
                validate_sqlite_profile_version_database,
                self._sqlite_fs_path,
            )

    @asynccontextmanager
    async def transaction(
        self,
        *,
        acquire_timeout_seconds: float | None = None,
    ):
        """Open the configured backend's transaction context."""
        if not self._initialized:
            await self.initialize()
        async with self._transaction_context(acquire_timeout_seconds) as conn:
            yield conn

    @asynccontextmanager
    async def _transaction_context(self, acquire_timeout_seconds: float | None):
        """Implement backend-specific transaction lifecycle and translation."""
        postgres_pool = self.pool
        if postgres_pool is not None:
            primary_failure: BaseException | None = None
            failure_operation = "acquire"
            acquire_context = None
            transaction_context = None
            conn = None
            acquired = False
            try:
                acquire_context = (
                    postgres_pool.acquire()
                    if acquire_timeout_seconds is None
                    else postgres_pool.acquire(timeout=acquire_timeout_seconds)
                )
                conn = await acquire_context.__aenter__()
                acquired = True
            except BaseException as exc:  # noqa: BLE001 - classify after ordered cleanup
                primary_failure = exc

            transaction_entered = False
            if acquired and primary_failure is None:
                try:
                    transaction_context = conn.transaction()
                    await transaction_context.__aenter__()
                    transaction_entered = True
                except BaseException as exc:  # noqa: BLE001 - release must still run
                    primary_failure = exc
                    failure_operation = "transaction_enter"

            if transaction_entered:
                try:
                    yield conn
                except BaseException as exc:  # noqa: BLE001 - preserve body control flow
                    primary_failure = exc
                    failure_operation = "transaction"

                try:
                    if primary_failure is None:
                        await await_cancellation_safe_cleanup(
                            transaction_context.__aexit__(None, None, None)
                        )
                    else:
                        await await_cancellation_safe_cleanup(
                            transaction_context.__aexit__(
                                type(primary_failure),
                                primary_failure,
                                primary_failure.__traceback__,
                            )
                        )
                except BaseException as cleanup_exc:  # noqa: BLE001 - ordered cleanup selection
                    cleanup_operation = (
                        "commit" if primary_failure is None else "rollback"
                    )
                    selected_failure, should_log = select_transaction_cleanup_failure(
                        primary_failure,
                        cleanup_exc,
                    )
                    if selected_failure is cleanup_exc:
                        failure_operation = cleanup_operation
                    primary_failure = selected_failure
                    if should_log:
                        logger.bind(
                            backend="postgresql",
                            operation=cleanup_operation,
                            error_type=type(cleanup_exc).__name__,
                        ).error("PostgreSQL transaction cleanup failed")

            if acquired and acquire_context is not None:
                try:
                    if primary_failure is None:
                        await await_cancellation_safe_cleanup(
                            acquire_context.__aexit__(None, None, None)
                        )
                    else:
                        await await_cancellation_safe_cleanup(
                            acquire_context.__aexit__(
                                type(primary_failure),
                                primary_failure,
                                primary_failure.__traceback__,
                            )
                        )
                except BaseException as cleanup_exc:  # noqa: BLE001 - ordered cleanup selection
                    selected_failure, should_log = select_transaction_cleanup_failure(
                        primary_failure,
                        cleanup_exc,
                    )
                    if selected_failure is cleanup_exc:
                        failure_operation = "release"
                    primary_failure = selected_failure
                    if should_log:
                        logger.bind(
                            backend="postgresql",
                            operation="release",
                            error_type=type(cleanup_exc).__name__,
                        ).error("PostgreSQL transaction cleanup failed")

            if primary_failure is None:
                logger.debug("PostgreSQL transaction committed successfully")
                return
            if _is_transaction_control_exception(primary_failure):
                raise primary_failure from None
            if not acquired and isinstance(
                primary_failure,
                (asyncpg.exceptions.TooManyConnectionsError, TimeoutError),
            ):
                raise ConnectionPoolExhaustedError() from None
            if failure_operation in {"transaction", "commit"} and (
                _has_postgres_concurrency_sqlstate(primary_failure)
            ):
                raise DatabaseConcurrencyConflict() from None
            if isinstance(primary_failure, Exception):
                logger.bind(
                    backend="postgresql",
                    operation=failure_operation,
                    error_type=type(primary_failure).__name__,
                ).error("PostgreSQL transaction failed")
                raise TransactionError("PostgreSQL transaction") from None
            raise primary_failure from None

        conn = None
        failure: BaseException | None = None
        failure_operation = "transaction"
        transaction_started = False
        try:
            conn = await aiosqlite.connect(self.db_path, uri=self._sqlite_uri)
            await configure_sqlite_connection_async(conn)
            conn.row_factory = aiosqlite.Row
            await conn.execute("BEGIN IMMEDIATE")
            transaction_started = True

            yield _GuardedSQLiteConnection(conn)
            await conn.commit()
        except BaseException as exc:  # noqa: BLE001 - cleanup must cover cancellation/control flow
            failure = exc
            if conn is not None and transaction_started:
                try:
                    await await_cancellation_safe_cleanup(conn.rollback())
                except BaseException as cleanup_exc:  # noqa: BLE001 - ordered cleanup selection
                    selected_failure, should_log = select_transaction_cleanup_failure(
                        failure,
                        cleanup_exc,
                    )
                    if selected_failure is cleanup_exc:
                        failure_operation = "rollback"
                    failure = selected_failure
                    if should_log:
                        logger.bind(
                            backend="sqlite",
                            operation="rollback",
                            error_type=type(cleanup_exc).__name__,
                        ).error("SQLite transaction cleanup failed")
        finally:
            if conn is not None:
                try:
                    await await_cancellation_safe_cleanup(conn.close())
                except BaseException as cleanup_exc:  # noqa: BLE001 - ordered cleanup selection
                    selected_failure, should_log = select_transaction_cleanup_failure(
                        failure,
                        cleanup_exc,
                    )
                    if selected_failure is cleanup_exc:
                        failure_operation = "close"
                    failure = selected_failure
                    if should_log:
                        logger.bind(
                            backend="sqlite",
                            operation="close",
                            error_type=type(cleanup_exc).__name__,
                        ).error("SQLite transaction cleanup failed")

        if failure is None:
            return
        if _is_transaction_control_exception(failure):
            raise failure from None
        if (
            failure_operation == "transaction"
            and isinstance(failure, aiosqlite.OperationalError)
            and "database is locked" in str(failure).lower()
        ):
            raise DatabaseLockError() from None
        if isinstance(failure, Exception):
            logger.bind(
                backend="sqlite",
                operation=failure_operation,
                error_type=type(failure).__name__,
            ).error("SQLite transaction failed")
            raise TransactionError("SQLite transaction") from None
        raise failure from None

    @asynccontextmanager
    async def acquire(self, *, timeout: float | None = None):
        """Acquire a connection, optionally bounding PostgreSQL pool wait time."""
        if not self._initialized:
            await self.initialize()

        postgres_pool = self.pool
        if postgres_pool is not None:
            # PostgreSQL connection
            try:
                if timeout is None:
                    conn = await postgres_pool.acquire()
                else:
                    conn = await postgres_pool.acquire(timeout=timeout)
            except asyncpg.exceptions.TooManyConnectionsError:
                raise ConnectionPoolExhaustedError() from None

            failure: BaseException | None = None
            try:
                yield conn
            except BaseException as exc:  # noqa: BLE001 - release before propagation
                failure = exc

            had_primary_failure = failure is not None
            failure = await _release_postgres_connection(
                postgres_pool,
                conn,
                timeout,
                failure,
            )
            if failure is not None:
                if not had_primary_failure and isinstance(failure, TransactionError):
                    raise failure from None
                if isinstance(failure, Exception):
                    raise failure
                raise failure from None
        else:
            # SQLite connection
            conn = await aiosqlite.connect(self.db_path, uri=self._sqlite_uri)

            failure: BaseException | None = None
            try:
                await configure_sqlite_connection_async(conn)
                conn.row_factory = aiosqlite.Row
                yield _GuardedSQLiteConnection(conn)
            except BaseException as exc:  # noqa: BLE001 - close before propagation
                failure = exc

            had_primary_failure = failure is not None
            failure = await _close_sqlite_connection(conn, failure)
            if failure is not None:
                if not had_primary_failure and isinstance(failure, TransactionError):
                    raise failure from None
                if isinstance(failure, Exception):
                    raise failure
                raise failure from None

    @asynccontextmanager
    async def acquire_statement_autocommit(self, *, timeout: float | None = None):
        """Acquire a connection whose standalone writes commit per statement.

        Asyncpg already runs statements outside an explicit transaction in
        autocommit mode. SQLite needs ``isolation_level=None`` so login-time
        password and timestamp writes cannot retain a lock while session and
        audit services use their own connections.
        """
        if not self._initialized:
            await self.initialize()

        if self.pool is not None:
            async with self.acquire(timeout=timeout) as conn:
                yield conn
            return

        conn = None
        try:
            conn = await aiosqlite.connect(
                self.db_path,
                uri=self._sqlite_uri,
                isolation_level=None,
            )
            await configure_sqlite_connection_async(conn)
            conn.row_factory = aiosqlite.Row
            yield _GuardedSQLiteConnection(conn)
        finally:
            if conn:
                await conn.close()

    async def execute(self, query: str, *args) -> Any:
        """Execute a query without returning results"""
        async with self.acquire() as conn:
            is_postgres = self.pool is not None
            if is_postgres:
                # PostgreSQL
                params = _flatten_params(args)
                pg_query = _convert_question_mark_to_dollar(query, params)
                return await conn.execute(pg_query, *params)
            else:
                # SQLite
                # Flatten args if a single list/tuple was provided by an adapter
                params = args[0] if (len(args) == 1 and isinstance(args[0], (list, tuple))) else args
                q = _normalize_sqlite_sql(query)
                cursor = await conn.execute(q, tuple(params))
                await conn.commit()
                return cursor

    async def fetchone(self, query: str, *args) -> dict[str, Any] | None:
        """Fetch a single row"""
        async with self.acquire() as conn:
            is_postgres = self.pool is not None
            if is_postgres:
                # PostgreSQL
                params = _flatten_params(args)
                pg_query = _convert_question_mark_to_dollar(query, params)
                row = await conn.fetchrow(pg_query, *params)
                return dict(row) if row else None
            else:
                # SQLite
                params = args[0] if (len(args) == 1 and isinstance(args[0], (list, tuple))) else args
                q = _normalize_sqlite_sql(query)
                cursor = await conn.execute(q, tuple(params))
                row = await cursor.fetchone()
                if row:
                    # Convert sqlite row objects defensively across row_factory variants.
                    if isinstance(row, dict):
                        return row
                    with suppress(Exception):
                        # Row iteration yields values; keys() is required here.
                        return {
                            key: row[key] for key in row.keys()  # noqa: SIM118
                        }
                    with suppress(Exception):
                        return dict(row)
                    values = tuple(row)
                    with suppress(Exception):
                        if values and all(
                            isinstance(item, (tuple, list)) and len(item) == 2 for item in values
                        ):
                            return {str(item[0]): item[1] for item in values}
                    return {str(index): value for index, value in enumerate(values)}
                return None

    # Compatibility aliases for callers expecting asyncpg-like API
    async def fetchrow(self, query: str, *args) -> dict[str, Any] | None:
        """Alias for fetchone to match asyncpg-style interfaces."""
        return await self.fetchone(query, *args)

    async def fetchall(self, query: str, *args) -> list[Any]:
        """Fetch all rows.

        PostgreSQL returns a list of dict-like records (converted via dict(row)).
        SQLite returns aiosqlite.Row objects (supporting both dict-style and index access)
        to maximize compatibility with tests that may use numeric indexing (r[0])
        or key access (r['col']).
        """
        async with self.acquire() as conn:
            is_postgres = self.pool is not None
            if is_postgres:
                # PostgreSQL
                params = _flatten_params(args)
                pg_query = _convert_question_mark_to_dollar(query, params)
                rows = await conn.fetch(pg_query, *params)
                return [dict(row) for row in rows]
            else:
                # SQLite
                params = args[0] if (len(args) == 1 and isinstance(args[0], (list, tuple))) else args
                q = _normalize_sqlite_sql(query)
                cursor = await conn.execute(q, tuple(params))
                rows = await cursor.fetchall()
                # Return native Row objects to support both index and key access
                return list(rows)

    async def fetch(self, query: str, *args) -> list[Any]:
        """Alias for fetchall to match asyncpg-style interfaces."""
        return await self.fetchall(query, *args)

    async def fetchval(self, query: str, *args) -> Any:
        """Fetch a single value"""
        async with self.acquire() as conn:
            is_postgres = self.pool is not None
            if is_postgres:
                # PostgreSQL
                params = _flatten_params(args)
                pg_query = _convert_question_mark_to_dollar(query, params)
                return await conn.fetchval(pg_query, *params)
            else:
                # SQLite
                params = args[0] if (len(args) == 1 and isinstance(args[0], (list, tuple))) else args
                q = _normalize_sqlite_sql(query)
                cursor = await conn.execute(q, tuple(params))
                row = await cursor.fetchone()
                return row[0] if row else None

    @asynccontextmanager
    async def acquire_openai_credential_lock_connection(
        self,
        *,
        timeout: float | None = None,
    ) -> AsyncIterator[Any]:
        """Acquire a PostgreSQL session dedicated to OpenAI credential locking."""
        if not self._initialized:
            await self.initialize()

        lock_pool = self._openai_credential_lock_pool
        if lock_pool is None:
            raise DatabaseError("OpenAI credential lock pool unavailable")

        try:
            if timeout is None:
                conn = await lock_pool.acquire()
            else:
                conn = await lock_pool.acquire(timeout=timeout)
        except asyncpg.exceptions.TooManyConnectionsError:
            raise ConnectionPoolExhaustedError() from None

        failure: BaseException | None = None
        try:
            yield conn
        except BaseException as exc:  # noqa: BLE001 - release before propagation
            failure = exc

        had_primary_failure = failure is not None
        failure = await _release_postgres_connection(
            lock_pool,
            conn,
            timeout,
            failure,
        )
        if failure is not None:
            if not had_primary_failure and isinstance(failure, TransactionError):
                raise failure from None
            if isinstance(failure, Exception):
                raise failure
            raise failure from None

    async def _close_postgres_pools(self) -> bool:
        """Close both PostgreSQL pools and report caller cancellation."""
        cancelled = False

        async def _drain(pool: asyncpg.Pool) -> None:
            nonlocal cancelled
            close_task = asyncio.create_task(pool.close())
            while True:
                try:
                    await asyncio.shield(close_task)
                    return
                except asyncio.CancelledError:
                    if close_task.cancelled():
                        raise
                    cancelled = True

        lock_pool = self._openai_credential_lock_pool
        if lock_pool is not None:
            try:
                await _drain(lock_pool)
            except _AUTHNZ_DB_NONCRITICAL_EXCEPTIONS as exc:
                logger.bind(exception_type=type(exc).__name__).debug(
                    "Ignoring OpenAI credential lock pool close error during shutdown"
                )
            finally:
                if self._openai_credential_lock_pool is lock_pool:
                    self._openai_credential_lock_pool = None

        main_pool = self.pool
        if main_pool is not None:
            try:
                await _drain(main_pool)
            except _AUTHNZ_DB_NONCRITICAL_EXCEPTIONS as exc:
                logger.bind(exception_type=type(exc).__name__).debug(
                    "Ignoring pool.close() error during shutdown"
                )
            finally:
                if self.pool is main_pool:
                    self.pool = None
                    self._loop = None
        return cancelled

    async def close(self):
        """Close database connections"""
        cancelled = await self._close_postgres_pools()
        self._initialized = False
        logger.info("Database pool closed")
        if cancelled:
            raise asyncio.CancelledError

    async def health_check(self) -> dict[str, Any]:
        """Perform database health check"""
        database_type = "postgresql" if self.pool is not None else "sqlite"
        try:
            postgres_pool = self.pool
            if postgres_pool is not None:
                # PostgreSQL health check
                async with postgres_pool.acquire() as conn:
                    await conn.fetchval("SELECT 1")
                    pool_size = postgres_pool.get_size()
                    idle_size = postgres_pool.get_idle_size()

                    return {
                        "status": "healthy",
                        "type": "postgresql",
                        "pool_size": pool_size,
                        "idle_connections": idle_size,
                        "active_connections": pool_size - idle_size
                    }
            else:
                # SQLite health check
                async with aiosqlite.connect(self.db_path, uri=self._sqlite_uri) as conn:
                    await conn.execute("SELECT 1")

                    # Get database file size
                    fs_path = self._sqlite_fs_path
                    db_size = 0
                    if fs_path and fs_path != ":memory:" and os.path.exists(fs_path):
                        db_size = os.path.getsize(fs_path)

                    return {
                        "status": "healthy",
                        "type": "sqlite",
                        "database_size_mb": round(db_size / (1024 * 1024), 2)
                    }

        except (_AUTHNZ_DB_NONCRITICAL_EXCEPTIONS + (sqlite3.Error,)) as exc:
            logger.bind(
                database_type=database_type,
                exception_type=type(exc).__name__,
            ).error("Database health check failed")
            return {
                "status": "unhealthy",
                "type": database_type,
                "error": "database_unavailable",
            }


#######################################################################################################################
#
# Dependency Injection

# Global database pool instance
_db_pool: DatabasePool | None = None
_db_pool_lifecycle_lock = threading.Lock()


async def _acquire_db_pool_lifecycle_lock() -> None:
    """Acquire the process-wide lifecycle lock without blocking an event loop."""
    while not _db_pool_lifecycle_lock.acquire(blocking=False):
        await asyncio.sleep(0.001)


@asynccontextmanager
async def _db_pool_lifecycle() -> AsyncIterator[None]:
    await _acquire_db_pool_lifecycle_lock()
    try:
        yield
    finally:
        _db_pool_lifecycle_lock.release()


async def _create_initialized_db_pool(settings: Settings) -> DatabasePool:
    candidate = DatabasePool(settings)
    await candidate.initialize()
    if not candidate._initialized:
        raise DatabaseError("Database initialization did not complete")
    return candidate


async def _get_db_pool_locked(current_settings: Settings) -> DatabasePool:
    global _db_pool

    if _db_pool is None or not _db_pool._initialized:
        _db_pool = await _create_initialized_db_pool(current_settings)
        return _db_pool

    previous_settings: Settings | None = getattr(_db_pool, "settings", None)
    if previous_settings:
        auth_mode_changed = previous_settings.AUTH_MODE != current_settings.AUTH_MODE
        db_url_changed = previous_settings.DATABASE_URL != current_settings.DATABASE_URL
        if auth_mode_changed or db_url_changed:
            logger.info(
                "AuthNZ database configuration changed "
                "(auth_mode_changed={}, database_backend_changed={}, "
                "previous_backend={}, current_backend={}) - recreating pool",
                auth_mode_changed,
                _database_url_backend_label(previous_settings.DATABASE_URL)
                != _database_url_backend_label(current_settings.DATABASE_URL),
                _database_url_backend_label(previous_settings.DATABASE_URL),
                _database_url_backend_label(current_settings.DATABASE_URL),
            )
            try:
                await _db_pool.close()
            except _AUTHNZ_DB_NONCRITICAL_EXCEPTIONS as exc:
                logger.bind(exception_type=type(exc).__name__).debug(
                    "Ignoring error while closing pool during config change"
                )
            _db_pool = await _create_initialized_db_pool(current_settings)
            return _db_pool
    else:
        _db_pool.settings = current_settings

    if _db_pool.settings is not current_settings:
        # Keep pool's settings reference in sync with latest resolved Settings object
        _db_pool.settings = current_settings

    # Ensure the pool is compatible with the current running loop (Postgres path)
    try:
        current_loop = asyncio.get_running_loop()
    except RuntimeError:
        current_loop = None
    # If an existing Postgres pool is bound to a different loop, recreate it
    if getattr(_db_pool, 'pool', None) is not None and getattr(_db_pool, '_loop', None) is not None:
        if _db_pool._loop is not None and current_loop is not None and _db_pool._loop is not current_loop:
            logger.info("Detected DB pool bound to a different event loop; recreating for current loop")
            try:
                await _db_pool.close()
            except _AUTHNZ_DB_NONCRITICAL_EXCEPTIONS as e:
                logger.debug(f"Ignoring error while closing incompatible pool: {e}")
            _db_pool = await _create_initialized_db_pool(current_settings)
    return _db_pool


async def get_db_pool() -> DatabasePool:
    """Get the process-wide database pool singleton instance."""
    async with _db_pool_lifecycle():
        return await _get_db_pool_locked(get_settings())


def _database_url_backend_label(database_url: object) -> str:
    """Return a non-sensitive backend label for configuration diagnostics."""
    if not isinstance(database_url, str):
        return "unknown"
    try:
        scheme = (urlparse(database_url).scheme or "").lower()
    except (TypeError, ValueError):
        return "unknown"
    if scheme.startswith("postgres"):
        return "postgresql"
    if scheme.startswith("sqlite") or scheme == "file":
        return "sqlite"
    return "unknown"


async def reset_db_pool():
    """Reset database pool (mainly for testing)"""
    global _db_pool
    async with _db_pool_lifecycle():
        # Ensure subsequent get_db_pool() picks up environment changes.
        try:
            from tldw_Server_API.app.core.AuthNZ.settings import reset_settings as _reset_settings
            _reset_settings()
        except _AUTHNZ_DB_NONCRITICAL_EXCEPTIONS as e:
            logger.debug(f"reset_db_pool: ignoring settings reset error: {e}")
        # Keep backend helpers synchronized with the settings reset.
        try:
            from tldw_Server_API.app.core.AuthNZ.db_config import AuthDatabaseConfig as _AuthDatabaseConfig
            cfg = _AuthDatabaseConfig()
            reset_lazy = getattr(cfg, "reset_lazy", None)
            if callable(reset_lazy):
                reset_lazy()
            else:
                cfg.reset()
        except _AUTHNZ_DB_NONCRITICAL_EXCEPTIONS as e:
            logger.debug(f"reset_db_pool: ignoring AuthDatabaseConfig reset error: {e}")
        pool_to_close = _db_pool
        if pool_to_close:
            try:
                await pool_to_close.close()
            except _AUTHNZ_DB_NONCRITICAL_EXCEPTIONS as e:
                # The loop might already be closed by a TestClient.
                logger.debug(f"reset_db_pool: ignoring close error: {e}")
            finally:
                if _db_pool is pool_to_close:
                    _db_pool = None
    try:
        from tldw_Server_API.app.core.MCP_unified.auth.authnz_rbac import reset_rbac_policy as _reset_rbac_policy
        _reset_rbac_policy()
    except _AUTHNZ_DB_NONCRITICAL_EXCEPTIONS as e:
        logger.debug(f"reset_db_pool: ignoring RBAC policy reset error: {e}")
    # Reset MCP cached configuration/filters so tests pick up new DB/config values
    try:
        from tldw_Server_API.app.core.MCP_unified.config import get_config as _get_mcp_config
        if hasattr(_get_mcp_config, "cache_clear"):
            _get_mcp_config.cache_clear()
    except _AUTHNZ_DB_NONCRITICAL_EXCEPTIONS as e:
        logger.debug(f"reset_db_pool: ignoring MCP config cache reset error: {e}")
    try:
        from tldw_Server_API.app.core.MCP_unified.security.ip_filter import (
            get_ip_access_controller as _get_ip_controller,
        )
        if hasattr(_get_ip_controller, "cache_clear"):
            _get_ip_controller.cache_clear()
    except _AUTHNZ_DB_NONCRITICAL_EXCEPTIONS as e:
        logger.debug(f"reset_db_pool: ignoring MCP IP access controller reset error: {e}")
    try:

        from tldw_Server_API.app.core.MCP_unified.server import reset_mcp_server as _reset_mcp_server
        await _reset_mcp_server()
    except _AUTHNZ_DB_NONCRITICAL_EXCEPTIONS as e:
        logger.debug(f"reset_db_pool: ignoring MCP server reset error: {e}")
    try:
        from tldw_Server_API.app.core.AuthNZ.api_key_manager import reset_api_key_manager as _reset_api_manager
        await _reset_api_manager()
    except _AUTHNZ_DB_NONCRITICAL_EXCEPTIONS as e:
        logger.debug(f"reset_db_pool: ignoring API key manager reset error: {e}")
    try:
        from tldw_Server_API.app.core.DB_Management.Users_DB import reset_users_db as _reset_users_db
        await _reset_users_db()
    except _AUTHNZ_DB_NONCRITICAL_EXCEPTIONS as e:
        logger.debug(f"reset_db_pool: ignoring UsersDB reset error: {e}")
    try:
        from tldw_Server_API.app.core.AuthNZ.llm_provider_overrides import (
            set_llm_provider_overrides_cache_for_tests as _reset_llm_overrides_cache,
        )
        _reset_llm_overrides_cache({})
    except _AUTHNZ_DB_NONCRITICAL_EXCEPTIONS as e:
        logger.debug(f"reset_db_pool: ignoring LLM provider overrides cache reset error: {e}")

async def get_db():
    """FastAPI dependency to get database connection"""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        yield conn


async def get_db_transaction():
    """FastAPI dependency to get database transaction"""
    pool = await get_db_pool()
    async with pool.transaction() as conn:
        yield conn


#######################################################################################################################
#
# Utility Functions

async def test_database_connection() -> bool:
    """Test database connection"""
    try:
        pool = await get_db_pool()
        health = await pool.health_check()
        return health.get("status") == "healthy"
    except _AUTHNZ_DB_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Database connection test failed: {e}")
        return False


async def execute_migration(migration_sql: str) -> bool:
    """Execute a database migration"""
    try:
        pool = await get_db_pool()
        await pool.execute(migration_sql)
        logger.info("Migration executed successfully")
        return True
    except _AUTHNZ_DB_NONCRITICAL_EXCEPTIONS as e:
        logger.error(f"Migration failed: {e}")
    return False


# --- Internal helpers ---
_DOLLAR_PARAM = re.compile(r"\$\d+")

#######################################################################################################################
#
# Shared backend detection helper

async def is_postgres_backend() -> bool:
    """Return True if the configured AuthNZ database backend is PostgreSQL.

    Uses the presence of an asyncpg pool on the DatabasePool singleton as the
    definitive signal, avoiding fragile attribute checks on per-request
    connections.
    """
    try:
        pool = await get_db_pool()
    except DatabaseError as exc:
        logger.debug("AuthNZ backend detection falling back to SQLite due to pool error: {}", exc)
        return False
    except _AUTHNZ_DB_NONCRITICAL_EXCEPTIONS as exc:  # pragma: no cover - defensive
        logger.debug("AuthNZ backend detection encountered unexpected error: {}", exc)
        return False
    return getattr(pool, "pool", None) is not None

def _normalize_sqlite_sql(query: str) -> str:
    """Convert Postgres-style $1 placeholders to SQLite '?' when needed.

    The admin endpoints and services generally branch on backend, but this
    normalization provides a safety net to avoid aiosqlite warnings when a
    $-style query slips through the SQLite path.
    """
    if "$" not in query:
        return query
    # Replace all occurrences of $N with '?' keeping ordering intact
    return _DOLLAR_PARAM.sub("?", query)


def _flatten_params(args: tuple[Any, ...]) -> tuple[Any, ...]:
    """Support both variadic and single-sequence parameter passing."""
    if len(args) == 1 and isinstance(args[0], (list, tuple)):
        return tuple(args[0])
    return tuple(args)


def _convert_question_mark_to_dollar(query: str, params: tuple[Any, ...]) -> str:
    """Convert '?' placeholders to Postgres-style '$N' placeholders when needed."""
    if "?" not in query or "$" in query:
        return query
    count = query.count("?")
    if count != len(params):
        logger.warning(
            "Query placeholder count mismatch (found {} '?', got {} params). Leaving query unchanged.",
            count,
            len(params),
        )
        return query
    parts = query.split("?")
    rebuilt = []
    for idx, part in enumerate(parts[:-1]):
        rebuilt.append(part)
        rebuilt.append(f"${idx + 1}")
    rebuilt.append(parts[-1])
    return "".join(rebuilt)


#
# End of database.py
#######################################################################################################################
