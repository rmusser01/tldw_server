"""
PostgreSQL backend implementation for the database abstraction layer.

This module provides a concrete implementation of the DatabaseBackend
interface for PostgreSQL databases, enabling the application to use
PostgreSQL as an alternative to SQLite.

Note: This implementation requires psycopg (v3) to be installed:
    pip install "psycopg[binary]"
    # optional pooling extras:
    pip install psycopg-pool
"""

import os
import time
from collections.abc import Generator
from contextlib import contextmanager, suppress
from typing import Any, Optional, Union

from loguru import logger

from tldw_Server_API.app.core.DB_Management.scope_context import get_scope
from tldw_Server_API.app.core.DB_Management.sql_utils import split_sql_statements
from tldw_Server_API.app.core.testing import is_truthy

from .base import (
    BackendFeatures,
    BackendType,
    ConnectionPool,
    DatabaseBackend,
    DatabaseConfig,
    DatabaseError,
    FTSQuery,
    QueryResult,
)
from .fts_translator import FTSQueryTranslator
from .query_utils import (
    prepare_backend_many_statement,
    prepare_backend_statement,
)

# Try to import psycopg v3. Keep the legacy flag name for test compatibility.
try:
    import psycopg  # type: ignore
    from psycopg.rows import dict_row  # type: ignore
    try:
        import psycopg_pool  # type: ignore
    except ImportError:  # pool is optional
        psycopg_pool = None  # type: ignore
    PSYCOPG2_AVAILABLE = True  # Legacy name used by tests to simulate missing driver
except ImportError:
    PSYCOPG2_AVAILABLE = False
    logger.warning("psycopg (v3) not available. PostgreSQL backend will not work.")

if PSYCOPG2_AVAILABLE:
    _PSYCOPG_DRIVER_EXCEPTIONS: tuple[type[BaseException], ...] = tuple(
        exc
        for exc in (
            getattr(psycopg, "Error", None),
            getattr(psycopg, "Warning", None),
        )
        if isinstance(exc, type) and issubclass(exc, BaseException)
    )
else:
    _PSYCOPG_DRIVER_EXCEPTIONS = ()

_POSTGRES_BACKEND_NONCRITICAL_EXCEPTIONS = (
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
    DatabaseError,
    *_PSYCOPG_DRIVER_EXCEPTIONS,
)


_WRITE_COMMANDS = {
    "INSERT",
    "UPDATE",
    "DELETE",
    "MERGE",
    "CREATE",
    "ALTER",
    "DROP",
    "TRUNCATE",
    "GRANT",
    "REVOKE",
    "COMMENT",
    "ANALYZE",
    "VACUUM",
    "REFRESH",
    "COPY",
    "CALL",
    "DO",
    "REINDEX",
    "CLUSTER",
}


class PostgreSQLConnectionPool(ConnectionPool):
    """PostgreSQL connection pool using psycopg (v3).

    Uses psycopg_pool when available; otherwise falls back to a simple
    on-demand pool creating connections up to pool_size.
    """

    def __init__(self, config: DatabaseConfig):
        if not PSYCOPG2_AVAILABLE:
            # Keep message for backward-compatible tests
            raise DatabaseError("psycopg2 is not installed. Install with: pip install psycopg[binary]")

        self.config = config
        self._closed = False
        self._connections: list[Any] = []
        self._free: list[Any] = []
        self._max = max(1, int(config.pool_size or 10))

        # Prefer an explicit connection string when provided (e.g. DATABASE_URL)
        # Fall back to composing a DSN from individual pg_* fields
        if getattr(config, "connection_string", None):
            self._dsn = str(config.connection_string)
        else:
            dsn = (
                f"host={config.pg_host or 'localhost'} "
                f"port={config.pg_port or 5432} "
                f"dbname={config.pg_database or 'tldw'} "
                f"user={config.pg_user or 'tldw_user'} "
                f"password={config.pg_password or ''} "
                f"sslmode={config.pg_sslmode or 'prefer'} "
                f"connect_timeout={config.connect_timeout or 10}"
            )
            self._dsn = dsn
        self._use_psycopg_pool = psycopg_pool is not None
        if self._use_psycopg_pool:
            # Create a psycopg_pool.ConnectionPool with sane production defaults
            max_size = max(1, int(self.config.pool_size or 10))
            timeout = float(self.config.pool_timeout or 30.0)
            recycle = int(self.config.pool_recycle or 3600)
            try:
                self._pool = psycopg_pool.ConnectionPool(
                    self._dsn,
                    min_size=1,
                    max_size=max_size,
                    timeout=timeout,
                    max_lifetime=recycle,
                    max_idle=recycle,
                    # Ensure JSON is parsed into Python objects consistently
                    configure=lambda conn: setattr(conn, 'row_factory', dict_row),
                )
            except _POSTGRES_BACKEND_NONCRITICAL_EXCEPTIONS:
                # Fallback to defaults if parameters unsupported
                self._pool = psycopg_pool.ConnectionPool(self._dsn)
        else:
            self._pool = None

    def _new_connection(self) -> Any:
        conn = psycopg.connect(self._dsn)
        # Ensure rows are dicts by default
        conn.row_factory = dict_row
        return conn

    def get_connection(self) -> Any:
        if self._closed:
            raise DatabaseError("Connection pool is closed")
        if self._use_psycopg_pool:
            # Standardize on getconn/putconn for clarity
            conn = self._pool.getconn()
            if hasattr(conn, 'row_factory'):
                conn.row_factory = dict_row
            try:
                self._apply_scope_settings(conn)
            except _POSTGRES_BACKEND_NONCRITICAL_EXCEPTIONS as scope_exc:
                logger.debug(f"Scope config failed for pooled connection: {scope_exc}")
            return conn
        # Fallback minimal pool
        if self._free:
            conn = self._free.pop()
            try:
                self._apply_scope_settings(conn)
            except _POSTGRES_BACKEND_NONCRITICAL_EXCEPTIONS as scope_exc:
                logger.debug(f"Scope config failed for pooled connection: {scope_exc}")
            return conn
        if len(self._connections) < self._max:
            conn = self._new_connection()
            self._connections.append(conn)
            try:
                self._apply_scope_settings(conn)
            except _POSTGRES_BACKEND_NONCRITICAL_EXCEPTIONS as scope_exc:
                logger.debug(f"Scope config failed for new connection: {scope_exc}")
            return conn
        # As a last resort, create a new connection (no hard block)
        conn = self._new_connection()
        try:
            self._apply_scope_settings(conn)
        except _POSTGRES_BACKEND_NONCRITICAL_EXCEPTIONS as scope_exc:
            logger.debug(f"Scope config failed for fallback connection: {scope_exc}")
        return conn

    def return_connection(self, connection: Any) -> None:
        if self._closed or connection is None:
            with suppress(_POSTGRES_BACKEND_NONCRITICAL_EXCEPTIONS):
                connection.close()
            return
        if self._use_psycopg_pool:
            # Always return via putconn
            try:
                self._pool.putconn(connection)
            except _POSTGRES_BACKEND_NONCRITICAL_EXCEPTIONS:
                with suppress(_POSTGRES_BACKEND_NONCRITICAL_EXCEPTIONS):
                    connection.close()
            return
        # Minimal pool:
        # - keep only managed connections in the reusable free list
        # - close overflow/fallback connections immediately to avoid growth
        is_managed = connection in self._connections
        if is_managed and len(self._free) < self._max:
            self._free.append(connection)
            return
        with suppress(_POSTGRES_BACKEND_NONCRITICAL_EXCEPTIONS):
            connection.close()

    @contextmanager
    def connection(self) -> Generator[Any, None, None]:
        conn = self.get_connection()
        try:
            yield conn
        finally:
            self.return_connection(conn)

    def close_all(self) -> None:
        self._closed = True
        if self._use_psycopg_pool:
            with suppress(_POSTGRES_BACKEND_NONCRITICAL_EXCEPTIONS):
                self._pool.close()
            return
        # Close all tracked connections (both managed and currently free).
        seen_ids: set[int] = set()
        for conn in [*self._connections, *self._free]:
            conn_id = id(conn)
            if conn_id in seen_ids:
                continue
            seen_ids.add(conn_id)
            with suppress(_POSTGRES_BACKEND_NONCRITICAL_EXCEPTIONS):
                conn.close()
        self._connections.clear()
        self._free.clear()

    def get_stats(self) -> dict[str, Any]:
        return {"closed": self._closed, "backend": "postgresql"}


class PostgreSQLBackend(DatabaseBackend):
    """PostgreSQL implementation of the database backend."""

    def __init__(self, config: DatabaseConfig):
        super().__init__(config)
        self._managed_tx_depths: dict[int, int] = {}
        # Track FTS table names to source tables/columns for Postgres.
        self._fts_table_map: dict[str, dict[str, str]] = {}

    @property
    def backend_type(self) -> BackendType:
        """Get the backend type."""
        return BackendType.POSTGRESQL

    def _get_features(self) -> BackendFeatures:
        """Get PostgreSQL feature support."""
        return BackendFeatures(
            full_text_search=True,   # tsvector/tsquery
            json_support=True,        # JSON/JSONB
            array_support=True,       # Native arrays
            window_functions=True,    # Full support
            cte_support=True,         # WITH queries
            partial_indexes=True,     # Partial indexes
            generated_columns=True,   # GENERATED columns
            upsert_support=True,      # ON CONFLICT
            returning_clause=True,    # RETURNING
            listen_notify=True        # LISTEN/NOTIFY
        )

    # Public API for callers to reapply scope safely
    def apply_scope(self, connection: Optional[Any] = None) -> None:
        try:
            if connection is not None:
                self._apply_scope_settings(connection)
        except _POSTGRES_BACKEND_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug(f"apply_scope: unable to apply scope settings: {exc}")

    def _apply_scope_settings(self, connection: Any) -> None:
        """Apply scope-related GUC settings for row-level security.

        Avoid rolling back active transactions unless necessary; if setting
        parameters fails due to a bad transaction state, rollback and retry once.
        """
        try:
            scope = get_scope()
        except _POSTGRES_BACKEND_NONCRITICAL_EXCEPTIONS:
            scope = None

        user_id = ""
        org_ids = ""
        team_ids = ""
        is_admin = "0"
        session_role: Optional[str] = None

        if scope:
            if scope.user_id is not None:
                try:
                    user_id = str(int(scope.user_id))
                except _POSTGRES_BACKEND_NONCRITICAL_EXCEPTIONS:
                    user_id = str(scope.user_id)
            if scope.org_ids:
                try:
                    org_ids = ",".join(str(int(oid)) for oid in scope.org_ids if oid is not None)
                except _POSTGRES_BACKEND_NONCRITICAL_EXCEPTIONS:
                    org_ids = ",".join(str(oid) for oid in scope.org_ids if oid is not None)
            if scope.team_ids:
                try:
                    team_ids = ",".join(str(int(tid)) for tid in scope.team_ids if tid is not None)
                except _POSTGRES_BACKEND_NONCRITICAL_EXCEPTIONS:
                    team_ids = ",".join(str(tid) for tid in scope.team_ids if tid is not None)
            if scope.is_admin:
                is_admin = "1"
            session_role = getattr(scope, "session_role", None) or None

        # Gate role switching behind explicit configuration and whitelist.
        # Default: disabled; rely on set_config GUCs and row-level security predicates.
        # Role switching must be explicitly enabled via env.
        _role_env = os.getenv("TLDW_CONTENT_PG_ROLE_SWITCH", "").strip().lower()
        allow_role_switch = False if _role_env == "" else is_truthy(_role_env)
        allowed_roles_env = os.getenv("TLDW_CONTENT_PG_ROLE_WHITELIST", "").strip()
        allowed_roles = {r.strip() for r in allowed_roles_env.split(',') if r.strip()}
        if not allow_role_switch or not session_role or (allowed_roles and session_role not in allowed_roles):
            if session_role and (not allow_role_switch or (allowed_roles and session_role not in allowed_roles)):
                logger.debug(f"Session role '{session_role}' not allowed by configuration; skipping role switch")
            session_role = None

        statements = [
            # Maintain both keys for compatibility across modules/tests
            ("SELECT set_config('app.current_user_id', %s, false)", (user_id,)),
            ("SELECT set_config('app.user_id', %s, false)", (user_id,)),
            ("SELECT set_config('app.org_ids', %s, false)", (org_ids,)),
            ("SELECT set_config('app.team_ids', %s, false)", (team_ids,)),
            ("SELECT set_config('app.is_admin', %s, false)", (is_admin,)),
        ]

        def _run_with_cursor(cur) -> None:
            if session_role:
                escaped_role = session_role.replace('"', '""')
                try:
                    cur.execute(f'SET SESSION AUTHORIZATION "{escaped_role}"')
                except _POSTGRES_BACKEND_NONCRITICAL_EXCEPTIONS:
                    cur.execute(f'SET ROLE "{escaped_role}"')
            else:
                try:
                    cur.execute("RESET SESSION AUTHORIZATION")
                except _POSTGRES_BACKEND_NONCRITICAL_EXCEPTIONS:
                    cur.execute("RESET ROLE")

            with suppress(_POSTGRES_BACKEND_NONCRITICAL_EXCEPTIONS):
                cur.execute("SET row_security = on")

            for sql_stmt, params in statements:
                cur.execute(sql_stmt, params)

        cursor_factory = getattr(connection, "cursor", None)

        # First attempt: try to set configs without touching transaction state
        try:
            if cursor_factory:
                with cursor_factory() as cur:
                    _run_with_cursor(cur)
            else:
                # Fallback if connection supports execute directly
                if session_role:
                    escaped_role = session_role.replace('"', '""')
                    try:
                        connection.execute(f'SET SESSION AUTHORIZATION "{escaped_role}"')
                    except _POSTGRES_BACKEND_NONCRITICAL_EXCEPTIONS:
                        connection.execute(f'SET ROLE "{escaped_role}"')
                else:
                    try:
                        connection.execute("RESET SESSION AUTHORIZATION")
                    except _POSTGRES_BACKEND_NONCRITICAL_EXCEPTIONS:
                        connection.execute("RESET ROLE")
                with suppress(_POSTGRES_BACKEND_NONCRITICAL_EXCEPTIONS):
                    connection.execute("SET row_security = on")
                for sql_stmt, params in statements:
                    try:
                        connection.execute(sql_stmt, params)
                    except _POSTGRES_BACKEND_NONCRITICAL_EXCEPTIONS as cfg_exc:
                        logger.debug(f"Unable to apply scope settings via execute: {cfg_exc}")
        except _POSTGRES_BACKEND_NONCRITICAL_EXCEPTIONS:
            # If we failed (e.g., transaction aborted), rollback and try once more
            try:
                connection.rollback()
            except _POSTGRES_BACKEND_NONCRITICAL_EXCEPTIONS as rollback_exc:
                logger.debug("Rollback while configuring scope failed: {}", rollback_exc)
            try:
                if cursor_factory:
                    with cursor_factory() as cur:
                        _run_with_cursor(cur)
                else:
                    if session_role:
                        escaped_role = session_role.replace('"', '""')
                        try:
                            connection.execute(f'SET SESSION AUTHORIZATION "{escaped_role}"')
                        except _POSTGRES_BACKEND_NONCRITICAL_EXCEPTIONS:
                            connection.execute(f'SET ROLE "{escaped_role}"')
                    else:
                        try:
                            connection.execute("RESET SESSION AUTHORIZATION")
                        except _POSTGRES_BACKEND_NONCRITICAL_EXCEPTIONS:
                            connection.execute("RESET ROLE")
                    with suppress(_POSTGRES_BACKEND_NONCRITICAL_EXCEPTIONS):
                        connection.execute("SET row_security = on")
                    for sql_stmt, params in statements:
                        try:
                            connection.execute(sql_stmt, params)
                        except _POSTGRES_BACKEND_NONCRITICAL_EXCEPTIONS as cfg_exc:
                            logger.debug(f"Unable to apply scope settings via execute (after rollback): {cfg_exc}")
            except _POSTGRES_BACKEND_NONCRITICAL_EXCEPTIONS as final_exc:
                logger.debug(f"Failed to configure session scope settings: {final_exc}")

    def _tx_depth(self, connection: Any) -> int:
        return self._managed_tx_depths.get(id(connection), 0)

    def _tx_depth_inc(self, connection: Any) -> None:
        key = id(connection)
        self._managed_tx_depths[key] = self._managed_tx_depths.get(key, 0) + 1

    def _tx_depth_dec(self, connection: Any) -> None:
        key = id(connection)
        current = self._managed_tx_depths.get(key, 0)
        if current <= 1:
            self._managed_tx_depths.pop(key, None)
        else:
            self._managed_tx_depths[key] = current - 1

    @staticmethod
    def _strip_leading_comments(sql: str) -> str:
        """Remove leading SQL comments so keyword detection is reliable."""
        text = sql.lstrip()
        while text:
            if text.startswith("--"):
                newline = text.find("\n")
                if newline == -1:
                    return ""
                text = text[newline + 1 :].lstrip()
                continue
            if text.startswith("/*"):
                end = text.find("*/", 2)
                if end == -1:
                    return ""
                text = text[end + 2 :].lstrip()
                continue
            break
        return text

    @staticmethod
    def _split_parenthesized_block(text: str) -> tuple[str, str]:
        """Return the inner content and remainder after a balanced (...) block."""
        if not text.startswith("("):
            return "", text

        depth = 1
        idx = 1
        length = len(text)
        while idx < length and depth > 0:
            ch = text[idx]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    idx += 1
                    break
            elif ch in ("'", '"'):
                quote = ch
                idx += 1
                while idx < length:
                    c = text[idx]
                    if c == quote:
                        if idx + 1 < length and text[idx + 1] == quote:
                            idx += 2
                            continue
                        break
                    idx += 1
            idx += 1

        if depth > 0:
            return "", ""

        body = text[1:idx - 1]
        remainder = text[idx:]
        return body, remainder

    @staticmethod
    def _skip_parenthesized_block(text: str) -> str:
        """Return the substring after the first balanced parenthesized block."""
        _, remainder = PostgreSQLBackend._split_parenthesized_block(text)
        return remainder

    @staticmethod
    def _command_after_cte(sql: str) -> str:
        """Return the first command keyword following an optional CTE block."""
        text = PostgreSQLBackend._strip_leading_comments(sql)
        if not text:
            return ""

        upper = text.upper()
        if not upper.startswith("WITH"):
            return text.split(None, 1)[0].upper()

        # Strip leading WITH (and optional RECURSIVE) keyword.
        text = text[4:].lstrip()
        if text.upper().startswith("RECURSIVE"):
            text = text[len("RECURSIVE"):].lstrip()

        # Skip one or more CTE definitions separated by commas.
        while text:
            # Skip identifier / optional schema qualifier.
            idx = 0
            length = len(text)
            while idx < length and (text[idx].isalnum() or text[idx] in ('_', '.', '"')):
                idx += 1
            original_text = text
            text = PostgreSQLBackend._strip_leading_comments(text[idx:].lstrip())

            # Optional column list e.g., WITH c(id, name) AS ...
            if text.startswith("("):
                _, remainder = PostgreSQLBackend._split_parenthesized_block(text)
                if remainder == "":
                    return ""
                potential_tail = PostgreSQLBackend._strip_leading_comments(remainder.lstrip())
                text = potential_tail if potential_tail.upper().startswith("AS") else original_text

            upper_tail = text.upper()
            if upper_tail.startswith("AS"):
                text = text[2:].lstrip()
                upper_tail = text.upper()
                if upper_tail.startswith("NOT MATERIALIZED"):
                    text = text[len("NOT MATERIALIZED"):].lstrip()
                elif upper_tail.startswith("MATERIALIZED"):
                    text = text[len("MATERIALIZED"):].lstrip()
                text = PostgreSQLBackend._strip_leading_comments(text)

            if not text.startswith("("):
                break

            _, remainder = PostgreSQLBackend._split_parenthesized_block(text)
            if remainder == "":
                return ""
            text = PostgreSQLBackend._strip_leading_comments(remainder.lstrip())
            if text.startswith(","):
                text = text[1:].lstrip()
                continue
            break

        if not text:
            return ""
        text = PostgreSQLBackend._strip_leading_comments(text)
        if not text:
            return ""
        return text.split(None, 1)[0].upper()

    @staticmethod
    def _cte_contains_write(sql: str) -> bool:
        """Check whether any CTE body contains a write command."""
        text = PostgreSQLBackend._strip_leading_comments(sql)
        if not text or not text.upper().startswith("WITH"):
            return False

        text = text[4:].lstrip()
        if text.upper().startswith("RECURSIVE"):
            text = text[len("RECURSIVE"):].lstrip()

        while text:
            idx = 0
            length = len(text)
            while idx < length and (text[idx].isalnum() or text[idx] in ('_', '.', '"')):
                idx += 1
            original_text = text
            text = PostgreSQLBackend._strip_leading_comments(text[idx:].lstrip())

            if text.startswith("("):
                body_candidate, remainder = PostgreSQLBackend._split_parenthesized_block(text)
                if remainder == "":
                    return False
                potential_tail = PostgreSQLBackend._strip_leading_comments(remainder.lstrip())
                text = potential_tail if potential_tail.upper().startswith("AS") else original_text

            upper_tail = text.upper()
            if upper_tail.startswith("AS"):
                text = text[2:].lstrip()
                upper_tail = text.upper()
                if upper_tail.startswith("NOT MATERIALIZED"):
                    text = text[len("NOT MATERIALIZED"):].lstrip()
                elif upper_tail.startswith("MATERIALIZED"):
                    text = text[len("MATERIALIZED"):].lstrip()
                text = PostgreSQLBackend._strip_leading_comments(text)

            if not text.startswith("("):
                break

            body, remainder = PostgreSQLBackend._split_parenthesized_block(text)
            if body == "" and remainder == "":
                return False

            body_head = PostgreSQLBackend._strip_leading_comments(body)
            if body_head:
                command = body_head.split(None, 1)[0].upper()
                if command in _WRITE_COMMANDS:
                    return True

            text = PostgreSQLBackend._strip_leading_comments(remainder.lstrip())
            if text.startswith(","):
                text = text[1:].lstrip()
                continue
            break

        return False

    def _is_write_command(self, command_tag: str, sql: str) -> bool:
        """Determine whether a statement should be treated as a write."""
        normalized = (command_tag or "").upper()
        if not normalized or normalized == "WITH":
            normalized = self._command_after_cte(sql)
        if normalized in _WRITE_COMMANDS:
            return True
        return PostgreSQLBackend._cte_contains_write(sql)

    def connect(self) -> Any:
        """Create a new PostgreSQL connection."""
        if not PSYCOPG2_AVAILABLE:
            # Keep message for compatibility with existing tests
            raise DatabaseError("psycopg2 is not installed")

        # Match pool initialization precedence: prefer explicit connection_string when present.
        if getattr(self.config, "connection_string", None):
            dsn = str(self.config.connection_string)
        else:
            dsn = (
                f"host={self.config.pg_host or 'localhost'} "
                f"port={self.config.pg_port or 5432} "
                f"dbname={self.config.pg_database or 'tldw'} "
                f"user={self.config.pg_user or 'tldw_user'} "
                f"password={self.config.pg_password or ''} "
                f"sslmode={self.config.pg_sslmode or 'prefer'} "
                f"connect_timeout={self.config.connect_timeout or 10}"
            )
        conn = psycopg.connect(dsn)
        conn.row_factory = dict_row
        try:
            self._apply_scope_settings(conn)
        except _POSTGRES_BACKEND_NONCRITICAL_EXCEPTIONS as scope_exc:
            logger.debug(f"Scope config failed for direct connection: {scope_exc}")
        return conn

    def disconnect(self, connection: Any) -> None:
        """Close a PostgreSQL connection."""
        self._managed_tx_depths.pop(id(connection), None)
        if connection and not connection.closed:
            connection.close()

    @contextmanager
    def transaction(self, connection: Optional[Any] = None) -> Generator[Any, None, None]:
        """PostgreSQL transaction context manager."""
        if connection:
            conn = connection
            owns_connection = False
        else:
            conn = self.get_pool().get_connection()
            owns_connection = True

        try:
            # Track managed transaction depth per-connection so we can
            # reliably commit/rollback only at the outermost boundary,
            # regardless of whether we own the connection or it was
            # supplied by the caller.
            self._tx_depth_inc(conn)
            is_outermost = self._tx_depth(conn) == 1
            # PostgreSQL uses implicit transactions
            yield conn
            # Commit only when we're at the outermost depth for this connection.
            if is_outermost:
                try:
                    conn.commit()
                except _POSTGRES_BACKEND_NONCRITICAL_EXCEPTIONS as e:
                    # Surface commit failures consistently
                    logger.error(f"Transaction commit failed: {e}")
                    raise
        except _POSTGRES_BACKEND_NONCRITICAL_EXCEPTIONS as e:
            # Roll back only when we're at the outermost depth for this connection.
            try:
                if self._tx_depth(conn) == 1:
                    conn.rollback()
            except _POSTGRES_BACKEND_NONCRITICAL_EXCEPTIONS:
                # Swallow rollback errors to avoid masking the original
                pass
            logger.error(f"Transaction failed: {e}")
            raise
        finally:
            self._tx_depth_dec(conn)
            if owns_connection:
                self.get_pool().return_connection(conn)

    def get_pool(self) -> ConnectionPool:
        """Get or create the connection pool."""
        if self._pool is None:
            pool = PostgreSQLConnectionPool(self.config)
            # Propagate scope configuration helper so the pool can refresh session state.
            pool._apply_scope_settings = self._apply_scope_settings
            self._pool = pool
        return self._pool

    def _prepare_query(
        self,
        query: str,
        params: Optional[Union[tuple, dict]]
    ) -> tuple[str, Optional[Union[tuple, dict]]]:
        """Prepare SQL and params for psycopg with robust placeholder handling.

        Delegates to shared query utils to safely convert SQLite-style
        placeholders ('?') to Postgres ('%s'), avoiding replacements inside
        quoted strings. Also normalizes params.
        """
        prepared_sql, prepared_params = prepare_backend_statement(
            BackendType.POSTGRESQL,
            query,
            params,
            apply_default_transform=True,
            ensure_returning=False,
        )
        return prepared_sql, prepared_params

    def execute(
        self,
        query: str,
        params: Optional[Union[tuple, dict]] = None,
        connection: Optional[Any] = None
    ) -> QueryResult:
        """Execute a query and return results."""
        start_time = time.time()
        query, params = self._prepare_query(query, params)

        if connection:
            conn = connection
            external_conn = True
        else:
            conn = self.get_pool().get_connection()
            external_conn = False

        try:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)

            has_description = cursor.description is not None

            # Some wrapped cursors used in higher-level transactions may not expose
            # a psycopg-like `statusmessage` attribute. Guard access to avoid raising
            # during metadata fetches executed within an external transaction.
            statusmsg = getattr(cursor, "statusmessage", None)
            status = (statusmsg or "").split()
            command_tag = status[0].upper() if status else ""
            is_write = self._is_write_command(command_tag, query)

            if has_description:
                rows = cursor.fetchall()
                # psycopg v3 will yield dicts if row_factory is set; otherwise adapt
                if rows and isinstance(rows[0], dict):
                    result_rows = rows  # already dicts
                else:
                    column_names = [col[0] for col in (cursor.description or [])]
                    result_rows = [
                        {column_names[idx]: value for idx, value in enumerate(row)}
                        for row in rows
                    ]
            else:
                result_rows = []

            managed_depth = self._tx_depth(conn)
            if managed_depth == 0 and not external_conn:
                if is_write:
                    conn.commit()
                else:
                    # Close implicit read-only transaction to avoid idle-in-transaction sessions.
                    try:
                        conn.rollback()
                    except _POSTGRES_BACKEND_NONCRITICAL_EXCEPTIONS as rollback_exc:  # noqa: BLE001
                        logger.debug(f"Rollback after read-only execute() failed: {rollback_exc}")

            execution_time = time.time() - start_time

            lastrowid = None
            if result_rows:
                first_row = result_rows[0]
                lastrowid = first_row.get('id') if isinstance(first_row, dict) else None

            return QueryResult(
                rows=result_rows,
                rowcount=cursor.rowcount,
                lastrowid=lastrowid,
                description=cursor.description,
                execution_time=execution_time
            )

        except _POSTGRES_BACKEND_NONCRITICAL_EXCEPTIONS as e:
            if not external_conn:
                try:
                    conn.rollback()
                except _POSTGRES_BACKEND_NONCRITICAL_EXCEPTIONS as rollback_exc:  # noqa: BLE001
                    logger.debug(f"Rollback after failed execute() also failed: {rollback_exc}")
            logger.error(f"Query execution failed: {e}")
            raise DatabaseError(f"PostgreSQL error: {e}") from e
        finally:
            if not external_conn:
                self.get_pool().return_connection(conn)

    def execute_many(
        self,
        query: str,
        params_list: list[Union[tuple, dict]],
        connection: Optional[Any] = None
    ) -> QueryResult:
        """Execute a query multiple times with different parameters."""
        start_time = time.time()

        if connection:
            conn = connection
            external_conn = True
        else:
            conn = self.get_pool().get_connection()
            external_conn = False

        try:
            cursor = conn.cursor()
            normalized_query, normalized_params = prepare_backend_many_statement(
                BackendType.POSTGRESQL,
                query,
                params_list,
                apply_default_transform=True,
                ensure_returning=False,
            )

            cursor.executemany(normalized_query, normalized_params)

            statusmsg = getattr(cursor, "statusmessage", None)
            status = (statusmsg or "").split()
            command_tag = status[0].upper() if status else ""
            is_write = self._is_write_command(command_tag, normalized_query)

            managed_depth = self._tx_depth(conn)
            if managed_depth == 0 and not external_conn:
                if is_write:
                    conn.commit()
                else:
                    # Close implicit read-only transaction to avoid idle-in-transaction sessions.
                    try:
                        conn.rollback()
                    except _POSTGRES_BACKEND_NONCRITICAL_EXCEPTIONS as rollback_exc:  # noqa: BLE001
                        logger.debug(f"Rollback after read-only execute_many() failed: {rollback_exc}")

            execution_time = time.time() - start_time

            return QueryResult(
                rows=[],
                rowcount=cursor.rowcount,
                lastrowid=None,
                description=cursor.description,
                execution_time=execution_time
            )

        except _POSTGRES_BACKEND_NONCRITICAL_EXCEPTIONS as e:
            if not external_conn:
                try:
                    conn.rollback()
                except _POSTGRES_BACKEND_NONCRITICAL_EXCEPTIONS as rollback_exc:  # noqa: BLE001
                    logger.debug(f"Rollback after failed execute_many() also failed: {rollback_exc}")
            logger.error(f"Batch execution failed: {e}")
            raise DatabaseError(f"PostgreSQL error: {e}") from e
        finally:
            if not external_conn:
                self.get_pool().return_connection(conn)

    def create_tables(self, schema: str, connection: Optional[Any] = None) -> None:
        """Create tables from a schema definition."""
        # PostgreSQL doesn't support multiple statements in execute()
        # Split safely to handle dollar-quoted bodies.
        statements = split_sql_statements(schema)

        if connection:
            conn = connection
            external_conn = True
        else:
            conn = self.get_pool().get_connection()
            external_conn = False

        try:
            cursor = conn.cursor()
            for statement in statements:
                if statement:
                    cursor.execute(statement)
            if not external_conn:
                conn.commit()
        except _POSTGRES_BACKEND_NONCRITICAL_EXCEPTIONS as e:
            if not external_conn:
                conn.rollback()
            logger.error(f"Schema creation failed: {e}")
            raise DatabaseError(f"Failed to create schema: {e}") from e
        finally:
            if not external_conn:
                self.get_pool().return_connection(conn)

    def table_exists(self, table_name: str, connection: Optional[Any] = None) -> bool:
        """Check if a table exists."""
        query = (
            "SELECT EXISTS ("
            " SELECT FROM information_schema.tables"
            " WHERE table_schema = 'public' AND table_name = %s)"
        )
        result = self.execute(query, (table_name,), connection=connection)
        if result.scalar:
            return True

        folded_name = table_name.lower()
        if folded_name == table_name:
            return False

        # PostgreSQL folds unquoted identifiers to lowercase. Allow mixed-case
        # callers like "MediaFiles" to resolve those tables without losing the
        # ability to distinguish deliberately quoted mixed-case tables.
        folded_result = self.execute(query, (folded_name,), connection=connection)
        return bool(folded_result.scalar)

    def get_table_info(
        self,
        table_name: str,
        connection: Optional[Any] = None
    ) -> list[dict[str, Any]]:
        """Get information about a table's columns."""
        query = """
            SELECT
                column_name as name,
                data_type as type,
                is_nullable = 'YES' as nullable,
                column_default as default,
                false as primary_key
            FROM information_schema.columns
            WHERE table_schema = 'public'
            AND table_name = %s
            ORDER BY ordinal_position
        """
        result = self.execute(query, (table_name,), connection)
        return result.rows

    def create_fts_table(
        self,
        table_name: str,
        source_table: str,
        columns: list[str],
        connection: Optional[Any] = None
    ) -> None:
        """
        Create PostgreSQL full-text search setup.

        Instead of a virtual table, PostgreSQL uses tsvector columns
        and GIN indexes.
        """
        self.features.require("full_text_search")

        if connection:
            conn = connection
            external_conn = True
        else:
            conn = self.get_pool().get_connection()
            external_conn = False

        try:
            cursor = conn.cursor()

            # Add tsvector column to source table if not exists
            fts_column = f"{table_name}_tsv"
            cursor.execute(f"""
                ALTER TABLE {self.escape_identifier(source_table)}
                ADD COLUMN IF NOT EXISTS {self.escape_identifier(fts_column)} tsvector
            """)

            # Create update function for tsvector
            # Build columns concat for both contexts
            columns_concat_set = " || ' ' || ".join([
                f"coalesce({self.escape_identifier(col)}, '')"
                for col in columns
            ])
            columns_concat_new = " || ' ' || ".join([
                f"coalesce(NEW.{self.escape_identifier(col)}, '')"
                for col in columns
            ])

            source_table_ident = self.escape_identifier(source_table)
            fts_column_ident = self.escape_identifier(fts_column)
            refresh_tsv_sql = f"""
                UPDATE {source_table_ident}
                SET {fts_column_ident} =
                    to_tsvector('english', {columns_concat_set})
            """  # nosec B608
            cursor.execute(refresh_tsv_sql)

            # Create GIN index for fast searching
            index_name = f"idx_{source_table}_{fts_column}"
            cursor.execute(f"""
                CREATE INDEX IF NOT EXISTS {self.escape_identifier(index_name)}
                ON {self.escape_identifier(source_table)}
                USING gin({self.escape_identifier(fts_column)})
            """)

            # Create trigger to keep tsvector updated
            trigger_name = f"update_{fts_column}_trigger"
            function_name = f"update_{fts_column}_function"

            cursor.execute(f"""
                CREATE OR REPLACE FUNCTION {self.escape_identifier(function_name)}()
                RETURNS trigger AS $$
                BEGIN
                    NEW.{self.escape_identifier(fts_column)} :=
                        to_tsvector('english', {columns_concat_new});
                    RETURN NEW;
                END;
                $$ LANGUAGE plpgsql
            """)

            cursor.execute(f"""
                DROP TRIGGER IF EXISTS {self.escape_identifier(trigger_name)}
                ON {self.escape_identifier(source_table)}
            """)

            cursor.execute(f"""
                CREATE TRIGGER {self.escape_identifier(trigger_name)}
                BEFORE INSERT OR UPDATE ON {self.escape_identifier(source_table)}
                FOR EACH ROW EXECUTE FUNCTION {self.escape_identifier(function_name)}()
            """)

            # Record the mapping for fts_search when table_name != source_table.
            with suppress(_POSTGRES_BACKEND_NONCRITICAL_EXCEPTIONS):
                self._fts_table_map[table_name] = {
                    "source_table": source_table,
                    "fts_column": fts_column,
                }

            if not external_conn:
                conn.commit()

        except _POSTGRES_BACKEND_NONCRITICAL_EXCEPTIONS as e:
            if not external_conn:
                conn.rollback()
            logger.error(f"FTS setup failed: {e}")
            raise DatabaseError(f"Failed to create FTS: {e}") from e
        finally:
            if not external_conn:
                self.get_pool().return_connection(conn)

    def fts_search(
        self,
        fts_query: FTSQuery,
        connection: Optional[Any] = None
    ) -> QueryResult:
        """Perform a PostgreSQL full-text search."""
        self.features.require("full_text_search")

        if not fts_query.table:
            raise DatabaseError("Table name required for FTS")

        # Build the FTS query
        fts_table = fts_query.table
        fts_column = f"{fts_table}_tsv"
        source_table = fts_table
        mapping = self._fts_table_map.get(fts_table)
        if mapping:
            source_table = mapping.get("source_table", source_table)
            fts_column = mapping.get("fts_column", fts_column)

        normalized_query = FTSQueryTranslator.normalize_query(
            fts_query.query_text,
            "postgresql",
        )
        if not normalized_query:
            raise DatabaseError("FTS query normalized to empty for postgresql")

        query_parts = [
            f"SELECT *, ts_rank({self.escape_identifier(fts_column)}, query) AS rank",
            f"FROM {self.escape_identifier(source_table)},",
            "to_tsquery('english', %s) query",
            f"WHERE {self.escape_identifier(fts_column)} @@ query"
        ]

        params = [normalized_query]

        # Add additional filters
        for key, value in fts_query.filters.items():
            query_parts.append(f"AND {self.escape_identifier(key)} = %s")
            params.append(value)

        # Add ORDER BY
        query_parts.append("ORDER BY rank DESC")

        # Add LIMIT/OFFSET
        if fts_query.limit:
            query_parts.append(f"LIMIT {fts_query.limit}")
        if fts_query.offset:
            query_parts.append(f"OFFSET {fts_query.offset}")

        query = " ".join(query_parts)

        return self.execute(query, tuple(params), connection)

    # --- Optional FTS synonyms support (table + function) ---
    def ensure_synonyms_support(self, connection: Optional[Any] = None) -> None:
        """Create a simple synonyms table and expansion function if not present.

        - Table: fts_synonyms(term TEXT PRIMARY KEY, synonyms TEXT[])
        - Function: synonyms_expand(text) -> text (original + synonyms appended)

        Enables index-time expansion by calling synonyms_expand(title/content)
        before to_tsvector when enabled.
        """
        if not PSYCOPG2_AVAILABLE:
            raise DatabaseError("psycopg not available; cannot ensure synonyms support")
        external_conn = connection is not None
        conn = connection or self.get_pool().get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS fts_synonyms (
                    term TEXT PRIMARY KEY,
                    synonyms TEXT[]
                )
                """
            )
            cursor.execute(
                """
                CREATE OR REPLACE FUNCTION synonyms_expand(input TEXT)
                RETURNS TEXT AS $$
                DECLARE
                    arr TEXT[];
                    tok TEXT;
                    out TEXT := '';
                    syns TEXT[];
                BEGIN
                    arr := regexp_split_to_array(coalesce(input,''), '[^[:alnum:]]+');
                    FOREACH tok IN ARRAY arr LOOP
                        IF length(tok) > 0 THEN
                            out := out || tok || ' ';
                            SELECT s.synonyms INTO syns FROM fts_synonyms s WHERE s.term = lower(tok);
                            IF syns IS NOT NULL THEN
                                out := out || array_to_string(syns, ' ') || ' ';
                            END IF;
                        END IF;
                    END LOOP;
                    RETURN trim(out);
                END;
                $$ LANGUAGE plpgsql IMMUTABLE;
                """
            )
            if not external_conn:
                conn.commit()
        except _POSTGRES_BACKEND_NONCRITICAL_EXCEPTIONS as exc:
            try:
                if not external_conn:
                    conn.rollback()
            except _POSTGRES_BACKEND_NONCRITICAL_EXCEPTIONS:
                pass
            logger.error(f"Failed to ensure FTS synonyms support: {exc}")
            raise DatabaseError(f"Failed to ensure FTS synonyms support: {exc}") from exc
        finally:
            if not external_conn:
                self.get_pool().return_connection(conn)

    def update_fts_index(
        self,
        table_name: str,
        connection: Optional[Any] = None
    ) -> None:
        """Update PostgreSQL FTS index (trigger-based, so automatic)."""
        # PostgreSQL FTS is updated automatically via triggers
        # This method exists for API compatibility
        pass

    def escape_identifier(self, identifier: str) -> str:
        """Escape a PostgreSQL identifier."""
        # PostgreSQL uses double quotes for identifiers
        return '"' + identifier.replace('"', '""') + '"'

    def get_last_insert_id(self, connection: Optional[Any] = None) -> Optional[int]:
        """Get the last inserted row ID using RETURNING clause."""
        # PostgreSQL doesn't have a direct equivalent to SQLite's lastrowid
        # Use RETURNING clause in INSERT statements instead
        logger.warning("PostgreSQL doesn't support last_insert_id. Use RETURNING clause in INSERT.")
        return None

    def vacuum(self, connection: Optional[Any] = None) -> None:
        """Vacuum the PostgreSQL database."""
        # VACUUM can't run in a transaction
        if connection:
            conn = connection
            external_conn = True
        else:
            conn = self.connect()  # Need a separate connection
            external_conn = False

        try:
            # Use autocommit for VACUUM
            old_autocommit = getattr(conn, 'autocommit', False)
            with suppress(_POSTGRES_BACKEND_NONCRITICAL_EXCEPTIONS):
                conn.autocommit = True
            cursor = conn.cursor()
            cursor.execute("VACUUM ANALYZE")
            with suppress(_POSTGRES_BACKEND_NONCRITICAL_EXCEPTIONS):
                conn.autocommit = old_autocommit
        finally:
            if not external_conn:
                conn.close()

    def get_database_size(self, connection: Optional[Any] = None) -> int:
        """Get the database size in bytes."""
        query = "SELECT pg_database_size(current_database())"
        result = self.execute(query, connection=connection)
        return result.scalar or 0

    def export_schema(self, connection: Optional[Any] = None) -> str:
        """Export the database schema as SQL."""
        # This would require pg_dump or complex queries
        # Simplified version that gets table definitions
        query = """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_type = 'BASE TABLE'
        """
        result = self.execute(query, connection=connection)

        schema_parts = []
        for row in result.rows:
            table_name = row['table_name']
            # This is a simplified version - full implementation would need pg_dump
            schema_parts.append(f"-- Table: {table_name}")

        return "\n".join(schema_parts)

    def export_data(
        self,
        table_name: str,
        connection: Optional[Any] = None
    ) -> Generator[dict[str, Any], None, None]:
        """Export data from a table."""
        query = f"SELECT * FROM {self.escape_identifier(table_name)}"  # nosec B608

        if connection:
            conn = connection
            external_conn = True
        else:
            conn = self.get_pool().get_connection()
            external_conn = False

        try:
            cursor = conn.cursor()
            cursor.execute(query)

            for row in cursor:
                yield dict(row)
        finally:
            if not external_conn:
                self.get_pool().return_connection(conn)

    def import_data(
        self,
        table_name: str,
        data: list[dict[str, Any]],
        connection: Optional[Any] = None
    ) -> int:
        """Import data into a table."""
        if not data:
            return 0

        # Get column names from first row
        columns = list(data[0].keys())
        columns_str = ", ".join([self.escape_identifier(col) for col in columns])
        placeholders = ", ".join(["%s" for _ in columns])

        query = f"""
            INSERT INTO {self.escape_identifier(table_name)} ({columns_str})
            VALUES ({placeholders})
            ON CONFLICT DO NOTHING
        """  # nosec B608

        # Convert dicts to tuples
        params_list = [tuple(row.get(col) for col in columns) for row in data]

        result = self.execute_many(query, params_list, connection)
        return result.rowcount
