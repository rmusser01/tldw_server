"""
SQLite backend implementation for the database abstraction layer.

This module provides a concrete implementation of the DatabaseBackend
interface for SQLite databases, maintaining compatibility with the
existing codebase while enabling multi-backend support.
"""

import re
import sqlite3
import threading
import time
import urllib.parse as _url
import weakref
from collections.abc import Generator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any, Optional, Union

from loguru import logger as _loguru_logger

from ..sqlite_policy import configure_sqlite_connection
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

logger = _loguru_logger

_NUMERIC_RE = re.compile(r"^-?\d+(?:\.\d+)?$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_QUOTED_IDENTIFIER_RE = re.compile(r'^"[^"]+"$')


def _classify_sqlite_path(db_path: str) -> tuple[bool, bool]:
    """Return (is_memory, use_uri) for a SQLite path/URI."""
    raw = (db_path or "").strip()
    lowered = raw.lower()
    use_uri = lowered.startswith("file:")
    if raw == ":memory:":
        return True, False
    if use_uri and (":memory:" in lowered or "mode=memory" in lowered):
        return True, True
    return False, use_uri


def _sqlite_file_path_from_uri(db_uri: str) -> Optional[Path]:
    """Extract a filesystem path from a file: URI, if present."""
    try:
        parsed = _url.urlparse(db_uri)
    except (TypeError, ValueError):
        return None
    if parsed.scheme != "file":
        return None
    path = _url.unquote(parsed.path or "")
    if not path or path in {":memory:", "/:memory:"}:
        return None
    candidate = Path(path).expanduser()
    try:
        if not candidate.is_absolute():
            return (Path.cwd() / candidate).resolve()
        return candidate.resolve()
    except (OSError, RuntimeError, ValueError):
        return candidate

class SQLiteConnectionPool(ConnectionPool):
    """SQLite-specific connection pool using thread-local storage."""

    def __init__(self, db_path: str, config: DatabaseConfig):
        """
        Initialize SQLite connection pool.

        Args:
            db_path: Path to SQLite database file
            config: Database configuration
        """
        # Normalize to absolute path to avoid CWD-related open errors under tests.
        # Detect in-memory/URI databases and avoid path resolution.
        self._is_memory, self._use_uri = _classify_sqlite_path(db_path)
        try:
            if self._is_memory or self._use_uri:
                self.db_path = db_path
            else:
                self.db_path = str(Path(db_path).resolve())
        except (OSError, RuntimeError, ValueError):
            self.db_path = db_path
        self.config = config
        self._local = threading.local()
        self._connections: dict[int, sqlite3.Connection] = {}
        self._thread_refs: dict[int, weakref.ReferenceType[threading.Thread]] = {}
        self._lock = threading.RLock()
        self._closed = False

    def _prune_dead_threads(self) -> None:
        """Close connections owned by threads that have exited."""
        with self._lock:
            stale_ids: list[int] = []
            for tid, ref in self._thread_refs.items():
                thread_obj = ref()
                if thread_obj is None or not thread_obj.is_alive():
                    stale_ids.append(tid)
            for tid in stale_ids:
                conn = self._connections.pop(tid, None)
                if conn:
                    with suppress(OSError, RuntimeError, sqlite3.Error):
                        conn.close()
                self._thread_refs.pop(tid, None)

    def get_connection(self) -> sqlite3.Connection:
        """Get a thread-local connection."""
        if self._closed:
            raise DatabaseError("Connection pool is closed")
        thread_id = threading.get_ident()
        self._prune_dead_threads()

        # Check if we have a connection for this thread
        if not hasattr(self._local, 'connection') or self._local.connection is None:
            with self._lock:
                if thread_id not in self._connections or self._connections[thread_id] is None:
                    conn = self._create_connection()
                    self._connections[thread_id] = conn
                    self._local.connection = conn
                    self._thread_refs[thread_id] = weakref.ref(threading.current_thread())
                else:
                    self._local.connection = self._connections[thread_id]

        return self._local.connection

    def _create_connection(self) -> sqlite3.Connection:
        """Create a new SQLite connection with optimal settings."""
        # Ensure database directory exists for file-backed DBs
        if not self._is_memory:
            try:
                dbp = _sqlite_file_path_from_uri(self.db_path) if self._use_uri else Path(self.db_path)
                if dbp and dbp.parent and not dbp.parent.exists():
                    dbp.parent.mkdir(parents=True, exist_ok=True)
            except OSError:
                pass

        conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False,
            isolation_level=None,  # Autocommit mode
            uri=self._use_uri,
        )

        # Set row factory for dict-like access
        conn.row_factory = sqlite3.Row

        configure_sqlite_connection(
            conn,
            use_wal=bool(self.config.sqlite_wal_mode),
            synchronous="NORMAL" if self.config.sqlite_wal_mode else None,
            foreign_keys=bool(self.config.sqlite_foreign_keys),
            busy_timeout_ms=10000,
            cache_size=-2000,
        )

        return conn

    def return_connection(self, connection: sqlite3.Connection) -> None:
        """SQLite connections are thread-local, no action needed."""
        pass

    def clear_thread_local_connection(self) -> None:
        """Clear the current thread's connection reference from the pool.

        This provides a safe way for higher layers to invalidate a broken
        connection without reaching into private attributes.
        """
        thread_id = threading.get_ident()
        with self._lock:
            try:
                conn = self._connections.get(thread_id)
                if conn:
                    with suppress(OSError, RuntimeError, sqlite3.Error):
                        conn.close()
                self._connections[thread_id] = None
            except (AttributeError, KeyError, RuntimeError, TypeError):
                pass
            with suppress(AttributeError, KeyError, RuntimeError, TypeError):
                self._thread_refs.pop(thread_id, None)
            try:
                if hasattr(self._local, 'connection'):
                    self._local.connection = None
            except (AttributeError, RuntimeError, TypeError):
                pass
            # Prune stale entries to avoid unbounded growth
            try:
                stale_keys = [tid for tid, conn in self._connections.items() if conn is None]
                for tid in stale_keys:
                    self._connections.pop(tid, None)
            except (AttributeError, RuntimeError, TypeError):
                pass

    @contextmanager
    def connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for connection handling."""
        conn = self.get_connection()
        try:
            yield conn
        except Exception as e:
            logger.exception(f"Error in connection context: {e}")
            raise

    def close_all(self) -> None:
        """Close all connections in the pool."""
        with self._lock:
            self._closed = True
            for conn in self._connections.values():
                if conn:
                    try:
                        conn.close()
                    except (OSError, RuntimeError, sqlite3.Error) as e:
                        logger.exception(f"Error closing connection: {e}")
            self._connections.clear()
            self._thread_refs.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get pool statistics."""
        self._prune_dead_threads()
        with self._lock:
            active = len([c for c in self._connections.values() if c])
            # Keep "active_threads" for backward compatibility; prefer "active_connections"
            return {
                "total_connections": len(self._connections),
                "active_connections": active,
                "active_threads": active,  # deprecated alias
                "closed": self._closed,
                "db_path": self.db_path,
            }


class SQLiteBackend(DatabaseBackend):
    """SQLite implementation of the database backend."""

    def __init__(self, config: DatabaseConfig):
        super().__init__(config)
        self._pool_lock = threading.Lock()
        self._retired = False

    @property
    def backend_type(self) -> BackendType:
        """Get the backend type."""
        return BackendType.SQLITE

    def _get_features(self) -> BackendFeatures:
        """Get SQLite feature support."""
        return BackendFeatures(
            full_text_search=True,  # FTS5
            json_support=True,       # JSON1 extension
            array_support=False,     # No native arrays
            window_functions=True,   # Since 3.25.0
            cte_support=True,        # Common Table Expressions
            partial_indexes=True,    # Since 3.8.0
            generated_columns=True,  # Since 3.31.0
            upsert_support=True,     # INSERT OR REPLACE
            returning_clause=True,   # Since 3.35.0
            listen_notify=False      # No LISTEN/NOTIFY
        )

    def connect(self) -> sqlite3.Connection:
        """Create a new SQLite connection."""
        if not self.config.sqlite_path:
            raise DatabaseError("SQLite path not configured")
        raw_path = self.config.sqlite_path
        # Handle in-memory/URI DB distinctly
        is_memory, use_uri = _classify_sqlite_path(raw_path)

        # Ensure database directory exists for file-backed DBs
        if not is_memory:
            if use_uri:
                db_path = _sqlite_file_path_from_uri(raw_path)
                if db_path is not None:
                    db_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                db_path = Path(raw_path)
                db_path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(
            raw_path if use_uri or is_memory else str(db_path),
            check_same_thread=False,
            isolation_level=None,
            uri=use_uri,
        )

        conn.row_factory = sqlite3.Row

        configure_sqlite_connection(
            conn,
            use_wal=bool(self.config.sqlite_wal_mode),
            synchronous="NORMAL" if self.config.sqlite_wal_mode else None,
            foreign_keys=bool(self.config.sqlite_foreign_keys),
            busy_timeout_ms=10000,
        )

        return conn

    def disconnect(self, connection: sqlite3.Connection) -> None:
        """Close a SQLite connection."""
        if connection:
            connection.close()

    @contextmanager
    def transaction(self, connection: Optional[sqlite3.Connection] = None) -> Generator[sqlite3.Connection, None, None]:
        """SQLite transaction context manager.

        Uses explicit BEGIN IMMEDIATE/COMMIT/ROLLBACK and guards with in_transaction to
        avoid errors when statements (e.g., executescript) implicitly end a txn.
        """
        conn = connection or self.get_pool().get_connection()

        started = False
        try:
            if not getattr(conn, "in_transaction", False):
                conn.execute("BEGIN IMMEDIATE")
                started = True
            yield conn
            if started and getattr(conn, "in_transaction", False):
                conn.execute("COMMIT")
        except Exception as e:
            if started and getattr(conn, "in_transaction", False):
                try:
                    conn.execute("ROLLBACK")
                except sqlite3.OperationalError:
                    # Best effort; ignore if no active transaction
                    pass
            logger.exception(f"Transaction failed: {e}")
            raise

    def get_pool(self) -> ConnectionPool:
        """Get or create the connection pool."""
        if self._retired:
            raise DatabaseError("SQLite backend has been retired")
        if self._pool is None:
            with self._pool_lock:
                if self._retired:
                    raise DatabaseError("SQLite backend has been retired")
                if self._pool is None:
                    if not self.config.sqlite_path:
                        raise DatabaseError("SQLite path not configured")
                    self._pool = SQLiteConnectionPool(self.config.sqlite_path, self.config)
        return self._pool

    def execute(
        self,
        query: str,
        params: Optional[Union[tuple, dict]] = None,
        connection: Optional[sqlite3.Connection] = None
    ) -> QueryResult:
        """Execute a query and return results."""
        start_time = time.time()

        conn = connection or self.get_pool().get_connection()

        try:
            cursor = conn.cursor()

            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)

            # Determine whether to fetch rows: any statement that returns rows
            if cursor.description is not None:
                rows = cursor.fetchall()
                result_rows = [dict(row) for row in rows]
            else:
                result_rows = []

            execution_time = time.time() - start_time

            return QueryResult(
                rows=result_rows,
                rowcount=cursor.rowcount,
                lastrowid=cursor.lastrowid,
                description=cursor.description,
                execution_time=execution_time
            )

        except sqlite3.Error as e:
            logger.exception(f"Query execution failed: {e}")
            raise DatabaseError(f"SQLite error: {e}") from e

    def execute_many(
        self,
        query: str,
        params_list: list[Union[tuple, dict]],
        connection: Optional[sqlite3.Connection] = None
    ) -> QueryResult:
        """Execute a query multiple times with different parameters."""
        start_time = time.time()

        conn = connection or self.get_pool().get_connection()

        try:
            cursor = conn.cursor()
            cursor.executemany(query, params_list)

            execution_time = time.time() - start_time

            return QueryResult(
                rows=[],
                rowcount=cursor.rowcount,
                lastrowid=cursor.lastrowid,
                description=cursor.description,
                execution_time=execution_time
            )

        except sqlite3.Error as e:
            logger.exception(f"Batch execution failed: {e}")
            raise DatabaseError(f"SQLite error: {e}") from e

    def create_tables(self, schema: str, connection: Optional[sqlite3.Connection] = None) -> None:
        """Create tables from a schema definition."""
        conn = connection or self.get_pool().get_connection()

        try:
            # Execute the schema as a script
            conn.executescript(schema)
        except sqlite3.Error as e:
            logger.exception(f"Schema creation failed: {e}")
            raise DatabaseError(f"Failed to create schema: {e}") from e

    def table_exists(self, table_name: str, connection: Optional[sqlite3.Connection] = None) -> bool:
        """Check if a table exists."""
        query = """
            SELECT name FROM sqlite_master
            WHERE type='table' AND name=?
        """
        result = self.execute(query, (table_name,), connection)
        return len(result.rows) > 0

    def get_table_info(
        self,
        table_name: str,
        connection: Optional[sqlite3.Connection] = None
    ) -> list[dict[str, Any]]:
        """Get information about a table's columns."""
        query = f"PRAGMA table_info({self.escape_identifier(table_name)})"
        result = self.execute(query, connection=connection)

        # Convert to standard format
        columns = []
        for row in result.rows:
            columns.append({
                "name": row["name"],
                "type": row["type"],
                "nullable": not row["notnull"],
                "default": row["dflt_value"],
                "primary_key": bool(row["pk"])
            })

        return columns

    def create_fts_table(
        self,
        table_name: str,
        source_table: str,
        columns: list[str],
        connection: Optional[sqlite3.Connection] = None
    ) -> None:
        """Create a FTS5 virtual table."""
        self.features.require("full_text_search")

        # Build FTS5 table creation query
        columns_str = ", ".join([self.escape_identifier(c) for c in columns])
        query = f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS {self.escape_identifier(table_name)}
            USING fts5({columns_str}, content='{source_table}')
        """

        try:
            self.execute(query, connection=connection)

            self._ensure_fts_triggers(
                table_name=table_name,
                source_table=source_table,
                columns=columns,
                connection=connection,
            )
            # Rebuild index from content table to avoid duplicate rowids on re-init.
            fts_table_ident = self.escape_identifier(table_name)
            rebuild_query_template = "INSERT INTO {table}({table}) VALUES('rebuild')"
            rebuild_query = rebuild_query_template.format(table=fts_table_ident)  # nosec B608
            self.execute(rebuild_query, connection=connection)

        except sqlite3.Error as e:
            logger.exception(f"FTS table creation failed: {e}")
            raise DatabaseError(f"Failed to create FTS table: {e}") from e

    def _ensure_fts_triggers(
        self,
        *,
        table_name: str,
        source_table: str,
        columns: list[str],
        connection: Optional[sqlite3.Connection] = None,
    ) -> None:
        """Create FTS sync triggers for external-content tables (SQLite)."""
        if not columns:
            return

        fts_table_ident = self.escape_identifier(table_name)
        source_table_ident = self.escape_identifier(source_table)
        columns_ident = [self.escape_identifier(col) for col in columns]
        columns_str = ", ".join(columns_ident)
        new_values = ", ".join([f"new.{self.escape_identifier(col)}" for col in columns])
        old_values = ", ".join([f"old.{self.escape_identifier(col)}" for col in columns])

        trigger_base = f"{table_name}_fts_sync"
        insert_trigger = self.escape_identifier(f"{trigger_base}_ai")
        update_trigger = self.escape_identifier(f"{trigger_base}_au")
        delete_trigger = self.escape_identifier(f"{trigger_base}_ad")
        fts_column_ident = self.escape_identifier(table_name)

        insert_sql_template = """
        CREATE TRIGGER IF NOT EXISTS {insert_trigger}
        AFTER INSERT ON {source_table_ident} BEGIN
            INSERT INTO {fts_table_ident}(rowid, {columns_str})
            VALUES (new.rowid, {new_values});
        END;
        """
        insert_sql = insert_sql_template.format_map(locals())  # nosec B608
        delete_sql_template = """
        CREATE TRIGGER IF NOT EXISTS {delete_trigger}
        AFTER DELETE ON {source_table_ident} BEGIN
            INSERT INTO {fts_table_ident}({fts_column_ident}, rowid, {columns_str})
            VALUES ('delete', old.rowid, {old_values});
        END;
        """
        delete_sql = delete_sql_template.format_map(locals())  # nosec B608
        update_sql_template = """
        CREATE TRIGGER IF NOT EXISTS {update_trigger}
        AFTER UPDATE ON {source_table_ident} BEGIN
            INSERT INTO {fts_table_ident}({fts_column_ident}, rowid, {columns_str})
            VALUES ('delete', old.rowid, {old_values});
            INSERT INTO {fts_table_ident}(rowid, {columns_str})
            VALUES (new.rowid, {new_values});
        END;
        """
        update_sql = update_sql_template.format_map(locals())  # nosec B608

        self.execute(insert_sql, connection=connection)
        self.execute(delete_sql, connection=connection)
        self.execute(update_sql, connection=connection)

    def fts_search(
        self,
        fts_query: FTSQuery,
        connection: Optional[sqlite3.Connection] = None
    ) -> QueryResult:
        """Perform a FTS5 search."""
        self.features.require("full_text_search")

        if not fts_query.table:
            raise DatabaseError("FTS table name required")

        # Build the FTS query
        query_parts = [f"SELECT * FROM {self.escape_identifier(fts_query.table)}"]  # nosec B608
        params = []

        # Add MATCH clause
        query_parts.append(f"WHERE {self.escape_identifier(fts_query.table)} MATCH ?")
        normalized_query = (
            FTSQueryTranslator.normalize_query(fts_query.query_text, "sqlite")
            or fts_query.query_text
        )
        params.append(normalized_query)

        # Add additional filters
        for key, value in fts_query.filters.items():
            query_parts.append(f"AND {self.escape_identifier(key)} = ?")
            params.append(value)

        # Add ORDER BY using bm25() by default for better relevance
        rank_expression = self._safe_fts_rank_expression(fts_query)
        if fts_query.rank_expression and rank_expression is None:
            logger.warning(
                'Ignoring unsafe FTS rank_expression for table {}',
                fts_query.table,
            )
        if rank_expression:
            query_parts.append(f"ORDER BY {rank_expression}")
        else:
            # bm25 returns lower scores for more relevant rows; sort ASC
            query_parts.append(
                f"ORDER BY bm25({self.escape_identifier(fts_query.table)}) ASC"
            )

        # Add LIMIT/OFFSET
        if fts_query.limit:
            query_parts.append(f"LIMIT {fts_query.limit}")
        if fts_query.offset:
            query_parts.append(f"OFFSET {fts_query.offset}")

        query = " ".join(query_parts)

        return self.execute(query, tuple(params), connection)

    def update_fts_index(
        self,
        table_name: str,
        connection: Optional[sqlite3.Connection] = None
    ) -> None:
        """Update the FTS5 index (rebuild if needed)."""
        query = f"INSERT INTO {self.escape_identifier(table_name)}({self.escape_identifier(table_name)}) VALUES('rebuild')"  # nosec B608
        self.execute(query, connection=connection)

    def escape_identifier(self, identifier: str) -> str:
        """Escape a SQLite identifier."""
        # SQLite uses double quotes for identifiers
        escaped = identifier.replace('"', '""')
        return f'"{escaped}"'

    def _safe_fts_rank_expression(self, fts_query: FTSQuery) -> Optional[str]:
        expr = (fts_query.rank_expression or "").strip()
        if not expr or not fts_query.table:
            return None

        direction = None
        parts = expr.rsplit(None, 1)
        if len(parts) == 2 and parts[1].upper() in {"ASC", "DESC"}:
            expr_core = parts[0].strip()
            direction = parts[1].upper()
        else:
            expr_core = expr

        bm25_match = re.match(r"^bm25(?:\((.*)\))?$", expr_core, re.IGNORECASE)
        if bm25_match:
            inner = (bm25_match.group(1) or "").strip()
            weights: list[str] = []
            if inner:
                tokens = [token.strip() for token in inner.split(",") if token.strip()]
                if tokens and (_IDENTIFIER_RE.match(tokens[0]) or _QUOTED_IDENTIFIER_RE.match(tokens[0])):
                    tokens = tokens[1:]
                for token in tokens:
                    if not _NUMERIC_RE.fullmatch(token):
                        return None
                weights = tokens

            expr_safe = f"bm25({self.escape_identifier(fts_query.table)}"
            if weights:
                expr_safe += ", " + ", ".join(weights)
            expr_safe += ")"
            if direction:
                expr_safe += f" {direction}"
            return expr_safe

        if _IDENTIFIER_RE.match(expr_core):
            expr_safe = self.escape_identifier(expr_core)
            if direction:
                expr_safe += f" {direction}"
            return expr_safe

        return None

    def get_last_insert_id(self, connection: Optional[sqlite3.Connection] = None) -> Optional[int]:
        """Get the last inserted row ID."""
        result = self.execute("SELECT last_insert_rowid()", connection=connection)
        return result.scalar

    def vacuum(self, connection: Optional[sqlite3.Connection] = None) -> None:
        """Vacuum the SQLite database."""
        self.execute("VACUUM", connection=connection)

    def get_database_size(self, connection: Optional[sqlite3.Connection] = None) -> int:
        """Get the database size in bytes."""
        if not self.config.sqlite_path:
            return 0

        is_memory, use_uri = _classify_sqlite_path(self.config.sqlite_path)
        if is_memory:
            return 0
        db_path = _sqlite_file_path_from_uri(self.config.sqlite_path) if use_uri else Path(self.config.sqlite_path)
        if db_path and db_path.exists():
            return db_path.stat().st_size
        return 0

    def export_schema(self, connection: Optional[sqlite3.Connection] = None) -> str:
        """Export the database schema as SQL."""
        query = """
            SELECT sql FROM sqlite_master
            WHERE type IN ('table', 'index', 'trigger', 'view')
            AND sql IS NOT NULL
            ORDER BY type, name
        """
        result = self.execute(query, connection=connection)

        schema_parts = []
        for row in result.rows:
            if row["sql"]:
                schema_parts.append(row["sql"] + ";")

        return "\n\n".join(schema_parts)

    def export_data(
        self,
        table_name: str,
        connection: Optional[sqlite3.Connection] = None
    ) -> Generator[dict[str, Any], None, None]:
        """Export data from a table."""
        query = f"SELECT * FROM {self.escape_identifier(table_name)}"  # nosec B608

        conn = connection or self.get_pool().get_connection()

        cursor = conn.cursor()
        cursor.execute(query)

        # Get column names
        columns = [desc[0] for desc in cursor.description]

        # Yield rows as dictionaries
        for row in cursor:
            yield dict(zip(columns, row))

    def import_data(
        self,
        table_name: str,
        data: list[dict[str, Any]],
        connection: Optional[sqlite3.Connection] = None
    ) -> int:
        """Import data into a table."""
        if not data:
            return 0

        # Get column names from first row
        columns = list(data[0].keys())
        columns_str = ", ".join([self.escape_identifier(col) for col in columns])
        placeholders = ", ".join(["?" for _ in columns])

        query = f"""
            INSERT OR REPLACE INTO {self.escape_identifier(table_name)} ({columns_str})
            VALUES ({placeholders})
        """

        # Convert dicts to tuples
        params_list = [tuple(row.get(col) for col in columns) for row in data]

        result = self.execute_many(query, params_list, connection)
        return result.rowcount
