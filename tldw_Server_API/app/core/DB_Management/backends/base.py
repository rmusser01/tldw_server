"""
Abstract base classes for database backend implementations.

This module defines the interface that all database backends must implement
to ensure compatibility with the application.
"""

import threading
from abc import ABC, abstractmethod
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Union

from tldw_Server_API.app.core.testing import is_truthy


class DatabaseError(Exception):
    """Base exception for database-related errors."""
    pass


class NotSupportedError(DatabaseError):
    """Raised when a feature is not supported by the backend."""
    pass


class BackendType(Enum):
    """Supported database backend types."""
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"  # Future support


_BACKEND_ALIASES = {
    "postgres": "postgresql",
    "postgresql": "postgresql",
    "sqlite3": "sqlite",
    "sqlite": "sqlite",
}


def normalize_backend_name(value: Optional[str]) -> Optional[str]:
    """Normalize backend identifiers (accept common aliases)."""
    if value is None:
        return None
    cleaned = str(value).strip().lower()
    if not cleaned:
        return cleaned
    return _BACKEND_ALIASES.get(cleaned, cleaned)


@dataclass
class BackendFeatures:
    """Features supported by a database backend."""
    full_text_search: bool = False
    json_support: bool = False
    array_support: bool = False
    window_functions: bool = False
    cte_support: bool = False
    partial_indexes: bool = False
    generated_columns: bool = False
    upsert_support: bool = False
    returning_clause: bool = False
    listen_notify: bool = False

    def require(self, feature: str) -> None:
        """Check if a feature is supported, raise if not."""
        if not getattr(self, feature, False):
            raise NotSupportedError(f"Feature '{feature}' is not supported by this backend")


@dataclass
class DatabaseConfig:
    """Configuration for a database backend."""
    backend_type: BackendType
    connection_string: Optional[str] = None
    # Optional client identifier for logging/telemetry correlation (non-functional)
    client_id: Optional[str] = None

    # Connection pool settings
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: float = 30.0
    pool_recycle: int = 3600

    # SQLite specific
    sqlite_path: Optional[str] = None
    sqlite_wal_mode: bool = True
    sqlite_foreign_keys: bool = True

    # PostgreSQL specific
    pg_host: Optional[str] = None
    pg_port: int = 5432
    pg_database: Optional[str] = None
    pg_user: Optional[str] = None
    pg_password: Optional[str] = None
    pg_sslmode: str = "prefer"

    # Common settings
    echo: bool = False
    isolation_level: Optional[str] = None
    connect_timeout: int = 10

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items() if v is not None}

    # Backwards-compatible helpers expected by some modules/tests
    @property
    def backend(self) -> BackendType:
        """Alias for backend_type for older call sites."""
        return self.backend_type

    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """
        Build a DatabaseConfig from common environment variables.

        Supports:
          - DATABASE_URL (postgresql://..., postgres://..., sqlite:///path, sqlite:///:memory:)
          - TLDW_DB_BACKEND ("sqlite" | "postgresql" | "postgres") and related TLDW_* vars
          - PG* variables (PGHOST, PGPORT, PGDATABASE, PGUSER, PGPASSWORD)
        """
        import os
        import urllib.parse as _url

        db_url = os.getenv("DATABASE_URL", "").strip()
        if db_url:
            parsed = _url.urlparse(db_url)
            scheme = (parsed.scheme or "").lower()
            base_scheme = scheme.split("+", 1)[0]
            # Normalize common aliases
            if base_scheme in {"postgres", "postgresql"}:
                cfg = cls(backend_type=BackendType.POSTGRESQL)
                cfg.connection_string = db_url
                cfg.pg_host = parsed.hostname or "localhost"
                try:
                    cfg.pg_port = int(parsed.port or 5432)
                except Exception:
                    cfg.pg_port = 5432
                cfg.pg_database = (parsed.path or "/").lstrip("/") or None
                cfg.pg_user = parsed.username or None
                cfg.pg_password = parsed.password or None
                # sslmode from query if present
                q = _url.parse_qs(parsed.query or "")
                if "sslmode" in q and q["sslmode"]:
                    cfg.pg_sslmode = q["sslmode"][0]
                return cfg
            elif base_scheme in {"sqlite", "file"}:
                # sqlite:///absolute/path, sqlite:///:memory:, or file: URIs
                cfg = cls(backend_type=BackendType.SQLITE)
                cfg.connection_string = db_url
                raw_path = (parsed.path or "")
                netloc = parsed.netloc or ""
                combined = f"{netloc}{raw_path}" if netloc else raw_path
                combined = _url.unquote(combined or "")
                query = parsed.query or ""
                if combined in {":memory:", "/:memory:"}:
                    if query:
                        cfg.sqlite_path = f"file::memory:?{query}"
                    else:
                        cfg.sqlite_path = ":memory:"
                elif combined.startswith("/./"):
                    cfg.sqlite_path = combined[1:]
                else:
                    # Normalize file: URIs (sqlite:///file:... or file:...)
                    if combined.startswith("/file:"):
                        combined = combined[1:]
                    if combined.startswith("file:"):
                        cfg.sqlite_path = combined + (f"?{query}" if query else "")
                    elif query:
                        # Preserve URI query args via file: prefix for sqlite3.connect(uri=True)
                        normalized = combined
                        if normalized.startswith("//"):
                            normalized = "/" + normalized.lstrip("/")
                        cfg.sqlite_path = f"file:{normalized}?{query}"
                    elif combined:
                        cfg.sqlite_path = combined
                    else:
                        try:
                            from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
                            cfg.sqlite_path = str(DatabasePaths.get_media_db_path(DatabasePaths.get_single_user_id()))
                        except Exception as exc:
                            raise ValueError("Failed to resolve default SQLite path via DatabasePaths") from exc
                return cfg
            # Fallback to TLDW_* handling if unknown scheme

        # TLDW_* environment style
        backend_env = normalize_backend_name(os.getenv("TLDW_DB_BACKEND", "sqlite"))
        backend_env = (backend_env or "sqlite").lower()
        try:
            backend_type = BackendType(backend_env)
        except ValueError:
            backend_type = BackendType.SQLITE

        cfg = cls(backend_type=backend_type)
        # Common settings
        try:
            cfg.pool_size = int(os.getenv("TLDW_DB_POOL_SIZE", "10"))
        except Exception:
            cfg.pool_size = 10
        try:
            cfg.pool_timeout = float(os.getenv("TLDW_DB_POOL_TIMEOUT", "30.0"))
        except Exception:
            cfg.pool_timeout = 30.0
        cfg.echo = is_truthy(os.getenv("TLDW_DB_ECHO", "false"))

        if backend_type == BackendType.SQLITE:
            if os.getenv("TLDW_SQLITE_PATH"):
                cfg.sqlite_path = os.getenv("TLDW_SQLITE_PATH")
            else:
                try:
                    from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
                    cfg.sqlite_path = str(DatabasePaths.get_media_db_path(DatabasePaths.get_single_user_id()))
                except Exception as exc:
                    raise ValueError("Failed to resolve default SQLite path via DatabasePaths") from exc
            cfg.sqlite_wal_mode = is_truthy(os.getenv("TLDW_SQLITE_WAL_MODE", "true"))
            cfg.sqlite_foreign_keys = is_truthy(os.getenv("TLDW_SQLITE_FOREIGN_KEYS", "true"))
        elif backend_type == BackendType.POSTGRESQL:
            # Prefer explicit PG* envs when DATABASE_URL not set
            cfg.pg_host = os.getenv("TLDW_PG_HOST") or os.getenv("PGHOST") or "localhost"
            try:
                cfg.pg_port = int(os.getenv("TLDW_PG_PORT") or os.getenv("PGPORT") or "5432")
            except Exception:
                cfg.pg_port = 5432
            cfg.pg_database = os.getenv("TLDW_PG_DATABASE") or os.getenv("PGDATABASE") or "tldw"
            cfg.pg_user = os.getenv("TLDW_PG_USER") or os.getenv("PGUSER") or None
            cfg.pg_password = os.getenv("TLDW_PG_PASSWORD") or os.getenv("PGPASSWORD") or None
            cfg.pg_sslmode = os.getenv("TLDW_PG_SSLMODE") or os.getenv("PGSSLMODE") or cfg.pg_sslmode

        return cfg


@dataclass
class QueryResult:
    """Result of a database query."""
    rows: list[dict[str, Any]]
    rowcount: int
    lastrowid: Optional[int] = None
    description: Optional[list[tuple]] = None
    execution_time: Optional[float] = None

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self):
        return iter(self.rows)

    def __getitem__(self, index):
        return self.rows[index]

    @property
    def first(self) -> Optional[dict[str, Any]]:
        """Get the first row or None."""
        return self.rows[0] if self.rows else None

    @property
    def one(self) -> dict[str, Any]:
        """Get exactly one row, raise if not exactly one."""
        if len(self.rows) != 1:
            raise DatabaseError(f"Expected 1 row, got {len(self.rows)}")
        return self.rows[0]

    @property
    def scalar(self) -> Any:
        """Get the first column of the first row."""
        if not self.rows:
            return None
        first_row = self.rows[0]
        if not first_row:
            return None
        return next(iter(first_row.values()))


@dataclass
class FTSQuery:
    """Full-text search query representation."""
    # Public attribute name expected by tests and callers
    query: str
    columns: list[str] = field(default_factory=list)
    table: Optional[str] = None
    limit: Optional[int] = None
    offset: Optional[int] = None
    highlight_config: Optional[dict[str, Any]] = None
    rank_expression: Optional[str] = None
    filters: dict[str, Any] = field(default_factory=dict)

    # Backward-compat alias used by some backends
    @property
    def query_text(self) -> str:
        return self.query


class ConnectionPool(ABC):
    """Abstract connection pool interface."""

    @abstractmethod
    def get_connection(self) -> Any:
        """Get a connection from the pool."""
        pass

    @abstractmethod
    def return_connection(self, connection: Any) -> None:
        """Return a connection to the pool."""
        pass

    @abstractmethod
    @contextmanager
    def connection(self) -> Generator[Any, None, None]:
        """Context manager for connection handling."""
        pass

    @abstractmethod
    def close_all(self) -> None:
        """Close all connections in the pool."""
        pass

    @abstractmethod
    def get_stats(self) -> dict[str, Any]:
        """Get pool statistics."""
        pass


class DatabaseBackend(ABC):
    """
    Abstract base class for database backends.

    All database backends must implement this interface to ensure
    compatibility with the application.
    """

    def __init__(self, config: DatabaseConfig):
        """
        Initialize the database backend.

        Args:
            config: Database configuration
        """
        self.config = config
        self._features = self._get_features()
        self._pool: Optional[ConnectionPool] = None
        self._local = threading.local()

    @abstractmethod
    def _get_features(self) -> BackendFeatures:
        """Get the features supported by this backend."""
        pass

    @property
    def features(self) -> BackendFeatures:
        """Get backend features."""
        return self._features

    @property
    @abstractmethod
    def backend_type(self) -> BackendType:
        """Get the backend type."""
        pass

    # Connection Management

    @abstractmethod
    def connect(self) -> Any:
        """Create a new database connection."""
        pass

    @abstractmethod
    def disconnect(self, connection: Any) -> None:
        """Close a database connection."""
        pass

    @abstractmethod
    @contextmanager
    def transaction(self, connection: Optional[Any] = None) -> Generator[Any, None, None]:
        """
        Transaction context manager.

        Args:
            connection: Optional existing connection to use

        Yields:
            Connection object for the transaction
        """
        pass

    @abstractmethod
    def get_pool(self) -> ConnectionPool:
        """Get or create the connection pool."""
        pass

    # Query Execution

    @abstractmethod
    def execute(
        self,
        query: str,
        params: Optional[Union[tuple, dict]] = None,
        connection: Optional[Any] = None,
        *,
        log_errors: bool = True,
    ) -> QueryResult:
        """
        Execute a query and return results.

        Args:
            query: SQL query to execute
            params: Query parameters
            connection: Optional connection to use
            log_errors: Whether backend errors may include raw driver details

        Returns:
            QueryResult object
        """
        pass

    @abstractmethod
    def execute_many(
        self,
        query: str,
        params_list: list[Union[tuple, dict]],
        connection: Optional[Any] = None
    ) -> QueryResult:
        """
        Execute a query multiple times with different parameters.

        Args:
            query: SQL query to execute
            params_list: List of parameter sets
            connection: Optional connection to use

        Returns:
            QueryResult object
        """
        pass

    # Schema Management

    @abstractmethod
    def create_tables(self, schema: str, connection: Optional[Any] = None) -> None:
        """
        Create tables from a schema definition.

        Args:
            schema: SQL schema definition
            connection: Optional connection to use
        """
        pass

    @abstractmethod
    def table_exists(self, table_name: str, connection: Optional[Any] = None) -> bool:
        """
        Check if a table exists.

        Args:
            table_name: Name of the table
            connection: Optional connection to use

        Returns:
            True if table exists, False otherwise
        """
        pass

    @abstractmethod
    def get_table_info(
        self,
        table_name: str,
        connection: Optional[Any] = None
    ) -> list[dict[str, Any]]:
        """
        Get information about a table's columns.

        Args:
            table_name: Name of the table
            connection: Optional connection to use

        Returns:
            List of column information dictionaries
        """
        pass

    # Full-Text Search

    @abstractmethod
    def create_fts_table(
        self,
        table_name: str,
        source_table: str,
        columns: list[str],
        connection: Optional[Any] = None
    ) -> None:
        """
        Create a full-text search table.

        Args:
            table_name: Name for the FTS table
            source_table: Source table to index
            columns: Columns to include in FTS
            connection: Optional connection to use
        """
        pass

    @abstractmethod
    def fts_search(
        self,
        fts_query: FTSQuery,
        connection: Optional[Any] = None
    ) -> QueryResult:
        """
        Perform a full-text search.

        Args:
            fts_query: Full-text search query configuration
            connection: Optional connection to use

        Returns:
            QueryResult with search results
        """
        pass

    @abstractmethod
    def update_fts_index(
        self,
        table_name: str,
        connection: Optional[Any] = None
    ) -> None:
        """
        Update the full-text search index.

        Args:
            table_name: FTS table to update
            connection: Optional connection to use
        """
        pass

    # Utility Methods

    @abstractmethod
    def escape_identifier(self, identifier: str) -> str:
        """
        Escape a database identifier (table/column name).

        Args:
            identifier: Identifier to escape

        Returns:
            Escaped identifier
        """
        pass

    @abstractmethod
    def get_last_insert_id(self, connection: Optional[Any] = None) -> Optional[int]:
        """
        Get the last inserted row ID.

        Args:
            connection: Optional connection to use

        Returns:
            Last insert ID or None
        """
        pass

    @abstractmethod
    def vacuum(self, connection: Optional[Any] = None) -> None:
        """
        Vacuum/optimize the database.

        Args:
            connection: Optional connection to use
        """
        pass

    @abstractmethod
    def get_database_size(self, connection: Optional[Any] = None) -> int:
        """
        Get the database size in bytes.

        Args:
            connection: Optional connection to use

        Returns:
            Database size in bytes
        """
        pass

    # Scope/Session helpers
    def apply_scope(self, connection: Optional[Any] = None) -> None:
        """Apply backend-specific session scope settings (no-op by default).

        Backends that support row-level security or session-scoped settings
        can override this to (re)apply the current request/user scope to a
        borrowed connection. SQLite backends typically do nothing.
        """
        return None

    # Migration Support

    @abstractmethod
    def export_schema(self, connection: Optional[Any] = None) -> str:
        """
        Export the database schema as SQL.

        Args:
            connection: Optional connection to use

        Returns:
            SQL schema definition
        """
        pass

    @abstractmethod
    def export_data(
        self,
        table_name: str,
        connection: Optional[Any] = None
    ) -> Generator[dict[str, Any], None, None]:
        """
        Export data from a table.

        Args:
            table_name: Table to export
            connection: Optional connection to use

        Yields:
            Row dictionaries
        """
        pass

    @abstractmethod
    def import_data(
        self,
        table_name: str,
        data: list[dict[str, Any]],
        connection: Optional[Any] = None
    ) -> int:
        """
        Import data into a table.

        Args:
            table_name: Table to import into
            data: List of row dictionaries
            connection: Optional connection to use

        Returns:
            Number of rows imported
        """
        pass
