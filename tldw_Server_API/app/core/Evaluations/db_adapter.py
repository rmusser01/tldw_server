"""
Database adapter interface for Evaluations module.

Provides an abstraction layer to support multiple database backends
(SQLite, PostgreSQL, etc.) without changing application code.
"""

import sqlite3
import threading
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from tldw_Server_API.app.core.DB_Management.sqlite_policy import (
    configure_sqlite_connection,
)
from tldw_Server_API.app.core.DB_Management.backends.base import (
    BackendType as UnifiedBackendType,
    DatabaseBackend as UnifiedDatabaseBackend,
)
from tldw_Server_API.app.core.DB_Management.backends.query_utils import (
    prepare_backend_many_statement,
    prepare_backend_statement,
)


class DatabaseType(Enum):
    """Supported database types."""
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"


@dataclass
class DatabaseConfig:
    """Database configuration."""
    db_type: DatabaseType
    connection_string: str
    pool_size: int = 10
    max_overflow: int = 20
    echo: bool = False
    options: dict[str, Any] = None

    def __post_init__(self):
        if self.options is None:
            self.options = {}


class DatabaseAdapter(ABC):
    """Abstract base class for database adapters."""

    @abstractmethod
    def execute(self, query: str, params: Optional[tuple] = None) -> Any:
        """Execute a query and return results."""
        pass

    @abstractmethod
    def execute_many(self, query: str, params_list: list[tuple]) -> Any:
        """Execute a query with multiple parameter sets."""
        pass

    @abstractmethod
    def fetch_one(self, query: str, params: Optional[tuple] = None) -> Optional[dict]:
        """Fetch a single row as a dictionary."""
        pass

    @abstractmethod
    def fetch_all(self, query: str, params: Optional[tuple] = None) -> list[dict]:
        """Fetch all rows as a list of dictionaries."""
        pass

    @abstractmethod
    def fetch_value(self, query: str, params: Optional[tuple] = None) -> Any:
        """Fetch a single value from the first row."""
        pass

    @abstractmethod
    @contextmanager
    def transaction(self):
        """Context manager for database transactions."""
        pass

    @abstractmethod
    def init_schema(self, schema_sql: str):
        """Initialize database schema."""
        pass

    @abstractmethod
    def insert(self, query: str, params: Optional[tuple] = None) -> int:
        """Execute an insert query and return the last inserted row id."""
        pass

    @abstractmethod
    def update(self, query: str, params: Optional[tuple] = None) -> int:
        """Execute an update query and return number of affected rows."""
        pass

    @abstractmethod
    def close(self):
        """Close database connection."""
        pass


class SQLiteAdapter(DatabaseAdapter):
    """SQLite database adapter."""

    def __init__(self, config: DatabaseConfig):
        """Initialize SQLite adapter."""
        self.config = config
        self.db_path = config.connection_string

        # Ensure database directory exists
        db_path = Path(self.db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize connection with optimal settings
        self._init_connection()

    def _init_connection(self):
        """Initialize SQLite connection with optimal settings."""
        self.conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False,  # Allow sharing between threads
            timeout=30.0
        )
        self.conn.row_factory = sqlite3.Row  # Return rows as dictionaries

        configure_sqlite_connection(self.conn, busy_timeout_ms=30000)
        self.conn.execute("PRAGMA mmap_size=268435456")

        self.conn.commit()

    def execute(self, query: str, params: Optional[tuple] = None) -> Any:
        """Execute a query and return cursor."""
        cursor = self.conn.cursor()
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        return cursor

    def execute_many(self, query: str, params_list: list[tuple]) -> Any:
        """Execute a query with multiple parameter sets."""
        cursor = self.conn.cursor()
        return cursor.executemany(query, params_list)

    def fetch_one(self, query: str, params: Optional[tuple] = None) -> Optional[dict]:
        """Fetch a single row as a dictionary."""
        cursor = self.execute(query, params)
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None

    def fetch_all(self, query: str, params: Optional[tuple] = None) -> list[dict]:
        """Fetch all rows as a list of dictionaries."""
        cursor = self.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

    def fetch_value(self, query: str, params: Optional[tuple] = None) -> Any:
        """Fetch a single value from the first row."""
        cursor = self.execute(query, params)
        row = cursor.fetchone()
        if row:
            return row[0]
        return None

    @contextmanager
    def transaction(self):
        """Context manager for database transactions."""
        try:
            yield self
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def init_schema(self, schema_sql: str):
        """Initialize database schema."""
        # Split schema into individual statements
        statements = schema_sql.split(';')
        for statement in statements:
            statement = statement.strip()
            if statement:
                self.execute(statement)
        self.conn.commit()

    def insert(self, query: str, params: Optional[tuple] = None) -> int:
        """Execute an insert query and return the last inserted row id."""
        cursor = self.conn.cursor()
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        return cursor.lastrowid

    def update(self, query: str, params: Optional[tuple] = None) -> int:
        """Execute an update query and return number of affected rows."""
        cursor = self.conn.cursor()
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        return cursor.rowcount

    def close(self):
        """Close database connection."""
        if hasattr(self, 'conn'):
            self.conn.close()


class BackendAdapter(DatabaseAdapter):
    """Adapter that bridges DatabaseBackend to the Evaluations adapter interface."""

    def __init__(self, backend: UnifiedDatabaseBackend):
        if backend is None:
            raise ValueError("backend must not be None")
        self.backend = backend
        self.backend_type = getattr(backend, "backend_type", None)
        self._local = threading.local()

    def _active_connection(self) -> Any:
        return getattr(self._local, "connection", None)

    def _prepare_statement(
        self,
        query: str,
        params: Optional[tuple] = None,
        *,
        ensure_returning: bool = False,
    ) -> tuple[str, Any]:
        backend_type = getattr(self.backend, "backend_type", None)
        if backend_type == UnifiedBackendType.POSTGRESQL:
            return prepare_backend_statement(
                backend_type,
                query,
                params,
                apply_default_transform=True,
                ensure_returning=ensure_returning,
            )
        return query, params

    def _prepare_many(
        self,
        query: str,
        params_list: list[tuple],
    ) -> tuple[str, list[Any]]:
        backend_type = getattr(self.backend, "backend_type", None)
        if backend_type == UnifiedBackendType.POSTGRESQL:
            return prepare_backend_many_statement(
                backend_type,
                query,
                params_list,
                apply_default_transform=True,
            )
        return query, params_list

    def execute(self, query: str, params: Optional[tuple] = None) -> Any:
        prepared_query, prepared_params = self._prepare_statement(query, params)
        return self.backend.execute(prepared_query, prepared_params, connection=self._active_connection())

    def execute_many(self, query: str, params_list: list[tuple]) -> Any:
        prepared_query, prepared_params = self._prepare_many(query, params_list)
        return self.backend.execute_many(prepared_query, prepared_params, connection=self._active_connection())

    def fetch_one(self, query: str, params: Optional[tuple] = None) -> Optional[dict]:
        result = self.execute(query, params)
        return result.first if getattr(result, "first", None) else None

    def fetch_all(self, query: str, params: Optional[tuple] = None) -> list[dict]:
        result = self.execute(query, params)
        return list(getattr(result, "rows", []) or [])

    def fetch_value(self, query: str, params: Optional[tuple] = None) -> Any:
        result = self.execute(query, params)
        return getattr(result, "scalar", None)

    
    @contextmanager
    def transaction(self):
        existing = self._active_connection()
        if existing is not None:
            yield self
            return

        with self.backend.transaction() as connection:
            self._local.connection = connection
            try:
                yield self
            finally:
                self._local.connection = None

    def init_schema(self, schema_sql: str):
        statements = [statement.strip() for statement in schema_sql.split(";") if statement.strip()]
        with self.transaction():
            for statement in statements:
                self.execute(statement)

    def insert(self, query: str, params: Optional[tuple] = None) -> int:
        prepared_query, prepared_params = self._prepare_statement(
            query,
            params,
            ensure_returning=True,
        )
        result = self.backend.execute(prepared_query, prepared_params, connection=self._active_connection())

        lastrowid = getattr(result, "lastrowid", None)
        if lastrowid is not None:
            try:
                return int(lastrowid)
            except (TypeError, ValueError):
                return 0

        first = getattr(result, "first", None)
        if isinstance(first, dict):
            for key in ("id", "webhook_id"):
                value = first.get(key)
                if value is not None:
                    try:
                        return int(value)
                    except (TypeError, ValueError):
                        return 0

        scalar = getattr(result, "scalar", None)
        try:
            return int(scalar) if scalar is not None else 0
        except (TypeError, ValueError):
            return 0

    def update(self, query: str, params: Optional[tuple] = None) -> int:
        result = self.execute(query, params)
        try:
            return int(getattr(result, "rowcount", 0) or 0)
        except (TypeError, ValueError):
            return 0

    def close(self):
        """Backends are managed by the shared backend pool lifecycle."""
        return None


def create_adapter_from_backend(backend: UnifiedDatabaseBackend) -> DatabaseAdapter:
    """Create an evaluations adapter from a shared content backend instance."""
    return BackendAdapter(backend)


class PostgreSQLAdapter(DatabaseAdapter):
    """PostgreSQL database adapter (stub for future implementation)."""

    def __init__(self, config: DatabaseConfig):
        """Initialize PostgreSQL adapter."""
        self.config = config
        # TODO: Implement PostgreSQL connection using psycopg2 or asyncpg
        raise NotImplementedError("PostgreSQL adapter not yet implemented")

    def execute(self, query: str, params: Optional[tuple] = None) -> Any:
        """Execute a query and return results."""
        # Convert SQLite-style placeholders (?) to PostgreSQL style ($1, $2, etc.)
        raise NotImplementedError()

    def execute_many(self, query: str, params_list: list[tuple]) -> Any:
        """Execute a query with multiple parameter sets."""
        raise NotImplementedError()

    def fetch_one(self, query: str, params: Optional[tuple] = None) -> Optional[dict]:
        """Fetch a single row as a dictionary."""
        raise NotImplementedError()

    def fetch_all(self, query: str, params: Optional[tuple] = None) -> list[dict]:
        """Fetch all rows as a list of dictionaries."""
        raise NotImplementedError()

    def fetch_value(self, query: str, params: Optional[tuple] = None) -> Any:
        """Fetch a single value from the first row."""
        raise NotImplementedError()

    @contextmanager
    def transaction(self):
        """Context manager for database transactions."""
        raise NotImplementedError()

    def init_schema(self, schema_sql: str):
        """Initialize database schema."""
        raise NotImplementedError()

    def insert(self, query: str, params: Optional[tuple] = None) -> int:
        """Execute an insert query and return the last inserted row id."""
        raise NotImplementedError()

    def update(self, query: str, params: Optional[tuple] = None) -> int:
        """Execute an update query and return number of affected rows."""
        raise NotImplementedError()

    def close(self):
        """Close database connection."""
        raise NotImplementedError()


class DatabaseAdapterFactory:
    """Factory for creating database adapters."""

    _adapters = {
        DatabaseType.SQLITE: SQLiteAdapter,
        DatabaseType.POSTGRESQL: PostgreSQLAdapter,
    }

    @classmethod
    def create(cls, config: DatabaseConfig) -> DatabaseAdapter:
        """Create a database adapter based on configuration."""
        adapter_class = cls._adapters.get(config.db_type)
        if not adapter_class:
            raise ValueError(f"Unsupported database type: {config.db_type}")

        return adapter_class(config)

    @classmethod
    def register(cls, db_type: DatabaseType, adapter_class: type):
        """Register a new adapter class."""
        cls._adapters[db_type] = adapter_class


# Global adapter instance (initialized on first use)
_global_adapter: Optional[DatabaseAdapter] = None


def get_database_adapter(config: Optional[DatabaseConfig] = None) -> DatabaseAdapter:
    """Get or create the global database adapter."""
    global _global_adapter

    if _global_adapter is None:
        if config is None:
            # Default to SQLite per-user (single-user ID) for backward compatibility
            from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths as _DP
            db_path = _DP.get_evaluations_db_path(_DP.get_single_user_id())
            db_path.parent.mkdir(parents=True, exist_ok=True)
            config = DatabaseConfig(
                db_type=DatabaseType.SQLITE,
                connection_string=str(db_path)
            )

        _global_adapter = DatabaseAdapterFactory.create(config)
        logger.info(f"Initialized {config.db_type.value} database adapter")

    return _global_adapter


def set_database_adapter(adapter: DatabaseAdapter):
    """Set the global database adapter (mainly for testing)."""
    global _global_adapter
    _global_adapter = adapter


def close_database_adapter():
    """Close and cleanup the global database adapter."""
    global _global_adapter
    if _global_adapter:
        _global_adapter.close()
        _global_adapter = None
