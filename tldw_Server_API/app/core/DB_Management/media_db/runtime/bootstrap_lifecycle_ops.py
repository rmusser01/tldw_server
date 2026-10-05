"""Bootstrap lifecycle helpers owned by the package-native Media DB runtime."""

from __future__ import annotations

import os
import sqlite3
import threading
from collections.abc import Callable
from configparser import ConfigParser
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from types import MethodType
from typing import Any

from tldw_Server_API.app.core.DB_Management.backends.base import (
    BackendType,
    DatabaseBackend,
    DatabaseConfig,
)
from tldw_Server_API.app.core.DB_Management.backends.base import (
    DatabaseError as BackendDatabaseError,
)
from tldw_Server_API.app.core.DB_Management.backends.sqlite_backend import SQLiteBackend
from tldw_Server_API.app.core.DB_Management.media_db.errors import (
    DatabaseError,
    SchemaError,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime import backend_resolution
from tldw_Server_API.app.core.DB_Management.media_db.runtime.connection_lifecycle import (
    close_connection,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.noncritical import (
    MEDIA_NONCRITICAL_EXCEPTIONS,
)

try:
    from loguru import logger

    logging = logger
except ImportError:  # pragma: no cover - defensive fallback
    import logging as _stdlib_logging

    logger = _stdlib_logging.getLogger("media_db_bootstrap_lifecycle")
    logging = logger

_MEDIA_NONCRITICAL_EXCEPTIONS: tuple[type[BaseException], ...] = MEDIA_NONCRITICAL_EXCEPTIONS


def _resolve_existing_only_backend(
    *, backend: DatabaseBackend | None, config: ConfigParser | None
) -> DatabaseBackend | None:
    """Resolve PostgreSQL routing without ever retaining a creating SQLite backend."""
    if backend is not None:
        return backend
    parser = config
    if parser is None:
        try:
            parser = backend_resolution.load_comprehensive_config()
        except _MEDIA_NONCRITICAL_EXCEPTIONS:
            parser = None

    mode = (os.getenv("CONTENT_DB_MODE") or os.getenv("TLDW_CONTENT_DB_BACKEND") or "").strip().lower()
    postgres = mode in {"postgres", "postgresql"}
    if not postgres and parser is not None:
        try:
            postgres = backend_resolution.load_content_db_settings(parser).backend_type == BackendType.POSTGRESQL
        except _MEDIA_NONCRITICAL_EXCEPTIONS:
            pass
    if not postgres:
        return None
    if parser is None:
        raise DatabaseError("PostgreSQL content backend requested but configuration could not be loaded")
    resolved = backend_resolution.get_content_backend(parser)
    if resolved is None or resolved.backend_type != BackendType.POSTGRESQL:
        raise DatabaseError("PostgreSQL content backend requested but could not be initialized")
    return resolved


def _close_existing_only_database(self: Any) -> None:
    """Release the session and close its exclusively owned SQLite pool."""
    if self._get_txn_conn() is not None:
        return
    try:
        close_connection(self)
    finally:
        self.backend.get_pool().close_all()


def _existing_only_query(self: Any, execute: Callable[..., Any]) -> Callable[..., Any]:
    """Keep nontransactional queries off the legacy creating/ephemeral SQLite path."""
    @wraps(execute)
    def execute_with_existing_connection(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("connection") is None:
            kwargs["connection"] = self.get_connection()
        return execute(*args, **kwargs)

    return execute_with_existing_connection


def _open_existing_sqlite_database(self: Any) -> None:
    """Eagerly verify a private noncreating connection without touching the schema."""
    if self.is_memory_db:
        raise ValueError("existing_only requires a file-backed SQLite database")
    self.db_path.stat()
    self.backend = SQLiteBackend(
        DatabaseConfig(
            backend_type=BackendType.SQLITE,
            sqlite_path=self.db_path.as_uri() + "?mode=rw",
        )
    )
    self.close_connection = MethodType(_close_existing_only_database, self)
    pool = self.backend.get_pool()
    try:
        conn = pool.get_connection()
        conn.execute("PRAGMA schema_version").fetchone()
    except sqlite3.Error as exc:
        pool.close_all()
        # Only CANTOPEN plus a verified ENOENT can identify a deletion/open race.
        if getattr(exc, "sqlite_errorcode", None) == sqlite3.SQLITE_CANTOPEN:
            try:
                self.db_path.stat()
            except FileNotFoundError as missing:
                raise missing from exc
            except OSError:
                pass
        raise
    except BaseException:
        pool.close_all()
        raise
    self.execute_query = _existing_only_query(self, self.execute_query)
    self.execute_many = _existing_only_query(self, self.execute_many)


def initialize_media_database(
    self: Any,
    db_path: str | Path,
    client_id: str,
    *,
    backend: DatabaseBackend | None = None,
    config: ConfigParser | None = None,
    default_org_id: int | None = None,
    default_team_id: int | None = None,
    existing_only: bool = False,
) -> None:
    """Initialize a Media DB, skipping creation/bootstrap in existing-only mode."""

    if isinstance(db_path, str) and db_path.strip() == "":
        raise ValueError("db_path cannot be an empty string; pass an explicit path or ':memory:'")  # noqa: TRY003

    if isinstance(db_path, Path):
        self.is_memory_db = False
        self.db_path = db_path.resolve()
    else:
        self.is_memory_db = db_path == ":memory:"
        if self.is_memory_db:
            self.db_path = Path(":memory:")
        else:
            self.db_path = Path(db_path).resolve()

    self.db_path_str = str(self.db_path) if not self.is_memory_db else ":memory:"

    if not client_id:
        raise ValueError("Client ID cannot be empty or None.")  # noqa: TRY003
    self.client_id = client_id
    self.existing_only = existing_only

    if not self.is_memory_db and not existing_only:
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise DatabaseError(
                f"Failed to create database directory {self.db_path.parent}: {exc}"
            ) from exc  # noqa: TRY003

    logging.info(
        f"Initializing Database object for path: {self.db_path_str} [Client ID: {self.client_id}]"
    )

    if existing_only:
        self.backend = _resolve_existing_only_backend(backend=backend, config=config)
        self.backend_type = self.backend.backend_type if self.backend is not None else BackendType.SQLITE
    else:
        self.backend = self._resolve_backend(backend=backend, config=config)
        self.backend_type = self.backend.backend_type
    self.default_org_id = default_org_id
    self.default_team_id = default_team_id

    self._txn_conn_var = ContextVar(
        f"media_db_txn_conn_{id(self)}",
        default=None,
    )
    self._tx_depth_var = ContextVar(
        f"media_db_tx_depth_{id(self)}",
        default=0,
    )
    self._persistent_conn_var = ContextVar(
        f"media_db_persistent_conn_{id(self)}",
        default=None,
    )
    self._persistent_conn = None

    if self.backend_type == BackendType.SQLITE and self.is_memory_db and not existing_only:
        persistent_conn = sqlite3.connect(
            self.db_path_str,
            check_same_thread=False,
            isolation_level=None,
        )
        try:
            persistent_conn.row_factory = sqlite3.Row
            self._apply_sqlite_connection_pragmas(persistent_conn)
        except sqlite3.Error:
            pass
        self._persistent_conn = persistent_conn

    self._media_insert_lock = threading.Lock()
    self._scope_cache = (self.default_org_id, self.default_team_id)

    if existing_only:
        if self.backend_type == BackendType.SQLITE:
            _open_existing_sqlite_database(self)
        return

    initialization_successful = False
    try:
        self._initialize_schema()
        initialization_successful = True
    except (DatabaseError, SchemaError, sqlite3.Error, BackendDatabaseError) as exc:
        logging.critical(
            f"FATAL: DB Initialization failed for {self.db_path_str}: {exc}",
            exc_info=True,
        )
        self.close_connection()
        raise DatabaseError(f"Database initialization failed: {exc}") from exc  # noqa: TRY003
    except _MEDIA_NONCRITICAL_EXCEPTIONS as exc:
        logging.critical(
            f"FATAL: Unexpected error during DB Initialization for {self.db_path_str}: {exc}",
            exc_info=True,
        )
        self.close_connection()
        raise DatabaseError(f"Unexpected database initialization error: {exc}") from exc  # noqa: TRY003
    finally:
        if initialization_successful:
            logging.debug(
                f"Database initialization completed successfully for {self.db_path_str}"
            )
        else:
            logging.error(
                f"Database initialization block finished for {self.db_path_str}, but failed."
            )


def initialize_db(self: Any):
    """Revalidate schema state for legacy callers and return ``self``."""

    if getattr(self, "existing_only", False):
        return self
    try:
        self._initialize_schema()
    except _MEDIA_NONCRITICAL_EXCEPTIONS as exc:
        raise DatabaseError(f"Database initialization failed: {exc}") from exc  # noqa: TRY003
    return self


def _ensure_sqlite_backend(self: Any) -> None:
    """Compatibility no-op retained for legacy bootstrap callers."""

    if self.backend_type != BackendType.SQLITE:
        return


__all__ = [
    "initialize_media_database",
    "initialize_db",
    "_ensure_sqlite_backend",
]
