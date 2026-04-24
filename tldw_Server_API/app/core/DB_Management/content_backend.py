"""Utilities for loading and instantiating content database backends.

This module centralises configuration handling for the Media/ChaCha content
stores so they can run on either SQLite or PostgreSQL backends using the
shared DatabaseBackend abstraction.
"""

from __future__ import annotations

import os
import weakref
from configparser import ConfigParser
from dataclasses import dataclass

from loguru import logger

from tldw_Server_API.app.core.DB_Management.backends.base import (
    BackendType,
    DatabaseBackend,
    DatabaseConfig,
)
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory

_DEFAULT_SQLITE_PATH = ""
_DEFAULT_BACKUP_PATH = "./tldw_DB_Backups/"


@dataclass
class ContentDatabaseSettings:
    """Resolved configuration for the content database backend."""

    raw_backend_type: str
    backend_type: BackendType | None
    database_config: DatabaseConfig | None
    sqlite_path: str | None
    backup_path: str | None


def _normalise_backend_name(raw_backend: str) -> str:
    """Normalise backend identifier from config/environment."""
    if not raw_backend:
        return "sqlite"
    return raw_backend.strip().lower()


def _backend_type_from_raw(raw_backend: str) -> BackendType | None:
    """Map textual backend identifiers to BackendType enum where supported."""
    if raw_backend in {"sqlite", "sqlite3"}:
        return BackendType.SQLITE
    if raw_backend in {"postgres", "postgresql"}:
        return BackendType.POSTGRESQL
    # elasticsearch / opensearch are planned but handled elsewhere for now
    return None


def load_content_db_settings(config: ConfigParser) -> ContentDatabaseSettings:
    """Build ContentDatabaseSettings from config parser and environment."""
    raw_backend = _normalise_backend_name(
        os.getenv("TLDW_CONTENT_DB_BACKEND")
        or config.get("Database", "type", fallback="sqlite")
    )

    backend_type = _backend_type_from_raw(raw_backend)

    sqlite_path = os.getenv("TLDW_CONTENT_SQLITE_PATH") or config.get(
        "Database", "sqlite_path", fallback=_DEFAULT_SQLITE_PATH
    )
    backup_path = os.getenv("TLDW_DB_BACKUP_PATH") or config.get(
        "Database", "backup_path", fallback=_DEFAULT_BACKUP_PATH
    )

    database_config: DatabaseConfig | None
    if backend_type == BackendType.SQLITE:
        database_config = DatabaseConfig(
            backend_type=BackendType.SQLITE,
            sqlite_path=sqlite_path,
            sqlite_wal_mode=config.getboolean(
                "Database", "sqlite_wal_mode", fallback=True
            ),
            sqlite_foreign_keys=config.getboolean(
                "Database", "sqlite_foreign_keys", fallback=True
            ),
            pool_size=config.getint("Database", "pool_size", fallback=10),
            max_overflow=config.getint("Database", "max_overflow", fallback=20),
            pool_timeout=config.getfloat("Database", "pool_timeout", fallback=30.0),
        )
    elif backend_type == BackendType.POSTGRESQL:
        # Allow multiple environment variable conventions so local test suites
        # (which historically relied on POSTGRES_TEST_* values) work without
        # extra configuration. Explicit TLDW_* overrides always win.
        pg_dsn = os.getenv("TLDW_CONTENT_PG_DSN") or os.getenv("POSTGRES_TEST_DSN")

        def _env_chain(*names: str, fallback: str | None = None) -> str | None:
            for name in names:
                value = os.getenv(name)
                if value:
                    return value
            return fallback

        database_config = DatabaseConfig(
            backend_type=BackendType.POSTGRESQL,
            connection_string=pg_dsn
            or config.get("Database", "pg_connection_string", fallback=None),
            pg_host=_env_chain(
                "TLDW_CONTENT_PG_HOST",
                "TLDW_PG_HOST",
                "POSTGRES_TEST_HOST",
                fallback=config.get("Database", "pg_host", fallback="localhost"),
            ),
            pg_port=int(
                _env_chain(
                    "TLDW_CONTENT_PG_PORT",
                    "TLDW_PG_PORT",
                    "POSTGRES_TEST_PORT",
                )
                or config.get("Database", "pg_port", fallback=5432)
            ),
            pg_database=_env_chain(
                "TLDW_CONTENT_PG_DATABASE",
                "TLDW_PG_DATABASE",
                "POSTGRES_TEST_DATABASE",
                fallback=config.get("Database", "pg_database", fallback="tldw_content"),
            ),
            pg_user=_env_chain(
                "TLDW_CONTENT_PG_USER",
                "TLDW_PG_USER",
                "POSTGRES_TEST_USER",
                fallback=config.get("Database", "pg_user", fallback="tldw_user"),
            ),
            pg_password=_env_chain(
                "TLDW_CONTENT_PG_PASSWORD",
                "TLDW_PG_PASSWORD",
                "POSTGRES_TEST_PASSWORD",
                fallback=config.get("Database", "pg_password", fallback=""),
            ),
            pg_sslmode=_env_chain(
                "TLDW_CONTENT_PG_SSLMODE",
                "TLDW_PG_SSLMODE",
                fallback=config.get("Database", "pg_sslmode", fallback="prefer"),
            ),
            pool_size=config.getint("Database", "pg_pool_size", fallback=20),
            max_overflow=config.getint("Database", "pg_max_overflow", fallback=40),
            pool_timeout=config.getfloat("Database", "pg_pool_timeout", fallback=30.0),
        )
    else:
        database_config = None

    return ContentDatabaseSettings(
        raw_backend_type=raw_backend,
        backend_type=backend_type,
        database_config=database_config,
        sqlite_path=sqlite_path,
        backup_path=backup_path,
    )


def backend_target_key(backend: DatabaseBackend | None) -> str | None:
    """Return a stable identifier for a backend target across wrapper instances."""
    if backend is None:
        return None

    config = getattr(backend, "config", None)
    if backend.backend_type == BackendType.POSTGRESQL:
        if config and getattr(config, "connection_string", None):
            return str(config.connection_string)
        if config:
            host = getattr(config, "pg_host", None) or "localhost"
            port = getattr(config, "pg_port", None) or 5432
            database = getattr(config, "pg_database", None) or "<postgres>"
            return f"{host}:{port}/{database}"
        return f"postgres:{id(backend)}"

    if config and getattr(config, "sqlite_path", None):
        return str(config.sqlite_path)
    return f"sqlite:{id(backend)}"


_cached_backend = None
_cached_backend_signature: tuple | None = None
_retired_backend_finalizers: dict[int, weakref.finalize] = {}
try:
    from threading import RLock

    _cache_lock = RLock()
except Exception:  # pragma: no cover
    _cache_lock = None  # type: ignore


def _close_backend_pool(backend: DatabaseBackend | None) -> None:
    """Best-effort close for pooled shared content backends."""
    if backend is None:
        return

    try:
        pool = backend.get_pool()
        pool.close_all()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to close superseded content backend pool: {}", exc)


def _retire_backend_pool(backend: DatabaseBackend | None) -> None:
    """Close a superseded backend only after active references release it."""
    if backend is None:
        return

    backend_id = id(backend)

    try:
        pool = backend.get_pool()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to retire superseded content backend pool: {}", exc)
        return

    def _close_retired_pool() -> None:
        try:
            pool.close_all()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to close retired content backend pool: {}", exc)
        finally:
            if _cache_lock is not None:
                with _cache_lock:
                    _retired_backend_finalizers.pop(backend_id, None)
            else:
                _retired_backend_finalizers.pop(backend_id, None)

    if _cache_lock is not None:
        with _cache_lock:
            existing_finalizer = _retired_backend_finalizers.get(backend_id)
            if existing_finalizer is not None and existing_finalizer.alive:
                return
            try:
                _retired_backend_finalizers[backend_id] = weakref.finalize(
                    backend,
                    _close_retired_pool,
                )
            except TypeError:
                _close_retired_pool()
        return

    existing_finalizer = _retired_backend_finalizers.get(backend_id)
    if existing_finalizer is not None and existing_finalizer.alive:
        return
    try:
        _retired_backend_finalizers[backend_id] = weakref.finalize(
            backend,
            _close_retired_pool,
        )
    except TypeError:
        _close_retired_pool()


def clear_cached_backend() -> None:
    """Clear and retire the currently cached shared content backend.

    Acquires ``_cache_lock`` to avoid racing with concurrent
    ``get_content_backend()`` calls that also mutate the cache. The previous
    backend is retired after being removed from the cache so future callers
    cannot observe stale cached state while in-flight borrowers finish cleanly.

    This is a controlled reconfiguration hook for tests, startup, and
    admin-driven reloads. Consumers are expected to refresh shared PostgreSQL
    backends between operations; callers should not rotate the cache while an
    active operation is still holding a concrete backend reference.
    """
    global _cached_backend, _cached_backend_signature

    if _cache_lock is not None:
        with _cache_lock:
            old_backend = _cached_backend
            _cached_backend = None
            _cached_backend_signature = None
    else:
        old_backend = _cached_backend
        _cached_backend = None
        _cached_backend_signature = None

    _retire_backend_pool(old_backend)


def get_content_backend(config: ConfigParser):
    """Return a DatabaseBackend instance for content storage if supported.

    Thread-safe around cache check and creation to handle concurrent reloads.
    """
    global _cached_backend, _cached_backend_signature

    settings = load_content_db_settings(config)
    if not settings.database_config:
        return None

    # Only create a backend for PostgreSQL content mode. For SQLite, return None
    # so callers resolve per-user file paths instead of a root-level DB.
    if settings.backend_type != BackendType.POSTGRESQL:
        return None

    signature = (
        settings.backend_type,
        settings.database_config.connection_string,
        settings.database_config.sqlite_path,
        settings.database_config.pg_host,
        settings.database_config.pg_port,
        settings.database_config.pg_database,
        settings.database_config.pg_user,
        settings.database_config.pg_password,
        settings.database_config.pg_sslmode,
    )

    if _cache_lock is not None:
        with _cache_lock:
            if _cached_backend and _cached_backend_signature == signature:
                return _cached_backend
            old_backend = _cached_backend
            backend = DatabaseBackendFactory.create_backend(settings.database_config)
            _cached_backend = backend
            _cached_backend_signature = signature
        # Retire the superseded backend after the cache update so callers no
        # longer retrieve it from shared state, while existing borrowers can
        # finish on their captured backend instance.
        if old_backend is not None and old_backend is not backend:
            _retire_backend_pool(old_backend)
        return backend

    # Fallback without lock (environments without threading)
    if _cached_backend and _cached_backend_signature == signature:
        return _cached_backend
    old_backend = _cached_backend
    backend = DatabaseBackendFactory.create_backend(settings.database_config)
    _cached_backend = backend
    _cached_backend_signature = signature
    if old_backend is not None and old_backend is not backend:
        _retire_backend_pool(old_backend)
    return backend
