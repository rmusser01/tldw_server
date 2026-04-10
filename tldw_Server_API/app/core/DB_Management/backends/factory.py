"""
Database backend factory for creating and managing database backends.

This module provides a factory pattern implementation for creating
database backend instances based on configuration.
"""

import os
import time
import urllib.parse as _url
from dataclasses import replace
from pathlib import Path
from threading import RLock, Thread
from typing import Optional

from loguru import logger

from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths

from .base import BackendType, DatabaseBackend, DatabaseConfig, DatabaseError, normalize_backend_name
from .sqlite_backend import SQLiteBackend

# Try to import PostgreSQL backend if available
try:
    from .postgresql_backend import PostgreSQLBackend
    POSTGRESQL_AVAILABLE = True
except ImportError:
    POSTGRESQL_AVAILABLE = False


# Registry of available backends
_BACKEND_REGISTRY: dict[BackendType, type[DatabaseBackend]] = {
    BackendType.SQLITE: SQLiteBackend,
}

# Register PostgreSQL if available
if POSTGRESQL_AVAILABLE:
    _BACKEND_REGISTRY[BackendType.POSTGRESQL] = PostgreSQLBackend

# Global backend instances cache
_backend_instances: dict[str, DatabaseBackend] = {}
_backend_instances_lock = RLock()
_sqlite_backend_registry: dict[tuple, DatabaseBackend] = {}
_sqlite_registry_lock = RLock()
_SQLITE_EVICTION_GRACE_SECONDS = 5.0


def _describe_sqlite_target(config: DatabaseConfig) -> str:
    raw_path = (config.sqlite_path or "").strip()
    if raw_path == ":memory:":
        return ":memory:"
    if raw_path.lower().startswith("file:"):
        return raw_path
    return str(Path(raw_path).resolve()) if raw_path else "<default>"


def _snapshot_sqlite_config(config: DatabaseConfig) -> DatabaseConfig:
    return replace(config)


def _sqlite_uri_to_file_path(
    raw_uri: str,
    *,
    parsed: _url.ParseResult | None = None,
) -> Optional[Path]:
    if parsed is None:
        try:
            parsed = _url.urlparse(raw_uri)
        except (TypeError, ValueError):
            return None
    if parsed.scheme != "file":
        return None
    path = _url.unquote(parsed.path or "")
    if not path or path in {":memory:", "/:memory:"}:
        return None
    candidate = Path(path).expanduser()
    try:
        if candidate.is_absolute():
            return candidate.resolve()
        return (Path.cwd() / candidate).resolve()
    except (OSError, RuntimeError, ValueError):
        return candidate


def _is_anonymous_memory_uri(raw_path: str) -> bool:
    lowered = raw_path.lower()
    if not lowered.startswith("file:"):
        return False
    if lowered.startswith("file::memory:"):
        return True
    try:
        parsed = _url.urlparse(raw_path)
    except (TypeError, ValueError):
        return False
    query = _url.parse_qs(parsed.query or "")
    mode = (query.get("mode", [""])[0] or "").lower()
    if mode != "memory":
        return False
    return not bool((parsed.path or "").strip("/"))


def _is_named_shared_cache_memory_uri(raw_path: str) -> bool:
    lowered = raw_path.lower()
    if not lowered.startswith("file:"):
        return False
    if _is_anonymous_memory_uri(raw_path):
        return False
    try:
        parsed = _url.urlparse(raw_path)
    except (TypeError, ValueError):
        return False
    query = _url.parse_qs(parsed.query or "")
    mode = (query.get("mode", [""])[0] or "").lower()
    cache = (query.get("cache", [""])[0] or "").lower()
    if mode != "memory" or cache != "shared":
        return False
    return bool((parsed.path or "").strip("/"))


def _canonicalize_named_shared_cache_memory_uri(raw_uri: str) -> str:
    try:
        parsed = _url.urlparse(raw_uri)
    except (TypeError, ValueError):
        return raw_uri

    raw_name = _url.unquote(parsed.path or "")
    if not raw_name:
        return raw_uri

    extra_pairs = [
        (key, value)
        for key, value in _url.parse_qsl(parsed.query or "", keep_blank_values=True)
        if key.lower() not in {"cache", "mode"}
    ]
    extra_pairs.sort(key=lambda item: (item[0], item[1]))
    query_pairs = [("cache", "shared"), ("mode", "memory"), *extra_pairs]
    return f"file:{raw_name}?{_url.urlencode(query_pairs, doseq=True)}"


def _normalize_sqlite_target(raw_path: str) -> str:
    cleaned = (raw_path or "").strip()
    if cleaned.lower().startswith("file:"):
        if _is_anonymous_memory_uri(cleaned):
            return cleaned
        if _is_named_shared_cache_memory_uri(cleaned):
            return _canonicalize_named_shared_cache_memory_uri(cleaned)
        try:
            parsed = _url.urlparse(cleaned)
        except (TypeError, ValueError):
            parsed = None
        maybe_file = _sqlite_uri_to_file_path(cleaned, parsed=parsed)
        if maybe_file is not None:
            if parsed and parsed.query:
                # Keep behavior-affecting URI options distinct in canonical identity.
                query_pairs = sorted(_url.parse_qsl(parsed.query, keep_blank_values=True))
                encoded_query = _url.urlencode(query_pairs, doseq=True)
                return f"{maybe_file}?{encoded_query}"
            return str(maybe_file)
        return cleaned
    if cleaned:
        try:
            return str(Path(cleaned).resolve())
        except (OSError, RuntimeError, ValueError):
            return cleaned
    return "<default>"


def _sqlite_signature(config: DatabaseConfig) -> tuple | None:
    raw_path = (config.sqlite_path or "").strip()
    if raw_path == ":memory:":
        return None
    if _is_anonymous_memory_uri(raw_path):
        return None
    if raw_path.lower().startswith("file:"):
        try:
            parsed = _url.urlparse(raw_path)
        except (TypeError, ValueError):
            parsed = None
        if parsed is not None:
            mode = (_url.parse_qs(parsed.query or "").get("mode", [""])[0] or "").lower()
            if mode == "memory" and not _is_named_shared_cache_memory_uri(raw_path):
                return None
    return (
        BackendType.SQLITE,
        _normalize_sqlite_target(raw_path),
        bool(config.sqlite_wal_mode),
        bool(config.sqlite_foreign_keys),
    )


def _retire_backend_instance(backend: DatabaseBackend) -> None:
    if getattr(backend, "backend_type", None) != BackendType.SQLITE:
        return

    pool_lock = getattr(backend, "_pool_lock", None)
    if pool_lock is not None:
        with pool_lock:
            setattr(backend, "_retired", True)
        return

    if hasattr(backend, "_retired"):
        setattr(backend, "_retired", True)


def _close_backend_instance(name: str, backend: DatabaseBackend) -> None:
    try:
        _retire_backend_instance(backend)
        pool_lock = getattr(backend, "_pool_lock", None)
        if pool_lock is not None:
            with pool_lock:
                pool = getattr(backend, "_pool", None)
                if pool is not None:
                    pool.close_all()
                    backend._pool = None
        else:
            pool = getattr(backend, "_pool", None)
            if pool is not None:
                pool.close_all()
                if hasattr(backend, "_pool"):
                    backend._pool = None
        logger.info(f"Closed backend: {name}")
    except Exception as e:
        logger.error(f"Error closing backend {name}: {e}")


def _close_backends_deduplicated(backends: list[tuple[str, DatabaseBackend]]) -> None:
    seen: set[int] = set()
    for name, backend in backends:
        if id(backend) in seen:
            continue
        seen.add(id(backend))
        _close_backend_instance(name, backend)


def _retire_backends_deduplicated(backends: list[tuple[str, DatabaseBackend]]) -> None:
    seen: set[int] = set()
    for _, backend in backends:
        if id(backend) in seen:
            continue
        seen.add(id(backend))
        _retire_backend_instance(backend)


def _evict_backend_references() -> list[tuple[str, DatabaseBackend]]:
    with _backend_instances_lock:
        named = list(_backend_instances.items())
        _backend_instances.clear()
    with _sqlite_registry_lock:
        sqlite_managed = [(f"sqlite::{sig[1]}", backend) for sig, backend in _sqlite_backend_registry.items()]
        _sqlite_backend_registry.clear()
    return named + sqlite_managed


def _evict_selected_sqlite_backend_references(
    *,
    backends: list[DatabaseBackend] | None = None,
    sqlite_targets: list[str] | None = None,
) -> list[tuple[str, DatabaseBackend]]:
    selected_backend_ids = {
        id(backend)
        for backend in (backends or [])
        if backend is not None
    }
    normalized_targets = {
        _normalize_sqlite_target(str(target))
        for target in (sqlite_targets or [])
        if str(target).strip()
    }

    if not selected_backend_ids and not normalized_targets:
        return []

    sqlite_managed: list[tuple[str, DatabaseBackend]] = []
    with _sqlite_registry_lock:
        for signature, backend in list(_sqlite_backend_registry.items()):
            signature_target = signature[1]
            if (
                id(backend) in selected_backend_ids
                or signature_target in normalized_targets
            ):
                sqlite_managed.append((f"sqlite::{signature_target}", backend))
                selected_backend_ids.add(id(backend))
                _sqlite_backend_registry.pop(signature, None)

    if not selected_backend_ids:
        return sqlite_managed

    named: list[tuple[str, DatabaseBackend]] = []
    with _backend_instances_lock:
        for name, backend in list(_backend_instances.items()):
            if id(backend) in selected_backend_ids:
                named.append((name, backend))
                _backend_instances.pop(name, None)

    return named + sqlite_managed


def _close_evicted_backends(evicted: list[tuple[str, DatabaseBackend]], mode: str) -> None:
    if not evicted:
        return
    _retire_backends_deduplicated(evicted)
    if mode == "hard":
        _close_backends_deduplicated(evicted)
        return

    def _deferred_close(backends: list[tuple[str, DatabaseBackend]]) -> None:
        time.sleep(_SQLITE_EVICTION_GRACE_SECONDS)
        _close_backends_deduplicated(backends)

    Thread(target=_deferred_close, args=(evicted,), daemon=True).start()


def _get_or_create_shared_sqlite_backend(config: DatabaseConfig) -> DatabaseBackend:
    signature = _sqlite_signature(config)
    backend_class = _BACKEND_REGISTRY[BackendType.SQLITE]
    if signature is None:
        logger.debug("Creating sqlite backend for {}", _describe_sqlite_target(config))
        return backend_class(_snapshot_sqlite_config(config))

    with _sqlite_registry_lock:
        existing = _sqlite_backend_registry.get(signature)
        if existing is not None:
            return existing

        logger.debug("Creating sqlite backend for {}", _describe_sqlite_target(config))
        backend = backend_class(_snapshot_sqlite_config(config))
        _sqlite_backend_registry[signature] = backend
        return backend


def is_factory_managed_backend(backend: DatabaseBackend | None) -> bool:
    """Return True when *backend* is a shared SQLite backend tracked by the factory."""
    if backend is None:
        return False
    if getattr(backend, "backend_type", None) != BackendType.SQLITE:
        return False
    with _sqlite_registry_lock:
        return any(candidate is backend for candidate in _sqlite_backend_registry.values())


def release_managed_backend(backend: DatabaseBackend | None) -> bool:
    """Release a managed backend reference without directly closing its pool.

    Pool lifecycle for managed SQLite backends is owned centrally via
    ``reset_backend_registry()``; this helper intentionally does not close
    pools directly.
    """
    return is_factory_managed_backend(backend)


class DatabaseBackendFactory:
    """Factory for creating database backend instances."""

    @staticmethod
    def create_backend(config: DatabaseConfig) -> DatabaseBackend:
        """
        Create a database backend instance based on configuration.

        Args:
            config: Database configuration

        Returns:
            DatabaseBackend instance

        Raises:
            DatabaseError: If backend type is not supported
        """
        backend_type = config.backend_type

        if backend_type not in _BACKEND_REGISTRY:
            raise DatabaseError(f"Unsupported backend type: {backend_type}")

        if backend_type == BackendType.SQLITE:
            return _get_or_create_shared_sqlite_backend(config)

        backend_class = _BACKEND_REGISTRY[backend_type]
        logger.info("Creating {} backend", backend_type.value)
        return backend_class(config)


class BackendFactory:
    """
    Backward-compatible alias used by some tests/utilities.

    Provides a stricter type check that raises ValueError when an invalid
    backend type string is provided in the config.
    """

    @staticmethod
    def create_backend(config: DatabaseConfig) -> DatabaseBackend:
        bt = config.backend_type
        # Coerce string backend types to enum, raising ValueError on invalid input
        if isinstance(bt, str):
            try:
                normalized = normalize_backend_name(bt)
                config.backend_type = BackendType(normalized)
            except ValueError as e:
                # Match expected behavior in tests
                raise ValueError(f"Invalid backend type: {bt}") from e
        return DatabaseBackendFactory.create_backend(config)

    @staticmethod
    def create_from_env(
        backend_type: Optional[str] = None,
        config_overrides: Optional[dict] = None
    ) -> DatabaseBackend:
        """
        Create a backend from environment variables.

        Args:
            backend_type: Override backend type (default from env)
            config_overrides: Additional config overrides

        Returns:
            DatabaseBackend instance
        """
        # Determine backend type
        if backend_type is None:
            backend_type = os.getenv("TLDW_DB_BACKEND", "sqlite")

        try:
            normalized = normalize_backend_name(backend_type)
            backend_enum = BackendType(normalized)
        except ValueError:
            raise DatabaseError(f"Invalid backend type: {backend_type}") from None

        # Build configuration from environment
        config = DatabaseConfig(backend_type=backend_enum)

        # SQLite configuration
        if backend_enum == BackendType.SQLITE:
            config.sqlite_path = os.getenv(
                "TLDW_SQLITE_PATH",
                str(DatabasePaths.get_media_db_path(DatabasePaths.get_single_user_id()))
            )
            config.sqlite_wal_mode = os.getenv(
                "TLDW_SQLITE_WAL_MODE", "true"
            ).lower() == "true"
            config.sqlite_foreign_keys = os.getenv(
                "TLDW_SQLITE_FOREIGN_KEYS", "true"
            ).lower() == "true"

        # PostgreSQL configuration (future)
        elif backend_enum == BackendType.POSTGRESQL:
            config.pg_host = os.getenv("TLDW_PG_HOST", "localhost")
            config.pg_port = int(os.getenv("TLDW_PG_PORT", "5432"))
            config.pg_database = os.getenv("TLDW_PG_DATABASE", "tldw")
            config.pg_user = os.getenv("TLDW_PG_USER", "tldw_user")
            config.pg_password = os.getenv("TLDW_PG_PASSWORD", "")
            config.pg_sslmode = os.getenv("TLDW_PG_SSLMODE", "prefer")

        # Common configuration
        config.pool_size = int(os.getenv("TLDW_DB_POOL_SIZE", "10"))
        config.pool_timeout = float(os.getenv("TLDW_DB_POOL_TIMEOUT", "30.0"))
        config.echo = os.getenv("TLDW_DB_ECHO", "false").lower() == "true"

        # Apply overrides
        if config_overrides:
            for key, value in config_overrides.items():
                if hasattr(config, key):
                    setattr(config, key, value)

        return DatabaseBackendFactory.create_backend(config)

    @staticmethod
    def create_from_config_file(
        config_path: str,
        backend_override: Optional[str] = None
    ) -> DatabaseBackend:
        """
        Create a backend from a configuration file.

        Args:
            config_path: Path to configuration file
            backend_override: Override backend type from config

        Returns:
            DatabaseBackend instance
        """
        import yaml

        with open(config_path) as f:
            config_data = yaml.safe_load(f)

        db_config = config_data.get('database', {})

        # Determine backend type
        backend_type_str = backend_override or db_config.get('backend', 'sqlite')
        backend_type_str = normalize_backend_name(backend_type_str)

        try:
            backend_type = BackendType(backend_type_str)
        except ValueError:
            raise DatabaseError(f"Invalid backend type: {backend_type_str}") from None

        config = DatabaseConfig(backend_type=backend_type)

        # Load backend-specific configuration
        if backend_type == BackendType.SQLITE:
            sqlite_config = db_config.get('sqlite', {})
            config.sqlite_path = sqlite_config.get(
                'path', str(DatabasePaths.get_media_db_path(DatabasePaths.get_single_user_id()))
            )
            config.sqlite_wal_mode = sqlite_config.get('wal_mode', True)
            config.sqlite_foreign_keys = sqlite_config.get('foreign_keys', True)

        elif backend_type == BackendType.POSTGRESQL:
            pg_config = db_config.get('postgresql', {})
            config.pg_host = pg_config.get('host', 'localhost')
            config.pg_port = pg_config.get('port', 5432)
            config.pg_database = pg_config.get('database', 'tldw')
            config.pg_user = pg_config.get('user', 'tldw_user')
            config.pg_password = pg_config.get('password', '')
            config.pg_sslmode = pg_config.get('sslmode', 'prefer')
            config.pool_size = pg_config.get('pool_size', 20)
            config.max_overflow = pg_config.get('max_overflow', 40)

        return DatabaseBackendFactory.create_backend(config)


def register_backend(backend_type: BackendType, backend_class: type[DatabaseBackend]) -> None:
    """
    Register a new backend implementation.

    Args:
        backend_type: Backend type enum
        backend_class: Backend implementation class
    """
    _BACKEND_REGISTRY[backend_type] = backend_class
    logger.info(f"Registered backend: {backend_type.value}")


def get_backend(
    name: str = "default",
    config: Optional[DatabaseConfig] = None,
    create_if_missing: bool = True
) -> Optional[DatabaseBackend]:
    """
    Get or create a named backend instance.

    This function provides a singleton pattern for backend instances,
    ensuring that the same backend instance is reused across the application.

    Args:
        name: Backend instance name
        config: Configuration for creating new instance
        create_if_missing: Create instance if it doesn't exist

    Returns:
        DatabaseBackend instance or None
    """
    with _backend_instances_lock:
        existing = _backend_instances.get(name)
        if existing is not None:
            return existing

    if not create_if_missing:
        return None

    resolved_config = config if config is not None else DatabaseConfig.from_env()
    backend = DatabaseBackendFactory.create_backend(resolved_config)

    duplicate: DatabaseBackend | None = None
    with _backend_instances_lock:
        existing = _backend_instances.get(name)
        if existing is not None:
            if existing is not backend:
                duplicate = backend
            backend = existing
        else:
            _backend_instances[name] = backend

    if duplicate is not None:
        _close_backend_instance(name, duplicate)

    return backend


def reset_backend_registry(mode: str = "hard") -> None:
    """
    Reset backend registries and close backend pools.

    hard:
      - evict named and canonical sqlite references immediately
      - close pools synchronously
    graceful:
      - evict references immediately
      - close pools after a grace delay in a background thread
    """
    normalized = (mode or "hard").strip().lower()
    if normalized not in {"hard", "graceful"}:
        raise ValueError(f"Unsupported reset mode: {mode}")

    evicted = _evict_backend_references()
    _close_evicted_backends(evicted, normalized)


def reset_managed_sqlite_backends(
    mode: str = "hard",
    *,
    backends: list[DatabaseBackend] | None = None,
    sqlite_targets: list[str] | None = None,
) -> None:
    """Reset selected managed SQLite backends and their named aliases.

    Unlike ``reset_backend_registry()``, this only evicts canonical managed
    SQLite backends matched by backend object identity or normalized sqlite
    target path/URI.
    """
    normalized = (mode or "hard").strip().lower()
    if normalized not in {"hard", "graceful"}:
        raise ValueError(f"Unsupported reset mode: {mode}")

    evicted = _evict_selected_sqlite_backend_references(
        backends=backends,
        sqlite_targets=sqlite_targets,
    )
    _close_evicted_backends(evicted, normalized)


def close_all_backends() -> None:
    """Close all backends and clear named/canonical caches via hard reset."""
    reset_backend_registry(mode="hard")


# Convenience function for getting default backend
def get_default_backend() -> DatabaseBackend:
    """
    Get the default database backend.

    Returns:
        Default DatabaseBackend instance
    """
    return get_backend("default")
