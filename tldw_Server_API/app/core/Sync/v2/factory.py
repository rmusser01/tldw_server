from __future__ import annotations

"""Sync v2 service composition helpers shared by HTTP and non-HTTP entrypoints."""

import os
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

from tldw_Server_API.app.core.DB_Management.Sync_DB import SYNC_DB_FILENAME, SyncDatabase
from tldw_Server_API.app.core.DB_Management.db_path_utils import (
    DatabasePaths,
    _resolve_user_id_for_storage,
)

from .adapters import StaticSyncAdapter, SyncAdapterRegistry
from .models import M1_SYNC_DOMAINS
from .security import server_trusted_encryption_status_from_env
from .service import SyncV2Service, SyncV2Settings
from .store import SyncV2Store


@lru_cache(maxsize=1)
def default_sync_v2_registry() -> SyncAdapterRegistry:
    """Return the default Sync v2 domain adapter registry."""

    return SyncAdapterRegistry(
        [
            StaticSyncAdapter(domain=domain, supported_adapter_versions={1})
            for domain in M1_SYNC_DOMAINS
        ]
    )


def sync_v2_service_for_user(user_id: str) -> SyncV2Service:
    """Build a Sync v2 service for a user using cached storage wiring."""

    store = _sync_v2_store_for_user(
        user_id,
        os.getenv("SYNC_V2_DATABASE_URL", "").strip(),
        os.getenv("SYNC_V2_SQLITE_PATH", "").strip(),
    )
    return SyncV2Service(
        store=store,
        adapters=default_sync_v2_registry(),
        settings=SyncV2Settings(
            server_trusted_encryption=server_trusted_encryption_status_from_env(),
        ),
    )


def sync_v2_storage_exists_for_user(user_id: str) -> bool:
    """Return whether durable Sync v2 storage already exists for a user."""

    database_url = os.getenv("SYNC_V2_DATABASE_URL", "").strip()
    sqlite_path = os.getenv("SYNC_V2_SQLITE_PATH", "").strip()
    default_path = _default_sync_v2_path_for_user(user_id)
    if database_url:
        return _sync_v2_database_url_exists(database_url, default_path)
    if sqlite_path:
        return _sync_v2_sqlite_path_exists(sqlite_path)
    return default_path.exists()


@lru_cache(maxsize=256)
def _sync_v2_store_for_user(
    user_id: str,
    database_url: str,
    sqlite_path: str,
) -> SyncV2Store:
    del database_url, sqlite_path
    return SyncV2Store(SyncDatabase(user_id=user_id))


def _default_sync_v2_path_for_user(user_id: str) -> Path:
    base_dir = DatabasePaths.resolve_user_db_base_dir()
    safe_user_id = _resolve_user_id_for_storage(user_id)
    return (base_dir / safe_user_id / SYNC_DB_FILENAME).resolve()


def _sync_v2_database_url_exists(database_url: str, default_path: Path) -> bool:
    parsed = urlparse(database_url)
    scheme = (parsed.scheme or "").lower().split("+", 1)[0]
    if scheme in {"postgres", "postgresql"}:
        return True
    if scheme in {"sqlite", "file", ""}:
        return _sync_v2_sqlite_url_path(database_url, default_path).exists()
    return True


def _sync_v2_sqlite_path_exists(sqlite_path: str) -> bool:
    if sqlite_path == ":memory:":
        return True
    return Path(sqlite_path).expanduser().exists()


def _sync_v2_sqlite_url_path(database_url: str, default_path: Path) -> Path:
    parsed = urlparse(database_url)
    raw_path = parsed.path or ""
    if raw_path in {"/:memory:", ":memory:"}:
        return Path(":memory:")
    if raw_path.startswith("/./"):
        raw_path = raw_path[1:]
    if raw_path.startswith("/") and raw_path != "/:memory:":
        return Path(raw_path)
    return default_path.parent / (raw_path or default_path.name)


__all__ = [
    "default_sync_v2_registry",
    "sync_v2_service_for_user",
    "sync_v2_storage_exists_for_user",
]
