from __future__ import annotations

"""Sync v2 service composition helpers shared by HTTP and non-HTTP entrypoints."""

import os
from functools import lru_cache

from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase

from .adapters import SyncAdapterRegistry
from .domain_adapters import (
    ChatDomainAdapter,
    MediaCompatibilityAdapter,
    NotesDomainAdapter,
    SourceCacheAdapter,
    WorkspacesDomainAdapter,
)
from .service import SyncV2Service
from .store import SyncV2Store


@lru_cache(maxsize=1)
def default_sync_v2_registry() -> SyncAdapterRegistry:
    """Return the default Sync v2 domain adapter registry."""

    return SyncAdapterRegistry(
        [
            NotesDomainAdapter(),
            ChatDomainAdapter(),
            WorkspacesDomainAdapter(),
            SourceCacheAdapter(),
            MediaCompatibilityAdapter(),
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
    )


@lru_cache(maxsize=256)
def _sync_v2_store_for_user(
    user_id: str,
    database_url: str,
    sqlite_path: str,
) -> SyncV2Store:
    del database_url, sqlite_path
    return SyncV2Store(SyncDatabase(user_id=user_id))


__all__ = [
    "default_sync_v2_registry",
    "sync_v2_service_for_user",
]
