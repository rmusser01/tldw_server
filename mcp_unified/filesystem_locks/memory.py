"""Process-local advisory lock leases for MCP filesystem tools."""

from __future__ import annotations

import secrets
import threading
import time
from collections.abc import Mapping
from typing import Any

from .models import (
    FilesystemLockConflict,
    FilesystemLockLease,
    FilesystemLockManager,
    FilesystemLockMissing,
)


class InMemoryFilesystemLockManager:
    """Thread-safe, process-local advisory lock lease manager."""

    def __init__(self, *, token_bytes: int = 24, sweep_interval: int = 64, max_sweep_entries: int = 512) -> None:
        self._token_bytes = max(16, token_bytes)
        self._sweep_interval = max(1, sweep_interval)
        self._max_sweep_entries = max(1, max_sweep_entries)
        self._operation_count = 0
        self._leases: dict[tuple[str, str], FilesystemLockLease] = {}
        self._lock = threading.RLock()

    def acquire(
        self,
        *,
        workspace_key: str,
        path: str,
        owner: str,
        ttl_seconds: int,
        lease_id: str | None = None,
        workspace_id: str | None = None,
        session_id: str | None = None,
    ) -> tuple[FilesystemLockLease, bool]:
        """Acquire a new lease or renew the current lease when the token matches."""

        key = (workspace_key, path)
        now = time.time()
        ttl = max(1, ttl_seconds)
        expires_at = now + ttl
        with self._lock:
            self._maybe_sweep_expired_locked(now=now)
            active = self._active_lease_locked(key, now=now)
            if active is not None:
                if lease_id != active.lease_id:
                    raise FilesystemLockConflict(active)
                renewed = FilesystemLockLease(
                    workspace_key=workspace_key,
                    path=path,
                    lease_id=active.lease_id,
                    owner=owner,
                    expires_at=expires_at,
                    ttl_seconds=ttl,
                    workspace_id=workspace_id,
                    session_id=session_id,
                )
                self._leases[key] = renewed
                return renewed, True

            if lease_id is not None:
                raise FilesystemLockMissing()

            acquired = FilesystemLockLease(
                workspace_key=workspace_key,
                path=path,
                lease_id=secrets.token_urlsafe(self._token_bytes),
                owner=owner,
                expires_at=expires_at,
                ttl_seconds=ttl,
                workspace_id=workspace_id,
                session_id=session_id,
            )
            self._leases[key] = acquired
            return acquired, False

    def release(self, *, workspace_key: str, path: str, lease_id: str) -> FilesystemLockLease | None:
        """Release a lease when the caller presents the active token."""

        key = (workspace_key, path)
        with self._lock:
            now = time.time()
            self._maybe_sweep_expired_locked(now=now)
            active = self._active_lease_locked(key, now=now)
            if active is None:
                return None
            if active.lease_id != lease_id.strip():
                raise FilesystemLockConflict(active)
            del self._leases[key]
            return active

    def validate(self, *, workspace_key: str, path: str, lease_id: str) -> FilesystemLockLease:
        """Return the active lease when the caller presents the current token."""

        key = (workspace_key, path)
        with self._lock:
            now = time.time()
            self._maybe_sweep_expired_locked(now=now)
            active = self._active_lease_locked(key, now=now)
            if active is None:
                raise FilesystemLockMissing()
            if active.lease_id != lease_id:
                raise FilesystemLockConflict(active)
            return active

    def _active_lease_locked(self, key: tuple[str, str], *, now: float) -> FilesystemLockLease | None:
        lease = self._leases.get(key)
        if lease is None:
            return None
        if lease.expires_at <= now:
            del self._leases[key]
            return None
        return lease

    def _maybe_sweep_expired_locked(self, *, now: float) -> None:
        self._operation_count += 1
        if self._operation_count % self._sweep_interval != 0 and len(self._leases) <= self._max_sweep_entries:
            return

        for _ in range(min(self._max_sweep_entries, len(self._leases))):
            key, lease = next(iter(self._leases.items()))
            if lease.expires_at <= now:
                del self._leases[key]
            else:
                self._leases.pop(key)
                self._leases[key] = lease


def create_filesystem_lock_manager(settings: Mapping[str, Any] | None = None) -> FilesystemLockManager:
    """Create the configured filesystem lock manager backend.

    The first shipped backend remains process-local memory. Unsupported
    backends fail closed so future persistent backends can be added without
    silently downgrading operator intent.
    """

    raw_backend = (settings or {}).get("lock_manager_backend")
    if raw_backend is None:
        return InMemoryFilesystemLockManager()
    backend = str(raw_backend).strip().lower()
    if backend in {"memory", "in_memory"}:
        return InMemoryFilesystemLockManager()
    if backend == "sqlite":
        sqlite_path = (settings or {}).get("lock_manager_sqlite_path")
        if sqlite_path is None or not str(sqlite_path).strip():
            raise ValueError(
                "lock_manager_sqlite_path is required for sqlite filesystem lock manager"
            )
        try:
            from .sqlite import SQLiteFilesystemLockManager
        except ModuleNotFoundError as exc:
            if exc.name == "sqlalchemy":
                raise ImportError(
                    "SQLiteFilesystemLockManager requires the mcp-unified sqlite extra. "
                    "Install mcp-unified[sqlite] or mcp-unified[gateway]."
                ) from exc
            raise

        return SQLiteFilesystemLockManager(
            sqlite_path,
            timeout_seconds=float(
                (settings or {}).get("lock_manager_sqlite_timeout_seconds", 30.0)
            ),
            cleanup_interval=int((settings or {}).get("lock_manager_cleanup_interval", 64)),
            cleanup_limit=int((settings or {}).get("lock_manager_cleanup_limit", 512)),
        )
    raise ValueError(f"unsupported filesystem lock_manager_backend: {raw_backend!r}")


__all__ = [
    "InMemoryFilesystemLockManager",
    "create_filesystem_lock_manager",
]
