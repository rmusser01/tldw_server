"""Process-local advisory lock leases for MCP filesystem tools."""

from __future__ import annotations

import secrets
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class FilesystemLockLease:
    """A caller-held advisory lease for one workspace-relative path."""

    workspace_key: str
    path: str
    lease_id: str
    owner: str
    expires_at: float
    ttl_seconds: int
    workspace_id: str | None = None
    session_id: str | None = None

    def expires_at_iso(self) -> str:
        """Return the lease expiry timestamp in UTC ISO-8601 form."""

        return datetime.fromtimestamp(self.expires_at, tz=timezone.utc).isoformat()

    def safe_payload(self) -> dict[str, Any]:
        """Return non-sensitive lease metadata safe for tool responses."""

        return {
            "path": self.path,
            "lease_id": self.lease_id,
            "owner": self.owner,
            "expires_at": self.expires_at_iso(),
            "ttl_seconds": self.ttl_seconds,
        }

    def conflict_payload(self) -> dict[str, Any]:
        """Return non-sensitive lock holder details for conflict responses."""

        return {
            "reason_code": "lock_conflict",
            "path": self.path,
            "held": True,
            "held_owner": self.owner,
            "expires_at": self.expires_at_iso(),
        }


class FilesystemLockManager(Protocol):
    """Backend contract for advisory filesystem lock leases."""

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
        """Acquire or renew a lease for one workspace path."""
        ...

    def release(self, *, workspace_key: str, path: str, lease_id: str) -> FilesystemLockLease | None:
        """Release an active lease when the token matches."""
        ...

    def validate(self, *, workspace_key: str, path: str, lease_id: str) -> FilesystemLockLease:
        """Validate that the caller holds the active lease."""
        ...


class FilesystemLockConflict(ValueError):
    """Raised when an active lease is held by another token."""

    def __init__(self, lease: FilesystemLockLease) -> None:
        super().__init__("lock_conflict")
        self.lease = lease


class FilesystemLockMissing(ValueError):
    """Raised when no active lease exists for a requested token/path pair."""

    def __init__(self) -> None:
        super().__init__("lock_missing")


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
        expires_at = now + max(1, ttl_seconds)
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
                    ttl_seconds=max(1, ttl_seconds),
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
                ttl_seconds=max(1, ttl_seconds),
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
        raw_backend = "memory"
    backend = str(raw_backend).strip().lower()
    if backend in {"", "memory", "in_memory"}:
        return InMemoryFilesystemLockManager()
    raise ValueError(f"unsupported filesystem lock_manager_backend: {raw_backend!r}")


__all__ = [
    "FilesystemLockConflict",
    "FilesystemLockLease",
    "FilesystemLockManager",
    "FilesystemLockMissing",
    "InMemoryFilesystemLockManager",
    "create_filesystem_lock_manager",
]
