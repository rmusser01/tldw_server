"""Shared filesystem lock lease models and exceptions."""

from __future__ import annotations

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


__all__ = [
    "FilesystemLockConflict",
    "FilesystemLockLease",
    "FilesystemLockManager",
    "FilesystemLockMissing",
]
