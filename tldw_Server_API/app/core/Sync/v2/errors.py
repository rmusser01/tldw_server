from __future__ import annotations

"""Sync v2 core exceptions."""


class SyncV2Error(Exception):
    """Base exception for Sync v2 failures."""


class SyncStoreError(SyncV2Error):
    """Raised when the Sync v2 store cannot complete an operation."""


class SyncDatasetNotFoundError(SyncStoreError):
    """Raised when a requested dataset does not exist."""


class SyncConflictNotFoundError(SyncStoreError):
    """Raised when a requested conflict record does not exist."""


__all__ = [
    "SyncConflictNotFoundError",
    "SyncDatasetNotFoundError",
    "SyncStoreError",
    "SyncV2Error",
]
