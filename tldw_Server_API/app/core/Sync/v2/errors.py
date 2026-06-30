from __future__ import annotations

"""Sync v2 core exceptions."""


class SyncV2Error(Exception):
    """Base exception for Sync v2 failures."""


class SyncStoreError(SyncV2Error):
    """Raised when the Sync v2 store cannot complete an operation."""


class SyncIdempotencyConflictError(SyncStoreError):
    """Raised when an idempotent retry reuses an ID with different content."""


class SyncDatasetNotFoundError(SyncStoreError):
    """Raised when a requested dataset does not exist."""


class SyncInvalidDomainError(SyncStoreError):
    """Raised when an operation targets a domain not enrolled in a dataset."""


class SyncConflictNotFoundError(SyncStoreError):
    """Raised when a requested conflict record does not exist."""


__all__ = [
    "SyncConflictNotFoundError",
    "SyncDatasetNotFoundError",
    "SyncIdempotencyConflictError",
    "SyncInvalidDomainError",
    "SyncStoreError",
    "SyncV2Error",
]
