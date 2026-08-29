from __future__ import annotations

"""Sync v2 core exceptions."""


class SyncV2Error(Exception):
    """Base exception for Sync v2 failures."""


class SyncStoreError(SyncV2Error):
    """Raised when the Sync v2 store cannot complete an operation."""


class SyncIdempotencyConflictError(SyncStoreError):
    """Raised when an idempotent retry reuses an ID with different content."""


class SyncHeadConflictError(SyncStoreError):
    """Raised when append-time optimistic lineage no longer matches the head."""

    error_code = "sync_head_changed"

    def __init__(self) -> None:
        super().__init__(self.error_code)


class SyncMaterializationBusyError(SyncStoreError):
    """Raised when durable projection serialization cannot be acquired in time."""

    error_code = "sync_projection_busy"

    def __init__(self) -> None:
        super().__init__(self.error_code)


class SyncMaterializationContractError(SyncStoreError):
    """Raised when guarded projection input violates its mutation contract."""

    error_code = "sync_materialization_contract_invalid"

    def __init__(self) -> None:
        super().__init__(self.error_code)


class SyncMaterializationPredecessorError(SyncStoreError):
    """Raised when an earlier accepted projection has not reached applied state."""

    error_code = "sync_projection_predecessor_unresolved"

    def __init__(
        self,
        *,
        apply_status: str,
        conflict_id: str | None = None,
        domain: str | None = None,
        entity_id: str | None = None,
        server_sequence: int | None = None,
    ) -> None:
        super().__init__(self.error_code)
        self.apply_status = apply_status
        self.retryable = apply_status != "conflict"
        self.conflict_id = conflict_id
        self.domain = domain
        self.entity_id = entity_id
        self.server_sequence = server_sequence


class SyncDatasetNotFoundError(SyncStoreError):
    """Raised when a requested dataset does not exist."""


class SyncInvalidDomainError(SyncStoreError):
    """Raised when an operation targets a domain not enrolled in a dataset."""


class SyncConflictNotFoundError(SyncStoreError):
    """Raised when a requested conflict record does not exist."""


__all__ = [
    "SyncConflictNotFoundError",
    "SyncDatasetNotFoundError",
    "SyncHeadConflictError",
    "SyncIdempotencyConflictError",
    "SyncInvalidDomainError",
    "SyncMaterializationBusyError",
    "SyncMaterializationContractError",
    "SyncMaterializationPredecessorError",
    "SyncStoreError",
    "SyncV2Error",
]
