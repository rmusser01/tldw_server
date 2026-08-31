"""Typed persistence records for the Notes semantic-index authority ledger."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class SemanticDesiredState(str, Enum):
    ENABLED = "enabled"
    DISABLED = "disabled"


class SemanticGenerationState(str, Enum):
    STAGING = "staging"
    ACTIVE = "active"
    RETIRED = "retired"
    FAILED = "failed"
    DELETING = "deleting"


class SemanticDimensionState(str, Enum):
    PENDING = "pending"
    RESOLVED = "resolved"


class SemanticNoteState(str, Enum):
    PENDING = "pending"
    INDEXED = "indexed"
    EXCLUDED = "excluded"
    FAILED = "failed"
    TOMBSTONED = "tombstoned"


class SemanticWorkKind(str, Enum):
    INDEX_NOTE = "index_note"
    DELETE_NOTE_VECTORS = "delete_note_vectors"
    DELETE_GENERATION = "delete_generation"


class SemanticWorkClaimState(str, Enum):
    PENDING = "pending"
    CLAIMED = "claimed"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class SemanticObsoleteVectorClaim:
    """One bounded, generation-homogeneous physical vector cleanup claim."""

    owner_user_id: str
    dataset_id: str
    generation_id: str
    ledger_ids: tuple[str, ...]
    vector_ids: tuple[str, ...]
    claim_token: str
    attempt_count: int


@dataclass(frozen=True, slots=True)
class SemanticOperationReceipt:
    """Bounded durable receipt for a Notes-side semantic mutation."""

    owner_user_id: str
    dataset_id: str
    key_digest: str
    action: str
    request_fingerprint: str
    run_id: str | None
    expected_revision: int
    state: str
    response_json: str | None
    expires_at: str


class SemanticIndexingError(RuntimeError):
    """Stable, content-free semantic indexing or publication failure."""

    def __init__(self, code: str) -> None:
        self.code = code
        self.failure_code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class SemanticIndexConfig:
    owner_user_id: str
    dataset_id: str
    desired_state: SemanticDesiredState
    configuration_revision: int
    semantic_index_revision: int
    capability_revision: str | None
    disclosure_hash: str | None
    compatibility_hash: str | None
    provider: str | None
    model: str | None
    model_revision: str | None
    endpoint_origin_revision: str | None
    endpoint_origin_display: str | None
    data_boundary: str | None
    vector_backend: str | None
    storage_boundary: str | None
    storage_label: str | None
    metric: str
    dimension_state: SemanticDimensionState
    dimensions: int | None
    normalization_version: str
    chunker_version: str
    active_generation_id: str | None
    enabled_at: str | None
    disabled_at: str | None
    consented_at: str | None
    updated_at: str


@dataclass(frozen=True, slots=True)
class SemanticGeneration:
    id: str
    owner_user_id: str
    dataset_id: str
    configuration_revision: int
    state: SemanticGenerationState
    compatibility_hash: str | None
    model_revision: str | None
    dimension_state: SemanticDimensionState
    dimensions: int | None
    root_job_id: str | None
    expected_note_count: int
    expected_chunk_count: int
    published_note_count: int
    published_chunk_count: int
    manifest_hash: str | None
    publication_receipt: str | None
    terminal_error_code: str | None
    created_at: str
    published_at: str | None
    retired_at: str | None
    deleted_at: str | None


@dataclass(frozen=True, slots=True)
class SemanticNoteRecord:
    owner_user_id: str
    dataset_id: str
    generation_id: str
    note_id: str
    content_version: int
    content_fingerprint: str
    dirty_generation: int
    state: SemanticNoteState
    chunk_count: int
    manifest_hash: str | None
    error_code: str | None
    published_at: str | None


@dataclass(frozen=True, slots=True)
class SemanticWorkItem:
    id: str
    owner_user_id: str
    dataset_id: str
    kind: SemanticWorkKind
    note_id: str | None
    generation_id: str | None
    dirty_generation: int | None
    fencing_token: str
    claim_state: SemanticWorkClaimState
    attempt_count: int
    next_eligible_at: str
    claim_token: str | None
    claimed_at: str | None
    error_code: str | None
    created_at: str
    updated_at: str


@dataclass(frozen=True, slots=True)
class SemanticSnapshotSeed:
    """Content-free initial state for one Note in a generation snapshot."""

    note_id: str
    content_version: int
    content_fingerprint: str
    state: SemanticNoteState | str
    planned_chunk_count: int
    error_code: str | None


@dataclass(frozen=True, slots=True)
class SemanticChunkRecord:
    """One content-free chunk manifest row ready for Notes publication."""

    chunk_id: str
    generation_id: str
    note_id: str
    content_version: int
    ordinal: int
    field: str
    start_offset: int
    end_offset: int
    chunk_fingerprint: str
    normalization_version: str
    chunker_version: str


@dataclass(frozen=True, slots=True)
class SemanticProjectionChunk:
    """One current published chunk joined to its live canonical Note."""

    owner_user_id: str
    dataset_id: str
    generation_id: str
    vector_id: str
    note_id: str
    content_version: int
    content_fingerprint: str
    title: str
    content: str
    created_at: datetime | str
    updated_at: datetime | str
    ordinal: int
    field: str
    start_offset: int
    end_offset: int
    chunk_fingerprint: str
    normalization_version: str
    chunker_version: str


@dataclass(frozen=True, slots=True)
class SemanticManifestPublication:
    """Result of one transactional Note manifest or tombstone publication."""

    note_id: str
    generation_id: str
    old_vector_ids: tuple[str, ...]
    new_vector_ids: tuple[str, ...]
    dirty_generation: int
    manifest_hash: str | None


@dataclass(frozen=True, slots=True)
class SemanticGenerationIntegrity:
    """Content-free generation counts and canonical manifest identity."""

    generation_id: str
    generation_fencing_token: str
    expected_note_count: int
    expected_chunk_count: int
    published_note_count: int
    published_chunk_count: int
    terminal_note_count: int
    indexed_note_count: int
    excluded_note_count: int
    failed_note_count: int
    pending_note_count: int
    tombstoned_note_count: int
    eligible_note_count: int
    waived_chunk_count: int
    vector_ids: tuple[str, ...]
    manifest_hash: str
    dimensions: int
    compatibility_hash: str
    terminal_error_code: str | None

    @property
    def degraded(self) -> bool:
        return self.excluded_note_count > 0 or self.failed_note_count > 0
