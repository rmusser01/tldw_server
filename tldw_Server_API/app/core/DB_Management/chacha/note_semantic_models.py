"""Typed persistence records for the Notes semantic-index authority ledger."""

from __future__ import annotations

from dataclasses import dataclass
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
