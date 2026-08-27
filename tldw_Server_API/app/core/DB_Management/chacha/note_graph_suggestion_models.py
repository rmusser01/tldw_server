"""Typed persistence records for reviewable Notes graph suggestions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class NoteGraphSuggestionRunState(str, Enum):
    ADMITTING = "admitting"
    QUEUED = "queued"
    RUNNING = "running"
    CANCELLING = "cancelling"
    PUBLISHING = "publishing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    STALE = "stale"


class NoteGraphSuggestionState(str, Enum):
    STAGED = "staged"
    PENDING = "pending"
    ACCEPTING = "accepting"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    STALE = "stale"


class NoteGraphSuggestionOperationKind(str, Enum):
    RUN_ADMIT = "run_admit"
    RUN_CANCEL = "run_cancel"
    SUGGESTION_ACCEPT = "suggestion_accept"
    SUGGESTION_REJECT = "suggestion_reject"
    REJECTIONS_RESET = "rejections_reset"


class NoteGraphSuggestionReceiptState(str, Enum):
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class NoteGraphSuggestionKind(str, Enum):
    RELATED_NOTE = "related_note"
    TAG = "tag"


class NoteGraphSuggestionEvidenceSide(str, Enum):
    SOURCE = "source"
    TARGET = "target"


class NoteGraphSuggestionEvidenceField(str, Enum):
    TITLE = "title"
    CONTENT = "content"


@dataclass(frozen=True, slots=True)
class NoteGraphSuggestionRun:
    id: str
    owner_user_id: str
    dataset_id: str
    source_note_id: str
    source_fingerprint: str
    state: NoteGraphSuggestionRunState
    revision: int
    created_at: str
    expires_at: str
    provider: str
    model: str
    capability_revision: str
    prompt_contract_version: str
    admission_receipt_id: str | None = None
    job_id: str | None = None
    expected_completion_token: str | None = None
    result_digest: str | None = None
    suggestion_count: int = 0
    related_note_count: int = 0
    tag_count: int = 0
    invalid_item_count: int = 0
    error_code: str | None = None
    guidance_key: str | None = None
    started_at: str | None = None
    completed_at: str | None = None


@dataclass(frozen=True, slots=True)
class NoteGraphSuggestionOperationReceipt:
    id: str
    operation_kind: NoteGraphSuggestionOperationKind
    owner_user_id: str
    dataset_id: str
    source_note_id: str
    resource_identity: str
    idempotency_key_digest: str
    request_fingerprint: str
    state: NoteGraphSuggestionReceiptState
    created_at: str
    expires_at: str
    http_status: int | None = None
    replay_envelope: str | None = None
    completed_at: str | None = None


@dataclass(frozen=True, slots=True)
class NoteGraphSuggestionRejectionSet:
    owner_user_id: str
    dataset_id: str
    source_note_id: str
    source_fingerprint: str
    revision: int
    rejection_count: int
    updated_at: str


@dataclass(frozen=True, slots=True)
class NoteGraphSuggestion:
    id: str
    run_id: str
    owner_user_id: str
    dataset_id: str
    kind: NoteGraphSuggestionKind
    source_note_id: str
    source_fingerprint: str
    state: NoteGraphSuggestionState
    revision: int
    created_at: str
    updated_at: str
    target_note_id: str | None = None
    target_fingerprint: str | None = None
    normalized_tag: str | None = None
    display_tag: str | None = None
    keyword_sync_id: str | None = None
    match_strength: str | None = None
    rationale: str | None = None
    decision_reason: str | None = None
    accepted_resource_identity: str | None = None
    decision_at: str | None = None
    acceptance_lease_token: str | None = None
    acceptance_lease_expires_at: str | None = None
    decision_receipt_id: str | None = None
    expires_at: str | None = None


@dataclass(frozen=True, slots=True)
class NoteGraphSuggestionEvidence:
    suggestion_id: str
    owner_user_id: str
    dataset_id: str
    side: NoteGraphSuggestionEvidenceSide
    ordinal: int
    note_id: str
    field: NoteGraphSuggestionEvidenceField
    content_fingerprint: str
    start_offset: int
    end_offset: int
