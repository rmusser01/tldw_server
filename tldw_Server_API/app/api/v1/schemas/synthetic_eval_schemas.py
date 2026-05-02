"""Pydantic schemas for synthetic evaluation draft persistence."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tldw_Server_API.app.api.v1.schemas.pagination import (
    OffsetPaginationMeta,
    default_offset_pagination_aliases,
)


class SyntheticEvalProvenance(str, Enum):
    """Source provenance for a synthetic evaluation draft sample."""

    REAL = "real"
    REAL_EDITED = "real_edited"
    SYNTHETIC_FROM_CORPUS = "synthetic_from_corpus"
    SYNTHETIC_FROM_SEED_EXAMPLES = "synthetic_from_seed_examples"
    SYNTHETIC_HUMAN_EDITED = "synthetic_human_edited"


class SyntheticEvalReviewState(str, Enum):
    """Human review state for a synthetic draft sample."""

    DRAFT = "draft"
    IN_REVIEW = "in_review"
    EDITED = "edited"
    APPROVED = "approved"
    REJECTED = "rejected"


class SyntheticEvalReviewActionType(str, Enum):
    """Recorded review actions for synthetic draft samples."""

    EDIT = "edit"
    APPROVE = "approve"
    REJECT = "reject"
    REQUEST_CHANGES = "request_changes"
    EDIT_AND_APPROVE = "edit_and_approve"


class SyntheticEvalDraftSampleRecord(BaseModel):
    """Persisted synthetic draft sample row."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str
    recipe_kind: str
    provenance: SyntheticEvalProvenance
    review_state: SyntheticEvalReviewState
    sample_payload: dict[str, Any] = Field(default_factory=dict)
    sample_metadata: dict[str, Any] = Field(default_factory=dict)
    source_kind: str | None = None
    created_by: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class SyntheticEvalDraftSampleCreate(BaseModel):
    """Create payload for a synthetic draft sample."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str
    recipe_kind: str
    provenance: SyntheticEvalProvenance
    review_state: SyntheticEvalReviewState = SyntheticEvalReviewState.DRAFT
    sample_payload: dict[str, Any] = Field(default_factory=dict)
    sample_metadata: dict[str, Any] = Field(default_factory=dict)
    source_kind: str | None = None
    created_by: str | None = None


class SyntheticEvalReviewActionRecord(BaseModel):
    """Persisted review action history entry."""

    model_config = ConfigDict(extra="forbid")

    action_id: str
    sample_id: str
    action: SyntheticEvalReviewActionType
    reviewer_id: str | None = None
    notes: str | None = None
    action_payload: dict[str, Any] = Field(default_factory=dict)
    resulting_review_state: SyntheticEvalReviewState | None = None
    created_at: datetime | None = None


class SyntheticEvalReviewActionCreate(BaseModel):
    """Create payload for a synthetic draft review action."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str
    action: SyntheticEvalReviewActionType
    reviewer_id: str | None = None
    notes: str | None = None
    action_payload: dict[str, Any] = Field(default_factory=dict)
    resulting_review_state: SyntheticEvalReviewState | None = None


class SyntheticEvalPromotionRecord(BaseModel):
    """Persisted promotion record for an approved synthetic draft sample."""

    model_config = ConfigDict(extra="forbid")

    promotion_id: str
    sample_id: str
    dataset_id: str | None = None
    dataset_snapshot_ref: str | None = None
    promoted_by: str | None = None
    promotion_reason: str | None = None
    promotion_metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None


class SyntheticEvalPromotionCreate(BaseModel):
    """Create payload for a synthetic draft promotion record."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str
    dataset_id: str | None = None
    dataset_snapshot_ref: str | None = None
    promoted_by: str | None = None
    promotion_reason: str | None = None
    promotion_metadata: dict[str, Any] = Field(default_factory=dict)


SyntheticEvalReviewStateValue = Literal["draft", "in_review", "edited", "approved", "rejected"]


class SyntheticEvalGenerationRequest(BaseModel):
    """Request payload for generating a draft batch."""

    model_config = ConfigDict(extra="forbid")

    recipe_kind: str
    corpus_scope: dict[str, Any] | list[str] | None = None
    generation_metadata: dict[str, Any] = Field(default_factory=dict)
    context_snapshot_ref: str | None = None
    retrieval_baseline_ref: str | None = None
    reference_answer: str | None = None
    real_examples: list[dict[str, Any]] = Field(default_factory=list)
    seed_examples: list[dict[str, Any]] = Field(default_factory=list)
    target_sample_count: int = Field(default=0, ge=0, le=500)


class SyntheticEvalGenerationResponse(BaseModel):
    """Response payload for a generated draft batch."""

    model_config = ConfigDict(extra="forbid")

    generation_batch_id: str | None = None
    samples: list[SyntheticEvalDraftSampleRecord] = Field(default_factory=list)
    source_breakdown: dict[str, int] = Field(default_factory=dict)
    coverage: dict[str, list[str]] = Field(default_factory=dict)
    missing_coverage: dict[str, list[str]] = Field(default_factory=dict)
    corpus_scope: dict[str, Any] = Field(default_factory=dict)


class SyntheticEvalQueueResponse(BaseModel):
    """Filtered queue listing response."""

    model_config = ConfigDict(extra="forbid")

    data: list[SyntheticEvalDraftSampleRecord] = Field(default_factory=list)
    total: int = 0
    has_more: bool | None = Field(default=None, description="Alias for pagination.has_more")
    next_offset: int | None = Field(default=None, ge=0, description="Alias for pagination.next_offset")
    pagination: OffsetPaginationMeta

    @model_validator(mode="after")
    def _default_pagination_aliases(self):
        return default_offset_pagination_aliases(self)


class SyntheticEvalReviewRequest(BaseModel):
    """Apply a review action to one draft sample."""

    model_config = ConfigDict(extra="forbid")

    action: SyntheticEvalReviewActionType
    reviewer_id: str | None = None
    notes: str | None = None
    action_payload: dict[str, Any] = Field(default_factory=dict)
    resulting_review_state: SyntheticEvalReviewState | None = None


class SyntheticEvalPromotionRequest(BaseModel):
    """Promote approved synthetic review items into a dataset."""

    model_config = ConfigDict(extra="forbid")

    sample_ids: list[str] = Field(..., min_length=1)
    dataset_name: str
    dataset_description: str | None = None
    dataset_metadata: dict[str, Any] = Field(default_factory=dict)
    promoted_by: str | None = None
    promotion_reason: str | None = None


class SyntheticEvalPromotionResponse(BaseModel):
    """Response payload for dataset promotion."""

    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    dataset_snapshot_ref: str
    promotion_ids: list[str] = Field(default_factory=list)
    sample_count: int = 0
