"""Core-owned synthetic evaluation models, re-exported by the API schemas package (TASK-13950.15)."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


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


class SyntheticEvalReviewActionCreate(BaseModel):
    """Create payload for a synthetic draft review action."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str
    action: SyntheticEvalReviewActionType
    reviewer_id: str | None = None
    notes: str | None = None
    action_payload: dict[str, Any] = Field(default_factory=dict)
    resulting_review_state: SyntheticEvalReviewState | None = None


class SyntheticEvalPromotionCreate(BaseModel):
    """Create payload for a synthetic draft promotion record."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str
    dataset_id: str | None = None
    dataset_snapshot_ref: str | None = None
    promoted_by: str | None = None
    promotion_reason: str | None = None
    promotion_metadata: dict[str, Any] = Field(default_factory=dict)
