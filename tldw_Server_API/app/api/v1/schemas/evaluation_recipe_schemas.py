"""Pydantic schemas for evaluation recipe registry and run persistence."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field

from tldw_Server_API.app.api.v1.schemas.evaluation_schemas_unified import RunStatus


class ReviewState(str, Enum):
    """Human review state for a recipe run."""

    NOT_REQUIRED = "not_required"
    NEEDS_REVIEW = "needs_review"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    REJECTED = "rejected"


class ConfidenceSummary(BaseModel):
    """Typed confidence summary persisted with recipe runs."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["aggregate", "bootstrap", "judge", "heuristic"] = "aggregate"
    confidence: float = Field(..., ge=0.0, le=1.0)
    sample_count: int = Field(default=0, ge=0)
    spread: float | None = Field(default=None, ge=0.0)
    margin: float | None = Field(default=None, ge=0.0)
    judge_agreement: float | None = Field(default=None, ge=0.0, le=1.0)
    notes: str | None = None


class RecommendationSlot(BaseModel):
    """Recommendation slot payload that can represent a null winner explicitly."""

    model_config = ConfigDict(extra="forbid")

    candidate_run_id: str | None = None
    reason_code: str | None = None
    explanation: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


RecipeEvaluationMode = Literal["fixed_context", "live_end_to_end"]
RecipeSupervisionMode = Literal["rubric", "reference_answer", "pairwise", "mixed"]
RecipeCandidateDimension = Literal[
    "generation_model",
    "prompt_variant",
    "formatting_citation_mode",
]
EmbeddingCandidateStatus = Literal[
    "ready",
    "missing_key",
    "disallowed_provider",
    "disallowed_model",
    "quota_risk",
    "unknown",
]


class RagAnswerQualityCapabilities(TypedDict, total=False):
    """Launch contract for the answer-quality RAG recipe."""

    evaluation_modes: list[RecipeEvaluationMode]
    supervision_modes: list[RecipeSupervisionMode]
    candidate_dimensions: list[RecipeCandidateDimension]


class RagAnswerQualityDefaultRunConfig(TypedDict, total=False):
    """Default run config for the answer-quality RAG recipe."""

    evaluation_mode: RecipeEvaluationMode
    supervision_mode: RecipeSupervisionMode
    candidate_dimensions: list[RecipeCandidateDimension]


class RecipeManifest(BaseModel):
    """Recipe manifest exposed by the registry."""

    model_config = ConfigDict(extra="forbid")

    recipe_id: str
    recipe_version: str
    name: str
    description: str
    launchable: bool = True
    supported_modes: list[Literal["labeled", "unlabeled"]] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    capabilities: dict[str, Any] = Field(default_factory=dict)
    default_run_config: dict[str, Any] = Field(default_factory=dict)


class RecipeLaunchReadiness(BaseModel):
    """User-facing launch readiness for a recipe."""

    model_config = ConfigDict(extra="forbid")

    recipe_id: str
    ready: bool
    can_enqueue_runs: bool
    can_reuse_completed_runs: bool = True
    runtime_checks: dict[str, bool] = Field(default_factory=dict)
    message: str | None = None


class RecipeDatasetValidationRequest(BaseModel):
    """Typed request payload for recipe dataset validation."""

    model_config = ConfigDict(extra="forbid")

    dataset_id: str | None = None
    dataset: list[dict[str, Any]] | None = None
    run_config: dict[str, Any] | None = None


class RecipeRunCreateRequest(RecipeDatasetValidationRequest):
    """Typed request payload for recipe-run creation."""

    run_config: dict[str, Any] = Field(default_factory=dict)
    force_rerun: bool = False


class RecipeDatasetValidationResponse(BaseModel):
    """Normalized validation payload with recipe-specific extension fields."""

    model_config = ConfigDict(extra="allow")

    valid: bool
    errors: list[str] = Field(default_factory=list)
    dataset_mode: str | None = None
    sample_count: int = Field(default=0, ge=0)
    dataset_snapshot_ref: str | None = None
    dataset_content_hash: str | None = None


class EmbeddingRecipeCandidateHint(BaseModel):
    """Candidate hint for the embeddings model selection recipe."""

    model_config = ConfigDict(extra="forbid")

    provider: str
    model: str
    is_local: bool = False
    default: bool = False
    status: EmbeddingCandidateStatus = "unknown"
    status_reason: str | None = None
    dimensions: int | None = Field(default=None, ge=1)
    revision: str | None = None
    cost_hint: str | None = None


class EmbeddingRecipeCandidatesResponse(BaseModel):
    """Candidate discovery response for the embeddings model selection recipe."""

    model_config = ConfigDict(extra="forbid")

    recipe_id: Literal["embeddings_model_selection"] = "embeddings_model_selection"
    current: EmbeddingRecipeCandidateHint | None = None
    candidates: list[EmbeddingRecipeCandidateHint] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class RecipeRecommendationApplyPreviewRequest(BaseModel):
    """Request payload for previewing a recipe recommendation apply action."""

    model_config = ConfigDict(extra="forbid")

    slot_name: str = "best_overall"
    candidate_run_id: str | None = None


class RecipeRecommendationApplyRequest(RecipeRecommendationApplyPreviewRequest):
    """Request payload for applying a recipe recommendation."""

    confirmed_provider: str
    confirmed_model: str


class RecipeRecommendationApplyPreviewResponse(BaseModel):
    """Secret-free preview of applying a recipe recommendation."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    recipe_id: str
    slot_name: str
    candidate_run_id: str | None = None
    apply_eligible: bool
    apply_available: bool = False
    blocked_reason: str | None = None
    warnings: list[str] = Field(default_factory=list)
    current: dict[str, str | None] = Field(default_factory=dict)
    proposed: dict[str, str | None] = Field(default_factory=dict)
    affected_config: dict[str, str] = Field(default_factory=dict)
    copy_config: dict[str, dict[str, str]] = Field(default_factory=dict)
    reindex_required: bool = True


class RecipeRecommendationApplyResponse(RecipeRecommendationApplyPreviewResponse):
    """Response payload for a completed recipe recommendation apply action."""

    applied: bool = False
    backup_path: str | None = None
    audit_ref: str | None = None


class RecipeRunRecord(BaseModel):
    """Persistent recipe run record."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    recipe_id: str
    recipe_version: str
    status: RunStatus
    review_state: ReviewState = ReviewState.NOT_REQUIRED
    dataset_snapshot_ref: str | None = None
    dataset_content_hash: str | None = None
    confidence_summary: ConfidenceSummary | None = None
    recommendation_slots: dict[str, RecommendationSlot] = Field(default_factory=dict)
    child_run_ids: list[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
