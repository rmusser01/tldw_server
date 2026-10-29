"""Core-owned evaluation recipe models, re-exported by the API schemas package (TASK-13950.15)."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from tldw_Server_API.app.core.Evaluations.run_state import RunStatus


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
