from __future__ import annotations

"""Pydantic schemas for the companion personalization API."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tldw_Server_API.app.api.v1.schemas.pagination import OffsetPaginationMeta


def _default_offset_pagination_aliases(response):
    if response.has_more is None:
        response.has_more = response.pagination.has_more
    if response.next_offset is None:
        response.next_offset = response.pagination.next_offset
    return response


class CompanionActivityItem(BaseModel):
    """One persisted companion activity event."""

    id: str
    event_type: str
    source_type: str
    source_id: str
    surface: str
    tags: list[str] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class CompanionActivityListResponse(BaseModel):
    """Paginated response for companion activity items."""

    items: list[CompanionActivityItem]
    total: int
    limit: int
    offset: int
    has_more: bool | None = Field(default=None, description="Alias for pagination.has_more")
    next_offset: int | None = Field(default=None, ge=0, description="Alias for pagination.next_offset")
    pagination: OffsetPaginationMeta

    @model_validator(mode="after")
    def _default_pagination_aliases(self):
        return _default_offset_pagination_aliases(self)


class CompanionActivityDetail(CompanionActivityItem):
    """One detailed companion activity entry."""


class CompanionActivityCreate(BaseModel):
    """Request payload for explicit companion activity capture."""

    event_type: str
    source_type: str
    source_id: str
    surface: str
    dedupe_key: str | None = None
    tags: list[str] = Field(default_factory=list)
    provenance: dict[str, Any]
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_provenance(self) -> "CompanionActivityCreate":
        if not self.provenance:
            raise ValueError("provenance is required")
        return self


class CompanionCheckInCreate(BaseModel):
    """Request payload for a manual companion check-in."""

    title: str | None = None
    summary: str
    surface: str | None = None
    tags: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")

    @field_validator("title", "summary", "surface", mode="before")
    @classmethod
    def _strip_text(cls, value: Any) -> Any:
        return value.strip() if isinstance(value, str) else value

    @field_validator("tags", mode="before")
    @classmethod
    def _normalize_tags(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise TypeError("tags must be a list")
        return [str(item).strip() for item in value if str(item).strip()]

    @model_validator(mode="after")
    def validate_summary(self) -> "CompanionCheckInCreate":
        if not self.summary:
            raise ValueError("summary is required")
        if self.title == "":
            self.title = None
        if self.surface == "":
            self.surface = None
        return self


class CompanionKnowledgeCard(BaseModel):
    """One derived companion knowledge card."""

    id: str
    card_type: str
    title: str
    summary: str
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    score: float
    status: str
    updated_at: datetime


class CompanionKnowledgeListResponse(BaseModel):
    """List response for companion knowledge cards."""

    items: list[CompanionKnowledgeCard]
    total: int


class CompanionKnowledgeDetail(CompanionKnowledgeCard):
    """Detailed response for one companion knowledge card."""

    evidence_events: list[CompanionActivityItem] = Field(default_factory=list)
    evidence_goals: list["CompanionGoal"] = Field(default_factory=list)


class CompanionGoal(BaseModel):
    """One tracked companion goal."""

    id: str
    title: str
    description: str | None = None
    goal_type: str
    config: dict[str, Any] = Field(default_factory=dict)
    progress: dict[str, Any] = Field(default_factory=dict)
    origin_kind: str = "manual"
    progress_mode: str = "manual"
    derivation_key: str | None = None
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    status: str
    created_at: datetime
    updated_at: datetime


class CompanionGoalCreate(BaseModel):
    """Request payload for creating a companion goal."""

    title: str
    description: str | None = None
    goal_type: str
    config: dict[str, Any] = Field(default_factory=dict)
    progress: dict[str, Any] = Field(default_factory=dict)
    status: str = "active"


class CompanionGoalUpdate(BaseModel):
    """Request payload for updating a companion goal."""

    title: str | None = None
    description: str | None = None
    config: dict[str, Any] | None = None
    progress: dict[str, Any] | None = None
    status: str | None = None

    model_config = ConfigDict(extra="forbid")


class CompanionGoalListResponse(BaseModel):
    """List response for companion goals."""

    items: list[CompanionGoal]
    total: int


class CompanionReflectionItem(BaseModel):
    """One companion reflection entry shown in the workspace."""

    id: str
    cadence: str | None = None
    summary: str
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    delivery_decision: str | None = None
    delivery_reason: str | None = None
    theme_key: str | None = None
    signal_strength: float | None = None
    follow_up_prompts: list[dict[str, Any]] = Field(default_factory=list)
    created_at: datetime


class CompanionReflectionDetail(BaseModel):
    """Detailed response for one companion reflection entry."""

    id: str
    title: str
    cadence: str | None = None
    summary: str
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    delivery_decision: str | None = None
    delivery_reason: str | None = None
    theme_key: str | None = None
    signal_strength: float | None = None
    follow_up_prompts: list[dict[str, Any]] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    activity_events: list[CompanionActivityItem] = Field(default_factory=list)
    knowledge_cards: list[CompanionKnowledgeCard] = Field(default_factory=list)
    goals: list[CompanionGoal] = Field(default_factory=list)


class CompanionFollowUpPrompt(BaseModel):
    """One deterministic follow-up prompt suggestion."""

    prompt_id: str
    label: str
    prompt_text: str
    prompt_type: str
    source_reflection_id: str | None = None
    source_evidence_ids: list[str] = Field(default_factory=list)


class CompanionConversationPromptsResponse(BaseModel):
    """Prompt suggestions for the companion conversation surface."""

    prompt_source_kind: str
    prompt_source_id: str | None = None
    prompts: list[CompanionFollowUpPrompt] = Field(default_factory=list)


CompanionLifecycleScope = Literal["knowledge", "reflections", "derived_goals", "goal_progress"]


class CompanionPurgeRequest(BaseModel):
    """Request payload for purging a scoped slice of companion state."""

    scope: CompanionLifecycleScope

    model_config = ConfigDict(extra="forbid")


class CompanionRebuildRequest(BaseModel):
    """Request payload for rebuilding a scoped slice of companion state."""

    scope: CompanionLifecycleScope

    model_config = ConfigDict(extra="forbid")


class CompanionLifecycleResponse(BaseModel):
    """Response payload for purge and rebuild lifecycle operations."""

    status: str
    scope: CompanionLifecycleScope
    deleted_counts: dict[str, int] = Field(default_factory=dict)
    rebuilt_counts: dict[str, int] = Field(default_factory=dict)
    job_id: int | None = None
    job_uuid: str | None = None


CompanionKnowledgeDetail.model_rebuild()
