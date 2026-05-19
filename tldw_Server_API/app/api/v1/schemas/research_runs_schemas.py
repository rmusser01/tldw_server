"""Schemas for deep research session APIs."""

from __future__ import annotations

from typing import Any
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

MAX_RESEARCH_FOLLOW_UP_QUESTION_LENGTH = 4000
MAX_RESEARCH_FOLLOW_UP_OUTLINE_ITEMS = 7
MAX_RESEARCH_FOLLOW_UP_OUTLINE_TITLE_LENGTH = 500
MAX_RESEARCH_FOLLOW_UP_FOCUS_AREA_LENGTH = 200
MAX_RESEARCH_FOLLOW_UP_KEY_CLAIMS = 5
MAX_RESEARCH_FOLLOW_UP_CLAIM_ID_LENGTH = 128
MAX_RESEARCH_FOLLOW_UP_CLAIM_TEXT_LENGTH = 4000
MAX_RESEARCH_FOLLOW_UP_UNRESOLVED_QUESTIONS = 5
MAX_RESEARCH_FOLLOW_UP_UNRESOLVED_QUESTION_LENGTH = 1000

ResearchFollowUpUnresolvedQuestion = Annotated[
    str,
    Field(min_length=1, max_length=MAX_RESEARCH_FOLLOW_UP_UNRESOLVED_QUESTION_LENGTH),
]


class ResearchFollowUpOutlineItem(BaseModel):
    """Compact outline item carried with a bounded follow-up seed."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(..., min_length=1, max_length=MAX_RESEARCH_FOLLOW_UP_OUTLINE_TITLE_LENGTH)
    focus_area: str | None = Field(default=None, min_length=1, max_length=MAX_RESEARCH_FOLLOW_UP_FOCUS_AREA_LENGTH)


class ResearchFollowUpClaimItem(BaseModel):
    """Compact claim item carried with a bounded follow-up seed."""

    model_config = ConfigDict(extra="forbid")

    claim_id: str = Field(..., min_length=1, max_length=MAX_RESEARCH_FOLLOW_UP_CLAIM_ID_LENGTH)
    text: str = Field(..., min_length=1, max_length=MAX_RESEARCH_FOLLOW_UP_CLAIM_TEXT_LENGTH)


class ResearchFollowUpVerificationSummary(BaseModel):
    """Verification summary for bounded follow-up background."""

    model_config = ConfigDict(extra="forbid")

    supported_claim_count: int = Field(..., ge=0)
    unsupported_claim_count: int = Field(..., ge=0)


class ResearchFollowUpSourceTrustSummary(BaseModel):
    """Source trust summary for bounded follow-up background."""

    model_config = ConfigDict(extra="forbid")

    high_trust_count: int = Field(..., ge=0)
    low_trust_count: int = Field(..., ge=0)


class ResearchRunFollowUpBackground(BaseModel):
    """Bounded background attached to a follow-up research launch."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(..., min_length=1, max_length=MAX_RESEARCH_FOLLOW_UP_QUESTION_LENGTH)
    outline: list[ResearchFollowUpOutlineItem] = Field(default_factory=list, max_length=MAX_RESEARCH_FOLLOW_UP_OUTLINE_ITEMS)
    key_claims: list[ResearchFollowUpClaimItem] = Field(default_factory=list, max_length=MAX_RESEARCH_FOLLOW_UP_KEY_CLAIMS)
    unresolved_questions: list[ResearchFollowUpUnresolvedQuestion] = Field(
        default_factory=list,
        max_length=MAX_RESEARCH_FOLLOW_UP_UNRESOLVED_QUESTIONS,
    )
    verification_summary: ResearchFollowUpVerificationSummary
    source_trust_summary: ResearchFollowUpSourceTrustSummary


class ResearchRunFollowUpCreateRequest(BaseModel):
    """Optional follow-up seed used to launch a new deep research run from chat."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(..., min_length=1, max_length=MAX_RESEARCH_FOLLOW_UP_QUESTION_LENGTH)
    background: ResearchRunFollowUpBackground | None = None


class ResearchChatHandoffCreateRequest(BaseModel):
    """Optional originating chat linkage for research runs launched from chat."""

    chat_id: str = Field(..., min_length=1, max_length=255)
    launch_message_id: str | None = Field(default=None, min_length=1, max_length=255)


class ResearchRunCreateRequest(BaseModel):
    """Request body for creating a deep research session."""

    query: str = Field(..., min_length=1, max_length=4000)
    source_policy: str = Field(default="balanced", min_length=1, max_length=64)
    autonomy_mode: str = Field(default="checkpointed", min_length=1, max_length=64)
    limits_json: dict[str, Any] | None = None
    provider_overrides: dict[str, Any] | None = None
    chat_handoff: ResearchChatHandoffCreateRequest | None = None
    follow_up: ResearchRunFollowUpCreateRequest | None = None


class ResearchCheckpointPatchApproveRequest(BaseModel):
    """Optional user edits applied when approving a checkpoint."""

    patch_payload: dict[str, Any] | None = None


class ResearchRunResponse(BaseModel):
    """Current state returned for research session operations."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    status: str
    phase: str
    control_state: str = "running"
    progress_percent: float | None = None
    progress_message: str | None = None
    active_job_id: str | None = None
    latest_checkpoint_id: str | None = None
    completed_at: str | None = None
    chat_id: str | None = None


class ResearchRunListItemResponse(ResearchRunResponse):
    """List item returned for recent deep research run queries."""

    query: str
    created_at: str
    updated_at: str


class ResearchRunDeleteResponse(BaseModel):
    """Envelope returned after deleting a deep research run."""

    deleted: bool


class ResearchCheckpointSummary(BaseModel):
    """Current checkpoint summary included in research stream snapshots."""

    checkpoint_id: str
    checkpoint_type: str
    status: str
    proposed_payload: dict[str, Any]
    resolution: str | None = None


class ResearchArtifactManifestEntry(BaseModel):
    """Latest-version artifact metadata for research stream snapshots."""

    artifact_name: str
    artifact_version: int
    content_type: str
    phase: str
    job_id: str | None = None


class ResearchRunSnapshotResponse(BaseModel):
    """Reconnect-safe snapshot for live research progress streams."""

    run: ResearchRunResponse
    latest_event_id: int = 0
    checkpoint: ResearchCheckpointSummary | None = None
    artifacts: list[ResearchArtifactManifestEntry] = Field(default_factory=list)


class ResearchArtifactResponse(BaseModel):
    """Typed artifact response for deep research polling APIs."""

    artifact_name: str
    content_type: str
    content: Any
