"""Shared Pydantic schemas for study-suggestion status and action contracts."""

from typing import Any, Literal

from pydantic import BaseModel, Field


SuggestionApiStatus = Literal["none", "pending", "ready", "failed"]


class SuggestionRefreshRequest(BaseModel):
    """Request body for explicitly refreshing study suggestions."""

    reason: str | None = None


class SuggestionStatusResponse(BaseModel):
    """Status payload returned when checking suggestion state for an anchor."""

    anchor_type: str
    anchor_id: int
    status: SuggestionApiStatus
    job_id: int | None = None
    snapshot_id: int | None = None


class SuggestionActionResponse(BaseModel):
    """Response describing the concrete study artifact opened or generated."""

    disposition: Literal["opened_existing", "generated"]
    snapshot_id: int
    selection_fingerprint: str
    target_service: Literal["quiz", "flashcards"]
    target_type: str
    target_id: str


class SuggestionJobSummary(BaseModel):
    """Serialized Jobs metadata returned by study-suggestion endpoints."""

    id: int
    status: str


class SuggestionJobAcceptedResponse(BaseModel):
    """Accepted response for queued study-suggestion refresh work."""

    job: SuggestionJobSummary


class SuggestionSnapshotResource(BaseModel):
    """Frozen snapshot payload returned to clients."""

    id: int
    service: str
    activity_type: str
    anchor_type: str
    anchor_id: int
    suggestion_type: str
    status: str
    payload: dict[str, Any] | list[Any] | str
    user_selection: dict[str, Any] | list[Any] | str | None = None
    refreshed_from_snapshot_id: int | None = None
    created_at: str | None = None
    last_modified: str | None = None


class SuggestionSnapshotResponse(BaseModel):
    """Read response for a snapshot plus live evidence hydration."""

    snapshot: SuggestionSnapshotResource
    live_evidence: dict[str, Any]


class SuggestionActionRequest(BaseModel):
    """Request body for triggering a study-suggestion follow-up action."""

    target_service: Literal["quiz", "flashcards"]
    target_type: str = Field(..., min_length=1)
    action_kind: str = Field(..., min_length=1)
    selected_topic_ids: list[str] = Field(default_factory=list)
    selected_topic_edits: list[dict[str, str]] = Field(default_factory=list)
    manual_topic_labels: list[str] = Field(default_factory=list)
    has_explicit_selection: bool = False
    generator_version: str = Field(default="v1", min_length=1)
    force_regenerate: bool = False
