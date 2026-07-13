"""Shared contracts for per-occurrence playlist ingest persistence."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Conservative limits for explicitly reviewed metadata mutations.
MAX_METADATA_PATCH_TEXT_LENGTH = 500
MAX_METADATA_PATCH_KEYWORDS = 100
MAX_METADATA_PATCH_KEYWORD_LENGTH = 128
MAX_PLAYLIST_PREFLIGHT_SELECTIONS = 500


class DuplicatePolicy(str, Enum):
    """Explicit action choices for an existing media item."""

    SKIP = "skip"
    INCLUDE_EXISTING = "include_existing"
    UPDATE_METADATA_ONLY = "update_metadata_only"
    OVERWRITE = "overwrite"


class RunItemState(str, Enum):
    """Server-owned lifecycle states for one ingest occurrence."""

    STAGED = "staged"
    PREPARING = "preparing"
    AWAITING_UPLOAD = "awaiting_upload"
    SUBMIT_PENDING = "submit_pending"
    QUEUED = "queued"
    RUNNING = "running"
    CANCELLATION_REQUESTED = "cancellation_requested"
    STATUS_UNAVAILABLE = "status_unavailable"
    TERMINAL = "terminal"


class RunItemOutcome(str, Enum):
    """Terminal results, kept separate from lifecycle state."""

    COMPLETED = "completed"
    INCLUDED_EXISTING = "included_existing"
    METADATA_UPDATED = "metadata_updated"
    SKIPPED_EXISTING = "skipped_existing"
    SUBMIT_FAILED = "submit_failed"
    PROCESSING_FAILED = "processing_failed"
    METADATA_UPDATE_FAILED = "metadata_update_failed"
    CANCELLED = "cancelled"


class MetadataPatch(BaseModel):
    """Explicit, bounded metadata fields reviewed for an existing item."""

    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(default=None, min_length=1, max_length=MAX_METADATA_PATCH_TEXT_LENGTH)
    author: str | None = Field(default=None, min_length=1, max_length=MAX_METADATA_PATCH_TEXT_LENGTH)
    keywords_add: list[str] | None = Field(default=None, min_length=1, max_length=MAX_METADATA_PATCH_KEYWORDS)

    @field_validator("title", "author", mode="before")
    @classmethod
    def _strip_text(cls, value: object) -> object:
        if value is None:
            return None
        if type(value) is not str:
            raise ValueError("metadata patch text must be a string")
        return value.strip()

    @field_validator("keywords_add", mode="before")
    @classmethod
    def _normalize_keywords(cls, value: object) -> object:
        if value is None:
            return None
        if type(value) is not list:
            raise ValueError("keywords_add must be a list")
        normalized: list[str] = []
        for keyword in value:
            if type(keyword) is not str:
                raise ValueError("keywords_add entries must be strings")
            trimmed = keyword.strip()
            if not trimmed:
                raise ValueError("keywords_add entries must not be blank")
            if len(trimmed) > MAX_METADATA_PATCH_KEYWORD_LENGTH:
                raise ValueError(
                    f"keywords_add entries must be {MAX_METADATA_PATCH_KEYWORD_LENGTH} characters or fewer"
                )
            normalized.append(trimmed)
        return normalized

    @model_validator(mode="after")
    def _require_change(self) -> MetadataPatch:
        if self.title is None and self.author is None and self.keywords_add is None:
            raise ValueError("metadata_patch must contain at least one change")
        return self


class ReviewOverride(BaseModel):
    """Review-time action and optional metadata patch."""

    model_config = ConfigDict(extra="forbid")

    duplicate_policy: DuplicatePolicy
    metadata_patch: MetadataPatch | None = None

    @model_validator(mode="after")
    def _validate_policy_patch(self) -> ReviewOverride:
        if self.duplicate_policy is DuplicatePolicy.UPDATE_METADATA_ONLY and self.metadata_patch is None:
            raise ValueError("update_metadata_only requires metadata_patch")
        if (
            self.duplicate_policy in {DuplicatePolicy.SKIP, DuplicatePolicy.INCLUDE_EXISTING}
            and self.metadata_patch is not None
        ):
            raise ValueError(f"{self.duplicate_policy.value} does not allow metadata_patch")
        return self


class RunItemSnapshot(BaseModel):
    """Current server snapshot for one immutable run occurrence."""

    model_config = ConfigDict(extra="forbid")

    occurrence_id: str = Field(..., min_length=1, max_length=255)
    ordinal: int = Field(..., ge=1)
    state: RunItemState
    outcome: RunItemOutcome | None = None
    progress_percent: float | None = Field(default=None, ge=0, le=100)
    progress_message: str | None = Field(default=None, max_length=1000)
    job_id: int | None = Field(default=None, ge=1)
    media_id: int | None = Field(default=None, ge=1)
    attempt: int = Field(default=1, ge=1)
    retryable: bool = False

    @field_validator("occurrence_id", mode="before")
    @classmethod
    def _strip_occurrence_id(cls, value: object) -> object:
        if type(value) is not str:
            raise ValueError("occurrence_id must be a string")
        return value.strip()

    @model_validator(mode="after")
    def _validate_terminal_outcome(self) -> RunItemSnapshot:
        if (self.state is RunItemState.TERMINAL) != (self.outcome is not None):
            raise ValueError("outcome is required exactly when state is terminal")
        return self


class PlaylistPreflightCreateRequest(BaseModel):
    """Bounded asynchronous playlist inspection request."""

    model_config = ConfigDict(extra="forbid")

    url: str = Field(..., min_length=1, max_length=8192)
    max_items: int = Field(default=100, ge=1, le=500)
    timeout_seconds: int = Field(default=20, ge=1, le=60)

    @field_validator("url", mode="before")
    @classmethod
    def _strip_url(cls, value: object) -> object:
        if type(value) is not str:
            raise ValueError("url must be a string")
        return value.strip()


class PlaylistPreflightLimits(BaseModel):
    """Safe admission limits advertised with an accepted resource."""

    model_config = ConfigDict(extra="forbid")

    max_items: int = Field(..., ge=1, le=500)
    global_capacity: int = Field(..., ge=1)
    owner_capacity: int = Field(..., ge=1)


class PlaylistPreflightAcceptedResponse(BaseModel):
    """Versioned response returned after durable resource/job binding."""

    model_config = ConfigDict(extra="forbid")

    contract_version: Literal[2] = 2
    preflight_id: str
    status: Literal["pending"] = "pending"
    status_url: str
    items_url: str
    expires_at: datetime
    limits: PlaylistPreflightLimits


class PlaylistPreflightSummaryResponse(BaseModel):
    """Owner-scoped asynchronous preflight status."""

    model_config = ConfigDict(extra="forbid")

    contract_version: Literal[2] = 2
    preflight_id: str
    status: Literal["pending", "running", "ready", "blocked", "cancelled", "expired"]
    source_url: str
    source_kind: str
    playlist_id: str | None = None
    summary: dict[str, Any] | None = None
    error: dict[str, str] | None = None
    created_at: datetime
    updated_at: datetime
    expires_at: datetime


class PlaylistPreflightItemResponse(BaseModel):
    """One immutable server-issued playlist occurrence."""

    model_config = ConfigDict(extra="forbid")

    occurrence_id: str
    ordinal: int = Field(..., ge=1)
    occurrence_index_for_source: int | None = Field(default=None, ge=1)
    source_url: str | None = None
    normalized_source_id: str | None = None
    source_kind: str
    availability: str | None = None
    duplicate_status: str | None = None
    duplicate_of_occurrence_id: str | None = None
    selected_by_default: bool | None = None
    display_metadata: dict[str, Any] = Field(default_factory=dict)


class PlaylistPreflightItemsPageResponse(BaseModel):
    """Bounded immutable preflight item page."""

    model_config = ConfigDict(extra="forbid")

    contract_version: Literal[2] = 2
    preflight_id: str
    items: list[PlaylistPreflightItemResponse]
    next_cursor: str | None = None


class PlaylistMaterializationCreateRequest(BaseModel):
    """Only server occurrence identities may be selected for materialization."""

    model_config = ConfigDict(extra="forbid")

    occurrence_ids: list[str] = Field(
        ...,
        min_length=1,
        max_length=MAX_PLAYLIST_PREFLIGHT_SELECTIONS,
    )

    @field_validator("occurrence_ids", mode="before")
    @classmethod
    def _validate_occurrence_ids(cls, value: object) -> object:
        if type(value) is not list:
            raise ValueError("occurrence_ids must be a list")
        normalized: list[str] = []
        for occurrence_id in value:
            if type(occurrence_id) is not str:
                raise ValueError("occurrence_ids entries must be strings")
            trimmed = occurrence_id.strip()
            if not trimmed or len(trimmed) > 255:
                raise ValueError("occurrence_ids entries must be between 1 and 255 characters")
            normalized.append(trimmed)
        if len(set(normalized)) != len(normalized):
            raise ValueError("occurrence_ids must be unique")
        return normalized


class PlaylistMaterializationItemResponse(BaseModel):
    """Compact authoritative identity copied from a completed snapshot."""

    model_config = ConfigDict(extra="forbid")

    occurrence_id: str
    ordinal: int = Field(..., ge=1)
    source_url: str
    normalized_source_id: str | None = None
    source_kind: str
    display_metadata: dict[str, Any] = Field(default_factory=dict)


class PlaylistMaterializationResponse(BaseModel):
    """Owner-bound materialization for a staged Quick Ingest draft."""

    model_config = ConfigDict(extra="forbid")

    contract_version: Literal[2] = 2
    materialization_id: str
    preflight_id: str
    status: Literal["ready"] = "ready"
    items: list[PlaylistMaterializationItemResponse]
    expires_at: datetime
