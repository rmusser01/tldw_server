"""Shared contracts for per-occurrence playlist ingest persistence."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Conservative limits for explicitly reviewed metadata mutations.
MAX_METADATA_PATCH_TEXT_LENGTH = 500
MAX_METADATA_PATCH_KEYWORDS = 100
MAX_METADATA_PATCH_KEYWORD_LENGTH = 128


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
        return value.strip() if isinstance(value, str) else value

    @field_validator("keywords_add", mode="before")
    @classmethod
    def _normalize_keywords(cls, value: object) -> object:
        if not isinstance(value, list):
            return value
        normalized: list[str] = []
        for keyword in value:
            if not isinstance(keyword, str):
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
        return value.strip() if isinstance(value, str) else value

    @model_validator(mode="after")
    def _validate_terminal_outcome(self) -> RunItemSnapshot:
        if (self.state is RunItemState.TERMINAL) != (self.outcome is not None):
            raise ValueError("outcome is required exactly when state is terminal")
        return self
