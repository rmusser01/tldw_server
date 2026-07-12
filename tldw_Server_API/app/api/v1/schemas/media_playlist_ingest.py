"""Shared contracts for per-occurrence playlist ingest persistence."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


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


class ReviewOverride(BaseModel):
    """Review-time action and optional metadata patch."""

    model_config = ConfigDict(extra="forbid")

    duplicate_policy: DuplicatePolicy
    metadata_patch: dict[str, object] | None = None


class RunItemSnapshot(BaseModel):
    """Current server snapshot for one immutable run occurrence."""

    model_config = ConfigDict(extra="forbid")

    occurrence_id: str = Field(..., min_length=1, max_length=255)
    ordinal: int = Field(..., ge=1)
    state: RunItemState
    outcome: RunItemOutcome | None = None
    progress_percent: float | None = Field(default=None, ge=0, le=100)
    progress_message: str | None = Field(default=None, max_length=1000)
    job_id: int | None = None
    media_id: int | None = None
    attempt: int = Field(default=1, ge=1)
    retryable: bool = False
