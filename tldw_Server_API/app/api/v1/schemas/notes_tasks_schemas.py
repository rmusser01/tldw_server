"""Schemas for note-backed task REST operations."""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

TaskStatusValue = Literal["open", "done"]
ProjectionStatusValue = Literal["live", "unlinked", "ambiguous", "deleted"]
TaskProjectionDriftAction = Literal[
    "keep_task",
    "accept_markdown",
    "unlink",
    "dismiss",
]


class TaskMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    due_date: str | None = Field(None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    priority: Literal["high", "medium", "low"] | None = None
    estimate: str | None = Field(None, pattern=r"^\d+[mhd]$")

    @field_validator("due_date")
    @classmethod
    def validate_due_date(cls, value: str | None) -> str | None:
        if value is None:
            return value
        try:
            date.fromisoformat(value)
        except ValueError as exc:
            raise ValueError("due_date must be a real ISO date.") from exc
        return value

    def as_compact_dict(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)


class TaskProjectionResponse(BaseModel):
    note_id: str
    note_version: int
    line_number: int
    start_offset: int
    end_offset: int
    raw_line: str
    has_child_content: bool
    projection_status: ProjectionStatusValue


class TaskNoteSummaryResponse(BaseModel):
    id: str
    title: str
    version: int


class TaskResponse(BaseModel):
    id: str
    note_id: str
    text: str
    status: TaskStatusValue
    metadata: dict[str, Any] = Field(default_factory=dict)
    projection_status: ProjectionStatusValue
    version: int
    created_at: str | None = None
    updated_at: str | None = None
    completed_at: str | None = None
    note: TaskNoteSummaryResponse | None = None
    projection: TaskProjectionResponse | None = None


class TaskReconciliationSummaryResponse(BaseModel):
    status: Literal["clean", "warnings", "incomplete"]
    note_id: str | None = None
    note_version: int | None = None
    parsed_count: int | None = None
    created_count: int = 0
    updated_count: int = 0
    unlinked_count: int = 0
    ambiguous_count: int = 0
    warning_count: int = 0
    processed_notes: int = 0
    remaining_stale_notes: int = 0


class TaskListResponse(BaseModel):
    tasks: list[TaskResponse]
    reconciliation: TaskReconciliationSummaryResponse


class TaskCreateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    status: TaskStatusValue = "open"
    metadata: TaskMetadata = Field(default_factory=TaskMetadata)
    expected_note_version: int = Field(..., ge=1)

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Task text cannot be empty.")
        if "\n" in value or "\r" in value:
            raise ValueError("Task text cannot contain newline characters.")
        return stripped


class TaskStatusUpdateItem(BaseModel):
    task_id: str = Field(..., min_length=1, max_length=128)
    status: TaskStatusValue
    expected_task_version: int = Field(..., ge=1)
    expected_note_version: int | None = Field(None, ge=1)
    record_only: bool = False


class TaskStatusBatchRequest(BaseModel):
    updates: list[TaskStatusUpdateItem] = Field(..., min_length=1, max_length=50)


class TaskStatusBatchResponse(BaseModel):
    tasks: list[TaskResponse]


class TaskUpdateRequest(BaseModel):
    text: str | None = Field(None, min_length=1, max_length=2000)
    metadata: TaskMetadata | None = None
    expected_task_version: int = Field(..., ge=1)
    expected_note_version: int | None = Field(None, ge=1)
    record_only: bool = False

    @field_validator("text")
    @classmethod
    def validate_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return value
        stripped = value.strip()
        if not stripped:
            raise ValueError("Task text cannot be empty.")
        if "\n" in value or "\r" in value:
            raise ValueError("Task text cannot contain newline characters.")
        return stripped

    @model_validator(mode="after")
    def require_mutation(self) -> TaskUpdateRequest:
        if self.text is None and self.metadata is None:
            raise ValueError("At least one task field must be provided.")
        return self


class TaskDeleteRequest(BaseModel):
    expected_task_version: int = Field(..., ge=1)
    expected_note_version: int | None = Field(None, ge=1)
    record_only: bool = False


class TaskRestoreRequest(BaseModel):
    """Exact task and Sync tombstone claims required for restore."""

    model_config = ConfigDict(extra="forbid")

    expected_task_version: int = Field(..., ge=1)
    expected_note_version: int = Field(..., ge=1)
    expected_base_server_cursor: int = Field(..., ge=1)
    expected_base_revision: int = Field(..., ge=1)
    expected_base_hash: str = Field(..., pattern=r"^sha256:[0-9a-f]{64}$")


class TaskRelinkRequest(BaseModel):
    """Authorized immutable parent and product versions required for relink."""

    model_config = ConfigDict(extra="forbid")

    note_id: str = Field(..., min_length=1, max_length=128)
    expected_task_version: int = Field(..., ge=1)
    expected_note_version: int = Field(..., ge=1)


class TaskActivityResponse(BaseModel):
    id: str
    task_id: str | None = None
    note_id: str | None = None
    event_type: str
    actor_type: str
    actor_id: str | None = None
    tool_name: str | None = None
    policy_mode: str | None = None
    approval_id: str | None = None
    old_value: dict[str, Any] | None = None
    new_value: dict[str, Any] | None = None
    created_at: str
    read_at: str | None = None
    dismissed_at: str | None = None


class TaskActivityListResponse(BaseModel):
    events: list[TaskActivityResponse]


class TaskActivityPatchRequest(BaseModel):
    read: bool = False
    dismissed: bool = False

    @model_validator(mode="after")
    def require_state_change(self) -> TaskActivityPatchRequest:
        if not self.read and not self.dismissed:
            raise ValueError("Either read or dismissed must be true.")
        return self


class TaskActivityStateResponse(BaseModel):
    event_id: str
    user_id: str
    read_at: str | None = None
    dismissed_at: str | None = None


class TaskProjectionDriftResponse(BaseModel):
    """Privacy-safe projection drift claims and lifecycle state."""

    id: str
    note_id: str
    task_id: str
    marker_base_revision: int
    marker_base_hash: str
    note_head_cursor: int | None = None
    note_head_hash: str | None = None
    task_head_cursor: int | None = None
    task_head_hash: str | None = None
    reason_code: str
    status: Literal["open", "resolved", "dismissed"]
    lifecycle_revision: int
    created_at: str
    updated_at: str
    resolved_at: str | None = None


class TaskProjectionDriftListResponse(BaseModel):
    drifts: list[TaskProjectionDriftResponse]


class TaskProjectionDriftResolveRequest(BaseModel):
    """Exact current claims required to resolve one open drift."""

    model_config = ConfigDict(extra="forbid")

    note_id: str = Field(..., min_length=1, max_length=128)
    action: TaskProjectionDriftAction
    expected_lifecycle_revision: int = Field(..., ge=1)
    expected_note_head_cursor: int | None = Field(None, ge=1)
    expected_note_head_hash: str | None = Field(
        None,
        pattern=r"^sha256:[0-9a-f]{64}$",
    )
    expected_task_head_cursor: int | None = Field(None, ge=1)
    expected_task_head_hash: str | None = Field(
        None,
        pattern=r"^sha256:[0-9a-f]{64}$",
    )

    @model_validator(mode="after")
    def validate_head_claim_pairs(self) -> TaskProjectionDriftResolveRequest:
        for cursor, object_hash in (
            (self.expected_note_head_cursor, self.expected_note_head_hash),
            (self.expected_task_head_cursor, self.expected_task_head_hash),
        ):
            if (cursor is None) != (object_hash is None):
                raise ValueError("Projection drift cursor and hash claims must be paired.")
        return self
