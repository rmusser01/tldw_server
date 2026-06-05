"""Schemas for note-backed task REST operations."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

TaskStatusValue = Literal["open", "done"]
ProjectionStatusValue = Literal["live", "unlinked", "ambiguous", "deleted"]


class TaskMetadata(BaseModel):
    due_date: str | None = Field(None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    priority: Literal["high", "medium", "low"] | None = None
    estimate: str | None = Field(None, pattern=r"^\d+[mhd]$")

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
        return stripped

    @model_validator(mode="after")
    def require_mutation(self) -> "TaskUpdateRequest":
        if self.text is None and self.metadata is None:
            raise ValueError("At least one task field must be provided.")
        return self


class TaskDeleteRequest(BaseModel):
    expected_task_version: int = Field(..., ge=1)
    expected_note_version: int | None = Field(None, ge=1)
    record_only: bool = False


class TaskActivityResponse(BaseModel):
    id: str
    task_id: str | None = None
    note_id: str | None = None
    event_type: str
    actor_type: str
    actor_id: str | None = None
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
    def require_state_change(self) -> "TaskActivityPatchRequest":
        if not self.read and not self.dismissed:
            raise ValueError("Either read or dismissed must be true.")
        return self


class TaskActivityStateResponse(BaseModel):
    event_id: str
    user_id: str
    read_at: str | None = None
    dismissed_at: str | None = None
