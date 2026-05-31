from __future__ import annotations

"""Pydantic request/response models for reminders tasks and notifications APIs."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from tldw_Server_API.app.api.v1.schemas.pagination import (
    OffsetPaginationMeta,
    default_offset_pagination_aliases,
    validate_offset_pagination_aliases,
)

ReminderScheduleKind = Literal["one_time", "recurring"]


class ReminderTaskCreateRequest(BaseModel):
    """Payload for creating a reminder task."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(..., min_length=1, max_length=200)
    body: str | None = None
    schedule_kind: ReminderScheduleKind
    run_at: str | None = None
    cron: str | None = None
    timezone: str | None = None
    link_type: str | None = None
    link_id: str | None = None
    link_url: str | None = None
    enabled: bool = True

    @model_validator(mode="after")
    def _validate_schedule_fields(self) -> "ReminderTaskCreateRequest":
        if self.schedule_kind == "one_time":
            if not self.run_at:
                raise ValueError("run_at is required for one_time schedules")
            return self

        if not self.cron:
            raise ValueError("cron is required for recurring schedules")
        if not self.timezone:
            raise ValueError("timezone is required for recurring schedules")
        return self


class ReminderTaskUpdateRequest(BaseModel):
    """Patch payload for mutable reminder task fields."""

    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(default=None, min_length=1, max_length=200)
    body: str | None = None
    schedule_kind: ReminderScheduleKind | None = None
    run_at: str | None = None
    cron: str | None = None
    timezone: str | None = None
    link_type: str | None = None
    link_id: str | None = None
    link_url: str | None = None
    enabled: bool | None = None


class ReminderTaskResponse(BaseModel):
    """Reminder task representation returned by API endpoints."""

    id: str
    user_id: str
    tenant_id: str
    title: str
    body: str | None = None
    link_type: str | None = None
    link_id: str | None = None
    link_url: str | None = None
    schedule_kind: ReminderScheduleKind
    run_at: str | None = None
    cron: str | None = None
    timezone: str | None = None
    enabled: bool
    last_run_at: str | None = None
    next_run_at: str | None = None
    last_status: str | None = None
    created_at: str
    updated_at: str


class ReminderTaskListResponse(BaseModel):
    """Paginated-style response for reminder task listing."""

    items: list[ReminderTaskResponse]
    total: int


class ReminderTaskDeleteResponse(BaseModel):
    """Delete outcome for a reminder task."""

    deleted: bool


NotificationKind = Literal["reminder_due", "reminder_failed", "job_completed", "job_failed", "companion_reflection"]


class NotificationResponse(BaseModel):
    """Notification item returned in list and stream payloads."""

    id: int
    user_id: str
    kind: NotificationKind
    title: str
    message: str
    severity: str
    source_task_id: str | None = None
    source_task_run_id: int | None = None
    source_job_id: str | None = None
    source_domain: str | None = None
    source_job_type: str | None = None
    link_type: str | None = None
    link_id: str | None = None
    link_url: str | None = None
    dedupe_key: str | None = None
    retention_until: str | None = None
    archived_at: str | None = None
    created_at: str
    read_at: str | None = None
    dismissed_at: str | None = None
    snooze_until: str | None = None


class NotificationsListResponse(BaseModel):
    """List response for user notifications."""

    items: list[NotificationResponse]
    total: int = Field(..., ge=0)
    limit: int = Field(..., ge=1)
    offset: int = Field(..., ge=0)
    pagination: OffsetPaginationMeta
    has_more: bool | None = None
    next_offset: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _default_pagination_aliases(self) -> "NotificationsListResponse":
        return default_offset_pagination_aliases(self)

    @model_validator(mode="after")
    def _validate_pagination_aliases(self) -> "NotificationsListResponse":
        return validate_offset_pagination_aliases(self)


class NotificationsUnreadCountResponse(BaseModel):
    """Unread notification counter response."""

    unread_count: int


class NotificationsMarkReadRequest(BaseModel):
    """Request payload for marking notifications as read."""

    ids: list[int] = Field(default_factory=list, min_length=1)


class NotificationsMarkReadResponse(BaseModel):
    """Result payload for mark-read operations."""

    updated: int


class NotificationDismissResponse(BaseModel):
    """Result payload for dismiss operations."""

    dismissed: bool


class NotificationPreferencesResponse(BaseModel):
    """Current notification preference settings for a user."""

    user_id: str
    reminder_enabled: bool
    job_completed_enabled: bool
    job_failed_enabled: bool
    updated_at: str


class NotificationPreferencesUpdateRequest(BaseModel):
    """Patch payload for notification preference flags."""

    model_config = ConfigDict(extra="forbid")

    reminder_enabled: bool | None = None
    job_completed_enabled: bool | None = None
    job_failed_enabled: bool | None = None


class NotificationSnoozeRequest(BaseModel):
    """Request payload for creating a snoozed one-time reminder."""

    minutes: int = Field(default=30, ge=1, le=10080)


class NotificationSnoozeResponse(BaseModel):
    """Response payload for snooze task creation."""

    task_id: str
    run_at: str


class NotificationCancelSnoozeResponse(BaseModel):
    """Response payload for snooze cancellation."""

    cancelled: bool
    deleted_tasks: int
