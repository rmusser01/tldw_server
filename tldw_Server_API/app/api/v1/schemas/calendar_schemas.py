from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tldw_Server_API.app.api.v1.schemas.reminders_schemas import ReminderTaskCreateRequest
from tldw_Server_API.app.api.v1.schemas.scheduled_tasks_control_plane_schemas import ScheduledTask
from tldw_Server_API.app.core.Calendar.constants import (
    CALENDAR_SOURCE_OWNER_LINKED_PROJECTION,
)
from tldw_Server_API.app.core.Calendar.errors import CalendarValidationError
from tldw_Server_API.app.core.Calendar.recurrence import LocalRecurrenceRule
from tldw_Server_API.app.core.DB_Management.Calendar_DB import (
    CalendarAnnotationRow,
    CalendarItemRow,
    CalendarLinkRow,
    CalendarMembershipRow,
    CalendarRecurrenceRow,
    CalendarRow,
    ExternalCalendarAccountRow,
    ExternalCalendarBindingRow,
)

CalendarItemKind = Literal["event", "todo"]
CalendarRole = Literal["owner", "editor", "commenter", "viewer"]
CalendarPrincipalType = Literal["user", "org_role"]
CalendarSourceOwner = Literal["tldw", "provider", "linked_projection"]


def _json_value(raw: str | None, default: Any) -> Any:
    if raw is None:
        return default
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return default


class CalendarCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., min_length=1, max_length=200)
    description: str | None = None
    color: str | None = Field(default=None, max_length=64)
    timezone: str = Field(default="UTC", min_length=1, max_length=128)
    org_id: int | None = None
    visibility: str = Field(default="private", min_length=1, max_length=64)
    default_reminder_policy: dict[str, Any] | None = None
    rbac_policy_ref: str | None = None


class CalendarResponse(BaseModel):
    id: int
    tenant_id: str
    owner_user_id: int
    org_id: int | None = None
    name: str
    description: str | None = None
    color: str | None = None
    timezone: str
    visibility: str
    default_reminder_policy: dict[str, Any] | None = None
    rbac_policy_ref: str | None = None
    archived_at: str | None = None
    created_at: str
    updated_at: str

    @classmethod
    def from_row(cls, row: CalendarRow) -> "CalendarResponse":
        return cls(
            id=row.id,
            tenant_id=row.tenant_id,
            owner_user_id=row.owner_user_id,
            org_id=row.org_id,
            name=row.name,
            description=row.description,
            color=row.color,
            timezone=row.timezone,
            visibility=row.visibility,
            default_reminder_policy=_json_value(row.default_reminder_policy_json, None),
            rbac_policy_ref=row.rbac_policy_ref,
            archived_at=row.archived_at,
            created_at=row.created_at,
            updated_at=row.updated_at,
        )


class CalendarListResponse(BaseModel):
    items: list[CalendarResponse] = Field(default_factory=list)
    total: int = Field(..., ge=0)


class CalendarMembershipCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    principal_type: CalendarPrincipalType
    principal_id: str = Field(..., min_length=1, max_length=200)
    role: CalendarRole


class CalendarMembershipResponse(BaseModel):
    id: int
    calendar_id: int
    principal_type: str
    principal_id: str
    role: str
    created_at: str
    updated_at: str

    @classmethod
    def from_row(cls, row: CalendarMembershipRow) -> "CalendarMembershipResponse":
        return cls(**row.__dict__)


class CalendarMembershipListResponse(BaseModel):
    items: list[CalendarMembershipResponse] = Field(default_factory=list)
    total: int = Field(..., ge=0)


class CalendarMembershipDeleteResponse(BaseModel):
    removed: int = Field(..., ge=0)


class CalendarRecurrenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rrule: str | None = None
    rdate: list[str] | None = None
    exdate: list[str] | None = None
    timezone: str | None = None

    @model_validator(mode="after")
    def _validate_rrule(self) -> "CalendarRecurrenceRequest":
        if self.rrule:
            try:
                LocalRecurrenceRule.from_rrule(self.rrule)
            except CalendarValidationError as exc:
                raise ValueError(str(exc)) from exc
        return self


class CalendarRecurrenceResponse(BaseModel):
    id: int
    calendar_item_id: int
    rrule: str | None = None
    rdate: list[str] | None = None
    exdate: list[str] | None = None
    timezone: str | None = None
    created_at: str
    updated_at: str

    @classmethod
    def from_row(cls, row: CalendarRecurrenceRow) -> "CalendarRecurrenceResponse":
        return cls(
            id=row.id,
            calendar_item_id=row.calendar_item_id,
            rrule=row.rrule,
            rdate=_json_value(row.rdate_json, None),
            exdate=_json_value(row.exdate_json, None),
            timezone=row.timezone,
            created_at=row.created_at,
            updated_at=row.updated_at,
        )


class CalendarItemCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    calendar_id: int
    kind: CalendarItemKind
    title: str = Field(..., min_length=1, max_length=500)
    description: str | None = None
    location: str | None = None
    start_at: str | None = None
    end_at: str | None = None
    due_at: str | None = None
    timezone: str | None = None
    all_day: bool = False
    status: str = Field(default="confirmed", min_length=1, max_length=64)
    local_tags: list[str] | None = None
    metadata: dict[str, Any] | None = None
    recurrence: CalendarRecurrenceRequest | None = None

    @model_validator(mode="after")
    def _validate_item_time(self) -> "CalendarItemCreateRequest":
        if self.kind == "event" and not self.start_at:
            raise ValueError("Calendar events require start_at")
        if self.kind == "todo" and not (self.start_at or self.due_at):
            raise ValueError("Calendar todos require due_at or start_at")
        return self


class CalendarItemUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: CalendarItemKind | None = None
    title: str | None = Field(default=None, min_length=1, max_length=500)
    description: str | None = None
    location: str | None = None
    start_at: str | None = None
    end_at: str | None = None
    due_at: str | None = None
    timezone: str | None = None
    all_day: bool | None = None
    status: str | None = Field(default=None, min_length=1, max_length=64)
    local_tags: list[str] | None = None
    metadata: dict[str, Any] | None = None
    recurrence: CalendarRecurrenceRequest | None = None

    def service_updates(self) -> dict[str, Any]:
        updates = self.model_dump(exclude_unset=True, exclude={"recurrence"})
        if "local_tags" in updates:
            updates["local_tags_json"] = updates.pop("local_tags")
        if "metadata" in updates:
            updates["metadata_json"] = updates.pop("metadata")
        return updates


class CalendarItemResponse(BaseModel):
    id: int
    calendar_id: int
    kind: str
    source_owner: CalendarSourceOwner | str
    provider_owned: bool
    title: str
    description: str | None = None
    location: str | None = None
    start_at: str | None = None
    end_at: str | None = None
    due_at: str | None = None
    timezone: str | None = None
    all_day: bool
    status: str
    local_tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    external_binding_id: int | None = None
    source_uid: str | None = None
    source_etag: str | None = None
    source_ctag: str | None = None
    source_updated_at: str | None = None
    copied_from_item_id: int | None = None
    linked_projection_type: str | None = None
    linked_projection_id: str | None = None
    deleted_at: str | None = None
    remote_deleted_at: str | None = None
    created_at: str
    updated_at: str
    recurrence: CalendarRecurrenceResponse | None = None

    @classmethod
    def from_row(
        cls,
        row: CalendarItemRow,
        recurrence: CalendarRecurrenceRow | None = None,
    ) -> "CalendarItemResponse":
        return cls(
            id=row.id,
            calendar_id=row.calendar_id,
            kind=row.kind,
            source_owner=row.source_owner,
            provider_owned=row.provider_owned,
            title=row.title,
            description=row.description,
            location=row.location,
            start_at=row.start_at,
            end_at=row.end_at,
            due_at=row.due_at,
            timezone=row.timezone,
            all_day=row.all_day,
            status=row.status,
            local_tags=_json_value(row.local_tags_json, []),
            metadata=_json_value(row.metadata_json, {}),
            external_binding_id=row.external_binding_id,
            source_uid=row.source_uid,
            source_etag=row.source_etag,
            source_ctag=row.source_ctag,
            source_updated_at=row.source_updated_at,
            copied_from_item_id=row.copied_from_item_id,
            linked_projection_type=row.linked_projection_type,
            linked_projection_id=row.linked_projection_id,
            deleted_at=row.deleted_at,
            remote_deleted_at=row.remote_deleted_at,
            created_at=row.created_at,
            updated_at=row.updated_at,
            recurrence=CalendarRecurrenceResponse.from_row(recurrence) if recurrence else None,
        )


class CalendarAnnotationCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    body: str = Field(..., min_length=1)
    tags: list[str] | None = None


class CalendarLocalTagsUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tags: list[str] = Field(default_factory=list)


class CalendarAnnotationResponse(BaseModel):
    id: int
    calendar_item_id: int
    author_user_id: int
    body: str
    tags: list[str] = Field(default_factory=list)
    deleted_at: str | None = None
    created_at: str
    updated_at: str

    @classmethod
    def from_row(cls, row: CalendarAnnotationRow) -> "CalendarAnnotationResponse":
        return cls(
            id=row.id,
            calendar_item_id=row.calendar_item_id,
            author_user_id=row.author_user_id,
            body=row.body,
            tags=_json_value(row.tags_json, []),
            deleted_at=row.deleted_at,
            created_at=row.created_at,
            updated_at=row.updated_at,
        )


class CalendarLinkCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_type: str = Field(..., min_length=1, max_length=100)
    target_id: str = Field(..., min_length=1, max_length=300)
    label: str | None = None
    url: str | None = None
    metadata: dict[str, Any] | None = None


class CalendarLinkResponse(BaseModel):
    id: int
    calendar_item_id: int
    target_type: str
    target_id: str
    label: str | None = None
    url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str
    updated_at: str

    @classmethod
    def from_row(cls, row: CalendarLinkRow) -> "CalendarLinkResponse":
        return cls(
            id=row.id,
            calendar_item_id=row.calendar_item_id,
            target_type=row.target_type,
            target_id=row.target_id,
            label=row.label,
            url=row.url,
            metadata=_json_value(row.metadata_json, {}),
            created_at=row.created_at,
            updated_at=row.updated_at,
        )


class CalendarItemCopyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_calendar_id: int | None = None
    title: str | None = Field(default=None, min_length=1, max_length=500)


class CalendarViewLinkResponse(BaseModel):
    target_type: str
    target_id: str
    label: str | None = None
    url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CalendarViewItemResponse(BaseModel):
    id: str
    title: str
    kind: CalendarItemKind | str
    source_owner: str
    start_at: str | None = None
    end_at: str | None = None
    due_at: str | None = None
    calendar_id: int | None = None
    calendar_item_id: int | None = None
    description: str | None = None
    location: str | None = None
    all_day: bool = False
    status: str | None = None
    local_tags: list[str] = Field(default_factory=list)
    read_only_reason: str | None = None
    recurrence_id: int | None = None
    occurrence_index: int | None = None
    link: CalendarViewLinkResponse | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CalendarViewResponse(BaseModel):
    start_at: str
    end_at: str
    items: list[CalendarViewItemResponse] = Field(default_factory=list)
    partial: bool = False
    warnings: list[str] = Field(default_factory=list)


class CalendarReminderCreateRequest(ReminderTaskCreateRequest):
    calendar_item_id: int


class CalendarReminderProjectionResponse(BaseModel):
    source_owner: Literal["linked_projection"] = CALENDAR_SOURCE_OWNER_LINKED_PROJECTION
    link_type: Literal["calendar_item"] = "calendar_item"
    link_id: str
    next_run_at: str | None = None


class CalendarReminderResponse(BaseModel):
    calendar_item_id: int
    scheduled_task: ScheduledTask
    projection: CalendarReminderProjectionResponse


class ExternalCalendarAccountCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str = Field(..., min_length=1, max_length=100)
    display_name: str = Field(..., min_length=1, max_length=200)
    secret_ref: str | None = None
    account_metadata: dict[str, Any] | None = None


class ExternalCalendarAccountResponse(BaseModel):
    id: int
    tenant_id: str
    user_id: int
    provider: str
    display_name: str
    secret_ref: str | None = None
    account_metadata: dict[str, Any] | None = None
    status: str
    revoked_at: str | None = None
    deleted_at: str | None = None
    created_at: str
    updated_at: str

    @classmethod
    def from_row(cls, row: ExternalCalendarAccountRow) -> "ExternalCalendarAccountResponse":
        return cls(
            id=row.id,
            tenant_id=row.tenant_id,
            user_id=row.user_id,
            provider=row.provider,
            display_name=row.display_name,
            secret_ref=row.secret_ref,
            account_metadata=_json_value(row.account_metadata_json, None),
            status=row.status,
            revoked_at=row.revoked_at,
            deleted_at=row.deleted_at,
            created_at=row.created_at,
            updated_at=row.updated_at,
        )


class ExternalCalendarAccountListResponse(BaseModel):
    items: list[ExternalCalendarAccountResponse] = Field(default_factory=list)
    total: int = Field(..., ge=0)


class ExternalCalendarBindingCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    account_id: int
    calendar_id: int
    remote_calendar_id: str = Field(..., min_length=1, max_length=500)
    remote_display_name: str | None = None
    sync_enabled: bool = True
    sync_interval_minutes: int | None = Field(default=None, ge=1)
    lookback_days: int = Field(default=90, ge=0, le=3700)
    lookahead_days: int = Field(default=365, ge=0, le=3700)
    provider_capabilities: dict[str, Any] | None = None


class ExternalCalendarBindingResponse(BaseModel):
    id: int
    account_id: int
    calendar_id: int
    remote_calendar_id: str
    remote_display_name: str | None = None
    sync_enabled: bool
    sync_interval_minutes: int | None = None
    lookback_days: int
    lookahead_days: int
    provider_capabilities: dict[str, Any] | None = None
    sync_cursor: str | None = None
    last_sync_at: str | None = None
    next_scan_at: str | None = None
    last_error: str | None = None
    disabled_at: str | None = None
    deleted_at: str | None = None
    created_at: str
    updated_at: str

    @classmethod
    def from_row(cls, row: ExternalCalendarBindingRow) -> "ExternalCalendarBindingResponse":
        return cls(
            id=row.id,
            account_id=row.account_id,
            calendar_id=row.calendar_id,
            remote_calendar_id=row.remote_calendar_id,
            remote_display_name=row.remote_display_name,
            sync_enabled=row.sync_enabled,
            sync_interval_minutes=row.sync_interval_minutes,
            lookback_days=row.lookback_days,
            lookahead_days=row.lookahead_days,
            provider_capabilities=_json_value(row.provider_capabilities_json, None),
            sync_cursor=row.sync_cursor,
            last_sync_at=row.last_sync_at,
            next_scan_at=row.next_scan_at,
            last_error=row.last_error,
            disabled_at=row.disabled_at,
            deleted_at=row.deleted_at,
            created_at=row.created_at,
            updated_at=row.updated_at,
        )


class ExternalCalendarBindingListResponse(BaseModel):
    items: list[ExternalCalendarBindingResponse] = Field(default_factory=list)
    total: int = Field(..., ge=0)


class CalendarSyncTriggerResponse(BaseModel):
    binding_id: int
    queued: bool = False
    status: Literal["not_implemented"] = "not_implemented"
