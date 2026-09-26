"""Calendar API payloads, recurrence validation, and credential-free row responses."""

from __future__ import annotations

import json
from typing import Any, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dateutil import parser as date_parser
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
    CalendarSyncEventRow,
    ExternalCalendarAccountRow,
    ExternalCalendarBindingRow,
)

CalendarItemKind = Literal["event", "todo"]
CalendarRole = Literal["owner", "editor", "commenter", "viewer"]
CalendarPrincipalType = Literal["user", "org_role"]
CalendarSourceOwner = Literal["tldw", "provider", "linked_projection"]


def _json_value(raw: str | None, default: Any) -> Any:
    """Decode persisted JSON, returning the caller's fallback for null or invalid JSON."""
    if raw is None:
        return default
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return default


class CalendarCreateRequest(BaseModel):
    """Calendar creation fields with bounded labels and no unknown properties."""

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
    """Persisted calendar identity, ownership, display settings, and archive state."""

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
    def from_row(cls, row: CalendarRow) -> CalendarResponse:
        """Build a response, treating absent or malformed reminder-policy JSON as null."""
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
    """Calendar collection with a nonnegative total independent of the returned page."""

    items: list[CalendarResponse] = Field(default_factory=list)
    total: int = Field(..., ge=0)


class CalendarMembershipCreateRequest(BaseModel):
    """Grant a supported calendar role to a user or organization-role principal."""

    model_config = ConfigDict(extra="forbid")

    principal_type: CalendarPrincipalType
    principal_id: str = Field(..., min_length=1, max_length=200)
    role: CalendarRole


class CalendarMembershipResponse(BaseModel):
    """Persisted role assignment for one principal within a calendar."""

    id: int
    calendar_id: int
    principal_type: str
    principal_id: str
    role: str
    created_at: str
    updated_at: str

    @classmethod
    def from_row(cls, row: CalendarMembershipRow) -> CalendarMembershipResponse:
        """Validate and expose the membership row's identity, role, and timestamps."""
        return cls(**row.__dict__)


class CalendarMembershipListResponse(BaseModel):
    """Membership collection with a nonnegative assignment count."""

    items: list[CalendarMembershipResponse] = Field(default_factory=list)
    total: int = Field(..., ge=0)


class CalendarMembershipDeleteResponse(BaseModel):
    """Nonnegative count of memberships removed by a delete operation."""

    removed: int = Field(..., ge=0)


class CalendarRecurrenceRequest(BaseModel):
    """Local recurrence rule and ISO inclusion/exclusion dates with an optional IANA zone."""

    model_config = ConfigDict(extra="forbid")

    rrule: str | None = None
    rdate: list[str] | None = None
    exdate: list[str] | None = None
    timezone: str | None = None

    @model_validator(mode="after")
    def _validate_rrule(self) -> CalendarRecurrenceRequest:
        """Return this request or raise ValueError for unsupported rules, dates, or zones."""
        if self.rrule:
            try:
                LocalRecurrenceRule.from_rrule(self.rrule)
            except CalendarValidationError as exc:
                raise ValueError(str(exc)) from exc
        try:
            if self.timezone:
                ZoneInfo(self.timezone)
            for value in [*(self.rdate or []), *(self.exdate or [])]:
                date_parser.isoparse(value)
        except (ValueError, ZoneInfoNotFoundError) as exc:
            raise ValueError(
                "Recurrence dates and timezone must be valid ISO dates/timestamps and IANA timezone"
            ) from exc
        return self


class CalendarRecurrenceResponse(BaseModel):
    """Persisted recurrence definition with decoded date lists for a calendar item."""

    id: int
    calendar_item_id: int
    rrule: str | None = None
    rdate: list[str] | None = None
    exdate: list[str] | None = None
    timezone: str | None = None
    created_at: str
    updated_at: str

    @classmethod
    def from_row(cls, row: CalendarRecurrenceRow) -> CalendarRecurrenceResponse:
        """Build recurrence output, mapping missing or malformed date-list JSON to null."""
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
    """New event or todo fields with kind-specific minimum scheduling requirements."""

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
    def _validate_item_time(self) -> CalendarItemCreateRequest:
        """Return this request; raise ValueError when the item kind lacks a required time."""
        if self.kind == "event" and not self.start_at:
            raise ValueError("Calendar events require start_at")
        if self.kind == "todo" and not (self.start_at or self.due_at):
            raise ValueError("Calendar todos require due_at or start_at")
        return self


class CalendarItemUpdateRequest(BaseModel):
    """Partial item mutation preserving the distinction between omitted and null fields."""

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
        """Return supplied scalar updates with JSON storage keys; handle recurrence separately."""
        updates = self.model_dump(exclude_unset=True, exclude={"recurrence"})
        if "local_tags" in updates:
            updates["local_tags_json"] = updates.pop("local_tags")
        if "metadata" in updates:
            updates["metadata_json"] = updates.pop("metadata")
        return updates


class CalendarItemDeleteResponse(BaseModel):
    """Whether a calendar item was soft-deleted."""

    deleted: bool


class CalendarItemResponse(BaseModel):
    """Item content and provenance with decoded local metadata and optional recurrence."""

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
    ) -> CalendarItemResponse:
        """Build item output with safe JSON fallbacks and an optional recurrence response."""
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
    """Nonempty annotation text and optional local tags; unknown fields are rejected."""

    model_config = ConfigDict(extra="forbid")

    body: str = Field(..., min_length=1)
    tags: list[str] | None = None


class CalendarLocalTagsUpdateRequest(BaseModel):
    """Replace the actor's local tag overlay, allowing an empty list to clear tags."""

    model_config = ConfigDict(extra="forbid")

    tags: list[str] = Field(default_factory=list)


class CalendarAnnotationResponse(BaseModel):
    """Authored item annotation with decoded tags and soft-deletion state."""

    id: int
    calendar_item_id: int
    author_user_id: int
    body: str
    tags: list[str] = Field(default_factory=list)
    deleted_at: str | None = None
    created_at: str
    updated_at: str

    @classmethod
    def from_row(cls, row: CalendarAnnotationRow) -> CalendarAnnotationResponse:
        """Build annotation output, defaulting absent or malformed tags JSON to an empty list."""
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
    """Bounded target identity and optional display metadata for an item context link."""

    model_config = ConfigDict(extra="forbid")

    target_type: str = Field(..., min_length=1, max_length=100)
    target_id: str = Field(..., min_length=1, max_length=300)
    label: str | None = None
    url: str | None = None
    metadata: dict[str, Any] | None = None


class CalendarLinkResponse(BaseModel):
    """Persisted item context link with decoded target metadata."""

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
    def from_row(cls, row: CalendarLinkRow) -> CalendarLinkResponse:
        """Build link output, defaulting absent or malformed metadata JSON to an empty mapping."""
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


class CalendarLinkListResponse(BaseModel):
    """Context-link collection and nonnegative total."""

    items: list[CalendarLinkResponse] = Field(default_factory=list)
    total: int = Field(..., ge=0)


class CalendarLinkDeleteResponse(BaseModel):
    """Nonnegative count of context links physically removed."""

    removed: int = Field(..., ge=0)


class CalendarItemCopyRequest(BaseModel):
    """Optional destination calendar and bounded replacement title for a local item copy."""

    model_config = ConfigDict(extra="forbid")

    target_calendar_id: int | None = None
    title: str | None = Field(default=None, min_length=1, max_length=500)


class CalendarViewLinkResponse(BaseModel):
    """Source-navigation target for a linked calendar projection."""

    target_type: str
    target_id: str
    label: str | None = None
    url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CalendarViewItemResponse(BaseModel):
    """Expanded occurrence or linked projection with stable identity and read-only context."""

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
    links: list[CalendarLinkResponse] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CalendarViewResponse(BaseModel):
    """Windowed agenda results with partial-result state and expansion warnings."""

    start_at: str
    end_at: str
    items: list[CalendarViewItemResponse] = Field(default_factory=list)
    partial: bool = False
    warnings: list[str] = Field(default_factory=list)


class CalendarReminderCreateRequest(ReminderTaskCreateRequest):
    """Reminder-task creation fields anchored to a persisted calendar item."""

    calendar_item_id: int


class CalendarReminderProjectionResponse(BaseModel):
    """Fixed calendar-item linkage and next execution time for a reminder projection."""

    source_owner: Literal["linked_projection"] = CALENDAR_SOURCE_OWNER_LINKED_PROJECTION
    link_type: Literal["calendar_item"] = "calendar_item"
    link_id: str
    next_run_at: str | None = None


class CalendarReminderResponse(BaseModel):
    """Created scheduled reminder and its calendar projection for the source item."""

    calendar_item_id: int
    scheduled_task: ScheduledTask
    projection: CalendarReminderProjectionResponse


class ExternalCalendarAccountCreateRequest(BaseModel):
    """Provider account setup with bounded credentials or an existing opaque secret reference."""

    model_config = ConfigDict(extra="forbid")

    provider: str = Field(..., min_length=1, max_length=100)
    display_name: str = Field(..., min_length=1, max_length=200)
    server_url: str | None = Field(default=None, max_length=2048)
    username: str | None = Field(default=None, max_length=500)
    password: str | None = Field(default=None, max_length=2000)
    token: str | None = Field(default=None, max_length=4000)
    secret_ref: str | None = None
    account_metadata: dict[str, Any] | None = None


class CalDavAccountVerifyRequest(BaseModel):
    """Optional bounded credential overrides for verifying an existing CalDAV account."""

    model_config = ConfigDict(extra="forbid")

    server_url: str | None = Field(default=None, max_length=2048)
    username: str | None = Field(default=None, max_length=500)
    password: str | None = Field(default=None, max_length=2000)
    token: str | None = Field(default=None, max_length=4000)


class CalDavAccountVerifyResponse(BaseModel):
    """Verification outcome and optional provider error for a specific account."""

    account_id: int
    verified: bool
    status: str | None = None
    error: str | None = None


class CalDavAccountMutationResponse(BaseModel):
    """Revocation or deletion outcome without returning account credentials."""

    revoked: bool | None = None
    deleted: bool | None = None


class ExternalCalendarAccountResponse(BaseModel):
    """Public account state excluding both credential payloads and secret references."""

    id: int
    tenant_id: str
    user_id: int
    provider: str
    display_name: str
    account_metadata: dict[str, Any] | None = None
    status: str
    revoked_at: str | None = None
    deleted_at: str | None = None
    created_at: str
    updated_at: str

    @classmethod
    def from_row(cls, row: ExternalCalendarAccountRow) -> ExternalCalendarAccountResponse:
        """Build credential-free account output with nullable decoded account metadata."""
        return cls(
            id=row.id,
            tenant_id=row.tenant_id,
            user_id=row.user_id,
            provider=row.provider,
            display_name=row.display_name,
            account_metadata=_json_value(row.account_metadata_json, None),
            status=row.status,
            revoked_at=row.revoked_at,
            deleted_at=row.deleted_at,
            created_at=row.created_at,
            updated_at=row.updated_at,
        )


class ExternalCalendarAccountListResponse(BaseModel):
    """Credential-free account collection with a nonnegative total."""

    items: list[ExternalCalendarAccountResponse] = Field(default_factory=list)
    total: int = Field(..., ge=0)


class ExternalCalendarDiscoveryItem(BaseModel):
    """Discovered remote calendar identity, display name, and provider capabilities."""

    remote_calendar_id: str
    remote_display_name: str | None = None
    provider_capabilities: dict[str, Any] | None = None


class ExternalCalendarDiscoveryResponse(BaseModel):
    """Remote calendars discovered for an external account."""

    items: list[ExternalCalendarDiscoveryItem] = Field(default_factory=list)


class ExternalCalendarBindingCreateRequest(BaseModel):
    """Bind a remote calendar with a positive sync interval and bounded scan windows."""

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


class ExternalCalendarBindingUpdateRequest(BaseModel):
    """Partial sync-policy update with the same interval and scan-window bounds as creation."""

    model_config = ConfigDict(extra="forbid")

    sync_enabled: bool | None = None
    sync_interval_minutes: int | None = Field(default=None, ge=1)
    lookback_days: int | None = Field(default=None, ge=0, le=3700)
    lookahead_days: int | None = Field(default=None, ge=0, le=3700)
    provider_capabilities: dict[str, Any] | None = None

    def service_updates(self) -> dict[str, Any]:
        """Return only supplied binding updates, translating capabilities to the JSON storage key."""
        updates = self.model_dump(exclude_unset=True)
        if "provider_capabilities" in updates:
            updates["provider_capabilities_json"] = updates.pop("provider_capabilities")
        return updates


class ExternalCalendarBindingResponse(BaseModel):
    """Remote binding identity, sync policy, progress, errors, and lifecycle timestamps."""

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
    def from_row(cls, row: ExternalCalendarBindingRow) -> ExternalCalendarBindingResponse:
        """Build binding output, mapping absent or malformed capabilities JSON to null."""
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
    """External binding collection with a nonnegative total."""

    items: list[ExternalCalendarBindingResponse] = Field(default_factory=list)
    total: int = Field(..., ge=0)


class CalendarSyncEventResponse(BaseModel):
    """Recorded sync outcome, item counters, error details, and decoded diagnostic metadata."""

    id: int
    binding_id: int | None = None
    account_id: int | None = None
    event_type: str
    status: str
    started_at: str | None = None
    finished_at: str | None = None
    items_seen: int
    items_upserted: int
    items_tombstoned: int
    error_message: str | None = None
    metadata: dict[str, Any] | None = None
    created_at: str

    @classmethod
    def from_row(cls, row: CalendarSyncEventRow) -> CalendarSyncEventResponse:
        """Build sync-event output, mapping missing or malformed diagnostic JSON to null."""
        return cls(
            id=row.id,
            binding_id=row.binding_id,
            account_id=row.account_id,
            event_type=row.event_type,
            status=row.status,
            started_at=row.started_at,
            finished_at=row.finished_at,
            items_seen=row.items_seen,
            items_upserted=row.items_upserted,
            items_tombstoned=row.items_tombstoned,
            error_message=row.error_message,
            metadata=_json_value(row.metadata_json, None),
            created_at=row.created_at,
        )


class CalendarSyncEventListResponse(BaseModel):
    """Sync-event history page with a nonnegative total."""

    items: list[CalendarSyncEventResponse] = Field(default_factory=list)
    total: int = Field(..., ge=0)


class CalendarSyncTriggerRequest(BaseModel):
    """Manual sync reason and optional nonempty window-boundary overrides."""

    model_config = ConfigDict(extra="forbid")

    reason: str = Field(default="manual", min_length=1, max_length=64)
    window_start: str | None = Field(default=None, min_length=1)
    window_end: str | None = Field(default=None, min_length=1)


class CalendarSyncTriggerResponse(BaseModel):
    """Binding sync acceptance state with optional queued-job and deduplication identifiers."""

    binding_id: int
    queued: bool = False
    status: str
    job_id: int | None = None
    idempotency_key: str | None = None
