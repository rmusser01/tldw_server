from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator
from tldw_Server_API.app.api.v1.schemas.pagination import OffsetPaginationMeta


def _default_offset_pagination_aliases(response):
    if response.has_more is None:
        response.has_more = response.pagination.has_more
    if response.next_offset is None:
        response.next_offset = response.pagination.next_offset
    return response


class WatchlistRule(BaseModel):
    rule_id: str | None = Field(None, description="Optional stable rule identifier")
    pattern: str = Field(..., description="Literal or /regex/ pattern")
    category: str | None = Field(None, description="Logical category, e.g., 'adult', 'violence', 'self_harm'")
    severity: str | None = Field("info", description="info | warning | critical")
    note: str | None = Field(None, description="Free-text note for admins")
    tags: list[str] | None = Field(default=None, description="Optional tags")


class Watchlist(BaseModel):
    id: str | None = Field(None, description="Watchlist ID (UUID). If absent on create, a new one is generated")
    name: str
    description: str | None = None
    enabled: bool = True
    scope_type: str = Field("user", description="global | user | team | org")
    scope_id: str | None = Field(None, description="The corresponding subject id (null for global)")
    managed_by: str | None = Field(None, description="config | api")
    rules: list[WatchlistRule] = Field(default_factory=list)


class WatchlistListResponse(BaseModel):
    watchlists: list[Watchlist]


class WatchlistUpsertResponse(BaseModel):
    watchlist: Watchlist
    status: str = Field("ok")


class WatchlistDeleteResponse(BaseModel):
    status: str
    id: str


class WatchlistsReloadResponse(BaseModel):
    status: str = Field("ok", description="Status of the watchlist reload operation")


class AlertItem(BaseModel):
    id: int
    alert_identity: str | None = None
    created_at: str
    user_id: str | None = None
    scope_type: str | None = None
    scope_id: str | None = None
    source: str
    watchlist_id: str | None = None
    rule_id: str | None = None
    rule_category: str | None = None
    rule_severity: str | None = None
    pattern: str | None = None
    source_id: str | None = None
    chunk_id: str | None = None
    chunk_seq: int | None = None
    text_snippet: str | None = None
    metadata: dict[str, Any] | None = None
    is_read: bool = False
    read_at: str | None = None
    acknowledged_at: str | None = None
    dismissed_at: str | None = None
    assigned_to_user_id: int | None = None
    snoozed_until: str | None = None
    escalated_severity: str | None = None


class AlertsListResponse(BaseModel):
    items: list[AlertItem]
    total: int | None = None  # Optional future enhancement
    has_more: bool | None = Field(default=None, description="Alias for pagination.has_more")
    next_offset: int | None = Field(default=None, ge=0, description="Alias for pagination.next_offset")
    pagination: OffsetPaginationMeta

    @model_validator(mode="after")
    def _default_pagination_aliases(self):
        return _default_offset_pagination_aliases(self)


class MarkReadResponse(BaseModel):
    status: str = Field(
        description="Mutation succeeded; item contains the authoritative merged state after the mutation."
    )
    id: int = Field(description="Runtime alert row id for the mutated monitoring alert")
    item: AlertItem = Field(description="Authoritative merged alert state after the mutation")


class NotificationSettings(BaseModel):
    enabled: bool
    min_severity: str
    file: str
    webhook_url: str | None = None
    email_to: str | None = None
    # Optional SMTP fields exposed for completeness
    smtp_host: str | None = None
    smtp_port: int | None = None
    smtp_starttls: bool | None = None
    smtp_user: str | None = None
    email_from: str | None = None


class NotificationSettingsUpdate(BaseModel):
    enabled: bool | None = None
    min_severity: str | None = None
    file: str | None = None
    webhook_url: str | None = None
    email_to: str | None = None
    smtp_host: str | None = None
    smtp_port: int | None = None
    smtp_starttls: bool | None = None
    smtp_user: str | None = None
    smtp_password: str | None = None  # write-only
    email_from: str | None = None


class NotificationTestRequest(BaseModel):
    severity: str = Field("critical")
    message: str = Field("Test notification from admin panel")
    user_id: str | None = None


class NotificationTestResponse(BaseModel):
    status: str = Field("ok", description="Result of the test notification request")


class RecentNotificationsResponse(BaseModel):
    items: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Recent notification events from the JSONL log",
    )
