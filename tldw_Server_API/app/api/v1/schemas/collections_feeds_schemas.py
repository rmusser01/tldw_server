from __future__ import annotations

from typing import Any

from pydantic import BaseModel, HttpUrl, field_validator

from tldw_Server_API.app.api.v1.schemas._compat import Field
from tldw_Server_API.app.api.v1.schemas.pagination import PagePaginationMeta


class CollectionsFeedCreateRequest(BaseModel):
    url: HttpUrl = Field(example="https://example.com/feed.xml")
    name: str | None = Field(default=None, example="Example Feed")
    tags: list[str] = Field(default_factory=list, example=["news", "tech"])
    schedule_expr: str | None = Field(default=None, description="Cron expression for polling")
    timezone: str | None = Field(default=None, description="Timezone for schedule (IANA or UTC+/-)")
    active: bool = Field(default=True, description="Whether the feed job is active")
    settings: dict[str, Any] | None = Field(
        default=None,
        description="Optional watchlists source settings (rss/history prefs, limits)",
    )

    @field_validator("tags", mode="before")
    @classmethod
    def _strip_tags(cls, value: Any) -> Any:
        if not isinstance(value, list):
            return value
        return [item.strip() if isinstance(item, str) else item for item in value]


class CollectionsFeedUpdateRequest(BaseModel):
    name: str | None = Field(default=None, example="Example Feed")
    url: HttpUrl | None = Field(default=None, example="https://example.com/feed.xml")
    tags: list[str] | None = Field(default=None, example=["news", "tech"])
    schedule_expr: str | None = Field(default=None, description="Cron expression for polling")
    timezone: str | None = Field(default=None, description="Timezone for schedule (IANA or UTC+/-)")
    active: bool | None = Field(default=None, description="Whether the feed job is active")
    settings: dict[str, Any] | None = Field(
        default=None,
        description="Optional watchlists source settings (rss/history prefs, limits)",
    )

    @field_validator("tags", mode="before")
    @classmethod
    def _strip_tags(cls, value: Any) -> Any:
        if not isinstance(value, list):
            return value
        return [item.strip() if isinstance(item, str) else item for item in value]


class CollectionsFeed(BaseModel):
    id: int
    name: str
    url: str
    source_type: str = Field(example="rss")
    origin: str = Field(example="feed")
    tags: list[str] = Field(default_factory=list)
    active: bool
    settings: dict[str, Any] | None = None
    last_scraped_at: str | None = None
    etag: str | None = None
    last_modified: str | None = None
    defer_until: str | None = None
    status: str | None = None
    consec_not_modified: int | None = None
    consec_errors: int | None = None
    health_status: str | None = Field(default=None, description="Derived health: healthy, degraded, failing, disabled")
    promoted_at: str | None = Field(default=None, description="When schedule was auto-promoted from hourly to daily")
    created_at: str | None = None
    updated_at: str | None = None
    job_id: int | None = None
    schedule_expr: str | None = None
    timezone: str | None = None
    job_active: bool | None = None
    next_run_at: str | None = None
    wf_schedule_id: str | None = None


class CollectionsFeedsListResponse(BaseModel):
    items: list[CollectionsFeed]
    total: int
    pagination: PagePaginationMeta
