from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


MediaCollectionItemStatus = Literal[
    "planned",
    "processing",
    "completed",
    "skipped_existing",
    "submit_failed",
    "failed",
    "cancelled",
]


class MediaCollectionCreateRequest(BaseModel):
    name: str = Field(..., min_length=1)
    kind: str = Field("conference", min_length=1)
    description: str | None = None
    source_url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    default_tags: list[str] = Field(default_factory=list)

    @field_validator("name", "kind")
    @classmethod
    def _trim_required_string(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("value must not be empty")
        return trimmed


class MediaCollectionUpdateRequest(BaseModel):
    name: str | None = None
    kind: str | None = None
    description: str | None = None
    source_url: str | None = None
    metadata: dict[str, Any] | None = None
    default_tags: list[str] | None = None

    @field_validator("name", "kind")
    @classmethod
    def _trim_optional_required_string(cls, value: str | None) -> str | None:
        if value is None:
            return None
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("value must not be empty")
        return trimmed


class MediaCollectionItemCreateRequest(BaseModel):
    source_url: str = Field(..., min_length=1)
    normalized_source_id: str | None = None
    source_kind: str | None = None
    status: MediaCollectionItemStatus = "planned"
    ordinal: int | None = Field(default=None, ge=1)
    title: str | None = None
    speaker: str | None = None
    published_at: str | None = None
    track: str | None = None
    duplicate_status: str = "unknown"
    media_id: int | None = None
    content_item_id: int | None = None
    latest_job_id: str | None = None
    latest_run_id: int | None = None
    idempotency_key: str | None = None
    retry_count: int = Field(0, ge=0)
    error_summary: str | None = None
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)

    @field_validator("source_url")
    @classmethod
    def _trim_source_url(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("source_url must not be empty")
        return trimmed


class MediaCollectionItemUpdateRequest(BaseModel):
    ordinal: int | None = Field(default=None, ge=1)
    title: str | None = None
    speaker: str | None = None
    published_at: str | None = None
    track: str | None = None
    duplicate_status: str | None = None
    status: MediaCollectionItemStatus | None = None
    media_id: int | None = None
    content_item_id: int | None = None
    latest_job_id: str | None = None
    latest_run_id: int | None = None
    idempotency_key: str | None = None
    retry_count: int | None = Field(default=None, ge=0)
    error_summary: str | None = None
    warnings: list[str] | None = None
    metadata: dict[str, Any] | None = None
    tags: list[str] | None = None


class MediaCollectionItemResponse(BaseModel):
    id: int
    collection_id: int
    ordinal: int
    source_url: str
    normalized_source_id: str | None = None
    source_kind: str | None = None
    title: str | None = None
    speaker: str | None = None
    published_at: str | None = None
    track: str | None = None
    duplicate_status: str
    status: str
    media_id: int | None = None
    content_item_id: int | None = None
    latest_job_id: str | None = None
    latest_run_id: int | None = None
    idempotency_key: str | None = None
    retry_count: int
    error_summary: str | None = None
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    created_at: str
    updated_at: str


class MediaCollectionResponse(BaseModel):
    id: int
    name: str
    kind: str
    description: str | None = None
    source_url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    default_tags: list[str] = Field(default_factory=list)
    created_at: str
    updated_at: str
    items: list[MediaCollectionItemResponse] = Field(default_factory=list)


class MediaCollectionListResponse(BaseModel):
    items: list[MediaCollectionResponse]
    total: int
    page: int
    size: int
