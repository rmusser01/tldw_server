from __future__ import annotations

from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator


PlaylistDuplicateStatus = Literal[
    "new",
    "duplicate_in_batch",
    "duplicate_existing",
    "unknown",
]


class PlaylistPreflightRequest(BaseModel):
    url: str = Field(..., min_length=1, description="Playlist or playlist-context URL to inspect")
    max_items: int = Field(100, ge=1, le=500, description="Maximum playlist entries to return")
    timeout_seconds: int = Field(20, ge=1, le=60, description="Maximum preflight wait time")

    @field_validator("url")
    @classmethod
    def _validate_http_url(cls, value: str) -> str:
        trimmed = value.strip()
        parsed = urlparse(trimmed)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("url must be an http(s) URL")
        return trimmed


class PlaylistPreflightItem(BaseModel):
    ordinal: int = Field(..., ge=1)
    source_url: str
    normalized_source_id: str | None = None
    source_kind: str
    title: str | None = None
    speaker: str | None = None
    duration_seconds: int | None = None
    published_at: str | None = None
    thumbnail_url: str | None = None
    duplicate_status: PlaylistDuplicateStatus = "unknown"
    duplicate_of_ordinal: int | None = None
    selected: bool = True


class PlaylistPreflightResponse(BaseModel):
    source_url: str
    source_kind: str
    playlist_id: str | None = None
    playlist_title: str | None = None
    video_id: str | None = None
    item_count: int
    selected_count: int
    duplicate_count: int
    warnings: list[str] = Field(default_factory=list)
    items: list[PlaylistPreflightItem] = Field(default_factory=list)
