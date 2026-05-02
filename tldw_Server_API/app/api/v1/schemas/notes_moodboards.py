from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tldw_Server_API.app.api.v1.schemas.pagination import OffsetPaginationMeta


def _default_offset_pagination_aliases(response):
    if response.has_more is None:
        response.has_more = response.pagination.has_more
    if response.next_offset is None:
        response.next_offset = response.pagination.next_offset
    return response


def _normalize_nonempty_text(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError("Value cannot be empty.")
    return text


def _normalize_text_list(values: list[str] | None) -> list[str]:
    if not values:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        text = str(raw or "").strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


class MoodboardMembershipSource(str, Enum):
    manual = "manual"
    smart = "smart"
    both = "both"


class MoodboardSmartRuleDateRange(BaseModel):
    after: datetime | None = Field(default=None)
    before: datetime | None = Field(default=None)

    @model_validator(mode="after")
    def validate_range_order(self) -> "MoodboardSmartRuleDateRange":
        if self.after and self.before and self.after > self.before:
            raise ValueError("'after' cannot be later than 'before'.")
        return self


class MoodboardSmartRule(BaseModel):
    query: str | None = Field(default=None, max_length=2000)
    keyword_tokens: list[str] = Field(default_factory=list)
    notebook_collection_ids: list[int] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    updated: MoodboardSmartRuleDateRange | None = Field(default=None)

    @field_validator("query", mode="before")
    @classmethod
    def normalize_query(cls, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @field_validator("keyword_tokens", mode="before")
    @classmethod
    def normalize_keyword_tokens(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError("keyword_tokens must be a list of strings.")
        return _normalize_text_list([str(item) for item in value])

    @field_validator("sources", mode="before")
    @classmethod
    def normalize_sources(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError("sources must be a list of strings.")
        return _normalize_text_list([str(item) for item in value])

    @field_validator("notebook_collection_ids", mode="before")
    @classmethod
    def normalize_collection_ids(cls, value: Any) -> list[int]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError("notebook_collection_ids must be a list of integers.")
        out: list[int] = []
        seen: set[int] = set()
        for raw in value:
            number = int(raw)
            if number < 1:
                raise ValueError("notebook_collection_ids values must be >= 1.")
            if number in seen:
                continue
            seen.add(number)
            out.append(number)
        return out


class MoodboardBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: str | None = Field(default=None, max_length=2000)

    @field_validator("name", mode="before")
    @classmethod
    def normalize_name(cls, value: Any) -> str:
        return _normalize_nonempty_text(str(value or ""))

    @field_validator("description", mode="before")
    @classmethod
    def normalize_description(cls, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None


class MoodboardCreate(MoodboardBase):
    smart_rule: MoodboardSmartRule | None = Field(default=None)


class MoodboardUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=255)
    description: str | None = Field(default=None, max_length=2000)
    smart_rule: MoodboardSmartRule | None = Field(default=None)

    @field_validator("name", mode="before")
    @classmethod
    def normalize_name(cls, value: Any) -> str | None:
        if value is None:
            return None
        return _normalize_nonempty_text(str(value))

    @field_validator("description", mode="before")
    @classmethod
    def normalize_description(cls, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None


class MoodboardResponse(MoodboardBase):
    id: int = Field(..., ge=1)
    smart_rule: MoodboardSmartRule | None = Field(default=None)
    created_at: datetime
    last_modified: datetime
    version: int = Field(..., ge=1)
    client_id: str
    deleted: bool

    model_config = ConfigDict(from_attributes=True)


class MoodboardListResponse(BaseModel):
    items: list[MoodboardResponse] = Field(default_factory=list)
    moodboards: list[MoodboardResponse] = Field(default_factory=list)
    count: int = Field(default=0, ge=0)
    limit: int = Field(..., ge=1)
    offset: int = Field(..., ge=0)
    total: int | None = Field(default=None, ge=0)
    has_more: bool | None = Field(default=None, description="Alias for pagination.has_more")
    next_offset: int | None = Field(default=None, ge=0, description="Alias for pagination.next_offset")
    pagination: OffsetPaginationMeta

    @model_validator(mode="after")
    def _default_pagination_aliases(self):
        return _default_offset_pagination_aliases(self)


class MoodboardPinResponse(BaseModel):
    success: bool
    moodboard_id: int = Field(..., ge=1)
    note_id: str = Field(..., min_length=1)


class MoodboardNoteSummary(BaseModel):
    id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1, max_length=255)
    content_preview: str | None = Field(default=None)
    updated_at: datetime | None = Field(default=None)
    keywords: list[str] = Field(default_factory=list)
    cover_image_url: str | None = Field(default=None)
    membership_source: MoodboardMembershipSource


class MoodboardNotesListResponse(BaseModel):
    items: list[MoodboardNoteSummary] = Field(default_factory=list)
    notes: list[MoodboardNoteSummary] = Field(default_factory=list)
    count: int = Field(default=0, ge=0)
    limit: int = Field(..., ge=1)
    offset: int = Field(..., ge=0)
    total: int | None = Field(default=None, ge=0)
    has_more: bool | None = Field(default=None, description="Alias for pagination.has_more")
    next_offset: int | None = Field(default=None, ge=0, description="Alias for pagination.next_offset")
    pagination: OffsetPaginationMeta

    @model_validator(mode="after")
    def _default_pagination_aliases(self):
        return _default_offset_pagination_aliases(self)
