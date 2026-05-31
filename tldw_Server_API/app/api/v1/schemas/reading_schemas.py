from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, HttpUrl, model_validator, validator

from tldw_Server_API.app.api.v1.schemas._compat import Field
from tldw_Server_API.app.api.v1.schemas.pagination import OffsetPaginationMeta

_READING_SAVED_SEARCH_ALLOWED_QUERY_KEYS = {
    "q",
    "status",
    "tags",
    "favorite",
    "domain",
    "date_from",
    "date_to",
    "sort",
}
_READING_SAVED_SEARCH_ALLOWED_STATUSES = {"saved", "reading", "read", "archived"}
_READING_SAVED_SEARCH_ALLOWED_SORTS = {
    "updated_desc",
    "updated_asc",
    "created_desc",
    "created_asc",
    "title_asc",
    "title_desc",
    "relevance",
}


def _default_offset_pagination_aliases(response):
    if response.has_more is None:
        response.has_more = response.pagination.has_more
    elif response.has_more != response.pagination.has_more:
        raise ValueError(
            f"has_more alias mismatch: has_more={response.has_more} "
            f"pagination.has_more={response.pagination.has_more}"
        )
    if response.next_offset is None:
        response.next_offset = response.pagination.next_offset
    elif response.next_offset != response.pagination.next_offset:
        raise ValueError(
            f"next_offset alias mismatch: next_offset={response.next_offset} "
            f"pagination.next_offset={response.pagination.next_offset}"
        )
    return response


def _normalize_nonempty_string(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name}_must_be_string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name}_must_not_be_blank")
    return normalized


def _normalize_saved_search_sort(value: Any) -> str:
    normalized = _normalize_nonempty_string(value, field_name="sort").lower()
    if normalized not in _READING_SAVED_SEARCH_ALLOWED_SORTS:
        raise ValueError("sort_invalid")
    return normalized


def _normalize_saved_search_query(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("query_must_be_object")

    normalized: dict[str, Any] = {}
    for key, raw in value.items():
        if key not in _READING_SAVED_SEARCH_ALLOWED_QUERY_KEYS:
            raise ValueError(f"unsupported_query_key:{key}")
        if key in {"q", "domain", "date_from", "date_to"}:
            normalized[key] = _normalize_nonempty_string(raw, field_name=key)
            continue
        if key == "favorite":
            if not isinstance(raw, bool):
                raise ValueError("favorite_must_be_boolean")
            normalized[key] = raw
            continue
        if key == "status":
            if isinstance(raw, str):
                status = raw.strip().lower()
                if status not in _READING_SAVED_SEARCH_ALLOWED_STATUSES:
                    raise ValueError("status_invalid")
                normalized[key] = status
                continue
            if isinstance(raw, list):
                statuses = []
                for entry in raw:
                    if not isinstance(entry, str):
                        raise ValueError("status_values_must_be_strings")
                    status = entry.strip().lower()
                    if status not in _READING_SAVED_SEARCH_ALLOWED_STATUSES:
                        raise ValueError("status_invalid")
                    statuses.append(status)
                if not statuses:
                    raise ValueError("status_values_must_not_be_empty")
                normalized[key] = statuses
                continue
            raise ValueError("status_must_be_string_or_list")
        if key == "tags":
            if not isinstance(raw, list):
                raise ValueError("tags_must_be_list")
            tags = []
            for entry in raw:
                tags.append(_normalize_nonempty_string(entry, field_name="tag"))
            normalized[key] = tags
            continue
        if key == "sort":
            normalized[key] = _normalize_saved_search_sort(raw)
            continue
    return normalized


class ReadingSaveRequest(BaseModel):
    url: HttpUrl = Field(example="https://example.com/article")
    title: str | None = Field(default=None, example="Example Article")
    tags: list[str] = Field(default_factory=list, example=["ai", "reading"])
    status: str | None = Field(default="saved", description="saved|reading|read|archived", example="saved")
    archive_mode: Literal["use_default", "always", "never"] = Field(
        default="use_default",
        description="Archive policy override for this save request.",
        example="always",
    )
    favorite: bool = False
    summary: str | None = Field(default=None, example="Short summary for quick scan.")
    notes: str | None = Field(default=None, example="Why this matters for the project.")
    content: str | None = Field(
        default=None,
        description="Optional inline content override (testing/offline)",
        example="Inline article content used for testing.",
    )

    @validator("tags", pre=True, each_item=True)
    def _strip_tags(cls, value: str) -> str:
        return value.strip()


class ReadingItem(BaseModel):
    id: int
    media_id: int | None = Field(default=None, example=42)
    media_uuid: str | None = None
    title: str = Field(example="Example Article")
    url: str | None = Field(default=None, example="https://example.com/article")
    canonical_url: str | None = Field(default=None, example="https://example.com/article")
    domain: str | None = Field(default=None, example="example.com")
    summary: str | None = Field(default=None, example="Short summary for quick scan.")
    notes: str | None = Field(default=None, example="Why this matters for the project.")
    published_at: str | None = None
    status: str | None = Field(default=None, example="saved")
    processing_status: str | None = None
    archive_requested: bool = False
    has_archive_copy: bool = False
    last_fetch_error: str | None = None
    favorite: bool = False
    tags: list[str] = Field(default_factory=list, example=["ai", "reading"])
    created_at: str | None = None
    updated_at: str | None = None
    read_at: str | None = None


class ReadingItemDetail(ReadingItem):
    text: str | None = Field(default=None, example="Full extracted article text.")
    clean_html: str | None = Field(default=None, example="<p>Sanitized HTML...</p>")
    metadata: dict[str, Any] | None = None


class ReadingItemsListResponse(BaseModel):
    items: list[ReadingItem]
    total: int
    page: int
    size: int
    offset: int | None = None
    limit: int | None = None
    has_more: bool | None = Field(default=None, description="Alias for pagination.has_more")
    next_offset: int | None = Field(default=None, ge=0, description="Alias for pagination.next_offset")
    pagination: OffsetPaginationMeta

    @model_validator(mode="after")
    def _default_pagination_aliases(self):
        return _default_offset_pagination_aliases(self)


class ReadingSavedSearchCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255, example="Morning")
    query: dict[str, Any] = Field(default_factory=dict, example={"q": "ai", "status": ["saved"]})
    sort: str | None = Field(
        default=None,
        description="updated_desc|updated_asc|created_desc|created_asc|title_asc|title_desc|relevance",
    )

    @validator("name")
    def _normalize_name(cls, value: str) -> str:
        return _normalize_nonempty_string(value, field_name="name")

    @validator("query", pre=True)
    def _normalize_query(cls, value: Any) -> dict[str, Any]:
        return _normalize_saved_search_query(value)

    @validator("sort")
    def _normalize_sort(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _normalize_saved_search_sort(value)


class ReadingSavedSearchUpdateRequest(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=255)
    query: dict[str, Any] | None = None
    sort: str | None = Field(
        default=None,
        description="updated_desc|updated_asc|created_desc|created_asc|title_asc|title_desc|relevance",
    )

    @validator("name")
    def _normalize_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _normalize_nonempty_string(value, field_name="name")

    @validator("query", pre=True)
    def _normalize_query(cls, value: Any) -> dict[str, Any] | None:
        if value is None:
            return None
        return _normalize_saved_search_query(value)

    @validator("sort")
    def _normalize_sort(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _normalize_saved_search_sort(value)


class ReadingSavedSearchResponse(BaseModel):
    id: int
    name: str
    query: dict[str, Any]
    sort: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


class ReadingSavedSearchListResponse(BaseModel):
    items: list[ReadingSavedSearchResponse]
    total: int
    limit: int
    offset: int
    has_more: bool | None = Field(default=None, description="Alias for pagination.has_more")
    next_offset: int | None = Field(default=None, ge=0, description="Alias for pagination.next_offset")
    pagination: OffsetPaginationMeta

    @model_validator(mode="after")
    def _default_pagination_aliases(self):
        return _default_offset_pagination_aliases(self)


class ReadingNoteLinkCreateRequest(BaseModel):
    note_id: str = Field(..., min_length=1, max_length=255, example="8ef2f7ad-2a2f-4442-9adf-23c36dcf0f8d")


class ReadingNoteLinkResponse(BaseModel):
    item_id: int
    note_id: str
    created_at: str | None = None


class ReadingNoteLinksListResponse(BaseModel):
    item_id: int
    links: list[ReadingNoteLinkResponse]


class ReadingImportResponse(BaseModel):
    source: str
    imported: int
    updated: int
    skipped: int
    errors: list[str] = Field(default_factory=list)

ReadingImportJobState = Literal[
    "queued",
    "processing",
    "completed",
    "failed",
    "cancelled",
    "quarantined",
]


class ReadingImportJobResponse(BaseModel):
    """Response payload for a newly created reading import job."""

    job_id: int
    job_uuid: str | None = None
    status: ReadingImportJobState


class ReadingImportJobStatus(BaseModel):
    """Status payload for tracking a reading import job."""

    job_id: int
    job_uuid: str | None = None
    status: ReadingImportJobState
    created_at: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    progress_percent: float | None = None
    progress_message: str | None = None
    error_message: str | None = None
    result: ReadingImportResponse | None = None


class ReadingImportJobsListResponse(BaseModel):
    jobs: list[ReadingImportJobStatus]
    total: int
    limit: int | None = None
    offset: int | None = None
    has_more: bool | None = Field(default=None, description="Alias for pagination.has_more")
    next_offset: int | None = Field(default=None, ge=0, description="Alias for pagination.next_offset")
    pagination: OffsetPaginationMeta

    @model_validator(mode="after")
    def _default_pagination_aliases(self):
        return _default_offset_pagination_aliases(self)


class ReadingDigestSuggestionsConfig(BaseModel):
    enabled: bool = False
    limit: int | None = Field(default=None, ge=1, le=200)
    status: list[Literal["saved", "reading", "read", "archived"]] | None = None
    exclude_tags: list[str] | None = None
    max_age_days: int | None = Field(default=None, ge=1, le=3650)
    include_read: bool = False
    include_archived: bool = False

    @validator("status", pre=True)
    def _coerce_suggestions_status(cls, value: Any) -> Any:
        if isinstance(value, str):
            return [value]
        return value

    @validator("exclude_tags", pre=True)
    def _coerce_exclude_tags(cls, value: Any) -> Any:
        if isinstance(value, str):
            return [value]
        return value

    @validator("exclude_tags", pre=True, each_item=True)
    def _strip_exclude_tags(cls, value: str) -> str:
        return value.strip()


class ReadingDigestScheduleFilters(BaseModel):
    status: list[Literal["saved", "reading", "read", "archived"]] | None = None
    tags: list[str] | None = None
    favorite: bool | None = None
    domain: str | None = None
    q: str | None = None
    date_from: str | None = None
    date_to: str | None = None
    sort: str | None = Field(
        default=None,
        description="updated_desc|updated_asc|created_desc|created_asc|title_asc|title_desc|relevance",
    )
    limit: int | None = Field(default=None, ge=1, le=500)
    suggestions: ReadingDigestSuggestionsConfig | None = None

    @validator("status", pre=True)
    def _coerce_status_list(cls, value: Any) -> Any:
        if isinstance(value, str):
            return [value]
        return value

    @validator("tags", pre=True)
    def _coerce_tags_list(cls, value: Any) -> Any:
        if isinstance(value, str):
            return [value]
        return value

    @validator("tags", pre=True, each_item=True)
    def _strip_digest_tags(cls, value: str) -> str:
        return value.strip()


class ReadingDigestScheduleCreateRequest(BaseModel):
    name: str | None = None
    cron: str = Field(..., description="Cron expression, e.g., '0 8 * * *'")
    timezone: str | None = Field(
        None,
        description="IANA timezone name (e.g., 'UTC', 'America/New_York')",
    )
    enabled: bool = True
    require_online: bool = False
    format: Literal["md", "html"] = Field(default="md")
    template_id: int | None = None
    template_name: str | None = None
    retention_days: int | None = Field(
        default=None,
        ge=0,
        le=3650,
        deprecated=True,
        description="Deprecated. Use retention.default_seconds instead.",
    )
    filters: ReadingDigestScheduleFilters | None = None


class ReadingDigestScheduleUpdateRequest(BaseModel):
    name: str | None = None
    cron: str | None = None
    timezone: str | None = None
    enabled: bool | None = None
    require_online: bool | None = None
    format: Literal["md", "html"] | None = None
    template_id: int | None = None
    template_name: str | None = None
    retention_days: int | None = Field(
        default=None,
        ge=0,
        le=3650,
        deprecated=True,
        description="Deprecated. Use retention.default_seconds instead.",
    )
    filters: ReadingDigestScheduleFilters | None = None


class ReadingDigestScheduleResponse(BaseModel):
    id: str
    name: str | None = None
    cron: str
    timezone: str | None = None
    enabled: bool
    require_online: bool
    format: Literal["md", "html"]
    template_id: int | None = None
    template_name: str | None = None
    retention_days: int | None = Field(
        default=None,
        deprecated=True,
        description="Deprecated. Use retention.default_seconds instead.",
    )
    filters: ReadingDigestScheduleFilters | None = None
    last_run_at: str | None = None
    next_run_at: str | None = None
    last_status: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


class ReadingDigestOutput(BaseModel):
    output_id: int
    title: str
    format: Literal["md", "html"]
    created_at: str | None = None
    download_url: str
    schedule_id: str | None = None
    schedule_name: str | None = None
    item_count: int | None = None
    metadata: dict[str, Any] | None = None


class ReadingDigestOutputsListResponse(BaseModel):
    items: list[ReadingDigestOutput]
    total: int
    limit: int | None = None
    offset: int | None = None
    has_more: bool | None = Field(default=None, description="Alias for pagination.has_more")
    next_offset: int | None = Field(default=None, ge=0, description="Alias for pagination.next_offset")
    pagination: OffsetPaginationMeta

    @model_validator(mode="after")
    def _default_pagination_aliases(self):
        return _default_offset_pagination_aliases(self)


class ReadingArchiveCreateRequest(BaseModel):
    format: Literal["html", "md"] = Field(
        default="html",
        description="Archive format (html or md).",
        example="html",
    )
    source: Literal["auto", "clean_html", "text"] = Field(
        default="auto",
        description="Source content preference (auto uses clean_html then text).",
        example="auto",
    )
    title: str | None = Field(default=None, max_length=200, description="Optional archive title override")
    retention_days: int | None = Field(
        default=None,
        ge=0,
        le=3650,
        description="Retention window in days (0 to disable retention). Ignored if retention_until is provided.",
    )
    retention_until: str | None = Field(
        default=None,
        description="ISO timestamp when this archive can be purged. Takes precedence over retention_days when set.",
        example="2025-12-31T00:00:00Z",
    )


class ReadingArchiveResponse(BaseModel):
    output_id: int
    title: str
    format: Literal["html", "md"]
    storage_path: str
    created_at: str | None = None
    retention_until: str | None = None
    download_url: str


class ReadingDeleteResponse(BaseModel):
    status: str = Field(example="archived")
    item_id: int = Field(example=123)
    hard: bool = False


class ReadingSummarizeRequest(BaseModel):
    provider: str | None = Field(
        default=None,
        description="LLM provider (e.g., openai, anthropic)",
        example="openai",
    )
    model: str | None = Field(default=None, description="Optional model override", example="gpt-4o-mini")
    prompt: str | None = Field(
        default=None,
        description="Optional user prompt override",
        example="Summarize for a product brief.",
    )
    system_prompt: str | None = Field(
        default=None,
        description="Optional system prompt override",
        example="You are a concise research assistant.",
    )
    temperature: float | None = Field(default=None, ge=0.0, le=2.0, example=0.4)
    recursive: bool = Field(default=False, description="Enable recursive summarization")
    chunked: bool = Field(default=False, description="Summarize chunks and concatenate")


class ReadingCitation(BaseModel):
    item_id: int = Field(example=123)
    url: str | None = Field(default=None, example="https://example.com/article")
    canonical_url: str | None = Field(default=None, example="https://example.com/article")
    title: str | None = Field(default=None, example="Example Article")
    source: str = "reading"


class ReadingSummaryResponse(BaseModel):
    item_id: int = Field(example=123)
    summary: str = Field(example="Short summary text...")
    provider: str = Field(example="openai")
    model: str | None = Field(default=None, example="gpt-4o-mini")
    citations: list[ReadingCitation] = Field(default_factory=list)
    generated_at: str | None = None


class ReadingTTSRequest(BaseModel):
    model: str = Field(..., min_length=1, description="TTS model identifier", example="kokoro")
    voice: str = Field(default="af_heart", description="TTS voice identifier", example="af_heart")
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = Field(default="mp3")
    stream: bool = Field(default=True)
    speed: float | None = Field(default=None, ge=0.25, le=4.0, example=1.0)
    max_chars: int | None = Field(default=None, ge=1, le=200000, example=12000)
    text_source: Literal["text", "summary", "notes"] | None = Field(
        default=None,
        description="Optional override for which field to render",
        example="text",
    )


class ReadingUpdateRequest(BaseModel):
    status: str | None = Field(
        default=None,
        description="saved|reading|read|archived",
        example="read",
    )
    favorite: bool | None = None
    tags: list[str] | None = Field(default=None, example=["ai", "priority"])
    notes: str | None = Field(default=None, example="Follow up in the next sprint.")
    title: str | None = Field(default=None, example="Updated Article Title")

    @validator("tags", pre=True, each_item=True)
    def _strip_tags(cls, value: str) -> str:
        return value.strip()
