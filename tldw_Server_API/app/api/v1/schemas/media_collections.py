from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, StringConstraints, field_validator, model_validator

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
    source_url: str | None = Field(default=None, min_length=1, description="Correct the source of an unresolved retry using its attempt and idempotency key.")
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


def positive_quick_ingest_media_id(value: Any) -> int | None:
    """Only admit a positive database identifier with a portable integer representation."""
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        return None
    text = str(value)
    if not text.isascii() or not text.isdecimal() or len(text) > 19:
        return None
    value = int(text)
    return value if 0 < value <= 2**63 - 1 else None


def _validate_plan_key(value: str) -> str:
    """A source key must address a single, stable HTTP path segment."""
    if value in {".", ".."}:
        raise ValueError("Plan and source keys cannot be relative path segments.")
    return value


PlanKey = Annotated[
    str,
    StringConstraints(pattern=r"^[A-Za-z0-9._:-]+$", min_length=1, max_length=200),
    AfterValidator(_validate_plan_key),
]


class QuickIngestPlanCollection(MediaCollectionCreateRequest):
    """Collection attributes permitted on the initial plan write."""

    model_config = ConfigDict(extra="forbid")


class QuickIngestPlanItem(MediaCollectionItemCreateRequest):
    """A source description cannot introduce worker or media receipts."""

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_initial_state(self) -> QuickIngestPlanItem:
        if (
            self.status not in {"planned", "skipped_existing"}
            or self.retry_count
            or self.warnings
            or any(
                value is not None
                for value in (
                    self.media_id,
                    self.content_item_id,
                    self.latest_job_id,
                    self.latest_run_id,
                    self.idempotency_key,
                    self.error_summary,
                )
            )
        ):
            raise ValueError("A plan item can describe a source but cannot supply a processing receipt.")
        return self


class QuickIngestPlanSource(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_key: PlanKey
    item: QuickIngestPlanItem


class QuickIngestPlanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    collection: QuickIngestPlanCollection
    items: list[QuickIngestPlanSource] = Field(min_length=1, max_length=5000)

    @field_validator("items")
    @classmethod
    def validate_distinct_keys(cls, items: list[QuickIngestPlanSource]) -> list[QuickIngestPlanSource]:
        if len({source.source_key for source in items}) != len(items):
            raise ValueError("Each source key must appear only once.")
        return items


QuickIngestDispatchState = Literal["unclaimed", "reserved", "jobs", "web", "fallback_ready", "add", "settled"]


class QuickIngestPlanItemResponse(BaseModel):
    dispatch_state: QuickIngestDispatchState
    source_key: str
    item: MediaCollectionItemResponse


class QuickIngestSourceStatusResponse(QuickIngestPlanItemResponse):
    collection_id: int


class QuickIngestPlanResponse(BaseModel):
    collection_id: int
    items: list[QuickIngestPlanItemResponse]


class QuickIngestRetryClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")
    retry_count: int = Field(ge=1)
    idempotency_key: str = Field(min_length=1, max_length=255)
    source_url: str | None = Field(default=None, min_length=1)

    @field_validator("source_url", "idempotency_key")
    @classmethod
    def trim_claim_strings(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("Expected a non-empty string.")
        return value.strip() if value is not None else None


class QuickIngestClaimResponse(BaseModel):
    dispatch_state: QuickIngestDispatchState
    claimed: bool
    item: MediaCollectionItemResponse
