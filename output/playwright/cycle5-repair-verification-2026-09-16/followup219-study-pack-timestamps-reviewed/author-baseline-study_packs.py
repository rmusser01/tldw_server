"""Pydantic schemas for study-pack requests, jobs, and provenance responses."""

from collections.abc import Mapping
from typing import Any, Literal, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator

from tldw_Server_API.app.api.v1.schemas.pagination import OffsetPaginationMeta


StudyPackSourceType = Literal["note", "media", "message"]
StudyPackStatus = Literal["active", "superseded"]


class StudyPackSourceSelection(BaseModel):
    """API-facing source selection for study-pack generation requests."""

    model_config = ConfigDict(populate_by_name=True)

    source_type: StudyPackSourceType
    source_id: str = Field(..., min_length=1)
    label: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("label", "source_title"),
    )
    excerpt_text: Optional[str] = None
    locator: dict[str, Any] = Field(default_factory=dict)

    @field_validator("source_id", mode="before")
    @classmethod
    def validate_source_id(cls, value: Any) -> str:
        if isinstance(value, str):
            value = value.strip()
        if not value:
            raise ValueError("source_id must not be blank")
        return str(value)

    @field_validator("label", "excerpt_text", mode="before")
    @classmethod
    def validate_optional_text(cls, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @field_validator("locator", mode="before")
    @classmethod
    def validate_locator(cls, value: Any) -> dict[str, Any]:
        if value in (None, "", [], ()):
            return {}
        if not isinstance(value, Mapping):
            raise ValueError("locator must be a mapping")
        return {
            str(key): item
            for key, item in value.items()
            if item not in (None, "", [], {})
        }


class StudyPackCreateJobRequest(BaseModel):
    """Request body for enqueuing a new study-pack generation job."""

    title: str = Field(..., min_length=1)
    workspace_id: Optional[str] = None
    deck_mode: Literal["new"] = "new"
    source_items: list[StudyPackSourceSelection] = Field(..., min_length=1)


class StudyPackSummaryResponse(BaseModel):
    """Serialized study-pack metadata returned by the API."""

    id: int
    workspace_id: Optional[str] = None
    title: str
    deck_id: Optional[int] = None
    source_bundle_json: dict[str, Any] = Field(default_factory=dict)
    generation_options_json: Optional[dict[str, Any]] = None
    status: StudyPackStatus
    superseded_by_pack_id: Optional[int] = None
    created_at: Optional[str] = None
    last_modified: Optional[str] = None
    deleted: bool
    client_id: str
    version: int


StudyPackJobApiStatus = Literal["queued", "running", "completed", "failed", "cancelled"]


class StudyPackJobSummaryResponse(BaseModel):
    """Summary fields for a study-pack generation job."""

    id: int
    status: StudyPackJobApiStatus
    domain: str
    queue: str
    job_type: str


class StudyPackJobAcceptedResponse(BaseModel):
    """Envelope returned when a study-pack job is accepted."""

    job: StudyPackJobSummaryResponse


class StudyPackJobStatusResponse(BaseModel):
    """Job status plus any completed study-pack result payload."""

    job: StudyPackJobSummaryResponse
    study_pack: Optional[StudyPackSummaryResponse] = None
    error: Optional[str] = None


class StudyPackJobListResponse(BaseModel):
    """Paginated summary envelope for study-pack job discovery."""

    jobs: list[StudyPackJobSummaryResponse] = Field(default_factory=list)
    total: int = Field(default=0, ge=0)
    pagination: OffsetPaginationMeta


class FlashcardCitationResponse(BaseModel):
    """Serialized flashcard citation row used by remediation UI."""

    id: int
    flashcard_uuid: str
    source_type: StudyPackSourceType
    source_id: str
    citation_text: Optional[str] = None
    locator: Optional[str] = None
    ordinal: int = Field(default=0, ge=0)
    created_at: Optional[str] = None
    last_modified: Optional[str] = None
    deleted: bool
    client_id: str
    version: int


class FlashcardDeepDiveTarget(BaseModel):
    """Resolved deep-dive target for a provenance-backed flashcard."""

    source_type: StudyPackSourceType
    source_id: str
    citation_ordinal: Optional[int] = Field(default=None, ge=0)
    route_kind: Optional[Literal["exact_locator", "workspace_route", "citation_only"]] = None
    route: Optional[str] = None
    available: bool = True
    fallback_reason: Optional[str] = None
