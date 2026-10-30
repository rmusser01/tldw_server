"""Pydantic schemas for study-pack requests, jobs, and provenance responses."""

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from tldw_Server_API.app.api.v1.schemas.pagination import OffsetPaginationMeta

# Core-owned since TASK-13950.15; re-exported for the API layer.
from tldw_Server_API.app.core.StudyPacks.study_pack_schemas import (  # noqa: F401 - re-exported (TASK-13950.15)
    StudyPackCreateJobRequest,
    StudyPackSourceSelection,
    StudyPackSourceType,
)

StudyPackStatus = Literal["active", "superseded"]


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

    @field_validator("created_at", "last_modified", mode="before")
    @classmethod
    def _serialize_timestamps(cls, value: Any) -> Any:
        """Keep database datetimes compatible with the public string contract."""
        return value.isoformat() if isinstance(value, datetime) else value


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

    @field_validator("created_at", "last_modified", mode="before")
    @classmethod
    def _serialize_timestamps(cls, value: Any) -> Any:
        """Keep database datetimes compatible with the public string contract."""
        return value.isoformat() if isinstance(value, datetime) else value


class FlashcardDeepDiveTarget(BaseModel):
    """Resolved deep-dive target for a provenance-backed flashcard."""

    source_type: StudyPackSourceType
    source_id: str
    citation_ordinal: Optional[int] = Field(default=None, ge=0)
    route_kind: Optional[Literal["exact_locator", "workspace_route", "citation_only"]] = None
    route: Optional[str] = None
    available: bool = True
    fallback_reason: Optional[str] = None
