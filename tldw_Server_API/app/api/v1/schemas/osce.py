"""Strict public API contracts for OSCE station authoring and practice."""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from tldw_Server_API.app.api.v1.schemas.pagination import (
    OffsetPaginationMeta,
    default_offset_pagination_aliases,
)


class StrictModel(BaseModel):
    """Base model that rejects unknown fields and implicit coercion."""

    model_config = ConfigDict(extra="forbid", strict=True)


class QuizActivityType(str, Enum):
    """Persisted quiz activity discriminator."""

    QUESTIONS = "questions"
    OSCE = "osce"


class OsceVerificationState(str, Enum):
    """Verification lifecycle for evidence-bearing station content."""

    SOURCE_VERIFIED = "source_verified"
    MODIFIED_AFTER_VERIFICATION = "modified_after_verification"
    MANUALLY_AUTHORED = "manually_authored"


class OsceAttemptState(str, Enum):
    """Allowed lifecycle states for an OSCE practice attempt."""

    IN_PROGRESS = "in_progress"
    SELF_ASSESSMENT = "self_assessment"
    COMPLETED = "completed"


class OsceCitationSourceType(str, Enum):
    """Source kinds supported by strict OSCE citations."""

    MEDIA = "media"
    DOCUMENT = "document"
    URL = "url"
    NOTE = "note"
    FLASHCARD_DECK = "flashcard_deck"
    FLASHCARD_CARD = "flashcard_card"
    QUIZ_ATTEMPT = "quiz_attempt"
    QUIZ_ATTEMPT_QUESTION = "quiz_attempt_question"


class OsceStationOrigin(str, Enum):
    """Server-managed origin of a persisted OSCE station."""

    GENERATED = "generated"
    MANUAL = "manual"


BoundedSourceId = Annotated[str, StringConstraints(min_length=1, max_length=512)]
BoundedLabel = Annotated[str, StringConstraints(min_length=1, max_length=200)]
BoundedTimestamp = Annotated[str, StringConstraints(min_length=1, max_length=64)]
OsceUuid = Annotated[UUID, Field(strict=False)]
OsceCitationSource = Annotated[OsceCitationSourceType, Field(strict=False)]
OsceVerificationStatus = Annotated[OsceVerificationState, Field(strict=False)]
OsceOrigin = Annotated[OsceStationOrigin, Field(strict=False)]
OsceLifecycleState = Annotated[OsceAttemptState, Field(strict=False)]


class OsceCitation(StrictModel):
    """Source citation with locators constrained by source kind."""

    source_type: OsceCitationSource
    source_id: BoundedSourceId
    label: Annotated[str, StringConstraints(max_length=200)] | None = None
    quote: Annotated[str, StringConstraints(max_length=1000)] | None = None
    media_id: int | None = Field(default=None, ge=1)
    chunk_id: Annotated[str, StringConstraints(min_length=1, max_length=512)] | None = None
    timestamp_seconds: float | None = Field(default=None, ge=0)
    page_number: int | None = Field(default=None, ge=1)
    source_url: Annotated[str, StringConstraints(min_length=1, max_length=2048)] | None = None

    @model_validator(mode="after")
    def validate_source_locator(self) -> OsceCitation:
        populated = {
            name
            for name in (
                "media_id",
                "chunk_id",
                "timestamp_seconds",
                "page_number",
                "source_url",
            )
            if getattr(self, name) is not None
        }
        allowed = {
            OsceCitationSourceType.MEDIA: {"media_id", "chunk_id", "timestamp_seconds"},
            OsceCitationSourceType.DOCUMENT: {"chunk_id", "page_number"},
            OsceCitationSourceType.URL: {"source_url"},
            OsceCitationSourceType.NOTE: {"chunk_id"},
            OsceCitationSourceType.FLASHCARD_DECK: {"chunk_id"},
            OsceCitationSourceType.FLASHCARD_CARD: {"chunk_id"},
            OsceCitationSourceType.QUIZ_ATTEMPT: {"chunk_id"},
            OsceCitationSourceType.QUIZ_ATTEMPT_QUESTION: {"chunk_id"},
        }[self.source_type]
        inconsistent = populated - allowed
        if inconsistent:
            fields = ", ".join(sorted(inconsistent))
            raise ValueError(f"invalid locator fields for {self.source_type.value}: {fields}")

        if self.source_type is OsceCitationSourceType.URL:
            if self.source_url is None:
                raise ValueError("url citations require source_url")
            parsed = urlsplit(self.source_url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError("source_url must be an absolute HTTP(S) URL")
        return self


class OscePatientContext(StrictModel):
    """Evidence-bearing simulated-patient context."""

    text: Annotated[str, StringConstraints(min_length=1, max_length=10000)]
    citations: list[OsceCitation] = Field(default_factory=list)


class OsceChecklistItemCreate(StrictModel):
    """Checklist item accepted during station creation."""

    label: Annotated[str, StringConstraints(min_length=1, max_length=1000)]
    rationale: Annotated[str, StringConstraints(max_length=2000)] | None = None
    citations: list[OsceCitation] = Field(default_factory=list)


class OsceChecklistItemUpdate(OsceChecklistItemCreate):
    """Checklist replacement item with an optional server identity."""

    id: OsceUuid | None = None


class OsceChecklistItemStored(OsceChecklistItemCreate):
    """Persisted checklist item with a required server identity."""

    id: OsceUuid


class OsceRubricLevelCreate(StrictModel):
    """Rubric level accepted during station creation."""

    label: BoundedLabel
    description: Annotated[str, StringConstraints(min_length=1, max_length=2000)]


class OsceRubricLevelUpdate(OsceRubricLevelCreate):
    """Rubric replacement level with an optional server identity."""

    id: OsceUuid | None = None


class OsceRubricLevelStored(OsceRubricLevelCreate):
    """Persisted rubric level with a required server identity."""

    id: OsceUuid


def _reject_duplicate_level_labels(levels: list[Any]) -> None:
    normalized = [level.label.strip().casefold() for level in levels]
    if len(normalized) != len(set(normalized)):
        raise ValueError("duplicate rubric level label")


class OsceRubricDomainCreate(StrictModel):
    """Ordered rubric domain accepted during station creation."""

    label: BoundedLabel
    levels: list[OsceRubricLevelCreate] = Field(min_length=2, max_length=6)

    @model_validator(mode="after")
    def validate_level_labels(self) -> OsceRubricDomainCreate:
        _reject_duplicate_level_labels(self.levels)
        return self


class OsceRubricDomainUpdate(StrictModel):
    """Rubric replacement domain with optional nested identities."""

    id: OsceUuid | None = None
    label: BoundedLabel
    levels: list[OsceRubricLevelUpdate] = Field(min_length=2, max_length=6)

    @model_validator(mode="after")
    def validate_level_labels(self) -> OsceRubricDomainUpdate:
        _reject_duplicate_level_labels(self.levels)
        return self


class OsceRubricDomainStored(StrictModel):
    """Persisted rubric domain with required nested identities."""

    id: OsceUuid
    label: BoundedLabel
    levels: list[OsceRubricLevelStored] = Field(min_length=2, max_length=6)

    @model_validator(mode="after")
    def validate_level_labels(self) -> OsceRubricDomainStored:
        _reject_duplicate_level_labels(self.levels)
        return self


class OsceKeyPointCreate(StrictModel):
    """Evidence-bearing key point accepted during station creation."""

    text: Annotated[str, StringConstraints(min_length=1, max_length=2000)]
    citations: list[OsceCitation] = Field(default_factory=list)


class OsceKeyPointUpdate(OsceKeyPointCreate):
    """Key-point replacement with an optional server identity."""

    id: OsceUuid | None = None


class OsceKeyPointStored(OsceKeyPointCreate):
    """Persisted key point with a required server identity."""

    id: OsceUuid


def _nested_ids(content: Any) -> list[UUID]:
    ids: list[UUID] = []
    for item in content.checklist_items or []:
        if item.id is not None:
            ids.append(item.id)
    for domain in content.rubric_domains or []:
        if domain.id is not None:
            ids.append(domain.id)
        ids.extend(level.id for level in domain.levels if level.id is not None)
    for point in content.expected_key_points or []:
        if point.id is not None:
            ids.append(point.id)
    return ids


def _reject_duplicate_nested_ids(content: Any) -> None:
    ids = _nested_ids(content)
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate UUID in station content")


class OsceStationCreateContent(StrictModel):
    """Complete editable content required to create one station."""

    schema_version: Literal["osce.station.v1"] = "osce.station.v1"
    title: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    candidate_instructions: Annotated[str, StringConstraints(min_length=1, max_length=4000)]
    candidate_task: Annotated[str, StringConstraints(min_length=1, max_length=4000)]
    patient_context: OscePatientContext
    recommended_duration_seconds: int = Field(ge=60, le=7200)
    checklist_items: list[OsceChecklistItemCreate] = Field(min_length=1, max_length=50)
    rubric_domains: list[OsceRubricDomainCreate] = Field(min_length=1, max_length=12)
    expected_key_points: list[OsceKeyPointCreate] = Field(min_length=1, max_length=50)


class OsceStationUpdateContent(StrictModel):
    """Partial station content whose supplied collections replace atomically."""

    schema_version: Literal["osce.station.v1"] | None = None
    title: Annotated[str, StringConstraints(min_length=1, max_length=200)] | None = None
    candidate_instructions: Annotated[
        str, StringConstraints(min_length=1, max_length=4000)
    ] | None = None
    candidate_task: Annotated[str, StringConstraints(min_length=1, max_length=4000)] | None = None
    patient_context: OscePatientContext | None = None
    recommended_duration_seconds: int | None = Field(default=None, ge=60, le=7200)
    checklist_items: list[OsceChecklistItemUpdate] | None = Field(
        default=None, min_length=1, max_length=50
    )
    rubric_domains: list[OsceRubricDomainUpdate] | None = Field(
        default=None, min_length=1, max_length=12
    )
    expected_key_points: list[OsceKeyPointUpdate] | None = Field(
        default=None, min_length=1, max_length=50
    )

    @model_validator(mode="after")
    def validate_unique_nested_ids(self) -> OsceStationUpdateContent:
        supplied_nulls = {
            field_name
            for field_name in self.model_fields_set
            if getattr(self, field_name) is None
        }
        if supplied_nulls:
            fields = ", ".join(sorted(supplied_nulls))
            raise ValueError(f"station update fields cannot be null: {fields}")
        _reject_duplicate_nested_ids(self)
        return self


class OsceStationStoredContent(StrictModel):
    """Fully materialized station content with server-owned nested IDs."""

    schema_version: Literal["osce.station.v1"] = "osce.station.v1"
    title: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    candidate_instructions: Annotated[str, StringConstraints(min_length=1, max_length=4000)]
    candidate_task: Annotated[str, StringConstraints(min_length=1, max_length=4000)]
    patient_context: OscePatientContext
    recommended_duration_seconds: int = Field(ge=60, le=7200)
    checklist_items: list[OsceChecklistItemStored] = Field(min_length=1, max_length=50)
    rubric_domains: list[OsceRubricDomainStored] = Field(min_length=1, max_length=12)
    expected_key_points: list[OsceKeyPointStored] = Field(min_length=1, max_length=50)

    @model_validator(mode="after")
    def validate_unique_nested_ids(self) -> OsceStationStoredContent:
        _reject_duplicate_nested_ids(self)
        return self


class OsceStationCreateRequest(StrictModel):
    """Request envelope for manually creating an ordered station."""

    content: OsceStationCreateContent
    order_index: int = Field(default=0, ge=0)


class OsceStationPatchRequest(StrictModel):
    """Optimistic request envelope for a partial station update."""

    expected_version: int = Field(ge=1)
    content: OsceStationUpdateContent
    order_index: int | None = Field(default=None, ge=0)


class _OsceOffsetPage(StrictModel):
    """Shared offset pagination fields for OSCE list responses."""

    count: int = Field(ge=0)
    has_more: bool | None = None
    next_offset: int | None = Field(default=None, ge=0)
    pagination: OffsetPaginationMeta

    @model_validator(mode="after")
    def populate_pagination_aliases(self) -> _OsceOffsetPage:
        """Populate legacy top-level pagination aliases."""

        return default_offset_pagination_aliases(self)


class OsceStationSummary(StrictModel):
    """Compact station metadata that omits marking-guide content."""

    id: int = Field(ge=1)
    quiz_id: int = Field(ge=1)
    title: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    recommended_duration_seconds: int = Field(ge=60, le=7200)
    order_index: int = Field(ge=0)
    version: int = Field(ge=1)
    checklist_count: int = Field(ge=1, le=50)
    rubric_domain_count: int = Field(ge=1, le=12)
    verification_state: OsceVerificationStatus
    created_at: BoundedTimestamp
    updated_at: BoundedTimestamp


class OsceStationSummaryPage(_OsceOffsetPage):
    """Paginated station summaries without authoring content."""

    items: list[OsceStationSummary]


class OsceStationAuthoringResponse(StrictModel):
    """Full station detail available through authoring routes."""

    id: int = Field(ge=1)
    quiz_id: int = Field(ge=1)
    content: OsceStationStoredContent
    order_index: int = Field(ge=0)
    version: int = Field(ge=1)
    origin: OsceOrigin
    provenance: dict[str, Any] | None = None
    source_bundle: list[dict[str, Any]] = Field(default_factory=list)
    verification_state: OsceVerificationStatus
    verification_timestamp: BoundedTimestamp | None = None
    verification_summary: Annotated[str, StringConstraints(max_length=2000)] | None = None
    deleted: bool = False
    created_at: BoundedTimestamp
    updated_at: BoundedTimestamp


ChecklistSelection = Literal["met", "not_met"]


class OsceAttemptCreate(StrictModel):
    """Retry-safe attempt creation request."""

    client_attempt_id: OsceUuid


class OsceAttemptPatch(StrictModel):
    """Optimistic phase-appropriate attempt update."""

    expected_version: int = Field(ge=1)
    notes: Annotated[str, StringConstraints(max_length=10000)] | None = None
    checklist_selections: dict[OsceUuid, ChecklistSelection] | None = None
    rubric_selections: dict[OsceUuid, OsceUuid] | None = None


class OsceAttemptTransition(StrictModel):
    """Optimistic attempt lifecycle transition request."""

    expected_version: int = Field(ge=1)


class OsceCandidateCitation(StrictModel):
    """Candidate-safe citation projection without source locators or quotes."""

    source_type: OsceCitationSource
    source_id: BoundedSourceId
    label: Annotated[str, StringConstraints(max_length=200)] | None = None


class OsceCandidatePatientContext(StrictModel):
    """Candidate-visible patient context with sanitized citations."""

    text: Annotated[str, StringConstraints(min_length=1, max_length=10000)]
    citations: list[OsceCandidateCitation] = Field(default_factory=list)


class OsceCandidateStation(StrictModel):
    """Candidate-phase station projection without a marking guide."""

    schema_version: Literal["osce.station.v1"] = "osce.station.v1"
    title: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    candidate_instructions: Annotated[str, StringConstraints(min_length=1, max_length=4000)]
    candidate_task: Annotated[str, StringConstraints(min_length=1, max_length=4000)]
    patient_context: OsceCandidatePatientContext
    recommended_duration_seconds: int = Field(ge=60, le=7200)


class OsceCandidateAttemptResponse(StrictModel):
    """Attempt response available before self-assessment begins."""

    id: int = Field(ge=1)
    quiz_id: int = Field(ge=1)
    station_id: int = Field(ge=1)
    client_attempt_id: OsceUuid
    state: Literal[OsceAttemptState.IN_PROGRESS] = OsceAttemptState.IN_PROGRESS
    version: int = Field(ge=1)
    station: OsceCandidateStation
    notes: Annotated[str, StringConstraints(max_length=10000)] = ""
    started_at: BoundedTimestamp
    last_modified_at: BoundedTimestamp
    server_time: BoundedTimestamp


class OsceRevealedAttemptResponse(StrictModel):
    """Attempt response after the learner reveals the marking guide."""

    id: int = Field(ge=1)
    quiz_id: int = Field(ge=1)
    station_id: int = Field(ge=1)
    client_attempt_id: OsceUuid
    state: Literal[OsceAttemptState.SELF_ASSESSMENT, OsceAttemptState.COMPLETED]
    version: int = Field(ge=1)
    station: OsceStationStoredContent
    notes: Annotated[str, StringConstraints(max_length=10000)] = ""
    checklist_selections: dict[OsceUuid, ChecklistSelection] = Field(default_factory=dict)
    rubric_selections: dict[OsceUuid, OsceUuid] = Field(default_factory=dict)
    started_at: BoundedTimestamp
    self_assessment_started_at: BoundedTimestamp
    completed_at: BoundedTimestamp | None = None
    elapsed_seconds: int = Field(ge=0)
    last_modified_at: BoundedTimestamp
    server_time: BoundedTimestamp


class OsceRubricResult(StrictModel):
    """Selected rubric level for one domain in a completed attempt."""

    domain_id: OsceUuid
    domain_label: BoundedLabel
    level_id: OsceUuid
    level_label: BoundedLabel


class OsceAttemptSummary(StrictModel):
    """Note-free attempt summary for resume and results lists."""

    id: int = Field(ge=1)
    quiz_id: int = Field(ge=1)
    station_id: int = Field(ge=1)
    client_attempt_id: OsceUuid
    station_title: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    state: OsceLifecycleState
    version: int = Field(ge=1)
    started_at: BoundedTimestamp
    self_assessment_started_at: BoundedTimestamp | None = None
    completed_at: BoundedTimestamp | None = None
    last_modified_at: BoundedTimestamp
    elapsed_seconds: int | None = Field(default=None, ge=0)
    checklist_met_count: int | None = Field(default=None, ge=0, le=50)
    checklist_total: int | None = Field(default=None, ge=1, le=50)
    rubric_results: list[OsceRubricResult] = Field(default_factory=list, max_length=12)


class OsceAttemptSummaryPage(_OsceOffsetPage):
    """Paginated attempt summaries without notes or station guides."""

    items: list[OsceAttemptSummary]
