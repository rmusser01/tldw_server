"""Bounded wire models for publishing generated content to native Study."""

from __future__ import annotations

import re
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator, model_validator

ResearchStudyKind = Literal["flashcards", "quiz"]
StudyIdentifier = Annotated[
    str, StringConstraints(strict=True, max_length=256, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
]
StudyOwnerId = Annotated[int, Field(strict=True, ge=1, le=2**63 - 1)]
_CatalogId = Annotated[str, StringConstraints(strict=True, pattern=r"^[1-9][0-9]{0,18}$")]
_Fingerprint = Annotated[str, StringConstraints(strict=True, pattern=r"^[0-9a-f]{64}$")]
_StudyText = Annotated[
    str, StringConstraints(strict=True, strip_whitespace=True, min_length=1, max_length=6000, pattern=r"^[^\x00]+$")
]
_Label = Annotated[str, StringConstraints(strict=True, max_length=256)]
_Revision = Annotated[int, Field(strict=True, ge=1, le=2**63 - 1)]
_SecretLabel = re.compile(
    r"(?i)(://|[\x00-\x1f]|\b(?:bearer\s+\S+|sk-[A-Za-z0-9_-]+|(?:api[_-]?key|password|secret|token)\s*[:=]))"
)


class _StudyModel(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)


class ResearchStudySource(_StudyModel):
    """Exact version of a selected Workspace Media association."""

    source_id: StudyIdentifier
    catalog_item_id: _CatalogId
    source_type: StudyIdentifier
    association_version: _Revision
    catalog_item_version: _Revision | None = None
    ready_modes: tuple[Literal["fts", "semantic", "hybrid"], ...] = Field(default=(), max_length=3)

    @field_validator("catalog_item_id")
    @classmethod
    def _int64_catalog_id(cls, value: str) -> str:
        if int(value) > 2**63 - 1:
            raise ValueError("Media ID is out of range")
        return value


class ResearchStudyGeneration(_StudyModel):
    """Payload-free provenance alongside the canonical owner."""

    operation_id: StudyIdentifier
    output_kind: ResearchStudyKind
    source_snapshot: tuple[ResearchStudySource, ...] = Field(min_length=1, max_length=100)
    settings_fingerprint: _Fingerprint
    processing_route_fingerprint: _Fingerprint
    provider: _Label
    model: _Label
    status: Literal["complete"] = "complete"
    supersedes_owner_id: _CatalogId | None = None

    @field_validator("provider", "model")
    @classmethod
    def _processing_label(cls, value: str) -> str:
        if _SecretLabel.search(value):
            raise ValueError("Processing label is invalid")
        return value

    @field_validator("supersedes_owner_id")
    @classmethod
    def _int64_owner_id(cls, value: str | None) -> str | None:
        if value is not None and int(value) > 2**63 - 1:
            raise ValueError("Study owner ID is out of range")
        return value

    @field_validator("source_snapshot")
    @classmethod
    def _distinct_sources(cls, sources: tuple[ResearchStudySource, ...]) -> tuple[ResearchStudySource, ...]:
        if (
            len({source.source_id for source in sources}) != len(sources)
            or len({source.catalog_item_id for source in sources}) != len(sources)
            or any(len(set(source.ready_modes)) != len(source.ready_modes) for source in sources)
            or not any(source.ready_modes for source in sources)
        ):
            raise ValueError("Choose distinct associations with at least one ready source")
        return sources


class _StudyEntry(_StudyModel):
    citations: tuple[StudyIdentifier, ...] = Field(default=(), max_length=20)


class _Flashcard(_StudyEntry):
    front: _StudyText = Field(repr=False)
    back: _StudyText = Field(repr=False)


class _QuizQuestion(_StudyEntry):
    question: _StudyText = Field(repr=False)
    options: tuple[_StudyText, ...] = Field(min_length=2, max_length=8, repr=False)
    correct_index: Annotated[int, Field(strict=True, ge=0)]
    explanation: _StudyText | None = Field(default=None, repr=False)

    @model_validator(mode="after")
    def _valid_answer(self) -> Self:
        if self.correct_index >= len(self.options) or len(set(self.options)) != len(self.options):
            raise ValueError("Choose one answer from distinct options")
        return self


class _StudyPublication(ResearchStudyGeneration):
    title: Annotated[
        str, StringConstraints(strict=True, strip_whitespace=True, max_length=120, pattern=r"^[^\x00-\x1f]*$")
    ] = Field(default="", repr=False)

    @model_validator(mode="after")
    def _citations_belong_to_ready_sources(self) -> Self:
        ready = {source.source_id for source in self.source_snapshot if source.ready_modes}
        if any(not ready.issuperset(item.citations) for item in self.items):
            raise ValueError("Each citation must refer to a ready source in the snapshot")
        return self


class ResearchFlashcardPublication(_StudyPublication):
    """Create a native deck and its generated cards."""

    output_kind: Literal["flashcards"]
    items: tuple[_Flashcard, ...] = Field(min_length=1, max_length=50, repr=False)


class ResearchQuizPublication(_StudyPublication):
    """Create a native quiz and its generated questions."""

    output_kind: Literal["quiz"]
    items: tuple[_QuizQuestion, ...] = Field(min_length=1, max_length=50, repr=False)


ResearchStudyPublication = Annotated[
    ResearchFlashcardPublication | ResearchQuizPublication, Field(discriminator="output_kind")
]


class ResearchStudyOutput(_StudyModel):
    """Current scoped native owner with its original generation provenance."""

    workspace_id: str
    owner_id: StudyOwnerId
    version: _Revision
    name: str
    created_at: str
    generation: ResearchStudyGeneration


class ResearchStudyOutputsPage(_StudyModel):
    """Bounded projection carrying the publication contract version."""

    contract: Literal["research-study.v1"] = "research-study.v1"
    items: list[ResearchStudyOutput]
    limit: int
    offset: int
    total: int
