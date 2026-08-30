import re
from enum import StrEnum
from typing import Annotated

from pydantic import AfterValidator, Field, field_validator, model_validator

from .canonical import VersionOne
from .enums import ProposalOperation
from .models import ProfileControls, SemanticKey
from .payloads import (
    FrozenModel,
    LegacyUnclassifiedPayload,
    ProfilePayload,
    reject_blank,
    reject_secret_material,
)


def _bounded(max_length: int):
    return Annotated[
        str, Field(min_length=1, max_length=max_length), AfterValidator(reject_blank)
    ]


InterviewId = _bounded(128)
InterviewTopic = _bounded(128)
QuestionText = _bounded(1_000)
AnswerText = _bounded(16_384)

_COMPOUND_CLAUSE = re.compile(
    r"\b(?:and|or)\s+(?:why|how|what|when|where|who|which|describe|explain|tell|"
    r"list|state|provide|share|identify|summarize|outline|give|your)\b",
    re.IGNORECASE,
)


class InterviewAudience(StrEnum):
    PERSONAL = "personal"
    WORKSPACE = "workspace"


class InterviewQuestion(FrozenModel):
    schema_version: VersionOne = 1
    question_id: InterviewId
    topic: InterviewTopic
    text: QuestionText

    @field_validator("text")
    @classmethod
    def validate_question(cls, value: str) -> str:
        reject_secret_material(value)
        if value.count("?") > 1 or ";" in value or _COMPOUND_CLAUSE.search(value):
            raise ValueError("question must ask one thing")
        return value


class InterviewTurn(FrozenModel):
    schema_version: VersionOne = 1
    question_id: InterviewId
    answer: AnswerText

    @field_validator("answer")
    @classmethod
    def validate_answer(cls, value: str) -> str:
        return reject_secret_material(value)


class InterviewPack(FrozenModel):
    schema_version: VersionOne = 1
    pack_id: InterviewId
    pack_version: VersionOne
    audience: InterviewAudience
    coverage_version: VersionOne
    coverage_topics: tuple[InterviewTopic, ...] = Field(min_length=1, max_length=32)
    questions: tuple[InterviewQuestion, ...] = Field(max_length=20)

    @model_validator(mode="after")
    def validate_pack(self):
        if len(set(self.coverage_topics)) != len(self.coverage_topics):
            raise ValueError("coverage topics must be unique")
        question_ids = [question.question_id for question in self.questions]
        if len(set(question_ids)) != len(question_ids):
            raise ValueError("question IDs must be unique")
        if any(
            question.topic not in self.coverage_topics for question in self.questions
        ):
            raise ValueError("question topic is not covered by the pack")
        return self


class InterviewProposedChange(FrozenModel):
    schema_version: VersionOne = 1
    operation: ProposalOperation
    target_record_id: InterviewId | None = None
    base_version_id: InterviewId | None = None
    proposed_payload: ProfilePayload | None = None
    controls: ProfileControls | None = None
    semantic_key: SemanticKey | None = None

    @model_validator(mode="after")
    def shape(self):
        expected = {
            ProposalOperation.CREATE: (False, False, True, True),
            ProposalOperation.UPDATE: (True, True, True, True),
            ProposalOperation.ARCHIVE: (True, True, False, False),
            ProposalOperation.PROMOTE: (True, True, False, False),
        }[self.operation]
        actual = (
            self.target_record_id is not None,
            self.base_version_id is not None,
            self.proposed_payload is not None,
            self.controls is not None,
        )
        if actual != expected:
            raise ValueError("invalid interview proposed change shape")
        if self.operation in (ProposalOperation.CREATE, ProposalOperation.UPDATE):
            is_legacy = isinstance(self.proposed_payload, LegacyUnclassifiedPayload)
            if (self.semantic_key is None) != is_legacy:
                raise ValueError("semantic key does not match payload kind")
        elif self.semantic_key is not None:
            raise ValueError("archive and promote changes cannot carry semantic keys")
        if self.proposed_payload is not None:
            reject_secret_material(str(self.proposed_payload.model_dump()))
        return self


class InterviewProposalBatch(FrozenModel):
    pack_id: InterviewId
    pack_version: VersionOne
    audience: InterviewAudience
    changes: tuple[InterviewProposedChange, ...] = Field(max_length=20)
