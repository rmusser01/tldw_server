import re
from typing import Annotated, Literal, Union

from pydantic import AfterValidator, BaseModel, ConfigDict, Field

from .canonical import VersionOne


def reject_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("value must not be blank")
    return value


_SECRET_MATERIAL_PATTERNS = (
    re.compile(r"-----BEGIN(?: [A-Z0-9]+)? PRIVATE KEY-----", re.IGNORECASE),
    re.compile(r"\b(?:sk-|gh[pousr]_|xox[baprs]-)[A-Za-z0-9_-]{20,}\b", re.IGNORECASE),
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{20,}\b", re.IGNORECASE),
    re.compile(
        r"\btoken\s*(?:=|:)\s*"
        r"(?!\d+(?=[^A-Za-z0-9._~+/=-]|$))[A-Za-z0-9._~+/=-]{6,}\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:password|private[_ -]?key|api[_ -]?key|access[_ -]?token|credentials?)"
        r"\s*(?:is|=|:)\s*\S{6,}",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:what(?:\s+is|'s|’s)|enter|provide|tell\s+me|share|may\s+i\s+have)"
        r"\s+(?:your|my)\s+"
        r"(?:password|private[_ -]?key|api[_ -]?key|access[_ -]?token)\b",
        re.IGNORECASE,
    ),
)


def reject_secret_material(value: str) -> str:
    if any(pattern.search(value) for pattern in _SECRET_MATERIAL_PATTERNS):
        raise ValueError("recognized secret material is not allowed")
    return value


BoundedText = Annotated[
    str, Field(min_length=1, max_length=16_384), AfterValidator(reject_blank)
]


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class IdentityPayload(FrozenModel):
    schema_version: VersionOne = 1
    kind: Literal["identity"] = "identity"
    subject: BoundedText
    value: BoundedText


class PreferencePayload(FrozenModel):
    schema_version: VersionOne = 1
    kind: Literal["preference"] = "preference"
    subject: BoundedText
    polarity: Literal["like", "dislike"]
    value: BoundedText


class RelationshipPayload(FrozenModel):
    schema_version: VersionOne = 1
    kind: Literal["relationship"] = "relationship"
    subject: BoundedText
    value: BoundedText


class CorrectionPayload(FrozenModel):
    schema_version: VersionOne = 1
    kind: Literal["correction"] = "correction"
    subject: BoundedText
    value: BoundedText


class ConstraintPayload(FrozenModel):
    schema_version: VersionOne = 1
    kind: Literal["constraint"] = "constraint"
    subject: BoundedText
    value: BoundedText


class GoalPayload(FrozenModel):
    schema_version: VersionOne = 1
    kind: Literal["goal"] = "goal"
    subject: BoundedText
    outcome: BoundedText


class ConventionPayload(FrozenModel):
    schema_version: VersionOne = 1
    kind: Literal["convention"] = "convention"
    subject: BoundedText
    value: BoundedText


class WorkingContextPayload(FrozenModel):
    schema_version: VersionOne = 1
    kind: Literal["working_context"] = "working_context"
    subject: BoundedText
    value: BoundedText


class LegacyUnclassifiedPayload(FrozenModel):
    schema_version: VersionOne = 1
    kind: Literal["legacy_unclassified"] = "legacy_unclassified"
    text: BoundedText


ProfilePayload = Annotated[
    Union[
        IdentityPayload,
        PreferencePayload,
        RelationshipPayload,
        CorrectionPayload,
        ConstraintPayload,
        GoalPayload,
        ConventionPayload,
        WorkingContextPayload,
        LegacyUnclassifiedPayload,
    ],
    Field(discriminator="kind"),
]
