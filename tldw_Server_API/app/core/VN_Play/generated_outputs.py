"""Strict parsing for scripted VN generation model outputs."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

CHOICE_ID_PATTERN = r"^[a-zA-Z0-9_-]{1,80}$"
MAX_TEXT_LENGTH = 2000
MAX_SPEAKER_LENGTH = 128
MAX_METADATA_BYTES = 4096
MAX_VISUAL_LABELS_BYTES = 4096
DEFAULT_NARRATIVE_LINE_CAP = 12
DEFAULT_DIALOGUE_LINE_CAP = 24
DEFAULT_CHOICE_CAP = 8
DEFAULT_VISUAL_DIRECTIVE_CAP = 12
VISUAL_ASSET_TYPES = {"background", "sprite", "depth_companion", "cg"}

AttachedCharacterValidator = Callable[[str], bool]


class VNGenerationOutputParseError(ValueError):
    """Raised when a scripted generation output fails strict validation."""


@dataclass(frozen=True, slots=True)
class VNGenerationOutputParseResult:
    """Normalized public generation output after strict schema validation."""

    schema: str
    public_payload: dict[str, Any]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _NarrativeLine(_StrictModel):
    text: str = Field(min_length=1, max_length=MAX_TEXT_LENGTH)
    speaker: str | None = Field(default=None, max_length=MAX_SPEAKER_LENGTH)
    character_id: str | None = Field(default=None, min_length=1, max_length=128)

    @field_validator("text", "speaker", "character_id", mode="before")
    @classmethod
    def _strip_string(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip()
        return value


class _DialogueLine(_NarrativeLine):
    pass


class _Choice(_StrictModel):
    id: str = Field(pattern=CHOICE_ID_PATTERN)
    text: str = Field(min_length=1, max_length=MAX_TEXT_LENGTH)
    metadata: dict[str, Any] | None = None

    @field_validator("id", "text", mode="before")
    @classmethod
    def _strip_string(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("metadata")
    @classmethod
    def _metadata_cap(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        return _validate_json_object_cap(value, MAX_METADATA_BYTES, "metadata_too_large")


class _VisualDirective(_StrictModel):
    asset_type: Literal["background", "sprite", "depth_companion", "cg"]
    slot_key: str | None = Field(default=None, min_length=1, max_length=MAX_TEXT_LENGTH)
    labels: dict[str, Any] | None = None

    @field_validator("slot_key", mode="before")
    @classmethod
    def _strip_string(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("labels")
    @classmethod
    def _labels_cap(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        return _validate_json_object_cap(value, MAX_VISUAL_LABELS_BYTES, "visual_labels_too_large")


class _NarrativeDialogueOutput(_StrictModel):
    schema: Literal["narrative_dialogue"]
    narrative: list[_NarrativeLine] = Field(default_factory=list, max_length=DEFAULT_NARRATIVE_LINE_CAP)
    dialogue: list[_DialogueLine] = Field(default_factory=list, max_length=DEFAULT_DIALOGUE_LINE_CAP)

    @model_validator(mode="after")
    def _requires_content(self) -> "_NarrativeDialogueOutput":
        if not self.narrative and not self.dialogue:
            raise ValueError("empty_narrative_dialogue")
        return self


class _ChoiceSetOutput(_StrictModel):
    schema: Literal["choice_set"]
    lead_in: str | None = Field(default=None, max_length=MAX_TEXT_LENGTH)
    choices: list[_Choice] = Field(min_length=1, max_length=DEFAULT_CHOICE_CAP)

    @field_validator("lead_in", mode="before")
    @classmethod
    def _strip_string(cls, value: Any) -> Any:
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return value

    @model_validator(mode="after")
    def _unique_choice_ids(self) -> "_ChoiceSetOutput":
        choice_ids = [choice.id for choice in self.choices]
        if len(choice_ids) != len(set(choice_ids)):
            raise ValueError("duplicate_choice_id")
        return self


class _SceneUpdateOutput(_StrictModel):
    schema: Literal["scene_update"]
    narrative: list[_NarrativeLine] = Field(default_factory=list, max_length=DEFAULT_NARRATIVE_LINE_CAP)
    dialogue: list[_DialogueLine] = Field(default_factory=list, max_length=DEFAULT_DIALOGUE_LINE_CAP)
    visual_directives: list[_VisualDirective] = Field(
        default_factory=list,
        max_length=DEFAULT_VISUAL_DIRECTIVE_CAP,
    )


def parse_vn_generation_output(
    raw: Any,
    *,
    output_schema: str,
    attached_character_validator: AttachedCharacterValidator | None = None,
) -> VNGenerationOutputParseResult:
    """Parse model JSON into a bounded public payload for one generation schema."""
    payload = _load_payload(raw)
    if payload.get("schema") != output_schema:
        raise VNGenerationOutputParseError("schema_mismatch")

    try:
        if output_schema == "narrative_dialogue":
            parsed = _NarrativeDialogueOutput.model_validate(payload)
        elif output_schema == "choice_set":
            parsed = _ChoiceSetOutput.model_validate(payload)
        elif output_schema == "scene_update":
            parsed = _SceneUpdateOutput.model_validate(payload)
        else:
            raise VNGenerationOutputParseError("unsupported_generation_output_schema")
    except ValidationError as exc:
        raise VNGenerationOutputParseError(_validation_error_code(exc)) from exc

    public_payload = _public_payload(parsed)
    _validate_attached_characters(
        public_payload,
        attached_character_validator=attached_character_validator,
    )
    return VNGenerationOutputParseResult(schema=output_schema, public_payload=public_payload)


def _load_payload(raw: Any) -> dict[str, Any]:
    if isinstance(raw, Mapping):
        return dict(raw)
    if isinstance(raw, str):
        text = _strip_markdown_fences(raw.strip())
        try:
            loaded = json.loads(text)
        except json.JSONDecodeError as exc:
            raise VNGenerationOutputParseError("invalid_generation_json") from exc
        if isinstance(loaded, Mapping):
            return dict(loaded)
    raise VNGenerationOutputParseError("generation_output_must_be_object")


def _strip_markdown_fences(value: str) -> str:
    if not value.startswith("```"):
        return value
    lines = value.splitlines()
    if len(lines) >= 2 and lines[-1].strip() == "```":
        first = lines[0].strip()
        if first in {"```", "```json", "```JSON"}:
            return "\n".join(lines[1:-1]).strip()
    return value


def _validate_json_object_cap(
    value: dict[str, Any] | None,
    max_bytes: int,
    error_code: str,
) -> dict[str, Any] | None:
    if value is None:
        return None
    try:
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("metadata_must_be_json") from exc
    if len(encoded) > max_bytes:
        raise ValueError(error_code)
    return value


def _validation_error_code(exc: ValidationError) -> str:
    messages = " ".join(str(error.get("msg") or "") for error in exc.errors())
    if "empty_narrative_dialogue" in messages:
        return "empty_narrative_dialogue"
    if "metadata_too_large" in messages:
        return "metadata_too_large"
    if "visual_labels_too_large" in messages:
        return "visual_labels_too_large"
    return "invalid_generation_output"


def _public_payload(parsed: BaseModel) -> dict[str, Any]:
    data = parsed.model_dump(mode="json", exclude_none=True)
    if data.get("schema") == "choice_set":
        return data
    if data.get("schema") == "narrative_dialogue":
        data.setdefault("narrative", [])
        data.setdefault("dialogue", [])
        return data
    if data.get("schema") == "scene_update":
        data.setdefault("narrative", [])
        data.setdefault("dialogue", [])
        data.setdefault("visual_directives", [])
        return data
    raise VNGenerationOutputParseError("unsupported_generation_output_schema")


def _validate_attached_characters(
    public_payload: Mapping[str, Any],
    *,
    attached_character_validator: AttachedCharacterValidator | None,
) -> None:
    if attached_character_validator is None:
        return
    for section_name in ("narrative", "dialogue"):
        lines = public_payload.get(section_name)
        if not isinstance(lines, list):
            continue
        for line in lines:
            if not isinstance(line, Mapping):
                continue
            character_id = line.get("character_id")
            if isinstance(character_id, str) and not attached_character_validator(character_id):
                raise VNGenerationOutputParseError("character_not_attached")
