"""Structured VN Play model output parsing."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from tldw_Server_API.app.core.VN_Play.constants import MODE_STORY
from tldw_Server_API.app.core.VN_Play.models import TurnResult


class VNPlayParseError(ValueError):
    """Raised when model output cannot be normalized into a VN Play turn."""


@dataclass(frozen=True, slots=True)
class DialogueLine:
    """Normalized dialogue line from model output."""

    speaker: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {"speaker": self.speaker, "text": self.text}
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True, slots=True)
class TurnChoice:
    """Normalized story/freeform choice from model output."""

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {"id": self.id, "text": self.text}
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(frozen=True, slots=True)
class NormalizedTurnResult:
    """Typed parser output before conversion to durable event payloads."""

    narration: str
    dialogue: list[DialogueLine] = field(default_factory=list)
    choices: list[TurnChoice] = field(default_factory=list)
    scene_updates: dict[str, Any] = field(default_factory=dict)
    visual_directives: list[dict[str, Any]] = field(default_factory=list)
    summary: str | None = None
    warnings: list[dict[str, Any]] = field(default_factory=list)

    def to_turn_result(self) -> TurnResult:
        return TurnResult(
            narrative_text=self.narration,
            dialogue=[line.to_dict() for line in self.dialogue],
            choices=[choice.to_dict() for choice in self.choices],
            scene_updates=dict(self.scene_updates),
            visual_directives=[dict(item) for item in self.visual_directives],
            warnings=[dict(item) for item in self.warnings],
        )


def parse_model_turn(raw: Any, *, mode: str) -> NormalizedTurnResult:
    """Parse provider output into a normalized VN Play turn result."""
    payload = _load_payload(raw)
    narration = _text_field(payload, "narration", "narrative_text", "text")
    if not narration:
        raise VNPlayParseError("narration_required")

    dialogue = _normalize_dialogue(payload.get("dialogue"))
    choices = _normalize_choices(payload.get("choices"))
    if mode == MODE_STORY and not 2 <= len(choices) <= 5:
        raise VNPlayParseError("story_choices_must_be_two_to_five")

    scene_updates = _normalize_scene_updates(payload)
    visual_directives = _normalize_list_of_dicts(
        payload.get("visual_directives") or payload.get("visuals")
    )
    warnings = _normalize_list_of_dicts(payload.get("warnings"))
    summary = _text_field(payload, "summary") or None

    return NormalizedTurnResult(
        narration=narration,
        dialogue=dialogue,
        choices=choices,
        scene_updates=scene_updates,
        visual_directives=visual_directives,
        summary=summary,
        warnings=warnings,
    )


def coerce_turn_result(raw: Any) -> TurnResult:
    """Convert parser or adapter output to the event-safe TurnResult shape."""
    if isinstance(raw, TurnResult):
        return raw
    if isinstance(raw, NormalizedTurnResult):
        return raw.to_turn_result()
    if isinstance(raw, Mapping):
        return parse_model_turn(raw, mode=str(raw.get("mode") or "freeform")).to_turn_result()
    raise VNPlayParseError("unsupported_turn_result")


def _load_payload(raw: Any) -> dict[str, Any]:
    if isinstance(raw, Mapping):
        return dict(raw)
    if isinstance(raw, str):
        stripped = _strip_markdown_fences(raw.strip())
        try:
            loaded = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise VNPlayParseError("invalid_model_json") from exc
        if isinstance(loaded, Mapping):
            return dict(loaded)
    raise VNPlayParseError("model_turn_must_be_object")


def _strip_markdown_fences(value: str) -> str:
    if not value.startswith("```"):
        return value
    lines = value.splitlines()
    if len(lines) >= 2 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return value


def _text_field(payload: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _normalize_dialogue(value: Any) -> list[DialogueLine]:
    lines: list[DialogueLine] = []
    for index, item in enumerate(_normalize_list_of_dicts(value), start=1):
        text = _text_field(item, "text", "line", "content")
        if not text:
            continue
        speaker = _text_field(item, "speaker", "name") or "Narrator"
        metadata = {
            key: val
            for key, val in item.items()
            if key not in {"speaker", "name", "text", "line", "content"}
        }
        metadata.setdefault("index", index)
        lines.append(DialogueLine(speaker=speaker, text=text, metadata=metadata))
    return lines


def _normalize_choices(value: Any) -> list[TurnChoice]:
    choices: list[TurnChoice] = []
    for index, item in enumerate(_normalize_list_of_dicts(value), start=1):
        text = _text_field(item, "text", "label", "title")
        if not text:
            continue
        choice_id = _text_field(item, "id", "key") or f"choice-{index}"
        metadata = {
            key: val
            for key, val in item.items()
            if key not in {"id", "key", "text", "label", "title"}
        }
        choices.append(TurnChoice(id=choice_id, text=text, metadata=metadata))
    return choices


def _normalize_scene_updates(payload: Mapping[str, Any]) -> dict[str, Any]:
    updates = {}
    raw_updates = payload.get("scene_updates")
    if isinstance(raw_updates, Mapping):
        updates.update(dict(raw_updates))

    directives = payload.get("scene_directives")
    if isinstance(directives, Mapping):
        _apply_background_directive(updates, directives.get("background"))
        for key in ("mood", "time_of_day", "weather"):
            value = directives.get(key)
            if isinstance(value, str) and value.strip():
                updates[key] = value.strip()
    return updates


def _apply_background_directive(updates: dict[str, Any], value: Any) -> None:
    if not isinstance(value, Mapping):
        return
    labels = value.get("labels")
    if isinstance(labels, Mapping):
        location = labels.get("location") or labels.get("location_key")
        if isinstance(location, str) and location.strip():
            updates["location_key"] = location.strip()
    for source_key, target_key in (
        ("location_key", "location_key"),
        ("background_item_id", "current_background_item_id"),
        ("depth_item_id", "current_depth_item_id"),
    ):
        if source_key in value:
            updates[target_key] = value[source_key]


def _normalize_list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]
