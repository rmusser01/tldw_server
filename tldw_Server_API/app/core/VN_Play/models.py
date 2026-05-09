"""Internal models for VN Play runtime orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class SceneState:
    """Derived scene state used by replay and turn orchestration."""

    current_background_item_id: int | None = None
    current_depth_item_id: int | None = None
    active_sprite_items: list[dict[str, Any]] = field(default_factory=list)
    location_key: str | None = None
    mood: str | None = None
    time_of_day: str | None = None
    weather: str | None = None
    active_branch_node_id: int | None = None
    visible_choices: list[dict[str, Any]] = field(default_factory=list)
    transcript_cursor: int | None = None
    scene_version: int = 0
    warnings: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class GateResult:
    """Result of a runtime admission gate."""

    allowed: bool
    warnings: tuple[dict[str, Any], ...] = ()
    error_code: str | None = None
    error_message: str | None = None


@dataclass(frozen=True, slots=True)
class CharacterSafetyResult:
    """Character metadata safety result for VN Play admission."""

    allowed: bool
    status: str
    warning_code: str | None = None
    error_code: str | None = None
    message: str | None = None


@dataclass(frozen=True, slots=True)
class ResolvedAsset:
    """Approved VN asset selected for a runtime visual directive."""

    item_id: int
    slot_id: int
    slot_key: str
    asset_type: str
    variant_index: int
    storage_ref: str | None = None
    generated_file_id: int | None = None
    file_artifact_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class VisualDirectiveResolution:
    """Result of resolving a visual directive against an approved asset manifest."""

    applied: bool
    item: dict[str, Any] | None = None
    reason: str | None = None
    directive: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TurnResult:
    """Normalized model/runtime turn result before events are appended."""

    narrative_text: str
    dialogue: list[dict[str, Any]] = field(default_factory=list)
    choices: list[dict[str, Any]] = field(default_factory=list)
    scene_updates: dict[str, Any] = field(default_factory=dict)
    visual_directives: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[dict[str, Any]] = field(default_factory=list)
