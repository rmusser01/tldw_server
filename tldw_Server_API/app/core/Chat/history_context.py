"""Project self-contained saved behavior from an owner-transaction resume read.

No live cards, profiles, presets, lore, memory, or assets are loaded here.
"""

from __future__ import annotations

import stat
from copy import deepcopy
from pathlib import Path
from typing import Any

from tldw_Server_API.app.core.Character_Chat.modules.character_prompt_presets import (
    ST_DEFAULT_PROMPT_PRESET,
    build_character_system_prompt,
    build_custom_system_prompt,
)
from tldw_Server_API.app.core.Chat.history_selection import HistorySelectionError

# These settings select already materialized values or carry identity/readiness
# metadata. Other saved prompt effects need a dedicated, faithful projector.
_SUPPORTED_SETTINGS = {
    "model",
    "provider",
    "schemaVersion",
    "updatedAt",
    "greetingsChecksum",
    "participantCharacterIds",
    "presetScope",
    "chatPresetOverrideId",
    "promptPreset",
    "prompt_preset",
    "roleplayResumeV1",
    "roleplayBehaviorV1",
    "chatGenerationOverride",
    "generationOverrides",
    "turnTakingMode",
    "greetingEnabled",
    "greetingScope",
    "greetingSelectionId",
    "useCharacterDefault",
    "memoryScope",
}


def _unsupported(effect: str) -> None:
    raise HistorySelectionError("unsupported_history_context_" + effect)


def project_history_context(
    state: dict[str, Any],
    *,
    character_override: str | None = None,
) -> tuple[dict[str, Any] | None, int | None, dict[str, Any]]:
    """Return detached neutral or saved single-character prompt and sampling data."""
    conversation = state["conversation"]
    settings = state.get("settings") or {}
    materialized = state.get("materialized_settings")
    if "roleplayBehaviorV1" in settings and materialized is None:
        _unsupported("invalid_materialized_behavior")
    if set(settings) - _SUPPORTED_SETTINGS:
        _unsupported("saved_settings")
    snapshot = state.get("behavior_snapshot") or {}
    if materialized is not None:
        base = materialized.get("values", {}).get("base_snapshot")
        expected_base = {
            "schema_version": snapshot.get("schema_version"),
            "digest": snapshot.get("digest"),
        }
        if snapshot.get("status") != "valid" or base != expected_base:
            _unsupported("materialized_binding")
    values = (materialized or {}).get("values", {})
    controls = values.get("behavior_controls", {})
    if set(controls.get("applied_overrides", {})) - _SUPPORTED_SETTINGS:
        _unsupported("behavior_controls")
    if (
        controls.get("prompt_context")
        or controls.get("pinned_message_ids")
        or controls.get("auto_summary", {}).get("enabled")
        or controls.get("auto_summary", {}).get("summary")
    ):
        _unsupported("behavior_controls")
    author_note = controls.get("author_note") or {}
    note_active = author_note.get("enabled", True) and not author_note.get("exclude_from_prompt", False)
    note_bookkeeping = {"enabled", "exclude_from_prompt", "gm_only", "position"}
    if note_active and any(value for key, value in author_note.items() if key not in note_bookkeeping):
        _unsupported("author_note")
    for effect in ("assistant_overlay", "world_books", "memory", "greeting"):
        if values.get(effect):
            _unsupported(effect)
    kind = conversation.get("assistant_kind")
    expected_id = conversation.get("character_id") or conversation.get("assistant_id")
    override = str(character_override or "").strip()
    if override and override != str(expected_id):
        raise HistorySelectionError("assistant_mismatch")
    if expected_id is None and kind is None:
        if (state.get("behavior_snapshot") or {}).get("status") != "missing":
            _unsupported("unbound_snapshot")
        return None, None, {"assistant_kind": None, "assistant_id": None}
    if kind == "persona":
        _unsupported("persona")
    snapshot = state.get("behavior_snapshot") or {}
    if snapshot.get("status") != "valid":
        _unsupported("snapshot_" + str(snapshot.get("status", "missing")))
    payload = snapshot["payload"]
    participants = values.get("participants", payload["participants"])
    turn_mode = controls.get("turn_taking_mode", payload["routing_defaults"].get("turn_taking_mode"))
    if len(participants) != 1 or turn_mode != "single":
        _unsupported("participants")
    participant = participants[0]
    source = participant["source"]
    if source["kind"] != "character" or source["id"] != str(expected_id):
        _unsupported("source_identity")
    if settings.get("participantCharacterIds", [int(expected_id)]) != [int(expected_id)]:
        _unsupported("participants")
    for effect in ("world_books", "exemplars", "default_memory"):
        if participant.get(effect):
            _unsupported(effect)
    defaults = participant.get("generation_defaults") or {}
    if set(defaults) - {"source", "sampling"}:
        _unsupported("generation_defaults")
    prompt = participant["prompt"]
    extensions = prompt["prompt_relevant_extensions"]
    if set(extensions) - {"prompt_preset", "character_extensions"}:
        _unsupported("prompt_extensions")
    card_extensions = extensions.get("character_extensions") or {}
    if set(card_extensions) - {"prompt_preset", "promptPreset", "tldw"} or set(card_extensions.get("tldw") or {}) - {
        "prompt_preset",
        "promptPreset",
    }:
        _unsupported("character_extensions")
    card = deepcopy({key: value for key, value in prompt.items() if key != "prompt_relevant_extensions"})
    card["name"] = participant["identity"]["name"]
    preset = values.get("prompt_preset", extensions.get("prompt_preset"))
    if preset is not None:
        if not isinstance(preset.get("section_order"), list) or not isinstance(preset.get("section_templates"), dict):
            _unsupported("prompt_preset")
        card["system_prompt"] = build_custom_system_prompt(
            card,
            card["name"],
            "User",
            preset["section_order"],
            preset["section_templates"],
        )
    else:
        card["system_prompt"] = build_character_system_prompt(
            card, card["name"], "User", preset=ST_DEFAULT_PROMPT_PRESET
        )
    effective = values.get("effective_completion", state.get("effective_completion"))
    if not effective and any(key in settings for key in ("chatGenerationOverride", "generationOverrides")):
        _unsupported("unbound_sampling")
    sampling = (effective or {}).get("sampling", defaults.get("sampling", {}))
    return (
        card,
        int(expected_id),
        {
            "assistant_kind": kind or "character",
            "assistant_id": str(expected_id),
            "history_sampling": deepcopy(sampling),
        },
    )


def require_history_skill_absence(base_path: Path, *, registry_may_be_visible: bool) -> None:
    """Fail closed on potentially model-visible skills, without syncing or mkdir.

    Match immediate, non-dot, non-symlink installed directories and regular
    SKILL.md files. Disabled skills cannot add context. Eligible registry rows
    and interrupted replacements cannot establish absence without mutation or
    an unfenced integrity decision, so they explicitly require another adapter.
    """
    from tldw_Server_API.app.core.Skills.exceptions import SkillParseError
    from tldw_Server_API.app.core.Skills.skill_parser import SkillParser
    from tldw_Server_API.app.core.Skills.skills_service import _read_regular_file_bytes_no_follow

    if registry_may_be_visible:
        _unsupported("skills_registry")
    directory = base_path / "skills"
    try:
        if not directory.exists():
            return
        if directory.is_symlink():
            _unsupported("skills_unresolved")
        for candidate in directory.iterdir():
            if candidate.name.startswith(".replacing-"):
                _unsupported("skills_recovery")
            if candidate.name.startswith(".") or not stat.S_ISDIR(candidate.lstat().st_mode):
                continue
            source = candidate / "SKILL.md"
            try:
                mode = source.lstat().st_mode
            except FileNotFoundError:
                continue
            if not stat.S_ISREG(mode):
                continue
            parsed = SkillParser().parse_content(
                _read_regular_file_bytes_no_follow(source).decode("utf-8"), default_name=candidate.name
            )
            if parsed.frontmatter.user_invocable and not parsed.frontmatter.disable_model_invocation:
                _unsupported("skills_installed")
    except (OSError, UnicodeError, SkillParseError) as exc:
        raise HistorySelectionError("unsupported_history_context_skills_unresolved") from exc
