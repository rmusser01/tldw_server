"""Project self-contained saved behavior from an owner-transaction resume read.

No live cards, profiles, presets, lore, memory, or assets are loaded here.
"""

from __future__ import annotations

import stat
from copy import deepcopy
from pathlib import Path
from typing import Any

from tldw_Server_API.app.core.Character_Chat.modules.character_generation_presets import (
    supports_saved_character_extensions,
)
from tldw_Server_API.app.core.Character_Chat.modules.character_prompt_presets import (
    DEFAULT_PROMPT_PRESET,
    ST_DEFAULT_PROMPT_PRESET,
    build_character_system_prompt,
    build_custom_system_prompt,
    build_saved_builtin_system_prompt,
)
from tldw_Server_API.app.core.Chat.history_selection import HistorySelectionError


def selected_unanswered_text_question(
    snapshot: Any, content: tuple[dict[str, Any], ...], selection: dict[str, Any],
) -> dict[str, Any]:
    """Require the exact canonical unanswered text input for a new answer intent."""
    cursor = selection["cursor"]
    question = content[-1] if content else None
    node = next((item for item in snapshot.nodes if question and item["id"] == question["id"]), None)
    if (selection["purpose"] != "send" or cursor["kind"] != "after_message"
            or question is None or cursor["message_id"] != question["id"]
            or node is None or node["role"] != "user"):
        raise HistorySelectionError("invalid_existing_input")
    if question["images"]:
        raise HistorySelectionError("unsupported_history_context_existing_input_assets")
    if any(item["role"] == "assistant" and item.get("parent_id") == question["id"] for item in snapshot.nodes):
        raise HistorySelectionError("existing_input_already_answered")
    return question


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
    "chat_dictionary_id",
    "chat_dictionary_ids",
    "conversationContext",
}


def _unsupported(effect: str) -> None:
    raise HistorySelectionError("unsupported_history_context_" + effect)


def render_saved_character_prompt(
    participant: dict[str, Any],
    values: dict[str, Any],
    *,
    user_name: str = "User",
) -> dict[str, Any]:
    """Render a detached saved character and saved preset for both history contracts."""
    prompt = participant["prompt"]
    card = deepcopy({key: value for key, value in prompt.items() if key != "prompt_relevant_extensions"})
    card["name"] = participant["identity"]["name"]
    preset = values.get(
        "prompt_preset",
        participant["prompt"]["prompt_relevant_extensions"].get("prompt_preset"),
    )
    if preset is not None:
        if not isinstance(preset.get("section_order"), list) or not isinstance(preset.get("section_templates"), dict):
            _unsupported("prompt_preset")
        source = preset.get("source", {})
        if preset.get("builtin") is True or source.get("kind") == "builtin_prompt_preset":
            if (
                preset.get("builtin") is not True
                or preset.get("preset_id") not in {DEFAULT_PROMPT_PRESET, ST_DEFAULT_PROMPT_PRESET}
                or source
                != {
                    "kind": "builtin_prompt_preset",
                    "id": preset["preset_id"],
                    "version": 1,
                }
            ):
                _unsupported("prompt_preset")
            card["system_prompt"] = build_saved_builtin_system_prompt(card, card["name"], user_name, preset)
        else:
            card["system_prompt"] = build_custom_system_prompt(
                card,
                card["name"],
                user_name,
                preset["section_order"],
                preset["section_templates"],
            )
    else:
        card["system_prompt"] = build_character_system_prompt(
            card, card["name"], user_name, preset=ST_DEFAULT_PROMPT_PRESET
        )
    return card


def project_history_context(
    state: dict[str, Any],
    *,
    character_override: str | None = None,
) -> tuple[dict[str, Any] | None, int | None, dict[str, Any]]:
    """Project saved owner state into detached prompt and sampling data.

    Args:
        state: Coherent conversation, settings, and behavior snapshot from the
            owner transaction.
        character_override: Optional character identity requested by the send.

    Returns:
        A detached character card (or ``None``), its database ID, and the
        assistant identity and saved sampling context.

    Raises:
        HistorySelectionError: Saved behavior cannot be represented faithfully
            or the requested character differs from the saved owner.
    """
    conversation = state["conversation"]
    settings = state.get("settings") or {}
    from tldw_Server_API.app.core.Character_Chat.frozen_dictionary_context import (
        assigned_dictionary_ids,
        validate_frozen_dictionary_plan,
        without_dictionary_assignment,
    )
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import InputError

    try:
        dictionary_ids = assigned_dictionary_ids(settings)
    except InputError as exc:
        raise HistorySelectionError("unsupported_history_context_dictionaries") from exc
    if "conversationContext" in without_dictionary_assignment(settings):
        _unsupported("saved_settings")
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
    dictionary_plan = None
    if dictionary_ids or "dictionaries" in values:
        try:
            dictionary_plan = validate_frozen_dictionary_plan(values.get("dictionaries"), dictionary_ids)
            catalog_owner = state.get("dictionary_catalog_owner")
            if catalog_owner is not None and (
                dictionary_plan.get("catalog_owner") != catalog_owner
                or conversation.get("client_id") != catalog_owner
            ):
                raise InputError("Saved dictionary plan lacks current catalog ownership.")
        except InputError as exc:
            raise HistorySelectionError("unsupported_history_context_dictionaries") from exc
    controls = values.get("behavior_controls", {})
    if set(controls.get("applied_overrides", {})) - _SUPPORTED_SETTINGS:
        _unsupported("behavior_controls")
    if (
        without_dictionary_assignment(controls.get("prompt_context") or {})
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
    for effect in ("assistant_overlay", "memory", "greeting"):
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
    for effect in ("exemplars", "default_memory"):
        if participant.get(effect):
            _unsupported(effect)
    from tldw_Server_API.app.core.Character_Chat.world_book_prompt_context import build_frozen_world_book_plan

    try:
        world_book_plan = build_frozen_world_book_plan(participant.get("world_books", []), values.get("world_books", []))
    except (ValueError, TypeError, KeyError) as exc:
        raise HistorySelectionError("unsupported_history_context_world_books") from exc
    defaults = participant.get("generation_defaults") or {}
    if set(defaults) - {"source", "sampling"}:
        _unsupported("generation_defaults")
    prompt = participant["prompt"]
    extensions = prompt["prompt_relevant_extensions"]
    if set(extensions) - {"prompt_preset", "character_extensions"}:
        _unsupported("prompt_extensions")
    card_extensions = extensions.get("character_extensions") or {}
    if not supports_saved_character_extensions(card_extensions):
        _unsupported("character_extensions")
    card = render_saved_character_prompt(participant, values)
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
            **({"history_dictionary_plan": dictionary_plan} if dictionary_plan is not None else {}),
            **({"history_world_book_plan": world_book_plan}
               if participant.get("world_books") or values.get("world_books") else {}),
        },
    )


def require_history_skill_absence(base_path: Path, *, registry_may_be_visible: bool) -> None:
    """Fail closed on potentially model-visible skills, without syncing or mkdir.

    Match immediate, non-dot, non-symlink installed directories and regular
    SKILL.md files. Disabled skills cannot add context. Eligible registry rows
    and interrupted replacements cannot establish absence without mutation or
    an unfenced integrity decision, so they explicitly require another adapter.

    Args:
        base_path: User directory whose installed skills could affect the prompt.
        registry_may_be_visible: Whether an eligible registry row may add a skill.

    Raises:
        HistorySelectionError: Skill absence cannot be established safely.
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
