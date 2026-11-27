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
from functools import partial
import asyncio
import base64
import contextlib
import json as _json
import os
import re
import uuid as _uuid


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


async def build_context_and_messages(
    chat_db: Any,
    request_data: Any,
    loop: Any,
    metrics: Any,
    default_save_to_db: bool,
    final_conversation_id: str | None,
    save_message_fn: Any,
    runtime_state: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], int | None, str, bool, list[dict[str, Any]], bool]:
    """Resolve character/conversation context, load history, save current messages, and return LLM-ready payload.

    Returns (character_card, character_db_id, final_conversation_id, conversation_created, llm_payload_messages, should_persist)
    """
    request_metadata = getattr(request_data, "metadata", None)
    explicit_failed_retry = isinstance(request_metadata, dict) and request_metadata.get("tldw_retry_failed_turn") is True
    regenerate_reply_id = request_metadata.get("tldw_regenerate_from_message_id") if isinstance(request_metadata, dict) else None
    if regenerate_reply_id is not None:
        if not isinstance(regenerate_reply_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", regenerate_reply_id):
            # 400: the chat endpoint turns statuses outside its allow-list (422) into 500.
            raise cs.HTTPException(status_code=400, detail="A valid saved reply ID is required for regeneration.")
        if (not final_conversation_id or explicit_failed_retry
                or getattr(request_data, "tldw_continuation", None) is not None):
            raise cs.HTTPException(status_code=409, detail="Regeneration requires an existing saved answered turn.")
    from tldw_Server_API.app.core.Chat.history_continuation import (
        continued_reply_prefix,
        parse_continue_reply_id,
    )

    continue_reply_id = parse_continue_reply_id(
        request_data,
        conversation_id=final_conversation_id,
        regenerate_reply_id=regenerate_reply_id,
        explicit_failed_retry=explicit_failed_retry,
    )
    continue_prefix: str | None = None
    reuse_saved_turn = explicit_failed_retry or regenerate_reply_id is not None

    server_existing_input = getattr(request_data, "tldw_history_server_existing_input_v1", None)
    history_selection = getattr(request_data, "tldw_history_selection_v1", None) or server_existing_input
    if history_selection is not None and (explicit_failed_retry or (server_existing_input is not None and (regenerate_reply_id is not None or continue_reply_id is not None))):
        raise cs.HTTPException(409, detail={"status": "unsupported_history_capability", "code": "unsupported_history_legacy_retry"})
    versioned = history_selection is not None
    history_runtime = runtime_state or {}
    if versioned and (not final_conversation_id or not history_runtime.get("history_owner_key")):
        raise cs.HTTPException(409, detail={"code": "missing_authenticated_history_binding"})
    selected_content = ()
    history_snapshot = None
    admission = None
    if versioned:
        from tldw_Server_API.app.core.Chat.history_context import (
            project_history_context,
            selected_unanswered_text_question,
        )

        if getattr(request_data, "prompt_template_name", None) not in (None, "", cs.DEFAULT_RAW_PASSTHROUGH_TEMPLATE.name):
            raise cs.HTTPException(409, detail={"status": "unsupported_history_capability", "code": "unsupported_history_context_prompt_template"})
        selection = history_selection.model_dump(mode="json")
        edit_target = getattr(request_data, "tldw_history_edit_v1", None)
        edit_target = edit_target.model_dump(mode="json") if edit_target is not None else None
        owner = history_runtime["history_owner_key"]
        continuation_holder: dict[str, str] = {}

        def accept_history() -> tuple[
            cs.HistorySelectionSnapshotV1,
            tuple[dict[str, Any], ...],
            dict[str, Any] | None,
            dict[str, Any],
            tuple[dict[str, Any] | None, int | None, dict[str, Any]],
        ]:
            with chat_db.transaction() as conn:
                state = chat_db.conversation_resume_store.get_roleplay_resume_state(
                    final_conversation_id,
                    conn=conn,
                    lock_for_update=True,
                    owner_client_id=history_runtime["history_owner_client_id"],
                )
                snap, content = chat_db.validate_history_selection(
                    final_conversation_id,
                    selection,
                    owner_client_id=history_runtime["history_owner_client_id"],
                    owner_key=owner,
                    conn=conn,
                    history_edit=edit_target,
                )
                question = (selected_unanswered_text_question(snap, content, selection)
                    if server_existing_input is not None else None)
                context = project_history_context(state, character_override=getattr(request_data, "character_id", None))
                dictionary_plan = context[2].pop("history_dictionary_plan", None)
                if dictionary_plan is not None:
                    from tldw_Server_API.app.core.Character_Chat.frozen_dictionary_context import (
                        transform_frozen_user_content,
                    )
                    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import InputError

                    try:
                        context[2]["history_dictionary_user_content"] = transform_frozen_user_content(
                            ([{"role": "user", "content": question["message"]}] if question is not None
                                else [message.model_dump(exclude_none=True) for message in request_data.messages]), dictionary_plan,
                        )
                    except InputError as exc:
                        raise cs.HistorySelectionError("unsupported_history_context_dictionaries") from exc
                cs._format_selected_history_messages(snap, content, context[0] or {})
                reused_reply_id = regenerate_reply_id if regenerate_reply_id is not None else continue_reply_id
                unsupported_code = (
                    "unsupported_history_continuation" if continue_reply_id is not None
                    else "unsupported_history_regeneration"
                )
                if server_existing_input is not None:
                    accepted = None
                elif reused_reply_id is not None:
                    if getattr(request_data, "tldw_history_branch", None) is not True:
                        raise cs.reply_action_refusal(unsupported_code, cs.REPLY_ACTION_REFUSED_REQUEST)
                    inputs = [
                        message.model_dump(exclude_none=True)
                        for message in request_data.messages
                        if message.role != "system"
                    ]
                    if len(inputs) != 1 or inputs[0].get("role") != "user":
                        raise cs.HistorySelectionError("invalid_regeneration_input")
                    try:
                        accepted = chat_db.reuse_selected_history_admission(
                            final_conversation_id,
                            selection,
                            reused_reply_id,
                            owner_client_id=history_runtime["history_owner_client_id"],
                            owner_key=owner,
                            conn=conn,
                        )
                    except cs.HistorySelectionError as exc:
                        if exc.code == "unsupported_history_regeneration":
                            raise cs.HistorySelectionError(unsupported_code, details=exc.details) from exc
                        raise
                    original = next(item for item in content if item["id"] == accepted["input_message_id"])
                    extra = original.get("extra_metadata") or {}
                    if (
                        original.get("tool_calls") is not None
                        or extra.get("function_call") is not None
                        or extra.get("tool_call_id") is not None
                    ):
                        raise cs.reply_action_refusal(unsupported_code, cs.REPLY_ACTION_REFUSED_QUESTION)
                    text = cs._saved_image_text(
                        {
                            "sender": "user",
                            "version": original["_message_version"],
                            "content": original["message"],
                            "images": original["images"],
                        },
                        extra,
                    )
                    details = cs._saved_image_details(extra, len(original["images"]))
                    parts = ([{"type": "text", "text": text}] if text else []) + [
                        {"type": "image_url", "image_url": {"url": image, "detail": detail}}
                        for image, detail in zip(original["images"], details)
                    ]
                    if (
                        cs._retry_content_components(inputs[0].get("content")) != cs._retry_content_components(parts)
                        or any(
                            inputs[0].get(key) is not None for key in ("tool_calls", "function_call", "tool_call_id")
                        )
                        or inputs[0].get("name") not in (None, extra.get("sender_name"))
                    ):
                        raise cs.HistorySelectionError("regeneration_input_changed")
                    if continue_reply_id is not None:
                        # Continue keeps the saved reply as the final context message.
                        continuation_holder["prefix"] = continued_reply_prefix(
                            next(item for item in content if item["id"] == continue_reply_id)
                        )
                    else:
                        content = tuple(item for item in content if item["id"] != regenerate_reply_id)
                else:
                    accepted = chat_db.append_selected_history_inputs(
                        final_conversation_id,
                        selection,
                        history_runtime["history_inputs"],
                        owner_client_id=history_runtime["history_owner_client_id"],
                        owner_key=owner,
                        conn=conn,
                        history_branch=getattr(request_data, "tldw_history_branch", None),
                        history_edit=edit_target,
                    )
                return snap, content, accepted, state["conversation"], context

        try:
            history_snapshot, selected_content, admission, existing_conversation, context = await asyncio.to_thread(accept_history)
        except cs.HistorySelectionError as exc:
            failure = (
                "unsupported_history_capability"
                if (exc.code.startswith("unsupported_history_context")
                    or exc.code in {"unsupported_history_regeneration", "unsupported_history_continuation"})
                else "stale_selection"
            )
            raise cs.HTTPException(409, detail={"status": failure, "code": exc.code, **exc.details}) from exc
        continue_prefix = continuation_holder.get("prefix")
        character_card, character_db_id, assistant_context = context
        for field, value in assistant_context.pop("history_sampling", {}).items():
            if field not in request_data.model_fields_set:
                setattr(request_data, field, value)
    else:
        # Assistant context
        (
            character_card,
            character_db_id,
            existing_conversation,
            assistant_context,
        ) = await cs._resolve_assistant_context_for_chat(
            chat_db=chat_db,
            request_data=request_data,
            loop=loop,
            conversation_id=final_conversation_id,
        )
    if character_card:
        cs.logger.debug(
            "Loaded assistant context name={} system_prompt_summary={}",
            character_card.get("name"),
            cs.text_summary(character_card.get("system_prompt")),
        )

    if character_card and character_db_id is not None:
        with contextlib.suppress(cs._CHAT_NONCRITICAL_EXCEPTIONS):
            metrics.track_character_access(character_id=str(request_data.character_id or "default"), cache_hit=False)

    if not character_card:
        cs.logger.warning("No character context found; proceeding with ephemeral default context.")
        character_card = {"name": cs.DEFAULT_CHARACTER_NAME, "system_prompt": cs._DEFAULT_ASSISTANT_SYSTEM_PROMPT}
        character_db_id = None
        assistant_context = {
            "assistant_kind": None,
            "assistant_id": None,
            "persona_memory_mode": assistant_context.get("persona_memory_mode"),
        }

    # Persistence decision
    requested = getattr(request_data, "save_to_db", None)
    should_persist: bool = bool(requested) if (requested is not None) else bool(default_save_to_db)

    # Conversation resolution
    client_id_from_db = getattr(chat_db, "client_id", None)
    conversation_created = False
    conv_id = final_conversation_id
    is_existing_persona_conversation = bool(
        existing_conversation
        and assistant_context.get("assistant_kind") == "persona"
        and assistant_context.get("assistant_id")
    )
    # Workspace chats can intentionally have no saved assistant identity. An
    # implicit default character supplies prompt context, not a request to fork.
    is_owned_neutral_conversation = bool(
        existing_conversation
        and client_id_from_db is not None
        and existing_conversation.get("client_id") == client_id_from_db
        and getattr(request_data, "character_id", None) is None
        and existing_conversation.get("character_id") is None
        and existing_conversation.get("assistant_kind") is None
        and existing_conversation.get("assistant_id") is None
    )
    from tldw_Server_API.app.core.Buddy.publication import current_buddy_publication

    publication = current_buddy_publication.get()
    # A Buddy acceptance already captured one exact existing conversation. Its
    # transaction fence rechecks owner/revision; default-character resolution
    # must not fork a neutral conversation into a new target.
    is_accepted_buddy_conversation = bool(
        publication is not None
        and existing_conversation
        and conv_id == publication.turn["conversation_id"]
        and str(client_id_from_db) == publication.repository.user_id
    )
    # Ensure a valid assistant identity is present before attempting persistence
    if should_persist and character_db_id is None and not (
        is_existing_persona_conversation or is_accepted_buddy_conversation or is_owned_neutral_conversation or versioned
    ):
        cs.logger.warning(
            'Persistence requested but no compatible assistant identity is available; disabling persistence for conversation {}.',
            final_conversation_id or "<new>",
        )
        should_persist = False

    if regenerate_reply_id or continue_reply_id:
        if (not should_persist or not existing_conversation
                or existing_conversation.get("client_id") != client_id_from_db
                or not (is_existing_persona_conversation or is_accepted_buddy_conversation or is_owned_neutral_conversation or versioned
                        or existing_conversation.get("character_id") == character_db_id)):
            raise cs.HTTPException(status_code=409, detail="Regeneration requires the existing saved assistant and conversation.")
        # This operation can only reuse its existing target; it must never fork.
    elif should_persist:
        if (is_existing_persona_conversation or is_accepted_buddy_conversation or is_owned_neutral_conversation or versioned) and conv_id:
            conversation_created = False
        else:
            conv_id, conversation_created = await cs.get_or_create_conversation(
                chat_db,
                conv_id,
                character_db_id,
                character_card.get("name", "Chat"),
                client_id_from_db,
                loop,
                first_user_message=cs.first_user_message_text(request_data.messages),
            )
    else:
        if not conv_id:
            conv_id = str(_uuid.uuid4())
        conversation_created = False

    if conv_id:
        with contextlib.suppress(cs._CHAT_NONCRITICAL_EXCEPTIONS):
            metrics.track_conversation(conv_id, conversation_created)
    if not conv_id:
        raise cs.HTTPException(status_code=cs.status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to establish conversation context.")

    continuation_spec = cs._extract_tldw_continuation_spec(request_data)
    continuation_metadata: dict[str, Any] | None = None
    assistant_parent_message_id: str | None = None
    assistant_prefill: str | None = None
    if continuation_spec and not versioned:
        assistant_parent_message_id, _mode, assistant_prefill = continuation_spec
        if conversation_created:
            raise cs.HTTPException(
                status_code=cs.status.HTTP_404_NOT_FOUND,
                detail="conversation_id is required for continuation and must reference an existing conversation.",
            )

    # History loading (configurable limit/order; filter missing roles, normalize assistant names)
    requested_history_limit = getattr(request_data, "history_message_limit", None)
    if requested_history_limit is None:
        history_limit = cs.DEFAULT_HISTORY_MESSAGE_LIMIT
    else:
        try:
            history_limit = int(requested_history_limit)
        except cs._CHAT_NONCRITICAL_EXCEPTIONS:
            history_limit = cs.DEFAULT_HISTORY_MESSAGE_LIMIT
        history_limit = max(0, min(cs._MAX_HISTORY_MESSAGES, history_limit))

    requested_history_order = getattr(request_data, "history_message_order", None)
    if requested_history_order:
        history_order = str(requested_history_order).strip().lower()
        if history_order not in {"asc", "desc"}:
            history_order = cs.DEFAULT_HISTORY_MESSAGE_ORDER
    else:
        history_order = cs.DEFAULT_HISTORY_MESSAGE_ORDER
    db_order = "ASC" if history_order == "asc" else "DESC"

    historical_msgs: list[dict[str, Any]] = []
    historical_ids: list[str | None] = []
    retry_history_content: dict[int, Any] = {}
    raw_hist: list[dict[str, Any]] = []
    if conv_id and (not conversation_created) and not versioned:
        if continuation_spec:
            from_message_id, mode, assistant_prefill = continuation_spec
            resolved_hist, continuation_metadata = await cs._resolve_tldw_continuation_history(
                chat_db=chat_db,
                loop=loop,
                conversation_id=conv_id,
                from_message_id=from_message_id,
                mode=mode,
                history_limit=(history_limit if history_limit > 0 else cs._MAX_HISTORY_MESSAGES),
                history_order=history_order,
            )
            if history_limit > 0:
                raw_hist = resolved_hist
            continuation_metadata["anchor_message_id"] = from_message_id
        elif history_limit > 0:
            raw_hist = await asyncio.to_thread(partial(chat_db.get_messages_for_conversation,
                    conv_id, history_limit, 0, db_order,
                    **({"strict_images": True} if reuse_saved_turn else {})),
            )
        for db_msg in raw_hist:
            sender_val = str(db_msg.get("sender", "") or "")
            metadata = None
            try:
                metadata = await asyncio.to_thread(chat_db.get_message_metadata, db_msg.get("id"))
            except cs._CHAT_NONCRITICAL_EXCEPTIONS as meta_err:
                cs.logger.debug("Metadata lookup failed for message {}: {}", db_msg.get("id"), meta_err)

            tool_calls_meta = metadata.get("tool_calls") if metadata else None
            extra_meta = metadata.get("extra") if metadata else None
            if not isinstance(extra_meta, dict):
                extra_meta = {}
            function_call_meta = extra_meta.get("function_call")
            content_placeholder_reason = extra_meta.get("content_placeholder_reason")
            tool_call_id_meta = extra_meta.get("tool_call_id")
            stored_role = extra_meta.get("sender_role")
            stored_name = extra_meta.get("sender_name")

            role_initial = cs.map_sender_to_role(sender_val, character_card.get("name") if character_card else None)
            role = stored_role if stored_role in {"user", "assistant", "system", "tool"} else role_initial
            if not cs.should_persist_message_role(role):
                continue
            char_name_hist = character_card.get("name", "Char") if character_card else "Char"
            text_content = cs._saved_image_text(db_msg, metadata.get("extra") if isinstance(metadata, dict) else None)
            if role == "assistant":
                text_content = cs.strip_reasoning_blocks(text_content)
            raw_user_text = text_content
            # Normal/persona Retry compares and sends the literal saved user input.
            preserve_literal_retry = reuse_saved_turn and role == "user" and (is_owned_neutral_conversation or is_existing_persona_conversation)
            if text_content and role != "tool" and not preserve_literal_retry:
                text_content = cs.replace_placeholders(text_content, char_name_hist, "User")
            msg_parts = []
            if text_content:
                msg_parts.append({"type": "text", "text": text_content})
            raw_images = db_msg.get("images") or []
            if (not raw_images) and db_msg.get("image_data") and db_msg.get("image_mime_type"):
                raw_images = [{
                    "position": 0,
                    "image_data": db_msg.get("image_data"),
                    "image_mime_type": db_msg.get("image_mime_type"),
                }]

            image_details = cs._saved_image_details(metadata.get("extra") if isinstance(metadata, dict) else None, len(raw_images))
            for image_entry, image_detail in zip(raw_images, image_details):
                try:
                    img_bytes = image_entry.get("image_data")
                    if isinstance(img_bytes, memoryview):
                        img_bytes = img_bytes.tobytes()
                    if not img_bytes:
                        if reuse_saved_turn:
                            raise cs.HTTPException(status_code=409, detail="A saved chat attachment is incomplete. Reload after repairing the source message.")
                        continue
                    img_mime = image_entry.get("image_mime_type") or db_msg.get("image_mime_type") or "image/png"
                    b64_img = await loop.run_in_executor(None, base64.b64encode, img_bytes)
                    image_part = {
                        "type": "image_url",
                        "image_url": {"url": f"data:{img_mime};base64,{b64_img.decode('utf-8')}", "detail": image_detail}
                    }
                    if reuse_saved_turn:
                        cs._retry_content_components([image_part])
                    msg_parts.append(image_part)
                except cs._CHAT_NONCRITICAL_EXCEPTIONS as e:
                    if reuse_saved_turn:
                        raise cs.HTTPException(status_code=409, detail="A saved chat attachment could not be read completely.") from e
                    cs.logger.warning(
                        "Error encoding DB image for history msg_id={} error={}",
                        db_msg.get("id"),
                        cs.exception_summary(e),
                    )
            if msg_parts or (role == "assistant" and (tool_calls_meta or function_call_meta)):
                hist_entry = {"role": role}
                if role == "tool" and len(msg_parts) == 1 and msg_parts[0].get("type") == "text":
                    hist_entry["content"] = msg_parts[0].get("text", "")
                else:
                    hist_entry["content"] = msg_parts or None
                cs._attach_saved_moonshot_reasoning(hist_entry, extra_meta)
                if role == "assistant":
                    name_source = stored_name or (character_card.get("name") if character_card else None)
                    if name_source:
                        name = cs.sanitize_sender_name(name_source)
                        if name:
                            hist_entry["name"] = name

                if tool_calls_meta is not None:
                    hist_entry["tool_calls"] = tool_calls_meta
                if function_call_meta and not hist_entry.get("tool_calls"):
                    hist_entry["function_call"] = function_call_meta
                if content_placeholder_reason in {"tool_calls", "function_call"} or (
                    content_placeholder_reason == "empty_tool_result"
                    and role == "tool"
                    and db_msg.get("version") == 1
                    and db_msg.get("content") == "[empty tool result]"
                ):
                    hist_entry["content"] = ""
                if role == "tool" and tool_call_id_meta:
                    hist_entry["tool_call_id"] = tool_call_id_meta
                if role == "tool" and not tool_call_id_meta:
                    cs.logger.debug(
                        "Skipping tool message {} without tool_call_id metadata.",
                        db_msg.get("id"),
                    )
                    continue
                if reuse_saved_turn and role == "user":
                    retry_history_content[id(hist_entry)] = ([{"type": "text", "text": raw_user_text}] if raw_user_text else []) + [part for part in msg_parts if part.get("type") == "image_url"]
                historical_msgs.append(hist_entry)
                historical_ids.append(db_msg.get("id"))
        cs.logger.info(f"Loaded {len(historical_msgs)} historical messages for conv_id '{conv_id}'.")

    # Keep failure diagnostics in saved history, but never send them as model
    # replies. Only a matching retry of the trailing failure can reuse its user
    # turn; ordinary repeated user messages retain their existing semantics.
    history_ended_with_error = bool(
        historical_msgs
        and cs._is_saved_chat_error_envelope(historical_msgs[0 if history_order == "desc" else -1])
    )
    visible_history = [(message, message_id) for message, message_id in zip(historical_msgs, historical_ids)
                       if not cs._is_saved_chat_error_envelope(message)]
    historical_msgs = [message for message, _ in visible_history]
    historical_ids = [message_id for _, message_id in visible_history]

    # Process current turn messages (persist if needed)
    request_messages: list[dict[str, Any]] = []
    for msg_model in request_data.messages:
        if not cs.should_persist_message_role(msg_model.role):
            continue
        msg_dict = msg_model.model_dump(exclude_none=True)
        raw_name = msg_dict.get("name")
        if isinstance(raw_name, str) and raw_name.strip():
            sanitized = cs.sanitize_sender_name(raw_name)
            if sanitized:
                msg_dict["name"] = sanitized
            else:
                msg_dict.pop("name", None)
        request_messages.append(msg_dict)

    metadata = getattr(request_data, "metadata", None)
    client_message_id = metadata.get("tldw_client_message_id") if isinstance(metadata, dict) else None
    if not isinstance(client_message_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", client_message_id):
        client_message_id = None
    explicit_failed_retry = isinstance(metadata, dict) and metadata.get("tldw_retry_failed_turn") is True
    retry_user_message_id: str | None = None
    saved_retry_system_messages: list[dict[str, Any]] = []
    legacy_regenerated_reply: dict[str, Any] | None = None
    if regenerate_reply_id and not versioned:
        if not request_messages:
            raise cs.HTTPException(status_code=409, detail="Regeneration requires the saved user turn.")
        if any(message.get("role") != "system" for message in request_messages[:-1]):
            raise cs.HTTPException(status_code=409, detail="Regeneration uses saved history. Send only the original user turn and current system instructions.")
        # System instructions still participate in prompt construction through
        # request_data. Neither they nor a supplied history prefix are new turns.
        request_messages = request_messages[-1:]
        regenerated_user, legacy_regenerated_reply = await cs._resolve_saved_regeneration(
            chat_db, conv_id, regenerate_reply_id, request_messages[-1],
        )
        retry_user_message_id = str(regenerated_user["id"])
        # Prior variants are alternatives to the answer being generated, not
        # preceding conversation context. Retain earlier turns unchanged.
        variant_ids = {row["id"] for row in raw_hist if
                       row.get("parent_message_id") == retry_user_message_id
                       or row["id"] == regenerate_reply_id}
        retained = [(message, message_id) for message, message_id in zip(historical_msgs, historical_ids)
                    if message_id not in variant_ids]
        historical_msgs = [message for message, _ in retained]
        historical_ids = [message_id for _, message_id in retained]
    if explicit_failed_retry and should_persist and not conversation_created and request_messages:
        # Read the actual tail independently of the requested context window.
        # Retry is an explicit operation; equal text on an ordinary send is new.
        tail_rows = await asyncio.to_thread(partial(chat_db.get_messages_for_conversation, conv_id, 2, 0, "DESC", strict_images=True))
        if (any(message.get("role") == "system" for message in request_messages)
                or any(str(row.get("sender", "")).lower() == "system" for row in tail_rows)):
            tail_ids, saved_retry_system_messages = await asyncio.to_thread(chat_db.get_retry_message_context, conv_id)
            tail_rows = [await asyncio.to_thread(partial(chat_db.get_message_by_id, message_id, strict_images=True))
                         for message_id in tail_ids]
            if any(row is None for row in tail_rows):
                raise cs.HTTPException(status_code=409, detail="The failed turn changed. Reload before retrying.")
        tail_metadata = {row["id"]: await asyncio.to_thread(chat_db.get_message_metadata, row["id"]) for row in tail_rows}
        requested_user = request_messages[-1]
        if requested_user.get("role") != "user":
            raise cs.HTTPException(status_code=409, detail="The failed turn changed. Reload the conversation before retrying.")

        def matches_saved_user(row: dict[str, Any]) -> bool:
            if str(row.get("sender", "")).lower() != "user":
                return False
            stored_metadata = tail_metadata.get(row["id"]) or {}
            parts = cs._saved_user_content_parts(row, stored_metadata.get("extra"))
            return cs._retry_content_components(requested_user.get("content")) == cs._retry_content_components(parts)

        if tail_rows:
            tail = tail_rows[0]
            if str(tail.get("sender", "")).lower() == "user":
                if not matches_saved_user(tail):
                    raise cs.HTTPException(status_code=409, detail="The unanswered turn changed. Reload the conversation before retrying.")
                retry_user_message_id = str(tail["id"])
            elif len(tail_rows) > 1 and matches_saved_user(tail_rows[1]):
                if cs._is_saved_chat_error_envelope({"role": "assistant", "content": tail.get("content", "")}):
                    retry_user_message_id = str(tail_rows[1]["id"])
                else:
                    saved_extra = (tail_metadata.get(tail_rows[1]["id"]) or {}).get("extra")
                    saved_client_id = saved_extra.get("client_message_id") if isinstance(saved_extra, dict) else None
                    saved_identity_is_valid = (
                        isinstance(saved_client_id, str)
                        and re.fullmatch(r"[A-Za-z0-9_-]{1,128}", saved_client_id)
                    )
                    if not (client_message_id and saved_identity_is_valid and saved_client_id != client_message_id):
                        raise cs.HTTPException(status_code=409, detail="This turn already has an answer. Reload the conversation before retrying.")

    retry_system_messages: list[dict[str, Any]] = []
    if explicit_failed_retry and retry_user_message_id:
        # Instructions are the requested current block, even when the client
        # also supplies overlapping history. A matched user suffix must not
        # trim a deliberately changed system prefix from persistence.
        retry_system_messages = [message for message in request_messages if message.get("role") == "system"]
        request_messages = [message for message in request_messages if message.get("role") != "system"]

    # If the client included history with conversation_id, trim overlaps against DB history
    overlap_cut = 0
    if conv_id and historical_msgs and request_messages and not versioned:
        has_non_user_role = any(
            msg.get("role") in {"assistant", "tool"} for msg in request_messages
        )
        all_user_roles = all(msg.get("role") == "user" for msg in request_messages)
        def _msg_sig(msg: dict[str, Any]) -> str:
            payload = {
                "role": msg.get("role"),
                "content": (cs._retry_content_components(retry_history_content.get(id(msg), msg.get("content"))) if reuse_saved_turn and msg.get("role") == "user" else cs._normalize_history_overlap_content(msg.get("content"), assistant=msg.get("role") == "assistant")),
                "tool_calls": msg.get("tool_calls"),
                "function_call": msg.get("function_call"),
                "tool_call_id": msg.get("tool_call_id"),
            }
            try:
                return _json.dumps(payload, sort_keys=True, default=str)
            except cs._CHAT_NONCRITICAL_EXCEPTIONS:
                return str(payload)

        if has_non_user_role:
            # Overlap detection assumes chronological ordering for comparison.
            # If history is requested newest-first, reverse a copy for overlap checks
            # while preserving the requested order in the final payload.
            hist_for_overlap = (
                list(reversed(historical_msgs)) if history_order == "desc" else historical_msgs
            )
            hist_sigs = [_msg_sig(m) for m in hist_for_overlap]
            req_sigs = [_msg_sig(m) for m in request_messages]
            # An ordinary final user is a new turn, even when its text equals
            # an interrupted saved user. Only explicit Retry may reuse it.
            if request_messages[-1].get("role") == "user" and not retry_user_message_id:
                req_sigs = req_sigs[:-1]
            max_k = min(len(hist_sigs), len(req_sigs))
            overlap_start = None
            overlap_k = 0
            for k in range(max_k, 0, -1):
                tail = hist_sigs[-k:]
                for i in range(0, len(req_sigs) - k + 1):
                    if req_sigs[i:i + k] == tail:
                        overlap_start = i
                        overlap_k = k
                        break
                if overlap_k:
                    break
            if not overlap_k:
                # A restored client can lag behind an interrupted canonical
                # turn. Trim only its exact historical PREFIX, including an
                # assistant, proven within the loaded history window. Later
                # canonical rows remain model context; do not replay them.
                for k in range(max_k, 0, -1):
                    if not any(message.get("role") == "assistant" for message in request_messages[:k]):
                        continue
                    if any(hist_sigs[i:i + k] == req_sigs[:k] for i in range(len(hist_sigs) - k + 1)):
                        overlap_start = 0
                        overlap_k = k
                        break
            if overlap_k and overlap_start is not None:
                overlap_cut = overlap_start + overlap_k
                if overlap_cut > 0:
                    cs.logger.debug(
                        'Trimmed {} request messages overlapping history for conv_id {}.',
                        overlap_cut,
                        conv_id,
                    )
        else:
            try:
                user_only_trim = str(os.getenv("CHAT_TRIM_USER_ONLY_OVERLAP", "false")).strip().lower() in {
                    "1",
                    "true",
                    "yes",
                    "on",
                }
            except cs._CHAT_NONCRITICAL_EXCEPTIONS:
                user_only_trim = False
            retrying_failed_turn = (history_ended_with_error or retry_user_message_id is not None) and len(request_messages) == 1
            if (user_only_trim or retrying_failed_turn) and all_user_roles:
                hist_for_overlap = (
                    list(reversed(historical_msgs)) if history_order == "desc" else historical_msgs
                )
                hist_sigs = [_msg_sig(m) for m in hist_for_overlap]
                req_sigs = [_msg_sig(m) for m in request_messages]
                if req_sigs and len(req_sigs) <= len(hist_sigs):
                    if hist_sigs[-len(req_sigs):] == req_sigs:
                        overlap_cut = len(req_sigs)
                        cs.logger.debug(
                            'Trimmed {} user-only request messages overlapping history for conv_id {}.',
                            overlap_cut,
                            conv_id,
                        )

    if versioned:
        if regenerate_reply_id is not None or continue_reply_id is not None:
            overlap_cut = len(request_messages)
        formatted_history = cs._history_for_request(history_snapshot, selected_content, character_card, continue_prefix)
        if server_existing_input is not None:
            existing_dictionary_content = assistant_context.pop("history_dictionary_user_content", [])
            if existing_dictionary_content:
                formatted_history[-1]["content"] = existing_dictionary_content[0]
        historical_msgs.extend(formatted_history)
        assistant_parent_message_id = (selected_content[-1]["id"] if server_existing_input is not None
            else admission["input_message_id"])
        runtime_state["assistant_parent_message_id"] = assistant_parent_message_id
        if server_existing_input is not None:
            runtime_state["tldw_history_server_existing_input_v1"] = history_selection.model_dump(mode="json")
        else:
            runtime_state["tldw_history_admission_v1"] = admission
        if continue_prefix is not None:
            runtime_state["history_continue_prefix"] = continue_prefix

    if retry_user_message_id and any(message.get("role") == "user" for message in request_messages[overlap_cut:-1]):
        raise cs.HTTPException(status_code=409, detail="This retry contains extra unsaved user messages. Resolve the local conversation before retrying.")
    if retry_user_message_id and client_message_id:
        saved_metadata = await asyncio.to_thread(chat_db.get_message_metadata, retry_user_message_id)
        saved_extra = (saved_metadata or {}).get("extra")
        saved_client_id = saved_extra.get("client_message_id") if isinstance(saved_extra, dict) else None
        if saved_client_id and saved_client_id != client_message_id:
            raise cs.HTTPException(status_code=409, detail="The failed turn identity changed. Reload the conversation before retrying.")

    if legacy_regenerated_reply is not None:
        from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import ConflictError

        try:
            await asyncio.to_thread(
                chat_db.update_message, legacy_regenerated_reply["id"],
                {"parent_message_id": retry_user_message_id}, legacy_regenerated_reply["version"],
            )
        except ConflictError as exc:
            raise cs.HTTPException(status_code=409, detail="The saved reply changed. Reload before regenerating.") from exc

    persisted_user_message_id: str | None = assistant_parent_message_id if versioned else retry_user_message_id
    system_block_id = str(_uuid.uuid4())
    if (should_persist and retry_system_messages
            and cs._extract_system_messages(retry_system_messages) != cs._extract_system_messages(saved_retry_system_messages)):
        import sqlite3

        from tldw_Server_API.app.core.Buddy.publication import current_buddy_publication
        from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseError as BackendDatabaseError
        from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDBError

        publication = current_buddy_publication.get()

        def persist_retry_instructions() -> None:
            """Keep publication receipts and the complete block in one transaction."""
            with chat_db.transaction() as conn:
                if publication is not None:
                    publication.repository.assert_publication(conn, publication.turn, conv_id)
                message_ids = chat_db.add_retry_system_instruction_block(
                    conv_id, retry_system_messages, system_block_id,
                )
                if publication is not None:
                    for message_id in message_ids:
                        publication.repository.record_message(conn, publication.turn["id"], "system", message_id)

        try:
            await asyncio.to_thread(persist_retry_instructions)
        except (CharactersRAGDBError, sqlite3.Error, BackendDatabaseError) as exc:
            raise cs.HTTPException(
                status_code=503,
                detail="Unable to save updated retry instructions. Try again.",
            ) from exc
    current_turn: list[dict[str, Any]] = list(retry_system_messages)
    dictionary_user_content = iter(assistant_context.pop("history_dictionary_user_content", []))
    for index, msg_dict in enumerate(request_messages[overlap_cut:], start=overlap_cut):
        role = msg_dict.get("role")
        msg_for_db = msg_dict.copy()
        if role == "system":
            msg_for_db["system_instruction_block_id"] = system_block_id
        if role == "assistant" and character_card:
            # Persist assistant sender as sanitized character name
            msg_for_db["name"] = cs.sanitize_sender_name(character_card.get("name", "Assistant"))
        reused_retry = retry_user_message_id is not None and index == len(request_messages) - 1
        if should_persist and not reused_retry and not versioned:
            if role == "user" and index == len(request_messages) - 1 and client_message_id:
                msg_for_db["client_message_id"] = client_message_id
            saved_message_id = await save_message_fn(chat_db, conv_id, msg_for_db, use_transaction=True)
            if role == "user":
                persisted_user_message_id = str(saved_message_id) if saved_message_id else None
        msg_for_llm = msg_dict.copy()
        if versioned and role == "user":
            msg_for_llm["content"] = next(dictionary_user_content, msg_for_llm.get("content"))
        if role == "assistant" and character_card and character_card.get("name"):
            name = cs.sanitize_sender_name(character_card.get("name"))
            if name:
                msg_for_llm["name"] = name
        if not reused_retry or retry_user_message_id not in historical_ids:
            current_turn.append(msg_for_llm)

    if retry_user_message_id is not None and retry_user_message_id in historical_ids:
        # Overlap and identity checks above have accepted this exact saved turn.
        # Keep its validated content, but make it current even with DESC history.
        retry_index = historical_ids.index(retry_user_message_id)
        current_turn.append(historical_msgs.pop(retry_index))

    if continuation_spec and assistant_prefill and not versioned:
        prefill_payload: dict[str, Any] = {
            "role": "assistant",
            "content": assistant_prefill,
        }
        if character_card and character_card.get("name"):
            prefill_name = cs.sanitize_sender_name(character_card.get("name"))
            if prefill_name:
                prefill_payload["name"] = prefill_name
        current_turn.append(prefill_payload)
        if continuation_metadata is not None:
            continuation_metadata["assistant_prefill"] = assistant_prefill
            continuation_metadata["assistant_prefill_applied"] = True

    if runtime_state is not None:
        runtime_state["user_message_id"] = persisted_user_message_id
        if persisted_user_message_id:
            runtime_state["assistant_parent_message_id"] = persisted_user_message_id
        runtime_state["assistant_context"] = dict(assistant_context)
        if continuation_spec and continuation_metadata is not None:
            runtime_state["tldw_continuation"] = continuation_metadata
            if assistant_parent_message_id:
                runtime_state["assistant_parent_message_id"] = assistant_parent_message_id

    # Preserve requested history ordering (DB fetch already honors ASC/DESC).
    historical_msgs_for_payload = historical_msgs

    llm_payload_messages = historical_msgs_for_payload + current_turn
    if versioned and assistant_context.get("history_world_book_plan") is not None:
        from tldw_Server_API.app.core.Character_Chat.world_book_prompt_context import (
            build_frozen_world_book_prompt_context,
        )

        character_card["_history_world_book_context"] = build_frozen_world_book_prompt_context(
            llm_payload_messages, assistant_context["history_world_book_plan"],
        )

    return character_card, character_db_id, conv_id, conversation_created, llm_payload_messages, should_persist


# late imports: moved code reads these at call time (TASK-13950.25 R5)
from tldw_Server_API.app.core.Chat import chat_service as cs
