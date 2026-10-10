"""Detached saved behavior and ordered history for protected Roleplay dispatch."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from typing import Any

from tldw_Server_API.app.core.Character_Chat.character_conversation_factory import (
    PROMPT_COMPLETION_SETTING_CLASSIFICATION,
)
from tldw_Server_API.app.core.Character_Chat.modules.character_generation_presets import (
    supports_saved_character_extensions,
)
from tldw_Server_API.app.core.Character_Chat.modules.character_prompt_presets import (
    _safe_replace,
)
from tldw_Server_API.app.core.Character_Chat.modules.persona_exemplar_selector import (
    PersonaExemplarSelectorConfig,
    select_materialized_character_exemplars,
)
from tldw_Server_API.app.core.Character_Chat.roleplay_prompt_effects import (
    _inject_author_note_from_settings,
    _resolve_author_note_position,
    inject_character_memory_rows,
    insert_character_greeting,
)
from tldw_Server_API.app.core.Character_Chat.roleplay_resume_contract import (
    RoleplayResumeConflictError,
    roleplay_resume_sampling_arguments,
)
from tldw_Server_API.app.core.Character_Chat.world_book_manager import (
    WorldBookEntry,
    _validate_regex_safety,
    process_loaded_world_book_context,
)
from tldw_Server_API.app.core.Character_Chat.world_book_prompt_context import (
    apply_world_book_prompt_context,
    build_recent_world_book_scan_text,
    world_book_prompt_context_from_result,
)
from tldw_Server_API.app.core.Chat.history_context import render_saved_character_prompt
from tldw_Server_API.app.core.Chat.history_selection import HistorySelectionError
from tldw_Server_API.app.core.Chat.native_fork_projection import (
    validate_closed_tool_replay,
)

MAX_ROLEPLAY_PROMPT_BYTES = 32 * 1024 * 1024
_NOTE_EXTENSION_KEYS = frozenset(
    {
        "author_note",
        "authorNote",
        "default_author_note",
        "defaultAuthorNote",
        "memory_note",
        "memoryNote",
    }
)


def _unsupported() -> None:
    raise RoleplayResumeConflictError("resume_context_unsupported")


def _flag(value: Any) -> bool:
    if type(value) is bool:
        return value
    if type(value) is int and value in (0, 1):
        return bool(value)
    _unsupported()


def _positive_id(value: Any) -> int:
    if type(value) is not int or value < 1:
        _unsupported()
    return value


def _saved_lore(books: list[dict[str, Any]], history: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], Any]:
    loaded: list[dict[str, Any]] = []
    entries_by_book: dict[int, list[WorldBookEntry]] = {}
    for book in books:
        if not isinstance(book, dict):
            _unsupported()
        book_id = _positive_id(book.get("id"))
        if book_id in entries_by_book:
            continue
        if not _flag(book.get("enabled", True)):
            continue
        if type(book.get("scan_depth")) is not int or not isinstance(book.get("entries"), list):
            _unsupported()
        _flag(book.get("recursive_scanning", False))
        entries: list[WorldBookEntry] = []
        seen: set[int] = set()
        for row in book["entries"]:
            if not isinstance(row, dict):
                _unsupported()
            entry_id = _positive_id(row.get("id"))
            if row.get("world_book_id") != book_id or entry_id in seen:
                _unsupported()
            seen.add(entry_id)
            if (
                not isinstance(row.get("keywords"), list)
                or any(not isinstance(word, str) or not word for word in row["keywords"])
                or not isinstance(row.get("content"), str)
                or type(row.get("priority")) is not int
                or not isinstance(row.get("metadata"), dict)
            ):
                _unsupported()
            flags = {
                key: _flag(row.get(key))
                for key in (
                    "enabled",
                    "case_sensitive",
                    "regex_match",
                    "whole_word_match",
                )
            }
            if flags["regex_match"] and any(not _validate_regex_safety(word)[0] for word in row["keywords"]):
                _unsupported()
            entry = WorldBookEntry(
                entry_id=entry_id,
                world_book_id=book_id,
                keywords=list(row["keywords"]),
                content=row["content"],
                priority=row["priority"],
                metadata=deepcopy(row["metadata"]),
                **flags,
            )
            if len(entry._patterns) != len(row["keywords"]):
                _unsupported()
            entries.append(entry)
        loaded.append(book)
        entries_by_book[book_id] = entries
    # The v1 budget estimator is fixed, independent of deployment tokenizer settings.
    result, _ = process_loaded_world_book_context(
        build_recent_world_book_scan_text(history),
        loaded,
        entries_by_book,
        count_tokens=lambda text: (len(text.encode("utf-8")) + 3) // 4,
        include_diagnostics=True,
        strict_matching=True,
    )
    context = world_book_prompt_context_from_result(result)
    return apply_world_book_prompt_context(history, context), context


def _saved_exemplars(rows: Any, user_turn: str, character_id: int) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        _unsupported()
    seen: set[str] = set()
    for row in rows:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("id"), str)
            or row["id"] in seen
            or ("character_id" in row and (type(row["character_id"]) is not int or row["character_id"] != character_id))
            or not isinstance(row.get("text"), str)
            or type(row.get("length_tokens")) is not int
            or row["length_tokens"] < 1
        ):
            _unsupported()
        seen.add(row["id"])
        for key in ("rhetorical", "safety_allowed", "safety_blocked"):
            if not isinstance(row.get(key), list) or any(not isinstance(item, str) for item in row[key]):
                _unsupported()
    return select_materialized_character_exemplars(
        rows, user_turn, PersonaExemplarSelectorConfig(), character_id=character_id
    ).selected


def _build_saved_messages(
    state: dict[str, Any], history: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sampling_arguments = roleplay_resume_sampling_arguments(state["effective_completion"])
    conversation = state["conversation"]
    snapshot = state.get("behavior_snapshot") or {}
    materialized = state.get("materialized_settings") or {}
    values = materialized.get("values", {})
    if snapshot.get("status") != "valid" or values.get("base_snapshot") != {
        "schema_version": snapshot.get("schema_version"),
        "digest": snapshot.get("digest"),
    }:
        _unsupported()
    settings = state.get("settings") or {}
    if (
        set(settings)
        - set(PROMPT_COMPLETION_SETTING_CLASSIFICATION)
        - {
            "roleplayResumeV1",
            "roleplayBehaviorV1",
            "schemaVersion",
            "updatedAt",
            "greetingsChecksum",
        }
    ):
        _unsupported()
    payload = snapshot["payload"]
    participants = values.get("participants", payload["participants"])
    controls = values["behavior_controls"]
    if not isinstance(participants, list) or len(participants) != 1 or controls["turn_taking_mode"] != "single":
        _unsupported()
    participant = participants[0]
    source = participant["source"]
    if (
        source["kind"] != "character"
        or not source["id"].isdigit()
        or str(int(source["id"])) != source["id"]
        or int(source["id"]) < 1
    ):
        _unsupported()
    character_id = int(source["id"])
    snapshot_bound = (
        conversation.get("assistant_binding_mode") == "snapshot_v1"
        and conversation.get("character_id") is None
        and conversation.get("assistant_id") == "snapshot:" + str(conversation.get("id"))
        and source == payload["participants"][0]["source"]
    )
    if conversation.get("assistant_kind") not in (None, "character") or (
        not snapshot_bound and str(conversation.get("character_id") or conversation.get("assistant_id")) != source["id"]
    ):
        _unsupported()
    if (
        values.get("assistant_overlay")
        or controls["auto_summary"].get("enabled")
        or controls["auto_summary"].get("summary")
    ):
        _unsupported()
    prompt_context = controls.get("prompt_context", {})
    if set(prompt_context) - {"conversationContext"}:
        _unsupported()
    if prompt_context.get("conversationContext"):
        saved_context = prompt_context["conversationContext"]
        if (
            not isinstance(saved_context, dict)
            or set(saved_context) - {"world_book_ids"}
            or "world_books" not in values
        ):
            _unsupported()
    extensions = participant["prompt"]["prompt_relevant_extensions"]
    if set(extensions) - {"character_extensions", "prompt_preset"}:
        _unsupported()
    card_extensions = deepcopy(extensions.get("character_extensions") or {})
    if not isinstance(card_extensions, dict) or any(
        not isinstance(card_extensions[key], str) for key in _NOTE_EXTENSION_KEYS & card_extensions.keys()
    ):
        _unsupported()
    if not supports_saved_character_extensions(
        {key: value for key, value in card_extensions.items() if key not in _NOTE_EXTENSION_KEYS}
    ):
        _unsupported()
    user_name = conversation.get("user_name") or "User"
    if not isinstance(user_name, str):
        _unsupported()
    card = render_saved_character_prompt(participant, values, user_name=user_name)
    card.update({"id": character_id, "extensions": card_extensions})
    defaults = participant.get("default_memory")
    if defaults is not None and (not isinstance(defaults, dict) or not isinstance(defaults.get("content"), str)):
        _unsupported()
    overrides = controls["applied_overrides"]
    note = controls["author_note"]
    expected_note_position = _resolve_author_note_position(overrides)
    if note["position"] != expected_note_position:
        _unsupported()
    note_settings = {
        **overrides,
        "authorNoteEnabled": note["enabled"],
        "authorNoteGmOnly": note["gm_only"],
        "authorNoteExcludeFromPrompt": note["exclude_from_prompt"],
        "authorNotePosition": note["position"],
        "memoryScope": controls["memory_scope"],
    }
    if "memory" in values:
        memory = values["memory"]
        if (
            not isinstance(memory, dict)
            or not isinstance(memory.get("author_note"), str)
            or not isinstance(memory.get("character_memory_by_id"), dict)
        ):
            _unsupported()
        if any(not isinstance(value, str) for value in memory["character_memory_by_id"].values()):
            _unsupported()
        if (
            type(memory.get("author_note_enabled")) is not bool
            or memory.get("author_note_position") != expected_note_position
        ):
            _unsupported()
        note_settings.update(
            {
                "authorNote": memory["author_note"],
                "authorNoteEnabled": memory["author_note_enabled"],
                "authorNotePosition": memory["author_note_position"],
                "memoryScope": memory["scope"],
                "characterMemoryById": memory["character_memory_by_id"],
            }
        )
        if source["id"] in memory["character_memory_by_id"] and not memory["character_memory_by_id"][source["id"]]:
            card["extensions"] = {
                key: value for key, value in card_extensions.items() if key not in _NOTE_EXTENSION_KEYS
            }
    elif "characterMemoryById" not in note_settings and defaults and defaults["content"]:
        note_settings["characterMemoryById"] = {source["id"]: defaults["content"]}
    if note_settings.get("memoryScope") not in {"shared", "character", "both"} or (
        "authorNote" in note_settings and not isinstance(note_settings["authorNote"], str)
    ):
        _unsupported()
    for key in ("authorNoteEnabled", "authorNoteGmOnly", "authorNoteExcludeFromPrompt"):
        if type(note_settings[key]) is not bool:
            _unsupported()
    messages = [
        {"role": "system", "content": card["system_prompt"]},
        *deepcopy(history),
    ]
    greeting = values.get("greeting", participant.get("greeting"))
    if greeting is not None and (not isinstance(greeting, dict) or not isinstance(greeting.get("content"), str)):
        _unsupported()
    if type(controls["greeting"].get("enabled")) is not bool:
        _unsupported()
    if (
        controls["greeting"]["enabled"]
        and greeting
        and not any(message.get("role") == "assistant" for message in history)
    ):
        text = _safe_replace(greeting["content"], card["name"], user_name).strip()
        if text:
            messages = insert_character_greeting(messages, text)
    messages = _inject_author_note_from_settings(messages, {"settings": note_settings}, card, card["name"], user_name)
    memory_rows = (defaults or {}).get("persona_memory_entries", [])
    if not isinstance(memory_rows, list):
        _unsupported()
    for row in memory_rows:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("content"), str)
            or row.get("memory_type") not in {"fact", "relationship", "event", "preference", "manual"}
            or type(row.get("salience")) not in (int, float)
            or not math.isfinite(row["salience"])
            or not isinstance(row.get("last_modified"), str)
        ):
            _unsupported()
    messages = inject_character_memory_rows(messages, memory_rows, user_name)
    user_turn = next(
        (
            message["content"]
            for message in reversed(history)
            if message.get("role") == "user" and isinstance(message.get("content"), str)
        ),
        "",
    )
    selected = _saved_exemplars(participant.get("exemplars", []), user_turn, character_id)
    if selected:
        messages.insert(
            0,
            {
                "role": "system",
                "content": "Character dialogue examples:\n"
                + "\n\n".join(_safe_replace(row["text"], card["name"], user_name) for row in selected),
            },
        )
    books = values.get("world_books", [])
    attached = participant.get("world_books", [])
    if not isinstance(books, list) or not isinstance(attached, list):
        _unsupported()
    pre_lore_messages = tuple(messages)
    messages, lore = _saved_lore(
        [
            *books,
            *(book for book in attached if _flag(book.get("attachment_enabled", True))),
        ],
        messages,
    )
    return {
        "card": card,
        "character_id": character_id,
        "user_name": user_name,
        "lorebook_diagnostics": list(lore.legacy_diagnostics),
        "guardrail_messages": pre_lore_messages,
        "world_book_text": lore.text or None,
        "sampling_arguments": sampling_arguments,
    }, messages


def build_roleplay_resume_messages(
    state: dict[str, Any], history: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build saved character effects and complete history without live-source reads."""
    try:
        context, messages = _build_saved_messages(state, history)
        validate_closed_tool_replay(messages)
    except (
        HistorySelectionError,
        KeyError,
        TypeError,
        ValueError,
        AttributeError,
        TimeoutError,
    ) as exc:
        if isinstance(exc, RoleplayResumeConflictError):
            raise
        raise RoleplayResumeConflictError("resume_context_unsupported") from exc
    if len(json.dumps(messages, ensure_ascii=False, allow_nan=False).encode("utf-8")) > MAX_ROLEPLAY_PROMPT_BYTES:
        raise RoleplayResumeConflictError("resume_context_too_large")
    provider = state["effective_completion"]["provider"]
    arguments = {**context.pop("sampling_arguments"), "messages_payload": messages}
    if provider == "anthropic" and any(
        (message.get("role") == "tool" and not isinstance(message.get("content"), str))
        or any(
            not call["function"]["name"].strip() or not isinstance(json.loads(call["function"]["arguments"]), dict)
            for call in message.get("tool_calls", [])
        )
        for message in messages
    ):
        _unsupported()
    if provider in {"google", "cohere"} and any(
        message.get("tool_calls") or "tool_call_id" in message or message.get("role") == "tool" for message in messages
    ):
        _unsupported()
    if provider in {"anthropic", "google", "cohere"}:
        from tldw_Server_API.app.core.LLM_Calls.adapter_utils import (
            split_system_message,
        )

        try:
            system_message, remaining = split_system_message(messages, leading_only=True)
        except TypeError as exc:
            raise RoleplayResumeConflictError("resume_context_unsupported") from exc
        if any(message.get("role") == "system" for message in remaining):
            _unsupported()
        if provider == "cohere" and any(
            message.get("role") not in {"user", "assistant"} or not isinstance(message.get("content"), str)
            for message in remaining
        ):
            _unsupported()
        arguments.update({"system_message": system_message, "messages_payload": remaining})
    context["provider_arguments"] = arguments
    return context, messages
