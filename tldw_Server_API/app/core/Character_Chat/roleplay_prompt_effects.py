"""Pure character prompt effects shared by legacy and frozen Roleplay paths."""

from __future__ import annotations

import json
from typing import Any

from tldw_Server_API.app.core.Character_Chat.modules.character_prompt_presets import (
    _safe_replace as _safe_replace_placeholders,
)


def _normalize_note_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()


def _extract_character_default_author_note(character: dict[str, Any]) -> str:
    if not isinstance(character, dict):
        return ""

    # Support direct fields and extension-based storage.
    for key in ("author_note", "authorNote", "memory_note", "memoryNote"):
        direct = _normalize_note_text(character.get(key))
        if direct:
            return direct

    extensions = character.get("extensions")
    if isinstance(extensions, str):
        try:
            extensions = json.loads(extensions)
        except json.JSONDecodeError:
            extensions = {}
    if not isinstance(extensions, dict):
        return ""

    for key in (
        "author_note",
        "authorNote",
        "default_author_note",
        "defaultAuthorNote",
        "memory_note",
        "memoryNote",
    ):
        ext_value = _normalize_note_text(extensions.get(key))
        if ext_value:
            return ext_value
    return ""


def _extract_character_memory_note(settings: dict[str, Any], character_id: Any) -> str:
    memory_by_id = settings.get("characterMemoryById")
    if not isinstance(memory_by_id, dict):
        return ""

    # Accept both raw and stringified keys for robustness across clients.
    lookup_keys = [character_id]
    if character_id is not None:
        lookup_keys.append(str(character_id))

    for key in lookup_keys:
        if key not in memory_by_id:
            continue
        entry = memory_by_id.get(key)
        if isinstance(entry, dict):
            note = _normalize_note_text(entry.get("note"))
            if note:
                return note
        else:
            note = _normalize_note_text(entry)
            if note:
                return note
    return ""


def _normalize_memory_scope(settings: dict[str, Any]) -> str:
    scope_raw = settings.get("memoryScope")
    scope = str(scope_raw).strip().lower() if isinstance(scope_raw, str) else ""
    if scope in {"shared", "character", "both"}:
        return scope
    return "shared"


def _resolve_author_note_text(
    settings: dict[str, Any],
    character: dict[str, Any],
    *,
    for_prompt: bool = True,
) -> str:
    # Optional toggles reserved for Stage D2 behavior; defaults keep note enabled.
    if settings.get("authorNoteEnabled") is False:
        return ""
    # GM-only and exclude-from-prompt flags suppress injection into LLM context
    # but the note text should still be resolvable for UI display (for_prompt=False).
    if for_prompt:
        if settings.get("authorNoteGmOnly") is True:
            return ""
        if settings.get("authorNoteExcludeFromPrompt") is True:
            return ""

    shared_note = _normalize_note_text(settings.get("authorNote"))
    character_note = _extract_character_memory_note(settings, character.get("id"))
    if not character_note:
        character_note = _extract_character_default_author_note(character)

    scope = _normalize_memory_scope(settings)

    if scope == "character":
        note = character_note
    elif scope == "both":
        note = "\n\n".join(part for part in (shared_note, character_note) if part)
    else:
        # Shared scope: keep chat note precedence, fallback to character default.
        note = shared_note or character_note

    return note.strip()


def _resolve_author_note_position(settings: dict[str, Any]) -> Any:
    for key in (
        "authorNotePosition",
        "authorNotePlacement",
        "authorNoteInjectionPosition",
    ):
        if key in settings:
            return settings.get(key)
    return "before_system"


def _insert_author_note_at_depth(
    messages: list[dict[str, Any]],
    author_note_message: dict[str, Any],
    depth: int,
) -> list[dict[str, Any]]:
    if depth <= 0:
        return [author_note_message, *messages]

    non_system_count = 0
    for idx, message in enumerate(messages):
        role = str(message.get("role", "")).strip().lower()
        if role == "system":
            continue
        if non_system_count >= depth:
            return [*messages[:idx], author_note_message, *messages[idx:]]
        non_system_count += 1
    return [*messages, author_note_message]


def _insert_author_note_message(
    messages: list[dict[str, Any]],
    author_note_message: dict[str, Any],
    position: Any,
) -> list[dict[str, Any]]:
    if isinstance(position, int):
        return _insert_author_note_at_depth(messages, author_note_message, position)

    if isinstance(position, dict):
        mode = str(position.get("mode", "")).strip().lower()
        if mode in {"depth", "at_depth"}:
            depth = position.get("depth")
            if isinstance(depth, int):
                return _insert_author_note_at_depth(messages, author_note_message, depth)
        if mode in {"before_system", "before-system", "before"}:
            return [author_note_message, *messages]

    if isinstance(position, str):
        normalized = position.strip().lower()
        if normalized in {"before_system", "before-system", "before system", "before"}:
            return [author_note_message, *messages]
        if normalized.startswith("depth"):
            suffix = normalized.replace("depth", "", 1).strip()
            if suffix.startswith(":"):
                suffix = suffix[1:].strip()
            if suffix.isdigit():
                return _insert_author_note_at_depth(messages, author_note_message, int(suffix))

    return [author_note_message, *messages]


def _inject_author_note_from_settings(
    messages: list[dict[str, Any]],
    settings_row: Any,
    character: dict[str, Any],
    char_name: str,
    user_name: str,
) -> list[dict[str, Any]]:
    if not isinstance(settings_row, dict):
        return messages
    settings = settings_row.get("settings")
    if not isinstance(settings, dict):
        return messages

    note_text = _resolve_author_note_text(settings, character)
    if not note_text:
        return messages

    resolved_note = _safe_replace_placeholders(note_text, char_name, user_name).strip()
    if not resolved_note:
        return messages

    author_note_message = {
        "role": "system",
        "content": f"Author's note:\n{resolved_note}",
    }
    position = _resolve_author_note_position(settings)
    return _insert_author_note_message(messages, author_note_message, position)


def inject_character_memory_rows(
    messages: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    user_name: str,
    token_budget: int = 300,
) -> list[dict[str, Any]]:
    """Format detached character memories with the existing category and size budget."""
    # Sort by salience DESC, then last_modified DESC
    rows = sorted(
        rows,
        key=lambda r: (float(r.get("salience", 0)), r.get("last_modified", "")),
        reverse=True,
    )

    # Group by category
    categories: dict[str, list[str]] = {}
    for row in rows:
        cat = row.get("memory_type", "manual")
        content = row.get("content", "").strip()
        if content:
            categories.setdefault(cat, []).append(content)

    # Build text block
    category_labels = {
        "fact": "Facts",
        "relationship": "Relationship",
        "event": "Events",
        "preference": "Preferences",
        "manual": "Notes",
    }
    lines: list[str] = [f"Character memory about {user_name}:"]
    char_budget = token_budget * 4  # approximate chars
    total_chars = len(lines[0])

    for cat, label in category_labels.items():
        entries = categories.get(cat)
        if not entries:
            continue
        header = f"[{label}]"
        lines.append(header)
        total_chars += len(header) + 1
        for entry in entries:
            line = f"- {entry}"
            if total_chars + len(line) + 1 > char_budget:
                break
            lines.append(line)
            total_chars += len(line) + 1
        if total_chars >= char_budget:
            break

    if len(lines) <= 1:
        return messages

    memory_block = "\n".join(lines)
    memory_message = {"role": "system", "content": memory_block}
    return [*messages, memory_message]


def insert_character_greeting(messages: list[dict[str, Any]], greeting_text: str) -> list[dict[str, Any]]:
    """Insert one resolved greeting using the existing placement and duplicate rule."""
    insert_idx = len(messages)
    for idx in range(len(messages) - 1, -1, -1):
        role = str(messages[idx].get("role", "")).strip().lower()
        if role == "user":
            insert_idx = idx
            break

    if insert_idx == len(messages):
        insert_idx = 0
        while insert_idx < len(messages):
            role = str(messages[insert_idx].get("role", "")).strip().lower()
            if role != "system":
                break
            insert_idx += 1

    if insert_idx > 0:
        prev_message = messages[insert_idx - 1]
        prev_role = str(prev_message.get("role", "")).strip().lower()
        prev_content = str(prev_message.get("content", "")).strip()
        if prev_role == "assistant" and prev_content == greeting_text:
            return messages

    greeting_message = {
        "role": "assistant",
        "content": greeting_text,
    }
    return [*messages[:insert_idx], greeting_message, *messages[insert_idx:]]
