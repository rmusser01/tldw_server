"""
World-book prompt context helpers for character chat.

This module centralizes prompt assembly and bounded diagnostics for lorebook
injection. It does not change provider dispatch behavior.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from tldw_Server_API.app.core.Chat.prompt_cost_envelope import (
    estimate_segment_tokens,
    fingerprint_text,
)

_STATIC_HINT_KEYS = frozenset(
    {
        "static",
        "pinned",
        "always_on",
        "constant",
        "cache_static",
        "cache_pinned",
        "static_or_pinned",
    }
)
_STATIC_ACTIVATION_REASONS = frozenset({"static", "pinned", "always_on", "constant"})


@dataclass(frozen=True)
class WorldBookPromptContext:
    """Prompt-safe world-book context and diagnostics."""

    text: str = ""
    system_message: Mapping[str, str] | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    legacy_diagnostics: tuple[dict[str, Any], ...] = ()
    fingerprint: str = field(default_factory=lambda: fingerprint_text(""))
    estimated_tokens: int = 0


def build_recent_world_book_scan_text(
    messages: Sequence[Mapping[str, Any]],
    *,
    window_chars: int = 2000,
) -> str:
    """Build the recent text scanned for world-book triggers."""
    parts: list[str] = []
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        role = str(message.get("role", "")).strip().lower()
        if role not in {"user", "assistant"}:
            continue
        content = _content_to_scan_text(message.get("content"))
        if content:
            parts.append(content)
    recent = " ".join(parts)
    if window_chars <= 0:
        return recent
    return recent[-window_chars:]


def build_world_book_prompt_context(
    messages: Sequence[Mapping[str, Any]],
    *,
    db: Any = None,
    world_book_service: Any = None,
    character_id: Any = None,
    recent_text_window_chars: int = 2000,
) -> WorldBookPromptContext:
    """Process messages through world-book matching and return prompt diagnostics."""
    recent_text = build_recent_world_book_scan_text(
        messages,
        window_chars=recent_text_window_chars,
    )
    if not recent_text.strip():
        return _empty_context()

    if world_book_service is None:
        if db is None:
            return _empty_context()
        from tldw_Server_API.app.core.Character_Chat.world_book_manager import WorldBookService

        world_book_service = WorldBookService(db)

    result = world_book_service.process_context(
        text=recent_text,
        character_id=character_id,
        include_diagnostics=True,
    )
    return world_book_prompt_context_from_result(result)


def world_book_prompt_context_from_result(result: Any) -> WorldBookPromptContext:
    """Format already matched lore through the shared bounded diagnostics contract."""
    if not isinstance(result, Mapping):
        return _empty_context()

    processed_context = str(result.get("processed_context") or "").strip()
    if not processed_context:
        return _empty_context(result=result)

    text = f"World info:\n{processed_context}"
    legacy_diagnostics = tuple(
        _sanitize_legacy_diagnostics(result.get("diagnostics") or [])
    )
    fingerprint = fingerprint_text(text)
    estimated_tokens = estimate_segment_tokens(text)
    diagnostics = _build_prompt_diagnostics(
        result=result,
        legacy_diagnostics=legacy_diagnostics,
        fingerprint=fingerprint,
        estimated_tokens=estimated_tokens,
        text_length=len(text),
    )
    return WorldBookPromptContext(
        text=text,
        system_message={"role": "system", "content": text},
        diagnostics=diagnostics,
        legacy_diagnostics=legacy_diagnostics,
        fingerprint=fingerprint,
        estimated_tokens=estimated_tokens,
    )


@dataclass(frozen=True)
class FrozenWorldBookPlan:
    """Immutable, detached lore accepted under the native owner transaction."""

    canonical_books: bytes


def build_frozen_world_book_plan(
    participant_books: Any,
    explicit_books: Any = (),
) -> FrozenWorldBookPlan:
    """Validate complete saved matching inputs without reading current sources.

    Disabled attachments apply only to inherited books; an explicit accepted
    selection may still use that same enabled book. Conflicting copies cannot
    silently select a newer definition.
    """
    from tldw_Server_API.app.core.Character_Chat.character_conversation_factory import (
        MAX_MATERIALIZED_WORLD_BOOK_ENTRIES,
        MAX_MATERIALIZED_WORLD_BOOK_ENTRY_BYTES,
        MAX_MATERIALIZED_WORLD_BOOKS,
        _materialized_world_book_entry_usage,
    )

    book_keys = {"id", "name", "description", "scan_depth", "token_budget", "recursive_scanning", "enabled",
                 "created_at", "last_modified", "version", "attachment_enabled", "attachment_priority", "entries"}
    entry_keys = {"id", "world_book_id", "keywords", "content", "priority", "enabled", "case_sensitive", "regex_match",
                  "whole_word_match", "metadata", "created_at", "last_modified"}
    books: dict[int, dict[str, Any]] = {}
    active: set[int] = set()

    def require_flag(value: Any) -> None:
        if type(value) not in (bool, int) or value not in (0, 1):
            raise ValueError("Saved lore requires a complete boolean policy")

    for group, inherited in ((participant_books, True), (explicit_books, False)):
        if not isinstance(group, (list, tuple)):
            raise ValueError("Saved lore must contain a book list")
        for book in group:
            if not isinstance(book, dict) or set(book) - book_keys:
                raise ValueError("Saved lore contains an unsupported book")
            for policy_key in ("id", "version", "scan_depth", "token_budget"):
                value = book.get(policy_key)
                minimum = 1 if policy_key in ("id", "version") else 0
                if type(value) is not int or value < minimum:
                    raise ValueError("Saved lore has invalid identity or numeric policy")
            for policy_key in ("enabled", "recursive_scanning"):
                require_flag(book.get(policy_key))
            if inherited:
                require_flag(book.get("attachment_enabled"))
                if type(book.get("attachment_priority")) is not int:
                    raise ValueError("Saved lore has an invalid attachment policy")
            entries = book.get("entries")
            if not isinstance(entries, list):
                raise ValueError("Saved lore is missing its captured entries")
            seen_entries: set[int] = set()
            for entry in entries:
                if not isinstance(entry, dict) or set(entry) - entry_keys:
                    raise ValueError("Saved lore contains an unsupported entry")
                if (type(entry.get("id")) is not int or entry["id"] <= 0 or entry["id"] in seen_entries
                        or type(entry.get("world_book_id")) is not int or entry["world_book_id"] != book["id"]):
                    raise ValueError("Saved lore entry identity is invalid")
                seen_entries.add(entry["id"])
                if (not isinstance(entry.get("keywords"), list)
                        or not all(isinstance(keyword, str) for keyword in entry["keywords"])
                        or not isinstance(entry.get("content"), str)
                        or type(entry.get("priority")) is not int or not isinstance(entry.get("metadata"), dict)):
                    raise ValueError("Saved lore entry matching inputs are incomplete")
                for policy_key in ("enabled", "case_sensitive", "regex_match", "whole_word_match"):
                    require_flag(entry.get(policy_key))
            detached = {key: value for key, value in book.items() if key not in {"attachment_enabled", "attachment_priority"}}
            previous = books.get(book["id"])
            if previous is not None and previous != detached:
                raise ValueError("Saved lore contains conflicting book revisions")
            books[book["id"]] = detached
            if book["enabled"] and (not inherited or book["attachment_enabled"]):
                active.add(book["id"])
    entry_count, entry_bytes = _materialized_world_book_entry_usage(list(books.values()))
    if (len(books) > MAX_MATERIALIZED_WORLD_BOOKS or entry_count > MAX_MATERIALIZED_WORLD_BOOK_ENTRIES
            or entry_bytes > MAX_MATERIALIZED_WORLD_BOOK_ENTRY_BYTES):
        raise ValueError("Saved lore exceeds its materialization bounds")
    canonical = json.dumps([book for book_id, book in books.items() if book_id in active],
                           allow_nan=False, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return FrozenWorldBookPlan(canonical)


def build_frozen_world_book_prompt_context(
    messages: Sequence[Mapping[str, Any]],
    plan: FrozenWorldBookPlan,
) -> WorldBookPromptContext:
    """Compose the accepted catalog over the exact selected/current message text."""
    from tldw_Server_API.app.core.Character_Chat.world_book_manager import WorldBookEntry, process_loaded_world_book_context

    recent_text = build_recent_world_book_scan_text(messages)
    if not recent_text.strip():
        return _empty_context()
    books = json.loads(plan.canonical_books)
    entries_by_book = {book["id"]: [WorldBookEntry.from_dict(entry) for entry in book["entries"]] for book in books}
    from tldw_Server_API.app.core.Utils.tokenizer import count_tokens

    result, _matched = process_loaded_world_book_context(
        recent_text, books, entries_by_book, count_tokens=count_tokens, include_diagnostics=True
    )
    return world_book_prompt_context_from_result(result)


def apply_world_book_prompt_context(
    messages: Sequence[Mapping[str, Any]],
    context: WorldBookPromptContext,
) -> list[dict[str, Any]]:
    """Insert world-book system context after leading system messages."""
    output = [dict(message) for message in messages if isinstance(message, Mapping)]
    if not context.system_message:
        return output

    insert_pos = 0
    for idx, message in enumerate(output):
        role = str(message.get("role", "")).strip().lower()
        if role == "system":
            insert_pos = idx + 1
        else:
            break
    output.insert(insert_pos, dict(context.system_message))
    return output


def _empty_context(*, result: Mapping[str, Any] | None = None) -> WorldBookPromptContext:
    fingerprint = fingerprint_text("")
    diagnostics = _build_prompt_diagnostics(
        result=result or {},
        legacy_diagnostics=(),
        fingerprint=fingerprint,
        estimated_tokens=0,
        text_length=0,
    )
    return WorldBookPromptContext(
        diagnostics=diagnostics,
        fingerprint=fingerprint,
    )


def _build_prompt_diagnostics(
    *,
    result: Mapping[str, Any],
    legacy_diagnostics: Sequence[Mapping[str, Any]],
    fingerprint: str,
    estimated_tokens: int,
    text_length: int,
) -> dict[str, Any]:
    entry_ids = _entry_ids_from_result(result, legacy_diagnostics)
    world_book_ids = sorted(
        {
            world_book_id
            for item in legacy_diagnostics
            if (world_book_id := _safe_int(item.get("world_book_id"))) is not None
        }
    )
    static_entry_ids = [
        entry_id
        for item in legacy_diagnostics
        if _is_static_or_pinned(item)
        if (entry_id := _safe_int(item.get("entry_id"))) is not None
    ]
    dynamic_entry_ids = [
        entry_id
        for item in legacy_diagnostics
        if not _is_static_or_pinned(item)
        if (entry_id := _safe_int(item.get("entry_id"))) is not None
    ]

    return {
        "fingerprint": fingerprint,
        "estimated_tokens": int(estimated_tokens),
        "text_length": int(text_length),
        "entries_matched": _safe_int(result.get("entries_matched")) or len(entry_ids),
        "included_entry_count": len(entry_ids),
        "dropped_entry_count": _safe_int(result.get("skipped_entries_due_to_budget")) or 0,
        "skipped_entries_due_to_budget": _safe_int(result.get("skipped_entries_due_to_budget")) or 0,
        "books_used": _safe_int(result.get("books_used")) or len(world_book_ids),
        "entry_ids": entry_ids,
        "world_book_ids": world_book_ids,
        "token_budget": _safe_int(result.get("token_budget")),
        "tokens_used": _safe_int(result.get("tokens_used")),
        "budget_exhausted": bool(result.get("budget_exhausted", False)),
        "diagnostic_count": len(legacy_diagnostics),
        "static_entry_ids": static_entry_ids,
        "dynamic_entry_ids": dynamic_entry_ids,
    }


def _entry_ids_from_result(
    result: Mapping[str, Any],
    legacy_diagnostics: Sequence[Mapping[str, Any]],
) -> list[int]:
    result_ids = result.get("entry_ids")
    if isinstance(result_ids, list):
        entry_ids = [_safe_int(value) for value in result_ids]
        return [value for value in entry_ids if value is not None]

    entry_ids = [_safe_int(item.get("entry_id")) for item in legacy_diagnostics]
    return [value for value in entry_ids if value is not None]


def _sanitize_legacy_diagnostics(raw_diagnostics: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_diagnostics, list):
        return []

    sanitized: list[dict[str, Any]] = []
    for item in raw_diagnostics:
        if not isinstance(item, Mapping):
            continue
        clean: dict[str, Any] = {}
        for key in (
            "entry_id",
            "world_book_id",
            "activation_reason",
            "token_cost",
            "priority",
            "regex_match",
            "appendable",
            "depth_level",
        ):
            if key in item:
                clean[key] = _sanitize_diagnostic_value(item.get(key))
        clean["static_or_pinned"] = _is_static_or_pinned(item)
        clean["cache_classification"] = (
            "static_or_pinned" if clean["static_or_pinned"] else "dynamic"
        )
        sanitized.append(clean)
    return sanitized


def _sanitize_diagnostic_value(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else value
    if isinstance(value, str):
        return value[:120]
    return str(type(value).__name__)


def _is_static_or_pinned(item: Mapping[str, Any]) -> bool:
    for key in _STATIC_HINT_KEYS:
        if _coerce_bool(item.get(key)):
            return True
    activation_reason = str(item.get("activation_reason") or "").strip().lower()
    return activation_reason in _STATIC_ACTIVATION_REASONS


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, int):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return False


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _content_to_scan_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, Mapping):
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return " ".join(part for part in parts if part)
    return str(content)


__all__ = [
    "WorldBookPromptContext",
    "apply_world_book_prompt_context",
    "build_recent_world_book_scan_text",
    "build_world_book_prompt_context",
]
