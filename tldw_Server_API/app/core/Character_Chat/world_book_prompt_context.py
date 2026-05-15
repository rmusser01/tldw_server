"""
World-book prompt context helpers for character chat.

This module centralizes prompt assembly and bounded diagnostics for lorebook
injection. It does not change provider dispatch behavior.
"""
from __future__ import annotations

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
