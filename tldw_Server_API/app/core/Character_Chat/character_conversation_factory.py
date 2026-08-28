"""Atomic creation of resumable character conversations."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import TYPE_CHECKING, Any

from tldw_Server_API.app.core.Character_Chat.character_behavior_snapshot import (
    DEFAULT_MAX_SNAPSHOT_BYTES,
    BehaviorSnapshotV1,
    build_behavior_snapshot,
    is_credential_key,
)
from tldw_Server_API.app.core.Character_Chat.modules.character_generation_presets import (
    resolve_character_generation_settings,
)
from tldw_Server_API.app.core.Character_Chat.modules.character_prompt_presets import (
    DEFAULT_PROMPT_PRESET,
    get_builtin_presets,
    resolve_character_prompt_preset,
)
from tldw_Server_API.app.core.Character_Chat.world_book_manager import WorldBookService
from tldw_Server_API.app.core.Chat.chat_service import is_model_known_for_provider
from tldw_Server_API.app.core.DB_Management.chacha.conversation_resume_store import (
    build_materialized_behavior_settings,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import BackendType, InputError
from tldw_Server_API.app.core.LLM_Calls.adapter_registry import get_registry
from tldw_Server_API.app.core.LLM_Calls.adapter_utils import (
    ensure_app_config,
    normalize_provider,
    resolve_provider_model,
)

if TYPE_CHECKING:
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


_SAMPLING_FIELDS = frozenset({"temperature", "top_p", "repetition_penalty", "stop"})
MAX_MATERIALIZED_PARTICIPANTS = 33
MAX_MATERIALIZED_WORLD_BOOKS = 64
MAX_MATERIALIZED_EXEMPLARS = 256
MAX_MATERIALIZED_EXEMPLAR_BYTES = 256 * 1024
MAX_MATERIALIZED_WORLD_BOOK_ENTRIES = 512
MAX_MATERIALIZED_WORLD_BOOK_ENTRY_BYTES = 512 * 1024
_MATERIALIZED_SOURCE_ROW_OVERHEAD_BYTES = 256
PENDING_GREETING_SETTINGS_KEY = "roleplayPendingGreetingV1"
_MATERIALIZED_EXEMPLAR_TEXT_FIELDS = (
    "id",
    "text",
    "source_type",
    "source_url_or_id",
    "source_date",
    "novelty_hint",
    "emotion",
    "scenario",
    "rhetorical",
    "register",
    "safety_allowed",
    "safety_blocked",
    "rights_notes",
)
DEFAULT_AUTO_SUMMARY_THRESHOLD_MESSAGES = 40
DEFAULT_AUTO_SUMMARY_WINDOW_MESSAGES = 12
_EXACT_BOOLEAN_BEHAVIOR_FIELDS = frozenset(
    {
        "authorNoteEnabled",
        "authorNoteExcludeFromPrompt",
        "authorNoteGmOnly",
        "autoSummaryEnabled",
        "greetingEnabled",
        "useCharacterDefault",
    }
)

# This is the reviewable contract inventory for every conversation-setting field
# consumed by prompt construction or completion routing. Unknown legacy keys remain
# compatible, but only fields classified as behavior may affect a resumable chat.
PROMPT_COMPLETION_SETTING_CLASSIFICATION = dict.fromkeys(
    (
        "assistantOverlay",
        "authorNote",
        "authorNoteEnabled",
        "authorNoteExcludeFromPrompt",
        "authorNoteGmOnly",
        "authorNoteInjectionPosition",
        "authorNotePlacement",
        "authorNotePosition",
        "autoSummaryEnabled",
        "autoSummaryMessageThreshold",
        "autoSummaryRecentWindow",
        "autoSummaryThresholdMessages",
        "autoSummaryWindowMessages",
        "characterMemoryById",
        "chatGenerationOverride",
        "chatPresetOverrideId",
        "conversationContext",
        "deepResearchAttachment",
        "deepResearchAttachmentHistory",
        "deepResearchPinnedAttachment",
        "generationOverrides",
        "greetingEnabled",
        "greetingScope",
        "greetingSelectionId",
        "memoryScope",
        "model",
        "participantCharacterIds",
        "participant_character_ids",
        "pinnedMessageIds",
        "presetScope",
        "promptPreset",
        "prompt_preset",
        "provider",
        "summary",
        "turnTakingMode",
        "useCharacterDefault",
    ),
    "behavior",
)
PROMPT_COMPLETION_SETTING_CLASSIFICATION.update(
    {
        "characterMemoryExtraction": "side_effect",
        "greetingsChecksum": "metadata",
        "schemaVersion": "metadata",
        "updatedAt": "metadata",
    }
)
RESUMABLE_BEHAVIOR_SETTING_KEYS = frozenset(
    key
    for key, classification in PROMPT_COMPLETION_SETTING_CLASSIFICATION.items()
    if classification == "behavior"
)
_INELIGIBLE_SAFE_BEHAVIOR_KEYS = frozenset(
    {
        "autoSummaryEnabled",
        "autoSummaryMessageThreshold",
        "autoSummaryRecentWindow",
        "autoSummaryThresholdMessages",
        "autoSummaryWindowMessages",
        "greetingEnabled",
        "greetingScope",
        "greetingSelectionId",
        "pinnedMessageIds",
        "summary",
        "useCharacterDefault",
    }
)


class _SourceDrift(RuntimeError):
    pass


@dataclass(frozen=True)
class _MaterializedBehavior:
    snapshot: BehaviorSnapshotV1
    primary_character: dict[str, Any]


def _row_dict(row: Any, result: Any) -> dict[str, Any]:
    if isinstance(row, dict):
        return dict(row)
    mapping = getattr(row, "_mapping", None)
    if mapping is not None:
        return dict(mapping)
    try:
        return dict(row)
    except (TypeError, ValueError):
        keys = result.keys() if callable(getattr(result, "keys", None)) else []
        return dict(zip(keys, row, strict=False))


def _safe_json(value: Any) -> Any:
    """Copy JSON behavior data while rejecting credentials and binary payloads."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if is_credential_key(key_text):
                raise InputError(
                    f"materialized_behavior contains credential-bearing key {key_text!r}."
                )
            if isinstance(item, (bytes, bytearray, memoryview)):
                continue
            result[key_text] = _safe_json(item)
        return result
    if isinstance(value, (list, tuple, set)):
        return [
            _safe_json(item)
            for item in value
            if not isinstance(item, (bytes, bytearray, memoryview))
        ]
    return str(value)


def reject_resumable_behavior_credentials(
    value: Any,
    *,
    path: str = "conversation_settings",
) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            if is_credential_key(key_text):
                raise InputError(f"{path} contains credential-bearing key {key_text!r}.")
            reject_resumable_behavior_credentials(item, path=f"{path}.{key_text}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            reject_resumable_behavior_credentials(item, path=f"{path}[{index}]")


def normalize_materialized_participant_ids(
    primary_character_id: int,
    participant_character_ids: Sequence[Any],
) -> list[int]:
    """Return bounded, positive, de-duplicated participant IDs in stable order."""
    if isinstance(primary_character_id, bool):
        raise InputError("character_id must be a positive integer.")
    try:
        primary = int(primary_character_id)
    except (TypeError, ValueError) as exc:
        raise InputError("character_id must be a positive integer.") from exc
    if primary <= 0:
        raise InputError("character_id must be a positive integer.")
    ordered = [primary]
    for raw_id in participant_character_ids:
        if isinstance(raw_id, bool):
            raise InputError("participant character IDs must be positive integers.")
        try:
            character_id = int(raw_id)
        except (TypeError, ValueError) as exc:
            raise InputError(
                "participant character IDs must be positive integers."
            ) from exc
        if character_id <= 0:
            raise InputError("participant character IDs must be positive integers.")
        if character_id not in ordered:
            ordered.append(character_id)
        if len(ordered) > MAX_MATERIALIZED_PARTICIPANTS:
            raise InputError(
                f"A resumable chat supports at most {MAX_MATERIALIZED_PARTICIPANTS} participants."
            )
    return ordered


def normalize_materialized_world_book_ids(values: Sequence[Any]) -> list[int]:
    """Return bounded, positive, de-duplicated world-book IDs in stable order."""
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise InputError("conversationContext.world_book_ids must be a list.")
    ordered: list[int] = []
    for raw_id in values:
        if isinstance(raw_id, bool):
            raise InputError("World book IDs must be positive integers.")
        try:
            world_book_id = int(raw_id)
        except (TypeError, ValueError) as exc:
            raise InputError("World book IDs must be positive integers.") from exc
        if world_book_id <= 0:
            raise InputError("World book IDs must be positive integers.")
        if world_book_id not in ordered:
            ordered.append(world_book_id)
        if len(ordered) > MAX_MATERIALIZED_WORLD_BOOKS:
            raise InputError(
                f"A resumable chat supports at most {MAX_MATERIALIZED_WORLD_BOOKS} world books."
            )
    return ordered


def validate_resumable_behavior_boole(settings: Mapping[str, Any]) -> None:
    """Reject loose boolean representations at the resumable trust boundary."""
    for key in _EXACT_BOOLEAN_BEHAVIOR_FIELDS:
        if key in settings and type(settings[key]) is not bool:
            raise InputError(f"{key} must be a boolean.")
    for container_key in ("chatGenerationOverride", "generationOverrides", "summary"):
        container = settings.get(container_key)
        if (
            isinstance(container, Mapping)
            and "enabled" in container
            and type(container["enabled"]) is not bool
        ):
            raise InputError(f"{container_key}.enabled must be a boolean.")


def _behavior_bool(
    settings: Mapping[str, Any],
    key: str,
    *,
    default: bool,
) -> bool:
    value = settings.get(key, default)
    return value if type(value) is bool else default


def _is_postgres_connection(conn: Any) -> bool:
    backend = getattr(conn, "_backend", None)
    return getattr(backend, "backend_type", None) == BackendType.POSTGRESQL


def _bounded_behavior_int(
    value: Any,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    if isinstance(value, bool):
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(parsed, maximum))


def _author_note_position(settings: Mapping[str, Any]) -> Any:
    for key in (
        "authorNotePosition",
        "authorNotePlacement",
        "authorNoteInjectionPosition",
    ):
        if key not in settings:
            continue
        value = settings.get(key)
        if isinstance(value, Mapping):
            return _safe_json(value)
        if isinstance(value, (str, int)) and not isinstance(value, bool):
            return value
    # This must match the live prompt-construction default.
    return "before_system"


def _deduplicated_text_ids(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        if item is None:
            continue
        item_id = str(item).strip()
        if item_id and item_id not in result:
            result.append(item_id)
    return result


def build_materialized_behavior_controls(
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the closed prompt/completion control authority from merged settings."""
    validate_resumable_behavior_boole(settings)
    summary = settings.get("summary")
    summary = summary if isinstance(summary, Mapping) else {}
    threshold_raw = settings.get("autoSummaryThresholdMessages")
    if threshold_raw is None:
        threshold_raw = settings.get("autoSummaryMessageThreshold")
    if threshold_raw is None:
        threshold_raw = summary.get("thresholdMessages", summary.get("messageThreshold"))
    threshold = _bounded_behavior_int(
        threshold_raw,
        default=DEFAULT_AUTO_SUMMARY_THRESHOLD_MESSAGES,
        minimum=2,
        maximum=5_000,
    )
    window_raw = settings.get("autoSummaryWindowMessages")
    if window_raw is None:
        window_raw = settings.get("autoSummaryRecentWindow")
    if window_raw is None:
        window_raw = summary.get("windowMessages", summary.get("recentWindowMessages"))
    window = _bounded_behavior_int(
        window_raw,
        default=DEFAULT_AUTO_SUMMARY_WINDOW_MESSAGES,
        minimum=1,
        maximum=2_000,
    )
    if window >= threshold:
        window = max(1, threshold - 1)

    summary_enabled = settings.get("autoSummaryEnabled", summary.get("enabled", False))
    auto_summary: dict[str, Any] = {
        "enabled": summary_enabled if type(summary_enabled) is bool else False,
        "threshold_messages": threshold,
        "window_messages": window,
    }
    if summary:
        auto_summary["summary"] = _safe_json(summary)

    applied_overrides = {
        key: _safe_json(settings[key])
        for key in sorted(RESUMABLE_BEHAVIOR_SETTING_KEYS)
        if key in settings
    }
    prompt_context = {
        key: _safe_json(settings[key])
        for key in (
            "conversationContext",
            "deepResearchAttachment",
            "deepResearchAttachmentHistory",
            "deepResearchPinnedAttachment",
        )
        if key in settings
    }
    turn_mode = str(settings.get("turnTakingMode") or "single").strip().lower()
    if turn_mode in {"round-robin", "round robin"}:
        turn_mode = "round_robin"
    if turn_mode not in {"single", "round_robin"}:
        turn_mode = "single"

    return {
        "applied_overrides": applied_overrides,
        "author_note": {
            "enabled": _behavior_bool(
                settings,
                "authorNoteEnabled",
                default=True,
            ),
            "gm_only": _behavior_bool(
                settings,
                "authorNoteGmOnly",
                default=False,
            ),
            "exclude_from_prompt": _behavior_bool(
                settings,
                "authorNoteExcludeFromPrompt",
                default=False,
            ),
            "position": _author_note_position(settings),
        },
        "auto_summary": auto_summary,
        "greeting": {
            "enabled": _behavior_bool(
                settings,
                "greetingEnabled",
                default=True,
            ),
            "scope": (
                str(settings.get("greetingScope") or "chat").strip().lower()
                if str(settings.get("greetingScope") or "chat").strip().lower()
                in {"chat", "character"}
                else "chat"
            ),
            "selection_id": (
                str(settings["greetingSelectionId"])
                if settings.get("greetingSelectionId") is not None
                else None
            ),
            "use_character_default": _behavior_bool(
                settings,
                "useCharacterDefault",
                default=True,
            ),
        },
        "memory_scope": (
            str(settings.get("memoryScope") or "shared").strip().lower()
            if str(settings.get("memoryScope") or "shared").strip().lower()
            in {"shared", "character", "both"}
            else "shared"
        ),
        "pinned_message_ids": _deduplicated_text_ids(
            settings.get("pinnedMessageIds")
        ),
        "preset_scope": (
            str(settings.get("presetScope") or "character").strip().lower()
            if str(settings.get("presetScope") or "character").strip().lower()
            in {"chat", "character"}
            else "character"
        ),
        "prompt_context": prompt_context,
        "turn_taking_mode": turn_mode,
    }


def build_creation_materialized_behavior_settings(
    snapshot: BehaviorSnapshotV1,
    effective_completion: Mapping[str, Any],
    settings: Mapping[str, Any],
    *,
    max_bytes: int = DEFAULT_MAX_SNAPSHOT_BYTES,
) -> dict[str, Any]:
    """Bind v1 materialized settings to the immutable creation snapshot."""
    return build_materialized_behavior_settings(
        {
            "base_snapshot": {
                "schema_version": snapshot.schema_version,
                "digest": snapshot.digest,
            },
            "behavior_controls": build_materialized_behavior_controls(settings),
            "effective_completion": _safe_json(effective_completion),
        },
        max_bytes=max_bytes,
    )


def _normalize_prompt_preset_id(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise InputError("prompt_preset_id must be a non-empty string.")
    return value.strip()


def _character_prompt_preset_is_explicit(
    character: Mapping[str, Any],
    resolved_preset: str,
) -> bool:
    extensions = character.get("extensions")
    if not isinstance(extensions, Mapping):
        return False
    selected: Any = None
    tldw_extensions = extensions.get("tldw")
    if isinstance(tldw_extensions, Mapping):
        selected = tldw_extensions.get("prompt_preset") or tldw_extensions.get(
            "promptPreset"
        )
    if not selected:
        selected = extensions.get("prompt_preset") or extensions.get("promptPreset")
    return isinstance(selected, str) and selected.strip() == resolved_preset


def _decode_json(value: Any, default: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    return value if value is not None else default


def _normalize_greeting_values(value: Any) -> list[str]:
    def _strings(entries: Sequence[Any]) -> list[str]:
        return [
            item.strip()
            for item in entries
            if isinstance(item, str) and item.strip()
        ]

    if isinstance(value, str):
        trimmed = value.strip()
        if not trimmed:
            return []
        try:
            parsed = json.loads(trimmed)
        except json.JSONDecodeError:
            return [trimmed]
        if isinstance(parsed, list):
            return _strings(parsed)
        if isinstance(parsed, str):
            try:
                nested = json.loads(parsed)
            except json.JSONDecodeError:
                return [trimmed]
            if isinstance(nested, list):
                return _strings(nested)
        return [trimmed]
    if isinstance(value, list):
        return _strings(value)
    return []


def collect_character_greeting_texts(character: Mapping[str, Any]) -> list[str]:
    """Return live-picker greeting values: nonempty and ordered-deduplicated."""
    greetings: list[str] = []
    seen: set[str] = set()
    for field_name in (
        "greeting",
        "first_message",
        "firstMessage",
        "greet",
        "alternate_greetings",
        "alternateGreetings",
    ):
        for value in _normalize_greeting_values(character.get(field_name)):
            if value in seen:
                continue
            seen.add(value)
            greetings.append(value)
    return greetings


def compute_character_greetings_checksum(character: Mapping[str, Any]) -> str:
    """Compute the persisted staleness checksum from one character source view."""
    joined = "\n---\n".join(collect_character_greeting_texts(character))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def _select_creation_greeting(
    character: Mapping[str, Any],
    *,
    strategy: str,
    alternate_index: int | None,
    random_value: float,
) -> dict[str, Any]:
    greeting_text = character.get("first_message")
    source = "first_message"
    source_index = 0
    alternates = _decode_json(character.get("alternate_greetings"), [])
    if strategy in {"alternate_random", "alternate_index"} and isinstance(
        alternates,
        list,
    ) and alternates:
        if strategy == "alternate_random":
            selected_index = min(int(random_value * len(alternates)), len(alternates) - 1)
            greeting_text = alternates[selected_index]
            source_index = alternates.index(greeting_text)
            source = "alternate"
        elif isinstance(alternate_index, int) and alternate_index < len(alternates):
            greeting_text = alternates[alternate_index]
            source_index = alternate_index
            source = "alternate"
    if not isinstance(greeting_text, str) or not greeting_text.strip():
        return {"content": "", "source": "none", "source_index": 0}
    return {
        "content": greeting_text,
        "source": source,
        "source_index": source_index,
    }


def _character_sampling(character: Mapping[str, Any]) -> dict[str, Any]:
    resolved = resolve_character_generation_settings(dict(character))
    return {
        "temperature": resolved.get("temperature", 0.7),
        "top_p": resolved.get("top_p", 1.0),
        "repetition_penalty": resolved.get("repetition_penalty", 1.0),
        "stop": resolved.get("stop", []),
    }


def _select_rows(conn: Any, query: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
    result = conn.execute(query, params)
    return [_row_dict(row, result) for row in result.fetchall()]


def _source_byte_expression(conn: Any, columns: Sequence[str]) -> str:
    if _is_postgres_connection(conn):
        parts = [
            f"octet_length(COALESCE(CAST({column} AS TEXT), ''))"
            for column in columns
        ]
    else:
        parts = [
            f"length(CAST(COALESCE({column}, '') AS BLOB))"
            for column in columns
        ]
    return " + ".join(parts)


def _source_collection_stats(
    conn: Any,
    *,
    table: str,
    foreign_key: str,
    source_ids: Sequence[int],
    text_columns: Sequence[str],
    extra_predicate: str = "",
) -> tuple[int, int]:
    if not source_ids:
        return 0, 0
    placeholders = ",".join("?" for _ in source_ids)
    byte_expression = _source_byte_expression(conn, text_columns)
    query = (  # noqa: S608 - identifiers are closed internal constants.
        "SELECT COUNT(*) AS row_count, "  # nosec B608
        f"COALESCE(SUM({byte_expression}), 0) AS byte_count "
        f"FROM {table} WHERE {foreign_key} IN ({placeholders}) {extra_predicate}"
    )
    result = conn.execute(query, tuple(source_ids))
    row = result.fetchone()
    if row is None:
        return 0, 0
    record = _row_dict(row, result)
    row_count = int(record.get("row_count") or 0)
    source_bytes = int(record.get("byte_count") or 0)
    return (
        row_count,
        source_bytes + row_count * _MATERIALIZED_SOURCE_ROW_OVERHEAD_BYTES,
    )


def _enforce_source_collection_budget(
    *,
    label: str,
    row_count: int,
    byte_count: int,
    max_rows: int,
    max_bytes: int,
) -> None:
    if row_count > max_rows:
        raise InputError(
            f"Materialized {label} exceed the total count limit of {max_rows}."
        )
    if byte_count > max_bytes:
        raise InputError(
            f"Materialized {label} exceed the total byte limit of {max_bytes}."
        )


def _load_preset(
    conn: Any,
    preset_id: str,
    *,
    selection_source: str,
    owner_user_id: str,
) -> dict[str, Any]:
    """Load a preset through the backend's actual ownership boundary."""
    normalized = str(preset_id).strip()
    if normalized in {"default", "st_default"}:
        builtin = next(
            preset
            for preset in get_builtin_presets()
            if preset["preset_id"] == normalized
        )
        return {
            **_safe_json(builtin),
            "version": 1,
            "selection_source": selection_source,
            "source": {
                "kind": "builtin_prompt_preset",
                "id": normalized,
                "version": 1,
            },
        }
    if _is_postgres_connection(conn):
        result = conn.execute(
            """
            SELECT preset_id, name, section_order_json, section_templates_json,
                   last_modified, version
              FROM prompt_presets
             WHERE preset_id = ? AND client_id = ? AND deleted = FALSE
             LIMIT 1
            """,
            (normalized, owner_user_id),
        )
    else:
        result = conn.execute(
            """
            SELECT preset_id, name, section_order_json, section_templates_json,
                   last_modified, version
              FROM prompt_presets
             WHERE preset_id = ? AND deleted = FALSE
             LIMIT 1
            """,
            (normalized,),
        )
    row = result.fetchone()
    if row is None:
        raise InputError(f"Prompt preset '{normalized}' not found.")
    record = _row_dict(row, result)
    version = int(record.get("version") or 1)
    return {
        "preset_id": record["preset_id"],
        "name": record["name"],
        "builtin": False,
        "version": version,
        "updated_at": str(record.get("last_modified") or ""),
        "section_order": _decode_json(record.get("section_order_json"), []),
        "section_templates": _decode_json(record.get("section_templates_json"), {}),
        "selection_source": selection_source,
        "source": {
            "kind": "prompt_preset",
            "id": str(record["preset_id"]),
            "version": version,
        },
    }


def _load_world_books_for_participants(
    conn: Any,
    participant_character_ids: Sequence[int],
    *,
    owner_user_id: str,
    preexisting_world_book_ids: Sequence[int] = (),
    preexisting_world_books: Mapping[int, Mapping[str, Any]] | None = None,
    existing_entry_count: int = 0,
    existing_entry_bytes: int = 0,
) -> dict[int, list[dict[str, Any]]]:
    """Load bounded participant lore, expanding each shared book only once."""
    frozen_books = {
        int(book_id): dict(book)
        for book_id, book in (preexisting_world_books or {}).items()
    }
    raw_by_character: dict[int, list[dict[str, Any]]] = {}
    unique_books: dict[int, dict[str, Any]] = {}
    for character_id in participant_character_ids:
        if _is_postgres_connection(conn):
            books = _select_rows(
                conn,
                """
                SELECT wb.*, cwb.enabled AS attachment_enabled,
                       cwb.priority AS attachment_priority
                  FROM world_books wb
                  JOIN character_world_books cwb ON cwb.world_book_id = wb.id
                  JOIN character_cards cc ON cc.id = cwb.character_id
                 WHERE cwb.character_id = ? AND cc.client_id = ?
                   AND cc.deleted = FALSE AND wb.deleted = FALSE
                 ORDER BY cwb.priority DESC, wb.name, wb.id
                 LIMIT 65
                """,
                (character_id, owner_user_id),
            )
        else:
            books = _select_rows(
                conn,
                """
                SELECT wb.*, cwb.enabled AS attachment_enabled,
                       cwb.priority AS attachment_priority
                  FROM world_books wb
                  JOIN character_world_books cwb ON cwb.world_book_id = wb.id
                 WHERE cwb.character_id = ? AND wb.deleted = FALSE
                 ORDER BY cwb.priority DESC, wb.name, wb.id
                 LIMIT 65
                """,
                (character_id,),
            )
        raw_by_character[int(character_id)] = books
        for book in books:
            world_book_id = int(book["id"])
            unique_books.setdefault(world_book_id, book)
        if len({*preexisting_world_book_ids, *unique_books}) > MAX_MATERIALIZED_WORLD_BOOKS:
            raise InputError(
                f"A resumable chat supports at most {MAX_MATERIALIZED_WORLD_BOOKS} world books."
            )

    combined_world_book_ids = {
        *preexisting_world_book_ids,
        *unique_books,
    }
    if len(combined_world_book_ids) > MAX_MATERIALIZED_WORLD_BOOKS:
        raise InputError(
            f"A resumable chat supports at most {MAX_MATERIALIZED_WORLD_BOOKS} world books."
        )

    new_book_ids = [book_id for book_id in unique_books if book_id not in frozen_books]
    entries_by_book = _load_world_book_entries(
        conn,
        new_book_ids,
        existing_row_count=existing_entry_count,
        existing_byte_count=existing_entry_bytes,
    )

    materialized_by_character: dict[int, list[dict[str, Any]]] = {}
    for character_id, books in raw_by_character.items():
        materialized: list[dict[str, Any]] = []
        for book in books:
            world_book_id = int(book["id"])
            if world_book_id in frozen_books:
                clean_book = dict(frozen_books[world_book_id])
                clean_book["attachment_enabled"] = _safe_json(
                    book.get("attachment_enabled")
                )
                clean_book["attachment_priority"] = _safe_json(
                    book.get("attachment_priority")
                )
            else:
                clean_book = {
                    key: _safe_json(value)
                    for key, value in book.items()
                    if key not in {"client_id", "deleted"}
                }
                clean_book["entries"] = entries_by_book[world_book_id]
            materialized.append(clean_book)
        materialized_by_character[character_id] = materialized
    return materialized_by_character


def _load_world_book_entries(
    conn: Any,
    world_book_ids: Sequence[int],
    *,
    existing_row_count: int = 0,
    existing_byte_count: int = 0,
) -> dict[int, list[dict[str, Any]]]:
    ordered_ids = list(dict.fromkeys(int(item) for item in world_book_ids))
    if not ordered_ids:
        return {}
    text_columns = ("keywords", "content", "metadata")
    row_count, byte_count = _source_collection_stats(
        conn,
        table="world_book_entries",
        foreign_key="world_book_id",
        source_ids=ordered_ids,
        text_columns=text_columns,
    )
    total_rows = existing_row_count + row_count
    total_bytes = existing_byte_count + byte_count
    _enforce_source_collection_budget(
        label="world-book entries",
        row_count=total_rows,
        byte_count=total_bytes,
        max_rows=MAX_MATERIALIZED_WORLD_BOOK_ENTRIES,
        max_bytes=MAX_MATERIALIZED_WORLD_BOOK_ENTRY_BYTES,
    )

    placeholders = ",".join("?" for _ in ordered_ids)
    byte_expression = _source_byte_expression(conn, text_columns)
    # IDs are parameters; table names, projections, and limits are closed constants.
    query = (  # noqa: S608 - only closed SQL and parameter placeholders vary.
        "WITH source_stats AS ("  # nosec B608
        "SELECT COUNT(*) AS row_count, "
        f"COALESCE(SUM({byte_expression}), 0) "
        f"+ COUNT(*) * {_MATERIALIZED_SOURCE_ROW_OVERHEAD_BYTES} AS byte_count "
        "FROM world_book_entries "
        f"WHERE world_book_id IN ({placeholders})"
        ") "
        "SELECT entry.id, entry.world_book_id, entry.keywords, entry.content, "
        "entry.priority, entry.enabled, entry.case_sensitive, entry.regex_match, "
        "entry.whole_word_match, entry.metadata, entry.created_at, "
        "entry.last_modified FROM world_book_entries entry "
        "CROSS JOIN source_stats stats "
        f"WHERE entry.world_book_id IN ({placeholders}) "
        "AND stats.row_count + ? <= ? AND stats.byte_count + ? <= ? "
        "ORDER BY entry.world_book_id, entry.priority DESC, entry.id "
        f"LIMIT {MAX_MATERIALIZED_WORLD_BOOK_ENTRIES + 1}"
    )
    entries = _select_rows(
        conn,
        query,
        (
            *ordered_ids,
            *ordered_ids,
            existing_row_count,
            MAX_MATERIALIZED_WORLD_BOOK_ENTRIES,
            existing_byte_count,
            MAX_MATERIALIZED_WORLD_BOOK_ENTRY_BYTES,
        ),
    )
    current_count, current_bytes = _source_collection_stats(
        conn,
        table="world_book_entries",
        foreign_key="world_book_id",
        source_ids=ordered_ids,
        text_columns=text_columns,
    )
    _enforce_source_collection_budget(
        label="world-book entries",
        row_count=existing_row_count + current_count,
        byte_count=existing_byte_count + current_bytes,
        max_rows=MAX_MATERIALIZED_WORLD_BOOK_ENTRIES,
        max_bytes=MAX_MATERIALIZED_WORLD_BOOK_ENTRY_BYTES,
    )
    if len(entries) != current_count:
        raise _SourceDrift

    entries_by_book: dict[int, list[dict[str, Any]]] = {
        world_book_id: [] for world_book_id in ordered_ids
    }
    for entry in entries:
        world_book_id = int(entry["world_book_id"])
        entries_by_book[world_book_id].append(
            {
                key: _safe_json(
                    _decode_json(value, {} if key == "metadata" else [])
                    if key in {"keywords", "metadata"}
                    else value
                )
                for key, value in entry.items()
                if key not in {"client_id", "deleted"}
            }
        )
    return entries_by_book


def _load_exemplars_for_participants(
    conn: Any,
    participant_character_ids: Sequence[int],
    *,
    existing_row_count: int = 0,
    existing_byte_count: int = 0,
) -> dict[int, list[dict[str, Any]]]:
    ordered_ids = list(
        dict.fromkeys(int(item) for item in participant_character_ids)
    )
    if not ordered_ids:
        return {}
    text_columns = _MATERIALIZED_EXEMPLAR_TEXT_FIELDS
    row_count, byte_count = _source_collection_stats(
        conn,
        table="character_exemplars",
        foreign_key="character_id",
        source_ids=ordered_ids,
        text_columns=text_columns,
        extra_predicate="AND is_deleted = FALSE",
    )
    _enforce_source_collection_budget(
        label="exemplars",
        row_count=existing_row_count + row_count,
        byte_count=existing_byte_count + byte_count,
        max_rows=MAX_MATERIALIZED_EXEMPLARS,
        max_bytes=MAX_MATERIALIZED_EXEMPLAR_BYTES,
    )

    placeholders = ",".join("?" for _ in ordered_ids)
    byte_expression = _source_byte_expression(conn, text_columns)
    # IDs are parameters; table names, projections, and limits are closed constants.
    query = (  # noqa: S608 - only closed SQL and parameter placeholders vary.
        "WITH source_stats AS ("  # nosec B608
        "SELECT COUNT(*) AS row_count, "
        f"COALESCE(SUM({byte_expression}), 0) "
        f"+ COUNT(*) * {_MATERIALIZED_SOURCE_ROW_OVERHEAD_BYTES} AS byte_count "
        "FROM character_exemplars "
        f"WHERE character_id IN ({placeholders}) AND is_deleted = FALSE"
        ") "
        "SELECT exemplar.id, exemplar.character_id, exemplar.text, "
        "exemplar.source_type, exemplar.source_url_or_id, exemplar.source_date, "
        "exemplar.novelty_hint, exemplar.emotion, exemplar.scenario, "
        "exemplar.rhetorical, exemplar.register, exemplar.safety_allowed, "
        "exemplar.safety_blocked, exemplar.rights_public_figure, "
        "exemplar.rights_notes, exemplar.length_tokens, exemplar.created_at, "
        "exemplar.updated_at FROM character_exemplars exemplar "
        "CROSS JOIN source_stats stats "
        f"WHERE exemplar.character_id IN ({placeholders}) "
        "AND exemplar.is_deleted = FALSE "
        "AND stats.row_count + ? <= ? AND stats.byte_count + ? <= ? "
        "ORDER BY exemplar.character_id, exemplar.updated_at DESC, "
        "exemplar.created_at DESC, exemplar.id "
        f"LIMIT {MAX_MATERIALIZED_EXEMPLARS + 1}"
    )
    rows = _select_rows(
        conn,
        query,
        (
            *ordered_ids,
            *ordered_ids,
            existing_row_count,
            MAX_MATERIALIZED_EXEMPLARS,
            existing_byte_count,
            MAX_MATERIALIZED_EXEMPLAR_BYTES,
        ),
    )
    current_count, current_bytes = _source_collection_stats(
        conn,
        table="character_exemplars",
        foreign_key="character_id",
        source_ids=ordered_ids,
        text_columns=text_columns,
        extra_predicate="AND is_deleted = FALSE",
    )
    _enforce_source_collection_budget(
        label="exemplars",
        row_count=existing_row_count + current_count,
        byte_count=existing_byte_count + current_bytes,
        max_rows=MAX_MATERIALIZED_EXEMPLARS,
        max_bytes=MAX_MATERIALIZED_EXEMPLAR_BYTES,
    )
    if len(rows) != current_count:
        raise _SourceDrift

    by_character: dict[int, list[dict[str, Any]]] = {
        character_id: [] for character_id in ordered_ids
    }
    for row in rows:
        character_id = int(row["character_id"])
        by_character[character_id].append(
            {
                key: _safe_json(
                    _decode_json(value, [])
                    if key in {"rhetorical", "safety_allowed", "safety_blocked"}
                    else value
                )
                for key, value in row.items()
                if key not in {"character_id", "is_deleted"}
            }
        )
    return by_character


def _load_exemplars(conn: Any, character_id: int) -> list[dict[str, Any]]:
    return _load_exemplars_for_participants(conn, [character_id])[character_id]


def _load_persona_memory_entries(
    conn: Any,
    *,
    owner_user_id: str,
    character_id: int,
) -> list[dict[str, Any]]:
    rows = _select_rows(
        conn,
        """
        SELECT id, memory_type, content, source_conversation_id,
               salience, last_modified, version
          FROM persona_memory_entries
         WHERE user_id = ? AND persona_id = ?
           AND archived = FALSE AND deleted = FALSE
         ORDER BY last_modified DESC, id ASC
         LIMIT 200
        """,
        (owner_user_id, f"char:{character_id}"),
    )
    rows.sort(
        key=lambda row: (
            float(row.get("salience") or 0.0),
            str(row.get("last_modified") or ""),
        ),
        reverse=True,
    )
    return [
        {
            "id": str(row.get("id") or ""),
            "memory_type": str(row.get("memory_type") or "manual"),
            "content": str(row.get("content") or ""),
            "source_conversation_id": (
                str(row["source_conversation_id"])
                if row.get("source_conversation_id") is not None
                else None
            ),
            "salience": float(row.get("salience") or 0.0),
            "last_modified": str(row.get("last_modified") or ""),
            "version": int(row.get("version") or 1),
        }
        for row in rows
        if str(row.get("content") or "").strip()
    ]


def _aliases(character: Mapping[str, Any]) -> list[str]:
    extensions = _decode_json(character.get("extensions"), {})
    candidates: Any = extensions.get("aliases") if isinstance(extensions, dict) else None
    if not isinstance(candidates, list) and isinstance(extensions, dict):
        tldw = extensions.get("tldw")
        candidates = tldw.get("aliases") if isinstance(tldw, dict) else None
    return [str(item) for item in candidates] if isinstance(candidates, list) else []


def _greeting_for(
    character: Mapping[str, Any],
    selected_greeting: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if selected_greeting is not None:
        return {
            "content": str(selected_greeting.get("content") or ""),
            "source": str(selected_greeting.get("source") or "first_message"),
            "source_index": int(selected_greeting.get("source_index") or 0),
        }
    return {
        "content": str(character.get("first_message") or ""),
        "source": "first_message" if character.get("first_message") else "none",
        "source_index": 0,
    }


def _materialize_behavior(
    conn: Any,
    *,
    participant_character_ids: Sequence[int],
    prompt_preset_id: str | None,
    memory_by_character_id: Mapping[str, str],
    primary_greeting: Mapping[str, Any] | None,
    owner_user_id: str,
    max_snapshot_bytes: int,
    greeting_strategy: str = "default",
    greeting_alternate_index: int | None = None,
    greeting_random_value: float = 0.0,
    fallback_greeting: Mapping[str, Any] | None = None,
    preexisting_world_book_ids: Sequence[int] = (),
    preexisting_world_books: Mapping[int, Mapping[str, Any]] | None = None,
    existing_exemplar_count: int = 0,
    existing_exemplar_bytes: int = 0,
    existing_world_book_entry_count: int = 0,
    existing_world_book_entry_bytes: int = 0,
) -> _MaterializedBehavior:
    requested_preset = str(prompt_preset_id or "").strip() or None
    participants: list[dict[str, Any]] = []
    primary_character: dict[str, Any] | None = None
    characters: dict[int, dict[str, Any]] = {}
    for character_id in participant_character_ids:
        if _is_postgres_connection(conn):
            result = conn.execute(
                "SELECT * FROM character_cards "
                "WHERE id = ? AND client_id = ? AND deleted = FALSE LIMIT 1",
                (character_id, owner_user_id),
            )
        else:
            result = conn.execute(
                "SELECT * FROM character_cards "
                "WHERE id = ? AND deleted = FALSE LIMIT 1",
                (character_id,),
            )
        row = result.fetchone()
        if row is None:
            raise InputError(f"Character ID {character_id} not found.")
        character = _row_dict(row, result)
        character["alternate_greetings"] = _decode_json(
            character.get("alternate_greetings"), []
        )
        character["extensions"] = _decode_json(character.get("extensions"), {})
        characters[int(character_id)] = character
    world_books_by_character = _load_world_books_for_participants(
        conn,
        participant_character_ids,
        owner_user_id=owner_user_id,
        preexisting_world_book_ids=preexisting_world_book_ids,
        preexisting_world_books=preexisting_world_books,
        existing_entry_count=existing_world_book_entry_count,
        existing_entry_bytes=existing_world_book_entry_bytes,
    )
    exemplars_by_character = _load_exemplars_for_participants(
        conn,
        participant_character_ids,
        existing_row_count=existing_exemplar_count,
        existing_byte_count=existing_exemplar_bytes,
    )
    for index, character_id in enumerate(participant_character_ids):
        character = characters[int(character_id)]
        resolved_preset = requested_preset or resolve_character_prompt_preset(character)
        selection_source = "default"
        if requested_preset:
            selection_source = "creation_request"
        elif _character_prompt_preset_is_explicit(character, resolved_preset):
            selection_source = "character"
        preset = _load_preset(
            conn,
            resolved_preset,
            selection_source=selection_source,
            owner_user_id=owner_user_id,
        )
        if primary_character is None:
            primary_character = character
        extensions = _safe_json(character.get("extensions") or {})
        prompt_extensions: dict[str, Any] = {"prompt_preset": _safe_json(preset)}
        if isinstance(extensions, dict):
            prompt_extensions["character_extensions"] = extensions
        memory = memory_by_character_id.get(str(character_id))
        persona_memory_entries = _load_persona_memory_entries(
            conn,
            owner_user_id=owner_user_id,
            character_id=character_id,
        )
        source = {
            "kind": "character",
            "id": str(character_id),
            "version": int(character.get("version") or 1),
        }
        selected_greeting = (
            primary_greeting
            or _select_creation_greeting(
                character,
                strategy=greeting_strategy,
                alternate_index=greeting_alternate_index,
                random_value=greeting_random_value,
            )
        )
        if (
            index == 0
            and not str(selected_greeting.get("content") or "").strip()
            and fallback_greeting is not None
        ):
            selected_greeting = dict(fallback_greeting)
        participants.append(
            {
                "source": source,
                "identity": {
                    "name": str(character.get("name") or "Character"),
                    "aliases": _aliases(character),
                },
                "prompt": {
                    "system_prompt": str(character.get("system_prompt") or ""),
                    "description": str(character.get("description") or ""),
                    "personality": str(character.get("personality") or ""),
                    "scenario": str(character.get("scenario") or ""),
                    "message_example": str(character.get("message_example") or ""),
                    "post_history_instructions": str(
                        character.get("post_history_instructions") or ""
                    ),
                    "prompt_relevant_extensions": prompt_extensions,
                },
                "greeting": _greeting_for(
                    character,
                    selected_greeting if index == 0 else None,
                ),
                "generation_defaults": {
                    "source": source,
                    "sampling": _character_sampling(character),
                },
                "exemplars": exemplars_by_character.get(character_id, []),
                "world_books": world_books_by_character.get(character_id, []),
                "default_memory": (
                    {
                        "content": memory if isinstance(memory, str) else "",
                        "source": (
                            "creation_request"
                            if isinstance(memory, str)
                            else "persona_memory_entries"
                        ),
                        "version": 1,
                        "persona_memory_entries": persona_memory_entries,
                    }
                    if isinstance(memory, str) or persona_memory_entries
                    else None
                ),
            }
        )
    if primary_character is None:
        raise InputError("At least one character participant is required.")
    snapshot = build_behavior_snapshot(
        {
            "schema_version": 1,
            "participants": participants,
            "routing_defaults": {"turn_taking_mode": "single"},
        },
        max_bytes=max_snapshot_bytes,
    )
    return _MaterializedBehavior(snapshot=snapshot, primary_character=primary_character)


def _normalize_sampling(
    sampling: Mapping[str, Any] | None,
    character: Mapping[str, Any],
) -> dict[str, Any] | None:
    if sampling is None:
        candidate = _character_sampling(character)
    else:
        if not set(sampling).issubset(_SAMPLING_FIELDS):
            return None
        candidate = {**_character_sampling(character), **sampling}
    limits = {
        "temperature": (0.0, 2.0),
        "top_p": (0.0, 1.0),
        "repetition_penalty": (0.0, 3.0),
    }
    normalized: dict[str, Any] = {}
    for key, (minimum, maximum) in limits.items():
        value = candidate.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        value = float(value)
        if not math.isfinite(value) or not minimum <= value <= maximum:
            return None
        normalized[key] = value
    stop = candidate.get("stop")
    if not isinstance(stop, list) or len(stop) > 64 or any(
        not isinstance(item, str) for item in stop
    ):
        return None
    normalized["stop"] = list(stop)
    return normalized


def _resolve_effective_completion(
    *,
    provider: str | None,
    model: str | None,
    sampling: Mapping[str, Any] | None,
    character: Mapping[str, Any],
    app_config: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    configured_provider = ""
    for section_name in ("llm_api_settings", "API"):
        section = app_config.get(section_name)
        if isinstance(section, Mapping):
            configured_provider = str(
                section.get("default_api") or section.get("default_provider") or ""
            ).strip()
            if configured_provider:
                break
    if not configured_provider:
        configured_provider = str(
            app_config.get("default_api") or app_config.get("default_provider") or ""
        ).strip()
    provider_candidate = str(
        provider
        or configured_provider
        or os.getenv("DEFAULT_LLM_PROVIDER")
        or "openai"
    ).strip()
    registry = get_registry()
    resolved_provider = registry.resolve_provider_name(
        normalize_provider(provider_candidate)
    )
    resolved_model_value = model or (
        resolve_provider_model(resolved_provider, app_config)
        if resolved_provider
        else None
    )
    resolved_model = (
        resolved_model_value.strip()
        if isinstance(resolved_model_value, str)
        else ""
    )
    normalized_sampling = _normalize_sampling(sampling, character)
    adapter = registry.get_adapter(resolved_provider) if resolved_provider else None
    model_known = (
        is_model_known_for_provider(resolved_provider, resolved_model)
        if resolved_provider and resolved_model
        else None
    )
    if (
        not resolved_provider
        or adapter is None
        or not resolved_model
        or model_known is False
        or normalized_sampling is None
    ):
        return None, "incomplete_effective_settings"
    return {
        "provider": resolved_provider,
        "model": resolved_model,
        "sampling": normalized_sampling,
    }, None


def _snapshot_participants(resume_state: Mapping[str, Any]) -> list[dict[str, Any]]:
    snapshot = resume_state.get("behavior_snapshot")
    payload = snapshot.get("payload") if isinstance(snapshot, Mapping) else None
    participants = payload.get("participants") if isinstance(payload, Mapping) else None
    return [dict(item) for item in participants or [] if isinstance(item, Mapping)]


def _source_id(value: Any) -> str | None:
    if not isinstance(value, Mapping):
        return None
    source = value.get("source")
    if not isinstance(source, Mapping) or source.get("id") is None:
        return None
    return str(source["id"])


def _materialized_world_book_entry_usage(
    world_books: Sequence[Mapping[str, Any]],
) -> tuple[int, int]:
    seen_book_ids: set[int] = set()
    row_count = 0
    byte_count = 0
    for book in world_books:
        try:
            world_book_id = int(book["id"])
        except (KeyError, TypeError, ValueError):
            continue
        if world_book_id in seen_book_ids:
            continue
        seen_book_ids.add(world_book_id)
        entries = book.get("entries")
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            row_count += 1
            for key in ("keywords", "content", "metadata"):
                value = entry.get(key)
                if isinstance(value, (Mapping, list, tuple)):
                    encoded = json.dumps(
                        _safe_json(value),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                else:
                    encoded = str(value or "").encode("utf-8")
                byte_count += len(encoded)
            byte_count += _MATERIALIZED_SOURCE_ROW_OVERHEAD_BYTES
    return row_count, byte_count


def _materialized_exemplar_usage(
    participants: Sequence[Mapping[str, Any]],
) -> tuple[int, int]:
    seen_exemplars: set[tuple[str, str]] = set()
    row_count = 0
    byte_count = 0
    for participant in participants:
        participant_id = _source_id(participant) or ""
        exemplars = participant.get("exemplars")
        if not isinstance(exemplars, list):
            continue
        for exemplar in exemplars:
            if not isinstance(exemplar, Mapping):
                continue
            exemplar_id = str(exemplar.get("id") or "")
            identity = (participant_id, exemplar_id)
            if identity in seen_exemplars:
                continue
            seen_exemplars.add(identity)
            row_count += 1
            for key in _MATERIALIZED_EXEMPLAR_TEXT_FIELDS:
                value = exemplar.get(key)
                if isinstance(value, (Mapping, list, tuple)):
                    encoded = json.dumps(
                        _safe_json(value),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                else:
                    encoded = str(value or "").encode("utf-8")
                byte_count += len(encoded)
            byte_count += _MATERIALIZED_SOURCE_ROW_OVERHEAD_BYTES
    return row_count, byte_count


def _materialize_world_books_by_id(
    conn: Any,
    world_book_ids: Sequence[int],
    *,
    owner_user_id: str,
    participant_character_ids: Sequence[int],
    existing_entry_count: int = 0,
    existing_entry_bytes: int = 0,
) -> dict[int, dict[str, Any]]:
    """Materialize explicitly addressed books through the backend auth boundary."""
    ordered_ids = list(dict.fromkeys(int(item) for item in world_book_ids))
    books: dict[int, dict[str, Any]] = {}
    for world_book_id in ordered_ids:
        if _is_postgres_connection(conn):
            placeholders = ",".join("?" for _ in participant_character_ids)
            if not placeholders:
                raise InputError(f"World book ID {world_book_id} not found.")
            authorized = conn.execute(
                "SELECT 1 AS authorized FROM character_world_books cwb "
                "JOIN character_cards cc ON cc.id = cwb.character_id "
                "WHERE cwb.world_book_id = ? "
                f"AND cwb.character_id IN ({placeholders}) "  # nosec B608
                "AND cc.client_id = ? AND cc.deleted = FALSE LIMIT 1",
                (world_book_id, *participant_character_ids, owner_user_id),
            ).fetchone()
            if authorized is None:
                raise InputError(f"World book ID {world_book_id} not found.")
        result = conn.execute(
            "SELECT * FROM world_books WHERE id = ? AND deleted = FALSE LIMIT 1",
            (world_book_id,),
        )
        row = result.fetchone()
        if row is None:
            raise InputError(f"World book ID {world_book_id} not found.")
        books[world_book_id] = _row_dict(row, result)

    entries_by_book = _load_world_book_entries(
        conn,
        ordered_ids,
        existing_row_count=existing_entry_count,
        existing_byte_count=existing_entry_bytes,
    )
    return {
        world_book_id: {
            **{
                key: _safe_json(value)
                for key, value in book.items()
                if key not in {"client_id", "deleted"}
            },
            "entries": entries_by_book[world_book_id],
        }
        for world_book_id, book in books.items()
    }


def _materialize_world_book_by_id(
    conn: Any,
    world_book_id: int,
    *,
    owner_user_id: str,
    participant_character_ids: Sequence[int],
) -> dict[str, Any]:
    """Materialize one explicitly addressed book through the shared bounded path."""
    return _materialize_world_books_by_id(
        conn,
        [world_book_id],
        owner_user_id=owner_user_id,
        participant_character_ids=participant_character_ids,
    )[world_book_id]


def _find_materialized_preset(
    preset_id: str,
    *,
    current: Any,
    participants: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    candidates: list[Any] = [current]
    for participant in participants:
        prompt = participant.get("prompt")
        extensions = (
            prompt.get("prompt_relevant_extensions")
            if isinstance(prompt, Mapping)
            else None
        )
        if isinstance(extensions, Mapping):
            candidates.append(extensions.get("prompt_preset"))
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        if candidate.get("preset_id") == preset_id or _source_id(candidate) == preset_id:
            return dict(candidate)
    return None


def _frozen_character_prompt_preset(
    participants: Sequence[Mapping[str, Any]],
) -> tuple[str, str]:
    """Resolve character-scope preset authority from frozen character extensions."""
    primary = participants[0] if participants else {}
    prompt = primary.get("prompt") if isinstance(primary, Mapping) else None
    prompt_extensions = (
        prompt.get("prompt_relevant_extensions")
        if isinstance(prompt, Mapping)
        else None
    )
    character_extensions = (
        prompt_extensions.get("character_extensions")
        if isinstance(prompt_extensions, Mapping)
        else None
    )
    frozen_character = {
        "extensions": (
            dict(character_extensions)
            if isinstance(character_extensions, Mapping)
            else {}
        )
    }
    preset_id = resolve_character_prompt_preset(frozen_character)
    selection_source = (
        "character"
        if _character_prompt_preset_is_explicit(frozen_character, preset_id)
        else "default"
    )
    return preset_id, selection_source


def _materialize_overlay(
    conn: Any,
    overlay: Mapping[str, Any],
    *,
    owner_user_id: str,
    participants: Sequence[Mapping[str, Any]],
    current_overlay: Any,
) -> dict[str, Any]:
    kind = str(overlay.get("kind") or "")
    source_id = str(overlay.get("id") or "")
    current_source = (
        current_overlay.get("source") if isinstance(current_overlay, Mapping) else None
    )
    if (
        isinstance(current_source, Mapping)
        and current_source.get("kind") == kind
        and str(current_source.get("id")) == source_id
    ):
        source = dict(current_source)
    elif kind == "character":
        participant = next(
            (item for item in participants if _source_id(item) == source_id),
            None,
        )
        if participant is not None:
            source = dict(participant["source"])
        else:
            try:
                character_id = int(source_id)
            except (TypeError, ValueError) as exc:
                raise InputError("Assistant overlay character ID is invalid.") from exc
            result = conn.execute(
                "SELECT id, version FROM character_cards "
                "WHERE id = ? AND deleted = FALSE LIMIT 1",
                (character_id,),
            )
            row = result.fetchone()
            if row is None:
                raise InputError(f"Assistant overlay character '{source_id}' not found.")
            record = _row_dict(row, result)
            source = {
                "kind": kind,
                "id": str(record["id"]),
                "version": int(record.get("version") or 1),
            }
    else:
        # Persona profiles share a database, so the owner predicate is mandatory.
        result = conn.execute(
            """
            SELECT id, version FROM persona_profiles
             WHERE id = ? AND user_id = ? AND deleted = FALSE
             LIMIT 1
            """,
            (source_id, owner_user_id),
        )
        row = result.fetchone()
        if row is None:
            raise InputError(f"Assistant overlay {kind} '{source_id}' not found.")
        record = _row_dict(row, result)
        source = {
            "kind": kind,
            "id": str(record["id"]),
            "version": int(record.get("version") or 1),
        }
    return {
        "source": source,
        "name": str(overlay.get("name") or ""),
        "system_prompt": str(overlay.get("system_prompt_snapshot") or ""),
    }


def _participant_ids_from_settings(
    settings: Mapping[str, Any],
    primary_character_id: Any,
) -> list[int]:
    raw_ids = settings.get("participantCharacterIds")
    if raw_ids is None:
        raw_ids = settings.get("participant_character_ids")
    if isinstance(raw_ids, str):
        try:
            decoded = json.loads(raw_ids)
        except json.JSONDecodeError:
            decoded = [part.strip() for part in raw_ids.split(",") if part.strip()]
        raw_ids = decoded
    if not isinstance(raw_ids, list):
        raw_ids = []
    return normalize_materialized_participant_ids(primary_character_id, raw_ids)


def _selected_greeting_from_character(
    character: Mapping[str, Any],
    selection_id: Any,
) -> dict[str, Any] | None:
    if selection_id is None:
        return None
    if not isinstance(selection_id, str):
        raise InputError("Greeting selection ID is invalid.")
    normalized_selection_id = selection_id.strip()
    parts = normalized_selection_id.split(":")
    if len(parts) < 3 or parts[0] != "greeting":
        raise InputError("Greeting selection ID is invalid.")
    try:
        index = int(parts[1])
    except (IndexError, ValueError):
        raise InputError("Greeting selection ID is invalid.") from None
    greetings = collect_character_greeting_texts(character)
    if index < 0 or index >= len(greetings):
        raise InputError(f"Greeting index {index} is out of range.")
    first_messages = _normalize_greeting_values(character.get("first_message"))
    from_first_message = bool(first_messages and greetings[index] == first_messages[0])
    return {
        "content": greetings[index],
        "selection_id": normalized_selection_id,
        "source": "first_message" if from_first_message else "alternate_greeting",
        "source_index": 0 if from_first_message else index,
        "character_version": int(character.get("version") or 1),
    }


def _resolve_selected_greeting(
    conn: Any,
    *,
    character_id: int,
    selection_id: Any,
) -> dict[str, Any] | None:
    result = conn.execute(
        "SELECT first_message, alternate_greetings, version FROM character_cards "
        "WHERE id = ? AND deleted = FALSE LIMIT 1",
        (character_id,),
    )
    row = result.fetchone()
    if row is None:
        raise InputError(f"Character ID {character_id} not found.")
    return _selected_greeting_from_character(_row_dict(row, result), selection_id)


def _pending_greeting_digest(values: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        {"schemaVersion": 1, "values": dict(values)},
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(canonical) > DEFAULT_MAX_SNAPSHOT_BYTES:
        raise InputError("Pending greeting authority exceeds the size limit.")
    return f"sha256:{hashlib.sha256(canonical).hexdigest()}"


def build_pending_greeting_authority(
    conn: Any,
    *,
    conversation: Mapping[str, Any],
    resume_state: Mapping[str, Any],
    selection_id: str,
) -> dict[str, Any]:
    """Freeze a selected greeting while completion authority is incomplete."""
    snapshot = resume_state.get("behavior_snapshot")
    if not isinstance(snapshot, Mapping) or snapshot.get("status") != "valid":
        raise InputError("Conversation has no valid behavior snapshot.")
    character_id = conversation.get("character_id")
    result = conn.execute(
        "SELECT first_message, alternate_greetings, version FROM character_cards "
        "WHERE id = ? AND deleted = FALSE LIMIT 1",
        (int(character_id),),
    )
    row = result.fetchone()
    if row is None:
        raise InputError(f"Character ID {character_id} not found.")
    character = _row_dict(row, result)
    greeting = _selected_greeting_from_character(character, selection_id)
    if greeting is None:
        raise InputError("Greeting selection ID is invalid.")
    values = {
        "base_snapshot": {
            "schema_version": int(snapshot["schema_version"]),
            "digest": str(snapshot["digest"]),
        },
        "character_id": int(character_id),
        "greetings_checksum": compute_character_greetings_checksum(character),
        "greeting": greeting,
    }
    return {
        "schemaVersion": 1,
        "digest": _pending_greeting_digest(values),
        "values": values,
    }


def _validated_pending_greeting(
    pending: Any,
    *,
    conversation: Mapping[str, Any],
    snapshot: Mapping[str, Any],
    merged_settings: Mapping[str, Any],
) -> dict[str, Any] | None:
    if pending is None:
        return None
    if not isinstance(pending, Mapping) or set(pending) != {
        "schemaVersion",
        "digest",
        "values",
    }:
        raise InputError("Stored pending greeting authority is invalid.")
    values = pending.get("values")
    if not isinstance(values, Mapping) or set(values) != {
        "base_snapshot",
        "character_id",
        "greetings_checksum",
        "greeting",
    }:
        raise InputError("Stored pending greeting authority is invalid.")
    greeting = values.get("greeting")
    if not isinstance(greeting, Mapping) or set(greeting) != {
        "content",
        "selection_id",
        "source",
        "source_index",
        "character_version",
    }:
        raise InputError("Stored pending greeting authority is invalid.")
    if (
        pending.get("schemaVersion") != 1
        or not isinstance(pending.get("digest"), str)
        or not isinstance(values.get("greetings_checksum"), str)
        or not isinstance(greeting.get("content"), str)
        or not isinstance(greeting.get("selection_id"), str)
        or greeting.get("source") not in {"first_message", "alternate_greeting"}
        or type(greeting.get("source_index")) is not int
        or greeting["source_index"] < 0
        or type(greeting.get("character_version")) is not int
        or greeting["character_version"] < 1
    ):
        raise InputError("Stored pending greeting authority is invalid.")
    expected_binding = {
        "schema_version": int(snapshot["schema_version"]),
        "digest": str(snapshot["digest"]),
    }
    if (
        values.get("base_snapshot") != expected_binding
        or values.get("character_id") != int(conversation.get("character_id"))
        or greeting.get("selection_id") != merged_settings.get("greetingSelectionId")
        or values.get("greetings_checksum") != merged_settings.get("greetingsChecksum")
    ):
        raise InputError("Stored pending greeting authority is invalid.")
    expected_digest = _pending_greeting_digest(values)
    if pending.get("digest") != expected_digest:
        raise InputError("Stored pending greeting authority is invalid.")
    return dict(greeting)


def _materialize_roleplay_behavior_settings_once(
    conn: Any,
    *,
    conversation: Mapping[str, Any],
    resume_state: Mapping[str, Any],
    merged_settings: Mapping[str, Any],
    owner_user_id: str,
    changed_keys: set[str] | frozenset[str],
    max_bytes: int = DEFAULT_MAX_SNAPSHOT_BYTES,
) -> dict[str, Any] | None:
    """Rebuild the complete, immutable resumable behavior authority."""
    reject_resumable_behavior_credentials(merged_settings)
    stored_settings = resume_state.get("settings")
    current_materialized = resume_state.get("materialized_settings")
    if (
        isinstance(stored_settings, Mapping)
        and "roleplayBehaviorV1" in stored_settings
        and not isinstance(current_materialized, Mapping)
    ):
        raise InputError("Stored materialized behavior settings are invalid.")
    current_values = (
        dict(current_materialized.get("values") or {})
        if isinstance(current_materialized, Mapping)
        else {}
    )
    snapshot = resume_state.get("behavior_snapshot")
    if not isinstance(snapshot, Mapping) or snapshot.get("status") != "valid":
        raise InputError("Conversation has no valid behavior snapshot.")
    snapshot_participants = _snapshot_participants(resume_state)

    base_effective = current_values.get("effective_completion")
    if not isinstance(base_effective, Mapping):
        base_effective = resume_state.get("effective_completion")
    has_base_effective = isinstance(base_effective, Mapping)
    behavior_changed_keys = RESUMABLE_BEHAVIOR_SETTING_KEYS.intersection(changed_keys)
    if not has_base_effective and not {"provider", "model"}.intersection(changed_keys):
        # An ordinary behavior writer must remain available for explicitly
        # ineligible chats. It preserves the readiness marker and public setting;
        # a later complete provider/model mutation can build the authority.
        if behavior_changed_keys.issubset(_INELIGIBLE_SAFE_BEHAVIOR_KEYS):
            return None
        raise InputError(
            "Conversation requires valid effective completion settings before "
            "changing referenced behavior sources."
        )

    if has_base_effective:
        sampling = dict(base_effective.get("sampling") or {})
    else:
        defaults = (
            snapshot_participants[0].get("generation_defaults")
            if snapshot_participants
            else None
        )
        snapshot_sampling = (
            defaults.get("sampling") if isinstance(defaults, Mapping) else None
        )
        sampling = dict(snapshot_sampling or {})
    overrides = merged_settings.get("chatGenerationOverride")
    if not isinstance(overrides, Mapping):
        overrides = merged_settings.get("generationOverrides")
    if isinstance(overrides, Mapping):
        if overrides.get("enabled") is False and snapshot_participants:
            defaults = snapshot_participants[0].get("generation_defaults")
            snapshot_sampling = (
                defaults.get("sampling") if isinstance(defaults, Mapping) else None
            )
            if isinstance(snapshot_sampling, Mapping):
                sampling = dict(snapshot_sampling)
        else:
            sampling.update(
                {
                    key: overrides[key]
                    for key in _SAMPLING_FIELDS
                    if key in overrides and overrides[key] is not None
                }
            )
    requested_effective = {
        "provider": merged_settings.get("provider")
        or (base_effective.get("provider") if has_base_effective else None),
        "model": merged_settings.get("model")
        or (base_effective.get("model") if has_base_effective else None),
        "sampling": sampling,
    }
    if not has_base_effective and (
        not isinstance(requested_effective["provider"], str)
        or not requested_effective["provider"].strip()
        or not isinstance(requested_effective["model"], str)
        or not requested_effective["model"].strip()
    ):
        raise InputError(
            "Provider and model are both required to repair resumable settings."
        )
    if has_base_effective and requested_effective == dict(base_effective):
        effective = dict(base_effective)
    else:
        effective, _reason = _resolve_effective_completion(
            provider=str(requested_effective["provider"] or ""),
            model=str(requested_effective["model"] or ""),
            sampling=requested_effective["sampling"],
            character={},
            app_config=ensure_app_config(),
        )
        if effective is None:
            raise InputError("Provider, model, or sampling settings are invalid.")

    primary_character_id = conversation.get("character_id")
    desired_participant_ids = _participant_ids_from_settings(
        merged_settings,
        primary_character_id,
    )
    materialization_settings = dict(merged_settings)
    greeting_selection_id = materialization_settings.get("greetingSelectionId")
    if isinstance(greeting_selection_id, str):
        materialization_settings["greetingSelectionId"] = (
            greeting_selection_id.strip()
        )
    participant_keys = {"participantCharacterIds", "participant_character_ids"}
    if participant_keys.intersection(changed_keys):
        participant_ids = {str(item) for item in desired_participant_ids}
        memory_by_id = merged_settings.get("characterMemoryById")
        if isinstance(memory_by_id, Mapping):
            materialization_settings["characterMemoryById"] = {
                str(character_id): entry
                for character_id, entry in memory_by_id.items()
                if str(character_id) in participant_ids
            }
    known_participants = {
        _source_id(item): dict(item)
        for item in [*(current_values.get("participants") or []), *snapshot_participants]
        if _source_id(item) is not None
    }
    known_desired_participants = [
        known_participants[str(character_id)]
        for character_id in desired_participant_ids
        if str(character_id) in known_participants
    ]
    retained_context_books = [
        dict(book)
        for book in current_values.get("world_books") or []
        if isinstance(book, Mapping)
    ]
    if "conversationContext" in changed_keys:
        context = merged_settings.get("conversationContext")
        requested_context_ids = (
            set(normalize_materialized_world_book_ids(context.get("world_book_ids")))
            if isinstance(context, Mapping) and "world_book_ids" in context
            else set()
        )
        retained_context_books = [
            book
            for book in retained_context_books
            if isinstance(book.get("id"), int) and book["id"] in requested_context_ids
        ]
    preexisting_world_books: dict[int, dict[str, Any]] = {}
    for container in [
        *known_desired_participants,
        {"world_books": retained_context_books},
    ]:
        for world_book in container.get("world_books") or []:
            if not isinstance(world_book, Mapping):
                continue
            try:
                preexisting_world_books.setdefault(
                    int(world_book["id"]),
                    dict(world_book),
                )
            except (KeyError, TypeError, ValueError):
                continue
    preexisting_world_book_ids = set(preexisting_world_books)
    if len(preexisting_world_book_ids) > MAX_MATERIALIZED_WORLD_BOOKS:
        raise InputError(
            f"A resumable chat supports at most {MAX_MATERIALIZED_WORLD_BOOKS} world books."
        )
    existing_exemplar_count, existing_exemplar_bytes = _materialized_exemplar_usage(
        known_desired_participants
    )
    existing_entry_count, existing_entry_bytes = _materialized_world_book_entry_usage(
        list(preexisting_world_books.values())
    )

    missing_character_ids = [
        character_id
        for character_id in desired_participant_ids
        if str(character_id) not in known_participants
    ]
    if missing_character_ids:
        new_materialized = _materialize_behavior(
            conn,
            participant_character_ids=missing_character_ids,
            prompt_preset_id=None,
            memory_by_character_id={},
            primary_greeting=None,
            owner_user_id=owner_user_id,
            max_snapshot_bytes=max_bytes,
            preexisting_world_book_ids=tuple(preexisting_world_book_ids),
            preexisting_world_books=preexisting_world_books,
            existing_exemplar_count=existing_exemplar_count,
            existing_exemplar_bytes=existing_exemplar_bytes,
            existing_world_book_entry_count=existing_entry_count,
            existing_world_book_entry_bytes=existing_entry_bytes,
        )
        known_participants.update(
            {
                _source_id(participant): dict(participant)
                for participant in new_materialized.snapshot.payload["participants"]
                if _source_id(participant) is not None
            }
        )
    participants = [
        known_participants[str(character_id)]
        for character_id in desired_participant_ids
    ]
    final_participant_book_ids = {
        int(book["id"])
        for participant in participants
        for book in participant.get("world_books", [])
        if isinstance(book, Mapping) and isinstance(book.get("id"), int)
    }
    current_context_book_ids = {
        int(book["id"])
        for book in current_values.get("world_books", [])
        if isinstance(book, Mapping) and isinstance(book.get("id"), int)
    }
    if (
        "conversationContext" not in changed_keys
        and len(final_participant_book_ids | current_context_book_ids)
        > MAX_MATERIALIZED_WORLD_BOOKS
    ):
        raise InputError(
            f"A resumable chat supports at most {MAX_MATERIALIZED_WORLD_BOOKS} world books."
        )

    values: dict[str, Any] = {
        **current_values,
        "base_snapshot": {
            "schema_version": int(snapshot["schema_version"]),
            "digest": str(snapshot["digest"]),
        },
        "behavior_controls": build_materialized_behavior_controls(
            materialization_settings
        ),
        "effective_completion": effective,
        "participants": participants,
    }
    pending_greeting = _validated_pending_greeting(
        merged_settings.get(PENDING_GREETING_SETTINGS_KEY),
        conversation=conversation,
        snapshot=snapshot,
        merged_settings=merged_settings,
    )
    if pending_greeting is not None:
        values["greeting"] = pending_greeting
    elif (
        not isinstance(current_values.get("greeting"), Mapping)
        and merged_settings.get("greetingSelectionId") is not None
        and not {"greetingSelectionId", "useCharacterDefault"}.intersection(changed_keys)
    ):
        raise InputError("Stored pending greeting authority is missing.")

    preset_keys = {
        "chatPresetOverrideId",
        "promptPreset",
        "prompt_preset",
        "presetScope",
    }
    if preset_keys.intersection(changed_keys):
        preset_scope = (
            str(merged_settings.get("presetScope") or "character").strip().lower()
        )
        if preset_scope == "chat":
            preset_raw = (
                merged_settings.get("chatPresetOverrideId")
                or merged_settings.get("promptPreset")
                or merged_settings.get("prompt_preset")
            )
            preset_id = str(preset_raw).strip() if preset_raw else ""
            preset_id = preset_id or DEFAULT_PROMPT_PRESET
            reusable = _find_materialized_preset(
                preset_id,
                current=current_values.get("prompt_preset"),
                participants=snapshot_participants,
            )
            values["prompt_preset"] = reusable or _load_preset(
                conn,
                preset_id,
                selection_source="settings_mutation",
                owner_user_id=owner_user_id,
            )
        else:
            preset_id, selection_source = _frozen_character_prompt_preset(
                participants
            )
            values["prompt_preset"] = _load_preset(
                conn,
                preset_id,
                selection_source=selection_source,
                owner_user_id=owner_user_id,
            )

    if "conversationContext" in changed_keys:
        context = merged_settings.get("conversationContext")
        if isinstance(context, Mapping) and "world_book_ids" in context:
            book_ids = normalize_materialized_world_book_ids(
                context.get("world_book_ids")
            )
            if len(final_participant_book_ids | set(book_ids)) > MAX_MATERIALIZED_WORLD_BOOKS:
                raise InputError(
                    f"A resumable chat supports at most {MAX_MATERIALIZED_WORLD_BOOKS} world books."
                )
            reusable_books: dict[int, dict[str, Any]] = {}
            for book in [
                *(current_values.get("world_books") or []),
                *(
                    book
                    for participant in participants
                    for book in participant.get("world_books", [])
                ),
            ]:
                if isinstance(book, Mapping) and isinstance(book.get("id"), int):
                    reusable_books[int(book["id"])] = dict(book)
            retained_books: dict[int, dict[str, Any]] = {
                int(book["id"]): dict(book)
                for participant in participants
                for book in participant.get("world_books", [])
                if isinstance(book, Mapping) and isinstance(book.get("id"), int)
            }
            retained_books.update(
                {
                    book_id: reusable_books[book_id]
                    for book_id in book_ids
                    if book_id in reusable_books
                }
            )
            existing_entry_count, existing_entry_bytes = (
                _materialized_world_book_entry_usage(list(retained_books.values()))
            )
            missing_book_ids = [
                book_id for book_id in book_ids if book_id not in reusable_books
            ]
            new_books = _materialize_world_books_by_id(
                conn,
                missing_book_ids,
                owner_user_id=owner_user_id,
                participant_character_ids=desired_participant_ids,
                existing_entry_count=existing_entry_count,
                existing_entry_bytes=existing_entry_bytes,
            )
            values["world_books"] = [
                reusable_books.get(book_id)
                or new_books[book_id]
                for book_id in book_ids
            ]
        else:
            values.pop("world_books", None)

    memory_keys = {
        "authorNote",
        "authorNoteEnabled",
        "authorNoteExcludeFromPrompt",
        "authorNoteGmOnly",
        "authorNoteInjectionPosition",
        "authorNotePlacement",
        "authorNotePosition",
        "characterMemoryById",
        "memoryScope",
    }
    if (memory_keys | participant_keys).intersection(changed_keys):
        participant_ids = {str(item) for item in desired_participant_ids}
        memory_by_id = materialization_settings.get("characterMemoryById")
        materialized_memory: dict[str, str] = {}
        for raw_id, entry in (
            memory_by_id.items() if isinstance(memory_by_id, Mapping) else ()
        ):
            character_id = str(raw_id)
            if character_id not in participant_ids:
                raise InputError(
                    f"Character memory ID {character_id} is not a participant."
                )
            note = entry.get("note") if isinstance(entry, Mapping) else entry
            materialized_memory[character_id] = str(note or "")
        values["memory"] = {
            "character_memory_by_id": materialized_memory,
            "author_note": str(materialization_settings.get("authorNote") or ""),
            "author_note_enabled": _behavior_bool(
                materialization_settings,
                "authorNoteEnabled",
                default=True,
            ),
            "author_note_position": _author_note_position(materialization_settings),
            "scope": build_materialized_behavior_controls(materialization_settings)[
                "memory_scope"
            ],
        }

    if "assistantOverlay" in changed_keys:
        overlay = merged_settings.get("assistantOverlay")
        if isinstance(overlay, Mapping):
            values["assistant_overlay"] = _materialize_overlay(
                conn,
                overlay,
                owner_user_id=owner_user_id,
                participants=participants,
                current_overlay=current_values.get("assistant_overlay"),
            )
        else:
            values["assistant_overlay"] = None

    if {"greetingSelectionId", "useCharacterDefault"}.intersection(changed_keys):
        selection_id = materialization_settings.get("greetingSelectionId")
        current_greeting = current_values.get("greeting")
        if (
            isinstance(current_greeting, Mapping)
            and current_greeting.get("selection_id") == selection_id
        ):
            greeting = dict(current_greeting)
        else:
            greeting = _resolve_selected_greeting(
                conn,
                character_id=int(primary_character_id),
                selection_id=selection_id,
            )
        if greeting is not None:
            values["greeting"] = greeting
        elif materialization_settings.get("useCharacterDefault") is False:
            values["greeting"] = None
        else:
            values.pop("greeting", None)

    return build_materialized_behavior_settings(values, max_bytes=max_bytes)


def materialize_roleplay_behavior_settings(
    conn: Any,
    *,
    conversation: Mapping[str, Any],
    resume_state: Mapping[str, Any],
    merged_settings: Mapping[str, Any],
    owner_user_id: str,
    changed_keys: set[str] | frozenset[str],
    max_bytes: int = DEFAULT_MAX_SNAPSHOT_BYTES,
) -> dict[str, Any] | None:
    """Materialize one stable authority, retrying bounded PostgreSQL source drift."""
    attempts = 2 if _is_postgres_connection(conn) else 1
    for _attempt in range(attempts):
        try:
            first = _materialize_roleplay_behavior_settings_once(
                conn,
                conversation=conversation,
                resume_state=resume_state,
                merged_settings=merged_settings,
                owner_user_id=owner_user_id,
                changed_keys=changed_keys,
                max_bytes=max_bytes,
            )
            if attempts == 1:
                return first
            second = _materialize_roleplay_behavior_settings_once(
                conn,
                conversation=conversation,
                resume_state=resume_state,
                merged_settings=merged_settings,
                owner_user_id=owner_user_id,
                changed_keys=changed_keys,
                max_bytes=max_bytes,
            )
        except _SourceDrift:
            continue
        if first == second:
            return second
    raise InputError("Behavior sources changed during settings materialization.")


def create_character_conversation(
    db: CharactersRAGDB,
    *,
    conversation_data: Mapping[str, Any],
    participant_character_ids: Sequence[int] = (),
    prompt_preset_id: str | None = None,
    memory_by_character_id: Mapping[str, str] | None = None,
    provider: str | None = None,
    model: str | None = None,
    sampling: Mapping[str, Any] | None = None,
    initial_messages: Sequence[Mapping[str, Any]] = (),
    primary_greeting: Mapping[str, Any] | None = None,
    seed_first_message: bool = False,
    greeting_strategy: str = "default",
    greeting_alternate_index: int | None = None,
    fallback_greeting: Mapping[str, Any] | None = None,
    conversation_settings: Mapping[str, Any] | None = None,
    max_snapshot_bytes: int = DEFAULT_MAX_SNAPSHOT_BYTES,
) -> str:
    """Create conversation, settings, messages, and snapshot in one transaction."""
    owner_user_id = str(getattr(db, "client_id", "") or "").strip()
    if not owner_user_id:
        raise InputError("Character conversation requires a scoped database owner.")
    supplied_owner = conversation_data.get("client_id")
    if supplied_owner is not None and str(supplied_owner).strip() != owner_user_id:
        raise InputError(
            "Conversation client_id must match the scoped database owner."
        )
    conversation_payload = dict(conversation_data)
    conversation_payload["client_id"] = owner_user_id

    primary_id = conversation_data.get("character_id")
    if isinstance(primary_id, bool):
        raise InputError("character_id must be a positive integer.")
    try:
        primary_id = int(primary_id)
    except (TypeError, ValueError) as exc:
        raise InputError("character_id must be a positive integer.") from exc
    if primary_id <= 0:
        raise InputError("character_id must be a positive integer.")
    ordered_ids = normalize_materialized_participant_ids(
        primary_id,
        participant_character_ids,
    )

    prompt_preset_id = _normalize_prompt_preset_id(prompt_preset_id)

    creation_settings = dict(conversation_settings or {})
    reject_resumable_behavior_credentials(creation_settings)
    validate_resumable_behavior_boole(creation_settings)

    # Ensure optional world-book tables exist before opening the create transaction.
    WorldBookService(db)
    memory = dict(memory_by_character_id or {})
    app_config = ensure_app_config()
    effective_identity: tuple[str, str] | None = None
    identity_resolution_complete = False
    identity_resolution_reason: str | None = None
    greeting_random_value = random.SystemRandom().random()

    for attempt in range(2):
        try:
            with db.transaction() as conn:
                first = _materialize_behavior(
                    conn,
                    participant_character_ids=ordered_ids,
                    prompt_preset_id=prompt_preset_id,
                    memory_by_character_id=memory,
                    primary_greeting=primary_greeting,
                    owner_user_id=owner_user_id,
                    max_snapshot_bytes=max_snapshot_bytes,
                    greeting_strategy=greeting_strategy,
                    greeting_alternate_index=greeting_alternate_index,
                    greeting_random_value=greeting_random_value,
                    fallback_greeting=fallback_greeting,
                )
                if not identity_resolution_complete:
                    initial_effective, identity_resolution_reason = (
                        _resolve_effective_completion(
                            provider=provider,
                            model=model,
                            sampling=sampling,
                            character=first.primary_character,
                            app_config=app_config,
                        )
                    )
                    if initial_effective is not None:
                        effective_identity = (
                            initial_effective["provider"],
                            initial_effective["model"],
                        )
                    identity_resolution_complete = True
                current = _materialize_behavior(
                    conn,
                    participant_character_ids=ordered_ids,
                    prompt_preset_id=prompt_preset_id,
                    memory_by_character_id=memory,
                    primary_greeting=primary_greeting,
                    owner_user_id=owner_user_id,
                    max_snapshot_bytes=max_snapshot_bytes,
                    greeting_strategy=greeting_strategy,
                    greeting_alternate_index=greeting_alternate_index,
                    greeting_random_value=greeting_random_value,
                    fallback_greeting=fallback_greeting,
                )
                if current.snapshot.canonical_bytes != first.snapshot.canonical_bytes:
                    raise _SourceDrift
                if effective_identity is None:
                    effective = None
                    reason = identity_resolution_reason
                else:
                    effective, reason = _resolve_effective_completion(
                        provider=effective_identity[0],
                        model=effective_identity[1],
                        sampling=sampling,
                        character=current.primary_character,
                        app_config=app_config,
                    )

                settings = dict(creation_settings)
                settings["participantCharacterIds"] = ordered_ids
                settings["greetingsChecksum"] = compute_character_greetings_checksum(
                    current.primary_character
                )
                if prompt_preset_id:
                    normalized_preset = str(prompt_preset_id).strip()
                    settings["presetScope"] = "chat"
                    settings["chatPresetOverrideId"] = normalized_preset
                    settings["promptPreset"] = normalized_preset
                else:
                    settings["presetScope"] = "character"
                settings["roleplayResumeV1"] = {
                    "resumeEligible": effective is not None,
                    "resumeIneligibleReason": reason,
                    "effectiveCompletion": effective,
                }
                if effective is not None:
                    settings["chatGenerationOverride"] = {
                        "enabled": True,
                        **effective["sampling"],
                    }
                    settings["roleplayBehaviorV1"] = (
                        build_creation_materialized_behavior_settings(
                            current.snapshot,
                            effective,
                            settings,
                        )
                    )

                conversation_id = db.add_conversation(conversation_payload, conn=conn)
                if not conversation_id:
                    raise InputError("Failed to create character conversation.")
                db.conversation_resume_store.put_creation_settings(
                    conversation_id,
                    settings,
                    conn=conn,
                )
                messages_to_persist = [dict(item) for item in initial_messages]
                accepted_participants = current.snapshot.payload["participants"]
                accepted_greeting = accepted_participants[0]["greeting"]
                if seed_first_message and str(
                    accepted_greeting.get("content") or ""
                ).strip():
                    messages_to_persist.insert(
                        0,
                        {
                            "sender": str(
                                accepted_participants[0]["identity"].get("name")
                                or "Assistant"
                            ),
                            "content": str(accepted_greeting["content"]),
                        },
                    )
                for initial_message in messages_to_persist:
                    message = dict(initial_message)
                    message["conversation_id"] = conversation_id
                    db.add_message(message, conn=conn)
                db.conversation_resume_store.put_behavior_snapshot(
                    conversation_id,
                    current.snapshot,
                    conn=conn,
                )
                return conversation_id
        except _SourceDrift:
            if attempt == 1:
                raise InputError("Behavior sources changed during conversation creation.") from None
    raise InputError("Failed to create character conversation.")


__all__ = ["create_character_conversation"]
