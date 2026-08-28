"""Atomic creation of resumable character conversations."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import TYPE_CHECKING, Any

from tldw_Server_API.app.core.Character_Chat.character_behavior_snapshot import (
    DEFAULT_MAX_SNAPSHOT_BYTES,
    BehaviorSnapshotV1,
    build_behavior_snapshot,
)
from tldw_Server_API.app.core.Character_Chat.world_book_manager import WorldBookService
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import InputError

if TYPE_CHECKING:
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


_SAMPLING_FIELDS = frozenset({"temperature", "top_p", "repetition_penalty", "stop"})
_CREDENTIAL_FIELDS = frozenset(
    {
        "api_key",
        "apikey",
        "access_token",
        "auth_token",
        "authorization",
        "bearer_token",
        "client_secret",
        "credential",
        "credentials",
        "password",
        "private_key",
        "refresh_token",
        "secret",
        "token",
        "x_api_key",
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
    """Copy JSON behavior data while dropping credentials and binary payloads."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if key_text.lower() in _CREDENTIAL_FIELDS or isinstance(
                item, (bytes, bytearray, memoryview)
            ):
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


def _decode_json(value: Any, default: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    return value if value is not None else default


def _character_generation_defaults(character: Mapping[str, Any]) -> dict[str, Any]:
    """Read only the four completion sampling fields from character extensions."""
    extensions = _decode_json(character.get("extensions"), {})
    tldw = extensions.get("tldw") if isinstance(extensions, dict) else None
    generation = tldw.get("generation") if isinstance(tldw, dict) else None
    containers = [
        item
        for item in (generation, extensions, character)
        if isinstance(item, Mapping)
    ]
    result: dict[str, Any] = {}
    for key in ("temperature", "top_p", "repetition_penalty", "stop"):
        for container in containers:
            if key in container:
                result[key] = _safe_json(container[key])
                break
    return result


def _select_rows(conn: Any, query: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
    result = conn.execute(query, params)
    return [_row_dict(row, result) for row in result.fetchall()]


def _load_preset(conn: Any, preset_id: str | None) -> dict[str, Any]:
    normalized = str(preset_id or "default").strip() or "default"
    if normalized in {"default", "st_default"}:
        return {
            "preset_id": normalized,
            "name": normalized.replace("_", " ").title(),
            "builtin": True,
            "version": 1,
            "section_order": [],
            "section_templates": {},
        }
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
    return {
        "preset_id": record["preset_id"],
        "name": record["name"],
        "builtin": False,
        "version": int(record.get("version") or 1),
        "updated_at": str(record.get("last_modified") or ""),
        "section_order": _decode_json(record.get("section_order_json"), []),
        "section_templates": _decode_json(record.get("section_templates_json"), {}),
    }


def _load_world_books(conn: Any, character_id: int) -> list[dict[str, Any]]:
    books = _select_rows(
        conn,
        """
        SELECT wb.*, cwb.enabled AS attachment_enabled,
               cwb.priority AS attachment_priority
          FROM world_books wb
          JOIN character_world_books cwb ON cwb.world_book_id = wb.id
         WHERE cwb.character_id = ? AND wb.deleted = FALSE
         ORDER BY cwb.priority DESC, wb.name, wb.id
        """,
        (character_id,),
    )
    materialized: list[dict[str, Any]] = []
    for book in books:
        entries = _select_rows(
            conn,
            """
            SELECT * FROM world_book_entries
             WHERE world_book_id = ?
             ORDER BY priority DESC, id
            """,
            (book["id"],),
        )
        clean_book = {
            key: _safe_json(value)
            for key, value in book.items()
            if key not in {"client_id", "deleted"}
        }
        clean_book["entries"] = [
            {
                key: _safe_json(
                    _decode_json(value, {} if key == "metadata" else [])
                    if key in {"keywords", "metadata"}
                    else value
                )
                for key, value in entry.items()
                if key not in {"client_id", "deleted"}
            }
            for entry in entries
        ]
        materialized.append(clean_book)
    return materialized


def _load_exemplars(conn: Any, character_id: int) -> list[dict[str, Any]]:
    rows = _select_rows(
        conn,
        """
        SELECT * FROM character_exemplars
         WHERE character_id = ? AND is_deleted = FALSE
         ORDER BY updated_at DESC, created_at DESC, id
        """,
        (character_id,),
    )
    return [
        {
            key: _safe_json(_decode_json(value, []) if key in {
                "rhetorical",
                "safety_allowed",
                "safety_blocked",
            } else value)
            for key, value in row.items()
            if key not in {"character_id", "is_deleted"}
        }
        for row in rows
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
    max_snapshot_bytes: int,
) -> _MaterializedBehavior:
    preset = _load_preset(conn, prompt_preset_id)
    participants: list[dict[str, Any]] = []
    primary_character: dict[str, Any] | None = None
    for index, character_id in enumerate(participant_character_ids):
        result = conn.execute(
            "SELECT * FROM character_cards WHERE id = ? AND deleted = FALSE LIMIT 1",
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
        if primary_character is None:
            primary_character = character
        extensions = _safe_json(character.get("extensions") or {})
        prompt_extensions: dict[str, Any] = {"prompt_preset": _safe_json(preset)}
        if isinstance(extensions, dict):
            prompt_extensions["character_extensions"] = extensions
        memory = memory_by_character_id.get(str(character_id))
        participants.append(
            {
                "source": {
                    "kind": "character",
                    "id": str(character_id),
                    "version": int(character.get("version") or 1),
                },
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
                    primary_greeting if index == 0 else None,
                ),
                "generation_defaults": _safe_json(
                    _character_generation_defaults(character)
                ),
                "exemplars": _load_exemplars(conn, character_id),
                "world_books": _load_world_books(conn, character_id),
                "default_memory": (
                    {"content": memory, "source": "creation_request", "version": 1}
                    if isinstance(memory, str)
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
        resolved = _character_generation_defaults(character)
        candidate: dict[str, Any] = {
            "temperature": resolved.get("temperature", 0.7),
            "top_p": resolved.get("top_p", 1.0),
            "repetition_penalty": resolved.get("repetition_penalty", 1.0),
            "stop": resolved.get("stop", []),
        }
    else:
        if set(sampling) != _SAMPLING_FIELDS:
            return None
        candidate = dict(sampling)
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
) -> tuple[dict[str, Any] | None, str | None]:
    resolved_provider = str(provider or os.getenv("DEFAULT_LLM_PROVIDER") or "").strip()
    resolved_model = str(model or os.getenv("CHAR_CHAT_MODEL") or "").strip()
    if os.getenv("PYTEST_CURRENT_TEST"):
        resolved_provider = resolved_provider or "local-llm"
        resolved_model = resolved_model or "local-test"
    normalized_sampling = _normalize_sampling(sampling, character)
    if not resolved_provider or not resolved_model or normalized_sampling is None:
        return None, "incomplete_effective_settings"
    return {
        "provider": resolved_provider,
        "model": resolved_model,
        "sampling": normalized_sampling,
    }, None


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
    conversation_settings: Mapping[str, Any] | None = None,
    max_snapshot_bytes: int = DEFAULT_MAX_SNAPSHOT_BYTES,
) -> str:
    """Create conversation, settings, messages, and snapshot in one transaction."""
    primary_id = conversation_data.get("character_id")
    if isinstance(primary_id, bool):
        raise InputError("character_id must be a positive integer.")
    try:
        primary_id = int(primary_id)
    except (TypeError, ValueError) as exc:
        raise InputError("character_id must be a positive integer.") from exc
    ordered_ids = [primary_id]
    for value in participant_character_ids:
        try:
            normalized = int(value)
        except (TypeError, ValueError) as exc:
            raise InputError("participant character IDs must be integers.") from exc
        if normalized > 0 and normalized not in ordered_ids:
            ordered_ids.append(normalized)
    if primary_id <= 0:
        raise InputError("character_id must be a positive integer.")

    # Ensure optional world-book tables exist before opening the create transaction.
    WorldBookService(db)
    memory = dict(memory_by_character_id or {})
    with db.transaction() as conn:
        materialized = _materialize_behavior(
            conn,
            participant_character_ids=ordered_ids,
            prompt_preset_id=prompt_preset_id,
            memory_by_character_id=memory,
            primary_greeting=primary_greeting,
            max_snapshot_bytes=max_snapshot_bytes,
        )

    for attempt in range(2):
        try:
            with db.transaction() as conn:
                current = _materialize_behavior(
                    conn,
                    participant_character_ids=ordered_ids,
                    prompt_preset_id=prompt_preset_id,
                    memory_by_character_id=memory,
                    primary_greeting=primary_greeting,
                    max_snapshot_bytes=max_snapshot_bytes,
                )
                if current.snapshot.canonical_bytes != materialized.snapshot.canonical_bytes:
                    raise _SourceDrift

                effective, reason = _resolve_effective_completion(
                    provider=provider,
                    model=model,
                    sampling=sampling,
                    character=current.primary_character,
                )
                settings = dict(conversation_settings or {})
                settings["participantCharacterIds"] = ordered_ids
                if prompt_preset_id:
                    settings["promptPreset"] = prompt_preset_id
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

                conversation_id = db.add_conversation(dict(conversation_data), conn=conn)
                if not conversation_id:
                    raise InputError("Failed to create character conversation.")
                db.conversation_resume_store.put_creation_settings(
                    conversation_id,
                    settings,
                    conn=conn,
                )
                for initial_message in initial_messages:
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
            with db.transaction() as conn:
                materialized = _materialize_behavior(
                    conn,
                    participant_character_ids=ordered_ids,
                    prompt_preset_id=prompt_preset_id,
                    memory_by_character_id=memory,
                    primary_greeting=primary_greeting,
                    max_snapshot_bytes=max_snapshot_bytes,
                )
    raise InputError("Failed to create character conversation.")


__all__ = ["create_character_conversation"]
