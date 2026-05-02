"""
Persona memory integration helpers.

This module wires persona session activity to the per-user personalization DB
when personalization is enabled and the user has opted in.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.config import settings
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    ConflictError,
)
from tldw_Server_API.app.core.DB_Management.Personalization_DB import (
    PersonalizationDB,
    SemanticMemory,
    UsageEvent,
)
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.feature_flags import is_personalization_enabled

_DEFAULT_PERSONA_MEMORY_READ_MODE = "legacy_only"
_DEFAULT_PERSONA_MEMORY_WRITE_MODE = "legacy_only"
_ALLOWED_PERSONA_MEMORY_READ_MODES = {
    "legacy_only",
    "chacha_only",
    "chacha_first_fallback_legacy",
}
_ALLOWED_PERSONA_MEMORY_WRITE_MODES = {
    "legacy_only",
    "chacha_only",
    "dual_write",
}
_CHACHA_RETRIEVABLE_MEMORY_TYPES = {
    "summary",
    "semantic",
    "legacy_semantic",
}
_DEFAULT_PERSONA_TOOL_OUTCOME_SUMMARY_MAX_CHARS = 2_048
_TRUNCATION_SUFFIX = "... [truncated]"


@dataclass
class RetrievedMemory:
    content: str
    memory_id: str | None = None


@dataclass
class PersonaMemoryBackfillResult:
    processed_semantic: int
    inserted_semantic: int
    skipped_semantic: int
    processed_usage_events: int
    inserted_usage_events: int
    skipped_usage_events: int
    next_checkpoint: dict[str, int]
    completed: bool


def _normalize_personalization_user_id(user_id: str) -> str:
    raw = str(user_id or "").strip()
    if not raw:
        raise ValueError("user_id is required")
    if raw.isdigit():
        return str(int(raw))
    digest = hashlib.sha256(raw.encode("utf-8")).digest()
    return str(int.from_bytes(digest, byteorder="big", signed=False))


def _get_db_for_user(user_id: str) -> tuple[PersonalizationDB, str]:
    normalized_user_id = _normalize_personalization_user_id(user_id)
    return PersonalizationDB.for_user(normalized_user_id), normalized_user_id


def _open_chacha_db_for_user(user_id: str) -> tuple[CharactersRAGDB, str]:
    normalized_user_id = _normalize_personalization_user_id(user_id)
    db_path = DatabasePaths.get_chacha_db_path(normalized_user_id)
    return CharactersRAGDB(str(db_path), client_id=f"persona_memory_{normalized_user_id}"), normalized_user_id


def _is_profile_opted_in(db: PersonalizationDB, normalized_user_id: str) -> bool:
    profile = db.get_or_create_profile(normalized_user_id)
    return bool(profile.get("enabled", 0))


def _coerce_flag_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    candidate = str(value).strip().lower()
    if candidate in {"1", "true", "yes", "on", "enabled"}:
        return True
    if candidate in {"0", "false", "no", "off", "disabled"}:
        return False
    return default


def _normalize_namespace_value(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _persistent_scope_fallback_namespace_from_session(session_id: str | None) -> str | None:
    normalized_session_id = _normalize_namespace_value(session_id)
    if not normalized_session_id:
        return None
    digest = hashlib.sha256(normalized_session_id.encode("utf-8")).hexdigest()[:24]
    return f"persistent_fallback_sid_{digest}"


def _legacy_persistent_scope_namespace_for_persona(
    *,
    normalized_user_id: str,
    persona_id: str | None,
) -> str | None:
    normalized_persona_id = _normalize_namespace_value(persona_id)
    if not normalized_persona_id:
        return None
    digest = hashlib.sha256(
        f"{normalized_user_id}:{normalized_persona_id}".encode("utf-8")
    ).hexdigest()[:24]
    return f"persistent_legacy_pid_{digest}"


def _resolve_persona_memory_namespace(
    *,
    runtime_mode: str | None,
    scope_snapshot_id: str | None,
    session_id: str | None,
) -> tuple[str | None, str | None]:
    mode = str(runtime_mode or "").strip().lower()
    normalized_scope_snapshot_id = _normalize_namespace_value(scope_snapshot_id)
    normalized_session_id = _normalize_namespace_value(session_id)
    if mode == "persistent_scoped":
        if normalized_scope_snapshot_id:
            return normalized_scope_snapshot_id, None
        return _persistent_scope_fallback_namespace_from_session(normalized_session_id), None
    if mode == "session_scoped":
        return normalized_scope_snapshot_id, normalized_session_id
    return None, None


def _truncate_text(value: Any, *, max_chars: int) -> str:
    text = str(value or "")
    safe_limit = max(1, int(max_chars))
    if len(text) <= safe_limit:
        return text
    if safe_limit <= len(_TRUNCATION_SUFFIX):
        return text[:safe_limit]
    return f"{text[: safe_limit - len(_TRUNCATION_SUFFIX)]}{_TRUNCATION_SUFFIX}"


def _summarize_retention_value(value: Any) -> tuple[str, int | None, int | None, str]:
    value_type = type(value).__name__
    if value is None:
        return value_type, 0, None, "na"
    if isinstance(value, str):
        digest = hashlib.sha1(value.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]
        return value_type, len(value), None, digest
    if isinstance(value, (bytes, bytearray)):
        raw = bytes(value)
        digest = hashlib.sha1(raw, usedforsecurity=False).hexdigest()[:16]
        return value_type, len(raw), None, digest
    if isinstance(value, dict):
        signature = f"dict:{len(value)}"
        digest = hashlib.sha1(signature.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]
        return value_type, None, len(value), digest
    if isinstance(value, (list, tuple, set)):
        signature = f"{value_type}:{len(value)}"
        digest = hashlib.sha1(signature.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]
        return value_type, None, len(value), digest
    text = str(value)
    digest = hashlib.sha1(text.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]
    return value_type, len(text), None, digest


def _get_persona_tool_outcome_summary_max_chars() -> int:
    try:
        candidate = int(
            settings.get(
                "PERSONA_TOOL_OUTCOME_SUMMARY_MAX_CHARS",
                _DEFAULT_PERSONA_TOOL_OUTCOME_SUMMARY_MAX_CHARS,
            )
        )
    except Exception:
        candidate = _DEFAULT_PERSONA_TOOL_OUTCOME_SUMMARY_MAX_CHARS
    return max(256, min(candidate, 16_384))


def _summarize_tool_outcome_payload(outcome: dict[str, Any]) -> str:
    payload = dict(outcome or {})
    output_value = payload.get("output")
    if "output" not in payload:
        output_value = payload.get("result")
    output_type, output_char_count, output_item_count, output_digest = _summarize_retention_value(output_value)
    error_text = str(payload.get("error") or "").strip()
    error_digest = (
        hashlib.sha1(error_text.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]
        if error_text
        else "na"
    )
    summary = {
        "ok": bool(payload.get("ok", False)),
        "reason_code": str(payload.get("reason_code") or ""),
        "output_type": output_type,
        "output_char_count": output_char_count,
        "output_item_count": output_item_count,
        "output_digest": output_digest,
        "error_present": bool(error_text),
        "error_char_count": len(error_text),
        "error_digest": error_digest,
    }
    serialized = json.dumps(summary, ensure_ascii=True, sort_keys=True)
    return _truncate_text(serialized, max_chars=_get_persona_tool_outcome_summary_max_chars())


def _get_persona_memory_read_mode() -> str:
    try:
        candidate = str(settings.get("PERSONA_MEMORY_READ_MODE", _DEFAULT_PERSONA_MEMORY_READ_MODE)).strip().lower()
    except Exception:
        candidate = _DEFAULT_PERSONA_MEMORY_READ_MODE
    if candidate in _ALLOWED_PERSONA_MEMORY_READ_MODES:
        return candidate
    return _DEFAULT_PERSONA_MEMORY_READ_MODE


def _get_persona_memory_write_mode() -> str:
    try:
        candidate = str(settings.get("PERSONA_MEMORY_WRITE_MODE", _DEFAULT_PERSONA_MEMORY_WRITE_MODE)).strip().lower()
    except Exception:
        candidate = _DEFAULT_PERSONA_MEMORY_WRITE_MODE
    if candidate in _ALLOWED_PERSONA_MEMORY_WRITE_MODES:
        return candidate
    return _DEFAULT_PERSONA_MEMORY_WRITE_MODE


def _ensure_persona_profile_for_memory(
    chacha_db: CharactersRAGDB,
    *,
    user_id: str,
    persona_id: str,
) -> bool:
    profile = chacha_db.get_persona_profile(persona_id, user_id=user_id, include_deleted=False)
    if profile is not None:
        return True
    try:
        _ = chacha_db.create_persona_profile(
            {
                "id": str(persona_id),
                "user_id": str(user_id),
                "name": f"Persona {persona_id}",
                "mode": "session_scoped",
                "system_prompt": "",
                "is_active": True,
            }
        )
        return True
    except ConflictError:
        profile = chacha_db.get_persona_profile(persona_id, user_id=user_id, include_deleted=False)
        return profile is not None
    except (CharactersRAGDBError, OSError, RuntimeError, ValueError) as exc:
        logger.debug("persona memory profile ensure skipped: {}", exc)
        return False


def _write_legacy_persona_turn(
    *,
    user_id: str,
    normalized_user_id: str,
    session_id: str,
    persona_id: str,
    role: str,
    content: str,
    turn_type: str,
    metadata: dict[str, Any] | None,
    store_as_memory: bool,
) -> bool:
    db = PersonalizationDB.for_user(normalized_user_id)
    clean_metadata = dict(metadata or {})
    clean_metadata["persona_id"] = str(persona_id or "")
    clean_metadata["turn_type"] = str(turn_type or "text")
    clean_metadata["content_length"] = len(str(content or ""))
    _ = db.insert_usage_event(
        UsageEvent(
            user_id=normalized_user_id,
            type="persona.turn",
            resource_id=str(session_id or ""),
            tags=["persona", str(role or "unknown")],
            metadata=clean_metadata,
        )
    )
    if store_as_memory:
        memory_text = str(content or "").strip()
        if memory_text:
            if len(memory_text) > 1024:
                memory_text = memory_text[:1024]
            _ = db.add_semantic_memory(
                SemanticMemory(
                    user_id=normalized_user_id,
                    content=memory_text,
                    tags=["persona", str(role or "unknown"), str(persona_id or "")],
                    pinned=False,
                )
            )
    return True


def _write_chacha_persona_turn(
    *,
    user_id: str,
    normalized_user_id: str,
    session_id: str,
    persona_id: str,
    role: str,
    content: str,
    turn_type: str,
    metadata: dict[str, Any] | None,
    store_as_memory: bool,
    scope_snapshot_id: str | None,
    namespace_session_id: str | None,
) -> bool:
    chacha_db = None
    try:
        chacha_db, _ = _open_chacha_db_for_user(user_id)
        if not _ensure_persona_profile_for_memory(chacha_db, user_id=normalized_user_id, persona_id=persona_id):
            return False

        telemetry_payload = {
            "role": str(role or "unknown"),
            "turn_type": str(turn_type or "text"),
            "session_id": str(session_id or ""),
            "namespace_session_id": str(namespace_session_id or ""),
            "scope_snapshot_id": str(scope_snapshot_id or ""),
            "persona_id": str(persona_id or ""),
            "content_length": len(str(content or "")),
            "metadata": dict(metadata or {}),
        }
        _ = chacha_db.add_persona_memory_entry(
            {
                "persona_id": str(persona_id),
                "user_id": str(normalized_user_id),
                "memory_type": "usage_event",
                "content": json.dumps(telemetry_payload, ensure_ascii=True, sort_keys=True),
                "scope_snapshot_id": scope_snapshot_id,
                "session_id": namespace_session_id,
                "salience": 0.0,
            }
        )
        if store_as_memory:
            memory_text = str(content or "").strip()
            if memory_text:
                if len(memory_text) > 1024:
                    memory_text = memory_text[:1024]
                _ = chacha_db.add_persona_memory_entry(
                    {
                        "persona_id": str(persona_id),
                        "user_id": str(normalized_user_id),
                        "memory_type": "summary",
                        "content": memory_text,
                        "scope_snapshot_id": scope_snapshot_id,
                        "session_id": namespace_session_id,
                        "salience": 0.5,
                    }
                )
        return True
    except (CharactersRAGDBError, OSError, RuntimeError, ValueError, ConflictError) as exc:
        logger.debug("persona chacha memory write skipped: {}", exc)
        return False
    finally:
        if chacha_db is not None:
            try:
                chacha_db.close_all_connections()
                chacha_db.close_connection()
            except Exception as close_exc:
                logger.debug("persona chacha memory close skipped: {}", close_exc)


def _read_legacy_memories(
    *,
    user_id: str,
    normalized_user_id: str,
    query_text: str,
    top_k: int,
) -> list[RetrievedMemory]:
    db = PersonalizationDB.for_user(normalized_user_id)
    safe_top_k = max(1, min(int(top_k), 10))
    query = str(query_text or "").strip() or None
    items, _ = db.list_semantic_memories(
        user_id=normalized_user_id,
        q=query,
        limit=safe_top_k,
        offset=0,
    )
    if not items and query:
        items, _ = db.list_semantic_memories(
            user_id=normalized_user_id,
            q=None,
            limit=safe_top_k,
            offset=0,
        )
    out: list[RetrievedMemory] = []
    for item in items:
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        memory_id = item.get("id")
        out.append(
            RetrievedMemory(
                content=content,
                memory_id=(str(memory_id) if memory_id is not None else None),
            )
        )
    return out


def _read_chacha_memories(
    *,
    user_id: str,
    normalized_user_id: str,
    persona_id: str | None,
    query_text: str,
    top_k: int,
    runtime_mode: str | None,
    scope_snapshot_id: str | None,
    session_id: str | None,
) -> list[RetrievedMemory]:
    chacha_db = None
    try:
        chacha_db, _ = _open_chacha_db_for_user(user_id)
        normalized_runtime_mode = str(runtime_mode or "").strip().lower()
        normalized_persona_id = str(persona_id).strip() if persona_id else None
        missing_scope_requested = (
            normalized_runtime_mode == "persistent_scoped"
            and _normalize_namespace_value(scope_snapshot_id) is None
        )
        namespace_scope_snapshot_id, namespace_session_id = _resolve_persona_memory_namespace(
            runtime_mode=normalized_runtime_mode,
            scope_snapshot_id=scope_snapshot_id,
            session_id=session_id,
        )
        if normalized_runtime_mode == "persistent_scoped" and not namespace_scope_snapshot_id:
            return []
        if normalized_runtime_mode == "session_scoped" and not namespace_session_id:
            return []

        def _query_rows(
            *,
            scope_id: str | None,
            sid: str | None,
        ) -> list[dict[str, Any]]:
            return chacha_db.list_persona_memory_entries(
                user_id=str(normalized_user_id),
                persona_id=normalized_persona_id,
                scope_snapshot_id=scope_id,
                session_id=sid,
                include_archived=False,
                include_deleted=False,
                limit=candidate_limit,
                offset=0,
            )

        candidate_limit = max(25, min(500, int(top_k) * 25))
        rows = _query_rows(scope_id=namespace_scope_snapshot_id, sid=namespace_session_id)

        if missing_scope_requested and normalized_persona_id:
            legacy_scope_namespace = _legacy_persistent_scope_namespace_for_persona(
                normalized_user_id=str(normalized_user_id),
                persona_id=normalized_persona_id,
            )
            if legacy_scope_namespace:
                legacy_rows = _query_rows(scope_id=legacy_scope_namespace, sid=None)
                if not legacy_rows:
                    migrated_count = chacha_db.backfill_persona_memory_scope_namespace(
                        user_id=str(normalized_user_id),
                        persona_id=normalized_persona_id,
                        scope_snapshot_id=legacy_scope_namespace,
                        require_missing_session_id=True,
                        include_archived=False,
                        include_deleted=False,
                    )
                    if migrated_count > 0:
                        logger.debug(
                            "persona persistent scope namespace backfilled rows={} user={} persona={}",
                            migrated_count,
                            normalized_user_id,
                            normalized_persona_id,
                        )
                        legacy_rows = _query_rows(scope_id=legacy_scope_namespace, sid=None)
                if legacy_rows:
                    seen_ids: set[str] = set()
                    merged_rows: list[dict[str, Any]] = []
                    for row in [*rows, *legacy_rows]:
                        row_id = str(row.get("id") or "").strip()
                        if row_id and row_id in seen_ids:
                            continue
                        if row_id:
                            seen_ids.add(row_id)
                        merged_rows.append(row)
                    rows = merged_rows

        query = str(query_text or "").strip().lower()
        retrievable_rows = [
            row for row in rows
            if str(row.get("memory_type") or "").strip().lower() in _CHACHA_RETRIEVABLE_MEMORY_TYPES
        ]
        if query:
            filtered = [
                row for row in retrievable_rows
                if query in str(row.get("content") or "").strip().lower()
            ]
            if filtered:
                retrievable_rows = filtered

        out: list[RetrievedMemory] = []
        for row in retrievable_rows[: max(1, min(int(top_k), 10))]:
            content = str(row.get("content") or "").strip()
            if not content:
                continue
            memory_id = row.get("id")
            out.append(
                RetrievedMemory(
                    content=content,
                    memory_id=(str(memory_id) if memory_id is not None else None),
                )
            )
        return out
    except (CharactersRAGDBError, OSError, RuntimeError, ValueError) as exc:
        logger.debug("persona chacha memory retrieval skipped: {}", exc)
        return []
    finally:
        if chacha_db is not None:
            try:
                chacha_db.close_all_connections()
                chacha_db.close_connection()
            except Exception as close_exc:
                logger.debug("persona chacha memory close skipped: {}", close_exc)


def _deterministic_backfill_entry_id(*parts: str) -> str:
    joined = "|".join(str(part or "") for part in parts)
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()
    return f"persona_mem_{digest[:24]}"


def _is_unique_entry_conflict(exc: Exception) -> bool:
    message = str(exc).strip().lower()
    if "unique constraint failed" in message:
        return True
    if "duplicate key value violates unique constraint" in message:
        return True
    return False


def _coerce_backfill_checkpoint(checkpoint: dict[str, Any] | None) -> tuple[int, int]:
    if not isinstance(checkpoint, dict):
        return 0, 0
    try:
        semantic_offset = max(0, int(checkpoint.get("semantic_offset", 0)))
    except (TypeError, ValueError):
        semantic_offset = 0
    try:
        events_offset = max(0, int(checkpoint.get("events_offset", 0)))
    except (TypeError, ValueError):
        events_offset = 0
    return semantic_offset, events_offset


def retrieve_top_memories(
    *,
    user_id: str | None,
    query_text: str,
    top_k: int = 3,
    persona_id: str | None = None,
    runtime_mode: str | None = None,
    scope_snapshot_id: str | None = None,
    session_id: str | None = None,
) -> list[RetrievedMemory]:
    if not user_id or not is_personalization_enabled():
        return []
    try:
        legacy_db, normalized_user_id = _get_db_for_user(user_id)
        if not _is_profile_opted_in(legacy_db, normalized_user_id):
            return []

        read_mode = _get_persona_memory_read_mode()
        if read_mode == "chacha_only":
            return _read_chacha_memories(
                user_id=user_id,
                normalized_user_id=normalized_user_id,
                persona_id=persona_id,
                query_text=query_text,
                top_k=top_k,
                runtime_mode=runtime_mode,
                scope_snapshot_id=scope_snapshot_id,
                session_id=session_id,
            )
        if read_mode == "chacha_first_fallback_legacy":
            chacha_result = _read_chacha_memories(
                user_id=user_id,
                normalized_user_id=normalized_user_id,
                persona_id=persona_id,
                query_text=query_text,
                top_k=top_k,
                runtime_mode=runtime_mode,
                scope_snapshot_id=scope_snapshot_id,
                session_id=session_id,
            )
            if chacha_result:
                return chacha_result
            return _read_legacy_memories(
                user_id=user_id,
                normalized_user_id=normalized_user_id,
                query_text=query_text,
                top_k=top_k,
            )
        return _read_legacy_memories(
            user_id=user_id,
            normalized_user_id=normalized_user_id,
            query_text=query_text,
            top_k=top_k,
        )
    except Exception:
        logger.debug("persona memory retrieval skipped")
        return []


def persist_persona_turn(
    *,
    user_id: str | None,
    session_id: str,
    persona_id: str,
    role: str,
    content: str,
    turn_type: str,
    metadata: dict[str, Any] | None = None,
    store_as_memory: bool = False,
    runtime_mode: str | None = None,
    scope_snapshot_id: str | None = None,
) -> bool:
    if not user_id or not is_personalization_enabled():
        return False
    try:
        legacy_db, normalized_user_id = _get_db_for_user(user_id)
        if not _is_profile_opted_in(legacy_db, normalized_user_id):
            return False

        write_mode = _get_persona_memory_write_mode()
        legacy_ok = False
        chacha_ok = False
        namespace_scope_snapshot_id, namespace_session_id = _resolve_persona_memory_namespace(
            runtime_mode=runtime_mode,
            scope_snapshot_id=scope_snapshot_id,
            session_id=session_id,
        )

        if write_mode in {"legacy_only", "dual_write"}:
            legacy_ok = _write_legacy_persona_turn(
                user_id=user_id,
                normalized_user_id=normalized_user_id,
                session_id=session_id,
                persona_id=persona_id,
                role=role,
                content=content,
                turn_type=turn_type,
                metadata=metadata,
                store_as_memory=bool(store_as_memory),
            )
        if write_mode in {"chacha_only", "dual_write"}:
            chacha_ok = _write_chacha_persona_turn(
                user_id=user_id,
                normalized_user_id=normalized_user_id,
                session_id=session_id,
                persona_id=persona_id,
                role=role,
                content=content,
                turn_type=turn_type,
                metadata=metadata,
                store_as_memory=bool(store_as_memory),
                scope_snapshot_id=namespace_scope_snapshot_id,
                namespace_session_id=namespace_session_id,
            )

        if write_mode == "dual_write":
            return legacy_ok or chacha_ok
        if write_mode == "chacha_only":
            return chacha_ok
        return legacy_ok
    except Exception:
        logger.debug("persona turn persistence skipped")
        return False


def persist_tool_outcome(
    *,
    user_id: str | None,
    session_id: str,
    persona_id: str,
    tool_name: str,
    step_idx: int,
    outcome: dict[str, Any],
    store_as_memory: bool = True,
    runtime_mode: str | None = None,
    scope_snapshot_id: str | None = None,
) -> bool:
    safe_outcome = dict(outcome or {}) if isinstance(outcome, dict) else {"value": str(outcome)}
    serialized_summary = _summarize_tool_outcome_payload(safe_outcome)
    tool_summary = f"Tool={tool_name} step={step_idx} outcome={serialized_summary}"
    tool_summary = _truncate_text(tool_summary, max_chars=_get_persona_tool_outcome_summary_max_chars())
    return persist_persona_turn(
        user_id=user_id,
        session_id=session_id,
        persona_id=persona_id,
        role="tool",
        content=tool_summary,
        turn_type="tool_result",
        metadata={"tool_name": str(tool_name or ""), "step_idx": int(step_idx)},
        store_as_memory=bool(store_as_memory),
        runtime_mode=runtime_mode,
        scope_snapshot_id=scope_snapshot_id,
    )


def backfill_persona_memory_from_legacy(
    *,
    user_id: str | None,
    persona_id: str,
    batch_size: int = 100,
    checkpoint: dict[str, Any] | None = None,
    include_usage_events: bool = True,
) -> PersonaMemoryBackfillResult:
    """
    Backfill legacy personalization data to ChaCha persona memory entries.

    The operation is idempotent (deterministic IDs), resumable (checkpoint offsets),
    and opt-in gated (uses legacy personalization profile enabled state).
    """
    empty_result = PersonaMemoryBackfillResult(
        processed_semantic=0,
        inserted_semantic=0,
        skipped_semantic=0,
        processed_usage_events=0,
        inserted_usage_events=0,
        skipped_usage_events=0,
        next_checkpoint={"semantic_offset": 0, "events_offset": 0},
        completed=True,
    )
    if not user_id or not is_personalization_enabled():
        return empty_result

    safe_batch = max(1, min(int(batch_size), 1000))
    semantic_offset, events_offset = _coerce_backfill_checkpoint(checkpoint)

    chacha_db = None
    try:
        legacy_db, normalized_user_id = _get_db_for_user(user_id)
        if not _is_profile_opted_in(legacy_db, normalized_user_id):
            return empty_result
        chacha_db, _ = _open_chacha_db_for_user(user_id)
        if not _ensure_persona_profile_for_memory(chacha_db, user_id=normalized_user_id, persona_id=persona_id):
            return empty_result

        processed_semantic = 0
        inserted_semantic = 0
        skipped_semantic = 0

        semantic_items, _ = legacy_db.list_semantic_memories(
            user_id=normalized_user_id,
            q=None,
            limit=safe_batch,
            offset=semantic_offset,
            include_hidden=True,
        )
        for item in semantic_items:
            processed_semantic += 1
            legacy_memory_id = str(item.get("id") or "")
            entry_id = _deterministic_backfill_entry_id(
                "legacy_semantic",
                normalized_user_id,
                persona_id,
                legacy_memory_id,
            )
            content = str(item.get("content") or "").strip()
            if not content:
                skipped_semantic += 1
                continue
            try:
                _ = chacha_db.add_persona_memory_entry(
                    {
                        "id": entry_id,
                        "persona_id": str(persona_id),
                        "user_id": str(normalized_user_id),
                        "memory_type": "legacy_semantic",
                        "content": content[:1024],
                        "salience": 0.8 if bool(item.get("pinned")) else 0.4,
                    }
                )
                inserted_semantic += 1
            except Exception as exc:
                if _is_unique_entry_conflict(exc):
                    skipped_semantic += 1
                    continue
                raise

        processed_usage_events = 0
        inserted_usage_events = 0
        skipped_usage_events = 0
        usage_batch: list[dict[str, Any]] = []
        if include_usage_events and len(semantic_items) < safe_batch:
            usage_batch = legacy_db.list_recent_events(
                user_id=normalized_user_id,
                limit=safe_batch,
                offset=events_offset,
            )
            for event in usage_batch:
                processed_usage_events += 1
                legacy_event_id = str(event.get("id") or "")
                entry_id = _deterministic_backfill_entry_id(
                    "legacy_usage_event",
                    normalized_user_id,
                    persona_id,
                    legacy_event_id,
                )
                payload = {
                    "id": legacy_event_id,
                    "timestamp": event.get("timestamp"),
                    "type": event.get("type"),
                    "resource_id": event.get("resource_id"),
                    "tags": event.get("tags") or [],
                }
                try:
                    _ = chacha_db.add_persona_memory_entry(
                        {
                            "id": entry_id,
                            "persona_id": str(persona_id),
                            "user_id": str(normalized_user_id),
                            "memory_type": "legacy_usage_event",
                            "content": json.dumps(payload, ensure_ascii=True, sort_keys=True),
                            "salience": 0.0,
                            "archived": True,
                        }
                    )
                    inserted_usage_events += 1
                except Exception as exc:
                    if _is_unique_entry_conflict(exc):
                        skipped_usage_events += 1
                        continue
                    raise

        next_semantic_offset = semantic_offset + len(semantic_items)
        next_events_offset = events_offset
        if include_usage_events and len(semantic_items) < safe_batch:
            next_events_offset = events_offset + len(usage_batch)
        completed = (
            len(semantic_items) < safe_batch
            and (
                not include_usage_events
                or len(usage_batch) < safe_batch
            )
        )
        return PersonaMemoryBackfillResult(
            processed_semantic=processed_semantic,
            inserted_semantic=inserted_semantic,
            skipped_semantic=skipped_semantic,
            processed_usage_events=processed_usage_events,
            inserted_usage_events=inserted_usage_events,
            skipped_usage_events=skipped_usage_events,
            next_checkpoint={
                "semantic_offset": next_semantic_offset,
                "events_offset": next_events_offset,
            },
            completed=completed,
        )
    except Exception as exc:
        logger.debug("persona memory backfill skipped: {}", exc)
        return empty_result
    finally:
        if chacha_db is not None:
            try:
                chacha_db.close_all_connections()
                chacha_db.close_connection()
            except Exception as close_exc:
                logger.debug("persona memory backfill close skipped: {}", close_exc)
