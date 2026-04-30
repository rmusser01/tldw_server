from __future__ import annotations

"""Helpers for recording explicit companion activity from adjacent systems."""

import hashlib
import json
import sqlite3
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.DB_Management.Personalization_DB import PersonalizationDB
from tldw_Server_API.app.core.feature_flags import is_personalization_enabled


def _open_db_for_user(user_id: str | int) -> tuple[PersonalizationDB, str]:
    """Open the personalization DB for a user and return the normalized user id."""
    normalized_user_id = str(user_id or "").strip()
    if not normalized_user_id:
        raise ValueError("user_id is required")
    return PersonalizationDB.for_user(user_id), normalized_user_id


def _profile_opted_in(db: PersonalizationDB, user_id: str) -> bool:
    profile = db.get_or_create_profile(user_id)
    return bool(profile.get("enabled", 0))


def _unique_conflict(exc: sqlite3.IntegrityError) -> bool:
    return "unique constraint failed" in str(exc).strip().lower()


def _explicit_provenance(
    *,
    route: str,
    action: str,
    source_timestamp: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "capture_mode": "explicit",
        "route": route,
        "action": action,
    }
    if source_timestamp:
        payload["source_timestamp"] = source_timestamp
    return payload


def _payload_fingerprint(payload: dict[str, Any] | None) -> str:
    try:
        serialized = json.dumps(payload or {}, sort_keys=True, ensure_ascii=True, default=str)
    except Exception:
        serialized = json.dumps({"unserializable": True}, sort_keys=True, ensure_ascii=True)
    return hashlib.sha1(serialized.encode("utf-8"), usedforsecurity=False).hexdigest()[:12]


def record_companion_activity(
    *,
    user_id: str | int | None,
    event_type: str,
    source_type: str,
    source_id: str | int,
    surface: str,
    dedupe_key: str,
    tags: list[str] | None = None,
    provenance: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> str | None:
    """Persist one explicit companion activity event when the user has opted in."""
    if user_id is None or not is_personalization_enabled():
        return None

    try:
        db, normalized_user_id = _open_db_for_user(user_id)
        if not _profile_opted_in(db, normalized_user_id):
            return None
        safe_provenance = dict(provenance or {})
        safe_provenance.setdefault("capture_mode", "explicit")
        return db.insert_companion_activity_event(
            user_id=normalized_user_id,
            event_type=event_type,
            source_type=source_type,
            source_id=str(source_id),
            surface=surface,
            dedupe_key=dedupe_key,
            tags=tags,
            provenance=safe_provenance,
            metadata=metadata,
        )
    except sqlite3.IntegrityError as exc:
        if _unique_conflict(exc):
            logger.debug("companion activity duplicate skipped: {}", dedupe_key)
            return None
        logger.debug("companion activity insert skipped")
        return None
    except Exception:
        logger.debug("companion activity capture skipped")
        return None


def record_companion_activity_events_bulk(
    *,
    user_id: str | int | None,
    events: list[dict[str, Any]],
) -> list[str]:
    """Persist explicit companion activity events in bulk when the user opted in.

    This helper performs synchronous DB I/O and should be wrapped in
    ``asyncio.to_thread`` from async request handlers.
    """
    if user_id is None or not is_personalization_enabled() or not events:
        return []

    normalized_user_id = str(user_id or "").strip() or None
    try:
        db, normalized_user_id = _open_db_for_user(user_id)
        if not _profile_opted_in(db, normalized_user_id):
            return []
        safe_events: list[dict[str, Any]] = []
        for event in events:
            safe_event = dict(event)
            safe_provenance = dict(safe_event.get("provenance") or {})
            safe_provenance.setdefault("capture_mode", "explicit")
            safe_event["provenance"] = safe_provenance
            safe_events.append(safe_event)
        return db.insert_companion_activity_events_bulk(
            user_id=normalized_user_id,
            events=safe_events,
        )
    except Exception:
        logger.warning("companion activity bulk capture skipped")
        return []


def _truncate_text(value: str | None, *, max_length: int) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized:
        return None
    if len(normalized) <= max_length:
        return normalized
    return normalized[: max_length - 3].rstrip() + "..."


def _normalize_companion_tags(tags: list[str] | None) -> list[str]:
    seen: set[str] = set()
    normalized: list[str] = []
    for raw_tag in tags or []:
        tag = str(raw_tag).strip()
        if not tag:
            continue
        if tag in seen:
            continue
        seen.add(tag)
        normalized.append(tag)
    return normalized


_PERSONA_ACTIVITY_SURFACES = {
    "api.persona",
    "persona.sidepanel",
    "companion.conversation",
}
_DEFAULT_PERSONA_ACTIVITY_SURFACE = "api.persona"


def normalize_persona_activity_surface(surface: Any) -> str:
    """Normalize persona-related activity surfaces to the supported allowlist."""
    candidate = str(surface or "").strip().lower()
    if candidate in _PERSONA_ACTIVITY_SURFACES:
        return candidate
    return _DEFAULT_PERSONA_ACTIVITY_SURFACE


def build_manual_check_in_activity(
    *,
    source_id: str,
    title: str | None,
    summary: str,
    tags: list[str] | None = None,
    event_timestamp: str | None = None,
    route: str = "/api/v1/companion/check-ins",
    surface: str = "companion.workspace",
) -> dict[str, Any]:
    """Build a normalized companion activity payload for a manual check-in."""
    normalized_summary = str(summary or "").strip()
    if not normalized_summary:
        raise ValueError("summary is required")
    normalized_title = _truncate_text(title, max_length=120)
    display_title = normalized_title or _truncate_text(normalized_summary, max_length=120)
    normalized_tags = _normalize_companion_tags(tags)
    return {
        "event_type": "companion_check_in_recorded",
        "source_type": "companion_check_in",
        "source_id": str(source_id),
        "surface": surface,
        "dedupe_key": f"companion.check_in:{source_id}",
        "tags": normalized_tags,
        "provenance": _explicit_provenance(
            route=route,
            action="manual_check_in",
            source_timestamp=event_timestamp,
        ),
        "metadata": {
            "title": display_title,
            "summary": _truncate_text(normalized_summary, max_length=1000),
            "check_in_title": normalized_title,
        },
    }


def record_reading_item_saved(*, user_id: str | int | None, item: Any) -> str | None:
    item_id = str(getattr(item, "id"))
    source_timestamp = getattr(item, "updated_at", None) or getattr(item, "created_at", None)
    return record_companion_activity(
        user_id=user_id,
        event_type="reading_item_saved",
        source_type="reading_item",
        source_id=item_id,
        surface="api.reading",
        dedupe_key=f"reading.save:{item_id}",
        tags=list(getattr(item, "tags", None) or []),
        provenance=_explicit_provenance(
            route="/api/v1/reading/save",
            action="save",
            source_timestamp=source_timestamp,
        ),
        metadata={
            "title": getattr(item, "title", None),
            "url": getattr(item, "url", None),
            "canonical_url": getattr(item, "canonical_url", None),
            "status": getattr(item, "status", None),
            "favorite": bool(getattr(item, "favorite", False)),
        },
    )


def record_reading_item_updated(*, user_id: str | int | None, item: Any) -> str | None:
    item_id = str(getattr(item, "id"))
    source_timestamp = getattr(item, "updated_at", None) or getattr(item, "created_at", None)
    return record_companion_activity(
        user_id=user_id,
        event_type="reading_item_updated",
        source_type="reading_item",
        source_id=item_id,
        surface="api.reading",
        dedupe_key=f"reading.update:{item_id}:{source_timestamp or 'na'}",
        tags=list(getattr(item, "tags", None) or []),
        provenance=_explicit_provenance(
            route=f"/api/v1/reading/items/{item_id}",
            action="update",
            source_timestamp=source_timestamp,
        ),
        metadata={
            "title": getattr(item, "title", None),
            "status": getattr(item, "status", None),
            "favorite": bool(getattr(item, "favorite", False)),
            "notes": getattr(item, "notes", None),
        },
    )


def record_reading_item_archived(*, user_id: str | int | None, item: Any) -> str | None:
    item_id = str(getattr(item, "id"))
    source_timestamp = getattr(item, "updated_at", None) or getattr(item, "created_at", None)
    return record_companion_activity(
        user_id=user_id,
        event_type="reading_item_archived",
        source_type="reading_item",
        source_id=item_id,
        surface="api.reading",
        dedupe_key=f"reading.archive:{item_id}:{source_timestamp or 'na'}",
        tags=list(getattr(item, "tags", None) or []),
        provenance=_explicit_provenance(
            route=f"/api/v1/reading/items/{item_id}",
            action="archive",
            source_timestamp=source_timestamp,
        ),
        metadata={
            "title": getattr(item, "title", None),
            "status": getattr(item, "status", None),
            "hard_delete": False,
        },
    )


def record_reading_item_deleted(*, user_id: str | int | None, item_id: int) -> str | None:
    return record_companion_activity(
        user_id=user_id,
        event_type="reading_item_deleted",
        source_type="reading_item",
        source_id=str(item_id),
        surface="api.reading",
        dedupe_key=f"reading.delete:{item_id}",
        provenance=_explicit_provenance(
            route=f"/api/v1/reading/items/{item_id}",
            action="delete",
        ),
        metadata={"hard_delete": True},
    )


def record_reading_note_linked(
    *,
    user_id: str | int | None,
    item_id: int,
    note_id: str,
    item_title: str | None = None,
    link_created_at: str | None = None,
) -> str | None:
    source_id = f"{item_id}:{note_id}"
    return record_companion_activity(
        user_id=user_id,
        event_type="reading_note_linked",
        source_type="reading_note_link",
        source_id=source_id,
        surface="api.reading",
        dedupe_key=f"reading.note.link:{source_id}:{link_created_at or 'na'}",
        provenance=_explicit_provenance(
            route=f"/api/v1/reading/items/{item_id}/links/note",
            action="link_note",
            source_timestamp=link_created_at,
        ),
        metadata={
            "item_id": item_id,
            "note_id": str(note_id),
            "title": item_title,
            "item_title": item_title,
        },
    )


def record_reading_note_unlinked(
    *,
    user_id: str | int | None,
    item_id: int,
    note_id: str,
    item_title: str | None = None,
    link_created_at: str | None = None,
) -> str | None:
    source_id = f"{item_id}:{note_id}"
    return record_companion_activity(
        user_id=user_id,
        event_type="reading_note_unlinked",
        source_type="reading_note_link",
        source_id=source_id,
        surface="api.reading",
        dedupe_key=f"reading.note.unlink:{source_id}:{link_created_at or 'na'}",
        provenance=_explicit_provenance(
            route=f"/api/v1/reading/items/{item_id}/links/note/{note_id}",
            action="unlink_note",
            source_timestamp=link_created_at,
        ),
        metadata={
            "item_id": item_id,
            "note_id": str(note_id),
            "title": item_title,
            "item_title": item_title,
        },
    )


def _reading_highlight_metadata(highlight: Any, *, item_title: str | None = None) -> dict[str, Any]:
    return {
        "item_id": getattr(highlight, "item_id", None),
        "title": item_title,
        "item_title": item_title,
        "quote": getattr(highlight, "quote", None),
        "note": getattr(highlight, "note", None),
        "color": getattr(highlight, "color", None),
        "state": getattr(highlight, "state", None),
        "anchor_strategy": getattr(highlight, "anchor_strategy", None),
        "start_offset": getattr(highlight, "start_offset", None),
        "end_offset": getattr(highlight, "end_offset", None),
    }


def record_reading_highlight_created(
    *,
    user_id: str | int | None,
    highlight: Any,
    item_title: str | None = None,
) -> str | None:
    highlight_id = str(getattr(highlight, "id"))
    item_id = int(getattr(highlight, "item_id"))
    created_at = getattr(highlight, "created_at", None)
    return record_companion_activity(
        user_id=user_id,
        event_type="reading_highlight_created",
        source_type="reading_highlight",
        source_id=highlight_id,
        surface="api.reading",
        dedupe_key=f"reading.highlight.create:{highlight_id}:{created_at or 'na'}",
        provenance=_explicit_provenance(
            route=f"/api/v1/reading/items/{item_id}/highlight",
            action="create_highlight",
            source_timestamp=created_at,
        ),
        metadata=_reading_highlight_metadata(highlight, item_title=item_title),
    )


def record_reading_highlight_updated(
    *,
    user_id: str | int | None,
    highlight: Any,
    item_title: str | None = None,
    patch: dict[str, Any] | None = None,
) -> str | None:
    highlight_id = str(getattr(highlight, "id"))
    fingerprint = _payload_fingerprint(patch)
    return record_companion_activity(
        user_id=user_id,
        event_type="reading_highlight_updated",
        source_type="reading_highlight",
        source_id=highlight_id,
        surface="api.reading",
        dedupe_key=f"reading.highlight.update:{highlight_id}:{fingerprint}",
        provenance=_explicit_provenance(
            route=f"/api/v1/reading/highlights/{highlight_id}",
            action="update_highlight",
            source_timestamp=getattr(highlight, "created_at", None),
        ),
        metadata={
            **_reading_highlight_metadata(highlight, item_title=item_title),
            "patch": dict(patch or {}),
        },
    )


def record_reading_highlight_deleted(
    *,
    user_id: str | int | None,
    highlight: Any,
    item_title: str | None = None,
) -> str | None:
    highlight_id = str(getattr(highlight, "id"))
    return record_companion_activity(
        user_id=user_id,
        event_type="reading_highlight_deleted",
        source_type="reading_highlight",
        source_id=highlight_id,
        surface="api.reading",
        dedupe_key=f"reading.highlight.delete:{highlight_id}",
        provenance=_explicit_provenance(
            route=f"/api/v1/reading/highlights/{highlight_id}",
            action="delete_highlight",
            source_timestamp=getattr(highlight, "created_at", None),
        ),
        metadata=_reading_highlight_metadata(highlight, item_title=item_title),
    )


def _value(payload: Any, key: str, default: Any = None) -> Any:
    if isinstance(payload, dict):
        return payload.get(key, default)
    return getattr(payload, key, default)


def _content_preview(value: Any, *, max_chars: int = 240) -> str | None:
    text = " ".join(str(value or "").strip().split())
    if not text:
        return None
    safe_limit = max(8, int(max_chars))
    if len(text) <= safe_limit:
        return text
    suffix = "... [truncated]"
    if safe_limit <= len(suffix):
        return text[:safe_limit]
    return f"{text[: safe_limit - len(suffix)]}{suffix}"


def _note_tags(note: Any) -> list[str] | None:
    keywords = _value(note, "keywords")
    if not isinstance(keywords, list):
        return None
    tags: list[str] = []
    for keyword in keywords:
        if isinstance(keyword, dict):
            raw_value = keyword.get("keyword")
        else:
            raw_value = getattr(keyword, "keyword", None)
        normalized = str(raw_value or "").strip()
        if normalized and normalized not in tags:
            tags.append(normalized)
    return tags or None


def _note_metadata(note: Any) -> dict[str, Any]:
    return {
        "title": _value(note, "title"),
        "content_preview": _content_preview(_value(note, "content")),
        "version": _value(note, "version"),
        "conversation_id": _value(note, "conversation_id"),
        "message_id": _value(note, "message_id"),
    }


def _note_timestamp(note: Any, *, prefer_created: bool = False) -> str | None:
    if prefer_created:
        return _value(note, "created_at") or _value(note, "last_modified") or _value(note, "updated_at")
    return _value(note, "last_modified") or _value(note, "updated_at") or _value(note, "created_at")


def build_note_bulk_import_activity(
    *,
    note: Any,
    operation: str,
    route: str,
    surface: str,
    patch: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a note companion activity payload for explicit bulk/import flows."""
    normalized_operation = str(operation or "").strip().lower()
    note_id = str(_value(note, "id"))
    if normalized_operation in {"import_create", "bulk_create"}:
        source_timestamp = _note_timestamp(note, prefer_created=True)
        return {
            "event_type": "note_created",
            "source_type": "note",
            "source_id": note_id,
            "surface": surface,
            "dedupe_key": f"notes.create:{note_id}",
            "tags": _note_tags(note),
            "provenance": _explicit_provenance(
                route=route,
                action=normalized_operation,
                source_timestamp=source_timestamp,
            ),
            "metadata": _note_metadata(note),
        }
    if normalized_operation == "import_overwrite":
        source_timestamp = _note_timestamp(note)
        version = _value(note, "version")
        fingerprint = _payload_fingerprint(patch)
        changed_fields = sorted(str(key) for key in dict(patch or {}).keys())
        return {
            "event_type": "note_updated",
            "source_type": "note",
            "source_id": note_id,
            "surface": surface,
            "dedupe_key": f"notes.update:{note_id}:{version or source_timestamp or 'na'}:{fingerprint}",
            "tags": _note_tags(note),
            "provenance": _explicit_provenance(
                route=route,
                action=normalized_operation,
                source_timestamp=source_timestamp,
            ),
            "metadata": {
                **_note_metadata(note),
                "changed_fields": changed_fields,
            },
        }
    raise ValueError(f"unsupported note bulk/import operation: {operation}")


def record_note_created(*, user_id: str | int | None, note: Any) -> str | None:
    note_id = str(_value(note, "id"))
    source_timestamp = _note_timestamp(note, prefer_created=True)
    return record_companion_activity(
        user_id=user_id,
        event_type="note_created",
        source_type="note",
        source_id=note_id,
        surface="api.notes",
        dedupe_key=f"notes.create:{note_id}",
        tags=_note_tags(note),
        provenance=_explicit_provenance(
            route="/api/v1/notes/",
            action="create",
            source_timestamp=source_timestamp,
        ),
        metadata=_note_metadata(note),
    )


def record_note_updated(
    *,
    user_id: str | int | None,
    note: Any,
    route: str,
    action: str,
    patch: dict[str, Any] | None = None,
) -> str | None:
    note_id = str(_value(note, "id"))
    source_timestamp = _note_timestamp(note)
    version = _value(note, "version")
    fingerprint = _payload_fingerprint(patch)
    changed_fields = sorted(str(key) for key in dict(patch or {}).keys())
    return record_companion_activity(
        user_id=user_id,
        event_type="note_updated",
        source_type="note",
        source_id=note_id,
        surface="api.notes",
        dedupe_key=f"notes.update:{note_id}:{version or source_timestamp or 'na'}:{fingerprint}",
        tags=_note_tags(note),
        provenance=_explicit_provenance(
            route=route,
            action=action,
            source_timestamp=source_timestamp,
        ),
        metadata={
            **_note_metadata(note),
            "changed_fields": changed_fields,
        },
    )


def record_note_deleted(
    *,
    user_id: str | int | None,
    note: Any,
    deleted_version: int | None = None,
) -> str | None:
    note_id = str(_value(note, "id"))
    source_timestamp = _note_timestamp(note)
    effective_version = deleted_version if deleted_version is not None else _value(note, "version")
    return record_companion_activity(
        user_id=user_id,
        event_type="note_deleted",
        source_type="note",
        source_id=note_id,
        surface="api.notes",
        dedupe_key=f"notes.delete:{note_id}:{effective_version or source_timestamp or 'na'}",
        tags=_note_tags(note),
        provenance=_explicit_provenance(
            route=f"/api/v1/notes/{note_id}",
            action="delete",
            source_timestamp=source_timestamp,
        ),
        metadata={
            **_note_metadata(note),
            "version": effective_version,
            "deleted": True,
            "hard_delete": False,
        },
    )


def record_note_restored(*, user_id: str | int | None, note: Any) -> str | None:
    note_id = str(_value(note, "id"))
    source_timestamp = _note_timestamp(note)
    version = _value(note, "version")
    return record_companion_activity(
        user_id=user_id,
        event_type="note_restored",
        source_type="note",
        source_id=note_id,
        surface="api.notes",
        dedupe_key=f"notes.restore:{note_id}:{version or source_timestamp or 'na'}",
        tags=_note_tags(note),
        provenance=_explicit_provenance(
            route=f"/api/v1/notes/{note_id}/restore",
            action="restore",
            source_timestamp=source_timestamp,
        ),
        metadata={
            **_note_metadata(note),
            "deleted": False,
            "hard_delete": False,
        },
    )


def record_persona_session_started(
    *,
    user_id: str | int | None,
    session_id: str,
    persona_id: str,
    runtime_mode: str | None,
    scope_snapshot_id: str | None,
    surface: str | None = None,
) -> str | None:
    normalized_surface = normalize_persona_activity_surface(surface)
    return record_companion_activity(
        user_id=user_id,
        event_type="persona_session_started",
        source_type="persona_session",
        source_id=session_id,
        surface=normalized_surface,
        dedupe_key=f"persona.session.started:{session_id}",
        tags=[persona_id],
        provenance=_explicit_provenance(
            route="/api/v1/persona/session",
            action="start",
        ),
        metadata={
            "persona_id": persona_id,
            "runtime_mode": runtime_mode,
            "scope_snapshot_id": scope_snapshot_id,
        },
    )


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


def _persona_tool_outcome_metadata(outcome: dict[str, Any] | None) -> dict[str, Any]:
    payload = dict(outcome or {}) if isinstance(outcome, dict) else {"value": str(outcome)}
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
    return {
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


def record_persona_session_summarized(
    *,
    user_id: str | int | None,
    session_id: str,
    persona_id: str,
    plan_id: str,
    step_idx: int,
    runtime_mode: str | None,
    scope_snapshot_id: str | None,
    summary_text: str,
    surface: str | None = None,
) -> str | None:
    normalized_summary = str(summary_text or "").strip()
    if not normalized_summary:
        return None
    summary_digest = hashlib.sha1(
        normalized_summary.encode("utf-8"),
        usedforsecurity=False,
    ).hexdigest()[:12]
    normalized_surface = normalize_persona_activity_surface(surface)
    return record_companion_activity(
        user_id=user_id,
        event_type="persona_session_summarized",
        source_type="persona_session",
        source_id=str(session_id),
        surface=normalized_surface,
        dedupe_key=f"persona.session.summary:{session_id}:{plan_id}:{step_idx}:{summary_digest}",
        tags=[str(persona_id)],
        provenance=_explicit_provenance(
            route="/api/v1/persona/stream",
            action="session_summary",
        ),
        metadata={
            "persona_id": str(persona_id),
            "runtime_mode": runtime_mode,
            "scope_snapshot_id": scope_snapshot_id,
            "plan_id": str(plan_id),
            "step_idx": int(step_idx),
            "summary_preview": _content_preview(normalized_summary),
            "summary_char_count": len(normalized_summary),
        },
    )


def record_persona_tool_executed(
    *,
    user_id: str | int | None,
    session_id: str,
    persona_id: str,
    plan_id: str,
    step_idx: int,
    step_type: str,
    tool_name: str,
    runtime_mode: str | None,
    scope_snapshot_id: str | None,
    outcome: dict[str, Any] | None,
    surface: str | None = None,
) -> str | None:
    normalized_step_type = str(step_type or "").strip().lower()
    normalized_tool_name = str(tool_name or "").strip()
    if normalized_step_type not in {"mcp_tool", "skill"}:
        return None
    if not normalized_tool_name:
        return None
    metadata = _persona_tool_outcome_metadata(outcome)
    if not bool(metadata.get("ok")):
        return None
    source_id = f"{session_id}:{plan_id}:{int(step_idx)}"
    fingerprint = _payload_fingerprint(metadata)
    normalized_surface = normalize_persona_activity_surface(surface)
    return record_companion_activity(
        user_id=user_id,
        event_type="persona_tool_executed",
        source_type="persona_tool_step",
        source_id=source_id,
        surface=normalized_surface,
        dedupe_key=f"persona.tool.executed:{source_id}:{normalized_tool_name}:{fingerprint}",
        tags=[str(persona_id), normalized_tool_name],
        provenance=_explicit_provenance(
            route="/api/v1/persona/stream",
            action="tool_outcome",
        ),
        metadata={
            "persona_id": str(persona_id),
            "plan_id": str(plan_id),
            "step_idx": int(step_idx),
            "step_type": normalized_step_type,
            "tool_name": normalized_tool_name,
            "runtime_mode": runtime_mode,
            "scope_snapshot_id": scope_snapshot_id,
            **metadata,
        },
    )


def _watchlist_source_metadata(source: Any) -> dict[str, Any]:
    settings = _value(source, "settings")
    if settings is None:
        raw_settings_json = _value(source, "settings_json")
        if isinstance(raw_settings_json, str) and raw_settings_json.strip():
            try:
                settings = json.loads(raw_settings_json)
            except Exception:
                settings = None
    settings_keys = sorted(str(key) for key in settings.keys()) if isinstance(settings, dict) else []
    return {
        "name": _value(source, "name"),
        "url": _value(source, "url"),
        "source_type": _value(source, "source_type"),
        "active": _value(source, "active"),
        "status": _value(source, "status"),
        "group_ids": list(_value(source, "group_ids") or []),
        "settings_keys": settings_keys,
    }


def _watchlist_source_timestamp(source: Any) -> str | None:
    return _value(source, "updated_at") or _value(source, "created_at")


def build_watchlist_source_bulk_import_activity(
    *,
    source: Any,
    operation: str,
    route: str,
    surface: str,
    patch: dict[str, Any] | None = None,
    event_timestamp: str | None = None,
) -> dict[str, Any]:
    """Build a watchlist source companion activity payload for bulk/import flows."""
    normalized_operation = str(operation or "").strip().lower()
    source_id = str(_value(source, "id"))
    if normalized_operation in {"import_create", "bulk_create"}:
        source_timestamp = event_timestamp or _watchlist_source_timestamp(source)
        return {
            "event_type": "watchlist_source_created",
            "source_type": "watchlist_source",
            "source_id": source_id,
            "surface": surface,
            "dedupe_key": f"watchlists.source.create:{source_id}",
            "tags": list(_value(source, "tags") or []),
            "provenance": _explicit_provenance(
                route=route,
                action=normalized_operation,
                source_timestamp=source_timestamp,
            ),
            "metadata": _watchlist_source_metadata(source),
        }
    if normalized_operation == "import_update":
        source_timestamp = event_timestamp or _watchlist_source_timestamp(source)
        fingerprint = _payload_fingerprint(patch)
        changed_fields = sorted(str(key) for key in dict(patch or {}).keys())
        return {
            "event_type": "watchlist_source_updated",
            "source_type": "watchlist_source",
            "source_id": source_id,
            "surface": surface,
            "dedupe_key": f"watchlists.source.update:{source_id}:{source_timestamp or 'na'}:{fingerprint}",
            "tags": list(_value(source, "tags") or []),
            "provenance": _explicit_provenance(
                route=route,
                action=normalized_operation,
                source_timestamp=source_timestamp,
            ),
            "metadata": {
                **_watchlist_source_metadata(source),
                "changed_fields": changed_fields,
            },
        }
    raise ValueError(f"unsupported watchlist source bulk/import operation: {operation}")


def record_watchlist_source_created(
    *,
    user_id: str | int | None,
    source: Any,
    route: str = "/api/v1/watchlists/sources",
    surface: str = "api.watchlists",
) -> str | None:
    source_id = str(_value(source, "id"))
    source_timestamp = _watchlist_source_timestamp(source)
    return record_companion_activity(
        user_id=user_id,
        event_type="watchlist_source_created",
        source_type="watchlist_source",
        source_id=source_id,
        surface=surface,
        dedupe_key=f"watchlists.source.create:{source_id}",
        tags=list(getattr(source, "tags", None) or []),
        provenance=_explicit_provenance(
            route=route,
            action="create",
            source_timestamp=source_timestamp,
        ),
        metadata=_watchlist_source_metadata(source),
    )


def record_watchlist_source_updated(
    *,
    user_id: str | int | None,
    source: Any,
    patch: dict[str, Any] | None = None,
    event_timestamp: str | None = None,
    route: str | None = None,
    surface: str = "api.watchlists",
) -> str | None:
    source_id = str(_value(source, "id"))
    source_timestamp = event_timestamp or _watchlist_source_timestamp(source)
    fingerprint = _payload_fingerprint(patch)
    changed_fields = sorted(str(key) for key in dict(patch or {}).keys())
    return record_companion_activity(
        user_id=user_id,
        event_type="watchlist_source_updated",
        source_type="watchlist_source",
        source_id=source_id,
        surface=surface,
        dedupe_key=f"watchlists.source.update:{source_id}:{source_timestamp or 'na'}:{fingerprint}",
        tags=list(_value(source, "tags") or []),
        provenance=_explicit_provenance(
            route=route or f"/api/v1/watchlists/sources/{source_id}",
            action="update",
            source_timestamp=source_timestamp,
        ),
        metadata={
            **_watchlist_source_metadata(source),
            "changed_fields": changed_fields,
        },
    )


def record_watchlist_source_deleted(
    *,
    user_id: str | int | None,
    source: Any,
    event_timestamp: str | None = None,
    restore_window_seconds: int | None = None,
    route: str | None = None,
    surface: str = "api.watchlists",
    hard_delete: bool = False,
) -> str | None:
    source_id = str(_value(source, "id"))
    source_timestamp = event_timestamp or _watchlist_source_timestamp(source)
    return record_companion_activity(
        user_id=user_id,
        event_type="watchlist_source_deleted",
        source_type="watchlist_source",
        source_id=source_id,
        surface=surface,
        dedupe_key=f"watchlists.source.delete:{source_id}:{source_timestamp or 'na'}",
        tags=list(_value(source, "tags") or []),
        provenance=_explicit_provenance(
            route=route or f"/api/v1/watchlists/sources/{source_id}",
            action="delete",
            source_timestamp=source_timestamp,
        ),
        metadata={
            **_watchlist_source_metadata(source),
            "deleted": True,
            "hard_delete": hard_delete,
            "restore_window_seconds": restore_window_seconds,
        },
    )


def record_watchlist_source_restored(
    *,
    user_id: str | int | None,
    source: Any,
    event_timestamp: str | None = None,
) -> str | None:
    source_id = str(_value(source, "id"))
    source_timestamp = event_timestamp or _watchlist_source_timestamp(source)
    return record_companion_activity(
        user_id=user_id,
        event_type="watchlist_source_restored",
        source_type="watchlist_source",
        source_id=source_id,
        surface="api.watchlists",
        dedupe_key=f"watchlists.source.restore:{source_id}:{source_timestamp or 'na'}",
        tags=list(_value(source, "tags") or []),
        provenance=_explicit_provenance(
            route=f"/api/v1/watchlists/sources/{source_id}/restore",
            action="restore",
            source_timestamp=source_timestamp,
        ),
        metadata={
            **_watchlist_source_metadata(source),
            "deleted": False,
            "hard_delete": False,
        },
    )


def _watchlist_item_tags(item: Any) -> list[str]:
    tags_value = _value(item, "tags")
    if isinstance(tags_value, list):
        return [str(tag) for tag in tags_value if str(tag).strip()]

    tags_fn = getattr(item, "tags", None)
    if callable(tags_fn):
        try:
            tags = tags_fn()
            if isinstance(tags, list):
                return [str(tag) for tag in tags if str(tag).strip()]
        except Exception:
            logger.debug("watchlist item tags() lookup skipped")

    raw_tags_json = _value(item, "tags_json")
    if isinstance(raw_tags_json, str) and raw_tags_json.strip():
        try:
            parsed = json.loads(raw_tags_json)
            if isinstance(parsed, list):
                return [str(tag) for tag in parsed if str(tag).strip()]
        except Exception:
            logger.debug("watchlist item tags_json parse skipped")

    return []


def _watchlist_item_metadata(item: Any) -> dict[str, Any]:
    return {
        "run_id": _value(item, "run_id"),
        "job_id": _value(item, "job_id"),
        "source_id": _value(item, "source_id"),
        "media_id": _value(item, "media_id"),
        "media_uuid": _value(item, "media_uuid"),
        "url": _value(item, "url"),
        "title": _value(item, "title"),
        "summary_preview": _content_preview(_value(item, "summary"), max_chars=200),
        "published_at": _value(item, "published_at"),
        "status": _value(item, "status"),
        "reviewed": bool(_value(item, "reviewed", False)),
        "queued_for_briefing": bool(_value(item, "queued_for_briefing", False)),
    }


def _watchlist_item_timestamp(item: Any) -> str | None:
    return _value(item, "created_at")


def record_watchlist_item_added(
    *,
    user_id: str | int | None,
    item: Any,
    route: str,
) -> str | None:
    """Record a watchlist item that was explicitly ingested into collections."""
    return record_companion_activity(
        user_id=user_id,
        **build_watchlist_item_added_activity(item=item, route=route),
    )


def build_watchlist_item_added_activity(
    *,
    item: Any,
    route: str,
) -> dict[str, Any]:
    """Build a normalized companion payload for a watchlist item ingest event."""
    item_id = str(_value(item, "id"))
    source_timestamp = _watchlist_item_timestamp(item)
    return {
        "event_type": "watchlist_item_added",
        "source_type": "watchlist_item",
        "source_id": item_id,
        "surface": "api.watchlists",
        "dedupe_key": f"watchlists.item.add:{item_id}",
        "tags": _watchlist_item_tags(item),
        "provenance": _explicit_provenance(
            route=route,
            action="item_ingested",
            source_timestamp=source_timestamp,
        ),
        "metadata": _watchlist_item_metadata(item),
    }


def record_watchlist_item_updated(
    *,
    user_id: str | int | None,
    item: Any,
    patch: dict[str, Any] | None = None,
    event_timestamp: str | None = None,
) -> str | None:
    item_id = str(_value(item, "id"))
    source_timestamp = event_timestamp or _watchlist_item_timestamp(item)
    fingerprint = _payload_fingerprint(patch)
    changed_fields = sorted(str(key) for key in dict(patch or {}).keys())
    return record_companion_activity(
        user_id=user_id,
        event_type="watchlist_item_updated",
        source_type="watchlist_item",
        source_id=item_id,
        surface="api.watchlists",
        dedupe_key=f"watchlists.item.update:{item_id}:{source_timestamp or 'na'}:{fingerprint}",
        tags=_watchlist_item_tags(item),
        provenance=_explicit_provenance(
            route=f"/api/v1/watchlists/items/{item_id}",
            action="update",
            source_timestamp=source_timestamp,
        ),
        metadata={
            **_watchlist_item_metadata(item),
            "changed_fields": changed_fields,
        },
    )


def _reminder_task_metadata(task: Any) -> dict[str, Any]:
    return {
        "title": _value(task, "title"),
        "body_preview": _content_preview(_value(task, "body"), max_chars=160),
        "schedule_kind": _value(task, "schedule_kind"),
        "enabled": _value(task, "enabled"),
        "link_type": _value(task, "link_type"),
        "link_id": _value(task, "link_id"),
        "link_url": _value(task, "link_url"),
    }


def _reminder_task_timestamp(task: Any) -> str | None:
    return _value(task, "updated_at") or _value(task, "created_at")


def record_reminder_task_created(*, user_id: str | int | None, task: Any) -> str | None:
    task_id = str(_value(task, "id"))
    source_timestamp = _reminder_task_timestamp(task)
    return record_companion_activity(
        user_id=user_id,
        event_type="reminder_task_created",
        source_type="reminder_task",
        source_id=task_id,
        surface="api.tasks",
        dedupe_key=f"reminder.task.create:{task_id}",
        provenance=_explicit_provenance(
            route="/api/v1/tasks",
            action="create",
            source_timestamp=source_timestamp,
        ),
        metadata=_reminder_task_metadata(task),
    )


def record_reminder_task_updated(
    *,
    user_id: str | int | None,
    task: Any,
    patch: dict[str, Any] | None = None,
) -> str | None:
    task_id = str(_value(task, "id"))
    source_timestamp = _reminder_task_timestamp(task)
    fingerprint = _payload_fingerprint(patch)
    changed_fields = sorted(str(key) for key in dict(patch or {}).keys())
    return record_companion_activity(
        user_id=user_id,
        event_type="reminder_task_updated",
        source_type="reminder_task",
        source_id=task_id,
        surface="api.tasks",
        dedupe_key=f"reminder.task.update:{task_id}:{source_timestamp or 'na'}:{fingerprint}",
        provenance=_explicit_provenance(
            route=f"/api/v1/tasks/{task_id}",
            action="update",
            source_timestamp=source_timestamp,
        ),
        metadata={
            **_reminder_task_metadata(task),
            "changed_fields": changed_fields,
        },
    )


def record_reminder_task_deleted(*, user_id: str | int | None, task: Any) -> str | None:
    task_id = str(_value(task, "id"))
    source_timestamp = _reminder_task_timestamp(task)
    return record_companion_activity(
        user_id=user_id,
        event_type="reminder_task_deleted",
        source_type="reminder_task",
        source_id=task_id,
        surface="api.tasks",
        dedupe_key=f"reminder.task.delete:{task_id}:{source_timestamp or 'na'}",
        provenance=_explicit_provenance(
            route=f"/api/v1/tasks/{task_id}",
            action="delete",
            source_timestamp=source_timestamp,
        ),
        metadata={
            **_reminder_task_metadata(task),
            "hard_delete": True,
        },
    )
