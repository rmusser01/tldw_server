"""Persona Live control service for session summaries, focus, and lifecycle state."""

from __future__ import annotations

from threading import RLock
import time
from typing import Any

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.Persona.session_manager import SessionManager
from tldw_Server_API.app.core.Persona.session_materialization import (
    ensure_default_persona_profile,
    materialize_persona_session,
)
from tldw_Server_API.app.core.Personalization.companion_activity import normalize_persona_activity_surface


LIVE_CONTROL_PREFS_KEY = "persona_live_control"
_TERMINAL_STATUSES = {"closed", "archived"}
_CAPABILITIES = {"text": True, "voice": False, "browser_microphone_required": False}
_generation_lock = RLock()
_last_focus_generation = 0


class PersonaLiveStreamRegistry:
    """Process-local registry of active Persona Live WebSocket stream presence."""

    def __init__(self) -> None:
        self._connected: set[tuple[str, str]] = set()
        self._lock = RLock()

    def mark_connected(self, *, user_id: str, session_id: str) -> None:
        uid = str(user_id or "").strip()
        sid = str(session_id or "").strip()
        if not uid or not sid:
            return
        with self._lock:
            self._connected.add((uid, sid))

    def mark_disconnected(self, *, user_id: str, session_id: str) -> None:
        uid = str(user_id or "").strip()
        sid = str(session_id or "").strip()
        if not uid or not sid:
            return
        with self._lock:
            self._connected.discard((uid, sid))

    def is_connected(self, *, user_id: str, session_id: str) -> bool:
        uid = str(user_id or "").strip()
        sid = str(session_id or "").strip()
        if not uid or not sid:
            return False
        with self._lock:
            return (uid, sid) in self._connected

    def clear(self) -> None:
        with self._lock:
            self._connected.clear()


persona_live_stream_registry = PersonaLiveStreamRegistry()


def _next_focus_generation() -> int:
    global _last_focus_generation
    with _generation_lock:
        candidate = int(time.time() * 1000)
        _last_focus_generation = max(candidate, _last_focus_generation + 1)
        return _last_focus_generation


def _session_preferences(row: dict[str, Any]) -> dict[str, Any]:
    preferences = row.get("preferences")
    return dict(preferences) if isinstance(preferences, dict) else {}


def _live_control_preferences(row: dict[str, Any]) -> dict[str, Any]:
    payload = _session_preferences(row).get(LIVE_CONTROL_PREFS_KEY)
    return dict(payload) if isinstance(payload, dict) else {}


def _focus_metadata(row: dict[str, Any]) -> dict[str, Any]:
    focus = _live_control_preferences(row).get("focus")
    return dict(focus) if isinstance(focus, dict) else {}


def _is_terminal(row: dict[str, Any]) -> bool:
    return str(row.get("status") or "active").strip().lower() in _TERMINAL_STATUSES


def _live_preferences_patch(
    row: dict[str, Any],
    *,
    focus: dict[str, Any] | None = None,
    idempotency_key: str | None | object = ...,
) -> dict[str, Any]:
    preferences = _session_preferences(row)
    live = _live_control_preferences(row)
    if focus is not None:
        live["focus"] = focus
    if idempotency_key is not ...:
        if idempotency_key is None:
            live.pop("create_idempotency_key", None)
        else:
            live["create_idempotency_key"] = str(idempotency_key)
    if live:
        preferences[LIVE_CONTROL_PREFS_KEY] = live
    else:
        preferences.pop(LIVE_CONTROL_PREFS_KEY, None)
    return preferences


def _update_preferences(db: CharactersRAGDB, *, row: dict[str, Any], user_id: str, preferences: dict[str, Any]) -> None:
    session_id = str(row.get("id") or "").strip()
    if not session_id:
        return
    db.update_persona_session(
        session_id=session_id,
        user_id=user_id,
        update_data={"preferences_json": preferences},
    )


def _load_owned_session_or_raise(db: CharactersRAGDB, *, user_id: str, session_id: str) -> dict[str, Any]:
    row = db.get_persona_session(session_id, user_id=user_id, include_deleted=False)
    if row is not None:
        return row
    cursor = db.execute_query(
        "SELECT user_id FROM persona_sessions WHERE id = ? AND deleted = 0",
        (session_id,),
    )
    existing = cursor.fetchone()
    if existing:
        raise PermissionError("session ownership mismatch")
    raise FileNotFoundError("Persona session not found")


def _persona_name_for_row(db: CharactersRAGDB, *, row: dict[str, Any], user_id: str) -> str:
    persona_id = str(row.get("persona_id") or "").strip()
    profile = db.get_persona_profile(persona_id, user_id=user_id, include_deleted=False)
    if profile is None and persona_id == "research_assistant":
        profile = ensure_default_persona_profile(db, user_id=user_id)
    if isinstance(profile, dict):
        name = str(profile.get("name") or "").strip()
        if name:
            return name
    return persona_id


def build_live_session_summary(
    db: CharactersRAGDB,
    *,
    session_manager: SessionManager,
    user_id: str,
    row: dict[str, Any],
    is_focused: bool = False,
    stream_registry: PersonaLiveStreamRegistry = persona_live_stream_registry,
) -> dict[str, Any]:
    session_id = str(row.get("id") or "").strip()
    persona_id = str(row.get("persona_id") or "").strip()
    status = str(row.get("status") or "active").strip().lower() or "active"
    terminal = status in _TERMINAL_STATUSES
    if terminal:
        lifecycle = "stopped"
    elif stream_registry.is_connected(user_id=user_id, session_id=session_id):
        lifecycle = "connected"
    else:
        lifecycle = "idle"

    focus = _focus_metadata(row)
    return {
        "session_id": session_id,
        "persona_id": persona_id,
        "persona_name": _persona_name_for_row(db, row=row, user_id=user_id),
        "lifecycle": lifecycle,
        "status": status,
        "is_focused": bool(is_focused),
        "focused_at": str(focus.get("focused_at") or "").strip() or None if is_focused else None,
        "focus_generation": int(focus["focus_generation"]) if is_focused and focus.get("focus_generation") is not None else None,
        "last_activity_at": str(row.get("last_modified") or row.get("created_at") or "").strip() or None,
        "pending_approval_count": 0,
        "active_tool_name": None,
        "error_state": None,
        "recovery_hint": None,
        "suggested_visual_state": "idle" if not terminal else "offline",
        "allowed_actions": [] if terminal else ["focus", "stop", "send_text_ws"],
        "capabilities": dict(_CAPABILITIES),
    }


def _focused_session_id(rows: list[dict[str, Any]]) -> str | None:
    winner_id: str | None = None
    winner_generation: int | None = None
    for row in rows:
        if _is_terminal(row):
            continue
        focus = _focus_metadata(row)
        if not focus.get("focused"):
            continue
        try:
            generation = int(focus.get("focus_generation"))
        except (TypeError, ValueError):
            continue
        if winner_generation is None or generation > winner_generation:
            winner_generation = generation
            winner_id = str(row.get("id") or "").strip() or None
    return winner_id


def list_live_session_summaries(
    db: CharactersRAGDB,
    *,
    session_manager: SessionManager,
    user_id: str,
    persona_id: str | None = None,
    surface: str | None = None,
    limit: int = 100,
    stream_registry: PersonaLiveStreamRegistry = persona_live_stream_registry,
) -> dict[str, Any]:
    normalized_surface = normalize_persona_activity_surface(surface) if surface is not None else None
    rows = db.list_persona_sessions(
        user_id=user_id,
        persona_id=persona_id,
        activity_surface=normalized_surface,
        include_deleted=False,
        limit=limit,
        offset=0,
    )
    focused_id = _focused_session_id(rows)
    return {
        "sessions": [
            build_live_session_summary(
                db,
                session_manager=session_manager,
                user_id=user_id,
                row=row,
                is_focused=str(row.get("id") or "").strip() == focused_id,
                stream_registry=stream_registry,
            )
            for row in rows
        ],
        "focused_session_id": focused_id,
    }


def _find_session_by_idempotency_key(
    db: CharactersRAGDB,
    *,
    user_id: str,
    persona_id: str,
    surface: str | None,
    idempotency_key: str,
) -> dict[str, Any] | None:
    rows = db.list_persona_sessions(
        user_id=user_id,
        persona_id=persona_id,
        activity_surface=normalize_persona_activity_surface(surface) if surface is not None else None,
        include_deleted=False,
        limit=200,
        offset=0,
    )
    for row in rows:
        live = _live_control_preferences(row)
        if str(live.get("create_idempotency_key") or "") == idempotency_key:
            return row
    return None


def _find_resume_compatible_session(
    db: CharactersRAGDB,
    *,
    user_id: str,
    persona_id: str,
    surface: str | None,
) -> dict[str, Any] | None:
    rows = db.list_persona_sessions(
        user_id=user_id,
        persona_id=persona_id,
        activity_surface=normalize_persona_activity_surface(surface) if surface is not None else None,
        include_deleted=False,
        limit=100,
        offset=0,
    )
    for row in rows:
        if not _is_terminal(row):
            return row
    return None


def create_or_resume_live_session(
    db: CharactersRAGDB,
    *,
    session_manager: SessionManager,
    user_id: str,
    persona_id: str,
    reuse_policy: str = "resume_compatible",
    idempotency_key: str | None = None,
    surface: str | None = None,
    stream_registry: PersonaLiveStreamRegistry = persona_live_stream_registry,
) -> dict[str, Any]:
    requested_persona_id = str(persona_id or "").strip()
    requested_key = str(idempotency_key or "").strip() or None
    row: dict[str, Any] | None = None

    if requested_key:
        row = _find_session_by_idempotency_key(
            db,
            user_id=user_id,
            persona_id=requested_persona_id,
            surface=surface,
            idempotency_key=requested_key,
        )

    if row is None and reuse_policy == "resume_compatible":
        row = _find_resume_compatible_session(
            db,
            user_id=user_id,
            persona_id=requested_persona_id,
            surface=surface,
        )

    if row is None:
        materialized = materialize_persona_session(
            db,
            session_manager=session_manager,
            user_id=user_id,
            persona_id=requested_persona_id,
            surface=surface,
        )
        row = materialized.session_row
    else:
        materialized = materialize_persona_session(
            db,
            session_manager=session_manager,
            user_id=user_id,
            persona_id=requested_persona_id,
            resume_session_id=str(row.get("id") or ""),
            surface=surface,
        )
        row = materialized.session_row

    if requested_key:
        preferences = _live_preferences_patch(row, idempotency_key=requested_key)
        _update_preferences(db, row=row, user_id=user_id, preferences=preferences)
        row = db.get_persona_session(str(row.get("id") or ""), user_id=user_id, include_deleted=False) or row

    return focus_live_session(
        db,
        session_manager=session_manager,
        user_id=user_id,
        session_id=str(row.get("id") or ""),
        stream_registry=stream_registry,
    )


def focus_live_session(
    db: CharactersRAGDB,
    *,
    session_manager: SessionManager,
    user_id: str,
    session_id: str,
    stream_registry: PersonaLiveStreamRegistry = persona_live_stream_registry,
) -> dict[str, Any]:
    row = _load_owned_session_or_raise(db, user_id=user_id, session_id=session_id)
    if _is_terminal(row):
        raise ValueError("Cannot focus a terminal persona session.")
    generation = _next_focus_generation()
    focused_at = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.gmtime())
    rows = db.list_persona_sessions(user_id=user_id, include_deleted=False, limit=200, offset=0)
    for existing in rows:
        existing_id = str(existing.get("id") or "").strip()
        if not existing_id:
            continue
        if existing_id == session_id:
            focus = {"focused": True, "focused_at": focused_at, "focus_generation": generation}
        else:
            focus = {"focused": False}
        _update_preferences(
            db,
            row=existing,
            user_id=user_id,
            preferences=_live_preferences_patch(existing, focus=focus),
        )
    row = db.get_persona_session(session_id, user_id=user_id, include_deleted=False) or row
    return build_live_session_summary(
        db,
        session_manager=session_manager,
        user_id=user_id,
        row=row,
        is_focused=True,
        stream_registry=stream_registry,
    )


def stop_live_session(
    db: CharactersRAGDB,
    *,
    session_manager: SessionManager,
    user_id: str,
    session_id: str,
    stream_registry: PersonaLiveStreamRegistry = persona_live_stream_registry,
) -> dict[str, Any]:
    row = _load_owned_session_or_raise(db, user_id=user_id, session_id=session_id)
    preferences = _live_preferences_patch(row, focus={"focused": False})
    db.update_persona_session(
        session_id=session_id,
        user_id=user_id,
        update_data={"status": "closed", "preferences_json": preferences},
    )
    stream_registry.mark_disconnected(user_id=user_id, session_id=session_id)
    row = db.get_persona_session(session_id, user_id=user_id, include_deleted=False) or row
    return build_live_session_summary(
        db,
        session_manager=session_manager,
        user_id=user_id,
        row=row,
        is_focused=False,
        stream_registry=stream_registry,
    )
