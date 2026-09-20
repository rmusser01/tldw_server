"""Persona Live control service for session summaries, focus, and lifecycle state."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from threading import RLock
from typing import Any, Callable

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, ConflictError
from tldw_Server_API.app.core.Persona.live_conversation import persona_live_turn_registry
from tldw_Server_API.app.core.Persona.live_voice_runtime import persona_live_voice_registry
from tldw_Server_API.app.core.Persona.session_manager import SessionManager
from tldw_Server_API.app.core.Persona.session_materialization import (
    DEFAULT_PERSONA_ID,
    ensure_default_persona_profile,
    materialize_persona_session,
)
from tldw_Server_API.app.core.Personalization.companion_activity import normalize_persona_activity_surface

LIVE_CONTROL_PREFS_KEY = "persona_live_control"
_TERMINAL_STATUSES = {"closed", "archived"}
_CAPABILITIES = {"text": True, "voice": False, "browser_microphone_required": False}
_generation_lock = RLock()
_live_session_mutation_lock = RLock()
_last_focus_generation = 0
_PREFERENCE_UPDATE_RETRIES = 3
_FOCUS_RECONCILIATION_PASSES = 3
_SESSION_SCAN_PAGE_SIZE = 200
_FOCUSED_SESSION_LOOKUP_LIMIT = 1000


class PersonaLiveStreamRegistry:
    """Process-local registry of active Persona Live WebSocket stream presence."""

    def __init__(self) -> None:
        self._connected: dict[tuple[str, str], int] = {}
        self._stop_callbacks: dict[tuple[str, str, str], Callable[[], None]] = {}
        self._lock = RLock()

    def mark_connected(self, *, user_id: str, session_id: str,
                       connection_id: str = "", on_stop: Callable[[], None] | None = None) -> None:
        uid = str(user_id or "").strip()
        sid = str(session_id or "").strip()
        if not uid or not sid:
            return
        with self._lock:
            key = (uid, sid)
            self._connected[key] = self._connected.get(key, 0) + 1
            if on_stop is not None and connection_id:
                self._stop_callbacks[(uid, sid, connection_id)] = on_stop

    def mark_disconnected(self, *, user_id: str, session_id: str, connection_id: str = "") -> None:
        uid = str(user_id or "").strip()
        sid = str(session_id or "").strip()
        if not uid or not sid:
            return
        with self._lock:
            self._stop_callbacks.pop((uid, sid, connection_id), None)
            key = (uid, sid)
            count = self._connected.get(key, 0) - 1
            if count > 0:
                self._connected[key] = count
            else:
                self._connected.pop(key, None)

    def is_connected(self, *, user_id: str, session_id: str) -> bool:
        uid = str(user_id or "").strip()
        sid = str(session_id or "").strip()
        if not uid or not sid:
            return False
        with self._lock:
            return self._connected.get((uid, sid), 0) > 0

    def clear(self) -> None:
        with self._lock:
            self._connected.clear()
            self._stop_callbacks.clear()

    def stop(self, *, user_id: str, session_id: str) -> None:
        """Notify only connections bound to this owned session."""
        with self._lock:
            callbacks = [callback for (uid, sid, _), callback in self._stop_callbacks.items()
                         if (uid, sid) == (str(user_id), str(session_id))]
        for callback in callbacks:
            try:
                callback()
            except RuntimeError:
                # A disconnect may close the owning loop after this snapshot.
                continue


persona_live_stream_registry = PersonaLiveStreamRegistry()


def _next_focus_generation() -> int:
    global _last_focus_generation
    with _generation_lock:
        candidate = int(time.time() * 1000)
        _last_focus_generation = max(candidate, _last_focus_generation + 1)
        return _last_focus_generation


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    expected_version = row.get("version")
    db.update_persona_session(
        session_id=session_id,
        user_id=user_id,
        update_data={"preferences_json": preferences},
        expected_version=int(expected_version) if expected_version is not None else None,
    )


def _update_live_control_preferences(
    db: CharactersRAGDB,
    *,
    row: dict[str, Any],
    user_id: str,
    focus: dict[str, Any] | None = None,
    idempotency_key: str | None | object = ...,
) -> None:
    session_id = str(row.get("id") or "").strip()
    if not session_id:
        return
    last_conflict: ConflictError | None = None
    latest_row = row
    for _ in range(_PREFERENCE_UPDATE_RETRIES):
        fresh_row = db.get_persona_session(session_id, user_id=user_id, include_deleted=False) or latest_row
        try:
            _update_preferences(
                db,
                row=fresh_row,
                user_id=user_id,
                preferences=_live_preferences_patch(fresh_row, focus=focus, idempotency_key=idempotency_key),
            )
            return
        except ConflictError as exc:
            last_conflict = exc
            latest_row = db.get_persona_session(session_id, user_id=user_id, include_deleted=False) or fresh_row
    if last_conflict is not None:
        raise last_conflict


def _load_owned_session_or_raise(db: CharactersRAGDB, *, user_id: str, session_id: str) -> dict[str, Any]:
    row = db.get_persona_session(session_id, user_id=user_id, include_deleted=False)
    if row is not None:
        return row
    raise FileNotFoundError("Persona session not found")


def _persona_name_for_row(db: CharactersRAGDB, *, row: dict[str, Any], user_id: str) -> str:
    persona_id = str(row.get("persona_id") or "").strip()
    profile = db.get_persona_profile(persona_id, user_id=user_id, include_deleted=False)
    if profile is None and persona_id == DEFAULT_PERSONA_ID:
        profile = ensure_default_persona_profile(db, user_id=user_id)
    return _persona_name_from_profile(profile, fallback=persona_id)


def _persona_name_from_profile(profile: dict[str, Any] | None, *, fallback: str) -> str:
    if isinstance(profile, dict):
        name = str(profile.get("name") or "").strip()
        if name:
            return name
    return fallback


def _persona_names_for_rows(db: CharactersRAGDB, *, user_id: str, rows: list[dict[str, Any]]) -> dict[str, str]:
    persona_ids: list[str] = []
    seen: set[str] = set()
    for row in rows:
        persona_id = str(row.get("persona_id") or "").strip()
        if not persona_id or persona_id in seen:
            continue
        persona_ids.append(persona_id)
        seen.add(persona_id)
    if not persona_ids:
        return {}

    profile_lookup = getattr(db, "get_persona_profiles_by_ids", None)
    if callable(profile_lookup):
        profiles_by_id = profile_lookup(
            user_id=user_id,
            persona_ids=persona_ids,
            include_deleted=False,
        )
    else:
        profiles_by_id: dict[str, dict[str, Any]] = {}
        offset = 0
        while seen - profiles_by_id.keys():
            page = db.list_persona_profiles(
                user_id=user_id,
                include_deleted=False,
                limit=_SESSION_SCAN_PAGE_SIZE,
                offset=offset,
            )
            for profile in page:
                profile_id = str(profile.get("id") or "").strip()
                if profile_id in seen:
                    profiles_by_id[profile_id] = profile
            if len(page) < _SESSION_SCAN_PAGE_SIZE:
                break
            offset += _SESSION_SCAN_PAGE_SIZE
    if DEFAULT_PERSONA_ID in seen and DEFAULT_PERSONA_ID not in profiles_by_id:
        profiles_by_id[DEFAULT_PERSONA_ID] = ensure_default_persona_profile(db, user_id=user_id)
    return {
        persona_id: _persona_name_from_profile(profiles_by_id.get(persona_id), fallback=persona_id)
        for persona_id in persona_ids
    }


def _resolve_live_control_persona_id(db: CharactersRAGDB, *, user_id: str, persona_id: str | None) -> str:
    requested_persona_id = str(persona_id or "").strip()
    if not requested_persona_id:
        profile = ensure_default_persona_profile(db, user_id=user_id)
        return str((profile or {}).get("id") or DEFAULT_PERSONA_ID).strip() or DEFAULT_PERSONA_ID
    profile = db.get_persona_profile(requested_persona_id, user_id=user_id, include_deleted=False)
    if profile is None and requested_persona_id == DEFAULT_PERSONA_ID:
        profile = ensure_default_persona_profile(db, user_id=user_id)
    if profile is None:
        raise FileNotFoundError("Persona not found")
    return str((profile or {}).get("id") or DEFAULT_PERSONA_ID).strip() or DEFAULT_PERSONA_ID


def _iter_persona_session_rows(
    db: CharactersRAGDB,
    *,
    user_id: str,
    persona_id: str | None = None,
    surface: str | None = None,
    include_deleted: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        page = db.list_persona_sessions(
            user_id=user_id,
            persona_id=persona_id,
            activity_surface=normalize_persona_activity_surface(surface) if surface is not None else None,
            include_deleted=include_deleted,
            limit=_SESSION_SCAN_PAGE_SIZE,
            offset=offset,
        )
        rows.extend(page)
        if len(page) < _SESSION_SCAN_PAGE_SIZE:
            break
        offset += _SESSION_SCAN_PAGE_SIZE
    return rows


def _focused_session_rows(
    db: CharactersRAGDB,
    *,
    user_id: str,
    persona_id: str | None = None,
    surface: str | None = None,
) -> list[dict[str, Any]]:
    focused_lookup = getattr(db, "list_focused_persona_sessions", None)
    if callable(focused_lookup):
        return focused_lookup(
            user_id=user_id,
            persona_id=persona_id,
            activity_surface=normalize_persona_activity_surface(surface) if surface is not None else None,
            include_deleted=False,
            limit=_FOCUSED_SESSION_LOOKUP_LIMIT,
        )
    rows = _iter_persona_session_rows(
        db,
        user_id=user_id,
        persona_id=persona_id,
        surface=surface,
    )
    return [row for row in rows if bool(_focus_metadata(row).get("focused"))]


def build_live_session_summary(
    db: CharactersRAGDB,
    *,
    user_id: str,
    row: dict[str, Any],
    is_focused: bool = False,
    persona_name: str | None = None,
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
        "persona_name": (
            persona_name if persona_name is not None else _persona_name_for_row(db, row=row, user_id=user_id)
        ),
        "lifecycle": lifecycle,
        "status": status,
        "is_focused": bool(is_focused),
        "focused_at": str(focus.get("focused_at") or "").strip() or None if is_focused else None,
        "focus_generation": (
            int(focus["focus_generation"]) if is_focused and focus.get("focus_generation") is not None else None
        ),
        "last_activity_at": str(row.get("last_modified") or row.get("created_at") or "").strip() or None,
        "pending_approval_count": 0,
        "active_tool_name": None,
        "error_state": None,
        "recovery_hint": None,
        "suggested_visual_state": "idle" if not terminal else "offline",
        "allowed_actions": [] if terminal else ["focus", "stop", "send_text_ws"],
        "capabilities": {
            **_CAPABILITIES,
            "voice": not terminal and persona_live_voice_registry.is_ready(user_id=user_id, session_id=session_id),
            "browser_microphone_required": not terminal and persona_live_voice_registry.is_ready(user_id=user_id, session_id=session_id),
        },
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


def _clear_other_focused_sessions(
    db: CharactersRAGDB,
    *,
    user_id: str,
    target_session_id: str,
    focused_rows: list[dict[str, Any]],
) -> list[str]:
    """Clear known non-target focused rows and report rows that still conflicted."""
    conflicted_session_ids: list[str] = []
    for existing in focused_rows:
        existing_id = str(existing.get("id") or "").strip()
        if not existing_id or existing_id == target_session_id:
            continue
        try:
            _update_live_control_preferences(
                db,
                row=existing,
                user_id=user_id,
                focus={"focused": False},
            )
        except ConflictError:
            conflicted_session_ids.append(existing_id)
    return conflicted_session_ids


def _non_target_focused_rows(rows: list[dict[str, Any]], *, target_session_id: str) -> list[dict[str, Any]]:
    """Return focused rows that do not represent the desired target session."""
    return [
        row
        for row in rows
        if str(row.get("id") or "").strip() != target_session_id and _focus_metadata(row).get("focused")
    ]


def _reconcile_focused_sessions(
    db: CharactersRAGDB,
    *,
    user_id: str,
    target_session_id: str,
    initially_focused_rows: list[dict[str, Any]],
) -> None:
    """Run bounded focus cleanup until only the target row remains focused."""
    focused_rows = initially_focused_rows
    for _ in range(_FOCUS_RECONCILIATION_PASSES):
        _clear_other_focused_sessions(
            db,
            user_id=user_id,
            target_session_id=target_session_id,
            focused_rows=focused_rows,
        )
        remaining = _non_target_focused_rows(
            _focused_session_rows(db, user_id=user_id),
            target_session_id=target_session_id,
        )
        if not remaining:
            return
        focused_rows = remaining

    target_row = db.get_persona_session(target_session_id, user_id=user_id, include_deleted=False)
    if target_row is not None:
        try:
            _update_live_control_preferences(
                db,
                row=target_row,
                user_id=user_id,
                focus={"focused": False},
            )
        except ConflictError:
            pass
    raise ConflictError(
        "Unable to reconcile focused Persona Live sessions.",
        entity="persona_sessions",
        entity_id=target_session_id,
    )


def list_live_session_summaries(
    db: CharactersRAGDB,
    *,
    user_id: str,
    persona_id: str | None = None,
    surface: str | None = None,
    limit: int = 100,
    stream_registry: PersonaLiveStreamRegistry = persona_live_stream_registry,
) -> dict[str, Any]:
    """Return bounded Persona Live session summaries plus the current focused session id."""
    normalized_surface = normalize_persona_activity_surface(surface) if surface is not None else None
    rows = db.list_persona_sessions(
        user_id=user_id,
        persona_id=persona_id,
        activity_surface=normalized_surface,
        include_deleted=False,
        limit=limit,
        offset=0,
    )
    focus_rows = _focused_session_rows(
        db,
        user_id=user_id,
        persona_id=persona_id,
        surface=normalized_surface,
    )
    focused_id = _focused_session_id(focus_rows)
    persona_names = _persona_names_for_rows(db, user_id=user_id, rows=rows)
    return {
        "sessions": [
            build_live_session_summary(
                db,
                user_id=user_id,
                row=row,
                is_focused=str(row.get("id") or "").strip() == focused_id,
                persona_name=persona_names.get(str(row.get("persona_id") or "").strip()),
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
    rows = _iter_persona_session_rows(
        db,
        user_id=user_id,
        persona_id=persona_id,
        surface=normalize_persona_activity_surface(surface),
    )
    for row in rows:
        if _is_terminal(row):
            continue
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
    rows = _iter_persona_session_rows(
        db,
        user_id=user_id,
        persona_id=persona_id,
        surface=normalize_persona_activity_surface(surface),
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
    """Create or resume a Persona Live session using the requested reuse policy and focus it."""
    with _live_session_mutation_lock:
        return _create_or_resume_live_session_unlocked(
            db,
            session_manager=session_manager,
            user_id=user_id,
            persona_id=persona_id,
            reuse_policy=reuse_policy,
            idempotency_key=idempotency_key,
            surface=surface,
            stream_registry=stream_registry,
        )


def _create_or_resume_live_session_unlocked(
    db: CharactersRAGDB,
    *,
    session_manager: SessionManager,
    user_id: str,
    persona_id: str,
    reuse_policy: str,
    idempotency_key: str | None,
    surface: str | None,
    stream_registry: PersonaLiveStreamRegistry,
) -> dict[str, Any]:
    """Create or resume a live session while the caller holds the mutation lock."""
    resolved_persona_id = _resolve_live_control_persona_id(db, user_id=user_id, persona_id=persona_id)
    requested_key = str(idempotency_key or "").strip() or None
    row: dict[str, Any] | None = None

    if requested_key:
        row = _find_session_by_idempotency_key(
            db,
            user_id=user_id,
            persona_id=resolved_persona_id,
            surface=surface,
            idempotency_key=requested_key,
        )

    if row is None and reuse_policy == "resume_compatible":
        row = _find_resume_compatible_session(
            db,
            user_id=user_id,
            persona_id=resolved_persona_id,
            surface=surface,
        )

    if row is None:
        materialized = materialize_persona_session(
            db,
            session_manager=session_manager,
            user_id=user_id,
            persona_id=resolved_persona_id,
            surface=surface,
        )
        row = materialized.session_row
    else:
        materialized = materialize_persona_session(
            db,
            session_manager=session_manager,
            user_id=user_id,
            persona_id=resolved_persona_id,
            resume_session_id=str(row.get("id") or ""),
            surface=surface,
        )
        row = materialized.session_row

    if requested_key:
        _update_live_control_preferences(
            db,
            row=row,
            user_id=user_id,
            idempotency_key=requested_key,
        )
        row = db.get_persona_session(str(row.get("id") or ""), user_id=user_id, include_deleted=False) or row

    return focus_live_session(
        db,
        user_id=user_id,
        session_id=str(row.get("id") or ""),
        stream_registry=stream_registry,
    )


def focus_live_session(
    db: CharactersRAGDB,
    *,
    user_id: str,
    session_id: str,
    stream_registry: PersonaLiveStreamRegistry = persona_live_stream_registry,
) -> dict[str, Any]:
    """Mark one non-terminal Persona Live session as focused and clear prior focused rows."""
    with _live_session_mutation_lock:
        return _focus_live_session_unlocked(
            db,
            user_id=user_id,
            session_id=session_id,
            stream_registry=stream_registry,
        )


def _focus_live_session_unlocked(
    db: CharactersRAGDB,
    *,
    user_id: str,
    session_id: str,
    stream_registry: PersonaLiveStreamRegistry,
) -> dict[str, Any]:
    """Focus a live session while the caller holds the mutation lock."""
    row = _load_owned_session_or_raise(db, user_id=user_id, session_id=session_id)
    if _is_terminal(row):
        raise ValueError("Cannot focus a terminal persona session.")
    generation = _next_focus_generation()
    focused_at = _utc_now_iso()
    previously_focused_rows = _focused_session_rows(db, user_id=user_id)
    fresh_target_row = db.get_persona_session(session_id, user_id=user_id, include_deleted=False) or row
    _update_live_control_preferences(
        db,
        row=fresh_target_row,
        user_id=user_id,
        focus={"focused": True, "focused_at": focused_at, "focus_generation": generation},
    )
    # Reconcile the pre-focus snapshot first, then any rows that became focused
    # during concurrent requests so only the target session remains focused.
    _reconcile_focused_sessions(
        db,
        user_id=user_id,
        target_session_id=session_id,
        initially_focused_rows=previously_focused_rows,
    )
    row = db.get_persona_session(session_id, user_id=user_id, include_deleted=False) or row
    return build_live_session_summary(
        db,
        user_id=user_id,
        row=row,
        is_focused=True,
        stream_registry=stream_registry,
    )


def stop_live_session(
    db: CharactersRAGDB,
    *,
    user_id: str,
    session_id: str,
    stream_registry: PersonaLiveStreamRegistry = persona_live_stream_registry,
) -> dict[str, Any]:
    """Stop a Persona Live session and remove live-control focus/idempotency metadata."""
    row = _load_owned_session_or_raise(db, user_id=user_id, session_id=session_id)
    if _is_terminal(row):
        persona_live_turn_registry.cancel(user_id=user_id, session_id=session_id)
        persona_live_voice_registry.clear(user_id=user_id, session_id=session_id)
        stream_registry.stop(user_id=user_id, session_id=session_id)
        return build_live_session_summary(
            db,
            user_id=user_id,
            row=row,
            is_focused=False,
            stream_registry=stream_registry,
        )
    fresh_row = db.get_persona_session(session_id, user_id=user_id, include_deleted=False) or row
    preferences = _live_preferences_patch(fresh_row, focus={"focused": False}, idempotency_key=None)
    db.update_persona_session(
        session_id=session_id,
        user_id=user_id,
        update_data={"status": "closed", "preferences_json": preferences},
    )
    persona_live_turn_registry.cancel(user_id=user_id, session_id=session_id)
    persona_live_voice_registry.clear(user_id=user_id, session_id=session_id)
    stream_registry.stop(user_id=user_id, session_id=session_id)
    row = db.get_persona_session(session_id, user_id=user_id, include_deleted=False) or row
    return build_live_session_summary(
        db,
        user_id=user_id,
        row=row,
        is_focused=False,
        stream_registry=stream_registry,
    )
