"""Shared Persona session materialization helpers for REST and live control."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
import uuid

from loguru import logger

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    ConflictError,
)
from tldw_Server_API.app.core.Persona.session_manager import SessionManager
from tldw_Server_API.app.core.Personalization.companion_activity import normalize_persona_activity_surface


DEFAULT_PERSONA_ID = "research_assistant"
DEFAULT_PERSONA_NAME = "Research Assistant"
DEFAULT_PERSONA_DESCRIPTION = "Helps ingest, search, and summarize content"
DEFAULT_PERSONA_POLICY_RULES: list[dict[str, Any]] = [
    {"rule_kind": "mcp_tool", "rule_name": "media.search", "allowed": True, "require_confirmation": False},
    {"rule_kind": "mcp_tool", "rule_name": "chats.search", "allowed": True, "require_confirmation": False},
    {"rule_kind": "mcp_tool", "rule_name": "notes.search", "allowed": True, "require_confirmation": False},
    {"rule_kind": "mcp_tool", "rule_name": "notes.create", "allowed": True, "require_confirmation": True},
]
EXPLICIT_SCOPE_RULE_TYPES = {"conversation_id", "character_id", "media_id", "note_id"}
PERSONA_RUNTIME_MODES = {"session_scoped", "persistent_scoped"}


@dataclass(frozen=True)
class MaterializedPersonaSession:
    session_id: str
    persona_id: str
    profile: dict[str, Any]
    session_row: dict[str, Any]
    created_new_session: bool
    scope_audit: dict[str, object]
    activity_surface: str


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _coerce_bool(value: Any, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on", "enabled"}:
            return True
        if normalized in {"false", "0", "no", "off", "disabled"}:
            return False
    return default


def _get_persona_memory_top_k() -> int:
    try:
        from tldw_Server_API.app.core.config import settings as app_settings

        value = int(app_settings.get("PERSONA_MEMORY_TOP_K", 3))
    except Exception:
        value = 3
    return max(1, min(value, 10))


def scope_audit_from_snapshot(scope_snapshot: Any) -> dict[str, object]:
    if not isinstance(scope_snapshot, dict):
        return {}
    audit = scope_snapshot.get("audit")
    if isinstance(audit, dict):
        return {str(k): v for k, v in audit.items()}
    return {}


def scope_snapshot_id_from_snapshot(scope_snapshot: Any) -> str | None:
    if not isinstance(scope_snapshot, dict):
        return None
    candidate = str(scope_snapshot.get("scope_snapshot_id") or "").strip()
    if candidate:
        return candidate
    audit = scope_audit_from_snapshot(scope_snapshot)
    fallback = str(audit.get("scope_snapshot_id") or "").strip()
    return fallback or None


def ensure_default_persona_profile(db: CharactersRAGDB, *, user_id: str) -> dict[str, Any]:
    profile = db.get_persona_profile(DEFAULT_PERSONA_ID, user_id=user_id, include_deleted=False)
    if profile is None:
        try:
            _ = db.create_persona_profile(
                {
                    "id": DEFAULT_PERSONA_ID,
                    "user_id": user_id,
                    "name": DEFAULT_PERSONA_NAME,
                    "mode": "session_scoped",
                    "system_prompt": DEFAULT_PERSONA_DESCRIPTION,
                    "is_active": True,
                }
            )
        except ConflictError:
            pass
        profile = db.get_persona_profile(DEFAULT_PERSONA_ID, user_id=user_id, include_deleted=False)
    if profile is None:
        profiles = db.list_persona_profiles(user_id=user_id, active_only=True, limit=1)
        if not profiles:
            raise ConflictError(
                "Unable to resolve a default persona profile for user.",
                entity="persona_profiles",
                entity_id=DEFAULT_PERSONA_ID,
            )
        profile = profiles[0]

    if str(profile.get("id") or "") == DEFAULT_PERSONA_ID:
        try:
            existing = db.list_persona_policy_rules(persona_id=DEFAULT_PERSONA_ID, user_id=user_id)
            if not existing:
                _ = db.replace_persona_policy_rules(
                    persona_id=DEFAULT_PERSONA_ID,
                    user_id=user_id,
                    rules=DEFAULT_PERSONA_POLICY_RULES,
                )
        except CharactersRAGDBError as exc:
            logger.warning("Failed to ensure default persona policy rules: {}", exc)
    return profile


def build_scope_snapshot(rules: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, object]]:
    scope_snapshot_id = uuid.uuid4().hex
    materialized_at = _utc_now_iso()
    include_counts: dict[str, int] = {}
    explicit_ids: dict[str, list[str]] = {}
    selector_values: dict[str, list[str]] = {}
    include_rule_count = 0
    exclude_rule_count = 0
    explicit_id_rule_count = 0
    selector_rule_count = 0

    for rule in rules:
        if not isinstance(rule, dict):
            continue
        rule_type = str(rule.get("rule_type") or "").strip().lower()
        rule_value = str(rule.get("rule_value") or "").strip()
        include = bool(rule.get("include", True))
        if not rule_type:
            continue
        include_counts[rule_type] = include_counts.get(rule_type, 0) + 1
        if not include:
            exclude_rule_count += 1
            continue
        include_rule_count += 1
        if rule_type in EXPLICIT_SCOPE_RULE_TYPES:
            if rule_value:
                explicit_ids.setdefault(rule_type, []).append(rule_value)
                explicit_id_rule_count += 1
        elif rule_value:
            selector_values.setdefault(rule_type, []).append(rule_value)
            selector_rule_count += 1

    for values in explicit_ids.values():
        values[:] = sorted(set(values))
    for values in selector_values.values():
        values[:] = sorted(set(values))

    audit: dict[str, object] = {
        "scope_snapshot_id": scope_snapshot_id,
        "materialized_at": materialized_at,
        "source_rule_count": len(rules),
        "include_rule_count": include_rule_count,
        "exclude_rule_count": exclude_rule_count,
        "source_rule_type_counts": include_counts,
        "explicit_id_rule_count": explicit_id_rule_count,
        "selector_rule_count": selector_rule_count,
        "selector_rule_types": sorted(selector_values.keys()),
    }
    snapshot = {
        "scope_snapshot_id": scope_snapshot_id,
        "materialized_at": materialized_at,
        "materialized_scope": {
            "explicit_ids": explicit_ids,
            "selectors": selector_values,
        },
        "audit": audit,
    }
    return snapshot, audit


def normalize_persisted_persona_session_preferences(preferences: Any) -> dict[str, Any]:
    if not isinstance(preferences, dict):
        return {}
    normalized: dict[str, Any] = {}
    if "use_memory_context" in preferences:
        normalized["use_memory_context"] = _coerce_bool(preferences.get("use_memory_context"), default=True)
    if "use_companion_context" in preferences:
        normalized["use_companion_context"] = _coerce_bool(preferences.get("use_companion_context"), default=True)
    if "use_persona_state_context" in preferences:
        normalized["use_persona_state_context"] = _coerce_bool(
            preferences.get("use_persona_state_context"),
            default=True,
        )
    if "memory_top_k" in preferences:
        try:
            normalized_top_k = int(preferences.get("memory_top_k"))
        except (TypeError, ValueError):
            normalized_top_k = _get_persona_memory_top_k()
        normalized["memory_top_k"] = max(1, normalized_top_k)
    if "session_policy_rules" in preferences:
        normalized["session_policy_rules"] = preferences.get("session_policy_rules")
    if "companion_activity_surface" in preferences:
        normalized["companion_activity_surface"] = normalize_persona_activity_surface(
            preferences.get("companion_activity_surface")
        )
    return normalized


def merge_persisted_persona_session_preferences(*payloads: Any) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for payload in payloads:
        merged.update(normalize_persisted_persona_session_preferences(payload))
    return merged


def default_persisted_persona_session_preferences(
    profile: dict[str, Any] | None,
    *,
    activity_surface: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "use_memory_context": True,
        "use_companion_context": True,
        "use_persona_state_context": _coerce_bool(
            (profile or {}).get("use_persona_state_context_default"),
            default=True,
        ),
        "memory_top_k": _get_persona_memory_top_k(),
    }
    if activity_surface is not None:
        payload["companion_activity_surface"] = normalize_persona_activity_surface(activity_surface)
    return merge_persisted_persona_session_preferences(payload)


def get_session_preferences_with_activity_surface(
    *,
    session_manager: Any,
    session_id: str,
    user_id: str,
    persisted_preferences: Any = None,
    persisted_activity_surface: Any = None,
) -> tuple[dict[str, Any], str]:
    runtime_preferences = dict(
        session_manager.get_preferences(
            session_id=session_id,
            user_id=user_id,
        )
    )
    merged_preferences = dict(runtime_preferences)
    merged_preferences.update(normalize_persisted_persona_session_preferences(persisted_preferences))
    activity_surface = normalize_persona_activity_surface(
        runtime_preferences.get("companion_activity_surface", persisted_activity_surface)
    )
    merged_preferences["companion_activity_surface"] = activity_surface
    if merged_preferences == runtime_preferences:
        return merged_preferences, activity_surface

    updated_preferences = merged_preferences
    try:
        updated_preferences = session_manager.update_preferences(
            session_id=session_id,
            user_id=user_id,
            preferences=merged_preferences,
        )
    except Exception as exc:
        logger.debug(
            "Persona session preference sync skipped for session {}: {}",
            session_id,
            exc,
        )
    return dict(updated_preferences), activity_surface


def _persist_activity_surface_preference(
    db: CharactersRAGDB,
    *,
    session_row: dict[str, Any],
    user_id: str,
    activity_surface: str,
) -> dict[str, Any]:
    session_id = str(session_row.get("id") or "").strip()
    if not session_id:
        return session_row
    preferences = dict(session_row.get("preferences") or {})
    preferences["companion_activity_surface"] = activity_surface
    if preferences == session_row.get("preferences"):
        return session_row
    updated = db.update_persona_session(
        session_id=session_id,
        user_id=user_id,
        update_data={"preferences_json": preferences},
    )
    if not updated:
        return session_row
    return db.get_persona_session(session_id, user_id=user_id, include_deleted=False) or session_row


def materialize_persona_session(
    db: CharactersRAGDB,
    *,
    session_manager: SessionManager,
    user_id: str,
    persona_id: str | None,
    resume_session_id: str | None = None,
    project_id: str | None = None,
    surface: str | None = None,
) -> MaterializedPersonaSession:
    requested_persona_id = str(persona_id or "").strip() or DEFAULT_PERSONA_ID
    requested_activity_surface = normalize_persona_activity_surface(surface)
    profile = db.get_persona_profile(requested_persona_id, user_id=user_id, include_deleted=False)
    if profile is None:
        logger.info(
            "Unknown persona_id requested in API: {}; defaulting to {}",
            requested_persona_id,
            DEFAULT_PERSONA_ID,
        )
        profile = ensure_default_persona_profile(db, user_id=user_id)
    resolved_persona_id = str(profile.get("id") or DEFAULT_PERSONA_ID)

    if resume_session_id:
        local_session = session_manager.get(resume_session_id)
        if local_session is not None:
            if str(local_session.user_id) != user_id:
                raise PermissionError("session ownership mismatch")
            if str(local_session.persona_id) != resolved_persona_id:
                raise ConflictError(
                    "resume_session_id is bound to a different persona_id.",
                    entity="persona_sessions",
                    entity_id=str(resume_session_id),
                )

    session_row: dict[str, Any] | None = None
    if resume_session_id:
        session_row = db.get_persona_session(resume_session_id, user_id=user_id, include_deleted=False)
        if session_row is not None:
            bound_persona_id = str(session_row.get("persona_id") or "").strip()
            if bound_persona_id and bound_persona_id != resolved_persona_id:
                raise ConflictError(
                    "resume_session_id is bound to a different persona_id.",
                    entity="persona_sessions",
                    entity_id=str(resume_session_id),
                )

    created_new_session = session_row is None
    if session_row is None:
        scope_rules = db.list_persona_scope_rules(
            persona_id=resolved_persona_id,
            user_id=user_id,
            include_deleted=False,
        )
        scope_snapshot, scope_audit = build_scope_snapshot(scope_rules)
        create_data: dict[str, Any] = {
            "persona_id": resolved_persona_id,
            "user_id": user_id,
            "conversation_id": project_id,
            "mode": str(profile.get("mode") or "session_scoped"),
            "scope_snapshot_json": scope_snapshot,
            "preferences_json": default_persisted_persona_session_preferences(
                profile,
                activity_surface=requested_activity_surface,
            ),
            "activity_surface": requested_activity_surface,
        }
        if resume_session_id:
            create_data["id"] = str(resume_session_id)
        session_id = db.create_persona_session(create_data)
        session_row = db.get_persona_session(session_id, user_id=user_id, include_deleted=False)
        if session_row is None:
            raise RuntimeError("Failed to load created persona session")
    else:
        scope_audit = scope_audit_from_snapshot(session_row.get("scope_snapshot") or {})
        if surface is not None:
            current_surface = normalize_persona_activity_surface(session_row.get("activity_surface"))
            if current_surface != requested_activity_surface:
                _ = db.update_persona_session(
                    session_id=str(session_row.get("id") or resume_session_id or ""),
                    user_id=user_id,
                    update_data={"activity_surface": requested_activity_surface},
                )
                refreshed_row = db.get_persona_session(
                    str(session_row.get("id") or resume_session_id or ""),
                    user_id=user_id,
                    include_deleted=False,
                )
                if refreshed_row is not None:
                    session_row = refreshed_row

    session_id = str(session_row.get("id") or resume_session_id or "").strip()
    if not session_id:
        raise RuntimeError("Persona session missing session_id")

    try:
        _ = session_manager.create(
            user_id=user_id,
            persona_id=resolved_persona_id,
            resume_session_id=session_id,
        )
        if created_new_session or surface is not None:
            session_manager.update_preferences(
                session_id=session_id,
                user_id=user_id,
                preferences={"companion_activity_surface": requested_activity_surface},
            )
    except ValueError as exc:
        raise PermissionError(str(exc)) from exc

    _session_preferences, activity_surface = get_session_preferences_with_activity_surface(
        session_manager=session_manager,
        session_id=session_id,
        user_id=user_id,
        persisted_preferences=session_row.get("preferences"),
        persisted_activity_surface=session_row.get("activity_surface"),
    )
    session_row = _persist_activity_surface_preference(
        db,
        session_row=session_row,
        user_id=user_id,
        activity_surface=activity_surface,
    )

    return MaterializedPersonaSession(
        session_id=session_id,
        persona_id=resolved_persona_id,
        profile=dict(profile),
        session_row=session_row,
        created_new_session=created_new_session,
        scope_audit=scope_audit_from_snapshot(session_row.get("scope_snapshot") or {}),
        activity_surface=activity_surface,
    )
