"""Point-in-time current Persona admission without generation or fallback effects."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from fastapi import HTTPException

from tldw_Server_API.app.core import feature_flags
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, InputError

PersonaAdmissionCode = Literal[
    "persona_binding_invalid", "persona_not_found", "persona_unavailable", "persona_feature_disabled"
]


class PersonaAdmissionError(HTTPException):
    """Typed, content-free failure usable by generation and strict replay adapters."""

    def __init__(self, code: PersonaAdmissionCode) -> None:
        """Expose a bounded code/reason without stored profile identifiers or text."""
        statuses = {
            "persona_binding_invalid": 409,
            "persona_not_found": 404,
            "persona_unavailable": 409,
            "persona_feature_disabled": 503,
        }
        self.code = code
        self.reason = "persona_unavailable" if code == "persona_unavailable" else None
        detail = {"code": code}
        if self.reason is not None:
            detail["reason"] = self.reason
        super().__init__(status_code=statuses[code], detail=detail)


def require_current_persona(
    db: CharactersRAGDB, *, owner_id: str, conversation: Mapping[str, Any], conn: Any = None
) -> dict[str, Any] | None:
    """Require the immutable owner's active Persona; only non-Persona bypasses lookup.

    An optional supplied transaction locks the profile for startup/replay. Ordinary
    generation callers do an unlocked point-in-time check, never a network-long lock.
    Storage failures propagate rather than becoming absence or plain Assistant.
    """
    if owner_id != db.owner_user_id:
        raise PersonaAdmissionError("persona_not_found")
    try:
        kind, identity, _, _ = db.conversation_store._normalize_conversation_assistant_identity(
            character_id=conversation.get("character_id"),
            assistant_kind=conversation.get("assistant_kind"),
            assistant_id=conversation.get("assistant_id"),
            persona_memory_mode=conversation.get("persona_memory_mode"),
        )
    except InputError:
        raise PersonaAdmissionError("persona_binding_invalid") from None
    if kind != "persona":
        return None
    supplied_identity = conversation.get("assistant_id")
    if not isinstance(supplied_identity, str) or not supplied_identity.strip():
        raise PersonaAdmissionError("persona_binding_invalid")
    if not feature_flags.is_persona_enabled():
        raise PersonaAdmissionError("persona_feature_disabled")
    connection_kwargs = {"conn": conn, "for_update": True} if conn is not None else {}
    profile = db.get_persona_profile(identity, user_id=owner_id, include_deleted=False, **connection_kwargs)
    if profile is None or profile.get("deleted") or profile.get("user_id") != owner_id:
        raise PersonaAdmissionError("persona_not_found")
    if not bool(profile.get("is_active", False)):
        raise PersonaAdmissionError("persona_unavailable")
    return profile
