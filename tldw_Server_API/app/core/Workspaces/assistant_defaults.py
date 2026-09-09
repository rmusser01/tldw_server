"""Resolve Workspace Persona defaults exactly once at chat creation."""

from __future__ import annotations

from fastapi import HTTPException
from pydantic import ValidationError

from tldw_Server_API.app.api.v1.schemas.chat_session_schemas import ChatSessionCreate
from tldw_Server_API.app.api.v1.schemas.workspace_schemas import WorkspaceAssistantDefaults
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


def resolve_new_conversation_assistant(
    db: CharactersRAGDB, *, user_id: str, request: ChatSessionCreate
) -> ChatSessionCreate:
    """Apply an active owned default only when the caller omitted identity."""
    if request.scope_type != "workspace":
        return request
    workspace = db.get_workspace(request.workspace_id)
    if workspace is None or workspace.get("deleted"):
        raise HTTPException(status_code=404, detail="Workspace not found")
    if request.parent_conversation_id or request.model_fields_set & {"assistant_kind", "assistant_id", "character_id"}:
        return request
    stored = workspace.get("assistant_defaults_json")
    if stored is None:
        return request
    try:
        default = WorkspaceAssistantDefaults.model_validate(stored)
    except ValidationError as exc:
        raise HTTPException(
            status_code=409, detail="Workspace default Persona is unavailable; choose an assistant explicitly"
        ) from exc
    profile = db.get_persona_profile(default.assistant_id, user_id=user_id)
    if profile is None or profile.get("deleted") or not profile.get("is_active", True):
        raise HTTPException(
            status_code=409, detail="Workspace default Persona is unavailable; choose an assistant explicitly"
        )
    return request.model_copy(
        update={
            "assistant_kind": "persona",
            "assistant_id": default.assistant_id,
            "persona_memory_mode": default.persona_memory_mode,
        }
    )
