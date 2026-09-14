"""Shared Workspace default projection and one-time assistant startup selection."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException
from loguru import logger
from pydantic import ValidationError

from tldw_Server_API.app.api.v1.schemas.chat_session_schemas import ChatSessionCreate
from tldw_Server_API.app.api.v1.schemas.workspace_schemas import (
    WorkspaceAssistantDefaults,
    WorkspaceEffectiveAssistantDefault,
)
from tldw_Server_API.app.core import feature_flags
from tldw_Server_API.app.core.Chat.assistant_startup import AssistantStartup
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, InputError

WorkspacePersonaProfileCache = dict[tuple[str, str, bool], dict[str, Any] | None]


def parse_workspace_assistant_defaults(
    raw: Any, *, workspace_id: str | None = None,
) -> tuple[WorkspaceAssistantDefaults | None, bool]:
    """Parse normalized storage without logging private payloads or validation text."""
    if raw is None:
        return None, False
    try:
        return WorkspaceAssistantDefaults.model_validate(raw), False
    except ValidationError:
        logger.warning(
            "Ignoring invalid stored workspace assistant defaults "
            "category=schema_validation; workspace_id_present={workspace_id_present}; "
            "payload_type={payload_type}",
            workspace_id_present=bool(workspace_id),
            payload_type=type(raw).__name__ if type(raw) in (dict, list, str, int, float, bool) else "other",
        )
        return None, True


def get_workspace_persona_profile(
    *, db: CharactersRAGDB, assistant_id: str, user_id: str, include_deleted: bool,
    cache: WorkspacePersonaProfileCache | None = None, conn: Any = None,
) -> dict[str, Any] | None:
    """Read owned profiles; a locking read must never reuse an unlocked cache value."""
    if conn is not None:
        cache = None
    cache_key = (user_id, assistant_id, include_deleted)
    if cache is not None and cache_key in cache:
        return cache[cache_key]
    connection_kwargs = {"conn": conn, "for_update": True} if conn is not None else {}
    profile = db.get_persona_profile(
        assistant_id, user_id=user_id, include_deleted=include_deleted, **connection_kwargs,
    )
    if cache is not None:
        cache[cache_key] = profile
    return profile


def resolve_effective_workspace_assistant_default(
    db: CharactersRAGDB, *, workspace: Mapping[str, Any], user_id: str,
    persona_profile_cache: WorkspacePersonaProfileCache | None = None, conn: Any = None,
) -> WorkspaceEffectiveAssistantDefault:
    """Return permission-safe default state while letting DB errors reach HTTP mappers."""
    raw = workspace.get("assistant_defaults_json")
    invalid = bool(workspace.get("_assistant_defaults_invalid")) or (
        raw is not None and bool(workspace.get("assistant_defaults_explicit_none"))
    )
    stored = None
    if not invalid:
        stored, invalid = parse_workspace_assistant_defaults(raw, workspace_id=workspace.get("id"))
    if invalid:
        return WorkspaceEffectiveAssistantDefault(
            status="unavailable", source="workspace", degraded_reason="invalid_default",
        )
    if stored is None:
        return WorkspaceEffectiveAssistantDefault(status="none", source="none")
    if stored.assistant_kind != "persona":
        return WorkspaceEffectiveAssistantDefault(
            status="unavailable", source="workspace", degraded_reason="unsupported_assistant_kind",
        )
    if not feature_flags.is_persona_enabled():
        return WorkspaceEffectiveAssistantDefault(
            status="unavailable", source="workspace", degraded_reason="persona_feature_disabled",
        )
    profile = get_workspace_persona_profile(
        db=db, assistant_id=stored.assistant_id, user_id=user_id, include_deleted=False,
        cache=persona_profile_cache, conn=conn,
    )
    reference = {
        "assistant_kind": "persona", "assistant_id": stored.assistant_id,
        "persona_memory_mode": stored.persona_memory_mode,
    }
    if profile is not None:
        if profile.get("deleted") or not bool(profile.get("is_active", True)):
            return WorkspaceEffectiveAssistantDefault(
                status="unavailable", source="workspace", **reference,
                degraded_reason="persona_deleted" if profile.get("deleted") else "persona_unavailable",
            )
        return WorkspaceEffectiveAssistantDefault(
            status="available", source="workspace", **reference,
            label=str(profile.get("name") or stored.assistant_id),
        )
    deleted_profile = get_workspace_persona_profile(
        db=db, assistant_id=stored.assistant_id, user_id=user_id, include_deleted=True,
        cache=persona_profile_cache, conn=conn,
    )
    if deleted_profile is None:
        return WorkspaceEffectiveAssistantDefault(
            status="unavailable", source="workspace", degraded_reason="permission_denied",
        )
    return WorkspaceEffectiveAssistantDefault(
        status="unavailable", source="workspace", **reference, degraded_reason="persona_deleted",
    )


@dataclass(frozen=True)
class ResolvedConversationAssistant:
    """Selected request and local origin, with an ephemeral title label."""

    request: ChatSessionCreate
    startup: AssistantStartup
    display_name: str


def resolve_workspace_assistant_startup(
    db: CharactersRAGDB, *, user_id: str, request: ChatSessionCreate, conn: Any = None,
) -> ResolvedConversationAssistant:
    """Select once from original omission/null intent and current Workspace state.

    The caller must validate scope, ownership and parent/message lineage before
    persisting this result. A parent string is not itself authorization for a fork.
    Passing an existing transaction locks Workspace then selected Persona reads;
    legacy no-connection callers retain unlocked DB signatures.
    """
    if request.scope_type != "workspace":
        return ResolvedConversationAssistant(request, AssistantStartup(), "Assistant")
    connection_kwargs = {"conn": conn, "for_update": True} if conn is not None else {}
    workspace = db.get_workspace(request.workspace_id, **connection_kwargs)
    if workspace is None or workspace.get("deleted"):
        raise HTTPException(status_code=404, detail="Workspace not found")
    if request.parent_conversation_id or request.model_fields_set & {"assistant_kind", "assistant_id", "character_id"}:
        source = "fork" if request.parent_conversation_id else (
            "explicit" if request.assistant_kind is not None else "explicit_none"
        )
        display_name = "Assistant"
        # Transaction-time explicit Persona titles use the selected profile,
        # without consulting defaults or changing legacy preflight validation.
        if conn is not None and request.assistant_kind == "persona":
            profile = get_workspace_persona_profile(
                db=db, assistant_id=request.assistant_id, user_id=user_id,
                include_deleted=False, conn=conn,
            )
            if profile is None or profile.get("deleted"):
                raise HTTPException(status_code=404, detail="Persona not found")
            display_name = str(profile.get("name") or display_name)
        return ResolvedConversationAssistant(request, AssistantStartup(source=source), display_name)
    effective = resolve_effective_workspace_assistant_default(db, workspace=workspace, user_id=user_id, conn=conn)
    if effective.status == "unavailable":
        raise HTTPException(
            status_code=503 if effective.degraded_reason == "persona_feature_disabled" else 409,
            detail="Workspace default Persona is unavailable; choose an assistant explicitly",
        )
    try:
        startup = AssistantStartup(
            source="workspace_default" if effective.status == "available" else "system_fallback",
            workspace_id=workspace["id"], workspace_version=workspace["version"],
        )
    except ValidationError:
        raise InputError("Workspace assistant startup cannot be represented safely") from None
    if effective.status == "none":
        return ResolvedConversationAssistant(request, startup, "Assistant")
    resolved_request = request.model_copy(update={
        "assistant_kind": effective.assistant_kind,
        "assistant_id": effective.assistant_id,
        "persona_memory_mode": effective.persona_memory_mode,
    })
    return ResolvedConversationAssistant(resolved_request, startup, effective.label or "Assistant")


def resolve_new_conversation_assistant(
    db: CharactersRAGDB, *, user_id: str, request: ChatSessionCreate,
) -> ChatSessionCreate:
    """Compatibility preflight returning only the selected request, not trusted origin."""
    return resolve_workspace_assistant_startup(db, user_id=user_id, request=request).request


def create_workspace_persona_conversation(
    db: CharactersRAGDB, *, user_id: str, request: ChatSessionCreate,
    conversation_data: Mapping[str, Any], title_timestamp: str,
) -> str:
    """Atomically select and persist non-Character Workspace identity and origin.

    Scope, quota and parent/message admission belong to the caller. The internal
    payload must retain that validated authority; only its preflight identity and
    derived title are replaced. Workspace then Persona locks cover the INSERT.
    """
    if request.scope_type != "workspace" or request.assistant_kind == "character":
        raise InputError("Workspace Persona creation requires a non-Character Workspace request")
    if user_id != str(db.client_id) or conversation_data.get("client_id") != user_id:
        raise InputError("Conversation owner must match the scoped database owner")
    for field in ("scope_type", "workspace_id", "parent_conversation_id", "forked_from_message_id"):
        if conversation_data.get(field) != (getattr(request, field) or None):
            raise InputError("Conversation scope and lineage must match the validated request")
    parent = db.get_conversation_by_id(request.parent_conversation_id) if request.parent_conversation_id else None
    if request.parent_conversation_id and (
        parent is None or str(parent.get("client_id", "")).strip() != user_id.strip()
        or parent.get("scope_type") != "workspace" or parent.get("workspace_id") != request.workspace_id
    ):
        raise InputError("Conversation parent must match the validated scope and owner")
    root_id = (parent.get("root_id") or parent["id"]) if parent else conversation_data.get("id")
    if conversation_data.get("root_id") != root_id:
        raise InputError("Conversation root must match the validated lineage")
    with db.transaction() as conn:
        resolved = resolve_workspace_assistant_startup(db, user_id=user_id, request=request, conn=conn)
        payload = dict(conversation_data)
        payload.update(
            assistant_kind=resolved.request.assistant_kind,
            assistant_id=resolved.request.assistant_id,
            character_id=resolved.request.character_id,
            persona_memory_mode=resolved.request.persona_memory_mode,
        )
        payload["title"] = request.title or (
            f"{resolved.display_name} Chat ({title_timestamp})"
            if resolved.request.assistant_kind in {"persona", "character"}
            else f"Chat ({title_timestamp})"
        )
        return db.add_conversation(payload, conn=conn, assistant_startup=resolved.startup)
