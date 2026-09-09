"""Persistence helpers for chat completion responses."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import HTTPException

from tldw_Server_API.app.core.Buddy.publication import current_buddy_publication
from tldw_Server_API.app.core.Character_Chat.chat_settings_validation import validate_chat_settings_storage
from tldw_Server_API.app.core.Character_Chat.modules.character_utils import sanitize_sender_name
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, NotFoundError

SaveMessageFn = Callable[..., Awaitable[str | None]]


def save_workspace_chat_model_selection(
    *,
    chat_db: CharactersRAGDB,
    conversation_id: str,
    owner_client_id: str,
    provider: str,
    model: str,
) -> None:
    """Retain an explicit plain Chat selection for the same workspace conversation."""
    # A Buddy override belongs to that reply, not the conversation's Chat defaults.
    if current_buddy_publication.get() is not None:
        return
    conversation = chat_db.get_conversation_by_id(conversation_id)
    if not conversation or conversation.get("scope_type") != "workspace":
        return
    try:
        with chat_db.transaction() as conn:
            state = chat_db.get_roleplay_resume_state(
                conversation_id,
                conn=conn,
                lock_for_update=True,
                owner_client_id=owner_client_id,
            )
            conversation = state["conversation"]
            if (
                conversation.get("scope_type") != "workspace"
                or conversation.get("character_id") is not None
                or conversation.get("assistant_kind") is not None
                or conversation.get("assistant_id") is not None
                or (state.get("behavior_snapshot") or {}).get("status", "missing") != "missing"
            ):
                return
            settings = dict(state.get("settings") or {})
            if settings.get("provider") == provider and settings.get("model") == model:
                return
            settings.update(provider=provider, model=model)
            settings = validate_chat_settings_storage(settings, allow_internal=True)
            if not chat_db.upsert_conversation_settings(
                conversation_id,
                settings,
                conn=conn,
                expected_settings_version=state["settings_version"] or 0,
            ):
                raise HTTPException(status_code=409, detail="Conversation changed while saving Chat model settings.")
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail="Conversation not found.") from exc


def build_assistant_message_payload(
    *,
    character_card_for_context: dict[str, Any] | None,
    assistant_parent_message_id: str | None,
    content: Any | None,
    tool_calls: Any | None,
    function_call: Any | None,
) -> dict[str, Any]:
    """Build the persisted assistant message payload for a chat completion."""

    asst_name = sanitize_sender_name(character_card_for_context.get("name") if character_card_for_context else None)
    message_payload: dict[str, Any] = {"role": "assistant", "name": asst_name}
    if assistant_parent_message_id:
        message_payload["parent_message_id"] = assistant_parent_message_id
    if content is not None:
        message_payload["content"] = content
    if tool_calls is not None:
        message_payload["tool_calls"] = tool_calls
    if function_call is not None:
        message_payload["function_call"] = function_call
    return message_payload


async def save_assistant_message(
    *,
    chat_db: Any,
    conversation_id: str,
    save_message_fn: SaveMessageFn,
    payload: dict[str, Any],
) -> str | None:
    """Persist one assistant message payload."""

    return await save_message_fn(chat_db, conversation_id, payload, use_transaction=True)


async def save_tool_messages(
    *,
    chat_db: Any,
    conversation_id: str,
    save_message_fn: SaveMessageFn,
    tool_messages: list[dict[str, Any]],
) -> None:
    """Persist tool result messages in order."""

    for tool_message in tool_messages:
        await save_message_fn(chat_db, conversation_id, tool_message, use_transaction=True)
