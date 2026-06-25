"""Persistence helpers for chat completion responses."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from tldw_Server_API.app.core.Character_Chat.modules.character_utils import sanitize_sender_name


SaveMessageFn = Callable[..., Awaitable[str | None]]


def build_assistant_message_payload(
    *,
    character_card_for_context: dict[str, Any] | None,
    assistant_parent_message_id: str | None,
    content: Any | None,
    tool_calls: Any | None,
    function_call: Any | None,
) -> dict[str, Any]:
    """Build the persisted assistant message payload for a chat completion."""

    asst_name = sanitize_sender_name(
        character_card_for_context.get("name") if character_card_for_context else None
    )
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
