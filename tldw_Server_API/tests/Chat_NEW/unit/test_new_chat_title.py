"""XP-14: a chat the server creates for a saved completion is named after the user's first message."""

import asyncio

import pytest

from tldw_Server_API.app.api.v1.endpoints.chat import _save_message_turn_to_db
from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import ChatCompletionRequest
from tldw_Server_API.app.core.Chat.chat_metrics import get_chat_metrics
from tldw_Server_API.app.core.Chat.chat_service import build_context_and_messages


@pytest.mark.asyncio
async def test_saved_completion_titles_its_new_chat_from_the_first_user_message(populated_chacha_db):
    request = ChatCompletionRequest(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "How should I plan my vegetable garden this spring?"}],
        save_to_db=True,
    )

    _, _, conversation_id, _, _, _ = await build_context_and_messages(
        chat_db=populated_chacha_db,
        request_data=request,
        loop=asyncio.get_running_loop(),
        metrics=get_chat_metrics(),
        default_save_to_db=False,
        final_conversation_id=None,
        save_message_fn=_save_message_turn_to_db,
    )

    conversation = populated_chacha_db.get_conversation_by_id(conversation_id)
    assert conversation["title"] == "How should I plan my vegetable…"
