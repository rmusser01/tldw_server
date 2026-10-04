"""Regression coverage for fallback assistants preserving neutral conversations."""

import asyncio

import pytest

from tldw_Server_API.app.api.v1.endpoints.chat import _save_message_turn_to_db
from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import ChatCompletionRequest
from tldw_Server_API.app.core.Chat.chat_metrics import get_chat_metrics
from tldw_Server_API.app.core.Chat.chat_service import build_context_and_messages
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


@pytest.mark.asyncio
@pytest.mark.parametrize("scope_type", ["global", "workspace"])
async def test_fallback_assistant_preserves_neutral_conversation_and_history(
    populated_chacha_db: CharactersRAGDB, scope_type: str
) -> None:
    """Two persisted turns must stay on the caller's neutral conversation."""
    db = populated_chacha_db
    if scope_type == "workspace":
        db.upsert_workspace("workspace-continuity", "Continuity workspace")
    conversation_id = db.add_conversation(
        {
            "title": "Neutral chat",
            "client_id": db.client_id,
            "scope_type": scope_type,
            "workspace_id": "workspace-continuity" if scope_type == "workspace" else None,
        }
    )
    before = db.get_conversation_by_id(conversation_id)

    for content in ["Remember MARBLE-5831.", "What code did I give you?"]:
        request = ChatCompletionRequest(
            model="local-model",
            conversation_id=conversation_id,
            save_to_db=True,
            messages=[{"role": "user", "content": content}],
        )
        _, _, resolved_id, created, _, persisted = await build_context_and_messages(
            chat_db=db,
            request_data=request,
            loop=asyncio.get_running_loop(),
            metrics=get_chat_metrics(),
            default_save_to_db=False,
            final_conversation_id=conversation_id,
            save_message_fn=_save_message_turn_to_db,
        )
        assert (resolved_id, created, persisted) == (conversation_id, False, True)

    stored = db.get_messages_for_conversation(conversation_id, limit=20, order_by_timestamp="ASC")
    assert [message["content"] for message in stored] == ["Remember MARBLE-5831.", "What code did I give you?"]
    after = db.get_conversation_by_id(conversation_id)
    for key in ("character_id", "assistant_kind", "assistant_id", "scope_type", "workspace_id", "client_id"):
        assert after[key] == before[key]


@pytest.mark.asyncio
@pytest.mark.parametrize("mismatch", ["explicit-character", "foreign-owner"])
async def test_neutral_conversation_reuse_keeps_identity_and_owner_validation(
    populated_chacha_db: CharactersRAGDB, mismatch: str
) -> None:
    """Fallback reuse must not bypass explicit character or owner mismatches."""
    db = populated_chacha_db
    character_id = db.add_character_card(
        {
            "name": "Explicit character",
            "system_prompt": "You are the explicitly selected character.",
            "client_id": db.client_id,
        }
    )
    conversation_id = db.add_conversation(
        {
            "title": "Do not reuse",
            "client_id": "other-owner" if mismatch == "foreign-owner" else db.client_id,
        }
    )
    request = ChatCompletionRequest(
        model="local-model",
        conversation_id=conversation_id,
        character_id=str(character_id) if mismatch == "explicit-character" else None,
        save_to_db=True,
        messages=[{"role": "user", "content": "Start the intended chat."}],
    )
    _, _, resolved_id, created, _, _ = await build_context_and_messages(
        chat_db=db,
        request_data=request,
        loop=asyncio.get_running_loop(),
        metrics=get_chat_metrics(),
        default_save_to_db=False,
        final_conversation_id=conversation_id,
        save_message_fn=_save_message_turn_to_db,
    )

    assert resolved_id != conversation_id
    assert created
    assert db.get_messages_for_conversation(conversation_id) == []
    assert db.get_conversation_by_id(resolved_id)["client_id"] == db.client_id
