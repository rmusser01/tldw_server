"""Primitive endpoint coverage for client-managed conversation context."""

from __future__ import annotations

import uuid

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints import character_chat_sessions
from tldw_Server_API.app.api.v1.endpoints import chat_dictionaries
from tldw_Server_API.app.api.v1.endpoints import characters_endpoint
from tldw_Server_API.app.api.v1.schemas.chat_dictionary_schemas import ProcessTextRequest
from tldw_Server_API.app.api.v1.schemas.chat_session_schemas import ChatSettingsUpdate
from tldw_Server_API.app.api.v1.schemas.world_book_schemas import ProcessContextRequest
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
from tldw_Server_API.app.core.Character_Chat.chat_dictionary import ChatDictionaryService
from tldw_Server_API.app.core.Character_Chat.world_book_manager import WorldBookService
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


def _test_user() -> User:
    return User(
        id="conversation-context-user",
        username="conversation-context-user",
        email=None,
        is_active=True,
    )


def _create_blank_chat(db: CharactersRAGDB, user: User) -> str:
    chat_id = f"conversation-context-{uuid.uuid4()}"
    db.add_conversation(
        {
            "id": chat_id,
            "character_id": None,
            "title": "Conversation Context Primitive Test",
            "root_id": chat_id,
            "parent_id": None,
            "active": 1,
            "deleted": 0,
            "client_id": str(user.id),
            "version": 1,
            "assistant_kind": "persona",
            "assistant_id": f"persona-{chat_id}",
            "persona_memory_mode": "read_only",
        }
    )
    return chat_id


@pytest.mark.integration
@pytest.mark.asyncio
async def test_worldbook_process_accepts_explicit_ids_without_character(
    character_db: CharactersRAGDB,
) -> None:
    service = WorldBookService(character_db)
    world_book_id = service.create_world_book(name="Conversation Context Lore")
    entry_id = service.add_entry(
        world_book_id,
        ["Echo Vault"],
        "Echo Vault sits below the old Lumen rail station.",
        priority=100,
    )

    response = await characters_endpoint.process_context_with_world_info(
        ProcessContextRequest(
            text="Mira opens the Echo Vault.",
            world_book_ids=[world_book_id],
            character_id=None,
            token_budget=500,
        ),
        db=character_db,
    )

    assert response.entries_matched == 1
    assert response.entry_ids == [entry_id]
    assert "Echo Vault sits below" in response.injected_content
    assert response.diagnostics
    assert response.diagnostics[0].world_book_id == world_book_id


@pytest.mark.integration
@pytest.mark.asyncio
async def test_dictionary_process_accepts_client_ordered_dictionary_ids(
    character_db: CharactersRAGDB,
) -> None:
    service = ChatDictionaryService(character_db)
    dictionary_z = service.create_dictionary("Zeta Dictionary", None)
    dictionary_a = service.create_dictionary("Alpha Dictionary", None)
    service.add_entry(dictionary_z, pattern="token", replacement="Z")
    service.add_entry(dictionary_a, pattern="Z", replacement="A")

    request = ProcessTextRequest.model_validate(
        {
            "text": "token",
            "dictionary_ids": [dictionary_z, dictionary_a],
            "max_iterations": 1,
        }
    )
    response = await chat_dictionaries.process_text_with_dictionaries(
        request,
        db=character_db,
    )

    assert response.processed_text == "A"
    assert response.replacements == 2
    assert len(response.entries_used) == 2


@pytest.mark.integration
@pytest.mark.asyncio
async def test_chat_settings_preserve_conversation_context_shape(
    character_db: CharactersRAGDB,
) -> None:
    user = _test_user()
    chat_id = _create_blank_chat(character_db, user)
    payload = ChatSettingsUpdate(
        settings={
            "conversationContext": {
                "world_book_ids": [1, 2],
                "chat_dictionary_ids": [7, 42],
            },
            "chat_dictionary_ids": [7, 42],
        }
    )

    updated = await character_chat_sessions.update_chat_settings(
        payload=payload,
        chat_id=chat_id,
        scope_type=None,
        workspace_id=None,
        db=character_db,
        current_user=user,
    )
    fetched = await character_chat_sessions.get_chat_settings(
        chat_id=chat_id,
        scope_type=None,
        workspace_id=None,
        db=character_db,
        current_user=user,
    )

    assert updated.settings["conversationContext"] == {
        "world_book_ids": [1, 2],
        "chat_dictionary_ids": [7, 42],
    }
    assert fetched.settings["conversationContext"] == updated.settings["conversationContext"]
    assert fetched.settings["chat_dictionary_ids"] == [7, 42]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_invalid_context_primitive_ids_return_domain_errors(
    character_db: CharactersRAGDB,
) -> None:
    with pytest.raises(HTTPException) as worldbook_exc:
        await characters_endpoint.process_context_with_world_info(
            ProcessContextRequest(
                text="Echo Vault",
                world_book_ids=[999_999],
                token_budget=500,
            ),
            db=character_db,
        )

    assert worldbook_exc.value.status_code in {400, 404}

    dictionary_request = ProcessTextRequest.model_validate(
        {
            "text": "token",
            "dictionary_ids": [999_999],
        }
    )
    with pytest.raises(HTTPException) as dictionary_exc:
        await chat_dictionaries.process_text_with_dictionaries(
            dictionary_request,
            db=character_db,
        )

    assert dictionary_exc.value.status_code in {400, 404}
