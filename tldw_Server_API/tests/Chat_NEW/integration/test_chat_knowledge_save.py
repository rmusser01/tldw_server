"""Integration tests for the chat knowledge save endpoint."""

from __future__ import annotations

import pytest
from fastapi import status

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


@pytest.mark.integration
def test_save_chat_knowledge_creates_note_and_flashcard(
    test_client,
    chacha_db: CharactersRAGDB,
    auth_headers,
):
    """Successful save should create both note and flashcard."""
    from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user

    # Seed minimal character, conversation, and message owned by user id "1"
    character_id = chacha_db.add_character_card(
        {
            "name": "Test Character",
            "description": "desc",
            "personality": "helpful",
            "system_prompt": "You are helpful.",
            "client_id": "1",
        }
    )
    conversation_id = chacha_db.add_conversation(
        {
            "character_id": character_id,
            "title": "Test Conversation",
            "client_id": "1",
        }
    )
    message_id = chacha_db.add_message(
        {
            "conversation_id": conversation_id,
            "sender": "user",
            "content": "Hello from user",
            "client_id": "1",
        }
    )

    assert chacha_db.count_notes() == 0

    def override_get_db():

        return chacha_db

    test_client.app.dependency_overrides[get_chacha_db_for_user] = override_get_db

    try:
        payload = {
            "conversation_id": conversation_id,
            "message_id": message_id,
            "snippet": "Important snippet",
            "tags": ["foo", "bar"],
            "make_flashcard": True,
            "flashcard_front": "What should be remembered?",
            "flashcard_back": "Important snippet",
            "export_to": "none",
        }

        resp = test_client.post(
            "/api/v1/chat/knowledge/save",
            json=payload,
            headers=auth_headers,
        )

        assert resp.status_code == status.HTTP_201_CREATED
        data = resp.json()
        assert data["note_id"]
        assert data["flashcard_id"]

        note = chacha_db.get_note_by_id(data["note_id"])
        assert note is not None
        assert note.get("conversation_id") == conversation_id
    finally:
        test_client.app.dependency_overrides.pop(get_chacha_db_for_user, None)


@pytest.mark.integration
def test_save_chat_knowledge_rolls_back_on_flashcard_error(
    test_client,
    chacha_db: CharactersRAGDB,
    auth_headers,
):
    """If flashcard creation fails, the note insert should be rolled back."""
    from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user

    character_id = chacha_db.add_character_card(
        {
            "name": "Test Character",
            "description": "desc",
            "personality": "helpful",
            "system_prompt": "You are helpful.",
            "client_id": "1",
        }
    )
    conversation_id = chacha_db.add_conversation(
        {
            "character_id": character_id,
            "title": "Test Conversation",
            "client_id": "1",
        }
    )
    message_id = chacha_db.add_message(
        {
            "conversation_id": conversation_id,
            "sender": "user",
            "content": "Hello from user",
            "client_id": "1",
        }
    )

    assert chacha_db.count_notes() == 0

    def override_get_db():

        return chacha_db

    test_client.app.dependency_overrides[get_chacha_db_for_user] = override_get_db

    # Force flashcard creation to fail after the note insert.
    def _failing_add_flashcard(card_data):
        raise RuntimeError("flashcard failure")

    original_add_flashcard = chacha_db.add_flashcard
    chacha_db.add_flashcard = _failing_add_flashcard  # type: ignore[assignment]

    try:
        payload = {
            "conversation_id": conversation_id,
            "message_id": message_id,
            "snippet": "Snippet that should not persist on failure",
            "tags": ["foo"],
            "make_flashcard": True,
            "flashcard_front": "What should be remembered?",
            "flashcard_back": "Snippet that should not persist on failure",
            "export_to": "none",
        }

        resp = test_client.post(
            "/api/v1/chat/knowledge/save",
            json=payload,
            headers=auth_headers,
        )

        # Endpoint should surface a controlled 500, not leak the runtime error.
        assert resp.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        body = resp.json()
        assert body.get("detail") == "Failed to save snippet"

        # Transaction should have rolled back the note insert.
        assert chacha_db.count_notes() == 0
    finally:
        chacha_db.add_flashcard = original_add_flashcard  # type: ignore[assignment]
        test_client.app.dependency_overrides.pop(get_chacha_db_for_user, None)
