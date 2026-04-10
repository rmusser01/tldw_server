"""Integration regressions for manual character memory extraction ownership."""

from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.tests.Character_Chat_NEW.integration.test_character_chat_stream_and_persist import (
    _create_character_and_chat,
)


pytestmark = pytest.mark.integration


def test_extract_character_memories_allows_owned_chat_by_client_id(
    test_client: TestClient,
    auth_headers,
    character_db,
    monkeypatch: pytest.MonkeyPatch,
):
    character_id, chat_id = _create_character_and_chat(test_client, auth_headers)
    character_db.execute_query(
        "UPDATE conversations SET client_id = ? WHERE id = ?",
        ("1", chat_id),
        commit=True,
    )

    user_message_id = character_db.add_message(
        {
            "id": str(uuid.uuid4()),
            "conversation_id": chat_id,
            "sender": "user",
            "content": "Remember that I like tea.",
            "client_id": "1",
            "version": 1,
        }
    )
    character_db.add_message(
        {
            "id": str(uuid.uuid4()),
            "conversation_id": chat_id,
            "sender": "assistant",
            "content": "I will remember that.",
            "parent_message_id": user_message_id,
            "client_id": "1",
            "version": 1,
        }
    )

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Character_Chat.modules.character_memory_extraction.extract_character_memories",
        lambda **_: SimpleNamespace(unique=[], total_parsed=0, duplicates_skipped=0),
    )

    response = test_client.post(
        f"/api/v1/characters/{character_id}/memories/extract",
        json={"chat_id": chat_id, "provider": "openai", "model": "test-model"},
        headers=auth_headers,
    )

    assert response.status_code == 200
    assert response.json() == {"extracted": 0, "skipped_duplicates": 0, "memories": []}


def test_extract_character_memories_rejects_foreign_chat(
    test_client: TestClient,
    auth_headers,
    character_db,
    monkeypatch: pytest.MonkeyPatch,
):
    character_id, chat_id = _create_character_and_chat(test_client, auth_headers)
    character_db.execute_query(
        "UPDATE conversations SET client_id = ? WHERE id = ?",
        ("999", chat_id),
        commit=True,
    )

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Character_Chat.modules.character_memory_extraction.extract_character_memories",
        lambda **_: SimpleNamespace(unique=[], total_parsed=0, duplicates_skipped=0),
    )

    response = test_client.post(
        f"/api/v1/characters/{character_id}/memories/extract",
        json={"chat_id": chat_id, "provider": "openai", "model": "test-model"},
        headers=auth_headers,
    )

    assert response.status_code == 403
    assert response.json()["detail"] == "Not your chat session"


def test_extract_character_memories_rejects_chat_for_different_character(
    test_client: TestClient,
    auth_headers,
    character_db,
    monkeypatch: pytest.MonkeyPatch,
):
    character_id, chat_id = _create_character_and_chat(test_client, auth_headers)
    other_character_response = test_client.post(
        "/api/v1/characters/",
        json={
            "name": "Other Character",
            "description": "Secondary character",
            "personality": "Reserved",
            "first_message": "Hello there",
        },
        headers=auth_headers,
    )
    assert other_character_response.status_code == 201
    other_character_id = other_character_response.json()["id"]

    character_db.execute_query(
        "UPDATE conversations SET client_id = ?, character_id = ? WHERE id = ?",
        ("1", character_id, chat_id),
        commit=True,
    )

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Character_Chat.modules.character_memory_extraction.extract_character_memories",
        lambda **_: SimpleNamespace(unique=[], total_parsed=0, duplicates_skipped=0),
    )

    response = test_client.post(
        f"/api/v1/characters/{other_character_id}/memories/extract",
        json={"chat_id": chat_id, "provider": "openai", "model": "test-model"},
        headers=auth_headers,
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Chat session must belong to the requested character"
