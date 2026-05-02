"""Integration regressions for manual character memory extraction ownership."""

from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.integration


def _create_character(
    client: TestClient,
    headers,
    *,
    name: str = "Memory Character",
) -> int:
    """Create a character card for character memory extraction tests."""
    response = client.post(
        "/api/v1/characters/",
        json={
            "name": name,
            "description": f"{name} description",
            "personality": "Calm",
            "first_message": "Hello!",
        },
        headers=headers,
    )
    assert response.status_code == 201
    return int(response.json()["id"])


def _create_conversation(
    character_db,
    *,
    character_id: int,
    client_id: str,
    title: str,
) -> str:
    """Create a conversation through the DB abstraction with the requested ownership."""
    chat_id = character_db.add_conversation(
        {
            "character_id": character_id,
            "title": title,
            "client_id": client_id,
        }
    )
    assert chat_id is not None
    return chat_id


def test_extract_character_memories_allows_owned_chat_by_client_id(
    test_client: TestClient,
    auth_headers,
    character_db,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Allow extraction when stored and request user IDs are numerically equivalent."""
    character_id = _create_character(test_client, auth_headers, name="Owned Memory Character")
    chat_id = _create_conversation(
        character_db,
        character_id=character_id,
        client_id="001",
        title="Owned Memory Chat",
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


def test_list_character_memories_includes_canonical_pagination(
    test_client: TestClient,
    auth_headers,
) -> None:
    """List character memories preserves total and adds canonical pagination."""
    character_id = _create_character(test_client, auth_headers, name="Paginated Memory Character")
    for index in range(4):
        response = test_client.post(
            f"/api/v1/characters/{character_id}/memories",
            json={"content": f"Remembered fact {index}", "memory_type": "manual"},
            headers=auth_headers,
        )
        assert response.status_code == 201

    response = test_client.get(
        f"/api/v1/characters/{character_id}/memories?limit=2&offset=1",
        headers=auth_headers,
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 4
    assert len(payload["memories"]) == 2
    assert payload["pagination"] == {
        "mode": "offset",
        "limit": 2,
        "offset": 1,
        "total": 4,
        "has_more": True,
        "next_offset": 3,
    }
    assert payload["has_more"] is True
    assert payload["next_offset"] == 3


def test_extract_character_memories_allows_string_owner_client_id(
    test_client: TestClient,
    auth_headers,
    character_db,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Allow extraction when a tenant-style string user ID owns the chat."""
    from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_current_user
    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
    from tldw_Server_API.app.main import app

    tenant_user = User(
        id="tenant-user",
        username="tenant-user",
        email="tenant@example.com",
        is_active=True,
    )

    async def _tenant_user() -> User:
        return tenant_user

    monkeypatch.setitem(app.dependency_overrides, get_request_user, _tenant_user)
    monkeypatch.setitem(app.dependency_overrides, get_current_user, _tenant_user)

    character_id = _create_character(test_client, auth_headers, name="String Owner Memory Character")
    chat_id = _create_conversation(
        character_db,
        character_id=character_id,
        client_id="tenant-user",
        title="String Owner Memory Chat",
    )
    character_db.add_message(
        {
            "id": str(uuid.uuid4()),
            "conversation_id": chat_id,
            "sender": "user",
            "content": "Remember that I prefer string tenant IDs.",
            "client_id": "tenant-user",
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
) -> None:
    """Reject extraction when the stored client_id belongs to another user."""
    character_id = _create_character(test_client, auth_headers, name="Foreign Memory Character")
    chat_id = _create_conversation(
        character_db,
        character_id=character_id,
        client_id="999",
        title="Foreign Memory Chat",
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
) -> None:
    """Reject extraction when the chat belongs to a different character."""
    character_id = _create_character(test_client, auth_headers, name="Primary Memory Character")
    chat_id = _create_conversation(
        character_db,
        character_id=character_id,
        client_id="1",
        title="Primary Memory Chat",
    )
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


def test_extract_character_memories_allows_normalized_character_id_match(
    test_client: TestClient,
    auth_headers,
    character_db,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Allow extraction when the stored conversation character ID is zero-padded."""
    character_id = _create_character(test_client, auth_headers, name="Normalized Memory Character")
    chat_id = _create_conversation(
        character_db,
        character_id=character_id,
        client_id="1",
        title="Normalized Memory Chat",
    )

    original_get_conversation_by_id = character_db.get_conversation_by_id

    def _get_padded_character_conversation(conversation_id: str) -> dict:
        conversation = original_get_conversation_by_id(conversation_id)
        assert conversation is not None
        return {
            **conversation,
            "character_id": f"{character_id:03d}",
        }

    monkeypatch.setattr(character_db, "get_conversation_by_id", _get_padded_character_conversation)
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
