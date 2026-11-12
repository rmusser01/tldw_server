"""Trash -> Restore must bring a chat back with the messages it had (CS-N2, #3104).

Trashing a chat hides it, and its messages, through the conversation's own
``deleted`` flag. Restoring it must return the transcript exactly as it was
when it was trashed: every message, in the same order, unmodified. A message
the user deleted on its own before trashing the chat must stay deleted.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

pytestmark = pytest.mark.integration


def _create_chat_with_messages(
    test_client: TestClient,
    auth_headers: dict[str, str],
    contents: tuple[str, ...],
) -> str:
    """Create a character chat and append one message per content string."""
    character = test_client.post(
        "/api/v1/characters/",
        json={
            "name": "Trash Restore Character",
            "description": "Character for trash/restore tests",
            "personality": "Patient",
            "first_message": "Hello there!",
        },
        headers=auth_headers,
    )
    assert character.status_code == 201, character.text
    chat = test_client.post(
        "/api/v1/chats/",
        json={"character_id": character.json()["id"], "title": "Trash restore chat"},
        headers=auth_headers,
    )
    assert chat.status_code == 201, chat.text
    chat_id = chat.json()["id"]
    roles = ("user", "assistant")
    for index, content in enumerate(contents):
        sent = test_client.post(
            f"/api/v1/chats/{chat_id}/messages",
            json={"role": roles[index % 2], "content": content},
            headers=auth_headers,
        )
        assert sent.status_code == 201, sent.text
    return chat_id


def _list_messages(
    test_client: TestClient,
    auth_headers: dict[str, str],
    chat_id: str,
) -> list[dict[str, Any]]:
    """Return the chat's visible messages through the public API."""
    response = test_client.get(
        f"/api/v1/chats/{chat_id}/messages",
        params={"limit": 200},
        headers=auth_headers,
    )
    assert response.status_code == 200, response.text
    return response.json()["messages"]


def _snapshot(messages: list[dict[str, Any]]) -> list[tuple[str, str, str, int]]:
    """Reduce messages to the fields a restore must preserve, in order."""
    return [
        (message["id"], message["sender"], message["content"], message["version"])
        for message in messages
    ]


def _trash(test_client: TestClient, auth_headers: dict[str, str], chat_id: str) -> None:
    deleted = test_client.delete(f"/api/v1/chats/{chat_id}", headers=auth_headers)
    assert deleted.status_code == 204, deleted.text


def _restore(test_client: TestClient, auth_headers: dict[str, str], chat_id: str) -> dict[str, Any]:
    restored = test_client.post(f"/api/v1/chats/{chat_id}/restore", headers=auth_headers)
    assert restored.status_code == 200, restored.text
    return restored.json()


def test_restore_from_trash_returns_every_message_in_original_order(
    test_client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    contents = ("First question", "First answer", "Second question", "Second answer")
    chat_id = _create_chat_with_messages(test_client, auth_headers, contents)
    before = _list_messages(test_client, auth_headers, chat_id)
    assert [message["content"] for message in before][-len(contents):] == list(contents)

    _trash(test_client, auth_headers, chat_id)
    restored = _restore(test_client, auth_headers, chat_id)

    assert restored["message_count"] == len(before)
    assert _snapshot(_list_messages(test_client, auth_headers, chat_id)) == _snapshot(before)


def test_trashed_chat_hides_its_messages_until_restored(
    test_client: TestClient,
    auth_headers: dict[str, str],
    character_db: CharactersRAGDB,
) -> None:
    chat_id = _create_chat_with_messages(test_client, auth_headers, ("Kestrel-411 marker",))
    assert [row["conversation_id"] for row in character_db.search_messages_by_content("Kestrel")] == [chat_id]

    _trash(test_client, auth_headers, chat_id)

    hidden = test_client.get(f"/api/v1/chats/{chat_id}/messages", headers=auth_headers)
    assert hidden.status_code == 404, hidden.text
    assert character_db.get_messages_for_conversation(chat_id, include_deleted=True) == []
    assert character_db.search_messages_by_content("Kestrel") == []

    _restore(test_client, auth_headers, chat_id)

    assert [row["conversation_id"] for row in character_db.search_messages_by_content("Kestrel")] == [chat_id]


def test_message_deleted_before_trash_stays_deleted_after_restore(
    test_client: TestClient,
    auth_headers: dict[str, str],
    character_db: CharactersRAGDB,
) -> None:
    chat_id = _create_chat_with_messages(
        test_client,
        auth_headers,
        ("Keep this question", "Delete this answer", "Keep this follow-up"),
    )
    before = _list_messages(test_client, auth_headers, chat_id)
    removed = next(message for message in before if message["content"] == "Delete this answer")
    deleted_message = test_client.delete(
        f"/api/v1/messages/{removed['id']}",
        params={"expected_version": removed["version"]},
        headers=auth_headers,
    )
    assert deleted_message.status_code == 204, deleted_message.text
    survivors = _list_messages(test_client, auth_headers, chat_id)
    assert removed["id"] not in {message["id"] for message in survivors}

    _trash(test_client, auth_headers, chat_id)
    restored = _restore(test_client, auth_headers, chat_id)

    assert restored["message_count"] == len(survivors)
    assert _snapshot(_list_messages(test_client, auth_headers, chat_id)) == _snapshot(survivors)
    stored = character_db.get_message_by_id(removed["id"], include_deleted=True)
    assert stored is not None
    assert bool(stored["deleted"]) is True


def test_restore_is_idempotent(
    test_client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    chat_id = _create_chat_with_messages(test_client, auth_headers, ("Only question", "Only answer"))
    before = _list_messages(test_client, auth_headers, chat_id)

    _trash(test_client, auth_headers, chat_id)
    first = _restore(test_client, auth_headers, chat_id)
    second = _restore(test_client, auth_headers, chat_id)

    assert second["version"] == first["version"]
    assert second["message_count"] == first["message_count"] == len(before)
    assert _snapshot(_list_messages(test_client, auth_headers, chat_id)) == _snapshot(before)


def test_trash_restore_cycles_keep_the_transcript(
    test_client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    chat_id = _create_chat_with_messages(test_client, auth_headers, ("Cycle question", "Cycle answer"))
    before = _list_messages(test_client, auth_headers, chat_id)

    for _ in range(2):
        _trash(test_client, auth_headers, chat_id)
        _restore(test_client, auth_headers, chat_id)

    assert _snapshot(_list_messages(test_client, auth_headers, chat_id)) == _snapshot(before)


def test_permanent_delete_after_trash_removes_the_messages(
    test_client: TestClient,
    auth_headers: dict[str, str],
    character_db: CharactersRAGDB,
) -> None:
    chat_id = _create_chat_with_messages(test_client, auth_headers, ("Purge question", "Purge answer"))
    message_ids = [message["id"] for message in _list_messages(test_client, auth_headers, chat_id)]

    _trash(test_client, auth_headers, chat_id)
    purged = test_client.delete(
        f"/api/v1/chats/{chat_id}",
        params={"hard_delete": True},
        headers=auth_headers,
    )

    assert purged.status_code == 204, purged.text
    assert character_db.get_conversation_by_id(chat_id, include_deleted=True) is None
    assert [character_db.get_message_by_id(message_id, include_deleted=True) for message_id in message_ids] == [
        None
    ] * len(message_ids)
