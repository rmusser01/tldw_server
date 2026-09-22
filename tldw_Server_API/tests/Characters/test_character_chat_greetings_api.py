from unittest.mock import patch

import pytest
from fastapi import status

pytest_plugins = [
    "tldw_Server_API.tests._plugins.chat_fixtures",
]


def _create_character_with_alts(db):


    return db.add_character_card({
        "name": "GreeterAPI",
        "description": "A character with alternate greetings",
        "first_message": "Hello, {{user}}.",
        "alternate_greetings": ["Hey there, {{user}}!", "Welcome, {{user}}."],
    })


def test_create_chat_with_default_greeting(
    authenticated_client, mock_chacha_db, setup_dependencies, auth_headers, test_user
):


     # Arrange: create character with alt greetings
    char_id = _create_character_with_alts(mock_chacha_db)

    # Act: create chat, seed with default (first_message)
    resp = authenticated_client.post(
        "/api/v1/chats/",
        params={
            "seed_first_message": True,
            "greeting_strategy": "default",
        },
        json={"character_id": char_id},
    )

    assert resp.status_code == status.HTTP_201_CREATED
    data = resp.json()
    chat_id = data["id"]
    assert mock_chacha_db.client_id == str(test_user.id)
    assert mock_chacha_db.get_conversation_by_id(chat_id)["client_id"] == str(test_user.id)

    # Assert: first stored message equals first_message (raw with placeholders)
    r = authenticated_client.get(
        f"/api/v1/chats/{chat_id}/messages", params={"limit": 10}, headers=auth_headers
    )
    assert r.status_code == status.HTTP_200_OK
    msgs = r.json()
    assert isinstance(msgs, dict) and "total" in msgs
    first = msgs["messages"][0]
    assert first["sender"].lower() in {"assistant", "greeterapi"}
    # Endpoint resolves placeholders for display
    assert first["content"] == "Hello, User."


def test_create_chat_with_alternate_index_greeting(authenticated_client, mock_chacha_db, setup_dependencies, auth_headers):


     # Arrange: create character with alt greetings
    char_id = _create_character_with_alts(mock_chacha_db)

    # Act: create chat, seed with alternate_index=1
    resp = authenticated_client.post(
        "/api/v1/chats/",
        params={
            "seed_first_message": True,
            "greeting_strategy": "alternate_index",
            "alternate_index": 1,
        },
        json={"character_id": char_id},
    )

    assert resp.status_code == status.HTTP_201_CREATED
    data = resp.json()
    chat_id = data["id"]

    # Assert: first stored message equals the selected alternate (raw with placeholders)
    r = authenticated_client.get(
        f"/api/v1/chats/{chat_id}/messages", params={"limit": 10}, headers=auth_headers
    )
    assert r.status_code == status.HTTP_200_OK
    msgs = r.json()
    assert isinstance(msgs, dict) and "total" in msgs
    first = msgs["messages"][0]
    assert first["sender"].lower() in {"assistant", "greeterapi"}
    assert first["content"] == "Welcome, User."


def test_create_chat_with_alternate_random_greeting(authenticated_client, mock_chacha_db, setup_dependencies, auth_headers):


     # Arrange: create character with alt greetings
    char_id = _create_character_with_alts(mock_chacha_db)

    # Control the transactional factory's random value to select the first alternate.
    with patch(
        "tldw_Server_API.app.core.Character_Chat.character_conversation_factory.random.SystemRandom.random",
        return_value=0.0,
    ) as random_value:
        resp = authenticated_client.post(
            "/api/v1/chats/",
            params={
                "seed_first_message": True,
                "greeting_strategy": "alternate_random",
            },
            json={"character_id": char_id},
        )

    random_value.assert_called_once_with()
    assert resp.status_code == status.HTTP_201_CREATED
    data = resp.json()
    chat_id = data["id"]

    # Assert: first stored message equals the selected alternate
    r = authenticated_client.get(
        f"/api/v1/chats/{chat_id}/messages", params={"limit": 10}, headers=auth_headers
    )
    assert r.status_code == status.HTTP_200_OK
    msgs = r.json()
    assert isinstance(msgs, dict) and "total" in msgs
    first = msgs["messages"][0]
    assert first["sender"].lower() in {"assistant", "greeterapi"}
    assert first["content"] == "Hey there, User!"


def test_create_chat_with_alternate_index_out_of_range_falls_back_default(
    authenticated_client, mock_chacha_db, setup_dependencies, auth_headers
):


     # Arrange: create character with alt greetings
    char_id = _create_character_with_alts(mock_chacha_db)

    # Act: attempt to seed with an out-of-range index; should fall back to first_message
    resp = authenticated_client.post(
        "/api/v1/chats/",
        params={
            "seed_first_message": True,
            "greeting_strategy": "alternate_index",
            "alternate_index": 999,
        },
        json={"character_id": char_id},
    )

    assert resp.status_code == status.HTTP_201_CREATED
    data = resp.json()
    chat_id = data["id"]

    # Assert: first stored message equals first_message
    r = authenticated_client.get(
        f"/api/v1/chats/{chat_id}/messages", params={"limit": 10}, headers=auth_headers
    )
    assert r.status_code == status.HTTP_200_OK
    msgs = r.json()
    assert isinstance(msgs, dict) and "total" in msgs and msgs["total"] >= 1
    first = msgs["messages"][0]
    assert first["sender"].lower() in {"assistant", "greeterapi"}
    assert first["content"] == "Hello, User."


def test_message_list_total_reflects_full_conversation_count(
    authenticated_client, mock_chacha_db, setup_dependencies, auth_headers
):
    char_id = _create_character_with_alts(mock_chacha_db)

    resp = authenticated_client.post(
        "/api/v1/chats/",
        json={"character_id": char_id},
    )
    assert resp.status_code == status.HTTP_201_CREATED
    chat_id = resp.json()["id"]

    for idx in range(3):
        msg_resp = authenticated_client.post(
            f"/api/v1/chats/{chat_id}/messages",
            json={"role": "user", "content": f"msg {idx}"},
        )
        assert msg_resp.status_code == status.HTTP_201_CREATED

    resp = authenticated_client.get(
        f"/api/v1/chats/{chat_id}/messages",
        params={"limit": 1},
        headers=auth_headers,
    )
    assert resp.status_code == status.HTTP_200_OK
    payload = resp.json()
    assert payload["total"] == 3
    assert len(payload["messages"]) == 1

    resp = authenticated_client.get(
        f"/api/v1/chats/{chat_id}/messages",
        params={"limit": 1, "format_for_completions": True},
        headers=auth_headers,
    )
    assert resp.status_code == status.HTTP_200_OK
    payload = resp.json()
    assert payload["total"] == 3
    assert len(payload["messages"]) == 1


@pytest.mark.parametrize("database_owner", ["pytest_client", "2"])
def test_create_chat_rejects_mismatched_database_owner_without_writes(
    authenticated_client, mock_chacha_db, monkeypatch, database_owner, test_user
):
    char_id = _create_character_with_alts(mock_chacha_db)
    monkeypatch.setattr(mock_chacha_db, "client_id", database_owner)

    response = authenticated_client.post(
        "/api/v1/chats/",
        params={"seed_first_message": True, "greeting_strategy": "default"},
        json={"character_id": char_id},
    )

    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert response.json()["detail"] == "Conversation client_id must match the scoped database owner."
    assert mock_chacha_db.get_conversations_for_character(char_id) == []
    assert mock_chacha_db.get_conversations_for_character(char_id, client_id=str(test_user.id)) == []
