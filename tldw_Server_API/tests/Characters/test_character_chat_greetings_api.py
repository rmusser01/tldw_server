pytest_plugins = [
    "tldw_Server_API.tests._plugins.chat_fixtures",
]

import pytest
from fastapi import status
from unittest.mock import patch

from tldw_Server_API.app.api.v1.endpoints.character_chat_sessions import _compute_greetings_checksum


def _create_character_with_alts(db):


    return db.add_character_card({
        "name": "GreeterAPI",
        "description": "A character with alternate greetings",
        "first_message": "Hello, {{user}}.",
        "alternate_greetings": ["Hey there, {{user}}!", "Welcome, {{user}}."],
    })


def test_create_chat_with_default_greeting(authenticated_client, mock_chacha_db, setup_dependencies, auth_headers):


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

    # Patch random.choice to force a deterministic selection
    with patch(
        "tldw_Server_API.app.api.v1.endpoints.character_chat_sessions.random.choice",
        return_value="Hey there, {{user}}!",
    ):
        resp = authenticated_client.post(
            "/api/v1/chats/",
            params={
                "seed_first_message": True,
                "greeting_strategy": "alternate_random",
            },
            json={"character_id": char_id},
        )

    assert resp.status_code == status.HTTP_201_CREATED
    data = resp.json()
    chat_id = data["id"]

    # Assert: first stored message equals the patched random choice
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


def test_seeded_chat_creation_persists_greeting_checksum(
    authenticated_client, mock_chacha_db, setup_dependencies, auth_headers
):
    char_id = _create_character_with_alts(mock_chacha_db)
    character = mock_chacha_db.get_character_card_by_id(char_id)
    assert character is not None
    expected_checksum = _compute_greetings_checksum(character)

    create_resp = authenticated_client.post(
        "/api/v1/chats/",
        params={
            "seed_first_message": True,
            "greeting_strategy": "default",
        },
        json={"character_id": char_id},
    )
    assert create_resp.status_code == status.HTTP_201_CREATED
    chat_id = create_resp.json()["id"]

    detail_resp = authenticated_client.get(
        f"/api/v1/chats/{chat_id}",
        params={"include_settings": True},
        headers=auth_headers,
    )

    assert detail_resp.status_code == status.HTTP_200_OK
    detail = detail_resp.json()
    assert detail["settings"]["greetingsChecksum"] == expected_checksum


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
