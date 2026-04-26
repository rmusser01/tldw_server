"""
Integration tests for streaming stub (offline-sim) and persist endpoint.
"""

from datetime import datetime, timezone

import pytest
pytestmark = pytest.mark.integration
from fastapi.testclient import TestClient


def _create_character_and_chat(client: TestClient, headers) -> tuple[int, str]:
    char_resp = client.post(
        "/api/v1/characters/",
        json={
            "name": "StreamChar",
            "description": "Streaming test",
            "personality": "Calm",
            "first_message": "Hello!"
        },
        headers=headers,
    )
    assert char_resp.status_code == 201
    char_id = char_resp.json()["id"]
    chat_resp = client.post(
        "/api/v1/chats/",
        json={"character_id": char_id, "title": "Streaming Chat"},
        headers=headers,
    )
    assert chat_resp.status_code == 201
    return char_id, chat_resp.json()["id"]


def test_offline_streaming_and_persist_flow(test_client: TestClient, auth_headers) -> None:
    # Ensure offline-sim mode by not enabling any ENABLE_LOCAL_LLM_PROVIDER, etc.
    char_id, chat_id = _create_character_and_chat(test_client, auth_headers)

    # Request a streamed completion with an appended user message
    user_prompt = "This is a streamed test"
    with test_client.stream(
        "POST",
        f"/api/v1/chats/{chat_id}/complete-v2",
        json={
            "provider": "local-llm",
            "model": "local-test",
            "append_user_message": user_prompt,
            "stream": True,
            "include_character_context": False,
        },
        headers=auth_headers,
    ) as resp:
        assert resp.status_code == 200
        # Collect lines; expect at least one data line and a final [DONE]
        body = resp.iter_lines()
        lines = []
        for line in body:
            if not line:
                continue
            try:
                s = line.decode() if isinstance(line, (bytes, bytearray)) else str(line)
            except Exception:
                s = str(line)
            lines.append(s)
            if s.strip().lower() == "data: [done]":
                break

        assert any(l.startswith("data: ") for l in lines), "Expected SSE data lines"
        assert any(l.strip().lower() == "data: [done]" for l in lines), "Expected final [DONE]"
        # Combined payload should contain the user prompt content (echo behavior)
        combined = "\n".join(lines)
        assert user_prompt in combined

    # Persist the assistant content after streaming
    persist_resp = test_client.post(
        f"/api/v1/chats/{chat_id}/completions/persist",
        json={
            "assistant_content": user_prompt,
            "user_message_id": None,
        },
        headers=auth_headers,
    )
    assert persist_resp.status_code == 200
    data = persist_resp.json()
    assert data.get("saved") is True
    assert data.get("assistant_message_id")

    # Verify message was persisted
    messages_resp = test_client.get(
        f"/api/v1/chats/{chat_id}/messages",
        headers=auth_headers,
    )
    assert messages_resp.status_code == 200
    msgs = messages_resp.json().get("messages")
    assert any(m.get("content") == user_prompt and m.get("sender") == "StreamChar" for m in msgs)


def test_persist_streamed_message_preserves_active_speaker_identity(test_client: TestClient, auth_headers) -> None:
    """Persist endpoint should allow preserving the directed/active character speaker."""
    # Primary character + chat
    char_resp = test_client.post(
        "/api/v1/characters/",
        json={
            "name": "PrimarySpeaker",
            "description": "Primary character",
            "personality": "Calm",
            "first_message": "Hello!",
        },
        headers=auth_headers,
    )
    assert char_resp.status_code == 201
    primary = char_resp.json()
    primary_id = primary["id"]

    # Secondary character participant
    secondary_resp = test_client.post(
        "/api/v1/characters/",
        json={
            "name": "SecondarySpeaker",
            "description": "Secondary character",
            "personality": "Assertive",
            "first_message": "Hi!",
        },
        headers=auth_headers,
    )
    assert secondary_resp.status_code == 201
    secondary = secondary_resp.json()
    secondary_id = secondary["id"]
    secondary_name = secondary["name"]

    chat_resp = test_client.post(
        "/api/v1/chats/",
        json={"character_id": primary_id, "title": "Persist Speaker Identity"},
        headers=auth_headers,
    )
    assert chat_resp.status_code == 201
    chat_id = chat_resp.json()["id"]

    # Enable round-robin participant context
    settings_resp = test_client.put(
        f"/api/v1/chats/{chat_id}/settings",
        json={
            "settings": {
                "schemaVersion": 2,
                "updatedAt": datetime.now(timezone.utc).isoformat(),
                "turnTakingMode": "round_robin",
                "participantCharacterIds": [secondary_id],
            }
        },
        headers=auth_headers,
    )
    assert settings_resp.status_code == 200

    # Dry-run a directed turn to confirm active speaker resolution points at secondary
    directed_resp = test_client.post(
        f"/api/v1/chats/{chat_id}/complete-v2",
        json={
            "provider": "local-llm",
            "model": "local-test",
            "append_user_message": "direct this to secondary",
            "save_to_db": False,
            "directed_character_id": secondary_id,
        },
        headers=auth_headers,
    )
    assert directed_resp.status_code == 200
    directed_payload = directed_resp.json()
    assert directed_payload.get("speaker_character_id") == secondary_id
    assert directed_payload.get("speaker_character_name") == secondary_name

    # Persist streamed text and request secondary as speaker identity.
    # Current API ignores these speaker fields, causing persistence under primary sender.
    persist_resp = test_client.post(
        f"/api/v1/chats/{chat_id}/completions/persist",
        json={
            "assistant_content": "secondary streamed response",
            "speaker_character_id": secondary_id,
            "speaker_character_name": secondary_name,
            "mood_label": "happy",
            "mood_confidence": 0.92,
            "mood_topic": "celebration",
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 8,
                "total_tokens": 20,
            },
        },
        headers=auth_headers,
    )
    assert persist_resp.status_code == 200
    persisted = persist_resp.json()
    assistant_message_id = persisted.get("assistant_message_id")
    assert assistant_message_id

    # The persisted sender should match the requested active speaker.
    messages_resp = test_client.get(f"/api/v1/chats/{chat_id}/messages", headers=auth_headers)
    assert messages_resp.status_code == 200
    messages = messages_resp.json().get("messages") or []
    persisted_msg = next((m for m in messages if m.get("id") == assistant_message_id), None)
    assert persisted_msg is not None
    assert persisted_msg.get("sender") == secondary_name

    message_resp = test_client.get(
        f"/api/v1/messages/{assistant_message_id}",
        params={"include_metadata": "true"},
        headers=auth_headers,
    )
    assert message_resp.status_code == 200
    metadata_extra = message_resp.json().get("metadata_extra") or {}
    assert metadata_extra.get("speaker_character_id") == secondary_id
    assert metadata_extra.get("speaker_character_name") == secondary_name
    assert metadata_extra.get("turn_taking_mode") == "round_robin"
    assert metadata_extra.get("mood_label") == "happy"
    assert metadata_extra.get("mood_confidence") == pytest.approx(0.92)
    assert metadata_extra.get("mood_topic") == "celebration"
    assert metadata_extra.get("usage") == {
        "prompt_tokens": 12,
        "completion_tokens": 8,
        "total_tokens": 20,
    }


def test_persist_streamed_message_saves_then_returns_503_when_counting_degrades(
    test_client: TestClient,
    auth_headers,
    character_db,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, chat_id = _create_character_and_chat(test_client, auth_headers)

    original_count = character_db.count_messages_for_conversation

    def broken_count(_chat_id: str) -> int:
        raise RuntimeError("count unavailable")

    monkeypatch.setattr(character_db, "count_messages_for_conversation", broken_count)

    response = test_client.post(
        f"/api/v1/chats/{chat_id}/completions/persist",
        json={
            "assistant_content": "saved once despite degraded validation",
            "assistant_message_id": "assistant-count-degraded-1",
        },
        headers=auth_headers,
    )

    assert response.status_code == 503
    payload = response.json()["detail"]
    assert payload["code"] == "persist_validation_degraded"
    assert payload["saved"] is True
    assert payload["assistant_message_id"]

    monkeypatch.setattr(character_db, "count_messages_for_conversation", original_count)
    messages = test_client.get(f"/api/v1/chats/{chat_id}/messages", headers=auth_headers).json()["messages"]
    assert [m["content"] for m in messages].count("saved once despite degraded validation") == 1


def test_persist_streamed_message_retry_reuses_saved_degraded_outcome(
    test_client: TestClient,
    auth_headers,
    character_db,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, chat_id = _create_character_and_chat(test_client, auth_headers)
    monkeypatch.setattr(
        character_db,
        "count_messages_for_conversation",
        lambda _chat_id: (_ for _ in ()).throw(RuntimeError("count unavailable")),
    )

    first = test_client.post(
        f"/api/v1/chats/{chat_id}/completions/persist",
        json={
            "assistant_content": "duplicate guard text",
            "assistant_message_id": "assistant-retry-guard-1",
            "user_message_id": None,
        },
        headers=auth_headers,
    )
    second = test_client.post(
        f"/api/v1/chats/{chat_id}/completions/persist",
        json={
            "assistant_content": "duplicate guard text",
            "assistant_message_id": "assistant-retry-guard-1",
            "user_message_id": None,
        },
        headers=auth_headers,
    )

    assert first.status_code == 503
    assert second.status_code == 503
    assert first.json()["detail"]["assistant_message_id"] == second.json()["detail"]["assistant_message_id"]

    messages = test_client.get(f"/api/v1/chats/{chat_id}/messages", headers=auth_headers).json()["messages"]
    assert [m["content"] for m in messages].count("duplicate guard text") == 1


def test_persist_streamed_message_retry_reuses_assistant_message_id_when_metadata_write_fails(
    test_client: TestClient,
    auth_headers,
    character_db,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, chat_id = _create_character_and_chat(test_client, auth_headers)
    monkeypatch.setattr(character_db, "add_message_metadata", lambda *args, **kwargs: False)

    payload = {
        "assistant_content": "metadata failure duplicate guard",
        "assistant_message_id": "assistant-metadata-failure-1",
        "user_message_id": None,
    }

    first = test_client.post(
        f"/api/v1/chats/{chat_id}/completions/persist",
        json=payload,
        headers=auth_headers,
    )
    second = test_client.post(
        f"/api/v1/chats/{chat_id}/completions/persist",
        json=payload,
        headers=auth_headers,
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["assistant_message_id"] == payload["assistant_message_id"]
    assert second.json()["assistant_message_id"] == payload["assistant_message_id"]

    messages = test_client.get(f"/api/v1/chats/{chat_id}/messages", headers=auth_headers).json()["messages"]
    assert [m["content"] for m in messages].count("metadata failure duplicate guard") == 1


def test_persist_streamed_message_retry_reapplies_metadata_and_rating(
    test_client: TestClient,
    auth_headers,
    character_db,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, chat_id = _create_character_and_chat(test_client, auth_headers)
    real_add_message_metadata = character_db.add_message_metadata
    monkeypatch.setattr(character_db, "add_message_metadata", lambda *args, **kwargs: False)

    payload = {
        "assistant_content": "retry should restore metadata",
        "assistant_message_id": "assistant-retry-side-effects-1",
        "mood_label": "focused",
        "usage": {
            "prompt_tokens": 5,
            "completion_tokens": 7,
            "total_tokens": 12,
        },
        "chat_rating": 4,
    }

    first = test_client.post(
        f"/api/v1/chats/{chat_id}/completions/persist",
        json=payload,
        headers=auth_headers,
    )
    assert first.status_code == 200

    monkeypatch.setattr(character_db, "add_message_metadata", real_add_message_metadata)

    second = test_client.post(
        f"/api/v1/chats/{chat_id}/completions/persist",
        json=payload,
        headers=auth_headers,
    )

    assert second.status_code == 200

    message_resp = test_client.get(
        f"/api/v1/messages/{payload['assistant_message_id']}",
        params={"include_metadata": "true"},
        headers=auth_headers,
    )
    assert message_resp.status_code == 200
    metadata_extra = message_resp.json().get("metadata_extra") or {}
    assert metadata_extra.get("mood_label") == "focused"
    assert metadata_extra.get("usage") == {
        "prompt_tokens": 5,
        "completion_tokens": 7,
        "total_tokens": 12,
    }

    conversation = character_db.get_conversation_by_id(chat_id) or {}
    assert conversation.get("rating") == 4


def test_persist_streamed_message_rejects_non_object_usage(
    test_client: TestClient,
    auth_headers,
) -> None:
    _, chat_id = _create_character_and_chat(test_client, auth_headers)

    response = test_client.post(
        f"/api/v1/chats/{chat_id}/completions/persist",
        json={
            "assistant_content": "invalid usage payload",
            "usage": ["not", "an", "object"],
        },
        headers=auth_headers,
    )

    assert response.status_code == 422


def test_persist_streamed_message_allows_duplicate_content_without_idempotency_key(
    test_client: TestClient,
    auth_headers,
) -> None:
    _, chat_id = _create_character_and_chat(test_client, auth_headers)

    first = test_client.post(
        f"/api/v1/chats/{chat_id}/completions/persist",
        json={"assistant_content": "same visible reply", "user_message_id": None},
        headers=auth_headers,
    )
    second = test_client.post(
        f"/api/v1/chats/{chat_id}/completions/persist",
        json={"assistant_content": "same visible reply", "user_message_id": None},
        headers=auth_headers,
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["assistant_message_id"] != second.json()["assistant_message_id"]

    messages = test_client.get(f"/api/v1/chats/{chat_id}/messages", headers=auth_headers).json()["messages"]
    assert [m["content"] for m in messages].count("same visible reply") == 2


def test_persist_streamed_message_accepts_senderless_existing_message_for_idempotent_retry(
    test_client: TestClient,
    auth_headers,
    character_db,
) -> None:
    _, chat_id = _create_character_and_chat(test_client, auth_headers)
    payload = {
        "assistant_content": "senderless idempotent reply",
        "assistant_message_id": "assistant-senderless-retry-1",
    }

    first = test_client.post(
        f"/api/v1/chats/{chat_id}/completions/persist",
        json=payload,
        headers=auth_headers,
    )
    assert first.status_code == 200

    character_db.execute_query(
        "UPDATE messages SET sender = '' WHERE id = ?",
        (payload["assistant_message_id"],),
        commit=True,
    )

    second = test_client.post(
        f"/api/v1/chats/{chat_id}/completions/persist",
        json=payload,
        headers=auth_headers,
    )

    assert second.status_code == 200
    assert second.json()["assistant_message_id"] == payload["assistant_message_id"]


def test_persist_streamed_message_requires_request_body(test_client: TestClient, auth_headers) -> None:
    """Persist endpoint should return 422 when request body is missing."""
    _, chat_id = _create_character_and_chat(test_client, auth_headers)

    response = test_client.post(
        f"/api/v1/chats/{chat_id}/completions/persist",
        headers=auth_headers,
    )

    assert response.status_code == 422


def test_persist_streamed_message_rejects_empty_assistant_content(test_client: TestClient, auth_headers) -> None:
    """Persist endpoint should reject empty assistant_content with 422."""
    _, chat_id = _create_character_and_chat(test_client, auth_headers)

    response = test_client.post(
        f"/api/v1/chats/{chat_id}/completions/persist",
        json={"assistant_content": ""},
        headers=auth_headers,
    )

    assert response.status_code == 422


def test_persist_streamed_message_rejects_blank_assistant_message_id(
    test_client: TestClient,
    auth_headers,
) -> None:
    _, chat_id = _create_character_and_chat(test_client, auth_headers)

    response = test_client.post(
        f"/api/v1/chats/{chat_id}/completions/persist",
        json={
            "assistant_content": "valid reply",
            "assistant_message_id": "   ",
        },
        headers=auth_headers,
    )

    assert response.status_code == 422
