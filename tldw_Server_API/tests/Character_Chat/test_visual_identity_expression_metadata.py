"""Tests for visual identity expression metadata in character chat turns."""

from __future__ import annotations

from collections.abc import Generator
from typing import Any
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints import character_chat_sessions
from tldw_Server_API.app.api.v1.endpoints.character_chat_sessions import (
    _build_stream_persist_metadata_extra,
)


pytestmark = pytest.mark.integration


@pytest.fixture
def test_client() -> Generator[TestClient, None, None]:
    from tldw_Server_API.app.main import app

    with TestClient(app) as client:
        yield client


def _create_character_and_chat(client: TestClient, headers: dict[str, str]) -> tuple[int, str]:
    character_name = f"VisualMetaChar-{uuid4().hex}"
    character_response = client.post(
        "/api/v1/characters/",
        json={
            "name": character_name,
            "description": "Character chat visual identity metadata test",
            "personality": "Calm",
            "first_message": "Hello!",
        },
        headers=headers,
    )
    assert character_response.status_code == 201
    character_id = int(character_response.json()["id"])

    chat_response = client.post(
        "/api/v1/chats/",
        json={"character_id": character_id, "title": "Visual Metadata Chat"},
        headers=headers,
    )
    assert chat_response.status_code == 201
    return character_id, str(chat_response.json()["id"])


def _resolved_visual_metadata(actor_id: int, expression_key: str = "happy") -> dict[str, Any]:
    return {
        "visual_actor_kind": "character",
        "visual_actor_id": actor_id,
        "visual_pack_id": 10,
        "visual_pack_version_id": 4,
        "visual_expression_key": expression_key,
        "visual_asset_id": 88,
        "visual_fallback_reason": "mood",
    }


def test_stream_persist_metadata_includes_visual_identity_fields() -> None:
    extra = _build_stream_persist_metadata_extra(
        speaker_character_id=5,
        speaker_character_name="Ari",
        turn_taking_mode="single",
        validation_degraded=False,
        persist_fingerprint="fp",
        mood_label="happy",
        mood_confidence=0.8,
        mood_topic=None,
        usage=None,
        visual_identity=_resolved_visual_metadata(actor_id=5),
    )

    assert extra["visual_actor_kind"] == "character"
    assert extra["visual_actor_id"] == 5
    assert extra["visual_pack_id"] == 10
    assert extra["visual_pack_version_id"] == 4
    assert extra["visual_expression_key"] == "happy"
    assert extra["visual_asset_id"] == 88
    assert extra["visual_fallback_reason"] == "mood"
    assert extra["mood_label"] == "happy"


def test_complete_v2_persists_resolved_visual_identity_metadata(
    test_client: TestClient,
    auth_headers: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    character_id, chat_id = _create_character_and_chat(test_client, auth_headers)
    calls: list[dict[str, Any]] = []

    def fake_resolver(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return _resolved_visual_metadata(actor_id=character_id, expression_key="happy")

    monkeypatch.setattr(
        character_chat_sessions,
        "resolve_character_visual_identity",
        fake_resolver,
        raising=False,
    )

    response = test_client.post(
        f"/api/v1/chats/{chat_id}/complete-v2",
        json={
            "provider": "local-llm",
            "model": "local-test",
            "append_user_message": "hello",
            "save_to_db": True,
            "mood_label": "happy",
            "mood_confidence": 0.77,
        },
        headers=auth_headers,
    )

    assert response.status_code == 200
    assistant_message_id = response.json()["assistant_message_id"]
    assert calls
    assert calls[0]["actor_id"] == character_id
    assert calls[0]["mood_label"] == "happy"

    message_response = test_client.get(
        f"/api/v1/messages/{assistant_message_id}",
        params={"include_metadata": "true"},
        headers=auth_headers,
    )
    assert message_response.status_code == 200
    metadata_extra = message_response.json()["metadata_extra"]
    assert metadata_extra["mood_label"] == "happy"
    assert metadata_extra["mood_confidence"] == pytest.approx(0.77)
    assert metadata_extra["visual_actor_kind"] == "character"
    assert metadata_extra["visual_actor_id"] == character_id
    assert metadata_extra["visual_pack_id"] == 10
    assert metadata_extra["visual_pack_version_id"] == 4
    assert metadata_extra["visual_expression_key"] == "happy"
    assert metadata_extra["visual_asset_id"] == 88
    assert metadata_extra["visual_fallback_reason"] == "mood"


def test_visual_identity_resolution_failure_does_not_block_character_reply(
    test_client: TestClient,
    auth_headers: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    character_id, chat_id = _create_character_and_chat(test_client, auth_headers)
    calls: list[dict[str, Any]] = []

    def raising_resolver(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        raise RuntimeError("visual resolver unavailable")

    monkeypatch.setattr(
        character_chat_sessions,
        "resolve_character_visual_identity",
        raising_resolver,
        raising=False,
    )

    response = test_client.post(
        f"/api/v1/chats/{chat_id}/complete-v2",
        json={
            "provider": "local-llm",
            "model": "local-test",
            "append_user_message": "hello despite resolver failure",
            "save_to_db": True,
            "mood_label": "happy",
        },
        headers=auth_headers,
    )

    assert response.status_code == 200
    assistant_message_id = response.json()["assistant_message_id"]
    assert calls
    assert calls[0]["actor_id"] == character_id

    message_response = test_client.get(
        f"/api/v1/messages/{assistant_message_id}",
        params={"include_metadata": "true"},
        headers=auth_headers,
    )
    assert message_response.status_code == 200
    metadata_extra = message_response.json()["metadata_extra"]
    assert metadata_extra["mood_label"] == "happy"
    assert "visual_actor_kind" not in metadata_extra
    assert "visual_asset_id" not in metadata_extra


def test_complete_v2_without_visual_identity_binding_keeps_metadata_clean(
    test_client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    _, chat_id = _create_character_and_chat(test_client, auth_headers)

    response = test_client.post(
        f"/api/v1/chats/{chat_id}/complete-v2",
        json={
            "provider": "local-llm",
            "model": "local-test",
            "append_user_message": "hello without a visual pack",
            "save_to_db": True,
            "mood_label": "happy",
            "mood_confidence": 0.71,
        },
        headers=auth_headers,
    )

    assert response.status_code == 200
    assistant_message_id = response.json()["assistant_message_id"]

    message_response = test_client.get(
        f"/api/v1/messages/{assistant_message_id}",
        params={"include_metadata": "true"},
        headers=auth_headers,
    )
    assert message_response.status_code == 200
    metadata_extra = message_response.json()["metadata_extra"]
    assert metadata_extra["mood_label"] == "happy"
    assert metadata_extra["mood_confidence"] == pytest.approx(0.71)
    assert not any(key.startswith("visual_") for key in metadata_extra)


def test_stream_persist_endpoint_persists_resolved_visual_identity_metadata(
    test_client: TestClient,
    auth_headers: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    character_id, chat_id = _create_character_and_chat(test_client, auth_headers)
    calls: list[dict[str, Any]] = []

    def fake_resolver(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return _resolved_visual_metadata(actor_id=character_id, expression_key="surprised")

    monkeypatch.setattr(
        character_chat_sessions,
        "resolve_character_visual_identity",
        fake_resolver,
        raising=False,
    )

    persist_response = test_client.post(
        f"/api/v1/chats/{chat_id}/completions/persist",
        json={
            "assistant_content": "streamed response",
            "mood_label": "surprised",
            "mood_confidence": 0.61,
        },
        headers=auth_headers,
    )

    assert persist_response.status_code == 200
    assistant_message_id = persist_response.json()["assistant_message_id"]
    assert calls
    assert calls[0]["actor_id"] == character_id
    assert calls[0]["mood_label"] == "surprised"

    message_response = test_client.get(
        f"/api/v1/messages/{assistant_message_id}",
        params={"include_metadata": "true"},
        headers=auth_headers,
    )
    assert message_response.status_code == 200
    metadata_extra = message_response.json()["metadata_extra"]
    assert metadata_extra["mood_label"] == "surprised"
    assert metadata_extra["visual_actor_id"] == character_id
    assert metadata_extra["visual_expression_key"] == "surprised"
    assert metadata_extra["visual_asset_id"] == 88
