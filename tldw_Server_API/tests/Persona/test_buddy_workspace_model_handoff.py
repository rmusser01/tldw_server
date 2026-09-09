"""Ordinary workspace Chat retains its explicit model for a later Buddy reply."""

from __future__ import annotations

import pytest

from tldw_Server_API.app.api.v1.endpoints import chat as chat_endpoint
from tldw_Server_API.tests.Chat.integration.test_persona_backed_chat_conversations import (
    persona_chat_client as _persona_chat_client,
)
from tldw_Server_API.tests.Chat.integration.test_persona_backed_chat_conversations import (
    persona_chat_db as _persona_chat_db,
)
from tldw_Server_API.tests.Persona.test_buddy_turns import _attach, _send, _terminal

pytestmark = pytest.mark.integration
persona_chat_client = _persona_chat_client
persona_chat_db = _persona_chat_db


@pytest.mark.parametrize("existing_settings", [None, {"authorNote": "Keep this note", "pinnedMessageIds": ["pin-1"]}])
def test_workspace_chat_selection_reaches_buddy_without_manual_override(
    persona_chat_client, persona_chat_db, monkeypatch, existing_settings
):
    monkeypatch.setenv("CHAT_FORCE_MOCK", "1")
    client, headers, provider = persona_chat_client
    db = persona_chat_db
    db.upsert_workspace("handoff-workspace", "Research")
    created = client.post(
        "/api/v1/chats/",
        headers=headers,
        json={"title": "New research", "scope_type": "workspace", "workspace_id": "handoff-workspace"},
    )
    assert created.status_code == 201, created.text
    conversation_id = created.json()["id"]
    if existing_settings:
        assert db.upsert_conversation_settings(conversation_id, existing_settings)

    generated = client.post(
        "/api/v1/chat/completions",
        headers=headers,
        json={
            "conversation_id": conversation_id,
            "api_provider": "custom-openai-api",
            "model": "buddy-qualification",
            "messages": [{"role": "user", "content": "Start my research"}],
            "stream": False,
            "save_to_db": True,
        },
    )
    assert generated.status_code == 200, generated.text
    settings = client.get(
        f"/api/v1/chats/{conversation_id}/settings",
        headers=headers,
        params={"scope_type": "workspace", "workspace_id": "handoff-workspace"},
    )
    assert settings.status_code == 200, settings.text
    assert settings.json()["settings"] == {
        **(existing_settings or {}),
        "provider": "custom-openai-api",
        "model": "buddy-qualification",
    }
    conversation = db.get_conversation_by_id(conversation_id)
    assert conversation["assistant_id"] is None
    assert conversation["character_id"] is None
    assert conversation["workspace_id"] == "handoff-workspace"

    _attach(client, headers, conversation_id, workspace_id="handoff-workspace")
    reply = _send(client, headers, conversation_id, provider=None, model=None)
    assert reply.status_code == 202, reply.text
    assert _terminal(client, headers, reply.json()["id"])["status"] == "completed"
    assert provider.call_args.kwargs["api_endpoint"] == "custom-openai-api"
    assert provider.call_args.kwargs["model"] == "buddy-qualification"

    overridden = _send(client, headers, conversation_id, provider="openai", model="gpt-4")
    assert overridden.status_code == 202, overridden.text
    assert _terminal(client, headers, overridden.json()["id"])["status"] == "completed"
    assert db.get_conversation_settings(conversation_id)["settings"] == settings.json()["settings"]


@pytest.mark.parametrize("omitted", ["save_to_db", "api_provider", "model"])
def test_workspace_chat_does_not_save_implicit_or_ephemeral_selection(
    persona_chat_client, persona_chat_db, monkeypatch, omitted
):
    monkeypatch.setenv("CHAT_FORCE_MOCK", "1")
    monkeypatch.setattr(chat_endpoint, "_get_default_provider", lambda: "openai")
    monkeypatch.setattr(chat_endpoint, "_get_default_model_for_provider_name", lambda _provider: "gpt-4")
    client, headers, _ = persona_chat_client
    db = persona_chat_db
    db.upsert_workspace("ephemeral-workspace", "Research")
    conversation_id = db.add_conversation(
        {"title": "Ephemeral", "client_id": "1", "scope_type": "workspace", "workspace_id": "ephemeral-workspace"}
    )
    previous_settings = {"provider": "custom-openai-api", "model": "previous-model", "authorNote": "Keep"}
    assert db.upsert_conversation_settings(conversation_id, previous_settings)
    payload = {
        "conversation_id": conversation_id,
        "api_provider": "openai",
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Do not change my selection"}],
        "stream": False,
        "save_to_db": True,
    }
    if omitted == "save_to_db":
        payload["save_to_db"] = False
    else:
        payload.pop(omitted)
    generated = client.post("/api/v1/chat/completions", headers=headers, json=payload)
    assert generated.status_code == 200, generated.text
    assert db.get_conversation_settings(conversation_id)["settings"] == previous_settings


def test_workspace_model_handoff_rejects_a_foreign_conversation(persona_chat_client, persona_chat_db, monkeypatch):
    monkeypatch.setenv("CHAT_FORCE_MOCK", "1")
    client, headers, provider = persona_chat_client
    db = persona_chat_db
    db.upsert_workspace("foreign-workspace", "Other research")
    conversation_id = db.add_conversation(
        {"title": "Foreign", "client_id": "2", "scope_type": "workspace", "workspace_id": "foreign-workspace"}
    )
    previous_settings = {"provider": "custom-openai-api", "model": "private-model"}
    assert db.upsert_conversation_settings(conversation_id, previous_settings)
    generated = client.post(
        "/api/v1/chat/completions",
        headers=headers,
        json={
            "conversation_id": conversation_id,
            "api_provider": "openai",
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Wrong owner"}],
            "stream": False,
            "save_to_db": True,
        },
    )
    assert generated.status_code == 404, generated.text
    assert db.get_conversation_settings(conversation_id)["settings"] == previous_settings
    assert provider.call_count == 0
