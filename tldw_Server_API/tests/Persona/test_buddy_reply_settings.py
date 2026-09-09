"""Reply preflight reads the exact attached target's effective Chat selection."""

from __future__ import annotations

import pytest

from tldw_Server_API.app.core.Character_Chat.character_conversation_factory import create_character_conversation
from tldw_Server_API.tests.Persona.test_independent_buddies import _client, _create
from tldw_Server_API.tests.Persona.test_independent_buddies import db as _db

pytestmark = pytest.mark.integration
db = _db


def _attach(client, conversation_id, *, workspace_id=None):
    buddy = _create(client)
    response = client.put(
        "/api/v1/buddies/attachment",
        json={
            "buddy_id": buddy["id"],
            "expected_version": 0,
            "scope_type": "workspace" if workspace_id else "conversation",
            "scope_id": workspace_id or conversation_id,
        },
    )
    assert response.status_code == 200, response.text
    return buddy


@pytest.mark.parametrize(
    ("settings", "expected"),
    [
        ({}, {"provider": None, "model": None}),
        (
            {"provider": " custom-openai-api ", "model": " chosen-model "},
            {"provider": "custom-openai-api", "model": "chosen-model"},
        ),
        ({"provider": "openai"}, {"provider": "openai", "model": None}),
        ({"provider": 42, "model": "  "}, {"provider": None, "model": None}),
    ],
)
def test_reply_settings_projects_only_usable_selection_without_writes(db, settings, expected):
    conversation_id = db.add_conversation({"title": "Chosen", "client_id": "1"})
    db.upsert_conversation_settings(conversation_id, {**settings, "authorNote": "Private note"})
    before = db.get_conversation_settings(conversation_id)
    with _client(db) as client:
        _attach(client, conversation_id)
        response = client.get(f"/api/v1/buddies/conversation-targets/{conversation_id}/reply-settings")
    assert response.status_code == 200, response.text
    assert response.json() == expected
    assert db.get_conversation_settings(conversation_id) == before
    assert db.get_messages_for_conversation(conversation_id) == []


def test_reply_settings_uses_authoritative_roleplay_completion_before_raw_settings(db):
    character_id = db.add_character_card({"name": "Resume character", "first_message": "Hello"})
    conversation_id = create_character_conversation(
        db,
        conversation_data={"character_id": character_id, "title": "Resumable", "client_id": "1"},
        provider="local-llm",
        model="resume-model",
        sampling={"temperature": 0.7, "top_p": 1.0, "repetition_penalty": 1.0, "stop": []},
    )
    assert db.get_roleplay_resume_state(conversation_id)["resume_eligible"] is True
    stored = db.get_conversation_settings(conversation_id)["settings"]
    db.upsert_conversation_settings(conversation_id, {**stored, "provider": "openai", "model": "stale-model"})
    with _client(db) as client:
        _attach(client, conversation_id)
        response = client.get(f"/api/v1/buddies/conversation-targets/{conversation_id}/reply-settings")
    assert response.status_code == 200, response.text
    assert response.json() == {"provider": "local-llm", "model": "resume-model"}


@pytest.mark.parametrize("change", ["outside", "foreign", "detach", "deleted_workspace", "deleted_buddy"])
def test_reply_settings_rechecks_current_attachment_and_target_access(db, change):
    db.upsert_workspace("selected-workspace", "Research")
    conversation_id = db.add_conversation(
        {"title": "Attached", "client_id": "1", "scope_type": "workspace", "workspace_id": "selected-workspace"}
    )
    with _client(db) as client:
        buddy = _attach(client, conversation_id, workspace_id="selected-workspace")
        if change in {"outside", "foreign"}:
            conversation_id = db.add_conversation({"title": change, "client_id": "2" if change == "foreign" else "1"})
        elif change == "detach":
            assert client.delete("/api/v1/buddies/attachment?expected_version=1").status_code == 200
        elif change == "deleted_workspace":
            with db.transaction() as conn:
                conn.execute("UPDATE workspaces SET deleted = TRUE WHERE id = ?", ("selected-workspace",))
        else:
            assert client.delete(f"/api/v1/buddies/{buddy['id']}?expected_version=1").status_code == 204
        response = client.get(f"/api/v1/buddies/conversation-targets/{conversation_id}/reply-settings")
        with _client(db, user_id=2) as foreign_client:
            foreign_response = foreign_client.get(
                f"/api/v1/buddies/conversation-targets/{conversation_id}/reply-settings"
            )
    assert response.status_code == 404, response.text
    assert foreign_response.status_code == 404, foreign_response.text
