from datetime import datetime, timedelta, timezone

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


pytestmark = pytest.mark.unit


@pytest.fixture()
def db(tmp_path):
    instance = CharactersRAGDB(
        db_path=str(tmp_path / "conversation_store.sqlite"),
        client_id="conversation-store-user",
    )
    instance.add_character_card({"name": "Conversation Store Character"})
    instance.upsert_workspace("ws-store", "Conversation Store Workspace")
    return instance


def test_conversation_store_roundtrip_preserves_scope_and_settings(db):
    conversation_id = db.add_conversation(
        {
            "character_id": 1,
            "title": "Scoped conversation",
            "scope_type": "workspace",
            "workspace_id": "ws-store",
        }
    )

    assert conversation_id is not None

    created = db.get_conversation_by_id(conversation_id)
    assert created is not None
    assert created["scope_type"] == "workspace"
    assert created["workspace_id"] == "ws-store"
    assert created["state"] == "in-progress"

    assert db.upsert_conversation_settings(
        conversation_id,
        {"temperature": 0.2, "memory": {"enabled": True}},
    ) is True

    settings_row = db.get_conversation_settings(conversation_id)
    assert settings_row is not None
    assert settings_row["settings"] == {
        "temperature": 0.2,
        "memory": {"enabled": True},
    }
    assert settings_row["last_modified"] is not None

    refreshed = db.get_conversation_by_id(conversation_id)
    assert refreshed is not None
    assert refreshed["version"] == created["version"] + 1

    workspace_rows = db.search_conversations(
        None,
        client_id=db.client_id,
        scope_type="workspace",
        workspace_id="ws-store",
    )
    global_rows = db.search_conversations(
        None,
        client_id=db.client_id,
        scope_type="global",
    )

    assert [row["id"] for row in workspace_rows] == [conversation_id]
    assert global_rows == []
    assert db.count_conversations_for_user(
        db.client_id,
        scope_type="workspace",
        workspace_id="ws-store",
    ) == 1


def test_conversation_store_preserves_assistant_identity_updates(db):
    conversation_id = db.add_conversation(
        {
            "assistant_kind": "persona",
            "assistant_id": "persona-gardener",
            "persona_memory_mode": "read_only",
            "title": "Persona conversation",
            "root_id": "persona-conversation",
            "client_id": db.client_id,
        }
    )

    created = db.get_conversation_by_id(conversation_id)
    assert created is not None
    assert created["assistant_kind"] == "persona"
    assert created["assistant_id"] == "persona-gardener"
    assert created["persona_memory_mode"] == "read_only"
    assert created["character_id"] is None

    assert db.update_conversation(
        conversation_id,
        {"persona_memory_mode": "read_write"},
        expected_version=created["version"],
    ) is True

    updated = db.get_conversation_by_id(conversation_id)
    assert updated is not None
    assert updated["assistant_kind"] == "persona"
    assert updated["assistant_id"] == "persona-gardener"
    assert updated["persona_memory_mode"] == "read_write"


def test_conversation_store_title_search_prefers_newer_last_modified(db):
    older_id = db.add_conversation(
        {
            "assistant_kind": "persona",
            "assistant_id": "persona-older",
            "persona_memory_mode": "read_only",
            "title": "shared alpha title",
            "root_id": "older-conversation",
            "client_id": db.client_id,
        }
    )
    newer_id = db.add_conversation(
        {
            "assistant_kind": "persona",
            "assistant_id": "persona-newer",
            "persona_memory_mode": "read_only",
            "title": "shared alpha title",
            "root_id": "newer-conversation",
            "client_id": db.client_id,
        }
    )

    now = datetime.now(timezone.utc)
    db.execute_query(
        "UPDATE conversations SET created_at = ?, last_modified = ? WHERE id = ?",
        ((now - timedelta(days=3)).isoformat(), (now - timedelta(days=3)).isoformat(), older_id),
        commit=True,
    )
    db.execute_query(
        "UPDATE conversations SET created_at = ?, last_modified = ? WHERE id = ?",
        ((now - timedelta(minutes=10)).isoformat(), (now - timedelta(minutes=10)).isoformat(), newer_id),
        commit=True,
    )

    rows = db.search_conversations_by_title("alpha", limit=10, offset=0)

    assert [row["id"] for row in rows[:2]] == [newer_id, older_id]
