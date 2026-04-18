import inspect

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.chacha.conversation_store import ConversationStore


pytestmark = pytest.mark.unit


@pytest.fixture()
def db(tmp_path):
    instance = CharactersRAGDB(
        db_path=str(tmp_path / "conversation_store.sqlite"),
        client_id="conversation-store-user",
    )
    instance.add_character_card({"name": "Conversation Store Character"})
    instance.upsert_workspace("ws-store", "Conversation Store Workspace")
    try:
        yield instance
    finally:
        instance.close_connection()


def test_conversation_store_roundtrip_preserves_scope_and_settings(db):
    store = ConversationStore(db)

    conversation_id = store.add_conversation(
        {
            "character_id": 1,
            "title": "Scoped conversation",
            "scope_type": "workspace",
            "workspace_id": "ws-store",
        }
    )

    assert conversation_id is not None

    created = store.get_conversation_by_id(conversation_id)
    assert created is not None
    assert created["scope_type"] == "workspace"
    assert created["workspace_id"] == "ws-store"
    assert created["state"] == "in-progress"

    assert store.upsert_conversation_settings(
        conversation_id,
        {"temperature": 0.2, "memory": {"enabled": True}},
    ) is True

    settings_row = store.get_conversation_settings(conversation_id)
    assert settings_row is not None
    assert settings_row["settings"] == {
        "temperature": 0.2,
        "memory": {"enabled": True},
    }
    assert settings_row["last_modified"] is not None

    refreshed = store.get_conversation_by_id(conversation_id)
    assert refreshed is not None
    assert refreshed["version"] == created["version"] + 1

    workspace_rows = store.search_conversations(
        None,
        client_id=db.client_id,
        scope_type="workspace",
        workspace_id="ws-store",
    )
    global_rows = store.search_conversations(
        None,
        client_id=db.client_id,
        scope_type="global",
    )

    assert [row["id"] for row in workspace_rows] == [conversation_id]
    assert global_rows == []
    assert store.count_conversations_for_user(
        db.client_id,
        scope_type="workspace",
        workspace_id="ws-store",
    ) == 1


def test_conversation_store_preserves_assistant_identity_updates(db):
    store = ConversationStore(db)

    conversation_id = store.add_conversation(
        {
            "assistant_kind": "persona",
            "assistant_id": "persona-gardener",
            "persona_memory_mode": "read_only",
            "title": "Persona conversation",
            "root_id": "persona-conversation",
            "client_id": db.client_id,
        }
    )

    created = store.get_conversation_by_id(conversation_id)
    assert created is not None
    assert created["assistant_kind"] == "persona"
    assert created["assistant_id"] == "persona-gardener"
    assert created["persona_memory_mode"] == "read_only"
    assert created["character_id"] is None

    assert store.update_conversation(
        conversation_id,
        {"persona_memory_mode": "read_write"},
        expected_version=created["version"],
    ) is True

    updated = store.get_conversation_by_id(conversation_id)
    assert updated is not None
    assert updated["assistant_kind"] == "persona"
    assert updated["assistant_id"] == "persona-gardener"
    assert updated["persona_memory_mode"] == "read_write"


def test_conversation_facade_preserves_search_signature():
    signature = inspect.signature(CharactersRAGDB.search_conversations_page)

    assert all(
        parameter.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        for parameter in signature.parameters.values()
    )
    assert "query" in signature.parameters
    assert "order_by" in signature.parameters
    assert "limit" in signature.parameters
    assert "offset" in signature.parameters
    assert "scope_type" in signature.parameters
    assert "workspace_id" in signature.parameters
