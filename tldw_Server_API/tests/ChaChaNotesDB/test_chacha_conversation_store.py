from datetime import datetime, timedelta, timezone
import ast
import inspect
from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import BackendType, CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.chacha.conversation_store import ConversationStore


pytestmark = pytest.mark.unit


_DELEGATED_CONVERSATION_METHODS = {
    "_ensure_conversation_settings_table",
    "upsert_conversation_settings",
    "get_conversation_settings",
    "_normalize_conversation_state",
    "_normalize_conversation_character_scope",
    "_conversation_character_scope_clause",
    "_conversation_deleted_scope_clause",
    "_normalize_scope",
    "_normalize_conversation_assistant_identity",
    "add_conversation",
    "upsert_conversation_from_sync",
    "tombstone_conversation_from_sync",
    "get_conversation_by_id",
    "get_conversation_by_source_ref",
    "conversation_title_exists",
    "get_conversations_for_character",
    "count_conversations_for_user",
    "count_conversations_for_user_by_character",
    "get_conversations_for_user",
    "get_conversations_for_user_and_character",
    "get_conversation_cluster",
    "update_conversation",
    "soft_delete_conversation",
    "restore_conversation",
    "hard_delete_conversation",
    "search_conversations_by_title",
    "_normalize_conversation_search_order",
    "_build_conversation_search_filters",
    "_conversation_deleted_text_search_clause",
    "search_conversations",
    "search_conversations_page",
}


def _class_method_names(class_obj: type[object]) -> set[str]:
    source_path = Path(inspect.getsourcefile(class_obj) or "")
    assert source_path.exists()
    tree = ast.parse(source_path.read_text())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_obj.__name__:
            return {
                item.name
                for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
    raise AssertionError(f"Class {class_obj.__name__} not found in {source_path}")


@pytest.fixture()
def db(tmp_path):
    instance = CharactersRAGDB(
        db_path=str(tmp_path / "conversation_store.sqlite"),
        client_id="conversation-store-user",
    )
    instance.add_character_card({"name": "Conversation Store Character"})
    instance.upsert_workspace("ws-store", "Conversation Store Workspace")
    return instance


def test_conversation_store_owns_delegated_methods_without_monolith_duplicates(db):
    class_method_names = _class_method_names(CharactersRAGDB)
    assert _DELEGATED_CONVERSATION_METHODS.isdisjoint(class_method_names)


def test_conversation_store_sync_helpers_upsert_and_tombstone_include_deleted(db):
    assert db.upsert_conversation_from_sync(
        conversation_id="sync-conv-1",
        title="Synced conversation",
        sync_client_id="sync-device",
        object_revision=1,
        object_hash="sha256:conv-v1",
        assistant_kind="persona",
        assistant_id="sync-assistant",
        state="active",
    )

    created = db.get_conversation_by_id("sync-conv-1")
    assert created is not None
    assert created["title"] == "Synced conversation"
    assert created["version"] == 1
    assert created["client_id"] == "sync-device"

    assert db.upsert_conversation_from_sync(
        conversation_id="sync-conv-1",
        title="Synced conversation revised",
        sync_client_id="sync-device",
        object_revision=2,
        object_hash="sha256:conv-v2",
        assistant_kind="persona",
        assistant_id="sync-assistant",
        state="archived",
    )
    updated = db.get_conversation_by_id("sync-conv-1")
    assert updated is not None
    assert updated["title"] == "Synced conversation revised"
    assert updated["state"] == "resolved"
    assert updated["version"] == 2

    assert db.tombstone_conversation_from_sync(
        conversation_id="sync-conv-1",
        sync_client_id="sync-device",
        object_revision=3,
        object_hash="sha256:conv-delete",
    )

    assert db.get_conversation_by_id("sync-conv-1") is None
    deleted = db.get_conversation_by_id("sync-conv-1", include_deleted=True)
    assert deleted is not None
    assert bool(deleted["deleted"]) is True
    assert deleted["version"] == 3


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


def test_conversation_store_get_conversation_by_source_ref_scopes_client_and_deleted(db):
    conversation_id = db.add_conversation(
        {
            "character_id": 1,
            "title": "OpenWebUI import",
            "source": "openwebui",
            "external_ref": "chat-1",
            "client_id": db.client_id,
        }
    )
    other_id = db.add_conversation(
        {
            "character_id": 1,
            "title": "Other client import",
            "source": "openwebui",
            "external_ref": "chat-1",
            "client_id": "other-client",
        }
    )

    found = db.get_conversation_by_source_ref(
        "openwebui",
        "chat-1",
        client_id=db.client_id,
    )

    assert found is not None
    assert found["id"] == conversation_id
    assert db.get_conversation_by_source_ref(
        "openwebui",
        "chat-1",
        client_id="missing-client",
    ) is None

    created = db.get_conversation_by_id(conversation_id)
    assert created is not None
    assert db.soft_delete_conversation(conversation_id, expected_version=created["version"]) is True

    assert db.get_conversation_by_source_ref(
        "openwebui",
        "chat-1",
        client_id=db.client_id,
    ) is None
    deleted = db.get_conversation_by_source_ref(
        "openwebui",
        "chat-1",
        client_id=db.client_id,
        include_deleted=True,
    )
    assert deleted is not None
    assert deleted["id"] == conversation_id

    other = db.get_conversation_by_source_ref(
        "openwebui",
        "chat-1",
        client_id="other-client",
    )
    assert other is not None
    assert other["id"] == other_id


def test_conversation_store_maps_tuple_style_result_rows():
    class TupleResult:
        def keys(self):
            return ["id", "source", "external_ref", "deleted"]

    row = ("conversation-1", "openwebui", "chat-1", False)

    assert ConversationStore._result_row_to_dict(row, TupleResult()) == {
        "id": "conversation-1",
        "source": "openwebui",
        "external_ref": "chat-1",
        "deleted": False,
    }


def test_conversation_store_source_ref_maps_tuple_style_postgres_result():
    class FakeResult:
        first = ("conversation-1", "openwebui", "chat-1", False)

        def keys(self):
            return ["id", "source", "external_ref", "deleted"]

    class FakeBackend:
        query = None
        params = None

        def execute(self, query, params):
            self.query = query
            self.params = params
            return FakeResult()

    class FakeDB:
        backend_type = BackendType.POSTGRESQL
        client_id = "postgres-client"

        def __init__(self):
            self.backend = FakeBackend()

        @staticmethod
        def _normalize_nullable_text(value):
            return str(value).strip() or None

    fake_db = FakeDB()
    store = ConversationStore(fake_db)

    assert store.get_conversation_by_source_ref("openwebui", "chat-1") == {
        "id": "conversation-1",
        "source": "openwebui",
        "external_ref": "chat-1",
        "deleted": False,
    }
    assert "source = %s" in fake_db.backend.query
    assert "external_ref = %s" in fake_db.backend.query
    assert "client_id = %s" in fake_db.backend.query
    assert fake_db.backend.params == ("openwebui", "chat-1", "postgres-client")


def test_conversation_store_title_exists_uses_postgres_placeholders():
    class FakeResult:
        first = {"id": "conversation-1"}

    class FakeBackend:
        query = None
        params = None

        def execute(self, query, params):
            self.query = query
            self.params = params
            return FakeResult()

    class FakeDB:
        backend_type = BackendType.POSTGRESQL
        client_id = "postgres-client"

        def __init__(self):
            self.backend = FakeBackend()

        @staticmethod
        def _normalize_nullable_text(value):
            return str(value).strip() or None

    fake_db = FakeDB()
    store = ConversationStore(fake_db)

    assert store.conversation_title_exists("Existing title") is True
    assert "title = %s" in fake_db.backend.query
    assert "client_id = %s" in fake_db.backend.query
    assert "deleted = FALSE" in fake_db.backend.query
    assert "?" not in fake_db.backend.query
    assert fake_db.backend.params == ("Existing title", "postgres-client")


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


def test_conversation_store_owns_paginated_search_without_monolith_alias(db, monkeypatch):
    sentinel = ([{"id": "sentinel"}], 1, 0.75)

    def fake_search(query, **kwargs):
        assert query == "alpha"
        assert kwargs["limit"] == 1
        return sentinel

    monkeypatch.setattr(db.conversation_store, "search_conversations_page", fake_search)

    assert not hasattr(CharactersRAGDB, "_search_conversations_page_impl")
    assert db.search_conversations_page("alpha", limit=1) == sentinel
