import ast
import inspect
from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.chacha.message_store import MessageStore


pytestmark = pytest.mark.unit


_DELEGATED_MESSAGE_METHODS = {
    "add_message",
    "append_message_from_sync",
    "tombstone_message_from_sync",
    "get_messages_by_sync_stable_id",
    "_insert_message_images",
    "get_message_images",
    "get_message_conversation_id",
    "get_message_by_id",
    "get_messages_for_conversation",
    "count_root_messages_for_conversation",
    "get_root_messages_for_conversation",
    "get_messages_for_conversation_by_parent_ids",
    "has_system_message_for_conversation",
    "update_message",
    "soft_delete_message",
    "search_messages_by_content",
    "add_message_metadata",
    "get_message_metadata",
    "get_message_metadata_map",
    "set_message_metadata_extra",
    "set_message_rag_context",
    "get_message_rag_context",
    "get_messages_with_rag_context",
    "count_messages_for_conversation",
    "count_messages_for_conversations",
    "get_latest_message_for_conversation",
    "count_messages_since",
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
        db_path=str(tmp_path / "message_store.sqlite"),
        client_id="message-store-user",
    )
    character_id = instance.add_character_card({"name": "Message Store Character"})
    conversation_id = instance.add_conversation(
        {
            "character_id": character_id,
            "title": "Message Store Conversation",
        }
    )
    return {
        "db": instance,
        "store": MessageStore(instance),
        "conversation_id": conversation_id,
    }


def test_message_store_owns_delegated_methods_without_monolith_duplicates(db, monkeypatch):
    class_method_names = _class_method_names(CharactersRAGDB)
    assert _DELEGATED_MESSAGE_METHODS.isdisjoint(class_method_names)

    captured: dict[str, object] = {}

    def _fake_add_message(msg_data):
        captured["msg_data"] = msg_data
        return "message-from-store"

    monkeypatch.setattr(db["db"].message_store, "add_message", _fake_add_message)

    assert db["db"].add_message({"conversation_id": db["conversation_id"], "sender": "user", "content": "hi"}) == "message-from-store"
    assert captured["msg_data"] == {
        "conversation_id": db["conversation_id"],
        "sender": "user",
        "content": "hi",
    }


def test_message_store_sync_append_dedupe_divergent_versions_and_tombstone(db):
    store = db["db"]
    conversation_id = db["conversation_id"]

    first = store.append_message_from_sync(
        stable_message_id="sync-msg-1",
        conversation_id=conversation_id,
        sender="user",
        content="First synced message",
        timestamp="2026-05-23T18:13:00+00:00",
        sync_client_id="sync-device",
        object_revision=1,
        payload_hash="sha256:msg-v1",
    )

    assert first["message_id"] == "sync-msg-1"
    assert first["created"] is True
    assert first["idempotent"] is False
    assert first["conflict"] is False
    assert store.get_message_by_id("sync-msg-1")["content"] == "First synced message"

    duplicate = store.append_message_from_sync(
        stable_message_id="sync-msg-1",
        conversation_id=conversation_id,
        sender="user",
        content="First synced message",
        timestamp="2026-05-23T18:13:00+00:00",
        sync_client_id="sync-device",
        object_revision=1,
        payload_hash="sha256:msg-v1",
    )

    assert duplicate["message_id"] == "sync-msg-1"
    assert duplicate["created"] is False
    assert duplicate["idempotent"] is True
    assert duplicate["conflict"] is False
    assert store.count_messages_for_conversation(conversation_id, include_deleted=True) == 1

    divergent = store.append_message_from_sync(
        stable_message_id="sync-msg-1",
        conversation_id=conversation_id,
        sender="assistant",
        content="Conflicting synced message",
        timestamp="2026-05-23T18:14:00+00:00",
        sync_client_id="sync-device",
        object_revision=2,
        payload_hash="sha256:msg-v2",
        projection_message_id="sync-msg-1-conflict",
    )

    assert divergent["message_id"] == "sync-msg-1-conflict"
    assert divergent["created"] is True
    assert divergent["idempotent"] is False
    assert divergent["conflict"] is True
    versions = store.get_messages_by_sync_stable_id("sync-msg-1", include_deleted=True)
    assert [item["id"] for item in versions] == ["sync-msg-1", "sync-msg-1-conflict"]

    assert store.tombstone_message_from_sync(
        stable_message_id="sync-msg-1",
        sync_client_id="sync-device",
        object_revision=3,
        object_hash="sha256:msg-v1",
    )
    assert store.get_messages_by_sync_stable_id("sync-msg-1") == []
    assert store.get_message_by_id("sync-msg-1") is None
    deleted = store.get_message_by_id("sync-msg-1", include_deleted=True)
    assert deleted is not None
    assert bool(deleted["deleted"]) is True
    conflict = store.get_message_by_id("sync-msg-1-conflict", include_deleted=True)
    assert conflict is not None
    assert bool(conflict["deleted"]) is True
    assert store.get_conversation_by_id(conversation_id) is not None


def test_message_store_add_and_fetch_roundtrip(db):
    store = db["store"]
    conversation_id = db["conversation_id"]

    message_id = store.add_message(
        {
            "conversation_id": conversation_id,
            "sender": "assistant",
            "content": "Stored via MessageStore",
            "images": [
                {"data": b"first-image", "mime": "image/png"},
                {"data": b"second-image", "mime": "image/jpeg"},
            ],
        }
    )

    assert message_id is not None

    stored = store.get_message_by_id(message_id)
    assert stored is not None
    assert stored["content"] == "Stored via MessageStore"
    assert stored["image_mime_type"] == "image/png"
    assert [image["image_mime_type"] for image in stored["images"]] == [
        "image/png",
        "image/jpeg",
    ]
    assert [image["image_data"] for image in stored["images"]] == [
        b"first-image",
        b"second-image",
    ]
    assert store.get_message_conversation_id(message_id) == conversation_id


def test_message_store_adds_image_only_message(db):
    store = db["store"]
    conversation_id = db["conversation_id"]

    message_id = store.add_message(
        {
            "conversation_id": conversation_id,
            "sender": "user",
            "image_data": b"image-only",
            "image_mime_type": "image/png",
        }
    )

    stored = store.get_message_by_id(message_id)
    assert stored is not None
    assert stored["content"] == ""
    assert stored["image_data"] == b"image-only"


def test_message_store_metadata_and_citations_roundtrip(db):
    store = db["store"]
    conversation_id = db["conversation_id"]

    first_message_id = store.add_message(
        {
            "conversation_id": conversation_id,
            "sender": "assistant",
            "content": "Answer with citations",
        }
    )
    second_message_id = store.add_message(
        {
            "conversation_id": conversation_id,
            "sender": "assistant",
            "content": "Answer citing the same document",
        }
    )

    assert store.add_message_metadata(
        first_message_id,
        tool_calls=[{"id": "tool-1", "type": "function"}],
        extra={
            "rag_context": {
                "retrieved_documents": [
                    {"id": "doc-1", "title": "Document One"},
                    {"chunk_id": "chunk-2", "title": "Document Two"},
                ]
            }
        },
    )
    assert store.add_message_metadata(
        second_message_id,
        extra={
            "rag_context": {
                "retrieved_documents": [
                    {"id": "doc-1", "title": "Document One"},
                ]
            }
        },
    )

    metadata = store.get_message_metadata(first_message_id)
    assert metadata is not None
    assert metadata["tool_calls"] == [{"id": "tool-1", "type": "function"}]
    assert metadata["extra"]["rag_context"]["retrieved_documents"][0]["id"] == "doc-1"

    metadata_map = store.get_message_metadata_map([first_message_id, second_message_id, "missing-id"])
    assert sorted(metadata_map.keys()) == sorted([first_message_id, second_message_id])

    # get_conversation_citations not yet extracted to MessageStore — tested via CharactersRAGDB directly


def test_message_store_rag_context_helpers_latest_and_since(db):
    store = db["store"]
    conversation_id = db["conversation_id"]

    first_message_id = store.add_message(
        {
            "conversation_id": conversation_id,
            "sender": "user",
            "content": "First message",
            "timestamp": "2024-01-01T00:00:00Z",
        }
    )
    second_message_id = store.add_message(
        {
            "conversation_id": conversation_id,
            "sender": "assistant",
            "content": "Second message",
            "timestamp": "2024-01-01T00:00:01Z",
        }
    )
    third_message_id = store.add_message(
        {
            "conversation_id": conversation_id,
            "sender": "assistant",
            "content": "Third message",
            "timestamp": "2024-01-01T00:00:02Z",
        }
    )

    assert store.set_message_metadata_extra(
        first_message_id,
        {"tool_results": {"call-1": {"status": "ok"}}, "trace_id": "trace-1"},
    )
    assert store.set_message_metadata_extra(
        first_message_id,
        {"tool_results": {"call-2": {"status": "later"}}, "note": "merged"},
    )

    merged_metadata = store.get_message_metadata(first_message_id)
    assert merged_metadata is not None
    assert merged_metadata["extra"]["tool_results"]["call-1"]["status"] == "ok"
    assert merged_metadata["extra"]["tool_results"]["call-2"]["status"] == "later"
    assert merged_metadata["extra"]["trace_id"] == "trace-1"
    assert merged_metadata["extra"]["note"] == "merged"

    rag_context = {
        "search_query": "galaxy",
        "retrieved_documents": [{"id": "doc-1", "title": "Galaxy Notes"}],
    }
    assert store.set_message_rag_context(second_message_id, rag_context)
    assert store.get_message_rag_context(second_message_id) == rag_context

    with_rag_context = store.get_messages_with_rag_context(conversation_id, limit=10, offset=0)
    second_message = next(item for item in with_rag_context if item["id"] == second_message_id)
    assert second_message["rag_context"]["retrieved_documents"][0]["id"] == "doc-1"

    latest_message = store.get_latest_message_for_conversation(conversation_id)
    assert latest_message is not None
    assert latest_message["id"] == third_message_id
    assert store.count_messages_since(conversation_id, first_message_id) == 2


def test_message_store_counts_and_soft_delete_roundtrip(db):
    store = db["store"]
    conversation_id = db["conversation_id"]

    root_message_id = store.add_message(
        {
            "conversation_id": conversation_id,
            "sender": "user",
            "content": "Root",
        }
    )
    child_message_id = store.add_message(
        {
            "conversation_id": conversation_id,
            "parent_message_id": root_message_id,
            "sender": "assistant",
            "content": "Child",
        }
    )
    system_message_id = store.add_message(
        {
            "conversation_id": conversation_id,
            "sender": "system",
            "content": "System prompt",
        }
    )

    assert store.count_messages_for_conversation(conversation_id) == 3
    assert store.count_messages_for_conversations([conversation_id, "missing-conversation"]) == {
        conversation_id: 3,
        "missing-conversation": 0,
    }
    assert store.count_root_messages_for_conversation(conversation_id) == 2
    assert [row["id"] for row in store.get_root_messages_for_conversation(conversation_id, limit=10, offset=0)] == [
        root_message_id,
        system_message_id,
    ]
    assert [row["id"] for row in store.get_messages_for_conversation_by_parent_ids(conversation_id, [root_message_id])] == [
        child_message_id
    ]
    assert store.has_system_message_for_conversation(conversation_id) is True

    root_message = store.get_message_by_id(root_message_id)
    assert root_message is not None
    assert store.update_message(
        root_message_id,
        {"content": "Root updated"},
        expected_version=root_message["version"],
    ) is True

    updated_root = store.get_message_by_id(root_message_id)
    assert updated_root is not None
    assert updated_root["content"] == "Root updated"

    child_message = store.get_message_by_id(child_message_id)
    assert child_message is not None
    assert store.soft_delete_message(child_message_id, expected_version=child_message["version"]) is True
    assert store.get_message_by_id(child_message_id) is None
    assert store.count_messages_for_conversation(conversation_id) == 2
    assert store.count_messages_for_conversation(conversation_id, include_deleted=True) == 3
