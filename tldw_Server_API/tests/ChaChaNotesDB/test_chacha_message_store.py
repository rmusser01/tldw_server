import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.chacha.message_store import MessageStore


pytestmark = pytest.mark.unit


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

    citations = store.get_conversation_citations(conversation_id)
    citations_by_id = {citation["id"] if "id" in citation else citation["chunk_id"]: citation for citation in citations}
    assert sorted(citations_by_id.keys()) == ["chunk-2", "doc-1"]
    assert citations_by_id["doc-1"]["message_ids"] == [first_message_id, second_message_id]


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
