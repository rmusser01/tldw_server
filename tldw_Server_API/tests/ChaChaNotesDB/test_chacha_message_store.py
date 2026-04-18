import inspect

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.chacha.message_store import MessageStore


pytestmark = pytest.mark.unit


@pytest.fixture()
def character_id(db_instance):
    character_id = db_instance.add_character_card({"name": "Message Store Character"})
    assert character_id is not None
    return character_id


def test_message_store_add_and_fetch_roundtrip(db_instance, character_id):
    conversation_id = db_instance.add_conversation({"character_id": character_id, "title": "msg store"})
    store = MessageStore(db_instance)

    message_id = store.add_message(
        {
            "conversation_id": conversation_id,
            "sender": "assistant",
            "content": "hello from store",
        }
    )

    assert message_id is not None
    fetched = store.get_message_by_id(message_id)
    assert fetched is not None
    assert fetched["content"] == "hello from store"
    assert fetched["conversation_id"] == conversation_id


def test_message_store_preserves_tree_counts_and_metadata(db_instance, character_id):
    conversation_id = db_instance.add_conversation({"character_id": character_id, "title": "msg tree"})
    store = MessageStore(db_instance)

    root_id = store.add_message(
        {
            "conversation_id": conversation_id,
            "sender": "system",
            "content": "root",
            "timestamp": "2024-01-01T00:00:00Z",
        }
    )
    child_id = store.add_message(
        {
            "conversation_id": conversation_id,
            "parent_message_id": root_id,
            "sender": "assistant",
            "content": "child",
            "images": [{"data": b"png-bytes", "mime": "image/png"}],
            "timestamp": "2024-01-01T00:00:01Z",
        }
    )

    assert root_id is not None
    assert child_id is not None
    assert store.add_message_metadata(
        child_id,
        tool_calls=[{"id": "tool-1"}],
        extra={"source": "message-store"},
    ) is True

    child_message = store.get_message_by_id(child_id)
    metadata_map = store.get_message_metadata_map([root_id, child_id])
    root_rows = store.get_root_messages_for_conversation(conversation_id, limit=10, offset=0)
    child_rows = store.get_messages_for_conversation_by_parent_ids(conversation_id, [root_id])

    assert child_message is not None
    assert child_message["images"][0]["image_mime_type"] == "image/png"
    assert metadata_map[child_id]["extra"] == {"source": "message-store"}
    assert store.count_messages_for_conversation(conversation_id) == 2
    assert store.count_root_messages_for_conversation(conversation_id) == 1
    assert store.has_system_message_for_conversation(conversation_id) is True
    assert [row["id"] for row in root_rows] == [root_id]
    assert [row["id"] for row in child_rows] == [child_id]


def test_message_facade_preserves_get_messages_signature():
    signature = inspect.signature(CharactersRAGDB.get_messages_for_conversation)

    assert all(
        parameter.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        for parameter in signature.parameters.values()
    )
    assert "conversation_id" in signature.parameters
    assert "limit" in signature.parameters
    assert "offset" in signature.parameters
    assert "order_by_timestamp" in signature.parameters
    assert "include_deleted" in signature.parameters
