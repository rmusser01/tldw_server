import ast
import inspect
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from loguru import logger

from tldw_Server_API.app.core.DB_Management.chacha.message_store import MessageStore
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    BackendType,
    CharactersRAGDB,
    CharactersRAGDBError,
    InputError,
)

pytestmark = pytest.mark.unit


_DELEGATED_MESSAGE_METHODS = {
    "add_message",
    "append_message_from_sync",
    "tombstone_message_from_sync",
    "tombstone_unsynced_message_from_sync",
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
            return {item.name for item in node.body if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))}
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

    assert (
        db["db"].add_message({"conversation_id": db["conversation_id"], "sender": "user", "content": "hi"})
        == "message-from-store"
    )
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


def _tombstone_unsynced(store, stable_message_id, object_hash="sha256:tombstone"):
    return store.tombstone_unsynced_message_from_sync(
        stable_message_id=stable_message_id,
        sync_client_id="message-store-user",
        object_revision=1,
        object_hash=object_hash,
    )


def test_unsynced_tombstone_of_a_message_that_does_not_exist_changes_nothing(db):
    store = db["db"]
    conversation_id = db["conversation_id"]
    store.add_message({"id": "neighbour", "conversation_id": conversation_id, "sender": "user", "content": "stays"})

    assert _tombstone_unsynced(store, "never-existed") == "absent"

    assert store.get_message_by_id("never-existed", include_deleted=True) is None
    assert store.count_messages_for_conversation(conversation_id) == 1


def test_unsynced_tombstone_deletes_a_copy_kept_outside_sync_and_replays(db):
    store = db["db"]
    conversation_id = db["conversation_id"]
    store.add_message({"id": "outside-sync", "conversation_id": conversation_id, "sender": "user", "content": "copy"})
    store.add_message({"id": "neighbour", "conversation_id": conversation_id, "sender": "user", "content": "stays"})

    assert _tombstone_unsynced(store, "outside-sync") == "deleted"

    assert store.get_message_by_id("outside-sync") is None
    assert bool(store.get_message_by_id("outside-sync", include_deleted=True)["deleted"]) is True
    assert store.get_message_by_id("neighbour") is not None
    # The same tombstone again recognises its own work; a different one does not claim it.
    assert _tombstone_unsynced(store, "outside-sync") == "deleted"
    assert _tombstone_unsynced(store, "outside-sync", object_hash="sha256:another-tombstone") == "foreign"


def test_unsynced_tombstone_leaves_a_row_with_its_own_sync_history(db):
    store = db["db"]
    store.append_message_from_sync(
        stable_message_id="projected",
        conversation_id=db["conversation_id"],
        sender="user",
        content="Projected by Sync",
        timestamp="2026-05-23T18:13:00+00:00",
        sync_client_id="message-store-user",
        object_revision=1,
        payload_hash="sha256:projected",
    )

    assert _tombstone_unsynced(store, "projected") == "foreign"

    assert store.get_message_by_id("projected") is not None


def test_unsynced_tombstone_only_reaches_chats_the_projection_owner_holds(db):
    store = db["db"]
    with store.transaction() as conn:
        conn.execute(
            "INSERT INTO conversations (id, root_id, title, client_id) VALUES (?, ?, ?, ?)",
            ("their-chat", "their-chat", "Theirs", "another-owner"),
        )
        conn.execute(
            "INSERT INTO messages (id, conversation_id, sender, content, client_id) VALUES (?, ?, ?, ?, ?)",
            ("their-message", "their-chat", "user", "not yours", "another-owner"),
        )

    assert _tombstone_unsynced(store, "their-message") == "absent"

    assert bool(store.get_message_by_id("their-message", include_deleted=True)["deleted"]) is False
    assert [item["id"] for item in store.get_messages_by_sync_stable_id("their-message")] == ["their-message"]
    assert store.get_messages_by_sync_stable_id("their-message", owner_client_id="message-store-user") == []


def test_unsynced_tombstone_requires_a_message_id(db):
    with pytest.raises(InputError):
        _tombstone_unsynced(db["db"], "  ")


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


def test_message_store_orders_equal_timestamps_by_last_modified(db):
    store = db["store"]
    raw_db = db["db"]
    conversation_id = db["conversation_id"]
    same_timestamp = "2026-01-01T00:00:00Z"

    assistant_id = store.add_message(
        {
            "id": "tie-assistant",
            "conversation_id": conversation_id,
            "sender": "assistant",
            "content": "Assistant response",
            "timestamp": same_timestamp,
        }
    )
    user_id = store.add_message(
        {
            "id": "tie-user",
            "conversation_id": conversation_id,
            "sender": "user",
            "content": "User request",
            "timestamp": same_timestamp,
        }
    )
    system_id = store.add_message(
        {
            "id": "tie-system",
            "conversation_id": conversation_id,
            "sender": "system",
            "content": "System prompt",
            "timestamp": same_timestamp,
        }
    )

    for message_id, last_modified in [
        (system_id, "2026-01-01T00:00:00.000Z"),
        (user_id, "2026-01-01T00:00:00.100Z"),
        (assistant_id, "2026-01-01T00:00:00.200Z"),
    ]:
        raw_db.execute_query(
            "UPDATE messages SET last_modified = ? WHERE id = ?",
            (last_modified, message_id),
            commit=True,
        )

    ascending = store.get_messages_for_conversation(
        conversation_id,
        limit=10,
        offset=0,
        order_by_timestamp="ASC",
    )
    assert [row["id"] for row in ascending] == [system_id, user_id, assistant_id]

    descending = store.get_messages_for_conversation(
        conversation_id,
        limit=10,
        offset=0,
        order_by_timestamp="DESC",
    )
    assert [row["id"] for row in descending] == [assistant_id, user_id, system_id]


def test_message_store_preserves_append_order_when_insert_clock_ties(db, monkeypatch):
    store = db["store"]
    raw_db = db["db"]
    conversation_id = db["conversation_id"]
    same_timestamp = "2026-01-01T00:00:00.000Z"

    monkeypatch.setattr(raw_db, "_get_current_utc_timestamp_iso", lambda: same_timestamp)

    first_id = store.add_message(
        {
            "id": "order-b",
            "conversation_id": conversation_id,
            "sender": "user",
            "content": "First",
        }
    )
    second_id = store.add_message(
        {
            "id": "order-a",
            "conversation_id": conversation_id,
            "sender": "assistant",
            "content": "Second",
        }
    )
    third_id = store.add_message(
        {
            "id": "order-c",
            "conversation_id": conversation_id,
            "sender": "user",
            "content": "Third",
        }
    )

    messages = store.get_messages_for_conversation(conversation_id, limit=10, offset=0)
    assert [row["id"] for row in messages] == [first_id, second_id, third_id]
    assert [row["last_modified"] for row in messages] == [
        "2026-01-01T00:00:00.000Z",
        "2026-01-01T00:00:00.001Z",
        "2026-01-01T00:00:00.002Z",
    ]


def test_source_message_projection_is_bounded_ordered_and_never_loads_images(db, monkeypatch):
    store = db["store"]
    raw_db = db["db"]
    conversation_id = db["conversation_id"]
    timestamp = "2026-01-01T00:00:00.000Z"

    second_id = store.add_message(
        {
            "id": "source-message-b",
            "conversation_id": conversation_id,
            "sender": "assistant",
            "content": "second-content",
            "timestamp": timestamp,
            "image_data": b"secret-image",
            "image_mime_type": "image/png",
        }
    )
    first_id = store.add_message(
        {
            "id": "source-message-a",
            "conversation_id": conversation_id,
            "sender": "user",
            "content": "first-content",
            "timestamp": timestamp,
        }
    )
    image_only_id = store.add_message(
        {
            "id": "source-message-image-only",
            "conversation_id": conversation_id,
            "sender": "user",
            "image_data": b"image-only",
            "image_mime_type": "image/png",
            "timestamp": timestamp,
        }
    )
    assert second_id and first_id and image_only_id

    raw_db.execute_query(
        "UPDATE messages SET last_modified = ? WHERE id = ?",
        ("2026-01-01T00:00:00.100Z", second_id),
        commit=True,
    )
    raw_db.execute_query(
        "UPDATE messages SET last_modified = ? WHERE id = ?",
        ("2026-01-01T00:00:00.000Z", first_id),
        commit=True,
    )

    queries: list[str] = []
    original_execute = raw_db.execute_query

    def recording_execute(query, params=None, **kwargs):
        queries.append(str(query))
        return original_execute(query, params, **kwargs)

    monkeypatch.setattr(raw_db, "execute_query", recording_execute)
    monkeypatch.setattr(
        store,
        "get_message_images",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("source projection must never load message images")
        ),
    )

    projection = store.get_source_message_projection(
        conversation_id,
        max_chars=30,
    )

    assert [row["source_text"] for row in projection["rows"]] == [
        "user: first-content",
        "assistant: ",
    ]
    assert sum(len(row["source_text"]) for row in projection["rows"]) + 1 == 31
    assert projection["conversation_exists"] is True
    assert projection["invalid"] is False
    assert projection["truncated"] is True
    normalized_sql = " ".join(queries).lower()
    assert "select *" not in normalized_sql
    assert "image_data" not in normalized_sql
    assert "message_images" not in normalized_sql
    assert "substr" in normalized_sql
    assert "limit" in normalized_sql
    assert "offset" not in normalized_sql
    assert "row_number()" in normalized_sql
    assert normalized_sql.count("substr(") == 1
    assert len(queries) == 1


def test_source_message_projection_uses_one_statement_snapshot(db, monkeypatch):
    store = db["store"]
    raw_db = db["db"]
    conversation_id = db["conversation_id"]

    first_id = store.add_message(
        {
            "id": "snapshot-first",
            "conversation_id": conversation_id,
            "sender": "user",
            "content": "first",
            "timestamp": "2026-01-01T00:00:00.000Z",
        }
    )
    second_id = store.add_message(
        {
            "id": "snapshot-second",
            "conversation_id": conversation_id,
            "sender": "assistant",
            "content": "second",
            "timestamp": "2026-01-01T00:00:01.000Z",
        }
    )
    assert first_id and second_id

    queries: list[str] = []
    original_execute = raw_db.execute_query

    def recording_execute(query, params=None, **kwargs):
        queries.append(str(query))
        return original_execute(query, params, **kwargs)

    monkeypatch.setattr(raw_db, "execute_query", recording_execute)

    projection = store.get_source_message_projection(conversation_id, max_chars=100)

    assert [row["source_text"] for row in projection["rows"]] == [
        "user: first",
        "assistant: second",
    ]
    assert projection == {
        "rows": [
            {"source_text": "user: first"},
            {"source_text": "assistant: second"},
        ],
        "conversation_exists": True,
        "invalid": False,
        "truncated": False,
    }
    assert len(queries) == 1
    assert "after_cursor" not in inspect.signature(store.get_source_message_projection).parameters
    assert "high_water" not in inspect.signature(store.get_source_message_projection).parameters


@pytest.mark.parametrize("concurrent_write", ["backdated_insert", "content_update"])
def test_source_message_projection_snapshot_excludes_concurrent_writes(
    db,
    concurrent_write,
):
    store = db["store"]
    raw_db = db["db"]
    conversation_id = db["conversation_id"]
    first_id = store.add_message(
        {
            "id": "race-first",
            "conversation_id": conversation_id,
            "sender": "user",
            "content": "first",
            "timestamp": "2026-01-01T00:00:00.000Z",
        }
    )
    second_id = store.add_message(
        {
            "id": "race-second",
            "conversation_id": conversation_id,
            "sender": "assistant",
            "content": "second",
            "timestamp": "2026-01-01T00:00:02.000Z",
        }
    )
    hidden_id = store.add_message(
        {
            "id": "race-backdated",
            "conversation_id": conversation_id,
            "sender": "user",
            "content": "must not enter snapshot",
            "timestamp": "2026-01-01T00:00:01.000Z",
        }
    )
    assert first_id and second_id and hidden_id
    raw_db.execute_query(
        "UPDATE messages SET deleted = 1 WHERE id = ?",
        (hidden_id,),
        commit=True,
    )

    reader_connection = raw_db.get_connection()
    reader_connection.execute("PRAGMA journal_mode = WAL")
    projection_started = threading.Event()
    writer_finished = threading.Event()
    first_substr = True

    def blocking_substr(value, start, length):
        nonlocal first_substr
        if first_substr:
            first_substr = False
            projection_started.set()
            assert writer_finished.wait(timeout=5)
        offset = max(0, int(start) - 1)
        return value[offset : offset + int(length)]

    reader_connection.create_function("SUBSTR", 3, blocking_substr)

    def write_concurrently():
        assert projection_started.wait(timeout=5)
        connection = sqlite3.connect(raw_db.db_path_str, timeout=5)
        try:
            if concurrent_write == "backdated_insert":
                connection.execute(
                    "UPDATE messages SET deleted = 0 WHERE id = ?",
                    (hidden_id,),
                )
            else:
                connection.execute(
                    """
                    UPDATE messages
                    SET content = ?, last_modified = ?
                    WHERE id = ?
                    """,
                    (
                        "changed after snapshot",
                        "2026-01-01T00:00:03.000Z",
                        first_id,
                    ),
                )
            connection.commit()
        finally:
            connection.close()
            writer_finished.set()

    with ThreadPoolExecutor(max_workers=1) as executor:
        writer = executor.submit(write_concurrently)
        try:
            projection = store.get_source_message_projection(conversation_id, max_chars=100)
        finally:
            writer.result(timeout=5)
    assert [row["source_text"] for row in projection["rows"]] == [
        "user: first",
        "assistant: second",
    ]

    written = store.get_message_by_id(hidden_id if concurrent_write == "backdated_insert" else first_id)
    assert written is not None
    assert written["content"] == (
        "must not enter snapshot" if concurrent_write == "backdated_insert" else "changed after snapshot"
    )


def test_source_message_projection_hides_deleted_conversation_and_messages(db):
    store = db["store"]
    raw_db = db["db"]
    conversation_id = db["conversation_id"]
    message_id = store.add_message(
        {
            "conversation_id": conversation_id,
            "sender": "user",
            "content": "visible",
        }
    )
    assert message_id

    raw_db.execute_query(
        "UPDATE messages SET deleted = 1 WHERE id = ?",
        (message_id,),
        commit=True,
    )
    assert (
        store.get_source_message_projection(
            conversation_id,
            max_chars=20,
        )["rows"]
        == []
    )

    live_message_id = store.add_message(
        {
            "conversation_id": conversation_id,
            "sender": "assistant",
            "content": "hidden with conversation",
        }
    )
    assert live_message_id
    raw_db.execute_query(
        "UPDATE conversations SET deleted = 1 WHERE id = ?",
        (conversation_id,),
        commit=True,
    )
    assert (
        store.get_source_message_projection(
            conversation_id,
            max_chars=20,
        )["conversation_exists"]
        is False
    )


@pytest.mark.parametrize("max_chars", [True, 0, -1, "10"])
def test_source_message_projection_rejects_invalid_character_budget(db, max_chars):
    with pytest.raises(InputError):
        db["store"].get_source_message_projection(
            db["conversation_id"],
            max_chars=max_chars,
        )


def test_source_message_projection_marks_nul_content_invalid(db):
    store = db["store"]
    raw_db = db["db"]
    conversation_id = db["conversation_id"]
    message_id = store.add_message(
        {
            "conversation_id": conversation_id,
            "sender": "user",
            "content": "valid",
        }
    )
    assert message_id
    raw_db.execute_query(
        "UPDATE messages SET content = ? WHERE id = ?",
        ("prefix\0" + ("secret" * 1000), message_id),
        commit=True,
    )

    projection = store.get_source_message_projection(conversation_id, max_chars=20)

    assert projection["invalid"] is True


def test_postgres_source_message_projection_is_one_owner_scoped_statement():
    captured: dict[str, object] = {}

    class _Cursor:
        @staticmethod
        def fetchall():
            return []

    class _PostgresDb:
        backend_type = BackendType.POSTGRESQL

        @staticmethod
        def execute_query(query, params, **kwargs):
            captured["query"] = str(query)
            captured["params"] = params
            captured["kwargs"] = kwargs
            return _Cursor()

    projection = MessageStore(_PostgresDb()).get_source_message_projection(
        "conversation-1",
        max_chars=100,
        owner_user_id="owner-1",
    )

    sql = " ".join(str(captured["query"]).split()).lower()
    assert projection["conversation_exists"] is False
    assert "c.client_id = ?" in sql
    assert "c.deleted = false" in sql
    assert "m.deleted = false" in sql
    assert "instr(" not in sql
    assert sql.count("substr(") == 1
    assert captured["params"] == (101, "conversation-1", "owner-1", 21)
    assert captured["kwargs"] == {"log_params": False, "log_errors": False}


def test_postgres_source_message_projection_requires_owner_scope():
    class _PostgresDb:
        backend_type = BackendType.POSTGRESQL

        @staticmethod
        def execute_query(*_args, **_kwargs):
            raise AssertionError("unscoped PostgreSQL projection must not query")

    with pytest.raises(InputError, match="owner_user_id"):
        MessageStore(_PostgresDb()).get_source_message_projection(
            "conversation-1",
            max_chars=20,
        )


def test_source_message_projection_failure_drops_raw_exception_context():
    secret = "PRIVATE_MESSAGE_FRAGMENT"

    class _FailingDb:
        backend_type = BackendType.SQLITE

        @staticmethod
        def execute_query(*_args, **_kwargs):
            raise CharactersRAGDBError(secret)

    with pytest.raises(CharactersRAGDBError) as exc_info:
        MessageStore(_FailingDb()).get_source_message_projection(
            "conversation-1",
            max_chars=20,
        )

    assert str(exc_info.value) == "Source-message projection failed."
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert secret not in repr(exc_info.value)


def test_source_message_projection_redacts_delayed_fetch_failure():
    secret = "PRIVATE_MESSAGE_FETCH_FRAGMENT"
    messages: list[str] = []

    class _FailingCursor:
        @staticmethod
        def fetchall():
            raise sqlite3.OperationalError(f"Could not decode {secret}")

    class _FailingDb:
        backend_type = BackendType.SQLITE

        @staticmethod
        def execute_query(*_args, **_kwargs):
            return _FailingCursor()

    sink_id = logger.add(messages.append, level="DEBUG", format="{message}")
    try:
        with pytest.raises(CharactersRAGDBError) as exc_info:
            MessageStore(_FailingDb()).get_source_message_projection(
                "conversation-1",
                max_chars=20,
            )
    finally:
        logger.remove(sink_id)

    assert str(exc_info.value) == "Source-message projection failed."
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert secret not in repr(exc_info.value)
    assert secret not in "\n".join(messages)


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
    assert [
        row["id"] for row in store.get_messages_for_conversation_by_parent_ids(conversation_id, [root_message_id])
    ] == [child_message_id]
    assert store.has_system_message_for_conversation(conversation_id) is True

    root_message = store.get_message_by_id(root_message_id)
    assert root_message is not None
    assert (
        store.update_message(
            root_message_id,
            {"content": "Root updated"},
            expected_version=root_message["version"],
        )
        is True
    )

    updated_root = store.get_message_by_id(root_message_id)
    assert updated_root is not None
    assert updated_root["content"] == "Root updated"

    child_message = store.get_message_by_id(child_message_id)
    assert child_message is not None
    assert store.soft_delete_message(child_message_id, expected_version=child_message["version"]) is True
    assert store.get_message_by_id(child_message_id) is None
    assert store.count_messages_for_conversation(conversation_id) == 2
    assert store.count_messages_for_conversation(conversation_id, include_deleted=True) == 3


def test_history_snapshot_preserves_equal_text_distinct_ids(db):
    owner_db = db["db"]
    conversation_id = db["conversation_id"]
    first = owner_db.add_message({"conversation_id": conversation_id, "sender": "user", "content": "repeat"})
    second = owner_db.add_message({"conversation_id": conversation_id, "sender": "user", "content": "repeat"})
    snapshot = owner_db.get_conversation_history_snapshot(conversation_id, owner_client_id="message-store-user")
    assert {row["id"] for row in snapshot.nodes} == {first, second}


@pytest.mark.parametrize("invalid_message_id", [None, "", 7])
def test_inline_tool_call_backfill_skips_corrupt_identity_before_matching(db, monkeypatch, invalid_message_id):
    instance = db["db"]
    inline_content = 'Answer [tool_calls]: [{"id":"call-1","type":"function"}]'
    message_id = instance.add_message(
        {
            "conversation_id": db["conversation_id"],
            "sender": "assistant",
            "content": inline_content,
        }
    )
    before = instance.get_message_by_id(message_id)
    assert before is not None
    assert instance.get_message_metadata(message_id) is None
    execute_query = instance.execute_query

    def corrupt_scanner_read(query, params=None, **kwargs):
        if query == "SELECT id, content, version FROM messages WHERE sender = ? AND deleted = 0":
            # Corrupt only scanner input; real metadata and optimistic writers remain intact.
            return execute_query(
                "SELECT ? AS id, ? AS content, 1 AS version "
                "UNION ALL SELECT id, content, version FROM messages WHERE id = ?",
                (invalid_message_id, inline_content, message_id),
            )
        return execute_query(query, params, **kwargs)

    monkeypatch.setattr(instance, "execute_query", corrupt_scanner_read)

    counts = instance.backfill_tool_calls_from_inline(strip_inline=True)

    assert counts == {"scanned": 2, "matched": 1, "backfilled": 1, "stripped": 1}
    metadata = instance.get_message_metadata(message_id)
    assert metadata is not None
    assert metadata["tool_calls"] == [{"id": "call-1", "type": "function"}]
    message = instance.get_message_by_id(message_id)
    assert message is not None
    assert message["content"] == "Answer"
    assert message["version"] == before["version"] + 1


@pytest.mark.parametrize("mutation", ["live_content_update", "soft_delete_then_undelete"])
def test_message_fts_serial_mutation_keeps_native_writer_valid(db, mutation):
    store = db["store"]
    raw_db = db["db"]
    conversation_id = db["conversation_id"]
    first_id = store.add_message(
        {
            "id": "race-first",
            "conversation_id": conversation_id,
            "sender": "user",
            "content": "first",
            "timestamp": "2026-01-01T00:00:00.000Z",
        }
    )
    second_id = store.add_message(
        {
            "id": "race-second",
            "conversation_id": conversation_id,
            "sender": "assistant",
            "content": "second",
            "timestamp": "2026-01-01T00:00:02.000Z",
        }
    )
    hidden_id = store.add_message(
        {
            "id": "race-backdated",
            "conversation_id": conversation_id,
            "sender": "user",
            "content": "must not enter snapshot",
            "timestamp": "2026-01-01T00:00:01.000Z",
        }
    )
    assert first_id and second_id and hidden_id
    catalog = [
        dict(row)
        for row in raw_db.execute_query(
            "SELECT name, sql FROM sqlite_master "
            "WHERE type = 'trigger' AND name IN ('messages_au', 'messages_ad') "
            "ORDER BY name"
        ).fetchall()
    ]
    phase = "soft_delete_execute"
    try:
        raw_db.execute_query(
            "UPDATE messages SET deleted = 1 WHERE id = ?",
            (hidden_id,),
            commit=True,
        )
        phase = "connect"
        with sqlite3.connect(raw_db.db_path_str, timeout=5) as connection:
            if mutation == "soft_delete_then_undelete":
                phase = "undelete_execute"
                connection.execute("UPDATE messages SET deleted = 0 WHERE id = ?", (hidden_id,))
            else:
                phase = "live_update_execute"
                connection.execute(
                    """
                    UPDATE messages
                    SET content = ?, last_modified = ?
                    WHERE id = ?
                    """,
                    (
                        "changed after snapshot",
                        "2026-01-01T00:00:03.000Z",
                        first_id,
                    ),
                )
            phase = "mutation_commit"
            connection.commit()
    except (sqlite3.Error, CharactersRAGDBError) as exc:
        native_error = exc.__cause__ if isinstance(exc.__cause__, sqlite3.Error) else exc
        exc.add_note(
            f"mutation={mutation}; phase={phase}; sqlite_version={sqlite3.sqlite_version}; "
            f"sqlite_errorcode={getattr(native_error, 'sqlite_errorcode', None)!r}; "
            f"sqlite_errorname={getattr(native_error, 'sqlite_errorname', None)!r}; "
            f"preserved_db={raw_db.db_path_str}; trigger_catalog={catalog!r}"
        )
        raise

    written = store.get_message_by_id(hidden_id if mutation == "soft_delete_then_undelete" else first_id)
    assert written is not None
    assert written["content"] == (
        "must not enter snapshot" if mutation == "soft_delete_then_undelete" else "changed after snapshot"
    )
    assert store.get_message_by_id(second_id)["content"] == "second"


@pytest.mark.parametrize("reopen_legacy", [False, True])
def test_message_fts_lifecycle_preserves_live_term_ownership(db, reopen_legacy):
    instance = db["db"]
    conversation_id = db["conversation_id"]
    message_id = instance.add_message({"conversation_id": conversation_id, "sender": "user", "content": "originalterm"})
    assert message_id
    if reopen_legacy:
        # Reproduce the deployed V4/V11 catalog without changing its real index or data.
        connection = instance.get_connection()
        connection.execute("DROP TRIGGER messages_au")
        connection.execute("DROP TRIGGER messages_ad")
        connection.execute("""CREATE TRIGGER messages_au AFTER UPDATE ON messages BEGIN
            INSERT INTO messages_fts(messages_fts,rowid,content)
            VALUES('delete',old.rowid,old.content);
            INSERT INTO messages_fts(rowid,content)
            SELECT new.rowid,new.content WHERE new.deleted = 0;
            END""")
        connection.execute("""CREATE TRIGGER messages_ad AFTER DELETE ON messages BEGIN
            INSERT INTO messages_fts(messages_fts,rowid,content)
            VALUES('delete',old.rowid,old.content);
            END""")
        path = instance.db_path_str
        instance.close_all_connections()
        instance = CharactersRAGDB(path, client_id="message-store-user")
    try:
        connection = instance.get_connection()
        assert {
            row["name"]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'trigger' "
                "AND name IN ('messages_ai', 'messages_au', 'messages_ad')"
            )
        } == {"messages_ai", "messages_au", "messages_ad"}
        assert instance.get_message_by_id(message_id)["content"] == "originalterm"
        assert [row["id"] for row in instance.search_messages_by_content("originalterm")] == [message_id]
        connection.execute("UPDATE messages SET content = 'liveterm' WHERE id = ?", (message_id,))
        assert instance.search_messages_by_content("originalterm") == []
        assert [row["id"] for row in instance.search_messages_by_content("liveterm")] == [message_id]
        connection.execute("UPDATE messages SET deleted = 1 WHERE id = ?", (message_id,))
        assert instance.search_messages_by_content("liveterm") == []
        connection.execute("UPDATE messages SET content = 'hiddenterm' WHERE id = ?", (message_id,))
        assert list(connection.execute("SELECT rowid FROM messages_fts WHERE messages_fts MATCH 'hiddenterm'")) == []
        connection.execute("UPDATE messages SET deleted = 0 WHERE id = ?", (message_id,))
        assert [row["id"] for row in instance.search_messages_by_content("hiddenterm")] == [message_id]
        assert instance.search_messages_by_content("liveterm") == []
        connection.execute("UPDATE messages SET ranking = 7 WHERE id = ?", (message_id,))
        assert [row["id"] for row in instance.search_messages_by_content("hiddenterm")] == [message_id]
        connection.execute("UPDATE messages SET content = NULL WHERE id = ?", (message_id,))
        assert instance.search_messages_by_content("hiddenterm") == []
        connection.execute("UPDATE messages SET ranking = 8 WHERE id = ?", (message_id,))
        connection.execute("UPDATE messages SET content = 'restoredterm' WHERE id = ?", (message_id,))
        assert [row["id"] for row in instance.search_messages_by_content("restoredterm")] == [message_id]
        connection.execute("UPDATE messages SET deleted = 1 WHERE id = ?", (message_id,))
        connection.execute("DELETE FROM messages WHERE id = ?", (message_id,))
        assert instance.get_message_by_id(message_id, include_deleted=True) is None
        assert list(connection.execute("SELECT rowid FROM messages_fts WHERE messages_fts MATCH 'restoredterm'")) == []
        live_id = instance.add_message(
            {"conversation_id": conversation_id, "sender": "user", "content": "physicalterm"}
        )
        connection.execute("DELETE FROM messages WHERE id = ?", (live_id,))
        assert instance.search_messages_by_content("physicalterm") == []
    finally:
        if reopen_legacy:
            instance.close_all_connections()


def test_message_fts_v80_upgrade_preserves_native_data_and_term_ownership(tmp_path, monkeypatch):
    path = str(tmp_path / "message-fts-v80.sqlite")
    with monkeypatch.context() as pinned:
        pinned.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 80)
        instance = CharactersRAGDB(path, client_id="message-store-user")
        try:
            conversation_id = instance.add_conversation({"title": "Message FTS upgrade"})
            message_id = instance.add_message(
                {"conversation_id": conversation_id, "sender": "user", "content": "upgradeterm"}
            )
            connection = instance.get_connection()
            connection.execute("DROP TRIGGER messages_au")
            connection.execute("DROP TRIGGER messages_ad")
            connection.execute("""CREATE TRIGGER messages_au AFTER UPDATE ON messages BEGIN
                INSERT INTO messages_fts(messages_fts,rowid,content)
                VALUES('delete',old.rowid,old.content);
                INSERT INTO messages_fts(rowid,content)
                SELECT new.rowid,new.content WHERE new.deleted = 0; END""")
            connection.execute("""CREATE TRIGGER messages_ad AFTER DELETE ON messages BEGIN
                INSERT INTO messages_fts(messages_fts,rowid,content)
                VALUES('delete',old.rowid,old.content); END""")
            before = instance.get_message_by_id(message_id)
        finally:
            instance.close_all_connections()
    instance = CharactersRAGDB(path, client_id="message-store-user")
    try:
        assert instance.get_message_by_id(message_id) == before
        assert [row["id"] for row in instance.search_messages_by_content("upgradeterm")] == [message_id]
        connection = instance.get_connection()
        connection.execute("UPDATE messages SET deleted = 1 WHERE id = ?", (message_id,))
        connection.execute("UPDATE messages SET deleted = 0 WHERE id = ?", (message_id,))
        assert [row["id"] for row in instance.search_messages_by_content("upgradeterm")] == [message_id]
        assert (
            connection.execute(
                "SELECT version FROM db_schema_version WHERE schema_name = ?", (instance._SCHEMA_NAME,)
            ).fetchone()[0]
            == instance._CURRENT_SCHEMA_VERSION
        )
    finally:
        instance.close_all_connections()


def test_message_fts_normalization_failure_restores_previous_triggers(db):
    instance = db["db"]
    connection = instance.get_connection()
    before = list(
        connection.execute(
            "SELECT name, sql FROM sqlite_master WHERE type = 'trigger' "
            "AND name IN ('messages_ai', 'messages_au', 'messages_ad') ORDER BY name"
        )
    )
    assert not connection.in_transaction

    def deny_update_trigger(action, name, _table, _database, _origin):
        return (
            sqlite3.SQLITE_DENY
            if action == sqlite3.SQLITE_CREATE_TRIGGER and name == "messages_au"
            else sqlite3.SQLITE_OK
        )

    connection.set_authorizer(deny_update_trigger)
    try:
        with pytest.raises(CharactersRAGDBError, match="Failed ensuring messages FTS triggers") as error:
            instance._ensure_messages_fts_triggers_sqlite(connection)
    finally:
        connection.set_authorizer(None)
    assert isinstance(error.value.__cause__, sqlite3.DatabaseError)
    assert not connection.in_transaction
    assert (
        list(
            connection.execute(
                "SELECT name, sql FROM sqlite_master WHERE type = 'trigger' "
                "AND name IN ('messages_ai', 'messages_au', 'messages_ad') ORDER BY name"
            )
        )
        == before
    )
    message_id = instance.add_message(
        {"conversation_id": db["conversation_id"], "sender": "user", "content": "rollbackterm"}
    )
    assert [row["id"] for row in instance.search_messages_by_content("rollbackterm")] == [message_id]


def test_message_fts_normalization_does_not_commit_caller_pending_work(db):
    instance = db["db"]
    message_id = instance.add_message(
        {"conversation_id": db["conversation_id"], "sender": "user", "content": "pendingterm"}
    )
    connection = instance.get_connection()
    before = instance.get_message_by_id(message_id)
    connection.execute("BEGIN")
    try:
        connection.execute("UPDATE messages SET ranking = 19 WHERE id = ?", (message_id,))
        instance._ensure_messages_fts_triggers_sqlite(connection)
        assert connection.in_transaction
        assert instance.get_message_by_id(message_id)["ranking"] == 19
    finally:
        connection.rollback()
    assert instance.get_message_by_id(message_id) == before
    assert [row["id"] for row in instance.search_messages_by_content("pendingterm")] == [message_id]
