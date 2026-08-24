import ast
import inspect
import sqlite3
import threading
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

    writer = threading.Thread(target=write_concurrently)
    writer.start()
    projection = store.get_source_message_projection(conversation_id, max_chars=100)
    writer.join(timeout=5)

    assert not writer.is_alive()
    assert [row["source_text"] for row in projection["rows"]] == [
        "user: first",
        "assistant: second",
    ]


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
