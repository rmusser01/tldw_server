"""Tests for the extracted NoteStore."""

import ast
import inspect
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from loguru import logger

from tldw_Server_API.app.core.DB_Management.chacha.note_store import NoteStore
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    BackendType,
    CharactersRAGDB,
    CharactersRAGDBError,
    ConflictError,
    InputError,
)
from tldw_Server_API.app.core.DB_Management.Sync_DB import _envelope_from_row

pytestmark = pytest.mark.unit


_DELEGATED_NOTE_METHODS = {
    "add_note",
    "get_note_by_id",
    "list_notes",
    "count_notes",
    "update_note",
    "soft_delete_note",
    "restore_note",
    "list_deleted_notes",
    "search_notes",
    "link_note_to_keyword",
    "get_notes_batch",
    "get_all_note_ids_for_graph",
    "get_note_tag_edges",
    "count_user_notes",
    "count_notes_per_tag",
    "get_note_source_info",
    "get_keywords_for_note",
    "get_keywords_for_notes",
    "get_note_counts_for_keywords",
    "upsert_note_from_sync",
    "tombstone_note_from_sync",
}


def _postgres_datetime_envelope(server_timestamp: datetime):
    return _envelope_from_row(
        {
            "server_sequence": 1,
            "dataset_id": "dataset-1",
            "client_envelope_id": "env-postgres-timestamp",
            "domain": "notes.note",
            "entity_id": "sync-note-postgres-time",
            "operation": "upsert",
            "server_timestamp": server_timestamp,
            "status": "accepted",
        }
    )


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
    return CharactersRAGDB(
        db_path=str(tmp_path / "note_store.sqlite"),
        client_id="note-store-user",
    )


@pytest.fixture()
def store(db):
    return NoteStore(db)


def test_note_store_owns_delegated_methods_without_monolith_duplicates(db, monkeypatch):
    class_method_names = _class_method_names(CharactersRAGDB)
    assert _DELEGATED_NOTE_METHODS.isdisjoint(class_method_names)

    captured: dict[str, object] = {}

    def _fake_add_note(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return "note-from-store"

    monkeypatch.setattr(db.note_store, "add_note", _fake_add_note)

    assert db.add_note(title="Delegated Note", content="delegated body") == "note-from-store"
    assert captured["args"] == ()
    assert captured["kwargs"] == {"title": "Delegated Note", "content": "delegated body"}


def test_source_note_projection_is_active_bounded_and_explicit(store, db, monkeypatch):
    note_id = store.add_note(
        title="abcdef",
        content="123456789",
        note_id="source-note",
    )
    assert note_id == "source-note"

    queries: list[str] = []
    original_execute = db.execute_query

    def recording_execute(query, params=None, **kwargs):
        queries.append(str(query))
        return original_execute(query, params, **kwargs)

    monkeypatch.setattr(db, "execute_query", recording_execute)

    assert store.get_source_note_projection("source-note", max_chars=5) == {
        "id": "source-note",
        "source_text": "# abcd",
        "source_invalid": False,
    }
    normalized_sql = " ".join(queries).lower()
    assert "select *" not in normalized_sql
    assert "substr" in normalized_sql
    assert "deleted =" in normalized_sql
    assert "payload_json" not in normalized_sql


def test_source_note_projection_hides_deleted_note_but_not_deleted_backlink_conversation(
    store,
    db,
):
    character_id = db.add_character_card({"name": "Source note character"})
    conversation_id = db.add_conversation(
        {
            "character_id": character_id,
            "title": "Source note conversation",
        }
    )
    note_id = store.add_note(
        title="Visible note",
        content="Visible body",
        note_id="source-note-linked",
        conversation_id=conversation_id,
    )
    assert note_id

    db.execute_query(
        "UPDATE conversations SET deleted = 1 WHERE id = ?",
        (conversation_id,),
        commit=True,
    )
    assert store.get_source_note_projection(note_id, max_chars=50) == {
        "id": note_id,
        "source_text": "# Visible note\n\nVisible body",
        "source_invalid": False,
    }

    db.execute_query(
        "UPDATE notes SET deleted = 1 WHERE id = ?",
        (note_id,),
        commit=True,
    )
    assert store.get_source_note_projection(note_id, max_chars=50) is None


def test_source_note_projection_accepts_zero_as_sentinel_budget(store, db):
    populated_id = store.add_note(
        title="Populated",
        content="body",
        note_id="source-note-populated",
    )
    empty_id = store.add_note(
        title="Empty",
        content="placeholder",
        note_id="source-note-empty",
    )
    db.execute_query(
        "UPDATE notes SET title = '', content = '' WHERE id = ?",
        (empty_id,),
        commit=True,
    )

    assert store.get_source_note_projection(populated_id, max_chars=0) == {
        "id": populated_id,
        "source_text": "#",
        "source_invalid": False,
    }
    assert store.get_source_note_projection(empty_id, max_chars=0) == {
        "id": empty_id,
        "source_text": "",
        "source_invalid": False,
    }


@pytest.mark.parametrize("max_chars", [True, -1, "10"])
def test_source_note_projection_rejects_invalid_character_budget(store, max_chars):
    with pytest.raises(InputError):
        store.get_source_note_projection("note", max_chars=max_chars)


def test_postgres_source_note_projection_requires_owner_scope():
    class _PostgresDb:
        backend_type = BackendType.POSTGRESQL

        @staticmethod
        def execute_query(*_args, **_kwargs):
            raise AssertionError("unscoped PostgreSQL projection must not query")

    with pytest.raises(InputError, match="owner_user_id"):
        NoteStore(_PostgresDb()).get_source_note_projection("note-1", max_chars=20)


def test_source_note_projection_marks_nul_text_invalid(store, db):
    note_id = store.add_note(
        title="NUL note",
        content="valid",
        note_id="source-note-nul",
    )
    db.execute_query(
        "UPDATE notes SET content = ? WHERE id = ?",
        ("prefix\0" + ("secret" * 1000), note_id),
        commit=True,
    )

    projection = store.get_source_note_projection(note_id, max_chars=20)

    assert projection is not None
    assert projection["source_invalid"] is True


def test_source_note_projection_failure_is_fixed_and_redacted():
    secret = "PRIVATE_NOTE_FRAGMENT"
    messages: list[str] = []

    class _FailingDb:
        backend_type = BackendType.SQLITE

        @staticmethod
        def execute_query(*_args, **_kwargs):
            raise CharactersRAGDBError(secret)

    sink_id = logger.add(messages.append, level="DEBUG", format="{message}")
    try:
        with pytest.raises(CharactersRAGDBError) as exc_info:
            NoteStore(_FailingDb()).get_source_note_projection(
                "note-1",
                max_chars=20,
            )
    finally:
        logger.remove(sink_id)

    assert str(exc_info.value) == "Source-note projection failed."
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert secret not in repr(exc_info.value)
    assert secret not in "\n".join(messages)


def test_source_note_projection_redacts_delayed_fetch_failure():
    secret = "PRIVATE_NOTE_FETCH_FRAGMENT"
    messages: list[str] = []

    class _FailingCursor:
        @staticmethod
        def fetchone():
            raise sqlite3.OperationalError(f"Could not decode {secret}")

    class _FailingDb:
        backend_type = BackendType.SQLITE

        @staticmethod
        def execute_query(*_args, **_kwargs):
            return _FailingCursor()

    sink_id = logger.add(messages.append, level="DEBUG", format="{message}")
    try:
        with pytest.raises(CharactersRAGDBError) as exc_info:
            NoteStore(_FailingDb()).get_source_note_projection(
                "note-1",
                max_chars=20,
            )
    finally:
        logger.remove(sink_id)

    assert str(exc_info.value) == "Source-note projection failed."
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert secret not in repr(exc_info.value)
    assert secret not in "\n".join(messages)


class TestNoteStoreAdd:
    def test_add_note(self, store):
        note_id = store.add_note(
            title="Test Note",
            content="Some content",
        )
        assert note_id is not None

    def test_add_note_retrievable(self, store):
        note_id = store.add_note(
            title="Retrievable Note",
            content="Content here",
        )
        note = store.get_note_by_id(note_id)
        assert note is not None
        assert note["title"] == "Retrievable Note"


class TestNoteStoreList:
    def test_list_notes(self, store):
        store.add_note(title="Note A", content="A")
        store.add_note(title="Note B", content="B")
        notes = store.list_notes()
        assert len(notes) >= 2

    def test_count_notes(self, store):
        store.add_note(title="Count Test", content="C")
        count = store.count_notes()
        assert count >= 1


class TestNoteStoreUpdate:
    def test_update_note(self, store):
        note_id = store.add_note(title="Original", content="Original content")
        note = store.get_note_by_id(note_id)
        result = store.update_note(
            note_id,
            {"title": "Updated", "content": "Updated content"},
            expected_version=note["version"],
        )
        assert result is True
        updated = store.get_note_by_id(note_id)
        assert updated["title"] == "Updated"


class TestNoteStoreSyncHelpers:
    def test_upsert_note_from_sync_creates_note_with_stable_id_and_revision(self, db):
        result = db.upsert_note_from_sync(
            note_id="sync-note-1",
            title="Synced note",
            content="Synced body",
            conversation_id=None,
            message_id=None,
            sync_client_id="device-1",
            object_revision=4,
            object_hash="sha256:note-v4",
        )

        assert result is True
        note = db.get_note_by_id("sync-note-1")
        assert note is not None
        assert note["title"] == "Synced note"
        assert note["content"] == "Synced body"
        assert note["client_id"] == "device-1"
        assert note["version"] == 4
        assert bool(note["deleted"]) is False

    def test_upsert_note_from_sync_updates_note_without_changing_created_at(self, db):
        db.upsert_note_from_sync(
            note_id="sync-note-1",
            title="Synced note",
            content="Synced body",
            conversation_id=None,
            message_id=None,
            sync_client_id="device-1",
            object_revision=1,
            object_hash="sha256:note-v1",
        )
        before = db.get_note_by_id("sync-note-1")
        assert before is not None

        result = db.upsert_note_from_sync(
            note_id="sync-note-1",
            title="Synced note revised",
            content="Updated body",
            conversation_id=None,
            message_id=None,
            sync_client_id="device-2",
            object_revision=2,
            object_hash="sha256:note-v2",
        )

        assert result is True
        after = db.get_note_by_id("sync-note-1")
        assert after is not None
        assert after["title"] == "Synced note revised"
        assert after["content"] == "Updated body"
        assert after["client_id"] == "device-2"
        assert after["version"] == 2
        assert after["created_at"] == before["created_at"]

    def test_ingestion_guard_accepts_exact_intended_postcondition(self, db):
        db.add_note(
            title="Before",
            content="Original",
            note_id="sync-note-guard",
        )
        kwargs = {
            "note_id": "sync-note-guard",
            "title": "After",
            "content": "Intended",
            "conversation_id": None,
            "message_id": None,
            "sync_client_id": "note-store-user",
            "object_revision": 2,
            "object_hash": "sha256:note-v2",
            "expected_product_version": 1,
            "projection_timestamp": "2026-08-09T08:00:00+00:00",
        }

        assert db.upsert_note_from_sync(**kwargs) is True
        after_product_commit = db.get_note_by_id("sync-note-guard")
        assert db.upsert_note_from_sync(**kwargs) is False
        assert db.get_note_by_id("sync-note-guard") == after_product_commit

    def test_ingestion_guard_rejects_divergent_same_version_postcondition(self, db):
        db.add_note(
            title="Before",
            content="Original",
            note_id="sync-note-diverged",
        )
        kwargs = {
            "note_id": "sync-note-diverged",
            "title": "After",
            "content": "Intended",
            "conversation_id": None,
            "message_id": None,
            "sync_client_id": "note-store-user",
            "object_revision": 2,
            "object_hash": "sha256:note-v2",
            "expected_product_version": 1,
            "projection_timestamp": "2026-08-09T08:00:00+00:00",
        }
        db.upsert_note_from_sync(**kwargs)
        with db.transaction() as conn:
            conn.execute(
                "UPDATE notes SET content = ? WHERE id = ?",
                ("Diverged", "sync-note-diverged"),
            )

        with pytest.raises(ConflictError, match="changed after ingestion planning"):
            db.upsert_note_from_sync(**kwargs)

    def test_postgres_datetime_recovery_accepts_equivalent_utc_offset_without_write(
        self,
        db,
    ):
        envelope = _postgres_datetime_envelope(
            datetime(
                2026,
                8,
                9,
                1,
                tzinfo=timezone(timedelta(hours=-7)),
            )
        )
        db.add_note(
            title="Before",
            content="Original",
            note_id="sync-note-postgres-time",
        )
        kwargs = {
            "note_id": "sync-note-postgres-time",
            "title": "After",
            "content": "Intended",
            "conversation_id": None,
            "message_id": None,
            "sync_client_id": "note-store-user",
            "object_revision": 2,
            "object_hash": "sha256:note-v2",
            "expected_product_version": 1,
            "projection_timestamp": envelope.server_timestamp,
        }
        assert db.upsert_note_from_sync(**kwargs) is True
        with db.transaction() as conn:
            conn.execute(
                "UPDATE notes SET last_modified = ? WHERE id = ?",
                ("2026-08-09T08:00:00+00:00", "sync-note-postgres-time"),
            )
        after_product_commit = db.get_note_by_id("sync-note-postgres-time")

        assert db.upsert_note_from_sync(**kwargs) is False
        assert db.get_note_by_id("sync-note-postgres-time") == after_product_commit
        assert envelope.server_timestamp == "2026-08-09T08:00:00+00:00"

    @pytest.mark.parametrize(
        "stored_timestamp",
        ["2026-08-09T08:00:01+00:00", "invalid-timestamp"],
    )
    def test_postgres_datetime_recovery_rejects_different_timestamp(
        self,
        db,
        stored_timestamp,
    ):
        envelope = _postgres_datetime_envelope(
            datetime(2026, 8, 9, 8, tzinfo=timezone.utc)
        )
        db.add_note(
            title="Before",
            content="Original",
            note_id="sync-note-postgres-time",
        )
        kwargs = {
            "note_id": "sync-note-postgres-time",
            "title": "After",
            "content": "Intended",
            "conversation_id": None,
            "message_id": None,
            "sync_client_id": "note-store-user",
            "object_revision": 2,
            "object_hash": "sha256:note-v2",
            "expected_product_version": 1,
            "projection_timestamp": envelope.server_timestamp,
        }
        db.upsert_note_from_sync(**kwargs)
        with db.transaction() as conn:
            conn.execute(
                "UPDATE notes SET last_modified = ? WHERE id = ?",
                (stored_timestamp, "sync-note-postgres-time"),
            )
        divergent = db.get_note_by_id("sync-note-postgres-time")

        with pytest.raises(ConflictError, match="changed after ingestion planning"):
            db.upsert_note_from_sync(**kwargs)
        assert db.get_note_by_id("sync-note-postgres-time") == divergent

    def test_tombstone_note_from_sync_soft_deletes_existing_note(self, db):
        db.upsert_note_from_sync(
            note_id="sync-note-1",
            title="Synced note",
            content="Synced body",
            conversation_id=None,
            message_id=None,
            sync_client_id="device-1",
            object_revision=1,
            object_hash="sha256:note-v1",
        )

        result = db.tombstone_note_from_sync(
            note_id="sync-note-1",
            sync_client_id="device-1",
            object_revision=2,
            object_hash="sha256:note-delete",
        )

        assert result is True
        assert db.get_note_by_id("sync-note-1") is None
        deleted = db.get_note_by_id("sync-note-1", include_deleted=True)
        assert deleted is not None
        assert bool(deleted["deleted"]) is True
        assert deleted["version"] == 2
        assert deleted["client_id"] == "device-1"


class TestNoteStoreSoftDelete:
    def test_soft_delete_and_restore(self, store):
        note_id = store.add_note(title="Deletable", content="To be deleted")
        note = store.get_note_by_id(note_id)

        result = store.soft_delete_note(note_id, expected_version=note["version"])
        assert result is True

        deleted = store.get_note_by_id(note_id)
        assert deleted is None or deleted.get("deleted") == 1

        deleted_list = store.list_deleted_notes()
        assert any(n["id"] == note_id for n in deleted_list)

        restored = store.restore_note(note_id, expected_version=note["version"] + 1)
        assert restored is True
        assert store.get_note_by_id(note_id) is not None

    def test_list_deleted_notes_excludes_active_rows(self, store):
        deleted_id = store.add_note(title="Deleted", content="trash me")
        active_id = store.add_note(title="Active", content="keep me")

        deleted_note = store.get_note_by_id(deleted_id)
        assert deleted_note is not None
        assert store.soft_delete_note(deleted_id, expected_version=deleted_note["version"]) is True

        deleted_notes = store.list_deleted_notes()

        deleted_ids = {row["id"] for row in deleted_notes}
        assert deleted_id in deleted_ids
        assert active_id not in deleted_ids


class TestNoteStoreSearch:
    def test_search_notes(self, store):
        store.add_note(title="Searchable Item", content="Unique searchable content xyz123")
        results = store.search_notes("xyz123")
        assert len(results) >= 1

    def test_search_notes_by_title(self, store):
        note_id = store.add_note(title="Title Search", content="regular body")

        results = store.search_notes("Title Search")

        assert len(results) == 1
        assert results[0]["id"] == note_id


class TestNoteStoreGraphHelpers:
    def test_notes_batch_preserves_conversation_and_message_backlinks(self, store, db):
        conversation_id = db.add_conversation({"title": "Source conversation"})
        message_id = db.add_message(
            {
                "conversation_id": conversation_id,
                "sender": "user",
                "content": "Source message",
            }
        )
        note_id = store.add_note(
            title="Linked note",
            content="Linked content",
            conversation_id=conversation_id,
            message_id=message_id,
        )

        rows = store.get_notes_batch([note_id])

        assert rows == [
            {
                "id": note_id,
                "title": "Linked note",
                "content": "Linked content",
                "created_at": rows[0]["created_at"],
                "last_modified": rows[0]["last_modified"],
                "deleted": rows[0]["deleted"],
                "conversation_id": conversation_id,
                "message_id": message_id,
            }
        ]

    def test_graph_helpers_and_keyword_links(self, store, db):
        character_id = db.add_character_card({"name": "Note Source Character"})
        conversation_id = db.add_conversation(
            {
                "character_id": character_id,
                "title": "Source Conversation",
                "source": "youtube",
                "external_ref": "video-123",
            }
        )

        first_note_id = store.add_note(
            title="First linked note",
            content="content one",
            conversation_id=conversation_id,
        )
        second_note_id = store.add_note(
            title="Second linked note",
            content="content two",
            conversation_id=conversation_id,
        )
        keyword_one_id = db.add_keyword("astrophysics")
        keyword_two_id = db.add_keyword("cosmology")

        assert store.link_note_to_keyword(first_note_id, keyword_one_id)
        assert store.link_note_to_keyword(first_note_id, keyword_two_id)
        assert store.link_note_to_keyword(second_note_id, keyword_two_id)

        second_note = store.get_note_by_id(second_note_id)
        assert second_note is not None
        assert store.soft_delete_note(second_note_id, expected_version=second_note["version"]) is True

        active_batch = store.get_notes_batch([first_note_id, second_note_id], include_deleted=False)
        assert {row["id"] for row in active_batch} == {first_note_id}
        assert store.get_all_note_ids_for_graph(include_deleted=False) == [first_note_id]
        tag_edges = store.get_note_tag_edges([first_note_id, second_note_id])
        assert {(row["note_id"], row["keyword"]) for row in tag_edges} == {
            (first_note_id, "astrophysics"),
            (first_note_id, "cosmology"),
            (second_note_id, "cosmology"),
        }
        assert store.count_user_notes(include_deleted=True) == 2
        assert store.count_user_notes(include_deleted=False) == 1
        assert store.count_notes_per_tag() == {keyword_one_id: 1, keyword_two_id: 1}

        source_info = store.get_note_source_info([first_note_id, second_note_id])
        assert any(
            row["note_id"] == first_note_id
            and row["conversation_id"] == conversation_id
            and row["source"] == "youtube"
            and row["external_ref"] == "video-123"
            for row in source_info
        )

        assert {row["id"] for row in store.get_keywords_for_note(first_note_id)} == {
            keyword_one_id,
            keyword_two_id,
        }
        keyword_map = store.get_keywords_for_notes([first_note_id, second_note_id])
        assert {row["id"] for row in keyword_map[first_note_id]} == {keyword_one_id, keyword_two_id}
        assert {row["id"] for row in keyword_map[second_note_id]} == {keyword_two_id}
        assert store.get_note_counts_for_keywords([keyword_one_id, keyword_two_id]) == {
            keyword_one_id: 1,
            keyword_two_id: 1,
        }

        assert store.unlink_note_from_keyword(first_note_id, keyword_one_id)
        assert {row["id"] for row in store.get_keywords_for_note(first_note_id)} == {keyword_two_id}
        assert db.unlink_note_to_keyword(first_note_id, keyword_two_id)
        assert store.get_keywords_for_note(first_note_id) == []
