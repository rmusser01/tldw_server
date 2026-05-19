"""Tests for the extracted NoteStore."""

import ast
import inspect
from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.chacha.note_store import NoteStore


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
