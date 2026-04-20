"""Tests for the extracted NoteStore."""

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.chacha.note_store import NoteStore


pytestmark = pytest.mark.unit


@pytest.fixture()
def db(tmp_path):
    return CharactersRAGDB(
        db_path=str(tmp_path / "note_store.sqlite"),
        client_id="note-store-user",
    )


@pytest.fixture()
def store(db):
    return NoteStore(db)


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


class TestNoteStoreSearch:
    def test_search_notes(self, store):
        store.add_note(title="Searchable Item", content="Unique searchable content xyz123")
        results = store.search_notes("xyz123")
        assert len(results) >= 1
