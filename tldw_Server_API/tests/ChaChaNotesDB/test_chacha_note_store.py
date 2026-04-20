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
