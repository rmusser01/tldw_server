"""Tests for graph-related batch query methods on CharactersRAGDB."""

import uuid

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

pytestmark = pytest.mark.unit


@pytest.fixture()
def db(tmp_path):
    """Fresh DB for each test."""
    db_path = tmp_path / "test_graph_queries.db"
    _db = CharactersRAGDB(str(db_path), client_id="u1")
    yield _db


def _make_note(db, title="N", content="body"):
    return db.add_note(title=title, content=content)


def _make_uuid():
    return str(uuid.uuid4())


# ---------- get_manual_edges_for_notes ----------

class TestGetManualEdgesForNotes:
    def test_empty_ids(self, db):
        assert db.get_manual_edges_for_notes("u1", []) == []

    def test_returns_edges(self, db):
        n1 = _make_note(db, title="A")
        n2 = _make_note(db, title="B")
        db.create_manual_note_edge(
            user_id="u1", from_note_id=n1, to_note_id=n2,
            directed=False, weight=1.0, created_by="test",
        )
        edges = db.get_manual_edges_for_notes("u1", [n1])
        assert len(edges) == 1
        assert edges[0]["from_note_id"] in (n1, n2)

    def test_other_user_excluded(self, db):
        n1 = _make_note(db, title="A")
        n2 = _make_note(db, title="B")
        db.create_manual_note_edge(
            user_id="u1", from_note_id=n1, to_note_id=n2,
            directed=False, weight=1.0, created_by="test",
        )
        assert db.get_manual_edges_for_notes("u2", [n1]) == []


# ---------- get_notes_batch ----------

class TestGetNotesBatch:
    def test_empty_ids(self, db):
        assert db.get_notes_batch([]) == []

    def test_returns_data(self, db):
        n1 = _make_note(db, title="Hello", content="world")
        rows = db.get_notes_batch([n1])
        assert len(rows) == 1
        assert rows[0]["title"] == "Hello"
        assert rows[0]["content"] == "world"
        assert "last_modified" in rows[0]

    def test_include_deleted_false(self, db):
        n1 = _make_note(db, title="Alive")
        n2 = _make_note(db, title="Dead")
        db.soft_delete_note(n2, expected_version=1)
        rows = db.get_notes_batch([n1, n2], include_deleted=False)
        assert len(rows) == 1
        assert rows[0]["id"] == n1

    def test_batching_over_900(self, db):
        # Just verify it doesn't crash with >900 IDs (most won't exist)
        fake_ids = [_make_uuid() for _ in range(950)]
        rows = db.get_notes_batch(fake_ids)
        assert rows == []


# ---------- get_all_note_ids_for_graph ----------

class TestGetAllNoteIdsForGraph:
    def test_empty_db(self, db):
        assert db.get_all_note_ids_for_graph() == []

    def test_returns_ids(self, db):
        n1 = _make_note(db)
        n2 = _make_note(db, title="B")
        ids = db.get_all_note_ids_for_graph()
        assert set(ids) == {n1, n2}

    def test_limit(self, db):
        for i in range(5):
            _make_note(db, title=f"N{i}")
        ids = db.get_all_note_ids_for_graph(limit=3)
        assert len(ids) == 3

    def test_exclude_deleted(self, db):
        n1 = _make_note(db, title="A")
        n2 = _make_note(db, title="B")
        db.soft_delete_note(n2, expected_version=1)
        ids = db.get_all_note_ids_for_graph(include_deleted=False)
        assert ids == [n1]


# ---------- get_note_tag_edges ----------

class TestGetNoteTagEdges:
    def test_empty_ids(self, db):
        assert db.get_note_tag_edges([]) == []

    def test_returns_tag_links(self, db):
        n1 = _make_note(db, title="Tagged")
        kw_id = db.add_keyword("ml")
        db.link_note_to_keyword(n1, kw_id)
        rows = db.get_note_tag_edges([n1])
        assert len(rows) == 1
        assert rows[0]["note_id"] == n1
        assert rows[0]["keyword"] == "ml"

    def test_deleted_keyword_excluded(self, db):
        n1 = _make_note(db, title="Tagged")
        kw_id = db.add_keyword("old_tag")
        db.link_note_to_keyword(n1, kw_id)
        db.soft_delete_keyword(kw_id, expected_version=1)
        rows = db.get_note_tag_edges([n1])
        assert rows == []


class TestGraphSeedLookups:
    def test_get_note_ids_by_tag_for_graph_finds_older_match_with_limit(self, db):
        for index in range(5):
            _make_note(db, title=f"Recent {index}")
        matching = _make_note(db, title="Older tagged")
        kw_id = db.add_keyword("deep-tag")
        db.link_note_to_keyword(matching, kw_id)

        rows = db.get_note_ids_by_tag_for_graph("deep-tag", limit=1)

        assert rows == [matching]

    def test_get_note_ids_by_source_for_graph_accepts_canonical_source_id(self, db):
        character_id = db.add_character_card({"name": "Graph Source Character"})
        conversation_id = db.add_conversation(
            {
                "character_id": character_id,
                "title": "Source Conversation",
                "source": "youtube",
                "external_ref": "video-123",
            }
        )
        matching = db.add_note(
            title="Sourced note",
            content="body",
            conversation_id=conversation_id,
        )

        rows = db.get_note_ids_by_source_for_graph("source:youtube:video-123", limit=5)

        assert rows == [matching]


# ---------- count_notes_per_tag ----------

class TestCountNotesPerTag:
    def test_empty_db(self, db):
        assert db.count_notes_per_tag() == {}

    def test_counts(self, db):
        n1 = _make_note(db, title="A")
        n2 = _make_note(db, title="B")
        kw = db.add_keyword("shared")
        db.link_note_to_keyword(n1, kw)
        db.link_note_to_keyword(n2, kw)
        counts = db.count_notes_per_tag()
        assert counts[kw] == 2


# ---------- get_note_source_info ----------

class TestGetNoteSourceInfo:
    def test_empty_ids(self, db):
        assert db.get_note_source_info([]) == []

    def test_note_without_conversation(self, db):
        n1 = _make_note(db)
        assert db.get_note_source_info([n1]) == []


# ---------- count_user_notes ----------

class TestCountUserNotes:
    def test_empty(self, db):
        assert db.count_user_notes() == 0

    def test_counts_all(self, db):
        _make_note(db)
        _make_note(db, title="B")
        assert db.count_user_notes() == 2

    def test_exclude_deleted(self, db):
        n1 = _make_note(db)
        _make_note(db, title="B")
        db.soft_delete_note(n1, expected_version=1)
        assert db.count_user_notes(include_deleted=False) == 1


# ---------- _chunk_list ----------

class TestChunkList:
    def test_empty(self):
        assert CharactersRAGDB._chunk_list([], 10) == []

    def test_exact_size(self):
        assert CharactersRAGDB._chunk_list([1, 2, 3], 3) == [[1, 2, 3]]

    def test_smaller_chunks(self):
        result = CharactersRAGDB._chunk_list([1, 2, 3, 4, 5], 2)
        assert result == [[1, 2], [3, 4], [5]]
