"""Notes graph query results preserve behavior across supported backends."""

import pytest

from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=pytest.mark.postgres)])
def graph_db(request, tmp_path):
    backend = (
        DatabaseBackendFactory.create_backend(request.getfixturevalue("pg_database_config"))
        if request.param == "postgres"
        else None
    )
    db = CharactersRAGDB(tmp_path / "notes.db", client_id="3", backend=backend)
    try:
        yield db
    finally:
        db.close_all_connections()
        if backend:
            backend.get_pool().close_all()


@pytest.mark.parametrize("populated", [False, True], ids=["empty", "populated"])
@pytest.mark.parametrize("operation", ["all-ids", "tag-ids", "source-ids", "count-notes", "counts-per-tag"])
def test_graph_mapping_rows(graph_db, populated, operation):
    db = graph_db
    live = None
    deleted = None
    kw = None
    if populated:
        char = db.add_character_card({"name": "Graph probe character"})
        conv = db.add_conversation(
            {"character_id": char, "title": "Source fixture", "source": "youtube", "external_ref": "fixture-1"}
        )
        live = db.add_note("Live graph note", "body", conversation_id=conv)
        deleted = db.add_note("Deleted graph note", "body", conversation_id=conv)
        kw = db.add_keyword("graph-tag")
        db.link_note_to_keyword(live, kw)
        db.link_note_to_keyword(deleted, kw)
        db.soft_delete_note(deleted, expected_version=1)
    if operation == "all-ids":
        assert db.get_all_note_ids_for_graph(include_deleted=False) == ([live] if populated else [])
        assert set(db.get_all_note_ids_for_graph()) == ({live, deleted} if populated else set())
    elif operation == "tag-ids":
        assert db.get_note_ids_by_tag_for_graph("graph-tag", include_deleted=False) == ([live] if populated else [])
        assert set(db.get_note_ids_by_tag_for_graph("tag:graph-tag")) == ({live, deleted} if populated else set())
    elif operation == "source-ids":
        assert db.get_note_ids_by_source_for_graph("source:youtube:fixture-1", include_deleted=False) == (
            [live] if populated else []
        )
        assert set(db.get_note_ids_by_source_for_graph("youtube")) == ({live, deleted} if populated else set())
    elif operation == "count-notes":
        assert db.count_user_notes() == (2 if populated else 0)
        assert db.count_user_notes(include_deleted=False) == (1 if populated else 0)
    else:
        assert db.count_notes_per_tag() == ({kw: 1} if populated else {})
