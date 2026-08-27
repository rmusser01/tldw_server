from __future__ import annotations

import pytest

from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.Notes_Graph.suggestion_retrieval import SuggestionRetriever

pytestmark = pytest.mark.integration


def _seed_and_retrieve(db: CharactersRAGDB) -> list[str]:
    source = "00000000-0000-4000-8000-000000000001"
    candidate = "00000000-0000-4000-8000-000000000002"
    direct = "00000000-0000-4000-8000-000000000003"
    db.add_note("Neural retrieval delta-theory", f"lexical graph [[id:{direct}]]", note_id=source)
    db.add_note("Graph retrieval", "neural lexical", note_id=candidate)
    db.add_note("Neural graph", "retrieval link", note_id=direct)

    result = SuggestionRetriever(db.note_graph_suggestion_store).retrieve(
        dataset_id="dataset-1", source_note_id=source
    )

    assert result.backend_overfetch_count <= 60
    assert candidate in [item.note_id for item in result.candidates]
    assert direct not in [item.note_id for item in result.candidates]
    return [item.note_id for item in result.candidates]


def test_sqlite_retrieval_uses_owner_bound_fts_and_byte_guards(tmp_path) -> None:
    db = CharactersRAGDB(str(tmp_path / "retrieval.db"), client_id="owner-1")
    try:
        assert _seed_and_retrieve(db) == ["00000000-0000-4000-8000-000000000002"]
    finally:
        db.close_connection()


def test_postgres_retrieval_uses_owner_dataset_scope_and_fts(
    pg_database_config,
) -> None:
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(":memory:", client_id="owner-1", backend=backend)
    try:
        assert _seed_and_retrieve(db) == ["00000000-0000-4000-8000-000000000002"]
    finally:
        db.close_connection()
