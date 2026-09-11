from __future__ import annotations

import pytest

from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.chacha.note_graph_suggestion_store import (
    NotesGraphFTSNotReadyError,
    NotesGraphSourceTooLargeError,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    BackendConnectionWrapper,
    CharactersRAGDB,
)
from tldw_Server_API.app.core.Notes_Graph.suggestion_content import content_fingerprint
from tldw_Server_API.app.core.Notes_Graph.suggestion_retrieval import SuggestionRetriever

pytestmark = pytest.mark.integration


DATASET_ID = "dataset-1"


def _authorize_dataset(db: CharactersRAGDB) -> None:
    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO note_task_scope_authority(owner_user_id,dataset_id) VALUES (?,?)",
            (db.client_id, DATASET_ID),
        )


def _set_graph_scope(db: CharactersRAGDB, conn) -> None:
    if db.note_graph_suggestion_store.is_postgres:
        conn.execute("SELECT set_config('app.current_dataset_id', ?, true)", (DATASET_ID,))


def _seed_and_retrieve(db: CharactersRAGDB) -> list[str]:
    source = "00000000-0000-4000-8000-000000000001"
    candidate = "00000000-0000-4000-8000-000000000002"
    direct = "00000000-0000-4000-8000-000000000003"
    db.add_note("Neural retrieval delta-theory", f"lexical graph [[id:{direct}]]", note_id=source)
    db.add_note("Graph retrieval", "neural lexical", note_id=candidate)
    db.add_note("Neural graph", "retrieval link", note_id=direct)

    result = SuggestionRetriever(db.note_graph_suggestion_store).retrieve(
        dataset_id=DATASET_ID, source_note_id=source
    )

    assert result.backend_overfetch_count <= 60
    assert candidate in [item.note_id for item in result.candidates]
    assert direct not in [item.note_id for item in result.candidates]
    return [item.note_id for item in result.candidates]


def test_sqlite_retrieval_uses_owner_bound_fts_and_byte_guards(tmp_path) -> None:
    db = CharactersRAGDB(str(tmp_path / "retrieval.db"), client_id="owner-1")
    try:
        _authorize_dataset(db)
        assert _seed_and_retrieve(db) == ["00000000-0000-4000-8000-000000000002"]
    finally:
        db.close_connection()


def test_postgres_retrieval_uses_owner_dataset_scope_and_fts(
    pg_database_config,
) -> None:
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(":memory:", client_id="owner-1", backend=backend)
    try:
        _authorize_dataset(db)
        assert _seed_and_retrieve(db) == ["00000000-0000-4000-8000-000000000002"]
    finally:
        db.close_connection()


def test_postgres_retrieval_rejects_oversized_source_before_payload_transfer(
    pg_database_config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(":memory:", client_id="owner-1", backend=backend)
    try:
        _authorize_dataset(db)
        source = "00000000-0000-4000-8000-000000000010"
        db.add_note("retrieval", "x" * 1_000_001, note_id=source)
        executed_sql: list[str] = []
        original_execute = BackendConnectionWrapper.execute

        def capture_execute(self, query: str, params=None):
            executed_sql.append(query)
            return original_execute(self, query, params)

        monkeypatch.setattr(BackendConnectionWrapper, "execute", capture_execute)

        with pytest.raises(NotesGraphSourceTooLargeError) as exc:
            SuggestionRetriever(db.note_graph_suggestion_store).retrieve(
                dataset_id=DATASET_ID, source_note_id=source
            )

        assert str(exc.value) == "notes_graph_source_too_large"
        source_payload_queries = [
            query for query in executed_sql if "SELECT n.id, n.title, n.content" in query
        ]
        assert len(source_payload_queries) == 1
        assert "octet_length(COALESCE(n.title, '')) + octet_length(COALESCE(n.content, ''))" in source_payload_queries[0]
        assert "<= ?" in source_payload_queries[0]
        assert not any("notes_fts_tsv @@" in query for query in executed_sql)
    finally:
        db.close_connection()


def _exercise_ranked_shortlist_and_bounds(db: CharactersRAGDB) -> None:
    source = "00000000-0000-4000-8000-000000000100"
    direct_ids = [f"00000000-0000-4000-8000-{number:012d}" for number in range(200, 260)]
    outside_id = "00000000-0000-4000-8000-000000000260"
    outside_oversized_id = "00000000-0000-4000-8000-000000000261"
    db.add_note("alpha", "source", note_id=source)
    for note_id in direct_ids:
        db.add_note("alpha", "candidate", note_id=note_id)
    db.add_note("alpha", "outside", note_id=outside_id)
    db.add_note("alpha", "x" * 250_001, note_id=outside_oversized_id)
    db.update_note(
        source,
        {"content": " ".join(f"[[id:{note_id}]]" for note_id in direct_ids)},
        expected_version=1,
    )

    result = SuggestionRetriever(db.note_graph_suggestion_store).retrieve(
        dataset_id=DATASET_ID, source_note_id=source
    )

    assert result.backend_overfetch_count == 60
    assert result.candidates == ()
    assert result.excluded_oversized_candidate_count == 0

    source = "00000000-0000-4000-8000-000000000300"
    oversized = "00000000-0000-4000-8000-000000000301"
    db.add_note("beta", "source", note_id=source)
    db.add_note("beta", "x" * 250_001, note_id=oversized)
    expected_ids = []
    for number in range(302, 362):
        note_id = f"00000000-0000-4000-8000-{number:012d}"
        expected_ids.append(note_id)
        db.add_note("beta", "candidate", note_id=note_id)

    result = SuggestionRetriever(db.note_graph_suggestion_store).retrieve(
        dataset_id=DATASET_ID, source_note_id=source
    )

    assert result.backend_overfetch_count == 60
    assert [candidate.note_id for candidate in result.candidates] == expected_ids[:30]
    assert result.excluded_oversized_candidate_count == 1


def _exercise_exact_rejection_and_tag_limit(db: CharactersRAGDB) -> None:
    source = "00000000-0000-4000-8000-000000000400"
    target = "00000000-0000-4000-8000-000000000401"
    db.add_note("gamma", "source", note_id=source)
    db.add_note("gamma", "target", note_id=target)
    source_fingerprint = content_fingerprint("gamma", "source")
    target_fingerprint = content_fingerprint("gamma", "target")
    with db.transaction() as conn:
        _set_graph_scope(db, conn)
        conn.execute(
            "INSERT INTO note_graph_suggestion_runs("
            "id,owner_user_id,dataset_id,source_note_id,source_fingerprint,"
            "provider,model,capability_revision,prompt_contract_version,"
            "state,revision,created_at,expires_at"
            ") VALUES (?,?,?,?,?,?,?,?,?,'succeeded',1,CURRENT_TIMESTAMP,CURRENT_TIMESTAMP)",
            (
                "run-400",
                db.client_id,
                DATASET_ID,
                source,
                source_fingerprint,
                "provider",
                "model",
                "capability-v1",
                "prompt-v1",
            ),
        )
        conn.execute(
            "INSERT INTO note_graph_suggestions("
            "id,run_id,owner_user_id,dataset_id,kind,source_note_id,source_fingerprint,"
            "target_note_id,target_fingerprint,state,revision,created_at,updated_at"
            ") VALUES (?,?,?,?,?,?,?,?,?,'rejected',1,CURRENT_TIMESTAMP,CURRENT_TIMESTAMP)",
            (
                "suggestion-400", "run-400", db.client_id, DATASET_ID, "related_note", source,
                source_fingerprint, target, target_fingerprint,
            ),
        )
    for number in range(101):
        db.add_keyword(f"gamma-tag-{number:03d}")

    result = SuggestionRetriever(db.note_graph_suggestion_store).retrieve(
        dataset_id=DATASET_ID, source_note_id=source
    )
    assert target not in [candidate.note_id for candidate in result.candidates]
    assert len(result.tag_catalog) == 100

    db.update_note(target, {"content": "target changed"}, expected_version=1)
    result = SuggestionRetriever(db.note_graph_suggestion_store).retrieve(
        dataset_id=DATASET_ID, source_note_id=source
    )
    assert target in [candidate.note_id for candidate in result.candidates]


def _exercise_scope_and_fts_drift(db: CharactersRAGDB) -> None:
    source = "00000000-0000-4000-8000-000000000500"
    db.add_note("delta", "source", note_id=source)
    with pytest.raises(RuntimeError, match="notes_graph_dataset_scope_invalid"):
        SuggestionRetriever(db.note_graph_suggestion_store).retrieve(
            dataset_id="dataset-not-authorized", source_note_id=source
        )

    with db.transaction() as conn:
        if db.note_graph_suggestion_store.is_postgres:
            conn.execute("DROP TRIGGER update_notes_fts_tsv_trigger ON notes")
            conn.execute(
                "CREATE TRIGGER update_notes_fts_tsv_trigger BEFORE UPDATE ON notes "
                "FOR EACH ROW EXECUTE FUNCTION update_notes_fts_tsv_function()"
            )
        else:
            conn.execute("DROP TRIGGER notes_au")
            conn.execute("CREATE TRIGGER notes_au AFTER UPDATE ON notes BEGIN SELECT 1; END")

    with pytest.raises(NotesGraphFTSNotReadyError, match="notes_graph_fts_not_ready"):
        SuggestionRetriever(db.note_graph_suggestion_store).retrieve(
            dataset_id=DATASET_ID, source_note_id=source
        )


def _assert_fix_round_one_behavior(db: CharactersRAGDB) -> None:
    _authorize_dataset(db)
    _exercise_ranked_shortlist_and_bounds(db)
    _exercise_exact_rejection_and_tag_limit(db)
    _exercise_scope_and_fts_drift(db)


def test_sqlite_retrieval_fix_round_one_contracts(tmp_path) -> None:
    db = CharactersRAGDB(str(tmp_path / "retrieval-fix-round-one.db"), client_id="owner-1")
    try:
        _assert_fix_round_one_behavior(db)
    finally:
        db.close_connection()


def test_postgres_retrieval_fix_round_one_contracts(pg_database_config) -> None:
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(":memory:", client_id="owner-1", backend=backend)
    try:
        _assert_fix_round_one_behavior(db)
    finally:
        db.close_connection()
