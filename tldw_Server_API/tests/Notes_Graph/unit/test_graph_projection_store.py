from __future__ import annotations

from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.chacha.note_graph_projection_store import (
    NoteGraphProjectionStore,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    BackendType,
    CharactersRAGDB,
)
from tldw_Server_API.app.core.Notes.wikilinks import parse_wikilinks

pytestmark = pytest.mark.unit

SOURCE_ID = "11111111-1111-4111-8111-111111111111"
TARGET_ID = "22222222-2222-4222-8222-222222222222"
OTHER_ID = "33333333-3333-4333-8333-333333333333"


@pytest.fixture()
def graph_db(tmp_path: Path) -> CharactersRAGDB:
    db = CharactersRAGDB(str(tmp_path / "graph-projection.db"), client_id="owner-1")
    try:
        yield db
    finally:
        db.close_connection()


def test_projection_store_retains_unresolved_targets_and_clears_exact_generation(
    graph_db: CharactersRAGDB,
) -> None:
    graph_db.add_note("Source", f"[[id:{TARGET_ID}]]", note_id=SOURCE_ID)
    store = graph_db.note_graph_projection_store
    graph_db.execute_query(
        "UPDATE notes SET content = content, version = version + 1 WHERE id = ?",
        (SOURCE_ID,),
    )

    with graph_db.transaction() as conn:
        claim = store.claim_dirty(limit=1, conn=conn)[0]
        projection = parse_wikilinks(
            f"[[id:{TARGET_ID}]]",
            source_note_id=SOURCE_ID,
        )
        cleared = store.replace_projection(
            note_id=SOURCE_ID,
            source_version=2,
            projection=projection,
            claimed_generation=claim.generation,
            conn=conn,
        )

    assert cleared is True
    assert store.list_outgoing(SOURCE_ID) == (TARGET_ID,)
    assert store.get_note_state(SOURCE_ID).truncated is False
    assert store.count_dirty() == 0


def test_projection_store_generation_safe_completion_preserves_newer_dirty_work(
    graph_db: CharactersRAGDB,
) -> None:
    graph_db.add_note("Source", f"[[id:{TARGET_ID}]]", note_id=SOURCE_ID)
    store = graph_db.note_graph_projection_store
    graph_db.execute_query(
        "UPDATE notes SET content = content, version = version + 1 WHERE id = ?",
        (SOURCE_ID,),
    )
    with graph_db.transaction() as conn:
        stale_claim = store.claim_dirty(limit=1, conn=conn)[0]

    graph_db.execute_query(
        "UPDATE notes SET content = ?, version = version + 1 WHERE id = ?",
        (f"[[id:{OTHER_ID}]]", SOURCE_ID),
    )
    with graph_db.transaction() as conn:
        cleared = store.replace_projection(
            note_id=SOURCE_ID,
            source_version=1,
            projection=parse_wikilinks(
                f"[[id:{TARGET_ID}]]",
                source_note_id=SOURCE_ID,
            ),
            claimed_generation=stale_claim.generation,
            conn=conn,
        )

    assert cleared is False
    assert store.count_dirty() == 1
    assert store.claim_dirty(limit=1)[0].generation > stale_claim.generation


def test_projection_store_claim_is_bounded_and_deterministic(
    graph_db: CharactersRAGDB,
) -> None:
    for note_id in (OTHER_ID, SOURCE_ID, TARGET_ID):
        graph_db.add_note(note_id, "plain", note_id=note_id)
        graph_db.execute_query(
            "UPDATE notes SET content = content, version = version + 1 WHERE id = ?",
            (note_id,),
        )

    claims = graph_db.note_graph_projection_store.claim_dirty(limit=2)

    assert [claim.note_id for claim in claims] == sorted([OTHER_ID, SOURCE_ID, TARGET_ID])[:2]
    with pytest.raises(ValueError):
        graph_db.note_graph_projection_store.claim_dirty(limit=0)
    with pytest.raises(ValueError):
        graph_db.note_graph_projection_store.claim_dirty(limit=1_001)
    with graph_db.transaction() as conn, pytest.raises(ValueError):
        graph_db.note_graph_projection_store.queue_rebuild_page(limit=1_001, conn=conn)


def test_postgres_claim_is_owner_scoped_bounded_and_skip_locked() -> None:
    calls: list[tuple[str, tuple[object, ...]]] = []

    class _Cursor:
        @staticmethod
        def fetchall():
            return [{"note_id": SOURCE_ID, "generation": 3}]

    class _Connection:
        @staticmethod
        def execute(query: str, params: tuple[object, ...]):
            calls.append((query, params))
            return _Cursor()

    class _DB:
        backend_type = BackendType.POSTGRESQL
        client_id = "owner-1"

    claims = NoteGraphProjectionStore(_DB()).claim_dirty(
        limit=5,
        conn=_Connection(),
    )

    assert claims[0].generation == 3
    assert "owner_user_id = ?" in calls[0][0]
    assert "LIMIT ? FOR UPDATE SKIP LOCKED" in calls[0][0]
    assert calls[0][1] == ("owner-1", 5)


def test_postgres_scalar_projection_reads_use_mapping_columns() -> None:
    class _Cursor:
        def __init__(self, row: dict[str, object]) -> None:
            self._row = row

        def fetchone(self) -> dict[str, object]:
            return self._row

    class _DB:
        backend_type = BackendType.POSTGRESQL
        client_id = "owner-1"

        @staticmethod
        def execute_query(query: str, _params: tuple[object, ...] = ()) -> _Cursor:
            if "dirty_count" in query:
                return _Cursor({"dirty_count": 3})
            if "FROM note_graph_revisions" in query:
                return _Cursor({"revision": 7})
            return _Cursor(
                {
                    "parser_version": 2,
                    "rebuild_state": "running",
                    "rebuild_cursor": SOURCE_ID,
                }
            )

    store = NoteGraphProjectionStore(_DB())  # type: ignore[arg-type]

    assert store.count_dirty() == 3
    assert store.get_revision() == 7
    assert store.get_projection_status().parser_version == 2
    assert store.get_projection_status().rebuild_cursor == SOURCE_ID


def test_projection_source_read_is_owner_scoped_db_abstraction(
    graph_db: CharactersRAGDB,
) -> None:
    graph_db.add_note("Source", "body", note_id=SOURCE_ID)

    with graph_db.transaction() as conn:
        source = graph_db.note_graph_projection_store.get_projection_source(
            SOURCE_ID,
            conn=conn,
        )

    assert source is not None
    assert source.note_id == SOURCE_ID
    assert source.content == "body"
    assert source.version == 1


def test_orphan_query_plan_uses_endpoint_indexes_without_relationship_scans(
    graph_db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph_db.add_note("Source", "plain", note_id=SOURCE_ID)
    captured: list[tuple[str, tuple[object, ...]]] = []
    original_execute = graph_db.execute_query

    def execute(query: str, params: tuple[object, ...] = ()):
        captured.append((query, params))
        return original_execute(query, params)

    monkeypatch.setattr(graph_db, "execute_query", execute)
    assert graph_db.note_graph_projection_store.list_orphan_note_ids(
        after_note_id=None,
        limit=2,
    ) == (SOURCE_ID,)
    query, params = captured[-1]

    details = [str(row["detail"]) for row in original_execute(f"EXPLAIN QUERY PLAN {query}", params).fetchall()]

    assert not any("SCAN manual" in detail for detail in details)
    assert not any("SCAN derived" in detail for detail in details)
    assert any("idx_note_edges_from_live" in detail for detail in details)
    assert any("idx_note_edges_to_live" in detail for detail in details)
    assert any("idx_note_wikilink_edges_target" in detail for detail in details)
