from __future__ import annotations

from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.Notes_Graph.projection_service import (
    NoteGraphProjectionService,
)
from tldw_Server_API.app.services.notes_graph_projection_worker import (
    run_notes_graph_projection_maintenance_once,
)

pytestmark = pytest.mark.integration

SOURCE_ID = "11111111-1111-4111-8111-111111111111"
TARGET_ID = "22222222-2222-4222-8222-222222222222"
OTHER_ID = "33333333-3333-4333-8333-333333333333"


@pytest.fixture()
def graph_db(tmp_path: Path) -> CharactersRAGDB:
    db = CharactersRAGDB(str(tmp_path / "graph-maintenance.db"), client_id="owner-1")
    try:
        yield db
    finally:
        db.close_connection()


def test_normal_note_writes_update_projection_atomically_and_lifecycle_only_hides_edges(
    graph_db: CharactersRAGDB,
) -> None:
    store = graph_db.note_graph_projection_store
    before_revision = store.get_revision()

    graph_db.add_note("Source", f"[[id:{TARGET_ID}]]", note_id=SOURCE_ID)
    assert store.list_outgoing(SOURCE_ID) == (TARGET_ID,)
    assert store.count_dirty() == 0
    created_revision = store.get_revision()
    assert created_revision > before_revision

    graph_db.update_note(
        SOURCE_ID,
        {"content": f"[[id:{OTHER_ID}]]"},
        expected_version=1,
    )
    assert store.list_outgoing(SOURCE_ID) == (OTHER_ID,)
    assert store.count_dirty() == 0

    revision_before_trash = store.get_revision()
    assert graph_db.soft_delete_note(SOURCE_ID, expected_version=2)
    assert store.list_outgoing(SOURCE_ID) == (OTHER_ID,)
    assert store.get_revision() > revision_before_trash
    assert graph_db.restore_note(SOURCE_ID, expected_version=3)
    assert store.list_outgoing(SOURCE_ID) == (OTHER_ID,)


def test_note_and_projection_roll_back_together(
    graph_db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_projection(**_kwargs):
        raise RuntimeError("injected projection failure")

    monkeypatch.setattr(
        graph_db.note_graph_projection_store,
        "replace_projection",
        fail_projection,
    )
    with pytest.raises(RuntimeError, match="injected projection failure"):
        graph_db.add_note("Source", f"[[id:{TARGET_ID}]]", note_id=SOURCE_ID)

    assert graph_db.get_note_by_id(SOURCE_ID, include_deleted=True) is None


def test_direct_write_recovery_is_bounded_resumable_and_generation_safe(
    graph_db: CharactersRAGDB,
) -> None:
    graph_db.add_note("Source", "plain", note_id=SOURCE_ID)
    graph_db.execute_query(
        "UPDATE notes SET content = ?, version = version + 1 WHERE id = ?",
        (f"[[id:{TARGET_ID}]]", SOURCE_ID),
    )
    service = NoteGraphProjectionService(graph_db)

    assert service.process_dirty(limit=1) == 1
    assert graph_db.note_graph_projection_store.list_outgoing(SOURCE_ID) == (TARGET_ID,)
    assert graph_db.note_graph_projection_store.count_dirty() == 0
    assert service.process_dirty(limit=1) == 0


def test_owner_scoped_maintenance_entrypoint_processes_one_bounded_batch(
    graph_db: CharactersRAGDB,
) -> None:
    graph_db.add_note("Source", "plain", note_id=SOURCE_ID)
    graph_db.add_note("Other", "plain", note_id=OTHER_ID)
    graph_db.execute_query(
        "UPDATE notes SET content = ?, version = version + 1 WHERE id = ?",
        (f"[[id:{TARGET_ID}]]", SOURCE_ID),
    )
    graph_db.execute_query(
        "UPDATE notes SET content = content, version = version + 1 WHERE id = ?",
        (OTHER_ID,),
    )

    result = run_notes_graph_projection_maintenance_once(
        graph_db,
        batch_limit=1,
        rebuild_page_limit=1,
    )

    assert result == 1
    assert graph_db.note_graph_projection_store.count_dirty() == 1


def test_lifecycle_change_preserves_dirty_work_when_projection_state_is_missing(
    graph_db: CharactersRAGDB,
) -> None:
    graph_db.add_note("Source", "plain", note_id=SOURCE_ID)
    with graph_db.transaction() as conn:
        conn.execute("DELETE FROM note_wikilink_edges WHERE source_note_id = ?", (SOURCE_ID,))
        conn.execute("DELETE FROM note_graph_note_state WHERE note_id = ?", (SOURCE_ID,))
        conn.execute(
            "UPDATE notes SET content = ?, version = version + 1 WHERE id = ?",
            (f"[[id:{TARGET_ID}]]", SOURCE_ID),
        )

    assert graph_db.note_graph_projection_store.count_dirty() == 1
    assert graph_db.soft_delete_note(SOURCE_ID, expected_version=2)
    assert graph_db.note_graph_projection_store.get_note_state(SOURCE_ID) is None
    assert graph_db.note_graph_projection_store.count_dirty() == 1

    assert NoteGraphProjectionService(graph_db).process_dirty(limit=1) == 1
    state = graph_db.note_graph_projection_store.get_note_state(SOURCE_ID)
    assert state is not None
    assert state.source_version == 3
    assert graph_db.note_graph_projection_store.list_outgoing(SOURCE_ID) == (TARGET_ID,)
    assert graph_db.note_graph_projection_store.count_dirty() == 0


def test_unresolved_target_becomes_visible_without_reparse_and_parser_rebuild_resumes(
    graph_db: CharactersRAGDB,
) -> None:
    graph_db.add_note("Source", f"[[id:{TARGET_ID}]]", note_id=SOURCE_ID)
    store = graph_db.note_graph_projection_store
    state_before = store.get_note_state(SOURCE_ID)
    assert state_before is not None
    assert store.list_live_outgoing(SOURCE_ID) == ()

    graph_db.add_note("Target", "plain", note_id=TARGET_ID)
    assert store.list_live_outgoing(SOURCE_ID) == (TARGET_ID,)
    assert store.get_note_state(SOURCE_ID) == state_before

    service = NoteGraphProjectionService(graph_db, parser_version=2)
    assert service.prepare_rebuild() is True
    assert service.queue_rebuild_page(limit=1) == 1
    assert service.queue_rebuild_page(limit=1) == 1
    assert service.queue_rebuild_page(limit=1) == 0
    while service.process_dirty(limit=1):
        pass
    status = store.get_projection_status()
    assert status.parser_version == 2
    assert status.rebuild_state == "ready"
