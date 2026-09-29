import sqlite3

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

pytestmark = pytest.mark.unit


def test_sqlite_migration_adds_task_tables(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "tasks.db"
    # Seed a genuine v47 DB. Dropping the task tables from a current-schema DB is
    # not a real predecessor: later steps depend on the catalog they left behind
    # (v55->v56's note_graph_suggestion_evidence FK needs uq_notes_owner_id).
    with monkeypatch.context() as patch:
        patch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 47)
        db = CharactersRAGDB(db_path=str(db_path), client_id="bootstrap")
        db.close_connection()

    with sqlite3.connect(db_path) as conn:
        assert conn.execute(  # nosec B101
            "SELECT version FROM db_schema_version WHERE schema_name = ?",
            (CharactersRAGDB._SCHEMA_NAME,),
        ).fetchone()[0] == 47
        assert conn.execute(  # nosec B101
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'note_tasks'"
        ).fetchone() is None

    migrated = CharactersRAGDB(db_path=str(db_path), client_id="migrate")
    migrated.close_connection()

    with sqlite3.connect(db_path) as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        final_version = conn.execute(
            "SELECT version FROM db_schema_version WHERE schema_name = ?",
            (CharactersRAGDB._SCHEMA_NAME,),
        ).fetchone()[0]
        note_tasks_row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'note_tasks'"
        ).fetchone()
        projection_row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'task_note_projections'"
        ).fetchone()
    assert {  # nosec B101
        "note_tasks",
        "task_events",
        "task_event_read_state",
        "task_note_projections",
        "note_task_reconciliation_state",
        "note_task_scope_authority",
    } <= tables
    assert "tasks" not in tables  # nosec B101
    assert final_version == CharactersRAGDB._CURRENT_SCHEMA_VERSION  # nosec B101
    assert note_tasks_row is not None  # nosec B101
    assert projection_row is not None  # nosec B101
    note_tasks_sql = note_tasks_row[0]
    projection_sql = projection_row[0]
    assert "projection_status IN ('live','unlinked','ambiguous','deleted')" in note_tasks_sql  # nosec B101
    assert "projection_status IN ('live','unlinked','ambiguous','deleted')" in projection_sql  # nosec B101
