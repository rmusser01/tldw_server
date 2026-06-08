import sqlite3

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


pytestmark = pytest.mark.unit


def test_sqlite_migration_adds_task_tables(tmp_path) -> None:
    db_path = tmp_path / "tasks.db"
    db = CharactersRAGDB(db_path=str(db_path), client_id="bootstrap")
    db.close_connection()

    with sqlite3.connect(db_path) as conn:
        for table in (
            "task_event_read_state",
            "task_note_projections",
            "task_events",
            "note_task_reconciliation_state",
            "note_tasks",
            "tasks",
        ):
            conn.execute(f"DROP TABLE IF EXISTS {table}")  # nosec B608 - test-only fixed table list
        conn.execute(
            "UPDATE db_schema_version SET version = ? WHERE schema_name = ?",
            (47, CharactersRAGDB._SCHEMA_NAME),
        )
        conn.commit()

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
    } <= tables
    assert "tasks" not in tables  # nosec B101
    assert final_version == CharactersRAGDB._CURRENT_SCHEMA_VERSION  # nosec B101
    assert note_tasks_row is not None  # nosec B101
    assert projection_row is not None  # nosec B101
    note_tasks_sql = note_tasks_row[0]
    projection_sql = projection_row[0]
    assert "projection_status IN ('live','unlinked','ambiguous','deleted')" in note_tasks_sql  # nosec B101
    assert "projection_status IN ('live','unlinked','ambiguous','deleted')" in projection_sql  # nosec B101
