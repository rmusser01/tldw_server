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
            "tasks",
        ):
            conn.execute(f"DROP TABLE IF EXISTS {table}")  # nosec B608 - test-only fixed table list
        conn.execute(
            "UPDATE db_schema_version SET version = ? WHERE schema_name = ?",
            (CharactersRAGDB._CURRENT_SCHEMA_VERSION - 1, CharactersRAGDB._SCHEMA_NAME),
        )
        conn.commit()

    migrated = CharactersRAGDB(db_path=str(db_path), client_id="migrate")
    migrated.close_connection()

    with sqlite3.connect(db_path) as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert {  # nosec B101
        "tasks",
        "task_events",
        "task_event_read_state",
        "task_note_projections",
        "note_task_reconciliation_state",
    } <= tables
