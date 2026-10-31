import sqlite3

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

pytestmark = pytest.mark.unit


def test_sqlite_migration_adds_task_tables(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "tasks.db"
    note_id = "11111111-1111-4111-8111-111111111111"

    def initialize_historical(db: CharactersRAGDB) -> None:
        with db.transaction() as conn:
            db._apply_schema_v4(conn)
            steps = db._sqlite_linear_migration_steps()
            for version in range(4, 47):
                steps[version](conn)
                assert db._get_db_version(conn) == version + 1

    with monkeypatch.context() as patch:
        patch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 47)
        patch.setattr(CharactersRAGDB, "_initialize_schema", initialize_historical)
        seed = CharactersRAGDB(db_path, client_id="bootstrap")
    try:
        with seed.transaction() as conn:
            assert seed._get_db_version(conn) == 47
            tables = seed._sqlite_table_names(conn)
            assert {
                "note_tasks", "task_events", "task_event_read_state", "task_note_projections",
                "note_task_reconciliation_state", "note_task_scope_authority", "note_attachments",
                "note_graph_suggestion_evidence",
            }.isdisjoint(tables)
            conn.execute(
                "INSERT INTO notes (id, title, content, client_id) VALUES (?, ?, ?, ?)",
                (note_id, "Retained task note", "Original content", "bootstrap"),
            )
            note_before = dict(conn.execute("SELECT * FROM notes WHERE id = ?", (note_id,)).fetchone())
            seed._migrate_from_v47_to_v48(conn)
            assert seed._get_db_version(conn) == 48
            assert {
                "note_tasks", "task_events", "task_event_read_state", "task_note_projections",
                "note_task_reconciliation_state",
            } <= seed._sqlite_table_names(conn)
            assert dict(conn.execute("SELECT * FROM notes WHERE id = ?", (note_id,)).fetchone()) == note_before
            with pytest.raises(sqlite3.IntegrityError, match="CHECK constraint failed.*projection_status"):
                conn.execute(
                    "INSERT INTO note_tasks "
                    "(id, note_id, text, status, projection_status, created_at, updated_at, client_id) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    ("invalid-task", note_id, "Task", "open", "invalid", "2026-01-01", "2026-01-01", "bootstrap"),
                )
    finally:
        seed.close_all_connections()

    migrated = CharactersRAGDB(db_path=str(db_path), client_id="migrate")
    migrated.close_all_connections()

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        note_after = dict(conn.execute("SELECT * FROM notes WHERE id = ?", (note_id,)).fetchone())
        assert all(note_after[key] == value for key, value in note_before.items())
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
