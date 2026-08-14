"""SQLite schema-v60 contracts for owner/dataset-scoped Notes tasks."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    SchemaError,
)

pytestmark = pytest.mark.unit

OWNER = "task-v60-owner"
LOCAL_UNBOUND = "local-unbound"
NOTE_ID = "11111111-1111-4111-8111-111111111111"
TASK_ID = "22222222-2222-4222-8222-222222222222"
EVENT_ID = "33333333-3333-4333-8333-333333333333"


def _prepare_v59_database(db_path: Path) -> None:
    """Create the preceding real schema and one complete legacy task graph."""

    original_version = CharactersRAGDB._CURRENT_SCHEMA_VERSION
    CharactersRAGDB._CURRENT_SCHEMA_VERSION = 59
    try:
        db = CharactersRAGDB(str(db_path), client_id=OWNER)
        note_id = db.add_note("Task parent", "- [ ] Review source\n", note_id=NOTE_ID)
        assert note_id == NOTE_ID  # nosec B101
        now = "2026-08-13T00:00:00+00:00"
        with db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO note_tasks(
                    id, note_id, text, status, metadata_json, projection_status, deleted,
                    created_at, updated_at, completed_at, client_id, version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    TASK_ID,
                    NOTE_ID,
                    "Review source",
                    "open",
                    '{"due_date":"2026-08-15","estimate":"30m","priority":"high"}',
                    "live",
                    0,
                    now,
                    now,
                    None,
                    OWNER,
                    1,
                ),
            )
            conn.execute(
                """
                INSERT INTO task_note_projections(
                    task_id, note_id, note_version, line_number, start_offset, end_offset,
                    normalized_text_hash, occurrence_index, block_fingerprint, raw_line,
                    has_child_content, projection_status, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    TASK_ID,
                    NOTE_ID,
                    1,
                    1,
                    0,
                    19,
                    "sha256:" + "a" * 64,
                    0,
                    "sha256:" + "b" * 64,
                    "- [ ] Review source",
                    0,
                    "live",
                    now,
                ),
            )
            conn.execute(
                """
                INSERT INTO task_events(
                    id, task_id, note_id, event_type, actor_type, actor_id, tool_name,
                    policy_mode, approval_id, old_value_json, new_value_json, created_at,
                    client_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    EVENT_ID,
                    TASK_ID,
                    NOTE_ID,
                    "created",
                    "user",
                    OWNER,
                    None,
                    None,
                    None,
                    None,
                    '{"status":"open","text":"Review source"}',
                    now,
                    OWNER,
                ),
            )
            conn.execute(
                "INSERT INTO task_event_read_state(event_id, user_id, read_at) VALUES (?, ?, ?)",
                (EVENT_ID, OWNER, now),
            )
            conn.execute(
                """
                INSERT INTO note_task_reconciliation_state(
                    note_id, note_version, status, reconciled_at, item_count, warning_count,
                    cursor
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (NOTE_ID, 1, "clean", now, 1, 0, None),
            )
        db.close_all_connections()
    finally:
        CharactersRAGDB._CURRENT_SCHEMA_VERSION = original_version


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")}  # nosec B608


def test_sqlite_v60_upgrade_preserves_graph_under_local_unbound_scope(tmp_path: Path) -> None:
    db_path = tmp_path / "task-v59.sqlite"
    _prepare_v59_database(db_path)

    upgraded = CharactersRAGDB(str(db_path), client_id=OWNER)
    try:
        with upgraded.transaction() as conn:
            assert upgraded._get_db_version(conn) == 60  # nosec B101
            assert conn.execute("PRAGMA foreign_key_check").fetchall() == []  # nosec B101
            counts = {
                table: int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])  # nosec B608
                for table in (
                    "note_tasks",
                    "task_note_projections",
                    "task_events",
                    "task_event_read_state",
                    "note_task_reconciliation_state",
                    "task_projection_drifts",
                )
            }
            assert counts == {  # nosec B101
                "note_tasks": 1,
                "task_note_projections": 1,
                "task_events": 1,
                "task_event_read_state": 1,
                "note_task_reconciliation_state": 1,
                "task_projection_drifts": 0,
            }
        task = upgraded.get_task_scoped(
            owner_user_id=OWNER,
            dataset_id=LOCAL_UNBOUND,
            task_id=TASK_ID,
            include_deleted=True,
        )
        assert task is not None  # nosec B101
        assert task["id"] == TASK_ID  # nosec B101
        assert task["canonical_revision"] == task["version"] == 1  # nosec B101
        assert task["canonical_hash"].startswith("sha256:")  # nosec B101
    finally:
        upgraded.close_all_connections()


def test_sqlite_v60_has_exact_scoped_task_graph_columns(tmp_path: Path) -> None:
    db = CharactersRAGDB(str(tmp_path / "fresh.sqlite"), client_id=OWNER)
    try:
        with db.transaction() as conn:
            assert {"owner_user_id", "dataset_id", "canonical_revision", "canonical_hash"} <= _table_columns(  # nosec B101
                conn, "note_tasks"
            )
            assert {
                "owner_user_id",
                "dataset_id",
                "sync_revision",
                "sync_object_hash",
                "sync_server_cursor",
                "source_device_id",
                "client_occurred_at",
                "source_kind",
                "corrects_activity_id",
                "deleted",
                "deleted_at",
                "delete_reason",
            } <= _table_columns(conn, "task_events")  # nosec B101
            for table in (
                "task_note_projections",
                "task_event_read_state",
                "note_task_reconciliation_state",
                "task_projection_drifts",
            ):
                assert {"owner_user_id", "dataset_id"} <= _table_columns(conn, table)  # nosec B101
            foreign_keys = conn.execute("PRAGMA foreign_key_list(task_note_projections)").fetchall()
            assert any(str(row[5]).upper() == "CASCADE" for row in foreign_keys)  # nosec B101
    finally:
        db.close_all_connections()


@pytest.mark.parametrize("stage", ["create", "copy", "index", "verify"])
def test_sqlite_v60_injected_failure_rolls_back_original_graph_and_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    db_path = tmp_path / f"rollback-{stage}.sqlite"
    _prepare_v59_database(db_path)

    def fail_at_stage(self: CharactersRAGDB, current_stage: str) -> None:
        if current_stage == stage:
            raise SchemaError(f"injected v60 {stage} failure")

    monkeypatch.setattr(
        CharactersRAGDB,
        "_note_task_v60_migration_checkpoint",
        fail_at_stage,
        raising=False,
    )
    with pytest.raises(CharactersRAGDBError, match=f"injected v60 {stage} failure"):
        CharactersRAGDB(str(db_path), client_id=OWNER)

    with sqlite3.connect(db_path) as conn:
        assert conn.execute(  # nosec B101
            "SELECT version FROM db_schema_version WHERE schema_name = ?",
            (CharactersRAGDB._SCHEMA_NAME,),
        ).fetchone()[0] == 59
        assert "owner_user_id" not in _table_columns(conn, "note_tasks")  # nosec B101
        assert conn.execute("SELECT id FROM note_tasks").fetchone()[0] == TASK_ID  # nosec B101
        assert conn.execute("SELECT id FROM task_events").fetchone()[0] == EVENT_ID  # nosec B101


def test_sqlite_v60_rejects_target_table_collision_without_version_change(tmp_path: Path) -> None:
    db_path = tmp_path / "collision.sqlite"
    _prepare_v59_database(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE task_projection_drifts(id TEXT PRIMARY KEY)")

    with pytest.raises(CharactersRAGDBError, match="collision"):
        CharactersRAGDB(str(db_path), client_id=OWNER)

    with sqlite3.connect(db_path) as conn:
        assert conn.execute(  # nosec B101
            "SELECT version FROM db_schema_version WHERE schema_name = ?",
            (CharactersRAGDB._SCHEMA_NAME,),
        ).fetchone()[0] == 59
        assert "owner_user_id" not in _table_columns(conn, "note_tasks")  # nosec B101


@pytest.mark.parametrize(
    ("column", "value"),
    [("canonical_revision", 1.5), ("version", 1.5), ("deleted", 1.5)],
)
def test_sqlite_v60_rejects_noninteger_task_storage_classes(
    tmp_path: Path,
    column: str,
    value: object,
) -> None:
    db = CharactersRAGDB(str(tmp_path / f"integer-{column}.sqlite"), client_id=OWNER)
    try:
        note_id = db.add_note("Task parent", "Body", note_id=NOTE_ID)
        assert note_id == NOTE_ID  # nosec B101
        with pytest.raises(sqlite3.IntegrityError), db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO note_tasks(
                    owner_user_id, dataset_id, id, note_id, text, status, metadata_json,
                    projection_status, deleted, created_at, updated_at, completed_at,
                    client_id, version, canonical_revision, canonical_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,  # nosec B608 - column is test-parametrized from a fixed set.
                (
                    OWNER,
                    LOCAL_UNBOUND,
                    TASK_ID,
                    NOTE_ID,
                    "Review source",
                    "open",
                    "{}",
                    "live",
                    value if column == "deleted" else 0,
                    "2026-08-13T00:00:00+00:00",
                    "2026-08-13T00:00:00+00:00",
                    None,
                    OWNER,
                    value if column == "version" else 1,
                    value if column == "canonical_revision" else 1,
                    "sha256:" + "a" * 64,
                ),
            )
    finally:
        db.close_all_connections()
