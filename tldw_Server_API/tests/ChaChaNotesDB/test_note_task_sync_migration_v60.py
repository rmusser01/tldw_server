"""SQLite schema-v60 contracts for owner/dataset-scoped Notes tasks."""

from __future__ import annotations

import inspect
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
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


def test_postgres_initializer_authority_remains_v59_until_postgres_v60_lands() -> None:
    assert CharactersRAGDB._POSTGRES_SCHEMA_VERSION == 59
    source = inspect.getsource(CharactersRAGDB._initialize_schema_postgres)
    assert "target_version = self._POSTGRES_SCHEMA_VERSION" in source


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


def _rewrite_sqlite_catalog(db_path: Path, *, object_name: str, old: str, new: str) -> None:
    with sqlite3.connect(db_path) as conn:
        current = conn.execute(
            "SELECT sql FROM sqlite_master WHERE name=?", (object_name,)
        ).fetchone()[0]
        assert old in current  # nosec B101 - proves the adversarial mutation was applied.
        conn.execute("PRAGMA writable_schema=ON")
        conn.execute(
            "UPDATE sqlite_master SET sql=? WHERE name=?",
            (current.replace(old, new, 1), object_name),
        )
        schema_version = int(conn.execute("PRAGMA schema_version").fetchone()[0])
        conn.execute(f"PRAGMA schema_version={schema_version + 1}")  # nosec B608
        conn.execute("PRAGMA writable_schema=OFF")


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
        task = upgraded.get_task(
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


def test_sqlite_v60_upgrade_derives_missing_event_note_from_task(tmp_path: Path) -> None:
    db_path = tmp_path / "task-v59-null-event-note.sqlite"
    _prepare_v59_database(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute("UPDATE task_events SET note_id=NULL WHERE id=?", (EVENT_ID,))

    upgraded = CharactersRAGDB(str(db_path), client_id=OWNER)
    try:
        with upgraded.transaction() as conn:
            row = conn.execute("SELECT note_id FROM task_events WHERE id=?", (EVENT_ID,)).fetchone()
            assert row[0] == NOTE_ID  # nosec B101
            note_column = next(
                column for column in conn.execute("PRAGMA table_xinfo(task_events)")
                if column[1] == "note_id"
            )
            assert note_column[3] == 1  # nosec B101 - NOT NULL is catalog authority.
    finally:
        upgraded.close_all_connections()


def test_sqlite_v60_upgrade_rejects_mismatched_event_task_and_note(tmp_path: Path) -> None:
    db_path = tmp_path / "task-v59-mismatched-event-note.sqlite"
    _prepare_v59_database(db_path)
    other_note_id = "44444444-4444-4444-8444-444444444444"
    original_version = CharactersRAGDB._CURRENT_SCHEMA_VERSION
    CharactersRAGDB._CURRENT_SCHEMA_VERSION = 59
    try:
        legacy = CharactersRAGDB(str(db_path), client_id=OWNER)
        assert legacy.add_note("Other", "Body", note_id=other_note_id) == other_note_id  # nosec B101
        with legacy.transaction() as conn:
            conn.execute("UPDATE task_events SET note_id=? WHERE id=?", (other_note_id, EVENT_ID))
        legacy.close_all_connections()
    finally:
        CharactersRAGDB._CURRENT_SCHEMA_VERSION = original_version

    with pytest.raises(CharactersRAGDBError, match="mismatched event parents"):
        CharactersRAGDB(str(db_path), client_id=OWNER)
    with sqlite3.connect(db_path) as conn:
        assert conn.execute(  # nosec B101
            "SELECT version FROM db_schema_version WHERE schema_name=?",
            (CharactersRAGDB._SCHEMA_NAME,),
        ).fetchone()[0] == 59


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


def test_sqlite_v60_verifier_rejects_same_name_weak_index(tmp_path: Path) -> None:
    db_path = tmp_path / "weak-index.sqlite"
    db = CharactersRAGDB(str(db_path), client_id=OWNER)
    db.close_all_connections()
    with sqlite3.connect(db_path) as conn:
        conn.execute("DROP INDEX idx_task_events_scope_task_page")
        conn.execute(
            "CREATE INDEX idx_task_events_scope_task_page ON task_events(owner_user_id)"
        )

    with pytest.raises(CharactersRAGDBError, match="index catalog drifted"):
        CharactersRAGDB(str(db_path), client_id=OWNER)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("note_id TEXT NOT NULL", "note_id BLOB NOT NULL"),
        ("CHECK(length(trim(note_id)) > 0)", "CHECK(1)"),
        ("ON UPDATE CASCADE ON DELETE RESTRICT", "ON UPDATE NO ACTION ON DELETE RESTRICT"),
    ],
    ids=("type", "check", "foreign-key-action"),
)
def test_sqlite_v60_verifier_rejects_table_catalog_drift(
    tmp_path: Path,
    old: str,
    new: str,
) -> None:
    db_path = tmp_path / f"table-drift-{old[:4]}.sqlite"
    db = CharactersRAGDB(str(db_path), client_id=OWNER)
    db.close_all_connections()
    _rewrite_sqlite_catalog(db_path, object_name="task_events", old=old, new=new)

    with pytest.raises(CharactersRAGDBError, match="table catalog drifted"):
        CharactersRAGDB(str(db_path), client_id=OWNER)


@pytest.mark.parametrize(
    "ddl",
    [
        (
            "CREATE TRIGGER unexpected_task_delete AFTER INSERT ON note_tasks "
            "BEGIN DELETE FROM note_tasks WHERE id = NEW.id; END"
        ),
        "CREATE VIEW unexpected_task_view AS SELECT id,note_id FROM note_tasks",
    ],
    ids=("trigger", "view"),
)
def test_sqlite_v60_verifier_rejects_related_trigger_or_view(
    tmp_path: Path,
    ddl: str,
) -> None:
    db_path = tmp_path / "related-catalog-drift.sqlite"
    db = CharactersRAGDB(str(db_path), client_id=OWNER)
    db.close_all_connections()
    with sqlite3.connect(db_path) as conn:
        conn.execute(ddl)

    with pytest.raises(CharactersRAGDBError, match="related catalog drifted"):
        CharactersRAGDB(str(db_path), client_id=OWNER)


@pytest.mark.parametrize("migrated", [False, True], ids=("fresh", "migrated"))
@pytest.mark.parametrize(
    ("filter_column", "filter_value", "expected_index"),
    [
        ("task_id", TASK_ID, "idx_task_events_scope_task_created"),
        ("note_id", NOTE_ID, "idx_task_events_scope_note_created"),
    ],
    ids=("task-page", "note-page"),
)
def test_sqlite_v60_activity_pages_use_scoped_created_indexes(
    tmp_path: Path,
    migrated: bool,
    filter_column: str,
    filter_value: str,
    expected_index: str,
) -> None:
    db_path = tmp_path / f"activity-plan-{migrated}-{filter_column}.sqlite"
    if migrated:
        _prepare_v59_database(db_path)
    db = CharactersRAGDB(str(db_path), client_id=OWNER)
    try:
        with db.transaction() as conn:
            plan = [
                str(row[3])
                for row in conn.execute(
                    "EXPLAIN QUERY PLAN SELECT * FROM task_events "
                    f"WHERE owner_user_id=? AND dataset_id=? AND {filter_column}=? "  # nosec B608
                    "ORDER BY created_at ASC,rowid ASC LIMIT ?",
                    (OWNER, LOCAL_UNBOUND, filter_value, 100),
                )
            ]
        assert any(expected_index in detail for detail in plan), plan  # nosec B101
        assert not any("TEMP B-TREE" in detail.upper() for detail in plan), plan  # nosec B101
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


def test_sqlite_v60_same_path_concurrent_openers_complete_one_migration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "concurrent-openers.sqlite"
    _prepare_v59_database(db_path)
    start = threading.Barrier(2)
    checkpoint_lock = threading.Lock()
    checkpoints: list[str] = []
    original_checkpoint = CharactersRAGDB._note_task_v60_migration_checkpoint

    def track_checkpoint(_db: CharactersRAGDB, stage: str) -> None:
        with checkpoint_lock:
            checkpoints.append(stage)
        original_checkpoint(stage)

    monkeypatch.setattr(
        CharactersRAGDB,
        "_note_task_v60_migration_checkpoint",
        track_checkpoint,
    )

    def open_same_path() -> int:
        start.wait(timeout=10)
        db = CharactersRAGDB(str(db_path), client_id=OWNER)
        try:
            with db.transaction() as conn:
                return db._get_db_version(conn)
        finally:
            db.close_all_connections()

    with ThreadPoolExecutor(max_workers=2) as executor:
        versions = list(executor.map(lambda _index: open_same_path(), range(2)))

    assert versions == [60, 60]  # nosec B101
    assert checkpoints == ["create", "copy", "index", "verify"]  # nosec B101
    with sqlite3.connect(db_path) as conn:
        tables = {str(row[0]) for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        assert conn.execute(  # nosec B101
            "SELECT version FROM db_schema_version WHERE schema_name=?",
            (CharactersRAGDB._SCHEMA_NAME,),
        ).fetchone()[0] == 60
        assert not any(name.endswith("_v60") for name in tables)  # nosec B101
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []  # nosec B101


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
