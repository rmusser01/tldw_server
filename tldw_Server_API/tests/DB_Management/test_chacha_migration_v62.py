"""SQLite schema-v62 contracts for staged Workspace clone targets."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, SchemaError

pytestmark = pytest.mark.unit


def _initialize(path: Path) -> None:
    db = CharactersRAGDB(str(path), client_id="user-1")
    db.close_all_connections()


def test_sqlite_v62_fresh_schema_has_clone_markers_constraint_and_index(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "chacha-v62-fresh.sqlite"
    _initialize(db_path)

    with sqlite3.connect(db_path) as conn:
        version = conn.execute(
            "SELECT version FROM db_schema_version WHERE schema_name = ?",
            (CharactersRAGDB._SCHEMA_NAME,),
        ).fetchone()[0]
        columns = {str(row[1]): str(row[2]).upper() for row in conn.execute("PRAGMA table_info(workspaces)")}
        index_columns = [
            str(row[2])
            for row in conn.execute("PRAGMA index_info(idx_workspaces_system_operation)")
        ]
        conn.execute(
            "INSERT INTO workspaces(id, name, system_operation_state) VALUES (?, ?, ?)",
            ("workspace-valid", "Valid", "staged"),
        )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO workspaces(id, name, system_operation_state) VALUES (?, ?, ?)",
                ("workspace-invalid", "Invalid", "published"),
            )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO workspaces(id, name, system_operation_kind) VALUES (?, ?, ?)",
                ("workspace-invalid-kind", "Invalid Kind", "other_operation"),
            )

    assert version == 62
    assert columns["system_operation_id"] == "TEXT"
    assert columns["system_operation_kind"] == "TEXT"
    assert columns["system_operation_state"] == "TEXT"
    assert columns["system_request_fingerprint"] == "TEXT"
    assert index_columns == [
        "system_operation_kind",
        "system_operation_state",
        "system_operation_id",
    ]


def test_sqlite_v61_upgrade_preserves_workspace_and_is_rerunnable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "chacha-v61-upgrade.sqlite"
    monkeypatch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 61)
    db = CharactersRAGDB(str(db_path), client_id="user-1")
    db.close_all_connections()
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO workspaces(id, name, client_id) VALUES (?, ?, ?)",
            ("workspace-existing", "Existing Workspace", "user-1"),
        )

    monkeypatch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 62)
    _initialize(db_path)
    _initialize(db_path)

    with sqlite3.connect(db_path) as conn:
        version = conn.execute(
            "SELECT version FROM db_schema_version WHERE schema_name = ?",
            (CharactersRAGDB._SCHEMA_NAME,),
        ).fetchone()[0]
        workspace = conn.execute(
            "SELECT name, system_operation_state FROM workspaces WHERE id = ?",
            ("workspace-existing",),
        ).fetchone()

    assert version == 62
    assert workspace == ("Existing Workspace", None)


def test_sqlite_v62_failure_rolls_back_partial_ddl_and_retry_succeeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "chacha-v62-interrupted.sqlite"
    monkeypatch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 61)
    db = CharactersRAGDB(str(db_path), client_id="user-1")
    original_sql = CharactersRAGDB._MIGRATION_SQL_V61_TO_V62
    injected_sql = original_sql.replace(
        "ALTER TABLE workspaces ADD COLUMN system_operation_state TEXT",
        "SELECT * FROM injected_v62_failure;\n"
        "ALTER TABLE workspaces ADD COLUMN system_operation_state TEXT",
        1,
    )

    try:
        monkeypatch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 62)
        monkeypatch.setattr(CharactersRAGDB, "_MIGRATION_SQL_V61_TO_V62", injected_sql)

        with pytest.raises(SchemaError, match="Workspace clone lifecycle v62 SQLite migration failed"):
            db._initialize_schema_sqlite()

        conn = db.get_connection()
        version_after_failure = conn.execute(
            "SELECT version FROM db_schema_version WHERE schema_name = ?",
            (CharactersRAGDB._SCHEMA_NAME,),
        ).fetchone()[0]
        columns_after_failure = {str(row[1]) for row in conn.execute("PRAGMA table_info(workspaces)")}
        indexes_after_failure = {
            str(row[1]) for row in conn.execute("PRAGMA index_list(workspaces)")
        }

        assert version_after_failure == 61
        assert not {
            "system_operation_id",
            "system_operation_kind",
            "system_operation_state",
            "system_request_fingerprint",
        }.intersection(columns_after_failure)
        assert "idx_workspaces_system_operation" not in indexes_after_failure

        monkeypatch.setattr(CharactersRAGDB, "_MIGRATION_SQL_V61_TO_V62", original_sql)
        db._initialize_schema_sqlite()

        version_after_retry = conn.execute(
            "SELECT version FROM db_schema_version WHERE schema_name = ?",
            (CharactersRAGDB._SCHEMA_NAME,),
        ).fetchone()[0]
        columns_after_retry = {str(row[1]) for row in conn.execute("PRAGMA table_info(workspaces)")}
        indexes_after_retry = {str(row[1]) for row in conn.execute("PRAGMA index_list(workspaces)")}

        assert version_after_retry == 62
        assert {
            "system_operation_id",
            "system_operation_kind",
            "system_operation_state",
            "system_request_fingerprint",
        } <= columns_after_retry
        assert "idx_workspaces_system_operation" in indexes_after_retry
    finally:
        db.close_all_connections()
