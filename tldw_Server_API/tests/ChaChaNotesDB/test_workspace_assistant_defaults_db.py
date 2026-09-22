"""Tests for Workspace Assistant Defaults schema and ChaChaNotes storage."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from pydantic import ValidationError

from tldw_Server_API.app.api.v1.schemas.workspace_schemas import WorkspaceAssistantDefaults
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, InputError


def _workspace_columns(db_path: Path) -> set[str]:
    with sqlite3.connect(str(db_path)) as conn:
        return {row[1] for row in conn.execute("PRAGMA table_info('workspaces')").fetchall()}


@pytest.fixture
def chacha_db(tmp_path: Path) -> CharactersRAGDB:
    """Provide a fresh ChaChaNotes DB for workspace assistant-default tests."""
    db = CharactersRAGDB(db_path=str(tmp_path / "chacha.db"), client_id="user-1")
    try:
        yield db
    finally:
        db.close_connection()


def test_workspace_assistant_defaults_accepts_persona_read_only() -> None:
    payload = WorkspaceAssistantDefaults(
        assistant_kind="persona",
        assistant_id="persona-1",
        persona_memory_mode="read_only",
    )

    assert payload.model_dump(exclude_none=True) == {
        "assistant_kind": "persona",
        "assistant_id": "persona-1",
        "persona_memory_mode": "read_only",
    }


def test_workspace_assistant_defaults_rejects_deferred_fields() -> None:
    with pytest.raises(ValidationError, match="voice must be null"):
        WorkspaceAssistantDefaults(
            assistant_kind="persona",
            assistant_id="persona-1",
            persona_memory_mode="read_only",
            voice={"provider": "openai"},
        )


def test_new_sqlite_db_has_workspace_assistant_defaults_column(
    chacha_db: CharactersRAGDB,
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "chacha.db"
    assert chacha_db is not None
    assert "assistant_defaults_json" in _workspace_columns(db_path)


def test_v48_sqlite_migration_adds_workspace_assistant_defaults_column(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "chacha.db"
    def initialize_historical(db: CharactersRAGDB) -> None:
        with db.transaction() as conn:
            db._apply_schema_v4(conn)
            steps = db._sqlite_linear_migration_steps()
            for version in range(4, 48):
                steps[version](conn)
                assert db._get_db_version(conn) == version + 1

    with monkeypatch.context() as patch:
        patch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 48)
        patch.setattr(CharactersRAGDB, "_initialize_schema", initialize_historical)
        seed = CharactersRAGDB(db_path, "historical-fixture")
    try:
        with seed.transaction() as conn:
            assert seed._get_db_version(conn) == 48
            assert "note_attachments" not in seed._sqlite_table_names(conn)
            # The maintained v35 workspace definition includes this later v49 addition.
            conn.execute("ALTER TABLE workspaces DROP COLUMN assistant_defaults_json")
            assert "assistant_defaults_json" not in {
                row["name"] for row in conn.execute("PRAGMA table_info(workspaces)")
            }
            conn.execute(
                "INSERT INTO workspaces (id, name, client_id) VALUES (?, ?, ?)",
                ("retained-workspace", "Historical workspace", seed.client_id),
            )
            before = dict(conn.execute("SELECT * FROM workspaces").fetchone())
            seed._migrate_from_v48_to_v49(conn)
            assert seed._get_db_version(conn) == 49
            after = dict(conn.execute("SELECT * FROM workspaces").fetchone())
            assert after == {**before, "assistant_defaults_json": None}
    finally:
        seed.close_all_connections()

    migrated = CharactersRAGDB(db_path=str(db_path), client_id="user-1")
    try:
        assert "assistant_defaults_json" in _workspace_columns(db_path)
        conn = migrated.get_connection()
        assert migrated._get_db_version(conn) == CharactersRAGDB._CURRENT_SCHEMA_VERSION
        current = dict(conn.execute("SELECT * FROM workspaces WHERE id = ?", (before["id"],)).fetchone())
        assert all(current[key] == value for key, value in after.items())
    finally:
        migrated.close_all_connections()


def test_workspace_assistant_defaults_round_trip_and_increment_version(chacha_db: CharactersRAGDB) -> None:
    created = chacha_db.upsert_workspace("ws-1", "Research")
    assert created["assistant_defaults_json"] is None

    assistant_defaults = {
        "assistant_kind": "persona",
        "assistant_id": "persona-1",
        "persona_memory_mode": "read_only",
    }
    updated = chacha_db.update_workspace(
        "ws-1",
        {"assistant_defaults_json": assistant_defaults},
        expected_version=created["version"],
    )

    assert updated["version"] == created["version"] + 1
    assert updated["assistant_defaults_json"] == assistant_defaults
    assert "persona_name" not in updated["assistant_defaults_json"]
    assert "source_persona" not in updated["assistant_defaults_json"]

    reloaded = chacha_db.get_workspace("ws-1")
    assert reloaded is not None
    assert reloaded["assistant_defaults_json"] == assistant_defaults


def test_workspace_assistant_defaults_clear_and_malformed_json_returns_none(chacha_db: CharactersRAGDB) -> None:
    created = chacha_db.upsert_workspace("ws-1", "Research")
    updated = chacha_db.update_workspace(
        "ws-1",
        {
            "assistant_defaults_json": {
                "assistant_kind": "persona",
                "assistant_id": "persona-1",
                "persona_memory_mode": "read_only",
            }
        },
        expected_version=created["version"],
    )
    cleared = chacha_db.update_workspace(
        "ws-1",
        {"assistant_defaults_json": None},
        expected_version=updated["version"],
    )
    assert cleared["assistant_defaults_json"] is None

    with chacha_db.transaction() as conn:
        conn.execute(
            "UPDATE workspaces SET assistant_defaults_json = ? WHERE id = ?",
            ("{bad-json", "ws-1"),
        )

    reloaded = chacha_db.get_workspace("ws-1")
    assert reloaded is not None
    assert reloaded["assistant_defaults_json"] is None


def test_workspace_assistant_defaults_update_rejects_invalid_json_string(chacha_db: CharactersRAGDB) -> None:
    created = chacha_db.upsert_workspace("ws-1", "Research")

    with pytest.raises(InputError, match="assistant_defaults_json must be valid JSON"):
        chacha_db.update_workspace(
            "ws-1",
            {"assistant_defaults_json": "{bad-json"},
            expected_version=created["version"],
        )


def test_workspace_update_rejects_unknown_only_payload(chacha_db: CharactersRAGDB) -> None:
    created = chacha_db.upsert_workspace("ws-1", "Research")

    with pytest.raises(InputError, match="No recognized workspace fields to update"):
        chacha_db.update_workspace(
            "ws-1",
            {"assistant_defaults": {"assistant_kind": "persona", "assistant_id": "persona-1"}},
            expected_version=created["version"],
        )
