"""Tests for Workspace Assistant Defaults schema and ChaChaNotes storage."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from pydantic import ValidationError

from tldw_Server_API.app.api.v1.schemas.workspace_schemas import WorkspaceAssistantDefaults
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


def _workspace_columns(db_path: Path) -> set[str]:
    with sqlite3.connect(str(db_path)) as conn:
        return {row[1] for row in conn.execute("PRAGMA table_info('workspaces')").fetchall()}


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


def test_new_sqlite_db_has_workspace_assistant_defaults_column(tmp_path: Path) -> None:
    db_path = tmp_path / "chacha.db"
    db = CharactersRAGDB(db_path=str(db_path), client_id="user-1")
    try:
        assert "assistant_defaults_json" in _workspace_columns(db_path)
    finally:
        db.close_connection()


def test_v48_sqlite_migration_adds_workspace_assistant_defaults_column(tmp_path: Path) -> None:
    db_path = tmp_path / "chacha.db"
    db = CharactersRAGDB(db_path=str(db_path), client_id="user-1")
    db.close_connection()

    with sqlite3.connect(str(db_path)) as conn:
        columns = _workspace_columns(db_path)
        if "assistant_defaults_json" in columns:
            conn.execute("ALTER TABLE workspaces DROP COLUMN assistant_defaults_json")
        conn.execute(
            "UPDATE db_schema_version SET version = ? WHERE schema_name = ?",
            (48, CharactersRAGDB._SCHEMA_NAME),
        )
        conn.commit()

    migrated = CharactersRAGDB(db_path=str(db_path), client_id="user-1")
    try:
        assert "assistant_defaults_json" in _workspace_columns(db_path)
        version_row = migrated.execute_query(
            "SELECT version FROM db_schema_version WHERE schema_name = ?",
            (CharactersRAGDB._SCHEMA_NAME,),
        ).fetchone()
        assert version_row["version"] == CharactersRAGDB._CURRENT_SCHEMA_VERSION
    finally:
        migrated.close_connection()


def test_workspace_assistant_defaults_round_trip_and_increment_version(tmp_path: Path) -> None:
    db = CharactersRAGDB(db_path=str(tmp_path / "chacha.db"), client_id="user-1")
    try:
        created = db.upsert_workspace("ws-1", "Research")
        assert created["assistant_defaults_json"] is None

        assistant_defaults = {
            "assistant_kind": "persona",
            "assistant_id": "persona-1",
            "persona_memory_mode": "read_only",
        }
        updated = db.update_workspace(
            "ws-1",
            {"assistant_defaults_json": assistant_defaults},
            expected_version=created["version"],
        )

        assert updated["version"] == created["version"] + 1
        assert updated["assistant_defaults_json"] == assistant_defaults
        assert "persona_name" not in json.dumps(updated["assistant_defaults_json"])
        assert "source_persona" not in json.dumps(updated["assistant_defaults_json"])

        reloaded = db.get_workspace("ws-1")
        assert reloaded is not None
        assert reloaded["assistant_defaults_json"] == assistant_defaults
    finally:
        db.close_connection()


def test_workspace_assistant_defaults_clear_and_malformed_json_returns_none(tmp_path: Path) -> None:
    db = CharactersRAGDB(db_path=str(tmp_path / "chacha.db"), client_id="user-1")
    try:
        created = db.upsert_workspace("ws-1", "Research")
        updated = db.update_workspace(
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
        cleared = db.update_workspace(
            "ws-1",
            {"assistant_defaults_json": None},
            expected_version=updated["version"],
        )
        assert cleared["assistant_defaults_json"] is None

        with db.transaction() as conn:
            conn.execute(
                "UPDATE workspaces SET assistant_defaults_json = ? WHERE id = ?",
                ("{bad-json", "ws-1"),
            )

        reloaded = db.get_workspace("ws-1")
        assert reloaded is not None
        assert reloaded["assistant_defaults_json"] is None
    finally:
        db.close_connection()
