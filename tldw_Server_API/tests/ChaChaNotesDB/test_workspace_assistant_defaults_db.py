"""Tests for Workspace Assistant Defaults schema and ChaChaNotes storage."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator
from contextlib import closing
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from tldw_Server_API.app.api.v1.schemas.workspace_schemas import WorkspaceAssistantDefaults
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, InputError


def _workspace_columns(db_path: Path) -> set[str]:
    """Read workspace column names without bootstrapping or repairing the schema."""
    with closing(sqlite3.connect(db_path)) as conn:
        return {row[1] for row in conn.execute("PRAGMA table_info('workspaces')").fetchall()}


@pytest.fixture
def chacha_db(tmp_path: Path) -> Iterator[CharactersRAGDB]:
    """Provide a fresh ChaChaNotes DB for workspace assistant-default tests."""
    db = CharactersRAGDB(db_path=str(tmp_path / "chacha.db"), client_id="user-1")
    try:
        yield db
    finally:
        db.close_connection()


@pytest.fixture
def workspace_with_stored_assistant_defaults() -> Callable[..., dict[str, Any]]:
    """Seed raw storage here so API projection tests need no SQL outside the DB test layer."""

    def create(
        db: CharactersRAGDB,
        stored_value: Any,
    ) -> dict[str, Any]:
        """Return a normally loaded workspace after bypassing storage write validation."""
        workspace = db.upsert_workspace("ws-stored-defaults", "Stored defaults")
        with db.transaction() as conn:
            conn.execute(
                "UPDATE workspaces SET assistant_defaults_json = ? WHERE id = ?",
                (stored_value, workspace["id"]),
            )
        loaded = db.get_workspace(workspace["id"])
        assert loaded is not None
        return loaded

    return create


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


def test_v48_sqlite_migration_adds_workspace_assistant_defaults_column(tmp_path: Path) -> None:
    """Exercise v48->v49 alone so later schema repair cannot mask a missing column migration."""
    db_path = tmp_path / "v48-workspace.db"
    migration = object.__new__(CharactersRAGDB)
    migration.db_path_str = str(db_path)
    with closing(sqlite3.connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("CREATE TABLE db_schema_version (schema_name TEXT PRIMARY KEY, version INTEGER NOT NULL)")
        conn.execute("CREATE TABLE workspaces (id TEXT PRIMARY KEY, name TEXT NOT NULL)")
        conn.execute("INSERT INTO workspaces (id, name) VALUES ('ws-legacy', 'Legacy research')")
        conn.execute(
            "INSERT INTO db_schema_version (schema_name, version) VALUES (?, 48)",
            (CharactersRAGDB._SCHEMA_NAME,),
        )
        conn.commit()
        assert "assistant_defaults_json" not in _workspace_columns(db_path)

        migration._migrate_from_v48_to_v49(conn)

        assert "assistant_defaults_json" in _workspace_columns(db_path)
        assert migration._get_db_version(conn) == 49
        assert dict(conn.execute("SELECT * FROM workspaces").fetchone()) == {
            "id": "ws-legacy",
            "name": "Legacy research",
            "assistant_defaults_json": None,
        }


def test_historical_v4_sqlite_full_upgrade_preserves_data_and_adds_workspace_defaults(tmp_path: Path) -> None:
    """Upgrade the retained v4 schema, not a current database with a rewound version marker."""
    db_path = tmp_path / "historical-v4.db"
    with closing(sqlite3.connect(db_path)) as conn:
        conn.executescript(CharactersRAGDB._FULL_SCHEMA_SQL_V4)
        conn.execute(
            "INSERT INTO notes (id, title, content, client_id) VALUES (?, ?, ?, ?)",
            ("legacy-note", "Historical research", "Keep this research", "user-1"),
        )
        conn.commit()
        assert (
            conn.execute(
                "SELECT version FROM db_schema_version WHERE schema_name = ?",
                (CharactersRAGDB._SCHEMA_NAME,),
            ).fetchone()[0]
            == 4
        )
        assert (
            conn.execute("SELECT name FROM sqlite_master WHERE name IN ('workspaces', 'note_attachments')").fetchall()
            == []
        )

    migrated = CharactersRAGDB(db_path=str(db_path), client_id="user-1")
    try:
        assert "assistant_defaults_json" in _workspace_columns(db_path)
        version_row = migrated.execute_query(
            "SELECT version FROM db_schema_version WHERE schema_name = ?",
            (CharactersRAGDB._SCHEMA_NAME,),
        ).fetchone()
        assert version_row["version"] == CharactersRAGDB._CURRENT_SCHEMA_VERSION
        note = migrated.get_note_by_id("legacy-note")
        assert note["title"] == "Historical research"
        assert note["content"] == "Keep this research"
        created = migrated.upsert_workspace("ws-upgraded", "Upgraded research")
        defaults = {"assistant_kind": "persona", "assistant_id": "persona-1"}
        migrated.update_workspace(
            created["id"], {"assistant_defaults_json": defaults}, expected_version=created["version"]
        )
        assert migrated.get_workspace(created["id"])["assistant_defaults_json"] == defaults
    finally:
        migrated.close_connection()


def test_workspace_assistant_defaults_round_trip_and_increment_version(chacha_db: CharactersRAGDB) -> None:
    """Persist only reference defaults and increment the workspace version."""
    created = chacha_db.upsert_workspace("ws-1", "Research")
    assert created["assistant_defaults_json"] is None
    assert created["_assistant_defaults_invalid"] is False

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
    assert reloaded["_assistant_defaults_invalid"] is False


@pytest.mark.parametrize(
    "stored_value",
    [
        "{private-malformed-marker",
        "null",
        "[]",
        '["private-marker"]',
        '"private-marker"',
        "42",
        "0",
        "true",
        "false",
        "",
        " \t ",
    ],
)
def test_workspace_get_and_list_preserve_malformed_storage_diagnostic(
    chacha_db: CharactersRAGDB,
    workspace_with_stored_assistant_defaults: Callable[..., dict[str, Any]],
    stored_value: str,
) -> None:
    """Distinguish persisted non-objects from SQL NULL without returning the raw payload."""
    workspace = workspace_with_stored_assistant_defaults(chacha_db, stored_value)
    listed = chacha_db.list_workspaces()

    assert workspace["assistant_defaults_json"] is None
    assert workspace["_assistant_defaults_invalid"] is True
    assert listed == [workspace]
    assert "private-marker" not in repr(workspace)
    assert "private-malformed-marker" not in repr(workspace)
    assert (
        chacha_db.execute_query(
            "SELECT assistant_defaults_json FROM workspaces WHERE id = ?", (workspace["id"],)
        ).fetchone()[0]
        == stored_value
    )


@pytest.mark.parametrize("replacement", [None, {"assistant_kind": "persona", "assistant_id": "persona-1"}])
def test_workspace_malformed_defaults_repair_or_clear_recomputes_diagnostic(
    chacha_db: CharactersRAGDB,
    workspace_with_stored_assistant_defaults: Callable[..., dict[str, Any]],
    replacement: dict[str, Any] | None,
) -> None:
    """Repair or clear corrupt storage, ignoring a stale private flag supplied in an update."""
    workspace = workspace_with_stored_assistant_defaults(chacha_db, "{bad-json")
    assert workspace["_assistant_defaults_invalid"] is True
    updated = chacha_db.update_workspace(
        workspace["id"],
        {"assistant_defaults_json": replacement, "_assistant_defaults_invalid": True},
        expected_version=workspace["version"],
    )
    assert updated["version"] == workspace["version"] + 1
    assert updated["assistant_defaults_json"] == replacement
    assert updated["_assistant_defaults_invalid"] is False
    chacha_db.close_connection()
    assert chacha_db.get_workspace(workspace["id"]) == updated
    assert chacha_db.list_workspaces() == [updated]
    raw = dict(chacha_db.execute_query("SELECT * FROM workspaces WHERE id = ?", (workspace["id"],)).fetchone())
    assert "_assistant_defaults_invalid" not in raw
    if replacement is None:
        assert raw["assistant_defaults_json"] is None
    assert "_assistant_defaults_invalid" not in repr(chacha_db.get_sync_log_entries())


def test_workspace_unrelated_update_preserves_malformed_diagnostic(
    chacha_db: CharactersRAGDB,
    workspace_with_stored_assistant_defaults: Callable[..., dict[str, Any]],
) -> None:
    """Editing an unrelated setting must neither clear corruption nor persist its private flag."""
    workspace = workspace_with_stored_assistant_defaults(chacha_db, "{bad-json")
    updated = chacha_db.update_workspace(workspace["id"], {"name": "Renamed"}, expected_version=workspace["version"])
    assert updated["_assistant_defaults_invalid"] is True
    assert updated["assistant_defaults_json"] is None
    assert (
        chacha_db.execute_query(
            "SELECT assistant_defaults_json FROM workspaces WHERE id = ?", (workspace["id"],)
        ).fetchone()[0]
        == "{bad-json"
    )


@pytest.mark.parametrize(
    "stored_value",
    [None, "{private-malformed-marker", '["private-marker"]'],
    ids=["sql-null", "malformed-json", "non-object-json"],
)
def test_workspace_storage_projects_unset_or_invalid_without_private_payload(
    chacha_db: CharactersRAGDB,
    workspace_with_stored_assistant_defaults: Callable[..., dict[str, Any]],
    stored_value: str | None,
) -> None:
    """Project actual DB reads as unset or invalid without leaking storage values or the private flag."""
    from tldw_Server_API.app.api.v1.endpoints import workspaces

    workspace = workspace_with_stored_assistant_defaults(chacha_db, stored_value)
    loaded = chacha_db.get_workspace(workspace["id"])
    assert loaded is not None
    payload = workspaces._ws_to_response(loaded, db=chacha_db, current_user=SimpleNamespace(id="user-1")).model_dump()

    assert payload["assistant_defaults"] is None
    assert payload["effective_assistant_default"] == {
        "status": "none" if stored_value is None else "unavailable",
        "source": "none" if stored_value is None else "workspace",
        "assistant_kind": None,
        "assistant_id": None,
        "label": None,
        "persona_memory_mode": None,
        "degraded_reason": None if stored_value is None else "invalid_default",
    }
    assert "_assistant_defaults_invalid" not in payload
    assert "assistant_defaults_json" not in payload
    assert "private-malformed-marker" not in repr(payload)
    assert "private-marker" not in repr(payload)


@pytest.mark.parametrize(
    ("stored_value", "expected_defaults", "invalid"),
    [
        (None, None, False),
        ("{}", {}, False),
        ({}, {}, False),
        (MappingProxyType({"assistant_id": "persona-1"}), {"assistant_id": "persona-1"}, False),
        ({"unknown": "value"}, {"unknown": "value"}, False),
        ("null", None, True),
        ("[]", None, True),
        ('"text"', None, True),
        ("", None, True),
        ("   ", None, True),
        ([], None, True),
        (0, None, True),
        (False, None, True),
        (b"{}", None, True),
    ],
)
def test_workspace_mapping_row_computes_diagnostic_from_storage(
    stored_value: Any, expected_defaults: dict[str, Any] | None, invalid: bool
) -> None:
    """Normalize Mapping rows without mutating inputs or trusting an incoming diagnostic flag."""
    row = {"id": "ws-mapping", "assistant_defaults_json": stored_value, "_assistant_defaults_invalid": not invalid}
    normalized = CharactersRAGDB._workspace_row_to_dict(MappingProxyType(row))
    assert normalized["assistant_defaults_json"] == expected_defaults
    assert normalized["_assistant_defaults_invalid"] is invalid
    assert row["assistant_defaults_json"] is stored_value
    assert row["_assistant_defaults_invalid"] is not invalid


def test_workspace_mapping_without_defaults_is_unset() -> None:
    """Treat a missing storage column as unset and discard any stale corruption diagnostic."""
    normalized = CharactersRAGDB._workspace_row_to_dict({"id": "ws-old", "_assistant_defaults_invalid": True})
    assert normalized == {"id": "ws-old", "_assistant_defaults_invalid": False}


def test_workspace_update_rejects_private_diagnostic_only_payload(chacha_db: CharactersRAGDB) -> None:
    """A computed diagnostic is not a writable workspace field."""
    workspace = chacha_db.upsert_workspace("ws-1", "Research")
    with pytest.raises(InputError, match="No recognized workspace fields to update"):
        chacha_db.update_workspace(
            workspace["id"], {"_assistant_defaults_invalid": True}, expected_version=workspace["version"]
        )


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
