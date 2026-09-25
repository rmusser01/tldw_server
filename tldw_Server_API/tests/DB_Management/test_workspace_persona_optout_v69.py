"""Durable Workspace Persona opt-out contracts for SQLite and PostgreSQL."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError, ConflictError

pytestmark = pytest.mark.integration

DEFAULT = {"assistant_kind": "persona", "assistant_id": "persona-1", "persona_memory_mode": "read_only"}
FLAG = "assistant_defaults_explicit_none"


@pytest.fixture(params=["sqlite", "postgres"])
def db_factory(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[Callable[[], CharactersRAGDB]]:
    """Open/reopen one isolated database using the repository's backend fixtures."""
    config = request.getfixturevalue("pg_database_config") if request.param == "postgres" else None
    opened: list[CharactersRAGDB] = []

    def create() -> CharactersRAGDB:
        """Create a fresh handle, including the normal schema initialization path."""
        backend = DatabaseBackendFactory.create_backend(config) if config else None
        database = CharactersRAGDB(tmp_path / "choice.db", client_id="user-1", backend=backend)
        opened.append(database)
        return database

    try:
        yield create
    finally:
        for database in opened:
            database.close_all_connections()


def test_choice_updates_are_atomic_and_survive_restart(db_factory: Callable[[], CharactersRAGDB]) -> None:
    """Clear/set/omit and stale updates preserve the paired flag and version contract."""
    db = db_factory()
    fresh = db.upsert_workspace("ws-choice", "Choice")
    assert fresh[FLAG] is False
    cleared = db.update_workspace(fresh["id"], {"assistant_defaults_json": None}, fresh["version"])
    assert cleared[FLAG] is True
    assert cleared["version"] == fresh["version"] + 1
    again = db.update_workspace(fresh["id"], {"assistant_defaults_json": None}, cleared["version"])
    assert again[FLAG] is True
    assert again["version"] == cleared["version"] + 1
    renamed = db.upsert_workspace(fresh["id"], "Renamed")
    assert renamed[FLAG] is True
    archived = db.update_workspace(fresh["id"], {"archived": True, FLAG: False}, renamed["version"])
    assert archived[FLAG] is True
    with pytest.raises(ConflictError):
        db.update_workspace(fresh["id"], {"assistant_defaults_json": DEFAULT}, again["version"])
    assert db.get_workspace(fresh["id"])[FLAG] is True
    saved = db.update_workspace(fresh["id"], {"assistant_defaults_json": DEFAULT, FLAG: True}, archived["version"])
    assert saved[FLAG] is False
    assert saved["assistant_defaults_json"] == DEFAULT
    with pytest.raises(ConflictError):
        db.update_workspace(fresh["id"], {"assistant_defaults_json": None}, archived["version"])
    assert db.get_workspace(fresh["id"])["assistant_defaults_json"] == DEFAULT
    cleared = db.update_workspace(fresh["id"], {"assistant_defaults_json": None}, saved["version"])
    db.close_all_connections()
    reopened = db_factory()
    assert reopened.get_workspace(fresh["id"])[FLAG] is True
    assert reopened.get_workspace(fresh["id"])["version"] == cleared["version"]
    assert reopened.list_workspaces()[0][FLAG] is True
    reopened.delete_workspace(fresh["id"], expected_version=cleared["version"])
    assert reopened.get_workspace(fresh["id"], include_deleted=True)[FLAG] is True


def test_pre_optout_upgrade_preserves_storage_and_conservatively_opts_out(
    db_factory: Callable[[], CharactersRAGDB],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Build storage at each backend's preceding version, then migrate."""
    with monkeypatch.context() as historical:
        historical.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 68)
        historical.setattr(CharactersRAGDB, "_POSTGRES_SCHEMA_VERSION", 72)
        legacy = db_factory()
    values = [None, '{"assistant_kind":"persona","assistant_id":"persona-1"}', "broken-json", "null", "[]", ""]
    with legacy.transaction() as conn:
        for index, value in enumerate(values):
            conn.execute(
                "INSERT INTO workspaces (id, name, client_id, assistant_defaults_json) VALUES (?, ?, ?, ?)",
                (f"legacy-{index}", "Legacy", "user-1", value),
            )
    assert FLAG not in dict(legacy.execute_query("SELECT * FROM workspaces LIMIT 1").fetchone())
    legacy.close_all_connections()
    upgraded = db_factory()
    for index, value in enumerate(values):
        row = upgraded.get_workspace(f"legacy-{index}")
        stored = dict(upgraded.execute_query("SELECT * FROM workspaces WHERE id = ?", (row["id"],)).fetchone())
        assert stored["assistant_defaults_json"] == value
        assert row[FLAG] is (value is None)
        assert row["_assistant_defaults_invalid"] is (index >= 2)
        assert row["version"] == 1
    assert upgraded.upsert_workspace("new-after-upgrade", "New")[FLAG] is False
    opted_out = upgraded.get_workspace("legacy-0")
    repaired = upgraded.update_workspace(opted_out["id"], {"assistant_defaults_json": DEFAULT}, opted_out["version"])
    assert repaired[FLAG] is False
    upgraded.close_all_connections()
    assert db_factory().get_workspace("legacy-0")[FLAG] is False


def test_inconsistent_choice_is_invalid_until_owner_repair(db_factory: Callable[[], CharactersRAGDB]) -> None:
    """A saved default plus an opt-out bit cannot be resolved as an available default."""
    db = db_factory()
    row = db.upsert_workspace("ws-inconsistent", "Inconsistent")
    row = db.update_workspace(row["id"], {"assistant_defaults_json": DEFAULT}, row["version"])
    with db.transaction() as conn:
        conn.execute("UPDATE workspaces SET assistant_defaults_explicit_none = ? WHERE id = ?", (True, row["id"]))
    assert db.get_workspace(row["id"])["_assistant_defaults_invalid"] is True
    for value in (DEFAULT, None):
        row = db.update_workspace(row["id"], {"assistant_defaults_json": value}, row["version"])
        assert row["_assistant_defaults_invalid"] is False
        assert row[FLAG] is (value is None)


def test_clone_and_import_conservatively_opt_out_without_overwriting_existing_choices(
    db_factory: Callable[[], CharactersRAGDB],
) -> None:
    """Unrepresented source choices never turn into new provisioning consent."""
    db = db_factory()
    clone = db.reserve_clone_target(
        workspace_id="cloned",
        operation_id="clone-operation",
        request_fingerprint="clone-fingerprint",
        name="Clone",
        description=None,
        workspace_profile="research",
    )
    assert clone[FLAG] is True
    assert clone["assistant_defaults_json"] is None
    db.publish_clone_target(workspace_id="cloned", operation_id="clone-operation")
    db.confirm_clone_target_publication(workspace_id="cloned", operation_id="clone-operation")
    assert db.get_workspace("cloned")[FLAG] is True
    existing = db.upsert_workspace("existing", "Existing")
    db.update_workspace(existing["id"], {"assistant_defaults_json": DEFAULT}, existing["version"])
    for target in ("imported", "existing"):
        data: dict[str, Any] = {
            "id": f"migration-{target}",
            "idempotency_key": f"key-{target}",
            "target_workspace_id": target,
            "target_workspace_name": "Imported",
            "manifest_hash": "a" * 64,
        }
        db.upsert_workspace_migration_session(data)
        db.upsert_workspace_migration_session(data)
    assert db.get_workspace("imported")[FLAG] is True
    assert db.get_workspace("existing")[FLAG] is False
    assert db.get_workspace("existing")["assistant_defaults_json"] == DEFAULT


def test_cached_legacy_writer_is_not_fenced_by_the_upgrade(
    db_factory: Callable[[], CharactersRAGDB],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Demonstrate why old writers must stop before the offline opt-out upgrade."""
    with monkeypatch.context() as historical:
        historical.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 68)
        historical.setattr(CharactersRAGDB, "_POSTGRES_SCHEMA_VERSION", 72)
        cached_legacy = db_factory()
    with cached_legacy.transaction() as conn:
        conn.execute(
            "INSERT INTO workspaces (id, name, client_id, assistant_defaults_json) VALUES (?, ?, ?, ?)",
            ("legacy-writer", "Legacy", "user-1", '{"assistant_id":"persona-1"}'),
        )
    upgraded = db_factory()
    assert upgraded.get_workspace("legacy-writer")[FLAG] is False
    # An old cached writer's clear omits the new bit: SQL NULL alone is ambiguous.
    with cached_legacy.transaction() as conn:
        conn.execute("UPDATE workspaces SET assistant_defaults_json = NULL WHERE id = ?", ("legacy-writer",))
    assert upgraded.get_workspace("legacy-writer")[FLAG] is False
    row = upgraded.get_workspace("legacy-writer")
    assert upgraded.update_workspace(row["id"], {"assistant_defaults_json": None}, row["version"])[FLAG] is True
    cached_legacy.close_all_connections()
    upgraded.close_all_connections()
    with monkeypatch.context() as historical:
        historical.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 68)
        historical.setattr(CharactersRAGDB, "_POSTGRES_SCHEMA_VERSION", 72)
        with pytest.raises(CharactersRAGDBError, match="newer than supported by code"):
            db_factory()


def test_failed_upgrade_rolls_back_column_and_backfill(
    db_factory: Callable[[], CharactersRAGDB],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed backfill must leave the preceding schema unchanged."""
    with monkeypatch.context() as historical:
        historical.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 68)
        historical.setattr(CharactersRAGDB, "_POSTGRES_SCHEMA_VERSION", 72)
        legacy = db_factory()
    with legacy.transaction() as conn:
        conn.execute("INSERT INTO workspaces (id, name, client_id) VALUES (?, ?, ?)", ("legacy", "Legacy", "user-1"))
    method = (
        "_migrate_from_v72_to_v73_postgres" if legacy.backend_type.value == "postgresql" else "_migrate_from_v68_to_v69"
    )
    migrate = getattr(CharactersRAGDB, method)
    legacy.close_all_connections()

    def fail_after_backfill(self: CharactersRAGDB, conn: Any) -> None:
        """Inject a transaction failure only after the real migration executed."""
        migrate(self, conn)
        raise CharactersRAGDBError("injected upgrade failure")

    with monkeypatch.context() as failing:
        failing.setattr(CharactersRAGDB, method, fail_after_backfill)
        with pytest.raises(CharactersRAGDBError, match="injected upgrade failure"):
            db_factory()
    with monkeypatch.context() as historical:
        historical.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 68)
        historical.setattr(CharactersRAGDB, "_POSTGRES_SCHEMA_VERSION", 72)
        rolled_back = db_factory()
    row = dict(rolled_back.execute_query("SELECT * FROM workspaces WHERE id = ?", ("legacy",)).fetchone())
    assert FLAG not in row
    assert row["assistant_defaults_json"] is None
    assert row["version"] == 1


@pytest.mark.parametrize("failure", ["catalog_drift", "late_initializer", "generic_initializer"])
def test_sqlite_initialization_failure_does_not_commit_optout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    """Later legacy checks/helpers must succeed before the opt-out migration commits."""
    path = tmp_path / "late-failure.db"
    with monkeypatch.context() as historical:
        historical.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 68)
        legacy = CharactersRAGDB(path, client_id="user-1")
    with legacy.transaction() as conn:
        conn.execute("INSERT INTO workspaces (id, name, client_id) VALUES (?, ?, ?)", ("legacy", "Legacy", "user-1"))
        if failure == "catalog_drift":
            conn.execute("CREATE INDEX unexpected_studio_index ON note_studio_documents(note_id)")
    legacy.close_all_connections()

    def fail_late(self: CharactersRAGDB, conn: Any = None) -> None:
        """Fail after legacy helpers which use executescript have already run."""
        raise CharactersRAGDBError("late initialization failure")

    with monkeypatch.context() as failing:
        if failure == "late_initializer":
            failing.setattr(CharactersRAGDB, "_self_heal_character_cards_fts_sqlite", fail_late)
        elif failure == "generic_initializer":
            failing.setattr(CharactersRAGDB, "_ensure_conversation_settings_table", fail_late)
        with pytest.raises(CharactersRAGDBError, match="catalog drifted|late initialization failure"):
            CharactersRAGDB(path, client_id="user-1")
    with sqlite3.connect(path) as conn:
        assert conn.execute(
            "SELECT version FROM db_schema_version WHERE schema_name = ?", (CharactersRAGDB._SCHEMA_NAME,)
        ).fetchone() == (68,)
        assert FLAG not in {row[1] for row in conn.execute("PRAGMA table_info(workspaces)")}
        assert conn.execute(
            "SELECT assistant_defaults_json, version FROM workspaces WHERE id = 'legacy'"
        ).fetchone() == (None, 1)


def test_sqlite_compatible_initializer_can_finish_during_version_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recheck a compatible committed upgrade without rejecting or backfilling twice."""
    path = tmp_path / "initializer-race.db"
    with monkeypatch.context() as historical:
        historical.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 68)
        legacy = CharactersRAGDB(path, client_id="user-1")
        legacy.close_all_connections()
    initialize = CharactersRAGDB._initialize_schema_sqlite_legacy
    raced = False

    def finish_competing_upgrade(self: CharactersRAGDB, *, target_version: int) -> None:
        """Interleave another handle between the preliminary probe and transaction."""
        nonlocal raced
        if not raced:
            raced = True
            peer = CharactersRAGDB(path, client_id="user-1")
            try:
                assert peer.upsert_workspace("fresh-after-upgrade", "Fresh")[FLAG] is False
            finally:
                peer.close_all_connections()
        initialize(self, target_version=target_version)

    monkeypatch.setattr(CharactersRAGDB, "_initialize_schema_sqlite_legacy", finish_competing_upgrade)
    db = CharactersRAGDB(path, client_id="user-1")
    try:
        assert db.get_workspace("fresh-after-upgrade")[FLAG] is False
    finally:
        db.close_all_connections()
