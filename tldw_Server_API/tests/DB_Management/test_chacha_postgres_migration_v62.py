"""PostgreSQL schema-v62 contracts for staged Workspace clone targets."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

pytestmark = pytest.mark.unit


class _ReachedV62(Exception):
    pass


class _FakeTransaction:
    def __enter__(self) -> object:
        return object()

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        return False


class _FakeBackend:
    backend_type = BackendType.POSTGRESQL

    def transaction(self) -> _FakeTransaction:
        return _FakeTransaction()

    def table_exists(self, _name: str, connection: object = None) -> bool:
        return True


def test_postgres_initializer_routes_schema_v61_through_v62(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = CharactersRAGDB.__new__(CharactersRAGDB)
    db._backend = _FakeBackend()
    db._uses_shared_content_backend = False
    db._backend_refresh_suspended = False
    db._local = SimpleNamespace()

    monkeypatch.setattr(CharactersRAGDB, "_POSTGRES_SCHEMA_VERSION", 62)
    monkeypatch.setattr(db, "_get_schema_version_postgres", lambda _conn, lock=False: 61)
    monkeypatch.setattr(db, "_verify_note_attachment_schema_postgres", lambda _conn: None)
    monkeypatch.setattr(db, "_verify_note_task_schema_postgres", lambda _conn: None)

    def _reached_v62(_conn: object) -> None:
        raise _ReachedV62

    monkeypatch.setattr(db, "_migrate_from_v61_to_v62_postgres", _reached_v62, raising=False)

    with pytest.raises(_ReachedV62):
        db._initialize_schema_postgres()


def test_postgres_v62_migration_versions_after_applying_ddl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = CharactersRAGDB.__new__(CharactersRAGDB)
    applied: list[tuple[str, int]] = []
    versions = iter((61, 62))
    monkeypatch.setattr(db, "_get_schema_version_postgres", lambda _conn: next(versions))
    monkeypatch.setattr(
        db,
        "_apply_postgres_migration_script",
        lambda script, _conn, *, expected_version: applied.append((script, expected_version)),
    )

    db._migrate_from_v61_to_v62_postgres(object())

    assert applied == [(CharactersRAGDB._MIGRATION_SQL_V61_TO_V62_POSTGRES, 62)]


def test_postgres_v62_ddl_matches_sqlite_marker_contract() -> None:
    sql = " ".join(CharactersRAGDB._MIGRATION_SQL_V61_TO_V62_POSTGRES.split())

    for clause in (
        "ALTER TABLE workspaces ADD COLUMN IF NOT EXISTS system_operation_id TEXT",
        "ALTER TABLE workspaces ADD COLUMN IF NOT EXISTS system_operation_kind TEXT",
        "ALTER TABLE workspaces ADD COLUMN IF NOT EXISTS system_operation_state TEXT",
        "ALTER TABLE workspaces ADD COLUMN IF NOT EXISTS system_request_fingerprint TEXT",
        "CHECK (system_operation_kind IS NULL OR system_operation_kind = 'shared_workspace_clone')",
        "CHECK (system_operation_state IS NULL OR system_operation_state IN ('staged', 'publication_pending'))",
        "idx_workspaces_system_operation ON workspaces(system_operation_kind, system_operation_state, system_operation_id)",
    ):
        assert clause in sql
