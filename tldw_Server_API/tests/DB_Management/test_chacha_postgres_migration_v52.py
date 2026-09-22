"""Tests for ChaChaNotes EMQ group metadata schema migration."""

import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.chacha import schema_bootstrap
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

pytestmark = pytest.mark.unit


class _FakeTransaction:
    def __init__(self, connection):
        self.connection = connection
        self.exit_exception = None

    def __enter__(self):
        return self.connection

    def __exit__(self, exc_type, exc, tb):
        self.exit_exception = exc_type
        return False


class _FakeBackend:
    backend_type = BackendType.POSTGRESQL

    def __init__(self):
        self.executed_statements: list[str] = []

    def transaction(self):
        return _FakeTransaction(self)

    def table_exists(self, _name: str, connection=None) -> bool:
        return True

    def execute(self, statement, *_args, **_kwargs):
        self.executed_statements.append(str(statement))
        return None

    @staticmethod
    def escape_identifier(identifier: str) -> str:
        return '"' + identifier.replace('"', '""') + '"'


def test_sqlite_v52_to_v53_preserves_existing_group_column_and_is_rerunnable(tmp_path: Path) -> None:
    db_path = tmp_path / "chacha-v52.db"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE db_schema_version (
                schema_name TEXT PRIMARY KEY,
                version INTEGER NOT NULL
            );
            INSERT INTO db_schema_version(schema_name, version)
            VALUES('rag_char_chat_schema', 52);
            CREATE TABLE quiz_questions (
                id INTEGER PRIMARY KEY,
                group_id TEXT
            );
            """
        )

    db = CharactersRAGDB.__new__(CharactersRAGDB)
    db.db_path_str = str(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        migrated_version = db._run_sqlite_linear_migration_step(
            conn,
            from_version=52,
            target_version=53,
            initial_version=52,
        )
        db._migrate_from_v52_to_v53(conn)
        columns = {row["name"] for row in conn.execute("PRAGMA table_info('quiz_questions')")}
        stored_version = conn.execute(
            "SELECT version FROM db_schema_version WHERE schema_name = ?",
            (CharactersRAGDB._SCHEMA_NAME,),
        ).fetchone()["version"]

    assert migrated_version == 53
    assert stored_version == 53
    assert {"group_id", "group_prompt"} <= columns


def test_postgres_initializer_routes_historical_v52_through_v53_script(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _ReachedV53Error(Exception):
        pass

    db = CharactersRAGDB.__new__(CharactersRAGDB)
    db._backend = _FakeBackend()
    db._uses_shared_content_backend = False
    db._backend_refresh_suspended = False
    db._local = SimpleNamespace()

    # Isolate routing from session-lock ownership, exercised with real PostgreSQL
    # in test_chacha_postgres_schema_lock.
    migration_transaction = _FakeTransaction(object())
    coordinator_calls: list[tuple[object, str]] = []

    def _schema_migration(backend, lock_timeout):
        coordinator_calls.append((backend, lock_timeout))
        return migration_transaction

    monkeypatch.setattr(schema_bootstrap, "postgres_schema_migration", _schema_migration)
    applied_scripts: list[tuple[str, int | None]] = []
    schema_version_reads: list[tuple[object, bool]] = []

    def _schema_version(_conn: object, *, lock: bool = False) -> int:
        schema_version_reads.append((_conn, lock))
        return 52

    monkeypatch.setattr(db, "_get_schema_version_postgres", _schema_version)
    monkeypatch.setattr(db, "_ensure_postgres_fts", lambda conn: None)

    def _record_script(script: str, conn, expected_version=None):
        assert conn is migration_transaction.connection
        applied_scripts.append((script, expected_version))
        if expected_version == 53:
            raise _ReachedV53Error

    monkeypatch.setattr(db, "_apply_postgres_migration_script", _record_script)

    with pytest.raises(_ReachedV53Error):
        db._initialize_schema_postgres()

    assert applied_scripts == [(CharactersRAGDB._MIGRATION_SQL_V52_TO_V53_POSTGRES, 53)]
    assert schema_version_reads == [
        (db._backend, False),
        (migration_transaction.connection, False),
        (migration_transaction.connection, True),
    ]
    assert coordinator_calls == [
        (db._backend, db._NOTES_MOODBOARD_STUDIO_V61_POSTGRES_LOCK_TIMEOUT),
    ]
    assert migration_transaction.exit_exception is _ReachedV53Error


def test_postgres_v53_script_adds_emq_group_columns_and_updates_version() -> None:
    script = CharactersRAGDB._MIGRATION_SQL_V52_TO_V53_POSTGRES

    assert "ALTER TABLE quiz_questions ADD COLUMN IF NOT EXISTS group_id TEXT" in script
    assert "ALTER TABLE quiz_questions ADD COLUMN IF NOT EXISTS group_prompt TEXT" in script
    assert "SET version = 53" in script
