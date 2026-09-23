from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.chacha import schema_bootstrap
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

pytestmark = pytest.mark.unit


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "persona_exemplar_migration.sqlite"


def test_migration_v32_to_latest_creates_persona_exemplar_table(db_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def initialize_historical(db: CharactersRAGDB) -> None:
        with db.transaction() as conn:
            db._apply_schema_v4(conn)
            steps = db._sqlite_linear_migration_steps()
            for version in range(4, 32):
                steps[version](conn)
                assert db._get_db_version(conn) == version + 1

    with monkeypatch.context() as patch:
        patch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 32)
        patch.setattr(CharactersRAGDB, "_initialize_schema", initialize_historical)
        seed = CharactersRAGDB(db_path, "historical-fixture")
    try:
        with seed.transaction() as conn:
            assert seed._get_db_version(conn) == 32
            tables = seed._sqlite_table_names(conn)
            assert {"persona_exemplars", "note_attachments"}.isdisjoint(tables)
            conn.execute(
                "INSERT INTO persona_profiles (id, user_id, name, system_prompt) VALUES (?, ?, ?, ?)",
                ("retained-persona", "user-1", "Historical persona", "Retained prompt"),
            )
            before = dict(conn.execute("SELECT * FROM persona_profiles").fetchone())
    finally:
        seed.close_all_connections()

    migrated = CharactersRAGDB(db_path, "persona-exemplar-migration-check")
    try:
        conn = migrated.get_connection()

        version = conn.execute(
            "SELECT version FROM db_schema_version WHERE schema_name = ?",
            (CharactersRAGDB._SCHEMA_NAME,),
        ).fetchone()["version"]
        assert version == CharactersRAGDB._CURRENT_SCHEMA_VERSION

        table = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='persona_exemplars'"
        ).fetchone()
        assert table is not None

        columns = {
            row["name"] for row in conn.execute("PRAGMA table_info('persona_exemplars')").fetchall()
        }
        assert {
            "id",
            "persona_id",
            "user_id",
            "kind",
            "content",
            "tone",
            "scenario_tags_json",
            "capability_tags_json",
            "priority",
            "enabled",
            "source_type",
            "source_ref",
            "notes",
            "created_at",
            "last_modified",
            "deleted",
            "version",
        }.issubset(columns)

        indexes = {
            row["name"] for row in conn.execute("PRAGMA index_list('persona_exemplars')").fetchall()
        }
        assert "idx_persona_exemplars_persona" in indexes
        assert "idx_persona_exemplars_user" in indexes
        assert "idx_persona_exemplars_kind" in indexes
        assert "idx_persona_exemplars_enabled" in indexes

        after = dict(conn.execute("SELECT * FROM persona_profiles WHERE id = ?", (before["id"],)).fetchone())
        assert all(after[key] == value for key, value in before.items())
    finally:
        migrated.close_all_connections()


class _FakeTransaction:
    def __init__(self, connection):
        self.connection = connection

    def __enter__(self):
        return self.connection

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeBackend:
    backend_type = BackendType.POSTGRESQL

    def __init__(self):
        self.executed_statements: list[str] = []

    def transaction(self):
        return _FakeTransaction(self)

    def table_exists(self, _name: str, connection=None) -> bool:
        return True

    def escape_identifier(self, identifier: str) -> str:
        return '"' + identifier.replace('"', '""') + '"'

    def execute(self, statement, *_args, **_kwargs):
        self.executed_statements.append(str(statement))
        return None


def test_postgres_initializer_uses_postgres_safe_v33_migration(monkeypatch):
    db = CharactersRAGDB.__new__(CharactersRAGDB)
    db._backend = _FakeBackend()
    db._uses_shared_content_backend = False
    db._backend_refresh_suspended = False
    db._local = SimpleNamespace()

    applied_scripts: list[str] = []

    migration_connection = object()
    coordinator_calls: list[tuple[object, str]] = []
    version_reads: list[tuple[object, bool]] = []

    @contextmanager
    def coordinator(backend, lock_timeout):
        coordinator_calls.append((backend, lock_timeout))
        with backend.transaction():
            yield migration_connection

    def schema_version(conn, *, lock=False):
        version_reads.append((conn, lock))
        return 32

    monkeypatch.setattr(schema_bootstrap, "postgres_schema_migration", coordinator)
    monkeypatch.setattr(db, "_get_schema_version_postgres", schema_version)
    monkeypatch.setattr(db, "_ensure_postgres_fts", lambda conn: None)

    class _ReachedV35(Exception):
        pass

    def _record_script(script: str, conn, expected_version=None):
        applied_scripts.append(script)
        if expected_version == 35:
            raise _ReachedV35

    monkeypatch.setattr(db, "_apply_postgres_migration_script", _record_script)

    with pytest.raises(_ReachedV35):
        db._initialize_schema_postgres()

    assert CharactersRAGDB._MIGRATION_SQL_V32_TO_V33_POSTGRES in applied_scripts
    assert CharactersRAGDB._MIGRATION_SQL_V34_TO_V35 in applied_scripts
    assert "PRAGMA foreign_keys" not in CharactersRAGDB._MIGRATION_SQL_V32_TO_V33_POSTGRES
    assert coordinator_calls == [(db._backend, db._NOTES_MOODBOARD_STUDIO_V61_POSTGRES_LOCK_TIMEOUT)]
    assert version_reads[:3] == [(db._backend, False), (migration_connection, False), (migration_connection, True)]
