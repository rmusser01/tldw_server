import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.backends.base import BackendType


pytestmark = pytest.mark.unit


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "persona_exemplar_migration.sqlite"


def test_migration_v32_to_latest_creates_persona_exemplar_table(db_path: Path):
    db = CharactersRAGDB(db_path, "persona-exemplar-migration-seed")
    db.close_connection()

    with sqlite3.connect(str(db_path)) as conn:
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute(
            "UPDATE db_schema_version SET version = ? WHERE schema_name = ?",
            (32, CharactersRAGDB._SCHEMA_NAME),
        )
        conn.execute("DROP TABLE IF EXISTS persona_exemplars")
        conn.commit()

    migrated = CharactersRAGDB(db_path, "persona-exemplar-migration-check")
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

    migrated.close_connection()


class _FakeTransaction:
    def __enter__(self):
        return object()

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeBackend:
    backend_type = BackendType.POSTGRESQL

    def transaction(self):
        return _FakeTransaction()

    def table_exists(self, _name: str, connection=None) -> bool:
        return True

    def execute(self, *_args, **_kwargs):
        return None


def test_postgres_initializer_uses_postgres_safe_v33_migration(monkeypatch):
    db = CharactersRAGDB.__new__(CharactersRAGDB)
    db._backend = _FakeBackend()
    db._uses_shared_content_backend = False
    db._backend_refresh_suspended = False
    db._local = SimpleNamespace()

    applied_scripts: list[str] = []

    monkeypatch.setattr(db, "_get_schema_version_postgres", lambda conn: 32)
    monkeypatch.setattr(db, "_ensure_postgres_fts", lambda conn: None)

    def _record_script(script: str, conn, expected_version=None):
        applied_scripts.append(script)

    monkeypatch.setattr(db, "_apply_postgres_migration_script", _record_script)

    db._initialize_schema_postgres()

    assert CharactersRAGDB._MIGRATION_SQL_V32_TO_V33_POSTGRES in applied_scripts
    assert CharactersRAGDB._MIGRATION_SQL_V34_TO_V35 in applied_scripts
    assert "PRAGMA foreign_keys" not in CharactersRAGDB._MIGRATION_SQL_V32_TO_V33_POSTGRES
