import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.DB_Management import db_migration as db_migration_module
from tldw_Server_API.app.core.DB_Management.db_migration import DatabaseMigrator, MigrationError


def test_migrate_to_version_rejects_missing_intermediate_versions(tmp_path: Path):
    migrations_dir = tmp_path / "migrations"
    migrations_dir.mkdir()
    db_path = tmp_path / "app.db"
    db_path.touch()

    (migrations_dir / "001_first.json").write_text(
        json.dumps({"version": 1, "name": "first", "up_sql": "SELECT 1"})
    )
    (migrations_dir / "003_third.json").write_text(
        json.dumps({"version": 3, "name": "third", "up_sql": "SELECT 3"})
    )

    migrator = DatabaseMigrator(str(db_path), str(migrations_dir))

    with pytest.raises(MigrationError, match=r"Missing migration versions: \[2\]"):
        migrator.migrate_to_version(3, create_backup=False)


def test_execute_migration_rolls_back_failed_multi_statement_script(tmp_path: Path) -> None:
    db_path = tmp_path / "app.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE schema_version (version INTEGER)")
        conn.execute("INSERT INTO schema_version VALUES (0)")
        conn.commit()

    migrator = DatabaseMigrator(str(db_path), str(tmp_path / "migrations"))
    migrator.initialize_migration_table()
    migration = db_migration_module.Migration(
        version=1,
        name="partial_failure_demo",
        up_sql=(
            "CREATE TABLE kept_after_failure (id INTEGER); "
            "INSERT INTO missing_table VALUES (1);"
        ),
    )

    with pytest.raises(MigrationError, match="missing_table"):
        migrator.execute_migration(migration)

    with sqlite3.connect(db_path) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        rows = conn.execute(
            "SELECT version, success, error_message FROM schema_migrations"
        ).fetchall()
        version = conn.execute("SELECT version FROM schema_version").fetchone()[0]

    assert "kept_after_failure" not in tables
    assert rows == [(1, 0, "no such table: missing_table")]
    assert version == 0


def test_migrate_to_version_rejects_rollback_without_down_sql(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    db_path = tmp_path / "app.db"
    db_path.touch()
    migrator = DatabaseMigrator(str(db_path), str(tmp_path / "unused"))

    monkeypatch.setattr(migrator, "get_current_version", lambda: 2)
    monkeypatch.setattr(
        migrator,
        "load_migrations",
        lambda: [
            SimpleNamespace(
                version=1,
                name="first",
                up_sql="SELECT 1",
                down_sql="SELECT 1",
                checksum="a",
                idempotent=False,
            ),
            SimpleNamespace(
                version=2,
                name="second",
                up_sql="SELECT 2",
                down_sql=None,
                checksum="b",
                idempotent=False,
            ),
        ],
    )

    with pytest.raises(MigrationError, match="down_sql"):
        migrator.migrate_to_version(0, create_backup=False)


def test_migrate_to_version_rejects_rollback_with_missing_intermediate_versions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    db_path = tmp_path / "app.db"
    db_path.touch()
    migrator = DatabaseMigrator(str(db_path), str(tmp_path / "unused"))

    monkeypatch.setattr(migrator, "get_current_version", lambda: 3)
    monkeypatch.setattr(
        migrator,
        "load_migrations",
        lambda: [
            SimpleNamespace(
                version=1,
                name="first",
                up_sql="SELECT 1",
                down_sql="SELECT 1",
                checksum="a",
                idempotent=False,
            ),
            SimpleNamespace(
                version=3,
                name="third",
                up_sql="SELECT 3",
                down_sql="SELECT 3",
                checksum="c",
                idempotent=False,
            ),
        ],
    )

    with pytest.raises(MigrationError, match=r"Missing migration versions: \[2\]"):
        migrator.migrate_to_version(0, create_backup=False)


def test_migrate_to_version_allows_redis_file_fallback_in_test_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "app.db"
    db_path.touch()
    migrations_dir = tmp_path / "migrations"
    migrations_dir.mkdir()
    migrator = DatabaseMigrator(str(db_path), str(migrations_dir))
    recorded_lock_kwargs: dict[str, object] = {}

    @contextmanager
    def fake_acquire_migration_lock(**kwargs: object):
        recorded_lock_kwargs.update(kwargs)
        yield object()

    monkeypatch.setenv("REDIS_URL", "redis://127.0.0.1:1/0")
    monkeypatch.setattr(
        db_migration_module,
        "acquire_migration_lock",
        fake_acquire_migration_lock,
    )
    monkeypatch.setattr(
        DatabaseMigrator,
        "_allow_redis_file_lock_fallback",
        staticmethod(lambda: True),
    )
    monkeypatch.setattr(
        migrator,
        "_migrate_to_version_locked",
        lambda target_version=None, create_backup=True: {"status": "ok"},
    )

    assert migrator.migrate_to_version(3, create_backup=False) == {"status": "ok"}
    assert recorded_lock_kwargs["redis_url"] == "redis://127.0.0.1:1/0"
    assert recorded_lock_kwargs["lock_dir"] == str(tmp_path)
    assert recorded_lock_kwargs["allow_file_fallback_on_redis_error"] is True


def test_migrate_to_version_keeps_redis_fail_closed_outside_test_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "app.db"
    db_path.touch()
    migrations_dir = tmp_path / "migrations"
    migrations_dir.mkdir()
    migrator = DatabaseMigrator(str(db_path), str(migrations_dir))
    recorded_lock_kwargs: dict[str, object] = {}

    @contextmanager
    def fake_acquire_migration_lock(**kwargs: object):
        recorded_lock_kwargs.update(kwargs)
        yield object()

    monkeypatch.setenv("REDIS_URL", "redis://127.0.0.1:1/0")
    monkeypatch.setattr(
        db_migration_module,
        "acquire_migration_lock",
        fake_acquire_migration_lock,
    )
    monkeypatch.setattr(
        DatabaseMigrator,
        "_allow_redis_file_lock_fallback",
        staticmethod(lambda: False),
    )
    monkeypatch.setattr(
        migrator,
        "_migrate_to_version_locked",
        lambda target_version=None, create_backup=True: {"status": "ok"},
    )

    assert migrator.migrate_to_version(3, create_backup=False) == {"status": "ok"}
    assert recorded_lock_kwargs["allow_file_fallback_on_redis_error"] is False
