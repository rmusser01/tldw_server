import json
from pathlib import Path
from types import SimpleNamespace

import pytest

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
