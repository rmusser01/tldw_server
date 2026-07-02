import json
from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.db_migration import DatabaseMigrator, MigrationError


def test_load_migrations_raises_on_duplicate_versions(tmp_path: Path):
    migrations_dir = tmp_path / "migrations"
    migrations_dir.mkdir()
    db_path = tmp_path / "app.db"
    db_path.touch()

    (migrations_dir / "001_first.json").write_text(
        json.dumps({"version": 1, "name": "first", "up_sql": "SELECT 1"})
    )
    (migrations_dir / "001_second.json").write_text(
        json.dumps({"version": 1, "name": "second", "up_sql": "SELECT 2"})
    )

    migrator = DatabaseMigrator(str(db_path), str(migrations_dir))

    with pytest.raises(MigrationError, match="Duplicate migration version 1"):
        migrator.load_migrations()


def test_load_migrations_raises_on_malformed_artifact(tmp_path: Path):
    migrations_dir = tmp_path / "migrations"
    migrations_dir.mkdir()
    db_path = tmp_path / "app.db"
    db_path.touch()

    (migrations_dir / "001_first.json").write_text("{not valid json")

    migrator = DatabaseMigrator(str(db_path), str(migrations_dir))

    with pytest.raises(MigrationError, match="Invalid migration set: 001_first.json"):
        migrator.load_migrations()


def test_load_migrations_accepts_packaged_media_schema_migration_dir(tmp_path: Path):
    from tldw_Server_API.app.core.DB_Management.media_db.schema.backends import (
        sqlite_helpers as sqlite_helpers_module,
    )

    db_path = tmp_path / "app.db"
    db_path.touch()
    migrations_dir = (
        Path(sqlite_helpers_module.__file__).resolve().parents[1]
        / "migrations"
        / "sqlite"
    )

    migrator = DatabaseMigrator(str(db_path), str(migrations_dir))
    migrations = migrator.load_migrations()

    assert [migration.version for migration in migrations] == [23]
    assert [migration.name for migration in migrations] == ["transcript_run_history"]
