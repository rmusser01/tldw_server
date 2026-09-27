import json
import sqlite3
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


def test_idempotent_sql_migration_skips_existing_first_add_column_after_comments(
    tmp_path: Path,
):
    migrations_dir = tmp_path / "migrations"
    migrations_dir.mkdir()
    db_path = tmp_path / "app.db"

    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE widgets (
                id INTEGER PRIMARY KEY,
                job_id INTEGER
            );
            CREATE TABLE schema_version (version INTEGER NOT NULL);
            INSERT INTO schema_version (version) VALUES (0);
            """
        )

    (migrations_dir / "001_add_widget_job.sql").write_text(
        """-- version: 1
-- description: Add widget job linkage
-- idempotent: true
ALTER TABLE widgets ADD COLUMN job_id INTEGER;
UPDATE schema_version SET version = 1;
"""
    )

    migrator = DatabaseMigrator(str(db_path), str(migrations_dir))

    result = migrator.migrate_to_version(1, create_backup=False)
    applied_migrations = migrator.get_applied_migrations()

    assert result["status"] == "success"
    assert result["current_version"] == 1
    assert len(applied_migrations) == 1
    assert applied_migrations[0]["version"] == 1
    assert applied_migrations[0]["name"] == "add_widget_job"

    with sqlite3.connect(db_path) as conn:
        job_id_columns = [
            row for row in conn.execute("PRAGMA table_info(widgets)") if row[1] == "job_id"
        ]

    assert len(job_id_columns) == 1
