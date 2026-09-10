# test_migrations.py - Test database migration system
"""
Test suite for database migration functionality.
"""

import json
import os
import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.db_migration import DatabaseMigrator, Migration, MigrationError


class TestMigrations(unittest.TestCase):
    """Test database migration system"""

    def setUp(self):
        """Set up test database and migrations"""
        # Create temporary directory for test
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, "test.db")
        self.migrations_dir = os.path.join(self.test_dir, "migrations")
        os.makedirs(self.migrations_dir)

        # Create test database with initial schema
        conn = sqlite3.connect(self.db_path)
        conn.execute("CREATE TABLE test_table (id INTEGER PRIMARY KEY)")
        conn.close()

        # Create migrator
        self.migrator = DatabaseMigrator(self.db_path, self.migrations_dir)

    def tearDown(self):
        """Clean up test directory"""
        shutil.rmtree(self.test_dir)

    def create_test_migration(self, version: int, name: str):
        """Create a test migration file"""
        migration = {
            "version": version,
            "name": name,
            "description": f"Test migration {version}",
            "up_sql": f"ALTER TABLE test_table ADD COLUMN col_{version} TEXT;",
            "down_sql": f"ALTER TABLE test_table DROP COLUMN col_{version};"
        }

        filepath = os.path.join(self.migrations_dir, f"{version:03d}_{name}.json")
        with open(filepath, 'w') as f:
            json.dump(migration, f)

    def test_migration_creation(self):
        """Test creating migration objects"""
        migration = Migration(
            version=1,
            name="test_migration",
            up_sql="CREATE TABLE test (id INTEGER);",
            down_sql="DROP TABLE test;",
            description="Test migration"
        )

        self.assertEqual(migration.version, 1)
        self.assertEqual(migration.name, "test_migration")
        self.assertIsNotNone(migration.checksum)

    def test_initialize_migration_table(self):
        """Test migration table initialization"""
        self.migrator.initialize_migration_table()

        # Check table exists
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_migrations'"
        )
        self.assertIsNotNone(cursor.fetchone())
        conn.close()

    def test_get_current_version(self):
        """Test getting current schema version"""
        # Initially should be 0
        version = self.migrator.get_current_version()
        self.assertEqual(version, 0)

        # Add a migration record
        self.migrator.initialize_migration_table()
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO schema_migrations (version, name, checksum, applied_at, execution_time, success) "
            "VALUES (1, 'test', 'abc123', datetime('now'), 0.1, 1)"
        )
        conn.commit()
        conn.close()

        # Now should be 1
        version = self.migrator.get_current_version()
        self.assertEqual(version, 1)

    def test_load_migrations(self):
        """Test loading migrations from directory"""
        # Create test migrations
        self.create_test_migration(1, "add_column_1")
        self.create_test_migration(2, "add_column_2")

        migrations = self.migrator.load_migrations()
        self.assertEqual(len(migrations), 2)
        self.assertEqual(migrations[0].version, 1)
        self.assertEqual(migrations[1].version, 2)

    def test_execute_migration_up(self):
        """Test executing migration up"""
        self.migrator.initialize_migration_table()

        migration = Migration(
            version=1,
            name="add_column",
            up_sql="ALTER TABLE test_table ADD COLUMN new_col TEXT;",
            down_sql="ALTER TABLE test_table DROP COLUMN new_col;"
        )

        execution_time = self.migrator.execute_migration(migration, "up")
        self.assertGreater(execution_time, 0)

        # Check column was added
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("PRAGMA table_info(test_table)")
        columns = [row[1] for row in cursor.fetchall()]
        self.assertIn("new_col", columns)

        # Check migration was recorded
        cursor = conn.execute(
            "SELECT version FROM schema_migrations WHERE version = 1 AND success = 1"
        )
        self.assertIsNotNone(cursor.fetchone())
        conn.close()

    def test_execute_migration_down(self):
        """Test executing migration down"""
        # First apply migration up
        self.test_execute_migration_up()

        migration = Migration(
            version=1,
            name="add_column",
            up_sql="ALTER TABLE test_table ADD COLUMN new_col TEXT;",
            down_sql=None,
        )

        with self.assertRaises(MigrationError):
            self.migrator.execute_migration(migration, "down")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "SELECT version FROM schema_migrations WHERE version = 1"
        )
        self.assertIsNotNone(cursor.fetchone())
        conn.close()

    def test_migrate_to_version(self):
        """Test migrating to specific version"""
        # Create migrations
        self.create_test_migration(1, "migration_1")
        self.create_test_migration(2, "migration_2")

        # Migrate to version 2
        result = self.migrator.migrate_to_version(2)

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["previous_version"], 0)
        self.assertEqual(result["current_version"], 2)
        self.assertEqual(len(result["migrations_applied"]), 2)

    def test_rollback(self):
        """Test rolling back migrations"""
        # First migrate up
        self.create_test_migration(1, "migration_1")
        self.create_test_migration(2, "migration_2")
        self.migrator.migrate_to_version(2)

        # Then rollback to version 1
        result = self.migrator.migrate_to_version(1)

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["previous_version"], 2)
        self.assertEqual(result["current_version"], 1)
        self.assertEqual(len(result["migrations_applied"]), 1)
        self.assertEqual(result["migrations_applied"][0]["direction"], "down")

    @pytest.mark.unit
    def test_no_migrations_upgrade_rejects_missing_chain(self):
        """Reject an unavailable upgrade before changing the schema version."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER)")
            conn.execute("DELETE FROM schema_version")
            conn.execute("INSERT INTO schema_version (version) VALUES (?)", (8,))

        with self.assertRaises(MigrationError) as error:
            self.migrator.migrate_to_version(20)

        self.assertEqual(str(error.exception), f"Missing migration versions: {list(range(9, 21))}")
        self.assertEqual(self.migrator.get_current_version(), 8)
        self.assertEqual(self.migrator.get_applied_migrations(), [])

    def test_sql_migration_without_down_sql_prevents_downgrade(self):
        """Ensure SQL migrations without down_sql cannot be downgraded."""
        sql_path = Path(self.migrations_dir) / "001_add_flag.sql"
        sql_path.write_text("ALTER TABLE test_table ADD COLUMN flag INTEGER;")

        up_result = self.migrator.migrate_to_version(1)
        self.assertEqual(up_result["status"], "success")
        self.assertEqual(self.migrator.get_current_version(), 1)

        with self.assertRaises(MigrationError):
            self.migrator.migrate_to_version(0)

        self.assertEqual(self.migrator.get_current_version(), 1)

    def test_create_backup(self):
        """Test database backup creation"""
        backup_path = self.migrator.create_backup("test_backup")

        self.assertTrue(os.path.exists(backup_path))
        self.assertTrue(os.path.exists(backup_path + ".json"))

        # Check backup metadata
        with open(backup_path + ".json") as f:
            metadata = json.load(f)

        self.assertEqual(
            Path(metadata["original_path"]).resolve(),
            Path(self.db_path).resolve(),
        )
        self.assertEqual(metadata["description"], "test_backup")

    def test_verify_migrations(self):
        """Test migration verification"""
        # Apply a migration
        self.create_test_migration(1, "test_migration")
        self.migrator.migrate_to_version(1)

        # Verify - should be no issues
        issues = self.migrator.verify_migrations()
        self.assertEqual(len(issues), 0)

        # Delete migration file to create an issue
        os.remove(os.path.join(self.migrations_dir, "001_test_migration.json"))

        # Verify again - should find issue
        issues = self.migrator.verify_migrations()
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0]["issue"], "migration_file_missing")


if __name__ == "__main__":
    unittest.main()
