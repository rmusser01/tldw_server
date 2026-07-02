import sqlite3
from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.db_migration import (
    DatabaseMigrator,
    MigrationError,
)


def test_failed_multi_statement_migration_rolls_back_body_ledger_and_schema_version(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "app.db"
    migrations_dir = tmp_path / "migrations"
    migrations_dir.mkdir()

    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE schema_version (version INTEGER)")
        conn.execute("INSERT INTO schema_version (version) VALUES (0)")
        conn.commit()

    (migrations_dir / "001_fail_after_create.sql").write_text(
        """
        -- version: 1
        -- description: fail after creating a table
        CREATE TABLE created_before_failure (id INTEGER PRIMARY KEY);
        INSERT INTO missing_table(id) VALUES (1);
        """
    )

    migrator = DatabaseMigrator(str(db_path), str(migrations_dir))

    with pytest.raises(MigrationError):
        migrator.migrate_to_version(1, create_backup=False)

    with sqlite3.connect(db_path) as conn:
        assert (
            conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name='created_before_failure'"
            ).fetchone()
            is None
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM schema_migrations WHERE success = 1"
            ).fetchone()[0]
            == 0
        )
        assert conn.execute("SELECT version FROM schema_version").fetchone()[0] == 0


def test_migration_sql_with_transaction_control_is_rejected(tmp_path: Path) -> None:
    db_path = tmp_path / "app.db"
    migrations_dir = tmp_path / "migrations"
    migrations_dir.mkdir()

    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE schema_version (version INTEGER)")
        conn.execute("INSERT INTO schema_version (version) VALUES (0)")
        conn.commit()

    (migrations_dir / "001_embedded_transaction.sql").write_text(
        """
        -- version: 1
        -- description: embedded transaction control
        BEGIN TRANSACTION;
        CREATE TABLE should_not_run (id INTEGER PRIMARY KEY);
        COMMIT;
        """
    )

    migrator = DatabaseMigrator(str(db_path), str(migrations_dir))

    with pytest.raises(MigrationError, match="transaction control statements"):
        migrator.migrate_to_version(1, create_backup=False)

    with sqlite3.connect(db_path) as conn:
        assert (
            conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name='should_not_run'"
            ).fetchone()
            is None
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM schema_migrations WHERE success = 1"
            ).fetchone()[0]
            == 0
        )
        assert conn.execute("SELECT version FROM schema_version").fetchone()[0] == 0


def test_end_transaction_is_rejected_before_partial_ddl_can_commit(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "app.db"
    migrations_dir = tmp_path / "migrations"
    migrations_dir.mkdir()

    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE schema_version (version INTEGER)")
        conn.execute("INSERT INTO schema_version (version) VALUES (0)")
        conn.commit()

    (migrations_dir / "001_end_transaction.sql").write_text(
        """
        -- version: 1
        -- description: end transaction should be rejected
        CREATE TABLE should_not_persist (id INTEGER PRIMARY KEY);
        END TRANSACTION;
        INSERT INTO missing_table(id) VALUES (1);
        """
    )

    migrator = DatabaseMigrator(str(db_path), str(migrations_dir))

    with pytest.raises(MigrationError, match="transaction control statements"):
        migrator.migrate_to_version(1, create_backup=False)

    with sqlite3.connect(db_path) as conn:
        assert (
            conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name='should_not_persist'"
            ).fetchone()
            is None
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM schema_migrations WHERE success = 1"
            ).fetchone()[0]
            == 0
        )
        assert conn.execute("SELECT version FROM schema_version").fetchone()[0] == 0


def test_savepoint_control_is_rejected(tmp_path: Path) -> None:
    db_path = tmp_path / "app.db"
    migrations_dir = tmp_path / "migrations"
    migrations_dir.mkdir()

    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE schema_version (version INTEGER)")
        conn.execute("INSERT INTO schema_version (version) VALUES (0)")
        conn.commit()

    (migrations_dir / "001_savepoint_control.sql").write_text(
        """
        -- version: 1
        -- description: savepoint control should be rejected
        CREATE TABLE savepoint_should_not_persist (id INTEGER PRIMARY KEY);
        SAVEPOINT migration_inner;
        INSERT INTO missing_table(id) VALUES (1);
        """
    )

    migrator = DatabaseMigrator(str(db_path), str(migrations_dir))

    with pytest.raises(MigrationError, match="transaction control statements"):
        migrator.migrate_to_version(1, create_backup=False)

    with sqlite3.connect(db_path) as conn:
        assert (
            conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name='savepoint_should_not_persist'"
            ).fetchone()
            is None
        )
        assert conn.execute("SELECT version FROM schema_version").fetchone()[0] == 0


def test_one_line_multi_statement_migration_executes_each_statement(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "app.db"
    migrations_dir = tmp_path / "migrations"
    migrations_dir.mkdir()

    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE schema_version (version INTEGER)")
        conn.execute("INSERT INTO schema_version (version) VALUES (0)")
        conn.commit()

    (migrations_dir / "001_one_line_multi_statement.sql").write_text(
        """
        -- version: 1
        -- description: one line multi-statement migration
        CREATE TABLE one_line_a (id INTEGER PRIMARY KEY); CREATE TABLE one_line_b (id INTEGER PRIMARY KEY);
        """
    )

    migrator = DatabaseMigrator(str(db_path), str(migrations_dir))
    result = migrator.migrate_to_version(1, create_backup=False)

    assert result["status"] == "success"

    with sqlite3.connect(db_path) as conn:
        assert (
            conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name='one_line_a'"
            ).fetchone()
            is not None
        )
        assert (
            conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name='one_line_b'"
            ).fetchone()
            is not None
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM schema_migrations WHERE success = 1"
            ).fetchone()[0]
            == 1
        )
        assert conn.execute("SELECT version FROM schema_version").fetchone()[0] == 1


def test_trigger_body_begin_end_is_not_rejected_as_transaction_control(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "app.db"
    migrations_dir = tmp_path / "migrations"
    migrations_dir.mkdir()

    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE schema_version (version INTEGER)")
        conn.execute("INSERT INTO schema_version (version) VALUES (0)")
        conn.commit()

    (migrations_dir / "001_trigger_body.sql").write_text(
        """
        -- version: 1
        -- description: trigger body begin/end is not transaction control
        CREATE TABLE trigger_items (id INTEGER PRIMARY KEY);
        CREATE TABLE trigger_log (item_id INTEGER NOT NULL);
        CREATE TRIGGER trigger_items_ai AFTER INSERT ON trigger_items
        BEGIN
            INSERT INTO trigger_log(item_id) VALUES (NEW.id);
        END;
        INSERT INTO trigger_items(id) VALUES (7);
        """
    )

    migrator = DatabaseMigrator(str(db_path), str(migrations_dir))
    result = migrator.migrate_to_version(1, create_backup=False)

    assert result["status"] == "success"

    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT item_id FROM trigger_log").fetchone()[0] == 7
        assert conn.execute("SELECT version FROM schema_version").fetchone()[0] == 1
