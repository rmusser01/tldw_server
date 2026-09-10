"""Regression coverage for SQLite migration planning and atomic execution."""

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from hypothesis import given
from hypothesis import strategies as st

from tldw_Server_API.app.core.DB_Management import db_migration as db_migration_module
from tldw_Server_API.app.core.DB_Management.db_migration import DatabaseMigrator, MigrationError


@pytest.fixture
def migration_db(tmp_path: Path) -> tuple[Path, DatabaseMigrator]:
    """Provide an isolated file-backed database with an initialized migration ledger."""
    db_path = tmp_path / "app.db"
    migrator = DatabaseMigrator(str(db_path), str(tmp_path / "migrations"))
    migrator.initialize_migration_table()
    return db_path, migrator


@pytest.fixture
def versioned_migration_db(
    migration_db: tuple[Path, DatabaseMigrator],
) -> tuple[Path, DatabaseMigrator]:
    """Add legacy version metadata to the isolated migration database."""
    db_path, _ = migration_db
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE schema_version (version INTEGER)")
        conn.execute("INSERT INTO schema_version VALUES (0)")
    return migration_db


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


@pytest.mark.unit
def test_execute_migration_rolls_back_failed_multi_statement_script(
    versioned_migration_db: tuple[Path, DatabaseMigrator],
) -> None:
    """A failed statement leaves no partial schema or advanced version behind."""
    db_path, migrator = versioned_migration_db
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


@pytest.mark.unit
@pytest.mark.parametrize("enforcement", ["OFF", "ON"], ids=["disabled", "enabled"])
def test_migration_applies_function_style_foreign_key_pragmas(
    migration_db: tuple[Path, DatabaseMigrator], enforcement: str,
) -> None:
    """Function-style PRAGMAs control foreign key enforcement before migration SQL."""
    db_path, migrator = migration_db
    migration = db_migration_module.Migration(1, "foreign_keys", f"""
        PRAGMA foreign_keys({enforcement});
        CREATE TABLE parent (id INTEGER PRIMARY KEY);
        CREATE TABLE child (parent_id INTEGER REFERENCES parent(id));
        INSERT INTO child VALUES (7);
        PRAGMA foreign_keys(ON);
    """)
    if enforcement == "ON":
        with pytest.raises(MigrationError, match="FOREIGN KEY constraint failed"):
            migrator.execute_migration(migration)
        with sqlite3.connect(db_path) as conn:
            assert conn.execute("SELECT name FROM sqlite_master WHERE name='child'").fetchall() == []
    else:
        migrator.execute_migration(migration)
        with sqlite3.connect(db_path) as conn:
            assert conn.execute("SELECT parent_id FROM child").fetchall() == [(7,)]


@pytest.mark.unit
@pytest.mark.parametrize("wrapped", [False, True], ids=["unwrapped", "wrapped"])
def test_migration_bookkeeping_failure_rolls_back_sql_and_allows_retry(
    versioned_migration_db: tuple[Path, DatabaseMigrator], wrapped: bool,
) -> None:
    """SQL, successful ledger entry, and schema version share one commit."""
    db_path, migrator = versioned_migration_db
    with sqlite3.connect(db_path) as conn:
        conn.executescript("""
            CREATE TRIGGER reject_version BEFORE UPDATE ON schema_version
            BEGIN SELECT RAISE(ABORT, 'version write failed'); END;
        """)
    sql = "CREATE TABLE widgets (id INTEGER); INSERT INTO widgets VALUES (7);"
    if wrapped:
        sql = "PRAGMA foreign_keys=OFF; BEGIN TRANSACTION; " + sql + " COMMIT; PRAGMA foreign_keys=ON;"
    migration = db_migration_module.Migration(1, "widgets", sql)

    with pytest.raises(MigrationError, match="version write failed"):
        migrator.execute_migration(migration)
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT name FROM sqlite_master WHERE name='widgets'").fetchall() == []
        assert conn.execute("SELECT version, success FROM schema_migrations").fetchall() == [(1, 0)]
        assert conn.execute("SELECT version FROM schema_version").fetchone() == (0,)
        conn.execute("DROP TRIGGER reject_version")

    migrator.execute_migration(migration)
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT id FROM widgets").fetchall() == [(7,)]
        assert conn.execute("SELECT version, success FROM schema_migrations").fetchall() == [(1, 1)]
        assert conn.execute("SELECT version FROM schema_version").fetchone() == (1,)


@pytest.mark.unit
@pytest.mark.parametrize("control", [
    pytest.param("/* boundary */ COMMIT;", id="leading-comment-commit"),
    pytest.param("COMMIT /* boundary */;", id="inline-comment-commit"),
    pytest.param("END -- boundary\n;", id="line-comment-end"),
    pytest.param("\ufeffCOMMIT;", id="embedded-bom-commit"),
    pytest.param("SAVEPOINT 'with spaces';", id="quoted-savepoint"),
    pytest.param("PRAGMA main.foreign_keys = 'OFF';", id="qualified-foreign-keys"),
])
def test_migration_body_cannot_escape_owned_transaction(
    migration_db: tuple[Path, DatabaseMigrator], control: str,
) -> None:
    """SQLite syntax variants must not bypass the migration transaction owner."""
    db_path, migrator = migration_db
    migration = db_migration_module.Migration(
        1, "escape", "CREATE TABLE partial (id INTEGER); " + control,
    )
    with pytest.raises(MigrationError):
        migrator.execute_migration(migration)
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT name FROM sqlite_master WHERE name='partial'").fetchall() == []
        assert conn.execute("SELECT success FROM schema_migrations").fetchall() == [(0,)]


@pytest.mark.unit
def test_migration_preserves_trigger_bodies_and_semicolons_in_literals(
    migration_db: tuple[Path, DatabaseMigrator],
) -> None:
    """Trigger statements and SQL-like literal text survive migration execution."""
    db_path, migrator = migration_db
    migrator.execute_migration(db_migration_module.Migration(1, "trigger", """
        BEGIN;
        CREATE TABLE widgets (value TEXT);
        CREATE TABLE audit (value TEXT);
        CREATE TRIGGER record_widget AFTER INSERT ON widgets
        BEGIN
            INSERT INTO audit VALUES ('COMMIT; /* literal */');
            INSERT INTO audit VALUES (NEW.value);
        END;
        INSERT INTO widgets VALUES ('one;two');
        COMMIT;
    """))
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT value FROM audit ORDER BY rowid").fetchall() == [
            ("COMMIT; /* literal */",), ("one;two",),
        ]


@pytest.mark.unit
@given(st.lists(st.text(alphabet="abc;'-/*\n", max_size=30), max_size=8))
def test_split_statements_preserves_sql_literal_values(values: list[str]) -> None:
    """Comment markers, quotes and semicolons in values are not SQL boundaries."""
    statements = ["CREATE TABLE widgets (value TEXT);"]
    statements.extend(
        "INSERT INTO widgets VALUES ('" + value.replace("'", "''") + "');"
        for value in values
    )
    with sqlite3.connect(":memory:") as conn:
        for statement in DatabaseMigrator._split_sql_statements("\n".join(statements)):
            conn.execute(statement)
        assert conn.execute("SELECT value FROM widgets ORDER BY rowid").fetchall() == [
            (value,) for value in values
        ]


@pytest.mark.unit
def test_failed_downgrade_preserves_schema_data_and_success_ledger(
    migration_db: tuple[Path, DatabaseMigrator],
) -> None:
    """A failed downgrade preserves the last successful schema, data, and ledger."""
    db_path, migrator = migration_db
    migration = db_migration_module.Migration(
        1, "widgets", "CREATE TABLE widgets (id INTEGER); INSERT INTO widgets VALUES (7);",
        "DROP TABLE widgets; INSERT INTO missing VALUES (1);",
    )
    migrator.execute_migration(migration)

    with pytest.raises(MigrationError, match="missing"):
        migrator.execute_migration(migration, "down")
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT id FROM widgets").fetchall() == [(7,)]
        assert conn.execute("SELECT version, success FROM schema_migrations").fetchall() == [(1, 1)]


@pytest.mark.unit
def test_success_ledger_failure_rolls_back_migration_sql(
    migration_db: tuple[Path, DatabaseMigrator],
) -> None:
    """Failure to record success rolls back the schema and records a failed attempt."""
    db_path, migrator = migration_db
    with sqlite3.connect(db_path) as conn:
        conn.executescript("""
            CREATE TRIGGER reject_success BEFORE INSERT ON schema_migrations
            WHEN NEW.success = 1
            BEGIN SELECT RAISE(ABORT, 'ledger write failed'); END;
        """)
    with pytest.raises(MigrationError, match="ledger write failed"):
        migrator.execute_migration(db_migration_module.Migration(
            1, "widgets", "CREATE TABLE widgets (id INTEGER);",
        ))
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT name FROM sqlite_master WHERE name='widgets'").fetchall() == []
        assert conn.execute("SELECT version, success FROM schema_migrations").fetchall() == [(1, 0)]


@pytest.mark.unit
def test_migration_executes_statements_after_leading_comments(
    migration_db: tuple[Path, DatabaseMigrator],
) -> None:
    """Comments before complete statements do not suppress their database effects."""
    db_path, migrator = migration_db
    migrator.execute_migration(db_migration_module.Migration(1, "comments", """
        -- comment without semicolon
        CREATE TABLE first (id INTEGER);
        INSERT INTO first VALUES (1);
    """))
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT id FROM first").fetchall() == [(1,)]


@pytest.mark.unit
@pytest.mark.parametrize("header", [
    pytest.param("", id="before-wrapper"),
    pytest.param("-- wrapped migration\n", id="before-comment"),
    pytest.param("PRAGMA foreign_keys(OFF);\n", id="before-pragma"),
])
def test_migrate_bom_prefixed_wrapped_file_preserves_source_and_checksum(
    versioned_migration_db: tuple[Path, DatabaseMigrator], header: str,
) -> None:
    """A leading BOM permits legacy wrappers without changing recorded source integrity."""
    db_path, migrator = versioned_migration_db
    source_sql = "\ufeff" + header + """BEGIN TRANSACTION;
CREATE TABLE widgets (value TEXT);
INSERT INTO widgets VALUES ('keep \ufeff inside literal');
COMMIT;
PRAGMA foreign_keys(ON);
"""
    source_path = Path(migrator.migrations_dir) / "001_widgets.sql"
    source_path.write_text(source_sql, encoding="utf-8")
    original = db_migration_module.Migration(1, "widgets", source_sql)

    result = migrator.migrate_to_version(1, create_backup=False)

    assert result["status"] == "success"
    assert source_path.read_text(encoding="utf-8") == source_sql
    assert migrator.load_migrations()[0].up_sql == source_sql
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT value FROM widgets").fetchall() == [("keep \ufeff inside literal",)]
        assert conn.execute("SELECT checksum, success FROM schema_migrations").fetchall() == [
            (original.checksum, 1),
        ]
        assert conn.execute("SELECT version FROM schema_version").fetchone() == (1,)


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
