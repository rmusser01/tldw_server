from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any, List, Tuple

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType, DatabaseConfig
from tldw_Server_API.app.core.DB_Management import migration_tools


class _StubPool:
    def close_all(self) -> None:
        pass


class _StubBackend:
    def __init__(self) -> None:
        self.executed: List[Tuple[str, Any]] = []
        self.executed_many: List[Tuple[str, List[Tuple[Any, ...]]]] = []

    def transaction(self):

        class _Ctx:
            def __init__(self, backend: "_StubBackend") -> None:
                self.backend = backend

            def __enter__(self):

                return self.backend

            def __exit__(self, exc_type, exc, tb) -> None:

                return False

        return _Ctx(self)

    def execute(self, sql: str, params: Any = None, connection: Any = None):
        self.executed.append((sql, params))
        return None

    def execute_many(self, sql: str, params_list: List[Tuple[Any, ...]], connection: Any = None):
        self.executed_many.append((sql, params_list))
        return None

    @staticmethod
    def escape_identifier(identifier: str) -> str:
        escaped = str(identifier).replace('"', '""')
        return f'"{escaped}"'

    def get_pool(self) -> _StubPool:

        return _StubPool()


@pytest.fixture
def sqlite_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "source.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("CREATE TABLE parent (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL)")
    conn.execute(
        "CREATE TABLE child (id INTEGER PRIMARY KEY AUTOINCREMENT, parent_id INTEGER NOT NULL, "
        "FOREIGN KEY(parent_id) REFERENCES parent(id))"
    )
    conn.executemany("INSERT INTO parent (name) VALUES (?)", [("p1",), ("p2",)])
    conn.executemany("INSERT INTO child (parent_id) VALUES (?)", [(1,), (2,)])
    conn.commit()
    conn.close()
    return db_path


def test_migrate_sqlite_to_postgres_uses_backend(monkeypatch: pytest.MonkeyPatch, sqlite_db: Path) -> None:
    stub_backend = _StubBackend()

    def fake_create_backend(config: DatabaseConfig) -> _StubBackend:
        return stub_backend

    monkeypatch.setattr(migration_tools.DatabaseBackendFactory, "create_backend", fake_create_backend)

    config = DatabaseConfig(backend_type=BackendType.POSTGRESQL)
    migration_tools.migrate_sqlite_to_postgres(sqlite_db, config, batch_size=10, label="test")

    delete_statements = [sql for sql, _params in stub_backend.executed if sql.startswith("DELETE FROM")]
    assert delete_statements == ['DELETE FROM "child"', 'DELETE FROM "parent"']

    inserted_tables = [call[0].split()[2].strip('"') for call in stub_backend.executed_many]
    assert inserted_tables == ["parent", "child"]

    parent_rows = stub_backend.executed_many[0][1]
    assert parent_rows == [(1, "p1"), (2, "p2")]
    child_rows = stub_backend.executed_many[1][1]
    assert child_rows == [(1, 1), (2, 2)]

    sequence_updates = [sql for sql, _params in stub_backend.executed if sql.startswith("SELECT setval")]
    assert any("parent" in sql for sql in sequence_updates)
    assert any("child" in sql for sql in sequence_updates)


def test_migrate_sqlite_to_postgres_user_scoped_truncate(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    db_path = tmp_path / "source_user.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("CREATE TABLE parent (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, name TEXT NOT NULL)")
    conn.execute("CREATE TABLE child (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, parent_id INTEGER NOT NULL)")
    conn.executemany("INSERT INTO parent (user_id, name) VALUES (?, ?)", [("u1", "p1"), ("u2", "p2")])
    conn.executemany("INSERT INTO child (user_id, parent_id) VALUES (?, ?)", [("u1", 1), ("u2", 2)])
    conn.commit()
    conn.close()

    stub_backend = _StubBackend()

    def fake_create_backend(config: DatabaseConfig) -> _StubBackend:
        return stub_backend

    monkeypatch.setattr(migration_tools.DatabaseBackendFactory, "create_backend", fake_create_backend)

    config = DatabaseConfig(backend_type=BackendType.POSTGRESQL)
    migration_tools.migrate_sqlite_to_postgres(
        db_path,
        config,
        batch_size=10,
        label="test",
        user_id="u1",
    )

    delete_statements = [(sql, params) for sql, params in stub_backend.executed if sql.startswith("DELETE FROM")]
    assert delete_statements == [
        ('DELETE FROM "child" WHERE "user_id" = %s', ("u1",)),
        ('DELETE FROM "parent" WHERE "user_id" = %s', ("u1",)),
    ]

    inserted_tables = [call[0].split()[2].strip('"') for call in stub_backend.executed_many]
    assert inserted_tables == ["parent", "child"]
    parent_rows = stub_backend.executed_many[0][1]
    assert parent_rows == [(1, "u1", "p1")]
    child_rows = stub_backend.executed_many[1][1]
    assert child_rows == [(1, "u1", 1)]


def test_copy_table_escapes_double_quote_identifiers(tmp_path: Path) -> None:
    table_name = 'odd"name'
    col_name = 'col"name'
    db_path = tmp_path / "source_identifiers.db"

    conn = sqlite3.connect(str(db_path))
    conn.execute(
        'CREATE TABLE "odd""name" ("id" INTEGER PRIMARY KEY AUTOINCREMENT, "col""name" TEXT NOT NULL)'
    )
    conn.execute(
        'INSERT INTO "odd""name" ("col""name") VALUES (?)',
        ("value",),
    )
    conn.commit()
    conn.close()

    sqlite_conn = sqlite3.connect(str(db_path))
    sqlite_conn.row_factory = sqlite3.Row

    meta = migration_tools.TableMeta(
        name=table_name,
        source_name=table_name,
        columns=["id", col_name],
        pg_columns=["id", col_name],
        pk_columns=["id"],
        sequence_columns=["id"],
    )

    stub_backend = _StubBackend()
    migration_tools._copy_table(sqlite_conn, stub_backend, object(), meta, batch_size=50)
    sqlite_conn.close()

    assert len(stub_backend.executed_many) == 1
    insert_sql, inserted_rows = stub_backend.executed_many[0]
    assert insert_sql.startswith('INSERT INTO "odd""name" ("id", "col""name") VALUES')
    assert inserted_rows == [(1, "value")]


def test_sync_sequences_escapes_single_quote_literals() -> None:
    stub_backend = _StubBackend()
    meta = migration_tools.TableMeta(
        name="odd'name",
        source_name="odd'name",
        columns=["id'name"],
        pg_columns=["id'name"],
        pk_columns=["id'name"],
        sequence_columns=["id'name"],
    )

    migration_tools._sync_sequences(stub_backend, object(), {meta.name: meta})

    sequence_sql = [sql for sql, _params in stub_backend.executed if sql.startswith("SELECT setval")]
    assert len(sequence_sql) == 1
    sql = sequence_sql[0]
    assert "pg_get_serial_sequence('odd''name', 'id''name')" in sql
    assert 'MAX("id\'name") FROM "odd\'name"' in sql


def test_migrate_sqlite_to_postgres_does_not_log_sensitive_label(
    monkeypatch: pytest.MonkeyPatch,
    sqlite_db: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    stub_backend = _StubBackend()

    def fake_create_backend(config: DatabaseConfig) -> _StubBackend:
        return stub_backend

    monkeypatch.setattr(migration_tools.DatabaseBackendFactory, "create_backend", fake_create_backend)

    config = DatabaseConfig(backend_type=BackendType.POSTGRESQL)
    sensitive_label = "postgresql://alice:super-secret@example.test/app"

    with caplog.at_level(logging.DEBUG, logger=migration_tools.logger.name):
        migration_tools.migrate_sqlite_to_postgres(
            sqlite_db,
            config,
            batch_size=10,
            label=sensitive_label,
        )

    assert sensitive_label not in caplog.text
    assert "super-secret" not in caplog.text
