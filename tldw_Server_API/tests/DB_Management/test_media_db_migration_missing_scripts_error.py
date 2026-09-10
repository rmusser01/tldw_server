"""Tests for MediaDatabase upgrade diagnostics when migration scripts are missing."""

import pathlib
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from typing import Any, cast

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import (
    BackendType,
    ConnectionPool,
    DatabaseConfig,
)
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.backends.sqlite_backend import SQLiteConnectionPool
from tldw_Server_API.app.core.DB_Management.media_db.errors import DatabaseError
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.DB_Management.media_db.schema.backends import (
    sqlite_helpers as sqlite_helpers_module,
)


@pytest.fixture
def uncached_sqlite_pool(tmp_path: pathlib.Path) -> ConnectionPool:
    """Provide a compliant injected pool without SQLite-specific cache methods."""
    class UncachedPool(ConnectionPool):
        """Open one SQLite connection per borrow without retaining cached handles."""

        def get_connection(self) -> sqlite3.Connection:
            """Open an independently owned connection."""
            conn = sqlite3.connect(tmp_path / "injected.db")
            conn.row_factory = sqlite3.Row
            return conn

        def return_connection(self, connection: sqlite3.Connection) -> None:
            """Close an independently borrowed connection."""
            connection.close()

        @contextmanager
        def connection(self) -> Iterator[sqlite3.Connection]:
            """Close the borrowed connection when its context ends."""
            with closing(self.get_connection()) as conn:
                yield conn

        def close_all(self) -> None:
            """There are no cached connections to close."""

        def get_stats(self) -> dict[str, Any]:
            """Report that this pool retains no cached handles."""
            return {"total_connections": 0}

    return UncachedPool()


@pytest.mark.unit
def test_injected_sqlite_pool_retains_legacy_guidance_and_closes_connection(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    uncached_sqlite_pool: ConnectionPool,
) -> None:
    """An injected generic pool closes failed startup handles and preserves recovery guidance."""
    db_path = tmp_path / "injected.db"
    with uncached_sqlite_pool.connection() as conn:
        conn.execute("CREATE TABLE schema_version (version INTEGER)")
        conn.execute("INSERT INTO schema_version VALUES (21)")
        conn.commit()

    backend = DatabaseBackendFactory.create_backend(
        DatabaseConfig(backend_type=BackendType.SQLITE, sqlite_path=str(db_path)),
    )
    monkeypatch.setattr(backend, "get_pool", lambda: uncached_sqlite_pool)
    with closing(uncached_sqlite_pool.get_connection()) as startup_conn:
        monkeypatch.setattr(uncached_sqlite_pool, "get_connection", lambda: startup_conn)
        with pytest.raises(DatabaseError, match="unsupported legacy Media DB schema version 21") as exc_info:
            MediaDatabase(db_path=str(db_path), client_id="injected-legacy", backend=backend)
        assert "Docs/Database_Migrations.md" in str(exc_info.value)
        with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
            startup_conn.execute("SELECT 1")


@pytest.mark.unit
def test_media_db_upgrade_no_migrations_reports_explicit_diagnostics(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that upgrading with no migration scripts raises DatabaseError with full diagnostics."""
    db_path = tmp_path / "Media_DB_v2.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE schema_version (version INTEGER)")
        conn.execute("INSERT INTO schema_version (version) VALUES (22)")
        conn.commit()

    fake_migrations_dir = str(tmp_path / "missing_migrations")

    class _FakeMigrator:
        def __init__(self, _db_path: str, migrations_dir: str | None = None) -> None:
            self.migrations_dir = migrations_dir or fake_migrations_dir

        def migrate_to_version(
            self,
            target_version: int,
            _create_backup: bool = True,
        ) -> dict[str, Any]:
            return {
                "status": "no_migrations",
                "current_version": 22,
                "target_version": target_version,
                "migrations_applied": [],
                "migrations_dir": fake_migrations_dir,
                "available_versions": [22],
                "missing_versions": list(range(23, target_version + 1)),
            }

    monkeypatch.setattr(sqlite_helpers_module, "DatabaseMigrator", _FakeMigrator)

    with pytest.raises(DatabaseError) as exc_info:
        MediaDatabase(db_path=str(db_path), client_id="migration-diagnostics-test")

    msg = str(exc_info.value)
    target_version = MediaDatabase._CURRENT_SCHEMA_VERSION
    assert (
        f"No migration scripts available to upgrade database schema from version 22 to {target_version}"
        in msg
    )
    assert f"migrations_dir={fake_migrations_dir}" in msg
    assert "discovered_versions=[22]" in msg
    expected_missing = str(list(range(23, target_version + 1)))
    assert f"missing_versions={expected_missing}" in msg


@pytest.mark.unit
@pytest.mark.parametrize("legacy_version", [1, 8, 21], ids=["oldest-schema", "schema-eight", "below-supported-boundary"])
def test_media_db_rejects_unsupported_legacy_schema_before_packaged_migrations(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    legacy_version: int,
) -> None:
    """Legacy rejection preserves data and releases the shared pool's thread connection."""
    db_path = tmp_path / "Media_DB_v2.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE schema_version (version INTEGER)")
        conn.execute("INSERT INTO schema_version (version) VALUES (?)", (legacy_version,))
        conn.execute("CREATE TABLE preserved_content (content TEXT)")
        conn.execute("INSERT INTO preserved_content VALUES ('keep this content')")
        conn.commit()

    class _UnexpectedMigrator:
        """Fail if automatic migration is attempted for an unsupported legacy schema."""

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            """Expose accidental migrator construction as a test failure."""
            raise AssertionError("unsupported legacy schemas must not invoke DatabaseMigrator")

    monkeypatch.setattr(sqlite_helpers_module, "DatabaseMigrator", _UnexpectedMigrator)

    backend = DatabaseBackendFactory.create_backend(
        DatabaseConfig(backend_type=BackendType.SQLITE, sqlite_path=str(db_path)),
    )
    pool = cast(SQLiteConnectionPool, backend.get_pool())
    original_conn = pool.get_connection()
    try:
        with pytest.raises(DatabaseError) as exc_info:
            MediaDatabase(db_path=str(db_path), client_id="legacy-boundary-test")

        assert pool.get_stats()["active_connections"] == 0
        with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
            original_conn.execute("SELECT 1")
        replacement = pool.get_connection()
        assert replacement is not original_conn
        assert replacement.execute("SELECT version FROM schema_version").fetchone()[0] == legacy_version
    finally:
        pool.clear_thread_local_connection()

    msg = str(exc_info.value)
    assert f"unsupported legacy Media DB schema version {legacy_version}" in msg
    assert "minimum supported automatic upgrade version is 22" in msg
    assert "backup" in msg
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT version FROM schema_version").fetchone() == (legacy_version,)
        assert conn.execute("SELECT content FROM preserved_content").fetchall() == [("keep this content",)]
        assert conn.execute("SELECT name FROM sqlite_master WHERE name='schema_migrations'").fetchall() == []
