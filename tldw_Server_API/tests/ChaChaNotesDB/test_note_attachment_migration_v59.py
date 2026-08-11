"""Schema-v59 contract tests for the canonical Notes attachment registry."""

from __future__ import annotations

import contextlib
import inspect
import sqlite3
import threading
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType, QueryResult
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    SchemaError,
)

pytestmark = pytest.mark.unit

OWNER = "attachment-migration-owner"
NOTE_ID = "11111111-1111-4111-8111-111111111111"
ATTACHMENT_ID = "22222222-2222-4222-8222-222222222222"
DATASET_ID = "dataset-notes-default"
CREATED_AT = "2026-08-11T12:00:00+00:00"
BLOB_HASH = "sha256:" + "a" * 64
OBJECT_HASH = "sha256:" + "b" * 64


def _prepare_v58_database(db_path: Path) -> None:
    """Create the preceding real schema, then remove only v59 state."""

    db = CharactersRAGDB(str(db_path), client_id=OWNER)
    try:
        if db.get_note_by_id(NOTE_ID) is None:
            db.add_note("Attachment parent", "Body", note_id=NOTE_ID)
    finally:
        db.close_all_connections()

    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute("DROP TABLE IF EXISTS note_attachments")
        conn.execute(
            "UPDATE db_schema_version SET version = 58 WHERE schema_name = ?",
            (CharactersRAGDB._SCHEMA_NAME,),
        )


def _registry_schema(conn: sqlite3.Connection) -> tuple[str, tuple[tuple[str, str], ...]]:
    table_sql = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'note_attachments'"
    ).fetchone()[0]
    indexes = tuple(
        sorted(
            (str(row[0]), str(row[1]))
            for row in conn.execute(
                "SELECT name, sql FROM sqlite_master "
                "WHERE type = 'index' AND tbl_name = 'note_attachments' AND sql IS NOT NULL"
            ).fetchall()
        )
    )
    return " ".join(str(table_sql).split()), indexes


def _valid_insert_values(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "client_id": OWNER,
        "dataset_id": DATASET_ID,
        "attachment_id": ATTACHMENT_ID,
        "note_id": NOTE_ID,
        "file_name": "Report.pdf",
        "normalized_file_name": "report.pdf",
        "original_file_name": "Report.pdf",
        "content_type": "application/pdf",
        "size_bytes": 42,
        "blob_hash": BLOB_HASH,
        "object_hash": OBJECT_HASH,
        "version": 1,
        "deleted": 0,
        "deleted_at": None,
        "delete_reason": None,
        "created_at": CREATED_AT,
        "last_modified": CREATED_AT,
        "created_by": "device-a",
        "source_kind": "sync",
    }
    values.update(overrides)
    return values


def _raw_insert(conn: sqlite3.Connection, **overrides: object) -> None:
    values = _valid_insert_values(**overrides)
    columns = tuple(values)
    conn.execute(
        f"INSERT INTO note_attachments({', '.join(columns)}) "  # nosec B608 - fixed test keys.
        f"VALUES ({', '.join('?' for _ in columns)})",
        tuple(values[column] for column in columns),
    )


def test_sqlite_v58_to_v59_creates_empty_canonical_registry(tmp_path: Path) -> None:
    db_path = tmp_path / "attachments-v58.sqlite"
    _prepare_v58_database(db_path)

    migrated = CharactersRAGDB(str(db_path), client_id=OWNER)
    try:
        with migrated.transaction() as conn:
            assert migrated._get_db_version(conn) == 59
            columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(note_attachments)")}
            assert columns == {
                "client_id",
                "dataset_id",
                "attachment_id",
                "note_id",
                "file_name",
                "normalized_file_name",
                "original_file_name",
                "content_type",
                "size_bytes",
                "blob_hash",
                "object_hash",
                "version",
                "deleted",
                "deleted_at",
                "delete_reason",
                "created_at",
                "last_modified",
                "created_by",
                "source_kind",
            }
            assert conn.execute("SELECT COUNT(*) FROM note_attachments").fetchone()[0] == 0
            assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
            foreign_keys = conn.execute("PRAGMA foreign_key_list(note_attachments)").fetchall()
            assert any(
                str(row[2]) == "notes"
                and str(row[3]) == "note_id"
                and str(row[6]).upper() == "RESTRICT"
                for row in foreign_keys
            )
            indexes = {str(row[1]) for row in conn.execute("PRAGMA index_list(note_attachments)")}
            assert {
                "uq_note_attachments_live_name",
                "idx_note_attachments_owner_dataset_note_page",
                "idx_note_attachments_owner_dataset_blob",
            } <= indexes
    finally:
        migrated.close_all_connections()


def test_sqlite_fresh_and_v58_upgrade_registry_schema_are_identical(tmp_path: Path) -> None:
    fresh_path = tmp_path / "attachments-fresh.sqlite"
    upgrade_path = tmp_path / "attachments-upgrade.sqlite"
    _prepare_v58_database(upgrade_path)

    fresh = CharactersRAGDB(str(fresh_path), client_id=OWNER)
    upgraded = CharactersRAGDB(str(upgrade_path), client_id=OWNER)
    try:
        with sqlite3.connect(fresh_path) as fresh_conn, sqlite3.connect(upgrade_path) as upgraded_conn:
            assert _registry_schema(fresh_conn) == _registry_schema(upgraded_conn)
    finally:
        fresh.close_all_connections()
        upgraded.close_all_connections()


@pytest.mark.parametrize(
    "overrides",
    [
        {"attachment_id": "not-a-uuid"},
        {"attachment_id": "22222222-2222-1222-8222-222222222222"},
        {"note_id": "not-a-uuid"},
        {"note_id": "11111111-1111-4111-7111-111111111111"},
        {"size_bytes": 0},
        {"version": 0},
        {"blob_hash": "sha256:" + "A" * 64},
        {"blob_hash": "a" * 64},
        {"object_hash": "sha256:" + "B" * 64},
        {"file_name": "x" * 181},
        {"normalized_file_name": "x" * 181},
        {"original_file_name": "x" * 256},
        {"original_file_name": "é" * 600},
        {"content_type": "x" * 256},
        {"delete_reason": "x" * 257, "deleted": 1, "deleted_at": CREATED_AT},
        {"source_kind": "filesystem"},
        {"deleted": 1, "deleted_at": None},
        {"deleted": 0, "deleted_at": CREATED_AT},
        {"deleted": 0, "delete_reason": "reason"},
    ],
)
def test_sqlite_v59_rejects_noncanonical_registry_rows(
    tmp_path: Path,
    overrides: dict[str, object],
) -> None:
    db = CharactersRAGDB(str(tmp_path / f"invalid-{uuid4()}.sqlite"), client_id=OWNER)
    try:
        db.add_note("Parent", "Body", note_id=NOTE_ID)
        with pytest.raises(sqlite3.IntegrityError), db.transaction() as conn:
            _raw_insert(conn, **overrides)
    finally:
        db.close_all_connections()


def test_sqlite_v59_restricts_hard_note_delete(tmp_path: Path) -> None:
    db = CharactersRAGDB(str(tmp_path / "restrict-note-delete.sqlite"), client_id=OWNER)
    try:
        db.add_note("Parent", "Body", note_id=NOTE_ID)
        with db.transaction() as conn:
            _raw_insert(conn)
        with pytest.raises(sqlite3.IntegrityError), db.transaction() as conn:
            conn.execute("DELETE FROM notes WHERE id = ?", (NOTE_ID,))
    finally:
        db.close_all_connections()


def test_sqlite_v58_to_v59_fails_closed_on_preexisting_registry_collision(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "attachments-collision.sqlite"
    _prepare_v58_database(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE note_attachments(attachment_id TEXT PRIMARY KEY)")

    with pytest.raises(CharactersRAGDBError, match="note_attachments|registry|collision"):
        CharactersRAGDB(str(db_path), client_id=OWNER)

    with sqlite3.connect(db_path) as conn:
        version = conn.execute(
            "SELECT version FROM db_schema_version WHERE schema_name = ?",
            (CharactersRAGDB._SCHEMA_NAME,),
        ).fetchone()[0]
        assert version == 58
        assert [row[1] for row in conn.execute("PRAGMA table_info(note_attachments)")] == [
            "attachment_id"
        ]


def test_sqlite_v59_post_ddl_failure_rolls_back_registry_and_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "attachments-post-ddl-failure.sqlite"
    _prepare_v58_database(db_path)
    create_schema = CharactersRAGDB._create_note_attachment_schema_sqlite

    def fail_after_ddl(self: CharactersRAGDB, conn: sqlite3.Connection) -> None:
        create_schema(self, conn)
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'note_attachments'"
        ).fetchone() is not None
        raise SchemaError("injected post-DDL registry failure")

    monkeypatch.setattr(
        CharactersRAGDB,
        "_create_note_attachment_schema_sqlite",
        fail_after_ddl,
    )
    with pytest.raises(CharactersRAGDBError, match="injected post-DDL registry failure"):
        CharactersRAGDB(str(db_path), client_id=OWNER)

    with sqlite3.connect(db_path) as conn:
        assert conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'note_attachments'"
        ).fetchone() is None
        assert conn.execute(
            "SELECT version FROM db_schema_version WHERE schema_name = ?",
            (CharactersRAGDB._SCHEMA_NAME,),
        ).fetchone()[0] == 58


def test_sqlite_v59_initializer_serializes_on_one_schema_authority(tmp_path: Path) -> None:
    db_path = tmp_path / "attachments-concurrent.sqlite"
    _prepare_v58_database(db_path)
    barrier = threading.Barrier(3)
    versions: list[int] = []
    errors: list[BaseException] = []

    def initialize() -> None:
        barrier.wait()
        try:
            db = CharactersRAGDB(str(db_path), client_id=OWNER)
            try:
                with db.transaction() as conn:
                    versions.append(db._get_db_version(conn))
            finally:
                db.close_all_connections()
        except Exception as exc:  # noqa: BLE001  # pragma: no cover - asserted below
            errors.append(exc)

    threads = [threading.Thread(target=initialize) for _ in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=10)

    assert not any(thread.is_alive() for thread in threads)
    assert errors == []
    assert sorted(versions) == [59, 59]
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM note_attachments").fetchone()[0] == 0


class _PostgresMigrationBackend:
    backend_type = BackendType.POSTGRESQL

    def __init__(
        self,
        *,
        schema_owner: bool = True,
        notes_forced: bool = True,
        registry_exists: bool = False,
    ) -> None:
        self.schema_owner = schema_owner
        self.notes_forced = notes_forced
        self.registry_exists = registry_exists
        self.calls: list[tuple[str, object]] = []

    def table_exists(self, table_name: str, *, connection: object) -> bool:
        assert connection is not None
        if table_name == "note_attachments":
            return self.registry_exists
        return True

    def execute(
        self,
        statement: str,
        params: object = None,
        *,
        connection: object,
    ) -> QueryResult:
        normalized = " ".join(statement.split())
        self.calls.append((normalized, params))
        if "FROM pg_class AS table_row" in normalized:
            relation_names = ["notes"]
            if "table_row.relname IN ('notes', 'note_attachments')" in normalized:
                relation_names.append("note_attachments")
            return QueryResult(
                rows=[
                    {
                        "table_name": table_name,
                        "relrowsecurity": self.notes_forced,
                        "relforcerowsecurity": self.notes_forced,
                        "is_schema_owner": self.schema_owner,
                        "is_current_schema_owner": self.schema_owner,
                    }
                    for table_name in relation_names
                ],
                rowcount=len(relation_names),
            )
        if "FROM pg_namespace" in normalized and "is_current_schema_owner" in normalized:
            return QueryResult(
                rows=[{"is_current_schema_owner": self.schema_owner}],
                rowcount=1,
            )
        if normalized.startswith("SELECT COUNT("):
            return QueryResult(rows=[{"count": 0}], rowcount=1)
        return QueryResult(rows=[], rowcount=0)


def _postgres_db(backend: _PostgresMigrationBackend) -> CharactersRAGDB:
    db = CharactersRAGDB.__new__(CharactersRAGDB)
    db._local = type("Local", (), {})()
    db._backend = backend
    db._uses_shared_content_backend = False
    return db


def test_postgres_v59_migration_uses_verified_lock_rls_and_version_order() -> None:
    backend = _PostgresMigrationBackend()
    db = _postgres_db(backend)

    assert hasattr(db, "_migrate_from_v58_to_v59_postgres")
    db._migrate_from_v58_to_v59_postgres(object())

    statements = [statement for statement, _ in backend.calls]
    lock_index = next(i for i, sql in enumerate(statements) if sql.startswith("LOCK TABLE"))
    catalog_index = next(i for i, sql in enumerate(statements) if "FROM pg_class AS table_row" in sql)
    no_force_index = statements.index("ALTER TABLE notes NO FORCE ROW LEVEL SECURITY")
    create_index = next(i for i, sql in enumerate(statements) if sql.startswith("CREATE TABLE note_attachments"))
    force_index = statements.index("ALTER TABLE notes FORCE ROW LEVEL SECURITY")
    policy_index = next(
        i
        for i, sql in enumerate(statements)
        if "CREATE POLICY note_attachments_tenant_isolation" in sql
    )
    final_catalog_index = max(
        i for i, sql in enumerate(statements) if "FROM pg_class AS table_row" in sql
    )
    version_index = next(i for i, sql in enumerate(statements) if "INSERT INTO db_schema_version" in sql)

    assert statements[lock_index].startswith("LOCK TABLE notes IN ACCESS EXCLUSIVE MODE")
    assert (
        lock_index
        < catalog_index
        < no_force_index
        < create_index
        < force_index
        < policy_index
        < final_catalog_index
        < version_index
    )
    joined = "\n".join(statements)
    assert "ON DELETE RESTRICT" in joined
    assert (
        "CREATE UNIQUE INDEX uq_note_attachments_live_name "
        "ON note_attachments(client_id, dataset_id, note_id, normalized_file_name) "
        "WHERE deleted = FALSE"
    ) in statements


def test_postgres_v59_rejects_unverified_owner_before_ddl_or_force_relaxation() -> None:
    backend = _PostgresMigrationBackend(schema_owner=False)
    db = _postgres_db(backend)

    with pytest.raises(SchemaError, match="schema-owner|schema owner"):
        db._migrate_from_v58_to_v59_postgres(object())

    statements = [statement for statement, _ in backend.calls]
    assert not any("NO FORCE ROW LEVEL SECURITY" in statement for statement in statements)
    assert not any(statement.startswith("CREATE TABLE note_attachments") for statement in statements)
    assert not any("INSERT INTO db_schema_version" in statement for statement in statements)


def test_postgres_v59_rejects_preexisting_registry_before_version_advance() -> None:
    backend = _PostgresMigrationBackend(registry_exists=True)
    db = _postgres_db(backend)

    with pytest.raises(SchemaError, match="note_attachments|registry|collision"):
        db._migrate_from_v58_to_v59_postgres(object())

    assert not any(
        "INSERT INTO db_schema_version" in statement for statement, _ in backend.calls
    )


def test_postgres_initializer_locks_schema_version_before_v59_migration() -> None:
    source = inspect.getsource(CharactersRAGDB._initialize_schema_postgres)
    assert "_get_schema_version_postgres(conn, lock=True)" in source
    assert "if current_version < 59:" in source
    assert source.index("_get_schema_version_postgres(conn, lock=True)") < source.index(
        "_migrate_from_v58_to_v59_postgres(conn)"
    )


class _FailingPostgresTransactionBackend(_PostgresMigrationBackend):
    def __init__(self) -> None:
        super().__init__()
        self.rolled_back = False

    @contextlib.contextmanager
    def transaction(self):
        forced_before = self.notes_forced
        try:
            yield object()
        except Exception:
            self.notes_forced = forced_before
            self.rolled_back = True
            raise

    def execute(
        self,
        statement: str,
        params: object = None,
        *,
        connection: object,
    ) -> QueryResult:
        normalized = " ".join(statement.split())
        if "SELECT version FROM db_schema_version" in normalized:
            self.calls.append((normalized, params))
            return QueryResult(rows=[{"version": 58}], rowcount=1)
        result = super().execute(statement, params, connection=connection)
        if normalized == "ALTER TABLE notes NO FORCE ROW LEVEL SECURITY":
            self.notes_forced = False
        if normalized.startswith("CREATE TABLE note_attachments"):
            raise SchemaError("injected registry DDL failure")
        return result


def test_postgres_v59_failure_rolls_back_temporary_force_relaxation_and_version() -> None:
    backend = _FailingPostgresTransactionBackend()
    db = _postgres_db(backend)

    with pytest.raises(SchemaError, match="injected registry DDL failure"):
        db._initialize_schema_postgres()

    statements = [statement for statement, _ in backend.calls]
    assert backend.rolled_back is True
    assert backend.notes_forced is True
    assert "ALTER TABLE notes NO FORCE ROW LEVEL SECURITY" in statements
    assert not any("INSERT INTO db_schema_version" in statement for statement in statements)


class _RollbackBackend:
    backend_type = BackendType.POSTGRESQL

    def __init__(self) -> None:
        self.rolled_back = False

    @contextlib.contextmanager
    def transaction(self):
        try:
            yield object()
        except Exception:
            self.rolled_back = True
            raise

    def table_exists(self, table_name: str, *, connection: object) -> bool:
        return table_name == "db_schema_version"

    def execute(
        self,
        statement: str,
        params: object = None,
        *,
        connection: object,
    ) -> QueryResult:
        del params, connection
        if "SELECT version FROM db_schema_version" in statement:
            return QueryResult(rows=[{"version": 58}], rowcount=1)
        return QueryResult(rows=[], rowcount=0)


def test_postgres_initializer_rolls_back_failed_v59_migration(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _RollbackBackend()
    db = CharactersRAGDB.__new__(CharactersRAGDB)
    db._local = type("Local", (), {})()
    db._backend = backend
    db._uses_shared_content_backend = False
    db._CURRENT_SCHEMA_VERSION = 59

    def fail(_conn: Any) -> None:
        raise SchemaError("injected v59 failure")

    monkeypatch.setattr(db, "_migrate_from_v58_to_v59_postgres", fail, raising=False)
    with pytest.raises(SchemaError, match="injected"):
        db._initialize_schema_postgres()
    assert backend.rolled_back is True
