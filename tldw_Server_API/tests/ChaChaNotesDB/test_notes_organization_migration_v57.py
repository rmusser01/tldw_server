"""Schema v57 coverage for shared PostgreSQL Notes organization tenancy."""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import (
    BackendType,
    QueryResult,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    SchemaError,
)

_V57_LOCKED_TABLES = (
    "chacha_keywords",
    "collection_keywords",
    "conversation_keywords",
    "conversations",
    "keyword_collections",
    "note_folder_memberships",
    "note_folder_source_memberships",
    "note_folder_sync_suppressions",
    "note_folders",
    "note_keywords",
    "notes",
)


class _MigrationBackend:
    backend_type = BackendType.POSTGRESQL

    def __init__(
        self,
        *,
        invalid_owner_table: str | None = None,
        invalid_owner_value: str | None = None,
        collision_marker: str | None = None,
        cross_owner_marker: str | None = None,
        forced_rls_tables: set[str] | None = None,
        unowned_table: str | None = None,
    ) -> None:
        self.invalid_owner_table = invalid_owner_table
        self.invalid_owner_value = invalid_owner_value
        self.collision_marker = collision_marker
        self.cross_owner_marker = cross_owner_marker
        self.forced_rls_tables = forced_rls_tables or set()
        self.unowned_table = unowned_table
        self.calls: list[tuple[str, object]] = []

    def execute(
        self,
        statement: str,
        params: object = None,
        connection: object = None,
    ) -> QueryResult:
        normalized = " ".join(statement.split())
        self.calls.append((normalized, params))
        if "FROM pg_class AS table_row" in normalized:
            return QueryResult(
                rows=[
                    {
                        "table_name": table_name,
                        "relrowsecurity": table_name in self.forced_rls_tables,
                        "relforcerowsecurity": table_name in self.forced_rls_tables,
                        "is_schema_owner": table_name != self.unowned_table,
                    }
                    for table_name in _V57_LOCKED_TABLES
                ],
                rowcount=len(_V57_LOCKED_TABLES),
            )
        if "FROM pg_constraint" in normalized:
            return QueryResult(
                rows=[
                    {
                        "table_name": "chacha_keywords",
                        "conname": "chacha_keywords_keyword_key",
                        "columns": ["keyword"],
                    },
                    {
                        "table_name": "keyword_collections",
                        "conname": "keyword_collections_name_key",
                        "columns": ["name"],
                    },
                    {
                        "table_name": "note_folders",
                        "conname": "note_folders_path_key",
                        "columns": ["path"],
                    },
                ],
                rowcount=3,
            )
        count = 0
        if (
            self.invalid_owner_table is not None
            and f"FROM {self.invalid_owner_table} WHERE" in normalized
            and "client_id" in normalized
            and self.invalid_owner_value is not None
            and re.fullmatch(r"[1-9][0-9]*", self.invalid_owner_value) is None
        ):
            count = 1
        if self.collision_marker and self.collision_marker in normalized:
            count = 1
        if self.cross_owner_marker and self.cross_owner_marker in normalized:
            count = 1
        if normalized.startswith("SELECT COUNT(*)"):
            return QueryResult(rows=[{"count": count}], rowcount=1)
        return QueryResult(rows=[], rowcount=0)


def _postgres_db(backend: _MigrationBackend) -> CharactersRAGDB:
    db = CharactersRAGDB.__new__(CharactersRAGDB)
    db._local = type("Local", (), {})()
    db._backend = backend
    db._uses_shared_content_backend = False
    return db


@pytest.mark.parametrize("client_id", ["1", "desktop-client"])
def test_sqlite_v56_to_v57_is_a_non_destructive_version_step(
    tmp_path: Path,
    client_id: str,
) -> None:
    db_path = tmp_path / "notes-organization-v56.sqlite"
    initial = CharactersRAGDB(str(db_path), client_id=client_id)
    try:
        with initial.transaction() as conn:
            conn.execute(
                "INSERT INTO keywords(sync_id, keyword, client_id) VALUES (?, ?, ?)",
                ("11111111-1111-4111-8111-111111111111", "Portable", client_id),
            )
            keyword_id = conn.execute(
                "SELECT id FROM keywords WHERE sync_id = ?",
                ("11111111-1111-4111-8111-111111111111",),
            ).fetchone()["id"]
            conn.execute("DROP TABLE note_attachments")
            conn.execute(
                "UPDATE db_schema_version SET version = 56 WHERE schema_name = ?",
                (CharactersRAGDB._SCHEMA_NAME,),
            )
    finally:
        initial.close_connection()

    migrated = CharactersRAGDB(str(db_path), client_id=client_id)
    try:
        with migrated.transaction() as conn:
            assert migrated._get_db_version(conn) == CharactersRAGDB._CURRENT_SCHEMA_VERSION
            row = conn.execute(
                "SELECT id, keyword, client_id FROM keywords WHERE sync_id = ?",
                ("11111111-1111-4111-8111-111111111111",),
            ).fetchone()
            assert dict(row) == {
                "id": keyword_id,
                "keyword": "Portable",
                "client_id": client_id,
            }
        assert migrated._sqlite_linear_migration_steps()[56].__name__ == ("_migrate_from_v56_to_v57")
    finally:
        migrated.close_connection()


def test_postgres_v56_to_v57_relaxes_only_global_organization_uniqueness() -> None:
    backend = _MigrationBackend()
    db = _postgres_db(backend)

    assert hasattr(db, "_migrate_from_v56_to_v57_postgres")
    db._migrate_from_v56_to_v57_postgres(object())

    sql = "\n".join(statement for statement, _ in backend.calls)
    assert backend.calls[0][0].startswith("LOCK TABLE")
    for table_name in _V57_LOCKED_TABLES:
        assert table_name in backend.calls[0][0]
    assert "SHARE ROW EXCLUSIVE MODE" in backend.calls[0][0]
    first_select = next(
        index for index, (statement, _) in enumerate(backend.calls) if statement.startswith("SELECT COUNT(*)")
    )
    assert first_select > 0
    assert 'DROP CONSTRAINT IF EXISTS "chacha_keywords_keyword_key"' in sql
    assert 'DROP CONSTRAINT IF EXISTS "keyword_collections_name_key"' in sql
    assert 'DROP CONSTRAINT IF EXISTS "note_folders_path_key"' in sql
    assert "ON chacha_keywords(client_id, sync_id)" in sql
    assert "ON chacha_keywords(client_id, LOWER(keyword))" in sql
    assert "ON keyword_collections(client_id, sync_id)" in sql
    assert "ON keyword_collections(client_id, LOWER(name))" in sql
    assert "ON note_folders(client_id, sync_id)" in sql
    assert "ON note_folders(client_id, LOWER(path)) WHERE deleted = FALSE" in sql
    assert "client_id <> BTRIM(client_id)" in sql
    assert "BTRIM(client_id) !~ '^[1-9][0-9]*$'" in sql
    assert not any(
        statement.startswith(
            (
                "DELETE FROM",
                "UPDATE chacha_keywords",
                "UPDATE keyword_collections",
                "UPDATE note_folders",
            )
        )
        for statement, _ in backend.calls
    )
    assert any(
        "INSERT INTO db_schema_version" in statement and params == (CharactersRAGDB._SCHEMA_NAME, 57)
        for statement, params in backend.calls
    )


def test_postgres_v56_to_v57_temporarily_unforces_verified_rls_tables() -> None:
    forced_tables = {
        "chacha_keywords",
        "note_folder_source_memberships",
        "notes",
    }
    backend = _MigrationBackend(forced_rls_tables=forced_tables)
    db = _postgres_db(backend)

    db._migrate_from_v56_to_v57_postgres(object())

    statements = [statement for statement, _ in backend.calls]
    catalog_index = next(
        index for index, statement in enumerate(statements) if "FROM pg_class AS table_row" in statement
    )
    first_scan_index = next(
        index for index, statement in enumerate(statements) if statement.startswith("SELECT COUNT(*)")
    )
    version_index = next(
        index for index, statement in enumerate(statements) if "INSERT INTO db_schema_version" in statement
    )
    no_force_tables = {
        statement.split()[2].strip('"') for statement in statements if " NO FORCE ROW LEVEL SECURITY" in statement
    }
    force_tables = {
        statement.split()[2].strip('"')
        for statement in statements
        if " FORCE ROW LEVEL SECURITY" in statement and " NO FORCE " not in statement
    }

    assert statements[0].startswith("LOCK TABLE")
    assert catalog_index < first_scan_index
    assert no_force_tables == forced_tables
    assert force_tables == forced_tables
    assert (
        max(index for index, statement in enumerate(statements) if " NO FORCE ROW LEVEL SECURITY" in statement)
        < first_scan_index
    )
    assert (
        min(
            index
            for index, statement in enumerate(statements)
            if " FORCE ROW LEVEL SECURITY" in statement and " NO FORCE " not in statement
        )
        < version_index
    )


def test_postgres_v56_to_v57_rejects_unverified_schema_owner_before_scans() -> None:
    backend = _MigrationBackend(
        forced_rls_tables={"notes"},
        unowned_table="notes",
    )
    db = _postgres_db(backend)

    with pytest.raises(SchemaError, match="schema-owner migration path"):
        db._migrate_from_v56_to_v57_postgres(object())

    sql = "\n".join(statement for statement, _ in backend.calls)
    assert "NO FORCE ROW LEVEL SECURITY" not in sql
    assert "SELECT COUNT(*)" not in sql
    assert "INSERT INTO db_schema_version" not in sql


def test_postgres_v55_to_v57_installs_full_rls_after_both_migrations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _MigrationBackend()
    db = _postgres_db(backend)
    import tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB as chacha_module

    monkeypatch.setattr(
        chacha_module,
        "build_chacha_rls_sql",
        lambda: ["SELECT 'full chacha rls policy set'"],
        raising=False,
    )

    db._ensure_chacha_rls_postgres(object())

    assert backend.calls == [("SELECT 'full chacha rls policy set'", None)]
    initializer_source = inspect.getsource(CharactersRAGDB._initialize_schema_postgres)
    assert initializer_source.index("self._migrate_from_v55_to_v56_postgres(conn)") < (
        initializer_source.index("self._migrate_from_v56_to_v57_postgres(conn)")
    )
    assert initializer_source.index("self._migrate_from_v56_to_v57_postgres(conn)") < (
        initializer_source.index("self._ensure_chacha_rls_postgres(conn)")
    )


@pytest.mark.parametrize(
    "invalid_owner_table",
    ["chacha_keywords", "keyword_collections", "note_folders"],
)
@pytest.mark.parametrize(
    "invalid_owner_value",
    [" 1", "1 ", "0", "01", "server-origin", "device-worker"],
)
def test_postgres_v56_to_v57_rejects_unattributable_owner_before_schema_changes(
    invalid_owner_table: str,
    invalid_owner_value: str,
) -> None:
    backend = _MigrationBackend(
        invalid_owner_table=invalid_owner_table,
        invalid_owner_value=invalid_owner_value,
    )
    db = _postgres_db(backend)

    assert hasattr(db, "_migrate_from_v56_to_v57_postgres")
    with pytest.raises(SchemaError, match="authenticated owner"):
        db._migrate_from_v56_to_v57_postgres(object())

    sql = "\n".join(statement for statement, _ in backend.calls)
    assert "DROP CONSTRAINT" not in sql
    assert "DROP INDEX" not in sql
    assert "INSERT INTO db_schema_version" not in sql


@pytest.mark.parametrize(
    "collision_marker",
    [
        "chacha_keywords GROUP BY client_id, sync_id",
        "chacha_keywords GROUP BY client_id, LOWER(keyword)",
        "keyword_collections GROUP BY client_id, sync_id",
        "keyword_collections GROUP BY client_id, LOWER(name)",
        "note_folders GROUP BY client_id, sync_id",
        "note_folders WHERE deleted = FALSE GROUP BY client_id, LOWER(path)",
    ],
)
def test_postgres_v56_to_v57_rejects_same_owner_collision_before_schema_changes(
    collision_marker: str,
) -> None:
    backend = _MigrationBackend(collision_marker=collision_marker)
    db = _postgres_db(backend)

    assert hasattr(db, "_migrate_from_v56_to_v57_postgres")
    with pytest.raises(SchemaError, match="owner-scoped uniqueness"):
        db._migrate_from_v56_to_v57_postgres(object())

    sql = "\n".join(statement for statement, _ in backend.calls)
    assert "DROP CONSTRAINT" not in sql
    assert "DROP INDEX" not in sql
    assert "INSERT INTO db_schema_version" not in sql


@pytest.mark.parametrize(
    "cross_owner_marker",
    [
        "FROM keyword_collections AS child JOIN keyword_collections AS parent",
        "FROM note_folders AS child JOIN note_folders AS parent",
        "FROM note_keywords AS link JOIN notes AS note",
        "FROM conversation_keywords AS link JOIN conversations AS conversation",
        "FROM collection_keywords AS link JOIN keyword_collections AS collection",
        "FROM note_folder_memberships AS link JOIN notes AS note",
        "FROM note_folder_source_memberships AS link JOIN notes AS note",
        "FROM note_folder_sync_suppressions AS link JOIN notes AS note",
    ],
)
def test_postgres_v56_to_v57_rejects_cross_owner_relations_before_schema_changes(
    cross_owner_marker: str,
) -> None:
    backend = _MigrationBackend(cross_owner_marker=cross_owner_marker)
    db = _postgres_db(backend)

    with pytest.raises(SchemaError, match="cross-owner"):
        db._migrate_from_v56_to_v57_postgres(object())

    sql = "\n".join(statement for statement, _ in backend.calls)
    assert backend.calls[0][0].startswith("LOCK TABLE")
    assert "IS DISTINCT FROM" in sql
    assert "DROP CONSTRAINT" not in sql
    assert "DROP INDEX" not in sql
    assert "INSERT INTO db_schema_version" not in sql


def test_postgres_folder_startup_dedupe_and_index_are_owner_scoped() -> None:
    backend = _MigrationBackend()
    db = _postgres_db(backend)

    db._ensure_note_folder_schema_postgres(object())

    sql = "\n".join(statement for statement, _ in backend.calls)
    assert "PARTITION BY client_id, LOWER(path)" in sql
    assert (
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_note_folders_path_lower "
        "ON note_folders(client_id, LOWER(path)) WHERE deleted = FALSE"
    ) in sql


def test_postgres_v57_migration_ddl_is_idempotent() -> None:
    backend = _MigrationBackend()
    db = _postgres_db(backend)

    assert hasattr(db, "_migrate_from_v56_to_v57_postgres")
    db._migrate_from_v56_to_v57_postgres(object())
    db._migrate_from_v56_to_v57_postgres(object())

    ddl = [statement for statement, _ in backend.calls if statement.startswith(("DROP INDEX", "CREATE UNIQUE INDEX"))]
    assert ddl
    assert all("IF EXISTS" in statement for statement in ddl if statement.startswith("DROP"))
    assert all("IF NOT EXISTS" in statement for statement in ddl if statement.startswith("CREATE"))
