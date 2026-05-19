import importlib
import sqlite3
from types import SimpleNamespace
import uuid

import pytest
from tldw_Server_API.app.core.DB_Management.backends.base import (
    DatabaseError as BackendDatabaseError,
)
from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.DB_Management.media_db.schema.bootstrap import ensure_media_schema
from tldw_Server_API.app.core.DB_Management.media_db.schema import bootstrap as bootstrap_module
from tldw_Server_API.app.core.DB_Management.media_db.schema.backends import postgres as postgres_backend_module
from tldw_Server_API.app.core.DB_Management.media_db.schema.backends import sqlite as sqlite_backend_module


@pytest.mark.unit
def test_ensure_media_schema_dispatches_sqlite(monkeypatch) -> None:
    db = SimpleNamespace(backend_type=BackendType.SQLITE)
    calls: list[object] = []

    monkeypatch.setattr(bootstrap_module, "initialize_sqlite_schema", lambda value: calls.append(value))
    monkeypatch.setattr(
        bootstrap_module,
        "initialize_postgres_schema",
        lambda value: pytest.fail(f"unexpected postgres dispatch for {value!r}"),
    )

    ensure_media_schema(db)

    assert calls == [db]


@pytest.mark.unit
def test_ensure_media_schema_dispatches_postgres(monkeypatch) -> None:
    db = SimpleNamespace(backend_type=BackendType.POSTGRESQL)
    calls: list[object] = []

    monkeypatch.setattr(
        bootstrap_module,
        "initialize_sqlite_schema",
        lambda value: pytest.fail(f"unexpected sqlite dispatch for {value!r}"),
    )
    monkeypatch.setattr(bootstrap_module, "initialize_postgres_schema", lambda value: calls.append(value))

    ensure_media_schema(db)

    assert calls == [db]


@pytest.mark.unit
def test_initialize_schema_uses_bootstrap_entrypoint(monkeypatch) -> None:
    db = MediaDatabase.__new__(MediaDatabase)
    db.backend_type = BackendType.SQLITE
    calls: list[object] = []

    monkeypatch.setattr(bootstrap_module, "initialize_sqlite_schema", lambda value: calls.append(value))
    monkeypatch.setattr(
        bootstrap_module,
        "initialize_postgres_schema",
        lambda value: pytest.fail(f"unexpected postgres dispatch for {value!r}"),
    )

    MediaDatabase._initialize_schema(db)

    assert calls == [db]


@pytest.mark.unit
def test_initialize_schema_sqlite_uses_package_helper() -> None:
    from tldw_Server_API.app.core.DB_Management.media_db.schema.backends import (
        sqlite_helpers as sqlite_helpers_module,
    )

    assert MediaDatabase.__dict__["_initialize_schema_sqlite"].__globals__["__name__"] == (
        sqlite_helpers_module.__name__
    )


@pytest.mark.unit
def test_initialize_schema_postgres_uses_package_helper() -> None:
    from tldw_Server_API.app.core.DB_Management.media_db.schema.backends import (
        postgres_helpers as postgres_helpers_module,
    )

    assert MediaDatabase.__dict__["_initialize_schema_postgres"].__globals__["__name__"] == (
        postgres_helpers_module.__name__
    )


@pytest.mark.unit
def test_run_postgres_migrations_uses_package_helper() -> None:
    from tldw_Server_API.app.core.DB_Management.media_db.schema import migrations as migrations_module

    assert MediaDatabase.__dict__["_run_postgres_migrations"].__globals__["__name__"] == (
        migrations_module.__name__
    )


@pytest.mark.unit
def test_get_postgres_migrations_uses_package_helper() -> None:
    from tldw_Server_API.app.core.DB_Management.media_db.media_database import MediaDatabase
    from tldw_Server_API.app.core.DB_Management.media_db.schema import migrations as migrations_module

    assert MediaDatabase.__dict__["_get_postgres_migrations"].__globals__["__name__"] == (
        migrations_module.__name__
    )


@pytest.mark.unit
def test_initialize_sqlite_schema_bridge_routes_through_package_coordinator(monkeypatch) -> None:
    from tldw_Server_API.app.core.DB_Management.media_db.schema.backends import (
        sqlite_helpers as sqlite_helpers_module,
    )

    legacy_calls: list[str] = []
    coordinator_calls: list[object] = []

    db = SimpleNamespace(_initialize_schema_sqlite=lambda: legacy_calls.append("legacy"))

    monkeypatch.setattr(
        sqlite_helpers_module,
        "bootstrap_sqlite_schema",
        lambda value: coordinator_calls.append(value),
        raising=False,
    )

    sqlite_backend_module.initialize_sqlite_schema(db)

    assert coordinator_calls == [db]
    assert legacy_calls == []


@pytest.mark.unit
def test_initialize_postgres_schema_bridge_routes_through_package_coordinator(monkeypatch) -> None:
    from tldw_Server_API.app.core.DB_Management.media_db.schema.backends import (
        postgres_helpers as postgres_helpers_module,
    )

    legacy_calls: list[str] = []
    coordinator_calls: list[object] = []

    db = SimpleNamespace(_initialize_schema_postgres=lambda: legacy_calls.append("legacy"))

    monkeypatch.setattr(
        postgres_helpers_module,
        "bootstrap_postgres_schema",
        lambda value: coordinator_calls.append(value),
        raising=False,
    )

    postgres_backend_module.initialize_postgres_schema(db)

    assert coordinator_calls == [db]
    assert legacy_calls == []


@pytest.mark.unit
def test_ensure_postgres_post_core_structures_runs_followup_ensures(monkeypatch) -> None:
    from tldw_Server_API.app.core.DB_Management.media_db.schema.backends import (
        postgres_helpers as postgres_helpers_module,
    )

    conn = object()
    calls: list[object] = []
    db = SimpleNamespace(
        _ensure_postgres_collections_tables=lambda value: calls.append(("collections", value)),
        _ensure_postgres_tts_history=lambda value: calls.append(("tts_history", value)),
        _ensure_postgres_data_tables=lambda value: calls.append(("data_tables", value)),
        _ensure_postgres_source_hash_column=lambda value: calls.append(("source_hash", value)),
        _ensure_postgres_claims_extensions=lambda value: calls.append(("claims_extensions", value)),
        _ensure_postgres_email_schema=lambda value: calls.append(("email_schema", value)),
        _sync_postgres_sequences=lambda value: calls.append(("sequence_sync", value)),
    )

    monkeypatch.setattr(
        postgres_helpers_module,
        "ensure_postgres_policies",
        lambda value, connection: calls.append(("policies", value, connection)),
    )

    postgres_helpers_module.ensure_postgres_post_core_structures(db, conn)

    assert calls == [
        ("collections", conn),
        ("tts_history", conn),
        ("data_tables", conn),
        ("source_hash", conn),
        ("claims_extensions", conn),
        ("email_schema", conn),
        ("sequence_sync", conn),
        ("policies", db, conn),
    ]


@pytest.mark.unit
def test_run_postgres_migrate_to_v16_invokes_source_hash_ensure() -> None:
    from tldw_Server_API.app.core.DB_Management.media_db.schema.migration_bodies import (
        postgres_source_hash as postgres_source_hash_module,
    )

    conn = object()
    calls: list[object] = []
    db = SimpleNamespace(
        _ensure_postgres_source_hash_column=lambda value: calls.append(value),
    )

    postgres_source_hash_module.run_postgres_migrate_to_v16(db, conn)

    assert calls == [conn]


@pytest.mark.unit
def test_run_postgres_migrate_to_v10_invokes_claims_helpers_in_order() -> None:
    from tldw_Server_API.app.core.DB_Management.media_db.schema.migration_bodies import (
        postgres_claims as postgres_claims_module,
    )

    conn = object()
    calls: list[tuple[str, object]] = []
    db = SimpleNamespace(
        _ensure_postgres_claims_tables=lambda value: calls.append(("claims_tables", value)),
        _ensure_postgres_claims_extensions=lambda value: calls.append(("claims_extensions", value)),
    )

    postgres_claims_module.run_postgres_migrate_to_v10(db, conn)

    assert calls == [("claims_tables", conn), ("claims_extensions", conn)]


@pytest.mark.unit
def test_run_postgres_migrate_to_v11_executes_converted_mediafiles_statements_in_order() -> None:
    from tldw_Server_API.app.core.DB_Management.media_db.schema.migration_bodies import (
        postgres_mediafiles as postgres_mediafiles_module,
    )

    conn = object()
    calls: list[tuple[str, tuple[object, ...] | None, object]] = []

    class FakeBackend:
        def execute(
            self,
            query: str,
            params: tuple[object, ...] | None = None,
            *,
            connection: object,
        ) -> None:
            calls.append((query, params, connection))

    db = SimpleNamespace(
        _MEDIA_FILES_TABLE_SQL="mediafiles sql",
        backend=FakeBackend(),
        _convert_sqlite_sql_to_postgres_statements=lambda sql: [
            "CREATE TABLE mediafiles (...)",
            "CREATE INDEX idx_media_files_media_id ON mediafiles(media_id)",
        ],
    )

    postgres_mediafiles_module.run_postgres_migrate_to_v11(db, conn)

    assert calls == [
        ("CREATE TABLE mediafiles (...)", None, conn),
        ("CREATE INDEX idx_media_files_media_id ON mediafiles(media_id)", None, conn),
    ]


@pytest.mark.unit
def test_get_db_version_returns_row_version_value() -> None:
    from tldw_Server_API.app.core.DB_Management.media_db.schema import (
        sqlite_schema_version as sqlite_schema_version_module,
    )

    class FakeCursor:
        def fetchone(self):
            return {"version": 11}

    class FakeConn:
        def execute(self, query: str):
            assert query == "SELECT version FROM schema_version LIMIT 1"
            return FakeCursor()

    assert sqlite_schema_version_module.get_db_version(SimpleNamespace(), FakeConn()) == 11


@pytest.mark.unit
def test_get_db_version_returns_zero_when_row_missing() -> None:
    from tldw_Server_API.app.core.DB_Management.media_db.schema import (
        sqlite_schema_version as sqlite_schema_version_module,
    )

    class FakeCursor:
        def fetchone(self):
            return None

    class FakeConn:
        def execute(self, _query: str):
            return FakeCursor()

    assert sqlite_schema_version_module.get_db_version(SimpleNamespace(), FakeConn()) == 0


@pytest.mark.unit
def test_get_db_version_returns_zero_when_schema_version_table_missing() -> None:
    from tldw_Server_API.app.core.DB_Management.media_db.schema import (
        sqlite_schema_version as sqlite_schema_version_module,
    )

    class FakeConn:
        def execute(self, _query: str):
            raise sqlite3.OperationalError("no such table: schema_version")

    assert sqlite_schema_version_module.get_db_version(SimpleNamespace(), FakeConn()) == 0


@pytest.mark.unit
def test_get_db_version_wraps_other_sqlite_errors() -> None:
    from tldw_Server_API.app.core.DB_Management.media_db.errors import DatabaseError
    from tldw_Server_API.app.core.DB_Management.media_db.schema import (
        sqlite_schema_version as sqlite_schema_version_module,
    )

    class FakeConn:
        def execute(self, _query: str):
            raise sqlite3.OperationalError("database disk image is malformed")

    with pytest.raises(DatabaseError, match="Could not determine database schema version"):
        sqlite_schema_version_module.get_db_version(SimpleNamespace(), FakeConn())


@pytest.mark.unit
def test_runtime_chunk_fts_ops_ensure_chunk_fts_creates_virtual_table_and_rebuilds_only_when_new() -> None:
    from tldw_Server_API.app.core.DB_Management.media_db.runtime import (
        chunk_fts_ops as chunk_fts_ops_module,
    )

    class FakeCursor:
        def __init__(self, row):
            self._row = row

        def fetchone(self):
            return self._row

    calls: list[tuple[str, bool]] = []

    def execute_query(sql: str, commit: bool = False):
        calls.append((sql, commit))
        if "sqlite_master" in sql:
            return FakeCursor(None)
        return FakeCursor(None)

    db = SimpleNamespace(
        backend_type=BackendType.SQLITE,
        execute_query=execute_query,
    )

    chunk_fts_ops_module.ensure_chunk_fts(db)

    assert calls == [
        (
            "SELECT 1 AS exists_flag FROM sqlite_master "
            "WHERE type = 'table' AND name = 'unvectorized_chunks_fts'",
            False,
        ),
        (
            "CREATE VIRTUAL TABLE IF NOT EXISTS unvectorized_chunks_fts "
            "USING fts5(\n"
            "  chunk_text,\n"
            "  content='UnvectorizedMediaChunks',\n"
            "  content_rowid='id'\n"
            ")",
            True,
        ),
        (
            "INSERT INTO unvectorized_chunks_fts(unvectorized_chunks_fts) VALUES('rebuild')",
            True,
        ),
    ]


@pytest.mark.unit
def test_runtime_chunk_fts_ops_maybe_rebuild_chunk_fts_if_empty_creates_missing_table_then_rebuilds() -> None:
    from tldw_Server_API.app.core.DB_Management.media_db.runtime import (
        chunk_fts_ops as chunk_fts_ops_module,
    )

    class FakeCursor:
        def __init__(self, row):
            self._row = row

        def fetchone(self):
            return self._row

    calls: list[tuple[str, bool]] = []
    missing_once = {"value": True}
    ensure_calls: list[object] = []

    def execute_query(sql: str, commit: bool = False):
        calls.append((sql, commit))
        if "SELECT count(*) AS c FROM unvectorized_chunks_fts" in sql and missing_once["value"]:
            missing_once["value"] = False
            raise sqlite3.OperationalError("missing table")
        if "SELECT count(*) AS c FROM unvectorized_chunks_fts" in sql:
            return FakeCursor((0,))
        return FakeCursor(None)

    db = SimpleNamespace(
        backend_type=BackendType.SQLITE,
        execute_query=execute_query,
        ensure_chunk_fts=lambda: ensure_calls.append("ensure"),
    )

    chunk_fts_ops_module.maybe_rebuild_chunk_fts_if_empty(db)

    assert ensure_calls == ["ensure"]
    assert calls == [
        ("SELECT count(*) AS c FROM unvectorized_chunks_fts", False),
        ("SELECT count(*) AS c FROM unvectorized_chunks_fts", False),
        ("INSERT INTO unvectorized_chunks_fts(unvectorized_chunks_fts) VALUES('rebuild')", True),
    ]


@pytest.mark.unit
def test_runtime_chunk_fts_ops_maybe_rebuild_chunk_fts_if_empty_skips_rebuild_when_not_empty() -> None:
    from tldw_Server_API.app.core.DB_Management.media_db.runtime import (
        chunk_fts_ops as chunk_fts_ops_module,
    )

    class FakeCursor:
        def __init__(self, row):
            self._row = row

        def fetchone(self):
            return self._row

    calls: list[tuple[str, bool]] = []

    def execute_query(sql: str, commit: bool = False):
        calls.append((sql, commit))
        return FakeCursor((3,))

    db = SimpleNamespace(
        backend_type=BackendType.SQLITE,
        execute_query=execute_query,
        ensure_chunk_fts=lambda: pytest.fail("ensure_chunk_fts should not run when table exists"),
    )

    chunk_fts_ops_module.maybe_rebuild_chunk_fts_if_empty(db)

    assert calls == [("SELECT count(*) AS c FROM unvectorized_chunks_fts", False)]


@pytest.mark.unit
def test_convert_sqlite_sql_to_postgres_statements_filters_sqlite_only_lines_and_collects_statements(
    monkeypatch,
) -> None:
    from tldw_Server_API.app.core.DB_Management.media_db.schema import (
        postgres_sqlite_conversion as postgres_sqlite_conversion_module,
    )

    seen_statements: list[str] = []
    db = SimpleNamespace()
    sql = """
    -- comment
    PRAGMA foreign_keys = ON;
    CREATE TABLE demo (
        id INTEGER PRIMARY KEY AUTOINCREMENT
    );
    CREATE VIRTUAL TABLE demo_fts USING fts5(content);
    CREATE TRIGGER demo_ai AFTER INSERT ON demo BEGIN SELECT 1; END;
    INSERT OR IGNORE INTO demo(name) VALUES ('a');
    """

    def fake_transform(_db, statement: str):
        seen_statements.append(statement)
        return f"converted::{len(seen_statements)}"

    monkeypatch.setattr(
        postgres_sqlite_conversion_module,
        "_transform_sqlite_statement_to_postgres",
        fake_transform,
    )

    result = postgres_sqlite_conversion_module._convert_sqlite_sql_to_postgres_statements(db, sql)

    assert result == ["converted::1", "converted::2"]
    assert seen_statements == [
        "    CREATE TABLE demo (\n        id INTEGER PRIMARY KEY AUTOINCREMENT\n    );",
        "    INSERT OR IGNORE INTO demo(name) VALUES ('a');",
    ]


@pytest.mark.unit
def test_transform_sqlite_statement_to_postgres_rewrites_insert_ignore_and_collation() -> None:
    from tldw_Server_API.app.core.DB_Management.media_db.schema import (
        postgres_sqlite_conversion as postgres_sqlite_conversion_module,
    )

    db = SimpleNamespace()

    transformed = postgres_sqlite_conversion_module._transform_sqlite_statement_to_postgres(
        db,
        "INSERT OR IGNORE INTO demo(name) VALUES ('a') COLLATE NOCASE",
    )

    assert transformed is not None
    assert "INSERT OR IGNORE" not in transformed.upper()
    assert "ON CONFLICT DO NOTHING" in transformed.upper()
    assert "COLLATE NOCASE" not in transformed.upper()
    assert transformed.endswith(";")


@pytest.mark.unit
def test_run_postgres_migrate_to_v11_swallows_per_statement_backend_errors_and_continues() -> None:
    from tldw_Server_API.app.core.DB_Management.media_db.schema.migration_bodies import (
        postgres_mediafiles as postgres_mediafiles_module,
    )

    conn = object()
    calls: list[str] = []

    class FakeBackend:
        def execute(
            self,
            query: str,
            params: tuple[object, ...] | None = None,
            *,
            connection: object,
        ) -> None:
            calls.append(query)
            if query == "bad stmt":
                raise BackendDatabaseError("boom")

    db = SimpleNamespace(
        _MEDIA_FILES_TABLE_SQL="mediafiles sql",
        backend=FakeBackend(),
        _convert_sqlite_sql_to_postgres_statements=lambda sql: [
            "good stmt",
            "bad stmt",
            "later stmt",
        ],
    )

    postgres_mediafiles_module.run_postgres_migrate_to_v11(db, conn)

    assert calls == ["good stmt", "bad stmt", "later stmt"]


@pytest.mark.unit
def test_run_postgres_migrate_to_v11_swallows_outer_noncritical_conversion_failures() -> None:
    from tldw_Server_API.app.core.DB_Management.media_db.schema.migration_bodies import (
        postgres_mediafiles as postgres_mediafiles_module,
    )

    conn = object()

    class FakeBackend:
        def execute(
            self,
            query: str,
            params: tuple[object, ...] | None = None,
            *,
            connection: object,
        ) -> None:
            raise AssertionError("backend should not be used when conversion fails")

    db = SimpleNamespace(
        _MEDIA_FILES_TABLE_SQL="mediafiles sql",
        backend=FakeBackend(),
        _convert_sqlite_sql_to_postgres_statements=lambda sql: (_ for _ in ()).throw(
            TypeError("conversion failed")
        ),
    )

    postgres_mediafiles_module.run_postgres_migrate_to_v11(db, conn)


@pytest.mark.unit
def test_update_schema_version_postgres_executes_expected_sql_and_params() -> None:
    from tldw_Server_API.app.core.DB_Management.media_db.schema import (
        postgres_schema_version as postgres_schema_version_module,
    )

    conn = object()
    calls: list[tuple[str, tuple[object, ...], object]] = []

    class FakeBackend:
        def execute(
            self,
            query: str,
            params: tuple[object, ...],
            *,
            connection: object,
        ) -> None:
            calls.append((query, params, connection))

    db = SimpleNamespace(backend=FakeBackend())

    postgres_schema_version_module.update_schema_version_postgres(db, conn, 11)

    assert calls == [("UPDATE schema_version SET version = %s", (11,), conn)]


@pytest.mark.unit
def test_sync_postgres_sequences_skips_incomplete_rows() -> None:
    from tldw_Server_API.app.core.DB_Management.media_db.schema import (
        postgres_sequence_maintenance as postgres_sequence_maintenance_module,
    )

    conn = object()
    execute_calls: list[tuple[str, tuple[object, ...] | None, object]] = []

    class FakeBackend:
        @staticmethod
        def escape_identifier(value: str) -> str:
            return f'"{value}"'

        def execute(
            self,
            query: str,
            params: tuple[object, ...] | None = None,
            *,
            connection: object,
        ) -> SimpleNamespace:
            execute_calls.append((query.strip(), params, connection))
            return SimpleNamespace(
                rows=[
                    {
                        "sequence_schema": "public",
                        "sequence_name": "media_id_seq",
                        "table_name": None,
                        "column_name": "id",
                    },
                    {
                        "sequence_schema": "public",
                        "sequence_name": None,
                        "table_name": "media",
                        "column_name": "id",
                    },
                ]
            )

    db = SimpleNamespace(backend=FakeBackend())

    postgres_sequence_maintenance_module.sync_postgres_sequences(db, conn)

    assert len(execute_calls) == 1


@pytest.mark.unit
def test_sync_postgres_sequences_invalid_scalar_uses_safe_setval_branch() -> None:
    from tldw_Server_API.app.core.DB_Management.media_db.schema import (
        postgres_sequence_maintenance as postgres_sequence_maintenance_module,
    )

    conn = object()
    execute_calls: list[tuple[str, tuple[object, ...] | None, object]] = []

    class FakeBackend:
        @staticmethod
        def escape_identifier(value: str) -> str:
            return f'"{value}"'

        def execute(
            self,
            query: str,
            params: tuple[object, ...] | None = None,
            *,
            connection: object,
        ) -> SimpleNamespace:
            execute_calls.append((query.strip(), params, connection))
            if len(execute_calls) == 1:
                return SimpleNamespace(
                    rows=[
                        {
                            "sequence_schema": "public",
                            "sequence_name": "media_id_seq",
                            "table_name": "media",
                            "column_name": "id",
                        }
                    ]
                )
            if len(execute_calls) == 2:
                return SimpleNamespace(scalar="not-an-int")
            return SimpleNamespace()

    db = SimpleNamespace(backend=FakeBackend())

    postgres_sequence_maintenance_module.sync_postgres_sequences(db, conn)

    assert execute_calls == [
        (
            (
                "SELECT\n"
                "            sequence_ns.nspname AS sequence_schema,\n"
                "            seq.relname AS sequence_name,\n"
                "            tab.relname AS table_name,\n"
                "            col.attname AS column_name\n"
                "        FROM pg_class seq\n"
                "        JOIN pg_namespace sequence_ns ON sequence_ns.oid = seq.relnamespace\n"
                "        JOIN pg_depend dep ON dep.objid = seq.oid AND dep.deptype = 'a'\n"
                "        JOIN pg_class tab ON tab.oid = dep.refobjid\n"
                "        JOIN pg_namespace tab_ns ON tab_ns.oid = tab.relnamespace\n"
                "        JOIN pg_attribute col ON col.attrelid = tab.oid AND col.attnum = dep.refobjsubid\n"
                "        WHERE seq.relkind = 'S' AND tab_ns.nspname = 'public';"
            ),
            None,
            conn,
        ),
        ('SELECT COALESCE(MAX("id"), 0) AS max_id FROM "media"', None, conn),
        ("SELECT setval(%s, %s, false)", ("public.media_id_seq", 1), conn),
    ]


@pytest.mark.unit
def test_sync_postgres_sequences_positive_scalar_uses_max_id_branch() -> None:
    from tldw_Server_API.app.core.DB_Management.media_db.schema import (
        postgres_sequence_maintenance as postgres_sequence_maintenance_module,
    )

    conn = object()
    execute_calls: list[tuple[str, tuple[object, ...] | None, object]] = []

    class FakeBackend:
        @staticmethod
        def escape_identifier(value: str) -> str:
            return f'"{value}"'

        def execute(
            self,
            query: str,
            params: tuple[object, ...] | None = None,
            *,
            connection: object,
        ) -> SimpleNamespace:
            execute_calls.append((query.strip(), params, connection))
            if len(execute_calls) == 1:
                return SimpleNamespace(
                    rows=[
                        {
                            "sequence_schema": "custom_schema",
                            "sequence_name": "media_id_seq",
                            "table_name": "media",
                            "column_name": "id",
                        }
                    ]
                )
            if len(execute_calls) == 2:
                return SimpleNamespace(scalar=7)
            return SimpleNamespace()

    db = SimpleNamespace(backend=FakeBackend())

    postgres_sequence_maintenance_module.sync_postgres_sequences(db, conn)

    assert execute_calls[-1] == (
        "SELECT setval(%s, %s)",
        ("custom_schema.media_id_seq", 7),
        conn,
    )


@pytest.mark.unit
def test_run_postgres_migrate_to_v17_invokes_claims_helpers_in_order() -> None:
    from tldw_Server_API.app.core.DB_Management.media_db.schema.migration_bodies import (
        postgres_claims as postgres_claims_module,
    )

    conn = object()
    calls: list[tuple[str, object]] = []
    db = SimpleNamespace(
        _ensure_postgres_claims_tables=lambda value: calls.append(("claims_tables", value)),
        _ensure_postgres_claims_extensions=lambda value: calls.append(("claims_extensions", value)),
    )

    postgres_claims_module.run_postgres_migrate_to_v17(db, conn)

    assert calls == [("claims_tables", conn), ("claims_extensions", conn)]


@pytest.mark.unit
def test_run_postgres_migrate_to_v5_adds_safe_metadata_column() -> None:
    from tldw_Server_API.app.core.DB_Management.media_db.schema.migration_bodies import (
        postgres_early_schema as postgres_early_schema_module,
    )

    conn = object()
    calls: list[tuple[str, object]] = []

    class FakeBackend:
        def escape_identifier(self, name: str) -> str:
            return f'"{name}"'

        def execute(self, query: str, *, connection: object) -> None:
            calls.append((query, connection))

    db = SimpleNamespace(backend=FakeBackend())

    postgres_early_schema_module.run_postgres_migrate_to_v5(db, conn)

    assert calls == [
        (
            'ALTER TABLE "documentversions" ADD COLUMN IF NOT EXISTS "safe_metadata" TEXT',
            conn,
        )
    ]


@pytest.mark.unit
def test_run_postgres_migrate_to_v6_creates_identifier_table_and_indexes() -> None:
    from tldw_Server_API.app.core.DB_Management.media_db.schema.migration_bodies import (
        postgres_early_schema as postgres_early_schema_module,
    )

    conn = object()
    calls: list[tuple[str, object]] = []

    class FakeBackend:
        def escape_identifier(self, name: str) -> str:
            return f'"{name}"'

        def execute(self, query: str, *, connection: object) -> None:
            calls.append((query, connection))

    db = SimpleNamespace(backend=FakeBackend())

    postgres_early_schema_module.run_postgres_migrate_to_v6(db, conn)

    assert calls[0] == (
        'CREATE TABLE IF NOT EXISTS "documentversionidentifiers" ("dv_id" BIGINT PRIMARY KEY REFERENCES "documentversions"("id") ON DELETE CASCADE,"doi" TEXT,"pmid" TEXT,"pmcid" TEXT,"arxiv_id" TEXT,"s2_paper_id" TEXT)',
        conn,
    )
    assert calls[1:] == [
        ('CREATE INDEX IF NOT EXISTS "idx_dvi_doi" ON "documentversionidentifiers" ("doi")', conn),
        ('CREATE INDEX IF NOT EXISTS "idx_dvi_pmid" ON "documentversionidentifiers" ("pmid")', conn),
        ('CREATE INDEX IF NOT EXISTS "idx_dvi_pmcid" ON "documentversionidentifiers" ("pmcid")', conn),
        ('CREATE INDEX IF NOT EXISTS "idx_dvi_arxiv" ON "documentversionidentifiers" ("arxiv_id")', conn),
        ('CREATE INDEX IF NOT EXISTS "idx_dvi_s2" ON "documentversionidentifiers" ("s2_paper_id")', conn),
    ]


@pytest.mark.unit
def test_run_postgres_migrate_to_v7_creates_structure_index_table_and_indexes() -> None:
    from tldw_Server_API.app.core.DB_Management.media_db.schema.migration_bodies import (
        postgres_early_schema as postgres_early_schema_module,
    )

    conn = object()
    calls: list[tuple[str, object]] = []

    class FakeBackend:
        def escape_identifier(self, name: str) -> str:
            return f'"{name}"'

        def execute(self, query: str, *, connection: object) -> None:
            calls.append((query, connection))

    db = SimpleNamespace(backend=FakeBackend())

    postgres_early_schema_module.run_postgres_migrate_to_v7(db, conn)

    assert calls[0] == (
        'CREATE TABLE IF NOT EXISTS "documentstructureindex" ("id" BIGSERIAL PRIMARY KEY,"media_id" BIGINT NOT NULL REFERENCES "media"("id") ON DELETE CASCADE,"parent_id" BIGINT NULL,"kind" TEXT NOT NULL,"level" INTEGER,"title" TEXT,"start_char" BIGINT NOT NULL,"end_char" BIGINT NOT NULL,"order_index" INTEGER,"path" TEXT,"created_at" TIMESTAMP DEFAULT CURRENT_TIMESTAMP,"last_modified" TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,"version" INTEGER NOT NULL DEFAULT 1,"client_id" TEXT NOT NULL,"deleted" BOOLEAN NOT NULL DEFAULT FALSE)',
        conn,
    )
    assert calls[1:] == [
        ('CREATE INDEX IF NOT EXISTS "idx_dsi_media_kind" ON "documentstructureindex" (media_id, kind)', conn),
        ('CREATE INDEX IF NOT EXISTS "idx_dsi_media_start" ON "documentstructureindex" (media_id, start_char)', conn),
        ('CREATE INDEX IF NOT EXISTS "idx_dsi_media_parent" ON "documentstructureindex" (parent_id)', conn),
    ]


@pytest.mark.unit
def test_run_postgres_migrate_to_v8_adds_scope_columns_to_media_and_sync_log() -> None:
    from tldw_Server_API.app.core.DB_Management.media_db.schema.migration_bodies import (
        postgres_early_schema as postgres_early_schema_module,
    )

    conn = object()
    calls: list[tuple[str, object]] = []

    class FakeBackend:
        def escape_identifier(self, name: str) -> str:
            return f'"{name}"'

        def execute(self, query: str, *, connection: object) -> None:
            calls.append((query, connection))

    db = SimpleNamespace(backend=FakeBackend())

    postgres_early_schema_module.run_postgres_migrate_to_v8(db, conn)

    assert calls == [
        ('ALTER TABLE "media" ADD COLUMN IF NOT EXISTS "org_id" BIGINT', conn),
        ('ALTER TABLE "media" ADD COLUMN IF NOT EXISTS "team_id" BIGINT', conn),
        ('ALTER TABLE "sync_log" ADD COLUMN IF NOT EXISTS "org_id" BIGINT', conn),
        ('ALTER TABLE "sync_log" ADD COLUMN IF NOT EXISTS "team_id" BIGINT', conn),
    ]


@pytest.mark.unit
def test_run_postgres_migrate_to_v14_invokes_data_tables_ensure() -> None:
    from tldw_Server_API.app.core.DB_Management.media_db.schema.migration_bodies import (
        postgres_data_tables as postgres_data_tables_module,
    )

    conn = object()
    calls: list[object] = []
    db = SimpleNamespace(
        _ensure_postgres_data_tables=lambda value: calls.append(value),
    )

    postgres_data_tables_module.run_postgres_migrate_to_v14(db, conn)

    assert calls == [conn]


@pytest.mark.unit
def test_run_postgres_migrate_to_v15_invokes_data_tables_ensure() -> None:
    from tldw_Server_API.app.core.DB_Management.media_db.schema.migration_bodies import (
        postgres_data_tables as postgres_data_tables_module,
    )

    conn = object()
    calls: list[object] = []
    db = SimpleNamespace(
        _ensure_postgres_data_tables=lambda value: calls.append(value),
    )

    postgres_data_tables_module.run_postgres_migrate_to_v15(db, conn)

    assert calls == [conn]


@pytest.mark.unit
def test_postgres_data_tables_structures_ensure_postgres_data_tables_runs_create_then_columns_then_other() -> None:
    from tldw_Server_API.app.core.DB_Management.media_db.schema import (
        postgres_data_table_structures as postgres_data_table_structures_module,
    )

    conn = object()
    calls: list[tuple[str, object]] = []

    class FakeBackend:
        def execute(self, query: str, *, connection: object) -> None:
            calls.append((query, connection))

    db = SimpleNamespace(
        _DATA_TABLES_SQL="data tables sql",
        backend=FakeBackend(),
        _convert_sqlite_sql_to_postgres_statements=lambda sql: [
            "CREATE TABLE data_tables (...)",
            "CREATE TABLE data_table_columns (...)",
            "CREATE INDEX idx_data_tables_workspace_tag ON data_tables(workspace_tag)",
        ],
        _ensure_postgres_data_tables_columns=lambda value: calls.append(("ensure_columns", value)),
    )

    postgres_data_table_structures_module.ensure_postgres_data_tables(db, conn)

    assert calls == [
        ("CREATE TABLE data_tables (...)", conn),
        ("CREATE TABLE data_table_columns (...)", conn),
        ("ensure_columns", conn),
        ("CREATE INDEX idx_data_tables_workspace_tag ON data_tables(workspace_tag)", conn),
    ]


@pytest.mark.unit
def test_postgres_data_tables_structures_ensure_postgres_columns_adds_only_missing_columns() -> None:
    from tldw_Server_API.app.core.DB_Management.media_db.schema import (
        postgres_data_table_structures as postgres_data_table_structures_module,
    )

    conn = object()
    calls: list[tuple[str, object]] = []

    class FakeBackend:
        def escape_identifier(self, name: str) -> str:
            return f'"{name}"'

        def table_exists(self, table: str, *, connection: object) -> bool:
            assert table == "data_tables"
            assert connection is conn
            return True

        def get_table_info(self, table: str, *, connection: object) -> list[dict[str, str]]:
            assert table == "data_tables"
            assert connection is conn
            return [{"name": "workspace_tag"}]

        def execute(self, query: str, *, connection: object) -> None:
            calls.append((query, connection))

    db = SimpleNamespace(backend=FakeBackend())

    postgres_data_table_structures_module.ensure_postgres_columns(
        db,
        conn,
        table="data_tables",
        column_defs={"workspace_tag": "TEXT", "column_hints_json": "TEXT"},
    )

    assert calls == [
        ('ALTER TABLE "data_tables" ADD COLUMN IF NOT EXISTS "column_hints_json" TEXT', conn)
    ]


@pytest.mark.unit
def test_postgres_data_tables_structures_ensure_postgres_data_tables_columns_repairs_columns_and_index() -> None:
    from tldw_Server_API.app.core.DB_Management.media_db.schema import (
        postgres_data_table_structures as postgres_data_table_structures_module,
    )

    conn = object()
    calls: list[tuple[str, object, object | tuple[object, ...] | None]] = []
    ensure_calls: list[tuple[str, dict[str, str]]] = []

    class FakeBackend:
        def escape_identifier(self, name: str) -> str:
            return f'"{name}"'

        def table_exists(self, table: str, *, connection: object) -> bool:
            assert connection is conn
            return table == "data_tables"

        def execute(
            self,
            query: str,
            params: tuple[object, ...] | None = None,
            *,
            connection: object,
        ) -> None:
            calls.append((query, connection, params))

    db = SimpleNamespace(
        backend=FakeBackend(),
        client_id="tests-client",
        _ensure_postgres_columns=lambda value, *, table, column_defs: ensure_calls.append(
            (table, column_defs)
        ),
    )

    postgres_data_table_structures_module.ensure_postgres_data_tables_columns(db, conn)

    assert [table for table, _ in ensure_calls] == [
        "data_tables",
        "data_table_columns",
        "data_table_rows",
        "data_table_sources",
    ]
    assert calls == [
        (
            'UPDATE "data_tables" SET "client_id" = %s WHERE "client_id" IS NULL OR "client_id" = \'\'',
            conn,
            ("tests-client",),
        ),
        (
            'UPDATE "data_tables" SET "last_modified" = CURRENT_TIMESTAMP WHERE "last_modified" IS NULL',
            conn,
            None,
        ),
        (
            'CREATE INDEX IF NOT EXISTS "idx_data_tables_workspace_tag" ON "data_tables" ("workspace_tag")',
            conn,
            None,
        ),
    ]


@pytest.mark.unit
def test_postgres_tts_source_hash_structures_ensure_postgres_tts_history_emits_table_then_indexes() -> None:
    from tldw_Server_API.app.core.DB_Management.media_db.schema import (
        postgres_tts_source_hash_structures as postgres_tts_source_hash_structures_module,
    )

    conn = object()
    calls: list[tuple[str, object]] = []

    class FakeBackend:
        def execute(self, query: str, *, connection: object) -> None:
            calls.append((query, connection))

    db = SimpleNamespace(backend=FakeBackend())

    postgres_tts_source_hash_structures_module.ensure_postgres_tts_history(db, conn)

    assert calls == [
        (
            "CREATE TABLE IF NOT EXISTS tts_history ("
            "id BIGSERIAL PRIMARY KEY, "
            "user_id TEXT NOT NULL, "
            "created_at TIMESTAMPTZ NOT NULL, "
            "text TEXT, "
            "text_hash TEXT NOT NULL, "
            "text_length INTEGER, "
            "provider TEXT, "
            "model TEXT, "
            "voice_id TEXT, "
            "voice_name TEXT, "
            "voice_info TEXT, "
            "format TEXT, "
            "duration_ms INTEGER, "
            "generation_time_ms INTEGER, "
            "params_json TEXT, "
            "status TEXT, "
            "segments_json TEXT, "
            "favorite BOOLEAN NOT NULL DEFAULT FALSE, "
            "job_id BIGINT, "
            "output_id BIGINT, "
            "artifact_ids TEXT, "
            "artifact_deleted_at TIMESTAMPTZ, "
            "error_message TEXT, "
            "deleted BOOLEAN NOT NULL DEFAULT FALSE, "
            "deleted_at TIMESTAMPTZ"
            ")",
            conn,
        ),
        ("CREATE INDEX IF NOT EXISTS idx_tts_history_user_created ON tts_history(user_id, created_at DESC)", conn),
        ("CREATE INDEX IF NOT EXISTS idx_tts_history_user_favorite ON tts_history(user_id, favorite)", conn),
        ("CREATE INDEX IF NOT EXISTS idx_tts_history_user_provider ON tts_history(user_id, provider)", conn),
        ("CREATE INDEX IF NOT EXISTS idx_tts_history_user_model ON tts_history(user_id, model)", conn),
        ("CREATE INDEX IF NOT EXISTS idx_tts_history_user_voice_id ON tts_history(user_id, voice_id)", conn),
        ("CREATE INDEX IF NOT EXISTS idx_tts_history_user_text_hash ON tts_history(user_id, text_hash)", conn),
    ]


@pytest.mark.unit
def test_postgres_tts_source_hash_structures_ensure_postgres_source_hash_emits_column_then_index() -> None:
    from tldw_Server_API.app.core.DB_Management.media_db.schema import (
        postgres_tts_source_hash_structures as postgres_tts_source_hash_structures_module,
    )

    conn = object()
    calls: list[tuple[str, object]] = []

    class FakeBackend:
        def escape_identifier(self, name: str) -> str:
            return f'"{name}"'

        def execute(self, query: str, *, connection: object) -> None:
            calls.append((query, connection))

    db = SimpleNamespace(backend=FakeBackend())

    postgres_tts_source_hash_structures_module.ensure_postgres_source_hash_column(db, conn)

    assert calls == [
        ('ALTER TABLE "media" ADD COLUMN IF NOT EXISTS "source_hash" TEXT', conn),
        (
            'CREATE INDEX IF NOT EXISTS "idx_media_source_hash" ON "media" ("source_hash")',
            conn,
        ),
    ]


@pytest.mark.unit
def test_run_postgres_migrate_to_v18_invokes_sequence_sync() -> None:
    from tldw_Server_API.app.core.DB_Management.media_db.schema.migration_bodies import (
        postgres_sequence_sync as postgres_sequence_sync_module,
    )

    conn = object()
    calls: list[object] = []
    db = SimpleNamespace(
        _sync_postgres_sequences=lambda value: calls.append(value),
    )

    postgres_sequence_sync_module.run_postgres_migrate_to_v18(db, conn)

    assert calls == [conn]


@pytest.mark.unit
def test_run_postgres_migrate_to_v19_invokes_fts_then_rls() -> None:
    from tldw_Server_API.app.core.DB_Management.media_db.schema.migration_bodies import (
        postgres_fts_rls as postgres_fts_rls_module,
    )

    conn = object()
    calls: list[object] = []
    db = SimpleNamespace(
        _ensure_postgres_fts=lambda value: calls.append(("fts", value)),
        _ensure_postgres_rls=lambda value: calls.append(("rls", value)),
    )

    postgres_fts_rls_module.run_postgres_migrate_to_v19(db, conn)

    assert calls == [("fts", conn), ("rls", conn)]


@pytest.mark.unit
def test_run_postgres_migrate_to_v20_invokes_tts_history_ensure() -> None:
    from tldw_Server_API.app.core.DB_Management.media_db.schema.migration_bodies import (
        postgres_tts_history as postgres_tts_history_module,
    )

    conn = object()
    calls: list[object] = []
    db = SimpleNamespace(
        _ensure_postgres_tts_history=lambda value: calls.append(value),
    )

    postgres_tts_history_module.run_postgres_migrate_to_v20(db, conn)

    assert calls == [conn]


@pytest.mark.unit
def test_run_postgres_migrate_to_v9_emits_visibility_owner_sql_in_order() -> None:
    from tldw_Server_API.app.core.DB_Management.media_db.schema.migration_bodies import (
        postgres_visibility_owner as postgres_visibility_owner_module,
    )

    conn = object()
    calls: list[tuple[str, object]] = []

    class FakeBackend:
        def escape_identifier(self, name: str) -> str:
            return f'"{name}"'

        def execute(self, query: str, *, connection: object) -> None:
            calls.append((query, connection))

    db = SimpleNamespace(backend=FakeBackend())

    postgres_visibility_owner_module.run_postgres_migrate_to_v9(db, conn)

    assert calls[0] == (
        'ALTER TABLE "media" ADD COLUMN IF NOT EXISTS "visibility" TEXT DEFAULT \'personal\'',
        conn,
    )
    assert "conname = 'chk_media_visibility'" in calls[1][0]
    assert 'ALTER TABLE "media"' in calls[1][0]
    assert 'CHECK ("visibility" IN (\'personal\', \'team\', \'org\'))' in calls[1][0]
    assert calls[1][1] is conn
    assert calls[2] == (
        'ALTER TABLE "media" ADD COLUMN IF NOT EXISTS "owner_user_id" BIGINT',
        conn,
    )
    assert 'UPDATE "media"' in calls[3][0]
    assert 'SET "owner_user_id" = CAST("client_id" AS BIGINT)' in calls[3][0]
    assert 'WHERE "owner_user_id" IS NULL' in calls[3][0]
    assert '"client_id" ~ \'^[0-9]+$\'' in calls[3][0]
    assert calls[3][1] is conn
    assert calls[4:] == [
        ('CREATE INDEX IF NOT EXISTS idx_media_visibility ON "media"("visibility")', conn),
        ('CREATE INDEX IF NOT EXISTS idx_media_owner_user_id ON "media"("owner_user_id")', conn),
    ]


@pytest.mark.unit
def test_run_postgres_migrate_to_v21_creates_structure_and_visual_indexes() -> None:
    from tldw_Server_API.app.core.DB_Management.media_db.schema.migration_bodies import (
        postgres_structure_visual_indexes as postgres_structure_visual_module,
    )

    conn = object()
    calls: list[object] = []

    class FakeBackend:
        def escape_identifier(self, name: str) -> str:
            return f'"{name}"'

        def table_exists(self, table_name: str, *, connection: object) -> bool:
            calls.append(("table_exists", table_name, connection))
            return table_name in {"DocumentStructureIndex", "VisualDocuments"}

        def execute(self, query: str, *, connection: object) -> None:
            calls.append(("execute", query, connection))

    db = SimpleNamespace(backend=FakeBackend())

    postgres_structure_visual_module.run_postgres_migrate_to_v21(db, conn)

    assert calls[0:2] == [
        ("table_exists", "documentstructureindex", conn),
        ("table_exists", "DocumentStructureIndex", conn),
    ]
    assert calls[2] == (
        "execute",
        'CREATE INDEX IF NOT EXISTS "idx_dsi_media_path" ON "DocumentStructureIndex" ("media_id", "path")',
        conn,
    )
    assert calls[3:5] == [
        ("table_exists", "visualdocuments", conn),
        ("table_exists", "VisualDocuments", conn),
    ]
    assert calls[5] == (
        "execute",
        'CREATE INDEX IF NOT EXISTS "idx_visualdocs_caption" ON "VisualDocuments" ("caption")',
        conn,
    )
    assert calls[6] == (
        "execute",
        'CREATE INDEX IF NOT EXISTS "idx_visualdocs_tags" ON "VisualDocuments" ("tags")',
        conn,
    )


@pytest.mark.unit
def test_run_postgres_migrate_to_v22_invokes_email_schema_ensure() -> None:
    from tldw_Server_API.app.core.DB_Management.media_db.schema.migration_bodies import (
        postgres_email_schema as postgres_email_schema_module,
    )

    conn = object()
    calls: list[object] = []
    db = SimpleNamespace(
        _ensure_postgres_email_schema=lambda value: calls.append(value),
    )

    postgres_email_schema_module.run_postgres_migrate_to_v22(db, conn)

    assert calls == [conn]


@pytest.mark.unit
def test_schema_features_ensure_sqlite_fts_structures_routes_through_package_helper(monkeypatch) -> None:
    from tldw_Server_API.app.core.DB_Management.media_db.schema.features import (
        fts as fts_feature_module,
    )

    db = SimpleNamespace(backend_type=BackendType.SQLITE)
    conn = object()
    calls: list[tuple[object, object]] = []

    monkeypatch.setattr(
        fts_feature_module,
        "_ensure_fts_structures",
        lambda value, connection: calls.append((value, connection)),
    )

    fts_feature_module.ensure_sqlite_fts_structures(db, conn)

    assert calls == [(db, conn)]


@pytest.mark.unit
def test_schema_features_ensure_postgres_fts_routes_through_package_helper(monkeypatch) -> None:
    from tldw_Server_API.app.core.DB_Management.media_db.schema.features import (
        fts as fts_feature_module,
    )

    db = SimpleNamespace(backend=object())
    conn = object()
    calls: list[tuple[object, object]] = []

    monkeypatch.setattr(
        fts_feature_module,
        "_ensure_postgres_fts",
        lambda value, connection: calls.append((value, connection)),
    )

    fts_feature_module.ensure_postgres_fts(db, conn)

    assert calls == [(db, conn)]


@pytest.mark.unit
def test_ensure_sqlite_post_core_structures_runs_followup_ensures(monkeypatch) -> None:
    from tldw_Server_API.app.core.DB_Management.media_db.schema.backends import (
        sqlite_helpers as sqlite_helpers_module,
    )

    calls: list[object] = []

    class FakeConn:
        def executescript(self, script: str) -> None:
            calls.append(("collections_sql", script))

    conn = FakeConn()
    db = SimpleNamespace(
        _ensure_sqlite_data_tables=lambda value: calls.append(("data_tables", value)),
        _ensure_sqlite_visibility_columns=lambda value: calls.append(("visibility", value)),
        _ensure_sqlite_source_hash_column=lambda value: calls.append(("source_hash", value)),
        _ensure_sqlite_claims_extensions=lambda value: calls.append(("claims_extensions", value)),
        _ensure_sqlite_email_schema=lambda value: calls.append(("email_schema", value)),
    )

    monkeypatch.setattr(
        sqlite_helpers_module,
        "ensure_sqlite_fts_structures",
        lambda value, connection: calls.append(("fts", value, connection)),
    )

    sqlite_helpers_module.ensure_sqlite_post_core_structures(db, conn)

    assert [entry[0] for entry in calls] == [
        "data_tables",
        "fts",
        "collections_sql",
        "visibility",
        "source_hash",
        "claims_extensions",
        "email_schema",
    ]
    assert calls[0] == ("data_tables", conn)
    assert calls[1] == ("fts", db, conn)
    assert "CREATE TABLE IF NOT EXISTS output_templates" in calls[2][1]
    assert "CREATE TABLE IF NOT EXISTS content_items" in calls[2][1]
    assert "CREATE VIRTUAL TABLE IF NOT EXISTS content_items_fts" in calls[2][1]
    assert calls[3:] == [
        ("visibility", conn),
        ("source_hash", conn),
        ("claims_extensions", conn),
        ("email_schema", conn),
    ]


@pytest.mark.unit
def test_fts_structures_dispatches_to_sqlite_helper() -> None:
    from tldw_Server_API.app.core.DB_Management.media_db.schema import (
        fts_structures as fts_structures_module,
    )

    conn = object()
    calls: list[tuple[str, object]] = []
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        fts_structures_module,
        "ensure_sqlite_fts",
        lambda value, connection: calls.append(("sqlite", connection)),
    )
    try:
        db = SimpleNamespace(backend_type=BackendType.SQLITE)
        fts_structures_module.ensure_fts_structures(db, conn)
    finally:
        monkeypatch.undo()

    assert calls == [("sqlite", conn)]


@pytest.mark.unit
def test_fts_structures_dispatches_to_postgres_helper() -> None:
    from tldw_Server_API.app.core.DB_Management.media_db.schema import (
        fts_structures as fts_structures_module,
    )

    conn = object()
    calls: list[tuple[str, object]] = []
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        fts_structures_module,
        "ensure_postgres_fts",
        lambda value, connection: calls.append(("postgres", connection)),
    )
    try:
        db = SimpleNamespace(backend_type=BackendType.POSTGRESQL)
        fts_structures_module.ensure_fts_structures(db, conn)
    finally:
        monkeypatch.undo()

    assert calls == [("postgres", conn)]


@pytest.mark.unit
def test_fts_structures_raises_for_unknown_backend() -> None:
    from tldw_Server_API.app.core.DB_Management.media_db.schema import (
        fts_structures as fts_structures_module,
    )

    db = SimpleNamespace(backend_type="mystery")

    with pytest.raises(NotImplementedError):
        fts_structures_module.ensure_fts_structures(db, object())


@pytest.mark.unit
def test_fts_structures_ensure_sqlite_fts_runs_scripts_verifies_tables_and_commits() -> None:
    from tldw_Server_API.app.core.DB_Management.media_db.schema import (
        fts_structures as fts_structures_module,
    )

    class FakeCursor:
        def fetchall(self):
            return [("media_fts",), ("keyword_fts",)]

    class FakeConn:
        def __init__(self) -> None:
            self.scripts: list[str] = []
            self.queries: list[str] = []
            self.commit_calls = 0

        def executescript(self, script: str) -> None:
            self.scripts.append(script)

        def execute(self, query: str):
            self.queries.append(query)
            return FakeCursor()

        def commit(self) -> None:
            self.commit_calls += 1

    conn = FakeConn()
    db = SimpleNamespace(
        _FTS_TABLES_SQL="fts tables",
        _CLAIMS_FTS_TRIGGERS_SQL="claims triggers",
    )

    fts_structures_module.ensure_sqlite_fts(db, conn)

    assert conn.scripts == ["fts tables", "claims triggers"]
    assert "SELECT name FROM sqlite_master" in conn.queries[0]
    assert conn.commit_calls == 1


@pytest.mark.unit
def test_fts_structures_ensure_sqlite_fts_raises_when_required_tables_missing() -> None:
    from tldw_Server_API.app.core.DB_Management.media_db.errors import DatabaseError
    from tldw_Server_API.app.core.DB_Management.media_db.schema import (
        fts_structures as fts_structures_module,
    )

    class FakeCursor:
        def fetchall(self):
            return [("media_fts",)]

    class FakeConn:
        def __init__(self) -> None:
            self.commit_calls = 0

        def executescript(self, _script: str) -> None:
            return None

        def execute(self, _query: str):
            return FakeCursor()

        def commit(self) -> None:
            self.commit_calls += 1

    conn = FakeConn()
    db = SimpleNamespace(
        _FTS_TABLES_SQL="fts tables",
        _CLAIMS_FTS_TRIGGERS_SQL="claims triggers",
    )

    with pytest.raises(DatabaseError):
        fts_structures_module.ensure_sqlite_fts(db, conn)

    assert conn.commit_calls == 1


@pytest.mark.unit
def test_fts_structures_ensure_postgres_fts_creates_core_tables_and_tolerates_chunk_failure() -> None:
    from tldw_Server_API.app.core.DB_Management.media_db.schema import (
        fts_structures as fts_structures_module,
    )

    calls: list[tuple[str, str, tuple[str, ...], object]] = []

    class FakeBackend:
        def create_fts_table(
            self,
            table_name: str,
            source_table: str,
            columns: list[str],
            *,
            connection: object,
        ) -> None:
            calls.append((table_name, source_table, tuple(columns), connection))
            if table_name == "unvectorized_chunks_fts":
                raise BackendDatabaseError("chunk fts failed")

    conn = object()
    db = SimpleNamespace(backend=FakeBackend())

    fts_structures_module.ensure_postgres_fts(db, conn)

    assert calls == [
        ("media_fts", "media", ("title", "content"), conn),
        ("keyword_fts", "keywords", ("keyword",), conn),
        ("claims_fts", "claims", ("claim_text",), conn),
        ("unvectorized_chunks_fts", "unvectorizedmediachunks", ("chunk_text",), conn),
    ]


@pytest.mark.unit
def test_email_schema_structures_ensure_sqlite_email_schema_executes_scripts_in_order_and_rebuilds_only_for_new_fts() -> None:
    module_name = "tldw_Server_API.app.core.DB_Management.media_db.schema.email_schema_structures"
    try:
        email_schema_structures_module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        pytest.fail(f"missing package email schema helper module: {exc}")

    class FakeCursor:
        def __init__(self, exists: bool) -> None:
            self._exists = exists

        def fetchone(self):
            return (1,) if self._exists else None

    class FakeConn:
        def __init__(self, *, fts_exists: bool) -> None:
            self.fts_exists = fts_exists
            self.queries: list[str] = []
            self.scripts: list[str] = []

        def execute(self, query: str):
            self.queries.append(query)
            if query.startswith("SELECT 1 FROM sqlite_master"):
                return FakeCursor(self.fts_exists)
            return None

        def executescript(self, script: str) -> None:
            self.scripts.append(script)

    db = SimpleNamespace(
        _EMAIL_SCHEMA_SQL="email schema sql",
        _EMAIL_INDICES_SQL="email indexes sql",
        _EMAIL_SQLITE_FTS_SQL="email fts sql",
    )
    missing_fts_conn = FakeConn(fts_exists=False)
    existing_fts_conn = FakeConn(fts_exists=True)

    email_schema_structures_module.ensure_sqlite_email_schema(db, missing_fts_conn)
    email_schema_structures_module.ensure_sqlite_email_schema(db, existing_fts_conn)

    assert missing_fts_conn.scripts == [
        "email schema sql",
        "email indexes sql",
        "email fts sql",
    ]
    assert existing_fts_conn.scripts == [
        "email schema sql",
        "email indexes sql",
        "email fts sql",
    ]
    assert missing_fts_conn.queries == [
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='email_fts' LIMIT 1",
        "INSERT INTO email_fts(email_fts) VALUES ('rebuild')",
    ]
    assert existing_fts_conn.queries == [
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='email_fts' LIMIT 1",
    ]


@pytest.mark.unit
def test_email_schema_structures_ensure_postgres_email_schema_executes_converted_statements_in_order_and_tolerates_failures() -> None:
    module_name = "tldw_Server_API.app.core.DB_Management.media_db.schema.email_schema_structures"
    try:
        email_schema_structures_module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        pytest.fail(f"missing package email schema helper module: {exc}")

    calls: list[tuple[str, object]] = []

    class FakeBackend:
        def execute(self, query: str, *, connection: object) -> None:
            calls.append((query, connection))
            if query == "CREATE INDEX idx_email_messages_tenant_date_id":
                raise BackendDatabaseError("index create failed")

    def _convert(sql: str) -> list[str]:
        if sql == "email schema sql":
            return [
                "CREATE TABLE email_sources (...)",
                "CREATE TABLE email_messages (...)",
            ]
        if sql == "email indexes sql":
            return [
                "CREATE INDEX idx_email_messages_tenant_date_id",
                "CREATE INDEX idx_email_messages_labels_gin",
            ]
        raise AssertionError(f"unexpected sql blob {sql!r}")

    conn = object()
    db = SimpleNamespace(
        _EMAIL_SCHEMA_SQL="email schema sql",
        _EMAIL_INDICES_SQL="email indexes sql",
        _convert_sqlite_sql_to_postgres_statements=_convert,
        backend=FakeBackend(),
    )

    email_schema_structures_module.ensure_postgres_email_schema(db, conn)

    assert calls == [
        ("CREATE TABLE email_sources (...)", conn),
        ("CREATE TABLE email_messages (...)", conn),
        ("CREATE INDEX idx_email_messages_tenant_date_id", conn),
        ("CREATE INDEX idx_email_messages_labels_gin", conn),
    ]


@pytest.mark.unit
def test_sqlite_post_core_structures_ensure_sqlite_visibility_columns_emits_missing_artifacts_and_noops_when_present() -> None:
    module_name = (
        "tldw_Server_API.app.core.DB_Management.media_db.schema.sqlite_post_core_structures"
    )
    try:
        sqlite_post_core_structures_module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        pytest.fail(f"missing package sqlite post-core helper module: {exc}")

    class FakeCursor:
        def __init__(self, rows):
            self._rows = rows

        def fetchall(self):
            return self._rows

    class FakeConn:
        def __init__(self, *, columns, indexes) -> None:
            self.columns = columns
            self.indexes = indexes
            self.queries: list[str] = []
            self.scripts: list[str] = []

        def execute(self, query: str):
            self.queries.append(query)
            if query == "PRAGMA table_info(Media)":
                return FakeCursor([(0, name) for name in self.columns])
            if query == "PRAGMA index_list(Media)":
                return FakeCursor([(0, name) for name in self.indexes])
            raise AssertionError(f"unexpected query {query!r}")

        def executescript(self, script: str) -> None:
            self.scripts.append(script)

    missing_conn = FakeConn(columns={"id", "title"}, indexes=set())
    present_conn = FakeConn(
        columns={"id", "title", "visibility", "owner_user_id"},
        indexes={"idx_media_visibility", "idx_media_owner_user_id"},
    )

    sqlite_post_core_structures_module.ensure_sqlite_visibility_columns(
        SimpleNamespace(),
        missing_conn,
    )
    sqlite_post_core_structures_module.ensure_sqlite_visibility_columns(
        SimpleNamespace(),
        present_conn,
    )

    assert missing_conn.queries == [
        "PRAGMA table_info(Media)",
        "PRAGMA index_list(Media)",
    ]
    assert missing_conn.scripts == [
        "\n".join(
            [
                "ALTER TABLE Media ADD COLUMN visibility TEXT DEFAULT 'personal' CHECK (visibility IN ('personal', 'team', 'org'));",
                "ALTER TABLE Media ADD COLUMN owner_user_id INTEGER;",
                "CREATE INDEX IF NOT EXISTS idx_media_visibility ON Media(visibility);",
                "CREATE INDEX IF NOT EXISTS idx_media_owner_user_id ON Media(owner_user_id);",
            ]
        )
    ]
    assert present_conn.queries == [
        "PRAGMA table_info(Media)",
        "PRAGMA index_list(Media)",
    ]
    assert present_conn.scripts == []


@pytest.mark.unit
def test_sqlite_post_core_structures_ensure_sqlite_source_hash_column_emits_missing_artifacts_and_noops_when_present() -> None:
    module_name = (
        "tldw_Server_API.app.core.DB_Management.media_db.schema.sqlite_post_core_structures"
    )
    try:
        sqlite_post_core_structures_module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        pytest.fail(f"missing package sqlite post-core helper module: {exc}")

    class FakeCursor:
        def __init__(self, rows):
            self._rows = rows

        def fetchall(self):
            return self._rows

    class FakeConn:
        def __init__(self, *, columns, indexes) -> None:
            self.columns = columns
            self.indexes = indexes
            self.queries: list[str] = []
            self.scripts: list[str] = []

        def execute(self, query: str):
            self.queries.append(query)
            if query == "PRAGMA table_info(Media)":
                return FakeCursor([(0, name) for name in self.columns])
            if query == "PRAGMA index_list(Media)":
                return FakeCursor([(0, name) for name in self.indexes])
            raise AssertionError(f"unexpected query {query!r}")

        def executescript(self, script: str) -> None:
            self.scripts.append(script)

    missing_conn = FakeConn(columns={"id", "title"}, indexes=set())
    present_conn = FakeConn(
        columns={"id", "title", "source_hash"},
        indexes={"idx_media_source_hash"},
    )

    sqlite_post_core_structures_module.ensure_sqlite_source_hash_column(
        SimpleNamespace(),
        missing_conn,
    )
    sqlite_post_core_structures_module.ensure_sqlite_source_hash_column(
        SimpleNamespace(),
        present_conn,
    )

    assert missing_conn.queries == [
        "PRAGMA table_info(Media)",
        "PRAGMA index_list(Media)",
    ]
    assert missing_conn.scripts == [
        "\n".join(
            [
                "ALTER TABLE Media ADD COLUMN source_hash TEXT;",
                "CREATE INDEX IF NOT EXISTS idx_media_source_hash ON Media(source_hash);",
            ]
        )
    ]
    assert present_conn.queries == [
        "PRAGMA table_info(Media)",
        "PRAGMA index_list(Media)",
    ]
    assert present_conn.scripts == []


@pytest.mark.unit
def test_sqlite_post_core_structures_ensure_sqlite_data_tables_executes_sql_and_tolerates_sqlite_errors() -> None:
    import sqlite3

    module_name = (
        "tldw_Server_API.app.core.DB_Management.media_db.schema.sqlite_post_core_structures"
    )
    try:
        sqlite_post_core_structures_module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        pytest.fail(f"missing package sqlite post-core helper module: {exc}")

    class RecordingConn:
        def __init__(self) -> None:
            self.scripts: list[str] = []

        def executescript(self, script: str) -> None:
            self.scripts.append(script)

    class FailingConn:
        def executescript(self, _script: str) -> None:
            raise sqlite3.Error("boom")

    db = SimpleNamespace(_DATA_TABLES_SQL="data tables sql")
    ok_conn = RecordingConn()

    sqlite_post_core_structures_module.ensure_sqlite_data_tables(db, ok_conn)
    sqlite_post_core_structures_module.ensure_sqlite_data_tables(db, FailingConn())

    assert ok_conn.scripts == ["data tables sql"]


@pytest.mark.unit
def test_sqlite_claims_extensions_missing_claims_table_executes_claims_schema_sql_and_returns() -> None:
    module_name = (
        "tldw_Server_API.app.core.DB_Management.media_db.schema.sqlite_claims_extensions"
    )
    try:
        sqlite_claims_extensions_module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        pytest.fail(f"missing package sqlite claims extension helper module: {exc}")

    class FakeCursor:
        def __init__(self, row):
            self._row = row

        def fetchone(self):
            return self._row

    class FakeConn:
        def __init__(self) -> None:
            self.queries: list[str] = []
            self.scripts: list[str] = []

        def execute(self, query: str):
            self.queries.append(query)
            if query == "SELECT name FROM sqlite_master WHERE type='table' AND name='Claims'":
                return FakeCursor(None)
            raise AssertionError(f"unexpected query {query!r}")

        def executescript(self, script: str) -> None:
            self.scripts.append(script)

    conn = FakeConn()
    db = SimpleNamespace(_CLAIMS_TABLE_SQL="claims schema sql")

    sqlite_claims_extensions_module.ensure_sqlite_claims_extensions(db, conn)

    assert conn.queries == [
        "SELECT name FROM sqlite_master WHERE type='table' AND name='Claims'",
    ]
    assert conn.scripts == ["claims schema sql"]


@pytest.mark.unit
def test_sqlite_claims_extensions_repairs_missing_claim_columns_and_events_delivery_artifacts() -> None:
    module_name = (
        "tldw_Server_API.app.core.DB_Management.media_db.schema.sqlite_claims_extensions"
    )
    try:
        sqlite_claims_extensions_module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        pytest.fail(f"missing package sqlite claims extension helper module: {exc}")

    class FakeCursor:
        def __init__(self, *, row=None, rows=None):
            self._row = row
            self._rows = rows or []

        def fetchone(self):
            return self._row

        def fetchall(self):
            return self._rows

    class FakeConn:
        def __init__(self) -> None:
            self.queries: list[str] = []
            self.scripts: list[str] = []

        def execute(self, query: str):
            self.queries.append(query)
            if query == "SELECT name FROM sqlite_master WHERE type='table' AND name='Claims'":
                return FakeCursor(row=("Claims",))
            if query == "PRAGMA table_info(Claims)":
                return FakeCursor(
                    rows=[
                        (0, "id"),
                        (1, "review_group"),
                        (2, "reviewed_at"),
                        (3, "review_notes"),
                        (4, "review_version"),
                        (5, "review_reason_code"),
                    ]
                )
            if query == "PRAGMA table_info(claims_monitoring_events)":
                return FakeCursor(rows=[(0, "id"), (1, "event_type")])
            if query == (
                "CREATE INDEX IF NOT EXISTS idx_claims_monitoring_events_delivered "
                "ON claims_monitoring_events(delivered_at);"
            ):
                return None
            raise AssertionError(f"unexpected query {query!r}")

        def executescript(self, script: str) -> None:
            self.scripts.append(script)

    conn = FakeConn()
    db = SimpleNamespace(_CLAIMS_TABLE_SQL="claims schema sql")

    sqlite_claims_extensions_module.ensure_sqlite_claims_extensions(db, conn)

    assert conn.queries == [
        "SELECT name FROM sqlite_master WHERE type='table' AND name='Claims'",
        "PRAGMA table_info(Claims)",
        "PRAGMA table_info(claims_monitoring_events)",
        "CREATE INDEX IF NOT EXISTS idx_claims_monitoring_events_delivered ON claims_monitoring_events(delivered_at);",
    ]
    assert conn.scripts == [
        "\n".join(
            [
                "ALTER TABLE Claims ADD COLUMN review_status TEXT NOT NULL DEFAULT 'pending';",
                "ALTER TABLE Claims ADD COLUMN reviewer_id INTEGER;",
                "ALTER TABLE Claims ADD COLUMN claim_cluster_id INTEGER;",
            ]
        ),
        "claims schema sql",
        "ALTER TABLE claims_monitoring_events ADD COLUMN delivered_at DATETIME;",
    ]


@pytest.mark.integration
def test_ensure_media_schema_keeps_sqlite_schema_intact() -> None:
    db = MediaDatabase(db_path=":memory:", client_id="schema-bootstrap")
    try:
        ensure_media_schema(db)

        table = db.execute_query(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='Media'"
        ).fetchone()
        version = db.execute_query("SELECT version FROM schema_version").fetchone()

        assert table is not None
        assert version["version"] == db._CURRENT_SCHEMA_VERSION
    finally:
        db.close_connection()


@pytest.mark.integration
def test_fresh_sqlite_bootstrap_includes_transcript_run_history_columns_and_indexes() -> None:
    db = MediaDatabase(db_path=":memory:", client_id="schema-run-history-bootstrap")
    try:
        conn = db.get_connection()
        media_columns = {row[1] for row in conn.execute("PRAGMA table_info(Media)").fetchall()}
        transcript_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(Transcripts)").fetchall()
        }
        transcript_indexes = {
            row[1] for row in conn.execute("PRAGMA index_list(Transcripts)").fetchall()
        }
        media_indexes = {row[1] for row in conn.execute("PRAGMA index_list(Media)").fetchall()}
        transcript_table_sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='Transcripts'"
        ).fetchone()[0]

        assert db._CURRENT_SCHEMA_VERSION == 23
        assert {
            "latest_transcription_run_id",
            "next_transcription_run_id",
        }.issubset(media_columns)
        assert {
            "transcription_run_id",
            "supersedes_run_id",
            "idempotency_key",
        }.issubset(transcript_columns)
        assert "UNIQUE (media_id, whisper_model)" not in transcript_table_sql
        assert {
            "idx_media_latest_transcription_run_id",
            "idx_media_next_transcription_run_id",
        }.issubset(media_indexes)
        assert {
            "idx_transcripts_media_run_id",
            "idx_transcripts_supersedes_run_id",
            "idx_transcripts_media_idempotency_key",
        }.issubset(transcript_indexes)

        now = db._get_current_utc_timestamp_str()
        media_uuid = str(uuid.uuid4())
        conn.execute(
            """
            INSERT INTO Media (uuid, title, type, content_hash, last_modified, version, client_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                media_uuid,
                "Bootstrap uniqueness",
                "audio",
                f"hash-{media_uuid}",
                now,
                1,
                db.client_id,
            ),
        )
        media_id = conn.execute(
            "SELECT id FROM Media WHERE uuid = ?",
            (media_uuid,),
        ).fetchone()[0]

        conn.execute(
            """
            INSERT INTO Transcripts (
                media_id,
                whisper_model,
                transcription,
                created_at,
                transcription_run_id,
                idempotency_key,
                uuid,
                last_modified,
                version,
                client_id,
                deleted
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                media_id,
                "small",
                "baseline transcript",
                now,
                1,
                "job-1",
                str(uuid.uuid4()),
                now,
                1,
                db.client_id,
                0,
            ),
        )
        conn.execute(
            """
            INSERT INTO Transcripts (
                media_id,
                whisper_model,
                transcription,
                created_at,
                transcription_run_id,
                idempotency_key,
                uuid,
                last_modified,
                version,
                client_id,
                deleted
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                media_id,
                "medium",
                "null idempotency transcript",
                now,
                2,
                None,
                str(uuid.uuid4()),
                now,
                1,
                db.client_id,
                0,
            ),
        )
        conn.execute(
            """
            INSERT INTO Transcripts (
                media_id,
                whisper_model,
                transcription,
                created_at,
                transcription_run_id,
                idempotency_key,
                uuid,
                last_modified,
                version,
                client_id,
                deleted
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                media_id,
                "large",
                "second null idempotency transcript",
                now,
                3,
                None,
                str(uuid.uuid4()),
                now,
                1,
                db.client_id,
                0,
            ),
        )

        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO Transcripts (
                    media_id,
                    whisper_model,
                    transcription,
                    created_at,
                    transcription_run_id,
                    idempotency_key,
                    uuid,
                    last_modified,
                    version,
                    client_id,
                    deleted
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    media_id,
                    "duplicate-run",
                    "duplicate run transcript",
                    now,
                    1,
                    "job-2",
                    str(uuid.uuid4()),
                    now,
                    1,
                    db.client_id,
                    0,
                ),
            )

        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO Transcripts (
                    media_id,
                    whisper_model,
                    transcription,
                    created_at,
                    transcription_run_id,
                    idempotency_key,
                    uuid,
                    last_modified,
                    version,
                    client_id,
                    deleted
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    media_id,
                    "duplicate-key",
                    "duplicate idempotency transcript",
                    now,
                    4,
                    "job-1",
                    str(uuid.uuid4()),
                    now,
                    1,
                    db.client_id,
                    0,
                ),
            )
    finally:
        db.close_connection()


@pytest.mark.integration
def test_on_disk_sqlite_migration_to_v23_backfills_transcript_run_history(tmp_path) -> None:
    db_path = tmp_path / "media_v22.sqlite"
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            """
            PRAGMA foreign_keys = ON;
            CREATE TABLE schema_version (
                version INTEGER PRIMARY KEY NOT NULL
            );
            INSERT INTO schema_version(version) VALUES (22);

            CREATE TABLE Media (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT UNIQUE,
                title TEXT NOT NULL,
                type TEXT NOT NULL,
                content TEXT,
                author TEXT,
                ingestion_date DATETIME,
                transcription_model TEXT,
                is_trash BOOLEAN DEFAULT 0 NOT NULL,
                trash_date DATETIME,
                vector_embedding BLOB,
                chunking_status TEXT DEFAULT 'pending' NOT NULL,
                vector_processing INTEGER DEFAULT 0 NOT NULL,
                content_hash TEXT NOT NULL,
                source_hash TEXT,
                uuid TEXT UNIQUE NOT NULL,
                last_modified DATETIME NOT NULL,
                version INTEGER NOT NULL DEFAULT 1,
                org_id INTEGER,
                team_id INTEGER,
                visibility TEXT DEFAULT 'personal' CHECK (visibility IN ('personal', 'team', 'org')),
                owner_user_id INTEGER,
                client_id TEXT NOT NULL,
                deleted BOOLEAN NOT NULL DEFAULT 0,
                prev_version INTEGER,
                merge_parent_uuid TEXT
            );

            CREATE TABLE Keywords (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                keyword TEXT NOT NULL UNIQUE COLLATE NOCASE,
                uuid TEXT UNIQUE NOT NULL,
                last_modified DATETIME NOT NULL,
                version INTEGER NOT NULL DEFAULT 1,
                client_id TEXT NOT NULL,
                deleted BOOLEAN NOT NULL DEFAULT 0,
                prev_version INTEGER,
                merge_parent_uuid TEXT
            );

            CREATE TABLE Transcripts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                media_id INTEGER NOT NULL,
                whisper_model TEXT,
                transcription TEXT,
                created_at DATETIME,
                uuid TEXT UNIQUE NOT NULL,
                last_modified DATETIME NOT NULL,
                version INTEGER NOT NULL DEFAULT 1,
                client_id TEXT NOT NULL,
                deleted BOOLEAN NOT NULL DEFAULT 0,
                prev_version INTEGER,
                merge_parent_uuid TEXT,
                UNIQUE (media_id, whisper_model),
                FOREIGN KEY (media_id) REFERENCES Media(id) ON DELETE CASCADE
            );

            CREATE TABLE Claims (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                media_id INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                claim_text TEXT NOT NULL,
                claim_type TEXT,
                confidence REAL,
                source_excerpt TEXT,
                source_start INTEGER,
                source_end INTEGER,
                normalized_value TEXT,
                context_json TEXT,
                extractor TEXT NOT NULL,
                extractor_version TEXT,
                chunk_hash TEXT NOT NULL,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME,
                review_status TEXT NOT NULL DEFAULT 'pending',
                reviewer_id INTEGER,
                reviewed_at DATETIME,
                review_notes TEXT,
                review_group TEXT,
                review_version INTEGER,
                review_reason_code TEXT,
                claim_cluster_id INTEGER,
                uuid TEXT UNIQUE NOT NULL,
                last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                version INTEGER NOT NULL DEFAULT 1,
                client_id TEXT NOT NULL,
                deleted BOOLEAN NOT NULL DEFAULT 0,
                prev_version INTEGER,
                merge_parent_uuid TEXT,
                FOREIGN KEY (media_id) REFERENCES Media(id) ON DELETE CASCADE,
                UNIQUE (media_id, chunk_hash, extractor, extractor_version)
            );

            INSERT INTO Media (
                id, title, type, content_hash, source_hash, uuid, last_modified, version, visibility, owner_user_id, client_id
            ) VALUES
                (1, 'Media One', 'audio', 'hash-1', 'source-1', 'media-uuid-1', '2024-01-01T00:00:00Z', 1, 'personal', 11, 'client-a'),
                (2, 'Media Two', 'audio', 'hash-2', 'source-2', 'media-uuid-2', '2024-01-01T00:00:00Z', 1, 'personal', 12, 'client-a'),
                (3, 'Media Three', 'audio', 'hash-3', 'source-3', 'media-uuid-3', '2024-01-01T00:00:00Z', 1, 'personal', 13, 'client-a');

            INSERT INTO Keywords (id, keyword, uuid, last_modified, version, client_id)
            VALUES (1, 'bootstrap', 'keyword-uuid-1', '2024-01-01T00:00:00Z', 1, 'client-a');

            INSERT INTO Transcripts (
                id, media_id, whisper_model, transcription, created_at, uuid, last_modified, version, client_id, deleted
            ) VALUES
                (1, 1, 'small', 'newer undeleted transcript', '2024-01-03T00:00:00Z', 'transcript-uuid-1', '2024-01-03T00:00:00Z', 1, 'client-a', 0),
                (2, 1, 'large', 'older undeleted transcript', '2024-01-01T00:00:00Z', 'transcript-uuid-2', '2024-01-01T00:00:00Z', 1, 'client-a', 0),
                (3, 1, 'xlarge', 'deleted newest transcript', '2024-01-04T00:00:00Z', 'transcript-uuid-4', '2024-01-04T00:00:00Z', 1, 'client-a', 1),
                (4, 2, 'small', 'third transcript', '2024-01-02T00:00:00Z', 'transcript-uuid-3', '2024-01-02T00:00:00Z', 1, 'client-a', 0);
            """
        )
        conn.commit()
    finally:
        conn.close()

    db = MediaDatabase(str(db_path), client_id="schema-run-history-migration")
    try:
        db.close_connection()
        db.backend.get_pool().close_all()
        verification_db_path = tmp_path / "media_v23_verification.sqlite"
        with sqlite3.connect(db_path) as source_conn, sqlite3.connect(verification_db_path) as dest_conn:
            source_conn.backup(dest_conn)

        with sqlite3.connect(verification_db_path) as raw_conn:
            raw_conn.row_factory = sqlite3.Row
            media_columns = {row[1] for row in raw_conn.execute("PRAGMA table_info(Media)").fetchall()}
            transcript_columns = {
                row[1] for row in raw_conn.execute("PRAGMA table_info(Transcripts)").fetchall()
            }
            transcript_indexes = {
                row[1] for row in raw_conn.execute("PRAGMA index_list(Transcripts)").fetchall()
            }
            transcript_table_sql = raw_conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='Transcripts'"
            ).fetchone()[0]
            transcript_rows = [
                dict(row)
                for row in raw_conn.execute(
                    """
                    SELECT id, media_id, whisper_model, created_at, deleted, transcription_run_id, supersedes_run_id, idempotency_key
                    FROM Transcripts
                    ORDER BY media_id, id
                    """
                ).fetchall()
            ]
            media_rows = [
                dict(row)
                for row in raw_conn.execute(
                    """
                    SELECT id, latest_transcription_run_id, next_transcription_run_id
                    FROM Media
                    ORDER BY id
                    """
                ).fetchall()
            ]
            version_row = raw_conn.execute(
                "SELECT version FROM schema_version LIMIT 1"
            ).fetchone()
            assert version_row["version"] == 23
            assert {
                "latest_transcription_run_id",
                "next_transcription_run_id",
            }.issubset(media_columns)
            assert {
                "transcription_run_id",
                "supersedes_run_id",
                "idempotency_key",
            }.issubset(transcript_columns)
            assert "UNIQUE (media_id, whisper_model)" not in transcript_table_sql
            assert {
                "idx_transcripts_media_run_id",
                "idx_transcripts_supersedes_run_id",
                "idx_transcripts_media_idempotency_key",
            }.issubset(transcript_indexes)
            assert transcript_rows == [
                {
                    "id": 1,
                    "media_id": 1,
                    "whisper_model": "small",
                    "created_at": "2024-01-03T00:00:00Z",
                    "deleted": 0,
                    "transcription_run_id": 2,
                    "supersedes_run_id": None,
                    "idempotency_key": None,
                },
                {
                    "id": 2,
                    "media_id": 1,
                    "whisper_model": "large",
                    "created_at": "2024-01-01T00:00:00Z",
                    "deleted": 0,
                    "transcription_run_id": 1,
                    "supersedes_run_id": None,
                    "idempotency_key": None,
                },
                {
                    "id": 3,
                    "media_id": 1,
                    "whisper_model": "xlarge",
                    "created_at": "2024-01-04T00:00:00Z",
                    "deleted": 1,
                    "transcription_run_id": 3,
                    "supersedes_run_id": None,
                    "idempotency_key": None,
                },
                {
                    "id": 4,
                    "media_id": 2,
                    "whisper_model": "small",
                    "created_at": "2024-01-02T00:00:00Z",
                    "deleted": 0,
                    "transcription_run_id": 1,
                    "supersedes_run_id": None,
                    "idempotency_key": None,
                },
            ]
            assert media_rows == [
                {"id": 1, "latest_transcription_run_id": 2, "next_transcription_run_id": 4},
                {"id": 2, "latest_transcription_run_id": 1, "next_transcription_run_id": 2},
                {"id": 3, "latest_transcription_run_id": None, "next_transcription_run_id": 1},
            ]

            with pytest.raises(sqlite3.IntegrityError):
                raw_conn.execute(
                    """
                    INSERT INTO Transcripts (
                        media_id,
                        whisper_model,
                        transcription,
                        created_at,
                        transcription_run_id,
                        idempotency_key,
                        uuid,
                        last_modified,
                        version,
                        client_id,
                        deleted
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        1,
                        "duplicate-run",
                        "duplicate run transcript",
                        "2024-01-05T00:00:00Z",
                        2,
                        "job-dup-run",
                        str(uuid.uuid4()),
                        "2024-01-05T00:00:00Z",
                        1,
                        "client-a",
                        0,
                    ),
                )
            raw_conn.rollback()

            raw_conn.execute(
                """
                INSERT INTO Transcripts (
                    media_id,
                    whisper_model,
                    transcription,
                    created_at,
                    transcription_run_id,
                    idempotency_key,
                    uuid,
                    last_modified,
                    version,
                    client_id,
                    deleted
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    1,
                    "nullable-key-one",
                    "nullable key transcript",
                    "2024-01-05T00:00:00Z",
                    4,
                    None,
                    str(uuid.uuid4()),
                    "2024-01-05T00:00:00Z",
                    1,
                    "client-a",
                    0,
                ),
            )
            raw_conn.execute(
                """
                INSERT INTO Transcripts (
                    media_id,
                    whisper_model,
                    transcription,
                    created_at,
                    transcription_run_id,
                    idempotency_key,
                    uuid,
                    last_modified,
                    version,
                    client_id,
                    deleted
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    1,
                    "idempotency-anchor",
                    "idempotency key anchor",
                    "2024-01-06T00:00:00Z",
                    5,
                    "job-unique",
                    str(uuid.uuid4()),
                    "2024-01-06T00:00:00Z",
                    1,
                    "client-a",
                    0,
                ),
            )
            raw_conn.commit()

            with pytest.raises(sqlite3.IntegrityError):
                raw_conn.execute(
                    """
                    INSERT INTO Transcripts (
                        media_id,
                        whisper_model,
                        transcription,
                        created_at,
                        transcription_run_id,
                        idempotency_key,
                        uuid,
                        last_modified,
                        version,
                        client_id,
                        deleted
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        1,
                        "duplicate-key",
                        "duplicate key transcript",
                        "2024-01-07T00:00:00Z",
                        6,
                        "job-unique",
                        str(uuid.uuid4()),
                        "2024-01-07T00:00:00Z",
                        1,
                        "client-a",
                        0,
                    ),
                )
            raw_conn.rollback()

            with pytest.raises(sqlite3.IntegrityError, match="Sync Error \\(Transcripts\\)"):
                raw_conn.execute(
                    "UPDATE Transcripts SET version = version, client_id = client_id WHERE id = 1"
                )
            raw_conn.rollback()

    finally:
        db.close_connection()
