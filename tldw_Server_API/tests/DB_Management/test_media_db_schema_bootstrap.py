import importlib
import sqlite3
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType, DatabaseConfig
from tldw_Server_API.app.core.DB_Management.backends.base import (
    DatabaseError as BackendDatabaseError,
)
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.DB_Management.media_db.schema import bootstrap as bootstrap_module
from tldw_Server_API.app.core.DB_Management.media_db.schema.backends import postgres as postgres_backend_module
from tldw_Server_API.app.core.DB_Management.media_db.schema.backends import sqlite as sqlite_backend_module
from tldw_Server_API.app.core.DB_Management.media_db.schema.bootstrap import ensure_media_schema


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
def test_real_sqlite_schema_bootstrap_creates_core_objects(tmp_path) -> None:
    """Run the REAL bootstrap end-to-end against a fresh SQLite database.

    The dispatch tests above verify routing with the initializers stubbed;
    this one closes the gap flagged in
    audits/2026-07-04-test-suite-audit-round2.md (RA2): nothing here ever
    executed the actual schema DDL, so a broken bootstrap kept the suite
    green.
    """
    db = MediaDatabase(db_path=str(tmp_path / "media.db"), client_id="schema-bootstrap-test")
    try:
        rows = db.execute_query(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
        tables = {row["name"] for row in rows}
        assert {"Media", "schema_version", "sync_log", "Keywords", "MediaKeywords"}.issubset(
            tables
        ), f"core tables missing from real bootstrap: {sorted(tables)}"

        fts_rows = db.execute_query(
            "SELECT name FROM sqlite_master WHERE name LIKE '%_fts%'"
        ).fetchall()
        assert fts_rows, "FTS structures missing from real bootstrap"

        version_row = db.execute_query("SELECT version FROM schema_version LIMIT 1").fetchone()
        assert version_row is not None and int(version_row["version"]) >= 1

        index_rows = db.execute_query(
            "SELECT name FROM sqlite_master WHERE type = 'index' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        assert index_rows, "no indexes created by real bootstrap"

        # idempotence: re-running the bootstrap on an initialized DB is a no-op
        ensure_media_schema(db)
    finally:
        db.close_connection()


@pytest.mark.unit
def test_ensure_postgres_post_core_structures_runs_followup_ensures(monkeypatch) -> None:
    from tldw_Server_API.app.core.DB_Management.media_db.schema.backends import (
        postgres_helpers as postgres_helpers_module,
    )

    conn = object()
    calls: list[object] = []
    backend = SimpleNamespace(
        execute=lambda statement, *, connection: calls.append(("safe_transcript_extractor", statement, connection))
    )
    db = SimpleNamespace(
        backend=backend,
        _ensure_postgres_collections_tables=lambda value: calls.append(("collections", value)),
        _ensure_postgres_tts_history=lambda value: calls.append(("tts_history", value)),
        _ensure_postgres_audio_presets=lambda value: calls.append(("audio_presets", value)),
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
    monkeypatch.setattr(
        postgres_helpers_module,
        "ensure_postgres_document_workspace_schema",
        lambda connection: calls.append(("document_workspace", connection)),
        raising=False,
    )

    postgres_helpers_module.ensure_postgres_post_core_structures(db, conn)

    assert calls == [
        ("collections", conn),
        ("tts_history", conn),
        ("audio_presets", conn),
        ("data_tables", conn),
        (
            "safe_transcript_extractor",
            postgres_helpers_module._SAFE_TRANSCRIPT_TEXT_FUNCTION_SQL,
            conn,
        ),
        ("document_workspace", conn),
        ("source_hash", conn),
        ("claims_extensions", conn),
        ("email_schema", conn),
        ("sequence_sync", conn),
        ("policies", db, conn),
    ]


@pytest.mark.unit
def test_postgres_safe_transcript_extractor_is_pg13_compatible_and_non_throwing() -> None:
    from tldw_Server_API.app.core.DB_Management.media_db.schema.backends import (
        postgres_helpers as postgres_helpers_module,
    )

    ddl = " ".join(postgres_helpers_module._SAFE_TRANSCRIPT_TEXT_FUNCTION_SQL.split()).lower()

    assert "create or replace function public.tldw_try_extract_normalized_transcript_text" in ddl
    assert "language plpgsql" in ddl
    assert "immutable" in ddl
    assert "strict" in ddl
    assert "set search_path = pg_catalog" in ddl
    assert "exception when data_exception then return null" in ddl
    assert "json_typeof(parsed) <> 'object'" in ddl
    assert "value_type := json_typeof(parsed -> 'text')" in ddl
    assert "value_type is null or value_type = 'null'" in ddl
    assert "if value_type = 'boolean'" in ddl
    assert "pg_input_is_valid" not in ddl
    assert " is json" not in ddl


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
def test_audio_preset_partial_indexes_convert_to_postgres_boolean_predicates() -> None:
    from tldw_Server_API.app.core.DB_Management.media_db.schema import (
        postgres_sqlite_conversion as postgres_sqlite_conversion_module,
    )

    statements = postgres_sqlite_conversion_module._convert_sqlite_sql_to_postgres_statements(
        SimpleNamespace(),
        MediaDatabase._AUDIO_PRESETS_TABLE_SQL,
    )
    converted_sql = "\n".join(statements).upper()

    assert "WHERE DELETED = FALSE" in converted_sql
    assert "WHERE IS_DEFAULT = TRUE AND DELETED = FALSE" in converted_sql
    assert "WHERE DELETED = 0" not in converted_sql
    assert "WHERE IS_DEFAULT = 1" not in converted_sql


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
    audio_presets_sql = "CREATE TABLE IF NOT EXISTS audio_presets (id INTEGER PRIMARY KEY);"

    class FakeConn:
        def executescript(self, script: str) -> None:
            label = "audio_presets_sql" if script == audio_presets_sql else "collections_sql"
            calls.append((label, script))

    conn = FakeConn()
    db = SimpleNamespace(
        _AUDIO_PRESETS_TABLE_SQL=audio_presets_sql,
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
    monkeypatch.setattr(
        sqlite_helpers_module,
        "ensure_sqlite_document_workspace_schema",
        lambda connection: calls.append(("document_workspace", connection)),
        raising=False,
    )

    sqlite_helpers_module.ensure_sqlite_post_core_structures(db, conn)

    assert [entry[0] for entry in calls] == [
        "data_tables",
        "fts",
        "collections_sql",
        "audio_presets_sql",
        "document_workspace",
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
    assert calls[3] == ("audio_presets_sql", audio_presets_sql)
    assert calls[4:] == [
        ("document_workspace", conn),
        ("visibility", conn),
        ("source_hash", conn),
        ("claims_extensions", conn),
        ("email_schema", conn),
    ]


@pytest.mark.unit
def test_ensure_sqlite_document_workspace_schema_creates_tables_indexes_and_columns() -> None:
    from tldw_Server_API.app.core.DB_Management.media_db.schema.document_workspace_schema import (
        ensure_sqlite_document_workspace_schema,
    )

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    try:
        ensure_sqlite_document_workspace_schema(conn)

        table_names = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name LIKE 'document_%'"
            ).fetchall()
        }
        assert {
            "document_reading_progress",
            "document_annotations",
            "document_parsed_references_cache",
        }.issubset(table_names)

        progress_columns = {row["name"] for row in conn.execute("PRAGMA table_info(document_reading_progress)")}
        annotation_columns = {row["name"] for row in conn.execute("PRAGMA table_info(document_annotations)")}
        cache_columns = {row["name"] for row in conn.execute("PRAGMA table_info(document_parsed_references_cache)")}

        assert {"cfi", "percentage"}.issubset(progress_columns)
        assert {"chapter_title", "percentage"}.issubset(annotation_columns)
        assert {"references_json", "total_detected", "updated_at"}.issubset(cache_columns)

        annotation_indexes = {row["name"] for row in conn.execute("PRAGMA index_list(document_annotations)")}
        cache_indexes = {row["name"] for row in conn.execute("PRAGMA index_list(document_parsed_references_cache)")}
        assert "idx_annotations_media_user" in annotation_indexes
        assert "idx_doc_refs_cache_lookup" in cache_indexes
    finally:
        conn.close()


@pytest.mark.unit
def test_ensure_sqlite_document_workspace_schema_migrates_old_tables_idempotently() -> None:
    from tldw_Server_API.app.core.DB_Management.media_db.schema.document_workspace_schema import (
        ensure_sqlite_document_workspace_schema,
    )

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(
            """
            CREATE TABLE document_reading_progress (
                media_id INTEGER NOT NULL,
                user_id TEXT NOT NULL,
                current_page INTEGER NOT NULL DEFAULT 1,
                total_pages INTEGER NOT NULL DEFAULT 1,
                zoom_level INTEGER NOT NULL DEFAULT 100,
                view_mode TEXT NOT NULL DEFAULT 'single',
                last_read_at TEXT NOT NULL,
                PRIMARY KEY (media_id, user_id)
            );
            CREATE TABLE document_annotations (
                id TEXT PRIMARY KEY,
                media_id INTEGER NOT NULL,
                user_id TEXT NOT NULL,
                location TEXT NOT NULL,
                text TEXT NOT NULL,
                color TEXT NOT NULL DEFAULT 'yellow',
                note TEXT,
                annotation_type TEXT NOT NULL DEFAULT 'highlight',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                deleted INTEGER NOT NULL DEFAULT 0
            );
            """
        )

        ensure_sqlite_document_workspace_schema(conn)
        ensure_sqlite_document_workspace_schema(conn)

        progress_columns = [row["name"] for row in conn.execute("PRAGMA table_info(document_reading_progress)")]
        annotation_columns = [row["name"] for row in conn.execute("PRAGMA table_info(document_annotations)")]

        assert progress_columns.count("cfi") == 1
        assert progress_columns.count("percentage") == 1
        assert annotation_columns.count("chapter_title") == 1
        assert annotation_columns.count("percentage") == 1
    finally:
        conn.close()


@pytest.mark.unit
def test_ensure_postgres_document_workspace_schema_executes_expected_ddl() -> None:
    from tldw_Server_API.app.core.DB_Management.media_db.schema.document_workspace_schema import (
        ensure_postgres_document_workspace_schema,
    )

    statements: list[str] = []

    class FakePostgresConn:
        def execute(self, statement: str) -> None:
            statements.append(" ".join(statement.split()))

    ensure_postgres_document_workspace_schema(FakePostgresConn())

    joined = "\n".join(statements).lower()
    assert "create table if not exists document_reading_progress" in joined
    assert "primary key (media_id, user_id)" in joined
    assert "create table if not exists document_annotations" in joined
    assert "create index if not exists idx_annotations_media_user" in joined
    assert "create table if not exists document_parsed_references_cache" in joined
    assert "primary key (media_id, user_id, parser_version, content_hash)" in joined
    assert "create index if not exists idx_doc_refs_cache_lookup" in joined
    assert "alter table document_reading_progress add column if not exists cfi text" in joined
    assert "alter table document_reading_progress add column if not exists percentage double precision" in joined
    assert "alter table document_annotations add column if not exists chapter_title text" in joined
    assert "alter table document_annotations add column if not exists percentage double precision" in joined


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
            if query == "PRAGMA table_info(claims_analytics_exports)":
                return FakeCursor(rows=[(0, "export_id"), (1, "snapshot_at")])
            if query == (
                "CREATE INDEX IF NOT EXISTS "
                "idx_claims_analytics_exports_user_status_export_id "
                "ON claims_analytics_exports(user_id, status, export_id);"
            ):
                return None
            if query == (
                "CREATE INDEX IF NOT EXISTS "
                "idx_claims_analytics_exports_user_status_updated_export_id "
                "ON claims_analytics_exports(user_id, status, updated_at, export_id);"
            ):
                return None
            if query == (
                "CREATE INDEX IF NOT EXISTS idx_claims_monitoring_events_delivered "
                "ON claims_monitoring_events(delivered_at);"
            ):
                return None
            if query == (
                "CREATE INDEX IF NOT EXISTS idx_claims_monitoring_events_user_created_id "
                "ON claims_monitoring_events(user_id, created_at, id);"
            ):
                return None
            if query == (
                "CREATE INDEX IF NOT EXISTS idx_claims_monitoring_events_user_id "
                "ON claims_monitoring_events(user_id, id);"
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
        "PRAGMA table_info(claims_analytics_exports)",
        "CREATE INDEX IF NOT EXISTS idx_claims_analytics_exports_user_status_export_id ON claims_analytics_exports(user_id, status, export_id);",
        "CREATE INDEX IF NOT EXISTS idx_claims_analytics_exports_user_status_updated_export_id ON claims_analytics_exports(user_id, status, updated_at, export_id);",
        "PRAGMA table_info(claims_monitoring_events)",
        "CREATE INDEX IF NOT EXISTS idx_claims_monitoring_events_delivered ON claims_monitoring_events(delivered_at);",
        "CREATE INDEX IF NOT EXISTS idx_claims_monitoring_events_user_created_id ON claims_monitoring_events(user_id, created_at, id);",
        "CREATE INDEX IF NOT EXISTS idx_claims_monitoring_events_user_id ON claims_monitoring_events(user_id, id);",
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
        "ALTER TABLE claims_analytics_exports ADD COLUMN snapshot_event_id INTEGER;",
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

        assert db._CURRENT_SCHEMA_VERSION == 26
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

            CREATE TABLE claims_analytics_exports (
                export_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                format TEXT NOT NULL,
                status TEXT NOT NULL,
                payload_json TEXT,
                payload_csv TEXT,
                filters_json TEXT,
                pagination_json TEXT,
                error_message TEXT,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
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
        verification_db_path = tmp_path / "media_v24_verification.sqlite"
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
            event_index_columns = {
                row["name"]: [
                    column["name"]
                    for column in raw_conn.execute(
                        f"PRAGMA index_info({row['name']})"
                    ).fetchall()
                ]
                for row in raw_conn.execute(
                    "PRAGMA index_list(claims_monitoring_events)"
                ).fetchall()
            }
            assert version_row["version"] == 26
            assert event_index_columns["idx_claims_monitoring_events_user_created_id"] == [
                "user_id",
                "created_at",
                "id",
            ]
            assert event_index_columns["idx_claims_monitoring_events_user_id"] == [
                "user_id",
                "id",
            ]
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


@pytest.mark.integration
def test_fresh_sqlite_bootstrap_includes_claims_analytics_export_job_fields() -> None:
    db = MediaDatabase(db_path=":memory:", client_id="claims-export-jobs-bootstrap")
    try:
        conn = db.get_connection()
        columns = {
            row[1]: {"type": row[2], "notnull": row[3]}
            for row in conn.execute(
                "PRAGMA table_info(claims_analytics_exports)"
            ).fetchall()
        }
        indexes = {
            row[1]
            for row in conn.execute(
                "PRAGMA index_list(claims_analytics_exports)"
            ).fetchall()
        }
        event_indexes = {
            row[1]
            for row in conn.execute(
                "PRAGMA index_list(claims_monitoring_events)"
            ).fetchall()
        }
        version = conn.execute(
            "SELECT version FROM schema_version LIMIT 1"
        ).fetchone()[0]

        assert version == 26
        assert db._CURRENT_SCHEMA_VERSION == 26
        assert columns["job_id"] == {"type": "INTEGER", "notnull": 0}
        assert columns["error_code"] == {"type": "TEXT", "notnull": 0}
        assert columns["snapshot_at"] == {"type": "TEXT", "notnull": 0}
        assert columns["snapshot_event_id"] == {"type": "INTEGER", "notnull": 0}
        assert "job_status" not in columns
        assert "idx_claims_analytics_exports_job_id" in indexes
        assert {
            "idx_claims_monitoring_events_user_created_id",
            "idx_claims_monitoring_events_user_id",
        }.issubset(event_indexes)
    finally:
        db.close_connection()


@pytest.mark.integration
def test_on_disk_sqlite_migration_to_v24_adds_claims_export_job_fields_and_preserves_rows(
    tmp_path,
) -> None:
    db_path = tmp_path / "media_v23_claims_exports.sqlite"
    db = MediaDatabase(str(db_path), client_id="claims-export-jobs-migration")
    db.close_connection()

    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            DROP INDEX IF EXISTS idx_claims_analytics_exports_job_id;
            ALTER TABLE claims_analytics_exports RENAME TO claims_analytics_exports_v24;
            CREATE TABLE claims_analytics_exports (
                export_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                format TEXT NOT NULL,
                status TEXT NOT NULL,
                payload_json TEXT,
                payload_csv TEXT,
                filters_json TEXT,
                pagination_json TEXT,
                error_message TEXT,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            INSERT INTO claims_analytics_exports (
                export_id,
                user_id,
                format,
                status,
                payload_json,
                filters_json,
                created_at,
                updated_at
            ) VALUES (
                'export-existing',
                'user-1',
                'json',
                'ready',
                '{"claim_count": 1}',
                '{"status": "verified"}',
                '2026-01-01T00:00:00Z',
                '2026-01-01T00:00:00Z'
            );
            DROP TABLE claims_analytics_exports_v24;
            CREATE INDEX IF NOT EXISTS idx_claims_analytics_exports_user
                ON claims_analytics_exports(user_id);
            UPDATE schema_version SET version = 23;
            """
        )

    try:
        db._initialize_schema()

        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            columns = {
                row["name"]: {"type": row["type"], "notnull": row["notnull"]}
                for row in conn.execute(
                    "PRAGMA table_info(claims_analytics_exports)"
                ).fetchall()
            }
            indexes = {
                row["name"]
                for row in conn.execute(
                    "PRAGMA index_list(claims_analytics_exports)"
                ).fetchall()
            }
            monitoring_index_columns = [
                row["name"]
                for row in conn.execute(
                    "PRAGMA index_info(idx_claims_monitoring_events_user_created_id)"
                ).fetchall()
            ]
            version = conn.execute(
                "SELECT version FROM schema_version LIMIT 1"
            ).fetchone()["version"]

        assert version == 26
        assert columns["job_id"] == {"type": "INTEGER", "notnull": 0}
        assert columns["error_code"] == {"type": "TEXT", "notnull": 0}
        assert columns["snapshot_at"] == {"type": "TEXT", "notnull": 0}
        assert columns["snapshot_event_id"] == {"type": "INTEGER", "notnull": 0}
        assert "job_status" not in columns
        assert "idx_claims_analytics_exports_job_id" in indexes
        assert "idx_claims_analytics_exports_user_status_export_id" in indexes
        assert "idx_claims_analytics_exports_user_status_updated_export_id" in indexes
        assert monitoring_index_columns == ["user_id", "created_at", "id"]

        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            export_row = dict(
                conn.execute(
                    """
                    SELECT export_id, user_id, format, status, payload_json,
                           filters_json, job_id, error_code, snapshot_at, snapshot_event_id
                    FROM claims_analytics_exports
                    WHERE export_id = 'export-existing'
                    """
                ).fetchone()
            )
        assert export_row == {
            "export_id": "export-existing",
            "user_id": "user-1",
            "format": "json",
            "status": "ready",
            "payload_json": '{"claim_count": 1}',
            "filters_json": '{"status": "verified"}',
            "job_id": None,
            "error_code": None,
            "snapshot_at": None,
            "snapshot_event_id": None,
        }
    finally:
        db.close_connection()


@pytest.mark.integration
def test_current_v24_sqlite_bootstrap_repairs_export_snapshot_column_and_event_indexes(
    tmp_path,
) -> None:
    db_path = tmp_path / "media_v24_claims_events.sqlite"
    db = MediaDatabase(str(db_path), client_id="claims-event-snapshot-index")
    try:
        with sqlite3.connect(db_path) as conn:
            conn.executescript(
                """
                DROP INDEX idx_claims_monitoring_events_user_created_id;
                DROP INDEX IF EXISTS idx_claims_monitoring_events_user_id;
                ALTER TABLE claims_analytics_exports RENAME TO claims_analytics_exports_v24;
                CREATE TABLE claims_analytics_exports (
                    export_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    format TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload_json TEXT,
                    payload_csv TEXT,
                    filters_json TEXT,
                    pagination_json TEXT,
                    error_message TEXT,
                    job_id INTEGER,
                    error_code TEXT,
                    snapshot_at TEXT,
                    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                DROP TABLE claims_analytics_exports_v24;
                """
            )

        db._initialize_schema()
        created = db.create_claims_analytics_export(
            export_id="current-v24-repaired",
            user_id="owner-1",
            format="json",
            status="queued",
            snapshot_event_id=0,
        )

        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            event_index_columns = [
                row["name"]
                for row in conn.execute(
                    "PRAGMA index_info(idx_claims_monitoring_events_user_created_id)"
                ).fetchall()
            ]
            high_water_index_columns = [
                row["name"]
                for row in conn.execute(
                    "PRAGMA index_info(idx_claims_monitoring_events_user_id)"
                ).fetchall()
            ]
            export_columns = {
                row["name"]
                for row in conn.execute(
                    "PRAGMA table_info(claims_analytics_exports)"
                ).fetchall()
            }
            export_indexes = {
                row["name"]
                for row in conn.execute(
                    "PRAGMA index_list(claims_analytics_exports)"
                ).fetchall()
            }
        assert created["snapshot_event_id"] == 0
        assert "snapshot_event_id" in export_columns
        assert "idx_claims_analytics_exports_user_status_export_id" in export_indexes
        assert (
            "idx_claims_analytics_exports_user_status_updated_export_id"
            in export_indexes
        )
        assert event_index_columns == ["user_id", "created_at", "id"]
        assert high_water_index_columns == ["user_id", "id"]
    finally:
        db.close_connection()


@pytest.mark.unit
def test_sqlite_migration_024_loads_as_idempotent(tmp_path) -> None:
    from tldw_Server_API.app.core.DB_Management.db_migration import DatabaseMigrator

    migrator = DatabaseMigrator(str(tmp_path / "migration-loader.sqlite"))
    migration = next(item for item in migrator.load_migrations() if item.version == 24)

    assert migration.name == "claims_analytics_export_jobs"
    assert migration.idempotent is True


@pytest.mark.unit
def test_sqlite_migration_024_defers_monitoring_event_indexes_to_extension() -> None:
    migration_path = (
        Path(__file__).parents[2]
        / "app/core/DB_Management/migrations/024_claims_analytics_export_jobs.sql"
    )

    assert "claims_monitoring_events" not in migration_path.read_text(encoding="utf-8")


@pytest.mark.unit
def test_postgres_migration_v24_defers_monitoring_event_indexes_to_extension() -> None:
    from tldw_Server_API.app.core.DB_Management.media_db.schema.migration_bodies.postgres_claims_analytics_export_jobs import (
        run_postgres_migrate_to_v24,
    )

    statements: list[str] = []

    class Backend:
        @staticmethod
        def escape_identifier(name: str) -> str:
            return f'"{name}"'

        @staticmethod
        def execute(
            query: str,
            params=None,
            *,
            connection,
        ) -> None:
            del params, connection
            statements.append(query)

    run_postgres_migrate_to_v24(
        SimpleNamespace(backend=Backend()),
        object(),
    )

    assert any("idx_claims_analytics_exports_job_id" in query for query in statements)
    assert any(
        "idx_claims_analytics_exports_user_status_export_id" in query
        for query in statements
    )
    assert any(
        "idx_claims_analytics_exports_user_status_updated_export_id" in query
        for query in statements
    )
    assert all("claims_monitoring_events" not in query for query in statements)


@pytest.mark.integration
@pytest.mark.parametrize(
    ("present_column_definitions", "job_index_present"),
    [
        (("job_id INTEGER",), False),
        (("job_id INTEGER", "error_code TEXT", "snapshot_at TEXT"), True),
    ],
    ids=("partial-v24-ddl", "full-v24-ddl"),
)
def test_on_disk_sqlite_migration_to_v24_recovers_idempotently_from_present_ddl(
    tmp_path,
    present_column_definitions: tuple[str, ...],
    job_index_present: bool,
) -> None:
    db_path = tmp_path / "media_v23_partial_claims_exports.sqlite"
    db = MediaDatabase(str(db_path), client_id="claims-export-jobs-recovery")
    db.close_connection()

    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            DROP INDEX IF EXISTS idx_claims_analytics_exports_job_id;
            ALTER TABLE claims_analytics_exports RENAME TO claims_analytics_exports_v24;
            CREATE TABLE claims_analytics_exports (
                export_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                format TEXT NOT NULL,
                status TEXT NOT NULL,
                payload_json TEXT,
                payload_csv TEXT,
                filters_json TEXT,
                pagination_json TEXT,
                error_message TEXT,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            INSERT INTO claims_analytics_exports (
                export_id,
                user_id,
                format,
                status,
                payload_json,
                created_at,
                updated_at
            ) VALUES (
                'export-recovery',
                'user-1',
                'json',
                'ready',
                '{"claim_count": 2}',
                '2026-01-02T00:00:00Z',
                '2026-01-02T00:00:00Z'
            );
            DROP TABLE claims_analytics_exports_v24;
            CREATE INDEX IF NOT EXISTS idx_claims_analytics_exports_user
                ON claims_analytics_exports(user_id);

            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                checksum TEXT NOT NULL,
                applied_at TIMESTAMP NOT NULL,
                execution_time REAL NOT NULL,
                success BOOLEAN NOT NULL DEFAULT 1,
                error_message TEXT
            );
            DELETE FROM schema_migrations;
            INSERT INTO schema_migrations (
                version,
                name,
                checksum,
                applied_at,
                execution_time,
                success,
                error_message
            ) VALUES (
                23,
                'transcript_run_history',
                'test-v23-checksum',
                '2026-01-02T00:00:00Z',
                0,
                1,
                NULL
            );
            UPDATE schema_version SET version = 23;
            """
        )
        for column_definition in present_column_definitions:
            conn.execute(
                f"ALTER TABLE claims_analytics_exports ADD COLUMN {column_definition}"
            )
        if job_index_present:
            conn.execute(
                """
                CREATE INDEX idx_claims_analytics_exports_job_id
                    ON claims_analytics_exports(job_id)
                """
            )
        conn.commit()

    try:
        db._initialize_schema()

        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            columns = {
                row["name"]: {"type": row["type"], "notnull": row["notnull"]}
                for row in conn.execute(
                    "PRAGMA table_info(claims_analytics_exports)"
                ).fetchall()
            }
            indexes = {
                row["name"]
                for row in conn.execute(
                    "PRAGMA index_list(claims_analytics_exports)"
                ).fetchall()
            }
            monitoring_index_columns = [
                row["name"]
                for row in conn.execute(
                    "PRAGMA index_info(idx_claims_monitoring_events_user_created_id)"
                ).fetchall()
            ]
            version = conn.execute(
                "SELECT version FROM schema_version LIMIT 1"
            ).fetchone()["version"]
            migration_row = dict(
                conn.execute(
                    """
                    SELECT version, name, success
                    FROM schema_migrations
                    WHERE version = 24
                    """
                ).fetchone()
            )
            export_row = dict(
                conn.execute(
                    """
                    SELECT export_id, payload_json, job_id, error_code, snapshot_at, snapshot_event_id
                    FROM claims_analytics_exports
                    WHERE export_id = 'export-recovery'
                    """
                ).fetchone()
            )

        assert version == 26
        assert columns["job_id"] == {"type": "INTEGER", "notnull": 0}
        assert columns["error_code"] == {"type": "TEXT", "notnull": 0}
        assert columns["snapshot_at"] == {"type": "TEXT", "notnull": 0}
        assert columns["snapshot_event_id"] == {"type": "INTEGER", "notnull": 0}
        assert "idx_claims_analytics_exports_job_id" in indexes
        assert "idx_claims_analytics_exports_user_status_export_id" in indexes
        assert "idx_claims_analytics_exports_user_status_updated_export_id" in indexes
        assert monitoring_index_columns == ["user_id", "created_at", "id"]
        assert migration_row == {
            "version": 24,
            "name": "claims_analytics_export_jobs",
            "success": 1,
        }
        assert export_row == {
            "export_id": "export-recovery",
            "payload_json": '{"claim_count": 2}',
            "job_id": None,
            "error_code": None,
            "snapshot_at": None,
            "snapshot_event_id": None,
        }
    finally:
        db.close_connection()


_INVALID_OPERATION_OWNERSHIP_MARKERS = (
    ("operation-only", None, None, None),
    ("operation-null-kind", None, "source-null-kind", "a" * 64),
    ("", "shared_workspace_clone", "source-empty-operation", "a" * 64),
    ("o" * 256, "shared_workspace_clone", "source-long-operation", "a" * 64),
    ("operation-empty-source", "shared_workspace_clone", "", "a" * 64),
    ("operation-long-source", "shared_workspace_clone", "s" * 256, "a" * 64),
    ("operation-wrong-kind", "other_kind", "source-wrong-kind", "a" * 64),
    ("operation-uppercase-hash", "shared_workspace_clone", "source-hash", "A" * 64),
)


def _assert_sqlite_rejects_invalid_operation_markers(
    connection: sqlite3.Connection,
    *,
    client_id: str,
) -> None:
    for index, marker_set in enumerate(_INVALID_OPERATION_OWNERSHIP_MARKERS):
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO Media (uuid, title, type, content_hash, last_modified, client_id, "
                "system_operation_id, system_operation_kind, system_source_identity, "
                "system_content_hash) VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?)",
                (
                    str(uuid.uuid4()),
                    f"invalid owned media {index}",
                    "text",
                    f"invalid-{index}",
                    client_id,
                    *marker_set,
                ),
            )


@pytest.mark.integration
def test_fresh_sqlite_bootstrap_includes_operation_owned_media_schema_v26() -> None:
    db = MediaDatabase(db_path=":memory:", client_id="owned-media-v26-bootstrap")
    try:
        conn = db.get_connection()
        columns = {
            row[1]: {"type": row[2], "notnull": row[3]}
            for row in conn.execute("PRAGMA table_info(Media)").fetchall()
        }
        index_rows = {
            row[1]: {"unique": row[2], "partial": row[4]}
            for row in conn.execute("PRAGMA index_list(Media)").fetchall()
        }
        index_sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'index' "
            "AND name = 'ux_media_system_operation_source'"
        ).fetchone()[0]
        hold_columns = {
            row[1]: {"type": row[2], "notnull": row[3], "pk": row[5]}
            for row in conn.execute(
                "PRAGMA table_info(OperationOwnedCloneKeywords)"
            ).fetchall()
        }
        version = conn.execute("SELECT version FROM schema_version LIMIT 1").fetchone()[0]

        assert version == 26
        assert db._CURRENT_SCHEMA_VERSION == 26
        assert {
            "system_operation_id",
            "system_operation_kind",
            "system_source_identity",
            "system_content_hash",
        }.issubset(columns)
        assert all(
            columns[column] == {"type": "TEXT", "notnull": 0}
            for column in (
                "system_operation_id",
                "system_operation_kind",
                "system_source_identity",
                "system_content_hash",
            )
        )
        assert index_rows["ux_media_system_operation_source"] == {
            "unique": 1,
            "partial": 1,
        }
        assert "WHERE system_operation_id IS NOT NULL" in index_sql
        assert set(hold_columns) == {
            "media_id",
            "keyword",
            "operation_id",
            "source_identity",
            "client_id",
        }
        assert hold_columns["keyword"] == {
            "type": "TEXT",
            "notnull": 1,
            "pk": 2,
        }

        now = db._get_current_utc_timestamp_str()
        conn.execute(
            "INSERT INTO Media (uuid, title, type, content_hash, last_modified, client_id) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), "ordinary", "text", "ordinary-hash", now, db.client_id),
        )
        _assert_sqlite_rejects_invalid_operation_markers(
            conn,
            client_id=db.client_id,
        )
        conn.execute(
            "INSERT INTO Media (uuid, title, type, content_hash, last_modified, client_id, "
            "system_operation_id, system_operation_kind, system_source_identity, "
            "system_content_hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                str(uuid.uuid4()),
                "owned",
                "text",
                "owned-hash",
                now,
                db.client_id,
                "operation-valid",
                "shared_workspace_clone",
                "source-valid",
                "a" * 64,
            ),
        )
    finally:
        db.close_connection()


@pytest.mark.integration
@pytest.mark.postgres
@pytest.mark.timeout(120)
def test_fresh_postgres_v26_enforces_media_markers_and_pending_keyword_shape(
    pg_database_config: DatabaseConfig,
) -> None:
    from tldw_Server_API.app.core.DB_Management.scope_context import scoped_context

    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = MediaDatabase(db_path=":memory:", client_id="901", backend=backend)
    try:
        with scoped_context(user_id=901, org_ids=[], team_ids=[], is_admin=True):
            for index, marker_set in enumerate(_INVALID_OPERATION_OWNERSHIP_MARKERS):
                with pytest.raises(BackendDatabaseError):
                    with db.transaction() as connection:
                        backend.execute(
                            "INSERT INTO Media (uuid, title, type, content_hash, last_modified, "
                            "client_id, system_operation_id, system_operation_kind, "
                            "system_source_identity, system_content_hash) VALUES "
                            "(%s, %s, %s, %s, CURRENT_TIMESTAMP, %s, %s, %s, %s, %s)",
                            (
                                str(uuid.uuid4()),
                                f"invalid owned media {index}",
                                "text",
                                f"invalid-{index}",
                                "901",
                                *marker_set,
                            ),
                            connection=connection,
                        )

            with db.transaction() as connection:
                media_id = backend.execute(
                    "INSERT INTO Media (uuid, title, type, content_hash, last_modified, "
                    "client_id, is_trash, system_operation_id, system_operation_kind, "
                    "system_source_identity, system_content_hash) VALUES "
                    "(%s, %s, %s, %s, CURRENT_TIMESTAMP, %s, TRUE, %s, %s, %s, %s) "
                    "RETURNING id",
                    (
                        str(uuid.uuid4()),
                        "valid owned media",
                        "text",
                        "valid-content",
                        "901",
                        "operation-valid",
                        "shared_workspace_clone",
                        "source-valid",
                        "a" * 64,
                    ),
                    connection=connection,
                ).rows[0]["id"]

            invalid_pending_rows = (
                ("Uppercase", "operation-valid", "source-valid", "901"),
                ("valid", "", "source-valid", "901"),
                ("valid", "operation-valid", "s" * 256, "901"),
                ("valid", "operation-valid", "source-valid", ""),
            )
            for keyword, operation_id, source_identity, client_id in invalid_pending_rows:
                with pytest.raises(BackendDatabaseError):
                    with db.transaction() as connection:
                        backend.execute(
                            "INSERT INTO OperationOwnedCloneKeywords "
                            "(media_id, keyword, operation_id, source_identity, client_id) "
                            "VALUES (%s, %s, %s, %s, %s)",
                            (media_id, keyword, operation_id, source_identity, client_id),
                            connection=connection,
                        )
    finally:
        db.close_connection()
        backend.get_pool().close_all()


def _create_minimal_media_v24_database(db_path: Path) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE schema_version (version INTEGER PRIMARY KEY NOT NULL);
            INSERT INTO schema_version(version) VALUES (24);
            CREATE TABLE schema_migrations (
                version INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                checksum TEXT NOT NULL,
                applied_at TIMESTAMP NOT NULL,
                execution_time REAL NOT NULL,
                success BOOLEAN NOT NULL DEFAULT 1,
                error_message TEXT
            );
            INSERT INTO schema_migrations (
                version, name, checksum, applied_at, execution_time, success
            ) VALUES (24, 'claims_analytics_export_jobs', 'test-v24', CURRENT_TIMESTAMP, 0, 1);
            CREATE TABLE Media (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT UNIQUE,
                title TEXT NOT NULL,
                type TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                uuid TEXT UNIQUE NOT NULL,
                last_modified TEXT NOT NULL,
                client_id TEXT NOT NULL,
                deleted INTEGER NOT NULL DEFAULT 0,
                is_trash INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE Keywords (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                keyword TEXT NOT NULL UNIQUE COLLATE NOCASE,
                uuid TEXT UNIQUE NOT NULL,
                last_modified TEXT NOT NULL,
                version INTEGER NOT NULL DEFAULT 1,
                client_id TEXT NOT NULL,
                deleted INTEGER NOT NULL DEFAULT 0
            );
            INSERT INTO Media (
                url, title, type, content_hash, uuid, last_modified, client_id
            ) VALUES (
                'https://ordinary.example.test/v24', 'ordinary v24', 'text',
                'ordinary-v24-hash', 'ordinary-v24-uuid', CURRENT_TIMESTAMP, 'client-v24'
            );
            """
        )


@pytest.mark.integration
def test_sqlite_migration_v25_recovers_partial_ddl_and_preserves_ordinary_rows(
    tmp_path: Path,
) -> None:
    from tldw_Server_API.app.core.DB_Management.db_migration import DatabaseMigrator

    db_path = tmp_path / "media-v24-operation-owned-partial.sqlite"
    _create_minimal_media_v24_database(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute("ALTER TABLE Media ADD COLUMN system_operation_id TEXT")
        conn.execute("ALTER TABLE Media ADD COLUMN system_operation_kind TEXT")

    migrator = DatabaseMigrator(str(db_path))
    result = migrator.migrate_to_version(25, create_backup=False)

    assert result["status"] == "success"
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(Media)")}
        indexes = {row["name"] for row in conn.execute("PRAGMA index_list(Media)")}
        hold_table = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' "
            "AND name = 'OperationOwnedCloneKeywords'"
        ).fetchone()
        version = conn.execute("SELECT version FROM schema_version").fetchone()[0]
        migration = dict(
            conn.execute(
                "SELECT version, name, success FROM schema_migrations WHERE version = 25"
            ).fetchone()
        )
        ordinary = dict(
            conn.execute(
                "SELECT title, system_operation_id, system_operation_kind, "
                "system_source_identity, system_content_hash FROM Media WHERE id = 1"
            ).fetchone()
        )
        _assert_sqlite_rejects_invalid_operation_markers(
            conn,
            client_id="client-v24",
        )

    assert version == 25
    assert {
        "system_operation_id",
        "system_operation_kind",
        "system_source_identity",
        "system_content_hash",
    }.issubset(columns)
    assert "ux_media_system_operation_source" in indexes
    assert hold_table is not None
    assert migration == {
        "version": 25,
        "name": "operation_owned_clone_media",
        "success": 1,
    }
    assert ordinary == {
        "title": "ordinary v24",
        "system_operation_id": None,
        "system_operation_kind": None,
        "system_source_identity": None,
        "system_content_hash": None,
    }


@pytest.mark.integration
def test_sqlite_migration_v25_does_not_advance_version_when_ddl_rolls_back(
    tmp_path: Path,
) -> None:
    from tldw_Server_API.app.core.DB_Management.db_migration import (
        DatabaseMigrator,
        MigrationError,
    )

    db_path = tmp_path / "media-v24-operation-owned-interrupted.sqlite"
    _create_minimal_media_v24_database(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TRIGGER reject_media_v25_version
            BEFORE UPDATE ON schema_version
            WHEN NEW.version = 25
            BEGIN
                SELECT RAISE(ABORT, 'simulated version write interruption');
            END;
            """
        )

    migrator = DatabaseMigrator(str(db_path))
    with pytest.raises(MigrationError):
        migrator.migrate_to_version(25, create_backup=False)

    with sqlite3.connect(db_path) as conn:
        columns_after_failure = {
            row[1] for row in conn.execute("PRAGMA table_info(Media)").fetchall()
        }
        version_after_failure = conn.execute(
            "SELECT version FROM schema_version"
        ).fetchone()[0]
        conn.execute("DROP TRIGGER reject_media_v25_version")

    assert version_after_failure == 24
    assert "system_operation_id" not in columns_after_failure

    result = migrator.migrate_to_version(25, create_backup=False)
    assert result["status"] == "success"
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT version FROM schema_version").fetchone()[0] == 25
        assert "system_content_hash" in {
            row[1] for row in conn.execute("PRAGMA table_info(Media)").fetchall()
        }


@pytest.mark.integration
def test_sqlite_migration_v24_to_v26_repairs_partial_v25_and_preserves_rows(
    tmp_path: Path,
) -> None:
    from tldw_Server_API.app.core.DB_Management.db_migration import DatabaseMigrator

    db_path = tmp_path / "media-v24-to-v26.sqlite"
    _create_minimal_media_v24_database(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute("ALTER TABLE Media ADD COLUMN system_operation_id TEXT")

    result = DatabaseMigrator(str(db_path)).migrate_to_version(26, create_backup=False)

    assert result["status"] == "success"
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        assert conn.execute("SELECT version FROM schema_version").fetchone()[0] == 26
        assert conn.execute("SELECT title FROM Media WHERE id = 1").fetchone()[0] == "ordinary v24"
        assert {row["name"] for row in conn.execute(
            "PRAGMA table_info(OperationOwnedCloneKeywords)"
        )} == {"media_id", "keyword", "operation_id", "source_identity", "client_id"}
        _assert_sqlite_rejects_invalid_operation_markers(conn, client_id="client-v24")


@pytest.mark.integration
@pytest.mark.parametrize("with_old_keyword_holds", [False, True])
def test_sqlite_migration_v25_variants_to_v26_preserve_pending_values(
    tmp_path: Path,
    with_old_keyword_holds: bool,
) -> None:
    from tldw_Server_API.app.core.DB_Management.db_migration import DatabaseMigrator

    db_path = tmp_path / f"media-v25-variant-{int(with_old_keyword_holds)}.sqlite"
    _create_minimal_media_v24_database(db_path)
    migrator = DatabaseMigrator(str(db_path))
    assert migrator.migrate_to_version(25, create_backup=False)["status"] == "success"

    with sqlite3.connect(db_path) as conn:
        if with_old_keyword_holds:
            conn.executescript(
                """
                INSERT INTO Keywords (
                    keyword, uuid, last_modified, client_id
                ) VALUES (' Pending Value ', 'pending-keyword-uuid', CURRENT_TIMESTAMP, 'client-v25');
                INSERT INTO Media (
                    title, type, content_hash, uuid, last_modified, client_id, is_trash,
                    system_operation_id, system_operation_kind,
                    system_source_identity, system_content_hash
                ) VALUES (
                    'pending v25', 'text', 'pending-content', 'pending-media-uuid',
                    CURRENT_TIMESTAMP, 'client-v25', 1, 'operation-v25',
                    'shared_workspace_clone', 'source-v25',
                    'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
                );
                INSERT INTO OperationOwnedCloneKeywords (
                    media_id, keyword_id, operation_id, source_identity, created_by_clone
                ) VALUES (2, 1, 'operation-v25', 'source-v25', 1);
                """
            )
        else:
            conn.execute("DROP TABLE OperationOwnedCloneKeywords")

    assert migrator.migrate_to_version(26, create_backup=False)["status"] == "success"
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        columns = {row["name"] for row in conn.execute(
            "PRAGMA table_info(OperationOwnedCloneKeywords)"
        )}
        pending = [dict(row) for row in conn.execute(
            "SELECT media_id, keyword, operation_id, source_identity, client_id "
            "FROM OperationOwnedCloneKeywords"
        )]
        assert columns == {"media_id", "keyword", "operation_id", "source_identity", "client_id"}
        assert pending == ([{
            "media_id": 2,
            "keyword": "pending value",
            "operation_id": "operation-v25",
            "source_identity": "source-v25",
            "client_id": "client-v25",
        }] if with_old_keyword_holds else [])


@pytest.mark.integration
def test_sqlite_migration_v26_rolls_back_and_retries_after_interruption(
    tmp_path: Path,
) -> None:
    from tldw_Server_API.app.core.DB_Management.db_migration import (
        DatabaseMigrator,
        MigrationError,
    )

    db_path = tmp_path / "media-v25-to-v26-interrupted.sqlite"
    _create_minimal_media_v24_database(db_path)
    migrator = DatabaseMigrator(str(db_path))
    assert migrator.migrate_to_version(25, create_backup=False)["status"] == "success"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TRIGGER reject_media_v26_version
            BEFORE UPDATE ON schema_version
            WHEN NEW.version = 26
            BEGIN
                SELECT RAISE(ABORT, 'simulated v26 version write interruption');
            END;
            """
        )

    with pytest.raises(MigrationError):
        migrator.migrate_to_version(26, create_backup=False)

    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT version FROM schema_version").fetchone()[0] == 25
        assert "keyword_id" in {
            row[1] for row in conn.execute("PRAGMA table_info(OperationOwnedCloneKeywords)")
        }
        conn.execute("DROP TRIGGER reject_media_v26_version")

    assert migrator.migrate_to_version(26, create_backup=False)["status"] == "success"
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT version FROM schema_version").fetchone()[0] == 26
        assert "keyword" in {
            row[1] for row in conn.execute("PRAGMA table_info(OperationOwnedCloneKeywords)")
        }


@pytest.mark.integration
def test_sqlite_migration_v26_replays_after_committed_script_missing_bookkeeping(
    tmp_path: Path,
) -> None:
    from tldw_Server_API.app.core.DB_Management.db_migration import DatabaseMigrator

    db_path = tmp_path / "media-v26-committed-script-replay.sqlite"
    _create_minimal_media_v24_database(db_path)
    migrator = DatabaseMigrator(str(db_path))
    assert migrator.migrate_to_version(25, create_backup=False)["status"] == "success"

    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE MediaKeywords (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                media_id INTEGER NOT NULL,
                keyword_id INTEGER NOT NULL,
                UNIQUE (media_id, keyword_id),
                FOREIGN KEY (media_id) REFERENCES Media(id) ON DELETE CASCADE,
                FOREIGN KEY (keyword_id) REFERENCES Keywords(id) ON DELETE CASCADE
            );
            INSERT INTO Keywords (
                keyword, uuid, last_modified, client_id
            ) VALUES (' Replay Pending ', 'replay-keyword-uuid', CURRENT_TIMESTAMP, 'client-v25');
            INSERT INTO Media (
                title, type, content_hash, uuid, last_modified, client_id, is_trash,
                system_operation_id, system_operation_kind,
                system_source_identity, system_content_hash
            ) VALUES (
                'replay pending v25', 'text', 'replay-content', 'replay-media-uuid',
                CURRENT_TIMESTAMP, 'client-v25', 1, 'operation-replay-v25',
                'shared_workspace_clone', 'source-replay-v25',
                'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
            );
            INSERT INTO MediaKeywords (media_id, keyword_id) VALUES (2, 1);
            INSERT INTO OperationOwnedCloneKeywords (
                media_id, keyword_id, operation_id, source_identity, created_by_clone
            ) VALUES (2, 1, 'operation-replay-v25', 'source-replay-v25', 0);
            """
        )

    migration = next(item for item in migrator.load_migrations() if item.version == 26)
    migrator.execute_migration(migration)

    def read_v26_state() -> dict[str, object]:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            return {
                "version": conn.execute(
                    "SELECT version FROM schema_version"
                ).fetchone()[0],
                "pending": [
                    dict(row)
                    for row in conn.execute(
                        "SELECT media_id, keyword, operation_id, source_identity, client_id "
                        "FROM OperationOwnedCloneKeywords ORDER BY media_id, keyword"
                    )
                ],
                "media": dict(
                    conn.execute(
                        "SELECT system_operation_id, system_operation_kind, "
                        "system_source_identity, system_content_hash "
                        "FROM Media WHERE id = 2"
                    ).fetchone()
                ),
                "links": conn.execute(
                    "SELECT COUNT(*) FROM MediaKeywords WHERE media_id = 2"
                ).fetchone()[0],
                "keywords": [
                    row[0]
                    for row in conn.execute(
                        "SELECT keyword FROM Keywords ORDER BY id"
                    )
                ],
                "indexes": [
                    row[0]
                    for row in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'index' "
                        "AND name IN (?, ?) ORDER BY name",
                        (
                            "idx_owned_clone_keywords_keyword",
                            "idx_owned_clone_keywords_operation",
                        ),
                    )
                ],
                "triggers": [
                    row[0]
                    for row in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'trigger' "
                        "AND name IN (?, ?) ORDER BY name",
                        (
                            "media_validate_system_operation_insert_v26",
                            "media_validate_system_operation_update_v26",
                        ),
                    )
                ],
            }

    state_after_first_run = read_v26_state()
    with sqlite3.connect(db_path) as conn:
        conn.execute("DELETE FROM schema_migrations WHERE version = 26")
        conn.commit()

    migrator.execute_migration(migration)

    assert read_v26_state() == state_after_first_run
    assert state_after_first_run == {
        "version": 26,
        "pending": [
            {
                "media_id": 2,
                "keyword": "replay pending",
                "operation_id": "operation-replay-v25",
                "source_identity": "source-replay-v25",
                "client_id": "client-v25",
            }
        ],
        "media": {
            "system_operation_id": "operation-replay-v25",
            "system_operation_kind": "shared_workspace_clone",
            "system_source_identity": "source-replay-v25",
            "system_content_hash": "a" * 64,
        },
        "links": 0,
        "keywords": [" Replay Pending "],
        "indexes": [
            "idx_owned_clone_keywords_keyword",
            "idx_owned_clone_keywords_operation",
        ],
        "triggers": [
            "media_validate_system_operation_insert_v26",
            "media_validate_system_operation_update_v26",
        ],
    }


@pytest.mark.unit
def test_sqlite_migration_025_loads_as_idempotent(tmp_path: Path) -> None:
    from tldw_Server_API.app.core.DB_Management.db_migration import DatabaseMigrator

    migrator = DatabaseMigrator(str(tmp_path / "migration-v25-loader.sqlite"))
    migration = next(item for item in migrator.load_migrations() if item.version == 25)

    assert migration.name == "operation_owned_clone_media"
    assert migration.idempotent is True


@pytest.mark.unit
def test_sqlite_migration_026_loads_as_atomic_non_idempotent_script(tmp_path: Path) -> None:
    from tldw_Server_API.app.core.DB_Management.db_migration import DatabaseMigrator

    migrator = DatabaseMigrator(str(tmp_path / "migration-v26-loader.sqlite"))
    migration = next(item for item in migrator.load_migrations() if item.version == 26)

    assert migration.name == "finalize_staged_clone_persistence"
    assert migration.idempotent is False


@pytest.mark.unit
def test_postgres_migration_v25_body_adds_owned_media_columns_constraint_and_index() -> None:
    from tldw_Server_API.app.core.DB_Management.media_db.schema.migration_bodies.postgres_operation_owned_clone_media import (
        run_postgres_migrate_to_v25,
    )

    statements: list[str] = []

    class Backend:
        @staticmethod
        def escape_identifier(name: str) -> str:
            return f'"{name}"'

        @staticmethod
        def execute(query: str, params=None, *, connection) -> None:
            del params, connection
            statements.append(query)

    run_postgres_migrate_to_v25(SimpleNamespace(backend=Backend()), object())

    combined_sql = "\n".join(statements)
    assert sum("ADD COLUMN IF NOT EXISTS" in query for query in statements) == 4
    assert any("ck_media_system_operation_ownership" in query for query in statements)
    assert any("ux_media_system_operation_source" in query for query in statements)
    assert any("operationownedclonekeywords" in query.lower() for query in statements)
    assert '"system_content_hash" IS NOT NULL' in combined_sql
    assert "^[0-9a-f]{64}$" in combined_sql


@pytest.mark.unit
def test_postgres_migration_v26_body_repairs_markers_and_pending_keywords() -> None:
    from tldw_Server_API.app.core.DB_Management.media_db.schema.migration_bodies.postgres_staged_clone_persistence import (
        run_postgres_migrate_to_v26,
    )

    statements: list[str] = []

    class Backend:
        @staticmethod
        def escape_identifier(name: str) -> str:
            return f'"{name}"'

        @staticmethod
        def execute(query: str, params=None, *, connection):
            del params, connection
            statements.append(query)
            normalized = " ".join(query.split()).lower()
            if "select index_row.relname as index_name" in normalized:
                return SimpleNamespace(rows=[])
            if "as table_name" in normalized and "from pg_class" in normalized:
                return SimpleNamespace(
                    rows=[
                        {
                            "table_name": name,
                            "rls_enabled": name in {"media", "operationownedclonekeywords"},
                            "rls_forced": name in {"media", "operationownedclonekeywords"},
                            "is_table_owner": True,
                            "is_schema_owner": True,
                        }
                        for name in (
                            "media",
                            "keywords",
                            "mediakeywords",
                            "operationownedclonekeywords",
                        )
                    ]
                )
            if "select table_row.relname" in normalized:
                return SimpleNamespace(
                    rows=[
                        {"relname": name}
                        for name in (
                            "media",
                            "keywords",
                            "mediakeywords",
                            "operationownedclonekeywords",
                        )
                    ]
                )
            if "select column_name" in normalized:
                return SimpleNamespace(
                    rows=[
                        {"column_name": name}
                        for name in (
                            "media_id",
                            "keyword_id",
                            "operation_id",
                            "source_identity",
                            "created_by_clone",
                        )
                    ]
                )
            if "select count(*) as count" in normalized:
                return SimpleNamespace(rows=[{"count": 0}])
            return SimpleNamespace(rows=[])

    run_postgres_migrate_to_v26(SimpleNamespace(backend=Backend()), object())

    combined_sql = "\n".join(statements).lower()
    assert "operationownedclonekeywords_v25" in combined_sql
    assert "join \"keywords\"" in combined_sql
    assert '"keyword" text not null' in combined_sql
    assert '"client_id" text not null' in combined_sql
    assert "drop constraint if exists" in combined_sql
    assert "ck_media_system_operation_ownership" in combined_sql
    assert "^[0-9a-f]{64}$" in combined_sql
    assert "in access exclusive mode" in combined_sql
    assert "no force row level security" in combined_sql
    assert "force row level security" in combined_sql


@pytest.mark.integration
@pytest.mark.postgres
@pytest.mark.timeout(120)
@pytest.mark.parametrize("with_old_keyword_holds", [False, True])
def test_postgres_migration_v26_repairs_both_committed_v25_shapes(
    pg_database_config: DatabaseConfig,
    with_old_keyword_holds: bool,
) -> None:
    from tldw_Server_API.app.core.DB_Management.media_db.schema.migration_bodies.postgres_staged_clone_persistence import (
        run_postgres_migrate_to_v26,
    )
    from tldw_Server_API.app.core.DB_Management.scope_context import scoped_context

    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = MediaDatabase(db_path=":memory:", client_id="901", backend=backend)
    pending_rows: list[dict[str, object]] = []
    try:
        with scoped_context(user_id=901, org_ids=[], team_ids=[], is_admin=True):
            with db.transaction() as connection:
                backend.execute(
                    "DROP TABLE operationownedclonekeywords CASCADE",
                    connection=connection,
                )
                if with_old_keyword_holds:
                    backend.execute(
                        """
                        CREATE TABLE operationownedclonekeywords (
                            media_id BIGINT NOT NULL REFERENCES media(id) ON DELETE CASCADE,
                            keyword_id BIGINT NOT NULL REFERENCES keywords(id) ON DELETE CASCADE,
                            operation_id TEXT NOT NULL,
                            source_identity TEXT NOT NULL,
                            created_by_clone BOOLEAN NOT NULL,
                            PRIMARY KEY (media_id, keyword_id)
                        )
                        """,
                        connection=connection,
                    )
                    backend.execute(
                        "CREATE INDEX idx_owned_clone_keywords_keyword "
                        "ON operationownedclonekeywords (keyword_id)",
                        connection=connection,
                    )
                    backend.execute(
                        "CREATE INDEX idx_owned_clone_keywords_operation "
                        "ON operationownedclonekeywords (operation_id, source_identity)",
                        connection=connection,
                    )
                    media_row = backend.execute(
                        "INSERT INTO Media (title, type, content_hash, uuid, last_modified, "
                        "client_id, is_trash, system_operation_id, system_operation_kind, "
                        "system_source_identity, system_content_hash) VALUES "
                        "(%s, %s, %s, %s, CURRENT_TIMESTAMP, %s, TRUE, %s, %s, %s, %s) "
                        "RETURNING id",
                        (
                            "pending pg v25",
                            "text",
                            "pending-content",
                            str(uuid.uuid4()),
                            "901",
                            "operation-pg-v25",
                            "shared_workspace_clone",
                            "source-pg-v25",
                            "a" * 64,
                        ),
                        connection=connection,
                    ).rows[0]
                    keyword_row = backend.execute(
                        "INSERT INTO Keywords (keyword, uuid, last_modified, client_id, deleted) "
                        "VALUES (%s, %s, CURRENT_TIMESTAMP, %s, FALSE) RETURNING id",
                        (" Pending PG Value ", str(uuid.uuid4()), "901"),
                        connection=connection,
                    ).rows[0]
                    backend.execute(
                        "INSERT INTO operationownedclonekeywords "
                        "(media_id, keyword_id, operation_id, source_identity, created_by_clone) "
                        "VALUES (%s, %s, %s, %s, TRUE)",
                        (
                            media_row["id"],
                            keyword_row["id"],
                            "operation-pg-v25",
                            "source-pg-v25",
                        ),
                        connection=connection,
                    )

                run_postgres_migrate_to_v26(db, connection)
                columns = {
                    row["column_name"]
                    for row in backend.execute(
                        "SELECT column_name FROM information_schema.columns "
                        "WHERE table_schema = current_schema() "
                        "AND table_name = 'operationownedclonekeywords'",
                        connection=connection,
                    ).rows
                }
                pending_rows = backend.execute(
                    "SELECT media_id, keyword, operation_id, source_identity, client_id "
                    "FROM operationownedclonekeywords",
                    connection=connection,
                ).rows
                index_owners = {
                    row["index_name"]: row["table_name"]
                    for row in backend.execute(
                        "SELECT index_row.relname AS index_name, "
                        "indexed_table.relname AS table_name "
                        "FROM pg_class AS index_row "
                        "JOIN pg_namespace AS namespace_row "
                        "ON namespace_row.oid = index_row.relnamespace "
                        "JOIN pg_index AS index_meta "
                        "ON index_meta.indexrelid = index_row.oid "
                        "JOIN pg_class AS indexed_table "
                        "ON indexed_table.oid = index_meta.indrelid "
                        "WHERE namespace_row.nspname = current_schema() "
                        "AND index_row.relname = ANY(%s)",
                        (
                            [
                                "idx_owned_clone_keywords_keyword",
                                "idx_owned_clone_keywords_operation",
                            ],
                        ),
                        connection=connection,
                    ).rows
                }

            assert columns == {
                "media_id",
                "keyword",
                "operation_id",
                "source_identity",
                "client_id",
            }
            assert pending_rows == ([{
                "media_id": media_row["id"],
                "keyword": "pending pg value",
                "operation_id": "operation-pg-v25",
                "source_identity": "source-pg-v25",
                "client_id": "901",
            }] if with_old_keyword_holds else [])
            assert index_owners == {
                "idx_owned_clone_keywords_keyword": "operationownedclonekeywords",
                "idx_owned_clone_keywords_operation": "operationownedclonekeywords",
            }
    finally:
        db.close_connection()
        backend.get_pool().close_all()


@pytest.mark.integration
@pytest.mark.postgres
@pytest.mark.timeout(120)
def test_postgres_v26_migration_releases_real_v25_keyword_graph(
    pg_database_config: DatabaseConfig,
) -> None:
    from tldw_Server_API.app.core.DB_Management.media_db.schema.migration_bodies.postgres_staged_clone_persistence import (
        run_postgres_migrate_to_v26,
    )
    from tldw_Server_API.app.core.DB_Management.scope_context import scoped_context

    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = MediaDatabase(db_path=":memory:", client_id="901", backend=backend)
    try:
        with scoped_context(user_id=901, org_ids=[], team_ids=[], is_admin=True):
            with db.transaction() as connection:
                backend.execute(
                    "DROP TABLE operationownedclonekeywords CASCADE",
                    connection=connection,
                )
                backend.execute(
                    """
                    CREATE TABLE operationownedclonekeywords (
                        media_id BIGINT NOT NULL REFERENCES media(id) ON DELETE CASCADE,
                        keyword_id BIGINT NOT NULL REFERENCES keywords(id) ON DELETE CASCADE,
                        operation_id TEXT NOT NULL,
                        source_identity TEXT NOT NULL,
                        created_by_clone BOOLEAN NOT NULL,
                        PRIMARY KEY (media_id, keyword_id)
                    )
                    """,
                    connection=connection,
                )
                staged_id = backend.execute(
                    "INSERT INTO Media (title, type, content_hash, uuid, last_modified, "
                    "client_id, is_trash, system_operation_id, system_operation_kind, "
                    "system_source_identity, system_content_hash) VALUES "
                    "(%s, %s, %s, %s, CURRENT_TIMESTAMP, %s, TRUE, %s, %s, %s, %s) "
                    "RETURNING id",
                    (
                        "pending pg fix1 v25",
                        "text",
                        "pending-content",
                        str(uuid.uuid4()),
                        "901",
                        "operation-pg-fix1-v25",
                        "shared_workspace_clone",
                        "source-pg-fix1-v25",
                        "a" * 64,
                    ),
                    connection=connection,
                ).rows[0]["id"]
                ordinary_id = backend.execute(
                    "INSERT INTO Media (title, type, content_hash, uuid, last_modified, client_id) "
                    "VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP, %s) RETURNING id",
                    ("ordinary keyword owner", "text", "ordinary", str(uuid.uuid4()), "901"),
                    connection=connection,
                ).rows[0]["id"]
                keyword_ids: dict[str, int] = {}
                for keyword in ("clone-orphan-pg", "recipient-existing-pg", "clone-shared-pg"):
                    keyword_ids[keyword] = int(
                        backend.execute(
                            "INSERT INTO Keywords "
                            "(keyword, uuid, last_modified, client_id, deleted) "
                            "VALUES (%s, %s, CURRENT_TIMESTAMP, %s, FALSE) RETURNING id",
                            (keyword, str(uuid.uuid4()), "901"),
                            connection=connection,
                        ).rows[0]["id"]
                    )
                for keyword, created_by_clone in (
                    ("clone-orphan-pg", True),
                    ("recipient-existing-pg", False),
                    ("clone-shared-pg", True),
                ):
                    backend.execute(
                        "INSERT INTO MediaKeywords (media_id, keyword_id) VALUES (%s, %s)",
                        (staged_id, keyword_ids[keyword]),
                        connection=connection,
                    )
                    backend.execute(
                        "INSERT INTO operationownedclonekeywords "
                        "(media_id, keyword_id, operation_id, source_identity, created_by_clone) "
                        "VALUES (%s, %s, %s, %s, %s)",
                        (
                            staged_id,
                            keyword_ids[keyword],
                            "operation-pg-fix1-v25",
                            "source-pg-fix1-v25",
                            created_by_clone,
                        ),
                        connection=connection,
                    )
                backend.execute(
                    "INSERT INTO MediaKeywords (media_id, keyword_id) VALUES (%s, %s)",
                    (ordinary_id, keyword_ids["clone-shared-pg"]),
                    connection=connection,
                )

                run_postgres_migrate_to_v26(db, connection)

                assert {
                    row["keyword"]
                    for row in backend.execute(
                        "SELECT keyword FROM operationownedclonekeywords WHERE media_id = %s",
                        (staged_id,),
                        connection=connection,
                    ).rows
                } == {"clone-orphan-pg", "recipient-existing-pg", "clone-shared-pg"}
                assert backend.execute(
                    "SELECT COUNT(*) AS count FROM MediaKeywords WHERE media_id = %s",
                    (staged_id,),
                    connection=connection,
                ).rows[0]["count"] == 0
                assert {
                    row["keyword"]
                    for row in backend.execute(
                        "SELECT keyword FROM Keywords WHERE keyword = ANY(%s)",
                        (["clone-orphan-pg", "recipient-existing-pg", "clone-shared-pg"],),
                        connection=connection,
                    ).rows
                } == {"recipient-existing-pg", "clone-shared-pg"}
                assert backend.execute(
                    "SELECT COUNT(*) AS count FROM MediaKeywords "
                    "WHERE media_id = %s AND keyword_id = %s",
                    (ordinary_id, keyword_ids["clone-shared-pg"]),
                    connection=connection,
                ).rows[0]["count"] == 1
    finally:
        db.close_connection()
        backend.get_pool().close_all()


@pytest.mark.integration
@pytest.mark.postgres
@pytest.mark.timeout(120)
def test_postgres_v26_migration_clears_weak_v25_partial_null_markers(
    pg_database_config: DatabaseConfig,
) -> None:
    from tldw_Server_API.app.core.DB_Management.scope_context import scoped_context

    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = MediaDatabase(db_path=":memory:", client_id="901", backend=backend)
    partial_rows = [
        (str(uuid.uuid4()), "partial-operation", None, None, None),
        (str(uuid.uuid4()), None, "shared_workspace_clone", None, "a" * 64),
        (str(uuid.uuid4()), "partial-operation", None, "partial-source", "b" * 64),
    ]
    try:
        with scoped_context(user_id=901, org_ids=[], team_ids=[], is_admin=True):
            with db.transaction() as connection:
                backend.execute(
                    "ALTER TABLE Media DROP CONSTRAINT ck_media_system_operation_ownership",
                    connection=connection,
                )
                backend.execute(
                    """
                    ALTER TABLE Media
                    ADD CONSTRAINT ck_media_system_operation_ownership
                    CHECK (
                        (
                            system_operation_id IS NULL
                            AND system_operation_kind IS NULL
                            AND system_source_identity IS NULL
                            AND system_content_hash IS NULL
                        )
                        OR
                        (
                            system_operation_kind = 'shared_workspace_clone'
                            AND system_content_hash ~ '^[0-9a-f]{64}$'
                        )
                    )
                    """,
                    connection=connection,
                )
                for index, marker_row in enumerate(partial_rows):
                    backend.execute(
                        "INSERT INTO Media "
                        "(uuid, title, type, content_hash, last_modified, client_id, "
                        "system_operation_id, system_operation_kind, "
                        "system_source_identity, system_content_hash) VALUES "
                        "(%s, %s, %s, %s, CURRENT_TIMESTAMP, %s, %s, %s, %s, %s)",
                        (
                            marker_row[0],
                            f"weak v25 partial markers {index}",
                            "text",
                            f"partial-content-{index}",
                            "901",
                            *marker_row[1:],
                        ),
                        connection=connection,
                    )
                backend.execute(
                    "UPDATE schema_version SET version = 25",
                    connection=connection,
                )

            db._initialize_schema()

            with db.transaction() as connection:
                rows = backend.execute(
                    "SELECT system_operation_id, system_operation_kind, "
                    "system_source_identity, system_content_hash "
                    "FROM Media WHERE uuid = ANY(%s)",
                    ([row[0] for row in partial_rows],),
                    connection=connection,
                ).rows
                version = int(
                    backend.execute(
                        "SELECT version FROM schema_version LIMIT 1",
                        connection=connection,
                    ).scalar
                )
                constraint_validated = backend.execute(
                    "SELECT convalidated FROM pg_constraint "
                    "WHERE conname = 'ck_media_system_operation_ownership' "
                    "AND conrelid = 'media'::regclass",
                    connection=connection,
                ).rows[0]["convalidated"]

            assert len(rows) == len(partial_rows)
            assert all(
                row
                == {
                    "system_operation_id": None,
                    "system_operation_kind": None,
                    "system_source_identity": None,
                    "system_content_hash": None,
                }
                for row in rows
            )
            assert version == 26
            assert constraint_validated is True
    finally:
        db.close_connection()
        backend.get_pool().close_all()
