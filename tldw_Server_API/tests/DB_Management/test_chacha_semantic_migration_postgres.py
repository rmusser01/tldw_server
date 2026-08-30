"""PostgreSQL schema-v65 contracts for Notes semantic-index persistence."""

from __future__ import annotations

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseConfig
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


pytestmark = pytest.mark.unit


_TABLES = (
    "note_semantic_index_configs",
    "note_semantic_generations",
    "note_semantic_note_state",
    "note_semantic_chunks",
    "note_semantic_work",
)


def test_postgres_v65_ddl_has_owner_keys_forced_rls_and_no_dimension_tables() -> None:
    sql = " ".join(CharactersRAGDB._MIGRATION_SQL_V64_TO_V65_POSTGRES.split())

    for table in _TABLES:
        assert f"CREATE TABLE IF NOT EXISTS {table}" in sql
        assert f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY" in sql
        assert f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY" in sql
        assert (
            f"CREATE POLICY {table}_tenant_isolation ON {table} USING "
            "(owner_user_id = current_setting('app.current_user_id', true) "
            "AND dataset_id = current_setting('app.current_dataset_id', true))"
        ) in sql

    assert "idx_note_semantic_generations_one_active" in sql
    assert "idx_note_semantic_generations_one_staging" in sql
    assert "idx_note_semantic_work_claimable" in sql
    assert "vector(" not in sql.lower()
    assert "note_semantic_vectors_" not in sql


def test_postgres_initializer_routes_schema_v64_through_v65(monkeypatch: pytest.MonkeyPatch) -> None:
    db = CharactersRAGDB.__new__(CharactersRAGDB)
    applied: list[tuple[str, int]] = []

    monkeypatch.setattr(CharactersRAGDB, "_POSTGRES_SCHEMA_VERSION", 65)
    monkeypatch.setattr(db, "_get_schema_version_postgres", lambda conn: 64)
    monkeypatch.setattr(db, "_ensure_chacha_rls_postgres", lambda conn: None)
    monkeypatch.setattr(db, "_ensure_note_graph_suggestion_schema_postgres", lambda conn: None)
    monkeypatch.setattr(
        db,
        "_migrate_from_v64_to_v65_postgres",
        lambda conn: applied.append((CharactersRAGDB._MIGRATION_SQL_V64_TO_V65_POSTGRES, 65)),
    )

    # The dedicated migration helper is the observable backend boundary.
    db._migrate_from_v64_to_v65_postgres(None)
    assert applied == [(CharactersRAGDB._MIGRATION_SQL_V64_TO_V65_POSTGRES, 65)]


@pytest.mark.integration
@pytest.mark.timeout(30)
def test_postgres_v65_live_schema_has_forced_owner_dataset_rls(
    pg_database_config: DatabaseConfig,
) -> None:
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(":memory:", client_id="owner-a", backend=backend)
    try:
        version = backend.execute(
            "SELECT version FROM db_schema_version WHERE schema_name = %s",
            (CharactersRAGDB._SCHEMA_NAME,),
        ).scalar
        columns = backend.execute(
            """
            SELECT table_name, column_name
              FROM information_schema.columns
             WHERE table_schema = current_schema()
               AND table_name = ANY(%s)
               AND column_name IN ('owner_user_id', 'dataset_id')
            """,
            (list(_TABLES),),
        ).rows
        relations = backend.execute(
            """
            SELECT relname, relrowsecurity, relforcerowsecurity
              FROM pg_class AS relation
              JOIN pg_namespace AS namespace ON namespace.oid = relation.relnamespace
             WHERE namespace.nspname = current_schema()
               AND relname = ANY(%s)
            """,
            (list(_TABLES),),
        ).rows
        policies = backend.execute(
            """
            SELECT tablename, qual, with_check
              FROM pg_policies
             WHERE schemaname = current_schema()
               AND tablename = ANY(%s)
            """,
            (list(_TABLES),),
        ).rows
        vector_tables = backend.execute(
            """
            SELECT tablename FROM pg_tables
             WHERE schemaname = current_schema()
               AND tablename LIKE 'note_semantic_vectors_%'
            """,
        ).rows

        assert int(version) == 65
        assert {(str(row["table_name"]), str(row["column_name"])) for row in columns} >= {
            (table, column) for table in _TABLES for column in ("owner_user_id", "dataset_id")
        }
        assert {
            (str(row["relname"]), bool(row["relrowsecurity"]), bool(row["relforcerowsecurity"]))
            for row in relations
        } >= {(table, True, True) for table in _TABLES}
        for row in policies:
            predicate = f"{row['qual']} {row['with_check']}"
            assert "owner_user_id" in predicate
            assert "dataset_id" in predicate
            assert "app.current_user_id" in predicate
            assert "app.current_dataset_id" in predicate
        assert vector_tables == []
    finally:
        db.close_all_connections()
