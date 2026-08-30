"""PostgreSQL schema-v66 contracts for durable semantic cleanup authority."""

from __future__ import annotations

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseConfig
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

pytestmark = [pytest.mark.integration, pytest.mark.timeout(30)]

_TABLE = "note_semantic_obsolete_vectors"


def test_postgres_v66_ddl_is_owner_scoped_forced_rls_and_has_no_cascading_authority() -> None:
    sql = " ".join(CharactersRAGDB._MIGRATION_SQL_V65_TO_V66_POSTGRES.split())

    assert f"CREATE TABLE IF NOT EXISTS {_TABLE}" in sql
    assert f"ALTER TABLE {_TABLE} ENABLE ROW LEVEL SECURITY" in sql
    assert f"ALTER TABLE {_TABLE} FORCE ROW LEVEL SECURITY" in sql
    assert f"CREATE POLICY {_TABLE}_tenant_isolation" in sql
    assert "idx_note_semantic_obsolete_vectors_claimable" in sql
    assert "idx_note_semantic_obsolete_vectors_generation" in sql
    assert "REFERENCES notes" not in sql
    assert "REFERENCES note_semantic_generations" not in sql


def test_postgres_v66_live_schema_has_cleanup_constraints_indexes_and_forced_rls(
    pg_database_config: DatabaseConfig,
) -> None:
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(":memory:", client_id="owner-a", backend=backend)
    try:
        version = backend.execute(
            "SELECT version FROM db_schema_version WHERE schema_name = %s",
            (CharactersRAGDB._SCHEMA_NAME,),
        ).scalar
        relation = backend.execute(
            """
            SELECT relrowsecurity,relforcerowsecurity
              FROM pg_class
             WHERE oid = to_regclass(%s)
            """,
            (_TABLE,),
        ).rows[0]
        columns = {
            str(row["column_name"])
            for row in backend.execute(
                """
                SELECT column_name FROM information_schema.columns
                 WHERE table_schema=current_schema() AND table_name=%s
                """,
                (_TABLE,),
            ).rows
        }
        indexes = {
            str(row["indexname"])
            for row in backend.execute(
                "SELECT indexname FROM pg_indexes WHERE schemaname=current_schema() AND tablename=%s",
                (_TABLE,),
            ).rows
        }
        foreign_keys = backend.execute(
            """
            SELECT confrelid::regclass::text AS referenced_table
              FROM pg_constraint
             WHERE conrelid=to_regclass(%s) AND contype='f'
            """,
            (_TABLE,),
        ).rows

        assert int(version) == 66
        assert relation == {"relrowsecurity": True, "relforcerowsecurity": True}
        assert {"owner_user_id", "dataset_id", "generation_id", "vector_id"} <= columns
        assert {
            "idx_note_semantic_obsolete_vectors_claimable",
            "idx_note_semantic_obsolete_vectors_generation",
        } <= indexes
        assert foreign_keys == []
    finally:
        db.close_all_connections()
        backend.get_pool().close_all()
