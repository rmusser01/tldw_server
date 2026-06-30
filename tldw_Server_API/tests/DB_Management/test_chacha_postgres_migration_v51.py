"""Unit tests for ChaChaNotes PostgreSQL v51 migration conversion."""

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


pytestmark = pytest.mark.unit


def test_postgres_v51_migration_keeps_dollar_quoted_function_intact(tmp_path):
    db = CharactersRAGDB(db_path=str(tmp_path / "dummy.db"), client_id="test")

    stmts = db._convert_sqlite_schema_to_postgres_statements(
        db._MIGRATION_SQL_V50_TO_V51_POSTGRES
    )

    function_statements = [
        stmt
        for stmt in stmts
        if stmt.startswith("CREATE OR REPLACE FUNCTION manuscript_annotations_sync_log_fn")
    ]
    assert len(function_statements) == 1
    function_stmt = function_statements[0]
    assert function_stmt.count("$$") == 2
    assert "RETURN NEW;" in function_stmt
    assert "LANGUAGE plpgsql" in function_stmt

    trigger_statements = [
        stmt
        for stmt in stmts
        if stmt.startswith("CREATE TRIGGER manuscript_annotations_sync_log")
    ]
    assert len(trigger_statements) == 1
    assert "EXECUTE FUNCTION manuscript_annotations_sync_log_fn()" in trigger_statements[0]
