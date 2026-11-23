import re

import pytest

from tldw_Server_API.app.core.DB_Management.chacha import postgres_schema
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

pytestmark = pytest.mark.unit


def test_postgres_statement_conversion_includes_writing_playground_tables(tmp_path):
    db = CharactersRAGDB(db_path=str(tmp_path / "dummy.db"), client_id="test")

    sql = db._MIGRATION_SQL_V14_TO_V15
    assert isinstance(sql, str) and "writing_sessions" in sql

    stmts = postgres_schema._convert_sqlite_schema_to_postgres_statements(db.ctx, sql)
    assert isinstance(stmts, list) and len(stmts) > 0

    full = "\n".join(stmts)
    assert "CREATE TABLE IF NOT EXISTS writing_sessions" in full
    assert "CREATE TABLE IF NOT EXISTS writing_templates" in full
    assert "CREATE TABLE IF NOT EXISTS writing_themes" in full
    assert "idx_writing_sessions_last_modified" in full
    assert "idx_writing_templates_last_modified" in full
    assert "idx_writing_themes_order" in full
    assert re.search(r"SET\s+version\s*=\s*15", full, flags=re.IGNORECASE)
