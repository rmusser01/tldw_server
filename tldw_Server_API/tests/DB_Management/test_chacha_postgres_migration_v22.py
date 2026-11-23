import re

import pytest

from tldw_Server_API.app.core.DB_Management.chacha import postgres_schema
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

pytestmark = pytest.mark.unit


def test_postgres_statement_conversion_skips_sqlite_fts_rebuild_statement(tmp_path):
    db = CharactersRAGDB(db_path=str(tmp_path / "dummy.db"), client_id="test")

    sql = db._MIGRATION_SQL_V21_TO_V22
    assert isinstance(sql, str) and "character_exemplars_fts" in sql

    stmts = postgres_schema._convert_sqlite_schema_to_postgres_statements(db.ctx, sql)
    assert isinstance(stmts, list) and len(stmts) > 0

    full = "\n".join(stmts)
    assert "CREATE TABLE IF NOT EXISTS character_exemplars" in full
    assert "character_exemplars_fts(character_exemplars_fts) VALUES('rebuild')" not in full
    assert re.search(r"SET\s+version\s*=\s*22", full, flags=re.IGNORECASE)
