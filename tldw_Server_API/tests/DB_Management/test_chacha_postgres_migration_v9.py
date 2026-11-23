import re

import pytest

from tldw_Server_API.app.core.DB_Management.chacha import postgres_schema
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

pytestmark = pytest.mark.unit


def test_postgres_statement_conversion_contains_note_edges(tmp_path):


     # Create a dummy DB to instantiate class and access converter
    db = CharactersRAGDB(db_path=str(tmp_path / "dummy.db"), client_id="test")

    sql = getattr(db, "_MIGRATION_SQL_V8_TO_V9")
    assert isinstance(sql, str) and "note_edges" in sql

    stmts = postgres_schema._convert_sqlite_schema_to_postgres_statements(db.ctx, sql)
    assert isinstance(stmts, list) and len(stmts) > 0

    full = "\n".join(stmts)
    # Ensure table creation and NOT NULL created_by make it through conversion
    assert re.search(r"CREATE\s+TABLE\s+IF\s+NOT\s+EXISTS\s+note_edges", full, flags=re.IGNORECASE)
    assert re.search(r"created_by\s+TEXT\s+NOT\s+NULL", full, flags=re.IGNORECASE)
    # Ensure the undirected unique index appears
    assert "uniq_note_edges_undirected" in full
