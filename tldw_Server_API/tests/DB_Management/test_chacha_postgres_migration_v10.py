import re

import pytest

from tldw_Server_API.app.core.DB_Management.chacha import postgres_schema
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

pytestmark = pytest.mark.unit


def test_postgres_statement_conversion_includes_chat_metadata_and_backlinks(tmp_path):

    db = CharactersRAGDB(db_path=str(tmp_path / "dummy.db"), client_id="test")

    sql = db._MIGRATION_SQL_V9_TO_V10
    assert isinstance(sql, str) and "state" in sql and "conversation_id" in sql

    stmts = postgres_schema._convert_sqlite_schema_to_postgres_statements(db.ctx, sql)
    assert isinstance(stmts, list) and len(stmts) > 0

    full = "\n".join(stmts)
    assert "ALTER TABLE conversations ADD COLUMN state TEXT NOT NULL" in full
    assert "CHECK(state IN ('in-progress','resolved','backlog','non-viable'))" in full
    assert "ALTER TABLE conversations ADD COLUMN topic_label" in full
    assert "ALTER TABLE notes ADD COLUMN conversation_id" in full
    assert "idx_conversations_state" in full
    assert "idx_notes_conversation" in full
    assert re.search(r"SET\s+version\s*=\s*10", full, flags=re.IGNORECASE)
