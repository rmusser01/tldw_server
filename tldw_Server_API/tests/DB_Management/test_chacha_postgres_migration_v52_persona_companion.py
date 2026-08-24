"""Unit tests for the ChaChaNotes PostgreSQL v62 companion migration."""

import re

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


pytestmark = pytest.mark.unit


def test_postgres_v62_migration_declares_companion_constraints(tmp_path) -> None:
    """PostgreSQL v62 DDL keeps the SQLite companion constraints."""
    db = CharactersRAGDB(tmp_path / "persona_companion_v62_postgres.sqlite", "persona-companion-v62-postgres")
    try:
        statements = db._convert_sqlite_schema_to_postgres_statements(
            db._MIGRATION_SQL_V61_TO_V62_POSTGRES
        )
    finally:
        db.close_connection()

    sql = "\n".join(statements)
    assert "ALTER TABLE persona_visual_packs ADD COLUMN IF NOT EXISTS companion_behavior_json TEXT" in sql
    assert "CREATE TABLE IF NOT EXISTS persona_buddy_preferences" in sql
    assert "ambient_mode IN ('off', 'expressive', 'roaming')" in sql
    assert "CREATE TABLE IF NOT EXISTS persona_visual_pack_reviews" in sql
    assert "UNIQUE(pack_id, fingerprint)" in sql
    assert re.search(r"SET\s+version\s*=\s*62", sql, flags=re.IGNORECASE)
