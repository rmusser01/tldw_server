"""Regression tests for the ChaChaNotes SQLite v62 companion migration."""

import sqlite3
from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


pytestmark = pytest.mark.unit


def test_v62_adds_companion_tables_and_pack_column(tmp_path: Path) -> None:
    """Migrating a v61 database adds companion persistence artifacts."""
    migrated = CharactersRAGDB(tmp_path / "current.sqlite", "persona-companion-v62-migration")
    try:
        with sqlite3.connect(tmp_path / "persona_companion_v62.sqlite") as conn:
            conn.row_factory = sqlite3.Row
            conn.execute(
                "CREATE TABLE db_schema_version (schema_name TEXT PRIMARY KEY, version INTEGER NOT NULL)"
            )
            conn.execute(
                "INSERT INTO db_schema_version (schema_name, version) VALUES (?, 61)",
                (migrated._SCHEMA_NAME,),
            )
            conn.execute("CREATE TABLE persona_visual_packs (id TEXT PRIMARY KEY)")

            migrated._migrate_from_v61_to_v62_sqlite(conn)

            pack_columns = migrated._sqlite_column_names(conn, "persona_visual_packs")
            tables = migrated._sqlite_table_names(conn)

            assert "companion_behavior_json" in pack_columns
            assert "persona_buddy_preferences" in tables
            assert "persona_visual_pack_reviews" in tables
            assert migrated._get_db_version(conn) == 62
    finally:
        migrated.close_connection()


def test_v62_preference_mode_constraint_rejects_unknown_mode(tmp_path: Path) -> None:
    """The database rejects ambient modes outside the wire contract."""
    db = CharactersRAGDB(tmp_path / "persona_companion_constraint.sqlite", "persona-companion-constraint")
    try:
        with pytest.raises(Exception):
            db.get_connection().execute(
                "INSERT INTO persona_buddy_preferences "
                "(user_id, ambient_mode, version, created_at, updated_at) VALUES (?, ?, 1, ?, ?)",
                ("user-1", "chaotic", "2026-08-23T00:00:00Z", "2026-08-23T00:00:00Z"),
            )
    finally:
        db.close_connection()
