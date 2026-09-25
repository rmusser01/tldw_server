"""Regression tests for the ChaChaNotes SQLite v69 companion migration."""

import sqlite3
from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


pytestmark = pytest.mark.unit


def _seed_v68_database(db_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Initialize through the version dispatcher at the pre-v69 schema."""
    with monkeypatch.context() as version_patch:
        version_patch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 68)
        seeded = CharactersRAGDB(db_path, "persona-companion-v68-seed")
        try:
            assert seeded._get_db_version(seeded.get_connection()) == 68
        finally:
            seeded.close_connection()


def test_v69_migrates_v68_and_enforces_companion_constraints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The v68 dispatcher migrates to v69 and SQLite enforces its constraints."""
    db_path = tmp_path / "persona_companion_v69.sqlite"
    _seed_v68_database(db_path, monkeypatch)

    migrated = CharactersRAGDB(db_path, "persona-companion-v69-migration")
    try:
        conn = migrated.get_connection()
        pack_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info('persona_visual_packs')")
        }
        tables = migrated._sqlite_table_names(conn)

        assert "companion_behavior_json" in pack_columns
        assert "persona_buddy_preferences" in tables
        assert "persona_visual_pack_reviews" in tables
        assert migrated._get_db_version(conn) == 69
        workspace_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info('workspaces')")
        }
        assert "system_operation_id" in workspace_columns

        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO persona_buddy_preferences "
                "(user_id, ambient_mode, version, created_at, updated_at) VALUES (?, ?, 1, ?, ?)",
                ("user-1", "chaotic", "2026-08-23T00:00:00Z", "2026-08-23T00:00:00Z"),
            )
        conn.execute(
            "INSERT INTO persona_buddy_preferences "
            "(user_id, ambient_mode, version, created_at, updated_at) VALUES (?, ?, 1, ?, ?)",
            ("user-1", "off", "2026-08-23T00:00:00Z", "2026-08-23T00:00:00Z"),
        )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO persona_buddy_preferences "
                "(user_id, ambient_mode, version, created_at, updated_at) VALUES (?, ?, 1, ?, ?)",
                ("user-1", "expressive", "2026-08-23T00:00:00Z", "2026-08-23T00:00:00Z"),
            )

        persona_id = migrated.create_persona_profile({"user_id": "user-1", "name": "Migrated Persona"})
        pack = migrated.create_persona_visual_pack(
            persona_id=persona_id,
            user_id="user-1",
            title="Migrated Pack",
            manifest={"manifest_version": 1, "renderer_type": "sprite_frames"},
        )
        review_params = (
            "review-1",
            pack["id"],
            "user-1",
            "reviewer-1",
            "a" * 64,
            1,
            "2026-08-23T00:00:00Z",
            "2026-08-23T00:00:00Z",
        )
        conn.execute(
            "INSERT INTO persona_visual_pack_reviews "
            "(id, pack_id, user_id, reviewer_user_id, fingerprint, pack_version, reviewed_at, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            review_params,
        )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO persona_visual_pack_reviews "
                "(id, pack_id, user_id, reviewer_user_id, fingerprint, pack_version, reviewed_at, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                ("review-2", *review_params[1:]),
            )
    finally:
        migrated.close_connection()
