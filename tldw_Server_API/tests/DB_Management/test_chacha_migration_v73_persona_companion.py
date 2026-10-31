"""Verify SQLite companion upgrades remain valid through the current registry."""

import sqlite3
from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.sql_utils import split_sql_statements

pytestmark = pytest.mark.unit


def _seed_database(db_path: Path, monkeypatch: pytest.MonkeyPatch, version: int) -> None:
    """Initialize through the existing version dispatcher before companion storage."""
    with monkeypatch.context() as version_patch:
        version_patch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", version)
        seeded = CharactersRAGDB(db_path, "persona-companion-v68-seed")
        try:
            assert seeded._get_db_version(seeded.get_connection()) == version
        finally:
            seeded.close_connection()


@pytest.mark.parametrize("source_version", [68, 72])
def test_v73_migrates_prior_dev_and_enforces_companion_constraints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_version: int,
) -> None:
    """Prior catalogs reach the current version and retain companion constraints."""
    db_path = tmp_path / "persona_companion_v73.sqlite"
    _seed_database(db_path, monkeypatch, source_version)

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
        assert migrated._get_db_version(conn) == CharactersRAGDB._CURRENT_SCHEMA_VERSION
        workspace_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info('workspaces')")
        }
        assert "system_operation_id" in workspace_columns
        assert "assistant_defaults_explicit_none" in workspace_columns
        assert "native_chat_admission_closed" in workspace_columns
        assert "conversation_history_projections" in tables

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


def test_published_companion_v69_lineage_preserves_preferences_and_reopens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The old companion v69 lineage reaches the current registry without data loss."""
    db_path = tmp_path / "published_companion.sqlite"
    _seed_database(db_path, monkeypatch, 68)
    with sqlite3.connect(db_path) as conn:
        # Reconstruct the published PR catalog: companion DDL after dev v68,
        # not the newer workspace opt-out/provenance/history/native lineage.
        for statement in split_sql_statements(CharactersRAGDB._MIGRATION_SQL_V72_TO_V73_PERSONA_COMPANION):
            if not statement.lstrip().startswith("UPDATE db_schema_version"):
                conn.execute(statement)
        conn.execute("UPDATE db_schema_version SET version = 69")
        conn.execute(
            "INSERT INTO persona_buddy_preferences VALUES (?, ?, ?, ?, ?)",
            ("legacy-owner", "roaming", 4, "2026-08-23T00:00:00Z", "2026-08-23T00:00:00Z"),
        )
    for _ in range(2):
        db = CharactersRAGDB(db_path, "legacy-owner")
        try:
            conn = db.get_connection()
            assert db._get_db_version(conn) == CharactersRAGDB._CURRENT_SCHEMA_VERSION
            assert tuple(conn.execute(
                "SELECT ambient_mode, version FROM persona_buddy_preferences WHERE user_id = ?",
                ("legacy-owner",),
            ).fetchone()) == ("roaming", 4)
            assert "assistant_defaults_explicit_none" in {
                row["name"] for row in conn.execute("PRAGMA table_info(workspaces)")
            }
            assert "native_chat_operations" in db._sqlite_table_names(conn)
        finally:
            db.close_connection()


def test_companion_upgrade_failure_rolls_back_new_catalog_and_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failure after companion writes leaves the pre-upgrade dev catalog intact."""
    db_path = tmp_path / "companion_rollback.sqlite"
    _seed_database(db_path, monkeypatch, 72)
    migrate = CharactersRAGDB._migrate_from_v72_to_v73_persona_companion

    def interrupted(self: CharactersRAGDB, conn: sqlite3.Connection) -> None:
        migrate(self, conn)
        raise RuntimeError("companion migration interrupted")

    with monkeypatch.context() as patch:
        patch.setattr(CharactersRAGDB, "_migrate_from_v72_to_v73_persona_companion", interrupted)
        with pytest.raises(Exception, match="companion migration interrupted"):
            CharactersRAGDB(db_path, "rollback-owner")
    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT version FROM db_schema_version").fetchone()[0] == 72
        assert "companion_behavior_json" not in {
            row[1] for row in conn.execute("PRAGMA table_info(persona_visual_packs)")
        }
        assert conn.execute(
            "SELECT name FROM sqlite_master WHERE name = 'persona_buddy_preferences'"
        ).fetchone() is None
