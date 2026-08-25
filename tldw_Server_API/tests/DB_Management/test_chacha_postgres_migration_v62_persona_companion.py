"""Tests for the ChaChaNotes PostgreSQL v62 companion migration."""

import re

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.backends.base import (
    BackendType,
    DatabaseConfig,
    DatabaseError as BackendDatabaseError,
)
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory


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


@pytest.mark.integration
def test_postgres_v62_migrates_v61_and_enforces_companion_constraints(
    pg_database_config: DatabaseConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The live PostgreSQL migration adds and enforces the v62 companion contract."""
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db: CharactersRAGDB | None = None
    try:
        with monkeypatch.context() as version_patch:
            version_patch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 61)
            version_patch.setattr(CharactersRAGDB, "_POSTGRES_SCHEMA_VERSION", 61)
            db = CharactersRAGDB(db_path=":memory:", client_id="persona-companion-v61-postgres", backend=backend)
            with backend.transaction() as conn:
                assert db._get_schema_version_postgres(conn) == 61

        db._initialize_schema_postgres()
        with backend.transaction() as conn:
            assert db._get_schema_version_postgres(conn) == 62

        timestamp = "2026-08-23T00:00:00Z"
        preference_insert = (
            "INSERT INTO persona_buddy_preferences "
            "(user_id, ambient_mode, version, created_at, updated_at) VALUES (%s, %s, 1, %s, %s)"
        )
        with pytest.raises(BackendDatabaseError):
            with backend.transaction() as conn:
                backend.execute(
                    preference_insert,
                    ("user-1", "chaotic", timestamp, timestamp),
                    connection=conn,
                )
        with backend.transaction() as conn:
            backend.execute(
                preference_insert,
                ("user-1", "off", timestamp, timestamp),
                connection=conn,
            )
        with pytest.raises(BackendDatabaseError):
            with backend.transaction() as conn:
                backend.execute(
                    preference_insert,
                    ("user-1", "expressive", timestamp, timestamp),
                    connection=conn,
                )

        persona_id = db.create_persona_profile({"user_id": "user-1", "name": "Migrated PostgreSQL Persona"})
        pack = db.create_persona_visual_pack(
            persona_id=persona_id,
            user_id="user-1",
            title="Migrated PostgreSQL Pack",
            manifest={"manifest_version": 1, "renderer_type": "sprite_frames"},
        )
        review_insert = (
            "INSERT INTO persona_visual_pack_reviews "
            "(id, pack_id, user_id, reviewer_user_id, fingerprint, pack_version, reviewed_at, created_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
        )
        review_params = (
            "review-1",
            pack["id"],
            "user-1",
            "reviewer-1",
            "a" * 64,
            1,
            timestamp,
            timestamp,
        )
        with backend.transaction() as conn:
            backend.execute(review_insert, review_params, connection=conn)
        with pytest.raises(BackendDatabaseError):
            with backend.transaction() as conn:
                backend.execute(review_insert, ("review-2", *review_params[1:]), connection=conn)
    finally:
        if db is not None:
            db.close_connection()
        if backend.backend_type == BackendType.POSTGRESQL:
            backend.get_pool().close_all()
