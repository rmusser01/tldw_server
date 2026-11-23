from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.chacha import sqlite_schema


pytestmark = pytest.mark.unit


def _build_legacy_v31_db_with_stale_conversations_fts(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE db_schema_version(
              schema_name TEXT PRIMARY KEY NOT NULL,
              version INTEGER NOT NULL
            );

            INSERT INTO db_schema_version(schema_name, version)
            VALUES ('rag_char_chat_schema', 31);

            CREATE TABLE conversations(
              id TEXT PRIMARY KEY,
              root_id TEXT NOT NULL,
              character_id INTEGER,
              title TEXT,
              deleted BOOLEAN NOT NULL DEFAULT 0,
              client_id TEXT NOT NULL,
              version INTEGER NOT NULL DEFAULT 1
            );

            CREATE VIRTUAL TABLE conversations_fts
            USING fts5(title, content='conversations', content_rowid='rowid');

            CREATE TRIGGER conversations_au
            AFTER UPDATE ON conversations BEGIN
              INSERT INTO conversations_fts(conversations_fts,rowid,title)
              VALUES('delete',old.rowid,old.title);

              INSERT INTO conversations_fts(rowid,title)
              SELECT new.rowid,new.title
              WHERE new.deleted = 0 AND new.title IS NOT NULL;
            END;

            CREATE TRIGGER conversations_ad
            AFTER DELETE ON conversations BEGIN
              INSERT INTO conversations_fts(conversations_fts,rowid,title)
              VALUES('delete',old.rowid,old.title);
            END;
            """
        )

        conn.execute(
            """
            INSERT INTO conversations(id, root_id, character_id, title, deleted, client_id, version)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("conv-1", "conv-1", 7, "Legacy chat", 0, "user-1", 1),
        )

        conn.execute(
            """
            CREATE TRIGGER conversations_ai
            AFTER INSERT ON conversations BEGIN
              INSERT INTO conversations_fts(rowid,title)
              SELECT new.rowid,new.title
              WHERE new.deleted = 0 AND new.title IS NOT NULL;
            END;
            """
        )

        conn.commit()
    finally:
        conn.close()


def _make_db_for_migration(db_path: Path) -> CharactersRAGDB:
    db = CharactersRAGDB.__new__(CharactersRAGDB)
    db.db_path = db_path
    db.db_path_str = str(db_path)
    return db


def _add_minimal_recent_persona_tables(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE persona_profiles(
              id TEXT PRIMARY KEY,
              user_id TEXT NOT NULL,
              name TEXT NOT NULL,
              character_card_id INTEGER,
              mode TEXT NOT NULL DEFAULT 'session_scoped',
              system_prompt TEXT,
              is_active BOOLEAN NOT NULL DEFAULT 1,
              created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
              last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
              deleted BOOLEAN NOT NULL DEFAULT 0,
              version INTEGER NOT NULL DEFAULT 1
            );

            CREATE TABLE persona_sessions(
              id TEXT PRIMARY KEY,
              persona_id TEXT NOT NULL,
              user_id TEXT NOT NULL,
              conversation_id TEXT,
              mode TEXT NOT NULL DEFAULT 'session_scoped',
              reuse_allowed BOOLEAN NOT NULL DEFAULT 0,
              status TEXT NOT NULL DEFAULT 'active',
              scope_snapshot_json TEXT NOT NULL DEFAULT '{}',
              created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
              last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
              deleted BOOLEAN NOT NULL DEFAULT 0,
              version INTEGER NOT NULL DEFAULT 1
            );
            """
        )
        conn.commit()
    finally:
        conn.close()


def test_v31_to_v32_migration_succeeds_on_legacy_sqlite_db_with_stale_conversations_fts(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "legacy-v31-chacha.db"
    _build_legacy_v31_db_with_stale_conversations_fts(db_path)

    db = _make_db_for_migration(db_path)

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row

        db._migrate_from_v31_to_v32(conn)

        assistant_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info('conversations')")
        }
        assert {"assistant_kind", "assistant_id", "persona_memory_mode"}.issubset(
            assistant_columns
        )

        conversation = conn.execute(
            """
            SELECT assistant_kind, assistant_id
            FROM conversations
            WHERE id = ?
            """,
            ("conv-1",),
        ).fetchone()
        assert conversation is not None
        assert conversation["assistant_kind"] == "character"
        assert conversation["assistant_id"] == "7"

        version = conn.execute(
            """
            SELECT version
            FROM db_schema_version
            WHERE schema_name = ?
            """,
            ("rag_char_chat_schema",),
        ).fetchone()
        assert version is not None
        assert version["version"] == 32


def test_recent_persona_schema_backfill_succeeds_on_legacy_sqlite_db_with_stale_conversations_fts(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "legacy-recent-persona-chacha.db"
    _build_legacy_v31_db_with_stale_conversations_fts(db_path)
    _add_minimal_recent_persona_tables(db_path)

    db = _make_db_for_migration(db_path)

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row

        sqlite_schema._ensure_recent_persona_schema_sqlite(db.ctx, conn)

        profile_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info('persona_profiles')")
        }
        assert {
            "origin_character_id",
            "origin_character_name",
            "origin_character_snapshot_at",
        }.issubset(profile_columns)

        conversation_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info('conversations')")
        }
        assert {"assistant_kind", "assistant_id", "persona_memory_mode"}.issubset(
            conversation_columns
        )

        session_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info('persona_sessions')")
        }
        assert {"activity_surface", "preferences_json"}.issubset(session_columns)

        conversation = conn.execute(
            """
            SELECT assistant_kind, assistant_id
            FROM conversations
            WHERE id = ?
            """,
            ("conv-1",),
        ).fetchone()
        assert conversation is not None
        assert conversation["assistant_kind"] == "character"
        assert conversation["assistant_id"] == "7"
