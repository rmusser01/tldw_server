import sqlite3

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

pytestmark = pytest.mark.unit


def _bootstrap_v9_sqlite_db(db_path, monkeypatch: pytest.MonkeyPatch) -> CharactersRAGDB:
    """Build the complete v9 catalog, including the v8-to-v9 note_edges table."""
    def initialize_historical(db: CharactersRAGDB) -> None:
        with db.transaction() as conn:
            db._apply_schema_v4(conn)
            steps = db._sqlite_linear_migration_steps()
            for version in range(4, 9):
                steps[version](conn)
                assert db._get_db_version(conn) == version + 1

    with monkeypatch.context() as patch:
        patch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 9)
        patch.setattr(CharactersRAGDB, "_initialize_schema", initialize_historical)
        return CharactersRAGDB(db_path, client_id="user-1")


def test_sqlite_migration_v9_to_v10_backfills_and_indexes(tmp_path, monkeypatch):
    db_path = tmp_path / "chacha_v9.db"
    note_id = "11111111-1111-4111-8111-111111111111"
    seed = _bootstrap_v9_sqlite_db(db_path, monkeypatch)
    try:
        with seed.transaction() as conn:
            assert seed._get_db_version(conn) == 9
            tables = seed._sqlite_table_names(conn)
            assert "note_edges" in tables
            assert {"note_attachments", "note_tasks"}.isdisjoint(tables)
            assert "state" not in {row["name"] for row in conn.execute("PRAGMA table_info(conversations)")}
            # Retain the legacy FTS repair scenario without omitting unrelated v9 tables.
            conn.executescript("""
                DROP TRIGGER character_cards_ai;
                DROP TRIGGER character_cards_au;
                DROP TRIGGER character_cards_ad;
                DROP TRIGGER character_cards_sync_create;
                DROP TRIGGER character_cards_sync_update;
                DROP TRIGGER character_cards_sync_delete;
                DROP TRIGGER character_cards_sync_undelete;
                ALTER TABLE character_cards DROP COLUMN description;
                ALTER TABLE character_cards DROP COLUMN personality;
                ALTER TABLE character_cards DROP COLUMN scenario;
                ALTER TABLE character_cards DROP COLUMN system_prompt;
            """)
            character_columns = {row["name"] for row in conn.execute("PRAGMA table_info(character_cards)")}
            assert {"description", "personality", "scenario", "system_prompt"}.isdisjoint(character_columns)
            character_before = dict(conn.execute("SELECT * FROM character_cards WHERE id = 1").fetchone())
            conn.execute(
                "INSERT INTO conversations (id, root_id, character_id, title, rating, client_id) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                ("conv-1", "conv-1", 1, "Legacy Chat", 5, "user-1"),
            )
            conn.execute(
                "INSERT INTO notes (id, title, content, client_id) VALUES (?, ?, ?, ?)",
                (note_id, "Legacy Note", "content", "user-1"),
            )
            conversation_before = dict(conn.execute("SELECT * FROM conversations WHERE id = 'conv-1'").fetchone())
            note_before = dict(conn.execute("SELECT * FROM notes WHERE id = ?", (note_id,)).fetchone())
            seed._migrate_from_v9_to_v10(conn)
            assert seed._get_db_version(conn) == 10
            assert dict(conn.execute("SELECT * FROM conversations WHERE id = 'conv-1'").fetchone()) == {
                **conversation_before, "state": "in-progress", "topic_label": None,
                "cluster_id": None, "source": None, "external_ref": None,
            }
            assert dict(conn.execute("SELECT * FROM notes WHERE id = ?", (note_id,)).fetchone()) == {
                **note_before, "conversation_id": None, "message_id": None,
            }
    finally:
        seed.close_all_connections()

    # Trigger migration to current schema version
    db = CharactersRAGDB(db_path=str(db_path), client_id="user-1")
    db.close_all_connections()

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        version = conn.execute(
            "SELECT version FROM db_schema_version WHERE schema_name = ?", ("rag_char_chat_schema",)
        ).fetchone()[0]
        assert version == CharactersRAGDB._CURRENT_SCHEMA_VERSION

        # Character card FTS dependencies should be present for legacy minimal schemas.
        character_cols = [row[1] for row in conn.execute("PRAGMA table_info('character_cards')").fetchall()]
        assert {"description", "personality", "scenario", "system_prompt"}.issubset(set(character_cols))

        # Conversations should have state backfilled and new columns/indexes
        conv_cols = [row[1] for row in conn.execute("PRAGMA table_info('conversations')").fetchall()]
        assert {
            "state",
            "topic_label",
            "cluster_id",
            "source",
            "external_ref",
            "topic_label_source",
            "topic_last_tagged_at",
            "topic_last_tagged_message_id",
        }.issubset(set(conv_cols))
        state_value = conn.execute("SELECT state FROM conversations WHERE id = 'conv-1'").fetchone()[0]
        assert state_value == "in-progress"

        conv_indexes = {row[1] for row in conn.execute("PRAGMA index_list('conversations')").fetchall()}
        assert {
            "idx_conversations_state",
            "idx_conversations_cluster",
            "idx_conversations_last_modified",
            "idx_conversations_topic_label",
            "idx_conversations_source_external_ref",
        }.issubset(conv_indexes)

        # Notes should have backlink columns
        note_cols = [row[1] for row in conn.execute("PRAGMA table_info('notes')").fetchall()]
        assert {"conversation_id", "message_id"}.issubset(set(note_cols))

        note_indexes = {row[1] for row in conn.execute("PRAGMA index_list('notes')").fetchall()}
        assert {"idx_notes_conversation", "idx_notes_message"}.issubset(note_indexes)

        # Flashcards should have backlink columns
        flash_cols = [row[1] for row in conn.execute("PRAGMA table_info('flashcards')").fetchall()]
        assert {"conversation_id", "message_id"}.issubset(set(flash_cols))

        flash_indexes = {row[1] for row in conn.execute("PRAGMA index_list('flashcards')").fetchall()}
        assert {"idx_flashcards_conversation", "idx_flashcards_message"}.issubset(flash_indexes)

        # Conversation clusters table exists
        cluster_table = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'conversation_clusters'"
        ).fetchone()
        assert cluster_table is not None

        character_after = dict(conn.execute("SELECT * FROM character_cards WHERE id = 1").fetchone())
        conversation_after = dict(conn.execute("SELECT * FROM conversations WHERE id = 'conv-1'").fetchone())
        note_after = dict(conn.execute("SELECT * FROM notes WHERE id = ?", (note_id,)).fetchone())
        assert all(character_after[key] == value for key, value in character_before.items())
        assert all(conversation_after[key] == value for key, value in conversation_before.items())
        assert all(note_after[key] == value for key, value in note_before.items())
        assert conn.execute(
            "SELECT rowid FROM character_cards_fts WHERE character_cards_fts MATCH ?", ("Default",),
        ).fetchone()[0] == 1
