from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

pytestmark = pytest.mark.unit


def test_migration_from_v35_adds_conversation_list_indexes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "conversation_list_indexes.sqlite"
    expected_indexes = {
        "idx_conversations_client_deleted_last_modified": [
            ("client_id", 0), ("deleted", 0), ("last_modified", 1),
        ],
        "idx_conversations_client_character_deleted_last_modified": [
            ("client_id", 0), ("character_id", 0), ("deleted", 0), ("last_modified", 1),
        ],
        "idx_conversations_client_deleted_created_at": [
            ("client_id", 0), ("deleted", 0), ("created_at", 1),
        ],
    }

    def initialize_historical_v35(db: CharactersRAGDB) -> None:
        with db.transaction() as conn:
            db._apply_schema_v4(conn)
            steps = db._sqlite_linear_migration_steps()
            for version in range(4, 35):
                steps[version](conn)
                assert db._get_db_version(conn) == version + 1

    # These indexes were introduced after v35, independently of later schema bumps.
    with monkeypatch.context() as patch:
        patch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 35)
        patch.setattr(CharactersRAGDB, "_initialize_schema", initialize_historical_v35)
        seed = CharactersRAGDB(db_path, "conversation-index-seed")
    try:
        with seed.transaction() as conn:
            assert seed._get_db_version(conn) == 35
            existing_indexes = {row["name"] for row in conn.execute("PRAGMA index_list('conversations')")}
            assert expected_indexes.keys().isdisjoint(existing_indexes)
            conn.execute(
                "INSERT INTO conversations (id, root_id, title, client_id) VALUES (?, ?, ?, ?)",
                ("retained-conversation", "retained-conversation", "Retained list entry", seed.client_id),
            )
            before = dict(conn.execute("SELECT * FROM conversations").fetchone())
    finally:
        seed.close_all_connections()

    migrated = CharactersRAGDB(db_path, "conversation-index-migration-check")
    try:
        conn = migrated.get_connection()
        assert migrated._get_db_version(conn) == CharactersRAGDB._CURRENT_SCHEMA_VERSION
        existing_indexes = {row["name"] for row in conn.execute("PRAGMA index_list('conversations')")}
        assert expected_indexes.keys() <= existing_indexes
        for index_name, expected_columns in expected_indexes.items():
            columns = [
                (row["name"], row["desc"])
                for row in conn.execute(f"PRAGMA index_xinfo('{index_name}')")
                if row["key"]
            ]
            assert columns == expected_columns
        after = dict(conn.execute("SELECT * FROM conversations WHERE id = ?", (before["id"],)).fetchone())
        assert all(after[key] == value for key, value in before.items())
    finally:
        migrated.close_all_connections()
