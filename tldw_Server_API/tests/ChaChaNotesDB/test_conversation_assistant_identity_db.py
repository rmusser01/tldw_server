from collections.abc import Iterator
from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.tests.Characters.test_character_functionality_db import sample_card_data

pytestmark = pytest.mark.unit


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "conversation_assistant_identity.sqlite"


@pytest.fixture
def db_instance(db_path: Path) -> Iterator[CharactersRAGDB]:
    db = CharactersRAGDB(db_path, "assistant-identity-test-client")
    yield db
    db.close_connection()


@pytest.fixture
def character_id(db_instance: CharactersRAGDB) -> int:
    card_id = db_instance.add_character_card(sample_card_data(name="Assistant Identity Source"))
    assert card_id is not None
    return card_id


def test_legacy_character_conversation_backfills_assistant_identity(
    db_instance: CharactersRAGDB,
    character_id: int,
) -> None:
    conv_id = db_instance.add_conversation(
        {
            "id": "conv-character-1",
            "character_id": character_id,
            "title": "Legacy chat",
            "root_id": "conv-character-1",
            "client_id": db_instance.client_id,
        }
    )

    row = db_instance.get_conversation_by_id(conv_id)
    assert row is not None
    assert row["assistant_kind"] == "character"
    assert row["assistant_id"] == str(character_id)
    assert row["persona_memory_mode"] is None

    conversations = db_instance.get_conversations_for_user(db_instance.client_id)
    assert len(conversations) == 1
    assert conversations[0]["assistant_kind"] == "character"
    assert conversations[0]["assistant_id"] == str(character_id)


def test_persona_conversation_round_trips_assistant_identity(db_instance: CharactersRAGDB) -> None:
    conv_id = db_instance.add_conversation(
        {
            "id": "conv-persona-1",
            "assistant_kind": "persona",
            "assistant_id": "garden-helper",
            "persona_memory_mode": "read_only",
            "title": "Persona chat",
            "root_id": "conv-persona-1",
            "client_id": db_instance.client_id,
        }
    )

    row = db_instance.get_conversation_by_id(conv_id)
    assert row is not None
    assert row["character_id"] is None
    assert row["assistant_kind"] == "persona"
    assert row["assistant_id"] == "garden-helper"
    assert row["persona_memory_mode"] == "read_only"

    assert db_instance.update_conversation(
        conv_id,
        {"persona_memory_mode": "read_write"},
        expected_version=row["version"],
    )

    updated = db_instance.get_conversation_by_id(conv_id)
    assert updated is not None
    assert updated["assistant_kind"] == "persona"
    assert updated["assistant_id"] == "garden-helper"
    assert updated["persona_memory_mode"] == "read_write"


def test_migration_v31_to_v32_backfills_assistant_identity_for_legacy_rows(
    db_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def initialize_historical_v31(db: CharactersRAGDB) -> None:
        with db.transaction() as conn:
            db._apply_schema_v4(conn)
            steps = db._sqlite_linear_migration_steps()
            for version in range(4, 31):
                steps[version](conn)
                assert db._get_db_version(conn) == version + 1

    # Construct the historical schema without current compatibility repairs.
    with monkeypatch.context() as patch:
        patch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 31)
        patch.setattr(CharactersRAGDB, "_initialize_schema", initialize_historical_v31)
        seed = CharactersRAGDB(db_path, "assistant-identity-test-client")
    conv_id = "conv-migration-1"
    try:
        with seed.transaction() as conn:
            assert seed._get_db_version(conn) == 31
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(conversations)")}
            assert {"assistant_kind", "assistant_id", "persona_memory_mode"}.isdisjoint(columns)
            assert conn.execute(
                "SELECT 1 FROM sqlite_master WHERE name = 'note_attachments'"
            ).fetchone() is None
            character_id = conn.execute(
                "INSERT INTO character_cards (name, description, client_id) VALUES (?, ?, ?)",
                ("Migration Source", "Retained character content", seed.client_id),
            ).lastrowid
            conn.execute(
                "INSERT INTO conversations (id, root_id, character_id, title, client_id) "
                "VALUES (?, ?, ?, ?, ?)",
                (conv_id, conv_id, character_id, "Legacy migration chat", seed.client_id),
            )
            before = dict(conn.execute("SELECT * FROM conversations WHERE id = ?", (conv_id,)).fetchone())

            seed._migrate_from_v31_to_v32(conn)
            assert seed._get_db_version(conn) == 32
            after = dict(conn.execute("SELECT * FROM conversations WHERE id = ?", (conv_id,)).fetchone())
            assert after == {
                **before,
                "assistant_kind": "character",
                "assistant_id": str(character_id),
                "persona_memory_mode": None,
            }
    finally:
        seed.close_all_connections()

    migrated = CharactersRAGDB(db_path, "assistant-identity-test-client")
    try:
        conn = migrated.get_connection()
        assert migrated._get_db_version(conn) == CharactersRAGDB._CURRENT_SCHEMA_VERSION
        current_row = dict(conn.execute("SELECT * FROM conversations WHERE id = ?", (conv_id,)).fetchone())
        assert all(current_row[key] == value for key, value in after.items())
        migrated_row = migrated.get_conversation_by_id(conv_id)
        assert migrated_row is not None
        assert migrated_row["assistant_kind"] == "character"
        assert migrated_row["assistant_id"] == str(character_id)
        assert migrated_row["persona_memory_mode"] is None
    finally:
        migrated.close_all_connections()
