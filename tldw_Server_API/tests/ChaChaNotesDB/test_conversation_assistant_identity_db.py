"""Conversation identity normalization and genuine historical schema upgrade contracts."""

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


def _initialize_genuine_v31(db: CharactersRAGDB) -> None:
    """Build only registered historical migrations, never downgrade a modern marker."""
    conn = db.get_connection()
    db._apply_schema_v4(conn)
    version = 4
    while version < 31:
        version = db._run_sqlite_linear_migration_step(conn, from_version=version, target_version=31, initial_version=4)
    conn.commit()


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
    """Registered upgrades retain a real pre-identity row and leave new origin unknown."""
    with monkeypatch.context() as historical:
        historical.setattr(CharactersRAGDB, "_initialize_schema_sqlite", _initialize_genuine_v31)
        seed = CharactersRAGDB(db_path, "assistant-identity-test-client")
    character_id = seed.execute_query("SELECT id FROM character_cards ORDER BY id LIMIT 1").fetchone()[0]
    conv_id = "conv-migration-1"
    with seed.transaction() as conn:
        conn.execute(
            "INSERT INTO conversations (id, root_id, character_id, title, client_id) VALUES (?, ?, ?, ?, ?)",
            (conv_id, conv_id, character_id, "Legacy migration chat", seed.client_id),
        )
    assert "assistant_kind" not in seed.get_conversation_by_id(conv_id)
    seed.close_connection()

    migrated = CharactersRAGDB(db_path, "assistant-identity-test-client")
    conn = migrated.get_connection()
    version_row = conn.execute(
        "SELECT version FROM db_schema_version WHERE schema_name = ?",
        (CharactersRAGDB._SCHEMA_NAME,),
    ).fetchone()
    assert version_row is not None
    assert version_row["version"] == CharactersRAGDB._CURRENT_SCHEMA_VERSION

    migrated_row = migrated.get_conversation_by_id(conv_id)
    assert migrated_row is not None
    assert migrated_row["assistant_kind"] == "character"
    assert migrated_row["assistant_id"] == str(character_id)
    assert migrated_row["persona_memory_mode"] is None
    assert migrated_row["assistant_startup_json"] is None
    migrated.close_connection()
