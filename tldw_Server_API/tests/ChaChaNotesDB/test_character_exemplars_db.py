from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


@pytest.fixture
def db_path(tmp_path) -> Path:
    return tmp_path / "character_exemplars_test.sqlite"


@pytest.fixture
def db_instance(db_path):
    db = CharactersRAGDB(db_path, "test_character_exemplars_client")
    yield db
    db.close_connection()


def _create_character(db: CharactersRAGDB, name: str = "Exemplar Test Character") -> int:
    char_id = db.add_character_card(
        {
            "name": name,
            "description": "Character used for exemplar tests",
            "personality": "Calm and concise",
            "first_message": "Hello",
        }
    )
    assert isinstance(char_id, int)
    return char_id


def test_migration_v21_to_v22_creates_exemplar_tables(db_path: Path, monkeypatch):
    def initialize_historical_v21(db):
        with db.transaction() as conn:
            db._apply_schema_v4(conn)
            steps = db._sqlite_linear_migration_steps()
            for version in range(4, 21):
                steps[version](conn)
                assert db._get_db_version(conn) == version + 1

    with monkeypatch.context() as patch:
        patch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 21)
        patch.setattr(CharactersRAGDB, "_initialize_schema", initialize_historical_v21)
        seeded = CharactersRAGDB(db_path, "migration_seed_client")
        with seeded.transaction() as conn:
            assert seeded._get_db_version(conn) == 21
            tables = {row["name"] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            assert {"character_exemplars", "character_exemplars_fts", "note_attachments"}.isdisjoint(tables)
            conn.execute(
                "INSERT INTO character_cards (name, description, client_id) VALUES (?, ?, ?)",
                ("Retained v21 character", "Historical content", "migration_seed_client"),
            )
            before = dict(conn.execute("SELECT * FROM character_cards WHERE name = ?", ("Retained v21 character",)).fetchone())

    # Verify the exact historical step before the ordinary initializer reaches head.
    with seeded.transaction() as conn:
        seeded._migrate_from_v21_to_v22(conn)
        assert seeded._get_db_version(conn) == 22
        tables = {row["name"] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"character_exemplars", "character_exemplars_fts"}.issubset(tables)
        assert dict(conn.execute("SELECT * FROM character_cards WHERE id = ?", (before["id"],)).fetchone()) == before
    seeded.close_connection()

    migrated_db = CharactersRAGDB(db_path, "migration_check_client")
    try:
        conn = migrated_db.get_connection()
        assert migrated_db._get_db_version(conn) == CharactersRAGDB._CURRENT_SCHEMA_VERSION
        tables = {row["name"] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"character_exemplars", "character_exemplars_fts"}.issubset(tables)
        after = dict(conn.execute("SELECT * FROM character_cards WHERE id = ?", (before["id"],)).fetchone())
        assert all(after[key] == value for key, value in before.items())
    finally:
        migrated_db.close_all_connections()
        seeded.close_all_connections()


def test_character_exemplar_crud_and_search(db_instance: CharactersRAGDB):
    char_id = _create_character(db_instance, name="CRUD Exemplar Character")

    created = db_instance.add_character_exemplar(
        char_id,
        {
            "text": "Opening statement from the character with a clear point.",
            "source_type": "article",
            "emotion": "neutral",
            "scenario": "debate",
            "rhetorical": ["opener", "emphasis"],
            "safety_allowed": ["general"],
            "safety_blocked": ["harmful"],
            "novelty_hint": "unknown",
        },
    )

    exemplar_id = created["id"]
    assert created["character_id"] == char_id
    assert created["text"].startswith("Opening statement")

    fetched = db_instance.get_character_exemplar_by_id(char_id, exemplar_id)
    assert fetched is not None
    assert fetched["id"] == exemplar_id

    listed = db_instance.list_character_exemplars(char_id, limit=10, offset=0)
    assert any(item["id"] == exemplar_id for item in listed)

    updated = db_instance.update_character_exemplar(
        char_id,
        exemplar_id,
        {
            "text": "Updated opener line tailored for a debate response.",
            "emotion": "happy",
            "rhetorical": ["opener", "ender"],
            "length_tokens": 12,
        },
    )

    assert updated is not None
    assert updated["emotion"] == "happy"
    assert "ender" in (updated.get("rhetorical") or [])
    assert int(updated["length_tokens"]) == 12

    search_results, search_total = db_instance.search_character_exemplars(
        char_id,
        query="debate response",
        limit=10,
        offset=0,
    )
    assert search_total >= 1
    assert any(item["id"] == exemplar_id for item in search_results)

    rhetorical_results, rhetorical_total = db_instance.search_character_exemplars(
        char_id,
        rhetorical=["opener"],
        limit=10,
        offset=0,
    )
    assert rhetorical_total >= 1
    assert any(item["id"] == exemplar_id for item in rhetorical_results)

    deleted = db_instance.soft_delete_character_exemplar(char_id, exemplar_id)
    assert deleted is True
    assert db_instance.get_character_exemplar_by_id(char_id, exemplar_id) is None


def test_character_exemplar_filters_by_scenario_and_emotion(db_instance: CharactersRAGDB):
    char_id = _create_character(db_instance, name="Filter Exemplar Character")

    db_instance.add_character_exemplar(
        char_id,
        {
            "text": "Boardroom response with pragmatic tone.",
            "scenario": "boardroom",
            "emotion": "neutral",
            "rhetorical": ["emphasis"],
        },
    )
    db_instance.add_character_exemplar(
        char_id,
        {
            "text": "Fan banter line with upbeat energy.",
            "scenario": "fan_banter",
            "emotion": "happy",
            "rhetorical": ["opener"],
        },
    )

    boardroom_results, boardroom_total = db_instance.search_character_exemplars(
        char_id,
        scenario="boardroom",
        emotion="neutral",
        limit=10,
        offset=0,
    )

    assert boardroom_total == 1
    assert len(boardroom_results) == 1
    assert boardroom_results[0]["scenario"] == "boardroom"
    assert boardroom_results[0]["emotion"] == "neutral"
