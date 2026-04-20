"""Tests for the extracted CharacterStore."""

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    InputError,
    ConflictError,
)
from tldw_Server_API.app.core.DB_Management.chacha.character_store import CharacterStore


pytestmark = pytest.mark.unit


@pytest.fixture()
def db(tmp_path):
    return CharactersRAGDB(
        db_path=str(tmp_path / "character_store.sqlite"),
        client_id="character-store-user",
    )


@pytest.fixture()
def store(db):
    return CharacterStore(db)


class TestCharacterStoreAdd:
    def test_add_character_card(self, store):
        card_id = store.add_character_card({"name": "Test Character"})
        assert card_id is not None
        assert isinstance(card_id, int)

    def test_add_requires_name(self, store):
        with pytest.raises(InputError, match="name"):
            store.add_character_card({})

    def test_add_duplicate_name_raises_conflict(self, store):
        store.add_character_card({"name": "Unique Name"})
        with pytest.raises(ConflictError):
            store.add_character_card({"name": "Unique Name"})


class TestCharacterStoreGet:
    def test_get_by_id(self, store):
        card_id = store.add_character_card({"name": "Get By ID"})
        card = store.get_character_card_by_id(card_id)
        assert card is not None
        assert card["name"] == "Get By ID"

    def test_get_by_id_nonexistent(self, store):
        card = store.get_character_card_by_id(99999)
        assert card is None

    def test_get_by_name(self, store):
        store.add_character_card({"name": "Named Card"})
        card = store.get_character_card_by_name("Named Card")
        assert card is not None
        assert card["name"] == "Named Card"


class TestCharacterStoreList:
    def test_list_empty(self, store):
        cards = store.list_character_cards()
        # May include default card from DB init
        assert isinstance(cards, list)

    def test_list_after_add(self, store):
        store.add_character_card({"name": "Card A"})
        store.add_character_card({"name": "Card B"})
        cards = store.list_character_cards()
        names = [c["name"] for c in cards]
        assert "Card A" in names
        assert "Card B" in names

    def test_list_respects_limit(self, store):
        for i in range(5):
            store.add_character_card({"name": f"Limited {i}"})
        cards = store.list_character_cards(limit=2)
        assert len(cards) <= 2


class TestCharacterStoreUpdate:
    def test_update_card(self, store):
        card_id = store.add_character_card({"name": "Original"})
        card = store.get_character_card_by_id(card_id)
        result = store.update_character_card(
            card_id,
            {"name": "Updated", "description": "New desc"},
            expected_version=card["version"],
        )
        assert result is True
        updated = store.get_character_card_by_id(card_id)
        assert updated["name"] == "Updated"
        assert updated["description"] == "New desc"


class TestCharacterStoreSoftDelete:
    def test_soft_delete(self, store):
        card_id = store.add_character_card({"name": "To Delete"})
        card = store.get_character_card_by_id(card_id)
        result = store.soft_delete_character_card(card_id, expected_version=card["version"])
        assert result is True
        deleted = store.get_character_card_by_id(card_id)
        assert deleted is None or deleted.get("deleted") == 1


class TestCharacterStoreSearch:
    def test_search_by_term(self, store):
        store.add_character_card({"name": "Searchable Hero"})
        results = store.search_character_cards("Searchable")
        assert len(results) >= 1
        assert any("Searchable" in r["name"] for r in results)
