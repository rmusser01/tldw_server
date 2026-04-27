"""Tests for the extracted CharacterStore."""

import ast
import inspect
from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    InputError,
    ConflictError,
)
from tldw_Server_API.app.core.DB_Management.chacha.character_store import CharacterStore


pytestmark = pytest.mark.unit


_DELEGATED_CHARACTER_METHODS = {
    "add_character_card",
    "get_character_card_by_id",
    "get_character_card_by_name",
    "list_character_cards",
    "query_character_cards",
    "_normalize_character_tags_for_operation",
    "_apply_character_tag_operation_to_list",
    "manage_character_tags",
    "update_character_card",
    "soft_delete_character_card",
    "restore_character_card",
    "search_character_cards",
    "search_character_cards_by_tags",
    "_check_json_support",
    "_search_cards_by_tags_json",
    "_search_cards_by_tags_fallback",
}


def _class_method_names(class_obj: type[object]) -> set[str]:
    source_path = Path(inspect.getsourcefile(class_obj) or "")
    assert source_path.exists()
    tree = ast.parse(source_path.read_text())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_obj.__name__:
            return {
                item.name
                for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
    raise AssertionError(f"Class {class_obj.__name__} not found in {source_path}")


@pytest.fixture()
def db(tmp_path):
    return CharactersRAGDB(
        db_path=str(tmp_path / "character_store.sqlite"),
        client_id="character-store-user",
    )


@pytest.fixture()
def store(db):
    return CharacterStore(db)


def test_character_store_owns_delegated_methods_without_monolith_duplicates(db, monkeypatch):
    class_method_names = _class_method_names(CharactersRAGDB)
    assert _DELEGATED_CHARACTER_METHODS.isdisjoint(class_method_names)

    captured: dict[str, object] = {}

    def _fake_add_character_card(card_data):
        captured["card_data"] = card_data
        return 987

    monkeypatch.setattr(db.character_store, "add_character_card", _fake_add_character_card)

    assert db.add_character_card({"name": "Delegated Character"}) == 987
    assert captured["card_data"] == {"name": "Delegated Character"}


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

    def test_search_by_term_uses_escaped_phrase(self, store):
        store.add_character_card({"name": 'Quoted "Hero"'})
        results = store.search_character_cards('Quoted "Hero"')
        assert any(row["name"] == 'Quoted "Hero"' for row in results)

    def test_tag_search_handles_legacy_non_json_tags(self, store):
        card_id = store.add_character_card({"name": "Legacy Tagged", "tags": "legacy-tag"})
        results = store.search_character_cards_by_tags(["legacy-tag"])
        assert {row["id"] for row in results} == {card_id}

    def test_manage_tags_rename_updates_tag_search_results(self, store):
        first_id = store.add_character_card({"name": "Tagged One", "tags": ["old-tag", "shared"]})
        second_id = store.add_character_card({"name": "Tagged Two", "tags": ["shared"]})

        before = store.search_character_cards_by_tags(["old-tag"])
        assert {row["id"] for row in before} == {first_id}

        summary = store.manage_character_tags(
            operation="rename",
            source_tag="old-tag",
            target_tag="new-tag",
        )

        assert summary["matched_count"] == 1
        assert summary["updated_count"] == 1
        assert summary["updated_character_ids"] == [first_id]

        renamed = store.get_character_card_by_id(first_id)
        untouched = store.get_character_card_by_id(second_id)
        assert renamed is not None
        assert untouched is not None
        assert set(renamed["tags"]) == {"new-tag", "shared"}
        assert set(untouched["tags"]) == {"shared"}
        assert store.search_character_cards_by_tags(["old-tag"]) == []
        assert {row["id"] for row in store.search_character_cards_by_tags(["new-tag"])} == {first_id}
