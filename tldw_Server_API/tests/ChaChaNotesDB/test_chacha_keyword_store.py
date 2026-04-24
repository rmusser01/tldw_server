"""Tests for the extracted KeywordStore."""

import ast
import inspect
from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    ConflictError,
)
from tldw_Server_API.app.core.DB_Management.chacha.keyword_store import KeywordStore


pytestmark = pytest.mark.unit


_DELEGATED_KEYWORD_METHODS = {
    "add_keyword",
    "get_keyword_by_id",
    "get_keyword_by_text",
    "list_keywords",
    "count_keywords",
    "search_keywords",
    "add_keyword_collection",
    "get_keyword_collection_by_id",
    "list_keyword_collections",
    "rename_keyword",
    "merge_keywords",
    "link_conversation_to_keyword",
    "link_collection_to_keyword",
    "get_keywords_for_conversation",
    "get_notes_for_keyword",
    "get_collections_for_keyword",
    "unlink_conversation_from_keyword",
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
        db_path=str(tmp_path / "keyword_store.sqlite"),
        client_id="keyword-store-user",
    )


@pytest.fixture()
def store(db):
    return KeywordStore(db)


def test_keyword_store_owns_delegated_methods_without_monolith_duplicates(db, monkeypatch):
    class_method_names = _class_method_names(CharactersRAGDB)
    assert _DELEGATED_KEYWORD_METHODS.isdisjoint(class_method_names)

    captured: dict[str, object] = {}

    def _fake_add_keyword(keyword_text):
        captured["keyword_text"] = keyword_text
        return 1234

    monkeypatch.setattr(db.keyword_store, "add_keyword", _fake_add_keyword)

    assert db.add_keyword("delegated-keyword") == 1234
    assert captured["keyword_text"] == "delegated-keyword"


class TestKeywordStoreAdd:
    def test_add_keyword(self, store):
        kw_id = store.add_keyword("test-keyword")
        assert kw_id is not None
        assert isinstance(kw_id, int)

    def test_add_duplicate_raises_conflict(self, store):
        store.add_keyword("duplicate")
        with pytest.raises(ConflictError, match="already exists and is active"):
            store.add_keyword("duplicate")


class TestKeywordStoreGet:
    def test_get_by_id(self, store):
        kw_id = store.add_keyword("by-id")
        kw = store.get_keyword_by_id(kw_id)
        assert kw is not None
        assert kw["keyword"] == "by-id"

    def test_get_by_text(self, store):
        store.add_keyword("by-text")
        kw = store.get_keyword_by_text("by-text")
        assert kw is not None

    def test_get_nonexistent(self, store):
        kw = store.get_keyword_by_id(99999)
        assert kw is None


class TestKeywordStoreList:
    def test_list_keywords(self, store):
        store.add_keyword("kw-a")
        store.add_keyword("kw-b")
        keywords = store.list_keywords()
        assert len(keywords) >= 2

    def test_count_keywords(self, store):
        store.add_keyword("count-kw")
        count = store.count_keywords()
        assert count >= 1


class TestKeywordStoreSearch:
    def test_search_keywords_returns_exact_match(self, store):
        keyword_id = store.add_keyword("unique-searchterm-xyz")

        results = store.search_keywords("unique-searchterm-xyz")

        assert len(results) == 1
        assert results[0]["id"] == keyword_id
        assert results[0]["keyword"] == "unique-searchterm-xyz"

    def test_search_keywords_allows_punctuation(self, store):
        keyword_id = store.add_keyword("C++")

        results = store.search_keywords("C++")

        assert any(row["id"] == keyword_id for row in results)


class TestKeywordCollectionCRUD:
    def test_add_collection(self, store):
        coll_id = store.add_keyword_collection("Test Collection")
        assert coll_id is not None

    def test_get_collection_by_id(self, store):
        coll_id = store.add_keyword_collection("Get Collection")
        coll = store.get_keyword_collection_by_id(coll_id)
        assert coll is not None
        assert coll["name"] == "Get Collection"

    def test_list_collections(self, store):
        store.add_keyword_collection("Coll A")
        store.add_keyword_collection("Coll B")
        colls = store.list_keyword_collections()
        assert len(colls) >= 2


def test_keyword_store_rename_merge_and_link_helpers(store, db):
    character_id = db.add_character_card({"name": "Keyword Link Character"})
    conversation_id = db.add_conversation(
        {
            "character_id": character_id,
            "title": "Keyword Link Conversation",
        }
    )
    note_id = db.add_note(title="Keyword linked note", content="linked content")
    collection_id = store.add_keyword_collection("Linked Collection")
    source_keyword_id = store.add_keyword("merge-source")
    target_keyword_id = store.add_keyword("merge-target")

    assert db.link_note_to_keyword(note_id, source_keyword_id)
    assert store.link_conversation_to_keyword(conversation_id, source_keyword_id)
    assert store.link_collection_to_keyword(collection_id, source_keyword_id)

    source_keyword = store.get_keyword_by_id(source_keyword_id)
    target_keyword = store.get_keyword_by_id(target_keyword_id)
    assert source_keyword is not None
    assert target_keyword is not None

    renamed_source = store.rename_keyword(
        source_keyword_id,
        "merge-source-renamed",
        expected_version=source_keyword["version"],
    )
    assert renamed_source["keyword"] == "merge-source-renamed"

    merged = store.merge_keywords(
        source_keyword_id=source_keyword_id,
        target_keyword_id=target_keyword_id,
        expected_source_version=renamed_source["version"],
        expected_target_version=target_keyword["version"],
    )

    assert merged["merged_note_links"] == 1
    assert merged["merged_conversation_links"] == 1
    assert merged["merged_collection_links"] == 1
    assert store.get_keyword_by_id(source_keyword_id) is None
    assert {row["id"] for row in store.get_keywords_for_conversation(conversation_id)} == {
        target_keyword_id
    }
    assert [row["id"] for row in store.get_notes_for_keyword(target_keyword_id)] == [note_id]
    assert [row["id"] for row in store.get_collections_for_keyword(target_keyword_id)] == [collection_id]

    assert store.unlink_conversation_from_keyword(conversation_id, target_keyword_id)
    assert store.get_keywords_for_conversation(conversation_id) == []
    assert db.unlink_collection_to_keyword(collection_id, target_keyword_id)
    assert store.get_collections_for_keyword(target_keyword_id) == []
