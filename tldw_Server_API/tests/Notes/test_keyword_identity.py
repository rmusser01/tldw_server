"""Case-insensitive ChaCha keyword identity (TASK-13950.5, slice 1).

Keywords are case-insensitive product-wide with display casing preserved: the
first-seen casing is stored and every lookup, dedupe and link goes through
``normalize_keyword_identity`` (NFC plus casefold).

The database alone cannot do that. SQLite ``COLLATE NOCASE`` folds ASCII only
("Éclair" != "éclair") and Postgres ``LOWER()`` does not fold the German sharp s
("straße" != "strasse"), so ``KeywordStore`` falls back to comparing helper
identities in Python. The non-ASCII cases below pin that alignment on both
backends. Known limit: ``rename_keyword``'s duplicate check is SQL-only, so a
rename into a Unicode-only variant of another keyword is not refused.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, ConflictError

NON_ASCII_VARIANTS = [
    pytest.param("Straße", "STRASSE", id="sharp-s"),
    pytest.param("Éclair", "éclair", id="accented-capital"),
    pytest.param("éclair", "éclair", id="nfc-composition"),
]


@pytest.fixture(
    params=[
        pytest.param("sqlite", marks=pytest.mark.unit),
        pytest.param("postgres", marks=pytest.mark.integration),
    ]
)
def db(request, tmp_path):
    if request.param == "sqlite":
        database = CharactersRAGDB(db_path=str(tmp_path / "keyword_identity.sqlite"), client_id="1")
    else:
        backend = DatabaseBackendFactory.create_backend(request.getfixturevalue("pg_database_config"))
        database = CharactersRAGDB(db_path=":memory:", client_id="913950", backend=backend)
    yield database
    database.close_connection()


def _note_keywords(db: CharactersRAGDB, note_id: str) -> list[str]:
    return [row["keyword"] for row in db.get_keywords_for_note(note_id)]


def _note_with_keyword(db: CharactersRAGDB, keyword: str) -> str:
    note_id = db.add_note(title="Keyword identity", content="Body")
    db.link_note_to_keyword(note_id, db.add_keyword(keyword))
    return note_id


# --- the helper ---------------------------------------------------------------


@pytest.mark.unit
def test_normalize_keyword_identity_is_nfc_plus_casefold() -> None:
    from tldw_Server_API.app.core.DB_Management.chacha.organization_sync_store import (
        NotesOrganizationSyncStore,
    )
    from tldw_Server_API.app.core.Utils.keyword_identity import normalize_keyword_identity

    assert normalize_keyword_identity("  Foo ") == normalize_keyword_identity("fOO") == "foo"
    assert normalize_keyword_identity("Straße") == normalize_keyword_identity("STRASSE")
    assert normalize_keyword_identity("Éclair") == normalize_keyword_identity("éclair")
    assert NotesOrganizationSyncStore.normalize_keyword_identity("Straße") == "strasse"


# --- the store ------------------------------------------------------------------


def test_adding_foo_then_foo_keeps_one_keyword_found_in_either_case(db) -> None:
    keyword_id = db.add_keyword("Foo")

    with pytest.raises(ConflictError):
        db.add_keyword("foo")

    assert db.count_keywords() == 1
    for probe in ("Foo", "foo", "FOO"):
        found = db.get_keyword_by_text(probe)
        assert found is not None, probe
        assert (found["id"], found["keyword"]) == (keyword_id, "Foo")


def test_readding_a_soft_deleted_keyword_in_another_case_restores_it(db) -> None:
    keyword_id = db.add_keyword("Foo")
    db.soft_delete_keyword(keyword_id, db.get_keyword_by_id(keyword_id)["version"])

    assert db.add_keyword("foo") == keyword_id
    assert db.count_keywords() == 1


@pytest.mark.parametrize(("stored", "probe"), NON_ASCII_VARIANTS)
def test_non_ascii_case_variants_share_one_keyword(db, stored: str, probe: str) -> None:
    keyword_id = db.add_keyword(stored)

    found = db.get_keyword_by_text(probe)
    assert found is not None
    assert (found["id"], found["keyword"]) == (keyword_id, stored)
    with pytest.raises(ConflictError):
        db.add_keyword(probe)
    assert db.count_keywords() == 1


def test_keyword_collection_names_match_in_either_case(db) -> None:
    collection_id = db.add_keyword_collection("Foo")

    with pytest.raises(ConflictError):
        db.add_keyword_collection("foo")
    found = db.get_keyword_collection_by_name("FOO")
    assert found is not None
    assert (found["id"], found["name"]) == (collection_id, "Foo")

    db.soft_delete_keyword_collection(collection_id, found["version"])
    assert db.add_keyword_collection("foo") == collection_id
    assert db.count_keyword_collections() == 1


def test_flashcard_tag_filters_match_in_either_case(db) -> None:
    card = db.add_flashcard({"front": "Q", "back": "A"})
    db.set_flashcard_tags(card, ["Foo"])

    assert [row["uuid"] for row in db.list_flashcards(tag="foo")] == [card]
    assert db.count_flashcards(tag="FOO") == 1
    db.review_flashcard(card, 4, review_mode="due", review_deck_id=None, review_tag_filter="foo")
    assert db.get_latest_flashcard_review(card) is not None


# --- the call sites ---------------------------------------------------------------


def test_notes_keyword_sync_links_case_variants_once(db) -> None:
    from tldw_Server_API.app.api.v1.endpoints.notes import (
        _get_or_create_keyword_row,
        _sync_note_keywords,
    )

    note_id = _note_with_keyword(db, "foo")
    foo_id = db.get_keyword_by_text("foo")["id"]

    summary = _sync_note_keywords(db, note_id, ["Foo", "foo"])

    assert summary["failed_count"] == 0
    assert _note_keywords(db, note_id) == ["foo"]
    assert _get_or_create_keyword_row(db, "FOO")["id"] == foo_id
    assert db.count_keywords() == 1


@pytest.mark.parametrize(("stored", "probe"), NON_ASCII_VARIANTS)
def test_notes_keyword_sync_treats_non_ascii_variants_as_one(db, stored: str, probe: str) -> None:
    from tldw_Server_API.app.api.v1.endpoints.notes import _sync_note_keywords

    note_id = _note_with_keyword(db, stored)

    _sync_note_keywords(db, note_id, [probe])

    assert _note_keywords(db, note_id) == [stored]
    assert db.count_keywords() == 1


def test_web_clipper_keyword_sync_links_case_variants_once(db) -> None:
    from tldw_Server_API.app.core.WebClipper.service import WebClipperService

    note_id = _note_with_keyword(db, "foo")
    warnings: list[str] = []

    WebClipperService(db=db, user_id="1")._sync_keywords(note_id, ["Foo", "foo", "Straße"], warnings)
    WebClipperService(db=db, user_id="1")._sync_keywords(note_id, ["FOO", "STRASSE"], warnings)

    assert warnings == []
    assert sorted(_note_keywords(db, note_id)) == ["Straße", "foo"]
    assert db.count_keywords() == 2


def test_conversation_keyword_replace_links_case_variants_once(db) -> None:
    from tldw_Server_API.app.api.v1.endpoints.chat import _replace_conversation_keywords

    conversation_id = db.add_conversation({"character_id": 1, "title": "Keyword identity"})
    db.link_conversation_to_keyword(conversation_id, db.add_keyword("foo"))
    db.link_conversation_to_keyword(conversation_id, db.add_keyword("Straße"))

    _replace_conversation_keywords(
        db, conversation_id, ["Foo", "foo", "STRASSE"], owner_user_id=db.client_id
    )

    assert sorted(row["keyword"] for row in db.get_keywords_for_conversation(conversation_id)) == [
        "Straße",
        "foo",
    ]
    assert db.count_keywords() == 2


@pytest.mark.asyncio
async def test_mcp_note_tags_keep_display_casing(db, monkeypatch) -> None:
    from tldw_Server_API.app.core.MCP_unified.modules.base import ModuleConfig
    from tldw_Server_API.app.core.MCP_unified.modules.implementations.notes_module import NotesModule

    module = NotesModule(ModuleConfig(name="notes"))
    monkeypatch.setattr(module, "_open_db", lambda _context: db)
    monkeypatch.setattr(db, "close_all_connections", lambda: None)
    context = SimpleNamespace(metadata={"roles": []}, user_id=db.client_id)
    note_id = db.add_note(title="MCP tags", content="Body")

    async def tool(name: str, tags: list[str]) -> list[str]:
        result = await module.execute_tool(name, {"note_id": note_id, "tags": tags}, context=context)
        return result["tags"]

    assert await tool("notes.tags.add", ["Foo", "foo"]) == ["Foo"]
    assert await tool("notes.tags.add", ["FOO", "Straße"]) == ["Foo", "Straße"]
    assert await tool("notes.tags.set", ["foo", "STRASSE", "Bar"]) == ["Bar", "Foo", "Straße"]
    assert await tool("notes.tags.remove", ["BAR", "strasse"]) == ["Foo"]
    assert sorted(row["keyword"] for row in db.list_keywords()) == ["Bar", "Foo", "Straße"]


@pytest.mark.asyncio
async def test_mcp_flashcard_tags_keep_display_casing(db, monkeypatch) -> None:
    from tldw_Server_API.app.core.MCP_unified.modules.base import ModuleConfig
    from tldw_Server_API.app.core.MCP_unified.modules.implementations.flashcards_module import (
        FlashcardsModule,
    )

    module = FlashcardsModule(ModuleConfig(name="flashcards"))
    monkeypatch.setattr(module, "_open_db", lambda _context: db)
    monkeypatch.setattr(db, "close_all_connections", lambda: None)
    context = SimpleNamespace(metadata={})
    card = db.add_flashcard({"front": "Q", "back": "A"})

    result = await module.execute_tool(
        "flashcards.tags.set", {"card_uuid": card, "tags": ["Foo", "foo", "Straße", "STRASSE"]}, context=context
    )
    listed = await module.execute_tool("flashcards.tags.get", {"card_uuid": card}, context=context)

    assert result["tags"] == ["Foo", "Straße"]
    assert sorted(listed["tags"]) == ["Foo", "Straße"]
    assert sorted(row["keyword"] for row in db.list_keywords()) == ["Foo", "Straße"]


# --- request parsing ----------------------------------------------------------------


@pytest.mark.unit
def test_request_keyword_lists_dedupe_by_identity_and_keep_first_casing() -> None:
    from tldw_Server_API.app.api.v1.endpoints.notes import (
        _normalize_import_keywords,
        _normalize_keyword_tokens,
    )
    from tldw_Server_API.app.api.v1.schemas.chat_conversation_schemas import ConversationUpdateRequest
    from tldw_Server_API.app.api.v1.schemas.chat_knowledge_schemas import KnowledgeSaveRequest
    from tldw_Server_API.app.api.v1.schemas.notes_schemas import NoteCreate, NoteUpdate

    raw = ["Foo", "foo", "Straße", "STRASSE", "Éclair", "éclair"]
    expected = ["Foo", "Straße", "Éclair"]

    assert ConversationUpdateRequest(version=1, keywords=raw).keywords == expected
    assert KnowledgeSaveRequest(conversation_id="c", snippet="s", tags=raw).tags == expected
    assert NoteCreate(title="t", content="c", keywords=raw).normalized_keywords == expected
    assert NoteUpdate(keywords=raw).normalized_keywords == expected
    assert _normalize_import_keywords(raw) == expected
    assert _normalize_keyword_tokens(raw) == expected
