from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


@pytest.fixture
def db(tmp_path: Path):
    db_instance = CharactersRAGDB(str(tmp_path / "ChaChaNotes.db"), client_id="test")
    try:
        yield db_instance
    finally:
        db_instance.close_all_connections()


def test_flashcards_basic_flow(db):
    # Schema version should be 5
    conn = db.get_connection()
    ver = conn.execute("SELECT version FROM db_schema_version WHERE schema_name='rag_char_chat_schema'").fetchone()
    assert ver and ver[0] >= 5

    # Create deck
    deck_id = db.add_deck("Test Deck", "desc")
    assert isinstance(deck_id, int)

    # Create card
    card_uuid = db.add_flashcard({
        "deck_id": deck_id,
        "front": "What is 2+2?",
        "back": "4",
        "tags_json": "[\"math\"]",
    })
    assert isinstance(card_uuid, str)

    # List and find it
    items = db.list_flashcards(deck_id=deck_id, limit=10)
    assert any(i["uuid"] == card_uuid for i in items)

    # Review with rating 4
    updated = db.review_flashcard(card_uuid, rating=4)
    assert updated["interval_days"] >= 1
    assert updated["due_at"] is not None

    # Export contains content
    data = db.export_flashcards_csv(deck_id=deck_id)
    text = data.decode("utf-8")
    assert "What is 2+2?" in text


def test_deck_workspace_id_persists_and_can_move_between_scopes(db):
    db.upsert_workspace("ws-1", "Workspace One")

    deck_id = db.add_deck("Scoped Deck", "desc", workspace_id="ws-1")
    deck = db.get_deck(deck_id)
    assert deck is not None
    assert deck["workspace_id"] == "ws-1"

    default_items = db.list_decks(limit=20, offset=0)
    assert all(item["id"] != deck_id for item in default_items)

    workspace_items = db.list_decks(workspace_id="ws-1", limit=20, offset=0)
    assert [item["id"] for item in workspace_items] == [deck_id]

    all_items = db.list_decks(include_workspace_items=True, limit=20, offset=0)
    assert any(item["id"] == deck_id for item in all_items)

    assert db.update_deck(deck_id, workspace_id=None, expected_version=deck["version"]) is True
    moved_to_general = db.get_deck(deck_id)
    assert moved_to_general is not None
    assert moved_to_general["workspace_id"] is None

    general_items = db.list_decks(limit=20, offset=0)
    assert any(item["id"] == deck_id for item in general_items)

    assert db.update_deck(deck_id, workspace_id="ws-1", expected_version=moved_to_general["version"]) is True
    moved_back = db.get_deck(deck_id)
    assert moved_back is not None
    assert moved_back["workspace_id"] == "ws-1"


def test_flashcard_visibility_defaults_to_general_only_and_honors_explicit_scope(db):
    db.upsert_workspace("ws-1", "Workspace One")

    general_deck = db.add_deck("General Deck", "desc")
    workspace_deck = db.add_deck("Workspace Deck", "desc", workspace_id="ws-1")

    workspace_uuid = db.add_flashcard({
        "deck_id": workspace_deck,
        "front": "Workspace Front",
        "back": "Workspace Back",
    })
    general_uuid = db.add_flashcard({
        "deck_id": general_deck,
        "front": "General Front",
        "back": "General Back",
    })

    default_items = db.list_flashcards(limit=10)
    assert any(item["uuid"] == general_uuid for item in default_items)
    assert all(item["deck_id"] != workspace_deck for item in default_items)

    workspace_items = db.list_flashcards(workspace_id="ws-1", limit=10)
    assert {item["uuid"] for item in workspace_items} == {workspace_uuid}

    all_items = db.list_flashcards(include_workspace_items=True, limit=10)
    assert {item["uuid"] for item in all_items} == {general_uuid, workspace_uuid}

    deck_items = db.list_flashcards(deck_id=workspace_deck, limit=10)
    assert {item["uuid"] for item in deck_items} == {workspace_uuid}

    assert db.count_flashcards() == 1
    assert db.count_flashcards(workspace_id="ws-1") == 1
    assert db.count_flashcards(include_workspace_items=True) == 2
    assert db.count_flashcards(deck_id=workspace_deck) == 1

    next_card, reason = db.get_next_review_card()
    assert reason == "new"
    assert next_card is not None
    assert next_card["uuid"] == general_uuid

    workspace_next, workspace_reason = db.get_next_review_card(workspace_id="ws-1")
    assert workspace_reason == "new"
    assert workspace_next is not None
    assert workspace_next["uuid"] == workspace_uuid

    deck_next, deck_reason = db.get_next_review_card(deck_id=workspace_deck)
    assert deck_reason == "new"
    assert deck_next is not None
    assert deck_next["uuid"] == workspace_uuid

    default_export = db.export_flashcards_csv()
    default_text = default_export.decode("utf-8")
    assert "General Front" in default_text
    assert "Workspace Front" not in default_text

    workspace_export = db.export_flashcards_csv(workspace_id="ws-1")
    workspace_text = workspace_export.decode("utf-8")
    assert "Workspace Front" in workspace_text
    assert "General Front" not in workspace_text

    deck_export = db.export_flashcards_csv(deck_id=workspace_deck)
    deck_text = deck_export.decode("utf-8")
    assert "Workspace Front" in deck_text
    assert "General Front" not in deck_text


def test_flashcard_analytics_summary_respects_workspace_visibility(db):
    db.upsert_workspace("ws-1", "Workspace One")

    general_deck = db.add_deck("General Deck", "desc")
    workspace_deck = db.add_deck("Workspace Deck", "desc", workspace_id="ws-1")

    db.add_flashcard({
        "deck_id": general_deck,
        "front": "General Front",
        "back": "General Back",
    })
    workspace_uuid = db.add_flashcard({
        "deck_id": workspace_deck,
        "front": "Workspace Front",
        "back": "Workspace Back",
    })
    db.review_flashcard(workspace_uuid, rating=4)

    default_summary = db.get_flashcard_analytics_summary()
    assert default_summary["reviewed_today"] == 0
    assert all(deck["deck_id"] != workspace_deck for deck in default_summary["decks"])

    workspace_summary = db.get_flashcard_analytics_summary(workspace_id="ws-1")
    assert workspace_summary["reviewed_today"] == 1
    assert {deck["deck_id"] for deck in workspace_summary["decks"]} == {workspace_deck}

    all_summary = db.get_flashcard_analytics_summary(include_workspace_items=True)
    assert all_summary["reviewed_today"] == 1
    assert {deck["deck_id"] for deck in all_summary["decks"]} == {general_deck, workspace_deck}

    deck_summary = db.get_flashcard_analytics_summary(deck_id=workspace_deck)
    assert deck_summary["reviewed_today"] == 1
    assert {deck["deck_id"] for deck in deck_summary["decks"]} == {workspace_deck}


def test_flashcard_analytics_summary_keeps_empty_targeted_deck_visible(db):
    empty_deck = db.add_deck("Empty Deck", "desc")

    summary = db.get_flashcard_analytics_summary(deck_id=empty_deck)

    assert summary["reviewed_today"] == 0
    assert len(summary["decks"]) == 1
    assert summary["decks"][0]["deck_id"] == empty_deck
    assert summary["decks"][0]["total"] == 0
    assert summary["decks"][0]["new"] == 0
    assert summary["decks"][0]["learning"] == 0
    assert summary["decks"][0]["due"] == 0
    assert summary["decks"][0]["mature"] == 0


def test_flashcard_analytics_summary_excludes_reviews_from_deleted_decks(db):
    deck_id = db.add_deck("Deleted Deck", "desc")
    card_uuid = db.add_flashcard({
        "deck_id": deck_id,
        "front": "Front",
        "back": "Back",
    })
    db.review_flashcard(card_uuid, rating=4)

    conn = db.get_connection()
    conn.execute("UPDATE decks SET deleted = 1 WHERE id = ?", (deck_id,))
    conn.commit()

    summary = db.get_flashcard_analytics_summary(include_workspace_items=True)

    assert summary["reviewed_today"] == 0
    assert all(deck["deck_id"] != deck_id for deck in summary["decks"])
