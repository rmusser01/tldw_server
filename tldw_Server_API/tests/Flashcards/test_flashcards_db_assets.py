import hashlib

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.Flashcards.asset_refs import build_flashcard_asset_markdown


@pytest.fixture
def chacha_db(tmp_path):
    db = CharactersRAGDB(str(tmp_path / "flashcards-assets.db"), client_id="flashcard-asset-tests")
    try:
        yield db
    finally:
        db.close_connection()


def test_add_flashcard_asset_round_trips_metadata_and_content(chacha_db):
    image_bytes = b"\x89PNG\r\n\x1a\nasset-bytes"
    asset_uuid = chacha_db.add_flashcard_asset(
        image_bytes=image_bytes,
        mime_type="image/png",
        original_filename="slide.png",
        width=640,
        height=480,
    )

    asset = chacha_db.get_flashcard_asset(asset_uuid)

    assert asset is not None
    assert asset["uuid"] == asset_uuid
    assert asset["card_uuid"] is None
    assert asset["mime_type"] == "image/png"
    assert asset["original_filename"] == "slide.png"
    assert asset["byte_size"] == len(image_bytes)
    assert asset["width"] == 640
    assert asset["height"] == 480
    assert asset["sha256"] == hashlib.sha256(image_bytes).hexdigest()
    assert chacha_db.get_flashcard_asset_content(asset_uuid) == image_bytes


def test_reconcile_flashcard_asset_refs_attaches_and_detaches_assets(chacha_db):
    deck_id = chacha_db.add_deck("Histology")
    first_asset_uuid = chacha_db.add_flashcard_asset(
        image_bytes=b"first-image",
        mime_type="image/png",
        original_filename="first.png",
    )
    second_asset_uuid = chacha_db.add_flashcard_asset(
        image_bytes=b"second-image",
        mime_type="image/png",
        original_filename="second.png",
    )

    card_uuid = chacha_db.add_flashcard(
        {
            "deck_id": deck_id,
            "front": "Question",
            "back": "Answer",
            "notes": "",
            "extra": "",
        }
    )

    chacha_db.reconcile_flashcard_asset_refs(
        card_uuid,
        front=build_flashcard_asset_markdown(first_asset_uuid, "Front slide"),
        back="Back " + build_flashcard_asset_markdown(second_asset_uuid, "Back slide"),
        extra="",
        notes="",
    )

    first_asset = chacha_db.get_flashcard_asset(first_asset_uuid)
    second_asset = chacha_db.get_flashcard_asset(second_asset_uuid)
    assert first_asset["card_uuid"] == card_uuid
    assert second_asset["card_uuid"] == card_uuid

    chacha_db.reconcile_flashcard_asset_refs(
        card_uuid,
        front="Question only",
        back="Answer only",
        extra="",
        notes="",
    )

    first_asset = chacha_db.get_flashcard_asset(first_asset_uuid)
    second_asset = chacha_db.get_flashcard_asset(second_asset_uuid)
    assert first_asset["card_uuid"] is None
    assert second_asset["card_uuid"] is None


def test_flashcard_search_shadow_columns_strip_asset_refs(chacha_db):
    asset_uuid = chacha_db.add_flashcard_asset(
        image_bytes=b"searchable-image",
        mime_type="image/png",
        original_filename="search.png",
    )
    deck_id = chacha_db.add_deck("Anatomy")
    card_uuid = chacha_db.add_flashcard(
        {
            "deck_id": deck_id,
            "front": f"Start {build_flashcard_asset_markdown(asset_uuid, 'Histology slide')} end",
            "back": "Answer text",
            "notes": f"Notes {build_flashcard_asset_markdown(asset_uuid, 'Annotated diagram')}",
            "extra": f"Extra {build_flashcard_asset_markdown(asset_uuid, 'Should not affect search shadow')}",
        }
    )

    row = chacha_db.execute_query(
        "SELECT front_search, back_search, notes_search FROM flashcards WHERE uuid = ?",
        (card_uuid,),
    ).fetchone()

    assert row is not None
    assert row["front_search"] == "Start Histology slide end"
    assert row["back_search"] == "Answer text"
    assert row["notes_search"] == "Notes Annotated diagram"


def test_list_flashcards_search_matches_sanitized_alt_text(chacha_db):
    asset_uuid = chacha_db.add_flashcard_asset(
        image_bytes=b"fts-image",
        mime_type="image/png",
        original_filename="fts.png",
    )
    card_uuid = chacha_db.add_flashcard(
        {
            "front": build_flashcard_asset_markdown(asset_uuid, "renal cortex"),
            "back": "Kidney structure",
            "notes": "",
            "extra": "",
        }
    )

    results = chacha_db.list_flashcards(q="renal")
    total = chacha_db.count_flashcards(q="renal")

    assert total == 1
    assert [row["uuid"] for row in results] == [card_uuid]


def test_list_flashcards_search_accepts_hyphenated_plain_text(chacha_db):
    """Hyphenated flashcard search terms are normalized without FTS5 syntax errors."""
    card_uuid = chacha_db.add_flashcard(
        {
            "front": "codex-flashcards-ux front",
            "back": "Regression coverage for FTS5 punctuation handling",
            "notes": "",
            "extra": "",
        }
    )

    results = chacha_db.list_flashcards(q="codex-flashcards-ux")
    total = chacha_db.count_flashcards(q="codex-flashcards-ux")

    assert total == 1
    assert [row["uuid"] for row in results] == [card_uuid]


def test_list_flashcards_search_accepts_negative_hyphenated_terms(chacha_db):
    """Negative hyphenated flashcard search terms exclude matching cards safely."""
    keep_uuid = chacha_db.add_flashcard(
        {
            "front": "codex-flashcards-ux clean",
            "back": "",
            "notes": "",
            "extra": "",
        }
    )
    chacha_db.add_flashcard(
        {
            "front": "codex-flashcards-ux state-of-the-art",
            "back": "",
            "notes": "",
            "extra": "",
        }
    )

    results = chacha_db.list_flashcards(q="codex-flashcards-ux -state-of-the-art")

    assert [row["uuid"] for row in results] == [keep_uuid]

    scoped_results = chacha_db.list_flashcards(q="codex-flashcards-ux -front:state-of-the-art")

    assert [row["uuid"] for row in scoped_results] == [keep_uuid]
