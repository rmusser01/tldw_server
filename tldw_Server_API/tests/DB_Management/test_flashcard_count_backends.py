"""Flashcard counts must consume actual SQLite and PostgreSQL result rows."""

import pytest

from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

pytestmark = pytest.mark.integration


@pytest.fixture(params=["sqlite", "postgresql"])
def count_db(request, tmp_path):
    """Build a normal database using the official isolated PostgreSQL fixture."""
    backend = None
    if request.param == "postgresql":
        backend = DatabaseBackendFactory.create_backend(
            request.getfixturevalue("pg_database_config")
        )
    db = CharactersRAGDB(str(tmp_path / "count.db"), client_id="1", backend=backend)
    try:
        yield db
    finally:
        db.close_connection()
        if backend is not None:
            backend.get_pool().close_all()


def test_empty_flashcard_count_is_zero(count_db):
    """An empty result is a valid count, not a mapping-index exception."""
    assert count_db.count_flashcards() == 0


@pytest.mark.parametrize("filtered", [False, True])
def test_flashcard_count_matches_deck_filter(count_db, filtered):
    """Named aggregate access preserves both total and selected-deck counts."""
    first_deck = count_db.add_deck("Citrine")
    second_deck = count_db.add_deck("Rowan")
    for deck_id, front in [(first_deck, "Amber"), (first_deck, "Tuesday"), (second_deck, "East")]:
        count_db.add_flashcard({"deck_id": deck_id, "front": front, "back": "Fixture"})
    filters = {"deck_id": first_deck} if filtered else {}
    assert count_db.count_flashcards(**filters) == (2 if filtered else 3)


def test_flashcard_asset_reconciliation_round_trips_real_rows(count_db):
    """Real asset rows attach, reject foreign cards, and detach on both backends."""
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import ConflictError
    from tldw_Server_API.app.core.Flashcards.asset_refs import build_flashcard_asset_markdown

    asset_uuid = count_db.add_flashcard_asset(
        image_bytes=b"fixture-image", mime_type="image/png", original_filename="fixture.png"
    )
    card_uuid = count_db.add_flashcard({"front": "Citrine", "back": "Amber"})
    other_uuid = count_db.add_flashcard({"front": "Rowan", "back": "East"})
    reference = build_flashcard_asset_markdown(asset_uuid, "Fixture")
    assert count_db.reconcile_flashcard_asset_refs(
        card_uuid, front=reference, back="Amber", extra="", notes=""
    ) == [asset_uuid]
    assert count_db.get_flashcard_asset(asset_uuid)["card_uuid"] == card_uuid
    with pytest.raises(ConflictError, match="different card"):
        count_db.reconcile_flashcard_asset_refs(
            other_uuid, front=reference, back="East", extra="", notes=""
        )
    count_db.reconcile_flashcard_asset_refs(
        card_uuid, front="Citrine", back="Amber", extra="", notes=""
    )
    assert count_db.get_flashcard_asset(asset_uuid)["card_uuid"] is None


def test_flashcard_noop_update_preserves_version_guard(count_db):
    """An empty update checks the current version without modifying the card."""
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import ConflictError

    card_uuid = count_db.add_flashcard({"front": "Marker?", "back": "Amber"})
    before = count_db.get_flashcard(card_uuid)
    assert count_db.update_flashcard(card_uuid, {}, expected_version=1)
    with pytest.raises(ConflictError, match="Version mismatch"):
        count_db.update_flashcard(card_uuid, {}, expected_version=2)
    assert count_db.get_flashcard(card_uuid) == before
