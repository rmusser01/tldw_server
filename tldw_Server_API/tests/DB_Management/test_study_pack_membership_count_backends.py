"""Study Pack append counts and rollback use real rows on both backends."""

import pytest

from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError

pytestmark = pytest.mark.integration


@pytest.fixture(params=["sqlite", "postgresql"])
def pack_db(request, tmp_path):
    backend = None
    if request.param == "postgresql":
        backend = DatabaseBackendFactory.create_backend(request.getfixturevalue("pg_database_config"))
    db = CharactersRAGDB(tmp_path / "membership.db", client_id="1", backend=backend)
    try:
        yield db
    finally:
        db.close_all_connections()
        if backend is not None:
            backend.get_pool().close_all()


def create_pack(db, title="Citrine"):
    return db.create_study_pack(
        title=title,
        workspace_id=None,
        deck_id=None,
        source_bundle_json={"items": [{"source_type": "note", "source_id": "fixture-note"}]},
        generation_options_json={},
    )


def test_empty_append_does_not_change_membership(pack_db):
    pack = create_pack(pack_db)
    assert pack_db.add_study_pack_cards(pack, []) == 0
    assert pack_db.list_study_pack_cards(pack) == []


def test_append_counts_new_members_only_and_scopes_each_pack(pack_db):
    first = create_pack(pack_db)
    second = create_pack(pack_db, "Rowan")
    cards = [pack_db.add_flashcard({"front": front, "back": "Fixture"}) for front in ("Amber", "East")]
    assert pack_db.add_study_pack_cards(first, [cards[0]]) == 1
    assert pack_db.add_study_pack_cards(first, [cards[0], cards[0]]) == 0
    assert pack_db.add_study_pack_cards(first, cards) == 1
    assert pack_db.add_study_pack_cards(second, cards) == 2
    assert [row["flashcard_uuid"] for row in pack_db.list_study_pack_cards(first)] == cards
    assert [row["flashcard_uuid"] for row in pack_db.list_study_pack_cards(second)] == cards


def test_outer_rollback_removes_successful_append(pack_db):
    pack = create_pack(pack_db)
    card = pack_db.add_flashcard({"front": "Amber", "back": "Fixture"})
    with pytest.raises(RuntimeError, match="caller rollback"):
        with pack_db.transaction():
            assert pack_db.add_study_pack_cards(pack, [card]) == 1
            raise RuntimeError("caller rollback")
    assert pack_db.list_study_pack_cards(pack) == []


def test_invalid_card_rolls_back_batch_and_preserves_existing_members(pack_db):
    pack = create_pack(pack_db)
    first = pack_db.add_flashcard({"front": "Amber", "back": "Fixture"})
    second = pack_db.add_flashcard({"front": "East", "back": "Fixture"})
    assert pack_db.add_study_pack_cards(pack, [first]) == 1
    with pytest.raises(CharactersRAGDBError):
        pack_db.add_study_pack_cards(pack, [second, "missing-fixture-card"])
    assert [row["flashcard_uuid"] for row in pack_db.list_study_pack_cards(pack)] == [first]
