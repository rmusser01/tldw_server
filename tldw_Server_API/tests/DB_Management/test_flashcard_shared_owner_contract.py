"""Private deck/card access must agree with the owner-scoped database instance."""

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints import flashcards
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

pytestmark = pytest.mark.integration


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=pytest.mark.postgres)])
def owners(request, tmp_path):
    backend = None
    if request.param == "postgres":
        backend = DatabaseBackendFactory.create_backend(request.getfixturevalue("pg_database_config"))
    alice = CharactersRAGDB(tmp_path / "2" / "ChaChaNotes.db", client_id="2", backend=backend)
    bob = CharactersRAGDB(tmp_path / "3" / "ChaChaNotes.db", client_id="3", backend=backend)
    try:
        alice.add_deck("Alice padding")
        private_deck = alice.add_deck("Alice private Citrine", visibility="private")
        private_card = alice.add_flashcard({"deck_id": private_deck, "front": "Alice confidential Citrine", "back": "Ten percent", "interval_days": 4, "repetitions": 2, "queue_state": "review"})
        alice.add_flashcard({"front": "Alice orphan private", "back": "Private answer"})
        bob_deck = bob.add_deck("Bob private Rowan")
        bob_card = bob.add_flashcard({"deck_id": bob_deck, "front": "Bob Rowan", "back": "Twenty percent"})
        alice.close_connection()
        bob.close_connection()
        yield SimpleNamespace(alice=alice, bob=bob, deck=private_deck, card=private_card, bob_deck=bob_deck, bob_card=bob_card, postgres=request.param == "postgres")
    finally:
        alice.close_all_connections()
        bob.close_all_connections()
        if backend is not None:
            backend.get_pool().close_all()


def _client(db):
    app = FastAPI()
    app.include_router(flashcards.router, prefix="/api/v1")
    app.dependency_overrides[flashcards.get_chacha_db_for_user] = lambda: db
    return TestClient(app, raise_server_exceptions=False)


def _raw_card(db, card):
    row = db.execute_query("SELECT * FROM flashcards WHERE uuid = ?", (card,), read_only=True).fetchone()
    return dict(row) if row else None


def _raw_deck(db, deck):
    row = db.execute_query("SELECT * FROM decks WHERE id = ?", (deck,), read_only=True).fetchone()
    return dict(row) if row else None


@pytest.mark.parametrize("route", ["decks", "cards", "cards-by-deck", "card-detail", "cards-search"])
def test_private_flashcard_routes_exclude_foreign_owner(owners, route):
    with _client(owners.bob) as client:
        if route == "decks":
            response = client.get("/api/v1/flashcards/decks", params={"include_workspace_items": False})
            assert response.status_code == 200, response.text
            assert [row["id"] for row in response.json()] == [owners.bob_deck]
        elif route == "card-detail":
            response = client.get(f"/api/v1/flashcards/id/{owners.card}")
            assert response.status_code == 404, response.text
        else:
            params = {"include_workspace_items": False, "limit": 1, "offset": 0}
            if route == "cards-by-deck":
                params["deck_id"] = owners.deck
            elif route == "cards-search":
                params["q"] = "Citrine"
            response = client.get("/api/v1/flashcards", params=params)
            assert response.status_code == 200, response.text
            body = response.json()
            if route == "cards":
                assert body["total"] == 1
                assert [row["uuid"] for row in body["items"]] == [owners.bob_card]
            else:
                assert body["items"] == [] and body["total"] == 0


@pytest.mark.parametrize("operation", ["card-patch", "card-delete", "card-reset", "deck-patch", "deck-delete", "insert-foreign-deck", "card-tags"])
def test_foreign_flashcard_write_routes_preserve_private_rows(owners, operation):
    before_card = _raw_card(owners.alice, owners.card)
    before_deck = _raw_deck(owners.alice, owners.deck)
    before_cards = owners.alice.count_flashcards(deck_id=owners.deck)
    with _client(owners.bob) as client:
        if operation == "card-patch":
            response = client.patch(f"/api/v1/flashcards/{owners.card}", json={"front": "Unauthorized edit", "expected_version": 1})
        elif operation == "card-delete":
            response = client.delete(f"/api/v1/flashcards/{owners.card}", params={"expected_version": 1})
        elif operation == "card-reset":
            response = client.post(f"/api/v1/flashcards/{owners.card}/reset-scheduling", json={"expected_version": 1})
        elif operation == "deck-patch":
            response = client.patch(f"/api/v1/flashcards/decks/{owners.deck}", json={"description": "Unauthorized edit", "expected_version": 1})
        elif operation == "deck-delete":
            response = client.delete(f"/api/v1/flashcards/decks/{owners.deck}", params={"expected_version": 1})
        elif operation == "card-tags":
            response = client.patch(f"/api/v1/flashcards/{owners.card}", json={"tags": ["Unauthorized"], "expected_version": 1})
        else:
            response = client.post("/api/v1/flashcards", json={"deck_id": owners.deck, "front": "Injected card", "back": "Unauthorized"})
    assert _raw_card(owners.alice, owners.card) == before_card
    assert _raw_deck(owners.alice, owners.deck) == before_deck
    assert owners.alice.count_flashcards(deck_id=owners.deck) == before_cards
    assert 400 <= response.status_code < 500, response.text


def test_same_owner_reads_updates_and_workspace_filters_remain_available(owners):
    db = owners.bob
    db.upsert_workspace("bob-workspace", "Bob Workspace")
    workspace_deck = db.add_deck("Bob workspace deck", workspace_id="bob-workspace")
    workspace_card = db.add_flashcard({"deck_id": workspace_deck, "front": "Bob workspace", "back": "Owned"})
    with _client(db) as client:
        response = client.patch(f"/api/v1/flashcards/{owners.bob_card}", json={"front": "Owned edit", "expected_version": 1})
        assert response.status_code == 200, response.text
        assert response.json()["client_id"] == "3"
        default_cards = client.get("/api/v1/flashcards").json()
        assert owners.bob_card in [row["uuid"] for row in default_cards["items"]]
        assert workspace_card not in [row["uuid"] for row in default_cards["items"]]
        workspace_cards = client.get("/api/v1/flashcards", params={"workspace_id": "bob-workspace"}).json()
        assert [row["uuid"] for row in workspace_cards["items"]] == [workspace_card]
        all_cards = client.get("/api/v1/flashcards", params={"include_workspace_items": True}).json()
        assert workspace_card in [row["uuid"] for row in all_cards["items"]]


def test_foreign_workspace_id_alone_does_not_grant_private_deck_access(owners):
    owners.alice.upsert_workspace("alice-workspace", "Alice Workspace")
    deck = owners.alice.add_deck("Alice workspace deck", workspace_id="alice-workspace")
    owners.alice.add_flashcard({"deck_id": deck, "front": "Workspace private", "back": "Owner only"})
    with _client(owners.bob) as client:
        cards = client.get("/api/v1/flashcards", params={"workspace_id": "alice-workspace"})
        decks = client.get("/api/v1/flashcards/decks", params={"workspace_id": "alice-workspace"})
    assert cards.status_code == decks.status_code == 200
    assert cards.json()["items"] == [] and cards.json()["total"] == 0
    assert decks.json() == []


def test_explicit_deck_share_lookup_in_owner_repository_remains_available(owners):
    # The existing per-file sharing contract queries the owner's DB after access resolution.
    db = owners.alice
    db.upsert_deck_share(owners.deck, user_id=3, shared_by=2, role="viewer")
    assert [row["id"] for row in db.list_decks(shared_with_user_id=3)] == [owners.deck]
    assert db.get_deck_share(owners.deck, user_id=3)["role"] == "viewer"
    assert db.list_decks(shared_with_user_id=4) == []


def test_sqlite_per_user_file_preserves_sync_device_access(tmp_path):
    path = tmp_path / "owned.db"
    first = CharactersRAGDB(path, client_id="device-a")
    second = CharactersRAGDB(path, client_id="device-b")
    try:
        deck = first.add_deck("One user multiple devices")
        card = first.add_flashcard({"deck_id": deck, "front": "Original", "back": "Same owner"})
        assert second.get_deck(deck)["client_id"] == "device-a"
        assert second.update_flashcard(card, {"front": "Synced edit"}, expected_version=1)
        assert first.get_flashcard(card)["client_id"] == "device-b"
    finally:
        first.close_all_connections()
        second.close_all_connections()




def test_resource_edge_foreign_asset_http_content_is_not_readable(owners):
    asset = owners.alice.add_flashcard_asset(image_bytes=b"private image bytes", mime_type="image/png")
    with _client(owners.alice) as client:
        own = client.get(f"/api/v1/flashcards/assets/{asset}/content")
    assert own.status_code == 200 and own.content == b"private image bytes"
    with _client(owners.bob) as client:
        foreign = client.get(f"/api/v1/flashcards/assets/{asset}/content")
    assert foreign.status_code == 404, foreign.content


def test_resource_edge_foreign_asset_cannot_attach_to_owned_card(owners):
    asset = owners.alice.add_flashcard_asset(image_bytes=b"private image bytes", mime_type="image/png")
    before = owners.alice.get_flashcard_asset(asset)
    with _client(owners.bob) as client:
        response = client.patch(
            f"/api/v1/flashcards/{owners.bob_card}",
            json={"front": f"![private](flashcard-asset://{asset})", "expected_version": 1},
        )
    assert owners.alice.get_flashcard_asset(asset) == before
    assert 400 <= response.status_code < 500, response.text


def test_resource_edge_global_review_session_is_owner_local(owners):
    kwargs = {"deck_id": None, "review_mode": "due", "tag_filter": None, "scope_key": "due:global"}
    alice_session = owners.alice.get_or_create_flashcard_review_session(**kwargs)
    before = owners.alice.get_flashcard_review_session(alice_session["id"])
    bob_session = owners.bob.get_or_create_flashcard_review_session(**kwargs)
    assert bob_session["client_id"] == "3"
    assert owners.alice.get_flashcard_review_session(alice_session["id"]) == before
    assert [row["client_id"] for row in owners.bob.list_flashcard_review_sessions()] == ["3"]


def test_resource_edge_foreign_review_http_preserves_schedule(owners):
    before = _raw_card(owners.alice, owners.card)
    with _client(owners.bob) as client:
        response = client.post("/api/v1/flashcards/review", json={"card_uuid": owners.card, "rating": 3})
    assert _raw_card(owners.alice, owners.card) == before
    assert response.status_code == 404, response.text


def test_resource_edge_foreign_deck_parent_is_rejected(owners):
    with _client(owners.bob) as client:
        response = client.patch(
            f"/api/v1/flashcards/decks/{owners.bob_deck}",
            json={"parent_deck_id": owners.deck, "expected_version": 1},
        )
    assert owners.bob.get_deck(owners.bob_deck)["parent_deck_id"] is None
    assert 400 <= response.status_code < 500, response.text


def test_resource_edge_two_owners_can_use_the_same_deck_name(owners):
    name = owners.alice.get_deck(owners.deck)["name"]
    bob_deck = owners.bob.add_deck(name)
    assert owners.bob.get_deck(bob_deck)["client_id"] == "3"
    assert owners.alice.get_deck(owners.deck)["client_id"] == "2"


def test_resource_edge_owned_deck_undelete_preserves_identity(owners):
    owners.alice.soft_delete_deck_by_id(owners.deck, expected_version=1)
    restored = owners.alice.add_deck("Alice private Citrine")
    assert restored == owners.deck
    assert owners.alice.get_deck(restored)["client_id"] == "2"
    assert not owners.alice.get_deck(restored)["deleted"]


def test_tag_edge_same_text_resolves_in_current_owner_repository(owners):
    owners.alice.set_flashcard_tags(owners.card, ["Shared text"])
    alice_keywords = owners.alice.get_keywords_for_flashcard(owners.card)
    with _client(owners.bob) as client:
        response = client.patch(
            f"/api/v1/flashcards/{owners.bob_card}",
            json={"tags": ["Shared text"], "expected_version": 1},
        )
        assert response.status_code == 200, response.text
        keywords = client.get(f"/api/v1/flashcards/{owners.bob_card}/tags")
    assert keywords.status_code == 200, keywords.text
    assert [row["keyword"] for row in keywords.json()["items"]] == ["Shared text"]
    assert {row["client_id"] for row in keywords.json()["items"]} == {"3"}
    assert owners.alice.get_keywords_for_flashcard(owners.card) == alice_keywords
