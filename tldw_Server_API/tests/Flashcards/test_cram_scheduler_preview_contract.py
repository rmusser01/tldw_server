"""Cram's list projection previews the same schedule that an actual rating saves."""

from datetime import datetime
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints import flashcards as endpoint
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=pytest.mark.postgres)])
def cram_db(request, tmp_path):
    """Use fixture-owned SQLite or the official isolated PostgreSQL database."""
    backend = None
    if request.param == "postgres":
        backend = DatabaseBackendFactory.create_backend(request.getfixturevalue("pg_database_config"))
    db = CharactersRAGDB(tmp_path / "cram-preview.db", client_id="1", backend=backend)
    try:
        yield db
    finally:
        db.close_connection()
        if backend is not None:
            backend.get_pool().close_all()


def _client(db):
    app = FastAPI()
    app.include_router(endpoint.router, prefix="/api/v1")
    app.dependency_overrides[endpoint.get_chacha_db_for_user] = lambda: db
    return TestClient(app, raise_server_exceptions=False)


def _minutes(body):
    def parse(value):
        return datetime.fromisoformat(value.replace("Z", "+00:00"))

    return (parse(body["due_at"]) - parse(body["last_reviewed_at"])).total_seconds() / 60


@pytest.mark.integration
@pytest.mark.parametrize(
    ("rating", "label", "preview", "minutes"),
    [(0, "again", "1 min", 1), (2, "hard", "6 min", 6), (3, "good", "10 min", 10), (5, "easy", "4 days", 5760)],
)
def test_cram_list_preview_matches_new_card_committed_schedule(cram_db, rating, label, preview, minutes):
    db = cram_db
    deck = db.add_deck("Cram preview new card")
    uuid = db.add_flashcard({"deck_id": deck, "front": "Question", "back": "Answer"})
    before = db.get_flashcard(uuid)
    db.close_connection()
    with _client(db) as client:
        listed = client.get(
            "/api/v1/flashcards",
            params={"deck_id": deck, "due_status": "all", "include_scheduler_preview": True},
        )
        due = client.get("/api/v1/flashcards/review/next", params={"deck_id": deck})
        assert listed.status_code == due.status_code == 200
        assert listed.json()["count"] == 1
        card = listed.json()["items"][0]
        assert card["uuid"] == due.json()["card"]["uuid"] == uuid
        assert due.json()["card"]["next_intervals"][label] == preview
        assert db.get_flashcard(uuid) == before
        assert db.list_flashcard_review_sessions() == []

        # Inspect the actual saved schedule even while the old list projection
        # is wrong. Never retry a failed assertion by rating this card twice.
        response = client.post(
            "/api/v1/flashcards/review",
            json={"card_uuid": uuid, "rating": rating, "review_context": {"review_mode": "cram", "deck_id": deck}},
        )
        assert response.status_code == 200, response.text
        assert _minutes(response.json()) == minutes
        assert db.get_flashcard(uuid)["version"] == before["version"] + 1
        assert card["next_intervals"] == due.json()["card"]["next_intervals"]
        assert card["scheduler_type"] == "sm2_plus"


@pytest.mark.integration
@pytest.mark.parametrize("scheduler", ["custom-sm2", "fsrs"])
def test_cram_list_uses_each_decks_actual_scheduler_settings(cram_db, scheduler):
    db = cram_db
    settings = {"sm2_plus": {"new_steps_minutes": [2, 25]}}
    if scheduler == "fsrs":
        settings = {"fsrs": {"enable_fuzz": False, "target_retention": 0.85}}
    deck = db.add_deck(
        "Configured preview", scheduler_type="fsrs" if scheduler == "fsrs" else "sm2_plus", scheduler_settings=settings
    )
    card_data = {"deck_id": deck, "front": "Configured question", "back": "Answer"}
    if scheduler == "fsrs":
        card_data.update(
            queue_state="review",
            interval_days=12,
            repetitions=7,
            last_reviewed_at="2026-03-01T00:00:00Z",
            due_at="2026-03-13T00:00:00Z",
        )
    uuid = db.add_flashcard(card_data)
    before = db.get_flashcard(uuid)
    db.close_connection()
    with _client(db) as client:
        params = {"deck_id": deck, "include_scheduler_preview": True}
        listed = client.get("/api/v1/flashcards", params=params)
        due = client.get("/api/v1/flashcards/review/next", params={"deck_id": deck})
        assert listed.status_code == due.status_code == 200
        assert db.get_flashcard(uuid) == before
        actual = client.post("/api/v1/flashcards/review", json={"card_uuid": uuid, "rating": 3})
        assert actual.status_code == 200, actual.text
        if scheduler == "custom-sm2":
            assert due.json()["card"]["next_intervals"]["good"] == "25 min"
            assert _minutes(actual.json()) == 25
        else:
            assert due.json()["card"]["scheduler_type"] == "fsrs"
            days = int(_minutes(actual.json()) / 1440)
            assert due.json()["card"]["next_intervals"]["good"] == f"{days} days"
        assert listed.json()["items"][0]["next_intervals"] == due.json()["card"]["next_intervals"]
        assert listed.json()["items"][0]["scheduler_type"] == due.json()["card"]["scheduler_type"]


@pytest.mark.integration
@pytest.mark.parametrize("opt_in", [None, False])
def test_ordinary_list_remains_unenriched_and_read_only(cram_db, opt_in):
    db = cram_db
    uuid = db.add_flashcard({"front": "Ordinary list", "back": "Answer"})
    before = db.get_flashcard(uuid)
    db.close_connection()
    with _client(db) as client:
        params = {} if opt_in is None else {"include_scheduler_preview": opt_in}
        response = client.get("/api/v1/flashcards", params=params)
    assert response.status_code == 200
    body = response.json()
    assert body["items"][0]["uuid"] == uuid
    assert body["items"][0]["scheduler_type"] is None
    assert body["items"][0]["next_intervals"] is None
    assert db.get_flashcard(uuid) == before
    assert db.list_flashcard_review_sessions() == []


@pytest.mark.integration
def test_preview_preserves_mixed_deck_page_and_reads_each_deck_once(cram_db, monkeypatch):
    db = cram_db
    first = db.add_deck("Default steps")
    second = db.add_deck("Custom steps", scheduler_settings={"sm2_plus": {"new_steps_minutes": [2, 25]}})
    uuids = [
        db.add_flashcard(
            {
                "deck_id": deck,
                "front": f"Card {index}",
                "back": "Answer",
                "due_at": f"2026-09-{index + 1:02d}T00:00:00Z",
            }
        )
        for index, deck in enumerate([first, first, first, second])
    ]
    before = {uuid: db.get_flashcard(uuid) for uuid in uuids}
    lookup = Mock(wraps=db.get_deck)
    monkeypatch.setattr(db, "get_deck", lookup)
    db.close_connection()
    with _client(db) as client:
        params = {"limit": 3, "offset": 1, "order_by": "due_at", "due_status": "all"}
        ordinary = client.get("/api/v1/flashcards", params=params)
        assert ordinary.status_code == 200
        lookup.assert_not_called()
        preview = client.get("/api/v1/flashcards", params={**params, "include_scheduler_preview": True})
    assert preview.status_code == 200
    body = preview.json()
    assert {key: value for key, value in body.items() if key != "items"} == {
        key: value for key, value in ordinary.json().items() if key != "items"
    }
    assert [card["uuid"] for card in body["items"]] == uuids[1:]
    assert {uuid: db.get_flashcard(uuid) for uuid in uuids} == before
    assert db.list_flashcard_review_sessions() == []
    assert [card["next_intervals"]["good"] for card in body["items"]] == ["10 min", "10 min", "25 min"]
    assert [call.args for call in lookup.call_args_list] == [(first,), (second,)]


@pytest.mark.integration
@pytest.mark.parametrize("failure", ["sm2-settings", "fsrs-settings", "database"])
def test_preview_errors_do_not_change_ordinary_listing_or_card_state(cram_db, monkeypatch, failure):
    db = cram_db
    deck = db.add_deck("Preview error")
    uuid = db.add_flashcard({"deck_id": deck, "front": "Keep visible", "back": "Answer", "queue_state": "review"})
    before = db.get_flashcard(uuid)
    get_deck = db.get_deck

    def fail_lookup(deck_id):
        if failure == "database":
            raise CharactersRAGDBError("controlled deck read failure")
        result = get_deck(deck_id)
        if failure == "sm2-settings":
            return {
                **result,
                "scheduler_type": "sm2_plus",
                "scheduler_settings_json": {"sm2_plus": {"new_steps_minutes": [0]}},
            }
        return {**result, "scheduler_type": "fsrs", "scheduler_settings_json": {"fsrs": {"target_retention": 2}}}

    monkeypatch.setattr(db, "get_deck", fail_lookup)
    db.close_connection()
    with _client(db) as client:
        for params in ({}, {"include_scheduler_preview": False}):
            ordinary = client.get("/api/v1/flashcards", params=params)
            assert ordinary.status_code == 200
            assert ordinary.json()["items"][0]["uuid"] == uuid
        preview = client.get("/api/v1/flashcards", params={"include_scheduler_preview": True})
    assert preview.status_code == (500 if failure == "database" else 400), preview.text
    assert isinstance(preview.json()["detail"], str)
    assert db.get_flashcard(uuid) == before
    assert db.list_flashcard_review_sessions() == []


@pytest.mark.integration
def test_empty_opt_in_page_does_not_lookup_decks(cram_db, monkeypatch):
    lookup = Mock(side_effect=AssertionError("No cards require a deck read"))
    monkeypatch.setattr(cram_db, "get_deck", lookup)
    cram_db.close_connection()
    with _client(cram_db) as client:
        response = client.get("/api/v1/flashcards", params={"include_scheduler_preview": True})
    assert response.status_code == 200
    assert response.json()["items"] == []
    assert response.json()["total"] == 0
    lookup.assert_not_called()
