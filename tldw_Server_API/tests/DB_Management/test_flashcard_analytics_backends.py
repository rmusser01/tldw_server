"""Study analytics preserve UTC metrics and caller-owned transactions."""

from datetime import datetime, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.DB_Management import ChaChaNotes_DB as database_module
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def fixed_analytics_clock(monkeypatch):
    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 9, 16, 12, tzinfo=timezone.utc).astimezone(tz)

    monkeypatch.setattr(database_module, "datetime", FixedDatetime)


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=pytest.mark.postgres)])
def analytics_db(request, tmp_path):
    backend = None
    if request.param == "postgres":
        backend = DatabaseBackendFactory.create_backend(request.getfixturevalue("pg_database_config"))
    db = CharactersRAGDB(tmp_path / "analytics.db", client_id="1", backend=backend)
    try:
        yield db
    finally:
        db.close_connection()
        if backend is not None:
            backend.get_pool().close_all()


@pytest.fixture
def pg_analytics(pg_database_config, tmp_path):
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(tmp_path / "analytics.db", client_id="1", backend=backend)
    try:
        yield db
    finally:
        db.close_connection()
        backend.get_pool().close_all()


def _seed_review(db, card_uuid, reviewed_at="2026-09-16T00:30:00Z", rating=3, answer_time_ms=2000, was_lapse=False):
    with db.transaction() as conn:
        conn.execute(
            """INSERT INTO flashcard_reviews(card_id, reviewed_at, rating, answer_time_ms, client_id, was_lapse)
               SELECT id, ?, ?, ?, ?, ? FROM flashcards WHERE uuid = ?""",
            (reviewed_at, rating, answer_time_ms, db.client_id, was_lapse, card_uuid),
        )


def _saved_card(db):
    deck = db.add_deck("Rowan")
    card = db.add_flashcard({"deck_id": deck, "front": "Marker?", "back": "Amber"})
    return deck, card


@pytest.mark.parametrize("populated", [False, True], ids=["empty-history", "populated-history"])
def test_saved_card_analytics_http_and_following_reads(analytics_db, populated):
    from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
    from tldw_Server_API.app.api.v1.endpoints.flashcards import router

    db = analytics_db
    deck, card = _saved_card(db)
    if populated:
        _seed_review(db, card)
        _seed_review(db, card, "2026-09-16T00:45:00Z", rating=1, answer_time_ms=4000, was_lapse=True)
        _seed_review(db, card, "2026-09-15T00:30:00Z")
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    app.dependency_overrides[get_chacha_db_for_user] = lambda: db
    with TestClient(app) as client:
        response = client.get("/api/v1/flashcards/analytics/summary")
        assert response.status_code == 200
        result = response.json()
        assert result["reviewed_today"] == (2 if populated else 0)
        assert result["study_streak_days"] == (2 if populated else 0)
        assert result["retention_rate_today"] == (50.0 if populated else None)
        assert result["lapse_rate_today"] == (50.0 if populated else None)
        assert result["avg_answer_time_ms_today"] == (3000.0 if populated else None)
        assert result["decks"][0]["total"] == 1
        assert client.get("/api/v1/flashcards/decks").status_code == 200
        queue = client.get("/api/v1/flashcards", params={"deck_id": deck, "due_status": "all", "order_by": "due_at"})
        assert queue.status_code == 200
        assert [row["uuid"] for row in queue.json()["items"]] == [card]
        assert client.get("/api/v1/flashcards/review-sessions", params={"status": "completed"}).json() == []


def test_analytics_preserves_visibility_and_deleted_filters(analytics_db):
    db = analytics_db
    db.upsert_workspace("ws-rowan", "Workspace")
    global_deck, global_card = _saved_card(db)
    other_deck = db.add_deck("Citrine")
    workspace_deck = db.add_deck("Workspace deck", workspace_id="ws-rowan")
    deleted_deck = db.add_deck("Deleted deck")
    cards = [global_card]
    for deck in [other_deck, workspace_deck, deleted_deck, global_deck]:
        cards.append(db.add_flashcard({"deck_id": deck, "front": "Scope?", "back": "Fixture"}))
    for card in cards:
        _seed_review(db, card)
    # Seed historical deletion state without exercising independent delete endpoints.
    with db.transaction() as conn:
        conn.execute("UPDATE flashcards SET deleted = ? WHERE uuid = ?", (True, cards[-1]))
        conn.execute("UPDATE decks SET deleted = ? WHERE id = ?", (True, deleted_deck))
    cases = [
        ({}, {global_deck, other_deck}),
        ({"deck_id": global_deck}, {global_deck}),
        ({"workspace_id": "ws-rowan"}, {workspace_deck}),
        ({"include_workspace_items": True}, {global_deck, other_deck, workspace_deck}),
    ]
    for filters, expected_decks in cases:
        result = db.get_flashcard_analytics_summary(**filters)
        assert result["reviewed_today"] == len(expected_decks)
        assert result["study_streak_days"] == 1
        assert {row["deck_id"] for row in result["decks"]} == expected_decks
        assert all(row["total"] == 1 for row in result["decks"])


@pytest.mark.parametrize("session_timezone", ["Pacific/Honolulu", "Pacific/Kiritimati"])
def test_postgres_streak_uses_utc_days_with_offset_input(pg_analytics, session_timezone):
    db = pg_analytics
    _, card = _saved_card(db)
    # One hour apart across UTC midnight; both are Sep15 in Honolulu local time.
    _seed_review(db, card, "2026-09-16T14:30:00+14:00")
    _seed_review(db, card, "2026-09-15T13:30:00-10:00")
    raw = db._get_thread_connection()
    raw.execute("SELECT set_config('TimeZone', %s, false)", (session_timezone,))
    raw.commit()
    result = db.get_flashcard_analytics_summary()
    assert result["reviewed_today"] == 1
    assert result["study_streak_days"] == 2
    assert raw.info.transaction_status.name == "IDLE"


@pytest.mark.parametrize("failed_select", ["reviewed_today", "review_day", "total_count"])
def test_failed_owned_analytics_read_recovers_without_poisoning_queue(pg_analytics, monkeypatch, failed_select):
    db = pg_analytics
    deck, card = _saved_card(db)
    original_execute = db.backend.execute

    def fail_selected_query(query, *args, **kwargs):
        if f"AS {failed_select}" in query:
            query = "SELECT uat177_missing_column FROM flashcard_reviews"
            # Drop the original query's parameters for the real invalid SELECT.
            args = ((), *args[1:]) if args else args
        return original_execute(query, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(db.backend, "execute", fail_selected_query)
        with pytest.raises(CharactersRAGDBError):
            db.get_flashcard_analytics_summary()
    assert db._get_thread_connection().info.transaction_status.name == "IDLE"
    assert db.get_flashcard_analytics_summary()["reviewed_today"] == 0
    assert [row["uuid"] for row in db.list_flashcards(deck_id=deck)] == [card]
    assert len(db.list_decks()) == 1
    assert db.list_flashcard_review_sessions(status="completed") == []


@pytest.mark.parametrize("explicit", [False, True], ids=["implicit-owner", "explicit-owner"])
@pytest.mark.parametrize("commit", [False, True], ids=["rollback", "commit"])
def test_analytics_preserves_caller_writes(pg_analytics, explicit, commit):
    from contextlib import nullcontext

    db = pg_analytics
    deck, _ = _saved_card(db)
    raw = db._get_thread_connection()
    with db.transaction() if explicit else nullcontext():
        db.execute_query("UPDATE decks SET name = ? WHERE id = ?", ("Pending", deck))
        assert db.get_flashcard_analytics_summary()["decks"][0]["deck_name"] == "Pending"
        assert raw.info.transaction_status.name == "INTRANS"
        assert db.backend.execute("SELECT name FROM decks WHERE id = %s", (deck,)).scalar == "Rowan"
        if commit:
            raw.commit()
        else:
            raw.rollback()
    expected = "Pending" if commit else "Rowan"
    assert db.backend.execute("SELECT name FROM decks WHERE id = %s", (deck,)).scalar == expected


@pytest.mark.parametrize("scheduler", ["sm2_plus", "fsrs"])
@pytest.mark.parametrize("queue", ["new", "review"])
@pytest.mark.parametrize("rating", range(6))
def test_actual_review_metrics_match_scheduler_outcome(analytics_db, scheduler, queue, rating):
    """Hard is recall; Again while learning is incorrect without a mature lapse."""
    db = analytics_db
    deck = db.add_deck("Recall metrics", scheduler_type=scheduler)
    card = db.add_flashcard({
        "deck_id": deck, "front": "Recall?", "back": "Recorded outcome",
        "queue_state": queue, "interval_days": 10 if queue == "review" else 0,
        "repetitions": 3 if queue == "review" else 0,
        "last_reviewed_at": "2026-09-15T12:00:00Z" if queue == "review" else None,
        "due_at": "2026-09-16T12:00:00Z",
    })
    updated = db.review_flashcard(card, rating, answer_time_ms=700)
    review = db.get_latest_flashcard_review(card)
    expected_lapse = rating == 0 and queue == "review"
    assert bool(review["was_lapse"]) is expected_lapse
    assert updated["lapses"] == int(expected_lapse)
    summary = db.get_flashcard_analytics_summary(deck_id=deck)
    assert summary["reviewed_today"] == 1
    assert summary["lapse_rate_today"] == (100.0 if expected_lapse else 0.0)
    assert summary["retention_rate_today"] == (0.0 if expected_lapse else 100.0)
    session_id = updated["review_session_id"]
    expected_correct = int(rating != 0)
    session = db.get_flashcard_review_session(session_id)
    assert session["correct_count"] == expected_correct
    with db.transaction() as conn:
        conn.execute(
            "UPDATE flashcard_review_sessions SET cards_reviewed = 99, correct_count = 99 WHERE id = ?",
            (session_id,),
        )
    db.get_flashcard_review_session_rollup(session_id, repair_session_aggregates=True)
    repaired = db.get_flashcard_review_session(session_id)
    assert (repaired["cards_reviewed"], repaired["correct_count"]) == (1, expected_correct)


@pytest.mark.parametrize("rating,stored_lapse", [(0, False), (1, True), (2, False), (3, True)])
def test_historical_metrics_trust_recorded_outcome(analytics_db, rating, stored_lapse):
    """Historical outcomes are not reinterpreted with current rating thresholds."""
    db = analytics_db
    _, card = _saved_card(db)
    _seed_review(db, card, rating=rating, was_lapse=stored_lapse)
    summary = db.get_flashcard_analytics_summary()
    assert summary["lapse_rate_today"] == (100.0 if stored_lapse else 0.0)


def test_repeated_scheduled_hard_reviews_preserve_recall_metrics(analytics_db):
    db = analytics_db
    _, card = _saved_card(db)
    for rating in [5, 3, 2, 2]:
        updated = db.review_flashcard(card, rating)
    summary = db.get_flashcard_analytics_summary()
    assert summary["reviewed_today"] == 4
    assert summary["lapse_rate_today"] == 0.0
    assert summary["retention_rate_today"] == 100.0
    session = db.get_flashcard_review_session(updated["review_session_id"])
    assert (session["cards_reviewed"], session["correct_count"]) == (4, 4)


@pytest.mark.parametrize("rating,was_lapse,correct", [(1, False, 1), (3, True, 0), (0, False, 0)])
def test_historical_session_reconstruction_matches_recorded_recall(analytics_db, rating, was_lapse, correct):
    db = analytics_db
    deck, card = _saved_card(db)
    session = db.get_or_create_flashcard_review_session(
        deck_id=deck, review_mode="due", tag_filter=None, scope_key=f"due:deck:{deck}"
    )
    _seed_review(db, card, rating=rating, was_lapse=was_lapse)
    with db.transaction() as conn:
        conn.execute("UPDATE flashcard_reviews SET review_session_id = ?", (session["id"],))
    rollup = db.get_flashcard_review_session_rollup(session["id"])
    assert (rollup["cards_reviewed"], rollup["correct_count"]) == (1, correct)


def test_lapse_metrics_keep_utc_day_boundaries(analytics_db):
    db = analytics_db
    _, card = _saved_card(db)
    _seed_review(db, card, "2026-09-15T23:59:59Z", rating=0, was_lapse=True)
    _seed_review(db, card, "2026-09-16T00:00:00Z", rating=2, was_lapse=False)
    _seed_review(db, card, "2026-09-16T23:59:59Z", rating=0, was_lapse=True)
    _seed_review(db, card, "2026-09-17T00:00:00Z", rating=0, was_lapse=True)
    summary = db.get_flashcard_analytics_summary()
    assert summary["reviewed_today"] == 2
    assert summary["lapse_rate_today"] == 50.0


def test_postgres_lapse_metrics_exclude_other_owner(pg_analytics, tmp_path):
    db = pg_analytics
    _, card = _saved_card(db)
    db.review_flashcard(card, 2)
    other = CharactersRAGDB(tmp_path / "other-owner.db", client_id="2", backend=db.backend)
    try:
        _, other_card = _saved_card(other)
        other.review_flashcard(other_card, 5)
        other.review_flashcard(other_card, 0)
        own_summary = db.get_flashcard_analytics_summary()
        other_summary = other.get_flashcard_analytics_summary()
        assert (own_summary["reviewed_today"], own_summary["lapse_rate_today"]) == (1, 0.0)
        assert (other_summary["reviewed_today"], other_summary["lapse_rate_today"]) == (2, 50.0)
    finally:
        other.close_connection()
