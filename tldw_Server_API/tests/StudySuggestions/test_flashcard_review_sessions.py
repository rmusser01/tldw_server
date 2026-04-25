from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from loguru import logger
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints.flashcards import router as flashcards_router
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    ConflictError,
    InputError,
)
from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseConfig
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory


AUTH_HEADERS = {"X-API-KEY": "test-key"}


@pytest.fixture
def db(tmp_path):
    chacha = CharactersRAGDB(str(tmp_path / "flashcard-review-sessions.db"), client_id="flashcard-review-session-tests")
    try:
        yield chacha
    finally:
        chacha.close_connection()


@pytest.fixture
def client(db: CharactersRAGDB):
    from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user

    app = FastAPI()
    app.include_router(flashcards_router, prefix="/api/v1")

    def override_get_db():
        try:
            yield db
        finally:
            pass

    async def override_user():
        return User(
            id=1,
            username="testuser",
            email="test@example.com",
            is_active=True,
            roles=["admin"],
            is_admin=True,
        )

    app.dependency_overrides[get_chacha_db_for_user] = override_get_db
    app.dependency_overrides[get_request_user] = override_user
    with TestClient(app, headers=AUTH_HEADERS) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def _create_card(db: CharactersRAGDB, *, deck_name: str = "Session Deck") -> tuple[int, str]:
    deck_id = db.add_deck(deck_name, "desc")
    card_uuid = db.add_flashcard(
        {
            "deck_id": deck_id,
            "front": "Front",
            "back": "Back",
        }
    )
    return deck_id, card_uuid


def _create_cards(
    db: CharactersRAGDB,
    *,
    count: int,
    deck_name: str = "Session Deck",
) -> tuple[int, list[str]]:
    deck_id = db.add_deck(deck_name, "desc")
    card_uuids: list[str] = []
    for index in range(count):
        card_uuid = db.add_flashcard(
            {
                "deck_id": deck_id,
                "front": f"Front {index}",
                "back": f"Back {index}",
            }
        )
        card_uuids.append(card_uuid)
    return deck_id, card_uuids


def test_first_review_submission_in_scope_creates_active_session_and_links_review(db: CharactersRAGDB):
    deck_id, card_uuid = _create_card(db)

    session = db.get_or_create_flashcard_review_session(
        deck_id=deck_id,
        review_mode="due",
        tag_filter=None,
        scope_key=f"due:deck:{deck_id}",
    )

    assert session["status"] == "active"  # nosec B101

    updated = db.review_flashcard(
        card_uuid,
        rating=1,
        answer_time_ms=900,
        review_session_id=session["id"],
    )
    review_row = db.get_latest_flashcard_review(card_uuid)

    assert updated["review_session_id"] == session["id"]  # nosec B101
    assert review_row["review_session_id"] == session["id"]  # nosec B101


def test_stale_active_sessions_older_than_thirty_minutes_are_marked_abandoned(db: CharactersRAGDB):
    deck_id, _ = _create_card(db)

    stale = db.get_or_create_flashcard_review_session(
        deck_id=deck_id,
        review_mode="due",
        tag_filter=None,
        scope_key=f"due:deck:{deck_id}",
    )
    stale_at = (datetime.now(timezone.utc) - timedelta(minutes=31)).isoformat(timespec="seconds").replace("+00:00", "Z")
    db.execute_query(
        """
        UPDATE flashcard_review_sessions
           SET started_at = ?, last_activity_at = ?
         WHERE id = ?
        """,
        (stale_at, stale_at, stale["id"]),
    )

    refreshed = db.get_or_create_flashcard_review_session(
        deck_id=deck_id,
        review_mode="due",
        tag_filter=None,
        scope_key=f"due:deck:{deck_id}",
    )

    stale_row = db.execute_query(
        "SELECT status FROM flashcard_review_sessions WHERE id = ?",
        (stale["id"],),
    ).fetchone()

    assert refreshed["id"] != stale["id"]  # nosec B101
    assert refreshed["status"] == "active"  # nosec B101
    assert stale_row["status"] == "abandoned"  # nosec B101


def test_same_scope_duplicates_collapse_to_newest_active_session(db: CharactersRAGDB):
    deck_id, _ = _create_card(db)

    oldest = db.get_or_create_flashcard_review_session(
        deck_id=deck_id,
        review_mode="due",
        tag_filter=None,
        scope_key=f"due:deck:{deck_id}",
    )
    newer_started_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    db.execute_query(
        """
        INSERT INTO flashcard_review_sessions(
            deck_id,
            review_mode,
            tag_filter,
            scope_key,
            status,
            started_at,
            last_activity_at,
            client_id
        )
        VALUES(?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            deck_id,
            "due",
            None,
            f"due:deck:{deck_id}",
            "active",
            newer_started_at,
            newer_started_at,
            db.client_id,
        ),
    )
    newest = db.execute_query(
        """
        SELECT id, status
          FROM flashcard_review_sessions
         WHERE scope_key = ?
         ORDER BY id DESC
         LIMIT 1
        """,
        (f"due:deck:{deck_id}",),
    ).fetchone()

    authoritative = db.get_or_create_flashcard_review_session(
        deck_id=deck_id,
        review_mode="due",
        tag_filter=None,
        scope_key=f"due:deck:{deck_id}",
    )
    statuses = db.execute_query(
        """
        SELECT id, status
          FROM flashcard_review_sessions
         WHERE scope_key = ?
         ORDER BY id ASC
        """,
        (f"due:deck:{deck_id}",),
    ).fetchall()

    assert authoritative["id"] == newest["id"]  # nosec B101
    assert statuses[0]["id"] == oldest["id"]  # nosec B101
    assert statuses[0]["status"] == "abandoned"  # nosec B101
    assert statuses[-1]["id"] == newest["id"]  # nosec B101
    assert statuses[-1]["status"] == "active"  # nosec B101


def test_manual_end_marks_session_completed(db: CharactersRAGDB):
    deck_id, _ = _create_card(db)

    session = db.get_or_create_flashcard_review_session(
        deck_id=deck_id,
        review_mode="due",
        tag_filter=None,
        scope_key=f"due:deck:{deck_id}",
    )

    db.mark_flashcard_review_session_completed(session["id"])

    sessions = db.list_flashcard_review_sessions(deck_id=deck_id)

    assert sessions[0]["id"] == session["id"]  # nosec B101
    assert sessions[0]["status"] == "completed"  # nosec B101
    assert sessions[0]["completed_at"] is not None  # nosec B101


def test_review_flashcard_without_review_session_id_auto_creates_active_session(db: CharactersRAGDB):
    deck_id, card_uuid = _create_card(db, deck_name="Auto Session Deck")

    updated = db.review_flashcard(card_uuid, rating=4, answer_time_ms=700)
    review_row = db.get_latest_flashcard_review(card_uuid)
    sessions = db.list_flashcard_review_sessions(deck_id=deck_id)

    assert updated["review_session_id"] is not None  # nosec B101
    assert review_row["review_session_id"] == updated["review_session_id"]  # nosec B101
    assert sessions[0]["id"] == updated["review_session_id"]  # nosec B101
    assert sessions[0]["scope_key"] == f"due:deck:{deck_id}"  # nosec B101


def test_review_flashcard_rejects_unknown_or_wrong_deck_review_session_id(db: CharactersRAGDB):
    deck_id, card_uuid = _create_card(db, deck_name="Primary Session Deck")
    other_deck_id, _ = _create_card(db, deck_name="Other Session Deck")
    other_session = db.get_or_create_flashcard_review_session(
        deck_id=other_deck_id,
        review_mode="due",
        tag_filter=None,
        scope_key=f"due:deck:{other_deck_id}",
    )

    with pytest.raises(ConflictError, match="Flashcard review session not found"):
        db.review_flashcard(card_uuid, rating=4, answer_time_ms=500, review_session_id=999999)

    with pytest.raises(InputError, match="deck scope"):
        db.review_flashcard(card_uuid, rating=4, answer_time_ms=500, review_session_id=other_session["id"])

    sessions = db.list_flashcard_review_sessions(deck_id=deck_id)
    assert sessions == []  # nosec B101


def test_review_flashcard_updates_session_aggregates_and_rollup(db: CharactersRAGDB):
    deck_id, card_uuids = _create_cards(db, count=3, deck_name="Aggregate Session Deck")
    session = db.get_or_create_flashcard_review_session(
        deck_id=deck_id,
        review_mode="due",
        tag_filter=None,
        scope_key=f"due:deck:{deck_id}",
    )

    db.review_flashcard(card_uuids[0], rating=4, answer_time_ms=600, review_session_id=session["id"])
    db.review_flashcard(card_uuids[1], rating=1, answer_time_ms=750, review_session_id=session["id"])
    db.review_flashcard(card_uuids[2], rating=3, answer_time_ms=900, review_session_id=session["id"])

    session_row = db.get_flashcard_review_session(session["id"])
    rollup = db.get_flashcard_review_session_rollup(session["id"])

    assert session_row["cards_reviewed"] == 3  # nosec B101
    assert session_row["correct_count"] == 2  # nosec B101
    assert rollup["cards_reviewed"] == 3  # nosec B101
    assert rollup["correct_count"] == 2  # nosec B101
    assert rollup["aggregate_source"] in {"session", "reconstructed"}  # nosec B101


def test_flashcard_review_session_rollup_reconstructs_missing_aggregates(db: CharactersRAGDB):
    deck_id, card_uuids = _create_cards(db, count=3, deck_name="Legacy Aggregate Deck")
    session = db.get_or_create_flashcard_review_session(
        deck_id=deck_id,
        review_mode="due",
        tag_filter=None,
        scope_key=f"due:deck:{deck_id}",
    )

    db.review_flashcard(card_uuids[0], rating=4, answer_time_ms=600, review_session_id=session["id"])
    db.review_flashcard(card_uuids[1], rating=2, answer_time_ms=750, review_session_id=session["id"])
    db.review_flashcard(card_uuids[2], rating=3, answer_time_ms=900, review_session_id=session["id"])
    db.execute_query(
        """
        UPDATE flashcard_review_sessions
           SET cards_reviewed = NULL,
               correct_count = NULL
         WHERE id = ?
        """,
        (session["id"],),
        commit=True,
    )

    rollup = db.get_flashcard_review_session_rollup(session["id"], repair_session_aggregates=False)

    assert rollup["cards_reviewed"] == 3  # nosec B101
    assert rollup["correct_count"] == 2  # nosec B101
    assert rollup["aggregate_source"] == "reconstructed"  # nosec B101


def test_flashcard_review_session_rollup_reconstructs_impossible_aggregates(db: CharactersRAGDB):
    deck_id, card_uuids = _create_cards(db, count=3, deck_name="Impossible Aggregate Deck")
    session = db.get_or_create_flashcard_review_session(
        deck_id=deck_id,
        review_mode="due",
        tag_filter=None,
        scope_key=f"due:deck:{deck_id}",
    )

    db.review_flashcard(card_uuids[0], rating=4, answer_time_ms=600, review_session_id=session["id"])
    db.review_flashcard(card_uuids[1], rating=1, answer_time_ms=750, review_session_id=session["id"])
    db.review_flashcard(card_uuids[2], rating=3, answer_time_ms=900, review_session_id=session["id"])
    db.execute_query(
        """
        UPDATE flashcard_review_sessions
           SET cards_reviewed = 2,
               correct_count = 3
         WHERE id = ?
        """,
        (session["id"],),
        commit=True,
    )

    rollup = db.get_flashcard_review_session_rollup(session["id"], repair_session_aggregates=False)

    assert rollup["cards_reviewed"] == 3  # nosec B101
    assert rollup["correct_count"] == 2  # nosec B101
    assert rollup["aggregate_source"] == "reconstructed"  # nosec B101


def test_flashcard_review_session_rollup_repairs_stale_aggregates_from_reviews(db: CharactersRAGDB):
    deck_id, card_uuids = _create_cards(db, count=3, deck_name="Repair Aggregate Deck")
    session = db.get_or_create_flashcard_review_session(
        deck_id=deck_id,
        review_mode="due",
        tag_filter=None,
        scope_key=f"due:deck:{deck_id}",
    )

    db.review_flashcard(card_uuids[0], rating=4, answer_time_ms=600, review_session_id=session["id"])
    db.review_flashcard(card_uuids[1], rating=1, answer_time_ms=750, review_session_id=session["id"])
    db.review_flashcard(card_uuids[2], rating=3, answer_time_ms=900, review_session_id=session["id"])
    db.execute_query(
        """
        UPDATE flashcard_review_sessions
           SET cards_reviewed = 17,
               correct_count = 1
         WHERE id = ?
        """,
        (session["id"],),
        commit=True,
    )

    rollup = db.get_flashcard_review_session_rollup(session["id"], repair_session_aggregates=True)
    repaired = db.get_flashcard_review_session(session["id"])

    assert rollup["cards_reviewed"] == 3  # nosec B101
    assert rollup["correct_count"] == 2  # nosec B101
    assert rollup["aggregate_source"] == "reconstructed"  # nosec B101
    assert repaired["cards_reviewed"] == 3  # nosec B101
    assert repaired["correct_count"] == 2  # nosec B101


def test_flashcard_review_session_rollup_reload_after_conditional_repair_race(
    db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
):
    deck_id, card_uuids = _create_cards(db, count=4, deck_name="Repair Race Deck")
    session = db.get_or_create_flashcard_review_session(
        deck_id=deck_id,
        review_mode="due",
        tag_filter=None,
        scope_key=f"due:deck:{deck_id}",
    )

    db.review_flashcard(card_uuids[0], rating=4, answer_time_ms=600, review_session_id=session["id"])
    db.review_flashcard(card_uuids[1], rating=1, answer_time_ms=750, review_session_id=session["id"])
    db.review_flashcard(card_uuids[2], rating=3, answer_time_ms=900, review_session_id=session["id"])
    db.execute_query(
        """
        UPDATE flashcard_review_sessions
           SET cards_reviewed = 17,
               correct_count = 1
         WHERE id = ?
        """,
        (session["id"],),
        commit=True,
    )

    def _lose_repair_race(conn, session_id, *, expected_cards_reviewed, expected_correct_count, repaired_cards_reviewed, repaired_correct_count):
        conn.execute(
            """
            INSERT INTO flashcard_reviews(
                card_id, reviewed_at, rating, answer_time_ms, scheduled_interval_days,
                new_ef, new_repetitions, was_lapse, client_id, scheduler_type,
                previous_queue_state, next_queue_state, previous_due_at, next_due_at, review_session_id
            )
            SELECT
                id, ?, 4, 500, NULL,
                2.6, 2, 0, ?, 'sm2_plus',
                'learning', 'review', NULL, ?, ?
              FROM flashcards
             WHERE uuid = ?
            """,
            (
                datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
                db.client_id,
                datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
                session_id,
                card_uuids[3],
            ),
        )
        conn.execute(
            """
            UPDATE flashcard_review_sessions
               SET cards_reviewed = 4,
                   correct_count = 3
             WHERE id = ?
            """,
            (session_id,),
        )
        return False

    monkeypatch.setattr(
        db,
        "_repair_flashcard_review_session_aggregates_if_unchanged",
        _lose_repair_race,
    )

    rollup = db.get_flashcard_review_session_rollup(session["id"], repair_session_aggregates=True)
    repaired = db.get_flashcard_review_session(session["id"])

    assert rollup["cards_reviewed"] == 4  # nosec B101
    assert rollup["correct_count"] == 3  # nosec B101
    assert rollup["aggregate_source"] == "session"  # nosec B101
    assert repaired["cards_reviewed"] == 4  # nosec B101
    assert repaired["correct_count"] == 3  # nosec B101


def test_legacy_rows_with_null_review_session_id_remain_readable(db: CharactersRAGDB):
    deck_id, card_uuid = _create_card(db)
    card_row = db.get_flashcard(card_uuid)
    assert deck_id == card_row["deck_id"]  # nosec B101

    reviewed_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    db.execute_query(
        """
        INSERT INTO flashcard_reviews(
            card_id,
            reviewed_at,
            rating,
            answer_time_ms,
            scheduled_interval_days,
            new_ef,
            new_repetitions,
            was_lapse,
            client_id,
            scheduler_type,
            previous_queue_state,
            next_queue_state,
            previous_due_at,
            next_due_at,
            review_session_id
        )
        SELECT
            id,
            ?,
            3,
            850,
            NULL,
            2.5,
            1,
            0,
            ?,
            'sm2_plus',
            'new',
            'learning',
            NULL,
            ?,
            NULL
          FROM flashcards
         WHERE uuid = ?
        """,
        (reviewed_at, db.client_id, reviewed_at, card_uuid),
    )

    review_row = db.get_latest_flashcard_review(card_uuid)

    assert review_row["rating"] == 3  # nosec B101
    assert review_row["review_session_id"] is None  # nosec B101


def test_review_endpoint_returns_review_session_id_and_persists_linkage(
    client: TestClient,
    db: CharactersRAGDB,
):
    deck_id, card_uuid = _create_card(db, deck_name="Endpoint Session Deck")

    response = client.post(
        "/api/v1/flashcards/review",
        json={"card_uuid": card_uuid, "rating": 4, "answer_time_ms": 1200},
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 200  # nosec B101
    payload = response.json()
    review_row = db.get_latest_flashcard_review(card_uuid)
    sessions = db.list_flashcard_review_sessions(deck_id=deck_id)

    assert payload["review_session_id"] == review_row["review_session_id"]  # nosec B101
    assert payload["review_session_id"] == sessions[0]["id"]  # nosec B101
    assert sessions[0]["scope_key"] == f"due:deck:{deck_id}"  # nosec B101


@pytest.mark.integration
def test_flashcard_review_sessions_postgres_round_trip(pg_database_config: DatabaseConfig):
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(db_path=":memory:", client_id="flashcard-review-session-pg", backend=backend)

    try:
        deck_id, card_uuid = _create_card(db, deck_name="Postgres Session Deck")
        session = db.get_or_create_flashcard_review_session(
            deck_id=deck_id,
            review_mode="due",
            tag_filter=None,
            scope_key=f"due:deck:{deck_id}",
        )

        updated = db.review_flashcard(
            card_uuid,
            rating=4,
            answer_time_ms=1100,
            review_session_id=session["id"],
        )
        review_row = db.get_latest_flashcard_review(card_uuid)
        sessions = db.list_flashcard_review_sessions(deck_id=deck_id)

        assert updated["review_session_id"] == session["id"]  # nosec B101
        assert review_row["review_session_id"] == session["id"]  # nosec B101
        assert sessions[0]["id"] == session["id"]  # nosec B101
    finally:
        try:
            db.close_connection()
            if db.backend_type.name == "POSTGRESQL":
                db.backend.get_pool().close_all()
        except Exception as e:
            logger.warning(f"Cleanup error: {e}")
