"""Scheduled review responses serialize committed PostgreSQL and SQLite state."""

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from tldw_Server_API.app.api.v1.endpoints import flashcards as endpoint
from tldw_Server_API.app.api.v1.schemas.flashcards import FlashcardReviewResponse
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=pytest.mark.postgres)])
def review_db(request, tmp_path):
    """Use the official isolated PostgreSQL fixture and a real SQLite control."""
    backend = None
    if request.param == "postgres":
        backend = DatabaseBackendFactory.create_backend(request.getfixturevalue("pg_database_config"))
    db = CharactersRAGDB(tmp_path / "review.db", client_id="1", backend=backend)
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


def _review_rows(db, card_uuid):
    return db.execute_query(
        "SELECT fr.* FROM flashcard_reviews fr JOIN flashcards f ON f.id = fr.card_id WHERE f.uuid = ?",
        (card_uuid,),
        read_only=True,
    ).fetchall()


def _time(value):
    return value if isinstance(value, datetime) else datetime.fromisoformat(value.replace("Z", "+00:00"))


@pytest.mark.integration
@pytest.mark.parametrize("mode", ["due", "cram"])
def test_review_http_serializes_exact_committed_card_and_session(review_db, mode):
    db = review_db
    deck = db.add_deck("Citrine review response")
    card_uuid = db.add_flashcard(
        {
            "deck_id": deck,
            "front": "Meeting time?",
            "back": "Tuesday at 14:00",
            "source_ref_type": "note",
            "source_ref_id": "review-source",
        }
    )
    before = db.get_flashcard(card_uuid)
    db.close_connection()
    with _client(db) as client:
        response = client.post(
            "/api/v1/flashcards/review",
            json={
                "card_uuid": card_uuid,
                "rating": 3,
                "answer_time_ms": 1200,
                "review_context": {"review_mode": mode, "deck_id": deck, "tag_filter": None},
            },
        )
        # A failed response is not permission to submit another rating. Inspect
        # the independent thread's committed state before the response assertion.
        saved = db.get_flashcard(card_uuid)
        rows = _review_rows(db, card_uuid)
        assert saved["version"] == before["version"] + 1
        assert len(rows) == 1
        assert rows[0]["rating"] == 3
        assert rows[0]["answer_time_ms"] == 1200
        session = db.get_flashcard_review_session(rows[0]["review_session_id"])
        assert session["cards_reviewed"] == 1
        assert session["status"] == "active"
        assert session["review_mode"] == mode
        assert session["deck_id"] == deck
        for field in ("front", "back", "source_ref_type", "source_ref_id", "deck_id"):
            assert saved[field] == before[field]

        timestamp_types = {
            field: type(saved[field]).__name__ for field in ("due_at", "last_reviewed_at", "last_modified")
        }
        assert response.status_code == 200, f"{response.text}; persisted timestamp types: {timestamp_types}"
        body = response.json()
        assert body["uuid"] == card_uuid
        assert body["review_session_id"] == session["id"]
        for field in (
            "ef",
            "interval_days",
            "repetitions",
            "lapses",
            "version",
            "queue_state",
            "step_index",
            "suspended_reason",
        ):
            assert body[field] == saved[field]
        assert body["scheduler_type"] == "sm2_plus"
        assert all(body["next_intervals"][key] for key in ("again", "hard", "good", "easy"))
        for field in ("due_at", "last_reviewed_at", "last_modified"):
            assert isinstance(body[field], str)
            assert _time(body[field]) == _time(saved[field])
        assert _time(body["due_at"]) > _time(body["last_reviewed_at"])
        reopened = client.get(f"/api/v1/flashcards/{card_uuid}")
        assert reopened.status_code == 200
        assert reopened.json()["version"] == body["version"]
        assert len(_review_rows(db, card_uuid)) == 1


@pytest.mark.integration
@pytest.mark.parametrize("invalid", ["missing-card", "invalid-rating"])
def test_rejected_review_http_does_not_create_review_or_session(review_db, invalid):
    db = review_db
    card_uuid = db.add_flashcard({"front": "Question", "back": "Answer"})
    before = db.get_flashcard(card_uuid)
    db.close_connection()
    with _client(db) as client:
        response = client.post(
            "/api/v1/flashcards/review",
            json={
                "card_uuid": str(uuid4()) if invalid == "missing-card" else card_uuid,
                "rating": 6 if invalid == "invalid-rating" else 3,
            },
        )
    assert response.status_code == (404 if invalid == "missing-card" else 422)
    assert _review_rows(db, card_uuid) == []
    assert db.list_flashcard_review_sessions() == []
    assert db.get_flashcard(card_uuid)["version"] == before["version"]


def _response_payload():
    return {
        "uuid": "aabca35a-d48d-482c-9d51-ed20da4c21a5",
        "ef": 2.5,
        "interval_days": 0,
        "repetitions": 1,
        "lapses": 0,
        "version": 2,
        "scheduler_type": "sm2_plus",
        "queue_state": "learning",
        "step_index": 1,
        "review_session_id": 7,
        "next_intervals": {"again": "1m", "hard": "6m", "good": "10m", "easy": "4d"},
    }


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (datetime(2026, 9, 17, 0, 51, tzinfo=timezone.utc), "2026-09-17T00:51:00+00:00"),
        (datetime(2026, 9, 17, 6, 36, tzinfo=timezone(timedelta(hours=5, minutes=45))), "2026-09-17T06:36:00+05:45"),
        (datetime(2026, 9, 17, 0, 51), "2026-09-17T00:51:00"),
        ("2026-09-17T00:51:18.308Z", "2026-09-17T00:51:18.308Z"),
        (None, None),
    ],
    ids=["utc", "offset", "naive", "sqlite-string", "null"],
)
def test_review_response_preserves_timestamp_and_non_time_values(value, expected):
    fields = ("due_at", "last_reviewed_at", "last_modified")
    payload = _response_payload()
    baseline = FlashcardReviewResponse.model_validate(payload).model_dump(mode="json")
    result = FlashcardReviewResponse.model_validate({**payload, **dict.fromkeys(fields, value)}).model_dump(mode="json")
    assert {field: result[field] for field in fields} == dict.fromkeys(fields, expected)
    assert {key: val for key, val in result.items() if key not in fields} == {
        key: val for key, val in baseline.items() if key not in fields
    }
    assert FlashcardReviewResponse.model_validate(result).model_dump(mode="json") == result


@pytest.mark.parametrize("value", [123, {"unexpected": "object"}])
def test_review_response_rejects_unrelated_timestamp_types(value):
    fields = ("due_at", "last_reviewed_at", "last_modified")
    with pytest.raises(ValidationError) as exc:
        FlashcardReviewResponse.model_validate({**_response_payload(), **dict.fromkeys(fields, value)})
    assert {error["loc"][0] for error in exc.value.errors()} == set(fields)
