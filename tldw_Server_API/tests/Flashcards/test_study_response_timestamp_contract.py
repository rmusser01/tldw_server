"""Real Study history responses retain string timestamps on both databases."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from tldw_Server_API.app.api.v1.endpoints import flashcards as endpoint
from tldw_Server_API.app.api.v1.schemas.flashcards import (
    FlashcardReviewSessionSummary,
    StudyAssistantMessage,
    StudyAssistantThreadSummary,
)
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=pytest.mark.postgres)])
def study_db(request, tmp_path):
    backend = None
    if request.param == "postgres":
        backend = DatabaseBackendFactory.create_backend(request.getfixturevalue("pg_database_config"))
    db = CharactersRAGDB(tmp_path / "study.db", client_id="1", backend=backend)
    try:
        yield db
    finally:
        db.close_connection()
        if backend is not None:
            backend.get_pool().close_all()


@pytest.fixture
def study_client(study_db):
    app = FastAPI()
    app.include_router(endpoint.router, prefix="/api/v1")
    app.dependency_overrides[endpoint.get_chacha_db_for_user] = lambda: study_db
    app.dependency_overrides[endpoint.get_request_user] = lambda: SimpleNamespace(id=1)
    # Enqueuing downstream suggestions is independent of the real session write/response.
    app.dependency_overrides[endpoint.get_job_manager] = lambda: SimpleNamespace(create_job=lambda **kwargs: None)
    with TestClient(app) as client:
        yield client


def _assert_timestamp_strings(row, fields):
    for field in fields:
        assert isinstance(row[field], str), field
        assert datetime.fromisoformat(row[field].replace("Z", "+00:00")).tzinfo is not None


@pytest.mark.integration
@pytest.mark.parametrize("operation", ["active", "completed", "end"])
def test_review_session_http_lifecycle_preserves_timestamp_contract(study_db, study_client, operation):
    deck = study_db.add_deck("Rowan")
    session = study_db.get_or_create_flashcard_review_session(
        deck_id=deck, review_mode="cram", tag_filter=None, scope_key=f"cram:deck:{deck}"
    )
    if operation == "completed":
        study_db.mark_flashcard_review_session_completed(session["id"])
    if operation == "end":
        response = study_client.post("/api/v1/flashcards/review-sessions/end", json={"review_session_id": session["id"]})
    else:
        response = study_client.get("/api/v1/flashcards/review-sessions", params={"status": operation, "deck_id": deck})
    assert response.status_code == 200
    row = response.json() if operation == "end" else response.json()[0]
    assert row["id"] == session["id"]
    assert row["deck_id"] == deck
    assert row["review_mode"] == "cram"
    assert row["status"] == ("active" if operation == "active" else "completed")
    _assert_timestamp_strings(row, ["started_at", "last_activity_at"])
    if operation == "active":
        assert row["completed_at"] is None
    else:
        _assert_timestamp_strings(row, ["completed_at"])
        assert study_db.get_flashcard_review_session(session["id"])["status"] == "completed"


@pytest.mark.integration
@pytest.mark.parametrize("populated", [False, True], ids=["empty-thread", "populated-thread"])
def test_assistant_context_http_preserves_thread_and_message_timestamps(study_db, study_client, populated):
    card = study_db.add_flashcard({"front": "Marker?", "back": "Amber"})
    thread = study_db.get_or_create_study_assistant_thread(context_type="flashcard", flashcard_uuid=card)
    if populated:
        study_db.append_study_assistant_message(
            thread_id=thread["id"], role="user", action_type="freeform", input_modality="text",
            content="Explain the marker", structured_payload={"fixture": True},
            context_snapshot={"front": "Marker?"}, expected_thread_version=thread["version"],
        )
    response = study_client.get(f"/api/v1/flashcards/{card}/assistant")
    assert response.status_code == 200
    body = response.json()
    assert body["thread"]["id"] == thread["id"]
    assert body["thread"]["flashcard_uuid"] == card
    assert body["thread"]["message_count"] == int(populated)
    _assert_timestamp_strings(body["thread"], ["created_at", "last_modified"])
    if populated:
        _assert_timestamp_strings(body["thread"], ["last_message_at"])
        assert len(body["messages"]) == 1
        message = body["messages"][0]
        _assert_timestamp_strings(message, ["created_at"])
        assert message["content"] == "Explain the marker"
        assert message["structured_payload"] == {"fixture": True}
        assert message["context_snapshot"] == {"front": "Marker?"}
    else:
        assert body["thread"]["last_message_at"] is None
        assert body["messages"] == []


@pytest.fixture(params=["session", "thread", "message"])
def timestamp_model(request):
    if request.param == "session":
        return FlashcardReviewSessionSummary, {
            "id": 7, "review_mode": "cram", "scope_key": "cram:global", "status": "completed", "client_id": "1",
        }, ["started_at", "last_activity_at", "completed_at"]
    if request.param == "thread":
        return StudyAssistantThreadSummary, {
            "id": 3, "context_type": "flashcard", "deleted": False, "client_id": "1", "version": 2,
        }, ["last_message_at", "created_at", "last_modified"]
    return StudyAssistantMessage, {
        "id": 9, "thread_id": 3, "role": "user", "action_type": "freeform",
        "input_modality": "text", "content": "Question", "client_id": "1",
        "structured_payload": {"fixture": True}, "context_snapshot": {"front": "Marker?"},
    }, ["created_at"]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (datetime(2026, 9, 16, 23, 19, tzinfo=timezone.utc), "2026-09-16T23:19:00+00:00"),
        (datetime(2026, 9, 17, 5, 4, tzinfo=timezone(timedelta(hours=5, minutes=45))), "2026-09-17T05:04:00+05:45"),
        (datetime(2026, 9, 16, 23, 19), "2026-09-16T23:19:00"),
        ("2026-09-16T23:19:00.125Z", "2026-09-16T23:19:00.125Z"),
        (None, None),
    ],
    ids=["utc", "offset", "naive", "sqlite-string", "null"],
)
def test_study_timestamp_conversion_preserves_values_and_non_time_fields(timestamp_model, value, expected):
    model, payload, fields = timestamp_model
    baseline = model.model_validate(payload).model_dump(mode="json")
    payload.update(dict.fromkeys(fields, value))
    result = model.model_validate(payload).model_dump(mode="json")
    assert {field: result[field] for field in fields} == dict.fromkeys(fields, expected)
    assert {key: val for key, val in result.items() if key not in fields} == {
        key: val for key, val in baseline.items() if key not in fields
    }
    assert model.model_validate(result).model_dump(mode="json") == result


def test_study_timestamp_conversion_rejects_unrelated_types(timestamp_model):
    model, payload, fields = timestamp_model
    payload.update(dict.fromkeys(fields, 123))
    with pytest.raises(ValidationError) as exc:
        model.model_validate(payload)
    assert {error["loc"][0] for error in exc.value.errors()} == set(fields)
