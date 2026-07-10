from __future__ import annotations

import json
import os
import uuid
from copy import deepcopy
from datetime import datetime, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

os.environ.setdefault("READING_DIGEST_JOBS_WORKER_ENABLED", "0")
os.environ.setdefault("READING_DIGEST_SCHEDULER_ENABLED", "0")
os.environ.setdefault("TEST_MODE", "1")

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.endpoints import flashcards as flashcards_endpoint
from tldw_Server_API.app.api.v1.endpoints.flashcards import router as flashcards_router
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.tests.test_config import TestConfig


AUTH_HEADERS = {"X-API-KEY": TestConfig.TEST_API_KEY}
BASE_PATH = "/api/v1/flashcards/source-review-plans"


def _app() -> FastAPI:
    app = FastAPI()
    app.include_router(flashcards_router, prefix="/api/v1")
    return app


@pytest.fixture
def source_review_api_db(tmp_path) -> CharactersRAGDB:
    db = CharactersRAGDB(
        str(tmp_path / "source-review-api.db"),
        client_id=f"source-review-api-{uuid.uuid4().hex[:6]}",
    )
    yield db
    db.close_connection()


@pytest.fixture
def client_with_flashcards_db(source_review_api_db: CharactersRAGDB):
    TestConfig.setup_test_environment()
    app = _app()

    def override_db():
        yield source_review_api_db

    async def override_user():
        return User(
            id=1,
            username="source-review-user",
            email="source-review@example.com",
            is_active=True,
            roles=["admin"],
            is_admin=True,
        )

    app.dependency_overrides[get_chacha_db_for_user] = override_db
    app.dependency_overrides[get_request_user] = override_user
    with TestClient(app, headers=AUTH_HEADERS) as client:
        yield client
    app.dependency_overrides.clear()
    TestConfig.reset_settings()


def _valid_payload() -> dict:
    return {
        "title": "Cardiac physiology review",
        "starts_on": "2026-07-09",
        "timezone": "UTC",
        "source_items": [
            {
                "source_type": "note",
                "source_id": "note-42",
                "source_title": "Cardiac physiology",
                "excerpt_text": "Frank-Starling mechanism",
                "locator": {"section": "Hemodynamics"},
            }
        ],
        "schedule": [
            {"offset_value": 1, "offset_unit": "day", "activity_type": "reread"},
            {"offset_value": 3, "offset_unit": "day", "activity_type": "quiz"},
        ],
    }


def _create_plan(client: TestClient, payload: dict | None = None) -> dict:
    response = client.post(BASE_PATH, json=payload or _valid_payload())
    assert response.status_code == 200  # nosec B101
    return response.json()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("title", "   "),
        ("title", "x" * 201),
        ("timezone", "Not/A-Timezone"),
        ("source_items", []),
        (
            "source_items",
            [{"source_type": "note", "source_id": "   ", "label": "Invalid"}],
        ),
        ("schedule", []),
        (
            "schedule",
            [
                {
                    "offset_value": offset,
                    "offset_unit": "day",
                    "activity_type": "reread",
                }
                for offset in range(1, 26)
            ],
        ),
    ],
)
def test_create_source_review_plan_rejects_invalid_required_fields(
    client_with_flashcards_db: TestClient,
    field: str,
    value,
) -> None:
    payload = _valid_payload()
    payload[field] = value

    response = client_with_flashcards_db.post(BASE_PATH, json=payload)

    assert response.status_code == 422  # nosec B101


@pytest.mark.parametrize(
    "schedule_row",
    [
        {"offset_value": 0, "offset_unit": "day", "activity_type": "reread"},
        {"offset_value": 3651, "offset_unit": "day", "activity_type": "reread"},
        {"offset_value": 121, "offset_unit": "month", "activity_type": "reread"},
        {"offset_value": True, "offset_unit": "day", "activity_type": "reread"},
    ],
)
def test_create_source_review_plan_rejects_invalid_offsets(
    client_with_flashcards_db: TestClient,
    schedule_row: dict,
) -> None:
    payload = _valid_payload()
    payload["schedule"] = [schedule_row]

    response = client_with_flashcards_db.post(BASE_PATH, json=payload)

    assert response.status_code == 422  # nosec B101


def test_create_source_review_plan_rejects_source_size_limits(
    client_with_flashcards_db: TestClient,
) -> None:
    too_many = _valid_payload()
    too_many["source_items"] = too_many["source_items"] * 11
    long_excerpt = _valid_payload()
    long_excerpt["source_items"][0]["excerpt_text"] = "x" * 20_001
    large_locator = _valid_payload()
    large_locator["source_items"][0]["locator"] = {"value": "x" * 8_192}

    statuses = [
        client_with_flashcards_db.post(BASE_PATH, json=payload).status_code
        for payload in (too_many, long_excerpt, large_locator)
    ]

    assert statuses == [422, 422, 422]  # nosec B101


def test_create_source_review_plan_rejects_duplicate_computed_due_activity(
    client_with_flashcards_db: TestClient,
) -> None:
    payload = _valid_payload()
    payload["schedule"] = [
        {"offset_value": 1, "offset_unit": "day", "activity_type": "quiz"},
        {"offset_value": 1, "offset_unit": "day", "activity_type": "quiz"},
    ]

    response = client_with_flashcards_db.post(BASE_PATH, json=payload)

    assert response.status_code == 422  # nosec B101


def test_create_and_list_source_review_plans_return_canonical_snapshot_and_occurrences(
    client_with_flashcards_db: TestClient,
) -> None:
    created = _create_plan(client_with_flashcards_db)
    listed = client_with_flashcards_db.get(BASE_PATH)

    assert created["source_bundle"]["items"][0]["label"] == "Cardiac physiology"  # nosec B101
    assert "source_title" not in created["source_bundle"]["items"][0]  # nosec B101
    assert [item["activity_type"] for item in created["occurrences"]] == [  # nosec B101
        "reread",
        "quiz",
    ]
    assert listed.status_code == 200  # nosec B101
    assert listed.json()["total"] == 1  # nosec B101
    assert listed.json()["items"][0]["id"] == created["id"]  # nosec B101


def test_source_review_list_and_due_limits_are_capped_at_100(
    client_with_flashcards_db: TestClient,
    source_review_api_db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, int] = {}

    def list_plans(*, limit: int, offset: int):
        seen["plans"] = limit
        return [], 0

    def list_due(*, now_utc: str, limit: int, offset: int):
        seen["due"] = limit
        return [], 0

    monkeypatch.setattr(source_review_api_db, "list_source_review_plans", list_plans)
    monkeypatch.setattr(source_review_api_db, "list_due_source_review_occurrences", list_due)

    assert client_with_flashcards_db.get(BASE_PATH, params={"limit": 999}).status_code == 200  # nosec B101
    assert client_with_flashcards_db.get(f"{BASE_PATH}/due", params={"limit": 999}).status_code == 200  # nosec B101
    assert seen == {"plans": 100, "due": 100}  # nosec B101


def test_due_source_review_response_has_backend_utc_now(
    client_with_flashcards_db: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 7, 10, 12, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(flashcards_endpoint, "_source_review_now_utc", lambda: now)
    _create_plan(client_with_flashcards_db)

    response = client_with_flashcards_db.get(f"{BASE_PATH}/due")

    assert response.status_code == 200  # nosec B101
    body = response.json()
    assert body["now"] == "2026-07-10T12:00:00Z"  # nosec B101
    assert body["total"] == 1  # nosec B101
    assert body["items"][0]["activity_type"] == "reread"  # nosec B101
    assert body["items"][0]["source_summary"] == [  # nosec B101
        {
            "source_type": "note",
            "source_id": "note-42",
            "label": "Cardiac physiology",
            "excerpt_preview": "Frank-Starling mechanism",
        }
    ]
    assert "excerpt_text" not in json.dumps(body["items"][0]["source_summary"])  # nosec B101


def test_source_review_start_complete_and_stored_launch_state(
    client_with_flashcards_db: TestClient,
    source_review_api_db: CharactersRAGDB,
) -> None:
    created = _create_plan(client_with_flashcards_db)
    occurrence_id = created["occurrences"][0]["id"]

    pending_complete = client_with_flashcards_db.post(
        f"{BASE_PATH}/occurrences/{occurrence_id}/complete"
    )
    started = client_with_flashcards_db.post(
        f"{BASE_PATH}/occurrences/{occurrence_id}/start"
    )
    resumed = client_with_flashcards_db.post(
        f"{BASE_PATH}/occurrences/{occurrence_id}/start"
    )
    completed = client_with_flashcards_db.post(
        f"{BASE_PATH}/occurrences/{occurrence_id}/complete"
    )
    repeated = client_with_flashcards_db.post(
        f"{BASE_PATH}/occurrences/{occurrence_id}/complete"
    )
    skip_after_complete = client_with_flashcards_db.post(
        f"{BASE_PATH}/occurrences/{occurrence_id}/skip"
    )
    stored = source_review_api_db.execute_query(
        "SELECT launch_state_json FROM source_review_occurrences WHERE id = ?",
        (occurrence_id,),
    ).fetchone()
    thin_launch = json.loads(stored["launch_state_json"])

    assert pending_complete.status_code == 409  # nosec B101
    assert started.status_code == 200  # nosec B101
    assert started.json()["launch_state"]["source_bundle"] == created["source_bundle"]  # nosec B101
    assert started.json()["launch_state"]["completion_required"] is True  # nosec B101
    assert started.json()["launch_state"]["target_surface"] == "source_review_due_panel"  # nosec B101
    assert resumed.json()["version"] == started.json()["version"]  # nosec B101
    assert completed.json()["status"] == "completed"  # nosec B101
    assert repeated.json()["version"] == completed.json()["version"]  # nosec B101
    assert skip_after_complete.status_code == 409  # nosec B101
    assert "source_bundle" not in thin_launch  # nosec B101
    assert "source_items" not in thin_launch  # nosec B101


def test_source_review_skip_is_idempotent_and_conflicts_with_start_or_complete(
    client_with_flashcards_db: TestClient,
) -> None:
    created = _create_plan(client_with_flashcards_db)
    occurrence_id = created["occurrences"][0]["id"]

    skipped = client_with_flashcards_db.post(
        f"{BASE_PATH}/occurrences/{occurrence_id}/skip"
    )
    repeated = client_with_flashcards_db.post(
        f"{BASE_PATH}/occurrences/{occurrence_id}/skip"
    )
    start = client_with_flashcards_db.post(
        f"{BASE_PATH}/occurrences/{occurrence_id}/start"
    )
    complete = client_with_flashcards_db.post(
        f"{BASE_PATH}/occurrences/{occurrence_id}/complete"
    )

    assert skipped.status_code == 200  # nosec B101
    assert skipped.json()["status"] == "skipped"  # nosec B101
    assert repeated.json()["version"] == skipped.json()["version"]  # nosec B101
    assert start.status_code == 409  # nosec B101
    assert complete.status_code == 409  # nosec B101


def test_delete_source_review_plan_is_idempotent_and_missing_is_404(
    client_with_flashcards_db: TestClient,
) -> None:
    missing = client_with_flashcards_db.delete(f"{BASE_PATH}/999999")
    created = _create_plan(client_with_flashcards_db)
    first = client_with_flashcards_db.delete(f"{BASE_PATH}/{created['id']}")
    repeated = client_with_flashcards_db.delete(f"{BASE_PATH}/{created['id']}")

    assert missing.status_code == 404  # nosec B101
    assert first.json() == {"deleted": True}  # nosec B101
    assert repeated.json() == {"deleted": False}  # nosec B101


@pytest.mark.parametrize("action", ["start", "complete", "skip"])
def test_source_review_actions_against_deleted_plan_return_404(
    client_with_flashcards_db: TestClient,
    action: str,
) -> None:
    created = _create_plan(client_with_flashcards_db)
    occurrence_id = created["occurrences"][0]["id"]
    client_with_flashcards_db.delete(f"{BASE_PATH}/{created['id']}")

    response = client_with_flashcards_db.post(
        f"{BASE_PATH}/occurrences/{occurrence_id}/{action}"
    )

    assert response.status_code == 404  # nosec B101


@pytest.mark.parametrize("action", ["start", "complete", "skip"])
def test_source_review_actions_against_deleted_occurrence_return_404(
    client_with_flashcards_db: TestClient,
    source_review_api_db: CharactersRAGDB,
    action: str,
) -> None:
    created = _create_plan(client_with_flashcards_db)
    occurrence_id = created["occurrences"][0]["id"]
    source_review_api_db.execute_query(
        "UPDATE source_review_occurrences SET deleted = 1 WHERE id = ?",
        (occurrence_id,),
        commit=True,
    )

    response = client_with_flashcards_db.post(
        f"{BASE_PATH}/occurrences/{occurrence_id}/{action}"
    )

    assert response.status_code == 404  # nosec B101
