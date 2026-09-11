"""OSCE station authoring and practice endpoint contracts."""

from __future__ import annotations

import os
import time
from typing import Any
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("READING_DIGEST_JOBS_WORKER_ENABLED", "0")
os.environ.setdefault("READING_DIGEST_SCHEDULER_ENABLED", "0")
os.environ.setdefault("TEST_MODE", "1")

pytestmark = pytest.mark.integration

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import (  # noqa: E402
    get_chacha_db_for_user,
)
from tldw_Server_API.app.api.v1.schemas.osce import OsceStationStoredContent  # noqa: E402
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import (  # noqa: E402
    User,
    get_request_user,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (  # noqa: E402
    CharactersRAGDB,
)
from tldw_Server_API.app.main import app as fastapi_app  # noqa: E402
from tldw_Server_API.app.services.osce_practice import (  # noqa: E402
    materialize_station_content,
)
from tldw_Server_API.tests.test_config import TestConfig  # noqa: E402

AUTH_HEADERS = {"X-API-KEY": TestConfig.TEST_API_KEY}
STATION_SUMMARY_KEYS = {
    "id",
    "quiz_id",
    "title",
    "recommended_duration_seconds",
    "order_index",
    "version",
    "checklist_count",
    "rubric_domain_count",
    "verification_state",
    "created_at",
    "updated_at",
}


def station_content(title: str = "Warfarin counselling") -> dict[str, Any]:
    return {
        "title": title,
        "candidate_instructions": "You are speaking with a simulated patient.",
        "candidate_task": "Explain safe medicine use.",
        "patient_context": {
            "text": "A fictional adult has started warfarin.",
            "citations": [
                {
                    "source_type": "document",
                    "source_id": "doc-1",
                    "label": "Anticoagulation guide",
                    "quote": "Private source quote",
                    "chunk_id": "chunk-1",
                    "page_number": 4,
                }
            ],
        },
        "recommended_duration_seconds": 480,
        "checklist_items": [
            {
                "label": "Explains monitoring",
                "rationale": "Private monitoring rationale.",
                "citations": [],
            },
            {
                "label": "Explains warning signs",
                "rationale": "Private warning-sign rationale.",
                "citations": [],
            },
        ],
        "rubric_domains": [
            {
                "label": "Communication",
                "levels": [
                    {
                        "label": "Developing",
                        "description": "The explanation is incomplete.",
                    },
                    {"label": "Effective", "description": "The explanation is clear."},
                ],
            }
        ],
        "expected_key_points": [
            {"text": "Private expected key point.", "citations": []}
        ],
    }


def stored_content(title: str = "Warfarin counselling") -> OsceStationStoredContent:
    return materialize_station_content(station_content(title))


def create_station(
    db: CharactersRAGDB,
    *,
    quiz_id: int | None = None,
    title: str = "Warfarin counselling",
    order_index: int = 0,
) -> dict[str, Any]:
    if quiz_id is None:
        quiz_id = db.create_quiz(name="OSCE", activity_type="osce")
    return db.create_osce_station(
        quiz_id,
        stored_content(title),
        order_index=order_index,
        origin="generated",
        provenance={"provider": "private-provider"},
        source_bundle=[{"source_id": "source-1", "quote": "Private source bundle"}],
        verification_state="source_verified",
    )


@pytest.fixture
def quizzes_db(tmp_path) -> CharactersRAGDB:
    db = CharactersRAGDB(
        str(tmp_path / "osce-endpoints.db"),
        client_id=f"test-{uuid4().hex[:8]}",
    )
    yield db
    db.close_connection()


@pytest.fixture
def client(quizzes_db: CharactersRAGDB):
    TestConfig.setup_test_environment()

    def override_get_db():
        yield quizzes_db

    async def override_user():
        return User(
            id=1,
            username="testuser",
            email="test@example.com",
            is_active=True,
            roles=["admin"],
            is_admin=True,
        )

    fastapi_app.dependency_overrides[get_chacha_db_for_user] = override_get_db
    fastapi_app.dependency_overrides[get_request_user] = override_user
    with TestClient(fastapi_app, headers=AUTH_HEADERS) as test_client:
        yield test_client
    fastapi_app.dependency_overrides.clear()
    TestConfig.reset_settings()


def test_station_authoring_crud_and_compact_pagination(
    client: TestClient,
    quizzes_db: CharactersRAGDB,
) -> None:
    quiz_id = quizzes_db.create_quiz(name="Authoring", activity_type="osce")
    first = client.post(
        f"/api/v1/quizzes/{quiz_id}/osce-stations",
        json={"content": station_content("Second station"), "order_index": 2},
    )
    second = client.post(
        f"/api/v1/quizzes/{quiz_id}/osce-stations",
        json={"content": station_content("First station"), "order_index": 0},
    )

    assert first.status_code == 201
    assert second.status_code == 201
    assert first.json()["origin"] == "manual"
    assert first.json()["verification_state"] == "manually_authored"

    page = client.get(
        f"/api/v1/quizzes/{quiz_id}/osce-stations",
        params={"limit": 1, "offset": 0},
    )
    assert page.status_code == 200
    assert page.json()["count"] == 2
    assert page.json()["has_more"] is True
    assert page.json()["next_offset"] == 1
    assert page.json()["pagination"] == {
        "mode": "offset",
        "limit": 1,
        "offset": 0,
        "total": 2,
        "has_more": True,
        "next_offset": 1,
    }
    assert set(page.json()["items"][0]) == STATION_SUMMARY_KEYS
    assert page.json()["items"][0]["title"] == "First station"

    station_id = first.json()["id"]
    detail = client.get(f"/api/v1/quizzes/{quiz_id}/osce-stations/{station_id}")
    assert detail.status_code == 200
    assert detail.json()["content"]["expected_key_points"][0]["text"] == (
        "Private expected key point."
    )

    updated = client.patch(
        f"/api/v1/quizzes/{quiz_id}/osce-stations/{station_id}",
        json={
            "expected_version": first.json()["version"],
            "content": {"title": "Updated station"},
            "order_index": 3,
        },
    )
    assert updated.status_code == 200
    assert updated.json()["content"]["title"] == "Updated station"
    assert updated.json()["order_index"] == 3

    stale = client.patch(
        f"/api/v1/quizzes/{quiz_id}/osce-stations/{station_id}",
        json={"expected_version": first.json()["version"], "content": {"title": "Stale"}},
    )
    assert stale.status_code == 409

    deleted = client.delete(
        f"/api/v1/quizzes/{quiz_id}/osce-stations/{station_id}",
        params={"expected_version": updated.json()["version"]},
    )
    assert deleted.status_code == 200
    assert deleted.json() == {"status": "deleted"}
    assert client.get(
        f"/api/v1/quizzes/{quiz_id}/osce-stations/{station_id}"
    ).status_code == 404


def test_station_routes_map_activity_parent_absence_and_malformed_content(
    client: TestClient,
    quizzes_db: CharactersRAGDB,
) -> None:
    question_quiz_id = quizzes_db.create_quiz(name="Questions")
    osce_quiz_id = quizzes_db.create_quiz(name="OSCE", activity_type="osce")
    other_quiz_id = quizzes_db.create_quiz(name="Other OSCE", activity_type="osce")
    station = create_station(quizzes_db, quiz_id=osce_quiz_id)

    assert client.post(
        f"/api/v1/quizzes/{question_quiz_id}/osce-stations",
        json={"content": station_content()},
    ).status_code == 409
    assert client.get(
        f"/api/v1/quizzes/{question_quiz_id}/osce-stations"
    ).status_code == 409
    assert client.get(
        f"/api/v1/quizzes/{other_quiz_id}/osce-stations/{station['id']}"
    ).status_code == 404
    assert client.patch(
        f"/api/v1/quizzes/{other_quiz_id}/osce-stations/{station['id']}",
        json={"expected_version": 1, "content": {"title": "Wrong parent"}},
    ).status_code == 404
    assert client.delete(
        f"/api/v1/quizzes/{other_quiz_id}/osce-stations/{station['id']}"
    ).status_code == 404
    assert client.get("/api/v1/quizzes/999999/osce-stations").status_code == 404

    malformed = station_content()
    malformed["recommended_duration_seconds"] = "480"
    assert client.post(
        f"/api/v1/quizzes/{osce_quiz_id}/osce-stations",
        json={"content": malformed},
    ).status_code == 422


def test_attempt_retry_lifecycle_conflict_and_selection_validation(
    client: TestClient,
    quizzes_db: CharactersRAGDB,
) -> None:
    station = create_station(quizzes_db)
    client_attempt_id = str(uuid4())

    started = client.post(
        f"/api/v1/quizzes/osce-stations/{station['id']}/attempts",
        json={"client_attempt_id": client_attempt_id},
    )
    retried = client.post(
        f"/api/v1/quizzes/osce-stations/{station['id']}/attempts",
        json={"client_attempt_id": client_attempt_id},
    )
    assert started.status_code == 201
    assert retried.status_code == 201
    assert retried.json()["id"] == started.json()["id"]
    assert started.json()["state"] == "in_progress"

    attempt_id = started.json()["id"]
    note_patch = client.patch(
        f"/api/v1/quizzes/osce-attempts/{attempt_id}",
        json={"expected_version": 1, "notes": "Private candidate draft"},
    )
    assert note_patch.status_code == 200
    assert note_patch.json()["version"] == 2
    assert note_patch.json()["notes"] == "Private candidate draft"

    invalid_phase = client.patch(
        f"/api/v1/quizzes/osce-attempts/{attempt_id}",
        json={
            "expected_version": 2,
            "checklist_selections": {str(uuid4()): "met"},
        },
    )
    assert invalid_phase.status_code == 422
    assert client.post(
        f"/api/v1/quizzes/osce-attempts/{attempt_id}/complete",
        json={"expected_version": 2},
    ).status_code == 409

    revealed = client.post(
        f"/api/v1/quizzes/osce-attempts/{attempt_id}/begin-self-assessment",
        json={"expected_version": 2},
    )
    repeated_reveal = client.post(
        f"/api/v1/quizzes/osce-attempts/{attempt_id}/begin-self-assessment",
        json={"expected_version": 2},
    )
    assert revealed.status_code == 200
    assert repeated_reveal.status_code == 200
    assert revealed.json()["state"] == "self_assessment"
    assert repeated_reveal.json()["version"] == revealed.json()["version"]

    invalid_selection = client.patch(
        f"/api/v1/quizzes/osce-attempts/{attempt_id}",
        json={
            "expected_version": revealed.json()["version"],
            "checklist_selections": {str(uuid4()): "met"},
        },
    )
    assert invalid_selection.status_code == 422
    assert client.post(
        f"/api/v1/quizzes/osce-attempts/{attempt_id}/complete",
        json={"expected_version": revealed.json()["version"]},
    ).status_code == 422

    revealed_station = OsceStationStoredContent.model_validate(revealed.json()["station"])
    selections = client.patch(
        f"/api/v1/quizzes/osce-attempts/{attempt_id}",
        json={
            "expected_version": revealed.json()["version"],
            "checklist_selections": {
                str(item.id): "met" for item in revealed_station.checklist_items
            },
            "rubric_selections": {
                str(domain.id): str(domain.levels[-1].id)
                for domain in revealed_station.rubric_domains
            },
        },
    )
    assert selections.status_code == 200
    completed = client.post(
        f"/api/v1/quizzes/osce-attempts/{attempt_id}/complete",
        json={"expected_version": selections.json()["version"]},
    )
    repeated_complete = client.post(
        f"/api/v1/quizzes/osce-attempts/{attempt_id}/complete",
        json={"expected_version": selections.json()["version"]},
    )
    assert completed.status_code == 200
    assert repeated_complete.status_code == 200
    assert completed.json()["state"] == "completed"
    assert repeated_complete.json()["version"] == completed.json()["version"]


def test_attempt_list_preserves_repeatable_states_filters_and_recency_order(
    client: TestClient,
    quizzes_db: CharactersRAGDB,
) -> None:
    station = create_station(quizzes_db)
    first = client.post(
        f"/api/v1/quizzes/osce-stations/{station['id']}/attempts",
        json={"client_attempt_id": str(uuid4())},
    ).json()
    second = client.post(
        f"/api/v1/quizzes/osce-stations/{station['id']}/attempts",
        json={"client_attempt_id": str(uuid4())},
    ).json()
    client.post(
        f"/api/v1/quizzes/osce-attempts/{second['id']}/begin-self-assessment",
        json={"expected_version": second["version"]},
    )
    time.sleep(0.005)
    client.patch(
        f"/api/v1/quizzes/osce-attempts/{first['id']}",
        json={"expected_version": first["version"], "notes": "most recent"},
    )

    page = client.get(
        "/api/v1/quizzes/osce-attempts",
        params=[
            ("quiz_id", str(station["quiz_id"])),
            ("station_id", str(station["id"])),
            ("state", "in_progress"),
            ("state", "self_assessment"),
            ("limit", "1"),
            ("offset", "0"),
        ],
    )

    assert page.status_code == 200
    assert page.json()["count"] == 2
    assert page.json()["items"][0]["id"] == first["id"]
    assert page.json()["items"][0]["state"] == "in_progress"
    assert page.json()["next_offset"] == 1
    assert "notes" not in page.json()["items"][0]
    assert "station" not in page.json()["items"][0]


def test_deleted_station_blocks_new_attempt_but_snapshot_attempt_remains_readable(
    client: TestClient,
    quizzes_db: CharactersRAGDB,
) -> None:
    station = create_station(quizzes_db)
    retry_key = str(uuid4())
    attempt = client.post(
        f"/api/v1/quizzes/osce-stations/{station['id']}/attempts",
        json={"client_attempt_id": retry_key},
    ).json()
    assert client.delete(
        f"/api/v1/quizzes/{station['quiz_id']}/osce-stations/{station['id']}",
        params={"expected_version": station["version"]},
    ).status_code == 200

    assert client.get(
        f"/api/v1/quizzes/osce-attempts/{attempt['id']}"
    ).status_code == 200
    retried = client.post(
        f"/api/v1/quizzes/osce-stations/{station['id']}/attempts",
        json={"client_attempt_id": retry_key},
    )
    assert retried.status_code == 201
    assert retried.json()["id"] == attempt["id"]
    assert client.post(
        f"/api/v1/quizzes/osce-stations/{station['id']}/attempts",
        json={"client_attempt_id": str(uuid4())},
    ).status_code == 404


@pytest.mark.parametrize(
    ("method", "path", "json_body"),
    [
        ("get", "/api/v1/quizzes/osce-attempts/999999", None),
        (
            "patch",
            "/api/v1/quizzes/osce-attempts/999999",
            {"expected_version": 1, "notes": "missing"},
        ),
        (
            "post",
            "/api/v1/quizzes/osce-attempts/999999/begin-self-assessment",
            {"expected_version": 1},
        ),
        (
            "post",
            "/api/v1/quizzes/osce-attempts/999999/complete",
            {"expected_version": 1},
        ),
    ],
)
def test_absent_attempt_resources_return_404(
    client: TestClient,
    method: str,
    path: str,
    json_body: dict[str, Any] | None,
) -> None:
    response = client.request(method, path, json=json_body)
    assert response.status_code == 404


def test_openapi_owns_all_routes_once_and_discriminates_attempt_phases() -> None:
    schema = fastapi_app.openapi()
    expected_operations = {
        ("post", "/api/v1/quizzes/{quiz_id}/osce-stations"),
        ("get", "/api/v1/quizzes/{quiz_id}/osce-stations"),
        ("get", "/api/v1/quizzes/{quiz_id}/osce-stations/{station_id}"),
        ("patch", "/api/v1/quizzes/{quiz_id}/osce-stations/{station_id}"),
        ("delete", "/api/v1/quizzes/{quiz_id}/osce-stations/{station_id}"),
        ("post", "/api/v1/quizzes/osce-stations/{station_id}/attempts"),
        ("get", "/api/v1/quizzes/osce-attempts"),
        ("get", "/api/v1/quizzes/osce-attempts/{attempt_id}"),
        ("patch", "/api/v1/quizzes/osce-attempts/{attempt_id}"),
        ("post", "/api/v1/quizzes/osce-attempts/{attempt_id}/begin-self-assessment"),
        ("post", "/api/v1/quizzes/osce-attempts/{attempt_id}/complete"),
    }
    actual_operations = {
        (method, path)
        for path, path_item in schema["paths"].items()
        for method in path_item
        if method in {"get", "post", "patch", "delete"}
    }
    assert expected_operations <= actual_operations

    operation_ids = [
        path_item[method]["operationId"]
        for path_item in schema["paths"].values()
        for method in path_item
        if method in {"get", "post", "put", "patch", "delete"}
    ]
    assert len(operation_ids) == len(set(operation_ids))

    attempt_schema = schema["paths"][
        "/api/v1/quizzes/osce-attempts/{attempt_id}"
    ]["get"]["responses"]["200"]["content"]["application/json"]["schema"]
    serialized_schema = str(attempt_schema)
    assert "OsceCandidateAttemptResponse" in serialized_schema
    assert "OsceRevealedAttemptResponse" in serialized_schema
    assert attempt_schema["discriminator"]["propertyName"] == "state"
