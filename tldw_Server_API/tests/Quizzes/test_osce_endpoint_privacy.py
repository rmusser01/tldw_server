"""OSCE REST privacy and production projection boundaries."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from loguru import logger

os.environ.setdefault("READING_DIGEST_JOBS_WORKER_ENABLED", "0")
os.environ.setdefault("READING_DIGEST_SCHEDULER_ENABLED", "0")
os.environ.setdefault("TEST_MODE", "1")

pytestmark = pytest.mark.integration

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import (  # noqa: E402
    get_chacha_db_for_user,
)
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import (  # noqa: E402
    User,
    get_request_user,
)
from tldw_Server_API.app.core.DB_Management.backends.base import (  # noqa: E402
    DatabaseConfig,
)
from tldw_Server_API.app.core.DB_Management.backends.factory import (  # noqa: E402
    DatabaseBackendFactory,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (  # noqa: E402
    CharactersRAGDB,
)
from tldw_Server_API.app.main import app as fastapi_app  # noqa: E402
from tldw_Server_API.app.services.osce_practice import (  # noqa: E402
    materialize_station_content,
)
from tldw_Server_API.tests.Quizzes.test_osce_endpoints import (  # noqa: E402
    AUTH_HEADERS,
    STATION_SUMMARY_KEYS,
    station_content,
)
from tldw_Server_API.tests.test_config import TestConfig  # noqa: E402


@pytest.fixture
def owner_db(tmp_path) -> CharactersRAGDB:
    db = CharactersRAGDB(str(tmp_path / "owner.db"), client_id="owner")
    yield db
    db.close_connection()


@pytest.fixture
def client(owner_db: CharactersRAGDB):
    TestConfig.setup_test_environment()

    def override_get_db():
        yield owner_db

    async def override_user():
        return User(
            id=1,
            username="owner",
            email="owner@example.com",
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


def create_private_station(owner_db: CharactersRAGDB) -> dict[str, Any]:
    quiz_id = owner_db.create_quiz(name="Private OSCE", activity_type="osce")
    return owner_db.create_osce_station(
        quiz_id,
        materialize_station_content(station_content()),
        origin="generated",
        provenance={"private": "provenance secret"},
        source_bundle=[{"quote": "source bundle secret"}],
        verification_state="source_verified",
    )


def test_candidate_get_omits_marking_guide_fields(
    client: TestClient,
    owner_db: CharactersRAGDB,
) -> None:
    station = create_private_station(owner_db)
    attempt = owner_db.start_osce_attempt(station["id"], uuid4())
    assert attempt is not None

    response = client.get(f"/api/v1/quizzes/osce-attempts/{attempt['id']}")

    assert response.status_code == 200
    serialized = json.dumps(response.json())
    for secret in [
        "expected_key_points",
        "rubric_domains",
        "checklist_items",
        "rationale",
        '"quote"',
        "chunk_id",
        "page_number",
        "provenance secret",
        "source bundle secret",
    ]:
        assert secret not in serialized


def test_revealed_detail_contains_snapshot_guide_but_summary_omits_notes_and_guide(
    client: TestClient,
    owner_db: CharactersRAGDB,
) -> None:
    station = create_private_station(owner_db)
    attempt = owner_db.start_osce_attempt(station["id"], uuid4())
    assert attempt is not None
    owner_db.patch_osce_attempt(
        attempt["id"], expected_version=1, notes="candidate note secret"
    )
    revealed = owner_db.transition_osce_attempt(
        attempt["id"], "self_assessment", expected_version=2
    )
    assert revealed is not None

    detail = client.get(f"/api/v1/quizzes/osce-attempts/{attempt['id']}")
    page = client.get("/api/v1/quizzes/osce-attempts")

    assert detail.status_code == 200
    assert detail.json()["station"]["checklist_items"][0]["rationale"] == (
        "Private monitoring rationale."
    )
    assert detail.json()["notes"] == "candidate note secret"
    assert page.status_code == 200
    serialized_summary = json.dumps(page.json()["items"][0])
    assert "candidate note secret" not in serialized_summary
    assert "checklist_items" not in serialized_summary
    assert "expected_key_points" not in serialized_summary
    assert "rubric_domains" not in serialized_summary


def test_station_list_calls_compact_projection_for_every_internal_row(
    client: TestClient,
    owner_db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = create_private_station(owner_db)
    owner_db.create_osce_station(
        first["quiz_id"],
        materialize_station_content(station_content("Second private station")),
        origin="generated",
        provenance={"private": "second provenance secret"},
        source_bundle=[{"quote": "second source bundle secret"}],
        verification_state="source_verified",
    )
    route = next(
        route
        for route in fastapi_app.routes
        if getattr(route, "name", None) == "list_osce_stations"
    )
    endpoint_globals = route.endpoint.__globals__
    original_projection = endpoint_globals["project_station_summary"]
    projected_ids: list[int] = []

    def recording_projection(row: dict[str, Any]):
        projected_ids.append(int(row["id"]))
        return original_projection(row)

    monkeypatch.setitem(
        endpoint_globals,
        "project_station_summary",
        recording_projection,
    )

    response = client.get(f"/api/v1/quizzes/{first['quiz_id']}/osce-stations")

    assert response.status_code == 200
    assert projected_ids == [first["id"], first["id"] + 1]
    assert all(set(item) == STATION_SUMMARY_KEYS for item in response.json()["items"])
    serialized = json.dumps(response.json())
    for forbidden in [
        "content",
        "patient_context",
        "rationale",
        "expected_key_points",
        "rubric_domains",
        "provenance",
        "source_bundle",
        '"quote"',
    ]:
        assert forbidden not in serialized

    openapi = fastapi_app.openapi()
    response_schema = openapi["paths"][
        "/api/v1/quizzes/{quiz_id}/osce-stations"
    ]["get"]["responses"]["200"]["content"]["application/json"]["schema"]
    page_name = response_schema["$ref"].rsplit("/", 1)[-1]
    page_schema = openapi["components"]["schemas"][page_name]
    item_ref = page_schema["properties"]["items"]["items"]["$ref"]
    assert item_ref.endswith("/OsceStationSummary")


def test_cross_user_detail_and_mutation_resources_are_404(
    client: TestClient,
    owner_db: CharactersRAGDB,
    tmp_path,
) -> None:
    station = create_private_station(owner_db)
    attempt = owner_db.start_osce_attempt(station["id"], uuid4())
    assert attempt is not None
    other_db = CharactersRAGDB(str(tmp_path / "other.db"), client_id="other")

    def override_other_db():
        yield other_db

    fastapi_app.dependency_overrides[get_chacha_db_for_user] = override_other_db
    try:
        assert client.get(
            f"/api/v1/quizzes/{station['quiz_id']}/osce-stations/{station['id']}"
        ).status_code == 404
        assert client.patch(
            f"/api/v1/quizzes/{station['quiz_id']}/osce-stations/{station['id']}",
            json={"expected_version": station["version"], "content": {"title": "Denied"}},
        ).status_code == 404
        assert client.get(
            f"/api/v1/quizzes/osce-attempts/{attempt['id']}"
        ).status_code == 404
        assert client.patch(
            f"/api/v1/quizzes/osce-attempts/{attempt['id']}",
            json={"expected_version": 1, "notes": "Denied"},
        ).status_code == 404
        assert client.post(
            f"/api/v1/quizzes/osce-attempts/{attempt['id']}/begin-self-assessment",
            json={"expected_version": 1},
        ).status_code == 404
        assert client.post(
            f"/api/v1/quizzes/osce-stations/{station['id']}/attempts",
            json={"client_attempt_id": str(uuid4())},
        ).status_code == 404
    finally:
        other_db.close_connection()


def test_candidate_notes_are_not_logged(
    client: TestClient,
    owner_db: CharactersRAGDB,
) -> None:
    station = create_private_station(owner_db)
    attempt = owner_db.start_osce_attempt(station["id"], uuid4())
    assert attempt is not None
    note_secret = f"note-secret-{uuid4()}"
    messages: list[str] = []
    sink_id = logger.add(messages.append, format="{message}")
    try:
        response = client.patch(
            f"/api/v1/quizzes/osce-attempts/{attempt['id']}",
            json={"expected_version": 1, "notes": note_secret},
        )
    finally:
        logger.remove(sink_id)

    assert response.status_code == 200
    assert note_secret not in "".join(messages)


@pytest.mark.timeout(90)
def test_postgres_station_routes_hide_every_foreign_owner_path(
    pg_database_config: DatabaseConfig,
) -> None:
    owner_backend = DatabaseBackendFactory.create_backend(pg_database_config)
    attacker_backend = DatabaseBackendFactory.create_backend(pg_database_config)
    owner = CharactersRAGDB(Path(":memory:"), client_id="route-owner", backend=owner_backend)
    attacker = CharactersRAGDB(
        Path(":memory:"), client_id="route-other-user", backend=attacker_backend
    )
    try:
        quiz_id = owner.create_quiz(
            name="Private PostgreSQL OSCE",
            activity_type="osce",
            client_id=owner.client_id,
        )
        detail_station = owner.create_osce_station(
            quiz_id, materialize_station_content(station_content("Detail")), origin="manual"
        )
        update_station = owner.create_osce_station(
            quiz_id, materialize_station_content(station_content("Update")), origin="manual"
        )
        delete_station = owner.create_osce_station(
            quiz_id, materialize_station_content(station_content("Delete")), origin="manual"
        )

        def override_attacker_db():
            yield attacker

        async def override_attacker_user():
            return User(
                id=2,
                username="other-user",
                email="other@example.com",
                is_active=True,
                roles=["admin"],
                is_admin=True,
            )

        TestConfig.setup_test_environment()
        fastapi_app.dependency_overrides[get_chacha_db_for_user] = override_attacker_db
        fastapi_app.dependency_overrides[get_request_user] = override_attacker_user
        with TestClient(fastapi_app, headers=AUTH_HEADERS) as test_client:
            responses = [
                test_client.post(
                    f"/api/v1/quizzes/{quiz_id}/osce-stations",
                    json={"content": station_content("Injected")},
                ),
                test_client.get(f"/api/v1/quizzes/{quiz_id}/osce-stations"),
                test_client.get(
                    f"/api/v1/quizzes/{quiz_id}/osce-stations/{detail_station['id']}"
                ),
                test_client.patch(
                    f"/api/v1/quizzes/{quiz_id}/osce-stations/{update_station['id']}",
                    json={
                        "expected_version": update_station["version"],
                        "content": {"title": "Captured"},
                    },
                ),
                test_client.delete(
                    f"/api/v1/quizzes/{quiz_id}/osce-stations/{delete_station['id']}",
                    params={"expected_version": delete_station["version"]},
                ),
            ]

        assert [response.status_code for response in responses] == [404] * 5
        assert owner.list_osce_stations(quiz_id)["count"] == 3
        assert owner.get_osce_station(quiz_id, update_station["id"])["content"]["title"] == (
            "Update"
        )
        assert owner.get_osce_station(quiz_id, delete_station["id"]) is not None
    finally:
        fastapi_app.dependency_overrides.clear()
        TestConfig.reset_settings()
        attacker.close_all_connections()
        owner.close_all_connections()
