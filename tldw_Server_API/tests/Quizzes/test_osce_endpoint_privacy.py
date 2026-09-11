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


@pytest.mark.timeout(120)
def test_postgres_quiz_http_crud_is_owner_scoped_and_cascade_safe(
    pg_database_config: DatabaseConfig,
) -> None:
    owner_backend = DatabaseBackendFactory.create_backend(pg_database_config)
    attacker_backend = DatabaseBackendFactory.create_backend(pg_database_config)
    owner = CharactersRAGDB(Path(":memory:"), client_id="http-owner", backend=owner_backend)
    attacker = CharactersRAGDB(
        Path(":memory:"), client_id="http-attacker", backend=attacker_backend
    )
    active: dict[str, Any] = {
        "db": owner,
        "user": User(
            id=101,
            username="http-owner",
            email="http-owner@example.com",
            is_active=True,
            roles=["admin"],
            is_admin=True,
        ),
    }

    def override_active_db():
        yield active["db"]

    async def override_active_user():
        return active["user"]

    try:
        TestConfig.setup_test_environment()
        fastapi_app.dependency_overrides[get_chacha_db_for_user] = override_active_db
        fastapi_app.dependency_overrides[get_request_user] = override_active_user
        with TestClient(fastapi_app, headers=AUTH_HEADERS) as test_client:
            owner_ids = {}
            for label in [
                "read",
                "update",
                "foreign-soft-delete",
                "foreign-hard-delete",
                "owner-soft-delete",
                "owner-hard-delete",
            ]:
                response = test_client.post(
                    "/api/v1/quizzes",
                    json={"name": f"Owner {label}"},
                )
                assert response.status_code == 200, response.text
                owner_ids[label] = response.json()["id"]

            osce_id = owner.create_quiz(name="Owner OSCE", activity_type="osce")
            station = owner.create_osce_station(
                osce_id,
                materialize_station_content(station_content("Cascade target")),
                origin="manual",
            )

            active["db"] = attacker
            active["user"] = User(
                id=202,
                username="http-attacker",
                email="http-attacker@example.com",
                is_active=True,
                roles=["admin"],
                is_admin=True,
            )
            attacker_create = test_client.post(
                "/api/v1/quizzes",
                json={"name": "Attacker quiz"},
            )
            assert attacker_create.status_code == 200
            attacker_id = attacker_create.json()["id"]

            attacker_list = test_client.get(
                "/api/v1/quizzes",
                params={"include_workspace_items": True},
            )
            foreign_get = test_client.get(f"/api/v1/quizzes/{owner_ids['read']}")
            foreign_update = test_client.patch(
                f"/api/v1/quizzes/{owner_ids['update']}",
                json={"name": "Captured", "expected_version": 1},
            )
            foreign_soft_delete = test_client.delete(
                f"/api/v1/quizzes/{owner_ids['foreign-soft-delete']}",
                params={"expected_version": 1},
            )
            foreign_hard_delete = test_client.delete(
                f"/api/v1/quizzes/{owner_ids['foreign-hard-delete']}",
                params={"hard": True},
            )
            foreign_osce_delete = test_client.delete(
                f"/api/v1/quizzes/{osce_id}",
                params={"expected_version": 1},
            )

            active["db"] = owner
            active["user"] = User(
                id=101,
                username="http-owner",
                email="http-owner@example.com",
                is_active=True,
                roles=["admin"],
                is_admin=True,
            )
            owner_update = test_client.patch(
                f"/api/v1/quizzes/{owner_ids['update']}",
                json={"name": "Owner updated", "expected_version": 1},
            )
            owner_soft_delete = test_client.delete(
                f"/api/v1/quizzes/{owner_ids['owner-soft-delete']}",
                params={"expected_version": 1},
            )
            owner_hard_delete = test_client.delete(
                f"/api/v1/quizzes/{owner_ids['owner-hard-delete']}",
                params={"hard": True},
            )
            owner_list = test_client.get(
                "/api/v1/quizzes",
                params={"include_workspace_items": True},
            )

        assert attacker_list.status_code == 200
        assert {item["id"] for item in attacker_list.json()["items"]} == {attacker_id}
        assert [
            foreign_get.status_code,
            foreign_update.status_code,
            foreign_soft_delete.status_code,
            foreign_hard_delete.status_code,
            foreign_osce_delete.status_code,
        ] == [404] * 5
        assert owner_update.status_code == 200
        assert owner_update.json()["client_id"] == owner.client_id
        assert owner_soft_delete.status_code == 200
        assert owner_hard_delete.status_code == 200
        assert attacker_id not in {item["id"] for item in owner_list.json()["items"]}

        rows = owner.execute_query(
            "SELECT id, name, deleted, client_id FROM quizzes ORDER BY id"
        ).fetchall()
        by_id = {int(row["id"]): row for row in rows}
        assert by_id[owner_ids["update"]]["name"] == "Owner updated"
        assert by_id[owner_ids["update"]]["client_id"] == owner.client_id
        assert not bool(by_id[owner_ids["foreign-soft-delete"]]["deleted"])
        assert not bool(by_id[owner_ids["foreign-hard-delete"]]["deleted"])
        assert by_id[owner_ids["foreign-soft-delete"]]["client_id"] == owner.client_id
        assert by_id[owner_ids["foreign-hard-delete"]]["client_id"] == owner.client_id
        assert bool(by_id[owner_ids["owner-soft-delete"]]["deleted"])
        assert owner_ids["owner-hard-delete"] not in by_id
        assert not bool(by_id[osce_id]["deleted"])
        station_row = owner.execute_query(
            "SELECT deleted FROM osce_stations WHERE id = ?",
            (station["id"],),
        ).fetchone()
        assert station_row is not None
        assert not bool(station_row["deleted"])
    finally:
        fastapi_app.dependency_overrides.clear()
        TestConfig.reset_settings()
        attacker.close_all_connections()
        owner.close_all_connections()


@pytest.mark.timeout(120)
def test_postgres_question_and_attempt_http_routes_scope_children_to_owner(
    pg_database_config: DatabaseConfig,
) -> None:
    owner_backend = DatabaseBackendFactory.create_backend(pg_database_config)
    attacker_backend = DatabaseBackendFactory.create_backend(pg_database_config)
    owner = CharactersRAGDB(Path(":memory:"), client_id="child-http-owner", backend=owner_backend)
    attacker = CharactersRAGDB(
        Path(":memory:"), client_id="child-http-attacker", backend=attacker_backend
    )
    active: dict[str, Any] = {
        "db": owner,
        "user": User(
            id=301,
            username="child-http-owner",
            email="child-http-owner@example.com",
            is_active=True,
            roles=["admin"],
            is_admin=True,
        ),
    }

    def override_active_db():
        yield active["db"]

    async def override_active_user():
        return active["user"]

    try:
        quiz_id = owner.create_quiz(name="Private question quiz")
        other_quiz_id = owner.create_quiz(name="Other private quiz")
        foreign_update_id = owner.create_question(
            quiz_id, "true_false", "Foreign update target?", "true"
        )
        foreign_delete_id = owner.create_question(
            quiz_id, "true_false", "Foreign delete target?", "true"
        )
        mismatch_update_id = owner.create_question(
            quiz_id, "true_false", "Mismatch update target?", "true"
        )
        mismatch_delete_id = owner.create_question(
            quiz_id, "true_false", "Mismatch delete target?", "true"
        )
        attempt = owner.start_attempt(quiz_id)

        TestConfig.setup_test_environment()
        fastapi_app.dependency_overrides[get_chacha_db_for_user] = override_active_db
        fastapi_app.dependency_overrides[get_request_user] = override_active_user
        with TestClient(fastapi_app, headers=AUTH_HEADERS) as test_client:
            active["db"] = attacker
            active["user"] = User(
                id=302,
                username="child-http-attacker",
                email="child-http-attacker@example.com",
                is_active=True,
                roles=["admin"],
                is_admin=True,
            )
            foreign_update = test_client.patch(
                f"/api/v1/quizzes/{quiz_id}/questions/{foreign_update_id}",
                json={"question_text": "Captured", "expected_version": 1},
            )
            foreign_delete = test_client.delete(
                f"/api/v1/quizzes/{quiz_id}/questions/{foreign_delete_id}",
                params={"expected_version": 1},
            )
            foreign_list_questions = test_client.get(
                f"/api/v1/quizzes/{quiz_id}/questions"
            )
            foreign_start_attempt = test_client.post(
                f"/api/v1/quizzes/{quiz_id}/attempts"
            )
            foreign_get_attempt = test_client.get(
                f"/api/v1/quizzes/attempts/{attempt['id']}",
                params={"include_answers": True},
            )
            foreign_list_attempts = test_client.get("/api/v1/quizzes/attempts")
            foreign_submit_attempt = test_client.put(
                f"/api/v1/quizzes/attempts/{attempt['id']}",
                json={
                    "answers": [
                        {
                            "question_id": foreign_update_id,
                            "user_answer": "false",
                        }
                    ]
                },
            )

            active["db"] = owner
            active["user"] = User(
                id=301,
                username="child-http-owner",
                email="child-http-owner@example.com",
                is_active=True,
                roles=["admin"],
                is_admin=True,
            )
            mismatch_update = test_client.patch(
                f"/api/v1/quizzes/{other_quiz_id}/questions/{mismatch_update_id}",
                json={"question_text": "Wrong parent", "expected_version": 1},
            )
            mismatch_delete = test_client.delete(
                f"/api/v1/quizzes/{other_quiz_id}/questions/{mismatch_delete_id}",
                params={"expected_version": 1},
            )

        assert [
            foreign_update.status_code,
            foreign_delete.status_code,
            foreign_list_questions.status_code,
            foreign_start_attempt.status_code,
            foreign_get_attempt.status_code,
            foreign_submit_attempt.status_code,
            mismatch_update.status_code,
            mismatch_delete.status_code,
        ] == [404] * 8
        assert foreign_list_attempts.status_code == 200
        assert foreign_list_attempts.json()["items"] == []
        assert foreign_list_attempts.json()["count"] == 0

        quiz_row = owner.execute_query(
            "SELECT total_questions, client_id FROM quizzes WHERE id = ?",
            (quiz_id,),
        ).fetchone()
        question_rows = owner.execute_query(
            "SELECT id, question_text, deleted, client_id FROM quiz_questions "
            "WHERE quiz_id = ? ORDER BY id",
            (quiz_id,),
        ).fetchall()
        attempt_row = owner.execute_query(
            "SELECT completed_at, score, answers, client_id FROM quiz_attempts WHERE id = ?",
            (attempt["id"],),
        ).fetchone()
        assert quiz_row is not None
        assert int(quiz_row["total_questions"]) == 4
        assert quiz_row["client_id"] == owner.client_id
        assert [row["question_text"] for row in question_rows] == [
            "Foreign update target?",
            "Foreign delete target?",
            "Mismatch update target?",
            "Mismatch delete target?",
        ]
        assert all(not bool(row["deleted"]) for row in question_rows)
        assert all(row["client_id"] == owner.client_id for row in question_rows)
        assert attempt_row is not None
        assert attempt_row["completed_at"] is None
        assert attempt_row["score"] is None
        assert attempt_row["answers"] in ([], "[]")
        assert attempt_row["client_id"] == owner.client_id
    finally:
        fastapi_app.dependency_overrides.clear()
        TestConfig.reset_settings()
        attacker.close_all_connections()
        owner.close_all_connections()
