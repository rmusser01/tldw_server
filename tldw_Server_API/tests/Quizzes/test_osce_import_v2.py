"""Portable mixed quiz import contracts for OSCE scenario practice."""

from __future__ import annotations

import os
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

os.environ.setdefault("READING_DIGEST_JOBS_WORKER_ENABLED", "0")
os.environ.setdefault("READING_DIGEST_SCHEDULER_ENABLED", "0")
os.environ.setdefault("TEST_MODE", "1")

pytestmark = pytest.mark.integration

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import (  # noqa: E402
    get_chacha_db_for_user,
)
from tldw_Server_API.app.api.v1.schemas.quizzes import QuizExportV2  # noqa: E402
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import (  # noqa: E402
    User,
    get_request_user,
)
from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseConfig  # noqa: E402
from tldw_Server_API.app.core.DB_Management.backends.factory import (  # noqa: E402
    DatabaseBackendFactory,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (  # noqa: E402
    CharactersRAGDB,
    CharactersRAGDBError,
)
from tldw_Server_API.app.main import app as fastapi_app  # noqa: E402
from tldw_Server_API.tests.test_config import TestConfig  # noqa: E402

AUTH_HEADERS = {"X-API-KEY": TestConfig.TEST_API_KEY}
CHECKLIST_ID = "11111111-1111-4111-8111-111111111111"
DOMAIN_ID = "22222222-2222-4222-8222-222222222222"
LEVEL_ONE_ID = "33333333-3333-4333-8333-333333333333"
LEVEL_TWO_ID = "44444444-4444-4444-8444-444444444444"
KEY_POINT_ID = "55555555-5555-4555-8555-555555555555"


def _station_content(title: str = "Warfarin counselling") -> dict[str, Any]:
    return {
        "schema_version": "osce.station.v1",
        "title": title,
        "candidate_instructions": "Speak with a simulated patient.",
        "candidate_task": "Explain safe anticoagulant use.",
        "patient_context": {
            "text": "A fictional adult has recently started warfarin.",
            "citations": [
                {
                    "source_type": "note",
                    "source_id": "note-1",
                    "label": "Anticoagulation guide",
                    "quote": "Regular INR monitoring is required.",
                    "chunk_id": "note-1",
                }
            ],
        },
        "recommended_duration_seconds": 480,
        "checklist_items": [
            {
                "id": CHECKLIST_ID,
                "label": "Explains monitoring",
                "rationale": "Monitoring supports safe treatment.",
                "citations": [],
            }
        ],
        "rubric_domains": [
            {
                "id": DOMAIN_ID,
                "label": "Communication",
                "levels": [
                    {
                        "id": LEVEL_ONE_ID,
                        "label": "Developing",
                        "description": "The explanation is incomplete.",
                    },
                    {
                        "id": LEVEL_TWO_ID,
                        "label": "Effective",
                        "description": "The explanation is clear.",
                    },
                ],
            }
        ],
        "expected_key_points": [
            {
                "id": KEY_POINT_ID,
                "text": "Discusses monitoring and warning signs.",
                "citations": [],
            }
        ],
    }


def _station_export(title: str = "Warfarin counselling", order_index: int = 0) -> dict[str, Any]:
    return {
        "id": 700 + order_index,
        "quiz_id": 500,
        "content": _station_content(title),
        "order_index": order_index,
        "version": 8,
        "origin": "generated",
        "provenance": {
            "provider": "untrusted-provider",
            "verification": {"claimed": True},
        },
        "source_bundle": [{"source_type": "note", "source_id": "note-1"}],
        "verification_state": "source_verified",
        "verification_timestamp": "2026-09-10T12:00:00Z",
        "verification_summary": "Untrusted verification claim",
        "deleted": False,
        "created_at": "2026-09-10T11:00:00Z",
        "updated_at": "2026-09-10T12:00:00Z",
    }


def _osce_entry(*stations: dict[str, Any]) -> dict[str, Any]:
    return {
        "activity_type": "osce",
        "quiz": {
            "id": 500,
            "name": "Imported OSCE",
            "description": "Portable authoring content",
            "workspace_tag": None,
            "workspace_id": None,
            "media_id": None,
            "source_bundle_json": [{"source_type": "note", "source_id": "note-1"}],
            "activity_type": "osce",
            "generation_profile": "osce_scenario",
            "total_questions": 0,
            "total_stations": len(stations),
            "time_limit_seconds": None,
            "passing_score": None,
            "deleted": False,
            "client_id": "unknown",
            "version": 12,
            "created_at": "2026-09-10T11:00:00Z",
            "last_modified": "2026-09-10T12:00:00Z",
        },
        "stations": list(stations),
    }


def _question_entry() -> dict[str, Any]:
    return {
        "activity_type": "questions",
        "quiz": {
            "id": 900,
            "name": "Imported Questions",
            "description": "Legacy question content in v2",
            "workspace_tag": None,
            "workspace_id": None,
            "media_id": None,
            "source_bundle_json": None,
            "activity_type": "questions",
            "generation_profile": None,
            "total_questions": 1,
            "total_stations": 0,
            "time_limit_seconds": 600,
            "passing_score": 70,
            "deleted": False,
            "client_id": "untrusted-owner",
            "version": 4,
            "created_at": "2026-09-10T11:00:00Z",
            "last_modified": "2026-09-10T12:00:00Z",
        },
        "questions": [
            {
                "id": 901,
                "quiz_id": 900,
                "question_type": "true_false",
                "question_text": "Warfarin requires monitoring.",
                "group_id": None,
                "group_prompt": None,
                "options": None,
                "correct_answer": "true",
                "explanation": "Monitoring is required.",
                "hint": None,
                "hint_penalty_points": 0,
                "source_citations": None,
                "points": 1,
                "order_index": 0,
                "tags": ["anticoagulation"],
                "deleted": False,
                "client_id": "untrusted-owner",
                "version": 2,
                "created_at": "2026-09-10T11:00:00Z",
                "last_modified": "2026-09-10T12:00:00Z",
            }
        ],
    }


def _v2_payload(*entries: Any) -> dict[str, Any]:
    return {
        "export_format": "tldw.quiz.export.v2",
        "exported_at": "2026-09-10T12:00:00Z",
        "quizzes": list(entries),
    }


def _nested_ids(content: dict[str, Any]) -> set[UUID]:
    ids = {UUID(item["id"]) for item in content["checklist_items"]}
    for domain in content["rubric_domains"]:
        ids.add(UUID(domain["id"]))
        ids.update(UUID(level["id"]) for level in domain["levels"])
    ids.update(UUID(point["id"]) for point in content["expected_key_points"])
    return ids


@pytest.fixture
def quizzes_db(tmp_path) -> CharactersRAGDB:
    db = CharactersRAGDB(
        str(tmp_path / "osce-import-v2.db"),
        client_id=f"authenticated-{uuid4().hex[:8]}",
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


def test_v2_contract_is_strict_and_discriminated() -> None:
    payload = _v2_payload(_question_entry(), _osce_entry(_station_export()))
    payload["exported_at"] = datetime.now(timezone.utc)
    parsed = QuizExportV2.model_validate(payload)

    assert [entry.activity_type for entry in parsed.quizzes] == ["questions", "osce"]

    invalid = deepcopy(payload)
    invalid["quizzes"][1]["stations"][0]["candidate_notes"] = "must not be portable"
    with pytest.raises(ValidationError):
        QuizExportV2.model_validate(invalid)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("quiz", "name"), "n" * 256),
        (("quiz", "description"), "d" * 2001),
        (("quiz", "workspace_id"), "w" * 129),
        (("quiz", "workspace_tag"), "t" * 256),
        (("quiz", "client_id"), "c" * 256),
        (("quiz", "created_at"), "a" * 65),
        (("quiz", "source_bundle_json", 0, "source_id"), "s" * 513),
    ],
)
def test_v2_contract_bounds_portable_metadata(
    path: tuple[str | int, ...],
    value: str,
) -> None:
    payload = _v2_payload(_osce_entry(_station_export()))
    target: Any = payload["quizzes"][0]
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = value

    with pytest.raises(ValidationError):
        QuizExportV2.model_validate(payload)


def test_v2_mixed_import_rekeys_osce_and_downgrades_untrusted_state(
    client: TestClient,
    quizzes_db: CharactersRAGDB,
) -> None:
    exported_station = _station_export()
    response = client.post(
        "/api/v1/quizzes/import/json",
        json=_v2_payload(_question_entry(), _osce_entry(exported_station)),
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["imported_quizzes"] == 2
    assert payload["failed_quizzes"] == 0
    assert payload["imported_questions"] == 1
    assert payload["failed_questions"] == 0
    assert payload["imported_stations"] == 1
    assert payload["failed_stations"] == 0

    question_item, osce_item = payload["items"]
    assert question_item["imported_questions"] == 1
    assert question_item["station_ids"] == []
    assert osce_item["imported_stations"] == 1
    assert osce_item["failed_stations"] == 0
    assert len(osce_item["station_ids"]) == 1

    imported_quiz = quizzes_db.get_quiz(osce_item["quiz_id"])
    assert imported_quiz is not None
    assert imported_quiz["activity_type"] == "osce"
    assert imported_quiz["generation_profile"] is None
    assert imported_quiz["source_bundle_json"] is None
    assert imported_quiz["client_id"] == quizzes_db.client_id
    assert imported_quiz["client_id"] != "unknown"

    station_response = client.get(
        f"/api/v1/quizzes/{osce_item['quiz_id']}/osce-stations/{osce_item['station_ids'][0]}"
    )
    assert station_response.status_code == 200
    station = station_response.json()
    assert station["id"] != exported_station["id"]
    assert station["quiz_id"] != exported_station["quiz_id"]
    assert station["origin"] == "manual"
    assert station["provenance"] is None
    assert station["source_bundle"] == []
    assert station["verification_state"] == "manually_authored"
    assert station["verification_timestamp"] is None
    assert station["verification_summary"] is None
    assert _nested_ids(station["content"]).isdisjoint(
        _nested_ids(exported_station["content"])
    )
    assert quizzes_db.execute_query(
        "SELECT COUNT(*) AS count FROM osce_practice_attempts"
    ).fetchone()["count"] == 0


def test_v2_import_discards_malformed_protected_ids_before_validation(
    client: TestClient,
) -> None:
    station = _station_export()
    station["id"] = {"attacker": "station-id"}
    station["quiz_id"] = ["attacker-quiz-id"]
    content = station["content"]
    content["checklist_items"][0]["id"] = {"attacker": "checklist-id"}
    content["rubric_domains"][0]["id"] = ["attacker-domain-id"]
    content["rubric_domains"][0]["levels"][0]["id"] = "not-a-uuid"
    content["rubric_domains"][0]["levels"][1]["id"] = 123
    content["expected_key_points"][0]["id"] = None

    response = client.post(
        "/api/v1/quizzes/import/json",
        json=_v2_payload(_osce_entry(station)),
    )

    assert response.status_code == 200
    item = response.json()["items"][0]
    imported = client.get(
        f"/api/v1/quizzes/{item['quiz_id']}/osce-stations/{item['station_ids'][0]}"
    ).json()
    nested_ids = _nested_ids(imported["content"])
    assert len(nested_ids) == 5


def test_v1_import_remains_question_activity_and_reports_zero_station_counts(
    client: TestClient,
    quizzes_db: CharactersRAGDB,
) -> None:
    response = client.post(
        "/api/v1/quizzes/import/json",
        json={
            "export_format": "tldw.quiz.export.v1",
            "exported_at": "2026-09-10T12:00:00Z",
            "source": "quiz-manage-tab",
            "quiz_count": 1,
            "quizzes": [
                {
                    "quiz": {
                        "id": 42,
                        "name": "Legacy question quiz",
                        "activity_type": "osce",
                        "client_id": "unknown",
                        "total_questions": 0,
                    },
                    "questions": [],
                }
            ],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["imported_quizzes"] == 1
    assert payload["imported_stations"] == 0
    assert payload["failed_stations"] == 0
    assert payload["items"][0]["imported_questions"] == 0
    assert payload["items"][0]["imported_stations"] == 0
    assert payload["items"][0]["station_ids"] == []
    imported = quizzes_db.get_quiz(payload["items"][0]["quiz_id"])
    assert imported is not None
    assert imported["activity_type"] == "questions"
    assert imported["client_id"] == quizzes_db.client_id


def test_invalid_v2_osce_entry_does_not_leave_quiz_shell_and_valid_sibling_succeeds(
    client: TestClient,
    quizzes_db: CharactersRAGDB,
) -> None:
    invalid = _osce_entry(_station_export())
    invalid["stations"][0]["content"]["rubric_domains"][0]["levels"] = []
    invalid["stations"][0]["candidate_notes"] = "private candidate note /private/note.txt"

    response = client.post(
        "/api/v1/quizzes/import/json",
        json=_v2_payload(invalid, _question_entry()),
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["imported_quizzes"] == 1
    assert payload["failed_quizzes"] == 1
    assert payload["imported_questions"] == 1
    assert payload["failed_stations"] == 1
    assert len(payload["items"]) == 1
    assert payload["items"][0]["source_index"] == 1
    assert payload["errors"][0]["error"] == "Invalid OSCE quiz export entry"
    assert "private candidate note" not in str(payload)
    assert quizzes_db.execute_query(
        "SELECT COUNT(*) AS count FROM quizzes WHERE activity_type = 'osce'"
    ).fetchone()["count"] == 0
    assert quizzes_db.execute_query(
        "SELECT COUNT(*) AS count FROM osce_stations"
    ).fetchone()["count"] == 0
    assert quizzes_db.execute_query(
        "SELECT COUNT(*) AS count FROM osce_practice_attempts"
    ).fetchone()["count"] == 0


def test_v2_raw_entry_failures_are_isolated_and_count_raw_questions(
    client: TestClient,
) -> None:
    invalid_questions = _question_entry()
    invalid_questions["quiz"]["name"] = "Q" * 300 + "private-name-tail"
    invalid_questions["questions"] = [
        invalid_questions["questions"][0],
        {"question_type": "private-invalid-type", "question_text": "private question"},
    ]
    malformed_questions = _question_entry()
    malformed_questions["quiz"]["name"] = "Malformed questions"
    malformed_questions["questions"] = "private non-list question data"

    response = client.post(
        "/api/v1/quizzes/import/json",
        json=_v2_payload(
            "private non-mapping entry /private/import.json",
            invalid_questions,
            malformed_questions,
            _question_entry(),
        ),
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["imported_quizzes"] == 1
    assert payload["failed_quizzes"] == 3
    assert payload["imported_questions"] == 1
    assert payload["failed_questions"] == 2
    assert [item["source_index"] for item in payload["items"]] == [3]
    assert [error["error"] for error in payload["errors"]] == [
        "Invalid v2 quiz export entry",
        "Invalid question quiz export entry",
        "Invalid question quiz export entry",
    ]
    assert payload["errors"][0]["quiz_name"] is None
    assert len(payload["errors"][1]["quiz_name"]) == 255
    assert payload["errors"][2]["quiz_name"] == "Malformed questions"
    assert "private" not in str(payload).lower()


def test_v2_workspace_and_persistence_errors_are_fixed_and_bounded(
    client: TestClient,
    quizzes_db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing_workspace = _question_entry()
    missing_workspace["quiz"]["name"] = "Missing workspace"
    missing_workspace["quiz"]["workspace_id"] = "private-workspace-id"

    def fail_create_quiz(**_kwargs: Any) -> int:
        raise CharactersRAGDBError(
            "private persistence detail /private/import.db with sk-import-secret"
        )

    persistence_failure = _question_entry()
    persistence_failure["quiz"]["name"] = "Persistence failure"
    monkeypatch.setattr(quizzes_db, "create_quiz", fail_create_quiz)

    response = client.post(
        "/api/v1/quizzes/import/json",
        json=_v2_payload(missing_workspace, persistence_failure),
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["imported_quizzes"] == 0
    assert payload["failed_quizzes"] == 2
    assert payload["failed_questions"] == 2
    assert [error["error"] for error in payload["errors"]] == [
        "Failed to import question quiz",
        "Failed to import question quiz",
    ]
    assert "private-workspace-id" not in str(payload)
    assert "/private/import.db" not in str(payload)
    assert "sk-import-secret" not in str(payload)


def test_second_station_failure_rolls_back_quiz_and_all_stations(
    client: TestClient,
    quizzes_db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_insert = quizzes_db._insert_osce_station_row
    calls = 0

    def fail_second_insert(*args: Any, **kwargs: Any) -> int:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise CharactersRAGDBError(
                "private persistence detail /private/import.db with sk-import-secret"
            )
        return original_insert(*args, **kwargs)

    monkeypatch.setattr(quizzes_db, "_insert_osce_station_row", fail_second_insert)
    response = client.post(
        "/api/v1/quizzes/import/json",
        json=_v2_payload(
            _osce_entry(
                _station_export("First station", 0),
                _station_export("Second station", 1),
            )
        ),
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["imported_quizzes"] == 0
    assert payload["failed_quizzes"] == 1
    assert payload["imported_stations"] == 0
    assert payload["failed_stations"] == 2
    assert payload["items"] == []
    assert payload["errors"][0]["error"] == "Failed to import OSCE quiz"
    assert "/private/import.db" not in str(payload)
    assert "sk-import-secret" not in str(payload)
    assert quizzes_db.execute_query(
        "SELECT COUNT(*) AS count FROM quizzes"
    ).fetchone()["count"] == 0
    assert quizzes_db.execute_query(
        "SELECT COUNT(*) AS count FROM osce_stations"
    ).fetchone()["count"] == 0


@pytest.mark.timeout(120)
def test_postgres_v2_import_ignores_owner_and_provenance_claims(
    pg_database_config: DatabaseConfig,
) -> None:
    owner_backend = DatabaseBackendFactory.create_backend(pg_database_config)
    attacker_backend = DatabaseBackendFactory.create_backend(pg_database_config)
    owner = CharactersRAGDB(
        Path(":memory:"),
        client_id="postgres-import-owner",
        backend=owner_backend,
    )
    attacker = CharactersRAGDB(
        Path(":memory:"),
        client_id="postgres-import-attacker",
        backend=attacker_backend,
    )

    def override_get_db():
        yield owner

    async def override_user():
        return User(
            id=101,
            username="postgres-import-owner",
            email="postgres-import-owner@example.com",
            is_active=True,
            roles=["admin"],
            is_admin=True,
        )

    try:
        TestConfig.setup_test_environment()
        fastapi_app.dependency_overrides[get_chacha_db_for_user] = override_get_db
        fastapi_app.dependency_overrides[get_request_user] = override_user
        entry = _osce_entry(_station_export())
        entry["quiz"]["workspace_tag"] = "workspace:postgres-import"
        entry["quiz"]["client_id"] = "unknown"
        entry["stations"][0]["provenance"] = {
            "client_id": "postgres-import-attacker",
            "verification": "attacker-claimed",
        }

        with TestClient(fastapi_app, headers=AUTH_HEADERS) as test_client:
            response = test_client.post(
                "/api/v1/quizzes/import/json",
                json=_v2_payload(entry),
            )

        assert response.status_code == 200, response.text
        item = response.json()["items"][0]
        quiz_id = item["quiz_id"]
        station_id = item["station_ids"][0]
        stored_quiz = owner.get_quiz(quiz_id)
        stored_station = owner.get_osce_station(quiz_id, station_id)
        assert stored_quiz is not None
        assert stored_quiz["client_id"] == owner.client_id
        assert stored_quiz["client_id"] not in {"unknown", attacker.client_id}
        assert stored_station is not None
        assert stored_station["origin"] == "manual"
        assert stored_station["provenance"] is None
        assert stored_station["source_bundle"] == []
        assert stored_station["verification_state"] == "manually_authored"
        assert owner.list_quizzes(
            workspace_tag="workspace:postgres-import",
            activity_type="osce",
            include_workspace_items=True,
        )["count"] == 1
        assert attacker.get_quiz(quiz_id) is None
        assert attacker.list_quizzes(
            workspace_tag="workspace:postgres-import",
            activity_type="osce",
            include_workspace_items=True,
        ) == {"items": [], "count": 0}
    finally:
        fastapi_app.dependency_overrides.clear()
        TestConfig.reset_settings()
        attacker.close_all_connections()
        owner.close_all_connections()
