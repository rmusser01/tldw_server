import json
import os
import uuid

import pytest
from fastapi.testclient import TestClient
from loguru import logger

# Keep this module self-contained and deterministic by disabling optional
# reading-digest startup paths that pull heavyweight STT deps during app import.
os.environ.setdefault("READING_DIGEST_JOBS_WORKER_ENABLED", "0")
os.environ.setdefault("READING_DIGEST_SCHEDULER_ENABLED", "0")
os.environ.setdefault("TEST_MODE", "1")

from tldw_Server_API.app.main import app as fastapi_app
from tldw_Server_API.app.api.v1.endpoints.quizzes import (
    convert_attempt_remediation_conversions,
    get_attempt_remediation_conversions,
)
from tldw_Server_API.app.api.v1.schemas.quizzes import (
    QuizRemediationConversionListResponse,
    QuizRemediationConvertResponse,
)
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, ConflictError
from tldw_Server_API.app.services import quiz_generator
from tldw_Server_API.tests.test_config import TestConfig

AUTH_HEADERS = {"X-API-KEY": TestConfig.TEST_API_KEY}


@pytest.fixture(scope="function")
def quizzes_db(tmp_path):
    db_path = tmp_path / "quizzes.db"
    db = CharactersRAGDB(str(db_path), client_id=f"test-{uuid.uuid4().hex[:6]}")
    yield db
    db.close_connection()


@pytest.fixture
def client_with_quizzes_db(quizzes_db: CharactersRAGDB):
    TestConfig.setup_test_environment()
    from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user

    def override_get_db():

        logger.info("[TEST] override get_chacha_db_for_user -> quizzes_db")
        try:
            yield quizzes_db
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

    fastapi_app.dependency_overrides[get_chacha_db_for_user] = override_get_db
    fastapi_app.dependency_overrides[get_request_user] = override_user

    with TestClient(fastapi_app, headers=AUTH_HEADERS) as client:
        yield client
    fastapi_app.dependency_overrides.clear()
    TestConfig.reset_settings()


def _create_attempt_with_missed_questions(quizzes_db: CharactersRAGDB) -> tuple[int, list[int]]:
    quiz_id = quizzes_db.create_quiz(name="Remediation Source Quiz")
    question_ids = [
        quizzes_db.create_question(
            quiz_id=quiz_id,
            question_type="multiple_choice",
            question_text="Which structure filters blood?",
            correct_answer=1,
            options=["Loop of Henle", "Glomerulus", "Collecting duct"],
            explanation="The glomerulus performs the initial blood filtration step.",
            source_citations=[
                {
                    "source_type": "note",
                    "source_id": "renal-note",
                    "quote": "Glomeruli filter blood.",
                }
            ],
            points=2,
            order_index=0,
        ),
        quizzes_db.create_question(
            quiz_id=quiz_id,
            question_type="multiple_choice",
            question_text="Which structure concentrates urine?",
            correct_answer=0,
            options=["Loop of Henle", "Glomerulus", "Collecting duct"],
            explanation="The Loop of Henle builds the medullary osmotic gradient.",
            source_citations=[
                {
                    "source_type": "note",
                    "source_id": "renal-note",
                    "quote": "The loop of Henle helps concentrate urine.",
                }
            ],
            points=1,
            order_index=1,
        ),
    ]
    attempt = quizzes_db.start_attempt(quiz_id)
    quizzes_db.submit_attempt(
        int(attempt["id"]),
        answers=[
            {"question_id": question_ids[0], "user_answer": 2, "time_spent_ms": 1400},
            {"question_id": question_ids[1], "user_answer": 2, "time_spent_ms": 900},
        ],
    )
    return int(attempt["id"]), question_ids


def test_remediation_endpoints_declare_return_annotations():
    assert get_attempt_remediation_conversions.__annotations__["return"] is QuizRemediationConversionListResponse
    assert convert_attempt_remediation_conversions.__annotations__["return"] is QuizRemediationConvertResponse


def test_quiz_source_bundle_roundtrip_via_db_create_and_get(quizzes_db: CharactersRAGDB):
    quiz_id = quizzes_db.create_quiz(
        name="Source Bundle Quiz",
        source_bundle_json=[{"source_type": "note", "source_id": "note-1"}],
    )

    row = quizzes_db.get_quiz(quiz_id)
    assert row is not None
    assert row["source_bundle_json"] == [{"source_type": "note", "source_id": "note-1"}]


def test_quiz_response_includes_source_bundle(client_with_quizzes_db: TestClient):
    create_response = client_with_quizzes_db.post(
        "/api/v1/quizzes",
        json={
            "name": "API Source Bundle Quiz",
            "source_bundle_json": [{"source_type": "note", "source_id": "note-1"}],
        },
        headers=AUTH_HEADERS,
    )
    assert create_response.status_code == 200
    created = create_response.json()
    assert created["source_bundle_json"] == [{"source_type": "note", "source_id": "note-1"}]

    quiz_id = created["id"]
    fetch_response = client_with_quizzes_db.get(f"/api/v1/quizzes/{quiz_id}", headers=AUTH_HEADERS)
    assert fetch_response.status_code == 200
    fetched = fetch_response.json()
    assert fetched["source_bundle_json"] == [{"source_type": "note", "source_id": "note-1"}]


def test_quiz_endpoints_flow(client_with_quizzes_db: TestClient):
    r = client_with_quizzes_db.post(
        "/api/v1/quizzes",
        json={"name": "Quiz One", "description": "desc"},
        headers=AUTH_HEADERS,
    )
    assert r.status_code == 200
    quiz_id = r.json()["id"]

    r = client_with_quizzes_db.post(
        f"/api/v1/quizzes/{quiz_id}/questions",
        json={
            "question_type": "multiple_choice",
            "question_text": "What is 2+2?",
            "options": ["1", "2", "4", "5"],
            "correct_answer": 2,
            "points": 1,
        },
        headers=AUTH_HEADERS,
    )
    assert r.status_code == 200
    question_id = r.json()["id"]

    r = client_with_quizzes_db.get(f"/api/v1/quizzes/{quiz_id}/questions", headers=AUTH_HEADERS)
    assert r.status_code == 200
    questions_payload = r.json()
    questions = questions_payload["items"]
    assert questions
    assert "correct_answer" not in questions[0]

    r = client_with_quizzes_db.get(
        f"/api/v1/quizzes/{quiz_id}/questions",
        params={"include_answers": True},
        headers=AUTH_HEADERS,
    )
    assert r.status_code == 200
    admin_payload = r.json()
    assert admin_payload["items"][0]["correct_answer"] == 2

    r = client_with_quizzes_db.post(f"/api/v1/quizzes/{quiz_id}/attempts", headers=AUTH_HEADERS)
    assert r.status_code == 200
    attempt = r.json()
    assert attempt["quiz_id"] == quiz_id
    assert attempt["questions"]
    assert "correct_answer" not in attempt["questions"][0]

    r = client_with_quizzes_db.put(
        f"/api/v1/quizzes/attempts/{attempt['id']}",
        json={"answers": [{"question_id": question_id, "user_answer": 2}]},
        headers=AUTH_HEADERS,
    )
    assert r.status_code == 200
    result = r.json()
    assert result["score"] == 1
    assert result["answers"][0]["is_correct"] is True


def test_quiz_workspace_id_contract_and_scope_filters(
    client_with_quizzes_db: TestClient,
    quizzes_db: CharactersRAGDB,
):
    quizzes_db.upsert_workspace("ws-1", "Workspace One")

    create_response = client_with_quizzes_db.post(
        "/api/v1/quizzes",
        json={
            "name": "Scoped Quiz",
            "workspace_id": "ws-1",
            "workspace_tag": "workspace:legacy",
        },
        headers=AUTH_HEADERS,
    )
    assert create_response.status_code == 200
    created = create_response.json()
    assert created["workspace_id"] == "ws-1"
    assert created["workspace_tag"] == "workspace:legacy"

    quiz_id = created["id"]

    default_list = client_with_quizzes_db.get("/api/v1/quizzes", headers=AUTH_HEADERS)
    assert default_list.status_code == 200
    assert all(item["id"] != quiz_id for item in default_list.json()["items"])

    workspace_list = client_with_quizzes_db.get(
        "/api/v1/quizzes",
        params={"workspace_id": "ws-1"},
        headers=AUTH_HEADERS,
    )
    assert workspace_list.status_code == 200
    assert [item["id"] for item in workspace_list.json()["items"]] == [quiz_id]

    all_list = client_with_quizzes_db.get(
        "/api/v1/quizzes",
        params={"include_workspace_items": True},
        headers=AUTH_HEADERS,
    )
    assert all_list.status_code == 200
    assert any(item["id"] == quiz_id for item in all_list.json()["items"])

    move_to_general = client_with_quizzes_db.patch(
        f"/api/v1/quizzes/{quiz_id}",
        json={"workspace_id": None, "expected_version": created["version"]},
        headers=AUTH_HEADERS,
    )
    assert move_to_general.status_code == 200
    moved = move_to_general.json()
    assert moved["workspace_id"] is None
    assert moved["workspace_tag"] == "workspace:legacy"

    general_list = client_with_quizzes_db.get("/api/v1/quizzes", headers=AUTH_HEADERS)
    assert general_list.status_code == 200
    assert any(item["id"] == quiz_id for item in general_list.json()["items"])


def test_quiz_endpoints_reject_unknown_workspace_ids(
    client_with_quizzes_db: TestClient,
):
    create_response = client_with_quizzes_db.post(
        "/api/v1/quizzes",
        json={"name": "Bad Quiz", "workspace_id": "missing-ws"},
        headers=AUTH_HEADERS,
    )
    assert create_response.status_code == 404
    assert "missing-ws" in create_response.json()["detail"]

    valid_quiz = client_with_quizzes_db.post(
        "/api/v1/quizzes",
        json={"name": "Good Quiz"},
        headers=AUTH_HEADERS,
    )
    assert valid_quiz.status_code == 200
    quiz_id = valid_quiz.json()["id"]

    update_response = client_with_quizzes_db.patch(
        f"/api/v1/quizzes/{quiz_id}",
        json={"workspace_id": "missing-ws"},
        headers=AUTH_HEADERS,
    )
    assert update_response.status_code == 404
    assert "missing-ws" in update_response.json()["detail"]

    import_response = client_with_quizzes_db.post(
        "/api/v1/quizzes/import/json",
        json={
            "export_format": "tldw.quiz.export.v1",
            "quizzes": [
                {
                    "quiz": {
                        "name": "Imported Quiz",
                        "workspace_id": "missing-ws",
                    },
                    "questions": [],
                }
            ],
        },
        headers=AUTH_HEADERS,
    )
    assert import_response.status_code == 200
    payload = import_response.json()
    assert payload["imported_quizzes"] == 0
    assert payload["failed_quizzes"] == 1
    assert payload["errors"][0]["error"] == "Workspace 'missing-ws' not found"


def test_quiz_patch_updates_workspace_scope_fields_together(client_with_quizzes_db: TestClient):
    workspace_response = client_with_quizzes_db.put(
        "/api/v1/workspaces/ws-1",
        json={"name": "Workspace One"},
        headers=AUTH_HEADERS,
    )
    assert workspace_response.status_code == 200

    create_response = client_with_quizzes_db.post(
        "/api/v1/quizzes",
        json={"name": "Quiz Scope Update", "workspace_id": "ws-1", "workspace_tag": "workspace:ws-1"},
        headers=AUTH_HEADERS,
    )
    assert create_response.status_code == 200
    created = create_response.json()
    quiz_id = created["id"]

    update_response = client_with_quizzes_db.patch(
        f"/api/v1/quizzes/{quiz_id}",
        json={
            "workspace_id": None,
            "workspace_tag": None,
            "expected_version": created["version"],
        },
        headers=AUTH_HEADERS,
    )
    assert update_response.status_code == 200
    updated = update_response.json()
    assert updated["workspace_id"] is None
    assert updated["workspace_tag"] is None


def test_quiz_list_ignores_workspace_tag_query_param(client_with_quizzes_db: TestClient):
    first_response = client_with_quizzes_db.post(
        "/api/v1/quizzes",
        json={"name": "Quiz One", "workspace_tag": "workspace:first"},
        headers=AUTH_HEADERS,
    )
    second_response = client_with_quizzes_db.post(
        "/api/v1/quizzes",
        json={"name": "Quiz Two", "workspace_tag": "workspace:second"},
        headers=AUTH_HEADERS,
    )
    assert first_response.status_code == 200
    assert second_response.status_code == 200

    baseline = client_with_quizzes_db.get("/api/v1/quizzes", headers=AUTH_HEADERS)
    filtered = client_with_quizzes_db.get(
        "/api/v1/quizzes",
        params={"workspace_tag": "workspace:first"},
        headers=AUTH_HEADERS,
    )
    assert baseline.status_code == 200
    assert filtered.status_code == 200
    assert filtered.json()["count"] == baseline.json()["count"]


def test_attempts_list_route_is_not_shadowed_by_quiz_id(client_with_quizzes_db: TestClient):
    create_quiz_response = client_with_quizzes_db.post(
        "/api/v1/quizzes",
        json={"name": "Attempts Route Quiz"},
        headers=AUTH_HEADERS,
    )
    assert create_quiz_response.status_code == 200
    quiz_id = create_quiz_response.json()["id"]

    create_question_response = client_with_quizzes_db.post(
        f"/api/v1/quizzes/{quiz_id}/questions",
        json={
            "question_type": "multiple_choice",
            "question_text": "Pick two",
            "options": ["1", "2", "3"],
            "correct_answer": 1,
            "points": 1,
        },
        headers=AUTH_HEADERS,
    )
    assert create_question_response.status_code == 200
    question_id = create_question_response.json()["id"]

    start_attempt_response = client_with_quizzes_db.post(
        f"/api/v1/quizzes/{quiz_id}/attempts",
        headers=AUTH_HEADERS,
    )
    assert start_attempt_response.status_code == 200
    attempt_id = start_attempt_response.json()["id"]

    submit_attempt_response = client_with_quizzes_db.put(
        f"/api/v1/quizzes/attempts/{attempt_id}",
        json={"answers": [{"question_id": question_id, "user_answer": 1}]},
        headers=AUTH_HEADERS,
    )
    assert submit_attempt_response.status_code == 200

    list_attempts_response = client_with_quizzes_db.get(
        "/api/v1/quizzes/attempts",
        params={"quiz_id": quiz_id, "limit": 20, "offset": 0},
        headers=AUTH_HEADERS,
    )
    assert list_attempts_response.status_code == 200
    attempts_payload = list_attempts_response.json()
    assert attempts_payload["count"] >= 1
    assert any(item["id"] == attempt_id for item in attempts_payload["items"])
    assert all(item["quiz_id"] == quiz_id for item in attempts_payload["items"])


def test_get_quiz_attempt_question_assistant_returns_thread_messages_and_context(
    client_with_quizzes_db: TestClient,
    quizzes_db: CharactersRAGDB,
):
    attempt_id, question_ids = _create_attempt_with_missed_questions(quizzes_db)
    question_id = question_ids[0]

    thread = quizzes_db.get_or_create_study_assistant_thread(
        context_type="quiz_attempt_question",
        quiz_attempt_id=attempt_id,
        question_id=question_id,
    )
    quizzes_db.append_study_assistant_message(
        thread_id=thread["id"],
        role="assistant",
        action_type="explain",
        input_modality="text",
        content="Start by comparing your answer with the filtration site.",
        structured_payload={"kind": "hint"},
        context_snapshot={"attempt_id": attempt_id, "question_id": question_id},
    )

    response = client_with_quizzes_db.get(
        f"/api/v1/quizzes/attempts/{attempt_id}/questions/{question_id}/assistant",
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["thread"]["id"] == thread["id"]
    assert payload["messages"][0]["content"] == "Start by comparing your answer with the filtration site."
    assert payload["context_snapshot"]["attempt"]["id"] == attempt_id
    assert payload["context_snapshot"]["question"]["id"] == question_id
    assert "follow_up" in payload["available_actions"]


def test_get_attempt_remediation_conversions_returns_active_rows(
    client_with_quizzes_db: TestClient,
    quizzes_db: CharactersRAGDB,
):
    attempt_id, question_ids = _create_attempt_with_missed_questions(quizzes_db)
    conversion_payload = quizzes_db.convert_quiz_remediation_questions(
        attempt_id=attempt_id,
        question_ids=[question_ids[0]],
        create_deck_name="Renal Remediation",
        replace_active=False,
    )

    response = client_with_quizzes_db.get(
        f"/api/v1/quizzes/attempts/{attempt_id}/remediation-conversions",
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["attempt_id"] == attempt_id
    assert payload["count"] == 1
    assert payload["superseded_count"] == 0
    assert payload["items"][0]["question_id"] == question_ids[0]
    assert payload["items"][0]["target_deck_id"] == conversion_payload["target_deck"]["id"]
    assert payload["items"][0]["orphaned"] is False
    assert payload["items"][0]["superseded_count"] == 0


def test_convert_remediation_questions_creates_new_deck(
    client_with_quizzes_db: TestClient,
    quizzes_db: CharactersRAGDB,
):
    attempt_id, question_ids = _create_attempt_with_missed_questions(quizzes_db)

    response = client_with_quizzes_db.post(
        f"/api/v1/quizzes/attempts/{attempt_id}/remediation-conversions/convert",
        json={
            "question_ids": question_ids,
            "create_deck_name": "Quiz Remediation Deck",
            "replace_active": False,
        },
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["attempt_id"] == attempt_id
    assert payload["target_deck"]["name"] == "Quiz Remediation Deck"
    assert {result["status"] for result in payload["results"]} == {"created"}
    assert len(payload["created_flashcard_uuids"]) == 2


def test_convert_remediation_questions_creates_new_deck_with_scheduler_settings(
    client_with_quizzes_db: TestClient,
    quizzes_db: CharactersRAGDB,
):
    attempt_id, question_ids = _create_attempt_with_missed_questions(quizzes_db)

    response = client_with_quizzes_db.post(
        f"/api/v1/quizzes/attempts/{attempt_id}/remediation-conversions/convert",
        json={
            "question_ids": question_ids,
            "create_deck_name": "Quiz Remediation Deck",
            "create_deck_review_prompt_side": "back",
            "create_deck_scheduler_type": "fsrs",
            "create_deck_scheduler_settings": {
                "sm2_plus": {
                    "new_steps_minutes": [1, 5, 15],
                    "relearn_steps_minutes": [10],
                    "graduating_interval_days": 1,
                    "easy_interval_days": 3,
                    "easy_bonus": 1.15,
                    "interval_modifier": 0.9,
                    "max_interval_days": 3650,
                    "leech_threshold": 10,
                    "enable_fuzz": False,
                },
                "fsrs": {
                    "target_retention": 0.95,
                    "maximum_interval_days": 1825,
                    "enable_fuzz": True,
                },
            },
            "replace_active": False,
        },
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 200
    payload = response.json()
    created_deck = quizzes_db.get_deck(payload["target_deck"]["id"])
    assert created_deck is not None
    assert created_deck["review_prompt_side"] == "back"
    assert created_deck["scheduler_type"] == "fsrs"
    scheduler_settings = json.loads(created_deck["scheduler_settings_json"])
    assert scheduler_settings["sm2_plus"]["new_steps_minutes"] == [1, 5, 15]
    assert scheduler_settings["sm2_plus"]["easy_interval_days"] == 3
    assert scheduler_settings["fsrs"]["target_retention"] == pytest.approx(0.95)
    assert scheduler_settings["fsrs"]["maximum_interval_days"] == 1825
    assert scheduler_settings["fsrs"]["enable_fuzz"] is True


def test_convert_remediation_questions_rejects_null_review_prompt_side(
    client_with_quizzes_db: TestClient,
    quizzes_db: CharactersRAGDB,
):
    attempt_id, question_ids = _create_attempt_with_missed_questions(quizzes_db)

    response = client_with_quizzes_db.post(
        f"/api/v1/quizzes/attempts/{attempt_id}/remediation-conversions/convert",
        json={
            "question_ids": question_ids,
            "create_deck_name": "Quiz Remediation Deck",
            "create_deck_review_prompt_side": None,
            "replace_active": False,
        },
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 422


def test_convert_remediation_questions_preserves_orientation_when_omitted_and_allows_front_override(
    client_with_quizzes_db: TestClient,
    quizzes_db: CharactersRAGDB,
):
    attempt_id, question_ids = _create_attempt_with_missed_questions(quizzes_db)
    deck_id = quizzes_db.add_deck("Quiz Orientation Deck", review_prompt_side="back")
    quizzes_db.soft_delete_deck_by_id(deck_id)

    omitted_response = client_with_quizzes_db.post(
        f"/api/v1/quizzes/attempts/{attempt_id}/remediation-conversions/convert",
        json={
            "question_ids": question_ids,
            "create_deck_name": "Quiz Orientation Deck",
            "replace_active": False,
        },
        headers=AUTH_HEADERS,
    )
    assert omitted_response.status_code == 200
    omitted_deck = quizzes_db.get_deck(deck_id)
    assert omitted_deck is not None
    assert omitted_deck["review_prompt_side"] == "back"

    quizzes_db.soft_delete_deck_by_id(deck_id)
    attempt_id_front, question_ids_front = _create_attempt_with_missed_questions(quizzes_db)
    explicit_front_response = client_with_quizzes_db.post(
        f"/api/v1/quizzes/attempts/{attempt_id_front}/remediation-conversions/convert",
        json={
            "question_ids": question_ids_front,
            "create_deck_name": "Quiz Orientation Deck",
            "create_deck_review_prompt_side": "front",
            "replace_active": False,
        },
        headers=AUTH_HEADERS,
    )
    assert explicit_front_response.status_code == 200
    explicit_front_deck = quizzes_db.get_deck(deck_id)
    assert explicit_front_deck is not None
    assert explicit_front_deck["review_prompt_side"] == "front"


def test_convert_remediation_questions_returns_mixed_results_without_replace_active(
    client_with_quizzes_db: TestClient,
    quizzes_db: CharactersRAGDB,
):
    attempt_id, question_ids = _create_attempt_with_missed_questions(quizzes_db)
    deck_id = quizzes_db.add_deck("Renal Deck")
    quizzes_db.convert_quiz_remediation_questions(
        attempt_id=attempt_id,
        question_ids=[question_ids[0]],
        target_deck_id=deck_id,
        replace_active=False,
    )

    response = client_with_quizzes_db.post(
        f"/api/v1/quizzes/attempts/{attempt_id}/remediation-conversions/convert",
        json={
            "question_ids": question_ids,
            "target_deck_id": deck_id,
            "replace_active": False,
        },
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 200
    payload = response.json()
    statuses = {result["question_id"]: result["status"] for result in payload["results"]}
    assert statuses[question_ids[0]] == "already_exists"
    assert statuses[question_ids[1]] == "created"
    assert len(payload["created_flashcard_uuids"]) == 1


def test_convert_remediation_questions_replace_active_returns_superseded_and_created(
    client_with_quizzes_db: TestClient,
    quizzes_db: CharactersRAGDB,
):
    attempt_id, question_ids = _create_attempt_with_missed_questions(quizzes_db)
    deck_id = quizzes_db.add_deck("Renal Deck")
    quizzes_db.convert_quiz_remediation_questions(
        attempt_id=attempt_id,
        question_ids=[question_ids[0]],
        target_deck_id=deck_id,
        replace_active=False,
    )

    response = client_with_quizzes_db.post(
        f"/api/v1/quizzes/attempts/{attempt_id}/remediation-conversions/convert",
        json={
            "question_ids": [question_ids[0]],
            "target_deck_id": deck_id,
            "replace_active": True,
        },
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["results"][0]["status"] == "superseded_and_created"

    read_response = client_with_quizzes_db.get(
        f"/api/v1/quizzes/attempts/{attempt_id}/remediation-conversions",
        headers=AUTH_HEADERS,
    )
    assert read_response.status_code == 200
    read_payload = read_response.json()
    assert read_payload["count"] == 1
    assert read_payload["superseded_count"] == 1
    assert read_payload["items"][0]["superseded_count"] == 1


def test_convert_remediation_questions_returns_409_for_conflict(
    client_with_quizzes_db: TestClient,
    quizzes_db: CharactersRAGDB,
    monkeypatch,
):
    attempt_id, question_ids = _create_attempt_with_missed_questions(quizzes_db)
    deck_id = quizzes_db.add_deck("Renal Deck")

    def raise_conflict(**_kwargs):
        raise ConflictError("Active remediation conversion already exists.")

    monkeypatch.setattr(quizzes_db, "convert_quiz_remediation_questions", raise_conflict)

    response = client_with_quizzes_db.post(
        f"/api/v1/quizzes/attempts/{attempt_id}/remediation-conversions/convert",
        json={
            "question_ids": [question_ids[0]],
            "target_deck_id": deck_id,
            "replace_active": False,
        },
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 409
    assert "already exists" in response.json()["detail"].lower()


def test_get_attempt_remediation_conversions_marks_orphaned_linked_flashcards(
    client_with_quizzes_db: TestClient,
    quizzes_db: CharactersRAGDB,
    monkeypatch,
):
    attempt_id, question_ids = _create_attempt_with_missed_questions(quizzes_db)
    conversion_payload = quizzes_db.convert_quiz_remediation_questions(
        attempt_id=attempt_id,
        question_ids=[question_ids[0]],
        create_deck_name="Renal Remediation",
        replace_active=False,
    )
    flashcard_uuid = conversion_payload["created_flashcard_uuids"][0]
    flashcard = quizzes_db.get_flashcard(flashcard_uuid)
    assert flashcard is not None
    quizzes_db.soft_delete_flashcard(flashcard_uuid, expected_version=int(flashcard["version"]))

    batch_lookup_calls: list[list[str]] = []
    original_batch_lookup = quizzes_db.get_flashcards_by_uuids

    def track_batch_lookup(uuids: list[str]):
        batch_lookup_calls.append(list(uuids))
        return original_batch_lookup(uuids)

    def fail_single_lookup(_uuid: str):
        raise AssertionError("Expected remediation orphan detection to use batch flashcard lookup.")

    monkeypatch.setattr(quizzes_db, "get_flashcards_by_uuids", track_batch_lookup)
    monkeypatch.setattr(quizzes_db, "get_flashcard", fail_single_lookup)

    response = client_with_quizzes_db.get(
        f"/api/v1/quizzes/attempts/{attempt_id}/remediation-conversions",
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["items"][0]["question_id"] == question_ids[0]
    assert payload["items"][0]["orphaned"] is True
    assert batch_lookup_calls == [[flashcard_uuid]]


def test_quiz_attempt_question_assistant_respond_persists_user_and_assistant_messages(
    client_with_quizzes_db: TestClient,
    quizzes_db: CharactersRAGDB,
    monkeypatch,
):
    attempt_id, question_ids = _create_attempt_with_missed_questions(quizzes_db)
    question_id = question_ids[0]

    async def fake_generate_reply(*, action, context, message=None, provider=None, model=None):
        assert action == "explain"
        assert context["attempt"]["id"] == attempt_id
        assert context["question"]["id"] == question_id
        return {
            "assistant_text": "The glomerulus is where blood filtration begins.",
            "structured_payload": {},
            "provider": provider or "openai",
            "model": model or "gpt-test",
        }

    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.quizzes.generate_study_assistant_reply",
        fake_generate_reply,
    )

    response = client_with_quizzes_db.post(
        f"/api/v1/quizzes/attempts/{attempt_id}/questions/{question_id}/assistant/respond",
        json={"action": "explain"},
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["user_message"]["role"] == "user"
    assert payload["assistant_message"]["role"] == "assistant"
    assert payload["assistant_message"]["content"] == "The glomerulus is where blood filtration begins."
    assert payload["thread"]["message_count"] == 2


def test_quiz_attempt_question_assistant_respond_returns_409_for_stale_thread_version(
    client_with_quizzes_db: TestClient,
    quizzes_db: CharactersRAGDB,
    monkeypatch,
):
    attempt_id, question_ids = _create_attempt_with_missed_questions(quizzes_db)
    question_id = question_ids[0]

    context_response = client_with_quizzes_db.get(
        f"/api/v1/quizzes/attempts/{attempt_id}/questions/{question_id}/assistant",
        headers=AUTH_HEADERS,
    )
    assert context_response.status_code == 200
    stale_version = int(context_response.json()["thread"]["version"])
    thread_id = int(context_response.json()["thread"]["id"])

    quizzes_db.append_study_assistant_message(
        thread_id=thread_id,
        role="user",
        action_type="follow_up",
        input_modality="text",
        content="Concurrent message",
    )

    call_count = 0

    async def fake_generate_reply(*, action, context, message=None, provider=None, model=None):
        nonlocal call_count
        call_count += 1
        return {
            "assistant_text": "The glomerulus is where blood filtration begins.",
            "structured_payload": {},
            "provider": provider or "openai",
            "model": model or "gpt-test",
        }

    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.quizzes.generate_study_assistant_reply",
        fake_generate_reply,
    )

    response = client_with_quizzes_db.post(
        f"/api/v1/quizzes/attempts/{attempt_id}/questions/{question_id}/assistant/respond",
        json={"action": "explain", "expected_thread_version": stale_version},
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "Study assistant thread version mismatch"
    assert call_count == 0


def test_quiz_multi_select_endpoints_flow(client_with_quizzes_db: TestClient):
    response = client_with_quizzes_db.post(
        "/api/v1/quizzes",
        json={"name": "Multi Select Quiz"},
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    quiz_id = response.json()["id"]

    response = client_with_quizzes_db.post(
        f"/api/v1/quizzes/{quiz_id}/questions",
        json={
            "question_type": "multi_select",
            "question_text": "Select even numbers",
            "options": ["1", "2", "3", "4"],
            "correct_answer": [1, 3],
            "points": 2,
        },
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    question_id = response.json()["id"]
    assert response.json()["correct_answer"] == [1, 3]

    response = client_with_quizzes_db.post(
        f"/api/v1/quizzes/{quiz_id}/attempts",
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    attempt_id = response.json()["id"]

    response = client_with_quizzes_db.put(
        f"/api/v1/quizzes/attempts/{attempt_id}",
        json={"answers": [{"question_id": question_id, "user_answer": [3, 1]}]},
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["score"] == 2
    assert payload["answers"][0]["is_correct"] is True


def test_quiz_matching_endpoints_flow(client_with_quizzes_db: TestClient):
    response = client_with_quizzes_db.post(
        "/api/v1/quizzes",
        json={"name": "Matching Quiz"},
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    quiz_id = response.json()["id"]

    response = client_with_quizzes_db.post(
        f"/api/v1/quizzes/{quiz_id}/questions",
        json={
            "question_type": "matching",
            "question_text": "Match term to definition",
            "options": ["CPU", "RAM"],
            "correct_answer": {
                "CPU": "Processor",
                "RAM": "Temporary memory"
            },
            "points": 2,
        },
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    question_id = response.json()["id"]
    assert response.json()["correct_answer"] == {
        "CPU": "Processor",
        "RAM": "Temporary memory"
    }

    response = client_with_quizzes_db.post(
        f"/api/v1/quizzes/{quiz_id}/attempts",
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    attempt_id = response.json()["id"]

    response = client_with_quizzes_db.put(
        f"/api/v1/quizzes/attempts/{attempt_id}",
        json={
            "answers": [{
                "question_id": question_id,
                "user_answer": {"cpu": "processor", "ram": "temporary memory"}
            }]
        },
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["score"] == 2
    assert payload["answers"][0]["is_correct"] is True


def test_quiz_hint_penalty_scoring_flow(client_with_quizzes_db: TestClient):
    response = client_with_quizzes_db.post(
        "/api/v1/quizzes",
        json={"name": "Hint Penalty Quiz"},
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    quiz_id = response.json()["id"]

    response = client_with_quizzes_db.post(
        f"/api/v1/quizzes/{quiz_id}/questions",
        json={
            "question_type": "multiple_choice",
            "question_text": "Capital of France?",
            "options": ["Berlin", "Paris", "Rome"],
            "correct_answer": 1,
            "hint": "Think Eiffel Tower.",
            "hint_penalty_points": 2,
            "points": 5,
        },
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    question_id = response.json()["id"]
    assert response.json()["hint"] == "Think Eiffel Tower."
    assert response.json()["hint_penalty_points"] == 2

    response = client_with_quizzes_db.post(
        f"/api/v1/quizzes/{quiz_id}/attempts",
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    attempt_id = response.json()["id"]

    response = client_with_quizzes_db.put(
        f"/api/v1/quizzes/attempts/{attempt_id}",
        json={
            "answers": [{
                "question_id": question_id,
                "user_answer": 1,
                "hint_used": True
            }]
        },
        headers=AUTH_HEADERS,
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["score"] == 3
    assert payload["answers"][0]["is_correct"] is True
    assert payload["answers"][0]["points_awarded"] == 3
    assert payload["answers"][0]["hint_used"] is True
    assert payload["answers"][0]["hint_penalty_points"] == 2


def test_generate_quiz_from_quiz_attempt_question_sources(
    client_with_quizzes_db: TestClient,
    quizzes_db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
):
    attempt_id, question_ids = _create_attempt_with_missed_questions(quizzes_db)
    question_source_id = f"{attempt_id}:{question_ids[0]}"
    quizzes_db.upsert_workspace("ws-1", "Workspace One")

    async def fake_generate_llm(*, prompt: str, model: str | None = None):
        assert "Which structure filters blood?" in prompt
        return {
            "questions": [
                {
                    "question_type": "multiple_choice",
                    "question_text": "Which renal structure performs initial blood filtration?",
                    "options": ["Collecting duct", "Glomerulus", "Ureter", "Renal pelvis"],
                    "correct_answer": 1,
                    "explanation": "The glomerulus is the filtration tuft within the nephron.",
                    "source_citations": [
                        {
                            "source_type": "quiz_attempt_question",
                            "source_id": question_source_id,
                            "quote": "Glomeruli filter blood.",
                        }
                    ],
                    "points": 1,
                }
            ]
        }

    monkeypatch.setattr(quiz_generator, "_call_quiz_generation_llm", fake_generate_llm)

    response = client_with_quizzes_db.post(
        "/api/v1/quizzes/generate",
        json={
            "num_questions": 1,
            "sources": [{"source_type": "quiz_attempt_question", "source_id": question_source_id}],
            "workspace_id": "ws-1",
        },
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["quiz"]["name"] == "Quiz: Remediation"
    assert payload["quiz"]["description"] == "Auto-generated remediation quiz from missed questions"
    assert payload["quiz"]["workspace_id"] == "ws-1"
    assert payload["quiz"]["source_bundle_json"] == [
        {"source_type": "quiz_attempt_question", "source_id": question_source_id}
    ]
    citation = payload["questions"][0]["source_citations"][0]
    assert citation["source_type"] == "quiz_attempt_question"
    assert citation["source_id"] == question_source_id
    assert citation["label"] == "Source 1"
    assert citation["quote"] == "Glomeruli filter blood."


def test_generate_quiz_from_multiple_missed_questions_in_one_attempt(
    client_with_quizzes_db: TestClient,
    quizzes_db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
):
    attempt_id, question_ids = _create_attempt_with_missed_questions(quizzes_db)
    source_ids = [f"{attempt_id}:{question_id}" for question_id in question_ids]

    async def fake_generate_llm(*, prompt: str, model: str | None = None):
        assert "Which structure filters blood?" in prompt
        assert "Which structure concentrates urine?" in prompt
        return {
            "questions": [
                {
                    "question_type": "multiple_choice",
                    "question_text": "Which structure creates the medullary gradient?",
                    "options": ["Loop of Henle", "Glomerulus", "Bowman's capsule", "Podocyte"],
                    "correct_answer": 0,
                    "explanation": "The Loop of Henle establishes the osmotic gradient.",
                    "source_citations": [
                        {"source_type": "quiz_attempt_question", "source_id": source_ids[1], "quote": "Loop of Henle"}
                    ],
                    "points": 1,
                },
                {
                    "question_type": "multiple_choice",
                    "question_text": "Which structure filters blood before tubule processing?",
                    "options": ["Glomerulus", "Collecting duct", "Ureter", "Loop of Henle"],
                    "correct_answer": 0,
                    "explanation": "The glomerulus filters plasma into Bowman's space.",
                    "source_citations": [
                        {"source_type": "quiz_attempt_question", "source_id": source_ids[0], "quote": "Glomeruli filter blood."}
                    ],
                    "points": 1,
                },
            ]
        }

    monkeypatch.setattr(quiz_generator, "_call_quiz_generation_llm", fake_generate_llm)

    response = client_with_quizzes_db.post(
        "/api/v1/quizzes/generate",
        json={
            "num_questions": 2,
            "sources": [{"source_type": "quiz_attempt_question", "source_id": source_id} for source_id in source_ids],
        },
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["quiz"]["source_bundle_json"] == [
        {"source_type": "quiz_attempt_question", "source_id": source_ids[0]},
        {"source_type": "quiz_attempt_question", "source_id": source_ids[1]},
    ]
    citations = [question["source_citations"][0]["source_id"] for question in payload["questions"]]
    assert citations == [source_ids[1], source_ids[0]]


def test_generate_quiz_rejects_invalid_quiz_attempt_question_source_identifier(
    client_with_quizzes_db: TestClient,
):
    response = client_with_quizzes_db.post(
        "/api/v1/quizzes/generate",
        json={
            "num_questions": 1,
            "sources": [{"source_type": "quiz_attempt_question", "source_id": "not-an-attempt"}],
        },
        headers=AUTH_HEADERS,
    )

    assert response.status_code == 400
    assert "quiz_attempt_question source_id" in response.json()["detail"]


def test_quiz_json_import_roundtrip(client_with_quizzes_db: TestClient):
    workspace_response = client_with_quizzes_db.put(
        "/api/v1/workspaces/ws-1",
        json={"name": "Workspace One", "study_materials_policy": "workspace"},
        headers=AUTH_HEADERS,
    )
    assert workspace_response.status_code == 200

    source_quiz_response = client_with_quizzes_db.post(
        "/api/v1/quizzes",
        json={
            "name": "Roundtrip Quiz",
            "description": "roundtrip source",
            "workspace_id": "ws-1",
            "workspace_tag": "workspace:science",
            "media_id": 99,
            "time_limit_seconds": 900,
            "passing_score": 80,
        },
        headers=AUTH_HEADERS,
    )
    assert source_quiz_response.status_code == 200
    source_quiz = source_quiz_response.json()
    source_quiz_id = source_quiz["id"]

    source_questions_payload = [
        {
            "question_type": "multiple_choice",
            "question_text": "What is 2 + 2?",
            "options": ["3", "4", "5"],
            "correct_answer": 1,
            "hint": "Think of pairs.",
            "hint_penalty_points": 1,
            "source_citations": [
                {
                    "label": "Arithmetic primer",
                    "quote": "Two plus two equals four.",
                    "media_id": 99,
                    "chunk_id": "math-1",
                    "timestamp_seconds": 12.2,
                }
            ],
            "points": 1,
            "order_index": 0,
            "tags": ["math", "easy"],
        },
        {
            "question_type": "matching",
            "question_text": "Match hardware to role",
            "options": ["CPU", "RAM"],
            "correct_answer": {"CPU": "Processor", "RAM": "Memory"},
            "hint": "CPU computes, RAM stores temporary data.",
            "hint_penalty_points": 1,
            "source_citations": [
                {
                    "label": "Hardware glossary",
                    "quote": "RAM stores short-term data used by the CPU.",
                    "media_id": 99,
                    "chunk_id": "hardware-3",
                }
            ],
            "points": 2,
            "order_index": 1,
            "tags": ["hardware"],
        },
    ]

    for question in source_questions_payload:
        create_question_response = client_with_quizzes_db.post(
            f"/api/v1/quizzes/{source_quiz_id}/questions",
            json=question,
            headers=AUTH_HEADERS,
        )
        assert create_question_response.status_code == 200

    source_questions_response = client_with_quizzes_db.get(
        f"/api/v1/quizzes/{source_quiz_id}/questions",
        params={"include_answers": True},
        headers=AUTH_HEADERS,
    )
    assert source_questions_response.status_code == 200
    source_questions = source_questions_response.json()["items"]

    import_payload = {
        "export_format": "tldw.quiz.export.v1",
        "quizzes": [
            {
                "quiz": source_quiz,
                "questions": source_questions,
            }
        ],
    }

    import_response = client_with_quizzes_db.post(
        "/api/v1/quizzes/import/json",
        json=import_payload,
        headers=AUTH_HEADERS,
    )
    assert import_response.status_code == 200
    import_result = import_response.json()
    assert import_result["imported_quizzes"] == 1
    assert import_result["failed_quizzes"] == 0
    assert import_result["imported_questions"] == 2
    assert import_result["failed_questions"] == 0
    assert import_result["errors"] == []
    assert len(import_result["items"]) == 1

    imported_quiz_id = import_result["items"][0]["quiz_id"]
    assert imported_quiz_id != source_quiz_id

    imported_quiz_response = client_with_quizzes_db.get(
        f"/api/v1/quizzes/{imported_quiz_id}",
        headers=AUTH_HEADERS,
    )
    assert imported_quiz_response.status_code == 200
    imported_quiz = imported_quiz_response.json()
    assert imported_quiz["name"] == source_quiz["name"]
    assert imported_quiz["description"] == source_quiz["description"]
    assert imported_quiz["workspace_id"] == source_quiz["workspace_id"]
    assert imported_quiz["workspace_tag"] == source_quiz["workspace_tag"]
    assert imported_quiz["media_id"] == source_quiz["media_id"]
    assert imported_quiz["time_limit_seconds"] == source_quiz["time_limit_seconds"]
    assert imported_quiz["passing_score"] == source_quiz["passing_score"]

    imported_questions_response = client_with_quizzes_db.get(
        f"/api/v1/quizzes/{imported_quiz_id}/questions",
        params={"include_answers": True},
        headers=AUTH_HEADERS,
    )
    assert imported_questions_response.status_code == 200
    imported_questions = imported_questions_response.json()["items"]
    assert len(imported_questions) == len(source_questions)

    source_by_order = {
        int(question["order_index"]): question
        for question in source_questions
    }
    for imported_question in imported_questions:
        source_question = source_by_order[int(imported_question["order_index"])]
        assert imported_question["question_type"] == source_question["question_type"]
        assert imported_question["question_text"] == source_question["question_text"]
        assert imported_question["options"] == source_question["options"]
        assert imported_question["correct_answer"] == source_question["correct_answer"]
        assert imported_question["hint"] == source_question["hint"]
        assert imported_question["hint_penalty_points"] == source_question["hint_penalty_points"]
        assert imported_question["source_citations"] == source_question["source_citations"]
        assert imported_question["points"] == source_question["points"]
        assert imported_question["tags"] == source_question["tags"]

    imported_attempt_response = client_with_quizzes_db.post(
        f"/api/v1/quizzes/{imported_quiz_id}/attempts",
        headers=AUTH_HEADERS,
    )
    assert imported_attempt_response.status_code == 200
    imported_attempt = imported_attempt_response.json()

    question_ids_by_type = {
        question["question_type"]: question["id"]
        for question in imported_questions
    }
    submit_response = client_with_quizzes_db.put(
        f"/api/v1/quizzes/attempts/{imported_attempt['id']}",
        json={
            "answers": [
                {
                    "question_id": question_ids_by_type["multiple_choice"],
                    "user_answer": 1,
                },
                {
                    "question_id": question_ids_by_type["matching"],
                    "user_answer": {"cpu": "processor", "ram": "memory"},
                },
            ]
        },
        headers=AUTH_HEADERS,
    )
    assert submit_response.status_code == 200
    submit_payload = submit_response.json()
    assert submit_payload["score"] == 3
    assert all(answer["is_correct"] for answer in submit_payload["answers"])
    answer_by_question_id = {
        int(answer["question_id"]): answer
        for answer in submit_payload["answers"]
    }
    source_by_type = {
        question["question_type"]: question["source_citations"]
        for question in imported_questions
    }
    assert answer_by_question_id[question_ids_by_type["multiple_choice"]]["source_citations"] == source_by_type["multiple_choice"]
    assert answer_by_question_id[question_ids_by_type["matching"]]["source_citations"] == source_by_type["matching"]
