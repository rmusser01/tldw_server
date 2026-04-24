from __future__ import annotations

from importlib import import_module
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.Jobs.manager import JobManager


pytestmark = pytest.mark.integration


def _load_modules():
    try:
        endpoints_mod = import_module("tldw_Server_API.app.api.v1.endpoints.study_suggestions")
        quizzes_mod = import_module("tldw_Server_API.app.api.v1.endpoints.quizzes")
        flashcards_mod = import_module("tldw_Server_API.app.api.v1.endpoints.flashcards")
        snapshot_service_mod = import_module("tldw_Server_API.app.core.StudySuggestions.snapshot_service")
        jobs_mod = import_module("tldw_Server_API.app.core.StudySuggestions.jobs")
        actions_mod = import_module("tldw_Server_API.app.core.StudySuggestions.actions")
    except ModuleNotFoundError as exc:
        pytest.fail(f"Study suggestions endpoint modules are missing: {exc}")
    except ImportError as exc:
        pytest.fail(f"Study suggestions endpoint imports are not yet usable: {exc}")
    return endpoints_mod, quizzes_mod, flashcards_mod, snapshot_service_mod, jobs_mod, actions_mod


@pytest.fixture
def db(tmp_path):
    chacha = CharactersRAGDB(str(tmp_path / "study-suggestions-endpoints.db"), client_id="study-suggestions-endpoints-tests")
    try:
        yield chacha
    finally:
        chacha.close_connection()


@pytest.fixture
def jobs_db_path(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "study-suggestions-jobs.db"
    monkeypatch.setenv("JOBS_DB_PATH", str(path))
    return path


@pytest.fixture
def client(db: CharactersRAGDB, jobs_db_path: Path):
    endpoints_mod, quizzes_mod, flashcards_mod, _snapshot_service_mod, _jobs_mod, _actions_mod = _load_modules()
    app = FastAPI()
    app.include_router(endpoints_mod.router, prefix="/api/v1")
    app.include_router(quizzes_mod.router, prefix="/api/v1")
    app.include_router(flashcards_mod.router, prefix="/api/v1")
    def override_chacha_db():
        return db

    def override_media_db():
        return object()

    app.dependency_overrides[get_chacha_db_for_user] = override_chacha_db
    app.dependency_overrides[get_media_db_for_user] = override_media_db

    async def _override_user() -> User:
        return User(id=1, username="tester", email="t@example.com", is_active=True, roles=["admin"], is_admin=True)

    async def _override_principal() -> AuthPrincipal:
        return AuthPrincipal(kind="user", user_id=1, roles=["admin"], permissions=["*"], is_admin=True)

    app.dependency_overrides[get_request_user] = _override_user
    app.dependency_overrides[get_auth_principal] = _override_principal
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.clear()


def _jobs_manager(jobs_db_path: Path) -> JobManager:
    return JobManager(db_path=jobs_db_path)


def _create_snapshot(
    db: CharactersRAGDB,
    *,
    anchor_type: str = "quiz_attempt",
    anchor_id: int = 101,
    refreshed_from_snapshot_id: int | None = None,
) -> int:
    return db.create_suggestion_snapshot(
        service="quiz",
        activity_type="quiz_attempt",
        anchor_type=anchor_type,
        anchor_id=anchor_id,
        suggestion_type="study_suggestions",
        payload_json={
            "summary": {"score": 7},
            "topics": [
                {
                    "display_label": "Renal basics",
                    "source_type": "note",
                    "source_id": "note-7",
                    "selected": True,
                }
            ],
        },
        user_selection_json={"selected_topic_ids": ["topic-1"]},
        refreshed_from_snapshot_id=refreshed_from_snapshot_id,
    )


def _create_quiz_attempt(db: CharactersRAGDB) -> int:
    quiz_id = db.create_quiz(
        name="Renal Quiz",
        source_bundle_json=[{"source_type": "note", "source_id": "note-7", "label": "Renal basics"}],
    )
    db.create_question(
        quiz_id=quiz_id,
        question_type="multiple_choice",
        question_text="Which organ filters blood?",
        options=["Heart", "Kidney", "Lung"],
        correct_answer=1,
        explanation="The kidney filters blood.",
        tags=["Renal basics"],
        source_citations=[{"source_type": "note", "source_id": "note-7", "label": "Renal basics"}],
    )
    attempt = db.start_attempt(quiz_id)
    return int(attempt["id"])


def test_anchor_status_returns_none_pending_ready_and_failed_states(
    client: TestClient,
    db: CharactersRAGDB,
    jobs_db_path: Path,
):
    _endpoints_mod, _quizzes_mod, _flashcards_mod, _snapshot_service_mod, jobs_mod, _actions_mod = _load_modules()
    pending_payload = jobs_mod.build_study_suggestions_job_payload(
        job_type=jobs_mod.STUDY_SUGGESTIONS_REFRESH_JOB_TYPE,
        anchor_type="quiz_attempt",
        anchor_id=101,
    )
    failed_payload = jobs_mod.build_study_suggestions_job_payload(
        job_type=jobs_mod.STUDY_SUGGESTIONS_REFRESH_JOB_TYPE,
        anchor_type="quiz_attempt",
        anchor_id=202,
    )
    jm = _jobs_manager(jobs_db_path)
    pending_job = jm.create_job(
        domain=jobs_mod.STUDY_SUGGESTIONS_DOMAIN,
        queue=jobs_mod.study_suggestions_jobs_queue(),
        job_type=jobs_mod.STUDY_SUGGESTIONS_REFRESH_JOB_TYPE,
        payload=pending_payload,
        owner_user_id="1",
        priority=5,
        max_retries=1,
    )
    failed_job = jm.create_job(
        domain=jobs_mod.STUDY_SUGGESTIONS_DOMAIN,
        queue=jobs_mod.study_suggestions_jobs_queue(),
        job_type=jobs_mod.STUDY_SUGGESTIONS_REFRESH_JOB_TYPE,
        payload=failed_payload,
        owner_user_id="1",
        priority=5,
        max_retries=1,
    )
    acquired = jm.acquire_next_job(
        domain=jobs_mod.STUDY_SUGGESTIONS_DOMAIN,
        queue=jobs_mod.study_suggestions_jobs_queue(),
        lease_seconds=30,
        worker_id="study-suggestions-test",
    )
    assert acquired is not None  # nosec B101
    if int(acquired["id"]) != int(failed_job["id"]):
        acquired = jm.acquire_next_job(
            domain=jobs_mod.STUDY_SUGGESTIONS_DOMAIN,
            queue=jobs_mod.study_suggestions_jobs_queue(),
            lease_seconds=30,
            worker_id="study-suggestions-test-2",
        )
    assert acquired is not None  # nosec B101
    assert int(acquired["id"]) == int(failed_job["id"])  # nosec B101
    jm.fail_job(
        int(failed_job["id"]),
        error="refresh failed",
        retryable=False,
        worker_id=str(acquired["worker_id"]),
        lease_id=str(acquired["lease_id"]),
    )

    ready_snapshot_id = _create_snapshot(db, anchor_id=303)

    none_response = client.get("/api/v1/study-suggestions/anchors/quiz_attempt/404/status")
    pending_response = client.get("/api/v1/study-suggestions/anchors/quiz_attempt/101/status")
    ready_response = client.get("/api/v1/study-suggestions/anchors/quiz_attempt/303/status")
    failed_response = client.get("/api/v1/study-suggestions/anchors/quiz_attempt/202/status")

    assert none_response.status_code == 200  # nosec B101
    assert none_response.json()["status"] == "none"  # nosec B101

    assert pending_response.status_code == 200  # nosec B101
    assert pending_response.json()["status"] == "pending"  # nosec B101
    assert int(pending_response.json()["job_id"]) == int(pending_job["id"])  # nosec B101

    assert ready_response.status_code == 200  # nosec B101
    assert ready_response.json()["status"] == "ready"  # nosec B101
    assert int(ready_response.json()["snapshot_id"]) == ready_snapshot_id  # nosec B101

    assert failed_response.status_code == 200  # nosec B101
    assert failed_response.json()["status"] == "failed"  # nosec B101
    assert int(failed_response.json()["job_id"]) == int(failed_job["id"])  # nosec B101


def test_snapshot_read_returns_frozen_payload_plus_live_evidence(
    client: TestClient,
    db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
):
    _endpoints_mod, _quizzes_mod, _flashcards_mod, snapshot_service_mod, _jobs_mod, _actions_mod = _load_modules()
    snapshot_id = _create_snapshot(db)

    def fake_live_evidence(snapshot_row, *, note_db, principal):
        assert int(snapshot_row["id"]) == snapshot_id  # nosec B101
        return {
            "topic-1": {
                "source_available": True,
                "source_type": "note",
                "source_id": "note-7",
                "excerpt_text": "Live note excerpt",
            }
        }

    monkeypatch.setattr(snapshot_service_mod, "load_live_evidence_for_snapshot", fake_live_evidence)

    response = client.get(f"/api/v1/study-suggestions/snapshots/{snapshot_id}")

    assert response.status_code == 200  # nosec B101
    body = response.json()
    assert body["snapshot"]["id"] == snapshot_id  # nosec B101
    assert body["snapshot"]["payload"]["topics"][0]["display_label"] == "Renal basics"  # nosec B101
    assert body["live_evidence"]["topic-1"]["source_available"] is True  # nosec B101
    assert body["live_evidence"]["topic-1"]["excerpt_text"] == "Live note excerpt"  # nosec B101


def test_refresh_enqueues_job_for_existing_snapshot(
    client: TestClient,
    db: CharactersRAGDB,
    jobs_db_path: Path,
):
    _endpoints_mod, _quizzes_mod, _flashcards_mod, _snapshot_service_mod, jobs_mod, _actions_mod = _load_modules()
    snapshot_id = _create_snapshot(db)

    refresh = client.post(f"/api/v1/study-suggestions/snapshots/{snapshot_id}/refresh", json={})

    assert refresh.status_code == 202  # nosec B101
    assert refresh.json()["job"]["status"] == "queued"  # nosec B101
    job = _jobs_manager(jobs_db_path).get_job(int(refresh.json()["job"]["id"]))
    assert job is not None  # nosec B101
    assert job["domain"] == jobs_mod.STUDY_SUGGESTIONS_DOMAIN  # nosec B101
    assert job["job_type"] == jobs_mod.STUDY_SUGGESTIONS_REFRESH_JOB_TYPE  # nosec B101
    assert int(job["payload"]["snapshot_id"]) == snapshot_id  # nosec B101
    assert int(job["payload"]["anchor_id"]) == 101  # nosec B101


def test_refresh_maps_job_manager_validation_errors_to_http_400(
    client: TestClient,
    db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
):
    snapshot_id = _create_snapshot(db)

    def fail_create_job(self, *args, **kwargs):  # noqa: ANN001, ARG001
        raise ValueError("invalid refresh request")

    monkeypatch.setattr(JobManager, "create_job", fail_create_job)

    refresh = client.post(f"/api/v1/study-suggestions/snapshots/{snapshot_id}/refresh", json={})

    assert refresh.status_code == 400  # nosec B101
    assert refresh.json()["detail"] == "invalid refresh request"  # nosec B101


def test_live_evidence_permission_failures_degrade_to_source_unavailable(
    client: TestClient,
    db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
):
    _endpoints_mod, _quizzes_mod, _flashcards_mod, snapshot_service_mod, _jobs_mod, _actions_mod = _load_modules()
    snapshot_id = _create_snapshot(db)

    def fail_live_evidence(snapshot_row, *, note_db, principal):
        raise PermissionError("forbidden")

    monkeypatch.setattr(snapshot_service_mod, "load_live_evidence_for_snapshot", fail_live_evidence)

    response = client.get(f"/api/v1/study-suggestions/snapshots/{snapshot_id}")

    assert response.status_code == 200  # nosec B101
    body = response.json()
    assert body["snapshot"]["id"] == snapshot_id  # nosec B101
    assert body["live_evidence"]["source_available"] is False  # nosec B101
    assert body["live_evidence"]["reason"] == "unavailable"  # nosec B101


def test_submit_attempt_enqueues_study_suggestions_refresh_job(
    client: TestClient,
    db: CharactersRAGDB,
    jobs_db_path: Path,
):
    _endpoints_mod, _quizzes_mod, _flashcards_mod, _snapshot_service_mod, jobs_mod, _actions_mod = _load_modules()
    attempt_id = _create_quiz_attempt(db)

    response = client.put(
        f"/api/v1/quizzes/attempts/{attempt_id}",
        json={"answers": [{"question_id": 1, "user_answer": 1, "time_spent_ms": 800}]},
    )

    assert response.status_code == 200  # nosec B101
    job = _jobs_manager(jobs_db_path).list_jobs(
        domain=jobs_mod.STUDY_SUGGESTIONS_DOMAIN,
        queue=jobs_mod.study_suggestions_jobs_queue(),
        owner_user_id="1",
        job_type=jobs_mod.STUDY_SUGGESTIONS_REFRESH_JOB_TYPE,
        limit=10,
    )[0]
    assert job["job_type"] == jobs_mod.STUDY_SUGGESTIONS_REFRESH_JOB_TYPE  # nosec B101
    assert job["payload"]["anchor_type"] == "quiz_attempt"  # nosec B101
    assert int(job["payload"]["anchor_id"]) == attempt_id  # nosec B101


def test_review_session_end_completes_session_and_enqueues_suggestions(
    client: TestClient,
    db: CharactersRAGDB,
    jobs_db_path: Path,
):
    _endpoints_mod, _quizzes_mod, _flashcards_mod, _snapshot_service_mod, jobs_mod, _actions_mod = _load_modules()
    deck_id = db.add_deck("Session Deck", "desc")
    session = db.get_or_create_flashcard_review_session(
        deck_id=deck_id,
        review_mode="due",
        tag_filter=None,
        scope_key=f"due:deck:{deck_id}",
    )

    response = client.post(
        "/api/v1/flashcards/review-sessions/end",
        json={"review_session_id": int(session["id"])},
    )

    assert response.status_code == 200  # nosec B101
    assert response.json()["status"] == "completed"  # nosec B101
    assert int(response.json()["id"]) == int(session["id"])  # nosec B101
    job = _jobs_manager(jobs_db_path).list_jobs(
        domain=jobs_mod.STUDY_SUGGESTIONS_DOMAIN,
        queue=jobs_mod.study_suggestions_jobs_queue(),
        owner_user_id="1",
        job_type=jobs_mod.STUDY_SUGGESTIONS_REFRESH_JOB_TYPE,
        limit=10,
    )[0]
    assert job["payload"]["anchor_type"] == "flashcard_review_session"  # nosec B101
    assert int(job["payload"]["anchor_id"]) == int(session["id"])  # nosec B101


def test_review_sessions_list_route_returns_db_sessions_with_filters(
    client: TestClient,
    db: CharactersRAGDB,
):
    deck_id = db.add_deck("Review Route Deck", "desc")
    active_session = db.get_or_create_flashcard_review_session(
        deck_id=deck_id,
        review_mode="due",
        tag_filter=None,
        scope_key=f"due:deck:{deck_id}",
    )
    completed_session = db.get_or_create_flashcard_review_session(
        deck_id=deck_id,
        review_mode="due",
        tag_filter="renal",
        scope_key=f"due:deck:{deck_id}:tag:renal",
    )
    db.mark_flashcard_review_session_completed(int(completed_session["id"]))

    response = client.get(
        "/api/v1/flashcards/review-sessions",
        params={"deck_id": deck_id, "status": "completed", "limit": 5},
    )

    assert response.status_code == 200  # nosec B101
    body = response.json()
    assert isinstance(body, list)  # nosec B101
    assert len(body) == 1  # nosec B101
    assert int(body[0]["id"]) == int(completed_session["id"])  # nosec B101
    assert body[0]["status"] == "completed"  # nosec B101
    assert body[0]["tag_filter"] == "renal"  # nosec B101
    assert int(active_session["id"]) != int(body[0]["id"])  # nosec B101


def test_snapshot_actions_open_existing_generation_link_when_fingerprint_matches(
    client: TestClient,
    db: CharactersRAGDB,
):
    _endpoints_mod, _quizzes_mod, _flashcards_mod, _snapshot_service_mod, _jobs_mod, actions_mod = _load_modules()
    ancestor_quiz_id = db.create_quiz(name="Ancestor Quiz", source_bundle_json=[])
    unrelated_quiz_id = db.create_quiz(name="Unrelated Quiz", source_bundle_json=[])
    snapshot_id = db.create_suggestion_snapshot(
        service="quiz",
        activity_type="quiz_attempt",
        anchor_type="quiz_attempt",
        anchor_id=101,
        suggestion_type="study_suggestions",
        payload_json={
            "topics": [
                {
                    "id": "topic-1",
                    "topic_key": "renal:kidney-functions",
                    "canonical_label": "kidney functions",
                    "display_label": " Renal Basics ",
                    "selected": True,
                },
                {
                    "id": "topic-2",
                    "topic_key": "renal:acid-base",
                    "canonical_label": "acid base",
                    "display_label": "Acid Base",
                    "selected": True,
                },
            ]
        },
    )
    fingerprint = actions_mod.build_selection_fingerprint(
        snapshot_id=snapshot_id,
        target_service="quiz",
        target_type="quiz",
        selected_topics=["renal:acid-base", "renal:kidney-functions"],
        action_kind="follow_up_quiz",
        generator_version="v1",
    )
    db.create_suggestion_generation_link(
        snapshot_id=snapshot_id,
        target_service="quiz",
        target_type="quiz",
        target_id=str(ancestor_quiz_id),
        selection_fingerprint=fingerprint,
    )
    unrelated_snapshot_id = db.create_suggestion_snapshot(
        service="quiz",
        activity_type="quiz_attempt",
        anchor_type="quiz_attempt",
        anchor_id=101,
        suggestion_type="study_suggestions",
        payload_json={
            "topics": [
                {
                    "id": "topic-1",
                    "topic_key": "renal:kidney-functions",
                    "canonical_label": "kidney functions",
                    "display_label": " Renal Basics ",
                    "selected": True,
                },
                {
                    "id": "topic-2",
                    "topic_key": "renal:acid-base",
                    "canonical_label": "acid base",
                    "display_label": "Acid Base",
                    "selected": True,
                },
            ]
        },
    )
    db.create_suggestion_generation_link(
        snapshot_id=unrelated_snapshot_id,
        target_service="quiz",
        target_type="quiz",
        target_id=str(unrelated_quiz_id),
        selection_fingerprint=fingerprint,
    )

    response = client.post(
        f"/api/v1/study-suggestions/snapshots/{snapshot_id}/actions",
        json={
            "target_service": "quiz",
            "target_type": "quiz",
            "action_kind": "follow_up_quiz",
            "selected_topic_ids": ["topic-1", "topic-2"],
            "generator_version": "v1",
            "force_regenerate": False,
        },
    )

    assert response.status_code == 200  # nosec B101
    assert response.json()["disposition"] == "opened_existing"  # nosec B101
    assert response.json()["target_id"] == str(ancestor_quiz_id)  # nosec B101
    assert response.json()["selection_fingerprint"] == fingerprint  # nosec B101


def test_snapshot_actions_use_selected_topic_edits_and_manual_labels_for_generation_and_fingerprint(
    client: TestClient,
    db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
):
    endpoints_mod, _quizzes_mod, _flashcards_mod, _snapshot_service_mod, _jobs_mod, actions_mod = _load_modules()
    snapshot_id = db.create_suggestion_snapshot(
        service="quiz",
        activity_type="quiz_attempt",
        anchor_type="quiz_attempt",
        anchor_id=101,
        suggestion_type="study_suggestions",
        payload_json={
            "topics": [
                {
                    "id": "topic-1",
                    "topic_key": "renal:basics",
                    "canonical_label": "renal basics",
                    "display_label": "Renal Basics",
                    "selected": True,
                },
                {
                    "id": "topic-2",
                    "topic_key": "renal:acid-base",
                    "canonical_label": "acid base",
                    "display_label": "Acid Base",
                    "selected": True,
                },
            ]
        },
    )

    expected_topics = ["acid base remix", "electrolyte ladder"]
    expected_fingerprint = actions_mod.build_selection_fingerprint(
        snapshot_id=snapshot_id,
        target_service="quiz",
        target_type="quiz",
        selected_topics=["electrolyte ladder", "renal:acid-base"],
        action_kind="follow_up_quiz",
        generator_version="v2",
    )

    async def fake_dispatch_action(*, note_db, snapshot_row, request_body, media_db=None):
        assert int(snapshot_row["id"]) == snapshot_id  # nosec B101
        assert request_body["selected_topics"] == expected_topics  # nosec B101
        assert request_body["selected_topic_edits"] == [{"id": "topic-2", "label": " Acid Base Remix "}]  # nosec B101
        assert request_body["manual_topic_labels"] == [" Electrolyte Ladder "]  # nosec B101
        return {
            "target_service": "quiz",
            "target_type": "quiz",
            "target_id": "quiz-77",
        }

    monkeypatch.setattr(endpoints_mod, "_dispatch_follow_up_action", fake_dispatch_action)

    response = client.post(
        f"/api/v1/study-suggestions/snapshots/{snapshot_id}/actions",
        json={
            "target_service": "quiz",
            "target_type": "quiz",
            "action_kind": "follow_up_quiz",
            "selected_topic_ids": ["topic-2"],
            "selected_topic_edits": [{"id": "topic-2", "label": " Acid Base Remix "}],
            "manual_topic_labels": [" Electrolyte Ladder "],
            "has_explicit_selection": True,
            "generator_version": "v2",
            "force_regenerate": True,
        },
    )

    assert response.status_code == 200  # nosec B101
    assert response.json()["disposition"] == "generated"  # nosec B101
    assert response.json()["target_id"] == "quiz-77"  # nosec B101
    assert response.json()["selection_fingerprint"] == expected_fingerprint  # nosec B101


def test_snapshot_actions_respect_explicit_empty_selection_without_falling_back_to_defaults(
    client: TestClient,
    db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
):
    endpoints_mod, _quizzes_mod, _flashcards_mod, _snapshot_service_mod, _jobs_mod, actions_mod = _load_modules()
    snapshot_id = db.create_suggestion_snapshot(
        service="quiz",
        activity_type="quiz_attempt",
        anchor_type="quiz_attempt",
        anchor_id=101,
        suggestion_type="study_suggestions",
        payload_json={
            "topics": [
                {"id": "topic-1", "display_label": "Renal Basics", "selected": True},
                {"id": "topic-2", "display_label": "Acid Base", "selected": True},
            ]
        },
    )

    expected_fingerprint = actions_mod.build_selection_fingerprint(
        snapshot_id=snapshot_id,
        target_service="quiz",
        target_type="quiz",
        selected_topics=[],
        action_kind="follow_up_quiz",
        generator_version="v1",
    )

    async def fake_dispatch_action(*, note_db, snapshot_row, request_body, media_db=None):
        assert int(snapshot_row["id"]) == snapshot_id  # nosec B101
        assert request_body["selected_topics"] == []  # nosec B101
        return {
            "target_service": "quiz",
            "target_type": "quiz",
            "target_id": "quiz-empty",
        }

    monkeypatch.setattr(endpoints_mod, "_dispatch_follow_up_action", fake_dispatch_action)

    response = client.post(
        f"/api/v1/study-suggestions/snapshots/{snapshot_id}/actions",
        json={
            "target_service": "quiz",
            "target_type": "quiz",
            "action_kind": "follow_up_quiz",
            "selected_topic_ids": [],
            "selected_topic_edits": [],
            "manual_topic_labels": [],
            "has_explicit_selection": True,
            "generator_version": "v1",
            "force_regenerate": True,
        },
    )

    assert response.status_code == 200  # nosec B101
    assert response.json()["target_id"] == "quiz-empty"  # nosec B101
    assert response.json()["selection_fingerprint"] == expected_fingerprint  # nosec B101


def test_snapshot_actions_force_regenerate_bypasses_duplicate_open_behavior(
    client: TestClient,
    db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
):
    endpoints_mod, _quizzes_mod, _flashcards_mod, _snapshot_service_mod, _jobs_mod, actions_mod = _load_modules()
    snapshot_id = db.create_suggestion_snapshot(
        service="quiz",
        activity_type="quiz_attempt",
        anchor_type="quiz_attempt",
        anchor_id=101,
        suggestion_type="study_suggestions",
        payload_json={"topics": [{"id": "topic-1", "display_label": "Renal Basics", "selected": True}]},
    )
    fingerprint = actions_mod.build_selection_fingerprint(
        snapshot_id=snapshot_id,
        target_service="quiz",
        target_type="quiz",
        selected_topics=["renal basics"],
        action_kind="follow_up_quiz",
        generator_version="v1",
    )
    db.create_suggestion_generation_link(
        snapshot_id=snapshot_id,
        target_service="quiz",
        target_type="quiz",
        target_id="quiz-55",
        selection_fingerprint=fingerprint,
    )

    async def fake_dispatch_action(*, note_db, snapshot_row, request_body, media_db=None):
        assert int(snapshot_row["id"]) == snapshot_id  # nosec B101
        assert request_body["force_regenerate"] is True  # nosec B101
        return {
            "target_service": "quiz",
            "target_type": "quiz",
            "target_id": "quiz-99",
        }

    monkeypatch.setattr(endpoints_mod, "_dispatch_follow_up_action", fake_dispatch_action)

    response = client.post(
        f"/api/v1/study-suggestions/snapshots/{snapshot_id}/actions",
        json={
            "target_service": "quiz",
            "target_type": "quiz",
            "action_kind": "follow_up_quiz",
            "selected_topic_ids": ["topic-1"],
            "generator_version": "v1",
            "force_regenerate": True,
        },
    )

    assert response.status_code == 200  # nosec B101
    assert response.json()["disposition"] == "generated"  # nosec B101
    assert response.json()["target_id"] == "quiz-99"  # nosec B101
    created_link = db.find_suggestion_generation_link(
        snapshot_id=snapshot_id,
        target_service="quiz",
        target_type="quiz",
        target_id="quiz-99",
        selection_fingerprint=fingerprint,
    )
    assert created_link is not None  # nosec B101


def test_flashcard_follow_up_force_regenerate_creates_real_deck_and_link(
    client: TestClient,
    db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
):
    endpoints_mod, _quizzes_mod, _flashcards_mod, _snapshot_service_mod, _jobs_mod, actions_mod = _load_modules()
    attempt_id = _create_quiz_attempt(db)
    snapshot_id = db.create_suggestion_snapshot(
        service="quiz",
        activity_type="quiz_attempt",
        anchor_type="quiz_attempt",
        anchor_id=attempt_id,
        suggestion_type="study_suggestions",
        payload_json={"topics": [{"id": "topic-1", "display_label": "Renal Basics", "selected": True}]},
    )

    async def fake_generate_flashcards(payload, context):
        assert "which organ filters blood?" in str(payload["text"]).lower()  # nosec B101
        assert payload["focus_topics"] == ["renal basics"]  # nosec B101
        return {
            "flashcards": [
                {"front": "What does the kidney do?", "back": "Filters blood.", "tags": ["renal"]},
                {"front": "What maintains homeostasis?", "back": "Kidneys help regulate fluids.", "tags": ["renal"]},
            ]
        }

    monkeypatch.setattr(endpoints_mod, "run_flashcard_generate_adapter", fake_generate_flashcards)

    response = client.post(
        f"/api/v1/study-suggestions/snapshots/{snapshot_id}/actions",
        json={
            "target_service": "flashcards",
            "target_type": "deck",
            "action_kind": "follow_up_flashcards",
            "selected_topic_ids": ["topic-1"],
            "generator_version": "v1",
            "force_regenerate": True,
        },
    )

    assert response.status_code == 200  # nosec B101
    assert response.json()["disposition"] == "generated"  # nosec B101
    deck_id = int(response.json()["target_id"])
    deck = db.get_deck(deck_id)
    assert deck is not None  # nosec B101
    cards = db.list_flashcards(deck_id=deck_id)
    assert len(cards) == 2  # nosec B101

    fingerprint = actions_mod.build_selection_fingerprint(
        snapshot_id=snapshot_id,
        target_service="flashcards",
        target_type="deck",
        selected_topics=["renal basics"],
        action_kind="follow_up_flashcards",
        generator_version="v1",
    )
    linked = db.find_suggestion_generation_link(
        snapshot_id=snapshot_id,
        target_service="flashcards",
        target_type="deck",
        target_id=str(deck_id),
        selection_fingerprint=fingerprint,
    )
    assert linked is not None  # nosec B101


def test_flashcard_session_snapshot_can_generate_follow_up_quiz(
    client: TestClient,
    db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
):
    endpoints_mod, _quizzes_mod, _flashcards_mod, _snapshot_service_mod, _jobs_mod, _actions_mod = _load_modules()
    deck_id = db.add_deck("Session Deck", "desc")
    db.add_flashcard(
        {
            "deck_id": deck_id,
            "front": "Kidney",
            "back": "Filters blood",
            "tags_json": '["renal basics"]',
        }
    )
    session = db.get_or_create_flashcard_review_session(
        deck_id=deck_id,
        review_mode="due",
        tag_filter=None,
        scope_key=f"due:deck:{deck_id}",
    )
    db.mark_flashcard_review_session_completed(int(session["id"]))
    snapshot_id = db.create_suggestion_snapshot(
        service="flashcards",
        activity_type="flashcard_review_session",
        anchor_type="flashcard_review_session",
        anchor_id=int(session["id"]),
        suggestion_type="study_suggestions",
        payload_json={"topics": [{"id": "topic-1", "display_label": "Renal Basics", "selected": True}]},
    )

    async def fake_generate_quiz_from_sources(**kwargs):
        assert kwargs["sources"] == [{"source_type": "flashcard_deck", "source_id": str(deck_id)}]  # nosec B101
        assert kwargs["focus_topics"] == ["renal basics"]  # nosec B101
        return {"quiz": {"id": 88}}

    monkeypatch.setattr(endpoints_mod, "generate_quiz_from_sources", fake_generate_quiz_from_sources)

    response = client.post(
        f"/api/v1/study-suggestions/snapshots/{snapshot_id}/actions",
        json={
            "target_service": "quiz",
            "target_type": "quiz",
            "action_kind": "follow_up_quiz",
            "selected_topic_ids": ["topic-1"],
            "generator_version": "v1",
            "force_regenerate": True,
        },
    )

    assert response.status_code == 200  # nosec B101
    assert response.json()["disposition"] == "generated"  # nosec B101
    assert response.json()["target_service"] == "quiz"  # nosec B101
    assert response.json()["target_type"] == "quiz"  # nosec B101
    assert response.json()["target_id"] == "88"  # nosec B101


def test_snapshot_actions_reject_mismatched_action_contract(
    client: TestClient,
    db: CharactersRAGDB,
):
    _endpoints_mod, _quizzes_mod, _flashcards_mod, _snapshot_service_mod, _jobs_mod, _actions_mod = _load_modules()
    snapshot_id = _create_snapshot(db)

    response = client.post(
        f"/api/v1/study-suggestions/snapshots/{snapshot_id}/actions",
        json={
            "target_service": "quiz",
            "target_type": "deck",
            "action_kind": "follow_up_quiz",
            "selected_topic_ids": ["topic-1"],
            "generator_version": "v1",
            "force_regenerate": False,
        },
    )

    assert response.status_code == 400  # nosec B101
    assert "must target quiz/quiz" in response.json()["detail"].lower()  # nosec B101


def test_snapshot_actions_return_conflict_when_matching_generation_is_already_reserved(
    client: TestClient,
    db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
):
    endpoints_mod, _quizzes_mod, _flashcards_mod, _snapshot_service_mod, _jobs_mod, actions_mod = _load_modules()
    attempt_id = _create_quiz_attempt(db)
    snapshot_id = db.create_suggestion_snapshot(
        service="quiz",
        activity_type="quiz_attempt",
        anchor_type="quiz_attempt",
        anchor_id=attempt_id,
        suggestion_type="study_suggestions",
        payload_json={"topics": [{"id": "topic-1", "display_label": "Renal Basics", "selected": True}]},
    )
    fingerprint = actions_mod.build_selection_fingerprint(
        snapshot_id=snapshot_id,
        target_service="quiz",
        target_type="quiz",
        selected_topics=["renal basics"],
        action_kind="follow_up_quiz",
        generator_version="v1",
    )
    db.create_suggestion_generation_link(
        snapshot_id=snapshot_id,
        target_service="quiz",
        target_type="quiz",
        target_id=actions_mod.build_pending_generation_target_id(fingerprint),
        selection_fingerprint=fingerprint,
    )

    async def fail_if_called(*args, **kwargs):
        pytest.fail("Generation should not run when the selection fingerprint is already reserved")

    monkeypatch.setattr(endpoints_mod, "_dispatch_follow_up_action", fail_if_called)

    response = client.post(
        f"/api/v1/study-suggestions/snapshots/{snapshot_id}/actions",
        json={
            "target_service": "quiz",
            "target_type": "quiz",
            "action_kind": "follow_up_quiz",
            "selected_topic_ids": ["topic-1"],
            "generator_version": "v1",
            "force_regenerate": False,
        },
    )

    assert response.status_code == 409  # nosec B101
    assert "already in progress" in response.json()["detail"].lower()  # nosec B101


def test_snapshot_actions_use_snapshot_semantics_for_fingerprint_and_edit_labels_for_generation(
    client: TestClient,
    db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
):
    endpoints_mod, _quizzes_mod, _flashcards_mod, _snapshot_service_mod, _jobs_mod, _actions_mod = _load_modules()
    snapshot_id = db.create_suggestion_snapshot(
        service="quiz",
        activity_type="quiz_attempt",
        anchor_type="quiz_attempt",
        anchor_id=101,
        suggestion_type="study_suggestions",
        payload_json={
            "topics": [
                {
                    "id": "topic-1",
                    "topic_key": "renal:kidney-functions",
                    "normalization_version": "norm-v2",
                    "canonical_label": "kidney functions",
                    "display_label": "Kidney Functions",
                    "selected": True,
                }
            ]
        },
    )

    expected_fingerprint = "semantic-fingerprint"

    def fake_build_selection_fingerprint(
        *,
        snapshot_id: int,
        target_service: str,
        target_type: str,
        selected_topics,
        action_kind: str,
        generator_version: str | None = None,
        normalization_version: str | None = None,
        include_normalization_version: bool = True,
    ) -> str:
        assert snapshot_id == snapshot_id_value  # nosec B101
        assert target_service == "quiz"  # nosec B101
        assert target_type == "quiz"  # nosec B101
        assert list(selected_topics) == ["renal:kidney-functions"]  # nosec B101
        assert action_kind == "follow_up_quiz"  # nosec B101
        assert generator_version == "v2"  # nosec B101
        assert normalization_version == "norm-v2"  # nosec B101
        assert include_normalization_version in {True, False}  # nosec B101
        return expected_fingerprint

    snapshot_id_value = snapshot_id
    monkeypatch.setattr(endpoints_mod, "build_selection_fingerprint", fake_build_selection_fingerprint)

    async def fake_dispatch_action(*, note_db, snapshot_row, request_body, media_db=None):
        assert int(snapshot_row["id"]) == snapshot_id  # nosec B101
        assert request_body["selected_topics"] == ["renal remix"]  # nosec B101
        assert request_body["selected_topic_edits"] == [{"id": "topic-1", "label": "Renal Remix"}]  # nosec B101
        return {
            "target_service": "quiz",
            "target_type": "quiz",
            "target_id": "quiz-77",
        }

    monkeypatch.setattr(endpoints_mod, "_dispatch_follow_up_action", fake_dispatch_action)

    response = client.post(
        f"/api/v1/study-suggestions/snapshots/{snapshot_id}/actions",
        json={
            "target_service": "quiz",
            "target_type": "quiz",
            "action_kind": "follow_up_quiz",
            "selected_topic_ids": ["topic-1"],
            "selected_topic_edits": [{"id": "topic-1", "label": "Renal Remix"}],
            "manual_topic_labels": [],
            "has_explicit_selection": True,
            "generator_version": "v2",
            "force_regenerate": True,
        },
    )

    assert response.status_code == 200  # nosec B101
    assert response.json()["disposition"] == "generated"  # nosec B101
    assert response.json()["selection_fingerprint"] == expected_fingerprint  # nosec B101


def test_snapshot_actions_reopen_existing_links_written_before_normalization_version_fingerprints(
    client: TestClient,
    db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
):
    endpoints_mod, _quizzes_mod, _flashcards_mod, _snapshot_service_mod, _jobs_mod, actions_mod = _load_modules()
    existing_quiz_id = db.create_quiz(name="Existing Quiz", source_bundle_json=[])
    snapshot_id = db.create_suggestion_snapshot(
        service="quiz",
        activity_type="quiz_attempt",
        anchor_type="quiz_attempt",
        anchor_id=101,
        suggestion_type="study_suggestions",
        payload_json={
            "topics": [
                {
                    "id": "topic-1",
                    "topic_key": "renal:kidney-functions",
                    "normalization_version": "norm-v2",
                    "canonical_label": "kidney functions",
                    "display_label": "Kidney Functions",
                    "selected": True,
                }
            ]
        },
    )
    legacy_fingerprint = (
        f"snapshot_id={snapshot_id}|target_service=quiz|target_type=quiz|"
        "topics=renal:kidney-functions|action_kind=follow_up_quiz|generator_version=v1"
    )
    db.create_suggestion_generation_link(
        snapshot_id=snapshot_id,
        target_service="quiz",
        target_type="quiz",
        target_id=str(existing_quiz_id),
        selection_fingerprint=legacy_fingerprint,
    )

    async def fake_dispatch_action(*, note_db, snapshot_row, request_body, media_db=None):
        assert int(snapshot_row["id"]) == snapshot_id  # nosec B101
        return {
            "target_service": "quiz",
            "target_type": "quiz",
            "target_id": "quiz-generated",
        }

    monkeypatch.setattr(endpoints_mod, "_dispatch_follow_up_action", fake_dispatch_action)

    response = client.post(
        f"/api/v1/study-suggestions/snapshots/{snapshot_id}/actions",
        json={
            "target_service": "quiz",
            "target_type": "quiz",
            "action_kind": "follow_up_quiz",
            "selected_topic_ids": ["topic-1"],
            "generator_version": "v1",
            "force_regenerate": False,
        },
    )

    expected_fingerprint = actions_mod.build_selection_fingerprint(
        snapshot_id=snapshot_id,
        target_service="quiz",
        target_type="quiz",
        selected_topics=["renal:kidney-functions"],
        action_kind="follow_up_quiz",
        generator_version="v1",
        normalization_version="norm-v2",
    )

    assert response.status_code == 200  # nosec B101
    assert response.json()["disposition"] == "opened_existing"  # nosec B101
    assert response.json()["target_id"] == str(existing_quiz_id)  # nosec B101
    assert response.json()["selection_fingerprint"] == expected_fingerprint  # nosec B101


def test_snapshot_actions_prefer_live_legacy_links_when_stale_current_fingerprint_rows_exist(
    client: TestClient,
    db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
):
    endpoints_mod, _quizzes_mod, _flashcards_mod, _snapshot_service_mod, _jobs_mod, actions_mod = _load_modules()
    live_deck_id = db.add_deck("Live Deck", "desc")
    stale_deck_id = db.add_deck("Stale Deck", "desc")
    db.soft_delete_deck_by_id(stale_deck_id)
    snapshot_id = db.create_suggestion_snapshot(
        service="flashcards",
        activity_type="flashcard_review_session",
        anchor_type="flashcard_review_session",
        anchor_id=101,
        suggestion_type="study_suggestions",
        payload_json={
            "topics": [
                {
                    "id": "topic-1",
                    "topic_key": "renal:kidney-functions",
                    "normalization_version": "norm-v2",
                    "canonical_label": "kidney functions",
                    "display_label": "Kidney Functions",
                    "selected": True,
                }
            ]
        },
    )
    current_fingerprint = actions_mod.build_selection_fingerprint(
        snapshot_id=snapshot_id,
        target_service="flashcards",
        target_type="deck",
        selected_topics=["renal:kidney-functions"],
        action_kind="follow_up_flashcards",
        generator_version="v1",
        normalization_version="norm-v2",
    )
    legacy_fingerprint = actions_mod.build_selection_fingerprint(
        snapshot_id=snapshot_id,
        target_service="flashcards",
        target_type="deck",
        selected_topics=["renal:kidney-functions"],
        action_kind="follow_up_flashcards",
        generator_version="v1",
        normalization_version="norm-v2",
        include_normalization_version=False,
    )
    db.create_suggestion_generation_link(
        snapshot_id=snapshot_id,
        target_service="flashcards",
        target_type="deck",
        target_id=str(stale_deck_id),
        selection_fingerprint=current_fingerprint,
    )
    db.create_suggestion_generation_link(
        snapshot_id=snapshot_id,
        target_service="flashcards",
        target_type="deck",
        target_id=str(live_deck_id),
        selection_fingerprint=legacy_fingerprint,
    )

    async def fake_dispatch_action(*, note_db, snapshot_row, request_body, media_db=None):
        assert int(snapshot_row["id"]) == snapshot_id  # nosec B101
        return {
            "target_service": "flashcards",
            "target_type": "deck",
            "target_id": "deck-generated",
        }

    monkeypatch.setattr(endpoints_mod, "_dispatch_follow_up_action", fake_dispatch_action)

    response = client.post(
        f"/api/v1/study-suggestions/snapshots/{snapshot_id}/actions",
        json={
            "target_service": "flashcards",
            "target_type": "deck",
            "action_kind": "follow_up_flashcards",
            "selected_topic_ids": ["topic-1"],
            "generator_version": "v1",
            "force_regenerate": False,
        },
    )

    assert response.status_code == 200  # nosec B101
    assert response.json()["disposition"] == "opened_existing"  # nosec B101
    assert response.json()["target_id"] == str(live_deck_id)  # nosec B101
    assert response.json()["selection_fingerprint"] == current_fingerprint  # nosec B101


def test_snapshot_actions_reserve_conflict_rechecks_live_legacy_links_before_opening_existing(
    client: TestClient,
    db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
):
    endpoints_mod, _quizzes_mod, _flashcards_mod, _snapshot_service_mod, _jobs_mod, actions_mod = _load_modules()
    live_deck_id = db.add_deck("Live Deck", "desc")
    stale_deck_id = db.add_deck("Stale Deck", "desc")
    db.soft_delete_deck_by_id(stale_deck_id)
    snapshot_id = db.create_suggestion_snapshot(
        service="flashcards",
        activity_type="flashcard_review_session",
        anchor_type="flashcard_review_session",
        anchor_id=101,
        suggestion_type="study_suggestions",
        payload_json={
            "topics": [
                {
                    "id": "topic-1",
                    "topic_key": "renal:kidney-functions",
                    "normalization_version": "norm-v2",
                    "canonical_label": "kidney functions",
                    "display_label": "Kidney Functions",
                    "selected": True,
                }
            ]
        },
    )
    current_fingerprint = actions_mod.build_selection_fingerprint(
        snapshot_id=snapshot_id,
        target_service="flashcards",
        target_type="deck",
        selected_topics=["renal:kidney-functions"],
        action_kind="follow_up_flashcards",
        generator_version="v1",
        normalization_version="norm-v2",
    )
    legacy_fingerprint = actions_mod.build_selection_fingerprint(
        snapshot_id=snapshot_id,
        target_service="flashcards",
        target_type="deck",
        selected_topics=["renal:kidney-functions"],
        action_kind="follow_up_flashcards",
        generator_version="v1",
        normalization_version="norm-v2",
        include_normalization_version=False,
    )
    db.create_suggestion_generation_link(
        snapshot_id=snapshot_id,
        target_service="flashcards",
        target_type="deck",
        target_id=str(stale_deck_id),
        selection_fingerprint=current_fingerprint,
    )
    db.create_suggestion_generation_link(
        snapshot_id=snapshot_id,
        target_service="flashcards",
        target_type="deck",
        target_id=str(live_deck_id),
        selection_fingerprint=legacy_fingerprint,
    )

    def fake_reserve_generation_link(*args, **kwargs):
        raise RuntimeError("simulated reservation race")

    async def fake_dispatch_action(*, note_db, snapshot_row, request_body, media_db=None):
        assert int(snapshot_row["id"]) == snapshot_id  # nosec B101
        return {
            "target_service": "flashcards",
            "target_type": "deck",
            "target_id": "deck-generated",
        }

    monkeypatch.setattr(endpoints_mod, "reserve_generation_link", fake_reserve_generation_link)
    monkeypatch.setattr(endpoints_mod, "_dispatch_follow_up_action", fake_dispatch_action)

    response = client.post(
        f"/api/v1/study-suggestions/snapshots/{snapshot_id}/actions",
        json={
            "target_service": "flashcards",
            "target_type": "deck",
            "action_kind": "follow_up_flashcards",
            "selected_topic_ids": ["topic-1"],
            "generator_version": "v1",
            "force_regenerate": False,
        },
    )

    assert response.status_code == 200  # nosec B101
    assert response.json()["disposition"] == "opened_existing"  # nosec B101
    assert response.json()["target_id"] == str(live_deck_id)  # nosec B101
    assert response.json()["selection_fingerprint"] == current_fingerprint  # nosec B101


def test_snapshot_actions_reopen_refreshed_ancestor_link_without_writing_child_alias_rows(
    client: TestClient,
    db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
):
    endpoints_mod, _quizzes_mod, _flashcards_mod, _snapshot_service_mod, _jobs_mod, actions_mod = _load_modules()
    ancestor_quiz_id = db.create_quiz(name="Ancestor Quiz", source_bundle_json=[])
    unrelated_quiz_id = db.create_quiz(name="Unrelated Quiz", source_bundle_json=[])
    original_snapshot_id = db.create_suggestion_snapshot(
        service="quiz",
        activity_type="quiz_attempt",
        anchor_type="quiz_attempt",
        anchor_id=101,
        suggestion_type="study_suggestions",
        payload_json={
            "topics": [
                {
                    "id": "topic-1",
                    "topic_key": "renal:kidney-functions",
                    "normalization_version": "norm-v2",
                    "canonical_label": "kidney functions",
                    "display_label": "Kidney Functions",
                    "selected": True,
                }
            ]
        },
    )
    fingerprint = actions_mod.build_selection_fingerprint(
        snapshot_id=original_snapshot_id,
        target_service="quiz",
        target_type="quiz",
        selected_topics=["renal:kidney-functions"],
        action_kind="follow_up_quiz",
        generator_version="v1",
        normalization_version="norm-v2",
    )
    db.create_suggestion_generation_link(
        snapshot_id=original_snapshot_id,
        target_service="quiz",
        target_type="quiz",
        target_id=str(ancestor_quiz_id),
        selection_fingerprint=fingerprint,
    )
    unrelated_snapshot_id = db.create_suggestion_snapshot(
        service="quiz",
        activity_type="quiz_attempt",
        anchor_type="quiz_attempt",
        anchor_id=101,
        suggestion_type="study_suggestions",
        payload_json={
            "topics": [
                {
                    "id": "topic-1",
                    "topic_key": "renal:kidney-functions",
                    "normalization_version": "norm-v2",
                    "canonical_label": "kidney functions",
                    "display_label": "Kidney Functions",
                    "selected": True,
                }
            ]
        },
    )
    db.create_suggestion_generation_link(
        snapshot_id=unrelated_snapshot_id,
        target_service="quiz",
        target_type="quiz",
        target_id=str(unrelated_quiz_id),
        selection_fingerprint=fingerprint,
    )
    child_snapshot_id = db.create_suggestion_snapshot(
        service="quiz",
        activity_type="quiz_attempt",
        anchor_type="quiz_attempt",
        anchor_id=101,
        suggestion_type="study_suggestions",
        payload_json={
            "topics": [
                {
                    "id": "topic-1",
                    "topic_key": "renal:kidney-functions",
                    "normalization_version": "norm-v2",
                    "canonical_label": "kidney functions",
                    "display_label": "Kidney Functions",
                    "selected": True,
                }
            ]
        },
        refreshed_from_snapshot_id=original_snapshot_id,
    )

    async def fake_dispatch_action(*, note_db, snapshot_row, request_body, media_db=None):
        assert int(snapshot_row["id"]) == child_snapshot_id  # nosec B101
        return {
            "target_service": "quiz",
            "target_type": "quiz",
            "target_id": "quiz-generated",
        }

    monkeypatch.setattr(endpoints_mod, "_dispatch_follow_up_action", fake_dispatch_action)

    response = client.post(
        f"/api/v1/study-suggestions/snapshots/{child_snapshot_id}/actions",
        json={
            "target_service": "quiz",
            "target_type": "quiz",
            "action_kind": "follow_up_quiz",
            "selected_topic_ids": ["topic-1"],
            "generator_version": "v1",
            "force_regenerate": False,
        },
    )

    assert response.status_code == 200  # nosec B101
    assert response.json()["disposition"] == "opened_existing"  # nosec B101
    assert response.json()["target_id"] == str(ancestor_quiz_id)  # nosec B101
    expected_child_fingerprint = actions_mod.build_selection_fingerprint(
        snapshot_id=child_snapshot_id,
        target_service="quiz",
        target_type="quiz",
        selected_topics=["renal:kidney-functions"],
        action_kind="follow_up_quiz",
        generator_version="v1",
        normalization_version="norm-v2",
    )
    assert response.json()["selection_fingerprint"] == expected_child_fingerprint  # nosec B101

    child_link = db.execute_query(
        """
        SELECT COUNT(*) AS count
          FROM suggestion_generation_links
         WHERE snapshot_id = ? AND deleted = 0
        """,
        (child_snapshot_id,),
    ).fetchone()
    assert int(child_link["count"]) == 0  # nosec B101


def test_snapshot_actions_reopen_refreshed_ancestor_links_written_before_normalization_version_fingerprints(
    client: TestClient,
    db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
):
    endpoints_mod, _quizzes_mod, _flashcards_mod, _snapshot_service_mod, _jobs_mod, actions_mod = _load_modules()
    ancestor_quiz_id = db.create_quiz(name="Ancestor Quiz", source_bundle_json=[])
    original_snapshot_id = db.create_suggestion_snapshot(
        service="quiz",
        activity_type="quiz_attempt",
        anchor_type="quiz_attempt",
        anchor_id=101,
        suggestion_type="study_suggestions",
        payload_json={
            "topics": [
                {
                    "id": "topic-1",
                    "topic_key": "renal:kidney-functions",
                    "normalization_version": "norm-v2",
                    "canonical_label": "kidney functions",
                    "display_label": "Kidney Functions",
                    "selected": True,
                }
            ]
        },
    )
    legacy_fingerprint = (
        f"snapshot_id={original_snapshot_id}|target_service=quiz|target_type=quiz|"
        "topics=renal:kidney-functions|action_kind=follow_up_quiz|generator_version=v1"
    )
    db.create_suggestion_generation_link(
        snapshot_id=original_snapshot_id,
        target_service="quiz",
        target_type="quiz",
        target_id=str(ancestor_quiz_id),
        selection_fingerprint=legacy_fingerprint,
    )
    child_snapshot_id = db.create_suggestion_snapshot(
        service="quiz",
        activity_type="quiz_attempt",
        anchor_type="quiz_attempt",
        anchor_id=101,
        suggestion_type="study_suggestions",
        payload_json={
            "topics": [
                {
                    "id": "topic-1",
                    "topic_key": "renal:kidney-functions",
                    "normalization_version": "norm-v2",
                    "canonical_label": "kidney functions",
                    "display_label": "Kidney Functions",
                    "selected": True,
                }
            ]
        },
        refreshed_from_snapshot_id=original_snapshot_id,
    )

    async def fake_dispatch_action(*, note_db, snapshot_row, request_body, media_db=None):
        assert int(snapshot_row["id"]) == child_snapshot_id  # nosec B101
        return {
            "target_service": "quiz",
            "target_type": "quiz",
            "target_id": "quiz-generated",
        }

    monkeypatch.setattr(endpoints_mod, "_dispatch_follow_up_action", fake_dispatch_action)

    response = client.post(
        f"/api/v1/study-suggestions/snapshots/{child_snapshot_id}/actions",
        json={
            "target_service": "quiz",
            "target_type": "quiz",
            "action_kind": "follow_up_quiz",
            "selected_topic_ids": ["topic-1"],
            "generator_version": "v1",
            "force_regenerate": False,
        },
    )

    expected_fingerprint = actions_mod.build_selection_fingerprint(
        snapshot_id=child_snapshot_id,
        target_service="quiz",
        target_type="quiz",
        selected_topics=["renal:kidney-functions"],
        action_kind="follow_up_quiz",
        generator_version="v1",
        normalization_version="norm-v2",
    )

    assert response.status_code == 200  # nosec B101
    assert response.json()["disposition"] == "opened_existing"  # nosec B101
    assert response.json()["target_id"] == str(ancestor_quiz_id)  # nosec B101
    assert response.json()["selection_fingerprint"] == expected_fingerprint  # nosec B101


def test_snapshot_actions_ignore_deleted_targets_when_reopening_existing_links(
    client: TestClient,
    db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
):
    endpoints_mod, _quizzes_mod, _flashcards_mod, _snapshot_service_mod, _jobs_mod, actions_mod = _load_modules()
    deck_id = db.add_deck("Session Deck", "desc")
    snapshot_id = db.create_suggestion_snapshot(
        service="flashcards",
        activity_type="flashcard_review_session",
        anchor_type="flashcard_review_session",
        anchor_id=101,
        suggestion_type="study_suggestions",
        payload_json={
            "topics": [
                {
                    "id": "topic-1",
                    "topic_key": "renal:kidney-functions",
                    "canonical_label": "kidney functions",
                    "display_label": "Kidney Functions",
                    "selected": True,
                }
            ]
        },
    )
    fingerprint = actions_mod.build_selection_fingerprint(
        snapshot_id=snapshot_id,
        target_service="flashcards",
        target_type="deck",
        selected_topics=["renal:kidney-functions"],
        action_kind="follow_up_flashcards",
        generator_version="v1",
    )
    db.create_suggestion_generation_link(
        snapshot_id=snapshot_id,
        target_service="flashcards",
        target_type="deck",
        target_id=str(deck_id),
        selection_fingerprint=fingerprint,
    )
    db.soft_delete_deck_by_id(deck_id)

    monkeypatch.setattr(endpoints_mod, "build_selection_fingerprint", lambda **_: fingerprint)

    async def fake_dispatch_action(*, note_db, snapshot_row, request_body, media_db=None):
        assert int(snapshot_row["id"]) == snapshot_id  # nosec B101
        return {
            "target_service": "flashcards",
            "target_type": "deck",
            "target_id": "deck-999",
        }

    monkeypatch.setattr(endpoints_mod, "_dispatch_follow_up_action", fake_dispatch_action)

    response = client.post(
        f"/api/v1/study-suggestions/snapshots/{snapshot_id}/actions",
        json={
            "target_service": "flashcards",
            "target_type": "deck",
            "action_kind": "follow_up_flashcards",
            "selected_topic_ids": ["topic-1"],
            "generator_version": "v1",
            "force_regenerate": False,
        },
    )

    assert response.status_code == 200  # nosec B101
    assert response.json()["disposition"] == "generated"  # nosec B101
    assert response.json()["target_id"] == "deck-999"  # nosec B101
    active_link = db.execute_query(
        """
        SELECT COUNT(*) AS count
          FROM suggestion_generation_links
         WHERE snapshot_id = ? AND deleted = 0
        """,
        (snapshot_id,),
    ).fetchone()
    assert int(active_link["count"]) == 1  # nosec B101


def test_snapshot_actions_ignore_deleted_ancestor_targets_in_refreshed_lineage_lookup(
    client: TestClient,
    db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
):
    endpoints_mod, _quizzes_mod, _flashcards_mod, _snapshot_service_mod, _jobs_mod, actions_mod = _load_modules()
    deleted_deck_id = db.add_deck("Deleted Ancestor Deck", "desc")
    original_snapshot_id = db.create_suggestion_snapshot(
        service="flashcards",
        activity_type="flashcard_review_session",
        anchor_type="flashcard_review_session",
        anchor_id=101,
        suggestion_type="study_suggestions",
        payload_json={
            "topics": [
                {
                    "id": "topic-1",
                    "topic_key": "renal:kidney-functions",
                    "normalization_version": "norm-v2",
                    "canonical_label": "kidney functions",
                    "display_label": "Kidney Functions",
                    "selected": True,
                }
            ]
        },
    )
    fingerprint = actions_mod.build_selection_fingerprint(
        snapshot_id=original_snapshot_id,
        target_service="flashcards",
        target_type="deck",
        selected_topics=["renal:kidney-functions"],
        action_kind="follow_up_flashcards",
        generator_version="v1",
        normalization_version="norm-v2",
    )
    db.create_suggestion_generation_link(
        snapshot_id=original_snapshot_id,
        target_service="flashcards",
        target_type="deck",
        target_id=str(deleted_deck_id),
        selection_fingerprint=fingerprint,
    )
    db.soft_delete_deck_by_id(deleted_deck_id)
    child_snapshot_id = db.create_suggestion_snapshot(
        service="flashcards",
        activity_type="flashcard_review_session",
        anchor_type="flashcard_review_session",
        anchor_id=101,
        suggestion_type="study_suggestions",
        payload_json={
            "topics": [
                {
                    "id": "topic-1",
                    "topic_key": "renal:kidney-functions",
                    "normalization_version": "norm-v2",
                    "canonical_label": "kidney functions",
                    "display_label": "Kidney Functions",
                    "selected": True,
                }
            ]
        },
        refreshed_from_snapshot_id=original_snapshot_id,
    )

    monkeypatch.setattr(endpoints_mod, "build_selection_fingerprint", lambda **_: fingerprint)

    async def fake_dispatch_action(*, note_db, snapshot_row, request_body, media_db=None):
        assert int(snapshot_row["id"]) == child_snapshot_id  # nosec B101
        return {
            "target_service": "flashcards",
            "target_type": "deck",
            "target_id": "deck-1234",
        }

    monkeypatch.setattr(endpoints_mod, "_dispatch_follow_up_action", fake_dispatch_action)

    response = client.post(
        f"/api/v1/study-suggestions/snapshots/{child_snapshot_id}/actions",
        json={
            "target_service": "flashcards",
            "target_type": "deck",
            "action_kind": "follow_up_flashcards",
            "selected_topic_ids": ["topic-1"],
            "generator_version": "v1",
            "force_regenerate": False,
        },
    )

    assert response.status_code == 200  # nosec B101
    assert response.json()["disposition"] == "generated"  # nosec B101
    assert response.json()["target_id"] == "deck-1234"  # nosec B101


def test_snapshot_actions_generate_new_output_for_refreshed_ancestor_requests_with_manual_topic_labels(
    client: TestClient,
    db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
):
    endpoints_mod, _quizzes_mod, _flashcards_mod, _snapshot_service_mod, _jobs_mod, actions_mod = _load_modules()
    ancestor_quiz_id = db.create_quiz(name="Ancestor Quiz", source_bundle_json=[])
    original_snapshot_id = db.create_suggestion_snapshot(
        service="quiz",
        activity_type="quiz_attempt",
        anchor_type="quiz_attempt",
        anchor_id=101,
        suggestion_type="study_suggestions",
        payload_json={
            "topics": [
                {
                    "id": "topic-1",
                    "topic_key": "renal:kidney-functions",
                    "normalization_version": "norm-v2",
                    "canonical_label": "kidney functions",
                    "display_label": "Kidney Functions",
                    "selected": True,
                }
            ]
        },
    )
    fingerprint = actions_mod.build_selection_fingerprint(
        snapshot_id=original_snapshot_id,
        target_service="quiz",
        target_type="quiz",
        selected_topics=["electrolyte ladder", "renal:kidney-functions"],
        action_kind="follow_up_quiz",
        generator_version="v1",
        normalization_version="norm-v2",
    )
    db.create_suggestion_generation_link(
        snapshot_id=original_snapshot_id,
        target_service="quiz",
        target_type="quiz",
        target_id=str(ancestor_quiz_id),
        selection_fingerprint=fingerprint,
    )
    child_snapshot_id = db.create_suggestion_snapshot(
        service="quiz",
        activity_type="quiz_attempt",
        anchor_type="quiz_attempt",
        anchor_id=101,
        suggestion_type="study_suggestions",
        payload_json={
            "topics": [
                {
                    "id": "topic-1",
                    "topic_key": "renal:kidney-functions",
                    "normalization_version": "norm-v2",
                    "canonical_label": "kidney functions",
                    "display_label": "Kidney Functions",
                    "selected": True,
                }
            ]
        },
        refreshed_from_snapshot_id=original_snapshot_id,
    )

    async def fake_dispatch_action(*, note_db, snapshot_row, request_body, media_db=None):
        assert int(snapshot_row["id"]) == child_snapshot_id  # nosec B101
        return {
            "target_service": "quiz",
            "target_type": "quiz",
            "target_id": "quiz-generated",
        }

    monkeypatch.setattr(endpoints_mod, "_dispatch_follow_up_action", fake_dispatch_action)

    response = client.post(
        f"/api/v1/study-suggestions/snapshots/{child_snapshot_id}/actions",
        json={
            "target_service": "quiz",
            "target_type": "quiz",
            "action_kind": "follow_up_quiz",
            "selected_topic_ids": ["topic-1"],
            "manual_topic_labels": [" Electrolyte Ladder "],
            "generator_version": "v1",
            "force_regenerate": False,
        },
    )

    expected_fingerprint = actions_mod.build_selection_fingerprint(
        snapshot_id=child_snapshot_id,
        target_service="quiz",
        target_type="quiz",
        selected_topics=["electrolyte ladder", "renal:kidney-functions"],
        action_kind="follow_up_quiz",
        generator_version="v1",
        normalization_version="norm-v2",
    )

    assert response.status_code == 200  # nosec B101
    assert response.json()["disposition"] == "generated"  # nosec B101
    assert response.json()["target_id"] == "quiz-generated"  # nosec B101
    assert response.json()["selection_fingerprint"] == expected_fingerprint  # nosec B101


def test_snapshot_actions_force_regenerate_replaces_active_direct_link(
    client: TestClient,
    db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
):
    endpoints_mod, _quizzes_mod, _flashcards_mod, _snapshot_service_mod, _jobs_mod, actions_mod = _load_modules()
    existing_quiz_id = db.create_quiz(name="Existing Quiz", source_bundle_json=[])
    snapshot_id = db.create_suggestion_snapshot(
        service="quiz",
        activity_type="quiz_attempt",
        anchor_type="quiz_attempt",
        anchor_id=101,
        suggestion_type="study_suggestions",
        payload_json={
            "topics": [
                {
                    "id": "topic-1",
                    "topic_key": "renal:kidney-functions",
                    "canonical_label": "kidney functions",
                    "display_label": "Kidney Functions",
                    "selected": True,
                }
            ]
        },
    )
    fingerprint = actions_mod.build_selection_fingerprint(
        snapshot_id=snapshot_id,
        target_service="quiz",
        target_type="quiz",
        selected_topics=["renal:kidney-functions"],
        action_kind="follow_up_quiz",
        generator_version="v1",
    )
    db.create_suggestion_generation_link(
        snapshot_id=snapshot_id,
        target_service="quiz",
        target_type="quiz",
        target_id=str(existing_quiz_id),
        selection_fingerprint=fingerprint,
    )

    monkeypatch.setattr(endpoints_mod, "build_selection_fingerprint", lambda **_: fingerprint)

    async def fake_dispatch_action(*, note_db, snapshot_row, request_body, media_db=None):
        assert int(snapshot_row["id"]) == snapshot_id  # nosec B101
        assert request_body["force_regenerate"] is True  # nosec B101
        return {
            "target_service": "quiz",
            "target_type": "quiz",
            "target_id": "quiz-99",
        }

    monkeypatch.setattr(endpoints_mod, "_dispatch_follow_up_action", fake_dispatch_action)

    response = client.post(
        f"/api/v1/study-suggestions/snapshots/{snapshot_id}/actions",
        json={
            "target_service": "quiz",
            "target_type": "quiz",
            "action_kind": "follow_up_quiz",
            "selected_topic_ids": ["topic-1"],
            "generator_version": "v1",
            "force_regenerate": True,
        },
    )

    assert response.status_code == 200  # nosec B101
    assert response.json()["disposition"] == "generated"  # nosec B101
    assert response.json()["target_id"] == "quiz-99"  # nosec B101
    active_links = db.execute_query(
        """
        SELECT COUNT(*) AS count
          FROM suggestion_generation_links
         WHERE snapshot_id = ? AND selection_fingerprint = ? AND deleted = 0
        """,
        (snapshot_id, fingerprint),
    ).fetchone()
    assert int(active_links["count"]) == 1  # nosec B101
