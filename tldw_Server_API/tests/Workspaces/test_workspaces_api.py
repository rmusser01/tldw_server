"""Tests for workspace CRUD endpoints and scoped chat session isolation."""
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.endpoints import workspaces as workspaces_endpoint
from tldw_Server_API.app.api.v1.endpoints.workspaces_rate_limit_policy import (
    WORKSPACES_DELETE_RATE_LIMIT,
    WORKSPACES_WRITE_RATE_LIMIT,
)
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    ConflictError,
)


class _CapturingJobManager:
    def __init__(self) -> None:
        self.created_jobs: list[dict[str, Any]] = []
        self.jobs_by_key: dict[str, dict[str, Any]] = {}

    def create_job(self, **kwargs: Any) -> dict[str, Any]:
        idempotency_key = str(kwargs.get("idempotency_key") or "")
        if idempotency_key and idempotency_key in self.jobs_by_key:
            return self.jobs_by_key[idempotency_key]
        self.created_jobs.append(kwargs)
        job = {"id": len(self.created_jobs), **kwargs}
        if idempotency_key:
            self.jobs_by_key[idempotency_key] = job
        return job


class _FailingJobManager:
    def create_job(self, **kwargs: Any) -> dict[str, Any]:
        _ = kwargs
        raise RuntimeError("jobs backend unavailable")


@pytest.fixture
def db(tmp_path):
    d = CharactersRAGDB(db_path=str(tmp_path / "chacha.db"), client_id="user-1")
    d.add_character_card({"name": "Test Char"})
    return d


@pytest.fixture
def workspace_fastapi_app():
    from tldw_Server_API.app.main import app

    return app


class TestWorkspaceLifecycle:
    def test_upsert_then_get(self, db):
        ws = db.upsert_workspace("ws-1", "My Workspace", study_materials_policy="workspace")
        assert ws["id"] == "ws-1"
        assert ws["study_materials_policy"] == "workspace"
        fetched = db.get_workspace("ws-1")
        assert fetched["name"] == "My Workspace"
        assert fetched["study_materials_policy"] == "workspace"

    def test_upsert_workspace_updates_existing_policy(self, db):
        original = db.upsert_workspace("ws-1", "Original Name", study_materials_policy="general")
        updated = db.upsert_workspace("ws-1", "Renamed Workspace", study_materials_policy="workspace")
        assert updated["id"] == original["id"]
        assert updated["name"] == "Renamed Workspace"
        assert updated["study_materials_policy"] == "workspace"
        assert updated["version"] == original["version"] + 1

    def test_patch_workspace_name(self, db):
        db.upsert_workspace("ws-1", "Old")
        ws = db.update_workspace("ws-1", {"name": "New"}, expected_version=1)
        assert ws["name"] == "New"
        assert ws["version"] == 2

    def test_archive_workspace(self, db):
        db.upsert_workspace("ws-1", "WS")
        ws = db.update_workspace("ws-1", {"archived": True}, expected_version=1)
        assert ws["archived"] in (True, 1)

    def test_delete_workspace_cascade(self, db):
        db.upsert_workspace("ws-1", "WS")
        conv_id = db.add_conversation({
            "title": "WS chat", "character_id": 1,
            "scope_type": "workspace", "workspace_id": "ws-1",
        })
        quiz_id = db.create_quiz(name="Workspace Quiz", workspace_id="ws-1")
        deck_id = db.add_deck("Workspace Deck", workspace_id="ws-1")
        db.delete_workspace("ws-1", expected_version=1)

        # Workspace is soft-deleted
        ws = db.get_workspace("ws-1")
        assert ws is None  # get_workspace excludes deleted

        # Conversation is also soft-deleted
        conv = db.get_conversation_by_id(conv_id)
        assert conv is None

        quiz = db.get_quiz(quiz_id)
        deck = db.get_deck(deck_id)
        assert quiz is not None
        assert deck is not None
        assert quiz["workspace_id"] is None
        assert deck["workspace_id"] is None

    def test_list_workspaces(self, db):
        for i in range(5):
            db.upsert_workspace(f"ws-{i}", f"WS {i}")
        result = db.list_workspaces()
        assert len(result) == 5

    def test_version_conflict_returns_error(self, db):
        db.upsert_workspace("ws-1", "WS")
        db.update_workspace("ws-1", {"name": "V2"}, expected_version=1)
        with pytest.raises((ConflictError, Exception)):
            db.update_workspace("ws-1", {"name": "V3"}, expected_version=1)

    def test_workspace_policy_updates(self, db):
        db.upsert_workspace("ws-1", "WS")
        ws = db.update_workspace("ws-1", {"study_materials_policy": "workspace"}, expected_version=1)
        assert ws["study_materials_policy"] == "workspace"


@pytest.mark.integration
def test_workspace_api_accepts_and_returns_study_materials_policy(workspace_fastapi_app, db):
    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user, User
    from tldw_Server_API.app.api.v1.endpoints.workspaces_rate_limit_policy import (
        WORKSPACES_READ_RATE_LIMIT,
        WORKSPACES_WRITE_RATE_LIMIT,
    )

    async def _allow_rate_limit() -> None:
        return None

    async def _user() -> User:
        return User(
            id=1,
            username="testuser",
            email="test@example.com",
            is_active=True,
            roles=["admin"],
            is_admin=True,
        )

    def _db() -> CharactersRAGDB:
        return db

    workspace_fastapi_app.dependency_overrides[get_request_user] = _user
    workspace_fastapi_app.dependency_overrides[get_chacha_db_for_user] = _db
    workspace_fastapi_app.dependency_overrides[WORKSPACES_READ_RATE_LIMIT] = _allow_rate_limit
    workspace_fastapi_app.dependency_overrides[WORKSPACES_WRITE_RATE_LIMIT] = _allow_rate_limit
    try:
        with TestClient(workspace_fastapi_app, raise_server_exceptions=False) as client:
            create_response = client.put(
                "/api/v1/workspaces/ws-api",
                json={"name": "API Workspace", "study_materials_policy": "workspace"},
            )
            assert create_response.status_code == 200, create_response.text
            created = create_response.json()
            assert created["study_materials_policy"] == "workspace"

            upsert_response = client.put(
                "/api/v1/workspaces/ws-api",
                json={"name": "API Workspace Renamed", "study_materials_policy": "general"},
            )
            assert upsert_response.status_code == 200, upsert_response.text
            upserted = upsert_response.json()
            assert upserted["name"] == "API Workspace Renamed"
            assert upserted["study_materials_policy"] == "general"

            patch_response = client.patch(
                f"/api/v1/workspaces/{created['id']}",
                json={"study_materials_policy": "workspace", "version": upserted["version"]},
            )
            assert patch_response.status_code == 200, patch_response.text
            patched = patch_response.json()
            assert patched["study_materials_policy"] == "workspace"
    finally:
        workspace_fastapi_app.dependency_overrides.pop(get_request_user, None)
        workspace_fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_READ_RATE_LIMIT, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_WRITE_RATE_LIMIT, None)


class TestScopedChatSessions:
    def test_workspace_chat_not_visible_in_global_list(self, db):
        db.upsert_workspace("ws-1", "WS")
        db.add_conversation({"title": "Global", "character_id": 1})
        db.add_conversation({
            "title": "WS Chat", "character_id": 1,
            "scope_type": "workspace", "workspace_id": "ws-1",
        })
        global_results = db.search_conversations(None, scope_type="global")
        assert all(r["scope_type"] == "global" for r in global_results)

    def test_global_chat_not_visible_in_workspace_list(self, db):
        db.upsert_workspace("ws-1", "WS")
        db.add_conversation({"title": "Global", "character_id": 1})
        ws_results = db.search_conversations(None, scope_type="workspace", workspace_id="ws-1")
        assert len(ws_results) == 0


@pytest.mark.integration
def test_delete_workspace_maps_conflict_to_409(workspace_fastapi_app):
    class _ConflictDB:
        def get_workspace(self, workspace_id: str):
            return {"id": workspace_id, "version": 1}

        def delete_workspace(self, workspace_id: str, expected_version: int) -> None:
            _ = (workspace_id, expected_version)
            raise ConflictError("Workspace 'ws-1' concurrent delete detected.")

    async def _allow_rate_limit() -> None:
        return None

    workspace_fastapi_app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)
    workspace_fastapi_app.dependency_overrides[get_chacha_db_for_user] = lambda: _ConflictDB()
    workspace_fastapi_app.dependency_overrides[WORKSPACES_DELETE_RATE_LIMIT] = _allow_rate_limit
    try:
        with TestClient(workspace_fastapi_app, raise_server_exceptions=False) as client:
            response = client.delete("/api/v1/workspaces/ws-1")
    finally:
        workspace_fastapi_app.dependency_overrides.pop(get_request_user, None)
        workspace_fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_DELETE_RATE_LIMIT, None)

    assert response.status_code == 409, response.text


@pytest.mark.integration
def test_add_workspace_source_enqueues_idempotent_ingest_job(workspace_fastapi_app, db):
    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User

    async def _allow_rate_limit() -> None:
        return None

    async def _user() -> User:
        return User(
            id=7,
            username="researcher",
            email="researcher@example.com",
            is_active=True,
            roles=["admin"],
            is_admin=True,
        )

    job_manager = _CapturingJobManager()
    db.upsert_workspace("ws-job-api", "Workspace Source Jobs")
    workspace_fastapi_app.dependency_overrides[get_request_user] = _user
    workspace_fastapi_app.dependency_overrides[get_chacha_db_for_user] = lambda: db
    workspace_fastapi_app.dependency_overrides[workspaces_endpoint.try_get_workspace_job_manager] = lambda: job_manager
    workspace_fastapi_app.dependency_overrides[WORKSPACES_WRITE_RATE_LIMIT] = _allow_rate_limit
    try:
        with TestClient(workspace_fastapi_app, raise_server_exceptions=False) as client:
            response = client.post(
                "/api/v1/workspaces/ws-job-api/sources",
                json={
                    "id": "src-job-1",
                    "media_id": 55,
                    "title": "NotebookLM Migration PDF",
                    "source_type": "pdf",
                    "url": "file:///imports/notebooklm.pdf",
                },
            )
            duplicate_response = client.post(
                "/api/v1/workspaces/ws-job-api/sources",
                json={
                    "id": "src-job-1",
                    "media_id": 55,
                    "title": "NotebookLM Migration PDF",
                    "source_type": "pdf",
                    "url": "file:///imports/notebooklm.pdf",
                },
            )
    finally:
        workspace_fastapi_app.dependency_overrides.pop(get_request_user, None)
        workspace_fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(workspaces_endpoint.try_get_workspace_job_manager, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_WRITE_RATE_LIMIT, None)

    assert response.status_code == 201, response.text
    assert duplicate_response.status_code == 201, duplicate_response.text
    assert duplicate_response.json()["id"] == "src-job-1"
    assert len(job_manager.created_jobs) == 1
    job = job_manager.created_jobs[0]
    assert job["domain"] == "media_ingest"
    assert job["queue"] == "default"
    assert job["job_type"] == "workspace_source_ingest"
    assert job["owner_user_id"] == "7"
    assert job["idempotency_key"] == "workspace-source:ws-job-api:src-job-1:55"
    assert job["payload"] == {
        "workspace_id": "ws-job-api",
        "workspace_source_id": "src-job-1",
        "source_id": "src-job-1",
        "media_id": 55,
        "source_type": "pdf",
        "title": "NotebookLM Migration PDF",
        "url": "file:///imports/notebooklm.pdf",
        "requested_stages": ["ingestion", "extraction", "chunking", "indexing"],
    }


@pytest.mark.integration
def test_add_workspace_source_preserves_row_when_job_enqueue_fails(workspace_fastapi_app, db):
    async def _allow_rate_limit() -> None:
        return None

    db.upsert_workspace("ws-job-fail-api", "Workspace Source Job Failure")
    workspace_fastapi_app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=8)
    workspace_fastapi_app.dependency_overrides[get_chacha_db_for_user] = lambda: db
    workspace_fastapi_app.dependency_overrides[workspaces_endpoint.try_get_workspace_job_manager] = (
        lambda: _FailingJobManager()
    )
    workspace_fastapi_app.dependency_overrides[WORKSPACES_WRITE_RATE_LIMIT] = _allow_rate_limit
    try:
        with TestClient(workspace_fastapi_app, raise_server_exceptions=False) as client:
            response = client.post(
                "/api/v1/workspaces/ws-job-fail-api/sources",
                json={
                    "id": "src-job-fail",
                    "media_id": 56,
                    "title": "Resilient Source",
                    "source_type": "web",
                    "url": "https://example.test/resilient-source",
                },
            )
    finally:
        workspace_fastapi_app.dependency_overrides.pop(get_request_user, None)
        workspace_fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(workspaces_endpoint.try_get_workspace_job_manager, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_WRITE_RATE_LIMIT, None)

    assert response.status_code == 201, response.text
    assert response.json()["id"] == "src-job-fail"
    assert [src["id"] for src in db.list_workspace_sources("ws-job-fail-api")] == ["src-job-fail"]


@pytest.mark.integration
def test_add_workspace_source_preserves_row_when_job_manager_dependency_fails(
    workspace_fastapi_app,
    db,
    monkeypatch: pytest.MonkeyPatch,
):
    async def _allow_rate_limit() -> None:
        return None

    def _raise_job_manager() -> None:
        raise RuntimeError("jobs manager construction failed")

    db.upsert_workspace("ws-job-dep-fail-api", "Workspace Source Job Dependency Failure")
    monkeypatch.setattr(workspaces_endpoint, "get_job_manager", _raise_job_manager)
    workspace_fastapi_app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=9)
    workspace_fastapi_app.dependency_overrides[get_chacha_db_for_user] = lambda: db
    workspace_fastapi_app.dependency_overrides[WORKSPACES_WRITE_RATE_LIMIT] = _allow_rate_limit
    try:
        with TestClient(workspace_fastapi_app, raise_server_exceptions=False) as client:
            response = client.post(
                "/api/v1/workspaces/ws-job-dep-fail-api/sources",
                json={
                    "id": "src-job-dep-fail",
                    "media_id": 57,
                    "title": "Dependency Resilient Source",
                    "source_type": "pdf",
                },
            )
    finally:
        workspace_fastapi_app.dependency_overrides.pop(get_request_user, None)
        workspace_fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_WRITE_RATE_LIMIT, None)

    assert response.status_code == 201, response.text
    assert response.json()["id"] == "src-job-dep-fail"
    assert [src["id"] for src in db.list_workspace_sources("ws-job-dep-fail-api")] == ["src-job-dep-fail"]
