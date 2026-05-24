"""Tests for workspace CRUD endpoints and scoped chat session isolation."""
import json
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.endpoints.workspaces_rate_limit_policy import (
    WORKSPACES_DELETE_RATE_LIMIT,
    WORKSPACES_READ_RATE_LIMIT,
    WORKSPACES_WRITE_RATE_LIMIT,
)
from tldw_Server_API.app.api.v1.endpoints import workspaces as workspaces_endpoint
from tldw_Server_API.app.api.v1.API_Deps.jobs_deps import get_job_manager
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDBError,
    CharactersRAGDB,
    ConflictError,
    InputError,
)


class _CapturingJobManager:
    def __init__(self) -> None:
        self.created_jobs: list[dict[str, Any]] = []

    def create_job(self, **kwargs: Any) -> dict[str, Any]:
        self.created_jobs.append(kwargs)
        return {"id": len(self.created_jobs), **kwargs}


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


@pytest.mark.integration
def test_workspace_root_endpoints_happy_path(workspace_fastapi_app, db):
    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User

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
            upsert_response = client.put(
                "/api/v1/workspaces/ws-root-api",
                json={"name": "Root Workspace", "study_materials_policy": "workspace"},
            )
            assert upsert_response.status_code == 200, upsert_response.text
            upserted = upsert_response.json()
            assert upserted["id"] == "ws-root-api"
            assert upserted["name"] == "Root Workspace"

            get_response = client.get("/api/v1/workspaces/ws-root-api")
            assert get_response.status_code == 200, get_response.text
            fetched = get_response.json()
            assert fetched["id"] == "ws-root-api"
            assert fetched["study_materials_policy"] == "workspace"

            list_response = client.get("/api/v1/workspaces/")
            assert list_response.status_code == 200, list_response.text
            payload = list_response.json()
            assert payload["total"] == 1
            assert payload["items"][0]["id"] == "ws-root-api"
    finally:
        workspace_fastapi_app.dependency_overrides.pop(get_request_user, None)
        workspace_fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_READ_RATE_LIMIT, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_WRITE_RATE_LIMIT, None)


@pytest.mark.integration
def test_list_workspaces_maps_database_error_to_contextual_500(workspace_fastapi_app):
    class _DatabaseErrorDB:
        def list_workspaces(self):
            raise CharactersRAGDBError("sqlite backend unavailable")

    async def _allow_rate_limit() -> None:
        return None

    workspace_fastapi_app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)
    workspace_fastapi_app.dependency_overrides[get_chacha_db_for_user] = lambda: _DatabaseErrorDB()
    workspace_fastapi_app.dependency_overrides[WORKSPACES_READ_RATE_LIMIT] = _allow_rate_limit
    try:
        with TestClient(workspace_fastapi_app, raise_server_exceptions=False) as client:
            response = client.get("/api/v1/workspaces/")
    finally:
        workspace_fastapi_app.dependency_overrides.pop(get_request_user, None)
        workspace_fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_READ_RATE_LIMIT, None)

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to fetch workspaces"


@pytest.mark.integration
def test_get_workspace_maps_database_error_to_contextual_500(workspace_fastapi_app):
    class _DatabaseErrorDB:
        def get_workspace(self, workspace_id: str):
            _ = workspace_id
            raise CharactersRAGDBError("sqlite backend unavailable")

    async def _allow_rate_limit() -> None:
        return None

    workspace_fastapi_app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)
    workspace_fastapi_app.dependency_overrides[get_chacha_db_for_user] = lambda: _DatabaseErrorDB()
    workspace_fastapi_app.dependency_overrides[WORKSPACES_READ_RATE_LIMIT] = _allow_rate_limit
    try:
        with TestClient(workspace_fastapi_app, raise_server_exceptions=False) as client:
            response = client.get("/api/v1/workspaces/ws-1")
    finally:
        workspace_fastapi_app.dependency_overrides.pop(get_request_user, None)
        workspace_fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_READ_RATE_LIMIT, None)

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to fetch workspace"


@pytest.mark.integration
def test_upsert_workspace_maps_input_error_to_400(workspace_fastapi_app):
    class _InputErrorDB:
        def upsert_workspace(self, workspace_id: str, name: str, *, study_materials_policy: str):
            _ = (workspace_id, name, study_materials_policy)
            raise InputError("invalid workspace create")

    async def _allow_rate_limit() -> None:
        return None

    workspace_fastapi_app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)
    workspace_fastapi_app.dependency_overrides[get_chacha_db_for_user] = lambda: _InputErrorDB()
    workspace_fastapi_app.dependency_overrides[WORKSPACES_WRITE_RATE_LIMIT] = _allow_rate_limit
    try:
        with TestClient(workspace_fastapi_app, raise_server_exceptions=False) as client:
            response = client.put(
                "/api/v1/workspaces/ws-1",
                json={"name": "Workspace", "study_materials_policy": "workspace"},
            )
    finally:
        workspace_fastapi_app.dependency_overrides.pop(get_request_user, None)
        workspace_fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_WRITE_RATE_LIMIT, None)

    assert response.status_code == 400, response.text
    assert response.json()["detail"] == "invalid workspace create"


@pytest.mark.integration
def test_upsert_workspace_maps_database_error_to_contextual_500(workspace_fastapi_app):
    class _DatabaseErrorDB:
        def upsert_workspace(self, workspace_id: str, name: str, *, study_materials_policy: str):
            _ = (workspace_id, name, study_materials_policy)
            raise CharactersRAGDBError("sqlite backend unavailable")

    async def _allow_rate_limit() -> None:
        return None

    workspace_fastapi_app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)
    workspace_fastapi_app.dependency_overrides[get_chacha_db_for_user] = lambda: _DatabaseErrorDB()
    workspace_fastapi_app.dependency_overrides[WORKSPACES_WRITE_RATE_LIMIT] = _allow_rate_limit
    try:
        with TestClient(workspace_fastapi_app, raise_server_exceptions=False) as client:
            response = client.put(
                "/api/v1/workspaces/ws-1",
                json={"name": "Workspace", "study_materials_policy": "workspace"},
            )
    finally:
        workspace_fastapi_app.dependency_overrides.pop(get_request_user, None)
        workspace_fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_WRITE_RATE_LIMIT, None)

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to create or update workspace"


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
def test_patch_workspace_maps_input_error_to_400(workspace_fastapi_app):
    class _InputErrorDB:
        def update_workspace(self, workspace_id: str, updates: dict, expected_version: int):
            _ = (workspace_id, updates, expected_version)
            raise InputError("invalid workspace patch")

    async def _allow_rate_limit() -> None:
        return None

    workspace_fastapi_app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)
    workspace_fastapi_app.dependency_overrides[get_chacha_db_for_user] = lambda: _InputErrorDB()
    workspace_fastapi_app.dependency_overrides[WORKSPACES_WRITE_RATE_LIMIT] = _allow_rate_limit
    try:
        with TestClient(workspace_fastapi_app, raise_server_exceptions=False) as client:
            response = client.patch(
                "/api/v1/workspaces/ws-1",
                json={"name": "Renamed", "version": 1},
            )
    finally:
        workspace_fastapi_app.dependency_overrides.pop(get_request_user, None)
        workspace_fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_WRITE_RATE_LIMIT, None)

    assert response.status_code == 400, response.text
    assert response.json()["detail"] == "invalid workspace patch"


@pytest.mark.integration
def test_delete_workspace_maps_database_error_to_contextual_500(workspace_fastapi_app):
    class _DatabaseErrorDB:
        def get_workspace(self, workspace_id: str):
            return {"id": workspace_id, "version": 1}

        def delete_workspace(self, workspace_id: str, expected_version: int) -> None:
            _ = (workspace_id, expected_version)
            raise CharactersRAGDBError("sqlite backend unavailable")

    async def _allow_rate_limit() -> None:
        return None

    workspace_fastapi_app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)
    workspace_fastapi_app.dependency_overrides[get_chacha_db_for_user] = lambda: _DatabaseErrorDB()
    workspace_fastapi_app.dependency_overrides[WORKSPACES_DELETE_RATE_LIMIT] = _allow_rate_limit
    try:
        with TestClient(workspace_fastapi_app, raise_server_exceptions=False) as client:
            response = client.delete("/api/v1/workspaces/ws-1")
    finally:
        workspace_fastapi_app.dependency_overrides.pop(get_request_user, None)
        workspace_fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_DELETE_RATE_LIMIT, None)

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to delete workspace"


@pytest.mark.integration
def test_update_workspace_source_maps_input_error_to_400(workspace_fastapi_app):
    class _InputErrorDB:
        def get_workspace(self, workspace_id: str):
            return {"id": workspace_id, "version": 1}

        def update_workspace_source(
            self,
            workspace_id: str,
            source_id: str,
            updates: dict,
            *,
            expected_version: int,
        ):
            _ = (workspace_id, source_id, updates, expected_version)
            raise InputError("invalid workspace source patch")

    async def _allow_rate_limit() -> None:
        return None

    workspace_fastapi_app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)
    workspace_fastapi_app.dependency_overrides[get_chacha_db_for_user] = lambda: _InputErrorDB()
    workspace_fastapi_app.dependency_overrides[WORKSPACES_WRITE_RATE_LIMIT] = _allow_rate_limit
    try:
        with TestClient(workspace_fastapi_app, raise_server_exceptions=False) as client:
            response = client.put(
                "/api/v1/workspaces/ws-1/sources/src-1",
                json={"title": "Renamed", "version": 1},
            )
    finally:
        workspace_fastapi_app.dependency_overrides.pop(get_request_user, None)
        workspace_fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_WRITE_RATE_LIMIT, None)

    assert response.status_code == 400, response.text
    assert response.json()["detail"] == "invalid workspace source patch"


@pytest.mark.integration
def test_workspace_artifact_endpoints_happy_path(workspace_fastapi_app, db):
    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User

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
    workspace_fastapi_app.dependency_overrides[WORKSPACES_DELETE_RATE_LIMIT] = _allow_rate_limit
    try:
        db.upsert_workspace("ws-art-api", "Workspace Artifacts")

        with TestClient(workspace_fastapi_app, raise_server_exceptions=False) as client:
            add_response = client.post(
                "/api/v1/workspaces/ws-art-api/artifacts",
                json={
                    "id": "art-1",
                    "artifact_type": "summary",
                    "title": "Draft Summary",
                    "content": "Initial summary",
                },
            )
            assert add_response.status_code == 201, add_response.text
            added = add_response.json()
            assert added["id"] == "art-1"
            assert added["status"] == "pending"
            assert added["version"] == 1

            list_response = client.get("/api/v1/workspaces/ws-art-api/artifacts")
            assert list_response.status_code == 200, list_response.text
            listed = list_response.json()
            assert len(listed) == 1
            assert listed[0]["id"] == "art-1"
            assert listed[0]["title"] == "Draft Summary"

            update_response = client.put(
                "/api/v1/workspaces/ws-art-api/artifacts/art-1",
                json={
                    "title": "Final Summary",
                    "status": "completed",
                    "content": "Completed summary",
                    "version": added["version"],
                },
            )
            assert update_response.status_code == 200, update_response.text
            updated = update_response.json()
            assert updated["title"] == "Final Summary"
            assert updated["status"] == "completed"
            assert updated["content"] == "Completed summary"
            assert updated["version"] == 2

            delete_response = client.delete("/api/v1/workspaces/ws-art-api/artifacts/art-1")
            assert delete_response.status_code == 204, delete_response.text

            final_list_response = client.get("/api/v1/workspaces/ws-art-api/artifacts")
            assert final_list_response.status_code == 200, final_list_response.text
            assert final_list_response.json() == []
    finally:
        workspace_fastapi_app.dependency_overrides.pop(get_request_user, None)
        workspace_fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_READ_RATE_LIMIT, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_WRITE_RATE_LIMIT, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_DELETE_RATE_LIMIT, None)


@pytest.mark.integration
def test_list_workspace_artifacts_maps_database_error_to_contextual_500(workspace_fastapi_app):
    class _DatabaseErrorDB:
        def get_workspace(self, workspace_id: str):
            return {"id": workspace_id, "version": 1}

        def list_workspace_artifacts(self, workspace_id: str):
            _ = workspace_id
            raise CharactersRAGDBError("sqlite backend unavailable")

    async def _allow_rate_limit() -> None:
        return None

    workspace_fastapi_app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)
    workspace_fastapi_app.dependency_overrides[get_chacha_db_for_user] = lambda: _DatabaseErrorDB()
    workspace_fastapi_app.dependency_overrides[WORKSPACES_READ_RATE_LIMIT] = _allow_rate_limit
    try:
        with TestClient(workspace_fastapi_app, raise_server_exceptions=False) as client:
            response = client.get("/api/v1/workspaces/ws-1/artifacts")
    finally:
        workspace_fastapi_app.dependency_overrides.pop(get_request_user, None)
        workspace_fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_READ_RATE_LIMIT, None)

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to fetch workspace artifacts"


@pytest.mark.integration
def test_add_workspace_artifact_maps_input_error_to_400(workspace_fastapi_app):
    class _InputErrorDB:
        def get_workspace(self, workspace_id: str):
            return {"id": workspace_id, "version": 1}

        def add_workspace_artifact(self, workspace_id: str, data: dict):
            _ = (workspace_id, data)
            raise InputError("invalid workspace artifact create")

    async def _allow_rate_limit() -> None:
        return None

    workspace_fastapi_app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)
    workspace_fastapi_app.dependency_overrides[get_chacha_db_for_user] = lambda: _InputErrorDB()
    workspace_fastapi_app.dependency_overrides[WORKSPACES_WRITE_RATE_LIMIT] = _allow_rate_limit
    try:
        with TestClient(workspace_fastapi_app, raise_server_exceptions=False) as client:
            response = client.post(
                "/api/v1/workspaces/ws-1/artifacts",
                json={
                    "id": "art-1",
                    "artifact_type": "summary",
                    "title": "Draft Summary",
                },
            )
    finally:
        workspace_fastapi_app.dependency_overrides.pop(get_request_user, None)
        workspace_fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_WRITE_RATE_LIMIT, None)

    assert response.status_code == 400, response.text
    assert response.json()["detail"] == "invalid workspace artifact create"


@pytest.mark.integration
def test_delete_workspace_artifact_maps_database_error_to_contextual_500(workspace_fastapi_app):
    class _DatabaseErrorDB:
        def get_workspace(self, workspace_id: str):
            return {"id": workspace_id, "version": 1}

        def delete_workspace_artifact(self, workspace_id: str, artifact_id: str) -> None:
            _ = (workspace_id, artifact_id)
            raise CharactersRAGDBError("sqlite backend unavailable")

    async def _allow_rate_limit() -> None:
        return None

    workspace_fastapi_app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)
    workspace_fastapi_app.dependency_overrides[get_chacha_db_for_user] = lambda: _DatabaseErrorDB()
    workspace_fastapi_app.dependency_overrides[WORKSPACES_DELETE_RATE_LIMIT] = _allow_rate_limit
    try:
        with TestClient(workspace_fastapi_app, raise_server_exceptions=False) as client:
            response = client.delete("/api/v1/workspaces/ws-1/artifacts/art-1")
    finally:
        workspace_fastapi_app.dependency_overrides.pop(get_request_user, None)
        workspace_fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_DELETE_RATE_LIMIT, None)

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to delete workspace artifact"


@pytest.mark.integration
def test_update_workspace_artifact_maps_database_error_to_contextual_500(workspace_fastapi_app):
    class _DatabaseErrorDB:
        def get_workspace(self, workspace_id: str):
            return {"id": workspace_id, "version": 1}

        def update_workspace_artifact(
            self,
            workspace_id: str,
            artifact_id: str,
            updates: dict,
            *,
            expected_version: int,
        ):
            _ = (workspace_id, artifact_id, updates, expected_version)
            raise CharactersRAGDBError("sqlite backend unavailable")

    async def _allow_rate_limit() -> None:
        return None

    workspace_fastapi_app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)
    workspace_fastapi_app.dependency_overrides[get_chacha_db_for_user] = lambda: _DatabaseErrorDB()
    workspace_fastapi_app.dependency_overrides[WORKSPACES_WRITE_RATE_LIMIT] = _allow_rate_limit
    try:
        with TestClient(workspace_fastapi_app, raise_server_exceptions=False) as client:
            response = client.put(
                "/api/v1/workspaces/ws-1/artifacts/art-1",
                json={"title": "Updated", "version": 1},
            )
    finally:
        workspace_fastapi_app.dependency_overrides.pop(get_request_user, None)
        workspace_fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_WRITE_RATE_LIMIT, None)

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to update workspace artifact"


@pytest.mark.integration
def test_update_workspace_note_maps_input_error_to_400(workspace_fastapi_app):
    class _InputErrorDB:
        def get_workspace(self, workspace_id: str):
            return {"id": workspace_id, "version": 1}

        def update_workspace_note(
            self,
            workspace_id: str,
            note_id: int,
            updates: dict,
            *,
            expected_version: int,
        ):
            _ = (workspace_id, note_id, updates, expected_version)
            raise InputError("invalid workspace note patch")

    async def _allow_rate_limit() -> None:
        return None

    workspace_fastapi_app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)
    workspace_fastapi_app.dependency_overrides[get_chacha_db_for_user] = lambda: _InputErrorDB()
    workspace_fastapi_app.dependency_overrides[WORKSPACES_WRITE_RATE_LIMIT] = _allow_rate_limit
    try:
        with TestClient(workspace_fastapi_app, raise_server_exceptions=False) as client:
            response = client.put(
                "/api/v1/workspaces/ws-1/notes/42",
                json={"title": "Updated", "version": 1},
            )
    finally:
        workspace_fastapi_app.dependency_overrides.pop(get_request_user, None)
        workspace_fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_WRITE_RATE_LIMIT, None)

    assert response.status_code == 400, response.text
    assert response.json()["detail"] == "invalid workspace note patch"


@pytest.mark.integration
def test_workspace_note_endpoints_happy_path(workspace_fastapi_app, db):
    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User

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
    workspace_fastapi_app.dependency_overrides[WORKSPACES_DELETE_RATE_LIMIT] = _allow_rate_limit
    try:
        db.upsert_workspace("ws-note-api", "Workspace Notes")

        with TestClient(workspace_fastapi_app, raise_server_exceptions=False) as client:
            add_response = client.post(
                "/api/v1/workspaces/ws-note-api/notes",
                json={
                    "title": "Draft Note",
                    "content": "Initial note body",
                    "keywords": ["alpha", "beta"],
                },
            )
            assert add_response.status_code == 201, add_response.text
            added = add_response.json()
            assert added["title"] == "Draft Note"
            assert added["content"] == "Initial note body"
            assert json.loads(added["keywords_json"]) == ["alpha", "beta"]
            assert added["version"] == 1

            list_response = client.get("/api/v1/workspaces/ws-note-api/notes")
            assert list_response.status_code == 200, list_response.text
            listed = list_response.json()
            assert len(listed) == 1
            assert listed[0]["id"] == added["id"]

            update_response = client.put(
                f"/api/v1/workspaces/ws-note-api/notes/{added['id']}",
                json={
                    "title": "Final Note",
                    "content": "Updated note body",
                    "keywords_json": json.dumps(["gamma"]),
                    "version": added["version"],
                },
            )
            assert update_response.status_code == 200, update_response.text
            updated = update_response.json()
            assert updated["title"] == "Final Note"
            assert updated["content"] == "Updated note body"
            assert json.loads(updated["keywords_json"]) == ["gamma"]
            assert updated["version"] == 2

            delete_response = client.delete(f"/api/v1/workspaces/ws-note-api/notes/{added['id']}")
            assert delete_response.status_code == 204, delete_response.text

            final_list_response = client.get("/api/v1/workspaces/ws-note-api/notes")
            assert final_list_response.status_code == 200, final_list_response.text
            assert final_list_response.json() == []
    finally:
        workspace_fastapi_app.dependency_overrides.pop(get_request_user, None)
        workspace_fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_READ_RATE_LIMIT, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_WRITE_RATE_LIMIT, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_DELETE_RATE_LIMIT, None)


@pytest.mark.integration
def test_list_workspace_notes_maps_database_error_to_contextual_500(workspace_fastapi_app):
    class _DatabaseErrorDB:
        def get_workspace(self, workspace_id: str):
            return {"id": workspace_id, "version": 1}

        def list_workspace_notes(self, workspace_id: str):
            _ = workspace_id
            raise CharactersRAGDBError("sqlite backend unavailable")

    async def _allow_rate_limit() -> None:
        return None

    workspace_fastapi_app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)
    workspace_fastapi_app.dependency_overrides[get_chacha_db_for_user] = lambda: _DatabaseErrorDB()
    workspace_fastapi_app.dependency_overrides[WORKSPACES_READ_RATE_LIMIT] = _allow_rate_limit
    try:
        with TestClient(workspace_fastapi_app, raise_server_exceptions=False) as client:
            response = client.get("/api/v1/workspaces/ws-1/notes")
    finally:
        workspace_fastapi_app.dependency_overrides.pop(get_request_user, None)
        workspace_fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_READ_RATE_LIMIT, None)

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to fetch workspace notes"


@pytest.mark.integration
def test_add_workspace_note_maps_input_error_to_400(workspace_fastapi_app):
    class _InputErrorDB:
        def get_workspace(self, workspace_id: str):
            return {"id": workspace_id, "version": 1}

        def add_workspace_note(self, workspace_id: str, data: dict):
            _ = (workspace_id, data)
            raise InputError("invalid workspace note create")

    async def _allow_rate_limit() -> None:
        return None

    workspace_fastapi_app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)
    workspace_fastapi_app.dependency_overrides[get_chacha_db_for_user] = lambda: _InputErrorDB()
    workspace_fastapi_app.dependency_overrides[WORKSPACES_WRITE_RATE_LIMIT] = _allow_rate_limit
    try:
        with TestClient(workspace_fastapi_app, raise_server_exceptions=False) as client:
            response = client.post(
                "/api/v1/workspaces/ws-1/notes",
                json={
                    "title": "Draft Note",
                    "content": "Initial note body",
                    "keywords": ["alpha"],
                },
            )
    finally:
        workspace_fastapi_app.dependency_overrides.pop(get_request_user, None)
        workspace_fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_WRITE_RATE_LIMIT, None)

    assert response.status_code == 400, response.text
    assert response.json()["detail"] == "invalid workspace note create"


@pytest.mark.integration
def test_delete_workspace_note_maps_database_error_to_contextual_500(workspace_fastapi_app):
    class _DatabaseErrorDB:
        def get_workspace(self, workspace_id: str):
            return {"id": workspace_id, "version": 1}

        def delete_workspace_note(self, workspace_id: str, note_id: int) -> None:
            _ = (workspace_id, note_id)
            raise CharactersRAGDBError("sqlite backend unavailable")

    async def _allow_rate_limit() -> None:
        return None

    workspace_fastapi_app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)
    workspace_fastapi_app.dependency_overrides[get_chacha_db_for_user] = lambda: _DatabaseErrorDB()
    workspace_fastapi_app.dependency_overrides[WORKSPACES_DELETE_RATE_LIMIT] = _allow_rate_limit
    try:
        with TestClient(workspace_fastapi_app, raise_server_exceptions=False) as client:
            response = client.delete("/api/v1/workspaces/ws-1/notes/42")
    finally:
        workspace_fastapi_app.dependency_overrides.pop(get_request_user, None)
        workspace_fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_DELETE_RATE_LIMIT, None)

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to delete workspace note"


@pytest.mark.integration
def test_workspace_source_endpoints_happy_path(workspace_fastapi_app, db):
    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User

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

    job_manager = _CapturingJobManager()
    workspace_fastapi_app.dependency_overrides[get_request_user] = _user
    workspace_fastapi_app.dependency_overrides[get_chacha_db_for_user] = _db
    workspace_fastapi_app.dependency_overrides[workspaces_endpoint.try_get_workspace_job_manager] = lambda: job_manager
    workspace_fastapi_app.dependency_overrides[WORKSPACES_READ_RATE_LIMIT] = _allow_rate_limit
    workspace_fastapi_app.dependency_overrides[WORKSPACES_WRITE_RATE_LIMIT] = _allow_rate_limit
    workspace_fastapi_app.dependency_overrides[WORKSPACES_DELETE_RATE_LIMIT] = _allow_rate_limit
    try:
        db.upsert_workspace("ws-src-api", "Workspace Sources")

        with TestClient(workspace_fastapi_app, raise_server_exceptions=False) as client:
            add_response = client.post(
                "/api/v1/workspaces/ws-src-api/sources",
                json={
                    "id": "src-1",
                    "media_id": 1,
                    "title": "Video Source",
                    "source_type": "video",
                },
            )
            assert add_response.status_code == 201, add_response.text
            added = add_response.json()
            assert added["id"] == "src-1"
            assert added["selected"] is True

            duplicate_add_response = client.post(
                "/api/v1/workspaces/ws-src-api/sources",
                json={
                    "id": "src-1",
                    "media_id": 1,
                    "title": "Video Source",
                    "source_type": "video",
                },
            )
            assert duplicate_add_response.status_code == 201, duplicate_add_response.text
            duplicate_added = duplicate_add_response.json()
            assert duplicate_added["id"] == "src-1"
            assert duplicate_added["media_id"] == 1

            list_response = client.get("/api/v1/workspaces/ws-src-api/sources")
            assert list_response.status_code == 200, list_response.text
            assert [item["id"] for item in list_response.json()] == ["src-1"]

            second_add_response = client.post(
                "/api/v1/workspaces/ws-src-api/sources",
                json={
                    "id": "src-2",
                    "media_id": 2,
                    "title": "Article Source",
                    "source_type": "article",
                },
            )
            assert second_add_response.status_code == 201, second_add_response.text

            selection_response = client.put(
                "/api/v1/workspaces/ws-src-api/sources/selection",
                json={"selected_ids": ["src-2"]},
            )
            assert selection_response.status_code == 200, selection_response.text

            reorder_response = client.put(
                "/api/v1/workspaces/ws-src-api/sources/reorder",
                json={"ordered_ids": ["src-2", "src-1"]},
            )
            assert reorder_response.status_code == 200, reorder_response.text

            reordered_list_response = client.get("/api/v1/workspaces/ws-src-api/sources")
            assert reordered_list_response.status_code == 200, reordered_list_response.text
            reordered_sources = reordered_list_response.json()
            assert [item["id"] for item in reordered_sources] == ["src-2", "src-1"]
            assert reordered_sources[0]["selected"] is True
            assert reordered_sources[1]["selected"] is False

            delete_response = client.delete("/api/v1/workspaces/ws-src-api/sources/src-1")
            assert delete_response.status_code == 204, delete_response.text

            final_list_response = client.get("/api/v1/workspaces/ws-src-api/sources")
            assert final_list_response.status_code == 200, final_list_response.text
            assert [item["id"] for item in final_list_response.json()] == ["src-2"]
    finally:
        workspace_fastapi_app.dependency_overrides.pop(get_request_user, None)
        workspace_fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(workspaces_endpoint.try_get_workspace_job_manager, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_READ_RATE_LIMIT, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_WRITE_RATE_LIMIT, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_DELETE_RATE_LIMIT, None)


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
    finally:
        workspace_fastapi_app.dependency_overrides.pop(get_request_user, None)
        workspace_fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(workspaces_endpoint.try_get_workspace_job_manager, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_WRITE_RATE_LIMIT, None)

    assert response.status_code == 201, response.text
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
def test_add_workspace_source_preserves_row_when_job_manager_dependency_fails(workspace_fastapi_app, db):
    async def _allow_rate_limit() -> None:
        return None

    def _raise_job_manager() -> None:
        raise RuntimeError("jobs manager construction failed")

    db.upsert_workspace("ws-job-dep-fail-api", "Workspace Source Job Dependency Failure")
    workspace_fastapi_app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=9)
    workspace_fastapi_app.dependency_overrides[get_chacha_db_for_user] = lambda: db
    workspace_fastapi_app.dependency_overrides[get_job_manager] = _raise_job_manager
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
        workspace_fastapi_app.dependency_overrides.pop(get_job_manager, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_WRITE_RATE_LIMIT, None)

    assert response.status_code == 201, response.text
    assert response.json()["id"] == "src-job-dep-fail"
    assert [src["id"] for src in db.list_workspace_sources("ws-job-dep-fail-api")] == ["src-job-dep-fail"]


@pytest.mark.integration
def test_list_workspace_sources_maps_database_error_to_contextual_500(workspace_fastapi_app):
    class _DatabaseErrorDB:
        def get_workspace(self, workspace_id: str):
            return {"id": workspace_id, "version": 1}

        def list_workspace_sources(self, workspace_id: str):
            _ = workspace_id
            raise CharactersRAGDBError("sqlite backend unavailable")

    async def _allow_rate_limit() -> None:
        return None

    workspace_fastapi_app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)
    workspace_fastapi_app.dependency_overrides[get_chacha_db_for_user] = lambda: _DatabaseErrorDB()
    workspace_fastapi_app.dependency_overrides[WORKSPACES_READ_RATE_LIMIT] = _allow_rate_limit
    try:
        with TestClient(workspace_fastapi_app, raise_server_exceptions=False) as client:
            response = client.get("/api/v1/workspaces/ws-1/sources")
    finally:
        workspace_fastapi_app.dependency_overrides.pop(get_request_user, None)
        workspace_fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_READ_RATE_LIMIT, None)

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to fetch workspace sources"


@pytest.mark.integration
def test_add_workspace_source_maps_input_error_to_400(workspace_fastapi_app):
    class _InputErrorDB:
        def get_workspace(self, workspace_id: str):
            return {"id": workspace_id, "version": 1}

        def add_workspace_source(self, workspace_id: str, data: dict):
            _ = (workspace_id, data)
            raise InputError("invalid workspace source create")

    async def _allow_rate_limit() -> None:
        return None

    workspace_fastapi_app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)
    workspace_fastapi_app.dependency_overrides[get_chacha_db_for_user] = lambda: _InputErrorDB()
    workspace_fastapi_app.dependency_overrides[WORKSPACES_WRITE_RATE_LIMIT] = _allow_rate_limit
    try:
        with TestClient(workspace_fastapi_app, raise_server_exceptions=False) as client:
            response = client.post(
                "/api/v1/workspaces/ws-1/sources",
                json={"id": "src-1", "media_id": 1, "title": "Video", "source_type": "video"},
            )
    finally:
        workspace_fastapi_app.dependency_overrides.pop(get_request_user, None)
        workspace_fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_WRITE_RATE_LIMIT, None)

    assert response.status_code == 400, response.text
    assert response.json()["detail"] == "invalid workspace source create"


@pytest.mark.integration
def test_delete_workspace_source_maps_database_error_to_contextual_500(workspace_fastapi_app):
    class _DatabaseErrorDB:
        def get_workspace(self, workspace_id: str):
            return {"id": workspace_id, "version": 1}

        def delete_workspace_source(self, workspace_id: str, source_id: str):
            _ = (workspace_id, source_id)
            raise CharactersRAGDBError("sqlite backend unavailable")

    async def _allow_rate_limit() -> None:
        return None

    workspace_fastapi_app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)
    workspace_fastapi_app.dependency_overrides[get_chacha_db_for_user] = lambda: _DatabaseErrorDB()
    workspace_fastapi_app.dependency_overrides[WORKSPACES_DELETE_RATE_LIMIT] = _allow_rate_limit
    try:
        with TestClient(workspace_fastapi_app, raise_server_exceptions=False) as client:
            response = client.delete("/api/v1/workspaces/ws-1/sources/src-1")
    finally:
        workspace_fastapi_app.dependency_overrides.pop(get_request_user, None)
        workspace_fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_DELETE_RATE_LIMIT, None)

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to delete workspace source"


@pytest.mark.integration
def test_update_workspace_source_selection_maps_input_error_to_400(workspace_fastapi_app):
    class _InputErrorDB:
        def get_workspace(self, workspace_id: str):
            return {"id": workspace_id, "version": 1}

        def update_workspace_source_selection(self, workspace_id: str, *, selected_ids: list[str]):
            _ = (workspace_id, selected_ids)
            raise InputError("invalid workspace source selection")

    async def _allow_rate_limit() -> None:
        return None

    workspace_fastapi_app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)
    workspace_fastapi_app.dependency_overrides[get_chacha_db_for_user] = lambda: _InputErrorDB()
    workspace_fastapi_app.dependency_overrides[WORKSPACES_WRITE_RATE_LIMIT] = _allow_rate_limit
    try:
        with TestClient(workspace_fastapi_app, raise_server_exceptions=False) as client:
            response = client.put(
                "/api/v1/workspaces/ws-1/sources/selection",
                json={"selected_ids": ["src-1"]},
            )
    finally:
        workspace_fastapi_app.dependency_overrides.pop(get_request_user, None)
        workspace_fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_WRITE_RATE_LIMIT, None)

    assert response.status_code == 400, response.text
    assert response.json()["detail"] == "invalid workspace source selection"


@pytest.mark.integration
def test_reorder_workspace_sources_maps_database_error_to_contextual_500(workspace_fastapi_app):
    class _DatabaseErrorDB:
        def get_workspace(self, workspace_id: str):
            return {"id": workspace_id, "version": 1}

        def reorder_workspace_sources(self, workspace_id: str, ordered_ids: list[str]):
            _ = (workspace_id, ordered_ids)
            raise CharactersRAGDBError("sqlite backend unavailable")

    async def _allow_rate_limit() -> None:
        return None

    workspace_fastapi_app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)
    workspace_fastapi_app.dependency_overrides[get_chacha_db_for_user] = lambda: _DatabaseErrorDB()
    workspace_fastapi_app.dependency_overrides[WORKSPACES_WRITE_RATE_LIMIT] = _allow_rate_limit
    try:
        with TestClient(workspace_fastapi_app, raise_server_exceptions=False) as client:
            response = client.put(
                "/api/v1/workspaces/ws-1/sources/reorder",
                json={"ordered_ids": ["src-2", "src-1"]},
            )
    finally:
        workspace_fastapi_app.dependency_overrides.pop(get_request_user, None)
        workspace_fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_WRITE_RATE_LIMIT, None)

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to reorder workspace sources"
