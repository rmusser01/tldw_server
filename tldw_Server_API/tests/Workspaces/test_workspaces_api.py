"""Tests for workspace CRUD endpoints and scoped chat session isolation."""
import base64
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import Response

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import try_get_media_db_for_user
from tldw_Server_API.app.api.v1.endpoints.workspaces_rate_limit_policy import (
    WORKSPACES_DELETE_RATE_LIMIT,
    WORKSPACES_READ_RATE_LIMIT,
    WORKSPACES_WRITE_RATE_LIMIT,
)
from tldw_Server_API.app.api.v1.endpoints import workspaces as workspaces_endpoint
from tldw_Server_API.app.api.v1.API_Deps.jobs_deps import get_job_manager
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    ConflictError,
    InputError,
)
from tldw_Server_API.app.core.Sandbox.store import IdempotencyConflict, InMemoryStore
from tldw_Server_API.app.core.Sandbox.workspace_volumes import SandboxWorkspaceVolumeService
from tldw_Server_API.app.core.Workspaces import root_binding_service


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


class _ConflictingSandboxVolumeService:
    def provision_workspace_volume(self, **kwargs: Any) -> None:
        _ = kwargs
        raise IdempotencyConflict("volume-previous", key="workspace-root:previous")


@pytest.fixture
def db(tmp_path: Path) -> CharactersRAGDB:
    d = CharactersRAGDB(db_path=str(tmp_path / "chacha.db"), client_id="user-1")
    d.add_character_card({"name": "Test Char"})
    return d


@pytest.fixture
def workspace_fastapi_app() -> FastAPI:
    app = FastAPI()
    app.include_router(workspaces_endpoint.router, prefix="/api/v1/workspaces")
    return app


def _get_workspace_roots_response(
    workspace_fastapi_app: FastAPI,
    db_like: Any,
    workspace_id: str = "ws-root",
) -> Response:
    async def _allow_rate_limit() -> None:
        return None

    workspace_fastapi_app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)
    workspace_fastapi_app.dependency_overrides[get_chacha_db_for_user] = lambda: db_like
    workspace_fastapi_app.dependency_overrides[WORKSPACES_READ_RATE_LIMIT] = _allow_rate_limit
    try:
        with TestClient(workspace_fastapi_app, raise_server_exceptions=False) as client:
            return client.get(f"/api/v1/workspaces/{workspace_id}/roots")
    finally:
        workspace_fastapi_app.dependency_overrides.pop(get_request_user, None)
        workspace_fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_READ_RATE_LIMIT, None)


def _get_workspace_capabilities_response(
    workspace_fastapi_app: FastAPI,
    db_like: Any,
    workspace_id: str = "ws-root",
) -> Response:
    async def _allow_rate_limit() -> None:
        return None

    workspace_fastapi_app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)
    workspace_fastapi_app.dependency_overrides[get_chacha_db_for_user] = lambda: db_like
    workspace_fastapi_app.dependency_overrides[try_get_media_db_for_user] = lambda: None
    workspace_fastapi_app.dependency_overrides[
        workspaces_endpoint.try_get_workspace_job_manager
    ] = lambda: None
    workspace_fastapi_app.dependency_overrides[WORKSPACES_READ_RATE_LIMIT] = _allow_rate_limit
    try:
        with TestClient(workspace_fastapi_app, raise_server_exceptions=False) as client:
            return client.get(f"/api/v1/workspaces/{workspace_id}/capabilities")
    finally:
        workspace_fastapi_app.dependency_overrides.pop(get_request_user, None)
        workspace_fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(try_get_media_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(
            workspaces_endpoint.try_get_workspace_job_manager,
            None,
        )
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_READ_RATE_LIMIT, None)


def _get_workspace_context_response(
    workspace_fastapi_app: FastAPI,
    db_like: Any,
    workspace_id: str = "ws-root",
) -> Response:
    async def _allow_rate_limit() -> None:
        return None

    workspace_fastapi_app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)
    workspace_fastapi_app.dependency_overrides[get_chacha_db_for_user] = lambda: db_like
    workspace_fastapi_app.dependency_overrides[try_get_media_db_for_user] = lambda: None
    workspace_fastapi_app.dependency_overrides[
        workspaces_endpoint.try_get_workspace_job_manager
    ] = lambda: None
    workspace_fastapi_app.dependency_overrides[WORKSPACES_READ_RATE_LIMIT] = _allow_rate_limit
    try:
        with TestClient(workspace_fastapi_app, raise_server_exceptions=False) as client:
            return client.get(f"/api/v1/workspaces/{workspace_id}/context")
    finally:
        workspace_fastapi_app.dependency_overrides.pop(get_request_user, None)
        workspace_fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(try_get_media_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(
            workspaces_endpoint.try_get_workspace_job_manager,
            None,
        )
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_READ_RATE_LIMIT, None)


def _put_workspace_primary_root_response(
    workspace_fastapi_app: FastAPI,
    db_like: Any,
    payload: dict[str, Any],
    workspace_id: str = "ws-root",
) -> Response:
    async def _allow_rate_limit() -> None:
        return None

    workspace_fastapi_app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)
    workspace_fastapi_app.dependency_overrides[get_chacha_db_for_user] = lambda: db_like
    workspace_fastapi_app.dependency_overrides[WORKSPACES_WRITE_RATE_LIMIT] = _allow_rate_limit
    try:
        with TestClient(workspace_fastapi_app, raise_server_exceptions=False) as client:
            return client.put(
                f"/api/v1/workspaces/{workspace_id}/roots/primary",
                json=payload,
            )
    finally:
        workspace_fastapi_app.dependency_overrides.pop(get_request_user, None)
        workspace_fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_WRITE_RATE_LIMIT, None)


def _post_workspace_sandbox_root_response(
    workspace_fastapi_app: FastAPI,
    db_like: Any,
    payload: dict[str, Any],
    workspace_id: str = "ws-root",
    idempotency_key: str | None = "root-key",
    sandbox_service: Any | None = None,
) -> Response:
    async def _allow_rate_limit() -> None:
        return None

    sandbox_service = sandbox_service or SandboxWorkspaceVolumeService(store=InMemoryStore())
    workspace_fastapi_app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)
    workspace_fastapi_app.dependency_overrides[get_chacha_db_for_user] = lambda: db_like
    workspace_fastapi_app.dependency_overrides[
        workspaces_endpoint.get_workspace_sandbox_volume_service
    ] = lambda: sandbox_service
    workspace_fastapi_app.dependency_overrides[WORKSPACES_WRITE_RATE_LIMIT] = _allow_rate_limit
    headers = {"Idempotency-Key": idempotency_key} if idempotency_key is not None else {}
    try:
        with TestClient(workspace_fastapi_app, raise_server_exceptions=False) as client:
            return client.post(
                f"/api/v1/workspaces/{workspace_id}/roots/primary/sandbox-volume",
                json=payload,
                headers=headers,
            )
    finally:
        workspace_fastapi_app.dependency_overrides.pop(get_request_user, None)
        workspace_fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(
            workspaces_endpoint.get_workspace_sandbox_volume_service,
            None,
        )
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_WRITE_RATE_LIMIT, None)


def _get_workspace_operation_response(
    workspace_fastapi_app: FastAPI,
    db_like: Any,
    workspace_id: str,
    operation_id: str,
) -> Response:
    async def _allow_rate_limit() -> None:
        return None

    workspace_fastapi_app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)
    workspace_fastapi_app.dependency_overrides[get_chacha_db_for_user] = lambda: db_like
    workspace_fastapi_app.dependency_overrides[WORKSPACES_READ_RATE_LIMIT] = _allow_rate_limit
    try:
        with TestClient(workspace_fastapi_app, raise_server_exceptions=False) as client:
            return client.get(f"/api/v1/workspaces/{workspace_id}/operations/{operation_id}")
    finally:
        workspace_fastapi_app.dependency_overrides.pop(get_request_user, None)
        workspace_fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_READ_RATE_LIMIT, None)


def test_root_path_hint_redacts_relative_path_segments() -> None:
    assert workspaces_endpoint._root_path_hint({"path_hint": "client/acme/repo"}) == "repo"
    assert workspaces_endpoint._root_path_hint({"display_name": "client\\acme\\repo"}) == "repo"


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
    from tldw_Server_API.app.api.v1.endpoints.workspaces_rate_limit_policy import (
        WORKSPACES_READ_RATE_LIMIT,
        WORKSPACES_WRITE_RATE_LIMIT,
    )
    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user

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
                json={
                    "name": "API Workspace",
                    "study_materials_policy": "workspace",
                    "workspace_profile": "project",
                },
            )
            assert create_response.status_code == 200, create_response.text
            created = create_response.json()
            assert created["study_materials_policy"] == "workspace"
            assert created["workspace_profile"] == "project"

            upsert_response = client.put(
                "/api/v1/workspaces/ws-api",
                json={
                    "name": "API Workspace Renamed",
                    "study_materials_policy": "general",
                },
            )
            assert upsert_response.status_code == 200, upsert_response.text
            upserted = upsert_response.json()
            assert upserted["name"] == "API Workspace Renamed"
            assert upserted["study_materials_policy"] == "general"
            assert upserted["workspace_profile"] == "project"

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
def test_workspace_api_patches_assistant_defaults(workspace_fastapi_app, db):
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

    workspace = db.upsert_workspace("ws-assistant", "Assistant Defaults")
    workspace_fastapi_app.dependency_overrides[get_request_user] = _user
    workspace_fastapi_app.dependency_overrides[get_chacha_db_for_user] = lambda: db
    workspace_fastapi_app.dependency_overrides[WORKSPACES_WRITE_RATE_LIMIT] = _allow_rate_limit
    try:
        with TestClient(workspace_fastapi_app, raise_server_exceptions=False) as client:
            response = client.patch(
                "/api/v1/workspaces/ws-assistant",
                json={
                    "version": workspace["version"],
                    "assistant_defaults": {
                        "assistant_kind": "persona",
                        "assistant_id": "persona-1",
                        "persona_memory_mode": "read_only",
                    },
                },
            )
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["assistant_defaults"]["assistant_kind"] == "persona"
        assert payload["assistant_defaults"]["assistant_id"] == "persona-1"
        assert payload["assistant_defaults"]["persona_memory_mode"] == "read_only"
        assert payload["assistant_defaults"]["voice"] is None
        assert payload["assistant_defaults"]["style"] is None
        assert payload["assistant_defaults"]["tool_policy_profile_id"] is None
        persisted = db.get_workspace("ws-assistant")
        assert persisted is not None
        assert persisted["assistant_defaults_json"] == {
            "assistant_kind": "persona",
            "assistant_id": "persona-1",
            "persona_memory_mode": "read_only",
        }
    finally:
        workspace_fastapi_app.dependency_overrides.pop(get_request_user, None)
        workspace_fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_WRITE_RATE_LIMIT, None)


@pytest.mark.integration
def test_workspace_api_requires_confirmation_for_read_write_assistant_default(workspace_fastapi_app, db):
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

    workspace = db.upsert_workspace("ws-assistant", "Assistant Defaults")
    workspace_fastapi_app.dependency_overrides[get_request_user] = _user
    workspace_fastapi_app.dependency_overrides[get_chacha_db_for_user] = lambda: db
    workspace_fastapi_app.dependency_overrides[WORKSPACES_WRITE_RATE_LIMIT] = _allow_rate_limit
    try:
        with TestClient(workspace_fastapi_app, raise_server_exceptions=False) as client:
            missing_confirmation = client.patch(
                "/api/v1/workspaces/ws-assistant",
                json={
                    "version": workspace["version"],
                    "assistant_defaults": {
                        "assistant_kind": "persona",
                        "assistant_id": "persona-1",
                        "persona_memory_mode": "read_write",
                    },
                },
            )
            assert missing_confirmation.status_code == 422, missing_confirmation.text

            confirmed = client.patch(
                "/api/v1/workspaces/ws-assistant",
                json={
                    "version": workspace["version"],
                    "assistant_defaults": {
                        "assistant_kind": "persona",
                        "assistant_id": "persona-1",
                        "persona_memory_mode": "read_write",
                    },
                    "confirm_read_write_assistant_default": True,
                },
            )
        assert confirmed.status_code == 200, confirmed.text
        assert confirmed.json()["assistant_defaults"]["persona_memory_mode"] == "read_write"
    finally:
        workspace_fastapi_app.dependency_overrides.pop(get_request_user, None)
        workspace_fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_WRITE_RATE_LIMIT, None)


@pytest.mark.integration
def test_workspace_api_rejects_confirmation_only_patch(workspace_fastapi_app, db):
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

    workspace = db.upsert_workspace("ws-assistant", "Assistant Defaults")
    workspace_fastapi_app.dependency_overrides[get_request_user] = _user
    workspace_fastapi_app.dependency_overrides[get_chacha_db_for_user] = lambda: db
    workspace_fastapi_app.dependency_overrides[WORKSPACES_WRITE_RATE_LIMIT] = _allow_rate_limit
    try:
        with TestClient(workspace_fastapi_app, raise_server_exceptions=False) as client:
            response = client.patch(
                "/api/v1/workspaces/ws-assistant",
                json={
                    "version": workspace["version"],
                    "confirm_read_write_assistant_default": True,
                },
            )
        assert response.status_code == 422, response.text
    finally:
        workspace_fastapi_app.dependency_overrides.pop(get_request_user, None)
        workspace_fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)
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
def test_workspace_roots_endpoint_returns_primary_root_contract(workspace_fastapi_app, db):
    async def _allow_rate_limit() -> None:
        return None

    db.upsert_workspace("ws-root", "Rooted Workspace")
    db.upsert_workspace_primary_root(
        "ws-root",
        {
            "root_id": "root-1",
            "backend": "host_local",
            "display_name": "Local root",
            "absolute_root": "/Users/example/project",
            "root_state": "attached",
            "indexing_state": "ready",
        },
    )

    workspace_fastapi_app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)
    workspace_fastapi_app.dependency_overrides[get_chacha_db_for_user] = lambda: db
    workspace_fastapi_app.dependency_overrides[WORKSPACES_READ_RATE_LIMIT] = _allow_rate_limit
    try:
        with TestClient(workspace_fastapi_app, raise_server_exceptions=False) as client:
            response = client.get("/api/v1/workspaces/ws-root/roots")
    finally:
        workspace_fastapi_app.dependency_overrides.pop(get_request_user, None)
        workspace_fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_READ_RATE_LIMIT, None)

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["workspace_id"] == "ws-root"
    assert payload["workspace_profile"] == "project"
    assert payload["primary_root"]["root_id"] == "root-1"
    assert payload["primary_root"]["backend"] == "host_local"
    assert payload["primary_root"]["state"] == "attached"
    assert payload["primary_root"]["path_hint"] == "Local root"
    assert "absolute_root" not in payload["primary_root"]
    assert [root["root_id"] for root in payload["roots"]] == ["root-1"]


@pytest.mark.integration
def test_list_workspace_roots_maps_database_error_to_contextual_500(workspace_fastapi_app):
    class _DatabaseErrorDB:
        def get_workspace(self, workspace_id: str):
            return {"id": workspace_id, "workspace_profile": "project"}

        def list_workspace_project_roots(self, workspace_id: str):
            _ = workspace_id
            raise CharactersRAGDBError("sqlite backend unavailable")

    async def _allow_rate_limit() -> None:
        return None

    workspace_fastapi_app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)
    workspace_fastapi_app.dependency_overrides[get_chacha_db_for_user] = lambda: _DatabaseErrorDB()
    workspace_fastapi_app.dependency_overrides[WORKSPACES_READ_RATE_LIMIT] = _allow_rate_limit
    try:
        with TestClient(workspace_fastapi_app, raise_server_exceptions=False) as client:
            response = client.get("/api/v1/workspaces/ws-root/roots")
    finally:
        workspace_fastapi_app.dependency_overrides.pop(get_request_user, None)
        workspace_fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_READ_RATE_LIMIT, None)

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to fetch workspace roots"


@pytest.mark.integration
def test_list_workspace_roots_maps_workspace_lookup_error_to_contextual_500(workspace_fastapi_app):
    class _DatabaseErrorDB:
        def get_workspace(self, workspace_id: str):
            _ = workspace_id
            raise CharactersRAGDBError("sqlite backend unavailable")

        def list_workspace_project_roots(self, workspace_id: str):
            _ = workspace_id
            pytest.fail("roots should not be listed when workspace lookup fails")

    response = _get_workspace_roots_response(workspace_fastapi_app, _DatabaseErrorDB())

    assert response.status_code == 500, response.text
    assert response.json()["detail"] == "Failed to fetch workspace roots"


@pytest.mark.integration
def test_workspace_roots_endpoint_fails_closed_for_unknown_root_state_and_backend(workspace_fastapi_app):
    class _InvalidRootDB:
        def get_workspace(self, workspace_id: str):
            return {"id": workspace_id, "workspace_profile": "project"}

        def list_workspace_project_roots(self, workspace_id: str):
            return [
                {
                    "workspace_id": workspace_id,
                    "root_id": "root-1",
                    "backend": "legacy_backend",
                    "root_state": "ready",
                    "display_name": "Legacy root",
                    "is_primary": True,
                    "version": 1,
                }
            ]

    response = _get_workspace_roots_response(workspace_fastapi_app, _InvalidRootDB())

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["primary_root"]["state"] == "failed"
    assert payload["primary_root"]["backend"] is None
    assert payload["roots"][0]["state"] == "failed"
    assert payload["roots"][0]["backend"] is None


@pytest.mark.integration
@pytest.mark.parametrize(
    ("absolute_root", "expected_hint"),
    [
        ("/Users/example/project", "project"),
        (r"C:\Users\example\project", "project"),
        ("C:\\", "project_root"),
        (r"\\server\share\project", "project"),
        (r"\\server\share", "project_root"),
        (r"\Users\example\project", "project"),
        ("relative/secret/project", "project"),
    ],
)
def test_workspace_roots_endpoint_redacts_absolute_root_fallback(
    workspace_fastapi_app,
    absolute_root,
    expected_hint,
):
    class _RootPathDB:
        def get_workspace(self, workspace_id: str):
            return {"id": workspace_id, "workspace_profile": "project"}

        def list_workspace_project_roots(self, workspace_id: str):
            return [
                {
                    "workspace_id": workspace_id,
                    "root_id": "root-1",
                    "backend": "host_local",
                    "root_state": "attached",
                    "absolute_root": absolute_root,
                    "is_primary": True,
                    "version": 1,
                }
            ]

    response = _get_workspace_roots_response(workspace_fastapi_app, _RootPathDB())

    assert response.status_code == 200, response.text
    root = response.json()["primary_root"]
    assert root["path_hint"] == expected_hint
    assert "absolute_root" not in root


@pytest.mark.integration
def test_attach_workspace_primary_host_local_root_returns_redacted_roots_response(
    workspace_fastapi_app,
    db,
    tmp_path,
    monkeypatch,
):
    allowed = tmp_path / "allowed"
    project = allowed / "project"
    project.mkdir(parents=True)
    db.upsert_workspace("ws-root", "Rooted Workspace")
    monkeypatch.setattr(
        root_binding_service.config,
        "get_workspace_project_root_allowed_roots",
        lambda: (allowed,),
        raising=True,
    )

    response = _put_workspace_primary_root_response(
        workspace_fastapi_app,
        db,
        {"backend": "host_local", "absolute_root": str(project)},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["workspace_id"] == "ws-root"
    assert payload["workspace_profile"] == "project"
    assert payload["primary_root"]["root_id"] == "primary"
    assert payload["primary_root"]["backend"] == "host_local"
    assert payload["primary_root"]["path_hint"] == "project"
    assert "absolute_root" not in payload["primary_root"]


@pytest.mark.integration
def test_attach_workspace_primary_host_local_root_outside_allowed_returns_service_error(
    workspace_fastapi_app,
    db,
    tmp_path,
    monkeypatch,
):
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside" / "project"
    allowed.mkdir()
    outside.mkdir(parents=True)
    db.upsert_workspace("ws-root", "Rooted Workspace")
    monkeypatch.setattr(
        root_binding_service.config,
        "get_workspace_project_root_allowed_roots",
        lambda: (allowed,),
        raising=True,
    )

    response = _put_workspace_primary_root_response(
        workspace_fastapi_app,
        db,
        {"backend": "host_local", "absolute_root": str(outside)},
    )

    assert response.status_code == 403, response.text
    assert response.json()["detail"]["code"] == "workspace_project_root_outside_allowed_roots"


@pytest.mark.integration
def test_attach_workspace_primary_host_local_root_without_configured_roots_returns_503(
    workspace_fastapi_app,
    db,
    tmp_path,
    monkeypatch,
):
    project = tmp_path / "project"
    project.mkdir()
    db.upsert_workspace("ws-root", "Rooted Workspace")
    monkeypatch.setattr(
        root_binding_service.config,
        "get_workspace_project_root_allowed_roots",
        lambda: (),
        raising=True,
    )

    response = _put_workspace_primary_root_response(
        workspace_fastapi_app,
        db,
        {"backend": "host_local", "absolute_root": str(project)},
    )

    assert response.status_code == 503, response.text
    assert response.json()["detail"]["code"] == "workspace_project_roots_not_configured"


@pytest.mark.integration
def test_attach_workspace_primary_different_root_without_replace_returns_409(
    workspace_fastapi_app,
    db,
    tmp_path,
    monkeypatch,
):
    allowed = tmp_path / "allowed"
    original = allowed / "original"
    replacement = allowed / "replacement"
    original.mkdir(parents=True)
    replacement.mkdir()
    db.upsert_workspace("ws-root", "Rooted Workspace")
    db.upsert_workspace_primary_root(
        "ws-root",
        {
            "root_id": "primary",
            "backend": "host_local",
            "absolute_root": str(original.resolve()),
            "root_state": "attached",
            "is_primary": True,
        },
    )
    monkeypatch.setattr(
        root_binding_service.config,
        "get_workspace_project_root_allowed_roots",
        lambda: (allowed,),
        raising=True,
    )

    response = _put_workspace_primary_root_response(
        workspace_fastapi_app,
        db,
        {"backend": "host_local", "absolute_root": str(replacement)},
    )

    assert response.status_code == 409, response.text
    assert response.json()["detail"]["code"] == "workspace_primary_root_exists"


@pytest.mark.integration
def test_attach_workspace_primary_replacement_with_replace_existing_returns_new_primary(
    workspace_fastapi_app,
    db,
    tmp_path,
    monkeypatch,
):
    allowed = tmp_path / "allowed"
    original = allowed / "original"
    replacement = allowed / "replacement"
    original.mkdir(parents=True)
    replacement.mkdir()
    db.upsert_workspace("ws-root", "Rooted Workspace")
    db.upsert_workspace_primary_root(
        "ws-root",
        {
            "root_id": "primary",
            "backend": "host_local",
            "absolute_root": str(original.resolve()),
            "root_state": "attached",
            "is_primary": True,
        },
    )
    monkeypatch.setattr(
        root_binding_service.config,
        "get_workspace_project_root_allowed_roots",
        lambda: (allowed,),
        raising=True,
    )

    response = _put_workspace_primary_root_response(
        workspace_fastapi_app,
        db,
        {
            "backend": "host_local",
            "absolute_root": str(replacement),
            "replace_existing": True,
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["primary_root"]["root_id"] == "primary"
    assert payload["primary_root"]["backend"] == "host_local"
    assert payload["primary_root"]["path_hint"] == "replacement"
    assert "absolute_root" not in payload["primary_root"]


@pytest.mark.integration
def test_attach_workspace_primary_sandbox_volume_returns_not_configured_mount_state(
    workspace_fastapi_app,
    db,
):
    db.upsert_workspace("ws-root", "Rooted Workspace")

    response = _put_workspace_primary_root_response(
        workspace_fastapi_app,
        db,
        {
            "backend": "sandbox_volume",
            "sandbox_volume_id": "volume-123",
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["workspace_profile"] == "project"
    assert payload["primary_root"]["root_id"] == "primary"
    assert payload["primary_root"]["backend"] == "sandbox_volume"
    assert payload["primary_root"]["path_hint"] == "volume-123"
    assert payload["primary_root"]["sandbox_mount_state"] == "not_configured"


@pytest.mark.integration
def test_provision_workspace_sandbox_root_requires_idempotency_key(workspace_fastapi_app, db):
    db.upsert_workspace("ws-root", "Rooted Workspace")

    response = _post_workspace_sandbox_root_response(
        workspace_fastapi_app,
        db,
        {"display_name": "Project root", "requested_runtime": "docker"},
        idempotency_key=None,
    )

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "workspace_idempotency_key_required"


@pytest.mark.integration
def test_provision_workspace_sandbox_root_returns_active_operation_and_pollable_status(
    workspace_fastapi_app,
    db,
):
    db.upsert_workspace("ws-root", "Rooted Workspace")
    sandbox_service = SandboxWorkspaceVolumeService(store=InMemoryStore())

    response = _post_workspace_sandbox_root_response(
        workspace_fastapi_app,
        db,
        {"display_name": "Project root", "requested_runtime": "docker"},
        idempotency_key="root-key",
        sandbox_service=sandbox_service,
    )

    assert response.status_code == 202, response.text
    payload = response.json()
    assert payload["workspace_profile"] == "project"
    assert payload["primary_root"]["backend"] == "sandbox_volume"
    assert payload["primary_root"]["sandbox_mount_state"] == "not_configured"
    assert payload["operation"]["status"] == "running"
    assert payload["operation"]["retryable"] is True
    operation_id = payload["operation"]["operation_id"]

    retry = _post_workspace_sandbox_root_response(
        workspace_fastapi_app,
        db,
        {"display_name": "Project root", "requested_runtime": "docker"},
        idempotency_key="root-key",
        sandbox_service=sandbox_service,
    )
    assert retry.status_code == 202, retry.text
    assert retry.json()["operation"]["operation_id"] == operation_id

    status_response = _get_workspace_operation_response(
        workspace_fastapi_app,
        db,
        "ws-root",
        operation_id,
    )
    assert status_response.status_code == 200, status_response.text
    assert status_response.json()["operation_id"] == operation_id

    context_response = _get_workspace_context_response(workspace_fastapi_app, db, "ws-root")
    assert context_response.status_code == 200, context_response.text
    active_operations = context_response.json()["active_operations"]
    assert [operation["operation_id"] for operation in active_operations] == [operation_id]


@pytest.mark.integration
def test_provision_workspace_sandbox_root_conflicts_for_changed_idempotent_request(
    workspace_fastapi_app,
    db,
):
    db.upsert_workspace("ws-root", "Rooted Workspace")
    sandbox_service = SandboxWorkspaceVolumeService(store=InMemoryStore())
    first = _post_workspace_sandbox_root_response(
        workspace_fastapi_app,
        db,
        {"display_name": "Project root", "requested_runtime": "docker"},
        idempotency_key="root-key",
        sandbox_service=sandbox_service,
    )
    assert first.status_code == 202, first.text

    changed = _post_workspace_sandbox_root_response(
        workspace_fastapi_app,
        db,
        {"display_name": "Project root", "requested_runtime": "vz_linux"},
        idempotency_key="root-key",
        sandbox_service=sandbox_service,
    )

    assert changed.status_code == 409, changed.text


@pytest.mark.integration
def test_provision_workspace_sandbox_root_maps_volume_idempotency_conflict_to_409(
    workspace_fastapi_app: FastAPI,
    db: CharactersRAGDB,
) -> None:
    db.upsert_workspace("ws-root", "Rooted Workspace")

    response = _post_workspace_sandbox_root_response(
        workspace_fastapi_app,
        db,
        {"display_name": "Project root", "requested_runtime": "docker"},
        idempotency_key="root-key",
        sandbox_service=_ConflictingSandboxVolumeService(),
    )

    assert response.status_code == 409, response.text
    operation = db.get_workspace_operation_by_idempotency(
        workspace_id="ws-root",
        user_id="1",
        command="provision_sandbox_root",
        idempotency_key="root-key",
    )
    assert operation is not None
    assert operation["status"] == "conflicted"
    assert operation["diagnostics"]["code"] == "workspace_sandbox_volume_idempotency_conflict"


@pytest.mark.integration
def test_attached_host_local_primary_root_is_redacted_across_read_contracts(
    workspace_fastapi_app,
    db,
    tmp_path,
    monkeypatch,
):
    secret_parent = tmp_path / "tenant-secret-123" / "user-token-456"
    allowed = secret_parent / "allowed-project-roots"
    project = allowed / "public-project"
    project.mkdir(parents=True)
    db.upsert_workspace("ws-root", "Rooted Workspace")
    monkeypatch.setattr(
        root_binding_service.config,
        "get_workspace_project_root_allowed_roots",
        lambda: (allowed,),
        raising=True,
    )

    attach_response = _put_workspace_primary_root_response(
        workspace_fastapi_app,
        db,
        {"backend": "host_local", "absolute_root": str(project)},
    )
    assert attach_response.status_code == 200, attach_response.text

    roots_response = _get_workspace_roots_response(workspace_fastapi_app, db)
    capabilities_response = _get_workspace_capabilities_response(workspace_fastapi_app, db)
    context_response = _get_workspace_context_response(workspace_fastapi_app, db)

    assert roots_response.status_code == 200, roots_response.text
    assert capabilities_response.status_code == 200, capabilities_response.text
    assert context_response.status_code == 200, context_response.text

    roots = roots_response.json()
    capabilities = capabilities_response.json()
    context = context_response.json()
    assert roots["workspace_profile"] == "project"
    assert capabilities["workspace_profile"] == "project"
    assert context["workspace_profile"] == "project"
    assert capabilities["workspace_kind"] == "project_workspace"
    assert context["workspace_kind"] == "project_workspace"

    roots_primary = roots["primary_root"]
    capabilities_root = capabilities["project_root"]
    context_root = context["project_root"]
    for root in (roots_primary, capabilities_root, context_root):
        assert root["root_id"] == "primary"
        assert root["backend"] == "host_local"
        assert root["path_hint"] == project.name
        assert root["file_inventory"]["available"] is True
        assert "absolute_root" not in root

    context_capability_root = context["capabilities"]["project_root"]
    assert context_capability_root["root_id"] == "primary"
    assert context_capability_root["backend"] == "host_local"
    assert context_capability_root["path_hint"] == project.name
    assert context_capability_root["file_inventory"]["available"] is True

    serialized_payloads = [
        json.dumps(payload, sort_keys=True)
        for payload in (roots, capabilities, context)
    ]
    forbidden_values = [
        str(project),
        str(allowed),
        str(secret_parent),
        "tenant-secret-123",
        "user-token-456",
        allowed.name,
    ]
    for serialized in serialized_payloads:
        assert project.name in serialized
        for forbidden_value in forbidden_values:
            assert forbidden_value not in serialized


@pytest.mark.integration
def test_workspace_capabilities_and_context_include_file_inventory_summary(
    workspace_fastapi_app,
    db,
    tmp_path,
    monkeypatch,
):
    allowed = tmp_path / "allowed-project-roots"
    project = allowed / "public-project"
    project.mkdir(parents=True)
    db.upsert_workspace("ws-root", "Rooted Workspace")
    monkeypatch.setattr(
        root_binding_service.config,
        "get_workspace_project_root_allowed_roots",
        lambda: (allowed,),
        raising=True,
    )

    attach_response = _put_workspace_primary_root_response(
        workspace_fastapi_app,
        db,
        {"backend": "host_local", "absolute_root": str(project)},
    )
    assert attach_response.status_code == 200, attach_response.text
    root = db.get_workspace_primary_root("ws-root")
    assert root is not None
    scan = db.begin_workspace_file_inventory_scan(
        "ws-root",
        str(root["root_id"]),
        int(root["version"]),
        "policy-fingerprint",
        requested_by="test-user",
    )
    db.complete_workspace_file_inventory_scan(
        str(scan["scan_id"]),
        "current",
        {
            "files": 7,
            "directories": 2,
            "symlinks": 1,
            "ignored": 3,
            "indexing_candidates": 5,
            "diagnostics": 0,
            "total_entries": 10,
        },
        [],
        root_snapshot_token="snapshot-1",
    )

    capabilities_response = _get_workspace_capabilities_response(workspace_fastapi_app, db)
    context_response = _get_workspace_context_response(workspace_fastapi_app, db)

    assert capabilities_response.status_code == 200, capabilities_response.text
    assert context_response.status_code == 200, context_response.text
    capabilities = capabilities_response.json()
    context = context_response.json()
    for payload in (capabilities, context, context["capabilities"]):
        inventory = payload["project_root"]["file_inventory"]
        assert inventory["state"] == "current"
        assert inventory["total_file_count"] == 7
        assert inventory["indexed_file_count"] == 0
        assert isinstance(inventory["updated_at"], str)
        assert inventory["available"] is True
        assert payload["allowed_actions"]["scan_files"] == {
            "allowed": True,
            "reason_code": None,
        }
        assert payload["allowed_actions"]["view_file_inventory"] == {
            "allowed": True,
            "reason_code": None,
        }
        assert payload["allowed_actions"]["index_file_content"] == {
            "allowed": False,
            "reason_code": "file_indexing_disabled",
        }


@pytest.mark.integration
def test_sandbox_volume_primary_root_fails_closed_across_read_contracts(
    workspace_fastapi_app,
    db,
):
    volume_id = "volume-123"
    db.upsert_workspace("ws-root", "Rooted Workspace")

    attach_response = _put_workspace_primary_root_response(
        workspace_fastapi_app,
        db,
        {
            "backend": "sandbox_volume",
            "sandbox_volume_id": volume_id,
        },
    )
    assert attach_response.status_code == 200, attach_response.text

    roots_response = _get_workspace_roots_response(workspace_fastapi_app, db)
    capabilities_response = _get_workspace_capabilities_response(workspace_fastapi_app, db)
    context_response = _get_workspace_context_response(workspace_fastapi_app, db)

    assert roots_response.status_code == 200, roots_response.text
    assert capabilities_response.status_code == 200, capabilities_response.text
    assert context_response.status_code == 200, context_response.text

    roots = roots_response.json()
    capabilities = capabilities_response.json()
    context = context_response.json()
    assert roots["primary_root"]["backend"] == "sandbox_volume"
    assert roots["primary_root"]["path_hint"] == volume_id
    assert roots["primary_root"]["sandbox_mount_state"] == "not_configured"
    assert capabilities["project_root"]["sandbox_mount_state"] == "not_configured"
    assert context["project_root"]["sandbox_mount_state"] == "not_configured"
    assert context["capabilities"]["project_root"]["sandbox_mount_state"] == "not_configured"
    assert context["project_root"]["file_inventory"]["available"] is False
    assert context["attention_state"] == "needs_attention"
    assert context["active_operations"] == []

    for payload in (capabilities, context):
        assert payload["workspace_kind"] == "project_workspace"
        assert payload["project_root"]["backend"] == "sandbox_volume"
        assert payload["project_root"]["path_hint"] == volume_id
        for action_name in (
            "write_files",
            "run_sandbox",
            "use_sandbox",
            "use_acp_agents",
        ):
            action = payload["allowed_actions"][action_name]
            assert action["allowed"] is False
            assert action["reason_code"] == "sandbox_mount_not_configured"


@pytest.mark.integration
def test_workspace_context_manager_defaults_for_research_workspace(
    workspace_fastapi_app,
    db,
):
    db.upsert_workspace("ws-root", "Research Workspace", workspace_profile="research")

    context_response = _get_workspace_context_response(workspace_fastapi_app, db)

    assert context_response.status_code == 200, context_response.text
    context = context_response.json()
    assert context["workspace_profile"] == "research"
    assert context["attention_state"] == "ready"
    assert context["project_root"]["state"] == "not_configured"
    assert context["project_root"]["file_inventory"]["available"] is False
    assert context["active_operations"] == []


@pytest.mark.integration
def test_workspace_context_project_shell_without_root_is_setup_pending(
    workspace_fastapi_app,
    db,
):
    db.upsert_workspace("ws-root", "Project Workspace", workspace_profile="project")

    context_response = _get_workspace_context_response(workspace_fastapi_app, db)

    assert context_response.status_code == 200, context_response.text
    context = context_response.json()
    assert context["workspace_profile"] == "project"
    assert context["attention_state"] == "setup_pending"
    assert context["project_root"]["state"] == "not_configured"
    assert context["project_root"]["file_inventory"]["available"] is False
    assert context["active_operations"] == []


@pytest.mark.integration
def test_workspace_context_project_inventory_scan_is_working(
    workspace_fastapi_app,
    db,
    tmp_path,
    monkeypatch,
):
    allowed = tmp_path / "allowed-project-roots"
    project = allowed / "active-inventory"
    project.mkdir(parents=True)
    db.upsert_workspace("ws-root", "Project Workspace", workspace_profile="project")
    monkeypatch.setattr(
        root_binding_service.config,
        "get_workspace_project_root_allowed_roots",
        lambda: (allowed,),
        raising=True,
    )
    attach_response = _put_workspace_primary_root_response(
        workspace_fastapi_app,
        db,
        {"backend": "host_local", "absolute_root": str(project)},
    )
    assert attach_response.status_code == 200, attach_response.text
    root = db.get_workspace_primary_root("ws-root")
    assert root is not None
    db.begin_workspace_file_inventory_scan(
        "ws-root",
        str(root["root_id"]),
        int(root["version"]),
        "policy-fingerprint",
        requested_by="test-user",
    )

    context_response = _get_workspace_context_response(workspace_fastapi_app, db)

    assert context_response.status_code == 200, context_response.text
    context = context_response.json()
    assert context["project_root"]["file_inventory"]["state"] == "queued"
    assert context["project_root"]["file_inventory"]["available"] is True
    assert context["attention_state"] == "working"
    assert context["active_operations"] == []


@pytest.mark.integration
def test_workspace_context_archived_workspace_attention_state(
    workspace_fastapi_app,
    db,
):
    workspace = db.upsert_workspace("ws-root", "Archived Project", workspace_profile="project")
    db.update_workspace("ws-root", {"archived": True}, expected_version=int(workspace["version"]))

    context_response = _get_workspace_context_response(workspace_fastapi_app, db)

    assert context_response.status_code == 200, context_response.text
    context = context_response.json()
    assert context["workspace"]["archived"] is True
    assert context["attention_state"] == "archived"
    assert context["active_operations"] == []


@pytest.mark.integration
def test_attach_workspace_primary_stale_expected_workspace_version_returns_409(
    workspace_fastapi_app,
    db,
    tmp_path,
    monkeypatch,
):
    allowed = tmp_path / "allowed"
    project = allowed / "project"
    project.mkdir(parents=True)
    workspace = db.upsert_workspace("ws-root", "Rooted Workspace")
    monkeypatch.setattr(
        root_binding_service.config,
        "get_workspace_project_root_allowed_roots",
        lambda: (allowed,),
        raising=True,
    )

    response = _put_workspace_primary_root_response(
        workspace_fastapi_app,
        db,
        {
            "backend": "host_local",
            "absolute_root": str(project),
            "expected_workspace_version": workspace["version"] + 1,
        },
    )

    assert response.status_code == 409, response.text
    assert response.json()["detail"]["code"] == "workspace_version_mismatch"


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
        def upsert_workspace(
            self,
            workspace_id: str,
            name: str,
            *,
            study_materials_policy: str,
            workspace_profile: str,
        ):
            _ = (workspace_id, name, study_materials_policy, workspace_profile)
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
        def upsert_workspace(
            self,
            workspace_id: str,
            name: str,
            *,
            study_materials_policy: str,
            workspace_profile: str,
        ):
            _ = (workspace_id, name, study_materials_policy, workspace_profile)
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


def test_workspace_artifact_response_defaults_null_version_for_version_id():
    from tldw_Server_API.app.api.v1.endpoints.workspaces import _art_to_response

    response = _art_to_response({
        "id": "art-null-version",
        "workspace_id": "ws-1",
        "artifact_type": "summary",
        "title": "Summary",
        "version": None,
        "created_at": "2026-05-15T00:00:00Z",
    })

    assert response.version == 1
    assert response.artifact_version_id == "art-null-version:v1"


def test_workspace_artifact_redaction_schema_requires_typed_posture():
    from pydantic import ValidationError

    from tldw_Server_API.app.api.v1.schemas.workspace_schemas import (
        WorkspaceArtifactCreateRequest,
        WorkspaceArtifactResponse,
    )

    with pytest.raises(ValidationError):
        WorkspaceArtifactCreateRequest(
            id="brief-1",
            artifact_type="workspace_brief",
            title="Brief",
            redaction={"support_safe": "yes", "redacted": False},
        )

    schema = WorkspaceArtifactResponse.model_json_schema()
    redaction_ref = schema["properties"]["redaction"]["$ref"]
    redaction_schema = schema["$defs"][redaction_ref.rsplit("/", 1)[-1]]
    assert redaction_schema["properties"]["support_safe"]["type"] == "boolean"
    assert redaction_schema["properties"]["redacted"]["type"] == "boolean"


@pytest.mark.integration
def test_workspace_artifact_export_accepted_version_preserves_identity_and_refs(workspace_fastapi_app, db):
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
        db.upsert_workspace("ws-export-api", "Workspace Exports")
        db.add_workspace_artifact(
            "ws-export-api",
            {
                "id": "brief-1",
                "artifact_type": "workspace_brief",
                "title": "ACP Research Brief --> Review",
                "status": "completed",
                "content": "# Brief\nGrounded <answer>.",
                "content_type": "text/markdown",
                "review_state": "accepted",
                "owner_scope": "workspace",
                "owner_id": "ws-export-api",
                "producer_metadata": {
                    "producer_type": "acp",
                    "producer_id": "task-42",
                    "marker": "json --> comment boundary",
                    "run_id": "run-7",
                    "session_id": "session-abc",
                },
                "source_lineage": {
                    "sources": [
                        {"source_id": "src-1", "source_type": "media", "label": "Transcript"}
                    ]
                },
                "review_metadata": {"decision": "accepted"},
                "version_metadata": {"revision_reason": "initial"},
                "export_refs": [{"format": "legacy", "artifact_version_id": "brief-1:v1"}],
                "redaction": {"support_safe": True, "redacted": False},
            },
        )

        with TestClient(workspace_fastapi_app, raise_server_exceptions=False) as client:
            by_format = {}
            for export_format in ("md", "html", "json"):
                request_json = {"format": export_format}
                if export_format == "json":
                    request_json["artifact_version_id"] = "brief-1:v1"
                response = client.post(
                    "/api/v1/workspaces/ws-export-api/artifacts/brief-1/exports",
                    json=request_json,
                )
                assert response.status_code == 200, response.text
                payload = response.json()
                by_format[export_format] = payload
                assert payload["workspace_id"] == "ws-export-api"
                assert payload["artifact_id"] == "brief-1"
                assert payload["artifact_version_id"] == "brief-1:v1"
                assert payload["review_state"] == "accepted"
                assert payload["format"] == export_format
                assert payload["bytes"] == len(payload["content"].encode("utf-8"))
                assert payload["metadata"]["source_lineage"]["sources"][0]["source_id"] == "src-1"
                assert payload["metadata"]["producer_metadata"]["run_id"] == "run-7"
                assert payload["export_ref"]["artifact_version_id"] == "brief-1:v1"

            md_content = by_format["md"]["content"]
            assert "artifact_id: brief-1" in md_content
            assert "tldw-artifact-metadata-base64:" in md_content
            marker = md_content.split("<!-- tldw-artifact-metadata-base64: ", 1)[1].split(" -->", 1)[0]
            decoded_metadata = json.loads(base64.b64decode(marker).decode("utf-8"))
            assert decoded_metadata["artifact"]["title"] == "ACP Research Brief --> Review"
            assert decoded_metadata["producer_metadata"]["marker"] == "json --> comment boundary"
            assert 'data-artifact-id="brief-1"' in by_format["html"]["content"]
            assert "<h1>Brief</h1>" in by_format["html"]["content"]
            assert "&lt;answer&gt;" in by_format["html"]["content"]
            exported_json = json.loads(by_format["json"]["content"])
            assert exported_json["artifact"]["id"] == "brief-1"
            assert exported_json["metadata"]["source_lineage"]["sources"][0]["source_id"] == "src-1"

            fetch_response = client.get("/api/v1/workspaces/ws-export-api/artifacts")
            assert fetch_response.status_code == 200, fetch_response.text
            exported_artifact = fetch_response.json()[0]
            export_refs = exported_artifact["export_refs"]
            assert export_refs[0]["format"] == "legacy"
            assert [ref["format"] for ref in export_refs[-3:]] == ["md", "html", "json"]
            assert {ref["artifact_version_id"] for ref in export_refs[-3:]} == {"brief-1:v1"}
    finally:
        workspace_fastapi_app.dependency_overrides.pop(get_request_user, None)
        workspace_fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_READ_RATE_LIMIT, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_WRITE_RATE_LIMIT, None)


@pytest.mark.integration
def test_workspace_artifact_export_missing_version_snapshot_fails_loudly(workspace_fastapi_app, db):
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
        db.upsert_workspace("ws-export-api", "Workspace Exports")
        db.add_workspace_artifact(
            "ws-export-api",
            {
                "id": "brief-1",
                "artifact_type": "workspace_brief",
                "title": "ACP Research Brief",
                "content": "# Brief\nGrounded answer.",
                "review_state": "accepted",
                "source_lineage": {"sources": [{"source_id": "src-1"}]},
            },
        )
        with db.transaction() as conn:
            conn.execute(
                "DELETE FROM workspace_artifact_versions "
                "WHERE workspace_id = ? AND artifact_id = ? AND artifact_version_id = ?",
                ("ws-export-api", "brief-1", "brief-1:v1"),
            )

        with TestClient(workspace_fastapi_app, raise_server_exceptions=False) as client:
            response = client.post(
                "/api/v1/workspaces/ws-export-api/artifacts/brief-1/exports",
                json={"format": "md"},
            )
            assert response.status_code == 409, response.text
            assert "missing" in response.json()["detail"].lower()

            fetch_response = client.get("/api/v1/workspaces/ws-export-api/artifacts")
            assert fetch_response.status_code == 200, fetch_response.text
            assert fetch_response.json()[0]["export_refs"] == []
    finally:
        workspace_fastapi_app.dependency_overrides.pop(get_request_user, None)
        workspace_fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_READ_RATE_LIMIT, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_WRITE_RATE_LIMIT, None)


@pytest.mark.integration
def test_workspace_artifact_export_rejects_non_accepted_state(workspace_fastapi_app, db):
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
        db.upsert_workspace("ws-export-api", "Workspace Exports")
        db.add_workspace_artifact(
            "ws-export-api",
            {
                "id": "brief-1",
                "artifact_type": "workspace_brief",
                "title": "ACP Research Brief",
                "content": "# Draft\nNeeds more evidence.",
                "review_state": "needs_revision",
                "source_lineage": {"sources": [{"source_id": "src-1"}]},
            },
        )

        with TestClient(workspace_fastapi_app, raise_server_exceptions=False) as client:
            response = client.post(
                "/api/v1/workspaces/ws-export-api/artifacts/brief-1/exports",
                json={"format": "md"},
            )
            assert response.status_code == 409, response.text
            assert response.json()["detail"] == "workspace_artifact_not_accepted"

            fetch_response = client.get("/api/v1/workspaces/ws-export-api/artifacts")
            assert fetch_response.status_code == 200, fetch_response.text
            assert fetch_response.json()[0]["export_refs"] == []
    finally:
        workspace_fastapi_app.dependency_overrides.pop(get_request_user, None)
        workspace_fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_READ_RATE_LIMIT, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_WRITE_RATE_LIMIT, None)


@pytest.mark.integration
def test_workspace_artifact_api_exposes_traceable_contract_fields(workspace_fastapi_app, db):
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
                    "id": "brief-1",
                    "artifact_type": "workspace_brief",
                    "title": "ACP Research Brief",
                    "status": "completed",
                    "content": "# Brief\nGrounded answer.",
                    "content_type": "text/markdown",
                    "preview_text": "Grounded answer.",
                    "summary": "Executive summary",
                    "review_state": "accepted",
                    "owner_scope": "workspace",
                    "owner_id": "ws-art-api",
                    "producer_metadata": {
                        "producer_type": "acp",
                        "producer_id": "task-42",
                        "run_id": "run-7",
                        "session_id": "session-abc",
                    },
                    "source_lineage": {
                        "sources": [
                            {"source_id": "src-1", "source_type": "media", "label": "Transcript"}
                        ]
                    },
                    "root_artifact_id": "forged-root",
                    "artifact_version_id": "forged:v99",
                    "previous_version_id": "forged:v98",
                    "review_metadata": {"reviewer_id": "reviewer-1", "decision": "accepted"},
                    "version_metadata": {"revision_reason": "initial"},
                    "export_refs": [{"format": "md", "file_id": 101}],
                    "redaction": {"support_safe": True, "redacted": False, "retention_class": "standard"},
                },
            )

            assert add_response.status_code == 201, add_response.text
            added = add_response.json()
            assert added["review_state"] == "accepted"
            assert added["root_artifact_id"] == "brief-1"
            assert added["artifact_version_id"] == "brief-1:v1"
            assert added["producer_metadata"]["producer_type"] == "acp"
            assert added["source_lineage"]["sources"][0]["source_id"] == "src-1"
            assert added["redaction"]["support_safe"] is True

            fetch_response = client.get("/api/v1/workspaces/ws-art-api/artifacts")
            assert fetch_response.status_code == 200, fetch_response.text
            fetched = fetch_response.json()[0]
            assert fetched["artifact_version_id"] == "brief-1:v1"
            assert fetched["review_metadata"]["decision"] == "accepted"
            assert fetched["export_refs"][0]["file_id"] == 101
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
def test_add_workspace_source_enqueues_workspace_ingest_job(workspace_fastapi_app, db):
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
    assert len(job_manager.created_jobs) == 2
    first_job = job_manager.created_jobs[0]
    assert first_job["domain"] == "media_ingest"
    assert first_job["queue"] == "default"
    assert first_job["job_type"] == "workspace_source_ingest"
    assert first_job["owner_user_id"] == "7"
    assert first_job["idempotency_key"] == "workspace-source:ws-job-api:src-job-1:55"
    assert first_job["max_retries"] == 3
    assert first_job["payload"] == {
        "workspace_id": "ws-job-api",
        "workspace_source_id": "src-job-1",
        "source_id": "src-job-1",
        "media_id": 55,
        "source_type": "pdf",
        "title": "NotebookLM Migration PDF",
        "url": "file:///imports/notebooklm.pdf",
        "requested_stages": ["ingestion", "extraction", "chunking", "indexing"],
    }
    assert job_manager.created_jobs[1]["idempotency_key"] == first_job["idempotency_key"]


@pytest.mark.integration
def test_add_workspace_source_ignores_unused_job_manager_override(workspace_fastapi_app, db):
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
def test_add_workspace_source_does_not_construct_job_manager(workspace_fastapi_app, db):
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
