"""API coverage for Workspace Assistant Defaults effective state."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import (
    get_chacha_db_for_user,
)
from tldw_Server_API.app.api.v1.endpoints import workspaces as workspaces_endpoint
from tldw_Server_API.app.api.v1.endpoints.workspaces_rate_limit_policy import (
    WORKSPACES_READ_RATE_LIMIT,
    WORKSPACES_WRITE_RATE_LIMIT,
)
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


@pytest.fixture
def db(tmp_path: Path) -> CharactersRAGDB:
    database = CharactersRAGDB(db_path=str(tmp_path / "chacha.db"), client_id="1")
    database.add_character_card(
        {
            "name": "Workspace Persona Source",
            "description": "Source character for workspace persona default tests",
            "personality": "Focused",
            "scenario": "Research",
            "system_prompt": "You support workspace research.",
            "first_message": "Ready.",
            "creator_notes": "Test fixture",
        }
    )
    return database


@pytest.fixture
def workspace_app() -> FastAPI:
    app = FastAPI()
    app.include_router(workspaces_endpoint.router, prefix="/api/v1/workspaces")
    return app


def _allow_rate_limit() -> None:
    return None


def _install_workspace_overrides(
    app: FastAPI,
    db: CharactersRAGDB,
    *,
    user_id: int = 1,
    write: bool = False,
) -> None:
    app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=user_id)
    app.dependency_overrides[get_chacha_db_for_user] = lambda: db
    app.dependency_overrides[WORKSPACES_READ_RATE_LIMIT] = _allow_rate_limit
    if write:
        app.dependency_overrides[WORKSPACES_WRITE_RATE_LIMIT] = _allow_rate_limit


def _clear_workspace_overrides(app: FastAPI) -> None:
    app.dependency_overrides.pop(get_request_user, None)
    app.dependency_overrides.pop(get_chacha_db_for_user, None)
    app.dependency_overrides.pop(WORKSPACES_READ_RATE_LIMIT, None)
    app.dependency_overrides.pop(WORKSPACES_WRITE_RATE_LIMIT, None)


def _create_persona(
    db: CharactersRAGDB,
    *,
    persona_id: str = "persona-1",
    user_id: str = "1",
    name: str = "Literature Review Assistant",
) -> str:
    character = db.get_character_card_by_name("Workspace Persona Source")
    assert character is not None  # nosec B101
    return db.create_persona_profile(
        {
            "id": persona_id,
            "user_id": user_id,
            "name": name,
            "character_card_id": int(character["id"]),
            "mode": "session_scoped",
            "system_prompt": "You support literature review workflows.",
            "is_active": True,
        }
    )


def _assistant_defaults_payload(
    persona_id: str = "persona-1",
    memory_mode: str = "read_only",
) -> dict[str, Any]:
    return {
        "assistant_kind": "persona",
        "assistant_id": persona_id,
        "persona_memory_mode": memory_mode,
    }


@pytest.mark.integration
def test_patch_workspace_returns_effective_default_for_existing_persona(
    workspace_app: FastAPI,
    db: CharactersRAGDB,
) -> None:
    workspace = db.upsert_workspace("ws-assistant", "Assistant Defaults")
    persona_id = _create_persona(db)
    _install_workspace_overrides(workspace_app, db, write=True)

    try:
        with TestClient(workspace_app, raise_server_exceptions=False) as client:
            response = client.patch(
                "/api/v1/workspaces/ws-assistant",
                json={
                    "version": workspace["version"],
                    "assistant_defaults": _assistant_defaults_payload(persona_id),
                },
            )
    finally:
        _clear_workspace_overrides(workspace_app)

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["assistant_defaults"] == {
        "assistant_kind": "persona",
        "assistant_id": persona_id,
        "persona_memory_mode": "read_only",
        "voice": None,
        "style": None,
        "tool_policy_profile_id": None,
    }
    assert payload["effective_assistant_default"] == {
        "status": "available",
        "source": "workspace",
        "assistant_kind": "persona",
        "assistant_id": persona_id,
        "label": "Literature Review Assistant",
        "persona_memory_mode": "read_only",
        "degraded_reason": None,
    }


@pytest.mark.integration
def test_get_and_list_workspaces_include_effective_default(
    workspace_app: FastAPI,
    db: CharactersRAGDB,
) -> None:
    workspace = db.upsert_workspace("ws-assistant", "Assistant Defaults")
    persona_id = _create_persona(db)
    db.update_workspace(
        "ws-assistant",
        {"assistant_defaults_json": _assistant_defaults_payload(persona_id)},
        expected_version=int(workspace["version"]),
    )
    _install_workspace_overrides(workspace_app, db)

    try:
        with TestClient(workspace_app, raise_server_exceptions=False) as client:
            get_response = client.get("/api/v1/workspaces/ws-assistant")
            list_response = client.get("/api/v1/workspaces/")
    finally:
        _clear_workspace_overrides(workspace_app)

    assert get_response.status_code == 200, get_response.text
    assert list_response.status_code == 200, list_response.text
    assert get_response.json()["effective_assistant_default"]["status"] == "available"
    [listed] = list_response.json()["items"]
    assert listed["id"] == "ws-assistant"
    assert listed["effective_assistant_default"]["assistant_id"] == persona_id
    assert listed["effective_assistant_default"]["label"] == "Literature Review Assistant"


@pytest.mark.integration
def test_patch_workspace_rejects_missing_persona_default(
    workspace_app: FastAPI,
    db: CharactersRAGDB,
) -> None:
    workspace = db.upsert_workspace("ws-assistant", "Assistant Defaults")
    _install_workspace_overrides(workspace_app, db, write=True)

    try:
        with TestClient(workspace_app, raise_server_exceptions=False) as client:
            response = client.patch(
                "/api/v1/workspaces/ws-assistant",
                json={
                    "version": workspace["version"],
                    "assistant_defaults": _assistant_defaults_payload("missing-persona"),
                },
            )
    finally:
        _clear_workspace_overrides(workspace_app)

    assert response.status_code == 422, response.text
    assert "assistant_defaults.assistant_id" in response.text
    assert db.get_workspace("ws-assistant")["assistant_defaults_json"] is None


@pytest.mark.integration
def test_effective_default_redacts_deleted_persona_drift(
    workspace_app: FastAPI,
    db: CharactersRAGDB,
) -> None:
    workspace = db.upsert_workspace("ws-assistant", "Assistant Defaults")
    persona_id = _create_persona(db)
    db.update_workspace(
        "ws-assistant",
        {"assistant_defaults_json": _assistant_defaults_payload(persona_id)},
        expected_version=int(workspace["version"]),
    )
    profile = db.get_persona_profile(persona_id, user_id="1")
    assert profile is not None  # nosec B101
    assert db.soft_delete_persona_profile(
        persona_id=persona_id,
        user_id="1",
        expected_version=int(profile["version"]),
    )
    _install_workspace_overrides(workspace_app, db)

    try:
        with TestClient(workspace_app, raise_server_exceptions=False) as client:
            response = client.get("/api/v1/workspaces/ws-assistant")
    finally:
        _clear_workspace_overrides(workspace_app)

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["assistant_defaults"]["assistant_id"] == persona_id
    assert payload["effective_assistant_default"] == {
        "status": "unavailable",
        "source": "workspace",
        "assistant_kind": "persona",
        "assistant_id": persona_id,
        "label": None,
        "persona_memory_mode": "read_only",
        "degraded_reason": "persona_deleted",
    }
