"""Workspace defaults are resolved once at server conversation creation."""

from __future__ import annotations

import importlib

import pytest

from tldw_Server_API.app.api.v1.schemas.chat_session_schemas import ChatSessionCreate
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

pytestmark = pytest.mark.unit


@pytest.fixture
def db(tmp_path):
    database = CharactersRAGDB(tmp_path / "defaults.sqlite", "1")
    character_id = database.add_character_card({"name": "Default source", "system_prompt": "Help."})
    database.create_persona_profile(
        {
            "id": "workspace-persona",
            "user_id": "1",
            "name": "Workspace Persona",
            "character_card_id": character_id,
            "mode": "session_scoped",
            "is_active": True,
        }
    )
    workspace = database.upsert_workspace("ws", "Workspace")
    database.update_workspace(
        "ws",
        {
            "assistant_defaults_json": {
                "assistant_kind": "persona",
                "assistant_id": "workspace-persona",
                "persona_memory_mode": "read_only",
            }
        },
        expected_version=workspace["version"],
    )
    yield database
    database.close_connection()


def _resolve(db, payload):
    module = importlib.import_module("tldw_Server_API.app.core.Workspaces.assistant_defaults")
    return module.resolve_new_conversation_assistant(db, user_id="1", request=ChatSessionCreate.model_validate(payload))


def test_omitted_choice_inherits_default_without_mutating_existing_conversations(db):
    existing_id = db.add_conversation(
        {"title": "Existing", "client_id": "1", "scope_type": "workspace", "workspace_id": "ws"}
    )
    result = _resolve(db, {"scope_type": "workspace", "workspace_id": "ws"})
    assert result.assistant_kind == "persona"
    assert result.assistant_id == "workspace-persona"
    assert result.persona_memory_mode == "read_only"
    assert db.get_conversation_by_id(existing_id)["assistant_id"] is None


@pytest.mark.parametrize(
    "choice",
    [
        {"assistant_kind": None},
        {"assistant_id": None},
        {"character_id": None},
        {"assistant_kind": "persona", "assistant_id": "explicit-persona"},
    ],
)
def test_explicit_choice_including_none_wins_over_workspace_default(db, choice):
    result = _resolve(db, {"scope_type": "workspace", "workspace_id": "ws", **choice})
    assert result.assistant_id == choice.get("assistant_id")


def test_global_and_fork_requests_do_not_inherit_workspace_default(db):
    assert _resolve(db, {}).assistant_id is None
    assert (
        _resolve(
            db, {"scope_type": "workspace", "workspace_id": "ws", "parent_conversation_id": "existing"}
        ).assistant_id
        is None
    )


def test_deleted_default_persona_fails_closed(db):
    profile = db.get_persona_profile("workspace-persona", user_id="1")
    db.soft_delete_persona_profile(persona_id="workspace-persona", user_id="1", expected_version=profile["version"])
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as error:
        _resolve(db, {"scope_type": "workspace", "workspace_id": "ws"})
    assert error.value.status_code == 409


def test_missing_workspace_fails_before_default_lookup(db):
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as error:
        _resolve(db, {"scope_type": "workspace", "workspace_id": "missing"})
    assert error.value.status_code == 404


def test_create_chat_endpoint_persists_inherited_and_explicit_none_choices(db, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_request_user
    from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
    from tldw_Server_API.app.api.v1.endpoints import character_chat_sessions as endpoint

    app = FastAPI()
    app.include_router(endpoint.router, prefix="/api/v1/chats")
    app.dependency_overrides[get_chacha_db_for_user] = lambda: db
    app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)
    app.dependency_overrides[endpoint.require_expected_user] = lambda: None
    monkeypatch.setattr(
        endpoint,
        "get_character_rate_limiter",
        lambda: SimpleNamespace(check_rate_limit=AsyncMock(), check_chat_limit=AsyncMock()),
    )
    monkeypatch.setattr(endpoint, "_active_chat_sync_service", lambda *args: None)
    with TestClient(app) as client:
        inherited = client.post("/api/v1/chats/", json={"scope_type": "workspace", "workspace_id": "ws"})
        assert inherited.status_code == 201, inherited.text
        row = db.get_conversation_by_id(inherited.json()["id"])
        assert row["assistant_kind"] == "persona"
        assert row["assistant_id"] == "workspace-persona"
        assert row["persona_memory_mode"] == "read_only"
        explicit = client.post(
            "/api/v1/chats/", json={"scope_type": "workspace", "workspace_id": "ws", "assistant_kind": None}
        )
        assert explicit.status_code == 201, explicit.text
        assert db.get_conversation_by_id(explicit.json()["id"])["assistant_id"] is None
