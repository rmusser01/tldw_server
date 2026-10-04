"""Scope and ownership regressions for lorebook diagnostics export."""

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.endpoints import character_chat_sessions
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "stored_scope,owner_id,params,expected_status",
    [
        pytest.param("global", "user-1", {}, 200, id="global-default"),
        pytest.param("global", "user-1", {"scope_type": "global"}, 200, id="global-explicit"),
        pytest.param("global", "user-1", {"workspace_id": "ws-1"}, 200, id="global-stray-workspace"),
        pytest.param("workspace", "user-1", {}, 404, id="workspace-no-scope"),
        pytest.param("workspace", "user-1", {"scope_type": "global"}, 404, id="workspace-global"),
        pytest.param("workspace", "user-1", {"workspace_id": "ws-1"}, 404, id="workspace-id-only"),
        pytest.param(
            "workspace",
            "user-1",
            {"scope_type": "workspace", "workspace_id": "ws-1"},
            200,
            id="workspace-matching",
        ),
        pytest.param(
            "workspace",
            "user-1",
            {"scope_type": "workspace", "workspace_id": "ws-2"},
            404,
            id="workspace-wrong-id",
        ),
        pytest.param(
            "global",
            "user-1",
            {"scope_type": "workspace", "workspace_id": "ws-1"},
            404,
            id="global-workspace-scope",
        ),
        pytest.param("global", "other-user", {}, 403, id="global-foreign-owner"),
        pytest.param(
            "workspace",
            "other-user",
            {"scope_type": "workspace", "workspace_id": "ws-1"},
            403,
            id="workspace-foreign-owner",
        ),
        pytest.param("workspace", "user-1", {"scope_type": "workspace"}, 400, id="missing-workspace-id"),
        pytest.param(
            "workspace",
            "user-1",
            {"scope_type": "workspace", "workspace_id": ""},
            400,
            id="empty-workspace-id",
        ),
        pytest.param("global", "user-1", {"scope_type": "invalid"}, 422, id="invalid-scope-type"),
    ],
)
def test_lorebook_diagnostics_enforces_requested_scope_and_owner(
    tmp_path,
    stored_scope,
    owner_id,
    params,
    expected_status,
):
    """Only the owning principal's matching scope may expose stored diagnostics."""
    db = CharactersRAGDB(db_path=str(tmp_path / "chacha.db"), client_id="user-1")
    try:
        db.upsert_workspace("ws-1", "Workspace One")
        character_id = db.add_character_card({"name": "Scope Character", "client_id": "user-1"})
        chat_id = db.add_conversation(
            {
                "id": "diagnostics-chat",
                "character_id": character_id,
                "client_id": owner_id,
                "scope_type": stored_scope,
                "workspace_id": "ws-1" if stored_scope == "workspace" else None,
            }
        )
        message_id = db.add_message(
            {
                "id": "diagnostics-message",
                "conversation_id": chat_id,
                "sender": "assistant",
                "content": "Workspace assistant turn",
                "client_id": owner_id,
            }
        )
        assert db.add_message_metadata(
            message_id,
            extra={"lorebook_diagnostics": [{"entry_id": 99, "keyword": "alpha"}]},
        )

        app = FastAPI()
        app.include_router(character_chat_sessions.router, prefix="/api/v1/chats")
        app.dependency_overrides[get_chacha_db_for_user] = lambda: db
        app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id="user-1")
        with TestClient(app) as client:
            response = client.get(
                f"/api/v1/chats/{chat_id}/diagnostics/lorebook",
                params=params,
            )

        assert response.status_code == expected_status, response.text
        if expected_status == 200:
            data = response.json()
            assert data["chat_id"] == "diagnostics-chat"
            assert data["total_turns_with_diagnostics"] == 1
            assert data["turns"][0]["message_id"] == "diagnostics-message"
            assert data["turns"][0]["diagnostics"] == [{"entry_id": 99, "keyword": "alpha"}]
        else:
            assert "turns" not in response.json()
    finally:
        db.close_all_connections()
