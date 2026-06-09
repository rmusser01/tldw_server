from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.Explainer_DB_Deps import get_explainer_db
from tldw_Server_API.app.api.v1.endpoints.explainer import router as explainer_router
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.Explainer_DB import ExplainerDatabase
from tldw_Server_API.app.core.Explainer.repository import ExplainerRepository

pytestmark = pytest.mark.integration


@pytest.fixture()
def explainer_client(tmp_path):
    app = FastAPI()
    app.include_router(explainer_router, prefix="/api/v1", tags=["explainer"])
    db = ExplainerDatabase(tmp_path / "Explainer.db", client_id="test")
    current_user_id = {"value": 7}

    async def _override_user():
        user_id = current_user_id["value"]
        return User(id=user_id, username=f"user-{user_id}", email=None, is_active=True, is_admin=True)

    async def _override_db():
        return db

    app.dependency_overrides[get_request_user] = _override_user
    app.dependency_overrides[get_explainer_db] = _override_db

    def _set_user(user_id: int) -> None:
        current_user_id["value"] = user_id

    with TestClient(app) as client:
        yield client, db, _set_user

    db.close_connection()
    app.dependency_overrides.clear()


def _create_goal_payload(**overrides):
    payload = {
        "title": "Learn attention",
        "mode": "goal",
        "outputIntent": "explain",
        "grounding": "open",
        "depthPreset": "standard",
        "selectedSources": [],
        "rootPrompt": "Explain transformer attention",
    }
    payload.update(overrides)
    return payload


def test_create_goal_session_persists_session(explainer_client) -> None:
    client, db, _set_user = explainer_client

    response = client.post("/api/v1/explainer/sessions", json=_create_goal_payload())

    assert response.status_code == 201
    body = response.json()
    assert body["title"] == "Learn attention"
    assert body["mode"] == "goal"
    assert body["outputIntent"] == "explain"
    assert body["rootNodeIds"]

    loaded = ExplainerRepository(db).get_session(body["id"], owner_user_id="7")
    assert loaded is not None
    assert loaded.root_node_ids == body["rootNodeIds"]


def test_create_session_rejects_source_only_without_selected_sources(explainer_client) -> None:
    client, _db, _set_user = explainer_client

    response = client.post(
        "/api/v1/explainer/sessions",
        json=_create_goal_payload(grounding="source_only"),
    )

    assert response.status_code in {400, 422}


def test_list_sessions_only_returns_current_user_sessions(explainer_client) -> None:
    client, _db, set_user = explainer_client

    owner_response = client.post(
        "/api/v1/explainer/sessions",
        json=_create_goal_payload(title="Owner session", rootPrompt="Owner topic"),
    )
    assert owner_response.status_code == 201

    set_user(8)
    other_response = client.post(
        "/api/v1/explainer/sessions",
        json=_create_goal_payload(title="Other session", rootPrompt="Other topic"),
    )
    assert other_response.status_code == 201

    set_user(7)
    response = client.get("/api/v1/explainer/sessions")

    assert response.status_code == 200
    titles = [item["title"] for item in response.json()["items"]]
    assert titles == ["Owner session"]


def test_patch_session_updates_output_intent_and_grounding(explainer_client) -> None:
    client, _db, _set_user = explainer_client
    created = client.post("/api/v1/explainer/sessions", json=_create_goal_payload())
    assert created.status_code == 201
    session_id = created.json()["id"]

    response = client.patch(
        f"/api/v1/explainer/sessions/{session_id}",
        json={"outputIntent": "both", "grounding": "source_led"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["outputIntent"] == "both"
    assert body["grounding"] == "source_led"


def test_delete_session_archives_instead_of_hard_deleting(explainer_client) -> None:
    client, db, _set_user = explainer_client
    created = client.post("/api/v1/explainer/sessions", json=_create_goal_payload())
    assert created.status_code == 201
    session_id = created.json()["id"]

    response = client.delete(f"/api/v1/explainer/sessions/{session_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "archived"
    assert body["archivedAt"]

    repo = ExplainerRepository(db)
    assert repo.get_session(session_id, owner_user_id="7") is None
    archived = repo.get_session(session_id, owner_user_id="7", include_archived=True)
    assert archived is not None
    assert archived.status == "archived"
