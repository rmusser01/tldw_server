from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.Explainer_DB_Deps import get_explainer_db
from tldw_Server_API.app.api.v1.API_Deps.jobs_deps import get_job_manager
from tldw_Server_API.app.api.v1.endpoints.explainer import router as explainer_router
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.DB_Management.Explainer_DB import ExplainerDatabase
from tldw_Server_API.app.core.Explainer.jobs import (
    EXPLAINER_DOMAIN,
    EXPLAINER_JOB_TYPE,
    EXPLAINER_QUEUE,
)
from tldw_Server_API.app.core.Explainer.repository import ExplainerRepository

pytestmark = pytest.mark.integration


class FakeJobManager:
    def __init__(self) -> None:
        self.jobs: dict[int, dict] = {}
        self.next_id = 700

    def create_job(self, **kwargs):
        job = {
            "id": self.next_id,
            "uuid": f"job-{self.next_id}",
            "status": "queued",
            "progress_percent": None,
            "progress_message": None,
            "result": None,
            "last_error": None,
            **kwargs,
        }
        self.jobs[self.next_id] = job
        self.next_id += 1
        return job

    def get_job(self, job_id: int):
        return self.jobs.get(int(job_id))


@pytest.fixture()
def explainer_client(tmp_path):
    app = FastAPI()
    app.include_router(explainer_router, prefix="/api/v1", tags=["explainer"])
    db = ExplainerDatabase(tmp_path / "Explainer.db", client_id="test")
    current_user_id = {"value": 7}
    fake_job_manager = FakeJobManager()

    async def _override_user():
        user_id = current_user_id["value"]
        return User(id=user_id, username=f"user-{user_id}", email=None, is_active=True, is_admin=True)

    async def _override_db():
        return db

    def _override_job_manager():
        return fake_job_manager

    app.dependency_overrides[get_request_user] = _override_user
    app.dependency_overrides[get_explainer_db] = _override_db
    app.dependency_overrides[get_job_manager] = _override_job_manager
    app.state.fake_job_manager = fake_job_manager

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


def test_list_sessions_returns_summary_page_without_nodes_or_citations(explainer_client) -> None:
    client, _db, set_user = explainer_client
    owner_response = client.post(
        "/api/v1/explainer/sessions",
        json=_create_goal_payload(title="Owner session", rootPrompt="Owner topic"),
    )
    assert owner_response.status_code == 201
    owner_session = owner_response.json()
    node_response = client.post(
        f"/api/v1/explainer/sessions/{owner_session['id']}/nodes",
        json={
            "parentId": owner_session["rootNodeIds"][0],
            "title": "Owner detail",
            "citations": [
                {
                    "sourceId": "media-42",
                    "sourceType": "media",
                    "title": "Attention notes",
                    "excerpt": "This excerpt must not appear in list responses.",
                }
            ],
        },
    )
    assert node_response.status_code == 201
    second_response = client.post(
        "/api/v1/explainer/sessions",
        json=_create_goal_payload(title="Second owner session", rootPrompt="Second topic"),
    )
    assert second_response.status_code == 201

    set_user(8)
    other_response = client.post(
        "/api/v1/explainer/sessions",
        json=_create_goal_payload(title="Other user session", rootPrompt="Other topic"),
    )
    assert other_response.status_code == 201

    set_user(7)
    response = client.get("/api/v1/explainer/sessions?limit=1&offset=0")

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 2
    assert len(body["items"]) == 1
    assert body["items"][0]["ownerUserId"] == "7"
    assert "nodes" not in body["items"][0]
    assert "rootNodeIds" not in body["items"][0]
    assert "citations" not in str(body)
    assert "This excerpt must not appear in list responses." not in str(body)


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


def test_create_node_accepts_and_persists_citation_snapshots(explainer_client) -> None:
    client, db, _set_user = explainer_client
    created = client.post(
        "/api/v1/explainer/sessions",
        json=_create_goal_payload(
            grounding="source_led",
            selectedSources=[
                {
                    "sourceId": "media-42",
                    "sourceType": "media",
                    "title": "Attention paper notes",
                }
            ],
        ),
    )
    assert created.status_code == 201
    session_body = created.json()

    response = client.post(
        f"/api/v1/explainer/sessions/{session_body['id']}/nodes",
        json={
            "parentId": session_body["rootNodeIds"][0],
            "title": "Scaled dot product attention",
            "body": "Attention compares every token against every other token.",
            "citations": [
                {
                    "sourceId": "media-42",
                    "sourceType": "media",
                    "title": "Attention paper notes",
                    "excerpt": "Attention weights are computed from query-key similarity.",
                    "locationLabel": "chunk 3",
                    "startOffset": 120,
                    "endOffset": 178,
                    "url": "https://example.test/attention",
                    "snapshotHash": "sha256:abc123",
                }
            ],
        },
    )

    assert response.status_code == 201
    body = response.json()
    assert body["citations"][0]["sourceId"] == "media-42"
    assert body["citations"][0]["excerpt"] == "Attention weights are computed from query-key similarity."

    loaded = ExplainerRepository(db).get_session(session_body["id"], owner_user_id="7")
    assert loaded is not None
    assert loaded.nodes[body["id"]].citations[0].snapshot_hash == "sha256:abc123"


def test_patch_node_replaces_citation_snapshots(explainer_client) -> None:
    client, db, _set_user = explainer_client
    created = client.post(
        "/api/v1/explainer/sessions",
        json=_create_goal_payload(
            grounding="source_led",
            selectedSources=[
                {
                    "sourceId": "media-42",
                    "sourceType": "media",
                    "title": "Attention paper notes",
                }
            ],
        ),
    )
    assert created.status_code == 201
    session_body = created.json()
    node_response = client.post(
        f"/api/v1/explainer/sessions/{session_body['id']}/nodes",
        json={
            "parentId": session_body["rootNodeIds"][0],
            "title": "Scaled dot product attention",
            "citations": [
                {
                    "sourceId": "media-42",
                    "sourceType": "media",
                    "title": "Attention paper notes",
                    "excerpt": "Initial citation.",
                }
            ],
        },
    )
    assert node_response.status_code == 201
    node_id = node_response.json()["id"]

    response = client.patch(
        f"/api/v1/explainer/sessions/{session_body['id']}/nodes/{node_id}",
        json={
            "citations": [
                {
                    "sourceId": "media-99",
                    "sourceType": "note",
                    "title": "Updated note",
                    "excerpt": "Replacement citation.",
                    "locationLabel": "paragraph 2",
                }
            ]
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["citations"][0]["sourceId"] == "media-99"
    assert body["citations"][0]["excerpt"] == "Replacement citation."

    loaded = ExplainerRepository(db).get_session(session_body["id"], owner_user_id="7")
    assert loaded is not None
    assert loaded.nodes[node_id].citations[0].location_label == "paragraph 2"


def test_patch_node_can_clear_nullable_fields_with_json_null(explainer_client) -> None:
    client, _db, _set_user = explainer_client
    created = client.post("/api/v1/explainer/sessions", json=_create_goal_payload())
    assert created.status_code == 201
    session_body = created.json()
    node_response = client.post(
        f"/api/v1/explainer/sessions/{session_body['id']}/nodes",
        json={
            "parentId": session_body["rootNodeIds"][0],
            "title": "Scaled dot product attention",
            "body": "Body to clear",
        },
    )
    assert node_response.status_code == 201
    node_id = node_response.json()["id"]
    populated = client.patch(
        f"/api/v1/explainer/sessions/{session_body['id']}/nodes/{node_id}",
        json={
            "selectedCustomAnswer": "Answer to clear",
            "generationMetadata": {"model": "test"},
        },
    )
    assert populated.status_code == 200
    assert populated.json()["body"] == "Body to clear"
    assert populated.json()["selectedCustomAnswer"] == "Answer to clear"
    assert populated.json()["generationMetadata"] == {"model": "test"}

    response = client.patch(
        f"/api/v1/explainer/sessions/{session_body['id']}/nodes/{node_id}",
        json={
            "body": None,
            "selectedCustomAnswer": None,
            "generationMetadata": None,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["body"] is None
    assert body["selectedCustomAnswer"] is None
    assert body["generationMetadata"] is None


def test_delete_node_removes_descendant_subtree_and_citations(explainer_client) -> None:
    client, _db, _set_user = explainer_client
    created = client.post(
        "/api/v1/explainer/sessions",
        json=_create_goal_payload(
            grounding="source_led",
            selectedSources=[
                {
                    "sourceId": "media-42",
                    "sourceType": "media",
                    "title": "Attention paper notes",
                }
            ],
        ),
    )
    assert created.status_code == 201
    session_body = created.json()
    parent_response = client.post(
        f"/api/v1/explainer/sessions/{session_body['id']}/nodes",
        json={
            "parentId": session_body["rootNodeIds"][0],
            "title": "Parent branch",
            "citations": [
                {
                    "sourceId": "media-42",
                    "sourceType": "media",
                    "title": "Attention paper notes",
                    "excerpt": "Parent citation.",
                }
            ],
        },
    )
    assert parent_response.status_code == 201
    parent_id = parent_response.json()["id"]
    child_response = client.post(
        f"/api/v1/explainer/sessions/{session_body['id']}/nodes",
        json={
            "parentId": parent_id,
            "title": "Child branch",
            "citations": [
                {
                    "sourceId": "media-42",
                    "sourceType": "media",
                    "title": "Attention paper notes",
                    "excerpt": "Child citation.",
                }
            ],
        },
    )
    assert child_response.status_code == 201
    child_id = child_response.json()["id"]
    grandchild_response = client.post(
        f"/api/v1/explainer/sessions/{session_body['id']}/nodes",
        json={
            "parentId": child_id,
            "title": "Grandchild branch",
            "citations": [
                {
                    "sourceId": "media-42",
                    "sourceType": "media",
                    "title": "Attention paper notes",
                    "excerpt": "Grandchild citation.",
                }
            ],
        },
    )
    assert grandchild_response.status_code == 201
    grandchild_id = grandchild_response.json()["id"]

    delete_response = client.delete(
        f"/api/v1/explainer/sessions/{session_body['id']}/nodes/{parent_id}",
    )
    assert delete_response.status_code == 200
    loaded = client.get(f"/api/v1/explainer/sessions/{session_body['id']}")

    assert loaded.status_code == 200
    body = loaded.json()
    assert parent_id not in body["nodes"]
    assert child_id not in body["nodes"]
    assert grandchild_id not in body["nodes"]
    assert "Parent citation." not in str(body)
    assert "Child citation." not in str(body)
    assert "Grandchild citation." not in str(body)


def test_expand_endpoint_returns_queued_job_and_marks_node(explainer_client) -> None:
    client, db, _set_user = explainer_client
    created = client.post("/api/v1/explainer/sessions", json=_create_goal_payload(outputIntent="both"))
    assert created.status_code == 201
    session_body = created.json()
    node_id = session_body["rootNodeIds"][0]

    response = client.post(
        f"/api/v1/explainer/sessions/{session_body['id']}/nodes/{node_id}/expand",
        json={"intent": "both"},
    )

    assert response.status_code == 202
    body = response.json()
    assert body["jobId"]
    assert body["sessionId"] == session_body["id"]
    assert body["nodeId"] == node_id
    assert body["status"] == "queued"
    loaded = ExplainerRepository(db).get_session(session_body["id"], owner_user_id="7")
    assert loaded is not None
    assert loaded.nodes[node_id].status == "queued"
    fake_job_manager = client.app.state.fake_job_manager
    [job] = fake_job_manager.jobs.values()
    assert job["domain"] == EXPLAINER_DOMAIN
    assert job["queue"] == EXPLAINER_QUEUE
    assert job["job_type"] == EXPLAINER_JOB_TYPE
    assert job["owner_user_id"] == "7"


def test_answer_question_endpoint_persists_selected_answer(explainer_client) -> None:
    client, db, _set_user = explainer_client
    created = client.post("/api/v1/explainer/sessions", json=_create_goal_payload())
    assert created.status_code == 201
    session_body = created.json()
    node_id = session_body["rootNodeIds"][0]

    response = client.post(
        f"/api/v1/explainer/sessions/{session_body['id']}/nodes/{node_id}/answer-question",
        json={"selectedOptionId": "option-1", "selectedCustomAnswer": "Use practical examples"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["selectedOptionId"] == "option-1"
    assert body["selectedCustomAnswer"] == "Use practical examples"
    loaded = ExplainerRepository(db).get_session(session_body["id"], owner_user_id="7")
    assert loaded is not None
    assert loaded.nodes[node_id].selected_option_id == "option-1"
    assert loaded.nodes[node_id].selected_custom_answer == "Use practical examples"


def test_job_status_returns_404_for_cross_user_job(explainer_client) -> None:
    client, _db, set_user = explainer_client
    fake_job_manager = client.app.state.fake_job_manager
    job = fake_job_manager.create_job(
        domain=EXPLAINER_DOMAIN,
        queue=EXPLAINER_QUEUE,
        job_type=EXPLAINER_JOB_TYPE,
        payload={"session_id": "session-1", "node_id": "node-1", "intent": "explain"},
        owner_user_id="7",
        priority=5,
        idempotency_key="explainer:session-1:node-1:explain:rev",
    )

    set_user(8)
    response = client.get(f"/api/v1/explainer/jobs/{job['id']}")

    assert response.status_code == 404
