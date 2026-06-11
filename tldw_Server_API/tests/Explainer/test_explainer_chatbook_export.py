from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.Explainer_DB_Deps import get_explainer_db
from tldw_Server_API.app.api.v1.endpoints import explainer as explainer_endpoint
from tldw_Server_API.app.api.v1.endpoints.explainer import router as explainer_router
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.Chatbooks.chatbook_models import ContentType
from tldw_Server_API.app.core.DB_Management.Explainer_DB import ExplainerDatabase
from tldw_Server_API.app.core.Explainer.chatbook_adapter import (
    build_explainer_chatbook_payload,
    restore_explainer_chatbook_payload,
)
from tldw_Server_API.app.core.Explainer.repository import ExplainerRepository

pytestmark = pytest.mark.unit


def _assert_text_absent(value: Any, forbidden: str) -> None:
    if isinstance(value, dict):
        for nested in value.values():
            _assert_text_absent(nested, forbidden)
    elif isinstance(value, list):
        for nested in value:
            _assert_text_absent(nested, forbidden)
    elif isinstance(value, str):
        assert forbidden not in value


def _create_complete_session(repo: ExplainerRepository, owner_user_id: str = "7"):
    session = repo.create_session(
        owner_user_id=owner_user_id,
        title="Learn attention",
        mode="goal",
        output_intent="both",
        grounding="source_led",
        depth_preset="standard",
        selected_sources=[
            {
                "source_id": "media-42",
                "source_type": "media",
                "title": "Attention paper notes",
                "snapshot_version": "snapshot-v3",
                "metadata": {
                    "mediaUuid": "uuid-42",
                    "snapshotHash": "sha256:sourcehash",
                    "capturedAt": "2026-06-09T12:00:00+00:00",
                },
            }
        ],
        root_prompt="Explain transformer attention",
    )
    root_id = session.root_node_ids[0]
    repo.update_node(
        session.id,
        root_id,
        owner_user_id=owner_user_id,
        kind="summary",
        intent="plan",
        body="Attention lets tokens route information to each other.",
        status="complete",
        evidence_state="supported",
        question_options=[
            {"id": "math", "label": "Focus on math"},
            {"id": "intuition", "label": "Focus on intuition"},
        ],
        selected_option_id="math",
        selected_custom_answer="Keep the equations but explain each symbol.",
        generation_metadata={
            "provider": "openai",
            "model": "gpt-test",
            "prompt_version": "explainer.v1",
            "api_key": "sk-test-secret",
            "system_prompt": "very secret prompt text",
        },
        citations=[
            {
                "source_id": "media-42",
                "source_type": "media",
                "title": "Attention paper notes",
                "excerpt": "Attention weights are computed from query-key similarity.",
                "location_label": "chunk 3",
                "start_offset": 120,
                "end_offset": 178,
                "url": "https://example.test/attention",
                "snapshot_hash": "sha256:citationhash",
            }
        ],
    )
    repo.create_node(
        session.id,
        owner_user_id=owner_user_id,
        parent_id=root_id,
        title="Scaled dot product attention",
        body="Queries and keys are compared, scaled, and normalized with softmax.",
        kind="explanation",
        intent="explain",
        status="complete",
        evidence_state="supported",
        outside_knowledge_used=False,
        citations=[
            {
                "source_id": "media-42",
                "source_type": "media",
                "title": "Attention paper notes",
                "excerpt": "The score is divided by the square root of the key dimension.",
                "location_label": "chunk 4",
                "snapshot_hash": "sha256:childhash",
            }
        ],
    )
    repo.create_node(
        session.id,
        owner_user_id=owner_user_id,
        parent_id=root_id,
        title="Practice plan",
        body="Work through one toy example by hand.",
        kind="step",
        intent="plan",
        status="complete",
        evidence_state="partially_supported",
        outside_knowledge_used=True,
        citations=[],
    )
    loaded = repo.get_session(session.id, owner_user_id=owner_user_id)
    assert loaded is not None
    return loaded


def test_chatbook_adapter_serializes_complete_session(tmp_path):
    repo = ExplainerRepository(ExplainerDatabase(tmp_path / "Explainer.db"))
    session = _create_complete_session(repo)

    payload = build_explainer_chatbook_payload(
        repo=repo,
        session_id=session.id,
        owner_user_id="7",
    )

    assert payload["format"] == "tldw.explainer_session.v1"
    assert payload["type"] == "explainer_session"
    assert payload["structured"]["session"]["id"] == session.id
    assert payload["structured"]["session"]["settings"] == {
        "mode": "goal",
        "outputIntent": "both",
        "grounding": "source_led",
        "depthPreset": "standard",
    }
    assert payload["structured"]["session"]["createdAt"]
    assert payload["structured"]["selectedSources"][0]["snapshotVersion"] == "snapshot-v3"
    assert payload["structured"]["selectedSources"][0]["metadata"]["snapshotHash"] == "sha256:sourcehash"

    nodes = payload["structured"]["nodes"]
    assert [node["title"] for node in nodes] == [
        "Explain transformer attention",
        "Scaled dot product attention",
        "Practice plan",
    ]
    root = nodes[0]
    assert root["kind"] == "summary"
    assert root["intent"] == "plan"
    assert root["body"] == "Attention lets tokens route information to each other."
    assert root["status"] == "complete"
    assert root["evidenceState"] == "supported"
    assert root["questionOptions"][0]["id"] == "math"
    assert root["selectedOptionId"] == "math"
    assert root["selectedCustomAnswer"] == "Keep the equations but explain each symbol."
    assert root["generationMetadata"] == {
        "provider": "openai",
        "model": "gpt-test",
        "prompt_version": "explainer.v1",
    }
    assert root["citations"][0]["excerpt"] == "Attention weights are computed from query-key similarity."
    assert root["citations"][0]["locationLabel"] == "chunk 3"
    assert root["citations"][0]["snapshotHash"] == "sha256:citationhash"
    assert payload["structured"]["citations"][0]["nodeId"] == root["id"]
    assert payload["rendered"]["markdown"].startswith("# Learn attention")
    assert "Scaled dot product attention" in payload["rendered"]["markdown"]
    assert "Attention weights are computed" in payload["rendered"]["markdown"]
    _assert_text_absent(payload, "sk-test-secret")
    _assert_text_absent(payload, "very secret prompt text")


def test_chatbook_adapter_rejects_cross_user_session_access(tmp_path):
    repo = ExplainerRepository(ExplainerDatabase(tmp_path / "Explainer.db"))
    session = _create_complete_session(repo, owner_user_id="7")

    with pytest.raises(LookupError):
        build_explainer_chatbook_payload(
            repo=repo,
            session_id=session.id,
            owner_user_id="8",
        )


def test_chatbook_adapter_restores_explainer_session_for_importing_user(tmp_path):
    source_repo = ExplainerRepository(ExplainerDatabase(tmp_path / "source.db"))
    source_session = _create_complete_session(source_repo, owner_user_id="7")
    payload = build_explainer_chatbook_payload(
        repo=source_repo,
        session_id=source_session.id,
        owner_user_id="7",
    )
    target_repo = ExplainerRepository(ExplainerDatabase(tmp_path / "target.db"))

    restored = restore_explainer_chatbook_payload(
        repo=target_repo,
        payload=payload,
        owner_user_id="8",
    )

    assert restored.id != source_session.id
    assert restored.owner_user_id == "8"
    assert restored.title == "Learn attention"
    assert restored.output_intent == "both"
    assert restored.selected_sources[0].metadata["originalSourceId"] == "media-42"
    assert restored.selected_sources[0].metadata["resolutionStatus"] == "unresolved"
    restored_titles = [restored.nodes[node_id].title for node_id in restored.root_node_ids]
    assert restored_titles == ["Explain transformer attention"]
    restored_root = restored.nodes[restored.root_node_ids[0]]
    assert restored_root.kind == "summary"
    assert restored_root.intent == "plan"
    assert restored_root.generation_metadata["import"]["originalNodeId"] == source_session.root_node_ids[0]
    assert restored_root.question_options[0]["id"] == "math"
    assert restored_root.citations[0].excerpt == "Attention weights are computed from query-key similarity."
    assert [restored.nodes[child_id].title for child_id in restored_root.child_node_ids] == [
        "Scaled dot product attention",
        "Practice plan",
    ]


def test_generated_document_subtype_payload_restores_explainer_session(tmp_path):
    source_repo = ExplainerRepository(ExplainerDatabase(tmp_path / "source.db"))
    source_session = _create_complete_session(source_repo, owner_user_id="7")
    payload = build_explainer_chatbook_payload(
        repo=source_repo,
        session_id=source_session.id,
        owner_user_id="7",
    )
    fallback_payload = {
        "type": "generated_document",
        "metadata": {"subtype": "explainer_session"},
        "content": payload,
    }
    target_repo = ExplainerRepository(ExplainerDatabase(tmp_path / "target.db"))

    restored = restore_explainer_chatbook_payload(
        repo=target_repo,
        payload=fallback_payload,
        owner_user_id="8",
    )

    assert restored.owner_user_id == "8"
    assert restored.title == "Learn attention"
    assert restored.nodes[restored.root_node_ids[0]].title == "Explain transformer attention"


@pytest.mark.integration
def test_export_chatbook_endpoint_delegates_to_chatbook_service(tmp_path):
    app = FastAPI()
    app.include_router(explainer_router, prefix="/api/v1")
    db = ExplainerDatabase(tmp_path / "Explainer.db")
    repo = ExplainerRepository(db)
    session = _create_complete_session(repo, owner_user_id="7")
    chatbook_service = MagicMock()
    chatbook_service.create_chatbook = AsyncMock(
        return_value=(True, "Export job started: job-123", "job-123")
    )

    async def _override_user():
        return User(id=7, username="user-7", email=None, is_active=True, is_admin=True)

    async def _override_db():
        return db

    def _override_chatbook_service():
        return chatbook_service

    app.dependency_overrides[get_request_user] = _override_user
    app.dependency_overrides[get_explainer_db] = _override_db
    app.dependency_overrides[explainer_endpoint.get_chatbook_service] = _override_chatbook_service

    with TestClient(app) as client:
        response = client.post(f"/api/v1/explainer/sessions/{session.id}/export-chatbook")

    assert response.status_code == 200
    assert response.json() == {
        "success": True,
        "message": "Export job started: job-123",
        "job_id": "job-123",
        "download_url": None,
    }
    chatbook_service.create_chatbook.assert_awaited_once()
    call_kwargs = chatbook_service.create_chatbook.await_args.kwargs
    assert call_kwargs["content_selections"] == {ContentType.EXPLAINER_SESSION: [session.id]}
    assert call_kwargs["include_media"] is False
    assert call_kwargs["include_embeddings"] is False
    assert call_kwargs["include_generated_content"] is True
    assert call_kwargs["async_mode"] is True


@pytest.mark.integration
def test_export_chatbook_endpoint_honors_sync_export_mode(tmp_path, monkeypatch):
    app = FastAPI()
    app.include_router(explainer_router, prefix="/api/v1")
    db = ExplainerDatabase(tmp_path / "Explainer.db")
    repo = ExplainerRepository(db)
    session = _create_complete_session(repo, owner_user_id="7")
    chatbook_service = MagicMock()
    archive_path = tmp_path / "exports" / "explainer.chatbook"
    chatbook_service.create_chatbook = AsyncMock(
        return_value=(True, "Chatbook created successfully", str(archive_path))
    )

    async def _override_user():
        return User(id=7, username="user-7", email=None, is_active=True, is_admin=True)

    async def _override_db():
        return db

    def _override_chatbook_service():
        return chatbook_service

    def _persist_sync(**kwargs):
        assert kwargs["user_id"] == "7"
        assert kwargs["chatbook_name"] == "Learn attention Explainer Session"
        assert kwargs["output_path"] == str(archive_path)
        return "sync-job-1", "/api/v1/chatbooks/download/sync-job-1", archive_path, 123

    chatbook_service.register_completed_sync_export = MagicMock(side_effect=_persist_sync)

    app.dependency_overrides[get_request_user] = _override_user
    app.dependency_overrides[get_explainer_db] = _override_db
    app.dependency_overrides[explainer_endpoint.get_chatbook_service] = _override_chatbook_service

    with TestClient(app) as client:
        response = client.post(
            f"/api/v1/explainer/sessions/{session.id}/export-chatbook",
            json={"asyncMode": False},
        )

    assert response.status_code == 200
    assert response.json() == {
        "success": True,
        "message": "Chatbook created successfully",
        "job_id": "sync-job-1",
        "download_url": "/api/v1/chatbooks/download/sync-job-1",
    }
    call_kwargs = chatbook_service.create_chatbook.await_args.kwargs
    assert call_kwargs["async_mode"] is False


@pytest.mark.integration
def test_export_chatbook_endpoint_rechecks_session_ownership(tmp_path):
    app = FastAPI()
    app.include_router(explainer_router, prefix="/api/v1")
    db = ExplainerDatabase(tmp_path / "Explainer.db")
    repo = ExplainerRepository(db)
    session = _create_complete_session(repo, owner_user_id="7")
    chatbook_service = MagicMock()
    chatbook_service.create_chatbook = AsyncMock()

    async def _override_user():
        return User(id=8, username="user-8", email=None, is_active=True, is_admin=True)

    async def _override_db():
        return db

    def _override_chatbook_service():
        return chatbook_service

    app.dependency_overrides[get_request_user] = _override_user
    app.dependency_overrides[get_explainer_db] = _override_db
    app.dependency_overrides[explainer_endpoint.get_chatbook_service] = _override_chatbook_service

    with TestClient(app) as client:
        response = client.post(f"/api/v1/explainer/sessions/{session.id}/export-chatbook")

    assert response.status_code == 404
    chatbook_service.create_chatbook.assert_not_awaited()


def test_restore_failure_does_not_leave_partial_session(tmp_path):
    source_repo = ExplainerRepository(ExplainerDatabase(tmp_path / "source.db"))
    source_session = _create_complete_session(source_repo, owner_user_id="7")
    payload = build_explainer_chatbook_payload(
        repo=source_repo,
        session_id=source_session.id,
        owner_user_id="7",
    )
    payload["structured"]["nodes"].append(
        {
            "id": "node-orphan",
            "parentId": "node-that-does-not-exist",
            "title": "Orphaned node",
            "kind": "explanation",
            "intent": "explain",
        }
    )
    target_repo = ExplainerRepository(ExplainerDatabase(tmp_path / "target.db"))

    with pytest.raises(ValueError):
        restore_explainer_chatbook_payload(
            repo=target_repo,
            payload=payload,
            owner_user_id="8",
        )

    _summaries, total = target_repo.list_session_summaries(
        owner_user_id="8",
        include_archived=True,
    )
    assert total == 0
