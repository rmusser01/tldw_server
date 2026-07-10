"""Tests for the workspace context API contract."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_request_user
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import try_get_media_db_for_user
from tldw_Server_API.app.api.v1.endpoints import workspaces as workspaces_endpoint
from tldw_Server_API.app.api.v1.endpoints.workspaces_rate_limit_policy import WORKSPACES_READ_RATE_LIMIT
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


async def _allow_rate_limit() -> None:
    return None


async def _empty_service_capabilities(
    *,
    workspace_id: str,
    user_id: int | str | None,
) -> dict[str, Any]:
    _ = (workspace_id, user_id)
    return {}


@pytest.mark.integration
def test_workspace_context_source_items_include_review_fields(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = CharactersRAGDB(db_path=str(tmp_path / "context.db"), client_id="user-1")
    db.upsert_workspace("ws-context", "Context Workspace")
    db.add_workspace_source(
        "ws-context",
        {
            "id": "src-context",
            "media_id": 42,
            "title": "Context source",
            "source_type": "pdf",
            "review_state": "needs_review",
        },
    )
    monkeypatch.setattr(
        workspaces_endpoint,
        "collect_workspace_service_capabilities",
        _empty_service_capabilities,
    )
    app = FastAPI()
    app.include_router(workspaces_endpoint.router, prefix="/api/v1/workspaces")
    app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id="user-1")
    app.dependency_overrides[get_chacha_db_for_user] = lambda: db
    app.dependency_overrides[try_get_media_db_for_user] = lambda: None
    app.dependency_overrides[workspaces_endpoint.try_get_workspace_job_manager] = lambda: None
    app.dependency_overrides[WORKSPACES_READ_RATE_LIMIT] = _allow_rate_limit

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/api/v1/workspaces/ws-context/context")

    assert response.status_code == 200, response.text
    source = response.json()["sources"]["items"][0]
    assert source["review_state"] == "needs_review"
    assert source["review_state_updated_at"]
    assert source["reviewed_at"] is None
    assert source["reviewed_by_user_id"] is None
