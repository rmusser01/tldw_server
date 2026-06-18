from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_request_user
from tldw_Server_API.app.api.v1.endpoints import workspaces as workspaces_endpoint
from tldw_Server_API.app.api.v1.endpoints.workspaces_rate_limit_policy import (
    WORKSPACES_DELETE_RATE_LIMIT,
    WORKSPACES_READ_RATE_LIMIT,
    WORKSPACES_WRITE_RATE_LIMIT,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


@pytest.fixture
def db(tmp_path: Path) -> Iterator[CharactersRAGDB]:
    database = CharactersRAGDB(db_path=str(tmp_path / "chacha.db"), client_id="user-1")
    try:
        yield database
    finally:
        database.close_connection()


@pytest.fixture
def workspace_fastapi_app() -> FastAPI:
    app = FastAPI()
    app.include_router(workspaces_endpoint.router, prefix="/api/v1/workspaces")
    return app


@pytest.fixture
def workspace_client(workspace_fastapi_app: FastAPI, db: CharactersRAGDB) -> Iterator[TestClient]:
    async def _allow_rate_limit() -> None:
        return None

    workspace_fastapi_app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)
    workspace_fastapi_app.dependency_overrides[get_chacha_db_for_user] = lambda: db
    workspace_fastapi_app.dependency_overrides[WORKSPACES_READ_RATE_LIMIT] = _allow_rate_limit
    workspace_fastapi_app.dependency_overrides[WORKSPACES_WRITE_RATE_LIMIT] = _allow_rate_limit
    workspace_fastapi_app.dependency_overrides[WORKSPACES_DELETE_RATE_LIMIT] = _allow_rate_limit
    try:
        with TestClient(workspace_fastapi_app, raise_server_exceptions=False) as client:
            yield client
    finally:
        workspace_fastapi_app.dependency_overrides.pop(get_request_user, None)
        workspace_fastapi_app.dependency_overrides.pop(get_chacha_db_for_user, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_READ_RATE_LIMIT, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_WRITE_RATE_LIMIT, None)
        workspace_fastapi_app.dependency_overrides.pop(WORKSPACES_DELETE_RATE_LIMIT, None)


def _descriptor_payload(**overrides: Any) -> dict[str, Any]:
    payload = {
        "binding_id": "acp-session-1",
        "binding_kind": "acp_session",
        "owner_domain": "acp",
        "locator_ref": "session-1",
        "label": "ACP Session",
        "status": "runtime_missing",
        "path_hint": "/Users/example/agent-workspace",
        "portability": "metadata_only",
        "metadata": {
            "agent": "codex",
            "absolute_root": "/Users/example/private/agent-workspace",
        },
    }
    payload.update(overrides)
    return payload


def test_workspace_runtime_binding_api_upserts_lists_gets_and_archives(
    workspace_client: TestClient,
    db: CharactersRAGDB,
) -> None:
    db.upsert_workspace("ws-runtime", "Runtime Workspace")

    response = workspace_client.post(
        "/api/v1/workspaces/ws-runtime/runtime-bindings",
        json=_descriptor_payload(),
    )
    assert response.status_code == 201
    created = response.json()
    assert created["binding_id"] == "acp-session-1"
    assert created["status"] == "runtime-missing"
    assert created["path_hint"] == "agent-workspace"
    assert created["metadata"] == {"agent": "codex", "absolute_root": "agent-workspace"}
    assert created["redaction_report"]["redacted_fields"] == [
        "metadata.absolute_root",
        "path_hint",
    ]
    assert "absolute_root" not in created

    listed = workspace_client.get(
        "/api/v1/workspaces/ws-runtime/runtime-bindings?binding_kind=acp_session"
    )
    assert listed.status_code == 200
    assert listed.json()["total"] == 1
    assert listed.json()["items"][0]["binding_id"] == "acp-session-1"

    fetched = workspace_client.get(
        "/api/v1/workspaces/ws-runtime/runtime-bindings/acp-session-1"
    )
    assert fetched.status_code == 200
    assert fetched.json()["owner_domain"] == "acp"

    deleted = workspace_client.delete(
        "/api/v1/workspaces/ws-runtime/runtime-bindings/acp-session-1"
    )
    assert deleted.status_code == 204
    assert (
        workspace_client.get(
            "/api/v1/workspaces/ws-runtime/runtime-bindings/acp-session-1"
        ).status_code
        == 404
    )


def test_workspace_runtime_binding_api_rejects_unknown_binding_kind(
    workspace_client: TestClient,
    db: CharactersRAGDB,
) -> None:
    db.upsert_workspace("ws-runtime", "Runtime Workspace")

    response = workspace_client.post(
        "/api/v1/workspaces/ws-runtime/runtime-bindings",
        json=_descriptor_payload(binding_kind="unknown_runtime"),
    )

    assert response.status_code == 422


def test_workspace_runtime_binding_api_rejects_secret_metadata_key(
    workspace_client: TestClient,
    db: CharactersRAGDB,
) -> None:
    db.upsert_workspace("ws-runtime", "Runtime Workspace")

    response = workspace_client.post(
        "/api/v1/workspaces/ws-runtime/runtime-bindings",
        json=_descriptor_payload(metadata={"OPENAI_API_KEY": "sk-secret"}),
    )

    assert response.status_code == 422


def test_workspace_runtime_binding_api_rejects_client_redaction_report(
    workspace_client: TestClient,
    db: CharactersRAGDB,
) -> None:
    db.upsert_workspace("ws-runtime", "Runtime Workspace")

    response = workspace_client.post(
        "/api/v1/workspaces/ws-runtime/runtime-bindings",
        json=_descriptor_payload(
            redaction_report={"redacted": True, "redacted_fields": ["metadata.agent"]},
        ),
    )

    assert response.status_code == 422


def test_workspace_runtime_binding_api_rejects_archived_status_on_upsert(
    workspace_client: TestClient,
    db: CharactersRAGDB,
) -> None:
    db.upsert_workspace("ws-runtime", "Runtime Workspace")

    response = workspace_client.post(
        "/api/v1/workspaces/ws-runtime/runtime-bindings",
        json=_descriptor_payload(status="archived"),
    )

    assert response.status_code == 422


def test_workspace_runtime_binding_api_returns_404_for_missing_workspace(
    workspace_client: TestClient,
) -> None:
    response = workspace_client.post(
        "/api/v1/workspaces/missing/runtime-bindings",
        json=_descriptor_payload(),
    )

    assert response.status_code == 404


def test_workspace_runtime_binding_api_rejects_writes_to_archived_workspace(
    workspace_client: TestClient,
    db: CharactersRAGDB,
) -> None:
    workspace = db.upsert_workspace("ws-archived", "Archived Workspace")
    db.update_workspace("ws-archived", {"archived": True}, expected_version=workspace["version"])

    response = workspace_client.post(
        "/api/v1/workspaces/ws-archived/runtime-bindings",
        json=_descriptor_payload(),
    )

    assert response.status_code == 409
