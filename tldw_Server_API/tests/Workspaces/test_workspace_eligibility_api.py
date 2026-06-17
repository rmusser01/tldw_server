from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import try_get_media_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_request_user
from tldw_Server_API.app.api.v1.endpoints import workspace_eligibility as workspace_eligibility_endpoint
from tldw_Server_API.app.api.v1.endpoints.workspaces_rate_limit_policy import WORKSPACES_READ_RATE_LIMIT


class FakeEligibilityDB:
    def __init__(self) -> None:
        self.workspaces: dict[str, dict[str, object]] = {
            "workspace-1": {"id": "workspace-1", "archived": False},
            "workspace-2": {"id": "workspace-2", "archived": False},
            "workspace-archived": {"id": "workspace-archived", "archived": True},
        }
        self.memberships: dict[tuple[str, str, str], dict[str, object]] = {
            ("workspace-1", "chat", "chat-1"): {
                "workspace_id": "workspace-1",
                "resource_type": "chat",
                "resource_id": "chat-1",
                "role": "conversation",
                "label": None,
                "transfer_policy": "link",
                "provenance": {},
                "metadata": {},
                "created_at": "2026-06-17T12:00:00Z",
                "updated_at": "2026-06-17T12:00:00Z",
                "version": 1,
                "deleted": False,
            },
            ("workspace-2", "chat", "chat-2"): {
                "workspace_id": "workspace-2",
                "resource_type": "chat",
                "resource_id": "chat-2",
                "role": "conversation",
                "label": None,
                "transfer_policy": "link",
                "provenance": {},
                "metadata": {},
                "created_at": "2026-06-17T12:01:00Z",
                "updated_at": "2026-06-17T12:01:00Z",
                "version": 1,
                "deleted": False,
            },
        }

    def get_workspace(self, workspace_id: str) -> dict[str, object] | None:
        return self.workspaces.get(workspace_id)

    def get_workspace_resource_membership(
        self,
        workspace_id: str,
        resource_type: str,
        resource_id: str,
        *,
        include_deleted: bool = False,
    ) -> dict[str, object] | None:
        row = self.memberships.get((workspace_id, resource_type, resource_id))
        if row is None or (row.get("deleted") and not include_deleted):
            return None
        return dict(row)

    def list_resource_workspace_memberships(
        self,
        resource_type: str,
        resource_id: str,
        *,
        include_deleted: bool = False,
        limit: int = 100,
        cursor: tuple[str, str] | None = None,
    ) -> list[dict[str, object]]:
        _ = cursor
        rows = [
            row
            for (_, row_resource_type, row_resource_id), row in self.memberships.items()
            if row_resource_type == resource_type
            and row_resource_id == resource_id
            and (include_deleted or not row.get("deleted"))
        ]
        rows.sort(key=lambda row: (str(row["updated_at"]), str(row["workspace_id"])), reverse=True)
        return [dict(row) for row in rows[:limit]]


@pytest.fixture
def db() -> FakeEligibilityDB:
    return FakeEligibilityDB()


@pytest.fixture
def client(db: FakeEligibilityDB) -> TestClient:
    app = FastAPI()
    app.include_router(workspace_eligibility_endpoint.router, prefix="/api/v1/workspace-eligibility")

    async def _allow_rate_limit() -> None:
        return None

    app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id="user-1")
    app.dependency_overrides[get_chacha_db_for_user] = lambda: db
    app.dependency_overrides[try_get_media_db_for_user] = lambda: None
    app.dependency_overrides[WORKSPACES_READ_RATE_LIMIT] = _allow_rate_limit
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


def _payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "active_workspace_id": "workspace-1",
        "resource_type": "chat",
        "resource_id": "chat-1",
        "operation": "rag_ground",
        "runtime_state": "not_required",
        "permission_state": "granted",
    }
    payload.update(overrides)
    return payload


def test_post_check_allows_visibility_without_active_workspace(client: TestClient) -> None:
    response = client.post(
        "/api/v1/workspace-eligibility/check",
        json=_payload(
            active_workspace_id=None,
            resource_type="prompt",
            resource_id="prompt-1",
            operation="open",
        ),
    )

    assert response.status_code == 200
    body = response.json()
    assert body["allowed"] is True
    assert body["operation_category"] == "visibility"
    assert body["reason_code"] == "visibility_allowed"
    assert body["global_visibility_preserved"] is True


def test_post_check_allows_active_operation_for_linked_resource(client: TestClient) -> None:
    response = client.post("/api/v1/workspace-eligibility/check", json=_payload())

    assert response.status_code == 200
    body = response.json()
    assert body["allowed"] is True
    assert body["reason_code"] == "allowed"
    assert body["membership"]["workspace_id"] == "workspace-1"


def test_post_check_denies_cross_workspace_resource(client: TestClient) -> None:
    response = client.post(
        "/api/v1/workspace-eligibility/check",
        json=_payload(resource_id="chat-2"),
    )

    assert response.status_code == 200
    body = response.json()
    assert body["allowed"] is False
    assert body["reason_code"] == "cross_workspace_resource"
    assert body["resource_workspace_ids"] == ["workspace-2"]
    assert {action["action"] for action in body["recovery_actions"]} == {
        "copy_to_active_workspace",
        "switch_workspace",
    }


def test_post_check_denies_missing_runtime(client: TestClient) -> None:
    response = client.post(
        "/api/v1/workspace-eligibility/check",
        json=_payload(operation="acp_run", runtime_state="not_required"),
    )

    assert response.status_code == 200
    body = response.json()
    assert body["allowed"] is False
    assert body["reason_code"] == "missing_runtime"
    assert body["recovery_actions"][0]["action"] == "configure_workspace_runtime"


def test_post_check_requires_explicit_runtime_and_permission_state(client: TestClient) -> None:
    payload = _payload()
    payload.pop("runtime_state")
    payload.pop("permission_state")

    response = client.post("/api/v1/workspace-eligibility/check", json=payload)

    assert response.status_code == 422
    missing_fields = {error["loc"][-1] for error in response.json()["detail"]}
    assert {"runtime_state", "permission_state"} <= missing_fields
