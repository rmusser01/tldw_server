from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import try_get_media_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_request_user
from tldw_Server_API.app.api.v1.endpoints import workspace_memberships as workspace_memberships_endpoint
from tldw_Server_API.app.api.v1.endpoints import workspaces as workspaces_endpoint
from tldw_Server_API.app.api.v1.endpoints.workspaces_rate_limit_policy import (
    WORKSPACES_DELETE_RATE_LIMIT,
    WORKSPACES_READ_RATE_LIMIT,
    WORKSPACES_WRITE_RATE_LIMIT,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import ConflictError


class FakeMembershipDB:
    def __init__(self) -> None:
        self.workspaces: dict[str, dict[str, object]] = {
            "workspace-1": {"id": "workspace-1", "archived": False},
            "workspace-2": {"id": "workspace-2", "archived": False},
            "workspace-archived": {"id": "workspace-archived", "archived": True},
        }
        self.conversations: dict[str, dict[str, object]] = {
            "chat-1": {
                "id": "chat-1",
                "title": "Research chat",
                "scope_type": "global",
                "workspace_id": None,
                "last_modified": "2026-06-07T12:00:00Z",
                "version": 1,
            }
        }
        self.memberships: dict[tuple[str, str, str], dict[str, object]] = {}
        self._clock = 0

    def _timestamp(self) -> str:
        self._clock += 1
        return f"2026-06-07T12:00:{self._clock:02d}Z"

    def get_workspace(self, workspace_id: str) -> dict[str, object] | None:
        return self.workspaces.get(workspace_id)

    def get_conversation_for_workspace_membership(self, conversation_id: str) -> dict[str, object] | None:
        return self.conversations.get(conversation_id)

    def add_workspace_resource_membership(
        self,
        workspace_id: str,
        data: dict[str, object],
        *,
        user_id: str | None = None,
    ) -> dict[str, object]:
        key = (workspace_id, str(data["resource_type"]), str(data["resource_id"]))
        existing = self.memberships.get(key)
        if existing is not None:
            if existing.get("deleted"):
                if data.get("restore_deleted") is not True:
                    return dict(existing)
                existing.update(
                    {
                        "role": data.get("role", "member"),
                        "label": data.get("label"),
                        "transfer_policy": data.get("transfer_policy", "link"),
                        "provenance": data.get("provenance", {}),
                        "metadata": data.get("metadata", {}),
                        "updated_by_user_id": user_id,
                        "updated_at": self._timestamp(),
                        "version": int(existing.get("version", 1)) + 1,
                        "deleted": False,
                    }
                )
                return dict(existing)
            if (
                existing.get("role") != data.get("role", "member")
                or existing.get("label") != data.get("label")
                or existing.get("transfer_policy") != data.get("transfer_policy", "link")
            ):
                raise ConflictError(entity="workspace_resource_memberships")
            return dict(existing)

        now = self._timestamp()
        row = {
            "workspace_id": workspace_id,
            "resource_type": str(data["resource_type"]),
            "resource_id": str(data["resource_id"]),
            "role": data.get("role", "member"),
            "label": data.get("label"),
            "transfer_policy": data.get("transfer_policy", "link"),
            "provenance": data.get("provenance", {}),
            "metadata": data.get("metadata", {}),
            "created_at": now,
            "updated_at": now,
            "version": 1,
            "deleted": False,
            "created_by_user_id": user_id,
            "updated_by_user_id": user_id,
        }
        self.memberships[key] = row
        return dict(row)

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

    def list_workspace_resource_memberships(
        self,
        workspace_id: str,
        *,
        resource_type: str | None = None,
        role: str | None = None,
        include_deleted: bool = False,
        limit: int = 100,
        cursor: tuple[str, str, str] | None = None,
    ) -> list[dict[str, object]]:
        _ = cursor
        rows = [
            row
            for (row_workspace_id, _, _), row in self.memberships.items()
            if row_workspace_id == workspace_id
            and (resource_type is None or row["resource_type"] == resource_type)
            and (role is None or row["role"] == role)
            and (include_deleted or not row.get("deleted"))
        ]
        rows.sort(
            key=lambda row: (
                str(row["updated_at"]),
                str(row["resource_type"]),
                str(row["resource_id"]),
            ),
            reverse=True,
        )
        return [dict(row) for row in rows[:limit]]

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

    def delete_workspace_resource_membership(
        self,
        workspace_id: str,
        resource_type: str,
        resource_id: str,
        *,
        user_id: str | None = None,
    ) -> dict[str, object] | None:
        row = self.memberships.get((workspace_id, resource_type, resource_id))
        if row is None or row.get("deleted"):
            return None
        row["deleted"] = True
        row["updated_by_user_id"] = user_id
        row["updated_at"] = self._timestamp()
        row["version"] = int(row.get("version", 1)) + 1
        return dict(row)


@pytest.fixture
def db() -> FakeMembershipDB:
    return FakeMembershipDB()


@pytest.fixture
def membership_app(db: FakeMembershipDB) -> FastAPI:
    app = FastAPI()
    app.include_router(workspaces_endpoint.router, prefix="/api/v1/workspaces")
    app.include_router(workspace_memberships_endpoint.router, prefix="/api/v1/workspace-memberships")

    async def _allow_rate_limit() -> None:
        return None

    app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id="user-1")
    app.dependency_overrides[get_chacha_db_for_user] = lambda: db
    app.dependency_overrides[try_get_media_db_for_user] = lambda: object()
    app.dependency_overrides[WORKSPACES_READ_RATE_LIMIT] = _allow_rate_limit
    app.dependency_overrides[WORKSPACES_WRITE_RATE_LIMIT] = _allow_rate_limit
    app.dependency_overrides[WORKSPACES_DELETE_RATE_LIMIT] = _allow_rate_limit
    return app


@pytest.fixture
def client(membership_app: FastAPI) -> TestClient:
    with TestClient(membership_app, raise_server_exceptions=False) as test_client:
        yield test_client


def _membership_payload(**overrides: Any) -> dict[str, Any]:
    payload = {
        "resource_type": "chat",
        "resource_id": "chat-1",
        "role": "conversation",
        "label": "Research chat",
        "transfer_policy": "link",
        "provenance": {"source_surface": "test"},
    }
    payload.update(overrides)
    return payload


def test_post_creates_membership(client: TestClient) -> None:
    response = client.post("/api/v1/workspaces/workspace-1/memberships", json=_membership_payload())

    assert response.status_code == 201
    body = response.json()
    assert body["workspace_id"] == "workspace-1"
    assert body["resource_type"] == "chat"
    assert body["resource_id"] == "chat-1"
    assert body["role"] == "conversation"
    assert body["summary"]["title"] == "Research chat"


def test_duplicate_same_request_is_idempotent(client: TestClient, db: FakeMembershipDB) -> None:
    first = client.post("/api/v1/workspaces/workspace-1/memberships", json=_membership_payload())
    second = client.post("/api/v1/workspaces/workspace-1/memberships", json=_membership_payload())

    assert first.status_code == 201
    assert second.status_code == 201
    assert second.json()["version"] == first.json()["version"]
    assert second.json()["created_at"] == first.json()["created_at"]
    assert len(db.memberships) == 1


def test_duplicate_conflicting_request_returns_409(client: TestClient) -> None:
    created = client.post("/api/v1/workspaces/workspace-1/memberships", json=_membership_payload())

    response = client.post(
        "/api/v1/workspaces/workspace-1/memberships",
        json=_membership_payload(role="reference"),
    )

    assert created.status_code == 201
    assert response.status_code == 409


def test_archived_workspace_write_returns_stable_error_code(client: TestClient) -> None:
    response = client.post(
        "/api/v1/workspaces/workspace-archived/memberships",
        json=_membership_payload(),
    )

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "workspace_archived"


def test_workspace_list_filters_and_resolve_false(client: TestClient, db: FakeMembershipDB) -> None:
    db.conversations["chat-2"] = {
        "id": "chat-2",
        "title": "Reference chat",
        "scope_type": "global",
        "workspace_id": None,
        "last_modified": "2026-06-07T12:01:00Z",
    }
    client.post("/api/v1/workspaces/workspace-1/memberships", json=_membership_payload())
    client.post(
        "/api/v1/workspaces/workspace-1/memberships",
        json=_membership_payload(resource_id="chat-2", role="reference", label="Reference chat"),
    )

    response = client.get(
        "/api/v1/workspaces/workspace-1/memberships",
        params={"resource_type": "chat", "role": "conversation", "resolve": "false"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["workspace_id"] == "workspace-1"
    assert body["total"] == 1
    assert body["items"][0]["resource_id"] == "chat-1"
    assert body["items"][0]["summary"] is None
    assert body["summary"]["by_role"] == {"conversation": 1}


def test_get_one_membership_and_missing_returns_stable_404(client: TestClient) -> None:
    client.post("/api/v1/workspaces/workspace-1/memberships", json=_membership_payload())

    found = client.get("/api/v1/workspaces/workspace-1/memberships/chat/chat-1")
    missing = client.get("/api/v1/workspaces/workspace-1/memberships/chat/missing-chat")

    assert found.status_code == 200
    assert found.json()["resource_id"] == "chat-1"
    assert missing.status_code == 404
    assert missing.json()["detail"]["code"] == "workspace_membership_not_found"


def test_delete_soft_deletes_and_default_list_hides_membership(client: TestClient) -> None:
    client.post("/api/v1/workspaces/workspace-1/memberships", json=_membership_payload())

    delete_response = client.delete("/api/v1/workspaces/workspace-1/memberships/chat/chat-1")
    list_response = client.get("/api/v1/workspaces/workspace-1/memberships")

    assert delete_response.status_code == 204
    assert list_response.status_code == 200
    assert list_response.json()["items"] == []


def test_relink_after_delete_restores_membership(client: TestClient) -> None:
    client.post("/api/v1/workspaces/workspace-1/memberships", json=_membership_payload())
    client.delete("/api/v1/workspaces/workspace-1/memberships/chat/chat-1")

    response = client.post(
        "/api/v1/workspaces/workspace-1/memberships",
        json=_membership_payload(role="reference"),
    )

    assert response.status_code == 201
    body = response.json()
    assert body["role"] == "reference"
    assert body["deleted"] is False
    assert body["version"] == 3


def test_reverse_resource_route_returns_resource_memberships(client: TestClient) -> None:
    client.post("/api/v1/workspaces/workspace-1/memberships", json=_membership_payload())
    client.post("/api/v1/workspaces/workspace-2/memberships", json=_membership_payload())

    response = client.get("/api/v1/workspace-memberships/resources/chat/chat-1", params={"resolve": "false"})

    assert response.status_code == 200
    body = response.json()
    assert body["resource_type"] == "chat"
    assert body["resource_id"] == "chat-1"
    assert "workspace_id" not in body
    assert body["total"] == 2
    assert {item["workspace_id"] for item in body["items"]} == {"workspace-1", "workspace-2"}
    assert all(item["summary"] is None for item in body["items"])


def test_unsupported_resource_type_returns_stable_400_code(client: TestClient) -> None:
    response = client.get("/api/v1/workspace-memberships/resources/note/1")

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "unsupported_resource_type"


def test_post_unsupported_resource_type_returns_stable_400_code(client: TestClient) -> None:
    response = client.post(
        "/api/v1/workspaces/workspace-1/memberships",
        json=_membership_payload(resource_type="note", resource_id="1"),
    )

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "unsupported_resource_type"


def test_missing_media_db_on_media_link_returns_503_code(membership_app: FastAPI) -> None:
    membership_app.dependency_overrides[try_get_media_db_for_user] = lambda: None
    with TestClient(membership_app, raise_server_exceptions=False) as test_client:
        response = test_client.post(
            "/api/v1/workspaces/workspace-1/memberships",
            json=_membership_payload(resource_type="media", resource_id="42", role="source"),
        )

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "media_db_unavailable"
