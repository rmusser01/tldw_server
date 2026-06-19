from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import try_get_media_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.Prompts_DB_Deps import try_get_prompts_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.Watchlists_DB_Deps import try_get_watchlists_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_request_user
from tldw_Server_API.app.api.v1.endpoints import workspaces as workspaces_endpoint
from tldw_Server_API.app.api.v1.endpoints.workspaces_rate_limit_policy import WORKSPACES_READ_RATE_LIMIT
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


def test_workspace_activity_events_are_timestamped_newest_first_and_safe(tmp_path) -> None:
    db = CharactersRAGDB(db_path=str(tmp_path / "chacha.db"), client_id="test-client")
    db.upsert_workspace("workspace-1", "Activity Workspace")

    db.record_workspace_activity_event(
        "workspace-1",
        {
            "event_type": "membership.linked",
            "category": "membership",
            "resource_type": "chat",
            "resource_id": "chat-1",
            "summary": "Linked chat",
            "metadata": {
                "role": "conversation",
                "absolute_root": "/Users/alice/private/project",
                "api_key": "sk-secret-value",
            },
        },
        user_id="user-1",
    )
    db.record_workspace_activity_event(
        "workspace-1",
        {
            "event_type": "runtime_binding.upserted",
            "category": "runtime_binding",
            "resource_type": "workspace_runtime_binding",
            "resource_id": "repo-main",
            "summary": "Updated runtime binding",
            "metadata": {"binding_kind": "repo", "status": "missing"},
        },
        user_id="user-1",
    )

    rows = db.list_workspace_activity_events("workspace-1", limit=10)

    assert [row["event_type"] for row in rows] == [
        "runtime_binding.upserted",
        "membership.linked",
    ]
    assert rows[0]["category"] == "runtime_binding"
    assert rows[0]["actor_user_id"] == "user-1"
    assert rows[0]["created_at"]
    assert rows[1]["metadata"]["absolute_root"] == "project"
    assert "api_key" not in rows[1]["metadata"]
    assert "/Users/alice" not in repr(rows)
    assert "sk-secret" not in repr(rows)
    assert [
        row["event_type"]
        for row in db.list_workspace_activity_events("workspace-1", limit=10, category="runtime_binding")
    ] == ["runtime_binding.upserted"]


class FakeWorkspaceIndexDB:
    def __init__(self) -> None:
        self.workspaces: dict[str, dict[str, object]] = {
            "workspace-1": {
                "id": "workspace-1",
                "name": "Activity Workspace",
                "archived": False,
                "deleted": False,
                "workspace_profile": "project",
                "study_materials_policy": "workspace",
                "created_at": "2026-06-18T12:00:00Z",
                "last_modified": "2026-06-18T12:00:00Z",
                "version": 3,
            }
        }
        self.conversations: dict[str, dict[str, object]] = {
            "chat-1": {
                "id": "chat-1",
                "title": "Research chat",
                "scope_type": "global",
                "workspace_id": None,
                "last_modified": "2026-06-18T12:05:00Z",
                "version": 1,
            }
        }
        self.memberships: list[dict[str, object]] = [
            {
                "workspace_id": "workspace-1",
                "resource_type": "chat",
                "resource_id": "chat-1",
                "role": "conversation",
                "label": "Research chat",
                "transfer_policy": "link",
                "provenance": {},
                "metadata": {},
                "created_at": "2026-06-18T12:10:00Z",
                "updated_at": "2026-06-18T12:10:00Z",
                "version": 1,
                "deleted": False,
            },
            {
                "workspace_id": "workspace-1",
                "resource_type": "prompt",
                "resource_id": "7",
                "role": "reference",
                "label": "Research Prompt",
                "transfer_policy": "link",
                "provenance": {},
                "metadata": {},
                "created_at": "2026-06-18T12:09:00Z",
                "updated_at": "2026-06-18T12:09:00Z",
                "version": 1,
                "deleted": False,
            },
        ]

    def get_workspace(self, workspace_id: str, *, include_deleted: bool = False) -> dict[str, object] | None:
        _ = include_deleted
        return self.workspaces.get(workspace_id)

    def get_conversation_for_workspace_membership(self, conversation_id: str) -> dict[str, object] | None:
        return self.conversations.get(conversation_id)

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
            for row in self.memberships
            if row["workspace_id"] == workspace_id
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

    def workspace_resource_membership_summary(self, workspace_id: str) -> dict[str, object]:
        rows = [
            row
            for row in self.memberships
            if row["workspace_id"] == workspace_id and not row.get("deleted")
        ]
        by_resource_type: dict[str, int] = {}
        by_role: dict[str, int] = {}
        for row in rows:
            by_resource_type[str(row["resource_type"])] = by_resource_type.get(str(row["resource_type"]), 0) + 1
            by_role[str(row["role"])] = by_role.get(str(row["role"]), 0) + 1
        return {"total": len(rows), "by_resource_type": by_resource_type, "by_role": by_role}

    def list_workspace_runtime_bindings(
        self,
        workspace_id: str,
        *,
        binding_kind: str | None = None,
        owner_domain: str | None = None,
        include_deleted: bool = False,
        limit: int = 100,
    ) -> list[dict[str, object]]:
        _ = workspace_id, binding_kind, owner_domain, include_deleted, limit
        return [
            {
                "workspace_id": "workspace-1",
                "binding_id": "repo-main",
                "binding_kind": "repo",
                "owner_domain": "workspaces",
                "locator_ref": "repo-1",
                "label": "Main Repo",
                "status": "missing",
                "path_hint": "/Users/alice/private/project",
                "portability": "reference",
                "metadata": {"branch": "dev"},
                "redaction_report": {"redacted": True, "redacted_fields": ["path_hint"], "rejected_fields": []},
                "created_at": "2026-06-18T12:00:00Z",
                "updated_at": "2026-06-18T12:00:00Z",
                "deleted": False,
                "version": 1,
            }
        ]

    def list_workspace_activity_events(
        self,
        workspace_id: str,
        *,
        limit: int = 50,
        category: str | None = None,
    ) -> list[dict[str, object]]:
        _ = workspace_id, category
        return [
            {
                "workspace_id": "workspace-1",
                "event_id": "evt-1",
                "event_type": "membership.linked",
                "category": "membership",
                "actor_user_id": "user-1",
                "resource_type": "chat",
                "resource_id": "chat-1",
                "summary": "Linked chat",
                "metadata": {"role": "conversation"},
                "created_at": "2026-06-18T12:10:00Z",
                "version": 1,
            }
        ][:limit]


class FakePromptsDB:
    def fetch_prompt_details(self, prompt_id_or_name_or_uuid: object, include_deleted: bool = False) -> dict[str, object] | None:
        _ = include_deleted
        if str(prompt_id_or_name_or_uuid) != "7":
            return None
        return {
            "id": 7,
            "uuid": "11111111-1111-4111-8111-111111111111",
            "name": "Research Prompt",
            "author": "analyst",
            "details": "do not expose",
            "last_modified": "2026-06-18T12:04:00Z",
            "deleted": 0,
        }


def test_workspace_index_api_groups_resources_activity_and_warnings() -> None:
    db = FakeWorkspaceIndexDB()
    app = FastAPI()
    app.include_router(workspaces_endpoint.router, prefix="/api/v1/workspaces")

    async def _allow_rate_limit() -> None:
        return None

    app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id="user-1", tenant_id="tenant-a")
    app.dependency_overrides[get_chacha_db_for_user] = lambda: db
    app.dependency_overrides[try_get_media_db_for_user] = lambda: None
    app.dependency_overrides[try_get_prompts_db_for_user] = lambda: FakePromptsDB()
    app.dependency_overrides[workspaces_endpoint.try_get_workflows_db_for_user] = lambda: None
    app.dependency_overrides[try_get_watchlists_db_for_user] = lambda: None
    app.dependency_overrides[WORKSPACES_READ_RATE_LIMIT] = _allow_rate_limit

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/api/v1/workspaces/workspace-1/index", params={"group_limit": "2", "activity_limit": "5"})

    assert response.status_code == 200
    body = response.json()
    assert body["schema_version"] == 1
    assert body["workspace"]["id"] == "workspace-1"
    assert body["workspace"]["workspace_profile"] == "project"
    assert body["membership_summary"] == {
        "total": 2,
        "by_resource_type": {"chat": 1, "prompt": 1},
        "by_role": {"conversation": 1, "reference": 1},
    }
    groups = {group["resource_type"]: group for group in body["resource_groups"]}
    assert groups["chat"]["count"] == 1
    assert groups["chat"]["owner_surface"]["href"]
    assert groups["chat"]["items"][0]["summary"]["title"] == "Research chat"
    assert groups["prompt"]["items"][0]["summary"]["title"] == "Research Prompt"
    assert body["runtime_summary"]["total"] == 1
    assert body["runtime_summary"]["by_status"] == {"missing": 1}
    assert body["recent_activity"][0]["event_type"] == "membership.linked"
    assert body["recent_activity"][0]["created_at"] == "2026-06-18T12:10:00Z"
    assert any(warning["reason_code"] == "runtime_binding_missing" for warning in body["warnings"])
    assert "/Users/alice" not in response.text
    assert "do not expose" not in response.text


def test_workspace_index_api_returns_deleted_warning_for_soft_deleted_workspace(tmp_path) -> None:
    db = CharactersRAGDB(db_path=str(tmp_path / "chacha.db"), client_id="test-client")
    try:
        workspace = db.upsert_workspace("workspace-deleted", "Deleted Workspace")
        db.record_workspace_activity_event(
            "workspace-deleted",
            {
                "event_type": "workspace.deleted",
                "category": "workspace",
                "summary": "Deleted workspace",
            },
            user_id="user-1",
        )
        db.delete_workspace("workspace-deleted", expected_version=workspace["version"])
        app = FastAPI()
        app.include_router(workspaces_endpoint.router, prefix="/api/v1/workspaces")

        async def _allow_rate_limit() -> None:
            return None

        app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id="user-1")
        app.dependency_overrides[get_chacha_db_for_user] = lambda: db
        app.dependency_overrides[try_get_media_db_for_user] = lambda: None
        app.dependency_overrides[try_get_prompts_db_for_user] = lambda: None
        app.dependency_overrides[workspaces_endpoint.try_get_workflows_db_for_user] = lambda: None
        app.dependency_overrides[try_get_watchlists_db_for_user] = lambda: None
        app.dependency_overrides[WORKSPACES_READ_RATE_LIMIT] = _allow_rate_limit

        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.get("/api/v1/workspaces/workspace-deleted/index")

        assert response.status_code == 200
        body = response.json()
        assert body["workspace"]["deleted"] in (True, 1)
        assert body["membership_summary"] == {"total": 0, "by_resource_type": {}, "by_role": {}}
        assert body["resource_groups"] == []
        assert body["runtime_summary"] == {"total": 0, "by_kind": {}, "by_status": {}, "bindings": []}
        assert body["recent_activity"][0]["event_type"] == "workspace.deleted"
        assert any(warning["reason_code"] == "workspace_deleted" for warning in body["warnings"])
    finally:
        db.close_connection()
