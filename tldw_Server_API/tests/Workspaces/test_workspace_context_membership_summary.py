from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import try_get_media_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_request_user
from tldw_Server_API.app.api.v1.endpoints import workspaces as workspaces_endpoint
from tldw_Server_API.app.api.v1.endpoints.workspaces_rate_limit_policy import WORKSPACES_READ_RATE_LIMIT
from tldw_Server_API.app.core.Workspaces import membership_adapters
from tldw_Server_API.app.core.Workspaces.membership_service import (
    WorkspaceMembershipService,
    WorkspaceMembershipServiceError,
)


class BackfillFakeDB:
    def __init__(self) -> None:
        self.workspaces = {"workspace-1": _workspace_row("workspace-1")}
        self.sources = [
            {
                "id": "source-1",
                "workspace_id": "workspace-1",
                "media_id": 42,
                "title": "Source title",
                "source_type": "pdf",
                "added_at": "2026-06-07T12:01:00Z",
            }
        ]
        self.artifacts = [
            {
                "id": "artifact-1",
                "workspace_id": "workspace-1",
                "title": "Artifact title",
                "artifact_type": "report",
                "review_state": "accepted",
                "created_at": "2026-06-07T12:02:00Z",
            }
        ]
        self.notes = [
            {
                "id": 7,
                "workspace_id": "workspace-1",
                "title": "Note title",
                "last_modified": "2026-06-07T12:03:00Z",
                "deleted": 0,
            }
        ]
        self.conversations = [
            {
                "id": "chat-1",
                "workspace_id": "workspace-1",
                "scope_type": "workspace",
                "title": "Workspace chat",
                "last_modified": "2026-06-07T12:04:00Z",
                "deleted": 0,
                "version": 1,
            }
        ]
        self.memberships: dict[tuple[str, str, str], dict[str, object]] = {}
        self.backfill_list_calls = 0
        self._clock = 0

    def _timestamp(self) -> str:
        self._clock += 1
        return f"2026-06-07T12:30:{self._clock:02d}Z"

    def get_workspace(self, workspace_id: str) -> dict[str, object] | None:
        return self.workspaces.get(workspace_id)

    def list_workspace_sources(self, workspace_id: str) -> list[dict[str, object]]:
        self.backfill_list_calls += 1
        return [dict(row) for row in self.sources if row["workspace_id"] == workspace_id]

    def get_workspace_source(self, workspace_id: str, source_id: str) -> dict[str, object] | None:
        return next(
            (
                dict(row)
                for row in self.sources
                if row["workspace_id"] == workspace_id and str(row["id"]) == source_id
            ),
            None,
        )

    def list_workspace_artifacts(self, workspace_id: str) -> list[dict[str, object]]:
        self.backfill_list_calls += 1
        return [dict(row) for row in self.artifacts if row["workspace_id"] == workspace_id]

    def get_workspace_artifact(self, workspace_id: str, artifact_id: str) -> dict[str, object] | None:
        return next(
            (
                dict(row)
                for row in self.artifacts
                if row["workspace_id"] == workspace_id and str(row["id"]) == artifact_id
            ),
            None,
        )

    def list_workspace_notes(self, workspace_id: str) -> list[dict[str, object]]:
        self.backfill_list_calls += 1
        return [dict(row) for row in self.notes if row["workspace_id"] == workspace_id and not row.get("deleted")]

    def get_workspace_note(self, workspace_id: str, note_id: int) -> dict[str, object] | None:
        return next(
            (
                dict(row)
                for row in self.notes
                if row["workspace_id"] == workspace_id and int(row["id"]) == note_id and not row.get("deleted")
            ),
            None,
        )

    def search_conversations(
        self,
        query: str | None,
        *,
        scope_type: str | None = None,
        workspace_id: str | None = None,
    ) -> list[dict[str, object]]:
        assert query is None
        self.backfill_list_calls += 1
        return [
            dict(row)
            for row in self.conversations
            if row.get("scope_type") == scope_type
            and row.get("workspace_id") == workspace_id
            and not row.get("deleted")
        ]

    def get_conversation_for_workspace_membership(self, conversation_id: str) -> dict[str, object] | None:
        return next(
            (
                dict(row)
                for row in self.conversations
                if str(row["id"]) == conversation_id and not row.get("deleted")
            ),
            None,
        )

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
            if existing.get("deleted") and data.get("restore_deleted") is True:
                existing.update(
                    {
                        "role": data.get("role", "member"),
                        "label": data.get("label"),
                        "transfer_policy": data.get("transfer_policy", "link"),
                        "provenance": data.get("provenance", {}),
                        "metadata": data.get("metadata", {}),
                        "updated_at": self._timestamp(),
                        "updated_by_user_id": user_id,
                        "version": int(existing.get("version", 1)) + 1,
                        "deleted": False,
                    }
                )
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

    def workspace_resource_membership_summary(self, workspace_id: str) -> dict[str, object]:
        rows = [
            row
            for (row_workspace_id, _, _), row in self.memberships.items()
            if row_workspace_id == workspace_id and not row.get("deleted")
        ]
        by_resource_type: dict[str, int] = {}
        by_role: dict[str, int] = {}
        for row in rows:
            by_resource_type[str(row["resource_type"])] = by_resource_type.get(str(row["resource_type"]), 0) + 1
            by_role[str(row["role"])] = by_role.get(str(row["role"]), 0) + 1
        return {
            "total": len(rows),
            "by_resource_type": dict(sorted(by_resource_type.items())),
            "by_role": dict(sorted(by_role.items())),
        }


class ContextFakeDB(BackfillFakeDB):
    def __init__(self) -> None:
        super().__init__()
        self.sources = []
        self.artifacts = []
        self.notes = []
        self.conversations = []

    def get_workspace_primary_root(self, workspace_id: str) -> None:
        _ = workspace_id
        return None

    def list_active_workspace_operations(self, workspace_id: str) -> list[dict[str, object]]:
        _ = workspace_id
        return []


def _workspace_row(workspace_id: str) -> dict[str, object]:
    return {
        "id": workspace_id,
        "name": "Workspace",
        "archived": False,
        "study_materials_policy": "general",
        "workspace_profile": "research",
        "deleted": False,
        "created_at": "2026-06-07T12:00:00Z",
        "last_modified": "2026-06-07T12:00:00Z",
        "version": 1,
    }


def _membership_row(
    workspace_id: str,
    resource_type: str,
    resource_id: str,
    role: str,
) -> dict[str, object]:
    return {
        "workspace_id": workspace_id,
        "resource_type": resource_type,
        "resource_id": resource_id,
        "role": role,
        "label": None,
        "transfer_policy": "link",
        "provenance": {"source_surface": "test"},
        "metadata": {},
        "created_at": "2026-06-07T12:00:00Z",
        "updated_at": "2026-06-07T12:00:00Z",
        "version": 1,
        "deleted": False,
    }


@pytest.fixture
def media_db(monkeypatch: pytest.MonkeyPatch) -> object:
    media_db_instance = object()

    def fake_get_media_by_id(
        passed_db: object,
        media_id: int,
        *,
        include_deleted: bool = False,
        include_trash: bool = False,
    ) -> dict[str, object]:
        _ = include_deleted, include_trash
        assert passed_db is media_db_instance
        if media_id == 42:
            return {
                "id": 42,
                "title": "Media title",
                "type": "pdf",
                "last_modified": "2026-06-07T12:05:00Z",
            }
        raise RuntimeError("media missing")

    monkeypatch.setattr(membership_adapters.media_db_api, "get_media_by_id", fake_get_media_by_id)
    return media_db_instance


def test_backfill_creates_memberships_from_existing_workspace_rows(media_db: object) -> None:
    db = BackfillFakeDB()
    service = WorkspaceMembershipService(db)

    result = service.backfill_workspace_memberships("workspace-1", user_id="user-1", media_db=media_db)

    assert result["status"] == "complete"
    assert result["created"] == 5
    assert result["existing"] == 0
    assert result["restored"] == 0
    assert result["skipped"] == 0
    assert result["errors"] == []
    assert set(db.memberships) == {
        ("workspace-1", "workspace_source", "source-1"),
        ("workspace-1", "media", "42"),
        ("workspace-1", "workspace_artifact", "artifact-1"),
        ("workspace-1", "workspace_note", "7"),
        ("workspace-1", "chat", "chat-1"),
    }
    assert db.memberships[("workspace-1", "workspace_source", "source-1")]["role"] == "source"
    assert db.memberships[("workspace-1", "media", "42")]["role"] == "source"
    assert db.memberships[("workspace-1", "workspace_artifact", "artifact-1")]["role"] == "artifact"
    assert db.memberships[("workspace-1", "workspace_note", "7")]["role"] == "reference"
    assert db.memberships[("workspace-1", "chat", "chat-1")]["role"] == "conversation"
    assert result["summary"]["by_resource_type"] == {
        "chat": 1,
        "media": 1,
        "workspace_artifact": 1,
        "workspace_note": 1,
        "workspace_source": 1,
    }


def test_backfill_is_idempotent(media_db: object) -> None:
    db = BackfillFakeDB()
    service = WorkspaceMembershipService(db)

    first = service.backfill_workspace_memberships("workspace-1", user_id="user-1", media_db=media_db)
    second = service.backfill_workspace_memberships("workspace-1", user_id="user-1", media_db=media_db)

    assert first["created"] == 5
    assert first["restored"] == 0
    assert second["created"] == 0
    assert second["existing"] == 5
    assert second["restored"] == 0
    assert len(db.memberships) == 5


def test_backfill_restores_soft_deleted_membership_without_reporting_created(media_db: object) -> None:
    db = BackfillFakeDB()
    db.memberships[("workspace-1", "workspace_source", "source-1")] = {
        "workspace_id": "workspace-1",
        "resource_type": "workspace_source",
        "resource_id": "source-1",
        "role": "source",
        "label": "Old source label",
        "transfer_policy": "link",
        "provenance": {"source_surface": "previous"},
        "metadata": {},
        "created_at": "2026-06-07T11:00:00Z",
        "updated_at": "2026-06-07T11:30:00Z",
        "version": 2,
        "deleted": True,
    }
    service = WorkspaceMembershipService(db)

    result = service.backfill_workspace_memberships("workspace-1", user_id="user-1", media_db=media_db)

    assert result["created"] == 4
    assert result["existing"] == 0
    assert result["restored"] == 1
    restored = db.memberships[("workspace-1", "workspace_source", "source-1")]
    assert restored["deleted"] is False
    assert restored["version"] == 3


def test_backfill_rejects_archived_workspace_before_listing_rows(media_db: object) -> None:
    db = BackfillFakeDB()
    db.workspaces["workspace-1"] = {**db.workspaces["workspace-1"], "archived": True}
    service = WorkspaceMembershipService(db)

    with pytest.raises(WorkspaceMembershipServiceError) as exc_info:
        service.backfill_workspace_memberships("workspace-1", user_id="user-1", media_db=media_db)

    assert exc_info.value.code == "workspace_archived"
    assert db.backfill_list_calls == 0
    assert db.memberships == {}


def test_backfill_reports_bounded_unresolved_rows_without_rewriting_resources(media_db: object) -> None:
    db = BackfillFakeDB()
    db.sources = [
        {
            "id": "source-missing-media",
            "workspace_id": "workspace-1",
            "media_id": 999,
            "title": "Source with missing media",
            "source_type": "pdf",
        }
    ]
    db.notes = [
        {
            "id": f"bad-note-{index}",
            "workspace_id": "workspace-1",
            "title": f"Bad note {index}",
            "deleted": 0,
        }
        for index in range(30)
    ]
    original_sources = [dict(row) for row in db.sources]
    original_notes = [dict(row) for row in db.notes]
    service = WorkspaceMembershipService(db)

    result = service.backfill_workspace_memberships("workspace-1", user_id="user-1", media_db=media_db)

    assert result["status"] == "partial"
    assert result["created"] == 3
    assert result["restored"] == 0
    assert len(result["errors"]) == 25
    assert result["errors"][0].keys() == {"resource_type", "resource_id", "code", "message"}
    assert all("/" not in error["message"] for error in result["errors"])
    assert db.sources == original_sources
    assert db.notes == original_notes
    assert ("workspace-1", "workspace_source", "source-missing-media") in db.memberships
    assert ("workspace-1", "media", "999") not in db.memberships


def test_context_returns_compact_membership_totals_without_membership_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = ContextFakeDB()
    db.memberships[("workspace-1", "media", "42")] = _membership_row("workspace-1", "media", "42", "source")
    db.memberships[("workspace-1", "workspace_note", "7")] = _membership_row(
        "workspace-1",
        "workspace_note",
        "7",
        "reference",
    )
    app = _context_app(db, monkeypatch)

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/api/v1/workspaces/workspace-1/context")

    assert response.status_code == 200, response.text
    memberships = response.json()["memberships"]
    assert memberships == {
        "total": 2,
        "by_resource_type": {"media": 1, "workspace_note": 1},
        "by_role": {"reference": 1, "source": 1},
    }
    assert "items" not in memberships


def test_context_membership_summary_failure_returns_partial_error_and_empty_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = ContextFakeDB()
    app = _context_app(db, monkeypatch)

    def fail_summary(self: WorkspaceMembershipService, workspace_id: str) -> dict[str, object]:
        _ = self, workspace_id
        raise RuntimeError("summary backend unavailable at /tmp/secret")

    monkeypatch.setattr(WorkspaceMembershipService, "workspace_membership_summary", fail_summary)

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/api/v1/workspaces/workspace-1/context")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["memberships"] == {"total": 0, "by_resource_type": {}, "by_role": {}}
    assert {
        "scope": "memberships",
        "code": "membership_summary_unavailable",
        "message": "Workspace membership summary is unavailable.",
    } in body["partial_errors"]


def test_context_mcp_permissions_are_not_derived_from_membership_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = ContextFakeDB()
    for index in range(10):
        db.memberships[("workspace-1", "chat", f"chat-{index}")] = _membership_row(
            "workspace-1",
            "chat",
            f"chat-{index}",
            "conversation",
        )
    app = _context_app(db, monkeypatch)

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/api/v1/workspaces/workspace-1/context")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["memberships"]["total"] == 10
    assert body["allowed_actions"]["run_mcp_tools"] == {
        "allowed": False,
        "reason_code": "mcp_not_configured",
    }
    assert body["services"]["mcp"] == {
        "state": "not_configured",
        "reason_code": "no_workspace_mcp_binding",
        "management_surface": "mcp_hub",
    }


def _context_app(db: ContextFakeDB, monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    app = FastAPI()
    app.include_router(workspaces_endpoint.router, prefix="/api/v1/workspaces")

    async def allow_rate_limit() -> None:
        return None

    async def fake_capabilities(*, workspace_id: str, user_id: str | int | None) -> dict[str, object]:
        _ = workspace_id, user_id
        return {
            "workspace_services": {},
            "allowed_actions": {},
        }

    monkeypatch.setattr(workspaces_endpoint, "collect_workspace_service_capabilities", fake_capabilities)
    app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id="user-1")
    app.dependency_overrides[get_chacha_db_for_user] = lambda: db
    app.dependency_overrides[try_get_media_db_for_user] = lambda: None
    app.dependency_overrides[workspaces_endpoint.try_get_workspace_job_manager] = lambda: None
    app.dependency_overrides[WORKSPACES_READ_RATE_LIMIT] = allow_rate_limit
    return app
