"""Integration tests for the sharing API endpoints."""
from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import Field

from tldw_Server_API.app.api.v1.endpoints.sharing import router
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User

pytestmark = pytest.mark.integration


class _TestUser(User):
    """Subclass with team/org membership for testing shared-with-me."""
    team_ids: list[int] = Field(default_factory=list)
    org_ids: list[int] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}


@pytest.fixture
def test_user():
    return _TestUser(
        id=1, username="alice", email="alice@test.com",
        password_hash="hash", team_ids=[10], org_ids=[5],
    )


@pytest.fixture
def test_app(test_user):
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")

    async def _fake_user():
        return test_user

    from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user
    app.dependency_overrides[get_request_user] = _fake_user

    return app


@pytest.fixture
def client(test_app):
    return TestClient(test_app)


@pytest.fixture
def mock_repo(repo, tmp_path):
    """Patch repo and security helpers while keeping the real audit service wiring."""
    async def _noop_verify(*args, **kwargs):
        pass

    def _no_rate_limit(scope: str):
        async def _noop():
            pass
        return _noop

    shared_audit_path = tmp_path / "audit_shared.db"

    with patch("tldw_Server_API.app.api.v1.endpoints.sharing._get_repo", return_value=repo), \
         patch("tldw_Server_API.app.api.v1.endpoints.sharing.rbac_rate_limit", _no_rate_limit), \
         patch("tldw_Server_API.app.api.v1.endpoints.sharing._verify_workspace_ownership", _noop_verify), \
         patch("tldw_Server_API.app.api.v1.endpoints.sharing._validate_user_has_share_access", _noop_verify), \
         patch("tldw_Server_API.app.api.v1.endpoints.sharing._get_token_service") as mock_ts, \
         patch("tldw_Server_API.app.core.DB_Management.db_path_utils.DatabasePaths.get_shared_audit_db_path", return_value=shared_audit_path):
        from tldw_Server_API.app.core.Sharing.share_token_service import ShareTokenService
        mock_ts.return_value = ShareTokenService(repo)
        yield repo


class TestWorkspaceSharing:
    def test_share_workspace(self, client, mock_repo):
        resp = client.post("/api/v1/sharing/workspaces/ws-1/share", json={
            "share_scope_type": "team",
            "share_scope_id": 10,
            "access_level": "view_chat",
            "allow_clone": True,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["workspace_id"] == "ws-1"
        assert data["access_level"] == "view_chat"

    def test_share_workspace_duplicate(self, client, mock_repo):
        client.post("/api/v1/sharing/workspaces/ws-dup/share", json={
            "share_scope_type": "team",
            "share_scope_id": 10,
        })
        resp = client.post("/api/v1/sharing/workspaces/ws-dup/share", json={
            "share_scope_type": "team",
            "share_scope_id": 10,
        })
        assert resp.status_code == 409

    def test_share_workspace_generic_failure_log_is_sanitized(
        self,
        client,
        mock_repo,
        monkeypatch,
    ):
        from tldw_Server_API.app.api.v1.endpoints import sharing

        async def _fail_create_share(*args, **kwargs):
            raise RuntimeError("share backend exploded at /private/sharing.db")

        fake_logger = MagicMock()
        monkeypatch.setattr(mock_repo, "create_share", _fail_create_share)
        monkeypatch.setattr(sharing, "logger", fake_logger)

        resp = client.post("/api/v1/sharing/workspaces/private-ws/share", json={
            "share_scope_type": "team",
            "share_scope_id": 10,
        })

        assert resp.status_code == 500
        assert resp.json()["detail"] == "An internal error occurred while creating the share."
        fake_logger.error.assert_called_once_with("Failed to create share")

    def test_list_workspace_shares(self, client, mock_repo):
        client.post("/api/v1/sharing/workspaces/ws-list/share", json={
            "share_scope_type": "team",
            "share_scope_id": 10,
        })
        resp = client.get("/api/v1/sharing/workspaces/ws-list/shares")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1

    def test_update_share(self, client, mock_repo):
        create = client.post("/api/v1/sharing/workspaces/ws-upd/share", json={
            "share_scope_type": "team",
            "share_scope_id": 10,
            "access_level": "view_chat",
        })
        share_id = create.json()["id"]
        resp = client.patch(f"/api/v1/sharing/shares/{share_id}", json={
            "access_level": "full_edit",
        })
        assert resp.status_code == 200
        assert resp.json()["access_level"] == "full_edit"

    def test_revoke_share(self, client, mock_repo):
        create = client.post("/api/v1/sharing/workspaces/ws-rev/share", json={
            "share_scope_type": "team",
            "share_scope_id": 10,
        })
        share_id = create.json()["id"]
        resp = client.delete(f"/api/v1/sharing/shares/{share_id}")
        assert resp.status_code == 200

    def test_update_nonexistent_share(self, client, mock_repo):
        resp = client.patch("/api/v1/sharing/shares/9999", json={
            "access_level": "full_edit",
        })
        assert resp.status_code == 404


class TestSharedWithMe:
    def test_shared_with_me_empty(self, client, mock_repo):
        resp = client.get("/api/v1/sharing/shared-with-me")
        assert resp.status_code == 200
        assert resp.json()["total"] == 0

    def test_get_shared_workspace(self, client, mock_repo):
        create = client.post("/api/v1/sharing/workspaces/ws-view/share", json={
            "share_scope_type": "team",
            "share_scope_id": 10,
        })
        share_id = create.json()["id"]
        resp = client.get(f"/api/v1/sharing/shared-with-me/{share_id}/workspace")
        assert resp.status_code == 200

    def test_get_shared_workspace_media_releases_owner_session(self, client, mock_repo):
        create = client.post("/api/v1/sharing/workspaces/ws-media/share", json={
            "share_scope_type": "team",
            "share_scope_id": 10,
        })
        share_id = create.json()["id"]
        events: list[str] = []

        class _FakeChaCha:
            def list_workspace_sources(self, workspace_id: str):
                assert workspace_id == "ws-media"
                return [{"media_id": 123}]

        class _FakeMediaDb:
            def get_media_by_id(self, media_id: int):
                assert media_id == 123
                return {
                    "id": 123,
                    "title": "Shared Item",
                    "url": "https://example.com/shared",
                    "type": "article",
                    "content": "shared content",
                    "author": "author",
                    "ingestion_date": "2025-01-01T00:00:00",
                }

        @contextmanager
        def _managed_media_db_for_owner(owner_user_id: int):
            assert owner_user_id == 1
            events.append("enter")
            try:
                yield _FakeMediaDb()
            finally:
                events.append("exit")

        with patch(
            "tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps.get_chacha_db_for_owner",
            return_value=_FakeChaCha(),
        ), patch(
            "tldw_Server_API.app.api.v1.API_Deps.DB_Deps.managed_media_db_for_owner",
            _managed_media_db_for_owner,
        ):
            resp = client.get(f"/api/v1/sharing/shared-with-me/{share_id}/media/123")

        assert resp.status_code == 200
        assert resp.json()["id"] == 123
        assert events == ["enter", "exit"]


class TestClone:
    def test_clone_shared_workspace(self, client, mock_repo):
        create = client.post("/api/v1/sharing/workspaces/ws-clone/share", json={
            "share_scope_type": "team",
            "share_scope_id": 10,
            "allow_clone": True,
        })
        share_id = create.json()["id"]
        resp = client.post(f"/api/v1/sharing/shared-with-me/{share_id}/clone", json={
            "new_name": "My Clone",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "pending"
        assert "job_id" in data

    def test_clone_not_allowed(self, client, mock_repo):
        create = client.post("/api/v1/sharing/workspaces/ws-noclone/share", json={
            "share_scope_type": "team",
            "share_scope_id": 10,
            "allow_clone": False,
        })
        share_id = create.json()["id"]
        resp = client.post(f"/api/v1/sharing/shared-with-me/{share_id}/clone", json={})
        assert resp.status_code == 403

    def test_clone_nonexistent_share(self, client, mock_repo):
        resp = client.post("/api/v1/sharing/shared-with-me/9999/clone", json={})
        assert resp.status_code == 404


class TestShareTokens:
    def test_create_token(self, client, mock_repo):
        resp = client.post("/api/v1/sharing/tokens", json={
            "resource_type": "workspace",
            "resource_id": "ws-1",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "raw_token" in data
        assert data["resource_type"] == "workspace"

    def test_list_tokens(self, client, mock_repo):
        client.post("/api/v1/sharing/tokens", json={
            "resource_type": "workspace",
            "resource_id": "ws-1",
        })
        resp = client.get("/api/v1/sharing/tokens")
        assert resp.status_code == 200
        assert resp.json()["total"] >= 1

    def test_revoke_token(self, client, mock_repo):
        create = client.post("/api/v1/sharing/tokens", json={
            "resource_type": "workspace",
            "resource_id": "ws-1",
        })
        token_id = create.json()["id"]
        resp = client.delete(f"/api/v1/sharing/tokens/{token_id}")
        assert resp.status_code == 200


class TestPublicEndpoints:
    def test_public_preview_valid(self, client, mock_repo):
        create = client.post("/api/v1/sharing/tokens", json={
            "resource_type": "workspace",
            "resource_id": "ws-1",
        })
        raw_token = create.json()["raw_token"]
        resp = client.get(f"/api/v1/sharing/public/{raw_token}")
        assert resp.status_code == 200
        assert resp.json()["resource_type"] == "workspace"

    def test_public_preview_invalid(self, client, mock_repo):
        resp = client.get("/api/v1/sharing/public/not-a-valid-token-here-12345678")
        assert resp.status_code == 404

    def test_password_verify_success(self, client, mock_repo):
        create = client.post("/api/v1/sharing/tokens", json={
            "resource_type": "workspace",
            "resource_id": "ws-1",
            "password": "mysecret",
        })
        raw_token = create.json()["raw_token"]
        resp = client.post(f"/api/v1/sharing/public/{raw_token}/verify", json={
            "password": "mysecret",
        })
        assert resp.status_code == 200
        assert resp.json()["verified"] is True

    def test_password_verify_failure(self, client, mock_repo):
        create = client.post("/api/v1/sharing/tokens", json={
            "resource_type": "workspace",
            "resource_id": "ws-1",
            "password": "mysecret",
        })
        raw_token = create.json()["raw_token"]
        resp = client.post(f"/api/v1/sharing/public/{raw_token}/verify", json={
            "password": "wrongpassword",
        })
        assert resp.status_code == 403

    def test_public_import(self, client, mock_repo):
        create = client.post("/api/v1/sharing/tokens", json={
            "resource_type": "workspace",
            "resource_id": "ws-1",
        })
        raw_token = create.json()["raw_token"]
        resp = client.post(f"/api/v1/sharing/public/{raw_token}/import")
        assert resp.status_code == 200
        assert resp.json()["resource_id"] == "ws-1"

    def test_public_import_blocked_when_password_protected(self, client, mock_repo):
        """Password-protected tokens cannot be imported without verification."""
        create = client.post("/api/v1/sharing/tokens", json={
            "resource_type": "workspace",
            "resource_id": "ws-pw",
            "password": "secret123",
        })
        raw_token = create.json()["raw_token"]
        resp = client.post(f"/api/v1/sharing/public/{raw_token}/import")
        assert resp.status_code == 403
        assert "Password verification required" in resp.json()["detail"]


class TestAdmin:
    def test_admin_list_shares(self, client, mock_repo):
        client.post("/api/v1/sharing/workspaces/ws-admin/share", json={
            "share_scope_type": "team",
            "share_scope_id": 10,
        })
        resp = client.get("/api/v1/sharing/admin/shares")
        assert resp.status_code == 200
        assert resp.json()["total"] >= 1

    def test_admin_update_config(self, client, mock_repo):
        resp = client.patch("/api/v1/sharing/admin/config", json={
            "config": {"default_access_level": "view_chat"},
        })
        assert resp.status_code == 200

    def test_admin_audit_log(self, client, mock_repo):
        resp = client.get("/api/v1/sharing/admin/audit")
        assert resp.status_code == 200
        assert "events" in resp.json()

    def test_admin_audit_log_returns_unified_backed_rows(self, client, mock_repo, sharing_db):
        create = client.post("/api/v1/sharing/workspaces/ws-1/share", json={
            "share_scope_type": "team",
            "share_scope_id": 10,
        })
        assert create.status_code == 200

        sharing_db.execute("DELETE FROM share_audit_log")
        sharing_db.commit()

        resp = client.get("/api/v1/sharing/admin/audit")
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["total"] >= 1
        assert any(
            isinstance(event.get("id"), int)
            and str(event.get("event_type", "")).startswith("share.")
            for event in payload["events"]
        )
