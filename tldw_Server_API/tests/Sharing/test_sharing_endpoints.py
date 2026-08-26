"""Integration tests for the sharing API endpoints."""
from __future__ import annotations

import asyncio
import builtins
import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient
from pydantic import Field

from tldw_Server_API.app.api.v1.endpoints import sharing as sharing_endpoints
from tldw_Server_API.app.api.v1.endpoints.sharing import router
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Sharing.shared_workspace_access_service import (
    SharedWorkspaceAccessService,
)

pytestmark = pytest.mark.integration


async def _resolve_factory(value):
    if inspect.isawaitable(value):
        return await value
    return value


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
def clone_client(test_app, mock_repo, monkeypatch, tmp_path):
    class _UsersRepo:
        async def get_user_by_id(self, user_id: int):
            assert user_id == 1
            return {"id": user_id, "username": "alice"}

    class _OwnerDb:
        def get_workspace(self, workspace_id: str):
            return {"id": workspace_id, "name": workspace_id}

    async def _owner_db(owner_user_id: int):
        assert owner_user_id == 1
        return _OwnerDb()

    monkeypatch.setenv("JOBS_ALLOWED_QUEUES_SHARING", "workspace-clone")
    monkeypatch.setenv("JOBS_METRICS_GAUGES_ENABLED", "false")
    manager = JobManager(tmp_path / "sharing-clone-jobs.db")
    access_service = SharedWorkspaceAccessService(
        mock_repo,
        _UsersRepo(),
        _owner_db,
    )
    test_app.dependency_overrides[
        sharing_endpoints.try_get_job_manager
    ] = lambda: manager
    test_app.dependency_overrides[
        sharing_endpoints.get_shared_workspace_access_service
    ] = lambda: access_service
    return TestClient(test_app), manager


@pytest.mark.asyncio
async def test_verify_workspace_ownership_multi_user_failure_log_is_sanitized(monkeypatch):
    from tldw_Server_API.app.api.v1.API_Deps import ChaCha_Notes_DB_Deps as chacha_deps
    from tldw_Server_API.app.api.v1.endpoints import sharing
    from tldw_Server_API.app.core.AuthNZ import settings as auth_settings

    async def _fail_get_chacha_db_for_user_id(user_id: int):
        assert user_id == 1
        raise RuntimeError("workspace DB exploded at /private/workspaces.db")

    fake_logger = MagicMock()
    monkeypatch.setattr(chacha_deps, "get_chacha_db_for_user_id", _fail_get_chacha_db_for_user_id)
    monkeypatch.setattr(auth_settings, "get_settings", lambda: SimpleNamespace(auth_mode="multi_user"))
    monkeypatch.setattr(sharing, "logger", fake_logger)

    with pytest.raises(HTTPException) as exc_info:
        await sharing._verify_workspace_ownership("private-ws", SimpleNamespace(id=1))

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "Could not verify workspace ownership due to a database error."
    fake_logger.error.assert_called_once_with("Workspace ownership check failed")


@pytest.mark.asyncio
async def test_verify_workspace_ownership_single_user_skip_log_is_sanitized(monkeypatch):
    from tldw_Server_API.app.api.v1.API_Deps import ChaCha_Notes_DB_Deps as chacha_deps
    from tldw_Server_API.app.api.v1.endpoints import sharing
    from tldw_Server_API.app.core.AuthNZ import settings as auth_settings

    async def _fail_get_chacha_db_for_user_id(user_id: int):
        assert user_id == 1
        raise RuntimeError("workspace DB exploded at /private/workspaces.db")

    fake_logger = MagicMock()
    monkeypatch.setattr(chacha_deps, "get_chacha_db_for_user_id", _fail_get_chacha_db_for_user_id)
    monkeypatch.setattr(auth_settings, "get_settings", lambda: SimpleNamespace(auth_mode="single_user"))
    monkeypatch.setattr(sharing, "logger", fake_logger)

    await sharing._verify_workspace_ownership("private-ws", SimpleNamespace(id=1))

    fake_logger.warning.assert_called_once_with("Workspace ownership check skipped in single-user mode")


@pytest.mark.asyncio
async def test_shared_with_me_workspace_name_preload_log_is_sanitized(
    repo,
    test_user,
    monkeypatch,
):
    from tldw_Server_API.app.api.v1.API_Deps import ChaCha_Notes_DB_Deps as chacha_deps
    from tldw_Server_API.app.api.v1.endpoints import sharing

    await repo.create_share(
        workspace_id="private-ws",
        owner_user_id=2,
        share_scope_type="team",
        share_scope_id=10,
        created_by=2,
    )

    async def _fail_get_chacha_db_for_owner(owner_user_id: int):
        assert owner_user_id == 2
        raise RuntimeError("owner workspace DB exploded at /private/owner-workspaces.db")

    fake_logger = MagicMock()
    monkeypatch.setattr(sharing, "_get_repo", lambda: repo)
    monkeypatch.setattr(chacha_deps, "get_chacha_db_for_owner", _fail_get_chacha_db_for_owner)
    monkeypatch.setattr(sharing, "logger", fake_logger)

    response = await sharing.shared_with_me(user=test_user)

    assert response.total == 1
    fake_logger.debug.assert_called_once_with("Skipping shared workspace name preload")


@pytest.mark.asyncio
async def test_shared_with_me_uses_authoritative_active_share_listing(
    test_user,
    monkeypatch,
):
    from tldw_Server_API.app.api.v1.API_Deps import ChaCha_Notes_DB_Deps as chacha_deps
    from tldw_Server_API.app.api.v1.endpoints import sharing

    events: list[str] = []

    class _Repo:
        async def list_active_shares_for_user(self, user_id: int):
            events.append(f"list:{user_id}")
            return [
                {
                    "id": 44,
                    "workspace_id": "authoritative-workspace",
                    "owner_user_id": 2,
                    "access_level": "view_chat",
                    "allow_clone": False,
                    "created_at": "2026-08-20T18:00:00+00:00",
                }
            ]

        async def list_shares_for_scope(self, *_args, **_kwargs):
            raise AssertionError("claim-loop discovery must not be used")

    class _OwnerDb:
        def get_workspace(self, workspace_id: str):
            events.append(f"workspace:{workspace_id}")
            return {"name": "Authoritative workspace"}

    async def _owner_db(owner_user_id: int):
        events.append(f"owner-db:{owner_user_id}")
        return _OwnerDb()

    monkeypatch.setattr(sharing, "_get_repo", lambda: _Repo())
    monkeypatch.setattr(chacha_deps, "get_chacha_db_for_owner", _owner_db)

    response = await sharing.shared_with_me(user=test_user)

    assert response.model_dump(mode="json") == {
        "items": [
            {
                "share_id": 44,
                "workspace_id": "authoritative-workspace",
                "workspace_name": "Authoritative workspace",
                "owner_user_id": 2,
                "owner_username": None,
                "access_level": "view_chat",
                "allow_clone": False,
                "shared_at": "2026-08-20T18:00:00+00:00",
            }
        ],
        "total": 1,
    }
    assert events == [
        "list:1",
        "owner-db:2",
        "workspace:authoritative-workspace",
    ]


@pytest.mark.asyncio
async def test_verify_chatbook_ownership_uses_storage_user_id_for_numeric_users(monkeypatch):
    from tldw_Server_API.app.api.v1.API_Deps import ChaCha_Notes_DB_Deps as chacha_deps
    from tldw_Server_API.app.api.v1.endpoints import sharing

    fake_db = object()
    captured: dict[str, object] = {}

    async def _get_chacha_db_for_user_id(user_id: int):
        captured["db_user_id"] = user_id
        return fake_db

    def _owned_sync(
        *,
        normalized_id: str,
        user_id: int | str,
        user_id_int: int | None,
        db: object | None,
    ) -> bool:
        captured["normalized_id"] = normalized_id
        captured["helper_user_id"] = user_id
        captured["helper_user_id_int"] = user_id_int
        captured["helper_db"] = db
        return True

    monkeypatch.setattr(chacha_deps, "get_chacha_db_for_user_id", _get_chacha_db_for_user_id)
    monkeypatch.setattr(sharing, "_chatbook_ownership_exists_sync", _owned_sync)

    await sharing._verify_chatbook_ownership(
        "export-123",
        SimpleNamespace(id="1", id_int=1),
    )

    assert captured == {
        "db_user_id": 1,
        "normalized_id": "export-123",
        "helper_user_id": 1,
        "helper_user_id_int": 1,
        "helper_db": fake_db,
    }


@pytest.mark.asyncio
async def test_validate_share_target_scope_rejects_invalid_scope_id():
    from tldw_Server_API.app.api.v1.endpoints import sharing

    body = SimpleNamespace(
        share_scope_id="not-an-int",
        share_scope_type=SimpleNamespace(value="team"),
    )

    with pytest.raises(HTTPException) as exc_info:
        await sharing._validate_share_target_scope(
            body,
            SimpleNamespace(team_ids=[1], org_ids=[]),
        )

    assert exc_info.value.status_code == 422
    assert exc_info.value.detail == "Invalid share scope ID."


@pytest.mark.asyncio
async def test_shared_with_me_workspace_name_resolution_log_is_sanitized(
    repo,
    test_user,
    monkeypatch,
):
    from tldw_Server_API.app.api.v1.API_Deps import ChaCha_Notes_DB_Deps as chacha_deps
    from tldw_Server_API.app.api.v1.endpoints import sharing

    class _FailingWorkspaceDb:
        def get_workspace(self, workspace_id: str):
            assert workspace_id == "private-ws"
            raise RuntimeError("workspace lookup exploded at /private/owner-workspaces.db")

    await repo.create_share(
        workspace_id="private-ws",
        owner_user_id=2,
        share_scope_type="team",
        share_scope_id=10,
        created_by=2,
    )

    async def _get_chacha_db_for_owner(owner_user_id: int):
        assert owner_user_id == 2
        return _FailingWorkspaceDb()

    fake_logger = MagicMock()
    monkeypatch.setattr(sharing, "_get_repo", lambda: repo)
    monkeypatch.setattr(chacha_deps, "get_chacha_db_for_owner", _get_chacha_db_for_owner)
    monkeypatch.setattr(sharing, "logger", fake_logger)

    response = await sharing.shared_with_me(user=test_user)

    assert response.total == 1
    fake_logger.debug.assert_called_once_with("Failed to resolve shared workspace name")


@pytest.mark.asyncio
async def test_shared_with_me_workspace_name_population_log_is_sanitized(
    repo,
    test_user,
    monkeypatch,
):
    from tldw_Server_API.app.api.v1.endpoints import sharing

    await repo.create_share(
        workspace_id="private-ws",
        owner_user_id=2,
        share_scope_type="team",
        share_scope_id=10,
        created_by=2,
    )

    real_import = builtins.__import__

    def fail_chacha_owner_import(name, globals=None, locals=None, fromlist=(), level=0):
        if "ChaCha_Notes_DB_Deps" in name:
            raise RuntimeError("workspace name import exploded at /private/owner-workspaces.db")
        return real_import(name, globals, locals, fromlist, level)

    fake_logger = MagicMock()
    monkeypatch.setattr(sharing, "_get_repo", lambda: repo)
    monkeypatch.setattr(builtins, "__import__", fail_chacha_owner_import)
    monkeypatch.setattr(sharing, "logger", fake_logger)

    response = await sharing.shared_with_me(user=test_user)

    assert response.total == 1
    fake_logger.debug.assert_called_once_with("Shared workspace name population skipped")


@pytest.mark.asyncio
async def test_admin_list_shares_includes_canonical_offset_pagination(monkeypatch):
    """Admin share listing should expose full totals and canonical offset metadata."""
    from tldw_Server_API.app.api.v1.endpoints import sharing

    class _FakeRepo:
        async def list_all_shares(self, *, limit: int, offset: int, include_revoked: bool):
            assert limit == 1
            assert offset == 1
            assert include_revoked is False
            return [
                {
                    "id": 2,
                    "workspace_id": "ws-2",
                    "owner_user_id": 1,
                    "share_scope_type": "team",
                    "share_scope_id": 20,
                    "access_level": "view_chat",
                    "allow_clone": True,
                    "created_by": 1,
                    "created_at": "2026-01-01T00:00:00Z",
                    "updated_at": "2026-01-01T00:00:00Z",
                    "revoked_at": None,
                    "is_revoked": False,
                }
            ]

        async def count_all_shares(self, *, include_revoked: bool):
            assert include_revoked is False
            return 3

    monkeypatch.setattr(sharing, "_get_repo", lambda: _FakeRepo())

    response = await sharing.admin_list_shares(
        limit=1,
        offset=1,
        include_revoked=False,
        user=SimpleNamespace(id=1),
    )

    assert response.total == 3
    assert response.limit == 1
    assert response.offset == 1
    assert response.has_more is True
    assert response.next_offset == 2
    assert response.pagination.total == 3
    assert response.pagination.has_more is True
    assert response.pagination.next_offset == 2


@pytest.mark.asyncio
async def test_admin_audit_log_includes_canonical_offset_pagination(monkeypatch):
    """Admin audit listing should expose full totals and canonical offset metadata."""
    from tldw_Server_API.app.api.v1.endpoints import sharing

    class _FakeAuditService:
        async def query(
            self,
            *,
            owner_user_id: int | None,
            resource_type: str | None,
            resource_id: str | None,
            limit: int,
            offset: int,
        ):
            assert owner_user_id == 1
            assert resource_type == "workspace"
            assert resource_id is None
            assert limit == 1
            assert offset == 1
            return [
                {
                    "id": 2,
                    "event_type": "share.created",
                    "actor_user_id": 1,
                    "resource_type": "workspace",
                    "resource_id": "ws-2",
                    "owner_user_id": 1,
                    "share_id": 2,
                    "token_id": None,
                    "metadata": {},
                    "ip_address": None,
                    "user_agent": None,
                    "created_at": "2026-01-01T00:00:00Z",
                }
            ]

        async def count(
            self,
            *,
            owner_user_id: int | None,
            resource_type: str | None,
            resource_id: str | None,
        ):
            assert owner_user_id == 1
            assert resource_type == "workspace"
            assert resource_id is None
            return 3

    monkeypatch.setattr(sharing, "_get_audit_service", lambda: _FakeAuditService())

    response = await sharing.admin_audit_log(
        owner_user_id=1,
        resource_type="workspace",
        resource_id=None,
        limit=1,
        offset=1,
        user=SimpleNamespace(id=1),
    )

    assert response.total == 3
    assert response.limit == 1
    assert response.offset == 1
    assert response.has_more is True
    assert response.next_offset == 2
    assert response.pagination.total == 3
    assert response.pagination.has_more is True
    assert response.pagination.next_offset == 2


@pytest.mark.asyncio
async def test_lazy_shared_repo_awaits_db_pool(monkeypatch):
    fake_pool = object()

    async def fake_get_db_pool():
        return fake_pool

    from tldw_Server_API.app.core.AuthNZ import database

    monkeypatch.setattr(database, "get_db_pool", fake_get_db_pool)

    repo = await _resolve_factory(sharing_endpoints._get_repo())
    db_pool = repo.db_pool
    if inspect.isawaitable(db_pool):
        await db_pool

    assert db_pool is fake_pool


@pytest.mark.asyncio
async def test_lazy_prototype_repo_awaits_db_pool(monkeypatch):
    fake_pool = object()

    async def fake_get_db_pool():
        return fake_pool

    from tldw_Server_API.app.core.AuthNZ import database

    monkeypatch.setattr(database, "get_db_pool", fake_get_db_pool)

    repo = await _resolve_factory(sharing_endpoints._get_prototype_repo())
    db_pool = repo.db_pool
    if inspect.isawaitable(db_pool):
        await db_pool

    assert db_pool is fake_pool


@pytest.fixture
def mock_repo(repo, tmp_path):
    """Patch repo and security helpers while keeping the real audit service wiring."""
    class _PrototypeRepoStub:
        async def get_workspace(self, prototype_workspace_id: str):
            return {"id": prototype_workspace_id, "owner_user_id": 1}

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
         patch("tldw_Server_API.app.api.v1.endpoints.sharing._get_prototype_repo", return_value=_PrototypeRepoStub()), \
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

    def test_share_workspace_accepts_current_user_org_scope(self, client, mock_repo):
        resp = client.post("/api/v1/sharing/workspaces/ws-org/share", json={
            "share_scope_type": "org",
            "share_scope_id": 5,
            "access_level": "view_chat_add",
            "allow_clone": False,
        })

        assert resp.status_code == 200
        data = resp.json()
        assert data["workspace_id"] == "ws-org"
        assert data["share_scope_type"] == "org"
        assert data["share_scope_id"] == 5
        assert data["access_level"] == "view_chat_add"
        assert data["allow_clone"] is False

    def test_share_workspace_rejects_team_scope_outside_current_user_membership(self, client, mock_repo):
        resp = client.post("/api/v1/sharing/workspaces/ws-unknown-team/share", json={
            "share_scope_type": "team",
            "share_scope_id": 999,
        })

        assert resp.status_code == 403
        assert resp.json()["detail"] == "You can only share with teams you belong to."

    def test_share_workspace_rejects_org_scope_outside_current_user_membership(self, client, mock_repo):
        resp = client.post("/api/v1/sharing/workspaces/ws-unknown-org/share", json={
            "share_scope_type": "org",
            "share_scope_id": 999,
        })

        assert resp.status_code == 403
        assert resp.json()["detail"] == "You can only share with organizations you belong to."

    def test_share_workspace_succeeds_when_audit_write_fails(
        self,
        client,
        mock_repo,
        monkeypatch,
    ):
        class _FailingAuditService:
            async def log(self, *args, **kwargs):
                raise RuntimeError("audit backend unavailable")

        monkeypatch.setattr(
            sharing_endpoints,
            "_get_audit_service",
            lambda: _FailingAuditService(),
        )

        resp = client.post("/api/v1/sharing/workspaces/ws-audit/share", json={
            "share_scope_type": "team",
            "share_scope_id": 10,
            "access_level": "view_chat",
            "allow_clone": True,
        })

        assert resp.status_code == 200
        assert resp.json()["workspace_id"] == "ws-audit"

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

    def test_get_shared_workspace(self, client, mock_repo, monkeypatch):
        create = client.post("/api/v1/sharing/workspaces/ws-view/share", json={
            "share_scope_type": "team",
            "share_scope_id": 10,
        })
        share_id = create.json()["id"]

        class _AccessService:
            async def resolve(self, *, share_id: int, recipient_user_id: int):
                return SimpleNamespace(
                    share_id=share_id,
                    workspace_id="ws-view",
                    owner_user_id=2,
                    recipient_user_id=recipient_user_id,
                    access_level="view_chat",
                    allow_clone=False,
                    owner_display_name="Workspace owner",
                    shared_at="2026-08-21T12:00:00+00:00",
                    workspace={"name": "Shared workspace", "description": ""},
                    policy_actions={
                        "inspect_sources": {"allowed": True, "reason_code": None},
                        "ask_grounded_questions": {"allowed": True, "reason_code": None},
                        "add_sources": {
                            "allowed": False,
                            "reason_code": "shared_write_not_available",
                        },
                        "edit_workspace": {
                            "allowed": False,
                            "reason_code": "shared_write_not_available",
                        },
                        "clone_workspace": {
                            "allowed": False,
                            "reason_code": "clone_deferred",
                        },
                    },
                )

        async def _sources(_context):
            return []

        async def _projection(_context, _sources):
            return {
                "sources": [],
                "summary": {"total": 0, "queryable": 0, "processing": 0, "failed": 0},
                "partial_errors": [],
            }

        async def _history(_context, *, before, limit):
            assert before is None
            assert limit == 30
            raise RuntimeError("empty test history")

        client.app.dependency_overrides[
            sharing_endpoints.get_shared_workspace_access_service
        ] = lambda: _AccessService()
        monkeypatch.setattr(sharing_endpoints, "_load_recipient_workspace_sources", _sources)
        monkeypatch.setattr(sharing_endpoints, "_project_recipient_source_status", _projection)
        monkeypatch.setattr(sharing_endpoints, "_load_recipient_chat_history", _history)
        resp = client.get(f"/api/v1/sharing/shared-with-me/{share_id}/workspace")

        assert resp.status_code == 200
        data = resp.json()
        assert data["sources"]["items"] == []
        assert data["allowed_actions"]["inspect_sources"]["allowed"] is True
        assert "owner_user_id" not in str(data)

    @pytest.mark.parametrize(
        "path",
        [
            "/api/v1/sharing/shared-with-me/12/media",
            "/api/v1/sharing/shared-with-me/12/media/99",
            "/api/v1/sharing/shared-with-me/12/full-media",
            "/api/v1/sharing/shared-with-me/12/full-media/99",
        ],
    )
    def test_removed_recipient_media_routes_are_plain_404s(
        self,
        client,
        mock_repo,
        path,
    ):
        response = client.get(path, follow_redirects=False)

        assert response.status_code == 404
        assert "location" not in response.headers

    def test_recipient_openapi_has_no_owner_media_operation(self, client, mock_repo):
        schema = client.get("/openapi.json").json()
        removed_paths = {
            "/api/v1/sharing/shared-with-me/{share_id}/media",
            "/api/v1/sharing/shared-with-me/{share_id}/media/{media_id}",
            "/api/v1/sharing/shared-with-me/{share_id}/full-media",
            "/api/v1/sharing/shared-with-me/{share_id}/full-media/{media_id}",
        }

        assert removed_paths.isdisjoint(schema["paths"])
        assert all(
            "SharedMediaResponse" not in operation.get("operationId", "")
            for path_item in schema["paths"].values()
            for method, operation in path_item.items()
            if method in {"get", "post", "put", "patch", "delete"}
        )
        assert "SharedMediaResponse" not in schema["components"]["schemas"]


class TestClone:
    def test_clone_shared_workspace(self, clone_client, mock_repo):
        client, manager = clone_client
        create = client.post("/api/v1/sharing/workspaces/ws-clone/share", json={
            "share_scope_type": "team",
            "share_scope_id": 10,
            "allow_clone": True,
        })
        share_id = create.json()["id"]
        resp = client.post(
            f"/api/v1/sharing/shared-with-me/{share_id}/clone",
            headers={"Idempotency-Key": "sharing-clone-test-0001"},
            json={"name": "My Clone"},
        )
        assert resp.status_code == 202
        data = resp.json()
        assert data["status"] == "queued"
        assert data["command"] == "shared_workspace_clone"
        assert data["share_id"] == share_id
        assert len(manager.list_jobs(domain="sharing", owner_user_id="1")) == 1

    def test_clone_not_allowed(self, clone_client, mock_repo):
        client, manager = clone_client
        create = client.post("/api/v1/sharing/workspaces/ws-noclone/share", json={
            "share_scope_type": "team",
            "share_scope_id": 10,
            "allow_clone": False,
        })
        share_id = create.json()["id"]
        resp = client.post(
            f"/api/v1/sharing/shared-with-me/{share_id}/clone",
            headers={"Idempotency-Key": "sharing-clone-test-0002"},
            json={},
        )
        assert resp.status_code == 403
        assert manager.list_jobs(domain="sharing") == []

    def test_clone_nonexistent_share(self, clone_client, mock_repo):
        client, manager = clone_client
        resp = client.post(
            "/api/v1/sharing/shared-with-me/9999/clone",
            headers={"Idempotency-Key": "sharing-clone-test-0003"},
            json={},
        )
        assert resp.status_code == 404
        assert manager.list_jobs(domain="sharing") == []


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

    def test_create_workspace_token_requires_owned_workspace(
        self,
        client,
        mock_repo,
    ):
        calls: list[tuple[str, int]] = []

        async def _record_workspace_ownership(workspace_id: str, user: User):
            calls.append((workspace_id, user.id))

        with patch.object(
            sharing_endpoints,
            "_verify_workspace_ownership",
            _record_workspace_ownership,
        ):
            resp = client.post("/api/v1/sharing/tokens", json={
                "resource_type": "workspace",
                "resource_id": "ws-owned",
            })

        assert resp.status_code == 200
        assert calls == [("ws-owned", 1)]

    def test_create_chatbook_token_requires_owned_chatbook(
        self,
        client,
        mock_repo,
        monkeypatch,
    ):
        calls: list[tuple[str, int]] = []

        async def _record_chatbook_ownership(chatbook_id: str, user: User):
            calls.append((chatbook_id, user.id))

        monkeypatch.setattr(
            sharing_endpoints,
            "_verify_chatbook_ownership",
            _record_chatbook_ownership,
            raising=False,
        )

        resp = client.post("/api/v1/sharing/tokens", json={
            "resource_type": "chatbook",
            "resource_id": "chatbook-1",
        })

        assert resp.status_code == 200
        assert calls == [("chatbook-1", 1)]

    def test_create_prototype_workspace_token(self, client, mock_repo):
        resp = client.post("/api/v1/sharing/tokens", json={
            "resource_type": "prototype_workspace",
            "resource_id": "pws-1",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "raw_token" in data
        assert data["resource_type"] == "prototype_workspace"
        assert data["resource_id"] == "pws-1"

    def test_create_prototype_workspace_token_requires_owner(self, client, mock_repo):
        class _ForeignPrototypeRepo:
            async def get_workspace(self, prototype_workspace_id: str):
                return {"id": prototype_workspace_id, "owner_user_id": 2}

        with patch.object(
            sharing_endpoints,
            "_get_prototype_repo",
            return_value=_ForeignPrototypeRepo(),
        ):
            resp = client.post("/api/v1/sharing/tokens", json={
                "resource_type": "prototype_workspace",
                "resource_id": "pws-foreign",
            })

        assert resp.status_code == 404

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


def test_request_is_secure_uses_first_forwarded_proto_value():
    request = Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/",
            "scheme": "http",
            "server": ("testserver", 80),
            "client": ("testclient", 50000),
            "headers": [(b"x-forwarded-proto", b"http,https")],
        }
    )

    assert sharing_endpoints._request_is_secure(request) is False


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

    def test_public_preview_prototype_workspace_token(self, client, mock_repo):
        create = client.post("/api/v1/sharing/tokens", json={
            "resource_type": "prototype_workspace",
            "resource_id": "pws-1",
        })
        raw_token = create.json()["raw_token"]
        resp = client.get(f"/api/v1/sharing/public/{raw_token}")
        assert resp.status_code == 200
        assert resp.json()["resource_type"] == "prototype_workspace"

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

    def test_public_import_accepts_empty_json_for_unprotected_token(self, client, mock_repo):
        create = client.post("/api/v1/sharing/tokens", json={
            "resource_type": "workspace",
            "resource_id": "ws-empty-json",
        })
        raw_token = create.json()["raw_token"]
        resp = client.post(f"/api/v1/sharing/public/{raw_token}/import", json={})
        assert resp.status_code == 200
        assert resp.json()["resource_id"] == "ws-empty-json"

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

    def test_public_import_accepts_password_for_protected_token(self, client, mock_repo):
        create = client.post("/api/v1/sharing/tokens", json={
            "resource_type": "workspace",
            "resource_id": "ws-pw-ok",
            "password": "secret123",
        })
        raw_token = create.json()["raw_token"]
        resp = client.post(
            f"/api/v1/sharing/public/{raw_token}/import",
            json={"password": "secret123"},
        )
        assert resp.status_code == 200
        assert resp.json()["resource_id"] == "ws-pw-ok"

    def test_public_import_uses_atomic_claim_for_use_limit(
        self,
        client,
        mock_repo,
        monkeypatch,
    ):
        create = client.post("/api/v1/sharing/tokens", json={
            "resource_type": "workspace",
            "resource_id": "ws-limit",
            "max_uses": 1,
        })
        raw_token = create.json()["raw_token"]

        async def _claim_lost_race(token_id: int) -> bool:
            return False

        monkeypatch.setattr(mock_repo, "claim_token_use", _claim_lost_race)

        resp = client.post(f"/api/v1/sharing/public/{raw_token}/import")

        assert resp.status_code == 404

    def test_public_import_rejects_prototype_workspace_token(self, client, mock_repo):
        create = client.post("/api/v1/sharing/tokens", json={
            "resource_type": "prototype_workspace",
            "resource_id": "pws-import",
        })
        token_id = create.json()["id"]
        raw_token = create.json()["raw_token"]

        resp = client.post(f"/api/v1/sharing/public/{raw_token}/import")
        assert resp.status_code == 422
        assert "prototype-session" in resp.json()["detail"]

        token_row = asyncio.run(mock_repo.get_token(token_id))
        assert token_row is not None
        assert token_row["use_count"] == 0


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
