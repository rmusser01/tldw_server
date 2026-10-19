"""Captured workspace owners are checked against real authenticated JWT users."""

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.API_Deps import kanban_deps
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.api.v1.endpoints.admin.admin_system import get_db_transaction
from tldw_Server_API.app.core.AuthNZ.jwt_service import get_jwt_service, reset_jwt_service
from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
from tldw_Server_API.app.core.DB_Management.Users_DB import create_user
from tldw_Server_API.tests.AuthNZ.integration.test_single_user_cookie_session import (
    single_user_cookie_client as single_user_cookie_client,
)

pytestmark = pytest.mark.integration


def test_private_workspaces_reject_a_real_changed_account(single_user_cookie_client, monkeypatch):
    """Real JWT auth remains intact; sentinels record protected database admission."""
    client, _api_key = single_user_cookie_client
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    reset_settings()
    reset_jwt_service()

    async def accounts():
        return [
            await create_user(name, f"{name}@workspace.test", "unused-test-hash", is_verified=True)
            for name in ("alice", "bob")
        ]

    alice, bob = client.portal.call(accounts)
    token = get_jwt_service().create_access_token(bob["id"], "bob", "user")
    authorization = {"Authorization": f"Bearer {token}"}
    accesses = []

    async def database():
        accesses.append("database")
        raise HTTPException(409, "Database admission reached")

    async def kanban_database(_user_id):
        return await database()

    monkeypatch.setattr(kanban_deps, "_get_or_init_db_instance", kanban_database)
    app = client.app
    previous = dict(app.dependency_overrides)
    app.dependency_overrides[get_media_db_for_user] = database
    app.dependency_overrides[get_db_transaction] = database
    try:
        for path in ("/api/v1/claims", "/api/v1/kanban/boards", "/api/v1/admin/stats"):
            accesses.clear()
            denied = client.get(path, headers={**authorization, "X-TLDW-Expected-User-ID": str(alice["id"])})
            assert denied.status_code == 412, denied.text  # nosec B101 # real authenticated ownership fence
            assert denied.json()["detail"]["code"] == "request_config_scope_changed"  # nosec B101
            assert accesses == []  # nosec B101
            for expected_owner in (None, str(bob["id"])):
                headers = dict(authorization)
                if expected_owner is not None:
                    headers["X-TLDW-Expected-User-ID"] = expected_owner
                admitted = client.get(path, headers=headers)
                # Ordinary users still cannot cross the admin RBAC boundary.
                expected_status = 403 if path.startswith("/api/v1/admin/") else 409
                assert admitted.status_code == expected_status, admitted.text  # nosec B101
    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(previous)
