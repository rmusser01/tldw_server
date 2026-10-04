"""Every storage quota writer lands in limits.storage_quota_mb (spec 2 §5)."""

import asyncio
import uuid

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_registration_service_dep
from tldw_Server_API.app.core.Usage import quota_resolver
from tldw_Server_API.app.main import app
from tldw_Server_API.tests.UserProfile._storage_quota_helpers import (
    KEY,
    me,
    org_and_team_with_member,
    patch_quota,
    quota,
)

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _quotas_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """Quotas on, resolver cache empty."""
    monkeypatch.setenv("USAGE_QUOTAS_ENABLED", "1")
    quota_resolver.invalidate_all()


def test_profile_patch_writes_override_and_null_deletes(auth_headers: dict) -> None:
    """PATCH limits.storage_quota_mb=300 enforces 300; null returns the user to unlimited."""
    with TestClient(app) as client:
        user_id = me(client, auth_headers)
        try:
            assert patch_quota(client, auth_headers, user_id, 300).status_code == 200
            assert quota(user_id) == 300
            assert patch_quota(client, auth_headers, user_id, None).status_code == 200
            assert quota(user_id) is None
        finally:
            patch_quota(client, auth_headers, user_id, None)


def test_admin_update_quota_absent_vs_null(auth_headers: dict) -> None:
    """An admin user update without storage_quota_mb keeps the quota; explicit null clears it; 0 is accepted."""
    with TestClient(app) as client:
        user_id = me(client, auth_headers)
        url = f"/api/v1/admin/users/{user_id}"
        try:
            assert client.put(url, headers=auth_headers, json={"storage_quota_mb": 0, "reason": "quota test set"}).status_code == 200
            assert quota(user_id) == 0
            assert client.put(url, headers=auth_headers, json={"is_verified": True, "reason": "unrelated change"}).status_code == 200
            assert quota(user_id) == 0
            assert client.put(url, headers=auth_headers, json={"storage_quota_mb": None, "reason": "quota test clear"}).status_code == 200
            assert quota(user_id) is None
            # A users-table write and the override write in one PUT must share the request transaction.
            both = {"is_verified": True, "storage_quota_mb": 64, "reason": "both fields at once"}
            assert client.put(url, headers=auth_headers, json=both).status_code == 200
            assert quota(user_id) == 64
        finally:
            patch_quota(client, auth_headers, user_id, None)


def test_admin_create_with_quota_writes_override(auth_headers: dict) -> None:
    """Creating a user with storage_quota_mb=250 yields an enforced 250; without it, unlimited."""
    with TestClient(app) as client:
        admin_id = me(client, auth_headers)
        suffix = uuid.uuid4().hex[:8]

        async def _create(username: str, email: str, quota_override: int | None) -> int:
            """Register a user the same way admin_users_service.create_user does."""
            svc = await get_registration_service_dep()
            info = await svc.register_user(
                username=username,
                email=email,
                password="Extra@Pass#2024!",
                created_by=admin_id,
                role_override="user",
                is_active_override=True,
                is_verified_override=True,
                storage_quota_override=quota_override,
            )
            return int(info["user_id"])

        new_id = asyncio.run(_create(f"quota250{suffix}", f"quota250-{suffix}@example.com", 250))
        other_id = asyncio.run(_create(f"quotanone{suffix}", f"quotanone-{suffix}@example.com", None))
        assert quota(new_id) == 250
        assert quota(other_id) is None


def test_team_storage_value_enforced_and_user_value_wins(auth_headers: dict) -> None:
    """A team limits.storage_quota_mb of 200 applies to a member; their own 500 overrides it."""
    with TestClient(app) as client:
        user_id = me(client, auth_headers)
        _org_id, team_id = org_and_team_with_member(user_id)
        team_url = f"/api/v1/admin/teams/{team_id}/profile/overrides/{KEY}"
        try:
            assert client.put(team_url, headers=auth_headers, json={"value": 200}).status_code == 200
            assert quota(user_id) == 200
            assert patch_quota(client, auth_headers, user_id, 500).status_code == 200
            assert quota(user_id) == 500
        finally:
            patch_quota(client, auth_headers, user_id, None)
            client.delete(team_url, headers=auth_headers)
