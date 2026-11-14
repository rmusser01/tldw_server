"""Every reader shows the enforced storage quota; unlimited is null (spec 2 §5)."""

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.Usage import quota_resolver
from tldw_Server_API.app.main import app
from tldw_Server_API.app.services.storage_quota_service import StorageQuotaService
from tldw_Server_API.tests.UserProfile._storage_quota_helpers import me, patch_quota

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _quotas_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """Quotas on, resolver cache empty."""
    monkeypatch.setenv("USAGE_QUOTAS_ENABLED", "1")
    quota_resolver.invalidate_all()


def _assert_unlimited(body: dict) -> None:
    """The storage response says unlimited (null), not 0 or 5120."""
    assert body["storage_quota_mb"] is None
    assert body["available_mb"] is None
    assert body["usage_percentage"] is None


def test_users_storage_unlimited_is_null(auth_headers: dict) -> None:
    """GET /users/storage reports null quota fields when no limits.storage_quota_mb applies."""
    with TestClient(app) as client:
        _assert_unlimited(client.get("/api/v1/users/storage", headers=auth_headers).json())


def test_users_storage_fallback_unlimited_is_null(auth_headers: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """When the live calculation fails, the fallback path also reports null, not 5120."""

    async def _fail(self, *args, **kwargs):
        """A calculation failure the endpoint catches."""
        raise RuntimeError("calculation failed")

    monkeypatch.setattr(StorageQuotaService, "calculate_user_storage", _fail)
    with TestClient(app) as client:
        _assert_unlimited(client.get("/api/v1/users/storage", headers=auth_headers).json())


def test_me_admin_list_and_profile_show_resolved_value(auth_headers: dict) -> None:
    """/users/me, GET /admin/users and the profile quotas section show 250 when set, null when cleared."""
    with TestClient(app) as client:
        user_id = me(client, auth_headers)
        try:
            assert patch_quota(client, auth_headers, user_id, 250).status_code == 200
            assert client.get("/api/v1/users/me", headers=auth_headers).json()["storage_quota_mb"] == 250
            listing = client.get("/api/v1/admin/users", headers=auth_headers, params={"limit": 100}).json()
            row = next(u for u in listing["users"] if int(u["id"]) == user_id)
            assert row["storage_quota_mb"] == 250
            profile = client.get("/api/v1/users/me/profile", headers=auth_headers, params={"sections": "quotas"}).json()
            assert profile["quotas"]["storage_quota_mb"] == 250
            assert patch_quota(client, auth_headers, user_id, None).status_code == 200
            assert client.get("/api/v1/users/me", headers=auth_headers).json()["storage_quota_mb"] is None
            profile = client.get("/api/v1/users/me/profile", headers=auth_headers, params={"sections": "quotas"}).json()
            # /me/profile is response_model_exclude_none=True: a null quota is an absent key, not {"storage_quota_mb": null}.
            assert profile["quotas"].get("storage_quota_mb") is None
        finally:
            patch_quota(client, auth_headers, user_id, None)


def test_admin_stats_total_quota_is_null(auth_headers: dict) -> None:
    """Admin system stats report total_quota_mb null instead of summing the legacy column."""
    with TestClient(app) as client:
        storage = client.get("/api/v1/admin/stats", headers=auth_headers).json()["storage"]
    assert storage["total_quota_mb"] is None
