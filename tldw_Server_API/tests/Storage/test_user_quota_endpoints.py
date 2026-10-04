"""Both per-user storage quota admin endpoints write and read limits.storage_quota_mb (spec 2 §5)."""

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.AuthNZ.repos.storage_quotas_repo import AuthnzStorageQuotasRepo
from tldw_Server_API.app.core.Usage import quota_resolver
from tldw_Server_API.app.main import app
from tldw_Server_API.tests.UserProfile._storage_quota_helpers import me, patch_quota, quota

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _quotas_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """Quotas on, resolver cache empty."""
    monkeypatch.setenv("USAGE_QUOTAS_ENABLED", "1")
    quota_resolver.invalidate_all()


def test_storage_admin_put_accepts_zero_and_null(auth_headers: dict) -> None:
    """PUT /storage/admin/quotas/user/{id} accepts 0 (blocks) and null (unlimited); 404 missing user; 422 negative."""
    with TestClient(app) as client:
        user_id = me(client, auth_headers)
        url = f"/api/v1/storage/admin/quotas/user/{user_id}"
        try:
            resp = client.put(url, headers=auth_headers, json={"quota_mb": 0})
            assert resp.status_code == 200 and resp.json()["quota"]["quota_mb"] == 0
            assert quota(user_id) == 0
            resp = client.put(url, headers=auth_headers, json={"quota_mb": None})
            assert resp.status_code == 200 and resp.json()["quota"]["quota_mb"] is None
            assert quota(user_id) is None
            assert client.put(url, headers=auth_headers, json={"quota_mb": -1}).status_code == 422
            missing = client.put("/api/v1/storage/admin/quotas/user/987654321", headers=auth_headers, json={"quota_mb": 5})
            assert missing.status_code == 404
        finally:
            patch_quota(client, auth_headers, user_id, None)


def test_admin_storage_quotas_user_routes_use_the_user_not_an_org(auth_headers: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    """PUT then GET /admin/storage-quotas/users/{id} set and read the user's quota and never touch an org pool."""
    org_writes: list[tuple] = []

    async def _record(self, *args, **kwargs) -> dict:
        """Record any org-pool write."""
        org_writes.append((args, kwargs))
        return {}

    monkeypatch.setattr(AuthnzStorageQuotasRepo, "upsert_org_quota", _record)
    with TestClient(app) as client:
        user_id = me(client, auth_headers)
        url = f"/api/v1/admin/storage-quotas/users/{user_id}"
        try:
            assert client.put(url, headers=auth_headers, json={"quota_mb": 400}).status_code == 200
            assert client.get(url, headers=auth_headers).json()["quota_mb"] == 400
            assert client.put(url, headers=auth_headers, json={"quota_mb": None}).status_code == 200
            assert client.get(url, headers=auth_headers).json()["quota_mb"] is None
            assert org_writes == []
        finally:
            patch_quota(client, auth_headers, user_id, None)
