from __future__ import annotations

from fastapi.testclient import TestClient

from tldw_Server_API.app.main import app


def _get_user_id(client: TestClient, auth_headers) -> int:
    resp = client.get("/api/v1/users/me/profile", headers=auth_headers)
    assert resp.status_code == 200
    return resp.json()["user"]["id"]


def test_admin_bulk_profile_update_dry_run(auth_headers) -> None:
    with TestClient(app) as client:
        user_id = _get_user_id(client, auth_headers)
        resp = client.post(
            "/api/v1/admin/users/profile/bulk",
            headers=auth_headers,
            json={
                "user_ids": [user_id],
                "dry_run": True,
                "updates": [
                    {"key": "limits.storage_quota_mb", "value": 8192},
                ],
            },
        )
        assert resp.status_code == 200
        payload = resp.json()

        assert payload["total_targets"] == 1
        assert payload["dry_run"] is True
        assert payload["updated"] == 1
        assert payload["results"][0]["user_id"] == user_id
        diffs = payload["results"][0].get("diffs", [])
        assert diffs
        assert diffs[0]["key"] == "limits.storage_quota_mb"
        assert diffs[0]["after"] == 8192

        profile_resp = client.get(
            "/api/v1/users/me/profile",
            params={"sections": "quotas"},
            headers=auth_headers,
        )
        assert profile_resp.status_code == 200
        profile = profile_resp.json()

    assert profile.get("quotas", {}).get("storage_quota_mb") != 8192


def test_admin_bulk_profile_update_apply(auth_headers) -> None:
    from tldw_Server_API.tests.UserProfile._storage_quota_helpers import patch_quota

    with TestClient(app) as client:
        user_id = _get_user_id(client, auth_headers)
        try:
            resp = client.post(
                "/api/v1/admin/users/profile/bulk",
                headers=auth_headers,
                json={
                    "user_ids": [user_id],
                    "updates": [
                        {"key": "limits.storage_quota_mb", "value": 2048},
                    ],
                },
            )
            assert resp.status_code == 200
            payload = resp.json()
            assert payload["total_targets"] == 1
            assert payload["updated"] == 1
            assert payload["results"][0]["user_id"] == user_id
            assert "limits.storage_quota_mb" in payload["results"][0]["applied"]
            diffs = payload["results"][0].get("diffs", [])
            assert diffs
            assert diffs[0]["key"] == "limits.storage_quota_mb"
            assert diffs[0]["after"] == 2048

            profile_resp = client.get(
                "/api/v1/users/me/profile",
                params={"sections": "quotas"},
                headers=auth_headers,
            )
            assert profile_resp.status_code == 200
            profile = profile_resp.json()
        finally:
            # This sets the shared single-user id's override; clear it so later tests
            # in this worker don't see a stale 2048 quota (spec 2 Sec. 5).
            patch_quota(client, auth_headers, user_id, None)

    assert profile.get("quotas", {}).get("storage_quota_mb") == 2048
