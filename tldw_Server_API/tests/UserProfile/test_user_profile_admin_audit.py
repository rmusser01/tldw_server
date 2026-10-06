from __future__ import annotations

from typing import Any, Dict, List

from fastapi.testclient import TestClient

from tldw_Server_API.app.main import app


def _get_user_id(client: TestClient, auth_headers) -> int:
    resp = client.get("/api/v1/users/me/profile", headers=auth_headers)
    assert resp.status_code == 200
    return int(resp.json()["user"]["id"])


def test_admin_profile_read_emits_audit(auth_headers, monkeypatch) -> None:
    calls: List[Dict[str, Any]] = []

    async def _stub_emit(*_args, **kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.admin._emit_admin_audit_event",
        _stub_emit,
    )

    with TestClient(app) as client:
        user_id = _get_user_id(client, auth_headers)
        resp = client.get(
            f"/api/v1/admin/users/{user_id}/profile",
            headers=auth_headers,
        )
        assert resp.status_code == 200

    assert calls
    assert calls[0].get("action") == "user_profile.read"


def test_admin_profile_update_emits_audit(auth_headers, monkeypatch) -> None:
    calls: List[Dict[str, Any]] = []

    async def _stub_emit(*_args, **kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.admin._emit_admin_audit_event",
        _stub_emit,
    )

    from tldw_Server_API.tests.UserProfile._storage_quota_helpers import patch_quota

    with TestClient(app) as client:
        user_id = _get_user_id(client, auth_headers)
        try:
            resp = client.patch(
                f"/api/v1/admin/users/{user_id}/profile",
                headers=auth_headers,
                json={"updates": [{"key": "limits.storage_quota_mb", "value": 2048}]},
            )
            assert resp.status_code == 200
        finally:
            # This sets the shared single-user id's override; clear it so later tests
            # in this worker don't see a stale 2048 quota (spec 2 Sec. 5).
            patch_quota(client, auth_headers, user_id, None)

    assert calls
    assert calls[0].get("action") == "user_profile.update"
