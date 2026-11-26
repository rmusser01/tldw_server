"""Only an older browser session may silence a new-sign-in alert, on isolated PostgreSQL (TASK-15649)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.AuthNZ.jwt_service import get_jwt_service
from tldw_Server_API.app.core.AuthNZ.sign_in_alert import drain_pending_alerts

pytestmark = [pytest.mark.integration, pytest.mark.asyncio, pytest.mark.sign_in_alerts]

PASSWORD = "Str0ngP@ssw0rd!"


def _bearer(tokens: dict) -> dict:
    return {"Authorization": f"Bearer {tokens['access_token']}"}


async def test_only_an_older_session_dismisses_the_alert_postgres(
    isolated_test_environment: tuple[TestClient, str], monkeypatch, tmp_path
) -> None:
    client, _db = isolated_test_environment
    monkeypatch.setenv("USER_DB_BASE_DIR", str((tmp_path / "user_dbs").resolve()))
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("JWT_SECRET_KEY", "X" * 40)
    reg = client.post(
        "/api/v1/auth/register",
        json={"username": "pgalice", "email": "pgalice@example.com", "password": PASSWORD},
    )
    assert reg.status_code == 200, reg.text

    def login(device: str) -> dict:
        r = client.post(
            "/api/v1/auth/login",
            data={"username": "pgalice", "password": PASSWORD},
            headers={"X-TLDW-Device-Id": device},
        )
        assert r.status_code == 200, r.text
        return r.json()

    a = login("dev-aaaaaaaa")
    b = login("dev-bbbbbbbb")
    await drain_pending_alerts()
    items = client.get("/api/v1/notifications?kind=new_sign_in", headers=_bearer(a)).json()["items"]
    (alert,) = items
    assert alert["source_job_id"] == str(get_jwt_service().decode_access_token(b["access_token"])["session_id"])
    url = f"/api/v1/notifications/{alert['id']}/dismiss"

    own = client.post(url, headers=_bearer(b))
    assert own.status_code == 403 and own.json()["detail"]["code"] == "new_sign_in_protected"
    still = client.get("/api/v1/notifications?kind=new_sign_in", headers=_bearer(a)).json()["items"]
    assert [i["dismissed_at"] for i in still] == [None]

    key = client.post("/api/v1/users/api-keys", json={"name": "t", "scope": "write"}, headers=_bearer(a))
    assert key.status_code == 201, key.text
    keyed = client.post(url, headers={"X-API-KEY": key.json()["key"]})
    assert keyed.status_code == 403 and keyed.json()["detail"]["code"] == "new_sign_in_protected"

    assert client.post(url, headers=_bearer(a)).json() == {"dismissed": True}
