"""Sessions API against isolated PostgreSQL (TASK-15649)."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.integration

PASSWORD = "Str0ngP@ssw0rd!"
CHROME_MAC = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
SAFARI_IPHONE = "Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Mobile/15E148 Safari/604.1"


def _login(client, name, device, ua=None) -> dict:
    headers = {"X-TLDW-Device-Id": device}
    if ua:
        headers["User-Agent"] = ua
    r = client.post("/api/v1/auth/login", data={"username": name, "password": PASSWORD}, headers=headers)
    assert r.status_code == 200, r.text
    return r.json()


def _bearer(tokens) -> dict:
    return {"Authorization": f"Bearer {tokens['access_token']}"}


@pytest.mark.asyncio
async def test_sessions_api_postgres(isolated_test_environment: tuple[TestClient, str], monkeypatch) -> None:
    client, _db = isolated_test_environment
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("JWT_SECRET_KEY", "X" * 40)
    for name in ("pgalice", "pgbob"):
        reg = client.post("/api/v1/auth/register", json={"username": name, "email": f"{name}@example.com", "password": PASSWORD})
        assert reg.status_code == 200, reg.text

    a = _login(client, "pgalice", "dev-aaaaaaaa", CHROME_MAC)
    b = _login(client, "pgalice", "dev-bbbbbbbb", SAFARI_IPHONE)
    c = _login(client, "pgalice", "dev-aaaaaaaa", CHROME_MAC)
    rows = {r["id"]: r for r in client.get("/api/v1/auth/sessions", headers=_bearer(a)).json()}
    assert len(rows) == 3
    current = [i for i, r in rows.items() if r["is_current"]]
    assert len(current) == 1
    assert sorted(r["same_install"] for r in rows.values()) == [False, True, True]
    assert sorted(r["device_label"] for r in rows.values()) == ["Chrome on macOS", "Chrome on macOS", "Safari on iOS"]
    assert "dev-aaaaaaaa" not in json.dumps(rows)

    # an API key lists sessions but may not end them
    key = client.post("/api/v1/users/api-keys", json={"name": "t", "scope": "write"}, headers=_bearer(a)).json()["key"]
    keyed = client.get("/api/v1/auth/sessions", headers={"X-API-KEY": key}).json()
    assert len(keyed) == 3 and not any(r["is_current"] or r["same_install"] for r in keyed)
    denied = client.post("/api/v1/auth/sessions/revoke-all", headers={"X-API-KEY": key})
    assert denied.status_code == 403 and denied.json()["detail"]["code"] == "interactive_session_required"
    assert len(client.get("/api/v1/auth/sessions", headers=_bearer(a)).json()) == 3

    # another user can neither see nor end alice's sessions
    bob = _login(client, "pgbob", "dev-cccccccc")
    assert not set(rows) & {r["id"] for r in client.get("/api/v1/auth/sessions", headers=_bearer(bob)).json()}
    client.post("/api/v1/auth/sessions/revoke-all?keep_current=true", headers=_bearer(bob))
    assert len(client.get("/api/v1/auth/sessions", headers=_bearer(a)).json()) == 3

    # keep_current ends the other two only
    kept = client.post("/api/v1/users/sessions/revoke-all?keep_current=true", headers=_bearer(a))
    assert kept.json()["details"]["sessions_revoked"] == 2
    assert client.get("/api/v1/auth/sessions", headers=_bearer(a)).status_code == 200
    assert client.get("/api/v1/auth/sessions", headers=_bearer(b)).status_code == 401
    assert client.get("/api/v1/auth/sessions", headers=_bearer(c)).status_code == 401

    # the default cap evicts the oldest of five sign-ins with reason session_cap; revoke-all then
    # counts only the four live ones and leaves that reason alone
    toks = [_login(client, "pgalice", f"dev-{i:08d}") for i in range(5)]
    ended = client.post("/api/v1/auth/sessions/revoke-all", headers=_bearer(toks[-1]))
    assert ended.json()["details"]["sessions_revoked"] == 4

    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool

    pool = await get_db_pool()
    reasons = [r["revoke_reason"] for r in await pool.fetch("SELECT revoke_reason FROM sessions")]
    assert "session_cap" in reasons
