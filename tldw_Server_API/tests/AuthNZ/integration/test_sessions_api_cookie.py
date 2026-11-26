"""Sessions API for single-user cookie sessions (TASK-15649)."""

from __future__ import annotations

import pytest

from tldw_Server_API.tests.AuthNZ.integration.test_single_user_cookie_session import (  # noqa: F401
    _mint,
    single_user_cookie_client,
)

pytestmark = pytest.mark.integration

LIST = "/api/v1/auth/sessions"


def _use(client, cookie: str) -> None:
    csrf = client.cookies.get("csrf_token")
    client.cookies.clear()
    if csrf:
        client.cookies.set("csrf_token", csrf, path="/")
    client.cookies.set("tldw_single_user_session", cookie, path="/api")


def _mint_cookie(client, api_key, device=None) -> str:
    client.cookies.clear()
    assert _mint(client, api_key, device).status_code == 200
    return client.cookies["tldw_single_user_session"]


def _csrf(client) -> dict:
    return {"X-CSRF-Token": client.cookies["csrf_token"]}


def _two_cookies(client, api_key, device1=None, device2=None):
    c1 = _mint_cookie(client, api_key, device1)
    csrf1 = client.cookies["csrf_token"]
    c2 = _mint_cookie(client, api_key, device2)
    _use(client, c1)
    client.cookies.set("csrf_token", csrf1, path="/")
    return c1, c2


def test_cookie_session_lists_itself_as_current_and_api_key_does_not(single_user_cookie_client):
    client, api_key = single_user_cookie_client
    c1, c2 = _two_cookies(client, api_key, "dev-cookie-0001", "dev-cookie-0002")
    rows = client.get(LIST).json()
    assert len(rows) == 2 and sum(r["is_current"] for r in rows) == 1
    _use(client, c2)
    other = [r["id"] for r in client.get(LIST).json() if r["is_current"]]
    _use(client, c1)
    mine = [r["id"] for r in client.get(LIST).json() if r["is_current"]]
    assert len(mine) == len(other) == 1 and mine != other
    client.cookies.clear()
    keyed = client.get(LIST, headers={"X-API-KEY": api_key}).json()
    assert len(keyed) == 2 and not any(r["is_current"] for r in keyed)


def test_cookie_session_ends_another_cookie_session(single_user_cookie_client):
    client, api_key = single_user_cookie_client
    c1, c2 = _two_cookies(client, api_key, "dev-cookie-0001", "dev-cookie-0002")
    _use(client, c2)
    other = next(r["id"] for r in client.get(LIST).json() if r["is_current"])
    _use(client, c1)
    r = client.delete(f"{LIST}/{other}", headers=_csrf(client))
    assert r.json()["message"] == "Session revoked successfully"
    _use(client, c2)
    assert client.get(LIST).status_code == 401


def test_api_key_cannot_end_single_user_sessions(single_user_cookie_client):
    client, api_key = single_user_cookie_client
    c1, c2 = _two_cookies(client, api_key, "dev-cookie-0001", "dev-cookie-0002")
    key = {"X-API-KEY": api_key}
    client.cookies.clear()
    ids = [r["id"] for r in client.get(LIST, headers=key).json()]
    one = client.delete(f"{LIST}/{ids[0]}", headers=key)
    all_ = client.post(f"{LIST}/revoke-all", headers=key)
    assert one.status_code == all_.status_code == 403
    assert one.json()["detail"]["code"] == all_.json()["detail"]["code"] == "interactive_session_required"
    for cookie in (c1, c2):
        _use(client, cookie)
        assert len(client.get(LIST).json()) == 2


def test_cookie_revoke_all_keep_current(single_user_cookie_client):
    client, api_key = single_user_cookie_client
    c1, c2 = _two_cookies(client, api_key, "dev-cookie-0001", "dev-cookie-0002")
    r = client.post(f"{LIST}/revoke-all?keep_current=true", headers=_csrf(client))
    assert r.json()["details"]["sessions_revoked"] == 1
    assert len(client.get(LIST).json()) == 1
    _use(client, c2)
    assert client.get(LIST).status_code == 401


def test_cookie_sessions_without_an_install_header_are_never_same_install(single_user_cookie_client):
    client, api_key = single_user_cookie_client
    _two_cookies(client, api_key)
    rows = client.get(LIST).json()
    assert len(rows) == 2 and not any(r["same_install"] for r in rows)
    assert sum(r["is_current"] for r in rows) == 1
    _use(client, _mint_cookie(client, api_key, "dev-cookie-0001"))
    _two = _mint_cookie(client, api_key, "dev-cookie-0001")  # a second sign-in from the same install
    rows = client.get(LIST).json()
    assert sum(r["is_current"] for r in rows) == 1
    assert sorted(r["same_install"] for r in rows) == [False, False, True, True]


def _single_user_alerts(user_id: int) -> list:
    from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase

    with CollectionsDatabase.for_user(user_id=user_id) as db:
        rows = db.list_user_notifications()
    return [r for r in rows if r.kind == "new_sign_in"]


@pytest.mark.sign_in_alerts
async def test_single_user_cookie_mint_alerts_the_other_install(single_user_cookie_client, tmp_path, monkeypatch):
    from tldw_Server_API.app.core.AuthNZ.settings import get_settings
    from tldw_Server_API.app.core.AuthNZ.sign_in_alert import drain_pending_alerts

    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "user_dbs"))
    client, api_key = single_user_cookie_client
    uid = int(get_settings().SINGLE_USER_FIXED_ID)
    _mint_cookie(client, api_key, "dev-cookie-0001")
    await drain_pending_alerts()
    assert _single_user_alerts(uid) == []
    _mint_cookie(client, api_key, "dev-cookie-0002")
    await drain_pending_alerts()
    (alert,) = _single_user_alerts(uid)
    assert alert.severity == "warning" and alert.link_url == "/settings/sessions"


@pytest.mark.sign_in_alerts
async def test_cookie_mints_without_an_install_header_still_alert(single_user_cookie_client, tmp_path, monkeypatch):
    from tldw_Server_API.app.core.AuthNZ.settings import get_settings
    from tldw_Server_API.app.core.AuthNZ.sign_in_alert import drain_pending_alerts

    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "user_dbs"))
    client, api_key = single_user_cookie_client
    uid = int(get_settings().SINGLE_USER_FIXED_ID)
    _mint_cookie(client, api_key)
    _mint_cookie(client, api_key)
    await drain_pending_alerts()
    assert len(_single_user_alerts(uid)) == 1
