"""The new-sign-in alert over a real multi-user SQLite app (TASK-15649)."""

from __future__ import annotations

import ast
import json
from datetime import datetime, timezone

import pytest
from loguru import logger

from tldw_Server_API.app.core.AuthNZ import session_manager as session_manager_module
from tldw_Server_API.app.core.AuthNZ import sign_in_alert
from tldw_Server_API.app.core.AuthNZ.repos.sessions_repo import AuthnzSessionsRepo
from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
from tldw_Server_API.app.core.AuthNZ.sign_in_alert import drain_pending_alerts, new_sign_in_message
from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseError as BackendDatabaseError
from tldw_Server_API.app.core.exceptions import DatabaseError as AuthnzDatabaseError
from tldw_Server_API.app.core.exceptions import StorageUnavailableError
from tldw_Server_API.tests.AuthNZ.unit.test_device_label import CHROME_MAC, SAFARI_IPHONE
from tldw_Server_API.tests.AuthNZ_SQLite._sessions_http_fixtures import bearer, login, session_id_of
from tldw_Server_API.tests.lint._ast_cache import APP_ROOT

pytestmark = [pytest.mark.asyncio, pytest.mark.sign_in_alerts]


async def notifications(client, tokens) -> list[dict]:
    await drain_pending_alerts()
    response = client.get("/api/v1/notifications", headers=bearer(tokens))
    assert response.status_code == 200, response.text
    return response.json()["items"]


async def test_first_sign_in_writes_nothing(two_user_client):
    client, pw = two_user_client.client, two_user_client.password
    A = login(client, "alice", pw, device="dev-aaaaaaaa")
    assert await notifications(client, A) == []


async def test_second_sign_in_writes_one_warning_notification_with_the_spec_fields(two_user_client):
    client, pw = two_user_client.client, two_user_client.password
    A = login(client, "alice", pw, device="dev-aaaaaaaa", user_agent=SAFARI_IPHONE)
    B = login(client, "alice", pw, device="dev-bbbbbbbb", user_agent=CHROME_MAC)
    (n,) = await notifications(client, A)
    ip = next(r["ip_address"] for r in client.get("/api/v1/auth/sessions", headers=bearer(B)).json() if r["is_current"])
    assert (n["kind"], n["severity"], n["title"]) == ("new_sign_in", "warning", "New sign-in to your account")
    assert (n["source_domain"], n["source_job_id"], n["link_url"]) == ("auth_session", str(session_id_of(B)), "/settings/sessions")
    where = ip if ip not in (None, "", "unknown") else "an unknown address"
    assert n["message"].startswith(f"Chrome on macOS (as the browser reported it) signed in from {where} at ") and " UTC." in n["message"]
    assert n["read_at"] is None and n["dismissed_at"] is None


async def test_same_browser_inside_the_bucket_collapses_to_one_notification(two_user_client):
    client, pw = two_user_client.client, two_user_client.password
    A = login(client, "alice", pw, device="dev-aaaaaaaa")
    for _ in range(3):
        B = login(client, "alice", pw, device="dev-bbbbbbbb")
    assert len(await notifications(client, A)) == 1
    login(client, "alice", pw, device="dev-cccccccc")  # at the cap of 4 this evicts A, so read as B
    assert len(await notifications(client, B)) == 2


async def test_alert_carries_no_credentials_or_install_id(two_user_client):
    client, pw = two_user_client.client, two_user_client.password
    A = login(client, "alice", pw, device="dev-aaaaaaaa")
    B = login(client, "alice", pw, device="dev-bbbbbbbb")
    raw = json.dumps(await notifications(client, A))
    for secret in (B["access_token"], B["refresh_token"], "dev-bbbbbbbb", pw):
        assert secret not in raw


async def test_alert_is_written_with_every_notification_preference_off(two_user_client):
    client, pw = two_user_client.client, two_user_client.password
    A = login(client, "alice", pw, device="dev-aaaaaaaa")
    patched = client.patch(
        "/api/v1/notifications/preferences",
        json={"reminder_enabled": False, "job_completed_enabled": False, "job_failed_enabled": False},
        headers=bearer(A),
    )
    assert patched.status_code == 200, patched.text
    assert [patched.json()[k] for k in ("reminder_enabled", "job_completed_enabled", "job_failed_enabled")] == [False] * 3
    login(client, "alice", pw, device="dev-bbbbbbbb")
    assert len(await notifications(client, A)) == 1


@pytest.mark.parametrize("error", [BackendDatabaseError("db down"), StorageUnavailableError("no base dir")])
async def test_failed_alert_write_does_not_fail_sign_in_and_logs_no_ids(two_user_client, monkeypatch, error):
    client, pw = two_user_client.client, two_user_client.password
    login(client, "alice", pw, device="dev-aaaaaaaa")

    def boom(**_):
        raise error

    monkeypatch.setattr(sign_in_alert.CollectionsDatabase, "for_user", staticmethod(boom))
    lines = []  # only this module's records, so an unrelated DB-layer debug line cannot match
    sink = logger.add(lines.append, level="DEBUG", filter=lambda r: r["name"] == sign_in_alert.__name__)
    try:
        B = login(client, "alice", pw, device="dev-bbbbbbbb")
        await drain_pending_alerts()
    finally:
        logger.remove(sink)
    text = "".join(map(str, lines))
    assert client.get("/api/v1/auth/sessions", headers=bearer(B)).status_code == 200
    assert "New sign-in alert not written" in text
    for secret in (B["access_token"], B["refresh_token"], "dev-bbbbbbbb"):
        assert secret not in text


async def test_sessions_read_failure_does_not_fail_sign_in(two_user_client, monkeypatch):
    client, pw = two_user_client.client, two_user_client.password
    A = login(client, "alice", pw, device="dev-aaaaaaaa")

    async def down(self, user_id):
        raise AuthnzDatabaseError("sessions unavailable")  # the class AuthnzSessionsRepo raises, not the backend one

    monkeypatch.setattr(AuthnzSessionsRepo, "get_active_sessions_for_user", down)
    counters = []
    monkeypatch.setattr(sign_in_alert, "log_counter", lambda name, *a, **k: counters.append(name))
    login(client, "alice", pw, device="dev-bbbbbbbb")  # login() asserts 200
    await drain_pending_alerts()
    assert "auth_sign_in_alert_failed" in counters
    assert await notifications(client, A) == []


async def test_scheduling_failure_does_not_fail_sign_in(two_user_client, monkeypatch):
    client, pw = two_user_client.client, two_user_client.password
    login(client, "alice", pw, device="dev-aaaaaaaa")

    def boom(**_):
        raise RuntimeError("scheduler down")

    monkeypatch.setattr(session_manager_module, "schedule_new_sign_in_alert", boom)
    counters = []
    monkeypatch.setattr(session_manager_module, "log_counter", lambda name, *a, **k: counters.append(name))
    B = login(client, "alice", pw, device="dev-bbbbbbbb")  # no SessionError after the row was inserted
    assert session_id_of(B) in {r["id"] for r in client.get("/api/v1/auth/sessions", headers=bearer(B)).json()}
    assert "auth_sign_in_alert_failed" in counters


async def test_alert_is_written_when_the_cap_evicted_the_only_other_session(two_user_client, monkeypatch):
    client, pw = two_user_client.client, two_user_client.password
    monkeypatch.setenv("MAX_SESSIONS_PER_USER", "1")
    reset_settings()
    A = login(client, "alice", pw, device="dev-aaaaaaaa")
    B = login(client, "alice", pw, device="dev-bbbbbbbb")  # evicts A (another install)
    evicted = client.get("/api/v1/auth/sessions", headers=bearer(A))
    assert evicted.status_code == 401 and evicted.json()["detail"]["code"] == "session_replaced"
    (n,) = await notifications(client, B)
    assert n["source_job_id"] == str(session_id_of(B))


async def test_evicting_the_same_install_writes_nothing(two_user_client, monkeypatch):
    client, pw = two_user_client.client, two_user_client.password
    monkeypatch.setenv("MAX_SESSIONS_PER_USER", "1")
    reset_settings()
    A = login(client, "alice", pw, device="dev-aaaaaaaa")
    B = login(client, "alice", pw, device="dev-aaaaaaaa")  # evicts A, same install signing in again
    evicted = client.get("/api/v1/auth/sessions", headers=bearer(A))
    assert evicted.status_code == 401 and evicted.json()["detail"]["code"] == "session_replaced"
    assert await notifications(client, B) == []


async def test_signing_in_again_from_an_install_with_a_live_session_writes_nothing_new(two_user_client):
    client, pw = two_user_client.client, two_user_client.password
    A = login(client, "alice", pw, device="dev-aaaaaaaa")
    B = login(client, "alice", pw, device="dev-bbbbbbbb")  # one alert, for B
    login(client, "alice", pw, device="dev-aaaaaaaa")  # A's install again
    items = await notifications(client, A)
    assert [n["source_job_id"] for n in items] == [str(session_id_of(B))]


async def test_sign_ins_without_an_install_id_still_alert(two_user_client):
    client, pw = two_user_client.client, two_user_client.password
    A = login(client, "alice", pw)  # no X-TLDW-Device-Id header: device_id is None
    B = login(client, "alice", pw)
    assert [n["source_job_id"] for n in await notifications(client, A)] == [str(session_id_of(B))]


def test_message_cuts_a_forwarded_ip_to_64_characters():
    text = new_sign_in_message(
        user_agent=CHROME_MAC, ip_address="1" * 100, now=datetime(2026, 11, 25, 14, 3, tzinfo=timezone.utc)
    )
    assert ("1" * 64) in text and ("1" * 65) not in text and text.endswith("sign out your other sessions.")


async def test_alert_goes_only_to_the_signing_in_users_inbox(two_user_client):
    client, pw = two_user_client.client, two_user_client.password
    login(client, "alice", pw, device="dev-aaaaaaaa")
    login(client, "alice", pw, device="dev-bbbbbbbb")
    bob = login(client, "bob", pw)
    assert await notifications(client, bob) == []


def test_every_real_sign_in_call_site_requests_the_alert():
    """AST pin: all four production create_session calls pass alert_new_sign_in=True."""
    for rel, expected in (("api/v1/endpoints/auth.py", 3), ("core/AuthNZ/single_user_session.py", 1)):
        calls = [
            n
            for n in ast.walk(ast.parse((APP_ROOT / rel).read_text()))
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "create_session"
        ]
        assert len(calls) == expected
        assert all(
            any(k.arg == "alert_new_sign_in" and getattr(k.value, "value", None) is True for k in c.keywords)
            for c in calls
        )
