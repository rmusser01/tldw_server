"""Sessions API over a real multi-user SQLite app: is_current, labels, keep_current, interactive guard (TASK-15649)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from tldw_Server_API.tests.AuthNZ.unit.test_device_label import CHROME_MAC, SAFARI_IPHONE
from tldw_Server_API.tests.AuthNZ_SQLite._sessions_http_fixtures import (
    api_key_for,
    bearer,
    login,
    revoke_reason_of,
    seed_session_revoked_with_reason,
    session_id_of,
)

pytestmark = pytest.mark.asyncio


@pytest.fixture
def alice_sessions(two_user_client):
    """Alice signed in twice from two installs: A (Chrome on macOS) and B (Safari on iOS)."""
    ctx = two_user_client
    a = login(ctx.client, "alice", ctx.password, device="dev-aaaaaaaa", user_agent=CHROME_MAC)
    b = login(ctx.client, "alice", ctx.password, device="dev-bbbbbbbb", user_agent=SAFARI_IPHONE)
    return SimpleNamespace(ctx=ctx, client=ctx.client, A=a, B=b, pw=ctx.password)


async def test_list_marks_exactly_the_callers_session_current(alice_sessions):
    client, A, B = alice_sessions.client, alice_sessions.A, alice_sessions.B
    rows = client.get("/api/v1/auth/sessions", headers=bearer(A)).json()
    assert [r["id"] for r in rows if r["is_current"]] == [session_id_of(A)]
    assert {r["id"]: r["device_label"] for r in rows} == {session_id_of(A): "Chrome on macOS", session_id_of(B): "Safari on iOS"}


async def test_client_cannot_claim_current_session(alice_sessions):
    client, A, B = alice_sessions.client, alice_sessions.A, alice_sessions.B
    other = session_id_of(B)
    r = client.get(f"/api/v1/auth/sessions?is_current=true&current={other}&session_id={other}",
                   headers={**bearer(A), "X-TLDW-Device-Id": "dev-bbbbbbbb"})
    assert [x["id"] for x in r.json() if x["is_current"]] == [session_id_of(A)]


async def test_same_install_marks_only_sessions_of_the_callers_install(alice_sessions):
    client, A, B, pw = alice_sessions.client, alice_sessions.A, alice_sessions.B, alice_sessions.pw
    C = login(client, "alice", pw, device="dev-aaaaaaaa", user_agent=CHROME_MAC)      # a second sign-in from A's install
    rows = {r["id"]: r for r in client.get("/api/v1/auth/sessions", headers=bearer(A)).json()}
    assert {i: r["same_install"] for i, r in rows.items()} == {session_id_of(A): True, session_id_of(C): True, session_id_of(B): False}
    raw = json.dumps(rows)
    assert "dev-aaaaaaaa" not in raw and "dev-bbbbbbbb" not in raw     # the install id is never returned


async def test_api_key_lists_sessions_but_none_is_current(alice_sessions):
    client, A = alice_sessions.client, alice_sessions.A
    rows = client.get("/api/v1/auth/sessions", headers={"X-API-KEY": api_key_for(client, A)}).json()
    assert len(rows) == 2 and not any(r["is_current"] or r["same_install"] for r in rows)


@pytest.mark.parametrize("prefix", ["/api/v1/auth", "/api/v1/users"])
async def test_api_key_cannot_end_sessions(alice_sessions, prefix):
    client, A, B = alice_sessions.client, alice_sessions.A, alice_sessions.B
    key = {"X-API-KEY": api_key_for(client, A)}
    one = client.delete(f"{prefix}/sessions/{session_id_of(B)}", headers=key)
    all_ = client.post(f"{prefix}/sessions/revoke-all", headers=key)
    assert one.status_code == all_.status_code == 403
    assert one.json()["detail"]["code"] == all_.json()["detail"]["code"] == "interactive_session_required"
    assert len(client.get("/api/v1/auth/sessions", headers=bearer(A)).json()) == 2   # nothing was revoked


async def test_scoped_virtual_key_cannot_end_sessions(alice_sessions):
    client, A, B = alice_sessions.client, alice_sessions.A, alice_sessions.B
    minted = client.post("/api/v1/auth/virtual-key", json={"scope": "workflows", "ttl_minutes": 5}, headers=bearer(A)).json()["token"]
    r = client.delete(f"/api/v1/auth/sessions/{session_id_of(B)}", headers={"Authorization": f"Bearer {minted}"})
    assert r.status_code == 403 and "Scoped token" in r.json()["detail"]
    assert {x["id"] for x in client.get("/api/v1/auth/sessions", headers=bearer(B)).json()} >= {session_id_of(A), session_id_of(B)}   # B survived


@pytest.mark.parametrize("prefix", ["/api/v1/auth", "/api/v1/users"])
async def test_session_token_ends_another_session_which_then_stops_working(alice_sessions, prefix):
    client, A, B = alice_sessions.client, alice_sessions.A, alice_sessions.B
    assert client.delete(f"{prefix}/sessions/{session_id_of(B)}", headers=bearer(A)).json()["message"] == "Session revoked successfully"
    assert client.get("/api/v1/auth/sessions", headers=bearer(B)).status_code == 401
    assert client.get("/api/v1/auth/sessions", headers=bearer(A)).status_code == 200


@pytest.mark.parametrize("prefix", ["/api/v1/auth", "/api/v1/users"])
async def test_revoke_all_keep_current_keeps_only_the_callers_session(alice_sessions, prefix):
    client, A, B = alice_sessions.client, alice_sessions.A, alice_sessions.B
    r = client.post(f"{prefix}/sessions/revoke-all?keep_current=true", headers=bearer(A))
    assert r.json()["details"]["sessions_revoked"] == 1
    assert client.get("/api/v1/auth/sessions", headers=bearer(A)).status_code == 200
    assert client.get("/api/v1/auth/sessions", headers=bearer(B)).status_code == 401


async def test_revoke_all_without_the_flag_still_signs_out_everything(alice_sessions):
    client, A = alice_sessions.client, alice_sessions.A
    assert client.post("/api/v1/auth/sessions/revoke-all", headers=bearer(A)).json()["details"]["sessions_revoked"] == 2
    assert client.get("/api/v1/auth/sessions", headers=bearer(A)).status_code == 401


async def test_revoke_all_counts_only_live_sessions_and_keeps_old_reasons(alice_sessions):
    ctx, A = alice_sessions.ctx, alice_sessions.A
    dead = await seed_session_revoked_with_reason(ctx.alice_id, "session_cap")   # raw INSERT/UPDATE helper in _sessions_http_fixtures.py
    r = ctx.client.post("/api/v1/auth/sessions/revoke-all", headers=bearer(A))
    assert r.json()["details"]["sessions_revoked"] == 2
    assert await revoke_reason_of(dead) == "session_cap"


async def test_users_sessions_alias_reports_is_current(alice_sessions):
    client, A = alice_sessions.client, alice_sessions.A
    rows = client.get("/api/v1/users/sessions", headers=bearer(A)).json()
    assert [r["id"] for r in rows if r["is_current"]] == [session_id_of(A)]


async def test_one_user_cannot_see_or_end_another_users_sessions(alice_sessions):
    client, A, B, password = alice_sessions.client, alice_sessions.A, alice_sessions.B, alice_sessions.pw
    bob = login(client, "bob", password, device="dev-cccccccc")
    alice_ids = {session_id_of(A), session_id_of(B)}
    assert alice_ids.isdisjoint({r["id"] for r in client.get("/api/v1/auth/sessions", headers=bearer(bob)).json()})
    client.delete(f"/api/v1/auth/sessions/{session_id_of(A)}", headers=bearer(bob))          # idempotent no-op
    client.post("/api/v1/auth/sessions/revoke-all?keep_current=true", headers=bearer(bob))
    assert {r["id"] for r in client.get("/api/v1/auth/sessions", headers=bearer(A)).json()} == alice_ids
