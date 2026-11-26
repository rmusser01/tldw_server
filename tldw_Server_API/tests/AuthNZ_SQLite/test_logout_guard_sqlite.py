"""POST /auth/logout cannot be used with an API key to end sessions (TASK-15649)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.AuthNZ.jwt_service import get_jwt_service
from tldw_Server_API.tests.AuthNZ_SQLite._sessions_http_fixtures import (
    api_key_for,
    bearer,
    login,
    session_id_of,
)


@pytest.fixture
def alice_sessions(two_user_client):
    """Alice signed in twice: A and B."""
    ctx = two_user_client
    a = login(ctx.client, "alice", ctx.password, device="dev-aaaaaaaa")
    b = login(ctx.client, "alice", ctx.password, device="dev-bbbbbbbb")
    return SimpleNamespace(client=ctx.client, A=a, B=b)


def test_api_key_as_bearer_cannot_log_everyone_out(alice_sessions):
    client, A, B = alice_sessions.client, alice_sessions.A, alice_sessions.B
    key = api_key_for(client, A)
    r = client.post("/api/v1/auth/logout", headers={"Authorization": f"Bearer {key}"})
    assert r.status_code == 403 and r.json()["detail"]["code"] == "interactive_session_required"
    assert {x["id"] for x in client.get("/api/v1/auth/sessions", headers=bearer(A)).json()} == {session_id_of(A), session_id_of(B)}


def test_api_key_header_cannot_log_everyone_out_with_all_devices(alice_sessions):
    client, A, B = alice_sessions.client, alice_sessions.A, alice_sessions.B
    r = client.post("/api/v1/auth/logout", json={"all_devices": True}, headers={"X-API-KEY": api_key_for(client, A)})
    assert r.status_code == 403 and r.json()["detail"]["code"] == "interactive_session_required"
    assert {x["id"] for x in client.get("/api/v1/auth/sessions", headers=bearer(B)).json()} == {session_id_of(A), session_id_of(B)}


def test_session_token_logout_ends_only_its_own_session(alice_sessions):
    client, A, B = alice_sessions.client, alice_sessions.A, alice_sessions.B
    assert client.post("/api/v1/auth/logout", headers=bearer(A)).json()["message"] == "Successfully logged out"
    assert client.get("/api/v1/auth/sessions", headers=bearer(A)).status_code == 401
    assert [x["id"] for x in client.get("/api/v1/auth/sessions", headers=bearer(B)).json()] == [session_id_of(B)]


def test_refreshed_token_can_still_log_out(alice_sessions):
    client, A, B = alice_sessions.client, alice_sessions.A, alice_sessions.B
    new = client.post("/api/v1/auth/refresh", json={"refresh_token": A["refresh_token"]}).json()["access_token"]
    assert client.post("/api/v1/auth/logout", headers={"Authorization": f"Bearer {new}"}).json()["message"] == "Successfully logged out"
    assert [x["id"] for x in client.get("/api/v1/auth/sessions", headers=bearer(B)).json()] == [session_id_of(B)]


def test_session_token_logout_all_devices_still_works(alice_sessions):
    client, A, B = alice_sessions.client, alice_sessions.A, alice_sessions.B
    assert client.post("/api/v1/auth/logout", json={"all_devices": True}, headers=bearer(A)).json()["message"].startswith("Logged out from")
    assert client.get("/api/v1/auth/sessions", headers=bearer(B)).status_code == 401


def test_token_without_a_session_claim_cannot_log_anyone_out(two_user_client, alice_sessions):
    client, A, B = alice_sessions.client, alice_sessions.A, alice_sessions.B
    bare = get_jwt_service().create_access_token(user_id=two_user_client.alice_id, username="alice", role="user")
    r = client.post("/api/v1/auth/logout", headers={"Authorization": f"Bearer {bare}"})
    assert r.status_code == 403 and r.json()["detail"]["code"] == "interactive_session_required"
    assert {x["id"] for x in client.get("/api/v1/auth/sessions", headers=bearer(A)).json()} == {session_id_of(A), session_id_of(B)}
