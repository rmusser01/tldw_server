"""Interactive-session resolution over a real multi-user SQLite app (TASK-15649)."""

from __future__ import annotations

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from tldw_Server_API.app.core.AuthNZ.interactive_session import install_id_or_none, resolve_current_session_id
from tldw_Server_API.app.core.AuthNZ.jwt_service import get_jwt_service
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.repos.sessions_repo import AuthnzSessionsRepo
from tldw_Server_API.app.core.AuthNZ.session_manager import get_session_manager
from tldw_Server_API.app.core.exceptions import DatabaseError
from tldw_Server_API.tests.AuthNZ_SQLite._sessions_http_fixtures import bearer, login, session_id_of

pytestmark = pytest.mark.asyncio


def _request(headers: dict) -> Request:
    return Request({"type": "http", "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()]})


def _access(tokens_or_jwt) -> dict:
    jwt = tokens_or_jwt["access_token"] if isinstance(tokens_or_jwt, dict) else tokens_or_jwt
    return {"Authorization": f"Bearer {jwt}"}


async def test_two_user_client_users_may_read_notifications(two_user_client):
    # smoke test for the fixture's role grant: a bare users row would get 403 here. It runs first.
    ctx = two_user_client
    tokens = login(ctx.client, "alice", ctx.password)
    r = ctx.client.get("/api/v1/notifications", headers=bearer(tokens))
    assert r.status_code == 200 and r.json()["items"] == []
    assert ctx.client.get("/api/v1/notifications/unread-count", headers=bearer(tokens)).json()["unread_count"] == 0


async def test_access_token_session_is_current(two_user_client):
    ctx = two_user_client
    manager = await get_session_manager()
    tokens = login(ctx.client, "alice", ctx.password)
    principal = AuthPrincipal(kind="user", user_id=ctx.alice_id, token_type="access")
    got = await resolve_current_session_id(_request(_access(tokens)), principal, manager)
    assert got == session_id_of(tokens)


@pytest.mark.parametrize("principal_kwargs", [
    {"kind": "service", "subject": "service:workflow-engine", "token_type": "service"},
    {"kind": "user", "token_type": "api_key"},
    {"kind": "user", "token_type": "access", "impersonation": True, "impersonated_by": 99},
])
async def test_service_api_key_and_impersonation_principals_are_not_interactive(two_user_client, principal_kwargs):
    ctx = two_user_client
    manager = await get_session_manager()
    tokens = login(ctx.client, "alice", ctx.password)
    principal = AuthPrincipal(user_id=ctx.alice_id, **principal_kwargs)
    request = _request(_access(tokens))  # a valid session token is on the request
    assert await resolve_current_session_id(request, principal, manager) is None


async def test_claim_naming_another_users_session_is_not_current(two_user_client):
    ctx = two_user_client
    manager = await get_session_manager()
    bob_tokens = login(ctx.client, "bob", ctx.password)
    forged = get_jwt_service().create_access_token(
        user_id=ctx.alice_id, username="alice", role="user",
        additional_claims={"session_id": session_id_of(bob_tokens)})   # validly signed, wrong owner
    principal = AuthPrincipal(kind="user", user_id=ctx.alice_id, token_type="access")
    assert await resolve_current_session_id(_request(_access(forged)), principal, manager) is None


async def test_claim_naming_a_revoked_session_is_not_current(two_user_client):
    ctx = two_user_client
    manager = await get_session_manager()
    tokens = login(ctx.client, "alice", ctx.password)
    await manager.revoke_session(session_id_of(tokens), revoked_by=ctx.alice_id, reason="test")
    principal = AuthPrincipal(kind="user", user_id=ctx.alice_id, token_type="access")
    assert await resolve_current_session_id(_request(_access(tokens)), principal, manager) is None


async def test_token_without_a_session_claim_is_not_interactive(two_user_client):
    ctx = two_user_client
    manager = await get_session_manager()
    bare = get_jwt_service().create_access_token(user_id=ctx.alice_id, username="alice", role="user")
    principal = AuthPrincipal(kind="user", user_id=ctx.alice_id, token_type="access")
    assert await resolve_current_session_id(_request(_access(bare)), principal, manager) is None


async def test_refreshed_token_is_still_interactive(two_user_client):
    ctx = two_user_client
    manager = await get_session_manager()
    tokens = login(ctx.client, "alice", ctx.password)
    new = ctx.client.post("/api/v1/auth/refresh", json={"refresh_token": tokens["refresh_token"]}).json()["access_token"]
    principal = AuthPrincipal(kind="user", user_id=ctx.alice_id, token_type="access")
    assert await resolve_current_session_id(_request(_access(new)), principal, manager) == session_id_of(tokens)


async def test_magic_link_token_is_interactive(two_user_client):
    # modelled on AuthNZ_SQLite/test_magic_link_redemption.py (token minted with create_magic_link_token, redeemed at /auth/magic-link/verify)
    ctx = two_user_client
    manager = await get_session_manager()
    link = get_jwt_service().create_magic_link_token(email="alice@example.com", user_id=ctx.alice_id, expires_in_minutes=5)
    r = ctx.client.post("/api/v1/auth/magic-link/verify", json={"token": link}, headers={"X-TLDW-Device-Id": "dev-magic-0001"})
    assert r.status_code == 200, r.text
    got = await resolve_current_session_id(_request(_access(r.json())), AuthPrincipal(kind="user", user_id=ctx.alice_id, token_type="access"), manager)
    magic_rows = [row["id"] for row in await manager.get_user_sessions(ctx.alice_id) if row["device_id"] == "dev-magic-0001"]
    assert magic_rows and got == magic_rows[0]


async def test_resolver_without_a_manager_uses_the_app_manager(two_user_client):
    ctx = two_user_client
    tokens = login(ctx.client, "alice", ctx.password)
    principal = AuthPrincipal(kind="user", user_id=ctx.alice_id, token_type="access")
    assert await resolve_current_session_id(_request(_access(tokens)), principal) == session_id_of(tokens)


async def test_session_read_failure_is_a_503_not_a_read_only_key(two_user_client, monkeypatch):
    ctx = two_user_client
    manager = await get_session_manager()
    tokens = login(ctx.client, "alice", ctx.password)

    async def down(self, user_id):
        raise DatabaseError("sessions unavailable")      # core.exceptions.DatabaseError, the class the repo raises

    monkeypatch.setattr(AuthnzSessionsRepo, "get_active_sessions_for_user", down)
    principal = AuthPrincipal(kind="user", user_id=ctx.alice_id, token_type="access")
    with pytest.raises(HTTPException) as caught:
        await resolve_current_session_id(_request(_access(tokens)), principal, manager)
    assert caught.value.status_code == 503 and caught.value.detail["code"] == "session_store_unavailable"


@pytest.mark.parametrize("stored,expected", [
    (None, None), ("", None), ("single-user-cookie:v1", None),
    ("single-user-cookie:v1:abcd1234", "single-user-cookie:v1:abcd1234"), ("dev-aaaaaaaa", "dev-aaaaaaaa"),
])
def test_install_id_or_none_ignores_the_bare_cookie_tag(stored, expected):
    assert install_id_or_none(stored) == expected
