"""TASK-15679: no token or sign-in of a locked account works, and an unlock does not revive old tokens."""

from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime, timedelta

import pytest
from fastapi import Depends

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
    get_auth_principal,
    get_password_service_dep,
    require_expected_user,
)
from tldw_Server_API.app.core.Audit.unified_audit_service import AuditEventType
from tldw_Server_API.app.core.AuthNZ import api_key_audit
from tldw_Server_API.app.core.AuthNZ.account_lock import AccountSecurityLockedError, set_lock
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
from tldw_Server_API.app.core.AuthNZ.email_change import authorize_own_email_change
from tldw_Server_API.app.core.AuthNZ.password_service import get_password_service
from tldw_Server_API.app.core.AuthNZ.jwt_service import get_jwt_service
from tldw_Server_API.app.core.AuthNZ.repos.sessions_repo import AuthnzSessionsRepo
from tldw_Server_API.app.core.AuthNZ.session_manager import get_session_manager
from tldw_Server_API.app.core.AuthNZ.token_blacklist import get_token_blacklist, reset_token_blacklist
from tldw_Server_API.app.core.DB_Management.user_profile_writes import update_user_email
from tldw_Server_API.app.services.auth_service import store_password_reset_token
from tldw_Server_API.tests.AuthNZ_SQLite._account_lock_fixtures import (
    assert_locked,
    lock_user,
    repair_reset_tokens_table,
    unlock_user,
)
from tldw_Server_API.tests.AuthNZ_SQLite._sessions_http_fixtures import bearer, login, session_id_of

pytestmark = pytest.mark.asyncio

CREDENTIALS = ["access_token", "refresh_token", "virtual_jwt", "impersonation_jwt", "password"]
VIRTUAL_KEY_ROUTE = "/api/v1/scheduler/workflows"  # guarded by require_token_scope("workflows")
PROFILE_ROUTE = "/api/v1/users/me/profile"
NEW_PASSWORD = "N3w-Str0ng-Pass!2026"
DEAD_TOKEN = {"detail": "Could not validate credentials"}


def _virtual_jwt(ctx) -> dict:
    token = get_jwt_service().create_virtual_access_token(
        user_id=ctx.alice_id, username="alice", role="user", scope="workflows"
    )
    return {"Authorization": f"Bearer {token}"}


def _impersonation_jwt(ctx) -> dict:
    """Bob impersonating alice, minted as admin_impersonation.create_impersonation_token mints it."""
    token = get_jwt_service().create_access_token(
        user_id=ctx.alice_id,
        username="alice",
        role="user",
        expires_delta=timedelta(minutes=15),
        additional_claims={"impersonated_by": ctx.bob_id, "impersonation": True},
    )
    return {"Authorization": f"Bearer {token}"}


def _credential(ctx, kind: str):
    """Arrange one of alice's credentials (before any lock); return the request that uses it."""
    client = ctx.client
    if kind == "password":
        return lambda: client.post("/api/v1/auth/login", data={"username": "alice", "password": ctx.password})
    if kind == "access_token":
        tokens = login(client, "alice", ctx.password)
        return lambda: client.get("/api/v1/auth/sessions", headers=bearer(tokens))
    if kind == "refresh_token":
        tokens = login(client, "alice", ctx.password)
        return lambda: client.post("/api/v1/auth/refresh", json={"refresh_token": tokens["refresh_token"]})
    if kind == "virtual_jwt":
        headers = _virtual_jwt(ctx)
        return lambda: client.get(VIRTUAL_KEY_ROUTE, headers=headers)
    headers = _impersonation_jwt(ctx)
    return lambda: client.get(PROFILE_ROUTE, headers=headers)


async def _password_hash(user_id: int) -> str:
    row = await (await get_db_pool()).fetchone("SELECT password_hash FROM users WHERE id = ?", user_id)
    return row["password_hash"]


async def _reset_token(user_id: int, email: str) -> str:
    """A reset token and its stored record, as forgot-password makes them."""
    await repair_reset_tokens_table()
    jwt_service = get_jwt_service()
    token = jwt_service.create_password_reset_token(user_id=user_id, email=email, expires_in_hours=1)
    async with (await get_db_pool()).transaction() as conn:
        await store_password_reset_token(
            conn,
            user_id=user_id,
            token_hash=jwt_service.hash_password_reset_token(token),
            expires_at=datetime.utcnow() + timedelta(hours=1),
            ip_address="127.0.0.1",
        )
    return token


def _reset(client, token: str):
    return client.post("/api/v1/auth/reset-password", json={"token": token, "new_password": NEW_PASSWORD})


async def _live_sessions(user_id: int) -> list[dict]:
    return await AuthnzSessionsRepo(await get_db_pool()).get_active_sessions_for_user(user_id)


@pytest.mark.parametrize("kind", CREDENTIALS)
async def test_no_credential_works_while_locked(two_user_client, kind) -> None:
    use = _credential(two_user_client, kind)
    await lock_user(two_user_client.alice_id)
    assert_locked(use())


async def test_virtual_and_impersonation_jwts_work_before_the_lock(two_user_client) -> None:
    """The routes the lock tests use accept these tokens, so the refusals above are the lock's."""
    ctx = two_user_client
    assert ctx.client.get(VIRTUAL_KEY_ROUTE, headers=_virtual_jwt(ctx)).json() == []
    profile = ctx.client.get(PROFILE_ROUTE, headers=_impersonation_jwt(ctx))
    assert profile.status_code == 200 and profile.json()["user"]["id"] == ctx.alice_id


async def test_wrong_password_on_a_locked_account_answers_like_any_wrong_password(two_user_client) -> None:
    await lock_user(two_user_client.alice_id)
    response = two_user_client.client.post(
        "/api/v1/auth/login", data={"username": "alice", "password": "Wrong-Pass-2026!"}
    )
    assert response.status_code == 401 and response.json()["detail"] == "Incorrect username or password"


async def test_token_minted_before_unlock_stays_dead_after_unlock(two_user_client) -> None:
    ctx = two_user_client
    virtual, impersonation = _virtual_jwt(ctx), _impersonation_jwt(ctx)
    pre_lock = login(ctx.client, "alice", ctx.password)
    await lock_user(ctx.alice_id)
    await unlock_user(ctx.alice_id)

    for route, headers in ((VIRTUAL_KEY_ROUTE, virtual), (PROFILE_ROUTE, impersonation)):
        response = ctx.client.get(route, headers=headers)
        assert response.status_code == 401 and response.json() == DEAD_TOKEN, route
    # Its blacklist reason is still "account_locked", but the account is no longer locked.
    old_session = ctx.client.get("/api/v1/auth/sessions", headers=bearer(pre_lock))
    assert old_session.status_code == 401 and old_session.json() == DEAD_TOKEN

    fresh = login(ctx.client, "alice", ctx.password)
    sessions = ctx.client.get("/api/v1/auth/sessions", headers=bearer(fresh))
    assert sessions.status_code == 200 and [s["id"] for s in sessions.json()] == [session_id_of(fresh)]


async def test_refresh_of_a_locked_account_is_423_not_500(two_user_client) -> None:
    """The lock only: the refresh token is not revoked, so the refusal comes after the user fetch."""
    tokens = login(two_user_client.client, "alice", two_user_client.password)
    await set_lock(two_user_client.alice_id)
    assert_locked(two_user_client.client.post("/api/v1/auth/refresh", json={"refresh_token": tokens["refresh_token"]}))


async def test_reset_password_token_minted_before_the_lock_is_refused_while_locked(two_user_client) -> None:
    ctx = two_user_client
    token = await _reset_token(ctx.alice_id, "alice@example.com")
    before = await _password_hash(ctx.alice_id)
    await lock_user(ctx.alice_id)
    assert_locked(_reset(ctx.client, token))
    assert await _password_hash(ctx.alice_id) == before


async def test_reset_token_minted_before_unlock_is_refused_after_unlock(two_user_client) -> None:
    ctx = two_user_client
    token = await _reset_token(ctx.alice_id, "alice@example.com")
    before = await _password_hash(ctx.alice_id)
    await lock_user(ctx.alice_id)
    await unlock_user(ctx.alice_id)
    response = _reset(ctx.client, token)
    assert response.status_code == 400 and response.json()["detail"] == "Invalid or expired reset token"
    assert await _password_hash(ctx.alice_id) == before


async def test_reset_token_for_an_address_the_account_no_longer_has_is_refused(two_user_client) -> None:
    ctx = two_user_client
    token = await _reset_token(ctx.alice_id, "alice@example.com")
    before = await _password_hash(ctx.alice_id)
    async with (await get_db_pool()).transaction() as conn:
        await update_user_email(conn, backend="sqlite", user_id=ctx.alice_id, email="other@example.com")
    response = _reset(ctx.client, token)
    assert response.status_code == 400 and response.json()["detail"] == "Invalid or expired reset token"
    assert await _password_hash(ctx.alice_id) == before


async def _reset_signs_alice_out(ctx, *, cold_blacklist: bool) -> None:
    """A reset token for the current address resets the password and signs every session out."""
    signed_in = login(ctx.client, "alice", ctx.password)
    token = await _reset_token(ctx.alice_id, "Alice@Example.com")
    before = await _password_hash(ctx.alice_id)
    if cold_blacklist:
        await reset_token_blacklist()  # the reset is the process's first use of the blacklist
    else:
        ctx.client.portal.call(get_token_blacklist().initialize)
    response = _reset(ctx.client, token)
    assert response.status_code == 200, response.text
    assert response.json() == {"message": "Password has been reset successfully"}
    assert await _password_hash(ctx.alice_id) != before
    old = ctx.client.get("/api/v1/auth/sessions", headers=bearer(signed_in))
    assert old.status_code == 401 and old.json() == DEAD_TOKEN
    rows = await (await get_db_pool()).fetchall(
        "SELECT is_revoked, revoke_reason FROM sessions WHERE user_id = ?", ctx.alice_id
    )
    assert [(bool(r["is_revoked"]), r["revoke_reason"]) for r in rows] == [(True, "Password reset")]


async def test_reset_token_still_works_for_an_unlocked_account(two_user_client) -> None:
    """The new checks refuse only what they must: a current token for the current address passes."""
    await _reset_signs_alice_out(two_user_client, cold_blacklist=False)


async def test_reset_password_works_as_the_first_blacklist_use_of_the_process(two_user_client) -> None:
    await _reset_signs_alice_out(two_user_client, cold_blacklist=True)


@contextmanager
def _overridden(ctx, dependency, override):
    ctx.client.app.dependency_overrides[dependency] = override
    try:
        yield
    finally:
        ctx.client.app.dependency_overrides.pop(dependency, None)


async def _email(user_id: int) -> str:
    return (await (await get_db_pool()).fetchone("SELECT email FROM users WHERE id = ?", user_id))["email"]


async def test_change_password_started_before_a_lock_is_refused_at_its_write(two_user_client) -> None:
    """The lock lands after the request is signed in and before its transaction begins (the route resolves
    the password service between the two)."""
    ctx = two_user_client
    tokens = login(ctx.client, "alice", ctx.password)
    before = await _password_hash(ctx.alice_id)

    async def _lock_then_the_password_service():
        await set_lock(ctx.alice_id)
        return get_password_service()

    with _overridden(ctx, get_password_service_dep, _lock_then_the_password_service):
        response = ctx.client.post(
            "/api/v1/users/change-password",
            json={"current_password": ctx.password, "new_password": NEW_PASSWORD},
            headers=bearer(tokens),
        )
    assert_locked(response)
    assert await _password_hash(ctx.alice_id) == before


async def test_email_change_is_refused_while_locked(two_user_client) -> None:
    """At the route (the lock lands between sign-in and the transaction: ``require_expected_user`` resolves
    there), then the gate itself on a transaction of the now locked account."""
    ctx = two_user_client
    tokens = login(ctx.client, "alice", ctx.password)

    async def _lock_once_signed_in(principal=Depends(get_auth_principal)) -> None:
        await set_lock(ctx.alice_id)

    with _overridden(ctx, require_expected_user, _lock_once_signed_in):
        response = ctx.client.patch(
            PROFILE_ROUTE,
            json={
                "updates": [{"key": "identity.email", "value": "alice2@example.com"}],
                "current_password": ctx.password,
            },
            headers=bearer(tokens),
        )
    assert_locked(response)
    assert await _email(ctx.alice_id) == "alice@example.com"
    async with (await get_db_pool()).transaction() as conn:
        with pytest.raises(AccountSecurityLockedError):
            await authorize_own_email_change(
                db_conn=conn, is_postgres=False, user_id=ctx.alice_id,
                new_emails=("alice2@example.com",), current_password=ctx.password,
            )


async def test_admin_cannot_impersonate_a_locked_user(two_user_client) -> None:
    ctx = two_user_client
    from tldw_Server_API.app.core.AuthNZ.repos.users_repo import AuthnzUsersRepo

    await AuthnzUsersRepo(await get_db_pool()).assign_role_if_missing(user_id=ctx.bob_id, role_name="admin")
    admin = bearer(login(ctx.client, "bob", ctx.password))
    await lock_user(ctx.alice_id)
    assert_locked(ctx.client.post(f"/api/v1/admin/impersonate/{ctx.alice_id}/token", headers=admin))


async def test_capped_insert_refuses_a_locked_user(two_user_client) -> None:
    ctx = two_user_client
    login(ctx.client, "alice", ctx.password)
    count_before = len(await _live_sessions(ctx.alice_id))
    await set_lock(ctx.alice_id)  # the lock only, sessions untouched
    with pytest.raises(AccountSecurityLockedError):
        await (await get_session_manager()).create_session(user_id=ctx.alice_id, access_token="a", refresh_token="r")
    assert len(await _live_sessions(ctx.alice_id)) == count_before == 1


@pytest.mark.real_audit
async def test_cap_eviction_is_audited(real_audit_service, two_user_client) -> None:
    ctx = two_user_client
    signed_in = [login(ctx.client, "alice", ctx.password, device=f"lock-dev-{i:05d}") for i in range(5)]

    async def _evicted_events() -> list[dict]:
        service = await api_key_audit._get_or_create_audit_service(ctx.alice_id)
        return await service.query_events(
            user_id=str(ctx.alice_id), event_types=[AuditEventType.AUTH_SESSION_EVICTED]
        )

    events = ctx.client.portal.call(_evicted_events)
    assert [(e["context_session_id"], json.loads(e["metadata"])) for e in events] == [
        (str(session_id_of(signed_in[-1])), {"evicted_session_ids": [session_id_of(signed_in[0])]})
    ]


async def test_other_user_is_untouched_by_alices_lock(two_user_client) -> None:
    ctx = two_user_client
    bob = login(ctx.client, "bob", ctx.password)
    await lock_user(ctx.alice_id)
    bob_sessions_response = ctx.client.get("/api/v1/auth/sessions", headers=bearer(bob))
    assert bob_sessions_response.status_code == 200 and len(bob_sessions_response.json()) == 1
