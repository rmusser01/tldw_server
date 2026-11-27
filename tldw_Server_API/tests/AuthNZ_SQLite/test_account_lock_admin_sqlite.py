"""The administrator's side of the account lock: security unlock, listings, own-email step-up, the CLI."""

from __future__ import annotations

import pytest

from tldw_Server_API.app.core.AuthNZ.account_lock import set_lock
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
from tldw_Server_API.app.core.AuthNZ.password_service import PasswordService
from tldw_Server_API.app.core.AuthNZ.reset_admin_password import reset_user_password
from tldw_Server_API.app.core.AuthNZ.repos.users_repo import AuthnzUsersRepo
from tldw_Server_API.tests.AuthNZ_SQLite._account_lock_fixtures import lock_user
from tldw_Server_API.tests.AuthNZ_SQLite._sessions_http_fixtures import api_key_for, bearer, login

pytestmark = pytest.mark.asyncio

REASON = "owner called support"


def _unlock(user_id: int) -> str:
    return f"/api/v1/admin/users/{user_id}/security-unlock"


def _user(user_id: int) -> str:
    return f"/api/v1/admin/users/{user_id}"


async def _row(user_id: int) -> dict:
    return dict(await (await get_db_pool()).fetchone("SELECT * FROM users WHERE id = ?", user_id))


@pytest.fixture
async def admin_client(two_user_client):
    """``two_user_client`` with bob holding the ``admin`` role; his password is ``ctx.password``."""
    await AuthnzUsersRepo(await get_db_pool()).assign_role_if_missing(user_id=two_user_client.bob_id, role_name="admin")
    return two_user_client


def _admin_headers(ctx) -> dict:
    return bearer(login(ctx.client, "bob", ctx.password))


async def test_admin_unlock_needs_step_up_and_clears_the_lock(admin_client) -> None:
    ctx = admin_client
    admin = _admin_headers(ctx)
    await lock_user(ctx.alice_id)
    refused = ctx.client.post(_unlock(ctx.alice_id), json={"reason": REASON}, headers=admin)
    still_locked = (await _row(ctx.alice_id))["security_locked_at"]
    done = ctx.client.post(
        _unlock(ctx.alice_id), json={"reason": REASON, "admin_password": ctx.password}, headers=admin
    )
    again = ctx.client.post(
        _unlock(ctx.alice_id), json={"reason": REASON, "admin_password": ctx.password}, headers=admin
    )
    assert refused.status_code == 400 and still_locked is not None
    assert done.json() == {"unlocked": True}
    assert again.json() == {"unlocked": False}
    assert (await _row(ctx.alice_id))["security_locked_at"] is None
    assert login(ctx.client, "alice", ctx.password)["access_token"]


async def test_admin_unlock_also_suspends_recent_keys_and_kills_old_tokens(admin_client) -> None:
    ctx = admin_client
    admin = _admin_headers(ctx)
    old = login(ctx.client, "alice", ctx.password)
    api_key_for(ctx.client, old)
    await lock_user(ctx.alice_id)
    response = ctx.client.post(
        _unlock(ctx.alice_id), json={"reason": REASON, "admin_password": ctx.password}, headers=admin
    )
    keys = await (await get_db_pool()).fetchall("SELECT suspended_at FROM api_keys WHERE user_id = ?", ctx.alice_id)
    stale = ctx.client.get("/api/v1/users/me/profile", headers=bearer(old))
    assert response.json() == {"unlocked": True}
    assert len(keys) == 1 and keys[0]["suspended_at"] is not None
    assert stale.status_code == 401 and "detail" in stale.json()


async def test_admin_listing_and_detail_show_the_lock(admin_client) -> None:
    ctx = admin_client
    admin = _admin_headers(ctx)
    await set_lock(ctx.alice_id)
    listed = {u["id"]: u for u in ctx.client.get("/api/v1/admin/users", headers=admin).json()["users"]}
    detail = ctx.client.get(_user(ctx.alice_id), headers=admin).json()
    assert listed[ctx.alice_id]["security_locked_at"] is not None
    assert listed[ctx.bob_id]["security_locked_at"] is None
    assert detail["security_locked_at"] is not None


async def test_is_locked_is_no_longer_an_accepted_field(admin_client) -> None:
    ctx = admin_client
    response = ctx.client.put(_user(ctx.bob_id), json={"is_locked": True}, headers=_admin_headers(ctx))
    assert response.status_code == 400 and response.json()["detail"] == "No fields to update"


async def test_admin_changing_their_own_email_needs_step_up(admin_client) -> None:
    ctx = admin_client
    admin = _admin_headers(ctx)
    refused = ctx.client.put(_user(ctx.bob_id), json={"email": "new-admin@example.com"}, headers=admin)
    unchanged = (await _row(ctx.bob_id))["email"]
    done = ctx.client.put(
        _user(ctx.bob_id),
        json={"email": "new-admin@example.com", "reason": "moved mailbox", "admin_password": ctx.password},
        headers=admin,
    )
    assert refused.status_code == 400 and unchanged == "bob@example.com"
    assert "updated successfully" in done.json()["message"]
    assert (await _row(ctx.bob_id))["email"] == "new-admin@example.com"


async def test_admin_changing_another_users_email_is_unchanged(admin_client) -> None:
    ctx = admin_client
    response = ctx.client.put(_user(ctx.alice_id), json={"email": "alice2@example.com"}, headers=_admin_headers(ctx))
    assert "updated successfully" in response.json()["message"]
    assert (await _row(ctx.alice_id))["email"] == "alice2@example.com"


async def test_reset_admin_password_cli_clears_a_lock(admin_client) -> None:
    ctx = admin_client
    await set_lock(ctx.bob_id)
    ok = await reset_user_password("bob", "Tr1cky-Harbor-Quartz!")
    row = await _row(ctx.bob_id)
    verified, _ = PasswordService().verify_password("Tr1cky-Harbor-Quartz!", row["password_hash"])
    assert ok is True and verified
    assert row["security_locked_at"] is None and row["credentials_valid_after"] is not None
