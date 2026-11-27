"""API keys obey the account lock: refused while locked, recent ones suspended after an unlock."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tldw_Server_API.app.core.Audit.unified_audit_service import MandatoryAuditWriteError
from tldw_Server_API.app.core.AuthNZ import api_key_audit
from tldw_Server_API.app.core.AuthNZ.api_key_manager import get_api_key_manager

from tldw_Server_API.app.core.AuthNZ.account_lock import set_lock
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
from tldw_Server_API.app.core.AuthNZ.profile_version import VersionedUserWriteGateway
from tldw_Server_API.app.core.AuthNZ.repos.api_keys_repo import AuthnzApiKeysRepo
from tldw_Server_API.app.core.AuthNZ.repos.users_repo import AuthnzUsersRepo
from tldw_Server_API.tests.AuthNZ_SQLite._account_lock_fixtures import lock_user, unlock_user
from tldw_Server_API.tests.AuthNZ_SQLite._sessions_http_fixtures import api_key_for, bearer, login

pytestmark = pytest.mark.asyncio

PROFILE = "/api/v1/users/me/profile"
MCP_TOOLS = "/api/v1/mcp/tools"
KEYS = "/api/v1/users/api-keys"


def _unsuspend(key_id: int) -> str:
    return f"{KEYS}/{key_id}/unsuspend"


def _anonymous_answer(ctx) -> dict:
    """What the MCP tools route answers a key it does not accept: the permission refusal of an anonymous caller."""
    garbage = ctx.client.get(MCP_TOOLS, headers={"X-API-KEY": "tldw_00000000.not-a-key"})
    assert garbage.status_code == 403 and garbage.json()["detail"]["reason_code"] == "permission_denied"
    return garbage.json()


async def _key_row(key: str) -> dict:
    """The api_keys row of a plaintext key, found by its identifier."""
    identifier = key.removeprefix("tldw_").split(".")[0]
    return dict(await (await get_db_pool()).fetchone("SELECT * FROM api_keys WHERE key_id = ?", identifier))


async def _deactivate(user_id: int) -> None:
    async with (await get_db_pool()).transaction() as conn:
        await VersionedUserWriteGateway("sqlite").execute_update(
            conn,
            user_id=user_id,
            profile_visible_fields=("is_active",),
            statement="UPDATE users SET is_active = ? WHERE id = ?",
            parameters=(False, user_id),
        )


async def _backdate(key_id: int, days: int) -> None:
    created = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    await (await get_db_pool()).execute("UPDATE api_keys SET created_at = ? WHERE id = ?", created, key_id)


async def _two_keys_across_a_lock(ctx):
    """alice's old key (10 days old) and new key, then lock and unlock; returns ids, keys and fresh tokens."""
    tokens = login(ctx.client, "alice", ctx.password)
    old_key, new_key = api_key_for(ctx.client, tokens), api_key_for(ctx.client, tokens)
    old_id, new_id = (await _key_row(old_key))["id"], (await _key_row(new_key))["id"]
    await _backdate(old_id, 10)
    await lock_user(ctx.alice_id)
    await unlock_user(ctx.alice_id)
    return old_id, old_key, new_id, new_key, login(ctx.client, "alice", ctx.password)


async def test_sqlite_lookup_carries_the_owners_state(two_user_client) -> None:
    ctx = two_user_client
    key = api_key_for(ctx.client, login(ctx.client, "alice", ctx.password))
    repo = AuthnzApiKeysRepo(db_pool=await get_db_pool())
    identifier = key.removeprefix("tldw_").split(".")[0]
    before_by_id = await repo.fetch_active_by_key_id(identifier)
    before_by_hash = await repo.fetch_active_by_hash_candidates([before_by_id["key_hash"]])
    await set_lock(ctx.alice_id)
    after_by_id = await repo.fetch_active_by_key_id(identifier)
    after_by_hash = await repo.fetch_active_by_hash_candidates([before_by_id["key_hash"]])
    for before in (before_by_id, before_by_hash):
        assert before["owner_security_locked_at"] is None and before["owner_is_active"] in (1, True)
        assert before["suspended_at"] is None
    for after in (after_by_id, after_by_hash):
        assert after["owner_security_locked_at"] is not None


async def test_api_key_of_a_locked_owner_is_invalid_and_usage_is_not_recorded(two_user_client) -> None:
    ctx = two_user_client
    key = api_key_for(ctx.client, login(ctx.client, "alice", ctx.password))
    working = ctx.client.get(PROFILE, headers={"X-API-KEY": key})
    assert working.status_code == 200 and working.json()["user"]["id"] == ctx.alice_id
    before = await _key_row(key)
    await lock_user(ctx.alice_id)
    # Clear the per-key usage throttle so a usage write would happen if the key were accepted.
    (await get_api_key_manager())._clear_usage_touch(before["id"])
    response = ctx.client.get(PROFILE, headers={"X-API-KEY": key})
    after = await _key_row(key)
    assert response.status_code == 401 and response.json()["detail"] == "Invalid API key"
    assert after["usage_count"] == before["usage_count"] and after["last_used_at"] == before["last_used_at"]


async def test_mcp_http_refuses_api_key_of_locked_owner(two_user_client) -> None:
    ctx = two_user_client
    key = api_key_for(ctx.client, login(ctx.client, "alice", ctx.password))
    before = ctx.client.get(MCP_TOOLS, headers={"X-API-KEY": key})
    assert before.status_code == 200 and "tools" in before.json()
    await lock_user(ctx.alice_id)
    after = ctx.client.get(MCP_TOOLS, headers={"X-API-KEY": key})
    assert after.status_code == 403 and after.json() == _anonymous_answer(ctx)


async def test_api_key_of_an_inactive_owner_is_invalid_on_mcp_http(two_user_client) -> None:
    ctx = two_user_client
    key = api_key_for(ctx.client, login(ctx.client, "alice", ctx.password))
    before = ctx.client.get(MCP_TOOLS, headers={"X-API-KEY": key})
    assert before.status_code == 200 and "tools" in before.json()
    await _deactivate(ctx.alice_id)
    after = ctx.client.get(MCP_TOOLS, headers={"X-API-KEY": key})
    assert after.status_code == 403 and after.json() == _anonymous_answer(ctx)


async def test_recent_api_key_stays_suspended_after_unlock(two_user_client) -> None:
    ctx = two_user_client
    old_id, old_key, new_id, new_key, tokens = await _two_keys_across_a_lock(ctx)
    old_response = ctx.client.get(PROFILE, headers={"X-API-KEY": old_key})
    assert old_response.status_code == 200 and old_response.json()["user"]["id"] == ctx.alice_id
    new_response = ctx.client.get(PROFILE, headers={"X-API-KEY": new_key})
    assert new_response.status_code == 401 and new_response.json()["detail"] == "Invalid API key"
    listed = {k["id"]: k for k in ctx.client.get(KEYS, headers=bearer(tokens)).json()}
    assert listed[new_id]["suspended_at"] is not None and listed[old_id]["suspended_at"] is None


async def test_unsuspend_restores_one_key_and_needs_a_browser_session(two_user_client) -> None:
    ctx = two_user_client
    old_id, old_key, new_id, new_key, tokens = await _two_keys_across_a_lock(ctx)
    refused = ctx.client.post(_unsuspend(new_id), headers={"X-API-KEY": old_key})
    assert refused.status_code == 403 and refused.json()["detail"]["code"] == "interactive_session_required"
    assert (await _key_row(new_key))["suspended_at"] is not None
    restored = ctx.client.post(_unsuspend(new_id), headers=bearer(tokens))
    assert restored.status_code == 200 and restored.json()["id"] == new_id
    assert restored.json()["suspended_at"] is None and "key_hash" not in restored.json()
    working = ctx.client.get(PROFILE, headers={"X-API-KEY": new_key})
    assert working.status_code == 200 and working.json()["user"]["id"] == ctx.alice_id


async def test_cannot_unsuspend_another_users_key(two_user_client) -> None:
    ctx = two_user_client
    _old_id, _old_key, new_id, new_key, _tokens = await _two_keys_across_a_lock(ctx)
    bob = login(ctx.client, "bob", ctx.password)
    response = ctx.client.post(_unsuspend(new_id), headers=bearer(bob))
    assert response.status_code == 404 and response.json()["detail"]["code"] == "api_key_not_suspended"
    assert (await _key_row(new_key))["suspended_at"] is not None


async def test_a_suspended_key_cannot_be_rotated(two_user_client) -> None:
    ctx = two_user_client
    old_id, old_key, new_id, _new_key, _tokens = await _two_keys_across_a_lock(ctx)
    response = ctx.client.post(f"{KEYS}/{new_id}/rotate", json={}, headers={"X-API-KEY": old_key})
    rows = await (await get_db_pool()).fetchall(
        "SELECT id, status FROM api_keys WHERE user_id = ? AND status = 'active'", ctx.alice_id
    )
    assert response.status_code == 409 and response.json()["detail"]["code"] == "api_key_suspended"
    assert sorted(r["id"] for r in rows) == sorted([old_id, new_id])


async def test_bobs_key_keeps_working_while_alice_is_locked(two_user_client) -> None:
    ctx = two_user_client
    key = api_key_for(ctx.client, login(ctx.client, "bob", ctx.password))
    await lock_user(ctx.alice_id)
    response = ctx.client.get(PROFILE, headers={"X-API-KEY": key})
    assert response.status_code == 200 and response.json()["user"]["id"] == ctx.bob_id


async def test_unsuspend_is_undone_when_its_audit_cannot_be_written(two_user_client, monkeypatch) -> None:
    ctx = two_user_client
    _old_id, old_key, new_id, new_key, tokens = await _two_keys_across_a_lock(ctx)

    async def _failing_audit(**_kwargs) -> None:
        raise MandatoryAuditWriteError("Mandatory audit persistence unavailable")

    monkeypatch.setattr(api_key_audit, "emit_mandatory_api_key_management_audit", _failing_audit)
    response = ctx.client.post(_unsuspend(new_id), headers=bearer(tokens))
    assert response.status_code == 503 and response.json()["detail"] == "Mandatory audit persistence unavailable"
    listed = {k["id"]: k for k in ctx.client.get(KEYS, headers=bearer(tokens)).json()}
    assert listed[new_id]["suspended_at"] is not None
    refused = ctx.client.get(PROFILE, headers={"X-API-KEY": new_key})
    assert refused.status_code == 401 and refused.json()["detail"] == "Invalid API key"


async def test_unsuspend_writes_its_audit_event(two_user_client, monkeypatch) -> None:
    ctx = two_user_client
    _old_id, _old_key, new_id, _new_key, tokens = await _two_keys_across_a_lock(ctx)
    calls: list[dict] = []
    real = api_key_audit.emit_mandatory_api_key_management_audit

    async def _recording_audit(**kwargs) -> None:
        calls.append(kwargs)
        await real(**kwargs)

    monkeypatch.setattr(api_key_audit, "emit_mandatory_api_key_management_audit", _recording_audit)
    response = ctx.client.post(_unsuspend(new_id), headers=bearer(tokens))
    assert response.status_code == 200 and response.json()["suspended_at"] is None
    assert [(c["action"], c["resource_id"], c["user_id"], c["actor_kind"]) for c in calls] == [
        ("api_key.unsuspend", str(new_id), ctx.alice_id, "user")
    ]


async def test_admin_cannot_rotate_a_suspended_key(two_user_client) -> None:
    ctx = two_user_client
    _old_id, _old_key, new_id, new_key, _tokens = await _two_keys_across_a_lock(ctx)
    await AuthnzUsersRepo(await get_db_pool()).assign_role_if_missing(user_id=ctx.bob_id, role_name="admin")
    admin = login(ctx.client, "bob", ctx.password)
    response = ctx.client.post(
        f"/api/v1/admin/users/{ctx.alice_id}/api-keys/{new_id}/rotate", json={}, headers=bearer(admin)
    )
    rows = await (await get_db_pool()).fetchall(
        "SELECT id FROM api_keys WHERE user_id = ? AND status = 'active'", ctx.alice_id
    )
    assert response.status_code == 409 and response.json()["detail"]["code"] == "api_key_suspended"
    assert new_id in [r["id"] for r in rows] and len(rows) == 2
    assert (await _key_row(new_key))["suspended_at"] is not None
