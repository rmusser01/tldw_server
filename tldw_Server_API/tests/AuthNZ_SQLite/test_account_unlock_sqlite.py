"""Unlocking a locked account through the mailed link, and the anonymous routes that mail the link again."""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.Audit.unified_audit_service import AuditEventType
from tldw_Server_API.app.core.AuthNZ import account_lock, api_key_audit
from tldw_Server_API.app.core.AuthNZ.account_lock import clear_lock, is_user_security_locked
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
from tldw_Server_API.app.core.AuthNZ.jwt_service import JWTService, get_jwt_service
from tldw_Server_API.app.core.AuthNZ.profile_version import VersionedUserWriteGateway
from tldw_Server_API.app.core.AuthNZ.repos.sessions_repo import AuthnzSessionsRepo
from tldw_Server_API.app.core.AuthNZ.session_manager import SessionManager
from tldw_Server_API.tests.AuthNZ_SQLite._account_lock_fixtures import (  # noqa: F401 - lockable is a fixture
    assert_locked,
    lock_user,
    lockable,
    repair_reset_tokens_table,
    unlock_token_from,
)
from tldw_Server_API.tests.AuthNZ_SQLite._sessions_http_fixtures import api_key_for, bearer, login
from tldw_Server_API.tests.AuthNZ_SQLite._user_fixtures import create_authnz_test_user
from tldw_Server_API.tests.AuthNZ_SQLite.test_account_lock_schema_sqlite import _identity_links, _link_identity

pytestmark = pytest.mark.asyncio

LOCK = "/api/v1/auth/account/lock"
VERIFY = "/api/v1/auth/magic-link/verify"
REQUEST_LINK = "/api/v1/auth/magic-link/request"
FORGOT = "/api/v1/auth/forgot-password"
SESSIONS = "/api/v1/auth/sessions"
LOGIN = "/api/v1/auth/login"
PROFILE = "/api/v1/users/me/profile"
ME = "/api/v1/users/me"
KEYS = "/api/v1/users/api-keys"
OWNER, INTRUDER = "alice@example.com", "new@example.com"
INVALID_LINK = {"detail": "Invalid magic link token"}
DEAD_TOKEN = {"detail": "Could not validate credentials"}
INVALID_KEY = "Invalid API key"


async def _write_user(user_id: int, statement: str, *values) -> None:
    """Run a users UPDATE of the email columns through the versioned gateway (a raw one is refused)."""
    async with (await get_db_pool()).transaction() as conn:
        await VersionedUserWriteGateway("sqlite").execute_update(
            conn,
            user_id=user_id,
            profile_visible_fields=("email",),
            statement=statement,
            parameters=(*values, user_id),
        )


async def _set_email(user_id: int, email: str, *, previous: str | None = None, days_ago: int = 1) -> None:
    """Set the email; with ``previous``, record it as the address changed away from ``days_ago`` days ago."""
    if previous is None:
        await _write_user(user_id, "UPDATE users SET email = ? WHERE id = ?", email)
        return
    changed = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()
    await _write_user(
        user_id,
        "UPDATE users SET email = ?, previous_email = ?, email_changed_at = ? WHERE id = ?",
        email,
        previous,
        changed,
    )


async def _row(user_id: int) -> dict:
    return dict(await (await get_db_pool()).fetchone("SELECT * FROM users WHERE id = ?", user_id))


async def _live_sessions(user_id: int) -> list[dict]:
    return await AuthnzSessionsRepo(await get_db_pool()).get_active_sessions_for_user(user_id)


async def _blacklisted(jti: str) -> bool:
    return await (await get_db_pool()).fetchval("SELECT 1 FROM token_blacklist WHERE jti = ?", jti) is not None


async def _key_id(key: str) -> int:
    identifier = key.removeprefix("tldw_").split(".")[0]
    return int(await (await get_db_pool()).fetchval("SELECT id FROM api_keys WHERE key_id = ?", identifier))


def _claims(token: str) -> dict:
    return get_jwt_service().verify_token(token)


def _mailed(outbox: list[tuple[str, str]]) -> list[tuple[str, str, str | None]]:
    """``(to, token type, purpose)`` of every mailed link."""
    found = []
    for to, body in outbox:
        claims = _claims(re.search(r"token=([\w.\-]+)", body).group(1))
        found.append((to, claims["type"], claims.get("purpose")))
    return found


def _lock(ctx) -> dict:
    """alice signs in and locks her account (the lock mail goes out); her now dead tokens."""
    tokens = login(ctx.client, "alice", ctx.password)
    response = ctx.client.post(LOCK, headers=bearer(tokens))
    assert response.status_code == 200 and response.json()["locked"] is True, response.text
    return tokens


def _redeem(ctx, outbox, address: str, **kwargs):
    return ctx.client.post(VERIFY, json={"token": unlock_token_from(outbox, address)}, **kwargs)


async def test_unlock_link_signs_in_with_one_session_and_clears_the_lock(two_user_client, lockable):
    ctx = two_user_client
    tokens = _lock(ctx)
    response = _redeem(ctx, lockable, OWNER)
    assert response.status_code == 200, response.text
    sessions = ctx.client.get(SESSIONS, headers=bearer(response.json())).json()
    assert len(sessions) == 1 and sessions[0]["is_current"] is True
    old = ctx.client.get(SESSIONS, headers=bearer(tokens))
    assert old.status_code == 401 and old.json() == DEAD_TOKEN  # dead, and not "locked"
    assert (await _row(ctx.alice_id))["security_locked_at"] is None


async def test_unlock_link_is_single_use(two_user_client, lockable):
    ctx = two_user_client
    _lock(ctx)
    token = unlock_token_from(lockable, OWNER)
    first = ctx.client.post(VERIFY, json={"token": token})
    second = ctx.client.post(VERIFY, json={"token": token})
    assert first.status_code == 200, first.text
    assert second.status_code == 400 and second.json() == INVALID_LINK


async def test_plain_magic_link_does_not_unlock(two_user_client, lockable):
    ctx = two_user_client
    plain = JWTService().create_magic_link_token(email=OWNER, user_id=ctx.alice_id, expires_in_minutes=5)
    _lock(ctx)
    assert_locked(ctx.client.post(VERIFY, json={"token": plain}))
    assert (await _row(ctx.alice_id))["security_locked_at"] is not None
    assert await _blacklisted(_claims(plain)["jti"]) is False  # refused before it was spent


async def test_link_to_previous_address_takes_the_account_back(two_user_client, lockable):
    ctx = two_user_client
    await _set_email(ctx.alice_id, INTRUDER, previous=OWNER)
    _lock(ctx)
    response = _redeem(ctx, lockable, OWNER)
    assert response.status_code == 200, response.text
    row = await _row(ctx.alice_id)
    assert row["email"] == OWNER and row["previous_email"] is None and row["security_locked_at"] is None
    refused = ctx.client.post(LOGIN, data={"username": "alice", "password": ctx.password})
    assert refused.status_code == 401 and refused.json()["detail"] == "Incorrect username or password"


async def test_link_to_newer_address_is_dead_after_take_back(two_user_client, lockable):
    ctx = two_user_client
    await _set_email(ctx.alice_id, INTRUDER, previous=OWNER)
    _lock(ctx)
    taken_back = _redeem(ctx, lockable, OWNER)
    assert taken_back.status_code == 200, taken_back.text
    response = _redeem(ctx, lockable, INTRUDER)
    assert response.status_code == 400 and response.json() == INVALID_LINK
    assert len(await _live_sessions(ctx.alice_id)) == 1  # the owner's only


async def test_take_back_after_the_newer_address_unlocked_first_leaves_the_intruder_nothing(two_user_client, lockable):
    ctx = two_user_client
    await _set_email(ctx.alice_id, INTRUDER, previous=OWNER)
    owner = login(ctx.client, "alice", ctx.password)
    unsuspended_key = api_key_for(ctx.client, owner)  # made in the week before the lock: the unlock suspends it
    ctx.client.post(LOCK, headers=bearer(owner))
    intruder_signed_in = _redeem(ctx, lockable, INTRUDER)
    assert intruder_signed_in.status_code == 200, intruder_signed_in.text
    intruder = intruder_signed_in.json()
    created_key = api_key_for(ctx.client, intruder)
    unsuspend = ctx.client.post(f"{KEYS}/{await _key_id(unsuspended_key)}/unsuspend", headers=bearer(intruder))
    assert unsuspend.status_code == 200 and unsuspend.json()["suspended_at"] is None, unsuspend.text

    response = _redeem(ctx, lockable, OWNER)
    assert response.status_code == 200, response.text
    assert (await _row(ctx.alice_id))["email"] == OWNER
    dead_session = ctx.client.get(SESSIONS, headers=bearer(intruder))
    assert dead_session.status_code == 401 and dead_session.json() == DEAD_TOKEN
    for key in (created_key, unsuspended_key):
        refused = ctx.client.get(PROFILE, headers={"X-API-KEY": key})
        assert refused.status_code == 401 and refused.json()["detail"] == INVALID_KEY


@pytest.mark.parametrize("locked_again", [False, True], ids=["unlocked", "locked-again"])
async def test_take_back_link_wins_after_the_intruder_unlocked_and_reverted_the_email(
    two_user_client, lockable, locked_again
):
    """The intruder redeems the newer address's link first, changes the email back to the owner's (the stored
    previous address is then the email too) and makes a key and an SSO link. The owner's link, mailed with the
    first lock, still takes the account back, also when the account was locked again meanwhile (TASK-15679)."""
    ctx = two_user_client
    await _set_email(ctx.alice_id, INTRUDER, previous=OWNER)
    _lock(ctx)
    signed_in = _redeem(ctx, lockable, INTRUDER)
    assert signed_in.status_code == 200, signed_in.text
    intruder = signed_in.json()
    reverted = ctx.client.put(ME, json={"email": OWNER, "current_password": ctx.password}, headers=bearer(intruder))
    assert reverted.status_code == 200 and reverted.json()["email"] == OWNER, reverted.text
    assert (await _row(ctx.alice_id))["previous_email"] == OWNER
    created_key = api_key_for(ctx.client, intruder)
    alice = SimpleNamespace(pool=await get_db_pool(), user_id=ctx.alice_id)
    await _link_identity(alice, "intruder")
    if locked_again:  # a lock now mails one plain link; the take-back link of the first lock is still valid
        relocked = ctx.client.post(LOCK, headers=bearer(intruder))
        assert relocked.status_code == 200 and relocked.json()["locked"] is True, relocked.text

    response = _redeem(ctx, lockable, OWNER)
    assert response.status_code == 200, response.text
    row = await _row(ctx.alice_id)
    assert (row["email"], row["previous_email"], row["security_locked_at"]) == (OWNER, None, None)
    refused = ctx.client.post(LOGIN, data={"username": "alice", "password": ctx.password})
    assert refused.status_code == 401 and refused.json()["detail"] == "Incorrect username or password"
    dead_session = ctx.client.get(SESSIONS, headers=bearer(intruder))
    assert dead_session.status_code == 401 and dead_session.json() == DEAD_TOKEN
    refused_key = ctx.client.get(PROFILE, headers={"X-API-KEY": created_key})
    assert refused_key.status_code == 401 and refused_key.json()["detail"] == INVALID_KEY
    assert await _identity_links(alice) == []


async def test_take_back_committed_during_the_other_links_verify_revokes_its_session(
    two_user_client, lockable, monkeypatch
):
    ctx = two_user_client
    await _set_email(ctx.alice_id, INTRUDER, previous=OWNER)
    _lock(ctx)
    real_create_session = SessionManager.create_session
    took_back: list[tuple[bool, object]] = []

    async def _owner_takes_back_first(self, *args, **kwargs):
        """Between the intruder's unlock and its session: the owner's take-back runs to the end."""
        if not took_back:
            locked = await is_user_security_locked(ctx.alice_id)
            took_back.append((locked, await clear_lock(ctx.alice_id, take_back=OWNER)))
        return await real_create_session(self, *args, **kwargs)

    monkeypatch.setattr(SessionManager, "create_session", _owner_takes_back_first)
    response = _redeem(ctx, lockable, INTRUDER)
    assert response.status_code == 400 and response.json() == INVALID_LINK
    # The intruder's unlock had committed (not locked) and the take-back won (it returned its valid_after).
    assert len(took_back) == 1 and took_back[0][0] is False and took_back[0][1] is not None
    assert (await _row(ctx.alice_id))["email"] == OWNER
    assert len(await _live_sessions(ctx.alice_id)) == 0


async def test_take_back_link_is_refused_before_it_is_spent_when_another_account_holds_the_address(
    two_user_client, lockable
):
    ctx = two_user_client
    await _set_email(ctx.alice_id, INTRUDER, previous=OWNER)
    _lock(ctx)
    await _set_email(ctx.bob_id, OWNER)  # direct setup: the email-change route refuses it
    token = unlock_token_from(lockable, OWNER)
    response = ctx.client.post(VERIFY, json={"token": token})
    assert response.status_code == 400 and response.json() == INVALID_LINK
    assert (await _row(ctx.alice_id))["security_locked_at"] is not None
    assert await _blacklisted(_claims(token)["jti"]) is False  # the link was not consumed


async def test_take_back_refused_when_the_address_is_taken_after_the_link_was_checked(
    two_user_client, lockable, monkeypatch
):
    ctx = two_user_client
    await _set_email(ctx.alice_id, INTRUDER, previous=OWNER)
    _lock(ctx)
    real_clear_lock = account_lock.clear_lock

    async def _bob_takes_the_address_first(user_id, **kwargs):
        await _set_email(ctx.bob_id, OWNER)
        return await real_clear_lock(user_id, **kwargs)

    monkeypatch.setattr(account_lock, "clear_lock", _bob_takes_the_address_first)
    token = unlock_token_from(lockable, OWNER)
    response = ctx.client.post(VERIFY, json={"token": token})
    assert response.status_code == 400 and response.json() == INVALID_LINK
    row = await _row(ctx.alice_id)
    assert row["security_locked_at"] is not None and row["email"] == INTRUDER
    assert await _blacklisted(_claims(token)["jti"]) is True  # checked, then spent: the owner asks for a new link
    assert len(await _live_sessions(ctx.alice_id)) == 0


async def test_take_back_link_past_its_window_does_not_unlock(two_user_client):
    """The window can close between rule 1 and the redeem: the link then unlocks nothing."""
    ctx = two_user_client
    await _set_email(ctx.alice_id, INTRUDER, previous=OWNER, days_ago=8)
    await lock_user(ctx.alice_id)
    take_back = {"email": OWNER, "restore_email": True, "purpose": "account_unlock"}
    redeemed = await account_lock.redeem_unlock(
        await _row(ctx.alice_id), take_back, via="magic_link", ip_address=None, user_agent=None
    )
    row = await _row(ctx.alice_id)
    assert redeemed is False
    assert row["security_locked_at"] is not None and row["email"] == INTRUDER


async def test_second_take_back_link_unlocks_once_its_address_is_current(two_user_client):
    """Two links to the previous address (the lock mail and a resent one): the first takes the account back;
    once it is locked again, the second finds its address is the account's email and unlocks it plainly."""
    ctx = two_user_client
    await _set_email(ctx.alice_id, INTRUDER, previous=OWNER)
    await lock_user(ctx.alice_id)
    take_back = {"email": OWNER, "restore_email": True, "purpose": "account_unlock"}

    async def _redeem_take_back_link() -> bool:
        return await account_lock.redeem_unlock(
            await _row(ctx.alice_id), take_back, via="magic_link", ip_address=None, user_agent=None
        )

    first = await _redeem_take_back_link()
    await lock_user(ctx.alice_id)
    second = await _redeem_take_back_link()
    row = await _row(ctx.alice_id)
    assert (first, second) == (True, True)
    assert (row["security_locked_at"], row["email"], row["previous_email"]) == (None, OWNER, None)


async def test_send_me_a_new_link_for_a_username_mails_nothing(two_user_client, lockable):
    """A username is not an address of the account: no link goes to any of its addresses."""
    ctx = two_user_client
    carol = await create_authnz_test_user(
        await get_db_pool(), username="carol@example.com", email="carol-mail@example.com", is_verified=True
    )
    await _set_email(carol, "carol-mail@example.com", previous="carol-old@example.com")
    await lock_user(carol)
    response = ctx.client.post(REQUEST_LINK, json={"email": "carol@example.com"})
    assert response.json() == {"message": "If the account exists, a sign-in link has been sent", "details": None}
    assert lockable == []


async def test_send_me_a_new_link_mails_the_lock_template_to_the_typed_address_only(two_user_client, lockable):
    ctx = two_user_client
    await _set_email(ctx.alice_id, OWNER, previous="old@example.com")
    await lock_user(ctx.alice_id)  # no mail yet
    response = ctx.client.post(REQUEST_LINK, json={"email": "old@example.com"})
    assert response.status_code == 200
    assert response.json() == {"message": "If the account exists, a sign-in link has been sent", "details": None}
    assert [to for to, _ in lockable] == ["old@example.com"]
    claims = _claims(unlock_token_from(lockable, "old@example.com"))
    assert claims["purpose"] == "account_unlock" and claims["restore_email"] is True


async def test_forgot_password_on_a_locked_account_mails_the_unlock_link_not_a_reset_link(two_user_client, lockable):
    ctx = two_user_client
    await repair_reset_tokens_table()  # a reset link could be stored, so the difference is the lock's
    await lock_user(ctx.alice_id)
    response = ctx.client.post(FORGOT, json={"email": OWNER})
    assert response.json() == {"message": "If the email exists, a reset link has been sent"}
    assert [to for to, _ in lockable] == [OWNER]
    claims = _claims(unlock_token_from(lockable, OWNER))
    assert claims["type"] == "magic_link" and claims["purpose"] == "account_unlock"
    assert await (await get_db_pool()).fetchval("SELECT COUNT(*) FROM password_reset_tokens") == 0


async def test_anonymous_answers_do_not_reveal_the_lock(two_user_client, lockable):
    ctx = two_user_client
    await repair_reset_tokens_table()  # bob's forgot-password really works
    await lock_user(ctx.alice_id)

    def _answers(username: str, email: str) -> list[tuple[int, dict]]:
        forgot = ctx.client.post(FORGOT, json={"email": email})
        request_link = ctx.client.post(REQUEST_LINK, json={"email": email})
        wrong_password = ctx.client.post(LOGIN, data={"username": username, "password": "Wrong-Pass-2026!"})
        return [(r.status_code, r.json()) for r in (forgot, request_link, wrong_password)]

    locked = _answers("alice", OWNER)
    unlocked = _answers("bob", "bob@example.com")
    nobody = _answers("nobody", "nobody@example.com")
    assert locked == unlocked == nobody
    assert [m for m in _mailed(lockable) if m[0] != "nobody@example.com"] == [
        (OWNER, "magic_link", "account_unlock"),
        (OWNER, "magic_link", "account_unlock"),
        ("bob@example.com", "password_reset", None),
        ("bob@example.com", "magic_link", None),
    ]


@pytest.mark.real_audit
async def test_unlock_writes_an_audit_event(real_audit_service, two_user_client, lockable):
    ctx = two_user_client
    _lock(ctx)
    response = _redeem(ctx, lockable, OWNER, headers={"User-Agent": "unlock-agent/1.0"})
    assert response.status_code == 200, response.text

    async def _events() -> list[dict]:
        service = await api_key_audit._get_or_create_audit_service(ctx.alice_id)
        return await service.query_events(
            user_id=str(ctx.alice_id), event_types=[AuditEventType.AUTH_ACCOUNT_UNLOCKED]
        )

    events = ctx.client.portal.call(_events)
    assert [(e["event_type"], e["context_user_agent"], json.loads(e["metadata"])) for e in events] == [
        ("auth.account.unlocked", "unlock-agent/1.0", {"via": "magic_link", "took_back_address": False})
    ]
