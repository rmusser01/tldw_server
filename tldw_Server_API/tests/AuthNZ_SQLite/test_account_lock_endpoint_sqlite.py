"""POST /auth/account/lock, GET /auth/account/lock-availability and the lock mail."""

from __future__ import annotations

import json
import smtplib
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.Audit.unified_audit_service import AuditEventType
from tldw_Server_API.app.core.AuthNZ import account_lock, api_key_audit, email_service
from tldw_Server_API.app.core.AuthNZ.account_lock import LockAvailability, send_lock_mail
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
from tldw_Server_API.app.core.AuthNZ.jwt_service import get_jwt_service
from tldw_Server_API.app.core.AuthNZ.profile_version import VersionedUserWriteGateway
from tldw_Server_API.app.core.AuthNZ.session_manager import SessionManager
from tldw_Server_API.app.core.AuthNZ.exceptions import SessionError
from tldw_Server_API.tests.AuthNZ_SQLite._account_lock_fixtures import assert_locked, lockable, unlock_token_from  # noqa: F401
from tldw_Server_API.tests.AuthNZ_SQLite._sessions_http_fixtures import api_key_for, bearer, login

pytestmark = pytest.mark.asyncio

LOCK = "/api/v1/auth/account/lock"
AVAILABILITY = "/api/v1/auth/account/lock-availability"
SESSIONS = "/api/v1/auth/sessions"
LOGIN = "/api/v1/auth/login"


async def _locked_at(user_id: int):
    row = await (await get_db_pool()).fetchone("SELECT security_locked_at FROM users WHERE id = ?", user_id)
    return row["security_locked_at"]


async def _write_user(user_id: int, field: str, statement: str, *values) -> None:
    """Run a users UPDATE through the versioned gateway (a raw UPDATE of profile-visible columns is refused)."""
    async with (await get_db_pool()).transaction() as conn:
        await VersionedUserWriteGateway("sqlite").execute_update(
            conn,
            user_id=user_id,
            profile_visible_fields=(field,),
            statement=statement,
            parameters=(*values, user_id),
        )


def _claims(token: str) -> dict:
    return get_jwt_service().verify_token(token, token_type="magic_link")


async def test_lock_signs_everything_out_and_mails_the_owner(two_user_client, lockable):
    ctx = two_user_client
    first, second = login(ctx.client, "alice", ctx.password), login(ctx.client, "alice", ctx.password)
    body = ctx.client.post(LOCK, headers=bearer(second)).json()
    assert body == {"locked": True, "email_sent": True, "sent_to": ["a***@example.com"]}
    assert_locked(ctx.client.get(SESSIONS, headers=bearer(first)))
    assert_locked(ctx.client.get(SESSIONS, headers=bearer(second)))
    assert [to for to, _ in lockable] == ["alice@example.com"]
    assert "https://app.example.com/auth/magic-link?token=" in lockable[0][1]
    assert "Locked at " in lockable[0][1]
    rows = await (await get_db_pool()).fetchall(
        "SELECT is_revoked, revoke_reason, revoked_by FROM sessions WHERE user_id = ?", ctx.alice_id
    )
    assert [(bool(r["is_revoked"]), r["revoke_reason"], r["revoked_by"]) for r in rows] == [
        (True, "account_locked", ctx.alice_id)
    ] * 2


async def test_lock_mail_also_goes_to_a_recent_previous_address(two_user_client, lockable):
    ctx = two_user_client
    changed = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    await _write_user(
        ctx.alice_id,
        "email",
        "UPDATE users SET email = email, previous_email = ?, email_changed_at = ? WHERE id = ?",
        "old@example.com",
        changed,
    )
    ctx.client.post(LOCK, headers=bearer(login(ctx.client, "alice", ctx.password)))
    assert sorted(to for to, _ in lockable) == ["alice@example.com", "old@example.com"]
    assert _claims(unlock_token_from(lockable, "old@example.com"))["restore_email"] is True
    assert _claims(unlock_token_from(lockable, "alice@example.com"))["restore_email"] is False
    assert _claims(unlock_token_from(lockable, "alice@example.com"))["purpose"] == "account_unlock"
    restores = "makes this address the account's email again"
    assert restores in dict(lockable)["old@example.com"]
    assert restores not in dict(lockable)["alice@example.com"]


async def test_api_key_cannot_lock(two_user_client, lockable):
    ctx = two_user_client
    key = api_key_for(ctx.client, login(ctx.client, "alice", ctx.password))
    response = ctx.client.post(LOCK, headers={"X-API-KEY": key})
    assert response.status_code == 403 and response.json()["detail"]["code"] == "interactive_session_required"
    assert await _locked_at(ctx.alice_id) is None


@pytest.mark.parametrize(
    "setup, reason",
    [
        ("mock_mail", "mail_not_configured"),
        ("no_public_url", "public_url_not_configured"),
        ("unverified", "email_not_verified"),
    ],
)
async def test_lock_is_refused_when_the_link_could_not_be_delivered(
    two_user_client, lockable, monkeypatch, setup, reason
):
    ctx = two_user_client
    tokens = login(ctx.client, "alice", ctx.password)
    if setup == "mock_mail":
        monkeypatch.setenv("EMAIL_PROVIDER", "mock")
    elif setup == "no_public_url":
        monkeypatch.setenv("PUBLIC_WEB_BASE_URL", "")
    else:
        await _write_user(ctx.alice_id, "is_verified", "UPDATE users SET is_verified = ? WHERE id = ?", False)
    monkeypatch.setattr(email_service, "_email_service", None)
    response = ctx.client.post(LOCK, headers=bearer(tokens))
    assert response.status_code == 409 and response.json()["detail"] == {"code": "lock_unavailable", "reason": reason}
    assert ctx.client.get(SESSIONS, headers=bearer(tokens)).status_code == 200
    assert await _locked_at(ctx.alice_id) is None


async def test_lock_holds_when_mail_fails(two_user_client, lockable, monkeypatch):
    ctx = two_user_client
    tokens = login(ctx.client, "alice", ctx.password)

    def _fail(self, message) -> None:
        raise smtplib.SMTPException("down")

    monkeypatch.setattr(email_service.EmailService, "_deliver_smtp_message", _fail)
    response = ctx.client.post(LOCK, headers=bearer(tokens))
    assert response.json() == {"locked": True, "email_sent": False, "sent_to": []}
    assert_locked(ctx.client.post(LOGIN, data={"username": "alice", "password": ctx.password}))


async def test_lock_holds_when_session_revoke_fails(two_user_client, lockable, monkeypatch):
    ctx = two_user_client
    tokens = login(ctx.client, "alice", ctx.password)

    async def _fail(self, *args, **kwargs):
        raise SessionError("boom")

    monkeypatch.setattr(SessionManager, "revoke_all_user_sessions", _fail)
    response = ctx.client.post(LOCK, headers=bearer(tokens))
    assert response.json()["locked"] is True
    assert_locked(ctx.client.get(SESSIONS, headers=bearer(tokens)))


async def test_availability_answer(two_user_client, lockable):
    ctx = two_user_client
    tokens = login(ctx.client, "alice", ctx.password)
    key = api_key_for(ctx.client, tokens)
    assert ctx.client.get(AVAILABILITY, headers=bearer(tokens)).json() == {
        "available": True, "reason": None, "sent_to": ["a***@example.com"],
    }
    assert ctx.client.get(AVAILABILITY, headers={"X-API-KEY": key}).json()["detail"]["code"] == "interactive_session_required"


@pytest.mark.real_audit
async def test_lock_writes_audit_events(real_audit_service, two_user_client, lockable):
    ctx = two_user_client
    tokens = login(ctx.client, "alice", ctx.password)
    session_id = int(get_jwt_service().decode_access_token(tokens["access_token"])["session_id"])
    ctx.client.post(LOCK, headers={**bearer(tokens), "User-Agent": "audit-agent/1.0"})

    async def _events() -> list[dict]:
        service = await api_key_audit._get_or_create_audit_service(ctx.alice_id)
        return await service.query_events(
            user_id=str(ctx.alice_id),
            event_types=[AuditEventType.AUTH_ACCOUNT_LOCKED, AuditEventType.AUTH_ACCOUNT_LOCK_EMAIL_SENT],
        )

    events = ctx.client.portal.call(_events)
    by_type = {e["event_type"]: e for e in events}
    assert sorted(by_type) == ["auth.account.lock_email_sent", "auth.account.locked"]
    locked = by_type["auth.account.locked"]
    assert locked["context_session_id"] == str(session_id)
    assert (locked["context_ip_address"], locked["context_user_agent"]) == ("unknown", "audit-agent/1.0")  # what the route resolves for a TestClient
    assert json.loads(by_type["auth.account.lock_email_sent"]["metadata"]) == {"recipient_count": 1}


async def test_resent_link_has_no_when_and_where(two_user_client, lockable):
    user = {"id": two_user_client.alice_id, "username": "alice", "email": "alice@example.com"}
    sent = await send_lock_mail(user, locked_at=None, device="Chrome on macOS", ip_address="203.0.113.9")
    assert sent == ["a***@example.com"]
    assert "token=" in lockable[0][1] and "Locked at" not in lockable[0][1]


async def test_resent_link_goes_only_to_the_named_address(two_user_client, lockable):
    user = {
        "id": two_user_client.alice_id, "username": "alice", "email": "alice@example.com",
        "previous_email": "old@example.com", "email_changed_at": datetime.now(timezone.utc) - timedelta(days=1),
    }

    async def _send(only_address: str) -> tuple[list[str], list[str]]:
        lockable.clear()
        sent = await send_lock_mail(
            user, locked_at=None, device="Chrome on macOS", ip_address="203.0.113.9", only_address=only_address
        )
        return sent, [to for to, _ in lockable]

    assert await _send("  ALICE@example.com ") == (["a***@example.com"], ["alice@example.com"])
    assert await _send("nobody@example.com") == ([], [])


def test_single_user_mode_cannot_lock(monkeypatch):
    monkeypatch.setattr(account_lock, "get_settings", lambda: SimpleNamespace(AUTH_MODE="single_user"))
    user = {"email": "alice@example.com", "is_verified": True}
    assert account_lock.lock_availability(user) == LockAvailability(
        available=False, reason="single_user_mode", sent_to=["a***@example.com"]
    )


async def test_lock_holds_when_the_mail_step_raises(two_user_client, lockable, monkeypatch):
    ctx = two_user_client
    tokens = login(ctx.client, "alice", ctx.password)

    async def _raise(*args, **kwargs):
        raise OSError("mail down")

    monkeypatch.setattr(account_lock, "send_lock_mail", _raise)
    response = ctx.client.post(LOCK, headers=bearer(tokens))
    assert response.json() == {"locked": True, "email_sent": False, "sent_to": []}
    assert await _locked_at(ctx.alice_id) is not None


def test_smtp_connection_has_a_timeout(monkeypatch):
    recorded: dict = {}

    def _fake_smtp(host, port, **kwargs):
        recorded.update(kwargs)
        return nullcontext(
            SimpleNamespace(starttls=lambda: None, login=lambda *a: None, send_message=lambda *a, **k: None)
        )

    monkeypatch.setattr(smtplib, "SMTP", _fake_smtp)
    service = email_service.EmailService()
    service._deliver_smtp_message(object())
    assert recorded["timeout"] == 15
