from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import Request, Response

from tldw_Server_API.app.core.AuthNZ import single_user_session
from tldw_Server_API.app.core.AuthNZ.exceptions import SessionError
from tldw_Server_API.app.core.AuthNZ.single_user_session import (
    MintedSingleUserSession,
    SingleUserSessionIdentity,
    clear_single_user_session_cookie,
    mint_single_user_session,
    set_single_user_session_cookie,
    validate_single_user_session,
)


def _request(*, cookie: str | None = None) -> Request:
    headers = [(b"user-agent", b"test-browser")]
    if cookie is not None:
        headers.append((b"cookie", f"tldw_single_user_session={cookie}".encode()))
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/api/v1/test",
            "headers": headers,
            "client": ("127.0.0.1", 12345),
        }
    )


@pytest.fixture
def session_settings(monkeypatch):
    settings = SimpleNamespace(
        AUTH_MODE="single_user",
        SINGLE_USER_FIXED_ID=1,
        SINGLE_USER_SESSION_COOKIE_NAME="tldw_single_user_session",
        SINGLE_USER_SESSION_EXPIRE_DAYS=30,
        SESSION_COOKIE_SECURE=True,
        AUTH_TRUST_X_FORWARDED_FOR=False,
    )
    monkeypatch.setattr(single_user_session, "get_settings", lambda: settings)
    return settings


@pytest.mark.asyncio
async def test_mint_uses_random_tokens_constant_type_and_bounded_expiry(session_settings):
    manager = AsyncMock()
    manager.create_session.return_value = {
        "session_id": 7,
        "user_id": 1,
        "expires_at": "2026-08-09T00:00:00+00:00",
    }
    before = datetime.now(timezone.utc)

    created = await mint_single_user_session(_request(), manager)

    after = datetime.now(timezone.utc)
    kwargs = manager.create_session.await_args.kwargs
    assert kwargs["access_token"] != kwargs["refresh_token"]
    assert len(kwargs["access_token"]) >= 43
    assert kwargs["device_id"] == "single-user-cookie:v1"
    assert kwargs["user_id"] == 1
    assert kwargs["ip_address"] == "127.0.0.1"
    assert kwargs["user_agent"] == "test-browser"
    assert before + timedelta(days=30) <= kwargs["expires_at_override"] <= after + timedelta(days=30)
    assert kwargs["refresh_expires_at_override"] == kwargs["expires_at_override"]
    assert created.identity.session_id == 7
    assert created.identity.expires_at == kwargs["expires_at_override"]
    assert created.cookie_token == kwargs["access_token"]


@pytest.mark.asyncio
async def test_validate_rejects_wrong_type_and_accepts_cookie_session(session_settings):
    manager = AsyncMock()
    manager.validate_session.side_effect = [
        {"id": 3, "user_id": 1, "device_id": "browser"},
        {
            "id": 4,
            "user_id": 1,
            "device_id": "single-user-cookie:v1",
            "expires_at": "2026-08-09T00:00:00+00:00",
        },
    ]

    assert await validate_single_user_session(_request(cookie="opaque"), manager) is None
    identity = await validate_single_user_session(_request(cookie="opaque"), manager)

    assert identity is not None
    assert identity.session_id == 4
    assert identity.user_id == 1
    assert identity.expires_at == datetime(2026, 8, 9, tzinfo=timezone.utc)


@pytest.mark.asyncio
async def test_validate_fails_closed_outside_single_user_mode(session_settings):
    session_settings.AUTH_MODE = "multi_user"
    manager = AsyncMock()

    assert await validate_single_user_session(_request(cookie="opaque"), manager) is None

    manager.validate_session.assert_not_awaited()


@pytest.mark.asyncio
async def test_non_strict_validation_fails_closed_on_unexpected_error(session_settings):
    manager = AsyncMock()
    manager.validate_session.side_effect = RuntimeError("session backend unavailable")

    assert await validate_single_user_session(_request(cookie="opaque"), manager) is None

    manager.validate_session.assert_awaited_once_with("opaque")


@pytest.mark.asyncio
@pytest.mark.parametrize("strict", [False, True])
async def test_validation_contains_manager_acquisition_failure(
    session_settings,
    monkeypatch,
    strict,
):
    async def unavailable_manager():
        raise RuntimeError("session manager unavailable")

    monkeypatch.setattr(single_user_session, "get_session_manager", unavailable_manager)

    if strict:
        with pytest.raises(SessionError):
            await validate_single_user_session(_request(cookie="opaque"), strict=True)
    else:
        assert await validate_single_user_session(_request(cookie="opaque")) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("strict", [False, True])
async def test_validation_contains_malformed_session_row(session_settings, strict):
    manager = AsyncMock()
    manager.validate_session.return_value = {
        "id": "not-an-integer",
        "user_id": 1,
        "device_id": "single-user-cookie:v1",
        "expires_at": "2026-08-09T00:00:00+00:00",
    }

    if strict:
        with pytest.raises(SessionError):
            await validate_single_user_session(
                _request(cookie="opaque"),
                manager,
                strict=True,
            )
    else:
        assert await validate_single_user_session(_request(cookie="opaque"), manager) is None


def test_cookie_helpers_set_and_clear_exact_host_only_contract(session_settings):
    expires_at = datetime.now(timezone.utc) + timedelta(days=30)
    minted = MintedSingleUserSession(
        identity=SingleUserSessionIdentity(session_id=7, user_id=1, expires_at=expires_at),
        cookie_token="opaque",
    )
    response = Response()

    set_single_user_session_cookie(response, minted)
    clear_single_user_session_cookie(response)

    set_header, clear_header = response.headers.getlist("set-cookie")
    assert set_header.startswith("tldw_single_user_session=opaque;")
    assert "HttpOnly" in set_header
    assert "Max-Age=2592000" in set_header
    assert "Path=/api" in set_header
    assert "SameSite=lax" in set_header
    assert "Secure" in set_header
    assert "Domain=" not in set_header
    assert clear_header.startswith('tldw_single_user_session="')
    assert "HttpOnly" in clear_header
    assert "Max-Age=0" in clear_header
    assert "Path=/api" in clear_header
    assert "SameSite=lax" in clear_header
    assert "Secure" in clear_header
    assert "Domain=" not in clear_header
