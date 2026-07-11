from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest
from starlette.websockets import WebSocket

from tldw_Server_API.app.core.AuthNZ import websocket_session_auth as session_auth
from tldw_Server_API.app.core.AuthNZ.single_user_session import SingleUserSessionIdentity


def _websocket(
    *,
    origin: str | None = "http://127.0.0.1:3000",
    cookie: str | None = "opaque",
    headers: dict[str, str] | None = None,
    query_string: bytes = b"",
) -> WebSocket:
    raw_headers: list[tuple[bytes, bytes]] = []
    if origin is not None:
        raw_headers.append((b"origin", origin.encode("latin-1")))
    if cookie is not None:
        raw_headers.append((b"cookie", f"tldw_single_user_session={cookie}".encode("latin-1")))
    raw_headers.extend(
        (key.lower().encode("latin-1"), value.encode("latin-1"))
        for key, value in (headers or {}).items()
    )
    return WebSocket(
        {
            "type": "websocket",
            "path": "/api/v1/test/ws",
            "query_string": query_string,
            "headers": raw_headers,
            "client": ("127.0.0.1", 50000),
            "server": ("127.0.0.1", 8000),
            "scheme": "ws",
            "subprotocols": [],
            "state": {},
        },
        receive=AsyncMock(),
        send=AsyncMock(),
    )


def _identity(session_id: int = 9, user_id: int = 1) -> SingleUserSessionIdentity:
    return SingleUserSessionIdentity(
        session_id=session_id,
        user_id=user_id,
        expires_at=datetime(2026, 8, 9, tzinfo=timezone.utc),
    )


@pytest.fixture
def single_user_settings(monkeypatch):
    settings = type(
        "Settings",
        (),
        {
            "AUTH_MODE": "single_user",
            "SINGLE_USER_SESSION_COOKIE_NAME": "tldw_single_user_session",
        },
    )()
    monkeypatch.setattr(session_auth, "get_settings", lambda: settings)
    monkeypatch.setattr(
        session_auth,
        "trusted_webui_origins",
        lambda: {"http://127.0.0.1:3000"},
    )
    return settings


@pytest.mark.asyncio
@pytest.mark.parametrize("origin", [None, "null", "https://evil.example", "not-a-url"])
async def test_cookie_websocket_rejects_untrusted_origin(
    origin,
    monkeypatch,
    single_user_settings,
):
    validate = AsyncMock(return_value=_identity())
    monkeypatch.setattr(session_auth, "validate_single_user_session", validate)
    websocket = _websocket(origin=origin)

    assert await session_auth.resolve_single_user_cookie_websocket(websocket) is None
    assert websocket.state.single_user_session_id is None
    assert websocket.state.user_id is None
    assert websocket.state.single_user_cookie_websocket_attempted is True
    assert websocket.state.single_user_cookie_websocket_close_code == 4403
    validate.assert_not_awaited()


@pytest.mark.asyncio
async def test_cookie_websocket_accepts_exact_trusted_origin(
    monkeypatch,
    single_user_settings,
):
    identity = _identity()
    validate = AsyncMock(return_value=identity)
    monkeypatch.setattr(session_auth, "validate_single_user_session", validate)
    websocket = _websocket()

    assert await session_auth.resolve_single_user_cookie_websocket(websocket) == identity
    assert websocket.state.single_user_session_id == 9
    assert websocket.state.user_id == 1
    assert websocket.state.auth_principal.user_id == 1
    assert websocket.state.auth_principal.token_type == "single_user_session"
    assert websocket.state.auth_principal.roles == ["admin"]
    assert websocket.state.auth_principal.is_admin is True
    assert websocket.state.single_user_cookie_websocket_close_code is None
    validate.assert_awaited_once_with(websocket)


@pytest.mark.asyncio
async def test_cookie_websocket_rejects_invalid_session_with_4401(
    monkeypatch,
    single_user_settings,
):
    monkeypatch.setattr(session_auth, "validate_single_user_session", AsyncMock(return_value=None))
    websocket = _websocket()

    assert await session_auth.resolve_single_user_cookie_websocket(websocket) is None
    assert websocket.state.single_user_cookie_websocket_attempted is True
    assert websocket.state.single_user_cookie_websocket_close_code == 4401


@pytest.mark.asyncio
async def test_wildcard_never_authorizes_cookie_websocket(
    monkeypatch,
    single_user_settings,
):
    monkeypatch.setattr(session_auth, "trusted_webui_origins", lambda: {"*"})
    validate = AsyncMock(return_value=_identity())
    monkeypatch.setattr(session_auth, "validate_single_user_session", validate)
    websocket = _websocket()

    assert await session_auth.resolve_single_user_cookie_websocket(websocket) is None
    assert websocket.state.single_user_cookie_websocket_close_code == 4403
    validate.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("headers", "query_string"),
    [
        ({"Authorization": "Bearer explicit"}, b""),
        ({"X-API-KEY": "explicit"}, b""),
        ({"Sec-WebSocket-Protocol": "bearer, explicit"}, b""),
        ({}, b"token=explicit"),
        ({}, b"api_key=explicit"),
        ({}, b"auth_token=explicit"),
    ],
)
async def test_explicit_credentials_suppress_cookie_fallback(
    headers,
    query_string,
    monkeypatch,
    single_user_settings,
):
    validate = AsyncMock(return_value=_identity())
    monkeypatch.setattr(session_auth, "validate_single_user_session", validate)
    websocket = _websocket(headers=headers, query_string=query_string)

    assert await session_auth.resolve_single_user_cookie_websocket(websocket) is None
    assert websocket.state.single_user_cookie_websocket_attempted is False
    assert websocket.state.single_user_cookie_websocket_close_code is None
    validate.assert_not_awaited()


@pytest.mark.asyncio
async def test_absent_cookie_is_not_a_failed_cookie_auth_attempt(
    monkeypatch,
    single_user_settings,
):
    validate = AsyncMock(return_value=_identity())
    monkeypatch.setattr(session_auth, "validate_single_user_session", validate)
    websocket = _websocket(cookie=None)

    assert await session_auth.resolve_single_user_cookie_websocket(websocket) is None
    assert websocket.state.single_user_cookie_websocket_attempted is False
    assert websocket.state.single_user_cookie_websocket_close_code is None
    validate.assert_not_awaited()


@pytest.mark.asyncio
async def test_cookie_fallback_is_disabled_outside_single_user_mode(
    monkeypatch,
    single_user_settings,
):
    single_user_settings.AUTH_MODE = "multi_user"
    validate = AsyncMock(return_value=_identity())
    monkeypatch.setattr(session_auth, "validate_single_user_session", validate)
    websocket = _websocket()

    assert await session_auth.resolve_single_user_cookie_websocket(websocket) is None
    assert websocket.state.single_user_cookie_websocket_attempted is False
    validate.assert_not_awaited()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("HTTP://LOCALHOST:80", "http://localhost"),
        ("https://Example.COM:443", "https://example.com"),
        ("http://[::1]:3000", "http://[::1]:3000"),
        ("http://localhost:3000/", "http://localhost:3000"),
        ("http://localhost:3000/path", None),
        ("https://user@example.com", None),
        ("ws://localhost:3000", None),
        ("null", None),
    ],
)
def test_normalize_http_origin(raw, expected):
    assert session_auth.normalize_http_origin(raw) == expected
