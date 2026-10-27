"""Branches of core/AuthNZ/websocket_principal.resolve_ws_principal (TASK-13950.7).

The route matrix (tests/AuthNZ/test_websocket_route_auth_matrix.py) covers missing,
revoked and valid credentials end to end. These pin the branches it cannot reach
cheaply: API-key scope, and a failing credential store.
"""

from __future__ import annotations

import contextlib
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException, WebSocket, WebSocketException
from starlette.types import Message

from tldw_Server_API.app.core.AuthNZ import websocket_principal
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal

pytestmark = pytest.mark.unit


def _websocket_with_api_key() -> WebSocket:
    scope = {
        "type": "websocket",
        "path": "/ws",
        "headers": [(b"x-api-key", b"tldw-test-key")],
        "query_string": b"",
        "client": ("127.0.0.1", 50000),
    }
    return WebSocket(scope, receive=AsyncMock(), send=AsyncMock())


def _api_key_principal(monkeypatch: pytest.MonkeyPatch, key_scope: str, **claims: Any) -> None:
    """Stand in for get_auth_principal: a multi-user key, its scope where HTTP auth puts it."""

    async def authenticate(request):
        request.state._api_key_scope = key_scope
        return AuthPrincipal(kind="api_key", user_id=7, api_key_id=3, **claims)

    monkeypatch.setattr(websocket_principal, "get_auth_principal", authenticate)


@pytest.mark.asyncio
async def test_a_multi_user_api_key_with_the_scope_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    _api_key_principal(monkeypatch, "read")

    principal = await websocket_principal.resolve_ws_principal(_websocket_with_api_key(), required_scope="read")

    assert (principal.user_id, principal.api_key_id) == (7, 3)


@pytest.mark.asyncio
async def test_an_api_key_without_the_scope_is_refused_4403(monkeypatch: pytest.MonkeyPatch) -> None:
    _api_key_principal(monkeypatch, "read")

    with pytest.raises(WebSocketException) as refused:
        await websocket_principal.resolve_ws_principal(_websocket_with_api_key(), required_scope="write")

    assert (refused.value.code, refused.value.reason) == (4403, "insufficient_scope")


@pytest.mark.asyncio
async def test_admin_claims_do_not_bypass_the_api_key_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stricter than HTTP's require_api_key_scope, on purpose; see the resolver's docstring."""
    _api_key_principal(monkeypatch, "read", roles=["admin"], is_admin=True)

    with pytest.raises(WebSocketException) as refused:
        await websocket_principal.resolve_ws_principal(_websocket_with_api_key(), required_scope="write")

    assert refused.value.code == 4403


@pytest.mark.asyncio
async def test_a_failing_credential_store_closes_1011(monkeypatch: pytest.MonkeyPatch) -> None:
    async def store_down(request):
        raise HTTPException(status_code=500, detail="Authentication failed due to internal error")

    monkeypatch.setattr(websocket_principal, "get_auth_principal", store_down)

    with pytest.raises(WebSocketException) as refused:
        await websocket_principal.resolve_ws_principal(_websocket_with_api_key())

    assert (refused.value.code, refused.value.reason) == (1011, "auth_unavailable")


def _websocket(headers: list[tuple[bytes, bytes]]) -> WebSocket:
    scope = {"type": "websocket", "path": "/ws", "headers": headers, "query_string": b"", "client": ("127.0.0.1", 50000)}
    return WebSocket(scope, receive=AsyncMock(), send=AsyncMock())


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        ([(b"authorization", b"Bearer t")], True),
        ([(b"authorization", b"bEaReR  t ")], True),  # the scheme is case-insensitive
        ([(b"authorization", b"Basic dXNlcjpwdw==")], False),  # a proxy's, not ours
        ([(b"authorization", b"Bearer ")], False),  # a scheme with no credential
        ([(b"x-api-key", b"k")], True),
        ([(b"sec-websocket-protocol", b"bearer, t")], True),
        ([(b"sec-websocket-protocol", b"x-api-key, k")], False),  # not a channel this module reads
        ([(b"cookie", b"session=s")], False),  # ambient, not presented
    ],
)
def test_has_ws_credential_counts_only_the_resolver_channels(headers: list[tuple[bytes, bytes]], expected: bool) -> None:
    assert websocket_principal.has_ws_credential(_websocket(headers)) is expected


@pytest.mark.asyncio
async def test_verify_ws_credential_hands_a_credential_to_the_http_verifier(monkeypatch: pytest.MonkeyPatch) -> None:
    """A route shim's credential is verified as the header it stands for, never parsed."""
    seen: list[tuple[str | None, str | None]] = []

    async def verifier(request):
        seen.append((request.headers.get("authorization"), request.headers.get("x-api-key")))
        return AuthPrincipal(kind="user", user_id=7)

    monkeypatch.setattr(websocket_principal, "get_auth_principal", verifier)

    by_token = await websocket_principal.verify_ws_credential(_websocket([]), token="a.b.c")
    by_key = await websocket_principal.verify_ws_credential(_websocket([]), api_key="tldw_key")

    assert (by_token.user_id, by_key.user_id) == (7, 7)
    assert seen == [("Bearer a.b.c", None), (None, "tldw_key")]


@pytest.mark.asyncio
async def test_verify_ws_credential_refuses_a_credential_no_header_can_carry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Query strings decode to any text; a header is latin-1. Refuse instead of crashing."""
    verifier = AsyncMock()
    monkeypatch.setattr(websocket_principal, "get_auth_principal", verifier)

    with pytest.raises(WebSocketException) as refused:
        await websocket_principal.verify_ws_credential(_websocket([]), token="€")

    assert (refused.value.code, refused.value.reason) == (4401, "invalid_token")
    verifier.assert_not_awaited()


@pytest.mark.parametrize("credential", [{}, {"token": "", "api_key": ""}])
@pytest.mark.asyncio
async def test_verify_ws_credential_without_a_credential_never_reaches_the_cookie(
    monkeypatch: pytest.MonkeyPatch, credential: dict[str, str]
) -> None:
    """The cookie needs resolve_ws_principal's Origin check; this helper must not verify it."""
    verifier = AsyncMock()
    monkeypatch.setattr(websocket_principal, "get_auth_principal", verifier)
    cross_site_cookie = _websocket([(b"cookie", b"tldw_session=s"), (b"origin", b"https://evil.example")])

    with pytest.raises(WebSocketException) as refused:
        await websocket_principal.verify_ws_credential(cross_site_cookie, **credential)

    assert (refused.value.code, refused.value.reason) == (4401, "authentication_required")
    verifier.assert_not_awaited()


@pytest.mark.asyncio
async def test_verify_ws_credential_replaces_a_proxy_authorization_header(monkeypatch: pytest.MonkeyPatch) -> None:
    """A browser behind a Basic-auth proxy sends ``Authorization: Basic``; the token must win."""
    seen: list[str | None] = []

    async def verifier(request):
        seen.append(request.headers.get("authorization"))
        return AuthPrincipal(kind="user", user_id=7)

    monkeypatch.setattr(websocket_principal, "get_auth_principal", verifier)
    behind_proxy = _websocket([(b"authorization", b"Basic dXNlcjpwdw==")])

    principal = await websocket_principal.verify_ws_credential(behind_proxy, token="a.b.c")

    assert principal.user_id == 7
    assert seen == ["Bearer a.b.c"]


@pytest.mark.parametrize("signed_url", [True, False])
@pytest.mark.asyncio
async def test_a_signed_url_token_hides_the_cookie_only_when_the_route_says_it_is_not_a_credential(
    monkeypatch: pytest.MonkeyPatch, signed_url: bool
) -> None:
    """Sandbox's signed log-stream URL carries its HMAC in ``?token=``. That is no credential, so
    it must not stop the single-user cookie; any other ``?token=`` still does."""
    from datetime import datetime, timedelta, timezone
    from types import SimpleNamespace

    from tldw_Server_API.app.core.AuthNZ import websocket_session_auth
    from tldw_Server_API.app.core.AuthNZ.single_user_session import SingleUserSessionIdentity

    settings = SimpleNamespace(AUTH_MODE="single_user", SINGLE_USER_SESSION_COOKIE_NAME="tldw_session")
    identity = SingleUserSessionIdentity(
        session_id=9, user_id=1, expires_at=datetime.now(timezone.utc) + timedelta(hours=1)
    )
    monkeypatch.setattr(websocket_session_auth, "get_settings", lambda: settings)
    monkeypatch.setattr(websocket_session_auth, "trusted_webui_origins", lambda: {"http://testserver"})
    monkeypatch.setattr(websocket_session_auth, "validate_single_user_session", AsyncMock(return_value=identity))
    monkeypatch.setattr(
        websocket_session_auth,
        "get_single_user_instance",
        lambda: SimpleNamespace(username="single-user", email=None, roles=[], role="user", permissions=[], is_admin=False),
    )
    websocket = _websocket([(b"cookie", b"tldw_session=s"), (b"origin", b"http://testserver")])
    websocket.scope["query_string"] = b"token=0123abcd&exp=4102444800"

    if signed_url:
        principal = await websocket_principal.resolve_ws_principal(websocket, ignore_signed_url_token=True)
        assert principal.user_id == 1
    else:
        with pytest.raises(WebSocketException) as refused:
            await websocket_principal.resolve_ws_principal(websocket)
        assert refused.value.code == 4401


@pytest.mark.parametrize(
    ("offered", "echoed"),
    [
        (b"bearer, t", "bearer"),
        (b"Bearer, t", "Bearer"),  # a browser compares the echo exactly
        (b"bearer", None),  # no credential follows the marker
        (b"x-api-key, k", None),
    ],
)
def test_auth_subprotocol_echoes_the_offered_marker_verbatim(offered: bytes, echoed: str | None) -> None:
    assert websocket_principal.auth_subprotocol(_websocket([(b"sec-websocket-protocol", offered)])) == echoed


_ACCESS = AuthPrincipal(kind="user", user_id=7, token_type="access")
_SCOPED_KEY = AuthPrincipal(kind="api_key", user_id=7, api_key_id=3, token_type="api_key")
_SINGLE_USER_KEY = AuthPrincipal(kind="user", user_id=1, token_type="api_key", subject="single_user")


@pytest.mark.parametrize(
    ("headers", "principal", "forwarded", "scopes"),
    [
        # HTTP took the JWT, so the key beside it is not the caller's credential.
        ([(b"authorization", b"Bearer a.b.c"), (b"x-api-key", b"other")], _ACCESS, {"authorization": "Bearer a.b.c"}, None),
        ([(b"sec-websocket-protocol", b"bearer, a.b.c")], _ACCESS, {"authorization": "Bearer a.b.c"}, None),
        # HTTP took the X-API-KEY over a refused bearer.
        ([(b"authorization", b"Bearer stale"), (b"x-api-key", b"key")], _SCOPED_KEY, {"x-api-key": "key"}, ["read", "write"]),
        # A non-JWT bearer is an API key; only a scoped (multi-user) key has scopes to check.
        ([(b"authorization", b"Bearer key")], _SINGLE_USER_KEY, {"x-api-key": "key"}, None),
    ],
)
@pytest.mark.asyncio
async def test_the_verified_credential_and_a_scoped_keys_scopes_are_recorded(
    monkeypatch: pytest.MonkeyPatch,
    headers: list[tuple[bytes, bytes]],
    principal: AuthPrincipal,
    forwarded: dict[str, str],
    scopes: list[str] | None,
) -> None:
    """Persona forwards its caller to an internal chat call and checks a key's scopes per tool, so it
    needs what HTTP verified: exactly the winning credential, never a loser beside it."""

    async def verifier(request):
        request.state._api_key_scope = '["write", "read"]'
        return principal

    monkeypatch.setattr(websocket_principal, "get_auth_principal", verifier)
    websocket = _websocket(headers)

    assert await websocket_principal.resolve_ws_principal(websocket) is principal
    assert websocket.state.ws_credential_headers == forwarded
    assert websocket.state.ws_api_key_scopes == scopes


@pytest.mark.asyncio
async def test_a_refused_credential_records_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    async def refuse(request):
        raise HTTPException(status_code=401, detail="Could not validate credentials")

    monkeypatch.setattr(websocket_principal, "get_auth_principal", refuse)
    websocket = _websocket([(b"authorization", b"Bearer a.b.c")])

    with pytest.raises(WebSocketException):
        await websocket_principal.resolve_ws_principal(websocket)

    assert not hasattr(websocket.state, "ws_credential_headers")


@pytest.mark.asyncio
async def test_the_query_shim_warns_once_per_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    """Persona re-verifies its caller every few seconds; the deprecation is still one line per socket."""
    from loguru import logger

    async def verifier(request):
        return _ACCESS

    monkeypatch.setattr(websocket_principal, "get_auth_principal", verifier)
    websocket = _websocket([])
    websocket.scope["query_string"] = b"token=a.b.c"
    warned: list[str] = []
    sink = logger.add(lambda message: warned.append(message.record["message"]), level="WARNING")
    try:
        for _ in range(3):
            await websocket_principal.resolve_ws_principal_or_deprecated_query(websocket)
    finally:
        logger.remove(sink)

    assert len([message for message in warned if "Deprecated WebSocket credential channel" in message]) == 1
    assert not [message for message in warned if "a.b.c" in message]


@pytest.mark.asyncio
async def test_verify_ws_credential_verifies_only_the_credential_it_is_given(monkeypatch: pytest.MonkeyPatch) -> None:
    """A first-frame route's token stands alone: the handshake's own X-API-KEY must not rescue it when
    HTTP refuses the token with 401. The bearer subprotocol still rides with X-API-KEY, as the two
    HTTP headers do."""
    seen: list[tuple[str | None, str | None]] = []

    async def verifier(request):
        seen.append((request.headers.get("authorization"), request.headers.get("x-api-key")))
        return _ACCESS

    monkeypatch.setattr(websocket_principal, "get_auth_principal", verifier)
    handshake = [(b"authorization", b"Bearer header-token"), (b"x-api-key", b"header-key")]

    frame = _websocket(handshake)
    by_frame = await websocket_principal.verify_ws_credential(frame, token="frame-token")
    by_subprotocol = await websocket_principal.resolve_ws_principal(
        _websocket([(b"sec-websocket-protocol", b"bearer, a.b.c"), (b"x-api-key", b"header-key")])
    )

    assert (by_frame.user_id, by_subprotocol.user_id) == (7, 7)
    assert seen == [("Bearer frame-token", None), ("Bearer a.b.c", "header-key")]
    assert frame.state.ws_credential_headers == {"authorization": "Bearer frame-token"}  # what persona would forward


# First-message-frame auth: resolve_ws_principal_first_frame (audio streams, voice assistant).

_COOKIE = AuthPrincipal(kind="user", user_id=1, token_type="single_user_session", subject="single_user")


def _frame_socket(*frames: object, headers: list[tuple[bytes, bytes]] | None = None) -> tuple[WebSocket, list[Message]]:
    """A socket whose client connects, then sends ``frames`` (text, or a dict sent as JSON), then waits.

    Returns the socket and the messages the server sent, so a test can see whether it accepted."""
    import asyncio
    import json

    inbox: list[Message] = [{"type": "websocket.connect"}]
    for frame in frames:
        inbox.append({"type": "websocket.receive", "text": frame if isinstance(frame, str) else json.dumps(frame)})
    sent: list[Message] = []

    async def receive() -> Message:
        if inbox:
            return inbox.pop(0)
        await asyncio.Event().wait()  # the client says nothing more
        raise AssertionError("unreachable")

    async def send(message: Message) -> None:
        sent.append(message)

    scope = {
        "type": "websocket",
        "path": "/ws",
        "headers": headers or [],
        "query_string": b"",
        "client": ("127.0.0.1", 50000),
    }
    socket = WebSocket(scope, receive=receive, send=send)
    socket.scope["_inbox"] = inbox  # what the server left unread
    return socket, sent


def _cookie(monkeypatch: pytest.MonkeyPatch, principal: AuthPrincipal | None) -> None:
    """Stand in for the cookie resolver: a session (Origin already checked), or none."""

    async def resolve(websocket, **_kwargs):
        if principal is None:
            return None
        websocket.state.auth_principal = principal
        return object()

    monkeypatch.setattr(websocket_principal, "resolve_single_user_cookie_websocket", resolve)


def _frame_verifier(monkeypatch: pytest.MonkeyPatch, principal: AuthPrincipal = _ACCESS) -> list[str | None]:
    seen: list[str | None] = []

    async def verifier(request):
        seen.append(request.headers.get("authorization"))
        if request.headers.get("authorization") == "Bearer revoked":
            raise HTTPException(status_code=401, detail="Could not validate credentials")
        return principal

    monkeypatch.setattr(websocket_principal, "get_auth_principal", verifier)
    return seen


def _accepted(sent: list[Message]) -> bool:
    return any(message["type"] == "websocket.accept" for message in sent)


@pytest.mark.asyncio
async def test_first_frame_token_is_verified_alone_and_nothing_after_it_is_read(monkeypatch: pytest.MonkeyPatch) -> None:
    _cookie(monkeypatch, None)
    seen = _frame_verifier(monkeypatch)
    websocket, sent = _frame_socket({"type": "auth", "token": "a.b.c"}, {"type": "config"})

    principal = await websocket_principal.resolve_ws_principal_first_frame(websocket, required_scope="read")

    assert principal is _ACCESS
    assert seen == ["Bearer a.b.c"]
    assert _accepted(sent)  # a frame cannot be read otherwise
    assert websocket.scope["_inbox"] == [{"type": "websocket.receive", "text": '{"type": "config"}'}]


@pytest.mark.parametrize("first", [{"type": "config"}, {"type": "audio", "data": "AAAA"}, "not json", ["auth"]])
@pytest.mark.asyncio
async def test_a_first_frame_that_is_not_auth_is_refused_4400_unprocessed(
    monkeypatch: pytest.MonkeyPatch, first: object
) -> None:
    _cookie(monkeypatch, None)
    seen = _frame_verifier(monkeypatch)
    websocket, _sent = _frame_socket(first, {"type": "auth", "token": "a.b.c"})

    with pytest.raises(WebSocketException) as refused:
        await websocket_principal.resolve_ws_principal_first_frame(websocket)

    assert (refused.value.code, refused.value.reason) == (4400, "expected_auth_frame")
    assert seen == []  # the auth frame behind it is never read
    assert len(websocket.scope["_inbox"]) == 1
    assert not hasattr(websocket.state, "ws_unread_frame")


@pytest.mark.asyncio
async def test_a_revoked_frame_token_is_refused_4401(monkeypatch: pytest.MonkeyPatch) -> None:
    _cookie(monkeypatch, None)
    _frame_verifier(monkeypatch)
    websocket, _sent = _frame_socket({"type": "auth", "token": "revoked"})

    with pytest.raises(WebSocketException) as refused:
        await websocket_principal.resolve_ws_principal_first_frame(websocket)

    assert (refused.value.code, refused.value.reason) == (4401, "invalid_token")


@pytest.mark.asyncio
async def test_no_frame_in_time_is_refused_4401(monkeypatch: pytest.MonkeyPatch) -> None:
    _cookie(monkeypatch, None)
    websocket, _sent = _frame_socket()

    with pytest.raises(WebSocketException) as refused:
        await websocket_principal.resolve_ws_principal_first_frame(websocket, timeout=0.05)

    assert (refused.value.code, refused.value.reason) == (4401, "authentication_timeout")


@pytest.mark.asyncio
async def test_a_tokenless_auth_frame_falls_back_to_the_handshake(monkeypatch: pytest.MonkeyPatch) -> None:
    """With nothing else presented it is refused; the route may pass its own fallback (voice: the
    deprecated query shim)."""
    _cookie(monkeypatch, None)
    _frame_verifier(monkeypatch)
    refused_socket, _sent = _frame_socket({"type": "auth"})
    fallback_socket, _sent = _frame_socket({"type": "auth"})
    fallback_calls: list[str | None] = []

    async def fallback(websocket, *, required_scope=None):
        fallback_calls.append(required_scope)
        return _ACCESS

    with pytest.raises(WebSocketException) as refused:
        await websocket_principal.resolve_ws_principal_first_frame(refused_socket)
    principal = await websocket_principal.resolve_ws_principal_first_frame(
        fallback_socket, required_scope="read", tokenless_frame=fallback
    )

    assert (refused.value.code, refused.value.reason) == (4401, "authentication_required")
    assert (principal, fallback_calls) == (_ACCESS, ["read"])


@pytest.mark.asyncio
async def test_the_cookie_answers_a_tokenless_frame_and_a_token_overrides_it(monkeypatch: pytest.MonkeyPatch) -> None:
    _cookie(monkeypatch, _COOKIE)
    seen = _frame_verifier(monkeypatch)
    tokenless, _sent = _frame_socket({"type": "auth"})
    token, _sent = _frame_socket({"type": "auth", "token": "a.b.c"})
    revoked, _sent = _frame_socket({"type": "auth", "token": "revoked"})

    assert await websocket_principal.resolve_ws_principal_first_frame(tokenless) is _COOKIE
    assert await websocket_principal.resolve_ws_principal_first_frame(token) is _ACCESS
    with pytest.raises(WebSocketException) as refused:  # a refused token is not rescued by the cookie
        await websocket_principal.resolve_ws_principal_first_frame(revoked)

    assert refused.value.code == 4401
    assert seen == ["Bearer a.b.c", "Bearer revoked"]


@pytest.mark.asyncio
async def test_data_first_after_a_cookie_is_kept_unread_only_where_the_route_allows_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Audio's cookie clients may send their config first; the route reads it from ``ws_unread_frame``."""
    _cookie(monkeypatch, _COOKIE)
    allowed, _sent = _frame_socket('{"type": "config"}')
    strict, _sent = _frame_socket('{"type": "config"}')
    silent, _sent = _frame_socket()

    principal = await websocket_principal.resolve_ws_principal_first_frame(allowed, cookie_may_send_data_first=True)
    kept_by_cookie = await websocket_principal.resolve_ws_principal_first_frame(
        silent, timeout=0.05, cookie_may_send_data_first=True
    )
    with pytest.raises(WebSocketException) as refused:
        await websocket_principal.resolve_ws_principal_first_frame(strict)

    assert (principal, allowed.state.ws_unread_frame) == (_COOKIE, '{"type": "config"}')
    assert kept_by_cookie is _COOKIE
    assert refused.value.code == 4400


@pytest.mark.asyncio
async def test_a_rejected_cookie_refuses_before_accept(monkeypatch: pytest.MonkeyPatch) -> None:
    async def cross_site(websocket, **_kwargs):
        websocket.state.single_user_cookie_websocket_close_code = 4403
        return None

    monkeypatch.setattr(websocket_principal, "resolve_single_user_cookie_websocket", cross_site)
    websocket, sent = _frame_socket({"type": "auth"})

    with pytest.raises(WebSocketException) as refused:
        await websocket_principal.resolve_ws_principal_first_frame(websocket)

    assert (refused.value.code, refused.value.reason) == (4403, "cookie_auth_rejected")
    assert not _accepted(sent)


@pytest.mark.asyncio
async def test_the_frame_credential_is_never_logged(monkeypatch: pytest.MonkeyPatch) -> None:
    from loguru import logger

    _cookie(monkeypatch, None)
    _frame_verifier(monkeypatch)
    logged: list[str] = []
    sink = logger.add(lambda message: logged.append(message.record["message"]), level="TRACE")
    try:
        for frame in ({"type": "auth", "token": "revoked"}, {"type": "auth", "token": "a.b.c"}):
            websocket, _sent = _frame_socket(frame)
            with contextlib.suppress(WebSocketException):
                await websocket_principal.resolve_ws_principal_first_frame(websocket)
    finally:
        logger.remove(sink)

    assert not [message for message in logged if "revoked" in message or "a.b.c" in message]
