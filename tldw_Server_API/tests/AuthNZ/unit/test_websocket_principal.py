"""Branches of core/AuthNZ/websocket_principal.resolve_ws_principal (TASK-13950.7).

The route matrix (tests/AuthNZ/test_websocket_route_auth_matrix.py) covers missing,
revoked and valid credentials end to end. These pin the branches it cannot reach
cheaply: API-key scope, and a failing credential store.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException, WebSocket, WebSocketException

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
