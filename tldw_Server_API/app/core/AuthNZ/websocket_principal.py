"""Resolve the caller of a WebSocket before it is accepted (TASK-13950.7).

WebSocket routes used to authenticate themselves, each with its own copy of token
sniffing and bearer parsing, and the copies drifted: the workflows route verified AuthNZ
tokens with the MCP JWT manager, so it rejected every one and never checked revocation.
This is the one resolver. Explicit credentials go through the HTTP resolver,
``get_auth_principal``, on a request view of the handshake, so WebSockets get the same
JWT verification, blacklist check, API-key handling and single-user rules as HTTP.

Credentials, in order:

- ``Authorization: Bearer <token>``, or ``Sec-WebSocket-Protocol: bearer, <token>`` for
  browsers, which cannot set headers on a WebSocket. Echo the marker with
  ``auth_subprotocol``: a browser fails a handshake whose offered protocol is ignored.
  Only the Bearer scheme counts: a browser behind a Basic-auth proxy attaches the proxy's
  ``Authorization: Basic`` to the upgrade, which is not a credential here.
- ``X-API-KEY``.
- With neither, the single-user session cookie, behind its trusted-Origin check
  (``websocket_session_auth``), since cookies are sent cross-site.

Query-string tokens are not read: they end up in access logs (TASK-12113). Routes whose
shipped clients still send ``?token=`` / ``?api_key=`` call
``resolve_ws_principal_or_deprecated_query`` instead, which hands that credential to
``verify_ws_credential``, so it is verified exactly as the header it stands for; a route with
another deprecated channel reads it itself and does the same.

Routes whose browser clients authenticate with their first message, ``{"type": "auth", "token":
...}`` (the audio streams and the voice assistant), call ``resolve_ws_principal_first_frame``:
the socket has to be accepted before a frame can be read, so it is a separate function rather
than a flag on ``resolve_ws_principal``, which is called before accept.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable

from fastapi import HTTPException, WebSocket, WebSocketDisconnect, WebSocketException, status
from loguru import logger
from starlette.requests import Request

from tldw_Server_API.app.core.AuthNZ.api_key_manager import has_scope, normalize_scope
from tldw_Server_API.app.core.AuthNZ.auth_principal_resolver import _extract_bearer_token, get_auth_principal
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.websocket_session_auth import (
    cookie_websocket_rejection_code,
    resolve_single_user_cookie_websocket,
)

WS_UNAUTHENTICATED = 4401
WS_FORBIDDEN = 4403
WS_EXPECTED_AUTH_FRAME = 4400
FIRST_FRAME_TIMEOUT_S = 10.0
_BEARER_SUBPROTOCOL = "bearer"
# Logged once per connection that authenticates through a deprecated channel; names the channel
# and path, never the credential.
DEPRECATED_WS_CHANNEL = "Deprecated WebSocket credential channel {} on {}; send Authorization or the bearer subprotocol"


def _subprotocol_token(websocket: WebSocket) -> str | None:
    """Return ``<token>`` from ``Sec-WebSocket-Protocol: bearer, <token>``."""
    parts = [part.strip() for part in (websocket.headers.get("sec-websocket-protocol") or "").split(",")]
    if len(parts) >= 2 and parts[0].lower() == _BEARER_SUBPROTOCOL and parts[1]:
        return parts[1]
    return None


def auth_subprotocol(websocket: WebSocket) -> str | None:
    """The subprotocol to accept with: the client's own ``bearer`` marker, spelled as offered (a
    browser compares the echo exactly), never the credential after it."""
    if not _subprotocol_token(websocket):
        return None
    return websocket.headers["sec-websocket-protocol"].split(",", 1)[0].strip()


def has_ws_credential(websocket: WebSocket) -> bool:
    """Whether the handshake presents a credential ``resolve_ws_principal`` reads.

    That is ``Authorization: Bearer``, ``X-API-KEY`` or the bearer subprotocol. Another
    ``Authorization`` scheme (a proxy's Basic auth) does not count, nor does the ambient cookie.
    """
    return bool(_subprotocol_token(websocket) or _extract_bearer_token(websocket)) or "x-api-key" in websocket.headers


async def resolve_ws_principal(
    websocket: WebSocket,
    *,
    required_scope: str | None = None,
    ignore_signed_url_token: bool = False,
) -> AuthPrincipal:
    """Return the authenticated caller of ``websocket``; call it before ``accept()``.

    Args:
        websocket: The connection, not yet accepted.
        required_scope: API-key scope the route needs ("read", "write", ...). JWT and
            single-user callers are not scoped, as on HTTP. The check is HTTP's
            ``require_api_key_scope`` without its admin-claim bypass: that helper lives in
            the API layer, which core may not import, so a WebSocket stays on the stricter
            side and an admin's key needs the scope too.
        ignore_signed_url_token: The route's ``?token=`` is a signed-URL signature, not a
            credential (sandbox's signed log-stream URL), so it must not stop the cookie.

    Raises:
        WebSocketException: 4401 when credentials are missing or rejected, 4403 when they
            are valid but not allowed, the cookie check's own code, or 1011 when the
            credential store failed. Uncaught, Starlette closes the socket with it.
    """
    token = _subprotocol_token(websocket)
    if token and not _extract_bearer_token(websocket):
        # The browser's Authorization header: it rides with X-API-KEY as the two HTTP headers do.
        api_key = websocket.headers.get("x-api-key")
        return await verify_ws_credential(websocket, token=token, api_key=api_key, required_scope=required_scope)
    if not has_ws_credential(websocket):
        cookie = await resolve_single_user_cookie_websocket(websocket, ignore_signed_url_token=ignore_signed_url_token)
        if cookie is not None:
            cookie_principal: AuthPrincipal = websocket.state.auth_principal
            return cookie_principal
        cookie_code = cookie_websocket_rejection_code(websocket)
        if cookie_code is not None:
            raise WebSocketException(code=cookie_code, reason="cookie_auth_rejected")
        raise WebSocketException(code=WS_UNAUTHENTICATED, reason="authentication_required")
    return await _verify(websocket, list(websocket.scope.get("headers") or []), required_scope)


async def resolve_ws_principal_or_deprecated_query(
    websocket: WebSocket,
    *,
    required_scope: str | None = None,
) -> AuthPrincipal:
    """``resolve_ws_principal``, plus the DEPRECATED ``?token=`` / ``?api_key=`` channels that
    shipped clients still send (ACP, prompt-studio, watchlists and persona; TASK-13950.7).

    The query is read only when no resolver channel competes (``has_ws_credential``), so a header
    or bearer-subprotocol credential always wins. It logs ``DEPRECATED_WS_CHANNEL`` once per connection
    and is verified by ``verify_ws_credential``. Delete it once those clients send the bearer subprotocol.
    """
    if not has_ws_credential(websocket):
        token = websocket.query_params.get("token") or None
        api_key = websocket.query_params.get("api_key") or None
        names = [name for name, value in (("token", token), ("api_key", api_key)) if value]
        if names:
            if not getattr(websocket.state, "ws_deprecation_logged", False):  # persona re-verifies its caller
                websocket.state.ws_deprecation_logged = True
                logger.warning(DEPRECATED_WS_CHANNEL, f"query {' and '.join(names)}", websocket.url.path)
            return await verify_ws_credential(websocket, token=token, api_key=api_key, required_scope=required_scope)
    return await resolve_ws_principal(websocket, required_scope=required_scope)


async def resolve_ws_principal_first_frame(
    websocket: WebSocket,
    *,
    required_scope: str | None = None,
    timeout: float = FIRST_FRAME_TIMEOUT_S,
    cookie_may_send_data_first: bool = False,
    tokenless_frame: Callable[..., Awaitable[AuthPrincipal]] = resolve_ws_principal,
) -> AuthPrincipal:
    """Return the caller of a route whose clients authenticate with their first message,
    ``{"type": "auth", "token": <JWT or API key>}``; call it before ``accept()``.

    The single-user cookie is checked before accept, behind its Origin check. The socket is then
    accepted, because a frame cannot be read otherwise, and exactly one frame is read within
    ``timeout`` seconds. Nothing after it is read, so no data frame is processed before the caller
    is known. The frame's token is verified alone by ``verify_ws_credential``; it is never logged.

    Args:
        websocket: The connection, not yet accepted.
        required_scope: API-key scope the route needs, as for ``resolve_ws_principal``.
        timeout: How long to wait for the frame.
        cookie_may_send_data_first: The route's cookie clients may skip the auth frame and send data
            first (the audio streams). With a cookie, a frame that is not auth is then left unread in
            ``websocket.state.ws_unread_frame`` for the route, and a silent client keeps the cookie.
        tokenless_frame: Resolves an auth frame without a token, after the cookie: by default the
            handshake (``resolve_ws_principal``); voice passes the deprecated query shim.

    Raises:
        WebSocketException: the cookie check's own code before accept; after it, 4400 when the first
            frame is not an auth frame, 4401 when the token is missing, refused or revoked or no frame
            arrives in time, and ``resolve_ws_principal``'s other codes.
        WebSocketDisconnect: the client left before sending a frame.
    """
    cookie: AuthPrincipal | None = None
    if await resolve_single_user_cookie_websocket(websocket) is not None:
        cookie = websocket.state.auth_principal
    cookie_code = cookie_websocket_rejection_code(websocket)
    if cookie_code is not None:
        raise WebSocketException(code=cookie_code, reason="cookie_auth_rejected")
    await websocket.accept(subprotocol=auth_subprotocol(websocket))
    data_first_ok = cookie if cookie_may_send_data_first else None  # the cookie, where data may come first
    try:
        message = await asyncio.wait_for(websocket.receive(), timeout)
    except asyncio.TimeoutError:
        if data_first_ok is not None:
            return data_first_ok
        raise WebSocketException(code=WS_UNAUTHENTICATED, reason="authentication_timeout") from None
    if message["type"] == "websocket.disconnect":
        raise WebSocketDisconnect(message.get("code", 1000))
    raw = message.get("text")
    try:
        frame = json.loads(raw) if raw is not None else None
    except ValueError:
        frame = None
    if not isinstance(frame, dict) or frame.get("type") != "auth":
        if data_first_ok is not None and raw is not None:
            websocket.state.ws_unread_frame = raw
            return data_first_ok
        raise WebSocketException(code=WS_EXPECTED_AUTH_FRAME, reason="expected_auth_frame")
    token = frame.get("token")
    if token:
        return await verify_ws_credential(websocket, token=str(token), required_scope=required_scope)
    if cookie is not None:
        return cookie
    return await tokenless_frame(websocket, required_scope=required_scope)


async def verify_ws_credential(
    websocket: WebSocket,
    *,
    token: str | None = None,
    api_key: str | None = None,
    required_scope: str | None = None,
) -> AuthPrincipal:
    """Verify a credential found elsewhere (the bearer subprotocol, a deprecated query channel, a
    first-message frame) as ``Authorization: Bearer`` or ``X-API-KEY``, on the same path as
    ``resolve_ws_principal``.

    Only the credentials given are verified: the handshake's own ``Authorization`` (a proxy's
    Basic auth, say) and ``X-API-KEY`` are dropped, so a first-frame token stands alone and is
    not rescued by HTTP's fall-back to a handshake key. A route shim calls it when
    ``has_ws_credential`` is False.

    Raises ``WebSocketException`` as ``resolve_ws_principal`` does, and 4401 when given no
    credential: the handshake's own headers are never verified here, or a bare cookie would
    reach the session check without its Origin check.
    """
    if not token and not api_key:
        raise WebSocketException(code=WS_UNAUTHENTICATED, reason="authentication_required")
    handshake = websocket.scope.get("headers") or []
    headers = [(name, value) for name, value in handshake if name not in (b"authorization", b"x-api-key")]
    try:
        if token:
            headers.append((b"authorization", f"Bearer {token}".encode("latin-1")))
        if api_key:
            headers.append((b"x-api-key", api_key.encode("latin-1")))
    except UnicodeEncodeError:  # no header could carry it, so no such credential exists
        raise WebSocketException(code=WS_UNAUTHENTICATED, reason="invalid_token") from None
    return await _verify(websocket, headers, required_scope)


async def _verify(
    websocket: WebSocket, headers: list[tuple[bytes, bytes]], required_scope: str | None
) -> AuthPrincipal:
    request = Request({**websocket.scope, "type": "http", "method": "GET", "headers": headers, "state": {}})
    try:
        principal = await get_auth_principal(request)
    except HTTPException as exc:
        if exc.status_code >= status.HTTP_500_INTERNAL_SERVER_ERROR:
            raise WebSocketException(code=status.WS_1011_INTERNAL_ERROR, reason="auth_unavailable") from None
        code = WS_FORBIDDEN if exc.status_code == status.HTTP_403_FORBIDDEN else WS_UNAUTHENTICATED
        raise WebSocketException(code=code, reason="invalid_token") from None
    key_scopes = normalize_scope(getattr(request.state, "_api_key_scope", None))
    if required_scope and principal.api_key_id is not None and not has_scope(key_scopes, required_scope):
        raise WebSocketException(code=WS_FORBIDDEN, reason="insufficient_scope")
    # What HTTP verified, for a route that checks a key's scopes per action or forwards its caller to
    # an internal HTTP call (persona): a scoped key's scopes, and the credential that won (the JWT, or
    # the key) as the one header that carries it, never a loser beside it. In process; never log it.
    websocket.state.ws_api_key_scopes = sorted(key_scopes) if principal.api_key_id is not None else None
    bearer = _extract_bearer_token(request) or ""
    websocket.state.ws_credential_headers = (
        {"x-api-key": request.headers.get("x-api-key") or bearer}
        if principal.token_type == "api_key"
        else {"authorization": f"Bearer {bearer}"}
    )
    return principal
