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
- ``X-API-KEY``.
- With neither, the single-user session cookie, behind its trusted-Origin check
  (``websocket_session_auth``), since cookies are sent cross-site.

Query-string tokens are not read: they end up in access logs (TASK-12113).
First-message-frame auth arrives with the audio slice of TASK-13950.7, with its consumer.
"""

from __future__ import annotations

from fastapi import HTTPException, WebSocket, WebSocketException, status
from starlette.requests import Request

from tldw_Server_API.app.core.AuthNZ.api_key_manager import has_scope, normalize_scope
from tldw_Server_API.app.core.AuthNZ.auth_principal_resolver import get_auth_principal
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.websocket_session_auth import (
    cookie_websocket_rejection_code,
    resolve_single_user_cookie_websocket,
)

WS_UNAUTHENTICATED = 4401
WS_FORBIDDEN = 4403
_BEARER_SUBPROTOCOL = "bearer"


def _subprotocol_token(websocket: WebSocket) -> str | None:
    """Return ``<token>`` from ``Sec-WebSocket-Protocol: bearer, <token>``."""
    parts = [part.strip() for part in (websocket.headers.get("sec-websocket-protocol") or "").split(",")]
    if len(parts) >= 2 and parts[0].lower() == _BEARER_SUBPROTOCOL and parts[1]:
        return parts[1]
    return None


def auth_subprotocol(websocket: WebSocket) -> str | None:
    """The subprotocol to accept with: the ``bearer`` marker, never the credential after it."""
    return _BEARER_SUBPROTOCOL if _subprotocol_token(websocket) else None


async def resolve_ws_principal(
    websocket: WebSocket,
    *,
    required_scope: str | None = None,
) -> AuthPrincipal:
    """Return the authenticated caller of ``websocket``; call it before ``accept()``.

    Args:
        websocket: The connection, not yet accepted.
        required_scope: API-key scope the route needs ("read", "write", ...). JWT and
            single-user callers are not scoped, as on HTTP.

    Raises:
        WebSocketException: 4401 when credentials are missing or rejected, 4403 when they
            are valid but not allowed, the cookie check's own code, or 1011 when the
            credential store failed. Uncaught, Starlette closes the socket with it.
    """
    headers = list(websocket.scope.get("headers") or [])
    token = _subprotocol_token(websocket)
    if token and "authorization" not in websocket.headers:
        headers.append((b"authorization", f"Bearer {token}".encode("latin-1")))
    elif not token and "authorization" not in websocket.headers and "x-api-key" not in websocket.headers:
        if await resolve_single_user_cookie_websocket(websocket) is not None:
            cookie_principal: AuthPrincipal = websocket.state.auth_principal
            return cookie_principal
        cookie_code = cookie_websocket_rejection_code(websocket)
        if cookie_code is not None:
            raise WebSocketException(code=cookie_code, reason="cookie_auth_rejected")
        raise WebSocketException(code=WS_UNAUTHENTICATED, reason="authentication_required")

    request = Request({**websocket.scope, "type": "http", "method": "GET", "headers": headers, "state": {}})
    try:
        principal = await get_auth_principal(request)
    except HTTPException as exc:
        if exc.status_code >= status.HTTP_500_INTERNAL_SERVER_ERROR:
            raise WebSocketException(code=status.WS_1011_INTERNAL_ERROR, reason="auth_unavailable") from None
        code = WS_FORBIDDEN if exc.status_code == status.HTTP_403_FORBIDDEN else WS_UNAUTHENTICATED
        raise WebSocketException(code=code, reason="invalid_token") from None
    if (
        required_scope
        and principal.api_key_id is not None
        and not has_scope(normalize_scope(getattr(request.state, "_api_key_scope", None)), required_scope)
    ):
        raise WebSocketException(code=WS_FORBIDDEN, reason="insufficient_scope")
    return principal
