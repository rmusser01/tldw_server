"""The voice assistant resolves its caller from the AUTH frame on the core first-frame resolver (TASK-13950.7).

``get_auth_principal`` is faked here; tests/AuthNZ/test_websocket_route_auth_matrix.py runs the
route against the real AuthNZ store (revoked, single-user, IP allowlist, deprecated query).
"""

from __future__ import annotations

import asyncio
import json

import pytest
from fastapi import HTTPException, WebSocketException
from starlette.types import Message
from starlette.websockets import WebSocket

from tldw_Server_API.app.api.v1.endpoints import voice_assistant
from tldw_Server_API.app.core.AuthNZ import websocket_principal
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal

pytestmark = pytest.mark.unit


def _socket(frame: dict, *, query: bytes = b"") -> WebSocket:
    """A voice client that connects and sends ``frame``, then waits."""
    inbox: list[Message] = [{"type": "websocket.connect"}, {"type": "websocket.receive", "text": json.dumps(frame)}]

    async def receive() -> Message:
        if inbox:
            return inbox.pop(0)
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    async def send(_message: Message) -> None:
        return None

    scope = {"type": "websocket", "path": "/api/v1/voice/assistant", "headers": [], "query_string": query, "client": ("127.0.0.1", 50000)}
    return WebSocket(scope, receive=receive, send=send)


def _verifier(monkeypatch: pytest.MonkeyPatch, principal: AuthPrincipal, key_scope: str | None = None) -> list[str | None]:
    seen: list[str | None] = []

    async def verify(request):
        seen.append(request.headers.get("authorization"))
        request.state._api_key_scope = key_scope
        return principal

    monkeypatch.setattr(websocket_principal, "get_auth_principal", verify)
    return seen


@pytest.mark.parametrize(
    ("principal", "key_scope"),
    [
        (AuthPrincipal(kind="user", user_id=7, token_type="access"), None),
        (AuthPrincipal(kind="api_key", user_id=7, api_key_id=3, token_type="api_key"), "read"),
    ],
)
@pytest.mark.asyncio
async def test_the_frame_token_is_verified_as_a_bearer_credential(
    monkeypatch: pytest.MonkeyPatch, principal: AuthPrincipal, key_scope: str | None
) -> None:
    seen = _verifier(monkeypatch, principal, key_scope)

    assert await voice_assistant._authenticate_websocket(_socket({"type": "auth", "token": "frame-token"})) == 7
    assert seen == ["Bearer frame-token"]


@pytest.mark.parametrize(
    ("principal", "key_scope", "refusal"),
    [
        # A key needs the read scope, as before.
        (AuthPrincipal(kind="api_key", user_id=7, api_key_id=3, token_type="api_key"), "write", (4403, "insufficient_scope")),
        # A service token has no user to own a voice session.
        (AuthPrincipal(kind="service", user_id=None, token_type="service"), None, (4401, "user_principal_required")),
    ],
)
@pytest.mark.asyncio
async def test_a_principal_voice_cannot_serve_is_refused(
    monkeypatch: pytest.MonkeyPatch, principal: AuthPrincipal, key_scope: str | None, refusal: tuple[int, str]
) -> None:
    _verifier(monkeypatch, principal, key_scope)

    with pytest.raises(WebSocketException) as refused:
        await voice_assistant._authenticate_websocket(_socket({"type": "auth", "token": "frame-token"}))

    assert (refused.value.code, refused.value.reason) == refusal


@pytest.mark.parametrize(("status_code", "code"), [(401, 4401), (500, 1011)])
@pytest.mark.asyncio
async def test_a_refused_or_unverifiable_token_is_refused(monkeypatch: pytest.MonkeyPatch, status_code: int, code: int) -> None:
    async def refuse(request):
        raise HTTPException(status_code=status_code, detail="refused")

    monkeypatch.setattr(websocket_principal, "get_auth_principal", refuse)

    with pytest.raises(WebSocketException) as refused:
        await voice_assistant._authenticate_websocket(_socket({"type": "auth", "token": "revoked.jwt.token"}))

    assert refused.value.code == code


@pytest.mark.asyncio
async def test_a_tokenless_frame_falls_back_to_the_deprecated_query_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """The voice SDKs also send ``?token=``; it is read, through the shared shim, only when the frame has none."""
    seen = _verifier(monkeypatch, AuthPrincipal(kind="user", user_id=7, token_type="access"))

    assert await voice_assistant._authenticate_websocket(_socket({"type": "auth"}, query=b"token=q.r.s")) == 7
    assert seen == ["Bearer q.r.s"]
