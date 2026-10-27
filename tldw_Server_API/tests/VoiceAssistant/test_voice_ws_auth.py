"""The voice assistant verifies its AUTH-frame token on the shared WebSocket path (TASK-13950.7).

``get_auth_principal`` is faked here; tests/AuthNZ/test_websocket_route_auth_matrix.py runs the
route against the real AuthNZ store (revoked, single-user, IP allowlist, deprecated query).
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from starlette.websockets import WebSocket

from tldw_Server_API.app.api.v1.endpoints import voice_assistant
from tldw_Server_API.app.core.AuthNZ import websocket_principal
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal

pytestmark = pytest.mark.unit


def _socket() -> WebSocket:
    scope = {"type": "websocket", "path": "/api/v1/voice/assistant", "headers": [], "query_string": b"", "client": ("127.0.0.1", 50000)}
    return WebSocket(scope, receive=AsyncMock(), send=AsyncMock())


@pytest.mark.parametrize(
    ("principal", "key_scope", "expected"),
    [
        (AuthPrincipal(kind="user", user_id=7, token_type="access"), None, (True, 7)),
        (AuthPrincipal(kind="api_key", user_id=7, api_key_id=3, token_type="api_key"), "read", (True, 7)),
        # A key needs the read scope, as before.
        (AuthPrincipal(kind="api_key", user_id=7, api_key_id=3, token_type="api_key"), "write", (False, None)),
        # A service token has no user to own a voice session.
        (AuthPrincipal(kind="service", user_id=None, token_type="service"), None, (False, None)),
    ],
)
@pytest.mark.asyncio
async def test_the_frame_token_is_verified_as_a_bearer_credential(
    monkeypatch: pytest.MonkeyPatch, principal: AuthPrincipal, key_scope: str | None, expected: tuple[bool, int | None]
) -> None:
    seen: list[str | None] = []

    async def verify(request):
        seen.append(request.headers.get("authorization"))
        request.state._api_key_scope = key_scope
        return principal

    monkeypatch.setattr(websocket_principal, "get_auth_principal", verify)

    assert await voice_assistant._authenticate_websocket(_socket(), "frame-token") == expected
    assert seen == ["Bearer frame-token"]


@pytest.mark.parametrize("status_code", [401, 500])
@pytest.mark.asyncio
async def test_a_refused_or_unverifiable_token_is_not_authenticated(monkeypatch: pytest.MonkeyPatch, status_code: int) -> None:
    async def refuse(request):
        raise HTTPException(status_code=status_code, detail="refused")

    monkeypatch.setattr(websocket_principal, "get_auth_principal", refuse)

    assert await voice_assistant._authenticate_websocket(_socket(), "revoked.jwt.token") == (False, None)
