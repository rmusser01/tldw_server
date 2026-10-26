"""Persona's stream resolves its caller with the core WebSocket resolver (TASK-13950.7).

The HTTP verifier (``get_auth_principal``) is faked here; the route matrix,
tests/AuthNZ/test_websocket_route_auth_matrix.py, runs persona against the real AuthNZ store.
"""

import json
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from starlette.websockets import WebSocket, WebSocketDisconnect

from tldw_Server_API.app.api.v1.endpoints import persona as persona_ep
from tldw_Server_API.app.core.AuthNZ import websocket_principal
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal

pytestmark = pytest.mark.unit


fastapi_app = FastAPI()
fastapi_app.include_router(persona_ep.router, prefix="/api/v1/persona")


def _socket(headers: list[tuple[bytes, bytes]]) -> WebSocket:
    scope = {
        "type": "websocket",
        "path": "/api/v1/persona/stream",
        "headers": headers,
        "query_string": b"",
        "client": ("127.0.0.1", 54321),
    }
    return WebSocket(scope, receive=AsyncMock(), send=AsyncMock())


def _http_verifier(monkeypatch: pytest.MonkeyPatch, principals: dict[str, AuthPrincipal]) -> None:
    """Stand in for get_auth_principal: the principal for a known bearer credential, else HTTP's 401."""

    async def verify(request):
        credential = (request.headers.get("authorization") or "").partition(" ")[2]
        if credential in principals:
            return principals[credential]
        raise HTTPException(status_code=401, detail="Could not validate credentials")

    monkeypatch.setattr(websocket_principal, "get_auth_principal", verify)


@pytest.mark.asyncio
async def test_a_jwt_caller_is_resolved_without_per_tool_key_scopes(monkeypatch: pytest.MonkeyPatch) -> None:
    _http_verifier(monkeypatch, {"a.b.c": AuthPrincipal(kind="user", user_id=7, token_type="access")})
    ws = _socket([(b"authorization", b"Bearer a.b.c")])

    assert await persona_ep._resolve_authenticated_user_id(ws) == ("7", True, True)
    assert ws.state.persona_auth_method == "access"
    assert ws.state.persona_api_key_scopes is None  # only a scoped API key is checked per tool


@pytest.mark.asyncio
async def test_a_service_token_is_not_a_persona_user(monkeypatch: pytest.MonkeyPatch) -> None:
    """Persona state is owned by a user; a service principal has none to own it."""
    _http_verifier(monkeypatch, {"s.v.c": AuthPrincipal(kind="service", user_id=None, token_type="service")})

    assert await persona_ep._resolve_authenticated_user_id(_socket([(b"authorization", b"Bearer s.v.c")])) == (
        None,
        True,
        False,
    )


@pytest.mark.asyncio
async def test_missing_and_refused_credentials_are_told_apart(monkeypatch: pytest.MonkeyPatch) -> None:
    _http_verifier(monkeypatch, {})

    assert await persona_ep._resolve_authenticated_user_id(_socket([])) == (None, False, False)
    refused = _socket([(b"authorization", b"Bearer revoked")])
    assert await persona_ep._resolve_authenticated_user_id(refused) == (None, True, False)
    # Until the socket closes: no credential to forward, and an empty scope list denies scoped tools.
    assert (refused.state.persona_auth_method, refused.state.persona_api_key_scopes) == ("", [])


@pytest.mark.parametrize("credentials_supplied", [False, True])
def test_persona_stream_auth_failure_rejects_before_stream_start(monkeypatch, credentials_supplied: bool):
    monkeypatch.setattr(persona_ep, "is_persona_enabled", lambda: True)

    async def _reject_auth(*_args, **_kwargs):
        return None, credentials_supplied, False

    started = False

    async def _record_start(_self):
        nonlocal started
        started = True

    monkeypatch.setattr(persona_ep, "_resolve_authenticated_user_id", _reject_auth)
    monkeypatch.setattr(persona_ep.WebSocketStream, "start", _record_start, raising=True)

    with TestClient(fastapi_app) as client:
        try:
            with client.websocket_connect("/api/v1/persona/stream") as ws:
                with pytest.raises(WebSocketDisconnect):
                    ws.receive_text()
        except WebSocketDisconnect:
            pass

    assert started is False


@pytest.mark.parametrize("marker", [None, "bearer", "Bearer", "unsupported"])
@pytest.mark.parametrize("valid_credential", [True, False])
def test_persona_stream_negotiates_only_safe_auth_marker(
    monkeypatch: pytest.MonkeyPatch, marker: str | None, valid_credential: bool
) -> None:
    """Echo the bearer marker as offered (a browser compares it exactly), never the credential."""
    monkeypatch.setattr(persona_ep, "is_persona_enabled", lambda: True)
    _http_verifier(monkeypatch, {"test_api_key": AuthPrincipal(kind="user", user_id=99, token_type="api_key")})

    credential = "test_api_key" if valid_credential else "invalid_api_key"
    headers = {}
    protocols = []
    selected_protocol = None
    if marker in ("bearer", "Bearer"):
        protocols = [marker, credential]
        selected_protocol = marker
    else:
        headers = {"Authorization": f"Bearer {credential}"}
        if marker:
            protocols = [marker, credential]

    with TestClient(fastapi_app) as client:
        if not valid_credential:
            with pytest.raises(WebSocketDisconnect) as rejected:
                with client.websocket_connect(
                    "/api/v1/persona/stream", headers=headers, subprotocols=protocols
                ):
                    pytest.fail("Invalid credentials must not complete a handshake")
            assert rejected.value.code == 4401
            return
        with client.websocket_connect(
            "/api/v1/persona/stream",
            headers=headers,
            subprotocols=protocols,
        ) as ws:
            assert ws.accepted_subprotocol == selected_protocol
            first_event = json.loads(ws.receive_text())
            assert first_event.get("event") == "notice"
            assert "connected" in str(first_event.get("message", "")).lower()
