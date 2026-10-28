"""The realtime speech routes authenticate on the core resolver from the handshake alone: their
protocol is opened by the server, so the auth step must never read a client event (TASK-13950.7)."""

import asyncio

import pytest

from tldw_Server_API.app.core.Audio import streaming_service
from tldw_Server_API.app.core.AuthNZ import websocket_principal
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.tests.Audio.ws_test_helpers import scripted_websocket

pytestmark = pytest.mark.unit

_COOKIE = AuthPrincipal(kind="user", user_id=1, token_type="single_user_session", subject="single_user")
_SESSION_UPDATE = '{"type":"session.update","session":{"type":"realtime"}}'


def _cookie(monkeypatch: pytest.MonkeyPatch, principal: AuthPrincipal | None) -> None:
    async def resolve(websocket, **_kwargs):
        if principal is None:
            return None
        websocket.state.auth_principal = principal
        return object()

    monkeypatch.setattr(websocket_principal, "resolve_single_user_cookie_websocket", resolve)


def _single_user_key(monkeypatch: pytest.MonkeyPatch) -> None:
    async def verify(request):
        if request.headers.get("x-api-key") != "single-user-secret":
            from fastapi import HTTPException

            raise HTTPException(status_code=401, detail="Could not validate credentials")
        return AuthPrincipal(kind="user", user_id=1, token_type="api_key", subject="single_user")

    monkeypatch.setattr(websocket_principal, "get_auth_principal", verify)


@pytest.mark.asyncio
async def test_cookie_realtime_auth_preserves_first_protocol_event(monkeypatch):
    from tldw_Server_API.app.core.Audio.Realtime.auth import authenticate_realtime_websocket

    _cookie(monkeypatch, _COOKIE)
    ws, sent = scripted_websocket(_SESSION_UPDATE, path="/api/v1/audio/realtime")

    assert await authenticate_realtime_websocket(ws, "native") == (True, 1)
    assert len(ws.scope["_inbox"]) == 2  # neither the connect nor the client's event was read
    assert sent == []  # the handler accepts, after its own quota check


@pytest.mark.asyncio
async def test_realtime_openai_compat_auth_does_not_consume_session_update(monkeypatch: pytest.MonkeyPatch) -> None:
    from tldw_Server_API.app.core.Audio.Realtime.auth import authenticate_realtime_websocket

    _cookie(monkeypatch, None)
    ws, sent = scripted_websocket(_SESSION_UPDATE, path="/v1/realtime")

    auth_ok, user_id = await authenticate_realtime_websocket(ws, route_kind="openai_compat")

    assert (auth_ok, user_id) == (False, None)
    assert len(ws.scope["_inbox"]) == 2
    assert sent == [{"type": "websocket.close", "code": 4401, "reason": ""}]  # refused before accept


@pytest.mark.asyncio
async def test_realtime_takes_a_handshake_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from tldw_Server_API.app.core.Audio.Realtime.auth import authenticate_realtime_websocket

    _cookie(monkeypatch, None)
    _single_user_key(monkeypatch)
    ws, sent = scripted_websocket(_SESSION_UPDATE, headers=[(b"x-api-key", b"single-user-secret")], path="/v1/realtime")

    assert await authenticate_realtime_websocket(ws, route_kind="openai_compat") == (True, 1)
    assert (len(ws.scope["_inbox"]), sent) == (2, [])


@pytest.mark.asyncio
async def test_existing_audio_stream_auth_still_accepts_initial_auth_message(monkeypatch: pytest.MonkeyPatch) -> None:
    _cookie(monkeypatch, None)

    async def verify(request):
        assert request.headers.get("authorization") == "Bearer single-user-secret"
        return AuthPrincipal(kind="user", user_id=1, token_type="api_key", subject="single_user")

    monkeypatch.setattr(websocket_principal, "get_auth_principal", verify)
    ws, sent = scripted_websocket('{"type":"auth","token":"single-user-secret"}')

    auth_ok, user_id = await streaming_service._audio_ws_authenticate(
        ws,
        endpoint_id="audio.stream.transcribe",
        ws_path="/api/v1/audio/stream/transcribe",
    )

    assert (auth_ok, user_id) == (True, 1)
    assert ws.scope["_inbox"] == []
    assert [message["type"] for message in sent] == ["websocket.accept"]


@pytest.mark.asyncio
async def test_realtime_auth_binds_the_user_for_the_handler_task_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """A task that authenticates sees the user for the rest of its life; the parent task does not."""
    from tldw_Server_API.app.core.Audio.Realtime.auth import authenticate_realtime_websocket
    from tldw_Server_API.app.core.Usage.llm_call_context import current_llm_user_id

    _cookie(monkeypatch, _COOKIE)
    ws, _sent = scripted_websocket(_SESSION_UPDATE, path="/api/v1/audio/realtime")

    async def _handler() -> tuple[tuple[bool, int | None], int | None]:
        """Authenticate, then report the user this task sees."""
        result = await authenticate_realtime_websocket(ws, "native")
        return result, current_llm_user_id()

    assert await asyncio.create_task(_handler()) == ((True, 1), 1)
    assert current_llm_user_id() is None


@pytest.mark.asyncio
async def test_failed_realtime_auth_binds_no_user(monkeypatch: pytest.MonkeyPatch) -> None:
    """The openai-compat route with no credentials is refused and no user is bound."""
    from tldw_Server_API.app.core.Audio.Realtime.auth import authenticate_realtime_websocket
    from tldw_Server_API.app.core.Usage.llm_call_context import current_llm_user_id

    _cookie(monkeypatch, None)
    ws, _sent = scripted_websocket(_SESSION_UPDATE, path="/v1/realtime")

    assert await authenticate_realtime_websocket(ws, route_kind="openai_compat") == (False, None)
    assert current_llm_user_id() is None


@pytest.mark.asyncio
async def test_auth_with_no_user_id_succeeds_and_binds_no_user(monkeypatch: pytest.MonkeyPatch) -> None:
    """Authentication that succeeds without a user id leaves no user bound."""
    from tldw_Server_API.app.core.Usage.llm_call_context import current_llm_user_id

    async def _auth_without_user(*_args: object, **_kwargs: object) -> tuple[bool, None]:
        """Succeed, naming no user."""
        return True, None

    monkeypatch.setattr(streaming_service, "_audio_ws_authenticate_impl", _auth_without_user)
    ws, _sent = scripted_websocket()

    result = await streaming_service._audio_ws_authenticate(ws, endpoint_id="audio.test", ws_path="/test")

    assert result == (True, None)
    assert current_llm_user_id() is None
