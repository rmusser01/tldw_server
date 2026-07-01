from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.Audio import streaming_service


pytestmark = pytest.mark.unit


class DummyWebSocket:
    def __init__(self, *, first_message: str | None = None):
        self.headers = {}
        self.query_params = {}
        self.client = SimpleNamespace(host="127.0.0.1")
        self.state = SimpleNamespace()
        self.first_message = first_message
        self.receive_text_calls = 0
        self.closed = False
        self.close_code = None
        self.sent_json = []

    async def receive_text(self):
        self.receive_text_calls += 1
        if self.first_message is None:
            raise AssertionError("auth helper unexpectedly consumed the first client event")
        return self.first_message

    async def send_json(self, payload):
        self.sent_json.append(payload)

    async def close(self, code=1000, reason=None):  # noqa: ARG002
        self.closed = True
        self.close_code = code


def _single_user_settings():
    return SimpleNamespace(
        SINGLE_USER_API_KEY="single-user-secret",
        SINGLE_USER_ALLOWED_IPS=[],
        SINGLE_USER_FIXED_ID=1,
    )


@pytest.mark.asyncio
async def test_realtime_openai_compat_auth_does_not_consume_session_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.Audio.Realtime.auth import authenticate_realtime_websocket
    from tldw_Server_API.app.core.AuthNZ import ip_allowlist, settings as auth_settings

    ws = DummyWebSocket(first_message='{"type":"session.update","session":{"type":"realtime"}}')

    monkeypatch.setattr(streaming_service, "is_multi_user_mode", lambda: False)
    monkeypatch.setattr(auth_settings, "get_settings", _single_user_settings)
    monkeypatch.setattr(ip_allowlist, "resolve_client_ip", lambda *_args, **_kwargs: "127.0.0.1")

    auth_ok, user_id = await authenticate_realtime_websocket(ws, route_kind="openai_compat")

    assert (auth_ok, user_id) == (False, None)
    assert ws.receive_text_calls == 0
    assert ws.sent_json == []
    assert ws.closed is True
    assert ws.close_code == 4401


@pytest.mark.asyncio
async def test_existing_audio_stream_auth_still_accepts_initial_auth_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.AuthNZ import ip_allowlist, settings as auth_settings

    ws = DummyWebSocket(first_message='{"type":"auth","token":"single-user-secret"}')

    monkeypatch.setattr(streaming_service, "is_multi_user_mode", lambda: False)
    monkeypatch.setattr(auth_settings, "get_settings", _single_user_settings)
    monkeypatch.setattr(ip_allowlist, "resolve_client_ip", lambda *_args, **_kwargs: "127.0.0.1")

    auth_ok, user_id = await streaming_service._audio_ws_authenticate(
        ws,
        None,
        endpoint_id="audio.stream.transcribe",
        ws_path="/api/v1/audio/stream/transcribe",
    )

    assert (auth_ok, user_id) == (True, 1)
    assert ws.receive_text_calls == 1
    assert ws.closed is False
