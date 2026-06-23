import asyncio
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.Audio import streaming_service


pytestmark = pytest.mark.unit


class DummyWebSocket:
    def __init__(self):
        self.headers = {}
        self.query_params = {}
        self.client = SimpleNamespace(host="127.0.0.1")
        self.state = SimpleNamespace()
        self.closed = False
        self.close_code = None
        self.sent_json = []

    async def receive_text(self):
        raise RuntimeError("no auth frame")

    async def send_json(self, payload):
        self.sent_json.append(payload)

    async def send_bytes(self, _data: bytes):
        return None

    async def close(self, code=1000, reason=None):  # noqa: ARG002
        self.closed = True
        self.close_code = code


@pytest.mark.asyncio
async def test_audio_ws_query_token_auth_rejected_by_default(monkeypatch: pytest.MonkeyPatch):
    from tldw_Server_API.app.core.AuthNZ import ip_allowlist, settings as auth_settings

    ws = DummyWebSocket()
    ws.query_params = {"token": "single-user-secret"}

    monkeypatch.delenv("AUDIO_WS_ALLOW_QUERY_TOKEN_AUTH", raising=False)
    monkeypatch.setattr(streaming_service, "is_multi_user_mode", lambda: False)
    monkeypatch.setattr(
        auth_settings,
        "get_settings",
        lambda: SimpleNamespace(
            SINGLE_USER_API_KEY="single-user-secret",
            SINGLE_USER_ALLOWED_IPS=[],
            SINGLE_USER_FIXED_ID=1,
        ),
    )
    monkeypatch.setattr(ip_allowlist, "resolve_client_ip", lambda *_args, **_kwargs: "127.0.0.1")

    auth_ok, user_id = await streaming_service._audio_ws_authenticate(
        ws,
        None,
        endpoint_id="audio.stream.tts",
        ws_path="/api/v1/audio/stream/tts",
    )

    assert (auth_ok, user_id) == (False, None)
    assert ws.closed is True


@pytest.mark.asyncio
async def test_audio_ws_query_token_auth_can_be_enabled_explicitly(monkeypatch: pytest.MonkeyPatch):
    from tldw_Server_API.app.core.AuthNZ import ip_allowlist, settings as auth_settings

    ws = DummyWebSocket()
    ws.query_params = {"token": "single-user-secret"}

    monkeypatch.setenv("AUDIO_WS_ALLOW_QUERY_TOKEN_AUTH", "1")
    monkeypatch.setattr(streaming_service, "is_multi_user_mode", lambda: False)
    monkeypatch.setattr(
        auth_settings,
        "get_settings",
        lambda: SimpleNamespace(
            SINGLE_USER_API_KEY="single-user-secret",
            SINGLE_USER_ALLOWED_IPS=[],
            SINGLE_USER_FIXED_ID=1,
        ),
    )
    monkeypatch.setattr(ip_allowlist, "resolve_client_ip", lambda *_args, **_kwargs: "127.0.0.1")

    auth_ok, user_id = await streaming_service._audio_ws_authenticate(
        ws,
        None,
        endpoint_id="audio.stream.tts",
        ws_path="/api/v1/audio/stream/tts",
    )

    assert (auth_ok, user_id) == (True, 1)
    assert ws.closed is False


@pytest.mark.asyncio
async def test_stream_tts_to_websocket_cancels_producer_when_consumer_send_fails():
    class HangingTTSService:
        def __init__(self):
            self.cancelled = asyncio.Event()

        async def generate_speech(self, *_args, **_kwargs):  # noqa: ARG002
            try:
                yield b"first-chunk"
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                raise

    class FailingWebSocket:
        async def send_bytes(self, _data: bytes):
            raise RuntimeError("client disconnected while reading stream")

        async def close(self, code=1000, reason=None):  # noqa: ARG002
            return None

    class DummyRegistry:
        def increment(self, *_args, **_kwargs):  # noqa: ARG002
            return None

    service = HangingTTSService()

    await asyncio.wait_for(
        streaming_service._stream_tts_to_websocket(
            websocket=FailingWebSocket(),
            speech_req=SimpleNamespace(model="test-model"),
            tts_service=service,
            provider="test-provider",
            outer_stream=None,
            reg=DummyRegistry(),
            route="audio.stream.tts",
            component_label="audio_tts_ws",
        ),
        timeout=0.25,
    )

    assert service.cancelled.is_set()


@pytest.mark.asyncio
async def test_stream_tts_to_websocket_drains_queue_when_producer_finishes():
    class FastTTSService:
        async def generate_speech(self, *_args, **_kwargs):  # noqa: ARG002
            yield b"first-chunk"
            yield b"second-chunk"

    class SlowWebSocket:
        def __init__(self):
            self.sent = []

        async def send_bytes(self, data: bytes):
            await asyncio.sleep(0.01)
            self.sent.append(data)

    class DummyRegistry:
        def increment(self, *_args, **_kwargs):  # noqa: ARG002
            return None

    websocket = SlowWebSocket()

    await asyncio.wait_for(
        streaming_service._stream_tts_to_websocket(
            websocket=websocket,
            speech_req=SimpleNamespace(model="test-model"),
            tts_service=FastTTSService(),
            provider="test-provider",
            outer_stream=None,
            reg=DummyRegistry(),
            route="audio.stream.tts",
            component_label="audio_tts_ws",
        ),
        timeout=0.5,
    )

    assert websocket.sent == [b"first-chunk", b"second-chunk"]
