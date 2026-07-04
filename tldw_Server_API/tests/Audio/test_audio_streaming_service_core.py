import asyncio
import json
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.Audio import streaming_service


pytestmark = pytest.mark.unit

TEST_API_KEY = "audio-test-" + "key"


class DummyWebSocket:
    def __init__(self, messages=None):
        self.headers = {}
        self.query_params = {}
        self.client = SimpleNamespace(host="127.0.0.1")
        self.state = SimpleNamespace()
        self.closed = False
        self.close_code = None
        self.sent_json = []
        self._messages = list(messages or [])

    async def receive_text(self):
        if not self._messages:
            raise RuntimeError("no auth frame")
        return self._messages.pop(0)

    async def send_json(self, payload):
        self.sent_json.append(payload)

    async def send_bytes(self, _data: bytes):
        return None

    async def close(self, code=1000, reason=None):  # noqa: ARG002
        self.closed = True
        self.close_code = code


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("endpoint_id", "ws_path"),
    [
        ("audio.stream.transcribe", "/api/v1/audio/stream/transcribe"),
        ("audio.chat.stream", "/api/v1/audio/chat/stream"),
        ("audio.stream.tts", "/api/v1/audio/stream/tts"),
    ],
)
async def test_audio_ws_query_token_auth_rejected_by_default(
    monkeypatch: pytest.MonkeyPatch,
    endpoint_id: str,
    ws_path: str,
):
    from tldw_Server_API.app.core.AuthNZ import ip_allowlist, settings as auth_settings

    ws = DummyWebSocket()
    ws.query_params = {"token": TEST_API_KEY}

    monkeypatch.delenv("AUDIO_WS_ALLOW_QUERY_TOKEN_AUTH", raising=False)
    monkeypatch.setattr(streaming_service, "is_multi_user_mode", lambda: False)
    monkeypatch.setattr(
        auth_settings,
        "get_settings",
        lambda: SimpleNamespace(
            SINGLE_USER_API_KEY=TEST_API_KEY,
            SINGLE_USER_ALLOWED_IPS=[],
            SINGLE_USER_FIXED_ID=1,
        ),
    )
    monkeypatch.setattr(ip_allowlist, "resolve_client_ip", lambda *_args, **_kwargs: "127.0.0.1")

    auth_ok, user_id = await streaming_service._audio_ws_authenticate(
        ws,
        None,
        endpoint_id=endpoint_id,
        ws_path=ws_path,
    )

    assert (auth_ok, user_id) == (False, None)
    assert ws.closed is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("endpoint_id", "ws_path"),
    [
        ("audio.stream.transcribe", "/api/v1/audio/stream/transcribe"),
        ("audio.chat.stream", "/api/v1/audio/chat/stream"),
        ("audio.stream.tts", "/api/v1/audio/stream/tts"),
    ],
)
async def test_audio_ws_initial_auth_frame_supported_for_audio_routes(
    monkeypatch: pytest.MonkeyPatch,
    endpoint_id: str,
    ws_path: str,
):
    from tldw_Server_API.app.core.AuthNZ import ip_allowlist, settings as auth_settings

    ws = DummyWebSocket([json.dumps({"type": "auth", "token": TEST_API_KEY})])

    monkeypatch.delenv("AUDIO_WS_ALLOW_QUERY_TOKEN_AUTH", raising=False)
    monkeypatch.setattr(streaming_service, "is_multi_user_mode", lambda: False)
    monkeypatch.setattr(
        auth_settings,
        "get_settings",
        lambda: SimpleNamespace(
            SINGLE_USER_API_KEY=TEST_API_KEY,
            SINGLE_USER_ALLOWED_IPS=[],
            SINGLE_USER_FIXED_ID=1,
        ),
    )
    monkeypatch.setattr(ip_allowlist, "resolve_client_ip", lambda *_args, **_kwargs: "127.0.0.1")

    auth_ok, user_id = await streaming_service._audio_ws_authenticate(
        ws,
        None,
        endpoint_id=endpoint_id,
        ws_path=ws_path,
    )

    assert (auth_ok, user_id) == (True, 1)
    assert ws.closed is False


@pytest.mark.asyncio
async def test_audio_ws_query_token_auth_can_be_enabled_explicitly(monkeypatch: pytest.MonkeyPatch):
    from tldw_Server_API.app.core.AuthNZ import ip_allowlist, settings as auth_settings

    ws = DummyWebSocket()
    ws.query_params = {"token": TEST_API_KEY}

    monkeypatch.setenv("AUDIO_WS_ALLOW_QUERY_TOKEN_AUTH", "1")
    monkeypatch.setattr(streaming_service, "is_multi_user_mode", lambda: False)
    monkeypatch.setattr(
        auth_settings,
        "get_settings",
        lambda: SimpleNamespace(
            SINGLE_USER_API_KEY=TEST_API_KEY,
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
async def test_stream_tts_to_websocket_propagates_parent_cancellation_during_cleanup():
    class SlowCancelTTSService:
        def __init__(self):
            self.cleanup_started = asyncio.Event()

        async def generate_speech(self, *_args, **_kwargs):  # noqa: ARG002
            try:
                yield b"first-chunk"
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cleanup_started.set()
                await asyncio.sleep(0.2)
                raise

    class FailingWebSocket:
        async def send_bytes(self, _data: bytes):
            raise RuntimeError("client disconnected while reading stream")

        async def close(self, code=1000, reason=None):  # noqa: ARG002
            return None

    class DummyRegistry:
        def increment(self, *_args, **_kwargs):  # noqa: ARG002
            return None

    service = SlowCancelTTSService()
    stream_task = asyncio.create_task(
        streaming_service._stream_tts_to_websocket(
            websocket=FailingWebSocket(),
            speech_req=SimpleNamespace(model="test-model"),
            tts_service=service,
            provider="test-provider",
            outer_stream=None,
            reg=DummyRegistry(),
            route="audio.stream.tts",
            component_label="audio_tts_ws",
        )
    )

    await asyncio.wait_for(service.cleanup_started.wait(), timeout=0.5)
    stream_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(stream_task, timeout=0.5)


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


@pytest.mark.asyncio
async def test_stream_tts_to_websocket_observes_consumer_when_both_tasks_done():
    class OneChunkTTSService:
        async def generate_speech(self, *_args, **_kwargs):  # noqa: ARG002
            yield b"first-chunk"

    class FailingWebSocket:
        async def send_bytes(self, _data: bytes):
            raise ZeroDivisionError("consumer task failed")

    class DummyRegistry:
        def increment(self, *_args, **_kwargs):  # noqa: ARG002
            return None

    async def wait_for_both(tasks, return_when=None):  # noqa: ARG001
        await asyncio.gather(*tasks, return_exceptions=True)
        return set(tasks), set()

    fake_asyncio = SimpleNamespace(
        Queue=asyncio.Queue,
        QueueFull=asyncio.QueueFull,
        create_task=asyncio.create_task,
        wait=wait_for_both,
        FIRST_COMPLETED=asyncio.FIRST_COMPLETED,
    )

    with pytest.raises(ZeroDivisionError, match="consumer task failed"):
        await asyncio.wait_for(
            streaming_service._stream_tts_to_websocket(
                websocket=FailingWebSocket(),
                speech_req=SimpleNamespace(model="test-model"),
                tts_service=OneChunkTTSService(),
                provider="test-provider",
                outer_stream=None,
                reg=DummyRegistry(),
                route="audio.stream.tts",
                component_label="audio_tts_ws",
                asyncio_module=fake_asyncio,
            ),
            timeout=0.5,
        )


@pytest.mark.asyncio
async def test_stream_tts_to_websocket_cancels_consumer_when_completion_signal_fails():
    consumer_cancelled = asyncio.Event()

    class BrokenSentinelQueue:
        def __init__(self, maxsize=0):  # noqa: ARG002
            self.items = []

        def put_nowait(self, item):
            self.items.append(item)

        async def put(self, item):
            if item is None:
                raise RuntimeError("sentinel enqueue failed")
            self.items.append(item)

        async def get(self):
            if self.items:
                return self.items.pop(0)
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                consumer_cancelled.set()
                raise

    class EmptyTTSService:
        async def generate_speech(self, *_args, **_kwargs):  # noqa: ARG002
            if False:
                yield b"unreachable"

    class PassiveWebSocket:
        async def send_bytes(self, _data: bytes):
            return None

    class DummyRegistry:
        def increment(self, *_args, **_kwargs):  # noqa: ARG002
            return None

    fake_asyncio = SimpleNamespace(
        Queue=BrokenSentinelQueue,
        QueueFull=asyncio.QueueFull,
        create_task=asyncio.create_task,
        wait=asyncio.wait,
        FIRST_COMPLETED=asyncio.FIRST_COMPLETED,
    )

    await asyncio.wait_for(
        streaming_service._stream_tts_to_websocket(
            websocket=PassiveWebSocket(),
            speech_req=SimpleNamespace(model="test-model"),
            tts_service=EmptyTTSService(),
            provider="test-provider",
            outer_stream=None,
            reg=DummyRegistry(),
            route="audio.stream.tts",
            component_label="audio_tts_ws",
            asyncio_module=fake_asyncio,
        ),
        timeout=0.5,
    )

    assert consumer_cancelled.is_set()
