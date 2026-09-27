import asyncio
import json
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.Audio import streaming_service

pytestmark = pytest.mark.unit


class DummyWebSocket:
    def __init__(self, messages=None):
        self.headers = {}
        self.query_params = {}
        self.client = SimpleNamespace(host="127.0.0.1")
        self.state = SimpleNamespace()
        self.closed = False
        self.close_code = None
        self.sent_json = []
        self.messages = list(messages or [])
        self.receive_count = 0

    async def receive_text(self):
        self.receive_count += 1
        if self.messages:
            return self.messages.pop(0)
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
    from tldw_Server_API.app.core.AuthNZ import ip_allowlist
    from tldw_Server_API.app.core.AuthNZ import settings as auth_settings

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
    from tldw_Server_API.app.core.AuthNZ import ip_allowlist
    from tldw_Server_API.app.core.AuthNZ import settings as auth_settings

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


def _single_user_audio_settings():
    return SimpleNamespace(
        SINGLE_USER_API_KEY="single-user-secret",
        SINGLE_USER_ALLOWED_IPS=[],
        SINGLE_USER_FIXED_ID=1,
    )


def _configure_cookie_audio_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    from tldw_Server_API.app.core.AuthNZ import ip_allowlist
    from tldw_Server_API.app.core.AuthNZ import settings as auth_settings

    monkeypatch.setattr(streaming_service, "is_multi_user_mode", lambda: False)
    monkeypatch.setattr(auth_settings, "get_settings", _single_user_audio_settings)
    monkeypatch.setattr(ip_allowlist, "resolve_client_ip", lambda *_args, **_kwargs: "127.0.0.1")


@pytest.mark.asyncio
async def test_cookie_audio_auth_buffers_first_non_auth_application_frame(
    monkeypatch: pytest.MonkeyPatch,
):
    _configure_cookie_audio_auth(monkeypatch)
    config_frame = json.dumps({"type": "config", "sample_rate": 16000})
    ws = DummyWebSocket([config_frame])
    ws.state.single_user_session_id = 9
    ws.state.user_id = 1

    auth_result = await streaming_service._audio_ws_authenticate(
        ws,
        None,
        endpoint_id="audio.stream.transcribe",
        ws_path="/api/v1/audio/stream/transcribe",
    )

    assert auth_result == (True, 1)
    assert await streaming_service.receive_audio_websocket_text(ws) == config_frame
    assert ws.receive_count == 1


@pytest.mark.asyncio
async def test_valid_initial_audio_auth_takes_precedence_over_cookie(
    monkeypatch: pytest.MonkeyPatch,
):
    _configure_cookie_audio_auth(monkeypatch)
    ws = DummyWebSocket([json.dumps({"type": "auth", "token": "single-user-secret"})])
    ws.state.single_user_session_id = 9
    ws.state.user_id = 1

    auth_result = await streaming_service._audio_ws_authenticate(
        ws,
        None,
        endpoint_id="audio.stream.tts",
        ws_path="/api/v1/audio/stream/tts",
    )

    assert auth_result == (True, 1)
    assert ws.state.auth_principal.token_type == streaming_service.AUTH_TOKEN_TYPE_ACCESS
    assert ws.receive_count == 1


@pytest.mark.asyncio
async def test_invalid_initial_audio_auth_suppresses_cookie_fallback(
    monkeypatch: pytest.MonkeyPatch,
):
    _configure_cookie_audio_auth(monkeypatch)
    ws = DummyWebSocket([json.dumps({"type": "auth", "token": "wrong"})])
    ws.state.single_user_session_id = 9
    ws.state.user_id = 1

    auth_result = await streaming_service._audio_ws_authenticate(
        ws,
        None,
        endpoint_id="audio.stream.tts.realtime",
        ws_path="/api/v1/audio/stream/tts/realtime",
    )

    assert auth_result == (False, None)
    assert ws.closed is True
    assert ws.close_code == 4401
    assert ws.receive_count == 1


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


@pytest.mark.asyncio
async def test_stream_tts_to_websocket_keeps_concurrent_credential_snapshots_isolated():
    """Concurrent TTS producers dispatch and account against only their own snapshot."""

    entered = {user_id: asyncio.Event() for user_id in (101, 202)}
    release = {user_id: asyncio.Event() for user_id in (101, 202)}
    calls: list[tuple[int, dict[str, object]]] = []
    marked: list[int] = []

    class RecordingTTSService:
        async def generate_speech(self, _request, **kwargs):  # noqa: ANN001
            user_id = int(kwargs["user_id"])
            calls.append((user_id, dict(kwargs)))
            entered[user_id].set()
            await release[user_id].wait()
            yield f"audio-{user_id}".encode()

    class RecordingWebSocket:
        def __init__(self) -> None:
            self.sent: list[bytes] = []

        async def send_bytes(self, data: bytes) -> None:
            self.sent.append(data)

    class DummyRegistry:
        def increment(self, *_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
            return None

    async def run_one(user_id: int) -> RecordingWebSocket:
        websocket = RecordingWebSocket()

        async def mark_first_output() -> None:
            marked.append(user_id)

        await streaming_service._stream_tts_to_websocket(
            websocket=websocket,
            speech_req=SimpleNamespace(model="tts-1"),
            tts_service=RecordingTTSService(),
            provider="openai",
            provider_overrides={
                "credentials_resolved": True,
                "openai_api_key": f"user-{user_id}-key",
            },
            user_id=user_id,
            on_first_output=mark_first_output,
            outer_stream=None,
            reg=DummyRegistry(),
            route="audio.stream.tts",
            component_label="audio_tts_ws",
        )
        return websocket

    first = asyncio.create_task(run_one(101))
    second = asyncio.create_task(run_one(202))
    try:
        await asyncio.wait_for(
            asyncio.gather(*(event.wait() for event in entered.values())),
            timeout=1.0,
        )
        release[202].set()
        second_ws = await asyncio.wait_for(second, timeout=1.0)
        assert second_ws.sent == [b"audio-202"]
        assert marked == [202]

        release[101].set()
        first_ws = await asyncio.wait_for(first, timeout=1.0)
        assert first_ws.sent == [b"audio-101"]
    finally:
        for event in release.values():
            event.set()
        await asyncio.gather(first, second, return_exceptions=True)

    assert sorted(
        (user_id, kwargs["provider_overrides"]["openai_api_key"])
        for user_id, kwargs in calls
    ) == [(101, "user-101-key"), (202, "user-202-key")]
    assert all(kwargs["fallback"] is False for _user_id, kwargs in calls)
    assert sorted(marked) == [101, 202]


@pytest.mark.asyncio
async def test_stream_tts_to_websocket_closes_inner_iterator_before_scope_on_disconnect():
    """A disconnected consumer cannot release credential scope before stream close."""

    second_next_started = asyncio.Event()
    close_started = asyncio.Event()
    close_release = asyncio.Event()
    send_attempted = asyncio.Event()
    lifecycle: list[str] = []

    class SpeechIterator:
        def __init__(self) -> None:
            self._first = True

        def __aiter__(self):  # noqa: ANN204
            return self

        async def __anext__(self) -> bytes:
            if self._first:
                self._first = False
                return b"first"
            second_next_started.set()
            await asyncio.Event().wait()
            raise StopAsyncIteration

        async def aclose(self) -> None:
            close_started.set()
            await close_release.wait()
            lifecycle.append("iterator_close")

    iterator = SpeechIterator()

    class TTSService:
        def generate_speech(self, *_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
            return iterator

    class DisconnectingWebSocket:
        async def send_bytes(self, _data: bytes) -> None:
            send_attempted.set()
            raise RuntimeError("client disconnected")

        async def close(self, **_kwargs) -> None:  # noqa: ANN003
            return None

    class DummyRegistry:
        def increment(self, *_args, **_kwargs):  # noqa: ANN002, ANN003, ANN202
            return None

    async def run_stream() -> None:
        try:
            await streaming_service._stream_tts_to_websocket(
                websocket=DisconnectingWebSocket(),
                speech_req=SimpleNamespace(model="tts-1"),
                tts_service=TTSService(),
                provider="openai",
                provider_overrides={
                    "credentials_resolved": True,
                    "openai_api_key": "disconnect-key",
                },
                user_id=101,
                on_first_output=lambda: None,
                outer_stream=None,
                reg=DummyRegistry(),
                route="audio.stream.tts",
                component_label="audio_tts_ws",
            )
        finally:
            lifecycle.append("scope_close")

    task = asyncio.create_task(run_stream())
    try:
        await asyncio.wait_for(send_attempted.wait(), timeout=1.0)
        await asyncio.wait_for(second_next_started.wait(), timeout=1.0)
        close_waiter = asyncio.create_task(close_started.wait())
        done, _pending = await asyncio.wait(
            {task, close_waiter},
            timeout=1.0,
            return_when=asyncio.FIRST_COMPLETED,
        )
        assert close_waiter in done
        assert task not in done
        assert lifecycle == []
    finally:
        close_release.set()
        await asyncio.gather(task, return_exceptions=True)

    assert lifecycle == ["iterator_close", "scope_close"]
