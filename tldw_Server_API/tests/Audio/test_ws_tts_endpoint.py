import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.api.v1.endpoints import audio
from tldw_Server_API.app.api.v1.endpoints.audio import (
    audio_streaming as audio_streaming_module,
)
from tldw_Server_API.app.core.Audio import streaming_service


class DummyWebSocket:
    def __init__(self, prompt_payload: dict):
        self.headers = {}
        self.query_params = {}
        self.client = SimpleNamespace(host="127.0.0.1")
        self.state = SimpleNamespace()
        self._messages = [json.dumps(prompt_payload)]
        self.sent_bytes = []
        self.sent_json = []
        self.accepted = False
        self.closed = False
        self.close_code = None

    async def accept(self):
        # Allow idempotent accept
        self.accepted = True

    async def receive_text(self):
        if not self._messages:
            raise RuntimeError("No more messages")  # noqa: TRY003
        return self._messages.pop(0)

    async def send_bytes(self, data: bytes):
        self.sent_bytes.append(data)

    async def send_json(self, payload):
        self.sent_json.append(payload)

    async def close(self, code=1000, reason=None):  # noqa: ARG002
        self.closed = True
        self.close_code = code


class _DummyTTSService:
    def __init__(self, chunks):
        self._chunks = chunks

    async def generate_speech(self, *_args, **_kwargs):  # noqa: ARG002
        for chunk in self._chunks:
            yield chunk


@asynccontextmanager
async def _resolved_tts_credential_scope(**_kwargs):
    """Keep adapter-focused tests independent of the real credential store."""

    class Runtime:
        async def mark_used(self, _credentials):
            return None

    credentials = SimpleNamespace(provider="kitten_tts", credentials_resolved=True)
    yield 1, {"credentials_resolved": True}, Runtime(), credentials


@pytest.mark.unit
@pytest.mark.asyncio
async def test_websocket_tts_streams_audio(monkeypatch: pytest.MonkeyPatch):
    prompt = {"type": "prompt", "text": "hello", "format": "pcm"}
    ws = DummyWebSocket(prompt)

    # Stub auth + quotas
    async def _auth_stub(*_args, **_kwargs):
        return True, 1

    async def _can_start_stream_stub(_user_id):
        return True, None

    async def _finish_stream_stub(_user_id):
        return None

    monkeypatch.setattr(audio, "_audio_ws_authenticate", _auth_stub)
    monkeypatch.setattr(audio, "can_start_stream", _can_start_stream_stub)
    monkeypatch.setattr(audio, "finish_stream", _finish_stream_stub)
    monkeypatch.setattr(
        audio_streaming_module,
        "tts_provider_credential_scope",
        _resolved_tts_credential_scope,
    )

    dummy_service = _DummyTTSService([b"abc", b"def"])

    async def _get_tts_service_stub():
        return dummy_service

    monkeypatch.setattr(audio, "get_tts_service", _get_tts_service_stub)

    # Run handler
    await audio.websocket_tts(ws, token=None)

    assert ws.sent_bytes == [b"abc", b"def"]
    # WebSocketStream.done sends a done frame before closing
    assert any(msg.get("type") == "done" for msg in ws.sent_json)
    assert ws.closed is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_websocket_tts_records_underrun(monkeypatch: pytest.MonkeyPatch):
    prompt = {"type": "prompt", "text": "hello", "format": "pcm"}
    ws = DummyWebSocket(prompt)

    class QueueStub:
        def __init__(self, *_args, **_kwargs):
            self.items = [b"stale"]
            self.first_full = True

        def put_nowait(self, item):

            if self.first_full:
                self.first_full = False
                raise asyncio.QueueFull
            self.items.append(item)

        async def put(self, item):
            self.items.append(item)

        async def get(self):
            while not self.items:
                await asyncio.sleep(0)
            return self.items.pop(0)

        def get_nowait(self):

            if not self.items:
                raise asyncio.QueueEmpty
            return self.items.pop(0)

    class DummyRegistry:
        def __init__(self):
            self.increments = []

        def increment(self, name, value=1, labels=None):

            self.increments.append((name, value, labels or {}))

        def observe(self, *_args, **_kwargs):  # noqa: ARG002
            return None

    reg = DummyRegistry()

    async def _auth_stub(*_args, **_kwargs):
        return True, 1

    async def _can_start_stream_stub(_user_id):
        return True, None

    async def _finish_stream_stub(_user_id):
        return None

    dummy_service = _DummyTTSService([b"a", b"b"])

    async def _get_tts_service_stub():
        return dummy_service

    monkeypatch.setattr(audio, "_audio_ws_authenticate", _auth_stub)
    monkeypatch.setattr(audio, "can_start_stream", _can_start_stream_stub)
    monkeypatch.setattr(audio, "finish_stream", _finish_stream_stub)
    monkeypatch.setattr(
        audio_streaming_module,
        "tts_provider_credential_scope",
        _resolved_tts_credential_scope,
    )
    monkeypatch.setattr(audio, "get_tts_service", _get_tts_service_stub)
    monkeypatch.setattr(audio, "get_metrics_registry", lambda: reg)
    monkeypatch.setattr(
        audio,
        "asyncio",
        SimpleNamespace(
            Queue=QueueStub,
            QueueFull=asyncio.QueueFull,
            QueueEmpty=asyncio.QueueEmpty,
            create_task=asyncio.create_task,
            wait=asyncio.wait,
            wait_for=asyncio.wait_for,
            FIRST_EXCEPTION=asyncio.FIRST_EXCEPTION,
            sleep=asyncio.sleep,
        ),
    )

    await audio.websocket_tts(ws, token=None)

    # First put_nowait raises, triggering underrun counter
    assert any(name == "audio_stream_underruns_total" for name, _, _ in reg.increments)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_websocket_tts_disconnect_releases_stream_slot(monkeypatch: pytest.MonkeyPatch):
    prompt = {"type": "prompt", "text": "hello disconnect", "format": "pcm"}
    ws = DummyWebSocket(prompt)

    async def _send_bytes_disconnect(_data: bytes):
        raise RuntimeError("client disconnected while reading stream")  # noqa: TRY003

    ws.send_bytes = _send_bytes_disconnect

    async def _auth_stub(*_args, **_kwargs):
        return True, 1

    async def _can_start_stream_stub(_user_id):
        return True, None

    finish_calls = []

    async def _finish_stream_stub(user_id):
        finish_calls.append(user_id)
        return None

    dummy_service = _DummyTTSService([b"a", b"b"])

    async def _get_tts_service_stub():
        return dummy_service

    monkeypatch.setattr(audio, "_audio_ws_authenticate", _auth_stub)
    monkeypatch.setattr(audio, "can_start_stream", _can_start_stream_stub)
    monkeypatch.setattr(audio, "finish_stream", _finish_stream_stub)
    monkeypatch.setattr(
        audio_streaming_module,
        "tts_provider_credential_scope",
        _resolved_tts_credential_scope,
    )
    monkeypatch.setattr(audio, "get_tts_service", _get_tts_service_stub)

    await audio.websocket_tts(ws, token=None)

    assert finish_calls == [1]
    assert ws.closed is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_websocket_tts_keeps_two_user_credential_snapshots_isolated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The WS endpoint resolves one user-scoped TTS snapshot per connection."""
    from tldw_Server_API.app.core.Audio import tts_service as tts_credential_service

    first_ws = DummyWebSocket(
        {"type": "prompt", "text": "first", "format": "pcm", "provider": "openai", "model": "tts-1"}
    )
    second_ws = DummyWebSocket(
        {"type": "prompt", "text": "second", "format": "pcm", "provider": "openai", "model": "tts-1"}
    )
    user_ids = {id(first_ws): 101, id(second_ws): 202}
    entered = {user_id: asyncio.Event() for user_id in user_ids.values()}
    release = {user_id: asyncio.Event() for user_id in user_ids.values()}
    calls: list[tuple[int, dict[str, object]]] = []
    runtimes: list[object] = []

    class Runtime:
        def __init__(self, **kwargs):
            self.user_id = int(kwargs["user_id"])
            self.handles = []
            self.marked = []
            self.closed = False
            runtimes.append(self)

        async def resolve(self, provider, *, model=None):
            handle = SimpleNamespace(
                provider=provider,
                api_key=f"ws-user-{self.user_id}-key",
                app_config={
                    "openai_api": {
                        "api_base_url": f"https://ws-user-{self.user_id}.example/v1",
                        "model": model,
                    }
                },
                auth_source="api_key",
                credentials_resolved=True,
            )
            self.handles.append(handle)
            return handle

        async def mark_used(self, handle):
            self.marked.append(handle)

        async def close(self):
            self.closed = True

    class Service:
        async def generate_speech(self, _request, **kwargs):
            user_id = int(kwargs["user_id"])
            calls.append((user_id, dict(kwargs)))
            entered[user_id].set()
            await release[user_id].wait()
            yield f"audio-{user_id}".encode()

    async def auth_stub(websocket, *_args, **_kwargs):
        return True, user_ids[id(websocket)]

    async def can_start_stream_stub(_user_id):
        return True, None

    async def finish_stream_stub(_user_id):
        return None

    async def get_service():
        return Service()

    monkeypatch.setenv("OPENAI_API_KEY", "global-ws-key-must-not-dispatch")
    monkeypatch.setattr(audio_streaming_module, "is_multi_user_mode", lambda: True)
    monkeypatch.setattr(audio, "_audio_ws_authenticate", auth_stub)
    monkeypatch.setattr(audio, "can_start_stream", can_start_stream_stub)
    monkeypatch.setattr(audio, "finish_stream", finish_stream_stub)
    monkeypatch.setattr(audio, "get_tts_service", get_service)
    monkeypatch.setattr(tts_credential_service, "ProviderCredentialRuntime", Runtime, raising=False)
    monkeypatch.setattr(
        tts_credential_service,
        "load_server_config_snapshot",
        lambda: {"openai_api": {"api_key": "global-ws-key-must-not-dispatch"}},
    )
    monkeypatch.setattr(
        tts_credential_service,
        "_capture_tts_provider_config",
        lambda _provider: {"enabled": True},
    )

    first = asyncio.create_task(audio.websocket_tts(first_ws, token=None))
    second = asyncio.create_task(audio.websocket_tts(second_ws, token=None))
    try:
        await asyncio.wait_for(
            asyncio.gather(*(event.wait() for event in entered.values())),
            timeout=1.0,
        )
        release[202].set()
        await asyncio.wait_for(second, timeout=1.0)
        release[101].set()
        await asyncio.wait_for(first, timeout=1.0)
    finally:
        for event in release.values():
            event.set()
        await asyncio.gather(first, second, return_exceptions=True)

    assert sorted(
        (user_id, kwargs["provider_overrides"]["openai_api_key"])
        for user_id, kwargs in calls
    ) == [(101, "ws-user-101-key"), (202, "ws-user-202-key")]
    assert all(kwargs["fallback"] is False for _user_id, kwargs in calls)
    assert "global-ws-key-must-not-dispatch" not in repr(calls)
    assert "global-ws-key-must-not-dispatch" not in repr(first_ws.sent_json + second_ws.sent_json)
    assert len(runtimes) == 2
    assert all(runtime.marked == runtime.handles for runtime in runtimes)
    assert all(runtime.closed for runtime in runtimes)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_websocket_tts_disconnect_closes_iterator_before_credential_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A WS disconnect keeps the credential runtime alive through iterator close."""
    from tldw_Server_API.app.core.Audio import tts_service as tts_credential_service

    ws = DummyWebSocket(
        {"type": "prompt", "text": "disconnect", "format": "pcm", "provider": "openai", "model": "tts-1"}
    )
    send_attempted = asyncio.Event()
    second_next_started = asyncio.Event()
    close_started = asyncio.Event()
    close_release = asyncio.Event()
    runtime_closed = asyncio.Event()
    lifecycle: list[str] = []

    async def send_bytes_disconnect(_data: bytes) -> None:
        send_attempted.set()
        raise RuntimeError("client disconnected")

    ws.send_bytes = send_bytes_disconnect

    class Runtime:
        async def resolve(self, provider, *, model=None):
            return SimpleNamespace(
                provider=provider,
                api_key="disconnect-user-key",
                app_config={"openai_api": {"model": model}},
                auth_source="api_key",
                credentials_resolved=True,
            )

        async def mark_used(self, _handle):
            lifecycle.append("mark_used")

        async def close(self):
            lifecycle.append("runtime_close")
            runtime_closed.set()

    class SpeechIterator:
        def __init__(self):
            self.first = True

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.first:
                self.first = False
                return b"first"
            second_next_started.set()
            await asyncio.Event().wait()
            raise StopAsyncIteration

        async def aclose(self):
            close_started.set()
            await close_release.wait()
            lifecycle.append("iterator_close")

    class Service:
        def generate_speech(self, *_args, **_kwargs):
            return SpeechIterator()

    async def auth_stub(*_args, **_kwargs):
        return True, 101

    async def can_start_stream_stub(_user_id):
        return True, None

    async def finish_stream_stub(_user_id):
        return None

    async def get_service():
        return Service()

    monkeypatch.setattr(audio, "_audio_ws_authenticate", auth_stub)
    monkeypatch.setattr(audio_streaming_module, "is_multi_user_mode", lambda: True)
    monkeypatch.setattr(audio, "can_start_stream", can_start_stream_stub)
    monkeypatch.setattr(audio, "finish_stream", finish_stream_stub)
    monkeypatch.setattr(audio, "get_tts_service", get_service)
    monkeypatch.setattr(
        tts_credential_service,
        "ProviderCredentialRuntime",
        lambda **_kwargs: Runtime(),
        raising=False,
    )
    monkeypatch.setattr(
        tts_credential_service,
        "load_server_config_snapshot",
        lambda: {"openai_api": {"api_key": "server-key"}},
    )
    monkeypatch.setattr(
        tts_credential_service,
        "_capture_tts_provider_config",
        lambda _provider: {"enabled": True},
    )

    task = asyncio.create_task(audio.websocket_tts(ws, token=None))
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
        assert runtime_closed.is_set() is False
    finally:
        close_release.set()
        await asyncio.gather(task, return_exceptions=True)

    assert lifecycle == ["mark_used", "iterator_close", "runtime_close"]


@pytest.mark.unit
def test_ws_tts_queue_maxsize_is_clamped(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("AUDIO_TTS_WS_QUEUE_MAXSIZE", raising=False)
    monkeypatch.delenv("AUDIO_WS_TTS_QUEUE_MAXSIZE", raising=False)
    assert streaming_service._get_tts_ws_queue_maxsize() == 8

    monkeypatch.setenv("AUDIO_TTS_WS_QUEUE_MAXSIZE", "1")
    assert streaming_service._get_tts_ws_queue_maxsize() == 2

    monkeypatch.setenv("AUDIO_TTS_WS_QUEUE_MAXSIZE", "999")
    assert streaming_service._get_tts_ws_queue_maxsize() == 256

    monkeypatch.setenv("AUDIO_TTS_WS_QUEUE_MAXSIZE", "bad-int")
    assert streaming_service._get_tts_ws_queue_maxsize() == 8
