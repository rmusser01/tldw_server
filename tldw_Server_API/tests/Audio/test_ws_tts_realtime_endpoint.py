import asyncio
import importlib.machinery
import json
import sys
import types
from types import SimpleNamespace
from typing import Any

import pytest

# Keep module imports deterministic in environments where torch-backed deps abort.
if "torch" not in sys.modules:
    _fake_torch = types.ModuleType("torch")
    _fake_torch.__spec__ = importlib.machinery.ModuleSpec("torch", loader=None)
    _fake_torch.Tensor = object
    _fake_torch.nn = SimpleNamespace(Module=object)
    sys.modules["torch"] = _fake_torch

if "faster_whisper" not in sys.modules:
    _fake_fw = types.ModuleType("faster_whisper")
    _fake_fw.__spec__ = importlib.machinery.ModuleSpec("faster_whisper", loader=None)

    class _StubWhisperModel:
        def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
            return None

    _fake_fw.WhisperModel = _StubWhisperModel
    _fake_fw.BatchedInferencePipeline = _StubWhisperModel
    sys.modules["faster_whisper"] = _fake_fw

if "transformers" not in sys.modules:
    _fake_tf = types.ModuleType("transformers")
    _fake_tf.__spec__ = importlib.machinery.ModuleSpec("transformers", loader=None)

    class _StubProcessor:
        @classmethod
        def from_pretrained(cls, *args: Any, **kwargs: Any):  # noqa: ANN206, ARG002
            return cls()

    class _StubModel:
        @classmethod
        def from_pretrained(cls, *args: Any, **kwargs: Any):  # noqa: ANN206, ARG002
            return cls()

    _fake_tf.AutoProcessor = _StubProcessor
    _fake_tf.Qwen2AudioForConditionalGeneration = _StubModel
    sys.modules["transformers"] = _fake_tf

from tldw_Server_API.app.api.v1.endpoints import audio
from tldw_Server_API.app.api.v1.endpoints.audio import (
    audio_streaming as audio_streaming_module,
)
from tldw_Server_API.app.core.TTS.realtime_session import RealtimeSessionHandle, RealtimeTTSSession


class DummyWebSocket:
    def __init__(self, payloads, *, headers=None, query_params=None):
        self.headers = dict(headers or {})
        self.query_params = dict(query_params or {})
        self.client = SimpleNamespace(host="127.0.0.1")
        self.state = SimpleNamespace()
        self._messages = [json.dumps(p) for p in payloads]
        self.sent_bytes = []
        self.sent_json = []
        self.sent_events = []
        self.accepted = False
        self.closed = False
        self.close_code = None

    async def accept(self):
        self.accepted = True

    async def receive_text(self):
        if not self._messages:
            raise RuntimeError("No more messages")  # noqa: TRY003
        return self._messages.pop(0)

    async def send_bytes(self, data: bytes):
        self.sent_bytes.append(data)
        self.sent_events.append(("bytes", data))

    async def send_json(self, payload):
        self.sent_json.append(payload)
        self.sent_events.append(("json", dict(payload)))

    async def close(self, code=1000, reason=None):  # noqa: ARG002
        self.closed = True
        self.close_code = code


class DummyRealtimeSession(RealtimeTTSSession):
    def __init__(self, chunks):
        self._queue: asyncio.Queue = asyncio.Queue()
        self._chunks = chunks
        self._closed = False
        self.finish_count = 0

    async def push_text(self, delta: str) -> None:  # noqa: ARG002
        return None

    async def commit(self) -> None:
        for chunk in self._chunks:
            await self._queue.put(chunk)

    async def finish(self) -> None:
        if self._closed:
            self.finish_count += 1
            return
        self._closed = True
        self.finish_count += 1
        await self._queue.put(None)

    async def audio_stream(self):
        while True:
            item = await self._queue.get()
            if item is None:
                break
            yield item


class DummyRealtimeService:
    async def open_realtime_session(self, *_args, **_kwargs):
        session = DummyRealtimeSession([b"aa", b"bb"])
        return RealtimeSessionHandle(
            session=session,
            provider="vibevoice_realtime",
            warning="fallback to buffered session",
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_websocket_tts_realtime_streams_audio(monkeypatch: pytest.MonkeyPatch):
    payloads = [
        {"type": "config", "model": "vibevoice-realtime-0.5b", "format": "pcm"},
        {"type": "text", "delta": "hello"},
        {"type": "commit"},
        {"type": "final"},
    ]
    ws = DummyWebSocket(payloads)

    async def _auth_stub(*_args, **_kwargs):
        return True, 1

    async def _can_start_stream_stub(_user_id):
        return True, None

    async def _finish_stream_stub(_user_id):
        return None

    async def _get_tts_service_stub():
        return DummyRealtimeService()

    monkeypatch.setattr(audio, "_audio_ws_authenticate", _auth_stub)
    monkeypatch.setattr(audio, "can_start_stream", _can_start_stream_stub)
    monkeypatch.setattr(audio, "finish_stream", _finish_stream_stub)
    monkeypatch.setattr(audio, "get_tts_service", _get_tts_service_stub)

    await audio.websocket_tts_realtime(ws, token=None)

    assert ws.sent_bytes == [b"aa", b"bb"]
    assert any(msg.get("type") == "warning" for msg in ws.sent_json)
    assert any(msg.get("type") == "done" for msg in ws.sent_json)
    assert ws.closed is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_websocket_tts_realtime_interrupt_cancels_without_close(monkeypatch: pytest.MonkeyPatch):
    payloads = [
        {"type": "config", "model": "vibevoice-realtime-0.5b", "format": "pcm"},
        {"type": "text", "delta": "hello"},
        {"type": "commit"},
        {"type": "interrupt", "reason": "barge_in"},
        {"type": "ping"},
        {"type": "final"},
    ]
    ws = DummyWebSocket(payloads)

    async def _auth_stub(*_args, **_kwargs):
        return True, 1

    async def _can_start_stream_stub(_user_id):
        return True, None

    async def _finish_stream_stub(_user_id):
        return None

    async def _get_tts_service_stub():
        return DummyRealtimeService()

    monkeypatch.setattr(audio, "_audio_ws_authenticate", _auth_stub)
    monkeypatch.setattr(audio, "can_start_stream", _can_start_stream_stub)
    monkeypatch.setattr(audio, "finish_stream", _finish_stream_stub)
    monkeypatch.setattr(audio, "get_tts_service", _get_tts_service_stub)

    await audio.websocket_tts_realtime(ws, token=None)

    assert any(msg.get("type") == "interrupted" for msg in ws.sent_json)
    assert any(msg.get("type") == "pong" for msg in ws.sent_json)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_websocket_tts_realtime_accepts_text_after_interrupt(monkeypatch: pytest.MonkeyPatch):
    payloads = [
        {"type": "config", "model": "vibevoice-realtime-0.5b", "format": "pcm"},
        {"type": "text", "delta": "hello"},
        {"type": "commit"},
        {"type": "interrupt", "reason": "barge_in"},
        {"type": "text", "delta": "after interrupt"},
        {"type": "commit"},
        {"type": "final"},
    ]
    ws = DummyWebSocket(payloads)

    async def _auth_stub(*_args, **_kwargs):
        return True, 1

    async def _can_start_stream_stub(_user_id):
        return True, None

    async def _finish_stream_stub(_user_id):
        return None

    async def _get_tts_service_stub():
        return DummyRealtimeService()

    monkeypatch.setattr(audio, "_audio_ws_authenticate", _auth_stub)
    monkeypatch.setattr(audio, "can_start_stream", _can_start_stream_stub)
    monkeypatch.setattr(audio, "finish_stream", _finish_stream_stub)
    monkeypatch.setattr(audio, "get_tts_service", _get_tts_service_stub)

    await audio.websocket_tts_realtime(ws, token=None)

    assert any(msg.get("type") == "interrupted" for msg in ws.sent_json)
    interrupted_idx = next(
        i for i, event in enumerate(ws.sent_events)
        if event[0] == "json" and event[1].get("type") == "interrupted"
    )
    bytes_after_interrupt = [
        event for event in ws.sent_events[interrupted_idx + 1:]
        if event[0] == "bytes"
    ]
    assert bytes_after_interrupt
    assert any(msg.get("type") == "done" for msg in ws.sent_json)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_websocket_tts_realtime_error_frame_shape(monkeypatch: pytest.MonkeyPatch):
    request_id = "req-error-shape"
    payloads = [
        {"type": "config", "model": "vibevoice-realtime-0.5b", "format": "aac"},
    ]
    ws = DummyWebSocket(payloads, headers={"x-request-id": request_id})

    async def _auth_stub(*_args, **_kwargs):
        return True, 1

    async def _can_start_stream_stub(_user_id):
        return True, None

    async def _finish_stream_stub(_user_id):
        return None

    async def _get_tts_service_stub():
        return DummyRealtimeService()

    monkeypatch.setattr(audio, "_audio_ws_authenticate", _auth_stub)
    monkeypatch.setattr(audio, "can_start_stream", _can_start_stream_stub)
    monkeypatch.setattr(audio, "finish_stream", _finish_stream_stub)
    monkeypatch.setattr(audio, "get_tts_service", _get_tts_service_stub)

    await audio.websocket_tts_realtime(ws, token=None)

    err = next((msg for msg in ws.sent_json if msg.get("type") == "error"), None)
    assert err is not None
    assert err.get("request_id") == request_id
    assert err.get("error_type") == "bad_request"
    assert ws.close_code == 4400


@pytest.mark.unit
@pytest.mark.asyncio
async def test_websocket_tts_realtime_provider_format_mismatch(monkeypatch: pytest.MonkeyPatch):
    request_id = "req-provider-mismatch"
    payloads = [
        {"type": "config", "model": "vibevoice-realtime-0.5b", "format": "flac"},
    ]
    ws = DummyWebSocket(payloads, headers={"x-request-id": request_id})
    session = DummyRealtimeSession([b"aa"])

    class MismatchRealtimeService:
        async def open_realtime_session(self, *_args, **_kwargs):
            return RealtimeSessionHandle(session=session, provider="elevenlabs")

    async def _auth_stub(*_args, **_kwargs):
        return True, 1

    async def _can_start_stream_stub(_user_id):
        return True, None

    async def _finish_stream_stub(_user_id):
        return None

    async def _get_tts_service_stub():
        return MismatchRealtimeService()

    monkeypatch.setattr(audio, "_audio_ws_authenticate", _auth_stub)
    monkeypatch.setattr(audio, "can_start_stream", _can_start_stream_stub)
    monkeypatch.setattr(audio, "finish_stream", _finish_stream_stub)
    monkeypatch.setattr(audio, "get_tts_service", _get_tts_service_stub)

    await audio.websocket_tts_realtime(ws, token=None)

    err = next((msg for msg in ws.sent_json if msg.get("type") == "error"), None)
    assert err is not None
    assert err.get("request_id") == request_id
    assert err.get("error_type") == "bad_request"
    assert ws.close_code == 4400
    assert session.finish_count >= 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_websocket_tts_realtime_error_without_compat_alias(monkeypatch: pytest.MonkeyPatch):
    request_id = "req-no-compat-alias"
    payloads = [{"type": "config", "model": "vibevoice-realtime-0.5b", "format": "aac"}]
    ws = DummyWebSocket(payloads, headers={"x-request-id": request_id})

    async def _auth_stub(*_args, **_kwargs):
        return True, 1

    async def _can_start_stream_stub(_user_id):
        return True, None

    async def _finish_stream_stub(_user_id):
        return None

    async def _get_tts_service_stub():
        return DummyRealtimeService()

    monkeypatch.setattr(audio, "_audio_ws_authenticate", _auth_stub)
    monkeypatch.setattr(audio, "can_start_stream", _can_start_stream_stub)
    monkeypatch.setattr(audio, "finish_stream", _finish_stream_stub)
    monkeypatch.setattr(audio, "get_tts_service", _get_tts_service_stub)
    monkeypatch.setenv("AUDIO_WS_COMPAT_ERROR_TYPE", "0")

    await audio.websocket_tts_realtime(ws, token=None)

    err = next((msg for msg in ws.sent_json if msg.get("type") == "error"), None)
    assert err is not None
    assert err.get("request_id") == request_id
    assert err.get("code") == "bad_request"
    assert err.get("error_type") is None
    assert ws.close_code == 4400


@pytest.mark.unit
@pytest.mark.asyncio
async def test_websocket_tts_realtime_keeps_two_user_snapshots_across_interrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Initial and reopened realtime sessions retain only their connection snapshot."""
    from tldw_Server_API.app.core.Audio import tts_service as credential_service
    from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal

    def payloads() -> list[dict[str, Any]]:
        return [
            {
                "type": "config",
                "provider": "openai",
                "model": "tts-1",
                "voice": "alloy",
                "format": "pcm",
            },
            {"type": "text", "delta": "before interrupt"},
            {"type": "commit"},
            {"type": "interrupt", "reason": "test"},
            {"type": "text", "delta": "after interrupt"},
            {"type": "commit"},
            {"type": "final"},
        ]

    first_ws = DummyWebSocket(payloads())
    second_ws = DummyWebSocket(payloads())
    user_ids = {id(first_ws): 101, id(second_ws): 202}
    first_commit_entered = {
        user_id: asyncio.Event() for user_id in user_ids.values()
    }
    first_commit_release = {
        user_id: asyncio.Event() for user_id in user_ids.values()
    }
    open_calls: list[tuple[int, dict[str, Any]]] = []
    runtimes: list[Any] = []

    class Runtime:
        def __init__(self, **kwargs: Any) -> None:
            self.user_id = int(kwargs["user_id"])
            self.handles: list[Any] = []
            self.marked: list[Any] = []
            self.closed = False
            runtimes.append(self)

        async def resolve(self, provider: str, *, model: str | None = None) -> Any:
            handle = SimpleNamespace(
                provider=provider,
                api_key=f"realtime-user-{self.user_id}-key",
                app_config={
                    "openai_api": {
                        "api_base_url": f"https://realtime-user-{self.user_id}.example/v1",
                        "model": model,
                    }
                },
                credentials_resolved=True,
            )
            self.handles.append(handle)
            return handle

        async def mark_used(self, handle: object) -> None:
            self.marked.append(handle)

        async def close(self) -> None:
            self.closed = True

    class Session(RealtimeTTSSession):
        def __init__(self, user_id: int, generation: int) -> None:
            self.user_id = user_id
            self.generation = generation
            self.queue: asyncio.Queue[bytes | None] = asyncio.Queue()
            self.closed = False

        async def push_text(self, _delta: str) -> None:
            return None

        async def commit(self) -> None:
            if self.generation == 1:
                first_commit_entered[self.user_id].set()
                await first_commit_release[self.user_id].wait()
            await self.queue.put(
                f"audio-{self.user_id}-{self.generation}".encode()
            )

        async def finish(self) -> None:
            if self.closed:
                return
            self.closed = True
            await self.queue.put(None)

        async def audio_stream(self):  # noqa: ANN202
            while True:
                item = await self.queue.get()
                if item is None:
                    break
                yield item

    class Service:
        async def open_realtime_session(self, **kwargs: Any) -> RealtimeSessionHandle:
            user_id = int(kwargs["user_id"])
            open_calls.append((user_id, dict(kwargs)))
            generation = sum(1 for called_user, _call in open_calls if called_user == user_id)
            return RealtimeSessionHandle(
                session=Session(user_id, generation),
                provider="openai",
            )

    async def auth_stub(websocket: Any, *_args: Any, **_kwargs: Any) -> tuple[bool, int]:
        user_id = user_ids[id(websocket)]
        websocket.state.auth_principal = AuthPrincipal(
            kind="user",
            user_id=user_id,
            subject=f"user:{user_id}",
        )
        return True, user_id

    async def can_start_stream_stub(_user_id: int) -> tuple[bool, None]:
        return True, None

    async def finish_stream_stub(_user_id: int) -> None:
        return None

    async def get_service() -> Service:
        return Service()

    monkeypatch.setenv("OPENAI_API_KEY", "global-realtime-key-must-not-dispatch")
    monkeypatch.setattr(audio_streaming_module, "is_multi_user_mode", lambda: True)
    monkeypatch.setattr(audio, "_audio_ws_authenticate", auth_stub)
    monkeypatch.setattr(audio, "can_start_stream", can_start_stream_stub)
    monkeypatch.setattr(audio, "finish_stream", finish_stream_stub)
    monkeypatch.setattr(audio, "get_tts_service", get_service)
    monkeypatch.setattr(credential_service, "ProviderCredentialRuntime", Runtime)
    monkeypatch.setattr(
        credential_service,
        "load_server_config_snapshot",
        lambda: {"openai_api": {"api_key": "global-realtime-key-must-not-dispatch"}},
    )
    monkeypatch.setattr(
        credential_service,
        "_capture_tts_provider_config",
        lambda _provider: {"enabled": True},
    )

    first = asyncio.create_task(audio.websocket_tts_realtime(first_ws, token=None))
    second = asyncio.create_task(audio.websocket_tts_realtime(second_ws, token=None))
    try:
        await asyncio.wait_for(
            asyncio.gather(
                *(event.wait() for event in first_commit_entered.values())
            ),
            timeout=1.0,
        )
        first_commit_release[202].set()
        await asyncio.wait_for(second, timeout=1.0)
        first_commit_release[101].set()
        await asyncio.wait_for(first, timeout=1.0)
    finally:
        for event in first_commit_release.values():
            event.set()
        await asyncio.gather(first, second, return_exceptions=True)

    assert len(open_calls) == 4
    for user_id in (101, 202):
        calls = [kwargs for called_user, kwargs in open_calls if called_user == user_id]
        assert len(calls) == 2
        assert calls[0]["provider_overrides"] is calls[1]["provider_overrides"]
        assert calls[0]["provider_overrides"]["openai_api_key"] == (
            f"realtime-user-{user_id}-key"
        )
        assert all(call["user_id"] == user_id for call in calls)

    assert "global-realtime-key-must-not-dispatch" not in repr(open_calls)
    assert "global-realtime-key-must-not-dispatch" not in repr(
        first_ws.sent_json + second_ws.sent_json
    )
    assert len(runtimes) == 2
    assert all(runtime.marked == runtime.handles for runtime in runtimes)
    assert all(runtime.closed for runtime in runtimes)
