"""Tests for the audio chat WebSocket streaming endpoint."""

import asyncio
import base64
import concurrent.futures
import importlib.machinery
import json
import struct
import sys
import threading
import time
import types
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any, AsyncIterator, Dict, Iterable, List, Optional

import pytest
from starlette.websockets import WebSocketDisconnect

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
from tldw_Server_API.app.api.v1.endpoints.audio import audio_streaming as audio_streaming_module
from tldw_Server_API.app.core.Audio.transcription_service import (
    _map_openai_audio_model_to_whisper,
)
from tldw_Server_API.app.core.AuthNZ import provider_credential_runtime
from tldw_Server_API.app.core.AuthNZ.byok_runtime import (
    ByokResolutionStatus,
    ResolvedByokCredentials,
)
from tldw_Server_API.app.core.AuthNZ.llm_provider_overrides import (
    set_llm_provider_overrides_cache_for_tests,
)
from tldw_Server_API.app.core.AuthNZ.provider_credential_runtime import (
    PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY,
    ProviderCredentialRuntime as RealProviderCredentialRuntime,
    is_runtime_issued_provider_call_credentials,
)
from tldw_Server_API.app.core.Character_Chat.character_conversation_factory import (
    build_materialized_behavior_controls,
)
from tldw_Server_API.app.core.Character_Chat.chat_settings_validation import (
    build_pending_greeting_record,
)
from tldw_Server_API.app.core.DB_Management.chacha.conversation_resume_store import (
    build_materialized_behavior_settings,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    ConflictError,
    InputError,
)

_AUDIO_LABEL_TO_SAMPLE = {
    "abc": 100,
    "abcd": 200,
    "live": 300,
    "one": 400,
    "two": 500,
    "queued": 600,
}
_AUDIO_SAMPLE_TO_LABEL = {value: key for key, value in _AUDIO_LABEL_TO_SAMPLE.items()}


@pytest.mark.asyncio
@pytest.mark.parametrize("abandonment", ["timeout", "caller_cancel"])
async def test_audio_sync_call_never_starts_late_when_default_executor_is_saturated(
    monkeypatch: pytest.MonkeyPatch,
    abandonment: str,
) -> None:
    """Audio sync work starts before its caller can time out or cancel."""
    from tldw_Server_API.app.core.Chat.bounded_daemon import BoundedDaemonPool

    loop = asyncio.get_running_loop()
    executor_release = threading.Event()
    call_release = threading.Event()
    executor_started = asyncio.Event()
    call_started = asyncio.Event()
    cleanup_finished = asyncio.Event()
    invocation_count = 0

    def occupy_default_executor() -> None:
        loop.call_soon_threadsafe(executor_started.set)
        executor_release.wait(timeout=2.0)

    def provider_call() -> str:
        nonlocal invocation_count
        invocation_count += 1
        loop.call_soon_threadsafe(call_started.set)
        call_release.wait(timeout=2.0)
        return "late"

    async def cleanup() -> None:
        cleanup_finished.set()

    previous_executor = getattr(loop, "_default_executor", None)
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    loop.set_default_executor(executor)
    executor_worker = loop.run_in_executor(None, occupy_default_executor)
    await executor_started.wait()
    monkeypatch.setattr(
        audio_streaming_module,
        "STREAM_DAEMON_POOL",
        BoundedDaemonPool(1),
    )

    operation = asyncio.create_task(
        audio_streaming_module._run_bounded_audio_sync_call(
            provider_call,
            name="audio-default-executor-regression",
            timeout_seconds=0.5 if abandonment == "timeout" else 30.0,
            on_abandoned=cleanup,
        )
    )
    try:
        await asyncio.wait_for(call_started.wait(), timeout=2.0)

        if abandonment == "caller_cancel":
            operation.cancel()
            with pytest.raises(asyncio.CancelledError):
                await operation
        else:
            with pytest.raises(TimeoutError, match="audio-default-executor-regression timed out"):
                await operation

        assert invocation_count == 1
    finally:
        executor_release.set()
        call_release.set()
        await executor_worker
        if not operation.done():
            operation.cancel()
            await asyncio.gather(operation, return_exceptions=True)
        if invocation_count:
            await asyncio.wait_for(cleanup_finished.wait(), timeout=2.0)
        loop.set_default_executor(
            previous_executor or concurrent.futures.ThreadPoolExecutor()
        )
        executor.shutdown(wait=True)


@pytest.mark.asyncio
async def test_audio_stream_cleanup_continues_after_child_self_cancellation() -> None:
    """A self-cancelled iterator close cannot skip source and runtime cleanup."""
    lifecycle: list[str] = []

    class Iterator:
        async def aclose(self) -> None:
            lifecycle.append("iterator_cancel")
            raise asyncio.CancelledError

    class Source:
        async def aclose(self) -> None:
            lifecycle.append("source_close")

    async def cleanup_scope() -> None:
        try:
            await audio_streaming_module._close_audio_provider_stream(
                Source(),
                {"iterator": Iterator()},
                owned_cleanup=True,
            )
        finally:
            lifecycle.append("runtime_close")

    await cleanup_scope()

    assert lifecycle == ["iterator_cancel", "source_close", "runtime_close"]


@pytest.mark.asyncio
@pytest.mark.parametrize("stream_mode", ["async", "sync"])
@pytest.mark.parametrize("abandonment", ["timeout", "caller_cancel"])
@pytest.mark.parametrize(
    ("chunk", "expected_marks"),
    [
        ('data: {"choices":[{"delta":{"content":"late"}}]}\n\n', 1),
        ('data: {"choices":[{"delta":{"content":""}}]}\n\n', 0),
        ('data: {"error":{"message":"private"}}\n\n', 0),
    ],
    ids=["content", "empty", "error"],
)
async def test_audio_late_chunk_usage_matches_normal_success_semantics(
    monkeypatch: pytest.MonkeyPatch,
    stream_mode: str,
    abandonment: str,
    chunk: str,
    expected_marks: int,
) -> None:
    """Late next results mark only non-empty successful provider content."""
    loop = asyncio.get_running_loop()
    async_release = asyncio.Event()
    sync_release = threading.Event()
    next_started = asyncio.Event()
    cleanup_finished = asyncio.Event()
    cleanup_claimed = threading.Event()
    usage_claimed = threading.Event()
    marks = 0

    class Runtime:
        async def mark_used(self, _handle: object) -> None:
            nonlocal marks
            marks += 1

    class AsyncStream:
        def __aiter__(self) -> "AsyncStream":
            return self

        async def __anext__(self) -> str:
            next_started.set()
            await async_release.wait()
            return chunk

    class SyncStream:
        def __iter__(self) -> "SyncStream":
            return self

        def __next__(self) -> str:
            loop.call_soon_threadsafe(next_started.set)
            sync_release.wait(timeout=2.0)
            return chunk

    async def mark_late_chunk(raw_chunk: Any) -> None:
        if audio_streaming_module._audio_provider_chunk_has_nonempty_content(raw_chunk):
            await audio_streaming_module._mark_audio_credentials_used_once(
                Runtime(),
                object(),
                usage_claimed,
            )

    async def cleanup() -> None:
        cleanup_finished.set()

    monkeypatch.setattr(audio_streaming_module, "AUDIO_STREAM_NEXT_TIMEOUT_SECONDS", 0.5)
    stream: Any = AsyncStream() if stream_mode == "async" else SyncStream()
    iterator = audio_streaming_module._iterate_audio_provider_stream(
        stream,
        resource_holder={},
        on_abandoned=cleanup,
        cleanup_claimed=cleanup_claimed,
        on_late_chunk=mark_late_chunk,
    )
    operation = asyncio.create_task(iterator.__anext__())
    try:
        await asyncio.wait_for(next_started.wait(), timeout=2.0)

        if abandonment == "caller_cancel":
            operation.cancel()
            with pytest.raises(asyncio.CancelledError):
                await operation
        else:
            with pytest.raises(TimeoutError, match="audio-stream-next timed out"):
                await operation
        assert marks == 0
    finally:
        async_release.set()
        sync_release.set()
        if not operation.done():
            operation.cancel()
            await asyncio.gather(operation, return_exceptions=True)

    await asyncio.wait_for(cleanup_finished.wait(), timeout=2.0)
    assert marks == expected_marks


@pytest.mark.asyncio
@pytest.mark.parametrize("abandonment", ["timeout", "caller_cancel"])
async def test_audio_late_sync_iterator_is_closed_with_its_source(
    monkeypatch: pytest.MonkeyPatch,
    abandonment: str,
) -> None:
    """A distinct iterator created after abandonment remains cleanup-owned."""
    loop = asyncio.get_running_loop()
    iterator_started = asyncio.Event()
    release_iterator = threading.Event()
    cleanup_finished = asyncio.Event()
    cleanup_claimed = threading.Event()
    lifecycle: list[str] = []
    holder: dict[str, Any] = {}

    class Iterator:
        def __next__(self) -> str:
            raise StopIteration

        def close(self) -> None:
            lifecycle.append("iterator_close")

    iterator = Iterator()

    class Source:
        def __iter__(self) -> Iterator:
            loop.call_soon_threadsafe(iterator_started.set)
            release_iterator.wait(timeout=2.0)
            return iterator

        def close(self) -> None:
            lifecycle.append("source_close")

    source = Source()

    async def cleanup() -> None:
        await audio_streaming_module._close_audio_provider_stream(
            source,
            holder,
            owned_cleanup=True,
        )
        lifecycle.append("runtime_close")
        cleanup_finished.set()

    monkeypatch.setattr(
        audio_streaming_module,
        "AUDIO_STREAM_ITERATOR_TIMEOUT_SECONDS",
        0.5 if abandonment == "timeout" else 30.0,
    )
    stream = audio_streaming_module._iterate_audio_provider_stream(
        source,
        resource_holder=holder,
        on_abandoned=cleanup,
        cleanup_claimed=cleanup_claimed,
    )
    operation = asyncio.create_task(stream.__anext__())
    await asyncio.wait_for(iterator_started.wait(), timeout=2.0)
    if abandonment == "caller_cancel":
        operation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await operation
    else:
        with pytest.raises(TimeoutError, match="audio-stream-iterator timed out"):
            await operation
    assert lifecycle == []

    release_iterator.set()
    await asyncio.wait_for(cleanup_finished.wait(), timeout=2.0)

    assert lifecycle == ["iterator_close", "source_close", "runtime_close"]


@pytest.mark.asyncio
async def test_audio_async_next_timeout_defers_resource_release_until_next_exits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A resistant async iterator keeps its stream/runtime lease after timeout."""
    from tldw_Server_API.app.core.Chat import streaming_utils

    release = asyncio.Event()
    next_started = asyncio.Event()
    cleanup_finished = asyncio.Event()
    cleanup_claimed = threading.Event()
    lifecycle: list[str] = []

    class BlockingAsyncStream:
        def __aiter__(self) -> "BlockingAsyncStream":
            return self

        async def __anext__(self) -> str:
            next_started.set()
            await release.wait()
            lifecycle.append("next_exit")
            return "late"

        async def aclose(self) -> None:
            lifecycle.append("stream_close")

    stream = BlockingAsyncStream()

    async def cleanup() -> None:
        await audio_streaming_module._close_audio_provider_stream(
            stream,
            holder,
            owned_cleanup=True,
        )
        lifecycle.append("runtime_close")
        cleanup_finished.set()

    holder: dict[str, Any] = {}
    monkeypatch.setattr(audio_streaming_module, "AUDIO_STREAM_NEXT_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(streaming_utils, "STREAM_CLEANUP_TASK_MAX_ACTIVE", 1)
    iterator = audio_streaming_module._iterate_audio_provider_stream(
        stream,
        resource_holder=holder,
        on_abandoned=cleanup,
        cleanup_claimed=cleanup_claimed,
    )

    with pytest.raises(TimeoutError, match="audio-stream-next timed out"):
        await iterator.__anext__()
    assert next_started.is_set()
    assert cleanup_claimed.is_set()
    assert lifecycle == []

    release.set()
    await asyncio.wait_for(cleanup_finished.wait(), timeout=1.0)
    assert lifecycle == ["next_exit", "stream_close", "runtime_close"]


@pytest.mark.asyncio
async def test_audio_sync_timeout_defers_resource_release_until_worker_exits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A timed-out sync boundary exposes true release to late cleanup."""
    from tldw_Server_API.app.core.Chat.bounded_daemon import (
        BoundedDaemonPool,
        DaemonCapacityError,
    )

    started = threading.Event()
    release = threading.Event()
    cleanup_finished = asyncio.Event()
    cleanup_claimed = threading.Event()
    lifecycle: list[str] = []
    rejected_started = False

    def blocked() -> None:
        started.set()
        release.wait(timeout=2.0)
        lifecycle.append("worker_exit")

    async def cleanup() -> None:
        lifecycle.append("resource_close")
        cleanup_finished.set()

    pool = BoundedDaemonPool(1)
    monkeypatch.setattr(audio_streaming_module, "STREAM_DAEMON_POOL", pool)
    with pytest.raises(TimeoutError, match="audio-factory timed out"):
        await audio_streaming_module._run_bounded_audio_sync_call(
            blocked,
            name="audio-factory",
            timeout_seconds=0.01,
            on_abandoned=cleanup,
            cleanup_claimed=cleanup_claimed,
        )
    assert started.is_set()
    assert cleanup_claimed.is_set()
    assert lifecycle == []

    def rejected() -> None:
        nonlocal rejected_started
        rejected_started = True

    with pytest.raises(DaemonCapacityError):
        await audio_streaming_module._run_bounded_audio_sync_call(
            rejected,
            name="audio-factory",
            timeout_seconds=0.1,
        )
    assert rejected_started is False

    release.set()
    await asyncio.wait_for(cleanup_finished.wait(), timeout=1.0)
    assert lifecycle == ["worker_exit", "resource_close"]
    assert pool.active_count == 0
    assert await audio_streaming_module._run_bounded_audio_sync_call(
        lambda: "recovered",
        name="audio-factory",
        timeout_seconds=0.1,
    ) == "recovered"


class DummyWebSocket:
    """In-memory WebSocket stub used for audio chat WebSocket tests."""

    def __init__(self, messages: Iterable[Dict[str, Any] | str]) -> None:
        self.headers: Dict[str, str] = {}
        self.query_params: Dict[str, str] = {}
        self.client = SimpleNamespace(host="127.0.0.1")
        self.state = SimpleNamespace()
        self._messages: List[str] = [
            json.dumps(_normalize_ws_test_message(m)) if isinstance(m, dict) else m
            for m in messages
        ]
        self.sent_bytes: List[bytes] = []
        self.sent_json: List[Dict[str, Any]] = []
        self.sent_events: List[tuple[str, Any]] = []
        self.accepted: bool = False
        self.closed: bool = False
        self.close_code: Optional[int] = None
        self.close_calls: List[int] = []

    async def accept(self) -> None:
        """Mark the WebSocket as accepted."""
        self.accepted = True

    async def receive_text(self) -> str:
        """Return the next queued text frame, or raise when exhausted."""
        if not self._messages:
            raise RuntimeError("No more messages")  # noqa: TRY003
        return self._messages.pop(0)

    async def send_bytes(self, data: bytes) -> None:
        """Record bytes sent over the WebSocket."""
        self.sent_bytes.append(data)
        self.sent_events.append(("bytes", data))

    async def send_json(self, payload: Dict[str, Any]) -> None:
        """Record JSON payloads sent over the WebSocket."""
        self.sent_json.append(payload)
        self.sent_events.append(("json", dict(payload)))

    async def close(self, code: int = 1000, reason: Optional[str] = None) -> None:  # noqa: ARG002
        """Record the close code and mark the WebSocket as closed."""
        self.close_calls.append(code)
        if not self.closed:
            self.close_code = code
        self.closed = True


class _DummyTranscriber:
    """Minimal streaming transcriber stub used in audio chat WebSocket tests."""

    def __init__(self, config: Any) -> None:  # noqa: ARG002
        self.reset_called = False

    def initialize(self) -> None:

        """Simulate transcriber initialization."""
        return None

    async def process_audio_chunk(self, audio_bytes: bytes) -> Dict[str, Any]:  # noqa: ARG002
        """Return a fixed partial transcription payload."""
        return {"type": "partial", "text": "hi"}

    def get_full_transcript(self) -> str:

        """Return a fixed full transcript."""
        return "hello world"

    def reset(self) -> None:

        """Record that reset was called."""
        self.reset_called = True

    def cleanup(self) -> None:
        return None


class _EchoTranscriber:
    """Streaming transcriber stub that reflects input bytes into transcript state."""

    instances: List["_EchoTranscriber"] = []

    def __init__(self, config: Any) -> None:  # noqa: ARG002
        self.current_chunks: List[str] = []
        self.processed_history: List[str] = []
        self.reset_calls = 0
        type(self).instances.append(self)

    def initialize(self) -> None:
        return None

    async def process_audio_chunk(self, audio_bytes: bytes) -> Dict[str, Any]:
        if len(audio_bytes) >= 4:
            first_sample = int(round(struct.unpack("<f", audio_bytes[:4])[0] * 32768))
            text = _AUDIO_SAMPLE_TO_LABEL.get(first_sample, f"pcm16:{first_sample}")
        else:
            text = "pcm16:empty"
        self.current_chunks.append(text)
        self.processed_history.append(text)
        return {"type": "partial", "text": text}

    def get_full_transcript(self) -> str:
        return "|".join(self.current_chunks)

    def reset(self) -> None:
        self.reset_calls += 1
        self.current_chunks = []

    def cleanup(self) -> None:
        return None


class _DummyVAD:
    """Simple VAD stub that never triggers an auto-commit."""

    available: bool = True
    unavailable_reason: Optional[str] = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
        self.last_trigger_at = time.time()

    def observe(self, audio_bytes: bytes) -> bool:  # noqa: ARG002
        """Update the last trigger timestamp and return False (no commit)."""
        self.last_trigger_at = time.time()
        return False


class _DummyRegistry:
    """Simple in-memory metrics registry stub."""

    def __init__(self) -> None:

        self.records: List[tuple[str, str, Any, Optional[Dict[str, Any]]]] = []
        self.registered: List[Any] = []

    def increment(self, name: str, value: int = 1, labels: Optional[Dict[str, Any]] = None) -> None:
        """Record an increment call."""
        self.records.append(("inc", name, value, labels))

    def observe(self, name: str, value: float, labels: Optional[Dict[str, Any]] = None) -> None:
        """Record an observe call."""
        self.records.append(("obs", name, value, labels))

    def register_metric(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
        """Record that a metric registration was requested."""
        self.registered.append(args)


class _DummyTTSService:
    """TTS service stub that yields a fixed sequence of chunks."""

    def __init__(self, chunks: Iterable[bytes]) -> None:
        """Initialize the stub with a fixed sequence of chunks."""
        self._chunks = list(chunks)

    async def generate_speech(self, *args: Any, **kwargs: Any) -> AsyncIterator[bytes]:  # noqa: ARG002
        """Yield preconfigured audio chunks."""
        for chunk in self._chunks:
            yield chunk


class _DummyRealtimeSession:
    """Minimal realtime TTS session used by overlap tests."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue = asyncio.Queue()
        self._buffer = ""
        self._closed = False

    async def push_text(self, delta: str) -> None:
        if self._closed:
            return
        self._buffer += str(delta or "")

    async def commit(self) -> None:
        if self._closed:
            return
        text = self._buffer.strip()
        self._buffer = ""
        if text:
            await self._queue.put(f"rt:{text}".encode("utf-8"))

    async def finish(self) -> None:
        if self._closed:
            return
        if self._buffer.strip():
            await self.commit()
        self._closed = True
        await self._queue.put(None)

    async def audio_stream(self) -> AsyncIterator[bytes]:
        while True:
            item = await self._queue.get()
            if item is None:
                break
            yield item


class _DummyRealtimeCapableTTSService:
    """TTS service exposing both realtime and buffered methods for overlap tests."""

    def __init__(self) -> None:
        self.session = _DummyRealtimeSession()

    async def open_realtime_session(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG002
        return SimpleNamespace(session=self.session, provider="stub-realtime", warning=None)

    async def generate_speech(self, *args: Any, **kwargs: Any) -> AsyncIterator[bytes]:  # noqa: ARG002
        # Legacy fallback path (pre-overlap implementation).
        yield b"legacy-tts"


async def _llm_stub(**kwargs: Any) -> AsyncIterator[str]:  # noqa: ARG002
    """Stubbed streaming LLM generator returning a short response."""

    async def _gen() -> AsyncIterator[str]:
        yield 'data: {"choices":[{"delta":{"content":"hey "}}]}\n\n'
        yield 'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
        yield "data: [DONE]\n\n"

    return _gen()


def _enable_chat_ws_control_v2(monkeypatch: pytest.MonkeyPatch) -> None:
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Unified as unified
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.ws_control_protocol import (
        WSControlProtocolConfig,
    )

    monkeypatch.setattr(
        unified,
        "_get_ws_control_protocol_config",
        lambda: WSControlProtocolConfig(
            ws_control_v2_enabled=True,
            paused_audio_queue_cap_seconds=2.0,
            overflow_warning_interval_seconds=5.0,
        ),
        raising=False,
    )


def _pcm16_audio(samples: Iterable[int] = (0,)) -> str:
    """Return base64-encoded little-endian PCM16 samples."""
    sample_list = list(samples)
    raw = struct.pack("<" + "h" * len(sample_list), *sample_list)
    return base64.b64encode(raw).decode("ascii")


def _strict_chat_config(
    *,
    mode: str = "voice_chat",
    stt: Optional[Dict[str, Any]] = None,
    llm: Optional[Dict[str, Any]] = None,
    tts: Optional[Dict[str, Any]] = None,
    **overrides: Any,
) -> Dict[str, Any]:
    """Return a strict v1 audio chat config frame."""
    payload: Dict[str, Any] = {
        "type": "config",
        "protocol_version": 1,
        "mode": mode,
        "audio_format": "pcm16",
        "sample_rate": 16000,
        "channels": 1,
        "stt": stt if stt is not None else {"model": "parakeet"},
        "llm": llm if llm is not None else {"provider": "stub", "model": "stub-model"},
        "tts": tts if tts is not None else {"voice": "af_heart", "format": "pcm"},
    }
    payload.update(overrides)
    return payload


def _normalize_ws_test_message(message: Dict[str, Any]) -> Dict[str, Any]:
    """Upgrade legacy positive-path test config frames to strict v1."""
    if message.get("type") != "config" or "protocol_version" in message:
        return message
    return {
        "protocol_version": 1,
        "mode": "voice_chat",
        "audio_format": "pcm16",
        "sample_rate": 16000,
        "channels": 1,
        **message,
    }


@pytest.fixture(autouse=True)
def mock_audio_ws_dependencies(monkeypatch: pytest.MonkeyPatch) -> _DummyRegistry:
    """Fixture that sets up common mocks for audio streaming WebSocket tests."""
    set_llm_provider_overrides_cache_for_tests({}, healthy=True)

    async def _auth(*_args: Any, **_kwargs: Any) -> tuple[bool, int]:
        return True, 1

    async def _can_start_stream(_user_id: int) -> tuple[bool, Optional[str]]:
        return True, None

    async def _finish_stream(_user_id: int) -> None:
        return None

    async def _allow_minutes(_uid: int, _minutes: float) -> tuple[bool, Optional[float]]:
        return True, None

    async def _add_minutes(_uid: int, _minutes: float) -> None:
        return None

    async def _hb(_uid: int) -> None:
        return None

    monkeypatch.setattr(audio, "_audio_ws_authenticate", _auth)
    monkeypatch.setattr(audio, "can_start_stream", _can_start_stream)
    monkeypatch.setattr(audio, "finish_stream", _finish_stream)
    monkeypatch.setattr(audio, "check_daily_minutes_allow", _allow_minutes)
    monkeypatch.setattr(audio, "add_daily_minutes", _add_minutes)
    monkeypatch.setattr(audio, "heartbeat_stream", _hb)
    monkeypatch.setattr(audio, "UnifiedStreamingTranscriber", _DummyTranscriber)
    monkeypatch.setattr(audio, "SileroTurnDetector", _DummyVAD)
    monkeypatch.setattr(audio, "chat_api_call_async", _llm_stub)
    monkeypatch.setattr(audio, "get_api_keys", lambda: {"stub": "fake"})
    monkeypatch.setattr(
        provider_credential_runtime,
        "load_server_config_snapshot",
        lambda: {"stub_api": {"api_key": "fake", "model": "stub-model"}},
    )

    registry = _DummyRegistry()
    monkeypatch.setattr(audio, "get_metrics_registry", lambda: registry)

    return registry


@pytest.mark.integration
async def test_audio_chat_ws_rejects_audio_before_strict_config() -> None:
    ws = DummyWebSocket(
        [
            {"type": "audio", "data": _pcm16_audio()},
            {"type": "stop"},
        ]
    )

    await audio.websocket_audio_chat_stream(ws, token=None)

    assert any(msg.get("type") == "error" and msg.get("code") == "bad_request" for msg in ws.sent_json)
    assert ws.close_code == 4400


@pytest.mark.integration
async def test_audio_chat_ws_rejects_transcribe_only_mode() -> None:
    ws = DummyWebSocket(
        [
            _strict_chat_config(mode="dictate"),
            {"type": "stop"},
        ]
    )

    await audio.websocket_audio_chat_stream(ws, token=None)

    assert any("not allowed" in msg.get("message", "") for msg in ws.sent_json)
    assert ws.close_code == 4400


@pytest.mark.integration
async def test_audio_chat_ws_push_to_talk_release_commits_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ws = DummyWebSocket(
        [
            _strict_chat_config(mode="push_to_talk"),
            {"type": "audio", "data": _pcm16_audio([1000])},
            {"type": "push_to_talk_release"},
            {"type": "stop"},
        ]
    )

    async def _get_tts_service():
        return _DummyTTSService([b"tts"])

    monkeypatch.setattr(audio, "get_tts_service", _get_tts_service)

    await audio.websocket_audio_chat_stream(ws, token=None)

    full_transcripts = [msg for msg in ws.sent_json if msg.get("type") == "full_transcript"]
    assert full_transcripts
    assert full_transcripts[0].get("commit_source") == "push_to_talk_release"


@pytest.mark.integration
async def test_audio_chat_ws_push_to_talk_ignores_vad_auto_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _TriggeringVAD(_DummyVAD):
        def observe(self, audio_bytes: bytes) -> bool:  # noqa: ARG002
            self.last_trigger_at = 1234.5
            return True

    ws = DummyWebSocket(
        [
            _strict_chat_config(mode="push_to_talk"),
            {"type": "audio", "data": _pcm16_audio([1000])},
            {"type": "stop"},
        ]
    )

    async def _get_tts_service():
        return _DummyTTSService([b"tts"])

    monkeypatch.setattr(audio, "SileroTurnDetector", _TriggeringVAD)
    monkeypatch.setattr(audio, "get_tts_service", _get_tts_service)

    await audio.websocket_audio_chat_stream(ws, token=None)

    assert not [msg for msg in ws.sent_json if msg.get("type") == "full_transcript"]


@pytest.mark.integration
async def test_audio_chat_ws_streams_llm_and_tts(monkeypatch: pytest.MonkeyPatch) -> None:
    audio_payload = _pcm16_audio([_AUDIO_LABEL_TO_SAMPLE["abc"]])
    messages = [
        {
            "type": "config",
            "stt": {"model": "parakeet"},
            "llm": {"provider": "stub", "model": "stub-model"},
            "tts": {"voice": "af_heart", "format": "pcm"},
        },
        {"type": "audio", "data": audio_payload},
        {"type": "commit"},
        {"type": "stop"},
    ]
    ws = DummyWebSocket(messages)

    async def _get_tts_service():
        return _DummyTTSService([b"tts1", b"tts2"])

    monkeypatch.setattr(audio, "get_tts_service", _get_tts_service)

    await audio.websocket_audio_chat_stream(ws, token=None)

    # Assert LLM delta and transcript were sent
    assert any(msg.get("type") == "full_transcript" for msg in ws.sent_json)
    assert any(msg.get("type") == "llm_delta" for msg in ws.sent_json), json.dumps(
        ws.sent_json,
        indent=2,
    )
    assert any(msg.get("type") == "tts_done" for msg in ws.sent_json)
    assert ws.sent_bytes == [b"tts1", b"tts2"]
    assert ws.closed is True


@pytest.mark.integration
async def test_audio_chat_ws_tts_keeps_two_user_runtime_snapshots_isolated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Buffered WS chat TTS cannot dispatch a global or neighboring user's key."""
    from tldw_Server_API.app.core.Audio import tts_service as tts_credential_service
    from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal

    def messages() -> list[dict[str, Any]]:
        return [
            _strict_chat_config(
                llm={"provider": "stub", "model": "stub-model"},
                tts={
                    "provider": "openai",
                    "model": "tts-1",
                    "voice": "alloy",
                    "format": "pcm",
                },
            ),
            {"type": "audio", "data": _pcm16_audio([1000])},
            {"type": "commit"},
            {"type": "stop"},
        ]

    first_ws = DummyWebSocket(messages())
    second_ws = DummyWebSocket(messages())
    user_ids = {id(first_ws): 101, id(second_ws): 202}
    entered = {user_id: asyncio.Event() for user_id in user_ids.values()}
    release = {user_id: asyncio.Event() for user_id in user_ids.values()}
    calls: list[tuple[int, dict[str, Any]]] = []
    runtimes: list[Any] = []

    class LLMRuntime:
        def __init__(self, **kwargs: Any) -> None:
            self.user_id = int(kwargs["user_id"])

        async def resolve(self, provider: str, *, model: str | None = None) -> Any:
            return SimpleNamespace(
                provider=provider,
                api_key=f"llm-user-{self.user_id}-key",
                app_config={"stub_api": {"model": model}},
                auth_source="api_key",
                credentials_resolved=True,
            )

        async def mark_used(self, _handle: object) -> None:
            return None

        async def close(self) -> None:
            return None

    class TTSRuntime:
        def __init__(self, **kwargs: Any) -> None:
            self.user_id = int(kwargs["user_id"])
            self.handles: list[Any] = []
            self.marked: list[Any] = []
            self.closed = False
            runtimes.append(self)

        async def resolve(self, provider: str, *, model: str | None = None) -> Any:
            handle = SimpleNamespace(
                provider=provider,
                api_key=f"chat-tts-user-{self.user_id}-key",
                app_config={
                    "openai_api": {
                        "api_base_url": f"https://chat-tts-user-{self.user_id}.example/v1",
                        "model": model,
                    }
                },
                auth_source="api_key",
                credentials_resolved=True,
            )
            self.handles.append(handle)
            return handle

        async def mark_used(self, handle: object) -> None:
            self.marked.append(handle)

        async def close(self) -> None:
            self.closed = True

    class RecordingTTSService:
        async def generate_speech(self, _request: Any, **kwargs: Any) -> AsyncIterator[bytes]:
            user_id = int(kwargs["user_id"])
            calls.append((user_id, dict(kwargs)))
            entered[user_id].set()
            await release[user_id].wait()
            yield f"tts-{user_id}".encode()

    async def auth_stub(websocket: Any, *_args: Any, **_kwargs: Any) -> tuple[bool, int]:
        user_id = user_ids[id(websocket)]
        websocket.state.auth_principal = AuthPrincipal(
            kind="user",
            user_id=user_id,
            subject=f"user:{user_id}",
        )
        return True, user_id

    async def get_tts_service() -> RecordingTTSService:
        return RecordingTTSService()

    monkeypatch.setenv("OPENAI_API_KEY", "global-chat-tts-key-must-not-dispatch")
    monkeypatch.setattr(audio_streaming_module, "is_multi_user_mode", lambda: True)
    monkeypatch.setattr(audio, "_audio_ws_authenticate", auth_stub)
    monkeypatch.setattr(
        audio_streaming_module,
        "ProviderCredentialRuntime",
        LLMRuntime,
    )
    monkeypatch.setattr(audio, "get_tts_service", get_tts_service)
    monkeypatch.setattr(
        tts_credential_service,
        "ProviderCredentialRuntime",
        TTSRuntime,
        raising=False,
    )
    monkeypatch.setattr(
        tts_credential_service,
        "load_server_config_snapshot",
        lambda: {"openai_api": {"api_key": "global-chat-tts-key-must-not-dispatch"}},
    )
    monkeypatch.setattr(
        tts_credential_service,
        "_capture_tts_provider_config",
        lambda _provider: {"enabled": True},
    )

    first = asyncio.create_task(audio.websocket_audio_chat_stream(first_ws, token=None))
    second = asyncio.create_task(audio.websocket_audio_chat_stream(second_ws, token=None))
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
    ) == [
        (101, "chat-tts-user-101-key"),
        (202, "chat-tts-user-202-key"),
    ]
    assert all(kwargs["fallback"] is False for _user_id, kwargs in calls)
    assert "global-chat-tts-key-must-not-dispatch" not in repr(calls)
    assert "global-chat-tts-key-must-not-dispatch" not in repr(
        first_ws.sent_json + second_ws.sent_json
    )
    assert len(runtimes) == 2
    assert all(runtime.marked == runtime.handles for runtime in runtimes)
    assert all(runtime.closed for runtime in runtimes)


@pytest.mark.integration
async def test_audio_chat_ws_realtime_tts_keeps_runtime_through_first_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Realtime overlap receives authoritative overrides and marks its first audio."""
    from tldw_Server_API.app.core.Audio import tts_service as tts_credential_service

    ws = DummyWebSocket(
        [
            _strict_chat_config(
                llm={"provider": "stub", "model": "stub-model"},
                tts={
                    "provider": "openai",
                    "model": "tts-1",
                    "voice": "alloy",
                    "format": "pcm",
                },
            ),
            {"type": "audio", "data": _pcm16_audio([1000])},
            {"type": "commit"},
            {"type": "stop"},
        ]
    )
    lifecycle: list[str] = []
    open_kwargs: list[dict[str, Any]] = []

    class LLMRuntime:
        async def resolve(self, provider: str, *, model: str | None = None) -> Any:
            return SimpleNamespace(
                provider=provider,
                api_key="llm-key",
                app_config={"stub_api": {"model": model}},
                auth_source="api_key",
                credentials_resolved=True,
            )

        async def mark_used(self, _handle: object) -> None:
            return None

        async def close(self) -> None:
            return None

    class TTSRuntime:
        async def resolve(self, provider: str, *, model: str | None = None) -> Any:
            return SimpleNamespace(
                provider=provider,
                api_key="realtime-user-key",
                app_config={"openai_api": {"model": model}},
                auth_source="api_key",
                credentials_resolved=True,
            )

        async def mark_used(self, _handle: object) -> None:
            lifecycle.append("mark_used")

        async def close(self) -> None:
            lifecycle.append("runtime_close")

    class RealtimeService:
        def __init__(self) -> None:
            self.session = _DummyRealtimeSession()

        async def open_realtime_session(self, **kwargs: Any) -> Any:
            open_kwargs.append(dict(kwargs))
            return SimpleNamespace(
                session=self.session,
                provider="openai",
                warning=None,
            )

        async def generate_speech(self, *_args: Any, **_kwargs: Any) -> AsyncIterator[bytes]:
            raise AssertionError("realtime success must not redispatch buffered TTS")
            yield b""  # pragma: no cover

    service = RealtimeService()

    async def get_tts_service() -> RealtimeService:
        return service

    monkeypatch.setattr(
        audio_streaming_module,
        "ProviderCredentialRuntime",
        lambda **_kwargs: LLMRuntime(),
    )
    monkeypatch.setattr(audio, "get_tts_service", get_tts_service)
    monkeypatch.setattr(
        tts_credential_service,
        "ProviderCredentialRuntime",
        lambda **_kwargs: TTSRuntime(),
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

    await audio.websocket_audio_chat_stream(ws, token=None)

    assert ws.sent_bytes
    assert open_kwargs
    assert open_kwargs[0]["provider_overrides"] == {
        "credentials_resolved": True,
        "api_key": "realtime-user-key",
        "openai_api_key": "realtime-user-key",
    }
    assert open_kwargs[0]["user_id"] == 1
    assert lifecycle == ["mark_used", "runtime_close"]


@pytest.mark.integration
@pytest.mark.parametrize("dispatch", ["adapter", "no-adapter", "not-implemented"])
@pytest.mark.parametrize("captured_key", ["audio-key-a", None], ids=["a-to-b", "absent-to-b"])
async def test_audio_chat_ws_keeps_static_snapshot_at_llm_boundary(
    monkeypatch: pytest.MonkeyPatch,
    dispatch: str,
    captured_key: str | None,
) -> None:
    """Every WebSocket LLM dispatch branch must keep one credential snapshot."""
    config_a = {"stub": {"model": "stub-model", "api_key": "config-key-a"}}
    boundary_requests: list[dict[str, Any]] = []
    adapter_requests: list[dict[str, Any]] = []
    fallback_requests: list[dict[str, Any]] = []
    adapter_timeouts: list[float | None] = []
    lifecycle: list[object] = []

    class FakeRuntime:
        def __init__(self, **kwargs: Any) -> None:
            lifecycle.append(("init", kwargs))

        async def resolve(self, provider: str, *, model: str | None = None):
            lifecycle.append(("resolve", provider, model))
            return SimpleNamespace(
                api_key=captured_key,
                app_config=config_a,
                credentials_resolved=True,
            )

        async def mark_used(self, _handle: object) -> None:
            lifecycle.append("mark_used")

        async def close(self) -> None:
            lifecycle.append("close")

    async def forbidden_low_level_resolver(*_args: Any, **_kwargs: Any):
        raise AssertionError("audio WebSocket bypassed ProviderCredentialRuntime")

    async def fallback_call(**kwargs: Any) -> AsyncIterator[str]:
        request = dict(kwargs)
        boundary_requests.append(request)
        fallback_requests.append(request)
        return await _llm_stub(**kwargs)

    class RecordingAdapter:
        def astream(
            self,
            request: dict[str, Any],
            *,
            timeout: float | None = None,
        ) -> AsyncIterator[str]:
            captured = dict(request)
            boundary_requests.append(captured)
            adapter_requests.append(captured)
            adapter_timeouts.append(timeout)

            async def _stream() -> AsyncIterator[str]:
                async for line in await _llm_stub():
                    yield line

            return _stream()

    class NotImplementedAdapter:
        async def astream(
            self,
            request: dict[str, Any],
            *,
            timeout: float | None = None,
        ) -> AsyncIterator[str]:
            captured = dict(request)
            boundary_requests.append(captured)
            adapter_requests.append(captured)
            adapter_timeouts.append(timeout)
            raise NotImplementedError

    adapter = None
    if dispatch == "adapter":
        adapter = RecordingAdapter()
    elif dispatch == "not-implemented":
        adapter = NotImplementedAdapter()

    monkeypatch.setattr(audio_streaming_module, "ProviderCredentialRuntime", FakeRuntime)
    monkeypatch.setattr(
        audio_streaming_module,
        "derive_trusted_credential_scope",
        lambda _request, _user: (1, [2], [3], True),
    )
    monkeypatch.setattr(
        audio_streaming_module,
        "resolve_byok_credentials",
        forbidden_low_level_resolver,
        raising=False,
    )
    monkeypatch.setattr(
        audio_streaming_module,
        "resolve_provider_api_key_from_config",
        lambda *_args: "audio-key-b",
        raising=False,
    )
    monkeypatch.setattr(audio_streaming_module, "provider_requires_api_key", lambda _provider: False, raising=False)
    monkeypatch.setattr(audio_streaming_module, "get_registry", lambda: SimpleNamespace(get_adapter=lambda _p: adapter))
    monkeypatch.setattr(audio, "chat_api_call_async", fallback_call)
    monkeypatch.setattr(audio, "get_api_keys", lambda: {"stub": "audio-key-b"})
    monkeypatch.setattr(audio, "get_tts_service", lambda: asyncio.sleep(0, result=_DummyTTSService([b"tts"])))

    config_frame = _strict_chat_config()
    config_frame["llm"]["extra_params"] = {
        "base_url": "https://attacker.invalid/v1",
        "api_key": "client-injected-key",
        "app_config": {"stub": {"base_url": "https://attacker.invalid/v1"}},
        "seed": 7,
    }
    ws = DummyWebSocket(
        [
            config_frame,
            {"type": "audio", "data": _pcm16_audio([100])},
            {"type": "commit"},
            {"type": "stop"},
        ]
    )
    await audio.websocket_audio_chat_stream(ws, token=None)

    assert boundary_requests
    assert all(request["api_key"] == captured_key for request in boundary_requests)
    assert all(request["app_config"] == config_a for request in boundary_requests)
    assert all(request["credentials_resolved"] is True for request in boundary_requests)
    assert all("timeout" not in request for request in adapter_requests)
    assert all(
        timeout == audio_streaming_module.AUDIO_STREAM_FACTORY_TIMEOUT_SECONDS
        for timeout in adapter_timeouts
    )
    assert all(
        request["timeout"]
        == audio_streaming_module.AUDIO_STREAM_FACTORY_TIMEOUT_SECONDS
        for request in fallback_requests
    )
    assert all("base_url" not in request for request in boundary_requests)
    assert all(request.get("seed") == 7 or request.get("extra_body", {}).get("seed") == 7 for request in boundary_requests)
    init_kwargs = lifecycle[0][1]
    assert init_kwargs["user_id"] == 1
    assert init_kwargs["team_ids"] == [2]
    assert init_kwargs["org_ids"] == [3]
    assert init_kwargs["trusted_base_url_override"] is True
    assert lifecycle[1:] == [
        ("resolve", "stub", "stub-model"),
        "mark_used",
        "close",
    ]


@pytest.mark.integration
async def test_audio_chat_ws_sync_stream_runs_off_loop_and_closes_before_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lifecycle: list[str] = []
    next_thread_ids: list[int] = []
    bounded_calls: list[tuple[str, float, str]] = []
    loop_thread_id = threading.get_ident()
    real_bounded_call = audio_streaming_module.await_bounded_daemon_with_timeout

    async def _bounded_call(
        call,
        *,
        pool: Any,
        name: str,
        timeout_seconds: float,
        timeout_message: str,
        released_event: threading.Event | None = None,
        retain_result_after_timeout: bool = False,
    ) -> Any:
        bounded_calls.append((name, timeout_seconds, timeout_message))
        return await real_bounded_call(
            call,
            pool=pool,
            name=name,
            timeout_seconds=timeout_seconds,
            timeout_message=timeout_message,
            released_event=released_event,
            retain_result_after_timeout=retain_result_after_timeout,
        )

    class Runtime:
        async def resolve(self, _provider: str, *, model: str | None = None):
            return SimpleNamespace(
                api_key="audio-key",
                app_config={"stub": {"model": model}},
                credentials_resolved=True,
            )

        async def mark_used(self, _handle: object) -> None:
            lifecycle.append("mark_used")

        async def close(self) -> None:
            lifecycle.append("runtime_close")

    class SyncStream:
        def __init__(self) -> None:
            self._items = iter(
                [
                    'data: {"choices":[{"delta":{"content":"sync reply"}}]}\n\n',
                    "data: [DONE]\n\n",
                ]
            )
            self._closed = False

        def __iter__(self):
            return self

        def __next__(self) -> str:
            next_thread_ids.append(threading.get_ident())
            return next(self._items)

        def close(self) -> None:
            if not self._closed:
                self._closed = True
                lifecycle.append("stream_close")

    stream = SyncStream()

    async def _fallback(**_kwargs: Any) -> SyncStream:
        return stream

    monkeypatch.setattr(
        audio_streaming_module,
        "ProviderCredentialRuntime",
        lambda **_kwargs: Runtime(),
    )
    monkeypatch.setattr(
        audio_streaming_module,
        "await_bounded_daemon_with_timeout",
        _bounded_call,
    )
    monkeypatch.setattr(
        audio_streaming_module,
        "get_registry",
        lambda: SimpleNamespace(get_adapter=lambda _provider: None),
    )
    monkeypatch.setattr(audio, "chat_api_call_async", _fallback)
    monkeypatch.setattr(
        audio,
        "get_tts_service",
        lambda: asyncio.sleep(0, result=_DummyTTSService([b"tts"])),
    )

    ws = DummyWebSocket(
        [
            _strict_chat_config(),
            {"type": "audio", "data": _pcm16_audio([100])},
            {"type": "commit"},
            {"type": "stop"},
        ]
    )
    await audio.websocket_audio_chat_stream(ws, token=None)

    assert next_thread_ids
    assert all(thread_id != loop_thread_id for thread_id in next_thread_ids)
    assert {name for name, _timeout, _message in bounded_calls} == {
        "audio-stream-iterator",
        "audio-stream-next",
    }
    assert all(timeout > 0 for _name, timeout, _message in bounded_calls)
    assert all(message.endswith(" timed out") for _name, _timeout, message in bounded_calls)
    assert "mark_used" in lifecycle
    assert lifecycle.index("stream_close") < lifecycle.index("runtime_close")


@pytest.mark.integration
async def test_audio_chat_ws_closes_distinct_iterator_and_source_before_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A wrapper and its distinct iterator both retain runtime ownership."""
    lifecycle: list[str] = []

    class Runtime:
        async def resolve(self, _provider: str, *, model: str | None = None):
            return SimpleNamespace(
                api_key="audio-key",
                app_config={"stub": {"model": model}},
                credentials_resolved=True,
            )

        async def mark_used(self, _handle: object) -> None:
            lifecycle.append("mark_used")

        async def close(self) -> None:
            lifecycle.append("runtime_close")

    class Iterator:
        def __init__(self) -> None:
            self._items = iter(
                [
                    'data: {"choices":[{"delta":{"content":"reply"}}]}',
                    "data: [DONE]",
                ]
            )

        def __aiter__(self):
            return self

        async def __anext__(self) -> str:
            try:
                return next(self._items)
            except StopIteration as exc:
                raise StopAsyncIteration from exc

        async def aclose(self) -> None:
            lifecycle.append("iterator_close")

    class Source:
        def __init__(self) -> None:
            self.iterator = Iterator()

        def __aiter__(self) -> Iterator:
            return self.iterator

        async def aclose(self) -> None:
            lifecycle.append("source_close")

    async def _fallback(**_kwargs: Any) -> Source:
        return Source()

    monkeypatch.setattr(
        audio_streaming_module,
        "ProviderCredentialRuntime",
        lambda **_kwargs: Runtime(),
    )
    monkeypatch.setattr(
        audio_streaming_module,
        "get_registry",
        lambda: SimpleNamespace(get_adapter=lambda _provider: None),
    )
    monkeypatch.setattr(audio, "chat_api_call_async", _fallback)
    monkeypatch.setattr(
        audio,
        "get_tts_service",
        lambda: asyncio.sleep(0, result=_DummyTTSService([b"tts"])),
    )

    ws = DummyWebSocket(
        [
            _strict_chat_config(),
            {"type": "audio", "data": _pcm16_audio([100])},
            {"type": "commit"},
            {"type": "stop"},
        ]
    )
    await audio.websocket_audio_chat_stream(ws, token=None)

    assert "iterator_close" in lifecycle
    assert "source_close" in lifecycle
    assert lifecycle.index("iterator_close") < lifecycle.index("runtime_close")
    assert lifecycle.index("source_close") < lifecycle.index("runtime_close")


@pytest.mark.integration
async def test_audio_chat_ws_close_timeout_retains_runtime_until_close_exits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A resistant normal close keeps ownership after its diagnostic deadline."""
    from tldw_Server_API.app.core.Chat import streaming_utils

    close_started = asyncio.Event()
    release_close = asyncio.Event()
    runtime_closed = asyncio.Event()
    lifecycle: list[str] = []

    class Runtime:
        async def resolve(self, _provider: str, *, model: str | None = None):
            return SimpleNamespace(
                api_key="audio-key",
                app_config={"stub": {"model": model}},
                credentials_resolved=True,
            )

        async def mark_used(self, _handle: object) -> None:
            lifecycle.append("mark_used")

        async def close(self) -> None:
            lifecycle.append("runtime_close")
            runtime_closed.set()

    class ResistantCloseStream:
        def __init__(self) -> None:
            self._items = iter(
                [
                    'data: {"choices":[{"delta":{"content":"reply"}}]}',
                    "data: [DONE]",
                ]
            )

        def __aiter__(self) -> "ResistantCloseStream":
            return self

        async def __anext__(self) -> str:
            try:
                return next(self._items)
            except StopIteration as exc:
                raise StopAsyncIteration from exc

        async def aclose(self) -> None:
            lifecycle.append("close_start")
            close_started.set()
            try:
                await release_close.wait()
            except asyncio.CancelledError:
                lifecycle.append("close_cancel_received")
                await release_close.wait()
            lifecycle.append("close_exit")

    class GatedStopWebSocket(DummyWebSocket):
        runtime_released_before_close = False

        async def receive_text(self) -> str:
            if self._messages:
                payload = json.loads(self._messages[0])
                if payload.get("type") == "stop":
                    await close_started.wait()
                    runtime_waiter = asyncio.create_task(runtime_closed.wait())
                    done, _pending = await asyncio.wait(
                        {runtime_waiter},
                        timeout=0.05,
                    )
                    self.runtime_released_before_close = runtime_waiter in done
                    if not runtime_waiter.done():
                        runtime_waiter.cancel()
                    await asyncio.gather(runtime_waiter, return_exceptions=True)
                    release_close.set()
            return await super().receive_text()

    async def fallback(**_kwargs: Any) -> ResistantCloseStream:
        return ResistantCloseStream()

    monkeypatch.setattr(
        audio_streaming_module,
        "ProviderCredentialRuntime",
        lambda **_kwargs: Runtime(),
    )
    monkeypatch.setattr(
        audio_streaming_module,
        "get_registry",
        lambda: SimpleNamespace(get_adapter=lambda _provider: None),
    )
    monkeypatch.setattr(
        audio_streaming_module,
        "AUDIO_STREAM_CLOSE_TIMEOUT_SECONDS",
        0.005,
        raising=False,
    )
    monkeypatch.setattr(streaming_utils, "STREAM_CLEANUP_TIMEOUT_SECONDS", 0.005)
    monkeypatch.setattr(streaming_utils, "STREAM_TASK_CANCEL_DRAIN_SECONDS", 0.005)
    monkeypatch.setattr(audio, "chat_api_call_async", fallback)
    monkeypatch.setattr(
        audio,
        "get_tts_service",
        lambda: asyncio.sleep(0, result=_DummyTTSService([b"tts"])),
    )

    ws = GatedStopWebSocket(
        [
            _strict_chat_config(),
            {"type": "audio", "data": _pcm16_audio([100])},
            {"type": "commit"},
            {"type": "stop"},
        ]
    )
    await audio.websocket_audio_chat_stream(ws, token=None)
    await asyncio.wait_for(runtime_closed.wait(), timeout=1.0)

    assert ws.runtime_released_before_close is False
    assert lifecycle.index("close_exit") < lifecycle.index("runtime_close")


@pytest.mark.integration
@pytest.mark.parametrize("stream_kind", ["empty", "error"])
@pytest.mark.parametrize("outer_stream_available", [True, False])
async def test_audio_chat_ws_empty_or_error_stream_is_unmarked_and_sanitized(
    monkeypatch: pytest.MonkeyPatch,
    stream_kind: str,
    outer_stream_available: bool,
) -> None:
    lifecycle: list[str] = []
    secret = "private-provider-secret"

    class Runtime:
        async def resolve(self, _provider: str, *, model: str | None = None):
            return SimpleNamespace(
                api_key="audio-key",
                app_config={"stub": {"model": model}},
                credentials_resolved=True,
            )

        async def mark_used(self, _handle: object) -> None:
            lifecycle.append("mark_used")

        async def close(self) -> None:
            lifecycle.append("runtime_close")

    async def _fallback(**_kwargs: Any) -> AsyncIterator[str]:
        async def _stream() -> AsyncIterator[str]:
            if stream_kind == "error":
                yield f'data: {{"error":{{"message":"{secret}"}}}}\n\n'

        return _stream()

    monkeypatch.setattr(
        audio_streaming_module,
        "ProviderCredentialRuntime",
        lambda **_kwargs: Runtime(),
    )
    monkeypatch.setattr(
        audio_streaming_module,
        "get_registry",
        lambda: SimpleNamespace(get_adapter=lambda _provider: None),
    )
    monkeypatch.setattr(audio, "chat_api_call_async", _fallback)
    monkeypatch.setattr(
        audio,
        "get_tts_service",
        lambda: asyncio.sleep(0, result=_DummyTTSService([b"tts"])),
    )
    if not outer_stream_available:
        import tldw_Server_API.app.core.Streaming.streams as streams

        class _UnavailableOuterStream:
            def __init__(self, *_args: Any, **_kwargs: Any) -> None:
                raise RuntimeError("outer stream unavailable")

        monkeypatch.setattr(streams, "WebSocketStream", _UnavailableOuterStream)

    ws = DummyWebSocket(
        [
            _strict_chat_config(),
            {"type": "audio", "data": _pcm16_audio([100])},
            {"type": "commit"},
            {"type": "stop"},
        ]
    )
    await audio.websocket_audio_chat_stream(ws, token=None)

    assert "mark_used" not in lifecycle
    assert lifecycle[-1] == "runtime_close"
    assert secret not in str(ws.sent_json)
    error_codes = [
        payload.get("code")
        for payload in ws.sent_json
        if payload.get("type") == "error"
    ]
    expected_code = "provider_unavailable" if stream_kind == "error" else "empty_assistant"
    assert error_codes == [expected_code]


@pytest.mark.integration
async def test_audio_chat_ws_accepts_bedrock_default_chain_runtime_auth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lifecycle: list[str] = []
    requests: list[dict[str, Any]] = []

    class Runtime:
        async def resolve(self, provider: str, *, model: str | None = None):
            assert (provider, model) == ("bedrock", "bedrock-model")
            return SimpleNamespace(
                api_key=None,
                app_config={
                    "bedrock_api": {
                        "model": model,
                        "_runtime_auth_source": "aws_default_chain",
                    }
                },
                credentials_resolved=True,
            )

        async def mark_used(self, _handle: object) -> None:
            lifecycle.append("mark_used")

        async def close(self) -> None:
            lifecycle.append("runtime_close")

    class Adapter:
        def astream(
            self,
            request: dict[str, Any],
            *,
            timeout: float | None = None,
        ) -> AsyncIterator[str]:
            assert "timeout" not in request
            assert timeout == audio_streaming_module.AUDIO_STREAM_FACTORY_TIMEOUT_SECONDS
            requests.append(request)

            async def _stream() -> AsyncIterator[str]:
                yield 'data: {"choices":[{"delta":{"content":"bedrock reply"}}]}\n\n'
                yield "data: [DONE]\n\n"

            return _stream()

    monkeypatch.setattr(
        audio_streaming_module,
        "ProviderCredentialRuntime",
        lambda **_kwargs: Runtime(),
    )
    monkeypatch.setattr(
        audio_streaming_module,
        "get_registry",
        lambda: SimpleNamespace(get_adapter=lambda _provider: Adapter()),
    )
    monkeypatch.setattr(
        audio,
        "get_tts_service",
        lambda: asyncio.sleep(0, result=_DummyTTSService([b"tts"])),
    )

    ws = DummyWebSocket(
        [
            _strict_chat_config(
                llm={"provider": "bedrock", "model": "bedrock-model"}
            ),
            {"type": "audio", "data": _pcm16_audio([100])},
            {"type": "commit"},
            {"type": "stop"},
        ]
    )
    await audio.websocket_audio_chat_stream(ws, token=None)

    assert requests
    assert requests[0]["api_key"] is None
    assert requests[0]["credentials_resolved"] is True
    assert lifecycle == ["mark_used", "runtime_close"]


@pytest.mark.integration
async def test_audio_chat_ws_partial_success_cancellation_drains_before_runtime_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Interrupting a blocked sync stream must retain its runtime until close."""
    lifecycle: list[str] = []
    waiting_for_release = threading.Event()
    release_stream = threading.Event()

    class Runtime:
        async def resolve(self, _provider: str, *, model: str | None = None):
            return SimpleNamespace(
                api_key="audio-key",
                app_config={"stub": {"model": model}},
                credentials_resolved=True,
            )

        async def mark_used(self, _handle: object) -> None:
            lifecycle.append("mark_used")

        async def close(self) -> None:
            lifecycle.append("runtime_close")

    class BlockingSyncStream:
        def __init__(self) -> None:
            self._index = 0

        def __iter__(self):
            return self

        def __next__(self) -> str:
            self._index += 1
            if self._index == 1:
                return 'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
            if self._index == 2:
                waiting_for_release.set()
                assert release_stream.wait(timeout=5)
                return "data: [DONE]\n\n"
            raise StopIteration

        def close(self) -> None:
            lifecycle.append("stream_close")

    class GatedWebSocket(DummyWebSocket):
        async def receive_text(self) -> str:
            if self._messages:
                payload = json.loads(self._messages[0])
                if payload.get("type") == "interrupt":
                    assert await asyncio.to_thread(waiting_for_release.wait, 5)
            return await super().receive_text()

        async def send_json(self, payload: dict[str, Any]) -> None:
            await super().send_json(payload)
            if payload.get("type") == "interrupted":
                release_stream.set()

    async def _fallback(**_kwargs: Any) -> BlockingSyncStream:
        return BlockingSyncStream()

    monkeypatch.setattr(
        audio_streaming_module,
        "ProviderCredentialRuntime",
        lambda **_kwargs: Runtime(),
    )
    monkeypatch.setattr(
        audio_streaming_module,
        "get_registry",
        lambda: SimpleNamespace(get_adapter=lambda _provider: None),
    )
    monkeypatch.setattr(audio, "chat_api_call_async", _fallback)
    monkeypatch.setattr(
        audio,
        "get_tts_service",
        lambda: asyncio.sleep(0, result=_DummyTTSService([b"tts"])),
    )

    ws = GatedWebSocket(
        [
            _strict_chat_config(),
            {"type": "audio", "data": _pcm16_audio([100])},
            {"type": "commit"},
            {"type": "interrupt", "reason": "test_cancel"},
            {"type": "stop"},
        ]
    )
    await audio.websocket_audio_chat_stream(ws, token=None)

    assert any(message.get("type") == "llm_delta" for message in ws.sent_json)
    assert any(message.get("type") == "interrupted" for message in ws.sent_json)
    for _ in range(100):
        if "runtime_close" in lifecycle:
            break
        await asyncio.sleep(0.01)
    assert lifecycle.count("mark_used") == 1
    assert lifecycle.index("stream_close") < lifecycle.index("runtime_close")


@pytest.mark.integration
async def test_audio_chat_ws_disconnect_cancels_inflight_provider_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A disconnected client must not leave its provider turn running."""
    lifecycle: list[str] = []
    waiting_for_release = threading.Event()
    release_stream = threading.Event()
    runtime_closed = threading.Event()

    class Runtime:
        async def resolve(self, _provider: str, *, model: str | None = None):
            return SimpleNamespace(
                api_key="audio-key",
                app_config={"stub": {"model": model}},
                credentials_resolved=True,
            )

        async def mark_used(self, _handle: object) -> None:
            lifecycle.append("mark_used")

        async def close(self) -> None:
            lifecycle.append("runtime_close")
            runtime_closed.set()

    class BlockingSyncStream:
        def __init__(self) -> None:
            self._index = 0

        def __iter__(self):
            return self

        def __next__(self) -> str:
            self._index += 1
            if self._index == 1:
                return 'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'
            if self._index == 2:
                waiting_for_release.set()
                assert release_stream.wait(timeout=5)
                return 'data: {"choices":[{"delta":{"content":"late"}}]}\n\n'
            if self._index == 3:
                return "data: [DONE]\n\n"
            raise StopIteration

        def close(self) -> None:
            lifecycle.append(f"stream_close_at_{self._index}")

    class DisconnectWebSocket(DummyWebSocket):
        async def receive_text(self) -> str:
            if self._messages:
                return await super().receive_text()
            assert await asyncio.to_thread(waiting_for_release.wait, 5)
            raise WebSocketDisconnect(code=1001)

    async def _fallback(**_kwargs: Any) -> BlockingSyncStream:
        return BlockingSyncStream()

    monkeypatch.setattr(
        audio_streaming_module,
        "ProviderCredentialRuntime",
        lambda **_kwargs: Runtime(),
    )
    monkeypatch.setattr(
        audio_streaming_module,
        "get_registry",
        lambda: SimpleNamespace(get_adapter=lambda _provider: None),
    )
    monkeypatch.setattr(audio, "chat_api_call_async", _fallback)
    monkeypatch.setattr(
        audio,
        "get_tts_service",
        lambda: asyncio.sleep(0, result=_DummyTTSService([b"tts"])),
    )

    ws = DisconnectWebSocket(
        [
            _strict_chat_config(),
            {"type": "audio", "data": _pcm16_audio([100])},
            {"type": "commit"},
        ]
    )
    await audio.websocket_audio_chat_stream(ws, token=None)

    release_stream.set()
    assert await asyncio.to_thread(runtime_closed.wait, 5)
    assert "stream_close_at_2" in lifecycle
    assert lifecycle.index("stream_close_at_2") < lifecycle.index("runtime_close")
    assert not any(
        message.get("type") == "llm_delta" and message.get("text") == "late"
        for message in ws.sent_json
    )


@pytest.mark.integration
async def test_audio_chat_ws_concurrent_turns_keep_runtime_snapshots_isolated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent WebSockets must not mix resolved keys or provider config."""
    boundary_requests: list[dict[str, Any]] = []
    adapter_timeouts: dict[str, float | None] = {}
    lifecycle: list[tuple[str, str]] = []
    runtimes: list[Any] = []
    acquisition_barrier = threading.Barrier(2)

    class Runtime:
        def __init__(self, **_kwargs: Any) -> None:
            self.model = "unresolved"
            self.handles: list[Any] = []
            self.inner: RealProviderCredentialRuntime | None = None
            runtimes.append(self)

        async def resolve(self, provider: str, *, model: str | None = None):
            self.model = str(model)
            app_config = {
                "stub": {
                    "model": self.model,
                    "snapshot": f"config-{self.model}",
                }
            }

            async def resolver(
                normalized_provider: str,
                **_resolver_kwargs: Any,
            ) -> ResolvedByokCredentials:
                return ResolvedByokCredentials(
                    provider=normalized_provider,
                    api_key=f"key-{self.model}",
                    app_config=app_config,
                    credential_fields={},
                    source="user",
                    allowlisted=True,
                    status=ByokResolutionStatus.RESOLVED,
                    auth_source="api_key",
                )

            self.inner = RealProviderCredentialRuntime(
                user_id=1,
                team_ids=(),
                org_ids=(),
                trusted_base_url_override=False,
                server_config_snapshot={},
                resolver=resolver,
            )
            handle = await self.inner.resolve(provider, model=model)
            self.handles.append(handle)
            return handle

        async def mark_used(self, handle: Any) -> None:
            assert handle in self.handles
            lifecycle.append(("mark_used", self.model))

        async def close(self) -> None:
            if self.inner is not None:
                await self.inner.close()
            lifecycle.append(("runtime_close", self.model))

    class Adapter:
        def astream(
            self,
            request: dict[str, Any],
            *,
            timeout: float | None = None,
        ) -> AsyncIterator[str]:
            acquisition_barrier.wait(timeout=5)
            boundary_requests.append(dict(request))
            model = str(request["model"])
            adapter_timeouts[model] = timeout

            async def _stream() -> AsyncIterator[str]:
                yield json.dumps(
                    {"choices": [{"delta": {"content": f"reply-{model}"}}]}
                )
                yield "data: [DONE]"

            return _stream()

    monkeypatch.setattr(audio_streaming_module, "ProviderCredentialRuntime", Runtime)
    monkeypatch.setattr(
        audio_streaming_module,
        "get_registry",
        lambda: SimpleNamespace(get_adapter=lambda _provider: Adapter()),
    )
    monkeypatch.setattr(
        audio,
        "get_tts_service",
        lambda: asyncio.sleep(0, result=_DummyTTSService([b"tts"])),
    )

    def websocket_for(model: str) -> DummyWebSocket:
        return DummyWebSocket(
            [
                _strict_chat_config(llm={"provider": "stub", "model": model}),
                {"type": "audio", "data": _pcm16_audio([100])},
                {"type": "commit"},
                {"type": "stop"},
            ]
        )

    await asyncio.gather(
        audio.websocket_audio_chat_stream(websocket_for("model-a"), token=None),
        audio.websocket_audio_chat_stream(websocket_for("model-b"), token=None),
    )

    observed = {
        (
            request["model"],
            request["api_key"],
            request["app_config"]["stub"]["snapshot"],
        )
        for request in boundary_requests
    }
    assert observed == {
        ("model-a", "key-model-a", "config-model-a"),
        ("model-b", "key-model-b", "config-model-b"),
    }
    assert all("timeout" not in request for request in boundary_requests)
    assert adapter_timeouts == {
        "model-a": audio_streaming_module.AUDIO_STREAM_FACTORY_TIMEOUT_SECONDS,
        "model-b": audio_streaming_module.AUDIO_STREAM_FACTORY_TIMEOUT_SECONDS,
    }
    assert all(
        is_runtime_issued_provider_call_credentials(
            request[PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY],
            provider="stub",
        )
        for request in boundary_requests
    )
    assert len({id(request[PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY]) for request in boundary_requests}) == 2
    assert {
        id(request[PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY])
        for request in boundary_requests
    } == {
        id(handle)
        for runtime in runtimes
        for handle in runtime.handles
    }
    assert sorted(event for event in lifecycle if event[0] == "mark_used") == [
        ("mark_used", "model-a"),
        ("mark_used", "model-b"),
    ]
    assert sorted(event for event in lifecycle if event[0] == "runtime_close") == [
        ("runtime_close", "model-a"),
        ("runtime_close", "model-b"),
    ]


@pytest.mark.integration
async def test_audio_chat_ws_keeps_empty_runtime_config_frozen_after_live_reload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty turn snapshot must not reload process-global provider config."""
    from tldw_Server_API.app.core.LLM_Calls import chat_calls

    handles_created = asyncio.Event()
    release_resolve = asyncio.Event()
    handles: dict[str, Any] = {}
    requests: list[dict[str, Any]] = []
    live_config = {"generation": "before-resolve"}
    loader_calls: list[str] = []

    class Runtime:
        def __init__(self, **_kwargs: Any) -> None:
            self.inner: RealProviderCredentialRuntime | None = None

        async def resolve(self, provider: str, *, model: str | None = None):
            model_name = str(model)

            async def resolver(
                normalized_provider: str,
                **_resolver_kwargs: Any,
            ) -> ResolvedByokCredentials:
                return ResolvedByokCredentials(
                    provider=normalized_provider,
                    api_key=f"{model_name}-key",
                    app_config=None,
                    credential_fields={},
                    source="user",
                    allowlisted=True,
                    status=ByokResolutionStatus.RESOLVED,
                    auth_source="api_key",
                )

            self.inner = RealProviderCredentialRuntime(
                user_id=1,
                team_ids=(),
                org_ids=(),
                trusted_base_url_override=False,
                server_config_snapshot={},
                resolver=resolver,
            )
            handle = await self.inner.resolve(provider, model=model)
            handles[model_name] = handle
            if len(handles) == 2:
                handles_created.set()
            await release_resolve.wait()
            return handle

        async def mark_used(self, handle: object) -> None:
            assert handle in handles.values()

        async def close(self) -> None:
            if self.inner is not None:
                await self.inner.close()

    class Adapter:
        async def astream(
            self,
            request: dict[str, Any],
            *,
            timeout: float | None = None,
        ) -> AsyncIterator[str]:
            assert "timeout" not in request
            assert timeout == audio_streaming_module.AUDIO_STREAM_FACTORY_TIMEOUT_SECONDS
            requests.append(dict(request))

            async def stream() -> AsyncIterator[str]:
                yield json.dumps(
                    {
                        "choices": [
                            {
                                "delta": {
                                    "content": f"reply-{request['model']}",
                                }
                            }
                        ]
                    }
                )
                yield "data: [DONE]"

            return stream()

    def live_loader() -> dict[str, dict[str, str]]:
        loader_calls.append(live_config["generation"])
        return {"stub_api": dict(live_config)}

    monkeypatch.setattr(audio_streaming_module, "ProviderCredentialRuntime", Runtime)
    monkeypatch.setattr(
        audio_streaming_module,
        "get_registry",
        lambda: SimpleNamespace(get_adapter=lambda _provider: Adapter()),
    )
    monkeypatch.setattr(chat_calls, "load_and_log_configs", live_loader)
    monkeypatch.setattr(
        audio,
        "get_tts_service",
        lambda: asyncio.sleep(0, result=_DummyTTSService([b"tts"])),
    )

    def websocket_for(model: str) -> DummyWebSocket:
        return DummyWebSocket(
            [
                _strict_chat_config(llm={"provider": "stub", "model": model}),
                {"type": "audio", "data": _pcm16_audio([100])},
                {"type": "commit"},
                {"type": "stop"},
            ]
        )

    first = asyncio.create_task(
        audio.websocket_audio_chat_stream(websocket_for("model-a"), token=None)
    )
    second = asyncio.create_task(
        audio.websocket_audio_chat_stream(websocket_for("model-b"), token=None)
    )
    try:
        await asyncio.wait_for(handles_created.wait(), timeout=1.0)
        live_config["generation"] = "after-resolve"
        release_resolve.set()
        await asyncio.wait_for(asyncio.gather(first, second), timeout=2.0)
    finally:
        release_resolve.set()
        await asyncio.gather(first, second, return_exceptions=True)

    assert len(requests) == 2
    assert loader_calls == []
    for request in requests:
        model = request["model"]
        assert request["api_key"] == f"{model}-key"
        assert request["app_config"] == {}
        assert request[PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY] is handles[model]
    assert "after-resolve" not in repr(requests)


@pytest.mark.integration
async def test_audio_chat_ws_emits_bounded_stt_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    import tldw_Server_API.app.core.Metrics.metrics_manager as metrics_manager

    metrics_manager._metrics_registry = None
    registry = metrics_manager.get_metrics_registry()

    audio_payload = _pcm16_audio([_AUDIO_LABEL_TO_SAMPLE["abc"]])
    messages = [
        {
            "type": "config",
            "stt": {"model": "parakeet-ctc-0.6b"},
            "llm": {"provider": "stub", "model": "stub-model"},
            "tts": {"voice": "af_heart", "format": "pcm"},
        },
        {"type": "audio", "data": audio_payload},
        {"type": "commit"},
        {"type": "stop"},
    ]
    ws = DummyWebSocket(messages)

    async def _get_tts_service():
        return _DummyTTSService([b"tts1"])

    monkeypatch.setattr(audio, "get_tts_service", _get_tts_service)

    try:
        await audio.websocket_audio_chat_stream(ws, token=None)

        assert registry.get_cumulative_counter_total("audio_stt_streaming_sessions_started_total") == 1
        assert registry.get_cumulative_counter_totals_by_label(
            "audio_stt_streaming_sessions_started_total",
            "provider",
        ) == {"nemo": 1.0}
        assert registry.get_cumulative_counter_total("audio_stt_streaming_sessions_ended_total") == 1
        assert registry.get_cumulative_counter_total("audio_stt_requests_total") == 1
        assert registry.get_cumulative_counter_totals_by_label("audio_stt_requests_total", "endpoint") == {
            "audio.chat.stream": 1.0
        }
        assert registry.get_cumulative_counter_totals_by_label("audio_stt_requests_total", "provider") == {
            "nemo": 1.0
        }
        assert registry.get_cumulative_counter_totals_by_label("audio_stt_requests_total", "status") == {
            "ok": 1.0
        }
        assert registry.get_cumulative_counter(
            "audio_stt_redaction_total",
            {"endpoint": "audio.chat.stream", "redaction_outcome": "not_requested"},
        ) >= 1
        metrics_text = registry.export_prometheus_format()
        assert 'stt_final_latency_seconds_count{endpoint="audio.chat.stream",model="parakeet"' in metrics_text
        assert "parakeet-ctc-0.6b" not in metrics_text
    finally:
        metrics_manager._metrics_registry = None


@pytest.mark.integration
async def test_audio_chat_ws_counts_final_frames_in_redaction_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tldw_Server_API.app.core.Metrics.metrics_manager as metrics_manager

    metrics_manager._metrics_registry = None
    registry = metrics_manager.get_metrics_registry()

    class _FinalFrameTranscriber(_DummyTranscriber):
        async def process_audio_chunk(self, audio_bytes: bytes) -> Dict[str, Any]:  # noqa: ARG002
            return {"type": "final", "text": "final chunk", "is_final": True}

        def get_full_transcript(self) -> str:
            return "final chunk"

    ws = DummyWebSocket(
        [
            _strict_chat_config(),
            {"type": "audio", "data": _pcm16_audio([_AUDIO_LABEL_TO_SAMPLE["abc"]])},
            {"type": "commit"},
            {"type": "stop"},
        ]
    )

    async def _get_tts_service():
        return _DummyTTSService([b"tts1"])

    monkeypatch.setattr(audio, "UnifiedStreamingTranscriber", _FinalFrameTranscriber)
    monkeypatch.setattr(audio, "get_tts_service", _get_tts_service)

    try:
        await audio.websocket_audio_chat_stream(ws, token=None)

        assert registry.get_cumulative_counter(
            "audio_stt_redaction_total",
            {"endpoint": "audio.chat.stream", "redaction_outcome": "not_requested"},
        ) >= 2
    finally:
        metrics_manager._metrics_registry = None


@pytest.mark.integration
async def test_audio_chat_ws_normalizes_whisper_alias_before_transcriber_init(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ws = DummyWebSocket(
        [
            {
                "type": "config",
                "stt": {"model": "whisper-1"},
                "llm": {"provider": "stub", "model": "stub-model"},
                "tts": {"voice": "af_heart", "format": "pcm"},
            },
            {"type": "stop"},
        ]
    )
    captured: Dict[str, Any] = {}

    class _CapturingTranscriber(_DummyTranscriber):
        def __init__(self, config: Any) -> None:
            captured["model"] = getattr(config, "model", None)
            captured["variant"] = getattr(config, "model_variant", None)
            captured["whisper_model_size"] = getattr(config, "whisper_model_size", None)
            super().__init__(config)

    monkeypatch.setattr(audio, "UnifiedStreamingTranscriber", _CapturingTranscriber)

    await audio.websocket_audio_chat_stream(ws, token=None)

    assert captured["model"] == "whisper"
    assert captured["whisper_model_size"] == _map_openai_audio_model_to_whisper("whisper-1")


@pytest.mark.integration
async def test_audio_chat_ws_normalizes_parakeet_variant_before_transcriber_init(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ws = DummyWebSocket(
        [
            {
                "type": "config",
                "stt": {"model": "parakeet-onnx"},
                "llm": {"provider": "stub", "model": "stub-model"},
                "tts": {"voice": "af_heart", "format": "pcm"},
            },
            {"type": "stop"},
        ]
    )
    captured: Dict[str, Any] = {}

    class _CapturingTranscriber(_DummyTranscriber):
        def __init__(self, config: Any) -> None:
            captured["model"] = getattr(config, "model", None)
            captured["variant"] = getattr(config, "model_variant", None)
            super().__init__(config)

    monkeypatch.setattr(audio, "UnifiedStreamingTranscriber", _CapturingTranscriber)

    await audio.websocket_audio_chat_stream(ws, token=None)

    assert captured["model"] == "parakeet"
    assert captured["variant"] == "onnx"


@pytest.mark.integration
async def test_audio_chat_ws_rejects_protocol_version_2() -> None:
    ws = DummyWebSocket(
        [
            {
                "type": "config",
                "protocol_version": 2,
                "mode": "voice_chat",
                "audio_format": "pcm16",
                "sample_rate": 16000,
                "channels": 1,
                "stt": {"model": "parakeet"},
                "llm": {"provider": "stub", "model": "stub-model"},
                "tts": {"voice": "af_heart", "format": "pcm"},
            }
        ]
    )

    await audio.websocket_audio_chat_stream(ws, token=None)

    assert any(
        msg.get("type") == "error"
        and msg.get("code") == "bad_request"
        and "protocol_version" in msg.get("message", "")
        for msg in ws.sent_json
    )
    assert not [msg for msg in ws.sent_json if msg.get("protocol_version") == 2]
    assert ws.close_code == 4400


@pytest.mark.integration
async def test_audio_chat_ws_rejects_protocol_version_2_when_legacy_control_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_chat_ws_control_v2(monkeypatch)
    ws = DummyWebSocket(
        [
            {
                "type": "config",
                "protocol_version": 2,
                "mode": "voice_chat",
                "audio_format": "pcm16",
                "sample_rate": 16000,
                "channels": 1,
                "stt": {"model": "parakeet"},
                "llm": {"provider": "stub", "model": "stub-model"},
                "tts": {"voice": "af_heart", "format": "pcm"},
            },
            {"type": "control", "action": "stop"},
        ]
    )

    await audio.websocket_audio_chat_stream(ws, token=None)

    assert any(
        msg.get("type") == "error"
        and msg.get("code") == "bad_request"
        and "protocol_version" in msg.get("message", "")
        for msg in ws.sent_json
    )
    assert not [msg for msg in ws.sent_json if msg.get("type") == "status"]
    assert ws.close_code == 4400


@pytest.mark.integration
async def test_audio_chat_ws_closes_on_malformed_json_after_strict_config() -> None:
    ws = DummyWebSocket(
        [
            _strict_chat_config(),
            "{not valid json",
        ]
    )

    await audio.websocket_audio_chat_stream(ws, token=None)

    assert any(
        msg.get("type") == "error"
        and msg.get("code") == "validation_error"
        and msg.get("message") == "Invalid JSON message"
        for msg in ws.sent_json
    )
    assert ws.close_code == 4400


@pytest.mark.integration
async def test_audio_chat_ws_records_failure_metrics_for_invalid_audio_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_statuses: list[str] = []
    session_close_reasons: list[str] = []
    monkeypatch.setattr(
        audio_streaming_module,
        "emit_stt_request_total",
        lambda **kwargs: request_statuses.append(kwargs["status"]),
    )
    monkeypatch.setattr(
        audio_streaming_module,
        "emit_stt_session_end_total",
        lambda **kwargs: session_close_reasons.append(kwargs["session_close_reason"]),
    )
    ws = DummyWebSocket(
        [
            _strict_chat_config(),
            {"type": "audio", "data": "not base64 ***"},
        ]
    )

    await audio.websocket_audio_chat_stream(ws, token=None)

    assert any(msg.get("type") == "error" and msg.get("code") == "bad_request" for msg in ws.sent_json)
    assert ws.close_code == 4400
    assert request_statuses[-1] == "bad_request"
    assert session_close_reasons[-1] == "error"


@pytest.mark.integration
async def test_audio_chat_ws_records_failure_metrics_for_push_to_talk_violation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_statuses: list[str] = []
    session_close_reasons: list[str] = []
    monkeypatch.setattr(
        audio_streaming_module,
        "emit_stt_request_total",
        lambda **kwargs: request_statuses.append(kwargs["status"]),
    )
    monkeypatch.setattr(
        audio_streaming_module,
        "emit_stt_session_end_total",
        lambda **kwargs: session_close_reasons.append(kwargs["session_close_reason"]),
    )
    ws = DummyWebSocket(
        [
            _strict_chat_config(mode="voice_chat"),
            {"type": "push_to_talk_release"},
        ]
    )

    await audio.websocket_audio_chat_stream(ws, token=None)

    assert any(
        msg.get("type") == "error" and "only valid in push_to_talk mode" in msg.get("message", "")
        for msg in ws.sent_json
    )
    assert ws.close_code == 4400
    assert request_statuses[-1] == "bad_request"
    assert session_close_reasons[-1] == "error"


@pytest.mark.integration
async def test_audio_chat_ws_interrupt_without_active_turn_is_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    ws = DummyWebSocket(
        [
            {
                "type": "config",
                "stt": {"model": "parakeet"},
                "llm": {"provider": "stub", "model": "stub-model"},
                "tts": {"voice": "af_heart", "format": "pcm"},
            },
            {"type": "interrupt", "reason": "user_stop"},
            {"type": "stop"},
        ]
    )

    async def _get_tts_service():
        return _DummyTTSService([b"tts"])

    monkeypatch.setattr(audio, "get_tts_service", _get_tts_service)

    await audio.websocket_audio_chat_stream(ws, token=None)

    assert any(msg.get("type") == "interrupted" for msg in ws.sent_json)


@pytest.mark.integration
async def test_audio_chat_ws_overlap_starts_tts_before_final_llm_message(monkeypatch: pytest.MonkeyPatch) -> None:
    audio_payload = _pcm16_audio([_AUDIO_LABEL_TO_SAMPLE["abc"]])
    ws = DummyWebSocket(
        [
            {
                "type": "config",
                "stt": {"model": "parakeet"},
                "llm": {"provider": "stub", "model": "stub-model"},
                "tts": {"voice": "af_heart", "format": "pcm"},
            },
            {"type": "audio", "data": audio_payload},
            {"type": "commit"},
            {"type": "stop"},
        ]
    )

    async def _overlap_llm_stub(**kwargs: Any) -> AsyncIterator[str]:  # noqa: ARG002
        async def _gen() -> AsyncIterator[str]:
            yield 'data: {"choices":[{"delta":{"content":"Hello world. "}}]}\n\n'
            await asyncio.sleep(0)
            yield 'data: {"choices":[{"delta":{"content":"How are you?"}}]}\n\n'
            await asyncio.sleep(0)
            yield 'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
            yield "data: [DONE]\n\n"

        return _gen()

    realtime_service = _DummyRealtimeCapableTTSService()

    async def _get_tts_service():
        return realtime_service

    monkeypatch.setattr(audio, "chat_api_call_async", _overlap_llm_stub)
    monkeypatch.setattr(audio, "get_tts_service", _get_tts_service)

    await audio.websocket_audio_chat_stream(ws, token=None)

    llm_message_idx = next(
        i for i, event in enumerate(ws.sent_events)
        if event[0] == "json" and event[1].get("type") == "llm_message"
    )
    tts_start_idx = next(
        i for i, event in enumerate(ws.sent_events)
        if event[0] == "json" and event[1].get("type") == "tts_start"
    )
    first_audio_idx = next(i for i, event in enumerate(ws.sent_events) if event[0] == "bytes")

    assert tts_start_idx < llm_message_idx
    assert first_audio_idx < llm_message_idx
    assert ws.sent_bytes


@pytest.mark.integration
async def test_audio_chat_ws_overlap_warning_sanitizes_internal_message(monkeypatch: pytest.MonkeyPatch) -> None:
    audio_payload = _pcm16_audio([_AUDIO_LABEL_TO_SAMPLE["abc"]])
    ws = DummyWebSocket(
        [
            {
                "type": "config",
                "stt": {"model": "parakeet"},
                "llm": {"provider": "stub", "model": "stub-model"},
                "tts": {"voice": "af_heart", "format": "pcm"},
            },
            {"type": "audio", "data": audio_payload},
            {"type": "commit"},
            {"type": "stop"},
        ]
    )

    class _WarningRealtimeService(_DummyRealtimeCapableTTSService):
        async def open_realtime_session(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG002
            return SimpleNamespace(
                session=self.session,
                provider="stub-realtime",
                warning=RuntimeError("tts provider degraded at /private/tts/cache"),
            )

    async def _get_tts_service():
        return _WarningRealtimeService()

    monkeypatch.setattr(audio, "get_tts_service", _get_tts_service)

    await audio.websocket_audio_chat_stream(ws, token=None)

    warnings = [message for message in ws.sent_json if message.get("type") == "warning"]
    assert warnings
    assert warnings[-1]["message"] == "Realtime TTS session warning"
    assert "tts provider degraded" not in str(warnings)
    assert "/private/tts/cache" not in str(warnings)


@pytest.mark.integration
async def test_audio_chat_ws_interrupt_cancels_inflight_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    audio_payload = _pcm16_audio([_AUDIO_LABEL_TO_SAMPLE["abc"]])
    ws = DummyWebSocket(
        [
            {
                "type": "config",
                "stt": {"model": "parakeet"},
                "llm": {"provider": "stub", "model": "stub-model"},
                "tts": {"voice": "af_heart", "format": "pcm"},
            },
            {"type": "audio", "data": audio_payload},
            {"type": "commit"},
            {"type": "interrupt", "reason": "barge_in"},
            {"type": "stop"},
        ]
    )

    async def _slow_llm_stub(**kwargs: Any) -> AsyncIterator[str]:  # noqa: ARG002
        async def _gen() -> AsyncIterator[str]:
            await asyncio.sleep(0.05)
            yield 'data: {"choices":[{"delta":{"content":"This should be cancelled."}}]}\n\n'
            await asyncio.sleep(0.05)
            yield 'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
            yield "data: [DONE]\n\n"

        return _gen()

    realtime_service = _DummyRealtimeCapableTTSService()

    async def _get_tts_service():
        return realtime_service

    monkeypatch.setattr(audio, "chat_api_call_async", _slow_llm_stub)
    monkeypatch.setattr(audio, "get_tts_service", _get_tts_service)

    await audio.websocket_audio_chat_stream(ws, token=None)

    interrupted = [msg for msg in ws.sent_json if msg.get("type") == "interrupted"]
    assert interrupted
    assert interrupted[-1].get("turn_id")


@pytest.mark.integration
async def test_audio_chat_ws_drops_stale_audio_after_interrupt(monkeypatch: pytest.MonkeyPatch) -> None:
    audio_payload = _pcm16_audio([_AUDIO_LABEL_TO_SAMPLE["abc"]])
    ws = DummyWebSocket(
        [
            {
                "type": "config",
                "stt": {"model": "parakeet"},
                "llm": {"provider": "stub", "model": "stub-model"},
                "tts": {"voice": "af_heart", "format": "pcm"},
            },
            {"type": "audio", "data": audio_payload},
            {"type": "commit"},
            {"type": "interrupt", "reason": "barge_in"},
            {"type": "stop"},
        ]
    )

    class _DelayedRealtimeSession(_DummyRealtimeSession):
        async def commit(self) -> None:
            if self._closed:
                return
            text = self._buffer.strip()
            self._buffer = ""
            if text:
                await asyncio.sleep(0.05)
                await self._queue.put(f"rt:{text}".encode("utf-8"))

    class _DelayedRealtimeService(_DummyRealtimeCapableTTSService):
        def __init__(self) -> None:
            self.session = _DelayedRealtimeSession()

    async def _slow_llm_stub(**kwargs: Any) -> AsyncIterator[str]:  # noqa: ARG002
        async def _gen() -> AsyncIterator[str]:
            yield 'data: {"choices":[{"delta":{"content":"Chunk one. "}}]}\n\n'
            await asyncio.sleep(0.05)
            yield 'data: {"choices":[{"delta":{"content":"Chunk two."}}]}\n\n'
            yield 'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
            yield "data: [DONE]\n\n"

        return _gen()

    async def _get_tts_service():
        return _DelayedRealtimeService()

    monkeypatch.setattr(audio, "chat_api_call_async", _slow_llm_stub)
    monkeypatch.setattr(audio, "get_tts_service", _get_tts_service)

    await audio.websocket_audio_chat_stream(ws, token=None)

    interrupted_idx = next(
        i for i, event in enumerate(ws.sent_events)
        if event[0] == "json" and event[1].get("type") == "interrupted"
    )
    stale_bytes_after_interrupt = [
        event for event in ws.sent_events[interrupted_idx + 1:]
        if event[0] == "bytes"
    ]
    assert stale_bytes_after_interrupt == []


@pytest.mark.integration
async def test_audio_chat_ws_auto_commit_uses_eos_timestamp(monkeypatch: pytest.MonkeyPatch) -> None:
    audio_payload = _pcm16_audio([_AUDIO_LABEL_TO_SAMPLE["abc"]])
    messages = [
        {
            "type": "config",
            "stt": {"model": "parakeet"},
            "llm": {"provider": "stub", "model": "stub-model"},
            "tts": {"voice": "af_heart", "format": "pcm"},
        },
        {"type": "audio", "data": audio_payload},
        {"type": "audio", "data": audio_payload},
        {"type": "stop"},
    ]
    ws = DummyWebSocket(messages)

    class _TriggeringVAD(_DummyVAD):
        def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
            super().__init__(*args, **kwargs)
            self._count = 0
            self.last_trigger_at = None

        def observe(self, audio_bytes: bytes) -> bool:  # noqa: ARG002
            self._count += 1
            if self._count >= 2:
                self.last_trigger_at = 4321.25
                return True
            return False

    async def _get_tts_service():
        return _DummyTTSService([b"tts"])

    monkeypatch.setattr(audio, "SileroTurnDetector", _TriggeringVAD)
    monkeypatch.setattr(audio, "get_tts_service", _get_tts_service)

    await audio.websocket_audio_chat_stream(ws, token=None)

    full_transcripts = [msg for msg in ws.sent_json if msg.get("type") == "full_transcript"]
    assert full_transcripts
    assert full_transcripts[0].get("auto_commit") is True
    assert full_transcripts[0].get("voice_to_voice_start") == pytest.approx(4321.25)


@pytest.mark.integration
async def test_audio_chat_ws_persists_turn_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    audio_payload = _pcm16_audio([_AUDIO_LABEL_TO_SAMPLE["abc"]])
    ws = DummyWebSocket(
        [
            {
                "type": "config",
                "stt": {"model": "parakeet"},
                "llm": {"provider": "stub", "model": "stub-model", "extra_params": {"action": "demo_tool"}},
                "tts": {"voice": "af_heart", "format": "pcm"},
                "metadata": {"persist_history": True},
            },
            {"type": "audio", "data": audio_payload},
            {"type": "commit"},
            {"type": "stop"},
        ]
    )

    class _DummyChatDB:
        def __init__(self) -> None:
            self.messages: List[Dict[str, Any]] = []
            self.settings: List[tuple[str, Dict[str, Any]]] = []

        def add_message(self, msg_data: Dict[str, Any]) -> str:
            self.messages.append(dict(msg_data))
            return "msg-id"

        def upsert_conversation_settings(self, conversation_id: str, settings: Dict[str, Any]) -> bool:
            self.settings.append((conversation_id, settings))
            return True

    persisted_db = _DummyChatDB()

    async def _get_tts_service():
        return _DummyTTSService([b"tts"])

    async def _get_db_for_user_id(_user_id: int, client_id: Optional[str] = None):  # noqa: ARG001
        return persisted_db

    async def _character_context(_db: Any, _character_id: Any, _loop: Any):
        return {"id": 42, "name": "Helpful AI Assistant"}, 42

    async def _conversation_context(
        _db: Any,
        _conversation_id: Optional[str],
        _character_id: int,
        _character_name: str,
        _client_id: str,
        _loop: Any,
    ):
        return "ws-session-001", True

    async def _execute_action(_action: str, _transcript: str, _user: Any) -> Dict[str, Any]:
        return {"action": "demo_tool", "status": "ok", "payload": {"value": 1}}

    monkeypatch.setattr(audio, "get_tts_service", _get_tts_service)
    monkeypatch.setattr(audio, "get_chacha_db_for_user_id", _get_db_for_user_id, raising=False)
    monkeypatch.setattr(audio, "get_or_create_character_context", _character_context, raising=False)
    monkeypatch.setattr(audio, "get_or_create_conversation", _conversation_context, raising=False)
    monkeypatch.setattr(audio_streaming_module.speech_chat_service, "_actions_enabled", lambda: True)
    monkeypatch.setattr(audio_streaming_module.speech_chat_service, "_execute_action", _execute_action)

    await audio.websocket_audio_chat_stream(ws, token=None)

    assert any(msg.get("type") == "session" and msg.get("session_id") == "ws-session-001" for msg in ws.sent_json)
    assert [m.get("sender") for m in persisted_db.messages] == ["user", "assistant", "tool"]
    assert all(m.get("conversation_id") == "ws-session-001" for m in persisted_db.messages)
    assert persisted_db.settings
    _, settings = persisted_db.settings[0]
    assert settings.get("audio_chat_ws", {}).get("action_hint") == "demo_tool"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_audio_chat_ws_existing_session_merges_settings_with_version_cas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_payload = _pcm16_audio([_AUDIO_LABEL_TO_SAMPLE["abc"]])
    ws = DummyWebSocket(
        [
            {
                "type": "config",
                "session_id": "resume-chat",
                "stt": {"model": "parakeet"},
                "llm": {"provider": "stub", "model": "stub-model"},
                "tts": {"voice": "af_heart", "format": "pcm"},
                "metadata": {"persist_history": True},
            },
            {"type": "audio", "data": audio_payload},
            {"type": "commit"},
            {"type": "stop"},
        ]
    )

    existing_settings, existing_resume_state = _valid_roleplay_settings_with_blob(16)
    existing_settings["userSetting"] = "preserve-me"
    existing_resume_state["settings"] = existing_settings

    class _ExistingChatDB:
        client_id = "1"

        def __init__(self) -> None:
            self.messages: list[dict[str, Any]] = []
            self.current = {
                "settings": existing_settings,
                "settings_version": 7,
            }
            self.upsert_calls: list[tuple[dict[str, Any], int | None]] = []

        def add_message(self, msg_data: dict[str, Any]) -> str:
            self.messages.append(dict(msg_data))
            return "msg-id"

        def get_conversation_settings(self, conversation_id: str) -> dict[str, Any]:
            assert conversation_id == "resume-chat"
            return {
                "settings": dict(self.current["settings"]),
                "settings_version": self.current["settings_version"],
            }

        def transaction(self):
            return nullcontext(object())

        def get_roleplay_resume_state(
            self,
            conversation_id: str,
            **_kwargs: Any,
        ) -> dict[str, Any]:
            assert conversation_id == "resume-chat"
            return {
                **existing_resume_state,
                "conversation": _owned_audio_test_conversation(conversation_id),
            }

        def get_conversation_by_id(self, conversation_id: str) -> dict[str, Any]:
            return _owned_audio_test_conversation(conversation_id)

        def upsert_conversation_settings(
            self,
            conversation_id: str,
            settings: dict[str, Any],
            *,
            conn=None,
            expected_settings_version: int | None = None,
        ) -> bool:
            assert conversation_id == "resume-chat"
            self.upsert_calls.append((dict(settings), expected_settings_version))
            return True

    persisted_db = _ExistingChatDB()

    async def _get_tts_service():
        return _DummyTTSService([b"tts"])

    async def _get_db_for_user_id(_user_id: int, client_id: Optional[str] = None):  # noqa: ARG001
        return persisted_db

    async def _character_context(_db: Any, _character_id: Any, _loop: Any):
        return {"id": 42, "name": "Helpful AI Assistant"}, 42

    async def _conversation_context(*_args: Any, **_kwargs: Any):
        return "resume-chat", False

    monkeypatch.setattr(audio, "get_tts_service", _get_tts_service)
    monkeypatch.setattr(audio, "get_chacha_db_for_user_id", _get_db_for_user_id, raising=False)
    monkeypatch.setattr(audio, "get_or_create_character_context", _character_context, raising=False)
    monkeypatch.setattr(audio, "get_or_create_conversation", _conversation_context, raising=False)

    await audio.websocket_audio_chat_stream(ws, token=None)

    assert len(persisted_db.upsert_calls) == 1
    merged, expected_version = persisted_db.upsert_calls[0]
    assert expected_version == 7
    assert merged["userSetting"] == "preserve-me"
    assert merged["roleplayResumeV1"] == existing_settings["roleplayResumeV1"]
    assert merged["roleplayBehaviorV1"] == existing_settings["roleplayBehaviorV1"]
    assert merged["audio_chat_ws"]["session_id"] == "resume-chat"


def _nested_audio_metadata(depth: int) -> dict[str, Any]:
    value: dict[str, Any] = {"leaf": "value"}
    for _index in range(depth):
        value = {"nested": value}
    return value


def _owned_audio_test_conversation(conversation_id: str) -> dict[str, Any]:
    return {
        "id": conversation_id,
        "character_id": 42,
        "client_id": "1",
        "deleted": 0,
    }


def _valid_roleplay_settings_with_blob(blob_size: int) -> tuple[dict[str, Any], dict[str, Any]]:
    snapshot_digest = "sha256:" + ("a" * 64)
    effective = {
        "provider": "local-llm",
        "model": "local-test",
        "sampling": {
            "temperature": 0.7,
            "top_p": 1.0,
            "repetition_penalty": 1.0,
            "stop": [],
        },
    }
    behavior = build_materialized_behavior_settings(
        {
            "base_snapshot": {"schema_version": 1, "digest": snapshot_digest},
            "behavior_controls": build_materialized_behavior_controls({}),
            "effective_completion": effective,
            "participants": [{"frozen_prompt": "x" * blob_size}],
        }
    )
    settings = {
        "userSetting": "preserve-me",
        "roleplayResumeV1": {
            "resumeEligible": True,
            "resumeIneligibleReason": None,
            "effectiveCompletion": effective,
        },
        "roleplayBehaviorV1": behavior,
    }
    resume_state = {
        "settings": settings,
        "settings_version": 7,
        "behavior_snapshot": {
            "status": "valid",
            "schema_version": 1,
            "digest": snapshot_digest,
        },
        "materialized_settings": {
            "schema_version": 1,
            "digest": behavior["digest"],
            "values": behavior["values"],
        },
        "resume_eligible": True,
        "resume_ineligible_reason": None,
        "effective_completion": effective,
    }
    return settings, resume_state


@pytest.mark.unit
def test_audio_writer_rejects_pending_greeting_for_different_character() -> None:
    snapshot_digest = "sha256:" + ("a" * 64)
    pending = build_pending_greeting_record(
        {
            "base_snapshot": {
                "schema_version": 1,
                "digest": snapshot_digest,
            },
            "character_id": 99,
            "greetings_checksum": "sha256:frozen-greetings",
            "greeting": {
                "content": "Frozen alternate greeting",
                "selection_id": "greeting:1:selected",
                "source": "alternate_greeting",
                "source_index": 1,
                "character_version": 1,
            },
        }
    )
    settings = {
        "greetingSelectionId": "greeting:1:selected",
        "greetingsChecksum": "sha256:frozen-greetings",
        "roleplayResumeV1": {
            "resumeEligible": False,
            "resumeIneligibleReason": "incomplete_effective_settings",
            "effectiveCompletion": None,
        },
        "roleplayPendingGreetingV1": pending,
    }
    resume_state = {
        "settings": settings,
        "settings_version": 7,
        "history_version": 11,
        "behavior_snapshot": {
            "status": "valid",
            "schema_version": 1,
            "digest": snapshot_digest,
        },
    }

    class _SettingsDB:
        client_id = "1"

        def __init__(self) -> None:
            self.current = dict(resume_state)
            self.upsert_calls: list[dict[str, Any]] = []

        def transaction(self):
            return nullcontext(object())

        def get_roleplay_resume_state(
            self,
            _conversation_id: str,
            *,
            conn,
            lock_for_update: bool,
            owner_client_id: str | None = None,
        ) -> dict[str, Any]:
            assert conn is not None
            assert lock_for_update is True
            assert owner_client_id == "1"
            return {
                **self.current,
                "conversation": _owned_audio_test_conversation("pending-audio"),
            }

        def get_conversation_by_id(self, _conversation_id: str) -> dict[str, Any]:
            return _owned_audio_test_conversation("pending-audio")

        def upsert_conversation_settings(
            self,
            _conversation_id: str,
            updated: dict[str, Any],
            *,
            conn=None,
            expected_settings_version: int | None = None,  # noqa: ARG002
        ) -> bool:
            self.upsert_calls.append(dict(updated))
            self.current["settings"] = dict(updated)
            self.current["settings_version"] += 1
            return True

    db = _SettingsDB()
    before = dict(db.current)

    with pytest.raises(InputError, match="pending greeting"):
        audio_streaming_module._persist_audio_chat_settings(
            db,
            "pending-audio",
            {"audio_chat_ws": {"metadata": {"mode": "tiny"}}},
            conversation_created=False,
        )

    assert db.upsert_calls == []
    assert db.current == before


@pytest.mark.unit
def test_audio_writer_uses_transactional_conversation_identity(monkeypatch) -> None:
    stale_conversation = _owned_audio_test_conversation("racing-audio")
    current_conversation = {
        **stale_conversation,
        "version": 2,
        "character_id": 2,
        "assistant_id": "2",
    }

    class _SettingsDB:
        client_id = "1"

        def __init__(self) -> None:
            self.owner_client_id: str | None = None

        def get_conversation_by_id(self, _conversation_id: str) -> dict[str, Any]:
            return stale_conversation

        def transaction(self):
            return nullcontext(object())

        def get_roleplay_resume_state(
            self,
            _conversation_id: str,
            *,
            conn,
            lock_for_update: bool,
            owner_client_id: str | None = None,
        ) -> dict[str, Any]:
            assert conn is not None
            assert lock_for_update is True
            self.owner_client_id = owner_client_id
            return {
                "conversation": current_conversation,
                "settings": {"preserved": True},
                "settings_version": 4,
                "behavior_snapshot": {"status": "missing"},
            }

        def upsert_conversation_settings(
            self,
            _conversation_id: str,
            _settings: dict[str, Any],
            *,
            conn,
            expected_settings_version: int,
        ) -> bool:
            assert conn is not None
            assert expected_settings_version == 4
            return True

    seen: dict[str, object] = {}

    def _capture_validation(settings, **kwargs):
        seen["conversation"] = kwargs.get("conversation")
        return settings

    monkeypatch.setattr(
        audio_streaming_module,
        "validate_chat_settings_storage",
        _capture_validation,
    )
    db = _SettingsDB()

    assert audio_streaming_module._persist_audio_chat_settings(
        db,
        "racing-audio",
        {"audio_chat_ws": {"metadata": {"mode": "tiny"}}},
        conversation_created=False,
    )

    assert seen["conversation"] == current_conversation
    assert db.owner_client_id == "1"


@pytest.mark.unit
def test_audio_writer_allows_small_public_update_with_large_valid_internal_authority() -> None:
    settings, resume_state = _valid_roleplay_settings_with_blob(210_000)

    class _SettingsDB:
        client_id = "1"

        def __init__(self) -> None:
            self.current = dict(settings)
            self.upsert_calls: list[tuple[dict[str, Any], int | None]] = []

        def get_conversation_settings(self, _conversation_id: str) -> dict[str, Any]:
            return {"settings": dict(self.current), "settings_version": 7}

        def transaction(self):
            return nullcontext(object())

        def get_roleplay_resume_state(
            self,
            conversation_id: str,
            **_kwargs: Any,
        ) -> dict[str, Any]:
            return {
                **resume_state,
                "conversation": _owned_audio_test_conversation(conversation_id),
            }

        def get_conversation_by_id(self, conversation_id: str) -> dict[str, Any]:
            return _owned_audio_test_conversation(conversation_id)

        def upsert_conversation_settings(
            self,
            _conversation_id: str,
            updated: dict[str, Any],
            *,
            conn=None,
            expected_settings_version: int | None = None,
        ) -> bool:
            self.upsert_calls.append((dict(updated), expected_settings_version))
            return True

    db = _SettingsDB()
    updated = audio_streaming_module._persist_audio_chat_settings(
        db,
        "large-resumable-audio",
        {"audio_chat_ws": {"metadata": {"mode": "tiny"}}},
        conversation_created=False,
    )

    assert updated is True
    assert len(db.upsert_calls) == 1
    persisted, expected_version = db.upsert_calls[0]
    assert expected_version == 7
    assert persisted["roleplayBehaviorV1"] == settings["roleplayBehaviorV1"]
    assert persisted["audio_chat_ws"]["metadata"] == {"mode": "tiny"}


@pytest.mark.unit
@pytest.mark.parametrize("internal_mutation", ["digest", "oversize"])
def test_audio_writer_rejects_invalid_preserved_internal_authority(
    internal_mutation: str,
) -> None:
    settings, resume_state = _valid_roleplay_settings_with_blob(16)
    behavior = dict(settings["roleplayBehaviorV1"])
    if internal_mutation == "digest":
        behavior["digest"] = "sha256:" + ("0" * 64)
    else:
        values = dict(behavior["values"])
        values["participants"] = [{"oversize": "x" * (1024 * 1024 + 1)}]
        behavior["values"] = values
    settings["roleplayBehaviorV1"] = behavior
    resume_state["settings"] = settings

    class _SettingsDB:
        client_id = "1"

        def __init__(self) -> None:
            self.upsert_calls: list[dict[str, Any]] = []

        def get_conversation_settings(self, _conversation_id: str) -> dict[str, Any]:
            return {"settings": dict(settings), "settings_version": 7}

        def transaction(self):
            return nullcontext(object())

        def get_roleplay_resume_state(
            self,
            conversation_id: str,
            **_kwargs: Any,
        ) -> dict[str, Any]:
            return {
                **resume_state,
                "conversation": _owned_audio_test_conversation(conversation_id),
            }

        def get_conversation_by_id(self, conversation_id: str) -> dict[str, Any]:
            return _owned_audio_test_conversation(conversation_id)

        def upsert_conversation_settings(
            self,
            _conversation_id: str,
            updated: dict[str, Any],
            *,
            conn=None,
            expected_settings_version: int | None = None,  # noqa: ARG002
        ) -> bool:
            self.upsert_calls.append(dict(updated))
            return True

    db = _SettingsDB()
    with pytest.raises(InputError, match="materialized|roleplay"):
        audio_streaming_module._persist_audio_chat_settings(
            db,
            "invalid-resumable-audio",
            {"audio_chat_ws": {"metadata": {"mode": "tiny"}}},
            conversation_created=False,
        )
    assert db.upsert_calls == []


@pytest.mark.unit
def test_audio_writer_rejects_caller_replacement_of_internal_authority() -> None:
    class _SettingsDB:
        def get_conversation_settings(self, _conversation_id: str) -> dict[str, Any]:
            return {"settings": {"userSetting": "preserve"}, "settings_version": 2}

        def upsert_conversation_settings(self, *_args: Any, **_kwargs: Any) -> bool:
            raise AssertionError("caller-controlled internal state must not reach upsert")

    with pytest.raises(InputError, match="reserved|server-owned"):
        audio_streaming_module._persist_audio_chat_settings(
            _SettingsDB(),
            "caller-internal-audio",
            {"roleplayBehaviorV1": {"caller": "replacement"}},
            conversation_created=False,
        )


@pytest.mark.unit
@pytest.mark.parametrize("conversation_created", [True, False], ids=["create", "merge"])
@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        ({"payload": "x" * 200_001}, "bytes"),
        (_nested_audio_metadata(40), "depth"),
        ({"score": float("nan")}, "finite"),
    ],
    ids=["oversize", "depth", "nonfinite"],
)
def test_audio_settings_writer_rejects_invalid_final_object_before_upsert(
    conversation_created: bool,
    metadata: dict[str, Any],
    message: str,
) -> None:
    class _SettingsDB:
        def __init__(self) -> None:
            self.current = {
                "settings": {"userSetting": "unchanged"},
                "settings_version": 4,
            }
            self.upsert_calls: list[tuple[dict[str, Any], int | None]] = []

        def get_conversation_settings(self, _conversation_id: str) -> dict[str, Any]:
            return {
                "settings": dict(self.current["settings"]),
                "settings_version": self.current["settings_version"],
            }

        def get_conversation_by_id(self, conversation_id: str) -> dict[str, Any]:
            return _owned_audio_test_conversation(conversation_id)

        def upsert_conversation_settings(
            self,
            _conversation_id: str,
            settings: dict[str, Any],
            *,
            expected_settings_version: int | None = None,
        ) -> bool:
            self.upsert_calls.append((dict(settings), expected_settings_version))
            self.current = {
                "settings": dict(settings),
                "settings_version": self.current["settings_version"] + 1,
            }
            return True

    db = _SettingsDB()
    before = dict(db.current)
    settings_payload = {
        "audio_chat_ws": {
            "session_id": "audio-validation",
            "metadata": metadata,
        }
    }

    with pytest.raises(InputError, match=message):
        audio_streaming_module._persist_audio_chat_settings(
            db,
            "audio-validation",
            settings_payload,
            conversation_created=conversation_created,
        )

    assert db.upsert_calls == []
    assert db.current == before


@pytest.mark.integration
@pytest.mark.asyncio
async def test_audio_chat_ws_existing_session_conflict_does_not_blind_replace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_payload = _pcm16_audio([_AUDIO_LABEL_TO_SAMPLE["abc"]])
    ws = DummyWebSocket(
        [
            {
                "type": "config",
                "session_id": "resume-chat-conflict",
                "stt": {"model": "parakeet"},
                "llm": {"provider": "stub", "model": "stub-model"},
                "tts": {"voice": "af_heart", "format": "pcm"},
                "metadata": {"persist_history": True},
            },
            {"type": "audio", "data": audio_payload},
            {"type": "commit"},
            {"type": "stop"},
        ]
    )

    existing_settings, existing_resume_state = _valid_roleplay_settings_with_blob(16)
    existing_settings["userSetting"] = "winner"
    existing_resume_state["settings"] = existing_settings
    existing_resume_state["settings_version"] = 9

    class _ConflictingChatDB:
        client_id = "1"

        def __init__(self) -> None:
            self.blind_writes = 0
            self.cas_versions: list[int | None] = []

        def add_message(self, _msg_data: dict[str, Any]) -> str:
            return "msg-id"

        def get_conversation_settings(self, _conversation_id: str) -> dict[str, Any]:
            return {
                "settings": existing_settings,
                "settings_version": 9,
            }

        def transaction(self):
            return nullcontext(object())

        def get_roleplay_resume_state(
            self,
            conversation_id: str,
            **_kwargs: Any,
        ) -> dict[str, Any]:
            return {
                **existing_resume_state,
                "conversation": _owned_audio_test_conversation(conversation_id),
            }

        def get_conversation_by_id(self, conversation_id: str) -> dict[str, Any]:
            return _owned_audio_test_conversation(conversation_id)

        def upsert_conversation_settings(
            self,
            _conversation_id: str,
            _settings: dict[str, Any],
            *,
            conn=None,
            expected_settings_version: int | None = None,
        ) -> bool:
            self.cas_versions.append(expected_settings_version)
            if expected_settings_version is None:
                self.blind_writes += 1
            raise ConflictError("concurrent settings winner")

    persisted_db = _ConflictingChatDB()

    async def _get_tts_service():
        return _DummyTTSService([b"tts"])

    async def _get_db_for_user_id(_user_id: int, client_id: Optional[str] = None):  # noqa: ARG001
        return persisted_db

    async def _character_context(_db: Any, _character_id: Any, _loop: Any):
        return {"id": 42, "name": "Helpful AI Assistant"}, 42

    async def _conversation_context(*_args: Any, **_kwargs: Any):
        return "resume-chat-conflict", False

    monkeypatch.setattr(audio, "get_tts_service", _get_tts_service)
    monkeypatch.setattr(audio, "get_chacha_db_for_user_id", _get_db_for_user_id, raising=False)
    monkeypatch.setattr(audio, "get_or_create_character_context", _character_context, raising=False)
    monkeypatch.setattr(audio, "get_or_create_conversation", _conversation_context, raising=False)

    await audio.websocket_audio_chat_stream(ws, token=None)

    assert persisted_db.cas_versions == [9]
    assert persisted_db.blind_writes == 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_audio_chat_ws_existing_resumable_session_rejects_credential_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_payload = _pcm16_audio([_AUDIO_LABEL_TO_SAMPLE["abc"]])
    ws = DummyWebSocket(
        [
            {
                "type": "config",
                "session_id": "resume-chat-secret",
                "stt": {"model": "parakeet"},
                "llm": {"provider": "stub", "model": "stub-model"},
                "tts": {"voice": "af_heart", "format": "pcm"},
                "metadata": {
                    "persist_history": True,
                    "nested": {"apiKey": "must-not-persist"},
                },
            },
            {"type": "audio", "data": audio_payload},
            {"type": "commit"},
            {"type": "stop"},
        ]
    )

    existing_settings, existing_resume_state = _valid_roleplay_settings_with_blob(16)
    existing_resume_state["settings_version"] = 4

    class _ExistingResumableDB:
        client_id = "1"

        def __init__(self) -> None:
            self.upsert_calls: list[dict[str, Any]] = []

        def add_message(self, _msg_data: dict[str, Any]) -> str:
            return "msg-id"

        def get_conversation_settings(self, _conversation_id: str) -> dict[str, Any]:
            return {
                "settings": existing_settings,
                "settings_version": 4,
            }

        def transaction(self):
            return nullcontext(object())

        def get_roleplay_resume_state(
            self,
            conversation_id: str,
            **_kwargs: Any,
        ) -> dict[str, Any]:
            return {
                **existing_resume_state,
                "conversation": _owned_audio_test_conversation(conversation_id),
            }

        def get_conversation_by_id(self, conversation_id: str) -> dict[str, Any]:
            return _owned_audio_test_conversation(conversation_id)

        def upsert_conversation_settings(
            self,
            _conversation_id: str,
            settings: dict[str, Any],
            *,
            conn=None,
            expected_settings_version: int | None = None,
        ) -> bool:
            self.upsert_calls.append(dict(settings))
            return True

    persisted_db = _ExistingResumableDB()

    async def _get_tts_service():
        return _DummyTTSService([b"tts"])

    async def _get_db_for_user_id(
        _user_id: int,
        client_id: Optional[str] = None,
    ):  # noqa: ARG001
        return persisted_db

    async def _character_context(_db: Any, _character_id: Any, _loop: Any):
        return {"id": 42, "name": "Helpful AI Assistant"}, 42

    async def _conversation_context(*_args: Any, **_kwargs: Any):
        return "resume-chat-secret", False

    monkeypatch.setattr(audio, "get_tts_service", _get_tts_service)
    monkeypatch.setattr(
        audio,
        "get_chacha_db_for_user_id",
        _get_db_for_user_id,
        raising=False,
    )
    monkeypatch.setattr(
        audio,
        "get_or_create_character_context",
        _character_context,
        raising=False,
    )
    monkeypatch.setattr(
        audio,
        "get_or_create_conversation",
        _conversation_context,
        raising=False,
    )

    await audio.websocket_audio_chat_stream(ws, token=None)

    assert persisted_db.upsert_calls == []


@pytest.mark.integration
async def test_audio_chat_ws_applies_stt_redaction_to_turn_output_and_persistence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_payload = _pcm16_audio([_AUDIO_LABEL_TO_SAMPLE["abc"]])
    ws = DummyWebSocket(
        [
            {
                "type": "config",
                "stt": {"model": "parakeet"},
                "llm": {"provider": "stub", "model": "stub-model"},
                "tts": {"voice": "af_heart", "format": "pcm"},
                "metadata": {"persist_history": True},
            },
            {"type": "audio", "data": audio_payload},
            {"type": "commit"},
            {"type": "stop"},
        ]
    )

    class _PiiTranscriber(_DummyTranscriber):
        def get_full_transcript(self) -> str:
            return "contact alice@example.com"

    class _DummyChatDB:
        def __init__(self) -> None:
            self.messages: List[Dict[str, Any]] = []
            self.settings: List[tuple[str, Dict[str, Any]]] = []

        def add_message(self, msg_data: Dict[str, Any]) -> str:
            self.messages.append(dict(msg_data))
            return "msg-id"

        def upsert_conversation_settings(self, conversation_id: str, settings: Dict[str, Any]) -> bool:
            self.settings.append((conversation_id, settings))
            return True

    persisted_db = _DummyChatDB()
    llm_user_messages: List[str] = []

    async def _get_tts_service():
        return _DummyTTSService([b"tts"])

    async def _get_db_for_user_id(_user_id: int, client_id: Optional[str] = None):  # noqa: ARG001
        return persisted_db

    async def _character_context(_db: Any, _character_id: Any, _loop: Any):
        return {"id": 42, "name": "Helpful AI Assistant"}, 42

    async def _conversation_context(
        _db: Any,
        _conversation_id: Optional[str],
        _character_id: int,
        _character_name: str,
        _client_id: str,
        _loop: Any,
    ):
        return "ws-session-002", True

    async def _llm_with_capture(**kwargs: Any) -> AsyncIterator[str]:
        messages_payload = kwargs.get("messages_payload") or []
        if messages_payload:
            llm_user_messages.append(str(messages_payload[-1].get("content")))
        return await _llm_stub(**kwargs)

    monkeypatch.setattr(audio, "UnifiedStreamingTranscriber", _PiiTranscriber)
    monkeypatch.setattr(audio, "get_tts_service", _get_tts_service)
    monkeypatch.setattr(audio, "get_chacha_db_for_user_id", _get_db_for_user_id, raising=False)
    monkeypatch.setattr(audio, "get_or_create_character_context", _character_context, raising=False)
    monkeypatch.setattr(audio, "get_or_create_conversation", _conversation_context, raising=False)
    monkeypatch.setattr(audio, "chat_api_call_async", _llm_with_capture)

    async def _resolve_policy(**kwargs: Any) -> Any:  # noqa: ARG001
        return SimpleNamespace(
            org_id=7,
            delete_audio_after_success=True,
            audio_retention_hours=0.0,
            redact_pii=True,
            allow_unredacted_partials=False,
            redact_categories=["pii_email"],
        )

    monkeypatch.setattr(audio_streaming_module, "resolve_effective_stt_policy", _resolve_policy)

    await audio.websocket_audio_chat_stream(ws, token=None)

    full_transcripts = [msg for msg in ws.sent_json if msg.get("type") == "full_transcript"]
    assert full_transcripts
    assert full_transcripts[0].get("text") == "contact [PII]"
    assert llm_user_messages == ["contact [PII]"]
    assert persisted_db.messages[0].get("content") == "contact [PII]"


@pytest.mark.integration
async def test_audio_chat_ws_persistence_failure_is_fail_soft(monkeypatch: pytest.MonkeyPatch) -> None:
    audio_payload = _pcm16_audio([_AUDIO_LABEL_TO_SAMPLE["abc"]])
    ws = DummyWebSocket(
        [
            {
                "type": "config",
                "stt": {"model": "parakeet"},
                "llm": {"provider": "stub", "model": "stub-model"},
                "tts": {"voice": "af_heart", "format": "pcm"},
                "metadata": {"persist_history": True},
            },
            {"type": "audio", "data": audio_payload},
            {"type": "commit"},
            {"type": "stop"},
        ]
    )

    async def _get_tts_service():
        return _DummyTTSService([b"tts"])

    async def _db_failure(_user_id: int, client_id: Optional[str] = None):  # noqa: ARG001
        raise RuntimeError("simulated ChaCha initialization failure")

    monkeypatch.setattr(audio, "get_tts_service", _get_tts_service)
    monkeypatch.setattr(audio, "get_chacha_db_for_user_id", _db_failure, raising=False)

    await audio.websocket_audio_chat_stream(ws, token=None)

    assert ws.sent_bytes == [b"tts"]
    assert any(
        msg.get("type") == "warning" and msg.get("warning_type") == "persistence_unavailable"
        for msg in ws.sent_json
    )
    assert any(msg.get("type") == "tts_done" for msg in ws.sent_json)
    assert ws.closed is True


@pytest.mark.integration
async def test_audio_chat_ws_quota_exceeded(monkeypatch: pytest.MonkeyPatch) -> None:
    audio_payload = _pcm16_audio([_AUDIO_LABEL_TO_SAMPLE["abc"]])
    ws = DummyWebSocket(
        [
            {"type": "config", "stt": {"model": "parakeet"}, "llm": {"model": "stub"}, "tts": {"format": "mp3"}},
            {"type": "audio", "data": audio_payload},
        ]
    )

    async def _check_minutes(uid: int, minutes: float) -> tuple[bool, Optional[float]]:  # noqa: ARG002
        return False, None

    monkeypatch.setattr(audio, "check_daily_minutes_allow", _check_minutes)

    monkeypatch.setattr(audio, "get_tts_service", lambda: _DummyTTSService([b"x"]))

    await audio.websocket_audio_chat_stream(ws, token=None)

    quota_errors = [msg for msg in ws.sent_json if msg.get("error_type") == "quota_exceeded"]
    assert quota_errors, "Expected quota_exceeded message"
    # Close code should reflect quota policy (default 4003 unless env flips to 1008)
    assert ws.close_code in {4003, 1008}
    assert ws.closed is True


@pytest.mark.integration
async def test_audio_chat_ws_records_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    audio_payload = _pcm16_audio([_AUDIO_LABEL_TO_SAMPLE["abcd"]])
    ws = DummyWebSocket(
        [
            {
                "type": "config",
                "stt": {"model": "parakeet", "sample_rate": 16000},
                "llm": {"provider": "stub", "model": "stub-model"},
                "tts": {"voice": "af_heart", "format": "mp3"},
            },
            {"type": "audio", "data": audio_payload},
            {"type": "commit"},
            {"type": "stop"},
        ]
    )

    class QueueStub:
        """Queue stub that simulates initial overflow and then enqueues items."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
            self.items = [b"stale"]
            self.first_full = True
            self.not_empty = asyncio.Event()
            self.not_empty.set()

        def put_nowait(self, item: Any) -> None:
            if self.first_full:
                self.first_full = False
                raise asyncio.QueueFull
            self.items.append(item)
            self.not_empty.set()

        async def put(self, item: Any) -> None:
            self.items.append(item)
            self.not_empty.set()

        async def get(self) -> Any:
            await self.not_empty.wait()
            item = self.items.pop(0)
            if not self.items:
                self.not_empty.clear()
            return item

        def get_nowait(self) -> Any:

            if not self.items:
                raise asyncio.QueueEmpty
            item = self.items.pop(0)
            if not self.items:
                self.not_empty.clear()
            return item

    class Registry:
        """Metrics registry stub used to capture increments and observations."""

        def __init__(self) -> None:

            self.increments = []
            self.observes = []
            self.registered = []

        def increment(self, name: str, value: int = 1, labels: Optional[Dict[str, Any]] = None) -> None:
            self.increments.append((name, value, labels or {}))

        def observe(self, name: str, value: float, labels: Optional[Dict[str, Any]] = None) -> None:
            self.observes.append((name, value, labels or {}))

        def register_metric(self, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
            self.registered.append(args)

    reg = Registry()

    async def _allow_minutes(uid: int, minutes: float) -> tuple[bool, float]:  # noqa: ARG002
        return True, 10.0

    async def _get_tts_service():
        class _Service:
            async def generate_speech(self, *args: Any, **kwargs: Any) -> AsyncIterator[bytes]:  # noqa: ARG002
                reg.observe(
                    "voice_to_voice_seconds",
                    0.5,
                    labels={"provider": "stub", "route": kwargs.get("voice_to_voice_route", "")},
                )
                yield b"a"
                yield b"b"

        return _Service()

    monkeypatch.setattr(audio, "check_daily_minutes_allow", _allow_minutes)
    monkeypatch.setattr(audio, "get_tts_service", _get_tts_service)
    monkeypatch.setattr(audio, "get_metrics_registry", lambda: reg)

    # Ensure WS helper uses the same registry
    import tldw_Server_API.app.core.Streaming.streams as streams

    monkeypatch.setattr(streams, "get_metrics_registry", lambda: reg)

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

    await audio.websocket_audio_chat_stream(ws, token=None)

    assert any(name == "audio_stream_underruns_total" for name, _, _ in reg.increments)
    assert any(name == "voice_to_voice_seconds" for name, _, _ in reg.observes)
    assert ws.closed is True
