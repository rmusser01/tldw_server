import sys
import types
import asyncio
import builtins
import gc
import threading
import time
import anyio
import numpy as np
import pytest

from tldw_Server_API.app.core.TTS.adapters import neutts_adapter as neutts_mod
from tldw_Server_API.app.core.TTS.adapters.neutts_adapter import NeuTTSAdapter
from tldw_Server_API.app.core.TTS.adapters.base import TTSRequest, AudioFormat
from tldw_Server_API.app.core.TTS.tts_exceptions import (
    TTSGenerationError,
    TTSModelLoadError,
    TTSValidationError,
)


class _FakeNeuTTSEngine:
    def __init__(self, *args, **kwargs):
        # Simulate HF transformers path (non-quantized)
        self._is_quantized_model = False

    def encode_reference(self, path):

        # Return some dummy codes
        return [1, 2, 3]

    def infer(self, text, ref_codes, ref_text):

        # Return a 0.5s of silence at 24kHz
        return np.zeros(12000, dtype=np.float32)


class _FakeNeuTTSEngineOnnx(_FakeNeuTTSEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Internal flag in upstream, irrelevant for adapter behavior here
        self._is_onnx_codec = True


class _FakeNeuTTSEngineError(_FakeNeuTTSEngine):
    def infer(self, text, ref_codes, ref_text):
        raise ValueError("No valid speech tokens found in the output.")


class _FakeNeuTTSEngineInitError(_FakeNeuTTSEngine):
    def __init__(self, *args, **kwargs):
        raise RuntimeError("RAW_NEUTTS_INIT_SECRET_MARKER token=secret")


class _FakeNeuTTSEngineSecretError(_FakeNeuTTSEngine):
    def infer(self, text, ref_codes, ref_text):
        raise RuntimeError("RAW_NEUTTS_GENERATION_SECRET_MARKER token=secret")


class _FakeNeuTTSEngineStreaming(_FakeNeuTTSEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._is_quantized_model = True

    def infer_stream(self, text, ref_codes, ref_text):
        yield np.zeros(480, dtype=np.float32)


class _FakeNeuTTSEngineStreamingError(_FakeNeuTTSEngineStreaming):
    def infer_stream(self, text, ref_codes, ref_text):
        raise RuntimeError("RAW_NEUTTS_STREAM_SECRET_MARKER token=secret")
        yield np.zeros(480, dtype=np.float32)


def _install_fake_engine(fake_cls):


    """Inject a fake NeuTTSAir into the vendored import path used by the adapter."""
    mod = types.ModuleType("tldw_Server_API.app.core.TTS.vendors.neuttsair.neutts")
    setattr(mod, "NeuTTSAir", fake_cls)
    sys.modules["tldw_Server_API.app.core.TTS.vendors.neuttsair.neutts"] = mod


def _capture_neutts_logs(level: str = "DEBUG") -> tuple[list[str], int]:
    messages: list[str] = []
    sink_id = neutts_mod.logger.add(
        lambda message: messages.append(
            f"{message.record['message']}\n{message.record.get('extra', {})}"
        ),
        level=level,
    )
    return messages, sink_id


@pytest.mark.asyncio
async def test_neutts_import_failure_log_sanitizes_exception_text(monkeypatch):
    raw_marker = "RAW_NEUTTS_IMPORT_SECRET_MARKER token=secret"
    real_import = builtins.__import__

    def fail_neutts_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "tldw_Server_API.app.core.TTS.vendors.neuttsair.neutts":
            raise ImportError(raw_marker)
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fail_neutts_import)
    adapter = NeuTTSAdapter(config={})
    messages, sink_id = _capture_neutts_logs(level="ERROR")

    try:
        with pytest.raises(TTSModelLoadError) as exc_info:
            await adapter.initialize()
    finally:
        neutts_mod.logger.remove(sink_id)

    assert raw_marker in exc_info.value.details["error"]
    rendered_logs = "\n".join(messages)
    assert "NeuTTS import error" in rendered_logs
    assert "RAW_NEUTTS_IMPORT_SECRET_MARKER" not in rendered_logs
    assert "token=secret" not in rendered_logs


@pytest.mark.asyncio
async def test_neutts_registration_failure_log_sanitizes_exception_extra(monkeypatch):
    _install_fake_engine(_FakeNeuTTSEngine)
    raw_marker = "RAW_NEUTTS_REGISTER_SECRET_MARKER token=secret"

    async def fail_resource_manager():
        raise RuntimeError(raw_marker)

    from tldw_Server_API.app.core.TTS import tts_resource_manager

    monkeypatch.setattr(tts_resource_manager, "get_resource_manager", fail_resource_manager)
    adapter = NeuTTSAdapter(config={})
    messages, sink_id = _capture_neutts_logs()

    try:
        assert await adapter.initialize() is True
    finally:
        neutts_mod.logger.remove(sink_id)

    rendered_logs = "\n".join(messages)
    assert "NeuTTS provider registration failed" in rendered_logs
    assert "RAW_NEUTTS_REGISTER_SECRET_MARKER" not in rendered_logs
    assert "token=secret" not in rendered_logs


@pytest.mark.asyncio
async def test_neutts_initialization_failure_log_sanitizes_exception_text():
    _install_fake_engine(_FakeNeuTTSEngineInitError)
    adapter = NeuTTSAdapter(config={})
    messages, sink_id = _capture_neutts_logs(level="ERROR")

    try:
        with pytest.raises(TTSModelLoadError) as exc_info:
            await adapter.initialize()
    finally:
        neutts_mod.logger.remove(sink_id)

    assert "RAW_NEUTTS_INIT_SECRET_MARKER" in exc_info.value.details["error"]
    rendered_logs = "\n".join(messages)
    assert "NeuTTS initialization failed" in rendered_logs
    assert "RAW_NEUTTS_INIT_SECRET_MARKER" not in rendered_logs
    assert "token=secret" not in rendered_logs


@pytest.mark.asyncio
async def test_neutts_generate_validation_failure_log_sanitizes_exception_text(monkeypatch):
    _install_fake_engine(_FakeNeuTTSEngine)
    raw_marker = "RAW_NEUTTS_VALIDATION_SECRET_MARKER token=secret"
    adapter = NeuTTSAdapter(config={})
    assert await adapter.ensure_initialized()

    def fail_validation(*args, **kwargs):
        raise TTSValidationError(raw_marker, provider="neutts")

    monkeypatch.setattr(neutts_mod, "validate_tts_request", fail_validation)
    request = TTSRequest(text="hello", format=AudioFormat.PCM, stream=False)
    messages, sink_id = _capture_neutts_logs(level="ERROR")

    try:
        with pytest.raises(TTSValidationError) as exc_info:
            await adapter.generate(request)
    finally:
        neutts_mod.logger.remove(sink_id)

    assert raw_marker in str(exc_info.value)
    rendered_logs = "\n".join(messages)
    assert "NeuTTS request validation failed" in rendered_logs
    assert "RAW_NEUTTS_VALIDATION_SECRET_MARKER" not in rendered_logs
    assert "token=secret" not in rendered_logs


@pytest.mark.asyncio
async def test_neutts_generation_failure_log_sanitizes_exception_text():
    _install_fake_engine(_FakeNeuTTSEngineSecretError)
    adapter = NeuTTSAdapter(config={})
    assert await adapter.ensure_initialized()
    request = TTSRequest(
        text="fail",
        format=AudioFormat.PCM,
        stream=False,
        extra_params={"reference_text": "text", "ref_codes": [1, 2]},
    )
    messages, sink_id = _capture_neutts_logs(level="ERROR")

    try:
        with pytest.raises(TTSGenerationError) as exc_info:
            await adapter.generate(request)
    finally:
        neutts_mod.logger.remove(sink_id)

    assert "RAW_NEUTTS_GENERATION_SECRET_MARKER" in exc_info.value.details["error"]
    rendered_logs = "\n".join(messages)
    assert "NeuTTS generation error" in rendered_logs
    assert "RAW_NEUTTS_GENERATION_SECRET_MARKER" not in rendered_logs
    assert "token=secret" not in rendered_logs


@pytest.mark.asyncio
async def test_neutts_stream_validation_failure_log_sanitizes_exception_text(monkeypatch):
    _install_fake_engine(_FakeNeuTTSEngineStreaming)
    raw_marker = "RAW_NEUTTS_STREAM_VALIDATION_SECRET_MARKER token=secret"
    adapter = NeuTTSAdapter(config={})
    assert await adapter.ensure_initialized()

    def fail_validation(*args, **kwargs):
        raise TTSValidationError(raw_marker, provider="neutts")

    monkeypatch.setattr(neutts_mod, "validate_tts_request", fail_validation)
    request = TTSRequest(text="hello", format=AudioFormat.PCM, stream=True)
    messages, sink_id = _capture_neutts_logs(level="ERROR")

    try:
        with pytest.raises(TTSValidationError) as exc_info:
            async for _chunk in adapter.generate_stream(request):
                pass
    finally:
        neutts_mod.logger.remove(sink_id)

    assert raw_marker in str(exc_info.value)
    rendered_logs = "\n".join(messages)
    assert "NeuTTS request validation failed" in rendered_logs
    assert "RAW_NEUTTS_STREAM_VALIDATION_SECRET_MARKER" not in rendered_logs
    assert "token=secret" not in rendered_logs


@pytest.mark.asyncio
async def test_neutts_streaming_failure_log_sanitizes_exception_text():
    _install_fake_engine(_FakeNeuTTSEngineStreamingError)
    adapter = NeuTTSAdapter(config={})
    assert await adapter.ensure_initialized()
    request = TTSRequest(
        text="stream",
        format=AudioFormat.PCM,
        stream=True,
        extra_params={"reference_text": "ref", "ref_codes": [1, 2, 3]},
    )
    messages, sink_id = _capture_neutts_logs(level="ERROR")

    try:
        with pytest.raises(TTSGenerationError) as exc_info:
            async for _chunk in adapter.generate_stream(request):
                pass
    finally:
        neutts_mod.logger.remove(sink_id)

    assert "RAW_NEUTTS_STREAM_SECRET_MARKER" in exc_info.value.details["error"]
    rendered_logs = "\n".join(messages)
    assert "NeuTTS streaming error" in rendered_logs
    assert "RAW_NEUTTS_STREAM_SECRET_MARKER" not in rendered_logs
    assert "token=secret" not in rendered_logs


@pytest.mark.asyncio
async def test_neutts_hf_path_generation(monkeypatch):
    _install_fake_engine(_FakeNeuTTSEngine)
    adapter = NeuTTSAdapter(config={
        "backbone_repo": "neuphonic/neutts-air",
        "codec_repo": "neuphonic/neucodec",
        "sample_rate": 24000,
    })
    assert await adapter.ensure_initialized()
    # Use ref_codes path to avoid validator requiring a real audio container
    req = TTSRequest(
        text="hello",
        format=AudioFormat.PCM,
        stream=False,
        extra_params={"reference_text": "hello world", "ref_codes": [1, 2, 3]},
    )
    resp = await adapter.generate(req)
    assert resp.audio_data and len(resp.audio_data) > 0
    assert resp.format == AudioFormat.PCM


@pytest.mark.asyncio
async def test_neutts_onnx_codec_generation(monkeypatch):
    _install_fake_engine(_FakeNeuTTSEngineOnnx)
    adapter = NeuTTSAdapter(config={
        "backbone_repo": "neuphonic/neutts-air",
        "codec_repo": "neuphonic/neucodec-onnx-decoder",
        "sample_rate": 24000,
    })
    assert await adapter.ensure_initialized()
    req = TTSRequest(
        text="check",
        format=AudioFormat.PCM,
        stream=False,
        extra_params={"reference_text": "ref", "ref_codes": [3, 2, 1]},
    )
    resp = await adapter.generate(req)
    assert resp.audio_data and len(resp.audio_data) > 0


@pytest.mark.asyncio
async def test_neutts_no_speech_tokens_error(monkeypatch):
    _install_fake_engine(_FakeNeuTTSEngineError)
    adapter = NeuTTSAdapter(config={
        "backbone_repo": "neuphonic/neutts-air",
        "codec_repo": "neuphonic/neucodec",
        "sample_rate": 24000,
    })
    assert await adapter.ensure_initialized()
    req = TTSRequest(
        text="fail",
        format=AudioFormat.PCM,
        stream=False,
        extra_params={"reference_text": "text", "ref_codes": [1, 2]},
    )
    with pytest.raises(TTSGenerationError):
        await adapter.generate(req)


@pytest.mark.asyncio
async def test_neutts_streaming_requires_gguf(monkeypatch):
    _install_fake_engine(_FakeNeuTTSEngine)
    adapter = NeuTTSAdapter(config={
        "backbone_repo": "neuphonic/neutts-air",
        "codec_repo": "neuphonic/neucodec",
        "sample_rate": 24000,
    })
    assert await adapter.ensure_initialized()
    req = TTSRequest(
        text="stream",
        format=AudioFormat.PCM,
        stream=True,
        extra_params={"reference_text": "ref", "ref_codes": [1, 2, 3]},
    )
    with pytest.raises(TTSGenerationError):
        await adapter.generate(req)


@pytest.mark.asyncio
async def test_neutts_streaming_pcm(monkeypatch):
    _install_fake_engine(_FakeNeuTTSEngineStreaming)
    adapter = NeuTTSAdapter(config={
        "backbone_repo": "neuphonic/neutts-air-q4-gguf",
        "codec_repo": "neuphonic/neucodec",
        "sample_rate": 24000,
    })
    assert await adapter.ensure_initialized()
    req = TTSRequest(
        text="stream",
        format=AudioFormat.PCM,
        stream=True,
        extra_params={"reference_text": "ref", "ref_codes": [1, 2, 3]},
    )
    resp = await adapter.generate(req)
    assert resp.audio_stream is not None
    chunks = []
    async for chunk in resp.audio_stream:
        chunks.append(chunk)
    assert chunks


class _ThreadRecordingEngine(_FakeNeuTTSEngineStreaming):
    """Records the thread each engine call runs on (TASK-13855)."""

    calls: list[tuple[str, int]] = []

    def encode_reference(self, path):
        self.calls.append(("encode_reference", threading.get_ident()))
        return [1, 2, 3]

    def infer(self, text, ref_codes, ref_text):
        self.calls.append(("infer", threading.get_ident()))
        return np.zeros(12000, dtype=np.float32)

    def infer_stream(self, text, ref_codes, ref_text):
        for _ in range(2):
            self.calls.append(("infer_stream", threading.get_ident()))
            yield np.zeros(480, dtype=np.float32)


@pytest.mark.asyncio
async def test_neutts_runs_engine_calls_off_the_event_loop(monkeypatch):
    """In-process long-form jobs share the server's loop; a blocking engine call would freeze it."""
    _install_fake_engine(_ThreadRecordingEngine)
    _ThreadRecordingEngine.calls = []
    monkeypatch.setattr(neutts_mod, "validate_tts_request", lambda *a, **k: None)
    adapter = NeuTTSAdapter(config={"backbone_repo": "neuphonic/neutts-air-q4-gguf"})
    assert await adapter.ensure_initialized()
    loop_thread = threading.get_ident()
    extras = {"reference_text": "ref"}

    resp = await adapter.generate(
        TTSRequest(text="hi", format=AudioFormat.PCM, stream=False, voice_reference=b"x", extra_params=extras)
    )
    assert resp.audio_data
    chunks = [
        c
        async for c in adapter.generate_stream(
            TTSRequest(text="hi", format=AudioFormat.PCM, stream=True, voice_reference=b"x", extra_params=extras)
        )
    ]
    assert chunks

    names = [name for name, _ in _ThreadRecordingEngine.calls]
    assert names == ["encode_reference", "infer", "encode_reference", "infer_stream", "infer_stream"]
    assert [name for name, ident in _ThreadRecordingEngine.calls if ident == loop_thread] == []


class _OverlapRecordingEngine(_FakeNeuTTSEngineStreaming):
    """Records how many engine calls run at once; a llama-cpp context is not thread-safe."""

    active = 0
    peak = 0
    guard = threading.Lock()

    @classmethod
    def _enter(cls):
        with cls.guard:
            cls.active += 1
            cls.peak = max(cls.peak, cls.active)

    @classmethod
    def _exit(cls):
        with cls.guard:
            cls.active -= 1

    def encode_reference(self, path):
        self._enter()
        time.sleep(0.02)
        self._exit()
        return [1, 2, 3]

    def infer(self, text, ref_codes, ref_text):
        self._enter()
        time.sleep(0.05)
        self._exit()
        return np.zeros(12000, dtype=np.float32)

    def infer_stream(self, text, ref_codes, ref_text):
        self._enter()
        try:
            for _ in range(3):
                time.sleep(0.02)
                yield np.zeros(480, dtype=np.float32)
        finally:
            self._exit()


@pytest.mark.asyncio
async def test_neutts_runs_one_engine_call_at_a_time(monkeypatch):
    """Off the loop, concurrent requests would otherwise share the engine across threads."""
    _install_fake_engine(_OverlapRecordingEngine)
    _OverlapRecordingEngine.active = _OverlapRecordingEngine.peak = 0
    monkeypatch.setattr(neutts_mod, "validate_tts_request", lambda *a, **k: None)
    adapter = NeuTTSAdapter(config={"backbone_repo": "neuphonic/neutts-air-q4-gguf"})
    assert await adapter.ensure_initialized()
    extras = {"reference_text": "ref"}

    async def _stream() -> list[bytes]:
        request = TTSRequest(text="hi", format=AudioFormat.PCM, stream=True, extra_params={**extras, "ref_codes": [1]})
        return [c async for c in adapter.generate_stream(request)]

    *responses, chunks = await asyncio.gather(
        *(
            adapter.generate(
                TTSRequest(text="hi", format=AudioFormat.PCM, stream=False, voice_reference=b"x", extra_params=extras)
            )
            for _ in range(3)
        ),
        _stream(),
    )
    assert all(r.audio_data for r in responses)
    assert chunks
    assert _OverlapRecordingEngine.peak == 1


@pytest.mark.asyncio
async def test_neutts_stream_keeps_the_engine_between_chunks():
    """llama-cpp keeps a stream's state in the shared context, so no other call may run mid-stream."""
    _install_fake_engine(_ThreadRecordingEngine)
    _ThreadRecordingEngine.calls = []
    adapter = NeuTTSAdapter(config={"backbone_repo": "neuphonic/neutts-air-q4-gguf"})
    assert await adapter.ensure_initialized()
    extras = {"reference_text": "ref", "ref_codes": [1, 2, 3]}

    stream = adapter.generate_stream(TTSRequest(text="hi", format=AudioFormat.PCM, stream=True, extra_params=extras))
    assert await stream.__anext__()
    other = asyncio.create_task(
        adapter.generate(TTSRequest(text="hi", format=AudioFormat.PCM, stream=False, extra_params=extras))
    )
    await asyncio.sleep(0.1)
    assert [name for name, _ in _ThreadRecordingEngine.calls] == ["infer_stream"]
    assert [c async for c in stream]
    assert (await other).audio_data
    assert [name for name, _ in _ThreadRecordingEngine.calls] == ["infer_stream", "infer_stream", "infer"]


class _BlockingEngine(_FakeNeuTTSEngineStreaming):
    """Engine calls wait for `release`, so a caller can be cancelled mid-call."""

    started: list[str] = []
    release = threading.Event()

    def infer(self, text, ref_codes, ref_text):
        self.started.append("infer")
        self.release.wait(5)
        return np.zeros(12000, dtype=np.float32)

    def infer_stream(self, text, ref_codes, ref_text):
        self.started.append("infer_stream")
        self.release.wait(5)
        yield np.zeros(480, dtype=np.float32)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("first_streams", "via_anyio_scope"),
    [(False, False), (True, False), (True, True)],
    # Starlette cancels a disconnected stream through an anyio cancel scope, which re-cancels on every await.
    ids=["generate", "stream", "stream-anyio-scope"],
)
async def test_neutts_cancelled_request_keeps_the_engine_until_its_call_returns(first_streams, via_anyio_scope):
    """A client disconnect cancels the request, but asyncio.to_thread cannot stop the engine call."""
    _install_fake_engine(_BlockingEngine)
    _BlockingEngine.started = []
    _BlockingEngine.release = threading.Event()
    adapter = NeuTTSAdapter(config={"backbone_repo": "neuphonic/neutts-air-q4-gguf"})
    assert await adapter.ensure_initialized()
    extras = {"reference_text": "ref", "ref_codes": [1, 2, 3]}
    scopes: list[anyio.CancelScope] = []

    async def _request():
        request = TTSRequest(text="hi", format=AudioFormat.PCM, stream=first_streams, extra_params=extras)
        if first_streams:
            return [c async for c in adapter.generate_stream(request)]
        return await adapter.generate(request)

    async def _first():
        if not via_anyio_scope:
            return await _request()
        with anyio.CancelScope() as scope:
            scopes.append(scope)
            await _request()

    try:
        first = asyncio.create_task(_first())
        while not _BlockingEngine.started:
            await asyncio.sleep(0.01)
        second = asyncio.create_task(
            adapter.generate(TTSRequest(text="hi", format=AudioFormat.PCM, stream=False, extra_params=extras))
        )
        await asyncio.sleep(0.05)
        if via_anyio_scope:
            scopes[0].cancel()
        else:
            first.cancel()
        await asyncio.sleep(0.1)
        # The cancelled request's engine call is still running, so the next one has not started.
        assert len(_BlockingEngine.started) == 1
    finally:
        _BlockingEngine.release.set()
    assert (await second).audio_data
    assert len(_BlockingEngine.started) == 2
    if via_anyio_scope:
        assert await first is None
    else:
        with pytest.raises(asyncio.CancelledError):
            await first


class _OrphanedErrorEngine(_BlockingEngine):
    """Fails the call whose caller was cancelled, after that caller has gone."""

    def infer(self, text, ref_codes, ref_text):
        self.started.append("infer")
        self.release.wait(5)
        if text == "fail":
            raise RuntimeError("RAW_NEUTTS_ORPHANED_SECRET_MARKER token=secret")
        return np.zeros(12000, dtype=np.float32)


@pytest.mark.asyncio
async def test_neutts_cancelled_request_engine_error_is_not_logged_raw():
    """Nobody awaits a cancelled caller's engine call, so asyncio would log its error unsanitised."""
    _install_fake_engine(_OrphanedErrorEngine)
    _OrphanedErrorEngine.started = []
    _OrphanedErrorEngine.release = threading.Event()
    adapter = NeuTTSAdapter(config={"backbone_repo": "neuphonic/neutts-air-q4-gguf"})
    assert await adapter.ensure_initialized()
    extras = {"reference_text": "ref", "ref_codes": [1, 2, 3]}
    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()
    unhandled: list[dict] = []
    loop.set_exception_handler(lambda _loop, context: unhandled.append(context))
    try:
        doomed = asyncio.create_task(
            adapter.generate(TTSRequest(text="fail", format=AudioFormat.PCM, stream=False, extra_params=extras))
        )
        while not _OrphanedErrorEngine.started:
            await asyncio.sleep(0.01)
        doomed.cancel()
        with pytest.raises(asyncio.CancelledError):
            await doomed
        _OrphanedErrorEngine.release.set()
        # The next call gets the engine only once the failed call has returned.
        assert (
            await adapter.generate(TTSRequest(text="ok", format=AudioFormat.PCM, stream=False, extra_params=extras))
        ).audio_data
        gc.collect()
        await asyncio.sleep(0)
    finally:
        _OrphanedErrorEngine.release.set()
        loop.set_exception_handler(previous_handler)
    assert unhandled == []


@pytest.mark.asyncio
async def test_neutts_streaming_rejects_wav(monkeypatch):
    _install_fake_engine(_FakeNeuTTSEngineStreaming)
    adapter = NeuTTSAdapter(config={
        "backbone_repo": "neuphonic/neutts-air-q4-gguf",
        "codec_repo": "neuphonic/neucodec",
        "sample_rate": 24000,
    })
    assert await adapter.ensure_initialized()
    req = TTSRequest(
        text="stream",
        format=AudioFormat.WAV,
        stream=True,
        extra_params={"reference_text": "ref", "ref_codes": [1, 2, 3]},
    )
    with pytest.raises(TTSValidationError):
        await adapter.generate(req)
