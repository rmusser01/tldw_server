import sys
import types
import asyncio
import builtins
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
