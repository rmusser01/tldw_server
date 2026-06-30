# test_pocket_tts_adapter_mock.py
# Description: Mock/Unit tests for PocketTTS ONNX adapter
#
import sys
import types

import numpy as np
import pytest

from tldw_Server_API.app.core.TTS.adapters.base import AudioFormat, ProviderStatus, TTSRequest
from tldw_Server_API.app.core.TTS.adapters import pocket_tts_adapter as pocket_mod
from tldw_Server_API.app.core.TTS.adapters.pocket_tts_adapter import PocketTTSOnnxAdapter
from tldw_Server_API.app.core.TTS.adapter_registry import TTSAdapterFactory, TTSProvider
from tldw_Server_API.app.core.TTS.tts_exceptions import (
    TTSGenerationError,
    TTSInvalidVoiceReferenceError,
    TTSModelNotFoundError,
    TTSProviderInitializationError,
)


class DummyPocketEngine:
    SAMPLE_RATE = 24000

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def generate(self, text, voice, max_frames=500):
        return np.zeros(240, dtype=np.float32)

    def stream(
        self,
        text,
        voice,
        max_frames=500,
        first_chunk_frames=2,
        target_buffer_sec=0.2,
        max_chunk_frames=15,
    ):
        for _ in range(2):
            yield np.zeros(240, dtype=np.float32)


class FailingGeneratePocketEngine(DummyPocketEngine):
    def __init__(self, marker):
        super().__init__()
        self.marker = marker

    def generate(self, text, voice, max_frames=500):
        raise RuntimeError(self.marker)


class FailingStreamPocketEngine(DummyPocketEngine):
    def __init__(self, marker):
        super().__init__()
        self.marker = marker

    def stream(
        self,
        text,
        voice,
        max_frames=500,
        first_chunk_frames=2,
        target_buffer_sec=0.2,
        max_chunk_frames=15,
    ):
        raise RuntimeError(self.marker)
        yield np.zeros(240, dtype=np.float32)


def _capture_pocket_logs(level="DEBUG"):
    messages = []
    sink_id = pocket_mod.logger.add(
        lambda message: messages.append(
            f"{message.record['message']}\n{message.record.get('extra', {})}"
        ),
        level=level,
    )
    return messages, sink_id


def _valid_voice_reference_bytes():
    # Minimal WAV header signature to satisfy validation checks.
    return b"RIFF\x24\x00\x00\x00WAVEfmt "


def _create_pocket_assets(tmp_path, precision="int8"):
    models_dir = tmp_path / "onnx"
    models_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = tmp_path / "tokenizer.model"
    tokenizer_path.write_bytes(b"token")

    suffix = "_int8" if precision == "int8" else ""
    required = [
        f"flow_lm_main{suffix}.onnx",
        f"flow_lm_flow{suffix}.onnx",
        f"mimi_decoder{suffix}.onnx",
        "mimi_encoder.onnx",
        "text_conditioner.onnx",
    ]
    for name in required:
        (models_dir / name).write_bytes(b"")

    return models_dir, tokenizer_path


@pytest.mark.asyncio
async def test_pocket_tts_capabilities_defaults():
    adapter = PocketTTSOnnxAdapter({})
    caps = await adapter.get_capabilities()

    assert caps.provider_name == "PocketTTS"
    assert caps.supports_streaming is True
    assert caps.supports_voice_cloning is True
    assert AudioFormat.WAV in caps.supported_formats
    assert caps.sample_rate == adapter.sample_rate


def test_pocket_tts_model_mapping():
    factory = TTSAdapterFactory({})
    provider = factory.get_provider_for_model("pocket-tts-onnx")
    assert provider == TTSProvider.POCKET_TTS


@pytest.mark.asyncio
async def test_pocket_tts_initialize_missing_assets(tmp_path):
    adapter = PocketTTSOnnxAdapter(
        {"model_path": str(tmp_path / "missing"), "tokenizer_path": str(tmp_path / "tokenizer.model")}
    )
    (tmp_path / "tokenizer.model").write_bytes(b"token")

    with pytest.raises(TTSModelNotFoundError):
        await adapter.initialize()


@pytest.mark.asyncio
async def test_pocket_tts_initialize_invalid_precision():
    adapter = PocketTTSOnnxAdapter({"precision": "bad"})
    with pytest.raises(TTSProviderInitializationError):
        await adapter.initialize()


@pytest.mark.asyncio
async def test_pocket_tts_initialize_invalid_device():
    adapter = PocketTTSOnnxAdapter({"device": "warp-drive"})
    with pytest.raises(TTSProviderInitializationError):
        await adapter.initialize()


@pytest.mark.asyncio
async def test_pocket_tts_initialize_success(tmp_path, monkeypatch):
    models_dir, tokenizer_path = _create_pocket_assets(tmp_path, precision="int8")
    dummy_module = types.SimpleNamespace(PocketTTSOnnx=DummyPocketEngine)
    monkeypatch.setitem(sys.modules, "pocket_tts_onnx", dummy_module)

    adapter = PocketTTSOnnxAdapter(
        {
            "model_path": str(models_dir),
            "tokenizer_path": str(tokenizer_path),
            "precision": "int8",
            "device": "cpu",
        }
    )

    success = await adapter.initialize()
    assert success is True
    assert adapter.status == ProviderStatus.AVAILABLE
    assert adapter._engine is not None


@pytest.mark.asyncio
async def test_pocket_tts_registration_failure_log_sanitizes_exception_extra(tmp_path, monkeypatch):
    raw_marker = "RAW_POCKET_REGISTER_SECRET_MARKER token=secret"
    models_dir, tokenizer_path = _create_pocket_assets(tmp_path, precision="int8")
    dummy_module = types.SimpleNamespace(PocketTTSOnnx=DummyPocketEngine)
    monkeypatch.setitem(sys.modules, "pocket_tts_onnx", dummy_module)

    class FailingResourceManager:
        def register_model(self, **kwargs):
            raise RuntimeError(raw_marker)

    async def get_failing_resource_manager():
        return FailingResourceManager()

    from tldw_Server_API.app.core.TTS import tts_resource_manager

    monkeypatch.setattr(tts_resource_manager, "get_resource_manager", get_failing_resource_manager)
    adapter = PocketTTSOnnxAdapter(
        {
            "model_path": str(models_dir),
            "tokenizer_path": str(tokenizer_path),
            "precision": "int8",
            "device": "cpu",
        }
    )
    messages, sink_id = _capture_pocket_logs()

    try:
        assert await adapter.initialize() is True
    finally:
        pocket_mod.logger.remove(sink_id)

    rendered_logs = "\n".join(messages)
    assert "PocketTTS provider registration failed" in rendered_logs
    assert "RAW_POCKET_REGISTER_SECRET_MARKER" not in rendered_logs
    assert "token=secret" not in rendered_logs


@pytest.mark.asyncio
async def test_pocket_tts_initialize_with_module_path(tmp_path):
    models_dir, tokenizer_path = _create_pocket_assets(tmp_path, precision="int8")
    module_dir = tmp_path / "module"
    module_dir.mkdir(parents=True, exist_ok=True)
    module_file = module_dir / "pocket_tts_onnx.py"
    module_file.write_text(
        "class PocketTTSOnnx:\n"
        "    SAMPLE_RATE = 24000\n"
        "    def __init__(self, **kwargs):\n"
        "        self.kwargs = kwargs\n",
        encoding="utf-8",
    )

    sys.modules.pop("pocket_tts_onnx", None)
    original_sys_path = list(sys.path)

    adapter = PocketTTSOnnxAdapter(
        {
            "model_path": str(models_dir),
            "tokenizer_path": str(tokenizer_path),
            "precision": "int8",
            "device": "cpu",
            "module_path": str(module_dir),
        }
    )

    try:
        success = await adapter.initialize()
    finally:
        sys.path[:] = original_sys_path
        sys.modules.pop("pocket_tts_onnx", None)

    assert success is True
    assert adapter._engine is not None


@pytest.mark.asyncio
async def test_pocket_tts_requires_voice_reference():
    adapter = PocketTTSOnnxAdapter({})
    adapter._initialized = True
    adapter._status = ProviderStatus.AVAILABLE
    adapter._engine = DummyPocketEngine()

    request = TTSRequest(
        text="hello",
        voice="clone",
        format=AudioFormat.PCM,
        stream=False,
    )

    with pytest.raises(TTSInvalidVoiceReferenceError):
        await adapter.generate(request)


@pytest.mark.asyncio
async def test_pocket_tts_generate_streaming_returns_bytes():
    adapter = PocketTTSOnnxAdapter({})
    adapter._initialized = True
    adapter._status = ProviderStatus.AVAILABLE
    adapter._engine = DummyPocketEngine()

    request = TTSRequest(
        text="hello",
        voice="clone",
        format=AudioFormat.PCM,
        stream=True,
        voice_reference=_valid_voice_reference_bytes(),
        extra_params={
            "validate_reference": False,
            "convert_reference": False,
        },
    )

    response = await adapter.generate(request)
    chunks = [chunk async for chunk in response.audio_stream]
    assert sum(len(chunk) for chunk in chunks) > 0


@pytest.mark.asyncio
async def test_pocket_tts_generate_pcm_bytes():
    adapter = PocketTTSOnnxAdapter({})
    adapter._initialized = True
    adapter._status = ProviderStatus.AVAILABLE
    adapter._engine = DummyPocketEngine()

    request = TTSRequest(
        text="hello",
        voice="clone",
        format=AudioFormat.PCM,
        stream=False,
        voice_reference=_valid_voice_reference_bytes(),
        extra_params={
            "validate_reference": False,
            "convert_reference": False,
        },
    )

    response = await adapter.generate(request)
    assert response.audio_data is not None
    assert len(response.audio_data) > 0


@pytest.mark.asyncio
async def test_pocket_tts_generation_failure_log_sanitizes_exception_text():
    raw_marker = "RAW_POCKET_GENERATION_SECRET_MARKER token=secret"
    adapter = PocketTTSOnnxAdapter({})
    adapter._initialized = True
    adapter._status = ProviderStatus.AVAILABLE
    adapter._engine = FailingGeneratePocketEngine(raw_marker)

    request = TTSRequest(
        text="hello",
        voice="clone",
        format=AudioFormat.PCM,
        stream=False,
        voice_reference=_valid_voice_reference_bytes(),
        extra_params={
            "validate_reference": False,
            "convert_reference": False,
        },
    )
    messages, sink_id = _capture_pocket_logs(level="ERROR")

    try:
        with pytest.raises(TTSGenerationError) as exc_info:
            await adapter.generate(request)
    finally:
        pocket_mod.logger.remove(sink_id)

    assert raw_marker in exc_info.value.details["error"]
    rendered_logs = "\n".join(messages)
    assert "PocketTTS generation failed" in rendered_logs
    assert "RAW_POCKET_GENERATION_SECRET_MARKER" not in rendered_logs
    assert "token=secret" not in rendered_logs


@pytest.mark.asyncio
async def test_pocket_tts_streaming_failure_log_sanitizes_exception_text():
    raw_marker = "RAW_POCKET_STREAM_SECRET_MARKER token=secret"
    adapter = PocketTTSOnnxAdapter({})
    adapter._initialized = True
    adapter._status = ProviderStatus.AVAILABLE
    adapter._engine = FailingStreamPocketEngine(raw_marker)

    request = TTSRequest(
        text="hello",
        voice="clone",
        format=AudioFormat.PCM,
        stream=True,
        voice_reference=_valid_voice_reference_bytes(),
        extra_params={
            "validate_reference": False,
            "convert_reference": False,
        },
    )
    response = await adapter.generate(request)
    messages, sink_id = _capture_pocket_logs(level="ERROR")

    try:
        with pytest.raises(TTSGenerationError) as exc_info:
            async for _chunk in response.audio_stream:
                pass
    finally:
        pocket_mod.logger.remove(sink_id)

    assert raw_marker in exc_info.value.details["error"]
    rendered_logs = "\n".join(messages)
    assert "PocketTTS streaming failed" in rendered_logs
    assert "RAW_POCKET_STREAM_SECRET_MARKER" not in rendered_logs
    assert "token=secret" not in rendered_logs


def test_pocket_tts_stream_chunk_conversion_log_sanitizes_exception_text():
    raw_marker = "RAW_POCKET_STREAM_CHUNK_SECRET_MARKER token=secret"

    class BadArray:
        def __array__(self, dtype=None):
            raise RuntimeError(raw_marker)

    adapter = PocketTTSOnnxAdapter({})
    writer = pocket_mod.StreamingAudioWriter(
        format=AudioFormat.PCM.value,
        sample_rate=adapter.sample_rate,
        channels=1,
    )
    messages, sink_id = _capture_pocket_logs(level="WARNING")

    try:
        result = adapter._convert_stream_chunk(BadArray(), adapter._audio_normalizer, writer)
    finally:
        writer.close()
        pocket_mod.logger.remove(sink_id)

    assert result == b""
    rendered_logs = "\n".join(messages)
    assert "PocketTTS stream chunk conversion error" in rendered_logs
    assert "RAW_POCKET_STREAM_CHUNK_SECRET_MARKER" not in rendered_logs
    assert "token=secret" not in rendered_logs
