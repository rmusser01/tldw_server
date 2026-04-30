import pytest
import numpy as np
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from tldw_Server_API.app.core.TTS.adapters.base import AudioFormat, TTSRequest
from tldw_Server_API.app.core.TTS.adapters import echo_tts_adapter as echo_mod
from tldw_Server_API.app.core.TTS.adapters.echo_tts_adapter import EchoTTSAdapter
from tldw_Server_API.app.core.TTS.tts_exceptions import TTSGenerationError


pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


class DummyTensor:
    def to(self, *args, **kwargs):
        return self


def _capture_echo_logs(level: str = "DEBUG") -> tuple[list[str], int]:
    messages: list[str] = []
    sink_id = echo_mod.logger.add(
        lambda message: messages.append(
            f"{message.record['message']}\n{message.record.get('extra', {})}"
        ),
        level=level,
    )
    return messages, sink_id


@pytest.mark.asyncio
async def test_echo_tts_initialize_failure_log_sanitizes_exception_text(monkeypatch):
    raw_marker = "RAW_ECHO_INIT_SECRET_MARKER token=secret"
    adapter = EchoTTSAdapter(config={})
    monkeypatch.setattr(adapter, "_import_torch", MagicMock(side_effect=RuntimeError(raw_marker)))
    messages, sink_id = _capture_echo_logs(level="ERROR")

    try:
        initialized = await adapter.initialize()
    finally:
        echo_mod.logger.remove(sink_id)

    assert initialized is False
    rendered_logs = "\n".join(messages)
    assert "initialization failed" in rendered_logs
    assert "RAW_ECHO_INIT_SECRET_MARKER" not in rendered_logs
    assert "token=secret" not in rendered_logs


@pytest.mark.asyncio
async def test_echo_tts_generation_failure_log_sanitizes_exception_text(monkeypatch):
    raw_marker = "RAW_ECHO_GENERATION_SECRET_MARKER token=secret"
    adapter = EchoTTSAdapter(config={})
    adapter._echo_inference = object()
    adapter.device = "cuda"
    adapter._model = SimpleNamespace(device="cuda", dtype="float32")
    adapter._fish_ae = object()
    adapter._pca_state = object()

    monkeypatch.setattr(adapter, "ensure_initialized", AsyncMock(return_value=True))
    monkeypatch.setattr(adapter, "_ensure_models_loaded", AsyncMock())
    monkeypatch.setattr(adapter, "_extract_voice_reference", lambda v: v)
    monkeypatch.setattr(adapter, "_prepare_voice_reference", AsyncMock(side_effect=lambda b, e: b))
    monkeypatch.setattr(adapter, "_get_cached_speaker_latent", AsyncMock(return_value=(DummyTensor(), DummyTensor())))
    monkeypatch.setattr(adapter, "_store_speaker_latent", AsyncMock())
    monkeypatch.setattr(adapter, "_import_torch", lambda: object())
    monkeypatch.setattr(adapter, "_prepare_text_inputs", MagicMock(return_value=("ids", "mask", "normalized")))
    monkeypatch.setattr(adapter, "_run_full_generation", MagicMock(side_effect=RuntimeError(raw_marker)))

    request = TTSRequest(
        text="hello",
        format=AudioFormat.WAV,
        stream=False,
        voice_reference=b"RIFF" + b"\x00" * 100,
    )
    messages, sink_id = _capture_echo_logs(level="ERROR")

    try:
        with pytest.raises(TTSGenerationError) as exc_info:
            await adapter.generate(request)
    finally:
        echo_mod.logger.remove(sink_id)

    assert raw_marker in exc_info.value.details["error"]
    rendered_logs = "\n".join(messages)
    assert "Echo-TTS generation failed" in rendered_logs
    assert "RAW_ECHO_GENERATION_SECRET_MARKER" not in rendered_logs
    assert "token=secret" not in rendered_logs


@pytest.mark.asyncio
async def test_echo_tts_generate_chunked_non_streaming(monkeypatch):
    adapter = EchoTTSAdapter(config={})
    adapter._echo_inference = object()
    adapter.device = "cuda"
    adapter._model = SimpleNamespace(device="cuda", dtype="float32")
    adapter._fish_ae = object()
    adapter._pca_state = object()

    monkeypatch.setattr(adapter, "ensure_initialized", AsyncMock(return_value=True))
    monkeypatch.setattr(adapter, "_ensure_models_loaded", AsyncMock())
    monkeypatch.setattr(adapter, "_extract_voice_reference", lambda v: v)
    monkeypatch.setattr(adapter, "_prepare_voice_reference", AsyncMock(side_effect=lambda b, e: b))
    monkeypatch.setattr(adapter, "_get_cached_speaker_latent", AsyncMock(return_value=(None, None)))
    monkeypatch.setattr(adapter, "_compute_speaker_latent", AsyncMock(return_value=(DummyTensor(), DummyTensor())))
    monkeypatch.setattr(adapter, "_store_speaker_latent", AsyncMock())
    monkeypatch.setattr(adapter, "_import_torch", lambda: object())
    monkeypatch.setattr(adapter, "_prepare_text_inputs", MagicMock(return_value=("ids", "mask", "normalized")))
    monkeypatch.setattr(adapter, "_run_full_generation", MagicMock(return_value="latent"))

    async def fake_latent_to_audio_np(*args, **kwargs):
        return np.array([0.1, 0.2], dtype=np.float32)

    async def fake_convert_audio_format(audio_data, source_format, target_format, sample_rate):
        return b"x" * len(audio_data)

    monkeypatch.setattr(adapter, "_latent_to_audio_np", fake_latent_to_audio_np)
    monkeypatch.setattr(adapter, "convert_audio_format", fake_convert_audio_format)

    alphabet = "abcdefghijklmnopqrstuvwxyz"
    long_text = "".join(alphabet[i % len(alphabet)] for i in range(adapter.MAX_TEXT_BYTES + 10))
    request = TTSRequest(
        text=long_text,
        format=AudioFormat.WAV,
        stream=False,
        voice_reference=b"RIFF" + b"\x00" * 100,
        extra_params={"chunk_text": True},
    )

    chunks = adapter._split_text_chunks(
        long_text,
        max_chars=adapter.MAX_TEXT_LENGTH,
        max_bytes=adapter.MAX_TEXT_BYTES,
    )

    response = await adapter.generate(request)

    assert response.audio_data is not None
    assert adapter._run_full_generation.call_count == len(chunks)
    assert len(response.audio_data) == len(chunks) * 2
