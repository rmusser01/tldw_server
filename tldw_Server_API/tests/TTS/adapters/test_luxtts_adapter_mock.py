import sys
import types

import numpy as np
import pytest

from tldw_Server_API.app.core.TTS.adapters.base import AudioFormat, TTSRequest
from tldw_Server_API.app.core.TTS.adapters import luxtts_adapter as luxtts_mod
from tldw_Server_API.app.core.TTS.adapters.luxtts_adapter import LuxTTSAdapter
from tldw_Server_API.app.core.TTS.tts_exceptions import TTSGenerationError


class _FakeLuxTTS:
    def __init__(self, model_path=None, device="cpu", threads=2, holder=None):
        self.model_path = model_path
        self.device = device
        self.threads = threads
        self.calls = []
        self.vocos = types.SimpleNamespace(return_48k=True)
        if isinstance(holder, dict):
            holder["instance"] = self

    def encode_prompt(self, prompt_audio, duration=5, rms=0.001):
        self.calls.append(("encode", prompt_audio, duration, rms))
        return {
            "prompt_tokens": [1],
            "prompt_features_lens": [1],
            "prompt_features": [1],
            "prompt_rms": [0.1],
        }

    def generate_speech(
        self,
        text,
        encode_dict,
        num_steps=4,
        guidance_scale=3.0,
        t_shift=0.5,
        speed=1.0,
        return_smooth=False,
    ):
        self.calls.append(
            ("generate", text, num_steps, guidance_scale, t_shift, speed, return_smooth)
        )
        return np.ones((1, 3200), dtype=np.float32) * 0.1


class _FakeLuxTTSGenerateError(_FakeLuxTTS):
    def generate_speech(self, *args, **kwargs):
        raise RuntimeError("RAW_LUXTTS_GENERATION_SECRET_MARKER token=secret")


def _inject_luxtts(monkeypatch, holder):
    module = types.ModuleType("zipvoice.luxvoice")

    class _LuxTTSFactory(_FakeLuxTTS):
        def __init__(self, model_path=None, device="cpu", threads=2):
            super().__init__(model_path=model_path, device=device, threads=threads, holder=holder)

    module.LuxTTS = _LuxTTSFactory
    pkg = types.ModuleType("zipvoice")
    monkeypatch.setitem(sys.modules, "zipvoice", pkg)
    monkeypatch.setitem(sys.modules, "zipvoice.luxvoice", module)


def _install_luxtts(monkeypatch, fake_cls, holder=None):
    holder = holder if holder is not None else {}
    module = types.ModuleType("zipvoice.luxvoice")

    class _LuxTTSFactory(fake_cls):
        def __init__(self, model_path=None, device="cpu", threads=2):
            super().__init__(model_path=model_path, device=device, threads=threads, holder=holder)

    module.LuxTTS = _LuxTTSFactory
    pkg = types.ModuleType("zipvoice")
    monkeypatch.setitem(sys.modules, "zipvoice", pkg)
    monkeypatch.setitem(sys.modules, "zipvoice.luxvoice", module)
    return holder


def _capture_luxtts_logs(level: str = "DEBUG") -> tuple[list[str], int]:
    messages: list[str] = []
    sink_id = luxtts_mod.logger.add(
        lambda message: messages.append(
            f"{message.record['message']}\n{message.record.get('extra', {})}"
        ),
        level=level,
    )
    return messages, sink_id


_VOICE_REF = b"RIFF" + b"\x00" * 12


@pytest.mark.asyncio
async def test_luxtts_registration_failure_log_sanitizes_exception_extra(monkeypatch):
    _install_luxtts(monkeypatch, _FakeLuxTTS)
    raw_marker = "RAW_LUXTTS_REGISTRATION_SECRET_MARKER token=secret"

    async def fail_resource_manager():
        raise RuntimeError(raw_marker)

    from tldw_Server_API.app.core.TTS import tts_resource_manager

    monkeypatch.setattr(tts_resource_manager, "get_resource_manager", fail_resource_manager)
    adapter = LuxTTSAdapter(
        {
            "device": "cpu",
            "validate_reference": False,
            "convert_reference": False,
        }
    )
    messages, sink_id = _capture_luxtts_logs()

    try:
        assert await adapter.initialize() is True
    finally:
        luxtts_mod.logger.remove(sink_id)
        await adapter.close()

    rendered_logs = "\n".join(messages)
    assert "LuxTTS provider registration failed" in rendered_logs
    assert "RAW_LUXTTS_REGISTRATION_SECRET_MARKER" not in rendered_logs
    assert "token=secret" not in rendered_logs


@pytest.mark.asyncio
async def test_luxtts_generation_failure_log_sanitizes_exception_text(monkeypatch):
    _install_luxtts(monkeypatch, _FakeLuxTTSGenerateError)
    adapter = LuxTTSAdapter(
        {
            "device": "cpu",
            "validate_reference": False,
            "convert_reference": False,
        }
    )
    request = TTSRequest(
        text="fail",
        format=AudioFormat.PCM,
        voice_reference=_VOICE_REF,
        stream=False,
    )
    messages, sink_id = _capture_luxtts_logs(level="ERROR")

    try:
        with pytest.raises(TTSGenerationError) as exc_info:
            await adapter.generate(request)
    finally:
        luxtts_mod.logger.remove(sink_id)
        await adapter.close()

    assert "RAW_LUXTTS_GENERATION_SECRET_MARKER" in exc_info.value.details["error"]
    rendered_logs = "\n".join(messages)
    assert "LuxTTS generation failed" in rendered_logs
    assert "RAW_LUXTTS_GENERATION_SECRET_MARKER" not in rendered_logs
    assert "token=secret" not in rendered_logs


def test_luxtts_sample_rate_probe_log_sanitizes_exception_extra():
    raw_marker = "RAW_LUXTTS_SAMPLE_RATE_SECRET_MARKER token=secret"

    class _FailingVocos:
        @property
        def return_48k(self):
            raise RuntimeError(raw_marker)

    adapter = LuxTTSAdapter({"sample_rate": 44100})
    adapter._engine = types.SimpleNamespace(vocos=_FailingVocos())
    messages, sink_id = _capture_luxtts_logs()

    try:
        sample_rate = adapter._resolve_output_sample_rate(return_smooth=False, extras={})
    finally:
        luxtts_mod.logger.remove(sink_id)

    assert sample_rate == 44100
    rendered_logs = "\n".join(messages)
    assert "LuxTTS sample-rate probe failed" in rendered_logs
    assert "RAW_LUXTTS_SAMPLE_RATE_SECRET_MARKER" not in rendered_logs
    assert "token=secret" not in rendered_logs


@pytest.mark.asyncio
async def test_luxtts_streaming_failure_log_sanitizes_exception_text(monkeypatch):
    raw_marker = "RAW_LUXTTS_STREAMING_SECRET_MARKER token=secret"

    class _FailingStreamingAudioWriter:
        def __init__(self, *args, **kwargs):
            pass

        def write_chunk(self, *args, **kwargs):
            raise RuntimeError(raw_marker)

        def close(self):
            pass

    monkeypatch.setattr(luxtts_mod, "StreamingAudioWriter", _FailingStreamingAudioWriter)
    adapter = LuxTTSAdapter({"stream_chunk_samples": 16})
    request = TTSRequest(
        text="stream",
        format=AudioFormat.PCM,
        voice_reference=_VOICE_REF,
        stream=True,
    )
    messages, sink_id = _capture_luxtts_logs(level="ERROR")

    try:
        with pytest.raises(TTSGenerationError) as exc_info:
            async for _chunk in adapter._stream_audio(np.ones(32, dtype=np.int16), request, 48000):
                pass
    finally:
        luxtts_mod.logger.remove(sink_id)

    assert raw_marker in exc_info.value.details["error"]
    rendered_logs = "\n".join(messages)
    assert "LuxTTS streaming failed" in rendered_logs
    assert "RAW_LUXTTS_STREAMING_SECRET_MARKER" not in rendered_logs
    assert "token=secret" not in rendered_logs


def test_luxtts_tensor_conversion_log_sanitizes_exception_extra():
    raw_marker = "RAW_LUXTTS_TENSOR_CONVERSION_SECRET_MARKER token=secret"

    class _TensorLike:
        def detach(self):
            raise RuntimeError(raw_marker)

        def __array__(self, dtype=None):
            return np.asarray([0.25], dtype=dtype)

    adapter = LuxTTSAdapter({})
    messages, sink_id = _capture_luxtts_logs()

    try:
        audio_np = adapter._coerce_audio_array(_TensorLike())
    finally:
        luxtts_mod.logger.remove(sink_id)

    assert audio_np.tolist() == [0.25]
    rendered_logs = "\n".join(messages)
    assert "LuxTTS tensor conversion failed" in rendered_logs
    assert "RAW_LUXTTS_TENSOR_CONVERSION_SECRET_MARKER" not in rendered_logs
    assert "token=secret" not in rendered_logs


@pytest.mark.asyncio
async def test_luxtts_generate_non_stream(monkeypatch):
    holder = {}
    _inject_luxtts(monkeypatch, holder)

    adapter = LuxTTSAdapter(
        {
            "device": "cpu",
            "lux_tts_threads": 1,
            "sample_rate": 48000,
            "validate_reference": False,
            "convert_reference": False,
        }
    )

    request = TTSRequest(
        text="Hello LuxTTS",
        format=AudioFormat.MP3,
        voice_reference=_VOICE_REF,
        stream=False,
        extra_params={"prompt_duration": 4.0, "prompt_rms": 0.002, "num_steps": 3},
    )

    response = await adapter.generate(request)
    assert response.audio_data and isinstance(response.audio_data, (bytes, bytearray))
    assert response.audio_stream is None
    assert response.provider == "lux_tts"
    assert response.sample_rate == 48000

    instance = holder.get("instance")
    assert instance is not None
    assert instance.device == "cpu"
    assert instance.calls and instance.calls[0][0] == "encode"
    assert instance.calls[1][0] == "generate"

    await adapter.close()


@pytest.mark.asyncio
async def test_luxtts_streaming_chunks(monkeypatch):
    holder = {}
    _inject_luxtts(monkeypatch, holder)

    adapter = LuxTTSAdapter(
        {
            "device": "cpu",
            "validate_reference": False,
            "convert_reference": False,
            "stream_chunk_samples": 512,
        }
    )

    request = TTSRequest(
        text="Stream LuxTTS",
        format=AudioFormat.PCM,
        voice_reference=_VOICE_REF,
        stream=True,
        extra_params={"stream_chunk_samples": 256},
    )

    response = await adapter.generate(request)
    assert response.audio_stream is not None
    chunks = [chunk async for chunk in response.audio_stream]
    assert len(chunks) >= 2
    assert all(isinstance(c, (bytes, bytearray)) for c in chunks)

    await adapter.close()
