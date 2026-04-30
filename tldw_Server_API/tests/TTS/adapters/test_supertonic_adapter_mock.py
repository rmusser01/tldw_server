import asyncio
import sys
import types

import numpy as np
import pytest

from tldw_Server_API.app.core.TTS.adapters import supertonic_adapter as supertonic_mod
from tldw_Server_API.app.core.TTS.adapters.base import AudioFormat, TTSRequest
from tldw_Server_API.app.core.TTS.adapters.supertonic_adapter import SupertonicOnnxAdapter
from tldw_Server_API.app.core.TTS.tts_exceptions import TTSModelNotFoundError


class _DummyEngine:
    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = sample_rate
        self.calls = []

    def __call__(self, text: str, style, total_step: int, speed: float):
        self.calls.append((text, style, total_step, speed))
        wav = np.ones((1, int(self.sample_rate * 0.25)), dtype=np.float32)
        duration = np.array([0.25])
        return wav, duration


def _inject_vendor(monkeypatch, engine: _DummyEngine, call_log: dict):
    module = types.SimpleNamespace()

    def load_text_to_speech(onnx_dir: str, use_gpu: bool = False):
        call_log["onnx_dir"] = onnx_dir
        call_log["use_gpu"] = use_gpu
        return engine

    def load_voice_style(paths, verbose: bool = False):
        call_log["voice_paths"] = list(paths)
        return {"style": "dummy"}

    module.load_text_to_speech = load_text_to_speech
    module.load_voice_style = load_voice_style
    monkeypatch.setitem(sys.modules, "tldw_Server_API.app.core.TTS.vendors.supertonic", module)


def _capture_supertonic_logs(level: str = "DEBUG") -> tuple[list[str], int]:
    messages: list[str] = []
    sink_id = supertonic_mod.logger.add(
        lambda message: messages.append(
            f"{message.record['message']}\n{message.record.get('extra', {})}"
        ),
        level=level,
    )
    return messages, sink_id


@pytest.mark.asyncio
async def test_supertonic_initialize_and_generate(monkeypatch, tmp_path):
    onnx_dir = tmp_path / "onnx"
    onnx_dir.mkdir()
    voices_dir = tmp_path / "voices"
    voices_dir.mkdir()
    default_voice = voices_dir / "M1.json"
    default_voice.write_text("{}")
    alt_voice = voices_dir / "F1.json"
    alt_voice.write_text("{}")

    engine = _DummyEngine(sample_rate=24000)
    call_log = {}
    _inject_vendor(monkeypatch, engine, call_log)

    adapter = SupertonicOnnxAdapter(
        {
            "model_path": str(onnx_dir),
            "sample_rate": 24000,
            "device": "cpu",
            "extra_params": {
                "voice_styles_dir": str(voices_dir),
                "default_voice": "supertonic_m1",
                "voice_files": {
                    "supertonic_m1": "M1.json",
                    "supertonic_f1": "F1.json",
                },
                "stream_chunk_size": 16,
                "default_total_step": 7,
            },
        }
    )

    request = TTSRequest(
        text="hello world",
        voice="supertonic_m1",
        format=AudioFormat.MP3,
        speed=1.1,
        stream=False,
    )

    response = await adapter.generate(request)
    assert response.audio_data and isinstance(response.audio_data, (bytes, bytearray))
    assert response.audio_stream is None
    assert response.voice_used == "supertonic_m1"
    assert engine.calls and engine.calls[0][0] == "hello world"
    assert engine.calls[0][2] == 7  # total_step from config
    assert pytest.approx(engine.calls[0][3], rel=1e-3) == 1.1
    assert call_log["onnx_dir"] == str(onnx_dir)
    assert call_log["use_gpu"] is False
    assert call_log["voice_paths"] == [str(default_voice)]

    await adapter.close()


@pytest.mark.asyncio
async def test_supertonic_registration_failure_log_sanitizes_exception_extra(monkeypatch, tmp_path):
    raw_marker = "RAW_SUPERTONIC_REGISTER_SECRET_MARKER token=secret"
    onnx_dir = tmp_path / "onnx"
    onnx_dir.mkdir()
    voices_dir = tmp_path / "voices"
    voices_dir.mkdir()
    default_voice = voices_dir / "M1.json"
    default_voice.write_text("{}")

    engine = _DummyEngine(sample_rate=24000)
    call_log = {}
    _inject_vendor(monkeypatch, engine, call_log)

    class FailingResourceManager:
        def register_model(self, **kwargs):
            raise RuntimeError(raw_marker)

    async def fake_get_resource_manager():
        return FailingResourceManager()

    from tldw_Server_API.app.core.TTS import tts_resource_manager

    monkeypatch.setattr(tts_resource_manager, "get_resource_manager", fake_get_resource_manager)
    adapter = SupertonicOnnxAdapter(
        {
            "model_path": str(onnx_dir),
            "sample_rate": 24000,
            "extra_params": {
                "voice_styles_dir": str(voices_dir),
                "default_voice": "supertonic_m1",
                "voice_files": {
                    "supertonic_m1": "M1.json",
                },
            },
        }
    )
    messages, sink_id = _capture_supertonic_logs()

    try:
        assert await adapter.initialize() is True
    finally:
        supertonic_mod.logger.remove(sink_id)
        await adapter.close()

    rendered_logs = "\n".join(messages)
    assert "Supertonic provider registration failed" in rendered_logs
    assert "RAW_SUPERTONIC_REGISTER_SECRET_MARKER" not in rendered_logs
    assert "token=secret" not in rendered_logs


@pytest.mark.asyncio
async def test_supertonic_streaming_chunks(monkeypatch, tmp_path):
    onnx_dir = tmp_path / "onnx"
    onnx_dir.mkdir()
    voices_dir = tmp_path / "voices"
    voices_dir.mkdir()
    default_voice = voices_dir / "M1.json"
    default_voice.write_text("{}")

    engine = _DummyEngine(sample_rate=24000)
    call_log = {}
    _inject_vendor(monkeypatch, engine, call_log)

    adapter = SupertonicOnnxAdapter(
        {
            "model_path": str(onnx_dir),
            "sample_rate": 24000,
            "extra_params": {
                "voice_styles_dir": str(voices_dir),
                "default_voice": "supertonic_m1",
                "voice_files": {
                    "supertonic_m1": "M1.json",
                },
                "stream_chunk_size": 8,
            },
        }
    )

    request = TTSRequest(
        text="stream me",
        voice="supertonic_m1",
        format=AudioFormat.WAV,
        speed=1.0,
        stream=True,
    )

    response = await adapter.generate(request)
    assert response.audio_stream is not None
    chunks = [chunk async for chunk in response.audio_stream]
    assert len(chunks) >= 2
    assert all(isinstance(c, (bytes, bytearray)) for c in chunks)

    await adapter.close()


def test_supertonic_audio_trim_fallback_log_sanitizes_exception_extra():
    raw_marker = "RAW_SUPERTONIC_TRIM_SECRET_MARKER token=secret"

    class BadDuration:
        def __float__(self):
            raise ValueError(raw_marker)

    adapter = SupertonicOnnxAdapter(config={"sample_rate": 24000})
    wav = np.arange(10, dtype=np.float32)
    messages, sink_id = _capture_supertonic_logs()

    try:
        audio = adapter._prepare_audio_array(wav, BadDuration())
    finally:
        supertonic_mod.logger.remove(sink_id)

    np.testing.assert_array_equal(audio, wav)
    rendered_logs = "\n".join(messages)
    assert "Supertonic audio trim by duration failed" in rendered_logs
    assert "RAW_SUPERTONIC_TRIM_SECRET_MARKER" not in rendered_logs
    assert "token=secret" not in rendered_logs


@pytest.mark.asyncio
async def test_supertonic_missing_default_voice(monkeypatch, tmp_path):
    onnx_dir = tmp_path / "onnx"
    onnx_dir.mkdir()
    voices_dir = tmp_path / "voices"
    voices_dir.mkdir()

    engine = _DummyEngine(sample_rate=24000)
    call_log = {}
    _inject_vendor(monkeypatch, engine, call_log)

    adapter = SupertonicOnnxAdapter(
        {
            "model_path": str(onnx_dir),
            "sample_rate": 24000,
            "extra_params": {
                "voice_styles_dir": str(voices_dir),
                "default_voice": "supertonic_m1",
                "voice_files": {
                    "supertonic_m1": "missing.json",
                },
            },
        }
    )

    with pytest.raises(TTSModelNotFoundError):
        await adapter.ensure_initialized()
