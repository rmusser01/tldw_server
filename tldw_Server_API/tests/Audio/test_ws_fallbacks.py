import asyncio
import importlib.machinery
import json
from configparser import ConfigParser
import sys
import types
import numpy as np
import pytest

# Stub heavyweight audio deps before importing unified streaming modules.
if "torch" not in sys.modules:
    _fake_torch = types.ModuleType("torch")
    _fake_torch.__spec__ = importlib.machinery.ModuleSpec("torch", loader=None)
    _fake_torch.Tensor = object
    _fake_torch.nn = types.SimpleNamespace(Module=object)
    _fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    sys.modules["torch"] = _fake_torch

if "faster_whisper" not in sys.modules:
    _fake_fw = types.ModuleType("faster_whisper")
    _fake_fw.__spec__ = importlib.machinery.ModuleSpec("faster_whisper", loader=None)

    class _StubWhisperModel:
        def __init__(self, *args, **kwargs):
            pass

    _fake_fw.WhisperModel = _StubWhisperModel
    _fake_fw.BatchedInferencePipeline = _StubWhisperModel
    sys.modules["faster_whisper"] = _fake_fw

if "transformers" not in sys.modules:
    _fake_tf = types.ModuleType("transformers")
    _fake_tf.__spec__ = importlib.machinery.ModuleSpec("transformers", loader=None)

    class _StubProcessor:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    class _StubModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    _fake_tf.AutoProcessor = _StubProcessor
    _fake_tf.Qwen2AudioForConditionalGeneration = _StubModel
    sys.modules["transformers"] = _fake_tf


class _DummyWebSocket:
    def __init__(self, frames):
        """
        Construct a dummy WebSocket preloaded with incoming frames for tests.

        Parameters:
            frames (iterable): An iterable of frames (typically strings) that will be copied into an internal queue and returned one-by-one by receive_text().
        """
        self._frames = list(frames)
        self.sent = []
        self.closed = False
        self.close_args = None

    async def receive_text(self):
        """
        Return the next queued text frame from the mock WebSocket.

        If no frames remain, simulates a client timeout by raising asyncio.TimeoutError.

        Returns:
            str: The next text frame.

        Raises:
            asyncio.TimeoutError: When there are no queued frames to deliver.
        """
        if not self._frames:
            # Simulate client gone
            await asyncio.sleep(0)
            raise asyncio.TimeoutError()
        return self._frames.pop(0)

    async def send_json(self, payload):
        """
        Record an outgoing JSON payload for the mock WebSocket.

        Parameters:
            payload: The JSON-serializable object to record; appended to the mock's `sent` list.
        """
        self.sent.append(payload)

    async def close(self, code: int | None = None, reason: str | None = None):
        """
        Mark the websocket as closed and record the close code and reason.

        Parameters:
            code (int | None): Optional numeric close code.
            reason (str | None): Optional human-readable reason for the close.
        """
        self.closed = True
        self.close_args = (code, reason)


def _make_cfg(fallback: bool) -> ConfigParser:
    """
    Create a ConfigParser with the "STT-Settings" section and the streaming_fallback_to_whisper flag.

    Parameters:
        fallback (bool): If True, sets "streaming_fallback_to_whisper" to 'true'; otherwise sets it to 'false'.

    Returns:
        ConfigParser: A parser containing the "STT-Settings" section with the configured "streaming_fallback_to_whisper" value.
    """
    cfg = ConfigParser()
    cfg.add_section('STT-Settings')
    cfg.set('STT-Settings', 'streaming_fallback_to_whisper', 'true' if fallback else 'false')
    return cfg


def _cfg_without_stt_section() -> ConfigParser:
    """Return an empty config parser (no STT-Settings section)."""
    return ConfigParser()


class _FakeWhisperModel:
    class _Seg:
        def __init__(self, t: str):
            """
            Initialize the segment with its transcribed text.

            Parameters:
                t (str): The transcribed text to store on the segment as `text`.
            """
            self.text = t

    class _Info:
        language = 'en'
        language_probability = 1.0

    def transcribe(self, path: str, **opts):
        # Return shape compatible with code: (segments, info)
        """
        Provide a minimal, test-only transcription result compatible with the expected (segments, info) shape.

        Parameters:
            path (str): Path to the audio file to transcribe (ignored by this fake implementation).
            **opts: Additional transcription options accepted for API compatibility (ignored).

        Returns:
            tuple: A pair (segments, info) where `segments` is a list containing a single object with a `text` attribute equal to `"ok"`, and `info` is an object with `language` set to `'en'` and `language_probability` set to `1.0`.
        """
        return [self._Seg("ok")], self._Info()


@pytest.mark.asyncio
async def test_model_unavailable_triggers_fallback_warning(monkeypatch):
    # Force Parakeet core variant builder to return None so adapter initialize fails
    """
    Verify that when the primary transcription model is unavailable and fallback to Whisper is enabled, the websocket handler emits at least one warning frame indicating a fallback.

    Sets up the environment so the primary model initialization fails, enables Whisper fallback, provides a fake Whisper model, and sends a config then stop frame to the handler; asserts that at least one sent message has type "warning" and that a warning with "fallback" == True is present.
    """
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Parakeet_Core_Streaming.transcriber as core_tx
    monkeypatch.setattr(core_tx, "_variant_decode_fn", lambda m, v: None)

    # Enable fallback to Whisper
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Unified as unified
    monkeypatch.setattr(unified, "load_comprehensive_config", lambda: _make_cfg(True))
    monkeypatch.setattr(unified, "get_whisper_model", lambda size, device: _FakeWhisperModel())

    # Prepare websocket with config (parakeet-onnx) then stop
    cfg = json.dumps({"type": "config", "model": "parakeet-onnx", "sample_rate": 16000})
    stop = json.dumps({"type": "stop"})
    ws = _DummyWebSocket([cfg, stop])

    # Run handler
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Unified import (
        handle_unified_websocket, UnifiedStreamingConfig
    )

    await handle_unified_websocket(ws, UnifiedStreamingConfig())

    warnings = [m for m in ws.sent if m.get("type") == "warning"]
    assert warnings, "Expected at least one warning frame"
    # Look for fallback notice
    fallback_msgs = [w for w in warnings if w.get("fallback") is True]
    assert fallback_msgs, "Fallback to Whisper warning not emitted"


@pytest.mark.asyncio
async def test_model_unavailable_without_fallback_emits_error(monkeypatch):
    # Force Parakeet core variant builder to return None so adapter initialize fails
    """
    Verify that when the primary transcription model is unavailable and Whisper fallback is disabled, the websocket handler emits an error frame indicating `model_unavailable`.

    This test patches the core variant decoder to force adapter initialization failure, disables streaming fallback to Whisper, sends a config and stop message via a dummy websocket, runs the unified websocket handler, and asserts that at least one sent frame has `"type": "error"` and an `"error_type"` equal to `"model_unavailable"`.
    """
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Parakeet_Core_Streaming.transcriber as core_tx
    monkeypatch.setattr(core_tx, "_variant_decode_fn", lambda m, v: None)

    # Disable fallback to Whisper
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Unified as unified
    monkeypatch.setattr(unified, "load_comprehensive_config", lambda: _make_cfg(False))

    cfg = json.dumps({"type": "config", "model": "parakeet-onnx", "sample_rate": 16000})
    stop = json.dumps({"type": "stop"})
    ws = _DummyWebSocket([cfg, stop])

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Unified import (
        handle_unified_websocket, UnifiedStreamingConfig
    )

    await handle_unified_websocket(ws, UnifiedStreamingConfig())

    errors = [m for m in ws.sent if m.get("type") == "error"]
    assert errors, "Expected an error frame when fallback disabled"
    assert any(e.get("error_type") == "model_unavailable" for e in errors)


@pytest.mark.asyncio
async def test_model_unavailable_without_fallback_sanitizes_internal_error(monkeypatch):
    """Model initialization failures should not expose backend exception details."""
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Unified as unified

    class _FailingTranscriber:
        def __init__(self, _config):
            pass

        def initialize(self):
            raise RuntimeError("streaming model exploded at /private/audio/model.bin")

        def cleanup(self):
            pass

    monkeypatch.setattr(unified, "UnifiedStreamingTranscriber", _FailingTranscriber)
    monkeypatch.setattr(unified, "load_comprehensive_config", lambda: _make_cfg(False))

    cfg = json.dumps({"type": "config", "model": "parakeet-onnx", "sample_rate": 16000})
    stop = json.dumps({"type": "stop"})
    ws = _DummyWebSocket([cfg, stop])

    await unified.handle_unified_websocket(ws, unified.UnifiedStreamingConfig())

    errors = [m for m in ws.sent if m.get("type") == "error"]
    warnings = [m for m in ws.sent if m.get("type") == "warning"]
    model_warnings = [w for w in warnings if w.get("error_type") == "model_unavailable"]
    assert errors
    assert model_warnings
    assert any(e.get("error_type") == "model_unavailable" for e in errors)
    assert "streaming model exploded" not in str(ws.sent)
    assert "/private/audio/model.bin" not in str(ws.sent)
    assert model_warnings[0]["details"]["error"] == "Streaming model initialization failed"
    assert errors[-1]["data"]["error"] == "Streaming model initialization failed"


@pytest.mark.asyncio
async def test_model_fallback_failure_sanitizes_internal_errors(monkeypatch):
    """Fallback model failures should not expose either backend exception."""
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Unified as unified

    class _FailingTranscriber:
        def __init__(self, config):
            self._model = config.model

        def initialize(self):
            if self._model == "whisper":
                raise RuntimeError("fallback whisper exploded at /private/audio/whisper.bin")
            raise RuntimeError("primary model exploded at /private/audio/parakeet.onnx")

        def cleanup(self):
            pass

    monkeypatch.setattr(unified, "UnifiedStreamingTranscriber", _FailingTranscriber)
    monkeypatch.setattr(unified, "load_comprehensive_config", lambda: _make_cfg(True))

    cfg = json.dumps({"type": "config", "model": "parakeet-onnx", "sample_rate": 16000})
    stop = json.dumps({"type": "stop"})
    ws = _DummyWebSocket([cfg, stop])

    await unified.handle_unified_websocket(ws, unified.UnifiedStreamingConfig())

    errors = [m for m in ws.sent if m.get("type") == "error"]
    warnings = [m for m in ws.sent if m.get("type") == "warning"]
    model_warnings = [w for w in warnings if w.get("error_type") == "model_unavailable"]
    assert errors
    assert model_warnings
    assert any(e.get("error_type") == "provider_error" for e in errors)
    assert "primary model exploded" not in str(ws.sent)
    assert "fallback whisper exploded" not in str(ws.sent)
    assert "/private/audio/parakeet.onnx" not in str(ws.sent)
    assert "/private/audio/whisper.bin" not in str(ws.sent)
    assert model_warnings[0]["details"]["error"] == "Streaming model initialization failed"
    assert errors[-1]["data"]["original_error"] == "Streaming model initialization failed"
    assert errors[-1]["data"]["fallback_error"] == "Streaming fallback initialization failed"


@pytest.mark.asyncio
async def test_stt_error_sentinel_sanitizes_raw_error_payload(monkeypatch):
    """STT error sentinels should not expose backend text in error frame data."""
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Unified as unified

    class _SentinelTranscriber:
        def __init__(self, _config):
            pass

        def initialize(self):
            pass

        async def process_audio_chunk(self, _audio_bytes):
            return {
                "text": "provider crashed while reading /private/audio/stt-model.bin",
                "is_final": False,
            }

        def get_full_transcript(self):
            return ""

        def reset(self):
            pass

        def cleanup(self):
            pass

    monkeypatch.setattr(unified, "UnifiedStreamingTranscriber", _SentinelTranscriber)
    monkeypatch.setattr(unified, "_is_transcription_error_message", lambda _text: True)

    cfg = json.dumps({"type": "config", "model": "whisper", "sample_rate": 16000})
    audio = json.dumps({"type": "audio", "data": "AA=="})
    ws = _DummyWebSocket([cfg, audio])

    await unified.handle_unified_websocket(ws, unified.UnifiedStreamingConfig())

    errors = [m for m in ws.sent if m.get("type") == "error"]
    assert errors
    assert errors[-1]["error_type"] == "provider_error"
    assert "provider crashed" not in str(ws.sent)
    assert "/private/audio/stt-model.bin" not in str(ws.sent)
    assert errors[-1]["data"]["raw_error"] == "Transcription provider returned an error"


@pytest.mark.asyncio
async def test_audio_processing_error_sanitizes_internal_error(monkeypatch):
    """Audio-frame processing failures should not expose backend exception text."""
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Unified as unified

    class _FailingChunkTranscriber:
        def __init__(self, _config):
            pass

        def initialize(self):
            pass

        async def process_audio_chunk(self, _audio_bytes):
            raise RuntimeError("chunk processor exploded at /private/audio/chunk.wav")

        def get_full_transcript(self):
            return ""

        def reset(self):
            pass

        def cleanup(self):
            pass

    monkeypatch.setattr(unified, "UnifiedStreamingTranscriber", _FailingChunkTranscriber)

    cfg = json.dumps({"type": "config", "model": "whisper", "sample_rate": 16000})
    audio = json.dumps({"type": "audio", "data": "AA=="})
    stop = json.dumps({"type": "stop"})
    ws = _DummyWebSocket([cfg, audio, stop])

    await unified.handle_unified_websocket(ws, unified.UnifiedStreamingConfig())

    errors = [m for m in ws.sent if m.get("type") == "error"]
    assert errors
    assert errors[0]["error_type"] == "internal_error"
    assert "chunk processor exploded" not in str(ws.sent)
    assert "/private/audio/chunk.wav" not in str(ws.sent)
    assert errors[0]["message"] == "Streaming audio processing failed"


@pytest.mark.asyncio
async def test_outer_websocket_handler_error_sanitizes_internal_error(monkeypatch):
    """Outer websocket failures should not expose backend exception text."""
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Unified as unified

    class _FailingControlSession:
        def __init__(self, _config):
            pass

        def apply_config(self, _config_payload):
            raise RuntimeError("control session exploded at /private/audio/control.db")

    monkeypatch.setattr(unified, "WSControlSession", _FailingControlSession)

    cfg = json.dumps({"type": "config", "model": "whisper", "sample_rate": 16000})
    ws = _DummyWebSocket([cfg])

    await unified.handle_unified_websocket(ws, unified.UnifiedStreamingConfig())

    errors = [m for m in ws.sent if m.get("type") == "error"]
    assert errors
    assert errors[-1]["error_type"] == "internal_error"
    assert "control session exploded" not in str(ws.sent)
    assert "/private/audio/control.db" not in str(ws.sent)
    assert errors[-1]["message"] == "Streaming server error"


@pytest.mark.asyncio
async def test_diarization_initialization_warning_sanitizes_details(monkeypatch):
    """Diarization init failures should not expose backend exception details."""
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Unified as unified

    class _ReadyTranscriber:
        def __init__(self, _config):
            pass

        def initialize(self):
            pass

        def get_full_transcript(self):
            return ""

        def reset(self):
            pass

        def cleanup(self):
            pass

    class _FailingDiarizer:
        def __init__(self, *_args, **_kwargs):
            pass

        async def ensure_ready(self):
            raise RuntimeError("diarizer exploded at /private/audio/diarizer.bin")

    monkeypatch.setattr(unified, "UnifiedStreamingTranscriber", _ReadyTranscriber)
    monkeypatch.setattr(unified, "StreamingDiarizer", _FailingDiarizer)

    cfg = json.dumps({
        "type": "config",
        "model": "whisper",
        "sample_rate": 16000,
        "diarization": {"enabled": True},
    })
    stop = json.dumps({"type": "stop"})
    ws = _DummyWebSocket([cfg, stop])

    await unified.handle_unified_websocket(ws, unified.UnifiedStreamingConfig())

    warnings = [
        m for m in ws.sent
        if m.get("type") == "warning" and m.get("state") == "diarization_unavailable"
    ]
    assert warnings
    assert "diarizer exploded" not in str(ws.sent)
    assert "/private/audio/diarizer.bin" not in str(ws.sent)
    assert warnings[-1]["details"] == "Diarization initialization failed"


@pytest.mark.asyncio
async def test_live_insights_initialization_warning_sanitizes_details(monkeypatch):
    """Live-insights init failures should not expose backend exception details."""
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Unified as unified

    class _ReadyTranscriber:
        def __init__(self, _config):
            pass

        def initialize(self):
            pass

        def get_full_transcript(self):
            return ""

        def reset(self):
            pass

        def cleanup(self):
            pass

    class _FailingLiveInsights:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("insights exploded at /private/audio/insights.db")

    monkeypatch.setattr(unified, "UnifiedStreamingTranscriber", _ReadyTranscriber)
    monkeypatch.setattr(unified, "LiveMeetingInsights", _FailingLiveInsights)

    cfg = json.dumps({
        "type": "config",
        "model": "whisper",
        "sample_rate": 16000,
        "insights_enabled": True,
    })
    stop = json.dumps({"type": "stop"})
    ws = _DummyWebSocket([cfg, stop])

    await unified.handle_unified_websocket(ws, unified.UnifiedStreamingConfig())

    warnings = [
        m for m in ws.sent
        if m.get("type") == "warning" and m.get("state") == "insights_unavailable"
    ]
    assert warnings
    assert "insights exploded" not in str(ws.sent)
    assert "/private/audio/insights.db" not in str(ws.sent)
    assert warnings[-1]["details"] == "Live insights initialization failed"


@pytest.mark.asyncio
async def test_model_unavailable_defaults_to_no_fallback_when_stt_section_missing(monkeypatch):
    """Missing STT-Settings should default to fail-fast (no Whisper fallback)."""
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Parakeet_Core_Streaming.transcriber as core_tx
    monkeypatch.setattr(core_tx, "_variant_decode_fn", lambda m, v: None)

    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Unified as unified
    monkeypatch.setattr(unified, "load_comprehensive_config", _cfg_without_stt_section)
    monkeypatch.setattr(
        unified,
        "get_whisper_model",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Whisper fallback should be disabled by default")),
    )

    cfg = json.dumps({"type": "config", "model": "parakeet-onnx", "sample_rate": 16000})
    stop = json.dumps({"type": "stop"})
    ws = _DummyWebSocket([cfg, stop])

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Unified import (
        UnifiedStreamingConfig,
        handle_unified_websocket,
    )

    await handle_unified_websocket(ws, UnifiedStreamingConfig())

    errors = [m for m in ws.sent if m.get("type") == "error"]
    assert errors, "Expected fail-fast error when STT fallback key is absent"
    assert any(e.get("error_type") == "model_unavailable" for e in errors)
    assert not any(m.get("fallback") is True for m in ws.sent)
