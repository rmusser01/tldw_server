import asyncio
import base64
import json
import time
import wave
from pathlib import Path

import numpy as np
import pytest


class _DummyWebSocket:
    def __init__(self, frames, delays=None):
        self._frames = list(frames)
        self.sent = []
        self.closed = False
        self.close_args = None
        self._delays = list(delays or [])

    async def receive_text(self):
        if not self._frames:
            await asyncio.sleep(0)
            raise asyncio.TimeoutError()
        if self._delays:
            await asyncio.sleep(self._delays.pop(0))
        return self._frames.pop(0)

    async def send_json(self, payload):
        self.sent.append(payload)

    async def close(self, code: int | None = None, reason: str | None = None):
        self.closed = True
        self.close_args = (code, reason)


@pytest.mark.asyncio
async def test_vad_auto_commit_triggers_full_transcript(monkeypatch):
    """Auto-commit should emit a full_transcript frame when VAD signals EOS."""
    class _StubTranscriber:
        def __init__(self, config):
            self.config = config

        def initialize(self):
            return None

        async def process_audio_chunk(self, _audio_bytes: bytes):
            return {"type": "partial", "text": "hi", "timestamp": time.time(), "is_final": False}

        def get_full_transcript(self):
            return "hello world"

        def reset(self):
            return None

        def cleanup(self):
            return None

    class _StubTurnDetector:
        def __init__(self, *args, **kwargs):
            self.available = True
            self.unavailable_reason = None
            self._count = 0
            self._last_trigger_at = None

        @property
        def last_trigger_at(self):
            return self._last_trigger_at

        def observe(self, _audio_bytes: bytes) -> bool:
            self._count += 1
            if self._count >= 2:
                self._last_trigger_at = 1234.5
                return True
            return False

    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Unified as unified

    monkeypatch.setattr(unified, "UnifiedStreamingTranscriber", _StubTranscriber)
    monkeypatch.setattr(unified, "SileroTurnDetector", _StubTurnDetector)

    cfg = json.dumps({"type": "config", "model": "parakeet", "sample_rate": 16000, "enable_vad": True})
    audio_frame = json.dumps({"type": "audio", "data": base64.b64encode(b'1234').decode("ascii")})
    stop = json.dumps({"type": "stop"})
    ws = _DummyWebSocket([cfg, audio_frame, audio_frame, stop])

    await unified.handle_unified_websocket(ws, unified.UnifiedStreamingConfig())

    full_transcripts = [m for m in ws.sent if m.get("type") == "full_transcript"]
    assert full_transcripts, f"Expected a full_transcript frame, saw {ws.sent}"
    assert full_transcripts[0].get("auto_commit") is True
    assert full_transcripts[0].get("vad_status") == "enabled"
    assert full_transcripts[0].get("diarization_status") == "disabled"
    assert full_transcripts[0].get("text") == "hello world"
    assert full_transcripts[0].get("voice_to_voice_start") == pytest.approx(1234.5)


@pytest.mark.asyncio
async def test_vad_fail_open_disables_auto_commit(monkeypatch):
    """When VAD is unavailable, the stream should continue without auto-commit."""
    class _StubTranscriber:
        def __init__(self, config):
            self.config = config

        def initialize(self):
            return None

        async def process_audio_chunk(self, _audio_bytes: bytes):
            return {"type": "partial", "text": "hi", "timestamp": time.time(), "is_final": False}

        def get_full_transcript(self):
            return "manual_after_fail_open"

        def reset(self):
            return None

        def cleanup(self):
            return None

    class _UnavailableVAD:
        def __init__(self, *args, **kwargs):
            self.available = False
            self.unavailable_reason = "no_silero"

        def observe(self, _audio_bytes: bytes):
            raise AssertionError("observe should not be called when VAD is unavailable")

    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Unified as unified

    monkeypatch.setattr(unified, "UnifiedStreamingTranscriber", _StubTranscriber)
    monkeypatch.setattr(unified, "SileroTurnDetector", _UnavailableVAD)

    cfg = json.dumps({"type": "config", "model": "parakeet", "sample_rate": 16000, "enable_vad": True})
    audio_frame = json.dumps({"type": "audio", "data": base64.b64encode(b'1234').decode("ascii")})
    commit = json.dumps({"type": "commit"})
    stop = json.dumps({"type": "stop"})
    ws = _DummyWebSocket([cfg, audio_frame, commit, stop])

    await unified.handle_unified_websocket(ws, unified.UnifiedStreamingConfig())

    full_transcripts = [m for m in ws.sent if m.get("type") == "full_transcript"]
    assert full_transcripts, f"Expected a full_transcript frame, saw {ws.sent}"
    assert full_transcripts[0].get("auto_commit") is False
    assert full_transcripts[0].get("vad_status") == "fail_open"
    assert full_transcripts[0].get("diarization_status") == "disabled"
    assert full_transcripts[0].get("text") == "manual_after_fail_open"


@pytest.mark.asyncio
async def test_manual_commit_with_vad_disabled_sets_deterministic_diagnostics(monkeypatch):
    class _StubTranscriber:
        def __init__(self, config):
            self.config = config

        def initialize(self):
            return None

        async def process_audio_chunk(self, _audio_bytes: bytes):
            return {"type": "partial", "text": "hi", "timestamp": time.time(), "is_final": False}

        def get_full_transcript(self):
            return "manual_vad_disabled"

        def reset(self):
            return None

        def cleanup(self):
            return None

    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Unified as unified

    monkeypatch.setattr(unified, "UnifiedStreamingTranscriber", _StubTranscriber)

    cfg = json.dumps({"type": "config", "model": "parakeet", "sample_rate": 16000, "enable_vad": False})
    audio_frame = json.dumps({"type": "audio", "data": base64.b64encode(b'1234').decode("ascii")})
    commit = json.dumps({"type": "commit"})
    stop = json.dumps({"type": "stop"})
    ws = _DummyWebSocket([cfg, audio_frame, commit, stop])

    await unified.handle_unified_websocket(ws, unified.UnifiedStreamingConfig())

    full_transcripts = [m for m in ws.sent if m.get("type") == "full_transcript"]
    assert full_transcripts, f"Expected a full_transcript frame, saw {ws.sent}"
    assert full_transcripts[0].get("auto_commit") is False
    assert full_transcripts[0].get("vad_status") == "disabled"
    assert full_transcripts[0].get("diarization_status") == "disabled"
    assert full_transcripts[0].get("text") == "manual_vad_disabled"


@pytest.mark.asyncio
async def test_vad_auto_commit_records_latency_metric(monkeypatch):
    """Auto-commit should record stt_final_latency_seconds with endpoint label."""
    class _StubTranscriber:
        def __init__(self, config):
            self.config = config

        def initialize(self):
            return None

        async def process_audio_chunk(self, _audio_bytes: bytes):
            return {"type": "partial", "text": "hi", "timestamp": time.time(), "is_final": False}

        def get_full_transcript(self):
            return "hello world"

        def reset(self):
            return None

        def cleanup(self):
            return None

    class _StubTurnDetector:
        def __init__(self, *args, **kwargs):
            self.available = True
            self.unavailable_reason = None
            self._count = 0
            self._last_trigger_at = None

        @property
        def last_trigger_at(self):
            return self._last_trigger_at

        def observe(self, _audio_bytes: bytes) -> bool:
            self._count += 1
            if self._count >= 2:
                # Pretend speech stopped just before this frame
                self._last_trigger_at = time.time()
                return True
            return False

    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Unified as unified
    from tldw_Server_API.app.core.Metrics.metrics_manager import get_metrics_registry

    reg = get_metrics_registry()
    reg.values["stt_final_latency_seconds"].clear()

    monkeypatch.setattr(unified, "UnifiedStreamingTranscriber", _StubTranscriber)
    monkeypatch.setattr(unified, "SileroTurnDetector", _StubTurnDetector)

    cfg = json.dumps({"type": "config", "model": "parakeet", "sample_rate": 16000, "enable_vad": True})
    audio_frame = json.dumps({"type": "audio", "data": base64.b64encode(b'1234').decode("ascii")})
    stop = json.dumps({"type": "stop"})
    ws = _DummyWebSocket([cfg, audio_frame, audio_frame, stop])

    await unified.handle_unified_websocket(ws, unified.UnifiedStreamingConfig())

    full_transcripts = [m for m in ws.sent if m.get("type") == "full_transcript"]
    assert len(full_transcripts) == 1
    values = list(reg.values.get("stt_final_latency_seconds", []))
    assert values, "Expected stt_final_latency_seconds metric to be recorded"
    latest = values[-1]
    assert latest.value < 0.5, f"Expected latency <0.5s, got {latest.value}"
    assert latest.labels.get("endpoint") == "audio_unified_ws"


def test_silero_turn_detector_triggers_after_silence(monkeypatch):


    """SileroTurnDetector should fire once speech is followed by configured silence."""
    class _FakeVADIterator:
        def __init__(self, *args, **kwargs):
            self.calls = 0

        def reset_states(self):

            self.calls = 0

        def __call__(self, _audio_in, return_seconds=False, **_kwargs):

            self.calls += 1
            if self.calls == 1:
                return {"speech_timestamps": [{"start": 0, "end": 100}]}
            return {}

    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.VAD_Lib as vlib
    monkeypatch.setattr(vlib, "_lazy_import_silero_vad", lambda: ("model", [None, None, None, _FakeVADIterator, None]))

    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Unified as unified

    detector = unified.SileroTurnDetector(
        sample_rate=16000,
        enabled=True,
        vad_threshold=0.5,
        min_silence_ms=200,
        turn_stop_secs=0.05,
        min_utterance_secs=0.0,
    )
    assert detector.available

    # First chunk marks speech
    assert detector.observe(b"\x00" * 160) is False
    # Wait beyond turn_stop_secs to simulate silence
    time.sleep(0.06)
    assert detector.observe(b"\x00" * 160) is True


def test_silero_turn_detector_honors_min_utterance(monkeypatch):


    """Auto-commit should not fire when speech duration is below min_utterance_secs."""
    class _FakeVADIterator:
        def __init__(self, *args, **kwargs):
            self.calls = 0

        def reset_states(self):

            self.calls = 0

        def __call__(self, _audio_in, return_seconds=False, **_kwargs):

            self.calls += 1
            if self.calls == 1:
                return {"speech_timestamps": [{"start": 0, "end": 20}]}
            return {}

    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.VAD_Lib as vlib
    monkeypatch.setattr(vlib, "_lazy_import_silero_vad", lambda: ("model", [None, None, None, _FakeVADIterator, None]))

    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Unified as unified

    detector = unified.SileroTurnDetector(
        sample_rate=16000,
        enabled=True,
        vad_threshold=0.5,
        min_silence_ms=200,
        turn_stop_secs=0.05,
        min_utterance_secs=0.5,
    )
    assert detector.available

    assert detector.observe(b"\x00" * 160) is False  # speech observed
    time.sleep(0.06)  # silence shorter than min_utterance guard
    assert detector.observe(b"\x00" * 160) is False
    time.sleep(0.5)  # now above min_utterance
    assert detector.observe(b"\x00" * 160) is True


def test_silero_turn_detector_real_vad_end_to_end():


    """
    Exercise SileroTurnDetector with real Silero VAD (no stubs) on a sample WAV plus trailing silence.

    Skips when Silero VAD is not available locally.
    """
    try:
        from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.VAD_Lib import _lazy_import_silero_vad

        model, utils = _lazy_import_silero_vad()
    except Exception as err:  # pragma: no cover - depends on local deps/cache
        pytest.skip(f"Silero VAD unavailable: {err}")

    if not model or not utils or len(utils) < 4:
        pytest.skip("Silero VAD unavailable")

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Unified import SileroTurnDetector

    # Use the shipped sample audio and append silence to trigger EOS
    wav_path = Path("tldw_Server_API/tests/Media_Ingestion_Modification/test_media/sample.wav")
    with wave.open(str(wav_path), "rb") as wf:
        data = wf.readframes(wf.getnframes())
        sr = wf.getframerate()
    # Normalize int16 to float32 [-1, 1]
    audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    silence = np.zeros(int(0.4 * sr), dtype=np.float32)
    audio = np.concatenate([audio, silence])

    detector = SileroTurnDetector(
        sample_rate=sr,
        enabled=True,
        vad_threshold=0.5,
        min_silence_ms=200,
        turn_stop_secs=0.2,
        min_utterance_secs=0.2,
    )
    if not detector.available:
        pytest.skip(f"Silero VAD not initialized: {detector.unavailable_reason}")

    frame_size = int(0.1 * sr)
    triggered = False
    for i in range(0, len(audio), frame_size):
        chunk = audio[i : i + frame_size].astype(np.float32).tobytes()
        if detector.observe(chunk):
            triggered = True
            break

    assert triggered, "Expected SileroTurnDetector to trigger on real VAD with trailing silence"


def test_silero_turn_detector_logs_fail_open(monkeypatch):


    """
    When Silero VAD cannot be initialized, we should log a warning and continue without auto-commit.
    """
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.VAD_Lib as vlib
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Unified as unified

    def _raise_import_error():

        raise ImportError("silero missing")

    monkeypatch.setattr(vlib, "_lazy_import_silero_vad", _raise_import_error)
    captured_warnings = []

    def _fake_warning(msg, *_args, **_kwargs):

        try:
            captured_warnings.append(msg.format(*_args))
        except (IndexError, KeyError, ValueError):
            captured_warnings.append(str(msg))

    monkeypatch.setattr(unified.logger, "warning", _fake_warning)

    detector = unified.SileroTurnDetector(
        sample_rate=16000,
        enabled=True,
        vad_threshold=0.5,
        min_silence_ms=200,
        turn_stop_secs=0.1,
        min_utterance_secs=0.2,
    )

    assert detector.available is False
    assert detector.unavailable_reason
    assert any("Silero VAD" in msg and "continuing without auto-commit" in msg for msg in captured_warnings)


def test_silero_turn_detector_surfaces_specific_loader_reason(monkeypatch):
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.VAD_Lib as vlib
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Unified as unified

    monkeypatch.setattr(vlib, "_lazy_import_silero_vad", lambda: (None, None))
    monkeypatch.setattr(
        vlib,
        "get_silero_vad_unavailable_reason",
        lambda: "missing_dependency: torchaudio",
        raising=False,
    )

    detector = unified.SileroTurnDetector(
        sample_rate=16000,
        enabled=True,
        vad_threshold=0.5,
        min_silence_ms=200,
        turn_stop_secs=0.1,
        min_utterance_secs=0.2,
    )

    assert detector.available is False
    assert detector.unavailable_reason == "missing_dependency: torchaudio"


@pytest.mark.asyncio
async def test_ws_streaming_pauses_emit_single_final(monkeypatch):
    """
    With VAD enabled by default, a stream with a pause should emit exactly one final quickly.
    """

    class _StubTranscriber:
        def __init__(self, config):
            self.config = config

        def initialize(self):
            return None

        async def process_audio_chunk(self, _audio_bytes: bytes):
            return {"type": "partial", "text": "hi", "timestamp": time.time(), "is_final": False}

        def get_full_transcript(self):
            return "pause-final"

        def reset(self):
            return None

        def cleanup(self):
            return None

    class _StubTurnDetector:
        def __init__(self, *args, **kwargs):
            self.available = True
            self.unavailable_reason = None
            self._last_trigger_at = None
            self._seen = 0

        @property
        def last_trigger_at(self):
            return self._last_trigger_at

        def observe(self, _audio_bytes: bytes) -> bool:
            # Trigger on the second audio chunk (after pause)
            self._seen += 1
            if self._seen >= 2:
                self._last_trigger_at = time.time()
                return True
            self._last_trigger_at = time.time()
            return False

    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Unified as unified

    monkeypatch.setattr(unified, "UnifiedStreamingTranscriber", _StubTranscriber)
    monkeypatch.setattr(unified, "SileroTurnDetector", _StubTurnDetector)

    cfg = json.dumps({"type": "config", "model": "parakeet", "sample_rate": 16000})
    audio_frame = json.dumps({"type": "audio", "data": base64.b64encode(b'1234').decode("ascii")})
    stop = json.dumps({"type": "stop"})
    # Insert a pause between two audio frames to mimic silence
    ws = _DummyWebSocket([cfg, audio_frame, audio_frame, stop], delays=[0.0, 0.3, 0.0, 0.0])

    start = time.time()
    await unified.handle_unified_websocket(ws, unified.UnifiedStreamingConfig())
    elapsed = time.time() - start

    finals = [m for m in ws.sent if m.get("type") == "full_transcript"]
    assert len(finals) == 1, f"Expected single final, saw {ws.sent}"
    assert finals[0].get("text") == "pause-final"
    assert elapsed < 3.0, f"Streaming with pause should complete quickly, took {elapsed}s"
