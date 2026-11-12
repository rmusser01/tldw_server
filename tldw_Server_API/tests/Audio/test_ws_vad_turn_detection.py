import asyncio
import base64
import json
import shutil
import time
import wave
from pathlib import Path
from types import SimpleNamespace

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
    assert detector.observe(b"\x00" * (512 * 4)) is False
    # Wait beyond turn_stop_secs to simulate silence
    time.sleep(0.06)
    assert detector.observe(b"\x00" * (512 * 4)) is True


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

    assert detector.observe(b"\x00" * (512 * 4)) is False  # speech observed
    time.sleep(0.06)  # silence shorter than min_utterance guard
    assert detector.observe(b"\x00" * (512 * 4)) is False
    time.sleep(0.5)  # now above min_utterance
    assert detector.observe(b"\x00" * (512 * 4)) is True


def test_silero_turn_detector_fails_open_on_torchscript_error(monkeypatch):
    """A TorchScript error (torch.jit.Error is not a RuntimeError) must fail open, not abort the stream."""

    class _FakeJitError(Exception):
        pass

    class _RaisingVADIterator:
        def __init__(self, *args, **kwargs):
            pass

        def reset_states(self):
            return None

        def __call__(self, _audio_in, return_seconds=False, **_kwargs):
            raise _FakeJitError("Input audio chunk is too short")

    import types

    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.VAD_Lib as vlib
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Unified as unified

    monkeypatch.setattr(vlib, "_lazy_import_silero_vad", lambda: ("model", [None, None, None, _RaisingVADIterator, None]))
    fake_torch = types.SimpleNamespace(jit=types.SimpleNamespace(Error=_FakeJitError), from_numpy=lambda arr: arr)
    monkeypatch.setattr(unified, "torch", fake_torch)

    detector = unified.SileroTurnDetector(
        sample_rate=16000,
        enabled=True,
        vad_threshold=0.5,
        min_silence_ms=200,
        turn_stop_secs=0.05,
    )
    assert detector.available

    assert detector.observe(np.zeros(512, dtype=np.float32).tobytes()) is False
    assert detector.available is False
    assert detector.unavailable_reason == "vad_runtime_error"


def test_silero_turn_detector_real_vad_end_to_end(monkeypatch, tmp_path):


    """
    Exercise real Silero VAD on tracked speech plus trailing silence.

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
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib import convert_to_wav

    # Convert the tracked speech fixture in pytest's private directory.
    video_path = tmp_path / "speech.mp4"
    shutil.copyfile(
        Path("tldw_Server_API/tests/Media_Ingestion_Modification/test_media/sample.mp4"),
        video_path,
    )
    wav_path = convert_to_wav(str(video_path), offset=1, end_time=7, base_dir=tmp_path)
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

    now = [1000.0]
    monkeypatch.setattr(time, "time", lambda: now[0])
    frame_size = int(0.1 * sr)
    triggered = False
    for i in range(0, len(audio), frame_size):
        now[0] = 1000.0 + (i // frame_size) * 0.1
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


class _RecordingVADIterator:
    def __init__(self, **kwargs):
        if kwargs["sampling_rate"] not in (8000, 16000):
            raise ValueError("unsupported Silero sample rate")
        self.windows = []
        self.reset_calls = 0

    def __call__(self, audio_in, **_kwargs):
        self.windows.append(np.asarray(audio_in).copy())
        return {"start": 0} if len(self.windows) == 1 else {}

    def reset_states(self):
        self.reset_calls += 1


@pytest.fixture
def silero_detector_factory(monkeypatch):
    """Exercise the real detector with a deterministic, optional-Torch-safe provider."""
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Unified as unified
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.VAD_Lib as vlib

    monkeypatch.setattr(unified, "load_comprehensive_config", lambda: None)
    monkeypatch.setattr(unified, "_get_torch_module", lambda: None)

    def make(
        iterator_cls=_RecordingVADIterator, *, sample_rate=16000,
        turn_stop_secs=0.1, min_utterance_secs=0.0,
    ):
        monkeypatch.setattr(
            vlib,
            "_lazy_import_silero_vad",
            lambda: (object(), [None, None, None, iterator_cls, None]),
        )
        return unified.SileroTurnDetector(
            sample_rate=sample_rate,
            enabled=True,
            vad_threshold=0.5,
            min_silence_ms=200,
            turn_stop_secs=turn_stop_secs,
            min_utterance_secs=min_utterance_secs,
        )

    return make


@pytest.mark.parametrize("sample_rate,window", [(8000, 256), (16000, 512)])
def test_silero_buffers_short_frames_and_preserves_ordered_tail(
    silero_detector_factory, sample_rate, window
):
    """Packet boundaries must not change the model windows, even without Torch."""
    detector = silero_detector_factory(sample_rate=sample_rate)
    audio = np.arange(window * 3, dtype=np.float32)
    assert detector.observe(audio[: window // 2].tobytes()) is False
    assert detector._iterator.windows == []
    detector.observe(audio[window // 2 : window * 2 + 17].tobytes())
    assert [len(chunk) for chunk in detector._iterator.windows] == [window, window]
    detector.observe(audio[window * 2 + 17 :].tobytes())
    np.testing.assert_array_equal(np.concatenate(detector._iterator.windows), audio)
    assert detector.available


def test_silero_consumes_all_windows_after_early_speech(silero_detector_factory):
    """A speech event must not short-circuit later stateful model calls."""
    detector = silero_detector_factory()
    audio = np.arange(512 * 3, dtype=np.float32)
    assert detector.observe(audio.tobytes()) is False
    assert [len(chunk) for chunk in detector._iterator.windows] == [512, 512, 512]
    np.testing.assert_array_equal(np.concatenate(detector._iterator.windows), audio)


def test_silero_turn_reset_discards_only_vad_tail(silero_detector_factory, monkeypatch):
    """The next turn must not inherit the preceding detector's incomplete window."""
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Unified as unified

    now = [1000.0]
    monkeypatch.setattr(unified.time, "time", lambda: now[0])
    detector = silero_detector_factory()
    assert detector.observe(np.zeros(512, dtype=np.float32).tobytes()) is False
    now[0] = 1000.2
    assert detector.observe(np.zeros(512 + 256, dtype=np.float32).tobytes()) is True
    assert detector._iterator.reset_calls == 1
    assert detector.observe(np.zeros(256, dtype=np.float32).tobytes()) is False
    assert len(detector._iterator.windows) == 2


@pytest.mark.parametrize("stage", ["initialization", "observe", "reset"])
def test_silero_typed_script_error_recovers_without_private_diagnostics(
    silero_detector_factory, monkeypatch, stage
):
    """The lazy Torch exception class belongs only to the detector's recovery boundary."""
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Unified as unified

    class _ScriptError(Exception):
        pass

    class _FailingIterator(_RecordingVADIterator):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            if stage == "initialization":
                raise _ScriptError("private-vad-sentinel /private/model/traceback")

        def __call__(self, audio_in, **kwargs):
            if stage == "observe":
                raise _ScriptError("private-vad-sentinel /private/model/traceback")
            return super().__call__(audio_in, **kwargs)

        def reset_states(self):
            super().reset_states()
            if stage == "reset":
                raise _ScriptError("private-vad-sentinel /private/model/traceback")

    fake_torch = SimpleNamespace(jit=SimpleNamespace(Error=_ScriptError), from_numpy=lambda x: x)
    monkeypatch.setattr(unified, "_get_torch_module", lambda: fake_torch)
    warnings = []
    monkeypatch.setattr(unified.logger, "warning", lambda message, *args: warnings.append(message.format(*args)))
    now = [1000.0]
    monkeypatch.setattr(unified.time, "time", lambda: now[0])
    detector = silero_detector_factory(_FailingIterator)
    if stage != "initialization":
        assert detector.available
        if stage == "reset":
            assert detector.observe(np.zeros(512, dtype=np.float32).tobytes()) is False
            now[0] = 1000.2
        assert detector.observe(np.zeros(513, dtype=np.float32).tobytes()) is False
    assert detector.available is False
    assert detector._vad_audio_remainder.size == 0
    assert detector.unavailable_reason == (
        "vad_initialization_error" if stage == "initialization" else "vad_runtime_error"
    )
    assert warnings
    assert "private-vad-sentinel" not in str(warnings + [detector.unavailable_reason])
    assert "/private/model/traceback" not in str(warnings + [detector.unavailable_reason])
    assert _ScriptError not in unified._AUDIO_UNIFIED_NONCRITICAL_EXCEPTIONS


@pytest.mark.parametrize("error_class", [None, "not an exception class", KeyboardInterrupt, RuntimeError])
def test_silero_rejects_invalid_torch_error_class(
    silero_detector_factory, monkeypatch, error_class
):
    """Optional/malformed Torch APIs must not expand recovery to unrelated exceptions."""
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Unified as unified

    class _UnrelatedError(Exception):
        pass

    class _UnexpectedIterator(_RecordingVADIterator):
        def __call__(self, *_args, **_kwargs):
            raise _UnrelatedError("unrelated failure")

    fake_torch = SimpleNamespace(jit=SimpleNamespace(Error=error_class), from_numpy=lambda x: x)
    monkeypatch.setattr(unified, "_get_torch_module", lambda: fake_torch)
    detector = silero_detector_factory(_UnexpectedIterator)
    with pytest.raises(_UnrelatedError, match="unrelated failure"):
        detector.observe(np.zeros(512, dtype=np.float32).tobytes())
    assert detector.available


def test_silero_preserves_unsupported_rate_rejection(silero_detector_factory):
    detector = silero_detector_factory(sample_rate=44100)
    assert detector.available is False
    assert detector.observe(np.zeros(4096, dtype=np.float32).tobytes()) is False


@pytest.mark.asyncio
async def test_typed_vad_failure_preserves_original_asr_frames_and_manual_commit(
    silero_detector_factory, monkeypatch
):
    """Only VAD's state is retired; all original PCM reaches ASR and stays manually committable."""
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Unified as unified

    class _ScriptError(Exception):
        pass

    class _FailingIterator(_RecordingVADIterator):
        def __call__(self, *_args, **_kwargs):
            raise _ScriptError("private-vad-sentinel /private/model/traceback")

    fake_torch = SimpleNamespace(jit=SimpleNamespace(Error=_ScriptError), from_numpy=lambda x: x)
    monkeypatch.setattr(unified, "_get_torch_module", lambda: fake_torch)
    detector = silero_detector_factory(_FailingIterator)
    monkeypatch.setattr(unified, "SileroTurnDetector", lambda **_kwargs: detector)
    received = []

    class _StubTranscriber:
        def __init__(self, _config):
            pass

        def initialize(self):
            pass

        async def process_audio_chunk(self, audio_bytes):
            received.append(audio_bytes)
            return {"type": "partial", "text": "asr_after_vad_error", "is_final": False}

        def get_full_transcript(self):
            return "asr_after_vad_error"

        def reset(self):
            pass

        def cleanup(self):
            pass

    monkeypatch.setattr(unified, "UnifiedStreamingTranscriber", _StubTranscriber)
    pcm_frames = [np.arange(513, dtype="<i2"), np.arange(7, dtype="<i2")]
    ws = _DummyWebSocket([
        json.dumps({"type": "config", "model": "parakeet", "sample_rate": 16000, "enable_vad": True}),
        *[json.dumps({"type": "audio", "data": base64.b64encode(pcm.tobytes()).decode("ascii")}) for pcm in pcm_frames],
        json.dumps({"type": "commit"}),
        json.dumps({"type": "stop"}),
    ])
    await unified.handle_unified_websocket(ws, unified.UnifiedStreamingConfig())
    assert received == [(pcm.astype(np.float32) / 32768.0).tobytes() for pcm in pcm_frames]
    full_transcripts = [frame for frame in ws.sent if frame.get("type") == "full_transcript"]
    assert full_transcripts[0]["text"] == "asr_after_vad_error"
    assert full_transcripts[0]["auto_commit"] is False
    assert full_transcripts[0]["vad_status"] == "fail_open"
    assert "private-vad-sentinel" not in str(ws.sent)
    assert "/private/model/traceback" not in str(ws.sent)


@pytest.mark.parametrize(
    "native_state,result,expected",
    [
        (True, None, True),
        (False, {"end": 512}, False),
        (False, {"start": 0}, False),
        (False, {"speech_probs": [0.9]}, False),
    ],
    ids=["ongoing-none", "native-end", "inactive-start", "inactive-probability"],
)
def test_silero_boolean_iterator_state_controls_speech(
    silero_detector_factory, native_state, result, expected
):
    """Native state is authoritative even when event-only output is contradictory."""
    detector = silero_detector_factory()
    detector._iterator.triggered = native_state
    assert detector._saw_speech(result) is expected


@pytest.mark.parametrize(
    "result,expected",
    [
        ({"start": 0}, True),
        ({"speech_probs": [0.9]}, True),
        ({"speech_timestamps": [{"start": 0, "end": 512}]}, True),
        ({"end": 512}, False),
    ],
    ids=["start-zero", "probability", "timestamps", "end-only"],
)
def test_silero_legacy_event_fallback_preserves_supported_shapes(
    silero_detector_factory, result, expected
):
    """Adapters without a boolean state retain the supported legacy output contract."""
    detector = silero_detector_factory()
    detector._iterator.triggered = "legacy-state"
    assert detector._saw_speech(result) is expected


def test_silero_active_speech_state_consumes_windows_then_commits_once(
    silero_detector_factory, monkeypatch
):
    """Ongoing event-free speech stays open until the native end and silence guards."""
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Streaming_Unified as unified

    class _StatefulIterator(_RecordingVADIterator):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.triggered = False
            self.speaking = True

        def __call__(self, audio_in, **kwargs):
            super().__call__(audio_in, **kwargs)
            was_active = self.triggered
            self.triggered = self.speaking
            if self.triggered and not was_active:
                return {"start": 0}
            if was_active and not self.triggered:
                return {"end": len(self.windows) * 512}
            return None

        def reset_states(self):
            super().reset_states()
            self.triggered = False

    now = [1000.0]
    monkeypatch.setattr(unified.time, "time", lambda: now[0])
    detector = silero_detector_factory(
        _StatefulIterator, turn_stop_secs=0.2, min_utterance_secs=0.2
    )
    audio = np.arange(512 * 9, dtype=np.float32)
    for packet, elapsed in enumerate((0.0, 0.3, 0.6)):
        now[0] = 1000.0 + elapsed
        assert detector.observe(audio[packet * 1536 : (packet + 1) * 1536].tobytes()) is False
    assert len(detector._iterator.windows) == 9
    np.testing.assert_array_equal(np.concatenate(detector._iterator.windows), audio)
    assert detector._iterator.reset_calls == 0

    detector._iterator.speaking = False
    now[0] = 1000.7
    silence = np.zeros(512 * 3, dtype=np.float32).tobytes()
    assert detector.observe(silence) is False
    now[0] = 1000.85
    assert detector.observe(silence) is True
    assert detector.last_trigger_at == 1000.85
    assert detector._iterator.reset_calls == 1
    now[0] = 1001.1
    assert detector.observe(silence) is False
    assert detector._iterator.reset_calls == 1
