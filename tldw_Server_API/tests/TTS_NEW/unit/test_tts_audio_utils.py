import asyncio
import subprocess
import sys
import threading
import types
from pathlib import Path

import numpy as np
import pytest

from tldw_Server_API.app.core.TTS import audio_utils
from tldw_Server_API.app.core.TTS.audio_utils import (
    analyze_audio_signal,
    compute_audio_peak,
    compute_audio_rms,
    crossfade_audio,
    evaluate_audio_quality,
    split_text_into_chunks,
    trailing_silence_duration_ms,
    trim_trailing_silence,
)


def _make_executable(path: Path) -> str:
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(0o755)
    return str(path.resolve())


def test_split_text_into_chunks_basic():
    text = "Hello world. This is a second sentence! And a third?"
    chunks = split_text_into_chunks(text, target_chars=20, max_chars=30, min_chars=5)
    assert chunks
    assert all(isinstance(c, str) for c in chunks)
    assert "Hello world." in chunks[0]


def test_crossfade_audio_length():
    left = np.ones(1000, dtype=np.int16) * 1000
    right = np.ones(1000, dtype=np.int16) * 2000
    merged = crossfade_audio(left, right, sample_rate=1000, crossfade_ms=100)
    # 100 ms at 1kHz = 100 samples overlap
    assert merged.shape[0] == (left.shape[0] + right.shape[0] - 100)
    assert merged.dtype == np.int16


def test_audio_metrics_and_trailing_silence():
    audio = np.concatenate(
        [np.ones(100, dtype=np.int16) * 1000, np.zeros(100, dtype=np.int16)]
    )
    rms = compute_audio_rms(audio)
    peak = compute_audio_peak(audio)
    trailing_ms = trailing_silence_duration_ms(audio, sample_rate=1000, threshold=0.01)
    trimmed = trim_trailing_silence(audio, sample_rate=1000, threshold=0.01)
    metrics = analyze_audio_signal(audio, sample_rate=1000, silence_threshold=0.01)

    assert rms > 0
    assert peak > 0
    assert trailing_ms == 100
    assert trimmed.shape[0] == 100
    assert metrics["trailing_silence_ms"] == 100.0


def test_evaluate_audio_quality_flags_silence_and_short_duration():
    audio = np.zeros(200, dtype=np.int16)
    metrics, warnings = evaluate_audio_quality(
        audio,
        sample_rate=1000,
        text_length=120,
        min_text_length=10,
        min_rms=0.001,
        min_peak=0.02,
        trailing_silence_ms=50,
        expected_chars_per_sec=10.0,
        min_duration_ratio=0.5,
        min_duration_seconds=0.4,
    )
    assert metrics["rms"] == 0.0
    assert any("low_levels" in w for w in warnings)
    assert any("trailing_silence_ms" in w for w in warnings)


def test_convert_audio_fallback_log_sanitizes_exception_text(monkeypatch):
    processor = audio_utils.AudioProcessor.__new__(audio_utils.AudioProcessor)
    processor.ffmpeg_available = False
    processor.librosa_available = True
    audio_bytes = b"original audio"
    secret_detail = "/Users/example/private/voice-clone-token-sk-test.wav"
    logged_messages: list[str] = []

    def fail_load(*args, **kwargs):
        raise RuntimeError(f"failed to decode {secret_detail}")

    fake_librosa = types.SimpleNamespace(load=fail_load)
    fake_soundfile = types.SimpleNamespace(write=lambda *args, **kwargs: None)
    monkeypatch.setitem(sys.modules, "librosa", fake_librosa)
    monkeypatch.setitem(sys.modules, "soundfile", fake_soundfile)

    sink_id = audio_utils.logger.add(
        lambda message: logged_messages.append(message.record["message"]),
        level="ERROR",
    )
    try:
        result = processor.convert_audio(audio_bytes)
    finally:
        audio_utils.logger.remove(sink_id)

    assert result == audio_bytes
    assert any("Audio conversion failed" in message for message in logged_messages)
    assert all(secret_detail not in message for message in logged_messages)


def test_convert_audio_strict_failure_raises_without_returning_original(monkeypatch):
    """Verify strict conversion failure raises instead of returning source bytes."""
    processor = audio_utils.AudioProcessor.__new__(audio_utils.AudioProcessor)
    processor.ffmpeg_available = False
    processor.librosa_available = True
    audio_bytes = b"original audio"

    def fail_load(*args, **kwargs):
        """Raise from the fallback decoder to simulate conversion failure."""
        raise RuntimeError("decode failed")

    fake_librosa = types.SimpleNamespace(load=fail_load)
    fake_soundfile = types.SimpleNamespace(write=lambda *args, **kwargs: None)
    monkeypatch.setitem(sys.modules, "librosa", fake_librosa)
    monkeypatch.setitem(sys.modules, "soundfile", fake_soundfile)

    with pytest.raises(RuntimeError, match="Audio conversion failed"):
        processor.convert_audio(audio_bytes, strict=True)


@pytest.mark.parametrize(
    "timeout_seconds",
    [0, -1, float("nan"), float("inf"), float("-inf"), "1", True],
)
def test_convert_audio_rejects_invalid_timeout_before_work(monkeypatch, timeout_seconds):
    """Invalid conversion timeouts fail before selecting a conversion backend."""
    processor = audio_utils.AudioProcessor.__new__(audio_utils.AudioProcessor)
    processor.ffmpeg_available = False
    processor.librosa_available = True
    monkeypatch.setattr(
        audio_utils.tempfile,
        "NamedTemporaryFile",
        lambda *args, **kwargs: pytest.fail("conversion work must not start"),
    )

    with pytest.raises(ValueError, match="timeout_seconds must be a finite positive number"):
        processor.convert_audio(b"audio", timeout_seconds=timeout_seconds)


def test_convert_audio_timeout_requires_ffmpeg_before_temp_files(monkeypatch):
    """Timed conversion never falls back to an uninterruptible local decoder."""
    processor = audio_utils.AudioProcessor.__new__(audio_utils.AudioProcessor)
    processor.ffmpeg_available = False
    processor.librosa_available = True
    monkeypatch.setattr(
        audio_utils.tempfile,
        "NamedTemporaryFile",
        lambda *args, **kwargs: pytest.fail("temp files must not be created"),
    )
    monkeypatch.setitem(
        sys.modules,
        "librosa",
        types.SimpleNamespace(load=lambda *args, **kwargs: pytest.fail("librosa must not run")),
    )

    with pytest.raises(RuntimeError, match="^Audio conversion failed$"):
        processor.convert_audio(b"audio", strict=True, timeout_seconds=1.0)


def test_convert_audio_timeout_is_forwarded_to_ffmpeg(monkeypatch, tmp_path):
    """Timed ffmpeg conversion forwards the exact timeout to subprocess.run."""
    processor = audio_utils.AudioProcessor.__new__(audio_utils.AudioProcessor)
    processor.ffmpeg_available = True
    processor.librosa_available = True
    processor.ffmpeg_path = _make_executable(tmp_path / "other-ffmpeg")
    pinned_path = _make_executable(tmp_path / "pinned-ffmpeg")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    seen: dict[str, object] = {}
    monkeypatch.setattr(audio_utils.tempfile, "tempdir", str(scratch))
    monkeypatch.setattr(
        audio_utils.shutil,
        "which",
        lambda _name: pytest.fail("timed conversion must not re-resolve PATH"),
    )

    def fake_run(command, **kwargs):
        seen["command"] = command
        seen["kwargs"] = kwargs
        Path(command[-1]).write_bytes(b"converted")
        return types.SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(audio_utils.subprocess, "run", fake_run)

    result = processor.convert_audio(
        b"audio",
        target_format="mp3",
        timeout_seconds=1.25,
        ffmpeg_path=pinned_path,
    )

    assert result == b"converted"
    assert seen["command"][0] == pinned_path
    assert seen["kwargs"]["timeout"] == 1.25
    assert not list(scratch.iterdir())


def test_convert_audio_timeout_maps_and_sanitizes_timeout_expired(monkeypatch, tmp_path):
    """Subprocess timeouts become sanitized, chained conversion failures."""
    processor = audio_utils.AudioProcessor.__new__(audio_utils.AudioProcessor)
    processor.ffmpeg_available = True
    processor.librosa_available = False
    processor.ffmpeg_path = _make_executable(tmp_path / "secret-ffmpeg-token-sk-test")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    logged_messages: list[str] = []
    monkeypatch.setattr(audio_utils.tempfile, "tempdir", str(scratch))

    def fail_run(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(audio_utils.subprocess, "run", fail_run)
    sink_id = audio_utils.logger.add(
        lambda message: logged_messages.append(message.record["message"]),
        level="ERROR",
    )
    try:
        with pytest.raises(RuntimeError, match="^Audio conversion failed$") as exc_info:
            processor.convert_audio(b"raw-input-token", strict=True, timeout_seconds=0.5)
    finally:
        audio_utils.logger.remove(sink_id)

    assert isinstance(exc_info.value.__cause__, subprocess.TimeoutExpired)
    assert "secret" not in str(exc_info.value)
    assert "raw-input-token" not in str(exc_info.value)
    assert all("secret" not in message for message in logged_messages)
    assert all("raw-input-token" not in message for message in logged_messages)
    assert not list(scratch.iterdir())


def test_convert_audio_timeout_maps_ffmpeg_launch_failure(monkeypatch, tmp_path):
    """Timed conversion fails if an advertised ffmpeg executable disappears."""
    processor = audio_utils.AudioProcessor.__new__(audio_utils.AudioProcessor)
    processor.ffmpeg_available = True
    processor.librosa_available = False
    processor.ffmpeg_path = _make_executable(tmp_path / "private-secret-ffmpeg")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(audio_utils.tempfile, "tempdir", str(scratch))

    def fail_run(*args, **kwargs):
        raise FileNotFoundError("private command")

    monkeypatch.setattr(audio_utils.subprocess, "run", fail_run)

    with pytest.raises(RuntimeError, match="^Audio conversion failed$") as exc_info:
        processor.convert_audio(b"audio", strict=True, timeout_seconds=0.5)

    assert isinstance(exc_info.value.__cause__, FileNotFoundError)
    assert "private" not in str(exc_info.value)
    assert not list(scratch.iterdir())


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize(
    "failure_kind",
    ["missing", "unlaunchable", "timeout", "nonzero"],
)
def test_timed_conversion_failures_follow_strict_contract(
    monkeypatch,
    tmp_path,
    strict,
    failure_kind,
):
    """Every timed ffmpeg runtime failure follows the existing strict contract."""
    processor = audio_utils.AudioProcessor.__new__(audio_utils.AudioProcessor)
    processor.ffmpeg_available = True
    processor.librosa_available = True
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(audio_utils.tempfile, "tempdir", str(scratch))

    if failure_kind == "missing":
        pinned_path = str((tmp_path / "missing-ffmpeg").resolve())
    else:
        pinned_path = _make_executable(tmp_path / "ffmpeg")

    def fail_run(command, **kwargs):
        if failure_kind == "unlaunchable":
            raise OSError("private command")
        if failure_kind == "timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        if failure_kind == "nonzero":
            return types.SimpleNamespace(returncode=1, stderr="private path")
        pytest.fail("missing executable must fail before subprocess")

    monkeypatch.setattr(audio_utils.subprocess, "run", fail_run)
    kwargs = {
        "strict": strict,
        "timeout_seconds": 0.5,
        "ffmpeg_path": pinned_path,
    }

    if strict:
        with pytest.raises(RuntimeError, match="^Audio conversion failed$") as exc_info:
            processor.convert_audio(b"original", **kwargs)
        assert exc_info.value.__cause__ is not None
        assert "private" not in str(exc_info.value)
    else:
        assert processor.convert_audio(b"original", **kwargs) == b"original"

    assert not list(scratch.iterdir())


@pytest.mark.parametrize("returncode", [0, 1])
def test_convert_audio_ffmpeg_cleans_temp_files_on_success_and_error(
    monkeypatch,
    tmp_path,
    returncode,
):
    """Converter-owned ffmpeg files are removed on both result paths."""
    processor = audio_utils.AudioProcessor.__new__(audio_utils.AudioProcessor)
    processor.ffmpeg_available = True
    processor.librosa_available = False
    processor.ffmpeg_path = _make_executable(tmp_path / "ffmpeg")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(audio_utils.tempfile, "tempdir", str(scratch))

    def fake_run(command, **kwargs):
        if returncode == 0:
            Path(command[-1]).write_bytes(b"converted")
        return types.SimpleNamespace(returncode=returncode, stderr="private path")

    monkeypatch.setattr(audio_utils.subprocess, "run", fake_run)

    if returncode == 0:
        assert processor.convert_audio(b"audio", strict=True) == b"converted"
    else:
        with pytest.raises(RuntimeError, match="^Audio conversion failed$"):
            processor.convert_audio(b"audio", strict=True)

    assert not list(scratch.iterdir())


def test_convert_audio_cleans_input_when_output_temp_creation_fails(monkeypatch, tmp_path):
    """A failed output allocation does not leak the already-written input file."""
    processor = audio_utils.AudioProcessor.__new__(audio_utils.AudioProcessor)
    processor.ffmpeg_available = True
    processor.librosa_available = False
    processor.ffmpeg_path = _make_executable(tmp_path / "ffmpeg")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    real_named_tempfile = audio_utils.tempfile.NamedTemporaryFile
    calls = 0

    def create_temp(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("output unavailable")
        return real_named_tempfile(*args, dir=scratch, **kwargs)

    monkeypatch.setattr(audio_utils.tempfile, "NamedTemporaryFile", create_temp)

    with pytest.raises(RuntimeError, match="^Audio conversion failed$"):
        processor.convert_audio(b"audio", strict=True)

    assert not list(scratch.iterdir())


def test_convert_audio_without_timeout_preserves_librosa_first_behavior(monkeypatch, tmp_path):
    """The legacy no-timeout call still prefers librosa when it is available."""
    processor = audio_utils.AudioProcessor.__new__(audio_utils.AudioProcessor)
    processor.ffmpeg_available = True
    processor.librosa_available = True
    processor.ffmpeg_path = "/usr/bin/ffmpeg"
    monkeypatch.setattr(audio_utils.tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(
        audio_utils.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("legacy path must keep preferring librosa"),
    )
    monkeypatch.setitem(
        sys.modules,
        "librosa",
        types.SimpleNamespace(load=lambda *args, **kwargs: (np.zeros(8), 24000)),
    )
    monkeypatch.setitem(
        sys.modules,
        "soundfile",
        types.SimpleNamespace(
            write=lambda path, *args, **kwargs: Path(path).write_bytes(b"legacy-converted")
        ),
    )

    assert processor.convert_audio(b"audio") == b"legacy-converted"
    assert not list(tmp_path.iterdir())


@pytest.mark.asyncio
async def test_convert_audio_async_forwards_timeout_and_strict(monkeypatch):
    """The async wrapper preserves timeout and strict conversion options."""
    processor = audio_utils.AudioProcessor.__new__(audio_utils.AudioProcessor)
    seen: dict[str, object] = {}

    def fake_convert(audio_bytes, **kwargs):
        seen["audio_bytes"] = audio_bytes
        seen.update(kwargs)
        return b"converted"

    monkeypatch.setattr(processor, "convert_audio", fake_convert)

    result = await processor.convert_audio_async(
        b"audio",
        target_format="flac",
        target_sample_rate=16000,
        provider="higgs",
        strict=True,
        timeout_seconds=2.5,
        ffmpeg_path="/opt/pinned/ffmpeg",
    )

    assert result == b"converted"
    assert seen == {
        "audio_bytes": b"audio",
        "target_format": "flac",
        "target_sample_rate": 16000,
        "provider": "higgs",
        "strict": True,
        "timeout_seconds": 2.5,
        "ffmpeg_path": "/opt/pinned/ffmpeg",
    }


@pytest.mark.asyncio
async def test_convert_audio_async_cancellation_allows_late_cleanup(monkeypatch, tmp_path):
    """Cancelling the awaiter does not prevent the worker's final cleanup."""
    processor = audio_utils.AudioProcessor.__new__(audio_utils.AudioProcessor)
    processor.ffmpeg_available = True
    processor.librosa_available = False
    processor.ffmpeg_path = _make_executable(tmp_path / "ffmpeg")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    started = threading.Event()
    release = threading.Event()
    monkeypatch.setattr(audio_utils.tempfile, "tempdir", str(scratch))

    def delayed_run(command, **kwargs):
        started.set()
        assert release.wait(2)
        Path(command[-1]).write_bytes(b"converted")
        return types.SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(audio_utils.subprocess, "run", delayed_run)
    task = asyncio.create_task(
        processor.convert_audio_async(b"audio", strict=True, timeout_seconds=1.0)
    )
    assert await asyncio.to_thread(started.wait, 1)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    release.set()
    for _ in range(100):
        if not list(scratch.iterdir()):
            break
        await asyncio.sleep(0.01)

    assert not list(scratch.iterdir())


def test_process_voice_reference_uses_strict_conversion(monkeypatch):
    """Verify voice-reference conversion requests strict failure behavior."""
    seen: dict[str, object] = {}

    class FailingConversionProcessor:
        """AudioProcessor stub that records strict conversion options."""

        PROVIDER_REQUIREMENTS = {"higgs": {"sample_rate": 24000}}

        def decode_base64_audio(self, _base64_audio):
            """Return decoded reference bytes for the conversion path."""
            return b"raw-reference"

        def convert_audio(self, audio_bytes, **kwargs):
            """Record conversion arguments and simulate a conversion failure."""
            seen["audio_bytes"] = audio_bytes
            seen["strict"] = kwargs.get("strict")
            raise RuntimeError("conversion failed")

    monkeypatch.setattr(audio_utils, "AudioProcessor", FailingConversionProcessor)

    processed, error = audio_utils.process_voice_reference(
        "cmF3LXJlZmVyZW5jZQ==",
        "higgs",
        validate=False,
        convert=True,
    )

    assert processed is None
    assert "conversion failed" in error
    assert seen == {"audio_bytes": b"raw-reference", "strict": True}


def test_audio_processor_constructor_pins_which_without_subprocess(monkeypatch, tmp_path):
    executable = _make_executable(tmp_path / "ffmpeg")
    monkeypatch.setattr(audio_utils.shutil, "which", lambda _name: executable)
    monkeypatch.setattr(
        audio_utils.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("constructor must not execute ffmpeg"),
    )
    monkeypatch.setattr(audio_utils.AudioProcessor, "_check_librosa", lambda self: False)

    processor = audio_utils.AudioProcessor()

    assert processor.ffmpeg_available is True
    assert processor.ffmpeg_path == executable


def test_audio_processor_constructor_accepts_pinned_ffmpeg_path(monkeypatch, tmp_path):
    executable = _make_executable(tmp_path / "ffmpeg")
    monkeypatch.setattr(
        audio_utils.shutil,
        "which",
        lambda _name: pytest.fail("injected identity must not search PATH"),
    )
    monkeypatch.setattr(
        audio_utils.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("constructor must not execute ffmpeg"),
    )
    monkeypatch.setattr(audio_utils.AudioProcessor, "_check_librosa", lambda self: False)

    processor = audio_utils.AudioProcessor(ffmpeg_path=executable)

    assert processor.ffmpeg_available is True
    assert processor.ffmpeg_path == executable


def test_validate_audio_failure_log_sanitizes_exception_text(monkeypatch):
    processor = audio_utils.AudioProcessor.__new__(audio_utils.AudioProcessor)
    processor.librosa_available = True
    secret_detail = "/Users/example/private/validation-token-sk-test.wav"
    logged_messages: list[str] = []

    def fail_load(*args, **kwargs):
        raise RuntimeError(f"cannot inspect {secret_detail}")

    fake_librosa = types.SimpleNamespace(load=fail_load)
    monkeypatch.setitem(sys.modules, "librosa", fake_librosa)

    sink_id = audio_utils.logger.add(
        lambda message: logged_messages.append(message.record["message"]),
        level="ERROR",
    )
    try:
        is_valid, error_msg, info = processor.validate_audio(b"audio", "higgs")
    finally:
        audio_utils.logger.remove(sink_id)

    assert is_valid is False
    assert secret_detail in error_msg
    assert info == {}
    assert any("Audio validation error" in message for message in logged_messages)
    assert all(secret_detail not in message for message in logged_messages)


def test_extract_clean_segment_fallback_log_sanitizes_exception_text(monkeypatch):
    processor = audio_utils.AudioProcessor.__new__(audio_utils.AudioProcessor)
    processor.librosa_available = True
    audio_bytes = b"original audio"
    secret_detail = "/Users/example/private/segment-token-sk-test.wav"
    logged_messages: list[str] = []

    def fail_load(*args, **kwargs):
        raise RuntimeError(f"cannot extract from {secret_detail}")

    fake_librosa = types.SimpleNamespace(load=fail_load)
    fake_soundfile = types.SimpleNamespace(write=lambda *args, **kwargs: None)
    monkeypatch.setitem(sys.modules, "librosa", fake_librosa)
    monkeypatch.setitem(sys.modules, "soundfile", fake_soundfile)

    sink_id = audio_utils.logger.add(
        lambda message: logged_messages.append(message.record["message"]),
        level="ERROR",
    )
    try:
        result = processor.extract_clean_segment(audio_bytes)
    finally:
        audio_utils.logger.remove(sink_id)

    assert result == audio_bytes
    assert any("Failed to extract clean segment" in message for message in logged_messages)
    assert all(secret_detail not in message for message in logged_messages)


def test_process_voice_reference_failure_log_sanitizes_exception_text(monkeypatch):
    secret_detail = "/Users/example/private/reference-token-sk-test.wav"
    logged_messages: list[str] = []

    class FailingProcessor:
        def decode_base64_audio(self, _base64_audio):
            raise ValueError(f"bad reference at {secret_detail}")

    monkeypatch.setattr(audio_utils, "AudioProcessor", FailingProcessor)

    sink_id = audio_utils.logger.add(
        lambda message: logged_messages.append(message.record["message"]),
        level="ERROR",
    )
    try:
        processed, error = audio_utils.process_voice_reference(
            "not-base64",
            "higgs",
            validate=False,
            convert=False,
        )
    finally:
        audio_utils.logger.remove(sink_id)

    assert processed is None
    assert secret_detail in error
    assert any("Failed to process voice reference" in message for message in logged_messages)
    assert all(secret_detail not in message for message in logged_messages)


@pytest.mark.asyncio
async def test_process_voice_reference_async_failure_log_sanitizes_exception_text(monkeypatch):
    secret_detail = "/Users/example/private/async-reference-token-sk-test.wav"
    logged_messages: list[str] = []

    class FailingProcessor:
        def decode_base64_audio(self, _base64_audio):
            raise ValueError(f"bad async reference at {secret_detail}")

    monkeypatch.setattr(audio_utils, "AudioProcessor", FailingProcessor)

    sink_id = audio_utils.logger.add(
        lambda message: logged_messages.append(message.record["message"]),
        level="ERROR",
    )
    try:
        processed, error = await audio_utils.process_voice_reference_async(
            "not-base64",
            "higgs",
            validate=False,
            convert=False,
        )
    finally:
        audio_utils.logger.remove(sink_id)

    assert processed is None
    assert secret_detail in error
    assert any("Failed to process voice reference (async)" in message for message in logged_messages)
    assert all(secret_detail not in message for message in logged_messages)
