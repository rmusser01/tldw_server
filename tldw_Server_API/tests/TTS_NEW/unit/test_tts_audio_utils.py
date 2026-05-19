import sys
import types

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


def test_check_ffmpeg_fallback_log_sanitizes_exception_text(monkeypatch):
    processor = audio_utils.AudioProcessor.__new__(audio_utils.AudioProcessor)
    processor.ffmpeg_path = None
    secret_detail = "/Users/example/private/ffmpeg-token-sk-test"
    logged_messages: list[str] = []

    def fail_run(*args, **kwargs):
        raise RuntimeError(f"ffmpeg probe failed at {secret_detail}")

    monkeypatch.setattr(audio_utils.shutil, "which", lambda _name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(audio_utils.subprocess, "run", fail_run)

    sink_id = audio_utils.logger.add(
        lambda message: logged_messages.append(message.record["message"]),
        level="WARNING",
    )
    try:
        result = processor._check_ffmpeg()
    finally:
        audio_utils.logger.remove(sink_id)

    assert result is False
    assert any("ffmpeg not found or not runnable" in message for message in logged_messages)
    assert all(secret_detail not in message for message in logged_messages)


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
