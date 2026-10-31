"""Integration tests for the native STT benchmark CLI."""

from __future__ import annotations

import shutil
import wave
from pathlib import Path

import Helper_Scripts.benchmarks.stt_bench as stt_bench
import pytest

pytestmark = pytest.mark.integration


def test_ffprobe_generated_wav_integration(tmp_path: Path) -> None:
    """Measure a generated WAV through the installed ffprobe executable."""
    if shutil.which("ffprobe") is None:
        pytest.skip("ffprobe is not installed")
    audio_path = tmp_path / "tone.wav"
    with wave.open(str(audio_path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(8_000)
        output.writeframes(b"\0\0" * 8_000)

    assert stt_bench.probe_audio_duration_ffprobe(audio_path) == pytest.approx(1.0)
