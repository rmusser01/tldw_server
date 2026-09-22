"""Regression: the buffered transcriber resampler must honour its post-condition.

_resample previously returned the INPUT unchanged when librosa was missing, while both
callers then set `sample_rate = 16000` unconditionally. A 48 kHz stream was therefore
relabelled as 16 kHz: 3x-speed audio fed to the model and every timestamp 3x short,
with an HTTP 200. The contract is: return audio AT target_sr, or raise.
"""

from __future__ import annotations

import builtins

import numpy as np
import pytest

from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Buffered_Transcription import (
    BufferedTranscriber,
)


@pytest.fixture
def no_librosa(monkeypatch: pytest.MonkeyPatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "librosa" or name.startswith("librosa."):
            raise ImportError("librosa disabled for this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    return None


def test_resample_returns_audio_at_target_rate_without_librosa(no_librosa) -> None:
    orig_sr, target_sr = 48000, 16000
    audio = np.sin(np.linspace(0, 20 * np.pi, orig_sr, dtype=np.float32))

    out = BufferedTranscriber._resample(None, audio, orig_sr, target_sr)

    expected_len = int(round(len(audio) * target_sr / orig_sr))
    assert len(out) == pytest.approx(expected_len, abs=1), (
        f"resampler returned {len(out)} samples; {expected_len} expected at {target_sr} Hz. "
        "Returning the input unchanged is what causes the 3x-speed corruption."
    )
    assert len(out) != len(audio), "output must not be the untouched input"


def test_resample_is_a_noop_when_rates_match(no_librosa) -> None:
    audio = np.zeros(1024, dtype=np.float32)
    out = BufferedTranscriber._resample(None, audio, 16000, 16000)
    assert len(out) == len(audio)
