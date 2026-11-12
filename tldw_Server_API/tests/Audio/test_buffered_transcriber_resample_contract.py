"""Regression: the buffered transcriber resampler must honour its post-condition.

_resample previously returned the INPUT unchanged when librosa was missing, while both
callers then set `sample_rate = 16000` unconditionally. A 48 kHz stream was therefore
relabelled as 16 kHz: 3x-speed audio fed to the model and every timestamp 3x short,
with an HTTP 200. The contract is: return audio AT target_sr, or raise.
"""

from __future__ import annotations

import builtins
import sys
import types

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


@pytest.mark.parametrize("variant", ["onnx", "mlx"])
def test_transcribe_long_audio_48k_without_librosa_keeps_real_timing(
    variant: str, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End to end: a 48 kHz file through transcribe_long_audio with librosa absent.

    Covers both entry paths (process_audio for onnx, the structured mlx branch).
    The model must receive 16 kHz audio (1/3 the samples) and chunk timestamps
    must match the real 12 s duration, not 36 s.
    """
    import soundfile as sf

    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio import (
        Audio_Buffered_Transcription as abt,
    )

    monkeypatch.setitem(sys.modules, "librosa", None)
    orig_sr, duration_s = 48000, 12.0
    path = tmp_path / "tone_48k.wav"
    t = np.arange(int(orig_sr * duration_s)) / orig_sr
    sf.write(str(path), (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), orig_sr)

    seen = {}
    real_create_chunks = abt.BufferedTranscriber._create_chunks

    def spy_create_chunks(self, audio_data):
        seen["resampled_len"] = len(audio_data)
        seen["chunks"] = real_create_chunks(self, audio_data)
        return seen["chunks"]

    monkeypatch.setattr(abt.BufferedTranscriber, "_create_chunks", spy_create_chunks)

    model_calls = []

    def stub_model(chunk_audio, sample_rate=16000, **_kwargs):
        model_calls.append(len(chunk_audio))
        if variant == "mlx":
            return {"tokens": [{"text": "w", "start": 0.0, "end": 0.1}]}
        return "w"

    pkg = "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio"
    monkeypatch.setitem(
        sys.modules,
        f"{pkg}.Audio_Transcription_Parakeet_ONNX",
        types.SimpleNamespace(transcribe_with_parakeet_onnx=stub_model),
    )
    monkeypatch.setitem(
        sys.modules,
        f"{pkg}.Audio_Transcription_Parakeet_MLX",
        types.SimpleNamespace(transcribe_with_parakeet_mlx=stub_model),
    )

    result = abt.transcribe_long_audio(
        path,
        variant=variant,
        chunk_duration=4.0,
        total_buffer=6.0,
        merge_algo="simple",
        return_structured=(variant == "mlx"),
    )

    assert seen["resampled_len"] == pytest.approx(duration_s * 16000, abs=1)
    chunks = seen["chunks"]
    assert chunks[-1]["end"] == pytest.approx(duration_s, abs=1e-3)
    assert len(model_calls) == len(chunks) == 4  # stride 3 s over 12 s, not 12 chunks over 36 s
    if variant == "mlx":
        assert max(tok["start"] for tok in result["tokens"]) < duration_s


def test_streaming_resample_raises_instead_of_returning_wrong_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio import (
        Audio_Streaming_Unified as asu,
    )

    monkeypatch.delitem(
        sys.modules,
        "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib",
        raising=False,
    )

    def broken_interp(*_args, **_kwargs):
        raise ValueError("interp failed")

    monkeypatch.setattr(asu.np, "interp", broken_interp)

    with pytest.raises(ValueError, match="48000 Hz to 16000 Hz"):
        asu._resample_audio_if_needed(np.zeros(4800, dtype=np.float32), 48000, target_sr=16000)
