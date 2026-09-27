"""TASK-13304: the buffered resampler returned its input unchanged without librosa.

```python
def _resample(self, audio, orig_sr, target_sr):
    try:
        import librosa
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    except ImportError:
        logger.warning("librosa not available, returning original audio")
        return audio
```

Both callers then assert the new rate unconditionally:

```python
if sample_rate != 16000:
    audio_data = self._resample(audio_data, sample_rate, 16000)
    sample_rate = 16000        # regardless of whether resampling happened
```

So without librosa, 48 kHz audio is *declared* to be 16 kHz. The transcriber receives samples
3x too dense: the transcript is garbage and every timestamp is 3x short. The request returns
**HTTP 200** with a warning buried in the logs — silent corruption, not a failure.

`Audio_Transcription_Lib.py` already solves this correctly, with a linear-interpolation
fallback that actually resamples, so declaring the target rate afterwards is truthful. This
follows that, reusing its helper rather than adding a third copy of the arithmetic.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Buffered_Transcription import (
    BufferedTranscriber,
)

pytestmark = pytest.mark.unit


def _transcriber() -> BufferedTranscriber:
    """A bare instance: _resample touches no configured state."""
    return BufferedTranscriber.__new__(BufferedTranscriber)


@pytest.fixture()
def no_librosa(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make `import librosa` raise, the way a deployment without it behaves.

    A None entry in sys.modules makes the import statement raise ImportError, which is the
    branch under test.
    """
    monkeypatch.setitem(sys.modules, "librosa", None)


def test_downsampling_without_librosa_actually_changes_the_length(no_librosa: None) -> None:
    """The headline defect: the caller relabels the rate, so the data must really change.

    48 kHz -> 16 kHz is a factor of 3. Returning the input unchanged while the caller sets
    sample_rate = 16000 is what made the transcript garbage and every timestamp 3x short.
    """
    audio = np.sin(np.linspace(0, 40 * np.pi, 48_000)).astype(np.float32)

    resampled = _transcriber()._resample(audio, 48_000, 16_000)

    assert len(resampled) != len(audio), (
        "the resampler returned its input unchanged, so the caller's `sample_rate = 16000` "
        "is a lie and the transcriber reads samples 3x too dense"
    )
    assert len(resampled) == pytest.approx(16_000, rel=0.01), len(resampled)


def test_upsampling_without_librosa_actually_changes_the_length(no_librosa: None) -> None:
    """The same defect in the other direction."""
    audio = np.sin(np.linspace(0, 20 * np.pi, 8_000)).astype(np.float32)

    resampled = _transcriber()._resample(audio, 8_000, 16_000)

    assert len(resampled) == pytest.approx(16_000, rel=0.01), len(resampled)


def test_the_result_is_float32(no_librosa: None) -> None:
    """Downstream feature extraction assumes float32; the fallback must not widen it."""
    audio = np.sin(np.linspace(0, 10 * np.pi, 24_000)).astype(np.float32)

    resampled = _transcriber()._resample(audio, 24_000, 16_000)

    assert resampled.dtype == np.float32, resampled.dtype


def test_the_signal_is_preserved_not_truncated(no_librosa: None) -> None:
    """Resampling must resample, not take the first N samples.

    A truncating implementation would also change the length and pass the tests above while
    discarding two thirds of the audio. Checked against a ramp, whose last value is
    position-dependent.
    """
    audio = np.linspace(0.0, 1.0, 48_000, dtype=np.float32)

    resampled = _transcriber()._resample(audio, 48_000, 16_000)

    # Bounded loosely on purpose. A polyphase resampler rings at the boundaries -- a ramp
    # has an implicit step at each end, and scipy's resample_poly overshoots to ~1.08 there.
    # That is correct filter behaviour, so asserting approx(1.0) would fail a working
    # implementation. What this test is actually for is that the output spans the whole
    # input: a truncating implementation would end near 1/3.
    assert resampled[0] < 0.2, f"the head is {resampled[0]}"
    assert resampled[-1] > 0.8, (
        f"the tail is {resampled[-1]}; a truncating implementation would end near 0.33, "
        "having discarded two thirds of the audio"
    )


def test_an_equal_rate_is_a_no_op(no_librosa: None) -> None:
    """Control: nothing to do when the rates already match."""
    audio = np.linspace(0.0, 1.0, 1_000, dtype=np.float32)

    resampled = _transcriber()._resample(audio, 16_000, 16_000)

    assert len(resampled) == len(audio)
    np.testing.assert_allclose(resampled, audio, atol=1e-6)


def test_empty_audio_does_not_raise(no_librosa: None) -> None:
    """Control: a zero-length buffer is a legitimate edge, not an error."""
    resampled = _transcriber()._resample(np.array([], dtype=np.float32), 48_000, 16_000)

    assert len(resampled) == 0


def test_empty_audio_does_not_raise_without_scipy_either(
    no_librosa: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The linear-interpolation leg cannot interpolate zero samples: np.interp raises."""
    monkeypatch.setitem(sys.modules, "scipy", None)

    resampled = _transcriber()._resample(np.array([], dtype=np.float32), 48_000, 16_000)

    assert len(resampled) == 0


@pytest.mark.parametrize(("orig_sr", "target_sr"), [(-48_000, 16_000), (0, 16_000), (48_000, 0)])
def test_a_non_positive_rate_is_rejected(no_librosa: None, orig_sr: int, target_sr: int) -> None:
    """A bad rate must fail loudly, not come back as one sample the caller labels 16 kHz."""
    audio = np.zeros(1_000, dtype=np.float32)

    with pytest.raises(ValueError, match="sample rate"):
        _transcriber()._resample(audio, orig_sr, target_sr)


def test_librosa_is_still_preferred_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """Control: the fallback must not displace librosa where it is installed.

    The fallback is a fallback -- librosa's resampling is higher quality, so it has to stay
    the primary path.
    """
    calls: list[tuple[int, int]] = []

    class _FakeLibrosa:
        """Stands in for librosa and records each resample call."""

        @staticmethod
        def resample(audio: np.ndarray, *, orig_sr: int, target_sr: int) -> np.ndarray:
            """Record the rates and return a zero buffer of the resampled length."""
            calls.append((orig_sr, target_sr))
            return np.zeros(len(audio) * target_sr // orig_sr, dtype=np.float32)

    monkeypatch.setitem(sys.modules, "librosa", _FakeLibrosa)

    audio = np.zeros(48_000, dtype=np.float32)
    resampled = _transcriber()._resample(audio, 48_000, 16_000)

    assert calls == [(48_000, 16_000)], calls
    assert len(resampled) == 16_000
