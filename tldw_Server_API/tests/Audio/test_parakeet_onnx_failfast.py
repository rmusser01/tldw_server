"""Regression tests for Parakeet ONNX fail-fast batch behavior."""

import importlib.machinery
import sys
import types
import wave
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

# Stub heavyweight audio deps before importing transcription library modules.
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

import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib as atlib


def _write_test_wav(
    audio_file: Path,
    *,
    sample_rate: int = 16000,
    duration_seconds: float = 0.25,
    channels: int = 1,
) -> None:
    """Write a small sine-wave PCM WAV fixture for audio-loader tests."""
    frame_count = int(sample_rate * duration_seconds)
    mono = 0.1 * np.sin(2.0 * np.pi * 440.0 * np.arange(frame_count) / sample_rate)
    if channels == 1:
        audio = mono[:, np.newaxis]
    else:
        audio = np.tile(mono[:, np.newaxis], (1, channels))
    pcm_i16 = np.clip(audio * 32767.0, -32768, 32767).astype("<i2")

    with wave.open(str(audio_file), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_i16.tobytes())


@pytest.mark.unit
def test_speech_to_text_parakeet_onnx_failure_fails_fast(monkeypatch, tmp_path):
    """Parakeet ONNX failures should surface as STTTranscriptionError."""
    audio_file = tmp_path / "sample.wav"
    audio_file.write_bytes(b"\x00" * 2048)

    def fake_parakeet(*_args, **_kwargs):
        raise RuntimeError("onnx init failed")

    monkeypatch.setattr(atlib, "speech_to_text_parakeet", fake_parakeet)
    monkeypatch.setattr(
        atlib,
        "get_whisper_model",
        Mock(
            side_effect=AssertionError(
                "Whisper fallback should not run for Parakeet ONNX"
            )
        ),
    )

    with pytest.raises(atlib.STTTranscriptionError):
        atlib.speech_to_text(
            str(audio_file),
            whisper_model="parakeet-onnx",
            selected_source_lang="en",
        )


@pytest.mark.unit
def test_speech_to_text_parakeet_onnx_error_sentinel_fails_fast(monkeypatch, tmp_path):
    """Parakeet ONNX error sentinel text should not be converted into transcript segments."""
    audio_file = tmp_path / "sample.wav"
    _write_test_wav(audio_file)

    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Nemo as nemo_mod

    sentinel = (
        "[Error: Transcription failed: Required inputs (['waveforms_lens']) "
        "are missing from input feed (['waveforms']).]"
    )
    monkeypatch.setattr(
        nemo_mod,
        "transcribe_with_parakeet",
        lambda *_args, **_kwargs: sentinel,
    )

    with pytest.raises(atlib.STTTranscriptionError, match="waveforms_lens") as exc_info:
        atlib.speech_to_text_parakeet(
            str(audio_file),
            variant="onnx",
            selected_source_lang="en",
            vad_filter=False,
        )
    assert str(exc_info.value) == sentinel  # nosec B101


@pytest.mark.unit
def test_speech_to_text_parakeet_onnx_loads_audio_without_librosa(monkeypatch, tmp_path):
    """Parakeet ONNX should not route file decoding through librosa."""
    audio_file = tmp_path / "sample.wav"
    _write_test_wav(audio_file, sample_rate=8000, channels=2)

    fake_librosa = types.SimpleNamespace(
        get_duration=Mock(
            side_effect=AssertionError(
                "librosa.get_duration should not be used for Parakeet ONNX"
            )
        ),
        load=Mock(
            side_effect=AssertionError(
                "librosa.load should not be used for Parakeet ONNX"
            )
        ),
    )
    monkeypatch.setitem(sys.modules, "librosa", fake_librosa)

    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Nemo as nemo_mod

    captured: dict[str, object] = {}

    def fake_transcribe_with_parakeet(audio_data, received_sample_rate, variant, **kwargs):
        captured["audio_data"] = audio_data
        captured["sample_rate"] = received_sample_rate
        captured["variant"] = variant
        captured["kwargs"] = kwargs
        return "parakeet onnx transcript"

    monkeypatch.setattr(nemo_mod, "transcribe_with_parakeet", fake_transcribe_with_parakeet)

    segments = atlib.speech_to_text_parakeet(
        str(audio_file),
        variant="onnx",
        selected_source_lang="en",
        vad_filter=False,
        base_dir=tmp_path,
    )

    audio_data = captured["audio_data"]
    assert isinstance(audio_data, np.ndarray)
    assert audio_data.dtype == np.float32
    assert audio_data.ndim == 1
    assert captured["sample_rate"] == 16000
    assert captured["variant"] == "onnx"
    assert segments[0]["Text"] == "parakeet onnx transcript"


@pytest.mark.unit
def test_speech_to_text_parakeet_onnx_passes_configured_chunking(monkeypatch, tmp_path):
    """Long-form Parakeet ONNX should receive bounded chunking settings."""
    audio_file = tmp_path / "sample.wav"
    _write_test_wav(audio_file, sample_rate=16000, duration_seconds=2.0)

    monkeypatch.setattr(
        atlib,
        "get_stt_config",
        lambda: {
            "nemo_chunk_duration": 1.0,
            "nemo_overlap_duration": 0.25,
        },
    )

    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Nemo as nemo_mod

    captured: dict[str, object] = {}

    def fake_transcribe_with_parakeet(audio_data, received_sample_rate, variant, **kwargs):
        captured["audio_data"] = audio_data
        captured["sample_rate"] = received_sample_rate
        captured["variant"] = variant
        captured["kwargs"] = kwargs
        return "chunked parakeet onnx transcript"

    monkeypatch.setattr(nemo_mod, "transcribe_with_parakeet", fake_transcribe_with_parakeet)

    segments = atlib.speech_to_text_parakeet(
        str(audio_file),
        variant="onnx",
        selected_source_lang="en",
        vad_filter=False,
        base_dir=tmp_path,
    )

    assert isinstance(captured["audio_data"], np.ndarray)
    assert captured["sample_rate"] == 16000
    assert captured["variant"] == "onnx"
    assert captured["kwargs"] == {
        "chunk_duration": 1.0,
        "overlap_duration": 0.25,
    }
    assert segments[0]["Text"] == "chunked parakeet onnx transcript"


@pytest.mark.unit
def test_parakeet_onnx_resample_uses_linear_fallback_for_large_polyphase(
    monkeypatch,
) -> None:
    """Pathological sample-rate ratios should not call SciPy polyphase resampling."""

    class FakeSignal:
        """Signal module stand-in that fails if polyphase resampling is used."""

        @staticmethod
        def resample_poly(*_args, **_kwargs):
            raise AssertionError("resample_poly should not be called")

    monkeypatch.setitem(sys.modules, "scipy", types.SimpleNamespace(signal=FakeSignal))
    audio = np.linspace(-0.25, 0.25, num=44101, endpoint=False, dtype=np.float32)

    result = atlib._resample_audio_without_librosa(audio, sample_rate=44101, target_sr=16000)

    assert result.dtype == np.float32
    assert result.shape == (16000,)


@pytest.mark.unit
def test_load_audio_for_parakeet_nemo_wraps_unexpected_soundfile_errors(
    monkeypatch,
    tmp_path,
) -> None:
    """Unexpected soundfile failures should be converted to STTTranscriptionError."""

    class CustomAudioLoadError(Exception):
        """Non-standard soundfile failure used to exercise broad wrapping."""

    fake_soundfile = types.SimpleNamespace(
        read=Mock(side_effect=CustomAudioLoadError("decoder exploded")),
    )
    monkeypatch.setitem(sys.modules, "soundfile", fake_soundfile)

    with pytest.raises(atlib.STTTranscriptionError, match="audio loading failed"):
        atlib._load_audio_for_parakeet_nemo(str(tmp_path / "input.wav"))
