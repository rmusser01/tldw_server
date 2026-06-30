"""Regression tests for Parakeet ONNX fail-fast batch behavior."""

import importlib.machinery
import sys
import types

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
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Whisper fallback should not run for Parakeet ONNX")
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
    audio_file.write_bytes(b"\x00" * 2048)

    fake_librosa = types.SimpleNamespace(
        get_duration=lambda path: 1.0,
        load=lambda path, sr=16000, mono=True: (np.zeros(1600, dtype=np.float32), sr),
    )
    monkeypatch.setitem(sys.modules, "librosa", fake_librosa)

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
