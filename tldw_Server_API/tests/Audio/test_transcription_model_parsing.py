import pytest

import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib as atlib

from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib import (
    parse_transcription_model,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("model_name", "expected_model"),
    [
        ("whisper-1", "large-v3"),
        ("whisper-large-v3", "large-v3"),
        ("whisper-large-v3-ct2", "large-v3"),
        ("whisper-base-ct2", "base"),
        ("whisper-tiny-ct2", "tiny"),
        ("whisper-large-v3-turbo", "deepdml/faster-whisper-large-v3-turbo-ct2"),
        ("whisper-large-v3-turbo-ct2", "deepdml/faster-whisper-large-v3-turbo-ct2"),
    ],
)
def test_parse_transcription_model_whisper_aliases(model_name: str, expected_model: str) -> None:
    provider, model, variant = parse_transcription_model(model_name)
    assert provider == "whisper"
    assert model == expected_model
    assert variant is None


@pytest.mark.unit
@pytest.mark.parametrize(
    ("model_name", "expected_model"),
    [
        ("vibevoice", "microsoft/VibeVoice-ASR"),
        ("vibevoice-asr", "microsoft/VibeVoice-ASR"),
        ("microsoft/VibeVoice-ASR", "microsoft/VibeVoice-ASR"),
    ],
)
def test_parse_transcription_model_vibevoice_aliases(
    monkeypatch: pytest.MonkeyPatch, model_name: str, expected_model: str
) -> None:
    monkeypatch.setattr(
        atlib,
        "get_stt_config",
        lambda: {"vibevoice_model_id": "microsoft/VibeVoice-ASR"},
        raising=True,
    )

    provider, model, variant = atlib.parse_transcription_model(model_name)
    assert provider == "vibevoice"
    assert model == expected_model
    assert variant is None


@pytest.mark.unit
@pytest.mark.parametrize("model_name", ["parakeet-tdt-0.6b-v3-onnx", "parakeet-onnx"])
def test_parse_transcription_model_parakeet_onnx_aliases(model_name: str) -> None:
    provider, model, variant = parse_transcription_model(model_name)
    assert provider == "parakeet"
    assert model == "parakeet"
    assert variant == "onnx"
