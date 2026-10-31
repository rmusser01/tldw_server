"""B05 missing local ONNX models must not look ready or consume minutes."""

from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.core import config
from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio import Audio_Files
from tldw_Server_API.tests.Audio.test_audio_transcription_retention_and_redaction import (
    TEST_API_KEY,
    _make_wav_bytes,
    _setup_stubbed_audio_app,
)

MODELS = ("parakeet-tdt-0.6b-v3-onnx", "parakeet-onnx")


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("location", ["absolute", "relative", "file", "windows"])
def test_b05_missing_local_onnx_bundle_is_unavailable(tmp_path, monkeypatch, model, location):
    monkeypatch.chdir(tmp_path)
    reference = {
        "absolute": str(tmp_path / "missing"),
        "relative": "./missing",
        "file": str(tmp_path / "not-a-directory"),
        "windows": r"C:\missing\parakeet",
    }[location]
    if location == "file":
        (tmp_path / "not-a-directory").write_text("not a model")
    monkeypatch.setattr(config, "get_stt_config", lambda: {"parakeet_onnx_model_id": reference})

    status = Audio_Files.check_transcription_model_status(model)

    assert (status["available"], status["usable"], status["on_demand"]) == (False, False, False)  # nosec B101 - Public availability contract.
    assert reference not in status["message"]  # nosec B101 - Private configured paths must not be disclosed.


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("location", ["remote", "directory"])
def test_b05_existing_or_remote_onnx_keeps_first_use_loading(tmp_path, monkeypatch, model, location):
    reference = "istupakov/parakeet-tdt-0.6b-v3-onnx"
    if location == "directory":
        reference = str(tmp_path)
    monkeypatch.setattr(config, "get_stt_config", lambda: {"parakeet_onnx_model_id": reference})

    status = Audio_Files.check_transcription_model_status(model)

    assert status["usable"] is True and status["on_demand"] is True  # nosec B101 - Keep lazy loading and validation, not an ASR qualification.


def test_b05_onnx_path_does_not_disable_other_parakeet_variants(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "get_stt_config", lambda: {"parakeet_onnx_model_id": str(tmp_path / "missing")})
    assert Audio_Files.check_transcription_model_status("parakeet-mlx")["usable"] is True  # nosec B101 - Variant isolation.


@pytest.mark.parametrize("model", MODELS)
def test_b05_missing_local_model_returns_503_before_execution_and_minutes(tmp_path, monkeypatch, model):
    from tldw_Server_API.app.api.v1.endpoints.audio import audio_transcriptions
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio import stt_provider_adapter

    actual_registry = stt_provider_adapter.get_stt_provider_registry()
    adapter = actual_registry.get_adapter("parakeet")
    calls = []

    def reject_execution(*_args, **_kwargs):
        calls.append("provider")
        raise RuntimeError("Unit provider boundary reached")

    app = _setup_stubbed_audio_app(monkeypatch, transcript_text="unused", temp_outputs_dir=tmp_path)
    monkeypatch.setattr(stt_provider_adapter, "get_stt_provider_registry", lambda: actual_registry)
    monkeypatch.setattr(adapter, "transcribe_batch", reject_execution)
    monkeypatch.setattr(config, "get_stt_config", lambda: {"parakeet_onnx_model_id": str(tmp_path / "missing")})
    minutes = AsyncMock(return_value=(True, None))
    monkeypatch.setattr(audio_transcriptions, "_consume_daily_minutes", minutes)
    with TestClient(app, headers={"X-API-KEY": TEST_API_KEY}) as client:
        response = client.post(
            "/api/v1/audio/transcriptions",
            files={"file": ("valid.wav", _make_wav_bytes(), "audio/wav")},
            data={"model": model, "response_format": "json"},
        )
    assert response.status_code == 503  # nosec B101 - Known missing model is unavailable, not transient500.
    assert response.json()["detail"]["status"] == "model_unavailable"  # nosec B101 - Existing stable taxonomy.
    assert str(tmp_path) not in response.text  # nosec B101 - No private configured path in public errors.
    assert calls == []  # nosec B101 - Reject before provider execution.
    minutes.assert_not_awaited()
