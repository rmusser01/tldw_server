from __future__ import annotations

from tldw_Server_API.app.core import config as config_module
from tldw_Server_API.app.core.config import load_and_log_configs


def test_auto_transcription_default_uses_mlx_on_macos(monkeypatch):
    monkeypatch.setattr(config_module.platform, "system", lambda: "Darwin")

    assert config_module.resolve_default_transcription_model_setting("auto") == "parakeet-mlx"


def test_auto_transcription_default_uses_onnx_off_macos(monkeypatch):
    monkeypatch.setattr(config_module.platform, "system", lambda: "Linux")
    assert (
        config_module.resolve_default_transcription_model_setting("auto")
        == "parakeet-tdt-0.6b-v3-onnx"
    )

    monkeypatch.setattr(config_module.platform, "system", lambda: "Windows")
    assert (
        config_module.resolve_default_transcription_model_setting("")
        == "parakeet-tdt-0.6b-v3-onnx"
    )


def test_explicit_transcription_default_is_preserved(monkeypatch):
    monkeypatch.setattr(config_module.platform, "system", lambda: "Darwin")

    assert (
        config_module.resolve_default_transcription_model_setting("parakeet-onnx")
        == "parakeet-onnx"
    )


def test_load_and_log_configs_includes_section_dicts():
    data = load_and_log_configs()
    assert isinstance(data, dict)
    assert isinstance(data.get("Redis"), dict)
    assert isinstance(data.get("Web-Scraping"), dict)


def test_load_and_log_configs_exposes_stt_default_model_keys():
    data = load_and_log_configs()
    stt = data.get("STT-Settings") or {}
    assert isinstance(stt, dict)
    assert "default_batch_transcription_model" in stt
    assert "default_streaming_transcription_model" in stt
    assert "parakeet_onnx_model_id" in stt
    assert "parakeet_onnx_revision" in stt
    assert "nemo_chunk_duration" in stt
    assert "nemo_overlap_duration" in stt
    expected_default = config_module.resolve_default_transcription_model_setting("auto")
    assert stt["default_batch_transcription_model"] == expected_default
    assert stt["default_streaming_transcription_model"] == expected_default
    assert stt["nemo_model_variant"] == "onnx"
    assert stt["nemo_chunk_duration"] == 120.0
    assert stt["nemo_overlap_duration"] == 15.0
