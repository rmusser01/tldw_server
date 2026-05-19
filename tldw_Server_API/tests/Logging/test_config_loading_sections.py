from __future__ import annotations

from tldw_Server_API.app.core.config import load_and_log_configs


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
    assert stt["default_batch_transcription_model"] == "parakeet-tdt-0.6b-v3-onnx"
    assert stt["default_streaming_transcription_model"] == "parakeet-tdt-0.6b-v3-onnx"
    assert stt["nemo_model_variant"] == "onnx"
