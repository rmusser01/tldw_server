from __future__ import annotations

import pytest

from tldw_Server_API.app.core.Ingestion_Media_Processing.transcription_models import (
    get_transcription_models_payload,
)


@pytest.mark.unit
def test_transcription_models_include_canonical_and_legacy_parakeet_onnx_aliases():
    payload = get_transcription_models_payload()
    all_models = payload["all_models"]

    assert "parakeet-tdt-0.6b-v3-onnx" in all_models
    assert "parakeet-onnx" in all_models

    parakeet_models = payload["categories"]["Parakeet Backends"]
    canonical = next(
        model for model in parakeet_models if model["value"] == "parakeet-tdt-0.6b-v3-onnx"
    )
    assert canonical["label"] == "Parakeet TDT 0.6B V3 ONNX"
