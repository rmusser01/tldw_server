from __future__ import annotations

import importlib.util
import json

import pytest


@pytest.mark.unit
def test_ocr_backend_auto_eligible_defaults_true():
    from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.base import OCRBackend

    assert OCRBackend.auto_eligible(False) is True  # nosec B101
    assert OCRBackend.auto_eligible(True) is True  # nosec B101


@pytest.mark.unit
def test_registry_explicit_hunyuan_selection_bypasses_auto_eligible(monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.registry import get_backend

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.hunyuan_ocr.HunyuanOCRBackend.available",
        classmethod(lambda cls: True),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.hunyuan_ocr.HunyuanOCRBackend.auto_eligible",
        classmethod(lambda cls, high_quality: False),
        raising=False,
    )

    backend = get_backend("hunyuan")

    assert backend is not None  # nosec B101
    assert getattr(backend, "name", None) == "hunyuan"  # nosec B101


@pytest.mark.unit
def test_hunyuan_native_availability_requires_explicit_transformers_intent(monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.hunyuan_ocr import (
        HunyuanOCRBackend,
    )

    monkeypatch.setenv("HUNYUAN_RUNTIME_FAMILY", "native")
    monkeypatch.setenv("HUNYUAN_MODE", "auto")
    monkeypatch.delenv("HUNYUAN_VLLM_URL", raising=False)
    monkeypatch.delenv("HUNYUAN_MODEL_PATH", raising=False)
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: object() if name in {"transformers", "torch", "PIL"} else None,
    )

    assert HunyuanOCRBackend.available() is False  # nosec B101


@pytest.mark.unit
def test_hunyuan_llamacpp_family_available_when_remote_runtime_is_configured(monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.hunyuan_ocr import (
        HunyuanOCRBackend,
    )

    monkeypatch.setenv("HUNYUAN_RUNTIME_FAMILY", "llamacpp")
    monkeypatch.setenv("HUNYUAN_LLAMACPP_MODE", "remote")
    monkeypatch.setenv("HUNYUAN_LLAMACPP_HOST", "127.0.0.1")
    monkeypatch.setenv("HUNYUAN_LLAMACPP_PORT", "8088")
    monkeypatch.setenv("HUNYUAN_LLAMACPP_MODEL", "ggml-org/HunyuanOCR-GGUF:Q8_0")
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)

    assert HunyuanOCRBackend.available() is True  # nosec B101


@pytest.mark.unit
def test_hunyuan_auto_eligible_uses_llamacpp_opt_in_flags(monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.hunyuan_ocr import (
        HunyuanOCRBackend,
    )

    monkeypatch.setenv("HUNYUAN_RUNTIME_FAMILY", "llamacpp")
    monkeypatch.setenv("HUNYUAN_LLAMACPP_MODE", "remote")
    monkeypatch.setenv("HUNYUAN_LLAMACPP_HOST", "127.0.0.1")
    monkeypatch.setenv("HUNYUAN_LLAMACPP_PORT", "8088")
    monkeypatch.setenv("HUNYUAN_LLAMACPP_MODEL", "ggml-org/HunyuanOCR-GGUF:Q8_0")
    monkeypatch.setenv("HUNYUAN_LLAMACPP_AUTO_ELIGIBLE", "false")
    monkeypatch.setenv("HUNYUAN_LLAMACPP_AUTO_HIGH_QUALITY_ELIGIBLE", "true")

    assert HunyuanOCRBackend.auto_eligible(False) is False  # nosec B101
    assert HunyuanOCRBackend.auto_eligible(True) is True  # nosec B101


@pytest.mark.unit
def test_hunyuan_llamacpp_remote_structured_json_uses_hunyuan_parser(monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.hunyuan_ocr import (
        HunyuanOCRBackend,
    )

    monkeypatch.setenv("HUNYUAN_RUNTIME_FAMILY", "llamacpp")
    monkeypatch.setenv("HUNYUAN_LLAMACPP_MODE", "remote")
    monkeypatch.setenv("HUNYUAN_LLAMACPP_HOST", "127.0.0.1")
    monkeypatch.setenv("HUNYUAN_LLAMACPP_PORT", "8088")
    monkeypatch.setenv("HUNYUAN_LLAMACPP_MODEL", "ggml-org/HunyuanOCR-GGUF:Q8_0")
    monkeypatch.setenv("HUNYUAN_LLAMACPP_USE_DATA_URL", "true")

    captured: dict[str, object] = {}

    def fake_fetch_json(*, method, url, json, timeout):
        captured["method"] = method
        captured["url"] = url
        captured["model"] = json["model"]
        return {
            "choices": [
                {
                    "message": {
                        "content": json_module.dumps(
                            {
                                "text": "hello from gguf",
                                "blocks": [{"text": "hello from gguf", "bbox": [1, 2, 3, 4]}],
                            }
                        )
                    }
                }
            ]
        }

    json_module = json
    monkeypatch.setattr("tldw_Server_API.app.core.http_client.fetch_json", fake_fetch_json)

    result = HunyuanOCRBackend().ocr_image_structured(
        b"png-bytes",
        output_format="json",
        prompt_preset="json",
    )

    assert captured["method"] == "POST"  # nosec B101
    assert captured["url"] == "http://127.0.0.1:8088/v1/chat/completions"  # nosec B101
    assert captured["model"] == "ggml-org/HunyuanOCR-GGUF:Q8_0"  # nosec B101
    assert result.text == "hello from gguf"  # nosec B101
    assert result.format == "json"  # nosec B101
    assert result.raw == {"text": "hello from gguf", "blocks": [{"text": "hello from gguf", "bbox": [1, 2, 3, 4]}]}  # nosec B101
