from __future__ import annotations

import pytest


@pytest.mark.unit
def test_registry_auto_requires_llamacpp_auto_eligible(monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.registry import get_backend

    monkeypatch.delenv("LLAMACPP_OCR_AUTO_ELIGIBLE", raising=False)
    monkeypatch.delenv("LLAMACPP_OCR_AUTO_HIGH_QUALITY_ELIGIBLE", raising=False)
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.llamacpp_ocr.LlamaCppOCRBackend.available",
        classmethod(lambda cls: True),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.tesseract_cli.TesseractCLIBackend.available",
        classmethod(lambda cls: False),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.nemotron_parse.NemotronParseBackend.available",
        classmethod(lambda cls: False),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.dots_ocr.DotsOCRBackend.available",
        classmethod(lambda cls: False),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.points_reader.PointsReaderBackend.available",
        classmethod(lambda cls: False),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.deepseek_ocr.DeepSeekOCRBackend.available",
        classmethod(lambda cls: False),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.hunyuan_ocr.HunyuanOCRBackend.available",
        classmethod(lambda cls: False),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.dolphin_ocr.DolphinOCRBackend.available",
        classmethod(lambda cls: False),
    )

    assert get_backend("auto") is None  # nosec B101


@pytest.mark.unit
def test_registry_auto_selects_llamacpp_when_opted_in(monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.registry import get_backend

    monkeypatch.setenv("LLAMACPP_OCR_AUTO_ELIGIBLE", "true")
    monkeypatch.delenv("LLAMACPP_OCR_AUTO_HIGH_QUALITY_ELIGIBLE", raising=False)
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.llamacpp_ocr.LlamaCppOCRBackend.available",
        classmethod(lambda cls: True),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.tesseract_cli.TesseractCLIBackend.available",
        classmethod(lambda cls: False),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.nemotron_parse.NemotronParseBackend.available",
        classmethod(lambda cls: False),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.dots_ocr.DotsOCRBackend.available",
        classmethod(lambda cls: False),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.points_reader.PointsReaderBackend.available",
        classmethod(lambda cls: False),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.deepseek_ocr.DeepSeekOCRBackend.available",
        classmethod(lambda cls: False),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.hunyuan_ocr.HunyuanOCRBackend.available",
        classmethod(lambda cls: False),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.dolphin_ocr.DolphinOCRBackend.available",
        classmethod(lambda cls: False),
    )

    backend = get_backend("auto")

    assert backend is not None  # nosec B101
    assert getattr(backend, "name", None) == "llamacpp"  # nosec B101


@pytest.mark.unit
def test_registry_auto_high_quality_requires_high_quality_opt_in(monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.registry import get_backend

    monkeypatch.setenv("LLAMACPP_OCR_AUTO_ELIGIBLE", "true")
    monkeypatch.delenv("LLAMACPP_OCR_AUTO_HIGH_QUALITY_ELIGIBLE", raising=False)
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.llamacpp_ocr.LlamaCppOCRBackend.available",
        classmethod(lambda cls: True),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.tesseract_cli.TesseractCLIBackend.available",
        classmethod(lambda cls: False),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.nemotron_parse.NemotronParseBackend.available",
        classmethod(lambda cls: False),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.dots_ocr.DotsOCRBackend.available",
        classmethod(lambda cls: False),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.points_reader.PointsReaderBackend.available",
        classmethod(lambda cls: False),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.deepseek_ocr.DeepSeekOCRBackend.available",
        classmethod(lambda cls: False),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.hunyuan_ocr.HunyuanOCRBackend.available",
        classmethod(lambda cls: False),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.dolphin_ocr.DolphinOCRBackend.available",
        classmethod(lambda cls: False),
    )

    assert get_backend("auto_high_quality") is None  # nosec B101


@pytest.mark.unit
def test_registry_auto_high_quality_selects_llamacpp_when_opted_in(monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.registry import get_backend

    monkeypatch.setenv("LLAMACPP_OCR_AUTO_ELIGIBLE", "true")
    monkeypatch.setenv("LLAMACPP_OCR_AUTO_HIGH_QUALITY_ELIGIBLE", "true")
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.llamacpp_ocr.LlamaCppOCRBackend.available",
        classmethod(lambda cls: True),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.tesseract_cli.TesseractCLIBackend.available",
        classmethod(lambda cls: False),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.nemotron_parse.NemotronParseBackend.available",
        classmethod(lambda cls: False),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.dots_ocr.DotsOCRBackend.available",
        classmethod(lambda cls: False),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.points_reader.PointsReaderBackend.available",
        classmethod(lambda cls: False),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.deepseek_ocr.DeepSeekOCRBackend.available",
        classmethod(lambda cls: False),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.hunyuan_ocr.HunyuanOCRBackend.available",
        classmethod(lambda cls: False),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.dolphin_ocr.DolphinOCRBackend.available",
        classmethod(lambda cls: False),
    )

    backend = get_backend("auto_high_quality")

    assert backend is not None  # nosec B101
    assert getattr(backend, "name", None) == "llamacpp"  # nosec B101


@pytest.mark.unit
def test_registry_auto_high_quality_prefers_llamacpp_in_default_priority(monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.registry import get_backend

    monkeypatch.setenv("LLAMACPP_OCR_AUTO_ELIGIBLE", "true")
    monkeypatch.setenv("LLAMACPP_OCR_AUTO_HIGH_QUALITY_ELIGIBLE", "true")
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.registry._resolve_priority_from_config",
        lambda: None,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.llamacpp_ocr.LlamaCppOCRBackend.available",
        classmethod(lambda cls: True),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.nemotron_parse.NemotronParseBackend.available",
        classmethod(lambda cls: False),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.hunyuan_ocr.HunyuanOCRBackend.available",
        classmethod(lambda cls: False),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.deepseek_ocr.DeepSeekOCRBackend.available",
        classmethod(lambda cls: False),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.points_reader.PointsReaderBackend.available",
        classmethod(lambda cls: True),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.tesseract_cli.TesseractCLIBackend.available",
        classmethod(lambda cls: True),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.dots_ocr.DotsOCRBackend.available",
        classmethod(lambda cls: False),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.dolphin_ocr.DolphinOCRBackend.available",
        classmethod(lambda cls: False),
    )

    backend = get_backend("auto_high_quality")

    assert backend is not None  # nosec B101
    assert getattr(backend, "name", None) == "llamacpp"  # nosec B101


@pytest.mark.unit
def test_registry_auto_requires_chatllm_auto_eligible(monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.registry import get_backend

    monkeypatch.delenv("CHATLLM_OCR_AUTO_ELIGIBLE", raising=False)
    monkeypatch.delenv("CHATLLM_OCR_AUTO_HIGH_QUALITY_ELIGIBLE", raising=False)
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.chatllm_ocr.ChatLLMOCRBackend.available",
        classmethod(lambda cls: True),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.tesseract_cli.TesseractCLIBackend.available",
        classmethod(lambda cls: False),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.nemotron_parse.NemotronParseBackend.available",
        classmethod(lambda cls: False),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.dots_ocr.DotsOCRBackend.available",
        classmethod(lambda cls: False),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.points_reader.PointsReaderBackend.available",
        classmethod(lambda cls: False),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.deepseek_ocr.DeepSeekOCRBackend.available",
        classmethod(lambda cls: False),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.hunyuan_ocr.HunyuanOCRBackend.available",
        classmethod(lambda cls: False),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.llamacpp_ocr.LlamaCppOCRBackend.available",
        classmethod(lambda cls: False),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.dolphin_ocr.DolphinOCRBackend.available",
        classmethod(lambda cls: False),
    )

    assert get_backend("auto") is None  # nosec B101


@pytest.mark.unit
def test_registry_auto_selects_chatllm_when_opted_in(monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.registry import get_backend

    monkeypatch.setenv("CHATLLM_OCR_AUTO_ELIGIBLE", "true")
    monkeypatch.delenv("CHATLLM_OCR_AUTO_HIGH_QUALITY_ELIGIBLE", raising=False)
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.chatllm_ocr.ChatLLMOCRBackend.available",
        classmethod(lambda cls: True),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.tesseract_cli.TesseractCLIBackend.available",
        classmethod(lambda cls: False),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.nemotron_parse.NemotronParseBackend.available",
        classmethod(lambda cls: False),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.dots_ocr.DotsOCRBackend.available",
        classmethod(lambda cls: False),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.points_reader.PointsReaderBackend.available",
        classmethod(lambda cls: False),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.deepseek_ocr.DeepSeekOCRBackend.available",
        classmethod(lambda cls: False),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.hunyuan_ocr.HunyuanOCRBackend.available",
        classmethod(lambda cls: False),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.llamacpp_ocr.LlamaCppOCRBackend.available",
        classmethod(lambda cls: False),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.dolphin_ocr.DolphinOCRBackend.available",
        classmethod(lambda cls: False),
    )

    backend = get_backend("auto")

    assert backend is not None  # nosec B101
    assert getattr(backend, "name", None) == "chatllm"  # nosec B101


@pytest.mark.unit
def test_registry_auto_high_quality_requires_chatllm_high_quality_opt_in(monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.registry import get_backend

    monkeypatch.setenv("CHATLLM_OCR_AUTO_ELIGIBLE", "true")
    monkeypatch.delenv("CHATLLM_OCR_AUTO_HIGH_QUALITY_ELIGIBLE", raising=False)
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.chatllm_ocr.ChatLLMOCRBackend.available",
        classmethod(lambda cls: True),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.tesseract_cli.TesseractCLIBackend.available",
        classmethod(lambda cls: False),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.nemotron_parse.NemotronParseBackend.available",
        classmethod(lambda cls: False),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.dots_ocr.DotsOCRBackend.available",
        classmethod(lambda cls: False),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.points_reader.PointsReaderBackend.available",
        classmethod(lambda cls: False),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.deepseek_ocr.DeepSeekOCRBackend.available",
        classmethod(lambda cls: False),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.hunyuan_ocr.HunyuanOCRBackend.available",
        classmethod(lambda cls: False),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.llamacpp_ocr.LlamaCppOCRBackend.available",
        classmethod(lambda cls: False),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.dolphin_ocr.DolphinOCRBackend.available",
        classmethod(lambda cls: False),
    )

    assert get_backend("auto_high_quality") is None  # nosec B101


@pytest.mark.unit
def test_registry_auto_high_quality_selects_chatllm_when_opted_in(monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.registry import get_backend

    monkeypatch.setenv("CHATLLM_OCR_AUTO_ELIGIBLE", "true")
    monkeypatch.setenv("CHATLLM_OCR_AUTO_HIGH_QUALITY_ELIGIBLE", "true")
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.chatllm_ocr.ChatLLMOCRBackend.available",
        classmethod(lambda cls: True),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.tesseract_cli.TesseractCLIBackend.available",
        classmethod(lambda cls: False),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.nemotron_parse.NemotronParseBackend.available",
        classmethod(lambda cls: False),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.dots_ocr.DotsOCRBackend.available",
        classmethod(lambda cls: False),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.points_reader.PointsReaderBackend.available",
        classmethod(lambda cls: False),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.deepseek_ocr.DeepSeekOCRBackend.available",
        classmethod(lambda cls: False),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.hunyuan_ocr.HunyuanOCRBackend.available",
        classmethod(lambda cls: False),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.llamacpp_ocr.LlamaCppOCRBackend.available",
        classmethod(lambda cls: False),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.dolphin_ocr.DolphinOCRBackend.available",
        classmethod(lambda cls: False),
    )

    backend = get_backend("auto_high_quality")

    assert backend is not None  # nosec B101
    assert getattr(backend, "name", None) == "chatllm"  # nosec B101


@pytest.mark.unit
def test_registry_auto_high_quality_prefers_llamacpp_before_chatllm(monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.registry import get_backend

    monkeypatch.setenv("LLAMACPP_OCR_AUTO_ELIGIBLE", "true")
    monkeypatch.setenv("LLAMACPP_OCR_AUTO_HIGH_QUALITY_ELIGIBLE", "true")
    monkeypatch.setenv("CHATLLM_OCR_AUTO_ELIGIBLE", "true")
    monkeypatch.setenv("CHATLLM_OCR_AUTO_HIGH_QUALITY_ELIGIBLE", "true")
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.registry._resolve_priority_from_config",
        lambda: None,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.llamacpp_ocr.LlamaCppOCRBackend.available",
        classmethod(lambda cls: True),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.chatllm_ocr.ChatLLMOCRBackend.available",
        classmethod(lambda cls: True),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.nemotron_parse.NemotronParseBackend.available",
        classmethod(lambda cls: True),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.hunyuan_ocr.HunyuanOCRBackend.available",
        classmethod(lambda cls: True),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.deepseek_ocr.DeepSeekOCRBackend.available",
        classmethod(lambda cls: True),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.points_reader.PointsReaderBackend.available",
        classmethod(lambda cls: True),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.tesseract_cli.TesseractCLIBackend.available",
        classmethod(lambda cls: True),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.dots_ocr.DotsOCRBackend.available",
        classmethod(lambda cls: True),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.dolphin_ocr.DolphinOCRBackend.available",
        classmethod(lambda cls: True),
    )

    backend = get_backend("auto_high_quality")

    assert backend is not None  # nosec B101
    assert getattr(backend, "name", None) == "llamacpp"  # nosec B101
