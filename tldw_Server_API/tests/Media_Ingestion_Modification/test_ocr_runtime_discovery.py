from __future__ import annotations


def test_list_ocr_backends_matches_response_schema(monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import ocr as ocr_mod
    from tldw_Server_API.app.api.v1.schemas.ocr_schemas import OCRBackendsResponse

    monkeypatch.setattr(
        ocr_mod,
        "_list_backends",
        lambda: {
            "llamacpp": {"available": True},
            "mineru": {"available": False},
        },
    )
    monkeypatch.setattr(
        ocr_mod,
        "_describe_mineru_backend",
        lambda: {
            "available": False,
            "pdf_only": True,
            "document_level": True,
            "opt_in_only": True,
            "supports_per_page_metrics": True,
            "mode": "cli",
            "timeout_sec": 30,
            "max_concurrency": 2,
            "tmp_root": "/tmp/mineru",
            "debug_save_raw": False,
        },
    )

    payload = ocr_mod.list_ocr_backends()
    validated = OCRBackendsResponse.model_validate(payload)

    assert validated.root["llamacpp"].available is True  # nosec B101
    assert validated.root["mineru"].timeout_sec == 30  # nosec B101


def test_list_ocr_backends_enriches_llamacpp_discovery(monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import ocr as ocr_mod

    class _StubLlamaCppBackend:
        def describe(self):
            return {
                "mode": "remote",
                "configured_mode": "auto",
                "model": "vision.gguf",
                "configured": True,
                "supports_structured_output": True,
                "supports_json": True,
                "configured_flags": "--ctx-size 4096",
                "auto_eligible": True,
                "auto_high_quality_eligible": True,
                "url_configured": True,
                "managed_configured": False,
                "managed_running": False,
                "allow_managed_start": False,
                "cli_configured": True,
                "backend_concurrency_cap": 3,
            }

    monkeypatch.setattr(
        ocr_mod,
        "_list_backends",
        lambda: {"llamacpp": {"available": True}},
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.llamacpp_ocr.LlamaCppOCRBackend",
        _StubLlamaCppBackend,
    )

    payload = ocr_mod.list_ocr_backends()

    assert payload["llamacpp"]["available"] is True  # nosec B101
    assert payload["llamacpp"]["mode"] == "remote"  # nosec B101
    assert payload["llamacpp"]["configured_mode"] == "auto"  # nosec B101
    assert payload["llamacpp"]["model"] == "vision.gguf"  # nosec B101
    assert payload["llamacpp"]["configured"] is True  # nosec B101
    assert payload["llamacpp"]["supports_structured_output"] is True  # nosec B101
    assert payload["llamacpp"]["supports_json"] is True  # nosec B101
    assert payload["llamacpp"]["configured_flags"] == "--ctx-size 4096"  # nosec B101
    assert payload["llamacpp"]["auto_eligible"] is True  # nosec B101
    assert payload["llamacpp"]["auto_high_quality_eligible"] is True  # nosec B101
    assert payload["llamacpp"]["url_configured"] is True  # nosec B101
    assert payload["llamacpp"]["managed_configured"] is False  # nosec B101
    assert payload["llamacpp"]["cli_configured"] is True  # nosec B101
    assert payload["llamacpp"]["backend_concurrency_cap"] == 3  # nosec B101


def test_list_ocr_backends_enriches_chatllm_discovery(monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import ocr as ocr_mod

    class _StubChatLLMBackend:
        def describe(self):
            return {
                "mode": "managed",
                "configured": True,
                "supports_structured_output": True,
                "supports_json": True,
                "auto_eligible": True,
                "auto_high_quality_eligible": False,
                "managed_configured": True,
                "managed_running": False,
                "allow_managed_start": True,
                "url_configured": False,
                "healthcheck_url_configured": True,
                "cli_configured": False,
                "backend_concurrency_cap": 2,
                "model": "/models/chatllm.gguf",
            }

    monkeypatch.setattr(
        ocr_mod,
        "_list_backends",
        lambda: {"chatllm": {"available": True}},
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.chatllm_ocr.ChatLLMOCRBackend",
        _StubChatLLMBackend,
    )

    payload = ocr_mod.list_ocr_backends()

    assert payload["chatllm"]["available"] is True  # nosec B101
    assert payload["chatllm"]["mode"] == "managed"  # nosec B101
    assert payload["chatllm"]["configured"] is True  # nosec B101
    assert payload["chatllm"]["supports_structured_output"] is True  # nosec B101
    assert payload["chatllm"]["supports_json"] is True  # nosec B101
    assert payload["chatllm"]["auto_eligible"] is True  # nosec B101
    assert payload["chatllm"]["auto_high_quality_eligible"] is False  # nosec B101
    assert payload["chatllm"]["managed_configured"] is True  # nosec B101
    assert payload["chatllm"]["healthcheck_url_configured"] is True  # nosec B101
    assert payload["chatllm"]["backend_concurrency_cap"] == 2  # nosec B101


def test_list_ocr_backends_records_llamacpp_discovery_errors(monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import ocr as ocr_mod

    class _BrokenLlamaCppBackend:
        def describe(self):
            raise ValueError("llama describe failed")

    monkeypatch.setattr(
        ocr_mod,
        "_list_backends",
        lambda: {"llamacpp": {"available": True}},
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.llamacpp_ocr.LlamaCppOCRBackend",
        _BrokenLlamaCppBackend,
    )

    payload = ocr_mod.list_ocr_backends()

    assert payload["llamacpp"]["available"] is True  # nosec B101
    assert payload["llamacpp"]["error"] == "llama describe failed"  # nosec B101


def test_list_ocr_backends_records_chatllm_discovery_errors(monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import ocr as ocr_mod

    class _BrokenChatLLMBackend:
        def describe(self):
            raise ValueError("chatllm describe failed")

    monkeypatch.setattr(
        ocr_mod,
        "_list_backends",
        lambda: {"chatllm": {"available": False}},
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.backends.chatllm_ocr.ChatLLMOCRBackend",
        _BrokenChatLLMBackend,
    )

    payload = ocr_mod.list_ocr_backends()

    assert payload["chatllm"]["available"] is False  # nosec B101
    assert payload["chatllm"]["error"] == "chatllm describe failed"  # nosec B101
