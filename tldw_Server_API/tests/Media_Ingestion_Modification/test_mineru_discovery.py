from __future__ import annotations


def test_list_ocr_backends_includes_mineru_capabilities(monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import ocr as ocr_mod

    monkeypatch.setattr(
        ocr_mod,
        "_list_backends",
        lambda: {"tesseract": {"available": True}},
    )
    monkeypatch.setattr(
        ocr_mod,
        "_describe_mineru_backend",
        lambda: {
            "available": True,
            "pdf_only": True,
            "document_level": True,
            "opt_in_only": True,
            "supports_per_page_metrics": True,
            "mode": "cli",
        },
        raising=False,
    )

    payload = ocr_mod.list_ocr_backends()

    assert payload["mineru"]["available"] is True
    assert payload["mineru"]["pdf_only"] is True
    assert payload["mineru"]["document_level"] is True
    assert payload["mineru"]["opt_in_only"] is True
    assert payload["mineru"]["supports_per_page_metrics"] is True
    assert payload["mineru"]["mode"] == "cli"


def test_list_ocr_backends_returns_safe_mineru_stub_on_describe_error(monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import ocr as ocr_mod

    def _raise_describe_error():
        raise ValueError("bad MinerU command")

    monkeypatch.setattr(
        ocr_mod,
        "_list_backends",
        lambda: {"tesseract": {"available": True}},
    )
    monkeypatch.setattr(
        ocr_mod,
        "_describe_mineru_backend",
        _raise_describe_error,
        raising=False,
    )

    payload = ocr_mod.list_ocr_backends()

    assert payload["tesseract"]["available"] is True
    assert payload["mineru"]["available"] is False
    assert payload["mineru"]["pdf_only"] is True
    assert payload["mineru"]["document_level"] is True
    assert payload["mineru"]["opt_in_only"] is True
    assert payload["mineru"]["mode"] == "cli"
    assert payload["mineru"]["error"] == "MinerU backend discovery failed"
    assert "bad MinerU command" not in payload["mineru"]["error"]
