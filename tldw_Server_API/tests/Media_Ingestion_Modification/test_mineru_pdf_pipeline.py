from __future__ import annotations

import pytest


def _build_minimal_pdf_bytes() -> bytes:
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page(width=300, height=200)
    page.insert_text((72, 72), "Hello from parser")
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


@pytest.mark.unit
@pytest.mark.asyncio
async def test_process_pdf_uses_mineru_document_adapter_for_always(monkeypatch):
    try:
        from tldw_Server_API.app.core.Ingestion_Media_Processing.PDF import PDF_Processing_Lib as pdf_lib
    except Exception as exc:
        pytest.skip(f"Dependencies not available: {exc}")

    monkeypatch.setattr(pdf_lib, "pymupdf4llm_parse_pdf", lambda path, **kwargs: "parser text")
    monkeypatch.setattr(
        pdf_lib,
        "_run_mineru_document_ocr",
        lambda **kwargs: {
            "text": "# MinerU Markdown",
            "structured": {
                "schema_version": 1,
                "format": "markdown",
                "text": "# MinerU Markdown",
                "pages": [{"page": 1, "text": "MinerU page"}],
                "tables": [],
                "artifacts": {},
                "meta": {"backend": "mineru", "supports_per_page_metrics": True},
            },
            "details": {"backend": "mineru", "mode": "always"},
            "warnings": [],
        },
        raising=False,
    )

    res = await pdf_lib.process_pdf_task(
        file_bytes=_build_minimal_pdf_bytes(),
        filename="mineru.pdf",
        parser="pymupdf4llm",
        perform_chunking=False,
        perform_analysis=False,
        enable_ocr=True,
        ocr_backend="mineru",
        ocr_mode="always",
    )

    assert res["content"] == "# MinerU Markdown"
    assert res["analysis_details"]["ocr"]["backend"] == "mineru"
    assert res["analysis_details"]["ocr"]["structured"]["schema_version"] == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_process_pdf_mineru_fallback_preserves_parser_text_on_failure(monkeypatch):
    try:
        from tldw_Server_API.app.core.Ingestion_Media_Processing.PDF import PDF_Processing_Lib as pdf_lib
    except Exception as exc:
        pytest.skip(f"Dependencies not available: {exc}")

    monkeypatch.setattr(pdf_lib, "pymupdf4llm_parse_pdf", lambda path, **kwargs: "parser text")

    def _raise_mineru(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        pdf_lib,
        "_run_mineru_document_ocr",
        _raise_mineru,
        raising=False,
    )

    res = await pdf_lib.process_pdf_task(
        file_bytes=_build_minimal_pdf_bytes(),
        filename="mineru.pdf",
        parser="pymupdf4llm",
        perform_chunking=False,
        perform_analysis=False,
        enable_ocr=True,
        ocr_backend="mineru",
        ocr_mode="fallback",
        ocr_min_page_text_chars=9999,
    )

    assert res["content"] == "parser text"
    assert any("OCR" in warning or "MinerU" in warning for warning in res["warnings"])


@pytest.mark.unit
@pytest.mark.asyncio
async def test_process_pdf_mineru_fallback_skips_ocr_when_parser_text_is_sufficient(monkeypatch):
    try:
        from tldw_Server_API.app.core.Ingestion_Media_Processing.PDF import PDF_Processing_Lib as pdf_lib
    except Exception as exc:
        pytest.skip(f"Dependencies not available: {exc}")

    mineru_calls = {"count": 0}

    def _fake_mineru(**kwargs):
        mineru_calls["count"] += 1
        return {"text": "unexpected mineru text", "structured": {}, "details": {}, "warnings": []}

    monkeypatch.setattr(pdf_lib, "pymupdf4llm_parse_pdf", lambda path, **kwargs: "parser text is already long enough")
    monkeypatch.setattr(
        pdf_lib,
        "_run_mineru_document_ocr",
        _fake_mineru,
        raising=False,
    )

    res = await pdf_lib.process_pdf_task(
        file_bytes=_build_minimal_pdf_bytes(),
        filename="mineru.pdf",
        parser="pymupdf4llm",
        perform_chunking=False,
        perform_analysis=False,
        enable_ocr=True,
        ocr_backend="mineru",
        ocr_mode="fallback",
        ocr_min_page_text_chars=5,
    )

    assert mineru_calls["count"] == 0
    assert res["content"] == "parser text is already long enough"
