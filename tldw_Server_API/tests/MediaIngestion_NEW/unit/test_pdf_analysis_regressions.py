"""Regression tests for PDF analysis edge cases in media ingestion."""

import pytest
from _pytest.monkeypatch import MonkeyPatch


@pytest.mark.asyncio
@pytest.mark.unit
async def test_pdf_analysis_empty_chunk_responses_do_not_crash(
    monkeypatch: MonkeyPatch,
) -> None:
    """Verify empty analyzer responses do not leave final_summary unbound."""
    from tldw_Server_API.app.core.Ingestion_Media_Processing.PDF import (
        PDF_Processing_Lib as pdf_mod,
    )

    monkeypatch.setattr(pdf_mod, "pymupdf4llm_parse_pdf", lambda path: "Some extracted content")
    monkeypatch.setattr(pdf_mod, "extract_metadata_from_pdf", lambda path: {})
    monkeypatch.setattr(pdf_mod, "analyze", lambda **kwargs: "")

    class _FakePdfDoc:
        """Minimal context-manager PDF document stub."""

        metadata = {}
        page_count = 1

        def __enter__(self) -> "_FakePdfDoc":
            """Return this fake PDF document."""
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
            """Do not suppress exceptions raised by the caller."""
            return False

    monkeypatch.setattr(pdf_mod.pymupdf, "open", lambda *args, **kwargs: _FakePdfDoc())

    out = await pdf_mod.process_pdf_task(
        file_bytes=b"%PDF-fake",
        filename="paper.pdf",
        parser="pymupdf4llm",
        perform_chunking=False,
        perform_analysis=True,
        api_name="openai",
        api_key=None,
    )

    assert out.get("status") in ("Success", "Warning"), out
    assert out.get("analysis") is None
    assert out["chunks"][0]["metadata"]["analysis"] is None


@pytest.mark.asyncio
async def test_pdf_cancellation_skips_pending_ocr_pages(monkeypatch, tmp_path):
    from threading import Event

    import pymupdf

    from tldw_Server_API.app.core.Ingestion_Media_Processing.PDF import PDF_Processing_Lib as pdf

    source = tmp_path / "scans.pdf"
    with pymupdf.open() as doc:
        for _ in range(3):
            doc.new_page(width=72, height=72)
        doc.save(source)

    cancelled = Event()
    calls = []

    class OCR:
        def ocr_image(self, *_args):
            calls.append("ocr")
            cancelled.set()
            return "Scanned text"

    monkeypatch.setattr(pdf, "pymupdf4llm_parse_pdf", lambda _: "")
    monkeypatch.setattr(pdf, "_get_ocr_backend", lambda _: OCR())
    monkeypatch.setenv("OCR_PAGE_CONCURRENCY", "1")

    from tldw_Server_API.app.core.Ingestion_Media_Processing.PDF.job_progress import pdf_job_progress

    with pdf_job_progress(cancelled.is_set):
        result = await pdf.process_pdf_task(
            file_bytes=source.read_bytes(),
            filename=source.name,
            enable_ocr=True,
            ocr_mode="always",
            ocr_dpi=72,
        )

    assert result["status"] == "Cancelled"
    assert calls == ["ocr"]
