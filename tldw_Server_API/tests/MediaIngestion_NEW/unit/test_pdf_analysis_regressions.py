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
