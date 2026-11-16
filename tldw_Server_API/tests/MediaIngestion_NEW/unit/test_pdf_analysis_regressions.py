"""Regression tests for PDF analysis edge cases in media ingestion."""

import pytest
from _pytest.monkeypatch import MonkeyPatch


@pytest.mark.unit
@pytest.mark.parametrize(
    "parser,docling_mode,expected_calls",
    [
        ("pymupdf4llm", "unused", ["pymupdf4llm"]),
        ("pymupdf", "unused", ["pymupdf"]),
        ("docling", "ready", ["docling"]),
        ("docling", "no_torch", ["pymupdf4llm"]),
        ("docling", "fails", ["docling", "pymupdf4llm"]),
        ("docling", "missing", []),
        ("unsupported", "unused", ["pymupdf4llm"]),
    ],
)
def test_pdf_parser_selection_preserves_fallback_and_page_metadata(
    tmp_path, monkeypatch: MonkeyPatch, parser: str, docling_mode: str, expected_calls: list[str],
) -> None:
    """Characterize public parser selection before extracting its dispatch helper."""
    import pymupdf

    from tldw_Server_API.app.core.Ingestion_Media_Processing.PDF import PDF_Processing_Lib as pdf

    path = tmp_path / "parser.pdf"
    with pymupdf.open() as document:
        document.new_page()
        document.save(path)
    calls = []

    def extract(name, supplied_path, *, page_chunks=False):
        assert supplied_path == str(path)
        calls.append(name)
        if name == "docling":
            if docling_mode == "fails":
                raise ValueError("parser unavailable")
            return "Docling extracted."
        assert page_chunks is True
        return [{"metadata": {"page": 1}, "text": "Native extracted."}]

    real_find_spec = pdf.importlib.util.find_spec
    monkeypatch.setattr(
        pdf.importlib.util, "find_spec",
        lambda name: (None if docling_mode == "missing" else object())
        if name == "docling.document_converter" else real_find_spec(name),
    )
    monkeypatch.setattr(pdf, "_is_usable_torch_module_for_docling", lambda: docling_mode != "no_torch")
    monkeypatch.setattr(pdf, "pymupdf4llm_parse_pdf", lambda *args, **kwargs: extract("pymupdf4llm", *args, **kwargs))
    monkeypatch.setattr(pdf, "extract_text_and_format_from_pdf", lambda *args, **kwargs: extract("pymupdf", *args, **kwargs))
    monkeypatch.setattr(pdf, "docling_parse_pdf", lambda *args: extract("docling", *args))

    result = pdf.process_pdf(path, path.name, parser=parser, perform_chunking=False)

    assert calls == expected_calls
    assert result["metadata"]["page_count"] == 1
    if not expected_calls:
        assert result["parser_used"] == "docling"
        assert any("Docling parser selected, but library is not installed" in warning for warning in result["warnings"])
    else:
        assert result["parser_used"] == expected_calls[-1]
        assert result["content"] == ("Docling extracted." if expected_calls[-1] == "docling" else "Native extracted.")
        assert bool(result["metadata"].get("page_map")) is (expected_calls[-1] != "docling")
        assert bool(result["warnings"]) is (parser == "unsupported")


@pytest.mark.asyncio
@pytest.mark.unit
async def test_pdf_analysis_empty_chunk_responses_do_not_crash(
    monkeypatch: MonkeyPatch,
) -> None:
    """Verify empty analyzer responses do not leave final_summary unbound."""
    from tldw_Server_API.app.core.Ingestion_Media_Processing.PDF import (
        PDF_Processing_Lib as pdf_mod,
    )

    monkeypatch.setattr(pdf_mod, "pymupdf4llm_parse_pdf", lambda path, **kwargs: "Some extracted content")
    monkeypatch.setattr(pdf_mod, "extract_metadata_from_pdf", lambda path, **kwargs: {})
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
