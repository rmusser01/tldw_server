import pytest

from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.types import OCRResult


class _StubOCRBackend:
    name = "stub_ocr"

    @classmethod
    def available(cls) -> bool:
        return True

    def ocr_image(self, image_bytes: bytes, lang=None) -> str:
        return "PLAIN TEXT"

    def ocr_image_structured(self, image_bytes: bytes, lang=None, output_format=None, prompt_preset=None):
        return OCRResult(
            text="PLAIN TEXT",
            format="text",
            raw={"labels": ["text"], "bboxes": [[1, 2, 3, 4]]},
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_pdf_ocr_attaches_structured_pages(monkeypatch):
    try:
        import pymupdf
        from tldw_Server_API.app.core.Ingestion_Media_Processing.PDF import PDF_Processing_Lib as pdf_lib
    except Exception as e:
        pytest.skip(f"Dependencies not available: {e}")

    monkeypatch.setattr(pdf_lib, "_get_ocr_backend", lambda name=None: _StubOCRBackend())
    monkeypatch.setattr(pdf_lib, "pymupdf4llm_parse_pdf", lambda path, **kwargs: "")

    doc = pymupdf.open()
    page = doc.new_page(width=300, height=200)
    page.insert_text((72, 72), "Hello")
    pdf_bytes = doc.tobytes()
    doc.close()

    res = await pdf_lib.process_pdf_task(
        file_bytes=pdf_bytes,
        filename="test.pdf",
        parser="pymupdf4llm",
        perform_chunking=False,
        perform_analysis=False,
        enable_ocr=True,
        ocr_backend="stub_ocr",
        ocr_mode="always",
        ocr_dpi=72,
        ocr_lang="eng",
    )

    ocr_details = res.get("analysis_details", {}).get("ocr", {})
    structured = ocr_details.get("structured")
    assert isinstance(structured, dict)
    pages = structured.get("pages")
    assert isinstance(pages, list)
    assert pages
    assert pages[0].get("raw") == {"labels": ["text"], "bboxes": [[1, 2, 3, 4]]}
