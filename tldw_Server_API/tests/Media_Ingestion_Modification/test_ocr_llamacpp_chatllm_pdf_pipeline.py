from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from tldw_Server_API.app.core.Ingestion_Media_Processing.OCR.types import OCRResult


def _build_minimal_pdf_bytes() -> bytes:
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page(width=300, height=200)
    page.insert_text((72, 72), "Hello from parser")
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


@dataclass
class _StubLLMOCROBackend:
    name: str
    backend_concurrency_cap: Any

    @classmethod
    def available(cls) -> bool:
        return True

    def describe(self) -> dict[str, Any]:
        return {
            "mode": "remote",
            "configured": True,
            "runtime": self.name,
            "backend_concurrency_cap": self.backend_concurrency_cap,
        }


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("backend_name", ["llamacpp", "chatllm"])
async def test_process_pdf_task_attaches_structured_ocr_and_preserves_backend_metadata(
    monkeypatch, backend_name: str
):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.PDF import PDF_Processing_Lib as pdf_lib

    stub_backend = _StubLLMOCROBackend(name=backend_name, backend_concurrency_cap=2)
    captured: dict[str, Any] = {}

    def _fake_get_backend(name=None):
        assert name == backend_name  # nosec B101
        return stub_backend

    def _fake_ocr_pdf_pages(**kwargs):
        captured.update(kwargs)
        return (
            "OCR PAGE TEXT",
            2,
            2,
            [
                {
                    "text": "OCR PAGE TEXT",
                    "raw": {"backend": backend_name, "page": 1},
                },
                {
                    "text": "OCR PAGE TEXT",
                    "raw": {"backend": backend_name, "page": 2},
                },
            ],
        )

    monkeypatch.setattr(pdf_lib, "_get_ocr_backend", _fake_get_backend)
    monkeypatch.setattr(pdf_lib, "_ocr_pdf_pages", _fake_ocr_pdf_pages)
    monkeypatch.setattr(pdf_lib, "pymupdf4llm_parse_pdf", lambda path: "parser text")
    monkeypatch.setenv("OCR_PAGE_CONCURRENCY", "8")

    result = await pdf_lib.process_pdf_task(
        file_bytes=_build_minimal_pdf_bytes(),
        filename=f"{backend_name}.pdf",
        parser="pymupdf4llm",
        perform_chunking=False,
        perform_analysis=False,
        enable_ocr=True,
        ocr_backend=backend_name,
        ocr_mode="always",
        ocr_output_format="json",
        ocr_prompt_preset="json",
    )

    assert captured["concurrency"] == 2  # nosec B101
    assert result["content"] == "OCR PAGE TEXT"  # nosec B101

    ocr_details = result["analysis_details"]["ocr"]
    assert ocr_details["backend"] == backend_name  # nosec B101
    assert ocr_details["runtime"] == backend_name  # nosec B101
    assert ocr_details["page_concurrency"] == 2  # nosec B101
    assert ocr_details["backend_concurrency_cap"] == 2  # nosec B101
    assert ocr_details["structured"]["pages"][0]["raw"] == {"backend": backend_name, "page": 1}  # nosec B101


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("backend_name", ["llamacpp", "chatllm"])
async def test_process_pdf_task_preserves_parser_text_when_ocr_not_triggered(
    monkeypatch, backend_name: str
):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.PDF import PDF_Processing_Lib as pdf_lib

    stub_backend = _StubLLMOCROBackend(name=backend_name, backend_concurrency_cap=1)
    ocr_calls = {"count": 0}

    def _fake_get_backend(name=None):
        assert name == backend_name  # nosec B101
        return stub_backend

    def _fake_ocr_pdf_pages(**kwargs):
        ocr_calls["count"] += 1
        return ("unexpected", 1, 1, [OCRResult(text="unexpected", raw={"backend": backend_name}).as_dict()])

    monkeypatch.setattr(pdf_lib, "_get_ocr_backend", _fake_get_backend)
    monkeypatch.setattr(pdf_lib, "_ocr_pdf_pages", _fake_ocr_pdf_pages)
    monkeypatch.setattr(pdf_lib, "pymupdf4llm_parse_pdf", lambda path: "parser text is already long enough")

    result = await pdf_lib.process_pdf_task(
        file_bytes=_build_minimal_pdf_bytes(),
        filename=f"{backend_name}.pdf",
        parser="pymupdf4llm",
        perform_chunking=False,
        perform_analysis=False,
        enable_ocr=True,
        ocr_backend=backend_name,
        ocr_mode="fallback",
        ocr_min_page_text_chars=5,
    )

    assert ocr_calls["count"] == 0  # nosec B101
    assert result["content"] == "parser text is already long enough"  # nosec B101


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("backend_name", ["llamacpp", "chatllm"])
async def test_process_pdf_task_ignores_invalid_backend_concurrency_cap(
    monkeypatch, backend_name: str
):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.PDF import PDF_Processing_Lib as pdf_lib

    stub_backend = _StubLLMOCROBackend(name=backend_name, backend_concurrency_cap="invalid")
    captured: dict[str, Any] = {}

    def _fake_get_backend(name=None):
        assert name == backend_name  # nosec B101
        return stub_backend

    def _fake_ocr_pdf_pages(**kwargs):
        captured.update(kwargs)
        return (
            "OCR PAGE TEXT",
            1,
            1,
            [{"text": "OCR PAGE TEXT", "raw": {"backend": backend_name, "page": 1}}],
        )

    monkeypatch.setattr(pdf_lib, "_get_ocr_backend", _fake_get_backend)
    monkeypatch.setattr(pdf_lib, "_ocr_pdf_pages", _fake_ocr_pdf_pages)
    monkeypatch.setattr(pdf_lib, "pymupdf4llm_parse_pdf", lambda path: "")
    monkeypatch.setenv("OCR_PAGE_CONCURRENCY", "4")

    result = await pdf_lib.process_pdf_task(
        file_bytes=_build_minimal_pdf_bytes(),
        filename=f"{backend_name}.pdf",
        parser="pymupdf4llm",
        perform_chunking=False,
        perform_analysis=False,
        enable_ocr=True,
        ocr_backend=backend_name,
        ocr_mode="always",
    )

    assert captured["concurrency"] == 4  # nosec B101
    assert result["analysis_details"]["ocr"]["page_concurrency"] == 4  # nosec B101
    assert "backend_concurrency_cap" not in result["analysis_details"]["ocr"]  # nosec B101


@pytest.mark.unit
@pytest.mark.asyncio
async def test_process_pdf_task_filters_internal_backend_details_from_output(monkeypatch):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.PDF import PDF_Processing_Lib as pdf_lib

    class _SensitiveBackend(_StubLLMOCROBackend):
        def describe(self) -> dict[str, Any]:
            return {
                "mode": "managed",
                "configured": True,
                "runtime": self.name,
                "backend_concurrency_cap": self.backend_concurrency_cap,
                "argv": ["llama-server", "--model", "/models/private.gguf"],
                "host": "127.0.0.1",
                "port": 19090,
                "url": "http://127.0.0.1:19090/v1/chat/completions",
                "prompt": "internal prompt",
                "model": "/models/private.gguf",
            }

    stub_backend = _SensitiveBackend(name="llamacpp", backend_concurrency_cap=2)

    monkeypatch.setattr(pdf_lib, "_get_ocr_backend", lambda name=None: stub_backend)
    monkeypatch.setattr(
        pdf_lib,
        "_ocr_pdf_pages",
        lambda **kwargs: ("OCR PAGE TEXT", 1, 1, [{"text": "OCR PAGE TEXT", "raw": {"page": 1}}]),
    )
    monkeypatch.setattr(pdf_lib, "pymupdf4llm_parse_pdf", lambda path: "parser text")
    monkeypatch.setenv("OCR_PAGE_CONCURRENCY", "4")

    result = await pdf_lib.process_pdf_task(
        file_bytes=_build_minimal_pdf_bytes(),
        filename="llamacpp-sensitive.pdf",
        parser="pymupdf4llm",
        perform_chunking=False,
        perform_analysis=False,
        enable_ocr=True,
        ocr_backend="llamacpp",
        ocr_mode="always",
    )

    ocr_details = result["analysis_details"]["ocr"]
    assert ocr_details["runtime"] == "llamacpp"  # nosec B101
    assert ocr_details["backend_concurrency_cap"] == 2  # nosec B101
    assert "argv" not in ocr_details  # nosec B101
    assert "host" not in ocr_details  # nosec B101
    assert "port" not in ocr_details  # nosec B101
    assert "url" not in ocr_details  # nosec B101
    assert "prompt" not in ocr_details  # nosec B101
    assert "model" not in ocr_details  # nosec B101


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("backend_name", ["llamacpp", "chatllm"])
async def test_process_pdf_task_appends_ocr_text_when_append_branch_is_taken(
    monkeypatch, backend_name: str
):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.PDF import PDF_Processing_Lib as pdf_lib

    stub_backend = _StubLLMOCROBackend(name=backend_name, backend_concurrency_cap=2)
    ocr_calls = {"count": 0}
    def _fake_get_backend(name=None):
        assert name == backend_name  # nosec B101
        return stub_backend

    def _fake_ocr_pdf_pages(**kwargs):
        ocr_calls["count"] += 1
        return (
            "OCR PAGE TEXT",
            1,
            1,
            [{"text": "OCR PAGE TEXT", "raw": {"backend": backend_name, "page": 1}}],
        )

    monkeypatch.setattr(pdf_lib, "_get_ocr_backend", _fake_get_backend)
    monkeypatch.setattr(pdf_lib, "_ocr_pdf_pages", _fake_ocr_pdf_pages)
    monkeypatch.setattr(pdf_lib, "pymupdf4llm_parse_pdf", lambda path: "parser text")
    monkeypatch.setattr(
        pdf_lib,
        "_should_replace_ocr_content",
        lambda content_text_len, ocr_mode, ocr_min_page_text_chars: False,
        raising=False,
    )

    result = await pdf_lib.process_pdf_task(
        file_bytes=_build_minimal_pdf_bytes(),
        filename=f"{backend_name}.pdf",
        parser="pymupdf4llm",
        perform_chunking=False,
        perform_analysis=False,
        enable_ocr=True,
        ocr_backend=backend_name,
        ocr_mode="fallback",
        ocr_min_page_text_chars=20,
    )

    assert ocr_calls["count"] == 1  # nosec B101
    assert result["content"] == "parser text\n\nOCR PAGE TEXT"  # nosec B101
    assert result["parser_used"].endswith("+ocr-appended") is True  # nosec B101
