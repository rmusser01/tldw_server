from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest


def _build_minimal_pdf_bytes() -> bytes:
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page(width=300, height=200)
    page.insert_text((72, 72), "Hello from parser")
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


@dataclass
class _StubHunyuanBackend:
    backend_concurrency_cap: Any

    name = "hunyuan"

    @classmethod
    def available(cls) -> bool:
        return True

    def describe(self) -> dict[str, Any]:
        return {
            "mode": "remote",
            "runtime_family": "llamacpp",
            "configured_family": "auto",
            "configured": True,
            "backend_concurrency_cap": self.backend_concurrency_cap,
            "native": {
                "mode": "transformers",
                "configured": False,
                "available": False,
                "model": "tencent/HunyuanOCR",
                "device": "cuda",
            },
            "llamacpp": {
                "mode": "remote",
                "configured_mode": "auto",
                "configured": True,
                "backend_concurrency_cap": self.backend_concurrency_cap,
                "argv": ["llama-server", "--model", "/models/private.gguf"],
                "host": "127.0.0.1",
                "port": 19092,
                "url": "http://127.0.0.1:19092/v1/chat/completions",
                "model": "ggml-org/HunyuanOCR-GGUF:Q8_0",
            },
        }


@pytest.mark.unit
@pytest.mark.asyncio
async def test_process_pdf_task_attaches_hunyuan_runtime_family_and_sanitizes_nested_details(
    monkeypatch,
):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.PDF import PDF_Processing_Lib as pdf_lib

    stub_backend = _StubHunyuanBackend(backend_concurrency_cap=2)

    monkeypatch.setattr(pdf_lib, "_get_ocr_backend", lambda name=None: stub_backend)
    monkeypatch.setattr(
        pdf_lib,
        "_ocr_pdf_pages",
        lambda **kwargs: (
            "OCR PAGE TEXT",
            1,
            1,
            [
                {
                    "text": "OCR PAGE TEXT",
                    "raw": {
                        "backend": "hunyuan",
                        "runtime_family": "llamacpp",
                        "page": 1,
                    },
                }
            ],
        ),
    )
    monkeypatch.setattr(pdf_lib, "pymupdf4llm_parse_pdf", lambda path: "parser text")
    monkeypatch.setenv("OCR_PAGE_CONCURRENCY", "4")

    result = await pdf_lib.process_pdf_task(
        file_bytes=_build_minimal_pdf_bytes(),
        filename="hunyuan-llamacpp.pdf",
        parser="pymupdf4llm",
        perform_chunking=False,
        perform_analysis=False,
        enable_ocr=True,
        ocr_backend="hunyuan",
        ocr_mode="always",
        ocr_output_format="json",
        ocr_prompt_preset="json",
    )

    ocr_details = result["analysis_details"]["ocr"]
    assert ocr_details["backend"] == "hunyuan"  # nosec B101
    assert ocr_details["runtime_family"] == "llamacpp"  # nosec B101
    assert ocr_details["configured_family"] == "auto"  # nosec B101
    assert ocr_details["backend_concurrency_cap"] == 2  # nosec B101
    assert ocr_details["structured"]["pages"][0]["raw"]["runtime_family"] == "llamacpp"  # nosec B101
    assert ocr_details["native"]["mode"] == "transformers"  # nosec B101
    assert "model" not in ocr_details["native"]  # nosec B101
    assert ocr_details["llamacpp"]["mode"] == "remote"  # nosec B101
    assert ocr_details["llamacpp"]["configured_mode"] == "auto"  # nosec B101
    assert ocr_details["llamacpp"]["backend_concurrency_cap"] == 2  # nosec B101
    assert "argv" not in ocr_details["llamacpp"]  # nosec B101
    assert "host" not in ocr_details["llamacpp"]  # nosec B101
    assert "port" not in ocr_details["llamacpp"]  # nosec B101
    assert "url" not in ocr_details["llamacpp"]  # nosec B101
    assert "model" not in ocr_details["llamacpp"]  # nosec B101
