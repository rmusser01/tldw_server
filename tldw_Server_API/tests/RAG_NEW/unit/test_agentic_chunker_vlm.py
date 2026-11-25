import os
import tempfile
import types

import pytest

from tldw_Server_API.app.core.RAG.rag_service.types import Document, DataSource
from tldw_Server_API.app.core.RAG.rag_service.agentic_chunker import agentic_rag_pipeline, AgenticConfig


@pytest.mark.asyncio
async def test_agentic_vlm_late_chunking_smoke(monkeypatch):
    # Create a temporary PDF path
    fd, pdf_path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)
    try:
        # Fake retriever returns a doc with a local PDF url
        class FakeRetriever:
            def __init__(self, *a, **k):
                pass
            async def retrieve(self, *a, **k):
                return [
                    Document(
                        id="pdf1",
                        content="",  # no content to encourage VLM-driven spans
                        metadata={"title": "PDF Doc", "url": pdf_path, "source": "media_db"},
                        source=DataSource.MEDIA_DB,
                        score=0.9,
                    )
                ]

        # Stub VLM backend registry to return a backend with process_pdf
        class StubRes:
            def __init__(self):
                self.extra = {
                    "by_page": [
                        {
                            "page": 1,
                            "detections": [
                                {"label": "table", "score": 0.95, "bbox": [10, 10, 100, 100]}
                            ],
                        }
                    ]
                }

        class StubBackend:
            def process_pdf(self, path, max_pages=None):  # noqa: ARG002
                return StubRes()

        import tldw_Server_API.app.core.RAG.rag_service.agentic_chunker as ac
        monkeypatch.setattr(ac, "MultiDatabaseRetriever", FakeRetriever)

        # Patch VLM registry get_backend
        vlm_mod_path = "tldw_Server_API.app.core.Ingestion_Media_Processing.VLM.registry"
        def fake_get_backend(name=None):  # noqa: ARG001
            return StubBackend()
        # Monkeypatch the import indirection: create a fake module with get_backend
        import sys
        # Use a real module object to avoid inserting unhashable objects into sys.modules
        m = types.ModuleType("vlm_registry_stub")
        setattr(m, "get_backend", fake_get_backend)
        monkeypatch.setitem(sys.modules, vlm_mod_path, m)

        res = await agentic_rag_pipeline(
            query="table results",
            sources=["media_db"],
            agentic=AgenticConfig(
                top_k_docs=5,
                enable_tools=True,
                agentic_enable_vlm_late_chunking=True,
                agentic_vlm_backend="hf_table_transformer",
                agentic_vlm_detect_tables_only=True,
                agentic_vlm_late_chunk_top_k_docs=2,
                time_budget_sec=2.0,
            ),
            enable_generation=False,
            enable_citations=False,
        )
        doc = res.documents[0]
        text = doc["content"] if isinstance(doc, dict) else doc.content
        assert "Detected table" in text or "table" in text.lower()
    finally:
        try:
            os.remove(pdf_path)
        except Exception:
            _ = None
