from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import tldw_Server_API.app.core.RAG.rag_service.unified_pipeline as up
from tldw_Server_API.app.api.v1.schemas.rag_schemas_unified import UnifiedRAGResponse
from tldw_Server_API.app.core.RAG.rag_service.types import DataSource, Document


pytestmark = pytest.mark.unit


class _CaptureGenerator:
    last_generate_kwargs = None

    def __init__(
        self,
        model=None,
        provider=None,
        system_prompt=None,
        credential_runtime=None,
    ):
        self.model = model
        self.provider = provider
        self.system_prompt = system_prompt
        self.credential_runtime = credential_runtime

    async def generate(self, **kwargs):
        _CaptureGenerator.last_generate_kwargs = kwargs
        return {"answer": "structured answer"}


def _doc_fixture() -> list[Document]:
    return [
        Document(
            id="doc1",
            content='Core finding with <xml-like> chars & evidence.',
            metadata={"title": 'Doc "One"', "url": "https://example.com/doc1"},
            source=DataSource.WEB_CONTENT,
            score=0.9,
        )
    ]


@pytest.mark.asyncio
async def test_structured_writer_quality_low_token_budget_marks_degraded_policy(monkeypatch):
    _CaptureGenerator.last_generate_kwargs = None
    monkeypatch.setattr(up, "AnswerGenerator", _CaptureGenerator)
    # Keep this unit test offline/deterministic: disable quality-mode HyDE helpers.
    monkeypatch.setattr(up, "generate_hypothetical_answer", None)
    monkeypatch.setattr(up, "hyde_embed_text", None)
    # Keep this unit test offline/deterministic: disable quality-mode multi-vector paths.
    monkeypatch.setattr(up, "apply_multi_vector_passages", None)
    monkeypatch.setattr(up, "MultiVectorConfig", None)
    monkeypatch.setattr(up, "apply_precomputed_spans", None)
    monkeypatch.setattr(up, "PrecomputedSpanConfig", None)

    with (
        patch("tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever") as mock_retriever,
        patch("tldw_Server_API.app.core.RAG.rag_service.database_retrievers.MultiDatabaseRetriever") as mock_db_retriever,
    ):
        retriever_instance = MagicMock()
        retriever_instance.retrieve = AsyncMock(return_value=_doc_fixture())
        mock_retriever.return_value = retriever_instance
        mock_db_retriever.return_value = retriever_instance

        result = await up.unified_rag_pipeline(
            query="Provide a deep research report",
            search_mode="fts",
            top_k=1,
            enable_cache=False,
            adaptive_cache=False,
            enable_reranking=False,
            enable_generation=True,
            enable_structured_response=True,
            search_depth_mode="quality",
            max_generation_tokens=800,
        )

    assert isinstance(result, UnifiedRAGResponse)
    assert isinstance(_CaptureGenerator.last_generate_kwargs, dict)
    prompt_template = _CaptureGenerator.last_generate_kwargs["prompt_template"]
    assert "strict 2000+ word minimum is likely not feasible" in prompt_template

    writer_meta = result.metadata.get("structured_writer", {})
    assert writer_meta.get("enabled") is True
    assert writer_meta.get("mode") == "quality"
    assert writer_meta.get("max_generation_tokens") == 800
    assert writer_meta.get("depth_policy", {}).get("degraded_due_to_token_budget") is True


@pytest.mark.asyncio
async def test_structured_writer_quality_high_token_budget_keeps_full_depth_policy(monkeypatch):
    _CaptureGenerator.last_generate_kwargs = None
    monkeypatch.setattr(up, "AnswerGenerator", _CaptureGenerator)
    # Keep this unit test offline/deterministic: disable quality-mode HyDE helpers.
    monkeypatch.setattr(up, "generate_hypothetical_answer", None)
    monkeypatch.setattr(up, "hyde_embed_text", None)
    # Keep this unit test offline/deterministic: disable quality-mode multi-vector paths.
    monkeypatch.setattr(up, "apply_multi_vector_passages", None)
    monkeypatch.setattr(up, "MultiVectorConfig", None)
    monkeypatch.setattr(up, "apply_precomputed_spans", None)
    monkeypatch.setattr(up, "PrecomputedSpanConfig", None)

    with (
        patch("tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever") as mock_retriever,
        patch("tldw_Server_API.app.core.RAG.rag_service.database_retrievers.MultiDatabaseRetriever") as mock_db_retriever,
    ):
        retriever_instance = MagicMock()
        retriever_instance.retrieve = AsyncMock(return_value=_doc_fixture())
        mock_retriever.return_value = retriever_instance
        mock_db_retriever.return_value = retriever_instance

        result = await up.unified_rag_pipeline(
            query="Provide a deep research report",
            search_mode="fts",
            top_k=1,
            enable_cache=False,
            adaptive_cache=False,
            enable_reranking=False,
            enable_generation=True,
            enable_structured_response=True,
            search_depth_mode="quality",
            max_generation_tokens=5000,
        )

    assert isinstance(result, UnifiedRAGResponse)
    assert isinstance(_CaptureGenerator.last_generate_kwargs, dict)
    prompt_template = _CaptureGenerator.last_generate_kwargs["prompt_template"]
    assert "supports the 2000+ word target" in prompt_template

    writer_meta = result.metadata.get("structured_writer", {})
    assert writer_meta.get("enabled") is True
    assert writer_meta.get("mode") == "quality"
    assert writer_meta.get("max_generation_tokens") == 5000
    assert writer_meta.get("depth_policy", {}).get("degraded_due_to_token_budget") is False
