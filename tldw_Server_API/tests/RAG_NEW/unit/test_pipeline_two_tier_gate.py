import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tldw_Server_API.app.core.RAG.rag_service.unified_pipeline import unified_rag_pipeline
from tldw_Server_API.app.core.RAG.rag_service.types import Document, DataSource
from tldw_Server_API.app.core.RAG.rag_service.advanced_reranking import (
    TwoTierReranker,
    RerankingConfig,
    BaseReranker,
    ScoredDocument,
)


class _FakeCross(BaseReranker):
    def __init__(self, config, scores_map):
        super().__init__(config)
        self._scores_map = scores_map
    async def rerank(self, query, documents, original_scores=None):
        out = []
        for d in documents:
            sc = float(self._scores_map.get(getattr(d, 'id', None), 0.0))
            out.append(ScoredDocument(document=d, original_score=getattr(d, 'score', 0.0), rerank_score=sc))
        out.sort(key=lambda x: x.rerank_score, reverse=True)
        return out[: self.config.top_k]


class _FakeLLM(BaseReranker):
    def __init__(self, config, scores_map):
        super().__init__(config)
        self._scores_map = scores_map
    async def rerank(self, query, documents, original_scores=None):
        out = []
        for d in documents:
            sc = float(self._scores_map.get(getattr(d, 'id', None), 0.0))
            out.append(ScoredDocument(document=d, original_score=getattr(d, 'score', 0.0), rerank_score=sc))
        out.sort(key=lambda x: x.rerank_score, reverse=True)
        return out[: self.config.top_k]


@pytest.mark.asyncio
async def test_unified_pipeline_two_tier_gates_generation(monkeypatch):
    # Tight thresholds to force generation gating
    monkeypatch.setenv("RAG_MIN_RELEVANCE_PROB", "0.99")
    monkeypatch.setenv("RAG_SENTINEL_MARGIN", "0.50")

    # Patch retrieval to return a small set of docs
    with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever') as mock_retriever:
        mock_retriever_instance = MagicMock()
        mock_retriever_instance.retrieve = AsyncMock(return_value=[
            Document(id="d1", content="alpha", metadata={}, source=DataSource.MEDIA_DB, score=0.2),
            Document(id="d2", content="beta", metadata={}, source=DataSource.MEDIA_DB, score=0.1),
        ])
        mock_retriever.return_value = mock_retriever_instance

        # Patch create_reranker to return a deterministic TwoTier with fake CE/LLM
        from tldw_Server_API.app.core.RAG.rag_service import advanced_reranking as ar

        def _fake_create(strategy, cfg, llm_client=None):

            if getattr(ar.RerankingStrategy, 'TWO_TIER') and strategy == ar.RerankingStrategy.TWO_TIER:
                ce_map = {"d1": 0.20, "d2": 0.10, "sentinel:irrelevant": 0.02}
                llm_map = {"d1": 0.40, "d2": 0.30, "sentinel:irrelevant": 0.05}
                return ar.TwoTierReranker(cfg, cross_reranker=_FakeCross(cfg, ce_map), llm_reranker=_FakeLLM(cfg, llm_map))
            return ar.create_reranker(strategy, cfg, llm_client=llm_client)

        with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.create_reranker', side_effect=_fake_create):
            # Do not patch AnswerGenerator; generation should be gated and not invoked
            res = await unified_rag_pipeline(
                query="What is RAG?",
                enable_reranking=True,
                reranking_strategy="two_tier",
                enable_generation=True,
                top_k=2,
            )

            # Check calibration metadata present and gating applied
            cal = res.metadata.get("reranking_calibration")
            assert isinstance(cal, dict)
            assert cal.get("strategy") == "two_tier"
            assert bool(cal.get("gated")) is True
            # Generation metadata should indicate gate
            gate = res.metadata.get("generation_gate")
            assert isinstance(gate, dict)
            assert gate.get("reason") == "low_relevance_probability"
            plan = res.metadata.get("retrieval_plan")
            assert isinstance(plan, dict)
            assert plan.get("query") == "What is RAG?"
            assert plan.get("top_k") == 2


@pytest.mark.asyncio
async def test_unified_pipeline_two_tier_request_overrides_gate(monkeypatch):
    # Do not set env thresholds; pass strict thresholds via request-level overrides
    with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever') as mock_retriever:
        mock_retriever_instance = MagicMock()
        mock_retriever_instance.retrieve = AsyncMock(return_value=[
            Document(id="d1", content="alpha", metadata={}, source=DataSource.MEDIA_DB, score=0.2),
            Document(id="d2", content="beta", metadata={}, source=DataSource.MEDIA_DB, score=0.1),
        ])
        mock_retriever.return_value = mock_retriever_instance

        from tldw_Server_API.app.core.RAG.rag_service import advanced_reranking as ar

        def _fake_create(strategy, cfg, llm_client=None):

            if getattr(ar.RerankingStrategy, 'TWO_TIER') and strategy == ar.RerankingStrategy.TWO_TIER:
                ce_map = {"d1": 0.20, "d2": 0.10, "sentinel:irrelevant": 0.02}
                llm_map = {"d1": 0.40, "d2": 0.30, "sentinel:irrelevant": 0.05}
                return ar.TwoTierReranker(cfg, cross_reranker=_FakeCross(cfg, ce_map), llm_reranker=_FakeLLM(cfg, llm_map))
            return ar.create_reranker(strategy, cfg, llm_client=llm_client)

        with patch('tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.create_reranker', side_effect=_fake_create):
            res = await unified_rag_pipeline(
                query="What is RAG?",
                enable_reranking=True,
                reranking_strategy="two_tier",
                enable_generation=True,
                top_k=2,
                # Request-level overrides (very strict to force gating)
                rerank_min_relevance_prob=0.99,
                rerank_sentinel_margin=0.50,
            )

            cal = res.metadata.get("reranking_calibration")
            assert isinstance(cal, dict)
            assert cal.get("strategy") == "two_tier"
            assert bool(cal.get("gated")) is True
            gate = res.metadata.get("generation_gate")
            assert isinstance(gate, dict)
            assert gate.get("reason") == "low_relevance_probability"
            plan = res.metadata.get("retrieval_plan")
            assert isinstance(plan, dict)
            assert plan.get("query") == "What is RAG?"
            assert plan.get("top_k") == 2


        def _fake_create(strategy, cfg, llm_client=None):


            if strategy == ar.RerankingStrategy.TWO_TIER:
                ce_map = {"d1": 0.20, "d2": 0.10, "sentinel:irrelevant": 0.02}
                llm_map = {"d1": 0.40, "d2": 0.30, "sentinel:irrelevant": 0.05}
                return ar.TwoTierReranker(cfg, cross_reranker=_FakeCross(cfg, ce_map), llm_reranker=_FakeLLM(cfg, llm_map))
            return ar.create_reranker(strategy, cfg, llm_client=llm_client)
