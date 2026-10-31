import asyncio
import threading

import pytest

from tldw_Server_API.app.core.RAG.rag_service.advanced_reranking import (
    BaseReranker,
    LLMReranker,
    RerankingConfig,
    ScoredDocument,
    TwoTierReranker,
    _has_reranker_score,
)
from tldw_Server_API.app.core.RAG.rag_service.types import DataSource, Document


class _FakeCross(BaseReranker):
    def __init__(self, config: RerankingConfig, scores_map):
        super().__init__(config)
        self._scores_map = scores_map

    async def rerank(self, query, documents, original_scores=None):
        out = []
        for d in documents:
            did = getattr(d, 'id', None)
            sc = float(self._scores_map.get(did, 0.0))
            out.append(ScoredDocument(
                document=d,
                original_score=getattr(d, 'score', 0.0),
                rerank_score=sc,
                relevance_score=sc,
                explanation="fake_ce",
            ))
        out.sort(key=lambda x: x.rerank_score, reverse=True)
        return out[: self.config.top_k]


class _FakeLLM(BaseReranker):
    def __init__(self, config: RerankingConfig, scores_map):
        super().__init__(config)
        self._scores_map = scores_map

    async def rerank(self, query, documents, original_scores=None):
        out = []
        for d in documents:
            did = getattr(d, 'id', None)
            sc = float(self._scores_map.get(did, 0.0))
            out.append(ScoredDocument(
                document=d,
                original_score=getattr(d, 'score', 0.0),
                rerank_score=sc,
                relevance_score=sc,
                explanation="fake_llm",
            ))
        out.sort(key=lambda x: x.rerank_score, reverse=True)
        return out[: self.config.top_k]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_two_tier_missing_llm_client_preserves_original_scores_without_gating():
    config = RerankingConfig(top_k=2)
    documents = [
        Document(
            id="d1",
            content="first",
            metadata={},
            source=DataSource.MEDIA_DB,
            score=0.1,
        ),
        Document(
            id="d2",
            content="second",
            metadata={},
            source=DataSource.MEDIA_DB,
            score=0.8,
        ),
    ]
    reranker = TwoTierReranker(
        config,
        cross_reranker=_FakeCross(
            config,
            {"d1": 0.2, "d2": 0.9, "sentinel:irrelevant": 0.0},
        ),
        llm_reranker=LLMReranker(config, llm_client=None),
    )

    scored = await reranker.rerank("query", documents)

    assert [item.document.id for item in scored] == ["d1", "d2"]
    assert [item.original_score for item in scored] == [0.1, 0.8]
    assert [item.rerank_score for item in scored] == [0.1, 0.8]
    assert [item.relevance_score for item in scored] == [0.1, 0.8]
    assert reranker.last_metadata == {
        "strategy": "two_tier",
        "degraded": True,
        "failure_code": "provider_unavailable",
        "verification_available": False,
    }
    assert {
        "top_doc_prob",
        "sentinel_scores",
        "fused_score",
        "calibrated_prob",
        "gated",
    }.isdisjoint(reranker.last_metadata)


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider_response",
    [
        "",
        "not a numeric score",
        "Error: upstream failed",
        "HTTP 500",
        "Error 500",
        "score: 0.75",
        "0.75 confidence",
        "-0.1",
        "1.1",
    ],
)
async def test_two_tier_malformed_runtime_score_preserves_originals_without_gating(
    provider_response: str,
):
    class _RuntimeClient:
        credentials_resolved = True
        used = False

        def analyze(self, _prompt: str) -> str:
            return provider_response

    config = RerankingConfig(top_k=2)
    documents = [
        Document(id="d1", content="first", metadata={}, source=DataSource.MEDIA_DB, score=0.1),
        Document(id="d2", content="second", metadata={}, source=DataSource.MEDIA_DB, score=0.8),
    ]
    client = _RuntimeClient()
    reranker = TwoTierReranker(
        config,
        cross_reranker=_FakeCross(
            config,
            {"d1": 0.2, "d2": 0.9, "sentinel:irrelevant": 0.0},
        ),
        llm_reranker=LLMReranker(config, llm_client=client),
    )

    scored = await reranker.rerank("query", documents)

    assert [item.document.id for item in scored] == ["d1", "d2"]
    assert [item.rerank_score for item in scored] == [0.1, 0.8]
    assert client.used is False
    assert reranker.last_metadata == {
        "strategy": "two_tier",
        "degraded": True,
        "failure_code": "provider_unavailable",
        "verification_available": False,
    }
    assert {"top_doc_prob", "sentinel_scores", "gated"}.isdisjoint(reranker.last_metadata)


@pytest.mark.unit
@pytest.mark.parametrize("provider_response", ["0", "1", "0.75"])
def test_runtime_reranker_score_gate_accepts_only_plain_bounded_numbers(
    provider_response: str,
) -> None:
    assert _has_reranker_score(provider_response)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_concurrent_runtime_rerankers_do_not_share_degraded_state():
    barrier = threading.Barrier(2)

    class _RuntimeClient:
        credentials_resolved = True

        def __init__(self, response: str) -> None:
            self.response = response
            self.used = False

        def analyze(self, _prompt: str) -> str:
            barrier.wait(timeout=2.0)
            return self.response

    config = RerankingConfig(top_k=1)

    async def run(response: str) -> tuple[list[ScoredDocument], dict[str, object], bool]:
        document = Document(
            id=response or "empty",
            content="passage",
            metadata={},
            source=DataSource.MEDIA_DB,
            score=0.4,
        )
        client = _RuntimeClient(response)
        reranker = LLMReranker(config, llm_client=client)
        scored = await reranker.rerank("query", [document])
        return scored, dict(reranker.last_metadata), client.used

    valid, malformed = await asyncio.gather(run("0.9"), run("malformed"))

    assert valid[0][0].rerank_score == 0.9
    assert valid[1] == {}
    assert valid[2] is True
    assert malformed[0][0].rerank_score == 0.4
    assert malformed[1] == {
        "degraded": True,
        "failure_code": "provider_unavailable",
        "verification_available": False,
    }
    assert malformed[2] is False


@pytest.mark.unit
def test_two_tier_reranker_calibration_and_gating(monkeypatch):
     # Force strict gating to validate the path
    monkeypatch.setenv("RAG_MIN_RELEVANCE_PROB", "0.95")
    monkeypatch.setenv("RAG_SENTINEL_MARGIN", "0.50")

    # Make three simple docs
    d1 = Document(id="d1", content="alpha", metadata={}, source=DataSource.MEDIA_DB, score=0.2)
    d2 = Document(id="d2", content="beta", metadata={}, source=DataSource.MEDIA_DB, score=0.1)
    d3 = Document(id="d3", content="gamma", metadata={}, source=DataSource.MEDIA_DB, score=0.05)
    docs = [d1, d2, d3]

    # Cross-encoder and LLM scores (sentinel appears later; filled by reranker)
    ce_map = {"d1": 0.20, "d2": 0.10, "d3": 0.05, "sentinel:irrelevant": 0.02}
    llm_map = {"d1": 0.40, "d2": 0.30, "d3": 0.10, "sentinel:irrelevant": 0.05}

    cfg = RerankingConfig(top_k=2)
    two = TwoTierReranker(cfg, cross_reranker=_FakeCross(cfg, ce_map), llm_reranker=_FakeLLM(cfg, llm_map))

    # Python 3.12+ uses no default loop in sync context; prefer asyncio.run
    scored = asyncio.run(two.rerank("q?", docs))

    # Returned docs should not include sentinel and should be <= top_k
    ids = [sd.document.id for sd in scored]
    assert "sentinel:irrelevant" not in ids
    assert len(ids) == 2

    # Calibrated probability is attached as rerank_score and criteria_scores
    assert all("calibrated_prob" in (sd.criteria_scores or {}) for sd in scored)
    assert all(isinstance(sd.rerank_score, float) for sd in scored)

    # Metadata exposes gating flag due to strict thresholds set above
    assert isinstance(two.last_metadata, dict)
    assert two.last_metadata.get("strategy") == "two_tier"
    assert bool(two.last_metadata.get("gated")) is True
