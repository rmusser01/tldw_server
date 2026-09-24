"""Reranker models are loaded once per process per construction key, not per request."""

import sys
import types

import pytest

from tldw_Server_API.app.core.RAG.rag_service import advanced_reranking as ar

pytestmark = pytest.mark.unit


def test_flashrank_model_loads_once_per_model(monkeypatch, tmp_path):
    loads: list[str] = []

    class _Ranker:
        def __init__(self, model_name, cache_dir):
            loads.append(model_name)

    monkeypatch.setitem(sys.modules, "flashrank", types.SimpleNamespace(Ranker=_Ranker))
    monkeypatch.setenv("RAG_FLASHRANK_MODEL_NAME", "tiny")
    monkeypatch.setenv("RAG_FLASHRANK_CACHE_DIR", str(tmp_path))

    for _ in range(3):
        ar.create_reranker(ar.RerankingStrategy.FLASHRANK)
    # Hybrid builds its own FlashRank reranker; it must reuse the same model.
    ar.create_reranker(ar.RerankingStrategy.HYBRID)
    assert loads == ["tiny"]

    ar.create_reranker(ar.RerankingStrategy.FLASHRANK, ar.RerankingConfig(model_name="other"))
    assert loads == ["tiny", "other"]


def test_cross_encoder_model_loads_once_per_key(monkeypatch):
    loads: list[tuple] = []

    class _CrossEncoder:
        def __init__(self, model_id, **kwargs):
            loads.append((model_id, kwargs.get("revision")))

    monkeypatch.setitem(
        sys.modules, "sentence_transformers", types.SimpleNamespace(CrossEncoder=_CrossEncoder)
    )

    def _cfg(**kw):
        return ar.RerankingConfig(
            strategy=ar.RerankingStrategy.CROSS_ENCODER,
            model_name="BAAI/bge-reranker-base",
            transformers_local_files_only=True,
            **kw,
        )

    first = ar.create_reranker(ar.RerankingStrategy.CROSS_ENCODER, _cfg())
    second = ar.create_reranker(ar.RerankingStrategy.CROSS_ENCODER, _cfg(top_k=3))
    assert loads == [("BAAI/bge-reranker-base", None)]
    assert first._ce is second._ce

    ar.create_reranker(ar.RerankingStrategy.CROSS_ENCODER, _cfg(hf_revision="abc123"))
    assert len(loads) == 2


def test_failed_load_is_not_cached(monkeypatch, tmp_path):
    attempts: list[int] = []

    class _FlakyRanker:
        def __init__(self, model_name, cache_dir):
            attempts.append(1)
            if len(attempts) == 1:
                raise RuntimeError("download failed")

    monkeypatch.setitem(sys.modules, "flashrank", types.SimpleNamespace(Ranker=_FlakyRanker))
    monkeypatch.setenv("RAG_FLASHRANK_MODEL_NAME", "tiny")
    monkeypatch.setenv("RAG_FLASHRANK_CACHE_DIR", str(tmp_path))

    assert ar.FlashRankReranker(ar.RerankingConfig())._ranker is None
    assert ar.FlashRankReranker(ar.RerankingConfig())._ranker is not None
    assert ar.FlashRankReranker(ar.RerankingConfig())._ranker is not None
    assert len(attempts) == 2
