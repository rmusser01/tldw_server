import asyncio
import types
import pytest

from tldw_Server_API.app.core.RAG.rag_service.result_model import RAGResult
from tldw_Server_API.app.core.RAG.rag_service.types import Document, DataSource
import tldw_Server_API.app.core.RAG.rag_service.unified_pipeline as up


class FakeRetriever:
    def __init__(self, *args, **kwargs):
        self.retrievers = {}

    async def retrieve(self, query: str, sources=None, config=None, index_namespace=None, **kwargs):
        return [
            Document(id="m1", content="First doc with content", source=DataSource.MEDIA_DB, metadata={}, score=0.6),
            Document(id="m2", content="Second doc with content", source=DataSource.MEDIA_DB, metadata={}, score=0.5),
        ]


class FakeTwoTierReranker:
    def __init__(self, *args, **kwargs):
             # Force gating
        self.last_metadata = {"gated": True, "top_doc_prob": 0.1}

    async def rerank(self, query, documents, original_scores=None):
        # return original as scored docs (simplified)
        return [types.SimpleNamespace(document=d, rerank_score=getattr(d, "score", 0.0)) for d in documents]


class FakeAnswerGenerator:
    def __init__(self, *args, **kwargs):
        pass

    async def generate(self, *, query: str, context: str, prompt_template=None, max_tokens=None, temperature=None):
        # Distinguish calls by prompt content to simulate draft/refine
        if "CRITIQUE:" in context:
            return {"answer": "refined answer"}
        return {"answer": "draft answer"}


@pytest.mark.asyncio
async def test_abstention_ask_behavior(monkeypatch):
    # Patch retriever and reranker
    monkeypatch.setattr(up, "MultiDatabaseRetriever", FakeRetriever)
    # create_reranker is imported symbol in unified_pipeline
    monkeypatch.setattr(up, "create_reranker", lambda *a, **k: FakeTwoTierReranker())

    res = await up.unified_rag_pipeline(
        query="What is the thing?",
        sources=["media_db"],
        enable_cache=False,
        search_mode="hybrid",
        enable_reranking=True,
        reranking_strategy="two_tier",
        enable_generation=True,
        enable_abstention=True,
        abstention_behavior="ask",
        top_k=3,
    )
    # Gated → abstention path should provide a clarifying answer
    ga = getattr(res, "generated_answer", None) or (res.get("generated_answer") if isinstance(res, dict) else None)
    assert ga and ("clarify" in ga.lower() or "clarification" in ga.lower())


@pytest.mark.asyncio
async def test_multi_turn_synthesis_happy_path(monkeypatch):
    # Patch retriever and generator; disable reranking
    monkeypatch.setattr(up, "MultiDatabaseRetriever", FakeRetriever)
    monkeypatch.setattr(up, "AnswerGenerator", FakeAnswerGenerator)

    res = await up.unified_rag_pipeline(
        query="Explain topic X",
        sources=["media_db"],
        enable_cache=False,
        enable_reranking=False,
        enable_generation=True,
        enable_multi_turn_synthesis=True,
        synthesis_time_budget_sec=5.0,
        synthesis_draft_tokens=64,
        synthesis_refine_tokens=64,
        top_k=3,
    )
    # Expect refined answer and synthesis metadata
    ga = getattr(res, "generated_answer", None) or (res.get("generated_answer") if isinstance(res, dict) else None)
    md = getattr(res, "metadata", None) or (res.get("metadata") if isinstance(res, dict) else {})
    assert ga == "refined answer"
    syn = md.get("synthesis") if isinstance(md, dict) else None
    assert isinstance(syn, dict) and syn.get("enabled") is True
    assert set((syn.get("durations") or {}).keys()) == {"draft", "critique", "refine"}


@pytest.mark.asyncio
async def test_generation_uses_explicit_provider_and_model(monkeypatch):
    monkeypatch.setattr(up, "MultiDatabaseRetriever", FakeRetriever)

    captured: dict[str, object] = {}

    class CapturingAnswerGenerator:
        def __init__(self, *args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs

        async def generate(self, *, query: str, context: str, prompt_template=None, max_tokens=None, temperature=None):
            return {"answer": "provider-model-ok"}

    monkeypatch.setattr(up, "AnswerGenerator", CapturingAnswerGenerator)

    res = await up.unified_rag_pipeline(
        query="Use selected provider/model",
        sources=["media_db"],
        enable_cache=False,
        enable_reranking=False,
        enable_generation=True,
        generation_provider="anthropic",
        generation_model="claude-3-5-haiku-latest",
        top_k=2,
    )

    init_kwargs = captured.get("kwargs")
    assert isinstance(init_kwargs, dict)
    assert init_kwargs.get("provider") == "anthropic"
    assert init_kwargs.get("model") == "claude-3-5-haiku-latest"
    ga = getattr(res, "generated_answer", None) or (res.get("generated_answer") if isinstance(res, dict) else None)
    assert ga == "provider-model-ok"


@pytest.mark.asyncio
async def test_standard_generation_routes_through_generation_executor(monkeypatch):
    monkeypatch.setattr(up, "MultiDatabaseRetriever", FakeRetriever)
    monkeypatch.setattr(up, "AnswerGenerator", FakeAnswerGenerator)

    captured: dict[str, object] = {}

    async def fake_execute_generation_phase(*, resolved_request, derived_evidence, generate_answer_fn):
        captured["resolved_request"] = resolved_request
        captured["derived_evidence"] = derived_evidence
        captured["generate_answer_fn"] = generate_answer_fn
        return RAGResult(
            documents=list(derived_evidence.documents),
            query=resolved_request.query,
            metadata={"model": "stub"},
            chunk_citations=[{"id": "doc-1"}],
            verification_report={"ok": True},
            generated_answer="executor answer",
        )

    monkeypatch.setattr(up, "execute_generation_phase", fake_execute_generation_phase)

    res = await up.unified_rag_pipeline(
        query="Use executor path",
        sources=["media_db"],
        enable_cache=False,
        enable_reranking=False,
        enable_generation=True,
        generation_prompt="concise",
        max_generation_tokens=64,
        top_k=2,
    )

    resolved_request = captured.get("resolved_request")
    derived_evidence = captured.get("derived_evidence")

    assert resolved_request is not None
    assert getattr(resolved_request, "query", None) == "Use executor path"
    assert getattr(resolved_request, "payload", {}).get("generation_prompt") == "concise"
    assert getattr(resolved_request, "payload", {}).get("max_generation_tokens") == 64
    assert derived_evidence is not None
    assert len(getattr(derived_evidence, "documents", [])) == 2

    ga = getattr(res, "generated_answer", None) or (res.get("generated_answer") if isinstance(res, dict) else None)
    md = getattr(res, "metadata", None) or (res.get("metadata") if isinstance(res, dict) else {})
    assert ga == "executor answer"
    assert md.get("chunk_citations") == [{"id": "doc-1"}]
