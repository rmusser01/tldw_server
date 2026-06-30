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


class ManyDocRetriever:
    def __init__(self, *args, **kwargs):
        self.retrievers = {}

    async def retrieve(self, query: str, sources=None, config=None, index_namespace=None, **kwargs):
        return [
            Document(
                id=f"m{i}",
                content=f"Document {i}",
                source=DataSource.MEDIA_DB,
                metadata={},
                score=1.0 - (i * 0.01),
            )
            for i in range(7)
        ]


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
            captured["context"] = context
            return {
                "answer": "provider-model-ok",
                "provider": "anthropic",
                "model": "claude-3-5-haiku-latest",
                "tokens_used": 22,
                "generation_time": 0.12,
            }

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
    md = getattr(res, "metadata", None) or (res.get("metadata") if isinstance(res, dict) else {})
    assert md.get("provider") == "anthropic"
    assert md.get("model") == "claude-3-5-haiku-latest"
    assert md.get("tokens_used") == 22
    assert md.get("generation_time") == 0.12


@pytest.mark.asyncio
async def test_standard_generation_routes_through_generation_executor(monkeypatch):
    monkeypatch.setattr(up, "MultiDatabaseRetriever", FakeRetriever)
    monkeypatch.setattr(up, "AnswerGenerator", FakeAnswerGenerator)

    captured: dict[str, object] = {}

    async def fake_execute_generation_phase(
        *,
        resolved_request,
        retrieval_plan,
        derived_evidence,
        generate_answer_fn,
        generation_context,
    ):
        captured["resolved_request"] = resolved_request
        captured["retrieval_plan"] = retrieval_plan
        captured["derived_evidence"] = derived_evidence
        captured["generate_answer_fn"] = generate_answer_fn
        captured["generation_context"] = generation_context
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
    assert captured.get("retrieval_plan") is not None
    assert derived_evidence is not None
    assert len(getattr(derived_evidence, "documents", [])) == 2
    assert captured.get("generation_context") == "First doc with content\n\nSecond doc with content"

    ga = getattr(res, "generated_answer", None) or (res.get("generated_answer") if isinstance(res, dict) else None)
    md = getattr(res, "metadata", None) or (res.get("metadata") if isinstance(res, dict) else {})
    assert ga == "executor answer"
    assert md.get("chunk_citations") == [{"id": "doc-1"}]


@pytest.mark.asyncio
async def test_generation_keeps_full_retrieval_set_when_executor_returns_smaller_document_sample(monkeypatch):
    monkeypatch.setattr(up, "MultiDatabaseRetriever", ManyDocRetriever)
    monkeypatch.setattr(up, "AnswerGenerator", FakeAnswerGenerator)

    async def fake_execute_generation_phase(
        *,
        resolved_request,
        retrieval_plan,
        derived_evidence,
        generate_answer_fn,
        generation_context,
    ):
        assert retrieval_plan is not None
        assert generation_context == "Document 0\n\nDocument 1\n\nDocument 2\n\nDocument 3\n\nDocument 4"
        return RAGResult(
            documents=list(derived_evidence.documents[:5]),
            query=resolved_request.query,
            metadata={"provider": "stub"},
            chunk_citations=[{"id": "doc-1"}],
            verification_report={"ok": True},
            generated_answer="executor answer",
        )

    monkeypatch.setattr(up, "execute_generation_phase", fake_execute_generation_phase)

    res = await up.unified_rag_pipeline(
        query="Keep all docs",
        sources=["media_db"],
        enable_cache=False,
        enable_reranking=False,
        enable_generation=True,
        top_k=7,
    )

    docs = getattr(res, "documents", None) or (res.get("documents") if isinstance(res, dict) else None)
    assert len(docs) == 7
    assert [doc["id"] if isinstance(doc, dict) else doc.id for doc in docs] == [f"m{i}" for i in range(7)]


@pytest.mark.asyncio
async def test_structured_response_generation_uses_writer_transformed_context(monkeypatch):
    monkeypatch.setattr(up, "MultiDatabaseRetriever", FakeRetriever)

    monkeypatch.setattr(up, "format_context_xml", lambda chunks: "XML::" + "|".join(chunk["content"] for chunk in chunks))
    monkeypatch.setattr(up, "build_writer_system_prompt", lambda mode, max_generation_tokens: f"SYSTEM::{mode}::{max_generation_tokens}")
    monkeypatch.setattr(up, "build_writer_user_prompt", lambda query, context_xml: f"USER::{query}::{context_xml}")
    monkeypatch.setattr(up, "get_writer_depth_policy", lambda **kwargs: {"mode": kwargs["mode"], "max_generation_tokens": kwargs["max_generation_tokens"]})

    captured: dict[str, object] = {}

    class CapturingStructuredAnswerGenerator:
        def __init__(self, *args, **kwargs):
            pass

        async def generate(self, *, query: str, context: str, prompt_template=None, max_tokens=None, temperature=None):
            captured["context"] = context
            captured["prompt_template"] = prompt_template
            return {"answer": "writer answer"}

    monkeypatch.setattr(up, "AnswerGenerator", CapturingStructuredAnswerGenerator)

    res = await up.unified_rag_pipeline(
        query="Use writer context",
        sources=["media_db"],
        enable_cache=False,
        enable_reranking=False,
        enable_generation=True,
        enable_structured_response=True,
        generation_prompt="base prompt",
        top_k=2,
    )

    assert captured["context"] == "USER::Use writer context::XML::First doc with content|Second doc with content"
    assert captured["prompt_template"] == "base prompt\n\nSYSTEM::balanced::500"
    ga = getattr(res, "generated_answer", None) or (res.get("generated_answer") if isinstance(res, dict) else None)
    assert ga == "writer answer"
