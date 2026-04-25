import asyncio
import time

import pytest

import tldw_Server_API.app.core.RAG.rag_service.agentic_execution as agentic_execution
from tldw_Server_API.app.core.RAG.rag_service.evidence_models import DerivedEvidence
from tldw_Server_API.app.core.RAG.rag_service.request_resolution import ResolvedRAGRequest
from tldw_Server_API.app.core.RAG.rag_service.retrieval_plan import (
    RetrievalPlan,
    build_retrieval_plan,
)
from tldw_Server_API.app.core.RAG.rag_service.types import Document, DataSource
from tldw_Server_API.app.core.RAG.rag_service.agentic_chunker import (
    _assemble_ephemeral_chunk,
    AgenticConfig,
    agentic_rag_pipeline,
)


def make_doc(doc_id: str, content: str, title: str = "Doc") -> Document:
    return Document(
        id=doc_id,
        content=content,
        metadata={"title": title, "source": "media_db", "ingestion_date": "2024-01-01"},
        source=DataSource.MEDIA_DB,
        score=0.9,
    )


def test_assemble_ephemeral_chunk_basic():


    query = "dropout prevents overfitting"
    content = (
        "Deep learning models often overfit. One method is dropout, which randomly removes units during training. "
        "Dropout helps prevent overfitting by reducing co-adaptation of neurons."
    )
    docs = [make_doc("d1", content, title="DL")]

    cfg = AgenticConfig(top_k_docs=1, window_chars=400, max_tokens_read=500)
    chunk, prov = _assemble_ephemeral_chunk(docs, query, cfg)

    assert "dropout" in chunk.lower()
    assert prov and prov[0]["document_id"] == "d1"
    assert prov[0]["start"] >= 0 and prov[0]["end"] > prov[0]["start"]


def test_agentic_chunker_reexports_execution_helpers():
    import tldw_Server_API.app.core.RAG.rag_service.agentic_chunker as ac

    assert ac._assemble_ephemeral_chunk is agentic_execution.assemble_ephemeral_chunk
    assert ac.AgenticToolbox is agentic_execution.AgenticToolbox
    assert ac._decompose_query is agentic_execution.decompose_query
    assert ac._tool_loop is agentic_execution.tool_loop
    assert ac._INTRA_DOC_VEC_CACHE is agentic_execution._INTRA_DOC_VEC_CACHE


@pytest.mark.asyncio
async def test_agentic_pipeline_cache_hit(monkeypatch):
    # Prepare fake docs returned by retriever
    query = "batch normalization effect"
    content = (
        "Batch Normalization reduces internal covariate shift and can speed up training. "
        "It also allows for higher learning rates."
    )
    docs = [make_doc("m1", content, title="BN")]

    calls = {"count": 0}

    class FakeRetriever:
        def __init__(self, *args, **kwargs):
            pass

        async def retrieve(self, *args, **kwargs):
            calls["count"] += 1
            return docs

    # Patch the retriever used inside agentic_chunker
    import tldw_Server_API.app.core.RAG.rag_service.agentic_chunker as ac
    monkeypatch.setattr(ac, "MultiDatabaseRetriever", FakeRetriever)

    # First call (miss -> assemble -> cache)
    res1 = await agentic_rag_pipeline(
        query=query,
        sources=["media_db"],
        media_db_path=None,
        notes_db_path=None,
        character_db_path=None,
        search_mode="fts",
        agentic=AgenticConfig(top_k_docs=1, cache_ttl_sec=60),
        enable_generation=False,
        enable_citations=False,
    )
    assert res1.documents and res1.documents[0].content
    assert res1.cache_hit is False
    assert calls["count"] == 1

    # Second call with same query/docs (hit)
    res2 = await agentic_rag_pipeline(
        query=query,
        sources=["media_db"],
        media_db_path=None,
        notes_db_path=None,
        character_db_path=None,
        search_mode="fts",
        agentic=AgenticConfig(top_k_docs=1, cache_ttl_sec=60),
        enable_generation=False,
        enable_citations=False,
    )
    assert res2.cache_hit is True
    assert calls["count"] == 2  # retriever still called, but assemble path uses cache
    assert res2.documents[0].content == res1.documents[0].content


@pytest.mark.asyncio
async def test_agentic_tool_loop_heuristic(monkeypatch):
    # Ensure tool loop can run and returns non-empty chunk around hits
    query = "transformer attention"
    content = (
        "Introduction. The Transformer architecture relies on attention mechanisms.\n"
        "Methods. Multi-head attention allows the model to jointly attend to information."
    )
    docs = [make_doc("t1", content, title="Transformer")]

    class FakeRetriever:
        def __init__(self, *args, **kwargs):
            pass
        async def retrieve(self, *args, **kwargs):
            return docs

    import tldw_Server_API.app.core.RAG.rag_service.agentic_chunker as ac
    monkeypatch.setattr(ac, "MultiDatabaseRetriever", FakeRetriever)

    res = await agentic_rag_pipeline(
        query=query,
        sources=["media_db"],
        search_mode="hybrid",
        agentic=AgenticConfig(top_k_docs=1, enable_tools=True, max_tool_calls=4, time_budget_sec=2.0, enable_semantic_within=True, enable_section_index=True),
        enable_generation=False,
        enable_citations=False,
    )
    assert res.documents and len(res.documents[0].content) > 0


@pytest.mark.asyncio
async def test_agentic_query_decomposition_merge(monkeypatch):
    query = "Explain residual connections and dropout"
    content = (
        "# Residuals\nResidual connections help gradient flow in deep networks.\n\n"
        "# Regularization\nDropout helps prevent overfitting by reducing co-adaptation of neurons."
    )
    docs = [make_doc("t2", content, title="ResNet & Dropout")]

    class FakeRetriever:
        def __init__(self, *args, **kwargs):
            pass
        async def retrieve(self, *args, **kwargs):
            return docs

    import tldw_Server_API.app.core.RAG.rag_service.agentic_chunker as ac
    monkeypatch.setattr(ac, "MultiDatabaseRetriever", FakeRetriever)

    res = await agentic_rag_pipeline(
        query=query,
        sources=["media_db"],
        search_mode="fts",
        agentic=AgenticConfig(top_k_docs=1, enable_tools=True, enable_query_decomposition=True, subgoal_max=2),
        enable_generation=False,
        enable_citations=False,
    )
    text = (res.documents[0]["content"] if isinstance(res.documents[0], dict) else res.documents[0].content)
    assert "Residual".lower()[:7] in text.lower()
    assert "Dropout".lower()[:7] in text.lower()


@pytest.mark.asyncio
async def test_agentic_pipeline_uses_shared_retrieval_plan_and_derived_evidence_boundary(monkeypatch):
    query = "shared request resolution"
    docs = [
        make_doc("d1", "The first coarse document explains the primary evidence.", title="Doc 1"),
        make_doc("d2", "The second coarse document provides corroboration.", title="Doc 2"),
    ]
    captured: dict[str, object] = {}

    class FakeRetriever:
        def __init__(self, *args, **kwargs):
            pass

        async def retrieve(self, *args, **kwargs):
            captured["retrieve_kwargs"] = kwargs
            return docs

    def fake_build_agentic_derived_evidence(*, retrieved_evidence, synthetic_chunk, derived_from_document_ids, coarse_docs_window):
        captured["retrieved_evidence"] = retrieved_evidence
        captured["build_kwargs"] = {
            "synthetic_chunk": synthetic_chunk,
            "derived_from_document_ids": tuple(derived_from_document_ids),
            "coarse_docs_window": list(coarse_docs_window),
        }
        return DerivedEvidence(
            retrieved=retrieved_evidence,
            documents=[*retrieved_evidence.documents, synthetic_chunk],
            metadata=dict(retrieved_evidence.metadata),
            citations=[],
            verification_report=None,
            derived_from_document_ids=tuple(derived_from_document_ids),
        )

    import tldw_Server_API.app.core.RAG.rag_service.agentic_chunker as ac

    monkeypatch.setattr(ac, "MultiDatabaseRetriever", FakeRetriever)
    monkeypatch.setattr(ac, "build_agentic_derived_evidence", fake_build_agentic_derived_evidence)

    resolved_request = ResolvedRAGRequest(
        query=query,
        strategy="agentic",
        payload={
            "query": query,
            "sources": ["notes", "media_db"],
            "search_mode": "hybrid",
            "top_k": 2,
            "min_score": 0.25,
            "index_namespace": "tenant-x",
        },
        index_namespace="tenant-x",
        rag_profile="fast",
        user_id="17",
        feedback_user_id="17",
    )
    retrieval_plan = build_retrieval_plan(resolved_request)

    res = await agentic_rag_pipeline(
        query=query,
        sources=["characters"],
        media_db_path=None,
        notes_db_path=None,
        character_db_path=None,
        search_mode="vector",
        top_k=9,
        min_score=0.9,
        index_namespace="wrong",
        agentic=AgenticConfig(top_k_docs=2, cache_ttl_sec=5),
        enable_generation=False,
        enable_citations=True,
        resolved_request=resolved_request,
        retrieval_plan=retrieval_plan,
    )

    retrieve_kwargs = captured["retrieve_kwargs"]
    assert isinstance(retrieve_kwargs, dict)
    assert retrieve_kwargs["config"].max_results == 2
    assert retrieve_kwargs["config"].min_score == 0.25
    assert retrieve_kwargs["config"].use_fts is True
    assert retrieve_kwargs["config"].use_vector is True
    assert retrieve_kwargs["sources"] == [DataSource.NOTES, DataSource.MEDIA_DB]
    assert retrieve_kwargs["index_namespace"] == "tenant-x"
    assert captured["retrieved_evidence"].metadata["retrieval_plan"]["top_k"] == 2
    assert captured["build_kwargs"]["derived_from_document_ids"] == ("d1", "d2")
    assert captured["build_kwargs"]["coarse_docs_window"] == [
        {"id": "d1", "title": "Doc 1", "score": 0.9},
        {"id": "d2", "title": "Doc 2", "score": 0.9},
    ]
    assert res.documents and res.documents[0].content
    assert res.metadata["derived_from_document_ids"] == ["d1", "d2"]
    assert res.metadata["strategy"] == "agentic"


@pytest.mark.asyncio
async def test_agentic_pipeline_lineage_only_tracks_docs_used_for_synthetic_chunk(monkeypatch):
    query = "lineage should match assembled docs"
    docs = [
        make_doc("d1", "Primary evidence for the assembled synthetic chunk.", title="Doc 1"),
        make_doc("d2", "Secondary corroboration that should stay outside the lineage.", title="Doc 2"),
        make_doc("d3", "Tertiary context that should not appear in derived lineage.", title="Doc 3"),
    ]

    class FakeRetriever:
        def __init__(self, *args, **kwargs):
            pass

        async def retrieve(self, *args, **kwargs):
            return docs

    import tldw_Server_API.app.core.RAG.rag_service.agentic_chunker as ac

    monkeypatch.setattr(ac, "MultiDatabaseRetriever", FakeRetriever)

    resolved_request = ResolvedRAGRequest(
        query=query,
        strategy="agentic",
        payload={
            "query": query,
            "sources": ["media_db"],
            "search_mode": "fts",
            "top_k": 3,
            "min_score": 0.0,
            "index_namespace": "tenant-y",
        },
        index_namespace="tenant-y",
        rag_profile="fast",
        user_id="17",
        feedback_user_id="17",
    )

    res = await agentic_rag_pipeline(
        query=query,
        sources=["media_db"],
        search_mode="fts",
        top_k=3,
        min_score=0.0,
        agentic=AgenticConfig(top_k_docs=1, cache_ttl_sec=5),
        enable_generation=False,
        enable_citations=False,
        resolved_request=resolved_request,
        retrieval_plan=RetrievalPlan(
            query=query,
            sources=("media_db",),
            search_mode="fts",
            top_k=3,
            min_score=0.0,
            index_namespace="tenant-y",
            collection_names={"media_db": "tenant-y_media_db"},
        ),
    )

    assert res.metadata["derived_from_document_ids"] == ["d1"]


@pytest.mark.asyncio
async def test_agentic_pipeline_preserves_legacy_coarse_docs_window_and_actual_lineage(monkeypatch):
    query = "legacy coarse docs compatibility"
    docs = [
        make_doc("d1", "Primary evidence " * 40, title="Doc 1"),
        make_doc("d2", "Secondary evidence", title="Doc 2"),
        make_doc("d3", "Tertiary evidence", title="Doc 3"),
    ]

    class FakeRetriever:
        def __init__(self, *args, **kwargs):
            pass

        async def retrieve(self, *args, **kwargs):
            return docs

    import tldw_Server_API.app.core.RAG.rag_service.agentic_chunker as ac

    monkeypatch.setattr(ac, "MultiDatabaseRetriever", FakeRetriever)

    resolved_request = ResolvedRAGRequest(
        query=query,
        strategy="agentic",
        payload={
            "query": query,
            "sources": ["media_db"],
            "search_mode": "fts",
            "top_k": 5,
            "min_score": 0.0,
            "index_namespace": "tenant-legacy",
        },
        index_namespace="tenant-legacy",
        rag_profile="fast",
        user_id="17",
        feedback_user_id="17",
    )

    res = await agentic_rag_pipeline(
        query=query,
        sources=["media_db"],
        search_mode="fts",
        top_k=5,
        min_score=0.0,
        agentic=AgenticConfig(top_k_docs=3, max_tokens_read=4, cache_ttl_sec=5),
        enable_generation=False,
        enable_citations=False,
        resolved_request=resolved_request,
        retrieval_plan=RetrievalPlan(
            query=query,
            sources=("media_db",),
            search_mode="fts",
            top_k=5,
            min_score=0.0,
            index_namespace="tenant-legacy",
            collection_names={"media_db": "tenant-legacy_media_db"},
        ),
    )

    assert [entry["id"] for entry in res.metadata["coarse_docs"]] == ["d1", "d2", "d3"]
    assert res.metadata["derived_from_document_ids"] == ["d1"]


@pytest.mark.asyncio
async def test_agentic_pipeline_uses_resolved_scope_for_post_generation_steps(monkeypatch):
    query = "raw query"
    docs = [
        make_doc("d1", "Tenant-scoped evidence says the answer is 42.", title="Doc 1"),
    ]
    captured: dict[str, object] = {
        "retriever_inits": [],
        "retriever_calls": [],
    }

    class FakeRetriever:
        def __init__(self, *args, user_id=None, **kwargs):
            captured["retriever_inits"].append(user_id)

        async def retrieve(self, query, **kwargs):
            captured["retriever_calls"].append({"query": query, **kwargs})
            if len(captured["retriever_calls"]) == 1:
                return docs
            return []

    class FakeAnswerGenerator:
        def __init__(self, *args, **kwargs):
            pass

        async def generate(self, *, query: str, context: str, prompt_template=None, max_tokens=None, temperature=None):  # noqa: ARG002
            captured["generation_query"] = query
            return {"answer": "The answer is 42."}

    class FakeClaimsEngine:
        def __init__(self, *args, **kwargs):
            pass

        async def run(self, **kwargs):
            captured["claims_query"] = kwargs["query"]
            return {"claims": [], "summary": {}}

    class FakeNumericFidelity:
        present = {"42"}
        missing = {"99"}
        union_source_numbers = {"42", "99"}

    class FakeVerifier:
        def __init__(self, *args, **kwargs):
            pass

        async def verify_and_maybe_fix(self, **kwargs):
            captured["post_verify_kwargs"] = kwargs

            class Outcome:
                unsupported_ratio = 0.0
                total_claims = 0
                unsupported_count = 0
                fixed = False
                reason = ""

            return Outcome()

    import tldw_Server_API.app.core.RAG.rag_service.agentic_chunker as ac
    import tldw_Server_API.app.core.RAG.rag_service.claims as claims_mod
    import tldw_Server_API.app.core.RAG.rag_service.generation as generation_mod
    import tldw_Server_API.app.core.RAG.rag_service.guardrails as guardrails_mod
    import tldw_Server_API.app.core.RAG.rag_service.post_generation_verifier as verifier_mod

    monkeypatch.setattr(ac, "MultiDatabaseRetriever", FakeRetriever)
    monkeypatch.setattr(generation_mod, "AnswerGenerator", FakeAnswerGenerator, raising=False)
    monkeypatch.setattr(claims_mod, "ClaimsEngine", FakeClaimsEngine, raising=False)
    monkeypatch.setattr(
        guardrails_mod,
        "build_hard_citations",
        lambda *args, **kwargs: {"coverage": 1.0, "sentences": []},
    )
    monkeypatch.setattr(
        guardrails_mod,
        "check_numeric_fidelity",
        lambda *args, **kwargs: FakeNumericFidelity(),
    )
    monkeypatch.setattr(verifier_mod, "PostGenerationVerifier", FakeVerifier, raising=False)

    resolved_request = ResolvedRAGRequest(
        query="resolved query",
        strategy="agentic",
        payload={
            "query": "resolved query",
            "sources": ["media_db"],
            "search_mode": "hybrid",
            "top_k": 2,
            "min_score": 0.25,
            "hybrid_alpha": 0.33,
            "index_namespace": "tenant-z",
        },
        index_namespace="tenant-z",
        rag_profile="fast",
        user_id="17",
        feedback_user_id="17",
    )

    await agentic_rag_pipeline(
        query=query,
        sources=["media_db"],
        media_db_path="tenant-z.db",
        search_mode="vector",
        hybrid_alpha=0.9,
        top_k=9,
        min_score=0.9,
        index_namespace="wrong",
        agentic=AgenticConfig(top_k_docs=1, cache_ttl_sec=5),
        enable_generation=True,
        enable_citations=False,
        enable_claims=True,
        enable_numeric_fidelity=True,
        numeric_fidelity_behavior="retry",
        resolved_request=resolved_request,
        retrieval_plan=RetrievalPlan(
            query="resolved query",
            sources=("media_db",),
            search_mode="hybrid",
            top_k=2,
            min_score=0.25,
            index_namespace="tenant-z",
            collection_names={"media_db": "tenant-z_media_db"},
        ),
    )

    assert captured["generation_query"] == "resolved query"
    assert captured["claims_query"] == "resolved query"
    assert captured["retriever_inits"] == ["17", "17"]
    assert captured["retriever_calls"][0]["query"] == "resolved query"
    assert captured["retriever_calls"][1]["query"] == "resolved query 99"
    assert captured["retriever_calls"][1]["config"].max_results == 2
    assert captured["retriever_calls"][1]["config"].min_score == 0.25
    assert captured["retriever_calls"][1]["index_namespace"] == "tenant-z"
    assert captured["post_verify_kwargs"]["query"] == "resolved query"
    assert captured["post_verify_kwargs"]["user_id"] == "17"
    assert captured["post_verify_kwargs"]["search_mode"] == "hybrid"
    assert captured["post_verify_kwargs"]["hybrid_alpha"] == 0.33
    assert captured["post_verify_kwargs"]["top_k"] == 2


def test_open_section_anchor_and_table_heuristic():


     # Build toolbox indirectly via private helpers (unit-level check)
    from tldw_Server_API.app.core.RAG.rag_service.agentic_chunker import AgenticToolbox
    doc = make_doc(
        "t3",
        "# Methods\nWe describe the approach.\n\n# Results\nCol A|Col B|Col C\n1|2|3\n4|5|6\n",
        title="Paper",
    )
    cfg = AgenticConfig(enable_section_index=True, enable_semantic_within=True, enable_table_support=True)
    tb = AgenticToolbox([doc], cfg)
    sec = tb.open_section(doc, "Results")
    assert sec and sec[0] < sec[1]
    # Table-like detection
    spans = tb.search_within(doc, "table of results", max_hits=3)
    assert spans, "Expected some spans from search_within"
    # Reorder with looks_table
    reordered = sorted(spans, key=lambda rng: int(not tb.looks_table((doc.content or "")[rng[0]:rng[1]])))
    # At least the first after reordering should be the table-like region
    text_first = (doc.content or "")[reordered[0][0]:reordered[0][1]]
    assert tb.looks_table(text_first)


def test_get_media_db_for_structure_uses_shared_factory(monkeypatch):
    import tldw_Server_API.app.core.RAG.rag_service.agentic_execution as ae

    captured: dict[str, object] = {}

    monkeypatch.setattr(ae, "_STRUCT_DB", None)
    monkeypatch.setattr(
        "tldw_Server_API.app.core.DB_Management.media_db.api.create_media_database",
        lambda client_id, **kwargs: captured.update({"client_id": client_id, **kwargs}) or "db-sentinel",
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.config.load_comprehensive_config",
        lambda: {"stub": True},
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.DB_Management.content_backend.get_content_backend",
        lambda _cfg: "backend-sentinel",
    )

    db = ae._get_media_db_for_structure()

    assert db == "db-sentinel"
    assert captured == {
        "client_id": "agentic_toolbox",
        "db_path": ":memory:",
        "backend": "backend-sentinel",
    }
