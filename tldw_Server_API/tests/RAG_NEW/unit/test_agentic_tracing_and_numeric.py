import pytest

import tldw_Server_API.app.core.RAG.rag_service.agentic_chunker as ac
from tldw_Server_API.app.core.RAG.rag_service.types import DataSource, Document


def _make_doc(doc_id: str, content: str, title: str = "Doc") -> Document:
    return Document(
        id=doc_id,
        content=content,
        metadata={"title": title, "source": "media_db", "ingestion_date": "2024-01-01"},
        source=DataSource.MEDIA_DB,
        score=0.9,
    )


@pytest.mark.asyncio
async def test_agentic_tool_trace_and_metrics(monkeypatch):
    # Fake retriever returns single document
    class FakeRetriever:
        def __init__(self, *args, **kwargs):
            pass
        async def retrieve(self, *args, **kwargs):
            return [
                _make_doc(
                    "m1",
                    "# Intro\nTransformers rely on attention.\n\n# Details\nMulti-head attention improves capacity.",
                    title="Transformer",
                )
            ]

    monkeypatch.setattr(ac, "MultiDatabaseRetriever", FakeRetriever)

    cfg = ac.AgenticConfig(
        top_k_docs=1,
        enable_tools=True,
        max_tool_calls=3,
        time_budget_sec=2.0,
        enable_semantic_within=True,
        enable_section_index=True,
        debug_trace=True,
    )

    res = await ac.agentic_rag_pipeline(
        query="explain attention",
        sources=["media_db"],
        search_mode="fts",
        agentic=cfg,
        enable_generation=False,
        debug_mode=True,
    )

    md = res.metadata or {}
    # Tool trace should be present when tools enabled and debug on
    assert isinstance(md.get("tool_trace"), list)
    # Agentic metrics should include coverage values
    am = md.get("agentic_metrics") or {}
    assert "term_coverage" in am and isinstance(am.get("term_coverage"), float)


@pytest.mark.asyncio
async def test_agentic_hard_citations_and_numeric(monkeypatch):
    # Patch retriever
    class FakeRetriever:
        def __init__(self, *args, **kwargs):
            pass
        async def retrieve(self, *args, **kwargs):
            # Include a numeric token 42 in the source
            return [_make_doc("m2", "We observed 42 experiments with consistent results.", title="Exp")]

    monkeypatch.setattr(ac, "MultiDatabaseRetriever", FakeRetriever)

    # Patch AnswerGenerator to return a small answer referencing the number
    class FakeAnswerGenerator:
        def __init__(self, *args, **kwargs):
            pass
        async def generate(self, *, query: str, context: str, prompt_template=None, max_tokens=None, temperature=None):  # noqa: ARG002
            return {"answer": "We observed 42 experiments with consistent results."}

    import tldw_Server_API.app.core.RAG.rag_service.generation as gen_mod
    monkeypatch.setattr(gen_mod, "AnswerGenerator", FakeAnswerGenerator)

    res = await ac.agentic_rag_pipeline(
        query="How many experiments were run?",
        sources=["media_db"],
        search_mode="fts",
        agentic=ac.AgenticConfig(top_k_docs=1),
        enable_generation=True,
        enable_numeric_fidelity=True,
        numeric_fidelity_behavior="continue",
        require_hard_citations=True,
        debug_mode=True,
    )

    md = res.metadata or {}
    # Hard citations should be present
    hc = md.get("hard_citations")
    assert isinstance(hc, dict)
    assert hc["coverage"] == 1.0
    # Numeric fidelity should show present token '42'
    nf = md.get("numeric_fidelity") or {}
    assert "present" in nf and any("42" in x for x in (nf.get("present") or []))
