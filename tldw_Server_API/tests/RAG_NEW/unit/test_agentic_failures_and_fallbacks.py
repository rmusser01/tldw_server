import pytest

import tldw_Server_API.app.core.RAG.rag_service.agentic_chunker as ac
from tldw_Server_API.app.core.RAG.rag_service.types import Document, DataSource


def _doc() -> Document:


    return Document(
        id="m_fail_1",
        content=("# Intro\nTransformers rely on attention.\n\n# Details\nMulti-head attention improves capacity."),
        metadata={"title": "Transformer"},
        source=DataSource.MEDIA_DB,
        score=0.8,
    )


@pytest.mark.asyncio
async def test_planner_failure_falls_back_to_heuristics(monkeypatch):
    class FakeRetriever:
        def __init__(self, *args, **kwargs):
            pass

        async def retrieve(self, *args, **kwargs):
            return [_doc()]

    monkeypatch.setattr(ac, "MultiDatabaseRetriever", FakeRetriever)

    class ExplodingPlanner:
        def __init__(self, *args, **kwargs):
            pass

        async def generate(self, *args, **kwargs):  # noqa: ARG002
            raise RuntimeError("planner explode")

    # Force LLM planner path to raise; tool loop should continue via heuristics and not crash
    monkeypatch.setattr(ac, "AnswerGenerator", ExplodingPlanner, raising=False)

    cfg = ac.AgenticConfig(top_k_docs=1, enable_tools=True, use_llm_planner=True, time_budget_sec=2.0, debug_trace=True)
    res = await ac.agentic_rag_pipeline(
        query="explain attention",
        sources=["media_db"],
        search_mode="fts",
        agentic=cfg,
        enable_generation=False,
        debug_mode=True,
    )
    # Should succeed with a non-empty synthetic chunk
    assert res.documents and len(res.documents[0].content if hasattr(res.documents[0], "content") else res.documents[0]["content"]) > 0


@pytest.mark.asyncio
async def test_time_budget_breach_early_stop(monkeypatch):
    class FakeRetriever:
        def __init__(self, *args, **kwargs):
            pass

        async def retrieve(self, *args, **kwargs):
            return [_doc()]

    monkeypatch.setattr(ac, "MultiDatabaseRetriever", FakeRetriever)

    # Set time budget to 0 to force immediate stop; fallback to at least one snippet
    cfg = ac.AgenticConfig(top_k_docs=1, enable_tools=True, max_tool_calls=10, time_budget_sec=0.0, debug_trace=True)
    res = await ac.agentic_rag_pipeline(
        query="transformer attention",
        sources=["media_db"],
        search_mode="hybrid",
        agentic=cfg,
        enable_generation=False,
        debug_mode=True,
    )
    # Synthetic doc exists
    assert res.documents and res.documents[0]
    syn = res.documents[0]
    text = syn.get("content") if isinstance(syn, dict) else syn.content
    assert isinstance(text, str) and len(text) > 0


@pytest.mark.asyncio
async def test_tool_loop_exhaustion_respects_max_calls(monkeypatch):
    class FakeRetriever:
        def __init__(self, *args, **kwargs):
            pass

        async def retrieve(self, *args, **kwargs):
            return [_doc()]

    monkeypatch.setattr(ac, "MultiDatabaseRetriever", FakeRetriever)

    cfg = ac.AgenticConfig(top_k_docs=1, enable_tools=True, max_tool_calls=1, time_budget_sec=2.0, debug_trace=True)
    res = await ac.agentic_rag_pipeline(
        query="attention windows",
        sources=["media_db"],
        search_mode="fts",
        agentic=cfg,
        enable_generation=False,
        debug_mode=True,
    )
    md = res.metadata or {}
    # Tool trace entries are only for expand_window steps; should be <= 1 when max_tool_calls=1
    trace = md.get("tool_trace") or []
    assert isinstance(trace, list)
    assert len([t for t in trace if t.get("tool") == "expand_window"]) <= 1


@pytest.mark.asyncio
async def test_planner_quota_refusal_falls_back_to_heuristics(monkeypatch):
    """A refused LLM planner degrades to heuristic planning instead of failing the search (spec §4a, §4b)."""
    import tldw_Server_API.app.core.RAG.rag_service.agentic_execution as agentic_execution
    from tldw_Server_API.app.core.exceptions import LLMQuotaExceededError

    class FakeRetriever:
        def __init__(self, *args, **kwargs):
            pass

        async def retrieve(self, *args, **kwargs):
            return [_doc()]

    class RefusedPlanner:
        def __init__(self, *args, **kwargs):
            pass

        async def generate(self, *args, **kwargs):  # noqa: ARG002
            raise LLMQuotaExceededError(
                category="llm_tokens_month", current=0, limit=0, message="Monthly LLM token limit reached", retry_after=60
            )

    monkeypatch.setattr(ac, "MultiDatabaseRetriever", FakeRetriever)
    monkeypatch.setattr(agentic_execution, "AnswerGenerator", RefusedPlanner)

    cfg = ac.AgenticConfig(top_k_docs=1, enable_tools=True, use_llm_planner=True, time_budget_sec=2.0)
    res = await ac.agentic_rag_pipeline(
        query="explain attention",
        sources=["media_db"],
        search_mode="fts",
        agentic=cfg,
        enable_generation=False,
    )

    assert res.documents
