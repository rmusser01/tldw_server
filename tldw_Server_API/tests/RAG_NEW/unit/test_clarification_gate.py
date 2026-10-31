import pytest

from tldw_Server_API.app.core.RAG.rag_service.clarification_gate import (
    ClarificationDecision,
    assess_query_for_clarification,
)


pytestmark = pytest.mark.unit


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query",
    [
        "When does Project Juniper launch, and who owns it? Cite the source.",
        "Describe Project Juniper. When does it launch?",
    ],
)
async def test_named_subject_is_context_for_later_pronoun(query):
    decision = await assess_query_for_clarification(query, chat_history=None)
    assert decision.required is False


@pytest.mark.asyncio
@pytest.mark.parametrize("query", ["Can you fix this?", "Tell Me About It", "When does it launch?"])
async def test_pronoun_without_context_requires_clarification(query):
    d = await assess_query_for_clarification(
        query=query,
        chat_history=None,
        timeout_sec=0.1,
        llm_call=None,
    )
    assert isinstance(d, ClarificationDecision)
    assert d.required is True
    assert "clarify" in (d.question or "").lower()
    assert d.detector in {"heuristic", "hybrid"}


@pytest.mark.asyncio
async def test_specific_query_does_not_require_clarification():
    d = await assess_query_for_clarification(
        query="Summarize the key findings from the 2025 RAG benchmark section",
        chat_history=[],
        timeout_sec=0.1,
        llm_call=None,
    )
    assert d.required is False


@pytest.mark.asyncio
async def test_borderline_query_uses_llm_and_fails_open_on_timeout():
    async def _slow_llm(_query, _history):
        raise TimeoutError("simulated timeout")

    d = await assess_query_for_clarification(
        query="What about that one?",
        chat_history=[{"role": "user", "content": "Discuss retrieval methods."}],
        timeout_sec=0.01,
        llm_call=_slow_llm,
    )
    assert d.required is False
    assert d.reason in {"llm_timeout_fallback", "fail_open"}
