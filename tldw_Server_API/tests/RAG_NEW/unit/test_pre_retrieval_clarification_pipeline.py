from unittest.mock import AsyncMock, patch
import asyncio

import pytest

import tldw_Server_API.app.core.RAG.rag_service.unified_pipeline as up


pytestmark = pytest.mark.unit


def test_rag_request_default_supports_reasoning_and_preserves_explicit_caps():
    from tldw_Server_API.app.core.RAG.rag_schemas_unified import UnifiedRAGRequest
    assert UnifiedRAGRequest(query="Education", generation_provider="deepseek").max_generation_tokens == 4096
    assert UnifiedRAGRequest(query="Education", max_generation_tokens=800).max_generation_tokens == 800
    assert UnifiedRAGRequest(query="Education", max_generation_tokens=8192).max_generation_tokens == 8192


@pytest.mark.asyncio
async def test_request_deadline_includes_work_before_retrieval():
    async def slow_spell_check(query):
        await asyncio.sleep(0.03)
        return query

    async def slow_retrieval(*args, **kwargs):
        await asyncio.sleep(0.04)
        return []

    with patch.object(up, "spell_check_query", slow_spell_check), patch.object(up, "MultiDatabaseRetriever") as factory:
        factory.return_value.retrieve = slow_retrieval
        with pytest.raises(TimeoutError):
            await up.unified_rag_pipeline(
                query="Education funding", spell_check=True, enable_generation=False,
                enable_cache=False, enable_reranking=False, timeout_seconds=0.05,
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", [{"include_media_ids": [1]}, {"include_note_ids": ["note-1"]}])
@pytest.mark.parametrize("override", [None, False, True])
async def test_selected_sources_disambiguate_only_the_default(scope, override):
    question = "What does this document propose for the Department of Education?"
    with patch.object(up, "assess_query_for_clarification", new=AsyncMock(
        return_value=up.ClarificationDecision(True, "Which document?", "ambiguous", 0.9, "heuristic")
    )) as assess:
        result = await up.unified_rag_pipeline(
            query=question, enable_generation=True, enable_cache=False,
            enable_pre_retrieval_clarification=override, **scope,
        )
    assert bool(result.metadata.get("clarification", {}).get("required")) is (override is True)
    assert assess.await_count == (1 if override is True else 0)


@pytest.mark.asyncio
async def test_pre_retrieval_clarification_short_circuits_retrieval():
    with patch.object(
        up,
        "assess_query_for_clarification",
        new=AsyncMock(
            return_value=up.ClarificationDecision(
                required=True,
                question="Could you clarify which document you mean?",
                reason="ambiguous",
                confidence=0.9,
                detector="heuristic",
            )
        ),
    ), patch.object(up, "MultiDatabaseRetriever") as mock_retriever:
        res = await up.unified_rag_pipeline(
            query="What about that one?",
            enable_generation=True,
        )
        assert res.generated_answer == "Could you clarify which document you mean?"
        assert res.metadata["clarification"]["required"] is True
        assert res.metadata["retrieval_bypassed"]["reason"] == "pre_retrieval_clarification"
        mock_retriever.assert_not_called()
