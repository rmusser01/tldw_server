import asyncio
import time
from types import SimpleNamespace
from typing import Any

import pytest

import tldw_Server_API.app.core.RAG.rag_service.agentic_chunker as agentic_chunker
import tldw_Server_API.app.core.RAG.rag_service.generation as generation_module
import tldw_Server_API.app.core.RAG.rag_service.post_generation_verifier as verifier_module
import tldw_Server_API.app.core.RAG.rag_service.unified_pipeline as unified_pipeline_module
from tldw_Server_API.app.core.AuthNZ.byok_runtime import ByokResolutionError
from tldw_Server_API.app.core.Chat.Chat_Deps import ChatAuthenticationError
from tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib import SummaryProviderError
from tldw_Server_API.app.core.RAG.rag_service.advanced_reranking import (
    LLMReranker,
    RerankingConfig,
    RerankingStrategy,
)
from tldw_Server_API.app.core.RAG.rag_service.agentic_chunker import AgenticConfig
from tldw_Server_API.app.core.RAG.rag_service.document_grader import (
    DocumentGrader,
    GradingConfig,
)
from tldw_Server_API.app.core.RAG.rag_service.evidence_models import DerivedEvidence, RetrievedEvidence
from tldw_Server_API.app.core.RAG.rag_service.generation import AnswerGenerator
from tldw_Server_API.app.core.RAG.rag_service.generation_executor import execute_generation_phase
from tldw_Server_API.app.core.RAG.rag_service.faithfulness import FaithfulnessEvaluator
from tldw_Server_API.app.core.RAG.rag_service.post_generation_verifier import (
    PostGenerationVerifier,
)
from tldw_Server_API.app.core.RAG.rag_service.request_resolution import ResolvedRAGRequest
from tldw_Server_API.app.core.RAG.rag_service.retrieval_plan import RetrievalPlan
from tldw_Server_API.app.core.RAG.rag_service.result_model import RAGResult
from tldw_Server_API.app.core.RAG.rag_service.quality_graders import (
    FastGroundednessGrader,
    UtilityGrader,
)
from tldw_Server_API.app.core.RAG.rag_service.types import DataSource, Document


pytestmark = pytest.mark.unit


class _RecordingCredentialRuntime:
    def __init__(self) -> None:
        self.handle = SimpleNamespace(
            provider="anthropic",
            api_key="runtime-only-key",
            app_config={"Anthropic": {"api_timeout": 12}},
            credentials_resolved=True,
        )
        self.resolved: list[str] = []
        self.marked: list[Any] = []

    async def resolve(self, provider: str) -> Any:
        self.resolved.append(provider)
        return self.handle

    async def mark_used(self, handle: Any) -> None:
        self.marked.append(handle)


def _stub_real_sgl_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    response: str,
) -> list[tuple[tuple[Any, ...], dict[str, Any]]]:
    import tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib as sgl

    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def dispatch(*args: Any, **kwargs: Any) -> str:
        calls.append((args, kwargs))
        return response

    monkeypatch.setattr(sgl, "_dispatch_to_api", dispatch)
    return calls


async def _run_unified_bound_sgl_stage(
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    runtime: _RecordingCredentialRuntime,
) -> Any:
    document = Document(
        id=f"doc-{stage}",
        content="Credential-scoped evidence for the requested answer.",
        metadata={},
        source=DataSource.MEDIA_DB,
        score=0.8,
    )

    class FakeRetriever:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def retrieve(self, *args: Any, **kwargs: Any) -> list[Document]:
            return [document]

    class FakeAnswerGenerator:
        def __init__(self, **kwargs: Any) -> None:
            pass

        async def generate(self, **kwargs: Any) -> dict[str, str]:
            return {"answer": "A grounded generated answer."}

    import tldw_Server_API.app.core.config as core_config

    monkeypatch.setattr(unified_pipeline_module, "MultiDatabaseRetriever", FakeRetriever)
    monkeypatch.setattr(unified_pipeline_module, "AnswerGenerator", FakeAnswerGenerator)
    monkeypatch.setattr(
        core_config,
        "load_and_log_configs",
        lambda: {
            "RAG_DEFAULT_LLM_PROVIDER": "anthropic",
            "RAG_DEFAULT_LLM_MODEL": "claude-test",
            "RAG_LLM_RERANKER_PROVIDER": "anthropic",
            "RAG_LLM_RERANKER_MODEL": "claude-rerank-test",
        },
    )

    options: dict[str, Any] = {
        "enable_cache": False,
        "enable_reranking": False,
        "enable_generation": False,
        "enable_pre_retrieval_clarification": False,
    }
    if stage == "gap":
        options["enable_gap_analysis"] = True
    elif stage == "reranker":
        options.update(enable_reranking=True, reranking_strategy="llm_scoring")
    elif stage == "critique":
        options.update(enable_generation=True, enable_multi_turn_synthesis=True)
    elif stage == "faithfulness":
        options.update(
            enable_generation=True,
            enable_faithfulness_eval=True,
            generation_provider="anthropic",
        )
    else:
        raise AssertionError(f"unsupported stage: {stage}")

    return await unified_pipeline_module.unified_rag_pipeline(
        query="How are request credentials applied?",
        sources=["media_db"],
        credential_runtime=runtime,
        **options,
    )


@pytest.mark.parametrize(
    ("stage", "dispatch_response"),
    [
        ("document", '{"is_relevant": true, "relevance_score": 0.9, "reasoning": "yes"}'),
        ("groundedness", '{"is_grounded": true, "confidence": 0.9, "reasoning": "yes"}'),
        ("utility", '{"utility_score": 5, "reasoning": "useful"}'),
    ],
)
@pytest.mark.asyncio
async def test_bound_grader_real_sgl_dispatches_nonempty_input_and_marks_once(
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    dispatch_response: str,
) -> None:
    import tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib as sgl

    runtime = _RecordingCredentialRuntime()
    calls = _stub_real_sgl_dispatch(monkeypatch, dispatch_response)
    document = Document(
        id=f"doc-{stage}",
        content="Evidence content is available for evaluation.",
        metadata={},
        source=DataSource.MEDIA_DB,
        score=0.8,
    )

    if stage == "document":
        result = await DocumentGrader(
            analyze_fn=sgl.analyze,
            config=GradingConfig(provider="anthropic"),
            credential_runtime=runtime,
        ).grade_document("is this relevant?", document)
        assert result.method == "llm"  # nosec B101
    elif stage == "groundedness":
        result = await FastGroundednessGrader(
            analyze_fn=sgl.analyze,
            provider="anthropic",
            credential_runtime=runtime,
        ).grade("query", "grounded answer", [document])
        assert result.method == "llm"  # nosec B101
    else:
        result = await UtilityGrader(
            analyze_fn=sgl.analyze,
            provider="anthropic",
            credential_runtime=runtime,
        ).grade("query", "useful answer")
        assert result.method == "llm"  # nosec B101

    assert len(calls) == 1  # nosec B101
    args, kwargs = calls[0]
    assert args[0].strip()  # nosec B101
    assert args[3] == "runtime-only-key"  # nosec B101
    assert kwargs["app_config"] == runtime.handle.app_config  # nosec B101
    assert kwargs["credentials_resolved"] is True  # nosec B101
    assert kwargs["raise_on_error"] is True  # nosec B101
    assert runtime.marked == [runtime.handle]  # nosec B101


@pytest.mark.parametrize("stage", ["document", "groundedness", "utility"])
@pytest.mark.asyncio
async def test_bound_grader_error_result_is_unavailable_and_unmarked(
    stage: str,
) -> None:
    runtime = _RecordingCredentialRuntime()

    def error_result(*args: Any, **kwargs: Any) -> str:
        return "Error: Could not extract text content. private-provider-detail"

    document = Document(
        id=f"doc-error-{stage}",
        content="Evidence content.",
        metadata={},
        source=DataSource.MEDIA_DB,
        score=0.8,
    )
    if stage == "document":
        result = await DocumentGrader(
            analyze_fn=error_result,
            config=GradingConfig(provider="anthropic"),
            credential_runtime=runtime,
        ).grade_document("query", document)
        assert result.method == "score_fallback"  # nosec B101
    elif stage == "groundedness":
        result = await FastGroundednessGrader(
            analyze_fn=error_result,
            provider="anthropic",
            credential_runtime=runtime,
        ).grade("query", "answer", [document])
        assert result.method == "heuristic"  # nosec B101
    else:
        result = await UtilityGrader(
            analyze_fn=error_result,
            provider="anthropic",
            credential_runtime=runtime,
        ).grade("query", "answer")
        assert result.method == "heuristic"  # nosec B101

    assert runtime.marked == []  # nosec B101
    assert result.metadata == {  # nosec B101
        "error": "provider_unavailable",
        "verification_available": False,
    }
    assert "private-provider-detail" not in str(result)  # nosec B101


@pytest.mark.parametrize(
    ("stage", "dispatch_response", "resolved_provider"),
    [
        ("gap", '["follow up safely"]', "anthropic"),
        ("reranker", "0.8", "anthropic"),
        ("critique", "- no unsupported claims", "openai"),
        ("faithfulness", "[]", "anthropic"),
    ],
)
@pytest.mark.asyncio
async def test_unified_bound_real_sgl_stage_dispatches_nonempty_input_and_marks_once(
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    dispatch_response: str,
    resolved_provider: str,
) -> None:
    runtime = _RecordingCredentialRuntime()
    calls = _stub_real_sgl_dispatch(monkeypatch, dispatch_response)

    result = await _run_unified_bound_sgl_stage(monkeypatch, stage, runtime)

    assert len(calls) == 1  # nosec B101
    args, kwargs = calls[0]
    assert args[0].strip()  # nosec B101
    assert args[3] == "runtime-only-key"  # nosec B101
    assert kwargs["app_config"] == runtime.handle.app_config  # nosec B101
    assert kwargs["credentials_resolved"] is True  # nosec B101
    assert kwargs["raise_on_error"] is True  # nosec B101
    assert runtime.resolved == [resolved_provider]  # nosec B101
    assert runtime.marked == [runtime.handle]  # nosec B101
    assert "verification_available" not in result.metadata.get(stage, {})  # nosec B101


@pytest.mark.parametrize(
    ("stage", "metadata_key"),
    [
        ("gap", "gap_analysis"),
        ("reranker", "reranking"),
        ("critique", "synthesis"),
        ("faithfulness", "faithfulness"),
    ],
)
@pytest.mark.asyncio
async def test_unified_bound_sgl_error_result_is_unavailable_and_unmarked(
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    metadata_key: str,
) -> None:
    runtime = _RecordingCredentialRuntime()
    sensitive = "private-provider-detail"
    calls = _stub_real_sgl_dispatch(monkeypatch, f"Error: provider failed {sensitive}")

    result = await _run_unified_bound_sgl_stage(monkeypatch, stage, runtime)

    assert len(calls) == 1  # nosec B101
    assert runtime.marked == []  # nosec B101
    assert result.metadata[metadata_key]["failure_code"] == "provider_unavailable"  # nosec B101
    assert result.metadata[metadata_key]["verification_available"] is False  # nosec B101
    assert sensitive not in str(result.metadata)  # nosec B101
    assert sensitive not in str(result.errors)  # nosec B101


@pytest.mark.asyncio
async def test_execute_generation_phase_builds_rag_result_from_derived_evidence():
    resolved = ResolvedRAGRequest(
        query="summarize",
        strategy="standard",
        payload={"enable_generation": True, "generation_prompt": "concise"},
        index_namespace="tenant-a",
        rag_profile=None,
        user_id="7",
        feedback_user_id="7",
    )
    derived = DerivedEvidence(
        retrieved=RetrievedEvidence(documents=[], metadata={"verification_report": {"ok": True}}),
        documents=[{"id": "doc-1", "content": "evidence"}],
        metadata={"chunk_citations": [{"id": "doc-1"}]},
        citations=[{"id": "doc-1"}],
        verification_report={"ok": True},
    )

    async def fake_generate_answer(**kwargs):
        assert kwargs["context"] == "writer context"
        return {
            "answer": "short answer",
            "provider": "stub-provider",
            "model": "stub-model",
            "tokens_used": 17,
            "generation_time": 0.25,
            "metadata": {"nested": "value"},
        }

    result = await execute_generation_phase(
        resolved_request=resolved,
        retrieval_plan=RetrievalPlan(
            query="summarize",
            sources=("media_db",),
            search_mode="hybrid",
            top_k=5,
            min_score=0.0,
            index_namespace="tenant-a",
        ),
        derived_evidence=derived,
        generate_answer_fn=fake_generate_answer,
        generation_context="writer context",
    )

    assert isinstance(result, RAGResult)
    assert result.generated_answer == "short answer"
    assert result.chunk_citations == [{"id": "doc-1"}]
    assert result.verification_report == {"ok": True}
    assert result.metadata["provider"] == "stub-provider"
    assert result.metadata["model"] == "stub-model"
    assert result.metadata["tokens_used"] == 17
    assert result.metadata["generation_time"] == 0.25
    assert result.metadata["nested"] == "value"


@pytest.mark.asyncio
async def test_answer_generator_runtime_uses_effective_provider_handle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _RecordingCredentialRuntime()
    captured: dict[str, Any] = {}

    async def fake_chat_call(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"choices": [{"message": {"content": "credentialized answer"}}]}

    import tldw_Server_API.app.core.Chat.chat_service as chat_service

    monkeypatch.setattr(chat_service, "perform_chat_api_call_async", fake_chat_call)

    result = await AnswerGenerator(
        provider="anthropic",
        model="claude-test",
        credential_runtime=runtime,
    ).generate(query="question", context="evidence")

    assert result["answer"] == "credentialized answer"  # nosec B101
    assert runtime.resolved == ["anthropic"]  # nosec B101
    assert runtime.marked == [runtime.handle]  # nosec B101
    assert captured["api_key"] == "runtime-only-key"  # nosec B101
    assert captured["app_config"] == {"Anthropic": {"api_timeout": 12}}  # nosec B101
    assert captured["credentials_resolved"] is True  # nosec B101


@pytest.mark.asyncio
async def test_answer_generator_runtime_propagates_typed_failure_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _RecordingCredentialRuntime()

    async def failing_chat_call(**kwargs: Any) -> None:  # noqa: ARG001
        raise ChatAuthenticationError("sensitive upstream body", provider="anthropic")

    import tldw_Server_API.app.core.Chat.chat_service as chat_service

    monkeypatch.setattr(chat_service, "perform_chat_api_call_async", failing_chat_call)

    with pytest.raises(ChatAuthenticationError):
        await AnswerGenerator(
            provider="anthropic",
            model="claude-test",
            credential_runtime=runtime,
        ).generate(query="question", context="evidence")

    assert runtime.marked == []  # nosec B101


@pytest.mark.asyncio
async def test_agentic_generation_consumes_task_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _RecordingCredentialRuntime()
    captured: dict[str, Any] = {}
    document = Document(
        id="doc-1",
        content="Credential runtimes keep provider calls execution-scoped.",
        metadata={"title": "Runtime", "source": "media_db"},
        source=DataSource.MEDIA_DB,
        score=0.9,
    )

    class FakeRetriever:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def retrieve(self, *args: Any, **kwargs: Any) -> list[Document]:
            return [document]

    class FakeAnswerGenerator:
        def __init__(self, **kwargs: Any) -> None:
            captured["runtime"] = kwargs.get("credential_runtime")

        async def generate(self, **kwargs: Any) -> dict[str, str]:
            return {"answer": "agentic answer"}

    monkeypatch.setattr(agentic_chunker, "MultiDatabaseRetriever", FakeRetriever)
    monkeypatch.setattr(generation_module, "AnswerGenerator", FakeAnswerGenerator)

    result = await agentic_chunker.agentic_rag_pipeline(
        query="How are credentials scoped?",
        sources=["media_db"],
        search_mode="fts",
        agentic=AgenticConfig(top_k_docs=1, enable_tools=False),
        enable_generation=True,
        generation_provider="anthropic",
        credential_runtime=runtime,
    )

    assert result.generated_answer == "agentic answer"  # nosec B101
    assert captured["runtime"] is runtime  # nosec B101


@pytest.mark.asyncio
async def test_document_grader_runtime_failure_uses_bounded_degraded_metadata() -> None:
    class FailingRuntime:
        async def resolve(self, provider: str) -> Any:
            raise ByokResolutionError("invalid_provider_credentials", provider)

    def unexpected_analyze(*args: Any, **kwargs: Any) -> str:
        raise AssertionError("configured analyzer must not run after runtime failure")

    document = Document(
        id="doc-grade",
        content="relevant evidence",
        metadata={},
        source=DataSource.MEDIA_DB,
        score=0.8,
    )
    result = await DocumentGrader(
        analyze_fn=unexpected_analyze,
        config=GradingConfig(provider="anthropic"),
        credential_runtime=FailingRuntime(),
    ).grade_document("query", document)

    assert result.method == "score_fallback"  # nosec B101
    assert result.metadata == {  # nosec B101
        "error": "invalid_provider_credentials",
        "verification_available": False,
    }
    assert "configured analyzer" not in str(result)  # nosec B101


@pytest.mark.asyncio
async def test_quality_graders_runtime_failure_lowers_verification_trust() -> None:
    class FailingRuntime:
        async def resolve(self, provider: str) -> Any:
            raise ByokResolutionError("credential_store_unavailable", provider)

    def unexpected_analyze(*args: Any, **kwargs: Any) -> str:
        raise AssertionError("configured analyzer must not run after runtime failure")

    document = Document(
        id="doc-quality",
        content="the answer is grounded in this evidence",
        metadata={},
        source=DataSource.MEDIA_DB,
        score=0.8,
    )
    groundedness = await FastGroundednessGrader(
        analyze_fn=unexpected_analyze,
        provider="anthropic",
        credential_runtime=FailingRuntime(),
    ).grade("query", "grounded answer", [document])
    utility = await UtilityGrader(
        analyze_fn=unexpected_analyze,
        provider="anthropic",
        credential_runtime=FailingRuntime(),
    ).grade("query", "grounded answer")

    expected = {
        "error": "credential_store_unavailable",
        "verification_available": False,
    }
    assert groundedness.method == "heuristic"  # nosec B101
    assert groundedness.metadata == expected  # nosec B101
    assert utility.method == "heuristic"  # nosec B101
    assert utility.metadata == expected  # nosec B101


@pytest.mark.asyncio
async def test_llm_reranker_provider_failure_skips_with_reduced_trust() -> None:
    class FailingBoundClient:
        def analyze(self, prompt: str) -> str:  # noqa: ARG002
            raise SummaryProviderError(code="authentication", provider="anthropic")

    document = Document(
        id="doc-rerank",
        content="evidence",
        metadata={},
        source=DataSource.MEDIA_DB,
        score=0.8,
    )
    reranker = LLMReranker(
        RerankingConfig(
            strategy=RerankingStrategy.LLM_SCORING,
            top_k=1,
            batch_size=1,
        ),
        llm_client=FailingBoundClient(),
    )

    reranked = await reranker.rerank("query", [document])

    assert reranked[0].rerank_score == document.score  # nosec B101
    assert reranker.last_metadata == {  # nosec B101
        "degraded": True,
        "failure_code": "provider_unavailable",
        "verification_available": False,
    }


@pytest.mark.parametrize(
    ("strategy_name", "strategy_enum"),
    [
        ("llm_scoring", RerankingStrategy.LLM_SCORING),
        ("two_tier", RerankingStrategy.TWO_TIER),
    ],
)
@pytest.mark.asyncio
async def test_unified_llm_reranker_runtime_failure_does_not_fail_over_provider(
    monkeypatch: pytest.MonkeyPatch,
    strategy_name: str,
    strategy_enum: RerankingStrategy,
) -> None:
    captured: dict[str, Any] = {}
    document = Document(
        id="doc-pipeline-rerank",
        content="pipeline evidence",
        metadata={},
        source=DataSource.MEDIA_DB,
        score=0.8,
    )

    class FakeRetriever:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def retrieve(self, *args: Any, **kwargs: Any) -> list[Document]:
            return [document]

    class FailingRuntime:
        async def resolve(self, provider: str) -> Any:
            captured.setdefault("resolved", []).append(provider)
            raise ByokResolutionError("invalid_provider_credentials", provider)

    class OriginalScoreReranker:
        async def rerank(
            self,
            query: str,
            documents: list[Document],
        ) -> list[Document]:
            return documents

    def fake_create_reranker(strategy: Any, config: Any, llm_client: Any = None) -> Any:
        captured["strategy"] = strategy
        captured["llm_client"] = llm_client
        return OriginalScoreReranker()

    import tldw_Server_API.app.core.config as core_config

    monkeypatch.setattr(unified_pipeline_module, "MultiDatabaseRetriever", FakeRetriever)
    monkeypatch.setattr(unified_pipeline_module, "create_reranker", fake_create_reranker)
    monkeypatch.setattr(
        core_config,
        "load_and_log_configs",
        lambda: {
            "RAG_LLM_RERANKER_PROVIDER": "anthropic",
            "RAG_LLM_RERANKER_MODEL": "rerank-model",
        },
    )

    result = await unified_pipeline_module.unified_rag_pipeline(
        query="rank this",
        sources=["media_db"],
        enable_cache=False,
        enable_reranking=True,
        reranking_strategy=strategy_name,
        enable_generation=False,
        credential_runtime=FailingRuntime(),
    )

    assert captured["resolved"] == ["anthropic"]  # nosec B101
    assert captured["strategy"] is strategy_enum  # nosec B101
    assert captured["llm_client"] is None  # nosec B101
    assert [item["id"] for item in result.documents] == [document.id]  # nosec B101
    assert result.metadata["reranking"] == {  # nosec B101
        "failure_code": "invalid_provider_credentials",
        "verification_available": False,
    }


@pytest.mark.asyncio
async def test_unified_llm_reranker_partial_stream_failure_is_bounded_and_unmarked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _RecordingCredentialRuntime()
    captured: dict[str, Any] = {}
    sensitive_partial = "partial private provider text"
    document = Document(
        id="doc-partial-rerank",
        content="pipeline evidence",
        metadata={},
        source=DataSource.MEDIA_DB,
        score=0.8,
    )

    class FakeRetriever:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def retrieve(self, *args: Any, **kwargs: Any) -> list[Document]:
            return [document]

    def partial_response() -> Any:
        yield sensitive_partial
        raise SummaryProviderError(code="authentication", provider="anthropic")

    def fake_analyze(*args: Any, **kwargs: Any) -> Any:
        captured["analyze_kwargs"] = kwargs
        return partial_response()

    real_create_reranker = unified_pipeline_module.create_reranker

    def capturing_create_reranker(
        strategy: Any,
        config: Any,
        llm_client: Any = None,
    ) -> Any:
        reranker = real_create_reranker(strategy, config, llm_client=llm_client)
        real_rerank = reranker.rerank

        async def capture_scores(query: str, documents: list[Document]) -> Any:
            reranked = await real_rerank(query, documents)
            captured["rerank_scores"] = [item.rerank_score for item in reranked]
            return reranked

        reranker.rerank = capture_scores
        return reranker

    import tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib as sgl
    import tldw_Server_API.app.core.config as core_config

    monkeypatch.setattr(unified_pipeline_module, "MultiDatabaseRetriever", FakeRetriever)
    monkeypatch.setattr(
        unified_pipeline_module,
        "create_reranker",
        capturing_create_reranker,
    )
    monkeypatch.setattr(sgl, "analyze", fake_analyze)
    monkeypatch.setattr(
        core_config,
        "load_and_log_configs",
        lambda: {
            "RAG_LLM_RERANKER_PROVIDER": "anthropic",
            "RAG_LLM_RERANKER_MODEL": "rerank-model",
        },
    )

    result = await unified_pipeline_module.unified_rag_pipeline(
        query="rank this safely",
        sources=["media_db"],
        enable_cache=False,
        enable_reranking=True,
        reranking_strategy="llm_scoring",
        enable_generation=False,
        credential_runtime=runtime,
    )

    assert runtime.resolved == ["anthropic"]  # nosec B101
    assert runtime.marked == []  # nosec B101
    assert captured["rerank_scores"] == [document.score]  # nosec B101
    assert captured["analyze_kwargs"]["api_key"] == "runtime-only-key"  # nosec B101
    assert captured["analyze_kwargs"]["credentials_resolved"] is True  # nosec B101
    assert result.metadata["reranking"] == {  # nosec B101
        "failure_code": "provider_unavailable",
        "verification_available": False,
    }
    assert sensitive_partial not in str(result.metadata)  # nosec B101
    assert sensitive_partial not in str(result.errors)  # nosec B101


@pytest.mark.asyncio
async def test_unified_generation_consumes_task_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _RecordingCredentialRuntime()
    captured: dict[str, Any] = {}
    document = Document(
        id="doc-unified-generation",
        content="runtime-scoped evidence",
        metadata={},
        source=DataSource.MEDIA_DB,
        score=0.8,
    )

    class FakeRetriever:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def retrieve(self, *args: Any, **kwargs: Any) -> list[Document]:
            return [document]

    class FakeAnswerGenerator:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

        async def generate(self, **kwargs: Any) -> dict[str, Any]:
            return {"answer": "runtime-bound answer"}

    monkeypatch.setattr(unified_pipeline_module, "MultiDatabaseRetriever", FakeRetriever)
    monkeypatch.setattr(unified_pipeline_module, "AnswerGenerator", FakeAnswerGenerator)

    result = await unified_pipeline_module.unified_rag_pipeline(
        query="answer this",
        sources=["media_db"],
        enable_cache=False,
        enable_reranking=False,
        enable_generation=True,
        enable_pre_retrieval_clarification=False,
        generation_provider="anthropic",
        credential_runtime=runtime,
    )

    assert result.generated_answer == "runtime-bound answer"  # nosec B101
    assert captured["credential_runtime"] is runtime  # nosec B101


@pytest.mark.asyncio
async def test_post_verifier_repair_consumes_task_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _RecordingCredentialRuntime()
    captured: dict[str, Any] = {}
    calls = 0

    async def claims_runner(**kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        summary = (
            {"supported": 0, "refuted": 1, "nei": 0}
            if calls == 1
            else {"supported": 1, "refuted": 0, "nei": 0}
        )
        return {"claims": [], "summary": summary}

    class FakeAnswerGenerator:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

        async def generate(self, **kwargs: Any) -> dict[str, Any]:
            return {"answer": "repaired answer"}

    monkeypatch.setattr(verifier_module, "AnswerGenerator", FakeAnswerGenerator)

    outcome = await PostGenerationVerifier(
        claims_runner=claims_runner,
        unsupported_threshold=0.1,
        max_retries=1,
        credential_runtime=runtime,
    ).verify_and_maybe_fix(
        query="question",
        answer="draft answer",
        base_documents=[],
        generation_provider="anthropic",
    )

    assert outcome.fixed is True  # nosec B101
    assert outcome.new_answer == "repaired answer"  # nosec B101
    assert captured["credential_runtime"] is runtime  # nosec B101


@pytest.mark.asyncio
async def test_post_verifier_partial_provider_stream_is_unavailable_and_not_marked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _RecordingCredentialRuntime()
    captured: dict[str, Any] = {}

    def partial_response():
        yield "partial sensitive provider text"
        raise SummaryProviderError(code="authentication", provider="anthropic")

    def fake_analyze(*args: Any, **kwargs: Any):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return partial_response()

    class FakeClaimsEngine:
        def __init__(self, analyze_fn: Any) -> None:
            self.analyze_fn = analyze_fn

        async def run(self, **kwargs: Any) -> dict[str, Any]:
            response = self.analyze_fn(
                "anthropic",
                kwargs["answer"],
                "verify",
                streaming=True,
            )
            return {"claims": list(response), "summary": {}}

    import tldw_Server_API.app.core.Claims_Extraction.claims_engine as claims_engine_module
    import tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib as sgl

    monkeypatch.setattr(
        claims_engine_module,
        "_resolve_claims_llm_config",
        lambda: ("anthropic", None, 0.1),
    )
    monkeypatch.setattr(sgl, "analyze", fake_analyze)
    monkeypatch.setattr(verifier_module, "ClaimsEngine", FakeClaimsEngine)

    outcome = await PostGenerationVerifier(
        max_retries=0,
        credential_runtime=runtime,
    ).verify_and_maybe_fix(
        query="question",
        answer="answer",
        base_documents=[],
    )

    assert runtime.resolved == ["anthropic"]  # nosec B101
    assert runtime.marked == []  # nosec B101
    assert captured["args"][3] == "runtime-only-key"  # nosec B101
    assert captured["kwargs"]["app_config"] == {  # nosec B101
        "Anthropic": {"api_timeout": 12}
    }
    assert captured["kwargs"]["credentials_resolved"] is True  # nosec B101
    assert captured["kwargs"]["raise_on_error"] is True  # nosec B101
    assert outcome.reason == "verification_unavailable"  # nosec B101
    assert outcome.verification_available is False  # nosec B101
    assert outcome.failure_code == "provider_unavailable"  # nosec B101
    assert "partial sensitive" not in str(outcome)  # nosec B101


@pytest.mark.parametrize("pipeline_kind", ["unified", "agentic"])
@pytest.mark.asyncio
async def test_claims_multicall_failure_marks_prior_completed_call(
    monkeypatch: pytest.MonkeyPatch,
    pipeline_kind: str,
) -> None:
    runtime = _RecordingCredentialRuntime()
    analyze_calls = 0
    document = Document(
        id=f"doc-claims-{pipeline_kind}",
        content="Claims evidence.",
        metadata={},
        source=DataSource.MEDIA_DB,
        score=0.8,
    )

    class FakeRetriever:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def retrieve(self, *args: Any, **kwargs: Any) -> list[Document]:
            return [document]

    class FakeAnswerGenerator:
        def __init__(self, **kwargs: Any) -> None:
            pass

        async def generate(self, **kwargs: Any) -> dict[str, str]:
            return {"answer": "A claim-bearing answer."}

    class MultiCallClaimsEngine:
        def __init__(self, analyze_fn: Any) -> None:
            self.analyze_fn = analyze_fn

        async def run(self, **kwargs: Any) -> dict[str, Any]:
            self.analyze_fn("anthropic", "first claim prompt", None)
            self.analyze_fn("anthropic", "second claim prompt", None)
            raise AssertionError("second call should fail")

    def fake_analyze(*args: Any, **kwargs: Any) -> str:
        nonlocal analyze_calls
        analyze_calls += 1
        if analyze_calls == 1:
            return "clean completed response"
        raise SummaryProviderError(code="authentication", provider="anthropic")

    import tldw_Server_API.app.core.Claims_Extraction.claims_engine as claims_engine_module
    import tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib as sgl
    import tldw_Server_API.app.core.RAG.rag_service.claims as rag_claims_module

    monkeypatch.setattr(
        claims_engine_module,
        "_resolve_claims_llm_config",
        lambda: ("anthropic", None, 0.1),
    )
    monkeypatch.setattr(sgl, "analyze", fake_analyze)
    monkeypatch.setattr(rag_claims_module, "ClaimsEngine", MultiCallClaimsEngine, raising=False)

    if pipeline_kind == "unified":
        monkeypatch.setattr(unified_pipeline_module, "MultiDatabaseRetriever", FakeRetriever)
        monkeypatch.setattr(unified_pipeline_module, "AnswerGenerator", FakeAnswerGenerator)
        monkeypatch.setattr(unified_pipeline_module, "ClaimsEngine", MultiCallClaimsEngine)
        result = await unified_pipeline_module.unified_rag_pipeline(
            query="verify claims",
            sources=["media_db"],
            enable_cache=False,
            enable_reranking=False,
            enable_generation=True,
            enable_claims=True,
            enable_pre_retrieval_clarification=False,
            credential_runtime=runtime,
        )
    else:
        monkeypatch.setattr(agentic_chunker, "MultiDatabaseRetriever", FakeRetriever)
        monkeypatch.setattr(generation_module, "AnswerGenerator", FakeAnswerGenerator)
        result = await agentic_chunker.agentic_rag_pipeline(
            query="verify claims",
            sources=["media_db"],
            search_mode="fts",
            agentic=AgenticConfig(top_k_docs=1, enable_tools=False),
            enable_generation=True,
            enable_claims=True,
            credential_runtime=runtime,
        )

    assert runtime.marked == [runtime.handle]  # nosec B101
    assert result.metadata["claims"] == {  # nosec B101
        "failure_code": "provider_unavailable",
        "verification_available": False,
    }


@pytest.mark.asyncio
async def test_post_verifier_initial_multicall_failure_marks_prior_completed_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _RecordingCredentialRuntime()

    class MultiCallClaimsEngine:
        def __init__(self, analyze_fn: Any) -> None:
            self.analyze_fn = analyze_fn

        async def run(self, **kwargs: Any) -> dict[str, Any]:
            self.analyze_fn("anthropic", "first claim prompt", None)
            self.analyze_fn("anthropic", "second claim prompt", None)
            raise AssertionError("second call should fail")

    analyze_calls = 0

    def fake_analyze(*args: Any, **kwargs: Any) -> str:
        nonlocal analyze_calls
        analyze_calls += 1
        if analyze_calls == 1:
            return "clean completed response"
        raise SummaryProviderError(code="authentication", provider="anthropic")

    import tldw_Server_API.app.core.Claims_Extraction.claims_engine as claims_engine_module
    import tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib as sgl

    monkeypatch.setattr(
        claims_engine_module,
        "_resolve_claims_llm_config",
        lambda: ("anthropic", None, 0.1),
    )
    monkeypatch.setattr(sgl, "analyze", fake_analyze)
    monkeypatch.setattr(verifier_module, "ClaimsEngine", MultiCallClaimsEngine)

    outcome = await PostGenerationVerifier(
        max_retries=0,
        credential_runtime=runtime,
    ).verify_and_maybe_fix(query="question", answer="answer", base_documents=[])

    assert runtime.marked == [runtime.handle]  # nosec B101
    assert outcome.reason == "verification_unavailable"  # nosec B101
    assert outcome.failure_code == "provider_unavailable"  # nosec B101
    assert outcome.verification_available is False  # nosec B101


@pytest.mark.asyncio
async def test_post_verifier_recheck_multicall_failure_marks_prior_completed_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _RecordingCredentialRuntime()
    engine_instances = 0
    analyze_calls = 0

    class MultiCallClaimsEngine:
        def __init__(self, analyze_fn: Any) -> None:
            nonlocal engine_instances
            engine_instances += 1
            self.instance = engine_instances
            self.analyze_fn = analyze_fn

        async def run(self, **kwargs: Any) -> dict[str, Any]:
            self.analyze_fn("anthropic", "first claim prompt", None)
            if self.instance == 1:
                return {
                    "claims": [],
                    "summary": {"supported": 0, "refuted": 1, "nei": 0},
                }
            self.analyze_fn("anthropic", "second claim prompt", None)
            raise AssertionError("second recheck call should fail")

    class FakeAnswerGenerator:
        def __init__(self, **kwargs: Any) -> None:
            pass

        async def generate(self, **kwargs: Any) -> dict[str, str]:
            return {"answer": "repaired answer"}

    def fake_analyze(*args: Any, **kwargs: Any) -> str:
        nonlocal analyze_calls
        analyze_calls += 1
        if analyze_calls < 3:
            return "clean completed response"
        raise SummaryProviderError(code="authentication", provider="anthropic")

    import tldw_Server_API.app.core.Claims_Extraction.claims_engine as claims_engine_module
    import tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib as sgl

    monkeypatch.setattr(
        claims_engine_module,
        "_resolve_claims_llm_config",
        lambda: ("anthropic", None, 0.1),
    )
    monkeypatch.setattr(sgl, "analyze", fake_analyze)
    monkeypatch.setattr(verifier_module, "ClaimsEngine", MultiCallClaimsEngine)
    monkeypatch.setattr(verifier_module, "AnswerGenerator", FakeAnswerGenerator)

    outcome = await PostGenerationVerifier(
        max_retries=1,
        unsupported_threshold=0.1,
        credential_runtime=runtime,
    ).verify_and_maybe_fix(query="question", answer="answer", base_documents=[])

    assert runtime.marked == [runtime.handle, runtime.handle]  # nosec B101
    assert outcome.reason == "verification_unavailable"  # nosec B101
    assert outcome.failure_code == "provider_unavailable"  # nosec B101
    assert outcome.verification_available is False  # nosec B101


@pytest.mark.asyncio
async def test_real_claims_engine_propagates_extraction_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.Claims_Extraction.claims_engine import ClaimsEngine

    engine = ClaimsEngine(lambda *args, **kwargs: "unused")

    async def cancelled_extract(*args: Any, **kwargs: Any) -> Any:
        raise asyncio.CancelledError

    monkeypatch.setattr(engine.extractor_llm, "extract", cancelled_extract)

    with pytest.raises(asyncio.CancelledError):
        await engine._extract_claims_by_mode(
            answer="A factual answer that should be cancelled.",
            claim_extractor="llm",
            claims_max=5,
            budget=None,
            job_context=None,
        )


@pytest.mark.parametrize("stage", ["document", "groundedness", "utility"])
@pytest.mark.asyncio
async def test_runtime_bound_grader_timeout_uses_unavailable_native_fallback(
    stage: str,
) -> None:
    runtime = _RecordingCredentialRuntime()

    def slow_analyze(*args: Any, **kwargs: Any) -> str:
        time.sleep(0.05)
        return "{}"

    document = Document(
        id=f"doc-timeout-{stage}",
        content="Timeout evidence.",
        metadata={},
        source=DataSource.MEDIA_DB,
        score=0.8,
    )
    if stage == "document":
        result = await DocumentGrader(
            analyze_fn=slow_analyze,
            config=GradingConfig(provider="anthropic", timeout_seconds=0.01),
            credential_runtime=runtime,
        ).grade_document("query", document)
        assert result.method == "score_fallback"  # nosec B101
    elif stage == "groundedness":
        result = await FastGroundednessGrader(
            analyze_fn=slow_analyze,
            provider="anthropic",
            timeout_sec=0.01,
            credential_runtime=runtime,
        ).grade("query", "answer", [document])
        assert result.method == "error_fallback"  # nosec B101
    else:
        result = await UtilityGrader(
            analyze_fn=slow_analyze,
            provider="anthropic",
            timeout_sec=0.01,
            credential_runtime=runtime,
        ).grade("query", "answer")
        assert result.method == "error_fallback"  # nosec B101

    assert runtime.marked == []  # nosec B101
    assert result.metadata == {  # nosec B101
        "error": "provider_unavailable",
        "verification_available": False,
    }


@pytest.mark.asyncio
async def test_runtime_bound_reranker_timeout_preserves_scores_with_reduced_trust(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SlowBoundClient:
        credentials_resolved = True

        def analyze(self, prompt: str) -> str:
            time.sleep(0.05)
            return "0.9"

    monkeypatch.setenv("RAG_LLM_RERANK_TIMEOUT_SEC", "0.01")
    document = Document(
        id="doc-rerank-timeout",
        content="Evidence.",
        metadata={},
        source=DataSource.MEDIA_DB,
        score=0.8,
    )
    reranker = LLMReranker(
        RerankingConfig(
            strategy=RerankingStrategy.LLM_SCORING,
            top_k=1,
            batch_size=1,
        ),
        llm_client=SlowBoundClient(),
    )

    reranked = await reranker.rerank("query", [document])

    assert reranked[0].rerank_score == document.score  # nosec B101
    assert reranker.last_metadata == {  # nosec B101
        "degraded": True,
        "failure_code": "provider_unavailable",
        "verification_available": False,
    }


@pytest.mark.asyncio
async def test_faithfulness_propagate_errors_flag_preserves_legacy_default() -> None:
    class FailingLLM:
        async def generate(self, prompt: str) -> str:
            raise RuntimeError("private faithfulness failure")

    legacy = await FaithfulnessEvaluator(FailingLLM()).evaluate_detailed(
        "A factual answer.",
        "Supporting context.",
    )
    assert legacy.reasoning == "Claim extraction failed."  # nosec B101

    with pytest.raises(RuntimeError):
        await FaithfulnessEvaluator(
            FailingLLM(),
            propagate_errors=True,
        ).evaluate_detailed(
            "A factual answer.",
            "Supporting context.",
        )


@pytest.mark.asyncio
async def test_unified_runtime_bound_faithfulness_generic_failure_is_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _RecordingCredentialRuntime()
    sensitive = "private faithfulness failure"

    def failing_analyze(*args: Any, **kwargs: Any) -> str:
        raise RuntimeError(sensitive)

    import tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib as sgl

    monkeypatch.setattr(sgl, "analyze", failing_analyze)
    result = await _run_unified_bound_sgl_stage(
        monkeypatch,
        "faithfulness",
        runtime,
    )

    assert runtime.marked == []  # nosec B101
    assert result.metadata["faithfulness"] == {  # nosec B101
        "failure_code": "provider_unavailable",
        "verification_available": False,
    }
    assert sensitive not in str(result.metadata)  # nosec B101
    assert sensitive not in str(result.errors)  # nosec B101
