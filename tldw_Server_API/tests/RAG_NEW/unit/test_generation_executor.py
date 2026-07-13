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
