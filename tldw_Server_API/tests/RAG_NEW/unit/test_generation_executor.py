import asyncio
import gc
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any

import pytest
from loguru import logger

import tldw_Server_API.app.core.RAG.rag_service.advanced_reranking as advanced_reranking_module
import tldw_Server_API.app.core.RAG.rag_service.agentic_chunker as agentic_chunker
import tldw_Server_API.app.core.RAG.rag_service.document_grader as document_grader_module
import tldw_Server_API.app.core.RAG.rag_service.generation as generation_module
import tldw_Server_API.app.core.RAG.rag_service.post_generation_verifier as verifier_module
import tldw_Server_API.app.core.RAG.rag_service.quality_graders as quality_graders_module
import tldw_Server_API.app.core.RAG.rag_service.unified_pipeline as unified_pipeline_module
from tldw_Server_API.app.core.AuthNZ.byok_runtime import (
    ByokResolutionError,
    ByokResolutionStatus,
    ResolvedByokCredentials,
)
from tldw_Server_API.app.core.AuthNZ.provider_credential_runtime import (
    PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY,
    ProviderCallCredentials,
    ProviderCredentialRuntime,
)
from tldw_Server_API.app.core.Chat.bounded_daemon import BoundedDaemonPool
from tldw_Server_API.app.core.Chat.Chat_Deps import ChatAuthenticationError
from tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib import SummaryProviderError
from tldw_Server_API.app.core.RAG.rag_service.advanced_reranking import (
    LLMReranker,
    RerankingConfig,
    RerankingStrategy,
    ScoredDocument,
    TwoTierReranker,
)
from tldw_Server_API.app.core.RAG.rag_service.agentic_chunker import AgenticConfig
from tldw_Server_API.app.core.RAG.rag_service.document_grader import (
    DocumentGrader,
    GradingConfig,
)
from tldw_Server_API.app.core.RAG.rag_service.evidence_models import DerivedEvidence, RetrievedEvidence
from tldw_Server_API.app.core.RAG.rag_service.faithfulness import FaithfulnessEvaluator
from tldw_Server_API.app.core.RAG.rag_service.generation import AnswerGenerator
from tldw_Server_API.app.core.RAG.rag_service.generation_executor import execute_generation_phase
from tldw_Server_API.app.core.RAG.rag_service.post_generation_verifier import (
    PostGenerationVerifier,
)
from tldw_Server_API.app.core.RAG.rag_service.quality_graders import (
    FastGroundednessGrader,
    UtilityGrader,
)
from tldw_Server_API.app.core.RAG.rag_service.query_classifier import QueryClassification
from tldw_Server_API.app.core.RAG.rag_service.request_resolution import ResolvedRAGRequest
from tldw_Server_API.app.core.RAG.rag_service.result_model import RAGResult
from tldw_Server_API.app.core.RAG.rag_service.retrieval_plan import RetrievalPlan
from tldw_Server_API.app.core.RAG.rag_service.types import DataSource, Document

pytestmark = pytest.mark.unit


class _RecordingCredentialRuntime:
    def __init__(self) -> None:
        self.handle: ProviderCallCredentials | None = None
        self.resolved: list[str] = []
        self.resolved_models: list[str | None] = []
        self.marked: list[Any] = []

    async def resolve(self, provider: str, *, model: str | None = None) -> Any:
        self.resolved.append(provider)
        self.resolved_models.append(model)
        if self.handle is None or self.handle.provider != provider:
            async def resolver(
                normalized_provider: str,
                **_kwargs: Any,
            ) -> ResolvedByokCredentials:
                return ResolvedByokCredentials(
                    provider=normalized_provider,
                    api_key="runtime-only-key",
                    app_config={"Anthropic": {"api_timeout": 12}},
                    credential_fields={},
                    source="user",
                    allowlisted=True,
                    status=ByokResolutionStatus.RESOLVED,
                    auth_source="api_key",
                )

            issuer = ProviderCredentialRuntime(
                user_id=41,
                team_ids=(),
                org_ids=(),
                trusted_base_url_override=True,
                server_config_snapshot={},
                resolver=resolver,
            )
            try:
                self.handle = await issuer.resolve(provider, model=model)
            finally:
                await issuer.close()
        return self.handle

    async def mark_used(self, handle: Any) -> None:
        self.marked.append(handle)


def _grader_module(stage: str) -> Any:
    return (
        document_grader_module
        if stage == "document"
        else quality_graders_module
    )


def _valid_grader_response(stage: str) -> str:
    return {
        "document": (
            '{"is_relevant": true, "relevance_score": 0.9, '
            '"reasoning": "yes"}'
        ),
        "groundedness": (
            '{"is_grounded": true, "confidence": 0.9, '
            '"rationale": "yes"}'
        ),
        "utility": '{"utility_score": 5, "explanation": "yes"}',
    }[stage]


async def _run_runtime_bound_grader(
    stage: str,
    *,
    analyze: Any,
    runtime: Any,
    document: Document,
    timeout_seconds: float = 1.0,
) -> Any:
    if stage == "document":
        return await DocumentGrader(
            analyze_fn=analyze,
            config=GradingConfig(
                provider="anthropic",
                timeout_seconds=timeout_seconds,
            ),
            credential_runtime=runtime,
        ).grade_document("query", document)
    if stage == "groundedness":
        return await FastGroundednessGrader(
            analyze_fn=analyze,
            provider="anthropic",
            timeout_sec=timeout_seconds,
            credential_runtime=runtime,
        ).grade("query", "answer", [document])
    return await UtilityGrader(
        analyze_fn=analyze,
        provider="anthropic",
        timeout_sec=timeout_seconds,
        credential_runtime=runtime,
    ).grade("query", "answer")


def _install_explicit_chat_capture(
    monkeypatch: pytest.MonkeyPatch,
    response: Any,
) -> dict[str, Any]:
    """Capture a real chat-service boundary while forbidding server fallback."""
    from tldw_Server_API.app.core.Chat import chat_service
    from tldw_Server_API.app.core.LLM_Calls import adapter_utils

    def fail_server_fallback(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("server credential fallback must not run")

    captured: dict[str, Any] = {}
    monkeypatch.setattr(adapter_utils, "ensure_app_config", fail_server_fallback)
    monkeypatch.setattr(
        adapter_utils,
        "resolve_provider_api_key_from_config",
        fail_server_fallback,
    )

    async def fake_chat_call(**kwargs: Any) -> Any:
        provider, request, _internal = chat_service._build_adapter_request_from_chat_args(
            kwargs
        )
        captured["provider"] = provider
        captured["kwargs"] = kwargs
        captured["request"] = request
        return response() if callable(response) else response

    monkeypatch.setattr(chat_service, "perform_chat_api_call_async", fake_chat_call)
    return captured


def _install_blocking_sync_chat_adapter(
    monkeypatch: pytest.MonkeyPatch,
    response: Any,
) -> tuple[threading.Event, threading.Event]:
    """Install a real sync-only adapter whose completion is event-gated."""

    from tldw_Server_API.app.core.Chat import chat_service

    entered = threading.Event()
    release = threading.Event()

    class BlockingSyncAdapter:
        async def achat(self, _request: dict[str, Any]) -> None:
            raise NotImplementedError

        def chat(self, _request: dict[str, Any]) -> Any:
            entered.set()
            assert release.wait(timeout=1.0)  # nosec B101
            return response

    monkeypatch.setattr(
        chat_service,
        "_get_llm_registry",
        lambda: SimpleNamespace(get_adapter=lambda _provider: BlockingSyncAdapter()),
    )
    return entered, release


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


@pytest.mark.asyncio
async def test_unified_pipeline_propagates_runtime_to_direct_auxiliaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _RecordingCredentialRuntime()
    captured: dict[str, Any] = {"chain_runtimes": []}
    document = Document(
        id="doc-aux-runtime",
        content="Credential runtime evidence.",
        metadata={},
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
            captured["generation_runtime"] = kwargs.get("credential_runtime")

        async def generate(self, **kwargs: Any) -> dict[str, str]:
            return {"answer": "Credential-scoped answer."}

    async def fake_classifier(**kwargs: Any) -> QueryClassification:
        captured["classifier_runtime"] = kwargs.get("credential_runtime")
        classification_metadata = kwargs.get("stage_metadata")
        if isinstance(classification_metadata, dict):
            classification_metadata.update(
                failure_code="provider_unavailable",
                verification_available=False,
            )
        return QueryClassification(
            search_local_db=True,
            standalone_query="credential runtime",
            confidence=0.9,
        )

    class FakeAccumulator:
        def __init__(self, **kwargs: Any) -> None:
            captured["accumulator_runtime"] = kwargs.get("credential_runtime")

        async def accumulate(self, **kwargs: Any) -> Any:
            return SimpleNamespace(
                documents=kwargs["initial_results"],
                total_rounds=1,
                is_sufficient=True,
                sufficiency_reason="enough",
                metadata={"initial_docs": 1, "docs_added": 0},
            )

    async def fake_knowledge_strips(**kwargs: Any) -> Any:
        captured["knowledge_runtime"] = kwargs.get("credential_runtime")
        return kwargs["documents"], {
            "total_strips": 1,
            "relevant_strips": 1,
            "filtered_strips": 1,
            "avg_relevance": 0.9,
        }

    class FakeChainBuilder:
        def __init__(self, **kwargs: Any) -> None:
            captured["chain_runtimes"].append(kwargs.get("credential_runtime"))

        async def build_chains(self, **kwargs: Any) -> Any:
            return SimpleNamespace(
                chains=[],
                overall_confidence=0.0,
                multi_hop_detected=False,
                metadata={"total_nodes": 0, "total_claims": 0, "supported_claims": 0},
            )

    async def fake_suggestions(**kwargs: Any) -> list[str]:
        captured["suggestion_runtime"] = kwargs.get("credential_runtime")
        suggestion_metadata = kwargs.get("stage_metadata")
        if isinstance(suggestion_metadata, dict):
            suggestion_metadata.update(
                failure_code="provider_unavailable",
                verification_available=False,
            )
        return ["What should I inspect next?"]

    monkeypatch.setattr(unified_pipeline_module, "MultiDatabaseRetriever", FakeRetriever)
    monkeypatch.setattr(unified_pipeline_module, "AnswerGenerator", FakeAnswerGenerator)
    monkeypatch.setattr(unified_pipeline_module, "classify_and_reformulate", fake_classifier)
    monkeypatch.setattr(unified_pipeline_module, "EvidenceAccumulator", FakeAccumulator)
    monkeypatch.setattr(unified_pipeline_module, "process_knowledge_strips", fake_knowledge_strips)
    monkeypatch.setattr(unified_pipeline_module, "EvidenceChainBuilder", FakeChainBuilder)
    monkeypatch.setattr(unified_pipeline_module, "generate_suggestions", fake_suggestions)

    result = await unified_pipeline_module.unified_rag_pipeline(
        query="credential runtime",
        sources=["media_db"],
        enable_cache=False,
        enable_query_classification=True,
        enable_research_loop=False,
        enable_evidence_accumulation=True,
        enable_knowledge_strips=True,
        enable_evidence_chains=True,
        enable_suggestions=True,
        enable_generation=True,
        enable_reranking=False,
        enable_pre_retrieval_clarification=False,
        credential_runtime=runtime,
    )

    assert captured["classifier_runtime"] is runtime
    assert captured["accumulator_runtime"] is runtime
    assert captured["knowledge_runtime"] is runtime
    assert captured["chain_runtimes"] and all(
        item is runtime for item in captured["chain_runtimes"]
    )
    assert captured["suggestion_runtime"] is runtime
    assert result.metadata["query_classification"]["verification_available"] is False
    assert result.metadata["query_classification"]["failure_code"] == "provider_unavailable"
    assert result.metadata["suggestion_generation"] == {
        "failure_code": "provider_unavailable",
        "verification_available": False,
    }


@pytest.mark.asyncio
async def test_unified_dedicated_reformulation_copies_runtime_trust(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.RAG.rag_service import query_classifier

    runtime = _RecordingCredentialRuntime()

    class FakeRetriever:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def retrieve(self, *args: Any, **kwargs: Any) -> list[Document]:
            return []

    async def fake_reformulate(**kwargs: Any) -> str:
        metadata = kwargs.get("stage_metadata")
        if isinstance(metadata, dict):
            metadata.update(
                failure_code="provider_unavailable",
                verification_available=False,
            )
        return kwargs["query"]

    monkeypatch.setattr(unified_pipeline_module, "MultiDatabaseRetriever", FakeRetriever)
    monkeypatch.setattr(query_classifier, "reformulate_query", fake_reformulate)

    result = await unified_pipeline_module.unified_rag_pipeline(
        query="what about it?",
        chat_history=[{"role": "user", "content": "Explain credential runtimes."}],
        sources=["media_db"],
        enable_cache=False,
        enable_query_classification=False,
        enable_query_reformulation=True,
        enable_generation=False,
        enable_reranking=False,
        enable_pre_retrieval_clarification=False,
        credential_runtime=runtime,
    )

    assert result.metadata["query_reformulation"] == {
        "failure_code": "provider_unavailable",
        "verification_available": False,
    }


@pytest.mark.asyncio
async def test_unified_research_unavailability_restores_standard_local_retrieval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _RecordingCredentialRuntime()
    retrieved = Document(
        id="local-fallback",
        content="Local fallback evidence.",
        source=DataSource.MEDIA_DB,
        metadata={},
    )
    retriever_calls: list[str] = []

    class FakeRetriever:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def retrieve(self, query: str, *args: Any, **kwargs: Any) -> list[Document]:
            retriever_calls.append(query)
            return [retrieved]

    async def fake_classifier(**_kwargs: Any) -> QueryClassification:
        return QueryClassification(
            skip_search=False,
            search_local_db=False,
            search_web=False,
            search_academic=True,
            standalone_query="external-only research",
            confidence=0.9,
        )

    async def fake_research_loop(**_kwargs: Any) -> Any:
        return SimpleNamespace(
            all_results=[],
            total_iterations=1,
            total_results=0,
            total_duration_sec=0.0,
            completed=False,
            final_reasoning="provider unavailable",
            metadata={
                "action_dedup": {},
                "url_dedup": {},
                "provider_stage": {
                    "failure_code": "provider_unavailable",
                    "verification_available": False,
                },
            },
            steps=[],
        )

    monkeypatch.setattr(unified_pipeline_module, "MultiDatabaseRetriever", FakeRetriever)
    monkeypatch.setattr(unified_pipeline_module, "classify_and_reformulate", fake_classifier)
    monkeypatch.setattr(unified_pipeline_module, "create_default_registry", lambda **_kwargs: object())
    monkeypatch.setattr(unified_pipeline_module, "research_loop", fake_research_loop)

    result = await unified_pipeline_module.unified_rag_pipeline(
        query="external-only research",
        sources=["media_db"],
        enable_cache=False,
        enable_query_classification=True,
        enable_research_loop=True,
        enable_generation=False,
        enable_reranking=False,
        enable_pre_retrieval_clarification=False,
        credential_runtime=runtime,
    )

    assert retriever_calls == ["external-only research"]
    assert result.documents
    assert "retrieval_bypassed" not in result.metadata
    assert (
        result.metadata["classification_local_retrieval"]
        == "fallback_after_research_unavailable"
    )


def test_research_unavailability_preserves_existing_full_stack_skip() -> None:
    assert not unified_pipeline_module._should_restore_classification_local_retrieval(
        skip_retrieval_stack=True,
        skip_local_retrieval=True,
        classification_local_retrieval="disabled",
    )


@pytest.mark.asyncio
async def test_unified_citation_chain_shortcut_receives_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _RecordingCredentialRuntime()
    captured: dict[str, Any] = {}
    document = Document(
        id="citation-runtime",
        content="Citation evidence.",
        source=DataSource.MEDIA_DB,
        metadata={"title": "Citation Evidence"},
    )

    class FakeRetriever:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def retrieve(self, *args: Any, **kwargs: Any) -> list[Document]:
            return [document]

    class FakeCitationGenerator:
        async def generate_citations_with_chains(self, **kwargs: Any) -> Any:
            captured["citation_runtime"] = kwargs.get("credential_runtime")
            captured["citation_provider"] = kwargs.get("llm_provider")
            captured["citation_model"] = kwargs.get("llm_model")
            dual = SimpleNamespace(
                academic_citations=[],
                chunk_citations=[],
                inline_markers={},
                citation_map={},
            )
            chain_result = SimpleNamespace(
                chains=[],
                overall_confidence=0.0,
                multi_hop_detected=False,
                metadata={
                    "total_nodes": 0,
                    "failure_code": "provider_unavailable",
                    "verification_available": False,
                },
            )
            return dual, chain_result

    class UnexpectedChainBuilder:
        def __init__(self, **_kwargs: Any) -> None:
            raise AssertionError("citation chain result should skip the later builder")

    monkeypatch.setattr(unified_pipeline_module, "MultiDatabaseRetriever", FakeRetriever)
    monkeypatch.setattr(unified_pipeline_module, "CitationGenerator", FakeCitationGenerator)
    monkeypatch.setattr(unified_pipeline_module, "EvidenceChainBuilder", UnexpectedChainBuilder)

    result = await unified_pipeline_module.unified_rag_pipeline(
        query="citation runtime",
        sources=["media_db"],
        generation_provider="anthropic",
        generation_model="claude-test",
        enable_cache=False,
        enable_citations=True,
        enable_evidence_chains=True,
        enable_generation=False,
        enable_reranking=False,
        enable_pre_retrieval_clarification=False,
        credential_runtime=runtime,
    )

    assert captured["citation_runtime"] is runtime
    assert captured["citation_provider"] == "anthropic"
    assert captured["citation_model"] == "claude-test"
    assert result.metadata["evidence_chains"]["verification_available"] is False
    assert result.metadata["evidence_chains"]["failure_code"] == "provider_unavailable"


@pytest.mark.asyncio
async def test_unified_pipeline_propagates_runtime_to_research_registry_and_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _RecordingCredentialRuntime()
    captured: dict[str, Any] = {}

    async def fake_classifier(**kwargs: Any) -> QueryClassification:
        captured["classifier_runtime"] = kwargs.get("credential_runtime")
        return QueryClassification(
            search_local_db=True,
            standalone_query="credential runtime research",
            confidence=0.9,
        )

    def fake_registry(**kwargs: Any) -> object:
        captured["registry_runtime"] = kwargs.get("credential_runtime")
        return object()

    async def fake_research_loop(**kwargs: Any) -> Any:
        captured["research_runtime"] = kwargs.get("credential_runtime")
        return SimpleNamespace(
            all_results=[],
            total_iterations=1,
            total_results=0,
            total_duration_sec=0.0,
            completed=True,
            final_reasoning="done",
            metadata={"action_dedup": {}, "url_dedup": {}},
            steps=[],
        )

    monkeypatch.setattr(unified_pipeline_module, "classify_and_reformulate", fake_classifier)
    monkeypatch.setattr(unified_pipeline_module, "create_default_registry", fake_registry)
    monkeypatch.setattr(unified_pipeline_module, "research_loop", fake_research_loop)

    await unified_pipeline_module.unified_rag_pipeline(
        query="credential runtime research",
        sources=["media_db"],
        enable_cache=False,
        enable_query_classification=True,
        enable_research_loop=True,
        enable_generation=False,
        enable_reranking=False,
        enable_pre_retrieval_clarification=False,
        credential_runtime=runtime,
    )

    assert captured["classifier_runtime"] is runtime
    assert captured["registry_runtime"] is runtime
    assert captured["research_runtime"] is runtime


@pytest.mark.asyncio
async def test_unified_research_aggregates_bounded_media_provider_trust(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _RecordingCredentialRuntime()
    image = {"title": "Image", "url": "https://example.test/image"}
    video = {"title": "Video", "url": "https://example.test/video"}

    async def fake_classifier(**_kwargs: Any) -> QueryClassification:
        return QueryClassification(
            search_local_db=True,
            standalone_query="media trust",
            confidence=0.9,
        )

    async def fake_research_loop(**_kwargs: Any) -> Any:
        return SimpleNamespace(
            all_results=[],
            total_iterations=3,
            total_results=2,
            total_duration_sec=0.0,
            completed=True,
            final_reasoning="done",
            metadata={"action_dedup": {}, "url_dedup": {}},
            steps=[
                SimpleNamespace(
                    iteration=1,
                    action_name="image_search",
                    reasoning="find an image",
                    duration_sec=0.0,
                    output=SimpleNamespace(
                        success=True,
                        results=[image],
                        result_count=1,
                        metadata={
                            "type": "images",
                            "verification_available": True,
                        },
                    ),
                ),
                SimpleNamespace(
                    iteration=2,
                    action_name="video_search",
                    reasoning="find a video",
                    duration_sec=0.0,
                    output=SimpleNamespace(
                        success=True,
                        results=[video],
                        result_count=1,
                        metadata={
                            "type": "videos",
                            "verification_available": True,
                        },
                    ),
                ),
                SimpleNamespace(
                    iteration=3,
                    action_name="image_search",
                    reasoning="no matching image",
                    duration_sec=0.0,
                    output=SimpleNamespace(
                        success=True,
                        results=[],
                        result_count=0,
                        metadata={
                            "type": "images",
                            "failure_code": "provider_unavailable",
                            "verification_available": False,
                            "raw_detail": "secret-key /private/provider-store.db",
                        },
                    ),
                ),
            ],
        )

    monkeypatch.setattr(unified_pipeline_module, "classify_and_reformulate", fake_classifier)
    monkeypatch.setattr(unified_pipeline_module, "create_default_registry", lambda **_kwargs: object())
    monkeypatch.setattr(unified_pipeline_module, "research_loop", fake_research_loop)

    result = await unified_pipeline_module.unified_rag_pipeline(
        query="media trust",
        sources=["media_db"],
        enable_cache=False,
        enable_query_classification=True,
        enable_research_loop=True,
        enable_image_search=True,
        enable_video_search=True,
        enable_generation=False,
        enable_reranking=False,
        enable_pre_retrieval_clarification=False,
        credential_runtime=runtime,
    )

    assert result.metadata["images"] == [image]
    assert result.metadata["videos"] == [video]
    assert result.metadata["research"]["media_provider_stage"] == {
        "failure_code": "provider_unavailable",
        "verification_available": False,
    }
    assert "raw_detail" not in str(
        result.metadata["research"]["media_provider_stage"]
    )


async def _run_unified_bound_sgl_stage(
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    runtime: _RecordingCredentialRuntime | None,
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
    assert kwargs["provider_credentials"] is runtime.handle  # nosec B101
    assert kwargs["raise_on_error"] is True  # nosec B101
    assert runtime.marked == [runtime.handle]  # nosec B101


@pytest.mark.parametrize("stage", ["document", "groundedness", "utility"])
@pytest.mark.parametrize("runtime_bound", [False, True])
@pytest.mark.asyncio
async def test_grader_error_result_preserves_runtime_trust_state(
    stage: str,
    runtime_bound: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _RecordingCredentialRuntime() if runtime_bound else None
    release_count = 0

    class TrackingPool(BoundedDaemonPool):
        def _release_capacity(self) -> None:
            nonlocal release_count
            release_count += 1
            super()._release_capacity()

    pool = TrackingPool(capacity=1)
    monkeypatch.setattr(
        _grader_module(stage),
        "SYNC_ADAPTER_CALL_POOL",
        pool,
        raising=False,
    )

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
    elif stage == "groundedness":
        result = await FastGroundednessGrader(
            analyze_fn=error_result,
            provider="anthropic",
            credential_runtime=runtime,
        ).grade("query", "answer", [document])
    else:
        result = await UtilityGrader(
            analyze_fn=error_result,
            provider="anthropic",
            credential_runtime=runtime,
        ).grade("query", "answer")

    if runtime_bound:
        assert runtime is not None  # nosec B101
        assert runtime.marked == []  # nosec B101
        expected_method = "score_fallback" if stage == "document" else "heuristic"
        assert result.method == expected_method  # nosec B101
        assert result.metadata == {  # nosec B101
            "error": "provider_unavailable",
            "verification_available": False,
        }
    elif stage == "document":
        assert result.method == "llm_heuristic"  # nosec B101
        assert result.metadata == {}  # nosec B101
    else:
        assert result.method == "error_fallback"  # nosec B101
        assert result.metadata == {"parse_error": True}  # nosec B101
    assert "private-provider-detail" not in str(result)  # nosec B101
    assert pool.active_count == 0  # nosec B101
    assert release_count == 1  # nosec B101 - one admitted call, one release


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
    assert kwargs["provider_credentials"] is runtime.handle  # nosec B101
    assert kwargs["raise_on_error"] is True  # nosec B101
    assert runtime.resolved == [resolved_provider]  # nosec B101
    assert runtime.marked == [runtime.handle]  # nosec B101
    assert "verification_available" not in result.metadata.get(stage, {})  # nosec B101


@pytest.mark.parametrize(
    ("stage", "dispatch_response", "metadata_key"),
    [
        ("gap", '["follow up safely"]', "gap_analysis"),
        ("critique", "- no unsupported claims", "synthesis"),
        ("faithfulness", "[]", "faithfulness"),
    ],
)
@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_unified_bound_sgl_cancellation_drains_and_marks_completed_call(
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    dispatch_response: str,
    metadata_key: str,
) -> None:
    runtime = _RecordingCredentialRuntime()
    entered = threading.Event()
    release = threading.Event()

    def blocking_analyze(*_args: Any, **_kwargs: Any) -> str:
        entered.set()
        assert release.wait(timeout=1.0)  # nosec B101
        return dispatch_response

    import tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib as sgl

    monkeypatch.setattr(sgl, "analyze", blocking_analyze)
    task = asyncio.create_task(
        _run_unified_bound_sgl_stage(monkeypatch, stage, runtime)
    )
    try:
        assert await asyncio.to_thread(entered.wait, 1.0)  # nosec B101
        task.cancel()
        await asyncio.sleep(0.03)
        assert not task.done()  # nosec B101
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1.0)
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert runtime.marked == [runtime.handle]  # nosec B101


@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_unified_bound_reranker_cancellation_drains_and_marks_before_return(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _RecordingCredentialRuntime()
    pool = BoundedDaemonPool(capacity=1)
    entered = threading.Event()
    release = threading.Event()

    def blocking_analyze(*_args: Any, **_kwargs: Any) -> str:
        entered.set()
        assert release.wait(timeout=1.0)  # nosec B101
        return "0.8"

    import tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib as sgl

    monkeypatch.setattr(sgl, "analyze", blocking_analyze)
    monkeypatch.setattr(advanced_reranking_module, "SYNC_ADAPTER_CALL_POOL", pool)
    task = asyncio.create_task(
        _run_unified_bound_sgl_stage(monkeypatch, "reranker", runtime)
    )
    try:
        assert await asyncio.to_thread(entered.wait, 1.0)  # nosec B101
        task.cancel()
        await asyncio.sleep(0.03)
        assert task.done() is False
        assert pool.active_count == 1  # nosec B101
        assert runtime.marked == []  # nosec B101

        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1.0)
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)

    assert pool.active_count == 0  # nosec B101
    assert runtime.marked == [runtime.handle]  # nosec B101


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


@pytest.mark.parametrize("stage", ["gap", "reranker", "critique", "faithfulness"])
@pytest.mark.asyncio
async def test_unified_legacy_sgl_error_result_preserves_fallback(
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    calls = _stub_real_sgl_dispatch(monkeypatch, "Error: legacy provider failure")

    result = await _run_unified_bound_sgl_stage(monkeypatch, stage, None)

    assert len(calls) == 1  # nosec B101
    metadata_key = {
        "gap": "gap_analysis",
        "reranker": "reranking",
        "critique": "synthesis",
        "faithfulness": "faithfulness",
    }[stage]
    metadata = result.metadata.get(metadata_key, {})
    assert "failure_code" not in metadata  # nosec B101
    assert "verification_available" not in metadata  # nosec B101


@pytest.mark.parametrize("stage", ["gap", "reranker", "critique", "faithfulness"])
@pytest.mark.parametrize(
    ("first_chunk", "expected_marked"),
    [
        ("valid provider content", True),
        (": keepalive\n\n", False),
        ("Error: streamed provider failure", False),
        ({"choices": [{"delta": {"content": ""}}]}, False),
    ],
)
@pytest.mark.asyncio
async def test_unified_bound_sgl_stream_failure_marks_only_after_content(
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    first_chunk: Any,
    expected_marked: bool,
) -> None:
    runtime = _RecordingCredentialRuntime()
    sensitive = "private streamed provider detail"

    def partial_response() -> Any:
        yield first_chunk
        raise SummaryProviderError(code="authentication", provider="anthropic")

    import tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib as sgl

    monkeypatch.setattr(sgl, "analyze", lambda *args, **kwargs: partial_response())

    result = await _run_unified_bound_sgl_stage(monkeypatch, stage, runtime)

    assert runtime.marked == ([runtime.handle] if expected_marked else [])  # nosec B101
    metadata_key = {
        "gap": "gap_analysis",
        "reranker": "reranking",
        "critique": "synthesis",
        "faithfulness": "faithfulness",
    }[stage]
    assert result.metadata[metadata_key]["failure_code"] == "provider_unavailable"  # nosec B101
    assert result.metadata[metadata_key]["verification_available"] is False  # nosec B101
    assert sensitive not in str(result.metadata)  # nosec B101
    assert sensitive not in str(result.errors)  # nosec B101


@pytest.mark.parametrize("stage", ["gap", "reranker", "critique", "faithfulness"])
@pytest.mark.asyncio
async def test_unified_bound_sgl_clean_empty_stream_marks_once(
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    runtime = _RecordingCredentialRuntime()

    import tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib as sgl

    monkeypatch.setattr(sgl, "analyze", lambda *args, **kwargs: iter(()))

    await _run_unified_bound_sgl_stage(monkeypatch, stage, runtime)

    assert runtime.marked == [runtime.handle]  # nosec B101


@pytest.mark.parametrize(
    "chunks",
    [
        ("valid provider content", "Error: private provider detail"),
        ("valid provider content\nError: private provider detail",),
    ],
)
def test_bound_sgl_rejects_streamed_error_after_valid_content(
    chunks: tuple[str, ...],
) -> None:
    used: list[bool] = []

    with pytest.raises(SummaryProviderError):
        unified_pipeline_module._consume_bound_sgl_response(
            iter(chunks),
            "anthropic",
            on_content=lambda: used.append(True),
            fail_closed=True,
        )

    assert used == [True]  # nosec B101


def test_legacy_sgl_consumer_preserves_error_string_response() -> None:
    used: list[bool] = []

    response = unified_pipeline_module._consume_bound_sgl_response(
        iter(("valid provider content\nError: legacy provider detail",)),
        "anthropic",
        on_content=lambda: used.append(True),
        fail_closed=False,
    )

    assert response == "valid provider content\nError: legacy provider detail"  # nosec B101
    assert used == [True]  # nosec B101


@pytest.mark.parametrize(
    ("stage", "metadata_key"),
    [("gap", "gap_analysis"), ("critique", "synthesis")],
)
@pytest.mark.asyncio
async def test_unified_runtime_bound_auxiliary_generic_failure_is_bounded(
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    metadata_key: str,
) -> None:
    runtime = _RecordingCredentialRuntime()
    sensitive = "private generic auxiliary failure"

    import tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib as sgl

    def failing_analyze(*args: Any, **kwargs: Any) -> str:
        raise RuntimeError(sensitive)

    monkeypatch.setattr(sgl, "analyze", failing_analyze)

    result = await _run_unified_bound_sgl_stage(monkeypatch, stage, runtime)

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
    assert captured[PROVIDER_CALL_CREDENTIALS_CONTEXT_KEY] is runtime.handle  # nosec B101


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
        async def resolve(self, provider: str, *, model: str | None = None) -> Any:
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
        async def resolve(self, provider: str, *, model: str | None = None) -> Any:
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


@pytest.mark.asyncio
async def test_llm_reranker_missing_client_is_explicitly_degraded() -> None:
    document = Document(
        id="doc-rerank-missing-client",
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
        llm_client=None,
    )

    reranked = await reranker.rerank("query", [document])

    assert reranked[0].rerank_score == document.score  # nosec B101
    assert reranker.last_metadata == {  # nosec B101
        "degraded": True,
        "failure_code": "provider_unavailable",
        "verification_available": False,
    }


@pytest.mark.asyncio
async def test_unified_llm_reranker_runtime_failure_does_not_fail_over_provider(
    monkeypatch: pytest.MonkeyPatch,
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
        async def resolve(self, provider: str, *, model: str | None = None) -> Any:
            captured.setdefault("resolved", []).append(provider)
            captured.setdefault("models", []).append(model)
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
        reranking_strategy="llm_scoring",
        enable_generation=False,
        credential_runtime=FailingRuntime(),
    )

    assert captured["resolved"] == ["anthropic"]  # nosec B101
    assert captured["models"] == ["rerank-model"]  # nosec B101
    assert captured["strategy"] is RerankingStrategy.LLM_SCORING  # nosec B101
    assert captured["llm_client"] is None  # nosec B101
    assert [item["id"] for item in result.documents] == [document.id]  # nosec B101
    assert result.metadata["reranking"] == {  # nosec B101
        "degraded": True,
        "failure_code": "invalid_provider_credentials",
        "verification_available": False,
    }


@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_two_tier_credential_failures_concurrently_degrade_without_gating(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generation_calls: list[str] = []
    analyze_calls: list[str] = []
    resolving_failure_codes: set[str] = set()
    all_resolving = asyncio.Event()

    class FakeRetriever:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def retrieve(self, *args: Any, **kwargs: Any) -> list[Document]:
            query = str(args[0] if args else kwargs.get("query", "query"))
            return [
                Document(
                    id=f"doc-{query}",
                    content="pipeline evidence",
                    metadata={},
                    source=DataSource.MEDIA_DB,
                    score=0.2,
                )
            ]

    class FailingRuntime:
        def __init__(self, failure_code: str) -> None:
            self.failure_code = failure_code
            self.resolved: list[tuple[str, str | None]] = []
            self.marked_used: list[Any] = []

        async def resolve(self, provider: str, *, model: str | None = None) -> Any:
            self.resolved.append((provider, model))
            resolving_failure_codes.add(self.failure_code)
            if len(resolving_failure_codes) == 3:
                all_resolving.set()
            await asyncio.wait_for(all_resolving.wait(), timeout=1)
            raise ByokResolutionError(self.failure_code, provider)

        async def mark_used(self, handle: Any) -> None:
            self.marked_used.append(handle)

    class StaticCrossReranker:
        def __init__(self, config: RerankingConfig) -> None:
            self.config = config

        async def rerank(
            self,
            query: str,
            documents: list[Document],
            original_scores: list[float] | None = None,
        ) -> list[ScoredDocument]:
            return [
                ScoredDocument(
                    document=document,
                    original_score=document.score,
                    rerank_score=(0.05 if document.id == "sentinel:irrelevant" else 0.2),
                )
                for document in documents
            ]

    class FakeAnswerGenerator:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        async def generate(self, **kwargs: Any) -> dict[str, str]:
            query = str(kwargs.get("query", "query"))
            generation_calls.append(query)
            return {"answer": f"generated:{query}"}

    def create_real_two_tier(
        strategy: RerankingStrategy,
        config: RerankingConfig,
        llm_client: Any = None,
    ) -> TwoTierReranker:
        assert strategy is RerankingStrategy.TWO_TIER  # nosec B101
        assert llm_client is None  # nosec B101
        return TwoTierReranker(
            config,
            llm_client=llm_client,
            cross_reranker=StaticCrossReranker(config),
        )

    def fail_analyze(*_args: Any, **_kwargs: Any) -> str:
        analyze_calls.append("analyze")
        raise AssertionError("provider dispatch reached after credential failure")

    import tldw_Server_API.app.core.config as core_config
    import tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib as sgl

    monkeypatch.setenv("RAG_MIN_RELEVANCE_PROB", "0.99")
    monkeypatch.setenv("RAG_SENTINEL_MARGIN", "0.50")
    monkeypatch.setattr(unified_pipeline_module, "MultiDatabaseRetriever", FakeRetriever)
    monkeypatch.setattr(unified_pipeline_module, "create_reranker", create_real_two_tier)
    monkeypatch.setattr(unified_pipeline_module, "AnswerGenerator", FakeAnswerGenerator)
    monkeypatch.setattr(sgl, "analyze", fail_analyze)
    monkeypatch.setattr(
        core_config,
        "load_and_log_configs",
        lambda: {
            "RAG_LLM_RERANKER_PROVIDER": "anthropic",
            "RAG_LLM_RERANKER_MODEL": "rerank-model",
        },
    )

    runtimes = [
        FailingRuntime("invalid_provider_credentials"),
        FailingRuntime("credential_store_unavailable"),
        FailingRuntime("credential_scope_revoked"),
    ]

    async def run(index: int) -> Any:
        return await unified_pipeline_module.unified_rag_pipeline(
            query=f"rank-{index}",
            sources=["media_db"],
            metadata={
                "reranking_calibration": {
                    "gated": True,
                    "fused_score": 0.01,
                    "source": "untrusted_inbound",
                }
            },
            enable_cache=False,
            enable_reranking=True,
            reranking_strategy="two_tier",
            enable_learned_fusion=True,
            enable_generation=True,
            enable_pre_retrieval_clarification=False,
            credential_runtime=runtimes[index],
        )

    results = await asyncio.gather(run(0), run(1), run(2))

    assert resolving_failure_codes == {  # nosec B101
        "invalid_provider_credentials",
        "credential_store_unavailable",
        "credential_scope_revoked",
    }
    for index, (failure_code, result) in enumerate(
        zip(
            (
                "invalid_provider_credentials",
                "credential_store_unavailable",
                "credential_scope_revoked",
            ),
            results,
            strict=True,
        )
    ):
        assert runtimes[index].resolved == [("anthropic", "rerank-model")]  # nosec B101
        assert result.metadata["reranking"] == {  # nosec B101
            "degraded": True,
            "failure_code": failure_code,
            "verification_available": False,
        }
        assert runtimes[index].marked_used == []  # nosec B101
        assert "reranking_calibration" not in result.metadata  # nosec B101
        assert "generation_gate" not in result.metadata  # nosec B101
        assert result.generated_answer == f"generated:rank-{index}"  # nosec B101

    assert set(generation_calls) == {"rank-0", "rank-1", "rank-2"}  # nosec B101
    assert analyze_calls == []  # nosec B101


@pytest.mark.asyncio
async def test_unified_llm_reranker_partial_stream_failure_is_bounded_and_marked(
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

    import tldw_Server_API.app.core.config as core_config
    import tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib as sgl

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
    assert runtime.marked == [runtime.handle]  # nosec B101
    assert captured["rerank_scores"] == [document.score]  # nosec B101
    assert captured["analyze_kwargs"]["api_key"] == "runtime-only-key"  # nosec B101
    assert captured["analyze_kwargs"]["credentials_resolved"] is True  # nosec B101
    assert (  # nosec B101
        captured["analyze_kwargs"]["provider_credentials"] is runtime.handle
    )
    assert result.metadata["reranking"] == {  # nosec B101
        "degraded": True,
        "failure_code": "provider_unavailable",
        "verification_available": False,
    }
    assert sensitive_partial not in str(result.metadata)  # nosec B101
    assert sensitive_partial not in str(result.errors)  # nosec B101


@pytest.mark.asyncio
async def test_unified_llm_reranker_marks_prior_success_before_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _RecordingCredentialRuntime()
    analyze_calls = 0
    documents = [
        Document(
            id=f"doc-rerank-cancel-{index}",
            content=f"Evidence {index}.",
            metadata={},
            source=DataSource.MEDIA_DB,
            score=0.8 - index * 0.1,
        )
        for index in range(2)
    ]

    class FakeRetriever:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def retrieve(self, *args: Any, **kwargs: Any) -> list[Document]:
            return documents

    class CancellingReranker:
        def __init__(self, llm_client: Any) -> None:
            self.llm_client = llm_client

        async def rerank(self, query: str, documents: list[Document]) -> list[Document]:
            self.llm_client.analyze("first rerank prompt")
            self.llm_client.analyze("second rerank prompt")
            raise AssertionError("second provider call should cancel")

    def fake_create_reranker(
        strategy: Any,
        config: Any,
        llm_client: Any = None,
    ) -> CancellingReranker:
        return CancellingReranker(llm_client)

    def fake_analyze(*args: Any, **kwargs: Any) -> str:
        nonlocal analyze_calls
        analyze_calls += 1
        if analyze_calls == 1:
            return "0.8"
        raise asyncio.CancelledError

    import tldw_Server_API.app.core.config as core_config
    import tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib as sgl

    monkeypatch.setattr(unified_pipeline_module, "MultiDatabaseRetriever", FakeRetriever)
    monkeypatch.setattr(unified_pipeline_module, "create_reranker", fake_create_reranker)
    monkeypatch.setattr(sgl, "analyze", fake_analyze)
    monkeypatch.setattr(
        core_config,
        "load_and_log_configs",
        lambda: {
            "RAG_LLM_RERANKER_PROVIDER": "anthropic",
            "RAG_LLM_RERANKER_MODEL": "rerank-model",
        },
    )

    with pytest.raises(asyncio.CancelledError):
        await unified_pipeline_module.unified_rag_pipeline(
            query="cancel reranking",
            sources=["media_db"],
            enable_cache=False,
            enable_reranking=True,
            reranking_strategy="llm_scoring",
            enable_generation=False,
            credential_runtime=runtime,
        )

    assert analyze_calls == 2  # nosec B101
    assert runtime.marked == [runtime.handle]  # nosec B101


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
async def test_post_verifier_partial_provider_stream_is_unavailable_and_unmarked(
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
    assert captured["kwargs"]["provider_credentials"] is runtime.handle  # nosec B101
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
    analyze_handles: list[Any] = []
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
        analyze_handles.append(kwargs["provider_credentials"])
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
    assert analyze_handles == [runtime.handle, runtime.handle]  # nosec B101
    assert result.metadata["claims"] == {  # nosec B101
        "failure_code": "provider_unavailable",
        "verification_available": False,
    }
    if pipeline_kind == "agentic":
        assert result.metadata["post_verification"] == {  # nosec B101
            "unsupported_ratio": 0.0,
            "total_claims": 0,
            "unsupported_count": 0,
            "fixed": False,
            "reason": "verification_unavailable",
            "verification_available": False,
            "failure_code": "provider_unavailable",
        }


@pytest.mark.asyncio
async def test_agentic_claims_creates_only_the_awaited_operation(
    monkeypatch: pytest.MonkeyPatch,
    recwarn: pytest.WarningsRecorder,
) -> None:
    """Agentic claims must not abandon a duplicate coroutine object."""

    runtime = _RecordingCredentialRuntime()
    document = Document(
        id="doc-agentic-claims-operation",
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

    class FakeClaimsEngine:
        def __init__(self, analyze_fn: Any) -> None:
            self.analyze_fn = analyze_fn

        async def run(self, **kwargs: Any) -> dict[str, Any]:
            return {"claims": [], "summary": {}}

    import tldw_Server_API.app.core.RAG.rag_service.claims as rag_claims_module

    monkeypatch.setattr(agentic_chunker, "MultiDatabaseRetriever", FakeRetriever)
    monkeypatch.setattr(generation_module, "AnswerGenerator", FakeAnswerGenerator)
    monkeypatch.setattr(rag_claims_module, "ClaimsEngine", FakeClaimsEngine)

    await agentic_chunker.agentic_rag_pipeline(
        query="verify claims",
        sources=["media_db"],
        search_mode="fts",
        agentic=AgenticConfig(top_k_docs=1, enable_tools=False),
        enable_generation=True,
        enable_claims=True,
        credential_runtime=runtime,
    )
    gc.collect()

    assert not [
        warning
        for warning in recwarn
        if issubclass(warning.category, RuntimeWarning)
        and "was never awaited" in str(warning.message)
    ]


@pytest.mark.parametrize("pipeline_kind", ["unified", "agentic"])
@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_pipeline_cancellation_drains_claims_and_marks_completed_call(
    monkeypatch: pytest.MonkeyPatch,
    pipeline_kind: str,
) -> None:
    runtime = _RecordingCredentialRuntime()
    entered = threading.Event()
    release = threading.Event()
    document = Document(
        id=f"doc-cancel-claims-{pipeline_kind}",
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
        def __init__(self, **_kwargs: Any) -> None:
            pass

        async def generate(self, **_kwargs: Any) -> dict[str, str]:
            return {"answer": "A claim-bearing answer."}

    class BlockingClaimsEngine:
        def __init__(self, analyze_fn: Any) -> None:
            self.analyze_fn = analyze_fn

        async def run(self, **_kwargs: Any) -> dict[str, Any]:
            def call_analyzer() -> dict[str, Any]:
                self.analyze_fn("anthropic", "claim prompt", None)
                return {"claims": [], "summary": {}}

            return await asyncio.to_thread(call_analyzer)

    def blocking_analyze(*_args: Any, **_kwargs: Any) -> str:
        entered.set()
        assert release.wait(timeout=1.0)  # nosec B101
        return "completed claims response"

    import tldw_Server_API.app.core.Claims_Extraction.claims_engine as claims_engine_module
    import tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib as sgl
    import tldw_Server_API.app.core.RAG.rag_service.claims as rag_claims_module

    monkeypatch.setattr(
        claims_engine_module,
        "_resolve_claims_llm_config",
        lambda: ("anthropic", None, 0.1),
    )
    monkeypatch.setattr(sgl, "analyze", blocking_analyze)
    monkeypatch.setattr(
        rag_claims_module,
        "ClaimsEngine",
        BlockingClaimsEngine,
        raising=False,
    )

    if pipeline_kind == "unified":
        monkeypatch.setattr(unified_pipeline_module, "MultiDatabaseRetriever", FakeRetriever)
        monkeypatch.setattr(unified_pipeline_module, "AnswerGenerator", FakeAnswerGenerator)
        monkeypatch.setattr(unified_pipeline_module, "ClaimsEngine", BlockingClaimsEngine)
        operation = unified_pipeline_module.unified_rag_pipeline(
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
        operation = agentic_chunker.agentic_rag_pipeline(
            query="verify claims",
            sources=["media_db"],
            search_mode="fts",
            agentic=AgenticConfig(top_k_docs=1, enable_tools=False),
            enable_generation=True,
            enable_claims=True,
            credential_runtime=runtime,
        )

    task = asyncio.create_task(operation)
    try:
        assert await asyncio.to_thread(entered.wait, 1.0)  # nosec B101
        task.cancel()
        await asyncio.sleep(0.03)
        assert not task.done()  # nosec B101
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1.0)
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert runtime.marked == [runtime.handle]  # nosec B101


@pytest.mark.parametrize("failure_kind", ["provider", "cancel", "partial"])
@pytest.mark.asyncio
async def test_preextracted_claims_mark_prior_completed_provider_call(
    monkeypatch: pytest.MonkeyPatch,
    failure_kind: str,
) -> None:
    runtime = _RecordingCredentialRuntime()
    analyze_calls = 0
    document = Document(
        id="doc-preextracted-claims",
        content="Stored claims evidence.",
        metadata={"media_id": 7},
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

    class FakeManagedDatabase:
        def __enter__(self) -> "FakeManagedDatabase":
            return self

        def __exit__(self, *args: Any) -> bool:
            return False

        def execute_query(self, *args: Any, **kwargs: Any) -> Any:
            return SimpleNamespace(
                fetchall=lambda: [("first stored claim",), ("second stored claim",)]
            )

    class DirectVerifierClaimsEngine:
        def __init__(self, analyze_fn: Any) -> None:
            self.analyze_fn = analyze_fn
            self.verifier = SimpleNamespace(verify=self._verify)

        async def _verify(self, **kwargs: Any) -> Any:
            self.analyze_fn("anthropic", kwargs["claim"].text, None)
            return SimpleNamespace(label="supported")

        async def run(self, **kwargs: Any) -> dict[str, Any]:
            raise AssertionError("pre-extracted claims must bypass engine.run")

    def fake_analyze(*args: Any, **kwargs: Any) -> Any:
        nonlocal analyze_calls
        analyze_calls += 1
        if failure_kind == "partial":
            def partial_response() -> Any:
                yield "partial private provider output"
                raise SummaryProviderError(code="authentication", provider="anthropic")

            return partial_response()
        if analyze_calls == 1:
            return "completed provider response"
        if failure_kind == "cancel":
            raise asyncio.CancelledError
        raise SummaryProviderError(code="authentication", provider="anthropic")

    import tldw_Server_API.app.core.Claims_Extraction.claims_engine as claims_engine_module
    import tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib as sgl

    monkeypatch.setattr(
        claims_engine_module,
        "_resolve_claims_llm_config",
        lambda: ("anthropic", None, 0.1),
    )
    monkeypatch.setattr(sgl, "analyze", fake_analyze)
    monkeypatch.setattr(unified_pipeline_module, "MultiDatabaseRetriever", FakeRetriever)
    monkeypatch.setattr(unified_pipeline_module, "AnswerGenerator", FakeAnswerGenerator)
    monkeypatch.setattr(unified_pipeline_module, "ClaimsEngine", DirectVerifierClaimsEngine)
    monkeypatch.setattr(
        unified_pipeline_module,
        "managed_media_database",
        lambda **kwargs: FakeManagedDatabase(),
    )

    pipeline_call = unified_pipeline_module.unified_rag_pipeline(
        query="verify stored claims",
        sources=["media_db"],
        enable_cache=False,
        enable_reranking=False,
        enable_generation=True,
        enable_claims=True,
        enable_pre_retrieval_clarification=False,
        media_db_path="/tmp/media.db",
        credential_runtime=runtime,
    )
    if failure_kind == "cancel":
        with pytest.raises(asyncio.CancelledError):
            await pipeline_call
    else:
        result = await pipeline_call
        assert result.metadata["claims"] == {  # nosec B101
            "failure_code": "provider_unavailable",
            "verification_available": False,
        }

    expected_calls = 1 if failure_kind == "partial" else 2
    expected_marks = [runtime.handle]
    assert analyze_calls == expected_calls  # nosec B101
    assert runtime.marked == expected_marks  # nosec B101


@pytest.mark.parametrize("runtime_bound", [False, True])
@pytest.mark.asyncio
async def test_preextracted_claims_generic_failure_sanitizes_runtime_log(
    monkeypatch: pytest.MonkeyPatch,
    runtime_bound: bool,
) -> None:
    runtime = _RecordingCredentialRuntime() if runtime_bound else None
    sensitive = "private pre-extracted verifier failure"
    document = Document(
        id="doc-preextracted-log",
        content="Stored claims evidence.",
        metadata={"media_id": 7},
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

    class FakeManagedDatabase:
        def __enter__(self) -> "FakeManagedDatabase":
            return self

        def __exit__(self, *args: Any) -> bool:
            return False

        def execute_query(self, *args: Any, **kwargs: Any) -> Any:
            return SimpleNamespace(fetchall=lambda: [("stored claim",)])

    class FallbackClaimsEngine:
        def __init__(self, analyze_fn: Any) -> None:
            self.verifier = SimpleNamespace(verify=self._verify)

        async def _verify(self, **kwargs: Any) -> Any:
            raise RuntimeError(sensitive)

        async def run(self, **kwargs: Any) -> dict[str, Any]:
            return {
                "claims": [{"text": "fallback claim"}],
                "summary": {"supported": 0, "refuted": 0, "nei": 1},
            }

    import tldw_Server_API.app.core.Claims_Extraction.claims_engine as claims_engine_module

    monkeypatch.setattr(
        claims_engine_module,
        "_resolve_claims_llm_config",
        lambda: ("anthropic", None, 0.1),
    )
    monkeypatch.setattr(unified_pipeline_module, "MultiDatabaseRetriever", FakeRetriever)
    monkeypatch.setattr(unified_pipeline_module, "AnswerGenerator", FakeAnswerGenerator)
    monkeypatch.setattr(unified_pipeline_module, "ClaimsEngine", FallbackClaimsEngine)
    monkeypatch.setattr(
        unified_pipeline_module,
        "managed_media_database",
        lambda **kwargs: FakeManagedDatabase(),
    )

    messages: list[str] = []
    sink_id = logger.add(
        lambda message: messages.append(str(message)),
        level="DEBUG",
        format="{message}",
    )
    try:
        result = await unified_pipeline_module.unified_rag_pipeline(
            query="verify stored claims",
            sources=["media_db"],
            enable_cache=False,
            enable_reranking=False,
            enable_generation=True,
            enable_claims=True,
            enable_pre_retrieval_clarification=False,
            media_db_path="/tmp/media.db",
            credential_runtime=runtime,
        )
    finally:
        logger.remove(sink_id)

    log_text = "".join(messages)
    assert result.metadata["claims"] == [{"text": "fallback claim"}]  # nosec B101
    assert sensitive not in str(result.metadata)  # nosec B101
    assert sensitive not in str(result.errors)  # nosec B101
    if runtime_bound:
        assert sensitive not in log_text  # nosec B101
    else:
        assert sensitive in log_text  # nosec B101


@pytest.mark.parametrize("pipeline_kind", ["unified", "agentic", "post"])
@pytest.mark.parametrize(
    "response_kind",
    [
        "error_string",
        "streamed_error_string",
        "content_then_error_string",
        "same_chunk_content_then_error_string",
        "empty_delta_then_partial_stream",
        "partial_stream",
    ],
)
@pytest.mark.asyncio
async def test_claims_provider_failure_is_unavailable_with_accurate_use(
    monkeypatch: pytest.MonkeyPatch,
    pipeline_kind: str,
    response_kind: str,
) -> None:
    runtime = _RecordingCredentialRuntime()
    sensitive = "private legacy provider detail"
    document = Document(
        id=f"doc-legacy-error-{pipeline_kind}",
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

    class LegacyErrorClaimsEngine:
        def __init__(self, analyze_fn: Any) -> None:
            self.analyze_fn = analyze_fn

        async def run(self, **kwargs: Any) -> dict[str, Any]:
            response = self.analyze_fn("anthropic", "claims prompt", None)
            return {
                "claims": [{"text": response}],
                "summary": {"supported": 1, "refuted": 0, "nei": 0},
            }

    def fake_analyze(*args: Any, **kwargs: Any) -> Any:
        if response_kind == "error_string":
            return f"Error: provider failed {sensitive}"

        if response_kind == "streamed_error_string":
            return iter((f"Error: provider failed {sensitive}",))

        if response_kind == "content_then_error_string":
            return iter(
                (
                    "valid provider content",
                    f"Error: provider failed {sensitive}",
                )
            )

        if response_kind == "same_chunk_content_then_error_string":
            return iter(
                (f"valid provider content\nError: provider failed {sensitive}",)
            )

        if response_kind == "empty_delta_then_partial_stream":
            def empty_partial_response() -> Any:
                yield {"choices": [{"delta": {"content": ""}}]}
                raise SummaryProviderError(code="authentication", provider="anthropic")

            return empty_partial_response()

        def partial_response() -> Any:
            yield "valid provider content"
            raise SummaryProviderError(code="authentication", provider="anthropic")

        return partial_response()

    import tldw_Server_API.app.core.Claims_Extraction.claims_engine as claims_engine_module
    import tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib as sgl
    import tldw_Server_API.app.core.RAG.rag_service.claims as rag_claims_module

    monkeypatch.setattr(
        claims_engine_module,
        "_resolve_claims_llm_config",
        lambda: ("anthropic", None, 0.1),
    )
    monkeypatch.setattr(sgl, "analyze", fake_analyze)
    monkeypatch.setattr(rag_claims_module, "ClaimsEngine", LegacyErrorClaimsEngine, raising=False)

    if pipeline_kind == "unified":
        monkeypatch.setattr(unified_pipeline_module, "MultiDatabaseRetriever", FakeRetriever)
        monkeypatch.setattr(unified_pipeline_module, "AnswerGenerator", FakeAnswerGenerator)
        monkeypatch.setattr(unified_pipeline_module, "ClaimsEngine", LegacyErrorClaimsEngine)
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
        failure = result.metadata["claims"]
        exposed = f"{result.metadata} {result.errors}"
    elif pipeline_kind == "agentic":
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
        failure = result.metadata["claims"]
        exposed = f"{result.metadata} {result.errors}"
    else:
        monkeypatch.setattr(verifier_module, "ClaimsEngine", LegacyErrorClaimsEngine)
        outcome = await PostGenerationVerifier(
            max_retries=0,
            credential_runtime=runtime,
        ).verify_and_maybe_fix(
            query="verify claims",
            answer="A claim-bearing answer.",
            base_documents=[document],
        )
        failure = {
            "failure_code": outcome.failure_code,
            "verification_available": outcome.verification_available,
        }
        exposed = str(outcome)

    assert failure == {  # nosec B101
        "failure_code": "provider_unavailable",
        "verification_available": False,
    }
    expected_marks = (
        [runtime.handle]
        if pipeline_kind != "post"
        and response_kind in {
            "content_then_error_string",
            "same_chunk_content_then_error_string",
            "partial_stream",
        }
        else []
    )
    assert runtime.marked == expected_marks  # nosec B101
    assert sensitive not in exposed  # nosec B101


@pytest.mark.asyncio
async def test_unified_runtime_bound_claims_generic_failure_is_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _RecordingCredentialRuntime()
    sensitive = "private generic claims failure"
    document = Document(
        id="doc-generic-claims",
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

    class FailingClaimsEngine:
        def __init__(self, analyze_fn: Any) -> None:
            pass

        async def run(self, **kwargs: Any) -> dict[str, Any]:
            raise RuntimeError(sensitive)

    import tldw_Server_API.app.core.Claims_Extraction.claims_engine as claims_engine_module

    monkeypatch.setattr(
        claims_engine_module,
        "_resolve_claims_llm_config",
        lambda: ("anthropic", None, 0.1),
    )
    monkeypatch.setattr(unified_pipeline_module, "MultiDatabaseRetriever", FakeRetriever)
    monkeypatch.setattr(unified_pipeline_module, "AnswerGenerator", FakeAnswerGenerator)
    monkeypatch.setattr(unified_pipeline_module, "ClaimsEngine", FailingClaimsEngine)

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

    assert result.metadata["claims"] == {  # nosec B101
        "failure_code": "provider_unavailable",
        "verification_available": False,
    }
    assert sensitive not in str(result.metadata)  # nosec B101
    assert sensitive not in str(result.errors)  # nosec B101


async def _run_agentic_claims_scenario(
    monkeypatch: pytest.MonkeyPatch,
    runtime: _RecordingCredentialRuntime | None,
    claims_engine: type[Any],
    post_verifier: type[Any],
) -> Any:
    document = Document(
        id="doc-agentic-failure",
        content="Agentic claims evidence.",
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

    import tldw_Server_API.app.core.Claims_Extraction.claims_engine as claims_engine_module
    import tldw_Server_API.app.core.RAG.rag_service.claims as rag_claims_module

    monkeypatch.setattr(
        claims_engine_module,
        "_resolve_claims_llm_config",
        lambda: ("anthropic", None, 0.1),
    )
    monkeypatch.setattr(agentic_chunker, "MultiDatabaseRetriever", FakeRetriever)
    monkeypatch.setattr(generation_module, "AnswerGenerator", FakeAnswerGenerator)
    monkeypatch.setattr(rag_claims_module, "ClaimsEngine", claims_engine, raising=False)
    monkeypatch.setattr(verifier_module, "PostGenerationVerifier", post_verifier)

    return await agentic_chunker.agentic_rag_pipeline(
        query="verify agentic claims",
        sources=["media_db"],
        search_mode="fts",
        agentic=AgenticConfig(top_k_docs=1, enable_tools=False),
        enable_generation=True,
        enable_claims=True,
        enable_citations=False,
        enable_numeric_fidelity=False,
        credential_runtime=runtime,
    )


@pytest.mark.parametrize("runtime_bound", [False, True])
@pytest.mark.asyncio
async def test_agentic_claims_generic_failure_preserves_runtime_trust_state(
    monkeypatch: pytest.MonkeyPatch,
    runtime_bound: bool,
) -> None:
    runtime = _RecordingCredentialRuntime() if runtime_bound else None
    sensitive = "private generic agentic claims failure"

    class FailingClaimsEngine:
        def __init__(self, analyze_fn: Any) -> None:
            pass

        async def run(self, **kwargs: Any) -> dict[str, Any]:
            raise RuntimeError(sensitive)

    class NoopPostVerifier:
        def __init__(self, **kwargs: Any) -> None:
            pass

        async def verify_and_maybe_fix(self, **kwargs: Any) -> Any:
            return SimpleNamespace(
                unsupported_ratio=0.0,
                total_claims=0,
                unsupported_count=0,
                fixed=False,
                reason="",
                verification_available=True,
                failure_code=None,
            )

    result = await _run_agentic_claims_scenario(
        monkeypatch,
        runtime,
        FailingClaimsEngine,
        NoopPostVerifier,
    )

    if runtime_bound:
        assert result.metadata["claims"] == {  # nosec B101
            "failure_code": "provider_unavailable",
            "verification_available": False,
        }
    else:
        assert "claims" not in result.metadata  # nosec B101
    assert sensitive not in str(result.metadata)  # nosec B101
    assert sensitive not in str(result.errors)  # nosec B101


@pytest.mark.parametrize("runtime_bound", [False, True])
@pytest.mark.asyncio
async def test_agentic_post_verification_generic_failure_preserves_runtime_trust_state(
    monkeypatch: pytest.MonkeyPatch,
    runtime_bound: bool,
) -> None:
    runtime = _RecordingCredentialRuntime() if runtime_bound else None
    sensitive = "private generic agentic NLI failure"

    class SuccessfulClaimsEngine:
        def __init__(self, analyze_fn: Any) -> None:
            pass

        async def run(self, **kwargs: Any) -> dict[str, Any]:
            return {"claims": [], "summary": {}}

    class FailingPostVerifier:
        def __init__(self, **kwargs: Any) -> None:
            pass

        async def verify_and_maybe_fix(self, **kwargs: Any) -> Any:
            raise RuntimeError(sensitive)

    result = await _run_agentic_claims_scenario(
        monkeypatch,
        runtime,
        SuccessfulClaimsEngine,
        FailingPostVerifier,
    )

    if runtime_bound:
        assert result.metadata["post_verification"] == {  # nosec B101
            "failure_code": "provider_unavailable",
            "verification_available": False,
        }
        assert sensitive not in str(result.errors)  # nosec B101
    else:
        assert "post_verification" not in result.metadata  # nosec B101
        assert any(sensitive in error for error in result.errors)  # nosec B101
    assert sensitive not in str(result.metadata)  # nosec B101


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
            return '{"claims": [{"text": "completed first claim"}]}'
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
async def test_post_verifier_initial_generic_failure_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _RecordingCredentialRuntime()
    sensitive = "private initial verification failure"

    class FailingClaimsEngine:
        def __init__(self, analyze_fn: Any) -> None:
            pass

        async def run(self, **kwargs: Any) -> dict[str, Any]:
            raise RuntimeError(sensitive)

    import tldw_Server_API.app.core.Claims_Extraction.claims_engine as claims_engine_module

    monkeypatch.setattr(
        claims_engine_module,
        "_resolve_claims_llm_config",
        lambda: ("anthropic", None, 0.1),
    )
    monkeypatch.setattr(verifier_module, "ClaimsEngine", FailingClaimsEngine)

    outcome = await PostGenerationVerifier(
        max_retries=0,
        credential_runtime=runtime,
    ).verify_and_maybe_fix(query="question", answer="answer", base_documents=[])

    assert outcome.reason == "verification_unavailable"  # nosec B101
    assert outcome.failure_code == "provider_unavailable"  # nosec B101
    assert outcome.verification_available is False  # nosec B101
    assert sensitive not in str(outcome)  # nosec B101


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
            return '{"claims": [{"text": "completed claim"}]}'
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
async def test_post_verifier_recheck_generic_failure_is_not_successful_repair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _RecordingCredentialRuntime()
    sensitive = "private recheck failure"
    engine_instances = 0

    class FailingRecheckClaimsEngine:
        def __init__(self, analyze_fn: Any) -> None:
            nonlocal engine_instances
            engine_instances += 1
            self.instance = engine_instances

        async def run(self, **kwargs: Any) -> dict[str, Any]:
            if self.instance == 1:
                return {
                    "claims": [],
                    "summary": {"supported": 0, "refuted": 1, "nei": 0},
                }
            raise RuntimeError(sensitive)

    class FakeAnswerGenerator:
        def __init__(self, **kwargs: Any) -> None:
            pass

        async def generate(self, **kwargs: Any) -> dict[str, str]:
            return {"answer": "repaired answer"}

    import tldw_Server_API.app.core.Claims_Extraction.claims_engine as claims_engine_module

    monkeypatch.setattr(
        claims_engine_module,
        "_resolve_claims_llm_config",
        lambda: ("anthropic", None, 0.1),
    )
    monkeypatch.setattr(verifier_module, "ClaimsEngine", FailingRecheckClaimsEngine)
    monkeypatch.setattr(verifier_module, "AnswerGenerator", FakeAnswerGenerator)

    outcome = await PostGenerationVerifier(
        max_retries=1,
        unsupported_threshold=0.1,
        credential_runtime=runtime,
    ).verify_and_maybe_fix(query="question", answer="answer", base_documents=[])

    assert outcome.fixed is False  # nosec B101
    assert outcome.reason == "verification_unavailable"  # nosec B101
    assert outcome.failure_code == "provider_unavailable"  # nosec B101
    assert outcome.verification_available is False  # nosec B101
    assert sensitive not in str(outcome)  # nosec B101


@pytest.mark.parametrize("phase", ["initial", "recheck"])
@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_post_verifier_cancellation_drains_claims_and_marks_completed_call(
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
) -> None:
    runtime = _RecordingCredentialRuntime()
    entered = threading.Event()
    release = threading.Event()
    engine_instances = 0

    class BlockingClaimsEngine:
        def __init__(self, analyze_fn: Any) -> None:
            nonlocal engine_instances
            engine_instances += 1
            self.instance = engine_instances
            self.analyze_fn = analyze_fn

        async def run(self, **_kwargs: Any) -> dict[str, Any]:
            if phase == "recheck" and self.instance == 1:
                return {
                    "claims": [],
                    "summary": {"supported": 0, "refuted": 1, "nei": 0},
                }

            def call_analyzer() -> dict[str, Any]:
                self.analyze_fn("anthropic", "claim prompt", None)
                return {
                    "claims": [],
                    "summary": {"supported": 1, "refuted": 0, "nei": 0},
                }

            return await asyncio.to_thread(call_analyzer)

    class FakeAnswerGenerator:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        async def generate(self, **_kwargs: Any) -> dict[str, str]:
            return {"answer": "repaired answer"}

    def blocking_analyze(*_args: Any, **_kwargs: Any) -> str:
        entered.set()
        assert release.wait(timeout=1.0)  # nosec B101
        return '{"claims": [{"text": "completed claim"}]}'

    import tldw_Server_API.app.core.Claims_Extraction.claims_engine as claims_engine_module
    import tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib as sgl

    monkeypatch.setattr(
        claims_engine_module,
        "_resolve_claims_llm_config",
        lambda: ("anthropic", None, 0.1),
    )
    monkeypatch.setattr(sgl, "analyze", blocking_analyze)
    monkeypatch.setattr(verifier_module, "ClaimsEngine", BlockingClaimsEngine)
    monkeypatch.setattr(verifier_module, "AnswerGenerator", FakeAnswerGenerator)

    task = asyncio.create_task(
        PostGenerationVerifier(
            max_retries=1 if phase == "recheck" else 0,
            unsupported_threshold=0.1,
            credential_runtime=runtime,
        ).verify_and_maybe_fix(
            query="question",
            answer="answer",
            base_documents=[],
        )
    )
    try:
        assert await asyncio.to_thread(entered.wait, 1.0)  # nosec B101
        task.cancel()
        await asyncio.sleep(0.03)
        assert not task.done()  # nosec B101
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1.0)
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert runtime.marked == [runtime.handle]  # nosec B101


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


@pytest.mark.asyncio
async def test_runtime_generation_cancellation_marks_completed_sync_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _RecordingCredentialRuntime()
    entered, release = _install_blocking_sync_chat_adapter(
        monkeypatch,
        {"choices": [{"message": {"content": "completed answer"}}]},
    )
    generator = generation_module.LLMGenerator(
        generation_module.GenerationConfig(
            provider="anthropic",
            model="claude-test",
            streaming=False,
        )
    )
    task = asyncio.create_task(
        generator._call_llm("runtime-bound prompt", credential_runtime=runtime)
    )
    try:
        assert await asyncio.to_thread(entered.wait, 1.0)  # nosec B101
        task.cancel()
        await asyncio.sleep(0.03)
        assert not task.done()  # nosec B101
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1.0)
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert runtime.marked == [runtime.handle]  # nosec B101


@pytest.mark.parametrize("stage", ["document", "groundedness", "utility"])
@pytest.mark.asyncio
async def test_runtime_bound_grader_bypasses_saturated_default_executor(
    stage: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _RecordingCredentialRuntime()
    pool = BoundedDaemonPool(capacity=1)
    analyzer_started = threading.Event()
    blocker_started = threading.Event()
    release_blocker = threading.Event()
    document = Document(
        id=f"doc-direct-{stage}",
        content="Direct adapter evidence.",
        metadata={},
        source=DataSource.MEDIA_DB,
        score=0.8,
    )

    def block_default_executor() -> None:
        blocker_started.set()
        release_blocker.wait(timeout=2.0)

    def analyze(*_args: Any, **_kwargs: Any) -> str:
        analyzer_started.set()
        return _valid_grader_response(stage)

    monkeypatch.setattr(
        _grader_module(stage),
        "SYNC_ADAPTER_CALL_POOL",
        pool,
        raising=False,
    )
    loop = asyncio.get_running_loop()
    previous_executor = getattr(loop, "_default_executor", None)
    saturated_executor = ThreadPoolExecutor(max_workers=1)
    loop.set_default_executor(saturated_executor)
    blocker = loop.run_in_executor(None, block_default_executor)
    while not blocker_started.is_set():
        await asyncio.sleep(0)

    task = asyncio.create_task(
        _run_runtime_bound_grader(
            stage,
            analyze=analyze,
            runtime=runtime,
            document=document,
        )
    )
    try:
        for _attempt in range(100):
            if analyzer_started.is_set():
                break
            await asyncio.sleep(0.001)
        started_before_release = analyzer_started.is_set()
    finally:
        release_blocker.set()
        await blocker
        replacement_executor = previous_executor or ThreadPoolExecutor()
        loop.set_default_executor(replacement_executor)
        saturated_executor.shutdown(wait=True, cancel_futures=True)

    result = await asyncio.wait_for(task, timeout=1.0)
    assert started_before_release is True  # nosec B101
    assert result.method == "llm"  # nosec B101
    assert runtime.marked == [runtime.handle]  # nosec B101
    assert pool.active_count == 0  # nosec B101


@pytest.mark.parametrize("stage", ["document", "groundedness", "utility"])
@pytest.mark.asyncio
async def test_runtime_bound_grader_capacity_rejects_before_dispatch(
    stage: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _RecordingCredentialRuntime()
    release_holder = threading.Event()
    holder_started = threading.Event()
    holder_released = threading.Event()
    call_count = 0
    release_count = 0
    private_secret = "grader-capacity-private-secret"

    class TrackingPool(BoundedDaemonPool):
        def _release_capacity(self) -> None:
            nonlocal release_count
            release_count += 1
            super()._release_capacity()

    def hold_capacity() -> None:
        holder_started.set()
        release_holder.wait(timeout=2.0)

    def analyze(*_args: Any, **_kwargs: Any) -> str:
        nonlocal call_count
        call_count += 1
        raise RuntimeError(private_secret)

    pool = TrackingPool(capacity=1)
    pool.start(
        hold_capacity,
        name="grader-capacity-holder",
        released_event=holder_released,
    )
    assert holder_started.wait(timeout=1.0)  # nosec B101
    monkeypatch.setattr(
        _grader_module(stage),
        "SYNC_ADAPTER_CALL_POOL",
        pool,
        raising=False,
    )
    document = Document(
        id=f"doc-capacity-{stage}",
        content="Capacity evidence.",
        metadata={},
        source=DataSource.MEDIA_DB,
        score=0.8,
    )

    try:
        result = await _run_runtime_bound_grader(
            stage,
            analyze=analyze,
            runtime=runtime,
            document=document,
        )
        await asyncio.sleep(0.03)
        assert call_count == 0  # nosec B101 - rejected calls never dispatch late
        assert runtime.marked == []  # nosec B101
        assert pool.active_count == 1  # nosec B101
        assert release_count == 0  # nosec B101
        assert result.method == (  # nosec B101
            "score_fallback" if stage == "document" else "heuristic"
        )
        assert result.metadata == {  # nosec B101
            "error": "provider_unavailable",
            "verification_available": False,
        }
        assert private_secret not in str(result)  # nosec B101
    finally:
        release_holder.set()
        assert holder_released.wait(timeout=1.0)  # nosec B101

    await asyncio.sleep(0.03)
    assert call_count == 0  # nosec B101
    assert pool.active_count == 0  # nosec B101
    assert release_count == 1  # nosec B101 - only the holder was released


@pytest.mark.parametrize("stage", ["document", "groundedness", "utility"])
@pytest.mark.asyncio
async def test_runtime_bound_grader_timeout_uses_unavailable_native_fallback(
    stage: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lifecycle: list[str] = []
    entered = threading.Event()
    release = threading.Event()

    class OrderedRuntime(_RecordingCredentialRuntime):
        async def mark_used(self, handle: Any) -> None:
            lifecycle.append("mark-used")
            await super().mark_used(handle)

    class TrackingPool(BoundedDaemonPool):
        def _release_capacity(self) -> None:
            lifecycle.append("capacity-release")
            super()._release_capacity()

    runtime = OrderedRuntime()
    pool = TrackingPool(capacity=1)
    monkeypatch.setattr(
        _grader_module(stage),
        "SYNC_ADAPTER_CALL_POOL",
        pool,
        raising=False,
    )

    def slow_analyze(*_args: Any, **_kwargs: Any) -> str:
        lifecycle.append("provider-start")
        entered.set()
        release.wait(timeout=2.0)
        lifecycle.append("provider-exit")
        return _valid_grader_response(stage)

    document = Document(
        id=f"doc-timeout-{stage}",
        content="Timeout evidence.",
        metadata={},
        source=DataSource.MEDIA_DB,
        score=0.8,
    )
    async def invoke_with_runtime() -> Any:
        try:
            return await _run_runtime_bound_grader(
                stage,
                analyze=slow_analyze,
                runtime=runtime,
                document=document,
                timeout_seconds=0.01,
            )
        finally:
            lifecycle.append("runtime-close")

    task = asyncio.create_task(invoke_with_runtime())
    try:
        assert await asyncio.to_thread(entered.wait, 1.0)  # nosec B101
        await asyncio.sleep(0.03)
        assert not task.done()  # nosec B101
        assert runtime.marked == []  # nosec B101
        assert pool.active_count == 1  # nosec B101
        assert lifecycle == ["provider-start"]  # nosec B101
        release.set()
        result = await asyncio.wait_for(task, timeout=1.0)
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert result.method == (  # nosec B101
        "score_fallback" if stage == "document" else "error_fallback"
    )
    assert runtime.marked == [runtime.handle]  # nosec B101
    assert pool.active_count == 0  # nosec B101
    assert result.metadata == {  # nosec B101
        "error": "provider_unavailable",
        "verification_available": False,
    }
    assert lifecycle == [  # nosec B101
        "provider-start",
        "provider-exit",
        "capacity-release",
        "mark-used",
        "runtime-close",
    ]


@pytest.mark.parametrize("stage", ["document", "groundedness", "utility"])
@pytest.mark.asyncio
async def test_runtime_bound_grader_cancellation_drains_before_runtime_close(
    stage: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lifecycle: list[str] = []
    entered = threading.Event()
    release = threading.Event()

    class OrderedRuntime(_RecordingCredentialRuntime):
        async def mark_used(self, handle: Any) -> None:
            lifecycle.append("mark-used")
            await super().mark_used(handle)

    class TrackingPool(BoundedDaemonPool):
        def _release_capacity(self) -> None:
            lifecycle.append("capacity-release")
            super()._release_capacity()

    runtime = OrderedRuntime()
    pool = TrackingPool(capacity=1)
    monkeypatch.setattr(
        _grader_module(stage),
        "SYNC_ADAPTER_CALL_POOL",
        pool,
        raising=False,
    )
    document = Document(
        id=f"doc-cancel-{stage}",
        content="Cancellation evidence.",
        metadata={},
        source=DataSource.MEDIA_DB,
        score=0.8,
    )

    def blocking_analyze(*_args: Any, **_kwargs: Any) -> str:
        lifecycle.append("provider-start")
        entered.set()
        release.wait(timeout=2.0)
        lifecycle.append("provider-exit")
        return _valid_grader_response(stage)

    async def invoke_with_runtime() -> Any:
        try:
            return await _run_runtime_bound_grader(
                stage,
                analyze=blocking_analyze,
                runtime=runtime,
                document=document,
            )
        finally:
            lifecycle.append("runtime-close")

    task = asyncio.create_task(invoke_with_runtime())
    try:
        assert await asyncio.to_thread(entered.wait, 1.0)  # nosec B101
        task.cancel()
        await asyncio.sleep(0.03)
        assert task.done() is False
        assert runtime.marked == []  # nosec B101
        assert pool.active_count == 1  # nosec B101
        assert lifecycle == ["provider-start"]  # nosec B101

        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1.0)
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert runtime.marked == [runtime.handle]  # nosec B101
    assert pool.active_count == 0  # nosec B101
    assert lifecycle == [  # nosec B101
        "provider-start",
        "provider-exit",
        "capacity-release",
        "mark-used",
        "runtime-close",
    ]


@pytest.mark.parametrize("runtime_bound", [False, True])
@pytest.mark.asyncio
async def test_filter_relevant_batch_timeout_preserves_runtime_trust_state(
    monkeypatch: pytest.MonkeyPatch,
    runtime_bound: bool,
) -> None:
    runtime = _RecordingCredentialRuntime() if runtime_bound else None
    document = Document(
        id="doc-batch-timeout",
        content="Batch timeout evidence.",
        metadata={},
        source=DataSource.MEDIA_DB,
        score=0.8,
    )
    grader = DocumentGrader(
        analyze_fn=lambda *args, **kwargs: "{}",
        config=GradingConfig(timeout_seconds=0.01),
        credential_runtime=runtime,
    )

    async def slow_grade_document(*args: Any, **kwargs: Any) -> Any:
        await asyncio.sleep(0.05)
        raise AssertionError("outer batch timeout should cancel this call")

    monkeypatch.setattr(grader, "grade_document", slow_grade_document)

    filtered, metadata = await grader.filter_relevant("query", [document])

    assert filtered == [document]  # nosec B101
    assert metadata["grading_results"][0]["method"] == "score_fallback"  # nosec B101
    if runtime_bound:
        assert metadata["failure_code"] == "provider_unavailable"  # nosec B101
        assert metadata["verification_available"] is False  # nosec B101
    else:
        assert "failure_code" not in metadata  # nosec B101
        assert "verification_available" not in metadata  # nosec B101


@pytest.mark.parametrize("runtime_bound", [False, True])
@pytest.mark.asyncio
async def test_document_grader_generic_failure_preserves_runtime_trust_state(
    runtime_bound: bool,
) -> None:
    runtime = _RecordingCredentialRuntime() if runtime_bound else None
    document = Document(
        id="doc-generic-grader",
        content="Generic failure evidence.",
        metadata={},
        source=DataSource.MEDIA_DB,
        score=0.8,
    )

    def failing_analyze(*args: Any, **kwargs: Any) -> str:
        raise RuntimeError("private document grader failure")

    result = await DocumentGrader(
        analyze_fn=failing_analyze,
        config=GradingConfig(provider="anthropic"),
        credential_runtime=runtime,
    ).grade_document("query", document)

    if runtime_bound:
        assert result.metadata == {  # nosec B101
            "error": "provider_unavailable",
            "verification_available": False,
        }
    else:
        assert result.metadata == {"error": "grading_error"}  # nosec B101


@pytest.mark.parametrize("stage", ["groundedness", "utility"])
@pytest.mark.parametrize("runtime_bound", [False, True])
@pytest.mark.asyncio
async def test_quality_grader_generic_failure_preserves_runtime_trust_state(
    stage: str,
    runtime_bound: bool,
) -> None:
    runtime = _RecordingCredentialRuntime() if runtime_bound else None
    document = Document(
        id=f"doc-generic-{stage}",
        content="Generic quality evidence.",
        metadata={},
        source=DataSource.MEDIA_DB,
        score=0.8,
    )

    def failing_analyze(*args: Any, **kwargs: Any) -> str:
        raise RuntimeError(f"private {stage} failure")

    if stage == "groundedness":
        result = await FastGroundednessGrader(
            analyze_fn=failing_analyze,
            provider="anthropic",
            credential_runtime=runtime,
        ).grade("query", "answer", [document])
    else:
        result = await UtilityGrader(
            analyze_fn=failing_analyze,
            provider="anthropic",
            credential_runtime=runtime,
        ).grade("query", "answer")

    if runtime_bound:
        assert result.metadata == {  # nosec B101
            "error": "provider_unavailable",
            "verification_available": False,
        }
    else:
        assert result.metadata == {}  # nosec B101


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
async def test_runtime_bound_reranker_timeout_drains_then_marks_before_runtime_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lifecycle: list[str] = []

    class OrderedRuntime(_RecordingCredentialRuntime):
        async def mark_used(self, handle: Any) -> None:
            lifecycle.append("mark-used")
            await super().mark_used(handle)

    class TrackingPool(BoundedDaemonPool):
        def _release_capacity(self) -> None:
            lifecycle.append("capacity-release")
            super()._release_capacity()

    runtime = OrderedRuntime()
    rejected_runtime = OrderedRuntime()
    pool = TrackingPool(capacity=1)
    entered = threading.Event()
    release = threading.Event()
    call_count = 0
    document = Document(
        id="doc-rerank-deadline",
        content="Deadline evidence.",
        metadata={},
        source=DataSource.MEDIA_DB,
        score=0.8,
    )

    class FakeRetriever:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def retrieve(self, *args: Any, **kwargs: Any) -> list[Document]:
            return [document]

    def slow_analyze(*_args: Any, **_kwargs: Any) -> str:
        nonlocal call_count
        call_count += 1
        lifecycle.append("provider-start")
        entered.set()
        assert release.wait(timeout=2.0)  # nosec B101
        lifecycle.append("provider-exit")
        return "0.9"

    async def invoke_pipeline(
        selected_runtime: _RecordingCredentialRuntime,
        query: str,
    ) -> Any:
        try:
            return await unified_pipeline_module.unified_rag_pipeline(
                query=query,
                sources=["media_db"],
                enable_cache=False,
                enable_reranking=True,
                reranking_strategy="llm_scoring",
                enable_generation=False,
                enable_pre_retrieval_clarification=False,
                credential_runtime=selected_runtime,
            )
        finally:
            lifecycle.append(f"runtime-close:{query}")

    import tldw_Server_API.app.core.config as core_config
    import tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib as sgl

    monkeypatch.setenv("RAG_LLM_RERANK_TIMEOUT_SEC", "0.02")
    monkeypatch.setenv("RAG_LLM_RERANK_TOTAL_BUDGET_SEC", "0.04")
    monkeypatch.setattr(unified_pipeline_module, "MultiDatabaseRetriever", FakeRetriever)
    monkeypatch.setattr(sgl, "analyze", slow_analyze)
    monkeypatch.setattr(
        core_config,
        "load_and_log_configs",
        lambda: {
            "RAG_LLM_RERANKER_PROVIDER": "anthropic",
            "RAG_LLM_RERANKER_MODEL": "rerank-model",
        },
    )
    monkeypatch.setattr(
        advanced_reranking_module,
        "SYNC_ADAPTER_CALL_POOL",
        pool,
        raising=False,
    )

    task = asyncio.create_task(invoke_pipeline(runtime, "deadline rerank"))
    result: Any = None
    try:
        assert await asyncio.to_thread(entered.wait, 1.0)  # nosec B101
        await asyncio.sleep(0.05)
        done, _pending = await asyncio.wait({task}, timeout=0.25)
        assert task not in done  # nosec B101 - runtime ownership drains the late worker
        assert pool.active_count == 1  # nosec B101
        assert runtime.marked == []  # nosec B101

        rejected = await asyncio.wait_for(
            invoke_pipeline(rejected_runtime, "capacity rejected"),
            timeout=1.0,
        )
        assert rejected.metadata["reranking"] == {  # nosec B101
            "degraded": True,
            "failure_code": "provider_unavailable",
            "verification_available": False,
        }
        assert call_count == 1  # nosec B101 - rejected work was never dispatched
        assert rejected_runtime.marked == []  # nosec B101

        release.set()
        result = await asyncio.wait_for(task, timeout=1.0)
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)

    assert pool.active_count == 0  # nosec B101
    assert runtime.marked == [runtime.handle]  # nosec B101
    assert result.metadata["reranking"] == {  # nosec B101
        "degraded": True,
        "failure_code": "provider_unavailable",
        "verification_available": False,
    }
    assert lifecycle == [  # nosec B101
        "provider-start",
        "runtime-close:capacity rejected",
        "provider-exit",
        "capacity-release",
        "mark-used",
        "runtime-close:deadline rerank",
    ]


@pytest.mark.asyncio
async def test_runtime_bound_reranker_cancellation_drains_success_before_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lifecycle: list[str] = []
    entered = threading.Event()
    release = threading.Event()

    class TrackingPool(BoundedDaemonPool):
        def _release_capacity(self) -> None:
            lifecycle.append("capacity-release")
            super()._release_capacity()

    class Client:
        credentials_resolved = True
        used = False

        def analyze(self, _prompt: str) -> str:
            lifecycle.append("provider-start")
            entered.set()
            assert release.wait(timeout=2.0)  # nosec B101
            lifecycle.append("provider-exit")
            return "0.9"

    client = Client()
    pool = TrackingPool(capacity=1)
    monkeypatch.setenv("RAG_LLM_RERANK_TIMEOUT_SEC", "1")
    monkeypatch.setattr(advanced_reranking_module, "SYNC_ADAPTER_CALL_POOL", pool)
    reranker = LLMReranker(
        RerankingConfig(
            strategy=RerankingStrategy.LLM_SCORING,
            top_k=1,
        ),
        llm_client=client,
    )
    document = Document(
        id="cancelled-rerank",
        content="Cancellation evidence.",
        metadata={},
        source=DataSource.MEDIA_DB,
        score=0.8,
    )

    async def invoke_with_runtime() -> Any:
        try:
            return await reranker.rerank("query", [document])
        finally:
            if client.used:
                lifecycle.append("mark-used")
            lifecycle.append("runtime-close")

    task = asyncio.create_task(invoke_with_runtime())
    try:
        assert await asyncio.to_thread(entered.wait, 1.0)  # nosec B101
        task.cancel()
        await asyncio.sleep(0.03)
        assert task.done() is False
        assert client.used is False  # nosec B101
        assert pool.active_count == 1  # nosec B101

        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1.0)
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert client.used is True  # nosec B101
    assert pool.active_count == 0  # nosec B101
    assert lifecycle == [  # nosec B101
        "provider-start",
        "provider-exit",
        "capacity-release",
        "mark-used",
        "runtime-close",
    ]


@pytest.mark.parametrize("late_outcome", ["empty", "error"])
@pytest.mark.asyncio
async def test_runtime_bound_reranker_late_failure_does_not_mark_used(
    monkeypatch: pytest.MonkeyPatch,
    late_outcome: str,
) -> None:
    entered = threading.Event()
    release = threading.Event()

    class Client:
        credentials_resolved = True
        used = False

        def analyze(self, _prompt: str) -> str:
            entered.set()
            assert release.wait(timeout=2.0)  # nosec B101
            if late_outcome == "error":
                raise RuntimeError("private late reranker failure")
            return ""

    client = Client()
    pool = BoundedDaemonPool(capacity=1)
    monkeypatch.setenv("RAG_LLM_RERANK_TIMEOUT_SEC", "0.01")
    monkeypatch.setattr(advanced_reranking_module, "SYNC_ADAPTER_CALL_POOL", pool)
    reranker = LLMReranker(
        RerankingConfig(
            strategy=RerankingStrategy.LLM_SCORING,
            top_k=1,
        ),
        llm_client=client,
    )
    document = Document(
        id=f"late-{late_outcome}",
        content="Late failure evidence.",
        metadata={},
        source=DataSource.MEDIA_DB,
        score=0.8,
    )

    task = asyncio.create_task(reranker.rerank("query", [document]))
    try:
        assert await asyncio.to_thread(entered.wait, 1.0)  # nosec B101
        await asyncio.sleep(0.03)
        assert task.done() is False
        release.set()
        reranked = await asyncio.wait_for(task, timeout=1.0)
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert reranked[0].rerank_score == document.score  # nosec B101
    assert client.used is False  # nosec B101
    assert pool.active_count == 0  # nosec B101


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


@pytest.mark.parametrize(
    ("stage", "dispatch_response", "metadata_key"),
    [
        ("gap", '["follow up safely"]', "gap_analysis"),
        ("critique", "- no unsupported claims", "synthesis"),
        ("faithfulness", "[]", "faithfulness"),
    ],
)
@pytest.mark.asyncio
async def test_unified_optional_sync_stage_bypasses_saturated_default_executor(
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    dispatch_response: str,
    metadata_key: str,
) -> None:
    runtime = _RecordingCredentialRuntime()
    analyzer_started = threading.Event()
    blocker_started = threading.Event()
    release_blocker = threading.Event()
    release_count = 0

    class TrackingPool(BoundedDaemonPool):
        def _release_capacity(self) -> None:
            nonlocal release_count
            release_count += 1
            super()._release_capacity()

    def block_default_executor() -> None:
        blocker_started.set()
        release_blocker.wait(timeout=2.0)

    def analyze(*_args: Any, **_kwargs: Any) -> str:
        analyzer_started.set()
        return dispatch_response

    import tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib as sgl

    pool = TrackingPool(capacity=1)
    monkeypatch.setattr(sgl, "analyze", analyze)
    monkeypatch.setattr(
        unified_pipeline_module,
        "SYNC_ADAPTER_CALL_POOL",
        pool,
        raising=False,
    )
    loop = asyncio.get_running_loop()
    previous_executor = getattr(loop, "_default_executor", None)
    saturated_executor = ThreadPoolExecutor(max_workers=1)
    loop.set_default_executor(saturated_executor)
    blocker = loop.run_in_executor(None, block_default_executor)
    while not blocker_started.is_set():
        await asyncio.sleep(0)

    task = asyncio.create_task(
        _run_unified_bound_sgl_stage(monkeypatch, stage, runtime)
    )
    try:
        for _attempt in range(100):
            if analyzer_started.is_set():
                break
            await asyncio.sleep(0.001)
        started_before_executor_release = analyzer_started.is_set()
    finally:
        release_blocker.set()
        await blocker
        loop.set_default_executor(previous_executor or ThreadPoolExecutor())
        saturated_executor.shutdown(wait=True, cancel_futures=True)

    result = await asyncio.wait_for(task, timeout=1.0)
    assert started_before_executor_release is True  # nosec B101
    assert runtime.marked == [runtime.handle]  # nosec B101
    assert pool.active_count == 0  # nosec B101
    assert release_count == 1  # nosec B101 - one admitted call, one release
    stage_metadata = result.metadata.get(metadata_key, {})
    assert "failure_code" not in stage_metadata  # nosec B101
    assert stage_metadata.get("verification_available") is not False  # nosec B101
    if stage == "gap":
        assert result.metadata["followups"] == ["follow up safely"]  # nosec B101
    elif stage == "critique":
        assert stage_metadata["enabled"] is True  # nosec B101
        assert result.generated_answer == "A grounded generated answer."  # nosec B101
    else:
        assert "faithfulness_score" in stage_metadata  # nosec B101


@pytest.mark.parametrize(
    ("stage", "metadata_key"),
    [
        ("gap", "gap_analysis"),
        ("critique", "synthesis"),
        ("faithfulness", "faithfulness"),
    ],
)
@pytest.mark.asyncio
async def test_unified_optional_sync_stage_capacity_rejects_without_late_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    metadata_key: str,
) -> None:
    runtime = _RecordingCredentialRuntime()
    holder_started = threading.Event()
    holder_released = threading.Event()
    release_holder = threading.Event()
    call_count = 0
    release_count = 0
    private_secret = f"private-{stage}-capacity-secret"

    class TrackingPool(BoundedDaemonPool):
        def _release_capacity(self) -> None:
            nonlocal release_count
            release_count += 1
            super()._release_capacity()

    def hold_capacity() -> None:
        holder_started.set()
        release_holder.wait(timeout=2.0)

    def analyze(*_args: Any, **_kwargs: Any) -> str:
        nonlocal call_count
        call_count += 1
        raise RuntimeError(private_secret)

    import tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib as sgl

    pool = TrackingPool(capacity=1)
    pool.start(
        hold_capacity,
        name=f"{stage}-capacity-holder",
        released_event=holder_released,
    )
    assert holder_started.wait(timeout=1.0)  # nosec B101
    monkeypatch.setattr(sgl, "analyze", analyze)
    monkeypatch.setattr(
        unified_pipeline_module,
        "SYNC_ADAPTER_CALL_POOL",
        pool,
        raising=False,
    )

    try:
        result = await _run_unified_bound_sgl_stage(monkeypatch, stage, runtime)
        await asyncio.sleep(0.03)
        assert call_count == 0  # nosec B101 - rejected work never dispatches
        assert runtime.marked == []  # nosec B101
        assert pool.active_count == 1  # nosec B101 - only the holder owns capacity
        assert release_count == 0  # nosec B101
        stage_metadata = result.metadata[metadata_key]
        assert stage_metadata["failure_code"] == "provider_unavailable"  # nosec B101
        assert stage_metadata["verification_available"] is False  # nosec B101
        assert private_secret not in str(result.metadata)  # nosec B101
        assert private_secret not in str(result.errors)  # nosec B101
    finally:
        release_holder.set()
        assert holder_released.wait(timeout=1.0)  # nosec B101

    await asyncio.sleep(0.03)
    assert call_count == 0  # nosec B101 - capacity release cannot start rejected work
    assert pool.active_count == 0  # nosec B101
    assert release_count == 1  # nosec B101 - only the holder was released


@pytest.mark.parametrize(
    ("stage", "dispatch_response"),
    [
        ("gap", '["follow up safely"]'),
        ("critique", "- no unsupported claims"),
        ("faithfulness", "[]"),
    ],
)
@pytest.mark.parametrize("termination", ["cancellation", "timeout"])
@pytest.mark.asyncio
async def test_unified_optional_sync_stage_owns_worker_until_actual_exit(
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    dispatch_response: str,
    termination: str,
) -> None:
    lifecycle: list[str] = []
    entered = threading.Event()
    release = threading.Event()

    class OrderedRuntime(_RecordingCredentialRuntime):
        async def mark_used(self, handle: Any) -> None:
            lifecycle.append("mark-used")
            await super().mark_used(handle)

    class TrackingPool(BoundedDaemonPool):
        def _release_capacity(self) -> None:
            lifecycle.append("capacity-release")
            super()._release_capacity()

    def blocking_analyze(*_args: Any, **_kwargs: Any) -> str:
        lifecycle.append("provider-start")
        entered.set()
        release.wait(timeout=2.0)
        lifecycle.append("provider-exit")
        return dispatch_response

    async def invoke_with_runtime() -> Any:
        try:
            return await _run_unified_bound_sgl_stage(monkeypatch, stage, runtime)
        finally:
            lifecycle.append("runtime-close")

    import tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib as sgl

    runtime = OrderedRuntime()
    pool = TrackingPool(capacity=1)
    monkeypatch.setattr(sgl, "analyze", blocking_analyze)
    monkeypatch.setattr(
        unified_pipeline_module,
        "SYNC_ADAPTER_CALL_POOL",
        pool,
        raising=False,
    )
    pipeline_task = asyncio.create_task(invoke_with_runtime())
    for _attempt in range(100):
        if entered.is_set():
            break
        await asyncio.sleep(0.001)
    assert entered.is_set()  # nosec B101

    if termination == "cancellation":
        terminal_task = pipeline_task
        terminal_task.cancel()
    else:
        terminal_task = asyncio.create_task(
            asyncio.wait_for(pipeline_task, timeout=0.01)
        )

    try:
        await asyncio.sleep(0.03)
        assert terminal_task.done() is False
        assert runtime.marked == []  # nosec B101
        assert pool.active_count == 1  # nosec B101
        assert lifecycle == ["provider-start"]  # nosec B101
        release.set()
        expected_error = (
            asyncio.CancelledError
            if termination == "cancellation"
            else asyncio.TimeoutError
        )
        with pytest.raises(expected_error):
            await asyncio.wait_for(terminal_task, timeout=1.0)
    finally:
        release.set()
        if not terminal_task.done():
            terminal_task.cancel()
        await asyncio.gather(terminal_task, pipeline_task, return_exceptions=True)

    assert runtime.marked == [runtime.handle]  # nosec B101
    assert pool.active_count == 0  # nosec B101
    assert lifecycle == [  # nosec B101
        "provider-start",
        "provider-exit",
        "capacity-release",
        "mark-used",
        "runtime-close",
    ]
