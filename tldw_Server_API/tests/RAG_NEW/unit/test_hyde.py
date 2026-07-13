import asyncio
import threading
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

import tldw_Server_API.app.core.LLM_Calls as llm_calls
from tldw_Server_API.app.core.RAG.rag_service import hyde
from tldw_Server_API.app.core.RAG.rag_service.types import DataSource, Document
from tldw_Server_API.app.core.RAG.rag_service.unified_pipeline import (
    unified_batch_pipeline,
    unified_rag_pipeline,
)


class _CredentialRuntime:
    def __init__(self, provider: str, *, base_url: str | None = None):
        section = "openai_api" if provider == "openai" else "huggingface_api"
        self.handle = SimpleNamespace(
            provider=provider,
            api_key="runtime-embedding-key",
            app_config={section: {"api_base_url": base_url}} if base_url else {},
            credentials_resolved=True,
        )
        self.resolved: list[str] = []
        self.marked: list[Any] = []

    async def resolve(self, provider: str):
        self.resolved.append(provider)
        return self.handle

    async def mark_used(self, handle: Any) -> None:
        self.marked.append(handle)


class _FakeLogger:
    def __init__(self):
        self.debugs = []
        self.warnings = []

    def debug(self, message):
        self.debugs.append(message)

    def warning(self, message):
        self.warnings.append(message)


@pytest.mark.unit
def test_embedding_degradation_preserves_invalid_configuration_taxonomy():
    from tldw_Server_API.app.core.Chat.Chat_Deps import ChatConfigurationError
    from tldw_Server_API.app.core.Embeddings.async_embeddings import EmbeddingEndpointError
    from tldw_Server_API.app.core.RAG.rag_service import unified_pipeline

    for error in (
        EmbeddingEndpointError("local_api"),
        ChatConfigurationError(
            "secret invalid endpoint detail",
            provider="local_api",
            error_code="provider_configuration_invalid",
        ),
    ):
        metadata = {}
        hyde._record_embedding_degraded(metadata, error)
        assert metadata == {
            "embedding_coverage": "degraded",
            "failure_code": "provider_configuration_invalid",
        }
        assert unified_pipeline._bounded_provider_failure_code(error) == (
            "provider_configuration_invalid"
            if isinstance(error, ChatConfigurationError)
            else "provider_unavailable"
        )


@pytest.mark.unit
def test_generate_with_llm_sanitizes_generation_failure(monkeypatch):
    secret = "sk-secret-hyde-generation"

    def fake_analyze(**kwargs):
        raise RuntimeError(secret)

    fake_sgl = SimpleNamespace(analyze=fake_analyze)
    fake_logger = _FakeLogger()
    monkeypatch.setattr(hyde, "logger", fake_logger)
    monkeypatch.setattr(llm_calls, "Summarization_General_Lib", fake_sgl)
    monkeypatch.setitem(
        __import__("sys").modules,
        "tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib",
        fake_sgl,
    )

    assert hyde._generate_with_llm("prompt", "openai", "gpt-4o-mini") == ""
    assert fake_logger.warnings == ["HyDE LLM generation failed"]
    assert secret not in "\n".join(fake_logger.warnings + fake_logger.debugs)


@pytest.mark.unit
def test_generate_with_llm_sanitizes_utility_unavailable(monkeypatch):
    secret = "sk-secret-hyde-utility"
    real_import = __import__("builtins").__import__

    def fake_import(name, *args, **kwargs):
        if name == "tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib":
            raise ImportError(secret)
        return real_import(name, *args, **kwargs)

    fake_logger = _FakeLogger()
    monkeypatch.setattr(hyde, "logger", fake_logger)
    monkeypatch.setattr("builtins.__import__", fake_import)

    assert hyde._generate_with_llm("prompt", "openai", "gpt-4o-mini") is None
    assert fake_logger.debugs == ["HyDE LLM utility unavailable"]
    assert secret not in "\n".join(fake_logger.warnings + fake_logger.debugs)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_embed_text_sanitizes_embedding_failure(monkeypatch):
    secret = "sk-secret-hyde-embedding"
    real_import = __import__("builtins").__import__

    def fake_import(name, *args, **kwargs):
        if name == "tldw_Server_API.app.core.Embeddings.Embeddings_Server.Embeddings_Create":
            raise RuntimeError(secret)
        return real_import(name, *args, **kwargs)

    fake_logger = _FakeLogger()
    monkeypatch.setattr(hyde, "logger", fake_logger)
    monkeypatch.setattr("builtins.__import__", fake_import)

    assert await hyde.embed_text("text") is None
    assert fake_logger.warnings == ["HyDE embedding failed"]
    assert secret not in "\n".join(fake_logger.warnings + fake_logger.debugs)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_embed_text_uses_runtime_credentials_for_hosted_openai(monkeypatch):
    from tldw_Server_API.app.core.Embeddings.Embeddings_Server import Embeddings_Create

    captured: dict[str, Any] = {}
    runtime = _CredentialRuntime("openai", base_url="https://user-embeddings.example/v1")
    config = {
        "embedding_config": {
            "default_model_id": "openai:text-embedding-3-small",
            "models": {
                "openai:text-embedding-3-small": SimpleNamespace(provider="openai"),
            },
        }
    }

    def fake_create(texts, user_app_config, model_id_override=None, **kwargs):
        captured.update(
            texts=texts,
            user_app_config=user_app_config,
            model_id_override=model_id_override,
            kwargs=kwargs,
        )
        return [[0.1, 0.2]]

    monkeypatch.setattr(Embeddings_Create, "get_embedding_config", lambda: config)
    monkeypatch.setattr(Embeddings_Create, "create_embeddings_batch", fake_create)

    assert await hyde.embed_text("hosted", credential_runtime=runtime) == [0.1, 0.2]
    assert runtime.resolved == ["openai"]
    assert runtime.marked == [runtime.handle]
    assert captured["kwargs"] == {
        "api_key_override": "runtime-embedding-key",
        "base_url_override": "https://user-embeddings.example/v1",
        "credentials_resolved": True,
    }


@pytest.mark.unit
@pytest.mark.asyncio
async def test_embed_text_does_not_resolve_local_huggingface(monkeypatch):
    from tldw_Server_API.app.core.Embeddings.Embeddings_Server import Embeddings_Create

    captured: dict[str, Any] = {}
    runtime = _CredentialRuntime("huggingface")
    config = {
        "embedding_config": {
            "default_model_id": "huggingface:local-model",
            "models": {
                "huggingface:local-model": SimpleNamespace(provider="huggingface"),
            },
        }
    }

    def fake_create(texts, user_app_config, model_id_override=None, **kwargs):
        captured["kwargs"] = kwargs
        return [[0.3, 0.4]]

    monkeypatch.setattr(Embeddings_Create, "get_embedding_config", lambda: config)
    monkeypatch.setattr(Embeddings_Create, "create_embeddings_batch", fake_create)

    assert await hyde.embed_text("local", credential_runtime=runtime) == [0.3, 0.4]
    assert runtime.resolved == []
    assert runtime.marked == []
    assert captured["kwargs"] == {}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_embed_text_runtime_local_api_uses_exact_endpoint_without_key(monkeypatch):
    from tldw_Server_API.app.core.Embeddings.Embeddings_Server import Embeddings_Create

    captured: dict[str, Any] = {}
    runtime = _CredentialRuntime("openai")
    config = {
        "embedding_config": {
            "default_model_id": "local_api:runtime-model",
            "models": {
                "local_api:runtime-model": SimpleNamespace(
                    provider="local_api",
                    api_url="https://runtime-local.example/embeddings",
                    api_key="configured-key-must-not-be-used",
                ),
            },
        }
    }

    def fake_create(texts, user_app_config, model_id_override=None, **kwargs):
        captured["kwargs"] = kwargs
        return [[0.1, 0.2]]

    monkeypatch.setattr(Embeddings_Create, "get_embedding_config", lambda: config)
    monkeypatch.setattr(Embeddings_Create, "create_embeddings_batch", fake_create)

    assert await hyde.embed_text("local api", credential_runtime=runtime) == [0.1, 0.2]
    assert runtime.resolved == []
    assert runtime.marked == []
    assert captured["kwargs"] == {
        "api_key_override": None,
        "base_url_override": "https://runtime-local.example/embeddings",
        "credentials_resolved": True,
    }


@pytest.mark.unit
def test_embedding_provider_resolution_matches_sync_ambiguous_bare_model_precedence():
    config = {
        "embedding_config": {
            "default_model_id": "shared-model",
            "models": {
                "openai:shared-model": SimpleNamespace(provider="openai"),
                "local_api:shared-model": SimpleNamespace(provider="local_api"),
            },
        }
    }

    assert hyde._embedding_provider_from_config(config) == "openai"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_embed_text_runtime_failure_degrades_with_bounded_metadata(monkeypatch):
    from tldw_Server_API.app.core.AuthNZ.byok_runtime import ByokResolutionError
    from tldw_Server_API.app.core.Embeddings.Embeddings_Server import Embeddings_Create

    class _FailingRuntime:
        async def resolve(self, provider: str):
            raise ByokResolutionError("credential_store_unavailable", provider)

    config = {
        "embedding_config": {
            "default_model_id": "openai:text-embedding-3-small",
            "models": {
                "openai:text-embedding-3-small": SimpleNamespace(provider="openai"),
            },
        }
    }
    stage_metadata: dict[str, Any] = {}
    monkeypatch.setattr(Embeddings_Create, "get_embedding_config", lambda: config)

    assert (
        await hyde.embed_text(
            "hosted",
            credential_runtime=_FailingRuntime(),
            stage_metadata=stage_metadata,
        )
        is None
    )
    assert stage_metadata == {
        "embedding_coverage": "degraded",
        "failure_code": "credential_store_unavailable",
    }


@pytest.mark.unit
@pytest.mark.asyncio
async def test_sync_embedding_thread_drains_before_cancellation_propagates():
    started = threading.Event()
    release = threading.Event()

    def blocking_call():
        started.set()
        release.wait(timeout=2)
        return [1.0]

    task = asyncio.create_task(hyde._run_sync_embedding_call(blocking_call))
    for _ in range(100):
        if started.is_set():
            break
        await asyncio.sleep(0.001)
    assert started.is_set()

    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.unit
@pytest.mark.asyncio
async def test_sync_embedding_thread_drains_after_repeated_cancellation():
    started = threading.Event()
    release = threading.Event()

    def blocking_call():
        started.set()
        release.wait(timeout=2)
        return [1.0]

    task = asyncio.create_task(hyde._run_sync_embedding_call(blocking_call))
    for _ in range(100):
        if started.is_set():
            break
        await asyncio.sleep(0.001)
    assert started.is_set()

    task.cancel()
    for _ in range(100):
        await asyncio.sleep(0)
        waiter = getattr(task, "_fut_waiter", None)
        if waiter is not None and not waiter.cancelled():
            break
    assert waiter is not None and not waiter.cancelled()

    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.unit
@pytest.mark.asyncio
async def test_sync_embedding_thread_records_completed_use_before_cancellation():
    started = threading.Event()
    release = threading.Event()
    marked: list[list[float]] = []

    def blocking_call():
        started.set()
        release.wait(timeout=2)
        return [1.0]

    async def record_success(result):
        marked.append(result)

    task = asyncio.create_task(
        hyde._run_sync_embedding_call(blocking_call, on_success=record_success)
    )
    for _ in range(100):
        if started.is_set():
            break
        await asyncio.sleep(0.001)
    assert started.is_set()

    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert marked == [[1.0]]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_sync_embedding_thread_drains_mark_used_after_worker_completion():
    mark_started = asyncio.Event()
    mark_release = asyncio.Event()
    mark_calls = 0
    mark_completions = 0

    async def record_success(result):
        nonlocal mark_calls, mark_completions
        mark_calls += 1
        mark_started.set()
        await mark_release.wait()
        mark_completions += 1

    task = asyncio.create_task(
        hyde._run_sync_embedding_call(lambda: [1.0], on_success=record_success)
    )
    await asyncio.wait_for(mark_started.wait(), timeout=1)

    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    assert mark_calls == 1
    assert mark_completions == 0

    mark_release.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert mark_calls == 1
    assert mark_completions == 1


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("vector", [["0.1", "0.2"], [10**10000]])
async def test_mark_runtime_used_rejects_malformed_vector(vector):
    from tldw_Server_API.app.core.Embeddings.async_embeddings import EmbeddingProviderError

    runtime = _CredentialRuntime("openai")

    with pytest.raises(EmbeddingProviderError) as exc_info:
        await hyde._mark_runtime_used_for_embeddings(
            [vector],
            credential_runtime=runtime,
            handle=runtime.handle,
        )

    assert exc_info.value.code == "provider_failure"
    assert runtime.marked == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_unified_pipeline_with_hyde_merges_results():
    """Ensure HyDE path runs and merged results are returned when enabled."""
    base_docs = [
        Document(id="base1", content="Base A", metadata={}, source=DataSource.MEDIA_DB, score=0.2),
        Document(id="base2", content="Base B", metadata={}, source=DataSource.MEDIA_DB, score=0.1),
    ]
    hyde_docs = [
        Document(id="hyde1", content="HyDE A", metadata={}, source=DataSource.MEDIA_DB, score=0.9),
        Document(id="hyde2", content="HyDE B", metadata={}, source=DataSource.MEDIA_DB, score=0.8),
    ]

    # Fake retriever that returns baseline docs and offers a MEDIA_DB retriever with retrieve_hybrid
    class _FakeMediaRetriever:
        async def retrieve_hybrid(self, *args, **kwargs):
            # Ensure HyDE vector was provided via kwargs for vector search
            assert "query_vector" in kwargs
            return hyde_docs

    runtime = object()
    captured: dict[str, Any] = {}

    class _FakeMultiRetriever:
        def __init__(self, *args, **kwargs):
            captured["retrieval_runtime"] = kwargs.get("credential_runtime")
            self.retrievers = {DataSource.MEDIA_DB: _FakeMediaRetriever()}
        async def retrieve(self, *args, **kwargs):
            return base_docs

    hyde_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3])
    with patch("tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.MultiDatabaseRetriever", _FakeMultiRetriever), \
         patch("tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.generate_hypothetical_answer", return_value="Hypo answer"), \
         patch("tldw_Server_API.app.core.RAG.rag_service.unified_pipeline.hyde_embed_text", new=hyde_embedding):
        result = await unified_rag_pipeline(
            query="test hyde",
            sources=["media_db"],
            top_k=10,
            enable_hyde=True,
            adaptive_hybrid_weights=False,
            credential_runtime=runtime,
            enable_generation=False,
        )

        # Response shape
        from tldw_Server_API.app.api.v1.schemas.rag_schemas_unified import UnifiedRAGResponse
        assert isinstance(result, UnifiedRAGResponse)
        # HyDE metadata present
        assert result.metadata.get("hyde_applied") is True
        assert result.metadata.get("hyde_merged_count") == len(hyde_docs)
        # Documents include both baseline and hyde docs (dedup by id)
        ids = {d["id"] for d in result.documents}
        for d in base_docs + hyde_docs:
            assert d.id in ids
        assert captured["retrieval_runtime"] is runtime
        assert hyde_embedding.await_args.kwargs["credential_runtime"] is runtime
        assert isinstance(hyde_embedding.await_args.kwargs["stage_metadata"], dict)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_unified_pipeline_propagates_required_retrieval_provider_failure(monkeypatch):
    from tldw_Server_API.app.core.Chat.Chat_Deps import ChatAuthenticationError
    from tldw_Server_API.app.core.RAG.rag_service import unified_pipeline

    provider_error = ChatAuthenticationError(
        "Embedding provider authentication failed.",
        provider="openai",
    )

    class _FailingRetriever:
        def __init__(self, *args, **kwargs):
            self.retrievers = {}

        async def retrieve(self, *args, **kwargs):
            raise provider_error

    logger = Mock()
    monkeypatch.setattr(unified_pipeline, "MultiDatabaseRetriever", _FailingRetriever)
    monkeypatch.setattr(unified_pipeline, "logger", logger)

    with pytest.raises(ChatAuthenticationError) as exc_info:
        await unified_pipeline.unified_rag_pipeline(
            query="required provider retrieval",
            sources=["media_db"],
            adaptive_hybrid_weights=False,
            enable_cache=False,
            enable_reranking=False,
            enable_generation=False,
            credential_runtime=object(),
        )

    assert exc_info.value is provider_error
    logger.error.assert_not_called()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_unified_pipeline_keeps_legacy_typed_retrieval_fts_fallback(monkeypatch):
    from tldw_Server_API.app.core.Chat.Chat_Deps import ChatAuthenticationError
    from tldw_Server_API.app.core.RAG.rag_service import database_retrievers, unified_pipeline

    fallback_doc = Document(
        id="fallback-typed",
        content="fallback",
        metadata={},
        source=DataSource.MEDIA_DB,
        score=0.7,
    )

    class _FailingRetriever:
        def __init__(self, *args, **kwargs):
            self.retrievers = {}

        async def retrieve(self, *args, **kwargs):
            raise ChatAuthenticationError("legacy provider failure", provider="openai")

    class _FallbackMediaRetriever:
        def __init__(self, *args, **kwargs):
            pass

        async def retrieve(self, *args, **kwargs):
            return [fallback_doc]

    monkeypatch.setattr(unified_pipeline, "MultiDatabaseRetriever", _FailingRetriever)
    monkeypatch.setattr(database_retrievers, "MediaDBRetriever", _FallbackMediaRetriever)

    result = await unified_pipeline.unified_rag_pipeline(
        query="legacy typed retrieval",
        sources=["media_db"],
        media_db_path="media.db",
        search_mode="hybrid",
        adaptive_hybrid_weights=False,
        enable_cache=False,
        enable_reranking=False,
        enable_generation=False,
    )

    assert [document["id"] for document in result.documents] == ["fallback-typed"]
    assert result.metadata["fallbacks"]["media_db_fts_on_error"] is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_unified_pipeline_keeps_ordinary_retrieval_fts_fallback(monkeypatch):
    from tldw_Server_API.app.core.RAG.rag_service import database_retrievers, unified_pipeline

    fallback_doc = Document(
        id="fallback-1",
        content="fallback",
        metadata={},
        source=DataSource.MEDIA_DB,
        score=0.7,
    )

    class _FailingRetriever:
        def __init__(self, *args, **kwargs):
            self.retrievers = {}

        async def retrieve(self, *args, **kwargs):
            raise RuntimeError("ordinary retrieval failure")

    class _FallbackMediaRetriever:
        def __init__(self, *args, **kwargs):
            pass

        async def retrieve(self, *args, **kwargs):
            return [fallback_doc]

    monkeypatch.setattr(unified_pipeline, "MultiDatabaseRetriever", _FailingRetriever)
    monkeypatch.setattr(database_retrievers, "MediaDBRetriever", _FallbackMediaRetriever)

    result = await unified_pipeline.unified_rag_pipeline(
        query="ordinary retrieval",
        sources=["media_db"],
        media_db_path="media.db",
        search_mode="hybrid",
        adaptive_hybrid_weights=False,
        enable_cache=False,
        enable_reranking=False,
        enable_generation=False,
    )

    assert [document["id"] for document in result.documents] == ["fallback-1"]
    assert result.metadata["fallbacks"]["media_db_fts_on_error"] is True


class _CountingBreaker:
    def __init__(self) -> None:
        self.failure_count = 0
        self.state = "closed"

    async def call(self, func, *args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception:
            self.failure_count += 1
            raise


class _TestCoordinator:
    def __init__(self) -> None:
        self.circuit_breakers: dict[str, _CountingBreaker] = {}

    def register_circuit_breaker(self, component: str, _config: Any) -> None:
        self.circuit_breakers[component] = _CountingBreaker()


class _TestRetryConfig:
    def __init__(self, *, max_attempts: int) -> None:
        self.max_attempts = max_attempts


class _TestRetryPolicy:
    def __init__(self, config: _TestRetryConfig) -> None:
        self.max_attempts = config.max_attempts

    async def execute(self, func):
        for attempt in range(self.max_attempts):
            try:
                return await func()
            except Exception:
                if attempt + 1 == self.max_attempts:
                    raise


def _install_counting_resilience(monkeypatch, unified_pipeline):
    coordinator = _TestCoordinator()
    monkeypatch.setattr(unified_pipeline, "get_coordinator", lambda: coordinator)
    monkeypatch.setattr(unified_pipeline, "CircuitBreakerConfig", lambda: object())
    monkeypatch.setattr(unified_pipeline, "RetryConfig", _TestRetryConfig)
    monkeypatch.setattr(unified_pipeline, "RetryPolicy", _TestRetryPolicy)
    return coordinator


@pytest.mark.unit
@pytest.mark.asyncio
async def test_required_provider_failure_bypasses_retry_and_circuit_accounting(monkeypatch):
    from tldw_Server_API.app.core.Chat.Chat_Deps import ChatAuthenticationError
    from tldw_Server_API.app.core.RAG.rag_service import unified_pipeline

    provider_error = ChatAuthenticationError(
        "Embedding provider authentication failed.",
        provider="openai",
    )
    calls = 0

    class _FailingRetriever:
        def __init__(self, *args, **kwargs):
            self.retrievers = {}

        async def retrieve(self, *args, **kwargs):
            nonlocal calls
            calls += 1
            raise provider_error

    coordinator = _install_counting_resilience(monkeypatch, unified_pipeline)
    monkeypatch.setattr(unified_pipeline, "MultiDatabaseRetriever", _FailingRetriever)

    with pytest.raises(ChatAuthenticationError) as exc_info:
        await unified_pipeline.unified_rag_pipeline(
            query="required provider retrieval",
            sources=["media_db"],
            credential_runtime=object(),
            enable_resilience=True,
            circuit_breaker=True,
            retry_attempts=3,
            adaptive_hybrid_weights=False,
            enable_cache=False,
            enable_reranking=False,
            enable_generation=False,
        )

    breaker = coordinator.circuit_breakers["retrieval"]
    assert exc_info.value is provider_error
    assert calls == 1
    assert breaker.failure_count == 0
    assert breaker.state == "closed"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_ordinary_retrieval_failure_keeps_retry_and_circuit_accounting(monkeypatch):
    from tldw_Server_API.app.core.RAG.rag_service import unified_pipeline

    calls = 0

    class _FailingRetriever:
        def __init__(self, *args, **kwargs):
            self.retrievers = {}

        async def retrieve(self, *args, **kwargs):
            nonlocal calls
            calls += 1
            raise RuntimeError("ordinary retrieval failure")

    coordinator = _install_counting_resilience(monkeypatch, unified_pipeline)
    monkeypatch.setattr(unified_pipeline, "MultiDatabaseRetriever", _FailingRetriever)

    result = await unified_pipeline.unified_rag_pipeline(
        query="ordinary resilient retrieval",
        sources=["media_db"],
        credential_runtime=object(),
        enable_resilience=True,
        circuit_breaker=True,
        retry_attempts=3,
        adaptive_hybrid_weights=False,
        enable_cache=False,
        enable_reranking=False,
        enable_generation=False,
    )

    breaker = coordinator.circuit_breakers["retrieval"]
    assert result.documents == []
    assert calls == 3
    assert breaker.failure_count == 3
    assert breaker.state == "closed"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_optional_expansion_provider_failure_degrades_with_bounded_metadata(monkeypatch):
    from tldw_Server_API.app.core.AuthNZ.byok_runtime import ByokResolutionError
    from tldw_Server_API.app.core.RAG.rag_service import unified_pipeline

    base_doc = Document(
        id="base-1",
        content="Base result",
        metadata={},
        source=DataSource.MEDIA_DB,
        score=0.9,
    )
    provider_error = ByokResolutionError("credential_store_unavailable", "openai")

    calls = 0

    class _ExpansionRetriever:
        def __init__(self, *args, **kwargs):
            self.retrievers = {}

        async def retrieve(self, *args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                return [base_doc]
            raise provider_error

    async def _expand(*args, **kwargs):
        return ["expanded query one", "expanded query two"]

    monkeypatch.setattr(unified_pipeline, "MultiDatabaseRetriever", _ExpansionRetriever)
    monkeypatch.setattr(unified_pipeline, "multi_strategy_expansion", _expand)

    result = await unified_pipeline.unified_rag_pipeline(
        query="base query and second part",
        sources=["media_db"],
        credential_runtime=object(),
        expand_query=True,
        expansion_strategies=["synonym"],
        max_query_variations=2,
        enable_query_decomposition=True,
        max_subqueries=2,
        adaptive_hybrid_weights=False,
        enable_cache=False,
        enable_reranking=False,
        enable_generation=False,
    )

    assert [document["id"] for document in result.documents] == ["base-1"]
    assert calls == 2
    assert result.metadata["retrieval_coverage"]["retrieval_expansion"] == {
        "coverage": "degraded",
        "failure_code": "credential_store_unavailable",
    }
    assert "credential_store_unavailable" not in result.errors


@pytest.mark.unit
@pytest.mark.asyncio
async def test_standard_numeric_retry_provider_failure_degrades_and_stops(monkeypatch):
    from tldw_Server_API.app.core.Chat.Chat_Deps import ChatConfigurationError
    from tldw_Server_API.app.core.RAG.rag_service import unified_pipeline

    calls = 0
    base_doc = Document(
        id="numeric-base",
        content="The source value is 42.",
        metadata={},
        source=DataSource.MEDIA_DB,
        score=0.9,
    )
    provider_error = ChatConfigurationError(
        "secret endpoint configuration detail",
        provider="local_api",
        error_code="provider_configuration_invalid",
    )

    class _Retriever:
        def __init__(self, *args, **kwargs):
            self.retrievers = {}

        async def retrieve(self, *args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                return [base_doc]
            raise provider_error

    class _Generator:
        def __init__(self, *args, **kwargs):
            pass

        async def generate(self, **kwargs):
            return {"answer": "The safe completed answer contains 99."}

    numeric = SimpleNamespace(
        present=set(),
        missing=("99", "100"),
        union_source_numbers={"42"},
    )
    monkeypatch.setattr(unified_pipeline, "MultiDatabaseRetriever", _Retriever)
    monkeypatch.setattr(unified_pipeline, "AnswerGenerator", _Generator)
    monkeypatch.setattr(unified_pipeline, "check_numeric_fidelity", lambda *args, **kwargs: numeric)

    result = await unified_pipeline.unified_rag_pipeline(
        query="numeric retry",
        sources=["media_db"],
        media_db_path="media.db",
        credential_runtime=object(),
        adaptive_hybrid_weights=False,
        enable_cache=False,
        enable_reranking=False,
        enable_generation=True,
        enable_numeric_fidelity=True,
        numeric_fidelity_behavior="retry",
    )

    assert calls == 2
    assert result.generated_answer == "The safe completed answer contains 99."
    assert result.metadata["numeric_fidelity"]["embedding_coverage"] == "degraded"
    assert result.metadata["numeric_fidelity"]["failure_code"] == "provider_configuration_invalid"
    assert "secret endpoint configuration detail" not in repr(result.metadata)
    assert "secret endpoint configuration detail" not in repr(result.errors)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_additional_evidence_provider_failure_degrades_callback_locally(monkeypatch):
    from tldw_Server_API.app.core.Chat.Chat_Deps import ChatConfigurationError
    from tldw_Server_API.app.core.RAG.rag_service import unified_pipeline

    calls = 0
    base_doc = Document(
        id="evidence-base",
        content="Base evidence remains available.",
        metadata={},
        source=DataSource.MEDIA_DB,
        score=0.9,
    )
    provider_error = ChatConfigurationError(
        "secret additional retrieval detail",
        provider="local_api",
        error_code="provider_configuration_invalid",
    )

    class _Retriever:
        def __init__(self, *args, **kwargs):
            self.retrievers = {}

        async def retrieve(self, *args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                return [base_doc]
            raise provider_error

    class _Accumulator:
        def __init__(self, *args, **kwargs):
            pass

        async def accumulate(self, *, initial_results, retrieval_fn, **kwargs):
            assert await retrieval_fn("gap query", set()) == []
            assert await retrieval_fn("second gap query", set()) == []
            return SimpleNamespace(
                documents=list(initial_results),
                total_rounds=1,
                is_sufficient=False,
                sufficiency_reason="provider unavailable",
                metadata={"initial_docs": len(initial_results), "docs_added": 0},
            )

    monkeypatch.setattr(unified_pipeline, "MultiDatabaseRetriever", _Retriever)
    monkeypatch.setattr(unified_pipeline, "EvidenceAccumulator", _Accumulator)

    result = await unified_pipeline.unified_rag_pipeline(
        query="additional evidence",
        sources=["media_db"],
        media_db_path="media.db",
        credential_runtime=object(),
        adaptive_hybrid_weights=False,
        enable_cache=False,
        enable_reranking=False,
        enable_generation=False,
        enable_evidence_accumulation=True,
    )

    assert calls == 2
    assert [document["id"] for document in result.documents] == ["evidence-base"]
    assert result.metadata["retrieval_coverage"]["evidence_accumulation"] == {
        "coverage": "degraded",
        "failure_code": "provider_configuration_invalid",
    }
    assert "secret additional retrieval detail" not in repr(result.metadata)
    assert "secret additional retrieval detail" not in repr(result.errors)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_per_claim_provider_failure_is_not_swallowed_by_nested_fallback(monkeypatch):
    import tldw_Server_API.app.core.Claims_Extraction.claims_engine as claims_engine
    from tldw_Server_API.app.core.Chat.Chat_Deps import ChatConfigurationError
    from tldw_Server_API.app.core.RAG.rag_service import unified_pipeline

    class _RuntimeConfigurationError(ChatConfigurationError, RuntimeError):
        pass

    provider_error = _RuntimeConfigurationError(
        "secret per-claim retrieval detail",
        provider="local_api",
        error_code="provider_configuration_invalid",
    )
    base_doc = Document(
        id="claim-base",
        content="Base claim evidence remains available.",
        metadata={},
        source=DataSource.NOTES,
        score=0.9,
    )

    class _ExplodingNotesRetriever:
        calls = 0

        async def retrieve(self, *args, **kwargs):
            type(self).calls += 1
            raise provider_error

    class _Retriever:
        def __init__(self, *args, **kwargs):
            self.retrievers = {DataSource.NOTES: _ExplodingNotesRetriever()}

        async def retrieve(self, *args, **kwargs):
            return [base_doc]

    class _Generator:
        def __init__(self, *args, **kwargs):
            pass

        async def generate(self, **kwargs):
            return {"answer": "A completed claim-bearing answer."}

    class _ClaimsEngine:
        def __init__(self, _analyze):
            pass

        async def run(self, **kwargs):
            first = await kwargs["retrieve_fn"]("claim text")
            second = await kwargs["retrieve_fn"]("second claim text")
            assert first == [base_doc]
            assert second == [base_doc]
            return {"claims": [], "summary": {}, "verifications": []}

    class _Runtime:
        async def resolve(self, provider):
            return SimpleNamespace(
                provider=provider,
                api_key="runtime-key",
                app_config={},
            )

        async def mark_used(self, handle):
            raise AssertionError("claims provider was not called")

    monkeypatch.setattr(unified_pipeline, "MultiDatabaseRetriever", _Retriever)
    monkeypatch.setattr(unified_pipeline, "AnswerGenerator", _Generator)
    monkeypatch.setattr(unified_pipeline, "ClaimsEngine", _ClaimsEngine)
    monkeypatch.setattr(
        claims_engine,
        "_resolve_claims_llm_config",
        lambda: ("local_api", None, 0.1),
    )

    result = await unified_pipeline.unified_rag_pipeline(
        query="per-claim retrieval",
        sources=["notes"],
        notes_db_path="notes.db",
        credential_runtime=_Runtime(),
        adaptive_hybrid_weights=False,
        enable_cache=False,
        enable_reranking=False,
        enable_generation=True,
        enable_claims=True,
    )

    assert [document["id"] for document in result.documents] == ["claim-base"]
    assert _ExplodingNotesRetriever.calls == 1
    assert result.metadata["retrieval_coverage"]["per_claim"] == {
        "coverage": "degraded",
        "failure_code": "provider_configuration_invalid",
    }
    assert "secret per-claim retrieval detail" not in repr(result.metadata)
    assert "secret per-claim retrieval detail" not in repr(result.errors)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_batch_clustering_uses_runtime_credentials_for_hosted_embeddings(monkeypatch):
    from tldw_Server_API.app.core.Embeddings.Embeddings_Server import Embeddings_Create
    from tldw_Server_API.app.core.RAG.rag_service import unified_pipeline
    from tldw_Server_API.app.core.RAG.rag_service.unified_pipeline import UnifiedSearchResult

    runtime = _CredentialRuntime("openai", base_url="https://batch-embeddings.example/v1")
    captured: dict[str, Any] = {}
    config = {
        "embedding_config": {
            "default_model_id": "openai:text-embedding-3-small",
            "models": {
                "openai:text-embedding-3-small": SimpleNamespace(provider="openai"),
            },
        }
    }

    def fake_create(texts, user_app_config, model_id_override=None, **kwargs):
        captured["kwargs"] = kwargs
        return [[1.0, 0.0], [0.0, 1.0]]

    async def fake_pipeline(**kwargs):
        return UnifiedSearchResult(documents=[], query=kwargs["query"])

    monkeypatch.setattr(unified_pipeline, "_shared_is_test_mode", lambda: False)
    monkeypatch.delenv("RAG_BATCH_DISABLE_CLUSTERING", raising=False)
    monkeypatch.setattr(Embeddings_Create, "get_embedding_config", lambda: config)
    monkeypatch.setattr(Embeddings_Create, "create_embeddings_batch", fake_create)
    monkeypatch.setattr(unified_pipeline, "unified_rag_pipeline", fake_pipeline)

    results = await unified_batch_pipeline(
        ["first question", "second question"],
        credential_runtime=runtime,
    )

    assert [result.query for result in results] == ["first question", "second question"]
    assert runtime.resolved == ["openai"]
    assert runtime.marked == [runtime.handle]
    assert captured["kwargs"] == {
        "api_key_override": "runtime-embedding-key",
        "base_url_override": "https://batch-embeddings.example/v1",
        "credentials_resolved": True,
    }


@pytest.mark.unit
@pytest.mark.asyncio
async def test_batch_clustering_runtime_local_api_uses_endpoint_without_key(monkeypatch):
    from tldw_Server_API.app.core.Embeddings.Embeddings_Server import Embeddings_Create
    from tldw_Server_API.app.core.RAG.rag_service import unified_pipeline
    from tldw_Server_API.app.core.RAG.rag_service.unified_pipeline import UnifiedSearchResult

    runtime = _CredentialRuntime("openai")
    captured: dict[str, Any] = {}
    config = {
        "embedding_config": {
            "default_model_id": "local_api:batch-model",
            "models": {
                "local_api:batch-model": SimpleNamespace(
                    provider="local_api",
                    api_url="https://batch-local.example/embeddings",
                    api_key="configured-key-must-not-be-used",
                ),
            },
        }
    }

    def fake_create(texts, user_app_config, model_id_override=None, **kwargs):
        captured["kwargs"] = kwargs
        return [[1.0, 0.0], [0.0, 1.0]]

    async def fake_pipeline(**kwargs):
        return UnifiedSearchResult(documents=[], query=kwargs["query"])

    monkeypatch.setattr(unified_pipeline, "_shared_is_test_mode", lambda: False)
    monkeypatch.delenv("RAG_BATCH_DISABLE_CLUSTERING", raising=False)
    monkeypatch.setattr(Embeddings_Create, "get_embedding_config", lambda: config)
    monkeypatch.setattr(Embeddings_Create, "create_embeddings_batch", fake_create)
    monkeypatch.setattr(unified_pipeline, "unified_rag_pipeline", fake_pipeline)

    await unified_batch_pipeline(
        ["first question", "second question"],
        credential_runtime=runtime,
    )

    assert runtime.resolved == []
    assert runtime.marked == []
    assert captured["kwargs"] == {
        "api_key_override": None,
        "base_url_override": "https://batch-local.example/embeddings",
        "credentials_resolved": True,
    }
