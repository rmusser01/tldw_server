import asyncio
import threading
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

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
    await asyncio.sleep(0.01)
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
