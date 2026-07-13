"""Sanitizer coverage for advanced retrieval fallback logs."""

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from tldw_Server_API.app.core.RAG.rag_service import advanced_retrieval as ar
from tldw_Server_API.app.core.RAG.rag_service import database_retrievers as dr
from tldw_Server_API.app.core.RAG.rag_service.types import Document

pytestmark = pytest.mark.unit


class _LoggerStub:
    def __init__(self):
        self.warnings: list[str] = []

    def warning(self, message):
        self.warnings.append(str(message))


class _QueryEmbeddingFailureService:
    async def create_embedding(self, *, text, user_id=None):
        raise RuntimeError("query embedding failed for /private/rag-query.db?token=secret")


class _SpanEmbeddingFailureService:
    async def create_embedding(self, *, text, user_id=None):
        return [1.0, 0.0]

    async def create_embeddings_batch(self, batch, *, user_id=None):
        raise RuntimeError("span embedding failed for /private/rag-span.db?token=secret")


class _CredentialRuntime:
    def __init__(self, provider: str):
        section = "openai_api" if provider == "openai" else "huggingface_api"
        self.handle = SimpleNamespace(
            provider=provider,
            api_key="runtime-embedding-key",
            app_config={section: {"api_url": "https://user-embeddings.example/v1"}},
            credentials_resolved=True,
        )
        self.resolved: list[str] = []
        self.marked: list[Any] = []

    async def resolve(self, provider: str):
        self.resolved.append(provider)
        return self.handle

    async def mark_used(self, handle: Any) -> None:
        if handle not in self.marked:
            self.marked.append(handle)


def _docs() -> list[Document]:
    return [
        Document(id="doc-1", content="alpha beta gamma", metadata={}, score=0.5),
        Document(id="doc-2", content="delta epsilon zeta", metadata={}, score=0.4),
    ]


def _assert_log_is_sanitized(logger_stub: _LoggerStub, expected_message: str) -> None:
    assert logger_stub.warnings == [expected_message]
    joined = "\n".join(logger_stub.warnings)
    assert "/private/" not in joined
    assert "secret" not in joined
    assert "rag-query.db" not in joined
    assert "rag-span.db" not in joined


@pytest.mark.asyncio
async def test_query_embedding_fallback_warning_omits_backend_exception(monkeypatch):
    logger_stub = _LoggerStub()
    documents = _docs()

    monkeypatch.setattr(ar, "logger", logger_stub)
    monkeypatch.setattr(ar, "get_async_embedding_service", lambda: _QueryEmbeddingFailureService())

    result = await ar.apply_multi_vector_passages("private query", documents)

    assert result is documents
    _assert_log_is_sanitized(
        logger_stub,
        "Query embedding failed; skipping multi-vector passages",
    )


@pytest.mark.asyncio
async def test_optional_embedding_auth_failure_reports_bounded_credential_code(monkeypatch):
    from tldw_Server_API.app.core.Embeddings.async_embeddings import EmbeddingProviderError

    class _AuthFailureService:
        async def create_embedding(self, **kwargs):
            raise EmbeddingProviderError("openai", code="authentication", status_code=401)

    metadata: dict[str, Any] = {}
    monkeypatch.setattr(ar, "get_async_embedding_service", lambda: _AuthFailureService())

    documents = _docs()
    result = await ar.apply_multi_vector_passages(
        "private query",
        documents,
        stage_metadata=metadata,
    )

    assert result is documents
    assert metadata == {
        "embedding_coverage": "degraded",
        "failure_code": "invalid_provider_credentials",
    }


@pytest.mark.asyncio
async def test_span_embedding_fallback_warning_omits_backend_exception(monkeypatch):
    logger_stub = _LoggerStub()
    documents = _docs()

    monkeypatch.setattr(ar, "logger", logger_stub)
    monkeypatch.setattr(ar, "get_async_embedding_service", lambda: _SpanEmbeddingFailureService())

    result = await ar.apply_multi_vector_passages("alpha", documents)

    assert result is documents
    _assert_log_is_sanitized(
        logger_stub,
        "Span embeddings failed; skipping multi-vector passages",
    )


@pytest.mark.asyncio
async def test_multi_vector_uses_runtime_credentials_for_hosted_huggingface(monkeypatch):
    captured: list[dict[str, Any]] = []
    runtime = _CredentialRuntime("huggingface")

    class _HostedService:
        config = SimpleNamespace(
            default_provider="huggingface",
            default_model="sentence-transformers/runtime-model",
        )

        def _resolve_provider_alias(self, provider: str) -> str:
            return provider

        async def create_embedding(self, **kwargs):
            callback = kwargs.pop("on_provider_success", None)
            captured.append(kwargs)
            embedding = [1.0, 0.0]
            if callback is not None:
                await callback()
            return embedding

        async def create_embeddings_batch(self, texts, **kwargs):
            callback = kwargs.pop("on_provider_success", None)
            captured.append({"texts": texts, **kwargs})
            embeddings = [[1.0, 0.0] for _ in texts]
            if callback is not None:
                await callback()
            return embeddings

    monkeypatch.setattr(ar, "get_async_embedding_service", lambda: _HostedService())

    result = await ar.apply_multi_vector_passages(
        "alpha",
        _docs(),
        credential_runtime=runtime,
    )

    assert result
    assert runtime.resolved == ["huggingface"]
    assert runtime.marked == [runtime.handle]
    expected = {
        "provider": "huggingface",
        "model": "sentence-transformers/runtime-model",
        "api_key_override": "runtime-embedding-key",
        "base_url_override": "https://user-embeddings.example/v1",
        "credentials_resolved": True,
    }
    assert captured[0] == {"text": "alpha", "user_id": None, **expected}
    assert captured[1]["user_id"] is None
    assert {key: captured[1][key] for key in expected} == expected


@pytest.mark.asyncio
async def test_multi_vector_does_not_resolve_local_embedding_provider(monkeypatch):
    runtime = _CredentialRuntime("openai")
    captured: list[dict[str, Any]] = []

    class _LocalService:
        config = SimpleNamespace(default_provider="local", default_model="local-model")

        def _resolve_provider_alias(self, provider: str) -> str:
            return provider

        async def create_embedding(self, **kwargs):
            captured.append(kwargs)
            return [1.0, 0.0]

        async def create_embeddings_batch(self, texts, **kwargs):
            captured.append({"texts": texts, **kwargs})
            return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr(ar, "get_async_embedding_service", lambda: _LocalService())

    await ar.apply_multi_vector_passages("alpha", _docs(), credential_runtime=runtime)

    assert runtime.resolved == []
    assert runtime.marked == []
    assert captured[0]["credentials_resolved"] is True
    assert "api_key_override" not in captured[0]
    assert captured[1]["credentials_resolved"] is True


@pytest.mark.asyncio
async def test_multi_vector_missing_hosted_key_degrades_with_bounded_metadata(monkeypatch):
    class _MissingKeyRuntime:
        async def resolve(self, provider: str):
            return SimpleNamespace(provider=provider, api_key=None, app_config={})

    class _HostedService:
        config = SimpleNamespace(default_provider="openai", default_model="embed-model")

        def _resolve_provider_alias(self, provider: str) -> str:
            return provider

    metadata: dict[str, Any] = {}
    documents = _docs()
    monkeypatch.setattr(ar, "get_async_embedding_service", lambda: _HostedService())

    result = await ar.apply_multi_vector_passages(
        "alpha",
        documents,
        credential_runtime=_MissingKeyRuntime(),
        stage_metadata=metadata,
    )

    assert result is documents
    assert metadata == {
        "embedding_coverage": "degraded",
        "failure_code": "missing_provider_credentials",
    }


@pytest.mark.asyncio
async def test_multi_vector_runtime_failure_sets_bounded_coverage_metadata(monkeypatch):
    from tldw_Server_API.app.core.AuthNZ.byok_runtime import ByokResolutionError

    class _FailingRuntime:
        async def resolve(self, provider: str):
            raise ByokResolutionError("invalid_provider_credentials", provider)

    class _HostedService:
        config = SimpleNamespace(default_provider="openai", default_model="embed-model")

        def _resolve_provider_alias(self, provider: str) -> str:
            return provider

    metadata: dict[str, Any] = {}
    documents = _docs()
    monkeypatch.setattr(ar, "get_async_embedding_service", lambda: _HostedService())

    result = await ar.apply_multi_vector_passages(
        "alpha",
        documents,
        credential_runtime=_FailingRuntime(),
        stage_metadata=metadata,
    )

    assert result is documents
    assert metadata == {
        "embedding_coverage": "degraded",
        "failure_code": "invalid_provider_credentials",
    }


@pytest.mark.asyncio
async def test_multi_vector_marks_successful_query_use_before_span_degradation(monkeypatch):
    runtime = _CredentialRuntime("openai")

    class _PartialHostedService:
        config = SimpleNamespace(default_provider="openai", default_model="embed-model")

        def _resolve_provider_alias(self, provider: str) -> str:
            return provider

        async def create_embedding(self, **kwargs):
            embedding = [1.0, 0.0]
            callback = kwargs.get("on_provider_success")
            if callback is not None:
                await callback()
            return embedding

        async def create_embeddings_batch(self, texts, **kwargs):
            raise RuntimeError("span provider failed with secret=runtime-embedding-key")

    metadata: dict[str, Any] = {}
    documents = _docs()
    monkeypatch.setattr(ar, "get_async_embedding_service", lambda: _PartialHostedService())

    result = await ar.apply_multi_vector_passages(
        "alpha",
        documents,
        credential_runtime=runtime,
        stage_metadata=metadata,
    )

    assert result is documents
    assert runtime.marked == [runtime.handle]
    assert metadata == {
        "embedding_coverage": "degraded",
        "failure_code": "provider_unavailable",
    }


@pytest.mark.asyncio
async def test_multi_vector_full_cache_hit_does_not_mark_runtime_used(monkeypatch):
    runtime = _CredentialRuntime("openai")

    class _CachedHostedService:
        config = SimpleNamespace(default_provider="openai", default_model="embed-model")

        def _resolve_provider_alias(self, provider: str) -> str:
            return provider

        async def create_embedding(self, **kwargs):
            return [1.0, 0.0]

        async def create_embeddings_batch(self, texts, **kwargs):
            return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr(ar, "get_async_embedding_service", lambda: _CachedHostedService())

    result = await ar.apply_multi_vector_passages(
        "alpha",
        _docs(),
        credential_runtime=runtime,
    )

    assert result
    assert runtime.marked == []


@pytest.mark.asyncio
async def test_multi_vector_span_dispatch_marks_after_query_cache_hit(monkeypatch):
    runtime = _CredentialRuntime("openai")

    class _PartiallyCachedHostedService:
        config = SimpleNamespace(default_provider="openai", default_model="embed-model")

        def _resolve_provider_alias(self, provider: str) -> str:
            return provider

        async def create_embedding(self, **kwargs):
            return [1.0, 0.0]

        async def create_embeddings_batch(self, texts, **kwargs):
            embeddings = [[1.0, 0.0] for _ in texts]
            callback = kwargs.get("on_provider_success")
            if callback is not None:
                await callback()
            return embeddings

    monkeypatch.setattr(ar, "get_async_embedding_service", lambda: _PartiallyCachedHostedService())

    result = await ar.apply_multi_vector_passages(
        "alpha",
        _docs(),
        credential_runtime=runtime,
    )

    assert result
    assert runtime.marked == [runtime.handle]


@pytest.mark.asyncio
async def test_multi_vector_credential_resolution_cancellation_propagates(monkeypatch):
    class _CancelledRuntime:
        async def resolve(self, provider: str):
            raise asyncio.CancelledError

    class _HostedService:
        config = SimpleNamespace(default_provider="openai", default_model="embed-model")

        def _resolve_provider_alias(self, provider: str) -> str:
            return provider

    monkeypatch.setattr(ar, "get_async_embedding_service", lambda: _HostedService())

    with pytest.raises(asyncio.CancelledError):
        await ar.apply_multi_vector_passages(
            "alpha",
            _docs(),
            credential_runtime=_CancelledRuntime(),
        )


@pytest.mark.asyncio
async def test_media_retriever_hosted_query_embedding_uses_runtime_credentials(monkeypatch):
    from tldw_Server_API.app.core.Embeddings.Embeddings_Server import Embeddings_Create

    captured: dict[str, Any] = {}
    runtime = _CredentialRuntime("openai")
    config = {
        "embedding_config": {
            "default_model_id": "openai:text-embedding-3-small",
            "models": {
                "openai:text-embedding-3-small": SimpleNamespace(provider="openai"),
            },
        }
    }

    class _VectorStore:
        _initialized = True

        async def search(self, **kwargs):
            return [SimpleNamespace(id="doc", content="alpha", metadata={}, score=0.9)]

    def fake_create(texts, user_app_config, model_id_override=None, **kwargs):
        captured.update(
            texts=texts,
            user_app_config=user_app_config,
            model_id_override=model_id_override,
            kwargs=kwargs,
        )
        return [[1.0, 0.0]]

    retriever = object.__new__(dr.MediaDBRetriever)
    retriever.vector_store = _VectorStore()
    retriever.user_id = "42"
    retriever.config = dr.RetrievalConfig(max_results=3, use_vector=True, use_fts=False)
    retriever.credential_runtime = runtime
    monkeypatch.setattr(Embeddings_Create, "get_embedding_config", lambda: config)
    monkeypatch.setattr(Embeddings_Create, "create_embeddings_batch", fake_create)

    documents = await retriever._retrieve_vector("alpha", index_namespace="runtime-index")

    assert [document.id for document in documents] == ["doc"]
    assert runtime.resolved == ["openai"]
    assert runtime.marked == [runtime.handle]
    assert captured["kwargs"] == {
        "api_key_override": "runtime-embedding-key",
        "base_url_override": "https://user-embeddings.example/v1",
        "credentials_resolved": True,
    }


@pytest.mark.asyncio
async def test_media_retriever_precomputed_query_vector_skips_runtime_resolution():
    runtime = _CredentialRuntime("openai")

    class _VectorStore:
        _initialized = True

        async def search(self, **kwargs):
            assert kwargs["query_vector"] == [0.5, 0.5]
            return []

    retriever = object.__new__(dr.MediaDBRetriever)
    retriever.vector_store = _VectorStore()
    retriever.user_id = "42"
    retriever.config = dr.RetrievalConfig(max_results=3, use_vector=True, use_fts=False)
    retriever.credential_runtime = runtime

    await retriever._retrieve_vector(
        "alpha",
        index_namespace="runtime-index",
        query_vector=[0.5, 0.5],
    )

    assert runtime.resolved == []
    assert runtime.marked == []


@pytest.mark.asyncio
async def test_required_media_embedding_auth_failure_raises_bounded_chat_error(monkeypatch):
    from tldw_Server_API.app.core.Chat.Chat_Deps import ChatAuthenticationError
    from tldw_Server_API.app.core.Embeddings.async_embeddings import EmbeddingProviderError
    from tldw_Server_API.app.core.Embeddings.Embeddings_Server import Embeddings_Create

    runtime = _CredentialRuntime("openai")
    config = {
        "embedding_config": {
            "default_model_id": "openai:text-embedding-3-small",
            "models": {
                "openai:text-embedding-3-small": SimpleNamespace(provider="openai"),
            },
        }
    }

    class _VectorStore:
        _initialized = True

    def fail_embedding(*args, **kwargs):
        raise EmbeddingProviderError("openai", code="authentication", status_code=401)

    retriever = object.__new__(dr.MediaDBRetriever)
    retriever.vector_store = _VectorStore()
    retriever.user_id = "42"
    retriever.config = dr.RetrievalConfig(max_results=3, use_vector=True, use_fts=False)
    retriever.credential_runtime = runtime
    monkeypatch.setattr(Embeddings_Create, "get_embedding_config", lambda: config)
    monkeypatch.setattr(Embeddings_Create, "create_embeddings_batch", fail_embedding)

    with pytest.raises(ChatAuthenticationError) as exc_info:
        await retriever._retrieve_vector("alpha", index_namespace="runtime-index")

    assert exc_info.value.provider == "openai"
    assert "runtime-embedding-key" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_required_media_missing_hosted_key_raises_bounded_configuration_error(monkeypatch):
    from tldw_Server_API.app.core.Chat.Chat_Deps import ChatConfigurationError
    from tldw_Server_API.app.core.Embeddings.Embeddings_Server import Embeddings_Create

    class _MissingKeyRuntime:
        async def resolve(self, provider: str):
            return SimpleNamespace(
                provider=provider,
                api_key=None,
                app_config={"openai_api": {"api_key": "server-secret-must-not-leak"}},
            )

    config = {
        "embedding_config": {
            "default_model_id": "openai:text-embedding-3-small",
            "models": {
                "openai:text-embedding-3-small": SimpleNamespace(provider="openai"),
            },
        }
    }

    retriever = object.__new__(dr.MediaDBRetriever)
    retriever.vector_store = SimpleNamespace(_initialized=True)
    retriever.user_id = "42"
    retriever.config = dr.RetrievalConfig(max_results=3, use_vector=True, use_fts=False)
    retriever.credential_runtime = _MissingKeyRuntime()
    monkeypatch.setattr(Embeddings_Create, "get_embedding_config", lambda: config)

    with pytest.raises(ChatConfigurationError) as exc_info:
        await retriever._retrieve_vector("alpha", index_namespace="runtime-index")

    assert exc_info.value.error_code == "missing_provider_credentials"
    assert "server-secret-must-not-leak" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_required_media_missing_local_api_endpoint_fails_closed(monkeypatch):
    from tldw_Server_API.app.core.Chat.Chat_Deps import ChatConfigurationError
    from tldw_Server_API.app.core.Embeddings.Embeddings_Server import Embeddings_Create

    config = {
        "embedding_config": {
            "default_model_id": "local_api:missing-endpoint",
            "models": {
                "local_api:missing-endpoint": SimpleNamespace(
                    provider="local_api",
                    api_key="configured-key-must-not-be-used",
                ),
            },
        }
    }

    retriever = object.__new__(dr.MediaDBRetriever)
    retriever.vector_store = SimpleNamespace(_initialized=True)
    retriever.user_id = "42"
    retriever.config = dr.RetrievalConfig(max_results=3, use_vector=True, use_fts=False)
    retriever.credential_runtime = object()
    monkeypatch.setattr(Embeddings_Create, "get_embedding_config", lambda: config)

    with pytest.raises(ChatConfigurationError) as exc_info:
        await retriever._retrieve_vector("alpha", index_namespace="runtime-index")

    assert exc_info.value.error_code == "provider_configuration_invalid"
    assert "configured-key-must-not-be-used" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_multi_database_retriever_propagates_required_media_auth_failure():
    from tldw_Server_API.app.core.Chat.Chat_Deps import ChatAuthenticationError
    from tldw_Server_API.app.core.RAG.rag_service.types import DataSource

    class _FailingMediaRetriever(dr.MediaDBRetriever):
        def __init__(self):
            self.config = dr.RetrievalConfig()

        async def _retrieve_vector(self, *args, **kwargs):
            raise ChatAuthenticationError(
                "Embedding provider authentication failed.",
                provider="openai",
            )

    class _SuccessfulRetriever:
        async def retrieve(self, query):
            return [Document(id="note-1", content="partial", metadata={}, score=0.9)]

    retriever = object.__new__(dr.MultiDatabaseRetriever)
    retriever.retrievers = {
        DataSource.MEDIA_DB: _FailingMediaRetriever(),
        DataSource.NOTES: _SuccessfulRetriever(),
    }
    retriever.credential_runtime = object()
    config = dr.RetrievalConfig(max_results=5, use_vector=True, use_fts=False)

    with pytest.raises(ChatAuthenticationError) as exc_info:
        await retriever.retrieve("alpha", config=config)

    assert exc_info.value.provider == "openai"
    assert str(exc_info.value) == "Embedding provider authentication failed."


@pytest.mark.asyncio
async def test_multi_database_retriever_keeps_legacy_partial_success_for_typed_failure():
    from tldw_Server_API.app.core.Chat.Chat_Deps import ChatAuthenticationError
    from tldw_Server_API.app.core.RAG.rag_service.types import DataSource

    class _FailingMediaRetriever(dr.MediaDBRetriever):
        def __init__(self):
            self.config = dr.RetrievalConfig()

        async def _retrieve_vector(self, *args, **kwargs):
            raise ChatAuthenticationError("legacy provider failure", provider="openai")

    class _SuccessfulRetriever:
        async def retrieve(self, query):
            return [Document(id="note-1", content="partial", metadata={}, score=0.9)]

    retriever = object.__new__(dr.MultiDatabaseRetriever)
    retriever.retrievers = {
        DataSource.MEDIA_DB: _FailingMediaRetriever(),
        DataSource.NOTES: _SuccessfulRetriever(),
    }
    retriever.credential_runtime = None

    documents = await retriever.retrieve(
        "alpha",
        config=dr.RetrievalConfig(max_results=5, use_vector=True, use_fts=False),
    )

    assert [document.id for document in documents] == ["note-1"]


@pytest.mark.asyncio
async def test_multi_database_retriever_keeps_ordinary_partial_source_success():
    from tldw_Server_API.app.core.RAG.rag_service.types import DataSource

    class _FailingRetriever:
        async def retrieve(self, query):
            raise RuntimeError("ordinary source failure")

    class _SuccessfulRetriever:
        async def retrieve(self, query):
            return [Document(id="note-1", content="partial", metadata={}, score=0.9)]

    retriever = object.__new__(dr.MultiDatabaseRetriever)
    retriever.retrievers = {
        DataSource.PROMPTS: _FailingRetriever(),
        DataSource.NOTES: _SuccessfulRetriever(),
    }

    documents = await retriever.retrieve("alpha")

    assert [document.id for document in documents] == ["note-1"]


@pytest.mark.asyncio
async def test_media_scoped_model_override_resolves_its_actual_hosted_provider(monkeypatch):
    from tldw_Server_API.app.core.Embeddings.Embeddings_Server import Embeddings_Create

    runtime = _CredentialRuntime("openai")
    captured: dict[str, Any] = {}
    config = {
        "embedding_config": {
            "default_model_id": "huggingface:local-model",
            "models": {
                "huggingface:local-model": SimpleNamespace(provider="huggingface"),
                "openai:text-embedding-3-small": SimpleNamespace(provider="openai"),
            },
        }
    }

    class _VectorStore:
        _initialized = True

        async def search(self, **kwargs):
            return []

    def fake_create(texts, user_app_config, model_id_override=None, **kwargs):
        captured["model_id_override"] = model_id_override
        captured["kwargs"] = kwargs
        return [[1.0, 0.0]]

    retriever = object.__new__(dr.MediaDBRetriever)
    retriever.vector_store = _VectorStore()
    retriever.user_id = "42"
    retriever.config = dr.RetrievalConfig(max_results=3, use_vector=True, use_fts=False)
    retriever.credential_runtime = runtime
    monkeypatch.setattr(
        retriever,
        "_resolve_scoped_query_embedding_override",
        lambda **kwargs: "openai:text-embedding-3-small",
    )
    monkeypatch.setattr(Embeddings_Create, "get_embedding_config", lambda: config)
    monkeypatch.setattr(Embeddings_Create, "create_embeddings_batch", fake_create)

    await retriever._retrieve_vector(
        "alpha",
        index_namespace="runtime-index",
        allowed_media_ids=[7],
    )

    assert runtime.resolved == ["openai"]
    assert captured["model_id_override"] == "openai:text-embedding-3-small"
    assert captured["kwargs"]["credentials_resolved"] is True


@pytest.mark.asyncio
async def test_media_scoped_local_api_uses_exact_endpoint_without_configured_key(monkeypatch):
    from tldw_Server_API.app.core.Embeddings.Embeddings_Server import Embeddings_Create

    runtime = _CredentialRuntime("openai")
    captured: dict[str, Any] = {}
    config = {
        "embedding_config": {
            "default_model_id": "local_api:default-model",
            "models": {
                "local_api:default-model": SimpleNamespace(
                    provider="local_api",
                    api_url="https://default-local.example/embeddings",
                    api_key="default-key-must-not-be-used",
                ),
                "local_api:scoped-model": SimpleNamespace(
                    provider="local_api",
                    api_url="https://scoped-local.example/embeddings",
                    api_key="scoped-key-must-not-be-used",
                ),
            },
        }
    }

    class _VectorStore:
        _initialized = True

        async def search(self, **kwargs):
            return []

    def fake_create(texts, user_app_config, model_id_override=None, **kwargs):
        captured["model_id_override"] = model_id_override
        captured["kwargs"] = kwargs
        return [[1.0, 0.0]]

    retriever = object.__new__(dr.MediaDBRetriever)
    retriever.vector_store = _VectorStore()
    retriever.user_id = "42"
    retriever.config = dr.RetrievalConfig(max_results=3, use_vector=True, use_fts=False)
    retriever.credential_runtime = runtime
    monkeypatch.setattr(
        retriever,
        "_resolve_scoped_query_embedding_override",
        lambda **kwargs: "local_api:scoped-model",
    )
    monkeypatch.setattr(Embeddings_Create, "get_embedding_config", lambda: config)
    monkeypatch.setattr(Embeddings_Create, "create_embeddings_batch", fake_create)

    await retriever._retrieve_vector(
        "alpha",
        index_namespace="runtime-index",
        allowed_media_ids=[7],
    )

    assert runtime.resolved == []
    assert runtime.marked == []
    assert captured["model_id_override"] == "local_api:scoped-model"
    assert captured["kwargs"] == {
        "api_key_override": None,
        "base_url_override": "https://scoped-local.example/embeddings",
        "credentials_resolved": True,
    }


@pytest.mark.asyncio
async def test_direct_media_retriever_keeps_legacy_embedding_provider_failure(monkeypatch):
    from tldw_Server_API.app.core.Embeddings.async_embeddings import EmbeddingProviderError
    from tldw_Server_API.app.core.Embeddings.Embeddings_Server import Embeddings_Create

    config = {
        "embedding_config": {
            "default_model_id": "openai:text-embedding-3-small",
            "models": {
                "openai:text-embedding-3-small": SimpleNamespace(provider="openai"),
            },
        }
    }

    def fail_embedding(*args, **kwargs):
        raise EmbeddingProviderError("openai", code="authentication", status_code=401)

    retriever = object.__new__(dr.MediaDBRetriever)
    retriever.vector_store = SimpleNamespace(_initialized=True)
    retriever.user_id = "42"
    retriever.config = dr.RetrievalConfig(max_results=3, use_vector=True, use_fts=False)
    retriever.credential_runtime = None
    monkeypatch.setattr(Embeddings_Create, "get_embedding_config", lambda: config)
    monkeypatch.setattr(Embeddings_Create, "create_embeddings_batch", fail_embedding)

    assert await retriever._retrieve_vector("alpha", index_namespace="legacy-index") == []
