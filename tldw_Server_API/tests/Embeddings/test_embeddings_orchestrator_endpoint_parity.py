from __future__ import annotations

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException, status
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_request_user
from tldw_Server_API.app.api.v1.schemas.embeddings_models import (
    CreateEmbeddingResponse,
    EmbeddingData,
    EmbeddingUsage,
)
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
from tldw_Server_API.app.core.Embeddings.orchestrator import EmbeddingExecutorOutput
from tldw_Server_API.app.core.Embeddings.request_types import (
    EmbeddingExecutionError,
    EmbeddingExecutionResult,
    EmbeddingInputError,
    EmbeddingProviderError,
    EmbeddingRateLimitError,
)
from tldw_Server_API.app.main import app


@pytest.fixture
def client(monkeypatch):
    original_overrides = dict(app.dependency_overrides)
    monkeypatch.setenv("TESTING", "true")
    monkeypatch.setenv("AUTO_DOWNLOAD_MODELS", "false")
    monkeypatch.delenv("EMBEDDINGS_ORCHESTRATOR_ENABLED", raising=False)

    async def override_user():
        return User(
            id=1,
            username="embedding-user",
            email="embedding-user@example.test",
            is_active=True,
            is_admin=False,
        )

    app.dependency_overrides[get_request_user] = override_user

    with TestClient(app) as test_client:
        csrf_token = f"test-csrf-{uuid.uuid4().hex}"
        test_client.cookies.set("csrf_token", csrf_token)
        test_client.headers["X-CSRF-Token"] = csrf_token
        test_client.headers["Authorization"] = "Bearer test-api-key"
        yield test_client

    app.dependency_overrides.clear()
    app.dependency_overrides.update(original_overrides)


def _ok_response(model: str = "patched-model") -> CreateEmbeddingResponse:
    return CreateEmbeddingResponse(
        data=[EmbeddingData(embedding=[0.1, 0.2], index=0)],
        model=model,
        usage=EmbeddingUsage(prompt_tokens=2, total_tokens=2),
    )


class FakePrepared:
    def __init__(self, total_tokens: int = 2) -> None:
        self.normalized_input = SimpleNamespace(total_tokens=total_tokens)


class FakeOrchestrator:
    def __init__(self, *, result=None, prepare_error=None, execute_error=None) -> None:
        self.result = result
        self.prepare_error = prepare_error
        self.execute_error = execute_error
        self.prepare_calls = []
        self.execute_calls = []

    def prepare(self, raw_input, context):
        self.prepare_calls.append((raw_input, context))
        if self.prepare_error is not None:
            raise self.prepare_error
        return FakePrepared(total_tokens=3)

    async def execute(self, prepared):
        self.execute_calls.append(prepared)
        if self.execute_error is not None:
            raise self.execute_error
        return self.result or EmbeddingExecutionResult(
            vectors=[[0.25, 0.75]],
            provider="huggingface",
            model="sentence-transformers/all-MiniLM-L6-v2",
            prompt_tokens=3,
            total_tokens=3,
            cache_hits=0,
            cache_misses=1,
        )


def _user() -> User:
    return User(
        id=1,
        username="embedding-user",
        email="embedding-user@example.test",
        is_active=True,
        is_admin=False,
    )


def _request():
    return SimpleNamespace()


class FakeCredentials:
    def __init__(
        self,
        *,
        api_key: str | None,
        source: str = "server",
        auth_source: str | None = None,
    ) -> None:
        self.api_key = api_key
        self.source = source
        self.auth_source = auth_source
        self.touch_last_used = AsyncMock()


def test_flag_unset_calls_legacy_path(client, monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import embeddings_v5_production_enhanced as mod

    legacy = AsyncMock(return_value=_ok_response("legacy-model"))
    orchestrator = AsyncMock(side_effect=AssertionError("orchestrator should not be called"))
    monkeypatch.setattr(mod, "_create_embedding_legacy", legacy, raising=False)
    monkeypatch.setattr(mod, "_create_embedding_with_orchestrator", orchestrator, raising=False)

    response = client.post(
        "/api/v1/embeddings",
        json={"model": "text-embedding-3-small", "input": "hello world"},
    )

    assert response.status_code == status.HTTP_200_OK
    assert response.json()["model"] == "legacy-model"
    assert legacy.await_count == 1
    assert orchestrator.await_count == 0


def test_flag_true_calls_orchestrator_path(client, monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import embeddings_v5_production_enhanced as mod

    monkeypatch.setenv("EMBEDDINGS_ORCHESTRATOR_ENABLED", "true")
    legacy = AsyncMock(side_effect=AssertionError("legacy should not be called"))
    orchestrator = AsyncMock(return_value=_ok_response("orchestrator-model"))
    monkeypatch.setattr(mod, "_create_embedding_legacy", legacy, raising=False)
    monkeypatch.setattr(mod, "_create_embedding_with_orchestrator", orchestrator, raising=False)

    response = client.post(
        "/api/v1/embeddings",
        json={"model": "text-embedding-3-small", "input": "hello world"},
    )

    assert response.status_code == status.HTTP_200_OK
    assert response.json()["model"] == "orchestrator-model"
    assert legacy.await_count == 0
    assert orchestrator.await_count == 1


def test_orchestrator_input_error_maps_to_current_400_shape(client, monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import embeddings_v5_production_enhanced as mod

    monkeypatch.setenv("EMBEDDINGS_ORCHESTRATOR_ENABLED", "true")
    fake_orchestrator = FakeOrchestrator(
        prepare_error=EmbeddingInputError("empty_input", "Input cannot be empty")
    )
    monkeypatch.setattr(
        mod,
        "_build_embedding_request_orchestrator",
        lambda *_args, **_kwargs: fake_orchestrator,
        raising=False,
    )

    response = client.post(
        "/api/v1/embeddings",
        json={"model": "text-embedding-3-small", "input": "   "},
    )

    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert response.json()["detail"] == "Input cannot be empty"


def test_orchestrator_token_limit_error_preserves_top_level_shape(client, monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import embeddings_v5_production_enhanced as mod

    monkeypatch.setenv("EMBEDDINGS_ORCHESTRATOR_ENABLED", "true")
    fake_orchestrator = FakeOrchestrator(
        prepare_error=EmbeddingInputError(
            "input_too_long",
            "One or more inputs exceed max tokens 2 for model text-embedding-3-small",
            details=[{"index": 0, "tokens": 3}],
        )
    )
    monkeypatch.setattr(
        mod,
        "_build_embedding_request_orchestrator",
        lambda *_args, **_kwargs: fake_orchestrator,
        raising=False,
    )

    response = client.post(
        "/api/v1/embeddings",
        json={"model": "text-embedding-3-small", "input": "three token input"},
    )

    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert response.json()["error"] == "input_too_long"
    assert response.json()["details"] == [{"index": 0, "tokens": 3}]


def test_orchestrator_missing_credentials_preserves_503_detail_dict(client, monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import embeddings_v5_production_enhanced as mod

    monkeypatch.setenv("EMBEDDINGS_ORCHESTRATOR_ENABLED", "true")
    fake_orchestrator = FakeOrchestrator(
        execute_error=EmbeddingProviderError(
            "missing_provider_credentials",
            "Embeddings provider 'cohere' requires an API key.",
            provider="cohere",
            model="embed-english-v3.0",
        )
    )
    monkeypatch.setattr(
        mod,
        "_build_embedding_request_orchestrator",
        lambda *_args, **_kwargs: fake_orchestrator,
        raising=False,
    )

    response = client.post(
        "/api/v1/embeddings",
        headers={"x-provider": "cohere"},
        json={"model": "embed-english-v3.0", "input": "hello world"},
    )

    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert response.json()["detail"]["error_code"] == "missing_provider_credentials"


def test_orchestrator_full_cache_hit_still_requires_provider_credentials(client, monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import embeddings_v5_production_enhanced as mod

    monkeypatch.setenv("EMBEDDINGS_ORCHESTRATOR_ENABLED", "true")

    async def fake_resolve(*_args, **_kwargs):
        return FakeCredentials(api_key=None, source="server")

    cache_get = AsyncMock(return_value=[0.0, 1.0])
    cache_set = AsyncMock()
    provider_call = AsyncMock(side_effect=AssertionError("provider path should not be called"))
    monkeypatch.setattr(mod, "_resolve_embeddings_byok", fake_resolve)
    monkeypatch.setattr(mod.embedding_cache, "get", cache_get)
    monkeypatch.setattr(mod.embedding_cache, "set", cache_set)
    monkeypatch.setattr(mod, "create_embeddings_with_circuit_breaker", provider_call)

    response = client.post(
        "/api/v1/embeddings",
        headers={"x-provider": "cohere"},
        json={"model": "embed-english-v3.0", "input": "cached but missing key"},
    )

    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert response.json()["detail"]["error_code"] == "missing_provider_credentials"
    assert cache_get.await_count == 0
    assert cache_set.await_count == 0
    assert provider_call.await_count == 0


def test_orchestrator_full_cache_hit_touches_resolved_provider_credentials(client, monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import embeddings_v5_production_enhanced as mod

    monkeypatch.setenv("EMBEDDINGS_ORCHESTRATOR_ENABLED", "true")
    credentials = FakeCredentials(api_key="cohere-key", source="user")

    async def fake_resolve(*_args, **_kwargs):
        return credentials

    cache_get = AsyncMock(return_value=[0.0, 1.0])
    cache_set = AsyncMock()
    provider_call = AsyncMock(side_effect=AssertionError("provider path should not be called"))
    monkeypatch.setattr(mod, "_resolve_embeddings_byok", fake_resolve)
    monkeypatch.setattr(mod.embedding_cache, "get", cache_get)
    monkeypatch.setattr(mod.embedding_cache, "set", cache_set)
    monkeypatch.setattr(mod, "create_embeddings_with_circuit_breaker", provider_call)

    response = client.post(
        "/api/v1/embeddings",
        headers={"x-provider": "cohere"},
        json={"model": "embed-english-v3.0", "input": "cached with key"},
    )

    assert response.status_code == status.HTTP_200_OK
    assert response.json()["data"][0]["embedding"] == [0.0, 1.0]
    assert cache_get.await_count == 1
    assert cache_set.await_count == 0
    assert provider_call.await_count == 0
    credentials.touch_last_used.assert_awaited_once()


def test_orchestrator_rate_limit_error_includes_retry_after(client, monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import embeddings_v5_production_enhanced as mod

    monkeypatch.setenv("EMBEDDINGS_ORCHESTRATOR_ENABLED", "true")
    fake_orchestrator = FakeOrchestrator(
        execute_error=EmbeddingRateLimitError(
            "provider_rate_limited",
            "Rate limit exceeded",
            retry_after=7,
            provider="openai",
            model="text-embedding-3-small",
        )
    )
    monkeypatch.setattr(
        mod,
        "_build_embedding_request_orchestrator",
        lambda *_args, **_kwargs: fake_orchestrator,
        raising=False,
    )

    response = client.post(
        "/api/v1/embeddings",
        json={"model": "text-embedding-3-small", "input": "hello world"},
    )

    assert response.status_code == status.HTTP_429_TOO_MANY_REQUESTS
    assert response.headers["Retry-After"] == "7"
    assert response.json()["detail"] == "Rate limit exceeded"


def test_orchestrator_response_headers_are_applied(client, monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import embeddings_v5_production_enhanced as mod

    monkeypatch.setenv("EMBEDDINGS_ORCHESTRATOR_ENABLED", "true")
    fake_orchestrator = FakeOrchestrator(
        result=EmbeddingExecutionResult(
            vectors=[[0.25, 0.75]],
            provider="huggingface",
            model="sentence-transformers/all-MiniLM-L6-v2",
            prompt_tokens=3,
            total_tokens=3,
            cache_hits=0,
            cache_misses=1,
            response_headers={"X-Embeddings-Provider": "huggingface"},
        )
    )
    monkeypatch.setattr(
        mod,
        "_build_embedding_request_orchestrator",
        lambda *_args, **_kwargs: fake_orchestrator,
        raising=False,
    )

    response = client.post(
        "/api/v1/embeddings",
        headers={"x-provider": "huggingface"},
        json={"model": "sentence-transformers/all-MiniLM-L6-v2", "input": "hello world"},
    )

    assert response.status_code == status.HTTP_200_OK
    assert response.headers["X-Embeddings-Provider"] == "huggingface"
    assert response.json()["model"] == "huggingface:sentence-transformers/all-MiniLM-L6-v2"


@pytest.mark.asyncio
async def test_executor_retries_openai_oauth_401_with_forced_refresh_and_touches_final_credentials(monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import embeddings_v5_production_enhanced as mod

    monkeypatch.setenv("TESTING", "true")
    monkeypatch.setenv("USE_REAL_OPENAI_IN_TESTS", "true")
    initial_credentials = FakeCredentials(
        api_key="oauth-initial-key",
        source="user",
        auth_source="oauth",
    )
    refreshed_credentials = FakeCredentials(
        api_key="oauth-refreshed-key",
        source="user",
        auth_source="oauth",
    )
    refresh_flags: list[bool] = []

    async def fake_resolve(provider, current_user, request, *, force_oauth_refresh=False):
        assert provider == "openai"
        assert current_user.id == 1
        assert request is not None
        refresh_flags.append(force_oauth_refresh)
        return refreshed_credentials if force_oauth_refresh else initial_credentials

    provider_calls: list[dict[str, object]] = []

    async def fake_provider(texts, provider, model, config, metadata=None, dimensions=None):
        provider_calls.append(
            {
                "texts": list(texts),
                "provider": provider,
                "model": model,
                "api_key": config.get("api_key"),
                "metadata": metadata,
                "dimensions": dimensions,
            }
        )
        if len(provider_calls) == 1:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="expired token")
        return [[0.9, 0.1]]

    monkeypatch.setattr(mod, "_resolve_embeddings_byok", fake_resolve)
    monkeypatch.setattr(mod, "create_embeddings_with_circuit_breaker", fake_provider)
    monkeypatch.setattr(mod, "_record_oauth_401_retry", lambda *_args, **_kwargs: None)

    executor = mod._EndpointEmbeddingExecutor(
        request=_request(),
        current_user=_user(),
        user_metadata={"user_id": 1},
    )

    vectors = await executor.create(
        ["refresh me"],
        provider="openai",
        model="text-embedding-3-small",
        dimensions=None,
    )

    assert vectors == [[0.9, 0.1]]
    assert refresh_flags == [False, True]
    assert [call["api_key"] for call in provider_calls] == [
        "oauth-initial-key",
        "oauth-refreshed-key",
    ]
    refreshed_credentials.touch_last_used.assert_awaited_once()
    initial_credentials.touch_last_used.assert_not_called()


def test_orchestrator_openai_oauth_second_401_propagates_original_auth_error(client, monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import embeddings_v5_production_enhanced as mod

    monkeypatch.setenv("EMBEDDINGS_ORCHESTRATOR_ENABLED", "true")
    monkeypatch.setenv("USE_REAL_OPENAI_IN_TESTS", "true")
    initial_credentials = FakeCredentials(
        api_key="oauth-initial-key",
        source="user",
        auth_source="oauth",
    )
    refreshed_credentials = FakeCredentials(
        api_key="oauth-refreshed-key",
        source="user",
        auth_source="oauth",
    )
    refresh_flags: list[bool] = []
    provider_call_count = 0

    async def fake_resolve(provider, current_user, request, *, force_oauth_refresh=False):
        assert provider == "openai"
        assert current_user.id == 1
        assert request is not None
        refresh_flags.append(force_oauth_refresh)
        return refreshed_credentials if force_oauth_refresh else initial_credentials

    async def fake_provider(texts, provider, model, config, metadata=None, dimensions=None):
        nonlocal provider_call_count
        _ = (texts, provider, model, config, metadata, dimensions)
        provider_call_count += 1
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="oauth auth failure")

    monkeypatch.setattr(mod, "_resolve_embeddings_byok", fake_resolve)
    monkeypatch.setattr(mod, "create_embeddings_with_circuit_breaker", fake_provider)
    monkeypatch.setattr(mod, "_record_oauth_401_retry", lambda *_args, **_kwargs: None)

    response = client.post(
        "/api/v1/embeddings",
        headers={"x-provider": "openai"},
        json={"model": "text-embedding-3-small", "input": "refresh fails"},
    )

    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    assert response.json()["detail"] == "oauth auth failure"
    assert refresh_flags == [False, True]
    assert provider_call_count == 2
    initial_credentials.touch_last_used.assert_not_called()
    refreshed_credentials.touch_last_used.assert_not_called()


@pytest.mark.asyncio
async def test_executor_missing_key_raises_missing_credentials_domain_error(monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import embeddings_v5_production_enhanced as mod

    async def fake_resolve(*_args, **_kwargs):
        return FakeCredentials(api_key=None, source="server")

    monkeypatch.setattr(mod, "_resolve_embeddings_byok", fake_resolve)

    executor = mod._EndpointEmbeddingExecutor(
        request=_request(),
        current_user=_user(),
        user_metadata=None,
    )

    with pytest.raises(EmbeddingProviderError) as exc_info:
        await executor.create(
            ["needs key"],
            provider="cohere",
            model="embed-english-v3.0",
            dimensions=None,
        )

    assert exc_info.value.code == "missing_provider_credentials"


@pytest.mark.asyncio
async def test_executor_batches_provider_calls_and_concatenates_vectors(monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import embeddings_v5_production_enhanced as mod

    async def fake_resolve(*_args, **_kwargs):
        return FakeCredentials(api_key=None, source="none")

    provider_batches: list[list[str]] = []

    async def fake_provider(texts, provider, model, config, metadata=None, dimensions=None):
        _ = (provider, model, config, metadata, dimensions)
        provider_batches.append(list(texts))
        return [[float(len(provider_batches)), float(index)] for index, _ in enumerate(texts)]

    monkeypatch.setattr(mod, "_resolve_embeddings_byok", fake_resolve)
    monkeypatch.setattr(mod, "create_embeddings_with_circuit_breaker", fake_provider)
    monkeypatch.setattr(mod, "MAX_BATCH_SIZE", 2)

    executor = mod._EndpointEmbeddingExecutor(
        request=_request(),
        current_user=_user(),
        user_metadata=None,
    )

    vectors = await executor.create(
        ["one", "two", "three", "four", "five"],
        provider="huggingface",
        model="sentence-transformers/all-MiniLM-L6-v2",
        dimensions=None,
    )

    assert provider_batches == [["one", "two"], ["three", "four"], ["five"]]
    assert vectors == [[1.0, 0.0], [1.0, 1.0], [2.0, 0.0], [2.0, 1.0], [3.0, 0.0]]


@pytest.mark.asyncio
async def test_executor_uses_adapter_registry_before_provider_execution(monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import embeddings_v5_production_enhanced as mod

    monkeypatch.setenv("LLM_EMBEDDINGS_ADAPTERS_ENABLED", "true")
    credentials = FakeCredentials(api_key="adapter-key", source="user")

    async def fake_resolve(*_args, **_kwargs):
        return credentials

    class FakeAdapter:
        def embed(self, request):
            assert request["input"] == ["one", "two"]
            assert request["model"] == "text-embedding-3-small"
            assert request["api_key"] == "adapter-key"
            return {
                "data": [
                    {"embedding": [0.1, 0.2]},
                    {"embedding": [0.3, 0.4]},
                ]
            }

    class FakeRegistry:
        def get_adapter(self, provider):
            assert provider == "openai"
            return FakeAdapter()

    provider_call = AsyncMock(side_effect=AssertionError("provider path should not be called"))
    monkeypatch.setattr(mod, "_resolve_embeddings_byok", fake_resolve)
    monkeypatch.setattr(mod, "get_embeddings_registry", lambda: FakeRegistry())
    monkeypatch.setattr(mod, "create_embeddings_with_circuit_breaker", provider_call)

    executor = mod._EndpointEmbeddingExecutor(
        request=_request(),
        current_user=_user(),
        user_metadata=None,
    )

    vectors = await executor.create(
        ["one", "two"],
        provider="openai",
        model="text-embedding-3-small",
        dimensions=None,
    )

    assert isinstance(vectors, EmbeddingExecutorOutput)
    assert vectors.vectors == [[0.1, 0.2], [0.3, 0.4]]
    assert vectors.embeddings_from_adapter is True
    assert provider_call.await_count == 0
    credentials.touch_last_used.assert_awaited_once()


def test_orchestrator_adapter_path_preserves_adapter_vector_scale(client, monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import embeddings_v5_production_enhanced as mod

    monkeypatch.setenv("EMBEDDINGS_ORCHESTRATOR_ENABLED", "true")
    monkeypatch.setenv("LLM_EMBEDDINGS_ADAPTERS_ENABLED", "true")
    credentials = FakeCredentials(api_key="adapter-key", source="user")

    async def fake_resolve(*_args, **_kwargs):
        return credentials

    class FakeAdapter:
        def embed(self, request):
            assert request["input"] == "adapter scale"
            assert request["api_key"] == "adapter-key"
            return {"data": [{"embedding": [3.0, 4.0]}]}

    class FakeRegistry:
        def get_adapter(self, provider):
            assert provider == "openai"
            return FakeAdapter()

    provider_call = AsyncMock(side_effect=AssertionError("provider path should not be called"))
    cache_get = AsyncMock(return_value=[0.6, 0.8])
    cache_set = AsyncMock()
    monkeypatch.setattr(mod, "_resolve_embeddings_byok", fake_resolve)
    monkeypatch.setattr(mod, "get_embeddings_registry", lambda: FakeRegistry())
    monkeypatch.setattr(mod, "create_embeddings_with_circuit_breaker", provider_call)
    monkeypatch.setattr(mod.embedding_cache, "get", cache_get)
    monkeypatch.setattr(mod.embedding_cache, "set", cache_set)

    response = client.post(
        "/api/v1/embeddings",
        json={"model": "text-embedding-3-small", "input": "adapter scale"},
    )

    assert response.status_code == status.HTTP_200_OK
    assert response.json()["data"][0]["embedding"] == [3.0, 4.0]
    assert provider_call.await_count == 0
    assert cache_get.await_count == 0
    assert cache_set.await_count == 0
    credentials.touch_last_used.assert_awaited_once()


def test_endpoint_real_orchestrator_applies_fallback_response_headers(client, monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import embeddings_v5_production_enhanced as mod

    monkeypatch.setenv("EMBEDDINGS_ORCHESTRATOR_ENABLED", "true")
    monkeypatch.setenv("EMBEDDINGS_ALLOW_FALLBACK_WITH_HEADER", "true")
    provider_calls: list[str] = []

    async def fake_create(self, texts, *, provider, model, dimensions):
        _ = (self, texts, model, dimensions)
        provider_calls.append(provider)
        if provider == "openai":
            raise EmbeddingProviderError(
                "provider_unavailable",
                "openai unavailable",
                retryable=True,
                provider=provider,
                model=model,
            )
        return [[0.25, 0.75]]

    monkeypatch.setattr(mod._EndpointEmbeddingExecutor, "create", fake_create)

    response = client.post(
        "/api/v1/embeddings",
        json={"model": "text-embedding-3-small", "input": "fallback headers"},
    )

    assert response.status_code == status.HTTP_200_OK
    assert provider_calls == ["openai", "huggingface"]
    assert response.headers["X-Embeddings-Provider"] == "huggingface"
    assert response.headers["X-Embeddings-Fallback-From"] == "openai"
    assert response.json()["model"] == "huggingface:sentence-transformers/all-MiniLM-L6-v2"


def test_endpoint_created_error_codes_are_part_of_domain_contract():
    for code in ("circuit_breaker_open", "internal_execution_failure"):
        exc = EmbeddingExecutionError(code, "execution failed")
        assert exc.code == code
