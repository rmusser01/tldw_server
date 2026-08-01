from __future__ import annotations

import pytest

from tldw_Server_API.app.core.Embeddings.provider_attempt import (
    EmbeddingProviderAttempt,
    EmbeddingProviderReadinessCheck,
    ProviderAttemptSuccess,
    ProviderCallFailure,
)
from tldw_Server_API.app.core.Embeddings.request_types import (
    EmbeddingExecutionPlan,
    EmbeddingExecutionError,
    EmbeddingExecutorOutput,
    EmbeddingPolicyDecision,
    EmbeddingProviderError,
    NormalizedEmbeddingInput,
    PreparedEmbeddingRequest,
    ProviderModelIntent,
)
from tldw_Server_API.app.core.Embeddings.vector_processing import (
    EmbeddingVectorProcessor,
)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_provider_readiness_check_delegates_to_preflight():
    calls: list[tuple[str, str]] = []

    async def preflight(provider: str, model: str) -> None:
        calls.append((provider, model))

    readiness = EmbeddingProviderReadinessCheck(preflight)

    await readiness.check("openai", "text-embedding-3-small")

    assert calls == [("openai", "text-embedding-3-small")]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_provider_readiness_check_propagates_exact_error():
    error = EmbeddingExecutionError(
        "circuit_breaker_open",
        "provider unavailable",
        provider="openai",
        model="text-embedding-3-small",
        retryable=True,
    )

    async def preflight(provider: str, model: str) -> None:
        del provider, model
        raise error

    readiness = EmbeddingProviderReadinessCheck(preflight)

    with pytest.raises(EmbeddingExecutionError) as exc_info:
        await readiness.check("openai", "text-embedding-3-small")

    assert exc_info.value is error


@pytest.mark.unit
def test_executor_output_contract_is_shared_from_request_types():
    output = EmbeddingExecutorOutput(
        vectors=[[0.1, 0.2]],
        embeddings_from_adapter=True,
    )

    assert output.vectors == [[0.1, 0.2]]
    assert output.embeddings_from_adapter is True


class RecordingCache:
    def __init__(self, values: dict[str, list[float]] | None = None) -> None:
        self.values = values or {}
        self.get_keys: list[str] = []
        self.set_calls: list[tuple[str, list[float]]] = []

    async def get(self, key: str) -> list[float] | None:
        self.get_keys.append(key)
        return self.values.get(key)

    async def set(self, key: str, value: list[float]) -> object:
        self.set_calls.append((key, value))
        self.values[key] = value
        return None


class RecordingExecutor:
    def __init__(self, vectors: list[list[float]] | None = None) -> None:
        self.vectors = vectors or []
        self.calls: list[dict[str, object]] = []

    async def create(
        self,
        texts: list[str],
        *,
        provider: str,
        model: str,
        dimensions: int | None,
    ) -> list[list[float]]:
        self.calls.append(
            {
                "texts": texts,
                "provider": provider,
                "model": model,
                "dimensions": dimensions,
            }
        )
        return self.vectors


def _cache_key(
    text: str,
    provider: str,
    model: str,
    dimensions: int | None,
    backend_identity: str | None,
) -> str:
    parts = [text, provider, model]
    if dimensions is not None:
        parts.append(str(dimensions))
    if backend_identity is not None:
        parts.append(backend_identity)
    return "|".join(parts)


def _prepared(
    texts: list[str],
    *,
    provider: str = "openai",
    model: str = "text-embedding-3-small",
    dimensions: int | None = None,
    backend_identity: str | None = "stale-plan-identity",
    cache_namespace: str | None = "ignored-namespace",
) -> PreparedEmbeddingRequest:
    return PreparedEmbeddingRequest(
        normalized_input=NormalizedEmbeddingInput(
            texts=texts,
            token_counts=[1 for _ in texts],
            total_tokens=len(texts),
        ),
        provider_intent=ProviderModelIntent(
            provider=provider,
            model=model,
            requested_provider=provider,
            requested_model=model,
            provider_was_explicit=True,
            model_was_provider_qualified=False,
        ),
        policy_decision=EmbeddingPolicyDecision(
            provider=provider,
            model=model,
            dimensions=dimensions,
            fallback_chain=[provider],
            fallback_allowed=True,
            enforce_policy=True,
        ),
        execution_plan=EmbeddingExecutionPlan(
            provider=provider,
            model=model,
            dimensions=dimensions,
            backend_identity=backend_identity,
            fallback_chain=[provider],
            cache_namespace=cache_namespace,
        ),
        effective_dimension_policy="reduce",
        prompt_tokens=len(texts),
        total_tokens=len(texts),
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_provider_attempt_preserves_order_and_executes_only_cache_misses():
    cache = RecordingCache(
        {
            "hit|openai|text-embedding-3-small|read-openai": [1.0, 0.0],
            "hit2|openai|text-embedding-3-small|read-openai": [0.0, 1.0],
        }
    )
    executor = RecordingExecutor(vectors=[[0.25, 0.75]])
    identity_calls: list[tuple[str, str]] = []

    def backend_identity(provider: str, model: str) -> str:
        identity_calls.append((provider, model))
        return "read-openai" if len(identity_calls) == 1 else "write-openai"

    attempt = EmbeddingProviderAttempt(
        cache_key_fn=_cache_key,
        cache=cache,
        executor=executor,
        backend_identity_resolver=backend_identity,
        vector_processor=EmbeddingVectorProcessor(),
    )

    result = await attempt.execute(
        _prepared(["hit", "miss", "hit2"]),
        provider="openai",
        model="text-embedding-3-small",
    )

    assert isinstance(result, ProviderAttemptSuccess)
    assert result.vectors == [[1.0, 0.0], [0.25, 0.75], [0.0, 1.0]]
    assert result.cache_hits == 2
    assert result.cache_misses == 1
    assert executor.calls == [
        {
            "texts": ["miss"],
            "provider": "openai",
            "model": "text-embedding-3-small",
            "dimensions": None,
        }
    ]
    assert cache.get_keys == [
        "hit|openai|text-embedding-3-small|read-openai",
        "miss|openai|text-embedding-3-small|read-openai",
        "hit2|openai|text-embedding-3-small|read-openai",
    ]
    assert cache.set_calls == [
        ("miss|openai|text-embedding-3-small|write-openai", [0.25, 0.75])
    ]
    assert identity_calls == [
        ("openai", "text-embedding-3-small"),
        ("openai", "text-embedding-3-small"),
    ]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_provider_attempt_cache_keys_exclude_plan_identity_and_namespace():
    cache = RecordingCache()
    executor = RecordingExecutor(vectors=[[0.1, 0.2]])
    cache_key_calls: list[tuple[str, str, str, int | None, str | None]] = []

    def cache_key_probe(
        text: str,
        provider: str,
        model: str,
        dimensions: int | None,
        backend_identity: str | None,
    ) -> str:
        cache_key_calls.append((text, provider, model, dimensions, backend_identity))
        return _cache_key(text, provider, model, dimensions, backend_identity)

    attempt = EmbeddingProviderAttempt(
        cache_key_fn=cache_key_probe,
        cache=cache,
        executor=executor,
        backend_identity_resolver=lambda provider, model: (
            f"runtime:{provider}:{model}"
        ),
        vector_processor=EmbeddingVectorProcessor(),
    )

    await attempt.execute(
        _prepared(
            ["one"],
            dimensions=2,
            backend_identity="stale-plan",
            cache_namespace="ns",
        ),
        provider="openai",
        model="text-embedding-3-small",
    )

    runtime_identity = "runtime:openai:text-embedding-3-small"
    assert cache_key_calls == [
        ("one", "openai", "text-embedding-3-small", 2, runtime_identity),
        ("one", "openai", "text-embedding-3-small", 2, runtime_identity),
    ]
    observed_keys = cache.get_keys + [key for key, _ in cache.set_calls]
    assert all("stale-plan" not in key for key in observed_keys)
    assert all("ns" not in key for key in observed_keys)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_provider_attempt_writes_provider_native_vectors_after_full_response_validation():
    cache = RecordingCache(
        {"hit|openai|text-embedding-3-small|2|read": [1.0, 0.0, 0.0]}
    )
    executor = RecordingExecutor(vectors=[[0.25, 0.75, 0.5]])
    identities = iter(["read", "write"])
    attempt = EmbeddingProviderAttempt(
        cache_key_fn=_cache_key,
        cache=cache,
        executor=executor,
        backend_identity_resolver=lambda provider, model: next(identities),
        vector_processor=EmbeddingVectorProcessor(),
    )

    result = await attempt.execute(
        _prepared(["hit", "miss"], dimensions=2),
        provider="openai",
        model="text-embedding-3-small",
    )

    assert isinstance(result, ProviderAttemptSuccess)
    assert result.vectors == [[1.0, 0.0], [0.25, 0.75]]
    assert cache.set_calls == [
        ("miss|openai|text-embedding-3-small|2|write", [0.25, 0.75, 0.5])
    ]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_provider_attempt_rejects_provider_malformed_response_before_writeback():
    cache = RecordingCache({"hit|openai|text-embedding-3-small|read": [1.0, 0.0]})
    executor = RecordingExecutor(vectors=[[0.25, 0.75, 0.5]])
    attempt = EmbeddingProviderAttempt(
        cache_key_fn=_cache_key,
        cache=cache,
        executor=executor,
        backend_identity_resolver=lambda provider, model: "read",
        vector_processor=EmbeddingVectorProcessor(),
    )

    with pytest.raises(EmbeddingProviderError) as exc_info:
        await attempt.execute(
            _prepared(["hit", "miss"]),
            provider="openai",
            model="text-embedding-3-small",
        )

    assert exc_info.value.code == "provider_malformed_response"
    assert cache.set_calls == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_provider_attempt_treats_malformed_cached_vector_as_miss():
    cache = RecordingCache(
        {"bad|openai|text-embedding-3-small|read": [float("nan")]}
    )
    executor = RecordingExecutor(vectors=[[0.1, 0.2]])
    attempt = EmbeddingProviderAttempt(
        cache_key_fn=_cache_key,
        cache=cache,
        executor=executor,
        backend_identity_resolver=lambda provider, model: "read",
        vector_processor=EmbeddingVectorProcessor(),
    )

    result = await attempt.execute(
        _prepared(["bad"]),
        provider="openai",
        model="text-embedding-3-small",
    )

    assert isinstance(result, ProviderAttemptSuccess)
    assert result.vectors == [[0.1, 0.2]]
    assert result.cache_hits == 0
    assert result.cache_misses == 1
    assert executor.calls[0]["texts"] == ["bad"]
    assert cache.set_calls == [
        ("bad|openai|text-embedding-3-small|read", [0.1, 0.2])
    ]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_provider_attempt_skips_cache_write_for_adapter_originated_executor_output():
    class AdapterOriginExecutor(RecordingExecutor):
        async def create(
            self,
            texts: list[str],
            *,
            provider: str,
            model: str,
            dimensions: int | None,
        ) -> EmbeddingExecutorOutput:
            self.calls.append(
                {
                    "texts": texts,
                    "provider": provider,
                    "model": model,
                    "dimensions": dimensions,
                }
            )
            return EmbeddingExecutorOutput(
                vectors=[[0.1, 0.2]],
                embeddings_from_adapter=True,
            )

    cache = RecordingCache()
    executor = AdapterOriginExecutor()
    attempt = EmbeddingProviderAttempt(
        cache_key_fn=_cache_key,
        cache=cache,
        executor=executor,
        backend_identity_resolver=lambda provider, model: "identity",
        vector_processor=EmbeddingVectorProcessor(),
    )

    result = await attempt.execute(
        _prepared(["adapter-origin"]),
        provider="openai",
        model="text-embedding-3-small",
    )

    assert isinstance(result, ProviderAttemptSuccess)
    assert result.vectors == [[0.1, 0.2]]
    assert result.embeddings_from_adapter is True
    assert cache.set_calls == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_provider_attempt_returns_exact_provider_call_failure_from_executor():
    error = EmbeddingProviderError(
        "provider_unavailable",
        "provider down",
        provider="openai",
        model="text-embedding-3-small",
        retryable=True,
    )

    class FailingExecutor(RecordingExecutor):
        async def create(
            self,
            texts: list[str],
            *,
            provider: str,
            model: str,
            dimensions: int | None,
        ) -> list[list[float]]:
            self.calls.append(
                {
                    "texts": texts,
                    "provider": provider,
                    "model": model,
                    "dimensions": dimensions,
                }
            )
            raise error

    attempt = EmbeddingProviderAttempt(
        cache_key_fn=_cache_key,
        cache=RecordingCache(),
        executor=FailingExecutor(),
        backend_identity_resolver=lambda provider, model: "identity",
        vector_processor=EmbeddingVectorProcessor(),
    )

    result = await attempt.execute(
        _prepared(["one"]),
        provider="openai",
        model="text-embedding-3-small",
    )

    assert isinstance(result, ProviderCallFailure)
    assert result.error is error


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["identity", "cache_key", "cache_get", "cache_set"])
async def test_provider_attempt_non_provider_failures_propagate_without_call_failure(
    boundary,
):
    original = RuntimeError(f"{boundary} failed")

    def identity(provider: str, model: str) -> str:
        del provider, model
        if boundary == "identity":
            raise original
        return "identity"

    def cache_key(
        text: str,
        provider: str,
        model: str,
        dimensions: int | None,
        backend_identity: str | None,
    ) -> str:
        if boundary == "cache_key":
            raise original
        return _cache_key(text, provider, model, dimensions, backend_identity)

    class BoundaryCache(RecordingCache):
        async def get(self, key: str) -> list[float] | None:
            if boundary == "cache_get":
                raise original
            return await super().get(key)

        async def set(self, key: str, value: list[float]) -> object:
            if boundary == "cache_set":
                raise original
            return await super().set(key, value)

    attempt = EmbeddingProviderAttempt(
        cache_key_fn=cache_key,
        cache=BoundaryCache(),
        executor=RecordingExecutor(vectors=[[0.1, 0.2]]),
        backend_identity_resolver=identity,
        vector_processor=EmbeddingVectorProcessor(),
    )

    with pytest.raises(RuntimeError) as exc_info:
        await attempt.execute(
            _prepared(["one"]),
            provider="openai",
            model="text-embedding-3-small",
        )

    assert exc_info.value is original
