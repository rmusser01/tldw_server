from __future__ import annotations

import pytest

from tldw_Server_API.app.core.Embeddings.request_types import (
    EmbeddingProviderError,
    EmbeddingRateLimitError,
    EmbeddingRequestContext,
)
from tldw_Server_API.app.core.Embeddings.orchestrator import (
    EmbeddingExecutorOutput,
    EmbeddingExecutionResult,
    EmbeddingRequestOrchestrator,
)


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
    def __init__(
        self,
        vectors: list[list[float]] | None = None,
        *,
        failures: dict[str, EmbeddingProviderError] | None = None,
        provider_vectors: dict[str, list[list[float]]] | None = None,
    ) -> None:
        self.vectors = vectors or []
        self.failures = failures or {}
        self.provider_vectors = provider_vectors or {}
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
        failure = self.failures.get(provider)
        if failure is not None:
            raise failure
        if provider in self.provider_vectors:
            return self.provider_vectors[provider]
        return self.vectors


class AdapterAwareExecutor(RecordingExecutor):
    def __init__(
        self,
        vectors: list[list[float]] | None = None,
        *,
        adapter_output: EmbeddingExecutorOutput | None = None,
        adapter_calls_return_none: bool = False,
    ) -> None:
        super().__init__(vectors=vectors)
        self.adapter_output = adapter_output
        self.adapter_calls_return_none = adapter_calls_return_none
        self.adapter_calls: list[dict[str, object]] = []

    async def create_adapter(
        self,
        texts: list[str],
        *,
        provider: str,
        model: str,
        dimensions: int | None,
    ) -> EmbeddingExecutorOutput | None:
        self.adapter_calls.append(
            {
                "texts": texts,
                "provider": provider,
                "model": model,
                "dimensions": dimensions,
            }
        )
        if self.adapter_calls_return_none:
            return None
        return self.adapter_output


def _count_tokens(text: str, model: str) -> int:
    del model
    return len(text.split())


def _tokens_to_texts(tokens_input: list[int] | list[list[int]], model: str):
    del model
    if tokens_input and isinstance(tokens_input[0], int):
        return ["decoded"], len(tokens_input), [len(tokens_input)]
    return [f"decoded-{index}" for index, _ in enumerate(tokens_input)], 0, [
        len(item) for item in tokens_input
    ]


def _cache_key(
    text: str,
    provider: str,
    model: str,
    dimensions: int | None = None,
    backend_identity: str | None = None,
) -> str:
    parts = [text, provider, model]
    if dimensions:
        parts.append(str(dimensions))
    if backend_identity:
        parts.append(backend_identity)
    return "|".join(parts)


def _context(
    *,
    model: str = "sentence-transformers/all-MiniLM-L6-v2",
    provider: str | None = "huggingface",
    dimensions: int | None = None,
    encoding_format: str | None = "float",
) -> EmbeddingRequestContext:
    return EmbeddingRequestContext(
        user_id="u1",
        model_field=model,
        provider_header=provider,
        dimensions=dimensions,
        encoding_format=encoding_format,
        request_id="req-1",
    )


def _orchestrator(
    *,
    cache: RecordingCache | None = None,
    executor: RecordingExecutor | None = None,
    settings_fallback_chain: dict[str, object] | None = None,
    settings_fallback_model_map: dict[str, object] | None = None,
    dimension_policy: str = "reduce",
    provider_preflight=None,
    execution_path: str = "legacy",
) -> EmbeddingRequestOrchestrator:
    return EmbeddingRequestOrchestrator(
        count_tokens=_count_tokens,
        tokens_to_texts=_tokens_to_texts,
        cache_key_fn=_cache_key,
        cache=cache or RecordingCache(),
        executor=executor or RecordingExecutor(),
        settings_config={},
        max_tokens=100,
        implemented_providers={"openai", "huggingface", "onnx", "local_api"},
        allowed_providers=None,
        allowed_models=None,
        enforce_policy=True,
        allow_fallback_with_header=True,
        settings_fallback_chain=settings_fallback_chain,
        settings_fallback_model_map=settings_fallback_model_map,
        dimension_policy=dimension_policy,
        backend_identity_resolver=lambda provider, model: f"{provider}:{model}:backend",
        provider_preflight=provider_preflight,
        execution_path=execution_path,  # type: ignore[arg-type]
    )


@pytest.mark.unit
def test_prepare_normalizes_input_and_returns_token_totals_without_execution():
    cache = RecordingCache()
    executor = RecordingExecutor()
    orchestrator = _orchestrator(cache=cache, executor=executor)

    prepared = orchestrator.prepare(["hello world", "again"], _context())

    assert prepared.normalized_input.texts == ["hello world", "again"]
    assert prepared.normalized_input.token_counts == [2, 1]
    assert prepared.prompt_tokens == 3
    assert prepared.total_tokens == 3
    assert prepared.execution_plan.provider == "huggingface"
    assert prepared.execution_plan.model == "sentence-transformers/all-MiniLM-L6-v2"
    assert cache.get_keys == []
    assert cache.set_calls == []
    assert executor.calls == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_execute_full_cache_hit_skips_executor_and_preserves_order():
    cache = RecordingCache(
        {
            "hit one|huggingface|sentence-transformers/all-MiniLM-L6-v2|huggingface:sentence-transformers/all-MiniLM-L6-v2:backend": [
                1.0,
                0.0,
            ],
            "hit two|huggingface|sentence-transformers/all-MiniLM-L6-v2|huggingface:sentence-transformers/all-MiniLM-L6-v2:backend": [
                0.0,
                1.0,
            ],
        }
    )
    executor = RecordingExecutor(vectors=[[9.0, 9.0]])
    orchestrator = _orchestrator(cache=cache, executor=executor)
    prepared = orchestrator.prepare(["hit one", "hit two"], _context())

    result = await orchestrator.execute(prepared)

    assert isinstance(result, EmbeddingExecutionResult)
    assert result.vectors == [[1.0, 0.0], [0.0, 1.0]]
    assert result.cache_hits == 2
    assert result.cache_misses == 0
    assert result.prompt_tokens == 4
    assert result.total_tokens == 4
    assert executor.calls == []
    assert cache.set_calls == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_execute_partial_cache_hit_executes_only_misses_and_writes_provider_native_vectors():
    cache = RecordingCache(
        {
            "hit|huggingface|sentence-transformers/all-MiniLM-L6-v2|2|huggingface:sentence-transformers/all-MiniLM-L6-v2:backend": [
                1.0,
                0.0,
            ],
        }
    )
    executor = RecordingExecutor(vectors=[[0.25, 0.75, 0.5]])
    orchestrator = _orchestrator(cache=cache, executor=executor)
    prepared = orchestrator.prepare(["hit", "miss"], _context(dimensions=2))

    result = await orchestrator.execute(prepared)

    assert result.vectors == [[1.0, 0.0], [0.25, 0.75]]
    assert result.cache_hits == 1
    assert result.cache_misses == 1
    assert executor.calls == [
        {
            "texts": ["miss"],
            "provider": "huggingface",
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "dimensions": 2,
        }
    ]
    set_key, cached_value = cache.set_calls[0]
    assert set_key == (
        "miss|huggingface|sentence-transformers/all-MiniLM-L6-v2|2|"
        "huggingface:sentence-transformers/all-MiniLM-L6-v2:backend"
    )
    assert cached_value == [0.25, 0.75, 0.5]
    assert all(isinstance(item, float) for item in cached_value)
    assert result.response_headers["X-Embeddings-Dimensions-Policy"] == "reduce"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_dimension_policy_is_applied_to_provider_native_cache_hits_per_request():
    cache = RecordingCache()
    executor = RecordingExecutor(vectors=[[0.25, 0.75]])
    orchestrator = _orchestrator(
        cache=cache,
        executor=executor,
        dimension_policy="pad",
    )
    first = orchestrator.prepare(
        "policy-sensitive",
        _context(dimensions=4, encoding_format="float"),
    )

    first_result = await orchestrator.execute(first)

    assert first_result.vectors == [[0.25, 0.75, 0.0, 0.0]]
    set_key, cached_value = cache.set_calls[0]
    assert cached_value == [0.25, 0.75]

    second_executor = RecordingExecutor(vectors=[[9.0, 9.0, 9.0, 9.0]])
    second_orchestrator = _orchestrator(
        cache=cache,
        executor=second_executor,
        dimension_policy="pad",
    )
    second = second_orchestrator.prepare(
        "policy-sensitive",
        _context(dimensions=4, encoding_format="base64"),
    )

    second_result = await second_orchestrator.execute(second)

    assert cache.get_keys[-1] == set_key
    assert second_result.vectors == [[0.25, 0.75]]
    assert second_result.response_headers["X-Embeddings-Dimensions-Policy"] == "reduce"
    assert second_executor.calls == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_adapter_preferred_path_uses_provider_cache_when_adapter_returns_no_vectors():
    cache = RecordingCache(
        {
            "cached adapter miss|openai|text-embedding-3-small|"
            "openai:text-embedding-3-small:backend": [0.5, 0.25],
        }
    )
    executor = AdapterAwareExecutor(
        vectors=[[9.0, 9.0]],
        adapter_calls_return_none=True,
    )
    orchestrator = _orchestrator(
        cache=cache,
        executor=executor,
        execution_path="adapter",
    )
    prepared = orchestrator.prepare(
        "cached adapter miss",
        _context(model="text-embedding-3-small", provider="openai"),
    )

    result = await orchestrator.execute(prepared)

    assert result.vectors == [[0.5, 0.25]]
    assert result.cache_hits == 1
    assert result.cache_misses == 0
    assert executor.adapter_calls == [
        {
            "texts": ["cached adapter miss"],
            "provider": "openai",
            "model": "text-embedding-3-small",
            "dimensions": None,
        }
    ]
    assert executor.calls == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_execute_rejects_provider_vector_count_mismatch_without_cache_writeback():
    cache = RecordingCache()
    executor = RecordingExecutor(vectors=[[0.1, 0.2]])
    orchestrator = _orchestrator(cache=cache, executor=executor)
    prepared = orchestrator.prepare(["one", "two"], _context())

    with pytest.raises(EmbeddingProviderError) as exc_info:
        await orchestrator.execute(prepared)

    assert exc_info.value.code == "provider_malformed_response"
    assert "returned 1 embeddings" in exc_info.value.message
    assert "expected 2" in exc_info.value.message
    assert cache.set_calls == []


@pytest.mark.unit
def test_execution_plan_repr_does_not_contain_raw_input_text():
    orchestrator = _orchestrator()
    prepared = orchestrator.prepare(["secret raw text"], _context())

    plan_repr = repr(prepared.execution_plan)

    assert "secret raw text" not in plan_repr
    assert "texts" not in plan_repr
    assert prepared.normalized_input.texts == ["secret raw text"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_fallback_execution_maps_model_and_returns_fallback_headers():
    openai_failure = EmbeddingProviderError(
        "provider_unavailable",
        "openai unavailable",
        provider="openai",
        model="text-embedding-3-small",
        retryable=True,
    )
    executor = RecordingExecutor(
        vectors=[[0.5, 0.25]],
        failures={"openai": openai_failure},
    )
    orchestrator = _orchestrator(
        executor=executor,
        settings_fallback_chain={"openai": ["huggingface"]},
        settings_fallback_model_map={
            "openai:text-embedding-3-small": {
                "huggingface": "sentence-transformers/all-MiniLM-L6-v2"
            }
        },
    )
    prepared = orchestrator.prepare(
        "fallback mapping",
        _context(model="text-embedding-3-small", provider="openai"),
    )

    result = await orchestrator.execute(prepared)

    assert executor.calls == [
        {
            "texts": ["fallback mapping"],
            "provider": "openai",
            "model": "text-embedding-3-small",
            "dimensions": None,
        },
        {
            "texts": ["fallback mapping"],
            "provider": "huggingface",
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "dimensions": None,
        },
    ]
    assert result.provider == "huggingface"
    assert result.model == "sentence-transformers/all-MiniLM-L6-v2"
    assert result.fallback_from == "openai"
    assert result.response_headers["X-Embeddings-Provider"] == "huggingface"
    assert result.response_headers["X-Embeddings-Fallback-From"] == "openai"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_fallback_full_cache_hit_skips_fallback_executor_and_sets_adapter_origin():
    cache = RecordingCache(
        {
            "fallback cached|huggingface|sentence-transformers/all-MiniLM-L6-v2|"
            "huggingface:sentence-transformers/all-MiniLM-L6-v2:backend": [0.75, 0.25],
        }
    )
    openai_failure = EmbeddingProviderError(
        "provider_unavailable",
        "openai unavailable",
        provider="openai",
        model="text-embedding-3-small",
        retryable=True,
    )
    executor = RecordingExecutor(failures={"openai": openai_failure})
    orchestrator = _orchestrator(
        cache=cache,
        executor=executor,
        settings_fallback_chain={"openai": ["huggingface"]},
        settings_fallback_model_map={
            "openai:text-embedding-3-small": {
                "huggingface": "sentence-transformers/all-MiniLM-L6-v2"
            }
        },
    )
    prepared = orchestrator.prepare(
        "fallback cached",
        _context(model="text-embedding-3-small", provider="openai"),
    )

    result = await orchestrator.execute(prepared)

    assert result.vectors == [[0.75, 0.25]]
    assert result.provider == "huggingface"
    assert result.fallback_from == "openai"
    assert result.embeddings_from_adapter is False
    assert executor.calls == [
        {
            "texts": ["fallback cached"],
            "provider": "openai",
            "model": "text-embedding-3-small",
            "dimensions": None,
        }
    ]
    assert cache.set_calls == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_missing_credentials_for_non_requested_fallback_provider_is_skipped():
    openai_failure = EmbeddingProviderError(
        "provider_unavailable",
        "openai unavailable",
        provider="openai",
        model="text-embedding-3-small",
        retryable=True,
    )
    executor = RecordingExecutor(
        failures={"openai": openai_failure},
        provider_vectors={"huggingface": [[0.5, 0.25]]},
    )
    preflight_calls: list[tuple[str, str]] = []

    async def provider_preflight(provider: str, model: str) -> None:
        preflight_calls.append((provider, model))
        if provider == "cohere":
            raise EmbeddingProviderError(
                "missing_provider_credentials",
                "Embeddings provider 'cohere' requires an API key.",
                provider=provider,
                model=model,
            )

    orchestrator = _orchestrator(
        executor=executor,
        provider_preflight=provider_preflight,
        settings_fallback_chain={"openai": ["cohere", "huggingface"]},
        settings_fallback_model_map={
            "openai:text-embedding-3-small": {
                "cohere": "embed-english-v3.0",
                "huggingface": "sentence-transformers/all-MiniLM-L6-v2",
            }
        },
    )
    prepared = orchestrator.prepare(
        "fallback after missing creds",
        _context(model="text-embedding-3-small", provider="openai"),
    )

    result = await orchestrator.execute(prepared)

    assert result.provider == "huggingface"
    assert result.vectors == [[0.5, 0.25]]
    assert preflight_calls == [
        ("openai", "text-embedding-3-small"),
        ("cohere", "embed-english-v3.0"),
        ("huggingface", "sentence-transformers/all-MiniLM-L6-v2"),
    ]
    assert executor.calls == [
        {
            "texts": ["fallback after missing creds"],
            "provider": "openai",
            "model": "text-embedding-3-small",
            "dimensions": None,
        },
        {
            "texts": ["fallback after missing creds"],
            "provider": "huggingface",
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "dimensions": None,
        },
    ]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_base64_encoding_format_still_caches_float_vectors_not_encoded_values():
    cache = RecordingCache()
    executor = RecordingExecutor(vectors=[[0.25, 0.75, 0.5]])
    orchestrator = _orchestrator(cache=cache, executor=executor)
    prepared = orchestrator.prepare(
        "cache me",
        _context(dimensions=2, encoding_format="base64"),
    )

    result = await orchestrator.execute(prepared)

    cached_value = cache.set_calls[0][1]
    assert result.vectors == [[0.25, 0.75]]
    assert cached_value == [0.25, 0.75, 0.5]
    assert all(isinstance(item, float) for item in cached_value)
    assert all(not isinstance(item, str) for item in cached_value)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_partial_primary_cache_hit_reexecutes_all_texts_when_miss_falls_back():
    cache = RecordingCache(
        {
            "hit|openai|text-embedding-3-small|openai:text-embedding-3-small:backend": [
                9.0,
                9.0,
            ],
        }
    )
    openai_failure = EmbeddingProviderError(
        "provider_unavailable",
        "openai unavailable",
        provider="openai",
        model="text-embedding-3-small",
        retryable=True,
    )
    executor = RecordingExecutor(
        failures={"openai": openai_failure},
        provider_vectors={"huggingface": [[0.5, 0.5], [0.25, 0.75]]},
    )
    orchestrator = _orchestrator(
        cache=cache,
        executor=executor,
        settings_fallback_chain={"openai": ["huggingface"]},
        settings_fallback_model_map={
            "openai:text-embedding-3-small": {
                "huggingface": "sentence-transformers/all-MiniLM-L6-v2"
            }
        },
    )
    prepared = orchestrator.prepare(
        ["hit", "miss"],
        _context(model="text-embedding-3-small", provider="openai"),
    )

    result = await orchestrator.execute(prepared)

    assert result.vectors == [[0.5, 0.5], [0.25, 0.75]]
    assert result.provider == "huggingface"
    assert result.model == "sentence-transformers/all-MiniLM-L6-v2"
    assert executor.calls == [
        {
            "texts": ["miss"],
            "provider": "openai",
            "model": "text-embedding-3-small",
            "dimensions": None,
        },
        {
            "texts": ["hit", "miss"],
            "provider": "huggingface",
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "dimensions": None,
        },
    ]
    assert cache.set_calls == [
        (
            "hit|huggingface|sentence-transformers/all-MiniLM-L6-v2|"
            "huggingface:sentence-transformers/all-MiniLM-L6-v2:backend",
            [0.5, 0.5],
        ),
        (
            "miss|huggingface|sentence-transformers/all-MiniLM-L6-v2|"
            "huggingface:sentence-transformers/all-MiniLM-L6-v2:backend",
            [0.25, 0.75],
        ),
    ]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_non_retryable_provider_error_does_not_attempt_fallback_and_is_raised_as_is():
    primary_error = EmbeddingProviderError(
        "provider_malformed_response",
        "bad response",
        provider="openai",
        model="text-embedding-3-small",
        retryable=False,
    )
    executor = RecordingExecutor(
        failures={"openai": primary_error},
        provider_vectors={"huggingface": [[0.25, 0.75]]},
    )
    orchestrator = _orchestrator(
        executor=executor,
        settings_fallback_chain={"openai": ["huggingface"]},
    )
    prepared = orchestrator.prepare(
        "no fallback",
        _context(model="text-embedding-3-small", provider="openai"),
    )

    with pytest.raises(EmbeddingProviderError) as exc_info:
        await orchestrator.execute(prepared)

    assert exc_info.value is primary_error
    assert executor.calls == [
        {
            "texts": ["no fallback"],
            "provider": "openai",
            "model": "text-embedding-3-small",
            "dimensions": None,
        }
    ]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_rate_limit_exhaustion_preserves_retry_after_and_retryability():
    rate_limit_error = EmbeddingRateLimitError(
        "provider_rate_limited",
        "rate limited",
        provider="openai",
        model="text-embedding-3-small",
        retryable=True,
        retry_after=42,
    )
    fallback_error = EmbeddingProviderError(
        "provider_unavailable",
        "fallback unavailable",
        provider="huggingface",
        model="sentence-transformers/all-MiniLM-L6-v2",
        retryable=True,
    )
    executor = RecordingExecutor(
        failures={"openai": rate_limit_error, "huggingface": fallback_error},
    )
    orchestrator = _orchestrator(
        executor=executor,
        settings_fallback_chain={"openai": ["huggingface"]},
        settings_fallback_model_map={
            "openai:text-embedding-3-small": {
                "huggingface": "sentence-transformers/all-MiniLM-L6-v2"
            }
        },
    )
    prepared = orchestrator.prepare(
        "rate limited",
        _context(model="text-embedding-3-small", provider="openai"),
    )

    with pytest.raises(EmbeddingRateLimitError) as exc_info:
        await orchestrator.execute(prepared)

    assert exc_info.value is rate_limit_error
    assert exc_info.value.retryable is True
    assert exc_info.value.retry_after == 42


@pytest.mark.unit
@pytest.mark.asyncio
async def test_base64_requested_dimensions_force_reduce_even_when_configured_policy_ignores():
    cache = RecordingCache()
    executor = RecordingExecutor(vectors=[[0.25, 0.75, 0.5]])
    orchestrator = _orchestrator(
        cache=cache,
        executor=executor,
        dimension_policy="ignore",
    )
    prepared = orchestrator.prepare(
        "cache me",
        _context(dimensions=2, encoding_format="base64"),
    )

    result = await orchestrator.execute(prepared)

    assert result.vectors == [[0.25, 0.75]]
    assert cache.set_calls[0][1] == [0.25, 0.75, 0.5]
    assert result.response_headers["X-Embeddings-Dimensions-Policy"] == "reduce"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_malformed_numeric_string_vector_is_rejected_without_cache_writeback():
    cache = RecordingCache()
    executor = RecordingExecutor(vectors=["123"])  # type: ignore[list-item]
    orchestrator = _orchestrator(cache=cache, executor=executor)
    prepared = orchestrator.prepare("malformed", _context())

    with pytest.raises(EmbeddingProviderError) as exc_info:
        await orchestrator.execute(prepared)

    assert exc_info.value.code == "provider_malformed_response"
    assert cache.set_calls == []
