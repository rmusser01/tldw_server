from __future__ import annotations

import asyncio

import pytest

import tldw_Server_API.app.core.Embeddings.preparation as preparation_module
from tldw_Server_API.app.core.Embeddings import orchestrator as orchestrator_module
from tldw_Server_API.app.core.Embeddings.orchestrator import (
    EmbeddingExecutionResult,
    EmbeddingExecutorOutput,
    EmbeddingRequestOrchestrator,
)
from tldw_Server_API.app.core.Embeddings.orchestrator import (
    PreparedEmbeddingRequest as OrchestratorPreparedEmbeddingRequest,
)
from tldw_Server_API.app.core.Embeddings.request_types import (
    EmbeddingExecutionError,
    EmbeddingProviderError,
    EmbeddingRateLimitError,
    EmbeddingRequestContext,
    PreparedEmbeddingRequest,
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
    cache_key_fn=None,
    settings_fallback_chain: dict[str, object] | None = None,
    settings_fallback_model_map: dict[str, object] | None = None,
    dimension_policy: str = "reduce",
    backend_identity_resolver=None,
    provider_preflight=None,
    execution_path: str = "legacy",
) -> EmbeddingRequestOrchestrator:
    return EmbeddingRequestOrchestrator(
        count_tokens=_count_tokens,
        tokens_to_texts=_tokens_to_texts,
        cache_key_fn=_cache_key if cache_key_fn is None else cache_key_fn,
        cache=cache or RecordingCache(),
        executor=executor or RecordingExecutor(),
        settings_config={},
        max_tokens=100,
        implemented_providers={
            "openai",
            "cohere",
            "huggingface",
            "onnx",
            "local_api",
        },
        allowed_providers=None,
        allowed_models=None,
        enforce_policy=True,
        allow_fallback_with_header=True,
        settings_fallback_chain=settings_fallback_chain,
        settings_fallback_model_map=settings_fallback_model_map,
        dimension_policy=dimension_policy,
        backend_identity_resolver=(
            backend_identity_resolver
            if backend_identity_resolver is not None
            else lambda provider, model: f"{provider}:{model}:backend"
        ),
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
def test_prepare_orders_intent_normalization_policy_and_plan_identity(monkeypatch):
    calls: list[str] = []
    real_resolve_provider_model = preparation_module.resolve_provider_model
    real_normalize_embedding_input = preparation_module.normalize_embedding_input
    real_enforce_embedding_policy = preparation_module.enforce_embedding_policy

    def resolve_provider_model_probe(*args, **kwargs):
        calls.append("resolve_intent")
        return real_resolve_provider_model(*args, **kwargs)

    def normalize_embedding_input_probe(*args, **kwargs):
        calls.append("normalize")
        return real_normalize_embedding_input(*args, **kwargs)

    def enforce_embedding_policy_probe(*args, **kwargs):
        calls.append("resolve_policy")
        return real_enforce_embedding_policy(*args, **kwargs)

    def backend_identity_probe(provider: str, model: str) -> str:
        calls.append("plan_identity")
        return f"{provider}:{model}:backend"

    monkeypatch.setattr(
        preparation_module,
        "resolve_provider_model",
        resolve_provider_model_probe,
    )
    monkeypatch.setattr(
        preparation_module,
        "normalize_embedding_input",
        normalize_embedding_input_probe,
    )
    monkeypatch.setattr(
        preparation_module,
        "enforce_embedding_policy",
        enforce_embedding_policy_probe,
    )
    orchestrator = _orchestrator(
        backend_identity_resolver=backend_identity_probe,
    )

    orchestrator.prepare("ordered preparation", _context())

    assert calls == [
        "resolve_intent",
        "normalize",
        "resolve_policy",
        "plan_identity",
    ]


@pytest.mark.unit
@pytest.mark.parametrize("use_default_backend_identity", [False, True])
def test_prepare_delegates_to_one_pipeline_without_phase_sink(
    monkeypatch,
    use_default_backend_identity,
):
    prepared_sentinel = object()
    created: list[dict[str, object]] = []
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    pipeline_values = {
        name: object()
        for name in (
            "count_tokens",
            "tokens_to_texts",
            "settings_config",
            "max_tokens",
            "implemented_providers",
            "allowed_providers",
            "allowed_models",
            "enforce_policy",
            "allow_fallback_with_header",
            "settings_fallback_chain",
            "settings_fallback_model_map",
            "dimension_policy",
            "require_model",
            "guess_provider",
            "backend_identity_resolver",
            "cache_namespace",
            "batch_size",
            "execution_path",
        )
    }

    def default_backend_identity(provider: str, model: str) -> None:
        del provider, model
        return None

    class RecordingPreparationPipeline:
        def __init__(self, **kwargs: object) -> None:
            created.append(kwargs)

        def prepare(self, *args: object, **kwargs: object) -> object:
            calls.append((args, kwargs))
            return prepared_sentinel

    monkeypatch.setattr(
        orchestrator_module,
        "EmbeddingPreparationPipeline",
        RecordingPreparationPipeline,
    )
    monkeypatch.setattr(
        orchestrator_module,
        "_no_backend_identity",
        default_backend_identity,
    )
    requested_backend_identity = (
        None
        if use_default_backend_identity
        else pipeline_values["backend_identity_resolver"]
    )
    expected_pipeline_values = {
        **pipeline_values,
        "backend_identity_resolver": (
            default_backend_identity
            if use_default_backend_identity
            else pipeline_values["backend_identity_resolver"]
        ),
    }
    orchestrator = EmbeddingRequestOrchestrator(
        count_tokens=pipeline_values["count_tokens"],
        tokens_to_texts=pipeline_values["tokens_to_texts"],
        cache_key_fn=_cache_key,
        cache=RecordingCache(),
        executor=RecordingExecutor(),
        settings_config=pipeline_values["settings_config"],
        max_tokens=pipeline_values["max_tokens"],
        implemented_providers=pipeline_values["implemented_providers"],
        allowed_providers=pipeline_values["allowed_providers"],
        allowed_models=pipeline_values["allowed_models"],
        enforce_policy=pipeline_values["enforce_policy"],
        allow_fallback_with_header=pipeline_values["allow_fallback_with_header"],
        settings_fallback_chain=pipeline_values["settings_fallback_chain"],
        settings_fallback_model_map=pipeline_values[
            "settings_fallback_model_map"
        ],
        dimension_policy=pipeline_values["dimension_policy"],
        require_model=pipeline_values["require_model"],
        guess_provider=pipeline_values["guess_provider"],
        backend_identity_resolver=requested_backend_identity,
        cache_namespace=pipeline_values["cache_namespace"],
        batch_size=pipeline_values["batch_size"],
        execution_path=pipeline_values["execution_path"],
    )
    raw_input = object()
    context = _context()

    assert orchestrator.prepare(raw_input, context) is prepared_sentinel
    assert orchestrator.prepare(raw_input, context) is prepared_sentinel
    assert len(created) == 1
    assert created[0].keys() == expected_pipeline_values.keys()
    assert all(
        created[0][name] is expected_value
        for name, expected_value in expected_pipeline_values.items()
    )
    assert calls == [((raw_input, context), {}), ((raw_input, context), {})]


@pytest.mark.unit
def test_orchestrator_reexports_prepared_embedding_request_contract():
    assert OrchestratorPreparedEmbeddingRequest is PreparedEmbeddingRequest


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
async def test_execute_rejects_mixed_width_full_cache_result():
    cache = RecordingCache(
        {
            "hit one|huggingface|sentence-transformers/all-MiniLM-L6-v2|huggingface:sentence-transformers/all-MiniLM-L6-v2:backend": [
                1.0,
                0.0,
            ],
            "hit two|huggingface|sentence-transformers/all-MiniLM-L6-v2|huggingface:sentence-transformers/all-MiniLM-L6-v2:backend": [
                1.0,
                0.0,
                0.5,
            ],
        }
    )
    executor = RecordingExecutor(vectors=[[9.0, 9.0]])
    orchestrator = _orchestrator(cache=cache, executor=executor)
    prepared = orchestrator.prepare(["hit one", "hit two"], _context())

    with pytest.raises(EmbeddingProviderError) as exc_info:
        await orchestrator.execute(prepared)

    assert exc_info.value.code == "provider_malformed_response"
    assert executor.calls == []
    assert cache.set_calls == []


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_concurrent_cache_assembly_keeps_malformed_and_valid_results_isolated():
    class GatedCache(RecordingCache):
        def __init__(self, values):
            super().__init__(values)
            self.arrivals = 0
            self.both_arrived = asyncio.Event()
            self.release = asyncio.Event()

        async def get(self, key: str) -> list[float] | None:
            value = await super().get(key)
            if key.startswith(("malformed one|", "valid one|")):
                self.arrivals += 1
                if self.arrivals == 2:
                    self.both_arrived.set()
                await asyncio.wait_for(self.release.wait(), timeout=10)
            return value

    suffix = (
        "huggingface|sentence-transformers/all-MiniLM-L6-v2|"
        "huggingface:sentence-transformers/all-MiniLM-L6-v2:backend"
    )
    cache = GatedCache(
        {
            f"malformed one|{suffix}": [1.0, 0.0],
            f"malformed two|{suffix}": [1.0, 0.0, 0.5],
            f"valid one|{suffix}": [0.0, 1.0],
            f"valid two|{suffix}": [1.0, 0.0],
        }
    )
    executor = RecordingExecutor(vectors=[[9.0, 9.0]])
    orchestrator = _orchestrator(cache=cache, executor=executor)
    malformed = orchestrator.prepare(
        ["malformed one", "malformed two"],
        _context(),
    )
    valid = orchestrator.prepare(["valid one", "valid two"], _context())
    tasks = [
        asyncio.create_task(orchestrator.execute(prepared))
        for prepared in (malformed, valid)
    ]
    try:
        await asyncio.wait_for(cache.both_arrived.wait(), timeout=10)
    finally:
        cache.release.set()
    malformed_result, valid_result = await asyncio.gather(*tasks, return_exceptions=True)

    assert isinstance(malformed_result, EmbeddingProviderError)
    assert malformed_result.code == "provider_malformed_response"
    assert isinstance(valid_result, EmbeddingExecutionResult)
    assert valid_result.vectors == [[0.0, 1.0], [1.0, 0.0]]
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
async def test_execute_rejects_mixed_width_partial_cache_and_provider_result_without_writeback():
    cache = RecordingCache(
        {
            "hit|huggingface|sentence-transformers/all-MiniLM-L6-v2|huggingface:sentence-transformers/all-MiniLM-L6-v2:backend": [
                1.0,
                0.0,
            ],
        }
    )
    executor = RecordingExecutor(vectors=[[0.25, 0.75, 0.5]])
    orchestrator = _orchestrator(cache=cache, executor=executor)
    prepared = orchestrator.prepare(["hit", "miss"], _context())

    with pytest.raises(EmbeddingProviderError) as exc_info:
        await orchestrator.execute(prepared)

    assert exc_info.value.code == "provider_malformed_response"
    assert executor.calls == [
        {
            "texts": ["miss"],
            "provider": "huggingface",
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "dimensions": None,
        }
    ]
    assert cache.set_calls == []


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
async def test_successful_adapter_reports_compatibility_cache_counts_and_bypasses_provider():
    cache = RecordingCache()
    preflight_calls: list[tuple[str, str]] = []

    async def provider_preflight(provider: str, model: str) -> None:
        preflight_calls.append((provider, model))

    executor = AdapterAwareExecutor(
        vectors=[[9.0, 9.0]],
        adapter_output=EmbeddingExecutorOutput(
            vectors=[[0.1, 0.2], [0.3, 0.4]],
            embeddings_from_adapter=True,
        ),
    )
    orchestrator = _orchestrator(
        cache=cache,
        executor=executor,
        execution_path="adapter",
        provider_preflight=provider_preflight,
    )
    prepared = orchestrator.prepare(
        ["one", "two"],
        _context(model="text-embedding-3-small", provider="openai"),
    )

    result = await orchestrator.execute(prepared)

    assert result.vectors == [[0.1, 0.2], [0.3, 0.4]]
    assert result.embeddings_from_adapter is True
    assert result.cache_hits == 0
    assert result.cache_misses == 2
    assert executor.adapter_calls == [
        {
            "texts": ["one", "two"],
            "provider": "openai",
            "model": "text-embedding-3-small",
            "dimensions": None,
        }
    ]
    assert preflight_calls == []
    assert cache.get_keys == []
    assert cache.set_calls == []
    assert executor.calls == []


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

    assert prepared.execution_plan.fallback_chain == ["openai", "huggingface"]
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
    assert [call["provider"] for call in executor.calls].count("openai") == 1
    assert result.provider == "huggingface"
    assert result.model == "sentence-transformers/all-MiniLM-L6-v2"
    assert result.fallback_from == "openai"
    assert result.response_headers["X-Embeddings-Provider"] == "huggingface"
    assert result.response_headers["X-Embeddings-Fallback-From"] == "openai"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_fallback_complete_result_is_validated_before_first_cache_write():
    primary_error = EmbeddingProviderError(
        "provider_unavailable",
        "primary unavailable",
        provider="openai",
        model="text-embedding-3-small",
        retryable=True,
    )
    fallback_hit_key = (
        "one|huggingface|sentence-transformers/all-MiniLM-L6-v2|"
        "huggingface:sentence-transformers/all-MiniLM-L6-v2:backend"
    )
    cache = RecordingCache({fallback_hit_key: [0.1, 0.2]})
    executor = RecordingExecutor(
        failures={"openai": primary_error},
        provider_vectors={
            "huggingface": [[0.3, 0.4, 0.5]],
        },
    )
    orchestrator = _orchestrator(
        cache=cache,
        executor=executor,
        settings_fallback_chain={"openai": ["huggingface"]},
        settings_fallback_model_map={
            "openai:text-embedding-3-small": {
                "huggingface": "sentence-transformers/all-MiniLM-L6-v2",
            }
        },
    )
    prepared = orchestrator.prepare(
        ["one", "two"],
        _context(model="text-embedding-3-small", provider="openai"),
    )

    with pytest.raises(EmbeddingProviderError) as exc_info:
        await orchestrator.execute(prepared)

    assert exc_info.value.code == "provider_malformed_response"
    assert exc_info.value.message == (
        "Embedding provider returned malformed embedding vectors"
    )
    assert exc_info.value.provider == "huggingface"
    assert exc_info.value.model == "sentence-transformers/all-MiniLM-L6-v2"
    assert executor.calls == [
        {
            "texts": ["one", "two"],
            "provider": "openai",
            "model": "text-embedding-3-small",
            "dimensions": None,
        },
        {
            "texts": ["two"],
            "provider": "huggingface",
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "dimensions": None,
        },
    ]
    assert cache.get_keys == [
        "one|openai|text-embedding-3-small|openai:text-embedding-3-small:backend",
        "two|openai|text-embedding-3-small|openai:text-embedding-3-small:backend",
        fallback_hit_key,
        (
            "two|huggingface|sentence-transformers/all-MiniLM-L6-v2|"
            "huggingface:sentence-transformers/all-MiniLM-L6-v2:backend"
        ),
    ]
    assert cache.set_calls == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_fallback_execution_uses_retryable_execution_failures():
    openai_failure = EmbeddingExecutionError(
        "circuit_breaker_open",
        "openai circuit open",
        provider="openai",
        model="text-embedding-3-small",
        retryable=True,
    )
    executor = RecordingExecutor(
        failures={"openai": openai_failure},
        provider_vectors={"huggingface": [[0.5, 0.25]]},
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
        "fallback after circuit breaker",
        _context(model="text-embedding-3-small", provider="openai"),
    )

    result = await orchestrator.execute(prepared)

    assert result.provider == "huggingface"
    assert result.fallback_from == "openai"
    assert [call["provider"] for call in executor.calls] == ["openai", "huggingface"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_primary_preflight_failure_propagates_without_cache_or_fallback():
    preflight_error = EmbeddingExecutionError(
        "circuit_breaker_open",
        "openai circuit open",
        provider="openai",
        model="text-embedding-3-small",
        retryable=True,
    )
    preflight_calls: list[tuple[str, str]] = []
    backend_identity_calls: list[tuple[str, str]] = []
    cache_key_calls: list[tuple[str, str, str, int | None, str | None]] = []
    cache = RecordingCache()
    executor = RecordingExecutor(provider_vectors={"huggingface": [[0.5, 0.25]]})

    async def provider_preflight(provider: str, model: str) -> None:
        preflight_calls.append((provider, model))
        raise preflight_error

    def backend_identity_resolver(provider: str, model: str) -> str:
        backend_identity_calls.append((provider, model))
        return f"{provider}:{model}:backend"

    def cache_key_probe(
        text: str,
        provider: str,
        model: str,
        dimensions: int | None = None,
        backend_identity: str | None = None,
    ) -> str:
        cache_key_calls.append(
            (text, provider, model, dimensions, backend_identity)
        )
        return _cache_key(text, provider, model, dimensions, backend_identity)

    orchestrator = _orchestrator(
        cache=cache,
        executor=executor,
        cache_key_fn=cache_key_probe,
        backend_identity_resolver=backend_identity_resolver,
        provider_preflight=provider_preflight,
        settings_fallback_chain={"openai": ["huggingface"]},
        settings_fallback_model_map={
            "openai:text-embedding-3-small": {
                "huggingface": "sentence-transformers/all-MiniLM-L6-v2"
            }
        },
    )
    prepared = orchestrator.prepare(
        "preflight failure",
        _context(model="text-embedding-3-small", provider="openai"),
    )
    assert backend_identity_calls == [("openai", "text-embedding-3-small")]
    backend_identity_calls.clear()

    with pytest.raises(EmbeddingExecutionError) as exc_info:
        await orchestrator.execute(prepared)

    assert exc_info.value is preflight_error
    assert preflight_calls == [("openai", "text-embedding-3-small")]
    assert backend_identity_calls == []
    assert cache_key_calls == []
    assert cache.get_keys == []
    assert cache.set_calls == []
    assert executor.calls == []


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "boundary",
    ["backend_identity", "cache_key", "cache_get"],
)
async def test_primary_runtime_infrastructure_failure_propagates_without_fallback(
    boundary,
):
    original = RuntimeError(f"{boundary} failed")
    events: list[tuple[str, str, str]] = []
    raised_errors: list[RuntimeError] = []
    identity_calls = 0

    def raise_original(provider: str, model: str) -> None:
        events.append((f"raise_{boundary}", provider, model))
        raised_errors.append(original)
        raise original

    def identity_resolver(provider: str, model: str) -> str:
        nonlocal identity_calls
        identity_calls += 1
        events.append(("backend_identity", provider, model))
        if boundary == "backend_identity" and identity_calls > 1:
            raise_original(provider, model)
        return f"{provider}:{model}:backend"

    def cache_key_fn(
        text: str,
        provider: str,
        model: str,
        dimensions: int | None = None,
        backend_identity: str | None = None,
    ) -> str:
        events.append(("cache_key", provider, model))
        if boundary == "cache_key":
            raise_original(provider, model)
        return _cache_key(text, provider, model, dimensions, backend_identity)

    class FailingGetCache(RecordingCache):
        async def get(self, key: str) -> list[float] | None:
            _, provider, model, _ = key.split("|", maxsplit=3)
            events.append(("cache_get", provider, model))
            if boundary == "cache_get":
                raise_original(provider, model)
            return await super().get(key)

    cache = FailingGetCache()
    executor = RecordingExecutor(
        provider_vectors={"huggingface": [[0.5, 0.25]]},
    )
    orchestrator = _orchestrator(
        cache=cache,
        executor=executor,
        cache_key_fn=cache_key_fn,
        backend_identity_resolver=identity_resolver,
        settings_fallback_chain={"openai": ["huggingface"]},
        settings_fallback_model_map={
            "openai:text-embedding-3-small": {
                "huggingface": "sentence-transformers/all-MiniLM-L6-v2"
            }
        },
    )
    prepared = orchestrator.prepare(
        "primary infrastructure failure",
        _context(model="text-embedding-3-small", provider="openai"),
    )
    assert events == [
        ("backend_identity", "openai", "text-embedding-3-small"),
    ]
    events.clear()

    with pytest.raises(RuntimeError) as exc_info:
        await orchestrator.execute(prepared)

    expected_events = {
        "backend_identity": [
            ("backend_identity", "openai", "text-embedding-3-small"),
            ("raise_backend_identity", "openai", "text-embedding-3-small"),
        ],
        "cache_key": [
            ("backend_identity", "openai", "text-embedding-3-small"),
            ("cache_key", "openai", "text-embedding-3-small"),
            ("raise_cache_key", "openai", "text-embedding-3-small"),
        ],
        "cache_get": [
            ("backend_identity", "openai", "text-embedding-3-small"),
            ("cache_key", "openai", "text-embedding-3-small"),
            ("cache_get", "openai", "text-embedding-3-small"),
            ("raise_cache_get", "openai", "text-embedding-3-small"),
        ],
    }
    assert exc_info.value is original
    assert raised_errors == [original]
    assert events == expected_events[boundary]
    assert executor.calls == []
    assert cache.set_calls == []


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
async def test_retryable_fallback_preflight_failure_continues_to_next_candidate():
    openai_failure = EmbeddingProviderError(
        "provider_unavailable",
        "openai unavailable",
        provider="openai",
        model="text-embedding-3-small",
        retryable=True,
    )
    events: list[tuple[str, str, str]] = []
    preflight_calls: list[tuple[str, str]] = []

    class EventRecordingCache(RecordingCache):
        async def get(self, key: str) -> list[float] | None:
            _, provider, model, _ = key.split("|", maxsplit=3)
            events.append(("cache_get", provider, model))
            return await super().get(key)

        async def set(self, key: str, value: list[float]) -> object:
            _, provider, model, _ = key.split("|", maxsplit=3)
            events.append(("cache_set", provider, model))
            return await super().set(key, value)

    class EventRecordingExecutor(RecordingExecutor):
        async def create(
            self,
            texts: list[str],
            *,
            provider: str,
            model: str,
            dimensions: int | None,
        ) -> list[list[float]]:
            events.append(("executor", provider, model))
            try:
                return await super().create(
                    texts,
                    provider=provider,
                    model=model,
                    dimensions=dimensions,
                )
            except EmbeddingProviderError:
                events.append(("executor_error", provider, model))
                raise

    cache = EventRecordingCache()
    executor = EventRecordingExecutor(
        failures={"openai": openai_failure},
        provider_vectors={"huggingface": [[0.5, 0.25]]},
    )

    async def provider_preflight(provider: str, model: str) -> None:
        preflight_calls.append((provider, model))
        events.append(("preflight", provider, model))
        if provider == "cohere":
            events.append(("preflight_error", provider, model))
            raise EmbeddingExecutionError(
                "circuit_breaker_open",
                "cohere circuit open",
                provider=provider,
                model=model,
                retryable=True,
            )

    def backend_identity_resolver(provider: str, model: str) -> str:
        events.append(("identity", provider, model))
        return f"{provider}:{model}:backend"

    def cache_key_probe(
        text: str,
        provider: str,
        model: str,
        dimensions: int | None = None,
        backend_identity: str | None = None,
    ) -> str:
        events.append(("cache_key", provider, model))
        return _cache_key(text, provider, model, dimensions, backend_identity)

    orchestrator = _orchestrator(
        cache=cache,
        executor=executor,
        cache_key_fn=cache_key_probe,
        backend_identity_resolver=backend_identity_resolver,
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
        "fallback after circuit breaker",
        _context(model="text-embedding-3-small", provider="openai"),
    )
    assert events == [("identity", "openai", "text-embedding-3-small")]
    events.clear()

    result = await orchestrator.execute(prepared)

    assert result.provider == "huggingface"
    assert result.vectors == [[0.5, 0.25]]
    assert preflight_calls == [
        ("openai", "text-embedding-3-small"),
        ("cohere", "embed-english-v3.0"),
        ("huggingface", "sentence-transformers/all-MiniLM-L6-v2"),
    ]
    primary = ("openai", "text-embedding-3-small")
    cohere = ("cohere", "embed-english-v3.0")
    fallback = ("huggingface", "sentence-transformers/all-MiniLM-L6-v2")

    def event_index(action: str, provider_model: tuple[str, str]) -> int:
        provider, model = provider_model
        return events.index((action, provider, model))

    primary_preflight = event_index("preflight", primary)
    primary_identity = event_index("identity", primary)
    primary_cache_key = event_index("cache_key", primary)
    primary_cache_get = event_index("cache_get", primary)
    primary_executor = event_index("executor", primary)
    primary_executor_error = event_index("executor_error", primary)
    assert (
        primary_preflight
        < primary_identity
        < primary_cache_key
        < primary_cache_get
        < primary_executor
        < primary_executor_error
    )

    cohere_preflight = event_index("preflight", cohere)
    cohere_preflight_error = event_index("preflight_error", cohere)
    assert primary_executor_error < cohere_preflight < cohere_preflight_error
    assert not any(
        action in {"identity", "cache_key", "cache_get", "cache_set", "executor"}
        and provider == cohere[0]
        for action, provider, _ in events
    )

    fallback_preflight = event_index("preflight", fallback)
    fallback_identity = event_index("identity", fallback)
    fallback_cache_key = event_index("cache_key", fallback)
    fallback_cache_get = event_index("cache_get", fallback)
    fallback_executor = event_index("executor", fallback)
    fallback_cache_set = event_index("cache_set", fallback)
    assert (
        cohere_preflight_error
        < fallback_preflight
        < fallback_identity
        < fallback_cache_key
        < fallback_cache_get
        < fallback_executor
        < fallback_cache_set
    )
    assert [call["provider"] for call in executor.calls] == [
        "openai",
        "huggingface",
    ]


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "boundary",
    ["backend_identity", "cache_key", "cache_get", "cache_set"],
)
async def test_current_fallback_wide_domain_catch_advances_after_infrastructure_failure(
    boundary,
):
    """Pin current behavior that Stage 2D will intentionally correct."""
    cohere_model = "embed-english-v3.0"
    original = EmbeddingExecutionError(
        "internal_execution_failure",
        f"{boundary} failed",
        provider="cohere",
        model=cohere_model,
        retryable=True,
    )
    primary_error = EmbeddingProviderError(
        "provider_unavailable",
        "primary unavailable",
        provider="openai",
        model="text-embedding-3-small",
        retryable=True,
    )
    events: list[tuple[str, str, str]] = []
    raised_errors: list[EmbeddingExecutionError] = []

    def raise_original(provider: str, model: str) -> None:
        events.append((f"raise_{boundary}", provider, model))
        raised_errors.append(original)
        raise original

    async def provider_preflight(provider: str, model: str) -> None:
        events.append(("candidate", provider, model))

    def identity_resolver(provider: str, model: str) -> str:
        events.append(("backend_identity", provider, model))
        if boundary == "backend_identity" and provider == "cohere":
            raise_original(provider, model)
        return f"{provider}:{model}:backend"

    def cache_key_fn(
        text: str,
        provider: str,
        model: str,
        dimensions: int | None = None,
        backend_identity: str | None = None,
    ) -> str:
        events.append(("cache_key", provider, model))
        if boundary == "cache_key" and provider == "cohere":
            raise_original(provider, model)
        return _cache_key(text, provider, model, dimensions, backend_identity)

    class BoundaryCache(RecordingCache):
        async def get(self, key: str) -> list[float] | None:
            _, provider, model, _ = key.split("|", maxsplit=3)
            events.append(("cache_get", provider, model))
            self.get_keys.append(key)
            if boundary == "cache_get" and provider == "cohere":
                raise_original(provider, model)
            return self.values.get(key)

        async def set(self, key: str, value: list[float]) -> object:
            _, provider, model, _ = key.split("|", maxsplit=3)
            events.append(("cache_set", provider, model))
            if boundary == "cache_set" and provider == "cohere":
                raise_original(provider, model)
            return await super().set(key, value)

    class EventRecordingExecutor(RecordingExecutor):
        async def create(
            self,
            texts: list[str],
            *,
            provider: str,
            model: str,
            dimensions: int | None,
        ) -> list[list[float]]:
            events.append(("executor", provider, model))
            return await super().create(
                texts,
                provider=provider,
                model=model,
                dimensions=dimensions,
            )

    cache = BoundaryCache()
    executor = EventRecordingExecutor(
        failures={"openai": primary_error},
        provider_vectors={
            "cohere": [[0.4, 0.6]],
            "huggingface": [[0.5, 0.25]],
        },
    )
    orchestrator = _orchestrator(
        cache=cache,
        executor=executor,
        cache_key_fn=cache_key_fn,
        backend_identity_resolver=identity_resolver,
        provider_preflight=provider_preflight,
        settings_fallback_chain={"openai": ["cohere", "huggingface"]},
        settings_fallback_model_map={
            "openai:text-embedding-3-small": {
                "cohere": cohere_model,
                "huggingface": "sentence-transformers/all-MiniLM-L6-v2",
            }
        },
    )
    prepared = orchestrator.prepare(
        "fallback infrastructure failure",
        _context(model="text-embedding-3-small", provider="openai"),
    )
    assert events == [
        ("backend_identity", "openai", "text-embedding-3-small"),
    ]
    events.clear()

    result = await orchestrator.execute(prepared)

    expected_executor_providers = (
        ["openai", "cohere", "huggingface"]
        if boundary == "cache_set"
        else ["openai", "huggingface"]
    )
    assert result.provider == "huggingface"
    assert result.vectors == [[0.5, 0.25]]
    assert result.fallback_from == "openai"
    assert [
        provider
        for action, provider, _ in events
        if action == "candidate"
    ] == ["openai", "cohere", "huggingface"]
    assert [call["provider"] for call in executor.calls] == expected_executor_providers
    assert [call["provider"] for call in executor.calls].count("openai") == 1
    assert [
        event
        for event in events
        if event == (boundary, "cohere", cohere_model)
    ] == [(boundary, "cohere", cohere_model)]
    assert [
        event
        for event in events
        if event == (f"raise_{boundary}", "cohere", cohere_model)
    ] == [(f"raise_{boundary}", "cohere", cohere_model)]
    assert raised_errors == [original]


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
@pytest.mark.parametrize(
    "cached_values",
    [
        pytest.param((), id="empty"),
        pytest.param(("not-a-number",), id="nonnumeric"),
        pytest.param((float("nan"), 0.0), id="nan"),
        pytest.param((float("inf"), 0.0), id="infinite"),
    ],
)
async def test_malformed_cached_vector_becomes_miss_and_is_replaced(
    cached_values: tuple[object, ...],
):
    cache_key = (
        "replace|huggingface|sentence-transformers/all-MiniLM-L6-v2|"
        "huggingface:sentence-transformers/all-MiniLM-L6-v2:backend"
    )
    cache = RecordingCache(
        {cache_key: list(cached_values)}  # type: ignore[dict-item]
    )
    executor = RecordingExecutor(vectors=[[0.25, 0.75]])
    orchestrator = _orchestrator(cache=cache, executor=executor)
    prepared = orchestrator.prepare("replace", _context())

    result = await orchestrator.execute(prepared)

    assert result.vectors == [[0.25, 0.75]]
    assert result.cache_hits == 0
    assert result.cache_misses == 1
    assert cache.get_keys == [cache_key]
    assert executor.calls == [
        {
            "texts": ["replace"],
            "provider": "huggingface",
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "dimensions": None,
        }
    ]
    assert cache.set_calls == [(cache_key, [0.25, 0.75])]
    assert cache.values[cache_key] == [0.25, 0.75]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_malformed_fallback_cached_vector_becomes_miss_and_is_replaced():
    fallback_model = "sentence-transformers/all-MiniLM-L6-v2"
    fallback_cache_key = (
        f"replace fallback|huggingface|{fallback_model}|"
        f"huggingface:{fallback_model}:backend"
    )
    cache = RecordingCache(
        {fallback_cache_key: [True, 0.0]}  # type: ignore[dict-item]
    )
    primary_error = EmbeddingProviderError(
        "provider_unavailable",
        "openai unavailable",
        provider="openai",
        model="text-embedding-3-small",
        retryable=True,
    )
    executor = RecordingExecutor(
        failures={"openai": primary_error},
        provider_vectors={"huggingface": [[0.25, 0.75]]},
    )
    identity_calls: list[tuple[str, str]] = []

    def backend_identity_resolver(provider: str, model: str) -> str:
        identity_calls.append((provider, model))
        return f"{provider}:{model}:backend"

    orchestrator = _orchestrator(
        cache=cache,
        executor=executor,
        backend_identity_resolver=backend_identity_resolver,
        settings_fallback_chain={"openai": ["huggingface"]},
        settings_fallback_model_map={
            "openai:text-embedding-3-small": {
                "huggingface": fallback_model,
            }
        },
    )
    prepared = orchestrator.prepare(
        "replace fallback",
        _context(model="text-embedding-3-small", provider="openai"),
    )

    result = await orchestrator.execute(prepared)

    assert result.vectors == [[0.25, 0.75]]
    assert result.provider == "huggingface"
    assert result.model == fallback_model
    assert result.fallback_from == "openai"
    assert result.cache_hits == 0
    assert result.cache_misses == 1
    assert cache.get_keys == [
        "replace fallback|openai|text-embedding-3-small|"
        "openai:text-embedding-3-small:backend",
        fallback_cache_key,
    ]
    assert executor.calls == [
        {
            "texts": ["replace fallback"],
            "provider": "openai",
            "model": "text-embedding-3-small",
            "dimensions": None,
        },
        {
            "texts": ["replace fallback"],
            "provider": "huggingface",
            "model": fallback_model,
            "dimensions": None,
        },
    ]
    assert cache.set_calls == [(fallback_cache_key, [0.25, 0.75])]
    assert cache.values[fallback_cache_key] == [0.25, 0.75]
    assert identity_calls == [
        ("openai", "text-embedding-3-small"),
        ("openai", "text-embedding-3-small"),
        ("huggingface", fallback_model),
    ]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_fallback_writeback_re_resolves_backend_identity_after_provider_execution():
    fallback_model = "sentence-transformers/all-MiniLM-L6-v2"
    primary_error = EmbeddingProviderError(
        "provider_unavailable",
        "openai unavailable",
        provider="openai",
        model="text-embedding-3-small",
        retryable=True,
    )
    cache = RecordingCache()
    executor = RecordingExecutor(
        failures={"openai": primary_error},
        provider_vectors={"huggingface": [[0.25, 0.75]]},
    )
    identity_calls: list[tuple[str, str]] = []

    def backend_identity_resolver(provider: str, model: str) -> str:
        identity_calls.append((provider, model))
        if provider == "huggingface":
            fallback_identity_count = sum(
                1 for call in identity_calls if call == (provider, model)
            )
            suffix = "read" if fallback_identity_count == 1 else "write"
            return f"{provider}:{model}:{suffix}"
        return f"{provider}:{model}:identity"

    orchestrator = _orchestrator(
        cache=cache,
        executor=executor,
        backend_identity_resolver=backend_identity_resolver,
        settings_fallback_chain={"openai": ["huggingface"]},
        settings_fallback_model_map={
            "openai:text-embedding-3-small": {
                "huggingface": fallback_model,
            }
        },
    )
    prepared = orchestrator.prepare(
        "identity correction",
        _context(model="text-embedding-3-small", provider="openai"),
    )

    result = await orchestrator.execute(prepared)

    assert result.provider == "huggingface"
    assert cache.get_keys == [
        "identity correction|openai|text-embedding-3-small|"
        "openai:text-embedding-3-small:identity",
        f"identity correction|huggingface|{fallback_model}|"
        f"huggingface:{fallback_model}:read",
    ]
    assert cache.set_calls == [
        (
            f"identity correction|huggingface|{fallback_model}|"
            f"huggingface:{fallback_model}:write",
            [0.25, 0.75],
        )
    ]
    assert identity_calls == [
        ("openai", "text-embedding-3-small"),
        ("openai", "text-embedding-3-small"),
        ("huggingface", fallback_model),
        ("huggingface", fallback_model),
    ]


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
