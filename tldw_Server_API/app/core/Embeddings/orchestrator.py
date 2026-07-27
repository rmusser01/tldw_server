"""Pure prepare and execute orchestration for embedding requests."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from tldw_Server_API.app.core.Embeddings.embedding_policy import (
    adjust_dimensions,
    map_model_for_provider,
)
from tldw_Server_API.app.core.Embeddings.preparation import (
    BackendIdentityResolver,
    EmbeddingPreparationPipeline,
    TokenCounter,
    TokenDecoder,
)
from tldw_Server_API.app.core.Embeddings.provider_resolution import ProviderGuesser
from tldw_Server_API.app.core.Embeddings.request_types import (
    EmbeddingDomainError,
    EmbeddingExecutionError,
    EmbeddingExecutionResult,
    EmbeddingProviderError,
    EmbeddingRequestContext,
    PreparedEmbeddingRequest,
)
from tldw_Server_API.app.core.Embeddings.vector_validation import (
    validated_embedding_vectors,
)


class EmbeddingCache(Protocol):
    async def get(self, key: str) -> list[float] | None:
        raise NotImplementedError

    async def set(self, key: str, value: list[float]) -> object:
        raise NotImplementedError


class EmbeddingExecutor(Protocol):
    async def create(
        self,
        texts: list[str],
        *,
        provider: str,
        model: str,
        dimensions: int | None,
    ) -> list[list[float]] | EmbeddingExecutorOutput:
        raise NotImplementedError


class EmbeddingAdapterExecutor(Protocol):
    async def create_adapter(
        self,
        texts: list[str],
        *,
        provider: str,
        model: str,
        dimensions: int | None,
    ) -> EmbeddingExecutorOutput | None:
        raise NotImplementedError


CacheKeyFn = Callable[[str, str, str, int | None, str | None], str]
DimensionAdjustmentRecorder = Callable[[str, str, str], None]
ProviderPreflight = Callable[[str, str], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class EmbeddingExecutorOutput:
    vectors: list[list[float]]
    embeddings_from_adapter: bool = False


@dataclass(frozen=True, slots=True)
class _ProviderExecution:
    vectors: list[list[float]]
    provider: str
    model: str
    fallback_from: str | None
    cache_vectors: list[list[float]] | None = None
    cache_hits: int = 0
    cache_misses: int = 0
    complete_response: bool = False
    embeddings_from_adapter: bool = False


class EmbeddingRequestOrchestrator:
    """Prepare and execute embedding requests without endpoint dependencies."""

    def __init__(
        self,
        *,
        count_tokens: TokenCounter,
        tokens_to_texts: TokenDecoder,
        cache_key_fn: CacheKeyFn,
        cache: EmbeddingCache,
        executor: EmbeddingExecutor,
        settings_config: Mapping[str, object] | None,
        max_tokens: int,
        implemented_providers: Sequence[str] | set[str],
        allowed_providers: Sequence[str] | set[str] | None,
        allowed_models: Sequence[str] | set[str] | None,
        enforce_policy: bool,
        allow_fallback_with_header: bool,
        settings_fallback_chain: Mapping[str, object] | None,
        settings_fallback_model_map: Mapping[str, object] | None,
        dimension_policy: str = "reduce",
        require_model: bool = True,
        guess_provider: ProviderGuesser | None = None,
        backend_identity_resolver: BackendIdentityResolver | None = None,
        record_dimension_adjustment: DimensionAdjustmentRecorder | None = None,
        provider_preflight: ProviderPreflight | None = None,
        cache_namespace: str | None = None,
        batch_size: int | None = None,
        execution_path: Literal["legacy", "adapter"] = "legacy",
    ) -> None:
        self._cache_key_fn = cache_key_fn
        self._cache = cache
        self._executor = executor
        self._settings_fallback_model_map = settings_fallback_model_map
        self._backend_identity_resolver = backend_identity_resolver or _no_backend_identity
        self._record_dimension_adjustment = record_dimension_adjustment
        self._provider_preflight = provider_preflight or _no_provider_preflight
        self._preparation_pipeline = EmbeddingPreparationPipeline(
            count_tokens=count_tokens,
            tokens_to_texts=tokens_to_texts,
            settings_config=settings_config,
            max_tokens=max_tokens,
            implemented_providers=implemented_providers,
            allowed_providers=allowed_providers,
            allowed_models=allowed_models,
            enforce_policy=enforce_policy,
            allow_fallback_with_header=allow_fallback_with_header,
            settings_fallback_chain=settings_fallback_chain,
            settings_fallback_model_map=settings_fallback_model_map,
            dimension_policy=dimension_policy,
            require_model=require_model,
            guess_provider=guess_provider,
            backend_identity_resolver=self._backend_identity_resolver,
            cache_namespace=cache_namespace,
            batch_size=batch_size,
            execution_path=execution_path,
        )

    def prepare(self, raw_input: Any, context: EmbeddingRequestContext) -> PreparedEmbeddingRequest:
        """Normalize, resolve, validate, and plan an embedding request."""
        return self._preparation_pipeline.prepare(raw_input, context)

    async def execute(self, prepared: PreparedEmbeddingRequest) -> EmbeddingExecutionResult:
        """Execute a prepared request through cache and provider executor."""
        plan = prepared.execution_plan
        adapter_execution = await self._execute_adapter(prepared)
        if adapter_execution is not None:
            headers = self._response_headers(
                actual_provider=adapter_execution.provider,
                fallback_from=adapter_execution.fallback_from,
                dimensions=plan.dimensions,
                dimension_policy=prepared.effective_dimension_policy,
            )
            return EmbeddingExecutionResult(
                vectors=adapter_execution.vectors,
                provider=adapter_execution.provider,
                model=adapter_execution.model,
                prompt_tokens=prepared.prompt_tokens,
                total_tokens=prepared.total_tokens,
                cache_hits=0,
                cache_misses=len(prepared.normalized_input.texts),
                fallback_from=adapter_execution.fallback_from,
                response_headers=headers,
                embeddings_from_adapter=True,
            )

        results: list[list[float] | None] = []
        miss_indices: list[int] = []
        miss_texts: list[str] = []

        await self._provider_preflight(plan.provider, plan.model)
        primary_backend_identity = self._backend_identity_resolver(plan.provider, plan.model)
        for index, text in enumerate(prepared.normalized_input.texts):
            key = self._cache_key(text, plan.provider, plan.model, plan.dimensions, primary_backend_identity)
            cached = await self._cache.get(key)
            validated_cached = validated_embedding_vectors([cached], expected=1) if cached is not None else None
            if validated_cached is not None:
                results.append(
                    self._postprocess_cached_vector(
                        validated_cached[0],
                        plan.provider,
                        plan.model,
                        plan.dimensions,
                        prepared.effective_dimension_policy,
                    )
                )
                continue
            results.append(None)
            miss_indices.append(index)
            miss_texts.append(text)

        actual_provider = plan.provider
        actual_model = plan.model
        fallback_from: str | None = None
        embeddings_from_adapter = False
        pending_cache_writes: list[tuple[str, list[float]]] = []
        if miss_texts:
            execution = await self._execute_misses(prepared, miss_texts)
            actual_provider = execution.provider
            actual_model = execution.model
            fallback_from = execution.fallback_from
            embeddings_from_adapter = execution.embeddings_from_adapter
            if execution.complete_response:
                results = list(execution.vectors)
                cache_hits = execution.cache_hits
                cache_misses = execution.cache_misses
            else:
                backend_identity = self._backend_identity_resolver(actual_provider, actual_model)
                cache_vectors = execution.cache_vectors or execution.vectors
                for index, text, vector, cache_vector in zip(
                    miss_indices,
                    miss_texts,
                    execution.vectors,
                    cache_vectors,
                ):
                    results[index] = vector
                    if not execution.embeddings_from_adapter:
                        key = self._cache_key(text, actual_provider, actual_model, plan.dimensions, backend_identity)
                        pending_cache_writes.append((key, cache_vector))
                cache_hits = len(results) - len(miss_indices)
                cache_misses = len(miss_indices)
        else:
            cache_hits = len(results)
            cache_misses = 0

        vectors = self._validate_vector_count(
            results,
            expected=len(prepared.normalized_input.texts),
            provider=actual_provider,
            model=actual_model,
        )
        for key, cache_vector in pending_cache_writes:
            await self._cache.set(key, cache_vector)
        headers = self._response_headers(
            actual_provider=actual_provider,
            fallback_from=fallback_from,
            dimensions=plan.dimensions,
            dimension_policy=prepared.effective_dimension_policy,
        )
        return EmbeddingExecutionResult(
            vectors=vectors,
            provider=actual_provider,
            model=actual_model,
            prompt_tokens=prepared.prompt_tokens,
            total_tokens=prepared.total_tokens,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            fallback_from=fallback_from,
            response_headers=headers,
            embeddings_from_adapter=embeddings_from_adapter,
        )

    async def _execute_misses(
        self,
        prepared: PreparedEmbeddingRequest,
        miss_texts: list[str],
    ) -> _ProviderExecution:
        plan = prepared.execution_plan
        chain = plan.fallback_chain or [plan.provider]
        errors: list[EmbeddingDomainError] = []

        for provider in chain:
            model = map_model_for_provider(
                plan.provider,
                provider,
                plan.model,
                settings_fallback_model_map=self._settings_fallback_model_map,
            )
            if provider != plan.provider:
                try:
                    return await self._execute_coherent_fallback(prepared, provider, model)
                except EmbeddingDomainError as exc:
                    if exc.code == "missing_provider_credentials":
                        continue
                    errors.append(exc)
                    if not _is_fallback_eligible(exc):
                        raise
                    continue

            try:
                output = await self._executor.create(
                    miss_texts,
                    provider=provider,
                    model=model,
                    dimensions=plan.dimensions,
                )
                vectors, embeddings_from_adapter = _coerce_executor_output(output)
            except EmbeddingDomainError as exc:
                errors.append(exc)
                if not prepared.policy_decision.fallback_allowed or not _is_fallback_eligible(exc):
                    raise
                continue

            vectors = self._validate_vector_count(
                vectors,
                expected=len(miss_texts),
                provider=provider,
                model=model,
            )
            cache_vectors = [_canonical_vector(vector) for vector in vectors]
            canonical_vectors = self._postprocess_vectors(
                cache_vectors,
                provider,
                model,
                plan.dimensions,
                prepared.effective_dimension_policy,
            )
            return _ProviderExecution(
                vectors=canonical_vectors,
                provider=provider,
                model=model,
                fallback_from=plan.provider if provider != plan.provider else None,
                cache_vectors=cache_vectors,
                embeddings_from_adapter=embeddings_from_adapter,
            )

        selected_error = _select_exhausted_error(errors)
        if selected_error is not None:
            raise selected_error
        raise EmbeddingExecutionError(
            "fallback_exhausted",
            "Embedding providers unavailable",
            retryable=True,
            provider=plan.provider,
            model=plan.model,
        )

    async def _execute_coherent_fallback(
        self,
        prepared: PreparedEmbeddingRequest,
        provider: str,
        model: str,
    ) -> _ProviderExecution:
        plan = prepared.execution_plan
        results: list[list[float] | None] = []
        miss_indices: list[int] = []
        miss_texts: list[str] = []

        await self._provider_preflight(provider, model)
        backend_identity = self._backend_identity_resolver(provider, model)
        for index, text in enumerate(prepared.normalized_input.texts):
            key = self._cache_key(text, provider, model, plan.dimensions, backend_identity)
            cached = await self._cache.get(key)
            validated_cached = validated_embedding_vectors([cached], expected=1) if cached is not None else None
            if validated_cached is not None:
                results.append(
                    self._postprocess_cached_vector(
                        validated_cached[0],
                        provider,
                        model,
                        plan.dimensions,
                        prepared.effective_dimension_policy,
                    )
                )
                continue
            results.append(None)
            miss_indices.append(index)
            miss_texts.append(text)

        embeddings_from_adapter = False
        pending_cache_writes: list[tuple[str, list[float]]] = []
        if miss_texts:
            output = await self._executor.create(
                miss_texts,
                provider=provider,
                model=model,
                dimensions=plan.dimensions,
            )
            vectors, embeddings_from_adapter = _coerce_executor_output(output)
            vectors = self._validate_vector_count(
                vectors,
                expected=len(miss_texts),
                provider=provider,
                model=model,
            )
            cache_vectors = [_canonical_vector(vector) for vector in vectors]
            canonical_vectors = self._postprocess_vectors(
                cache_vectors,
                provider,
                model,
                plan.dimensions,
                prepared.effective_dimension_policy,
            )
            for index, text, vector, cache_vector in zip(
                miss_indices,
                miss_texts,
                canonical_vectors,
                cache_vectors,
            ):
                results[index] = vector
                if not embeddings_from_adapter:
                    key = self._cache_key(text, provider, model, plan.dimensions, backend_identity)
                    pending_cache_writes.append((key, cache_vector))

        vectors = self._validate_vector_count(
            results,
            expected=len(prepared.normalized_input.texts),
            provider=provider,
            model=model,
        )
        for key, cache_vector in pending_cache_writes:
            await self._cache.set(key, cache_vector)

        return _ProviderExecution(
            vectors=vectors,
            provider=provider,
            model=model,
            fallback_from=plan.provider,
            cache_hits=len(results) - len(miss_indices),
            cache_misses=len(miss_indices),
            complete_response=True,
            embeddings_from_adapter=embeddings_from_adapter,
        )

    async def _execute_adapter(self, prepared: PreparedEmbeddingRequest) -> _ProviderExecution | None:
        plan = prepared.execution_plan
        if plan.execution_path != "adapter":
            return None
        create_adapter = getattr(self._executor, "create_adapter", None)
        if not callable(create_adapter):
            return None

        output = await create_adapter(
            prepared.normalized_input.texts,
            provider=plan.provider,
            model=plan.model,
            dimensions=plan.dimensions,
        )
        if output is None:
            return None
        vectors, embeddings_from_adapter = _coerce_executor_output(output)
        if not embeddings_from_adapter:
            return None
        vectors = self._validate_vector_count(
            vectors,
            expected=len(prepared.normalized_input.texts),
            provider=plan.provider,
            model=plan.model,
        )
        canonical_vectors = self._postprocess_vectors(
            vectors,
            plan.provider,
            plan.model,
            plan.dimensions,
            prepared.effective_dimension_policy,
        )
        return _ProviderExecution(
            vectors=canonical_vectors,
            provider=plan.provider,
            model=plan.model,
            fallback_from=None,
            complete_response=True,
            embeddings_from_adapter=True,
        )

    def _postprocess_vectors(
        self,
        vectors: list[list[float]],
        provider: str,
        model: str,
        dimensions: int | None,
        dimension_policy: str,
    ) -> list[list[float]]:
        canonical = [_canonical_vector(vector) for vector in vectors]
        if dimensions is None:
            return canonical
        adjusted = adjust_dimensions(
            canonical,
            dimensions,
            provider,
            model,
            dimension_policy=dimension_policy,
            record_adjustment=self._record_dimension_adjustment,
        )
        return [_canonical_vector(vector) for vector in adjusted]

    def _postprocess_cached_vector(
        self,
        vector: list[float],
        provider: str,
        model: str,
        dimensions: int | None,
        dimension_policy: str,
    ) -> list[float]:
        return self._postprocess_vectors(
            [_canonical_vector(vector)],
            provider,
            model,
            dimensions,
            dimension_policy,
        )[0]

    @staticmethod
    def _validate_vector_count(
        vectors: object,
        *,
        expected: int,
        provider: str,
        model: str,
    ) -> list[list[float]]:
        validated = validated_embedding_vectors(vectors, expected=expected)
        if validated is None:
            count: int | str = len(vectors) if isinstance(vectors, list) else "invalid"
            message = (
                f"Embedding provider returned {count} embeddings, expected {expected}"
                if count != expected
                else "Embedding provider returned malformed embedding vectors"
            )
            raise EmbeddingProviderError(
                "provider_malformed_response",
                message,
                provider=provider,
                model=model,
            )
        return validated

    def _cache_key(
        self,
        text: str,
        provider: str,
        model: str,
        dimensions: int | None,
        backend_identity: str | None,
    ) -> str:
        return self._cache_key_fn(text, provider, model, dimensions, backend_identity)

    def _response_headers(
        self,
        *,
        actual_provider: str,
        fallback_from: str | None,
        dimensions: int | None,
        dimension_policy: str,
    ) -> dict[str, str]:
        headers: dict[str, str] = {"X-Embeddings-Provider": actual_provider}
        if fallback_from and fallback_from != actual_provider:
            headers["X-Embeddings-Fallback-From"] = fallback_from
        if dimensions is not None:
            headers["X-Embeddings-Dimensions-Policy"] = dimension_policy
        return headers


def _canonical_vector(vector: object) -> list[float]:
    if not isinstance(vector, (list, tuple)):
        raise EmbeddingProviderError(
            "provider_malformed_response",
            "Embedding provider returned a malformed vector",
        )
    try:
        return [float(item) for item in vector]
    except (TypeError, ValueError) as exc:
        raise EmbeddingProviderError(
            "provider_malformed_response",
            "Embedding provider returned a malformed vector",
        ) from exc


def _coerce_executor_output(
    output: list[list[float]] | EmbeddingExecutorOutput,
) -> tuple[list[list[float]], bool]:
    if isinstance(output, EmbeddingExecutorOutput):
        return output.vectors, output.embeddings_from_adapter
    return output, False


def _no_backend_identity(provider: str, model: str) -> str | None:
    del provider, model
    return None


async def _no_provider_preflight(provider: str, model: str) -> None:
    del provider, model


_NON_FALLBACKABLE_ERROR_CODES = frozenset(
    {
        "empty_input",
        "invalid_input_type",
        "too_many_inputs",
        "input_too_long",
        "invalid_token_array",
        "unknown_provider",
        "provider_model_mismatch",
        "invalid_dimensions",
        "provider_denied",
        "model_denied",
        "model_required",
        "provider_unsupported",
        "missing_provider_credentials",
        "provider_malformed_response",
    }
)


def _is_fallback_eligible(error: EmbeddingDomainError) -> bool:
    return bool(error.retryable) and error.code not in _NON_FALLBACKABLE_ERROR_CODES


def _select_exhausted_error(errors: list[EmbeddingDomainError]) -> EmbeddingDomainError | None:
    for error in errors:
        if isinstance(error, EmbeddingProviderError) and not error.retryable:
            return error
    for error in errors:
        if error.code == "provider_rate_limited":
            return error
    return errors[-1] if errors else None


__all__ = [
    "EmbeddingCache",
    "EmbeddingAdapterExecutor",
    "EmbeddingExecutorOutput",
    "EmbeddingExecutionResult",
    "EmbeddingExecutor",
    "EmbeddingRequestOrchestrator",
    "PreparedEmbeddingRequest",
]
