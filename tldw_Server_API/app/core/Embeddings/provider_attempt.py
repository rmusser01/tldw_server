"""Provider readiness and single-provider execution attempts for embeddings."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol

from tldw_Server_API.app.core.Embeddings.preparation import BackendIdentityResolver
from tldw_Server_API.app.core.Embeddings.request_types import (
    EmbeddingDomainError,
    EmbeddingExecutorOutput,
    PreparedEmbeddingRequest,
)
from tldw_Server_API.app.core.Embeddings.vector_processing import (
    EmbeddingVectorProcessor,
)

CacheKeyFn = Callable[[str, str, str, int | None, str | None], str]
ProviderPreflight = Callable[[str, str], Awaitable[None]]


async def _no_provider_preflight(provider: str, model: str) -> None:
    del provider, model


class EmbeddingProviderReadinessCheck:
    """Run provider readiness checks without cache or execution side effects."""

    def __init__(self, provider_preflight: ProviderPreflight | None = None) -> None:
        self._provider_preflight = provider_preflight or _no_provider_preflight

    async def check(self, provider: str, model: str) -> None:
        await self._provider_preflight(provider, model)


class ProviderAttemptCache(Protocol):
    async def get(self, key: str) -> list[float] | None:
        raise NotImplementedError

    async def set(self, key: str, value: list[float]) -> object:
        raise NotImplementedError


class ProviderAttemptExecutor(Protocol):
    async def create(
        self,
        texts: list[str],
        *,
        provider: str,
        model: str,
        dimensions: int | None,
    ) -> list[list[float]] | EmbeddingExecutorOutput:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class ProviderAttemptSuccess:
    vectors: list[list[float]]
    provider: str
    model: str
    cache_hits: int
    cache_misses: int
    embeddings_from_adapter: bool = False


@dataclass(frozen=True, slots=True)
class ProviderCallFailure:
    error: EmbeddingDomainError


class EmbeddingProviderAttempt:
    """Execute one provider/model against ordered input and cache boundaries."""

    def __init__(
        self,
        *,
        cache_key_fn: CacheKeyFn,
        cache: ProviderAttemptCache,
        executor: ProviderAttemptExecutor,
        backend_identity_resolver: BackendIdentityResolver,
        vector_processor: EmbeddingVectorProcessor | None = None,
    ) -> None:
        self._cache_key_fn = cache_key_fn
        self._cache = cache
        self._executor = executor
        self._backend_identity_resolver = backend_identity_resolver
        self._vector_processor = vector_processor or EmbeddingVectorProcessor()

    async def execute(
        self,
        prepared: PreparedEmbeddingRequest,
        *,
        provider: str,
        model: str,
    ) -> ProviderAttemptSuccess | ProviderCallFailure:
        plan = prepared.execution_plan
        read_identity = self._backend_identity_resolver(provider, model)
        results: list[list[float] | None] = []
        miss_indices: list[int] = []
        miss_texts: list[str] = []

        for index, text in enumerate(prepared.normalized_input.texts):
            key = self._cache_key_fn(
                text,
                provider,
                model,
                plan.dimensions,
                read_identity,
            )
            cached = await self._cache.get(key)
            cached_vector = self._vector_processor.validate_cached_vector(cached)
            if cached_vector is None:
                results.append(None)
                miss_indices.append(index)
                miss_texts.append(text)
                continue
            results.append(
                self._vector_processor.process_cached_vector(
                    cached_vector,
                    provider=provider,
                    model=model,
                    dimensions=plan.dimensions,
                    dimension_policy=prepared.effective_dimension_policy,
                )
            )

        embeddings_from_adapter = False
        pending_cache_writes: list[tuple[str, list[float]]] = []
        if miss_texts:
            try:
                output = await self._executor.create(
                    miss_texts,
                    provider=provider,
                    model=model,
                    dimensions=plan.dimensions,
                )
            except EmbeddingDomainError as exc:
                return ProviderCallFailure(exc)

            miss_vectors, embeddings_from_adapter = _coerce_executor_output(output)
            cache_vectors = self._vector_processor.validate_vector_count(
                miss_vectors,
                expected=len(miss_texts),
                provider=provider,
                model=model,
            )
            processed_misses = self._vector_processor.process_vectors(
                cache_vectors,
                provider=provider,
                model=model,
                dimensions=plan.dimensions,
                dimension_policy=prepared.effective_dimension_policy,
            )
            write_identity = self._backend_identity_resolver(provider, model)
            for index, text, vector, cache_vector in zip(
                miss_indices,
                miss_texts,
                processed_misses,
                cache_vectors,
            ):
                results[index] = vector
                if not embeddings_from_adapter:
                    key = self._cache_key_fn(
                        text,
                        provider,
                        model,
                        plan.dimensions,
                        write_identity,
                    )
                    pending_cache_writes.append((key, cache_vector))

        vectors = self._vector_processor.validate_vector_count(
            results,
            expected=len(prepared.normalized_input.texts),
            provider=provider,
            model=model,
        )
        for key, vector in pending_cache_writes:
            await self._cache.set(key, vector)
        return ProviderAttemptSuccess(
            vectors=vectors,
            provider=provider,
            model=model,
            cache_hits=len(results) - len(miss_indices),
            cache_misses=len(miss_indices),
            embeddings_from_adapter=embeddings_from_adapter,
        )


def _coerce_executor_output(
    output: list[list[float]] | EmbeddingExecutorOutput,
) -> tuple[list[list[float]], bool]:
    if isinstance(output, EmbeddingExecutorOutput):
        return output.vectors, output.embeddings_from_adapter
    return output, False


__all__ = [
    "CacheKeyFn",
    "EmbeddingExecutorOutput",
    "EmbeddingProviderAttempt",
    "EmbeddingProviderReadinessCheck",
    "ProviderAttemptCache",
    "ProviderAttemptExecutor",
    "ProviderAttemptSuccess",
    "ProviderCallFailure",
    "ProviderPreflight",
]
