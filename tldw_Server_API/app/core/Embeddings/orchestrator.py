"""Dependency wiring and compatibility facade for embedding requests."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, Literal, Protocol

from tldw_Server_API.app.core.Embeddings.execution_coordinator import (
    EmbeddingAdapterAttempt,
    EmbeddingExecutionCoordinator,
    EmbeddingFallbackCoordinator,
)
from tldw_Server_API.app.core.Embeddings.preparation import (
    BackendIdentityResolver,
    EmbeddingPreparationPipeline,
    TokenCounter,
    TokenDecoder,
)
from tldw_Server_API.app.core.Embeddings.provider_attempt import (
    EmbeddingProviderAttempt,
    EmbeddingProviderReadinessCheck,
)
from tldw_Server_API.app.core.Embeddings.provider_resolution import ProviderGuesser
from tldw_Server_API.app.core.Embeddings.request_types import (
    EmbeddingExecutionResult,
    EmbeddingExecutorOutput,
    EmbeddingRequestContext,
    PreparedEmbeddingRequest,
)
from tldw_Server_API.app.core.Embeddings.result_mapping import (
    map_outcome_to_legacy_execution_result,
)
from tldw_Server_API.app.core.Embeddings.vector_processing import (
    DimensionAdjustmentRecorder,
    EmbeddingVectorProcessor,
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
ProviderPreflight = Callable[[str, str], Awaitable[None]]


class EmbeddingRequestOrchestrator:
    """Wire concrete components and preserve the temporary legacy result API."""

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
        resolved_backend_identity = backend_identity_resolver or _no_backend_identity
        resolved_provider_preflight = provider_preflight or _no_provider_preflight
        vector_processor = EmbeddingVectorProcessor(
            record_dimension_adjustment=record_dimension_adjustment,
        )
        readiness = EmbeddingProviderReadinessCheck(resolved_provider_preflight)
        provider_attempt = EmbeddingProviderAttempt(
            cache_key_fn=cache_key_fn,
            cache=cache,
            executor=executor,
            backend_identity_resolver=resolved_backend_identity,
            vector_processor=vector_processor,
        )
        adapter_attempt = EmbeddingAdapterAttempt(
            executor=executor,
            vector_processor=vector_processor,
        )
        fallback_coordinator = EmbeddingFallbackCoordinator(
            readiness=readiness,
            provider_attempt=provider_attempt,
            settings_fallback_model_map=settings_fallback_model_map,
        )
        self._execution_coordinator = EmbeddingExecutionCoordinator(
            adapter_attempt=adapter_attempt,
            readiness=readiness,
            provider_attempt=provider_attempt,
            fallback_coordinator=fallback_coordinator,
        )
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
            backend_identity_resolver=resolved_backend_identity,
            cache_namespace=cache_namespace,
            batch_size=batch_size,
            execution_path=execution_path,
        )

    @property
    def preparation_pipeline(self) -> EmbeddingPreparationPipeline:
        """Return the concrete preparation component for the workflow runner."""
        return self._preparation_pipeline

    @property
    def execution_coordinator(self) -> EmbeddingExecutionCoordinator:
        """Return the canonical execution component for the workflow runner."""
        return self._execution_coordinator

    def prepare(self, raw_input: Any, context: EmbeddingRequestContext) -> PreparedEmbeddingRequest:
        """Delegate preparation for compatibility callers."""
        return self._preparation_pipeline.prepare(raw_input, context)

    async def execute(self, prepared: PreparedEmbeddingRequest) -> EmbeddingExecutionResult:
        """Delegate execution and map its canonical outcome for compatibility callers."""
        outcome = await self._execution_coordinator.execute(prepared)
        return map_outcome_to_legacy_execution_result(outcome)


def _no_backend_identity(provider: str, model: str) -> str | None:
    del provider, model
    return None


async def _no_provider_preflight(provider: str, model: str) -> None:
    del provider, model


__all__ = [
    "EmbeddingCache",
    "EmbeddingAdapterExecutor",
    "EmbeddingExecutorOutput",
    "EmbeddingExecutionResult",
    "EmbeddingExecutor",
    "EmbeddingRequestOrchestrator",
    "PreparedEmbeddingRequest",
]
