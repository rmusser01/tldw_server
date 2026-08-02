"""Explicit execution boundaries for the embeddings workflow."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from tldw_Server_API.app.core.Embeddings.embedding_policy import (
    map_model_for_provider,
)
from tldw_Server_API.app.core.Embeddings.provider_attempt import (
    EmbeddingProviderAttempt,
    EmbeddingProviderReadinessCheck,
    ProviderAttemptExecutor,
    ProviderAttemptSuccess,
    ProviderCallFailure,
)
from tldw_Server_API.app.core.Embeddings.request_types import (
    EmbeddingDomainError,
    EmbeddingExecutionError,
    EmbeddingExecutorOutput,
    EmbeddingProviderError,
    PreparedEmbeddingRequest,
)
from tldw_Server_API.app.core.Embeddings.vector_processing import (
    EmbeddingVectorProcessor,
)


@dataclass(frozen=True, slots=True)
class AdapterAttemptResult:
    """Record whether the adapter path ran and whether it produced vectors."""

    attempted: bool
    success: ProviderAttemptSuccess | None = None

    def __post_init__(self) -> None:
        if self.success is not None and not self.attempted:
            raise ValueError("adapter success requires an attempted adapter path")


@dataclass(frozen=True, slots=True)
class FallbackExecutionSuccess:
    """Successful fallback result plus the candidates whose readiness began."""

    success: ProviderAttemptSuccess
    attempt_count: int


class EmbeddingAdapterAttempt:
    """Execute an optional adapter path for one prepared embedding request."""

    def __init__(
        self,
        *,
        executor: ProviderAttemptExecutor,
        vector_processor: EmbeddingVectorProcessor | None = None,
    ) -> None:
        self._executor = executor
        self._vector_processor = vector_processor or EmbeddingVectorProcessor()

    async def execute(self, prepared: PreparedEmbeddingRequest) -> AdapterAttemptResult:
        """Attempt adapter execution, preserving adapter exceptions for the caller."""
        plan = prepared.execution_plan
        if plan.execution_path != "adapter":
            return AdapterAttemptResult(attempted=False)

        create_adapter = getattr(self._executor, "create_adapter", None)
        if not callable(create_adapter):
            return AdapterAttemptResult(attempted=False)

        output = await create_adapter(
            prepared.normalized_input.texts,
            provider=plan.provider,
            model=plan.model,
            dimensions=plan.dimensions,
        )
        if not isinstance(output, EmbeddingExecutorOutput) or not output.embeddings_from_adapter:
            return AdapterAttemptResult(attempted=True)

        vectors = self._vector_processor.validate_vector_count(
            output.vectors,
            expected=len(prepared.normalized_input.texts),
            provider=plan.provider,
            model=plan.model,
        )
        processed_vectors = self._vector_processor.process_vectors(
            vectors,
            provider=plan.provider,
            model=plan.model,
            dimensions=plan.dimensions,
            dimension_policy=prepared.effective_dimension_policy,
        )
        return AdapterAttemptResult(
            attempted=True,
            success=ProviderAttemptSuccess(
                vectors=processed_vectors,
                provider=plan.provider,
                model=plan.model,
                cache_hits=0,
                cache_misses=len(prepared.normalized_input.texts),
                embeddings_from_adapter=True,
            ),
        )


class EmbeddingFallbackCoordinator:
    """Route a failed primary provider through ordered fallback candidates."""

    def __init__(
        self,
        *,
        readiness: EmbeddingProviderReadinessCheck,
        provider_attempt: EmbeddingProviderAttempt,
        settings_fallback_model_map: Mapping[str, object] | None,
    ) -> None:
        self._readiness = readiness
        self._provider_attempt = provider_attempt
        self._settings_fallback_model_map = settings_fallback_model_map

    async def execute(
        self,
        prepared: PreparedEmbeddingRequest,
        primary_failure: ProviderCallFailure,
    ) -> FallbackExecutionSuccess:
        """Attempt every non-primary candidate until one returns complete vectors."""
        plan = prepared.execution_plan
        errors: list[EmbeddingDomainError] = [primary_failure.error]
        attempt_count = 0

        for provider in plan.fallback_chain:
            if provider == plan.provider:
                continue

            model = map_model_for_provider(
                plan.provider,
                provider,
                plan.model,
                settings_fallback_model_map=self._settings_fallback_model_map,
            )
            attempt_count += 1
            try:
                await self._readiness.check(provider, model)
            except EmbeddingDomainError as error:
                if error.code == "missing_provider_credentials":
                    continue
                errors.append(error)
                if not _is_fallback_eligible(error):
                    raise
                continue

            attempt_result = await self._provider_attempt.execute(
                prepared,
                provider=provider,
                model=model,
            )
            if isinstance(attempt_result, ProviderAttemptSuccess):
                return FallbackExecutionSuccess(attempt_result, attempt_count)

            error = attempt_result.error
            if error.code == "missing_provider_credentials":
                continue
            errors.append(error)
            if not _is_fallback_eligible(error):
                raise error

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


def _select_exhausted_error(
    errors: Sequence[EmbeddingDomainError],
) -> EmbeddingDomainError | None:
    for error in errors:
        if isinstance(error, EmbeddingProviderError) and not error.retryable:
            return error
    for error in errors:
        if error.code == "provider_rate_limited":
            return error
    return errors[-1] if errors else None


__all__ = [
    "AdapterAttemptResult",
    "EmbeddingAdapterAttempt",
    "EmbeddingFallbackCoordinator",
    "FallbackExecutionSuccess",
]
