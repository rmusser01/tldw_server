"""Explicit execution boundaries for the embeddings workflow."""

from __future__ import annotations

from dataclasses import dataclass

from tldw_Server_API.app.core.Embeddings.provider_attempt import (
    ProviderAttemptExecutor,
    ProviderAttemptSuccess,
)
from tldw_Server_API.app.core.Embeddings.request_types import (
    EmbeddingExecutorOutput,
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


__all__ = ["AdapterAttemptResult", "EmbeddingAdapterAttempt"]
