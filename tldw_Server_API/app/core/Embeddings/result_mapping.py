"""Pure mappings from canonical embedding outcomes to boundary contracts.

HTTP response-header construction is endpoint-owned. The endpoint does not switch
to the pure header mapper until Stage 2E. ``map_outcome_to_legacy_execution_result``
is the sole temporary exception: it lives outside the canonical runner path and is
scheduled for removal in Stage 6.
"""

from __future__ import annotations

from collections.abc import Sequence

from tldw_Server_API.app.core.Embeddings.request_types import (
    EmbeddingExecutionOutcome,
    EmbeddingExecutionResult,
    PreparedEmbeddingRequest,
)


def assemble_embedding_execution_outcome(
    prepared: PreparedEmbeddingRequest,
    *,
    vectors: Sequence[Sequence[float]],
    provider: str,
    model: str,
    cache_hits: int,
    cache_misses: int,
    fallback_from: str | None,
    embeddings_from_adapter: bool,
    attempt_count: int,
    fallback_attempt_count: int,
) -> EmbeddingExecutionOutcome:
    """Assemble an immutable outcome from already-processed execution values."""
    return EmbeddingExecutionOutcome(
        vectors=tuple(tuple(vector) for vector in vectors),
        provider=provider,
        model=model,
        prompt_tokens=prepared.prompt_tokens,
        total_tokens=prepared.total_tokens,
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        requested_dimensions=prepared.execution_plan.dimensions,
        effective_dimension_policy=prepared.effective_dimension_policy,
        attempt_count=attempt_count,
        fallback_attempt_count=fallback_attempt_count,
        fallback_from=fallback_from,
        embeddings_from_adapter=embeddings_from_adapter,
    )


def map_embedding_response_headers(
    outcome: EmbeddingExecutionOutcome,
) -> dict[str, str]:
    """Build endpoint-owned headers for endpoint adoption deferred until Stage 2E."""
    headers = {"X-Embeddings-Provider": outcome.provider}
    if outcome.fallback_from and outcome.fallback_from != outcome.provider:
        headers["X-Embeddings-Fallback-From"] = outcome.fallback_from
    if outcome.requested_dimensions is not None:
        headers["X-Embeddings-Dimensions-Policy"] = outcome.effective_dimension_policy
    return headers


def map_outcome_to_legacy_execution_result(
    outcome: EmbeddingExecutionOutcome,
) -> EmbeddingExecutionResult:
    """Return the sole temporary exception to endpoint-owned header construction.

    This mapper lives outside the canonical runner path and is scheduled for
    removal in Stage 6. The endpoint remains on its existing header path until
    Stage 2E.
    """
    return EmbeddingExecutionResult(
        vectors=[list(vector) for vector in outcome.vectors],
        provider=outcome.provider,
        model=outcome.model,
        prompt_tokens=outcome.prompt_tokens,
        total_tokens=outcome.total_tokens,
        cache_hits=outcome.cache_hits,
        cache_misses=outcome.cache_misses,
        fallback_from=outcome.fallback_from,
        response_headers=map_embedding_response_headers(outcome),
        embeddings_from_adapter=outcome.embeddings_from_adapter,
    )


__all__ = [
    "assemble_embedding_execution_outcome",
    "map_embedding_response_headers",
    "map_outcome_to_legacy_execution_result",
]
