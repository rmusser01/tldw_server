from __future__ import annotations

import pytest

from tldw_Server_API.app.core.Embeddings.request_types import (
    EmbeddingExecutionOutcome,
    EmbeddingExecutionPlan,
    EmbeddingExecutionResult,
    EmbeddingPolicyDecision,
    NormalizedEmbeddingInput,
    PreparedEmbeddingRequest,
    ProviderModelIntent,
)
from tldw_Server_API.app.core.Embeddings.result_mapping import (
    assemble_embedding_execution_outcome,
    map_embedding_response_headers,
    map_outcome_to_legacy_execution_result,
)


def _prepared_request(*, dimensions: int | None = None) -> PreparedEmbeddingRequest:
    return PreparedEmbeddingRequest(
        normalized_input=NormalizedEmbeddingInput(
            texts=["one", "two"],
            token_counts=[1, 1],
            total_tokens=2,
        ),
        provider_intent=ProviderModelIntent(
            provider="openai",
            model="text-embedding-3-small",
            requested_provider="openai",
            requested_model="text-embedding-3-small",
            provider_was_explicit=True,
            model_was_provider_qualified=False,
        ),
        policy_decision=EmbeddingPolicyDecision(
            provider="openai",
            model="text-embedding-3-small",
            dimensions=dimensions,
            fallback_chain=["openai", "huggingface"],
            fallback_allowed=True,
            enforce_policy=True,
        ),
        execution_plan=EmbeddingExecutionPlan(
            provider="openai",
            model="text-embedding-3-small",
            dimensions=dimensions,
            backend_identity="openai:test",
            fallback_chain=["openai", "huggingface"],
        ),
        effective_dimension_policy="reduce",
        prompt_tokens=2,
        total_tokens=2,
    )


def _outcome(
    *,
    provider: str = "openai",
    fallback_from: str | None = None,
    requested_dimensions: int | None = None,
) -> EmbeddingExecutionOutcome:
    return EmbeddingExecutionOutcome(
        vectors=((0.1, 0.2),),
        provider=provider,
        model=("text-embedding-3-small" if provider == "openai" else "all-MiniLM-L6-v2"),
        prompt_tokens=2,
        total_tokens=2,
        cache_hits=0,
        cache_misses=1,
        requested_dimensions=requested_dimensions,
        effective_dimension_policy="reduce",
        attempt_count=2 if fallback_from else 1,
        fallback_attempt_count=1 if fallback_from else 0,
        fallback_from=fallback_from,
        embeddings_from_adapter=False,
    )


@pytest.mark.unit
def test_result_assembly_freezes_vectors_and_preserves_adapter_cache_counts():
    vectors = [[0.1, 0.2], [0.3, 0.4]]

    outcome = assemble_embedding_execution_outcome(
        _prepared_request(),
        vectors=vectors,
        provider="openai",
        model="text-embedding-3-small",
        cache_hits=0,
        cache_misses=2,
        fallback_from=None,
        embeddings_from_adapter=True,
        attempt_count=1,
        fallback_attempt_count=0,
    )
    vectors[0][0] = 9.0

    assert outcome.vectors == ((0.1, 0.2), (0.3, 0.4))
    assert (outcome.cache_hits, outcome.cache_misses) == (0, 2)
    assert outcome.embeddings_from_adapter is True


@pytest.mark.unit
def test_result_assembly_relies_on_canonical_outcome_invariants():
    with pytest.raises(ValueError) as exc_info:
        assemble_embedding_execution_outcome(
            _prepared_request(),
            vectors=[[0.1, 0.2], [0.3, 0.4]],
            provider="openai",
            model="text-embedding-3-small",
            cache_hits=0,
            cache_misses=1,
            fallback_from=None,
            embeddings_from_adapter=False,
            attempt_count=1,
            fallback_attempt_count=0,
        )

    assert str(exc_info.value) == ("cache_hits + cache_misses must equal the number of vectors")


@pytest.mark.unit
@pytest.mark.parametrize(
    ("provider", "fallback_from", "dimensions", "expected"),
    [
        ("openai", None, None, {"X-Embeddings-Provider": "openai"}),
        (
            "openai",
            None,
            2,
            {
                "X-Embeddings-Provider": "openai",
                "X-Embeddings-Dimensions-Policy": "reduce",
            },
        ),
        (
            "huggingface",
            "huggingface",
            None,
            {"X-Embeddings-Provider": "huggingface"},
        ),
        (
            "huggingface",
            "openai",
            2,
            {
                "X-Embeddings-Provider": "huggingface",
                "X-Embeddings-Fallback-From": "openai",
                "X-Embeddings-Dimensions-Policy": "reduce",
            },
        ),
    ],
)
def test_header_mapper_matches_legacy_contract(
    provider: str,
    fallback_from: str | None,
    dimensions: int | None,
    expected: dict[str, str],
):
    outcome = _outcome(
        provider=provider,
        fallback_from=fallback_from,
        requested_dimensions=dimensions,
    )

    assert map_embedding_response_headers(outcome) == expected


@pytest.mark.unit
def test_legacy_mapper_preserves_all_legacy_fields():
    outcome = _outcome(
        provider="huggingface",
        fallback_from="openai",
        requested_dimensions=2,
    )

    legacy = map_outcome_to_legacy_execution_result(outcome)

    assert legacy == EmbeddingExecutionResult(
        vectors=[[0.1, 0.2]],
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


@pytest.mark.unit
def test_legacy_mapper_returns_independent_vectors_and_headers():
    outcome = _outcome(
        provider="huggingface",
        fallback_from="openai",
        requested_dimensions=2,
    )
    first_result = map_outcome_to_legacy_execution_result(outcome)
    second_result = map_outcome_to_legacy_execution_result(outcome)

    first_result.vectors[0][0] = 9.0
    first_result.response_headers["X-Embeddings-Provider"] = "mutated"

    assert outcome.vectors == ((0.1, 0.2),)
    assert second_result.vectors == [[0.1, 0.2]]
    assert second_result.response_headers == {
        "X-Embeddings-Provider": "huggingface",
        "X-Embeddings-Fallback-From": "openai",
        "X-Embeddings-Dimensions-Policy": "reduce",
    }
