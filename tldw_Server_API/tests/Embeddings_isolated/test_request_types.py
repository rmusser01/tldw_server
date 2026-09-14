from __future__ import annotations

from dataclasses import MISSING, asdict, fields
from typing import Literal, get_type_hints

import pytest

from tldw_Server_API.app.core.Embeddings.orchestrator import (
    PreparedEmbeddingRequest as OrchestratorPreparedEmbeddingRequest,
)
from tldw_Server_API.app.core.Embeddings.request_types import (
    EmbeddingDomainError,
    EmbeddingExecutionOutcome,
    EmbeddingExecutionPlan,
    EmbeddingExecutionResult,
    EmbeddingPolicyDecision,
    EmbeddingRequestContext,
    NormalizedEmbeddingInput,
    PreparedEmbeddingRequest,
)

TelemetryScalar = str | int | float | bool | None


def _valid_execution_outcome_values() -> dict[str, object]:
    return {
        "vectors": ((0.1, 0.2),),
        "provider": "openai",
        "model": "text-embedding-3-small",
        "prompt_tokens": 1,
        "total_tokens": 1,
        "cache_hits": 0,
        "cache_misses": 1,
        "requested_dimensions": None,
        "effective_dimension_policy": "reduce",
        "attempt_count": 1,
        "fallback_attempt_count": 0,
        "embeddings_from_adapter": False,
    }


@pytest.mark.unit
def test_request_context_excludes_raw_input_and_secret_fields():
    context = EmbeddingRequestContext(
        user_id="user-1",
        model_field="openai/text-embedding-3-small",
        provider_header="openai",
        dimensions=1536,
        encoding_format="float",
        request_id="req-1",
        testing=True,
        adapters_enabled=True,
    )

    for forbidden_attribute in ("raw_input", "texts", "input", "api_key", "authorization"):
        assert not hasattr(context, forbidden_attribute)


@pytest.mark.unit
def test_execution_plan_serializes_only_sanitized_planning_metadata():
    plan = EmbeddingExecutionPlan(
        provider="openai",
        model="text-embedding-3-small",
        dimensions=1536,
        backend_identity=None,
        fallback_chain=["local"],
        cache_namespace="embeddings:v1",
        batch_size=32,
        execution_path="adapter",
        observability_tags={
            "request_mode": "sync",
            "fallback_count": 1,
            "cache_enabled": True,
            "sample_rate": 0.25,
            "tenant": None,
        },
    )

    assert asdict(plan) == {
        "provider": "openai",
        "model": "text-embedding-3-small",
        "dimensions": 1536,
        "backend_identity": None,
        "fallback_chain": ["local"],
        "cache_namespace": "embeddings:v1",
        "batch_size": 32,
        "execution_path": "adapter",
        "observability_tags": {
            "request_mode": "sync",
            "fallback_count": 1,
            "cache_enabled": True,
            "sample_rate": 0.25,
            "tenant": None,
        },
    }


@pytest.mark.unit
def test_request_contract_annotations_match_approved_plan_shapes():
    normalized_hints = get_type_hints(NormalizedEmbeddingInput)
    policy_hints = get_type_hints(EmbeddingPolicyDecision)
    plan_hints = get_type_hints(EmbeddingExecutionPlan)
    result_hints = get_type_hints(EmbeddingExecutionResult)

    assert normalized_hints["texts"] == list[str]
    assert normalized_hints["token_counts"] == list[int]
    assert normalized_hints["token_input_mode"] == Literal["none", "single", "batch"]
    assert policy_hints["fallback_chain"] == list[str]
    assert plan_hints["backend_identity"] == str | None
    assert plan_hints["fallback_chain"] == list[str]
    assert plan_hints["execution_path"] == Literal["legacy", "adapter"]
    assert plan_hints["observability_tags"] == dict[str, TelemetryScalar]
    assert result_hints["response_headers"] == dict[str, str]

    error_details_hint = get_type_hints(EmbeddingDomainError.__init__)["details"]
    assert error_details_hint == list[dict[str, TelemetryScalar]] | None


@pytest.mark.unit
def test_execution_outcome_is_deeply_immutable_and_has_no_http_headers():
    hints = get_type_hints(EmbeddingExecutionOutcome)
    outcome_fields = {field.name: field for field in fields(EmbeddingExecutionOutcome)}

    assert "response_headers" not in hints
    assert hints["vectors"] == tuple[tuple[float, ...], ...]
    assert hints["attempt_count"] is int
    assert hints["fallback_attempt_count"] is int
    assert outcome_fields["attempt_count"].default is MISSING
    assert outcome_fields["fallback_attempt_count"].default is MISSING
    assert EmbeddingExecutionOutcome.__dataclass_params__.frozen is True
    assert set(EmbeddingExecutionOutcome.__slots__) == set(hints)


@pytest.mark.unit
def test_execution_outcome_copies_mutable_vectors_into_nested_tuples():
    source_vectors = [[0.1, 0.2], [0.3, 0.4]]
    outcome = EmbeddingExecutionOutcome(
        vectors=source_vectors,
        provider="openai",
        model="text-embedding-3-small",
        prompt_tokens=2,
        total_tokens=2,
        cache_hits=0,
        cache_misses=2,
        requested_dimensions=None,
        effective_dimension_policy="reduce",
        attempt_count=1,
        fallback_attempt_count=0,
    )

    source_vectors[0][0] = 9.0
    source_vectors.append([0.5, 0.6])

    assert outcome.vectors == ((0.1, 0.2), (0.3, 0.4))
    assert type(outcome.vectors) is tuple
    assert all(type(vector) is tuple for vector in outcome.vectors)


@pytest.mark.unit
def test_execution_outcome_canonicalizes_numeric_vector_leaves_to_floats():
    values = _valid_execution_outcome_values()
    values.update(
        {
            "vectors": [[1, 2], [3.5, 4]],
            "cache_misses": 2,
        }
    )

    outcome = EmbeddingExecutionOutcome(**values)

    assert outcome.vectors == ((1.0, 2.0), (3.5, 4.0))
    assert all(type(value) is float for vector in outcome.vectors for value in vector)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("vectors", "cache_misses"),
    [
        ([[[1.0], 2.0]], 1),
        ([["not-a-number"]], 1),
        ([[True]], 1),
        ([[float("nan")]], 1),
        ([[float("inf")]], 1),
        ([[1.0], [2.0, 3.0]], 2),
        ([[]], 1),
        ([1.0], 1),
    ],
    ids=(
        "nested-mutable-leaf",
        "non-numeric-leaf",
        "boolean-leaf",
        "nan-leaf",
        "infinite-leaf",
        "non-rectangular",
        "empty-vector",
        "non-vector-item",
    ),
)
def test_execution_outcome_rejects_malformed_vectors(
    vectors: object,
    cache_misses: int,
):
    values = _valid_execution_outcome_values()
    values.update({"vectors": vectors, "cache_misses": cache_misses})

    with pytest.raises(ValueError) as exc_info:
        EmbeddingExecutionOutcome(**values)

    assert str(exc_info.value) == ("vectors must contain equally sized, non-empty vectors of finite numbers")


@pytest.mark.unit
def test_execution_outcome_accepts_zero_total_with_positive_prompt_tokens():
    outcome = EmbeddingExecutionOutcome(
        vectors=((0.1, 0.2),),
        provider="openai",
        model="text-embedding-3-small",
        prompt_tokens=2,
        total_tokens=0,
        cache_hits=0,
        cache_misses=1,
        requested_dimensions=None,
        effective_dimension_policy="reduce",
        attempt_count=1,
        fallback_attempt_count=0,
    )

    assert outcome.prompt_tokens == 2
    assert outcome.total_tokens == 0


@pytest.mark.unit
@pytest.mark.parametrize(
    ("overrides", "expected_message"),
    [
        ({"prompt_tokens": -1}, "prompt_tokens must be nonnegative"),
        ({"cache_hits": -1}, "cache_hits must be nonnegative"),
        ({"cache_misses": -1}, "cache_misses must be nonnegative"),
        ({"attempt_count": -1}, "attempt_count must be nonnegative"),
        (
            {"fallback_attempt_count": -1},
            "fallback_attempt_count must be nonnegative",
        ),
        ({"total_tokens": -1}, "total_tokens must be nonnegative"),
        (
            {"attempt_count": 0},
            "attempt_count must be at least 1 for a successful outcome",
        ),
        (
            {"fallback_attempt_count": 2},
            "fallback_attempt_count must be less than attempt_count",
        ),
        (
            {"fallback_attempt_count": 1},
            "fallback_attempt_count must be less than attempt_count",
        ),
        (
            {"cache_hits": 1, "cache_misses": 1},
            "cache_hits + cache_misses must equal the number of vectors",
        ),
    ],
)
def test_execution_outcome_rejects_invalid_success_invariants(
    overrides: dict[str, int],
    expected_message: str,
):
    values = _valid_execution_outcome_values()
    values.update(overrides)

    with pytest.raises(ValueError) as exc_info:
        EmbeddingExecutionOutcome(**values)

    assert str(exc_info.value) == expected_message


@pytest.mark.unit
@pytest.mark.parametrize(
    ("overrides", "expected_message"),
    [
        (
            {"fallback_from": "openai"},
            "fallback_from requires a positive fallback_attempt_count",
        ),
        (
            {"attempt_count": 2, "fallback_attempt_count": 1},
            "positive fallback_attempt_count requires fallback_from",
        ),
        (
            {"attempt_count": 99},
            "attempt_count - fallback_attempt_count must be 1 or 2",
        ),
        (
            {
                "attempt_count": 4,
                "fallback_attempt_count": 1,
                "fallback_from": "openai",
            },
            "attempt_count - fallback_attempt_count must be 1 or 2",
        ),
    ],
)
def test_execution_outcome_rejects_inconsistent_attempt_metadata(
    overrides: dict[str, object],
    expected_message: str,
):
    values = _valid_execution_outcome_values()
    values.update(overrides)

    with pytest.raises(ValueError) as exc_info:
        EmbeddingExecutionOutcome(**values)

    assert str(exc_info.value) == expected_message


@pytest.mark.unit
@pytest.mark.parametrize(
    ("attempt_count", "fallback_attempt_count", "fallback_from"),
    [
        (1, 0, None),
        (2, 0, None),
        (2, 1, "openai"),
        (3, 1, "openai"),
        (3, 2, "openai"),
        (4, 2, "openai"),
    ],
)
def test_execution_outcome_accepts_attempt_count_boundaries(
    attempt_count: int,
    fallback_attempt_count: int,
    fallback_from: str | None,
):
    values = _valid_execution_outcome_values()
    values.update(
        {
            "attempt_count": attempt_count,
            "fallback_attempt_count": fallback_attempt_count,
            "fallback_from": fallback_from,
        }
    )

    outcome = EmbeddingExecutionOutcome(**values)

    assert outcome.attempt_count - outcome.fallback_attempt_count in (1, 2)
    assert outcome.fallback_from == fallback_from


@pytest.mark.unit
@pytest.mark.parametrize(
    "field_name",
    (
        "prompt_tokens",
        "total_tokens",
        "cache_hits",
        "cache_misses",
        "attempt_count",
        "fallback_attempt_count",
    ),
)
@pytest.mark.parametrize("invalid_value", (True, 1.0), ids=("bool", "float"))
def test_execution_outcome_requires_exact_builtin_integer_counters(
    field_name: str,
    invalid_value: bool | float,
):
    values = _valid_execution_outcome_values()
    values[field_name] = invalid_value

    with pytest.raises(ValueError) as exc_info:
        EmbeddingExecutionOutcome(**values)

    assert str(exc_info.value) == f"{field_name} must be an exact int"


@pytest.mark.unit
def test_execution_outcome_accepts_mixed_cache_counts_with_adapter_origin():
    values = _valid_execution_outcome_values()
    values.update(
        {
            "vectors": ((0.1, 0.2), (0.3, 0.4)),
            "cache_hits": 1,
            "cache_misses": 1,
            "embeddings_from_adapter": True,
        }
    )

    outcome = EmbeddingExecutionOutcome(**values)

    assert outcome.embeddings_from_adapter is True
    assert (outcome.cache_hits, outcome.cache_misses) == (1, 1)
    assert len(outcome.vectors) == 2


@pytest.mark.unit
def test_orchestrator_reexports_prepared_request_contract():
    assert OrchestratorPreparedEmbeddingRequest is PreparedEmbeddingRequest


@pytest.mark.unit
def test_execution_plan_observability_tags_are_sanitized_to_safe_scalars():
    plan = EmbeddingExecutionPlan(
        provider="openai",
        model="text-embedding-3-small",
        dimensions=1536,
        backend_identity="openai:text-embedding-3-small:1536",
        fallback_chain=[],
        observability_tags={
            "request_mode": "sync",
            "fallback_count": 1,
            "cache_enabled": True,
            "sample_rate": 0.25,
            "tenant": None,
            "authorization": "Bearer provider-secret",
            "raw_body": '{"api_key":"sk-secret","text":"request text"}',
            "headers": {"x-api-key": "sk-secret"},
            "nested": {"secret": "provider-secret"},
            "labels": ["secret", "provider-secret"],
            "safe_but_secret_value": "contains sk-secret",
        },
    )

    serialized = asdict(plan)

    assert serialized["observability_tags"] == {
        "request_mode": "sync",
        "fallback_count": 1,
        "cache_enabled": True,
        "sample_rate": 0.25,
        "tenant": None,
        "authorization": "[redacted]",
        "raw_body": "[redacted]",
        "headers": "[redacted]",
        "safe_but_secret_value": "[redacted]",
    }
    assert "provider-secret" not in repr(serialized)
    assert "sk-secret" not in repr(serialized)
    assert "request text" not in repr(serialized)


@pytest.mark.unit
def test_domain_error_http_payload_uses_only_sanitized_fields():
    error = EmbeddingDomainError(
        "provider_unavailable",
        "Provider unavailable",
        retryable=True,
        provider="openai",
        model="text-embedding-3-small",
        retry_after=30,
        cause_class="TimeoutError",
    )
    error.__cause__ = RuntimeError("raw provider body: secret-key")
    error.raw_provider_body = "raw provider body: secret-key"
    error.extra_debug = {"authorization": "Bearer secret"}

    assert error.to_http_payload() == {
        "error_code": "provider_unavailable",
        "message": "Provider unavailable",
        "provider": "openai",
        "model": "text-embedding-3-small",
        "retryable": True,
        "retry_after": 30,
        "details": [],
        "cause_class": "TimeoutError",
    }
    assert "__cause__" not in error.to_http_payload()
    assert "raw_provider_body" not in error.to_http_payload()
    assert "extra_debug" not in error.to_http_payload()
    assert "raw provider body: secret-key" not in repr(error.to_http_payload())
    assert "Bearer secret" not in repr(error.to_http_payload())


@pytest.mark.unit
def test_domain_error_details_are_public_safe_and_redact_secret_bearing_values():
    error = EmbeddingDomainError(
        "provider_malformed_response",
        "Provider response could not be parsed",
        details=[
            {
                "summary": "malformed json",
                "status": 502,
                "retryable": False,
                "latency_seconds": 1.25,
                "provider": None,
                "raw_body": '{"api_key":"sk-secret","text":"request text"}',
                "authorization": "Bearer provider-secret",
                "headers": {"x-api-key": "sk-secret"},
                "nested": {"body": "secret-bearing nested data"},
            },
            "raw provider body with sk-secret",
        ],
    )

    payload = error.to_http_payload()

    assert payload["details"] == [
        {
            "summary": "malformed json",
            "status": 502,
            "retryable": False,
            "latency_seconds": 1.25,
            "provider": None,
            "raw_body": "[redacted]",
            "authorization": "[redacted]",
            "headers": "[redacted]",
        }
    ]
    assert "sk-secret" not in repr(payload)
    assert "provider-secret" not in repr(payload)
    assert "request text" not in repr(payload)
    assert "secret-bearing nested data" not in repr(payload)
    assert "raw provider body" not in repr(payload)


@pytest.mark.unit
def test_domain_error_payload_resanitizes_details_mutated_after_construction():
    error = EmbeddingDomainError(
        "provider_unavailable",
        "Provider unavailable",
        details=[{"summary": "initial"}],
    )
    error.details.append(
        {
            "authorization": "Bearer provider-secret",
            "raw_body": '{"api_key":"sk-secret","text":"request text"}',
            "headers": {"x-api-key": "sk-secret"},
            "nested": {"body": "secret-bearing nested data"},
        }
    )
    error.details[0]["summary"] = "contains sk-secret"

    payload = error.to_http_payload()

    assert error.details[0]["summary"] == "contains sk-secret"
    assert payload["details"] == [
        {
            "summary": "[redacted]",
        },
        {
            "authorization": "[redacted]",
            "raw_body": "[redacted]",
            "headers": "[redacted]",
        },
    ]
    assert "provider-secret" not in repr(payload)
    assert "sk-secret" not in repr(payload)
    assert "request text" not in repr(payload)
    assert "secret-bearing nested data" not in repr(payload)

    payload["details"][0]["summary"] = "mutated payload"
    assert error.details[0]["summary"] == "contains sk-secret"


@pytest.mark.unit
def test_domain_error_preserves_safe_numeric_token_count_details():
    error = EmbeddingDomainError(
        "input_too_long",
        "too long",
        details=[
            {"index": 2, "tokens": 1234},
            {"token": "sk-secret"},
            {"access_token": "Bearer secret"},
            {"prompt_tokens": 100, "total_tokens": 200, "token_count": 300},
        ],
    )

    payload = error.to_http_payload()

    assert payload["details"] == [
        {"index": 2, "tokens": 1234},
        {"token": "[redacted]"},
        {"access_token": "[redacted]"},
        {"prompt_tokens": 100, "total_tokens": 200, "token_count": 300},
    ]
    assert "sk-secret" not in repr(payload)
    assert "Bearer secret" not in repr(payload)


@pytest.mark.unit
def test_default_mutable_fields_are_not_shared_between_instances():
    first_plan = EmbeddingExecutionPlan(
        provider="openai",
        model="text-embedding-3-small",
        dimensions=1536,
        backend_identity="openai:text-embedding-3-small:1536",
        fallback_chain=[],
    )
    second_plan = EmbeddingExecutionPlan(
        provider="local",
        model="all-MiniLM-L6-v2",
        dimensions=None,
        backend_identity="local:all-MiniLM-L6-v2",
        fallback_chain=[],
    )
    first_plan.observability_tags["provider_family"] = "commercial"

    assert second_plan.observability_tags == {}

    first_result = EmbeddingExecutionResult(
        vectors=[[0.1, 0.2]],
        provider="openai",
        model="text-embedding-3-small",
        prompt_tokens=2,
        total_tokens=2,
        cache_hits=0,
        cache_misses=1,
    )
    second_result = EmbeddingExecutionResult(
        vectors=[[0.3, 0.4]],
        provider="local",
        model="all-MiniLM-L6-v2",
        prompt_tokens=3,
        total_tokens=3,
        cache_hits=1,
        cache_misses=0,
    )
    first_result.response_headers["retry-after"] = "30"

    assert second_result.response_headers == {}
