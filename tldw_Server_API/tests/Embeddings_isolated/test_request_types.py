from __future__ import annotations

from dataclasses import asdict
from typing import Literal, get_type_hints

import pytest

from tldw_Server_API.app.core.Embeddings.request_types import (
    EmbeddingDomainError,
    EmbeddingExecutionPlan,
    EmbeddingExecutionResult,
    EmbeddingPolicyDecision,
    EmbeddingRequestContext,
    NormalizedEmbeddingInput,
)

TelemetryScalar = str | int | float | bool | None


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
