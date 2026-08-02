from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pytest

from tldw_Server_API.app.core.Embeddings.execution_coordinator import (
    AdapterAttemptResult,
    EmbeddingAdapterAttempt,
    EmbeddingFallbackCoordinator,
)
from tldw_Server_API.app.core.Embeddings.provider_attempt import (
    ProviderAttemptSuccess,
    ProviderCallFailure,
)
from tldw_Server_API.app.core.Embeddings.request_types import (
    EmbeddingDomainError,
    EmbeddingExecutionError,
    EmbeddingExecutionPlan,
    EmbeddingExecutorOutput,
    EmbeddingPolicyDecision,
    NormalizedEmbeddingInput,
    PreparedEmbeddingRequest,
    ProviderModelIntent,
)
from tldw_Server_API.app.core.exceptions import (
    EmbeddingProviderError,
    EmbeddingRateLimitError,
)


def _prepared(
    *,
    texts: list[str] | None = None,
    provider: str = "openai",
    model: str = "text-embedding-3-small",
    fallback_chain: list[str] | None = None,
    fallback_allowed: bool = True,
    execution_path: Literal["legacy", "adapter"] = "legacy",
) -> PreparedEmbeddingRequest:
    ordered_texts = texts or ["alpha"]
    chain = list(fallback_chain or [provider])
    return PreparedEmbeddingRequest(
        normalized_input=NormalizedEmbeddingInput(
            texts=ordered_texts,
            token_counts=[1 for _ in ordered_texts],
            total_tokens=len(ordered_texts),
        ),
        provider_intent=ProviderModelIntent(
            provider=provider,
            model=model,
            requested_provider=provider,
            requested_model=model,
            provider_was_explicit=True,
            model_was_provider_qualified=False,
        ),
        policy_decision=EmbeddingPolicyDecision(
            provider=provider,
            model=model,
            dimensions=None,
            fallback_chain=chain,
            fallback_allowed=fallback_allowed,
            enforce_policy=True,
        ),
        execution_plan=EmbeddingExecutionPlan(
            provider=provider,
            model=model,
            dimensions=None,
            backend_identity="stale-plan-identity",
            fallback_chain=chain,
            execution_path=execution_path,
        ),
        effective_dimension_policy="reduce",
        prompt_tokens=len(ordered_texts),
        total_tokens=len(ordered_texts),
    )


@dataclass(frozen=True, slots=True)
class AttemptCall:
    prepared: PreparedEmbeddingRequest
    provider: str
    model: str


class RecordingExecutor:
    def __init__(
        self,
        *,
        adapter_output: EmbeddingExecutorOutput | None = None,
        adapter_error: Exception | None = None,
    ) -> None:
        self.adapter_output = adapter_output
        self.adapter_error = adapter_error
        self.adapter_calls: list[tuple[list[str], str, str, int | None]] = []

    async def create_adapter(
        self,
        texts: list[str],
        *,
        provider: str,
        model: str,
        dimensions: int | None,
    ) -> EmbeddingExecutorOutput | None:
        self.adapter_calls.append((texts, provider, model, dimensions))
        if self.adapter_error is not None:
            raise self.adapter_error
        return self.adapter_output


class ExecutorWithoutAdapter:
    pass


def _retryable(provider: str) -> EmbeddingProviderError:
    return EmbeddingProviderError(
        "provider_unavailable",
        "provider unavailable",
        provider=provider,
        model={
            "openai": "text-embedding-3-small",
            "cohere": "embed-english-v3.0",
            "huggingface": "sentence-transformers/all-MiniLM-L6-v2",
        }[provider],
        retryable=True,
    )


def _success(provider: str) -> ProviderAttemptSuccess:
    return ProviderAttemptSuccess(
        vectors=[[0.25, 0.75]],
        provider=provider,
        model={
            "cohere": "embed-english-v3.0",
            "huggingface": "sentence-transformers/all-MiniLM-L6-v2",
        }.get(provider, "text-embedding-3-small"),
        cache_hits=0,
        cache_misses=1,
    )


def _model_map() -> dict[str, object]:
    return {
        "openai:text-embedding-3-small": {
            "cohere": "embed-english-v3.0",
            "huggingface": "sentence-transformers/all-MiniLM-L6-v2",
        }
    }


class RecordingReadiness:
    def __init__(
        self,
        errors: dict[str, EmbeddingDomainError] | None = None,
        events: list[str] | None = None,
    ) -> None:
        self.errors = errors or {}
        self.events = events
        self.calls: list[tuple[str, str]] = []

    async def check(self, provider: str, model: str) -> None:
        self.calls.append((provider, model))
        if self.events is not None:
            self.events.append(f"readiness:{provider}")
        error = self.errors.get(provider)
        if error is not None:
            raise error


class RecordingProviderAttempt:
    def __init__(
        self,
        outcomes: dict[str, ProviderAttemptSuccess | ProviderCallFailure],
        raised: dict[str, EmbeddingDomainError] | None = None,
        events: list[str] | None = None,
    ) -> None:
        self.outcomes = outcomes
        self.raised = raised or {}
        self.events = events
        self.calls: list[AttemptCall] = []

    async def execute(
        self,
        prepared: PreparedEmbeddingRequest,
        *,
        provider: str,
        model: str,
    ) -> ProviderAttemptSuccess | ProviderCallFailure:
        self.calls.append(AttemptCall(prepared, provider, model))
        if self.events is not None:
            self.events.append(f"attempt:{provider}")
        error = self.raised.get(provider)
        if error is not None:
            raise error
        return self.outcomes[provider]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_adapter_attempt_skips_non_adapter_execution_plan():
    executor = RecordingExecutor()
    attempt = EmbeddingAdapterAttempt(executor=executor)

    result = await attempt.execute(_prepared(execution_path="legacy"))

    assert result == AdapterAttemptResult(attempted=False)
    assert executor.adapter_calls == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_adapter_attempt_skips_executor_without_create_adapter():
    executor = ExecutorWithoutAdapter()
    attempt = EmbeddingAdapterAttempt(executor=executor)

    result = await attempt.execute(_prepared(execution_path="adapter"))

    assert result == AdapterAttemptResult(attempted=False)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_adapter_attempt_decline_is_counted_without_provider_result():
    executor = RecordingExecutor(adapter_output=None)
    attempt = EmbeddingAdapterAttempt(executor=executor)

    result = await attempt.execute(_prepared(execution_path="adapter"))

    assert result == AdapterAttemptResult(attempted=True)
    assert len(executor.adapter_calls) == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_adapter_attempt_non_adapter_output_is_a_decline():
    executor = RecordingExecutor(
        adapter_output=EmbeddingExecutorOutput(
            vectors=[[3.0, 4.0]],
            embeddings_from_adapter=False,
        )
    )
    attempt = EmbeddingAdapterAttempt(executor=executor)

    result = await attempt.execute(_prepared(execution_path="adapter"))

    assert result == AdapterAttemptResult(attempted=True)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_adapter_attempt_validates_and_processes_success():
    executor = RecordingExecutor(
        adapter_output=EmbeddingExecutorOutput(
            vectors=[[3.0, 4.0]],
            embeddings_from_adapter=True,
        )
    )
    attempt = EmbeddingAdapterAttempt(executor=executor)

    result = await attempt.execute(_prepared(execution_path="adapter"))

    assert result.success == ProviderAttemptSuccess(
        vectors=[[3.0, 4.0]],
        provider="openai",
        model="text-embedding-3-small",
        cache_hits=0,
        cache_misses=1,
        embeddings_from_adapter=True,
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_adapter_attempt_reraises_exact_adapter_exception():
    error = RuntimeError("adapter failed")
    executor = RecordingExecutor(adapter_error=error)
    attempt = EmbeddingAdapterAttempt(executor=executor)

    with pytest.raises(RuntimeError) as exc_info:
        await attempt.execute(_prepared(execution_path="adapter"))

    assert exc_info.value is error


@pytest.mark.unit
@pytest.mark.asyncio
async def test_fallback_excludes_primary_and_preserves_candidate_order():
    readiness = RecordingReadiness()
    attempt = RecordingProviderAttempt(
        outcomes={
            "cohere": ProviderCallFailure(_retryable("cohere")),
            "huggingface": _success("huggingface"),
        }
    )
    coordinator = EmbeddingFallbackCoordinator(
        readiness=readiness,
        provider_attempt=attempt,
        settings_fallback_model_map=_model_map(),
    )

    result = await coordinator.execute(
        _prepared(fallback_chain=["openai", "cohere", "openai", "huggingface"]),
        ProviderCallFailure(_retryable("openai")),
    )

    assert readiness.calls == [
        ("cohere", "embed-english-v3.0"),
        ("huggingface", "sentence-transformers/all-MiniLM-L6-v2"),
    ]
    assert [call.provider for call in attempt.calls] == ["cohere", "huggingface"]
    assert result.attempt_count == 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_fallback_skips_missing_credentials_from_readiness_and_provider_call():
    missing_readiness = EmbeddingProviderError(
        "missing_provider_credentials",
        "cohere credentials missing",
        provider="cohere",
        model="embed-english-v3.0",
    )
    missing_attempt = EmbeddingProviderError(
        "missing_provider_credentials",
        "huggingface credentials missing",
        provider="huggingface",
        model="sentence-transformers/all-MiniLM-L6-v2",
    )
    readiness = RecordingReadiness(errors={"cohere": missing_readiness})
    attempt = RecordingProviderAttempt(
        outcomes={
            "huggingface": ProviderCallFailure(missing_attempt),
            "mistral": _success("mistral"),
        }
    )
    coordinator = EmbeddingFallbackCoordinator(
        readiness=readiness,
        provider_attempt=attempt,
        settings_fallback_model_map=_model_map(),
    )

    result = await coordinator.execute(
        _prepared(fallback_chain=["cohere", "huggingface", "mistral"]),
        ProviderCallFailure(_retryable("openai")),
    )

    assert readiness.calls == [
        ("cohere", "embed-english-v3.0"),
        ("huggingface", "sentence-transformers/all-MiniLM-L6-v2"),
        ("mistral", "text-embedding-3-small"),
    ]
    assert [call.provider for call in attempt.calls] == ["huggingface", "mistral"]
    assert result.attempt_count == 3


@pytest.mark.unit
@pytest.mark.asyncio
async def test_fallback_continues_after_eligible_readiness_failure():
    readiness = RecordingReadiness(errors={"cohere": _retryable("cohere")})
    attempt = RecordingProviderAttempt(outcomes={"huggingface": _success("huggingface")})
    coordinator = EmbeddingFallbackCoordinator(
        readiness=readiness,
        provider_attempt=attempt,
        settings_fallback_model_map=_model_map(),
    )

    result = await coordinator.execute(
        _prepared(fallback_chain=["cohere", "huggingface"]),
        ProviderCallFailure(_retryable("openai")),
    )

    assert result.success == _success("huggingface")
    assert result.attempt_count == 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_fallback_continues_after_eligible_provider_call_failure():
    readiness = RecordingReadiness()
    attempt = RecordingProviderAttempt(
        outcomes={
            "cohere": ProviderCallFailure(_retryable("cohere")),
            "huggingface": _success("huggingface"),
        }
    )
    coordinator = EmbeddingFallbackCoordinator(
        readiness=readiness,
        provider_attempt=attempt,
        settings_fallback_model_map=_model_map(),
    )

    result = await coordinator.execute(
        _prepared(fallback_chain=["cohere", "huggingface"]),
        ProviderCallFailure(_retryable("openai")),
    )

    assert result.success == _success("huggingface")
    assert result.attempt_count == 2


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["readiness", "provider_call"])
async def test_fallback_reraises_ineligible_failure_without_trying_another_candidate(
    source: str,
):
    error = EmbeddingProviderError(
        "provider_denied",
        "provider denied request",
        provider="cohere",
        model="embed-english-v3.0",
        retryable=False,
    )
    readiness = RecordingReadiness(errors={"cohere": error} if source == "readiness" else None)
    attempt = RecordingProviderAttempt(
        outcomes={
            "cohere": ProviderCallFailure(error),
            "huggingface": _success("huggingface"),
        }
    )
    coordinator = EmbeddingFallbackCoordinator(
        readiness=readiness,
        provider_attempt=attempt,
        settings_fallback_model_map=_model_map(),
    )

    with pytest.raises(EmbeddingProviderError) as exc_info:
        await coordinator.execute(
            _prepared(fallback_chain=["cohere", "huggingface"]),
            ProviderCallFailure(_retryable("openai")),
        )

    assert exc_info.value is error
    assert [call.provider for call in attempt.calls] == ([] if source == "readiness" else ["cohere"])


@pytest.mark.unit
@pytest.mark.asyncio
async def test_fallback_selects_original_rate_limit_error_on_exhaustion():
    rate_limit = EmbeddingRateLimitError(
        "provider_rate_limited",
        "huggingface rate limited",
        provider="huggingface",
        model="sentence-transformers/all-MiniLM-L6-v2",
        retryable=True,
        retry_after=37,
    )
    readiness = RecordingReadiness()
    attempt = RecordingProviderAttempt(
        outcomes={
            "cohere": ProviderCallFailure(_retryable("cohere")),
            "huggingface": ProviderCallFailure(rate_limit),
        }
    )
    coordinator = EmbeddingFallbackCoordinator(
        readiness=readiness,
        provider_attempt=attempt,
        settings_fallback_model_map=_model_map(),
    )

    with pytest.raises(EmbeddingRateLimitError) as exc_info:
        await coordinator.execute(
            _prepared(fallback_chain=["cohere", "huggingface"]),
            ProviderCallFailure(_retryable("openai")),
        )

    assert exc_info.value is rate_limit
    assert exc_info.value.retry_after == 37


@pytest.mark.unit
@pytest.mark.asyncio
async def test_fallback_uses_exact_primary_failure_when_candidates_are_skipped():
    primary_error = _retryable("openai")
    missing_credentials = EmbeddingProviderError(
        "missing_provider_credentials",
        "cohere credentials missing",
        provider="cohere",
        model="embed-english-v3.0",
    )
    readiness = RecordingReadiness(errors={"cohere": missing_credentials})
    attempt = RecordingProviderAttempt(outcomes={})
    coordinator = EmbeddingFallbackCoordinator(
        readiness=readiness,
        provider_attempt=attempt,
        settings_fallback_model_map=_model_map(),
    )

    with pytest.raises(EmbeddingProviderError) as exc_info:
        await coordinator.execute(
            _prepared(fallback_chain=["cohere"]),
            ProviderCallFailure(primary_error),
        )

    assert exc_info.value is primary_error


@pytest.mark.unit
@pytest.mark.asyncio
async def test_fallback_passes_the_complete_prepared_request_to_successful_attempt():
    prepared = _prepared(texts=["first", "second"], fallback_chain=["huggingface"])
    readiness = RecordingReadiness()
    attempt = RecordingProviderAttempt(outcomes={"huggingface": _success("huggingface")})
    coordinator = EmbeddingFallbackCoordinator(
        readiness=readiness,
        provider_attempt=attempt,
        settings_fallback_model_map=_model_map(),
    )

    await coordinator.execute(prepared, ProviderCallFailure(_retryable("openai")))

    assert attempt.calls == [
        AttemptCall(
            prepared,
            "huggingface",
            "sentence-transformers/all-MiniLM-L6-v2",
        )
    ]


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "boundary",
    ["backend_identity", "cache_key", "cache", "validation", "postprocessing", "writeback"],
)
async def test_fallback_propagates_attempt_boundary_errors_without_advancing(
    boundary: str,
):
    error = EmbeddingExecutionError(
        "internal_execution_failure",
        f"{boundary} failed",
        retryable=True,
    )
    readiness = RecordingReadiness()
    attempt = RecordingProviderAttempt(
        outcomes={"huggingface": _success("huggingface")},
        raised={"cohere": error},
    )
    coordinator = EmbeddingFallbackCoordinator(
        readiness=readiness,
        provider_attempt=attempt,
        settings_fallback_model_map=_model_map(),
    )

    with pytest.raises(EmbeddingExecutionError) as exc_info:
        await coordinator.execute(
            _prepared(fallback_chain=["cohere", "huggingface"]),
            ProviderCallFailure(_retryable("openai")),
        )

    assert exc_info.value is error
    assert [call.provider for call in attempt.calls] == ["cohere"]
