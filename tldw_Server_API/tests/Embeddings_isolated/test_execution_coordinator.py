from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pytest

from tldw_Server_API.app.core.Embeddings.execution_coordinator import (
    AdapterAttemptResult,
    EmbeddingAdapterAttempt,
    EmbeddingExecutionCoordinator,
    EmbeddingFallbackCoordinator,
    FallbackExecutionSuccess,
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


def _success(
    provider: str,
    *,
    vectors: list[list[float]] | None = None,
    cache_hits: int = 0,
    cache_misses: int = 1,
    embeddings_from_adapter: bool = False,
) -> ProviderAttemptSuccess:
    return ProviderAttemptSuccess(
        vectors=vectors or [[0.25, 0.75]],
        provider=provider,
        model={
            "cohere": "embed-english-v3.0",
            "huggingface": "sentence-transformers/all-MiniLM-L6-v2",
        }.get(provider, "text-embedding-3-small"),
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        embeddings_from_adapter=embeddings_from_adapter,
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


class RecordingAdapterAttempt:
    def __init__(self, result: AdapterAttemptResult, events: list[str]) -> None:
        self.result = result
        self.events = events

    async def execute(self, prepared: PreparedEmbeddingRequest) -> AdapterAttemptResult:
        del prepared
        if self.result.attempted:
            self.events.append("adapter")
        return self.result


class RecordingFallbackCoordinator:
    def __init__(
        self,
        result: FallbackExecutionSuccess | None = None,
    ) -> None:
        self.result = result or FallbackExecutionSuccess(_success("huggingface"), 1)
        self.calls: list[tuple[PreparedEmbeddingRequest, ProviderCallFailure]] = []

    async def execute(
        self,
        prepared: PreparedEmbeddingRequest,
        primary_failure: ProviderCallFailure,
    ) -> FallbackExecutionSuccess:
        self.calls.append((prepared, primary_failure))
        return self.result


def _coordinator(
    *,
    events: list[str] | None = None,
    adapter_declines: bool = False,
    adapter_result: AdapterAttemptResult | None = None,
    readiness_error: EmbeddingDomainError | None = None,
    primary_result: ProviderAttemptSuccess | ProviderCallFailure | None = None,
    fallback_result: FallbackExecutionSuccess | None = None,
    fallback: RecordingFallbackCoordinator | None = None,
    provider_attempt: RecordingProviderAttempt | None = None,
) -> EmbeddingExecutionCoordinator:
    ordered_events = events if events is not None else []
    adapter = RecordingAdapterAttempt(
        adapter_result or AdapterAttemptResult(attempted=adapter_declines),
        ordered_events,
    )
    readiness = RecordingReadiness(
        errors={"openai": readiness_error} if readiness_error is not None else None,
        events=ordered_events,
    )
    attempt = provider_attempt or RecordingProviderAttempt(
        outcomes={"openai": primary_result or _success("openai")}, events=ordered_events
    )
    fallback_coordinator = fallback or RecordingFallbackCoordinator(fallback_result)
    return EmbeddingExecutionCoordinator(
        adapter_attempt=adapter,
        readiness=readiness,
        provider_attempt=attempt,
        fallback_coordinator=fallback_coordinator,
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_execution_orders_adapter_before_primary_readiness_and_attempt():
    events: list[str] = []
    coordinator = _coordinator(events=events, adapter_declines=True)

    outcome = await coordinator.execute(_prepared(execution_path="adapter"))

    assert events == ["adapter", "readiness:openai", "attempt:openai"]
    assert outcome.attempt_count == 2
    assert outcome.fallback_attempt_count == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_execution_returns_adapter_success_with_adapter_cache_accounting():
    events: list[str] = []
    coordinator = _coordinator(
        events=events,
        adapter_result=AdapterAttemptResult(
            attempted=True,
            success=_success(
                "openai",
                vectors=[[0.1, 0.9], [0.3, 0.7]],
                cache_hits=0,
                cache_misses=2,
                embeddings_from_adapter=True,
            ),
        ),
    )

    outcome = await coordinator.execute(
        _prepared(texts=["first", "second"], execution_path="adapter")
    )

    assert events == ["adapter"]
    assert outcome.vectors == ((0.1, 0.9), (0.3, 0.7))
    assert outcome.provider == "openai"
    assert outcome.model == "text-embedding-3-small"
    assert outcome.cache_hits == 0
    assert outcome.cache_misses == 2
    assert outcome.embeddings_from_adapter is True
    assert outcome.attempt_count == 1
    assert outcome.fallback_attempt_count == 0
    assert outcome.fallback_from is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_execution_primary_readiness_failure_does_not_enter_fallback():
    error = _retryable("openai")
    fallback = RecordingFallbackCoordinator()
    coordinator = _coordinator(readiness_error=error, fallback=fallback)

    with pytest.raises(EmbeddingDomainError) as raised:
        await coordinator.execute(_prepared())

    assert raised.value is error
    assert fallback.calls == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_execution_reraises_primary_provider_attempt_exception_without_fallback():
    error = EmbeddingExecutionError(
        "internal_execution_failure",
        "primary provider attempt failed",
        retryable=True,
    )
    prepared = _prepared()
    attempt = RecordingProviderAttempt(
        outcomes={},
        raised={"openai": error},
    )
    fallback = RecordingFallbackCoordinator()
    coordinator = _coordinator(provider_attempt=attempt, fallback=fallback)

    with pytest.raises(EmbeddingExecutionError) as raised:
        await coordinator.execute(prepared)

    assert raised.value is error
    assert attempt.calls == [
        AttemptCall(prepared, "openai", "text-embedding-3-small")
    ]
    assert fallback.calls == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_execution_returns_full_primary_cache_hit_in_input_order():
    primary = _success(
        "openai",
        vectors=[[0.7, 0.3], [0.2, 0.8]],
        cache_hits=2,
        cache_misses=0,
    )
    coordinator = _coordinator(primary_result=primary)

    outcome = await coordinator.execute(_prepared(texts=["first", "second"]))

    assert outcome.vectors == ((0.7, 0.3), (0.2, 0.8))
    assert outcome.provider == "openai"
    assert outcome.model == "text-embedding-3-small"
    assert outcome.cache_hits == 2
    assert outcome.cache_misses == 0
    assert outcome.embeddings_from_adapter is False
    assert outcome.attempt_count == 1
    assert outcome.fallback_attempt_count == 0
    assert outcome.fallback_from is None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_execution_reraises_non_retryable_primary_failure_without_fallback():
    error = EmbeddingProviderError(
        "provider_denied",
        "provider denied request",
        provider="openai",
        model="text-embedding-3-small",
        retryable=False,
    )
    fallback = RecordingFallbackCoordinator()
    coordinator = _coordinator(
        primary_result=ProviderCallFailure(error),
        fallback=fallback,
    )

    with pytest.raises(EmbeddingProviderError) as raised:
        await coordinator.execute(_prepared())

    assert raised.value is error
    assert fallback.calls == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_execution_reraises_fallback_disabled_primary_failure():
    error = _retryable("openai")
    fallback = RecordingFallbackCoordinator()
    coordinator = _coordinator(
        primary_result=ProviderCallFailure(error),
        fallback=fallback,
    )

    with pytest.raises(EmbeddingProviderError) as raised:
        await coordinator.execute(_prepared(fallback_allowed=False))

    assert raised.value is error
    assert fallback.calls == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_execution_uses_complete_request_for_eligible_primary_failure_fallback():
    prepared = _prepared(texts=["first", "second"])
    fallback = RecordingFallbackCoordinator(
        FallbackExecutionSuccess(
            success=_success(
                "huggingface",
                vectors=[[0.8, 0.2], [0.6, 0.4]],
                cache_hits=1,
                cache_misses=1,
            ),
            attempt_count=2,
        )
    )
    coordinator = _coordinator(
        primary_result=ProviderCallFailure(_retryable("openai")),
        fallback=fallback,
    )

    outcome = await coordinator.execute(prepared)

    assert fallback.calls[0][0] is prepared
    assert outcome.vectors == ((0.8, 0.2), (0.6, 0.4))
    assert outcome.provider == "huggingface"
    assert outcome.model == "sentence-transformers/all-MiniLM-L6-v2"
    assert outcome.cache_hits == 1
    assert outcome.cache_misses == 1
    assert outcome.embeddings_from_adapter is False
    assert outcome.fallback_from == "openai"
    assert outcome.attempt_count == 3
    assert outcome.fallback_attempt_count == 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_execution_returns_successful_primary_execution():
    coordinator = _coordinator(
        primary_result=_success(
            "openai",
            vectors=[[0.4, 0.6]],
            cache_hits=0,
            cache_misses=1,
        )
    )

    outcome = await coordinator.execute(_prepared())

    assert outcome.vectors == ((0.4, 0.6),)
    assert outcome.provider == "openai"
    assert outcome.model == "text-embedding-3-small"
    assert outcome.cache_hits == 0
    assert outcome.cache_misses == 1
    assert outcome.embeddings_from_adapter is False
    assert outcome.attempt_count == 1
    assert outcome.fallback_attempt_count == 0
    assert outcome.fallback_from is None


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
async def test_fallback_propagates_attempt_exception_without_advancing():
    error = EmbeddingExecutionError(
        "internal_execution_failure",
        "attempt boundary failed",
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
