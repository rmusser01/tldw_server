from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pytest

from tldw_Server_API.app.core.Embeddings.request_types import (
    EmbeddingExecutionPlan,
    EmbeddingExecutorOutput,
    EmbeddingPolicyDecision,
    NormalizedEmbeddingInput,
    PreparedEmbeddingRequest,
    ProviderModelIntent,
)
from tldw_Server_API.app.core.Embeddings.provider_attempt import ProviderAttemptSuccess
from tldw_Server_API.app.core.Embeddings.execution_coordinator import (
    AdapterAttemptResult,
    EmbeddingAdapterAttempt,
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
