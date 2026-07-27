from __future__ import annotations

import pytest

from tldw_Server_API.app.core.Embeddings.provider_attempt import (
    EmbeddingProviderReadinessCheck,
)
from tldw_Server_API.app.core.Embeddings.request_types import (
    EmbeddingExecutionError,
    EmbeddingExecutorOutput,
)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_provider_readiness_check_delegates_to_preflight():
    calls: list[tuple[str, str]] = []

    async def preflight(provider: str, model: str) -> None:
        calls.append((provider, model))

    readiness = EmbeddingProviderReadinessCheck(preflight)

    await readiness.check("openai", "text-embedding-3-small")

    assert calls == [("openai", "text-embedding-3-small")]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_provider_readiness_check_propagates_exact_error():
    error = EmbeddingExecutionError(
        "circuit_breaker_open",
        "provider unavailable",
        provider="openai",
        model="text-embedding-3-small",
        retryable=True,
    )

    async def preflight(provider: str, model: str) -> None:
        del provider, model
        raise error

    readiness = EmbeddingProviderReadinessCheck(preflight)

    with pytest.raises(EmbeddingExecutionError) as exc_info:
        await readiness.check("openai", "text-embedding-3-small")

    assert exc_info.value is error


@pytest.mark.unit
def test_executor_output_contract_is_shared_from_request_types():
    output = EmbeddingExecutorOutput(
        vectors=[[0.1, 0.2]],
        embeddings_from_adapter=True,
    )

    assert output.vectors == [[0.1, 0.2]]
    assert output.embeddings_from_adapter is True
