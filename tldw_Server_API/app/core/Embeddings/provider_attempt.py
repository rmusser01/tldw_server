"""Provider readiness and single-provider execution attempts for embeddings."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from tldw_Server_API.app.core.Embeddings.request_types import (
    EmbeddingExecutorOutput,
)

ProviderPreflight = Callable[[str, str], Awaitable[None]]


async def _no_provider_preflight(provider: str, model: str) -> None:
    del provider, model


class EmbeddingProviderReadinessCheck:
    """Run provider readiness checks without cache or execution side effects."""

    def __init__(self, provider_preflight: ProviderPreflight | None = None) -> None:
        self._provider_preflight = provider_preflight or _no_provider_preflight

    async def check(self, provider: str, model: str) -> None:
        await self._provider_preflight(provider, model)


__all__ = [
    "EmbeddingExecutorOutput",
    "EmbeddingProviderReadinessCheck",
    "ProviderPreflight",
]
