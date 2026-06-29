"""Internal request contracts for the embeddings orchestration path."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from tldw_Server_API.app.core.exceptions import (
    EmbeddingDomainError,
    EmbeddingErrorCode,
    EmbeddingExecutionError,
    EmbeddingInputError,
    EmbeddingPolicyError,
    EmbeddingProviderError,
    EmbeddingRateLimitError,
    SafeDetail,
    SafeJsonScalar,
    sanitize_embedding_scalar_mapping,
)


@dataclass(frozen=True, slots=True)
class EmbeddingRequestContext:
    user_id: str | int | None
    model_field: str | None
    provider_header: str | None
    dimensions: int | None
    encoding_format: str | None
    request_id: str | None = None
    endpoint_path: str = "/api/v1/embeddings"
    testing: bool = False
    adapters_enabled: bool = False


@dataclass(frozen=True, slots=True)
class NormalizedEmbeddingInput:
    texts: list[str]
    token_counts: list[int]
    total_tokens: int
    provided_token_arrays: bool = False
    token_input_mode: Literal["none", "single", "batch"] = "none"


@dataclass(frozen=True, slots=True)
class ProviderModelIntent:
    provider: str
    model: str
    requested_provider: str | None
    requested_model: str | None
    provider_was_explicit: bool
    model_was_provider_qualified: bool


@dataclass(frozen=True, slots=True)
class EmbeddingPolicyDecision:
    provider: str
    model: str
    dimensions: int | None
    fallback_chain: list[str]
    fallback_allowed: bool
    enforce_policy: bool
    bypass_reason: str | None = None


@dataclass(frozen=True, slots=True)
class EmbeddingExecutionPlan:
    provider: str
    model: str
    dimensions: int | None
    backend_identity: str | None
    fallback_chain: list[str]
    cache_namespace: str | None = None
    batch_size: int | None = None
    execution_path: Literal["legacy", "adapter"] = "legacy"
    observability_tags: dict[str, SafeJsonScalar] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "observability_tags",
            sanitize_embedding_scalar_mapping(self.observability_tags),
        )


@dataclass(frozen=True, slots=True)
class EmbeddingExecutionResult:
    vectors: list[list[float]]
    provider: str
    model: str
    prompt_tokens: int
    total_tokens: int
    cache_hits: int
    cache_misses: int
    fallback_from: str | None = None
    response_headers: dict[str, str] = field(default_factory=dict)
    embeddings_from_adapter: bool = False


__all__ = [
    "EmbeddingDomainError",
    "EmbeddingErrorCode",
    "EmbeddingExecutionError",
    "EmbeddingExecutionPlan",
    "EmbeddingExecutionResult",
    "EmbeddingInputError",
    "EmbeddingPolicyDecision",
    "EmbeddingPolicyError",
    "EmbeddingProviderError",
    "EmbeddingRateLimitError",
    "EmbeddingRequestContext",
    "SafeDetail",
    "SafeJsonScalar",
    "NormalizedEmbeddingInput",
    "ProviderModelIntent",
]
