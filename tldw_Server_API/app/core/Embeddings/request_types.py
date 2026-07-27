"""Internal request contracts for the embeddings orchestration path."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from tldw_Server_API.app.core.Embeddings.vector_validation import (
    validated_embedding_vectors,
)
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
class PreparedEmbeddingRequest:
    normalized_input: NormalizedEmbeddingInput
    provider_intent: ProviderModelIntent
    policy_decision: EmbeddingPolicyDecision
    execution_plan: EmbeddingExecutionPlan
    effective_dimension_policy: str
    prompt_tokens: int
    total_tokens: int


@dataclass(frozen=True, slots=True)
class EmbeddingExecutorOutput:
    vectors: list[list[float]]
    embeddings_from_adapter: bool = False


@dataclass(frozen=True, slots=True)
class EmbeddingExecutionOutcome:
    vectors: tuple[tuple[float, ...], ...]
    provider: str
    model: str
    prompt_tokens: int
    total_tokens: int
    cache_hits: int
    cache_misses: int
    requested_dimensions: int | None
    effective_dimension_policy: str
    attempt_count: int
    fallback_attempt_count: int
    fallback_from: str | None = None
    embeddings_from_adapter: bool = False

    def __post_init__(self) -> None:
        invalid_vectors_message = "vectors must contain equally sized, non-empty vectors of finite numbers"
        try:
            vector_lists = [list(vector) for vector in self.vectors]
        except TypeError as exc:
            raise ValueError(invalid_vectors_message) from exc
        validated_vectors = validated_embedding_vectors(
            vector_lists,
            expected=len(vector_lists),
        )
        if validated_vectors is None:
            raise ValueError(invalid_vectors_message)
        object.__setattr__(
            self,
            "vectors",
            tuple(tuple(vector) for vector in validated_vectors),
        )

        for field_name, value in (
            ("prompt_tokens", self.prompt_tokens),
            ("total_tokens", self.total_tokens),
            ("cache_hits", self.cache_hits),
            ("cache_misses", self.cache_misses),
            ("attempt_count", self.attempt_count),
            ("fallback_attempt_count", self.fallback_attempt_count),
        ):
            if type(value) is not int:
                raise ValueError(f"{field_name} must be an exact int")
            if value < 0:
                raise ValueError(f"{field_name} must be nonnegative")
        if self.attempt_count < 1:
            raise ValueError("attempt_count must be at least 1 for a successful outcome")
        if self.fallback_attempt_count >= self.attempt_count:
            raise ValueError("fallback_attempt_count must be less than attempt_count")
        if self.fallback_from is not None and self.fallback_attempt_count == 0:
            raise ValueError("fallback_from requires a positive fallback_attempt_count")
        if self.fallback_attempt_count > 0 and self.fallback_from is None:
            raise ValueError("positive fallback_attempt_count requires fallback_from")
        non_fallback_attempt_count = self.attempt_count - self.fallback_attempt_count
        if non_fallback_attempt_count not in (1, 2):
            raise ValueError("attempt_count - fallback_attempt_count must be 1 or 2")
        if self.cache_hits + self.cache_misses != len(self.vectors):
            raise ValueError("cache_hits + cache_misses must equal the number of vectors")


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
    "EmbeddingExecutionOutcome",
    "EmbeddingExecutionPlan",
    "EmbeddingExecutionResult",
    "EmbeddingExecutorOutput",
    "EmbeddingInputError",
    "EmbeddingPolicyDecision",
    "EmbeddingPolicyError",
    "EmbeddingProviderError",
    "EmbeddingRateLimitError",
    "EmbeddingRequestContext",
    "SafeDetail",
    "SafeJsonScalar",
    "NormalizedEmbeddingInput",
    "PreparedEmbeddingRequest",
    "ProviderModelIntent",
]
