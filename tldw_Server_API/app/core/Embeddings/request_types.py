"""Internal request contracts for the embeddings orchestration path."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

SafeJsonScalar = str | int | float | bool | None
SafeDetail = dict[str, SafeJsonScalar]

EmbeddingErrorCode = Literal[
    "empty_input",
    "invalid_input_type",
    "too_many_inputs",
    "input_too_long",
    "invalid_token_array",
    "unknown_provider",
    "provider_model_mismatch",
    "invalid_dimensions",
    "provider_denied",
    "model_denied",
    "provider_unsupported",
    "missing_provider_credentials",
    "provider_malformed_response",
    "provider_rate_limited",
    "provider_unavailable",
    "fallback_exhausted",
    "circuit_breaker_open",
    "internal_execution_failure",
]

_REDACTED = "[redacted]"
_SENSITIVE_KEY_PARTS = (
    "api_key",
    "apikey",
    "access_token",
    "authorization",
    "body",
    "credential",
    "header",
    "input",
    "password",
    "raw",
    "secret",
    "text",
)
_SAFE_NUMERIC_TOKEN_KEYS = {"tokens", "token_count", "prompt_tokens", "total_tokens"}
_SENSITIVE_VALUE_PARTS = (
    "api_key",
    "authorization",
    "bearer ",
    "password",
    "secret",
    "sk-",
    "token",
)


def _is_sensitive_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    return any(part in normalized for part in _SENSITIVE_KEY_PARTS)


def _is_sensitive_string(value: str) -> bool:
    normalized = value.lower()
    return any(part in normalized for part in _SENSITIVE_VALUE_PARTS)


def _is_safe_numeric_token_count(key: str, value: object) -> bool:
    normalized = key.lower().replace("-", "_")
    return normalized in _SAFE_NUMERIC_TOKEN_KEYS and isinstance(value, (int, float)) and not isinstance(value, bool)


def _sanitize_public_details(details: list[SafeDetail] | None) -> list[SafeDetail]:
    if not isinstance(details, list):
        return []

    sanitized: list[SafeDetail] = []
    for item in details:
        if not isinstance(item, Mapping):
            continue

        safe_item: SafeDetail = {}
        for raw_key, raw_value in item.items():
            if not isinstance(raw_key, str):
                continue

            if _is_safe_numeric_token_count(raw_key, raw_value):
                safe_item[raw_key] = raw_value
                continue

            if _is_sensitive_key(raw_key):
                safe_item[raw_key] = _REDACTED
                continue

            if isinstance(raw_value, str):
                safe_item[raw_key] = _REDACTED if _is_sensitive_string(raw_value) else raw_value
            elif isinstance(raw_value, (int, float, bool)) or raw_value is None:
                safe_item[raw_key] = raw_value

        if safe_item:
            sanitized.append(safe_item)

    return sanitized


def _sanitize_scalar_mapping(values: Mapping[str, object] | None) -> dict[str, SafeJsonScalar]:
    if not isinstance(values, Mapping):
        return {}

    sanitized: dict[str, SafeJsonScalar] = {}
    for raw_key, raw_value in values.items():
        if not isinstance(raw_key, str):
            continue

        if _is_safe_numeric_token_count(raw_key, raw_value):
            sanitized[raw_key] = raw_value
            continue

        if _is_sensitive_key(raw_key):
            sanitized[raw_key] = _REDACTED
            continue

        if isinstance(raw_value, str):
            sanitized[raw_key] = _REDACTED if _is_sensitive_string(raw_value) else raw_value
        elif isinstance(raw_value, (int, float, bool)) or raw_value is None:
            sanitized[raw_key] = raw_value

    return sanitized


class EmbeddingDomainError(Exception):
    """Base domain error for embedding request planning and execution."""

    def __init__(
        self,
        code: EmbeddingErrorCode,
        message: str,
        *,
        retryable: bool = False,
        provider: str | None = None,
        model: str | None = None,
        retry_after: int | float | None = None,
        cause_class: str | None = None,
        details: list[SafeDetail] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retryable = retryable
        self.provider = provider
        self.model = model
        self.retry_after = retry_after
        self.cause_class = cause_class
        self.details = _sanitize_public_details(details)

    def to_http_payload(self) -> dict[str, SafeJsonScalar | list[SafeDetail]]:
        """Return a stable, sanitized payload safe for HTTP responses."""
        return {
            "error_code": self.code,
            "message": self.message,
            "provider": self.provider,
            "model": self.model,
            "retryable": self.retryable,
            "retry_after": self.retry_after,
            "details": _sanitize_public_details(self.details),
            "cause_class": self.cause_class,
        }


class EmbeddingInputError(EmbeddingDomainError):
    """Input validation failed before provider execution."""


class EmbeddingPolicyError(EmbeddingDomainError):
    """Provider or model policy rejected the request."""


class EmbeddingProviderError(EmbeddingDomainError):
    """Provider returned an error response or malformed result."""


class EmbeddingRateLimitError(EmbeddingProviderError):
    """Provider rate limit or quota throttled the request."""


class EmbeddingExecutionError(EmbeddingDomainError):
    """Embedding execution failed after request planning."""


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
        object.__setattr__(self, "observability_tags", _sanitize_scalar_mapping(self.observability_tags))


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
