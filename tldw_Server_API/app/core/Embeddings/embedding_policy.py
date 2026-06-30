"""Dependency-light policy helpers for embedding requests."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np

from tldw_Server_API.app.core.Embeddings.request_types import (
    EmbeddingPolicyDecision,
    EmbeddingPolicyError,
    EmbeddingRequestContext,
    ProviderModelIntent,
)

OPENAI_DIMENSIONS_MODELS = frozenset(
    {
        "text-embedding-3-small",
        "text-embedding-3-large",
    }
)
OPENAI_DIMENSION_MAXIMUMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}
KNOWN_EMBEDDING_PROVIDERS = frozenset(
    {
        "openai",
        "huggingface",
        "onnx",
        "local_api",
        "cohere",
        "voyage",
        "google",
        "mistral",
        "mlx",
    }
)
DEFAULT_FALLBACK_CHAINS = {
    "openai": ["openai", "huggingface", "onnx", "local_api"],
    "huggingface": ["huggingface", "onnx", "local_api"],
    "onnx": ["onnx", "huggingface", "local_api"],
    "local_api": ["local_api", "huggingface"],
}
DEFAULT_FALLBACK_MODEL_MAP = {
    "openai:text-embedding-3-small": {
        "huggingface": "sentence-transformers/all-MiniLM-L6-v2",
        "onnx": "sentence-transformers/all-MiniLM-L6-v2",
        "local_api": "sentence-transformers/all-MiniLM-L6-v2",
    },
    "openai:text-embedding-3-large": {
        "huggingface": "sentence-transformers/all-mpnet-base-v2",
        "onnx": "sentence-transformers/all-mpnet-base-v2",
        "local_api": "sentence-transformers/all-mpnet-base-v2",
    },
    "openai:text-embedding-ada-002": {
        "huggingface": "sentence-transformers/all-mpnet-base-v2",
        "onnx": "sentence-transformers/all-mpnet-base-v2",
        "local_api": "sentence-transformers/all-mpnet-base-v2",
    },
}

EnvGetter = Callable[[str, str | None], str | None]
AdjustmentRecorder = Callable[[str, str, str], None]


def dimension_policy(
    env_getter: EnvGetter | None = None,
    *,
    default: str = "reduce",
) -> str:
    """Return the configured dimension adjustment policy."""
    fallback = default if default in {"reduce", "pad", "ignore"} else "reduce"
    if env_getter is None:
        return fallback
    try:
        value = env_getter("EMBEDDINGS_DIMENSION_POLICY", fallback)
    except (RuntimeError, TypeError, ValueError):
        return fallback
    if isinstance(value, str):
        normalized = value.lower()
        if normalized in {"reduce", "pad", "ignore"}:
            return normalized
    return fallback


def supports_openai_dimensions(model: str) -> bool:
    """Return True when an OpenAI model supports the dimensions parameter."""
    model_key = (model or "").split(":", 1)[-1]
    return model_key in OPENAI_DIMENSIONS_MODELS


def validate_dimensions_request(provider: str, model: str, dimensions: int | None) -> int | None:
    """Validate requested dimensions for the provider/model pair."""
    if dimensions is None:
        return None
    try:
        dim = int(dimensions)
    except (TypeError, ValueError) as exc:
        raise EmbeddingPolicyError("invalid_dimensions", "dimensions must be an integer") from exc
    if dim <= 0:
        raise EmbeddingPolicyError("invalid_dimensions", f"dimensions must be positive, got {dim}")

    provider_key = (provider or "").lower()
    model_key = (model or "").split(":", 1)[-1]
    if provider_key == "openai":
        if not supports_openai_dimensions(model):
            raise EmbeddingPolicyError(
                "invalid_dimensions",
                "dimensions is only supported for OpenAI text-embedding-3-small and text-embedding-3-large models",
                provider=provider_key,
                model=model_key,
            )
        max_dim = OPENAI_DIMENSION_MAXIMUMS.get(model_key)
        if max_dim is not None and dim > max_dim:
            raise EmbeddingPolicyError(
                "invalid_dimensions",
                f"dimensions {dim} exceeds maximum {max_dim} for model {model_key}",
                provider=provider_key,
                model=model_key,
            )
    elif dim > 4096:
        raise EmbeddingPolicyError(
            "invalid_dimensions",
            "dimensions must be <= 4096 for non-OpenAI providers",
            provider=provider_key,
            model=model_key,
        )

    return dim


def adjust_dimensions(
    vectors: list[list[float]],
    target_dim: int | None,
    provider: str,
    model: str,
    *,
    dimension_policy: str = "reduce",
    record_adjustment: AdjustmentRecorder | None = None,
) -> list[list[float]]:
    """Apply reduce/pad/ignore dimension policy to numeric vectors."""
    if not target_dim or target_dim <= 0:
        return vectors

    policy = dimension_policy if dimension_policy in {"reduce", "pad", "ignore"} else "reduce"
    adjusted: list[list[float]] = []
    for vector in vectors:
        if not isinstance(vector, (list, tuple)):
            adjusted.append(vector)
            continue
        arr = np.asarray(vector, dtype=np.float32)
        current_dim = arr.shape[0]
        if current_dim == target_dim or policy == "ignore":
            adjusted.append(arr.tolist())
            continue
        if current_dim > target_dim:
            out = arr[:target_dim]
            adjusted.append(out.tolist())
            _record_adjustment(record_adjustment, provider, model, "reduce")
        elif policy == "pad":
            pad = np.zeros((target_dim - current_dim,), dtype=np.float32)
            out = np.concatenate([arr, pad], axis=0)
            adjusted.append(out.tolist())
            _record_adjustment(record_adjustment, provider, model, "pad")
        else:
            adjusted.append(arr.tolist())
    return adjusted


def decide_and_apply_l2(
    embedding: list[float] | np.ndarray,
    encoding_format: str,
    embeddings_from_adapter: bool,
    *,
    normalize_requested: bool | None = None,
) -> tuple[np.ndarray, bool]:
    """Decide and apply L2 normalization for a single embedding vector."""
    do_l2 = encoding_format != "base64"
    if embeddings_from_adapter:
        do_l2 = encoding_format != "base64" if normalize_requested is True else False

    try:
        arr = np.asarray(embedding, dtype=np.float32)
        if do_l2:
            norm = np.linalg.norm(arr)
            if norm > 0:
                arr = arr / norm
        return arr, do_l2
    except (RuntimeError, TypeError, ValueError):
        try:
            arr = np.asarray(embedding, dtype=np.float32)
        except (RuntimeError, TypeError, ValueError):
            arr = np.array(embedding)
        return arr, False


def resolve_fallback_chain(
    primary_provider: str,
    *,
    settings_fallback_chain: Mapping[str, object] | None = None,
) -> list[str]:
    """Return the configured or default fallback chain for a provider."""
    if isinstance(settings_fallback_chain, Mapping):
        chain = settings_fallback_chain.get(primary_provider)
        if isinstance(chain, list) and chain:
            return _dedupe_provider_chain(
                [primary_provider] + [provider for provider in chain if isinstance(provider, str)]
            )
    return _dedupe_provider_chain(DEFAULT_FALLBACK_CHAINS.get(primary_provider, [primary_provider]))


def fallback_model_map(
    settings_fallback_model_map: Mapping[str, object] | None = None,
) -> dict[str, dict[str, str]]:
    """Return the configured or default provider fallback model map."""
    if isinstance(settings_fallback_model_map, Mapping) and settings_fallback_model_map:
        return {
            str(key): {str(dst): str(model) for dst, model in value.items() if isinstance(model, str)}
            for key, value in settings_fallback_model_map.items()
            if isinstance(value, Mapping)
        }
    return {key: value.copy() for key, value in DEFAULT_FALLBACK_MODEL_MAP.items()}


def map_model_for_provider(
    src_provider: str,
    dst_provider: str,
    model_id: str,
    *,
    settings_fallback_model_map: Mapping[str, object] | None = None,
) -> str:
    """Map a model id to the destination provider if a mapping exists."""
    if not src_provider or not dst_provider:
        return model_id
    if src_provider == dst_provider:
        return model_id
    mapping = fallback_model_map(settings_fallback_model_map)
    dst_map = mapping.get(f"{src_provider}:{model_id}", {})
    mapped = dst_map.get(dst_provider)
    if isinstance(mapped, str) and mapped:
        return mapped
    return model_id


def enforce_embedding_policy(
    intent: ProviderModelIntent,
    context: EmbeddingRequestContext,
    *,
    allowed_providers: Sequence[str] | set[str] | None,
    allowed_models: Sequence[str] | set[str] | None,
    implemented_providers: Sequence[str] | set[str],
    enforce_policy: bool,
    allow_fallback_with_header: bool,
    settings_fallback_chain: Mapping[str, object] | None,
    settings_fallback_model_map: Mapping[str, object] | None,
) -> EmbeddingPolicyDecision:
    """Validate provider/model policy and return an execution policy decision."""
    provider = (intent.provider or "").lower()
    model = intent.model

    if provider not in KNOWN_EMBEDDING_PROVIDERS:
        raise EmbeddingPolicyError(
            "unknown_provider",
            f"Unknown provider: {provider}",
            provider=provider,
            model=model,
        )

    implemented = _normalize_allowlist(implemented_providers, lowercase=True) or set()
    if provider not in implemented:
        raise EmbeddingPolicyError(
            "provider_unsupported",
            f"Provider '{provider}' not implemented",
            provider=provider,
            model=model,
        )

    dimensions = validate_dimensions_request(provider, model, context.dimensions)

    if enforce_policy:
        provider_allowlist = _normalize_allowlist(allowed_providers, lowercase=True)
        if provider_allowlist is not None and provider not in provider_allowlist:
            raise EmbeddingPolicyError(
                "provider_denied",
                f"Provider '{provider}' is not allowed",
                provider=provider,
                model=model,
            )

        if allowed_models is not None and not is_model_allowed(
            provider,
            model,
            allowed_providers=None,
            allowed_models=allowed_models,
        ):
            raise EmbeddingPolicyError(
                "model_denied",
                f"Model '{model}' is not allowed",
                provider=provider,
                model=model,
            )

    fallback_allowed = not (context.provider_header is not None and not allow_fallback_with_header)
    fallback_chain = [provider] if not fallback_allowed else resolve_fallback_chain(
        provider,
        settings_fallback_chain=settings_fallback_chain,
    )
    if enforce_policy:
        provider_allowlist = _normalize_allowlist(allowed_providers, lowercase=True)
        if provider_allowlist is not None:
            fallback_chain = [p for p in fallback_chain if p.lower() in provider_allowlist or p == provider]

    return EmbeddingPolicyDecision(
        provider=provider,
        model=model,
        dimensions=dimensions,
        fallback_chain=fallback_chain,
        fallback_allowed=fallback_allowed,
        enforce_policy=enforce_policy,
    )


def is_model_allowed(
    provider: str,
    model: str,
    *,
    allowed_providers: Sequence[str] | set[str] | None,
    allowed_models: Sequence[str] | set[str] | None,
) -> bool:
    """Return whether provider/model satisfy endpoint-compatible allowlists."""
    provider_allowlist = _normalize_allowlist(allowed_providers, lowercase=True)
    if provider_allowlist is not None and provider.lower() not in provider_allowlist:
        return False
    if allowed_models is not None:
        for pattern in allowed_models:
            if not isinstance(pattern, str):
                continue
            if pattern.endswith("*") and model.startswith(pattern[:-1]):
                return True
            if model == pattern:
                return True
        return False
    return True


def _record_adjustment(
    record_adjustment: AdjustmentRecorder | None,
    provider: str,
    model: str,
    method: str,
) -> None:
    if record_adjustment is None:
        return
    record_adjustment(provider, model, method)


def _normalize_allowlist(values: Sequence[str] | set[str] | None, *, lowercase: bool) -> set[str] | None:
    if values is None:
        return None
    normalized = {str(value).lower() if lowercase else str(value) for value in values}
    return normalized if normalized else None


def _dedupe_provider_chain(values: Sequence[str]) -> list[str]:
    """Deduplicate provider fallback chains while preserving first-seen order."""
    chain: list[str] = []
    seen: set[str] = set()
    for value in values:
        provider = str(value or "").strip().lower()
        if not provider or provider in seen:
            continue
        seen.add(provider)
        chain.append(provider)
    return chain


_dimension_policy = dimension_policy
_supports_openai_dimensions = supports_openai_dimensions
_validate_dimensions_request = validate_dimensions_request
_fallback_model_map = fallback_model_map

__all__ = [
    "DEFAULT_FALLBACK_CHAINS",
    "DEFAULT_FALLBACK_MODEL_MAP",
    "KNOWN_EMBEDDING_PROVIDERS",
    "OPENAI_DIMENSIONS_MODELS",
    "adjust_dimensions",
    "decide_and_apply_l2",
    "dimension_policy",
    "enforce_embedding_policy",
    "fallback_model_map",
    "is_model_allowed",
    "map_model_for_provider",
    "resolve_fallback_chain",
    "supports_openai_dimensions",
    "validate_dimensions_request",
]
