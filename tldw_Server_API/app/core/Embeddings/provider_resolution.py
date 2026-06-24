"""Provider and model resolution helpers for embeddings requests."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from tldw_Server_API.app.core.Embeddings.request_types import (
    EmbeddingPolicyError,
    ProviderModelIntent,
)

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OPENAI_EMBEDDING_MODELS = frozenset(
    {
        "text-embedding-3-small",
        "text-embedding-3-large",
        "text-embedding-ada-002",
    }
)
HUGGINGFACE_MODEL_PREFIXES = (
    "sentence-transformers/",
    "BAAI/",
    "thenlper/",
    "intfloat/",
    "hkunlp/",
    "Qwen/",
    "microsoft/",
    "google/",
    "facebook/",
    "bert-",
    "roberta-",
    "xlm-",
    "distilbert-",
    "all-MiniLM-",
    "all-mpnet-",
)

ProviderGuesser = Callable[[str, str | None], str | None]


def split_provider_model(model: str) -> tuple[str | None, str]:
    """Split provider-qualified model IDs like ``openai:text-embedding-3-small``."""
    if not isinstance(model, str):
        return None, str(model)
    if ":" in model:
        prefix, rest = model.split(":", 1)
        prefix = prefix.strip().lower()
        rest = rest.strip()
        if prefix and rest:
            return prefix, rest
    return None, model


def resolve_provider_model(
    model,
    provider_header,
    *,
    settings_config: Mapping[str, object] | None,
    require_model: bool,
    guess_provider: ProviderGuesser | None = None,
) -> ProviderModelIntent:
    """Resolve the effective provider and unqualified model for an embeddings request."""
    requested_model = model if isinstance(model, str) else None
    requested_provider = provider_header if isinstance(provider_header, str) else None

    if model is None or (isinstance(model, str) and not model.strip()):
        if require_model:
            raise EmbeddingPolicyError("model_denied", "Model is required")
        model = _default_model(settings_config)

    prefix_provider, stripped_model = split_provider_model(model)
    explicit_provider = _normalize_provider(provider_header)
    if explicit_provider:
        if prefix_provider and prefix_provider != explicit_provider:
            raise EmbeddingPolicyError(
                "provider_model_mismatch",
                f"Model provider prefix '{prefix_provider}' does not match provider '{explicit_provider}'",
                provider=explicit_provider,
                model=stripped_model,
            )
        return ProviderModelIntent(
            provider=explicit_provider,
            model=stripped_model,
            requested_provider=requested_provider,
            requested_model=requested_model,
            provider_was_explicit=True,
            model_was_provider_qualified=prefix_provider is not None,
        )

    if prefix_provider:
        return ProviderModelIntent(
            provider=prefix_provider,
            model=stripped_model,
            requested_provider=requested_provider,
            requested_model=requested_model,
            provider_was_explicit=False,
            model_was_provider_qualified=True,
        )

    resolved_provider = _heuristic_provider(stripped_model)
    if resolved_provider is None and guess_provider is not None:
        guessed_provider = guess_provider(stripped_model, None)
        if guessed_provider:
            resolved_provider = str(guessed_provider).lower()
    if resolved_provider is None:
        resolved_provider = "openai"

    return ProviderModelIntent(
        provider=resolved_provider,
        model=stripped_model,
        requested_provider=requested_provider,
        requested_model=requested_model,
        provider_was_explicit=False,
        model_was_provider_qualified=False,
    )


def _default_model(settings_config: Mapping[str, object] | None) -> object:
    if isinstance(settings_config, Mapping):
        return (
            settings_config.get("embedding_model")
            or settings_config.get("default_model_id")
            or DEFAULT_EMBEDDING_MODEL
        )
    return DEFAULT_EMBEDDING_MODEL


def _normalize_provider(provider_header) -> str | None:
    if not isinstance(provider_header, str):
        return None
    provider = provider_header.strip().lower()
    return provider or None


def _heuristic_provider(model: str) -> str | None:
    if model in OPENAI_EMBEDDING_MODELS:
        return None
    if "/" in model or model.startswith(HUGGINGFACE_MODEL_PREFIXES):
        return "huggingface"
    return None


__all__ = [
    "DEFAULT_EMBEDDING_MODEL",
    "HUGGINGFACE_MODEL_PREFIXES",
    "OPENAI_EMBEDDING_MODELS",
    "ProviderGuesser",
    "resolve_provider_model",
    "split_provider_model",
]
