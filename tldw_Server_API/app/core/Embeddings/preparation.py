"""Ordered preparation pipeline for embedding requests."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

from tldw_Server_API.app.core.Embeddings.embedding_policy import enforce_embedding_policy
from tldw_Server_API.app.core.Embeddings.input_normalizer import normalize_embedding_input
from tldw_Server_API.app.core.Embeddings.provider_resolution import (
    ProviderGuesser,
    resolve_provider_model,
)
from tldw_Server_API.app.core.Embeddings.request_types import (
    EmbeddingExecutionPlan,
    EmbeddingRequestContext,
    PreparedEmbeddingRequest,
)
from tldw_Server_API.app.core.Embeddings.workflow_types import EmbeddingWorkflowPhase

TokenCounter = Callable[[str, str], int]
TokenDecoder = Callable[[list[int] | list[list[int]], str], object]
BackendIdentityResolver = Callable[[str, str], str | None]
PhaseSink = Callable[[EmbeddingWorkflowPhase], None]


def effective_dimension_policy(
    encoding_format: str | None,
    dimensions: int | None,
    configured_policy: str,
) -> str:
    """Return the request's effective dimension adjustment policy."""
    if dimensions is not None and encoding_format == "base64":
        return "reduce"
    return configured_policy


class EmbeddingPreparationPipeline:
    """Resolve, normalize, validate, and plan an embedding request."""

    def __init__(
        self,
        *,
        count_tokens: TokenCounter,
        tokens_to_texts: TokenDecoder,
        settings_config: Mapping[str, object] | None,
        max_tokens: int,
        implemented_providers: Sequence[str] | set[str],
        allowed_providers: Sequence[str] | set[str] | None,
        allowed_models: Sequence[str] | set[str] | None,
        enforce_policy: bool,
        allow_fallback_with_header: bool,
        settings_fallback_chain: Mapping[str, object] | None,
        settings_fallback_model_map: Mapping[str, object] | None,
        dimension_policy: str,
        require_model: bool,
        guess_provider: ProviderGuesser | None,
        backend_identity_resolver: BackendIdentityResolver,
        cache_namespace: str | None,
        batch_size: int | None,
        execution_path: Literal["legacy", "adapter"],
    ) -> None:
        self._count_tokens = count_tokens
        self._tokens_to_texts = tokens_to_texts
        self._settings_config = settings_config
        self._max_tokens = max_tokens
        self._implemented_providers = implemented_providers
        self._allowed_providers = allowed_providers
        self._allowed_models = allowed_models
        self._enforce_policy = enforce_policy
        self._allow_fallback_with_header = allow_fallback_with_header
        self._settings_fallback_chain = settings_fallback_chain
        self._settings_fallback_model_map = settings_fallback_model_map
        self._dimension_policy = dimension_policy
        self._require_model = require_model
        self._guess_provider = guess_provider
        self._backend_identity_resolver = backend_identity_resolver
        self._cache_namespace = cache_namespace
        self._batch_size = batch_size
        self._execution_path = execution_path

    def prepare(
        self,
        raw_input: Any,
        context: EmbeddingRequestContext,
        phase_sink: PhaseSink | None = None,
    ) -> PreparedEmbeddingRequest:
        """Prepare a request while reporting each phase before its boundary."""
        if phase_sink is not None:
            phase_sink("resolving_intent")
        intent = resolve_provider_model(
            context.model_field,
            context.provider_header,
            settings_config=self._settings_config,
            require_model=self._require_model,
            guess_provider=self._guess_provider,
        )

        if phase_sink is not None:
            phase_sink("normalizing")
        normalized_input = normalize_embedding_input(
            raw_input,
            model=intent.model,
            max_tokens=self._max_tokens,
            count_tokens=self._count_tokens,
            tokens_to_texts=self._tokens_to_texts,
        )

        if phase_sink is not None:
            phase_sink("resolving_policy")
        decision = enforce_embedding_policy(
            intent,
            context,
            allowed_providers=self._allowed_providers,
            allowed_models=self._allowed_models,
            implemented_providers=self._implemented_providers,
            enforce_policy=self._enforce_policy,
            allow_fallback_with_header=self._allow_fallback_with_header,
            settings_fallback_chain=self._settings_fallback_chain,
            settings_fallback_model_map=self._settings_fallback_model_map,
        )

        if phase_sink is not None:
            phase_sink("planning")
        dimension_policy = effective_dimension_policy(
            context.encoding_format,
            decision.dimensions,
            self._dimension_policy,
        )
        execution_plan = EmbeddingExecutionPlan(
            provider=decision.provider,
            model=decision.model,
            dimensions=decision.dimensions,
            backend_identity=self._backend_identity_resolver(decision.provider, decision.model),
            fallback_chain=list(decision.fallback_chain),
            cache_namespace=self._cache_namespace,
            batch_size=self._batch_size,
            execution_path=self._execution_path,
            observability_tags={
                "provider": decision.provider,
                "model": decision.model,
                "fallback_allowed": decision.fallback_allowed,
            },
        )
        return PreparedEmbeddingRequest(
            normalized_input=normalized_input,
            provider_intent=intent,
            policy_decision=decision,
            execution_plan=execution_plan,
            effective_dimension_policy=dimension_policy,
            prompt_tokens=normalized_input.total_tokens,
            total_tokens=normalized_input.total_tokens,
        )


__all__ = [
    "BackendIdentityResolver",
    "EmbeddingPreparationPipeline",
    "PhaseSink",
    "TokenCounter",
    "TokenDecoder",
    "effective_dimension_policy",
]
