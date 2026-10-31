"""Shared provider/model capability resolution for Notes graph suggestions."""

from __future__ import annotations

from dataclasses import dataclass

from tldw_Server_API.app.core.Chat.chat_target_resolution import (
    get_default_model_for_provider,
    get_default_provider,
)
from tldw_Server_API.app.core.config import loaded_config_data
from tldw_Server_API.app.core.LLM_Calls import adapter_registry
from tldw_Server_API.app.core.LLM_Calls.adapter_utils import _resolve_openai_api_base

from .suggestion_capabilities import (
    ProviderCapabilityContract,
    SuggestionCapabilities,
    build_suggestion_capabilities,
    build_unavailable_suggestion_capabilities,
)
from .suggestion_generation import GenerationProvider, build_provider_call_policy


@dataclass(frozen=True, slots=True)
class ResolvedSuggestionProvider:
    """One provider object paired with the exact disclosed capability revision."""

    capabilities: SuggestionCapabilities
    provider: GenerationProvider


unavailable_generation_capability = build_unavailable_suggestion_capabilities


class SuggestionProviderResolutionError(ValueError):
    """Sanitized unresolved provider/model facts for capability preflight."""

    def __init__(self, *, provider: str | None, model: str | None) -> None:
        self.provider = provider
        self.model = model
        super().__init__("notes_graph_provider_model_disallowed")


def resolve_generation_capability(
    *,
    provider: str | None,
    model: str | None,
) -> ResolvedSuggestionProvider:
    """Resolve the configured provider/model once for API and worker parity."""

    requested_provider = provider.strip() if isinstance(provider, str) else ""
    default_provider = get_default_provider()
    resolved_provider = requested_provider or (
        default_provider.strip() if isinstance(default_provider, str) else ""
    )
    requested_model = model.strip() if isinstance(model, str) else ""
    resolved_model = requested_model or get_default_model_for_provider(resolved_provider) or ""
    if not resolved_provider or not resolved_model:
        raise SuggestionProviderResolutionError(
            provider=resolved_provider or None,
            model=resolved_model or None,
        )

    registry = adapter_registry.get_registry()
    canonical = registry.resolve_provider_name(resolved_provider)
    config = dict(loaded_config_data)
    openai_config = dict(config.get("openai_api") or {})
    endpoint = _resolve_openai_api_base(openai_config).rstrip("/") + "/chat/completions"
    adapter = registry.get_adapter(canonical)
    provider_capabilities = adapter.capabilities() if adapter is not None else {}
    api_key = openai_config.get("api_key") if canonical == "openai" else None
    contract = ProviderCapabilityContract(
        adapter=canonical,
        model=resolved_model,
        endpoint_url=endpoint,
        call_policy=build_provider_call_policy(
            allow_response_format=True,
            endpoint_url=endpoint,
        ),
        data_boundary="remote",
        credentials_available=bool(api_key),
        provider_healthy=adapter is not None,
    )
    return ResolvedSuggestionProvider(
        capabilities=build_suggestion_capabilities(contract),
        provider=GenerationProvider(
            adapter=canonical,
            model=resolved_model,
            endpoint_url=endpoint,
            api_key=str(api_key) if api_key else None,
            app_config=config,
            provider_capabilities=provider_capabilities or {},
        ),
    )


__all__ = [
    "ResolvedSuggestionProvider",
    "SuggestionProviderResolutionError",
    "resolve_generation_capability",
    "unavailable_generation_capability",
]
