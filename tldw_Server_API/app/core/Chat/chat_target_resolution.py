"""Canonical provider and model resolution for direct chat calls."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable

from tldw_Server_API.app.api.v1.schemas.chat_request_schemas import (
    DEFAULT_LLM_PROVIDER,
)
from tldw_Server_API.app.core.AuthNZ.llm_provider_overrides import (
    get_llm_provider_override,
    get_override_default_model,
    validate_provider_override,
)
from tldw_Server_API.app.core.Chat.Chat_Deps import ChatConfigurationError
from tldw_Server_API.app.core.Chat.chat_service import (
    _split_inline_provider_model,
    resolve_provider_and_model,
)
from tldw_Server_API.app.core.config import (
    load_and_log_configs,
    load_comprehensive_config,
)
from tldw_Server_API.app.core.LLM_Calls import adapter_registry as _adapter_registry
from tldw_Server_API.app.core.testing import is_test_mode


@dataclass(frozen=True)
class ResolvedChatTarget:
    """One provider/model pair validated against server policy and adapters."""

    provider: str
    model: str


@dataclass
class _TargetRequest:
    api_provider: str | None
    model: str | None


@dataclass(frozen=True)
class _CandidateChatTarget:
    provider: str
    model: str | None


def config_default_llm_provider(
    config_data: dict[str, Any] | None = None,
) -> str | None:
    """Return the configured default provider without applying fallbacks."""

    cfg = load_and_log_configs() if config_data is None else config_data
    if not isinstance(cfg, dict):
        return None
    for section in ("llm_api_settings", "API"):
        section_data = cfg.get(section)
        if not isinstance(section_data, dict):
            continue
        value = section_data.get("default_api")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def get_default_provider(
    *,
    config_resolver: Callable[[], str | None] = config_default_llm_provider,
    test_mode_resolver: Callable[[], bool] = is_test_mode,
    fallback_provider: str = DEFAULT_LLM_PROVIDER,
) -> str:
    """Resolve the server default provider using the ordinary chat precedence."""

    configured = config_resolver()
    if isinstance(configured, str) and configured.strip():
        return configured.strip()
    environment = os.getenv("DEFAULT_LLM_PROVIDER")
    if isinstance(environment, str) and environment.strip():
        return environment.strip()
    if test_mode_resolver():
        return "local-llm"
    return str(fallback_provider or "").strip()


def get_default_model_for_provider(
    provider: str,
    *,
    override_default_resolver: Callable[[str], str | None] = get_override_default_model,
    override_resolver: Callable[[str], Any] = get_llm_provider_override,
    config_loader: Callable[[], Any] = load_comprehensive_config,
) -> str | None:
    """Resolve the ordinary server default model for one provider."""

    normalized_provider = str(provider or "").strip().lower()
    if not normalized_provider:
        return None

    override_default = override_default_resolver(normalized_provider)
    if isinstance(override_default, str) and override_default.strip():
        return override_default.strip()
    override = override_resolver(normalized_provider)
    allowed_models = getattr(override, "allowed_models", None)
    if isinstance(allowed_models, list):
        for allowed_model in allowed_models:
            if isinstance(allowed_model, str) and allowed_model.strip():
                return allowed_model.strip()

    config_name = normalized_provider.replace(".", "_").replace("-", "_")
    environment = os.getenv(f"DEFAULT_MODEL_{config_name.upper()}")
    if isinstance(environment, str) and environment.strip():
        return environment.strip()

    try:
        config = config_loader()
        if config is not None and config.has_section("Chat-Module"):
            configured = config.get(
                "Chat-Module",
                f"default_model_{config_name.lower()}",
                fallback=None,
            )
            if isinstance(configured, str) and configured.strip():
                return configured.strip()
    except (AttributeError, KeyError, TypeError, ValueError):
        return None
    return None


def _configuration_error(provider: str | None = None) -> ChatConfigurationError:
    return ChatConfigurationError(
        provider=provider,
        message="No usable chat provider and model are configured.",
    )


def _resolve_candidate_chat_target(
    *,
    requested_provider: str | None,
    requested_model: str | None,
    default_model_resolver: Callable[[str], str | None] | None = None,
) -> _CandidateChatTarget:
    """Normalize one provider/model candidate without checking target readiness."""

    provider_input = (
        requested_provider.strip()
        if isinstance(requested_provider, str) and requested_provider.strip()
        else None
    )
    model_input = (
        requested_model.strip()
        if isinstance(requested_model, str) and requested_model.strip()
        else None
    )
    default_provider = str(get_default_provider() or "").strip()
    registry = _adapter_registry.get_registry()

    preliminary_provider_input = provider_input
    if provider_input is None and isinstance(requested_model, str):
        inline_provider, inline_model, _separator = _split_inline_provider_model(
            requested_model
        )
        if inline_provider and inline_model:
            preliminary_provider_input = inline_provider
    preliminary_provider = registry.resolve_provider_name(
        preliminary_provider_input or default_provider
    )
    if not model_input and default_model_resolver is not None:
        model_input = default_model_resolver(preliminary_provider)
    if not model_input:
        return _CandidateChatTarget(provider=preliminary_provider, model=None)

    request_data = _TargetRequest(
        api_provider=provider_input,
        model=model_input,
    )
    _, _, selected_provider, selected_model, _ = resolve_provider_and_model(
        request_data=request_data,
        metrics_default_provider=default_provider,
        normalize_default_provider=default_provider,
        routing_decision=None,
    )
    return _CandidateChatTarget(
        provider=registry.resolve_provider_name(selected_provider),
        model=str(selected_model or "").strip() or None,
    )


def resolve_chat_provider_identity(
    *,
    requested_provider: str | None,
    requested_model: str | None,
) -> str:
    """Resolve only the local canonical provider identity for a request."""

    return _resolve_candidate_chat_target(
        requested_provider=requested_provider,
        requested_model=requested_model,
    ).provider


def resolve_chat_target(
    *,
    requested_provider: str | None,
    requested_model: str | None,
) -> ResolvedChatTarget:
    """Resolve one direct target without routing or provider fallback."""

    registry = _adapter_registry.get_registry()
    try:
        candidate = _resolve_candidate_chat_target(
            requested_provider=requested_provider,
            requested_model=requested_model,
            default_model_resolver=get_default_model_for_provider,
        )
    except Exception as exc:  # noqa: BLE001 - normalize to one safe core error.
        raise _configuration_error() from exc

    provider = candidate.provider
    model = candidate.model or ""
    if provider not in set(registry.list_providers()) or not model:
        raise _configuration_error(provider or None)
    if validate_provider_override(provider, model) is not None:
        raise _configuration_error(provider)
    try:
        adapter = registry.get_adapter(provider)
    except Exception as exc:  # noqa: BLE001 - adapter diagnostics stay private.
        raise _configuration_error(provider) from exc
    if adapter is None:
        raise _configuration_error(provider)
    return ResolvedChatTarget(provider=provider, model=model)


__all__ = [
    "ResolvedChatTarget",
    "config_default_llm_provider",
    "get_default_model_for_provider",
    "get_default_provider",
    "resolve_chat_provider_identity",
    "resolve_chat_target",
]
