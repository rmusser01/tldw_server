"""Runtime helpers for LLM-backed auto-routing."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Mapping, Sequence

from tldw_Server_API.app.core.config import load_and_log_configs
from tldw_Server_API.app.core.exceptions import LLMPolicyRefusal
from .llm_router import build_router_prompt
from .models import RouterRequest, RoutingPolicy


@dataclass(frozen=True)
class RouterModelConfig:
    provider: str
    model: str


def _normalize_string(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _normalize_provider(value: Any) -> str | None:
    text = _normalize_string(value)
    return text.lower() if text else None


def _coerce_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _provider_listing_entry(
    provider_listing: Mapping[str, Any] | None,
    provider_name: str,
) -> Mapping[str, Any] | None:
    if not isinstance(provider_listing, Mapping):
        return None
    providers = provider_listing.get("providers")
    if not isinstance(providers, Sequence):
        return None
    for provider in providers:
        if not isinstance(provider, Mapping):
            continue
        if _normalize_provider(provider.get("name")) == provider_name:
            return provider
    return None


def flatten_provider_listing_for_routing(
    provider_listing: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    """Flatten the providers payload into routing candidate records."""

    catalog: list[dict[str, Any]] = []
    if not isinstance(provider_listing, Mapping):
        return catalog
    providers = provider_listing.get("providers")
    if not isinstance(providers, Sequence):
        return catalog
    for provider in providers:
        if not isinstance(provider, Mapping):
            continue
        provider_name = _normalize_provider(provider.get("name"))
        if not provider_name:
            continue
        if provider.get("is_configured") is False:
            continue
        models_info = provider.get("models_info")
        if not isinstance(models_info, Sequence):
            continue
        for model_info in models_info:
            if not isinstance(model_info, Mapping):
                continue
            model_name = _normalize_string(model_info.get("name") or model_info.get("id"))
            if not model_name:
                continue
            candidate_record = dict(model_info)
            candidate_record["provider"] = provider_name
            candidate_record["model"] = model_name
            catalog.append(candidate_record)
    return catalog


def build_provider_order_for_routing(
    provider_listing: Mapping[str, Any] | None,
    *,
    objective: str,
    priority_resolver: Callable[[str, str], list[str] | None],
) -> dict[str, list[str]]:
    """Collect admin/provider-specific ordering hints for deterministic fallback."""

    provider_order: dict[str, list[str]] = {}
    if not isinstance(provider_listing, Mapping):
        return provider_order
    providers = provider_listing.get("providers")
    if not isinstance(providers, Sequence):
        return provider_order
    for provider in providers:
        if not isinstance(provider, Mapping):
            continue
        provider_name = _normalize_provider(provider.get("name"))
        if not provider_name:
            continue
        preferred_models = priority_resolver(provider_name, objective) or []
        if preferred_models:
            provider_order[provider_name] = preferred_models
    return provider_order


def _router_config_mapping(config: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not isinstance(config, Mapping):
        return {}
    model_routing = config.get("model_routing")
    if isinstance(model_routing, Mapping):
        return model_routing
    return config


def resolve_router_model_config(
    *,
    provider_listing: Mapping[str, Any] | None,
    server_default_provider: str,
    config: Mapping[str, Any] | None = None,
) -> RouterModelConfig | None:
    """Resolve the dedicated router provider/model, falling back to provider defaults."""

    runtime_config = _router_config_mapping(config or load_and_log_configs() or {})

    provider = (
        _normalize_provider(os.getenv("MODEL_ROUTING_ROUTER_PROVIDER"))
        or _normalize_provider(os.getenv("MODEL_ROUTER_PROVIDER"))
        or _normalize_provider(runtime_config.get("router_provider"))
    )
    provider = provider or _normalize_provider(server_default_provider)
    if provider is None:
        return None

    router_model = (
        _normalize_string(os.getenv("MODEL_ROUTING_ROUTER_MODEL"))
        or _normalize_string(os.getenv("MODEL_ROUTER_MODEL"))
        or _normalize_string(runtime_config.get("router_model"))
    )
    if router_model is None:
        provider_entry = _provider_listing_entry(provider_listing, provider)
        if provider_entry is not None:
            router_model = _normalize_string(provider_entry.get("default_model"))
            if router_model is None:
                models_info = provider_entry.get("models_info")
                if isinstance(models_info, Sequence):
                    for model_info in models_info:
                        if not isinstance(model_info, Mapping):
                            continue
                        router_model = _normalize_string(model_info.get("name"))
                        if router_model:
                            break
    if router_model is None:
        return None

    return RouterModelConfig(provider=provider, model=router_model)


def build_router_messages(prompt_payload: Mapping[str, Any]) -> list[dict[str, str]]:
    """Build a compact message list for the dedicated router model."""

    return [
        {
            "role": "system",
            "content": (
                "Choose the best model candidate for the request. "
                "Return only JSON with keys provider and model."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(prompt_payload, separators=(",", ":"), ensure_ascii=True),
        },
    ]


RouterExecutor = Callable[[RouterModelConfig, list[dict[str, str]]], Awaitable[Any]]
RouterUsageLogger = Callable[[RouterModelConfig, dict[str, int], int], Awaitable[None]]


async def select_llm_router_choice(
    *,
    router_request: RouterRequest,
    policy: RoutingPolicy,
    candidates: list[dict[str, Any]],
    provider_listing: Mapping[str, Any] | None,
    execute_router_call: RouterExecutor,
    log_router_usage: RouterUsageLogger | None = None,
) -> tuple[dict[str, str] | None, dict[str, Any]]:
    """Run the optional LLM-based router selection flow."""

    if policy.strategy != "llm_router":
        return None, {"skipped": "strategy_rules_router"}
    if len(candidates) <= 1:
        return None, {"skipped": "single_candidate"}

    router_model = resolve_router_model_config(
        provider_listing=provider_listing,
        server_default_provider=policy.server_default_provider,
    )
    if router_model is None:
        return None, {"skipped": "router_model_unavailable"}

    router_start = time.time()
    try:
        router_response = await execute_router_call(
            router_model,
            build_router_messages(
                build_router_prompt(
                    request=router_request,
                    policy=policy,
                    candidates=candidates,
                )
            ),
        )
    except LLMPolicyRefusal:
        raise
    except Exception as exc:
        return None, {
            "router_model": {
                "provider": router_model.provider,
                "model": router_model.model,
            },
            "error": type(exc).__name__,
        }

    usage = extract_router_usage(router_response)
    if log_router_usage is not None:
        await log_router_usage(
            router_model,
            usage,
            int((time.time() - router_start) * 1000),
        )

    llm_router_choice = extract_router_choice(router_response)
    return llm_router_choice, {
        "router_model": {
            "provider": router_model.provider,
            "model": router_model.model,
        },
        "choice_received": llm_router_choice is not None,
    }


def extract_router_usage(response: Any) -> dict[str, int]:
    if not isinstance(response, Mapping):
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    usage = response.get("usage")
    if not isinstance(usage, Mapping):
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    prompt_tokens = _coerce_int(usage.get("prompt_tokens"))
    completion_tokens = _coerce_int(usage.get("completion_tokens"))
    total_tokens = _coerce_int(usage.get("total_tokens")) or (prompt_tokens + completion_tokens)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def extract_router_choice(response: Any) -> dict[str, str] | None:
    """Extract the first provider/model JSON object from an LLM router response."""

    if isinstance(response, Mapping):
        direct_provider = _normalize_provider(response.get("provider"))
        direct_model = _normalize_string(response.get("model"))
        if direct_provider and direct_model:
            return {"provider": direct_provider, "model": direct_model}

    content_text: str | None = None
    if isinstance(response, Mapping):
        choices = response.get("choices")
        if isinstance(choices, Sequence) and choices:
            choice = choices[0]
            if isinstance(choice, Mapping):
                message = choice.get("message")
                if isinstance(message, Mapping):
                    content = message.get("content")
                    if isinstance(content, list):
                        content_text = "".join(
                            str(part.get("text", ""))
                            for part in content
                            if isinstance(part, Mapping)
                        )
                    elif content is not None:
                        content_text = str(content)
                if content_text is None and choice.get("text") is not None:
                    content_text = str(choice.get("text"))
        if content_text is None and response.get("content") is not None:
            content = response.get("content")
            if isinstance(content, list):
                content_text = "".join(
                    str(part.get("text", ""))
                    for part in content
                    if isinstance(part, Mapping)
                )
            else:
                content_text = str(content)
        if content_text is None and response.get("output_text") is not None:
            content_text = str(response.get("output_text"))
    elif response is not None:
        content_text = str(response)

    if not content_text:
        return None

    decoder = json.JSONDecoder()
    for start, char in enumerate(content_text):
        if char != "{":
            continue
        try:
            parsed, _end = decoder.raw_decode(content_text[start:])
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, Mapping):
            continue
        provider = _normalize_provider(parsed.get("provider"))
        model = _normalize_string(parsed.get("model"))
        if not provider or not model:
            continue
        return {"provider": provider, "model": model}
    return None
