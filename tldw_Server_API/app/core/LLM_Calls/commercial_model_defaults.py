"""Published limits and bundled budgets for supported commercial defaults."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit

# Provider, endpoint host, effective model, context, advertised output, bundled output.
_MODELS = {
    "anthropic": ("api.anthropic.com", "claude-sonnet-4-6", 1_000_000, 128_000, 128_000),
    "bedrock": ("bedrock-runtime.us-west-2.amazonaws.com", "openai.gpt-oss-20b-1:0", 131_072, 16_384, 16_384),
    "cohere": ("api.cohere.ai api.cohere.com", "command-a-plus-05-2026", 128_000, 64_000, 64_000),
    "deepseek": ("api.deepseek.com", "deepseek-flash", 1_048_576, 393_216, 200_000),
    "google": ("generativelanguage.googleapis.com", "gemini-3.5-flash-lite", 1_048_576, 65_536, 65_536),
    "groq": ("api.groq.com", "openai/gpt-oss-120b", 131_072, 65_536, 65_536),
    "huggingface": ("router.huggingface.co", "Qwen/Qwen3.6-35B-A3B", 262_144, None, 32_768),
    "mistral": ("api.mistral.ai", "mistral-small-2603", 262_144, None, 4096),
    "moonshot": ("api.moonshot.cn api.moonshot.ai", "kimi-k3.1-pro", 1_048_576, None, 131_072),
    "openai": ("api.openai.com eu.api.openai.com", "gpt-6-luna", 1_050_000, 128_000, 128_000),
    "openrouter": ("openrouter.ai", "google/gemini-3.1-flash-lite", 1_048_576, 65_536, 65_536),
    "qwen": ("dashscope-intl.aliyuncs.com", "qwen3.7-plus", 1_000_000, 131_072, 131_072),
    "zai": ("api.z.ai", "glm-5.3-flash", 1_000_000, 131_072, 131_072),
}


NATIVE_COMPLETION_TOKEN_PROVIDERS = frozenset({"openai", "groq", "moonshot", "qwen"})


# Verify these capabilities against both the model and the generation adapter.
_MODEL_FEATURES = {
    "anthropic": ("vision", "tool_use", "function_calling", "thinking"),
    "deepseek": ("tool_use", "function_calling", "json_mode", "thinking"),
    "google": ("vision", "tool_use", "function_calling", "json_mode", "thinking"),
    "groq": ("tool_use", "function_calling", "json_mode", "thinking"),
    "moonshot": ("tool_use", "function_calling", "json_mode", "thinking"),
    "openai": ("vision", "tool_use", "function_calling", "json_mode", "thinking"),
}


def commercial_model_capabilities(provider: str) -> dict[str, bool]:
    """Return verified generation capabilities for an already matched bundled model."""
    return dict.fromkeys(("streaming", *_MODEL_FEATURES.get(provider, ())), True)


def commercial_model_limits(provider: str, model: str) -> dict[str, int | None]:
    """Return published limits for a bundled commercial model."""
    entry = _MODELS.get(provider)
    if entry and entry[1] == model:
        return {"context_window": entry[2], "max_output_tokens": entry[3]}
    return {}


def is_bundled_model_endpoint(provider: str, model: Any, endpoint: str | None) -> bool:
    entry = _MODELS.get(provider)
    if entry is None or model != entry[1] or not endpoint:
        return False
    try:
        parsed = urlsplit(endpoint)
        return parsed.scheme == "https" and parsed.hostname in entry[0].split() and parsed.port in (None, 443)
    except ValueError:
        return False


def apply_commercial_output_default(provider: str, request: dict[str, Any]) -> None:
    """Fill a missing output budget after resolving the model and credential scope."""
    entry = _MODELS.get(provider)
    if not entry:
        return

    from tldw_Server_API.app.core.LLM_Calls.adapter_utils import resolve_provider_model, resolve_provider_section
    from tldw_Server_API.app.core.LLM_Calls.capability_registry import normalize_payload

    completion_field = provider in NATIVE_COMPLETION_TOKEN_PROVIDERS
    normalized = normalize_payload(provider, request)
    if normalized.get("max_completion_tokens") is not None:
        if not completion_field and normalized.get("max_tokens") is None:
            request["max_tokens"] = normalized["max_completion_tokens"]
        return
    if normalized.get("max_tokens") is not None:
        request["max_tokens"] = normalized["max_tokens"]
        return

    extra = request.get("extra_body")
    if isinstance(extra, Mapping):
        normalized_extra = normalize_payload(provider, extra)
        for key in ("max_completion_tokens", "max_tokens"):
            if normalized_extra.get(key) is not None:
                native_key = key if completion_field else "max_tokens"
                request[native_key] = normalized_extra[key]
                request["extra_body"] = {
                    field: value
                    for field, value in extra.items()
                    if not {"max_tokens", "max_completion_tokens"}.intersection(
                        normalize_payload(provider, {field: value})
                    )
                }
                return
    from tldw_Server_API.app.core.LLM_Calls.provider_model_inventory import resolve_provider_models_base_url

    config = request.get("app_config") or {}
    scoped = config.get(resolve_provider_section(provider)) or {}
    configured = scoped.get("max_output_tokens") if provider == "google" else None
    configured = configured if configured is not None else scoped.get("max_tokens")
    if configured not in (None, ""):
        # The provider-specific adapters retain their own handling of invalid configured values.
        if isinstance(configured, int) and not isinstance(configured, bool) and configured > 0:
            request["max_tokens"] = configured
        elif isinstance(configured, str) and configured.strip().isascii() and configured.strip().isdigit():
            request["max_tokens"] = int(configured)
        return

    _hosts, _model, context, _ceiling, budget = entry
    selected_model = request.get("model") or resolve_provider_model(provider, config)
    if provider == "bedrock":
        from tldw_Server_API.app.core.LLM_Calls.providers.bedrock_adapter import BedrockAdapter

        endpoint = BedrockAdapter()._base_url(request)
    else:
        endpoint = resolve_provider_models_base_url(
            provider,
            config,
            base_url=request.get("base_url"),
            credentials_resolved=request.get("credentials_resolved") is True,
        )
    if not is_bundled_model_endpoint(provider, selected_model, endpoint):
        # Older shipped configs supplied this cap even for another model or endpoint.
        if "max_tokens" in scoped and provider not in {"moonshot", "zai"}:
            request["max_tokens"] = 4096
        return

    if provider == "anthropic" and not request.get("stream"):
        budget = 21_333
    if provider in {"cohere", "google"}:  # Separate provider input and output limits.
        request["max_tokens"] = budget
        return

    from tldw_Server_API.app.core.Chat.chat_service import estimate_tokens_from_json

    prompt = {key: request[key] for key in ("messages", "system_message", "tools") if request.get(key)}
    remaining = context - estimate_tokens_from_json(json.dumps(prompt, default=str))
    request["max_tokens"] = max(1, min(budget, remaining))
