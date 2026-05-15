from __future__ import annotations

"""
Provider capability registry for chat adapters.

Defines the allowlist of supported request fields per provider, plus alias
normalization and blocked field enforcement used by adapters.
"""

import math
from collections.abc import Mapping
from typing import Any

from tldw_Server_API.app.core.Chat.Chat_Deps import ChatBadRequestError
from tldw_Server_API.app.core.custom_openai_providers import custom_openai_provider_number

SCHEMA_VERSION = 1
_VALIDATION_METRICS_REGISTERED = False


def _ensure_validation_metrics() -> None:
    global _VALIDATION_METRICS_REGISTERED
    if _VALIDATION_METRICS_REGISTERED:
        return
    try:
        from tldw_Server_API.app.core.Metrics.metrics_manager import (
            MetricDefinition,
            MetricType,
            get_metrics_registry,
        )
    except Exception:
        return
    try:
        registry = get_metrics_registry()
        metric_name = "llm_request_validation_rejections_total"
        if metric_name not in registry.metrics:
            registry.register_metric(
                MetricDefinition(
                    name=metric_name,
                    type=MetricType.COUNTER,
                    description="LLM request validation rejections",
                    labels=["provider"],
                )
            )
        _VALIDATION_METRICS_REGISTERED = True
    except Exception:
        return


def _record_validation_rejection(provider_key: str) -> None:
    try:
        _ensure_validation_metrics()
        from tldw_Server_API.app.core.Metrics import increment_counter

        increment_counter(
            "llm_request_validation_rejections_total",
            labels={"provider": provider_key or "unknown"},
        )
    except Exception:
        return

# Base OpenAI-compatible request fields supported across providers.
BASE_FIELDS: set[str] = {
    "messages",
    "model",
    "temperature",
    "top_p",
    "max_tokens",
    "max_completion_tokens",
    "n",
    "presence_penalty",
    "frequency_penalty",
    "logit_bias",
    "logprobs",
    "top_logprobs",
    "user",
    "tools",
    "tool_choice",
    "response_format",
    "seed",
    "stop",
    "stream",
    "system_message",
    # Internal/common adapter fields
    "api_key",
    "base_url",
    "app_config",
    "custom_prompt_arg",
    "extra_headers",
    "extra_body",
    "billing_prompt_cache_intent",
    "inference_prefix_cache_intent",
}

# Provider-specific extension fields (non-OpenAI keys).
PROVIDER_EXTENSIONS: dict[str, set[str]] = {
    "anthropic": {"top_k"},
    "google": {"top_k"},
    "huggingface": {"top_k"},
    "mistral": {"top_k", "safe_prompt"},
    "openrouter": {"top_k", "min_p"},
    "novita": {"top_k", "min_p"},
    "poe": {"top_k", "min_p"},
    "together": {"top_k", "min_p"},
    "custom-openai-api": {"top_k", "min_p"},
    "custom-openai-api-2": {"top_k", "min_p"},
    "mlx": {"top_k", "prompt_template"},
    "cohere": {"top_k", "num_generations"},
    "zai": {"do_sample", "request_id"},
    "llama.cpp": {"top_k", "min_p"},
    "kobold": {"top_k"},
    "ooba": {"top_k", "min_p"},
    "tabbyapi": {"top_k", "min_p"},
    "vllm": {"top_k", "min_p"},
    "local-llm": {"top_k", "min_p"},
    "ollama": {"top_k"},
    "aphrodite": {"top_k", "min_p"},
}

# Alias mappings from legacy or provider-specific field names to canonical keys.
ALIASES: dict[str, dict[str, str]] = {
    "*": {
        "temp": "temperature",
        "streaming": "stream",
        "maxp": "top_p",
        "topp": "top_p",
        "topk": "top_k",
        "minp": "min_p",
        "system_prompt": "system_message",
        "user_identifier": "user",
        "api_base_url": "base_url",
        "custom_prompt": "custom_prompt_arg",
        "custom_prompt_input": "custom_prompt_arg",
        "prompt_cache_intent": "billing_prompt_cache_intent",
        "local_prefix_cache_intent": "inference_prefix_cache_intent",
    },
    "bedrock": {"maxp": "top_p", "topp": "top_p"},
    "openai": {"maxp": "top_p"},
    "qwen": {"maxp": "top_p", "topp": "top_p"},
    "openrouter": {"maxp": "top_p", "topp": "top_p", "topk": "top_k", "minp": "min_p"},
    "novita": {"maxp": "top_p", "topp": "top_p", "topk": "top_k", "minp": "min_p"},
    "poe": {"maxp": "top_p", "topp": "top_p", "topk": "top_k", "minp": "min_p"},
    "together": {"maxp": "top_p", "topp": "top_p", "topk": "top_k", "minp": "min_p"},
    "mistral": {"topk": "top_k", "random_seed": "seed"},
    "google": {
        "max_output_tokens": "max_tokens",
        "stop_sequences": "stop",
        "candidate_count": "n",
    },
    "huggingface": {"max_new_tokens": "max_tokens"},
    "anthropic": {"stop_sequences": "stop"},
    "cohere": {"stop_sequences": "stop"},
    "llama.cpp": {"n_predict": "max_tokens"},
    "kobold": {"max_length": "max_tokens", "stop_sequence": "stop", "num_responses": "n"},
}

# Explicit denylist for unsafe or unsupported keys.
BLOCKED_FIELDS: dict[str, set[str]] = {
    "cohere": {"tool_choice"},
    "google": {"tool_choice"},
    "llama.cpp": {"tools", "tool_choice"},
}


def _normalize_provider(provider: str) -> str:
    return (provider or "").strip().lower()


def _alias_map(provider: str) -> dict[str, str]:
    merged: dict[str, str] = {}
    merged.update(ALIASES.get("*", {}))
    if provider:
        merged.update(ALIASES.get(provider, {}))
    return merged


def normalize_payload(provider: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a normalized payload with aliases applied.

    Alias precedence:
    - Canonical keys win if both are present and non-None.
    - If canonical is missing or None, alias value fills it.
    """
    normalized: dict[str, Any] = dict(payload or {})
    aliases = _alias_map(_normalize_provider(provider))
    for alias, canonical in aliases.items():
        if alias not in normalized:
            continue
        alias_val = normalized.get(alias)
        canonical_val = normalized.get(canonical)
        if canonical not in normalized or canonical_val is None:
            normalized[canonical] = alias_val
        # Always drop alias to avoid duplicate keys downstream.
        normalized.pop(alias, None)
    return normalized


def get_allowed_fields(provider: str) -> set[str]:
    provider_key = _normalize_provider(provider)
    provider_extensions = PROVIDER_EXTENSIONS.get(provider_key, set())
    if not provider_extensions and custom_openai_provider_number(provider_key) is not None:
        provider_extensions = PROVIDER_EXTENSIONS.get("custom-openai-api", set())
    return set(BASE_FIELDS) | set(provider_extensions)


def _raise_nested_error(provider_key: str, field: str, message: str) -> None:
    _record_validation_rejection(provider_key)
    raise ChatBadRequestError(
        message=f"Invalid {field}: {message}",
        provider=provider_key or None,
    )


def _validate_tools(provider_key: str, tools: Any) -> None:
    if tools is None:
        return
    if not isinstance(tools, list):
        _raise_nested_error(provider_key, "tools", "must be an array")
    for idx, tool in enumerate(tools):
        if not isinstance(tool, dict):
            _raise_nested_error(provider_key, "tools", f"item {idx} must be an object")
        tool_type = tool.get("type")
        if not isinstance(tool_type, str) or not tool_type.strip():
            _raise_nested_error(provider_key, "tools", f"item {idx} type must be a non-empty string")
        if tool_type != "function":
            continue
        func = tool.get("function")
        if not isinstance(func, dict):
            _raise_nested_error(provider_key, "tools", f"item {idx} function must be an object")
        name = func.get("name")
        if not isinstance(name, str) or not name.strip():
            _raise_nested_error(provider_key, "tools", f"item {idx} function.name must be a non-empty string")
        params = func.get("parameters")
        if params is not None and not isinstance(params, dict):
            _raise_nested_error(provider_key, "tools", f"item {idx} function.parameters must be an object")


def _validate_response_format(provider_key: str, response_format: Any) -> None:
    if response_format is None:
        return
    if not isinstance(response_format, Mapping):
        _raise_nested_error(provider_key, "response_format", "must be an object")
    resp_type = response_format.get("type")
    if not isinstance(resp_type, str) or not resp_type.strip():
        _raise_nested_error(provider_key, "response_format", "type must be a non-empty string")
    schema = response_format.get("json_schema")
    if resp_type == "json_schema" and not isinstance(schema, Mapping):
        _raise_nested_error(provider_key, "response_format", "json_schema must be an object")
    if schema is not None and not isinstance(schema, Mapping):
        _raise_nested_error(provider_key, "response_format", "json_schema must be an object")
    if isinstance(schema, Mapping):
        inner = schema.get("schema")
        if inner is not None and not isinstance(inner, Mapping):
            _raise_nested_error(provider_key, "response_format", "json_schema.schema must be an object")


def _validate_logit_bias(provider_key: str, logit_bias: Any) -> None:
    if logit_bias is None:
        return
    if not isinstance(logit_bias, Mapping):
        _raise_nested_error(provider_key, "logit_bias", "must be an object")
    for key, value in logit_bias.items():
        if isinstance(key, bool) or not isinstance(key, (int, str)):
            _raise_nested_error(provider_key, "logit_bias", "keys must be token id strings or integers")
        if isinstance(key, str):
            try:
                int(key)
            except ValueError:
                _raise_nested_error(provider_key, "logit_bias", f"invalid token id '{key}'")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            _raise_nested_error(provider_key, "logit_bias", f"invalid bias for token '{key}'")
        if isinstance(value, float) and math.isnan(value):
            _raise_nested_error(provider_key, "logit_bias", f"invalid bias for token '{key}'")


def _validate_nested_fields(provider_key: str, payload: Mapping[str, Any]) -> None:
    _validate_tools(provider_key, payload.get("tools"))
    _validate_response_format(provider_key, payload.get("response_format"))
    _validate_logit_bias(provider_key, payload.get("logit_bias"))


def validate_payload(provider: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate payload keys against the capability registry.

    Returns a normalized copy of the payload (aliases applied).
    Raises ChatBadRequestError for blocked or unsupported keys.
    """
    provider_key = _normalize_provider(provider)
    normalized = normalize_payload(provider_key, payload)
    tool_choice_present = "tool_choice" in normalized
    tool_choice_value = normalized.get("tool_choice")
    filtered = {k: v for k, v in normalized.items() if v is not None}
    blocked = set(BLOCKED_FIELDS.get(provider_key, set()))
    if tool_choice_present and tool_choice_value is not None and "tool_choice" in blocked:
        blocked_present = ["tool_choice"]
    else:
        blocked_present = sorted(set(filtered.keys()) & blocked)
    if blocked_present:
        _record_validation_rejection(provider_key)
        raise ChatBadRequestError(
            message=f"Blocked fields for provider '{provider_key}': {', '.join(blocked_present)}",
            provider=provider_key or None,
        )
    allowed = get_allowed_fields(provider_key)
    unsupported = sorted(set(filtered.keys()) - allowed)
    if unsupported:
        _record_validation_rejection(provider_key)
        raise ChatBadRequestError(
            message=f"Unsupported fields for provider '{provider_key}': {', '.join(unsupported)}",
            provider=provider_key or None,
        )
    _validate_nested_fields(provider_key, filtered)
    tools_value = normalized.get("tools")
    has_tools = isinstance(tools_value, list) and len(tools_value) > 0
    if isinstance(tools_value, list) and len(tools_value) == 0:
        normalized["tools"] = None

    tool_choice = normalized.get("tool_choice")
    if tool_choice is not None:
        if isinstance(tool_choice, str) and tool_choice.strip().lower() == "none":
            return normalized
        if not has_tools and isinstance(tool_choice, str) and tool_choice.strip().lower() == "auto":
            normalized["tool_choice"] = "none"
            return normalized
        if not has_tools:
            _record_validation_rejection(provider_key)
            raise ChatBadRequestError(
                message="tool_choice requires tools",
                provider=provider_key or None,
            )
    return normalized


__all__ = [
    "SCHEMA_VERSION",
    "BASE_FIELDS",
    "PROVIDER_EXTENSIONS",
    "ALIASES",
    "BLOCKED_FIELDS",
    "get_allowed_fields",
    "normalize_payload",
    "validate_payload",
]
