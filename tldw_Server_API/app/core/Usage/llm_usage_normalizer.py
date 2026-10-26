"""
Provider-agnostic LLM usage normalization helpers.

This module normalizes provider usage
payloads into stable fields without changing billing rates, persistence schema,
or provider request behavior. It reads the billing facts a provider reports
(cache tokens, service tier, OpenRouter's own cost); compute_costs prices them.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

MAX_RAW_USAGE_METADATA_CHARS = 4096
MAX_RAW_USAGE_DEPTH = 5
MAX_RAW_USAGE_ITEMS = 50
MAX_RAW_USAGE_STRING_CHARS = 256
REDACTED_VALUE = "[redacted]"

_SECRET_KEY_FRAGMENTS = (
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "password",
    "secret",
    "bearer",
    "x-api-key",
    "x_api_key",
    "access_key",
    "private_key",
    "client_secret",
)

_PROMPT_LIKE_KEYS = {
    "headers",
    "prompt",
    "prompts",
    "message",
    "messages",
    "content",
    "contents",
    "text",
    "input",
    "inputs",
    "output",
    "outputs",
    "system",
    "user",
    "assistant",
    "tool",
    "tools",
}


@dataclass(frozen=True)
class NormalizedLLMUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cached_input_tokens: int = 0
    cache_write_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    billable_input_tokens: int = 0
    reasoning_tokens: int = 0
    choice_count: int = 0
    estimate_source: str = "missing_usage"
    raw_usage_metadata: Mapping[str, Any] = field(default_factory=dict)
    # The 1-hour share of cache_write_input_tokens (Anthropic). Not in to_dict(); compute_costs reads it.
    cache_write_1h_input_tokens: int = 0
    # Anthropic request options billed above list price (lowercased; empty when not reported).
    speed: str = ""
    inference_geo: str = ""
    # The service tier the provider reports the call ran on (lowercased; empty when not reported).
    service_tier: str = ""
    # OpenRouter's own bill for the call in USD (its usage.cost), or None.
    reported_cost_usd: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "cache_write_input_tokens": self.cache_write_input_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
            "billable_input_tokens": self.billable_input_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "choice_count": self.choice_count,
            "estimate_source": self.estimate_source,
            "raw_usage_metadata": dict(self.raw_usage_metadata),
        }


def normalize_llm_usage(
    *,
    provider: str,
    usage: Mapping[str, Any] | None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    total_tokens: int | None = None,
    choices: Sequence[Any] | None = None,
    choice_count: int | None = None,
    estimate_source: str | None = None,
) -> NormalizedLLMUsage:
    """
    Normalize provider usage payloads into stable cost/cache observability fields.

    Supported inputs include OpenAI-compatible providers, Anthropic, Gemini,
    OpenRouter passthrough payloads, and local OpenAI-compatible servers such as
    vLLM and llama.cpp. Unknown providers fall back to OpenAI-style keys.
    """
    provider_key = str(provider or "").strip().lower()
    usage_mapping = usage if isinstance(usage, Mapping) else None
    raw_usage_metadata = _sanitize_raw_usage_metadata(usage_mapping or {})

    input_tokens = _first_int(
        prompt_tokens,
        _usage_int(usage_mapping, "prompt_tokens", "input_tokens", "promptTokenCount", "prompt_token_count"),
    )
    output_tokens = _first_int(
        completion_tokens,
        _usage_int(
            usage_mapping,
            "completion_tokens",
            "output_tokens",
            "candidatesTokenCount",
            "candidates_token_count",
        ),
    )
    reasoning_tokens = _usage_int(
        usage_mapping,
        "reasoning_tokens",
        "thoughtsTokenCount",
        "thoughts_token_count",
        "completion_tokens_details.reasoning_tokens",
        "output_tokens_details.reasoning_tokens",
    )

    cache_read_input_tokens = 0
    cache_write_input_tokens = 0
    cached_input_tokens = 0
    cache_write_1h_input_tokens = 0
    speed = ""
    inference_geo = ""
    # Anthropic reports it in the usage block; OpenAI beside it (usage_with_service_tier folds it in).
    service_tier = _usage_str(usage_mapping, "service_tier")
    reported_cost_usd = _openrouter_reported_cost(usage_mapping) if "openrouter" in provider_key else None

    if "anthropic" in provider_key:
        speed = _usage_str(usage_mapping, "speed")
        inference_geo = _usage_str(usage_mapping, "inference_geo")
        cache_write_input_tokens = _usage_int(usage_mapping, "cache_creation_input_tokens")
        cache_read_input_tokens = _usage_int(usage_mapping, "cache_read_input_tokens")
        cached_input_tokens = cache_read_input_tokens
        # The 1-hour share of the writes (billed at 2x input, the rest at 1.25x); never more than the writes.
        cache_write_1h_input_tokens = min(
            _usage_int(usage_mapping, "cache_creation.ephemeral_1h_input_tokens"), cache_write_input_tokens
        )
    elif "google" in provider_key or "gemini" in provider_key:
        cache_read_input_tokens = _usage_int(usage_mapping, "cachedContentTokenCount", "cached_content_token_count")
        cached_input_tokens = cache_read_input_tokens
    else:
        cached_input_tokens = _usage_int(
            usage_mapping,
            "cached_input_tokens",
            "prompt_tokens_details.cached_tokens",
            "input_tokens_details.cached_tokens",
        )
        cache_read_input_tokens = _usage_int(
            usage_mapping,
            "cache_read_input_tokens",
            "prompt_tokens_details.cached_tokens",
            "input_tokens_details.cached_tokens",
        )
        cache_write_input_tokens = _usage_int(
            usage_mapping,
            "cache_creation_input_tokens",
            "cache_write_input_tokens",
            "prompt_tokens_details.cache_creation_tokens",
            "input_tokens_details.cache_creation_tokens",
        )
        cached_input_tokens = max(cached_input_tokens, cache_read_input_tokens)

    total = _first_int(
        total_tokens,
        _usage_int(usage_mapping, "total_tokens", "totalTokenCount", "total_token_count"),
    )
    if total <= 0:
        total = max(0, input_tokens) + max(0, output_tokens)

    normal_rate_input_tokens = max(0, input_tokens - cache_read_input_tokens - cache_write_input_tokens)
    resolved_choice_count = _normalize_choice_count(choices=choices, choice_count=choice_count)
    resolved_source = _resolve_estimate_source(
        usage=usage_mapping,
        explicit_source=estimate_source,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )

    return NormalizedLLMUsage(
        input_tokens=max(0, input_tokens),
        output_tokens=max(0, output_tokens),
        total_tokens=max(0, total),
        cached_input_tokens=max(0, cached_input_tokens),
        cache_write_input_tokens=max(0, cache_write_input_tokens),
        cache_read_input_tokens=max(0, cache_read_input_tokens),
        billable_input_tokens=normal_rate_input_tokens,
        reasoning_tokens=max(0, reasoning_tokens),
        choice_count=resolved_choice_count,
        estimate_source=resolved_source,
        raw_usage_metadata=raw_usage_metadata,
        cache_write_1h_input_tokens=cache_write_1h_input_tokens,
        speed=speed,
        inference_geo=inference_geo,
        service_tier=service_tier,
        reported_cost_usd=reported_cost_usd,
    )


def usage_with_service_tier(response: Any) -> Any:
    """``response["usage"]``, with the response's top-level ``service_tier`` added when the block lacks one.

    OpenAI reports the tier a call ran on beside the usage block; Anthropic reports it inside. The result is a copy,
    so the response a caller passes on is unchanged; without a tier to add it is the usage block itself.
    """
    if not isinstance(response, Mapping):
        return None
    usage = response.get("usage")
    tier = response.get("service_tier")
    if isinstance(usage, Mapping) and isinstance(tier, str) and tier.strip() and "service_tier" not in usage:
        return {**usage, "service_tier": tier}
    return usage


def _openrouter_reported_cost(usage: Mapping[str, Any] | None) -> float | None:
    """OpenRouter's ``usage.cost`` in USD (its credits are dollars), or None when absent or not a number.

    With BYOK, ``cost`` is only OpenRouter's fee; ``cost_details.upstream_inference_cost`` is what the upstream
    provider charged the operator's own key, so it is added.
    """
    cost = _usage_float(usage, "cost")
    if cost is None or not usage:
        return None
    if usage.get("is_byok") is True:
        details = usage.get("cost_details")
        cost += _usage_float(details if isinstance(details, Mapping) else None, "upstream_inference_cost") or 0.0
    return cost


def _usage_float(usage: Mapping[str, Any] | None, key: str) -> float | None:
    """A finite non-negative number at ``key``, or None."""
    value = usage.get(key) if usage else None
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value < 0:
        return None
    return float(value)


def _first_int(*values: Any) -> int:
    for value in values:
        parsed = _coerce_nonnegative_int(value)
        if parsed > 0:
            return parsed
    return 0


def _usage_int(usage: Mapping[str, Any] | None, *paths: str) -> int:
    if not usage:
        return 0
    for path in paths:
        value: Any = usage
        for part in path.split("."):
            if not isinstance(value, Mapping) or part not in value:
                value = None
                break
            value = value[part]
        parsed = _coerce_nonnegative_int(value)
        if parsed > 0:
            return parsed
    return 0


def _usage_str(usage: Mapping[str, Any] | None, key: str) -> str:
    """A lowercased string usage field, or empty."""
    value = usage.get(key) if usage else None
    return value.strip().lower() if isinstance(value, str) else ""


def _coerce_nonnegative_int(value: Any) -> int:
    if value is None or isinstance(value, bool):
        return 0
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, parsed)


def _normalize_choice_count(*, choices: Sequence[Any] | None, choice_count: int | None) -> int:
    explicit_count = _coerce_nonnegative_int(choice_count)
    if explicit_count > 0:
        return explicit_count
    if choices is None:
        return 0
    try:
        return max(0, len(choices))
    except TypeError:
        return 0


def _resolve_estimate_source(
    *,
    usage: Mapping[str, Any] | None,
    explicit_source: str | None,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    total_tokens: int | None,
) -> str:
    if explicit_source:
        return str(explicit_source)
    if usage:
        return "provider_usage"
    if any(value is not None for value in (prompt_tokens, completion_tokens, total_tokens)):
        return "estimated"
    return "missing_usage"


def _sanitize_raw_usage_metadata(usage: Mapping[str, Any]) -> dict[str, Any]:
    sanitized = _sanitize_value(usage, depth=0, parent_prompt_like=False)
    if not isinstance(sanitized, dict):
        return {}
    return _bound_raw_usage_metadata(sanitized)


def _sanitize_value(value: Any, *, depth: int, parent_prompt_like: bool) -> Any:
    if depth >= MAX_RAW_USAGE_DEPTH:
        return "[truncated]"
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= MAX_RAW_USAGE_ITEMS:
                output["_truncated"] = True
                break
            key_text = str(key)
            key_is_prompt_like = _is_prompt_like_key(key_text)
            if _should_redact_child_in_prompt_like_parent(key_text, item, parent_prompt_like):
                output[key_text] = REDACTED_VALUE
            elif _should_redact_key(key_text, item):
                output[key_text] = REDACTED_VALUE
            else:
                output[key_text] = _sanitize_value(
                    item,
                    depth=depth + 1,
                    parent_prompt_like=parent_prompt_like or key_is_prompt_like,
                )
        return output
    if isinstance(value, list | tuple):
        output_list = []
        for index, item in enumerate(value):
            if index >= MAX_RAW_USAGE_ITEMS:
                output_list.append("[truncated]")
                break
            output_list.append(
                _sanitize_value(
                    item,
                    depth=depth + 1,
                    parent_prompt_like=parent_prompt_like,
                )
            )
        return output_list
    if isinstance(value, str):
        if len(value) > MAX_RAW_USAGE_STRING_CHARS:
            return value[:MAX_RAW_USAGE_STRING_CHARS] + "...[truncated]"
        return value
    if isinstance(value, bool) or value is None or isinstance(value, int | float):
        return value
    return str(value)[:MAX_RAW_USAGE_STRING_CHARS]


def _should_redact_key(key: str, value: Any) -> bool:
    normalized = key.strip().lower().replace("-", "_")
    if any(fragment in normalized for fragment in _SECRET_KEY_FRAGMENTS):
        return True
    if normalized == "headers":
        return True
    if _is_prompt_like_key(normalized):
        return not _looks_like_usage_counter(value)
    return False


def _is_prompt_like_key(key: str) -> bool:
    normalized = key.strip().lower().replace("-", "_")
    if normalized in _PROMPT_LIKE_KEYS:
        return True
    prompt_fragments = ("prompt", "message", "content", "text", "system", "user", "assistant", "tool")
    return any(fragment in normalized for fragment in prompt_fragments)


def _is_usage_descriptor_key(key: str) -> bool:
    normalized = key.strip().lower().replace("-", "_")
    return normalized in {"unit", "units", "metric", "counter", "counter_type"}


def _should_redact_child_in_prompt_like_parent(key: str, value: Any, parent_prompt_like: bool) -> bool:
    if not parent_prompt_like or not isinstance(value, str):
        return False
    if _looks_like_usage_counter(value):
        return False
    return not _is_usage_descriptor_key(key)


def _looks_like_usage_counter(value: Any) -> bool:
    if value is None or isinstance(value, bool):
        return False
    if isinstance(value, int | float):
        return True
    if isinstance(value, str):
        return value.isdigit()
    if isinstance(value, Mapping):
        return any(_looks_like_usage_counter(item) for item in value.values())
    return False


def _bound_raw_usage_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    try:
        serialized = json.dumps(metadata, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return {"_unserializable": True}
    if len(serialized) <= MAX_RAW_USAGE_METADATA_CHARS:
        return metadata

    bounded: dict[str, Any] = {}
    for key, value in metadata.items():
        candidate = dict(bounded)
        candidate[key] = value
        try:
            candidate_serialized = json.dumps(candidate, sort_keys=True, default=str)
        except (TypeError, ValueError):
            continue
        if len(candidate_serialized) > MAX_RAW_USAGE_METADATA_CHARS - 32:
            bounded["_truncated"] = True
            break
        bounded = candidate
    return bounded


__all__ = ["NormalizedLLMUsage", "normalize_llm_usage", "usage_with_service_tier"]
