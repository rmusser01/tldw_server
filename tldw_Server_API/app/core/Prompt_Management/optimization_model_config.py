"""Canonical, secret-free configuration contracts for Prompt Studio work."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

from tldw_Server_API.app.core.LLM_Calls.provider_identity import (
    canonical_provider_name,
)

_TOP_LEVEL_PARAMETER_KEYS = (
    "temperature",
    "max_tokens",
    "timeout_seconds",
    "top_p",
    "top_k",
    "min_p",
    "stop",
    "response_format",
    "tools",
    "tool_choice",
    "seed",
    "logit_bias",
    "logprobs",
    "top_logprobs",
    "presence_penalty",
    "frequency_penalty",
    "n",
)
_DROP = object()
MODEL_CONFIG_ALIASES = (
    "model_config",
    "model_configuration",
    "llm_model_config",
)
MAX_DURABLE_OPTIMIZATION_CONFIG_BYTES = 32 * 1024
MAX_OPTIMIZATION_ITERATIONS = 500
NATIVE_DURABLE_OPTIMIZATION_STRATEGIES = frozenset(
    {"mipro", "bootstrap", "iterative", "mcts"}
)
LEGACY_OPTIMIZATION_STRATEGIES = frozenset(
    {
        "hill_climbing",
        "random_search",
        "grid_search",
        "bayesian",
        "beam_search",
        "greedy",
        "simulated_annealing",
        "genetic",
        "hyperparameter",
    }
)
OPTIMIZATION_STRATEGY_ALIASES = {
    "hill_climb": "hill_climbing",
    "anneal": "simulated_annealing",
    "hyperparam": "hyperparameter",
    "hparam": "hyperparameter",
}
ACCEPTED_OPTIMIZATION_STRATEGIES = NATIVE_DURABLE_OPTIMIZATION_STRATEGIES
_CONVENTIONAL_SECRET_KEYS = frozenset(
    {
        "authorization",
        "auth",
        "password",
        "secret",
        "clientsecret",
        "accesstoken",
        "refreshtoken",
        "token",
        "cookie",
        "jwt",
    }
)


def _is_sensitive_key(key: object) -> bool:
    compact = re.sub(r"[^a-z0-9]", "", str(key).casefold())
    return (
        compact in _CONVENTIONAL_SECRET_KEYS
        or any(
            marker in compact
            for marker in (
                "apikey",
                "clientsecret",
                "accesstoken",
                "refreshtoken",
                "secretaccesskey",
                "sessiontoken",
            )
        )
        or compact.endswith(
            (
                "authorization",
                "password",
                "secret",
                "secretkey",
                "token",
                "cookie",
                "jwt",
                "auth",
            )
        )
        or "appconfig" in compact
        or compact in {"authsource", "authuser", "trustedbaseurloverride"}
        or "credential" in compact
        or ("runtime" in compact and "handle" in compact)
        or compact.endswith("accesskeyid")
    )


def _validate_serialized_size(value: Mapping[str, Any], *, label: str) -> None:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    if len(encoded) > MAX_DURABLE_OPTIMIZATION_CONFIG_BYTES:
        raise ValueError(f"{label} is too large")


def _clean_durable_value(value: Any, *, reject_sensitive: bool) -> Any:
    if isinstance(value, Mapping):
        cleaned: dict[str, Any] = {}
        removed = False
        for key, nested in value.items():
            if not isinstance(key, str):
                if reject_sensitive:
                    raise ValueError("Optimization configuration keys must be strings")
                removed = True
                continue
            if _is_sensitive_key(key):
                if reject_sensitive:
                    raise ValueError(
                        f"Server-managed credential field is not allowed: {key}"
                    )
                removed = True
                continue
            cleaned_nested = _clean_durable_value(
                nested,
                reject_sensitive=reject_sensitive,
            )
            if cleaned_nested is not _DROP:
                cleaned[key] = cleaned_nested
            else:
                removed = True
        return _DROP if removed and value and not cleaned else cleaned
    if isinstance(value, (list, tuple)):
        cleaned_items = [
            _clean_durable_value(item, reject_sensitive=reject_sensitive)
            for item in value
        ]
        cleaned_list = [item for item in cleaned_items if item is not _DROP]
        return _DROP if value and not cleaned_list else cleaned_list
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if reject_sensitive:
        raise ValueError("Optimization configuration must contain JSON values only")
    return _DROP


def strip_sensitive_optimization_config(value: Any) -> dict[str, Any]:
    """Return a JSON-only optimization config with credential internals removed."""
    return strip_sensitive_durable_mapping(value)


def strip_sensitive_durable_mapping(value: Any) -> dict[str, Any]:
    """Return a JSON-only durable mapping with recursive secret fields removed."""
    cleaned = _clean_durable_value(value, reject_sensitive=False)
    return cleaned if isinstance(cleaned, dict) else {}


def validate_secret_free_optimization_config(value: Any) -> dict[str, Any]:
    """Validate a durable config without changing its safe legacy shape."""
    if not isinstance(value, Mapping):
        raise ValueError("Optimization configuration must be an object")
    cleaned = _clean_durable_value(value, reject_sensitive=True)
    if not isinstance(cleaned, dict):
        raise ValueError("Optimization configuration must be an object")
    return cleaned


def normalize_optimization_strategy(value: Any) -> str:
    """Normalize one accepted strategy name and reject unknown strategies."""

    if not isinstance(value, str):
        raise ValueError("Optimization strategy must be a string")
    normalized = value.strip().lower()
    normalized = OPTIMIZATION_STRATEGY_ALIASES.get(normalized, normalized)
    if normalized not in ACCEPTED_OPTIMIZATION_STRATEGIES:
        raise ValueError(f"Unsupported optimization strategy: {value}")
    return normalized


def optimization_execution_strategy(value: Any) -> str:
    """Return the provider-bound engine used for an accepted strategy."""

    return normalize_optimization_strategy(value)


def reconcile_optimization_strategy(
    *values: Any,
    default: str = "mipro",
) -> str:
    """Return one canonical strategy or fail closed when sources disagree."""

    candidates = [
        normalize_optimization_strategy(value)
        for value in values
        if value is not None
    ]
    if not candidates:
        return normalize_optimization_strategy(default)
    if len(set(candidates)) != 1:
        raise ValueError("Optimization strategy mismatch")
    return candidates[0]


def normalize_optimization_model_config(
    value: Any,
    *,
    allow_default: bool,
    reject_sensitive: bool,
) -> dict[str, Any]:
    """Normalize model aliases to provider/model/parameters only."""
    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise ValueError("Model configuration must be an object")

    cleaned = _clean_durable_value(value, reject_sensitive=reject_sensitive)
    if not isinstance(cleaned, dict):
        cleaned = {}

    provider_values: list[str] = []
    for alias in ("provider", "api_name"):
        if alias not in cleaned:
            continue
        value_for_alias = cleaned[alias]
        if not isinstance(value_for_alias, str):
            raise ValueError("Model configuration provider must be a string")
        provider = canonical_provider_name(value_for_alias.strip())
        if not provider:
            raise ValueError("Model configuration requires a provider")
        provider_values.append(provider)
    if len(set(provider_values)) > 1:
        raise ValueError("Model configuration provider aliases conflict")

    model_values: list[str] = []
    for alias in ("model", "model_name"):
        if alias not in cleaned:
            continue
        value_for_alias = cleaned[alias]
        if not isinstance(value_for_alias, str):
            raise ValueError("Model configuration model must be a string")
        model = value_for_alias.strip()
        if not model:
            raise ValueError("Model configuration requires a model")
        model_values.append(model)
    if len(set(model_values)) > 1:
        raise ValueError("Model configuration model aliases conflict")

    provider = provider_values[0] if provider_values else ""
    model = model_values[0] if model_values else ""
    if not provider and not model and allow_default:
        provider = "openai"
        model = "gpt-3.5-turbo"
    if not provider:
        raise ValueError("Model configuration requires a provider")
    if not model:
        if provider == "openai" and allow_default:
            model = "gpt-3.5-turbo"
        else:
            raise ValueError("Model configuration requires a model")

    raw_parameters = cleaned.get("parameters") or {}
    if not isinstance(raw_parameters, Mapping):
        raise ValueError("Model configuration parameters must be an object")
    parameters = dict(raw_parameters)
    for key in _TOP_LEVEL_PARAMETER_KEYS:
        if key in cleaned and key not in parameters:
            parameters[key] = cleaned[key]

    normalized = {
        "provider": provider,
        "model": model,
        "parameters": parameters,
    }
    _validate_serialized_size(normalized, label="Model configuration")
    return normalized


def reconcile_optimization_model_config_aliases(
    value: Mapping[str, Any],
    *,
    aliases: tuple[str, ...] = MODEL_CONFIG_ALIASES,
    allow_default_when_missing: bool,
    reject_sensitive: bool,
) -> dict[str, Any]:
    """Return one canonical model config or reject conflicting aliases."""

    supplied = [alias for alias in aliases if alias in value]
    if not supplied:
        return normalize_optimization_model_config(
            None,
            allow_default=allow_default_when_missing,
            reject_sensitive=reject_sensitive,
        )

    normalized = [
        normalize_optimization_model_config(
            value[alias],
            allow_default=False,
            reject_sensitive=reject_sensitive,
        )
        for alias in supplied
    ]
    if any(candidate != normalized[0] for candidate in normalized[1:]):
        raise ValueError("Model configuration aliases conflict")
    return normalized[0]


def normalize_durable_optimization_config(
    value: Any,
    *,
    reject_sensitive: bool,
) -> dict[str, Any]:
    """Normalize one complete durable optimization configuration."""
    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise ValueError("Optimization configuration must be an object")

    model_supplied = any(alias in value for alias in MODEL_CONFIG_ALIASES)
    cleaned = _clean_durable_value(value, reject_sensitive=reject_sensitive)
    if not isinstance(cleaned, dict):
        cleaned = {}
    cleaned["model_config"] = reconcile_optimization_model_config_aliases(
        cleaned,
        allow_default_when_missing=not model_supplied,
        reject_sensitive=reject_sensitive,
    )
    for alias in MODEL_CONFIG_ALIASES[1:]:
        cleaned.pop(alias, None)

    max_iterations = cleaned.get("max_iterations")
    if max_iterations is not None:
        if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
            raise ValueError("max_iterations must be an integer")
        if not 1 <= max_iterations <= MAX_OPTIMIZATION_ITERATIONS:
            raise ValueError(
                f"max_iterations must be between 1 and {MAX_OPTIMIZATION_ITERATIONS}"
            )

    _validate_serialized_size(cleaned, label="Optimization configuration")
    return cleaned


def runtime_model_config(
    durable_config: Mapping[str, Any],
    *,
    api_key: str | None,
    app_config: dict[str, Any] | None,
    credentials_resolved: bool,
) -> dict[str, Any]:
    """Overlay one resolved credential snapshot without mutating durable state."""
    return {
        "provider": durable_config["provider"],
        "model": durable_config["model"],
        "parameters": dict(durable_config.get("parameters") or {}),
        "api_key": api_key,
        "app_config": app_config or {},
        "credentials_resolved": credentials_resolved,
    }


__all__ = [
    "ACCEPTED_OPTIMIZATION_STRATEGIES",
    "LEGACY_OPTIMIZATION_STRATEGIES",
    "MAX_DURABLE_OPTIMIZATION_CONFIG_BYTES",
    "MAX_OPTIMIZATION_ITERATIONS",
    "MODEL_CONFIG_ALIASES",
    "NATIVE_DURABLE_OPTIMIZATION_STRATEGIES",
    "OPTIMIZATION_STRATEGY_ALIASES",
    "normalize_durable_optimization_config",
    "normalize_optimization_model_config",
    "normalize_optimization_strategy",
    "optimization_execution_strategy",
    "reconcile_optimization_model_config_aliases",
    "reconcile_optimization_strategy",
    "runtime_model_config",
    "strip_sensitive_durable_mapping",
    "strip_sensitive_optimization_config",
    "validate_secret_free_optimization_config",
]
