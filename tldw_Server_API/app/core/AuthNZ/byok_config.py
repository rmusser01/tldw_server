from __future__ import annotations

import json
from typing import Any

from tldw_Server_API.app.core.AuthNZ.user_provider_secrets import normalize_provider_name
from tldw_Server_API.app.core.custom_openai_providers import (
    custom_openai_provider_number,
    custom_openai_section_name,
)

PROVIDER_APP_CONFIG_KEYS: dict[str, str] = {
    "openai": "openai_api",
    "anthropic": "anthropic_api",
    "cohere": "cohere_api",
    "groq": "groq_api",
    "openrouter": "openrouter_api",
    "novita": "novita_api",
    "poe": "poe_api",
    "deepseek": "deepseek_api",
    "together": "together_api",
    "mistral": "mistral_api",
    "google": "google_api",
    "huggingface": "huggingface_api",
    "qwen": "qwen_api",
    "bedrock": "bedrock_api",
    "moonshot": "moonshot_api",
    "zai": "zai_api",
    "custom-openai-api": "custom_openai_api",
    "custom-openai-api-2": "custom_openai_api_2",
    "voyage": "voyage_api",
    "elevenlabs": "elevenlabs_api",
    "fish_s2": "fish_s2_api",
    "llama.cpp": "llama_api",
    "kobold": "kobold_api",
    "ooba": "ooba_api",
    "tabbyapi": "tabby_api",
    "vllm": "vllm_api",
    "local-llm": "local_llm",
    "ollama": "ollama_api",
    "aphrodite": "aphrodite_api",
    "mlx": "mlx",
}

_LOCAL_ENDPOINT_CONFIG_FIELDS = {
    "local-llm": "api_ip",
    "llama.cpp": "api_ip",
    "kobold": "api_ip",
    "ooba": "api_ip",
    "tabbyapi": "api_ip",
    "vllm": "api_ip",
    "ollama": "api_url",
    "aphrodite": "api_ip",
}

_ENDPOINT_CONFIG_ALIASES = frozenset(
    {
        "base_url",
        "api_base_url",
        "api_base",
        "api_url",
        "api_ip",
        "endpoint",
        "runtime_endpoint",
    }
)

# Environment values that participate in one atomic execution credential
# snapshot. The first non-empty alias wins, matching the adapters' legacy
# precedence. Custom OpenAI slots are generated from their canonical helpers.
PROVIDER_RUNTIME_ENV_CONFIG_KEYS: dict[str, dict[str, tuple[str, ...]]] = {
    "anthropic": {
        "api_key": ("ANTHROPIC_API_KEY",),
        "api_base_url": ("ANTHROPIC_BASE_URL",),
    },
    "bedrock": {
        "api_key": ("BEDROCK_API_KEY", "AWS_BEARER_TOKEN_BEDROCK"),
        "runtime_endpoint": ("BEDROCK_RUNTIME_ENDPOINT",),
        "api_base_url": ("BEDROCK_API_BASE_URL", "BEDROCK_OPENAI_BASE_URL"),
        "region": ("BEDROCK_REGION",),
    },
    "cohere": {"api_key": ("COHERE_API_KEY",)},
    "deepseek": {
        "api_key": ("DEEPSEEK_API_KEY",),
        "api_base_url": ("DEEPSEEK_BASE_URL",),
    },
    "elevenlabs": {"api_key": ("ELEVENLABS_API_KEY",)},
    "fish_s2": {"api_key": ("FISH_AUDIO_API_KEY", "FISH_API_KEY")},
    "google": {
        "api_key": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
        "api_base_url": ("GOOGLE_GEMINI_BASE_URL",),
    },
    "groq": {
        "api_key": ("GROQ_API_KEY",),
        "api_base_url": ("GROQ_BASE_URL",),
    },
    "huggingface": {
        "api_key": ("HUGGINGFACE_API_KEY", "HF_TOKEN"),
        "api_base_url": ("HUGGINGFACE_INFERENCE_BASE_URL",),
    },
    "mistral": {
        "api_key": ("MISTRAL_API_KEY",),
        "api_base_url": ("MISTRAL_API_BASE",),
    },
    "moonshot": {"api_key": ("MOONSHOT_API_KEY",)},
    "novita": {
        "api_key": ("NOVITA_API_KEY",),
        "api_base_url": ("NOVITA_BASE_URL", "NOVITA_API_BASE_URL"),
    },
    "openai": {
        "api_key": ("OPENAI_API_KEY",),
        "api_base_url": (
            "OPENAI_API_BASE_URL",
            "OPENAI_API_BASE",
            "OPENAI_BASE_URL",
            "MOCK_OPENAI_BASE_URL",
        ),
    },
    "openrouter": {
        "api_key": ("OPENROUTER_API_KEY",),
        "api_base_url": ("OPENROUTER_BASE_URL",),
        "site_url": ("OPENROUTER_SITE_URL",),
        "site_name": ("OPENROUTER_SITE_NAME",),
    },
    "poe": {
        "api_key": ("POE_API_KEY",),
        "api_base_url": ("POE_BASE_URL", "POE_API_BASE_URL"),
    },
    "qwen": {
        "api_key": ("QWEN_API_KEY",),
        "api_base_url": ("QWEN_BASE_URL",),
        "region": ("QWEN_REGION",),
    },
    "together": {
        "api_key": ("TOGETHER_API_KEY",),
        "api_base_url": ("TOGETHER_BASE_URL", "TOGETHER_API_BASE_URL"),
    },
    "voyage": {"api_key": ("VOYAGE_API_KEY",)},
    "zai": {"api_key": ("ZAI_API_KEY",)},
    "llama.cpp": {"api_key": ("LLAMA_API_KEY", "LLAMA_CPP_API_KEY")},
    "kobold": {"api_key": ("KOBOLD_API_KEY",)},
    "ooba": {"api_key": ("OOBA_API_KEY",)},
    "tabbyapi": {"api_key": ("TABBY_API_KEY", "TABBYAPI_API_KEY")},
    "vllm": {"api_key": ("VLLM_API_KEY",)},
    "local-llm": {
        "api_key": ("LOCAL_LLM_API_KEY",),
        "api_ip": (
            "LOCAL_LLM_API_URL",
            "LOCAL_LLM_API_BASE",
            "LOCAL_LLM_API_IP",
            "LOCAL_LLM_BASE_URL",
        ),
        "model": ("LOCAL_LLM_MODEL",),
        "temperature": ("LOCAL_LLM_TEMPERATURE",),
        "streaming": ("LOCAL_LLM_STREAMING",),
        "top_p": ("LOCAL_LLM_TOP_P",),
        "top_k": ("LOCAL_LLM_TOP_K",),
        "min_p": ("LOCAL_LLM_MIN_P",),
        "max_tokens": ("LOCAL_LLM_MAX_TOKENS",),
        "seed": ("LOCAL_LLM_SEED",),
        "stop": ("LOCAL_LLM_STOP",),
        "response_format": ("LOCAL_LLM_RESPONSE_FORMAT",),
        "n": ("LOCAL_LLM_N",),
        "presence_penalty": ("LOCAL_LLM_PRESENCE_PENALTY",),
        "frequency_penalty": ("LOCAL_LLM_FREQUENCY_PENALTY",),
        "logprobs": ("LOCAL_LLM_LOGPROBS",),
        "top_logprobs": ("LOCAL_LLM_TOP_LOGPROBS",),
        "api_timeout": ("LOCAL_LLM_API_TIMEOUT",),
        "api_retries": ("LOCAL_LLM_API_RETRIES",),
        "api_retry_delay": ("LOCAL_LLM_API_RETRY_DELAY",),
        "strict_openai_compat": ("LOCAL_LLM_STRICT_OPENAI_COMPAT",),
    },
    "ollama": {"api_key": ("OLLAMA_API_KEY",)},
    "aphrodite": {"api_key": ("APHRODITE_API_KEY",)},
}

_LOCAL_LLM_BOOLEAN_CONFIG_FIELDS = frozenset(
    {"streaming", "logprobs", "strict_openai_compat"}
)
_LOCAL_LLM_FLOAT_CONFIG_DEFAULTS: dict[str, float | None] = {
    "temperature": 0.7,
    "top_p": None,
    "min_p": None,
    "presence_penalty": None,
    "frequency_penalty": None,
}
_LOCAL_LLM_INTEGER_CONFIG_DEFAULTS: dict[str, int | None] = {
    "top_k": None,
    "max_tokens": 4096,
    "n": None,
    "top_logprobs": None,
    "api_timeout": 120,
    "api_retries": 1,
    "api_retry_delay": 1,
}
_LOCAL_LLM_JSON_CONFIG_FIELDS = frozenset({"stop", "response_format"})


def normalize_runtime_environment_config_value(
    provider: str,
    field: str,
    value: str,
) -> Any:
    """Preserve loader-compatible types in an immutable environment view."""
    if normalize_provider_name(provider) != "local-llm":
        return value
    if field in _LOCAL_LLM_BOOLEAN_CONFIG_FIELDS:
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    if field in _LOCAL_LLM_FLOAT_CONFIG_DEFAULTS:
        try:
            return float(value)
        except (TypeError, ValueError):
            return _LOCAL_LLM_FLOAT_CONFIG_DEFAULTS[field]
    if field in _LOCAL_LLM_INTEGER_CONFIG_DEFAULTS:
        try:
            return int(value)
        except (TypeError, ValueError):
            return _LOCAL_LLM_INTEGER_CONFIG_DEFAULTS[field]
    if field in _LOCAL_LLM_JSON_CONFIG_FIELDS:
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value
    return value


class _RuntimeBaseUrlOverrideProvenance:
    """Opaque server provenance for a validated runtime credential endpoint."""

    __slots__ = ()

    def __copy__(self) -> _RuntimeBaseUrlOverrideProvenance:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> _RuntimeBaseUrlOverrideProvenance:
        memo[id(self)] = self
        return self

    def __repr__(self) -> str:
        return "<runtime-base-url-override>"


_RUNTIME_BASE_URL_OVERRIDE_PROVENANCE = _RuntimeBaseUrlOverrideProvenance()


def runtime_base_url_override_provenance() -> object:
    """Return the non-JSON server marker for a validated credential base URL."""
    return _RUNTIME_BASE_URL_OVERRIDE_PROVENANCE


def is_runtime_base_url_override(value: object) -> bool:
    """Return whether a value carries authentic server runtime provenance."""
    return value is _RUNTIME_BASE_URL_OVERRIDE_PROVENANCE


def build_app_config_overrides(
    provider: str,
    credential_fields: dict[str, Any] | None,
) -> dict[str, Any]:
    if not credential_fields:
        return {}

    provider_norm = normalize_provider_name(provider)
    section = PROVIDER_APP_CONFIG_KEYS.get(provider_norm)
    if not section:
        custom_number = custom_openai_provider_number(provider_norm)
        if custom_number is not None:
            section = custom_openai_section_name(custom_number)
    if not section:
        return {}

    cfg_section: dict[str, Any] = {}
    base_url = credential_fields.get("base_url")
    if isinstance(base_url, str) and base_url.strip():
        endpoint_field = _LOCAL_ENDPOINT_CONFIG_FIELDS.get(
            provider_norm,
            "api_base_url",
        )
        cfg_section[endpoint_field] = base_url.strip()

    if "org_id" in credential_fields and credential_fields.get("org_id") is not None:
        cfg_section["org_id"] = credential_fields.get("org_id")

    if "project_id" in credential_fields and credential_fields.get("project_id") is not None:
        cfg_section["project_id"] = credential_fields.get("project_id")

    return {section: cfg_section} if cfg_section else {}


def merge_app_config_overrides(
    base_config: dict[str, Any] | None,
    provider: str,
    credential_fields: dict[str, Any] | None,
) -> dict[str, Any]:
    overrides = build_app_config_overrides(provider, credential_fields)
    if not overrides:
        return dict(base_config or {})

    merged: dict[str, Any] = dict(base_config or {})
    for section, values in overrides.items():
        existing = merged.get(section)
        merged_section = dict(existing or {}) if isinstance(existing, dict) else {}
        if isinstance(values, dict):
            if any(field in values for field in _ENDPOINT_CONFIG_ALIASES):
                for field in _ENDPOINT_CONFIG_ALIASES:
                    merged_section.pop(field, None)
            merged_section.update(values)
        merged[section] = merged_section
    return merged
