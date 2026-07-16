"""Runtime config resolution helpers for LLM provider catalog assembly."""

from __future__ import annotations

import os
from configparser import ConfigParser
from dataclasses import dataclass
from typing import Optional

from tldw_Server_API.app.core.config import load_settings
from tldw_Server_API.app.core.custom_openai_providers import (
    custom_openai_api_key_env_keys,
    custom_openai_endpoint_env_keys,
    custom_openai_model_env_keys,
    custom_openai_provider_number,
)
from tldw_Server_API.app.core.Security.egress import ConfiguredEndpointScope


@dataclass(frozen=True)
class TrustedProviderEndpoint:
    """One fresh server-owned endpoint paired with its exact-origin scope."""

    base_url: str
    scope: ConfiguredEndpointScope


_LOCAL_ENDPOINT_FIELDS: dict[str, tuple[str, str]] = {
    "local-llm": ("local_llm", "api_ip"),
    "llama.cpp": ("llama_api", "api_ip"),
    "kobold": ("kobold_api", "api_ip"),
    "ooba": ("ooba_api", "api_ip"),
    "tabbyapi": ("tabby_api", "api_ip"),
    "vllm": ("vllm_api", "api_ip"),
    "ollama": ("ollama_api", "api_url"),
    "aphrodite": ("aphrodite_api", "api_ip"),
}

_LOCAL_ENDPOINT_ALIASES: dict[str, str] = {
    "local": "local-llm",
    "local_llm": "local-llm",
    "llama-cpp": "llama.cpp",
    "llama_cpp": "llama.cpp",
    "llamacpp": "llama.cpp",
    "kobold-cpp": "kobold",
    "kobold_cpp": "kobold",
    "koboldcpp": "kobold",
    "oobabooga": "ooba",
    "text-generation-webui": "ooba",
    "text_generation_webui": "ooba",
    "tabby-api": "tabbyapi",
    "tabby_api": "tabbyapi",
    "tabby": "tabbyapi",
}

_LOCAL_LLM_ENDPOINT_ENV_KEYS = (
    "LOCAL_LLM_API_URL",
    "LOCAL_LLM_API_BASE",
    "LOCAL_LLM_API_IP",
    "LOCAL_LLM_BASE_URL",
)


def _normalize_configured_local_provider(provider_name: str) -> str | None:
    """Normalize configured-local aliases without classifying public adapters."""
    raw = str(provider_name or "").strip().lower()
    custom_number = custom_openai_provider_number(raw)
    if custom_number is not None:
        return "custom-openai-api" if custom_number == 1 else f"custom-openai-api-{custom_number}"
    canonical = _LOCAL_ENDPOINT_ALIASES.get(raw, raw)
    return canonical if canonical in _LOCAL_ENDPOINT_FIELDS else None


def resolve_trusted_provider_endpoint(provider_name: str) -> TrustedProviderEndpoint | None:
    """Resolve a paired endpoint/scope solely from one current server snapshot.

    Caller app configuration and fallback URLs are deliberately not accepted.
    Numbered custom OpenAI aliases resolve only their matching configured slot.
    """
    canonical = _normalize_configured_local_provider(provider_name)
    if canonical is None:
        return None

    snapshot = load_settings() or {}
    if canonical == "local-llm":
        base_url = first_env_provider_value(_LOCAL_LLM_ENDPOINT_ENV_KEYS)
        if base_url is None:
            base_url = valid_provider_config_value(
                (snapshot.get("local_llm") or {}).get("api_ip")
            )
    else:
        custom_number = custom_openai_provider_number(canonical)
        if custom_number is not None:
            base_url = first_env_provider_value(
                custom_openai_endpoint_env_keys(custom_number)
            )
            if base_url is None:
                section = "custom_openai_api" if custom_number == 1 else f"custom_openai_api_{custom_number}"
                base_url = valid_provider_config_value(
                    (snapshot.get(section) or {}).get("api_ip")
                )
        else:
            section, field = _LOCAL_ENDPOINT_FIELDS[canonical]
            base_url = valid_provider_config_value((snapshot.get(section) or {}).get(field))

    if base_url is None:
        return None
    try:
        scope = ConfiguredEndpointScope.from_url(base_url)
    except ValueError:
        return None
    return TrustedProviderEndpoint(base_url=base_url.rstrip("/"), scope=scope)

_KNOWN_API_KEY_PLACEHOLDERS = {
    "REPLACE-ME",
    "REPLACE_ME",
    "<REPLACE-ME>",
    "<REPLACE_ME>",
    "YOUR_API_KEY",
    "YOUR_API_KEY_HERE",
    "<YOUR_API_KEY>",
    "<YOUR_API_KEY_HERE>",
    "API_KEY",
    "CHANGE_ME",
    "CHANGE_ME_TO_SECURE_API_KEY",
    "CHANGEME",
}


def valid_provider_config_value(value: Optional[str]) -> Optional[str]:
    """Return a non-placeholder config value after trimming, otherwise None."""
    if not isinstance(value, str):
        return None
    trimmed = value.strip()
    if not trimmed:
        return None
    if trimmed.startswith("<") and trimmed.endswith(">"):
        return None
    return trimmed


def valid_provider_api_key(value: Optional[str]) -> Optional[str]:
    """Return a usable provider API key after applying placeholder checks."""
    trimmed = valid_provider_config_value(value)
    if not trimmed:
        return None
    if trimmed.lower().startswith("change_me"):
        return None
    if trimmed.upper() in _KNOWN_API_KEY_PLACEHOLDERS:
        return None
    return trimmed


def first_env_provider_value(env_keys: tuple[str, ...]) -> Optional[str]:
    """Return the first non-placeholder value from the requested environment keys."""
    for env_key in env_keys:
        value = valid_provider_config_value(os.getenv(env_key))
        if value:
            return value
    return None


def provider_config_value(
    config_parser: ConfigParser,
    section_name: Optional[str],
    field_name: Optional[str],
) -> Optional[str]:
    """Read and validate a provider value from config.txt-style sections."""
    if (
        section_name
        and field_name
        and config_parser.has_section(section_name)
        and config_parser.has_option(section_name, field_name)
    ):
        return valid_provider_config_value(
            config_parser.get(section_name, field_name, fallback="")
        )
    return None


def resolve_provider_endpoint_url(
    provider_name: str,
    config_parser: ConfigParser,
    section_name: Optional[str],
    endpoint_field: Optional[str],
) -> Optional[str]:
    """Resolve the endpoint URL for a provider, preferring custom OpenAI env vars."""
    custom_number = custom_openai_provider_number(provider_name)
    if custom_number is not None:
        env_endpoint = first_env_provider_value(
            custom_openai_endpoint_env_keys(custom_number)
        )
        if env_endpoint:
            return env_endpoint
    return provider_config_value(config_parser, section_name, endpoint_field)


def resolve_provider_model_value(
    provider_name: str,
    config_parser: ConfigParser,
    section_name: Optional[str],
    model_field: Optional[str],
) -> Optional[str]:
    """Resolve the configured model list for a provider."""
    custom_number = custom_openai_provider_number(provider_name)
    if custom_number is not None:
        env_model = first_env_provider_value(custom_openai_model_env_keys(custom_number))
        if env_model:
            return env_model
    return provider_config_value(config_parser, section_name, model_field)


def resolve_provider_api_key_value(
    provider_name: str,
    config_parser: ConfigParser,
    section_name: Optional[str],
    api_key_field: Optional[str],
) -> Optional[str]:
    """Resolve the API key for a provider while ignoring placeholders."""
    custom_number = custom_openai_provider_number(provider_name)
    if custom_number is not None:
        env_api_key = valid_provider_api_key(
            first_env_provider_value(custom_openai_api_key_env_keys(custom_number))
        )
        if env_api_key:
            return env_api_key
    return valid_provider_api_key(
        provider_config_value(config_parser, section_name, api_key_field)
    )


def has_custom_openai_env_configuration(provider_name: str) -> bool:
    """Return whether a custom OpenAI provider has any usable env configuration."""
    custom_number = custom_openai_provider_number(provider_name)
    if custom_number is None:
        return False
    return bool(
        first_env_provider_value(custom_openai_endpoint_env_keys(custom_number))
        or first_env_provider_value(custom_openai_model_env_keys(custom_number))
        or valid_provider_api_key(
            first_env_provider_value(custom_openai_api_key_env_keys(custom_number))
        )
    )
