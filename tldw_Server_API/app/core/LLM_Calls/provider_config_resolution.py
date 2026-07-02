"""Runtime config resolution helpers for LLM provider catalog assembly."""

from __future__ import annotations

import os
from configparser import ConfigParser
from typing import Optional

from tldw_Server_API.app.core.custom_openai_providers import (
    custom_openai_api_key_env_keys,
    custom_openai_endpoint_env_keys,
    custom_openai_model_env_keys,
    custom_openai_provider_number,
)

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
