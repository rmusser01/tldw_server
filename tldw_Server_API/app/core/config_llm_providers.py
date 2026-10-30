"""Provider names and API keys from env and config, read at call time (TASK-13950.13; TASK-13950.14 adds the default provider)."""

import os
from typing import Optional

from tldw_Server_API.app.core.config import load_and_log_configs
from tldw_Server_API.app.core.custom_openai_providers import (
    custom_openai_api_key_env_keys,
    custom_openai_provider_number,
    custom_openai_section_name,
    iter_custom_openai_provider_names,
)

_BASE_SUPPORTED_PROVIDER_NAMES_LIST: list[str] = [
    "bedrock",
    "anthropic",
    "cohere",
    "deepseek",
    "google",
    "groq",
    "qwen",
    "huggingface",
    "mistral",
    "openai",
    "openrouter",
    "novita",
    "poe",
    "together",
    "llama.cpp",
    "kobold",
    "ollama",
    "ooba",
    "tabbyapi",
    "vllm",
    "local-llm",
    "aphrodite",
    "moonshot",
    "zai",
    "mlx",
]

ALL_SUPPORTED_PROVIDER_NAMES_LIST: list[str] = [
    *_BASE_SUPPORTED_PROVIDER_NAMES_LIST,
    *iter_custom_openai_provider_names(),
]


def get_api_keys() -> dict[str, Optional[str]]:
    """
    Get API keys dynamically to support runtime changes.
    This function reloads config and environment variables each time it's called,
    ensuring that test environment changes are properly reflected.
    """
    # Reload config to get latest values
    current_config = load_and_log_configs() or {}

    def _get_dynamic_setting(
        env_vars: str | tuple[str, ...],
        section: str,
        key: str,
        default: Optional[str] = "",
    ) -> Optional[str]:
        """Resolve an API key from env aliases before config-backed settings."""
        # First check environment variable
        env_candidates = (env_vars,) if isinstance(env_vars, str) else tuple(env_vars or ())
        for env_var in env_candidates:
            env_value = os.getenv(env_var)
            if env_value is not None:
                return env_value

        custom_number = custom_openai_provider_number(key)
        if section == "api_keys" and custom_number is not None:
            custom_section = current_config.get(custom_openai_section_name(custom_number))
            if isinstance(custom_section, dict):
                api_key = custom_section.get("api_key")
                if api_key:
                    return api_key

        # Check for API key in the config dict
        if section == "api_keys":
            # Look for provider-specific API config like 'openai_api'
            provider_api_key = f"{key}_api"
            if provider_api_key in current_config:
                api_config = current_config[provider_api_key]
                if isinstance(api_config, dict):
                    api_key = api_config.get("api_key")
                    if api_key:
                        return api_key

        # Fallback to checking section directly
        config_section = current_config.get(section)
        if config_section:
            config_value = config_section.get(key) if isinstance(config_section, dict) else None
            if config_value is not None:
                return config_value

        return default

    def _provider_env_keys(provider_name: str) -> tuple[str, ...]:
        custom_number = custom_openai_provider_number(provider_name)
        if custom_number is not None:
            return custom_openai_api_key_env_keys(custom_number)
        normalized = provider_name.upper().replace('.', '_').replace('-', '_')
        if normalized.endswith("_API"):
            normalized = normalized[: -len("_API")]
        return (f"{normalized}_API_KEY",)

    return {
        name: _get_dynamic_setting(_provider_env_keys(name), "api_keys", name)
        for name in ALL_SUPPORTED_PROVIDER_NAMES_LIST
    }
