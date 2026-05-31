from __future__ import annotations

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
}


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
        cfg_section["api_base_url"] = base_url.strip()

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
            merged_section.update(values)
        merged[section] = merged_section
    return merged
