"""Canonical provider identities shared by adapters and credential storage."""

from __future__ import annotations

from tldw_Server_API.app.core.custom_openai_providers import (
    custom_openai_aliases,
    custom_openai_provider_name,
    iter_custom_openai_provider_numbers,
)


PROVIDER_ALIASES: dict[str, tuple[str, ...]] = {
    "openai": ("oai",),
    "bedrock": ("aws-bedrock", "amazon-bedrock"),
    "custom-openai-api": (
        "custom_openai_api",
        "custom-openai",
        "openai-compatible",
        "customopenai",
    ),
    "custom-openai-api-2": (
        "custom_openai_api_2",
        "custom-openai-2",
        "openai-compatible-2",
        "customopenai2",
    ),
    "novita": ("novita-ai",),
    "poe": ("poe-api",),
    "together": ("together-ai", "togetherai"),
    "llama.cpp": ("llama-cpp", "llama_cpp", "llamacpp"),
    "kobold": ("kobold-cpp", "kobold_cpp", "koboldcpp"),
    "ooba": ("oobabooga", "text-generation-webui", "text_generation_webui"),
    "tabbyapi": ("tabby-api", "tabby_api", "tabby"),
    "local-llm": ("local_llm",),
    "zai": ("z-ai", "z.ai"),
}
PROVIDER_ALIASES.update(
    {
        custom_openai_provider_name(number): custom_openai_aliases(number)
        for number in iter_custom_openai_provider_numbers(start=3)
    }
)


def _normalize_provider_spelling(provider: str) -> str:
    return str(provider or "").strip().lower()


def _normalize_provider_token(provider: str) -> str:
    return _normalize_provider_spelling(provider).replace("_", "-")


_ALIAS_TO_CANONICAL = {
    _normalize_provider_token(alias): canonical for canonical, aliases in PROVIDER_ALIASES.items() for alias in aliases
}


def canonical_provider_name(provider: str) -> str:
    """Return the adapter-owned canonical identity for a provider or alias."""
    spelling = _normalize_provider_spelling(provider)
    return _ALIAS_TO_CANONICAL.get(_normalize_provider_token(spelling), spelling)


def _registered_storage_spellings(provider: str) -> tuple[str, ...]:
    """Return every hyphen/underscore spelling accepted by the adapter registry."""
    literal = _normalize_provider_spelling(provider)
    parts = _normalize_provider_token(literal).split("-")
    variants = [parts[0]]
    for part in parts[1:]:
        variants = [
            f"{prefix}{separator}{part}"
            for prefix in variants
            for separator in ("-", "_")
        ]
    return tuple(dict.fromkeys((literal, *variants)))


def provider_lookup_names(provider: str) -> tuple[str, ...]:
    """Return canonical storage name first, followed by deterministic legacy aliases."""
    canonical = canonical_provider_name(provider)
    aliases = PROVIDER_ALIASES.get(canonical)
    if aliases is None:
        return (canonical,)
    names: list[str] = []
    for registered_name in (canonical, *aliases):
        names.extend(_registered_storage_spellings(registered_name))
    return tuple(dict.fromkeys(names))
