"""Shared catalog for Chatterbox TTS model-family selection."""

from enum import Enum
from typing import Optional


class ChatterboxModelFamily(str, Enum):
    """Runtime families exposed by the upstream Chatterbox package."""

    STANDARD = "standard"
    MULTILINGUAL = "multilingual"
    TURBO = "turbo"


CHATTERBOX_PROVIDER_KEY = "chatterbox"

CHATTERBOX_LANGUAGE_CODES: set[str] = {
    "ar",
    "da",
    "de",
    "el",
    "en",
    "es",
    "fi",
    "fr",
    "he",
    "hi",
    "it",
    "ja",
    "ko",
    "ms",
    "nl",
    "no",
    "pl",
    "pt",
    "ru",
    "sv",
    "sw",
    "tr",
    "zh",
}

CHATTERBOX_MODEL_FAMILY_ALIASES: dict[str, ChatterboxModelFamily] = {
    "chatterbox": ChatterboxModelFamily.STANDARD,
    "chatterbox-original": ChatterboxModelFamily.STANDARD,
    "chatterbox-emotion": ChatterboxModelFamily.STANDARD,
    "chatterbox-tts": ChatterboxModelFamily.STANDARD,
    "original": ChatterboxModelFamily.STANDARD,
    "standard": ChatterboxModelFamily.STANDARD,
    "resembleai/chatterbox": ChatterboxModelFamily.STANDARD,
    "resemble-ai/chatterbox": ChatterboxModelFamily.STANDARD,
    "chatterbox-multilingual": ChatterboxModelFamily.MULTILINGUAL,
    "chatterbox-multi": ChatterboxModelFamily.MULTILINGUAL,
    "multilingual": ChatterboxModelFamily.MULTILINGUAL,
    "multi": ChatterboxModelFamily.MULTILINGUAL,
    "resembleai/chatterbox-multilingual": ChatterboxModelFamily.MULTILINGUAL,
    "resemble-ai/chatterbox-multilingual": ChatterboxModelFamily.MULTILINGUAL,
    "chatterbox-turbo": ChatterboxModelFamily.TURBO,
    "turbo": ChatterboxModelFamily.TURBO,
    "resembleai/chatterbox-turbo": ChatterboxModelFamily.TURBO,
    "resemble-ai/chatterbox-turbo": ChatterboxModelFamily.TURBO,
}

CHATTERBOX_MODEL_PROVIDER_ALIASES: dict[str, str] = {
    alias: CHATTERBOX_PROVIDER_KEY for alias in CHATTERBOX_MODEL_FAMILY_ALIASES
}


def normalize_chatterbox_model_id(value: object) -> Optional[str]:
    """Normalize model-family aliases while preserving repository separators."""
    if value is None:
        return None
    normalized = str(value).strip().casefold().replace("_", "-")
    return normalized or None


def resolve_chatterbox_model_family(
    model: object = None,
    *,
    language: Optional[str] = None,
    config_variant: object = None,
    use_multilingual: bool = False,
) -> ChatterboxModelFamily:
    """Resolve the Chatterbox runtime family from request and config hints."""
    model_key = normalize_chatterbox_model_id(model)
    if model_key in CHATTERBOX_MODEL_FAMILY_ALIASES:
        explicit_family = CHATTERBOX_MODEL_FAMILY_ALIASES[model_key]
        if explicit_family is not ChatterboxModelFamily.STANDARD:
            return explicit_family

    variant_key = normalize_chatterbox_model_id(config_variant)
    if variant_key in CHATTERBOX_MODEL_FAMILY_ALIASES:
        return CHATTERBOX_MODEL_FAMILY_ALIASES[variant_key]

    language_key = (language or "").strip().casefold()
    if use_multilingual and language_key and language_key != "en":
        return ChatterboxModelFamily.MULTILINGUAL

    return ChatterboxModelFamily.STANDARD
