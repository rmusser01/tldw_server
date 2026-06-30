"""Naming helpers for custom OpenAI-compatible provider slots.

The first slot keeps the historical provider name, env vars, and config keys.
Slots 2 through 99 use numbered names and only resolve their own endpoint
override env vars so one configured external endpoint cannot shadow another.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

CUSTOM_OPENAI_PROVIDER_MIN = 1
CUSTOM_OPENAI_PROVIDER_MAX = 99


def _validate_custom_openai_number(number: int) -> int:
    """Return a supported custom provider number or raise a descriptive error."""
    if not isinstance(number, int):
        raise TypeError("Custom OpenAI provider number must be an integer")
    if number < CUSTOM_OPENAI_PROVIDER_MIN or number > CUSTOM_OPENAI_PROVIDER_MAX:
        raise ValueError(
            f"Custom OpenAI provider number must be between "
            f"{CUSTOM_OPENAI_PROVIDER_MIN} and {CUSTOM_OPENAI_PROVIDER_MAX}"
        )
    return number


def iter_custom_openai_provider_numbers(
    *,
    start: int = CUSTOM_OPENAI_PROVIDER_MIN,
    stop: int = CUSTOM_OPENAI_PROVIDER_MAX,
) -> Iterator[int]:
    """Yield supported custom OpenAI provider numbers, inclusive."""
    start = max(CUSTOM_OPENAI_PROVIDER_MIN, int(start))
    stop = min(CUSTOM_OPENAI_PROVIDER_MAX, int(stop))
    yield from range(start, stop + 1)


def custom_openai_provider_name(number: int) -> str:
    """Return the canonical provider id for a custom OpenAI-compatible slot."""
    number = _validate_custom_openai_number(number)
    if number == 1:
        return "custom-openai-api"
    return f"custom-openai-api-{number}"


def iter_custom_openai_provider_names(
    *,
    start: int = CUSTOM_OPENAI_PROVIDER_MIN,
    stop: int = CUSTOM_OPENAI_PROVIDER_MAX,
) -> Iterator[str]:
    """Yield canonical custom OpenAI provider ids for the requested range."""
    for number in iter_custom_openai_provider_numbers(start=start, stop=stop):
        yield custom_openai_provider_name(number)


def custom_openai_section_name(number: int) -> str:
    """Return the `app_config` section name used by a custom provider slot."""
    number = _validate_custom_openai_number(number)
    if number == 1:
        return "custom_openai_api"
    return f"custom_openai_api_{number}"


def custom_openai_aliases(number: int) -> tuple[str, ...]:
    """Return accepted aliases that normalize to the canonical provider id."""
    number = _validate_custom_openai_number(number)
    if number == 1:
        return (
            "custom_openai_api",
            "custom-openai",
            "openai-compatible",
            "customopenai",
        )
    return (
        f"custom_openai_api_{number}",
        f"custom_openai{number}_api",
        f"custom-openai-{number}",
        f"openai-compatible-{number}",
        f"customopenai{number}",
    )


def custom_openai_provider_number(provider: str | None) -> int | None:
    """Resolve a provider id, alias, or config section name to its slot number."""
    raw = str(provider or "").strip().lower()
    if not raw:
        return None

    compact = raw.replace(" ", "")
    normalized = compact.replace("_", "-")
    if normalized in {
        "custom-openai-api",
        "custom-openai",
        "openai-compatible",
        "customopenai",
    }:
        return 1

    section_match = re.fullmatch(r"custom_openai_api_(\d{1,2})", compact)
    if section_match:
        number = int(section_match.group(1))
        return number if CUSTOM_OPENAI_PROVIDER_MIN <= number <= CUSTOM_OPENAI_PROVIDER_MAX else None

    legacy_section_match = re.fullmatch(r"custom_openai(\d{1,2})_api", compact)
    if legacy_section_match:
        number = int(legacy_section_match.group(1))
        return number if CUSTOM_OPENAI_PROVIDER_MIN <= number <= CUSTOM_OPENAI_PROVIDER_MAX else None

    for pattern in (
        r"custom-openai-api-(\d{1,2})",
        r"custom-openai-(\d{1,2})",
        r"openai-compatible-(\d{1,2})",
        r"customopenai(\d{1,2})",
    ):
        match = re.fullmatch(pattern, normalized)
        if match:
            number = int(match.group(1))
            return number if CUSTOM_OPENAI_PROVIDER_MIN <= number <= CUSTOM_OPENAI_PROVIDER_MAX else None

    return None


def custom_openai_endpoint_env_keys(number: int) -> tuple[str, ...]:
    """Return endpoint base URL env vars for one custom provider slot."""
    number = _validate_custom_openai_number(number)
    if number == 1:
        return (
            "CUSTOM_OPENAI_API_IP",
            "CUSTOM_OPENAI_API_BASE",
            "CUSTOM_OPENAI_API_URL",
            "CUSTOM_OPENAI_API_BASE_URL",
            "CUSTOM_OPENAI_BASE_URL",
            "CUSTOM_OPENAI_API_IP_1",
        )
    if number == 2:
        return (
            "CUSTOM_OPENAI2_API_IP",
            "CUSTOM_OPENAI2_API_BASE",
            "CUSTOM_OPENAI2_API_URL",
            "CUSTOM_OPENAI2_API_BASE_URL",
            "CUSTOM_OPENAI2_BASE_URL",
            "CUSTOM_OPENAI_API_2_IP",
            "CUSTOM_OPENAI_API_2_BASE",
            "CUSTOM_OPENAI_API_2_URL",
            "CUSTOM_OPENAI_API_2_BASE_URL",
            "CUSTOM_OPENAI_API_IP_2",
            "CUSTOM_OPENAI_API_BASE_2",
            "CUSTOM_OPENAI_API_URL_2",
            "CUSTOM_OPENAI_API_BASE_URL_2",
        )
    return (
        f"CUSTOM_OPENAI_API_IP_{number}",
        f"CUSTOM_OPENAI_API_BASE_{number}",
        f"CUSTOM_OPENAI_API_URL_{number}",
        f"CUSTOM_OPENAI_API_BASE_URL_{number}",
        f"CUSTOM_OPENAI{number}_API_IP",
        f"CUSTOM_OPENAI{number}_API_BASE",
        f"CUSTOM_OPENAI{number}_API_URL",
        f"CUSTOM_OPENAI{number}_API_BASE_URL",
        f"CUSTOM_OPENAI_API_{number}_IP",
        f"CUSTOM_OPENAI_API_{number}_BASE",
        f"CUSTOM_OPENAI_API_{number}_URL",
        f"CUSTOM_OPENAI_API_{number}_BASE_URL",
    )


def custom_openai_api_key_env_keys(number: int) -> tuple[str, ...]:
    """Return API-key env vars for one custom provider slot."""
    number = _validate_custom_openai_number(number)
    if number == 1:
        return (
            "CUSTOM_OPENAI_API_KEY",
            "CUSTOM_OPENAI_API_KEY_1",
            "CUSTOM_OPENAI1_API_KEY",
            "CUSTOM_OPENAI_API_1_API_KEY",
        )
    if number == 2:
        return (
            "CUSTOM_OPENAI2_API_KEY",
            "CUSTOM_OPENAI_API_KEY_2",
            "CUSTOM_OPENAI_API_2_API_KEY",
        )
    return (
        f"CUSTOM_OPENAI_API_KEY_{number}",
        f"CUSTOM_OPENAI{number}_API_KEY",
        f"CUSTOM_OPENAI_API_{number}_API_KEY",
    )


def custom_openai_model_env_keys(number: int) -> tuple[str, ...]:
    """Return default-model env vars for one custom provider slot."""
    number = _validate_custom_openai_number(number)
    if number == 1:
        return (
            "CUSTOM_OPENAI_API_MODEL",
            "CUSTOM_OPENAI_API_MODEL_1",
            "CUSTOM_OPENAI1_API_MODEL",
            "CUSTOM_OPENAI_API_1_MODEL",
        )
    if number == 2:
        return (
            "CUSTOM_OPENAI2_API_MODEL",
            "CUSTOM_OPENAI_API_MODEL_2",
            "CUSTOM_OPENAI_API_2_MODEL",
        )
    return (
        f"CUSTOM_OPENAI_API_MODEL_{number}",
        f"CUSTOM_OPENAI{number}_API_MODEL",
        f"CUSTOM_OPENAI_API_{number}_MODEL",
    )


def custom_openai_config_option_names(number: int, suffix: str) -> tuple[str, ...]:
    """Return config.txt option names for a slot-specific API setting suffix."""
    number = _validate_custom_openai_number(number)
    normalized_suffix = str(suffix).strip().lower()
    if not normalized_suffix:
        raise ValueError("Custom OpenAI config suffix must be non-empty")

    if number == 1:
        return (f"custom_openai_api_{normalized_suffix}",)
    if number == 2:
        return (
            f"custom_openai2_api_{normalized_suffix}",
            f"custom_openai_api_2_{normalized_suffix}",
        )
    return (
        f"custom_openai{number}_api_{normalized_suffix}",
        f"custom_openai_api_{number}_{normalized_suffix}",
    )
