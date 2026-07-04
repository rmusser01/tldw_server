"""Repository-backed settings helpers for chat macros."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .output_profiles import DEFAULT_OUTPUT_PROFILE, normalize_output_profile, profile_to_dict


def default_settings() -> dict[str, Any]:
    return {
        "disabled_builtins": [],
        "output_profiles": {
            "default": profile_to_dict(DEFAULT_OUTPUT_PROFILE),
        },
    }


def normalize_settings(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    settings = default_settings()
    raw = raw or {}

    disabled = raw.get("disabled_builtins", [])
    if isinstance(disabled, list):
        settings["disabled_builtins"] = sorted({str(name) for name in disabled})

    raw_profiles = raw.get("output_profiles", {})
    if isinstance(raw_profiles, list):
        raw_profiles = {
            str(item.get("name")): item
            for item in raw_profiles
            if isinstance(item, Mapping) and item.get("name")
        }
    if isinstance(raw_profiles, Mapping):
        for name, profile in raw_profiles.items():
            if isinstance(profile, Mapping):
                settings["output_profiles"][str(name)] = profile_to_dict(
                    normalize_output_profile(str(name), profile)
                )
    return settings
