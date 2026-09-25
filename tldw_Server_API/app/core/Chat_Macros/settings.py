"""Repository-backed settings helpers for chat macros."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from .output_profiles import DEFAULT_OUTPUT_PROFILE, normalize_output_profile, profile_to_dict


def default_settings() -> dict[str, Any]:
    """Return a fresh copy of the v1 chat macro settings defaults."""
    return {
        "disabled_builtins": [],
        "user_macro_enabled": {},
        "output_profiles": {
            "default": profile_to_dict(DEFAULT_OUTPUT_PROFILE),
        },
    }


def normalize_settings(raw: Mapping[str, Any] | None, *, from_storage: bool = False) -> dict[str, Any]:
    """Normalize settings, repairing legacy empty profiles only on storage reads."""
    raw = raw or {}
    settings = deepcopy(dict(raw))
    settings.update(default_settings())

    disabled = raw.get("disabled_builtins", [])
    if isinstance(disabled, list):
        settings["disabled_builtins"] = sorted({str(name) for name in disabled})

    user_enabled = raw.get("user_macro_enabled", {})
    if isinstance(user_enabled, Mapping):
        settings["user_macro_enabled"] = {
            str(name): enabled
            for name, enabled in user_enabled.items()
            if isinstance(enabled, bool)
        }

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
                if from_storage:
                    # Older versions accepted empty sections and whitespace-only
                    # headings. Keep those records readable so users can edit them.
                    profile = dict(profile)
                    if profile.get("sections") == []:
                        profile.pop("sections")
                    titles = profile.get("section_titles")
                    if isinstance(titles, Mapping):
                        profile["section_titles"] = {
                            section: title
                            for section, title in titles.items()
                            if not (isinstance(title, str) and title and not title.strip())
                        }
                settings["output_profiles"][str(name)] = profile_to_dict(
                    normalize_output_profile(str(name), profile)
                )
    return settings
