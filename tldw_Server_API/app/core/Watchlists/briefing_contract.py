"""Canonical Watchlists briefing preferences and legacy compatibility."""

from __future__ import annotations

import copy
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

BRIEFING_PIPELINE_KEY = "briefing_pipeline"
BRIEFING_PIPELINE_VERSION = 1
PROGRAM_FORMATS = {
    "concise_briefing",
    "solo_update",
    "host_discussion",
    "sportscast",
    "culture_roundtable",
    "custom",
}

_LEGACY_BRIEFING_KEYS = {
    "auto_output",
    "template",
    "template_name",
    "deliveries",
    "delivery_config",
    "generate_audio",
    "target_audio_minutes",
    "audio_language",
    "audio_provider",
    "audio_model",
    "audio_voice",
    "tts_provider",
    "tts_model",
    "tts_voice",
    "audio_speed",
    "audio_cast",
    "voice_map",
    "llm_provider",
    "llm_model",
    "persona_summarize",
    "persona_id",
    "persona_provider",
    "persona_model",
    "background_audio_uri",
    "background_volume",
    "background_delay_ms",
    "background_fade_seconds",
}


@dataclass(frozen=True)
class NormalizedBriefingContract:
    output_prefs: dict[str, Any]
    contract: dict[str, Any]
    warnings: tuple[str, ...]


def _clamped_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _coerce_bool(value: Any, *, default: bool = False) -> bool:
    """Return a deterministic boolean, falling back safely for invalid values."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off", ""}:
            return False
        return default
    if isinstance(value, (int, float)):
        if value == 1:
            return True
        if value == 0:
            return False
    return default


def _configured_selection_limit() -> int:
    return _clamped_int(os.getenv("WATCHLIST_BRIEFING_MAX_ITEMS"), 100, 1, 1000)


def _mapping_copy(value: Any) -> dict[str, Any]:
    return copy.deepcopy(dict(value)) if isinstance(value, Mapping) else {}


def _deep_merge(base: dict[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _base_contract(*, scheduled: bool) -> dict[str, Any]:
    return {
        "version": BRIEFING_PIPELINE_VERSION,
        "selection": {"mode": "automatic", "max_items": _configured_selection_limit()},
        "editorial": {"program_format": "concise_briefing", "outcome_noun": "briefing"},
        "text": {
            "enabled": _coerce_bool(scheduled),
            "type": "briefing_markdown",
            "format": "md",
            "template_name": "",
            "show_notes": False,
        },
        "audio": {
            "enabled": False,
            "target_minutes": 10,
            "language": "en",
        },
        "delivery": {
            "reports": {"enabled": True},
            "email": {"enabled": False, "recipients": []},
            "chatbook": {"enabled": False},
        },
        "test": {"external_delivery": False, "audio_sample_seconds": 60},
    }


def _legacy_contract(raw: Mapping[str, Any]) -> dict[str, Any]:
    legacy: dict[str, Any] = {}

    auto_output = _mapping_copy(raw.get("auto_output"))
    template = _mapping_copy(raw.get("template"))
    text = auto_output
    if template:
        for key, value in template.items():
            if key not in {"default_name", "default_version", "default_format"}:
                text.setdefault(key, value)
        text.setdefault("template_name", template.get("default_name"))
        text.setdefault("template_version", template.get("default_version"))
        text.setdefault("format", template.get("default_format"))
    if raw.get("template_name") is not None:
        text["template_name"] = copy.deepcopy(raw.get("template_name"))
    text = {key: value for key, value in text.items() if value is not None}
    if text:
        legacy["text"] = text

    audio_key_map = {
        "target_audio_minutes": "target_minutes",
        "audio_language": "language",
        "audio_provider": "provider",
        "audio_model": "model",
        "audio_voice": "voice",
        "audio_speed": "speed",
        "audio_cast": "cast",
        "voice_map": "voice_map",
        "llm_provider": "llm_provider",
        "llm_model": "llm_model",
        "persona_summarize": "persona_summarize",
        "persona_id": "persona_id",
        "persona_provider": "persona_provider",
        "persona_model": "persona_model",
        "background_audio_uri": "background_audio_uri",
        "background_volume": "background_volume",
        "background_delay_ms": "background_delay_ms",
        "background_fade_seconds": "background_fade_seconds",
    }
    audio: dict[str, Any] = {}
    if "generate_audio" in raw:
        audio["enabled"] = _coerce_bool(raw.get("generate_audio"))
    for old_key, new_key in audio_key_map.items():
        if old_key in raw and raw.get(old_key) is not None:
            audio[new_key] = copy.deepcopy(raw.get(old_key))
    for canonical_key, aliases in {
        "provider": ("tts_provider",),
        "model": ("tts_model",),
        "voice": ("tts_voice",),
    }.items():
        if canonical_key not in audio:
            for alias in aliases:
                if raw.get(alias) is not None:
                    audio[canonical_key] = copy.deepcopy(raw.get(alias))
                    break
    if audio:
        legacy["audio"] = audio

    delivery = _mapping_copy(raw.get("deliveries"))
    delivery_config = _mapping_copy(raw.get("delivery_config"))
    if delivery_config:
        email = _mapping_copy(delivery.get("email"))
        if "email_recipients" in delivery_config:
            email.setdefault("recipients", copy.deepcopy(delivery_config["email_recipients"]))
        if "email_enabled" in delivery_config:
            email.setdefault("enabled", _coerce_bool(delivery_config["email_enabled"]))
        if email:
            delivery["email"] = email
        chatbook = _mapping_copy(delivery.get("chatbook"))
        if "create_chatbook" in delivery_config:
            chatbook.setdefault("enabled", _coerce_bool(delivery_config["create_chatbook"]))
        if chatbook:
            delivery["chatbook"] = chatbook
        for key, value in delivery_config.items():
            if key not in {"email_recipients", "email_enabled", "create_chatbook"}:
                delivery.setdefault(key, copy.deepcopy(value))
    if delivery:
        legacy["delivery"] = delivery

    return legacy


def _finalize_contract(contract: dict[str, Any], *, scheduled: bool) -> dict[str, Any]:
    defaults = _base_contract(scheduled=scheduled)
    for section in ("selection", "editorial", "text", "audio", "delivery", "test"):
        value = contract.get(section)
        contract[section] = (
            _deep_merge(defaults[section], value) if isinstance(value, Mapping) else copy.deepcopy(defaults[section])
        )

    contract["version"] = BRIEFING_PIPELINE_VERSION
    selection = contract["selection"]
    selection["max_items"] = _clamped_int(selection.get("max_items"), _configured_selection_limit(), 1, 1000)
    if selection.get("mode") not in {"automatic", "manual_override"}:
        selection["mode"] = "automatic"

    editorial = contract["editorial"]
    if editorial.get("program_format") not in PROGRAM_FORMATS:
        editorial["program_format"] = "concise_briefing"
    if editorial.get("outcome_noun") not in {"briefing", "episode"}:
        editorial["outcome_noun"] = "briefing"

    audio = contract["audio"]
    audio["enabled"] = _coerce_bool(audio.get("enabled"))
    if "persona_summarize" in audio:
        audio["persona_summarize"] = _coerce_bool(audio.get("persona_summarize"))
    audio["target_minutes"] = _clamped_int(audio.get("target_minutes"), 10, 1, 60)
    audio["language"] = str(audio.get("language") or "en")

    text = contract["text"]
    text["enabled"] = _coerce_bool(scheduled) or _coerce_bool(text.get("enabled")) or audio["enabled"]
    text["type"] = str(text.get("type") or "briefing_markdown")
    text["format"] = text.get("format") if text.get("format") in {"md", "html"} else "md"
    text["template_name"] = str(text.get("template_name") or "")
    text["show_notes"] = _coerce_bool(text.get("show_notes"))

    delivery = contract["delivery"]
    delivery["reports"] = _deep_merge(_mapping_copy(delivery.get("reports")), {"enabled": True})
    for channel in ("email", "chatbook"):
        channel_config = _mapping_copy(delivery.get(channel))
        channel_config["enabled"] = _coerce_bool(channel_config.get("enabled"))
        delivery[channel] = channel_config
    if not isinstance(delivery["email"].get("recipients"), list):
        delivery["email"]["recipients"] = []

    test = contract["test"]
    test["external_delivery"] = False
    test["audio_sample_seconds"] = 60
    return contract


def normalize_briefing_output_prefs(
    raw: Mapping[str, Any] | None,
    *,
    scheduled: bool,
) -> NormalizedBriefingContract:
    """Return canonical briefing intent while preserving unrelated fields."""
    source = _mapping_copy(raw)
    canonical = _mapping_copy(source.get(BRIEFING_PIPELINE_KEY))
    contract = _base_contract(scheduled=scheduled)
    contract = _deep_merge(contract, _legacy_contract(source))
    contract = _deep_merge(contract, canonical)
    contract = _finalize_contract(contract, scheduled=scheduled)

    legacy_consumed = any(key in source for key in _LEGACY_BRIEFING_KEYS)
    output_prefs = copy.deepcopy(source)
    for key in _LEGACY_BRIEFING_KEYS:
        output_prefs.pop(key, None)
    output_prefs[BRIEFING_PIPELINE_KEY] = copy.deepcopy(contract)
    warnings = ("legacy_briefing_preferences_normalized",) if legacy_consumed else ()
    return NormalizedBriefingContract(output_prefs=output_prefs, contract=contract, warnings=warnings)


def get_briefing_contract(
    raw: Mapping[str, Any] | None,
    *,
    scheduled: bool,
) -> dict[str, Any]:
    """Read canonical or legacy briefing preferences without mutating input."""
    return normalize_briefing_output_prefs(raw, scheduled=scheduled).contract


def briefing_selection_limit(contract: Mapping[str, Any]) -> int:
    """Return max_items clamped to the configured 1..1000 range."""
    selection = contract.get("selection")
    value = selection.get("max_items") if isinstance(selection, Mapping) else None
    return _clamped_int(value, _configured_selection_limit(), 1, 1000)
