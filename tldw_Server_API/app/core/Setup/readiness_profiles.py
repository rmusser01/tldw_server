"""Curated first-run setup readiness profile helpers."""

from __future__ import annotations

import os
from typing import Any

from tldw_Server_API.app.core.Setup.readiness_models import (
    LANE_CHAT,
    LANE_EMBEDDINGS_RAG,
    LANE_IDS,
    LANE_SPEECH,
    LANE_STATUSES,
    OVERLAY_IDS,
    build_lane_summary,
)

PROFILE_IDS = (
    "local_light",
    "local_balanced",
    "local_performance",
    "hosted_plus_local_speech",
    "advanced_custom",
)

_PROFILE_LABELS = {
    "local_light": "Local Light",
    "local_balanced": "Local Balanced",
    "local_performance": "Local Performance",
    "hosted_plus_local_speech": "Hosted Plus Local Speech",
    "advanced_custom": "Advanced Custom",
}

_PROFILE_DESCRIPTIONS = {
    "local_light": "Lower disk and memory footprint for constrained machines.",
    "local_balanced": "Recommended local-first default for most machines.",
    "local_performance": "Larger local footprint for better quality or throughput.",
    "hosted_plus_local_speech": "Hosted chat or embeddings with local speech readiness.",
    "advanced_custom": "Expose exact provider, endpoint, and model controls.",
}

_PROFILE_RESOURCE_HINTS = {
    "local_light": "light",
    "local_balanced": "balanced",
    "local_performance": "performance",
    "hosted_plus_local_speech": "balanced",
    "advanced_custom": None,
}


def _field_lookup(config_snapshot: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for section in config_snapshot.get("sections") or []:
        section_name = str(section.get("name") or "")
        for field in section.get("fields") or []:
            key = str(field.get("key") or "")
            if section_name and key:
                lookup[(section_name, key)] = dict(field)
    return lookup


def _field_value(
    fields: dict[tuple[str, str], dict[str, Any]],
    section: str,
    key: str,
) -> str:
    value = fields.get((section, key), {}).get("value")
    return str(value or "").strip()


def _field_is_placeholder(
    fields: dict[tuple[str, str], dict[str, Any]],
    section: str,
    key: str,
) -> bool:
    return bool(fields.get((section, key), {}).get("placeholder"))


def _first_audio_recommendation(audio_recommendations: dict[str, Any]) -> dict[str, Any]:
    recommendations = audio_recommendations.get("recommendations")
    if isinstance(recommendations, list) and recommendations:
        first = recommendations[0]
        if isinstance(first, dict):
            return first
    return {}


def _speech_selection(audio_recommendations: dict[str, Any]) -> dict[str, Any]:
    recommendation = _first_audio_recommendation(audio_recommendations)
    if not recommendation:
        return {}

    profile = recommendation.get("profile") if isinstance(recommendation.get("profile"), dict) else {}
    tts_choice = profile.get("default_tts_choice")
    if not tts_choice:
        tts_choices = profile.get("tts_choices")
        if isinstance(tts_choices, list) and tts_choices:
            first_choice = tts_choices[0]
            if isinstance(first_choice, dict):
                tts_choice = first_choice.get("choice_id")

    return {
        "bundle_id": recommendation.get("bundle_id"),
        "resource_profile": recommendation.get("resource_profile"),
        "selection_key": recommendation.get("selection_key"),
        "tts_choice": tts_choice,
    }


def _chat_lane(fields: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    default_api = _field_value(fields, "API", "default_api")
    if not default_api:
        return build_lane_summary(
            LANE_CHAT,
            consequences=["Chat defaults can be configured later from provider settings."],
        )

    status = "ready_with_warnings"
    warnings: list[str] = []
    if default_api == "openai" and _field_is_placeholder(fields, "API", "openai_api_key"):
        warnings.append("Hosted provider key is not configured yet.")

    return build_lane_summary(
        LANE_CHAT,
        status=status,
        selection={"default_api": default_api},
        warnings=warnings,
    )


def _embeddings_lane(fields: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    provider = _field_value(fields, "Embeddings", "embedding_provider")
    model = _field_value(fields, "Embeddings", "embedding_model")
    if not provider or not model:
        return build_lane_summary(
            LANE_EMBEDDINGS_RAG,
            consequences=["RAG search will be limited until embeddings are configured."],
        )

    status = "ready_with_warnings"
    if _field_is_placeholder(fields, "Embeddings", "embedding_model"):
        status = "not_configured"

    return build_lane_summary(
        LANE_EMBEDDINGS_RAG,
        status=status,
        selection={"provider": provider, "model": model},
    )


def _speech_lane(audio_recommendations: dict[str, Any]) -> dict[str, Any]:
    selection = _speech_selection(audio_recommendations)
    if not selection:
        return build_lane_summary(
            LANE_SPEECH,
            consequences=["Transcription can be configured later from setup or admin settings."],
        )

    return build_lane_summary(
        LANE_SPEECH,
        status="ready_with_warnings",
        selection=selection,
        warnings=["Speech bundle is recommended but still needs explicit provisioning or verification."],
    )


def _build_profile(profile_id: str, speech_selection: dict[str, Any]) -> dict[str, Any]:
    resource_hint = _PROFILE_RESOURCE_HINTS[profile_id]
    profile_speech_selection = dict(speech_selection)
    if resource_hint and profile_speech_selection:
        profile_speech_selection["resource_profile"] = resource_hint

    return {
        "profile_id": profile_id,
        "label": _PROFILE_LABELS[profile_id],
        "description": _PROFILE_DESCRIPTIONS[profile_id],
        "lanes": {
            LANE_CHAT: {"mode": "skip" if profile_id.startswith("local_") else "hosted_or_local"},
            LANE_EMBEDDINGS_RAG: {
                "mode": "local" if profile_id.startswith("local_") else "hosted_or_local",
            },
            LANE_SPEECH: profile_speech_selection,
        },
        "advanced": profile_id == "advanced_custom",
    }


def _active_overlays(setup_status: dict[str, Any], audio_recommendations: dict[str, Any]) -> list[str]:
    overlays: list[str] = []
    if not setup_status.get("needs_setup"):
        overlays.append("requires_admin")

    machine_profile = audio_recommendations.get("machine_profile")
    if isinstance(machine_profile, dict) and machine_profile.get("network_available_for_downloads") is False:
        overlays.append("network_unavailable")

    if os.getenv("TLDW_SETUP_SKIP_DOWNLOADS", "").strip().lower() in {"1", "true", "yes", "on"}:
        overlays.append("downloads_disabled")
    if os.getenv("TLDW_SETUP_SKIP_PIP", "").strip().lower() in {"1", "true", "yes", "on"}:
        overlays.append("package_installs_disabled")

    return overlays


def _recommended_profile_id(audio_recommendations: dict[str, Any]) -> str:
    resource_profile = str(_first_audio_recommendation(audio_recommendations).get("resource_profile") or "")
    if resource_profile in {"light", "balanced", "performance"}:
        return f"local_{resource_profile}"
    return "local_balanced"


def build_readiness_profiles(
    *,
    setup_status: dict[str, Any],
    config_snapshot: dict[str, Any],
    audio_recommendations: dict[str, Any],
) -> dict[str, Any]:
    """Build the first-run setup readiness profile payload without mutations."""

    fields = _field_lookup(config_snapshot)
    speech_selection = _speech_selection(audio_recommendations)
    setup_mode = "first_run" if setup_status.get("needs_setup") else "admin"

    return {
        "setup_access": {
            "mode": setup_mode,
            "needs_setup": bool(setup_status.get("needs_setup")),
            "setup_completed": bool(setup_status.get("setup_completed")),
            "remote_access_active": bool(setup_status.get("remote_access_active")),
        },
        "machine_profile": audio_recommendations.get("machine_profile") or {},
        "lane_ids": list(LANE_IDS),
        "supported_statuses": list(LANE_STATUSES),
        "supported_overlays": list(OVERLAY_IDS),
        "active_overlays": _active_overlays(setup_status, audio_recommendations),
        "lanes": [
            _chat_lane(fields),
            _embeddings_lane(fields),
            _speech_lane(audio_recommendations),
        ],
        "profiles": [_build_profile(profile_id, speech_selection) for profile_id in PROFILE_IDS],
        "recommended_profile_id": _recommended_profile_id(audio_recommendations),
    }


__all__ = [
    "PROFILE_IDS",
    "build_readiness_profiles",
]
