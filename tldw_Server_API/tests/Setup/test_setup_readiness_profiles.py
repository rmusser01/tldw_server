from __future__ import annotations

from tldw_Server_API.app.core.Setup.readiness_models import (
    LANE_IDS,
    LANE_STATUSES,
    OVERLAY_IDS,
)
from tldw_Server_API.app.core.Setup.readiness_profiles import build_readiness_profiles


def _setup_status(needs_setup: bool = True) -> dict:
    return {
        "enabled": True,
        "setup_completed": not needs_setup,
        "needs_setup": needs_setup,
        "allow_remote_setup_access": False,
        "remote_access_active": False,
        "placeholder_fields": [],
    }


def _config_snapshot() -> dict:
    return {
        "sections": [
            {
                "name": "API",
                "fields": [
                    {"key": "default_api", "value": "openai", "placeholder": False},
                    {"key": "openai_api_key", "value": "", "placeholder": True, "is_secret": True},
                ],
            },
            {
                "name": "Embeddings",
                "fields": [
                    {"key": "embedding_provider", "value": "huggingface", "placeholder": False},
                    {
                        "key": "embedding_model",
                        "value": "Qwen/Qwen3-Embedding-0.6B",
                        "placeholder": False,
                    },
                ],
            },
        ]
    }


def _audio_recommendations() -> dict:
    return {
        "machine_profile": {
            "platform": "darwin",
            "arch": "arm64",
            "apple_silicon": True,
            "cuda_available": False,
            "free_disk_gb": 128.0,
            "network_available_for_downloads": True,
        },
        "recommendations": [
            {
                "bundle_id": "apple_silicon_local",
                "resource_profile": "balanced",
                "selection_key": "v2:apple_silicon_local:balanced:kokoro",
                "bundle": {"label": "Apple Silicon Local"},
                "profile": {
                    "profile_id": "balanced",
                    "label": "Balanced",
                    "default_tts_choice": "kokoro",
                    "tts_choices": [
                        {"choice_id": "kokoro", "label": "Kokoro"},
                        {"choice_id": "kitten_tts", "label": "Kitten TTS"},
                    ],
                },
            }
        ],
        "catalog": [],
        "excluded": [],
    }


def test_readiness_contract_uses_canonical_lanes_and_overlays():
    assert LANE_IDS == ("chat", "embeddings_rag", "speech")
    assert "restart_required" not in LANE_STATUSES
    assert "restart_required" in OVERLAY_IDS
    assert "requires_admin" in OVERLAY_IDS
    assert "remote_setup_blocked" in OVERLAY_IDS


def test_profiles_return_canonical_lanes_and_curated_profiles():
    response = build_readiness_profiles(
        setup_status=_setup_status(),
        config_snapshot=_config_snapshot(),
        audio_recommendations=_audio_recommendations(),
    )

    assert [lane["lane_id"] for lane in response["lanes"]] == list(LANE_IDS)
    assert response["recommended_profile_id"] == "local_balanced"
    assert [profile["profile_id"] for profile in response["profiles"]] == [
        "local_light",
        "local_balanced",
        "local_performance",
        "hosted_plus_local_speech",
        "advanced_custom",
    ]
    assert response["supported_overlays"] == list(OVERLAY_IDS)
    assert all(lane["status"] in LANE_STATUSES for lane in response["lanes"])


def test_speech_lane_carries_tts_as_secondary_metadata():
    response = build_readiness_profiles(
        setup_status=_setup_status(),
        config_snapshot=_config_snapshot(),
        audio_recommendations=_audio_recommendations(),
    )

    speech_lane = next(lane for lane in response["lanes"] if lane["lane_id"] == "speech")

    assert speech_lane["status"] == "ready_with_warnings"
    assert speech_lane["primary_capability"] == "transcription"
    assert speech_lane["secondary_capabilities"] == ["tts"]
    assert speech_lane["selection"]["bundle_id"] == "apple_silicon_local"
    assert speech_lane["selection"]["resource_profile"] == "balanced"
    assert speech_lane["selection"]["tts_choice"] == "kokoro"


def test_post_setup_profiles_report_admin_required_overlay():
    response = build_readiness_profiles(
        setup_status=_setup_status(needs_setup=False),
        config_snapshot=_config_snapshot(),
        audio_recommendations=_audio_recommendations(),
    )

    assert "requires_admin" in response["active_overlays"]
    assert response["setup_access"]["mode"] == "admin"
