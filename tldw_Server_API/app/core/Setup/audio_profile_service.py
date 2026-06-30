"""Machine profile detection and audio bundle recommendation helpers."""

from __future__ import annotations

import os
import platform
import shutil
from pathlib import Path

from loguru import logger
from pydantic import BaseModel

from tldw_Server_API.app.core.Setup.audio_bundle_catalog import (
    AudioBundle,
    AudioResourceProfile,
    build_audio_selection_key,
    get_audio_bundle_catalog,
)
from tldw_Server_API.app.core.Setup import install_manager


class MachineProfile(BaseModel):
    """Best-effort local machine capability snapshot for setup recommendations."""

    platform: str
    arch: str
    apple_silicon: bool
    cuda_available: bool
    ffmpeg_available: bool
    espeak_available: bool
    free_disk_gb: float
    network_available_for_downloads: bool


class BundleRecommendation(BaseModel):
    """A scored audio bundle recommendation."""

    bundle_id: str
    label: str
    resource_profile: str
    selection_key: str
    confidence: str
    reasons: list[str]
    score: int


def detect_machine_profile() -> MachineProfile:
    """Build a best-effort machine profile from local signals."""

    system_name = platform.system().lower()
    machine_arch = platform.machine().lower()
    project_root = Path.cwd()

    try:
        disk_usage = shutil.disk_usage(project_root)
        free_disk_gb = round(disk_usage.free / (1024 ** 3), 1)
    except OSError:
        logger.warning("Failed to inspect disk availability for setup profile")
        free_disk_gb = 0.0

    espeak_present = bool(shutil.which("espeak-ng") or shutil.which("espeak"))
    if not espeak_present:
        espeak_library = os.getenv("PHONEMIZER_ESPEAK_LIBRARY")
        espeak_present = bool(espeak_library and os.path.exists(espeak_library))

    force_offline = os.getenv("TLDW_SETUP_FORCE_OFFLINE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    return MachineProfile(
        platform=system_name,
        arch=machine_arch,
        apple_silicon=system_name == "darwin" and machine_arch in {"arm64", "aarch64"},
        cuda_available=install_manager.cuda_available(),
        ffmpeg_available=bool(shutil.which("ffmpeg")),
        espeak_available=espeak_present,
        free_disk_gb=free_disk_gb,
        network_available_for_downloads=not force_offline and install_manager.downloads_allowed(),
    )


def rank_audio_bundles(
    profile: MachineProfile,
    *,
    prefer_offline_runtime: bool,
    allow_hosted_fallbacks: bool,
) -> list[BundleRecommendation]:
    """Return ranked bundle recommendations for the supplied machine profile."""

    catalog = get_audio_bundle_catalog()
    recommendations: list[BundleRecommendation] = []

    for bundle in catalog.bundles:
        if bundle.bundle_id == "hosted_plus_local_backup" and not allow_hosted_fallbacks:
            continue
        if bundle.bundle_id == "nvidia_local" and not profile.cuda_available:
            continue
        if bundle.bundle_id == "apple_silicon_local" and not profile.apple_silicon:
            continue

        for resource_profile, profile_definition in bundle.resource_profiles.items():
            score, reasons, confidence = _score_bundle_profile(
                bundle,
                profile_definition,
                profile,
                prefer_offline_runtime=prefer_offline_runtime,
                allow_hosted_fallbacks=allow_hosted_fallbacks,
            )
            recommendations.append(
                BundleRecommendation(
                    bundle_id=bundle.bundle_id,
                    label=bundle.label,
                    resource_profile=resource_profile,
                    selection_key=build_audio_selection_key(bundle.bundle_id, resource_profile, bundle.catalog_version),
                    confidence=confidence,
                    reasons=reasons,
                    score=score,
                )
            )

    return sorted(recommendations, key=lambda item: item.score, reverse=True)


def _score_bundle_profile(
    bundle: AudioBundle,
    profile_definition: AudioResourceProfile,
    machine_profile: MachineProfile,
    *,
    prefer_offline_runtime: bool,
    allow_hosted_fallbacks: bool,
) -> tuple[int, list[str], str]:
    score = 50
    reasons: list[str] = []
    confidence = "medium"
    profile_id = profile_definition.profile_id
    estimated_disk = profile_definition.estimated_disk_gb or 0.0

    if prefer_offline_runtime and bundle.offline_runtime_supported:
        score += 15
        reasons.append("Supports offline runtime after provisioning")

    if bundle.bundle_id == "nvidia_local":
        score += 40
        reasons.append("CUDA detected")
    elif bundle.bundle_id == "apple_silicon_local":
        score += 40
        reasons.append("Apple Silicon detected")
    elif bundle.bundle_id == "cpu_local":
        score += 20
        reasons.append("Conservative local-first default")
    elif bundle.bundle_id == "hosted_plus_local_backup":
        if allow_hosted_fallbacks:
            score += 10
            reasons.append("Hosted fallbacks allowed")
        if prefer_offline_runtime:
            score -= 15
            reasons.append("Offline runtime preferred")

    if profile_id == "light":
        if machine_profile.free_disk_gb and estimated_disk and machine_profile.free_disk_gb <= max(estimated_disk * 1.5, 3.0):
            score += 25
            reasons.append("Low-disk machine profile favors the light tier")
            confidence = "high"
        else:
            score -= 5
            reasons.append("Balanced tier preferred when disk pressure is low")
            confidence = "medium"
    elif profile_id == "balanced":
        score += 8
        reasons.append("Balanced tier is the conservative default")
        confidence = "high"
    elif profile_id == "performance":
        confidence = "low"
        if estimated_disk and machine_profile.free_disk_gb >= estimated_disk + 2.0:
            score += 12
            reasons.append("Sufficient disk headroom for the performance tier")
        else:
            score -= 20
            reasons.append("Insufficient disk headroom for the performance tier")

        if bundle.bundle_id == "nvidia_local" and machine_profile.cuda_available:
            score += 12
            reasons.append("GPU-backed local profile can support the performance tier")
            confidence = "high"
        elif bundle.bundle_id == "apple_silicon_local" and machine_profile.apple_silicon:
            score += 6
            reasons.append("Apple Silicon can support the performance tier")
            confidence = "medium"
        else:
            score -= 10
            reasons.append("No strong acceleration signal for the performance tier")

    if estimated_disk and machine_profile.free_disk_gb < estimated_disk:
        score -= 15
        reasons.append("Available disk is below the estimated bundle footprint")
        confidence = "low"

    if not machine_profile.ffmpeg_available:
        score -= 5
        reasons.append("FFmpeg prerequisite still needed")
    if not machine_profile.espeak_available:
        score -= 5
        reasons.append("eSpeak prerequisite still needed")

    return score, reasons, confidence


def recommend_audio_bundles(
    profile: MachineProfile,
    *,
    prefer_offline_runtime: bool,
    allow_hosted_fallbacks: bool,
) -> dict[str, list[dict[str, object]]]:
    """Return ranked and excluded bundles for the current machine profile."""

    ranked = rank_audio_bundles(
        profile,
        prefer_offline_runtime=prefer_offline_runtime,
        allow_hosted_fallbacks=allow_hosted_fallbacks,
    )
    excluded: list[dict[str, object]] = []

    if not profile.cuda_available:
        excluded.append(
            {
                "bundle_id": "nvidia_local",
                "reasons": ["CUDA not detected"],
            }
        )
    if not profile.apple_silicon:
        excluded.append(
            {
                "bundle_id": "apple_silicon_local",
                "reasons": ["Apple Silicon not detected"],
            }
        )
    if not allow_hosted_fallbacks:
        excluded.append(
            {
                "bundle_id": "hosted_plus_local_backup",
                "reasons": ["Hosted fallbacks disabled by operator preference"],
            }
        )

    return {
        "recommendations": [recommendation.model_dump() for recommendation in ranked],
        "excluded": excluded,
    }


__all__ = [
    "BundleRecommendation",
    "MachineProfile",
    "detect_machine_profile",
    "rank_audio_bundles",
    "recommend_audio_bundles",
]
