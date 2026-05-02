from tldw_Server_API.app.core.Setup import audio_profile_service
from tldw_Server_API.app.core.Setup.audio_profile_service import (
    MachineProfile,
    detect_machine_profile,
    rank_audio_bundles,
    recommend_audio_bundles,
)


class _CapturingLogger:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def warning(self, message: str, *args, **_kwargs) -> None:
        if args:
            message = message.format(*args)
        self.messages.append(message)


def test_nvidia_machine_prefers_nvidia_bundle():
    profile = MachineProfile(
        platform="linux",
        arch="x86_64",
        apple_silicon=False,
        cuda_available=True,
        ffmpeg_available=True,
        espeak_available=True,
        free_disk_gb=80.0,
        network_available_for_downloads=True,
    )

    ranked = rank_audio_bundles(
        profile,
        prefer_offline_runtime=True,
        allow_hosted_fallbacks=True,
    )

    assert ranked[0].bundle_id == "nvidia_local"
    assert ranked[0].resource_profile in {"balanced", "performance"}


def test_disk_constrained_cpu_machine_prefers_light_profile():
    profile = MachineProfile(
        platform="linux",
        arch="x86_64",
        apple_silicon=False,
        cuda_available=False,
        ffmpeg_available=True,
        espeak_available=True,
        free_disk_gb=1.2,
        network_available_for_downloads=True,
    )

    ranked = rank_audio_bundles(
        profile,
        prefer_offline_runtime=True,
        allow_hosted_fallbacks=True,
    )

    assert ranked[0].bundle_id == "cpu_local"
    assert ranked[0].resource_profile == "light"


def test_hosted_bundle_drops_when_hosted_fallbacks_disabled():
    profile = MachineProfile(
        platform="linux",
        arch="x86_64",
        apple_silicon=False,
        cuda_available=False,
        ffmpeg_available=True,
        espeak_available=True,
        free_disk_gb=40.0,
        network_available_for_downloads=True,
    )

    ranked = rank_audio_bundles(
        profile,
        prefer_offline_runtime=True,
        allow_hosted_fallbacks=False,
    )

    assert all(bundle.bundle_id != "hosted_plus_local_backup" for bundle in ranked)


def test_unsupported_hardware_bundles_move_to_excluded_list():
    profile = MachineProfile(
        platform="linux",
        arch="x86_64",
        apple_silicon=False,
        cuda_available=False,
        ffmpeg_available=True,
        espeak_available=True,
        free_disk_gb=40.0,
        network_available_for_downloads=True,
    )

    result = recommend_audio_bundles(
        profile,
        prefer_offline_runtime=True,
        allow_hosted_fallbacks=True,
    )

    recommendation_ids = {bundle["bundle_id"] for bundle in result["recommendations"]}
    excluded_ids = {bundle["bundle_id"] for bundle in result["excluded"]}

    assert "nvidia_local" not in recommendation_ids
    assert "apple_silicon_local" not in recommendation_ids
    assert {"nvidia_local", "apple_silicon_local"} <= excluded_ids
    assert "resource_profile" in result["recommendations"][0]
    assert "confidence" in result["recommendations"][0]


def test_detect_machine_profile_disk_failure_log_is_sanitized(monkeypatch):
    capture = _CapturingLogger()

    def fail_disk_usage(_path):
        raise OSError("disk probe failed at /private/setup-profile")

    monkeypatch.setattr(audio_profile_service, "logger", capture)
    monkeypatch.setattr(audio_profile_service.platform, "system", lambda: "Linux")
    monkeypatch.setattr(audio_profile_service.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(audio_profile_service.shutil, "disk_usage", fail_disk_usage)
    monkeypatch.setattr(audio_profile_service.shutil, "which", lambda _name: None)
    monkeypatch.setattr(audio_profile_service.os.path, "exists", lambda _path: False)
    monkeypatch.setattr(audio_profile_service.install_manager, "cuda_available", lambda: False)
    monkeypatch.setattr(audio_profile_service.install_manager, "downloads_allowed", lambda: False)

    profile = detect_machine_profile()
    joined = "\n".join(capture.messages)

    assert profile.free_disk_gb == 0.0
    assert "Failed to inspect disk availability for setup profile" in joined
    assert "disk probe failed" not in joined
    assert "/private/setup-profile" not in joined
