import json
from pathlib import PurePosixPath

import pytest

import tldw_Server_API.app.api.v1.endpoints.setup as setup_endpoint
from tldw_Server_API.app.api.v1.schemas.setup_schemas import AudioPackExportRequest, AudioPackImportRequest
from tldw_Server_API.app.core.Setup.audio_profile_service import MachineProfile
from tldw_Server_API.app.core.Setup.audio_readiness_store import AudioReadinessStore


def _machine_profile() -> MachineProfile:
    return MachineProfile(
        platform="linux",
        arch="x86_64",
        apple_silicon=False,
        cuda_available=False,
        ffmpeg_available=True,
        espeak_available=True,
        free_disk_gb=64.0,
        network_available_for_downloads=True,
    )


def test_audio_pack_request_models_accept_legacy_pack_path_alias():
    export_payload = AudioPackExportRequest.model_validate(
        {"bundle_id": "cpu_local", "resource_profile": "balanced", "pack_path": "audio_packs/legacy-pack.json"}
    )
    import_payload = AudioPackImportRequest.model_validate({"pack_path": "audio_packs\\legacy-pack.json"})

    assert export_payload.pack_name == "legacy-pack.json"
    assert import_payload.pack_name == "legacy-pack.json"


@pytest.mark.asyncio
async def test_export_audio_pack_writes_to_managed_directory(mocker, tmp_path):
    config_root = tmp_path / "Config_Files"
    mocker.patch.object(
        setup_endpoint.setup_manager,
        "get_status_snapshot",
        return_value={"enabled": True, "needs_setup": True},
    )
    mocker.patch.object(
        setup_endpoint.audio_profile_service,
        "detect_machine_profile",
        return_value=_machine_profile(),
    )
    mocker.patch.object(setup_endpoint.audio_pack_service, "CONFIG_ROOT", config_root)

    result = await setup_endpoint.export_audio_pack(
        AudioPackExportRequest(
            bundle_id="cpu_local",
            resource_profile="balanced",
            pack_name="managed-pack.json",
        ),
        None,
    )

    assert PurePosixPath(result["pack_path"].replace("\\", "/")) == PurePosixPath(
        "audio_packs/managed-pack.json"
    )
    assert (config_root / "audio_packs" / "managed-pack.json").is_file()


@pytest.mark.asyncio
async def test_import_audio_pack_uses_managed_directory(mocker, tmp_path):
    config_root = tmp_path / "Config_Files"
    store = AudioReadinessStore(tmp_path / "audio_readiness.json")
    manifest = setup_endpoint.audio_pack_service.build_audio_pack_manifest(
        bundle_id="cpu_local",
        resource_profile="balanced",
        catalog_version="v2",
        compatibility={"platform": "linux", "arch": "x86_64", "python_version": "3.11"},
    )
    mocker.patch.object(setup_endpoint.audio_pack_service, "CONFIG_ROOT", config_root)
    pack_path = setup_endpoint.audio_pack_service.resolve_audio_pack_path("import-pack.json")
    pack_path.write_text(json.dumps(manifest), encoding="utf-8")

    mocker.patch.object(
        setup_endpoint.setup_manager,
        "get_status_snapshot",
        return_value={"enabled": True, "needs_setup": True},
    )
    mocker.patch.object(
        setup_endpoint.audio_profile_service,
        "detect_machine_profile",
        return_value=_machine_profile(),
    )
    mocker.patch.object(
        setup_endpoint.audio_readiness_store,
        "get_audio_readiness_store",
        return_value=store,
    )

    result = await setup_endpoint.import_audio_pack(
        AudioPackImportRequest(pack_name="import-pack.json"),
        None,
    )

    pack_path = result["audio_readiness"]["imported_packs"][0]["pack_path"]
    assert PurePosixPath(pack_path.replace("\\", "/")) == PurePosixPath("audio_packs/import-pack.json")
    assert result["selection_key"] == "v2:cpu_local:balanced"
    assert result["audio_readiness"]["machine_profile"]["ffmpeg_available"] is True
    assert result["audio_readiness"]["machine_profile"]["free_disk_gb"] == 64.0
