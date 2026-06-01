import hashlib
import json
from unittest.mock import MagicMock

import pytest

from tldw_Server_API.app.core.Setup.audio_pack_service import (
    build_audio_pack_manifest,
    register_imported_audio_pack,
    resolve_audio_pack_path,
    validate_audio_pack_manifest,
)


def test_audio_pack_manifest_captures_selection_identity():
    manifest = build_audio_pack_manifest(
        bundle_id="cpu_local",
        resource_profile="balanced",
        catalog_version="v2",
    )

    assert manifest["bundle_id"] == "cpu_local"
    assert manifest["resource_profile"] == "balanced"
    assert manifest["selection_key"] == "v2:cpu_local:balanced"
    assert "checksums" in manifest
    assert manifest["checksums"]["manifest_sha256"]


def test_audio_pack_manifest_captures_tts_choice_identity():
    manifest = build_audio_pack_manifest(
        bundle_id="cpu_local",
        resource_profile="balanced",
        tts_choice="kitten_tts",
        catalog_version="v2",
    )

    assert manifest["tts_choice"] == "kitten_tts"
    assert manifest["selection_key"] == "v2:cpu_local:balanced:kitten_tts"


def test_audio_pack_manifest_canonicalizes_default_tts_choice_identity():
    omitted = build_audio_pack_manifest(
        bundle_id="cpu_local",
        resource_profile="balanced",
        catalog_version="v2",
    )
    explicit_default = build_audio_pack_manifest(
        bundle_id="cpu_local",
        resource_profile="balanced",
        tts_choice="kokoro",
        catalog_version="v2",
    )

    assert omitted["selection_key"] == "v2:cpu_local:balanced"
    assert explicit_default["selection_key"] == "v2:cpu_local:balanced"
    assert explicit_default["tts_choice"] is None


def test_audio_pack_manifest_rejects_invalid_tts_choice():
    with pytest.raises(ValueError, match="Unknown curated TTS choice"):
        build_audio_pack_manifest(
            bundle_id="cpu_local",
            resource_profile="balanced",
            tts_choice="bogus_choice",
            catalog_version="v2",
        )


def test_validate_audio_pack_manifest_reports_platform_mismatch(tmp_path, monkeypatch):
    manifest = build_audio_pack_manifest(
        bundle_id="cpu_local",
        resource_profile="balanced",
        catalog_version="v2",
        compatibility={"platform": "linux", "arch": "x86_64", "python_version": "3.11"},
    )
    monkeypatch_root = tmp_path / "config"
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Setup.audio_pack_service.CONFIG_ROOT",
        monkeypatch_root,
    )
    pack_path = resolve_audio_pack_path("audio_pack.json")
    pack_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = validate_audio_pack_manifest(
        "audio_pack.json",
        machine_profile={"platform": "darwin", "arch": "arm64"},
        python_version="3.11.13",
    )

    assert result["compatible"] is False
    assert any("platform" in issue.lower() for issue in result["issues"])


def test_validate_audio_pack_manifest_rejects_invalid_tts_choice(tmp_path):
    manifest = build_audio_pack_manifest(
        bundle_id="cpu_local",
        resource_profile="balanced",
        catalog_version="v2",
    )
    manifest["tts_choice"] = "bogus_choice"
    manifest_without_checksums = dict(manifest)
    manifest_without_checksums["checksums"] = {}
    manifest["checksums"] = {
        "manifest_sha256": hashlib.sha256(
            json.dumps(manifest_without_checksums, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "assets": {},
    }
    pack_path = tmp_path / "audio_pack.json"
    pack_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = validate_audio_pack_manifest(pack_path)

    assert result["compatible"] is False
    assert any("curated TTS choice" in issue for issue in result["issues"])
    assert "invalid_tts_choice" in result["issue_codes"]
    assert result["selection_key"] == "v2:cpu_local:balanced"


def test_validate_audio_pack_manifest_rejects_noncanonical_selection_key(tmp_path):
    manifest = build_audio_pack_manifest(
        bundle_id="cpu_local",
        resource_profile="balanced",
        catalog_version="v2",
    )
    manifest["tts_choice"] = "kokoro"
    manifest["selection_key"] = "v2:cpu_local:balanced:kokoro"
    manifest_without_checksums = dict(manifest)
    manifest_without_checksums["checksums"] = {}
    manifest["checksums"] = {
        "manifest_sha256": hashlib.sha256(
            json.dumps(manifest_without_checksums, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "assets": {},
    }
    pack_path = tmp_path / "audio_pack.json"
    pack_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = validate_audio_pack_manifest(pack_path)

    assert result["compatible"] is False
    assert any("selection key" in issue.lower() for issue in result["issues"])
    assert "selection_key_mismatch" in result["issue_codes"]
    assert result["selection_key"] == "v2:cpu_local:balanced"


def test_register_imported_audio_pack_rejects_invalid_tts_choice_before_persistence(tmp_path):
    manifest = build_audio_pack_manifest(
        bundle_id="cpu_local",
        resource_profile="balanced",
        catalog_version="v2",
    )
    manifest["tts_choice"] = "bogus_choice"
    manifest_without_checksums = dict(manifest)
    manifest_without_checksums["checksums"] = {}
    manifest["checksums"] = {
        "manifest_sha256": hashlib.sha256(
            json.dumps(manifest_without_checksums, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "assets": {},
    }
    pack_path = tmp_path / "audio_pack.json"
    pack_path.write_text(json.dumps(manifest), encoding="utf-8")

    readiness_store = MagicMock()
    readiness_store.load.return_value = {"imported_packs": []}

    with pytest.raises(ValueError, match="curated TTS choice"):
        register_imported_audio_pack(pack_path, readiness_store=readiness_store)

    readiness_store.update.assert_not_called()


def test_register_imported_audio_pack_persists_canonical_default_tts_choice(tmp_path):
    manifest = build_audio_pack_manifest(
        bundle_id="cpu_local",
        resource_profile="balanced",
        catalog_version="v2",
    )
    manifest["tts_choice"] = "kokoro"
    manifest_without_checksums = dict(manifest)
    manifest_without_checksums["checksums"] = {}
    manifest["checksums"] = {
        "manifest_sha256": hashlib.sha256(
            json.dumps(manifest_without_checksums, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "assets": {},
    }
    pack_path = tmp_path / "audio_pack.json"
    pack_path.write_text(json.dumps(manifest), encoding="utf-8")

    readiness_store = MagicMock()
    readiness_store.load.return_value = {"imported_packs": []}

    result = register_imported_audio_pack(pack_path, readiness_store=readiness_store)

    assert result["tts_choice"] is None
    assert result["selection_key"] == "v2:cpu_local:balanced"

    update_kwargs = readiness_store.update.call_args.kwargs
    assert update_kwargs["tts_choice"] is None
    assert update_kwargs["selection_key"] == "v2:cpu_local:balanced"
    assert update_kwargs["imported_packs"][-1]["tts_choice"] is None
    assert update_kwargs["imported_packs"][-1]["selection_key"] == "v2:cpu_local:balanced"


def test_register_imported_audio_pack_rejects_noncanonical_selection_key(tmp_path):
    manifest = build_audio_pack_manifest(
        bundle_id="cpu_local",
        resource_profile="balanced",
        catalog_version="v2",
    )
    manifest["tts_choice"] = "kokoro"
    manifest["selection_key"] = "v2:cpu_local:balanced:kokoro"
    manifest_without_checksums = dict(manifest)
    manifest_without_checksums["checksums"] = {}
    manifest["checksums"] = {
        "manifest_sha256": hashlib.sha256(
            json.dumps(manifest_without_checksums, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "assets": {},
    }
    pack_path = tmp_path / "audio_pack.json"
    pack_path.write_text(json.dumps(manifest), encoding="utf-8")

    readiness_store = MagicMock()
    readiness_store.load.return_value = {"imported_packs": []}

    with pytest.raises(ValueError, match="selection key"):
        register_imported_audio_pack(pack_path, readiness_store=readiness_store)

    readiness_store.update.assert_not_called()


def test_validate_audio_pack_manifest_handles_missing_bundle_identity_fields(tmp_path):
    pack_path = tmp_path / "audio_pack.json"
    pack_path.write_text(
        json.dumps(
            {
                "format": "audio_bundle_pack_manifest_v1",
                "checksums": {},
                "compatibility": {"platform": "darwin", "arch": "arm64", "python_version": "3.11"},
                "assets": [],
            }
        ),
        encoding="utf-8",
    )

    result = validate_audio_pack_manifest(pack_path)

    assert result["compatible"] is False
    assert any("audio bundle or resource profile" in issue.lower() for issue in result["issues"])
    assert "unknown_bundle" in result["issue_codes"]


def test_register_imported_audio_pack_rejects_unknown_bundle_identity(tmp_path):
    manifest = build_audio_pack_manifest(
        bundle_id="cpu_local",
        resource_profile="balanced",
        catalog_version="v2",
    )
    manifest["bundle_id"] = "unknown_bundle"
    manifest["selection_key"] = "v2:unknown_bundle:balanced"
    manifest_without_checksums = dict(manifest)
    manifest_without_checksums["checksums"] = {}
    manifest["checksums"] = {
        "manifest_sha256": hashlib.sha256(
            json.dumps(manifest_without_checksums, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "assets": {},
    }
    pack_path = tmp_path / "audio_pack.json"
    pack_path.write_text(json.dumps(manifest), encoding="utf-8")

    readiness_store = MagicMock()
    readiness_store.load.return_value = {"imported_packs": []}

    with pytest.raises(ValueError, match="not available in this catalog"):
        register_imported_audio_pack(pack_path, readiness_store=readiness_store)

    readiness_store.update.assert_not_called()
