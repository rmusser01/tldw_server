"""The one-shot control command authenticates a bundle before creating instance state."""

from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from tldw_Server_API.app.core.AuthNZ.api_key_crypto import parse_api_key
from tldw_Server_API.scripts.app_bundle_control import (
    BundleControlError,
    initialize_bundle,
    main,
    verify_bundle,
)


@pytest.fixture
def release(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, bytes]]:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    payload = b"known local artifact"
    (bundle / "catalog.json").write_bytes(payload)
    artifacts = [
        {
            "id": "catalog-amd64",
            "kind": "file",
            "role": "catalog",
            "platform": "linux/amd64",
            "location": "https://example.invalid/catalog.json",
            "path": "catalog.json",
            "size_bytes": len(payload),
            "installed_size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    ]
    for name in (
        "README.md",
        "compose.yaml",
        "start.sh",
        "stop.sh",
        "status.sh",
        "start.ps1",
        "stop.ps1",
        "status.ps1",
    ):
        (bundle / name).write_bytes(payload)
        artifacts.append({**artifacts[0], "id": name, "role": "bundle-helper", "path": name})
    for role, digest in (("backend", "a"), ("webui", "b"), ("gateway", "c")):
        artifacts.append(
            {
                "id": f"{role}-amd64",
                "kind": "oci",
                "role": role,
                "platform": "linux/amd64",
                "location": f"registry.invalid/tldw/{role}@sha256:{digest * 64}",
                "image_digest": f"sha256:{digest * 64}",
                "sha256": digest * 64,
                "size_bytes": 1024,
                "installed_size_bytes": 4096,
            }
        )
    manifest = {
        "schema_version": 1,
        "version": "0.2.0",
        "source_commit": "d" * 40,
        "created_at": "2026-09-22T00:00:00Z",
        "channel": "preview",
        "signer_id": "test-key",
        "platforms": ["linux/amd64"],
        "compatibility": {
            "min_launcher": "0.1.0",
            "python_version": "3.12.7",
            "node_version": "24.6.0",
            "backend_generation": 1,
            "browser_generation": 1,
            "allowed_upgrade_sources": ["0.1.0"],
            "components": ["core"],
        },
        "artifacts": artifacts,
        "dependencies": {"lock_digests": {"core": "e" * 64}},
        "data": {
            "inventory_schema": 1,
            "migration_generation": 1,
            "rollback_eligible": True,
            "component_catalog_digest": "f" * 64,
        },
        "qualifications": {
            "gates": {"G2": True, "G4": True, "G10": True},
            "evidence_links": ["https://example.invalid/evidence"],
        },
    }
    key = Ed25519PrivateKey.from_private_bytes(bytes(range(32)))
    raw = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest_path = bundle / "manifest.json"
    signature_path = bundle / "manifest.sig"
    manifest_path.write_bytes(raw)
    signature_path.write_bytes(key.sign(raw))
    trusted_keys = {
        "test-key": key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
    }
    return bundle, manifest_path, signature_path, trusted_keys


def _verified(release: tuple[Path, Path, Path, dict[str, bytes]]):
    bundle, manifest, signature, keys = release
    return verify_bundle(manifest, signature, keys, platform="linux/amd64", bundle_root=bundle)


def test_first_init_creates_private_persistent_state(
    tmp_path: Path,
    release: tuple[Path, Path, Path, dict[str, bytes]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    state = tmp_path / "data" / "instance"
    config = initialize_bundle(state, _verified(release), public_port=18080)

    assert stat.S_IMODE(state.stat().st_mode) == 0o700
    assert stat.S_IMODE((state / "config.env").stat().st_mode) == 0o600
    assert stat.S_IMODE((state / "identity.json").stat().st_mode) == 0o600
    assert config.project_id.startswith("tldw_")
    assert parse_api_key(config.api_key) is not None
    assert config.session_cookie_name != config.csrf_cookie_name
    assert config.public_port == 18080
    assert all("@sha256:" in ref for ref in config.images.values())
    assert config.api_key not in capsys.readouterr().out


def test_repeat_init_reuses_exact_credentials(
    tmp_path: Path,
    release: tuple[Path, Path, Path, dict[str, bytes]],
) -> None:
    state = tmp_path / "instance"
    first = initialize_bundle(state, _verified(release))
    before = (state / "config.env").read_bytes()

    second = initialize_bundle(state, _verified(release))

    assert first == second
    assert (state / "config.env").read_bytes() == before


def test_bad_signature_does_not_create_state(
    tmp_path: Path,
    release: tuple[Path, Path, Path, dict[str, bytes]],
) -> None:
    bundle, manifest, signature, keys = release
    signature.write_bytes(b"\0" * 64)
    state = tmp_path / "instance"

    with pytest.raises(BundleControlError, match="signature"):
        verified = verify_bundle(manifest, signature, keys, platform="linux/amd64", bundle_root=bundle)
        initialize_bundle(state, verified)

    assert not state.exists()


def test_existing_state_conflicting_release_is_rejected_without_changes(
    tmp_path: Path,
    release: tuple[Path, Path, Path, dict[str, bytes]],
) -> None:
    state = tmp_path / "instance"
    initialize_bundle(state, _verified(release))
    before = (state / "config.env").read_bytes()
    bundle, manifest_path, signature_path, keys = release
    changed = json.loads(manifest_path.read_text())
    changed["version"] = "0.2.1"
    key = Ed25519PrivateKey.from_private_bytes(bytes(range(32)))
    raw = json.dumps(changed, sort_keys=True, separators=(",", ":")).encode()
    manifest_path.write_bytes(raw)
    signature_path.write_bytes(key.sign(raw))

    with pytest.raises(BundleControlError, match="different release"):
        initialize_bundle(
            state, verify_bundle(manifest_path, signature_path, keys, platform="linux/amd64", bundle_root=bundle)
        )

    assert (state / "config.env").read_bytes() == before


def test_missing_or_changed_artifact_fails_before_state_mutation(
    tmp_path: Path,
    release: tuple[Path, Path, Path, dict[str, bytes]],
) -> None:
    bundle, manifest, signature, keys = release
    (bundle / "catalog.json").write_bytes(b"changed local artifact")

    with pytest.raises(BundleControlError, match="artifact"):
        verify_bundle(manifest, signature, keys, platform="linux/amd64", bundle_root=bundle)

    assert not (tmp_path / "instance").exists()


def test_existing_symlink_state_is_rejected(
    tmp_path: Path,
    release: tuple[Path, Path, Path, dict[str, bytes]],
) -> None:
    target = tmp_path / "target"
    target.mkdir()
    state = tmp_path / "instance"
    state.symlink_to(target, target_is_directory=True)

    with pytest.raises(BundleControlError, match="unsafe"):
        initialize_bundle(state, _verified(release))

    assert list(target.iterdir()) == []


def test_verify_command_does_not_create_or_change_state(
    tmp_path: Path,
    release: tuple[Path, Path, Path, dict[str, bytes]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    bundle, manifest, signature, keys = release
    trusted = tmp_path / "trusted"
    trusted.mkdir()
    (trusted / "test-key.pub").write_bytes(keys["test-key"])
    state = tmp_path / "instance"
    arguments = [
        "--state",
        str(state),
        "--manifest",
        str(manifest),
        "--signature",
        str(signature),
        "--platform",
        "linux/amd64",
        "--bundle-root",
        str(bundle),
        "--trusted-keys",
        str(trusted),
    ]

    assert main(["verify", *arguments]) == 0
    assert not state.exists()
    config = initialize_bundle(state, _verified(release))
    before = (state / "config.env").read_bytes()
    assert main(["verify", *arguments]) == 0
    output = capsys.readouterr()
    assert (state / "config.env").read_bytes() == before
    assert config.api_key not in output.out + output.err
    assert config.gateway_hop_secret not in output.out + output.err


def test_helper_pinned_signer_must_match_manifest(
    tmp_path: Path,
    release: tuple[Path, Path, Path, dict[str, bytes]],
) -> None:
    bundle, manifest, signature, keys = release
    trusted = tmp_path / "trusted"
    trusted.mkdir()
    (trusted / "test-key.pub").write_bytes(keys["test-key"])
    state = tmp_path / "instance"

    assert (
        main(
            [
                "init",
                "--state",
                str(state),
                "--manifest",
                str(manifest),
                "--signature",
                str(signature),
                "--platform",
                "linux/amd64",
                "--bundle-root",
                str(bundle),
                "--trusted-keys",
                str(trusted),
                "--expected-signer",
                "different-key",
            ]
        )
        == 1
    )
    assert not state.exists()


def test_missing_required_platform_image_fails(
    release: tuple[Path, Path, Path, dict[str, bytes]],
) -> None:
    bundle, manifest_path, signature_path, keys = release
    changed = json.loads(manifest_path.read_text())
    changed["artifacts"] = [artifact for artifact in changed["artifacts"] if artifact["role"] != "gateway"]
    key = Ed25519PrivateKey.from_private_bytes(bytes(range(32)))
    raw = json.dumps(changed, sort_keys=True, separators=(",", ":")).encode()
    manifest_path.write_bytes(raw)
    signature_path.write_bytes(key.sign(raw))

    with pytest.raises(BundleControlError, match="gateway image"):
        verify_bundle(manifest_path, signature_path, keys, platform="linux/amd64", bundle_root=bundle)


def test_file_artifact_cannot_traverse_symlink(
    tmp_path: Path,
    release: tuple[Path, Path, Path, dict[str, bytes]],
) -> None:
    bundle, manifest_path, signature_path, keys = release
    original = bundle / "catalog.json"
    directory = tmp_path / "outside"
    directory.mkdir()
    original.rename(directory / "catalog.json")
    (bundle / "nested").symlink_to(directory, target_is_directory=True)
    changed = json.loads(manifest_path.read_text())
    changed["artifacts"][0]["path"] = "nested/catalog.json"
    key = Ed25519PrivateKey.from_private_bytes(bytes(range(32)))
    raw = json.dumps(changed, sort_keys=True, separators=(",", ":")).encode()
    manifest_path.write_bytes(raw)
    signature_path.write_bytes(key.sign(raw))

    with pytest.raises(BundleControlError, match="symlink"):
        verify_bundle(manifest_path, signature_path, keys, platform="linux/amd64", bundle_root=bundle)


def test_signed_image_location_cannot_inject_instance_env(
    release: tuple[Path, Path, Path, dict[str, bytes]],
) -> None:
    bundle, manifest_path, signature_path, keys = release
    changed = json.loads(manifest_path.read_text())
    next(a for a in changed["artifacts"] if a["role"] == "backend")["location"] = (
        "registry.invalid/tldw/backend\nSINGLE_USER_API_KEY=override@sha256:" + "a" * 64
    )
    key = Ed25519PrivateKey.from_private_bytes(bytes(range(32)))
    raw = json.dumps(changed, sort_keys=True, separators=(",", ":")).encode()
    manifest_path.write_bytes(raw)
    signature_path.write_bytes(key.sign(raw))

    with pytest.raises(BundleControlError, match="image location"):
        verify_bundle(manifest_path, signature_path, keys, platform="linux/amd64", bundle_root=bundle)


@pytest.mark.parametrize("inspection", [b"[]", b"not-json-private-sentinel", b"x" * (1024 * 1024 + 1)])
def test_ready_cli_refuses_bad_private_inspection_without_leaking_or_changing_state(
    tmp_path, release, monkeypatch, capsys, inspection
):
    import io
    import sys
    from types import SimpleNamespace

    bundle, manifest, signature, keys = release
    trusted = tmp_path / "trusted"
    trusted.mkdir()
    (trusted / "test-key.pub").write_bytes(keys["test-key"])
    state = tmp_path / "instance"
    config = initialize_bundle(state, _verified(release))
    before = (state / "config.env").read_bytes()
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(buffer=io.BytesIO(inspection)))
    result = main(
        [
            "ready",
            "--state",
            str(state),
            "--manifest",
            str(manifest),
            "--signature",
            str(signature),
            "--bundle-root",
            str(bundle),
            "--platform",
            "linux/amd64",
            "--trusted-keys",
            str(trusted),
        ]
    )
    assert result == 1
    output = capsys.readouterr()
    assert "private-sentinel" not in output.err + output.out
    assert config.api_key not in output.err + output.out
    assert (state / "config.env").read_bytes() == before


def test_established_origin_cannot_change_credentials_or_data(tmp_path, release):
    state = tmp_path / "instance"
    first = initialize_bundle(state, _verified(release), public_port=18080)
    before = (state / "config.env").read_bytes()
    sentinel = state / "browser-data-sentinel"
    sentinel.write_bytes(b"retain-me")
    with pytest.raises(BundleControlError, match="conflicts"):
        initialize_bundle(state, _verified(release), public_port=18081)
    assert (state / "config.env").read_bytes() == before
    assert sentinel.read_bytes() == b"retain-me"
    assert initialize_bundle(state, _verified(release)) == first
