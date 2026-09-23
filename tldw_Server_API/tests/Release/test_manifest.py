"""Release manifests must authenticate a complete, platform-matched bundle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from tldw_Server_API.app.core.Release.manifest import (
    ManifestError,
    verify_artifact,
    verify_manifest,
)


@pytest.fixture
def signing_key() -> Ed25519PrivateKey:
    return Ed25519PrivateKey.from_private_bytes(bytes(range(32)))


@pytest.fixture
def trusted_keys(signing_key: Ed25519PrivateKey) -> dict[str, bytes]:
    public_key = signing_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return {"test-key": public_key}


@pytest.fixture
def manifest_data() -> dict[str, object]:
    payload = b"backend artifact"
    return {
        "schema_version": 1,
        "version": "0.2.0",
        "source_commit": "a" * 40,
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
        "artifacts": [
            {
                "id": "backend-wheel",
                "kind": "file",
                "role": "backend",
                "platform": "linux/amd64",
                "location": "https://example.invalid/backend.whl",
                "path": "backend.whl",
                "size_bytes": len(payload),
                "installed_size_bytes": 32,
                "sha256": hashlib.sha256(payload).hexdigest(),
            },
            {
                "id": "webui-image",
                "kind": "oci",
                "role": "webui",
                "platform": "linux/amd64",
                "location": "registry.invalid/tldw/webui@sha256:" + "b" * 64,
                "size_bytes": 1024,
                "installed_size_bytes": 4096,
                "sha256": "b" * 64,
                "image_digest": "sha256:" + "b" * 64,
            },
        ],
        "dependencies": {"lock_digests": {"core": "c" * 64}},
        "data": {
            "inventory_schema": 1,
            "migration_generation": 1,
            "rollback_eligible": True,
            "component_catalog_digest": "d" * 64,
        },
        "qualifications": {
            "gates": {"G2": True, "G4": True, "G10": True},
            "evidence_links": ["https://example.invalid/evidence"],
        },
    }


def signed(data: dict[str, object], signing_key: Ed25519PrivateKey) -> tuple[bytes, bytes]:
    raw = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return raw, signing_key.sign(raw)


def test_valid_manifest_binds_platform_and_roles(
    manifest_data: dict[str, object],
    signing_key: Ed25519PrivateKey,
    trusted_keys: dict[str, bytes],
) -> None:
    result = verify_manifest(*signed(manifest_data, signing_key), trusted_keys, platform="linux/amd64")

    assert (result.version, {artifact.role for artifact in result.artifacts}) == (
        "0.2.0",
        {"backend", "webui"},
    )


def test_mutated_manifest_is_rejected_before_parsing(
    manifest_data: dict[str, object],
    signing_key: Ed25519PrivateKey,
    trusted_keys: dict[str, bytes],
) -> None:
    raw, signature = signed(manifest_data, signing_key)

    with pytest.raises(ManifestError, match="signature"):
        verify_manifest(
            raw.replace(b'"version":"0.2.0"', b'"version":"0.2.1"'),
            signature,
            trusted_keys,
            platform="linux/amd64",
        )


def test_unknown_signer_is_rejected(manifest_data: dict[str, object], signing_key: Ed25519PrivateKey) -> None:
    with pytest.raises(ManifestError, match="signer"):
        verify_manifest(*signed(manifest_data, signing_key), {}, platform="linux/amd64")


def test_wrong_platform_is_rejected(
    manifest_data: dict[str, object],
    signing_key: Ed25519PrivateKey,
    trusted_keys: dict[str, bytes],
) -> None:
    with pytest.raises(ManifestError, match="platform"):
        verify_manifest(
            *signed(manifest_data, signing_key),
            trusted_keys,
            platform="linux/arm64",
        )


@pytest.mark.parametrize("path", ["../backend.whl", "/tmp/backend.whl", "a/../../b"])
def test_artifact_path_cannot_escape_bundle(
    path: str,
    manifest_data: dict[str, object],
    signing_key: Ed25519PrivateKey,
    trusted_keys: dict[str, bytes],
) -> None:
    manifest_data["artifacts"][0]["path"] = path  # type: ignore[index]

    with pytest.raises(ManifestError, match="path"):
        verify_manifest(
            *signed(manifest_data, signing_key),
            trusted_keys,
            platform="linux/amd64",
        )


@pytest.mark.parametrize("field", ["role", "sha256", "size_bytes"])
def test_artifact_missing_required_field_is_rejected(
    field: str,
    manifest_data: dict[str, object],
    signing_key: Ed25519PrivateKey,
    trusted_keys: dict[str, bytes],
) -> None:
    del manifest_data["artifacts"][0][field]  # type: ignore[index]

    with pytest.raises(ManifestError, match="artifact"):
        verify_manifest(
            *signed(manifest_data, signing_key),
            trusted_keys,
            platform="linux/amd64",
        )


def test_duplicate_artifact_id_is_rejected(
    manifest_data: dict[str, object],
    signing_key: Ed25519PrivateKey,
    trusted_keys: dict[str, bytes],
) -> None:
    manifest_data["artifacts"][1]["id"] = "backend-wheel"  # type: ignore[index]

    with pytest.raises(ManifestError, match="duplicate"):
        verify_manifest(
            *signed(manifest_data, signing_key),
            trusted_keys,
            platform="linux/amd64",
        )


def test_duplicate_json_keys_are_rejected_even_with_valid_signature(
    manifest_data: dict[str, object],
    signing_key: Ed25519PrivateKey,
    trusted_keys: dict[str, bytes],
) -> None:
    raw, _ = signed(manifest_data, signing_key)
    duplicated = raw.replace(b'"version":"0.2.0"', b'"version":"0.2.0","version":"0.2.1"')
    assert duplicated != raw

    with pytest.raises(ManifestError, match="duplicate"):
        verify_manifest(
            duplicated,
            signing_key.sign(duplicated),
            trusted_keys,
            platform="linux/amd64",
        )


def test_artifact_bytes_must_match_manifest(
    tmp_path: Path,
    manifest_data: dict[str, object],
    signing_key: Ed25519PrivateKey,
    trusted_keys: dict[str, bytes],
) -> None:
    result = verify_manifest(*signed(manifest_data, signing_key), trusted_keys, platform="linux/amd64")
    path = tmp_path / "backend.whl"
    path.write_bytes(b"backend artifacT")

    with pytest.raises(ManifestError, match="digest"):
        verify_artifact(path, result.artifacts[0])


def test_matching_artifact_is_accepted(
    tmp_path: Path,
    manifest_data: dict[str, object],
    signing_key: Ed25519PrivateKey,
    trusted_keys: dict[str, bytes],
) -> None:
    result = verify_manifest(*signed(manifest_data, signing_key), trusted_keys, platform="linux/amd64")
    path = tmp_path / "backend.whl"
    path.write_bytes(b"backend artifact")

    verify_artifact(path, result.artifacts[0])


def test_symlinked_artifact_is_rejected(
    tmp_path: Path,
    manifest_data: dict[str, object],
    signing_key: Ed25519PrivateKey,
    trusted_keys: dict[str, bytes],
) -> None:
    result = verify_manifest(*signed(manifest_data, signing_key), trusted_keys, platform="linux/amd64")
    real = tmp_path / "real.whl"
    real.write_bytes(b"backend artifact")
    link = tmp_path / "backend.whl"
    link.symlink_to(real)

    with pytest.raises(ManifestError, match="unsafe"):
        verify_artifact(link, result.artifacts[0])
