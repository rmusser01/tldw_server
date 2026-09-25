"""Candidate signing and qualification must not promote partial image sets."""

from __future__ import annotations

from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from Helper_Scripts.build_app_bundle import build_candidate
from Helper_Scripts.verify_app_bundle import candidate_is_promotable


PLATFORMS = ("linux/amd64", "linux/arm64")


@pytest.fixture
def signing_key() -> Ed25519PrivateKey:
    return Ed25519PrivateKey.from_private_bytes(bytes(range(32)))


@pytest.fixture
def inventory() -> dict[str, object]:
    source_commit = "a" * 40
    artifacts = []
    for platform, letter in (("linux/amd64", "b"), ("linux/arm64", "c")):
        for role, role_letter in (("backend", "1"), ("webui", "2"), ("gateway", "3")):
            digest = letter + role_letter * 63
            artifacts.append(
                {
                    "id": f"{role}-{platform.split('/')[1]}",
                    "kind": "oci",
                    "role": role,
                    "platform": platform,
                    "source_commit": source_commit,
                    "location": f"localhost:5000/tldw/{role}@sha256:{digest}",
                    "image_digest": f"sha256:{digest}",
                    "sha256": digest,
                    "size_bytes": 1024,
                    "installed_size_bytes": 4096,
                }
            )
        artifacts.append(
            {
                "id": f"control-{platform.split('/')[1]}",
                "kind": "oci",
                "role": "control",
                "platform": platform,
                "source_commit": source_commit,
                "location": "localhost:5000/tldw/control@sha256:" + "d" * 64,
                "image_digest": "sha256:" + "d" * 64,
                "sha256": "d" * 64,
                "size_bytes": 512,
                "installed_size_bytes": 2048,
            }
        )
    return {
        "version": "0.2.0",
        "source_commit": source_commit,
        "created_at": "2026-09-25T00:00:00Z",
        "channel": "ci-candidate",
        "signer_id": "ci-test",
        "platforms": list(PLATFORMS),
        "control_image": "localhost:5000/tldw/control@sha256:" + "d" * 64,
        "artifact_base_url": "https://example.invalid/ci-bundle",
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
        "dependencies": {"lock_digests": {"gateway": "e" * 64}},
        "data": {
            "inventory_schema": 1,
            "migration_generation": 1,
            "rollback_eligible": True,
            "component_catalog_digest": "f" * 64,
        },
    }


@pytest.fixture
def evidence(inventory: dict[str, object]) -> dict[str, object]:
    return {
        "source_commit": inventory["source_commit"],
        "platforms": {
            platform: {
                "G2": True,
                "G4": True,
                "G10": True,
                "G12": True,
                "python_version": "3.12.7",
                "node_version": "24.6.0",
                "link": "https://example.invalid/ci-evidence",
            }
            for platform in PLATFORMS
        },
    }


def _build(
    tmp_path: Path,
    inventory: dict[str, object],
    evidence: dict[str, object],
    signing_key: Ed25519PrivateKey,
) -> tuple[Path, dict[str, bytes]]:
    output = tmp_path / "bundle"
    build_candidate(inventory, evidence, signing_key, output)
    public = signing_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return output, {"ci-test": public}


def test_complete_local_candidate_is_promotable(
    tmp_path: Path,
    inventory: dict[str, object],
    evidence: dict[str, object],
    signing_key: Ed25519PrivateKey,
) -> None:
    output, keys = _build(tmp_path, inventory, evidence, signing_key)

    assert candidate_is_promotable(output, keys, evidence, required_platforms=PLATFORMS)
    assert b"__CONTROL_IMAGE_DIGEST__" not in (output / "start.sh").read_bytes()
    assert not (output / "signing.key").exists()


def test_missing_webui_image_is_not_promotable(
    tmp_path: Path,
    inventory: dict[str, object],
    evidence: dict[str, object],
    signing_key: Ed25519PrivateKey,
) -> None:
    inventory["artifacts"] = [
        artifact
        for artifact in inventory["artifacts"]
        if not (artifact["platform"] == "linux/amd64" and artifact["role"] == "webui")
    ]
    output, keys = _build(tmp_path, inventory, evidence, signing_key)

    assert not candidate_is_promotable(output, keys, evidence, required_platforms=PLATFORMS)


def test_mixed_source_commits_are_rejected_before_signing(
    tmp_path: Path,
    inventory: dict[str, object],
    evidence: dict[str, object],
    signing_key: Ed25519PrivateKey,
) -> None:
    inventory["artifacts"][0]["source_commit"] = "0" * 40

    with pytest.raises(ValueError, match="source commit"):
        _build(tmp_path, inventory, evidence, signing_key)


def test_runtime_patch_mismatch_is_not_promotable(
    tmp_path: Path,
    inventory: dict[str, object],
    evidence: dict[str, object],
    signing_key: Ed25519PrivateKey,
) -> None:
    evidence["platforms"]["linux/arm64"]["node_version"] = "24.0.0"
    output, keys = _build(tmp_path, inventory, evidence, signing_key)

    assert not candidate_is_promotable(output, keys, evidence, required_platforms=PLATFORMS)


def test_unsupported_runtime_family_is_not_promotable(
    tmp_path: Path,
    inventory: dict[str, object],
    evidence: dict[str, object],
    signing_key: Ed25519PrivateKey,
) -> None:
    inventory["compatibility"]["node_version"] = "20.19.0"
    for platform in PLATFORMS:
        evidence["platforms"][platform]["node_version"] = "20.19.0"
    output, keys = _build(tmp_path, inventory, evidence, signing_key)

    assert not candidate_is_promotable(output, keys, evidence, required_platforms=PLATFORMS)


def test_missing_arm64_evidence_is_not_promotable(
    tmp_path: Path,
    inventory: dict[str, object],
    evidence: dict[str, object],
    signing_key: Ed25519PrivateKey,
) -> None:
    del evidence["platforms"]["linux/arm64"]
    output, keys = _build(tmp_path, inventory, evidence, signing_key)

    assert not candidate_is_promotable(output, keys, evidence, required_platforms=PLATFORMS)


def test_tampered_helper_digest_is_not_promotable(
    tmp_path: Path,
    inventory: dict[str, object],
    evidence: dict[str, object],
    signing_key: Ed25519PrivateKey,
) -> None:
    output, keys = _build(tmp_path, inventory, evidence, signing_key)
    (output / "start.sh").write_text((output / "start.sh").read_text() + "\n# tampered\n")

    assert not candidate_is_promotable(output, keys, evidence, required_platforms=PLATFORMS)


def test_corrupted_image_digest_is_rejected_before_signing(
    tmp_path: Path,
    inventory: dict[str, object],
    evidence: dict[str, object],
    signing_key: Ed25519PrivateKey,
) -> None:
    inventory["artifacts"][0]["sha256"] = "0" * 64

    with pytest.raises(ValueError, match="digest"):
        _build(tmp_path, inventory, evidence, signing_key)
