"""Contracts for digest-bound release evidence assembly and verification."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from Helper_Scripts.Supply_Chain.release_evidence import (
    EvidenceError,
    build_release_manifest,
    load_image_evidence,
    main,
    verify_release_manifest,
)

PROJECT_IMAGES = {
    "app": ("Dockerfiles/Dockerfile.prod", "promoted"),
    "worker": ("Dockerfiles/Dockerfile.worker", "promoted"),
    "audio-worker": ("Dockerfiles/Dockerfile.audio_gpu_worker", "promoted"),
    "webui": ("Dockerfiles/Dockerfile.webui", "build-and-scan-only"),
    "admin-ui": ("Dockerfiles/Dockerfile.admin-ui", "build-and-scan-only"),
}
REFERENCE_IMAGES = (
    "caddy",
    "postgres",
    "redis",
    "prometheus",
    "alertmanager",
    "grafana",
)
COMPONENT_NAMES = {
    **{name: f"image-{name}" for name in PROJECT_IMAGES},
    **{name: f"reference-{name}" for name in REFERENCE_IMAGES},
    "postgres": "reference-postgresql",
}


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _metadata() -> dict[str, object]:
    scan_started = datetime(2026, 9, 4, 18, 0, tzinfo=timezone.utc)
    return {
        "repository": "rmusser01/tldw_server",
        "source_commit": "a" * 40,
        "release_tag": "v0.1.0-rc.1",
        "workflow_run": "https://github.com/rmusser01/tldw_server/actions/runs/123",
        "platform": "linux/amd64",
        "policy_file": "vulnerability-exceptions.json",
        "scanner": {
            "name": "trivy",
            "version": "0.74.0",
            "image": "ghcr.io/aquasecurity/trivy:0.74.0@sha256:" + "f" * 64,
            "database_updated_at": (scan_started - timedelta(hours=12)).isoformat().replace("+00:00", "Z"),
            "database_downloaded_at": (scan_started - timedelta(hours=1)).isoformat().replace("+00:00", "Z"),
            "scan_started_at": scan_started.isoformat().replace("+00:00", "Z"),
        },
        "decision": "pass",
    }


def _write_image_fixture(
    root: Path,
    *,
    name: str,
    index: int,
    ownership: str,
    dockerfile: str | None,
    publication: str,
) -> None:
    subject_digit = "123456789abcdef"[index]
    child_digit = "123456789abcdef"[index + 1]
    subject = "sha256:" + subject_digit * 64
    child = "sha256:" + child_digit * 64
    reference = f"registry.example/tldw/{name}:v0.1.0@{subject}"
    sbom_file = f"sbom-image-{name}.cdx.json"
    scan_file = f"trivy-image-{name}.json"
    decision_file = f"scan-decision-image-{name}.json"

    _write_json(
        root / sbom_file,
        {
            "bomFormat": "CycloneDX",
            "specVersion": "1.6",
            "metadata": {
                "component": {
                    "type": "container",
                    "name": name,
                    "bom-ref": f"pkg:oci/{name}@{subject}?arch=amd64",
                }
            },
            "components": [{"type": "container", "name": name, "version": subject}],
        },
    )
    _write_json(
        root / scan_file,
        {
            "ArtifactName": reference,
            "ArtifactType": "container_image",
            "Metadata": {"ImageConfig": {"architecture": "amd64"}},
            "Results": [],
        },
    )
    _write_json(
        root / decision_file,
        {
            "component": COMPONENT_NAMES[name],
            "blocking": [],
            "excepted": [],
            "unmatched_exception_ids": [],
        },
    )
    _write_json(
        root / f"image-{name}.json",
        {
            "schema_version": 1,
            "name": name,
            "ownership": ownership,
            "platform": "linux/amd64",
            "subject_digest": subject,
            "platform_manifest_digest": child,
            "subject_media_type": "application/vnd.oci.image.index.v1+json",
            "scan_subject_digest": subject,
            "scan_platform_manifest_digest": child,
            "reference": reference,
            "dockerfile": dockerfile,
            "publication": publication,
            "sbom_file": sbom_file,
            "scan_file": scan_file,
            "decision_file": decision_file,
            "provenance_ref": (
                f"https://github.com/rmusser01/tldw_server/attestations/{name}"
                if ownership == "project-built"
                else None
            ),
        },
    )


def _write_complete_fixture(root: Path) -> None:
    _write_json(root / "vulnerability-exceptions.json", {"schema_version": 1, "exceptions": []})
    index = 0
    for name, (dockerfile, publication) in PROJECT_IMAGES.items():
        _write_image_fixture(
            root,
            name=name,
            index=index,
            ownership="project-built",
            dockerfile=dockerfile,
            publication=publication,
        )
        index += 1
    for name in REFERENCE_IMAGES:
        _write_image_fixture(
            root,
            name=name,
            index=index,
            ownership="third-party-reference",
            dockerfile=None,
            publication="build-and-scan-only",
        )
        index += 1


def _edit_json(path: Path, **changes: object) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(changes)
    _write_json(path, payload)


def test_release_manifest_requires_exact_image_sets(tmp_path: Path) -> None:
    _write_complete_fixture(tmp_path)
    (tmp_path / "image-admin-ui.json").unlink()

    with pytest.raises(EvidenceError, match="admin-ui"):
        build_release_manifest(tmp_path, _metadata())


def test_release_manifest_rejects_duplicate_image_name(tmp_path: Path) -> None:
    _write_complete_fixture(tmp_path)
    _edit_json(tmp_path / "image-admin-ui.json", name="app")

    with pytest.raises(EvidenceError, match="duplicate"):
        build_release_manifest(tmp_path, _metadata())


def test_release_manifest_rejects_wrong_platform(tmp_path: Path) -> None:
    _write_complete_fixture(tmp_path)
    _edit_json(tmp_path / "image-app.json", platform="linux/arm64")

    with pytest.raises(EvidenceError, match="platform"):
        build_release_manifest(tmp_path, _metadata())


def test_release_manifest_rejects_missing_referenced_file(tmp_path: Path) -> None:
    _write_complete_fixture(tmp_path)
    (tmp_path / "sbom-image-app.cdx.json").unlink()

    with pytest.raises(EvidenceError, match="sbom_file"):
        build_release_manifest(tmp_path, _metadata())


def test_release_manifest_detects_file_checksum_tampering(tmp_path: Path) -> None:
    _write_complete_fixture(tmp_path)
    manifest = build_release_manifest(tmp_path, _metadata())
    _write_json(
        tmp_path / "sbom-image-app.cdx.json",
        {"bomFormat": "CycloneDX", "specVersion": "1.6", "components": []},
    )

    with pytest.raises(EvidenceError, match="checksum"):
        verify_release_manifest(manifest, tmp_path)


def test_release_manifest_rejects_malformed_digest(tmp_path: Path) -> None:
    _write_complete_fixture(tmp_path)
    _edit_json(tmp_path / "image-app.json", subject_digest="sha256:abc")

    with pytest.raises(EvidenceError, match="subject_digest"):
        build_release_manifest(tmp_path, _metadata())


def test_release_manifest_rejects_subject_scan_mismatch(tmp_path: Path) -> None:
    _write_complete_fixture(tmp_path)
    _edit_json(tmp_path / "image-app.json", scan_subject_digest="sha256:" + "0" * 64)

    with pytest.raises(EvidenceError, match="scan_subject_digest"):
        build_release_manifest(tmp_path, _metadata())


def test_release_manifest_rejects_raw_scan_subject_mismatch(tmp_path: Path) -> None:
    _write_complete_fixture(tmp_path)
    _edit_json(
        tmp_path / "trivy-image-app.json",
        ArtifactName="registry.example/tldw/app:v0.1.0@sha256:" + "0" * 64,
    )

    with pytest.raises(EvidenceError, match="ArtifactName"):
        build_release_manifest(tmp_path, _metadata())


def test_release_manifest_rejects_sbom_subject_mismatch(tmp_path: Path) -> None:
    _write_complete_fixture(tmp_path)
    sbom_path = tmp_path / "sbom-image-app.cdx.json"
    sbom = json.loads(sbom_path.read_text(encoding="utf-8"))
    sbom["metadata"]["component"]["bom-ref"] = "pkg:oci/app@sha256:" + "0" * 64
    _write_json(sbom_path, sbom)

    with pytest.raises(EvidenceError, match="subject"):
        build_release_manifest(tmp_path, _metadata())


def test_release_manifest_rejects_stale_scanner_database(tmp_path: Path) -> None:
    _write_complete_fixture(tmp_path)
    metadata = _metadata()
    scanner = dict(metadata["scanner"])
    scanner["database_updated_at"] = "2026-09-03T17:59:59Z"
    metadata["scanner"] = scanner

    with pytest.raises(EvidenceError, match="database_updated_at"):
        build_release_manifest(tmp_path, metadata)


def test_release_manifest_rejects_frontend_publication(tmp_path: Path) -> None:
    _write_complete_fixture(tmp_path)
    _edit_json(tmp_path / "image-webui.json", publication="promoted")

    with pytest.raises(EvidenceError, match="publication"):
        build_release_manifest(tmp_path, _metadata())


def test_release_manifest_rejects_third_party_provenance(tmp_path: Path) -> None:
    _write_complete_fixture(tmp_path)
    _edit_json(
        tmp_path / "image-caddy.json",
        provenance_ref="https://github.com/rmusser01/tldw_server/attestations/caddy",
    )

    with pytest.raises(EvidenceError, match="provenance_ref"):
        build_release_manifest(tmp_path, _metadata())


def test_release_manifest_rejects_non_pass_decision(tmp_path: Path) -> None:
    _write_complete_fixture(tmp_path)
    metadata = _metadata()
    metadata["decision"] = "fail"

    with pytest.raises(EvidenceError, match="decision"):
        build_release_manifest(tmp_path, metadata)


def test_index_subject_and_platform_manifest_must_differ(tmp_path: Path) -> None:
    _write_complete_fixture(tmp_path)
    record_path = tmp_path / "image-app.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["platform_manifest_digest"] = record["subject_digest"]
    record["scan_platform_manifest_digest"] = record["subject_digest"]
    _write_json(record_path, record)

    with pytest.raises(EvidenceError, match="platform_manifest_digest"):
        load_image_evidence(record_path)


def test_single_platform_subject_may_equal_platform_manifest(tmp_path: Path) -> None:
    _write_complete_fixture(tmp_path)
    record_path = tmp_path / "image-app.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["subject_media_type"] = "application/vnd.oci.image.manifest.v1+json"
    record["platform_manifest_digest"] = record["subject_digest"]
    record["scan_platform_manifest_digest"] = record["subject_digest"]
    _write_json(record_path, record)

    evidence = load_image_evidence(record_path)

    assert evidence.platform_manifest_digest == evidence.subject_digest


def test_release_manifest_rejects_symlinked_evidence(tmp_path: Path) -> None:
    _write_complete_fixture(tmp_path)
    target = tmp_path / "real-sbom.json"
    (tmp_path / "sbom-image-app.cdx.json").replace(target)
    (tmp_path / "sbom-image-app.cdx.json").symlink_to(target.name)

    with pytest.raises(EvidenceError, match="sbom_file"):
        build_release_manifest(tmp_path, _metadata())


def test_release_manifest_rejects_path_traversal(tmp_path: Path) -> None:
    _write_complete_fixture(tmp_path)
    _edit_json(tmp_path / "image-app.json", sbom_file="../outside.json")

    with pytest.raises(EvidenceError, match="sbom_file"):
        build_release_manifest(tmp_path, _metadata())


def test_release_manifest_rejects_blocking_component_decision(tmp_path: Path) -> None:
    _write_complete_fixture(tmp_path)
    _edit_json(
        tmp_path / "scan-decision-image-app.json",
        blocking=[{"vulnerability_id": "CVE-2026-1000"}],
    )

    with pytest.raises(EvidenceError, match="decision_file"):
        build_release_manifest(tmp_path, _metadata())


def test_release_manifest_is_sorted_and_verifies(tmp_path: Path) -> None:
    _write_complete_fixture(tmp_path)

    manifest = build_release_manifest(tmp_path, _metadata())
    verify_release_manifest(manifest, tmp_path)

    assert [item.name for item in manifest.project_images] == sorted(PROJECT_IMAGES)
    assert [item.name for item in manifest.reference_images] == sorted(REFERENCE_IMAGES)
    assert list(manifest.files) == sorted(manifest.files)


def test_release_manifest_checksums_additional_source_evidence(tmp_path: Path) -> None:
    _write_complete_fixture(tmp_path)
    _write_json(tmp_path / "sbom-source-aggregate.cdx.json", {"bomFormat": "CycloneDX"})

    manifest = build_release_manifest(tmp_path, _metadata())

    assert "sbom-source-aggregate.cdx.json" in manifest.files
    verify_release_manifest(manifest, tmp_path)


def test_release_manifest_verifier_rejects_manifest_identity_change(tmp_path: Path) -> None:
    _write_complete_fixture(tmp_path)
    manifest = build_release_manifest(tmp_path, _metadata())
    changed = replace(manifest.project_images[0], reference="registry.example/tldw/app:v2@sha256:" + "1" * 64)
    tampered = replace(manifest, project_images=(changed, *manifest.project_images[1:]))

    with pytest.raises(EvidenceError, match="manifest"):
        verify_release_manifest(tampered, tmp_path)


def test_release_evidence_cli_assembles_and_verifies(tmp_path: Path) -> None:
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir()
    _write_complete_fixture(evidence_dir)
    metadata_path = tmp_path / "release-metadata.json"
    manifest_path = tmp_path / "release-manifest.json"
    _write_json(metadata_path, _metadata())

    assert (
        main(
            [
                "assemble",
                "--evidence-dir",
                str(evidence_dir),
                "--metadata",
                str(metadata_path),
                "--output",
                str(manifest_path),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "verify",
                "--manifest",
                str(manifest_path),
                "--evidence-dir",
                str(evidence_dir),
            ]
        )
        == 0
    )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["decision"] == "pass"
    assert len(payload["project_images"]) == 5
    assert len(payload["reference_images"]) == 6
