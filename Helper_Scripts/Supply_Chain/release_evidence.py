"""Assemble digest-bound release evidence and verify signed provenance with gh."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess  # nosec B404
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

_PROJECT_IMAGES = {
    "app": ("Dockerfiles/Dockerfile.prod", "promoted"),
    "worker": ("Dockerfiles/Dockerfile.worker", "promoted"),
    "audio-worker": ("Dockerfiles/Dockerfile.audio_gpu_worker", "promoted"),
    "webui": ("Dockerfiles/Dockerfile.webui", "build-and-scan-only"),
    "admin-ui": ("Dockerfiles/Dockerfile.admin-ui", "build-and-scan-only"),
}
_REFERENCE_IMAGES = frozenset(
    {
        "caddy",
        "postgres",
        "redis",
        "prometheus",
        "alertmanager",
        "grafana",
    }
)
_COMPONENT_NAMES = {
    **{name: f"image-{name}" for name in _PROJECT_IMAGES},
    **{name: f"reference-{name}" for name in _REFERENCE_IMAGES},
    "postgres": "reference-postgresql",
}
_IMAGE_FIELDS = frozenset(
    {
        "schema_version",
        "name",
        "ownership",
        "platform",
        "subject_digest",
        "platform_manifest_digest",
        "subject_media_type",
        "scan_subject_digest",
        "scan_platform_manifest_digest",
        "reference",
        "dockerfile",
        "publication",
        "sbom_file",
        "scan_file",
        "decision_file",
        "provenance_ref",
    }
)
_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "repository",
        "source_commit",
        "release_tag",
        "workflow_run",
        "platform",
        "policy_sha256",
        "scanner",
        "project_images",
        "reference_images",
        "files",
        "decision",
    }
)
_MANIFEST_IMAGE_FIELDS = frozenset(
    {
        "name",
        "ownership",
        "platform",
        "subject_digest",
        "platform_manifest_digest",
        "reference",
        "dockerfile",
        "publication",
        "sbom_file",
        "scan_file",
        "decision_file",
        "provenance_ref",
    }
)
_SCANNER_FIELDS = frozenset(
    {
        "name",
        "version",
        "image",
        "database_updated_at",
        "database_downloaded_at",
        "scan_started_at",
    }
)
_METADATA_FIELDS = frozenset(
    {
        "repository",
        "source_commit",
        "release_tag",
        "workflow_run",
        "platform",
        "policy_file",
        "scanner",
        "decision",
    }
)
_DECISION_FIELDS = frozenset(
    {"component", "blocking", "excepted", "unmatched_exception_ids"}
)
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_CHECKSUM = re.compile(r"^[0-9a-f]{64}$")
_IMAGE_REFERENCE = re.compile(r"^[^@\s]+:[^@\s]+@sha256:[0-9a-f]{64}$")
_NAME = re.compile(r"^[a-z][a-z0-9-]{0,63}$")
_REPOSITORY = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_RELEASE_TAG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,127}$")
_SOURCE_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_INDEX_MEDIA_TYPES = frozenset(
    {
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
    }
)
_MANIFEST_MEDIA_TYPES = frozenset(
    {
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    }
)
_MAX_EVIDENCE_BYTES = 64 * 1024 * 1024
_MAX_TEXT_LENGTH = 2048
_POLICY_FILE = "vulnerability-exceptions.json"


class EvidenceError(ValueError):
    """Raised when release evidence is missing, unsafe, or inconsistent."""


@dataclass(frozen=True)
class ImageEvidence:
    """Digest and evidence identity for one release image."""

    name: str
    ownership: str
    platform: str
    subject_digest: str
    platform_manifest_digest: str
    reference: str
    dockerfile: str | None
    publication: str
    sbom_file: str
    scan_file: str
    decision_file: str
    provenance_ref: str | None


@dataclass(frozen=True)
class ReleaseManifest:
    """Complete, checksummed evidence for one admitted release candidate."""

    schema_version: int
    repository: str
    source_commit: str
    release_tag: str
    workflow_run: str
    platform: str
    policy_sha256: str
    scanner: Mapping[str, str]
    project_images: tuple[ImageEvidence, ...]
    reference_images: tuple[ImageEvidence, ...]
    files: Mapping[str, str]
    decision: str


def _error(context: str, field: str) -> EvidenceError:
    return EvidenceError(f"{context} field {field}")


def _duplicate_rejecting_object(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise EvidenceError("evidence field json")
        result[key] = value
    return result


def _checked_file(path: Path, context: str) -> Path:
    try:
        if path.is_symlink() or not path.is_file():
            raise _error(context, "file")
        size = path.stat().st_size
    except OSError as error:
        raise _error(context, "file") from error
    if size <= 0 or size > _MAX_EVIDENCE_BYTES:
        raise _error(context, "size")
    return path


def _load_json_file(path: Path, context: str) -> object:
    checked = _checked_file(path, context)
    try:
        with checked.open("r", encoding="utf-8") as handle:
            return json.load(handle, object_pairs_hook=_duplicate_rejecting_object)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise _error(context, "json") from error


def _evidence_root(path: Path) -> Path:
    try:
        if path.is_symlink() or not path.is_dir():
            raise _error("evidence", "directory")
        return path.resolve(strict=True)
    except OSError as error:
        raise _error("evidence", "directory") from error


def _relative_file(root: Path, value: object, context: str) -> Path:
    if type(value) is not str or not value or len(value) > 256:
        raise _error(context, "file")
    relative = Path(value)
    if relative.is_absolute() or relative.name != value or value in {".", ".."}:
        raise _error(context, "file")
    candidate = root / relative
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as error:
        raise _error(context, "file") from error
    if candidate.is_symlink() or resolved != candidate:
        raise _error(context, "file")
    return _checked_file(candidate, context)


def _string(value: object, context: str, field: str) -> str:
    if (
        type(value) is not str
        or not value.strip()
        or value != value.strip()
        or len(value) > _MAX_TEXT_LENGTH
    ):
        raise _error(context, field)
    return value


def _nullable_string(value: object, context: str, field: str) -> str | None:
    if value is None:
        return None
    return _string(value, context, field)


def _digest(value: object, context: str, field: str) -> str:
    if type(value) is not str or not _DIGEST.fullmatch(value):
        raise _error(context, field)
    return value


def _validate_reference(value: object, subject: str, context: str) -> str:
    reference = _string(value, context, "reference")
    if not _IMAGE_REFERENCE.fullmatch(reference) or not reference.endswith("@" + subject):
        raise _error(context, "reference")
    tag = reference.rpartition("@sha256:")[0].rsplit("/", 1)[-1].rsplit(":", 1)[-1]
    if tag.lower() == "latest":
        raise _error(context, "reference")
    return reference


def _validate_sbom(path: Path, context: str, subject: str) -> None:
    payload = _load_json_file(path, context)
    if not isinstance(payload, Mapping):
        raise _error(context, "root")
    if payload.get("bomFormat") != "CycloneDX":
        raise _error(context, "bomFormat")
    if payload.get("specVersion") not in {"1.5", "1.6", "1.7"}:
        raise _error(context, "specVersion")
    components = payload.get("components")
    if type(components) is not list or not components:
        raise _error(context, "components")
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise _error(context, "metadata")
    component = metadata.get("component")
    if not isinstance(component, Mapping):
        raise _error(context, "metadata.component")
    identities = (component.get("bom-ref"), component.get("purl"))
    if not any(type(identity) is str and f"@{subject}" in identity for identity in identities):
        raise _error(context, "subject")


def _validate_scan(path: Path, context: str, subject: str) -> None:
    payload = _load_json_file(path, context)
    if not isinstance(payload, Mapping):
        raise _error(context, "root")
    artifact_name = _string(payload.get("ArtifactName"), context, "ArtifactName")
    if "@sha256:" in artifact_name and not artifact_name.endswith("@" + subject):
        raise _error(context, "ArtifactName")
    if payload.get("ArtifactType") != "container_image":
        raise _error(context, "ArtifactType")
    if type(payload.get("Results")) is not list:
        raise _error(context, "Results")
    metadata = payload.get("Metadata")
    if not isinstance(metadata, Mapping):
        raise _error(context, "Metadata")
    repo_digests = metadata.get("RepoDigests")
    if repo_digests is None:
        repo_digests = []
    if type(repo_digests) is not list or not all(type(item) is str for item in repo_digests):
        raise _error(context, "RepoDigests")
    image_id = metadata.get("ImageID")
    scan_matches_subject = (
        artifact_name.endswith("@" + subject)
        or image_id == subject
        or any(item.endswith("@" + subject) for item in repo_digests)
    )
    if not scan_matches_subject:
        raise _error(context, "ArtifactName")
    image_config = metadata.get("ImageConfig")
    if not isinstance(image_config, Mapping) or image_config.get("architecture") != "amd64":
        raise _error(context, "architecture")


def _validate_decision(path: Path, context: str, name: str) -> None:
    payload = _load_json_file(path, context)
    if not isinstance(payload, Mapping) or set(payload) != _DECISION_FIELDS:
        raise _error(context, "fields")
    if payload.get("component") != _COMPONENT_NAMES[name]:
        raise _error(context, "component")
    blocking = payload.get("blocking")
    excepted = payload.get("excepted")
    unmatched = payload.get("unmatched_exception_ids")
    if type(blocking) is not list or blocking:
        raise _error(context, "blocking")
    if type(excepted) is not list:
        raise _error(context, "excepted")
    if type(unmatched) is not list or unmatched:
        raise _error(context, "unmatched_exception_ids")


def load_image_evidence(path: Path) -> ImageEvidence:
    """Load and validate one image evidence record and its referenced files."""
    root = _evidence_root(path.parent)
    payload = _load_json_file(path, "image record")
    if not isinstance(payload, Mapping) or set(payload) != _IMAGE_FIELDS:
        raise _error("image record", "fields")
    if payload.get("schema_version") != 1:
        raise _error("image record", "schema_version")

    name = _string(payload.get("name"), "image record", "name")
    if not _NAME.fullmatch(name) or name not in set(_PROJECT_IMAGES) | _REFERENCE_IMAGES:
        raise _error("image record", "name")
    context = f"image {name}"
    ownership = _string(payload.get("ownership"), context, "ownership")
    expected_ownership = "project-built" if name in _PROJECT_IMAGES else "third-party-reference"
    if ownership != expected_ownership:
        raise _error(context, "ownership")
    if payload.get("platform") != "linux/amd64":
        raise _error(context, "platform")

    subject = _digest(payload.get("subject_digest"), context, "subject_digest")
    child = _digest(
        payload.get("platform_manifest_digest"),
        context,
        "platform_manifest_digest",
    )
    media_type = _string(payload.get("subject_media_type"), context, "subject_media_type")
    if media_type in _INDEX_MEDIA_TYPES:
        if subject == child:
            raise _error(context, "platform_manifest_digest")
    elif media_type in _MANIFEST_MEDIA_TYPES:
        if subject != child:
            raise _error(context, "platform_manifest_digest")
    else:
        raise _error(context, "subject_media_type")
    if payload.get("scan_subject_digest") != subject:
        raise _error(context, "scan_subject_digest")
    if payload.get("scan_platform_manifest_digest") != child:
        raise _error(context, "scan_platform_manifest_digest")

    reference = _validate_reference(payload.get("reference"), subject, context)
    dockerfile = _nullable_string(payload.get("dockerfile"), context, "dockerfile")
    publication = _string(payload.get("publication"), context, "publication")
    provenance_ref = _nullable_string(payload.get("provenance_ref"), context, "provenance_ref")
    if name in _PROJECT_IMAGES:
        expected_dockerfile, expected_publication = _PROJECT_IMAGES[name]
        if dockerfile != expected_dockerfile:
            raise _error(context, "dockerfile")
        if publication != expected_publication:
            raise _error(context, "publication")
        if provenance_ref is None or not provenance_ref.startswith("https://github.com/"):
            raise _error(context, "provenance_ref")
        _relative_file(root, f"provenance-image-{name}.jsonl", f"{context} provenance bundle")
        subject_path = _relative_file(root, f"subject-{name}.json", f"{context} provenance subject")
        if "sha256:" + _sha256(subject_path) != subject:
            raise _error(context, "provenance subject")
    else:
        if dockerfile is not None:
            raise _error(context, "dockerfile")
        if publication != "build-and-scan-only":
            raise _error(context, "publication")
        if provenance_ref is not None:
            raise _error(context, "provenance_ref")

    sbom_name = _string(payload.get("sbom_file"), context, "sbom_file")
    scan_name = _string(payload.get("scan_file"), context, "scan_file")
    decision_name = _string(payload.get("decision_file"), context, "decision_file")
    sbom_path = _relative_file(root, sbom_name, f"{context} sbom_file")
    scan_path = _relative_file(root, scan_name, f"{context} scan_file")
    decision_path = _relative_file(root, decision_name, f"{context} decision_file")
    _validate_sbom(sbom_path, f"{context} sbom_file", subject)
    _validate_scan(scan_path, f"{context} scan_file", subject)
    _validate_decision(decision_path, f"{context} decision_file", name)

    return ImageEvidence(
        name=name,
        ownership=ownership,
        platform="linux/amd64",
        subject_digest=subject,
        platform_manifest_digest=child,
        reference=reference,
        dockerfile=dockerfile,
        publication=publication,
        sbom_file=sbom_name,
        scan_file=scan_name,
        decision_file=decision_name,
        provenance_ref=provenance_ref,
    )


def _timestamp(value: object, field: str) -> datetime:
    timestamp = _string(value, "scanner", field)
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as error:
        raise _error("scanner", field) from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise _error("scanner", field)
    return parsed.astimezone(timezone.utc)


def _validate_scanner(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != _SCANNER_FIELDS:
        raise _error("metadata", "scanner")
    scanner = {field: _string(value.get(field), "scanner", field) for field in _SCANNER_FIELDS}
    if scanner["name"] != "trivy":
        raise _error("scanner", "name")
    if not _IMAGE_REFERENCE.fullmatch(scanner["image"]):
        raise _error("scanner", "image")
    updated = _timestamp(scanner["database_updated_at"], "database_updated_at")
    downloaded = _timestamp(scanner["database_downloaded_at"], "database_downloaded_at")
    started = _timestamp(scanner["scan_started_at"], "scan_started_at")
    if updated > downloaded or downloaded > started:
        raise _error("scanner", "database_downloaded_at")
    if started - updated > timedelta(hours=24):
        raise _error("scanner", "database_updated_at")
    return dict(sorted(scanner.items()))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise EvidenceError("evidence field checksum") from error
    return digest.hexdigest()


def _validate_metadata(metadata: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(metadata, Mapping) or set(metadata) != _METADATA_FIELDS:
        raise _error("metadata", "fields")
    repository = _string(metadata.get("repository"), "metadata", "repository")
    if not _REPOSITORY.fullmatch(repository):
        raise _error("metadata", "repository")
    source_commit = _string(metadata.get("source_commit"), "metadata", "source_commit")
    if not _SOURCE_COMMIT.fullmatch(source_commit):
        raise _error("metadata", "source_commit")
    release_tag = _string(metadata.get("release_tag"), "metadata", "release_tag")
    if not _RELEASE_TAG.fullmatch(release_tag):
        raise _error("metadata", "release_tag")
    workflow_run = _string(metadata.get("workflow_run"), "metadata", "workflow_run")
    if metadata.get("platform") != "linux/amd64":
        raise _error("metadata", "platform")
    policy_file = _string(metadata.get("policy_file"), "metadata", "policy_file")
    if policy_file != _POLICY_FILE:
        raise _error("metadata", "policy_file")
    if metadata.get("decision") != "pass":
        raise _error("metadata", "decision")
    return {
        "repository": repository,
        "source_commit": source_commit,
        "release_tag": release_tag,
        "workflow_run": workflow_run,
        "platform": "linux/amd64",
        "policy_file": policy_file,
        "scanner": _validate_scanner(metadata.get("scanner")),
        "decision": "pass",
    }


def _validate_image_sets(
    images: list[ImageEvidence],
) -> tuple[tuple[ImageEvidence, ...], tuple[ImageEvidence, ...]]:
    names = [item.name for item in images]
    if len(names) != len(set(names)):
        raise EvidenceError("image set duplicate name")
    actual = set(names)
    expected = set(_PROJECT_IMAGES) | _REFERENCE_IMAGES
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        detail = missing[0] if missing else extra[0]
        raise EvidenceError(f"image set missing or unexpected {detail}")
    project = tuple(sorted((item for item in images if item.name in _PROJECT_IMAGES), key=lambda item: item.name))
    reference = tuple(sorted((item for item in images if item.name in _REFERENCE_IMAGES), key=lambda item: item.name))
    return project, reference


def _verify_project_provenance(
    root: Path,
    image: ImageEvidence,
    metadata: Mapping[str, object],
    trusted_root: Path | None,
) -> None:
    """Require a valid signature for the exact retained OCI subject and source."""
    context = f"image {image.name} provenance"
    repository = str(metadata["repository"])
    suffixes = {"app": "", "worker": "-worker", "audio-worker": "-audio-worker"}
    subject_name = (
        f"ghcr.io/{repository.lower()}{suffixes[image.name]}"
        if image.name in suffixes else f"local.invalid/tldw/{image.name}"
    )
    if image.reference.split("@", 1)[0].rsplit(":", 1)[0] != subject_name:
        raise _error(context, "subject name")
    # The URL is a navigation aid, never the authority for accepting provenance.
    # Authenticity comes from verifying the retained bundle and subject bytes.
    if not re.fullmatch(
        rf"https://github\.com/{re.escape(repository)}/attestations/[1-9][0-9]*",
        image.provenance_ref or "",
    ):
        raise _error(context, "provenance_ref")
    command = [
        "gh", "attestation", "verify", str(root / f"subject-{image.name}.json"),
        "--bundle", str(root / f"provenance-image-{image.name}.jsonl"),
        "--repo", repository,
        "--signer-workflow", f"{repository}/.github/workflows/publish-docker.yml",
        "--source-digest", str(metadata["source_commit"]),
        "--signer-digest", str(metadata["source_commit"]),
        "--source-ref", f"refs/tags/{metadata['release_tag']}",
        "--predicate-type", "https://slsa.dev/provenance/v1",
        "--deny-self-hosted-runners", "--format", "json",
    ]
    if trusted_root is not None:
        command.extend(["--custom-trusted-root", str(_checked_file(trusted_root, "trusted root"))])
    try:
        # Fixed executable with validated arguments, never passed through a shell.
        result = subprocess.run(  # nosec B603
            command, check=True, capture_output=True, text=True, timeout=60,
        )
        if len(result.stdout) > _MAX_EVIDENCE_BYTES:
            raise _error(context, "verification size")
        verified = json.loads(result.stdout, object_pairs_hook=_duplicate_rejecting_object)
        expected_subject = {
            "name": subject_name,
            "digest": {"sha256": image.subject_digest.removeprefix("sha256:")},
        }
        if type(verified) is not list or len(verified) != 1:
            raise _error(context, "verification result")
        if verified[0]["verificationResult"]["statement"]["subject"] != [expected_subject]:
            raise _error(context, "subject")
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError) as error:
        raise _error(context, "verification") from error


def build_release_manifest(
    evidence_dir: Path,
    metadata: Mapping[str, object],
    *,
    trusted_root: Path | None = None,
) -> ReleaseManifest:
    """Build a stable release manifest after validating every evidence input."""
    root = _evidence_root(evidence_dir)
    validated_metadata = _validate_metadata(metadata)
    try:
        record_paths = sorted(root.glob("image-*.json"), key=lambda item: item.name)
    except OSError as error:
        raise _error("evidence", "records") from error
    record_names: list[str] = []
    for record_path in record_paths:
        record_payload = _load_json_file(record_path, "image record")
        if not isinstance(record_payload, Mapping):
            raise _error("image record", "root")
        record_names.append(_string(record_payload.get("name"), "image record", "name"))
    if len(record_names) != len(set(record_names)):
        raise EvidenceError("image set duplicate name")
    images = [load_image_evidence(path) for path in record_paths]
    for record_path, image in zip(record_paths, images, strict=True):
        if record_path.name != f"image-{image.name}.json":
            raise _error(f"image {image.name}", "record filename")
    project_images, reference_images = _validate_image_sets(images)
    for image in project_images:
        _verify_project_provenance(root, image, validated_metadata, trusted_root)

    policy_path = _relative_file(root, validated_metadata["policy_file"], "policy_file")
    claimed_file_names = {policy_path.name}
    for record_path, image in zip(record_paths, images, strict=True):
        for file_name in (
            record_path.name,
            image.sbom_file,
            image.scan_file,
            image.decision_file,
        ):
            if file_name in claimed_file_names:
                raise EvidenceError("evidence field duplicate file")
            _relative_file(root, file_name, f"image {image.name} file")
            claimed_file_names.add(file_name)
    files_by_name: dict[str, Path] = {}
    try:
        evidence_paths = sorted(root.iterdir(), key=lambda item: item.name)
    except OSError as error:
        raise _error("evidence", "files") from error
    for evidence_path in evidence_paths:
        files_by_name[evidence_path.name] = _checked_file(evidence_path, "evidence file")
    files = {name: _sha256(files_by_name[name]) for name in sorted(files_by_name)}

    return ReleaseManifest(
        schema_version=1,
        repository=validated_metadata["repository"],
        source_commit=validated_metadata["source_commit"],
        release_tag=validated_metadata["release_tag"],
        workflow_run=validated_metadata["workflow_run"],
        platform="linux/amd64",
        policy_sha256=_sha256(policy_path),
        scanner=validated_metadata["scanner"],
        project_images=project_images,
        reference_images=reference_images,
        files=files,
        decision="pass",
    )


def _manifest_image(value: object, context: str) -> ImageEvidence:
    if not isinstance(value, Mapping) or set(value) != _MANIFEST_IMAGE_FIELDS:
        raise _error(context, "fields")
    try:
        return ImageEvidence(**value)
    except TypeError as error:
        raise _error(context, "fields") from error


def _manifest_from_payload(value: object) -> ReleaseManifest:
    if not isinstance(value, Mapping) or set(value) != _MANIFEST_FIELDS:
        raise _error("manifest", "fields")
    project_values = value.get("project_images")
    reference_values = value.get("reference_images")
    files_value = value.get("files")
    scanner_value = value.get("scanner")
    if type(project_values) is not list or type(reference_values) is not list:
        raise _error("manifest", "images")
    if not isinstance(files_value, Mapping) or not isinstance(scanner_value, Mapping):
        raise _error("manifest", "files")
    files: dict[str, str] = {}
    for name, checksum in files_value.items():
        if type(name) is not str or Path(name).name != name:
            raise _error("manifest", "files")
        if type(checksum) is not str or not _CHECKSUM.fullmatch(checksum):
            raise _error("manifest", "files")
        files[name] = checksum
    return ReleaseManifest(
        schema_version=value.get("schema_version"),
        repository=value.get("repository"),
        source_commit=value.get("source_commit"),
        release_tag=value.get("release_tag"),
        workflow_run=value.get("workflow_run"),
        platform=value.get("platform"),
        policy_sha256=value.get("policy_sha256"),
        scanner=dict(scanner_value),
        project_images=tuple(_manifest_image(item, "manifest project_images") for item in project_values),
        reference_images=tuple(
            _manifest_image(item, "manifest reference_images") for item in reference_values
        ),
        files=dict(sorted(files.items())),
        decision=value.get("decision"),
    )


def _manifest_payload(manifest: ReleaseManifest) -> dict[str, object]:
    return {
        "schema_version": manifest.schema_version,
        "repository": manifest.repository,
        "source_commit": manifest.source_commit,
        "release_tag": manifest.release_tag,
        "workflow_run": manifest.workflow_run,
        "platform": manifest.platform,
        "policy_sha256": manifest.policy_sha256,
        "scanner": dict(manifest.scanner),
        "project_images": [asdict(item) for item in manifest.project_images],
        "reference_images": [asdict(item) for item in manifest.reference_images],
        "files": dict(manifest.files),
        "decision": manifest.decision,
    }


def verify_release_manifest(
    manifest: ReleaseManifest,
    evidence_dir: Path,
    *,
    trusted_root: Path | None = None,
) -> None:
    """Recompute every checksum and identity relationship in a release manifest."""
    root = _evidence_root(evidence_dir)
    if manifest.schema_version != 1 or manifest.platform != "linux/amd64" or manifest.decision != "pass":
        raise EvidenceError("manifest identity mismatch")
    if type(manifest.policy_sha256) is not str or not _CHECKSUM.fullmatch(
        manifest.policy_sha256
    ):
        raise EvidenceError("manifest checksum mismatch")
    if not isinstance(manifest.files, Mapping):
        raise EvidenceError("manifest checksum mismatch")
    for name, expected in manifest.files.items():
        if type(name) is not str or type(expected) is not str or not _CHECKSUM.fullmatch(expected):
            raise EvidenceError("manifest checksum mismatch")
        path = _relative_file(root, name, "manifest file")
        if _sha256(path) != expected:
            raise EvidenceError("manifest checksum mismatch")
    if manifest.files.get(_POLICY_FILE) != manifest.policy_sha256:
        raise EvidenceError("manifest policy checksum mismatch")

    metadata = {
        "repository": manifest.repository,
        "source_commit": manifest.source_commit,
        "release_tag": manifest.release_tag,
        "workflow_run": manifest.workflow_run,
        "platform": manifest.platform,
        "policy_file": _POLICY_FILE,
        "scanner": dict(manifest.scanner),
        "decision": manifest.decision,
    }
    rebuilt = build_release_manifest(root, metadata, trusted_root=trusted_root)
    if rebuilt != manifest:
        raise EvidenceError("manifest identity mismatch")


def _write_manifest(path: Path, manifest: ReleaseManifest) -> None:
    try:
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise _error("output", "file")
        if not path.parent.is_dir() or path.parent.is_symlink():
            raise _error("output", "directory")
        path.write_text(
            json.dumps(_manifest_payload(manifest), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except OSError as error:
        raise _error("output", "file") from error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    assemble = subparsers.add_parser("assemble", help="assemble a release evidence manifest")
    assemble.add_argument("--evidence-dir", required=True, type=Path)
    assemble.add_argument("--metadata", required=True, type=Path)
    assemble.add_argument("--output", required=True, type=Path)
    verify = subparsers.add_parser("verify", help="verify a release evidence manifest")
    verify.add_argument("--manifest", required=True, type=Path)
    verify.add_argument("--evidence-dir", required=True, type=Path)
    for command in (assemble, verify):
        command.add_argument(
            "--trusted-root", type=Path,
            help="independently trusted Sigstore root for offline gh verification",
        )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the release evidence command-line interface."""
    arguments = _parser().parse_args(argv)
    if arguments.command == "assemble":
        metadata = _load_json_file(arguments.metadata, "metadata")
        if not isinstance(metadata, Mapping):
            raise _error("metadata", "root")
        manifest = build_release_manifest(
            arguments.evidence_dir, metadata, trusted_root=arguments.trusted_root,
        )
        _write_manifest(arguments.output, manifest)
        return 0

    payload = _load_json_file(arguments.manifest, "manifest")
    manifest = _manifest_from_payload(payload)
    verify_release_manifest(manifest, arguments.evidence_dir, trusted_root=arguments.trusted_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EvidenceError",
    "ImageEvidence",
    "ReleaseManifest",
    "build_release_manifest",
    "load_image_evidence",
    "main",
    "verify_release_manifest",
]
