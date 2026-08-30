"""Verified, non-secret artifacts for production deployment recovery."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tarfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ARTIFACT_KINDS = frozenset({"postgresql", "redis", "app_data"})
_MANIFEST_KEYS = frozenset(
    {
        "created_at",
        "target_image",
        "rollback_image",
        "compose_file_sha256",
        "artifacts",
    }
)
_ARTIFACT_KEYS = frozenset({"kind", "path", "sha256", "size_bytes"})


@dataclass(frozen=True)
class ArtifactRecord:
    """One checksummed recovery artifact stored beside its manifest."""

    kind: str
    path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class DeploymentManifest:
    """Non-secret record tying images and verified backups to one deployment."""

    created_at: str
    target_image: str
    rollback_image: str
    compose_file_sha256: str
    artifacts: tuple[ArtifactRecord, ...]


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a regular file."""

    if path.is_symlink():
        raise ValueError("artifact must not be a symbolic link")
    if not path.is_file():
        raise ValueError("artifact must be a regular file")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_artifact_name(value: object) -> str:
    """Validate a manifest-relative artifact filename."""

    if not isinstance(value, str) or not value:
        raise ValueError("artifact path must be a non-empty relative filename")
    path = PurePosixPath(value)
    if path.is_absolute() or len(path.parts) != 1 or path.parts[0] in {".", ".."}:
        raise ValueError("artifact path must not contain traversal or directories")
    return value


def _record_from_json(value: object) -> ArtifactRecord:
    """Parse and structurally validate one artifact record."""

    if not isinstance(value, dict) or frozenset(value) != _ARTIFACT_KEYS:
        raise ValueError("artifact record has invalid fields")
    kind = value["kind"]
    if not isinstance(kind, str) or kind not in _ARTIFACT_KINDS:
        raise ValueError("artifact kind is unknown")
    path = _safe_artifact_name(value["path"])
    digest = value["sha256"]
    if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
        raise ValueError("artifact checksum is invalid")
    size = value["size_bytes"]
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise ValueError("artifact size must be greater than zero")
    return ArtifactRecord(kind=kind, path=path, sha256=digest, size_bytes=size)


def _validate_manifest(manifest: DeploymentManifest) -> None:
    """Reject incomplete or ambiguous manifest metadata."""

    if not manifest.created_at or not manifest.target_image or not manifest.rollback_image:
        raise ValueError("manifest metadata must be non-empty")
    if not _SHA256.fullmatch(manifest.compose_file_sha256):
        raise ValueError("manifest compose checksum is invalid")
    seen: set[str] = set()
    for raw_record in manifest.artifacts:
        record = _record_from_json(asdict(raw_record))
        if record.kind in seen:
            raise ValueError("manifest contains a duplicate artifact kind")
        seen.add(record.kind)


def write_manifest(path: Path, manifest: DeploymentManifest) -> None:
    """Write a deterministic owner-only manifest without secret material."""

    _validate_manifest(manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = asdict(manifest)
    data = (json.dumps(body, indent=2, sort_keys=True) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    descriptor = os.open(path, flags, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)


def _manifest_from_json(value: Any) -> DeploymentManifest:
    """Parse a manifest mapping without accepting additional fields."""

    if not isinstance(value, dict) or frozenset(value) != _MANIFEST_KEYS:
        raise ValueError("manifest has invalid fields")
    artifacts_raw = value["artifacts"]
    if not isinstance(artifacts_raw, list):
        raise ValueError("manifest artifacts must be a list")
    scalar_names = (
        "created_at",
        "target_image",
        "rollback_image",
        "compose_file_sha256",
    )
    if any(not isinstance(value[name], str) for name in scalar_names):
        raise ValueError("manifest metadata has invalid types")
    manifest = DeploymentManifest(
        created_at=value["created_at"],
        target_image=value["target_image"],
        rollback_image=value["rollback_image"],
        compose_file_sha256=value["compose_file_sha256"],
        artifacts=tuple(_record_from_json(item) for item in artifacts_raw),
    )
    _validate_manifest(manifest)
    return manifest


def load_verified_manifest(path: Path) -> DeploymentManifest:
    """Load a manifest and verify every referenced artifact in place."""

    if path.is_symlink():
        raise ValueError("manifest must not be a symbolic link")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("manifest could not be read or parsed") from exc
    manifest = _manifest_from_json(value)
    for record in manifest.artifacts:
        artifact = path.parent / record.path
        try:
            size = artifact.stat().st_size
        except OSError as exc:
            raise ValueError("manifest artifact is missing or unreadable") from exc
        if artifact.is_symlink():
            raise ValueError("manifest artifact must not be a symbolic link")
        if not artifact.is_file() or size != record.size_bytes:
            raise ValueError("artifact size does not match manifest")
        if sha256_file(artifact) != record.sha256:
            raise ValueError("artifact checksum does not match manifest")
    return manifest


def verify_tar_archive(path: Path) -> tuple[str, ...]:
    """Inspect an app-data tar without extracting any member."""

    if path.is_symlink():
        raise ValueError("archive must not be a symbolic link")
    names: list[str] = []
    readable_regular_file = False
    try:
        with tarfile.open(path, mode="r:*") as archive:
            for member in archive:
                member_path = PurePosixPath(member.name)
                if (
                    not member.name
                    or member_path.is_absolute()
                    or ".." in member_path.parts
                ):
                    raise ValueError("archive contains an unsafe member path")
                if member.issym() or member.islnk():
                    raise ValueError("archive contains a link member")
                if member.isdev():
                    raise ValueError("archive contains a device member")
                names.append(member.name)
                if member.isfile() and member.size > 0:
                    stream = archive.extractfile(member)
                    if stream is None or not stream.read(1):
                        raise ValueError("archive regular member is unreadable")
                    readable_regular_file = True
    except ValueError:
        raise
    except (OSError, tarfile.TarError) as exc:
        raise ValueError("archive could not be read") from exc
    if not readable_regular_file:
        raise ValueError("archive has no readable nonempty regular member")
    return tuple(names)
