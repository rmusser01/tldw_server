"""Authenticate and validate one paired application release manifest."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?\Z")
_VERSION = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?\Z")
_PLATFORM = re.compile(r"[a-z0-9_]+/[a-z0-9_]+\Z")
_TOP_FIELDS = frozenset(
    {
        "schema_version",
        "version",
        "source_commit",
        "created_at",
        "channel",
        "signer_id",
        "platforms",
        "compatibility",
        "artifacts",
        "dependencies",
        "data",
        "qualifications",
    }
)
_COMPATIBILITY_FIELDS = frozenset(
    {
        "min_launcher",
        "python_version",
        "node_version",
        "backend_generation",
        "browser_generation",
        "allowed_upgrade_sources",
        "components",
    }
)
_ARTIFACT_FIELDS = frozenset(
    {
        "id",
        "kind",
        "role",
        "platform",
        "location",
        "size_bytes",
        "installed_size_bytes",
        "sha256",
    }
)
_DATA_FIELDS = frozenset(
    {
        "inventory_schema",
        "migration_generation",
        "rollback_eligible",
        "component_catalog_digest",
    }
)


class ManifestError(ValueError):
    """A release manifest or artifact cannot be trusted."""


@dataclass(frozen=True)
class Artifact:
    """One file or immutable OCI artifact named by the release."""

    id: str
    kind: str
    role: str
    platform: str
    location: str
    size_bytes: int
    installed_size_bytes: int
    sha256: str
    path: str | None = None
    image_digest: str | None = None


@dataclass(frozen=True)
class ReleaseManifest:
    """Validated, signed application release metadata."""

    version: str
    source_commit: str
    created_at: str
    channel: str
    signer_id: str
    platforms: tuple[str, ...]
    compatibility: Mapping[str, Any]
    artifacts: tuple[Artifact, ...]
    dependencies: Mapping[str, Any]
    data: Mapping[str, Any]
    qualifications: Mapping[str, Any]


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ManifestError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _object(value: Any, fields: frozenset[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or frozenset(value) != fields:
        raise ManifestError(f"{label} fields are incomplete or unknown")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ManifestError(f"{label} must be a nonempty string")
    return value


def _positive_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ManifestError(f"{label} must be a positive integer")
    return value


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ManifestError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _version(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _VERSION.fullmatch(value):
        raise ManifestError(f"{label} must be a version")
    return value


def _strings(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ManifestError(f"{label} must be a nonempty list")
    values = tuple(_string(item, label) for item in value)
    if len(set(values)) != len(values):
        raise ManifestError(f"{label} contains duplicates")
    return values


def _artifact_path(value: Any) -> str:
    path = _string(value, "artifact path")
    parts = path.split("/")
    if (
        path.startswith("/")
        or "\\" in path
        or "\x00" in path
        or any(part in {"", ".", ".."} for part in parts)
        or ":" in parts[0]
    ):
        raise ManifestError("artifact path is unsafe")
    return path


def _artifact(value: Any, platforms: tuple[str, ...]) -> Artifact:
    if not isinstance(value, dict):
        raise ManifestError("artifact must be an object")
    kind = value.get("kind")
    if kind not in {"file", "oci"}:
        raise ManifestError("artifact kind is unsupported")
    required = _ARTIFACT_FIELDS | ({"path"} if kind == "file" else {"image_digest"})
    item = _object(value, frozenset(required), "artifact")
    platform = _string(item["platform"], "artifact platform")
    if platform not in platforms:
        raise ManifestError("artifact platform is not in release platforms")
    digest = _digest(item["sha256"], "artifact digest")
    location = _string(item["location"], "artifact location")
    if kind == "file":
        if not location.startswith("https://"):
            raise ManifestError("artifact download location must use HTTPS")
        path = _artifact_path(item["path"])
        image_digest = None
    else:
        path = None
        image_digest = _string(item["image_digest"], "artifact image digest")
        if image_digest != f"sha256:{digest}" or not location.endswith(f"@{image_digest}"):
            raise ManifestError("artifact image location must pin its digest")
    return Artifact(
        id=_string(item["id"], "artifact id"),
        kind=kind,
        role=_string(item["role"], "artifact role"),
        platform=platform,
        location=location,
        size_bytes=_positive_int(item["size_bytes"], "artifact size"),
        installed_size_bytes=_positive_int(item["installed_size_bytes"], "artifact installed size"),
        sha256=digest,
        path=path,
        image_digest=image_digest,
    )


def _manifest(value: Any, *, platform: str, current_version: str | None) -> ReleaseManifest:
    item = _object(value, _TOP_FIELDS, "manifest")
    if item["schema_version"] != 1 or isinstance(item["schema_version"], bool):
        raise ManifestError("unsupported manifest schema version")
    version = _version(item["version"], "release version")
    commit = _string(item["source_commit"], "source commit")
    if not _COMMIT.fullmatch(commit):
        raise ManifestError("source commit must be a full Git object ID")
    created_at = _string(item["created_at"], "creation time")
    try:
        if datetime.fromisoformat(created_at.replace("Z", "+00:00")).tzinfo is None:
            raise ValueError("missing timezone")
    except ValueError as exc:
        raise ManifestError("creation time must have a timezone") from exc
    platforms = _strings(item["platforms"], "platforms")
    if any(not _PLATFORM.fullmatch(target) for target in platforms):
        raise ManifestError("invalid platform tuple")
    if platform not in platforms:
        raise ManifestError("release does not support platform")
    compatibility = _object(item["compatibility"], _COMPATIBILITY_FIELDS, "compatibility")
    _version(compatibility["min_launcher"], "minimum launcher")
    _version(compatibility["python_version"], "Python version")
    _version(compatibility["node_version"], "Node version")
    _positive_int(compatibility["backend_generation"], "backend generation")
    _positive_int(compatibility["browser_generation"], "browser generation")
    allowed_sources = _strings(compatibility["allowed_upgrade_sources"], "upgrade sources")
    for source in allowed_sources:
        _version(source, "upgrade source")
    _strings(compatibility["components"], "components")
    if current_version is not None and current_version not in allowed_sources:
        raise ManifestError("current version is not an allowed upgrade source")
    artifacts_raw = item["artifacts"]
    if not isinstance(artifacts_raw, list) or not artifacts_raw:
        raise ManifestError("artifacts must be a nonempty list")
    artifacts = tuple(_artifact(raw, platforms) for raw in artifacts_raw)
    ids = [artifact.id for artifact in artifacts]
    if len(set(ids)) != len(ids):
        raise ManifestError("duplicate artifact ID")
    file_paths: dict[str, tuple[str, int, int, set[str]]] = {}
    for artifact in artifacts:
        if artifact.path is None:
            continue
        prior = file_paths.get(artifact.path)
        if prior is None:
            file_paths[artifact.path] = (
                artifact.sha256,
                artifact.size_bytes,
                artifact.installed_size_bytes,
                {artifact.platform},
            )
        elif (
            artifact.sha256 != prior[0]
            or artifact.size_bytes != prior[1]
            or artifact.installed_size_bytes != prior[2]
            or artifact.platform in prior[3]
        ):
            raise ManifestError("duplicate artifact path has conflicting contents or platform")
        else:
            prior[3].add(artifact.platform)
    dependencies = _object(item["dependencies"], frozenset({"lock_digests"}), "dependencies")
    locks = dependencies["lock_digests"]
    if not isinstance(locks, dict) or not locks:
        raise ManifestError("dependency locks are required")
    for name, digest in locks.items():
        _string(name, "dependency lock name")
        _digest(digest, "dependency lock digest")
    data = _object(item["data"], _DATA_FIELDS, "data")
    _positive_int(data["inventory_schema"], "inventory schema")
    _positive_int(data["migration_generation"], "migration generation")
    if not isinstance(data["rollback_eligible"], bool):
        raise ManifestError("rollback eligibility must be boolean")
    _digest(data["component_catalog_digest"], "component catalog digest")
    qualifications = _object(item["qualifications"], frozenset({"gates", "evidence_links"}), "qualifications")
    gates = qualifications["gates"]
    if (
        not isinstance(gates, dict)
        or not gates
        or any(
            not re.fullmatch(r"G(?:[1-9]|1[0-2])", key) or not isinstance(passed, bool) for key, passed in gates.items()
        )
    ):
        raise ManifestError("qualification gates are invalid")
    _strings(qualifications["evidence_links"], "evidence links")
    return ReleaseManifest(
        version=version,
        source_commit=commit,
        created_at=created_at,
        channel=_string(item["channel"], "release channel"),
        signer_id=_string(item["signer_id"], "signer ID"),
        platforms=platforms,
        compatibility=compatibility,
        artifacts=artifacts,
        dependencies=dependencies,
        data=data,
        qualifications=qualifications,
    )


def verify_manifest(
    manifest_bytes: bytes,
    signature_bytes: bytes,
    trusted_keys: Mapping[str, bytes],
    *,
    platform: str,
    current_version: str | None = None,
) -> ReleaseManifest:
    """Verify an exact-byte Ed25519 signature, then validate release contents."""
    if not isinstance(manifest_bytes, bytes) or len(manifest_bytes) > 1024 * 1024:
        raise ManifestError("manifest bytes are invalid or oversized")
    if not isinstance(signature_bytes, bytes) or len(signature_bytes) != 64:
        raise ManifestError("signature must contain 64 raw bytes")
    try:
        parsed = json.loads(manifest_bytes.decode("utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ManifestError("manifest is not valid UTF-8 JSON") from exc
    if not isinstance(parsed, dict):
        raise ManifestError("manifest must be a JSON object")
    signer_id = _string(parsed.get("signer_id"), "signer ID")
    key_bytes = trusted_keys.get(signer_id)
    if key_bytes is None:
        raise ManifestError("manifest signer is not trusted")
    try:
        Ed25519PublicKey.from_public_bytes(key_bytes).verify(signature_bytes, manifest_bytes)
    except (ValueError, InvalidSignature) as exc:
        raise ManifestError("manifest signature is invalid") from exc
    return _manifest(parsed, platform=platform, current_version=current_version)


def verify_artifact(path: Path, artifact: Artifact) -> None:
    """Check a regular file against its signed size and SHA-256 digest."""
    if artifact.kind != "file":
        raise ManifestError("artifact is not a file")
    if path.is_symlink():
        raise ManifestError("artifact file is unsafe")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ManifestError("artifact file is missing or unsafe") from exc
    try:
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_size != artifact.size_bytes:
            raise ManifestError("artifact size or file type does not match manifest")
        digest = hashlib.sha256()
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != artifact.sha256:
            raise ManifestError("artifact digest does not match manifest")
    finally:
        os.close(descriptor)
