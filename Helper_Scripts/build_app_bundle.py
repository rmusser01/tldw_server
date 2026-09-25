"""Build and sign one local paired-Docker candidate from a measured inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any, Mapping

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from tldw_Server_API.app.core.Release.manifest import verify_manifest


BUNDLE_TEMPLATE = Path(__file__).resolve().parents[1] / "Dockerfiles" / "app-bundle"
BUNDLE_FILES = (
    "README.md",
    "compose.yaml",
    "start.sh",
    "stop.sh",
    "status.sh",
    "start.ps1",
    "stop.ps1",
    "status.ps1",
)
REQUIRED_GATES = ("G2", "G4", "G10", "G12")
_IMAGE_REF = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9._:/-]*@sha256:[0-9a-f]{64}\Z")
_KEY_ID = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}\Z")


def _image_artifacts(inventory: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Accept only measured OCI records built from the one source commit."""
    source_commit = inventory["source_commit"]
    if not isinstance(source_commit, str) or not re.fullmatch(r"[0-9a-f]{40}(?:[0-9a-f]{24})?", source_commit):
        raise ValueError("source commit must be a full Git object ID")
    artifacts = inventory["artifacts"]
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("image artifact inventory is empty")
    cleaned = []
    for artifact in artifacts:
        if not isinstance(artifact, dict) or artifact.get("kind") != "oci":
            raise ValueError("candidate inventory accepts only OCI image artifacts")
        if artifact.get("source_commit") != source_commit:
            raise ValueError("image source commit differs from candidate source commit")
        cleaned.append({key: value for key, value in artifact.items() if key != "source_commit"})
    return cleaned


def _copy_bundle_files(output: Path, inventory: Mapping[str, Any]) -> None:
    """Copy an allowlisted template and embed the fixed bootstrap boundary."""
    control_image = inventory["control_image"]
    signer_id = inventory["signer_id"]
    if not isinstance(control_image, str) or not _IMAGE_REF.fullmatch(control_image):
        raise ValueError("control image must be pinned to a SHA-256 digest")
    if not isinstance(signer_id, str) or not _KEY_ID.fullmatch(signer_id):
        raise ValueError("signer ID is invalid")
    for name in BUNDLE_FILES:
        source = BUNDLE_TEMPLATE / name
        target = output / name
        if name in {"start.sh", "start.ps1"}:
            template = source.read_text()
            if template.count("__CONTROL_IMAGE_DIGEST__") != 1 or template.count("__TRUSTED_KEY_ID__") != 1:
                raise ValueError("start helper bootstrap placeholders are incomplete")
            target.write_text(
                template.replace("__CONTROL_IMAGE_DIGEST__", control_image).replace("__TRUSTED_KEY_ID__", signer_id)
            )
            target.chmod(source.stat().st_mode & 0o777)
        else:
            shutil.copy2(source, target)


def _file_artifacts(output: Path, inventory: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Record the exact generated helper bytes for every advertised platform."""
    base_url = inventory["artifact_base_url"]
    if not isinstance(base_url, str) or not base_url.startswith("https://") or base_url.endswith("/"):
        raise ValueError("artifact base URL must be a canonical HTTPS URL without a trailing slash")
    artifacts = []
    for name in BUNDLE_FILES:
        path = output / name
        contents = path.read_bytes()
        if not contents:
            raise ValueError("bundle file is empty")
        for platform in inventory["platforms"]:
            artifacts.append(
                {
                    "id": f"bundle-{name.replace('.', '-')}-{platform.split('/')[1]}",
                    "kind": "file",
                    "role": "bundle-helper",
                    "platform": platform,
                    "location": f"{base_url}/{name}",
                    "path": name,
                    "size_bytes": len(contents),
                    "installed_size_bytes": len(contents),
                    "sha256": hashlib.sha256(contents).hexdigest(),
                }
            )
    return artifacts


def _qualifications(evidence: Mapping[str, Any], platforms: list[str]) -> dict[str, Any]:
    """Represent missing or failed gates honestly in a provisional candidate."""
    by_platform = evidence.get("platforms", {})
    if not isinstance(by_platform, dict):
        raise ValueError("platform evidence must be an object")
    gates = {
        gate: all(
            isinstance(by_platform.get(platform), dict) and by_platform[platform].get(gate) is True
            for platform in platforms
        )
        for gate in REQUIRED_GATES
    }
    links = sorted(
        {
            entry.get("link")
            for entry in by_platform.values()
            if isinstance(entry, dict) and isinstance(entry.get("link"), str) and entry["link"].startswith("https://")
        }
    )
    if not links:
        raise ValueError("candidate evidence needs at least one HTTPS evidence link")
    return {"gates": gates, "evidence_links": links}


def build_candidate(
    inventory: Mapping[str, Any],
    evidence: Mapping[str, Any],
    signing_key: Ed25519PrivateKey,
    output: Path,
) -> Path:
    """Produce exact manifest bytes, signature, and allowlisted bundle files."""
    if evidence.get("source_commit") != inventory.get("source_commit"):
        raise ValueError("evidence source commit differs from candidate source commit")
    platforms = inventory["platforms"]
    if not isinstance(platforms, list) or not platforms or len(set(platforms)) != len(platforms):
        raise ValueError("candidate platforms are invalid")
    images = _image_artifacts(inventory)
    control_image = inventory["control_image"]
    for platform in platforms:
        controls = [item for item in images if item["platform"] == platform and item["role"] == "control"]
        if len(controls) != 1 or controls[0]["location"] != control_image:
            raise ValueError("control image inventory differs from fixed helper digest")
    output.mkdir(mode=0o700, parents=True, exist_ok=True)
    _copy_bundle_files(output, inventory)
    raw_manifest = {
        "schema_version": 1,
        "version": inventory["version"],
        "source_commit": inventory["source_commit"],
        "created_at": inventory["created_at"],
        "channel": inventory["channel"],
        "signer_id": inventory["signer_id"],
        "platforms": platforms,
        "compatibility": inventory["compatibility"],
        "artifacts": images + _file_artifacts(output, inventory),
        "dependencies": inventory["dependencies"],
        "data": inventory["data"],
        "qualifications": _qualifications(evidence, platforms),
    }
    manifest_bytes = json.dumps(raw_manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    signature = signing_key.sign(manifest_bytes)
    public_key = signing_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    for platform in platforms:
        verify_manifest(manifest_bytes, signature, {inventory["signer_id"]: public_key}, platform=platform)
    (output / "manifest.json").write_bytes(manifest_bytes)
    (output / "manifest.sig").write_bytes(signature)
    return output


def main() -> int:
    """Build a local candidate from JSON inventory/evidence and a raw private key."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--signing-key", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    key = Ed25519PrivateKey.from_private_bytes(args.signing_key.read_bytes())
    build_candidate(json.loads(args.artifacts.read_text()), json.loads(args.evidence.read_text()), key, args.output)
    print(f"Built signed local candidate at {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
