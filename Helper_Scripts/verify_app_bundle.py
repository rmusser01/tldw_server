"""Refuse qualification of incomplete or tampered paired-Docker candidates."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from tldw_Server_API.app.core.Release.manifest import (
    ManifestError,
    verify_artifact,
    verify_manifest,
)


REQUIRED_ROLES = {"backend", "webui", "gateway", "control"}
REQUIRED_GATES = ("G2", "G4", "G10", "G12")


def candidate_is_promotable(
    bundle: Path,
    trusted_keys: Mapping[str, bytes],
    evidence: Mapping[str, Any],
    *,
    required_platforms: Sequence[str],
) -> bool:
    """Check signed contents and per-platform evidence without changing state."""
    try:
        manifest_bytes = (bundle / "manifest.json").read_bytes()
        signature = (bundle / "manifest.sig").read_bytes()
        if not required_platforms or len(set(required_platforms)) != len(required_platforms):
            return False
        platform_evidence_map = evidence.get("platforms")
        if not isinstance(platform_evidence_map, dict):
            return False
        for platform in required_platforms:
            manifest = verify_manifest(manifest_bytes, signature, trusted_keys, platform=platform)
            if evidence.get("source_commit") != manifest.source_commit:
                return False
            if not set(required_platforms).issubset(set(manifest.platforms)):
                return False
            if any(manifest.qualifications["gates"].get(gate) is not True for gate in REQUIRED_GATES):
                return False
            if (
                re.fullmatch(r"3\.12\.[0-9]+", manifest.compatibility["python_version"]) is None
                or re.fullmatch(r"24\.[0-9]+\.[0-9]+", manifest.compatibility["node_version"]) is None
            ):
                return False
            platform_evidence = platform_evidence_map.get(platform)
            if not isinstance(platform_evidence, dict):
                return False
            if any(platform_evidence.get(gate) is not True for gate in REQUIRED_GATES):
                return False
            if (
                platform_evidence.get("python_version") != manifest.compatibility["python_version"]
                or platform_evidence.get("node_version") != manifest.compatibility["node_version"]
                or platform_evidence.get("link") not in manifest.qualifications["evidence_links"]
            ):
                return False
            selected = [artifact for artifact in manifest.artifacts if artifact.platform == platform]
            roles = [artifact.role for artifact in selected if artifact.kind == "oci"]
            if len(roles) != len(REQUIRED_ROLES) or set(roles) != REQUIRED_ROLES:
                return False
            for artifact in selected:
                if artifact.kind == "file":
                    current = bundle
                    for part in Path(artifact.path).parts[:-1]:
                        current = current / part
                        if current.is_symlink():
                            return False
                    verify_artifact(bundle / artifact.path, artifact)
        return True
    except (OSError, KeyError, TypeError, ValueError, ManifestError):
        return False


def main() -> int:
    """Check a local candidate with public keys supplied by the trusted job."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--signature", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--trusted-key-id", required=True)
    parser.add_argument("--trusted-key-file", type=Path, required=True)
    parser.add_argument("--platform", action="append", required=True)
    args = parser.parse_args()
    if args.signature.parent != args.manifest.parent:
        parser.error("manifest and signature must be in the same bundle directory")
    evidence = json.loads(args.evidence.read_text())
    trusted_keys = {args.trusted_key_id: args.trusted_key_file.read_bytes()}
    if not candidate_is_promotable(args.manifest.parent, trusted_keys, evidence, required_platforms=args.platform):
        print("Candidate qualification failed")
        return 1
    print("Candidate qualification passed for " + ", ".join(args.platform))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
