"""Verify same-checkout native artifacts before candidate-only integration.

The caller must obtain both artifacts from successful producer jobs in the same
trusted workflow run. These checks establish integrity and producer consistency,
not an attestation or qualification of any subsequently assembled image.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import stat
import subprocess  # nosec B404

DIRECTORY = Path(__file__).resolve().parent
VERSION = "2.8.4-1~deb13u1+tldw1"
SYSTEM_FILES = {
    f"expat-dbgsym_{VERSION}_amd64.deb",
    f"expat_{VERSION}.debian.tar.xz",
    f"expat_{VERSION}.dsc",
    f"expat_{VERSION}_amd64.buildinfo",
    f"expat_{VERSION}_amd64.changes",
    f"expat_{VERSION}_amd64.deb",
    f"expat_{VERSION}_source.buildinfo",
    f"expat_{VERSION}_source.changes",
    "expat_2.8.4.orig.tar.gz",
    f"libexpat1-dbgsym_{VERSION}_amd64.deb",
    f"libexpat1-dev_{VERSION}_amd64.deb",
    f"libexpat1-udeb_{VERSION}_amd64.udeb",
    f"libexpat1_{VERSION}_amd64.deb",
}
PYTHON_FILES = {"installed-binaries.sha256", "python-install.tar.gz", "python-source.tar.xz"}


def read_record(path: Path) -> str:
    """Read a bounded UTF-8 metadata record, not an arbitrary payload."""
    with path.open("rb") as handle:
        data = handle.read(65537)
    if len(data) > 65536:
        raise ValueError(f"metadata record too large: {path.name}")
    return data.decode("utf-8")


def verify_profile(root: Path, profile: str, commit: str) -> dict:
    """Recheck the existing phase gates and exact expected artifact inventory."""
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"invalid evidence directory: {profile}")
    # Reject links and special files before shell gates can follow/open them.
    for path in root.rglob("*"):
        mode = path.lstat().st_mode
        if not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
            raise ValueError(f"non-regular evidence: {path.relative_to(root)}")
    identity = {}
    for line in read_record(root / "identity/runner.txt").splitlines():
        key, separator, value = line.partition("=")
        if not separator or key in identity:
            raise ValueError("invalid or duplicate runner identity")
        identity[key] = value
    if set(identity) != {"commit", "kernel", "arch", "daemon_arch", "base", "base_arch", "prepared"}:
        raise ValueError("incomplete runner identity")
    if identity["commit"] != commit:
        raise ValueError("qualification checkout differs from expected commit")
    if (
        identity["kernel"] != "Linux"
        or identity["arch"] != "x86_64"
        or identity["daemon_arch"] not in {"amd64", "x86_64"}
        or identity["base_arch"] != "amd64"
        or any(not re.fullmatch(r"sha256:[0-9a-f]{64}", identity[k]) for k in ("base", "prepared"))
    ):
        raise ValueError("invalid native image identity")
    phases = ("prepare", "build", "sanitize", "install") if profile == "system" else ("prepare", "build", "install")
    script = DIRECTORY / ("qualify.sh" if profile == "system" else "python-qualify.sh")
    for phase in phases:
        if read_record(root / phase / "container.exit") != "0\n":
            raise ValueError(f"failed container: {profile}/{phase}")
        # Only the checked-out repository's gate is executable, never the input.
        result = subprocess.run(  # nosec B603
            ["/bin/bash", str(script), "verify-evidence", str(root / phase), phase],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode:
            raise ValueError(f"failed evidence gate: {profile}/{phase}: {result.stderr.strip()}")
    artifacts = root / "build/artifacts"
    expected = SYSTEM_FILES if profile == "system" else PYTHON_FILES
    if {p.name for p in artifacts.iterdir()} != expected | {"SHA256SUMS"}:
        raise ValueError(f"unexpected artifact inventory: {profile}")
    hashes = {}
    for line in read_record(artifacts / "SHA256SUMS").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  \./([^/]+)", line)
        if not match or match[2] not in expected or match[2] in hashes:
            raise ValueError("invalid or duplicate artifact checksum entry")
        hashes[match[2]] = match[1]
    if set(hashes) != expected:
        raise ValueError("incomplete artifact checksum inventory")
    for name, expected_hash in hashes.items():
        path = artifacts / name
        if not path.is_file() or not 0 < path.stat().st_size <= 1024**3:
            raise ValueError(f"invalid artifact size/type: {name}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected_hash:
            raise ValueError(f"artifact checksum mismatch: {name}")
    return {"identity": identity, "artifacts": hashes}


def verify(system: Path, python: Path, commit: str) -> dict:
    """Validate both profiles; do not extract, install, execute or modify inputs."""
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("expected commit must be a full lowercase Git SHA")
    return {
        "scope": "qualified-inputs-only",
        "commit": commit,
        "profiles": {
            "system": verify_profile(system, "system", commit),
            "python": verify_profile(python, "python", commit),
        },
    }


def main() -> None:
    """Emit an inventory only after both sets pass every check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("system", type=Path)
    parser.add_argument("python", type=Path)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args()
    try:
        report = verify(args.system, args.python, args.commit)
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
