"""Prepare pinned Expat candidate metadata; never install or qualify a binary.

Call source verification before extracting archives into a task-owned directory.
The metadata update prepares CPython's own refresh/SBOM tools; it does not replace
their execution, archive signature verification, or native parser tests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SOURCE_SHA256 = {
    "Python-3.12.14.tar.xz": "5c8462af5790baf43a321a1559dbe0db06d1be4300fb85fb53c40060668e548a",
    "expat-2.8.4.tar.gz": "b8ece2437692dad44d851c4532723390a5a330990007706be9c8d2b90d294f36",
    "expat_2.8.4-1.dsc": "b3dc30ff68a32b95746899d2c8e03cfbb5350b982916d649fab179e56ef5ed3e",
    "expat_2.8.4.orig.tar.gz": "a8a9c5cbba9110000b13cc9943f50fcd7e552a5cbad49cb191142c500a0a11b7",
    "expat_2.8.4-1.debian.tar.xz": "a90e0731e6ccdee5f4368a69ab12cf8cc9f5f29e7e61959e2e839d4ca00361fc",
}
BASELINE_SHA256 = "22920a86c83f32300b11463635b71f11137a917975af297725e55525027d4e50"
RELEASE_SHA256 = SOURCE_SHA256["expat-2.8.4.tar.gz"]
BASELINE_URL = "https://github.com/libexpat/libexpat/releases/download/R_2_8_3/expat-2.8.3.tar.gz"
RELEASE_URL = "https://github.com/libexpat/libexpat/releases/download/R_2_8_4/expat-2.8.4.tar.gz"
BASELINE_CPE = "cpe:2.3:a:libexpat_project:libexpat:2.8.3:*:*:*:*:*:*:*"


def verify_sources(directory: Path) -> dict[str, str]:
    """Require every pinned source file; reject absent, aliased or altered inputs."""
    verified = {}
    for filename, expected in SOURCE_SHA256.items():
        path = directory / filename
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"source input must be a regular file: {filename}")
        with path.open("rb") as stream:
            observed = hashlib.file_digest(stream, "sha256").hexdigest()
        if observed != expected:
            raise ValueError(f"source digest mismatch: {filename}")
        verified[filename] = observed
    return verified


def update_python_metadata(root: Path) -> None:
    """Prepare the authenticated 3.12.14 tree for its official Expat refresh.

    Validate all expected baseline metadata before writing either file. This is
    intentionally not idempotent: a reused or unexpected tree needs inspection.
    The caller owns a fresh extracted tree, never a live Python installation.
    """
    refresh_path = root / "Modules/expat/refresh.sh"
    sbom_path = root / "Misc/sbom.spdx.json"
    for path in (refresh_path, sbom_path):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"baseline metadata must be a regular file: {path}")
    refresh = refresh_path.read_text(encoding="utf-8")
    replacements = {
        "expected_libexpat_tag": ("R_2_8_3", "R_2_8_4"),
        "expected_libexpat_version": ("2.8.3", "2.8.4"),
        "expected_libexpat_sha256": (BASELINE_SHA256, RELEASE_SHA256),
    }
    lines = refresh.splitlines(keepends=True)
    for name, (old, new) in replacements.items():
        matches = [index for index, line in enumerate(lines) if line.startswith(name + "=")]
        if len(matches) != 1 or lines[matches[0]] != f'{name}="{old}"\n':
            raise ValueError(f"unexpected refresh baseline assignment: {name}")
        lines[matches[0]] = f'{name}="{new}"\n'

    sbom = json.loads(sbom_path.read_text(encoding="utf-8"))
    packages = [item for item in sbom["packages"] if item.get("name") == "expat"]
    if len(packages) != 1:
        raise ValueError("baseline SBOM must contain exactly one Expat package")
    package = packages[0]
    if (
        package.get("SPDXID") != "SPDXRef-PACKAGE-expat"
        or package.get("versionInfo") != "2.8.3"
        or package.get("downloadLocation") != BASELINE_URL
        or package.get("checksums") != [{"algorithm": "SHA256", "checksumValue": BASELINE_SHA256}]
    ):
        raise ValueError("unexpected Expat baseline SBOM metadata")
    cpes = [ref for ref in package.get("externalRefs", []) if ref.get("referenceType") == "cpe23Type"]
    if len(cpes) != 1 or cpes[0].get("referenceLocator") != BASELINE_CPE:
        raise ValueError("unexpected Expat baseline CPE")
    package.update(
        versionInfo="2.8.4",
        downloadLocation=RELEASE_URL,
        checksums=[{"algorithm": "SHA256", "checksumValue": RELEASE_SHA256}],
    )
    cpes[0]["referenceLocator"] = BASELINE_CPE.replace(":2.8.3:", ":2.8.4:")
    # Do not pretend to regenerate file hashes here: CPython's source SBOM tool
    # must run after refresh.sh has copied and namespaced the actual new sources.
    refresh_path.write_text("".join(lines), encoding="utf-8")
    sbom_path.write_text(json.dumps(sbom, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    """Expose explicit, separate source-verification and metadata-update steps."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify-sources", "update-python-metadata"))
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    if args.command == "verify-sources":
        print(json.dumps(verify_sources(args.directory), indent=2, sort_keys=True))
    else:
        update_python_metadata(args.directory)
        print("Metadata prepared; source refresh, SBOM regeneration and native qualification are still required.")


if __name__ == "__main__":
    main()
