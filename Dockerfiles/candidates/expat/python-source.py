"""Verify actual refreshed CPython sources and its regenerated upstream SBOM."""

import argparse
import hashlib
import json
from pathlib import Path

OLD_HASH = "22920a86c83f32300b11463635b71f11137a917975af297725e55525027d4e50"
NEW_HASH = "b8ece2437692dad44d851c4532723390a5a330990007706be9c8d2b90d294f36"
EXCLUDED = {"expat_config.h", "pyexpatns.h", "refresh.sh"}


def verify(root: Path, baseline: Path) -> dict:
    """Reject changed ownership boundaries, stale metadata and incomplete hashes."""
    expat = root / "Modules/expat"
    for name in ("expat_config.h", "pyexpatns.h"):
        if (expat / name).read_bytes() != (baseline / name).read_bytes():
            raise ValueError(f"CPython-owned header changed: {name}")
    expected = (baseline / "refresh.sh").read_text()
    for old, new in (("R_2_8_3", "R_2_8_4"), ("2.8.3", "2.8.4"), (OLD_HASH, NEW_HASH)):
        if expected.count('="' + old + '"') != 1:
            raise ValueError("unexpected refresh baseline")
        expected = expected.replace('="' + old + '"', '="' + new + '"')
    if (expat / "refresh.sh").read_text() != expected:
        raise ValueError("upstream refresh procedure changed")
    if (expat / "expat_external.h").read_text().count('#include "pyexpatns.h"') != 1:
        raise ValueError("missing or duplicate namespace include")
    before = json.loads((baseline / "sbom.spdx.json").read_text())
    after = json.loads((root / "Misc/sbom.spdx.json").read_text())
    expected_packages = json.loads(json.dumps(before["packages"]))
    packages = [p for p in expected_packages if p["name"] == "expat"]
    if len(packages) != 1 or packages[0]["versionInfo"] != "2.8.3":
        raise ValueError("unexpected SBOM baseline")
    package = packages[0]
    package.update(
        versionInfo="2.8.4",
        downloadLocation="https://github.com/libexpat/libexpat/releases/download/R_2_8_4/expat-2.8.4.tar.gz",
        checksums=[{"algorithm": "SHA256", "checksumValue": NEW_HASH}],
    )
    for ref in package["externalRefs"]:
        if ref["referenceType"] == "cpe23Type":
            ref["referenceLocator"] = "cpe:2.3:a:libexpat_project:libexpat:2.8.4:*:*:*:*:*:*:*"
    if after["packages"] != expected_packages:
        raise ValueError("unexpected source package identity changes")
    if json.loads((root / "Misc/externals.spdx.json").read_text()) != json.loads(
        (baseline / "externals.spdx.json").read_text()
    ):
        raise ValueError("unrelated external identities changed")
    # Regeneration may reorder entries, but must not change unrelated source files.
    for key, field in (("files", "fileName"), ("relationships", "spdxElementId")):
        prefix = "Modules/expat/" if key == "files" else "SPDXRef-PACKAGE-expat"
        original = [item for item in before[key] if not item[field].startswith(prefix)]
        current = [item for item in after[key] if not item[field].startswith(prefix)]
        if sorted(original, key=lambda x: json.dumps(x, sort_keys=True)) != sorted(
            current, key=lambda x: json.dumps(x, sort_keys=True)
        ):
            raise ValueError("unrelated source SBOM entries changed")
    expected_files = set()
    for path in expat.rglob("*"):
        if path.is_symlink():
            raise ValueError("aliased Expat source")
        if path.is_file() and path.name not in EXCLUDED:
            expected_files.add(path.relative_to(root).as_posix())
    files = [f for f in after["files"] if f["fileName"].startswith("Modules/expat/")]
    if not expected_files or len(files) != len(expected_files) or {f["fileName"] for f in files} != expected_files:
        raise ValueError("incomplete or duplicate Expat file inventory")
    relationships = [r for r in after["relationships"] if r["spdxElementId"] == "SPDXRef-PACKAGE-expat"]
    expected_relationships = []
    for entry in files:
        data = (root / entry["fileName"]).read_bytes()
        if b"\x00" not in data:
            data = data.replace(b"\r\n", b"\n")
        checksums = [
            {"algorithm": "SHA1", "checksumValue": hashlib.sha1(data, usedforsecurity=False).hexdigest()},
            {"algorithm": "SHA256", "checksumValue": hashlib.sha256(data).hexdigest()},
        ]
        if entry["checksums"] != checksums:
            raise ValueError("stale source checksum")
        expected_relationships.append(
            {
                "spdxElementId": "SPDXRef-PACKAGE-expat",
                "relatedSpdxElement": entry["SPDXID"],
                "relationshipType": "CONTAINS",
            }
        )
    if sorted(relationships, key=lambda x: x["relatedSpdxElement"]) != sorted(
        expected_relationships, key=lambda x: x["relatedSpdxElement"]
    ):
        raise ValueError("incomplete source relationships")
    return {"version": "2.8.4", "files": len(files)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("baseline", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.root, args.baseline), sort_keys=True))
