"""Fail-closed preparation of the two Expat candidate source copies."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "Helper_Scripts/Supply_Chain/expat_candidate.py"


def candidate_module():
    """Load the standalone candidate tool without changing import search paths."""
    assert SCRIPT.is_file(), "candidate source preparation is not implemented"
    spec = importlib.util.spec_from_file_location("expat_candidate", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metadata_fixture(tmp_path: Path) -> tuple[Path, Path]:
    """Create the actual metadata fields consumed by CPython's refresh tool."""
    refresh = tmp_path / "Modules/expat/refresh.sh"
    refresh.parent.mkdir(parents=True)
    refresh.write_text(
        '#!/bin/bash\nexpected_libexpat_tag="R_2_8_3"\n'
        'expected_libexpat_version="2.8.3"\n'
        'expected_libexpat_sha256="22920a86c83f32300b11463635b71f11137a917975af297725e55525027d4e50"\n'
        "# Preserve the existing refresh procedure.\n",
        encoding="utf-8",
    )
    sbom = tmp_path / "Misc/sbom.spdx.json"
    sbom.parent.mkdir()
    sbom.write_text(
        json.dumps(
            {
                "packages": [
                    {
                        "name": "expat",
                        "SPDXID": "SPDXRef-PACKAGE-expat",
                        "versionInfo": "2.8.3",
                        "downloadLocation": "https://github.com/libexpat/libexpat/releases/download/R_2_8_3/expat-2.8.3.tar.gz",
                        "checksums": [
                            {
                                "algorithm": "SHA256",
                                "checksumValue": "22920a86c83f32300b11463635b71f11137a917975af297725e55525027d4e50",
                            }
                        ],
                        "externalRefs": [
                            {
                                "referenceType": "cpe23Type",
                                "referenceLocator": "cpe:2.3:a:libexpat_project:libexpat:2.8.3:*:*:*:*:*:*:*",
                            }
                        ],
                    },
                    {"name": "unrelated", "versionInfo": "2.8.3"},
                ],
                "files": [{"fileName": "retained-until-upstream-regeneration"}],
            }
        ),
        encoding="utf-8",
    )
    return refresh, sbom


def test_refresh_and_sbom_use_the_same_complete_release(tmp_path):
    refresh, sbom = metadata_fixture(tmp_path)
    original = json.loads(sbom.read_text())
    candidate_module().update_python_metadata(tmp_path)
    updated = json.loads(sbom.read_text())
    package = updated["packages"][0]
    assert package["versionInfo"] == "2.8.4"
    assert package["checksums"][0]["checksumValue"] == (
        "b8ece2437692dad44d851c4532723390a5a330990007706be9c8d2b90d294f36"
    )
    assert 'expected_libexpat_tag="R_2_8_4"' in refresh.read_text()
    assert 'expected_libexpat_version="2.8.4"' in refresh.read_text()
    assert package["checksums"][0]["checksumValue"] in refresh.read_text()
    assert package["downloadLocation"].endswith("R_2_8_4/expat-2.8.4.tar.gz")
    assert ":2.8.4:" in package["externalRefs"][0]["referenceLocator"]
    assert updated["packages"][1:] == original["packages"][1:]
    assert updated["files"] == original["files"]
    assert refresh.read_text().endswith("# Preserve the existing refresh procedure.\n")


@pytest.mark.parametrize("mutation", ["version", "checksum", "duplicate", "url", "cpe"])
def test_inconsistent_sbom_is_rejected_before_any_write(tmp_path, mutation):
    refresh, sbom = metadata_fixture(tmp_path)
    data = json.loads(sbom.read_text())
    package = data["packages"][0]
    if mutation == "version":
        package["versionInfo"] = "2.8.2"
    elif mutation == "checksum":
        package["checksums"][0]["checksumValue"] = "0" * 64
    elif mutation == "duplicate":
        data["packages"].append(package.copy())
    elif mutation == "url":
        package["downloadLocation"] = "https://invalid.example/expat-2.8.3.tar.gz"
    else:
        package["externalRefs"][0]["referenceLocator"] = "unexpected"
    sbom.write_text(json.dumps(data))
    before = (refresh.read_bytes(), sbom.read_bytes())
    with pytest.raises(ValueError, match="baseline"):
        candidate_module().update_python_metadata(tmp_path)
    assert (refresh.read_bytes(), sbom.read_bytes()) == before


@pytest.mark.parametrize("mutation", ["changed", "duplicate", "missing"])
def test_unexpected_refresh_assignment_is_rejected_before_any_write(tmp_path, mutation):
    refresh, sbom = metadata_fixture(tmp_path)
    text = refresh.read_text()
    assignment = 'expected_libexpat_version="2.8.3"'
    replacement = {
        "changed": 'expected_libexpat_version="2.8.2"',
        "duplicate": assignment + "\n" + assignment,
        "missing": "# version omitted",
    }[mutation]
    refresh.write_text(text.replace(assignment, replacement))
    before = (refresh.read_bytes(), sbom.read_bytes())
    with pytest.raises(ValueError, match="baseline"):
        candidate_module().update_python_metadata(tmp_path)
    assert (refresh.read_bytes(), sbom.read_bytes()) == before


@pytest.mark.parametrize("kind", ["missing", "corrupt", "symlink"])
def test_source_bundle_rejects_untrusted_archive(tmp_path, kind):
    module = candidate_module()
    for filename in module.SOURCE_SHA256:
        path = tmp_path / filename
        if kind == "corrupt":
            path.write_bytes(b"not the pinned source archive")
        elif kind == "symlink":
            target = tmp_path / (filename + ".target")
            target.write_bytes(b"not a regular source input")
            path.symlink_to(target)
    with pytest.raises(ValueError, match="source"):
        module.verify_sources(tmp_path)


def test_source_contract_covers_system_and_bundled_archives():
    assert set(candidate_module().SOURCE_SHA256) == {
        "Python-3.12.14.tar.xz",
        "expat-2.8.4.tar.gz",
        "expat_2.8.4-1.dsc",
        "expat_2.8.4.orig.tar.gz",
        "expat_2.8.4-1.debian.tar.xz",
    }
