"""Exercise the fail-closed NLTK candidate source preparation boundary."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import tarfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "Dockerfiles/candidates/nltk/prepare.py"


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


@pytest.fixture
def helper():
    assert SCRIPT.is_file(), "NLTK candidate source preparer is not implemented"
    spec = importlib.util.spec_from_file_location("nltk_candidate_prepare", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _archive(path: Path, members: list[tuple[str, bytes | None, bytes]]) -> None:
    """Write controlled tar members as (name, content, type)."""
    with tarfile.open(path, "w:gz") as archive:
        for name, content, member_type in members:
            info = tarfile.TarInfo(name)
            info.type = member_type
            if member_type == tarfile.DIRTYPE:
                info.mode = 0o755
                archive.addfile(info)
            elif member_type in {tarfile.SYMTYPE, tarfile.LNKTYPE}:
                info.linkname = "pkg-1.0/pkg/model.py"
                archive.addfile(info)
            else:
                payload = content or b""
                info.mode = 0o644
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))


def _patch(extra: bool = False) -> bytes:
    result = b"""diff --git a/pkg/model.py b/pkg/model.py
--- a/pkg/model.py
+++ b/pkg/model.py
@@ -1 +1 @@
-VALUE = 1
+VALUE = 2
"""
    if extra:
        result += b"""diff --git a/pkg/unreviewed.py b/pkg/unreviewed.py
new file mode 100644
--- /dev/null
+++ b/pkg/unreviewed.py
@@ -0,0 +1 @@
+UNREVIEWED = True
"""
    return result


@pytest.fixture
def source_case(tmp_path, helper, monkeypatch):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    archive = inputs / "pkg-1.0.tar.gz"
    _archive(
        archive,
        [
            ("pkg-1.0", None, tarfile.DIRTYPE),
            ("pkg-1.0/pkg", None, tarfile.DIRTYPE),
            ("pkg-1.0/pkg/model.py", b"VALUE = 1\n", tarfile.REGTYPE),
            ("pkg-1.0/LICENSE", b"fixture license\n", tarfile.REGTYPE),
        ],
    )
    upstream = inputs / "upstream.patch"
    upstream.write_bytes(b"immutable upstream patch input\n")
    backport = tmp_path / "backport.patch"
    backport.write_bytes(_patch())
    manifest_path = tmp_path / "source-inputs.json"
    manifest = {
        "schemaVersion": 1,
        "packageRoot": "pkg-1.0",
        "releaseArtifacts": [
            {
                "role": "source",
                "filename": archive.name,
                "url": "https://example.invalid/pkg-1.0.tar.gz",
                "sha256": _sha256(archive),
            },
            {
                "role": "baseline-wheel",
                "filename": "pkg-1.0-py3-none-any.whl",
                "url": "https://example.invalid/pkg-1.0-py3-none-any.whl",
                "sha256": "f" * 64,
            },
        ],
        "preparationInputs": [
            {
                "kind": "upstream-patch",
                "filename": upstream.name,
                "url": "https://example.invalid/upstream.patch",
                "sha256": _sha256(upstream),
                "commit": "a" * 40,
            }
        ],
        "backport": {
            "filename": backport.name,
            "sha256": _sha256(backport),
            "changedFiles": [
                {
                    "path": "pkg/model.py",
                    "beforeSha256": _sha256_bytes(b"VALUE = 1\n"),
                    "afterSha256": _sha256_bytes(b"VALUE = 2\n"),
                }
            ],
        },
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(helper, "INPUTS_MANIFEST", manifest_path)
    monkeypatch.setattr(helper, "BACKPORT_PATCH", backport)
    return inputs, tmp_path / "output", manifest, manifest_path, backport


def test_prepares_controlled_source_and_records_exact_provenance(source_case, helper):
    inputs, output, manifest, _, _ = source_case
    result = helper.prepare(inputs, output)

    assert (output / "source/pkg/model.py").read_bytes() == b"VALUE = 2\n"
    assert (output / "source/LICENSE").read_bytes() == b"fixture license\n"
    assert json.loads((output / "source-provenance.json").read_text()) == result
    assert result == {
        "schemaVersion": 1,
        "scope": "nltk-candidate-source-not-release-admission",
        "admitted": False,
        "packageRoot": "pkg-1.0",
        "inputs": [
            {
                "kind": "source",
                "filename": "pkg-1.0.tar.gz",
                "url": "https://example.invalid/pkg-1.0.tar.gz",
                "sha256": manifest["releaseArtifacts"][0]["sha256"],
            },
            manifest["preparationInputs"][0],
        ],
        "backport": {
            "filename": "backport.patch",
            "sha256": manifest["backport"]["sha256"],
        },
        "changedFiles": manifest["backport"]["changedFiles"],
    }


def test_rejects_mutated_source_archive(source_case, helper):
    inputs, output, _, _, _ = source_case
    (inputs / "pkg-1.0.tar.gz").write_bytes(b"substituted source")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        helper.prepare(inputs, output)
    assert not output.exists()


def test_rejects_mutated_local_backport(source_case, helper):
    inputs, output, _, _, backport = source_case
    backport.write_bytes(backport.read_bytes() + b"# substitution\n")
    with pytest.raises(ValueError, match="backport.*SHA-256 mismatch"):
        helper.prepare(inputs, output)
    assert not output.exists()


def test_rejects_missing_upstream_patch_input(source_case, helper):
    inputs, output, _, _, _ = source_case
    (inputs / "upstream.patch").unlink()
    with pytest.raises(ValueError, match="missing.*upstream.patch"):
        helper.prepare(inputs, output)
    assert not output.exists()


def test_rejects_unreviewed_source_edit(source_case, helper):
    inputs, output, manifest, manifest_path, backport = source_case
    backport.write_bytes(_patch(extra=True))
    manifest["backport"]["sha256"] = _sha256(backport)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="unexpected changed source files"):
        helper.prepare(inputs, output)
    assert not output.exists()


@pytest.mark.parametrize(
    ("name", "member_type"),
    [
        ("pkg-1.0/../escape", tarfile.REGTYPE),
        ("pkg-1.0/pkg/link", tarfile.SYMTYPE),
        ("pkg-1.0/pkg/hardlink", tarfile.LNKTYPE),
    ],
)
def test_rejects_unsafe_archive_members(source_case, helper, name, member_type):
    inputs, output, manifest, manifest_path, _ = source_case
    archive = inputs / "pkg-1.0.tar.gz"
    _archive(
        archive,
        [
            ("pkg-1.0", None, tarfile.DIRTYPE),
            (name, b"unsafe", member_type),
        ],
    )
    manifest["releaseArtifacts"][0]["sha256"] = _sha256(archive)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="archive member"):
        helper.prepare(inputs, output)
    assert not output.exists()


def test_rejects_duplicate_archive_members(source_case, helper):
    inputs, output, manifest, manifest_path, _ = source_case
    archive = inputs / "pkg-1.0.tar.gz"
    _archive(
        archive,
        [
            ("pkg-1.0", None, tarfile.DIRTYPE),
            ("pkg-1.0/pkg/model.py", b"first", tarfile.REGTYPE),
            ("pkg-1.0/pkg/model.py", b"second", tarfile.REGTYPE),
        ],
    )
    manifest["releaseArtifacts"][0]["sha256"] = _sha256(archive)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate archive member"):
        helper.prepare(inputs, output)
    assert not output.exists()


def test_rejects_existing_output_without_modifying_it(tmp_path, source_case, helper):
    inputs, output, _, _, _ = source_case
    output.mkdir()
    sentinel = output / "sentinel"
    sentinel.write_bytes(b"keep")

    with pytest.raises(FileExistsError):
        helper.prepare(inputs, output)
    assert sentinel.read_bytes() == b"keep"
