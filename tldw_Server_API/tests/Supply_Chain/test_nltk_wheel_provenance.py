"""Verify the NLTK candidate wheel provenance boundary."""

from __future__ import annotations

import base64
import csv
import hashlib
import importlib.util
import io
import json
import zipfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "Dockerfiles/candidates/nltk/wheel-provenance.py"


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _record_hash(content: bytes) -> str:
    encoded = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=")
    return f"sha256={encoded.decode('ascii')}"


@pytest.fixture
def helper():
    assert SCRIPT.is_file(), "NLTK wheel provenance verifier is not implemented"
    spec = importlib.util.spec_from_file_location("nltk_wheel_provenance", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_record(files: dict[str, bytes], record_name: str) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    for name, content in files.items():
        writer.writerow((name, _record_hash(content), len(content)))
    writer.writerow((record_name, "", ""))
    return stream.getvalue().encode()


def _wheel(
    path: Path,
    *,
    module: bytes = b"VALUE = 2\n",
    version: str = "3.10.3",
    build: str = "1tldw1",
    metadata: bytes | None = None,
    wheel_metadata: bytes | None = None,
    record_mode: str = "valid",
    record_member_replacements: dict[str, str] | None = None,
    directory_members: list[str] | None = None,
    extra_members: list[tuple[str, bytes]] | None = None,
) -> None:
    dist_info = f"nltk-{version}.dist-info"
    record_name = f"{dist_info}/RECORD"
    files = {
        "nltk/__init__.py": module,
        "nltk/VERSION": b"3.10.3\n",
        f"{dist_info}/METADATA": metadata or (f"Metadata-Version: 2.4\nName: nltk\nVersion: {version}\n\n").encode(),
        f"{dist_info}/WHEEL": wheel_metadata
        or (
            "Wheel-Version: 1.0\nGenerator: fixture\nRoot-Is-Purelib: true\n" f"Build: {build}\nTag: py3-none-any\n"
        ).encode(),
    }
    if extra_members:
        files.update(extra_members)
    replacements = record_member_replacements or {}
    recorded_files = {replacements.get(name, name): content for name, content in files.items()}
    record = _write_record(recorded_files, record_name)
    if record_mode == "incorrect":
        record = record.replace(b"sha256=", b"sha256=wrong", 1)

    with zipfile.ZipFile(path, "w") as archive:
        for name in directory_members or []:
            archive.writestr(name, b"")
        for name, content in files.items():
            archive.writestr(name, content)
        if record_mode != "missing":
            archive.writestr(record_name, record)


@pytest.fixture
def wheel_case(tmp_path):
    source = tmp_path / "source"
    package = source / "nltk"
    package.mkdir(parents=True)
    module = b"VALUE = 2\n"
    (package / "__init__.py").write_bytes(module)
    (package / "VERSION").write_bytes(b"3.10.3\n")
    provenance = tmp_path / "source-provenance.json"
    provenance.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "scope": "nltk-candidate-source-not-release-admission",
                "admitted": False,
                "packageRoot": "nltk-3.10.3",
                "backport": {"filename": "backport.patch", "sha256": "d" * 64},
                "changedFiles": [
                    {
                        "path": "nltk/__init__.py",
                        "beforeSha256": "b" * 64,
                        "afterSha256": _sha256(module),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    wheel = tmp_path / "nltk-3.10.3-1tldw1-py3-none-any.whl"
    _wheel(wheel)
    return wheel, source, provenance, tmp_path / "wheel-provenance.json"


def test_verifies_hand_derived_wheel_fixture(wheel_case, helper):
    wheel, source, provenance, output = wheel_case
    _wheel(
        wheel,
        directory_members=[
            "nltk/",
            "nltk-3.10.3.dist-info/",
            "nltk-3.10.3.dist-info/licenses/",
        ],
        extra_members=[("nltk-3.10.3.dist-info/licenses/LICENSE.txt", b"fixture license\n")],
    )

    result = helper.verify_wheel(wheel, source, provenance, output)

    assert json.loads(output.read_text()) == result
    assert result == {
        "schemaVersion": 1,
        "scope": "nltk-candidate-wheel-provenance-not-release-admission",
        "admitted": False,
        "wheel": {"filename": wheel.name, "sha256": _sha256(wheel.read_bytes())},
        "metadata": {
            "name": "nltk",
            "version": "3.10.3",
            "build": "1tldw1",
            "tags": ["py3-none-any"],
        },
        "source": {
            "provenanceFilename": provenance.name,
            "provenanceSha256": _sha256(provenance.read_bytes()),
            "backportSha256": "d" * 64,
        },
        "verifiedModules": [{"path": "nltk/__init__.py", "sha256": _sha256(b"VALUE = 2\n")}],
    }


def test_rejects_changed_wheel_module_even_with_updated_record(wheel_case, helper):
    wheel, source, provenance, output = wheel_case
    _wheel(wheel, module=b"VALUE = 3\n")

    with pytest.raises(ValueError, match="source mismatch"):
        helper.verify_wheel(wheel, source, provenance, output)
    assert not output.exists()


def test_rejects_unrelated_top_level_payload_with_valid_record(wheel_case, helper):
    wheel, source, provenance, output = wheel_case
    _wheel(wheel, extra_members=[("unrelated.txt", b"not candidate payload\n")])

    with pytest.raises(ValueError, match="unexpected wheel payload"):
        helper.verify_wheel(wheel, source, provenance, output)
    assert not output.exists()


def test_rejects_changed_doctest_even_with_updated_record(wheel_case, helper):
    wheel, source, provenance, output = wheel_case
    doctest = source / "nltk/test/example.doctest"
    doctest.parent.mkdir()
    doctest.write_bytes(b">>> 1 + 1\n2\n")
    _wheel(
        wheel,
        extra_members=[("nltk/test/example.doctest", b">>> 1 + 1\n3\n")],
    )

    with pytest.raises(ValueError, match="source mismatch"):
        helper.verify_wheel(wheel, source, provenance, output)
    assert not output.exists()


@pytest.mark.parametrize("record_mode", ["missing", "incorrect"])
def test_rejects_missing_or_incorrect_record(wheel_case, helper, record_mode):
    wheel, source, provenance, output = wheel_case
    _wheel(wheel, record_mode=record_mode)

    with pytest.raises(ValueError, match="RECORD"):
        helper.verify_wheel(wheel, source, provenance, output)
    assert not output.exists()


def test_rejects_duplicate_member(wheel_case, helper):
    wheel, source, provenance, output = wheel_case
    with zipfile.ZipFile(wheel, "a") as archive:
        archive.writestr("nltk/__init__.py", b"duplicate\n")

    with pytest.raises(ValueError, match="duplicate wheel member"):
        helper.verify_wheel(wheel, source, provenance, output)


@pytest.mark.parametrize("name", ["../escape", "/absolute", "nltk\\escape.py"])
def test_rejects_unsafe_member(wheel_case, helper, name):
    wheel, source, provenance, output = wheel_case
    _wheel(wheel, extra_members=[(name, b"unsafe")])

    with pytest.raises(ValueError, match="unsafe wheel member"):
        helper.verify_wheel(wheel, source, provenance, output)


@pytest.mark.parametrize(
    "name",
    [
        "nltk-3.10.3.dist-info/./METADATA",
        "nltk-3.10.3.dist-info//METADATA",
    ],
)
def test_rejects_noncanonical_archive_member(wheel_case, helper, name):
    wheel, source, provenance, output = wheel_case
    _wheel(wheel, extra_members=[(name, b"Name: other\nVersion: 0\n")])

    with pytest.raises(ValueError, match="unsafe wheel member"):
        helper.verify_wheel(wheel, source, provenance, output)
    assert not output.exists()


@pytest.mark.parametrize(
    "name",
    [
        "nltk/./__init__.py",
        "nltk//__init__.py",
    ],
)
def test_rejects_noncanonical_record_member(wheel_case, helper, name):
    wheel, source, provenance, output = wheel_case
    _wheel(
        wheel,
        record_member_replacements={"nltk/__init__.py": name},
    )

    with pytest.raises(ValueError, match="unsafe wheel member"):
        helper.verify_wheel(wheel, source, provenance, output)
    assert not output.exists()


def test_rejects_file_directory_alias(wheel_case, helper):
    wheel, source, provenance, output = wheel_case
    alias = "nltk-3.10.3.dist-info/licenses"
    _wheel(
        wheel,
        directory_members=[f"{alias}/"],
        extra_members=[(alias, b"not a directory\n")],
    )

    with pytest.raises(ValueError, match="alias"):
        helper.verify_wheel(wheel, source, provenance, output)
    assert not output.exists()


@pytest.mark.parametrize(
    ("header", "duplicate"),
    [
        ("Name", "nltk"),
        ("Name", "other"),
        ("Version", "3.10.3"),
        ("Version", "0"),
    ],
)
def test_rejects_repeated_metadata_identity_header(wheel_case, helper, header, duplicate):
    wheel, source, provenance, output = wheel_case
    metadata = ("Metadata-Version: 2.4\n" "Name: nltk\n" "Version: 3.10.3\n" f"{header}: {duplicate}\n\n").encode()
    _wheel(wheel, metadata=metadata)

    with pytest.raises(ValueError, match="METADATA"):
        helper.verify_wheel(wheel, source, provenance, output)
    assert not output.exists()


@pytest.mark.parametrize("duplicate", ["1tldw1", "other"])
def test_rejects_repeated_wheel_build_header(wheel_case, helper, duplicate):
    wheel, source, provenance, output = wheel_case
    wheel_metadata = (
        "Wheel-Version: 1.0\n"
        "Generator: fixture\n"
        "Root-Is-Purelib: true\n"
        "Build: 1tldw1\n"
        f"Build: {duplicate}\n"
        "Tag: py3-none-any\n"
    ).encode()
    _wheel(wheel, wheel_metadata=wheel_metadata)

    with pytest.raises(ValueError, match="Build"):
        helper.verify_wheel(wheel, source, provenance, output)
    assert not output.exists()


@pytest.mark.parametrize(
    "version_headers",
    [
        "",
        "Wheel-Version: malformed\n",
        "Wheel-Version: 1.0\nWheel-Version: 1.0\n",
        "Wheel-Version: 999.0\n",
    ],
)
def test_rejects_missing_malformed_repeated_or_unsupported_wheel_version(wheel_case, helper, version_headers):
    wheel, source, provenance, output = wheel_case
    wheel_metadata = (
        f"{version_headers}" "Generator: fixture\n" "Root-Is-Purelib: true\n" "Build: 1tldw1\n" "Tag: py3-none-any\n"
    ).encode()
    _wheel(wheel, wheel_metadata=wheel_metadata)

    with pytest.raises(ValueError, match="Wheel-Version"):
        helper.verify_wheel(wheel, source, provenance, output)
    assert not output.exists()


@pytest.mark.parametrize(
    ("filename", "version", "build"),
    [
        ("nltk-3.10.2-1tldw1-py3-none-any.whl", "3.10.2", "1tldw1"),
        ("nltk-3.10.3-2local-py3-none-any.whl", "3.10.3", "2local"),
    ],
)
def test_rejects_wrong_version_or_build_tag(wheel_case, helper, filename, version, build):
    old_wheel, source, provenance, output = wheel_case
    wheel = old_wheel.with_name(filename)
    _wheel(wheel, version=version, build=build)

    with pytest.raises(ValueError, match="version|build"):
        helper.verify_wheel(wheel, source, provenance, output)


def test_rejects_unexpected_source_package_file(wheel_case, helper):
    wheel, source, provenance, output = wheel_case
    (source / "nltk/unreviewed.py").write_bytes(b"UNREVIEWED = True\n")

    with pytest.raises(ValueError, match="source inventory"):
        helper.verify_wheel(wheel, source, provenance, output)


def test_rejects_mismatched_changed_file_hash(wheel_case, helper):
    wheel, source, provenance, output = wheel_case
    document = json.loads(provenance.read_text())
    document["changedFiles"][0]["afterSha256"] = "f" * 64
    provenance.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="source-provenance hash mismatch"):
        helper.verify_wheel(wheel, source, provenance, output)
