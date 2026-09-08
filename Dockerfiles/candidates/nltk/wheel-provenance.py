#!/usr/bin/env python3
"""Bind an NLTK candidate wheel to caller-supplied prepared source."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import re
import stat
import zipfile
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from typing import Any

EXPECTED_NAME = "nltk"
EXPECTED_VERSION = "3.10.3"
EXPECTED_BUILD = "1tldw1"
EXPECTED_TAG = "py3-none-any"
WHEEL_NAME = re.compile(r"^nltk-(?P<version>[^-]+)-(?P<build>[^-]+)-(?P<tag>py3-none-any)\.whl$")


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _record_hash(content: bytes) -> str:
    digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=")
    return f"sha256={digest.decode('ascii')}"


def _safe_member(name: str) -> PurePosixPath:
    member = PurePosixPath(name)
    if (
        not name
        or "\\" in name
        or "\x00" in name
        or member.is_absolute()
        or any(part in {"", ".", ".."} for part in member.parts)
    ):
        raise ValueError(f"unsafe wheel member: {name!r}")
    return member


def _validate_members(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    members: dict[str, zipfile.ZipInfo] = {}
    for info in archive.infolist():
        _safe_member(info.filename.rstrip("/"))
        if info.filename in members:
            raise ValueError(f"duplicate wheel member: {info.filename}")
        mode = info.external_attr >> 16
        file_type = stat.S_IFMT(mode)
        if file_type and file_type not in {stat.S_IFREG, stat.S_IFDIR}:
            raise ValueError(f"unsafe wheel member type: {info.filename}")
        members[info.filename] = info
    return members


def _single_member(files: set[str], suffix: str) -> str:
    matches = sorted(name for name in files if name.endswith(suffix))
    if len(matches) != 1:
        raise ValueError(f"wheel must contain exactly one {suffix}: {matches}")
    return matches[0]


def _verify_record(archive: zipfile.ZipFile, files: set[str], record_name: str) -> None:
    try:
        rows = list(csv.reader(io.StringIO(archive.read(record_name).decode("utf-8"))))
    except (KeyError, UnicodeDecodeError, csv.Error) as exc:
        raise ValueError("missing or malformed RECORD") from exc

    recorded: dict[str, tuple[str, str]] = {}
    for row in rows:
        if len(row) != 3:
            raise ValueError("malformed RECORD row")
        name, digest, size = row
        _safe_member(name)
        if name in recorded:
            raise ValueError(f"duplicate RECORD member: {name}")
        recorded[name] = (digest, size)
    if set(recorded) != files:
        raise ValueError("RECORD inventory does not match wheel members")

    for name, (digest, size) in recorded.items():
        if name == record_name:
            if digest or size:
                raise ValueError("RECORD must not hash itself")
            continue
        content = archive.read(name)
        if digest != _record_hash(content) or size != str(len(content)):
            raise ValueError(f"RECORD hash/size mismatch: {name}")


def _expected_source_files(source: Path) -> set[str]:
    package = source / "nltk"
    if not package.is_dir():
        raise ValueError("prepared source does not contain nltk package")
    expected = {
        path.relative_to(source).as_posix()
        for path in package.rglob("*.py")
        if path.is_file() and "__pycache__" not in path.parts
    }
    version = package / "VERSION"
    if not version.is_file():
        raise ValueError("prepared source is missing nltk/VERSION")
    expected.add("nltk/VERSION")
    expected.update(
        path.relative_to(source).as_posix() for path in (package / "test").glob("*.doctest") if path.is_file()
    )
    return expected


def _verify_source_provenance(source: Path, document: dict[str, Any]) -> None:
    if (
        document.get("schemaVersion") != 1
        or document.get("scope") != "nltk-candidate-source-not-release-admission"
        or document.get("admitted") is not False
        or document.get("packageRoot") != "nltk-3.10.3"
    ):
        raise ValueError("invalid source-provenance identity")
    for entry in document.get("changedFiles", []):
        relative = entry.get("path")
        if not isinstance(relative, str) or _safe_member(relative).parts[0] != "nltk":
            raise ValueError("invalid source-provenance changed file")
        path = source / relative
        if not path.is_file() or _sha256(path) != entry.get("afterSha256"):
            raise ValueError(f"source-provenance hash mismatch: {relative}")


def verify_wheel(wheel: Path, source: Path, source_provenance: Path, output: Path) -> dict[str, Any]:
    """Verify *wheel* against supplied source, then write an evidence record."""
    if output.exists():
        raise FileExistsError(output)
    match = WHEEL_NAME.fullmatch(wheel.name)
    if not match or match["version"] != EXPECTED_VERSION:
        raise ValueError("wheel filename has wrong version")
    if match["build"] != EXPECTED_BUILD:
        raise ValueError("wheel filename has wrong build tag")

    provenance_bytes = source_provenance.read_bytes()
    provenance = json.loads(provenance_bytes)
    _verify_source_provenance(source, provenance)
    backport_sha = provenance.get("backport", {}).get("sha256")
    if not isinstance(backport_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", backport_sha):
        raise ValueError("invalid source-provenance backport hash")

    with zipfile.ZipFile(wheel) as archive:
        members = _validate_members(archive)
        files = {name for name, info in members.items() if not info.is_dir()}
        metadata_name = _single_member(files, ".dist-info/METADATA")
        wheel_metadata_name = _single_member(files, ".dist-info/WHEEL")
        record_name = _single_member(files, ".dist-info/RECORD")
        expected_dist_info = f"nltk-{EXPECTED_VERSION}.dist-info"
        if {PurePosixPath(name).parent.as_posix() for name in (metadata_name, wheel_metadata_name, record_name)} != {
            expected_dist_info
        }:
            raise ValueError("wrong version in dist-info directory")
        _verify_record(archive, files, record_name)
        unexpected_payload = sorted(name for name in files if not name.startswith(("nltk/", f"{expected_dist_info}/")))
        if unexpected_payload:
            raise ValueError(f"unexpected wheel payload members: {unexpected_payload!r}")

        metadata = BytesParser().parsebytes(archive.read(metadata_name))
        if metadata.get("Name") != EXPECTED_NAME or metadata.get("Version") != EXPECTED_VERSION:
            raise ValueError("wheel METADATA has wrong name or version")
        wheel_metadata = BytesParser().parsebytes(archive.read(wheel_metadata_name))
        build = wheel_metadata.get("Build")
        tags = wheel_metadata.get_all("Tag", [])
        if build != EXPECTED_BUILD:
            raise ValueError("wheel WHEEL metadata has wrong build tag")
        if tags != [EXPECTED_TAG]:
            raise ValueError(f"wheel WHEEL metadata has wrong compatibility tags: {tags}")

        wheel_package_files = {name for name in files if name.startswith("nltk/")}
        expected_source_files = _expected_source_files(source)
        if wheel_package_files != expected_source_files:
            raise ValueError(
                "wheel/source inventory mismatch; unexpected source-file changes "
                f"or packaging drift: missing={sorted(expected_source_files - wheel_package_files)!r}, "
                f"extra={sorted(wheel_package_files - expected_source_files)!r}"
            )

        verified_modules = []
        for name in sorted(wheel_package_files):
            wheel_content = archive.read(name)
            if wheel_content != (source / name).read_bytes():
                raise ValueError(f"wheel package source mismatch: {name}")
            if name.endswith(".py"):
                verified_modules.append({"path": name, "sha256": _sha256_bytes(wheel_content)})

        if archive.read("nltk/VERSION").decode("utf-8").strip() != EXPECTED_VERSION:
            raise ValueError("nltk/VERSION has wrong version")

    result = {
        "schemaVersion": 1,
        "scope": "nltk-candidate-wheel-provenance-not-release-admission",
        "admitted": False,
        "wheel": {"filename": wheel.name, "sha256": _sha256(wheel)},
        "metadata": {
            "name": EXPECTED_NAME,
            "version": EXPECTED_VERSION,
            "build": build,
            "tags": tags,
        },
        "source": {
            "provenanceFilename": source_provenance.name,
            "provenanceSha256": _sha256_bytes(provenance_bytes),
            "backportSha256": backport_sha,
        },
        "verifiedModules": verified_modules,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--source-provenance", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = verify_wheel(args.wheel, args.source, args.source_provenance, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
