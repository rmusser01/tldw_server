#!/usr/bin/env python3
"""Prepare a hash-bound NLTK candidate source tree from local inputs only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess  # nosec B404
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

HERE = Path(__file__).resolve().parent
INPUTS_MANIFEST = HERE / "source-inputs.json"
BACKPORT_PATCH = HERE / "backport.patch"
SCOPE = "nltk-candidate-source-not-release-admission"
SHA256_RE = re.compile(r"[0-9a-f]{64}")
COMMIT_RE = re.compile(r"[0-9a-f]{40}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _string(value: object, label: str) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{label} must be a non-empty canonical string")
    return value


def _digest(value: object, label: str) -> str:
    result = _string(value, label)
    if SHA256_RE.fullmatch(result) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return result


def _filename(value: object, label: str) -> str:
    result = _string(value, label)
    if result != Path(result).name or "/" in result or "\\" in result or "\x00" in result:
        raise ValueError(f"{label} must be a plain filename")
    return result


def _relative_path(value: object, label: str) -> str:
    result = _string(value, label)
    pure = PurePosixPath(result)
    if (
        pure.is_absolute()
        or "\\" in result
        or "\x00" in result
        or not pure.parts
        or any(part in {"", ".", ".."} for part in pure.parts)
        or pure.as_posix() != result
    ):
        raise ValueError(f"{label} must be a canonical relative POSIX path")
    return result


def _https_url(value: object, label: str) -> str:
    result = _string(value, label)
    parsed = urlsplit(result)
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise ValueError(f"{label} must be an HTTPS URL without credentials or fragment")
    return result


def _record(value: object, fields: set[str], label: str) -> dict:
    if type(value) is not dict or set(value) != fields:
        raise ValueError(f"{label} fields do not match schema version 1")
    return value


def _records(value: object, label: str, *, nonempty: bool = False) -> list:
    if type(value) is not list or (nonempty and not value):
        raise ValueError(f"{label} must be {'a non-empty' if nonempty else 'a'} list")
    return value


def _regular_file(root: Path, filename: str, label: str) -> Path:
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"{label} root must be a real directory")
    path = root / filename
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise ValueError(f"missing {label}: {filename}") from exc
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise ValueError(f"{label} must be a regular non-symlink file: {filename}")
    return path


def _verified_file(root: Path, record: dict, label: str) -> Path:
    filename = _filename(record["filename"], f"{label}.filename")
    expected = _digest(record["sha256"], f"{label}.sha256")
    path = _regular_file(root, filename, label)
    if _sha256(path) != expected:
        raise ValueError(f"SHA-256 mismatch for {label} {filename}")
    return path


def _load_manifest() -> tuple[dict, dict, list[dict], dict, list[dict]]:
    manifest = json.loads(INPUTS_MANIFEST.read_text(encoding="utf-8"))
    _record(
        manifest,
        {"schemaVersion", "packageRoot", "releaseArtifacts", "preparationInputs", "backport"},
        "manifest",
    )
    if manifest["schemaVersion"] != 1:
        raise ValueError("unknown source-input schema version")
    package_root = _filename(manifest["packageRoot"], "packageRoot")

    release_artifacts = _records(manifest["releaseArtifacts"], "releaseArtifacts", nonempty=True)
    roles: set[str] = set()
    source_record = None
    for index, value in enumerate(release_artifacts):
        record = _record(value, {"role", "filename", "url", "sha256"}, f"releaseArtifacts[{index}]")
        role = _string(record["role"], f"releaseArtifacts[{index}].role")
        if role in roles:
            raise ValueError(f"duplicate release artifact role: {role}")
        roles.add(role)
        _filename(record["filename"], f"releaseArtifacts[{index}].filename")
        _https_url(record["url"], f"releaseArtifacts[{index}].url")
        _digest(record["sha256"], f"releaseArtifacts[{index}].sha256")
        if role == "source":
            source_record = record
    if source_record is None or "baseline-wheel" not in roles:
        raise ValueError("releaseArtifacts must declare source and baseline-wheel")

    preparation_inputs = _records(manifest["preparationInputs"], "preparationInputs", nonempty=True)
    filenames: set[str] = {source_record["filename"]}
    validated_inputs: list[dict] = []
    for index, value in enumerate(preparation_inputs):
        record = _record(
            value,
            {"kind", "filename", "url", "sha256", "commit"},
            f"preparationInputs[{index}]",
        )
        _string(record["kind"], f"preparationInputs[{index}].kind")
        filename = _filename(record["filename"], f"preparationInputs[{index}].filename")
        if filename in filenames:
            raise ValueError(f"duplicate preparation input filename: {filename}")
        filenames.add(filename)
        _https_url(record["url"], f"preparationInputs[{index}].url")
        _digest(record["sha256"], f"preparationInputs[{index}].sha256")
        if COMMIT_RE.fullmatch(_string(record["commit"], f"preparationInputs[{index}].commit")) is None:
            raise ValueError(f"preparationInputs[{index}].commit must be a full commit SHA")
        validated_inputs.append(record)

    backport = _record(manifest["backport"], {"filename", "sha256", "changedFiles"}, "backport")
    _filename(backport["filename"], "backport.filename")
    _digest(backport["sha256"], "backport.sha256")
    changed_files: list[dict] = []
    changed_paths: set[str] = set()
    for index, value in enumerate(_records(backport["changedFiles"], "backport.changedFiles", nonempty=True)):
        record = _record(
            value,
            {"path", "beforeSha256", "afterSha256"},
            f"backport.changedFiles[{index}]",
        )
        path = _relative_path(record["path"], f"backport.changedFiles[{index}].path")
        if path in changed_paths:
            raise ValueError(f"duplicate changed source path: {path}")
        changed_paths.add(path)
        _digest(record["beforeSha256"], f"backport.changedFiles[{index}].beforeSha256")
        _digest(record["afterSha256"], f"backport.changedFiles[{index}].afterSha256")
        changed_files.append(record)
    manifest["packageRoot"] = package_root
    return manifest, source_record, validated_inputs, backport, changed_files


def _validated_members(archive: tarfile.TarFile, package_root: str) -> list[tarfile.TarInfo]:
    members = archive.getmembers()
    seen: set[str] = set()
    folded: dict[str, str] = {}
    for member in members:
        name = member.name
        try:
            canonical = _relative_path(name, "archive member")
        except ValueError as exc:
            raise ValueError(f"unsafe archive member {name!r}") from exc
        parts = PurePosixPath(canonical).parts
        if parts[0] != package_root:
            raise ValueError(f"archive member outside fixed package root: {name!r}")
        if canonical in seen:
            raise ValueError(f"duplicate archive member: {canonical}")
        seen.add(canonical)
        collision_key = canonical.casefold()
        if collision_key in folded and folded[collision_key] != canonical:
            raise ValueError(f"colliding archive members: {folded[collision_key]!r}, {canonical!r}")
        folded[collision_key] = canonical
        if not (member.isfile() or member.isdir()):
            raise ValueError(f"unsafe archive member type for {name!r}")
    return members


def _extract(archive_path: Path, target: Path, package_root: str) -> Path:
    with tarfile.open(archive_path, "r:gz") as archive:
        members = _validated_members(archive, package_root)
        for member in members:
            destination = target.joinpath(*PurePosixPath(member.name).parts)
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"cannot read archive member: {member.name}")
            with source, destination.open("xb") as output:
                shutil.copyfileobj(source, output)
            os.chmod(destination, member.mode & 0o777)
    root = target / package_root
    if not root.is_dir() or root.is_symlink():
        raise ValueError("archive did not produce the fixed package root")
    return root


def _inventory(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for directory, names, filenames in os.walk(root, topdown=True, followlinks=False):
        directory_path = Path(directory)
        for name in names:
            path = directory_path / name
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
                raise ValueError(f"source contains unsafe directory entry: {path.relative_to(root)}")
        for name in filenames:
            path = directory_path / name
            mode = path.lstat().st_mode
            relative = path.relative_to(root).as_posix()
            if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
                raise ValueError(f"source contains unsafe file entry: {relative}")
            result[relative] = _sha256(path)
    return result


def _verify_changed_files(before: dict[str, str], after: dict[str, str], changed_files: list[dict]) -> None:
    if set(before) != set(after):
        raise ValueError("unexpected changed source files: patch changed the file inventory")
    observed = {path for path in before if before[path] != after[path]}
    expected = {record["path"] for record in changed_files}
    if observed != expected:
        raise ValueError(f"unexpected changed source files: expected {sorted(expected)}, observed {sorted(observed)}")
    for record in changed_files:
        path = record["path"]
        if before.get(path) != record["beforeSha256"]:
            raise ValueError(f"source preimage SHA-256 mismatch for {path}")
        if after.get(path) != record["afterSha256"]:
            raise ValueError(f"source postimage SHA-256 mismatch for {path}")


def prepare(inputs: Path, output: Path) -> dict:
    """Validate local inputs, apply the reviewed patch exactly, and record provenance."""
    inputs = Path(inputs)
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"output already exists: {output}")
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise ValueError("output parent must be an existing non-symlink directory")

    manifest, source_record, preparation_inputs, backport, changed_files = _load_manifest()
    source_archive = _verified_file(inputs, source_record, "source archive")
    for index, record in enumerate(preparation_inputs):
        _verified_file(inputs, record, f"preparation input {index}")
    if BACKPORT_PATCH.name != backport["filename"]:
        raise ValueError("backport filename does not match the manifest")
    patch_path = _regular_file(BACKPORT_PATCH.parent, BACKPORT_PATCH.name, "backport patch")
    if _sha256(patch_path) != backport["sha256"]:
        raise ValueError("backport patch SHA-256 mismatch")

    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.prepare-", dir=output.parent))
    try:
        extracted = staging / "extracted"
        extracted.mkdir()
        source_root = _extract(source_archive, extracted, manifest["packageRoot"])
        before = _inventory(source_root)
        patch_executable = shutil.which("patch")
        if patch_executable is None:
            raise ValueError("the required patch utility is unavailable")
        # The executable is resolved above and argv is fixed; no shell is involved.
        process = subprocess.run(  # nosec B603
            [
                patch_executable,
                "--batch",
                "--forward",
                "--fuzz=0",
                "--strip=1",
                "--input",
                str(patch_path.resolve()),
            ],
            cwd=source_root,
            check=False,
            capture_output=True,
            text=True,
        )
        if process.returncode != 0:
            raise ValueError(
                "reviewed backport did not apply exactly: " + (process.stderr.strip() or process.stdout.strip())
            )
        after = _inventory(source_root)
        _verify_changed_files(before, after, changed_files)

        provenance = {
            "schemaVersion": 1,
            "scope": SCOPE,
            "admitted": False,
            "packageRoot": manifest["packageRoot"],
            "inputs": [
                {
                    "kind": "source",
                    "filename": source_record["filename"],
                    "url": source_record["url"],
                    "sha256": source_record["sha256"],
                },
                *preparation_inputs,
            ],
            "backport": {
                "filename": backport["filename"],
                "sha256": backport["sha256"],
            },
            "changedFiles": changed_files,
        }
        final_source = staging / "source"
        source_root.rename(final_source)
        (staging / "source-provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        shutil.rmtree(extracted)
        staging.rename(output)
        return provenance
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.inputs, args.output)


if __name__ == "__main__":
    main()
