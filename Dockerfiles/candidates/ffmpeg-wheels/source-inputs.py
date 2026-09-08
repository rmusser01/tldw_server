#!/usr/bin/env python3
"""Validate hash-bound source inputs for the candidate FFmpeg wheel builds."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import re
import stat
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

SCHEMA_VERSION = "ffmpeg-wheel-source/v1"
LOCK_FIELDS = {"schema_version", "inputs", "patches", "coverage", "builds", "evidence"}
INPUT_FIELDS = {"name", "url", "sha256", "authentication", "path"}
AUTHENTICATION_FIELDS = {"method", "evidence_paths"}
PATCH_FIELDS = {"commit", "sha256", "path", "requires", "source_paths"}
COVERAGE_FIELDS = {
    "input_id",
    "cve",
    "owner",
    "source_paths",
    "disposition",
    "repair_commits",
    "evidence_paths",
    "regression_id",
}
BUILD_FIELDS = {"owner", "version", "abi", "platform", "tools", "assets", "configuration"}
PLATFORM_FIELDS = {"os", "architecture", "python", "execution"}
CONFIGURATION_FIELDS = {"environment", "evidence_paths"}
EVIDENCE_FIELDS = {"path", "sha256"}
MATCH_FIELDS = {"input_id", "cve", "owner"}
DISPOSITIONS = {"repaired", "already_fixed", "absent_condition"}
EXPECTED_BUILDS = {
    "av": ("18.1.0", "cp311-abi3"),
    "opencv-python": ("5.0.0.93", "cp37-abi3"),
}
EXPECTED_PLATFORM = {
    "os": "linux",
    "architecture": "amd64",
    "python": "3.12",
    "execution": "native",
}
SHA256_RE = re.compile(r"[0-9a-f]{64}")
COMMIT_RE = re.compile(r"[0-9a-f]{40}")
CVE_RE = re.compile(r"CVE-[0-9]{4}-[0-9]{4,}")
IDENTIFIER_RE = re.compile(r"[a-z0-9][a-z0-9._-]*")


def _record(value: object, fields: set[str], label: str) -> dict:
    if type(value) is not dict or set(value) != fields:
        raise ValueError(f"{label} fields do not match the {SCHEMA_VERSION} schema")
    return value


def _records(value: object, label: str, *, nonempty: bool = False) -> list:
    if type(value) is not list or (nonempty and not value):
        requirement = "a non-empty list" if nonempty else "a list"
        raise ValueError(f"{label} must be {requirement}")
    return value


def _string(value: object, label: str, pattern: re.Pattern[str] | None = None) -> str:
    if type(value) is not str or not value or value != value.strip():
        raise ValueError(f"{label} must be a non-empty canonical string")
    if pattern is not None and pattern.fullmatch(value) is None:
        raise ValueError(f"{label} has an invalid value")
    return value


def _strings(
    value: object,
    label: str,
    *,
    nonempty: bool = False,
    pattern: re.Pattern[str] | None = None,
) -> list[str]:
    values = _records(value, label, nonempty=nonempty)
    result = [_string(item, f"{label} item", pattern) for item in values]
    if len(set(result)) != len(result):
        raise ValueError(f"duplicate {label} item")
    return result


def _relative_path(value: object, label: str) -> str:
    path = _string(value, label)
    pure = PurePosixPath(path)
    if (
        "\\" in path
        or "\x00" in path
        or pure.is_absolute()
        or not pure.parts
        or any(part in {"", ".", ".."} for part in pure.parts)
        or pure.as_posix() != path
    ):
        raise ValueError(f"{label} must be a canonical relative POSIX path")
    return path


def _digest(value: object, label: str) -> str:
    return _string(value, label, SHA256_RE)


def _https_url(value: object, label: str) -> str:
    url = _string(value, label)
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise ValueError(f"{label} must be an HTTPS URL without credentials or a fragment")
    return url


def _register_file(files: dict[str, str], path: str, digest: str) -> None:
    if path in files:
        raise ValueError(f"duplicate artifact path: {path}")
    files[path] = digest


def _artifact_file(root: Path, relative: str) -> Path:
    if root.is_symlink():
        raise ValueError("source root must not be a symlink")
    try:
        root_mode = root.lstat().st_mode
    except OSError as exc:
        raise ValueError(f"cannot inspect source root: {exc}") from exc
    if not stat.S_ISDIR(root_mode):
        raise ValueError("source root must be a directory")

    current = root
    parts = PurePosixPath(relative).parts
    for index, part in enumerate(parts):
        current = current / part
        try:
            mode = current.lstat().st_mode
        except OSError as exc:
            raise ValueError(f"missing declared artifact: {relative}") from exc
        if stat.S_ISLNK(mode):
            raise ValueError(f"declared artifact has a symlink path: {relative}")
        if index < len(parts) - 1 and not stat.S_ISDIR(mode):
            raise ValueError(f"declared artifact parent is not a directory: {relative}")
        if index == len(parts) - 1 and not stat.S_ISREG(mode):
            raise ValueError(f"declared artifact is not a regular file: {relative}")
    return current


def _verify_file(root: Path, relative: str, expected: str) -> None:
    path = _artifact_file(root, relative)
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ValueError(f"cannot read declared artifact: {relative}") from exc
    if not hmac.compare_digest(digest.hexdigest(), expected):
        raise ValueError(f"SHA-256 mismatch for {relative}")


def _files_on_disk(root: Path) -> set[str]:
    files: set[str] = set()

    def walk_error(error: OSError) -> None:
        raise ValueError(f"cannot inspect source root: {error}") from error

    for directory, names, filenames in os.walk(root, topdown=True, followlinks=False, onerror=walk_error):
        directory_path = Path(directory)
        for name in names:
            path = directory_path / name
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise ValueError(f"source root contains a symlink: {path.relative_to(root).as_posix()}")
            if not stat.S_ISDIR(mode):
                raise ValueError(f"source root contains a non-directory parent: {path.relative_to(root).as_posix()}")
        for name in filenames:
            path = directory_path / name
            mode = path.lstat().st_mode
            relative = path.relative_to(root).as_posix()
            if stat.S_ISLNK(mode):
                raise ValueError(f"source root contains a symlink: {relative}")
            if not stat.S_ISREG(mode):
                raise ValueError(f"source root contains a nonregular file: {relative}")
            files.add(relative)
    return files


def _validate_evidence(lock: dict, files: dict[str, str]) -> set[str]:
    evidence_paths: set[str] = set()
    for index, value in enumerate(_records(lock["evidence"], "evidence", nonempty=True)):
        record = _record(value, EVIDENCE_FIELDS, f"evidence[{index}]")
        path = _relative_path(record["path"], f"evidence[{index}].path")
        digest = _digest(record["sha256"], f"evidence[{index}].sha256")
        if path in evidence_paths:
            raise ValueError(f"duplicate evidence path: {path}")
        evidence_paths.add(path)
        _register_file(files, path, digest)
    return evidence_paths


def _validate_inputs(
    lock: dict,
    evidence_paths: set[str],
    evidence_references: set[str],
    files: dict[str, str],
) -> tuple[set[str], dict[str, str]]:
    names: set[str] = set()
    hashes: dict[str, str] = {}
    for index, value in enumerate(_records(lock["inputs"], "inputs", nonempty=True)):
        record = _record(value, INPUT_FIELDS, f"input[{index}]")
        name = _string(record["name"], f"input[{index}].name", IDENTIFIER_RE)
        if name in names:
            raise ValueError(f"duplicate input name: {name}")
        names.add(name)
        _https_url(record["url"], f"input[{index}].url")
        path = _relative_path(record["path"], f"input[{index}].path")
        digest = _digest(record["sha256"], f"input[{index}].sha256")
        authentication = _record(
            record["authentication"],
            AUTHENTICATION_FIELDS,
            f"input[{index}] authentication",
        )
        _string(authentication["method"], f"input[{index}].authentication.method", IDENTIFIER_RE)
        references = _strings(
            authentication["evidence_paths"],
            f"input[{index}].authentication.evidence_paths",
            nonempty=True,
        )
        unknown = set(references) - evidence_paths
        if unknown:
            raise ValueError(f"input authentication references unknown evidence: {sorted(unknown)}")
        evidence_references.update(references)
        _register_file(files, path, digest)
        hashes[name] = digest
    return names, hashes


def _validate_patches(lock: dict, files: dict[str, str]) -> tuple[dict[str, dict], dict[str, int]]:
    patches: dict[str, dict] = {}
    positions: dict[str, int] = {}
    digests: set[str] = set()
    for index, value in enumerate(_records(lock["patches"], "patches")):
        record = _record(value, PATCH_FIELDS, f"patch[{index}]")
        commit = _string(record["commit"], f"patch[{index}].commit", COMMIT_RE)
        if commit in patches:
            raise ValueError(f"duplicate patch commit: {commit}")
        path = _relative_path(record["path"], f"patch[{index}].path")
        digest = _digest(record["sha256"], f"patch[{index}].sha256")
        if digest in digests:
            raise ValueError(f"duplicate patch SHA-256: {digest}")
        digests.add(digest)
        requires = _strings(record["requires"], f"patch[{index}].requires", pattern=COMMIT_RE)
        missing = set(requires) - patches.keys()
        if missing:
            raise ValueError(f"patch prerequisite must be declared earlier: {sorted(missing)}")
        source_paths = _strings(record["source_paths"], f"patch[{index}].source_paths", nonempty=True)
        validated_record = dict(record)
        validated_record["_validated_source_paths"] = {
            _relative_path(item, f"patch[{index}].source_paths item") for item in source_paths
        }
        patches[commit] = validated_record
        positions[commit] = index
        _register_file(files, path, digest)
    return patches, positions


def _validated_identity(record: dict, label: str) -> tuple[str, str, str]:
    input_id = _string(record["input_id"], f"{label}.input_id", IDENTIFIER_RE)
    cve = _string(record["cve"], f"{label}.cve", CVE_RE)
    owner = _string(record["owner"], f"{label}.owner", IDENTIFIER_RE)
    if owner not in EXPECTED_BUILDS:
        raise ValueError(f"{label}.owner does not name an owning wheel")
    return input_id, cve, owner


def _validate_matches(original_matches: object) -> set[tuple[str, str, str]]:
    identities: set[tuple[str, str, str]] = set()
    ids: set[str] = set()
    for index, value in enumerate(_records(original_matches, "original matches", nonempty=True)):
        record = _record(value, MATCH_FIELDS, f"original match[{index}]")
        identity = _validated_identity(record, f"original match[{index}]")
        if identity in identities or identity[0] in ids:
            raise ValueError(f"duplicate original match identity: {identity[0]}")
        identities.add(identity)
        ids.add(identity[0])
    return identities


def _validate_coverage(
    lock: dict,
    original_identities: set[tuple[str, str, str]],
    evidence_paths: set[str],
    evidence_references: set[str],
    patches: dict[str, dict],
    patch_positions: dict[str, int],
) -> list[str]:
    identities: set[tuple[str, str, str]] = set()
    ids: set[str] = set()
    regression_ids: set[str] = set()
    for index, value in enumerate(_records(lock["coverage"], "coverage", nonempty=True)):
        record = _record(value, COVERAGE_FIELDS, f"coverage[{index}]")
        identity = _validated_identity(record, f"coverage[{index}]")
        if identity in identities or identity[0] in ids:
            raise ValueError(f"duplicate coverage identity: {identity[0]}")
        identities.add(identity)
        ids.add(identity[0])

        source_paths = {
            _relative_path(item, f"coverage[{index}].source_paths item")
            for item in _strings(record["source_paths"], f"coverage[{index}].source_paths", nonempty=True)
        }
        disposition = _string(record["disposition"], f"coverage[{index}].disposition")
        if disposition not in DISPOSITIONS:
            raise ValueError(f"unknown coverage disposition: {disposition}")
        regression_id = _string(record["regression_id"], f"coverage[{index}].regression_id", IDENTIFIER_RE)
        if regression_id in regression_ids:
            raise ValueError(f"duplicate regression id: {regression_id}")
        regression_ids.add(regression_id)

        references = _strings(
            record["evidence_paths"],
            f"coverage[{index}].evidence_paths",
            nonempty=True,
        )
        unknown_evidence = set(references) - evidence_paths
        if unknown_evidence:
            raise ValueError(f"coverage references unknown evidence: {sorted(unknown_evidence)}")
        evidence_references.update(references)

        repair_commits = _strings(
            record["repair_commits"],
            f"coverage[{index}].repair_commits",
            nonempty=disposition == "repaired",
            pattern=COMMIT_RE,
        )
        if disposition != "repaired" and repair_commits:
            raise ValueError(f"{disposition} coverage must not declare repair commits")
        if disposition == "repaired":
            unknown_repairs = set(repair_commits) - patches.keys()
            if unknown_repairs:
                raise ValueError(f"coverage references unknown repair commits: {sorted(unknown_repairs)}")
            if [patch_positions[item] for item in repair_commits] != sorted(
                patch_positions[item] for item in repair_commits
            ):
                raise ValueError("coverage repair commits are not in declared patch order")
            selected: set[str] = set()
            repaired_paths: set[str] = set()
            for commit in repair_commits:
                missing = set(patches[commit]["requires"]) - selected
                if missing:
                    raise ValueError(f"coverage repair is missing prerequisite commits: {sorted(missing)}")
                selected.add(commit)
                repaired_paths.update(patches[commit]["_validated_source_paths"])
            if not source_paths <= repaired_paths:
                raise ValueError("coverage source path is not bound to its repair commits")

    if identities != original_identities:
        missing = sorted(original_identities - identities)
        extra = sorted(identities - original_identities)
        raise ValueError(f"coverage does not exactly bind original matches; missing={missing}, extra={extra}")
    return sorted(ids)


def _validate_builds(
    lock: dict,
    input_names: set[str],
    evidence_paths: set[str],
    evidence_references: set[str],
) -> None:
    owners: set[str] = set()
    environments: set[str] = set()
    input_references: set[str] = set()
    for index, value in enumerate(_records(lock["builds"], "builds", nonempty=True)):
        record = _record(value, BUILD_FIELDS, f"build[{index}]")
        owner = _string(record["owner"], f"build[{index}].owner", IDENTIFIER_RE)
        if owner not in EXPECTED_BUILDS:
            raise ValueError(f"build[{index}] has an unknown owner")
        if owner in owners:
            raise ValueError(f"duplicate build owner: {owner}")
        owners.add(owner)
        version = _string(record["version"], f"build[{index}].version")
        abi = _string(record["abi"], f"build[{index}].abi")
        if (version, abi) != EXPECTED_BUILDS[owner]:
            raise ValueError(f"build identity mismatch for {owner}")

        platform = _record(record["platform"], PLATFORM_FIELDS, f"build[{index}] platform")
        if platform != EXPECTED_PLATFORM:
            raise ValueError(f"build platform mismatch for {owner}")

        for field in ("tools", "assets"):
            references = _strings(record[field], f"build[{index}].{field}", nonempty=True, pattern=IDENTIFIER_RE)
            unknown = set(references) - input_names
            if unknown:
                raise ValueError(f"build references unknown input: {sorted(unknown)}")
            input_references.update(references)

        configuration = _record(
            record["configuration"],
            CONFIGURATION_FIELDS,
            f"build[{index}] configuration",
        )
        environment = _string(
            configuration["environment"],
            f"build[{index}].configuration.environment",
            IDENTIFIER_RE,
        )
        if environment in environments:
            raise ValueError(f"builds must use separate environments; duplicate {environment}")
        environments.add(environment)
        configuration_evidence = _strings(
            configuration["evidence_paths"],
            f"build[{index}].configuration.evidence_paths",
            nonempty=True,
        )
        unknown_evidence = set(configuration_evidence) - evidence_paths
        if unknown_evidence:
            raise ValueError(f"build configuration references unknown evidence: {sorted(unknown_evidence)}")
        evidence_references.update(configuration_evidence)

    if owners != EXPECTED_BUILDS.keys():
        raise ValueError(f"build records must contain exactly these owners: {sorted(EXPECTED_BUILDS)}")
    if input_references != input_names:
        raise ValueError("build records do not bind the exact declared input set")


def _canonical_lock_sha256(lock: dict) -> str:
    canonical = json.dumps(lock, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def verify_sources(lock: dict, root: Path, original_matches: list[dict]) -> dict:
    """Validate a source lock and return only deterministic byte-binding metadata."""
    lock = _record(lock, LOCK_FIELDS, "source lock")
    if lock["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported source lock schema: {lock['schema_version']!r}")
    root = Path(root)
    files: dict[str, str] = {}
    evidence_references: set[str] = set()
    evidence_paths = _validate_evidence(lock, files)
    input_names, input_hashes = _validate_inputs(lock, evidence_paths, evidence_references, files)
    patches, patch_positions = _validate_patches(lock, files)
    original_identities = _validate_matches(original_matches)
    coverage_ids = _validate_coverage(
        lock,
        original_identities,
        evidence_paths,
        evidence_references,
        patches,
        patch_positions,
    )
    _validate_builds(lock, input_names, evidence_paths, evidence_references)
    if evidence_references != evidence_paths:
        raise ValueError("evidence set contains unreferenced hashed files")
    for path, expected in files.items():
        _verify_file(root, path, expected)
    actual_files = _files_on_disk(root)
    if actual_files != files.keys():
        missing = sorted(files.keys() - actual_files)
        extra = sorted(actual_files - files.keys())
        raise ValueError(f"source root has missing or unreviewed files; missing={missing}, extra={extra}")
    return {
        "schema_version": SCHEMA_VERSION,
        "source_lock_sha256": _canonical_lock_sha256(lock),
        "input_sha256": input_hashes,
        "coverage_ids": coverage_ids,
    }


def main() -> None:
    """Validate JSON inputs and emit one success record to stdout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--original-matches", type=Path, required=True)
    args = parser.parse_args()
    try:
        lock = json.loads(args.lock.read_text(encoding="utf-8"))
        original_matches = json.loads(args.original_matches.read_text(encoding="utf-8"))
        result = verify_sources(lock, args.root, original_matches)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
