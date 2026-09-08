"""Exercise the deterministic FFmpeg wheel source-input gate."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "Dockerfiles/candidates/ffmpeg-wheels/source-inputs.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def helper():
    assert SCRIPT.is_file(), "FFmpeg wheel source-input gate is not implemented"
    spec = importlib.util.spec_from_file_location("ffmpeg_wheel_source_inputs", SCRIPT)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.fixture
def source_case(tmp_path):
    root = tmp_path / "source-root"
    payloads = {
        "inputs/ffmpeg.tar.xz": b"synthetic ffmpeg source\n",
        "inputs/av.tar.gz": b"synthetic av source\n",
        "inputs/opencv.tar.gz": b"synthetic opencv source\n",
        "inputs/av-toolchain.json": b'{"tool":"av"}\n',
        "inputs/opencv-toolchain.json": b'{"tool":"opencv"}\n',
        "patches/prerequisite.patch": b"synthetic prerequisite patch\n",
        "patches/repair.patch": b"synthetic repair patch\n",
        "evidence/ffmpeg-auth.txt": b"trusted acquisition recorded signature verification\n",
        "evidence/av-auth.json": b'{"source":"index metadata"}\n',
        "evidence/opencv-auth.json": b'{"source":"index metadata"}\n',
        "evidence/toolchain-auth.json": b'{"source":"registry metadata"}\n',
        "evidence/repaired.json": b'{"result":"repaired"}\n',
        "evidence/already-fixed.json": b'{"result":"already fixed"}\n',
        "evidence/absent-condition.json": b'{"result":"absent condition"}\n',
        "evidence/av-build.txt": b"isolated av build configuration\n",
        "evidence/opencv-build.txt": b"isolated opencv build configuration\n",
    }
    for relative, content in payloads.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    evidence_paths = [path for path in payloads if path.startswith("evidence/")]
    evidence = [{"path": path, "sha256": _sha256(root / path)} for path in evidence_paths]

    def source_input(name, path, method, authentication_evidence):
        return {
            "name": name,
            "url": f"https://example.invalid/{Path(path).name}",
            "sha256": _sha256(root / path),
            "authentication": {"method": method, "evidence_paths": authentication_evidence},
            "path": path,
        }

    lock = {
        "schema_version": "ffmpeg-wheel-source/v1",
        "inputs": [
            source_input("ffmpeg-source", "inputs/ffmpeg.tar.xz", "detached-signature", ["evidence/ffmpeg-auth.txt"]),
            source_input("av-source", "inputs/av.tar.gz", "index-metadata", ["evidence/av-auth.json"]),
            source_input("opencv-source", "inputs/opencv.tar.gz", "index-metadata", ["evidence/opencv-auth.json"]),
            source_input(
                "av-toolchain",
                "inputs/av-toolchain.json",
                "registry-metadata",
                ["evidence/toolchain-auth.json"],
            ),
            source_input(
                "opencv-toolchain",
                "inputs/opencv-toolchain.json",
                "registry-metadata",
                ["evidence/toolchain-auth.json"],
            ),
        ],
        "patches": [
            {
                "commit": "1" * 40,
                "sha256": _sha256(root / "patches/prerequisite.patch"),
                "path": "patches/prerequisite.patch",
                "requires": [],
                "source_paths": ["libavcodec/example.c"],
            },
            {
                "commit": "2" * 40,
                "sha256": _sha256(root / "patches/repair.patch"),
                "path": "patches/repair.patch",
                "requires": ["1" * 40],
                "source_paths": ["libavcodec/example.c"],
            },
        ],
        "coverage": [
            {
                "input_id": "match-repaired",
                "cve": "CVE-2026-10001",
                "owner": "av",
                "source_paths": ["libavcodec/example.c"],
                "disposition": "repaired",
                "repair_commits": ["1" * 40, "2" * 40],
                "evidence_paths": ["evidence/repaired.json"],
                "regression_id": "regression-repaired",
            },
            {
                "input_id": "match-already-fixed",
                "cve": "CVE-2026-10002",
                "owner": "av",
                "source_paths": ["libavcodec/already.c"],
                "disposition": "already_fixed",
                "repair_commits": [],
                "evidence_paths": ["evidence/already-fixed.json"],
                "regression_id": "regression-already-fixed",
            },
            {
                "input_id": "match-absent",
                "cve": "CVE-2026-10003",
                "owner": "opencv-python",
                "source_paths": ["libavcodec/absent.c"],
                "disposition": "absent_condition",
                "repair_commits": [],
                "evidence_paths": ["evidence/absent-condition.json"],
                "regression_id": "regression-absent",
            },
        ],
        "builds": [
            {
                "owner": "av",
                "version": "18.1.0",
                "abi": "cp311-abi3",
                "platform": {"os": "linux", "architecture": "amd64", "python": "3.12", "execution": "native"},
                "tools": ["av-toolchain"],
                "assets": ["ffmpeg-source", "av-source"],
                "configuration": {
                    "environment": "av-native",
                    "evidence_paths": ["evidence/av-build.txt"],
                },
            },
            {
                "owner": "opencv-python",
                "version": "5.0.0.93",
                "abi": "cp37-abi3",
                "platform": {"os": "linux", "architecture": "amd64", "python": "3.12", "execution": "native"},
                "tools": ["opencv-toolchain"],
                "assets": ["ffmpeg-source", "opencv-source"],
                "configuration": {
                    "environment": "opencv-native",
                    "evidence_paths": ["evidence/opencv-build.txt"],
                },
            },
        ],
        "evidence": evidence,
    }
    matches = [
        {"input_id": "match-repaired", "cve": "CVE-2026-10001", "owner": "av"},
        {"input_id": "match-already-fixed", "cve": "CVE-2026-10002", "owner": "av"},
        {"input_id": "match-absent", "cve": "CVE-2026-10003", "owner": "opencv-python"},
    ]
    return lock, root, matches


def test_valid_sources_bind_all_dispositions_and_builds(source_case, helper):
    lock, root, matches = source_case
    result = helper.verify_sources(lock, root, matches)
    assert result["schema_version"] == "ffmpeg-wheel-source/v1"
    assert result["input_sha256"] == {item["name"]: item["sha256"] for item in lock["inputs"]}
    assert result["coverage_ids"] == ["match-absent", "match-already-fixed", "match-repaired"]
    assert len(result["source_lock_sha256"]) == 64


def test_canonical_lock_hash_ignores_mapping_key_order(source_case, helper):
    lock, root, matches = source_case
    reordered = {key: copy.deepcopy(lock[key]) for key in reversed(lock)}
    expected = helper.verify_sources(lock, root, matches)["source_lock_sha256"]
    assert helper.verify_sources(lock, root, matches)["source_lock_sha256"] == expected
    assert helper.verify_sources(reordered, root, matches)["source_lock_sha256"] == expected


def test_missing_component_coverage_fails(source_case, helper):
    lock, root, matches = source_case
    lock["coverage"].pop()
    with pytest.raises(ValueError, match="coverage"):
        helper.verify_sources(lock, root, matches)


@pytest.mark.parametrize(
    ("target", "field"),
    [
        ("lock", "unexpected"),
        ("input", "unexpected"),
        ("authentication", "signer"),
        ("patch", "unexpected"),
        ("coverage", "unexpected"),
        ("build", "unexpected"),
        ("platform", "unexpected"),
        ("configuration", "unexpected"),
        ("evidence", "unexpected"),
        ("match", "unexpected"),
    ],
)
def test_unknown_fields_fail_closed(source_case, helper, target, field):
    lock, root, matches = source_case
    targets = {
        "lock": lock,
        "input": lock["inputs"][0],
        "authentication": lock["inputs"][0]["authentication"],
        "patch": lock["patches"][0],
        "coverage": lock["coverage"][0],
        "build": lock["builds"][0],
        "platform": lock["builds"][0]["platform"],
        "configuration": lock["builds"][0]["configuration"],
        "evidence": lock["evidence"][0],
        "match": matches[0],
    }
    targets[target][field] = "unreviewed"
    with pytest.raises(ValueError, match="fields"):
        helper.verify_sources(lock, root, matches)


@pytest.mark.parametrize("value", ["ffmpeg-wheel-source/v2", 1, None])
def test_unknown_or_malformed_schema_fails(source_case, helper, value):
    lock, root, matches = source_case
    lock["schema_version"] = value
    with pytest.raises(ValueError, match="schema"):
        helper.verify_sources(lock, root, matches)


@pytest.mark.parametrize("value", ["../outside", "/absolute", "inputs/../source", "inputs\\source"])
def test_unsafe_artifact_paths_fail(source_case, helper, value):
    lock, root, matches = source_case
    lock["inputs"][0]["path"] = value
    with pytest.raises(ValueError, match="path"):
        helper.verify_sources(lock, root, matches)


@pytest.mark.parametrize("value", ["../source.c", "/source.c", "src/../source.c", "src\\source.c"])
def test_unsafe_source_paths_fail(source_case, helper, value):
    lock, root, matches = source_case
    lock["coverage"][1]["source_paths"] = [value]
    with pytest.raises(ValueError, match="path"):
        helper.verify_sources(lock, root, matches)


def test_wrong_file_hash_fails(source_case, helper):
    lock, root, matches = source_case
    (root / lock["inputs"][0]["path"]).write_bytes(b"substituted source\n")
    with pytest.raises(ValueError, match="SHA-256"):
        helper.verify_sources(lock, root, matches)


def test_symlink_file_fails(source_case, helper):
    lock, root, matches = source_case
    path = root / lock["inputs"][0]["path"]
    target = path.with_suffix(".real")
    path.rename(target)
    path.symlink_to(target.name)
    with pytest.raises(ValueError, match="symlink"):
        helper.verify_sources(lock, root, matches)


def test_symlink_parent_fails(source_case, helper, tmp_path):
    lock, root, matches = source_case
    external = tmp_path / "external"
    external.mkdir()
    external_file = external / "source.tar.xz"
    external_file.write_bytes((root / lock["inputs"][0]["path"]).read_bytes())
    link = root / "linked-inputs"
    link.symlink_to(external, target_is_directory=True)
    lock["inputs"][0]["path"] = "linked-inputs/source.tar.xz"
    with pytest.raises(ValueError, match="symlink"):
        helper.verify_sources(lock, root, matches)


def test_nonregular_declared_file_fails(source_case, helper):
    lock, root, matches = source_case
    path = root / lock["patches"][0]["path"]
    path.unlink()
    path.mkdir()
    with pytest.raises(ValueError, match="regular"):
        helper.verify_sources(lock, root, matches)


def test_extra_unreviewed_file_fails(source_case, helper):
    lock, root, matches = source_case
    (root / "unreviewed.txt").write_text("not in the lock\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unreviewed"):
        helper.verify_sources(lock, root, matches)


@pytest.mark.parametrize("duplicate", ["name", "path"])
def test_duplicate_input_identities_or_paths_fail(source_case, helper, duplicate):
    lock, root, matches = source_case
    lock["inputs"][1][duplicate] = lock["inputs"][0][duplicate]
    with pytest.raises(ValueError, match="duplicate"):
        helper.verify_sources(lock, root, matches)


def test_duplicate_cross_section_path_fails(source_case, helper):
    lock, root, matches = source_case
    lock["patches"][0]["path"] = lock["inputs"][0]["path"]
    lock["patches"][0]["sha256"] = lock["inputs"][0]["sha256"]
    with pytest.raises(ValueError, match="duplicate"):
        helper.verify_sources(lock, root, matches)


def test_bare_signer_claim_without_hashed_evidence_fails(source_case, helper):
    lock, root, matches = source_case
    lock["inputs"][0]["authentication"] = {
        "method": "detached-signature",
        "signer": "unverified signer text",
    }
    with pytest.raises(ValueError, match="authentication"):
        helper.verify_sources(lock, root, matches)


def test_unknown_authentication_evidence_fails(source_case, helper):
    lock, root, matches = source_case
    lock["inputs"][0]["authentication"]["evidence_paths"] = ["evidence/not-declared.txt"]
    with pytest.raises(ValueError, match="evidence"):
        helper.verify_sources(lock, root, matches)


@pytest.mark.parametrize("field", ["cve", "owner"])
def test_coverage_must_bind_match_cve_and_owner(source_case, helper, field):
    lock, root, matches = source_case
    lock["coverage"][0][field] = "CVE-2026-99999" if field == "cve" else "opencv-python"
    with pytest.raises(ValueError, match="coverage"):
        helper.verify_sources(lock, root, matches)


@pytest.mark.parametrize("section", ["coverage", "matches"])
def test_duplicate_coverage_identities_fail(source_case, helper, section):
    lock, root, matches = source_case
    records = lock["coverage"] if section == "coverage" else matches
    records.append(copy.deepcopy(records[0]))
    with pytest.raises(ValueError, match="duplicate"):
        helper.verify_sources(lock, root, matches)


def test_unknown_disposition_fails(source_case, helper):
    lock, root, matches = source_case
    lock["coverage"][0]["disposition"] = "not_applicable"
    with pytest.raises(ValueError, match="disposition"):
        helper.verify_sources(lock, root, matches)


def test_missing_regression_id_fails(source_case, helper):
    lock, root, matches = source_case
    lock["coverage"][0]["regression_id"] = ""
    with pytest.raises(ValueError, match="regression"):
        helper.verify_sources(lock, root, matches)


@pytest.mark.parametrize("disposition", ["already_fixed", "absent_condition"])
def test_non_repaired_dispositions_reject_repair_commits(source_case, helper, disposition):
    lock, root, matches = source_case
    record = next(item for item in lock["coverage"] if item["disposition"] == disposition)
    record["repair_commits"] = ["1" * 40]
    with pytest.raises(ValueError, match="repair"):
        helper.verify_sources(lock, root, matches)


def test_repaired_disposition_requires_commits(source_case, helper):
    lock, root, matches = source_case
    lock["coverage"][0]["repair_commits"] = []
    with pytest.raises(ValueError, match="repair"):
        helper.verify_sources(lock, root, matches)


def test_unknown_repair_reference_fails(source_case, helper):
    lock, root, matches = source_case
    lock["coverage"][0]["repair_commits"][-1] = "f" * 40
    with pytest.raises(ValueError, match="repair"):
        helper.verify_sources(lock, root, matches)


def test_coverage_repair_requires_declared_prerequisite(source_case, helper):
    lock, root, matches = source_case
    lock["coverage"][0]["repair_commits"] = ["2" * 40]
    with pytest.raises(ValueError, match="prerequisite"):
        helper.verify_sources(lock, root, matches)


def test_coverage_repair_must_cover_declared_source_paths(source_case, helper):
    lock, root, matches = source_case
    lock["coverage"][0]["source_paths"] = ["libavcodec/different.c"]
    with pytest.raises(ValueError, match="source path"):
        helper.verify_sources(lock, root, matches)


def test_patch_prerequisites_must_be_declared_earlier(source_case, helper):
    lock, root, matches = source_case
    lock["patches"].reverse()
    with pytest.raises(ValueError, match="prerequisite"):
        helper.verify_sources(lock, root, matches)


@pytest.mark.parametrize("field", ["commit", "path"])
def test_duplicate_patch_application_fails(source_case, helper, field):
    lock, root, matches = source_case
    lock["patches"][1][field] = lock["patches"][0][field]
    with pytest.raises(ValueError, match="duplicate"):
        helper.verify_sources(lock, root, matches)


@pytest.mark.parametrize(
    ("owner", "field", "value"),
    [
        ("av", "version", "18.1.1"),
        ("av", "abi", "cp312-cp312"),
        ("opencv-python", "version", "5.0.0.94"),
        ("opencv-python", "abi", "cp312-cp312"),
    ],
)
def test_build_identity_is_exact(source_case, helper, owner, field, value):
    lock, root, matches = source_case
    record = next(item for item in lock["builds"] if item["owner"] == owner)
    record[field] = value
    with pytest.raises(ValueError, match="build"):
        helper.verify_sources(lock, root, matches)


@pytest.mark.parametrize(
    ("field", "value"),
    [("os", "darwin"), ("architecture", "arm64"), ("python", "3.11"), ("execution", "emulated")],
)
def test_build_platform_is_native_linux_amd64_python312(source_case, helper, field, value):
    lock, root, matches = source_case
    lock["builds"][0]["platform"][field] = value
    with pytest.raises(ValueError, match="platform"):
        helper.verify_sources(lock, root, matches)


def test_builds_require_separate_environments(source_case, helper):
    lock, root, matches = source_case
    lock["builds"][1]["configuration"]["environment"] = lock["builds"][0]["configuration"]["environment"]
    with pytest.raises(ValueError, match="environment"):
        helper.verify_sources(lock, root, matches)


@pytest.mark.parametrize("field", ["tools", "assets"])
def test_build_references_declared_inputs(source_case, helper, field):
    lock, root, matches = source_case
    lock["builds"][0][field] = ["undeclared-input"]
    with pytest.raises(ValueError, match="input"):
        helper.verify_sources(lock, root, matches)


def test_cli_emits_json_only_after_success(source_case, helper, monkeypatch, capsys, tmp_path):
    lock, root, matches = source_case
    lock_path = tmp_path / "lock.json"
    matches_path = tmp_path / "matches.json"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    matches_path.write_text(json.dumps(matches), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["source-inputs.py", "--lock", str(lock_path), "--root", str(root), "--original-matches", str(matches_path)],
    )
    helper.main()
    output = capsys.readouterr()
    assert json.loads(output.out)["schema_version"] == "ffmpeg-wheel-source/v1"
    assert output.err == ""


def test_cli_invalid_input_has_no_success_output(source_case, helper, monkeypatch, capsys, tmp_path):
    lock, root, matches = source_case
    lock["coverage"].pop()
    lock_path = tmp_path / "invalid-lock.json"
    matches_path = tmp_path / "matches.json"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    matches_path.write_text(json.dumps(matches), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["source-inputs.py", "--lock", str(lock_path), "--root", str(root), "--original-matches", str(matches_path)],
    )
    with pytest.raises(SystemExit) as exit_info:
        helper.main()
    output = capsys.readouterr()
    assert exit_info.value.code != 0
    assert output.out == ""
