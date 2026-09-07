from __future__ import annotations

import os
import re
import stat

# Bandit: subprocess is required to execute fixed local qualification test commands.
import subprocess  # nosec B404
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = ROOT / ".github/workflows/util-linux-candidate.yml"
QUALIFY = ROOT / "Dockerfiles/candidates/util-linux/qualify.sh"
DOCKERFILE = ROOT / "Dockerfiles/candidates/util-linux/Dockerfile"
CHECKOUT_SHA = "34e114876b0b11c390a56381ad16ebd13914f8d5"
UPLOAD_SHA = "043fb46d1a93c77aae656e7c1c64a875d1fc6a0a"
LIBRARIES = ("libmount", "libblkid", "libuuid", "libsmartcols", "liblastlog2")
BASH = "/bin/bash"


def _load_workflow() -> dict[str, Any]:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def _qualification_run_script() -> str:
    workflow = _load_workflow()
    steps = workflow["jobs"]["qualify"]["steps"]
    return next(step["run"] for step in steps if step.get("name") == "Run native qualification")


def _write_executable(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _run_bash(
    arguments: list[str],
    *,
    cwd: Path = ROOT,
    env: dict[str, str] | None = None,
    timeout: int = 30,
) -> subprocess.CompletedProcess[str]:
    # Bandit: every caller uses fixed shell code or a repo-owned script plus controlled fixture paths.
    return subprocess.run(  # nosec B603
        [BASH, *arguments],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _fake_command_directory(tmp_path: Path) -> Path:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(
        fake_bin / "uname",
        """#!/bin/sh
case "$1" in
  -s) printf '%s\\n' "${FAKE_UNAME_S:-Linux}" ;;
  -m) printf '%s\\n' "${FAKE_UNAME_M:-x86_64}" ;;
  *) exit 2 ;;
esac
""",
    )
    _write_executable(
        fake_bin / "docker",
        """#!/bin/bash
set -eu
printf '%s\\n' "$*" >> "${FAKE_DOCKER_LOG}"
case "${1:-}:${2:-}" in
  info:*) printf '%s\\n' "${FAKE_DOCKER_ARCH:-x86_64}" ;;
  build:*) exit "${FAKE_DOCKER_BUILD_STATUS:-0}" ;;
  image:inspect)
    case "$*" in
      *Architecture*tldw-util-linux-candidate-prepared*) printf '%s\\n' "${FAKE_PREPARED_IMAGE_ARCH:-amd64}" ;;
      *Architecture*) printf '%s\\n' "${FAKE_BASE_IMAGE_ARCH:-amd64}" ;;
      *tldw-util-linux-candidate-prepared*) printf '%s\\n' 'sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb' ;;
      *) printf '%s\\n' 'sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa' ;;
    esac
    ;;
  create:*)
    case " $* " in
      *" prepare "*) printf '%s\\n' prep-container ;;
      *" build "*) printf '%s\\n' build-container ;;
      *" install "*) printf '%s\\n' install-container ;;
      *) exit 64 ;;
    esac
    ;;
  start:*)
    case "$*" in
      *prep-container*) exit "${FAKE_PREP_STATUS:-0}" ;;
      *build-container*) exit "${FAKE_BUILD_STATUS:-0}" ;;
      *install-container*) exit "${FAKE_INSTALL_STATUS:-0}" ;;
    esac
    ;;
  commit:*) printf '%s\\n' 'sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb' ;;
  cp:*)
    if [[ "${FAKE_WRITE_EVIDENCE:-1}" == 1 && "$*" == *build-container:* ]]; then
      mkdir -p "${FAKE_EVIDENCE_DIR}/logs" "${FAKE_EVIDENCE_DIR}/status"
      printf 'build log\\n' > "${FAKE_EVIDENCE_DIR}/logs/binary-package-build-and-tests.log"
      printf '0\\n' > "${FAKE_EVIDENCE_DIR}/status/source-build.exit"
      printf '%s\\n' "${FAKE_BUILD_STATUS:-0}" > "${FAKE_EVIDENCE_DIR}/status/binary-build.exit"
      if [[ "${FAKE_BUILD_STATUS:-0}" == 0 ]]; then
        mkdir -p "${FAKE_EVIDENCE_DIR}/artifacts/source" "${FAKE_EVIDENCE_DIR}/artifacts/binary" "${FAKE_EVIDENCE_DIR}/abi"
        printf 'source sums\\n' > "${FAKE_EVIDENCE_DIR}/artifacts/source/SHA256SUMS"
        printf 'binary sums\\n' > "${FAKE_EVIDENCE_DIR}/artifacts/binary/SHA256SUMS"
        printf 'compatible\\n' > "${FAKE_EVIDENCE_DIR}/abi/comparison.txt"
        : > "${FAKE_EVIDENCE_DIR}/artifacts/binary/candidate.deb"
      fi
    fi
    if [[ "${FAKE_WRITE_EVIDENCE:-1}" == 1 && "$*" == *install-container:* ]]; then
      mkdir -p "${FAKE_EVIDENCE_DIR}/install"
      printf 'installed\\n' > "${FAKE_EVIDENCE_DIR}/install/package-versions.txt"
      printf 'ok\\n' > "${FAKE_EVIDENCE_DIR}/install/apt-get-check.log"
      : > "${FAKE_EVIDENCE_DIR}/install/dpkg-audit.txt"
      printf 'smoke ok\\n' > "${FAKE_EVIDENCE_DIR}/install/smoke-tests.log"
      printf '0\\n' > "${FAKE_EVIDENCE_DIR}/status/install.exit"
    fi
    ;;
  rm:*) ;;
  *) exit 65 ;;
esac
""",
    )
    return fake_bin


def _run_workflow_shell(tmp_path: Path, **overrides: str) -> subprocess.CompletedProcess[str]:
    fake_bin = _fake_command_directory(tmp_path)
    evidence = tmp_path / "runner" / "util-linux-candidate-evidence"
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "RUNNER_TEMP": str(tmp_path / "runner"),
            "GITHUB_RUN_ID": "12345",
            "GITHUB_SHA": "0123456789abcdef0123456789abcdef01234567",
            "FAKE_DOCKER_LOG": str(tmp_path / "docker.log"),
            "FAKE_EVIDENCE_DIR": str(evidence),
        }
    )
    env.update(overrides)
    return _run_bash(
        ["-c", _qualification_run_script()],
        cwd=ROOT,
        env=env,
        timeout=30,
    )


def _run_qualify(*arguments: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return _run_bash([str(QUALIFY), *arguments], env=env)


def _package_metadata_environment(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    package_dir = tmp_path / "packages"
    package_dir.mkdir()
    fake_bin = tmp_path / "package-tools"
    fake_bin.mkdir()
    _write_executable(
        fake_bin / "dpkg-deb",
        """#!/bin/sh
set -eu
test "$1" = -f
sed -n "s/^$3: //p" "$2"
""",
    )
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    return package_dir, env


def _write_package_control(package_dir: Path, package: str, version: str, source: str) -> None:
    (package_dir / f"{package}.deb").write_text(
        f"Package: {package}\nVersion: {version}\nSource: {source}\n",
        encoding="utf-8",
    )


def test_workflow_rejects_non_native_host_before_docker_build(tmp_path: Path) -> None:
    result = _run_workflow_shell(tmp_path, FAKE_UNAME_M="arm64")

    assert result.returncode != 0
    assert "native x86_64" in result.stderr
    identity = (tmp_path / "runner/util-linux-candidate-evidence/identity/runner.txt").read_text()
    assert "runner_kernel=Linux\n" in identity
    assert "runner_architecture=arm64\n" in identity
    assert "checked_out_commit=0123456789abcdef0123456789abcdef01234567" not in identity
    assert re.search(r"^checked_out_commit=[0-9a-f]{40}$", identity, re.MULTILINE)
    assert not (tmp_path / "docker.log").exists()


def test_workflow_retains_rejected_daemon_identity_without_starting_build_or_install(tmp_path: Path) -> None:
    result = _run_workflow_shell(tmp_path, FAKE_DOCKER_ARCH="s390x")

    assert result.returncode != 0
    identity = (tmp_path / "runner/util-linux-candidate-evidence/identity/runner.txt").read_text()
    assert "docker_daemon_architecture=s390x\n" in identity
    docker_calls = (tmp_path / "docker.log").read_text()
    assert "build --platform" not in docker_calls
    assert "tldw-util-linux-build" not in docker_calls
    assert "tldw-util-linux-install" not in docker_calls


def test_workflow_retains_rejected_base_image_identity_without_starting_build_or_install(tmp_path: Path) -> None:
    result = _run_workflow_shell(tmp_path, FAKE_BASE_IMAGE_ARCH="arm64")

    assert result.returncode != 0
    identity = (tmp_path / "runner/util-linux-candidate-evidence/identity/runner.txt").read_text()
    assert "base_image_id=sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n" in identity
    assert "base_image_architecture=arm64\n" in identity
    docker_calls = (tmp_path / "docker.log").read_text()
    assert "tldw-util-linux-build" not in docker_calls
    assert "tldw-util-linux-install" not in docker_calls


def test_workflow_retains_rejected_prepared_image_identity_without_starting_build_or_install(tmp_path: Path) -> None:
    result = _run_workflow_shell(tmp_path, FAKE_PREPARED_IMAGE_ARCH="arm64")

    assert result.returncode != 0
    identity = (tmp_path / "runner/util-linux-candidate-evidence/identity/runner.txt").read_text()
    assert "prepared_image_id=sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\n" in identity
    assert "prepared_image_architecture=arm64\n" in identity
    docker_calls = (tmp_path / "docker.log").read_text()
    assert "tldw-util-linux-build" not in docker_calls
    assert "tldw-util-linux-install" not in docker_calls


def test_workflow_propagates_build_failure_after_collecting_evidence(tmp_path: Path) -> None:
    result = _run_workflow_shell(tmp_path, FAKE_BUILD_STATUS="23")

    assert result.returncode == 23
    docker_calls = (tmp_path / "docker.log").read_text(encoding="utf-8")
    assert "cp build-container:/work/evidence/." in docker_calls
    assert " install " not in docker_calls
    evidence = tmp_path / "runner" / "util-linux-candidate-evidence"
    assert (evidence / "logs/binary-package-build-and-tests.log").read_text() == "build log\n"
    assert (evidence / "status/binary-build.exit").read_text() == "23\n"
    assert (
        "checked_out_commit=0123456789abcdef0123456789abcdef01234567"
        not in (evidence / "identity/runner.txt").read_text()
    )


def test_workflow_rejects_missing_evidence_after_successful_build(tmp_path: Path) -> None:
    result = _run_workflow_shell(tmp_path, FAKE_WRITE_EVIDENCE="0")

    assert result.returncode != 0
    assert "required qualification evidence" in result.stderr
    docker_calls = (tmp_path / "docker.log").read_text(encoding="utf-8")
    assert " install " not in docker_calls


def test_source_hash_verification_rejects_modified_download(tmp_path: Path) -> None:
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    (source_dir / "util-linux_2.41.5-0+deb13u1.dsc").write_bytes(b"modified dsc")
    (source_dir / "util-linux_2.41.5.orig.tar.xz").write_bytes(b"modified orig")
    (source_dir / "util-linux_2.41.5-0+deb13u1.debian.tar.xz").write_bytes(b"modified debian")

    result = _run_qualify("verify-sources", str(source_dir))

    assert result.returncode != 0
    assert "FAILED" in result.stdout + result.stderr


def test_checksum_manifest_verifies_after_artifact_directory_is_relocated(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "generated-artifact"
    artifact_dir.mkdir()
    (artifact_dir / "candidate.deb").write_bytes(b"candidate package bytes")

    generated = _run_qualify("write-sums", str(artifact_dir))
    assert generated.returncode == 0, generated.stderr
    downloaded = tmp_path / "downloaded-artifact"
    artifact_dir.rename(downloaded)

    verified = _run_bash(["-c", 'cd "$1" && sha256sum -c SHA256SUMS', "verify-manifest", str(downloaded)])
    assert verified.returncode == 0, verified.stderr
    assert verified.stdout == "./candidate.deb: OK\n"


def test_source_family_versions_accept_exact_runtime_and_debug_package_mappings(tmp_path: Path) -> None:
    package_dir, env = _package_metadata_environment(tmp_path)
    candidate = "2.41.5-0+deb13u1+tldw1"
    source = f"util-linux ({candidate})"
    expected = {
        "bsdutils": f"1:{candidate}",
        "bsdutils-dbgsym": f"1:{candidate}",
        "login": f"1:4.16.0-2+really{candidate}",
        "login-dbgsym": f"1:4.16.0-2+really{candidate}",
    }
    for package, version in expected.items():
        _write_package_control(package_dir, package, version, source)
    output = tmp_path / "source-family-versions.txt"

    result = _run_qualify("verify-package-versions", str(package_dir), str(output), env=env)

    assert result.returncode == 0, result.stderr
    assert output.read_text().splitlines() == [
        f"{package}\t{expected[package]}\t{source}" for package in sorted(expected)
    ]


@pytest.mark.parametrize(
    ("package", "wrong_version", "expected_version"),
    [
        (
            "bsdutils-dbgsym",
            "2.41.5-0+deb13u1+tldw1",
            "1:2.41.5-0+deb13u1+tldw1",
        ),
        (
            "bsdutils-dbgsym-extra",
            "1:2.41.5-0+deb13u1+tldw1",
            "2.41.5-0+deb13u1+tldw1",
        ),
        (
            "login-dbgsym",
            "2.41.5-0+deb13u1+tldw1",
            "1:4.16.0-2+really2.41.5-0+deb13u1+tldw1",
        ),
        (
            "login-dbgsym-extra",
            "1:4.16.0-2+really2.41.5-0+deb13u1+tldw1",
            "2.41.5-0+deb13u1+tldw1",
        ),
    ],
)
def test_source_family_versions_reject_wrong_or_glob_like_versions(
    tmp_path: Path, package: str, wrong_version: str, expected_version: str
) -> None:
    package_dir, env = _package_metadata_environment(tmp_path)
    source = "util-linux (2.41.5-0+deb13u1+tldw1)"
    _write_package_control(package_dir, package, wrong_version, source)

    result = _run_qualify(
        "verify-package-versions",
        str(package_dir),
        str(tmp_path / "versions.txt"),
        env=env,
    )

    assert result.returncode != 0
    assert f"{package} has version {wrong_version}, expected {expected_version}" in result.stderr


def test_source_family_versions_reject_wrong_source(tmp_path: Path) -> None:
    package_dir, env = _package_metadata_environment(tmp_path)
    _write_package_control(
        package_dir,
        "login-dbgsym",
        "1:4.16.0-2+really2.41.5-0+deb13u1+tldw1",
        "not-util-linux (2.41.5-0+deb13u1+tldw1)",
    )

    result = _run_qualify(
        "verify-package-versions",
        str(package_dir),
        str(tmp_path / "versions.txt"),
        env=env,
    )

    assert result.returncode != 0
    assert "does not identify the candidate util-linux source" in result.stderr


def _write_abi_fixture(root: Path, *, missing_symbol: bool = False, changed_soname: bool = False) -> None:
    for library in LIBRARIES:
        library_dir = root / library
        library_dir.mkdir(parents=True, exist_ok=True)
        soname = f"{library}.so.99" if changed_soname and library == "libmount" else f"{library}.so.1"
        (library_dir / "soname.txt").write_text(f"{soname}\n", encoding="utf-8")
        symbols = [f"{library.upper()}_1.0 {library}_public"]
        if not missing_symbol or library != "libmount":
            symbols.append(f"{library.upper()}_1.0 {library}_second")
        (library_dir / "exported-symbols.txt").write_text("\n".join(symbols) + "\n", encoding="utf-8")


def test_abi_comparison_accepts_additions_but_rejects_missing_public_export(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_abi_fixture(baseline)
    _write_abi_fixture(candidate)
    with (candidate / "libmount/exported-symbols.txt").open("a", encoding="utf-8") as handle:
        handle.write("LIBMOUNT_2.0 new_public_symbol\n")

    accepted = _run_qualify("compare-abi", str(baseline), str(candidate), str(tmp_path / "accepted"))
    assert accepted.returncode == 0, accepted.stderr

    _write_abi_fixture(candidate, missing_symbol=True)
    rejected = _run_qualify("compare-abi", str(baseline), str(candidate), str(tmp_path / "rejected"))
    assert rejected.returncode != 0
    assert "libmount_second" in (tmp_path / "rejected/libmount-missing-symbols.txt").read_text()


def test_abi_comparison_rejects_changed_soname(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    _write_abi_fixture(baseline)
    _write_abi_fixture(candidate, changed_soname=True)

    result = _run_qualify("compare-abi", str(baseline), str(candidate), str(tmp_path / "result"))

    assert result.returncode != 0
    assert (tmp_path / "result/libmount-soname.diff").read_text()


def test_workflow_security_and_trigger_contract() -> None:
    workflow = _load_workflow()
    triggers = workflow[True]
    push = triggers["push"]

    assert set(triggers) == {"push", "workflow_dispatch"}
    assert push["branches"] == ["codex/task-13013-7-supply-chain-design"]
    assert set(push["paths"]) == {
        ".github/workflows/util-linux-candidate.yml",
        "Dockerfiles/candidates/util-linux/**",
        "Dockerfiles/candidates/ffmpeg/debian.sources",
        "tldw_Server_API/tests/Supply_Chain/test_util_linux_candidate.py",
    }
    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["concurrency"] == {
        "group": "util-linux-candidate-${{ github.workflow }}-${{ github.ref }}",
        "cancel-in-progress": False,
    }
    job = workflow["jobs"]["qualify"]
    assert job["runs-on"] == "ubuntu-24.04"
    assert job["timeout-minutes"] == 180
    assert "if" not in job
    steps = job["steps"]
    checkout = next(step for step in steps if step.get("name") == "Checkout candidate inputs")
    assert checkout == {
        "name": "Checkout candidate inputs",
        "uses": f"actions/checkout@{CHECKOUT_SHA}",
        "with": {
            "ref": "${{ github.sha }}",
            "persist-credentials": False,
        },
    }
    upload = next(step for step in steps if step.get("name") == "Upload qualification evidence")
    assert upload["uses"] == f"actions/upload-artifact@{UPLOAD_SHA}"
    assert upload["if"] == "${{ always() }}"
    assert upload["with"]["retention-days"] == 14
    assert upload["with"]["if-no-files-found"] == "error"


def test_build_drops_all_capabilities_but_package_install_keeps_ordinary_root_capabilities() -> None:
    script = _qualification_run_script()
    build_create = script.split('build_container_id="$(docker create', 1)[1].split("set +e", 1)[0]
    install_create = script.split('install_container_id="$(docker create', 1)[1].split("docker cp", 1)[0]

    assert "--cap-drop ALL" in build_create
    assert "--user 1000:1000" in build_create
    assert "--cap-drop ALL" not in install_create
    assert "--user 0:0" in install_create


def test_candidate_recipe_is_fixed_native_amd64_preparation_image() -> None:
    recipe = DOCKERFILE.read_text(encoding="utf-8")

    assert recipe.startswith(
        "FROM python:3.12.14-slim-trixie@sha256:" "78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea"
    )
    assert "COPY Dockerfiles/candidates/ffmpeg/debian.sources" in recipe
    assert "apt-get build-dep -y --no-install-recommends util-linux=2.41.5-0+deb13u1" in recipe
    assert "useradd --uid 1000" in recipe
    assert "QEMU_USER" not in recipe
    assert "nocheck" not in recipe


@pytest.mark.parametrize(
    "forbidden",
    [
        "pull_request_target",
        "pull_request",
        "schedule",
        "workflow_run",
        "--privileged",
        "--device",
        "SYS_ADMIN",
        "/var/run/docker.sock",
    ],
)
def test_workflow_has_no_privilege_escalation_or_host_escape(forbidden: str) -> None:
    assert forbidden not in WORKFLOW.read_text(encoding="utf-8")
