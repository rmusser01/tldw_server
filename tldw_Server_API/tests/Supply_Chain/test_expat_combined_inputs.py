"""Reject mixed or damaged qualification artifacts before combined-image use."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess  # nosec B404
import sys

import pytest
import yaml

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "Dockerfiles/candidates/expat/combined-inputs.py"
COMMIT = "1" * 40
SYSTEM_FILES = (
    "expat-dbgsym_2.8.4-1~deb13u1+tldw1_amd64.deb",
    "expat_2.8.4-1~deb13u1+tldw1.debian.tar.xz",
    "expat_2.8.4-1~deb13u1+tldw1.dsc",
    "expat_2.8.4-1~deb13u1+tldw1_amd64.buildinfo",
    "expat_2.8.4-1~deb13u1+tldw1_amd64.changes",
    "expat_2.8.4-1~deb13u1+tldw1_amd64.deb",
    "expat_2.8.4-1~deb13u1+tldw1_source.buildinfo",
    "expat_2.8.4-1~deb13u1+tldw1_source.changes",
    "expat_2.8.4.orig.tar.gz",
    "libexpat1-dbgsym_2.8.4-1~deb13u1+tldw1_amd64.deb",
    "libexpat1-dev_2.8.4-1~deb13u1+tldw1_amd64.deb",
    "libexpat1-udeb_2.8.4-1~deb13u1+tldw1_amd64.udeb",
    "libexpat1_2.8.4-1~deb13u1+tldw1_amd64.deb",
)
PYTHON_FILES = ("installed-binaries.sha256", "python-install.tar.gz", "python-source.tar.xz")


def load_tool():
    assert SCRIPT.exists(), "combined input gate is not implemented"
    spec = importlib.util.spec_from_file_location("combined_inputs", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evidence_fixture(tmp_path: Path) -> tuple[Path, Path]:
    """Small real files satisfying the existing shell evidence gates."""
    for profile in ("system", "python"):
        root = tmp_path / profile
        (root / "identity").mkdir(parents=True)
        (root / "identity/runner.txt").write_text(
            f"commit={COMMIT}\nkernel=Linux\narch=x86_64\ndaemon_arch=x86_64\n"
            f"base=sha256:{'2' * 64}\nbase_arch=amd64\nprepared=sha256:{'3' * 64}\n"
        )
        phases = ("prepare", "build", "sanitize", "install") if profile == "system" else ("prepare", "build", "install")
        for phase in phases:
            directory = root / phase
            directory.mkdir()
            (directory / "phase.exit").write_text("0\n")
            (directory / "container.exit").write_text("0\n")
            (directory / "complete.txt").write_text(phase + "\n")
        (root / "prepare/authentication.json").write_text("{}\n")
        (root / "build/abi.txt").write_text("compatible\n")
        (root / "build/scaling-comparison.log").write_text("PASS: bounded whole/incremental attribute scaling\n")
        (root / "install/apt-check.log").write_text("dependency check complete\n")
        (root / "install/dpkg-audit.txt").write_text("")
        if profile == "system":
            (root / "build/parser-verbose.log").write_text("PASS: test_default_attr_index_after_dtd_copy\n")
            (root / "build/wide-controls.log").write_text("PASS: wide XML controls\n")
            (root / "sanitize/sanitizer-tests.log").write_text("100% tests passed, 0 tests failed\n")
            (root / "install/versions.txt").write_text("libexpat.so.1 expat_2.8.4\nlibexpatw.so.1 expat_2.8.4\n")
        else:
            for relative in (
                "prepare/source-verification.log",
                "prepare/python-source.tar.xz",
                "prepare/baseline.xml",
                "build/xml-tests.log",
                "build/xml-results.xml",
                "build/suite-comparison.log",
                "build/python-controls.log",
                "install/python-controls.log",
                "install/binary-checksums.log",
            ):
                (root / relative).write_text("recorded worker evidence\n")
        artifacts = root / "build/artifacts"
        artifacts.mkdir()
        lines = []
        for name in SYSTEM_FILES if profile == "system" else PYTHON_FILES:
            data = ("payload for " + name).encode()
            (artifacts / name).write_bytes(data)
            lines.append(f"{hashlib.sha256(data).hexdigest()}  ./{name}\n")
        (artifacts / "SHA256SUMS").write_text("".join(lines))
    return tmp_path / "system", tmp_path / "python"


def test_combined_input_gate_verifies_both_profiles_without_modifying_payloads(tmp_path: Path) -> None:
    system, python = evidence_fixture(tmp_path)
    before = {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    result = load_tool().verify(system, python, COMMIT)
    assert result["commit"] == COMMIT and result["scope"] == "qualified-inputs-only"
    assert set(result["profiles"]["system"]["artifacts"]) == set(SYSTEM_FILES)
    assert set(result["profiles"]["python"]["artifacts"]) == set(PYTHON_FILES)
    assert before == {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}


@pytest.mark.parametrize("profile", ["system", "python"])
@pytest.mark.parametrize(
    "mutation",
    [
        "commit",
        "architecture",
        "duplicate-identity",
        "phase",
        "container",
        "evidence",
        "payload",
        "missing",
        "extra",
        "symlink",
        "directory-link",
        "traversal",
        "duplicate-hash",
        "missing-hash",
    ],
)
def test_combined_input_gate_rejects_mixed_failed_or_damaged_inputs(
    tmp_path: Path, profile: str, mutation: str
) -> None:
    system, python = evidence_fixture(tmp_path)
    root = tmp_path / profile
    identity = root / "identity/runner.txt"
    artifacts = root / "build/artifacts"
    manifest = artifacts / "SHA256SUMS"
    payload = artifacts / (SYSTEM_FILES[-1] if profile == "system" else "python-install.tar.gz")
    if mutation == "commit":
        identity.write_text(identity.read_text().replace(COMMIT, "4" * 40))
    elif mutation == "architecture":
        identity.write_text(identity.read_text().replace("base_arch=amd64", "base_arch=arm64"))
    elif mutation == "duplicate-identity":
        identity.write_text(identity.read_text() + f"commit={COMMIT}\n")
    elif mutation in {"phase", "container"}:
        (root / "build" / (mutation + ".exit")).write_text("1\n")
    elif mutation == "evidence":
        (root / "build/abi.txt").write_text("incompatible\n")
    elif mutation == "payload":
        payload.write_bytes(b"altered")
    elif mutation == "missing":
        payload.unlink()
    elif mutation == "extra":
        (artifacts / "extra.deb").write_bytes(b"unexpected")
    elif mutation == "symlink":
        outside = tmp_path / "outside"
        payload.rename(outside)
        payload.symlink_to(outside)
    elif mutation == "directory-link":
        outside = tmp_path / "outside"
        artifacts.rename(outside)
        artifacts.symlink_to(outside, target_is_directory=True)
    elif mutation == "traversal":
        manifest.write_text(manifest.read_text() + f"{'a' * 64}  ./../../outside\n")
    elif mutation == "duplicate-hash":
        manifest.write_text(manifest.read_text() + manifest.read_text().splitlines()[0] + "\n")
    else:
        manifest.write_text("\n".join(manifest.read_text().splitlines()[1:]) + "\n")
    with pytest.raises((ValueError, OSError)):
        load_tool().verify(system, python, COMMIT)


def test_combined_input_cli_emits_no_success_record_on_failure(tmp_path: Path) -> None:
    system, python = evidence_fixture(tmp_path)
    assert SCRIPT.exists(), "combined input gate is not implemented"
    result = subprocess.run(  # nosec B603
        [sys.executable, str(SCRIPT), str(system), str(python), "--commit", "not-a-commit"],
        capture_output=True,
        timeout=15,
    )
    assert result.returncode != 0 and result.stdout == b""


@pytest.mark.parametrize("damaged", [False, True])
def test_workflow_checks_same_run_inputs_before_reporting_success(tmp_path: Path, damaged: bool) -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/expat-candidate.yml").read_text())
    job = workflow["jobs"].get("combined-inputs")
    assert job is not None, "workflow does not consume the combined input gate"
    assert set(job["needs"]) == {"system", "python"}
    downloads = [step["with"] for step in job["steps"] if "actions/download-artifact@" in step.get("uses", "")]
    assert {step["name"] for step in downloads} == {
        "expat-system-candidate-${{ github.run_id }}",
        "expat-python-candidate-${{ github.run_id }}",
    }
    assert all("run-id" not in step and "repository" not in step for step in downloads)
    parent = tmp_path / "expat-combined-downloads"
    parent.mkdir()
    system, python = evidence_fixture(parent)
    # Fixed read-only Git query; no artifact data enters this command.
    checkout = subprocess.run(  # nosec B603
        ["/usr/bin/git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True, timeout=10
    )
    commit = checkout.stdout.strip()
    for root in (system, python):
        identity = root / "identity/runner.txt"
        identity.write_text(identity.read_text().replace(COMMIT, commit))
    if damaged:
        (system / "build/artifacts" / SYSTEM_FILES[-1]).write_bytes(b"changed during transport")
    command = next(step["run"] for step in job["steps"] if step.get("name") == "Verify qualified inputs")
    result = subprocess.run(  # nosec B603
        ["/bin/bash", "-c", command],
        cwd=ROOT,
        env={**os.environ, "RUNNER_TEMP": str(tmp_path), "GITHUB_SHA": commit},
        capture_output=True,
        timeout=30,
    )
    report = tmp_path / "expat-combined-inputs/inputs.json"
    if damaged:
        assert result.returncode != 0 and report.read_bytes() == b""
    else:
        assert result.returncode == 0, result.stderr
        assert json.loads(report.read_text())["scope"] == "qualified-inputs-only"
