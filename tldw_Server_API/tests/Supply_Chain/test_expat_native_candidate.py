"""Behavioral failure gates for candidate-only native Expat qualification."""

import os
import importlib.util
from pathlib import Path

# Execute only the repository-owned controller with isolated command doubles.
import subprocess  # nosec B404

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "Dockerfiles/candidates/expat/qualify.sh"
WORKFLOW = ROOT / ".github/workflows/expat-candidate.yml"


def scaling_module():
    path = ROOT / "Dockerfiles/candidates/expat/attribute-scaling.py"
    assert path.exists(), "bounded attribute-scaling control is not implemented"
    spec = importlib.util.spec_from_file_location("expat_scaling", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scaling_document_exercises_non_normalized_tokenized_attributes():
    from xml.parsers import expat

    seen = []
    parser = expat.ParserCreate()
    parser.StartElementHandler = lambda name, attributes: seen.append((name, attributes))
    parser.Parse(scaling_module().document(3), True)
    assert seen == [("tag", {"a0": "value", "a1": "value", "a2": "value"})]


@pytest.mark.parametrize("mutation", [None, "quadratic", "slower", "missing-mode", "wrong-version"])
def test_scaling_gate_requires_improvement_and_bounded_growth(mutation):
    module = scaling_module()
    baseline = {"version": "expat_2.8.3", "measurements": {"whole": [0.2, 0.8, 3.2], "incremental": [0.2, 0.8, 3.2]}}
    candidate = {
        "version": "expat_2.8.4",
        "measurements": {"whole": [0.01, 0.02, 0.04], "incremental": [0.01, 0.02, 0.04]},
    }
    if mutation == "quadratic":
        candidate["measurements"]["whole"] = [0.01, 0.04, 0.16]
    elif mutation == "slower":
        candidate["measurements"]["whole"] = [1, 2, 4]
    elif mutation == "missing-mode":
        del candidate["measurements"]["incremental"]
    elif mutation == "wrong-version":
        candidate["version"] = "expat_2.8.3"
    if mutation is None:
        module.compare(baseline, candidate)
    else:
        with pytest.raises(ValueError):
            module.compare(baseline, candidate)


def run_script(*args, env=None):
    """Run the real shell gate, never a test reimplementation of its decisions."""
    assert SCRIPT.exists(), "native Expat controller is not implemented"
    return subprocess.run(  # nosec B603
        ["/bin/bash", str(SCRIPT), *map(str, args)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


def evidence_fixture(root, phase):
    """Hand-authored success evidence at the container/controller boundary."""
    root.mkdir(parents=True)
    (root / "phase.exit").write_text("0\n")
    (root / "complete.txt").write_text(phase + "\n")
    if phase == "prepare":
        (root / "authentication.json").write_text('{"signatures":{}}')
    elif phase == "build":
        (root / "parser-verbose.log").write_text("PASS: test_default_attr_index_after_dtd_copy\n")
        (root / "wide-controls.log").write_text("PASS: wide XML controls\n")
        (root / "abi.txt").write_text("compatible\n")
        (root / "scaling-comparison.log").write_text("PASS: bounded whole/incremental attribute scaling\n")
        (root / "artifacts").mkdir()
        (root / "artifacts/SHA256SUMS").write_text("fixture hashes\n")
    elif phase == "sanitize":
        (root / "sanitizer-tests.log").write_text("100% tests passed, 0 tests failed\n")
    elif phase == "install":
        (root / "versions.txt").write_text("libexpat1 2.8.4\nlibexpatw 2.8.4\n")
        (root / "apt-check.log").write_text("Reading package lists...\n")
        (root / "dpkg-audit.txt").write_text("")


@pytest.mark.parametrize("phase", ["prepare", "build", "sanitize", "install"])
def test_evidence_accepts_completed_phase(tmp_path, phase):
    evidence_fixture(tmp_path / phase, phase)
    result = run_script("verify-evidence", tmp_path / phase, phase)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "mutation", ["missing-status", "failed-status", "missing-parser", "wrong-parser", "missing-wide", "missing-scaling"]
)
def test_evidence_never_accepts_missing_or_failed_parser_checks(tmp_path, mutation):
    evidence_fixture(tmp_path / "build", "build")
    directory = tmp_path / "build"
    if mutation == "missing-status":
        (directory / "phase.exit").unlink()
    elif mutation == "failed-status":
        (directory / "phase.exit").write_text("7\n")
    elif mutation == "missing-parser":
        (directory / "parser-verbose.log").unlink()
    elif mutation == "wrong-parser":
        (directory / "parser-verbose.log").write_text("PASS: some_other_test\n")
    elif mutation == "missing-wide":
        (directory / "wide-controls.log").unlink()
    else:
        (directory / "scaling-comparison.log").unlink()
    assert run_script("verify-evidence", directory, "build").returncode != 0


def test_install_evidence_rejects_dpkg_audit_findings(tmp_path):
    evidence_fixture(tmp_path / "install", "install")
    (tmp_path / "install/dpkg-audit.txt").write_text("unconfigured package\n")
    assert run_script("verify-evidence", tmp_path / "install", "install").returncode != 0


def controller_environment(tmp_path, **overrides):
    """Docker is external; keep controller execution, file copies and gates real."""
    binaries = tmp_path / "bin"
    binaries.mkdir()
    fixtures = tmp_path / "fixtures"
    for phase in ("prepare", "build", "sanitize", "install"):
        evidence_fixture(fixtures / phase, phase)
    commands = {
        "uname": '#!/bin/sh\ncase "$1" in -s) echo Linux;; -m) echo "${HOST_ARCH:-x86_64}";; esac\n',
        "docker": r"""#!/bin/bash
set -eu
printf '%s\n' "$*" >> "$COMMAND_LOG"
case "$1:$2" in
info:*) echo "${DAEMON_ARCH:-x86_64}" ;;
build:*) exit "${IMAGE_BUILD_STATUS:-0}" ;;
image:inspect)
  if [[ "$*" == *Architecture* ]]; then echo "${IMAGE_ARCH:-amd64}"; else echo sha256:fixture; fi ;;
create:*) echo "${@: -1}" ;;
start:*) phase="${@: -1}"; if [[ "$phase" == "${FAIL_PHASE:-}" ]]; then exit 7; fi ;;
cp:*)
  if [[ "$2" == *:/work/evidence/* ]]; then
    phase="${2#*:/work/evidence/}"; phase="${phase%/.}"
    mkdir -p "$3"; cp -R "$FIXTURES/$phase/." "$3/"
    if [[ "$phase" == "${FAIL_PHASE:-}" ]]; then echo 7 > "$3/phase.exit"; fi
    if [[ "$phase" == "${OMIT_PHASE:-}" ]]; then rm "$3/phase.exit"; fi
  fi ;;
commit:*) echo sha256:prepared ;;
rm:*) ;;
*) exit 65 ;;
esac
""",
    }
    for name, body in commands.items():
        path = binaries / name
        path.write_text(body)
        path.chmod(0o700)
    env = {
        **os.environ,
        "PATH": f"{binaries}:{os.environ['PATH']}",
        "COMMAND_LOG": str(tmp_path / "commands"),
        "FIXTURES": str(fixtures),
        "GITHUB_RUN_ID": "123",
    }
    env.update(overrides)
    return env


@pytest.mark.parametrize("phase", ["prepare", "build", "sanitize", "install"])
def test_controller_preserves_phase_failure_and_collected_evidence(tmp_path, phase):
    env = controller_environment(tmp_path, FAIL_PHASE=phase)
    result = run_script("controller", tmp_path / "evidence", env=env)
    assert result.returncode == 7, result.stderr
    assert (tmp_path / f"evidence/{phase}/phase.exit").read_text() == "7\n"
    assert not (tmp_path / "evidence/system-qualified.txt").exists()


@pytest.mark.parametrize("override", [{"HOST_ARCH": "arm64"}, {"DAEMON_ARCH": "aarch64"}, {"IMAGE_ARCH": "arm64"}])
def test_controller_rejects_non_native_execution(tmp_path, override):
    result = run_script("controller", tmp_path / "evidence", env=controller_environment(tmp_path, **override))
    assert result.returncode != 0
    assert "native" in result.stderr
    assert not (tmp_path / "evidence/system-qualified.txt").exists()


def test_controller_requires_evidence_even_after_successful_container(tmp_path):
    result = run_script("controller", tmp_path / "evidence", env=controller_environment(tmp_path, OMIT_PHASE="build"))
    assert result.returncode != 0
    assert not (tmp_path / "evidence/system-qualified.txt").exists()


def test_controller_only_qualifies_system_after_all_four_phases(tmp_path):
    result = run_script("controller", tmp_path / "evidence", env=controller_environment(tmp_path))
    assert result.returncode == 0, result.stderr
    assert "Python bundled parser remains unqualified" in (tmp_path / "evidence/system-qualified.txt").read_text()
    for phase in ("prepare", "build", "sanitize", "install"):
        assert (tmp_path / f"evidence/{phase}/container.exit").read_text() == "0\n"


def test_controller_propagates_base_image_build_failure(tmp_path):
    result = run_script(
        "controller", tmp_path / "evidence", env=controller_environment(tmp_path, IMAGE_BUILD_STATUS="8")
    )
    assert result.returncode == 8
    assert not (tmp_path / "evidence/prepare").exists()


def test_controller_rejects_preexisting_evidence_before_docker(tmp_path):
    directory = tmp_path / "evidence"
    directory.mkdir()
    (directory / "system-qualified.txt").write_text("old result")
    result = run_script("controller", directory, env=controller_environment(tmp_path))
    assert result.returncode != 0
    assert (directory / "system-qualified.txt").read_text() == "old result"
    assert not (tmp_path / "commands").exists()


def test_workflow_always_uploads_failure_evidence_with_read_only_permissions():
    assert WORKFLOW.exists(), "native Expat workflow is not implemented"
    workflow = yaml.safe_load(WORKFLOW.read_text())
    assert workflow["permissions"] == {"contents": "read"}
    job = workflow["jobs"]["system"]
    assert job["runs-on"] == "ubuntu-24.04"
    upload = next(step for step in job["steps"] if "upload-artifact@" in step.get("uses", ""))
    assert upload["if"] == "${{ always() }}"
    assert upload["with"]["if-no-files-found"] == "error"
