"""Keep approved app/audio CI risk acceptance exact and separate from release."""

import hashlib
import json
import subprocess  # nosec B404: execute fixed repository admission shell with controlled outcomes
from datetime import date
from pathlib import Path

import pytest
import yaml
from Helper_Scripts.Supply_Chain.exception_policy import PolicyError, evaluate_trivy_report, load_policy

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = ROOT / ".github/workflows/container-build-check.yml"
CI_POLICY = ".github/supply-chain/ci-image-risk-acceptance.json"
CANONICAL = ".github/supply-chain/vulnerability-exceptions.json"


def run_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, name: str = "app", mutation: str = "") -> dict:
    """Execute the actual workflow block with fixed scanner and approval inputs."""
    record = {
        "id": "TEST-CI-RISK",
        "component": "image-app",
        "vulnerability_id": "CVE-2099-12345",
        "purl": "pkg:deb/debian/example@1.0?arch=amd64",
        "installed_version": "1.0",
        "severity": "CRITICAL",
        "rationale": "Requester accepts this exact CI scanner risk.",
        "mitigation": "Release remains separately gated.",
        "owner": "rmusser01",
        "approval": "https://github.com/rmusser01/tldw_server/pull/2869",
        "created_on": "2026-09-11",
        "expires_on": "2026-09-17",
        "supersedes": None,
    }
    record["component"] = f"image-{name}" if name in {"app", "audio-worker"} else "image-app"
    finding = {
        "VulnerabilityID": record["vulnerability_id"],
        "InstalledVersion": "1.0",
        "Severity": "CRITICAL",
        "PkgIdentifier": {"PURL": record["purl"]},
    }
    if mutation == "version":
        finding["InstalledVersion"] = "1.1"
    elif mutation == "purl":
        finding["PkgIdentifier"] = {"PURL": record["purl"].replace("amd64", "arm64")}
    elif mutation == "advisory":
        finding["VulnerabilityID"] = "CVE-2099-99999"
    elif mutation == "severity":
        finding["Severity"] = "HIGH"
    elif mutation == "expired":
        record.update(created_on="2026-09-01", expires_on="2026-09-07")
    elif mutation == "scope":
        record["component"] = "image-worker"
    policy_dir = tmp_path / ".github/supply-chain"
    policy_dir.mkdir(parents=True)
    (tmp_path / CANONICAL).write_text(json.dumps({"schema_version": 1, "exceptions": []}))
    (tmp_path / CI_POLICY).write_text(json.dumps({"schema_version": 1, "exceptions": [record]}))
    if mutation == "unmatched":
        other = {**record, "id": "TEST-CANONICAL", "vulnerability_id": "CVE-2099-88888"}
        (tmp_path / CANONICAL).write_text(json.dumps({"schema_version": 1, "exceptions": [other]}))
    if mutation == "missing":
        (tmp_path / CI_POLICY).unlink()
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    report = {"Results": [{"Target": "fixture", "Vulnerabilities": [finding]}]}
    (evidence / f"trivy-image-{name}.json").write_text(json.dumps(report))
    steps = yaml.safe_load(WORKFLOW.read_text())["jobs"]["build-and-scan"]["steps"]
    script = next(s["run"] for s in steps if s.get("id") == "policy")
    block = script.split("python - <<'PY'\n", 1)[1].split("\nPY", 1)[0]
    block = block.replace("${{ matrix.name }}", name).replace("date.today()", "date(2026, 9, 11)")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("COMPONENT", f"image-{name}")
    exec(compile(block, str(WORKFLOW), "exec"), {})  # nosec B102: fixed repository workflow, controlled input files
    assert json.loads((evidence / f"trivy-image-{name}.json").read_text()) == report
    return json.loads((evidence / f"scan-decision-image-{name}.json").read_text())


@pytest.mark.parametrize("name", ["app", "audio-worker"])
def test_exact_approved_app_audio_findings_pass_ci(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    result = run_gate(tmp_path, monkeypatch, name=name)
    assert not result["blocking"] and not result["unmatched_exception_ids"]
    assert result["ci_risk_acceptance_ids"] == ["TEST-CI-RISK"]
    assert result["ci_risk_acceptance_policy_sha256"] == hashlib.sha256((tmp_path / CI_POLICY).read_bytes()).hexdigest()


@pytest.mark.parametrize("name", ["worker", "webui", "admin-ui"])
def test_other_images_cannot_use_app_audio_bypass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    with pytest.raises(SystemExit, match="rejected"):
        run_gate(tmp_path, monkeypatch, name=name)


@pytest.mark.parametrize(
    "mutation", ["version", "purl", "advisory", "severity", "expired", "scope", "unmatched", "missing"]
)
def test_ci_risk_acceptance_rejects_changed_or_invalid_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    with pytest.raises((SystemExit, PolicyError)):
        run_gate(tmp_path, monkeypatch, mutation=mutation)


@pytest.mark.parametrize(
    "runtime,policy,passed",
    [
        ("success", "success", True),
        ("failure", "success", False),
        ("success", "failure", False),
        ("skipped", "success", False),
    ],
)
def test_runtime_and_policy_success_remain_required(runtime: str, policy: str, passed: bool) -> None:
    steps = yaml.safe_load(WORKFLOW.read_text())["jobs"]["build-and-scan"]["steps"]
    script = next(s["run"] for s in steps if s.get("name") == "Require candidate admission")
    script = script.replace("${{ steps.runtime.outcome }}", runtime).replace("${{ steps.policy.outcome }}", policy)
    script = script.replace("${{ matrix.python_runtime }}", "true")
    result = subprocess.run(["/bin/bash", "-c", script], capture_output=True, check=False)  # nosec B603
    assert (result.returncode == 0) is passed


def test_canonical_policy_does_not_receive_ci_risk_acceptance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_gate(tmp_path, monkeypatch)
    report = json.loads((tmp_path / "evidence/trivy-image-app.json").read_text())
    decision = evaluate_trivy_report(
        report,
        component="image-app",
        policy=load_policy(tmp_path / CANONICAL, today=date(2026, 9, 11)),
        today=date(2026, 9, 11),
    )
    assert len(decision.blocking) == 1
    assert not decision.excepted


def test_ci_policy_is_not_loaded_by_other_workflows() -> None:
    for path in (ROOT / ".github/workflows").glob("*.yml"):
        if path != WORKFLOW:
            assert "ci-image-risk-acceptance.json" not in path.read_text()
