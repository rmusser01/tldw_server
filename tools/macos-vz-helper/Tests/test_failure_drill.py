"""Portable safety and result-contract tests for the opt-in fault workflow."""

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import NoReturn, Optional

import pytest


@pytest.fixture
def drill() -> ModuleType:
    """Load the operator entrypoint without starting a helper."""
    path = Path(__file__).resolve().parents[1] / "scripts/vz-failure-drill.py"
    assert path.is_file(), "The reproducible fault workflow is not implemented"
    spec = importlib.util.spec_from_file_location("failure_drill", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
def test_evidence_is_exclusive_and_private(drill: ModuleType, tmp_path: Path) -> None:
    """Re-running a command must not overwrite earlier receipts."""
    source = tmp_path / "source"
    source.mkdir()
    evidence = drill.create_evidence(tmp_path / "evidence", source)
    assert evidence.stat().st_mode & 0o777 == 0o700
    (evidence / "receipt.json").write_text("retained")
    with pytest.raises(FileExistsError):
        drill.create_evidence(evidence, source)
    assert (evidence / "receipt.json").read_text() == "retained"


@pytest.mark.unit
@pytest.mark.parametrize("symlink", [False, True])
def test_evidence_cannot_mutate_source(drill: ModuleType, tmp_path: Path, symlink: bool) -> None:
    """Containment checks must resolve aliases before creating output."""
    source = tmp_path / "source"
    source.mkdir()
    parent = source
    if symlink:
        parent = tmp_path / "alias"
        parent.symlink_to(source, target_is_directory=True)
    with pytest.raises(ValueError, match="source"):
        drill.create_evidence(parent / "evidence", source)
    assert list(source.iterdir()) == []


@pytest.mark.unit
@pytest.mark.parametrize("source", ["missing", "anchor anchor"])
def test_overlay_refuses_source_drift(drill: ModuleType, source: str) -> None:
    """An unrecognized or ambiguous source must not yield a healthy fault guest."""
    with pytest.raises(ValueError, match="anchor"):
        drill.replace_once(source, "anchor", "fault")


@pytest.mark.unit
def test_overlay_replaces_exactly_one_anchor(drill: ModuleType) -> None:
    """The replacement must preserve surrounding production code."""
    assert drill.replace_once("before anchor after", "anchor", "fault") == "before fault after"


def write_junit(path: Path, name: str, outcome: str = "", message: str = "", body: str = "") -> None:
    """Write an independently specified single-test report."""
    # Escaping locally constructed test output; no XML parser is invoked here.
    from xml.sax.saxutils import escape, quoteattr  # nosec B406

    result = f"<{outcome} message={quoteattr(message)}>{escape(body)}</{outcome}>" if outcome else ""
    path.write_text(f'<testsuites><testsuite><testcase name="{name}">{result}</testcase></testsuite></testsuites>')


@pytest.mark.unit
@pytest.mark.parametrize(
    "outcome,code,message,negative,accepted",
    [
        ("", 0, "", False, True),
        ("skipped", 0, "no helper", False, False),
        ("error", 1, "cleanup failed", False, False),
        ("failure", 1, "Failed: Readiness withholding did not fail: completed", True, True),
        ("failure", 1, "Failed: guest did not boot", True, False),
        ("", 0, "", True, False),
        ("failure", 2, "Failed: Readiness withholding did not fail: completed", True, False),
    ],
)
def test_acceptance_requires_exact_result(
    drill: ModuleType, tmp_path: Path, outcome: str, code: int, message: str, negative: bool, accepted: bool
) -> None:
    """Only the intended pass/failure is acceptance, never skips or other errors."""
    xml = tmp_path / "host.xml"
    write_junit(xml, "test_vz_linux_readiness_timeout_then_healthy_session_reuse", outcome, message)
    assert drill.test_result(xml, code, "readiness", negative)["ok"] is accepted


@pytest.mark.unit
def test_negative_control_does_not_match_source_in_traceback(drill: ModuleType, tmp_path: Path) -> None:
    """A traceback quoting the assertion must not hide an unrelated failure."""
    xml = tmp_path / "host.xml"
    write_junit(
        xml,
        "test_vz_linux_readiness_timeout_then_healthy_session_reuse",
        "failure",
        "boot failed",
        "Readiness withholding did not fail",
    )
    assert not drill.test_result(xml, 1, "readiness", True)["ok"]


@pytest.mark.unit
@pytest.mark.parametrize("variant", ["missing", "malformed", "empty", "wrong_test", "extra_test"])
def test_result_rejects_incomplete_or_wrong_report(drill: ModuleType, tmp_path: Path, variant: str) -> None:
    """A green process exit alone cannot prove that the selected drill ran."""
    xml = tmp_path / "host.xml"
    if variant == "malformed":
        xml.write_text("<")
    elif variant == "empty":
        xml.write_text("<testsuites/>")
    elif variant in ("wrong_test", "extra_test"):
        write_junit(xml, "wrong")
        if variant == "extra_test":
            xml.write_text(xml.read_text().replace("</testsuite>", '<testcase name="extra"/></testsuite>'))
    assert not drill.test_result(xml, 0, "readiness", False)["ok"]


@pytest.mark.unit
def test_cleanup_attempts_every_vm_and_records_remaining(drill: ModuleType) -> None:
    """One operational cleanup failure must not prevent later deletions."""
    remaining = {"a", "b"}

    def terminate(vm_id: str) -> bool:
        """Fail the first termination and remove subsequent VMs from the inventory."""
        if vm_id == "a":
            raise OSError("could not terminate a")
        remaining.remove(vm_id)
        return True

    helper = SimpleNamespace(
        list_vms=lambda: SimpleNamespace(vms=[SimpleNamespace(vm_id=x) for x in sorted(remaining)]),
        terminate_vm=terminate,
    )
    result = drill.cleanup_vms(helper)
    assert remaining == {"a"}
    assert result["remaining_vm_ids"] == ["a"]
    assert not result["ok"]
    assert "could not terminate a" in result["errors"][0]


@pytest.mark.unit
def test_cleanup_unknown_is_not_empty(drill: ModuleType) -> None:
    """Unavailable enumeration is not successful cleanup."""

    def unavailable() -> NoReturn:
        """Simulate an unavailable helper instead of an empty VM inventory."""
        raise OSError("helper unavailable")

    result = drill.cleanup_vms(SimpleNamespace(list_vms=unavailable))
    assert result["remaining_vm_ids"] is None
    assert not result["ok"]


@pytest.mark.unit
@pytest.mark.parametrize(
    "body_fails,start_fails,stop_fails", [(True, False, False), (False, True, False), (False, False, True)]
)
def test_managed_lifecycle_retains_failure_and_stops(
    drill: ModuleType, tmp_path: Path, body_fails: bool, start_fails: bool, stop_fails: bool
) -> None:
    """Failed starts or drill bodies must still stop the exclusively owned helper."""

    @dataclass
    class Result:
        """Represent the outcome returned by a simulated helper lifecycle action."""

        ok: bool
        reason: str = ""

    state = {"running": False}

    def start(*args: Path) -> Result:
        """Mark the helper running and return the configured startup outcome."""
        state["running"] = True
        return Result(not start_fails, "start failed" if start_fails else "")

    def stop(*args: Path, **kwargs: Path) -> Result:
        """Stop the simulated helper unless the configured shutdown fails."""
        if stop_fails:
            raise OSError("stop failed")
        state["running"] = False
        return Result(True)

    ctl = SimpleNamespace(
        collect_check_results=lambda *args: [("preflight", Result(True))],
        start_helper=start,
        stop_helper=stop,
        socket_accepts_connection=lambda path: state["running"],
    )
    helper = SimpleNamespace(list_vms=lambda: SimpleNamespace(vms=[]))
    receipt = {"errors": []}
    try:
        with drill.managed_helper(ctl, Path("helper"), tmp_path, receipt, [], lambda path: helper):
            if body_fails:
                raise ValueError("drill failed")
    except (RuntimeError, ValueError):
        pass
    assert state["running"] is stop_fails
    assert receipt["lifecycle"]["stop"]["ok"] is not stop_fails
    if stop_fails:
        assert receipt["errors"]


@pytest.mark.unit
def test_disk_probe_errors_are_not_absence(drill: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """lsof exit 1 with diagnostics means unknown, not no open handles."""
    monkeypatch.setattr(
        drill.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout="", stderr="permission denied"),
    )
    assert not drill.disk_handles([tmp_path / "rootfs.img"])["ok"]


@pytest.mark.unit
def test_cli_requires_opt_in_before_mutation(drill: ModuleType, tmp_path: Path) -> None:
    """An accidental copy/paste without consent must not allocate images or logs."""
    with pytest.raises(SystemExit) as result:
        drill.main(
            ["--source-bundle", str(tmp_path), "--helper", "helper", "--evidence-dir", str(tmp_path / "evidence")]
        )
    assert result.value.code == 2
    assert list(tmp_path.iterdir()) == []


@pytest.mark.unit
def test_source_verification_continues_after_canonical_failure(drill: ModuleType, tmp_path: Path) -> None:
    """Failure receipts must still check fault-source immutability independently."""
    fault = tmp_path / "image-store/runs/mismatch-source/bundle"
    fault.mkdir(parents=True)
    (fault / "rootfs.img").write_bytes(b"changed")
    materializer = SimpleNamespace(bundle_artifact_names=lambda path: ["rootfs.img"])
    receipt = {
        "errors": [],
        "source_before": {"rootfs.img": "original"},
        "fault_sources": {"mismatch": {"rootfs.img": "original"}},
    }
    drill.verify_sources(materializer, tmp_path / "missing", tmp_path, receipt)
    assert receipt["fault_sources_unchanged"] == {"mismatch": False}
    assert len(receipt["errors"]) == 2


@pytest.mark.unit
@pytest.mark.parametrize("profile", ["mismatch", "readiness", "protocol"])
@pytest.mark.parametrize(
    "fault", [None, "cancelled", "exit_code", "stdout", "dispatch", "handshake", "wire_version", "missing", "malformed"]
)
def test_negative_control_requires_actual_fault_execution(
    drill: ModuleType, tmp_path: Path, profile: str, fault: Optional[str]
) -> None:
    """Cancellation/empty output/another VM must not count as meaningful RED."""
    packet = tmp_path / "pytest/test_case0"
    packet.mkdir(parents=True)
    (tmp_path / "pytest/test_casecurrent").symlink_to(packet, target_is_directory=True)
    data = {
        "runs": [{"phase": "completed", "exit_code": 0, "stdout": profile + "-drill-first\n"}],
        "created_vms": [{"vm_id": "fault-vm"}],
        "exec_vm_ids": ["fault-vm"],
        "handshake_proof": {"vm_id": "fault-vm", "handshake_acknowledged": True},
        "protocol_proof": {"vm_id": "fault-vm", "protocol_version": "1"},
        "cleanup_errors": [],
        "remaining_owned_vms": [],
    }
    if fault == "cancelled":
        data["runs"][0].update(phase="killed", exit_code=None)
    elif fault == "exit_code":
        data["runs"][0]["exit_code"] = 1
    elif fault == "stdout":
        data["runs"][0]["stdout"] = ""
    elif fault == "dispatch":
        data["exec_vm_ids"] = ["healthy-vm"]
    elif fault == "handshake":
        data["handshake_proof"]["vm_id"] = "other-vm"
        data["protocol_proof"]["vm_id"] = "other-vm"
    elif fault == "wire_version":
        data["protocol_proof"]["protocol_version"] = "999"
    name = {"mismatch": "guest-mismatch.json", "readiness": "guest-readiness.json", "protocol": "guest-protocol.json"}[
        profile
    ]
    if fault != "missing":
        (packet / name).write_text("null" if fault == "malformed" else json.dumps(data))
    accepted = (
        fault is None
        or (profile == "mismatch" and fault == "handshake")
        or (profile != "protocol" and fault == "wire_version")
    )
    assert drill.negative_execution(tmp_path, profile)["ok"] is accepted
