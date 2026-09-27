"""Opt-in Apple Silicon drill for absent guest-agent startup and recovery.

It provisions real local VMs through an isolated helper, mutates only disposable
image clones, and explicitly cleans up owned VM and session resources.
"""

from __future__ import annotations

import json
import os
import platform
import stat
import sys
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from tldw_Server_API.app.core.Sandbox.models import RunPhase, RunSpec, RuntimeType, SessionSpec
from tldw_Server_API.app.core.Sandbox.runners.vz_linux_runner import VZLinuxRunner
from tldw_Server_API.app.core.Sandbox.service import SandboxService
from tldw_Server_API.app.core.Sandbox.streams import get_hub
from tldw_Server_API.app.core.testing import is_truthy
from tldw_Server_API.tests.sandbox.test_vz_linux_guest_mismatch_host_gated import _cleanup_owned_resources
from tldw_Server_API.tests.sandbox.test_vz_linux_real_host_e2e import _expect, _require_vz_linux_real_host_e2e

_STARTUP_TIMEOUT = 15
_START_AGENT = False  # The explicit negative-control plugin changes only this challenge.
_CHALLENGE = ".tldw-missing-agent-challenge"
_PROOF = ".tldw-missing-agent-proof.json"


def _require_missing_agent_bundle() -> Path:
    """Require separate manual opt-in and a disposable launcher bundle."""
    if not is_truthy(os.getenv("TLDW_SANDBOX_VZ_LINUX_MISSING_AGENT_DRILL")):
        pytest.skip("Set TLDW_SANDBOX_VZ_LINUX_MISSING_AGENT_DRILL=1 for this manual drill")
    raw = os.getenv("TLDW_SANDBOX_VZ_LINUX_MISSING_AGENT_BASE_IMAGE", "").strip()
    _expect(bool(raw), "A disposable missing-agent bundle is required")
    bundle = Path(raw).expanduser().resolve(strict=True)
    _expect(bundle.is_dir(), "Missing-agent image must be a disposable bundle directory")
    return bundle


def _read_startup_proof(path: Path, nonce: str, mode: str, expected_vm_id: str) -> dict[str, str]:
    """Require a bounded fresh guest-written marker from the service launcher."""
    _expect(not path.is_symlink(), "Invalid startup proof: symlink")
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0))
        try:
            _expect(stat.S_ISREG(os.fstat(fd).st_mode), "Invalid startup proof: not a regular file")
            with os.fdopen(fd, "rb", closefd=False) as handle:
                raw = handle.read(4097)
        finally:
            os.close(fd)
        _expect(len(raw) <= 4096, "Invalid startup proof: too large")
        proof = json.loads(raw)
    except (OSError, ValueError) as exc:
        pytest.fail(f"Invalid startup proof: {exc}")
    _expect(
        isinstance(proof, dict)
        and proof.get("nonce") == nonce
        and proof.get("mode") == mode
        and isinstance(proof.get("vm_id"), str)
        and proof["vm_id"] == expected_vm_id
        and bool(expected_vm_id),
        "Invalid startup proof: expected this VM and challenge mode",
    )
    return proof


@pytest.mark.unit
@pytest.mark.parametrize(
    "proof",
    [
        {"nonce": "stale", "vm_id": "vm-1", "mode": "no-agent"},
        {"nonce": "fresh", "vm_id": "", "mode": "no-agent"},
        {"nonce": "fresh", "vm_id": "other-vm", "mode": "no-agent"},
        {"nonce": "fresh", "vm_id": "vm-1", "mode": "start-original"},
        [],
    ],
)
def test_startup_proof_rejects_wrong_guest_or_mode(tmp_path: Path, proof: Any) -> None:
    """Reject markers from stale, unrelated, or wrong-mode VM starts."""
    path = tmp_path / "proof.json"
    path.write_text(json.dumps(proof))
    with pytest.raises(pytest.fail.Exception, match="Invalid startup proof"):
        _read_startup_proof(path, "fresh", "no-agent", "vm-1")


@pytest.mark.unit
@pytest.mark.parametrize("raw", [None, "not json", " " * 4097])
def test_startup_proof_requires_bounded_json(tmp_path: Path, raw: str | None) -> None:
    """Reject missing, malformed, or oversized startup markers."""
    path = tmp_path / "proof.json"
    if raw is not None:
        path.write_text(raw)
    with pytest.raises(pytest.fail.Exception, match="Invalid startup proof"):
        _read_startup_proof(path, "fresh", "no-agent", "vm-1")


@pytest.mark.unit
@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="POSIX proof files")
@pytest.mark.timeout(2, method="signal")
def test_startup_proof_rejects_fifo_without_blocking(tmp_path: Path) -> None:
    """Do not block on a FIFO substituted for the proof file."""
    path = tmp_path / "proof.json"
    os.mkfifo(path)
    with pytest.raises(pytest.fail.Exception, match="Invalid startup proof"):
        _read_startup_proof(path, "fresh", "no-agent", "vm-1")


@pytest.mark.unit
def test_startup_proof_rejects_symlink(tmp_path: Path) -> None:
    """Do not follow a proof-file symlink outside the workspace."""
    target = tmp_path / "target.json"
    target.write_text(json.dumps({"nonce": "fresh", "vm_id": "vm-1", "mode": "no-agent"}))
    path = tmp_path / "proof.json"
    path.symlink_to(target)
    with pytest.raises(pytest.fail.Exception, match="Invalid startup proof"):
        _read_startup_proof(path, "fresh", "no-agent", "vm-1")


@pytest.mark.integration
def test_startup_launcher_proves_service_reached_guest_workspace(tmp_path: Path) -> None:
    """The real test fixture writes a fresh proof before refusing to start VSock."""
    import subprocess  # nosec B404 - launch only this checked-in test fixture

    launcher = (
        Path(__file__).resolve().parents[3] / "tools/macos-vz-helper/Tests/failure_drill/missing-agent-launcher.sh"
    )
    (tmp_path / _CHALLENGE).write_text("abcd1234\nno-agent\n")
    environment = {**os.environ, "TLDW_AGENT_GUEST_WORKSPACE_ROOT": str(tmp_path), "TLDW_AGENT_GUEST_VM_ID": "AB12-34"}
    process = subprocess.Popen(["/bin/sh", str(launcher)], env=environment)  # nosec B603
    try:
        deadline = time.monotonic() + 10
        while not (tmp_path / _PROOF).exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        _expect(
            _read_startup_proof(tmp_path / _PROOF, "abcd1234", "no-agent", "AB12-34")["vm_id"] == "AB12-34",
            "Guest launcher did not record its VM ID",
        )
        _expect(process.poll() is None, "Guest launcher exited before readiness timeout")
    finally:
        process.terminate()
        process.wait(timeout=3)


@pytest.mark.unit
def test_missing_agent_drill_requires_separate_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep this real-VM drill gated independently of ordinary VZ smoke tests."""
    monkeypatch.setenv("TLDW_SANDBOX_VZ_LINUX_E2E", "1")
    monkeypatch.delenv("TLDW_SANDBOX_VZ_LINUX_MISSING_AGENT_DRILL", raising=False)
    with pytest.raises(pytest.skip.Exception, match="MISSING_AGENT_DRILL"):
        _require_missing_agent_bundle()


@pytest.mark.integration
@pytest.mark.vz_linux_host_failure_drill
@pytest.mark.skipif(sys.platform != "darwin", reason="macOS host only")
def test_vz_linux_missing_agent_then_healthy_session_reuse(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Observe real no-agent timeout, then real execution and healthy VM reuse."""
    fault_bundle = _require_missing_agent_bundle()
    if platform.machine() != "arm64":
        pytest.skip("Apple silicon host only")
    for name in (
        "TEST_MODE",
        "TLDW_SANDBOX_VZ_LINUX_FAKE_EXEC",
        "TLDW_SANDBOX_MACOS_HELPER_READY",
        "TLDW_SANDBOX_VZ_LINUX_TEMPLATE_READY",
        "TLDW_SANDBOX_VZ_LINUX_AVAILABLE",
    ):
        monkeypatch.delenv(name, raising=False)
    _expect(bool(os.getenv("TLDW_SANDBOX_MACOS_HELPER_SOCKET")), "An explicit isolated helper is required")
    healthy_bundle = _require_vz_linux_real_host_e2e(monkeypatch, tmp_path)
    _expect(Path(healthy_bundle).resolve() != fault_bundle, "Healthy and fault bundles must differ")
    service = SandboxService()
    helper = VZLinuxRunner.helper_client_cls()
    _expect(not helper.list_vms().vms, "Use an isolated helper with no existing VMs")
    generation = helper.ping().details.get("helper_instance_id")
    _expect(bool(generation), "Helper generation is required")
    evidence: dict[str, Any] = {
        "helper_instance_id": generation,
        "startup_timeout_sec": _STARTUP_TIMEOUT,
        "attempted_creates": [],
        "created_vms": [],
        "exec_vm_ids": [],
        "runs": [],
    }
    sessions: list[str] = []
    original_create = VZLinuxRunner.helper_client_cls.create_vm
    original_exec = VZLinuxRunner.helper_client_cls.exec_guest

    def record_create(client: Any, request: dict[str, Any]) -> Any:
        """Record create attempts and verify fresh proof from the fault guest."""
        attempt = {field: request[field] for field in ("owner", "runtime", "run_id", "session_id")}
        attempt["expected_vm_id"] = request["vm_name"]
        evidence["attempted_creates"].append(attempt)
        is_fault = Path(request["template"]).resolve() == fault_bundle
        nonce = uuid4().hex
        proof_path = Path(request["workspace_path"]) / _PROOF
        mode = "start-original" if _START_AGENT else "no-agent"
        if is_fault:
            _expect(not proof_path.exists(), "Proof must not predate this VM create")
            (proof_path.parent / _CHALLENGE).write_text(nonce + "\n" + mode + "\n")
        started = time.monotonic()
        try:
            vm = original_create(client, request)
            evidence["created_vms"].append({"vm_id": vm.vm_id})
            return vm
        except Exception as exc:
            attempt["error"] = str(exc)
            raise
        finally:
            attempt["elapsed_sec"] = time.monotonic() - started
            if is_fault:
                evidence["startup_proof"] = _read_startup_proof(proof_path, nonce, mode, request["vm_name"])

    def record_exec(client: Any, **kwargs: Any) -> Any:
        """Record whether execution was dispatched and to which VM."""
        evidence["exec_vm_ids"].append(kwargs["vm_id"])
        return original_exec(client, **kwargs)

    monkeypatch.setattr(VZLinuxRunner.helper_client_cls, "create_vm", record_create)
    monkeypatch.setattr(VZLinuxRunner.helper_client_cls, "exec_guest", record_exec)

    def run(session_id: str, bundle: str, token: str, startup_timeout: int) -> tuple[Any, str]:
        """Run one sandbox command and retain its observable result and stdout."""
        command = ["/bin/echo", token]
        result = service.start_run_scaffold(
            user_id="e2e-user",
            spec=RunSpec(
                session_id=session_id,
                runtime=RuntimeType.vz_linux,
                base_image=bundle,
                command=command,
                network_policy="deny_all",
                timeout_sec=30,
                startup_timeout_sec=startup_timeout,
            ),
            spec_version="1.0",
            idem_key=None,
            raw_body={"session_id": session_id, "runtime": "vz_linux", "command": command},
        )
        frames = get_hub().get_buffer_snapshot(result.id)
        stdout = "".join(str(frame.get("data", "")) for frame in frames if frame.get("type") == "stdout")
        evidence["runs"].append(
            {"phase": result.phase.value, "exit_code": result.exit_code, "message": result.message, "stdout": stdout}
        )
        return result, stdout

    def check_empty(label: str) -> None:
        """Require reconciliation and helper inventory to be empty after cleanup."""
        report = service.macos_diagnostics()["reconciliation"]
        evidence[label] = report
        _expect(
            report["computed"] and report["persisted_sessions"] == 0 and report["live_vms"] == 0,
            f"Reusable state remains: {report}",
        )
        _expect(not helper.list_vms().vms, "VM remains in helper registry")

    try:
        for bundle in (str(fault_bundle), healthy_bundle):
            session = service.create_session(
                user_id="e2e-user",
                spec=SessionSpec(runtime=RuntimeType.vz_linux, base_image=bundle, network_policy="deny_all"),
                spec_version="1.0",
                idem_key=None,
                raw_body={"runtime": "vz_linux", "base_image": bundle},
            )
            sessions.append(session.id)
            fault = bundle == str(fault_bundle)
            result, stdout = run(session.id, bundle, "missing_agent-drill-first", _STARTUP_TIMEOUT if fault else 60)
            if fault:
                proof = evidence["startup_proof"]
                _expect(proof["mode"] == ("start-original" if _START_AGENT else "no-agent"), "Wrong launcher mode")
                _expect(result.phase == RunPhase.failed, f"Missing-agent startup did not fail: {result}")
                _expect("guest_transport_timeout" in (result.message or ""), f"Wrong failure: {result.message}")
                _expect(not evidence["exec_vm_ids"] and stdout == "", "Execution reached an unready guest")
                _expect(not evidence["created_vms"], "An unready VM was returned as created")
                elapsed = evidence["attempted_creates"][0]["elapsed_sec"]
                _expect(_STARTUP_TIMEOUT <= elapsed < _STARTUP_TIMEOUT + 15, f"Unbounded/early timeout: {elapsed}")
                check_empty("reconciliation_after_timeout")
            else:
                _expect(result.phase == RunPhase.completed and result.exit_code == 0, f"Recovery failed: {result}")
                _expect(stdout == "missing_agent-drill-first\n", f"Wrong healthy output: {stdout!r}")
                second, output = run(session.id, bundle, "missing_agent-drill-reuse", 60)
                _expect(second.phase == RunPhase.completed and second.exit_code == 0, f"Reuse failed: {second}")
                _expect(output == "missing_agent-drill-reuse\n", f"Wrong reuse output: {output!r}")
                _expect(len(evidence["created_vms"]) == 1, "Healthy session provisioned an extra VM")
                vm_id = evidence["created_vms"][0]["vm_id"]
                _expect(evidence["exec_vm_ids"] == [vm_id, vm_id], "Healthy commands did not reuse one VM")
                _expect(vm_id != evidence["startup_proof"]["vm_id"], "Fault VM reused")
            _expect(service.destroy_session(session.id), "Session destruction failed")
            _expect(service.get_session(session.id) is None, "Session still exists")
            sessions.remove(session.id)
        _expect(helper.ping().details.get("helper_instance_id") == generation, "Helper changed during recovery")
        check_empty("reconciliation_after_cleanup")
    finally:
        try:
            _cleanup_owned_resources(service, helper, sessions, evidence)
        finally:
            (tmp_path / "guest-missing-agent.json").write_text(json.dumps(evidence, indent=2) + "\n")
