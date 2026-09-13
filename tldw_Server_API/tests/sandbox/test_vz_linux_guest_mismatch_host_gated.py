"""Manual real-VM proof of guest capability rejection and subsequent recovery."""

from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

import pytest

from tldw_Server_API.app.core.Sandbox.macos_virtualization.models import HelperVMMetadata, HelperVMStatusReply
from tldw_Server_API.app.core.Sandbox.models import RunPhase, RunSpec, RuntimeType, SessionSpec
from tldw_Server_API.app.core.Sandbox.runners.vz_linux_runner import VZLinuxRunner
from tldw_Server_API.app.core.Sandbox.service import SandboxService
from tldw_Server_API.app.core.Sandbox.streams import get_hub
from tldw_Server_API.app.core.Sandbox.vz_guest_agent import (
    VZ_LINUX_GUEST_AGENT_REQUIRED_CAPABILITY_MISSING,
    classify_vz_linux_guest_agent,
)
from tldw_Server_API.app.core.testing import is_truthy
from tldw_Server_API.tests.sandbox.test_vz_linux_real_host_e2e import (
    _expect,
    _require_vz_linux_real_host_e2e,
)


def _require_mismatch_bundle() -> Path:
    """Require separate opt-in before accessing a deliberately incompatible guest."""
    if not is_truthy(os.getenv("TLDW_SANDBOX_VZ_LINUX_GUEST_MISMATCH_DRILL")):
        pytest.skip("Set TLDW_SANDBOX_VZ_LINUX_GUEST_MISMATCH_DRILL=1 for this manual drill")
    path = os.getenv("TLDW_SANDBOX_VZ_LINUX_MISMATCH_BASE_IMAGE", "").strip()
    _expect(bool(path), "A disposable missing-exec guest bundle is required")
    bundle = Path(path).expanduser().resolve(strict=True)
    _expect(bundle.is_dir(), "Mismatch image must be a disposable bundle directory")
    return bundle


def _owned_vm_ids(vms: list[HelperVMStatusReply], evidence: dict[str, Any]) -> list[str]:
    """Include owned creates whose helper reply was lost, never unrelated VMs."""
    owned = {vm["vm_id"] for vm in evidence["created_vms"]}
    fields = ("owner", "runtime", "run_id", "session_id")
    attempts = {tuple(attempt[field] for field in fields) for attempt in evidence["attempted_creates"]}
    for vm in vms:
        identity = tuple(getattr(vm.metadata, field) for field in fields)
        if all(identity) and identity in attempts:
            owned.add(vm.vm_id)
    return [vm.vm_id for vm in vms if vm.vm_id in owned]


def test_guest_mismatch_cleanup_covers_lost_create_reply() -> None:
    """A timed-out create can still leave a VM with this run's ownership metadata."""
    evidence = {
        "created_vms": [],
        "attempted_creates": [{"owner": "tldw", "runtime": "vz_linux", "run_id": "mine", "session_id": "session"}],
    }
    vms = [
        HelperVMStatusReply(
            protocol_version="1",
            helper_version="test",
            vm_id="lost-reply",
            state="running",
            healthy=True,
            metadata=HelperVMMetadata(owner="tldw", runtime="vz_linux", run_id="mine", session_id="session"),
        ),
        HelperVMStatusReply(
            protocol_version="1",
            helper_version="test",
            vm_id="other-run",
            state="running",
            healthy=True,
            metadata=HelperVMMetadata(owner="tldw", runtime="vz_linux", run_id="other", session_id="session"),
        ),
        HelperVMStatusReply(
            protocol_version="1",
            helper_version="test",
            vm_id="other-owner",
            state="running",
            healthy=True,
            metadata=HelperVMMetadata(owner="other", runtime="vz_linux", run_id="mine", session_id="session"),
        ),
    ]
    _expect(_owned_vm_ids(vms, evidence) == ["lost-reply"], "Cleanup must select only the owned lost-reply VM")


def test_guest_mismatch_drill_requires_separate_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    """Normal E2E opt-in must not enable intentional guest fault injection."""
    monkeypatch.setenv("TLDW_SANDBOX_VZ_LINUX_E2E", "1")
    monkeypatch.delenv("TLDW_SANDBOX_VZ_LINUX_GUEST_MISMATCH_DRILL", raising=False)
    with pytest.raises(pytest.skip.Exception, match="GUEST_MISMATCH_DRILL"):
        _require_mismatch_bundle()


def test_guest_mismatch_drill_requires_fault_bundle(monkeypatch: pytest.MonkeyPatch) -> None:
    """Opting in without a fault bundle is a configuration error, not acceptance."""
    monkeypatch.setenv("TLDW_SANDBOX_VZ_LINUX_GUEST_MISMATCH_DRILL", "1")
    monkeypatch.delenv("TLDW_SANDBOX_VZ_LINUX_MISMATCH_BASE_IMAGE", raising=False)
    with pytest.raises(pytest.fail.Exception, match="missing-exec"):
        _require_mismatch_bundle()


@pytest.mark.integration
@pytest.mark.vz_linux_host_failure_drill
@pytest.mark.skipif(sys.platform != "darwin", reason="macOS host only")
def test_vz_linux_rejects_real_guest_missing_exec_then_runs_healthy_session(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Use an operator-owned helper and disposable bundles; retain JSON evidence.

    The caller starts/stops an isolated helper. This test only destroys sessions
    and VMs it creates, including on assertion failure. Observers delegate to the
    real client unchanged; no readiness, metadata, or execution is simulated.
    """
    mismatch_bundle = _require_mismatch_bundle()
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
    healthy_bundle = _require_vz_linux_real_host_e2e(monkeypatch, tmp_path)
    _expect(Path(healthy_bundle).resolve() != mismatch_bundle, "Healthy and fault bundles must differ")
    _expect(bool(os.getenv("TLDW_SANDBOX_MACOS_HELPER_SOCKET")), "An explicit isolated helper is required")
    service = SandboxService()
    helper = VZLinuxRunner.helper_client_cls()
    _expect(not helper.list_vms().vms, "Use an isolated helper with no existing VMs")
    generation = helper.ping().details.get("helper_instance_id")
    _expect(bool(generation), "Helper generation is required")
    evidence: dict[str, Any] = {
        "helper_instance_id": generation,
        "attempted_creates": [],
        "created_vms": [],
        "exec_vm_ids": [],
        "runs": [],
    }
    sessions: list[str] = []
    original_create = VZLinuxRunner.helper_client_cls.create_vm
    original_exec = VZLinuxRunner.helper_client_cls.exec_guest

    def record_create(client: Any, request: dict[str, Any]) -> Any:
        """Record real handshake metadata without changing the create reply."""
        evidence["attempted_creates"].append(
            {field: request[field] for field in ("owner", "runtime", "run_id", "session_id")}
        )
        vm = original_create(client, request)
        evidence["created_vms"].append({"vm_id": vm.vm_id, "guest_agent": classify_vz_linux_guest_agent(vm.details)})
        return vm

    def record_exec(client: Any, **kwargs: Any) -> Any:
        """Count actual dispatches, including any forbidden mismatched-guest exec."""
        evidence["exec_vm_ids"].append(kwargs["vm_id"])
        return original_exec(client, **kwargs)

    monkeypatch.setattr(VZLinuxRunner.helper_client_cls, "create_vm", record_create)
    monkeypatch.setattr(VZLinuxRunner.helper_client_cls, "exec_guest", record_exec)

    def run(session_id: str, bundle: str, token: str) -> tuple[Any, str]:
        """Run through the public service and retain exact output and failure reason."""
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
                startup_timeout_sec=60,
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

    try:
        for bundle in (str(mismatch_bundle), healthy_bundle):
            session = service.create_session(
                user_id="e2e-user",
                spec=SessionSpec(runtime=RuntimeType.vz_linux, base_image=bundle, network_policy="deny_all"),
                spec_version="1.0",
                idem_key=None,
                raw_body={"runtime": "vz_linux", "base_image": bundle},
            )
            sessions.append(session.id)
            result, stdout = run(session.id, bundle, "mismatch-drill-first")
            if bundle == str(mismatch_bundle):
                _expect(len(evidence["created_vms"]) == 1, "Expected a real fault VM, not a preflight failure")
                guest = evidence["created_vms"][0]["guest_agent"]
                _expect(
                    guest["reasons"] == [VZ_LINUX_GUEST_AGENT_REQUIRED_CAPABILITY_MISSING],
                    f"Wrong injected fault: {guest}",
                )
                _expect(result.phase == RunPhase.failed, f"Mismatched guest was not rejected: {result}")
                _expect(
                    VZ_LINUX_GUEST_AGENT_REQUIRED_CAPABILITY_MISSING in (result.message or ""),
                    f"Wrong rejection reason: {result.message}",
                )
                _expect(not evidence["exec_vm_ids"] and stdout == "", "Execution reached the mismatched guest")
                report = service.macos_diagnostics()["reconciliation"]
                evidence["reconciliation_after_rejection"] = report
                _expect(
                    report["computed"] and report["persisted_sessions"] == 0 and report["live_vms"] == 0,
                    f"Rejected guest left reusable state: {report}",
                )
                _expect(not helper.list_vms().vms, "Rejected VM leaked in helper registry")
            else:
                _expect(
                    result.phase == RunPhase.completed and result.exit_code == 0,
                    f"Healthy execution failed after rejection: {result}",
                )
                _expect(stdout == "mismatch-drill-first\n", f"Wrong healthy output: {stdout!r}")
                second, second_stdout = run(session.id, bundle, "mismatch-drill-reuse")
                _expect(second.phase == RunPhase.completed and second.exit_code == 0, f"Reuse failed: {second}")
                _expect(second_stdout == "mismatch-drill-reuse\n", f"Wrong reuse output: {second_stdout!r}")
                _expect(len(evidence["created_vms"]) == 2, "Healthy session booted an extra VM instead of reusing")
                healthy_id = evidence["created_vms"][1]["vm_id"]
                _expect(evidence["exec_vm_ids"] == [healthy_id, healthy_id], "Healthy session did not reuse its VM")
            _expect(service.destroy_session(session.id), "Session destruction failed")
            _expect(service.get_session(session.id) is None, "Session remained after destruction")
            sessions.remove(session.id)
        _expect(helper.ping().details.get("helper_instance_id") == generation, "Helper changed during recovery")
        report = service.macos_diagnostics()["reconciliation"]
        evidence["reconciliation_after_cleanup"] = report
        _expect(
            report["computed"] and report["persisted_sessions"] == 0 and report["live_vms"] == 0,
            f"Cleanup incomplete: {report}",
        )
    finally:
        try:
            for session_id in sessions:
                service.destroy_session(session_id)
            for vm_id in _owned_vm_ids(helper.list_vms().vms, evidence):
                helper.terminate_vm(vm_id)
            evidence["remaining_owned_vms"] = _owned_vm_ids(helper.list_vms().vms, evidence)
            _expect(not evidence["remaining_owned_vms"], "Owned VM cleanup failed")
        finally:
            (tmp_path / "guest-mismatch.json").write_text(json.dumps(evidence, indent=2) + "\n")
