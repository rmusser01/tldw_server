"""Manual advertised-workspace admission proof, not guest mount isolation."""

from __future__ import annotations

import json
import os
import platform
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
from tldw_Server_API.app.core.Sandbox.vz_guest_agent import classify_vz_linux_guest_agent
from tldw_Server_API.app.core.testing import is_truthy
from tldw_Server_API.tests.sandbox.test_vz_linux_guest_mismatch_host_gated import _cleanup_owned_resources
from tldw_Server_API.tests.sandbox.test_vz_linux_real_host_e2e import _expect, _require_vz_linux_real_host_e2e

_STARTUP_TIMEOUT = 60
_INJECT_MISMATCH = True  # Only the test plugin toggles this for the negative control.
_CHALLENGE_NAME = ".tldw-workspace-challenge.json"


def _require_workspace_bundle() -> Path:
    """Require independent consent and a disposable workspace fault bundle."""
    if not is_truthy(os.getenv("TLDW_SANDBOX_VZ_LINUX_WORKSPACE_DRILL")):
        pytest.skip("Set TLDW_SANDBOX_VZ_LINUX_WORKSPACE_DRILL=1 for this manual drill (tracked in #1442)")
    path = os.getenv("TLDW_SANDBOX_VZ_LINUX_WORKSPACE_BASE_IMAGE", "").strip()
    _expect(bool(path), "A disposable workspace-mismatch bundle is required")
    bundle = Path(path).expanduser().resolve(strict=True)
    _expect(bundle.is_dir(), "Workspace image must be a disposable bundle directory")
    return bundle


def _expect_workspace_metadata(created: dict[str, Any], attempt: dict[str, Any], generation: str) -> None:
    """Correlate real helper metadata with this create, not a guest proof file."""
    details = created["details"]
    guest = created["guest_agent"]
    _expect(bool(attempt["vm_name"]) and created["vm_id"] == attempt["vm_name"], "Wrong create VM identity")
    _expect(details.get("helper_instance_id") == generation, "Create came from a different helper generation")
    _expect(
        details.get("guest_workspace_root") == attempt["expected_workspace_root"]
        and guest["workspace_root"] == attempt["expected_workspace_root"],
        f"Wrong advertised workspace for this create: {created}",
    )
    _expect(
        guest["capabilities_known"] is True and {"exec", "output_cap_v1"}.issubset(guest["capabilities"]),
        f"Workspace drill must preserve supported capabilities: {guest}",
    )


@pytest.mark.unit
@pytest.mark.parametrize("opt_in", [None, "", "0", "false"])
def test_workspace_drill_requires_separate_opt_in(monkeypatch: pytest.MonkeyPatch, opt_in: str | None) -> None:
    """Other manual drill opt-ins cannot enable workspace fault injection."""
    for name in (
        "TLDW_SANDBOX_VZ_LINUX_E2E",
        "TLDW_SANDBOX_VZ_LINUX_READINESS_DRILL",
        "TLDW_SANDBOX_VZ_LINUX_GUEST_MISMATCH_DRILL",
        "TLDW_SANDBOX_VZ_LINUX_PROTOCOL_DRILL",
    ):
        monkeypatch.setenv(name, "1")
    if opt_in is None:
        monkeypatch.delenv("TLDW_SANDBOX_VZ_LINUX_WORKSPACE_DRILL", raising=False)
    else:
        monkeypatch.setenv("TLDW_SANDBOX_VZ_LINUX_WORKSPACE_DRILL", opt_in)
    with pytest.raises(pytest.skip.Exception, match="WORKSPACE_DRILL.*#1442"):
        _require_workspace_bundle()


@pytest.mark.unit
@pytest.mark.parametrize("bundle", [None, "", "   "])
def test_workspace_drill_requires_fault_bundle(monkeypatch: pytest.MonkeyPatch, bundle: str | None) -> None:
    """Missing workspace bundles fail explicitly rather than accepting a skip."""
    monkeypatch.setenv("TLDW_SANDBOX_VZ_LINUX_WORKSPACE_DRILL", "1")
    if bundle is None:
        monkeypatch.delenv("TLDW_SANDBOX_VZ_LINUX_WORKSPACE_BASE_IMAGE", raising=False)
    else:
        monkeypatch.setenv("TLDW_SANDBOX_VZ_LINUX_WORKSPACE_BASE_IMAGE", bundle)
    with pytest.raises(pytest.fail.Exception, match="workspace-mismatch bundle"):
        _require_workspace_bundle()


@pytest.mark.unit
@pytest.mark.parametrize("is_directory", [False, True])
def test_workspace_drill_requires_bundle_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, is_directory: bool
) -> None:
    """Resolve an explicit directory, but reject a file without touching a VM."""
    path = tmp_path / "bundle"
    if is_directory:
        path.mkdir()
    else:
        path.write_text("not a bundle")
    monkeypatch.setenv("TLDW_SANDBOX_VZ_LINUX_WORKSPACE_DRILL", "1")
    monkeypatch.setenv("TLDW_SANDBOX_VZ_LINUX_WORKSPACE_BASE_IMAGE", f" {path} ")
    if is_directory:
        _expect(_require_workspace_bundle() == path.resolve(), "Wrong workspace bundle")
    else:
        with pytest.raises(pytest.fail.Exception, match="bundle directory"):
            _require_workspace_bundle()


@pytest.mark.unit
@pytest.mark.parametrize("workspace_root", ["/workspace-mismatch/" + "a" * 32, "/workspace"])
@pytest.mark.parametrize(
    "updates,vm_id,error",
    [
        ({}, "vm-test", None),
        ({}, "other-vm", "VM identity"),
        ({"helper_instance_id": "other-helper"}, "vm-test", "helper generation"),
        ({"guest_workspace_root": "/workspace-mismatch/" + "b" * 32}, "vm-test", "advertised workspace"),
        ({"guest_workspace_root": None}, "vm-test", "advertised workspace"),
        ({"guest_capabilities_known": "false"}, "vm-test", "capabilities"),
        ({"guest_capabilities": "output_cap_v1"}, "vm-test", "capabilities"),
        ({"guest_capabilities": "exec"}, "vm-test", "capabilities"),
    ],
)
def test_workspace_metadata_requires_correlated_create(
    workspace_root: str, updates: dict[str, Any], vm_id: str, error: str | None
) -> None:
    """Both controls require exact VM/root correlation and intact capabilities."""
    details = {
        "helper_instance_id": "helper-test",
        "guest_workspace_root": workspace_root,
        "guest_capabilities_known": "true",
        "guest_capabilities": "exec,output_cap_v1",
        **updates,
    }
    created = {"vm_id": vm_id, "details": details, "guest_agent": classify_vz_linux_guest_agent(details)}
    attempt = {"vm_name": "vm-test", "expected_workspace_root": workspace_root}
    if error is None:
        _expect_workspace_metadata(created, attempt, "helper-test")
    else:
        with pytest.raises(pytest.fail.Exception, match=error):
            _expect_workspace_metadata(created, attempt, "helper-test")


@pytest.mark.integration
@pytest.mark.vz_linux_host_failure_drill
@pytest.mark.skipif(sys.platform != "darwin", reason="macOS host only (host coverage tracked in #1442)")
@pytest.mark.skipif(platform.machine() != "arm64", reason="Apple silicon host only (host coverage tracked in #1442)")
def test_vz_linux_workspace_mismatch_then_healthy_session_reuse(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Reject only advertised root mismatch; retain real execution and cleanup receipts.

    The caller owns the isolated helper and disposable overlay. Observers never
    replace helper replies or execution. A successful create already passed the
    helper's normal guest wire-protocol validation; the guest mount is unchanged.

    Args:
        monkeypatch: Restores temporary environment/settings changes and real
            helper-call observers after the test. Disables fake execution and
            configures synchronous execution with isolated SQLite state.
        tmp_path: Private test directory for sandbox state and the
            ``guest-workspace.json`` receipt written once the drill begins.

    Returns:
        None after rejection, healthy recovery, same-session reuse and cleanup
        meet all assertions. The negative control intentionally does not return.

    Raises:
        pytest.skip.Exception: The platform is unsupported, an opt-in is absent,
            or normal real-E2E helper/healthy-image preflight is unavailable.
        pytest.fail.Exception: Explicit fault-bundle/helper requirements or
            metadata, rejection, execution, reuse or cleanup assertions fail.
            The negative control deliberately fails the rejection assertion
            after supported-root execution succeeds with real admission enabled.
        OSError: Fault-bundle resolution or state/evidence file operations fail.

    Helper transport errors outside preflight propagate as test errors; neither
    errors nor skips count as acceptance in the enclosing operator workflow.
    """
    fault_bundle = _require_workspace_bundle()
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
        "profile": "workspace",
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
        """Challenge only the fault guest, preserving ownership even if its reply is lost."""
        is_fault = Path(request["template"]).resolve() == fault_bundle
        nonce = uuid4().hex if is_fault else None
        expected_root = f"/workspace-mismatch/{nonce}" if is_fault and _INJECT_MISMATCH else "/workspace"
        attempt = {field: request[field] for field in ("owner", "runtime", "run_id", "session_id", "vm_name")}
        attempt.update({"nonce": nonce, "expected_workspace_root": expected_root})
        evidence["attempted_creates"].append(attempt)
        if is_fault:
            (Path(request["workspace_path"]) / _CHALLENGE_NAME).write_text(
                json.dumps({"nonce": nonce, "inject_mismatch": _INJECT_MISMATCH})
            )
        started = time.monotonic()
        try:
            vm = original_create(client, request)
            evidence["created_vms"].append(
                {
                    **attempt,
                    "vm_id": vm.vm_id,
                    "details": dict(vm.details),
                    "guest_agent": classify_vz_linux_guest_agent(vm.details),
                }
            )
            return vm
        finally:
            attempt["elapsed_sec"] = time.monotonic() - started

    def record_exec(client: Any, **kwargs: Any) -> Any:
        """Observe real exec dispatches, including any forbidden fault-VM dispatch."""
        evidence["exec_vm_ids"].append(kwargs["vm_id"])
        return original_exec(client, **kwargs)

    monkeypatch.setattr(VZLinuxRunner.helper_client_cls, "create_vm", record_create)
    monkeypatch.setattr(VZLinuxRunner.helper_client_cls, "exec_guest", record_exec)

    def run(session_id: str, bundle: str, token: str) -> tuple[Any, str]:
        """Keep exact public service results and stdout for the parent receipt parser."""
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
                startup_timeout_sec=_STARTUP_TIMEOUT,
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
        """Use public reconciliation and helper inventory to exclude reusable state."""
        report = service.macos_diagnostics()["reconciliation"]
        evidence[label] = report
        _expect(
            report["computed"] and report["persisted_sessions"] == 0 and report["live_vms"] == 0,
            f"Reusable state remains: {report}",
        )
        _expect(not helper.list_vms().vms, "VM remains in helper registry")
        _expect(helper.ping().details.get("helper_instance_id") == generation, "Helper changed during recovery")

    try:
        for index, bundle in enumerate((str(fault_bundle), healthy_bundle)):
            session = service.create_session(
                user_id="e2e-user",
                spec=SessionSpec(runtime=RuntimeType.vz_linux, base_image=bundle, network_policy="deny_all"),
                spec_version="1.0",
                idem_key=None,
                raw_body={"runtime": "vz_linux", "base_image": bundle},
            )
            sessions.append(session.id)
            result, stdout = run(session.id, bundle, "workspace-drill-first")
            _expect(len(evidence["attempted_creates"]) == index + 1, "Expected exactly one create per session")
            _expect(len(evidence["created_vms"]) == index + 1, "Expected a real create reply, not preflight failure")
            created = evidence["created_vms"][index]
            _expect_workspace_metadata(created, evidence["attempted_creates"][index], generation)
            guest = created["guest_agent"]
            if index == 0:
                if not _INJECT_MISMATCH:
                    _expect(result.phase == RunPhase.completed and result.exit_code == 0, "Negative execution failed")
                    _expect(stdout == "workspace-drill-first\n", f"Wrong negative output: {stdout!r}")
                    _expect(evidence["exec_vm_ids"] == [created["vm_id"]], "Negative did not execute in the fault VM")
                # The root-only negative control must reach this unchanged assertion.
                _expect(result.phase == RunPhase.failed, f"Workspace mismatch was not rejected: {result}")
                _expect(
                    guest["reasons"] == ["vz_linux_guest_agent_workspace_mismatch"], f"Wrong injected fault: {guest}"
                )
                _expect(
                    result.message == "vz_linux execution error: vz_linux_guest_agent_workspace_mismatch",
                    f"Wrong rejection reason: {result.message}",
                )
                _expect(not evidence["exec_vm_ids"] and stdout == "", "Execution reached the mismatched guest")
                check_empty("reconciliation_after_rejection")
            else:
                fault = evidence["created_vms"][0]
                _expect(created["vm_id"] != fault["vm_id"], "Rejected VM reused")
                _expect(
                    guest["compatibility"] == "compatible" and guest["reasons"] == [], "Unhealthy replacement metadata"
                )
                _expect(
                    guest["version"] == fault["guest_agent"]["version"]
                    and guest["capabilities"] == fault["guest_agent"]["capabilities"],
                    "Workspace overlay changed guest version or capabilities",
                )
                _expect(result.phase == RunPhase.completed and result.exit_code == 0, f"Recovery failed: {result}")
                _expect(stdout == "workspace-drill-first\n", f"Wrong healthy output: {stdout!r}")
                second, output = run(session.id, bundle, "workspace-drill-reuse")
                _expect(second.phase == RunPhase.completed and second.exit_code == 0, f"Reuse failed: {second}")
                _expect(output == "workspace-drill-reuse\n", f"Wrong reuse output: {output!r}")
                _expect(len(evidence["attempted_creates"]) == 2, "Reuse attempted an extra create")
                _expect(len(evidence["created_vms"]) == 2, "Healthy session provisioned an extra VM")
                _expect(evidence["exec_vm_ids"] == [created["vm_id"], created["vm_id"]], "Healthy VM was not reused")
            _expect(service.destroy_session(session.id), "Session destruction failed")
            _expect(service.get_session(session.id) is None, "Session still exists")
            sessions.remove(session.id)
    finally:
        try:
            _cleanup_owned_resources(service, helper, sessions, evidence)
            check_empty("reconciliation_after_cleanup")
        finally:
            (tmp_path / "guest-workspace.json").write_text(json.dumps(evidence, indent=2) + "\n")
