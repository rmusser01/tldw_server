"""Opt-in real VZ initramfs stall, cleanup, recovery and same-session reuse.

The test-only PID1 wrapper proves early userspace entry via serial output before
rootfs mounting or agent startup. This is not a general kernel-hang test.
"""

from __future__ import annotations

import importlib.util
import json
import os
import platform
import re
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
_STALL_BOOT = True  # Only the explicit test plugin selects continue mode.


def _require_boot_stall_bundle() -> Path:
    """Require independent manual consent and a disposable initramfs bundle."""
    if not is_truthy(os.getenv("TLDW_SANDBOX_VZ_LINUX_BOOT_STALL_DRILL")):
        pytest.skip("Set TLDW_SANDBOX_VZ_LINUX_BOOT_STALL_DRILL=1 for this manual drill")
    raw = os.getenv("TLDW_SANDBOX_VZ_LINUX_BOOT_STALL_BASE_IMAGE", "").strip()
    _expect(bool(raw), "A disposable boot-stall bundle is required")
    bundle = Path(raw).expanduser().resolve(strict=True)
    _expect(bundle.is_dir(), "Boot-stall image must be a disposable bundle directory")
    return bundle


@pytest.mark.unit
def test_boot_stall_drill_requires_separate_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ordinary VZ smoke consent must not enable fault injection."""
    monkeypatch.setenv("TLDW_SANDBOX_VZ_LINUX_E2E", "1")
    monkeypatch.delenv("TLDW_SANDBOX_VZ_LINUX_BOOT_STALL_DRILL", raising=False)
    with pytest.raises(pytest.skip.Exception, match="BOOT_STALL_DRILL"):
        _require_boot_stall_bundle()


@pytest.mark.integration
@pytest.mark.vz_linux_host_failure_drill
@pytest.mark.skipif(sys.platform != "darwin", reason="macOS host only")
def test_vz_linux_boot_stall_then_healthy_session_reuse(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Prove a bounded early-userspace timeout and recovery without helper restart."""
    fault_bundle = _require_boot_stall_bundle()
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
    serial_root = os.getenv("TLDW_SANDBOX_VZ_LINUX_SERIAL_LOG_DIR", "").strip()
    _expect(bool(serial_root), "An isolated helper serial-log directory is required")
    serial_dir = Path(serial_root).resolve(strict=True)
    _expect(serial_dir.is_dir(), "Serial-log directory is unavailable")
    fixture_path = Path(__file__).resolve().parents[3] / "tools/macos-vz-helper/Tests/failure_drill/boot_stall.py"
    spec = importlib.util.spec_from_file_location("host_boot_stall", fixture_path)
    boot_fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(boot_fixture)
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
        """Challenge only this fault clone and require its fresh VM-correlated proof."""
        attempt = {field: request[field] for field in ("owner", "runtime", "run_id", "session_id")}
        vm_id = request["vm_name"]
        attempt["expected_vm_id"] = vm_id
        evidence["attempted_creates"].append(attempt)
        is_fault = Path(request["template"]).resolve() == fault_bundle
        mode = "stall" if _STALL_BOOT else "continue"
        nonce = uuid4().hex
        proof_path = serial_dir / (vm_id + ".serial.log")
        if is_fault:
            _expect(re.fullmatch(r"[a-zA-Z0-9._-]+", vm_id) is not None, "Unsafe VM log filename")
            _expect(not proof_path.exists() and not proof_path.is_symlink(), "Serial proof must not predate create")
            attempt["boot_nonce"] = nonce
            boot_fixture.append_archive(boot_fixture.initrd_path(fault_bundle), boot_fixture.challenge(nonce, mode))
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
                evidence["boot_proof"] = boot_fixture.read_proof(proof_path, nonce, mode, vm_id)
                evidence["serial_log"] = str(proof_path)

    def record_exec(client: Any, **kwargs: Any) -> Any:
        """Record real helper dispatch before delegating unchanged."""
        evidence["exec_vm_ids"].append(kwargs["vm_id"])
        return original_exec(client, **kwargs)

    monkeypatch.setattr(VZLinuxRunner.helper_client_cls, "create_vm", record_create)
    monkeypatch.setattr(VZLinuxRunner.helper_client_cls, "exec_guest", record_exec)

    def run(session_id: str, bundle: str, token: str, startup_timeout: int) -> tuple[Any, str]:
        """Record the actual sandbox result and buffered guest stdout."""
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
        stdout = "".join(
            str(frame.get("data", ""))
            for frame in get_hub().get_buffer_snapshot(result.id)
            if frame.get("type") == "stdout"
        )
        evidence["runs"].append(
            {"phase": result.phase.value, "exit_code": result.exit_code, "message": result.message, "stdout": stdout}
        )
        return result, stdout

    def check_empty(label: str) -> None:
        """Require both reusable control state and helper inventory to be empty."""
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
            result, stdout = run(session.id, bundle, "boot_stall-drill-first", _STARTUP_TIMEOUT if fault else 60)
            if fault:
                _expect(result.phase == RunPhase.failed, f"Boot stall did not fail: {result}")
                _expect("guest_transport_timeout" in (result.message or ""), f"Wrong failure: {result.message}")
                _expect(not evidence["exec_vm_ids"] and stdout == "", "Execution reached a stalled guest")
                _expect(not evidence["created_vms"], "An unready VM was returned as created")
                elapsed = evidence["attempted_creates"][0]["elapsed_sec"]
                _expect(_STARTUP_TIMEOUT <= elapsed < _STARTUP_TIMEOUT + 15, f"Unbounded/early timeout: {elapsed}")
                check_empty("reconciliation_after_timeout")
            else:
                _expect(result.phase == RunPhase.completed and result.exit_code == 0, f"Recovery failed: {result}")
                _expect(stdout == "boot_stall-drill-first\n", f"Wrong healthy output: {stdout!r}")
                second, output = run(session.id, bundle, "boot_stall-drill-reuse", 60)
                _expect(second.phase == RunPhase.completed and second.exit_code == 0, f"Reuse failed: {second}")
                _expect(output == "boot_stall-drill-reuse\n", f"Wrong reuse output: {output!r}")
                _expect(len(evidence["created_vms"]) == 1, "Healthy session provisioned an extra VM")
                vm_id = evidence["created_vms"][0]["vm_id"]
                _expect(evidence["exec_vm_ids"] == [vm_id, vm_id], "Healthy commands did not reuse one VM")
                _expect(vm_id != evidence["boot_proof"]["vm_id"], "Fault VM reused")
            _expect(service.destroy_session(session.id), "Session destruction failed")
            _expect(service.get_session(session.id) is None, "Session still exists")
            sessions.remove(session.id)
        _expect(helper.ping().details.get("helper_instance_id") == generation, "Helper changed during recovery")
        check_empty("reconciliation_after_cleanup")
    finally:
        try:
            _cleanup_owned_resources(service, helper, sessions, evidence)
        finally:
            (tmp_path / "guest-boot-stall.json").write_text(json.dumps(evidence, indent=2) + "\n")
