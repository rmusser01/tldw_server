"""Manual real-VM readiness timeout proof, with a fresh handshake challenge."""

from __future__ import annotations

import json
import os
import platform
import stat
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest

from tldw_Server_API.app.core.Sandbox.models import RunPhase, RunSpec, RuntimeType, SessionSpec
from tldw_Server_API.app.core.Sandbox.runners.vz_linux_runner import VZLinuxRunner
from tldw_Server_API.app.core.Sandbox.service import SandboxService
from tldw_Server_API.app.core.Sandbox.streams import get_hub
from tldw_Server_API.app.core.testing import is_truthy
from tldw_Server_API.tests.sandbox.test_vz_linux_guest_mismatch_host_gated import _owned_vm_ids
from tldw_Server_API.tests.sandbox.test_vz_linux_real_host_e2e import _expect, _require_vz_linux_real_host_e2e


_STARTUP_TIMEOUT = 15
_WITHHOLD_READY = True  # Test-only negative control may allow this guest to become ready.
_CHALLENGE_NAME = ".tldw-readiness-challenge.json"
_PROOF_NAME = ".tldw-readiness-proof.json"


def _cleanup_owned_resources(service: Any, helper: Any, sessions: list[str], evidence: dict[str, Any]) -> None:
    """Release only this test's resources, retaining failure evidence."""
    errors = []
    for session_id in sessions:
        try:
            service.destroy_session(session_id)
        except Exception as exc:
            errors.append(f"session {session_id}: {exc}")
        try:
            if service.get_session(session_id) is not None:
                errors.append(f"session {session_id}: still exists after deletion")
        except Exception as exc:
            errors.append(f"session {session_id} verification: {exc}")
    try:
        for vm_id in _owned_vm_ids(helper.list_vms().vms, evidence):
            try:
                helper.terminate_vm(vm_id)
            except Exception as exc:
                errors.append(f"VM {vm_id}: {exc}")
    except Exception as exc:
        errors.append(f"VM enumeration: {exc}")
    try:
        evidence["remaining_owned_vms"] = _owned_vm_ids(helper.list_vms().vms, evidence)
    except Exception as exc:
        evidence["remaining_owned_vms"] = None
        errors.append(f"VM verification: {exc}")
    evidence["cleanup_errors"] = errors
    _expect(not errors and evidence["remaining_owned_vms"] == [], f"Cleanup failed: {errors}")


def test_readiness_cleanup_continues_after_session_and_vm_errors() -> None:
    calls = []
    evidence = {"created_vms": [{"vm_id": "owned-a"}, {"vm_id": "owned-b"}], "attempted_creates": []}
    metadata = SimpleNamespace(owner="", runtime="", run_id="", session_id="")
    vms = [SimpleNamespace(vm_id=value, metadata=metadata) for value in ("owned-a", "owned-b", "other")]

    def destroy(session_id: str) -> None:
        calls.append(session_id)
        if session_id == "session-a":
            raise RuntimeError("session cleanup failed")

    def terminate(vm_id: str) -> None:
        calls.append(vm_id)
        if vm_id == "owned-a":
            raise RuntimeError("VM cleanup failed")
        vms[:] = [vm for vm in vms if vm.vm_id != vm_id]

    service = SimpleNamespace(destroy_session=destroy, get_session=lambda _: None)
    helper = SimpleNamespace(list_vms=lambda: SimpleNamespace(vms=vms), terminate_vm=terminate)
    with pytest.raises((RuntimeError, pytest.fail.Exception)):
        _cleanup_owned_resources(service, helper, ["session-a", "session-b"], evidence)
    _expect(calls == ["session-a", "session-b", "owned-a", "owned-b"], "Cleanup abandoned remaining owned resources")
    _expect(evidence["remaining_owned_vms"] == ["owned-a"], "Failed cleanup not recorded")


def test_readiness_cleanup_reports_session_that_survives_false_deletion() -> None:
    evidence = {"created_vms": [], "attempted_creates": []}
    service = SimpleNamespace(destroy_session=lambda _: False, get_session=lambda _: object())
    helper = SimpleNamespace(list_vms=lambda: SimpleNamespace(vms=[]))
    with pytest.raises(pytest.fail.Exception, match="Cleanup failed"):
        _cleanup_owned_resources(service, helper, ["remaining-session"], evidence)
    _expect(evidence["remaining_owned_vms"] == [], "VM verification was skipped")


def _require_readiness_bundle() -> Path:
    """Require a separate manual opt-in and a disposable fault bundle."""
    if not is_truthy(os.getenv("TLDW_SANDBOX_VZ_LINUX_READINESS_DRILL")):
        pytest.skip("Set TLDW_SANDBOX_VZ_LINUX_READINESS_DRILL=1 for this manual drill")
    path = os.getenv("TLDW_SANDBOX_VZ_LINUX_READINESS_BASE_IMAGE", "").strip()
    _expect(bool(path), "A disposable readiness-withholding bundle is required")
    bundle = Path(path).expanduser().resolve(strict=True)
    _expect(bundle.is_dir(), "Readiness image must be a disposable bundle directory")
    return bundle


def _read_handshake_proof(path: Path, nonce: str) -> dict[str, Any]:
    """Read the test-only guest's acknowledgement proof before workspace cleanup."""
    _expect(not path.is_symlink(), "Invalid handshake proof: symlink")
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0))
        try:
            _expect(stat.S_ISREG(os.fstat(fd).st_mode), "Invalid handshake proof: not a regular file")
            with os.fdopen(fd, "rb", closefd=False) as handle:
                raw = handle.read(4097)
        finally:
            os.close(fd)
        _expect(len(raw) <= 4096, "Invalid handshake proof: too large")
        proof = json.loads(raw)
    except (OSError, ValueError) as exc:
        pytest.fail(f"Invalid handshake proof: {exc}")
    _expect(
        isinstance(proof, dict)
        and proof.get("nonce") == nonce
        and proof.get("handshake_acknowledged") is True
        and isinstance(proof.get("vm_id"), str)
        and bool(proof["vm_id"].strip()),
        "Invalid handshake proof: expected this run's acknowledged VM",
    )
    return proof


@pytest.mark.parametrize(
    "payload",
    [
        {"nonce": "stale", "vm_id": "vm-test", "handshake_acknowledged": True},
        {"nonce": "fresh", "vm_id": "vm-test", "handshake_acknowledged": False},
        {"nonce": "fresh", "vm_id": "", "handshake_acknowledged": True},
    ],
)
def test_readiness_proof_rejects_stale_or_unacknowledged_guest(tmp_path: Path, payload: dict[str, Any]) -> None:
    path = tmp_path / "proof.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(pytest.fail.Exception, match="Invalid handshake proof"):
        _read_handshake_proof(path, "fresh")


def test_readiness_proof_accepts_matching_acknowledged_guest(tmp_path: Path) -> None:
    proof = {"nonce": "fresh", "vm_id": "vm-test", "handshake_acknowledged": True}
    path = tmp_path / "proof.json"
    path.write_text(json.dumps(proof))
    _expect(_read_handshake_proof(path, "fresh") == proof, "Proof changed")


@pytest.mark.parametrize("raw", [None, "not json", "[]", " " * 4097])
def test_readiness_proof_requires_bounded_json(tmp_path: Path, raw: str | None) -> None:
    path = tmp_path / "proof.json"
    if raw is not None:
        path.write_text(raw)
    with pytest.raises(pytest.fail.Exception, match="Invalid handshake proof"):
        _read_handshake_proof(path, "fresh")


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="POSIX proof files")
@pytest.mark.timeout(2, method="signal")
def test_readiness_proof_rejects_fifo_without_blocking(tmp_path: Path) -> None:
    path = tmp_path / "proof.json"
    os.mkfifo(path)
    with pytest.raises(pytest.fail.Exception, match="Invalid handshake proof"):
        _read_handshake_proof(path, "fresh")


def test_readiness_drill_requires_separate_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TLDW_SANDBOX_VZ_LINUX_E2E", "1")
    monkeypatch.delenv("TLDW_SANDBOX_VZ_LINUX_READINESS_DRILL", raising=False)
    with pytest.raises(pytest.skip.Exception, match="READINESS_DRILL"):
        _require_readiness_bundle()


def test_readiness_drill_requires_fault_bundle(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TLDW_SANDBOX_VZ_LINUX_READINESS_DRILL", "1")
    monkeypatch.delenv("TLDW_SANDBOX_VZ_LINUX_READINESS_BASE_IMAGE", raising=False)
    with pytest.raises(pytest.fail.Exception, match="readiness-withholding"):
        _require_readiness_bundle()


@pytest.mark.integration
@pytest.mark.vz_linux_host_failure_drill
@pytest.mark.skipif(sys.platform != "darwin", reason="macOS host only")
def test_vz_linux_readiness_timeout_then_healthy_session_reuse(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Use a caller-owned isolated helper; observe real calls and clean only owned VMs."""
    fault_bundle = _require_readiness_bundle()
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
        attempt = {field: request[field] for field in ("owner", "runtime", "run_id", "session_id")}
        evidence["attempted_creates"].append(attempt)
        is_fault = Path(request["template"]).resolve() == fault_bundle
        nonce = uuid4().hex
        proof_path = Path(request["workspace_path"]) / _PROOF_NAME
        if is_fault:
            _expect(not proof_path.exists(), "Proof must not predate this create")
            (proof_path.parent / _CHALLENGE_NAME).write_text(
                json.dumps({"nonce": nonce, "withhold_ready": _WITHHOLD_READY})
            )
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
                # Capture before the runner can delete a failed run's workspace.
                evidence["handshake_proof"] = _read_handshake_proof(proof_path, nonce)

    def record_exec(client: Any, **kwargs: Any) -> Any:
        evidence["exec_vm_ids"].append(kwargs["vm_id"])
        return original_exec(client, **kwargs)

    monkeypatch.setattr(VZLinuxRunner.helper_client_cls, "create_vm", record_create)
    monkeypatch.setattr(VZLinuxRunner.helper_client_cls, "exec_guest", record_exec)

    def run(session_id: str, bundle: str, token: str, startup_timeout: int) -> tuple[Any, str]:
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
            result, stdout = run(session.id, bundle, "readiness-drill-first", _STARTUP_TIMEOUT if fault else 60)
            if fault:
                _expect("handshake_proof" in evidence, "No real acknowledged guest handshake")
                _expect(result.phase == RunPhase.failed, f"Readiness withholding did not fail: {result}")
                _expect("guest_transport_timeout" in (result.message or ""), f"Wrong failure: {result.message}")
                _expect(not evidence["exec_vm_ids"] and stdout == "", "Execution reached an unready guest")
                _expect(not evidence["created_vms"], "An unready VM was returned as created")
                elapsed = evidence["attempted_creates"][0]["elapsed_sec"]
                _expect(_STARTUP_TIMEOUT <= elapsed < _STARTUP_TIMEOUT + 15, f"Unbounded/early timeout: {elapsed}")
                check_empty("reconciliation_after_timeout")
            else:
                _expect(result.phase == RunPhase.completed and result.exit_code == 0, f"Recovery failed: {result}")
                _expect(stdout == "readiness-drill-first\n", f"Wrong healthy output: {stdout!r}")
                second, output = run(session.id, bundle, "readiness-drill-reuse", 60)
                _expect(second.phase == RunPhase.completed and second.exit_code == 0, f"Reuse failed: {second}")
                _expect(output == "readiness-drill-reuse\n", f"Wrong reuse output: {output!r}")
                _expect(len(evidence["created_vms"]) == 1, "Healthy session provisioned an extra VM")
                vm_id = evidence["created_vms"][0]["vm_id"]
                _expect(evidence["exec_vm_ids"] == [vm_id, vm_id], "Healthy commands did not reuse one VM")
                _expect(vm_id != evidence["handshake_proof"]["vm_id"], "Fault VM reused")
            _expect(service.destroy_session(session.id), "Session destruction failed")
            _expect(service.get_session(session.id) is None, "Session still exists")
            sessions.remove(session.id)
        _expect(helper.ping().details.get("helper_instance_id") == generation, "Helper changed during recovery")
        check_empty("reconciliation_after_cleanup")
    finally:
        try:
            _cleanup_owned_resources(service, helper, sessions, evidence)
        finally:
            (tmp_path / "guest-readiness.json").write_text(json.dumps(evidence, indent=2) + "\n")
