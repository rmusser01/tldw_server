"""Manual real-VM wire protocol rejection proof, with a fresh guest challenge."""

from __future__ import annotations

import json
import os
import platform
import re
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

_STARTUP_TIMEOUT = 60
_INJECT_MISMATCH = True  # Test-only negative control changes only this flag.
_CHALLENGE_NAME = ".tldw-protocol-challenge.json"
_PROOF_NAME = ".tldw-protocol-proof.json"


def _require_protocol_bundle() -> Path:
    """Require a separate manual opt-in and a disposable protocol fault bundle."""
    if not is_truthy(os.getenv("TLDW_SANDBOX_VZ_LINUX_PROTOCOL_DRILL")):
        pytest.skip("Set TLDW_SANDBOX_VZ_LINUX_PROTOCOL_DRILL=1 for this manual drill (tracked in #1442)")
    path = os.getenv("TLDW_SANDBOX_VZ_LINUX_PROTOCOL_BASE_IMAGE", "").strip()
    _expect(bool(path), "A disposable protocol-mismatch bundle is required")
    bundle = Path(path).expanduser().resolve(strict=True)
    _expect(bundle.is_dir(), "Protocol image must be a disposable bundle directory")
    return bundle


def _read_protocol_proof(path: Path, nonce: str, vm_id: str, inject_mismatch: bool) -> dict[str, Any]:
    """Read bounded guest proof and correlate this create's requested handshake."""
    _expect(not path.is_symlink(), "Invalid protocol proof: symlink")
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0))
        try:
            _expect(stat.S_ISREG(os.fstat(fd).st_mode), "Invalid protocol proof: not a regular file")
            with os.fdopen(fd, "rb", closefd=False) as handle:
                raw = handle.read(4097)
        finally:
            os.close(fd)
        _expect(len(raw) <= 4096, "Invalid protocol proof: too large")
        proof = json.loads(raw)
    except (OSError, ValueError) as exc:
        pytest.fail(f"Invalid protocol proof: {exc}")
    _expect(
        isinstance(proof, dict)
        and re.fullmatch(r"[0-9a-f]{32}", nonce) is not None
        and proof.get("nonce") == nonce
        and isinstance(proof.get("vm_id"), str)
        and bool(vm_id.strip())
        and proof["vm_id"] == vm_id
        and proof.get("protocol_version") == ("999" if inject_mismatch else "1"),
        "Invalid protocol proof: expected this create's nonce, VM, and requested wire version",
    )
    return proof


@pytest.mark.unit
@pytest.mark.parametrize("opt_in", [None, "", "0", "false"])
def test_protocol_drill_requires_separate_opt_in(monkeypatch: pytest.MonkeyPatch, opt_in: str | None) -> None:
    """Neither normal E2E nor other fault drills may enable protocol injection."""
    for name in (
        "TLDW_SANDBOX_VZ_LINUX_E2E",
        "TLDW_SANDBOX_VZ_LINUX_READINESS_DRILL",
        "TLDW_SANDBOX_VZ_LINUX_GUEST_MISMATCH_DRILL",
    ):
        monkeypatch.setenv(name, "1")
    if opt_in is None:
        monkeypatch.delenv("TLDW_SANDBOX_VZ_LINUX_PROTOCOL_DRILL", raising=False)
    else:
        monkeypatch.setenv("TLDW_SANDBOX_VZ_LINUX_PROTOCOL_DRILL", opt_in)
    with pytest.raises(pytest.skip.Exception, match="PROTOCOL_DRILL"):
        _require_protocol_bundle()


@pytest.mark.unit
@pytest.mark.parametrize("bundle", [None, "", "   "])
def test_protocol_drill_requires_fault_bundle(monkeypatch: pytest.MonkeyPatch, bundle: str | None) -> None:
    """Opting in without the protocol bundle must fail rather than skip."""
    monkeypatch.setenv("TLDW_SANDBOX_VZ_LINUX_PROTOCOL_DRILL", "1")
    if bundle is None:
        monkeypatch.delenv("TLDW_SANDBOX_VZ_LINUX_PROTOCOL_BASE_IMAGE", raising=False)
    else:
        monkeypatch.setenv("TLDW_SANDBOX_VZ_LINUX_PROTOCOL_BASE_IMAGE", bundle)
    with pytest.raises(pytest.fail.Exception, match="protocol-mismatch bundle"):
        _require_protocol_bundle()


@pytest.mark.unit
def test_protocol_drill_rejects_non_directory_bundle(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A file is not a disposable guest bundle directory."""
    path = tmp_path / "bundle"
    path.write_text("not a bundle")
    monkeypatch.setenv("TLDW_SANDBOX_VZ_LINUX_PROTOCOL_DRILL", "1")
    monkeypatch.setenv("TLDW_SANDBOX_VZ_LINUX_PROTOCOL_BASE_IMAGE", str(path))
    with pytest.raises(pytest.fail.Exception, match="bundle directory"):
        _require_protocol_bundle()


@pytest.mark.unit
def test_protocol_drill_accepts_explicit_bundle(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Resolve an explicitly selected disposable bundle without touching a VM."""
    monkeypatch.setenv("TLDW_SANDBOX_VZ_LINUX_PROTOCOL_DRILL", "1")
    monkeypatch.setenv("TLDW_SANDBOX_VZ_LINUX_PROTOCOL_BASE_IMAGE", f" {tmp_path} ")
    _expect(_require_protocol_bundle() == tmp_path.resolve(), "Wrong protocol bundle")


@pytest.mark.unit
@pytest.mark.parametrize("inject_mismatch,version", [(True, "999"), (False, "1")])
def test_protocol_proof_accepts_requested_version(tmp_path: Path, inject_mismatch: bool, version: str) -> None:
    """The negative control's valid handshake must reach execution, not fail proof validation."""
    nonce = "a" * 32
    proof = {"nonce": nonce, "vm_id": "vm-test", "protocol_version": version}
    path = tmp_path / "proof.json"
    path.write_text(json.dumps(proof))
    _expect(_read_protocol_proof(path, nonce, "vm-test", inject_mismatch) == proof, "Proof changed")


@pytest.mark.unit
@pytest.mark.parametrize(
    "updates",
    [
        {"nonce": "b" * 32},
        {"nonce": None},
        {"vm_id": "other-vm"},
        {"vm_id": ""},
        {"vm_id": " vm-test "},
        {"vm_id": None},
        {"protocol_version": "1"},
        {"protocol_version": 999},
        {"protocol_version": None},
    ],
)
def test_protocol_proof_rejects_uncorrelated_guest(tmp_path: Path, updates: dict[str, Any]) -> None:
    """Stale nonces, wrong VMs, and wrong or untyped versions do not prove this attempt."""
    nonce = "a" * 32
    proof = {"nonce": nonce, "vm_id": "vm-test", "protocol_version": "999", **updates}
    path = tmp_path / "proof.json"
    path.write_text(json.dumps(proof))
    with pytest.raises(pytest.fail.Exception, match="Invalid protocol proof"):
        _read_protocol_proof(path, nonce, "vm-test", True)


@pytest.mark.unit
def test_protocol_proof_rejects_mismatch_for_negative_control(tmp_path: Path) -> None:
    """A negative run must prove it actually requested the supported wire version."""
    nonce = "a" * 32
    path = tmp_path / "proof.json"
    path.write_text(json.dumps({"nonce": nonce, "vm_id": "vm-test", "protocol_version": "999"}))
    with pytest.raises(pytest.fail.Exception, match="Invalid protocol proof"):
        _read_protocol_proof(path, nonce, "vm-test", False)


@pytest.mark.unit
@pytest.mark.parametrize("nonce", ["", "fresh", "g" * 32, "a" * 31, "a" * 33])
def test_protocol_proof_rejects_invalid_challenge_nonce(tmp_path: Path, nonce: str) -> None:
    """Even an echoed nonce must satisfy the 32-hex challenge contract."""
    path = tmp_path / "proof.json"
    path.write_text(json.dumps({"nonce": nonce, "vm_id": "vm-test", "protocol_version": "999"}))
    with pytest.raises(pytest.fail.Exception, match="Invalid protocol proof"):
        _read_protocol_proof(path, nonce, "vm-test", True)


@pytest.mark.unit
@pytest.mark.parametrize("raw", [None, b"not json", b"[]", b"null", b"{}", b"\xff", b" " * 4097])
def test_protocol_proof_requires_bounded_json(tmp_path: Path, raw: bytes | None) -> None:
    """Reject missing, malformed, non-object, incomplete, and oversized proof files."""
    path = tmp_path / "proof.json"
    if raw is not None:
        path.write_bytes(raw)
    with pytest.raises(pytest.fail.Exception, match="Invalid protocol proof"):
        _read_protocol_proof(path, "a" * 32, "vm-test", True)


@pytest.mark.unit
@pytest.mark.parametrize("size", [4096, 4097])
def test_protocol_proof_enforces_byte_boundary(tmp_path: Path, size: int) -> None:
    """Accept a complete proof at the limit, but reject even valid JSON beyond it."""
    nonce = "a" * 32
    proof = {"nonce": nonce, "vm_id": "vm-test", "protocol_version": "999"}
    path = tmp_path / "proof.json"
    path.write_text(json.dumps(proof).ljust(size))
    if size == 4096:
        _expect(_read_protocol_proof(path, nonce, "vm-test", True) == proof, "Boundary proof rejected")
    else:
        with pytest.raises(pytest.fail.Exception, match="Invalid protocol proof: too large"):
            _read_protocol_proof(path, nonce, "vm-test", True)


@pytest.mark.unit
@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="POSIX proof files (host coverage tracked in #1442)")
@pytest.mark.timeout(2, method="signal")
def test_protocol_proof_rejects_fifo_without_blocking(tmp_path: Path) -> None:
    """A guest-created FIFO must never wait for a writer during proof capture."""
    path = tmp_path / "proof.json"
    os.mkfifo(path)
    with pytest.raises(pytest.fail.Exception, match="Invalid protocol proof"):
        _read_protocol_proof(path, "a" * 32, "vm-test", True)


@pytest.mark.unit
@pytest.mark.skipif(os.name != "posix", reason="POSIX proof files (host coverage tracked in #1442)")
@pytest.mark.timeout(2, method="signal")
@pytest.mark.parametrize("target_kind", ["regular", "missing", "fifo"])
def test_protocol_proof_rejects_symlink(tmp_path: Path, target_kind: str) -> None:
    """Reject links to otherwise valid proof, absent files, and blocking FIFOs."""
    target = tmp_path / "target"
    if target_kind == "regular":
        target.write_text(json.dumps({"nonce": "a" * 32, "vm_id": "vm-test", "protocol_version": "999"}))
    elif target_kind == "fifo":
        os.mkfifo(target)
    path = tmp_path / "proof.json"
    path.symlink_to(target)
    with pytest.raises(pytest.fail.Exception, match="Invalid protocol proof: symlink"):
        _read_protocol_proof(path, "a" * 32, "vm-test", True)


@pytest.mark.unit
def test_protocol_proof_rejects_directory(tmp_path: Path) -> None:
    """Proof capture must reject a directory instead of treating it as a file."""
    with pytest.raises(pytest.fail.Exception, match="Invalid protocol proof"):
        _read_protocol_proof(tmp_path, "a" * 32, "vm-test", True)


@pytest.mark.integration
@pytest.mark.vz_linux_host_failure_drill
@pytest.mark.skipif(sys.platform != "darwin", reason="macOS host only (host coverage tracked in #1442)")
def test_vz_linux_protocol_mismatch_then_healthy_session_reuse(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Observe real rejection and recovery through a caller-owned isolated helper.

    The disposable overlay writes proof before its handshake. Observers capture
    that proof before runner cleanup, without replacing create or exec calls.
    Only this drill's sessions and VMs are cleaned, even on assertion failure.
    """
    fault_bundle = _require_protocol_bundle()
    if platform.machine() != "arm64":
        pytest.skip("Apple silicon host only (host coverage tracked in #1442)")
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
        """Challenge the fault guest and preserve proof even when create is rejected."""
        attempt = {field: request[field] for field in ("owner", "runtime", "run_id", "session_id")}
        evidence["attempted_creates"].append(attempt)
        is_fault = Path(request["template"]).resolve() == fault_bundle
        nonce = uuid4().hex
        proof_path = Path(request["workspace_path"]) / _PROOF_NAME
        if is_fault:
            _expect(not proof_path.exists() and not proof_path.is_symlink(), "Proof must not predate this create")
            attempt.update({"nonce": nonce, "vm_id": request["vm_name"], "inject_mismatch": _INJECT_MISMATCH})
            (proof_path.parent / _CHALLENGE_NAME).write_text(
                json.dumps({"nonce": nonce, "inject_mismatch": _INJECT_MISMATCH})
            )
        started = time.monotonic()
        try:
            vm = original_create(client, request)
            evidence["created_vms"].append({"vm_id": vm.vm_id})
            return vm
        except Exception as exc:
            attempt["error"] = str(exc)
            attempt["error_code"] = getattr(exc, "error_code", None)
            raise
        finally:
            attempt["elapsed_sec"] = time.monotonic() - started
            if is_fault:
                # Capture before the runner can delete a rejected create's workspace.
                evidence["protocol_proof"] = _read_protocol_proof(
                    proof_path, nonce, request["vm_name"], _INJECT_MISMATCH
                )

    def record_exec(client: Any, **kwargs: Any) -> Any:
        """Record actual dispatches without replacing the guest execution path."""
        evidence["exec_vm_ids"].append(kwargs["vm_id"])
        return original_exec(client, **kwargs)

    monkeypatch.setattr(VZLinuxRunner.helper_client_cls, "create_vm", record_create)
    monkeypatch.setattr(VZLinuxRunner.helper_client_cls, "exec_guest", record_exec)

    def run(session_id: str, bundle: str, token: str) -> tuple[Any, str]:
        """Execute through the service and retain the phase, exit status, and stdout."""
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
        """Record reconciliation and reject any reusable controls or registered VMs."""
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
            result, stdout = run(session.id, bundle, "protocol-drill-first")
            if bundle == str(fault_bundle):
                _expect("protocol_proof" in evidence, "No real guest protocol proof")
                # The negative control must get here after supported-version execution.
                _expect(result.phase == RunPhase.failed, f"Protocol mismatch was not rejected: {result}")
                _expect(len(evidence["attempted_creates"]) == 1, "Expected exactly one fault create attempt")
                attempt = evidence["attempted_creates"][0]
                _expect(attempt.get("error_code") == "guest_protocol_mismatch", f"Wrong create rejection: {attempt}")
                _expect(
                    result.message == f"vz_linux execution error: {attempt['error']}",
                    f"Wrong rejection reason: {result.message}",
                )
                _expect(not evidence["exec_vm_ids"] and stdout == "", "Execution reached the mismatched guest")
                _expect(not evidence["created_vms"], "A protocol-mismatched VM was returned as created")
                check_empty("reconciliation_after_rejection")
            else:
                _expect(result.phase == RunPhase.completed and result.exit_code == 0, f"Recovery failed: {result}")
                _expect(stdout == "protocol-drill-first\n", f"Wrong healthy output: {stdout!r}")
                second, output = run(session.id, bundle, "protocol-drill-reuse")
                _expect(second.phase == RunPhase.completed and second.exit_code == 0, f"Reuse failed: {second}")
                _expect(output == "protocol-drill-reuse\n", f"Wrong reuse output: {output!r}")
                _expect(len(evidence["attempted_creates"]) == 2, "Expected only fault and fresh healthy creates")
                _expect(len(evidence["created_vms"]) == 1, "Healthy session provisioned an extra VM")
                vm_id = evidence["created_vms"][0]["vm_id"]
                _expect(evidence["exec_vm_ids"] == [vm_id, vm_id], "Healthy commands did not reuse one VM")
                _expect(vm_id != evidence["protocol_proof"]["vm_id"], "Rejected VM reused")
            _expect(service.destroy_session(session.id), "Session destruction failed")
            _expect(service.get_session(session.id) is None, "Session still exists")
            sessions.remove(session.id)
        _expect(helper.ping().details.get("helper_instance_id") == generation, "Helper changed during recovery")
        check_empty("reconciliation_after_cleanup")
    finally:
        try:
            _cleanup_owned_resources(service, helper, sessions, evidence)
        finally:
            (tmp_path / "guest-protocol.json").write_text(json.dumps(evidence, indent=2) + "\n")
