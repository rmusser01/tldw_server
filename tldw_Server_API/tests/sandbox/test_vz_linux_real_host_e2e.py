from __future__ import annotations

import dataclasses
import os
import platform
import signal
import socket
import stat
import subprocess  # nosec B404
import sys
import time
from pathlib import Path

import pytest

from tldw_Server_API.app.core.config import clear_config_cache, settings as app_settings
from tldw_Server_API.app.core.Sandbox.macos_virtualization.helper_client import (
    MacOSVirtualizationHelperUnavailable,
)
from tldw_Server_API.app.core.Sandbox.macos_virtualization.models import HelperPingReply
from tldw_Server_API.app.core.Sandbox.models import RunPhase, RunSpec, RuntimeType, SessionSpec
from tldw_Server_API.app.core.Sandbox.runners.vz_linux_runner import VZLinuxRunner
from tldw_Server_API.app.core.Sandbox.service import SandboxService
from tldw_Server_API.app.core.Sandbox.streams import get_hub
from tldw_Server_API.app.core.testing import is_truthy


def _expect(condition: bool, message: str) -> None:
    if not condition:
        pytest.fail(message)


def _configure_sqlite_store(monkeypatch, tmp_path: Path) -> None:
    db_path = str(tmp_path / "sandbox_store.db")
    root_dir = str(tmp_path / "sandbox_root")
    snapshot_dir = str(tmp_path / "snapshots")
    monkeypatch.setenv("SANDBOX_STORE_BACKEND", "sqlite")
    monkeypatch.setenv("SANDBOX_STORE_DB_PATH", db_path)
    monkeypatch.setenv("SANDBOX_ROOT_DIR", root_dir)
    monkeypatch.setenv("SANDBOX_SNAPSHOT_PATH", snapshot_dir)
    if hasattr(app_settings, "SANDBOX_STORE_BACKEND"):
        monkeypatch.setattr(app_settings, "SANDBOX_STORE_BACKEND", "sqlite")
    if hasattr(app_settings, "SANDBOX_STORE_DB_PATH"):
        monkeypatch.setattr(app_settings, "SANDBOX_STORE_DB_PATH", db_path)
    if hasattr(app_settings, "SANDBOX_ROOT_DIR"):
        monkeypatch.setattr(app_settings, "SANDBOX_ROOT_DIR", root_dir)
    if hasattr(app_settings, "SANDBOX_SNAPSHOT_PATH"):
        monkeypatch.setattr(app_settings, "SANDBOX_SNAPSHOT_PATH", snapshot_dir)
    clear_config_cache()


def _require_vz_linux_real_host_e2e(monkeypatch, tmp_path: Path) -> str:
    _configure_sqlite_store(monkeypatch, tmp_path)
    if sys.platform != "darwin":
        pytest.skip("macOS host only")
    if platform.machine() != "arm64":
        pytest.skip("Apple silicon host only")
    if not is_truthy(os.getenv("TLDW_SANDBOX_VZ_LINUX_E2E")):
        pytest.skip("Set TLDW_SANDBOX_VZ_LINUX_E2E=1 to enable this test")
    base_image = str(os.getenv("TLDW_SANDBOX_VZ_LINUX_E2E_BASE_IMAGE") or "").strip()
    if not base_image:
        pytest.skip(
            "TLDW_SANDBOX_VZ_LINUX_E2E_BASE_IMAGE is required; prefer the "
            "canonical bundle output from tools/vz-linux-image/scripts/build-debian-bundle.sh"
        )
    monkeypatch.setenv("SANDBOX_ENABLE_EXECUTION", "1")
    monkeypatch.setenv("SANDBOX_BACKGROUND_EXECUTION", "0")
    helper = VZLinuxRunner.helper_client_cls()
    try:
        ping = helper.ping()
    except MacOSVirtualizationHelperUnavailable as exc:
        pytest.skip(f"vz_linux helper unavailable for ping: {exc}")
    _expect(bool(str(ping.protocol_version).strip()), "Expected helper protocol_version from ping")
    try:
        validation = helper.validate_template(
            {"runtime": RuntimeType.vz_linux.value, "template": base_image}
        )
    except MacOSVirtualizationHelperUnavailable as exc:
        pytest.skip(f"vz_linux helper unavailable for template validation: {exc}")
    if not bool(validation.get("ready")):
        template_reasons = [str(reason) for reason in validation.get("reasons", []) if str(reason).strip()]
        reason_text = ", ".join(template_reasons) if template_reasons else "template_invalid"
        pytest.skip(f"vz_linux template validation unavailable: {reason_text}")
    expected_boot_mode = "raw_disk" if Path(base_image).suffix == ".img" else "bundle"
    expected_validation_strength = "compatibility" if expected_boot_mode == "raw_disk" else "strong"
    _expect(
        validation.get("boot_mode") == expected_boot_mode,
        f"Expected helper validation boot_mode {expected_boot_mode!r}, got {validation.get('boot_mode')!r}",
    )
    _expect(
        validation.get("validation_strength") == expected_validation_strength,
        "Expected helper validation_strength "
        f"{expected_validation_strength!r}, got {validation.get('validation_strength')!r}",
    )
    return base_image


@dataclasses.dataclass(frozen=True)
class _HelperRestartLease:
    helper_path: Path
    socket_path: Path
    serial_log_dir: Path
    pid_file: Path


def _require_helper_restart_lease() -> _HelperRestartLease:
    if not is_truthy(os.getenv("TLDW_SANDBOX_VZ_LINUX_E2E_HELPER_RESTART_ALLOWED")):
        pytest.skip("Set TLDW_SANDBOX_VZ_LINUX_E2E_HELPER_RESTART_ALLOWED=1 to enable helper restart drill")
    helper_text = str(os.getenv("TLDW_SANDBOX_MACOS_HELPER_BINARY") or "").strip()
    socket_text = str(os.getenv("TLDW_SANDBOX_MACOS_HELPER_SOCKET") or "").strip()
    serial_log_text = str(os.getenv("TLDW_SANDBOX_VZ_LINUX_SERIAL_LOG_DIR") or "").strip()
    pid_file_text = str(os.getenv("TLDW_SANDBOX_VZ_LINUX_E2E_HELPER_PID_FILE") or "").strip()
    if not helper_text or not socket_text or not serial_log_text or not pid_file_text:
        pytest.skip("helper restart drill requires helper binary, socket, serial log dir, and pid file env")
    return _HelperRestartLease(
        helper_path=Path(helper_text).expanduser(),
        socket_path=Path(socket_text).expanduser(),
        serial_log_dir=Path(serial_log_text).expanduser(),
        pid_file=Path(pid_file_text).expanduser(),
    )


def _lookup_process_command(pid: int) -> str | None:
    completed = subprocess.run(  # nosec B603, B607
        ["ps", "-p", str(pid), "-o", "command="],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    command = completed.stdout.strip()
    return command or None


def _read_valid_restart_pid(
    lease: _HelperRestartLease,
    *,
    process_lookup=_lookup_process_command,
) -> int:
    socket_dir = lease.socket_path.parent.resolve()
    try:
        pid_parent = lease.pid_file.parent.resolve()
    except OSError as exc:
        pytest.fail(f"helper restart pid file parent is invalid: {exc}")
    if pid_parent != socket_dir:
        pytest.fail("helper restart pid file must be inside the private socket directory")
    try:
        stat_result = lease.pid_file.lstat()
    except OSError as exc:
        pytest.fail(f"helper restart pid file is unavailable: {exc}")
    if not stat.S_ISREG(stat_result.st_mode) or lease.pid_file.is_symlink():
        pytest.fail("helper restart pid file must be a regular non-symlink file")
    if stat_result.st_mode & 0o077:
        pytest.fail("helper restart pid file must be owner-only")
    try:
        raw_pid = lease.pid_file.read_text(encoding="utf-8").strip()
    except OSError as exc:
        pytest.fail(f"helper restart pid file could not be read: {exc}")
    if not raw_pid.isdigit() or int(raw_pid) <= 0:
        pytest.fail("helper restart pid file does not contain a positive PID")
    pid = int(raw_pid)
    command = process_lookup(pid)
    if command is None:
        pytest.skip("helper process exited before restart drill could stop it")
    if str(lease.helper_path) not in command and lease.helper_path.name not in command:
        pytest.fail("helper restart pid file points at a non-helper process")
    return pid


def _wait_for_helper_socket_unavailable(socket_path: Path, timeout_sec: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if not socket_path.exists():
            return
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                client.settimeout(0.2)
                client.connect(str(socket_path))
        except OSError:
            return
        time.sleep(0.05)
    pytest.fail(f"helper socket remained available after helper stop: {socket_path}")


def _wait_for_helper_ping(helper, timeout_sec: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_sec
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            helper.ping()
            return
        except Exception as exc:
            last_error = exc
            time.sleep(0.1)
    pytest.fail(f"replacement helper did not answer ping: {last_error}")


def _restart_helper_for_drill(
    lease: _HelperRestartLease,
    *,
    helper_client_factory=VZLinuxRunner.helper_client_cls,
    process_lookup=_lookup_process_command,
    startup_timeout_sec: float = 10.0,
) -> subprocess.Popen[str]:
    old_pid = _read_valid_restart_pid(lease, process_lookup=process_lookup)
    os.kill(old_pid, signal.SIGTERM)
    _wait_for_helper_socket_unavailable(lease.socket_path)

    env = os.environ.copy()
    env["TLDW_SANDBOX_MACOS_HELPER_SOCKET"] = str(lease.socket_path)
    env["TLDW_SANDBOX_VZ_LINUX_SERIAL_LOG_DIR"] = str(lease.serial_log_dir)
    stdout_path = lease.serial_log_dir / "helper.restart.stdout.log"
    stderr_path = lease.serial_log_dir / "helper.restart.stderr.log"
    with stdout_path.open("ab") as stdout, stderr_path.open("ab") as stderr:
        replacement = subprocess.Popen(  # nosec B603
            [str(lease.helper_path)],
            env=env,
            stdout=stdout,
            stderr=stderr,
        )
    lease.pid_file.write_text(f"{replacement.pid}\n", encoding="utf-8")
    lease.pid_file.chmod(0o600)
    try:
        _wait_for_helper_ping(helper_client_factory(), timeout_sec=startup_timeout_sec)
    except Exception:
        if replacement.poll() is None:
            replacement.terminate()
            try:
                replacement.wait(timeout=5)
            except subprocess.TimeoutExpired:
                replacement.kill()
                replacement.wait(timeout=5)
        raise
    return replacement


def test_vz_linux_real_host_e2e_requires_opt_in(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    monkeypatch.delenv("TLDW_SANDBOX_VZ_LINUX_E2E", raising=False)

    with pytest.raises(pytest.skip.Exception, match="TLDW_SANDBOX_VZ_LINUX_E2E"):
        _require_vz_linux_real_host_e2e(monkeypatch, tmp_path)


def test_vz_linux_real_host_e2e_requires_base_image_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    monkeypatch.setenv("TLDW_SANDBOX_VZ_LINUX_E2E", "1")
    monkeypatch.delenv("TLDW_SANDBOX_VZ_LINUX_E2E_BASE_IMAGE", raising=False)

    with pytest.raises(pytest.skip.Exception, match="BASE_IMAGE"):
        _require_vz_linux_real_host_e2e(monkeypatch, tmp_path)


def test_vz_linux_real_host_e2e_requires_helper_validated_template(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    monkeypatch.setenv("TLDW_SANDBOX_VZ_LINUX_E2E", "1")
    monkeypatch.setenv("TLDW_SANDBOX_VZ_LINUX_E2E_BASE_IMAGE", "bad-template")

    class _FailingHelper:
        def ping(self):
            return HelperPingReply(
                protocol_version="1",
                helper_version="test-mode",
                status="ok",
                details={"transport": "fake"},
            )

        def validate_template(self, request: dict[str, object]) -> dict[str, object]:
            assert request["template"] == "bad-template"
            return {"ready": False, "reasons": ["template_invalid"]}

    monkeypatch.setattr(VZLinuxRunner, "helper_client_cls", _FailingHelper)

    with pytest.raises(pytest.skip.Exception, match="template_invalid"):
        _require_vz_linux_real_host_e2e(monkeypatch, tmp_path)


def test_vz_linux_real_host_e2e_requires_helper_ping(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    monkeypatch.setenv("TLDW_SANDBOX_VZ_LINUX_E2E", "1")
    monkeypatch.setenv("TLDW_SANDBOX_VZ_LINUX_E2E_BASE_IMAGE", "ubuntu-24.04")

    class _UnavailableHelper:
        def ping(self):
            raise MacOSVirtualizationHelperUnavailable("macos_virtualization_helper_unavailable")

    monkeypatch.setattr(VZLinuxRunner, "helper_client_cls", _UnavailableHelper)

    with pytest.raises(pytest.skip.Exception, match="helper unavailable for ping"):
        _require_vz_linux_real_host_e2e(monkeypatch, tmp_path)


def test_helper_restart_lease_requires_explicit_opt_in(monkeypatch) -> None:
    monkeypatch.delenv("TLDW_SANDBOX_VZ_LINUX_E2E_HELPER_RESTART_ALLOWED", raising=False)

    with pytest.raises(pytest.skip.Exception, match="HELPER_RESTART_ALLOWED"):
        _require_helper_restart_lease()


def test_helper_restart_pid_file_rejects_symlink(tmp_path: Path) -> None:
    helper = tmp_path / "macos-vz-helper"
    helper.write_text("#!/bin/sh\n", encoding="utf-8")
    helper.chmod(0o755)
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir(mode=0o700)
    target = runtime_dir / "target.pid"
    target.write_text("1234\n", encoding="utf-8")
    target.chmod(0o600)
    pid_file = runtime_dir / "helper.pid"
    pid_file.symlink_to(target)
    lease = _HelperRestartLease(helper, runtime_dir / "helper.sock", runtime_dir / "serial", pid_file)

    with pytest.raises(pytest.fail.Exception, match="pid file"):
        _read_valid_restart_pid(lease, process_lookup=lambda _pid: str(helper))


def test_restart_helper_for_drill_replaces_pid_file_and_stops_old_helper(tmp_path: Path) -> None:
    helper = tmp_path / "macos-vz-helper"
    helper.write_text(
        "#!/usr/bin/env python3\n"
        "import signal\n"
        "import sys\n"
        "import time\n"
        "signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))\n"
        "while True:\n"
        "    time.sleep(0.1)\n",
        encoding="utf-8",
    )
    helper.chmod(0o755)
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir(mode=0o700)
    serial_log_dir = runtime_dir / "serial"
    serial_log_dir.mkdir(mode=0o700)
    pid_file = runtime_dir / "helper.pid"
    old_proc = subprocess.Popen(  # nosec B603
        [str(helper)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    pid_file.write_text(f"{old_proc.pid}\n", encoding="utf-8")
    pid_file.chmod(0o600)
    lease = _HelperRestartLease(helper, runtime_dir / "helper.sock", serial_log_dir, pid_file)

    class _PingHelper:
        def ping(self) -> HelperPingReply:
            return HelperPingReply(
                protocol_version="1",
                helper_version="test-mode",
                status="ok",
                details={"transport": "fake"},
            )

    replacement_proc: subprocess.Popen[str] | None = None
    try:
        replacement_proc = _restart_helper_for_drill(
            lease,
            helper_client_factory=_PingHelper,
            process_lookup=lambda _pid: str(helper),
            startup_timeout_sec=1.0,
        )
        _expect(replacement_proc.poll() is None, "Expected replacement helper to remain running")
        _expect(
            int(pid_file.read_text(encoding="utf-8").strip()) == replacement_proc.pid,
            "Expected restart helper to update pid file to replacement process",
        )
        old_proc.wait(timeout=5)
    finally:
        if replacement_proc is not None and replacement_proc.poll() is None:
            replacement_proc.terminate()
            replacement_proc.wait(timeout=5)
        if old_proc.poll() is None:
            old_proc.terminate()
            old_proc.wait(timeout=5)


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS host only")
@pytest.mark.vz_linux_host_smoke
def test_vz_linux_real_ephemeral_run_smoke(monkeypatch, tmp_path: Path) -> None:
    base_image = _require_vz_linux_real_host_e2e(monkeypatch, tmp_path)
    monkeypatch.delenv("TEST_MODE", raising=False)
    monkeypatch.delenv("TLDW_SANDBOX_VZ_LINUX_FAKE_EXEC", raising=False)
    monkeypatch.delenv("TLDW_SANDBOX_MACOS_HELPER_READY", raising=False)
    monkeypatch.delenv("TLDW_SANDBOX_VZ_LINUX_TEMPLATE_READY", raising=False)
    monkeypatch.delenv("TLDW_SANDBOX_VZ_LINUX_AVAILABLE", raising=False)
    runner = VZLinuxRunner()
    preflight = runner.preflight(network_policy="deny_all")
    if not preflight.available or preflight.execution_mode != "real":
        pytest.skip(f"vz_linux real execution unavailable: {preflight.reasons}")

    run_id = "vz-linux-real-ephemeral"
    hub = get_hub()
    hub._buffers.pop(run_id, None)  # type: ignore[attr-defined]
    status = runner.start_run(
        run_id,
        RunSpec(
            session_id=None,
            runtime=RuntimeType.vz_linux,
            base_image=base_image,
            command=["/bin/echo", "vz-linux-e2e"],
            network_policy="deny_all",
        ),
        session_workspace=None,
    )

    frames = list(hub._buffers.get(run_id, []))  # type: ignore[attr-defined]
    stdout_text = "".join(
        str(frame.get("data", ""))
        for frame in frames
        if frame.get("type") == "stdout"
    )
    _expect(status.phase == RunPhase.completed, f"Expected completed phase, got {status.phase!r}")
    _expect(status.exit_code == 0, f"Expected exit code 0, got {status.exit_code!r}")
    _expect("vz-linux-e2e" in stdout_text, f"Expected stdout token in output, got {stdout_text!r}")


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS host only")
@pytest.mark.vz_linux_host_smoke
def test_vz_linux_real_session_reuse_smoke(monkeypatch, tmp_path: Path) -> None:
    base_image = _require_vz_linux_real_host_e2e(monkeypatch, tmp_path)
    monkeypatch.delenv("TEST_MODE", raising=False)
    monkeypatch.delenv("TLDW_SANDBOX_VZ_LINUX_FAKE_EXEC", raising=False)
    monkeypatch.delenv("TLDW_SANDBOX_MACOS_HELPER_READY", raising=False)
    monkeypatch.delenv("TLDW_SANDBOX_VZ_LINUX_TEMPLATE_READY", raising=False)
    monkeypatch.delenv("TLDW_SANDBOX_VZ_LINUX_AVAILABLE", raising=False)
    runner = VZLinuxRunner()
    preflight = runner.preflight(network_policy="deny_all")
    if not preflight.available or preflight.execution_mode != "real":
        pytest.skip(f"vz_linux real execution unavailable: {preflight.reasons}")

    service = SandboxService()
    session_id: str | None = None
    destroyed = False
    try:
        session = service.create_session(
            user_id="e2e-user",
            spec=SessionSpec(
                runtime=RuntimeType.vz_linux,
                base_image=base_image,
                network_policy="deny_all",
            ),
            spec_version="1.0",
            idem_key=None,
            raw_body={"spec_version": "1.0", "runtime": "vz_linux", "base_image": base_image},
        )
        session_id = session.id

        first = service.start_run_scaffold(
            user_id="e2e-user",
            spec=RunSpec(
                session_id=session.id,
                runtime=RuntimeType.vz_linux,
                base_image=base_image,
                command=["/bin/echo", "first"],
                network_policy="deny_all",
            ),
            spec_version="1.0",
            idem_key=None,
            raw_body={"session_id": session.id, "runtime": "vz_linux", "command": ["/bin/echo", "first"]},
        )
        control_after_first = service._orch.get_vz_session_control(session.id)

        second = service.start_run_scaffold(
            user_id="e2e-user",
            spec=RunSpec(
                session_id=session.id,
                runtime=RuntimeType.vz_linux,
                base_image=base_image,
                command=["/bin/echo", "second"],
                network_policy="deny_all",
            ),
            spec_version="1.0",
            idem_key=None,
            raw_body={"session_id": session.id, "runtime": "vz_linux", "command": ["/bin/echo", "second"]},
        )
        control_after_second = service._orch.get_vz_session_control(session.id)

        _expect(first.phase == RunPhase.completed, f"Expected first run completed, got {first.phase!r}")
        _expect(second.phase == RunPhase.completed, f"Expected second run completed, got {second.phase!r}")
        _expect(control_after_first is not None, "Expected VZ session control after first run")
        _expect(control_after_second is not None, "Expected VZ session control after second run")
        _expect(
            control_after_first["vm_id"] == control_after_second["vm_id"],
            "Expected second run to reuse the same vz_linux vm_id",
        )
        _expect(service.destroy_session(session.id) is True, "Expected session destruction to succeed")
        destroyed = True
        _expect(service._orch.get_vz_session_control(session.id) is None, "Expected VZ session control cleanup")
    finally:
        if session_id and not destroyed:
            service.destroy_session(session_id)


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS host only")
@pytest.mark.vz_linux_host_smoke
def test_vz_linux_real_recovery_diagnostics_dry_run_smoke(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify real-helper diagnostics and dry-run repair planning stay non-destructive."""
    base_image = _require_vz_linux_real_host_e2e(monkeypatch, tmp_path)
    monkeypatch.delenv("TEST_MODE", raising=False)
    monkeypatch.delenv("TLDW_SANDBOX_VZ_LINUX_FAKE_EXEC", raising=False)
    monkeypatch.delenv("TLDW_SANDBOX_MACOS_HELPER_READY", raising=False)
    monkeypatch.delenv("TLDW_SANDBOX_VZ_LINUX_TEMPLATE_READY", raising=False)
    monkeypatch.delenv("TLDW_SANDBOX_VZ_LINUX_AVAILABLE", raising=False)

    service = SandboxService()
    stale_session_id = f"vz-linux-real-recovery-stale-{os.getpid()}"
    missing_vm_id = f"vz-linux-real-recovery-missing-vm-{os.getpid()}"
    service._orch.put_vz_session_control(
        session_id=stale_session_id,
        runtime=RuntimeType.vz_linux.value,
        vm_id=missing_vm_id,
        template_id=base_image,
        workspace_mount=None,
        agent_ready=False,
    )

    try:
        diagnostics = service.macos_diagnostics()
        reconciliation_raw = diagnostics.get("reconciliation")
        recovery_summary_raw = diagnostics.get("recovery_summary")
        _expect(isinstance(reconciliation_raw, dict), "Expected reconciliation data in diagnostics")
        _expect(isinstance(recovery_summary_raw, dict), "Expected recovery_summary in diagnostics")
        reconciliation = reconciliation_raw if isinstance(reconciliation_raw, dict) else {}
        recovery_summary = recovery_summary_raw if isinstance(recovery_summary_raw, dict) else {}
        _expect(
            reconciliation.get("computed") is True,
            f"Expected reconciliation to compute, got reasons={reconciliation.get('reasons')!r}",
        )
        _expect(
            stale_session_id in reconciliation.get("stale_session_ids", []),
            f"Expected stale session id in reconciliation, got {reconciliation.get('stale_session_ids')!r}",
        )
        _expect(
            recovery_summary.get("status") == "action_recommended",
            f"Expected recovery action recommendation, got {recovery_summary!r}",
        )
        _expect(
            "vz_stale_session_controls" in recovery_summary.get("codes", []),
            f"Expected stale-session recovery code, got {recovery_summary.get('codes')!r}",
        )

        repair = service.repair_macos_reconciliation(dry_run=True)
        planned_deletes = [
            action
            for action in repair.get("actions", [])
            if action.get("type") == "delete_session_control"
            and action.get("session_id") == stale_session_id
            and action.get("status") == "planned"
        ]
        _expect(repair.get("dry_run") is True, f"Expected dry-run repair, got {repair!r}")
        _expect(planned_deletes, f"Expected planned stale-session delete action, got {repair.get('actions')!r}")
        _expect(
            service._orch.get_vz_session_control(stale_session_id) is not None,
            "Expected dry-run reconciliation repair to leave session control unchanged",
        )
    finally:
        service._orch.delete_vz_session_control(stale_session_id)


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS host only")
@pytest.mark.vz_linux_host_failure_drill
def test_vz_linux_real_session_recreates_vm_after_helper_termination(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify a stale session VM is not reused after helper-side termination."""
    base_image = _require_vz_linux_real_host_e2e(monkeypatch, tmp_path)
    monkeypatch.delenv("TEST_MODE", raising=False)
    monkeypatch.delenv("TLDW_SANDBOX_VZ_LINUX_FAKE_EXEC", raising=False)
    monkeypatch.delenv("TLDW_SANDBOX_MACOS_HELPER_READY", raising=False)
    monkeypatch.delenv("TLDW_SANDBOX_VZ_LINUX_TEMPLATE_READY", raising=False)
    monkeypatch.delenv("TLDW_SANDBOX_VZ_LINUX_AVAILABLE", raising=False)

    service = SandboxService()
    helper = VZLinuxRunner.helper_client_cls()
    session_id: str | None = None
    destroyed = False
    try:
        session = service.create_session(
            user_id="e2e-user",
            spec=SessionSpec(
                runtime=RuntimeType.vz_linux,
                base_image=base_image,
                network_policy="deny_all",
            ),
            spec_version="1.0",
            idem_key=None,
            raw_body={"spec_version": "1.0", "runtime": "vz_linux", "base_image": base_image},
        )
        session_id = session.id

        first = service.start_run_scaffold(
            user_id="e2e-user",
            spec=RunSpec(
                session_id=session.id,
                runtime=RuntimeType.vz_linux,
                base_image=base_image,
                command=["/bin/echo", "failure-drill-first"],
                network_policy="deny_all",
            ),
            spec_version="1.0",
            idem_key=None,
            raw_body={
                "session_id": session.id,
                "runtime": "vz_linux",
                "command": ["/bin/echo", "failure-drill-first"],
            },
        )
        control_after_first = service._orch.get_vz_session_control(session.id)
        _expect(first.phase == RunPhase.completed, f"Expected first run completed, got {first.phase!r}")
        _expect(isinstance(control_after_first, dict), "Expected VZ session control after first run")
        first_vm_id = str(control_after_first.get("vm_id") or "").strip() if control_after_first else ""
        _expect(bool(first_vm_id), f"Expected first run VM id, got {control_after_first!r}")

        status_before_terminate = helper.get_vm_status(first_vm_id)
        _expect(
            bool(getattr(status_before_terminate, "healthy", False)),
            f"Expected drill VM {first_vm_id!r} healthy before termination, got {status_before_terminate!r}",
        )
        terminated = helper.terminate_vm(first_vm_id)
        status_after_terminate = helper.get_vm_status(first_vm_id)
        healthy_after_terminate = bool(getattr(status_after_terminate, "healthy", False))
        if not terminated and healthy_after_terminate:
            pytest.skip(
                f"Could not invalidate drill VM {first_vm_id!r}; helper returned False and VM remained healthy"
            )
        _expect(
            not healthy_after_terminate,
            f"Expected drill VM {first_vm_id!r} unhealthy or missing after termination, got {status_after_terminate!r}",
        )

        second = service.start_run_scaffold(
            user_id="e2e-user",
            spec=RunSpec(
                session_id=session.id,
                runtime=RuntimeType.vz_linux,
                base_image=base_image,
                command=["/bin/echo", "failure-drill-second"],
                network_policy="deny_all",
            ),
            spec_version="1.0",
            idem_key=None,
            raw_body={
                "session_id": session.id,
                "runtime": "vz_linux",
                "command": ["/bin/echo", "failure-drill-second"],
            },
        )
        control_after_second = service._orch.get_vz_session_control(session.id)
        _expect(second.phase == RunPhase.completed, f"Expected second run completed, got {second.phase!r}")
        _expect(isinstance(control_after_second, dict), "Expected VZ session control after second run")
        second_vm_id = str(control_after_second.get("vm_id") or "").strip() if control_after_second else ""
        _expect(bool(second_vm_id), f"Expected second run VM id, got {control_after_second!r}")
        _expect(
            second_vm_id != first_vm_id,
            f"Expected stale VM replacement after helper termination, got {first_vm_id!r}",
        )

        _expect(service.destroy_session(session.id) is True, "Expected session destruction to succeed")
        destroyed = True
        _expect(service._orch.get_vz_session_control(session.id) is None, "Expected VZ session control cleanup")
    finally:
        if session_id and not destroyed:
            service.destroy_session(session_id)


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS host only")
@pytest.mark.vz_linux_host_failure_drill
def test_vz_linux_real_session_recreates_vm_after_helper_restart(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify stale session VM control is replaced after helper process restart."""
    base_image = _require_vz_linux_real_host_e2e(monkeypatch, tmp_path)
    lease = _require_helper_restart_lease()
    monkeypatch.delenv("TEST_MODE", raising=False)
    monkeypatch.delenv("TLDW_SANDBOX_VZ_LINUX_FAKE_EXEC", raising=False)
    monkeypatch.delenv("TLDW_SANDBOX_MACOS_HELPER_READY", raising=False)
    monkeypatch.delenv("TLDW_SANDBOX_VZ_LINUX_TEMPLATE_READY", raising=False)
    monkeypatch.delenv("TLDW_SANDBOX_VZ_LINUX_AVAILABLE", raising=False)

    service = SandboxService()
    helper = VZLinuxRunner.helper_client_cls()
    session_id: str | None = None
    destroyed = False
    try:
        session = service.create_session(
            user_id="e2e-user",
            spec=SessionSpec(
                runtime=RuntimeType.vz_linux,
                base_image=base_image,
                network_policy="deny_all",
            ),
            spec_version="1.0",
            idem_key=None,
            raw_body={"spec_version": "1.0", "runtime": "vz_linux", "base_image": base_image},
        )
        session_id = session.id

        first = service.start_run_scaffold(
            user_id="e2e-user",
            spec=RunSpec(
                session_id=session.id,
                runtime=RuntimeType.vz_linux,
                base_image=base_image,
                command=["/bin/echo", "restart-drill-first"],
                network_policy="deny_all",
            ),
            spec_version="1.0",
            idem_key=None,
            raw_body={
                "session_id": session.id,
                "runtime": "vz_linux",
                "command": ["/bin/echo", "restart-drill-first"],
            },
        )
        control_after_first = service._orch.get_vz_session_control(session.id)
        _expect(first.phase == RunPhase.completed, f"Expected first run completed, got {first.phase!r}")
        _expect(isinstance(control_after_first, dict), "Expected VZ session control after first run")
        first_vm_id = str(control_after_first.get("vm_id") or "").strip() if control_after_first else ""
        _expect(bool(first_vm_id), f"Expected first VM id, got {control_after_first!r}")
        _expect(
            bool(getattr(helper.get_vm_status(first_vm_id), "healthy", False)),
            f"Expected first VM healthy before restart: {first_vm_id!r}",
        )

        _restart_helper_for_drill(lease)
        helper_after_restart = VZLinuxRunner.helper_client_cls()
        status_after_restart = helper_after_restart.get_vm_status(first_vm_id)
        _expect(
            not bool(getattr(status_after_restart, "healthy", False)),
            f"Expected old VM stale after helper restart: {status_after_restart!r}",
        )

        second = service.start_run_scaffold(
            user_id="e2e-user",
            spec=RunSpec(
                session_id=session.id,
                runtime=RuntimeType.vz_linux,
                base_image=base_image,
                command=["/bin/echo", "restart-drill-second"],
                network_policy="deny_all",
            ),
            spec_version="1.0",
            idem_key=None,
            raw_body={
                "session_id": session.id,
                "runtime": "vz_linux",
                "command": ["/bin/echo", "restart-drill-second"],
            },
        )
        control_after_second = service._orch.get_vz_session_control(session.id)
        _expect(second.phase == RunPhase.completed, f"Expected second run completed, got {second.phase!r}")
        _expect(isinstance(control_after_second, dict), "Expected VZ session control after second run")
        second_vm_id = str(control_after_second.get("vm_id") or "").strip() if control_after_second else ""
        _expect(bool(second_vm_id), f"Expected second VM id, got {control_after_second!r}")
        _expect(
            second_vm_id != first_vm_id,
            f"Expected helper restart to force fresh VM, got {first_vm_id!r}",
        )

        _expect(service.destroy_session(session.id) is True, "Expected session destruction to succeed")
        destroyed = True
        _expect(service._orch.get_vz_session_control(session.id) is None, "Expected VZ session control cleanup")
    finally:
        if session_id and not destroyed:
            service.destroy_session(session_id)
