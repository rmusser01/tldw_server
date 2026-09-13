"""Manual proof of live-session recovery through a drill-owned LaunchAgent."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import os
import platform
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from tldw_Server_API.app.core.Sandbox.models import RunPhase, RunSpec, RuntimeType, SessionSpec
from tldw_Server_API.app.core.Sandbox.runners.vz_linux_runner import VZLinuxRunner
from tldw_Server_API.app.core.Sandbox.service import SandboxService
from tldw_Server_API.app.core.Sandbox.streams import get_hub
from tldw_Server_API.app.core.testing import is_truthy
from tldw_Server_API.tests.sandbox.test_vz_linux_real_host_e2e import (
    _expect,
    _require_vz_linux_real_host_e2e,
    _wait_for_helper_socket_unavailable,
)


@pytest.mark.vz_linux_host_failure_drill
@pytest.mark.skipif(sys.platform != "darwin", reason="macOS host only")
def test_vz_linux_session_recovers_after_launchd_restart(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Reject stale VM reuse after launchd replaces a helper with a live session."""
    if not is_truthy(os.getenv("TLDW_SANDBOX_VZ_LINUX_LAUNCHD_RESTART_DRILL")):
        pytest.skip("Set TLDW_SANDBOX_VZ_LINUX_LAUNCHD_RESTART_DRILL=1 for this manual drill")
    if platform.machine() != "arm64":
        pytest.skip("Apple silicon host only")
    _expect(is_truthy(os.getenv("TLDW_SANDBOX_VZ_LINUX_E2E")), "Real E2E opt-in is required")
    _expect(bool(os.getenv("TLDW_SANDBOX_VZ_LINUX_E2E_BASE_IMAGE")), "A disposable bundle is required")
    helper_text = os.getenv("TLDW_SANDBOX_MACOS_HELPER_BINARY", "").strip()
    _expect(bool(helper_text), "A built, signed helper binary is required")
    helper_path = Path(helper_text).expanduser().resolve(strict=True)

    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "tools/macos-vz-helper/scripts/vz-helperctl.py"
    spec = importlib.util.spec_from_file_location("vz_helperctl_launchd_recovery", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {script}")
    helperctl = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, helperctl)
    spec.loader.exec_module(helperctl)

    # Short AF_UNIX parent; mkdtemp atomically creates a random 0700 directory.
    runtime_parent = "/tmp"  # nosec B108
    runtime_dir = Path(tempfile.mkdtemp(prefix="tvz-lr.", dir=runtime_parent)).resolve()
    socket_path = runtime_dir / "helper.sock"
    launchd_options = {
        "helper_path": helper_path,
        "socket_path": socket_path,
        "log_dir": tmp_path / "helper-logs",
        "plist_path": runtime_dir / "launchd.plist",
        "label": f"org.tldw.macos-vz-helper.drill.recovery.{uuid4().hex}",
    }
    monkeypatch.setenv("TLDW_SANDBOX_MACOS_HELPER_SOCKET", str(socket_path))
    for name in (
        "TEST_MODE",
        "TLDW_SANDBOX_VZ_LINUX_FAKE_EXEC",
        "TLDW_SANDBOX_MACOS_HELPER_READY",
        "TLDW_SANDBOX_VZ_LINUX_TEMPLATE_READY",
        "TLDW_SANDBOX_VZ_LINUX_AVAILABLE",
    ):
        monkeypatch.delenv(name, raising=False)
    observations: dict[str, Any] = {"label": launchd_options["label"], "runs": []}

    def exercise_session() -> Any:
        """Execute before/after restart through the unchanged SandboxService."""
        base_image = _require_vz_linux_real_host_e2e(monkeypatch, tmp_path)
        service = SandboxService()
        helper = VZLinuxRunner.helper_client_cls()
        original_exec = VZLinuxRunner.helper_client_cls.exec_guest

        def record_exec(client: Any, **kwargs: Any) -> Any:
            reply = original_exec(client, **kwargs)
            observations.setdefault("guest_replies", []).append(
                {
                    "stdout": repr(reply.stdout),
                    "stderr": repr(reply.stderr),
                    "details": reply.details,
                }
            )
            return reply

        monkeypatch.setattr(VZLinuxRunner.helper_client_cls, "exec_guest", record_exec)
        session = service.create_session(
            user_id="e2e-user",
            spec=SessionSpec(runtime=RuntimeType.vz_linux, base_image=base_image, network_policy="deny_all"),
            spec_version="1.0",
            idem_key=None,
            raw_body={"runtime": "vz_linux", "base_image": base_image},
        )

        def run_command(token: str) -> dict[str, Any]:
            """Check real stdout and exit status, then record the current VM control."""
            command = ["/bin/echo", token]
            result = service.start_run_scaffold(
                user_id="e2e-user",
                spec=RunSpec(
                    session_id=session.id,
                    runtime=RuntimeType.vz_linux,
                    base_image=base_image,
                    command=command,
                    network_policy="deny_all",
                ),
                spec_version="1.0",
                idem_key=None,
                raw_body={"session_id": session.id, "runtime": "vz_linux", "command": command},
            )
            frames = get_hub().get_buffer_snapshot(result.id)
            stdout = "".join(str(frame.get("data", "")) for frame in frames if frame.get("type") == "stdout")
            control = service._orch.get_vz_session_control(session.id)
            observations["runs"].append(
                {
                    "token": token,
                    "phase": result.phase.value,
                    "exit_code": result.exit_code,
                    "stdout": stdout,
                    "frames": frames,
                    "control": dict(control) if control else None,
                }
            )
            _expect(result.phase == RunPhase.completed, f"Run failed: {result}")
            _expect(result.exit_code == 0, f"Unexpected exit status: {result.exit_code}")
            _expect(stdout.strip() == token, f"Missing guest output: {stdout!r}")
            _expect(isinstance(control, dict) and bool(control.get("vm_id")), "Missing session VM control")
            return dict(control)

        try:
            before = run_command("launchd-before-restart")
            old_instance = str(helper.ping().details.get("helper_instance_id") or "")
            _expect(bool(old_instance), "Helper generation is required before restart")
            _expect(helper.get_vm_status(before["vm_id"]).healthy, "First VM must be live before restart")
            observations["helper_instance_before"] = old_instance
            restart = helperctl.run_launchd_action("kickstart", **launchd_options)
            observations["live_session_restart"] = dataclasses.asdict(restart)
            _expect(restart.ok, f"LaunchAgent restart failed: {restart}")

            def replacement_ping(path: Path) -> Any:
                """Do not mistake a still-answering old helper for its replacement."""
                state = helperctl.ping_helper_state(path)
                instance = (state.details or {}).get("helper_instance_id")
                if state.result.ok and (not instance or instance == old_instance):
                    return helperctl.PingState(helperctl.CheckResult(False, "helper_generation_unchanged"))
                return state

            replacement = helperctl.wait_for_ping(socket_path, ping_checker=replacement_ping)
            _expect(replacement.result.ok, f"Replacement helper unavailable: {replacement.result}")
            new_instance = replacement.details["helper_instance_id"]
            observations["helper_instance_after"] = new_instance
            _expect(not helper.get_vm_status(before["vm_id"]).healthy, "Old VM survived helper replacement")
            _expect(
                service._orch.get_vz_session_control(session.id) == before, "Restart bypassed stale control recovery"
            )

            after = run_command("launchd-after-restart")
            _expect(after["vm_id"] != before["vm_id"], "Stale VM was reused after helper restart")
            _expect(after.get("helper_instance_id") == new_instance, "Replacement generation was not persisted")
            reused = run_command("launchd-replacement-reuse")
            _expect(reused["vm_id"] == after["vm_id"], "Healthy replacement VM was not reused")
        finally:
            destroyed = service.destroy_session(session.id)
            observations["session_destroyed"] = destroyed
            _expect(destroyed, "Session destruction failed")
            _expect(service._orch.get_vz_session_control(session.id) is None, "Session control remained")
            remaining = helper.list_vms().vms
            observations["remaining_vm_ids"] = [vm.vm_id for vm in remaining]
            _expect(not remaining, "Drill left VMs in its helper registry")
        return helperctl.CheckResult(True)

    failure: BaseException | None = None

    def record_session_failure() -> Any:
        nonlocal failure
        try:
            return exercise_session()
        except (Exception, pytest.fail.Exception, pytest.skip.Exception) as exc:
            # Let the lifecycle driver return its bootout evidence before reraising.
            failure = exc
            observations["session_failure"] = str(exc)
            return helperctl.CheckResult(False, "session_drill_failed", str(exc))

    results = []
    try:
        results = helperctl.run_launchd_drill(
            **launchd_options,
            write_plist=True,
            create_dirs=True,
            smoke_runner=record_session_failure,
        )
        if failure is not None:
            raise failure
        _expect(all(result.ok for _, result in results), f"Launchd drill failed: {results}")
    finally:
        postflight = helperctl.launchd_service_loaded(launchd_options["label"])
        observations["launchd_steps"] = [{"name": name, **dataclasses.asdict(result)} for name, result in results]
        observations["launchd_after"] = dataclasses.asdict(postflight)
        observations["runtime_dir"] = str(runtime_dir)
        try:
            if launchd_options["plist_path"].is_file():
                shutil.copyfile(launchd_options["plist_path"], tmp_path / "launchd.plist")
            absent = postflight.ok and postflight.reason == "launchd_service_absent"
            _expect(absent, f"LaunchAgent cleanup unconfirmed; retain {runtime_dir}: {postflight}")
            _wait_for_helper_socket_unavailable(socket_path)
            observations["socket_unavailable"] = True
            shutil.rmtree(runtime_dir)
            observations["runtime_removed"] = not runtime_dir.exists()
        finally:
            (tmp_path / "launchd-recovery.json").write_text(json.dumps(observations, indent=2) + "\n")
