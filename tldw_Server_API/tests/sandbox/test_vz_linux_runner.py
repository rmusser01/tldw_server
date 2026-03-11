from __future__ import annotations

import tldw_Server_API.app.core.Sandbox.runners.vz_common as vz_common
import tldw_Server_API.app.core.Sandbox.runners.vz_linux_runner as vz_linux_module
from tldw_Server_API.app.core.Sandbox.macos_virtualization.models import HelperExecReply, HelperVMReply
from tldw_Server_API.app.core.Sandbox.models import RunPhase, RunSpec, RuntimeType
from tldw_Server_API.app.core.Sandbox.streams import get_hub
from tldw_Server_API.app.core.Sandbox.runners.vz_linux_runner import VZLinuxRunner


def test_vz_linux_fake_run_completes(monkeypatch) -> None:
    monkeypatch.setenv("TLDW_SANDBOX_VZ_LINUX_FAKE_EXEC", "1")

    runner = VZLinuxRunner()
    status = runner.start_run(
        run_id="run-123",
        spec=RunSpec(
            session_id=None,
            runtime=RuntimeType.vz_linux,
            base_image="ubuntu-24.04",
            command=["echo", "ok"],
            network_policy="deny_all",
        ),
    )

    assert status.phase == RunPhase.completed
    assert status.exit_code == 0


def test_vz_linux_preflight_requires_execution_readiness(monkeypatch) -> None:
    monkeypatch.setattr(vz_common.sys, "platform", "darwin")
    monkeypatch.setattr(vz_common.platform, "machine", lambda: "arm64")
    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("TLDW_SANDBOX_VZ_LINUX_AVAILABLE", "1")
    monkeypatch.setenv("TLDW_SANDBOX_MACOS_HELPER_READY", "1")
    monkeypatch.setenv("TLDW_SANDBOX_VZ_LINUX_TEMPLATE_READY", "1")
    monkeypatch.delenv("TLDW_SANDBOX_VZ_LINUX_FAKE_EXEC", raising=False)
    monkeypatch.delenv("TLDW_SANDBOX_VZ_LINUX_EXEC_READY", raising=False)

    result = VZLinuxRunner().preflight(network_policy="deny_all")

    assert result.available is True
    assert result.reasons == []
    assert result.execution_mode == "real"
    assert result.enforcement_ready == {"deny_all": True, "allowlist": False}


def test_vz_linux_start_run_executes_real_ephemeral_vm_command(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("TLDW_SANDBOX_VZ_LINUX_FAKE_EXEC", raising=False)
    calls: list[tuple[str, dict[str, object]]] = []

    class _FakeHelper:
        def create_vm(self, request: dict[str, object]) -> HelperVMReply:
            calls.append(("create_vm", dict(request)))
            return HelperVMReply(vm_id="vm-ephemeral-1", state="created", details={"transport": "vsock"})

        def exec_guest(self, *, vm_id: str, request: dict[str, object]) -> HelperExecReply:
            calls.append(("exec_guest", {"vm_id": vm_id, **request}))
            return HelperExecReply(exit_code=0, stdout=b"ok\n", stderr=b"warn\n")

        def terminate_vm(self, vm_id: str) -> bool:
            calls.append(("terminate_vm", {"vm_id": vm_id}))
            return True

    monkeypatch.setattr(vz_linux_module.VZLinuxRunner, "helper_client_cls", _FakeHelper)

    run_id = "vz-run-1"
    hub = get_hub()
    hub._buffers.pop(run_id, None)  # type: ignore[attr-defined]

    status = VZLinuxRunner().start_run(
        run_id=run_id,
        spec=RunSpec(
            session_id=None,
            runtime=RuntimeType.vz_linux,
            base_image="ubuntu-24.04",
            command=["/bin/echo", "ok"],
            network_policy="deny_all",
            env={"DEMO": "1"},
        ),
        session_workspace=str(tmp_path),
    )

    assert status.phase == RunPhase.completed
    assert status.exit_code == 0
    assert [name for name, _payload in calls] == ["create_vm", "exec_guest", "terminate_vm"]
    assert calls[0][1]["workspace_path"] == str(tmp_path)
    assert calls[0][1]["workspace_mount"] == "virtiofs"
    assert calls[1][1]["vm_id"] == "vm-ephemeral-1"
    assert calls[1][1]["argv"] == ["/bin/echo", "ok"]
    assert calls[1][1]["cwd"] == "/workspace"
    frames = list(hub._buffers.get(run_id, []))  # type: ignore[attr-defined]
    assert any(frame.get("type") == "stdout" and "ok" in frame.get("data", "") for frame in frames)
    assert any(frame.get("type") == "stderr" and "warn" in frame.get("data", "") for frame in frames)
