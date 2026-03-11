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
    monkeypatch.delenv("TEST_MODE", raising=False)
    monkeypatch.delenv("TLDW_SANDBOX_VZ_LINUX_FAKE_EXEC", raising=False)

    calls: list[dict[str, object]] = []

    class _FakeHelper:
        def validate_vz_linux_host(self, request: dict[str, object]) -> dict[str, object]:
            calls.append(dict(request))
            return {
                "protocol_version": "1",
                "helper_version": "0.1.0",
                "available": True,
                "reasons": [],
                "execution_mode": "real",
                "transport": "vsock",
                "details": {"runtime": "vz_linux"},
            }

    monkeypatch.setattr(vz_linux_module.VZLinuxRunner, "helper_client_cls", _FakeHelper)

    result = VZLinuxRunner().preflight(network_policy="deny_all")

    assert result.available is True
    assert result.reasons == []
    assert result.execution_mode == "real"
    assert result.enforcement_ready == {"deny_all": True, "allowlist": False}
    assert calls == [{"network_policy": "deny_all"}]


def test_vz_linux_start_run_executes_real_ephemeral_vm_command(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("TLDW_SANDBOX_VZ_LINUX_FAKE_EXEC", raising=False)
    calls: list[tuple[str, dict[str, object]]] = []

    class _FakeHelper:
        def validate_template(self, request: dict[str, object]) -> dict[str, object]:
            calls.append(("validate_template", dict(request)))
            return {
                "template_id": "vz_linux:validated-ubuntu",
                "ready": True,
                "reasons": [],
            }

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
    assert [name for name, _payload in calls] == ["validate_template", "create_vm", "exec_guest", "terminate_vm"]
    assert calls[0][1]["template"] == "ubuntu-24.04"
    assert calls[1][1]["workspace_path"] == str(tmp_path)
    assert calls[1][1]["workspace_mount"] == "virtiofs"
    assert calls[1][1]["template"] == "vz_linux:validated-ubuntu"
    assert calls[2][1]["vm_id"] == "vm-ephemeral-1"
    assert calls[2][1]["argv"] == ["/bin/echo", "ok"]
    assert calls[2][1]["cwd"] == "/workspace"
    frames = list(hub._buffers.get(run_id, []))  # type: ignore[attr-defined]
    assert any(frame.get("type") == "stdout" and "ok" in frame.get("data", "") for frame in frames)
    assert any(frame.get("type") == "stderr" and "warn" in frame.get("data", "") for frame in frames)


def test_vz_linux_start_run_fails_closed_when_template_validation_fails(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("TLDW_SANDBOX_VZ_LINUX_FAKE_EXEC", raising=False)
    calls: list[str] = []

    class _FakeHelper:
        def validate_template(self, request: dict[str, object]) -> dict[str, object]:
            calls.append("validate_template")
            return {
                "template_id": None,
                "ready": False,
                "reasons": ["template_invalid"],
            }

        def create_vm(self, request: dict[str, object]) -> HelperVMReply:
            calls.append("create_vm")
            raise AssertionError("create_vm should not run when template validation fails")

    monkeypatch.setattr(vz_linux_module.VZLinuxRunner, "helper_client_cls", _FakeHelper)

    status = VZLinuxRunner().start_run(
        run_id="vz-run-invalid-template",
        spec=RunSpec(
            session_id=None,
            runtime=RuntimeType.vz_linux,
            base_image="missing-template",
            command=["/bin/echo", "ok"],
            network_policy="deny_all",
        ),
        session_workspace=str(tmp_path),
    )

    assert status.phase == RunPhase.failed
    assert "template_invalid" in status.message
    assert calls == ["validate_template"]
