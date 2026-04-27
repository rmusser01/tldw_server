from __future__ import annotations

import tldw_Server_API.app.core.Sandbox.runners.vz_common as vz_common
import tldw_Server_API.app.core.Sandbox.runners.vz_linux_runner as vz_linux_module
from tldw_Server_API.app.core.Sandbox.macos_virtualization.models import (
    HelperExecReply,
    HelperVMReply,
    HelperVMStatusReply,
)
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
    assert calls == [{"runtime": "vz_linux", "network_policy": "deny_all"}]


def test_vz_linux_start_run_executes_real_ephemeral_vm_command(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("TLDW_SANDBOX_VZ_LINUX_FAKE_EXEC", raising=False)
    calls: list[tuple[str, dict[str, object]]] = []

    class _FakeHelper:
        def validate_template(self, request: dict[str, object]) -> dict[str, object]:
            calls.append(("validate_template", dict(request)))
            return {
                "template_id": "vz_linux:validated-ubuntu",
                "source": "ubuntu-24.04",
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
            startup_timeout_sec=23,
        ),
        session_workspace=str(tmp_path),
    )

    assert status.phase == RunPhase.completed
    assert status.exit_code == 0
    assert [name for name, _payload in calls] == ["validate_template", "create_vm", "exec_guest", "terminate_vm"]
    assert calls[0][1]["template"] == "ubuntu-24.04"
    assert calls[1][1]["workspace_path"] == str(tmp_path)
    assert calls[1][1]["workspace_mount"] == "virtiofs"
    assert calls[1][1]["template"] == "ubuntu-24.04"
    assert calls[1][1]["timeout_sec"] == 23
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


def test_vz_linux_session_run_reuses_only_healthy_vm(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("TLDW_SANDBOX_VZ_LINUX_FAKE_EXEC", raising=False)
    calls: list[str] = []

    class _Store:
        def get_vz_session_control(self, session_id: str) -> dict[str, object]:
            assert session_id == "sess-1"
            return {
                "runtime": "vz_linux",
                "vm_id": "vm-existing",
                "template_id": "vz_linux:existing",
                "workspace_mount": str(tmp_path),
                "agent_ready": True,
            }

        def delete_vz_session_control(self, session_id: str) -> bool:
            raise AssertionError(f"delete_vz_session_control should not be called for healthy vm: {session_id}")

        def put_vz_session_control(self, **kwargs) -> None:
            raise AssertionError(f"put_vz_session_control should not be called for healthy vm: {kwargs}")

    class _FakeHelper:
        def get_vm_status(self, vm_id: str) -> HelperVMStatusReply:
            calls.append("get_vm_status")
            return HelperVMStatusReply(
                protocol_version="1",
                helper_version="0.1.0",
                vm_id=vm_id,
                state="running",
                healthy=True,
            )

        def validate_template(self, request: dict[str, object]) -> dict[str, object]:
            raise AssertionError(f"validate_template should not run for healthy vm reuse: {request}")

        def create_vm(self, request: dict[str, object]) -> HelperVMReply:
            raise AssertionError(f"create_vm should not run for healthy vm reuse: {request}")

        def exec_guest(self, *, vm_id: str, request: dict[str, object]) -> HelperExecReply:
            calls.append("exec_guest")
            return HelperExecReply(exit_code=0, stdout=b"reuse-ok\n")

    monkeypatch.setattr(vz_linux_module.VZLinuxRunner, "helper_client_cls", _FakeHelper)

    status = VZLinuxRunner(session_control_store=_Store()).start_run(
        run_id="vz-run-reuse",
        spec=RunSpec(
            session_id="sess-1",
            runtime=RuntimeType.vz_linux,
            base_image="ubuntu-24.04",
            command=["/bin/echo", "ok"],
            network_policy="deny_all",
        ),
        session_workspace=str(tmp_path),
    )

    assert status.phase == RunPhase.completed
    assert status.exit_code == 0
    assert calls == ["get_vm_status", "exec_guest"]


def test_vz_linux_session_run_recreates_unhealthy_vm(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("TLDW_SANDBOX_VZ_LINUX_FAKE_EXEC", raising=False)
    calls: list[str] = []
    deleted: list[str] = []
    stored: list[dict[str, object]] = []

    class _Store:
        def get_vz_session_control(self, session_id: str) -> dict[str, object]:
            assert session_id == "sess-2"
            return {
                "runtime": "vz_linux",
                "vm_id": "vm-stale",
                "template_id": "vz_linux:stale",
                "workspace_mount": str(tmp_path),
                "agent_ready": True,
            }

        def delete_vz_session_control(self, session_id: str) -> bool:
            deleted.append(session_id)
            return True

        def put_vz_session_control(self, **kwargs) -> None:
            stored.append(dict(kwargs))

    class _FakeHelper:
        def get_vm_status(self, vm_id: str) -> HelperVMStatusReply:
            calls.append("get_vm_status")
            return HelperVMStatusReply(
                protocol_version="1",
                helper_version="0.1.0",
                vm_id=vm_id,
                state="missing",
                healthy=False,
            )

        def validate_template(self, request: dict[str, object]) -> dict[str, object]:
            calls.append("validate_template")
            return {
                "template_id": "vz_linux:new-template",
                "ready": True,
                "reasons": [],
            }

        def create_vm(self, request: dict[str, object]) -> HelperVMReply:
            calls.append("create_vm")
            return HelperVMReply(vm_id="vm-new", state="created")

        def exec_guest(self, *, vm_id: str, request: dict[str, object]) -> HelperExecReply:
            calls.append("exec_guest")
            return HelperExecReply(exit_code=0, stdout=b"recreated\n")

    monkeypatch.setattr(vz_linux_module.VZLinuxRunner, "helper_client_cls", _FakeHelper)

    status = VZLinuxRunner(session_control_store=_Store()).start_run(
        run_id="vz-run-recreate",
        spec=RunSpec(
            session_id="sess-2",
            runtime=RuntimeType.vz_linux,
            base_image="ubuntu-24.04",
            command=["/bin/echo", "ok"],
            network_policy="deny_all",
        ),
        session_workspace=str(tmp_path),
    )

    assert status.phase == RunPhase.completed
    assert status.exit_code == 0
    assert calls == ["get_vm_status", "validate_template", "create_vm", "exec_guest"]
    assert deleted == ["sess-2"]
    assert stored and stored[0]["vm_id"] == "vm-new"
    assert stored[0]["template_id"] == "vz_linux:new-template"
