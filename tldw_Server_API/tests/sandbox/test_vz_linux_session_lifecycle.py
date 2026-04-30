from __future__ import annotations

from tldw_Server_API.app.core.Sandbox.macos_virtualization.models import (
    HelperExecReply,
    HelperVMReply,
    HelperVMStatusReply,
)
from tldw_Server_API.app.core.Sandbox.models import RunPhase, RunSpec, RuntimeType
import tldw_Server_API.app.core.Sandbox.runners.vz_linux_runner as vz_linux_module
from tldw_Server_API.app.core.Sandbox.runners.vz_linux_runner import VZLinuxRunner


class _FakeSessionControlStore:
    def __init__(self) -> None:
        self._rows: dict[str, dict[str, object]] = {}

    def put_vz_session_control(
        self,
        *,
        session_id: str,
        runtime: str,
        vm_id: str,
        template_id: str | None,
        workspace_mount: str | None,
        agent_ready: bool,
    ) -> None:
        self._rows[str(session_id)] = {
            "id": str(session_id),
            "runtime": str(runtime),
            "vm_id": str(vm_id),
            "template_id": template_id,
            "workspace_mount": workspace_mount,
            "agent_ready": bool(agent_ready),
        }

    def get_vz_session_control(self, session_id: str) -> dict[str, object] | None:
        row = self._rows.get(str(session_id))
        return dict(row) if row else None

    def delete_vz_session_control(self, session_id: str) -> bool:
        return self._rows.pop(str(session_id), None) is not None


def test_vz_linux_session_reuses_existing_vm_for_second_run(monkeypatch, tmp_path) -> None:
    calls: list[str] = []

    class _FakeHelper:
        def validate_template(self, request: dict[str, object]) -> dict[str, object]:
            calls.append("validate_template")
            return {
                "ready": True,
                "template_id": str(request["template"]),
                "source": str(request["template"]),
            }

        def create_vm(self, request: dict[str, object]) -> HelperVMReply:
            assert request["owner"] == "tldw"
            assert request["runtime"] == "vz_linux"
            assert request["run_id"] == "run-1"
            assert request["session_id"] == "sess-vz-1"
            assert request["session_mode"] is True
            assert request["template"] == "ubuntu-24.04"
            calls.append("create_vm")
            return HelperVMReply(vm_id="vm-session-1", state="created", details={"session_mode": True})

        def get_vm_status(self, vm_id: str) -> HelperVMStatusReply:
            calls.append(f"get_vm_status:{vm_id}")
            return HelperVMStatusReply(
                protocol_version="1",
                helper_version="test",
                vm_id=vm_id,
                state="running",
                healthy=True,
            )

        def exec_guest(self, *, vm_id: str, request: dict[str, object]) -> HelperExecReply:
            del request
            calls.append(f"exec_guest:{vm_id}")
            return HelperExecReply(exit_code=0, stdout=b"ok\n")

        def terminate_vm(self, vm_id: str) -> bool:
            calls.append(f"terminate_vm:{vm_id}")
            return True

    monkeypatch.setattr(vz_linux_module.VZLinuxRunner, "helper_client_cls", _FakeHelper)

    store = _FakeSessionControlStore()
    runner = VZLinuxRunner(session_control_store=store)
    first_spec = RunSpec(
        session_id="sess-vz-1",
        runtime=RuntimeType.vz_linux,
        base_image="ubuntu-24.04",
        command=["/bin/echo", "ok"],
        network_policy="deny_all",
    )
    second_spec = RunSpec(
        session_id="sess-vz-1",
        runtime=RuntimeType.vz_linux,
        base_image="ubuntu-24.04",
        command=["/bin/echo", "ok"],
        network_policy="deny_all",
    )

    first = runner.start_run("run-1", first_spec, str(tmp_path))
    second = runner.start_run("run-2", second_spec, str(tmp_path))

    assert first.phase == RunPhase.completed
    assert second.phase == RunPhase.completed
    assert calls == [
        "validate_template",
        "create_vm",
        "exec_guest:vm-session-1",
        "get_vm_status:vm-session-1",
        "exec_guest:vm-session-1",
    ]
    control = store.get_vz_session_control("sess-vz-1")
    assert control is not None
    assert control["vm_id"] == "vm-session-1"
    assert control["agent_ready"] is True
