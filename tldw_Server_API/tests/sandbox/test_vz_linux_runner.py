from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

import tldw_Server_API.app.core.Sandbox.runners.vz_common as vz_common
import tldw_Server_API.app.core.Sandbox.runners.vz_linux_runner as vz_linux_module
from tldw_Server_API.app.core.Sandbox.image_store import SandboxImageStore
from tldw_Server_API.app.core.Sandbox.macos_virtualization.helper_client import (
    MacOSVirtualizationHelperFailure,
    MacOSVirtualizationHelperProtocolError,
    MacOSVirtualizationHelperUnavailable,
)
from tldw_Server_API.app.core.Sandbox.macos_virtualization.models import (
    HelperExecReply,
    HelperVMMetadata,
    HelperVMReply,
    HelperVMStatusReply,
)
from tldw_Server_API.app.core.Sandbox.models import RunPhase, RunSpec, RuntimeType
from tldw_Server_API.app.core.Sandbox.policy import SandboxPolicy, SandboxPolicyConfig
from tldw_Server_API.app.core.Sandbox.runners.vz_linux_runner import VZLinuxRunner
from tldw_Server_API.app.core.Sandbox.service import SandboxService
from tldw_Server_API.app.core.Sandbox.streams import get_hub


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


def test_vz_linux_preflight_classifies_protocol_mismatch(monkeypatch) -> None:
    monkeypatch.setattr(vz_common.sys, "platform", "darwin")
    monkeypatch.setattr(vz_common.platform, "machine", lambda: "arm64")
    monkeypatch.delenv("TEST_MODE", raising=False)

    class _FakeHelper:
        def validate_vz_linux_host(self, request: dict[str, object]) -> dict[str, object]:
            raise MacOSVirtualizationHelperProtocolError("macos_virtualization_helper_protocol_mismatch")

    monkeypatch.setattr(vz_linux_module.VZLinuxRunner, "helper_client_cls", _FakeHelper)

    result = VZLinuxRunner().preflight(network_policy="deny_all")

    assert result.available is False
    assert result.reasons == ["macos_virtualization_helper_protocol_mismatch"]
    assert result.execution_mode == "none"
    assert result.enforcement_ready == {"deny_all": False, "allowlist": False}


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
    assert calls[1][1]["owner"] == "tldw"
    assert calls[1][1]["runtime"] == "vz_linux"
    assert calls[1][1]["vm_name"] == run_id
    assert calls[1][1]["run_id"] == run_id
    assert calls[1][1]["session_id"] == ""
    assert calls[1][1]["session_mode"] is False
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


def test_vz_linux_start_run_uses_image_store_template_id_when_configured(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("TLDW_SANDBOX_VZ_LINUX_FAKE_EXEC", raising=False)
    store_root = tmp_path / "image-store"
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "kernel").write_bytes(b"kernel")
    (bundle / "rootfs.img").write_bytes(b"rootfs")
    (bundle / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "boot_mode": "linux_direct"}),
        encoding="utf-8",
    )
    template_id = SandboxImageStore(root_path=store_root).register_bundle(
        runtime="vz_linux",
        template_name="debian-bookworm-arm64",
        bundle_path=bundle,
    )
    monkeypatch.setenv("TLDW_SANDBOX_IMAGE_STORE_ROOT", str(store_root))
    calls: list[tuple[str, dict[str, object]]] = []

    class _FakeHelper:
        def validate_template(self, request: dict[str, object]) -> dict[str, object]:
            calls.append(("validate_template", dict(request)))
            return {
                "template_id": template_id,
                "source": str(bundle),
                "ready": True,
                "reasons": [],
            }

        def create_vm(self, request: dict[str, object]) -> HelperVMReply:
            calls.append(("create_vm", dict(request)))
            return HelperVMReply(vm_id="vm-ephemeral-1", state="created", details={"transport": "vsock"})

        def exec_guest(self, *, vm_id: str, request: dict[str, object]) -> HelperExecReply:
            calls.append(("exec_guest", {"vm_id": vm_id, **request}))
            return HelperExecReply(exit_code=0, stdout=b"ok\n", stderr=b"")

        def terminate_vm(self, vm_id: str) -> bool:
            calls.append(("terminate_vm", {"vm_id": vm_id}))
            return True

    monkeypatch.setattr(vz_linux_module.VZLinuxRunner, "helper_client_cls", _FakeHelper)

    status = VZLinuxRunner().start_run(
        run_id="vz-run-with-store",
        spec=RunSpec(
            session_id=None,
            runtime=RuntimeType.vz_linux,
            base_image=template_id,
            command=["/bin/echo", "ok"],
            network_policy="deny_all",
        ),
        session_workspace=str(tmp_path / "workspace"),
    )

    assert status.phase == RunPhase.completed
    assert calls[0][1]["template"] == str(bundle)
    assert calls[1][1]["template"] == str(bundle)
    assert calls[1][1]["template_id"] == template_id
    assert calls[1][1]["planning_source"] == "image_store"
    assert calls[1][1]["run_manifest_path"] == str(
        store_root / "runs" / "vz-run-with-store" / "manifest.json"
    )
    persisted_manifest = SandboxImageStore(root_path=store_root).get_run_clone_manifest(
        "vz-run-with-store"
    )
    assert persisted_manifest is not None
    assert persisted_manifest.template_id == template_id


def test_vz_linux_image_store_does_not_persist_manifest_for_healthy_session_reuse(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.delenv("TLDW_SANDBOX_VZ_LINUX_FAKE_EXEC", raising=False)
    store_root = tmp_path / "image-store"
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "kernel").write_bytes(b"kernel")
    (bundle / "rootfs.img").write_bytes(b"rootfs")
    template_id = SandboxImageStore(root_path=store_root).register_bundle(
        runtime="vz_linux",
        template_name="debian-bookworm-arm64",
        bundle_path=bundle,
    )
    monkeypatch.setenv("TLDW_SANDBOX_IMAGE_STORE_ROOT", str(store_root))

    class _Store:
        def get_vz_session_control(self, session_id: str) -> dict[str, object]:
            assert session_id == "sess-store-reuse"
            return {
                "runtime": "vz_linux",
                "vm_id": "vm-existing",
                "template_id": template_id,
                "workspace_mount": str(tmp_path),
                "agent_ready": True,
                "helper_instance_id": "helper-a",
                "helper_started_at": "2026-05-09T00:00:00Z",
            }

    class _FakeHelper:
        def get_vm_status(self, vm_id: str) -> HelperVMStatusReply:
            return HelperVMStatusReply(
                protocol_version="1",
                helper_version="0.1.0",
                vm_id=vm_id,
                state="running",
                healthy=True,
                metadata=HelperVMMetadata(
                    owner="tldw",
                    runtime="vz_linux",
                    session_id="sess-store-reuse",
                    session_mode=True,
                ),
                details={
                    "helper_instance_id": "helper-a",
                    "helper_started_at": "2026-05-09T00:00:00Z",
                },
            )

        def validate_template(self, request: dict[str, object]) -> dict[str, object]:
            raise AssertionError(f"validate_template should not run for healthy vm reuse: {request}")

        def create_vm(self, request: dict[str, object]) -> HelperVMReply:
            raise AssertionError(f"create_vm should not run for healthy vm reuse: {request}")

        def exec_guest(self, *, vm_id: str, request: dict[str, object]) -> HelperExecReply:
            return HelperExecReply(exit_code=0, stdout=b"reuse-ok\n")

    monkeypatch.setattr(vz_linux_module.VZLinuxRunner, "helper_client_cls", _FakeHelper)

    status = VZLinuxRunner(session_control_store=_Store()).start_run(
        run_id="vz-run-reused-store-template",
        spec=RunSpec(
            session_id="sess-store-reuse",
            runtime=RuntimeType.vz_linux,
            base_image=template_id,
            command=["/bin/echo", "ok"],
            network_policy="deny_all",
        ),
        session_workspace=str(tmp_path),
    )

    assert status.phase == RunPhase.completed
    assert (
        SandboxImageStore(root_path=store_root).get_run_clone_manifest("vz-run-reused-store-template")
        is None
    )


def test_vz_linux_image_store_does_not_persist_manifest_when_create_vm_fails(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.delenv("TLDW_SANDBOX_VZ_LINUX_FAKE_EXEC", raising=False)
    store_root = tmp_path / "image-store"
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "kernel").write_bytes(b"kernel")
    (bundle / "rootfs.img").write_bytes(b"rootfs")
    template_id = SandboxImageStore(root_path=store_root).register_bundle(
        runtime="vz_linux",
        template_name="debian-bookworm-arm64",
        bundle_path=bundle,
    )
    monkeypatch.setenv("TLDW_SANDBOX_IMAGE_STORE_ROOT", str(store_root))

    class _FakeHelper:
        def validate_template(self, request: dict[str, object]) -> dict[str, object]:
            return {
                "template_id": template_id,
                "source": str(bundle),
                "ready": True,
                "reasons": [],
            }

        def create_vm(self, request: dict[str, object]) -> HelperVMReply:
            raise RuntimeError("create_vm_failed")

    monkeypatch.setattr(vz_linux_module.VZLinuxRunner, "helper_client_cls", _FakeHelper)

    status = VZLinuxRunner().start_run(
        run_id="vz-run-create-fails",
        spec=RunSpec(
            session_id=None,
            runtime=RuntimeType.vz_linux,
            base_image=template_id,
            command=["/bin/echo", "ok"],
            network_policy="deny_all",
        ),
        session_workspace=str(tmp_path / "workspace"),
    )

    assert status.phase == RunPhase.failed
    assert "create_vm_failed" in status.message
    assert SandboxImageStore(root_path=store_root).get_run_clone_manifest("vz-run-create-fails") is None


def test_vz_linux_session_create_vm_readiness_failure_does_not_persist_reuse_state(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.delenv("TLDW_SANDBOX_VZ_LINUX_FAKE_EXEC", raising=False)
    calls: list[str] = []
    deleted: list[str] = []
    stored: list[dict[str, object]] = []
    terminated: list[str] = []

    class _Store:
        def get_vz_session_control(self, session_id: str) -> dict[str, object]:
            assert session_id == "sess-readiness-timeout"
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

        def put_vz_session_control(self, **kwargs: object) -> None:
            stored.append(dict(kwargs))

    class _FakeHelper:
        def get_vm_status(self, vm_id: str) -> HelperVMStatusReply:
            assert vm_id == "vm-stale"
            calls.append("get_vm_status")
            return HelperVMStatusReply(
                protocol_version="1",
                helper_version="0.1.0",
                vm_id=vm_id,
                state="booting",
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
            assert request["run_id"] == "vz-run-readiness-timeout"
            assert request["session_id"] == "sess-readiness-timeout"
            assert request["session_mode"] is True
            raise MacOSVirtualizationHelperFailure(
                "guest_readiness_timed_out",
                "guest readiness timed out",
            )

        def exec_guest(self, *, vm_id: str, request: dict[str, object]) -> HelperExecReply:
            raise AssertionError(f"exec_guest should not run after readiness failure: {vm_id} {request}")

        def terminate_vm(self, vm_id: str) -> bool:
            terminated.append(vm_id)
            return True

    monkeypatch.setattr(vz_linux_module.VZLinuxRunner, "helper_client_cls", _FakeHelper)

    run_id = "vz-run-readiness-timeout"
    try:
        status = VZLinuxRunner(session_control_store=_Store()).start_run(
            run_id=run_id,
            spec=RunSpec(
                session_id="sess-readiness-timeout",
                runtime=RuntimeType.vz_linux,
                base_image="ubuntu-24.04",
                command=["/bin/echo", "ok"],
                network_policy="deny_all",
            ),
            session_workspace=str(tmp_path),
        )

        assert status.phase == RunPhase.failed
        assert "guest_readiness_timed_out" in status.message
        assert calls == ["get_vm_status", "validate_template", "create_vm"]
        assert deleted == ["sess-readiness-timeout"]
        assert stored == []
        assert terminated == []
        with VZLinuxRunner._active_lock:  # type: ignore[attr-defined]
            assert run_id not in VZLinuxRunner._active_vm  # type: ignore[attr-defined]
            assert run_id not in VZLinuxRunner._active_run_dir  # type: ignore[attr-defined]
    finally:
        with VZLinuxRunner._active_lock:  # type: ignore[attr-defined]
            VZLinuxRunner._active_vm.pop(run_id, None)  # type: ignore[attr-defined]
            VZLinuxRunner._active_run_dir.pop(run_id, None)  # type: ignore[attr-defined]


def test_vz_linux_raw_template_path_ignores_unavailable_image_store(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("TLDW_SANDBOX_VZ_LINUX_FAKE_EXEC", raising=False)
    broken_store_root = tmp_path / "not-a-directory"
    broken_store_root.write_text("not a directory", encoding="utf-8")
    monkeypatch.setenv("TLDW_SANDBOX_IMAGE_STORE_ROOT", str(broken_store_root))
    calls: list[tuple[str, dict[str, object]]] = []

    class _FakeHelper:
        def validate_template(self, request: dict[str, object]) -> dict[str, object]:
            calls.append(("validate_template", dict(request)))
            return {
                "template_id": "vz_linux:raw-template",
                "source": "/tmp/raw-vz-bundle",
                "ready": True,
                "reasons": [],
            }

        def create_vm(self, request: dict[str, object]) -> HelperVMReply:
            calls.append(("create_vm", dict(request)))
            return HelperVMReply(vm_id="vm-raw", state="created", details={})

        def exec_guest(self, *, vm_id: str, request: dict[str, object]) -> HelperExecReply:
            calls.append(("exec_guest", {"vm_id": vm_id, **request}))
            return HelperExecReply(exit_code=0, stdout=b"ok\n")

        def terminate_vm(self, vm_id: str) -> bool:
            calls.append(("terminate_vm", {"vm_id": vm_id}))
            return True

    monkeypatch.setattr(vz_linux_module.VZLinuxRunner, "helper_client_cls", _FakeHelper)

    status = VZLinuxRunner().start_run(
        run_id="vz-run-raw-path",
        spec=RunSpec(
            session_id=None,
            runtime=RuntimeType.vz_linux,
            base_image="/tmp/raw-vz-bundle",
            command=["/bin/echo", "ok"],
            network_policy="deny_all",
        ),
        session_workspace=str(tmp_path / "workspace"),
    )

    assert status.phase == RunPhase.completed
    assert calls[0] == ("validate_template", {"runtime": "vz_linux", "template": "/tmp/raw-vz-bundle"})


def test_vz_linux_exec_failure_terminates_vm_and_removes_created_workspace(
    monkeypatch,
) -> None:
    monkeypatch.delenv("TLDW_SANDBOX_VZ_LINUX_FAKE_EXEC", raising=False)
    calls: list[tuple[str, object]] = []
    workspaces: list[Path] = []

    class _FakeHelper:
        def validate_template(self, request: dict[str, object]) -> dict[str, object]:
            calls.append(("validate_template", dict(request)))
            return {
                "template_id": "vz_linux:test-template",
                "source": "/tmp/raw-vz-bundle",
                "ready": True,
                "reasons": [],
            }

        def create_vm(self, request: dict[str, object]) -> HelperVMReply:
            calls.append(("create_vm", dict(request)))
            workspaces.append(Path(str(request["workspace_path"])))
            return HelperVMReply(vm_id="vm-exec-fails", state="created", details={})

        def exec_guest(self, *, vm_id: str, request: dict[str, object]) -> HelperExecReply:
            calls.append(("exec_guest", {"vm_id": vm_id, **request}))
            raise RuntimeError("guest_exec_failed")

        def terminate_vm(self, vm_id: str) -> bool:
            calls.append(("terminate_vm", vm_id))
            return True

    monkeypatch.setattr(vz_linux_module.VZLinuxRunner, "helper_client_cls", _FakeHelper)

    rid = "vz-run-exec-fails-cleanup"
    try:
        status = VZLinuxRunner().start_run(
            run_id=rid,
            spec=RunSpec(
                session_id=None,
                runtime=RuntimeType.vz_linux,
                base_image="/tmp/raw-vz-bundle",
                command=["/bin/echo", "ok"],
                network_policy="deny_all",
            ),
            session_workspace=None,
        )

        assert status.phase == RunPhase.failed
        assert "guest_exec_failed" in status.message
        assert ("terminate_vm", "vm-exec-fails") in calls
        assert workspaces
        assert not workspaces[0].exists()
        with VZLinuxRunner._active_lock:  # type: ignore[attr-defined]
            assert rid not in VZLinuxRunner._active_vm  # type: ignore[attr-defined]
            assert rid not in VZLinuxRunner._active_run_dir  # type: ignore[attr-defined]
    finally:
        with VZLinuxRunner._active_lock:  # type: ignore[attr-defined]
            VZLinuxRunner._active_vm.pop(rid, None)  # type: ignore[attr-defined]
            VZLinuxRunner._active_run_dir.pop(rid, None)  # type: ignore[attr-defined]
        if workspaces and workspaces[0].exists():
            shutil.rmtree(workspaces[0], ignore_errors=True)


def test_vz_linux_start_run_fails_when_image_store_template_id_has_no_source_path(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.delenv("TLDW_SANDBOX_VZ_LINUX_FAKE_EXEC", raising=False)
    store_root = tmp_path / "image-store"
    artifact = tmp_path / "rootfs.img"
    artifact.write_bytes(b"rootfs")
    template_id = SandboxImageStore(root_path=store_root).register_template(
        runtime="vz_linux",
        template_name="no-source",
        disk_paths=[str(artifact)],
    )
    monkeypatch.setenv("TLDW_SANDBOX_IMAGE_STORE_ROOT", str(store_root))

    class _FakeHelper:
        def validate_template(self, request: dict[str, object]) -> dict[str, object]:
            raise AssertionError("validate_template should not run without a source_path")

    monkeypatch.setattr(vz_linux_module.VZLinuxRunner, "helper_client_cls", _FakeHelper)

    status = VZLinuxRunner().start_run(
        run_id="vz-run-missing-source",
        spec=RunSpec(
            session_id=None,
            runtime=RuntimeType.vz_linux,
            base_image=template_id,
            command=["/bin/echo", "ok"],
            network_policy="deny_all",
        ),
        session_workspace=str(tmp_path / "workspace"),
    )

    assert status.phase == RunPhase.failed
    assert "image_store_template_source_missing" in status.message


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
                "helper_instance_id": "helper-a",
                "helper_started_at": "2026-05-09T00:00:00Z",
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
                metadata=HelperVMMetadata(
                    owner="tldw",
                    runtime="vz_linux",
                    session_id="sess-1",
                    session_mode=True,
                ),
                details={
                    "helper_instance_id": "helper-a",
                    "helper_started_at": "2026-05-09T00:00:00Z",
                },
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


def test_vz_linux_session_reuse_generation_mismatch_recreates_vm(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("TLDW_SANDBOX_VZ_LINUX_FAKE_EXEC", raising=False)
    calls: list[str] = []
    deleted: list[str] = []
    stored: list[dict[str, object]] = []

    class _Store:
        def get_vz_session_control(self, session_id: str) -> dict[str, object]:
            assert session_id == "sess-generation-mismatch"
            return {
                "runtime": "vz_linux",
                "vm_id": "vm-stale-generation",
                "template_id": "vz_linux:existing",
                "workspace_mount": str(tmp_path),
                "agent_ready": True,
                "helper_instance_id": "helper-old",
                "helper_started_at": "2026-05-09T00:00:00Z",
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
                state="running",
                healthy=True,
                metadata=HelperVMMetadata(
                    owner="tldw",
                    runtime="vz_linux",
                    session_id="sess-generation-mismatch",
                    session_mode=True,
                ),
                details={
                    "helper_instance_id": "helper-new",
                    "helper_started_at": "2026-05-09T01:00:00Z",
                },
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
            assert request["session_id"] == "sess-generation-mismatch"
            assert request["session_mode"] is True
            return HelperVMReply(
                vm_id="vm-new-generation",
                state="created",
                details={
                    "helper_instance_id": "helper-new",
                    "helper_started_at": "2026-05-09T01:00:00Z",
                },
            )

        def exec_guest(self, *, vm_id: str, request: dict[str, object]) -> HelperExecReply:
            calls.append("exec_guest")
            assert vm_id == "vm-new-generation"
            return HelperExecReply(exit_code=0, stdout=b"recreated\n")

    monkeypatch.setattr(vz_linux_module.VZLinuxRunner, "helper_client_cls", _FakeHelper)

    status = VZLinuxRunner(session_control_store=_Store()).start_run(
        run_id="vz-run-generation-mismatch",
        spec=RunSpec(
            session_id="sess-generation-mismatch",
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
    assert deleted == ["sess-generation-mismatch"]
    assert stored and stored[0]["vm_id"] == "vm-new-generation"
    assert stored[0]["helper_instance_id"] == "helper-new"
    assert stored[0]["helper_started_at"] == "2026-05-09T01:00:00Z"


@pytest.mark.parametrize(
    ("metadata", "case_name"),
    [
        (
            HelperVMMetadata(
                owner="other",
                runtime="vz_linux",
                session_id="sess-metadata-mismatch",
                session_mode=True,
            ),
            "owner",
        ),
        (
            HelperVMMetadata(
                owner="tldw",
                runtime="vz_macos",
                session_id="sess-metadata-mismatch",
                session_mode=True,
            ),
            "runtime",
        ),
        (
            HelperVMMetadata(
                owner="tldw",
                runtime="vz_linux",
                session_id="other-session",
                session_mode=True,
            ),
            "session",
        ),
    ],
)
def test_vz_linux_session_reuse_metadata_mismatch_recreates_vm(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    metadata: HelperVMMetadata,
    case_name: str,
) -> None:
    monkeypatch.delenv("TLDW_SANDBOX_VZ_LINUX_FAKE_EXEC", raising=False)
    calls: list[str] = []
    deleted: list[str] = []
    stored: list[dict[str, object]] = []

    class _Store:
        def get_vz_session_control(self, session_id: str) -> dict[str, object]:
            assert session_id == "sess-metadata-mismatch"
            return {
                "runtime": "vz_linux",
                "vm_id": f"vm-stale-{case_name}",
                "template_id": "vz_linux:existing",
                "workspace_mount": str(tmp_path),
                "agent_ready": True,
                "helper_instance_id": "helper-a",
                "helper_started_at": "2026-05-09T00:00:00Z",
            }

        def delete_vz_session_control(self, session_id: str) -> bool:
            deleted.append(session_id)
            return True

        def put_vz_session_control(self, **kwargs: object) -> None:
            stored.append(dict(kwargs))

    class _FakeHelper:
        def get_vm_status(self, vm_id: str) -> HelperVMStatusReply:
            calls.append("get_vm_status")
            return HelperVMStatusReply(
                protocol_version="1",
                helper_version="0.1.0",
                vm_id=vm_id,
                state="running",
                healthy=True,
                metadata=metadata,
                details={
                    "helper_instance_id": "helper-a",
                    "helper_started_at": "2026-05-09T00:00:00Z",
                },
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
            assert request["session_id"] == "sess-metadata-mismatch"
            assert request["session_mode"] is True
            return HelperVMReply(
                vm_id=f"vm-new-{case_name}",
                state="created",
                details={
                    "helper_instance_id": "helper-a",
                    "helper_started_at": "2026-05-09T00:00:00Z",
                },
            )

        def exec_guest(self, *, vm_id: str, request: dict[str, object]) -> HelperExecReply:
            calls.append("exec_guest")
            assert vm_id == f"vm-new-{case_name}"
            return HelperExecReply(exit_code=0, stdout=b"recreated\n")

    monkeypatch.setattr(vz_linux_module.VZLinuxRunner, "helper_client_cls", _FakeHelper)

    status = VZLinuxRunner(session_control_store=_Store()).start_run(
        run_id=f"vz-run-metadata-mismatch-{case_name}",
        spec=RunSpec(
            session_id="sess-metadata-mismatch",
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
    assert deleted == ["sess-metadata-mismatch"]
    assert stored and stored[0]["vm_id"] == f"vm-new-{case_name}"


def test_vz_linux_session_reuse_guest_agent_mismatch_recreates_vm(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("TLDW_SANDBOX_VZ_LINUX_FAKE_EXEC", raising=False)
    calls: list[str] = []
    deleted: list[str] = []
    stored: list[dict[str, object]] = []

    class _Store:
        def get_vz_session_control(self, session_id: str) -> dict[str, object]:
            assert session_id == "sess-guest-agent-mismatch"
            return {
                "runtime": "vz_linux",
                "vm_id": "vm-stale-guest-agent",
                "template_id": "vz_linux:existing",
                "workspace_mount": str(tmp_path),
                "agent_ready": True,
                "helper_instance_id": "helper-a",
                "helper_started_at": "2026-05-09T00:00:00Z",
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
                state="running",
                healthy=True,
                metadata=HelperVMMetadata(
                    owner="tldw",
                    runtime="vz_linux",
                    session_id="sess-guest-agent-mismatch",
                    session_mode=True,
                ),
                details={
                    "helper_instance_id": "helper-a",
                    "helper_started_at": "2026-05-09T00:00:00Z",
                    "guest_version": "0.9.0",
                    "guest_workspace_root": "/var/empty",
                    "guest_capabilities_known": "true",
                    "guest_capabilities": "output_cap_v1",
                },
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
            assert request["session_id"] == "sess-guest-agent-mismatch"
            assert request["session_mode"] is True
            return HelperVMReply(
                vm_id="vm-new-guest-agent",
                state="created",
                details={
                    "helper_instance_id": "helper-a",
                    "helper_started_at": "2026-05-09T00:00:00Z",
                },
            )

        def exec_guest(self, *, vm_id: str, request: dict[str, object]) -> HelperExecReply:
            calls.append("exec_guest")
            assert vm_id == "vm-new-guest-agent"
            return HelperExecReply(exit_code=0, stdout=b"recreated\n")

    monkeypatch.setattr(vz_linux_module.VZLinuxRunner, "helper_client_cls", _FakeHelper)

    status = VZLinuxRunner(session_control_store=_Store()).start_run(
        run_id="vz-run-guest-agent-mismatch",
        spec=RunSpec(
            session_id="sess-guest-agent-mismatch",
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
    assert deleted == ["sess-guest-agent-mismatch"]
    assert stored and stored[0]["vm_id"] == "vm-new-guest-agent"


def test_vz_linux_session_reuse_helper_unavailable_fails_closed(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("TLDW_SANDBOX_VZ_LINUX_FAKE_EXEC", raising=False)
    calls: list[str] = []
    deleted: list[str] = []
    stored: list[dict[str, object]] = []

    class _Store:
        def get_vz_session_control(self, session_id: str) -> dict[str, object]:
            assert session_id == "sess-helper-unavailable"
            return {
                "runtime": "vz_linux",
                "vm_id": "vm-candidate",
                "template_id": "vz_linux:existing",
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
            assert vm_id == "vm-candidate"
            calls.append("get_vm_status")
            raise MacOSVirtualizationHelperUnavailable("macos_virtualization_helper_unavailable")

        def validate_template(self, request: dict[str, object]) -> dict[str, object]:
            calls.append("validate_template")
            return {
                "template_id": "vz_linux:new-template",
                "ready": True,
                "reasons": [],
            }

        def create_vm(self, request: dict[str, object]) -> HelperVMReply:
            calls.append("create_vm")
            assert request["run_id"] == "vz-run-helper-unavailable"
            assert request["session_id"] == "sess-helper-unavailable"
            assert request["session_mode"] is True
            return HelperVMReply(vm_id="vm-new", state="created")

        def exec_guest(self, *, vm_id: str, request: dict[str, object]) -> HelperExecReply:
            calls.append("exec_guest")
            assert vm_id == "vm-new"
            return HelperExecReply(exit_code=0, stdout=b"recreated\n")

    monkeypatch.setattr(vz_linux_module.VZLinuxRunner, "helper_client_cls", _FakeHelper)

    status = VZLinuxRunner(session_control_store=_Store()).start_run(
        run_id="vz-run-helper-unavailable",
        spec=RunSpec(
            session_id="sess-helper-unavailable",
            runtime=RuntimeType.vz_linux,
            base_image="ubuntu-24.04",
            command=["/bin/echo", "ok"],
            network_policy="deny_all",
        ),
        session_workspace=str(tmp_path),
    )

    assert status.phase == RunPhase.failed
    assert status.exit_code is None
    assert "macos_virtualization_helper_unavailable" in status.message
    assert calls == ["get_vm_status"]
    assert deleted == []
    assert stored == []


def test_vz_linux_session_reuse_protocol_mismatch_fails_closed(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("TLDW_SANDBOX_VZ_LINUX_FAKE_EXEC", raising=False)
    calls: list[str] = []
    deleted: list[str] = []
    stored: list[dict[str, object]] = []

    class _Store:
        def get_vz_session_control(self, session_id: str) -> dict[str, object]:
            assert session_id == "sess-protocol-mismatch"
            return {
                "runtime": "vz_linux",
                "vm_id": "vm-candidate",
                "template_id": "vz_linux:existing",
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
            assert vm_id == "vm-candidate"
            calls.append("get_vm_status")
            raise MacOSVirtualizationHelperProtocolError("macos_virtualization_helper_protocol_mismatch")

        def validate_template(self, request: dict[str, object]) -> dict[str, object]:
            calls.append("validate_template")
            return {
                "template_id": "vz_linux:new-template",
                "ready": True,
                "reasons": [],
            }

        def create_vm(self, request: dict[str, object]) -> HelperVMReply:
            calls.append("create_vm")
            assert request["run_id"] == "vz-run-protocol-mismatch"
            assert request["session_id"] == "sess-protocol-mismatch"
            assert request["session_mode"] is True
            return HelperVMReply(vm_id="vm-new", state="created")

        def exec_guest(self, *, vm_id: str, request: dict[str, object]) -> HelperExecReply:
            calls.append("exec_guest")
            assert vm_id == "vm-new"
            return HelperExecReply(exit_code=0, stdout=b"recreated\n")

    monkeypatch.setattr(vz_linux_module.VZLinuxRunner, "helper_client_cls", _FakeHelper)

    status = VZLinuxRunner(session_control_store=_Store()).start_run(
        run_id="vz-run-protocol-mismatch",
        spec=RunSpec(
            session_id="sess-protocol-mismatch",
            runtime=RuntimeType.vz_linux,
            base_image="ubuntu-24.04",
            command=["/bin/echo", "ok"],
            network_policy="deny_all",
        ),
        session_workspace=str(tmp_path),
    )

    assert status.phase == RunPhase.failed
    assert status.exit_code is None
    assert "macos_virtualization_helper_protocol_mismatch" in status.message
    assert calls == ["get_vm_status"]
    assert deleted == []
    assert stored == []


def test_vz_linux_session_reuse_absent_status_recreates_vm(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("TLDW_SANDBOX_VZ_LINUX_FAKE_EXEC", raising=False)
    calls: list[str] = []
    deleted: list[str] = []
    stored: list[dict[str, object]] = []

    class _Store:
        def get_vz_session_control(self, session_id: str) -> dict[str, object]:
            assert session_id == "sess-absent-status"
            return {
                "runtime": "vz_linux",
                "vm_id": "vm-candidate",
                "template_id": "vz_linux:existing",
                "workspace_mount": str(tmp_path),
                "agent_ready": True,
            }

        def delete_vz_session_control(self, session_id: str) -> bool:
            deleted.append(session_id)
            return True

        def put_vz_session_control(self, **kwargs) -> None:
            stored.append(dict(kwargs))

    class _FakeHelper:
        def get_vm_status(self, vm_id: str) -> HelperVMStatusReply | None:
            assert vm_id == "vm-candidate"
            calls.append("get_vm_status")
            return None

        def validate_template(self, request: dict[str, object]) -> dict[str, object]:
            calls.append("validate_template")
            return {
                "template_id": "vz_linux:new-template",
                "ready": True,
                "reasons": [],
            }

        def create_vm(self, request: dict[str, object]) -> HelperVMReply:
            calls.append("create_vm")
            assert request["run_id"] == "vz-run-absent-status"
            assert request["session_id"] == "sess-absent-status"
            assert request["session_mode"] is True
            return HelperVMReply(
                vm_id="vm-new",
                state="created",
                details={
                    "helper_instance_id": "helper-new",
                    "helper_started_at": "2026-05-09T01:00:00Z",
                },
            )

        def exec_guest(self, *, vm_id: str, request: dict[str, object]) -> HelperExecReply:
            calls.append("exec_guest")
            assert vm_id == "vm-new"
            return HelperExecReply(exit_code=0, stdout=b"recreated\n")

    monkeypatch.setattr(vz_linux_module.VZLinuxRunner, "helper_client_cls", _FakeHelper)

    status = VZLinuxRunner(session_control_store=_Store()).start_run(
        run_id="vz-run-absent-status",
        spec=RunSpec(
            session_id="sess-absent-status",
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
    assert deleted == ["sess-absent-status"]
    assert stored and stored[0]["vm_id"] == "vm-new"
    assert stored[0]["helper_instance_id"] == "helper-new"
    assert stored[0]["helper_started_at"] == "2026-05-09T01:00:00Z"


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
            assert request["owner"] == "tldw"
            assert request["runtime"] == "vz_linux"
            assert request["run_id"] == "vz-run-recreate"
            assert request["session_id"] == "sess-2"
            assert request["session_mode"] is True
            assert request["template"] == "ubuntu-24.04"
            return HelperVMReply(
                vm_id="vm-new",
                state="created",
                details={
                    "helper_instance_id": "helper-new",
                    "helper_started_at": "2026-05-09T01:00:00Z",
                },
            )

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
    assert stored[0]["helper_instance_id"] == "helper-new"
    assert stored[0]["helper_started_at"] == "2026-05-09T01:00:00Z"


def test_vz_linux_start_run_passes_log_cap_to_helper(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("TLDW_SANDBOX_VZ_LINUX_FAKE_EXEC", raising=False)
    monkeypatch.setattr(
        vz_linux_module.SandboxPolicyConfig,
        "from_settings",
        classmethod(lambda cls: cls(max_log_bytes=5)),
    )
    exec_requests: list[dict[str, object]] = []

    class _FakeHelper:
        def validate_template(self, request: dict[str, object]) -> dict[str, object]:
            return {
                "template_id": "vz_linux:validated-ubuntu",
                "source": "ubuntu-24.04",
                "ready": True,
                "reasons": [],
            }

        def create_vm(self, request: dict[str, object]) -> HelperVMReply:
            return HelperVMReply(vm_id="vm-output-cap", state="created")

        def exec_guest(self, *, vm_id: str, request: dict[str, object]) -> HelperExecReply:
            exec_requests.append({"vm_id": vm_id, **request})
            return HelperExecReply(exit_code=0, stdout=b"ok\n")

        def terminate_vm(self, vm_id: str) -> bool:
            return True

    monkeypatch.setattr(vz_linux_module.VZLinuxRunner, "helper_client_cls", _FakeHelper)

    status = VZLinuxRunner().start_run(
        run_id="vz-run-output-cap",
        spec=RunSpec(
            session_id=None,
            runtime=RuntimeType.vz_linux,
            base_image="ubuntu-24.04",
            command=["/bin/echo", "ok"],
            network_policy="deny_all",
        ),
        session_workspace=str(tmp_path),
    )

    assert status.phase == RunPhase.completed
    assert exec_requests[0]["max_output_bytes"] == 5


def test_vz_linux_start_run_clamps_log_cap_to_helper_limit(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("TLDW_SANDBOX_VZ_LINUX_FAKE_EXEC", raising=False)
    monkeypatch.setattr(
        vz_linux_module.SandboxPolicyConfig,
        "from_settings",
        classmethod(lambda cls: cls(max_log_bytes=VZLinuxRunner.max_helper_output_bytes + 1)),
    )
    exec_requests: list[dict[str, object]] = []

    class _FakeHelper:
        def validate_template(self, request: dict[str, object]) -> dict[str, object]:
            return {
                "template_id": "vz_linux:validated-ubuntu",
                "source": "ubuntu-24.04",
                "ready": True,
                "reasons": [],
            }

        def create_vm(self, request: dict[str, object]) -> HelperVMReply:
            return HelperVMReply(vm_id="vm-output-cap", state="created")

        def exec_guest(self, *, vm_id: str, request: dict[str, object]) -> HelperExecReply:
            exec_requests.append({"vm_id": vm_id, **request})
            return HelperExecReply(exit_code=0, stdout=b"ok\n")

        def terminate_vm(self, vm_id: str) -> bool:
            return True

    monkeypatch.setattr(vz_linux_module.VZLinuxRunner, "helper_client_cls", _FakeHelper)

    status = VZLinuxRunner().start_run(
        run_id="vz-run-output-cap-clamped",
        spec=RunSpec(
            session_id=None,
            runtime=RuntimeType.vz_linux,
            base_image="ubuntu-24.04",
            command=["/bin/echo", "ok"],
            network_policy="deny_all",
        ),
        session_workspace=str(tmp_path),
    )

    assert status.phase == RunPhase.completed
    assert exec_requests[0]["max_output_bytes"] == VZLinuxRunner.max_helper_output_bytes


def test_feature_discovery_advertises_vz_linux_helper_log_cap() -> None:
    configured = VZLinuxRunner.max_helper_output_bytes + 1024
    service = SandboxService(
        policy=SandboxPolicy(SandboxPolicyConfig(max_log_bytes=configured)),
    )

    runtimes = {item["name"]: item for item in service.feature_discovery()}

    assert runtimes["vz_linux"]["max_log_bytes"] == VZLinuxRunner.max_helper_output_bytes
    assert runtimes["docker"]["max_log_bytes"] == configured


def test_vz_linux_collect_artifacts_uses_policy_caps(monkeypatch, tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "too-large.txt").write_bytes(b"1234")
    monkeypatch.setattr(
        vz_linux_module.SandboxPolicyConfig,
        "from_settings",
        classmethod(
            lambda cls: cls(
                max_artifact_file_bytes=3,
                max_artifact_total_bytes=10,
            )
        ),
    )

    artifacts = VZLinuxRunner._collect_artifacts(str(workspace), ["*.txt"])

    assert artifacts == {}


def test_output_counters_include_guest_enforcement_details() -> None:
    counters = VZLinuxRunner._output_counters_from_details(
        {
            "guest_output_limit_bytes": "16",
            "guest_output_limit_exceeded": "true",
            "guest_stdout_bytes_observed": "17",
            "guest_stderr_bytes_observed": "0",
            "guest_stdout_bytes_returned": "16",
            "guest_stderr_bytes_returned": "0",
            "guest_output_kill_reason": "output_limit",
            "ignored": "not-int",
        }
    )

    assert counters["guest_output_limit_bytes"] == 16
    assert counters["guest_output_limit_exceeded"] == 1
    assert counters["guest_stdout_bytes_observed"] == 17
    assert counters["guest_stdout_bytes_returned"] == 16
    assert "guest_output_kill_reason" not in counters


def test_vz_linux_start_run_records_output_limit_counters(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("TLDW_SANDBOX_VZ_LINUX_FAKE_EXEC", raising=False)
    monkeypatch.setattr(
        vz_linux_module.SandboxPolicyConfig,
        "from_settings",
        classmethod(lambda cls: cls(max_log_bytes=5)),
    )

    class _FakeHelper:
        def validate_template(self, request: dict[str, object]) -> dict[str, object]:
            return {
                "template_id": "vz_linux:validated-ubuntu",
                "source": "ubuntu-24.04",
                "ready": True,
                "reasons": [],
            }

        def create_vm(self, request: dict[str, object]) -> HelperVMReply:
            return HelperVMReply(vm_id="vm-output-counters", state="created")

        def exec_guest(self, *, vm_id: str, request: dict[str, object]) -> HelperExecReply:
            return HelperExecReply(
                exit_code=0,
                stdout=b"hello",
                stderr=b"",
                details={
                    "output_limit_bytes": "5",
                    "stdout_bytes_original": "11",
                    "stderr_bytes_original": "0",
                    "stdout_bytes_returned": "5",
                    "stderr_bytes_returned": "0",
                    "stdout_truncated": "true",
                    "stderr_truncated": "false",
                    "non_counter": "ignored",
                },
            )

        def terminate_vm(self, vm_id: str) -> bool:
            return True

    monkeypatch.setattr(vz_linux_module.VZLinuxRunner, "helper_client_cls", _FakeHelper)

    status = VZLinuxRunner().start_run(
        run_id="vz-run-output-counters",
        spec=RunSpec(
            session_id=None,
            runtime=RuntimeType.vz_linux,
            base_image="ubuntu-24.04",
            command=["/bin/echo", "ok"],
            network_policy="deny_all",
        ),
        session_workspace=str(tmp_path),
    )

    assert status.phase == RunPhase.completed
    assert status.resource_usage["output_limit_bytes"] == 5
    assert status.resource_usage["stdout_bytes_original"] == 11
    assert status.resource_usage["stdout_bytes_returned"] == 5
    assert status.resource_usage["stdout_truncated"] == 1
    assert status.resource_usage["stderr_truncated"] == 0
    assert "non_counter" not in status.resource_usage


def test_vz_linux_start_run_applies_artifact_capture_caps(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("TLDW_SANDBOX_VZ_LINUX_FAKE_EXEC", raising=False)
    monkeypatch.setattr(
        vz_linux_module.SandboxPolicyConfig,
        "from_settings",
        classmethod(
            lambda cls: cls(
                max_artifact_file_bytes=5,
                max_artifact_total_bytes=8,
            )
        ),
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    class _FakeHelper:
        def validate_template(self, request: dict[str, object]) -> dict[str, object]:
            return {
                "template_id": "vz_linux:validated-ubuntu",
                "source": "ubuntu-24.04",
                "ready": True,
                "reasons": [],
            }

        def create_vm(self, request: dict[str, object]) -> HelperVMReply:
            return HelperVMReply(vm_id="vm-artifact-caps", state="created")

        def exec_guest(self, *, vm_id: str, request: dict[str, object]) -> HelperExecReply:
            (workspace / "small.txt").write_bytes(b"1234")
            (workspace / "too-large.txt").write_bytes(b"123456")
            (workspace / "would-exceed-total.txt").write_bytes(b"56789")
            return HelperExecReply(exit_code=0, stdout=b"ok\n")

        def terminate_vm(self, vm_id: str) -> bool:
            return True

    monkeypatch.setattr(vz_linux_module.VZLinuxRunner, "helper_client_cls", _FakeHelper)

    status = VZLinuxRunner().start_run(
        run_id="vz-run-artifact-caps",
        spec=RunSpec(
            session_id=None,
            runtime=RuntimeType.vz_linux,
            base_image="ubuntu-24.04",
            command=["/bin/echo", "ok"],
            network_policy="deny_all",
            capture_patterns=["*.txt"],
        ),
        session_workspace=str(workspace),
    )

    assert status.phase == RunPhase.completed
    assert status.artifacts == {"small.txt": b"1234"}
    assert status.resource_usage["artifact_limit_file_bytes"] == 5
    assert status.resource_usage["artifact_limit_total_bytes"] == 8
    assert status.resource_usage["artifact_files_collected"] == 1
    assert status.resource_usage["artifact_files_skipped"] == 2
    assert status.resource_usage["artifact_skip_file_limit"] == 1
    assert status.resource_usage["artifact_skip_total_limit"] == 1
    assert status.resource_usage["artifact_bytes"] == 4
