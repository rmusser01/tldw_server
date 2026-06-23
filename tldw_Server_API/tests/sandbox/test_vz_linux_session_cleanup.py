from __future__ import annotations

from pathlib import Path

import tldw_Server_API.app.core.Sandbox.service as service_module
from tldw_Server_API.app.core.config import clear_config_cache, settings as app_settings
from tldw_Server_API.app.core.Sandbox.models import RuntimeType, SessionSpec
from tldw_Server_API.app.core.Sandbox.runtime_capabilities import RuntimePreflightResult
from tldw_Server_API.app.core.Sandbox.service import SandboxService


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


def _stub_vz_linux_preflight(monkeypatch) -> None:
    def _collect_runtime_preflights(
        self: SandboxService,
        *,
        network_policy: str | None,
    ) -> dict[RuntimeType, RuntimePreflightResult]:
        return {
            RuntimeType.vz_linux: RuntimePreflightResult(
                runtime=RuntimeType.vz_linux,
                available=True,
                reasons=[],
                execution_mode="helper",
                enforcement_ready={"deny_all": True, "allowlist": True},
            )
        }

    monkeypatch.setattr(SandboxService, "_collect_runtime_preflights", _collect_runtime_preflights)


def test_destroy_session_terminates_persisted_vz_linux_vm(monkeypatch, tmp_path: Path) -> None:
    _configure_sqlite_store(monkeypatch, tmp_path)
    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("TLDW_SANDBOX_VZ_LINUX_AVAILABLE", "1")
    monkeypatch.setenv("TLDW_SANDBOX_MACOS_HELPER_READY", "1")
    monkeypatch.setenv("TLDW_SANDBOX_VZ_LINUX_TEMPLATE_READY", "1")
    _stub_vz_linux_preflight(monkeypatch)
    terminated: list[str] = []

    class _FakeHelper:
        def terminate_vm(self, vm_id: str) -> bool:
            terminated.append(vm_id)
            return True

    monkeypatch.setattr(service_module, "MacOSVirtualizationHelperClient", _FakeHelper)

    svc = SandboxService()
    session = svc.create_session(
        user_id="user-55",
        spec=SessionSpec(runtime=RuntimeType.vz_linux, base_image="ubuntu-24.04"),
        spec_version="1.0",
        idem_key=None,
        raw_body={"spec_version": "1.0", "runtime": "vz_linux"},
    )
    workspace_mount = svc._orch.get_session_workspace_path(session.id)
    svc._orch.put_vz_session_control(
        session_id=session.id,
        runtime="vz_linux",
        vm_id="vm-session-1",
        template_id="vz_linux:ubuntu-24.04",
        workspace_mount=workspace_mount,
        agent_ready=True,
    )

    assert svc.destroy_session(session.id) is True
    assert terminated == ["vm-session-1"]
    assert svc._orch.get_vz_session_control(session.id) is None


def test_destroy_session_tolerates_already_absent_vz_linux_vm(monkeypatch, tmp_path: Path) -> None:
    _configure_sqlite_store(monkeypatch, tmp_path)
    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("TLDW_SANDBOX_VZ_LINUX_AVAILABLE", "1")
    monkeypatch.setenv("TLDW_SANDBOX_MACOS_HELPER_READY", "1")
    monkeypatch.setenv("TLDW_SANDBOX_VZ_LINUX_TEMPLATE_READY", "1")
    _stub_vz_linux_preflight(monkeypatch)
    terminated: list[str] = []

    class _FakeHelper:
        def terminate_vm(self, vm_id: str) -> bool:
            terminated.append(vm_id)
            return False

    monkeypatch.setattr(service_module, "MacOSVirtualizationHelperClient", _FakeHelper)

    svc = SandboxService()
    session = svc.create_session(
        user_id="user-56",
        spec=SessionSpec(runtime=RuntimeType.vz_linux, base_image="ubuntu-24.04"),
        spec_version="1.0",
        idem_key=None,
        raw_body={"spec_version": "1.0", "runtime": "vz_linux"},
    )
    workspace_mount = svc._orch.get_session_workspace_path(session.id)
    svc._orch.put_vz_session_control(
        session_id=session.id,
        runtime="vz_linux",
        vm_id="vm-missing",
        template_id="vz_linux:ubuntu-24.04",
        workspace_mount=workspace_mount,
        agent_ready=True,
    )

    assert svc.destroy_session(session.id) is True
    assert terminated == ["vm-missing"]
    assert svc._orch.get_vz_session_control(session.id) is None
