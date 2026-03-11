from __future__ import annotations

import os
import platform
import sys
from pathlib import Path

import pytest

from tldw_Server_API.app.core.config import clear_config_cache, settings as app_settings
from tldw_Server_API.app.core.Sandbox.models import RunPhase, RunSpec, RuntimeType, SessionSpec
from tldw_Server_API.app.core.Sandbox.runners.vz_linux_runner import VZLinuxRunner
from tldw_Server_API.app.core.Sandbox.service import SandboxService
from tldw_Server_API.app.core.Sandbox.streams import get_hub
from tldw_Server_API.app.core.testing import is_truthy


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
        pytest.skip("TLDW_SANDBOX_VZ_LINUX_E2E_BASE_IMAGE is required")
    monkeypatch.setenv("SANDBOX_ENABLE_EXECUTION", "1")
    monkeypatch.setenv("SANDBOX_BACKGROUND_EXECUTION", "0")
    return base_image


def test_vz_linux_real_host_e2e_requires_opt_in(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    monkeypatch.delenv("TLDW_SANDBOX_VZ_LINUX_E2E", raising=False)

    with pytest.raises(pytest.skip.Exception, match="TLDW_SANDBOX_VZ_LINUX_E2E"):
        _require_vz_linux_real_host_e2e(monkeypatch, tmp_path)


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS host only")
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
    assert status.phase == RunPhase.completed
    assert status.exit_code == 0
    assert "vz-linux-e2e" in stdout_text


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS host only")
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

        assert first.phase == RunPhase.completed
        assert second.phase == RunPhase.completed
        assert control_after_first is not None
        assert control_after_second is not None
        assert control_after_first["vm_id"] == control_after_second["vm_id"]
        assert service.destroy_session(session.id) is True
        destroyed = True
        assert service._orch.get_vz_session_control(session.id) is None
    finally:
        if session_id and not destroyed:
            service.destroy_session(session_id)
