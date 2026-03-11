from __future__ import annotations

import os
import platform
import sys
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
    _expect(status.phase == RunPhase.completed, f"Expected completed phase, got {status.phase!r}")
    _expect(status.exit_code == 0, f"Expected exit code 0, got {status.exit_code!r}")
    _expect("vz-linux-e2e" in stdout_text, f"Expected stdout token in output, got {stdout_text!r}")


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
