from __future__ import annotations

import tldw_Server_API.app.core.Sandbox.runners.vz_common as vz_common
from tldw_Server_API.app.core.Sandbox.models import RunPhase, RunSpec, RuntimeType
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
