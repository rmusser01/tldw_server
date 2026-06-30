from __future__ import annotations

import sys

import pytest

import tldw_Server_API.app.core.Sandbox.runners.seatbelt_runner as seatbelt_module
from tldw_Server_API.app.core.Sandbox.macos_diagnostics import collect_macos_diagnostics
from tldw_Server_API.app.core.Sandbox.models import RunPhase, RunSpec, RuntimeType, TrustLevel
from tldw_Server_API.app.core.Sandbox.runners.seatbelt_runner import SeatbeltRunner
from tldw_Server_API.app.core.Sandbox.streams import get_hub
from tldw_Server_API.app.core.Sandbox.runners.vz_linux_runner import VZLinuxRunner
from tldw_Server_API.app.core.Sandbox.runners.vz_macos_runner import VZMacOSRunner


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS host only")
def test_vz_linux_preflight_smoke_on_real_host() -> None:
    result = VZLinuxRunner().preflight(network_policy="deny_all")

    assert isinstance(result.available, bool)
    assert isinstance(result.host, dict)
    assert "os" in result.host
    assert result.execution_mode in {"real", "none"}
    if result.available:
        assert result.execution_mode == "real"


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS host only")
def test_vz_macos_preflight_smoke_on_real_host() -> None:
    result = VZMacOSRunner().preflight(network_policy="deny_all")

    assert isinstance(result.available, bool)
    assert isinstance(result.host, dict)
    assert "os" in result.host


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS host only")
def test_collect_macos_diagnostics_smoke_on_real_host() -> None:
    data = collect_macos_diagnostics()

    assert "host" in data
    assert "helper" in data
    assert "templates" in data
    assert "runtimes" in data
    assert "reconciliation" in data
    assert isinstance(data["host"].get("macos_version"), (str, type(None)))
    assert data["runtimes"]["vz_linux"]["execution_mode"] in {"real", "none", "fake"}
    assert isinstance(data["reconciliation"].get("computed"), bool)
    if data["runtimes"]["vz_linux"]["available"]:
        assert data["runtimes"]["vz_linux"]["execution_mode"] == "real"


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS host only")
def test_seatbelt_real_execution_smoke_on_real_host() -> None:
    if not seatbelt_module._sandbox_exec_exists():
        pytest.skip("sandbox-exec is unavailable on this host")

    run_id = "smoke-seatbelt-real-host"
    hub = get_hub()
    hub._buffers.pop(run_id, None)  # type: ignore[attr-defined]

    status = SeatbeltRunner().start_run(
        run_id,
        RunSpec(
            session_id=None,
            runtime=RuntimeType.seatbelt,
            base_image="host-local",
            command=["/usr/bin/true"],
            timeout_sec=5,
            network_policy="deny_all",
            trust_level=TrustLevel.trusted,
        ),
        session_workspace=None,
    )

    frames = list(hub._buffers.get(run_id, []))  # type: ignore[attr-defined]
    stderr_text = "\n".join(
        str(frame.get("data", ""))
        for frame in frames
        if frame.get("type") == "stderr"
    )

    if status.phase == RunPhase.failed and (
        "Operation not permitted" in stderr_text or status.exit_code == 71
    ):
        pytest.skip("sandbox-exec is blocked by the enclosing execution sandbox")

    assert status.phase == RunPhase.completed
