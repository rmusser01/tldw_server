from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import tldw_Server_API.app.core.Sandbox.runners.lima_runner as lima_module
from tldw_Server_API.app.core.Sandbox.models import RunPhase, RunSpec, RuntimeType
from tldw_Server_API.app.core.Sandbox.runners.lima_runner import LimaRunner


def test_lima_real_run_replays_logs_with_configured_cap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("SANDBOX_MAX_LOG_BYTES", "5")
    monkeypatch.setenv("TLDW_SANDBOX_LIMA_VERSION", "0.0-test")

    run_dir = tmp_path / "lima-run"
    run_dir.mkdir()
    workspace = run_dir / "workspace"

    monkeypatch.setattr(lima_module.tempfile, "mkdtemp", lambda prefix: str(run_dir))
    monkeypatch.setattr(LimaRunner, "_tail_log", staticmethod(lambda run_id, log_path, stop_flag: None))

    def _fake_run(args: list[str], **kwargs: object) -> SimpleNamespace:
        del kwargs
        if args[:2] == ["limactl", "shell"]:
            (workspace / "run.log").write_bytes(b"abcdef")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(lima_module.subprocess, "run", _fake_run)

    status = LimaRunner()._run_real(
        "run-lima-log-cap",
        RunSpec(
            session_id=None,
            runtime=RuntimeType.lima,
            base_image="ubuntu-24.04",
            command=["/bin/echo", "ok"],
            network_policy="deny_all",
            timeout_sec=5,
        ),
    )

    assert status.phase == RunPhase.completed
    assert status.resource_usage["log_bytes"] == 5
    assert status.resource_usage["log_limit_bytes"] == 5
    assert status.resource_usage["log_truncated"] == 1
