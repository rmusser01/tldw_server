from __future__ import annotations

from unittest.mock import patch

import pytest

from tldw_Server_API.app.core.Sandbox.models import RunSpec, RuntimeType
from tldw_Server_API.app.core.Sandbox.runtime_capabilities import RuntimePreflightResult
from tldw_Server_API.app.core.Sandbox.service import SandboxService


def _spec() -> RunSpec:
    return RunSpec(
        session_id=None,
        runtime=RuntimeType.docker,
        base_image="python:3.11-slim",
        command=["python", "-c", "print(1)"],
        env={},
        timeout_sec=5,
    )


def _stub_docker_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        SandboxService,
        "_collect_runtime_preflights",
        lambda self, network_policy=None: {
            RuntimeType.docker: RuntimePreflightResult(
                runtime=RuntimeType.docker,
                available=True,
                reasons=[],
                enforcement_ready={"deny_all": True, "allowlist": False},
            )
        },
    )


@pytest.mark.unit
def test_foreground_docker_runner_exception_marks_run_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SANDBOX_ENABLE_EXECUTION", "1")
    monkeypatch.setenv("SANDBOX_BACKGROUND_EXECUTION", "0")
    monkeypatch.setenv("TLDW_SANDBOX_DOCKER_FAKE_EXEC", "0")
    _stub_docker_preflight(monkeypatch)

    svc = SandboxService()

    with patch(
        "tldw_Server_API.app.core.Sandbox.runners.docker_runner.DockerRunner.start_run",
        side_effect=RuntimeError("boom"),
    ):
        status = svc.start_run_scaffold(
            user_id="1",
            spec=_spec(),
            spec_version="1.0",
            idem_key=None,
            raw_body={},
        )

    stored = svc.get_run(status.id)
    assert status.phase.value == "failed"
    assert status.message == "docker_failed"
    assert status.finished_at is not None
    assert stored is not None
    assert stored.phase.value == "failed"
    assert stored.message == "docker_failed"
    assert stored.finished_at is not None


@pytest.mark.unit
def test_background_docker_runner_exception_marks_run_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SANDBOX_ENABLE_EXECUTION", "1")
    monkeypatch.setenv("SANDBOX_BACKGROUND_EXECUTION", "1")
    monkeypatch.setenv("TLDW_SANDBOX_DOCKER_FAKE_EXEC", "0")
    _stub_docker_preflight(monkeypatch)

    svc = SandboxService()

    with patch(
        "tldw_Server_API.app.core.Sandbox.runners.docker_runner.DockerRunner.start_run",
        side_effect=RuntimeError("boom"),
    ):
        with patch.object(SandboxService, "_submit_background_worker", lambda self, fn: fn()):
            status = svc.start_run_scaffold(
                user_id="1",
                spec=_spec(),
                spec_version="1.0",
                idem_key=None,
                raw_body={},
            )

    stored = svc.get_run(status.id)
    assert status.phase.value == "failed"
    assert status.message == "docker_failed"
    assert status.finished_at is not None
    assert stored is not None
    assert stored.phase.value == "failed"
    assert stored.message == "docker_failed"
    assert stored.finished_at is not None
