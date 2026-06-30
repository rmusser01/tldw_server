from __future__ import annotations

import tldw_Server_API.app.core.Sandbox.service as service_module
from tldw_Server_API.app.core.Sandbox.models import RunPhase, RunSpec, RunStatus, RuntimeType, TrustLevel
from tldw_Server_API.app.core.Sandbox.policy import SandboxPolicy
from tldw_Server_API.app.core.Sandbox.service import SandboxService


def test_execute_single_runtime_scaffold_marks_policy_failures(monkeypatch) -> None:
    svc = SandboxService()
    status = RunStatus(id="run-1", phase=RunPhase.queued, runtime=RuntimeType.seatbelt)
    admitted = RunStatus(id="run-1", phase=RunPhase.starting, runtime=RuntimeType.seatbelt)
    updates: list[tuple[str, RunStatus]] = []

    def _fake_apply(target: RunStatus, source: RunStatus) -> None:
        target.phase = source.phase

    monkeypatch.setattr(svc, "_admit_run_starting", lambda run_id: admitted)
    monkeypatch.setattr(svc, "_apply_admitted_status", _fake_apply)
    monkeypatch.setattr(svc, "_run_with_claim_lease", lambda run_id, fn: fn())
    monkeypatch.setattr(svc._orch, "update_run", lambda run_id, state: updates.append((run_id, state)))

    def _raise_policy(run_id: str, spec: RunSpec, workspace_path: str | None):
        del run_id, spec, workspace_path
        raise SandboxPolicy.PolicyUnsupported(
            RuntimeType.seatbelt,
            requirement="standard",
            reasons=["seatbelt_standard_disabled"],
        )

    result = svc._execute_single_runtime_scaffold(
        status=status,
        spec=RunSpec(
            session_id=None,
            runtime=RuntimeType.seatbelt,
            base_image="host-local",
            command=["echo", "ok"],
            trust_level=TrustLevel.standard,
        ),
        workspace_path=None,
        start_run_fn=_raise_policy,
        policy_failed_reason="seatbelt_policy_failed",
        failed_reason="seatbelt_failed",
        policy_exceptions=(SandboxPolicy.RuntimeUnavailable, SandboxPolicy.PolicyUnsupported),
    )

    assert result.phase == RunPhase.failed
    assert result.message == "seatbelt_policy_failed"
    assert updates[-1][0] == "run-1"


def test_start_vz_linux_run_with_execution_preflight_dispatches_real_runner(monkeypatch) -> None:
    svc = SandboxService()
    calls: list[tuple[str, object]] = []

    class _FakePreflight:
        available = True
        reasons: list[str] = []

    class _FakeRunner:
        def __init__(self, session_control_store=None) -> None:
            calls.append(("init", session_control_store is svc._orch))

        def preflight(self, network_policy: str | None = None):
            calls.append(("preflight", network_policy))
            return _FakePreflight()

        def start_run(self, run_id: str, spec: RunSpec, workspace_path: str | None = None) -> RunStatus:
            calls.append(
                (
                    "start_run",
                    {
                        "run_id": run_id,
                        "workspace_path": workspace_path,
                        "command": list(spec.command),
                    },
                )
            )
            return RunStatus(
                id="",
                phase=RunPhase.completed,
                runtime=RuntimeType.vz_linux,
                exit_code=0,
            )

    monkeypatch.setattr(service_module, "VZLinuxRunner", _FakeRunner)

    result = svc._start_vz_linux_run_with_execution_preflight(
        "run-vz-1",
        RunSpec(
            session_id=None,
            runtime=RuntimeType.vz_linux,
            base_image="ubuntu-24.04",
            command=["/bin/echo", "ok"],
            network_policy="deny_all",
        ),
        "/tmp/vz-linux-workspace",
    )

    assert result.phase == RunPhase.completed
    assert calls == [
        ("init", True),
        ("preflight", "deny_all"),
        (
            "start_run",
            {
                "run_id": "run-vz-1",
                "workspace_path": "/tmp/vz-linux-workspace",
                "command": ["/bin/echo", "ok"],
            },
        ),
    ]
