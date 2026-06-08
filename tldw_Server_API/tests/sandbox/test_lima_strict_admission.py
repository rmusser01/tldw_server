from __future__ import annotations

import pytest

from tldw_Server_API.app.core.Sandbox.models import RunSpec, RuntimeType
from tldw_Server_API.app.core.Sandbox.policy import SandboxPolicy
from tldw_Server_API.app.core.Sandbox.runtime_capabilities import RuntimePreflightResult
from tldw_Server_API.app.core.Sandbox.runners.lima_runner import LimaRunner
from tldw_Server_API.app.core.Sandbox.service import SandboxService


def _stub_lima_preflight(
    monkeypatch: pytest.MonkeyPatch,
    *,
    allowlist_ready: bool = False,
    deny_all_ready: bool = True,
) -> None:
    def _fake_preflight(self, network_policy: str | None = None):
        return RuntimePreflightResult(
            runtime=RuntimeType.lima,
            available=True,
            reasons=[],
            host={"os": "linux", "variant": "native"},
            enforcement_ready={
                "deny_all": deny_all_ready,
                "allowlist": allowlist_ready,
            },
        )

    monkeypatch.setattr(LimaRunner, "preflight", _fake_preflight)


def test_lima_allowlist_rejected_when_strict_not_ready(monkeypatch) -> None:
    monkeypatch.setenv("SANDBOX_ENABLE_EXECUTION", "0")
    monkeypatch.setenv("TLDW_SANDBOX_LIMA_AVAILABLE", "1")
    monkeypatch.setenv("TLDW_SANDBOX_LIMA_ENFORCER_ALLOWLIST_READY", "0")
    _stub_lima_preflight(monkeypatch, allowlist_ready=False)

    svc = SandboxService()
    spec = RunSpec(
        session_id=None,
        runtime=RuntimeType.lima,
        base_image="ubuntu:24.04",
        command=["echo", "ok"],
        network_policy="allowlist",
    )

    with pytest.raises(SandboxPolicy.PolicyUnsupported):
        svc.start_run_scaffold(
            user_id="1",
            spec=spec,
            spec_version="1.0",
            idem_key=None,
            raw_body={},
        )


def test_lima_allowlist_rejected_even_when_override_reports_ready(monkeypatch) -> None:
    monkeypatch.setenv("SANDBOX_ENABLE_EXECUTION", "0")
    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setenv("TLDW_SANDBOX_LIMA_AVAILABLE", "1")
    monkeypatch.setenv("TLDW_SANDBOX_LIMA_ENFORCER_ALLOWLIST_READY", "1")
    _stub_lima_preflight(monkeypatch, allowlist_ready=True)

    svc = SandboxService()
    spec = RunSpec(
        session_id=None,
        runtime=RuntimeType.lima,
        base_image="ubuntu:24.04",
        command=["echo", "ok"],
        network_policy="allowlist",
    )

    with pytest.raises(SandboxPolicy.PolicyUnsupported) as exc:
        svc.start_run_scaffold(
            user_id="1",
            spec=spec,
            spec_version="1.0",
            idem_key=None,
            raw_body={},
        )
    assert "strict_allowlist_not_supported" in exc.value.reasons


def test_lima_allow_all_rejected_as_unsupported_policy(monkeypatch) -> None:
    monkeypatch.setenv("SANDBOX_ENABLE_EXECUTION", "0")
    monkeypatch.setenv("TLDW_SANDBOX_LIMA_AVAILABLE", "1")
    monkeypatch.setenv("TLDW_SANDBOX_LIMA_ENFORCER_DENY_ALL_READY", "1")
    monkeypatch.setenv("TLDW_SANDBOX_LIMA_ENFORCER_ALLOWLIST_READY", "1")
    _stub_lima_preflight(monkeypatch, allowlist_ready=True)

    svc = SandboxService()
    spec = RunSpec(
        session_id=None,
        runtime=RuntimeType.lima,
        base_image="ubuntu:24.04",
        command=["echo", "ok"],
        network_policy="allow_all",
    )

    with pytest.raises(SandboxPolicy.PolicyUnsupported) as exc:
        svc.start_run_scaffold(
            user_id="1",
            spec=spec,
            spec_version="1.0",
            idem_key=None,
            raw_body={},
        )
    assert "unsupported_network_policy" in exc.value.reasons


def test_lima_execution_worker_revalidates_preflight_before_run_start(monkeypatch) -> None:
    monkeypatch.setenv("SANDBOX_ENABLE_EXECUTION", "1")
    monkeypatch.setenv("SANDBOX_BACKGROUND_EXECUTION", "0")
    monkeypatch.setenv("TLDW_SANDBOX_LIMA_AVAILABLE", "1")

    preflight_calls = {"count": 0}
    start_called = {"value": False}

    def _fake_preflight(self, network_policy: str | None = None):
        preflight_calls["count"] += 1
        if preflight_calls["count"] == 1:
            return RuntimePreflightResult(
                runtime=RuntimeType.lima,
                available=True,
                reasons=[],
                host={"os": "darwin", "variant": "native"},
                enforcement_ready={"deny_all": True, "allowlist": True},
            )
        return RuntimePreflightResult(
            runtime=RuntimeType.lima,
            available=False,
            reasons=["strict_deny_all_not_supported"],
            host={"os": "darwin", "variant": "native"},
            enforcement_ready={"deny_all": False, "allowlist": True},
        )

    def _fake_start_run(self, run_id: str, spec: RunSpec, session_workspace: str | None = None):
        start_called["value"] = True
        raise AssertionError("Lima run should not start when execution preflight fails")

    monkeypatch.setattr(LimaRunner, "preflight", _fake_preflight)
    monkeypatch.setattr(LimaRunner, "start_run", _fake_start_run)

    svc = SandboxService()
    spec = RunSpec(
        session_id=None,
        runtime=RuntimeType.lima,
        base_image="ubuntu:24.04",
        command=["echo", "ok"],
        network_policy="deny_all",
    )

    status = svc.start_run_scaffold(
        user_id="1",
        spec=spec,
        spec_version="1.0",
        idem_key=None,
        raw_body={},
    )

    assert status.phase.value == "failed"
    assert status.message == "lima_policy_failed"
    assert preflight_calls["count"] == 2
    assert start_called["value"] is False
