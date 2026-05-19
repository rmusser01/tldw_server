from __future__ import annotations

import pytest

import tldw_Server_API.app.core.Sandbox.policy as policy_module
import tldw_Server_API.app.core.Sandbox.service as service_module

from tldw_Server_API.app.core.Sandbox.models import RunSpec, RuntimeType, TrustLevel
from tldw_Server_API.app.core.Sandbox.policy import SandboxPolicy, SandboxPolicyConfig
from tldw_Server_API.app.core.Sandbox.runtime_capabilities import RuntimePreflightResult
from tldw_Server_API.app.core.Sandbox.service import SandboxService


def test_policy_config_reads_lima_default_runtime(monkeypatch) -> None:
    monkeypatch.setattr(policy_module.app_settings, "SANDBOX_DEFAULT_RUNTIME", "lima", raising=False)
    cfg = SandboxPolicyConfig.from_settings()
    assert cfg.default_runtime == RuntimeType.lima


def test_policy_prefers_shared_runtime_preflight_over_legacy_booleans() -> None:
    policy = SandboxPolicy(SandboxPolicyConfig(default_runtime=RuntimeType.docker))

    with pytest.raises(SandboxPolicy.RuntimeUnavailable) as exc:
        policy.select_runtime(
            RuntimeType.firecracker,
            firecracker_available=True,
            runtime_preflights={
                RuntimeType.firecracker: RuntimePreflightResult(
                    runtime=RuntimeType.firecracker,
                    available=False,
                    reasons=["shared_preflight_unavailable"],
                )
            },
        )

    assert exc.value.reasons == ["shared_preflight_unavailable"]


def test_policy_rejects_worktree_for_untrusted_without_preflight_metadata() -> None:
    policy = SandboxPolicy(SandboxPolicyConfig(default_runtime=RuntimeType.docker))
    spec = RunSpec(
        session_id=None,
        runtime=RuntimeType.worktree,
        base_image=None,
        command=["echo", "ok"],
        trust_level=TrustLevel.untrusted,
    )

    with pytest.raises(SandboxPolicy.PolicyUnsupported) as exc:
        policy.apply_to_run(
            spec,
            firecracker_available=True,
            lima_available=True,
            runtime_preflights=None,
        )

    assert exc.value.runtime == RuntimeType.worktree
    assert exc.value.requirement == "untrusted_requires_vm_runtime"
    assert "trust_level_requires_vm_runtime" in exc.value.reasons


def test_service_start_run_passes_shared_runtime_preflight_to_policy(monkeypatch) -> None:
    monkeypatch.setenv("SANDBOX_ENABLE_EXECUTION", "0")

    shared_preflights = {
        RuntimeType.firecracker: RuntimePreflightResult(
            runtime=RuntimeType.firecracker,
            available=False,
            reasons=["shared_preflight_unavailable"],
        )
    }

    monkeypatch.setattr(
        service_module,
        "collect_runtime_preflights",
        lambda *, network_policy=None: shared_preflights,
    )

    seen: dict[str, object] = {}

    class CapturingPolicy(SandboxPolicy):
        def apply_to_run(
            self,
            spec: RunSpec,
            firecracker_available: bool,
            lima_available: bool = False,
            runtime_preflights=None,
        ) -> RunSpec:
            seen["runtime_preflights"] = runtime_preflights
            return super().apply_to_run(
                spec,
                firecracker_available=firecracker_available,
                lima_available=lima_available,
                runtime_preflights=runtime_preflights,
            )

    svc = SandboxService(policy=CapturingPolicy(SandboxPolicyConfig()))
    spec = RunSpec(
        session_id=None,
        runtime=RuntimeType.firecracker,
        base_image="python:3.11-slim",
        command=["echo", "ok"],
    )

    with pytest.raises(SandboxPolicy.RuntimeUnavailable) as exc:
        svc.start_run_scaffold(
            user_id="1",
            spec=spec,
            spec_version="1.0",
            idem_key=None,
            raw_body={},
        )

    assert seen["runtime_preflights"] is shared_preflights
    assert exc.value.reasons == ["shared_preflight_unavailable"]
