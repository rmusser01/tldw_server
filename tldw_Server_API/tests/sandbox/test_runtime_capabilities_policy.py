from __future__ import annotations

import pytest

import tldw_Server_API.app.core.Sandbox.policy as policy_module
import tldw_Server_API.app.core.Sandbox.service as service_module
from tldw_Server_API.app.core.Sandbox.models import RunSpec, RuntimeType, SessionSpec, TrustLevel
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


@pytest.mark.parametrize(
    "runtime",
    [RuntimeType.docker, RuntimeType.worktree, RuntimeType.seatbelt, RuntimeType.vz_macos],
)
@pytest.mark.parametrize("with_preflight", [False, True])
@pytest.mark.parametrize("session", [False, True])
def test_policy_rejects_non_vm_untrusted_regardless_of_advertised_trust(
    runtime: RuntimeType, with_preflight: bool, session: bool,
) -> None:
    policy = SandboxPolicy(SandboxPolicyConfig(default_runtime=RuntimeType.docker))
    spec = (
        SessionSpec(runtime=runtime, trust_level=TrustLevel.untrusted)
        if session else RunSpec(
            session_id=None,
            runtime=runtime,
            base_image=None,
            command=["echo", "ok"],
            trust_level=TrustLevel.untrusted,
        )
    )
    preflights = {
        runtime: RuntimePreflightResult(
            runtime=runtime,
            available=True,
            supported_trust_levels=["untrusted"],
            enforcement_ready={"deny_all": True},
        )
    } if with_preflight else None

    with pytest.raises(SandboxPolicy.PolicyUnsupported) as exc:
        (policy.apply_to_session if session else policy.apply_to_run)(
            spec,
            firecracker_available=True,
            lima_available=True,
            runtime_preflights=preflights,
        )

    assert exc.value.runtime == runtime
    assert exc.value.requirement == "untrusted_requires_vm_runtime"
    assert "trust_level_requires_vm_runtime" in exc.value.reasons


@pytest.mark.parametrize(
    "runtime, eligible",
    [
        (RuntimeType.docker, False),
        (RuntimeType.firecracker, True),
        (RuntimeType.lima, True),
        (RuntimeType.vz_linux, True),
        (RuntimeType.vz_macos, False),
        (RuntimeType.seatbelt, False),
        (RuntimeType.worktree, False),
    ],
)
def test_preflight_untrusted_support_requires_vm_and_eligibility(
    runtime: RuntimeType, eligible: bool,
) -> None:
    preflight = RuntimePreflightResult(runtime=runtime, available=True)

    assert ("untrusted" in preflight.supported_trust_levels) is eligible


def test_shared_firecracker_preflight_propagates_real_containment_failure(monkeypatch) -> None:
    from tldw_Server_API.app.core.Sandbox.runners import firecracker_runner
    from tldw_Server_API.app.core.Sandbox.runtime_capabilities import collect_runtime_preflights

    monkeypatch.setenv("TLDW_SANDBOX_FIRECRACKER_FAKE_EXEC", "0")
    monkeypatch.setattr(firecracker_runner, "firecracker_real_enabled", lambda: True)
    monkeypatch.setattr(firecracker_runner, "firecracker_available", lambda: True)
    monkeypatch.setattr(
        firecracker_runner, "_preflight_errors",
        lambda: ["firecracker_host_containment_not_implemented"],
    )

    preflight = collect_runtime_preflights()[RuntimeType.firecracker]

    assert preflight.available is False
    assert preflight.reasons == ["firecracker_host_containment_not_implemented"]
    assert preflight.enforcement_ready == {"deny_all": False, "allowlist": False}


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
