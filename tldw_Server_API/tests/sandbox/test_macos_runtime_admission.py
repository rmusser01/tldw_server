from __future__ import annotations

import pytest

from tldw_Server_API.app.core.Sandbox.models import RunSpec, RuntimeType, TrustLevel
from tldw_Server_API.app.core.Sandbox.policy import SandboxPolicy
from tldw_Server_API.app.core.Sandbox.runtime_capabilities import RuntimePreflightResult
from tldw_Server_API.app.core.Sandbox.service import SandboxService


def test_seatbelt_rejected_for_untrusted_runs(monkeypatch) -> None:
    monkeypatch.setenv("SANDBOX_ENABLE_EXECUTION", "0")

    def _preflights(
        self: SandboxService,
        *,
        network_policy: str | None,
    ) -> dict[RuntimeType, RuntimePreflightResult]:
        del self, network_policy
        return {
            RuntimeType.seatbelt: RuntimePreflightResult(
                runtime=RuntimeType.seatbelt,
                available=True,
                reasons=[],
                supported_trust_levels=["trusted", "standard"],
            )
        }

    monkeypatch.setattr(SandboxService, "_collect_runtime_preflights", _preflights)

    svc = SandboxService()
    spec = RunSpec(
        session_id=None,
        runtime=RuntimeType.seatbelt,
        base_image="python:3.11-slim",
        command=["echo", "ok"],
        trust_level=TrustLevel.untrusted,
    )

    with pytest.raises(SandboxPolicy.PolicyUnsupported) as exc:
        svc.start_run_scaffold(
            user_id="1",
            spec=spec,
            spec_version="1.0",
            idem_key=None,
            raw_body={},
        )

    assert exc.value.runtime == RuntimeType.seatbelt
    assert "trust_level_requires_vm_runtime" in exc.value.reasons


def test_runtime_preflight_trust_levels_are_enforced() -> None:
    policy = SandboxPolicy()
    spec = RunSpec(
        session_id=None,
        runtime=RuntimeType.vz_linux,
        base_image="ubuntu-24.04",
        command=["echo", "ok"],
        trust_level=TrustLevel.untrusted,
    )

    with pytest.raises(SandboxPolicy.PolicyUnsupported) as exc:
        policy.apply_to_run(
            spec,
            firecracker_available=True,
            runtime_preflights={
                RuntimeType.vz_linux: RuntimePreflightResult(
                    runtime=RuntimeType.vz_linux,
                    available=True,
                    supported_trust_levels=["trusted", "standard"],
                )
            },
        )

    assert exc.value.runtime == RuntimeType.vz_linux
    assert "trust_level_not_supported" in exc.value.reasons


def test_seatbelt_standard_requires_explicit_opt_in() -> None:
    policy = SandboxPolicy()
    spec = RunSpec(
        session_id=None,
        runtime=RuntimeType.seatbelt,
        base_image=None,
        command=["echo", "ok"],
        trust_level=TrustLevel.standard,
    )

    with pytest.raises(SandboxPolicy.PolicyUnsupported) as exc:
        policy.apply_to_run(
            spec,
            firecracker_available=True,
            runtime_preflights={
                RuntimeType.seatbelt: RuntimePreflightResult(
                    runtime=RuntimeType.seatbelt,
                    available=True,
                    supported_trust_levels=["trusted"],
                )
            },
        )

    assert exc.value.runtime == RuntimeType.seatbelt
    assert "seatbelt_standard_disabled" in exc.value.reasons
